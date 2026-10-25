"""Subsystem 06 GUI: navigation, source selection, extraction, discovery, safety."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.combined_features.helpers import write_joint_session

from combined_features import CombinedFeaturesConfig, ModelRegistrySettings
from combined_features.runner import CombinedFeaturesRequest
from ui.controllers.combined_features_controller import (
    CombinedFeaturesController,
    CombinedFeaturesUiError,
    CombinedFeaturesUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.combined_features_page import CombinedFeaturesPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


def _patch_upstream_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic joint sessions lack full S3/S1 trees; mark upstream eligible."""

    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "pose_finalization.run_discovery.is_s4_run_eligible_for_downstream",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "blob_features.run_discovery.is_s5_run_eligible_for_downstream",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "combined_features.runner.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
        raising=False,
    )
    monkeypatch.setattr(
        "combined_features.runner.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
        raising=False,
    )


def _write_mini_registry(root: Path) -> Path:
    dyadic = root / "dyadic" / "v1"
    focal = root / "focal" / "v1"
    (dyadic / "contact_intact").mkdir(parents=True)
    (dyadic / "contact").mkdir(parents=True)
    (focal / "approach").mkdir(parents=True)
    (dyadic / "contact_intact" / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "contact_intact",
                "selected_feature_cols": [
                    "distance_nose_i_nose_j_norm",
                    "cc_count",
                    "nose_any_imputed_i",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dyadic / "contact_intact" / "model.joblib").write_bytes(b"fake")
    (dyadic / "contact" / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "contact",
                "selected_feature_cols": ["nose_any_refined", "nose_any_imputed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (focal / "approach" / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "approach",
                "selected_feature_cols": [
                    "signed_kinematic_intent",
                    "nose_any_imputed_i",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _joint_session(tmp_path: Path, **kwargs):
    return write_joint_session(tmp_path, **kwargs)


def test_main_window_exposes_s6_navigation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    window = MainWindow(PreprocessSetupController())
    assert window.s6_button.isEnabled() is True
    window.open_combined_features(paths["session_root"])
    assert window.pages.currentWidget() is window.combined_features_page
    assert window.combined_features_page.controller.state.session_root is not None


def test_group_navigation_preserves_session_and_enables_s6(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ui.main_window import GROUP_DATA_PREPARATION, GROUP_FEATURE_PROCESSING

    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    session = paths["session_root"]
    window = MainWindow(PreprocessSetupController())
    # Remember session without opening S2 (joint fixture has no prepared video).
    window._remember_session(session)
    window.show_subsystem_1()
    assert window._active_group == GROUP_DATA_PREPARATION
    assert window.s6_button.isEnabled() is True

    # Open S6 directly — avoid auto-opening S5 against a fixture without video.
    window.open_combined_features(session)
    assert window._active_group == GROUP_FEATURE_PROCESSING
    assert window.s5_button.isHidden() is False
    assert window.s6_button.isEnabled() is True
    assert window.pages.currentWidget() is window.combined_features_page
    assert window.combined_features_page.controller.state.session_root == session.resolve()
    assert window._session_root == session.resolve()

    window.set_application_group(GROUP_DATA_PREPARATION)
    assert window._session_root == session.resolve()


def test_missing_s4_or_s5_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_upstream_eligible(monkeypatch)
    session = tmp_path / "empty_session"
    session.mkdir()
    controller = CombinedFeaturesController()
    controller.set_session(session)
    assert controller.can_run_extraction() is False
    reason = controller.cannot_run_reason() or ""
    assert "eligible S4" in reason or "eligible S5" in reason
    assert "no eligible S4" in controller.alignment_status_text().lower() or (
        "no eligible" in controller.alignment_status_text().lower()
    )


def test_compatible_default_pair_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    assert controller.selected_s4_source() is not None
    assert controller.selected_s5_source() is not None
    assert controller.state.alignment is not None
    assert controller.state.alignment.eligible is True
    assert controller.can_run_extraction() is True


def test_incompatible_pair_rejected_before_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path, n_samples=4)
    # Second S5 with mismatched sample count.
    write_joint_session(
        tmp_path / "alt",
        n_samples=8,
        s4_run_id="pose_finalization__ignored",
        s5_run_id="blob_features__bad",
    )
    # Copy mismatched S5 into the primary session.
    import shutil

    bad_s5 = tmp_path / "alt" / "session" / "blob_features" / "blob_features__bad"
    dest = paths["session_root"] / "blob_features" / "blob_features__bad"
    shutil.copytree(bad_s5, dest)
    # Rewrite S5 meta to claim the primary session + sync but keep 8-sample parquet.
    meta_path = dest / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["source_s1"]["session_root"] = str(paths["session_root"].resolve())
    meta["source_s1"]["prepared_sync"] = str(paths["prepared_sync"].resolve())
    from tests.combined_features.helpers import sha256_file

    meta["source_s1"]["prepared_sync_sha256"] = sha256_file(paths["prepared_sync"])
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    controller.select_s4_source(paths["s4_run_dir"].name, manual=True)
    controller.select_s5_source("blob_features__bad", manual=True)
    assert controller.can_run_extraction() is False
    reason = controller.cannot_run_reason() or ""
    assert "incompatible" in reason.lower() or "mismatch" in reason.lower()
    with pytest.raises(CombinedFeaturesUiError):
        controller.build_request(timestamp="20260101T120000")


def test_manual_selection_not_silently_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(
        tmp_path,
        s4_run_id="pose_finalization__20261021T120000",
        s5_run_id="blob_features__20261021T120100",
    )
    write_joint_session(
        tmp_path,
        s4_run_id="pose_finalization__20261021T130000",
        s5_run_id="blob_features__20261021T130100",
        sync_path=paths["prepared_sync"],
    )
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    older_s4 = "pose_finalization__20261021T120000"
    older_s5 = "blob_features__20261021T120100"
    controller.select_s4_source(older_s4, manual=True)
    controller.select_s5_source(older_s5, manual=True)
    controller.refresh_discovery()
    assert controller.state.selected_s4_run_id == older_s4
    assert controller.state.selected_s5_run_id == older_s5
    assert controller.state.s4_selection_manual is True


def test_readiness_validation_and_immutable_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    controller.update_settings(
        CombinedFeaturesUiSettings(smooth_win=9, diff_dt=1, iteration_name="suite")
    )
    request, identity = controller.build_request(timestamp="20260101T120000")
    assert isinstance(request, CombinedFeaturesRequest)
    assert request.config is not None
    snap = request.config.smooth_win
    # Mutate working settings while the snapshotted request is in flight (GUI
    # would block this; prove the request object itself stays immutable).
    controller.state.working_settings = CombinedFeaturesUiSettings(
        smooth_win=11, diff_dt=1, iteration_name="suite"
    )
    assert request.config.smooth_win == snap
    assert identity.settings.smooth_win == snap
    assert controller.state.active_extraction_identity is not None
    assert controller.state.active_extraction_identity.settings.smooth_win == snap
    controller.record_failure("cleanup", expected_identity=identity)


def test_full_extraction_launch_and_qc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path, n_samples=4)
    page = CombinedFeaturesPage(CombinedFeaturesController())
    page.set_session(paths["session_root"])
    assert page.run_button.isEnabled() is True
    page.run_button.click()
    _wait_until(lambda: page.controller.state.last_result is not None, timeout_sec=90.0)
    result = page.controller.state.last_result
    assert result is not None
    assert result.success is True
    assert result.dyadic_features_path is not None
    assert result.dyadic_features_path.is_file()
    rows = page.controller.qc_display_rows()
    assert any(name == "Historical status" for name, _ in rows)
    assert any(name == "Current eligibility" for name, _ in rows)
    assert page.controller.state.summary is not None
    assert page.controller.state.summary.completed_runs


def test_duplicate_task_prevention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    _request, identity = controller.build_request(timestamp="20260101T130000")
    assert controller.can_run_extraction() is False
    assert "already running" in (controller.cannot_run_reason() or "").lower()
    with pytest.raises(CombinedFeaturesUiError, match="already running"):
        controller.build_request(timestamp="20260101T130001")
    controller.record_failure("stop", expected_identity=identity)


def test_source_mutation_blocked_while_task_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    first = _joint_session(tmp_path / "a")["session_root"]
    second = _joint_session(tmp_path / "b")["session_root"]
    controller = CombinedFeaturesController()
    controller.set_session(first)
    _request, identity = controller.build_request(timestamp="20260101T140000")
    with pytest.raises(CombinedFeaturesUiError, match="Cannot change S6 session"):
        controller.set_session(second)
    with pytest.raises(CombinedFeaturesUiError, match="Cannot change S4 source"):
        controller.select_s4_source(None, manual=True)
    assert controller.state.session_root == first.resolve()
    controller.record_failure("stop", expected_identity=identity)


def test_group_navigation_allowed_during_s6_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ui.main_window import GROUP_DATA_PREPARATION, GROUP_FEATURE_PROCESSING

    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    window = MainWindow(PreprocessSetupController())
    window.open_combined_features(paths["session_root"])
    page = window.combined_features_page
    request, identity = page.controller.build_request(timestamp="20260101T150000")
    assert page.has_running_task is True
    window._update_navigation()
    assert window.group_data_button.isEnabled() is True
    assert window.s1_button.isEnabled() is True
    assert window.s6_button.isEnabled() is True

    window.set_application_group(GROUP_DATA_PREPARATION)
    assert window._active_group == GROUP_DATA_PREPARATION
    assert page.controller.state.active_extraction_identity is identity
    assert page.controller.state.task_running is True

    window.set_application_group(GROUP_FEATURE_PROCESSING)
    # Returning to Feature Processing preserves S6 when it was the active page.
    assert window.pages.currentWidget() is page
    assert page.controller.state.active_extraction_identity is identity
    assert page.run_button.isEnabled() is False
    page.controller.record_failure("cancelled-for-test", expected_identity=identity)
    _ = request


def test_completed_run_reopening_and_eligibility_distinction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path, n_samples=4)
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    request, identity = controller.build_request(timestamp="20260101T160000")
    result = controller.execute_request(request)
    assert result.success is True
    accepted = controller.record_success(result, expected_identity=identity)
    assert accepted is True
    run = controller.selected_s6_run()
    assert run is not None
    assert run.run_id == result.run_id
    rows = controller.qc_display_rows()
    labels = {name for name, _ in rows}
    assert "Historical status" in labels
    assert "Current eligibility" in labels
    # Stale classification is a discovery concept distinct from historical completion.
    assert run.status in ("completed", "completed_with_warnings")


def test_model_registry_absent_does_not_block_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    assert controller.state.working_settings.model_registry_root is None
    preview = controller.state.model_compat_preview
    assert isinstance(preview, dict)
    assert preview.get("registry_status") == "unavailable"
    assert controller.can_run_extraction() is True


def test_model_registry_dyadic_focal_and_obsolete_display(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path, n_samples=4)
    registry = _write_mini_registry(tmp_path / "models")
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    controller.update_settings(
        CombinedFeaturesUiSettings(
            smooth_win=9,
            diff_dt=1,
            model_registry_root=registry,
            include_focal=True,
        )
    )
    rows = controller.model_compat_display_rows()
    by_name = {name: (family, status, detail) for name, family, status, detail in rows}
    assert "contact" in by_name
    assert by_name["contact"][1] == "unsupported"
    assert "contact_intact" in by_name
    assert by_name["contact_intact"][0] == "dyadic"
    assert "approach" in by_name
    assert by_name["approach"][0] == "focal"
    assert "inference not run" in by_name["approach"][2]

    request, identity = controller.build_request(timestamp="20260101T170000")
    result = controller.execute_request(request)
    controller.record_success(result, expected_identity=identity)
    after = controller.model_compat_display_rows()
    after_by_name = {name: status for name, _family, status, _detail in after}
    assert after_by_name.get("contact") == "unsupported"
    assert "contact_intact" in after_by_name
    assert "approach" in after_by_name


def test_unknown_implantation_warning_surfaces_in_qc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    from tests.combined_features.helpers import build_final_pose_table

    pose = build_final_pose_table(n_samples=4, headstage_owner=None)
    paths = _joint_session(tmp_path, n_samples=4, final_pose=pose)
    controller = CombinedFeaturesController()
    controller.set_session(paths["session_root"])
    request, identity = controller.build_request(timestamp="20260101T180000")
    result = controller.execute_request(request)
    controller.record_success(result, expected_identity=identity)
    rows = dict(controller.qc_display_rows())
    assert "Implantation" in rows or any(
        "implant" in (rows.get("Warnings") or "").lower() for _ in [0]
    )
    warnings = rows.get("Warnings", "")
    implant = rows.get("Implantation", "")
    assert "implant" in (warnings + implant).lower() or "unresolved" in (
        warnings + implant
    ).lower()


def test_s1_s5_gui_regression_s6_does_not_break_s5_nav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Opening S6 must not disable S5 or steal Group 2 default unexpectedly."""

    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)
    window = MainWindow(PreprocessSetupController())
    # Switch groups without binding an S5-ineligible session into blob features.
    window._ensure_group("feature_processing")
    window.pages.setCurrentWidget(window.blob_features_page)
    window._update_navigation()
    assert window.pages.currentWidget() is window.blob_features_page
    window.open_combined_features(paths["session_root"])
    assert window.pages.currentWidget() is window.combined_features_page
    # Returning to S5 must remain possible; do not auto-load the S6-only fixture.
    window.pages.setCurrentWidget(window.blob_features_page)
    window._update_navigation()
    assert window.pages.currentWidget() is window.blob_features_page
    assert window.s5_button.isEnabled() is True
    assert window.s6_button.isEnabled() is True


def test_page_progress_and_error_reporting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_upstream_eligible(monkeypatch)
    paths = _joint_session(tmp_path)

    def _boom(request, progress_callback=None):
        raise RuntimeError("synthetic S6 failure")

    page = CombinedFeaturesPage(
        CombinedFeaturesController(runner=_boom)
    )
    # Avoid modal dialogs blocking the offscreen Qt test runner.
    monkeypatch.setattr(
        "ui.pages.combined_features_page.QMessageBox.warning",
        lambda *args, **kwargs: None,
    )
    seen: list[str] = []
    page.unexpected_error.connect(seen.append)
    page.set_session(paths["session_root"])
    page.run_button.click()
    _wait_until(lambda: page.controller.state.run_error is not None, timeout_sec=30.0)
    assert page.controller.state.last_result is None
    assert "synthetic S6 failure" in (page.controller.state.run_error or "")
    assert seen and "synthetic S6 failure" in seen[0]


def test_settings_config_roundtrip() -> None:
    settings = CombinedFeaturesUiSettings(
        smooth_win=9,
        diff_dt=2,
        iteration_name="lab",
        exp_name="demo",
        model_registry_root=None,
    ).validated()
    config = settings.to_config()
    assert isinstance(config, CombinedFeaturesConfig)
    assert config.diff_dt == 2
    assert isinstance(config.model_registry, ModelRegistrySettings)
    assert config.model_registry.root is None
    assert config.model_registry.include_focal is True
