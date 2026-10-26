"""Subsystem 07 GUI: navigation, S6 source, registry, detectors, inference safety."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)

from behavioral_inference import (
    BehavioralInferenceConfig,
    BehavioralInferenceRequest,
)
from behavioral_inference.config import DEFAULT_WORKING_THRESHOLDS
from combined_features.config import ModelRegistrySettings
from ui.controllers.behavioral_inference_controller import (
    BehavioralInferenceController,
    BehavioralInferenceUiError,
    BehavioralInferenceUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import (
    GROUP_BEHAVIORAL_INFERENCE,
    GROUP_DATA_PREPARATION,
    GROUP_FEATURE_PROCESSING,
    MainWindow,
)
from ui.pages.behavioral_inference_page import BehavioralInferencePage

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


def _write_s6_run(
    session: Path,
    *,
    run_id: str = "combined_features__20260101T000000",
    n_frames: int = 3,
) -> Path:
    s6_root = session / "combined_features" / run_id
    s6_root.mkdir(parents=True, exist_ok=True)
    table = make_two_mouse_feature_frame(n_frames=n_frames).copy()
    if "orig_frame_idx" not in table.columns:
        table["orig_frame_idx"] = table["dense_frame_idx"]
    table["schema_version"] = "dyadic_features_full_v1"
    table["session_root"] = str(session.resolve())
    table["source_s4_run_id"] = "pose_finalization__x"
    table["source_s5_run_id"] = "blob_features__x"
    identity = [
        "schema_version",
        "session_root",
        "focal_id",
        "partner_id",
        "sample_idx",
        "dense_frame_idx",
        "time_sec",
        "orig_frame_idx",
        "source_s4_run_id",
        "source_s5_run_id",
    ]
    ordered = identity + [c for c in table.columns if c not in identity]
    table = table.loc[:, ordered]
    parquet = s6_root / "dyadic_features_full.parquet"
    table.to_parquet(parquet, index=False)
    meta = {
        "schema_version": "combined_features_run_meta_v1",
        "run_id": run_id,
        "status": "completed",
        "session_root": str(session.resolve()),
        "source_s4": {
            "run_dir": str(session / "pose_finalization" / "pose_finalization__x"),
            "run_id": "pose_finalization__x",
            "session_root": str(session.resolve()),
            "final_pose": str(
                session / "pose_finalization" / "pose_finalization__x" / "final_pose.parquet"
            ),
            "final_pose_sha256": "a" * 64,
            "prepared_sync": str(session / "preprocess" / "prepared_sync.json"),
            "prepared_sync_sha256": "b" * 64,
        },
        "source_s5": {
            "run_dir": str(session / "blob_features" / "blob_features__x"),
            "run_id": "blob_features__x",
            "session_root": str(session.resolve()),
            "blob_features": str(
                session / "blob_features" / "blob_features__x" / "blob_features.parquet"
            ),
            "blob_features_sha256": "c" * 64,
            "prepared_sync_sha256": "b" * 64,
        },
        "adapter": {"n_tracks": 2, "n_samples": n_frames},
        "summary": {"sample_count": n_frames},
        "outputs": {"dyadic_features_parquet": str(parquet)},
    }
    (s6_root / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (s6_root / "qc.json").write_text(
        json.dumps(
            {
                "schema_version": "combined_features_qc_v1",
                "structural_ok": True,
                "metrics": {"n_tracks": 2, "n_samples": n_frames},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (s6_root / "model_compat.json").write_text("{}\n", encoding="utf-8")
    (s6_root / "settings_used.yaml").write_text("schema_version: x\n", encoding="utf-8")
    (s6_root / "processing_log.txt").write_text("ok\n", encoding="utf-8")
    return s6_root


def _write_fake_registry(root: Path, *, names: dict[str, str]) -> Path:
    for name, family in names.items():
        bundle = root / family / "v1" / name
        bundle.mkdir(parents=True, exist_ok=True)
        payload = f"fake-model::{family}::{name}".encode()
        (bundle / "model.joblib").write_bytes(payload)
        (bundle / "feature_spec.json").write_text(
            json.dumps(
                {
                    "detector_name": name,
                    "feature_cols": ["feat_a", "feat_b"],
                    "selected_feature_cols": ["feat_a", "feat_b"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    # Obsolete standalone contact for display tests.
    obsolete = root / "dyadic" / "v1" / "contact"
    obsolete.mkdir(parents=True, exist_ok=True)
    (obsolete / "model.joblib").write_bytes(b"obsolete")
    (obsolete / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "contact",
                "feature_cols": ["feat_a"],
                "selected_feature_cols": ["feat_a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _patch_s6_ok(monkeypatch: pytest.MonkeyPatch, s6_run_dir: Path, session: Path) -> None:
    def _validate(_path: Any) -> dict[str, Any]:
        return {
            "s6_run_dir": Path(s6_run_dir).resolve(),
            "status": "completed",
            "session_root": session.resolve(),
            "s4_run_dir": session / "pose_finalization" / "pose_finalization__x",
            "s5_run_dir": session / "blob_features" / "blob_features__x",
            "prepared_sync_sha256": "b" * 64,
            "final_pose_sha256": "a" * 64,
            "blob_features_sha256": "c" * 64,
            "s6_run_id": s6_run_dir.name,
        }

    for target in (
        "behavioral_inference.runner.validate_s6_run_for_downstream",
        "combined_features.runner.validate_s6_run_for_downstream",
        "ui.controllers.behavioral_inference_controller.validate_s6_run_for_downstream",
    ):
        monkeypatch.setattr(target, _validate, raising=False)
    for target in (
        "behavioral_inference.runner.is_s6_run_eligible_for_downstream",
        "behavioral_inference.run_discovery.is_s6_run_eligible_for_downstream",
        "combined_features.runner.is_s6_run_eligible_for_downstream",
        "combined_features.run_discovery.is_s6_run_eligible_for_downstream",
        "ui.controllers.behavioral_inference_controller.is_s6_run_eligible_for_downstream",
    ):
        monkeypatch.setattr(target, lambda _p: True, raising=False)


def _patch_runtime_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    from behavioral_inference.loader import RuntimeDependencyStatus

    status = RuntimeDependencyStatus(
        available=True,
        versions={"xgboost": "2.0", "sklearn": "1.4", "joblib": "1.3"},
        missing=(),
        import_errors={},
    )
    monkeypatch.setattr(
        "ui.controllers.behavioral_inference_controller.probe_runtime_dependencies",
        lambda: status,
    )
    monkeypatch.setattr(
        "behavioral_inference.loader.probe_runtime_dependencies",
        lambda: status,
    )


def _patch_models_to_fakes(monkeypatch: pytest.MonkeyPatch, registry: Path) -> None:
    from behavioral_inference.loader import LoadedModel

    def _fake_load(info: Any, *, allow_fallback: bool = False) -> LoadedModel:
        stub = make_loaded_model(
            info.detector_name,
            info.family,
            list(info.feature_cols) or ["feat_a", "feat_b"],
            tmp_path=registry,
            support=info.support,
        )
        return LoadedModel(
            bundle=info,
            estimator=stub.estimator,
            classes=stub.classes,
            positive_class_index=stub.positive_class_index,
            positive_class_label=stub.positive_class_label,
            n_features_in=stub.n_features_in,
            library_versions=stub.library_versions,
        )

    monkeypatch.setattr("behavioral_inference.engine.load_model_bundle", _fake_load)
    monkeypatch.setattr("behavioral_inference.loader.load_model_bundle", _fake_load)


def _ready_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    dyadic: tuple[str, ...] = ("nose_to_nose",),
    focal: tuple[str, ...] = ("approach",),
) -> tuple[BehavioralInferenceController, Path, Path]:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "session"
    session.mkdir(parents=True, exist_ok=True)
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    registry = _write_fake_registry(
        tmp_path / "models",
        names={
            "nose_to_nose": "dyadic",
            "approach": "focal",
            "contact_intact": "dyadic",
            "contact_occluded": "dyadic",
            "mutually_disengaged": "dyadic",
            "side_by_side": "dyadic",
            "nose_body_intact": "focal",
            "nose_body_occ": "focal",
            "nose_to_tail": "focal",
            "retract": "focal",
        },
    )
    _patch_models_to_fakes(monkeypatch, registry)
    controller = BehavioralInferenceController()
    controller.set_session(session)
    controller.update_settings(
        BehavioralInferenceUiSettings(
            parity_mode="legacy_parity",
            model_registry_root=registry,
            selected_detectors_dyadic=dyadic,
            selected_detectors_focal=focal,
            working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
        )
    )
    return controller, session, registry


def test_main_window_exposes_group3_and_s7(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, session, _registry = _ready_controller(monkeypatch, tmp_path)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_inference_controller=controller,
    )
    assert window.group_behavior_button.isEnabled() is True
    assert window.s7_button.isEnabled() is True
    window.open_behavioral_inference(session)
    assert window._active_group == GROUP_BEHAVIORAL_INFERENCE
    assert window.pages.currentWidget() is window.behavioral_inference_page
    assert window.s7_button.isHidden() is False
    assert window.s5_button.isHidden() is True
    assert window.s1_button.isHidden() is True


def test_group_navigation_preserves_session_across_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, session, _registry = _ready_controller(monkeypatch, tmp_path)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_inference_controller=controller,
    )
    window._remember_session(session)
    window.open_behavioral_inference(session)
    assert window._session_root == session.resolve()
    window.set_application_group(GROUP_DATA_PREPARATION)
    assert window._session_root == session.resolve()
    # Switch to Feature Processing without auto-loading S5 against a fixture
    # that has no prepared video (same caution as S6 GUI tests).
    window._ensure_group(GROUP_FEATURE_PROCESSING)
    window.pages.setCurrentWidget(window.combined_features_page)
    window._update_navigation()
    assert window._session_root == session.resolve()
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.pages.currentWidget() is window.behavioral_inference_page
    assert (
        window.behavioral_inference_page.controller.state.session_root
        == session.resolve()
    )


def test_s1_s6_regression_group3_does_not_break_feature_nav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    window = MainWindow(PreprocessSetupController())
    window._ensure_group(GROUP_FEATURE_PROCESSING)
    window.pages.setCurrentWidget(window.combined_features_page)
    window._update_navigation()
    assert window.s5_button.isHidden() is False
    assert window.s6_button.isHidden() is False
    assert window.s7_button.isHidden() is True
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.s7_button.isHidden() is False
    assert window.s6_button.isHidden() is True
    window.set_application_group(GROUP_FEATURE_PROCESSING)
    assert window.s6_button.isEnabled() is True
    assert window.s5_button.isEnabled() is True


def test_missing_s6_blocks_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "empty"
    session.mkdir()
    controller = BehavioralInferenceController()
    controller.set_session(session)
    assert controller.can_run_inference() is False
    reason = (controller.cannot_run_reason() or "").lower()
    assert "eligible s6" in reason or "no currently eligible" in reason


def test_default_and_manual_s6_selection_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    older = _write_s6_run(session, run_id="combined_features__20260101T010000", n_frames=2)
    newer = _write_s6_run(session, run_id="combined_features__20260101T020000", n_frames=2)
    _patch_s6_ok(monkeypatch, newer, session)
    # Make eligibility true for both.
    monkeypatch.setattr(
        "combined_features.run_discovery.is_s6_run_eligible_for_downstream",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "ui.controllers.behavioral_inference_controller.is_s6_run_eligible_for_downstream",
        lambda _p: True,
    )
    controller = BehavioralInferenceController()
    controller.set_session(session)
    assert controller.state.selected_s6_run_id == newer.name
    controller.select_s6_source(older.name, manual=True)
    controller.refresh_discovery()
    assert controller.state.selected_s6_run_id == older.name
    assert controller.state.s6_selection_manual is True


def test_ineligible_s6_not_listed_and_blocks_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    s6 = _write_s6_run(session, n_frames=2)
    monkeypatch.setattr(
        "combined_features.run_discovery.is_s6_run_eligible_for_downstream",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "ui.controllers.behavioral_inference_controller.is_s6_run_eligible_for_downstream",
        lambda _p: False,
    )
    controller = BehavioralInferenceController()
    controller.set_session(session)
    assert controller.state.s6_options == ()
    assert controller.can_run_inference() is False
    _ = s6


def test_model_registry_discovery_and_detector_vs_specialist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, registry = _ready_controller(monkeypatch, tmp_path)
    assert controller.state.registry_status in ("ok", "partial")
    names = {d.name: d for d in controller.state.detector_availability}
    assert "nose_to_nose" in names
    assert names["nose_to_nose"].ready is True
    assert "contact" in names
    assert names["contact"].specialist_deps == ("contact_intact", "contact_occluded")
    assert "contact_intact" not in names or names["contact_intact"].selected is False
    # Obsolete contact appears as non-executable identification.
    obsolete = [d for d in controller.state.detector_availability if d.name == "contact" and d.family == "dyadic"]
    assert obsolete
    assert registry.is_dir()


def test_missing_unsupported_model_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    registry = _write_fake_registry(
        tmp_path / "models",
        names={"nose_to_nose": "dyadic"},
    )
    controller = BehavioralInferenceController()
    controller.set_session(session)
    controller.update_settings(
        BehavioralInferenceUiSettings(
            model_registry_root=registry,
            selected_detectors_dyadic=("nose_to_nose", "side_by_side"),
            selected_detectors_focal=(),
            working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
        )
    )
    by_name = {d.name: d for d in controller.state.detector_availability if d.family == "dyadic"}
    assert by_name["nose_to_nose"].ready is True
    assert by_name["side_by_side"].ready is False
    assert by_name["side_by_side"].selected is True  # must not silently disappear
    assert controller.can_run_inference() is False


def test_threshold_override_and_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, _registry = _ready_controller(monkeypatch, tmp_path)
    rows_before = dict(
        (key, (value, source)) for key, value, source in controller.threshold_display_rows()
    )
    assert rows_before["approach"][1] == "inference_config"
    controller.set_threshold_override("approach", 0.9)
    rows = dict(
        (key, (value, source)) for key, value, source in controller.threshold_display_rows()
    )
    assert rows["approach"][1] == "user_override"
    assert float(rows["approach"][0]) == pytest.approx(0.9)
    controller.clear_threshold_overrides()
    rows_after = dict(
        (key, (value, source)) for key, value, source in controller.threshold_display_rows()
    )
    assert rows_after["approach"][1] == "inference_config"


def test_smoothing_mode_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    controller, _session, _registry = _ready_controller(monkeypatch, tmp_path)
    controller.update_settings(
        replace(controller.state.working_settings, parity_mode="raw_framewise")
    )
    assert controller.state.working_settings.parity_mode == "raw_framewise"
    assert any("raw_framewise" in w for w in controller.state.readiness_warnings)


def test_readiness_requires_registry_and_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, _registry = _ready_controller(monkeypatch, tmp_path)
    assert controller.can_run_inference() is True
    controller.set_model_registry_root(None)
    assert controller.can_run_inference() is False
    assert "registry" in (controller.cannot_run_reason() or "").lower()


def test_immutable_execution_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, _registry = _ready_controller(monkeypatch, tmp_path)
    request, identity = controller.build_request(timestamp="20260101T120000")
    assert isinstance(request, BehavioralInferenceRequest)
    assert request.config is not None
    snap = request.config.parity_mode
    controller.state.working_settings = BehavioralInferenceUiSettings(
        parity_mode="raw_framewise",
        model_registry_root=identity.settings.model_registry_root,
        selected_detectors_dyadic=("nose_to_nose",),
        selected_detectors_focal=("approach",),
        working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
    )
    assert request.config.parity_mode == snap
    assert identity.settings.parity_mode == snap
    controller.record_failure("cleanup", expected_identity=identity)


def test_duplicate_task_prevention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, _registry = _ready_controller(monkeypatch, tmp_path)
    _request, identity = controller.build_request(timestamp="20260101T130000")
    assert controller.can_run_inference() is False
    assert "already running" in (controller.cannot_run_reason() or "").lower()
    with pytest.raises(BehavioralInferenceUiError, match="already running"):
        controller.build_request(timestamp="20260101T130001")
    controller.record_failure("stop", expected_identity=identity)


def test_source_mutation_blocked_while_task_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, session, _registry = _ready_controller(monkeypatch, tmp_path / "a")
    other = tmp_path / "b" / "session"
    other.mkdir(parents=True)
    _request, identity = controller.build_request(timestamp="20260101T140000")
    with pytest.raises(BehavioralInferenceUiError, match="Cannot change S7 session"):
        controller.set_session(other)
    with pytest.raises(BehavioralInferenceUiError, match="Cannot change S6 source"):
        controller.select_s6_source(None, manual=True)
    assert controller.state.session_root == session.resolve()
    controller.record_failure("stop", expected_identity=identity)


def test_group_navigation_allowed_during_s7_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, session, _registry = _ready_controller(monkeypatch, tmp_path)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_inference_controller=controller,
    )
    window.open_behavioral_inference(session)
    page = window.behavioral_inference_page
    request, identity = page.controller.build_request(timestamp="20260101T150000")
    assert page.has_running_task is True
    window._update_navigation()
    assert window.group_data_button.isEnabled() is True
    assert window.group_feature_button.isEnabled() is True
    assert window.s7_button.isEnabled() is True

    window.set_application_group(GROUP_DATA_PREPARATION)
    assert window._active_group == GROUP_DATA_PREPARATION
    assert page.controller.state.active_inference_identity is identity
    assert page.controller.state.task_running is True

    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.pages.currentWidget() is page
    assert page.controller.state.active_inference_identity is identity
    assert page.run_button.isEnabled() is False
    page.controller.record_failure("cancelled-for-test", expected_identity=identity)
    _ = request


def test_full_inference_launch_qc_and_reopen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, session, _registry = _ready_controller(monkeypatch, tmp_path)
    page = BehavioralInferencePage(controller)
    monkeypatch.setattr(
        "ui.pages.behavioral_inference_page.QMessageBox.warning",
        lambda *args, **kwargs: None,
    )
    page.set_session(session)
    _wait_until(
        lambda: (
            not page.discovery_runner.is_running
            and not page.controller.state.discovery_running
            and page.controller.can_run_inference()
        ),
        timeout_sec=30.0,
    )
    assert page.run_button.isEnabled() is True
    page.run_button.click()
    _wait_until(lambda: page.controller.state.last_result is not None, timeout_sec=90.0)
    result = page.controller.state.last_result
    assert result is not None
    assert result.success is True
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()
    rows = dict(page.controller.qc_display_rows())
    assert "Historical status" in rows
    assert "Current eligibility" in rows
    assert "Source S6" in rows
    assert any(k.startswith("S8 handoff") for k in rows)
    run = page.controller.selected_s7_run()
    assert run is not None
    assert run.run_id == result.run_id


def test_page_progress_and_failure_presentation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(request, progress_callback=None):
        raise RuntimeError("synthetic S7 failure")

    controller, session, _registry = _ready_controller(monkeypatch, tmp_path)
    controller._runner = _boom
    page = BehavioralInferencePage(controller)
    monkeypatch.setattr(
        "ui.pages.behavioral_inference_page.QMessageBox.warning",
        lambda *args, **kwargs: None,
    )
    seen: list[str] = []
    page.unexpected_error.connect(seen.append)
    page.set_session(session)
    _wait_until(
        lambda: (
            not page.discovery_runner.is_running
            and not page.controller.state.discovery_running
            and page.controller.can_run_inference()
        ),
        timeout_sec=30.0,
    )
    assert page.run_button.isEnabled() is True
    page.run_button.click()
    _wait_until(lambda: page.controller.state.run_error is not None, timeout_sec=30.0)
    assert page.controller.state.last_result is None
    assert "synthetic S7 failure" in (page.controller.state.run_error or "")
    assert seen and "synthetic S7 failure" in seen[0]


def test_s8_required_view_availability_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, _registry = _ready_controller(monkeypatch, tmp_path)
    request, identity = controller.build_request(timestamp="20260101T160000")
    result = controller.execute_request(request)
    assert result.success is True
    controller.record_success(result, expected_identity=identity)
    rows = dict(controller.qc_display_rows())
    assert "S8 handoff (S7 eligible)" in rows
    assert "S8 required-view note" in rows
    assert "does not require all detectors" in rows["S8 required-view note"].lower()


def test_settings_config_roundtrip() -> None:
    settings = BehavioralInferenceUiSettings(
        parity_mode="legacy_parity",
        selected_detectors_dyadic=("nose_to_nose",),
        selected_detectors_focal=("approach",),
        working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
        exp_name="demo",
        model_registry_root=None,
    ).validated()
    config = settings.to_config()
    assert isinstance(config, BehavioralInferenceConfig)
    assert config.parity_mode == "legacy_parity"
    assert isinstance(config.model_registry, ModelRegistrySettings)
    assert settings.exp_name == "demo"
    assert settings.validated().exp_name == "demo"


def test_select_all_compatible_detectors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, _registry = _ready_controller(
        monkeypatch, tmp_path, dyadic=(), focal=()
    )
    controller.select_all_compatible_detectors()
    assert "nose_to_nose" in controller.state.working_settings.selected_detectors_dyadic
    assert "approach" in controller.state.working_settings.selected_detectors_focal
