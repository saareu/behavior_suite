"""Subsystem 08 GUI: navigation, S7 readiness, settings, task safety, QC/reopen."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication
from tests.behavioral_arbitration._fixtures import build_synthetic_session

from behavioral_arbitration import (
    BehavioralArbitrationConfig,
    BehavioralArbitrationRequest,
    BehavioralArbitrationRunResult,
)
from behavioral_arbitration.contracts import STATUS_COMPLETED
from ui.controllers.behavioral_arbitration_controller import (
    BehavioralArbitrationController,
    BehavioralArbitrationUiError,
    BehavioralArbitrationUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import (
    GROUP_BEHAVIORAL_INFERENCE,
    GROUP_DATA_PREPARATION,
    GROUP_FEATURE_PROCESSING,
    MainWindow,
)
from ui.pages.behavioral_arbitration_page import BehavioralArbitrationPage

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


def _patch_s7_s6_ok(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, Any]) -> None:
    s7 = fixture["s7_run_dir"]
    s6 = fixture["s6_run_dir"]
    session = fixture["session_root"]

    def _validate_s7(_path: Any) -> dict[str, Any]:
        return {
            "s7_run_dir": Path(s7).resolve(),
            "status": "completed",
            "session_root": session.resolve(),
            "source_s6_run_dir": Path(s6).resolve(),
            "source_s6_run_id": fixture["s6_run_id"],
            "dyadic_features_sha256": fixture["feat_sha"],
        }

    def _validate_s6(_path: Any) -> dict[str, Any]:
        return {
            "s6_run_dir": Path(s6).resolve(),
            "status": "completed",
            "session_root": session.resolve(),
            "run_id": fixture["s6_run_id"],
        }

    for target in (
        "behavioral_arbitration.intake.validate_s7_run_for_downstream",
        "behavioral_arbitration.runner.validate_s7_run_for_downstream",
        "behavioral_inference.runner.validate_s7_run_for_downstream",
        "behavioral_inference.run_discovery.validate_s7_run_for_downstream",
    ):
        monkeypatch.setattr(target, _validate_s7, raising=False)
    for target in (
        "behavioral_arbitration.intake.validate_s6_run_for_downstream",
        "behavioral_arbitration.runner.validate_s6_run_for_downstream",
        "combined_features.runner.validate_s6_run_for_downstream",
        "combined_features.run_discovery.validate_s6_run_for_downstream",
    ):
        monkeypatch.setattr(target, _validate_s6, raising=False)
    for target in (
        "behavioral_arbitration.intake.is_s7_run_eligible_for_downstream",
        "behavioral_arbitration.runner.is_s7_run_eligible_for_downstream",
        "behavioral_inference.runner.is_s7_run_eligible_for_downstream",
        "behavioral_inference.run_discovery.is_s7_run_eligible_for_downstream",
    ):
        monkeypatch.setattr(target, lambda _p: True, raising=False)
    for target in (
        "behavioral_arbitration.intake.is_s6_run_eligible_for_downstream",
        "behavioral_arbitration.runner.is_s6_run_eligible_for_downstream",
        "combined_features.runner.is_s6_run_eligible_for_downstream",
        "combined_features.run_discovery.is_s6_run_eligible_for_downstream",
        "behavioral_inference.runner.is_s6_run_eligible_for_downstream",
        "behavioral_inference.run_discovery.is_s6_run_eligible_for_downstream",
    ):
        monkeypatch.setattr(target, lambda _p: True, raising=False)


def _ready_controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **fixture_kwargs: Any
) -> tuple[BehavioralArbitrationController, dict[str, Any]]:
    fixture = build_synthetic_session(tmp_path, **fixture_kwargs)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller = BehavioralArbitrationController()
    controller.set_session(fixture["session_root"])
    return controller, fixture


def _wait_page_idle(page: BehavioralArbitrationPage, timeout_sec: float = 60.0) -> None:
    _wait_until(lambda: not page.has_running_task, timeout_sec=timeout_sec)


def test_main_window_exposes_group3_s7_and_s8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_arbitration_controller=controller,
    )
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.s7_button.isVisibleTo(window)
    assert window.s8_button.isVisibleTo(window)
    # Bind session on the S8 controller without re-entering async page discovery.
    window._session_root = None  # avoid S7 async discovery on synthetic tree
    window.pages.setCurrentWidget(window.behavioral_arbitration_page)
    window._last_behavior_page = "s8"
    window._ensure_group(GROUP_BEHAVIORAL_INFERENCE)
    window._update_navigation()
    assert window._last_behavior_page == "s8"
    assert window.pages.currentWidget() is window.behavioral_arbitration_page
    window.open_behavioral_inference()
    assert window._last_behavior_page == "s7"
    assert window.pages.currentWidget() is window.behavioral_inference_page
    window.set_application_group(GROUP_FEATURE_PROCESSING)
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window._last_behavior_page == "s7"
    window.open_behavioral_arbitration()
    assert window._last_behavior_page == "s8"
    assert window.pages.currentWidget() is window.behavioral_arbitration_page
    window.set_application_group(GROUP_DATA_PREPARATION)
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.pages.currentWidget() is window.behavioral_arbitration_page


def test_group_navigation_preserves_session_across_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_arbitration_controller=controller,
    )
    root = fixture["session_root"].resolve()
    # Keep MainWindow session unset so S5/S6 openers do not start discovery on
    # the synthetic tree; controller session is the shared scientific context.
    window._session_root = None
    window.pages.setCurrentWidget(window.behavioral_arbitration_page)
    window._last_behavior_page = "s8"
    window._ensure_group(GROUP_BEHAVIORAL_INFERENCE)
    assert controller.state.session_root == root
    window.set_application_group(GROUP_FEATURE_PROCESSING)
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert controller.state.session_root == root
    assert window._last_behavior_page == "s8"
    assert window.pages.currentWidget() is window.behavioral_arbitration_page


def test_s1_s7_regression_group3_still_has_s7() -> None:
    window = MainWindow(PreprocessSetupController())
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.s7_button.isVisibleTo(window)
    assert window.s8_button.isVisibleTo(window)
    window.open_behavioral_inference()
    assert window.pages.currentWidget() is window.behavioral_inference_page
    window.open_behavioral_arbitration()
    assert window.pages.currentWidget() is window.behavioral_arbitration_page
    window.open_behavioral_inference()
    assert window.pages.currentWidget() is window.behavioral_inference_page


def test_missing_s7_blocks_arbitration(tmp_path: Path) -> None:
    session = tmp_path / "empty"
    session.mkdir()
    controller = BehavioralArbitrationController()
    controller.set_session(session)
    assert not controller.can_run_arbitration()
    assert controller.cannot_run_reason() is not None


def test_ineligible_partial_detectors_not_s8_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _fixture = _ready_controller(
        monkeypatch, tmp_path, missing_detectors=("approach",)
    )
    source = controller.selected_s7_source()
    # Default prefers S8-ready; partial coverage should not auto-select as ready.
    if source is not None:
        assert not source.s8_ready
    assert not controller.can_run_arbitration()
    assert any(not opt.s8_ready for opt in controller.state.s7_options)


def test_default_and_manual_s7_selection_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)
    selected = controller.selected_s7_source()
    assert selected is not None
    assert selected.s8_ready
    assert selected.run_id == fixture["s7_run_id"]
    assert not controller.state.s7_selection_manual
    controller.select_s7_source(selected.run_id, manual=True)
    assert controller.state.s7_selection_manual
    controller.refresh_discovery()
    assert controller.state.selected_s7_run_id == selected.run_id
    assert controller.state.s7_selection_manual


def test_scientific_settings_loading_and_immutable_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _fixture = _ready_controller(monkeypatch, tmp_path)
    settings = controller.state.working_settings
    assert settings.fps == 120.0
    assert settings.scientific.contact_post_processing.max_gap_to_fill == 0
    assert settings.scientific.temporal.gap_fill_seconds == pytest.approx(0.05)
    request, identity = controller.build_request(timestamp="20260101T120000")
    assert identity.settings.scientific.decay_rate == settings.scientific.decay_rate
    assert request.config is not None
    assert request.run_purpose == "gui"
    snap_decay = request.config.scientific.decay_rate
    # Working-settings mutation is blocked while the task flag is active.
    with pytest.raises(BehavioralArbitrationUiError):
        controller.update_settings(settings)
    # The request retains its frozen config object independently of later state.
    assert request.config.scientific.decay_rate == snap_decay
    controller.record_failure("abort", expected_identity=identity)


def test_duplicate_task_prevention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _fixture = _ready_controller(monkeypatch, tmp_path)
    _request, _identity = controller.build_request(timestamp="20260101T120001")
    with pytest.raises(BehavioralArbitrationUiError):
        controller.build_request(timestamp="20260101T120002")


def test_source_mutation_blocked_while_task_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)
    controller.build_request(timestamp="20260101T120003")
    with pytest.raises(BehavioralArbitrationUiError):
        controller.select_s7_source(fixture["s7_run_id"], manual=True)


def test_group_navigation_allowed_during_s8_arbitration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)

    def _slow_runner(request: BehavioralArbitrationRequest, progress_callback=None):
        time.sleep(0.2)
        if progress_callback is not None:
            from preprocess.models import OperationProgress

            progress_callback(
                OperationProgress(
                    phase="arbitration",
                    message="working",
                    completed_units=1,
                    total_units=2,
                    total_is_estimate=False,
                    is_indeterminate=False,
                    processed_video_time_sec=None,
                )
            )
        return BehavioralArbitrationRunResult(
            success=True,
            status=STATUS_COMPLETED,
            run_id="behavioral_arbitration__fake",
            run_dir=fixture["session_root"] / "behavioral_arbitration" / "behavioral_arbitration__fake",
            run_meta_path=fixture["session_root"] / "x.json",
            settings_used_path=fixture["session_root"] / "x.yaml",
            processing_log_path=fixture["session_root"] / "x.txt",
            qc_path=fixture["session_root"] / "qc.json",
            behavior_states_path=None,
            behavior_bouts_path=None,
            diagnostics_path=None,
            session_root=fixture["session_root"],
            source_s7_run_id=fixture["s7_run_id"],
            source_s6_run_id=fixture["s6_run_id"],
            warnings=(),
            error_message=None,
            qc={
                "structural_ok": True,
                "metrics": {
                    "label_counts": {"engaged": 2, "undefined": 1},
                    "n_undefined": 1,
                    "directional": {"n_unique_frames": 2, "n_directional_rows": 4},
                    "parallel_coverage": {
                        "contact": {"n_positive": 1, "positive_fraction": 0.25},
                        "side_by_side": {"n_positive": 0, "positive_fraction": 0.0},
                    },
                    "bout_summary": {
                        "n_bouts": 1,
                        "by_channel": {"exclusive": 1},
                        "duration_frames": {"min": 2, "median": 2, "max": 2},
                    },
                },
                "warnings": [],
                "failures": [],
            },
        )

    controller = BehavioralArbitrationController(runner=_slow_runner)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller.set_session(fixture["session_root"])
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_arbitration_controller=controller,
    )
    page = window.behavioral_arbitration_page
    window._session_root = None
    window.pages.setCurrentWidget(page)
    window._last_behavior_page = "s8"
    window._ensure_group(GROUP_BEHAVIORAL_INFERENCE)
    page.refresh_from_state()
    assert controller.can_run_arbitration()
    page._run_arbitration()
    assert page.has_running_task
    window.set_application_group(GROUP_FEATURE_PROCESSING)
    assert not window._hard_navigation_blocked()
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    page.refresh_from_state()
    assert page.has_running_task or controller.state.last_result is not None
    _wait_page_idle(page, timeout_sec=15.0)
    rows = dict(controller.ethogram_summary_rows())
    assert any("engaged" in k for k in rows)
    assert any("undefined" in k.lower() for k in rows)
    assert any("Parallel contact" in k for k in rows)


def test_full_arbitration_launch_qc_and_reopen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)
    request, identity = controller.build_request(timestamp="20260101T130000")
    result = controller.execute_request(request)
    assert result.success
    assert result.behavior_states_path is not None
    assert result.behavior_bouts_path is not None
    accepted = controller.record_success(
        result, expected_identity=identity, refresh_discovery=True
    )
    assert accepted
    selected = controller.selected_s8_run()
    assert selected is not None
    assert selected.run_id == result.run_id
    qc_rows = dict(controller.qc_display_rows())
    assert "Historical status" in qc_rows
    assert "Current eligibility" in qc_rows
    ethogram = dict(controller.ethogram_summary_rows())
    assert ethogram
    bouts = dict(controller.bout_summary_rows())
    assert "Total bouts" in bouts


def test_page_progress_and_failure_presentation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_synthetic_session(tmp_path)
    _patch_s7_s6_ok(monkeypatch, fixture)

    def _failing_runner(request: BehavioralArbitrationRequest, progress_callback=None):
        raise RuntimeError("synthetic arbitration failure")

    controller = BehavioralArbitrationController(runner=_failing_runner)
    controller.set_session(fixture["session_root"])
    page = BehavioralArbitrationPage(controller)
    page.show()
    _wait_page_idle(page)
    page._run_arbitration()
    _wait_page_idle(page, timeout_sec=15.0)
    assert controller.state.run_stage == "Failed"
    assert controller.state.run_error is not None
    assert "synthetic arbitration failure" in controller.state.run_error


def test_settings_config_roundtrip() -> None:
    config = BehavioralArbitrationConfig().validated()
    settings = BehavioralArbitrationUiSettings.from_config(config)
    roundtrip = settings.to_config()
    assert roundtrip.fps == config.fps
    assert (
        roundtrip.scientific.contact_post_processing.max_gap_to_fill
        == config.scientific.contact_post_processing.max_gap_to_fill
    )
    assert (
        roundtrip.scientific.temporal.gap_fill_seconds
        == config.scientific.temporal.gap_fill_seconds
    )


def test_historical_status_vs_current_eligibility_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, fixture = _ready_controller(monkeypatch, tmp_path)
    request, identity = controller.build_request(timestamp="20260101T140000")
    result = controller.execute_request(request)
    controller.record_success(
        result, expected_identity=identity, refresh_discovery=True
    )
    rows = dict(controller.qc_display_rows())
    assert rows.get("Historical status") in {
        STATUS_COMPLETED,
        "completed_with_warnings",
    }
    assert "Current eligibility" in rows
