"""Subsystem 09 GUI: Group 4 nav, S8 readiness, settings, task safety, QC/reopen."""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication
from tests.results_packaging._fixtures import build_packaging_session

from behavioral_arbitration.contracts import STATUS_COMPLETED
from results_packaging import (
    ResultsPackageRunResult,
    ResultsPackagingConfig,
    ResultsPackagingRequest,
)
from results_packaging.config import ReviewEmitV1Defaults
from results_packaging.contracts import REVIEW_EMIT_PROFILE_ID, STATUS_COMPLETED_WITH_WARNINGS
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.results_packaging_controller import (
    ResultsPackagingController,
    ResultsPackagingUiError,
    ResultsPackagingUiSettings,
)
from ui.main_window import (
    GROUP_BEHAVIORAL_INFERENCE,
    GROUP_DATA_PREPARATION,
    GROUP_FEATURE_PROCESSING,
    GROUP_RESULTS_REVIEW,
    MainWindow,
)
from ui.pages.results_packaging_page import ResultsPackagingPage

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


def _eligible_patches(info: dict[str, Any]):
    def fake_validate(path: Path | str) -> dict[str, Any]:
        return {
            "s8_run_dir": Path(path).resolve(),
            "status": STATUS_COMPLETED,
            "session_root": info["session_root"].resolve(),
            "source_s7_run_dir": info["s7_run_dir"].resolve(),
            "source_s7_run_id": info["s7_run_id"],
            "source_s6_run_dir": info["s6_run_dir"].resolve(),
            "source_s6_run_id": info["s6_run_id"],
        }

    return (
        patch(
            "results_packaging.eligibility.validate_s8_run_for_downstream",
            side_effect=fake_validate,
        ),
        patch(
            "results_packaging.eligibility.is_s8_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.runner.validate_s8_run_for_downstream",
            side_effect=fake_validate,
            raising=False,
        ),
        patch(
            "behavioral_arbitration.runner.is_s8_run_eligible_for_downstream",
            return_value=True,
            raising=False,
        ),
        patch(
            "behavioral_arbitration.run_discovery.validate_s8_run_for_downstream",
            side_effect=fake_validate,
            raising=False,
        ),
        patch(
            "behavioral_arbitration.run_discovery.is_s8_run_eligible_for_downstream",
            return_value=True,
            raising=False,
        ),
    )


def _ready_controller(
    tmp_path: Path, **fixture_kwargs: Any
) -> tuple[ResultsPackagingController, dict[str, Any], ExitStack]:
    fixture = build_packaging_session(tmp_path, **fixture_kwargs)
    stack = ExitStack()
    for p in _eligible_patches(fixture):
        stack.enter_context(p)
    controller = ResultsPackagingController()
    controller.set_session(fixture["session_root"])
    return controller, fixture, stack


def _fake_result(fixture: dict[str, Any], *, run_id: str = "results_package__gui") -> ResultsPackageRunResult:
    run_dir = fixture["session_root"] / "results_package" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return ResultsPackageRunResult(
        success=True,
        status=STATUS_COMPLETED_WITH_WARNINGS,
        run_id=run_id,
        run_dir=run_dir,
        session_root=fixture["session_root"],
        source_s8_run_id=fixture["s8_run_id"],
        source_s8_run_dir=fixture["s8_run_dir"],
        run_meta_path=run_dir / "run_meta.json",
        settings_used_path=run_dir / "settings_used.yaml",
        processing_log_path=run_dir / "processing_log.txt",
        qc_path=run_dir / "qc.json",
        package_manifest_path=run_dir / "package_manifest.json",
        sync_index_path=run_dir / "sync_index.parquet",
        behavior_bouts_path=run_dir / "behavior_bouts.parquet",
        ethogram_summary_path=run_dir / "ethogram_summary.parquet",
        export_profile_dirs={REVIEW_EMIT_PROFILE_ID: run_dir / "export_profiles" / REVIEW_EMIT_PROFILE_ID},
        warnings=("inherited warning",),
        error_message=None,
        qc={
            "structural_ok": True,
            "warnings": ["Source S8 run completed_with_warnings (inherited)."],
            "failures": [],
            "informational": [
                "Optional profile review_emit_v1: included=1, excluded=0.",
                "Many-to-one video-frame mapping present (max multiplicity=2).",
            ],
            "metrics": {
                "n_bouts_packaged": 3,
                "channel_counts": {"exclusive": 2, "contact": 1, "side_by_side": 0},
                "sync_audit": {"n_dense": 48},
                "export_profiles": {
                    REVIEW_EMIT_PROFILE_ID: {"n_included": 1, "n_excluded": 2}
                },
            },
        },
        currently_eligible=True,
        classification="completed_with_warnings",
    )


def test_main_window_exposes_group4_s9(tmp_path: Path) -> None:
    controller, _fixture, stack = _ready_controller(tmp_path)
    with stack:
        window = MainWindow(
            PreprocessSetupController(),
            results_packaging_controller=controller,
        )
        window.set_application_group(GROUP_RESULTS_REVIEW)
        assert window.group_results_button.isVisibleTo(window)
        assert window.s9_button.isVisibleTo(window)
        assert window.s10_button.isVisibleTo(window)
        assert not window.s7_button.isVisibleTo(window)
        window._session_root = None
        window.pages.setCurrentWidget(window.results_packaging_page)
        window._ensure_group(GROUP_RESULTS_REVIEW)
        window._update_navigation()
        assert window.pages.currentWidget() is window.results_packaging_page
        window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
        assert window.s7_button.isVisibleTo(window)
        assert window.s8_button.isVisibleTo(window)
        assert not window.s9_button.isVisibleTo(window)
        assert not window.s10_button.isVisibleTo(window)
        window.set_application_group(GROUP_RESULTS_REVIEW)
        assert window.pages.currentWidget() is window.results_packaging_page


def test_group_navigation_preserves_session_across_groups(tmp_path: Path) -> None:
    controller, fixture, stack = _ready_controller(tmp_path)
    with stack:
        window = MainWindow(
            PreprocessSetupController(),
            results_packaging_controller=controller,
        )
        root = fixture["session_root"].resolve()
        window._session_root = None
        window.pages.setCurrentWidget(window.results_packaging_page)
        window._ensure_group(GROUP_RESULTS_REVIEW)
        assert controller.state.session_root == root
        window.set_application_group(GROUP_FEATURE_PROCESSING)
        window.set_application_group(GROUP_DATA_PREPARATION)
        window.set_application_group(GROUP_RESULTS_REVIEW)
        assert controller.state.session_root == root
        assert window.pages.currentWidget() is window.results_packaging_page


def test_s1_s8_regression_groups_still_present() -> None:
    window = MainWindow(PreprocessSetupController())
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert window.s7_button.isVisibleTo(window)
    assert window.s8_button.isVisibleTo(window)
    window.set_application_group(GROUP_RESULTS_REVIEW)
    assert window.s9_button.isVisibleTo(window)
    assert window.pages.currentWidget() is window.results_packaging_page


def test_missing_s8_blocks_packaging(tmp_path: Path) -> None:
    session = tmp_path / "empty"
    session.mkdir()
    controller = ResultsPackagingController()
    controller.set_session(session)
    assert not controller.can_run_packaging()
    assert controller.cannot_run_reason() is not None


def test_default_and_manual_s8_selection_preserved(tmp_path: Path) -> None:
    controller, fixture, stack = _ready_controller(tmp_path)
    with stack:
        selected = controller.selected_s8_source()
        assert selected is not None
        assert selected.s9_ready
        assert selected.run_id == fixture["s8_run_id"]
        assert not controller.state.s8_selection_manual
        controller.select_s8_source(selected.run_id, manual=True)
        assert controller.state.s8_selection_manual
        controller.refresh_discovery()
        assert controller.state.selected_s8_run_id == selected.run_id
        assert controller.state.s8_selection_manual


def test_default_full_package_and_optional_profile_settings(tmp_path: Path) -> None:
    controller, _fixture, stack = _ready_controller(tmp_path)
    with stack:
        settings = controller.state.working_settings
        assert settings.default_profile_id == "analysis_full"
        assert settings.enable_review_emit_v1 is False
        assert settings.review_emit_v1.include_contact is False
        assert settings.review_emit_v1.include_engaged is True
        assert settings.review_emit_v1.target_fps == 120.0
        assert settings.hash_prepared_video is False
        updated = ResultsPackagingUiSettings(
            enable_review_emit_v1=True,
            review_emit_v1=ReviewEmitV1Defaults(
                include_contact=False,
                include_engaged=True,
                include_side_by_side=True,
                min_emit_bout_frames=10,
                min_emit_bout_seconds=0.10,
                target_fps=120.0,
            ),
            require_prepared_video=True,
            hash_prepared_video=False,
        )
        controller.update_settings(updated)
        cfg = controller.state.working_settings.to_config()
        assert REVIEW_EMIT_PROFILE_ID in cfg.execution.selected_export_profiles
        rows = controller.settings_display_rows()
        assert any("Full package" in k for k, _ in rows)
        assert any("review_emit_v1" in k for k, _ in rows)


def test_immutable_settings_snapshot_and_duplicate_prevention(tmp_path: Path) -> None:
    controller, _fixture, stack = _ready_controller(tmp_path)
    with stack:
        settings = controller.state.working_settings
        identity = controller.begin_packaging_launch(settings)
        assert identity.settings.enable_review_emit_v1 == settings.enable_review_emit_v1
        with pytest.raises(ResultsPackagingUiError):
            controller.begin_packaging_launch(settings)
        with pytest.raises(ResultsPackagingUiError):
            controller.update_settings(settings)
        with pytest.raises(ResultsPackagingUiError):
            controller.select_s8_source(identity.s8_run_id, manual=True)
        controller.record_failure("abort", expected_identity=identity)


def test_group_navigation_allowed_during_s9_packaging(tmp_path: Path) -> None:
    controller, fixture, stack = _ready_controller(tmp_path)
    with stack:

        def _slow_runner(request: ResultsPackagingRequest, progress_callback=None):
            time.sleep(0.2)
            if progress_callback is not None:
                from preprocess.models import OperationProgress

                progress_callback(
                    OperationProgress(
                        phase="package",
                        message="working",
                        completed_units=1,
                        total_units=2,
                        total_is_estimate=False,
                        is_indeterminate=False,
                        processed_video_time_sec=None,
                    )
                )
            return _fake_result(fixture)

        controller._runner = _slow_runner
        page = ResultsPackagingPage(controller)
        window = MainWindow(
            PreprocessSetupController(),
            results_packaging_controller=controller,
        )
        window.results_packaging_page = page
        window._session_root = None
        page.show()
        page.refresh_from_state()
        page._run_packaging()
        assert controller.state.task_running
        window.set_application_group(GROUP_DATA_PREPARATION)
        window.set_application_group(GROUP_FEATURE_PROCESSING)
        window.set_application_group(GROUP_RESULTS_REVIEW)
        _wait_until(lambda: not page.has_running_task, timeout_sec=30.0)
        assert controller.state.last_result is not None


def test_full_packaging_launch_qc_and_reopen(tmp_path: Path) -> None:
    controller, fixture, stack = _ready_controller(tmp_path)
    with stack:

        def _runner(request: ResultsPackagingRequest, progress_callback=None):
            return _fake_result(fixture, run_id="results_package__reopen")

        controller._runner = _runner
        identity = controller.begin_packaging_launch()
        result = controller.execute_packaging_launch(identity)
        assert controller.record_success(result, expected_identity=identity, refresh_discovery=True)
        assert controller.state.selected_s9_run_id == result.run_id
        summary_rows = dict(controller.package_summary_rows())
        assert summary_rows["Total authoritative bouts (full package)"] == "3"
        assert "Optional profile" in summary_rows
        qc_rows = controller.qc_display_rows()
        labels = [name for name, _ in qc_rows]
        assert "Historical status" in labels
        assert "Current eligibility" in labels
        assert any(name == "Warning" for name, _ in qc_rows)
        assert any(name == "Information" for name, _ in qc_rows)
        assert "Structural error" not in labels


def test_page_progress_and_failure_presentation(tmp_path: Path) -> None:
    controller, fixture, stack = _ready_controller(tmp_path)
    with stack:

        def _failing_runner(request: ResultsPackagingRequest, progress_callback=None):
            raise RuntimeError("synthetic packaging failure")

        controller._runner = _failing_runner
        page = ResultsPackagingPage(controller)
        page.show()
        page.refresh_from_state()
        page._run_packaging()
        _wait_until(lambda: not page.has_running_task, timeout_sec=30.0)
        assert controller.state.run_error is not None
        assert "synthetic packaging failure" in controller.state.run_error


def test_settings_config_roundtrip() -> None:
    settings = ResultsPackagingUiSettings(
        enable_review_emit_v1=True,
        review_emit_v1=ReviewEmitV1Defaults(include_contact=True),
        hash_prepared_video=True,
    ).validated()
    cfg = settings.to_config()
    assert isinstance(cfg, ResultsPackagingConfig)
    restored = ResultsPackagingUiSettings.from_config(cfg)
    assert restored.enable_review_emit_v1 is True
    assert restored.review_emit_v1.include_contact is True
    assert restored.hash_prepared_video is True


def test_historical_status_vs_current_eligibility_rows(tmp_path: Path) -> None:
    controller, fixture, stack = _ready_controller(tmp_path)
    with stack:
        result = _fake_result(fixture)
        identity = controller.begin_packaging_launch()
        controller.record_success(result, expected_identity=identity, refresh_discovery=False)
        rows = dict(controller.qc_display_rows())
        assert rows["Historical status"] == STATUS_COMPLETED_WITH_WARNINGS
        assert "eligible" in rows["Current eligibility"] or "warnings" in rows["Current eligibility"]
