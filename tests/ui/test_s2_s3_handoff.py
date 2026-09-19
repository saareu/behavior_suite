"""Focused S2 → S3 GUI handoff: eligibility, path shape, open-vs-run."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from tests.tracking_correction.test_review import TIMESTAMP, _completed_s2

from pose_inference.run_discovery import (
    COMPLETE_REVIEWABLE,
    PoseInferenceProjectSummary,
    PoseInferenceRunSummary,
    S1HandoffStatus,
)
from tracking_correction.contracts import (
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    TrackingCorrectionResult,
)
from tracking_correction.run_discovery import find_reviewable_s3_run_for_s2
from tracking_correction.runner import run_tracking_correction, validate_s2_handoff
from ui.controllers.pose_inference_controller import (
    PoseInferenceController,
    S3PoseInput,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.tracking_review_controller import TrackingReviewController
from ui.main_window import MainWindow
from ui.pages.pose_inference_page import PoseInferencePage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _handoff_status(session_root: Path) -> S1HandoffStatus:
    preprocess = session_root / "preprocess"
    return S1HandoffStatus(
        session_root=session_root,
        preprocess_dir=preprocess,
        prepared_video=preprocess / "prepared_video.mp4",
        prepare_meta=preprocess / "prepare_meta.json",
        prepared_sync=preprocess / "prepared_sync.npz",
        artifact_presence={
            "prepared_video": True,
            "prepare_meta": True,
            "prepared_sync": True,
        },
        missing_files=(),
        is_complete=True,
    )


def _summary_for_s2(s2_run: Path, session_root: Path) -> PoseInferenceProjectSummary:
    run = PoseInferenceRunSummary(
        run_dir=s2_run,
        run_id=s2_run.name,
        status="completed",
        classification=COMPLETE_REVIEWABLE,
        artifact_presence={
            "pose_slp": (s2_run / "pose.slp").is_file(),
            "pose_parquet": True,
            "overlay_mp4": (s2_run / "overlay.mp4").is_file(),
            "pose_meta": True,
        },
        missing_required_artifacts=(),
        inference_mode="bottomup",
        pose_qc_outcome=None,
    )
    return PoseInferenceProjectSummary(
        session_root=session_root,
        s1_handoff=_handoff_status(session_root),
        runs=(run,),
    )


def _wait_until(predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


def test_selecting_valid_completed_s2_enables_s3_action(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    validate_s2_handoff(s2_run)

    controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root)
    )
    controller.set_session(session_root)
    controller.select_run(s2_run.name)
    page = PoseInferencePage(controller)
    page.refresh_from_state()

    assert controller.selected_run_can_feed_s3() is True
    assert page.continue_s3_button.isEnabled() is True
    assert page.continue_s3_button.text() == "Run Subsystem 3"
    assert "Subsystem 3 unavailable" not in page.selected_summary.text()
    page.close()


def test_invalid_s2_keeps_s3_disabled_with_reason(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    (s2_run / "pose.parquet").unlink()

    controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root)
    )
    controller.set_session(session_root)
    controller.select_run(s2_run.name)
    page = PoseInferencePage(controller)
    page.refresh_from_state()

    assert controller.selected_run_can_feed_s3() is False
    reason = controller.selected_run_s3_unavailable_reason()
    assert reason is not None
    assert "pose.parquet" in reason
    assert page.continue_s3_button.isEnabled() is False
    assert reason in page.selected_summary.text()
    page.close()


def test_handoff_passes_s2_run_directory_not_parquet(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root)
    )
    controller.set_session(session_root)
    controller.select_run(s2_run.name)

    handoff = controller.build_s3_handoff()

    assert handoff.selected_run_dir == s2_run.resolve()
    assert handoff.selected_run_dir.is_dir()
    assert handoff.selected_run_dir.name != "pose.parquet"
    assert handoff.pose_parquet_path == handoff.selected_run_dir / "pose.parquet"


def test_existing_s3_opens_without_rerun(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    created = run_tracking_correction(
        TrackingCorrectionRequest(s2_run_dir=s2_run, dry_run=False, timestamp=TIMESTAMP)
    )
    assert created.success
    found = find_reviewable_s3_run_for_s2(session_root, s2_run)
    assert found == created.run_dir.resolve()

    runner_calls: list[Path] = []

    def tracking_runner(request: TrackingCorrectionRequest) -> TrackingCorrectionResult:
        runner_calls.append(request.s2_run_dir)
        raise AssertionError("must not rerun when an S3 result already exists")

    controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root),
        s3_runner=tracking_runner,
    )
    controller.set_session(session_root)
    controller.select_run(s2_run.name)
    page = PoseInferencePage(controller)
    page.refresh_from_state()

    assert page.continue_s3_button.text() == "Open Subsystem 3"
    assert page.continue_s3_button.isEnabled() is True

    received: list[S3PoseInput] = []
    page.s3_handoff_requested.connect(received.append)
    page.continue_s3_button.click()
    _APPLICATION.processEvents()

    assert len(received) == 1
    assert received[0].existing_s3_run_dir == created.run_dir.resolve()
    assert received[0].selected_run_dir == s2_run.resolve()
    assert runner_calls == []
    page.close()


def test_new_s3_run_becomes_reviewable_via_main_window(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    release = threading.Event()

    def slow_runner(request: TrackingCorrectionRequest) -> TrackingCorrectionResult:
        release.wait(timeout=5)
        return run_tracking_correction(
            TrackingCorrectionRequest(
                s2_run_dir=request.s2_run_dir,
                dry_run=False,
                timestamp="20260301T150000",
                run_purpose=request.run_purpose,
            )
        )

    pose_controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root),
        s3_runner=slow_runner,
    )
    window = MainWindow(
        PreprocessSetupController(),
        pose_controller=pose_controller,
        review_controller=TrackingReviewController(),
    )
    window.show()
    window.open_pose_inference(session_root)
    pose_controller.select_run(s2_run.name)
    window.pose_page.refresh_from_state()
    assert window.pose_page.continue_s3_button.isEnabled() is True

    window.pose_page.continue_s3_button.click()
    _wait_until(lambda: window.pose_page.task_runner.is_running)
    release.set()
    _wait_until(lambda: not window.pose_page.task_runner.is_running, timeout_sec=30)
    _wait_until(lambda: window.pages.currentWidget() is window.s3_shell)

    assert window.review_page.controller.session is not None
    assert window.review_page.controller.session.run_dir.name.endswith("20260301T150000")
    window.close()


def test_s3_busy_indicator_disables_action_and_restores_on_success(
    tmp_path: Path,
) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    release = threading.Event()

    def slow_runner(request: TrackingCorrectionRequest) -> TrackingCorrectionResult:
        release.wait(timeout=5)
        return run_tracking_correction(
            TrackingCorrectionRequest(
                s2_run_dir=request.s2_run_dir,
                dry_run=False,
                timestamp="20260301T160000",
                run_purpose=request.run_purpose,
            )
        )

    pose_controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root),
        s3_runner=slow_runner,
    )
    window = MainWindow(
        PreprocessSetupController(),
        pose_controller=pose_controller,
        review_controller=TrackingReviewController(),
    )
    window.show()
    window.open_pose_inference(session_root)
    pose_controller.select_run(s2_run.name)
    page = window.pose_page
    page.refresh_from_state()

    page.continue_s3_button.click()
    _wait_until(lambda: page.task_runner.is_running)

    assert page.s3_progress.isHidden() is False
    assert page.s3_progress.minimum() == 0
    assert page.s3_progress.maximum() == 0
    assert page.continue_s3_button.isEnabled() is False
    assert "Running automatic tracking correction" in page.s3_stage_label.text()
    assert page.s3_stage_label.isHidden() is False

    release.set()
    _wait_until(lambda: not page.task_runner.is_running, timeout_sec=30)
    _wait_until(lambda: window.pages.currentWidget() is window.s3_shell)

    assert page.s3_progress.isHidden() is True
    assert page.s3_stage_label.isHidden() is True
    assert pose_controller.state.task_running is False
    window.close()


def test_s3_busy_indicator_restores_controls_on_failure(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    release = threading.Event()

    def failing_runner(_request: TrackingCorrectionRequest) -> TrackingCorrectionResult:
        release.wait(timeout=5)
        raise TrackingCorrectionError("simulated S3 failure")

    pose_controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root),
        s3_runner=failing_runner,
    )
    page = PoseInferencePage(pose_controller)
    pose_controller.set_session(session_root)
    pose_controller.select_run(s2_run.name)
    page.show()
    page.refresh_from_state()

    page.continue_s3_button.click()
    _wait_until(lambda: page.task_runner.is_running)
    assert page.s3_progress.isHidden() is False
    assert page.continue_s3_button.isEnabled() is False

    release.set()
    _wait_until(lambda: not page.task_runner.is_running, timeout_sec=5)

    assert page.s3_progress.isHidden() is True
    assert page.s3_stage_label.isHidden() is True
    assert pose_controller.state.task_running is False
    assert page.continue_s3_button.isEnabled() is True
    assert "simulated S3 failure" in page.error_label.text()
    page.close()


def test_single_selection_enables_s3_without_double_click(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    second = session_root / "pose_inference" / "run_b"
    second.mkdir(parents=True)
    (second / "pose.parquet").write_bytes(b"not-valid")
    (second / "pose_meta.json").write_text("{}", encoding="utf-8")

    first_summary = PoseInferenceRunSummary(
        run_dir=s2_run,
        run_id=s2_run.name,
        status="completed",
        classification=COMPLETE_REVIEWABLE,
        artifact_presence={
            "pose_slp": False,
            "pose_parquet": True,
            "overlay_mp4": False,
            "pose_meta": True,
        },
        missing_required_artifacts=(),
        inference_mode="bottomup",
        pose_qc_outcome=None,
    )
    second_summary = PoseInferenceRunSummary(
        run_dir=second,
        run_id=second.name,
        status="completed",
        classification=COMPLETE_REVIEWABLE,
        artifact_presence={
            "pose_slp": False,
            "pose_parquet": True,
            "overlay_mp4": False,
            "pose_meta": True,
        },
        missing_required_artifacts=(),
        inference_mode="bottomup",
        pose_qc_outcome="pass",
    )
    controller = PoseInferenceController(
        discovery=lambda _root: PoseInferenceProjectSummary(
            session_root=session_root,
            s1_handoff=_handoff_status(session_root),
            runs=(first_summary, second_summary),
        )
    )
    controller.set_session(session_root)
    page = PoseInferencePage(controller)
    page.show()
    _APPLICATION.processEvents()

    page.runs_table.selectRow(0)
    _APPLICATION.processEvents()
    assert controller.state.selected_run_id == s2_run.name
    assert page.continue_s3_button.isEnabled() is True

    page.runs_table.selectRow(1)
    _APPLICATION.processEvents()
    assert controller.state.selected_run_id == second.name
    assert page.continue_s3_button.isEnabled() is False
    reason = controller.selected_run_s3_unavailable_reason()
    assert reason is not None
    page.close()


def test_find_reviewable_s3_run_for_s2_matches_input_provenance(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    assert find_reviewable_s3_run_for_s2(session_root, s2_run) is None
    result = run_tracking_correction(
        TrackingCorrectionRequest(s2_run_dir=s2_run, dry_run=False, timestamp=TIMESTAMP)
    )
    assert find_reviewable_s3_run_for_s2(session_root, s2_run) == result.run_dir.resolve()


def test_incomplete_status_rejected_by_authoritative_validator(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    (s2_run / "pose_meta.json").write_text(
        '{"status": "running", "dry_run": false}\n',
        encoding="utf-8",
    )
    with pytest.raises(TrackingCorrectionError, match="not a completed"):
        validate_s2_handoff(s2_run)

    controller = PoseInferenceController(
        discovery=lambda _root: _summary_for_s2(s2_run, session_root)
    )
    controller.set_session(session_root)
    controller.select_run(s2_run.name)
    assert controller.selected_run_can_feed_s3() is False
    assert "not a completed" in (controller.selected_run_s3_unavailable_reason() or "")
