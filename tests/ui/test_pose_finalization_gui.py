"""Subsystem 04 GUI: discovery, settings validation, background run handoff."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from pose_finalization import (
    ELIGIBLE_DOWNSTREAM,
    PoseFinalizationRequest,
    PoseFinalizationResult,
    STALE_SOURCE,
    STATUS_COMPLETED,
    run_pose_finalization,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_NOT_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    FINAL_TRACKED_POSE_FILENAME,
    RUN_META_FILENAME as S3_RUN_META_FILENAME,
)
from ui.controllers.pose_finalization_controller import (
    PoseFinalizationController,
    PoseFinalizationUiError,
    PoseFinalizationUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.pose_finalization_page import PoseFinalizationPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sync(path: Path, *, frame_count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=120.0,
        raw_fps_effective=120.0,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / 120.0,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(path, sync)
    return path


def _write_accepted_session(
    session_root: Path,
    *,
    run_id: str = "current_lab_two_mouse_headstage_v1__20260101T000000",
    acceptance_state: str = ACCEPTANCE_ACCEPTED,
    n_frames: int = 24,
    completed_at: str = "2026-01-01T00:00:00",
    accepted_at: str = "2026-01-01T00:05:00",
) -> Path:
    session_root.mkdir(parents=True, exist_ok=True)
    sync = _write_sync(session_root / "preprocess" / "prepared_sync.npz", frame_count=n_frames)
    run_dir = session_root / "tracking_correction" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for frame_idx in range(n_frames):
        for track in (0, 1):
            for node in ("nose", "neck"):
                rows.append(
                    {
                        "frame": frame_idx,
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "track": track,
                        "node": node,
                        "x": float(track * 10 + frame_idx),
                        "y": float(track * 20 + 2 * frame_idx),
                        "point_score": 0.9,
                        "instance_score": 1.5,
                        "was_track_swapped": np.uint8(0),
                        "was_node_blanked": np.uint8(0),
                        "time_sec": frame_idx / 120.0,
                        "timestamp_source": "prepared_sync:prepared_time_sec",
                        "video_index": 0,
                        "video_fps": 120.0,
                    }
                )
    pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    pd.DataFrame(rows).to_parquet(pose_path, index=False)
    digest = _sha256(pose_path)
    meta = {
        "run_id": run_id,
        "status": "correction_complete",
        "backend_status": "completed",
        "acceptance_state": acceptance_state,
        "completed_at": completed_at,
        "profile_id": "current_lab_two_mouse_headstage_v1",
        "acceptance": {
            "accepted_at": accepted_at,
            "tracked_pose_sha256": digest,
        },
        "input": {
            "session_root": str(session_root),
            "prepared_sync": str(sync),
        },
    }
    (run_dir / S3_RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def _wait_until(predicate, timeout_sec: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


def test_controller_auto_selects_latest_accepted_s3(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_accepted_session(
        session,
        run_id="older__20260101T000000",
        completed_at="2026-01-01T00:00:00",
        accepted_at="2026-01-01T00:05:00",
    )
    newest = _write_accepted_session(
        session,
        run_id="newer__20260102T000000",
        completed_at="2026-01-02T00:00:00",
        accepted_at="2026-01-02T00:05:00",
    )
    controller = PoseFinalizationController()
    controller.set_session(session)
    selected = controller.selected_accepted_s3()
    assert selected is not None
    assert selected.run_id == newest.name
    assert controller.can_run_finalization() is True


def test_controller_explains_missing_unaccepted_superseded(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir(parents=True, exist_ok=True)
    controller = PoseFinalizationController()
    controller.set_session(session)
    reason = controller.cannot_run_reason()
    assert reason is not None
    assert "accepted" in reason.lower()

    _write_accepted_session(
        session,
        run_id="reviewable__20260101T000000",
        acceptance_state=ACCEPTANCE_NOT_ACCEPTED,
    )
    controller.refresh_discovery()
    reason = controller.cannot_run_reason()
    assert reason is not None
    assert "accepted" in reason.lower()

    _write_accepted_session(
        session,
        run_id="superseded__20260101T010000",
        acceptance_state=ACCEPTANCE_SUPERSEDED,
    )
    controller.refresh_discovery()
    assert controller.selected_accepted_s3() is None


def test_advanced_settings_validation_and_defaults(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_accepted_session(session)
    controller = PoseFinalizationController()
    controller.set_session(session)
    defaults = controller.state.settings
    assert defaults.linear_max_frames == 20
    assert defaults.kalman_max_frames == 140
    assert defaults.smooth_enabled is True
    assert defaults.savgol_window_length == 11
    assert defaults.savgol_polyorder == 2

    with pytest.raises(PoseFinalizationUiError, match="kalman_max_frames|odd|polyorder"):
        controller.update_settings(
            PoseFinalizationUiSettings(linear_max_frames=50, kalman_max_frames=10)
        )
    with pytest.raises(PoseFinalizationUiError, match="odd"):
        controller.update_settings(
            PoseFinalizationUiSettings(savgol_window_length=10)
        )
    request = controller.build_request(timestamp="20261019T200000")
    assert request.gap_fill_config is not None
    assert request.gap_fill_config.linear_max_frames == 20
    assert request.smooth_config is not None
    assert request.smooth_config.enabled is True


def test_page_run_finalization_background_and_qc(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=20)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    assert page.run_button.isEnabled() is True

    page._run_finalization()
    assert page.has_running_task is True
    assert page.run_button.isEnabled() is False

    _wait_until(lambda: not page.has_running_task)
    assert controller.state.last_result is not None
    assert controller.state.last_result.status in (
        STATUS_COMPLETED,
        "completed_with_warnings",
    )
    assert controller.state.last_result.final_pose_path is not None
    rows = dict(controller.qc_display_rows())
    assert "Status" in rows
    assert "Output" in rows
    page.close()


def test_page_duplicate_launch_blocked(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=30)

    def slow_runner(request: PoseFinalizationRequest) -> PoseFinalizationResult:
        time.sleep(0.4)
        return run_pose_finalization(request)

    controller = PoseFinalizationController(runner=slow_runner)
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    page._run_finalization()
    assert page.has_running_task is True
    page._run_finalization()  # second click while busy
    assert page.task_runner.is_running is True
    _wait_until(lambda: not page.has_running_task)
    page.close()


def test_completed_run_discovery_eligible_vs_stale(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=20)
    result = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20261019T201000")
    )
    controller = PoseFinalizationController()
    controller.set_session(session)
    summary = controller.state.summary
    assert summary is not None
    assert any(run.run_id == result.run_id for run in summary.eligible_runs)
    assert summary.eligible_runs[0].classification == ELIGIBLE_DOWNSTREAM

    meta_path = s3_run / S3_RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    controller.refresh_discovery()
    stale = controller.state.summary
    assert stale is not None
    match = next(run for run in stale.runs if run.run_id == result.run_id)
    assert match.classification == STALE_SOURCE
    assert result.final_pose_path is not None
    assert result.final_pose_path.is_file()


def test_main_window_exposes_s4_navigation(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=16)
    window = MainWindow(PreprocessSetupController())
    window.open_pose_finalization(session)
    assert window.pages.currentWidget() is window.s4_shell
    assert window.finalization_page.controller.selected_accepted_s3() is not None
    assert window.s4_button.isEnabled() is True
    window.close()


def test_failed_run_not_marked_completed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=16)

    def boom(_request: PoseFinalizationRequest) -> PoseFinalizationResult:
        raise RuntimeError("forced failure")

    controller = PoseFinalizationController(runner=boom)
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    page._run_finalization()
    _wait_until(lambda: not page.has_running_task)
    assert controller.state.last_result is None
    assert controller.state.run_stage == "Failed"
    assert controller.state.run_error is not None
    assert "forced failure" in controller.state.run_error
    page.close()
