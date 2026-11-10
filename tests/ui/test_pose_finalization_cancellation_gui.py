"""S4 GUI Cancel enablement, hard-navigation lock, and deferred-close tests."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from pose_finalization import (
    STATUS_CANCELED,
    PoseFinalizationRequest,
    PoseFinalizationResult,
    run_pose_finalization,
)
from preprocess.models import OperationProgress
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    FINAL_TRACKED_POSE_FILENAME,
)
from tracking_correction.contracts import (
    RUN_META_FILENAME as S3_RUN_META_FILENAME,
)
from ui.controllers.pose_finalization_controller import PoseFinalizationController
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.pose_finalization_page import PoseFinalizationPage
from ui.tasks import GuiTaskContext, GuiTaskKind

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
    n_frames: int = 20,
    run_id: str = "current_lab_two_mouse_headstage_v1__20260101T000000",
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
        "acceptance_state": ACCEPTANCE_ACCEPTED,
        "completed_at": "2026-01-01T00:00:00",
        "profile_id": "current_lab_two_mouse_headstage_v1",
        "acceptance": {
            "accepted_at": "2026-01-01T00:05:00",
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


def _wait_until(predicate, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


@pytest.fixture
def silence_modals(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))


class _CancelAfterNFalsePolls:
    """Return False for the first ``n`` polls, then True (post-bootstrap cancel)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


def _gated_cancelable_runner(
    *,
    worker_started: threading.Event,
    hold_until_cancel: bool = True,
):
    def runner(
        request: PoseFinalizationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> PoseFinalizationResult:
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="reconstructing_gaps",
                    message="Reconstructing gaps",
                    completed_units=3,
                    total_units=None,
                    is_indeterminate=True,
                )
            )
        if hold_until_cancel:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if cancellation_requested is not None and cancellation_requested():
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("cancel was never requested")
        # Skip the pre-bootstrap checkpoint so a canceled run directory is persisted.
        cancel_after_bootstrap = _CancelAfterNFalsePolls(1)
        return run_pose_finalization(
            request,
            progress_callback=progress_callback,
            cancellation_requested=cancel_after_bootstrap,
            publication_started=publication_started,
        )

    return runner


def test_cancel_button_enablement_and_feedback(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=16)
    worker_started = threading.Event()
    controller = PoseFinalizationController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    assert page.cancel_button.isEnabled() is False

    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)
    assert page.run_button.isEnabled() is False

    page.cancel_button.click()
    assert page.cancel_button.isEnabled() is False
    assert "Cancel requested" in page.stage_label.text()
    # Repeated Cancel clicks are harmless.
    page.cancel_button.click()
    page.cancel_button.click()

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED
    assert page.controller.state.summary is not None
    assert page.controller.state.summary.eligible_runs == ()
    assert page.run_button.isEnabled() is True
    assert page.cancel_button.isEnabled() is False
    page.close()


def test_cancel_unavailable_in_publication_critical_section(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=12)
    entered = threading.Event()
    release = threading.Event()

    def runner(
        request: PoseFinalizationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> PoseFinalizationResult:
        del cancellation_requested
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="writing_artifacts",
                    message="Writing artifacts",
                    completed_units=7,
                    total_units=None,
                    is_indeterminate=True,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        return run_pose_finalization(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page = PoseFinalizationPage(PoseFinalizationController(runner=runner))
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )
    assert page.cancel_button.isEnabled() is False
    page._cancel_finalization()
    assert "Publication in progress" in page.stage_label.text()
    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    page.close()


def test_hard_navigation_lock_survives_cancel_request(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=16)
    worker_started = threading.Event()
    controller = PoseFinalizationController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    window = MainWindow(
        PreprocessSetupController(),
        finalization_controller=controller,
    )
    window.open_pose_finalization(session)
    page = window.finalization_page
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    page.cancel_button.click()
    assert page._finalization_cancel_requested is True
    assert page.has_running_task is True
    assert window._hard_navigation_blocked() is True
    assert window.s5_button.isEnabled() is False
    assert window.group_feature_button.isEnabled() is False

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert window._hard_navigation_blocked() is False
    assert page.stage_label.text() == "Canceled"
    window.close()


def test_mainwindow_deferred_close_requests_cancel(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=16)
    worker_started = threading.Event()
    controller = PoseFinalizationController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    window = MainWindow(
        PreprocessSetupController(),
        finalization_controller=controller,
    )
    window.open_pose_finalization(session)
    page = window.finalization_page
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s4_task is True
    assert page._finalization_cancel_requested is True
    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted() is False
    assert window._close_after_s4_task is True

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"


def test_session_change_blocked_while_finalization_active(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    other = tmp_path / "other"
    _write_accepted_session(session, n_frames=12)
    _write_accepted_session(other, n_frames=12)
    worker_started = threading.Event()
    controller = PoseFinalizationController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    page.set_session(other)
    assert page.controller.state.session_root == session.resolve()
    page.cancel_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    page.close()


def test_stale_finalization_failure_rejected_for_superseded_identity(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=12)
    controller = PoseFinalizationController()
    controller.set_session(session)
    _request, identity = controller.build_request(timestamp="20260101T170000")
    from dataclasses import replace

    controller.state.active_finalization_identity = replace(
        identity, request_id="superseded-request"
    )
    assert (
        controller.record_failure(
            "canceled",
            expected_identity=identity,
            canceled=True,
        )
        is False
    )
    assert controller.state.run_stage != STATUS_CANCELED


def test_gui_task_context_publication_flag_defaults() -> None:
    context = GuiTaskContext(
        task_id="t",
        task_kind=GuiTaskKind.POSE_FINALIZATION,
        project_dir=None,
        raw_video_path=None,
        generation=0,
    )
    assert context.is_cancellation_requested() is False
    assert context.is_publication_critical_section() is False
    context.request_cancellation()
    context.request_cancellation()
    assert context.is_cancellation_requested() is True
    context.mark_publication_critical_section()
    assert context.is_publication_critical_section() is True


def test_cancel_after_task_completion_is_noop(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = tmp_path / "session"
    _write_accepted_session(session, n_frames=12)
    page = PoseFinalizationPage(PoseFinalizationController())
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    completed_stage = page.stage_label.text()
    page._cancel_finalization()
    assert page.stage_label.text() == completed_stage
    assert page.cancel_button.isEnabled() is False
    assert page.run_button.isEnabled() is True
    page.close()
