"""Visible Windows GUI smoke for PQ Sprint 3 Increment 3 (S4 cancellation).

Uses a disposable synthetic accepted-S3 session plus an instrumented slow worker
so Cancel can be exercised reliably (real synthetic S4 finalization can finish
before a human click). Does not modify scientific pose algorithms and does not
touch lab/test_cases scientific artifacts.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Force the native Windows Qt platform for visible validation.
os.environ.pop("QT_QPA_PLATFORM", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from pose_finalization import (  # noqa: E402
    STATUS_CANCELED,
    PoseFinalizationRequest,
    PoseFinalizationResult,
    is_s4_run_eligible_for_downstream,
    run_pose_finalization,
    summarize_pose_finalization_project,
)
from pose_finalization.contracts import (  # noqa: E402
    FINAL_POSE_FILENAME,
    RUN_META_FILENAME,
)
from preprocess.models import OperationProgress  # noqa: E402
from preprocess.sync_writer import (  # noqa: E402
    build_prepared_sync,
    write_prepared_sync_npz,
)
from tracking_correction.contracts import (  # noqa: E402
    ACCEPTANCE_ACCEPTED,
    FINAL_TRACKED_POSE_FILENAME,
    RUN_META_FILENAME as S3_RUN_META_FILENAME,
)
from ui.controllers.pose_finalization_controller import (  # noqa: E402
    PoseFinalizationController,
)
from ui.controllers.preprocess_setup_controller import (  # noqa: E402
    PreprocessSetupController,
)
from ui.main_window import MainWindow  # noqa: E402

WORK = ROOT / "tmp_pq_sprint3_s4_cancel_gui"
EVIDENCE = ROOT / "docs" / "roadmap" / "_pq_sprint3_s4_cancel_visible_gui_evidence.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_until(app: QApplication, predicate, timeout_sec: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for GUI state")
        time.sleep(0.02)
    app.processEvents()


def _write_accepted_session(session_root: Path, *, n_frames: int = 24) -> Path:
    session_root.mkdir(parents=True, exist_ok=True)
    sync_path = session_root / "preprocess" / "prepared_sync.npz"
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    sync = build_prepared_sync(
        prepared_frame_count=n_frames,
        start_frame=0,
        end_frame_exclusive=n_frames,
        fps_header=120.0,
        raw_fps_effective=120.0,
        raw_pts_time_sec=np.arange(n_frames, dtype=float) / 120.0,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=n_frames,
        prepared_frame_count_opencv_reported=n_frames,
        prepared_frame_count_opencv_readable=n_frames,
    )
    write_prepared_sync_npz(sync_path, sync)
    run_id = "current_lab_two_mouse_headstage_v1__20260101T000000"
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
            "prepared_sync": str(sync_path),
        },
    }
    (run_dir / S3_RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


class _CancelAfterNFalsePolls:
    """Return False for the first ``n`` polls, then True (post-bootstrap cancel)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


def _gated_runner(worker_started: threading.Event):
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
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if cancellation_requested is not None and cancellation_requested():
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("cancel was never requested")
        # Skip the pre-bootstrap checkpoint so a canceled run directory is persisted.
        return run_pose_finalization(
            request,
            progress_callback=progress_callback,
            cancellation_requested=_CancelAfterNFalsePolls(1),
            publication_started=publication_started,
        )

    return runner


def main() -> int:
    lines: list[str] = [
        "PQ Sprint 3 Increment 3 — S4 visible GUI cancellation smoke",
        f"work_root: {WORK}",
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    session = WORK / "session"
    if session.exists():
        # Disposable evidence workspace only.
        for path in sorted(session.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        session.rmdir()

    try:
        _write_accepted_session(session, n_frames=20)
        completed = run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=session
                / "tracking_correction"
                / "current_lab_two_mouse_headstage_v1__20260101T000000",
                run_purpose="smoke_prior_completed",
                timestamp="20260101T090000",
            )
        )
        assert completed.final_pose_path is not None
        assert is_s4_run_eligible_for_downstream(completed.run_dir)

        app = QApplication.instance() or QApplication([])
        for name in ("warning", "information", "critical", "question"):
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))

        worker_started = threading.Event()
        controller = PoseFinalizationController(
            runner=_gated_runner(worker_started)
        )
        window = MainWindow(
            PreprocessSetupController(),
            finalization_controller=controller,
        )
        window.show()
        window.open_pose_finalization(session)
        page = window.finalization_page
        _wait_until(app, lambda: not page.has_running_task)

        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        assert window._hard_navigation_blocked() is True
        lines.append("PASS: Cancel enabled during cancellable S4 processing")
        lines.append("PASS: hard navigation locked while worker active")

        page.cancel_button.click()
        assert page.cancel_button.isEnabled() is False
        assert "Cancel requested" in page.stage_label.text()
        assert window._hard_navigation_blocked() is True
        lines.append(
            "PASS: Cancel requested feedback visible; navigation still locked"
        )

        _wait_until(app, lambda: not page.has_running_task, timeout_sec=60.0)
        assert page.stage_label.text() == "Canceled"
        assert page.controller.state.run_stage == STATUS_CANCELED
        summary = summarize_pose_finalization_project(session)
        canceled_runs = [r for r in summary.runs if r.status == STATUS_CANCELED]
        assert canceled_runs
        assert not (canceled_runs[0].run_dir / FINAL_POSE_FILENAME).exists()
        assert (canceled_runs[0].run_dir / RUN_META_FILENAME).is_file()
        assert is_s4_run_eligible_for_downstream(canceled_runs[0].run_dir) is False
        assert is_s4_run_eligible_for_downstream(completed.run_dir) is True
        assert summary.eligible_runs
        assert any(r.run_id == completed.run_id for r in summary.eligible_runs)
        lines.append("PASS: worker-confirmed Canceled; unpublished artifacts removed")
        lines.append("PASS: canceled run ineligible; prior completed remains eligible")

        # Second run after cancel.
        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        page.cancel_button.click()
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted() is False
        assert window._close_after_s4_task is True
        assert page.task_runner.is_running is True
        lines.append("PASS: close during active cancel defers shutdown; worker retained")
        _wait_until(app, lambda: not page.has_running_task, timeout_sec=60.0)
        lines.append("PASS: deferred close waited for worker completion")
        lines.append("RESULT: PASS")
        EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        window.close()
        print("\n".join(lines))
        return 0
    except Exception:
        lines.append("RESULT: FAIL")
        lines.append(traceback.format_exc())
        EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
