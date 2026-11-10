"""Visible Windows GUI smoke for PQ Sprint 3 Increment 1 (S5 cancellation).

Creates a disposable synthetic eligible S1 session, exercises Cancel mid-run and
deferred close in the real Windows Qt GUI, and writes evidence. Does not touch
lab/test_cases scientific artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# Force the native Windows Qt platform for visible validation.
os.environ.pop("QT_QPA_PLATFORM", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from blob_features import (  # noqa: E402
    STATUS_CANCELED,
    is_s5_run_eligible_for_downstream,
    load_blob_features_config,
    run_blob_feature_extraction,
    summarize_blob_features_project,
)
from blob_features.contracts import (  # noqa: E402
    BLOB_FEATURES_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
)
from preprocess.sync_writer import (  # noqa: E402
    build_prepared_sync,
    write_prepared_sync_npz,
)
from ui.controllers.blob_features_controller import (  # noqa: E402
    BlobFeaturesUiSettings,
)
from ui.controllers.preprocess_setup_controller import (  # noqa: E402
    PreprocessSetupController,
)
from ui.main_window import MainWindow  # noqa: E402

WORK = ROOT / "tmp_pq_sprint3_s5_cancel_gui"
EVIDENCE = ROOT / "docs" / "roadmap" / "_pq_sprint3_s5_cancel_visible_gui_evidence.txt"
FRAME_COUNT = 48


def _soft_settings() -> BlobFeaturesUiSettings:
    from blob_features import BlobSegmentationSettings

    config = load_blob_features_config()
    soft = BlobSegmentationSettings(
        blob_threshold=-1,
        blob_blur_kernel_size=5,
        blob_kernel_size=5,
        blob_tail_open_kernel_size=3,
        blob_min_area=20,
        blob_max_area=0,
        blob_prune_min_radius=1,
        blob_prune_regrow_iters=1,
        blob_prune_max_regrow_dist=2,
        blob_post_prune_min_area=20,
        blob_post_prune_max_area=0,
        blob_final_smooth_kernel_size=3,
        abs_dark_threshold=250,
    )
    return BlobFeaturesUiSettings(
        segmentation=soft,
        temporal=config.temporal,
        execution=config.execution,
    ).validated()


def _write_disposable_session(root: Path, *, frame_count: int = FRAME_COUNT) -> Path:
    session = root / "disposable_session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True, exist_ok=True)
    width, height = 96, 64
    fps = 20.0
    background = np.full((height, width, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)
    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame_idx in range(frame_count):
        frame = background.copy()
        if frame_idx % 2 == 0:
            frame[12:36, 10:34] = 5
            frame[12:36, 55:79] = 5
        else:
            cv2.ellipse(frame, (48, 32), (22, 12), 0, 0, 360, (5, 5, 5), -1)
        writer.write(frame)
    writer.release()
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "width": width,
                    "height": height,
                    "opencv_reported_frame_count": frame_count,
                    "opencv_readable_frame_count": frame_count,
                    "frame_count_used_for_sleap": frame_count,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
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
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    return session


def _wait_until(app: QApplication, predicate, timeout_sec: float, label: str) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(0.02)
    raise TimeoutError(label)


def _visible_pause(app: QApplication, seconds: float = 1.2) -> None:
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        app.processEvents()
        time.sleep(0.05)


def main() -> int:
    notes: list[str] = []
    qt_warnings: list[str] = []

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        text = warnings.formatwarning(message, category, filename, lineno, line)
        qt_warnings.append(text.strip())
        notes.append(f"python_warning={text.strip()!r}")

    warnings.showwarning = _showwarning  # type: ignore[assignment]

    if WORK.exists():
        # Disposable workspace only.
        for path in sorted(WORK.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        WORK.rmdir()
    WORK.mkdir(parents=True)

    session = _write_disposable_session(WORK)
    notes.append(f"session={session}")
    notes.append(f"frame_count={FRAME_COUNT}")

    # Seed one completed run so canceled output cannot mask prior eligibility.
    from blob_features import (
        BlobExecutionSettings,
        BlobFeaturesConfig,
        BlobFeaturesRequest,
        BlobSegmentationSettings,
        BlobTemporalSettings,
    )

    soft_config = BlobFeaturesConfig(
        segmentation=BlobSegmentationSettings(
            blob_threshold=-1,
            blob_blur_kernel_size=5,
            blob_kernel_size=5,
            blob_tail_open_kernel_size=3,
            blob_min_area=20,
            blob_max_area=0,
            blob_prune_min_radius=1,
            blob_prune_regrow_iters=1,
            blob_prune_max_regrow_dist=2,
            blob_post_prune_min_area=20,
            blob_post_prune_max_area=0,
            blob_final_smooth_kernel_size=3,
            abs_dark_threshold=250,
        ).validated(),
        temporal=BlobTemporalSettings(cc_count_gap_fill_max_gap=2).validated(),
        execution=BlobExecutionSettings(n_workers=1, max_in_flight=4).validated(),
    ).validated()
    prior = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=soft_config,
            run_purpose="pq_sprint3_prior_completed",
            timestamp="20261109T180000",
        )
    )
    notes.append(f"prior_run_id={prior.run_id}")
    notes.append(f"prior_status={prior.status}")
    notes.append(f"prior_eligible={is_s5_run_eligible_for_downstream(prior.run_dir)}")

    app = QApplication.instance() or QApplication(sys.argv)
    notes.append(f"platform={app.platformName()}")
    # Auto-dismiss informational cancel dialogs during unattended smoke.
    QMessageBox.information = staticmethod(lambda *a, **k: 0)  # type: ignore[method-assign]
    QMessageBox.warning = staticmethod(lambda *a, **k: 0)  # type: ignore[method-assign]

    window = MainWindow(PreprocessSetupController())
    window.setWindowTitle("PQ Sprint 3 — S5 cancel visible smoke")
    window.resize(1280, 860)
    window.show()
    window.raise_()
    window.activateWindow()
    window.open_blob_features(session)
    page = window.blob_features_page
    page.controller.set_working_settings(_soft_settings())
    _wait_until(app, lambda: not page.has_running_task, 60.0, "initial open idle")
    notes.append(
        "pre_cancel_eligible="
        f"{len(page.controller.state.summary.eligible_runs) if page.controller.state.summary else 0}"
    )
    _visible_pause(app, 1.0)

    # --- Scenario A: cancel while frames are processing ---
    page.run_button.click()
    _wait_until(app, lambda: page.cancel_button.isEnabled(), 30.0, "cancel enabled")
    notes.append("cancel_enabled_after_start=True")
    # Wait until extraction progress is visibly advancing.
    _wait_until(
        app,
        lambda: "frame" in page.stage_label.text().lower()
        or page.progress.value() > 0,
        60.0,
        "frames processing",
    )
    notes.append(f"stage_before_cancel={page.stage_label.text()!r}")
    page.cancel_button.click()
    cancel_disabled = not page.cancel_button.isEnabled()
    notes.append(f"cancel_disabled_after_request={cancel_disabled}")
    notes.append(f"inline_feedback={page.stage_label.text()!r}")
    assert cancel_disabled
    assert "Cancel requested" in page.stage_label.text()
    _wait_until(app, lambda: not page.has_running_task, 120.0, "worker canceled")
    notes.append(f"stage_after_cancel={page.stage_label.text()!r}")
    notes.append(f"run_stage={page.controller.state.run_stage!r}")
    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED

    summary = summarize_blob_features_project(session)
    canceled_runs = [
        run
        for run in summary.runs
        if Path(run.run_dir).name != prior.run_dir.name
    ]
    assert canceled_runs, "expected a canceled run directory"
    canceled = max(canceled_runs, key=lambda r: Path(r.run_dir).name)
    meta = json.loads((Path(canceled.run_dir) / RUN_META_FILENAME).read_text(encoding="utf-8"))
    log_path = Path(canceled.run_dir) / "processing_log.txt"
    notes.append(f"canceled_run_id={canceled.run_id}")
    notes.append(f"canceled_meta_status={meta.get('status')!r}")
    notes.append(f"canceled_meta_error={meta.get('error')!r}")
    notes.append(f"canceled_has_parquet={(Path(canceled.run_dir) / BLOB_FEATURES_FILENAME).exists()}")
    notes.append(f"canceled_has_qc={(Path(canceled.run_dir) / QC_FILENAME).exists()}")
    notes.append(f"canceled_log_exists={log_path.is_file()}")
    notes.append(
        f"canceled_eligible={is_s5_run_eligible_for_downstream(canceled.run_dir)}"
    )
    notes.append(
        f"prior_still_eligible={is_s5_run_eligible_for_downstream(prior.run_dir)}"
    )
    notes.append(f"summary_eligible_count={len(summary.eligible_runs)}")
    assert meta["status"] == STATUS_CANCELED
    assert not (Path(canceled.run_dir) / BLOB_FEATURES_FILENAME).exists()
    assert is_s5_run_eligible_for_downstream(canceled.run_dir) is False
    assert is_s5_run_eligible_for_downstream(prior.run_dir) is True
    assert page.run_button.isEnabled() is True
    _visible_pause(app, 1.0)

    # --- Scenario A2: another extraction can start afterward (cancel again) ---
    page.run_button.click()
    _wait_until(app, lambda: page.cancel_button.isEnabled(), 30.0, "second cancel enabled")
    page.cancel_button.click()
    _wait_until(app, lambda: not page.has_running_task, 120.0, "second cancel done")
    notes.append(f"second_run_stage={page.controller.state.run_stage!r}")
    notes.append(f"run_enabled_after_second={page.run_button.isEnabled()}")
    assert page.controller.state.run_stage == STATUS_CANCELED
    assert page.run_button.isEnabled() is True
    _visible_pause(app, 0.8)

    # --- Scenario B: close while extraction is running ---
    page.run_button.click()
    _wait_until(app, lambda: page.cancel_button.isEnabled(), 30.0, "close-scenario start")
    worker_alive_before_close = page.task_runner.is_running
    notes.append(f"worker_running_before_close={worker_alive_before_close}")
    event = QCloseEvent()
    window.closeEvent(event)
    notes.append(f"close_accepted={event.isAccepted()}")
    notes.append(f"close_after_s5={window._close_after_s5_task}")
    notes.append(f"worker_running_after_close={page.task_runner.is_running}")
    notes.append(f"shutdown_feedback={page.stage_label.text()!r}")
    assert event.isAccepted() is False
    assert window._close_after_s5_task is True
    assert page.task_runner.is_running is True
    _wait_until(app, lambda: not page.has_running_task, 120.0, "deferred close worker done")
    # MainWindow closes itself via _s5_task_finished -> close().
    _wait_until(app, lambda: not window.isVisible(), 30.0, "window closed after worker")
    notes.append("deferred_close_completed=True")
    notes.append(f"qt_warning_count={len(qt_warnings)}")
    for warning in qt_warnings:
        notes.append(f"qt_warning={warning!r}")

    EVIDENCE.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"evidence={EVIDENCE}")
    # Keep process alive briefly so the closed state is observable.
    _visible_pause(app, 0.5)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
