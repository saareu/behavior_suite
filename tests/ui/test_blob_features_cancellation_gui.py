"""S5 GUI Cancel enablement, feedback, and MainWindow deferred-close tests."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from blob_features import (
    BlobFeaturesRequest,
    BlobFeaturesResult,
    load_blob_features_config,
    run_blob_feature_extraction,
)
from blob_features.contracts import STATUS_CANCELED
from preprocess.models import OperationProgress
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from ui.controllers.blob_features_controller import (
    BlobFeaturesController,
    BlobFeaturesUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.blob_features_page import BlobFeaturesPage
from ui.tasks import GuiTaskContext, GuiTaskKind

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _write_eligible_s1(
    root: Path,
    *,
    frame_count: int = 6,
    fps: float = 20.0,
    size: tuple[int, int] = (64, 48),
) -> Path:
    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    width, height = size
    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    background = np.full((height, width, 3), 220, dtype=np.uint8)
    for _ in range(frame_count):
        frame = background.copy()
        frame[10:30, 8:28] = 5
        frame[10:30, 36:56] = 5
        writer.write(frame)
    writer.release()
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)
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


def _gated_cancelable_runner(
    *,
    worker_started: threading.Event,
    hold_until_cancel: bool = True,
):
    def runner(
        request: BlobFeaturesRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BlobFeaturesResult:
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="extracting_frames",
                    message="Processed prepared-video frame 1/100",
                    completed_units=1,
                    total_units=100,
                    is_indeterminate=False,
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
        return run_blob_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: True,
            publication_started=publication_started,
        )

    return runner


def test_cancel_button_enablement_and_feedback(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = _write_eligible_s1(tmp_path, frame_count=4)
    worker_started = threading.Event()
    controller = BlobFeaturesController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    page = BlobFeaturesPage(controller)
    page.controller.set_working_settings(_soft_settings())
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
    # A new run is permitted after worker completion.
    assert page.run_button.isEnabled() is True
    assert page.cancel_button.isEnabled() is False


def test_cancel_unavailable_in_publication_critical_section(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = _write_eligible_s1(tmp_path, frame_count=3)
    entered = threading.Event()
    release = threading.Event()

    def runner(
        request: BlobFeaturesRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BlobFeaturesResult:
        del cancellation_requested
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="writing",
                    message="Writing blob_features.parquet and QC artifacts",
                    completed_units=3,
                    total_units=3,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        return run_blob_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page = BlobFeaturesPage(BlobFeaturesController(runner=runner))
    page.controller.set_working_settings(_soft_settings())
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )
    assert page.cancel_button.isEnabled() is False
    page._cancel_extraction()
    assert "Publication in progress" in page.stage_label.text()
    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True


def test_mainwindow_deferred_close_requests_cancel(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = _write_eligible_s1(tmp_path, frame_count=4)
    worker_started = threading.Event()
    controller = BlobFeaturesController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    window = MainWindow(
        PreprocessSetupController(),
        blob_features_controller=controller,
    )
    window.open_blob_features(session)
    page = window.blob_features_page
    _wait_until(lambda: not page.has_running_task)
    page.controller.set_working_settings(_soft_settings())
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s5_task is True
    assert page._extraction_cancel_requested is True
    # Repeated close attempts remain idempotent.
    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted() is False
    assert window._close_after_s5_task is True

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"


def test_stale_extraction_failure_rejected_for_superseded_identity(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    first = _write_eligible_s1(tmp_path / "a", frame_count=3)
    controller = BlobFeaturesController()
    controller.set_session(first)
    controller.set_working_settings(_soft_settings())
    _request, identity = controller.build_request(timestamp="20260101T170000")
    from dataclasses import replace

    # A newer extraction identity is now active; the older callback is stale.
    controller.state.active_extraction_identity = replace(
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
        task_kind=GuiTaskKind.BLOB_FEATURES_EXTRACTION,
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
    session = _write_eligible_s1(tmp_path, frame_count=3)
    page = BlobFeaturesPage(BlobFeaturesController())
    page.controller.set_working_settings(_soft_settings())
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    completed_stage = page.stage_label.text()
    page._cancel_extraction()
    assert page.stage_label.text() == completed_stage
    assert page.cancel_button.isEnabled() is False
    assert page.run_button.isEnabled() is True


def test_close_during_active_cancellation_keeps_worker(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    session = _write_eligible_s1(tmp_path, frame_count=4)
    worker_started = threading.Event()
    controller = BlobFeaturesController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    window = MainWindow(
        PreprocessSetupController(),
        blob_features_controller=controller,
    )
    window.open_blob_features(session)
    page = window.blob_features_page
    _wait_until(lambda: not page.has_running_task)
    page.controller.set_working_settings(_soft_settings())
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    page.cancel_button.click()
    assert page._extraction_cancel_requested is True
    assert page.task_runner.is_running is True

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s5_task is True
    assert page.task_runner.is_running is True
    assert page.has_running_task is True

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
