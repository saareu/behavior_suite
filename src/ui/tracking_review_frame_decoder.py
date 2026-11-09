"""Persistent S3 review frame decoder with worker-thread VideoCapture ownership.

GuiTaskRunner creates a new QThread per task. OpenCV ``VideoCapture`` handles are
not safe to share across those short-lived threads, and sequential playback on
NAS paths depends on keeping one capture warm. This module owns a dedicated
decoder thread for Subsystem 03 review only — not a shared video-player stack.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from preprocess.exceptions import VideoProbeError
from preprocess.video_probe import read_raw_frame_at_index
from tracking_correction.overlay_draw import draw_pose_overlay
from ui.controllers.tracking_review_controller import (
    FrameRenderRequest,
    FrameRenderSnapshot,
)

FrameReader = Callable[[Path, int], np.ndarray]

# Soft upper bound for cooperative shutdown waits. A stalled NAS seek may still
# hold the worker until OpenCV returns; we never force-terminate the thread.
DEFAULT_SHUTDOWN_WAIT_MS = 5_000


class _DecoderEngine(QObject):
    """Runs exclusively on the decoder QThread and owns the sequential capture."""

    snapshot_ready = Signal(object)
    decode_failed = Signal(object, str)
    became_idle = Signal()

    def __init__(
        self,
        *,
        frame_reader: FrameReader | None = None,
        use_sequential_capture: bool = True,
    ) -> None:
        super().__init__()
        self._frame_reader = frame_reader or read_raw_frame_at_index
        self._use_sequential_capture = use_sequential_capture
        self._capture: cv2.VideoCapture | None = None
        self._capture_path: Path | None = None
        self._capture_next_idx: int | None = None
        self._lock = threading.Lock()
        self._pending: FrameRenderRequest | None = None
        self._busy = False
        self._shutting_down = False

    def queue_request(self, request: FrameRenderRequest) -> None:
        """Thread-safe latest-wins enqueue (callable from the GUI thread)."""

        with self._lock:
            if self._shutting_down:
                return
            self._pending = request

    def mark_shutdown(self) -> None:
        """Thread-safe shutdown arming; capture release still happens on-thread."""

        with self._lock:
            self._shutting_down = True
            self._pending = None

    @Slot()
    def wake(self) -> None:
        """Start or continue draining on the decoder thread."""

        self._drain()

    @Slot()
    def release_capture(self) -> None:
        """Drop the warm capture without stopping the worker thread."""

        self._close_capture()

    @Slot()
    def shutdown(self) -> None:
        self.mark_shutdown()
        self._close_capture()

    def _drain(self) -> None:
        with self._lock:
            if self._busy or self._shutting_down:
                return
            self._busy = True
        try:
            while True:
                with self._lock:
                    request = self._pending
                    self._pending = None
                    shutting_down = self._shutting_down
                if shutting_down or request is None:
                    break
                try:
                    snapshot = self._render(request)
                except Exception as exc:  # noqa: BLE001 - surface to GUI thread
                    self.decode_failed.emit(request.identity, str(exc))
                else:
                    self.snapshot_ready.emit(snapshot)
        finally:
            with self._lock:
                self._busy = False
                again = self._pending is not None and not self._shutting_down
            if again:
                self._drain()
            else:
                self.became_idle.emit()

    def _render(self, request: FrameRenderRequest) -> FrameRenderSnapshot:
        try:
            raw = self._read_frame(request.video_path, request.identity.frame_idx)
        except VideoProbeError as exc:
            return FrameRenderSnapshot(
                identity=request.identity,
                image_bgr=None,
                error=str(exc),
            )
        image = draw_pose_overlay(
            raw,
            request.pose_points,
            skeleton_edges=request.skeleton_edges,
        )
        return FrameRenderSnapshot(
            identity=request.identity,
            image_bgr=image,
            error=None,
        )

    def _read_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        path = Path(video_path).resolve()
        if not self._use_sequential_capture:
            return self._frame_reader(path, frame_idx)

        if (
            self._capture is not None
            and self._capture_path == path
            and self._capture_next_idx is not None
            and frame_idx >= self._capture_next_idx
        ):
            while self._capture_next_idx < frame_idx:
                success, _discarded = self._capture.read()
                if not success:
                    self._close_capture()
                    break
                self._capture_next_idx += 1
            if self._capture is not None and self._capture_next_idx == frame_idx:
                success, frame = self._capture.read()
                if success and frame is not None:
                    self._capture_next_idx = frame_idx + 1
                    return frame.copy()
                self._close_capture()

        if frame_idx == 0:
            self._open_capture(path)
            assert self._capture is not None
            success, frame = self._capture.read()
            if not success or frame is None:
                self._close_capture()
                raise VideoProbeError("Prepared decode frame 0 is unreadable.")
            self._capture_next_idx = 1
            return frame.copy()

        # Non-sequential jump: verified random-access helper, then reopen for
        # subsequent sequential playback frames.
        frame = self._frame_reader(path, frame_idx)
        self._open_capture(path)
        assert self._capture is not None
        if self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1):
            self._capture_next_idx = frame_idx + 1
        else:
            self._close_capture()
        return frame

    def _open_capture(self, path: Path) -> None:
        self._close_capture()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoProbeError(f"Could not open prepared video: {path}")
        self._capture = capture
        self._capture_path = path
        self._capture_next_idx = 0

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._capture_path = None
        self._capture_next_idx = None


class _Bridge(QObject):
    """Queued-connection bridge so GUI callers never touch Qt slots cross-thread."""

    wake_requested = Signal()
    release_requested = Signal()
    shutdown_requested = Signal()


class TrackingReviewFrameDecoder(QObject):
    """GUI-thread facade for the dedicated S3 frame-decoder worker."""

    snapshot_ready = Signal(object)
    decode_failed = Signal(object, str)
    finished = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        frame_reader: FrameReader | None = None,
        use_sequential_capture: bool = True,
    ) -> None:
        super().__init__(parent)
        # Keep the QThread unparented so deleting this facade cannot destroy a
        # still-running thread during interpreter/Qt teardown.
        self._thread = QThread()
        self._engine = _DecoderEngine(
            frame_reader=frame_reader,
            use_sequential_capture=use_sequential_capture,
        )
        self._engine.moveToThread(self._thread)
        self._bridge = _Bridge()
        self._running = False
        self._shutting_down = False

        self._bridge.wake_requested.connect(self._engine.wake)
        self._bridge.release_requested.connect(self._engine.release_capture)
        self._bridge.shutdown_requested.connect(self._engine.shutdown)
        self._engine.snapshot_ready.connect(self._on_snapshot_ready)
        self._engine.decode_failed.connect(self._on_decode_failed)
        self._engine.became_idle.connect(self._on_became_idle)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    @property
    def is_running(self) -> bool:
        # A soft shutdown timeout must not look idle: the worker may still own
        # VideoCapture until OpenCV returns and the thread actually stops.
        return self._running or (self._shutting_down and self._thread.isRunning())

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def submit(self, request: FrameRenderRequest) -> None:
        if self._shutting_down:
            return
        self._running = True
        # Latest-wins even while the worker is blocked inside OpenCV seek/read.
        self._engine.queue_request(request)
        self._bridge.wake_requested.emit()

    def release_capture(self) -> None:
        """Ask the worker thread to release any open VideoCapture."""

        if self._shutting_down or not self._thread.isRunning():
            return
        self._bridge.release_requested.emit()

    def shutdown(self, *, wait_ms: int = DEFAULT_SHUTDOWN_WAIT_MS) -> bool:
        """Release capture on the worker thread and stop it cooperatively.

        Returns True when the thread stopped within ``wait_ms``. A stalled NAS
        seek may exceed this soft limit; the thread is left running rather than
        force-terminated. ``mark_shutdown`` is applied immediately so newer
        requests are ignored while OpenCV finishes the in-flight read.

        A False return is a soft timeout, not successful completion: ownership of
        the worker and its VideoCapture remains valid until the thread stops.
        """

        if self._shutting_down:
            if not self._thread.isRunning():
                self._running = False
                return True
            finished = bool(self._thread.wait(max(0, int(wait_ms))))
            if finished:
                self._running = False
                self._engine.moveToThread(self.thread())
            return finished

        self._shutting_down = True
        self._engine.mark_shutdown()
        if self._thread.isRunning():
            self._bridge.shutdown_requested.emit()
            self._thread.quit()
            finished = bool(self._thread.wait(max(0, int(wait_ms))))
        else:
            finished = True
        if finished:
            self._running = False
            # Thread has stopped; reclaim the engine onto the GUI thread before
            # Qt/interpreter teardown can touch a worker with dead affinity.
            self._engine.moveToThread(self.thread())
        else:
            # Soft timeout: keep the facade/thread/engine graph alive.
            self._running = True
        return finished

    @Slot(object)
    def _on_snapshot_ready(self, snapshot: object) -> None:
        if self._shutting_down:
            return
        self.snapshot_ready.emit(snapshot)

    @Slot(object, str)
    def _on_decode_failed(self, identity: object, message: str) -> None:
        if self._shutting_down:
            return
        self.decode_failed.emit(identity, message)

    @Slot()
    def _on_became_idle(self) -> None:
        if self._shutting_down:
            # Quit was already requested; remain "running" until the thread stops
            # so callers do not treat a stalled seek as completed shutdown.
            return
        self._running = False
        self.finished.emit()

    @Slot()
    def _on_thread_finished(self) -> None:
        self._running = False
        if not self._shutting_down:
            return
        # Eventual cleanup after a soft shutdown timeout: reclaim affinity and
        # notify observers that deferred application close may proceed.
        self._engine.moveToThread(self.thread())
        self.finished.emit()
