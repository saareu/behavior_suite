"""Headless coordination for Subsystem 03 tracking review and acceptance."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocess.exceptions import VideoProbeError
from preprocess.video_probe import read_raw_frame_at_index
from tracking_correction.contracts import TrackingCorrectionError
from tracking_correction.overlay_draw import draw_pose_overlay
from tracking_correction.review import (
    AcceptanceResult,
    CorrectionEpisode,
    PoseDisplaySource,
    PoseNodePoint,
    TrackingReviewSession,
    load_review_session,
)
from tracking_correction.run_discovery import (
    TrackingCorrectionProjectSummary,
    is_completed_s3_run_dir,
    summarize_tracking_correction_project,
)

FrameReader = Callable[[Path, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class ReviewFrameView:
    """One prepared-frame display payload for the review page."""

    frame_idx: int
    frame_count: int
    pose_source: str
    acceptance_state: str
    correction_types: tuple[str, ...]
    correction_episode: CorrectionEpisode | None
    pose_points: tuple[PoseNodePoint, ...]
    skeleton_edges: tuple[tuple[str, str], ...]
    image_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class PlaybackTimingState:
    """Wall-clock playback schedule for one play session."""

    active: bool
    anchor_frame: int
    anchor_monotonic: float
    source_fps: float
    playback_speed: float

    def target_frame(self, now_monotonic: float) -> int:
        elapsed = max(0.0, float(now_monotonic) - self.anchor_monotonic)
        return int(self.anchor_frame + elapsed * self.source_fps * self.playback_speed)


class TrackingReviewController:
    """Load an S3 run, navigate prepared frames, and accept tracking."""

    def __init__(
        self,
        *,
        frame_reader: FrameReader | None = None,
    ) -> None:
        self._frame_reader = frame_reader or read_prepared_frame
        self._use_sequential_capture = frame_reader is None
        self.session: TrackingReviewSession | None = None
        self.project: TrackingCorrectionProjectSummary | None = None
        self._last_error: str | None = None
        self._playback: PlaybackTimingState | None = None
        self._capture: cv2.VideoCapture | None = None
        self._capture_path: Path | None = None
        self._capture_next_idx: int | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def playback_active(self) -> bool:
        return self._playback is not None and self._playback.active

    def open_project(self, session_root: Path) -> TrackingCorrectionProjectSummary:
        """Discover S3 runs under a project/session directory."""

        try:
            summary = summarize_tracking_correction_project(session_root)
        except OSError as exc:
            self._last_error = str(exc)
            raise TrackingCorrectionError(f"Could not open project directory: {exc}") from exc
        self.project = summary
        self._last_error = None
        return summary

    def open_path(self, path: Path) -> TrackingReviewSession:
        """Open a session root or a direct completed S3 run directory."""

        target = Path(path).expanduser().resolve()
        if is_completed_s3_run_dir(target):
            return self.open_run(target)
        summary = self.open_project(target)
        reviewable = summary.reviewable_runs
        if not reviewable:
            raise TrackingCorrectionError(
                "No completed S3 runs were found under "
                f"{summary.session_root / 'tracking_correction'}."
            )
        if len(reviewable) == 1:
            return self.open_run(reviewable[0].run_dir)
        raise TrackingCorrectionError(
            "Multiple completed S3 runs are available; select one explicitly."
        )

    def open_run(self, s3_run_dir: Path) -> TrackingReviewSession:
        """Load a completed automatic S3 run into the review session."""

        self.stop_playback()
        self._close_capture()
        try:
            self.session = load_review_session(s3_run_dir)
        except TrackingCorrectionError as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None
        if self.session.session_root is not None:
            with suppress(OSError):
                self.project = summarize_tracking_correction_project(self.session.session_root)
        return self.session

    def require_session(self) -> TrackingReviewSession:
        if self.session is None:
            raise TrackingCorrectionError("No S3 review session is open.")
        return self.session

    def set_frame(self, frame_idx: int) -> int:
        return self.require_session().set_frame(frame_idx)

    def step_forward(self) -> int:
        self.stop_playback()
        return self.require_session().step_forward()

    def step_backward(self) -> int:
        self.stop_playback()
        return self.require_session().step_backward()

    def next_correction_frame(self) -> int | None:
        return self.require_session().next_correction_frame()

    def previous_correction_frame(self) -> int | None:
        return self.require_session().previous_correction_frame()

    def next_correction_episode(self) -> CorrectionEpisode | None:
        return self.require_session().next_correction_episode()

    def previous_correction_episode(self) -> CorrectionEpisode | None:
        return self.require_session().previous_correction_episode()

    def set_pose_source(self, source: PoseDisplaySource | str) -> PoseDisplaySource:
        return self.require_session().set_pose_source(source)

    def set_playback_speed(self, speed: float) -> float:
        return self.require_session().set_playback_speed(speed)

    def begin_playback(self, *, now_monotonic: float | None = None) -> PlaybackTimingState:
        """Anchor wall-clock playback to the current prepared frame."""

        session = self.require_session()
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        self._playback = PlaybackTimingState(
            active=True,
            anchor_frame=session.current_frame,
            anchor_monotonic=now,
            source_fps=session.fps_header,
            playback_speed=session.playback_speed,
        )
        return self._playback

    def stop_playback(self) -> None:
        if self._playback is not None:
            self._playback = PlaybackTimingState(
                active=False,
                anchor_frame=self._playback.anchor_frame,
                anchor_monotonic=self._playback.anchor_monotonic,
                source_fps=self._playback.source_fps,
                playback_speed=self._playback.playback_speed,
            )

    def playback_target_frame(self, now_monotonic: float | None = None) -> int | None:
        """Return the wall-clock target frame, or None when playback is inactive."""

        if self._playback is None or not self._playback.active:
            return None
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        return self._playback.target_frame(now)

    def sync_playback_frame(self, *, now_monotonic: float | None = None) -> int | None:
        """Advance to the wall-clock target frame, skipping as needed.

        Returns the displayed frame index, or ``None`` when playback reaches the
        end of the prepared-frame domain and stops.
        """

        session = self.require_session()
        target = self.playback_target_frame(now_monotonic)
        if target is None:
            return None
        if target >= session.frame_count:
            session.set_frame(session.max_frame_idx)
            self.stop_playback()
            return None
        if target != session.current_frame:
            session.set_frame(target)
        return session.current_frame

    def accept_tracking(self) -> AcceptanceResult:
        return self.require_session().accept_tracking()

    def current_view(self) -> ReviewFrameView:
        """Decode the current prepared frame and compose the pose overlay."""

        session = self.require_session()
        try:
            raw = self._read_frame(session.prepared_video_path, session.current_frame)
        except VideoProbeError as exc:
            raise TrackingCorrectionError(str(exc)) from exc
        points = session.pose_points_for_frame()
        image = draw_pose_overlay(raw, points, skeleton_edges=session.skeleton_edges)
        return ReviewFrameView(
            frame_idx=session.current_frame,
            frame_count=session.frame_count,
            pose_source=session.pose_source.value,
            acceptance_state=session.acceptance_state,
            correction_types=session.current_correction_types(),
            correction_episode=session.current_correction_episode(),
            pose_points=points,
            skeleton_edges=session.skeleton_edges,
            image_bgr=image,
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


def read_prepared_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read prepared-video frame ``frame_idx`` in S1 prepared-frame order."""

    return read_raw_frame_at_index(video_path, frame_idx)
