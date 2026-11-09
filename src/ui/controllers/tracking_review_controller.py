"""Headless coordination for Subsystem 03 tracking review and acceptance."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from preprocess.exceptions import VideoProbeError
from preprocess.video_probe import read_raw_frame_at_index
from tracking_correction.contracts import TrackingCorrectionError
from tracking_correction.manual_edits import ManualEditResult
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
    manual_edit_count: int = 0
    available_nodes: tuple[str, ...] = ()


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


@dataclass(frozen=True, slots=True)
class FrameRequestIdentity:
    """Identity used to reject stale composed-frame results."""

    request_id: str
    frame_generation: int
    session_generation: int
    run_id: str
    frame_idx: int
    pose_source: str
    edit_revision: int


@dataclass(frozen=True, slots=True)
class FrameRenderRequest:
    """Immutable decode + overlay payload for the decoder worker."""

    identity: FrameRequestIdentity
    video_path: Path
    pose_points: tuple[PoseNodePoint, ...]
    skeleton_edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FrameRenderSnapshot:
    """Worker result for one frame-render request."""

    identity: FrameRequestIdentity
    image_bgr: np.ndarray | None
    error: str | None = None


class TrackingReviewController:
    """Load an S3 run, navigate prepared frames, and accept tracking."""

    def __init__(
        self,
        *,
        frame_reader: FrameReader | None = None,
    ) -> None:
        self._frame_reader = frame_reader or read_prepared_frame
        # Sequential capture belongs on the decoder worker thread. Injected
        # readers (tests) skip it so fixtures stay deterministic.
        self._use_sequential_capture = frame_reader is None
        self.session: TrackingReviewSession | None = None
        self.project: TrackingCorrectionProjectSummary | None = None
        self._last_error: str | None = None
        self._playback: PlaybackTimingState | None = None
        self._session_generation = 0
        self._frame_generation = 0
        self._edit_revision = 0
        self._active_frame_identity: FrameRequestIdentity | None = None
        self._frame_pending = False
        self._displayed_frame_idx: int | None = None
        self._last_frame_error: str | None = None

    @property
    def frame_reader(self) -> FrameReader:
        return self._frame_reader

    @property
    def use_sequential_capture(self) -> bool:
        return self._use_sequential_capture

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_frame_error(self) -> str | None:
        return self._last_frame_error

    @property
    def playback_active(self) -> bool:
        return self._playback is not None and self._playback.active

    @property
    def session_generation(self) -> int:
        return self._session_generation

    @property
    def frame_generation(self) -> int:
        return self._frame_generation

    @property
    def edit_revision(self) -> int:
        return self._edit_revision

    @property
    def active_frame_identity(self) -> FrameRequestIdentity | None:
        return self._active_frame_identity

    @property
    def frame_pending(self) -> bool:
        return self._frame_pending

    @property
    def displayed_frame_idx(self) -> int | None:
        return self._displayed_frame_idx

    @property
    def display_matches_playhead(self) -> bool:
        """True when the last applied pixels match the scientific playhead."""

        session = self.session
        if session is None or self._frame_pending:
            return False
        return self._displayed_frame_idx == session.current_frame

    def abandon_pending_frame(self) -> None:
        """Drop in-flight display intent without changing scientific session state."""

        self._frame_pending = False
        self._active_frame_identity = None

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
        self._bump_session_generation()
        try:
            self.session = load_review_session(s3_run_dir)
        except TrackingCorrectionError as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None
        self._last_frame_error = None
        self._displayed_frame_idx = None
        # Prefer the project that actually contains this S3 run. Provenance
        # session_root may still point at a historical drive/NAS path.
        project_root: Path | None = None
        if self.session.run_dir.parent.name == "tracking_correction":
            candidate = self.session.run_dir.parent.parent
            if candidate.is_dir():
                project_root = candidate
        if project_root is None:
            project_root = self.session.session_root
        if project_root is not None:
            with suppress(OSError):
                self.project = summarize_tracking_correction_project(project_root)
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

    def jump_to_episode(self, episode: CorrectionEpisode) -> int:
        return self.require_session().jump_to_episode(episode)

    def next_correction_frame(self) -> int | None:
        return self.require_session().next_correction_frame()

    def previous_correction_frame(self) -> int | None:
        return self.require_session().previous_correction_frame()

    def next_correction_episode(self) -> CorrectionEpisode | None:
        return self.require_session().next_correction_episode()

    def previous_correction_episode(self) -> CorrectionEpisode | None:
        return self.require_session().previous_correction_episode()

    def set_pose_source(self, source: PoseDisplaySource | str) -> PoseDisplaySource:
        result = self.require_session().set_pose_source(source)
        self.invalidate_composed_frames()
        return result

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

    def swap_identities(self, start_frame: int, end_frame: int) -> ManualEditResult:
        result = self.require_session().swap_identities(start_frame, end_frame)
        self.invalidate_composed_frames()
        return result

    def swap_node(
        self,
        node: str,
        start_frame: int | None = None,
        end_frame: int | None = None,
        frame_idx: int | None = None,
    ) -> ManualEditResult:
        result = self.require_session().swap_node(
            node,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_idx=frame_idx,
        )
        self.invalidate_composed_frames()
        return result

    def blank_node(
        self,
        node: str,
        track: int,
        frame_idx: int | None = None,
    ) -> ManualEditResult:
        result = self.require_session().blank_node(node, track, frame_idx=frame_idx)
        self.invalidate_composed_frames()
        return result

    def undo_last_manual_edit(self) -> ManualEditResult:
        result = self.require_session().undo_last_manual_edit()
        self.invalidate_composed_frames()
        return result

    def reset_manual_edits(self) -> ManualEditResult:
        result = self.require_session().reset_manual_edits()
        self.invalidate_composed_frames()
        return result

    def invalidate_composed_frames(self) -> None:
        """Bump edit revision so in-flight composed overlays cannot apply."""

        self._edit_revision += 1
        self._frame_pending = False
        self._active_frame_identity = None
        # Pixels on screen are no longer authoritative for the current composition.
        self._displayed_frame_idx = None

    def begin_frame_request(self) -> FrameRenderRequest | None:
        """Snapshot pose points and mint a latest-wins frame-render identity."""

        session = self.session
        if session is None:
            return None
        self._frame_generation += 1
        points = session.pose_points_for_frame()
        identity = FrameRequestIdentity(
            request_id=uuid4().hex,
            frame_generation=self._frame_generation,
            session_generation=self._session_generation,
            run_id=session.run_id,
            frame_idx=session.current_frame,
            pose_source=session.pose_source.value,
            edit_revision=self._edit_revision,
        )
        self._active_frame_identity = identity
        self._frame_pending = True
        self._last_frame_error = None
        return FrameRenderRequest(
            identity=identity,
            video_path=Path(session.prepared_video_path),
            pose_points=points,
            skeleton_edges=session.skeleton_edges,
        )

    def apply_frame_snapshot(self, snapshot: FrameRenderSnapshot) -> bool:
        """Apply a worker snapshot only when it still matches current intent."""

        active = self._active_frame_identity
        session = self.session
        if (
            active is None
            or session is None
            or snapshot.identity.request_id != active.request_id
            or snapshot.identity.frame_generation != active.frame_generation
            or snapshot.identity.session_generation != self._session_generation
            or snapshot.identity.run_id != session.run_id
            or snapshot.identity.frame_idx != session.current_frame
            or snapshot.identity.pose_source != session.pose_source.value
            or snapshot.identity.edit_revision != self._edit_revision
        ):
            return False
        if snapshot.error is not None or snapshot.image_bgr is None:
            self._last_frame_error = snapshot.error or "Prepared frame decode failed."
            self._frame_pending = False
            if active.frame_generation == self._frame_generation:
                self._active_frame_identity = None
            return False
        self._displayed_frame_idx = snapshot.identity.frame_idx
        self._last_frame_error = None
        self._frame_pending = False
        if active.frame_generation == self._frame_generation:
            self._active_frame_identity = None
        return True

    def note_frame_failure(
        self,
        identity: FrameRequestIdentity,
        message: str,
    ) -> bool:
        """Record a decoder failure when the identity is still current."""

        active = self._active_frame_identity
        if (
            active is None
            or active.request_id != identity.request_id
            or active.frame_generation != identity.frame_generation
            or identity.session_generation != self._session_generation
            or identity.edit_revision != self._edit_revision
        ):
            return False
        self._last_frame_error = message
        self._frame_pending = False
        if active.frame_generation == self._frame_generation:
            self._active_frame_identity = None
        return True

    def current_view(self) -> ReviewFrameView:
        """Decode the current prepared frame synchronously (tests / headless).

        The GUI page uses :meth:`begin_frame_request` / async decode instead.
        This helper composes on the calling thread via the injected reader and
        never touches the worker-owned sequential capture.
        """

        session = self.require_session()
        try:
            raw = self._frame_reader(
                Path(session.prepared_video_path).resolve(),
                session.current_frame,
            )
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
            manual_edit_count=session.manual_edit_count,
            available_nodes=session.available_node_names,
        )

    def _bump_session_generation(self) -> None:
        self._session_generation += 1
        self._frame_generation += 1
        self._edit_revision += 1
        self._active_frame_identity = None
        self._frame_pending = False
        self._displayed_frame_idx = None
        self._last_frame_error = None


def read_prepared_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read prepared-video frame ``frame_idx`` in S1 prepared-frame order."""

    return read_raw_frame_at_index(video_path, frame_idx)
