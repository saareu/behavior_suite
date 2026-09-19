"""Sprint 3 review session: load an automatic S3 run, navigate, and accept.

This module owns review/acceptance orchestration. It does not call the
correction backend and does not mutate S1 or S2 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from preprocess.exceptions import SyncValidationError
from preprocess.sync_writer import load_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_NOT_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    AUTOMATIC_TRACKED_POSE_FILENAME,
    FINAL_TRACKED_POSE_FILENAME,
    MACHINE_CORRECTIONS_FILENAME,
    MANUAL_ACTION_BLANK_NODE,
    MANUAL_ACTION_SWAP_IDENTITIES,
    MANUAL_ACTION_SWAP_NODE,
    MANUAL_CORRECTIONS_FILENAME,
    POSE_SOURCE_CORRECTED,
    POSE_SOURCE_PROVISIONAL,
    PROCESSING_LOG_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_CORRECTION_COMPLETE,
    WORKING_TRACKED_POSE_FILENAME,
    TrackingCorrectionError,
    is_currently_accepted,
)
from tracking_correction.legacy_backend import load_current_profile_config
from tracking_correction.manual_edits import (
    ManualEditResult,
    apply_blank_node,
    apply_swap_identities,
    apply_swap_node,
    available_nodes,
    build_edit_record,
    ensure_automatic_baseline,
    frame_domain_signature,
    load_manual_corrections,
    read_working_pose,
    restore_automatic_baseline,
    undo_edit,
    write_manual_corrections,
    write_working_pose,
)
from tracking_correction.skeleton import (
    resolve_review_skeleton_edges,
    resolve_s2_pose_slp_path,
)

# Stable identity colors for the two-track MVP profile (BGR for OpenCV drawing).
TRACK_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (80, 180, 255),  # warm amber
    1: (255, 160, 60),  # cool blue
}

EPISODE_SOURCE_MACHINE = "machine"
EPISODE_SOURCE_MANUAL = "manual"

FPS_SOURCE_S1_TIMING = "S1 timing"
FPS_SOURCE_VIDEO_HEADER_FALLBACK = "video header fallback"

_DEFAULT_PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)


class PoseDisplaySource(StrEnum):
    """Which pose table the overlay reads."""

    CORRECTED = POSE_SOURCE_CORRECTED
    PROVISIONAL = POSE_SOURCE_PROVISIONAL


@dataclass(frozen=True, slots=True)
class PoseNodePoint:
    """One drawable node in prepared-video coordinates."""

    track: int
    node: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """Outcome of an Accept Tracking action."""

    already_accepted: bool
    tracked_pose_path: Path
    accepted_at: str
    working_tracked_pose_path: Path
    working_sha256: str
    manual_corrections_present: bool = False
    manual_correction_count: int = 0


@dataclass(frozen=True, slots=True)
class CorrectionEpisode:
    """Contiguous run of prepared frames that contain corrections.

    Machine episodes collapse contiguous automatic correction frames.
    Manual episodes represent one ``manual_corrections.json`` edit record.
    """

    start_frame: int
    end_frame: int
    correction_types: tuple[str, ...]
    source: str = EPISODE_SOURCE_MACHINE
    node: str | None = None
    track: int | None = None
    tracks: tuple[int, ...] = ()

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def is_single_frame(self) -> bool:
        return self.start_frame == self.end_frame


@dataclass(frozen=True, slots=True)
class TrackLegendEntry:
    """One track identity row for the review-workspace color legend."""

    track: int
    color_bgr: tuple[int, int, int]
    role_label: str | None = None

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        blue, green, red = self.color_bgr
        return (red, green, blue)

    @property
    def color_hex(self) -> str:
        red, green, blue = self.color_rgb
        return f"#{red:02x}{green:02x}{blue:02x}"


def track_legend_entries() -> tuple[TrackLegendEntry, ...]:
    """Return overlay track colors, with optional authoritative profile roles."""

    role_by_track = _profile_track_role_labels()
    tracks = sorted(TRACK_COLORS_BGR)
    return tuple(
        TrackLegendEntry(
            track=track,
            color_bgr=TRACK_COLORS_BGR[track],
            role_label=role_by_track.get(track),
        )
        for track in tracks
    )


def _profile_track_role_labels() -> dict[int, str]:
    """Map track indices to documented profile roles only (no inference)."""

    try:
        config = load_current_profile_config()
    except TrackingCorrectionError:
        return {}
    labels: dict[int, str] = {}
    implanted = config.get("implanted_track")
    if isinstance(implanted, bool) or not isinstance(implanted, int):
        return labels
    # Current MVP profile: track 0 owns the implanted/headstage animal.
    labels[int(implanted)] = "implanted/headstage"
    return labels


@dataclass(frozen=True, slots=True)
class ReviewPlaybackFps:
    """Constant review playback rate resolved from S1 timing when available."""

    fps: float
    source: str
    detail: str


@dataclass
class TrackingReviewSession:
    """In-memory review state for one completed automatic S3 run."""

    run_dir: Path
    run_id: str
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    prepared_video_path: Path
    s2_pose_path: Path
    working_tracked_pose_path: Path
    machine_corrections_path: Path
    tracked_pose_path: Path
    automatic_tracked_pose_path: Path
    manual_corrections_path: Path
    frame_count: int
    fps_header: float
    fps_source: str
    fps_source_detail: str
    s1_paths: tuple[Path, ...]
    s2_paths: tuple[Path, ...]
    correction_frames: tuple[int, ...]
    corrections_by_frame: dict[int, tuple[str, ...]]
    correction_episodes: tuple[CorrectionEpisode, ...]
    skeleton_edges: tuple[tuple[str, str], ...]
    skeleton_source: str
    session_root: Path | None
    s2_pose_slp_path: Path | None
    _corrected_by_frame: dict[int, tuple[PoseNodePoint, ...]]
    _provisional_by_frame: dict[int, tuple[PoseNodePoint, ...]]
    current_frame: int = 0
    pose_source: PoseDisplaySource = PoseDisplaySource.CORRECTED
    playback_speed: float = 1.0
    acceptance_state: str = ACCEPTANCE_NOT_ACCEPTED
    manual_edit_stack: list[dict[str, Any]] = field(default_factory=list)
    available_node_names: tuple[str, ...] = ()
    _playback_speeds: tuple[float, ...] = field(default=_DEFAULT_PLAYBACK_SPEEDS)
    _working_pose: pd.DataFrame | None = field(default=None, repr=False)
    _persist_generation: int = 0
    _persisted_generation: int = 0
    _persist_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _navigable_episodes: tuple[CorrectionEpisode, ...] = field(default_factory=tuple)

    @property
    def max_frame_idx(self) -> int:
        """Inclusive last prepared-frame index."""

        return max(0, self.frame_count - 1)

    @property
    def playback_speeds(self) -> tuple[float, ...]:
        return self._playback_speeds

    def set_frame(self, frame_idx: int) -> int:
        """Jump to an absolute prepared-frame index."""

        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
            raise TrackingCorrectionError("Frame index must be an integer.")
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise TrackingCorrectionError(
                f"Frame index {frame_idx} is outside prepared-frame domain "
                f"0..{self.frame_count - 1}."
            )
        self.current_frame = frame_idx
        return self.current_frame

    def step_forward(self) -> int:
        if self.current_frame < self.max_frame_idx:
            self.current_frame += 1
        return self.current_frame

    def step_backward(self) -> int:
        if self.current_frame > 0:
            self.current_frame -= 1
        return self.current_frame

    def set_pose_source(self, source: PoseDisplaySource | str) -> PoseDisplaySource:
        if isinstance(source, PoseDisplaySource):
            self.pose_source = source
        else:
            try:
                self.pose_source = PoseDisplaySource(str(source))
            except ValueError as exc:
                raise TrackingCorrectionError(
                    f"Unknown pose display source: {source!r}"
                ) from exc
        return self.pose_source

    def set_playback_speed(self, speed: float) -> float:
        if isinstance(speed, bool) or not isinstance(speed, int | float):
            raise TrackingCorrectionError("Playback speed must be numeric.")
        value = float(speed)
        if not math.isfinite(value) or value <= 0:
            raise TrackingCorrectionError("Playback speed must be a positive finite number.")
        self.playback_speed = value
        return self.playback_speed

    def pose_points_for_frame(
        self,
        frame_idx: int | None = None,
        *,
        source: PoseDisplaySource | None = None,
    ) -> tuple[PoseNodePoint, ...]:
        """Return drawable nodes for one prepared-frame index."""

        index = self.current_frame if frame_idx is None else int(frame_idx)
        if index < 0 or index >= self.frame_count:
            raise TrackingCorrectionError(
                f"Frame index {index} is outside prepared-frame domain "
                f"0..{self.frame_count - 1}."
            )
        selected = source or self.pose_source
        table = (
            self._corrected_by_frame
            if selected is PoseDisplaySource.CORRECTED
            else self._provisional_by_frame
        )
        return table.get(index, ())

    def current_correction_types(self) -> tuple[str, ...]:
        return self.corrections_by_frame.get(self.current_frame, ())

    def current_correction_episode(self) -> CorrectionEpisode | None:
        """Return the machine correction episode containing the current frame."""

        for episode in self.correction_episodes:
            if episode.start_frame <= self.current_frame <= episode.end_frame:
                return episode
        return None

    @property
    def navigable_episodes(self) -> tuple[CorrectionEpisode, ...]:
        """Manual edit episodes for the visible review list (not machine logs)."""

        return self._navigable_episodes

    def jump_to_episode(self, episode: CorrectionEpisode) -> int:
        """Jump to an edit/episode start frame (single-frame edits use that frame)."""

        return self.set_frame(episode.start_frame)

    def next_correction_episode(self) -> CorrectionEpisode | None:
        """Jump to the start of the next automatic machine episode."""

        for episode in self.correction_episodes:
            if episode.start_frame > self.current_frame:
                self.set_frame(episode.start_frame)
                return episode
        return None

    def previous_correction_episode(self) -> CorrectionEpisode | None:
        """Jump to the start of the previous automatic machine episode.

        When the playhead is inside an episode but past its start, this returns
        to that episode's start frame.
        """

        for episode in reversed(self.correction_episodes):
            if episode.start_frame < self.current_frame:
                self.set_frame(episode.start_frame)
                return episode
        return None

    def refresh_navigable_episodes(self) -> tuple[CorrectionEpisode, ...]:
        """Rebuild the visible manual-edit list from ``manual_corrections.json``."""

        self._navigable_episodes = _build_manual_edit_episodes(self.manual_edit_stack)
        return self._navigable_episodes

    # Legacy aliases used by older call sites; machine episode navigation is authoritative.
    def next_correction_frame(self) -> int | None:
        episode = self.next_correction_episode()
        return None if episode is None else episode.start_frame

    def previous_correction_frame(self) -> int | None:
        episode = self.previous_correction_episode()
        return None if episode is None else episode.start_frame

    @property
    def manual_edit_count(self) -> int:
        return len(self.manual_edit_stack)

    @property
    def has_manual_edits(self) -> bool:
        return self.manual_edit_count > 0

    @property
    def has_automatic_baseline(self) -> bool:
        return self.automatic_tracked_pose_path.is_file()

    @property
    def is_currently_accepted(self) -> bool:
        """True only while acceptance_state is accepted (not superseded)."""

        return is_currently_accepted(self.acceptance_state)

    def swap_identities(self, start_frame: int, end_frame: int) -> ManualEditResult:
        """Swap track 0 ↔ track 1 for all nodes over an inclusive interval."""

        def mutate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
            edited = apply_swap_identities(
                frame, start_frame=start_frame, end_frame=end_frame
            )
            record = build_edit_record(
                action=MANUAL_ACTION_SWAP_IDENTITIES,
                start_frame=start_frame,
                end_frame=end_frame,
                tracks=(0, 1),
            )
            return edited, record

        return self._commit_manual_edit(mutate)

    def swap_node(self, node: str, frame_idx: int | None = None) -> ManualEditResult:
        """Swap one node between tracks at the current (or given) frame."""

        index = self.current_frame if frame_idx is None else int(frame_idx)

        def mutate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
            edited = apply_swap_node(frame, frame_idx=index, node=node)
            record = build_edit_record(
                action=MANUAL_ACTION_SWAP_NODE,
                start_frame=index,
                end_frame=index,
                tracks=(0, 1),
                node=str(node).strip(),
            )
            return edited, record

        return self._commit_manual_edit(mutate)

    def blank_node(
        self,
        node: str,
        track: int,
        frame_idx: int | None = None,
    ) -> ManualEditResult:
        """Blank one incorrect node at the current (or given) frame."""

        index = self.current_frame if frame_idx is None else int(frame_idx)

        def mutate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
            edited, before = apply_blank_node(
                frame, frame_idx=index, track=track, node=node
            )
            after = {
                "frame_idx": index,
                "track": int(track),
                "node": str(node).strip(),
                "x": None,
                "y": None,
            }
            record = build_edit_record(
                action=MANUAL_ACTION_BLANK_NODE,
                start_frame=index,
                end_frame=index,
                tracks=(int(track),),
                node=str(node).strip(),
                track=int(track),
                before=before,
                after=after,
            )
            return edited, record

        return self._commit_manual_edit(mutate)

    def undo_last_manual_edit(self) -> ManualEditResult:
        """Undo the most recent manual edit."""

        if not self.manual_edit_stack:
            raise TrackingCorrectionError("No manual edits to undo.")
        protected = _snapshot_paths([*self.s1_paths, *self.s2_paths])
        record = self.manual_edit_stack[-1]
        working = self._require_working_pose()
        before_domain = frame_domain_signature(working)
        restored = undo_edit(working, record)
        after_domain = frame_domain_signature(restored)
        if after_domain != before_domain:
            raise TrackingCorrectionError(
                "Undo would change the prepared-frame domain; refusing."
            )
        self._working_pose = restored
        self.manual_edit_stack.pop()
        write_manual_corrections(self.manual_corrections_path, self.manual_edit_stack)
        self._apply_corrected_overlay_from_working(
            start_frame=int(record.get("start_frame", self.current_frame)),
            end_frame=int(record.get("end_frame", self.current_frame)),
        )
        self.refresh_navigable_episodes()
        self._queue_working_pose_persist()
        invalidated = self._invalidate_acceptance_if_needed()
        _assert_no_mutation(protected)
        return ManualEditResult(
            action="undo",
            start_frame=int(record.get("start_frame", self.current_frame)),
            end_frame=int(record.get("end_frame", self.current_frame)),
            edit_count=self.manual_edit_count,
            acceptance_invalidated=invalidated,
            record=record,
        )

    def reset_manual_edits(self) -> ManualEditResult:
        """Restore working pose from the automatic baseline and clear edits."""

        if not self.has_automatic_baseline and not self.manual_edit_stack:
            raise TrackingCorrectionError("No manual edits to reset.")
        protected = _snapshot_paths([*self.s1_paths, *self.s2_paths])
        working = self._require_working_pose()
        before_domain = frame_domain_signature(working)
        with self._persist_lock:
            # Invalidate in-flight persists before restoring the baseline on disk.
            self._persist_generation += 1
            restore_automatic_baseline(
                run_dir=self.run_dir,
                working_path=self.working_tracked_pose_path,
            )
            restored = read_working_pose(self.working_tracked_pose_path)
            self._working_pose = restored
            self._persisted_generation = self._persist_generation
        after_domain = frame_domain_signature(restored)
        if after_domain != before_domain:
            raise TrackingCorrectionError(
                "Reset would change the prepared-frame domain; refusing."
            )
        self.manual_edit_stack.clear()
        write_manual_corrections(self.manual_corrections_path, self.manual_edit_stack)
        self._corrected_by_frame = _index_pose_points_dataframe(
            restored,
            self.frame_count,
            label="working pose",
        )
        self.available_node_names = available_nodes(restored)
        self.refresh_navigable_episodes()
        invalidated = self._invalidate_acceptance_if_needed()
        _assert_no_mutation(protected)
        return ManualEditResult(
            action="reset",
            start_frame=0,
            end_frame=self.max_frame_idx,
            edit_count=0,
            acceptance_invalidated=invalidated,
        )

    def accept_tracking(self) -> AcceptanceResult:
        """Promote working_tracked_pose.parquet to tracked_pose.parquet."""

        self.flush_working_pose_persistence()
        return accept_tracking_result(self)

    def flush_working_pose_persistence(self) -> None:
        """Block until the latest in-memory working pose is on disk."""

        with self._persist_lock:
            if self._working_pose is None:
                return
            if self._persisted_generation != self._persist_generation:
                write_working_pose(self.working_tracked_pose_path, self._working_pose)
                self._persisted_generation = self._persist_generation

    def _commit_manual_edit(
        self,
        mutate: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, Any]]],
    ) -> ManualEditResult:
        protected = _snapshot_paths([*self.s1_paths, *self.s2_paths])
        ensure_automatic_baseline(self.run_dir, self.working_tracked_pose_path)
        working = self._require_working_pose()
        before_domain = frame_domain_signature(working)
        edited, record = mutate(working)
        after_domain = frame_domain_signature(edited)
        if after_domain != before_domain:
            raise TrackingCorrectionError(
                "Manual edit would change the prepared-frame domain; refusing."
            )
        self._working_pose = edited
        self.manual_edit_stack.append(dict(record))
        write_manual_corrections(self.manual_corrections_path, self.manual_edit_stack)
        self._apply_corrected_overlay_from_working(
            start_frame=int(record["start_frame"]),
            end_frame=int(record["end_frame"]),
        )
        self.available_node_names = available_nodes(edited)
        self.refresh_navigable_episodes()
        self._queue_working_pose_persist()
        invalidated = self._invalidate_acceptance_if_needed()
        _assert_no_mutation(protected)
        return ManualEditResult(
            action=str(record["action"]),
            start_frame=int(record["start_frame"]),
            end_frame=int(record["end_frame"]),
            edit_count=self.manual_edit_count,
            acceptance_invalidated=invalidated,
            record=record,
        )

    def _require_working_pose(self) -> pd.DataFrame:
        if self._working_pose is None:
            self._working_pose = read_working_pose(self.working_tracked_pose_path)
            with self._persist_lock:
                if self._persist_generation == 0:
                    self._persisted_generation = 0
        return self._working_pose

    def _apply_corrected_overlay_from_working(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> None:
        working = self._require_working_pose()
        patched = _index_pose_points_dataframe(
            working,
            self.frame_count,
            label="working pose",
            frame_min=start_frame,
            frame_max=end_frame,
        )
        for frame_idx in range(int(start_frame), int(end_frame) + 1):
            if frame_idx in patched:
                self._corrected_by_frame[frame_idx] = patched[frame_idx]
            else:
                self._corrected_by_frame.pop(frame_idx, None)

    def _queue_working_pose_persist(self) -> None:
        """Persist the working parquet off the caller thread with generation gating."""

        working = self._require_working_pose()
        with self._persist_lock:
            self._persist_generation += 1
            generation = self._persist_generation
        thread = threading.Thread(
            target=self._persist_working_pose_generation,
            args=(generation, working),
            daemon=True,
            name="s3-working-pose-persist",
        )
        thread.start()

    def _persist_working_pose_generation(
        self,
        generation: int,
        frame: pd.DataFrame,
    ) -> None:
        with self._persist_lock:
            if generation != self._persist_generation:
                return
            write_working_pose(self.working_tracked_pose_path, frame)
            self._persisted_generation = generation

    def _reload_corrected_overlay(self) -> None:
        working = self._require_working_pose()
        self._corrected_by_frame = _index_pose_points_dataframe(
            working,
            self.frame_count,
            label="working pose",
        )

    def _invalidate_acceptance_if_needed(self) -> bool:
        if not is_currently_accepted(self.acceptance_state):
            return False
        invalidated_at = _iso_now()
        meta = _load_json_object(self.run_meta_path, "S3 run_meta.json")
        previous = meta.get("acceptance")
        meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
        meta["acceptance_invalidated_at"] = invalidated_at
        if isinstance(previous, Mapping):
            meta["previous_acceptance"] = dict(previous)
        meta["acceptance"] = {
            "state": ACCEPTANCE_SUPERSEDED,
            "invalidated_at": invalidated_at,
            "reason": "working_pose_changed_after_acceptance",
        }
        # Keep any stale tracked_pose.parquet on disk, but metadata is authoritative:
        # superseded runs are not currently accepted merely because that file exists.
        _write_json(self.run_meta_path, meta)

        if self.settings_used_path.is_file():
            settings = _load_yaml_object(self.settings_used_path, "S3 settings_used.yaml")
            settings["acceptance_state"] = ACCEPTANCE_SUPERSEDED
            settings["acceptance_invalidated_at"] = invalidated_at
            if isinstance(previous, Mapping):
                settings["previous_acceptance"] = dict(previous)
            settings["acceptance"] = dict(meta["acceptance"])
            _write_yaml(self.settings_used_path, settings)

        _append_text_log(
            self.processing_log_path,
            [
                "",
                f"acceptance_state: {ACCEPTANCE_SUPERSEDED}",
                f"acceptance_invalidated_at: {invalidated_at}",
                "reason: working_pose_changed_after_acceptance",
            ],
        )
        self.acceptance_state = ACCEPTANCE_SUPERSEDED
        return True


def load_review_session(s3_run_dir: Path) -> TrackingReviewSession:
    """Load a completed automatic S3 run for video-centered review."""

    run_dir = Path(s3_run_dir).expanduser()
    if not run_dir.exists():
        raise TrackingCorrectionError(f"S3 run does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise TrackingCorrectionError(f"S3 run must be a directory: {run_dir}")
    run_dir = run_dir.resolve()

    run_meta_path = run_dir / RUN_META_FILENAME
    settings_path = run_dir / SETTINGS_USED_FILENAME
    log_path = run_dir / PROCESSING_LOG_FILENAME
    working_path = run_dir / WORKING_TRACKED_POSE_FILENAME
    corrections_path = run_dir / MACHINE_CORRECTIONS_FILENAME
    tracked_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    automatic_path = run_dir / AUTOMATIC_TRACKED_POSE_FILENAME
    manual_path = run_dir / MANUAL_CORRECTIONS_FILENAME

    meta = _load_json_object(run_meta_path, "S3 run_meta.json")
    _reject_incomplete_run(meta, working_path)

    input_block = meta.get("input")
    if not isinstance(input_block, Mapping):
        raise TrackingCorrectionError("S3 run_meta.json is missing input provenance.")

    prepared_video = _require_existing_file(
        _resolve_meta_path(input_block.get("prepared_video"), run_dir),
        "Prepared video",
    )
    s2_pose = _require_existing_file(
        _resolve_meta_path(input_block.get("pose_parquet"), run_dir),
        "S2 pose.parquet",
    )
    working_path = _require_existing_file(working_path, "Working tracked pose")
    corrections_path = _require_existing_file(corrections_path, "Machine corrections")

    timing = meta.get("timing")
    if not isinstance(timing, Mapping):
        raise TrackingCorrectionError("S3 run_meta.json is missing timing.")
    frame_count = _positive_int(timing.get("frame_count_used_for_sleap"), "frame_count")
    playback_fps = _resolve_review_playback_fps(
        input_block=input_block,
        run_dir=run_dir,
        timing_block=timing,
        prepared_video=prepared_video,
    )

    corrected_by_frame = _index_pose_points(working_path, frame_count, label="working pose")
    provisional_by_frame = _index_pose_points(s2_pose, frame_count, label="S2 pose")
    correction_frames, corrections_by_frame = _index_correction_frames(corrections_path)
    correction_episodes = _build_correction_episodes(correction_frames, corrections_by_frame)
    manual_stack = load_manual_corrections(manual_path)
    working_pose = read_working_pose(working_path)
    node_names = available_nodes(working_pose)

    s1_paths = _existing_paths(
        (
            input_block.get("prepared_video"),
            input_block.get("prepare_meta"),
            input_block.get("prepared_sync"),
            input_block.get("preprocess_dir"),
        ),
        run_dir,
    )
    s2_paths = _existing_paths(
        (
            input_block.get("s2_run_dir"),
            input_block.get("pose_parquet"),
        ),
        run_dir,
    )

    s2_run_dir = _optional_existing_dir(input_block.get("s2_run_dir"), run_dir)
    session_root = _optional_existing_dir(input_block.get("session_root"), run_dir)
    if session_root is None and s2_run_dir is not None:
        candidate = s2_run_dir.parent.parent
        if candidate.is_dir():
            session_root = candidate
    pose_slp = resolve_s2_pose_slp_path(s2_run_dir=s2_run_dir, pose_parquet=s2_pose)
    if pose_slp is not None and pose_slp not in s2_paths:
        s2_paths.append(pose_slp)
    skeleton_edges = resolve_review_skeleton_edges(pose_slp)
    if pose_slp is not None and skeleton_edges:
        skeleton_source = f"s2_pose_slp:{pose_slp}"
    elif pose_slp is not None:
        skeleton_source = f"unresolved:{pose_slp}"
    else:
        skeleton_source = "unavailable"

    acceptance_state = str(meta.get("acceptance_state") or ACCEPTANCE_NOT_ACCEPTED)
    run_id = str(meta.get("run_id") or run_dir.name)

    session = TrackingReviewSession(
        run_dir=run_dir,
        run_id=run_id,
        run_meta_path=run_meta_path,
        settings_used_path=settings_path if settings_path.is_file() else run_dir / SETTINGS_USED_FILENAME,
        processing_log_path=log_path if log_path.is_file() else run_dir / PROCESSING_LOG_FILENAME,
        prepared_video_path=prepared_video,
        s2_pose_path=s2_pose,
        working_tracked_pose_path=working_path,
        machine_corrections_path=corrections_path,
        tracked_pose_path=tracked_path,
        automatic_tracked_pose_path=automatic_path,
        manual_corrections_path=manual_path,
        frame_count=frame_count,
        fps_header=playback_fps.fps,
        fps_source=playback_fps.source,
        fps_source_detail=playback_fps.detail,
        s1_paths=tuple(s1_paths),
        s2_paths=tuple(s2_paths),
        correction_frames=correction_frames,
        corrections_by_frame=corrections_by_frame,
        correction_episodes=correction_episodes,
        skeleton_edges=skeleton_edges,
        skeleton_source=skeleton_source,
        session_root=session_root,
        s2_pose_slp_path=pose_slp,
        _corrected_by_frame=corrected_by_frame,
        _provisional_by_frame=provisional_by_frame,
        acceptance_state=acceptance_state,
        manual_edit_stack=manual_stack,
        available_node_names=node_names,
        _working_pose=working_pose,
    )
    session.refresh_navigable_episodes()
    return session


def accept_tracking_result(session: TrackingReviewSession) -> AcceptanceResult:
    """Copy the working pose to tracked_pose.parquet and record acceptance."""

    session.flush_working_pose_persistence()
    protected = _snapshot_paths([*session.s1_paths, *session.s2_paths])
    meta = _load_json_object(session.run_meta_path, "S3 run_meta.json")
    _reject_incomplete_run(meta, session.working_tracked_pose_path)

    working_sha = _sha256(session.working_tracked_pose_path)
    accepted_at = _iso_now()
    manual_present = session.has_manual_edits
    manual_count = session.manual_edit_count

    if is_currently_accepted(meta.get("acceptance_state")):
        if not session.tracked_pose_path.is_file():
            raise TrackingCorrectionError(
                "S3 run is marked accepted but tracked_pose.parquet is missing."
            )
        existing_sha = _sha256(session.tracked_pose_path)
        recorded = _optional_text(
            _mapping(meta.get("acceptance")).get("working_tracked_pose_sha256")
            if isinstance(meta.get("acceptance"), Mapping)
            else None
        )
        if existing_sha != working_sha or (recorded is not None and recorded != working_sha):
            raise TrackingCorrectionError(
                "S3 run is already accepted with a different tracked result; "
                "refusing to overwrite."
            )
        session.acceptance_state = ACCEPTANCE_ACCEPTED
        _assert_no_mutation(protected)
        return AcceptanceResult(
            already_accepted=True,
            tracked_pose_path=session.tracked_pose_path,
            accepted_at=str(
                _mapping(meta.get("acceptance")).get("accepted_at") or accepted_at
            ),
            working_tracked_pose_path=session.working_tracked_pose_path,
            working_sha256=working_sha,
            manual_corrections_present=bool(
                _mapping(meta.get("acceptance")).get("manual_corrections_present")
            ),
            manual_correction_count=int(
                _mapping(meta.get("acceptance")).get("manual_correction_count") or 0
            ),
        )

    try:
        shutil.copy2(session.working_tracked_pose_path, session.tracked_pose_path)
    except OSError as exc:
        raise TrackingCorrectionError(
            f"Could not write tracked_pose.parquet: {exc}"
        ) from exc

    if _sha256(session.tracked_pose_path) != working_sha:
        with suppress(OSError):
            session.tracked_pose_path.unlink()
        raise TrackingCorrectionError(
            "tracked_pose.parquet does not match the accepted working pose."
        )

    acceptance_block = {
        "accepted_at": accepted_at,
        "accepted_working_tracked_pose": _path_text(session.working_tracked_pose_path),
        "working_tracked_pose_sha256": working_sha,
        "tracked_pose_parquet": _path_text(session.tracked_pose_path),
        "tracked_pose_sha256": working_sha,
        "manual_corrections_present": manual_present,
        "manual_correction_count": manual_count,
    }
    meta["acceptance_state"] = ACCEPTANCE_ACCEPTED
    meta["acceptance"] = acceptance_block
    meta.pop("acceptance_invalidated_at", None)
    outputs = meta.get("outputs")
    if isinstance(outputs, dict):
        outputs["tracked_pose_parquet"] = _path_text(session.tracked_pose_path)
    _write_json(session.run_meta_path, meta)

    if session.settings_used_path.is_file():
        settings = _load_yaml_object(session.settings_used_path, "S3 settings_used.yaml")
        settings["acceptance_state"] = ACCEPTANCE_ACCEPTED
        settings["acceptance"] = dict(acceptance_block)
        settings.pop("acceptance_invalidated_at", None)
        _write_yaml(session.settings_used_path, settings)

    _append_acceptance_log(
        session.processing_log_path,
        accepted_at=accepted_at,
        working_path=session.working_tracked_pose_path,
        tracked_path=session.tracked_pose_path,
        working_sha=working_sha,
        manual_corrections_present=manual_present,
        manual_correction_count=manual_count,
    )
    session.acceptance_state = ACCEPTANCE_ACCEPTED
    _assert_no_mutation(protected)
    return AcceptanceResult(
        already_accepted=False,
        tracked_pose_path=session.tracked_pose_path,
        accepted_at=accepted_at,
        working_tracked_pose_path=session.working_tracked_pose_path,
        working_sha256=working_sha,
        manual_corrections_present=manual_present,
        manual_correction_count=manual_count,
    )


def _resolve_review_playback_fps(
    *,
    input_block: Mapping[str, Any],
    run_dir: Path,
    timing_block: Mapping[str, Any],
    prepared_video: Path,
) -> ReviewPlaybackFps:
    """Resolve constant review FPS from S1 artifacts, else video-header fallback.

    Prepared-frame domain identity is unchanged; only the wall-clock rate used for
    playback scheduling is selected here.
    """

    sync_fps = _fps_from_prepared_sync(input_block, run_dir)
    if sync_fps is not None:
        return ReviewPlaybackFps(
            fps=sync_fps,
            source=FPS_SOURCE_S1_TIMING,
            detail="prepared_sync.npz",
        )

    meta_fps = _fps_from_prepare_meta(input_block, run_dir)
    if meta_fps is not None:
        return ReviewPlaybackFps(
            fps=meta_fps,
            source=FPS_SOURCE_S1_TIMING,
            detail="prepare_meta.json",
        )

    video_fps = _fps_from_video_header(prepared_video)
    if video_fps is not None:
        return ReviewPlaybackFps(
            fps=video_fps,
            source=FPS_SOURCE_VIDEO_HEADER_FALLBACK,
            detail="opencv_fps",
        )

    recorded = _optional_positive_fps(timing_block.get("fps_header"))
    if recorded is not None:
        return ReviewPlaybackFps(
            fps=recorded,
            source=FPS_SOURCE_VIDEO_HEADER_FALLBACK,
            detail="run_meta.timing.fps_header",
        )
    raise TrackingCorrectionError(
        "Review playback FPS cannot be resolved from S1 timing or video header."
    )


def _fps_from_prepared_sync(input_block: Mapping[str, Any], run_dir: Path) -> float | None:
    path = _optional_existing_file(input_block.get("prepared_sync"), run_dir)
    if path is None:
        return None
    try:
        sync = load_prepared_sync_npz(path)
    except (OSError, SyncValidationError, ValueError, TypeError):
        return None
    return _optional_positive_fps(sync.fps_header)


def _fps_from_prepare_meta(input_block: Mapping[str, Any], run_dir: Path) -> float | None:
    path = _optional_existing_file(input_block.get("prepare_meta"), run_dir)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    prepared = payload.get("prepared_video")
    if not isinstance(prepared, Mapping):
        return None
    return _optional_positive_fps(prepared.get("fps_header"))


def _fps_from_video_header(prepared_video: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(prepared_video))
    try:
        if not capture.isOpened():
            return None
        reported = float(capture.get(cv2.CAP_PROP_FPS))
    except (cv2.error, TypeError, ValueError):
        return None
    finally:
        capture.release()
    return _optional_positive_fps(reported)


def _optional_positive_fps(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _optional_existing_file(value: object, base: Path) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    return resolved if resolved.is_file() else None


def _reject_incomplete_run(meta: Mapping[str, Any], working_path: Path) -> None:
    status = _optional_text(meta.get("status"))
    backend_status = _optional_text(meta.get("backend_status"))
    dry_run = meta.get("dry_run") is True
    if dry_run:
        raise TrackingCorrectionError("Cannot review a dry-run S3 workspace.")
    if status != STATUS_CORRECTION_COMPLETE:
        raise TrackingCorrectionError(
            f"S3 run is not ready for review/acceptance: status={status or 'unknown'}."
        )
    if backend_status != "completed":
        raise TrackingCorrectionError(
            "S3 run is not ready for review/acceptance: "
            f"backend_status={backend_status or 'unknown'}."
        )
    if not working_path.is_file():
        raise TrackingCorrectionError(
            f"Working tracked pose is missing: {working_path}"
        )


def _index_pose_points(
    path: Path,
    frame_count: int,
    *,
    label: str,
) -> dict[int, tuple[PoseNodePoint, ...]]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise TrackingCorrectionError(f"Could not read {label}: {exc}") from exc
    return _index_pose_points_dataframe(frame, frame_count, label=label)


def _index_pose_points_dataframe(
    frame: pd.DataFrame,
    frame_count: int,
    *,
    label: str,
    frame_min: int | None = None,
    frame_max: int | None = None,
) -> dict[int, tuple[PoseNodePoint, ...]]:
    frame_column = "frame_idx" if "frame_idx" in frame.columns else "frame"
    if frame_column not in frame.columns:
        raise TrackingCorrectionError(f"{label} is missing frame_idx/frame.")
    for required in ("track", "node", "x", "y"):
        if required not in frame.columns:
            raise TrackingCorrectionError(f"{label} is missing column {required}.")

    view = frame
    if frame_min is not None or frame_max is not None:
        frame_values_all = pd.to_numeric(frame[frame_column], errors="coerce")
        mask = pd.Series(True, index=frame.index)
        if frame_min is not None:
            mask &= frame_values_all >= int(frame_min)
        if frame_max is not None:
            mask &= frame_values_all <= int(frame_max)
        view = frame.loc[mask]

    frame_values = pd.to_numeric(view[frame_column], errors="coerce")
    x_values = pd.to_numeric(view["x"], errors="coerce")
    y_values = pd.to_numeric(view["y"], errors="coerce")
    track_values = view["track"].tolist()
    node_values = view["node"].tolist()

    by_frame: dict[int, list[PoseNodePoint]] = {}
    for frame_value, track_value, node_value, x_value, y_value in zip(
        frame_values.tolist(),
        track_values,
        node_values,
        x_values.tolist(),
        y_values.tolist(),
        strict=True,
    ):
        if pd.isna(frame_value) or pd.isna(x_value) or pd.isna(y_value):
            continue
        track = _parse_track_index(track_value)
        if track is None:
            continue
        index = int(frame_value)
        if index < 0 or index >= frame_count:
            raise TrackingCorrectionError(
                f"{label} frame_idx {index} is outside prepared-frame domain "
                f"0..{frame_count - 1}."
            )
        node = str(node_value).strip()
        if not node or node.lower() == "nan":
            continue
        by_frame.setdefault(index, []).append(
            PoseNodePoint(track=track, node=node, x=float(x_value), y=float(y_value))
        )
    return {key: tuple(value) for key, value in by_frame.items()}


def _parse_track_index(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    lowered = text.lower()
    if lowered.startswith("track_"):
        suffix = text.split("_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _index_correction_frames(
    path: Path,
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingCorrectionError(f"Machine corrections are unreadable: {exc}") from exc
    if not isinstance(payload, list):
        raise TrackingCorrectionError("machine_corrections.json must contain a list.")

    types_by_frame: dict[int, set[str]] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        frame_value = item.get("frame")
        if isinstance(frame_value, bool) or not isinstance(frame_value, int | float):
            continue
        if isinstance(frame_value, float) and not frame_value.is_integer():
            continue
        frame_idx = int(frame_value)
        correction_type = str(item.get("type") or "unknown").strip() or "unknown"
        types_by_frame.setdefault(frame_idx, set()).add(correction_type)

    frames = tuple(sorted(types_by_frame))
    by_frame = {
        frame_idx: tuple(sorted(types_by_frame[frame_idx])) for frame_idx in frames
    }
    return frames, by_frame


def _build_correction_episodes(
    frames: Sequence[int],
    corrections_by_frame: Mapping[int, Sequence[str]],
) -> tuple[CorrectionEpisode, ...]:
    """Collapse contiguous corrected frames into lightweight navigation episodes."""

    if not frames:
        return ()
    episodes: list[CorrectionEpisode] = []
    start = int(frames[0])
    previous = start
    types: set[str] = set(corrections_by_frame.get(start, ()))
    for frame_idx in frames[1:]:
        index = int(frame_idx)
        if index == previous + 1:
            types.update(corrections_by_frame.get(index, ()))
            previous = index
            continue
        episodes.append(
            CorrectionEpisode(
                start_frame=start,
                end_frame=previous,
                correction_types=tuple(sorted(types)),
                source=EPISODE_SOURCE_MACHINE,
            )
        )
        start = index
        previous = index
        types = set(corrections_by_frame.get(index, ()))
    episodes.append(
        CorrectionEpisode(
            start_frame=start,
            end_frame=previous,
            correction_types=tuple(sorted(types)),
            source=EPISODE_SOURCE_MACHINE,
        )
    )
    return tuple(episodes)


def _build_manual_edit_episodes(
    records: Sequence[Mapping[str, Any]],
) -> tuple[CorrectionEpisode, ...]:
    """One list entry per ``manual_corrections.json`` edit record."""

    episodes: list[CorrectionEpisode] = []
    for record in records:
        try:
            start = int(record["start_frame"])
            end = int(record["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        action = str(record.get("action") or "manual_edit").strip() or "manual_edit"
        node_value = record.get("node")
        node = (
            str(node_value).strip()
            if node_value is not None and str(node_value).strip()
            else None
        )
        track_value = record.get("track")
        track: int | None
        try:
            track = int(track_value) if track_value is not None else None
        except (TypeError, ValueError):
            track = None
        tracks_raw = record.get("tracks")
        tracks: list[int] = []
        if isinstance(tracks_raw, Sequence) and not isinstance(
            tracks_raw, (str, bytes)
        ):
            for item in tracks_raw:
                try:
                    tracks.append(int(item))
                except (TypeError, ValueError):
                    continue
        episodes.append(
            CorrectionEpisode(
                start_frame=start,
                end_frame=end,
                correction_types=(action,),
                source=EPISODE_SOURCE_MANUAL,
                node=node,
                track=track,
                tracks=tuple(tracks),
            )
        )
    return tuple(episodes)


def _optional_existing_dir(value: object, base: Path) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    return resolved if resolved.is_dir() else None


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise TrackingCorrectionError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingCorrectionError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingCorrectionError(f"{label} must contain an object.")
    return payload


def _load_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrackingCorrectionError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingCorrectionError(f"{label} must contain an object.")
    return payload


def _resolve_meta_path(value: object, base: Path) -> Path:
    text = _optional_text(value)
    if text is None:
        raise TrackingCorrectionError("S3 run_meta.json is missing a required path.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_existing_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise TrackingCorrectionError(f"{label} does not exist: {path}")
    return path


def _existing_paths(values: Sequence[object], base: Path) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = _optional_text(value)
        if text is None:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = base / path
        resolved = path.resolve()
        if resolved.exists():
            paths.append(resolved)
    return paths


def _snapshot_paths(paths: Sequence[Path]) -> dict[Path, tuple[str, int] | None]:
    snapshot: dict[Path, tuple[str, int] | None] = {}
    for path in paths:
        if path.is_file():
            snapshot[path] = (_sha256(path), path.stat().st_size)
        elif path.exists():
            snapshot[path] = None
        else:
            snapshot[path] = None
    return snapshot


def _assert_no_mutation(before: Mapping[Path, tuple[str, int] | None]) -> None:
    for path, prior in before.items():
        if prior is None:
            if path.is_file():
                raise TrackingCorrectionError(
                    f"Acceptance mutated protected artifact unexpectedly: {path}"
                )
            continue
        if not path.is_file():
            raise TrackingCorrectionError(f"Protected artifact disappeared: {path}")
        current = (_sha256(path), path.stat().st_size)
        if current != prior:
            raise TrackingCorrectionError(f"Acceptance mutated protected artifact: {path}")


def _append_acceptance_log(
    path: Path,
    *,
    accepted_at: str,
    working_path: Path,
    tracked_path: Path,
    working_sha: str,
    manual_corrections_present: bool = False,
    manual_correction_count: int = 0,
) -> None:
    lines = [
        "",
        f"acceptance_state: {ACCEPTANCE_ACCEPTED}",
        f"accepted_at: {accepted_at}",
        f"accepted_working_tracked_pose: {_path_text(working_path)}",
        f"working_tracked_pose_sha256: {working_sha}",
        f"tracked_pose_parquet: {_path_text(tracked_path)}",
        f"manual_corrections_present: {str(manual_corrections_present).lower()}",
        f"manual_correction_count: {manual_correction_count}",
    ]
    _append_text_log(path, lines)


def _append_text_log(path: Path, lines: Sequence[str]) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingCorrectionError(f"S3 timing {label} is invalid.")
    if value <= 0:
        raise TrackingCorrectionError(f"S3 timing {label} must be positive.")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrackingCorrectionError(f"S3 timing {label} is invalid.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise TrackingCorrectionError(f"S3 timing {label} must be positive.")
    return number


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
