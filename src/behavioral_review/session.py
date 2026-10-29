"""Read-only in-memory inspection session for synchronized behavioral review.

Seeking, filtering, and evidence inspection never modify behavioral results or
create review decisions. Sprint 1–2 do not create draft journals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.bouts import BoutBrowser, BoutFilter, BoutOverlay
from behavioral_review.contracts import (
    CHANNEL_VALUES,
    DEFAULT_ETHOGRAM_HALF_WINDOW,
    DEFAULT_EVIDENCE_HALF_WINDOW,
    DEFAULT_TIMELINE_MAX_SEGMENTS,
    IDENTITY_DIRECTIONAL,
    IDENTITY_SYMMETRIC_PAIR,
    INSPECTION_SESSION_SCHEMA_VERSION,
    MODE_INSPECTION,
    InspectionError,
    SessionError,
    VideoAccessError,
)
from behavioral_review.ethogram import EthogramAccessor, EthogramWindow
from behavioral_review.evidence import (
    ArbitrationProvenance,
    EvidenceAccessor,
    S7EvidenceWindow,
    S8EchoEvidenceWindow,
)
from behavioral_review.source import ReviewSourcePackage, open_review_source
from behavioral_review.sync import BoutSeekMapping, PlayheadMapping, SyncMap
from behavioral_review.timeline import TimelineView, build_timeline_view
from behavioral_review.video import DecodedFrame, PreparedFrameAccessor


@dataclass(frozen=True, slots=True)
class DirectionSelection:
    """Active exclusive direction (independent; no reciprocal tool in MVP)."""

    focal_id: int | None
    partner_id: int | None

    @property
    def is_set(self) -> bool:
        return self.focal_id is not None and self.partner_id is not None

    def as_pair(self) -> tuple[int, int] | None:
        if not self.is_set:
            return None
        return int(self.focal_id), int(self.partner_id)  # type: ignore[arg-type]


@dataclass
class InspectionSession:
    """Mutable inspection playhead + selections over an immutable source.

    One authoritative dense playhead drives ethogram, bout, evidence, and video
    surfaces. Seeking/filtering never writes science artifacts or review decisions.
    """

    schema_version: str
    mode: str
    source: ReviewSourcePackage
    sync: SyncMap
    playhead: PlayheadMapping
    ethogram: EthogramAccessor
    bouts: BoutBrowser
    evidence: EvidenceAccessor
    selected_channel: str | None = None
    selected_direction: DirectionSelection = field(
        default_factory=lambda: DirectionSelection(None, None)
    )
    selected_bout_id: str | None = None
    video: PreparedFrameAccessor | None = None
    last_decoded: DecodedFrame | None = None
    warnings: list[str] = field(default_factory=list)
    _previous_prepared_frame_idx: int | None = field(default=None, repr=False)

    @property
    def package_run_id(self) -> str:
        return self.source.package_run_id

    @property
    def currently_eligible(self) -> bool:
        return self.source.currently_eligible

    @property
    def n_dense_samples(self) -> int:
        return self.sync.n_dense

    @property
    def n_bouts(self) -> int:
        return self.source.n_bouts

    def seek_sample(self, sample_idx: int, *, decode: bool = False) -> PlayheadMapping:
        mapping = self.sync.mapping_at_sample(sample_idx)
        self._apply_playhead(mapping, decode=decode)
        return mapping

    def seek_dense_frame(
        self, dense_frame_idx: int, *, decode: bool = False
    ) -> PlayheadMapping:
        return self.seek_sample(dense_frame_idx, decode=decode)

    def seek_time_sec(self, time_sec: float, *, decode: bool = False) -> PlayheadMapping:
        mapping = self.sync.mapping_at_time_sec(time_sec)
        self._apply_playhead(mapping, decode=decode)
        return mapping

    def step(self, delta: int = 1, *, decode: bool = False) -> PlayheadMapping:
        if isinstance(delta, bool) or not isinstance(delta, (int, np.integer)):
            raise SessionError(f"step delta must be an integer; got {delta!r}.")
        nxt = int(self.playhead.sample_idx) + int(delta)
        nxt = max(0, min(self.sync.n_dense - 1, nxt))
        return self.seek_sample(nxt, decode=decode)

    def seek_bout_start(
        self,
        bout_id: str | None = None,
        *,
        decode: bool = False,
    ) -> BoutSeekMapping:
        try:
            mapped = self.bouts.jump_to_start(bout_id)
            bout = self.bouts.get(mapped.bout_id or self.bouts.selected_bout_id or "")
        except InspectionError as exc:
            raise SessionError(str(exc)) from exc
        self.selected_bout_id = str(bout["bout_id"])
        self._apply_channel_and_identity_from_bout(bout)
        self._apply_playhead(mapped.start, decode=decode)
        return mapped

    def seek_bout_end(
        self,
        bout_id: str | None = None,
        *,
        decode: bool = False,
    ) -> BoutSeekMapping:
        try:
            mapped = self.bouts.jump_to_end(bout_id)
            bout = self.bouts.get(mapped.bout_id or self.bouts.selected_bout_id or "")
        except InspectionError as exc:
            raise SessionError(str(exc)) from exc
        self.selected_bout_id = str(bout["bout_id"])
        self._apply_channel_and_identity_from_bout(bout)
        self._apply_playhead(mapped.end, decode=decode)
        return mapped

    def select_channel(self, channel: str | None) -> None:
        if channel is None:
            self.selected_channel = None
            return
        if channel not in CHANNEL_VALUES:
            raise SessionError(
                f"Unknown channel {channel!r}; expected one of {sorted(CHANNEL_VALUES)}."
            )
        self.selected_channel = channel

    def select_direction(
        self,
        *,
        focal_id: int | None,
        partner_id: int | None,
    ) -> None:
        if (focal_id is None) ^ (partner_id is None):
            raise SessionError(
                "focal_id and partner_id must both be set or both be None."
            )
        self.selected_direction = DirectionSelection(focal_id, partner_id)

    def select_bout(self, bout_id: str | None) -> pd.Series | None:
        if bout_id is None:
            self.selected_bout_id = None
            self.bouts.select(None)
            return None
        try:
            bout = self.bouts.select(bout_id)
        except InspectionError as exc:
            raise SessionError(str(exc)) from exc
        assert bout is not None
        self.selected_bout_id = str(bout["bout_id"])
        self._apply_channel_and_identity_from_bout(bout)
        return bout

    def set_bout_filter(self, bout_filter: BoutFilter | None) -> None:
        """Display/navigation filter only — never rewrites packaged bouts."""

        self.bouts.set_filter(bout_filter)

    def previous_bout(self, *, decode: bool = False) -> BoutSeekMapping | None:
        nxt = self.bouts.previous_bout(self.selected_bout_id)
        if nxt is None:
            return None
        return self.seek_bout_start(nxt, decode=decode)

    def next_bout(self, *, decode: bool = False) -> BoutSeekMapping | None:
        nxt = self.bouts.next_bout(self.selected_bout_id)
        if nxt is None:
            return None
        return self.seek_bout_start(nxt, decode=decode)

    def ethogram_window(
        self,
        *,
        half_window: int = DEFAULT_ETHOGRAM_HALF_WINDOW,
        start_sample_idx: int | None = None,
        end_sample_idx: int | None = None,
        include_echo_probs: bool = True,
        include_validity: bool = True,
    ) -> EthogramWindow:
        start, end = self._window_bounds(
            half_window=half_window,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
        )
        directions = None
        pair = self.selected_direction.as_pair()
        if pair is not None:
            # Always include both reciprocal directions when one is selected.
            directions = (pair, (pair[1], pair[0]))
        return self.ethogram.read_window(
            start,
            end,
            directions=directions,
            include_echo_probs=include_echo_probs,
            include_validity=include_validity,
        )

    def overlapping_bouts(
        self,
        *,
        half_window: int = DEFAULT_ETHOGRAM_HALF_WINDOW,
        start_sample_idx: int | None = None,
        end_sample_idx: int | None = None,
    ) -> pd.DataFrame:
        start, end = self._window_bounds(
            half_window=half_window,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
        )
        return self.bouts.overlapping(start, end)

    def selected_bout_overlay(self) -> BoutOverlay | None:
        if self.selected_bout_id is None:
            return None
        return self.bouts.overlay(self.selected_bout_id, selected=True)

    def s8_echoed_evidence(
        self,
        *,
        half_window: int = DEFAULT_EVIDENCE_HALF_WINDOW,
        start_sample_idx: int | None = None,
        end_sample_idx: int | None = None,
        detectors: list[str] | None = None,
    ) -> S8EchoEvidenceWindow:
        start, end = self._window_bounds(
            half_window=half_window,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
        )
        directions = None
        pair = self.selected_direction.as_pair()
        if pair is not None:
            directions = (pair, (pair[1], pair[0]))
        return self.evidence.s8_echoed_window(
            start, end, directions=directions, detectors=detectors
        )

    def s7_evidence(
        self,
        *,
        half_window: int = DEFAULT_EVIDENCE_HALF_WINDOW,
        start_sample_idx: int | None = None,
        end_sample_idx: int | None = None,
        detectors: list[str] | None = None,
    ) -> S7EvidenceWindow:
        start, end = self._window_bounds(
            half_window=half_window,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
        )
        pair = self.selected_direction.as_pair()
        return self.evidence.s7_prediction_window(
            start,
            end,
            detectors=detectors,
            focal_id=pair[0] if pair else None,
            partner_id=pair[1] if pair else None,
        )

    def provenance_at_playhead(self) -> ArbitrationProvenance | None:
        pair = self.selected_direction.as_pair()
        if pair is None:
            discovered = self.ethogram.discover_directions(
                sample_idx=int(self.playhead.sample_idx), half_window=1
            )
            if not discovered:
                return None
            pair = discovered[0]
        return self.evidence.provenance_at_sample(
            int(self.playhead.sample_idx),
            focal_id=pair[0],
            partner_id=pair[1],
        )

    def provenance_for_selected_bout(self) -> ArbitrationProvenance | None:
        if self.selected_bout_id is None:
            return None
        return self.evidence.provenance_for_bout(self.bouts.get(self.selected_bout_id))

    def timeline_view(
        self,
        *,
        half_window: int = DEFAULT_ETHOGRAM_HALF_WINDOW,
        start_sample_idx: int | None = None,
        end_sample_idx: int | None = None,
        max_segments: int = DEFAULT_TIMELINE_MAX_SEGMENTS,
        include_probability_traces: bool = True,
    ) -> TimelineView:
        window = self.ethogram_window(
            half_window=half_window,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
        )
        overlays = self.bouts.overlapping(
            window.start_sample_idx, window.end_sample_idx
        )
        bout_overlays = self.bouts.overlays_for_ids(
            [str(x) for x in overlays["bout_id"].tolist()],
            selected_bout_id=self.selected_bout_id,
        )
        evidence_points: tuple[Any, ...] = ()
        if include_probability_traces:
            pair = self.selected_direction.as_pair()
            directions = (pair, (pair[1], pair[0])) if pair is not None else None
            echo = self.evidence.s8_echoed_window(
                window.start_sample_idx,
                window.end_sample_idx,
                directions=directions,
            )
            evidence_points = echo.points
        return build_timeline_view(
            window,
            sync=self.sync,
            playhead=self.playhead,
            bout_overlays=bout_overlays,
            evidence_points=evidence_points,
            previous_prepared_frame_idx=self._previous_prepared_frame_idx,
            max_segments=max_segments,
            include_probability_traces=include_probability_traces,
        )

    def decode_current_frame(self) -> DecodedFrame:
        if self.video is None:
            raise VideoAccessError("Prepared video is not available for this session.")
        previous = (
            self.last_decoded.prepared_frame_idx if self.last_decoded is not None else None
        )
        changed = self.video.read_frame_if_changed(
            self.playhead.prepared_frame_idx,
            previous_prepared_frame_idx=previous,
        )
        if changed is None:
            if self.last_decoded is None:
                changed = self.video.read_frame(self.playhead.prepared_frame_idx)
            else:
                return self.last_decoded
        self.last_decoded = changed
        return changed

    def snapshot(self) -> dict[str, Any]:
        """Serializable inspection state (no review decisions)."""

        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "package_run_id": self.package_run_id,
            "package_dir": str(self.source.package_dir),
            "currently_eligible": self.currently_eligible,
            "package_status": self.source.package_status,
            "n_dense_samples": self.n_dense_samples,
            "n_bouts": self.n_bouts,
            "playhead": self.playhead.to_dict(),
            "selected_channel": self.selected_channel,
            "selected_direction": {
                "focal_id": self.selected_direction.focal_id,
                "partner_id": self.selected_direction.partner_id,
            },
            "selected_bout_id": self.selected_bout_id,
            "bout_filter": self.bouts.active_filter.to_dict(),
            "s8_states_available": self.source.s8_states_available,
            "s7_evidence_available": self.evidence.s7_available,
            "stage_diagnostics_availability": (
                self.evidence.stage_diagnostics_availability()
            ),
            "source_s8_run_id": self.source.source_s8.run_id,
            "source_s7_run_id": self.source.source_s7_run_id,
            "source_video_path": (
                str(self.source.source_video.path)
                if self.source.source_video.path is not None
                else None
            ),
            "video_available": self.source.video_available,
            "warnings": list(self.warnings),
        }

    def _window_bounds(
        self,
        *,
        half_window: int,
        start_sample_idx: int | None,
        end_sample_idx: int | None,
    ) -> tuple[int, int]:
        if start_sample_idx is not None and end_sample_idx is not None:
            return int(start_sample_idx), int(end_sample_idx)
        center = int(self.playhead.sample_idx)
        half = max(0, int(half_window))
        start = max(0, center - half)
        end = min(self.sync.n_dense - 1, center + half)
        return start, end

    def _apply_playhead(self, mapping: PlayheadMapping, *, decode: bool) -> None:
        previous_prepared = self.playhead.prepared_frame_idx
        self._previous_prepared_frame_idx = previous_prepared
        self.playhead = mapping
        if not decode:
            return
        if self.video is None:
            raise VideoAccessError("Prepared video is not available for this session.")
        changed = self.video.read_frame_if_changed(
            mapping.prepared_frame_idx,
            previous_prepared_frame_idx=previous_prepared,
        )
        if changed is not None:
            self.last_decoded = changed
        elif self.last_decoded is None:
            self.last_decoded = self.video.read_frame(mapping.prepared_frame_idx)

    def _apply_channel_and_identity_from_bout(self, bout: pd.Series) -> None:
        channel = str(bout["channel"])
        if channel in CHANNEL_VALUES:
            self.selected_channel = channel
        grain = str(bout["identity_grain"])
        if grain == IDENTITY_DIRECTIONAL:
            focal = bout["focal_id"]
            partner = bout["partner_id"]
            self.selected_direction = DirectionSelection(
                int(focal) if pd.notna(focal) else None,
                int(partner) if pd.notna(partner) else None,
            )
        elif grain == IDENTITY_SYMMETRIC_PAIR:
            self.selected_direction = DirectionSelection(None, None)


def open_inspection_session(
    package_dir: Path | str,
    *,
    require_video: bool = True,
    require_s8_states: bool = False,
    initial_sample_idx: int = 0,
    decode_initial_frame: bool = False,
    video_cache_size: int = 8,
) -> InspectionSession:
    """Open a read-only inspection session on an eligible S9 package.

    Does not create ``behavioral_review__*`` runs, draft journals, or
    acceptance decisions.
    """

    source = open_review_source(
        package_dir,
        require_video=require_video,
        require_s8_states=require_s8_states,
    )
    sync = SyncMap.from_sync_index(source.sync_index)
    if sync.n_dense <= 0:
        raise SessionError("Cannot open inspection session with empty sync_index.")
    sample = max(0, min(int(initial_sample_idx), sync.n_dense - 1))
    playhead = sync.mapping_at_sample(sample)

    video: PreparedFrameAccessor | None = None
    if source.video_available and source.source_video.path is not None:
        video = PreparedFrameAccessor(
            source.source_video.path,
            cache_size=video_cache_size,
        )
    elif require_video:
        raise VideoAccessError("Prepared video required but unavailable.")

    ethogram = EthogramAccessor(
        source.source_s8.states_path,
        sync=sync,
        available=source.source_s8.states_available,
    )
    bouts = BoutBrowser(source.behavior_bouts, sync)
    evidence = EvidenceAccessor.from_source(source, sync, ethogram)

    session = InspectionSession(
        schema_version=INSPECTION_SESSION_SCHEMA_VERSION,
        mode=MODE_INSPECTION,
        source=source,
        sync=sync,
        playhead=playhead,
        ethogram=ethogram,
        bouts=bouts,
        evidence=evidence,
        video=video,
        warnings=list(source.warnings),
    )
    if decode_initial_frame:
        session.decode_current_frame()
    return session
