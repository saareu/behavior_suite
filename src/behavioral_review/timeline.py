"""Visualization-oriented inspection data structures (no Qt widgets).

LOD / downsampling simplifies paint payloads only. Underlying behavioral values,
bout boundaries, and future review-editing precision remain unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from behavioral_review.bouts import BoutOverlay
from behavioral_review.contracts import (
    DEFAULT_TIMELINE_MAX_SEGMENTS,
    TIMELINE_VIEW_SCHEMA_VERSION,
)
from behavioral_review.ethogram import (
    DirectionEthogramTrack,
    EthogramLaneMeta,
    EthogramWindow,
    ParallelChannelTrack,
    run_length_segments,
)
from behavioral_review.evidence import DetectorEvidencePoint
from behavioral_review.sync import PlayheadMapping, SyncMap


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    """One contiguous label/pred span for a lane (inclusive dense endpoints)."""

    lane_id: str
    start_sample_idx: int
    end_sample_idx: int
    value: str
    n_samples: int
    lod_merged: bool = False


@dataclass(frozen=True, slots=True)
class ProbabilityTrace:
    """Optional probability/validity series for evidence panes."""

    detector_name: str
    source: str
    focal_id: int | None
    partner_id: int | None
    sample_idx: np.ndarray
    time_sec: np.ndarray
    prob: np.ndarray
    input_valid: np.ndarray
    display_state: np.ndarray
    downsampled: bool


@dataclass(frozen=True, slots=True)
class PlayheadSyncInfo:
    """Dense authority + prepared decode target for GUI sync."""

    sample_idx: int
    time_sec: float
    prepared_frame_idx: int
    prepared_time_sec: float
    plateau_sample_indices: tuple[int, ...]
    prepared_frame_unchanged: bool


@dataclass(frozen=True, slots=True)
class TimelineView:
    """GUI-ready inspection payload for a dense window."""

    schema_version: str
    start_sample_idx: int
    end_sample_idx: int
    lanes: tuple[EthogramLaneMeta, ...]
    segments: tuple[TimelineSegment, ...]
    bout_overlays: tuple[BoutOverlay, ...]
    probability_traces: tuple[ProbabilityTrace, ...]
    playhead: PlayheadSyncInfo
    lod_level: str  # "full" | "overview"
    lod_max_segments: int
    scientific_values_unchanged: bool


def build_playhead_sync(
    sync: SyncMap,
    playhead: PlayheadMapping,
    *,
    previous_prepared_frame_idx: int | None = None,
) -> PlayheadSyncInfo:
    plateau = tuple(int(x) for x in sync.plateau_sample_indices(playhead.prepared_frame_idx))
    unchanged = (
        previous_prepared_frame_idx is not None
        and int(previous_prepared_frame_idx) == int(playhead.prepared_frame_idx)
    )
    return PlayheadSyncInfo(
        sample_idx=int(playhead.sample_idx),
        time_sec=float(playhead.time_sec),
        prepared_frame_idx=int(playhead.prepared_frame_idx),
        prepared_time_sec=float(playhead.prepared_time_sec),
        plateau_sample_indices=plateau,
        prepared_frame_unchanged=unchanged,
    )


def build_timeline_view(
    ethogram: EthogramWindow,
    *,
    sync: SyncMap,
    playhead: PlayheadMapping,
    bout_overlays: Sequence[BoutOverlay] = (),
    evidence_points: Sequence[DetectorEvidencePoint] = (),
    previous_prepared_frame_idx: int | None = None,
    max_segments: int = DEFAULT_TIMELINE_MAX_SEGMENTS,
    include_probability_traces: bool = True,
) -> TimelineView:
    """Assemble lane segments + overlays for a future Qt viewer.

    When segment count exceeds ``max_segments``, adjacent same-value segments
    are merged for overview painting only (``lod_merged=True``). Dense ethogram
    accessors remain authoritative for science and future editing.
    """

    lanes: list[EthogramLaneMeta] = []
    raw_segments: list[TimelineSegment] = []

    for track in ethogram.directions:
        lanes.append(track.lane)
        raw_segments.extend(_segments_from_direction(track))
    parallel_tracks: list[ParallelChannelTrack] = []
    if ethogram.contact is not None:
        parallel_tracks.append(ethogram.contact)
    if ethogram.side_by_side is not None:
        parallel_tracks.append(ethogram.side_by_side)
    for parallel in parallel_tracks:
        lanes.append(parallel.lane)
        raw_segments.extend(_segments_from_parallel(parallel))

    lod_level = "full"
    segments = raw_segments
    if len(raw_segments) > int(max_segments):
        segments = _downsample_segments(raw_segments, max_segments=int(max_segments))
        lod_level = "overview"

    traces: tuple[ProbabilityTrace, ...] = ()
    if include_probability_traces and evidence_points:
        traces = _traces_from_evidence_points(
            evidence_points,
            sync=sync,
            max_points=max(256, int(max_segments)),
        )

    return TimelineView(
        schema_version=TIMELINE_VIEW_SCHEMA_VERSION,
        start_sample_idx=ethogram.start_sample_idx,
        end_sample_idx=ethogram.end_sample_idx,
        lanes=tuple(lanes),
        segments=tuple(segments),
        bout_overlays=tuple(bout_overlays),
        probability_traces=traces,
        playhead=build_playhead_sync(
            sync,
            playhead,
            previous_prepared_frame_idx=previous_prepared_frame_idx,
        ),
        lod_level=lod_level,
        lod_max_segments=int(max_segments),
        scientific_values_unchanged=True,
    )


def _segments_from_direction(track: DirectionEthogramTrack) -> list[TimelineSegment]:
    spans = run_length_segments(track.sample_idx, track.behavior_label)
    return [
        TimelineSegment(
            lane_id=track.lane.lane_id,
            start_sample_idx=start,
            end_sample_idx=end,
            value=str(value),
            n_samples=int(end - start + 1),
            lod_merged=False,
        )
        for start, end, value in spans
    ]


def _segments_from_parallel(track: ParallelChannelTrack) -> list[TimelineSegment]:
    values = np.where(track.pred.astype(np.int8) == 1, "positive", "negative")
    spans = run_length_segments(track.sample_idx, values)
    return [
        TimelineSegment(
            lane_id=track.lane.lane_id,
            start_sample_idx=start,
            end_sample_idx=end,
            value=str(value),
            n_samples=int(end - start + 1),
            lod_merged=False,
        )
        for start, end, value in spans
    ]


def _downsample_segments(
    segments: Sequence[TimelineSegment],
    *,
    max_segments: int,
) -> list[TimelineSegment]:
    """Merge adjacent same-lane/same-value segments until under budget.

    Never alters bout overlays or underlying ethogram values — paint only.
    """

    if len(segments) <= max_segments:
        return list(segments)
    # Group by lane, merge contiguous identical values (already mostly RLE).
    by_lane: dict[str, list[TimelineSegment]] = {}
    for seg in segments:
        by_lane.setdefault(seg.lane_id, []).append(seg)

    merged: list[TimelineSegment] = []
    for lane_id, lane_segs in by_lane.items():
        current = lane_segs[0]
        for nxt in lane_segs[1:]:
            if (
                nxt.value == current.value
                and nxt.start_sample_idx <= current.end_sample_idx + 1
            ):
                current = TimelineSegment(
                    lane_id=lane_id,
                    start_sample_idx=current.start_sample_idx,
                    end_sample_idx=max(current.end_sample_idx, nxt.end_sample_idx),
                    value=current.value,
                    n_samples=int(
                        max(current.end_sample_idx, nxt.end_sample_idx)
                        - current.start_sample_idx
                        + 1
                    ),
                    lod_merged=True,
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    if len(merged) <= max_segments:
        return merged

    # Coarse overview: stride-pick keeping first/last per lane.
    stride = max(1, int(np.ceil(len(merged) / max_segments)))
    out: list[TimelineSegment] = []
    for i, seg in enumerate(merged):
        if i % stride == 0 or i == len(merged) - 1:
            out.append(
                TimelineSegment(
                    lane_id=seg.lane_id,
                    start_sample_idx=seg.start_sample_idx,
                    end_sample_idx=seg.end_sample_idx,
                    value=seg.value,
                    n_samples=seg.n_samples,
                    lod_merged=True,
                )
            )
    return out


def _traces_from_evidence_points(
    points: Sequence[DetectorEvidencePoint],
    *,
    sync: SyncMap,
    max_points: int,
) -> tuple[ProbabilityTrace, ...]:
    grouped: dict[tuple[str, str, int | None, int | None], list[DetectorEvidencePoint]] = {}
    for pt in points:
        key = (pt.detector_name, pt.source, pt.focal_id, pt.partner_id)
        grouped.setdefault(key, []).append(pt)

    traces: list[ProbabilityTrace] = []
    for (det, source, focal, partner), group in grouped.items():
        group_sorted = sorted(group, key=lambda p: p.sample_idx)
        sample_idx = np.asarray([p.sample_idx for p in group_sorted], dtype=np.int64)
        probs = np.asarray(
            [np.nan if p.prob is None else float(p.prob) for p in group_sorted],
            dtype=np.float64,
        )
        valid = np.asarray(
            [
                True if p.input_valid is None else bool(p.input_valid)
                for p in group_sorted
            ],
            dtype=bool,
        )
        display = np.asarray([p.display_state for p in group_sorted], dtype=object)
        downsampled = False
        if sample_idx.size > max_points:
            sample_idx, probs, valid, display = _stride_downsample_preserve_extrema(
                sample_idx, probs, valid, display, max_points=max_points
            )
            downsampled = True
        times = np.full(sample_idx.shape[0], np.nan, dtype=np.float64)
        in_range = (sample_idx >= 0) & (sample_idx < sync.n_dense)
        times[in_range] = sync.time_sec[sample_idx[in_range]]
        traces.append(
            ProbabilityTrace(
                detector_name=det,
                source=source,
                focal_id=focal,
                partner_id=partner,
                sample_idx=sample_idx,
                time_sec=times,
                prob=probs,
                input_valid=valid,
                display_state=display,
                downsampled=downsampled,
            )
        )
    return tuple(traces)


def _stride_downsample_preserve_extrema(
    sample_idx: np.ndarray,
    probs: np.ndarray,
    valid: np.ndarray,
    display: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin by stride; keep first, argmin, argmax, last indices within each bin.

    Display-only; does not rewrite stored science tables.
    """

    n = int(sample_idx.shape[0])
    if n <= max_points:
        return sample_idx, probs, valid, display
    bin_count = max(1, max_points // 4)
    edges = np.linspace(0, n, bin_count + 1, dtype=np.int64)
    keep: list[int] = []
    for b in range(bin_count):
        lo = int(edges[b])
        hi = int(edges[b + 1])
        if lo >= hi:
            continue
        chunk = probs[lo:hi]
        keep.append(lo)
        if np.any(np.isfinite(chunk)):
            keep.append(lo + int(np.nanargmin(chunk)))
            keep.append(lo + int(np.nanargmax(chunk)))
        keep.append(hi - 1)
    uniq = np.unique(np.asarray(keep, dtype=np.int64))
    return sample_idx[uniq], probs[uniq], valid[uniq], display[uniq]


def timeline_view_to_dict(view: TimelineView) -> dict[str, Any]:
    """JSON-friendly summary (arrays omitted / summarized)."""

    return {
        "schema_version": view.schema_version,
        "start_sample_idx": view.start_sample_idx,
        "end_sample_idx": view.end_sample_idx,
        "n_lanes": len(view.lanes),
        "n_segments": len(view.segments),
        "n_bout_overlays": len(view.bout_overlays),
        "n_probability_traces": len(view.probability_traces),
        "lod_level": view.lod_level,
        "lod_max_segments": view.lod_max_segments,
        "scientific_values_unchanged": view.scientific_values_unchanged,
        "playhead": {
            "sample_idx": view.playhead.sample_idx,
            "time_sec": view.playhead.time_sec,
            "prepared_frame_idx": view.playhead.prepared_frame_idx,
            "prepared_time_sec": view.playhead.prepared_time_sec,
            "prepared_frame_unchanged": view.playhead.prepared_frame_unchanged,
            "plateau_n": len(view.playhead.plateau_sample_indices),
        },
        "lanes": [
            {
                "lane_id": lane.lane_id,
                "channel": lane.channel,
                "identity_grain": lane.identity_grain,
                "display_name": lane.display_name,
                "focal_id": lane.focal_id,
                "partner_id": lane.partner_id,
            }
            for lane in view.lanes
        ],
    }
