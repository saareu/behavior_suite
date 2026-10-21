"""Nearest-neighbor 120 Hz mapping onto authoritative S1 prepared timing.

Reuses ``pose_finalization.timing`` for source load and uniform target timeline
construction so S5 ``time_sec`` matches S4 for the same sync. Mapping differs
from S4: each target sample selects exactly one prepared frame (no pose
interpolation between frames).

Equidistant ties (identical absolute distance to two neighbors) select the
**earlier** source frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blob_features.contracts import TARGET_FPS_DEFAULT, BlobFeaturesError
from pose_finalization.contracts import PoseFinalizationError
from pose_finalization.timing import (
    SourceTiming,
    build_uniform_target_timeline,
    load_source_timing,
)

_TIME_ATOL_SEC = 1e-9


@dataclass(frozen=True, slots=True)
class NearestNeighborTimingMap:
    """Deterministic nearest-neighbor map from uniform 120 Hz targets to S1 frames."""

    target_fps: float
    target_frame_idx: np.ndarray
    target_time_sec: np.ndarray
    orig_frame_idx: np.ndarray
    source_time_sec: np.ndarray
    is_exact_source_sample: np.ndarray
    source_fps_header: float
    source_frame_count: int
    prepared_sync_path: Path | None = None

    @property
    def target_sample_count(self) -> int:
        return int(self.target_frame_idx.shape[0])


def build_nearest_neighbor_timing_map(
    *,
    source_frame_idx: np.ndarray,
    source_time_sec: np.ndarray,
    target_time_sec: np.ndarray | None = None,
    target_fps: float = TARGET_FPS_DEFAULT,
    source_fps_header: float | None = None,
    prepared_sync_path: Path | None = None,
) -> NearestNeighborTimingMap:
    """Map each target sample to the nearest prepared-frame index.

    When ``target_time_sec`` is omitted, builds the uniform lab timeline via
    ``build_uniform_target_timeline`` (same helper as S4).
    """

    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise BlobFeaturesError("target_fps must be finite and positive.")

    frames = np.asarray(source_frame_idx, dtype=np.int64)
    times = _as_finite_1d(source_time_sec, "source_time_sec")
    _validate_source_arrays(frames, times)

    if target_time_sec is None:
        try:
            targets = build_uniform_target_timeline(times, target_fps=target_fps)
        except PoseFinalizationError as exc:
            raise BlobFeaturesError(str(exc)) from exc
    else:
        targets = _as_finite_1d(target_time_sec, "target_time_sec")
        if targets.size == 0:
            raise BlobFeaturesError("target_time_sec must not be empty.")
        _require_strictly_increasing(targets, "target_time_sec")

    t0 = float(times[0])
    t1 = float(times[-1])
    if float(targets[0]) < t0 - _TIME_ATOL_SEC or float(targets[-1]) > t1 + _TIME_ATOL_SEC:
        raise BlobFeaturesError(
            "Target timeline extends outside the authoritative source time range."
        )

    orig_idx, exact = map_target_nearest_source_frame(
        source_time_sec=times,
        target_time_sec=targets,
    )
    orig_frames = frames[orig_idx]
    source_times_mapped = times[orig_idx]
    target_frame_idx = np.arange(targets.shape[0], dtype=np.int64)

    # Validate mapping contracts.
    if orig_frames.shape[0] != targets.shape[0]:
        raise BlobFeaturesError("Nearest-neighbor map length does not match target timeline.")
    if np.any(orig_frames < 0) or np.any(orig_frames >= frames.shape[0]):
        raise BlobFeaturesError("Nearest-neighbor map produced out-of-bounds frame indices.")
    if targets.shape[0] > 1 and not np.all(np.diff(orig_frames.astype(np.int64)) >= 0):
        raise BlobFeaturesError(
            "Nearest-neighbor orig_frame_idx must be monotonically non-decreasing."
        )
    # Uniform timelines start at t0 (exact) and end at or before t1.
    if abs(float(targets[0]) - t0) <= _TIME_ATOL_SEC and int(orig_frames[0]) != int(frames[0]):
        raise BlobFeaturesError(
            "Nearest-neighbor map must select the first source frame at the timeline start."
        )
    if abs(float(targets[-1]) - t1) <= _TIME_ATOL_SEC and int(orig_frames[-1]) != int(
        frames[-1]
    ):
        raise BlobFeaturesError(
            "Nearest-neighbor map must select the last source frame when the "
            "target timeline ends on the final source time."
        )

    for array in (target_frame_idx, targets, orig_frames, source_times_mapped, exact):
        array.setflags(write=False)

    resolved_fps = (
        float(source_fps_header)
        if source_fps_header is not None
        else _infer_source_fps(times)
    )
    return NearestNeighborTimingMap(
        target_fps=float(target_fps),
        target_frame_idx=target_frame_idx,
        target_time_sec=targets,
        orig_frame_idx=orig_frames,
        source_time_sec=source_times_mapped,
        is_exact_source_sample=exact,
        source_fps_header=resolved_fps,
        source_frame_count=int(frames.shape[0]),
        prepared_sync_path=prepared_sync_path,
    )


def map_target_nearest_source_frame(
    *,
    source_time_sec: np.ndarray,
    target_time_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(source_position, is_exact)`` for each target time.

    Positions are indices into ``source_time_sec``. Exact coincidence uses the
    same absolute tolerance as S4 (``1e-9`` s). Equidistant left/right neighbors
    select the earlier (left) source sample.
    """

    times = _as_finite_1d(source_time_sec, "source_time_sec")
    targets = _as_finite_1d(target_time_sec, "target_time_sec")
    if times.size == 0:
        raise BlobFeaturesError("source_time_sec must not be empty.")
    if targets.size == 0:
        raise BlobFeaturesError("target_time_sec must not be empty.")
    _require_strictly_increasing(times, "source_time_sec")
    _require_strictly_increasing(targets, "target_time_sec")

    t0 = float(times[0])
    t1 = float(times[-1])
    if float(targets[0]) < t0 - _TIME_ATOL_SEC or float(targets[-1]) > t1 + _TIME_ATOL_SEC:
        raise BlobFeaturesError(
            "Target timeline extends outside the authoritative source time range."
        )

    # Insertion index of the left neighbor for each target (may be -1 if below).
    left_pos = np.searchsorted(times, targets, side="right") - 1
    chosen = np.empty(targets.shape[0], dtype=np.int64)
    exact = np.zeros(targets.shape[0], dtype=bool)

    for candidate_offset in (0, 1):
        candidate = left_pos + candidate_offset
        valid = (candidate >= 0) & (candidate < times.shape[0])
        if not np.any(valid):
            continue
        idxs = np.nonzero(valid)[0]
        source_i = candidate[idxs]
        matches = np.abs(times[source_i] - targets[idxs]) <= _TIME_ATOL_SEC
        hit = idxs[matches]
        if hit.size == 0:
            continue
        chosen[hit] = candidate[hit]
        exact[hit] = True

    needs_nn = ~exact
    if np.any(needs_nn):
        interp_i = np.nonzero(needs_nn)[0]
        left = left_pos[interp_i]
        if np.any(left < 0) or np.any(left >= times.shape[0] - 1):
            raise BlobFeaturesError(
                "One or more target times are outside nearest-neighbor source bounds "
                f"[{t0}, {t1}]."
            )
        right = left + 1
        dist_left = np.abs(targets[interp_i] - times[left])
        dist_right = np.abs(targets[interp_i] - times[right])
        # Tie-break: earlier (left) wins when distances are equal.
        use_left = dist_left <= dist_right
        chosen[interp_i] = np.where(use_left, left, right)

    chosen.setflags(write=False)
    exact.setflags(write=False)
    return chosen, exact


def build_nearest_neighbor_timing_map_from_prepared_sync(
    prepared_sync_path: Path | str,
    *,
    target_fps: float = TARGET_FPS_DEFAULT,
) -> NearestNeighborTimingMap:
    """Load S1 timing and build the deterministic nearest-neighbor 120 Hz map."""

    try:
        source: SourceTiming = load_source_timing(prepared_sync_path)
    except PoseFinalizationError as exc:
        raise BlobFeaturesError(str(exc)) from exc

    return build_nearest_neighbor_timing_map(
        source_frame_idx=source.prepared_frame_idx,
        source_time_sec=source.prepared_time_sec,
        target_fps=target_fps,
        source_fps_header=source.fps_header,
        prepared_sync_path=source.prepared_sync_path,
    )


def _validate_source_arrays(
    prepared_frame_idx: np.ndarray,
    prepared_time_sec: np.ndarray,
) -> None:
    frames = np.asarray(prepared_frame_idx)
    times = _as_finite_1d(prepared_time_sec, "prepared_time_sec")
    if frames.ndim != 1:
        raise BlobFeaturesError("prepared_frame_idx must be one-dimensional.")
    if frames.shape[0] == 0:
        raise BlobFeaturesError("prepared_frame_idx must not be empty.")
    if frames.shape[0] != times.shape[0]:
        raise BlobFeaturesError(
            "prepared_frame_idx and prepared_time_sec lengths do not match."
        )
    if not np.issubdtype(frames.dtype, np.integer):
        raise BlobFeaturesError("prepared_frame_idx must use an integer dtype.")
    expected = np.arange(frames.shape[0], dtype=np.int64)
    if not np.array_equal(frames.astype(np.int64, copy=False), expected):
        raise BlobFeaturesError(
            "prepared_frame_idx must equal [0, 1, ..., N - 1]."
        )
    _require_strictly_increasing(times, "prepared_time_sec")


def _as_finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BlobFeaturesError(f"{name} must be a real numeric array.") from exc
    if array.ndim != 1:
        raise BlobFeaturesError(f"{name} must be one-dimensional.")
    if array.size and not np.all(np.isfinite(array)):
        raise BlobFeaturesError(f"{name} must contain only finite values.")
    return array


def _require_strictly_increasing(values: np.ndarray, name: str) -> None:
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise BlobFeaturesError(f"{name} must be strictly increasing.")


def _infer_source_fps(times: np.ndarray) -> float:
    if times.size < 2:
        return float("nan")
    dt = float(np.median(np.diff(times)))
    if dt <= 0.0 or not np.isfinite(dt):
        return float("nan")
    return 1.0 / dt
