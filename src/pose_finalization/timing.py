"""Authoritative S1 timing load and deterministic 120 Hz target mapping.

Source timeline is always S1 ``prepared_time_sec`` / ``prepared_frame_idx`` from
``prepared_sync.npz``. This module builds a uniform target clock and records
left/right source indices plus interpolation weights for later pose resampling.
It does not resample video, modify S1 artifacts, or interpolate coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose_finalization.contracts import TARGET_FPS_DEFAULT, PoseFinalizationError
from preprocess.exceptions import SyncValidationError
from preprocess.sync_writer import load_prepared_sync_npz

# Absolute time tolerance for treating a target sample as coinciding with a
# source frame. Chosen well below one microsecond so exact 1/fps grids match
# while non-exact samples still require temporal derivation.
_TIME_ATOL_SEC = 1e-9


@dataclass(frozen=True, slots=True)
class SourceTiming:
    """Authoritative prepared-frame timing from S1."""

    prepared_frame_idx: np.ndarray
    prepared_time_sec: np.ndarray
    fps_header: float
    frame_count: int
    prepared_sync_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TimingMap:
    """Deterministic mapping from a uniform target timeline onto S1 frames.

    For each target sample:
    - ``is_exact_source_sample`` is True when the target time coincides with an
      observed prepared frame (left == right, weight == 0).
    - Otherwise the sample requires temporal derivation between
      ``left_source_frame_idx`` and ``right_source_frame_idx`` with
      ``interpolation_weight`` in ``(0, 1)`` giving the fractional distance
      from left toward right.
    """

    target_fps: float
    target_frame_idx: np.ndarray
    target_time_sec: np.ndarray
    left_source_frame_idx: np.ndarray
    right_source_frame_idx: np.ndarray
    left_source_time_sec: np.ndarray
    right_source_time_sec: np.ndarray
    interpolation_weight: np.ndarray
    is_exact_source_sample: np.ndarray
    source_fps_header: float
    source_frame_count: int
    prepared_sync_path: Path | None = None

    @property
    def target_sample_count(self) -> int:
        return int(self.target_frame_idx.shape[0])


def load_source_timing(prepared_sync_path: Path | str) -> SourceTiming:
    """Load and validate authoritative S1 prepared timing (read-only)."""

    path = Path(prepared_sync_path).expanduser().resolve()
    try:
        sync = load_prepared_sync_npz(path)
    except SyncValidationError as exc:
        raise PoseFinalizationError(
            f"Could not load authoritative S1 prepared_sync.npz: {exc}"
        ) from exc

    prepared_frame_idx = np.asarray(sync.prepared_frame_idx, dtype=np.int64)
    prepared_time_sec = np.asarray(sync.prepared_time_sec, dtype=np.float64)
    _validate_source_arrays(prepared_frame_idx, prepared_time_sec)
    return SourceTiming(
        prepared_frame_idx=prepared_frame_idx,
        prepared_time_sec=prepared_time_sec,
        fps_header=float(sync.fps_header),
        frame_count=int(sync.prepared_frame_count_opencv_readable),
        prepared_sync_path=path,
    )


def build_uniform_target_timeline(
    source_time_sec: np.ndarray,
    *,
    target_fps: float = TARGET_FPS_DEFAULT,
) -> np.ndarray:
    """Build an exactly uniform target timeline spanning the source times.

    The timeline starts at ``source_time_sec[0]`` and ends at or before
    ``source_time_sec[-1]`` with spacing ``1 / target_fps``.
    """

    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise PoseFinalizationError("target_fps must be finite and positive.")

    times = _as_finite_1d(source_time_sec, "source_time_sec")
    if times.size == 0:
        raise PoseFinalizationError("source_time_sec must not be empty.")
    _require_strictly_increasing(times, "source_time_sec")

    t0 = float(times[0])
    t1 = float(times[-1])
    dt = 1.0 / float(target_fps)
    # Inclusive count of samples on [t0, t1] at exact target_fps spacing.
    sample_count = int(np.floor(((t1 - t0) * float(target_fps)) + _TIME_ATOL_SEC)) + 1
    if sample_count <= 0:
        raise PoseFinalizationError("Uniform target timeline produced no samples.")

    target_time_sec = t0 + np.arange(sample_count, dtype=np.float64) * dt
    # Guard floating-point overshoot past the last observed source time.
    if float(target_time_sec[-1]) > t1 + _TIME_ATOL_SEC:
        target_time_sec = target_time_sec[target_time_sec <= t1 + _TIME_ATOL_SEC]
    if target_time_sec.size == 0:
        raise PoseFinalizationError("Uniform target timeline produced no in-range samples.")
    target_time_sec.setflags(write=False)
    return target_time_sec


def map_target_to_source(
    *,
    source_frame_idx: np.ndarray,
    source_time_sec: np.ndarray,
    target_time_sec: np.ndarray,
    target_fps: float = TARGET_FPS_DEFAULT,
    source_fps_header: float | None = None,
    prepared_sync_path: Path | None = None,
) -> TimingMap:
    """Map each target sample onto left/right source frames and weights."""

    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise PoseFinalizationError("target_fps must be finite and positive.")

    frames = np.asarray(source_frame_idx, dtype=np.int64)
    times = _as_finite_1d(source_time_sec, "source_time_sec")
    targets = _as_finite_1d(target_time_sec, "target_time_sec")
    _validate_source_arrays(frames, times)
    if targets.size == 0:
        raise PoseFinalizationError("target_time_sec must not be empty.")
    _require_strictly_increasing(targets, "target_time_sec")

    t0 = float(times[0])
    t1 = float(times[-1])
    if float(targets[0]) < t0 - _TIME_ATOL_SEC or float(targets[-1]) > t1 + _TIME_ATOL_SEC:
        raise PoseFinalizationError(
            "Target timeline extends outside the authoritative source time range."
        )

    # Insertion index of the left neighbor for each target (may be -1 if below).
    left_pos = np.searchsorted(times, targets, side="right") - 1
    left_idx = np.empty(targets.shape[0], dtype=np.int64)
    right_idx = np.empty(targets.shape[0], dtype=np.int64)
    weights = np.empty(targets.shape[0], dtype=np.float64)
    exact = np.zeros(targets.shape[0], dtype=bool)

    # Exact coincidence: target matches a source sample within absolute tolerance.
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
        chosen = candidate[hit]
        left_idx[hit] = chosen
        right_idx[hit] = chosen
        weights[hit] = 0.0
        exact[hit] = True

    needs_interp = ~exact
    if np.any(needs_interp):
        interp_i = np.nonzero(needs_interp)[0]
        left = left_pos[interp_i]
        if np.any(left < 0) or np.any(left >= times.shape[0] - 1):
            raise PoseFinalizationError(
                "One or more target times are outside interpolable source bounds "
                f"[{t0}, {t1}]."
            )
        right = left + 1
        span = times[right] - times[left]
        if np.any(span <= 0.0):
            raise PoseFinalizationError(
                "Source times are not strictly increasing around a target sample."
            )
        weight = (targets[interp_i] - times[left]) / span
        if np.any(weight < 0.0) or np.any(weight > 1.0):
            raise PoseFinalizationError(
                "One or more interpolation weights fell outside [0, 1]."
            )
        left_idx[interp_i] = left
        right_idx[interp_i] = right
        weights[interp_i] = weight

    left_frames = frames[left_idx]
    right_frames = frames[right_idx]
    left_times = times[left_idx]
    right_times = times[right_idx]
    target_frame_idx = np.arange(targets.shape[0], dtype=np.int64)

    for array in (
        target_frame_idx,
        targets,
        left_frames,
        right_frames,
        left_times,
        right_times,
        weights,
        exact,
    ):
        array.setflags(write=False)

    resolved_fps = (
        float(source_fps_header)
        if source_fps_header is not None
        else _infer_source_fps(times)
    )
    return TimingMap(
        target_fps=float(target_fps),
        target_frame_idx=target_frame_idx,
        target_time_sec=targets,
        left_source_frame_idx=left_frames,
        right_source_frame_idx=right_frames,
        left_source_time_sec=left_times,
        right_source_time_sec=right_times,
        interpolation_weight=weights,
        is_exact_source_sample=exact,
        source_fps_header=resolved_fps,
        source_frame_count=int(frames.shape[0]),
        prepared_sync_path=prepared_sync_path,
    )


def build_timing_map_from_prepared_sync(
    prepared_sync_path: Path | str,
    *,
    target_fps: float = TARGET_FPS_DEFAULT,
) -> TimingMap:
    """Load S1 timing and build the deterministic uniform target mapping."""

    source = load_source_timing(prepared_sync_path)
    target_time_sec = build_uniform_target_timeline(
        source.prepared_time_sec,
        target_fps=target_fps,
    )
    return map_target_to_source(
        source_frame_idx=source.prepared_frame_idx,
        source_time_sec=source.prepared_time_sec,
        target_time_sec=target_time_sec,
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
        raise PoseFinalizationError("prepared_frame_idx must be one-dimensional.")
    if frames.shape[0] == 0:
        raise PoseFinalizationError("prepared_frame_idx must not be empty.")
    if frames.shape[0] != times.shape[0]:
        raise PoseFinalizationError(
            "prepared_frame_idx and prepared_time_sec lengths do not match."
        )
    if not np.issubdtype(frames.dtype, np.integer):
        raise PoseFinalizationError("prepared_frame_idx must use an integer dtype.")
    expected = np.arange(frames.shape[0], dtype=np.int64)
    if not np.array_equal(frames.astype(np.int64, copy=False), expected):
        raise PoseFinalizationError(
            "prepared_frame_idx must equal [0, 1, ..., N - 1]."
        )
    _require_strictly_increasing(times, "prepared_time_sec")


def _as_finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PoseFinalizationError(f"{name} must be a real numeric array.") from exc
    if array.ndim != 1:
        raise PoseFinalizationError(f"{name} must be one-dimensional.")
    if array.size and not np.all(np.isfinite(array)):
        raise PoseFinalizationError(f"{name} must contain only finite values.")
    return array


def _require_strictly_increasing(values: np.ndarray, name: str) -> None:
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise PoseFinalizationError(f"{name} must be strictly increasing.")


def _infer_source_fps(times: np.ndarray) -> float:
    if times.size < 2:
        return float("nan")
    dt = float(np.median(np.diff(times)))
    if dt <= 0.0 or not np.isfinite(dt):
        return float("nan")
    return 1.0 / dt
