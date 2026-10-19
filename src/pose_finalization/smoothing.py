"""Savitzky–Golay smoothing for dense 120 Hz pose trajectories.

Applied after temporal resampling and gap filling. Smooths X and Y
independently within continuous valid segments. Never smooths across
unresolved NaN gaps, protected manual blanks, or segment discontinuities.
Short segments that cannot support the configured window are left unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from pose_finalization.contracts import (
    SAVGOL_ENABLED_DEFAULT,
    SAVGOL_POLYORDER_DEFAULT,
    SAVGOL_WINDOW_LENGTH_DEFAULT,
    PoseFinalizationError,
)


@dataclass(frozen=True, slots=True)
class SmoothConfig:
    """Configurable Savitzky–Golay parameters."""

    enabled: bool = SAVGOL_ENABLED_DEFAULT
    window_length: int = SAVGOL_WINDOW_LENGTH_DEFAULT
    polyorder: int = SAVGOL_POLYORDER_DEFAULT

    def __post_init__(self) -> None:
        if self.window_length < 3:
            raise PoseFinalizationError("window_length must be >= 3.")
        if self.window_length % 2 == 0:
            raise PoseFinalizationError("window_length must be odd.")
        if self.polyorder < 0:
            raise PoseFinalizationError("polyorder must be non-negative.")
        if self.polyorder >= self.window_length:
            raise PoseFinalizationError(
                "polyorder must be strictly less than window_length."
            )


@dataclass(frozen=True, slots=True)
class SmoothResult:
    """Smoothed dense coordinates with pre-smooth retention and a change mask."""

    x: np.ndarray
    y: np.ndarray
    x_pre_smooth: np.ndarray
    y_pre_smooth: np.ndarray
    was_smoothed: np.ndarray
    config: SmoothConfig

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_tracks(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[2])


def apply_savgol_smoothing(
    *,
    x: np.ndarray,
    y: np.ndarray,
    protected: np.ndarray,
    config: SmoothConfig | None = None,
) -> SmoothResult:
    """Smooth finite, unprotected coordinates segment-wise with SavGol.

    Preserves pre-smoothing coordinates. Does not invent values for short
    segments, NaNs, or protected samples. Does not modify provenance or
    confidence (callers keep those arrays unchanged).
    """

    cfg = config if config is not None else SmoothConfig()
    if x.shape != y.shape or x.shape != protected.shape:
        raise PoseFinalizationError("Smoothing array shapes must match.")
    if x.ndim != 3:
        raise PoseFinalizationError("Smoothing arrays must be (sample, track, node).")

    x_pre = np.array(x, dtype=np.float64, copy=True)
    y_pre = np.array(y, dtype=np.float64, copy=True)
    out_x = np.array(x, dtype=np.float64, copy=True)
    out_y = np.array(y, dtype=np.float64, copy=True)
    was_smoothed = np.zeros(x.shape, dtype=bool)

    if not cfg.enabled:
        return _freeze_result(out_x, out_y, x_pre, y_pre, was_smoothed, cfg)

    for track_i in range(x.shape[1]):
        for node_i in range(x.shape[2]):
            _smooth_series(
                out_x[:, track_i, node_i],
                out_y[:, track_i, node_i],
                was_smoothed[:, track_i, node_i],
                protected[:, track_i, node_i],
                cfg,
            )

    return _freeze_result(out_x, out_y, x_pre, y_pre, was_smoothed, cfg)


def _smooth_series(
    x: np.ndarray,
    y: np.ndarray,
    was_smoothed: np.ndarray,
    protected: np.ndarray,
    config: SmoothConfig,
) -> None:
    """In-place SavGol over continuous valid segments of one (track, node)."""

    valid = np.isfinite(x) & np.isfinite(y) & ~protected
    for start, end in _continuous_true_runs(valid):
        length = end - start
        if length < config.window_length:
            # Too short for the configured window: leave coordinates unchanged.
            continue
        segment_x = x[start:end]
        segment_y = y[start:end]
        smoothed_x = savgol_filter(
            segment_x,
            window_length=config.window_length,
            polyorder=config.polyorder,
            mode="interp",
        )
        smoothed_y = savgol_filter(
            segment_y,
            window_length=config.window_length,
            polyorder=config.polyorder,
            mode="interp",
        )
        changed = (smoothed_x != segment_x) | (smoothed_y != segment_y)
        x[start:end] = smoothed_x
        y[start:end] = smoothed_y
        was_smoothed[start:end] = changed


def _continuous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-start / exclusive-end ranges of continuous True values."""

    if mask.size == 0:
        return []
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _freeze_result(
    x: np.ndarray,
    y: np.ndarray,
    x_pre: np.ndarray,
    y_pre: np.ndarray,
    was_smoothed: np.ndarray,
    config: SmoothConfig,
) -> SmoothResult:
    for array in (x, y, x_pre, y_pre, was_smoothed):
        array.setflags(write=False)
    return SmoothResult(
        x=x,
        y=y,
        x_pre_smooth=x_pre,
        y_pre_smooth=y_pre,
        was_smoothed=was_smoothed,
        config=config,
    )
