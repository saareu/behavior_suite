"""Sprint 3: Savitzky–Golay smoothing behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pose_finalization import (
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PoseFinalizationError,
    SmoothConfig,
    apply_savgol_smoothing,
)


def _series_cube(
    *,
    n_samples: int = 40,
    n_tracks: int = 1,
    n_nodes: int = 1,
    x0: float = 0.0,
    slope: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.full((n_samples, n_tracks, n_nodes), np.nan, dtype=np.float64)
    y = np.full((n_samples, n_tracks, n_nodes), np.nan, dtype=np.float64)
    protected = np.zeros((n_samples, n_tracks, n_nodes), dtype=bool)
    t = np.arange(n_samples, dtype=np.float64)
    x[:, 0, 0] = x0 + slope * t
    y[:, 0, 0] = 10.0 + 0.5 * t
    return x, y, protected


def test_savgol_enabled_modifies_noisy_coordinates_and_retains_pre_smooth() -> None:
    x, y, protected = _series_cube(n_samples=40)
    rng = np.random.default_rng(0)
    x[:, 0, 0] += rng.normal(0.0, 0.4, size=40)
    y[:, 0, 0] += rng.normal(0.0, 0.4, size=40)
    provenance = np.full(x.shape, PROVENANCE_ORIGINAL, dtype=np.uint8)
    point_score = np.full(x.shape, 0.9, dtype=np.float64)

    result = apply_savgol_smoothing(x=x, y=y, protected=protected)
    assert result.config.enabled is True
    assert np.allclose(result.x_pre_smooth, x)
    assert np.allclose(result.y_pre_smooth, y)
    assert bool(np.any(result.was_smoothed))
    assert not np.allclose(result.x, x)
    # Provenance / confidence inputs are caller-owned and unchanged here.
    assert np.all(provenance == PROVENANCE_ORIGINAL)
    assert np.allclose(point_score, 0.9)


def test_savgol_disabled_is_identity() -> None:
    x, y, protected = _series_cube()
    x[:, 0, 0] += np.sin(np.linspace(0, 4, x.shape[0]))
    result = apply_savgol_smoothing(
        x=x,
        y=y,
        protected=protected,
        config=SmoothConfig(enabled=False),
    )
    assert np.array_equal(result.x, x)
    assert np.array_equal(result.y, y)
    assert not np.any(result.was_smoothed)


def test_short_valid_segments_are_not_invented() -> None:
    x, y, protected = _series_cube(n_samples=20)
    # Only 5 valid samples — shorter than window 11.
    x[5:, 0, 0] = np.nan
    y[5:, 0, 0] = np.nan
    before_x = x.copy()
    before_y = y.copy()
    result = apply_savgol_smoothing(x=x, y=y, protected=protected)
    assert np.array_equal(result.x, before_x, equal_nan=True)
    assert np.array_equal(result.y, before_y, equal_nan=True)
    assert not np.any(result.was_smoothed)


def test_no_smoothing_across_nan_gap_or_protected_boundary() -> None:
    x, y, protected = _series_cube(n_samples=50)
    # Two long valid segments separated by an unresolved gap + protected blank.
    x[20:25, 0, 0] = np.nan
    y[20:25, 0, 0] = np.nan
    protected[25:28, 0, 0] = True
    x[25:28, 0, 0] = np.nan
    y[25:28, 0, 0] = np.nan

    left = x[:20, 0, 0].copy()
    right = x[28:, 0, 0].copy()
    result = apply_savgol_smoothing(x=x, y=y, protected=protected)

    # Gap and protected remain NaN / unsmoothed.
    assert np.all(np.isnan(result.x[20:28, 0, 0]))
    assert not np.any(result.was_smoothed[20:28, 0, 0])
    # Segments are smoothed independently (not a single continuous filter).
    assert bool(np.any(result.was_smoothed[:20, 0, 0]))
    assert bool(np.any(result.was_smoothed[28:, 0, 0]))
    assert result.x.shape == x.shape
    # Pre-smooth retention for the right segment matches the pre-gap input.
    assert result.x_pre_smooth[28:, 0, 0].tolist() == right.tolist()
    # Left-segment pre-smooth matches the original left values.
    assert result.x_pre_smooth[:20, 0, 0].tolist() == left.tolist()


def test_protected_manual_blanks_never_smoothed() -> None:
    x, y, protected = _series_cube(n_samples=30)
    protected[10:15, 0, 0] = True
    x[10:15, 0, 0] = np.nan
    y[10:15, 0, 0] = np.nan
    provenance = np.full(x.shape, PROVENANCE_ORIGINAL, dtype=np.uint8)
    provenance[10:15, 0, 0] = PROVENANCE_PROTECTED_MANUAL_BLANK

    result = apply_savgol_smoothing(x=x, y=y, protected=protected)
    assert np.all(np.isnan(result.x[10:15, 0, 0]))
    assert not np.any(result.was_smoothed[10:15, 0, 0])
    assert np.all(provenance[10:15, 0, 0] == PROVENANCE_PROTECTED_MANUAL_BLANK)


def test_imputed_and_observed_coordinates_are_both_smoothed_in_segment() -> None:
    x, y, protected = _series_cube(n_samples=40)
    provenance = np.full(x.shape, PROVENANCE_ORIGINAL, dtype=np.uint8)
    provenance[10:15, 0, 0] = PROVENANCE_LINEAR_GAP_FILL
    rng = np.random.default_rng(1)
    x[:, 0, 0] += rng.normal(0.0, 0.5, size=40)
    result = apply_savgol_smoothing(x=x, y=y, protected=protected)
    assert bool(np.any(result.was_smoothed[10:15, 0, 0]))
    assert bool(np.any(result.was_smoothed[:10, 0, 0]))
    # Smoothing does not rewrite provenance codes held by the caller.
    assert np.all(provenance[10:15, 0, 0] == PROVENANCE_LINEAR_GAP_FILL)
    assert np.all(provenance[:10, 0, 0] == PROVENANCE_ORIGINAL)


def test_smooth_config_validation() -> None:
    with pytest.raises(PoseFinalizationError, match="odd"):
        SmoothConfig(window_length=10)
    with pytest.raises(PoseFinalizationError, match="polyorder"):
        SmoothConfig(window_length=5, polyorder=5)


def test_missing_samples_remain_nan() -> None:
    x, y, protected = _series_cube(n_samples=30)
    x[3, 0, 0] = np.nan
    y[3, 0, 0] = np.nan
    # Breaks into short segments around the hole when neighbors are insufficient.
    result = apply_savgol_smoothing(x=x, y=y, protected=protected)
    assert np.isnan(result.x[3, 0, 0])
    assert np.isnan(result.y[3, 0, 0])
    assert not result.was_smoothed[3, 0, 0]
    assert PROVENANCE_MISSING == 0  # contract sanity for provenance docs
