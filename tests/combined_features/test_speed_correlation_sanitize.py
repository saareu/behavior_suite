"""Unit tests for Pearson speed-correlation sanitization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from combined_features.frame_features import sanitize_pearson_correlation


def test_sanitize_pearson_correlation_rejects_inf_and_wild_values() -> None:
    raw = pd.Series(
        [0.5, -1.0, 1.0, 1.0000001, -np.inf, np.inf, -57.8, np.nan, 0.0],
        dtype=float,
    )
    cleaned = sanitize_pearson_correlation(raw)
    expected = np.array(
        [0.5, -1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, 0.0],
        dtype=float,
    )
    np.testing.assert_allclose(
        cleaned.to_numpy(dtype=float),
        expected,
        equal_nan=True,
    )
    assert not np.isinf(cleaned.to_numpy(dtype=float)).any()


def test_sanitize_pearson_correlation_does_not_neighbor_fill() -> None:
    raw = pd.Series([0.9, np.inf, 0.8], dtype=float)
    cleaned = sanitize_pearson_correlation(raw)
    assert cleaned.iloc[0] == 0.9
    assert pd.isna(cleaned.iloc[1])
    assert cleaned.iloc[2] == 0.8
