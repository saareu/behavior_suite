"""Feature-matrix validation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    FeatureValidationError,
    require_feature_matrix,
    validate_feature_matrix,
)


def test_missing_features(feature_frame: pd.DataFrame) -> None:
    result = validate_feature_matrix(feature_frame, ["feat_a", "missing_col"])
    assert result.ok is False
    assert result.matrix is None
    assert any("missing" in e for e in result.errors)


def test_duplicate_feature_names(feature_frame: pd.DataFrame) -> None:
    result = validate_feature_matrix(feature_frame, ["feat_a", "feat_a"])
    assert result.ok is False
    assert any("duplicate" in e for e in result.errors)


def test_feature_ordering(feature_frame: pd.DataFrame) -> None:
    result = validate_feature_matrix(feature_frame, ["feat_b", "feat_a"])
    assert result.ok
    assert result.matrix is not None
    assert result.matrix.shape == (3, 2)
    np.testing.assert_allclose(result.matrix[0], [1.0, 0.1])


def test_numeric_dtype_rejection() -> None:
    frame = pd.DataFrame({"feat_a": ["x", "y"], "feat_b": [1.0, 2.0]})
    result = validate_feature_matrix(frame, ["feat_a", "feat_b"])
    assert result.ok is False
    assert any("non-numeric" in e for e in result.errors)


def test_nan_preserved(feature_frame: pd.DataFrame) -> None:
    result = validate_feature_matrix(feature_frame, ["feat_a", "feat_b"])
    assert result.ok
    assert result.matrix is not None
    assert np.isnan(result.matrix[2, 0])
    assert result.nan_counts["feat_a"] == 1
    assert any("NaN" in w for w in result.warnings)


def test_inf_rejected(feature_frame: pd.DataFrame) -> None:
    frame = feature_frame.copy()
    frame.loc[0, "feat_b"] = np.inf
    result = validate_feature_matrix(frame, ["feat_a", "feat_b"])
    assert result.ok is False
    assert any("Inf" in e for e in result.errors)
    with pytest.raises(FeatureValidationError, match="Inf"):
        require_feature_matrix(frame, ["feat_a", "feat_b"])


def test_empty_rows_rejected(feature_frame: pd.DataFrame) -> None:
    empty = feature_frame.iloc[0:0]
    result = validate_feature_matrix(empty, ["feat_a"], min_rows=1)
    assert result.ok is False
    assert any("row" in e for e in result.errors)


def test_shape_matches_spec(feature_frame: pd.DataFrame) -> None:
    matrix = require_feature_matrix(feature_frame, ["feat_a", "feat_b", "feat_c"])
    assert matrix.shape == (3, 3)
