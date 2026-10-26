"""Shared predict_proba / binarize helpers for Subsystem 07 inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.contracts import FeatureValidationError, InferenceError
from behavioral_inference.features import require_feature_matrix
from behavioral_inference.loader import LoadedModel


def binarize_probabilities(
    probabilities: np.ndarray | Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Return int8 predictions: 1 iff ``prob >= threshold`` (NaN compares False)."""

    probs = np.asarray(probabilities, dtype=float)
    # NaN >= threshold is False under IEEE / NumPy rules → prediction 0.
    return (probs >= float(threshold)).astype(np.int8)


def score_positive_probabilities(
    features: pd.DataFrame,
    model: LoadedModel,
    *,
    row_mask: np.ndarray | pd.Series | None = None,
    excluded_fill: float = 0.0,
) -> np.ndarray:
    """Score rows with a loaded model and return positive-class probabilities.

    When ``row_mask`` is provided, only masked rows are scored; excluded rows
    receive ``excluded_fill`` (legacy contact / nose-to-body routers use 0.0).
    Empty mask → all-fill array without calling ``predict_proba``.
    """

    n_rows = len(features)
    if n_rows == 0:
        return np.zeros(0, dtype=float)

    feature_cols = model.bundle.feature_cols
    if not feature_cols:
        raise InferenceError(
            f"{model.detector_name}: feature specification is empty; cannot score"
        )

    if row_mask is None:
        try:
            matrix = require_feature_matrix(features, feature_cols)
        except FeatureValidationError as exc:
            raise InferenceError(
                f"{model.detector_name}: feature validation failed: {exc}"
            ) from exc
        return np.asarray(model.positive_proba(matrix), dtype=float)

    mask = np.asarray(row_mask, dtype=bool)
    if mask.shape != (n_rows,):
        raise InferenceError(
            f"{model.detector_name}: row_mask length {mask.shape} does not match "
            f"feature rows ({n_rows},)"
        )

    out = np.full(n_rows, float(excluded_fill), dtype=float)
    if not bool(mask.any()):
        return out

    try:
        matrix = require_feature_matrix(features.loc[mask], feature_cols)
    except FeatureValidationError as exc:
        raise InferenceError(
            f"{model.detector_name}: feature validation failed for masked rows: {exc}"
        ) from exc
    scored = np.asarray(model.positive_proba(matrix), dtype=float)
    if scored.shape != (int(mask.sum()),):
        raise InferenceError(
            f"{model.detector_name}: predict_proba returned {scored.shape[0]} rows; "
            f"expected {int(mask.sum())}"
        )
    out[mask] = scored
    return out


def ensure_blob_helper_columns(
    features: pd.DataFrame,
    *,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Create NaN blob helper columns when absent (legacy Pipeline 70/90 defensive)."""

    out = features
    for col in ("cc_count", "min_inter_contour_dist"):
        if col not in out.columns:
            if warnings is not None:
                warnings.append(
                    f"Missing blob helper column '{col}'; filling with NaN for inference"
                )
            out = out.copy()
            out[col] = np.nan
    return out
