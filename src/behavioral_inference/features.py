"""Runtime feature-matrix validation against a model feature specification.

Schema compatibility (S6 ``model_compat``) only checks column presence / dtype
warnings. This module validates **executability**: exact names, exact order,
no duplicates, numeric dtypes, Inf rejection, and shape. NaN values are
preserved for XGBoost native missing-value handling — they are not imputed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.contracts import FeatureValidationError


@dataclass(frozen=True, slots=True)
class FeatureValidationResult:
    """Outcome of validating a feature table against an ordered feature list."""

    ok: bool
    feature_cols: tuple[str, ...]
    n_rows: int
    n_features: int
    matrix: np.ndarray | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    nan_counts: Mapping[str, int]
    inf_counts: Mapping[str, int]


def _as_feature_tuple(feature_cols: Sequence[str]) -> tuple[str, ...]:
    cols = tuple(str(c) for c in feature_cols)
    if not cols:
        raise FeatureValidationError("feature specification is empty")
    return cols


def validate_feature_names(
    feature_cols: Sequence[str],
    available_columns: Sequence[str] | set[str],
) -> list[str]:
    """Return error messages for missing / duplicate feature names."""

    cols = _as_feature_tuple(feature_cols)
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in cols:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        errors.append(f"duplicate feature names in specification: {duplicates}")

    available = set(available_columns)
    missing = [name for name in cols if name not in available]
    if missing:
        errors.append(f"missing required feature columns: {missing}")
    return errors


def validate_feature_matrix(
    features: pd.DataFrame | Mapping[str, Any],
    feature_cols: Sequence[str],
    *,
    allow_nan: bool = True,
    reject_inf: bool = True,
    min_rows: int = 1,
) -> FeatureValidationResult:
    """Validate and assemble an ordered numeric feature matrix.

    Parameters
    ----------
    features:
        Input table (typically S6 ``dyadic_features_full`` rows).
    feature_cols:
        Exact ordered feature names from ``feature_spec.json``.
    allow_nan:
        When True (default), NaNs are preserved for XGBoost. When False, any
        NaN is an error.
    reject_inf:
        When True (default), ``+/-Inf`` is rejected. Inf is not clipped or
        replaced.
    min_rows:
        Minimum accepted row count (default 1).
    """

    cols = _as_feature_tuple(feature_cols)
    frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    errors = validate_feature_names(cols, frame.columns)
    warnings: list[str] = []
    nan_counts: dict[str, int] = {}
    inf_counts: dict[str, int] = {}

    n_rows = int(len(frame))
    if n_rows < min_rows:
        errors.append(f"expected at least {min_rows} input row(s); got {n_rows}")

    if errors:
        return FeatureValidationResult(
            ok=False,
            feature_cols=cols,
            n_rows=n_rows,
            n_features=len(cols),
            matrix=None,
            errors=tuple(errors),
            warnings=tuple(warnings),
            nan_counts=nan_counts,
            inf_counts=inf_counts,
        )

    # Exact order subset — never reorder by name sort.
    subset = frame.loc[:, list(cols)]
    if list(subset.columns) != list(cols):
        errors.append(
            "feature column order mismatch after subsetting "
            f"(expected {list(cols)}, got {list(subset.columns)})"
        )

    for name in cols:
        series = subset[name]
        if not np.issubdtype(series.dtype, np.number):
            # Attempt safe numeric coercion for object columns that are numeric.
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.isna().all() and not series.isna().all():
                errors.append(f"feature '{name}' has non-numeric dtype {series.dtype}")
            else:
                if not np.issubdtype(series.dtype, np.number):
                    warnings.append(
                        f"feature '{name}' coerced from dtype {series.dtype} to numeric"
                    )
                subset = subset.copy()
                subset[name] = coerced

    if errors:
        return FeatureValidationResult(
            ok=False,
            feature_cols=cols,
            n_rows=n_rows,
            n_features=len(cols),
            matrix=None,
            errors=tuple(errors),
            warnings=tuple(warnings),
            nan_counts=nan_counts,
            inf_counts=inf_counts,
        )

    try:
        matrix = subset.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(
            f"Could not assemble float feature matrix: {exc}"
        ) from exc

    if matrix.shape != (n_rows, len(cols)):
        errors.append(
            f"feature matrix shape {matrix.shape} does not match "
            f"expected ({n_rows}, {len(cols)})"
        )

    for idx, name in enumerate(cols):
        col = matrix[:, idx]
        nan_counts[name] = int(np.isnan(col).sum())
        inf_counts[name] = int(np.isinf(col).sum())

    total_inf = int(sum(inf_counts.values()))
    total_nan = int(sum(nan_counts.values()))
    if reject_inf and total_inf:
        bad = {k: v for k, v in inf_counts.items() if v}
        errors.append(
            "unsupported non-finite Inf values present in features "
            f"(counts={bad}); Inf is rejected rather than replaced"
        )
    if not allow_nan and total_nan:
        bad = {k: v for k, v in nan_counts.items() if v}
        errors.append(f"NaN values are not allowed (counts={bad})")
    elif total_nan:
        warnings.append(
            f"preserved {total_nan} NaN value(s) for native model missing-value handling"
        )

    ok = not errors
    return FeatureValidationResult(
        ok=ok,
        feature_cols=cols,
        n_rows=n_rows,
        n_features=len(cols),
        matrix=matrix if ok else None,
        errors=tuple(errors),
        warnings=tuple(warnings),
        nan_counts=nan_counts,
        inf_counts=inf_counts,
    )


def require_feature_matrix(
    features: pd.DataFrame | Mapping[str, Any],
    feature_cols: Sequence[str],
    **kwargs: Any,
) -> np.ndarray:
    """Validate features and return the ordered float matrix, or raise."""

    result = validate_feature_matrix(features, feature_cols, **kwargs)
    if not result.ok or result.matrix is None:
        detail = "; ".join(result.errors) if result.errors else "validation failed"
        raise FeatureValidationError(detail)
    return result.matrix
