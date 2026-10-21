"""Structural and operational QC for Subsystem 05 blob feature outputs.

Structural failures block publication of an eligible parquet. Quality warnings
are recorded without blocking completion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from blob_features.contracts import (
    BLOB_FEATURES_COLUMNS,
    BLOB_FEATURES_SCHEMA_VERSION,
    OCCLUSION_STATIC_FEATURE_COLS,
    OCCLUSION_TEMPORAL_FEATURE_COLS,
    QC_SCHEMA_VERSION,
    BlobFeaturesError,
)
from blob_features.timing import NearestNeighborTimingMap

_NUMERIC_FEATURE_COLS = (
    "cc_count",
    "cc_count_raw",
    "min_inter_contour_dist",
    *OCCLUSION_STATIC_FEATURE_COLS,
    *OCCLUSION_TEMPORAL_FEATURE_COLS,
)


@dataclass(frozen=True, slots=True)
class QcSummary:
    """Machine-readable QC summary stored in ``qc.json`` / ``run_meta.json``."""

    structural_ok: bool
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": QC_SCHEMA_VERSION,
            "structural_ok": self.structural_ok,
            "warning_count": len(self.warnings),
            "failure_count": len(self.failures),
            "warnings": list(self.warnings),
            "failures": list(self.failures),
            "metrics": dict(self.metrics),
        }


def evaluate_blob_features_qc(
    *,
    table: pd.DataFrame,
    timing_map: NearestNeighborTimingMap,
    source_frame_count: int,
    frame_errors: Mapping[str, Any] | None = None,
) -> QcSummary:
    """Evaluate structural integrity and lightweight operational warnings."""

    failures: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    _check_schema_and_row_count(table, timing_map, failures, metrics)
    _check_index_integrity(table, timing_map, source_frame_count, failures, metrics)
    _check_numeric_types(table, failures, metrics)

    if failures:
        return QcSummary(
            structural_ok=False,
            warnings=tuple(warnings),
            failures=tuple(failures),
            metrics=metrics,
        )

    _check_component_distributions(table, warnings, metrics)
    _check_feature_missingness(table, warnings, metrics)
    if frame_errors:
        metrics["frame_processing_errors"] = dict(frame_errors)
        error_count = int(frame_errors.get("count", 0))
        if error_count > 0:
            warnings.append(
                f"{error_count} prepared-video frame(s) failed during feature extraction"
            )

    return QcSummary(
        structural_ok=True,
        warnings=tuple(warnings),
        failures=(),
        metrics=metrics,
    )


def _check_schema_and_row_count(
    table: pd.DataFrame,
    timing_map: NearestNeighborTimingMap,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    expected = int(timing_map.target_sample_count)
    metrics["row_count"] = int(len(table))
    metrics["expected_row_count"] = expected
    if list(table.columns) != list(BLOB_FEATURES_COLUMNS):
        failures.append("blob_features schema columns do not match the authoritative contract")
    if len(table) != expected:
        failures.append(
            f"blob_features row count {len(table)} != expected target sample count {expected}"
        )
    if "schema_version" in table.columns and len(table) > 0:
        versions = {str(v) for v in table["schema_version"].unique().tolist()}
        metrics["schema_versions"] = sorted(versions)
        if versions != {BLOB_FEATURES_SCHEMA_VERSION}:
            failures.append(
                "schema_version values do not match "
                f"{BLOB_FEATURES_SCHEMA_VERSION}: {sorted(versions)}"
            )


def _check_index_integrity(
    table: pd.DataFrame,
    timing_map: NearestNeighborTimingMap,
    source_frame_count: int,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    if table.empty:
        return

    for alias in ("sample_idx", "dense_frame_idx", "frame_idx"):
        if alias not in table.columns:
            failures.append(f"missing index column '{alias}'")
            continue
        values = table[alias].to_numpy()
        expected = np.arange(len(table), dtype=np.int64)
        if not np.array_equal(values, expected):
            failures.append(f"'{alias}' must equal [0 .. N-1] uniquely")
        if len(np.unique(values)) != len(values):
            failures.append(f"'{alias}' values are not unique")

    if (
        "sample_idx" in table.columns
        and "dense_frame_idx" in table.columns
        and not np.array_equal(
            table["sample_idx"].to_numpy(), table["dense_frame_idx"].to_numpy()
        )
    ):
        failures.append("sample_idx and dense_frame_idx must be identical")
    if (
        "dense_frame_idx" in table.columns
        and "frame_idx" in table.columns
        and not np.array_equal(
            table["dense_frame_idx"].to_numpy(), table["frame_idx"].to_numpy()
        )
    ):
        failures.append("dense_frame_idx and frame_idx must be identical after mapping")

    if "time_sec" in table.columns:
        times = np.asarray(table["time_sec"], dtype=np.float64)
        expected_times = np.asarray(timing_map.target_time_sec, dtype=np.float64)
        if times.shape != expected_times.shape or not np.allclose(
            times, expected_times, rtol=0.0, atol=1e-12
        ):
            failures.append("time_sec does not match the authoritative 120 Hz target timeline")

    if "orig_frame_idx" in table.columns:
        orig = np.asarray(table["orig_frame_idx"], dtype=np.int64)
        metrics["orig_frame_idx_min"] = int(orig.min()) if orig.size else None
        metrics["orig_frame_idx_max"] = int(orig.max()) if orig.size else None
        if np.any(orig < 0) or np.any(orig >= source_frame_count):
            failures.append(
                "orig_frame_idx is outside prepared-video frame bounds "
                f"[0, {source_frame_count})"
            )
        if orig.size > 1 and not np.all(np.diff(orig) >= 0):
            failures.append("orig_frame_idx must be monotonically non-decreasing")
        expected_orig = np.asarray(timing_map.orig_frame_idx, dtype=np.int64)
        if orig.shape == expected_orig.shape and not np.array_equal(orig, expected_orig):
            failures.append("orig_frame_idx does not match the nearest-neighbor timing map")


def _check_numeric_types(
    table: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    non_numeric: list[str] = []
    for col in _NUMERIC_FEATURE_COLS:
        if col not in table.columns:
            continue
        series = table[col]
        if not np.issubdtype(series.dtype, np.number):
            non_numeric.append(col)
    metrics["numeric_feature_columns"] = list(_NUMERIC_FEATURE_COLS)
    if non_numeric:
        failures.append(
            "non-numeric feature columns: " + ", ".join(sorted(non_numeric))
        )


def _check_component_distributions(
    table: pd.DataFrame,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    cc = pd.to_numeric(table["cc_count"], errors="coerce")
    raw = pd.to_numeric(table["cc_count_raw"], errors="coerce")
    metrics["cc_count"] = {
        "finite_fraction": float(np.isfinite(cc.to_numpy(dtype=float)).mean()),
        "value_counts": {
            str(k): int(v)
            for k, v in cc.value_counts(dropna=False).to_dict().items()
        },
    }
    metrics["cc_count_raw"] = {
        "value_counts": {
            str(k): int(v)
            for k, v in raw.value_counts(dropna=False).to_dict().items()
        },
        "fragmented_frames": int((raw > 2).sum()),
    }
    fragmented = int((raw > 2).sum())
    if fragmented > 0:
        warnings.append(
            f"{fragmented} target row(s) map to fragmented source frames (cc_count_raw > 2)"
        )


def _check_feature_missingness(
    table: pd.DataFrame,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    missingness: dict[str, float] = {}
    for col in _NUMERIC_FEATURE_COLS:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        missingness[col] = float(1.0 - np.isfinite(values).mean()) if values.size else 1.0
    metrics["feature_missing_fraction"] = missingness

    occlusion_missing = [
        name
        for name in OCCLUSION_STATIC_FEATURE_COLS
        if missingness.get(name, 1.0) >= 0.999
    ]
    if occlusion_missing and int((pd.to_numeric(table["cc_count"], errors="coerce") == 1).sum()) == 0:
        warnings.append(
            "No merged-blob (cc_count==1) frames produced occlusion static features"
        )


def assert_structural_qc(summary: QcSummary) -> None:
    if not summary.structural_ok:
        raise BlobFeaturesError(
            "S5 structural QC failed: " + "; ".join(summary.failures)
        )
