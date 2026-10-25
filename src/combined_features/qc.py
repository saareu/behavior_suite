"""Structural and operational QC for Subsystem 06 combined-feature outputs.

Structural failures block publication of an eligible parquet. Ordinary
scientific missingness and nonblocking warnings do not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.contracts import (
    BLOB_FEATURE_COLS,
    DYADIC_FEATURES_SCHEMA_VERSION,
    IMPLANTATION_RESOLVED,
    MANDATORY_BLOB_FEATURE_COLS,
    QC_SCHEMA_VERSION,
    REQUIRED_IDENTITY_COLUMNS,
    CombinedFeaturesError,
)

_MAJOR_FEATURE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "identity": REQUIRED_IDENTITY_COLUMNS,
    "blob": BLOB_FEATURE_COLS,
    "quality_flags": (
        "nose_any_imputed_i",
        "nose_any_imputed_j",
        "nose_i_is_kf",
        "nose_j_is_kf",
        "is_total_occlusion",
        "is_implanted",
    ),
    "geometry": (
        "distance_nose_i_nose_j_norm",
        "dyad_body_scale",
        "speed_i",
        "speed_j",
    ),
}


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


def evaluate_combined_features_qc(
    *,
    table: pd.DataFrame,
    n_tracks: int,
    n_samples: int,
    tracks: Sequence[int],
    nodes: Sequence[str],
    implantation_status: str | None,
    implantation_reason: str | None = None,
    implanted_track_id: int | None = None,
    alignment: Mapping[str, Any] | None = None,
    source_warnings: Sequence[str] = (),
) -> QcSummary:
    """Evaluate structural integrity and lightweight operational warnings."""

    failures: list[str] = []
    warnings: list[str] = list(source_warnings)
    metrics: dict[str, Any] = {
        "n_tracks": int(n_tracks),
        "n_samples": int(n_samples),
        "tracks": [int(t) for t in tracks],
        "nodes": [str(n) for n in nodes],
        "implantation_status": implantation_status,
        "implanted_track_id": implanted_track_id,
    }
    if alignment is not None:
        metrics["alignment"] = dict(alignment)

    _check_identity_schema(table, failures, metrics)
    _check_directional_cardinality(table, n_tracks, n_samples, failures, metrics)
    _check_index_and_timestamps(table, failures, metrics)
    _check_mandatory_blob_presence(table, failures, metrics)
    _check_implantation(table, implantation_status, implantation_reason, warnings, metrics)

    if failures:
        return QcSummary(
            structural_ok=False,
            warnings=tuple(warnings),
            failures=tuple(failures),
            metrics=metrics,
        )

    _check_feature_missingness(table, warnings, metrics)
    _check_nan_inf_counts(table, warnings, metrics)

    if implantation_status and implantation_status != IMPLANTATION_RESOLVED:
        warnings.append(
            f"implantation_status={implantation_status}; is_implanted uses NaN for "
            "unresolved identity (not encoded as 0)."
        )
        if implantation_reason:
            warnings.append(str(implantation_reason))

    return QcSummary(
        structural_ok=True,
        warnings=tuple(dict.fromkeys(warnings)),
        failures=(),
        metrics=metrics,
    )


def assert_structural_qc(summary: QcSummary) -> None:
    if not summary.structural_ok:
        raise CombinedFeaturesError(
            "S6 structural QC failed: " + "; ".join(summary.failures)
        )


def _check_identity_schema(
    table: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    metrics["row_count"] = int(len(table))
    metrics["n_feature_columns"] = int(len(table.columns))
    missing = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in table.columns]
    if missing:
        failures.append(
            "missing required identity columns: " + ", ".join(missing)
        )
    if list(table.columns[: len(REQUIRED_IDENTITY_COLUMNS)]) != list(
        REQUIRED_IDENTITY_COLUMNS
    ):
        failures.append("required identity columns are not leading / ordered")
    if "schema_version" in table.columns and len(table) > 0:
        versions = {str(v) for v in table["schema_version"].unique().tolist()}
        metrics["schema_versions"] = sorted(versions)
        if versions != {DYADIC_FEATURES_SCHEMA_VERSION}:
            failures.append(
                "schema_version values do not match "
                f"{DYADIC_FEATURES_SCHEMA_VERSION}: {sorted(versions)}"
            )


def _check_directional_cardinality(
    table: pd.DataFrame,
    n_tracks: int,
    n_samples: int,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    expected_per_sample = int(n_tracks * max(n_tracks - 1, 0))
    expected_rows = int(n_samples * expected_per_sample)
    metrics["expected_rows"] = expected_rows
    metrics["expected_directional_per_sample"] = expected_per_sample
    if len(table) != expected_rows:
        failures.append(
            f"row count {len(table)} != expected "
            f"N_samples * K * (K-1) = {expected_rows}"
        )
    required = ("focal_id", "partner_id", "dense_frame_idx")
    if all(c in table.columns for c in required):
        if table.duplicated(subset=list(required)).any():
            failures.append(
                "duplicate (focal_id, partner_id, dense_frame_idx) directional keys"
            )
        if (table["focal_id"] == table["partner_id"]).any():
            failures.append("rows with focal_id == partner_id are present")
        per_sample = (
            table.groupby("dense_frame_idx", sort=True)
            .size()
            .to_numpy(dtype=np.int64)
        )
        metrics["directional_rows_per_sample_unique"] = sorted(
            {int(v) for v in per_sample.tolist()}
        )
        if per_sample.size and not np.all(per_sample == expected_per_sample):
            failures.append(
                "not every sample has exactly K*(K-1) directional rows "
                f"(expected {expected_per_sample})"
            )


def _check_index_and_timestamps(
    table: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    if table.empty:
        return
    if (
        "sample_idx" in table.columns
        and "dense_frame_idx" in table.columns
        and not np.array_equal(
            table["sample_idx"].to_numpy(), table["dense_frame_idx"].to_numpy()
        )
    ):
        failures.append("sample_idx and dense_frame_idx must be identical")
    if "time_sec" in table.columns:
        times = np.asarray(table["time_sec"], dtype=np.float64)
        if not np.isfinite(times).all():
            failures.append("time_sec contains non-finite values")
        metrics["time_sec_min"] = float(np.nanmin(times)) if times.size else None
        metrics["time_sec_max"] = float(np.nanmax(times)) if times.size else None
    if "orig_frame_idx" in table.columns:
        orig = np.asarray(table["orig_frame_idx"], dtype=np.int64)
        metrics["orig_frame_idx_min"] = int(orig.min()) if orig.size else None
        metrics["orig_frame_idx_max"] = int(orig.max()) if orig.size else None


def _check_mandatory_blob_presence(
    table: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    missing = [c for c in MANDATORY_BLOB_FEATURE_COLS if c not in table.columns]
    metrics["mandatory_blob_columns_present"] = missing == []
    if missing:
        failures.append(
            "missing mandatory S5 blob measurement columns: " + ", ".join(missing)
        )


def _check_implantation(
    table: pd.DataFrame,
    implantation_status: str | None,
    implantation_reason: str | None,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    metrics["implantation_reason"] = implantation_reason
    if "is_implanted" not in table.columns:
        warnings.append("is_implanted column is absent from the feature table")
        return
    values = pd.to_numeric(table["is_implanted"], errors="coerce").to_numpy(dtype=float)
    metrics["is_implanted"] = {
        "nan_fraction": float(np.isnan(values).mean()) if values.size else 1.0,
        "unique_finite": sorted(
            {float(v) for v in values[np.isfinite(values)].tolist()}
        ),
    }
    if implantation_status == IMPLANTATION_RESOLVED:
        if np.isnan(values).any():
            warnings.append(
                "implantation marked resolved but is_implanted contains NaN rows"
            )
    elif (
        values.size
        and np.isfinite(values).any()
        and implantation_status
        and implantation_status != IMPLANTATION_RESOLVED
    ):
        # Unresolved should be all-NaN; any finite encoding is a policy violation.
        finite = values[np.isfinite(values)]
        if finite.size and set(np.unique(finite).tolist()) <= {0.0}:
            warnings.append(
                "unresolved implantation appears encoded as 0; prefer NaN"
            )


def _check_feature_missingness(
    table: pd.DataFrame,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    family_missing: dict[str, dict[str, float]] = {}
    for family, cols in _MAJOR_FEATURE_FAMILIES.items():
        present = [c for c in cols if c in table.columns]
        missingness: dict[str, float] = {}
        for col in present:
            series = table[col]
            if np.issubdtype(series.dtype, np.number):
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                missingness[col] = (
                    float(1.0 - np.isfinite(values).mean()) if values.size else 1.0
                )
            else:
                missingness[col] = float(series.isna().mean()) if len(series) else 1.0
        family_missing[family] = missingness
        absent = [c for c in cols if c not in table.columns]
        if absent and family != "identity":
            warnings.append(
                f"{family} feature family missing columns: " + ", ".join(absent[:8])
            )
    metrics["feature_family_missing_fraction"] = family_missing


def _check_nan_inf_counts(
    table: pd.DataFrame,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    numeric_cols = [
        c for c in table.columns if np.issubdtype(table[c].dtype, np.number)
    ]
    nan_total = 0
    inf_total = 0
    for col in numeric_cols:
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        nan_total += int(np.isnan(values).sum())
        inf_total += int(np.isinf(values).sum())
    metrics["numeric_column_count"] = len(numeric_cols)
    metrics["nan_value_count"] = nan_total
    metrics["inf_value_count"] = inf_total
    if inf_total > 0:
        warnings.append(f"{inf_total} Inf value(s) present in numeric feature columns")
