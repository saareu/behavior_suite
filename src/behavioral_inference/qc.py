"""Structural and operational QC for Subsystem 07 behavioral inference runs.

Structural failures block publication. Ordinary warnings (routing notes,
nonblocking NaN counts on invalid rows, model-load soft messages) do not —
a completed-with-warnings run remains eligible for downstream analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.contracts import (
    FAMILY_DYADIC,
    FAMILY_FOCAL,
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_PRIMARY,
    QC_SCHEMA_VERSION,
    PublicationError,
)
from behavioral_inference.results import BehavioralInferenceResult


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


def evaluate_behavioral_inference_qc(
    *,
    result: BehavioralInferenceResult,
    predictions: pd.DataFrame,
    specialist_predictions: pd.DataFrame,
    dyadic_wide: pd.DataFrame | None,
    focal_wide: pd.DataFrame | None,
    validation_metrics: Mapping[str, Any],
    source_provenance: Mapping[str, Any] | None = None,
    model_manifest: Mapping[str, Any] | None = None,
    source_warnings: Sequence[str] = (),
) -> QcSummary:
    """Evaluate structural integrity and lightweight operational warnings."""

    failures: list[str] = []
    warnings: list[str] = list(source_warnings)
    warnings.extend(result.warnings)

    metrics: dict[str, Any] = {
        "n_unique_frames": validation_metrics.get("n_unique_frames"),
        "n_prediction_rows": int(len(predictions)),
        "n_specialist_rows": int(len(specialist_predictions)),
        "selected_detectors": {
            "dyadic": list(result.selected_detectors_dyadic),
            "focal": list(result.selected_detectors_focal),
        },
        "model_families": sorted(
            {str(m.family) for m in result.detector_meta if m.family}
        ),
        "prediction_counts_per_detector": dict(
            validation_metrics.get("prediction_counts_per_detector", {})
        ),
        "invalid_reason_counts": dict(
            validation_metrics.get("invalid_reason_counts", {})
        ),
        "n_input_invalid_predictions": validation_metrics.get(
            "n_input_invalid_predictions", 0
        ),
        "thresholds": {k: dict(v) for k, v in result.thresholds.items()},
        "routing_diagnostics": dict(result.routing_diagnostics),
    }
    if source_provenance is not None:
        metrics["source_provenance"] = dict(source_provenance)
    if model_manifest is not None:
        metrics["model_manifest_summary"] = {
            "n_bundles": len(model_manifest.get("bundles", [])),
            "registry_root": model_manifest.get("registry_root"),
            "selected_detectors": model_manifest.get("selected_detectors"),
        }

    if predictions.empty:
        failures.append("canonical predictions table is empty")

    _detector_positive_stats(predictions, metrics)
    _probability_finite_stats(predictions, warnings, metrics)
    _coverage_stats(predictions, result, metrics, failures)
    _wide_view_stats(dyadic_wide, focal_wide, result, metrics, failures)

    # Deduplicate warnings while preserving order.
    warnings = list(dict.fromkeys(str(w) for w in warnings if str(w).strip()))

    if failures:
        return QcSummary(
            structural_ok=False,
            warnings=tuple(warnings),
            failures=tuple(failures),
            metrics=metrics,
        )
    return QcSummary(
        structural_ok=True,
        warnings=tuple(warnings),
        failures=(),
        metrics=metrics,
    )


def assert_structural_qc(summary: QcSummary) -> None:
    if not summary.structural_ok:
        raise PublicationError(
            "S7 structural QC failed: " + "; ".join(summary.failures)
        )


def _detector_positive_stats(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    stats: dict[str, Any] = {}
    primary = predictions[
        predictions["output_role"].isin([OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED])
    ]
    for detector, block in primary.groupby("detector_name", sort=True):
        n = int(len(block))
        n_pos = int(pd.to_numeric(block["prediction"], errors="coerce").fillna(0).sum())
        n_invalid = int((~block["input_valid"].astype(bool)).sum())
        stats[str(detector)] = {
            "n_rows": n,
            "n_positive": n_pos,
            "positive_fraction": (float(n_pos) / float(n)) if n else 0.0,
            "n_input_invalid": n_invalid,
        }
    metrics["detector_positive_stats"] = stats


def _probability_finite_stats(
    predictions: pd.DataFrame,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    eff = pd.to_numeric(predictions["prob_effective"], errors="coerce")
    raw = pd.to_numeric(predictions["prob_raw"], errors="coerce")
    n_nan_eff = int(eff.isna().sum())
    n_nan_raw = int(raw.isna().sum())
    n_nonfinite_eff = int((~np.isfinite(eff.to_numpy(dtype=float))).sum())
    metrics["n_nan_prob_effective"] = n_nan_eff
    metrics["n_nan_prob_raw"] = n_nan_raw
    metrics["n_nonfinite_prob_effective"] = n_nonfinite_eff
    # Contact fused uses NaN prob_raw by design — do not warn on that alone.
    fused_contact = predictions[
        (predictions["detector_name"] == "contact")
        & (predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED)
    ]
    expected_nan_raw = int(len(fused_contact))
    unexpected_nan_raw = n_nan_raw - expected_nan_raw
    if unexpected_nan_raw > 0:
        warnings.append(
            f"{unexpected_nan_raw} non-contact rows have NaN prob_raw"
        )
    valid = predictions["input_valid"].astype(bool)
    if valid.any():
        valid_eff = eff[valid]
        if (~np.isfinite(valid_eff.to_numpy(dtype=float))).any():
            # Structural — also caught by validate; keep as metric.
            metrics["valid_rows_nonfinite_prob_effective"] = int(
                (~np.isfinite(valid_eff.to_numpy(dtype=float))).sum()
            )


def _coverage_stats(
    predictions: pd.DataFrame,
    result: BehavioralInferenceResult,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    frames = (
        set(pd.to_numeric(predictions["dense_frame_idx"], errors="coerce").dropna().astype(int))
    )
    metrics["frame_coverage_n"] = len(frames)
    metrics["detector_coverage"] = {
        "dyadic": list(result.selected_detectors_dyadic),
        "focal": list(result.selected_detectors_focal),
        "note": (
            "Subset publication is allowed; S8 must check whether its required "
            "detectors are present in detector_coverage."
        ),
    }
    dyadic = predictions[predictions["family"] == FAMILY_DYADIC]
    focal = predictions[predictions["family"] == FAMILY_FOCAL]
    if not dyadic.empty and dyadic["prediction_grain"].nunique() != 1:
        failures.append("dyadic predictions have mixed prediction_grain")
    if not focal.empty and focal["prediction_grain"].nunique() != 1:
        failures.append("focal predictions have mixed prediction_grain")


def _wide_view_stats(
    dyadic_wide: pd.DataFrame | None,
    focal_wide: pd.DataFrame | None,
    result: BehavioralInferenceResult,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    metrics["dyadic_wide_rows"] = (
        int(len(dyadic_wide)) if dyadic_wide is not None else None
    )
    metrics["focal_wide_rows"] = int(len(focal_wide)) if focal_wide is not None else None
    if result.selected_detectors_dyadic:
        if dyadic_wide is None or dyadic_wide.empty:
            failures.append("dyadic detectors selected but dyadic wide view is empty")
        elif dyadic_wide.duplicated(
            subset=["dense_frame_idx", "mouse_a", "mouse_b"]
        ).any():
            failures.append("dyadic wide view has duplicate undirected keys")
    if result.selected_detectors_focal:
        if focal_wide is None or focal_wide.empty:
            failures.append("focal detectors selected but focal wide view is empty")
        elif focal_wide.duplicated(
            subset=["dense_frame_idx", "focal_id", "partner_id"]
        ).any():
            failures.append("focal wide view has duplicate directional keys")
