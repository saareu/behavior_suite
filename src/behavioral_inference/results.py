"""In-memory behavioral inference result contract (Sprint 2 science; Sprint 3 publishes)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.contracts import (
    INFERENCE_RESULT_SCHEMA_VERSION,
    PREDICTIONS_SCHEMA_VERSION,
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "sample_idx",
    "dense_frame_idx",
    "time_sec",
    "prediction_grain",
    "pair_id",
    "mouse_a",
    "mouse_b",
    "focal_id",
    "partner_id",
    "family",
    "detector_name",
    "specialist_name",
    "output_role",
    "model_version",
    "model_bundle_dir",
    "model_sha256",
    "prob_raw",
    "prob_effective",
    "threshold",
    "threshold_source",
    "prediction",
    "input_valid",
    "invalid_reason",
)


@dataclass(frozen=True, slots=True)
class DetectorRunMeta:
    """Per-detector / specialist execution metadata recorded on the result."""

    family: str
    detector_name: str
    specialist_name: str | None
    output_role: str
    model_version: str | None
    model_bundle_dir: str | None
    model_sha256: str | None
    threshold: float | None
    threshold_source: str | None
    n_rows: int
    n_scored: int
    n_positive: int
    routing_mode: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "detector_name": self.detector_name,
            "specialist_name": self.specialist_name,
            "output_role": self.output_role,
            "model_version": self.model_version,
            "model_bundle_dir": self.model_bundle_dir,
            "model_sha256": self.model_sha256,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "n_rows": self.n_rows,
            "n_scored": self.n_scored,
            "n_positive": self.n_positive,
            "routing_mode": self.routing_mode,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True, slots=True)
class BehavioralInferenceResult:
    """Headless Sprint 2 computation result (canonical long-form + diagnostics)."""

    schema_version: str
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    selected_detectors_dyadic: tuple[str, ...]
    selected_detectors_focal: tuple[str, ...]
    detector_meta: tuple[DetectorRunMeta, ...]
    thresholds: Mapping[str, Mapping[str, Any]]
    model_identities: Mapping[str, Mapping[str, Any]]
    warnings: tuple[str, ...]
    routing_diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_prediction_rows": int(len(self.predictions)),
            "n_diagnostic_rows": int(len(self.diagnostics)),
            "selected_detectors": {
                "dyadic": list(self.selected_detectors_dyadic),
                "focal": list(self.selected_detectors_focal),
            },
            "thresholds": {k: dict(v) for k, v in self.thresholds.items()},
            "model_identities": {k: dict(v) for k, v in self.model_identities.items()},
            "warnings": list(self.warnings),
            "routing_diagnostics": dict(self.routing_diagnostics),
            "metadata": dict(self.metadata),
            "detector_meta": [m.to_dict() for m in self.detector_meta],
        }


def empty_predictions_frame() -> pd.DataFrame:
    """Empty long-form predictions table with the Sprint 2 column contract."""

    return pd.DataFrame({col: pd.Series(dtype=object) for col in PREDICTION_COLUMNS})


def make_pair_id(mouse_a: int, mouse_b: int) -> str:
    lo, hi = (mouse_a, mouse_b) if mouse_a <= mouse_b else (mouse_b, mouse_a)
    return f"{lo}|{hi}"


def append_prediction_rows(
    base: pd.DataFrame,
    *,
    n_rows: int,
    dense_frame_idx: Sequence[Any] | np.ndarray,
    time_sec: Sequence[Any] | np.ndarray,
    sample_idx: Sequence[Any] | np.ndarray,
    prediction_grain: str,
    pair_id: Sequence[Any] | np.ndarray | None,
    mouse_a: Sequence[Any] | np.ndarray | None,
    mouse_b: Sequence[Any] | np.ndarray | None,
    focal_id: Sequence[Any] | np.ndarray | None,
    partner_id: Sequence[Any] | np.ndarray | None,
    family: str,
    detector_name: str,
    specialist_name: str | None,
    output_role: str,
    model_version: str | None,
    model_bundle_dir: str | None,
    model_sha256: str | None,
    prob_raw: Sequence[Any] | np.ndarray,
    prob_effective: Sequence[Any] | np.ndarray,
    threshold: float | Sequence[Any] | np.ndarray,
    threshold_source: str | Sequence[Any] | np.ndarray,
    prediction: Sequence[Any] | np.ndarray,
    input_valid: Sequence[Any] | np.ndarray,
    invalid_reason: Sequence[Any] | np.ndarray | None,
) -> pd.DataFrame:
    """Build one detector block of long-form prediction rows."""

    def _as_series(values: Sequence[Any] | np.ndarray | None, fill: Any = pd.NA) -> pd.Series:
        if values is None:
            return pd.Series([fill] * n_rows)
        return pd.Series(list(values) if not isinstance(values, np.ndarray) else values)

    def _scalar_or_series(values: float | str | Sequence[Any] | np.ndarray) -> pd.Series:
        if isinstance(values, (float, int, str, np.floating, np.integer)):
            return pd.Series([values] * n_rows)
        return pd.Series(list(values) if not isinstance(values, np.ndarray) else values)

    frame = pd.DataFrame(
        {
            "schema_version": [PREDICTIONS_SCHEMA_VERSION] * n_rows,
            "sample_idx": _as_series(sample_idx),
            "dense_frame_idx": _as_series(dense_frame_idx),
            "time_sec": _as_series(time_sec),
            "prediction_grain": [prediction_grain] * n_rows,
            "pair_id": _as_series(pair_id),
            "mouse_a": _as_series(mouse_a),
            "mouse_b": _as_series(mouse_b),
            "focal_id": _as_series(focal_id),
            "partner_id": _as_series(partner_id),
            "family": [family] * n_rows,
            "detector_name": [detector_name] * n_rows,
            "specialist_name": [specialist_name] * n_rows,
            "output_role": [output_role] * n_rows,
            "model_version": [model_version] * n_rows,
            "model_bundle_dir": [model_bundle_dir] * n_rows,
            "model_sha256": [model_sha256] * n_rows,
            "prob_raw": _as_series(prob_raw),
            "prob_effective": _as_series(prob_effective),
            "threshold": _scalar_or_series(threshold),
            "threshold_source": _scalar_or_series(threshold_source),
            "prediction": _as_series(prediction),
            "input_valid": _as_series(input_valid),
            "invalid_reason": _as_series(invalid_reason),
        }
    )
    if len(base) == 0:
        return frame.loc[:, list(PREDICTION_COLUMNS)]
    return pd.concat([base, frame], ignore_index=True).loc[:, list(PREDICTION_COLUMNS)]


def build_inference_result(
    *,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_detectors_dyadic: Sequence[str],
    selected_detectors_focal: Sequence[str],
    detector_meta: Sequence[DetectorRunMeta],
    warnings: Sequence[str],
    routing_diagnostics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> BehavioralInferenceResult:
    thresholds: dict[str, dict[str, Any]] = {}
    model_identities: dict[str, dict[str, Any]] = {}
    for meta in detector_meta:
        key = meta.specialist_name or meta.detector_name
        if meta.threshold is not None:
            thresholds[key] = {
                "value": meta.threshold,
                "source": meta.threshold_source,
                "detector_name": meta.detector_name,
                "specialist_name": meta.specialist_name,
                "output_role": meta.output_role,
            }
        if meta.model_sha256 is not None or meta.model_bundle_dir is not None:
            model_identities[key] = {
                "family": meta.family,
                "detector_name": meta.detector_name,
                "specialist_name": meta.specialist_name,
                "model_version": meta.model_version,
                "model_bundle_dir": meta.model_bundle_dir,
                "model_sha256": meta.model_sha256,
            }

    return BehavioralInferenceResult(
        schema_version=INFERENCE_RESULT_SCHEMA_VERSION,
        predictions=predictions.reset_index(drop=True),
        diagnostics=diagnostics.reset_index(drop=True),
        selected_detectors_dyadic=tuple(selected_detectors_dyadic),
        selected_detectors_focal=tuple(selected_detectors_focal),
        detector_meta=tuple(detector_meta),
        thresholds=thresholds,
        model_identities=model_identities,
        warnings=tuple(warnings),
        routing_diagnostics=dict(routing_diagnostics),
        metadata=dict(metadata),
    )
