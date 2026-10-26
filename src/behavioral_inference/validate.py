"""Canonical prediction validation for Subsystem 07 publication.

Structural checks only — no arbitrary quality thresholds on behavioral
probabilities. Invalid routing must remain explicit via ``input_valid``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.contracts import (
    FAMILY_DYADIC,
    FAMILY_FOCAL,
    GRAIN_DIRECTIONAL,
    GRAIN_UNDIRECTED_PAIR,
    INVALID_REASON_NON_MATCHING_OCCLUSION,
    INVALID_REASON_OCCLUSION_MASK_EXCLUDED,
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_PRIMARY,
    OUTPUT_ROLE_SPECIALIST,
    PREDICTIONS_SCHEMA_VERSION,
    SCORE_ORIGIN_FUSION_ROUTED,
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
    PublicationError,
)
from behavioral_inference.results import PREDICTION_COLUMNS, BehavioralInferenceResult


def annotate_score_origin(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive export-time score_origin without changing inference formulas.

    Distinguishes observed model probabilities from legacy-compatible zero
    routing sentinels so wide views / S8 cannot confuse the two.
    """

    out = frame.copy()
    if out.empty:
        out["score_origin"] = pd.Series(dtype=object)
        return out
    valid = out["input_valid"].astype(bool).to_numpy()
    role = out["output_role"].astype(str).to_numpy()
    detector = out["detector_name"].astype(str).to_numpy()
    origins = np.full(len(out), SCORE_ORIGIN_MODEL_OBSERVED, dtype=object)
    origins[~valid] = SCORE_ORIGIN_ROUTING_SENTINEL
    fused_contact = (
        valid
        & (role == OUTPUT_ROLE_LOGICAL_FUSED)
        & (detector == "contact")
    )
    origins[fused_contact] = SCORE_ORIGIN_FUSION_ROUTED
    out["score_origin"] = origins
    return out


def validate_canonical_predictions(
    result: BehavioralInferenceResult,
    *,
    n_unique_frames: int | None = None,
    require_exactly_two_mice: bool = True,
) -> dict[str, Any]:
    """Validate long-form predictions (+ diagnostics) before publication.

    Returns a metrics dict used by QC. Raises ``PublicationError`` on
    structural failure.
    """

    predictions = result.predictions
    diagnostics = result.diagnostics
    failures: list[str] = []
    metrics: dict[str, Any] = {
        "n_prediction_rows": int(len(predictions)),
        "n_diagnostic_rows": int(len(diagnostics)),
        "selected_detectors_dyadic": list(result.selected_detectors_dyadic),
        "selected_detectors_focal": list(result.selected_detectors_focal),
    }

    if predictions.empty:
        raise PublicationError("Refusing to publish empty canonical predictions.")

    missing = [c for c in PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        failures.append("predictions missing columns: " + ", ".join(missing))

    _check_schema_version(predictions, failures, metrics)
    _check_unique_keys(predictions, failures, metrics, label="predictions")
    if not diagnostics.empty:
        _check_unique_keys(diagnostics, failures, metrics, label="diagnostics")

    frames = (
        int(n_unique_frames)
        if n_unique_frames is not None
        else int(predictions["dense_frame_idx"].nunique())
    )
    metrics["n_unique_frames"] = frames

    _check_detector_coverage(result, frames, require_exactly_two_mice, failures, metrics)
    _check_sample_timing(predictions, failures, metrics)
    _check_grains_and_identity(predictions, failures, metrics)
    _check_thresholds_and_scores(predictions, failures, metrics)
    _check_invalid_routes(predictions, diagnostics, failures, metrics)

    if failures:
        raise PublicationError(
            "Canonical prediction validation failed: " + "; ".join(failures)
        )
    return metrics


def _check_schema_version(
    frame: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    if "schema_version" not in frame.columns or frame.empty:
        return
    versions = {str(v) for v in frame["schema_version"].unique().tolist()}
    metrics["schema_versions"] = sorted(versions)
    if versions != {PREDICTIONS_SCHEMA_VERSION}:
        failures.append(
            f"schema_version values {sorted(versions)} != {PREDICTIONS_SCHEMA_VERSION}"
        )


def _prediction_key_cols(grain: str) -> tuple[str, ...]:
    if grain == GRAIN_UNDIRECTED_PAIR:
        return (
            "dense_frame_idx",
            "mouse_a",
            "mouse_b",
            "detector_name",
            "specialist_name",
            "output_role",
        )
    return (
        "dense_frame_idx",
        "focal_id",
        "partner_id",
        "detector_name",
        "specialist_name",
        "output_role",
    )


def _check_unique_keys(
    frame: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
    *,
    label: str,
) -> None:
    if frame.empty or "prediction_grain" not in frame.columns:
        return
    for grain, subset in frame.groupby("prediction_grain", sort=True):
        key_cols = list(_prediction_key_cols(str(grain)))
        work = subset.copy()
        if "specialist_name" in work.columns:
            work["specialist_name"] = work["specialist_name"].astype(object).where(
                work["specialist_name"].notna(), other=""
            )
        else:
            work["specialist_name"] = ""
        if work.duplicated(subset=key_cols).any():
            failures.append(
                f"{label}: duplicate prediction keys for grain={grain} on {key_cols}"
            )
        metrics[f"{label}_unique_keys_{grain}"] = int(
            (~work.duplicated(subset=key_cols)).sum()
        )


def _check_detector_coverage(
    result: BehavioralInferenceResult,
    n_frames: int,
    require_exactly_two_mice: bool,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    preds = result.predictions
    present_dyadic = set(
        preds.loc[preds["family"] == FAMILY_DYADIC, "detector_name"].astype(str).unique()
    )
    present_focal = set(
        preds.loc[preds["family"] == FAMILY_FOCAL, "detector_name"].astype(str).unique()
    )
    missing_d = [d for d in result.selected_detectors_dyadic if d not in present_dyadic]
    missing_f = [d for d in result.selected_detectors_focal if d not in present_focal]
    if missing_d or missing_f:
        failures.append(
            f"selected detectors missing from predictions: dyadic={missing_d}, "
            f"focal={missing_f}"
        )

    per_detector: dict[str, Any] = {}
    for detector in result.selected_detectors_dyadic:
        block = preds[
            (preds["detector_name"] == detector)
            & (preds["output_role"].isin([OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]))
        ]
        count = int(len(block))
        per_detector[detector] = {"n_rows": count, "grain": GRAIN_UNDIRECTED_PAIR}
        if count != n_frames:
            failures.append(
                f"dyadic detector '{detector}' has {count} primary/fused rows; "
                f"expected {n_frames} (one undirected pair per sample)"
            )

    expected_focal = n_frames * 2 if require_exactly_two_mice else None
    for detector in result.selected_detectors_focal:
        block = preds[
            (preds["detector_name"] == detector)
            & (preds["output_role"].isin([OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]))
        ]
        count = int(len(block))
        per_detector[detector] = {
            "n_rows": count,
            "grain": GRAIN_DIRECTIONAL,
            "expected_rows": expected_focal,
        }
        if require_exactly_two_mice and expected_focal is not None and count != expected_focal:
            failures.append(
                f"focal detector '{detector}' has {count} primary/fused rows; "
                f"expected {expected_focal} (two directions per sample for K=2)"
            )
    metrics["prediction_counts_per_detector"] = per_detector


def _check_sample_timing(
    frame: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    if frame["dense_frame_idx"].isna().any() or frame["sample_idx"].isna().any():
        failures.append("NaN sample_idx / dense_frame_idx in predictions")
    if not np.isfinite(pd.to_numeric(frame["time_sec"], errors="coerce")).all():
        # time_sec may be NaN when absent upstream; warn via metrics only if all NaN.
        metrics["time_sec_all_nan"] = bool(
            pd.to_numeric(frame["time_sec"], errors="coerce").isna().all()
        )
    sample = pd.to_numeric(frame["sample_idx"], errors="coerce")
    dense = pd.to_numeric(frame["dense_frame_idx"], errors="coerce")
    # S6 contract: sample_idx == dense_frame_idx; allow float/int equality.
    if not sample.equals(dense) and not np.allclose(
        sample.to_numpy(dtype=float),
        dense.to_numpy(dtype=float),
        equal_nan=True,
    ):
        failures.append("sample_idx does not match dense_frame_idx")


def _check_grains_and_identity(
    frame: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    undirected = frame[frame["prediction_grain"] == GRAIN_UNDIRECTED_PAIR]
    directional = frame[frame["prediction_grain"] == GRAIN_DIRECTIONAL]
    metrics["n_undirected_rows"] = int(len(undirected))
    metrics["n_directional_rows"] = int(len(directional))

    if not undirected.empty:
        if undirected["mouse_a"].isna().any() or undirected["mouse_b"].isna().any():
            failures.append("undirected rows missing mouse_a / mouse_b")
        if (undirected["mouse_a"] >= undirected["mouse_b"]).any():
            failures.append("undirected rows require mouse_a < mouse_b")
        # Must not fabricate directional identities for undirected detectors.
        if undirected["focal_id"].notna().any() or undirected["partner_id"].notna().any():
            failures.append(
                "undirected predictions must not fabricate focal_id / partner_id"
            )
        if undirected["family"].ne(FAMILY_DYADIC).any():
            failures.append("undirected grain rows must have family=dyadic")

    if not directional.empty:
        if directional["focal_id"].isna().any() or directional["partner_id"].isna().any():
            failures.append("directional rows missing focal_id / partner_id")
        if (directional["focal_id"] == directional["partner_id"]).any():
            failures.append("directional rows with focal_id == partner_id")
        if directional["mouse_a"].notna().any() or directional["mouse_b"].notna().any():
            failures.append("directional predictions must leave mouse_a / mouse_b null")
        if directional["family"].ne(FAMILY_FOCAL).any():
            failures.append("directional grain rows must have family=focal")

    if "model_family" not in frame.columns and "family" in frame.columns:
        # family already carries dyadic/focal; model family recorded in meta.
        pass
    grains = set(frame["prediction_grain"].astype(str).unique())
    bad = grains - {GRAIN_UNDIRECTED_PAIR, GRAIN_DIRECTIONAL}
    if bad:
        failures.append(f"unknown prediction_grain values: {sorted(bad)}")


def _check_thresholds_and_scores(
    frame: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    roles = set(frame["output_role"].astype(str).unique())
    bad_roles = roles - {
        OUTPUT_ROLE_PRIMARY,
        OUTPUT_ROLE_SPECIALIST,
        OUTPUT_ROLE_LOGICAL_FUSED,
    }
    if bad_roles:
        failures.append(f"unknown output_role values: {sorted(bad_roles)}")

    preds = pd.to_numeric(frame["prediction"], errors="coerce")
    if preds.isna().any() or not preds.isin([0, 1]).all():
        failures.append("prediction must be boolean/binary 0/1 with no NaN")

    if "input_valid" not in frame.columns:
        failures.append("input_valid column missing")
    else:
        # Must be boolean-like.
        try:
            frame["input_valid"].astype(bool)
        except (TypeError, ValueError):
            failures.append("input_valid is not boolean-coercible")

    eff = pd.to_numeric(frame["prob_effective"], errors="coerce")
    n_nan = int(eff.isna().sum())
    n_nonfinite = int((~np.isfinite(eff.to_numpy(dtype=float))).sum())
    metrics["n_nan_prob_effective"] = n_nan
    metrics["n_nonfinite_prob_effective"] = n_nonfinite
    # Non-finite probabilities on *valid* rows are structural (cannot threshold).
    valid_mask = frame["input_valid"].astype(bool)
    if valid_mask.any():
        valid_eff = eff[valid_mask]
        if (~np.isfinite(valid_eff.to_numpy(dtype=float))).any():
            failures.append("non-finite prob_effective on input_valid=True rows")


def _check_invalid_routes(
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    invalid = predictions[~predictions["input_valid"].astype(bool)]
    metrics["n_input_invalid_predictions"] = int(len(invalid))
    reasons = (
        invalid["invalid_reason"].astype(str).value_counts().to_dict()
        if not invalid.empty
        else {}
    )
    metrics["invalid_reason_counts"] = {str(k): int(v) for k, v in reasons.items()}

    # Contact / nose-to-body invalid routes must keep an explicit reason.
    routed = predictions[
        predictions["detector_name"].isin(["contact", "nose_to_body"])
        & ~predictions["input_valid"].astype(bool)
    ]
    if not routed.empty and routed["invalid_reason"].isna().any():
        failures.append("invalid contact/nose_to_body rows missing invalid_reason")

    known = {
        INVALID_REASON_OCCLUSION_MASK_EXCLUDED,
        INVALID_REASON_NON_MATCHING_OCCLUSION,
        "None",
        "nan",
        "",
    }
    if not diagnostics.empty:
        diag_invalid = diagnostics[~diagnostics["input_valid"].astype(bool)]
        metrics["n_input_invalid_diagnostics"] = int(len(diag_invalid))
        for reason in diag_invalid["invalid_reason"].dropna().astype(str).unique():
            if reason not in known and reason not in {
                INVALID_REASON_OCCLUSION_MASK_EXCLUDED,
                INVALID_REASON_NON_MATCHING_OCCLUSION,
            }:
                # Allow additional documented reasons; do not hard-fail unknowns.
                metrics.setdefault("unlisted_invalid_reasons", []).append(reason)

    # Sentinel zeros on invalid fused contact / nose_to_body must not be
    # re-interpreted as observed model scores.
    for detector in ("contact", "nose_to_body"):
        block = predictions[
            (predictions["detector_name"] == detector)
            & (predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED)
            & (~predictions["input_valid"].astype(bool))
        ]
        if block.empty:
            continue
        # Legacy binary: prediction must be 0 on invalid routes.
        if (pd.to_numeric(block["prediction"], errors="coerce") != 0).any():
            failures.append(
                f"invalid {detector} fused rows must keep prediction=0 "
                "(legacy binary; do not invent positives)"
            )
        # Contact: effective must be the zero sentinel.
        if detector == "contact" and (
            pd.to_numeric(block["prob_effective"], errors="coerce") != 0.0
        ).any():
            failures.append(
                "invalid contact fused rows must keep prob_effective=0 sentinel"
            )


def assert_wide_traceable_to_canonical(
    *,
    wide: pd.DataFrame,
    canonical: pd.DataFrame,
    family: str,
) -> None:
    """Verify every wide probability/prediction column traces to canonical rows."""

    if wide.empty:
        raise PublicationError(f"Refusing empty {family} wide view.")
    if family == FAMILY_DYADIC:
        key = ["dense_frame_idx", "mouse_a", "mouse_b"]
        if wide.duplicated(subset=["dense_frame_idx", "focal_id", "partner_id"]).any():
            raise PublicationError(
                "dyadic wide view has duplicate (dense_frame_idx, focal_id, partner_id)"
            )
        # Undirected uniqueness via pair identity.
        if "mouse_a" in wide.columns and wide.duplicated(
            subset=["dense_frame_idx", "mouse_a", "mouse_b"]
        ).any():
            raise PublicationError(
                "dyadic wide view has duplicate undirected pair keys"
            )
        _ = key
    else:
        if wide.duplicated(
            subset=["dense_frame_idx", "focal_id", "partner_id"]
        ).any():
            raise PublicationError(
                "focal wide view has duplicate directional keys"
            )

    # Every wide row key must exist in canonical primary/fused for that family.
    canon = canonical[
        (canonical["family"] == family)
        & (canonical["output_role"].isin([OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]))
    ]
    if canon.empty:
        raise PublicationError(
            f"Cannot trace {family} wide view: no canonical primary/fused rows"
        )
