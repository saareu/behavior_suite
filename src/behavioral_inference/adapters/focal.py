"""Focal adapter: directional pass-through, nose-to-body routing / smoothing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.config import BehavioralInferenceConfig, resolve_working_threshold
from behavioral_inference.contracts import (
    FAMILY_FOCAL,
    GRAIN_DIRECTIONAL,
    LOGICAL_FOCAL_DETECTORS,
    NOSE_TO_BODY_SPECIALISTS,
    OCCLUSION_COLUMN,
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_PRIMARY,
    OUTPUT_ROLE_SPECIALIST,
    PREDICTIONS_SCHEMA_VERSION,
    REQUIRED_FEATURE_META_COLS,
    DetectorSelectionError,
    InferenceError,
    ThresholdConfigError,
)
from behavioral_inference.loader import LoadedModel
from behavioral_inference.predict import binarize_probabilities, score_positive_probabilities
from behavioral_inference.results import (
    DetectorRunMeta,
    append_prediction_rows,
    empty_predictions_frame,
)

STANDALONE_FOCAL_DETECTORS = ("nose_to_tail", "approach", "retract")


def prepare_directional_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Validate directional keys and return a stable-sorted copy (no collapse)."""

    missing = [c for c in REQUIRED_FEATURE_META_COLS if c not in features_df.columns]
    if missing:
        raise InferenceError(
            f"Focal inference requires metadata columns {list(REQUIRED_FEATURE_META_COLS)}; "
            f"missing {missing}"
        )
    if features_df[list(REQUIRED_FEATURE_META_COLS)].isna().any().any():
        raise InferenceError("Focal feature table has NaN values in required metadata keys")

    same_id = features_df["focal_id"] == features_df["partner_id"]
    if bool(same_id.any()):
        raise InferenceError(
            "Focal inference forbids rows where focal_id == partner_id "
            f"({int(same_id.sum())} rows)"
        )

    return features_df.sort_values(
        ["exp_name", "dense_frame_idx", "focal_id", "partner_id"],
        kind="stable",
    ).reset_index(drop=True)


def apply_nose_to_body_rolling_median(
    frame: pd.DataFrame,
    prob_column: str,
    *,
    window: int = 5,
    min_periods: int = 1,
) -> pd.Series:
    """Legacy Pipeline 90 rolling-median smoother on fused nose-to-body probs.

    Groups by ``(exp_name, focal_id, partner_id)`` after sorting by
    ``(exp_name, focal_id, partner_id, dense_frame_idx)``. Does not fill NaNs
    before smoothing. Returns a Series aligned to ``frame.index``.
    """

    if prob_column not in frame.columns:
        raise InferenceError(f"Missing probability column '{prob_column}' for smoothing")
    sort_cols = ["exp_name", "focal_id", "partner_id", "dense_frame_idx"]
    missing = [c for c in sort_cols if c not in frame.columns]
    if missing:
        raise InferenceError(f"Rolling median requires columns {missing}")

    work = frame.loc[:, sort_cols + [prob_column]].copy()
    sorted_idx = work.sort_values(sort_cols, kind="stable").index
    smoothed = (
        work.loc[sorted_idx]
        .groupby(["exp_name", "focal_id", "partner_id"], sort=False)[prob_column]
        .transform(lambda s: s.rolling(window=window, min_periods=min_periods).median())
    )
    out = pd.Series(index=frame.index, dtype=float)
    out.loc[sorted_idx] = smoothed.to_numpy(dtype=float, copy=False)
    return out


def _resolve_focal_selection(
    selected: Sequence[str] | None,
    config: BehavioralInferenceConfig,
) -> tuple[str, ...]:
    if selected is None:
        chosen = tuple(config.active_detectors_focal)
    else:
        chosen = tuple(str(x).strip() for x in selected if str(x).strip())
    unknown = [name for name in chosen if name not in LOGICAL_FOCAL_DETECTORS]
    if unknown:
        raise DetectorSelectionError(
            f"Unknown focal detector(s): {unknown}. "
            f"Supported logical names: {list(LOGICAL_FOCAL_DETECTORS)}"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for name in chosen:
        if name in seen:
            raise DetectorSelectionError(f"Duplicate focal detector selection: {name}")
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def run_focal_inference(
    features_df: pd.DataFrame,
    models: Mapping[str, LoadedModel],
    config: BehavioralInferenceConfig,
    *,
    selected_detectors: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[DetectorRunMeta, ...], list[str], dict[str, Any]]:
    """Run focal detectors on directional rows (both directions preserved)."""

    warnings: list[str] = []
    routing: dict[str, Any] = {}
    selected = _resolve_focal_selection(selected_detectors, config)
    if not selected:
        empty = empty_predictions_frame()
        return empty, empty.copy(), (), warnings, routing

    directional = prepare_directional_features(features_df)
    n = len(directional)
    unique_frames = int(directional["dense_frame_idx"].nunique())
    expected_double = unique_frames * 2
    if n != expected_double:
        warnings.append(
            f"Focal row count {n} is not exactly 2× unique frames ({expected_double}); "
            "continuing without auto-repair (Pipeline 90 soft note)"
        )

    time_sec = (
        directional["time_sec"].to_numpy(dtype=float)
        if "time_sec" in directional.columns
        else np.full(n, np.nan)
    )
    sample_idx = (
        directional["sample_idx"].to_numpy(dtype=float)
        if "sample_idx" in directional.columns
        else directional["dense_frame_idx"].to_numpy(dtype=float)
    )

    pred_chunks: list[pd.DataFrame] = []
    diag_chunks: list[pd.DataFrame] = []
    metas: list[DetectorRunMeta] = []

    for detector_name in selected:
        if detector_name == "nose_to_body":
            rows, diags, meta_list, warn, route_info = _run_nose_to_body(
                directional,
                models,
                config,
                time_sec=time_sec,
                sample_idx=sample_idx,
            )
            warnings.extend(warn)
            routing["nose_to_body"] = route_info
            pred_chunks.append(rows)
            diag_chunks.append(diags)
            metas.extend(meta_list)
            continue

        if detector_name not in STANDALONE_FOCAL_DETECTORS:
            raise DetectorSelectionError(f"Unsupported focal detector '{detector_name}'")

        model = models.get(detector_name)
        if model is None:
            raise InferenceError(
                f"Selected focal detector '{detector_name}' has no loaded model bundle. "
                "Refusing to omit it from a successful run."
            )
        threshold = resolve_working_threshold(detector_name, config)
        probs = score_positive_probabilities(directional, model)
        preds = binarize_probabilities(probs, threshold.value)
        chunk = append_prediction_rows(
            empty_predictions_frame(),
            n_rows=n,
            dense_frame_idx=directional["dense_frame_idx"].to_numpy(),
            time_sec=time_sec,
            sample_idx=sample_idx,
            prediction_grain=GRAIN_DIRECTIONAL,
            pair_id=None,
            mouse_a=None,
            mouse_b=None,
            focal_id=directional["focal_id"].to_numpy(),
            partner_id=directional["partner_id"].to_numpy(),
            family=FAMILY_FOCAL,
            detector_name=detector_name,
            specialist_name=None,
            output_role=OUTPUT_ROLE_PRIMARY,
            model_version=model.bundle.model_version,
            model_bundle_dir=str(model.bundle.bundle_dir),
            model_sha256=model.model_sha256,
            prob_raw=probs,
            prob_effective=probs,
            threshold=threshold.value,
            threshold_source=threshold.source,
            prediction=preds,
            input_valid=np.ones(n, dtype=bool),
            invalid_reason=None,
        )
        pred_chunks.append(chunk)
        metas.append(
            DetectorRunMeta(
                family=FAMILY_FOCAL,
                detector_name=detector_name,
                specialist_name=None,
                output_role=OUTPUT_ROLE_PRIMARY,
                model_version=model.bundle.model_version,
                model_bundle_dir=str(model.bundle.bundle_dir),
                model_sha256=model.model_sha256,
                threshold=threshold.value,
                threshold_source=threshold.source,
                n_rows=n,
                n_scored=n,
                n_positive=int(preds.sum()),
                routing_mode=None,
                extras={},
            )
        )

    predictions = (
        pd.concat(pred_chunks, ignore_index=True) if pred_chunks else empty_predictions_frame()
    )
    diagnostics = (
        pd.concat(diag_chunks, ignore_index=True) if diag_chunks else empty_predictions_frame()
    )
    predictions["schema_version"] = PREDICTIONS_SCHEMA_VERSION
    diagnostics["schema_version"] = PREDICTIONS_SCHEMA_VERSION
    return predictions, diagnostics, tuple(metas), warnings, routing


def _run_nose_to_body(
    directional: pd.DataFrame,
    models: Mapping[str, LoadedModel],
    config: BehavioralInferenceConfig,
    *,
    time_sec: np.ndarray,
    sample_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[DetectorRunMeta], list[str], dict[str, Any]]:
    warnings: list[str] = []
    n = len(directional)
    has_intact = "nose_body_intact" in models
    has_occ = "nose_body_occ" in models
    has_fallback = "nose_to_body" in models
    adapters = config.adapters

    if has_intact and has_occ:
        mode = "two_stage"
    elif adapters.allow_nose_to_body_single_stage_fallback and has_fallback:
        mode = "single_stage_fallback"
        warnings.append(
            "nose_to_body using documented single-stage fallback (no rolling median); "
            "specialist bundles incomplete"
        )
    else:
        missing = [s for s in NOSE_TO_BODY_SPECIALISTS if s not in models]
        raise InferenceError(
            "nose_to_body requires nose_body_intact + nose_body_occ specialists, "
            "or an explicitly allowed single-stage nose_to_body fallback bundle. "
            f"Missing specialists: {missing}."
        )

    route_info: dict[str, Any] = {"mode": mode}

    if mode == "single_stage_fallback":
        model = models["nose_to_body"]
        # Single-stage uses nested intact threshold only if flat missing — require
        # explicit scalar or fail. Legacy used _resolve_threshold on nested dict
        # without stage → default 0.5. S7 fail-closed: demand a dedicated key or
        # allow_default. Prefer intact nested value only when operator put a flat
        # nose_to_body scalar; otherwise fail unless allow_default.
        try:
            threshold = resolve_working_threshold("nose_to_body", config)
        except ThresholdConfigError:
            # Nested-only config has no scalar nose_to_body; use intact as the
            # documented single-stage threshold when fallback is selected.
            threshold = resolve_working_threshold("nose_body_intact", config)
            warnings.append(
                "single-stage nose_to_body threshold resolved from nose_to_body.intact "
                f"({threshold.value}); no scalar nose_to_body threshold configured"
            )
        probs = score_positive_probabilities(directional, model)
        preds = binarize_probabilities(probs, threshold.value)
        rows = append_prediction_rows(
            empty_predictions_frame(),
            n_rows=n,
            dense_frame_idx=directional["dense_frame_idx"].to_numpy(),
            time_sec=time_sec,
            sample_idx=sample_idx,
            prediction_grain=GRAIN_DIRECTIONAL,
            pair_id=None,
            mouse_a=None,
            mouse_b=None,
            focal_id=directional["focal_id"].to_numpy(),
            partner_id=directional["partner_id"].to_numpy(),
            family=FAMILY_FOCAL,
            detector_name="nose_to_body",
            specialist_name=None,
            output_role=OUTPUT_ROLE_PRIMARY,
            model_version=model.bundle.model_version,
            model_bundle_dir=str(model.bundle.bundle_dir),
            model_sha256=model.model_sha256,
            prob_raw=probs,
            # No smoothing on single-stage fallback (Pipeline 90).
            prob_effective=probs,
            threshold=threshold.value,
            threshold_source=threshold.source,
            prediction=preds,
            input_valid=np.ones(n, dtype=bool),
            invalid_reason=None,
        )
        meta = DetectorRunMeta(
            family=FAMILY_FOCAL,
            detector_name="nose_to_body",
            specialist_name=None,
            output_role=OUTPUT_ROLE_PRIMARY,
            model_version=model.bundle.model_version,
            model_bundle_dir=str(model.bundle.bundle_dir),
            model_sha256=model.model_sha256,
            threshold=threshold.value,
            threshold_source=threshold.source,
            n_rows=n,
            n_scored=n,
            n_positive=int(preds.sum()),
            routing_mode="single_stage_fallback",
            extras={"nose_to_body_mode": "single_stage_fallback", "smoothing": False},
        )
        route_info["smoothing"] = False
        return rows, empty_predictions_frame(), [meta], warnings, route_info

    # --- two-stage specialists ---
    if OCCLUSION_COLUMN not in directional.columns:
        raise InferenceError(
            "Two-stage nose_to_body inference requires 'is_total_occlusion' "
            "in dyadic features"
        )

    # Exact legacy masks: .eq(0) / .eq(1) without fillna.
    occlusion = directional[OCCLUSION_COLUMN]
    mask_intact = occlusion.eq(0).to_numpy()
    mask_occ = occlusion.eq(1).to_numpy()
    handled = mask_intact | mask_occ
    if not bool(handled.all()):
        n_bad = int((~handled).sum())
        warnings.append(
            f"nose_to_body routing: {n_bad} row(s) with occlusion state neither 0 nor 1 "
            "(including NaN); fused prob left at 0 with no invented specialist"
        )

    thr_intact = resolve_working_threshold("nose_body_intact", config)
    thr_occ = resolve_working_threshold("nose_body_occ", config)
    model_intact = models["nose_body_intact"]
    model_occ = models["nose_body_occ"]

    prob_fused_raw = np.zeros(n, dtype=float)
    diag_parts: list[pd.DataFrame] = []
    metas: list[DetectorRunMeta] = []

    for specialist, mask, model, thr in (
        ("nose_body_intact", mask_intact, model_intact, thr_intact),
        ("nose_body_occ", mask_occ, model_occ, thr_occ),
    ):
        probs = score_positive_probabilities(directional, model, row_mask=mask)
        # Specialist raw probs for diagnostics (0 where excluded, matching scored fill).
        # For diagnostics, store actual scored values on mask and NaN off-mask so
        # consumers can distinguish "scored 0" from "unscored".
        diag_probs = np.full(n, np.nan, dtype=float)
        if bool(mask.any()):
            diag_probs[mask] = probs[mask]
        preds = np.zeros(n, dtype=np.int8)
        if bool(mask.any()):
            preds[mask] = binarize_probabilities(probs[mask], thr.value)
        if bool(mask.any()):
            prob_fused_raw[mask] = probs[mask]

        valid = mask.astype(bool)
        invalid_reason = np.asarray(
            [None if flag else "occlusion_mask_excluded" for flag in valid],
            dtype=object,
        )
        diag_parts.append(
            append_prediction_rows(
                empty_predictions_frame(),
                n_rows=n,
                dense_frame_idx=directional["dense_frame_idx"].to_numpy(),
                time_sec=time_sec,
                sample_idx=sample_idx,
                prediction_grain=GRAIN_DIRECTIONAL,
                pair_id=None,
                mouse_a=None,
                mouse_b=None,
                focal_id=directional["focal_id"].to_numpy(),
                partner_id=directional["partner_id"].to_numpy(),
                family=FAMILY_FOCAL,
                detector_name="nose_to_body",
                specialist_name=specialist,
                output_role=OUTPUT_ROLE_SPECIALIST,
                model_version=model.bundle.model_version,
                model_bundle_dir=str(model.bundle.bundle_dir),
                model_sha256=model.model_sha256,
                prob_raw=diag_probs,
                prob_effective=diag_probs,
                threshold=thr.value,
                threshold_source=thr.source,
                prediction=preds,
                input_valid=valid,
                invalid_reason=invalid_reason,
            )
        )
        metas.append(
            DetectorRunMeta(
                family=FAMILY_FOCAL,
                detector_name="nose_to_body",
                specialist_name=specialist,
                output_role=OUTPUT_ROLE_SPECIALIST,
                model_version=model.bundle.model_version,
                model_bundle_dir=str(model.bundle.bundle_dir),
                model_sha256=model.model_sha256,
                threshold=thr.value,
                threshold_source=thr.source,
                n_rows=n,
                n_scored=int(mask.sum()),
                n_positive=int(preds[mask].sum()) if bool(mask.any()) else 0,
                routing_mode="occlusion_eq_mask",
                extras={},
            )
        )

    smooth_enabled = bool(adapters.nose_to_body_rolling_median_enabled)
    work = directional.copy()
    work["prob_nose_to_body"] = prob_fused_raw
    if smooth_enabled:
        smoothed = apply_nose_to_body_rolling_median(
            work,
            "prob_nose_to_body",
            window=adapters.nose_to_body_rolling_window,
            min_periods=adapters.nose_to_body_rolling_min_periods,
        )
        prob_effective = smoothed.to_numpy(dtype=float)
    else:
        prob_effective = prob_fused_raw.copy()
        warnings.append("nose_to_body rolling median disabled (raw_framewise / adapter flag)")

    # Re-binarize smoothed fused probs with stage-specific thresholds on original masks.
    pred_fused = np.zeros(n, dtype=np.int8)
    if bool(mask_intact.any()):
        pred_fused[mask_intact] = binarize_probabilities(
            prob_effective[mask_intact], thr_intact.value
        )
    if bool(mask_occ.any()):
        pred_fused[mask_occ] = binarize_probabilities(
            prob_effective[mask_occ], thr_occ.value
        )

    fused_threshold = np.full(n, np.nan, dtype=float)
    fused_threshold[mask_intact] = thr_intact.value
    fused_threshold[mask_occ] = thr_occ.value
    fused_source = np.full(n, None, dtype=object)
    fused_source[mask_intact] = thr_intact.source
    fused_source[mask_occ] = thr_occ.source
    fused_valid = handled.copy()
    fused_reason = np.asarray(
        [None if flag else "non_matching_occlusion_state" for flag in fused_valid],
        dtype=object,
    )

    fused_rows = append_prediction_rows(
        empty_predictions_frame(),
        n_rows=n,
        dense_frame_idx=directional["dense_frame_idx"].to_numpy(),
        time_sec=time_sec,
        sample_idx=sample_idx,
        prediction_grain=GRAIN_DIRECTIONAL,
        pair_id=None,
        mouse_a=None,
        mouse_b=None,
        focal_id=directional["focal_id"].to_numpy(),
        partner_id=directional["partner_id"].to_numpy(),
        family=FAMILY_FOCAL,
        detector_name="nose_to_body",
        specialist_name=None,
        output_role=OUTPUT_ROLE_LOGICAL_FUSED,
        model_version=model_intact.bundle.model_version,
        model_bundle_dir=None,
        model_sha256=None,
        # Fused from multiple specialists — raw is the pre-smooth stitch.
        prob_raw=prob_fused_raw,
        prob_effective=prob_effective,
        threshold=fused_threshold,
        threshold_source=fused_source,
        prediction=pred_fused,
        input_valid=fused_valid,
        invalid_reason=fused_reason,
    )
    metas.append(
        DetectorRunMeta(
            family=FAMILY_FOCAL,
            detector_name="nose_to_body",
            specialist_name=None,
            output_role=OUTPUT_ROLE_LOGICAL_FUSED,
            model_version=model_intact.bundle.model_version,
            model_bundle_dir=None,
            model_sha256=None,
            threshold=None,
            threshold_source="stage_specific",
            n_rows=n,
            n_scored=int(handled.sum()),
            n_positive=int(pred_fused.sum()),
            routing_mode="two_stage_occlusion",
            extras={
                "nose_to_body_mode": "two_stage",
                "smoothing": smooth_enabled,
                "rolling_window": adapters.nose_to_body_rolling_window
                if smooth_enabled
                else None,
                "intact_threshold": thr_intact.value,
                "occluded_threshold": thr_occ.value,
                "n_intact_mask": int(mask_intact.sum()),
                "n_occluded_mask": int(mask_occ.sum()),
                "n_unhandled_occlusion": int((~handled).sum()),
            },
        )
    )
    route_info.update(
        {
            "smoothing": smooth_enabled,
            "n_intact": int(mask_intact.sum()),
            "n_occluded": int(mask_occ.sum()),
            "n_unhandled": int((~handled).sum()),
        }
    )
    diagnostics = pd.concat(diag_parts, ignore_index=True) if diag_parts else empty_predictions_frame()
    return fused_rows, diagnostics, metas, warnings, route_info
