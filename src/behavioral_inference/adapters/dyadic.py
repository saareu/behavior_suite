"""Dyadic adapter: unordered-pair collapse, contact routing, standalone detectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.config import BehavioralInferenceConfig, resolve_working_threshold
from behavioral_inference.contracts import (
    CONTACT_SPECIALISTS,
    FAMILY_DYADIC,
    GRAIN_UNDIRECTED_PAIR,
    LOGICAL_DYADIC_DETECTORS,
    OCCLUSION_COLUMN,
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_PRIMARY,
    OUTPUT_ROLE_SPECIALIST,
    PREDICTIONS_SCHEMA_VERSION,
    REQUIRED_FEATURE_META_COLS,
    STANDALONE_DYADIC_DETECTORS,
    DetectorSelectionError,
    InferenceError,
)
from behavioral_inference.loader import LoadedModel
from behavioral_inference.predict import binarize_probabilities, score_positive_probabilities
from behavioral_inference.results import (
    DetectorRunMeta,
    append_prediction_rows,
    empty_predictions_frame,
    make_pair_id,
)


def collapse_to_unordered_pairs(features_df: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical undirected-pair row per sample (S7-B §1.2).

    Stable-sort by ``(exp_name, dense_frame_idx, focal_id, partner_id)``, derive
    ``mouse_a=min``, ``mouse_b=max``, and keep the unique row where
    ``focal_id == mouse_a`` and ``partner_id == mouse_b`` (lower ID as focal).
    Does **not** average directional feature vectors.
    """

    missing = [c for c in REQUIRED_FEATURE_META_COLS if c not in features_df.columns]
    if missing:
        raise InferenceError(
            f"Dyadic collapse requires metadata columns {list(REQUIRED_FEATURE_META_COLS)}; "
            f"missing {missing}"
        )
    if features_df["dense_frame_idx"].isna().any():
        raise InferenceError("Found NaN dense_frame_idx in dyadic feature table")

    work = features_df.sort_values(
        ["exp_name", "dense_frame_idx", "focal_id", "partner_id"],
        kind="stable",
    ).copy()
    mouse_a = work[["focal_id", "partner_id"]].min(axis=1)
    mouse_b = work[["focal_id", "partner_id"]].max(axis=1)
    work = work.assign(mouse_a=mouse_a, mouse_b=mouse_b)

    # Canonical selection: lower ID as focal / higher as partner.
    canonical = work[
        (work["focal_id"] == work["mouse_a"]) & (work["partner_id"] == work["mouse_b"])
    ].copy()

    # Parity reference: sort + drop_duplicates(keep='first') must match canonical.
    legacy_like = work.drop_duplicates(
        subset=["exp_name", "dense_frame_idx", "mouse_a", "mouse_b"],
        keep="first",
    )
    if len(canonical) != len(legacy_like):
        raise InferenceError(
            "Canonical dyadic collapse diverged from Pipeline 70 sort+keep-first "
            f"cardinality ({len(canonical)} vs {len(legacy_like)}). "
            "Investigate before changing the formula."
        )

    dup_keys = canonical.duplicated(
        subset=["exp_name", "dense_frame_idx", "mouse_a", "mouse_b"],
        keep=False,
    )
    if bool(dup_keys.any()):
        raise InferenceError(
            "Canonical dyadic collapse produced duplicate unordered-pair keys; "
            "refusing to invent an aggregation policy"
        )

    return canonical.reset_index(drop=True)


def _contact_occlusion_masks(
    occlusion: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy Pipeline 70 contact masks: NaN→0 coercion before eq(0)/eq(1)."""

    coerced = (
        pd.to_numeric(occlusion, errors="coerce").fillna(0).astype(int)
    )
    intact = coerced.eq(0).to_numpy()
    occluded = coerced.eq(1).to_numpy()
    return intact, occluded


def _resolve_dyadic_selection(
    selected: Sequence[str] | None,
    config: BehavioralInferenceConfig,
) -> tuple[str, ...]:
    if selected is None:
        chosen = tuple(config.active_detectors_dyadic)
    else:
        chosen = tuple(str(x).strip() for x in selected if str(x).strip())
    unknown = [name for name in chosen if name not in LOGICAL_DYADIC_DETECTORS]
    if unknown:
        raise DetectorSelectionError(
            f"Unknown dyadic detector(s): {unknown}. "
            f"Supported logical names: {list(LOGICAL_DYADIC_DETECTORS)}"
        )
    if not chosen:
        return ()
    # Preserve caller order; reject duplicates that would silently double-score.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in chosen:
        if name in seen:
            raise DetectorSelectionError(f"Duplicate dyadic detector selection: {name}")
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def run_dyadic_inference(
    features_df: pd.DataFrame,
    models: Mapping[str, LoadedModel],
    config: BehavioralInferenceConfig,
    *,
    selected_detectors: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[DetectorRunMeta, ...], list[str]]:
    """Run dyadic detectors on canonical unordered-pair rows.

    Returns ``(predictions, diagnostics, detector_meta, warnings)``.
    """

    warnings: list[str] = []
    selected = _resolve_dyadic_selection(selected_detectors, config)
    if not selected:
        empty = empty_predictions_frame()
        return empty, empty.copy(), (), warnings

    collapsed = collapse_to_unordered_pairs(features_df)
    n = len(collapsed)
    pair_ids = [
        make_pair_id(int(a), int(b))
        for a, b in zip(collapsed["mouse_a"], collapsed["mouse_b"], strict=True)
    ]
    time_sec = (
        collapsed["time_sec"].to_numpy(dtype=float)
        if "time_sec" in collapsed.columns
        else np.full(n, np.nan)
    )
    sample_idx = (
        collapsed["sample_idx"].to_numpy(dtype=float)
        if "sample_idx" in collapsed.columns
        else collapsed["dense_frame_idx"].to_numpy(dtype=float)
    )

    pred_chunks: list[pd.DataFrame] = []
    diag_chunks: list[pd.DataFrame] = []
    metas: list[DetectorRunMeta] = []

    for detector_name in selected:
        if detector_name == "contact":
            rows, diags, meta_list, warn = _run_contact(
                collapsed,
                models,
                config,
                pair_ids=pair_ids,
                time_sec=time_sec,
                sample_idx=sample_idx,
            )
            warnings.extend(warn)
            pred_chunks.append(rows)
            diag_chunks.append(diags)
            metas.extend(meta_list)
            continue

        if detector_name not in STANDALONE_DYADIC_DETECTORS:
            raise DetectorSelectionError(f"Unsupported dyadic detector '{detector_name}'")

        model = models.get(detector_name)
        if model is None:
            raise InferenceError(
                f"Selected dyadic detector '{detector_name}' has no loaded model bundle. "
                "Refusing to omit it from a successful run."
            )
        threshold = resolve_working_threshold(detector_name, config)
        probs = score_positive_probabilities(collapsed, model)
        preds = binarize_probabilities(probs, threshold.value)
        chunk = append_prediction_rows(
            empty_predictions_frame(),
            n_rows=n,
            dense_frame_idx=collapsed["dense_frame_idx"].to_numpy(),
            time_sec=time_sec,
            sample_idx=sample_idx,
            prediction_grain=GRAIN_UNDIRECTED_PAIR,
            pair_id=pair_ids,
            mouse_a=collapsed["mouse_a"].to_numpy(),
            mouse_b=collapsed["mouse_b"].to_numpy(),
            focal_id=None,
            partner_id=None,
            family=FAMILY_DYADIC,
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
                family=FAMILY_DYADIC,
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
    return predictions, diagnostics, tuple(metas), warnings


def _run_contact(
    collapsed: pd.DataFrame,
    models: Mapping[str, LoadedModel],
    config: BehavioralInferenceConfig,
    *,
    pair_ids: list[str],
    time_sec: np.ndarray,
    sample_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[DetectorRunMeta], list[str]]:
    warnings: list[str] = []
    if OCCLUSION_COLUMN not in collapsed.columns:
        raise InferenceError(
            "contact inference requires is_total_occlusion in dyadic features"
        )
    for specialist in CONTACT_SPECIALISTS:
        if specialist not in models:
            raise InferenceError(
                f"Logical detector 'contact' requires specialist bundle '{specialist}'. "
                "Refusing obsolete standalone contact and refusing silent omission."
            )

    intact_mask, occ_mask = _contact_occlusion_masks(collapsed[OCCLUSION_COLUMN])
    # Rows that are neither intact nor occluded after coercion (e.g. value 2).
    handled = intact_mask | occ_mask
    if not bool(handled.all()):
        n_unhandled = int((~handled).sum())
        warnings.append(
            f"contact routing: {n_unhandled} row(s) have occlusion state neither 0 nor 1 "
            "after NaN→0 coercion; specialists forced to 0 and fused contact stays 0 "
            "(no invented prediction)"
        )

    n = len(collapsed)
    diag_parts: list[pd.DataFrame] = []
    metas: list[DetectorRunMeta] = []
    specialist_probs: dict[str, np.ndarray] = {}
    specialist_preds: dict[str, np.ndarray] = {}
    specialist_valid: dict[str, np.ndarray] = {}
    specialist_thresholds: dict[str, Any] = {}

    for specialist, mask in (
        ("contact_intact", intact_mask),
        ("contact_occluded", occ_mask),
    ):
        model = models[specialist]
        threshold = resolve_working_threshold(specialist, config)
        probs = score_positive_probabilities(collapsed, model, row_mask=mask)
        preds = binarize_probabilities(probs, threshold.value)
        valid = mask.astype(bool)
        invalid_reason = np.asarray(
            [None if flag else "occlusion_mask_excluded" for flag in valid],
            dtype=object,
        )
        specialist_probs[specialist] = probs
        specialist_preds[specialist] = preds
        specialist_valid[specialist] = valid
        specialist_thresholds[specialist] = threshold
        diag_parts.append(
            append_prediction_rows(
                empty_predictions_frame(),
                n_rows=n,
                dense_frame_idx=collapsed["dense_frame_idx"].to_numpy(),
                time_sec=time_sec,
                sample_idx=sample_idx,
                prediction_grain=GRAIN_UNDIRECTED_PAIR,
                pair_id=pair_ids,
                mouse_a=collapsed["mouse_a"].to_numpy(),
                mouse_b=collapsed["mouse_b"].to_numpy(),
                focal_id=None,
                partner_id=None,
                family=FAMILY_DYADIC,
                detector_name="contact",
                specialist_name=specialist,
                output_role=OUTPUT_ROLE_SPECIALIST,
                model_version=model.bundle.model_version,
                model_bundle_dir=str(model.bundle.bundle_dir),
                model_sha256=model.model_sha256,
                prob_raw=probs,
                prob_effective=probs,
                threshold=threshold.value,
                threshold_source=threshold.source,
                prediction=preds,
                input_valid=valid,
                invalid_reason=invalid_reason,
            )
        )
        metas.append(
            DetectorRunMeta(
                family=FAMILY_DYADIC,
                detector_name="contact",
                specialist_name=specialist,
                output_role=OUTPUT_ROLE_SPECIALIST,
                model_version=model.bundle.model_version,
                model_bundle_dir=str(model.bundle.bundle_dir),
                model_sha256=model.model_sha256,
                threshold=threshold.value,
                threshold_source=threshold.source,
                n_rows=n,
                n_scored=int(mask.sum()),
                n_positive=int(preds[mask].sum()) if bool(mask.any()) else 0,
                routing_mode="occlusion_gated",
                extras={
                    "n_intact_mask": int(intact_mask.sum()),
                    "n_occluded_mask": int(occ_mask.sum()),
                },
            )
        )

    # Fuse: start from intact, overwrite where occluded (Pipeline 70).
    prob_fused = pd.to_numeric(
        pd.Series(specialist_probs["contact_intact"]), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    pred_fused = pd.to_numeric(
        pd.Series(specialist_preds["contact_intact"]), errors="coerce"
    ).fillna(0).astype(int).to_numpy()
    if bool(occ_mask.any()):
        prob_fused[occ_mask] = specialist_probs["contact_occluded"][occ_mask]
        pred_fused[occ_mask] = specialist_preds["contact_occluded"][occ_mask]

    # Unhandled occlusion states: no invented specialist; fused remains 0 from fill.
    unhandled = ~handled
    if bool(unhandled.any()):
        prob_fused[unhandled] = 0.0
        pred_fused[unhandled] = 0

    # Fused threshold reported as occluded when routed, else intact (diagnostic).
    thr_intact = specialist_thresholds["contact_intact"]
    thr_occ = specialist_thresholds["contact_occluded"]
    fused_threshold = np.full(n, thr_intact.value, dtype=float)
    fused_threshold[occ_mask] = thr_occ.value
    fused_source = np.full(n, thr_intact.source, dtype=object)
    fused_source[occ_mask] = thr_occ.source
    fused_valid = handled.copy()
    fused_reason = np.asarray(
        [None if flag else "non_matching_occlusion_state" for flag in fused_valid],
        dtype=object,
    )

    fused_rows = append_prediction_rows(
        empty_predictions_frame(),
        n_rows=n,
        dense_frame_idx=collapsed["dense_frame_idx"].to_numpy(),
        time_sec=time_sec,
        sample_idx=sample_idx,
        prediction_grain=GRAIN_UNDIRECTED_PAIR,
        pair_id=pair_ids,
        mouse_a=collapsed["mouse_a"].to_numpy(),
        mouse_b=collapsed["mouse_b"].to_numpy(),
        focal_id=None,
        partner_id=None,
        family=FAMILY_DYADIC,
        detector_name="contact",
        specialist_name=None,
        output_role=OUTPUT_ROLE_LOGICAL_FUSED,
        model_version=models["contact_intact"].bundle.model_version,
        model_bundle_dir=str(models["contact_intact"].bundle.bundle_dir),
        model_sha256=None,
        # Final contact is a fusion of multiple classifier outputs.
        prob_raw=np.full(n, np.nan),
        prob_effective=prob_fused,
        threshold=fused_threshold,
        threshold_source=fused_source,
        prediction=pred_fused.astype(np.int8),
        input_valid=fused_valid,
        invalid_reason=fused_reason,
    )
    metas.append(
        DetectorRunMeta(
            family=FAMILY_DYADIC,
            detector_name="contact",
            specialist_name=None,
            output_role=OUTPUT_ROLE_LOGICAL_FUSED,
            model_version=models["contact_intact"].bundle.model_version,
            model_bundle_dir=None,
            model_sha256=None,
            threshold=None,
            threshold_source="specialist_routed",
            n_rows=n,
            n_scored=int(handled.sum()),
            n_positive=int(pred_fused.sum()),
            routing_mode="occlusion_gated_overwrite",
            extras={
                "intact_threshold": thr_intact.value,
                "occluded_threshold": thr_occ.value,
                "n_intact_mask": int(intact_mask.sum()),
                "n_occluded_mask": int(occ_mask.sum()),
                "n_unhandled_occlusion": int(unhandled.sum()),
            },
        )
    )
    diagnostics = pd.concat(diag_parts, ignore_index=True) if diag_parts else empty_predictions_frame()
    return fused_rows, diagnostics, metas, warnings
