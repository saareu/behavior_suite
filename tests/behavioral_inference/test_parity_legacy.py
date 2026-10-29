"""Focused legacy parity checks vs Pipeline 70/90 formulas (lab models).

When ``models`` and ``[s7]`` deps are available, score a small S6
snippet with S7 and independently re-apply the Pipeline 70/90 scoring steps
using the same loaded estimators. Binary predictions must match exactly;
probabilities must match within a documented float tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    BehavioralInferenceConfig,
    collapse_to_unordered_pairs,
    discover_model_registry,
    load_behavioral_inference_config,
    load_model_bundle,
    run_behavioral_inference,
)
from behavioral_inference.adapters.focal import apply_nose_to_body_rolling_median
from behavioral_inference.contracts import OUTPUT_ROLE_LOGICAL_FUSED, OUTPUT_ROLE_PRIMARY
from behavioral_inference.features import require_feature_matrix
from combined_features.config import ModelRegistrySettings
from tests.behavioral_inference.conftest import (
    LAB_MODEL_ROOT,
    S6_FEATURES,
    lab_registry_available,
    s7_deps_available,
)

PROB_RTOL = 1e-6
PROB_ATOL = 1e-8
N_FRAMES = 48  # small focused snippet (~96 directional rows)


def _skip_unless_lab() -> None:
    if not lab_registry_available():
        pytest.skip("lab model registry (models) not available")
    if not s7_deps_available():
        pytest.skip("optional [s7] dependencies not installed")
    if not S6_FEATURES.is_file():
        pytest.skip(f"S6 feature artifact missing: {S6_FEATURES}")


def _load_snippet() -> pd.DataFrame:
    full = pd.read_parquet(S6_FEATURES)
    frames = sorted(full["dense_frame_idx"].unique())[:N_FRAMES]
    return full[full["dense_frame_idx"].isin(frames)].copy()


def _config_with_lab_registry() -> BehavioralInferenceConfig:
    base = load_behavioral_inference_config()
    return BehavioralInferenceConfig(
        schema_version=base.schema_version,
        model_iteration=base.model_iteration,
        parity_mode="legacy_parity",
        model_registry=ModelRegistrySettings(
            root=LAB_MODEL_ROOT,
            dyadic_subdir="dyadic/v1",
            focal_subdir="focal/v1",
            include_obsolete=False,
            include_focal=True,
        ),
        active_detectors_dyadic=base.active_detectors_dyadic,
        active_detectors_focal=base.active_detectors_focal,
        working_thresholds=dict(base.working_thresholds),
        user_threshold_overrides={},
        allow_default_threshold_0_5=False,
        adapters=base.adapters,
    ).validated()


@pytest.mark.s7_model_smoke
def test_parity_dyadic_standalone_vs_legacy_formula() -> None:
    _skip_unless_lab()
    features = _load_snippet()
    config = _config_with_lab_registry()
    snapshot = discover_model_registry(config.model_registry)
    model = load_model_bundle(snapshot.by_name()["nose_to_nose"])

    result = run_behavioral_inference(
        features,
        config,
        models={"nose_to_nose": model},
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=[],
    )
    s7 = result.predictions.sort_values(["dense_frame_idx", "pair_id"]).reset_index(
        drop=True
    )

    # Independent Pipeline 70 mirror.
    collapsed = collapse_to_unordered_pairs(features)
    matrix = require_feature_matrix(collapsed, model.bundle.feature_cols)
    legacy_probs = model.positive_proba(matrix)
    thr = float(config.working_thresholds["nose_to_nose"])
    legacy_preds = (legacy_probs >= thr).astype(np.int8)

    assert len(s7) == len(collapsed)
    np.testing.assert_allclose(
        s7["prob_effective"].to_numpy(dtype=float),
        legacy_probs,
        rtol=PROB_RTOL,
        atol=PROB_ATOL,
    )
    np.testing.assert_array_equal(
        s7["prediction"].to_numpy(dtype=np.int8),
        legacy_preds,
    )
    # Canonical keys: lower ID focal direction after collapse.
    assert (collapsed["focal_id"] <= collapsed["partner_id"]).all()


@pytest.mark.s7_model_smoke
def test_parity_contact_routing_vs_legacy_formula() -> None:
    _skip_unless_lab()
    features = _load_snippet()
    config = _config_with_lab_registry()
    snapshot = discover_model_registry(config.model_registry)
    by_name = snapshot.by_name()
    models = {
        "contact_intact": load_model_bundle(by_name["contact_intact"]),
        "contact_occluded": load_model_bundle(by_name["contact_occluded"]),
    }
    result = run_behavioral_inference(
        features,
        config,
        models=models,
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=[],
    )
    fused = (
        result.predictions[result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED]
        .sort_values("dense_frame_idx")
        .reset_index(drop=True)
    )

    collapsed = collapse_to_unordered_pairs(features)
    occ = (
        pd.to_numeric(collapsed["is_total_occlusion"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    intact_mask = occ.eq(0).to_numpy()
    occ_mask = occ.eq(1).to_numpy()

    thr_i = float(config.working_thresholds["contact"]["intact"])
    thr_o = float(config.working_thresholds["contact"]["occluded"])

    def _score(model, mask: np.ndarray) -> np.ndarray:
        out = np.zeros(len(collapsed), dtype=float)
        if mask.any():
            matrix = require_feature_matrix(
                collapsed.loc[mask], model.bundle.feature_cols
            )
            out[mask] = model.positive_proba(matrix)
        return out

    p_i = _score(models["contact_intact"], intact_mask)
    p_o = _score(models["contact_occluded"], occ_mask)
    pred_i = (p_i >= thr_i).astype(int)
    pred_o = (p_o >= thr_o).astype(int)
    legacy_prob = p_i.copy()
    legacy_pred = pred_i.copy()
    legacy_prob[occ_mask] = p_o[occ_mask]
    legacy_pred[occ_mask] = pred_o[occ_mask]

    np.testing.assert_allclose(
        fused["prob_effective"].to_numpy(dtype=float),
        legacy_prob,
        rtol=PROB_RTOL,
        atol=PROB_ATOL,
    )
    np.testing.assert_array_equal(
        fused["prediction"].to_numpy(dtype=np.int8),
        legacy_pred.astype(np.int8),
    )


@pytest.mark.s7_model_smoke
def test_parity_nose_to_body_vs_legacy_formula() -> None:
    _skip_unless_lab()
    features = _load_snippet()
    config = _config_with_lab_registry()
    snapshot = discover_model_registry(config.model_registry)
    by_name = snapshot.by_name()
    models = {
        "nose_body_intact": load_model_bundle(by_name["nose_body_intact"]),
        "nose_body_occ": load_model_bundle(by_name["nose_body_occ"]),
    }
    result = run_behavioral_inference(
        features,
        config,
        models=models,
        selected_detectors_dyadic=[],
        selected_detectors_focal=["nose_to_body"],
    )
    fused = (
        result.predictions[result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED]
        .sort_values(["dense_frame_idx", "focal_id", "partner_id"])
        .reset_index(drop=True)
    )

    directional = features.sort_values(
        ["exp_name", "dense_frame_idx", "focal_id", "partner_id"], kind="stable"
    ).reset_index(drop=True)
    mask_i = directional["is_total_occlusion"].eq(0).to_numpy()
    mask_o = directional["is_total_occlusion"].eq(1).to_numpy()
    thr_i = float(config.working_thresholds["nose_to_body"]["intact"])
    thr_o = float(config.working_thresholds["nose_to_body"]["occluded"])

    prob_raw = np.zeros(len(directional), dtype=float)
    if mask_i.any():
        mi = require_feature_matrix(
            directional.loc[mask_i], models["nose_body_intact"].bundle.feature_cols
        )
        prob_raw[mask_i] = models["nose_body_intact"].positive_proba(mi)
    if mask_o.any():
        mo = require_feature_matrix(
            directional.loc[mask_o], models["nose_body_occ"].bundle.feature_cols
        )
        prob_raw[mask_o] = models["nose_body_occ"].positive_proba(mo)

    work = directional.copy()
    work["prob_nose_to_body"] = prob_raw
    smoothed = apply_nose_to_body_rolling_median(work, "prob_nose_to_body")
    legacy_eff = smoothed.to_numpy(dtype=float)
    legacy_pred = np.zeros(len(directional), dtype=np.int8)
    legacy_pred[mask_i] = (legacy_eff[mask_i] >= thr_i).astype(np.int8)
    legacy_pred[mask_o] = (legacy_eff[mask_o] >= thr_o).astype(np.int8)

    # Align S7 to the same sort.
    s7 = fused.sort_values(
        ["dense_frame_idx", "focal_id", "partner_id"]
    ).reset_index(drop=True)
    assert list(zip(s7["focal_id"], s7["partner_id"], s7["dense_frame_idx"], strict=True)) == list(
        zip(
            directional["focal_id"],
            directional["partner_id"],
            directional["dense_frame_idx"],
            strict=True,
        )
    )
    np.testing.assert_allclose(
        s7["prob_raw"].to_numpy(dtype=float),
        prob_raw,
        rtol=PROB_RTOL,
        atol=PROB_ATOL,
    )
    np.testing.assert_allclose(
        s7["prob_effective"].to_numpy(dtype=float),
        legacy_eff,
        rtol=PROB_RTOL,
        atol=PROB_ATOL,
    )
    np.testing.assert_array_equal(
        s7["prediction"].to_numpy(dtype=np.int8),
        legacy_pred,
    )


@pytest.mark.s7_model_smoke
def test_parity_focal_approach_keys() -> None:
    _skip_unless_lab()
    features = _load_snippet()
    config = _config_with_lab_registry()
    snapshot = discover_model_registry(config.model_registry)
    model = load_model_bundle(snapshot.by_name()["approach"])
    result = run_behavioral_inference(
        features,
        config,
        models={"approach": model},
        selected_detectors_dyadic=[],
        selected_detectors_focal=["approach"],
    )
    preds = result.predictions[
        result.predictions["output_role"] == OUTPUT_ROLE_PRIMARY
    ]
    assert len(preds) == len(features)
    # Both directions present per frame.
    counts = preds.groupby("dense_frame_idx").size()
    assert (counts == 2).all()
    feat_sorted = features.sort_values(
        ["dense_frame_idx", "focal_id", "partner_id"], kind="stable"
    )
    # Recompute on sorted order for key alignment.
    matrix_sorted = require_feature_matrix(feat_sorted, model.bundle.feature_cols)
    legacy_sorted = model.positive_proba(matrix_sorted)
    s7 = preds.sort_values(
        ["dense_frame_idx", "focal_id", "partner_id"]
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        s7["prob_effective"].to_numpy(dtype=float),
        legacy_sorted,
        rtol=PROB_RTOL,
        atol=PROB_ATOL,
    )
    thr = float(config.working_thresholds["approach"])
    np.testing.assert_array_equal(
        s7["prediction"].to_numpy(dtype=np.int8),
        (legacy_sorted >= thr).astype(np.int8),
    )
    # Directional identity fields must match feature keys exactly.
    pd.testing.assert_series_equal(
        s7["focal_id"].reset_index(drop=True),
        feat_sorted["focal_id"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        s7["partner_id"].reset_index(drop=True),
        feat_sorted["partner_id"].reset_index(drop=True),
        check_names=False,
    )
