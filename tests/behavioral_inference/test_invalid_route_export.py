"""Sprint 3 review-point regressions: invalid routing sentinels and validity."""

from __future__ import annotations

import numpy as np

from behavioral_inference import (
    AdapterSettings,
    BehavioralInferenceConfig,
    annotate_score_origin,
    build_dyadic_predictions_wide,
    build_focal_predictions_wide,
    run_behavioral_inference,
    validate_canonical_predictions,
)
from behavioral_inference.contracts import (
    OUTPUT_ROLE_LOGICAL_FUSED,
    SCORE_ORIGIN_FUSION_ROUTED,
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
)
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)


def _contact_models():
    return {
        "contact_intact": make_loaded_model(
            "contact_intact", "dyadic", ["feat_a"], weight=2.0, bias=0.0
        ),
        "contact_occluded": make_loaded_model(
            "contact_occluded", "dyadic", ["feat_a"], weight=-2.0, bias=3.0
        ),
    }


def test_contact_invalid_route_preserves_input_valid_and_sentinel() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 2])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_contact_models(),
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=[],
    )
    validate_canonical_predictions(result, n_unique_frames=2)
    fused = result.predictions[
        result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED
    ].sort_values("dense_frame_idx")
    invalid = fused.loc[fused["dense_frame_idx"] == 1].iloc[0]
    assert not bool(invalid["input_valid"])
    assert float(invalid["prob_effective"]) == 0.0
    assert int(invalid["prediction"]) == 0
    annotated = annotate_score_origin(result.predictions)
    origin = annotated.loc[
        (annotated["dense_frame_idx"] == 1)
        & (annotated["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED),
        "score_origin",
    ].iloc[0]
    assert origin == SCORE_ORIGIN_ROUTING_SENTINEL

    # Observed zero on a valid intact frame must not share the sentinel origin.
    valid_row = annotated.loc[
        (annotated["dense_frame_idx"] == 0)
        & (annotated["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED)
    ].iloc[0]
    assert bool(valid_row["input_valid"])
    assert valid_row["score_origin"] == SCORE_ORIGIN_FUSION_ROUTED


def test_contact_excluded_specialist_zero_is_sentinel_not_observed() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 1])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_contact_models(),
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=[],
    )
    diags = annotate_score_origin(result.diagnostics)
    intact_excluded = diags[
        (diags["specialist_name"] == "contact_intact")
        & (diags["dense_frame_idx"] == 1)
    ].iloc[0]
    assert not bool(intact_excluded["input_valid"])
    assert float(intact_excluded["prob_effective"]) == 0.0
    assert intact_excluded["score_origin"] == SCORE_ORIGIN_ROUTING_SENTINEL

    intact_scored = diags[
        (diags["specialist_name"] == "contact_intact")
        & (diags["dense_frame_idx"] == 0)
    ].iloc[0]
    assert bool(intact_scored["input_valid"])
    assert intact_scored["score_origin"] == SCORE_ORIGIN_MODEL_OBSERVED


def test_wide_dyadic_preserves_contact_validity() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 2])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_contact_models(),
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=[],
    )
    wide = build_dyadic_predictions_wide(
        predictions=result.predictions,
        diagnostics=result.diagnostics,
        selected_detectors=["contact"],
    )
    assert "input_valid_contact" in wide.columns
    assert "score_origin_contact" in wide.columns
    row = wide.loc[wide["dense_frame_idx"] == 1].iloc[0]
    assert not bool(row["input_valid_contact"])
    assert float(row["prob_contact"]) == 0.0
    assert int(row["pred_contact"]) == 0
    assert row["score_origin_contact"] == SCORE_ORIGIN_ROUTING_SENTINEL
    # Collapse survivors only — undirected keys, no fabricated opposite direction.
    assert set(zip(wide["focal_id"], wide["partner_id"], strict=True)) == {(0, 1)}
    assert (wide["mouse_a"] < wide["mouse_b"]).all()


def test_nose_to_body_invalid_preserves_raw_vs_effective_and_blocks_rethreshold() -> None:
    frame = make_two_mouse_feature_frame(n_frames=4, occlusion=[0, 0, 0, 0])
    # Make frame 2 invalid (NaN occlusion) while neighbors are scored.
    frame.loc[frame["dense_frame_idx"] == 2, "is_total_occlusion"] = np.nan
    models = {
        "nose_body_intact": make_loaded_model(
            "nose_body_intact", "focal", ["feat_a"], weight=3.0, bias=0.0
        ),
        "nose_body_occ": make_loaded_model(
            "nose_body_occ", "focal", ["feat_a"], weight=0.5, bias=1.0
        ),
    }
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=(),
        active_detectors_focal=("nose_to_body",),
        adapters=AdapterSettings(nose_to_body_rolling_median_enabled=True),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=[],
        selected_detectors_focal=["nose_to_body"],
    )
    fused = result.predictions[
        (result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED)
        & (result.predictions["dense_frame_idx"] == 2)
    ]
    assert (fused["prob_raw"] == 0.0).all()
    assert (fused["prediction"].astype(int) == 0).all()
    assert (~fused["input_valid"].astype(bool)).all()
    # After smoothing, effective may be non-zero from neighbors — must not be
    # treated as an observed score eligible for blind re-thresholding.
    wide = build_focal_predictions_wide(
        predictions=result.predictions,
        diagnostics=result.diagnostics,
        selected_detectors=["nose_to_body"],
        features_df=frame,
    )
    bad = wide.loc[wide["dense_frame_idx"] == 2]
    assert (~bad["input_valid_nose_to_body"].astype(bool)).all()
    assert (bad["pred_nose_to_body"].astype(int) == 0).all()
    assert "prob_nose_to_body_raw" in wide.columns
    assert (bad["prob_nose_to_body_raw"] == 0.0).all()
    assert "score_origin_nose_to_body" in wide.columns
    assert (bad["score_origin_nose_to_body"] == SCORE_ORIGIN_ROUTING_SENTINEL).all()
