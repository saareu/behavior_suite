"""Canonical ↔ wide view consistency, cardinality, and subset publication."""

from __future__ import annotations

import pytest

from behavioral_inference import (
    BehavioralInferenceConfig,
    PublicationError,
    build_canonical_predictions_table,
    build_dyadic_predictions_wide,
    build_focal_predictions_wide,
    build_model_manifest,
    run_behavioral_inference,
    validate_canonical_predictions,
)
from behavioral_inference.contracts import (
    GRAIN_DIRECTIONAL,
    GRAIN_UNDIRECTED_PAIR,
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_PRIMARY,
)
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)


def _models_subset():
    return {
        "nose_to_nose": make_loaded_model("nose_to_nose", "dyadic", ["feat_a"]),
        "side_by_side": make_loaded_model("side_by_side", "dyadic", ["feat_a"]),
        "approach": make_loaded_model("approach", "focal", ["feat_a"]),
        "contact_intact": make_loaded_model("contact_intact", "dyadic", ["feat_a"]),
        "contact_occluded": make_loaded_model("contact_occluded", "dyadic", ["feat_a"]),
        "nose_body_intact": make_loaded_model("nose_body_intact", "focal", ["feat_a"]),
        "nose_body_occ": make_loaded_model("nose_body_occ", "focal", ["feat_a"]),
    }


def test_prediction_key_uniqueness_and_cardinality() -> None:
    frame = make_two_mouse_feature_frame(n_frames=3, occlusion=[0, 1, 0])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact", "nose_to_nose"),
        active_detectors_focal=("approach", "nose_to_body"),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_models_subset(),
        selected_detectors_dyadic=["contact", "nose_to_nose"],
        selected_detectors_focal=["approach", "nose_to_body"],
    )
    metrics = validate_canonical_predictions(result, n_unique_frames=3)
    assert metrics["n_unique_frames"] == 3
    # One undirected row per dyadic detector per frame.
    for detector in ("contact", "nose_to_nose"):
        block = result.predictions[
            (result.predictions["detector_name"] == detector)
            & (
                result.predictions["output_role"].isin(
                    [OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]
                )
            )
        ]
        assert len(block) == 3
        assert (block["prediction_grain"] == GRAIN_UNDIRECTED_PAIR).all()
    # Two directional rows per focal detector per frame.
    for detector in ("approach", "nose_to_body"):
        block = result.predictions[
            (result.predictions["detector_name"] == detector)
            & (
                result.predictions["output_role"].isin(
                    [OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]
                )
            )
        ]
        assert len(block) == 6
        assert (block["prediction_grain"] == GRAIN_DIRECTIONAL).all()


def test_canonical_and_wide_consistency() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 1])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact", "nose_to_nose"),
        active_detectors_focal=("approach",),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_models_subset(),
        selected_detectors_dyadic=["contact", "nose_to_nose"],
        selected_detectors_focal=["approach"],
    )
    canon = build_canonical_predictions_table(
        result,
        session_root="/tmp/session",
        source_s6_run_id="combined_features__test",
        n_unique_frames=2,
    )
    assert "score_origin" in canon.columns
    assert "source_s6_run_id" in canon.columns

    dyadic_wide = build_dyadic_predictions_wide(
        predictions=result.predictions,
        diagnostics=result.diagnostics,
        selected_detectors=["contact", "nose_to_nose"],
    )
    focal_wide = build_focal_predictions_wide(
        predictions=result.predictions,
        diagnostics=result.diagnostics,
        selected_detectors=["approach"],
        features_df=frame,
    )
    assert len(dyadic_wide) == 2
    assert not dyadic_wide.duplicated(
        subset=["dense_frame_idx", "mouse_a", "mouse_b"]
    ).any()
    assert len(focal_wide) == 4
    assert not focal_wide.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()

    # Trace nose_to_nose probs.
    for _, row in dyadic_wide.iterrows():
        match = result.predictions[
            (result.predictions["detector_name"] == "nose_to_nose")
            & (result.predictions["dense_frame_idx"] == row["dense_frame_idx"])
            & (result.predictions["mouse_a"] == row["mouse_a"])
            & (result.predictions["mouse_b"] == row["mouse_b"])
        ]
        assert len(match) == 1
        assert float(row["prob_nose_to_nose"]) == pytest.approx(
            float(match["prob_effective"].iloc[0])
        )


def test_detector_subset_publication() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose", "side_by_side"),
        active_detectors_focal=("approach",),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_models_subset(),
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
    )
    assert set(result.predictions["detector_name"]) == {"nose_to_nose", "approach"}
    wide = build_dyadic_predictions_wide(
        predictions=result.predictions,
        diagnostics=result.diagnostics,
        selected_detectors=["nose_to_nose"],
    )
    assert "prob_nose_to_nose" in wide.columns
    assert "prob_side_by_side" not in wide.columns
    assert "prob_contact" not in wide.columns


def test_undirected_never_fabricates_directional_identities() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_models_subset(),
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=[],
    )
    undirected = result.predictions[
        result.predictions["prediction_grain"] == GRAIN_UNDIRECTED_PAIR
    ]
    assert undirected["focal_id"].isna().all()
    assert undirected["partner_id"].isna().all()


def test_model_manifest_thresholds_and_digests() -> None:
    frame = make_two_mouse_feature_frame(n_frames=1)
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_models_subset(),
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=[],
    )
    manifest = build_model_manifest(
        result,
        config_settings=config.to_settings_dict(),
        registry_root="/models",
    )
    assert manifest["selected_detectors"]["dyadic"] == ["nose_to_nose"]
    assert "nose_to_nose" in manifest["thresholds"]
    assert manifest["dyadic_collapse_policy"] == "lower_id_as_focal"
    assert any(b["detector_name"] == "nose_to_nose" for b in manifest["bundles"])


def test_duplicate_keys_fail_validation() -> None:
    frame = make_two_mouse_feature_frame(n_frames=1)
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_models_subset(),
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=[],
    )
    import pandas as pd

    from behavioral_inference.results import build_inference_result

    bad_preds = pd.concat(
        [result.predictions, result.predictions.iloc[[0]]], ignore_index=True
    )
    bad = build_inference_result(
        predictions=bad_preds,
        diagnostics=result.diagnostics,
        selected_detectors_dyadic=result.selected_detectors_dyadic,
        selected_detectors_focal=result.selected_detectors_focal,
        detector_meta=result.detector_meta,
        warnings=result.warnings,
        routing_diagnostics=result.routing_diagnostics,
        metadata=result.metadata,
    )
    with pytest.raises(PublicationError, match="duplicate"):
        validate_canonical_predictions(bad, n_unique_frames=1)
