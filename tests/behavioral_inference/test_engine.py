"""Shared inference engine: selection, failures, schemas, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    PREDICTION_COLUMNS,
    BehavioralInferenceConfig,
    DetectorSelectionError,
    InferenceError,
    ThresholdConfigError,
    run_behavioral_inference,
)
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)


def _standalone_models():
    return {
        "nose_to_nose": make_loaded_model("nose_to_nose", "dyadic", ["feat_a"]),
        "side_by_side": make_loaded_model("side_by_side", "dyadic", ["feat_a"]),
        "approach": make_loaded_model("approach", "focal", ["feat_a"]),
        "retract": make_loaded_model("retract", "focal", ["feat_a"]),
    }


def test_selected_subset_does_not_change_unrelated_probs() -> None:
    frame = make_two_mouse_feature_frame(n_frames=3)
    models = _standalone_models()
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose", "side_by_side"),
        active_detectors_focal=("approach",),
    ).validated()
    full = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose", "side_by_side"],
        selected_detectors_focal=["approach"],
    )
    subset = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
    )
    full_n2n = full.predictions[
        full.predictions["detector_name"] == "nose_to_nose"
    ].sort_values("dense_frame_idx")
    sub_n2n = subset.predictions[
        subset.predictions["detector_name"] == "nose_to_nose"
    ].sort_values("dense_frame_idx")
    pd.testing.assert_series_equal(
        full_n2n["prob_effective"].reset_index(drop=True),
        sub_n2n["prob_effective"].reset_index(drop=True),
        check_names=False,
    )
    full_app = full.predictions[
        full.predictions["detector_name"] == "approach"
    ].sort_values(["dense_frame_idx", "focal_id"])
    sub_app = subset.predictions[
        subset.predictions["detector_name"] == "approach"
    ].sort_values(["dense_frame_idx", "focal_id"])
    pd.testing.assert_series_equal(
        full_app["prob_effective"].reset_index(drop=True),
        sub_app["prob_effective"].reset_index(drop=True),
        check_names=False,
    )
    assert "side_by_side" not in set(subset.predictions["detector_name"])


def test_missing_model_fails_fast() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
    ).validated()
    with pytest.raises(InferenceError, match="nose_to_nose"):
        run_behavioral_inference(
            frame,
            config,
            models={},
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=[],
        )


def test_missing_threshold_fails() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    models = {"nose_to_nose": make_loaded_model("nose_to_nose", "dyadic", ["feat_a"])}
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
        working_thresholds={},
    ).validated()
    with pytest.raises(ThresholdConfigError):
        run_behavioral_inference(
            frame,
            config,
            models=models,
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=[],
        )


def test_inf_feature_rejection() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    frame.loc[0, "feat_a"] = np.inf
    models = {"nose_to_nose": make_loaded_model("nose_to_nose", "dyadic", ["feat_a"])}
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
    ).validated()
    with pytest.raises(InferenceError, match="Inf|feature validation"):
        run_behavioral_inference(
            frame,
            config,
            models=models,
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=[],
        )


def test_nan_features_propagate_to_probability() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    # Put NaN on the canonical undirected row for frame 0 (focal=0, partner=1).
    mask = (frame["dense_frame_idx"] == 0) & (frame["focal_id"] == 0)
    frame.loc[mask, "feat_a"] = np.nan
    models = {"nose_to_nose": make_loaded_model("nose_to_nose", "dyadic", ["feat_a"])}
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=(),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=[],
    )
    row = result.predictions[result.predictions["dense_frame_idx"] == 0].iloc[0]
    assert np.isnan(float(row["prob_effective"]))
    assert int(row["prediction"]) == 0  # NaN >= thr → False


def test_require_exactly_two_mice() -> None:
    frame = make_two_mouse_feature_frame(n_frames=1)
    # Add a third mouse direction.
    extra = frame.copy()
    extra["partner_id"] = 2
    frame = pd.concat([frame, extra], ignore_index=True)
    models = {"approach": make_loaded_model("approach", "focal", ["feat_a"])}
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=(),
        active_detectors_focal=("approach",),
    ).validated()
    with pytest.raises(InferenceError, match="exactly two tracked mice"):
        run_behavioral_inference(
            frame,
            config,
            models=models,
            selected_detectors_dyadic=[],
            selected_detectors_focal=["approach"],
        )


def test_output_keys_and_cardinality() -> None:
    frame = make_two_mouse_feature_frame(n_frames=5)
    models = _standalone_models()
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=("approach", "retract"),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach", "retract"],
    )
    assert list(result.predictions.columns) == list(PREDICTION_COLUMNS)
    # Dyadic: 5 undirected; focal: 5*2 per detector × 2 detectors = 20; total 25
    assert len(result.predictions) == 5 + 20
    assert result.metadata["dyadic_rows_per_detector"] == 5
    assert result.metadata["focal_rows_per_detector"] == 10
    assert set(result.selected_detectors_dyadic) == {"nose_to_nose"}
    assert set(result.selected_detectors_focal) == {"approach", "retract"}
    assert "nose_to_nose" in result.thresholds
    assert "approach" in result.model_identities


def test_determinism() -> None:
    frame = make_two_mouse_feature_frame(n_frames=4)
    models = _standalone_models()
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=("approach",),
    ).validated()
    r1 = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
    )
    r2 = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
    )
    pd.testing.assert_frame_equal(r1.predictions, r2.predictions)


def test_unknown_detector_rejected() -> None:
    frame = make_two_mouse_feature_frame(n_frames=1)
    config = BehavioralInferenceConfig().validated()
    with pytest.raises(DetectorSelectionError):
        run_behavioral_inference(
            frame,
            config,
            models={},
            selected_detectors_dyadic=["not_a_detector"],
            selected_detectors_focal=[],
        )


def test_empty_selection_rejected() -> None:
    frame = make_two_mouse_feature_frame(n_frames=1)
    config = BehavioralInferenceConfig().validated()
    with pytest.raises(DetectorSelectionError, match="No detectors selected"):
        run_behavioral_inference(
            frame,
            config,
            models={},
            selected_detectors_dyadic=[],
            selected_detectors_focal=[],
        )
