"""Contact specialist routing and fusion tests."""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_inference import (
    BehavioralInferenceConfig,
    InferenceError,
    run_behavioral_inference,
)
from behavioral_inference.contracts import (
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_SPECIALIST,
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


def test_contact_intact_and_occluded_routing() -> None:
    # Frames: intact, occluded, intact, occluded
    frame = make_two_mouse_feature_frame(
        n_frames=4, occlusion=[0, 1, 0, 1]
    )
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
    fused = result.predictions[
        result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED
    ].sort_values("dense_frame_idx")
    assert len(fused) == 4
    assert fused["prob_raw"].isna().all()
    assert fused["prob_effective"].notna().all()

    diags = result.diagnostics
    intact = diags[diags["specialist_name"] == "contact_intact"].sort_values(
        "dense_frame_idx"
    )
    occ = diags[diags["specialist_name"] == "contact_occluded"].sort_values(
        "dense_frame_idx"
    )
    # Intact scored on frames 0,2; excluded on 1,3
    assert list(intact["input_valid"].astype(bool)) == [True, False, True, False]
    assert list(occ["input_valid"].astype(bool)) == [False, True, False, True]
    # Excluded specialist probs forced to 0
    assert float(intact.loc[~intact["input_valid"].astype(bool), "prob_effective"].iloc[0]) == 0.0


def test_contact_nan_occlusion_treated_as_intact() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 0])
    # Set NaN occlusion on frame 1 both directions
    frame.loc[frame["dense_frame_idx"] == 1, "is_total_occlusion"] = np.nan
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
    intact = result.diagnostics[
        result.diagnostics["specialist_name"] == "contact_intact"
    ].sort_values("dense_frame_idx")
    # NaN→0 coercion → intact route
    assert bool(intact.loc[intact["dense_frame_idx"] == 1, "input_valid"].iloc[0])


def test_contact_invalid_occlusion_no_invented_prediction() -> None:
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
    fused = result.predictions[
        result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED
    ].sort_values("dense_frame_idx")
    row = fused.loc[fused["dense_frame_idx"] == 1].iloc[0]
    assert float(row["prob_effective"]) == 0.0
    assert int(row["prediction"]) == 0
    assert not bool(row["input_valid"])
    assert row["invalid_reason"] == "non_matching_occlusion_state"
    assert any("neither 0 nor 1" in w for w in result.warnings)


def test_contact_missing_occlusion_column_fails() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    frame = frame.drop(columns=["is_total_occlusion"])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact",),
        active_detectors_focal=(),
    ).validated()
    with pytest.raises(InferenceError, match="is_total_occlusion"):
        run_behavioral_inference(
            frame,
            config,
            models=_contact_models(),
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=[],
        )


def test_contact_missing_specialist_fails() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact",),
        active_detectors_focal=(),
    ).validated()
    with pytest.raises(InferenceError, match="contact_occluded"):
        run_behavioral_inference(
            frame,
            config,
            models={"contact_intact": _contact_models()["contact_intact"]},
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=[],
        )


def test_specialist_thresholds_applied() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 1])
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("contact",),
        active_detectors_focal=(),
        working_thresholds={"contact": {"intact": 0.99, "occluded": 0.01}},
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=_contact_models(),
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=[],
    )
    intact = result.diagnostics[
        (result.diagnostics["specialist_name"] == "contact_intact")
        & (result.diagnostics["output_role"] == OUTPUT_ROLE_SPECIALIST)
    ]
    assert float(intact["threshold"].iloc[0]) == 0.99
    assert intact["threshold_source"].iloc[0] == "inference_config"
