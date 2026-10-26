"""Focal directional inference, nose-to-body routing, and smoothing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    AdapterSettings,
    BehavioralInferenceConfig,
    run_behavioral_inference,
)
from behavioral_inference.adapters.focal import apply_nose_to_body_rolling_median
from behavioral_inference.contracts import (
    GRAIN_DIRECTIONAL,
    OUTPUT_ROLE_LOGICAL_FUSED,
)
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)


def test_both_focal_directions_preserved() -> None:
    frame = make_two_mouse_feature_frame(n_frames=3)
    models = {
        "approach": make_loaded_model("approach", "focal", ["feat_a"]),
    }
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=(),
        active_detectors_focal=("approach",),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=[],
        selected_detectors_focal=["approach"],
    )
    preds = result.predictions
    assert len(preds) == 6  # 3 frames × 2 directions
    assert (preds["prediction_grain"] == GRAIN_DIRECTIONAL).all()
    assert preds["pair_id"].isna().all()
    pairs = set(zip(preds["focal_id"], preds["partner_id"], strict=True))
    assert pairs == {(0, 1), (1, 0)}


def test_focal_swap_changes_directional_input() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    models = {
        "approach": make_loaded_model("approach", "focal", ["feat_a"], weight=1.0, bias=0.0),
    }
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=(),
        active_detectors_focal=("approach",),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=[],
        selected_detectors_focal=["approach"],
    )
    preds = result.predictions.sort_values(["dense_frame_idx", "focal_id"])
    # feat_a is + for focal=0 and - for focal=1 → probabilities must differ.
    for frame_idx in (0, 1):
        rows = preds[preds["dense_frame_idx"] == frame_idx]
        p01 = float(rows.loc[rows["focal_id"] == 0, "prob_effective"].iloc[0])
        p10 = float(rows.loc[rows["focal_id"] == 1, "prob_effective"].iloc[0])
        assert p01 != p10


def test_nose_to_body_specialist_fusion_and_smoothing() -> None:
    frame = make_two_mouse_feature_frame(
        n_frames=6, occlusion=[0, 0, 0, 1, 1, 1]
    )
    models = {
        "nose_body_intact": make_loaded_model(
            "nose_body_intact", "focal", ["feat_a"], weight=1.0, bias=0.0
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
        result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED
    ]
    assert len(fused) == 12
    # Raw vs effective may differ when smoothing is on.
    assert "prob_raw" in fused.columns and "prob_effective" in fused.columns
    diags = result.diagnostics
    assert set(diags["specialist_name"].dropna()) == {
        "nose_body_intact",
        "nose_body_occ",
    }
    assert result.routing_diagnostics["nose_to_body"]["mode"] == "two_stage"
    assert result.routing_diagnostics["nose_to_body"]["smoothing"] is True


def test_nose_to_body_nan_occlusion_not_filled() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 0])
    frame.loc[frame["dense_frame_idx"] == 1, "is_total_occlusion"] = np.nan
    models = {
        "nose_body_intact": make_loaded_model(
            "nose_body_intact", "focal", ["feat_a"]
        ),
        "nose_body_occ": make_loaded_model("nose_body_occ", "focal", ["feat_a"]),
    }
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=(),
        active_detectors_focal=("nose_to_body",),
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
        & (result.predictions["dense_frame_idx"] == 1)
    ]
    # Raw fused stays 0 for NaN occlusion (no invented specialist).
    assert (fused["prob_raw"] == 0.0).all()
    # After legacy rolling median, effective may pull nonzero from neighbors;
    # prediction stays 0 because neither .eq(0) nor .eq(1) mask applies.
    assert (fused["prediction"].astype(int) == 0).all()
    assert (~fused["input_valid"].astype(bool)).all()
    assert (fused["invalid_reason"] == "non_matching_occlusion_state").all()


def test_rolling_median_edge_min_periods() -> None:
    frame = pd.DataFrame(
        {
            "exp_name": ["E"] * 5,
            "focal_id": [0] * 5,
            "partner_id": [1] * 5,
            "dense_frame_idx": [0, 1, 2, 3, 4],
            "prob_nose_to_body": [0.1, 0.9, 0.2, 0.8, 0.3],
        }
    )
    smoothed = apply_nose_to_body_rolling_median(frame, "prob_nose_to_body", window=5)
    # Frame 0: median of [0.1] = 0.1
    assert smoothed.iloc[0] == pytest.approx(0.1)
    # Frame 1: median of [0.1, 0.9] = 0.5
    assert smoothed.iloc[1] == pytest.approx(0.5)
    # Frame 4: median of all five
    assert smoothed.iloc[4] == pytest.approx(float(np.median([0.1, 0.9, 0.2, 0.8, 0.3])))


def test_rolling_median_preserves_all_nan_window() -> None:
    frame = pd.DataFrame(
        {
            "exp_name": ["E"] * 3,
            "focal_id": [0] * 3,
            "partner_id": [1] * 3,
            "dense_frame_idx": [0, 1, 2],
            "prob_nose_to_body": [np.nan, np.nan, np.nan],
        }
    )
    smoothed = apply_nose_to_body_rolling_median(frame, "prob_nose_to_body")
    assert smoothed.isna().all()


def test_single_stage_fallback_no_smoothing() -> None:
    frame = make_two_mouse_feature_frame(n_frames=3)
    models = {
        "nose_to_body": make_loaded_model(
            "nose_to_body", "focal", ["feat_a"], support="fallback"
        ),
    }
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=(),
        active_detectors_focal=("nose_to_body",),
        adapters=AdapterSettings(allow_nose_to_body_single_stage_fallback=True),
    ).validated()
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=[],
        selected_detectors_focal=["nose_to_body"],
    )
    fused = result.predictions
    assert (fused["prob_raw"] == fused["prob_effective"]).all()
    assert result.routing_diagnostics["nose_to_body"]["mode"] == "single_stage_fallback"
    assert result.routing_diagnostics["nose_to_body"]["smoothing"] is False


def test_raw_framewise_disables_smoothing() -> None:
    frame = make_two_mouse_feature_frame(n_frames=4, occlusion=[0, 0, 1, 1])
    models = {
        "nose_body_intact": make_loaded_model(
            "nose_body_intact", "focal", ["feat_a"]
        ),
        "nose_body_occ": make_loaded_model("nose_body_occ", "focal", ["feat_a"]),
    }
    config = BehavioralInferenceConfig(
        parity_mode="raw_framewise",
        active_detectors_dyadic=(),
        active_detectors_focal=("nose_to_body",),
        adapters=AdapterSettings(nose_to_body_rolling_median_enabled=True),
    ).validated()
    assert config.adapters.nose_to_body_rolling_median_enabled is False
    result = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=[],
        selected_detectors_focal=["nose_to_body"],
    )
    fused = result.predictions[
        result.predictions["output_role"] == OUTPUT_ROLE_LOGICAL_FUSED
    ]
    assert (fused["prob_raw"] == fused["prob_effective"]).all()
