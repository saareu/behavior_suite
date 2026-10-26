"""Canonical unordered-pair collapse and dyadic cardinality tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    BehavioralInferenceConfig,
    InferenceError,
    collapse_to_unordered_pairs,
    run_behavioral_inference,
)
from behavioral_inference.contracts import GRAIN_UNDIRECTED_PAIR, OUTPUT_ROLE_PRIMARY
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)


def test_collapse_keeps_lower_id_as_focal() -> None:
    frame = make_two_mouse_feature_frame(n_frames=3)
    collapsed = collapse_to_unordered_pairs(frame)
    assert len(collapsed) == 3
    assert (collapsed["focal_id"] == collapsed["mouse_a"]).all()
    assert (collapsed["partner_id"] == collapsed["mouse_b"]).all()
    assert (collapsed["mouse_a"] < collapsed["mouse_b"]).all()


def test_collapse_invariant_to_row_order() -> None:
    frame = make_two_mouse_feature_frame(n_frames=5)
    # Reverse and shuffle-like reorder including directional swaps first.
    reversed_rows = frame.iloc[::-1].reset_index(drop=True)
    permuted = pd.concat(
        [frame.iloc[1::2], frame.iloc[::2]], ignore_index=True
    )
    a = collapse_to_unordered_pairs(frame)
    b = collapse_to_unordered_pairs(reversed_rows)
    c = collapse_to_unordered_pairs(permuted)
    keys = ["dense_frame_idx", "mouse_a", "mouse_b", "feat_a"]
    pd.testing.assert_frame_equal(
        a[keys].reset_index(drop=True),
        b[keys].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        a[keys].reset_index(drop=True),
        c[keys].reset_index(drop=True),
    )


def test_collapse_rejects_nan_dense_frame_idx() -> None:
    frame = make_two_mouse_feature_frame(n_frames=2)
    frame.loc[0, "dense_frame_idx"] = np.nan
    with pytest.raises(InferenceError, match="NaN dense_frame_idx"):
        collapse_to_unordered_pairs(frame)


def test_one_undirected_row_per_sample_detector() -> None:
    frame = make_two_mouse_feature_frame(n_frames=4)
    models = {
        "nose_to_nose": make_loaded_model("nose_to_nose", "dyadic", ["feat_a"]),
    }
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
    preds = result.predictions
    assert len(preds) == 4
    assert (preds["prediction_grain"] == GRAIN_UNDIRECTED_PAIR).all()
    assert preds["focal_id"].isna().all()
    assert preds["partner_id"].isna().all()
    assert preds["pair_id"].nunique() == 1
    assert preds.duplicated(
        subset=["dense_frame_idx", "pair_id", "detector_name", "output_role"]
    ).sum() == 0
    assert (preds["output_role"] == OUTPUT_ROLE_PRIMARY).all()


def test_dyadic_inference_row_order_invariance() -> None:
    frame = make_two_mouse_feature_frame(n_frames=6)
    models = {
        "side_by_side": make_loaded_model(
            "side_by_side", "dyadic", ["feat_a"], weight=0.5, bias=-0.2
        ),
    }
    config = BehavioralInferenceConfig(
        active_detectors_dyadic=("side_by_side",),
        active_detectors_focal=(),
    ).validated()
    r1 = run_behavioral_inference(
        frame,
        config,
        models=models,
        selected_detectors_dyadic=["side_by_side"],
        selected_detectors_focal=[],
    )
    r2 = run_behavioral_inference(
        frame.iloc[::-1].reset_index(drop=True),
        config,
        models=models,
        selected_detectors_dyadic=["side_by_side"],
        selected_detectors_focal=[],
    )
    cols = ["dense_frame_idx", "pair_id", "prob_effective", "prediction"]
    a = r1.predictions[cols].sort_values("dense_frame_idx").reset_index(drop=True)
    b = r2.predictions[cols].sort_values("dense_frame_idx").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
