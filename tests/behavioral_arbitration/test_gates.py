"""Hard-distance gates, postural vetoes, NaN behavior, parallel channels."""

from __future__ import annotations

import numpy as np

from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.contracts import (
    EVIDENCE_RULE_VETOED,
    EVIDENCE_UNOBSERVED_INVALID_ROUTE,
    REQUIRED_HARD_GAP_COLUMNS,
)
from behavioral_arbitration.expand import expand_dyadic_to_directional
from behavioral_arbitration.gates import (
    apply_hard_distance_probability_gates,
    apply_postural_approach_veto,
    apply_postural_retract_veto,
    assert_parallel_channels_preserved,
    merge_s6_features,
)
from behavioral_arbitration.validity import normalize_arbitration_evidence
from tests.behavioral_arbitration._fixtures import (
    MOUSE_A,
    MOUSE_B,
    make_dyadic_wide,
    make_focal_wide,
    make_s6_features,
)


def _prepared_frames(
    *,
    distance_overrides=None,
    invalid_contact_frames=None,
    include_veto: bool = True,
):
    dyadic = make_dyadic_wide(invalid_contact_frames=invalid_contact_frames)
    focal = make_focal_wide()
    features = make_s6_features(
        distance_overrides=distance_overrides,
        include_veto=include_veto,
    )
    frames, _ = expand_dyadic_to_directional(dyadic, focal)
    frames, _ = normalize_arbitration_evidence(frames)
    frames, _ = merge_s6_features(
        frames, features, feature_columns=list(REQUIRED_HARD_GAP_COLUMNS)
        + [
            "body_contraction_rate_7",
            "behind_score_bl_j",
            "cos_bearing_i_to_j",
            "rolling_mean_projected_vel_i_to_j_60",
            "rolling_speed_i_15",
            "rolling_speed_j_15",
            "focal_effort_ratio_15",
            "spine_directed_vel_15_i",
            "rolling_mean_projected_vel_combined_i_to_j_7",
            "signed_kinematic_intent_nose_or_neck",
        ]
    )
    return frames


def test_hard_distance_gate_zeros_far_valid_rows() -> None:
    # Nose-to-nose threshold default 0.3; set far on frame 0 A→B.
    overrides = {
        (0, MOUSE_A, MOUSE_B): {"distance_nose_i_nose_j_norm": 0.5},
        (0, MOUSE_B, MOUSE_A): {"distance_nose_i_nose_j_norm": 0.5},
    }
    frames = _prepared_frames(distance_overrides=overrides)
    config = load_behavioral_arbitration_config(None)
    # Frame 0 was valid positive for nose_to_nose before gate.
    assert float(frames.loc[frames["dense_frame_idx"] == 0, "prob_nose_to_nose"].iloc[0]) > 0
    out, diags = apply_hard_distance_probability_gates(frames, config)
    gated = out[out["dense_frame_idx"] == 0]
    assert (gated["prob_nose_to_nose"] == 0.0).all()
    assert (gated["pred_nose_to_nose"] == 0).all()
    assert (gated["evidence_state_nose_to_nose"] == EVIDENCE_RULE_VETOED).all()
    assert any(d.detector == "nose_to_nose" and d.n_rows_changed >= 2 for d in diags)


def test_hard_distance_nan_distance_skips_gate() -> None:
    overrides = {
        (0, MOUSE_A, MOUSE_B): {"distance_nose_i_nose_j_norm": np.nan},
        (0, MOUSE_B, MOUSE_A): {"distance_nose_i_nose_j_norm": np.nan},
    }
    frames = _prepared_frames(distance_overrides=overrides)
    config = load_behavioral_arbitration_config(None)
    before = frames.loc[frames["dense_frame_idx"] == 0, "prob_nose_to_nose"].copy()
    out, diags = apply_hard_distance_probability_gates(frames, config)
    after = out.loc[out["dense_frame_idx"] == 0, "prob_nose_to_nose"]
    assert np.allclose(before.to_numpy(dtype=float), after.to_numpy(dtype=float))
    nn_diag = [d for d in diags if d.detector == "nose_to_nose"]
    assert not nn_diag or nn_diag[0].n_rows_changed == 0


def test_hard_distance_does_not_convert_sentinel_to_valid_negative() -> None:
    overrides = {
        (0, MOUSE_A, MOUSE_B): {"distance_spine_i_spine_j_norm": 2.0},
        (0, MOUSE_B, MOUSE_A): {"distance_spine_i_spine_j_norm": 2.0},
    }
    frames = _prepared_frames(
        distance_overrides=overrides,
        invalid_contact_frames={0},
    )
    config = load_behavioral_arbitration_config(None)
    out, _ = apply_hard_distance_probability_gates(frames, config)
    sentinel = out[out["dense_frame_idx"] == 0]
    assert (sentinel["evidence_state_contact"] == EVIDENCE_UNOBSERVED_INVALID_ROUTE).all()
    assert sentinel["prob_contact"].isna().all()


def test_postural_approach_veto() -> None:
    frames = _prepared_frames()
    # Force rearing + partner behind on A→B frame 1 (approach positive).
    mask = (frames["focal_id"] == MOUSE_A) & (frames["dense_frame_idx"] == 1)
    frames.loc[mask, "body_contraction_rate_7"] = 6.0
    frames.loc[mask, "behind_score_bl_j"] = -0.9
    frames.loc[mask, "cos_bearing_i_to_j"] = -0.5
    frames.loc[mask, "rolling_mean_projected_vel_i_to_j_60"] = 0.0
    assert int(frames.loc[mask, "pred_approach"].iloc[0]) == 1

    out, diag = apply_postural_approach_veto(frames)
    assert diag.n_rows_changed >= 1
    assert int(out.loc[mask, "pred_approach"].iloc[0]) == 0
    assert float(out.loc[mask, "prob_approach"].iloc[0]) == 0.0


def test_postural_approach_escape_hatch() -> None:
    frames = _prepared_frames()
    mask = (frames["focal_id"] == MOUSE_A) & (frames["dense_frame_idx"] == 1)
    frames.loc[mask, "body_contraction_rate_7"] = 6.0
    frames.loc[mask, "behind_score_bl_j"] = -0.9
    frames.loc[mask, "rolling_mean_projected_vel_i_to_j_60"] = 2.0  # escape
    out, _diag = apply_postural_approach_veto(frames)
    # Escape protects the genuine-translation frame; smear may still touch
    # later frames in the same direction (legacy rolling-max behavior).
    assert int(out.loc[mask, "pred_approach"].iloc[0]) == 1
    assert float(out.loc[mask, "prob_approach"].iloc[0]) > 0.0


def test_postural_retract_stationary_veto() -> None:
    frames = _prepared_frames()
    mask = (frames["focal_id"] == MOUSE_A) & (frames["dense_frame_idx"] == 0)
    frames.loc[mask, "prob_retract"] = 0.9
    frames.loc[mask, "pred_retract"] = 1
    frames.loc[mask, "s7_prob_retract"] = 0.9
    frames.loc[mask, "s7_pred_retract"] = 1
    frames.loc[mask, "arbitration_input_valid_retract"] = True
    frames.loc[mask, "evidence_state_retract"] = "valid_positive"
    frames.loc[mask, "rolling_speed_i_15"] = 5.0  # stationary
    frames.loc[mask, "spine_directed_vel_15_i"] = 0.0  # no escape
    frames.loc[mask, "rolling_mean_projected_vel_combined_i_to_j_7"] = 0.0
    frames.loc[mask, "signed_kinematic_intent_nose_or_neck"] = 0.0

    out, diag = apply_postural_retract_veto(frames)
    assert diag.n_rows_changed >= 1
    assert int(out.loc[mask, "pred_retract"].iloc[0]) == 0


def test_feature_merge_no_extra_rows() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    features = make_s6_features()
    frames, _ = expand_dyadic_to_directional(dyadic, focal)
    n = len(frames)
    merged, _ = merge_s6_features(
        frames, features, feature_columns=list(REQUIRED_HARD_GAP_COLUMNS)
    )
    assert len(merged) == n
    assert "distance_nose_i_nose_j_norm" in merged.columns


def test_parallel_channels_preserved_after_gates() -> None:
    frames = _prepared_frames()
    config = load_behavioral_arbitration_config(None)
    out, _ = apply_hard_distance_probability_gates(frames, config)
    out, _ = apply_postural_approach_veto(out)
    out, _ = apply_postural_retract_veto(out)
    assert_parallel_channels_preserved(out)
    # Contact and SBS still independent columns (not folded into ethogram).
    assert "prob_contact" in out.columns and "prob_side_by_side" in out.columns
