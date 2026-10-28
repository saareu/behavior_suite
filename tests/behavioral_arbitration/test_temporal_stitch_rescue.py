"""Stages 12–14 / 16–17 — stitch, idle-retract, rescue, approach close gate."""

from __future__ import annotations

from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.temporal import (
    _split_working_thresholds,
    apply_approach_probability_gate,
    rescue_dynamic_arcs,
    stitch_contextual_engaged_bouts,
    stitch_dynamic_transitions,
    stitch_idle_retract_idle_bouts,
)
from tests.behavioral_arbitration._fixtures import make_directional_timeline


def test_contextual_stitch_same_sniff() -> None:
    config = load_behavioral_arbitration_config(None)
    focal, dyadic = _split_working_thresholds(config)
    # nose_to_body — short engaged island — nose_to_body
    labels = (
        ["nose_to_body"] * 4
        + ["engaged"] * 2
        + ["nose_to_body"] * 4
    )
    df = make_directional_timeline(labels)
    # Contextual stitch also requires cc_count==1 when column present.
    df["cc_count"] = 1
    out, diag = stitch_contextual_engaged_bouts(
        df,
        focal_thresholds=focal,
        dyadic_thresholds=dyadic,
        max_sniff_gap=3,
        max_dynamic_gap=3,
    )
    mid = out.iloc[4:6]
    assert (mid["behavior_label"] == "nose_to_body").all()
    assert (mid["label_reconstruction"] == "stitched").all()
    assert diag.n_rows_changed == 2
    assert (out["pred_nose_to_body"].iloc[4:6] == 1).all()


def test_dynamic_stitch_approach_engaged_approach() -> None:
    config = load_behavioral_arbitration_config(None)
    focal, dyadic = _split_working_thresholds(config)
    labels = ["approach"] * 3 + ["engaged"] * 2 + ["approach"] * 3
    df = make_directional_timeline(labels)
    out, diag = stitch_dynamic_transitions(
        df,
        focal_thresholds=focal,
        dyadic_thresholds=dyadic,
        max_padding_frames=5,
    )
    mid = out.iloc[3:5]
    assert (mid["behavior_label"] == "approach").all()
    assert diag.n_rows_changed == 2
    assert (out["pred_approach"].iloc[3:5] == 1).all()


def test_idle_retract_idle_collapses_short_retract() -> None:
    labels = (
        ["mutually_disengaged"] * 3
        + ["retract"] * 2
        + ["mutually_disengaged"] * 3
    )
    df = make_directional_timeline(labels)
    df["distance_spine_i_spine_j_norm"] = 0.5  # not far; short blip still collapses
    out, diag = stitch_idle_retract_idle_bouts(df, max_retract_gap_frames=3)
    mid = out.iloc[3:5]
    assert (mid["behavior_label"] == "mutually_disengaged").all()
    assert diag.n_rows_changed == 2
    assert (out["pred_retract"].iloc[3:5] == 0).all()


def test_rescue_approach_realm_from_engaged() -> None:
    # Engaged bout followed by approach; high intent/speed rescues trailing motion.
    labels = ["engaged"] * 8 + ["approach"] * 3
    df = make_directional_timeline(labels)
    df["prob_approach"] = 0.6
    df["signed_kinematic_intent"] = 0.0
    df["signed_kinematic_intent_nose_or_neck"] = 0.0
    df["rolling_speed_i_60"] = 40.0
    # Peak + contiguous moving tail inside engaged.
    df.loc[df.index[2:8], "signed_kinematic_intent"] = 2.0
    df.loc[df.index[2:8], "signed_kinematic_intent_nose_or_neck"] = 2.0
    df.loc[df.index[2:8], "rolling_speed_i_60"] = 250.0
    out, diag = rescue_dynamic_arcs(
        df,
        min_peak_intent=1.0,
        min_peak_speed=200.0,
        initiation_intent=0.5,
        initiation_speed=50.0,
        approach_prob_hint=0.4,
        moving_intent_abs=0.1,
        moving_smooth_frames=3,
    )
    rescued = out.iloc[:8]
    assert (rescued["behavior_label"] == "approach").any()
    assert diag.n_rows_changed > 0
    assert "rescued_approach=" in (diag.detail or "")


def test_approach_close_gate_zeros_when_too_close() -> None:
    df = make_directional_timeline(["approach"] * 5)
    df["prob_approach"] = 0.9
    df["pred_approach"] = 1
    df["distance_nose_i_nose_j_norm"] = [0.05, 0.05, 1.0, 1.0, 0.05]
    out, diag = apply_approach_probability_gate(
        df,
        nose_to_nose_max_dist=0.15,
        nose_to_body_max_dist=None,
        nose_to_tail_max_dist=None,
    )
    assert out.loc[0, "prob_approach"] == 0.0
    assert out.loc[0, "pred_approach"] == 0
    assert out.loc[2, "prob_approach"] == 0.9
    assert out.loc[2, "pred_approach"] == 1
    assert diag.n_rows_changed == 3
    # Validity / score origin untouched.
    assert "arbitration_input_valid_approach" in out.columns
    assert out["arbitration_input_valid_approach"].all()
