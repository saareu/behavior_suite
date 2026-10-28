"""Stage 5 hierarchy / decay / residual policy tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.contracts import (
    EVIDENCE_UNOBSERVED_INVALID_ROUTE,
    LABEL_ENGAGED,
    LABEL_UNRESOLVED,
    RESIDUAL_ENGAGED_VALID,
    RESIDUAL_LAYER_WINNER,
    RESIDUAL_UNRESOLVED_INVALID,
)
from behavioral_arbitration.hierarchy import (
    calculate_normalized_margin,
    resolve_hierarchical_conflicts,
)
from behavioral_inference.contracts import (
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
)

HIERARCHY = (
    "nose_to_nose",
    "nose_to_body",
    "nose_to_tail",
    "approach",
    "retract",
    "mutually_disengaged",
)


def _base_row(
    *,
    frame: int,
    focal: int = 1,
    partner: int = 2,
    exp: str = "synth",
) -> dict:
    row: dict = {
        "exp_name": exp,
        "dense_frame_idx": frame,
        "focal_id": focal,
        "partner_id": partner,
        "cc_count": 2,
        "is_total_occlusion": 0,
    }
    for det in HIERARCHY:
        row[f"prob_{det}"] = 0.01
        row[f"pred_{det}"] = 0
        row[f"arbitration_input_valid_{det}"] = True
        row[f"input_valid_{det}"] = True
        row[f"score_origin_{det}"] = SCORE_ORIGIN_MODEL_OBSERVED
        row[f"evidence_state_{det}"] = "valid_negative"
    row["prob_contact"] = 0.01
    row["pred_contact"] = 0
    row["prob_side_by_side"] = 0.01
    row["pred_side_by_side"] = 0
    return row


def test_normalized_margin_legacy_formula() -> None:
    # Explicit numerical example: nose_to_nose T=0.0816, P=0.9
    s = calculate_normalized_margin(0.9, 0.0816)
    assert abs(s - (0.9 - 0.0816) / (1.0 - 0.0816)) < 1e-12
    assert calculate_normalized_margin(0.05, 0.0816) == 0.0
    assert calculate_normalized_margin(float("nan"), 0.5) == 0.0


def test_layer1_beats_layer2_and_tie_prefers_earlier() -> None:
    config = load_behavioral_arbitration_config(None)
    rows = [_base_row(frame=0), _base_row(frame=1)]
    # Frame 0: strong nose_to_nose and approach — Layer 1 wins.
    rows[0]["prob_nose_to_nose"] = 0.95
    rows[0]["prob_approach"] = 0.99
    # Frame 1: equal margins on nose_to_nose and nose_to_body — earlier wins.
    # Choose probs so S is identical above floor.
    # For T_nn=0.0816, T_nb_intact=0.6141: solve equal S is awkward; instead
    # give nose_to_nose slightly lower but still > floor and body higher — then
    # body wins. For tie: set both to same S via hand-picked values.
    rows[1]["prob_nose_to_nose"] = 0.20  # S≈0.129
    rows[1]["prob_nose_to_body"] = 0.20  # below NB threshold → S=0
    # Equal-S tie: force identical working thresholds, then earlier detector wins.
    config = replace(
        config,
        working_thresholds={
            "nose_to_nose": 0.5,
            "nose_to_body": 0.5,
            "nose_to_tail": 0.5,
            "approach": 0.5,
            "retract": 0.5,
            "mutually_disengaged": 0.5,
        },
    ).validated()
    rows[1]["prob_nose_to_nose"] = 0.80
    rows[1]["prob_nose_to_body"] = 0.80  # equal S; earlier (nn) wins

    out, _ = resolve_hierarchical_conflicts(pd.DataFrame(rows), config)
    assert out.loc[0, "behavior_label"] == "nose_to_nose"
    assert out.loc[0, "hierarchy_residual_kind"] == RESIDUAL_LAYER_WINNER
    assert int(out.loc[0, "pred_approach"]) == 0
    assert out.loc[1, "behavior_label"] == "nose_to_nose"


def test_decay_at_sequence_boundary_and_within_streak() -> None:
    config = load_behavioral_arbitration_config(None)
    # conf_frames=5, decay_rate=0.5. Sniff decay requires cc_count!=2
    # (legacy clears sniff memory when separated).
    rows = []
    for i in range(7):
        r = _base_row(frame=i)
        r["cc_count"] = 1
        r["prob_nose_to_nose"] = 0.90 if i < 5 else 0.01  # drop after confirm
        rows.append(r)
    # Separate direction must not inherit memory.
    other = _base_row(frame=0, focal=2, partner=1)
    other["cc_count"] = 1
    other["prob_nose_to_nose"] = 0.01
    rows.append(other)
    # Separated sequence: sniff memory must clear (no decay carry).
    for i in range(6):
        r = _base_row(frame=i, exp="sep")
        r["cc_count"] = 2
        r["prob_nose_to_nose"] = 0.90 if i < 5 else 0.01
        rows.append(r)

    out, diags = resolve_hierarchical_conflicts(pd.DataFrame(rows), config)
    # Frames 0-4: nose_to_nose winner; after streak>=5, frame 5 can decay-carry.
    ab = out[(out["exp_name"] == "synth") & (out["focal_id"] == 1)]
    assert all(ab.iloc[:5]["behavior_label"] == "nose_to_nose")
    row5 = ab[ab["dense_frame_idx"] == 5].iloc[0]
    assert row5["behavior_label"] == "nose_to_nose"
    assert bool(row5["hierarchy_decay_carry"]) is True
    # Effective prob = max(0.01, 0.90 * 0.5) = 0.45
    assert abs(float(row5["prob_nose_to_nose"]) - 0.45) < 1e-9
    # Reverse direction unaffected.
    rev = out[(out["exp_name"] == "synth") & (out["focal_id"] == 2)].iloc[0]
    assert rev["behavior_label"] == LABEL_ENGAGED
    # Separated streak does not decay-carry sniff after raw drop.
    sep_row = out[(out["exp_name"] == "sep") & (out["dense_frame_idx"] == 5)].iloc[0]
    assert sep_row["behavior_label"] != "nose_to_nose"
    assert bool(sep_row["hierarchy_decay_carry"]) is False
    assert any(d.rule_code == "DECAY_CARRY" and d.n_rows_changed >= 1 for d in diags)


def test_valid_engaged_residual_vs_unresolved_invalid() -> None:
    config = load_behavioral_arbitration_config(None)
    valid = _base_row(frame=0)
    invalid = _base_row(frame=1)
    for det in HIERARCHY:
        invalid[f"prob_{det}"] = np.nan
        invalid[f"arbitration_input_valid_{det}"] = False
        invalid[f"input_valid_{det}"] = False
        invalid[f"score_origin_{det}"] = SCORE_ORIGIN_ROUTING_SENTINEL
        invalid[f"evidence_state_{det}"] = EVIDENCE_UNOBSERVED_INVALID_ROUTE

    out, _ = resolve_hierarchical_conflicts(pd.DataFrame([valid, invalid]), config)
    assert out.loc[0, "behavior_label"] == LABEL_ENGAGED
    assert out.loc[0, "hierarchy_residual_kind"] == RESIDUAL_ENGAGED_VALID
    assert out.loc[1, "behavior_label"] == LABEL_UNRESOLVED
    assert out.loc[1, "hierarchy_residual_kind"] == RESIDUAL_UNRESOLVED_INVALID
    assert int(out.loc[1, "pred_engaged"]) == 0


def test_invalid_detector_does_not_win_via_zero_fill() -> None:
    config = load_behavioral_arbitration_config(None)
    row = _base_row(frame=0)
    # Sentinel contact-like: exported 0 but invalid — here on nose_to_nose.
    row["prob_nose_to_nose"] = np.nan
    row["arbitration_input_valid_nose_to_nose"] = False
    row["evidence_state_nose_to_nose"] = EVIDENCE_UNOBSERVED_INVALID_ROUTE
    # Valid approach below threshold → engaged residual, not nn.
    row["prob_approach"] = 0.10
    out, _ = resolve_hierarchical_conflicts(pd.DataFrame([row]), config)
    assert out.loc[0, "behavior_label"] == LABEL_ENGAGED
    assert int(out.loc[0, "pred_nose_to_nose"]) == 0


def test_contact_and_sbs_untouched_by_hierarchy() -> None:
    config = load_behavioral_arbitration_config(None)
    row = _base_row(frame=0)
    row["prob_contact"] = 0.77
    row["pred_contact"] = 1
    row["prob_side_by_side"] = 0.55
    row["pred_side_by_side"] = 1
    row["prob_nose_to_body"] = 0.95
    out, _ = resolve_hierarchical_conflicts(pd.DataFrame([row]), config)
    assert float(out.loc[0, "prob_contact"]) == 0.77
    assert int(out.loc[0, "pred_contact"]) == 1
    assert float(out.loc[0, "prob_side_by_side"]) == 0.55
    assert int(out.loc[0, "pred_side_by_side"]) == 1


def test_hierarchy_determinism_and_no_duplicate_keys() -> None:
    config = load_behavioral_arbitration_config(None)
    rows = [_base_row(frame=i, focal=f, partner=3 - f) for i in range(3) for f in (1, 2)]
    rows[0]["prob_approach"] = 0.9
    df = pd.DataFrame(rows)
    out1, _ = resolve_hierarchical_conflicts(df, config)
    out2, _ = resolve_hierarchical_conflicts(df, config)
    assert out1["behavior_label"].tolist() == out2["behavior_label"].tolist()
    assert not out1.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()
