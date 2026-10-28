"""Stage 8 contact cleaning + zero-gap policy + pass-2 interaction tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from behavioral_arbitration.config import (
    ContactPostProcessingSettings,
    load_behavioral_arbitration_config,
)
from behavioral_arbitration.contact_clean import (
    apply_contact_arbitration_no_remultiplex,
    get_cleaned_contact_bouts,
)
from behavioral_arbitration.contracts import EVIDENCE_UNOBSERVED_INVALID_ROUTE
from behavioral_arbitration.gates import apply_hard_distance_probability_gates
from behavioral_inference.contracts import SCORE_ORIGIN_ROUTING_SENTINEL


def _contact_series(
    probs: list[float],
    *,
    cc_count: int = 1,
    occlusion: int = 0,
    valid: list[bool] | None = None,
    focal: int = 1,
    partner: int = 2,
    spine_dist: float = 0.5,
) -> pd.DataFrame:
    valid = valid or [True] * len(probs)
    rows = []
    for i, (p, v) in enumerate(zip(probs, valid, strict=True)):
        rows.append(
            {
                "exp_name": "synth",
                "dense_frame_idx": i,
                "focal_id": focal,
                "partner_id": partner,
                "prob_contact": p,
                "pred_contact": 0,
                "prob_side_by_side": 0.01,
                "pred_side_by_side": 0,
                "cc_count": cc_count,
                "is_total_occlusion": occlusion,
                "distance_spine_i_spine_j_norm": spine_dist,
                "arbitration_input_valid_contact": v,
                "input_valid_contact": v,
                "score_origin_contact": (
                    "model_observed" if v else SCORE_ORIGIN_ROUTING_SENTINEL
                ),
                "evidence_state_contact": (
                    "valid_negative" if v else EVIDENCE_UNOBSERVED_INVALID_ROUTE
                ),
            }
        )
    return pd.DataFrame(rows)


def test_optical_assist_boosts_to_high_threshold() -> None:
    # Intact high = 0.9731; optical_assist_min = 0.30; cc_count==1.
    df = _contact_series([0.35] * 8, cc_count=1)
    cfg = ContactPostProcessingSettings()
    out = apply_contact_arbitration_no_remultiplex(df, cfg)
    assert float(out.loc[0, "prob_contact_fused"]) == cfg.prob_threshold_high_intact
    assert int(out.loc[0, "contact_optical_boost"]) == 1


def test_optical_assist_skipped_when_cc_count_not_one() -> None:
    df = _contact_series([0.35] * 4, cc_count=2)
    out = apply_contact_arbitration_no_remultiplex(df, ContactPostProcessingSettings())
    assert float(out.loc[0, "prob_contact_fused"]) == 0.35
    assert int(out.loc[0, "contact_optical_boost"]) == 0


def test_schmitt_hysteresis_transitions() -> None:
    # Force clear high then low trips without optical (cc_count=2).
    # High intact 0.9731, low 0.4.
    probs = [0.99, 0.99, 0.99, 0.50, 0.50, 0.10, 0.10, 0.10]
    df = _contact_series(probs, cc_count=2)
    cfg = replace(
        ContactPostProcessingSettings(),
        optical_assist_min=1.1,  # disable optical
        max_gap_to_fill=0,
        min_bout_duration=1,
        smoothing_window=1,
    )
    out = apply_contact_arbitration_no_remultiplex(df, cfg)
    # After high trip, mid 0.50 holds prior state (between low and high).
    assert float(out.loc[3, "is_contact_base"]) == 1.0
    # After low trip, state clears.
    assert float(out.loc[5, "is_contact_base"]) == 0.0


def test_max_gap_to_fill_zero_disables_morphological_close() -> None:
    # Schmitt positives with a 1-frame hole: W=0 must NOT fill.
    probs = [0.99, 0.99, 0.0, 0.99, 0.99, 0.99]
    df = _contact_series(probs, cc_count=2)
    cfg = replace(
        ContactPostProcessingSettings(),
        optical_assist_min=1.1,
        max_gap_to_fill=0,
        min_bout_duration=1,
        smoothing_window=1,
        prob_threshold_low=0.2,
    )
    out = apply_contact_arbitration_no_remultiplex(df, cfg)
    assert int(out["contact_morph_applied"].iloc[0]) == 0
    assert list(out["is_contact_gapped"].astype(int)) == list(
        out["is_contact_base"].astype(int)
    )
    # Hole remains (not a one-frame fill reinterpretation).
    assert int(out.loc[2, "is_contact_final"]) == 0


def test_morphological_closing_with_positive_window() -> None:
    probs = [0.99, 0.99, 0.0, 0.99, 0.99, 0.99]
    df = _contact_series(probs, cc_count=2)
    cfg = replace(
        ContactPostProcessingSettings(),
        optical_assist_min=1.1,
        max_gap_to_fill=3,
        min_bout_duration=1,
        smoothing_window=1,
        prob_threshold_low=0.2,
    )
    out = apply_contact_arbitration_no_remultiplex(df, cfg)
    assert int(out["contact_morph_applied"].iloc[0]) == 1
    assert int(out.loc[2, "is_contact_gapped"]) == 1


def test_no_remultiplex_even_if_specialist_columns_present() -> None:
    df = _contact_series([0.2] * 4, cc_count=1, occlusion=1)
    df["prob_contact_occluded"] = 0.9
    df["prob_contact_intact"] = 0.1
    cfg = replace(ContactPostProcessingSettings(), optical_assist_min=1.1)
    out = apply_contact_arbitration_no_remultiplex(df, cfg)
    # Fused contact must remain the S7 fused input (0.2), not occluded specialist.
    assert abs(float(out.loc[0, "prob_contact_fused"]) - 0.2) < 1e-12


def test_invalid_contact_sentinel_not_boosted_or_tripped() -> None:
    probs = [0.0, 0.0, 0.0, 0.0]
    df = _contact_series(probs, cc_count=1, valid=[False, False, False, False])
    out = apply_contact_arbitration_no_remultiplex(df, ContactPostProcessingSettings())
    assert np.isnan(out.loc[0, "prob_contact_fused"])
    assert int(out.loc[0, "contact_optical_boost"]) == 0
    assert int(out.loc[0, "is_contact_final"]) == 0


def test_cleaned_bouts_independent_of_sbs_and_pass2_gates() -> None:
    # Both directions; SBS stays put; pass-2 zeros far contact.
    ab = _contact_series([0.99] * 8, focal=1, partner=2, spine_dist=0.5)
    ba = _contact_series([0.99] * 8, focal=2, partner=1, spine_dist=2.0)
    df = pd.concat([ab, ba], ignore_index=True)
    cfg = replace(
        ContactPostProcessingSettings(),
        max_gap_to_fill=0,
        min_bout_duration=1,
        smoothing_window=1,
        optical_assist_min=1.1,
    )
    cleaned, diags = get_cleaned_contact_bouts(df, cfg)
    assert all(cleaned["pred_side_by_side"] == 0)
    assert "prob_side_by_side" in cleaned.columns
    assert any(d.rule_code == "CONTACT_SCHMITT" for d in diags)

    config = load_behavioral_arbitration_config(None)
    # Ensure arbitration_input_valid present for pass-2.
    gated, pass2 = apply_hard_distance_probability_gates(
        cleaned, config, pass_id=2
    )
    far = gated[(gated["focal_id"] == 2)]
    assert (far["pred_contact"] == 0).all()
    near = gated[(gated["focal_id"] == 1)]
    assert (near["pred_contact"] == 1).all()
    assert any(d.stage == "hard_distance_pass2" for d in pass2)


def test_min_bout_duration_drops_short_positives() -> None:
    probs = [0.99, 0.99, 0.0, 0.0, 0.0, 0.0]
    df = _contact_series(probs, cc_count=2)
    cfg = replace(
        ContactPostProcessingSettings(),
        optical_assist_min=1.1,
        max_gap_to_fill=0,
        min_bout_duration=3,
        smoothing_window=1,
        prob_threshold_low=0.2,
    )
    out = apply_contact_arbitration_no_remultiplex(df, cfg)
    assert int(out.loc[0, "is_contact_gapped"]) == 1
    assert int(out.loc[0, "is_contact_final"]) == 0
