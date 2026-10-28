"""Stages 6–7 adaptive approach / retract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from behavioral_arbitration.adaptive import (
    apply_adaptive_approach_thresholding,
    apply_adaptive_retract_thresholding,
)
from behavioral_arbitration.config import AdaptiveSettings


def _directional_frames(
    *,
    n: int = 5,
    approach_probs_ab: list[float] | None = None,
    approach_probs_ba: list[float] | None = None,
    retract_probs_ab: list[float] | None = None,
    retract_probs_ba: list[float] | None = None,
    dist_ab: float = 1.0,
    dist_ba: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(n):
        for focal, partner, dist, ap, rt in (
            (
                1,
                2,
                dist_ab,
                (approach_probs_ab or [0.0] * n)[i],
                (retract_probs_ab or [0.0] * n)[i],
            ),
            (
                2,
                1,
                dist_ba,
                (approach_probs_ba or [0.0] * n)[i],
                (retract_probs_ba or [0.0] * n)[i],
            ),
        ):
            rows.append(
                {
                    "exp_name": "synth",
                    "dense_frame_idx": i,
                    "focal_id": focal,
                    "partner_id": partner,
                    "distance_nose_i_spine_j_norm": dist,
                    "prob_approach": ap,
                    "pred_approach": 0,
                    "prob_retract": rt,
                    "pred_retract": 0,
                    "pred_contact": 0,
                    "pred_nose_to_body": 0,
                    "pred_nose_to_tail": 0,
                    "pred_nose_to_nose": 0,
                    "arbitration_input_valid_approach": np.isfinite(ap),
                    "arbitration_input_valid_retract": np.isfinite(rt),
                }
            )
    return pd.DataFrame(rows)


def test_adaptive_approach_envelope_both_directions() -> None:
    # Mid-frame weak approach (0.30) with neighbors at 0.40 → envelope ≥ 0.25.
    # Strict 0.5 fails; adaptive should fire when engaged and no sniff/contact.
    probs_ab = [0.40, 0.30, 0.40, 0.0, 0.0]
    probs_ba = [0.0, 0.0, 0.0, 0.0, 0.0]
    df = _directional_frames(
        approach_probs_ab=probs_ab,
        approach_probs_ba=probs_ba,
        dist_ab=1.0,
        dist_ba=10.0,  # reverse too far → not susceptible
    )
    out, diag = apply_adaptive_approach_thresholding(df, AdaptiveSettings())
    ab = out[(out["focal_id"] == 1) & (out["dense_frame_idx"] == 1)].iloc[0]
    ba = out[(out["focal_id"] == 2) & (out["dense_frame_idx"] == 1)].iloc[0]
    assert int(ab["pred_approach"]) == 1
    assert int(ab["y_approach_smoothed"]) == 1
    assert int(ba["pred_approach"]) == 0
    assert diag.rule_code == "ADAPTIVE_APPROACH"


def test_adaptive_retract_uses_wider_distance_window() -> None:
    # Approach engaged window is 3.0; retract uses 3.0 * 1.5 = 4.5.
    # Distance 4.0: approach not susceptible, retract is.
    df = _directional_frames(
        n=3,
        approach_probs_ab=[0.30, 0.30, 0.30],
        retract_probs_ab=[0.30, 0.30, 0.30],
        dist_ab=4.0,
    )
    out_a, _ = apply_adaptive_approach_thresholding(df, AdaptiveSettings())
    out_r, _ = apply_adaptive_retract_thresholding(df, AdaptiveSettings())
    mid_a = out_a[(out_a["focal_id"] == 1) & (out_a["dense_frame_idx"] == 1)].iloc[0]
    mid_r = out_r[(out_r["focal_id"] == 1) & (out_r["dense_frame_idx"] == 1)].iloc[0]
    assert int(mid_a["pred_approach"]) == 0
    assert int(mid_r["pred_retract"]) == 1


def test_adaptive_blocked_by_active_sniff_or_contact() -> None:
    df = _directional_frames(
        n=3,
        approach_probs_ab=[0.40, 0.30, 0.40],
        dist_ab=1.0,
    )
    df.loc[df["focal_id"] == 1, "pred_contact"] = 1
    out, _ = apply_adaptive_approach_thresholding(df, AdaptiveSettings())
    mid = out[(out["focal_id"] == 1) & (out["dense_frame_idx"] == 1)].iloc[0]
    assert int(mid["pred_approach"]) == 0


def test_adaptive_ignores_invalid_nan_probs() -> None:
    df = _directional_frames(
        n=3,
        approach_probs_ab=[0.40, float("nan"), 0.40],
        dist_ab=1.0,
    )
    out, _ = apply_adaptive_approach_thresholding(df, AdaptiveSettings())
    mid = out[(out["focal_id"] == 1) & (out["dense_frame_idx"] == 1)].iloc[0]
    assert int(mid["pred_approach"]) == 0


def test_strict_threshold_alone_sets_pred() -> None:
    df = _directional_frames(
        n=1,
        approach_probs_ab=[0.60],
        dist_ab=10.0,  # not susceptible for adaptive envelope
    )
    out, _ = apply_adaptive_approach_thresholding(df, AdaptiveSettings())
    assert int(out.loc[out["focal_id"] == 1, "pred_approach"].iloc[0]) == 1
