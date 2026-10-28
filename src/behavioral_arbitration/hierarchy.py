"""Stage 5 — hierarchical conflict resolution and temporal decay.

Migrates ``_resolve_hierarchical_conflicts`` from Pipeline 100 with S8
validity hardening (conflict C2/C3 / policy P3/P4):

- Invalid / routing-sentinel probs contribute margin 0 (never filled as 0.0).
- Valid residual with no Layer-1/2 winner remains legacy ``engaged``.
- Wholly unobserved hierarchy evidence yields provisional ``unresolved``
  (not silently ``engaged``); final publish undefined deferred to Sprint 3+.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.config import BehavioralArbitrationConfig
from behavioral_arbitration.contracts import (
    HIERARCHY_DETECTORS,
    HIERARCHY_LAYER1_DETECTORS,
    HIERARCHY_LAYER2_DETECTORS,
    LABEL_ENGAGED,
    LABEL_UNRESOLVED,
    RESIDUAL_ENGAGED_VALID,
    RESIDUAL_LAYER_WINNER,
    RESIDUAL_MD_BASELINE,
    RESIDUAL_UNRESOLVED_INVALID,
    RULE_HIERARCHY_DECAY,
    RULE_HIERARCHY_RESIDUAL,
    RULE_HIERARCHY_WIN,
)
from behavioral_arbitration.results import StageDiagnostic


def calculate_normalized_margin(prob: float, threshold: float) -> float:
    """``S = (P - T) / (1 - T)`` if ``P >= T`` else 0 (legacy formula)."""

    if not np.isfinite(prob) or not np.isfinite(threshold):
        return 0.0
    if prob >= threshold:
        denom = 1.0 - threshold
        if denom <= 0.0:
            return 0.0
        return (prob - threshold) / denom
    return 0.0


def resolve_threshold_value(
    threshold_spec: Any,
    *,
    stage: str | None = None,
    default: float = 0.5,
) -> float:
    if isinstance(threshold_spec, dict):
        if stage and stage in threshold_spec:
            return float(threshold_spec[stage])
        if "threshold" in threshold_spec:
            return float(threshold_spec["threshold"])
        return float(default)
    if threshold_spec is None:
        return float(default)
    return float(threshold_spec)


def resolve_threshold_for_row(
    threshold_spec: Any,
    row: pd.Series,
    *,
    default: float = 0.5,
) -> float:
    if not isinstance(threshold_spec, dict):
        return resolve_threshold_value(threshold_spec, default=default)

    if "is_total_occlusion" in row.index:
        if int(row.get("is_total_occlusion", 0) or 0) == 1:
            return resolve_threshold_value(
                threshold_spec, stage="occluded", default=default
            )
        return resolve_threshold_value(
            threshold_spec, stage="intact", default=default
        )

    return resolve_threshold_value(threshold_spec, stage="intact", default=default)


def _raw_prob_for_rank(df: pd.DataFrame, idx: int, det: str) -> float:
    """Return finite arbitration prob, or NaN when unobserved / missing."""

    arb_col = f"arbitration_input_valid_{det}"
    prob_col = f"prob_{det}"
    if arb_col in df.columns and not bool(df.at[idx, arb_col]):
        return float("nan")
    if prob_col not in df.columns:
        return float("nan")
    val = df.at[idx, prob_col]
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(fval):
        return float("nan")
    return fval


def _has_any_valid_hierarchy_evidence(df: pd.DataFrame, idx: int) -> bool:
    for det in HIERARCHY_DETECTORS:
        arb_col = f"arbitration_input_valid_{det}"
        if arb_col in df.columns and bool(df.at[idx, arb_col]):
            return True
        # Absent validity column → treat observed finite prob as valid (legacy).
        if arb_col not in df.columns:
            prob_col = f"prob_{det}"
            if prob_col in df.columns:
                try:
                    if np.isfinite(float(df.at[idx, prob_col])):
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def resolve_hierarchical_conflicts(
    df: pd.DataFrame,
    config: BehavioralArbitrationConfig,
) -> tuple[pd.DataFrame, list[StageDiagnostic]]:
    """Adjudicate exclusive ethogram winners with decay memory (Stage 5).

    Contact / side-by-side preds are untouched. Writes provisional
    ``behavior_label``, exclusive hierarchy ``pred_*``, optional decayed
    winner ``prob_*``, and residual diagnostics.
    """

    out = df.copy().reset_index(drop=True)
    sci = config.scientific
    layer1 = list(HIERARCHY_LAYER1_DETECTORS)
    layer2 = list(HIERARCHY_LAYER2_DETECTORS)
    hierarchy_dets = list(HIERARCHY_DETECTORS)
    thresholds = dict(config.working_thresholds)

    out["is_interpolated_by_decay"] = False
    out["pred_engaged"] = 0
    out["prob_engaged"] = 0.0
    out["behavior_label"] = LABEL_UNRESOLVED
    out["hierarchy_residual_kind"] = RESIDUAL_UNRESOLVED_INVALID
    out["hierarchy_layer"] = ""
    out["hierarchy_decay_carry"] = False

    if "exp_name" not in out.columns:
        out["exp_name"] = ""

    row_ids = (
        out["exp_name"].astype(str)
        + "_"
        + out["focal_id"].astype(str)
        + "_"
        + out["partner_id"].astype(str)
    )
    sort_order = np.lexsort(
        (out["dense_frame_idx"].to_numpy(), row_ids.to_numpy())
    )

    decay_rate = float(sci.decay_rate)
    decay_confirmation_frames = int(sci.conf_frames)
    engaged_min_threshold = float(sci.engaged_min_threshold)
    layer1_floor = float(sci.hierarchy.layer1_significance_floor)
    approach_penalty = float(sci.hierarchy.approach_contact_penalty)

    sniff_behaviors = {"nose_to_body", "nose_to_tail", "nose_to_nose"}
    memory_state: dict[str, dict[str, Any]] = {}

    n_layer_wins = 0
    n_decay_carry = 0
    n_engaged = 0
    n_md_baseline = 0
    n_unresolved = 0

    for idx in sort_order:
        rid = str(row_ids.iat[idx])
        state = memory_state.get(
            rid, {"winner": None, "winner_streak": 0, "winner_prob": 0.0}
        )

        raw_probs: dict[str, float] = {
            det: _raw_prob_for_rank(out, int(idx), det) for det in hierarchy_dets
        }
        eff_probs: dict[str, float] = {}
        row_margins: dict[str, float] = {}
        decay_used = False

        in_contact = (
            bool(out.at[idx, "cc_count"] == 1) if "cc_count" in out.columns else False
        )
        is_separated = (
            bool(out.at[idx, "cc_count"] == 2) if "cc_count" in out.columns else False
        )

        if is_separated and state["winner"] in sniff_behaviors:
            state = {"winner": None, "winner_streak": 0, "winner_prob": 0.0}

        row = out.iloc[idx]
        for det in hierarchy_dets:
            rank_prob = raw_probs[det]
            if np.isfinite(rank_prob) and det == "approach" and in_contact:
                rank_prob = max(0.0, rank_prob - approach_penalty)

            decayed_prob = 0.0
            if (
                state["winner"] == det
                and state["winner_streak"] >= decay_confirmation_frames
                and not (is_separated and det in sniff_behaviors)
            ):
                decayed_prob = float(state["winner_prob"]) * decay_rate

            if np.isfinite(rank_prob):
                eff = max(rank_prob, decayed_prob)
                if decayed_prob > rank_prob and decayed_prob > 0.0:
                    decay_used = True
            elif decayed_prob > 0.0:
                # Decay may bridge a transient missing observation for the
                # confirmed winner only; underlying invalidity mask is not erased.
                eff = decayed_prob
                decay_used = True
            else:
                eff = float("nan")

            eff_probs[det] = eff
            thresh_spec = thresholds.get(det, 0.5)
            thresh = resolve_threshold_for_row(thresh_spec, row, default=0.5)
            margin_prob = eff if np.isfinite(eff) else float("nan")
            row_margins[det] = calculate_normalized_margin(margin_prob, float(thresh))

        winner: str | None = None
        residual_kind = RESIDUAL_UNRESOLVED_INVALID
        layer = ""

        l1_winner, l1_max = None, layer1_floor
        for det in layer1:
            if row_margins.get(det, 0.0) > l1_max:
                l1_max = row_margins[det]
                l1_winner = det

        if l1_winner:
            winner = l1_winner
            residual_kind = RESIDUAL_LAYER_WINNER
            layer = "layer1"
        else:
            l2_winner, l2_max = None, 0.0
            for det in layer2:
                if row_margins.get(det, 0.0) > l2_max:
                    l2_max = row_margins[det]
                    l2_winner = det
            if l2_winner:
                winner = l2_winner
                residual_kind = RESIDUAL_LAYER_WINNER
                layer = "layer2"

        if winner is None:
            has_valid = _has_any_valid_hierarchy_evidence(out, int(idx))
            md_prob = raw_probs.get("mutually_disengaged", float("nan"))
            if (
                has_valid
                and np.isfinite(md_prob)
                and md_prob >= engaged_min_threshold
            ):
                winner = "mutually_disengaged"
                residual_kind = RESIDUAL_MD_BASELINE
                layer = "baseline"
            elif has_valid:
                winner = LABEL_ENGAGED
                residual_kind = RESIDUAL_ENGAGED_VALID
                layer = "baseline"
            else:
                winner = LABEL_UNRESOLVED
                residual_kind = RESIDUAL_UNRESOLVED_INVALID
                layer = "unresolved"

        out.at[idx, "behavior_label"] = winner
        out.at[idx, "hierarchy_residual_kind"] = residual_kind
        out.at[idx, "hierarchy_layer"] = layer
        out.at[idx, "hierarchy_decay_carry"] = bool(decay_used)
        if decay_used:
            out.at[idx, "is_interpolated_by_decay"] = True
            n_decay_carry += 1

        for det in hierarchy_dets:
            out.at[idx, f"pred_{det}"] = 1 if det == winner else 0
            if det == winner and np.isfinite(eff_probs.get(det, float("nan"))):
                out.at[idx, f"prob_{det}"] = eff_probs[det]

        if winner == LABEL_ENGAGED:
            out.at[idx, "pred_engaged"] = 1
            md_raw = raw_probs.get("mutually_disengaged", float("nan"))
            # Inverse confidence only when MD evidence is observed.
            if np.isfinite(md_raw):
                out.at[idx, "prob_engaged"] = 1.0 - md_raw
            else:
                out.at[idx, "prob_engaged"] = float("nan")
            n_engaged += 1
        else:
            out.at[idx, "pred_engaged"] = 0
            out.at[idx, "prob_engaged"] = 0.0

        if residual_kind == RESIDUAL_LAYER_WINNER:
            n_layer_wins += 1
        elif residual_kind == RESIDUAL_MD_BASELINE:
            n_md_baseline += 1
        elif residual_kind == RESIDUAL_UNRESOLVED_INVALID:
            n_unresolved += 1

        new_streak = (state["winner_streak"] + 1) if (winner == state["winner"]) else 1
        memory_state[rid] = {
            "winner": winner if winner not in (LABEL_ENGAGED, LABEL_UNRESOLVED) else None,
            "winner_streak": (
                new_streak if winner not in (LABEL_ENGAGED, LABEL_UNRESOLVED) else 0
            ),
            "winner_prob": (
                float(eff_probs.get(winner, 0.0))
                if winner in eff_probs and np.isfinite(eff_probs.get(winner, float("nan")))
                else 0.0
            ),
        }

    diagnostics = [
        StageDiagnostic(
            stage="hierarchy",
            rule_code=RULE_HIERARCHY_WIN,
            detector=None,
            n_rows_changed=n_layer_wins,
            detail=f"layer_wins={n_layer_wins}",
        ),
        StageDiagnostic(
            stage="hierarchy",
            rule_code=RULE_HIERARCHY_DECAY,
            detector=None,
            n_rows_changed=n_decay_carry,
            detail=f"decay_carry_frames={n_decay_carry}",
        ),
        StageDiagnostic(
            stage="hierarchy",
            rule_code=RULE_HIERARCHY_RESIDUAL,
            detector=None,
            n_rows_changed=n_engaged + n_md_baseline + n_unresolved,
            detail=(
                f"engaged_valid={n_engaged}, md_baseline={n_md_baseline}, "
                f"unresolved_invalid={n_unresolved}"
            ),
        ),
    ]
    return out, diagnostics
