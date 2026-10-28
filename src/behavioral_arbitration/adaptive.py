"""Stages 6–7 — adaptive approach / retract re-binarization.

Migrates ``_apply_adaptive_approach_thresholding`` and
``_apply_adaptive_retract_thresholding`` from Pipeline 100.

These overwrite arbitration ``pred_approach`` / ``pred_retract`` only; they do
not rewrite S7 provenance or ``behavior_label`` (Stage 11 sync is Sprint 3).
Invalid / NaN probabilities never trip strict or envelope thresholds.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.config import AdaptiveSettings
from behavioral_arbitration.contracts import (
    RULE_ADAPTIVE_APPROACH,
    RULE_ADAPTIVE_RETRACT,
)
from behavioral_arbitration.gates import _directional_rolling_max
from behavioral_arbitration.results import StageDiagnostic


def _prediction_flag(df: pd.DataFrame, detector: str) -> pd.Series:
    """Return binary ``pred_*`` flags; missing detector → 0."""

    pred_col = f"pred_{detector}"
    if pred_col not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)


def _no_contact_or_sniff(df: pd.DataFrame) -> pd.Series:
    return (
        (_prediction_flag(df, "contact") == 0)
        & (_prediction_flag(df, "nose_to_body") == 0)
        & (_prediction_flag(df, "nose_to_tail") == 0)
        & (_prediction_flag(df, "nose_to_nose") == 0)
    )


def apply_adaptive_approach_thresholding(
    df: pd.DataFrame,
    settings: AdaptiveSettings | None = None,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Boost flickering approach preds in close-but-not-contact context (Stage 6)."""

    cfg = settings or AdaptiveSettings()
    out = df.copy().reset_index(drop=True)
    empty = StageDiagnostic(
        stage="adaptive_approach",
        rule_code=RULE_ADAPTIVE_APPROACH,
        detector="approach",
        n_rows_changed=0,
        detail="skipped (prob_approach or distance absent)",
    )
    if (
        "prob_approach" not in out.columns
        or "distance_nose_i_spine_j_norm" not in out.columns
    ):
        return out, empty

    arb_col = "arbitration_input_valid_approach"
    observed = (
        out[arb_col].astype(bool)
        if arb_col in out.columns
        else out["prob_approach"].notna()
    )

    is_engaged = out["distance_nose_i_spine_j_norm"] < float(
        cfg.approach_engaged_distance
    )
    is_susceptible = is_engaged & _no_contact_or_sniff(out)

    # NaN probs do not contribute to the envelope as observed scores.
    prob_for_env = out["prob_approach"].astype(float).where(observed)
    prob_envelope = _directional_rolling_max(
        prob_for_env,
        out,
        window=int(cfg.window_frames),
        center=True,
    )

    y_strict = (out["prob_approach"].astype(float) >= float(cfg.strict_threshold)) & observed
    adaptive_mask = (
        is_susceptible
        & observed
        & (prob_envelope >= float(cfg.envelope_threshold))
    )
    y_final = (y_strict | adaptive_mask).astype(int)

    before = (
        pd.to_numeric(out.get("pred_approach", 0), errors="coerce").fillna(0).astype(int)
    )
    out["y_approach_smoothed"] = y_final
    if "pred_approach" in out.columns:
        out["pred_approach"] = y_final
    n_changed = int((before != y_final).sum())

    return out, StageDiagnostic(
        stage="adaptive_approach",
        rule_code=RULE_ADAPTIVE_APPROACH,
        detector="approach",
        n_rows_changed=n_changed,
        detail=(
            f"engaged_distance={cfg.approach_engaged_distance}, "
            f"strict={cfg.strict_threshold}, envelope={cfg.envelope_threshold}"
        ),
    )


def apply_adaptive_retract_thresholding(
    df: pd.DataFrame,
    settings: AdaptiveSettings | None = None,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Boost flickering retract preds with a wider engaged window (Stage 7).

    ``direction_window_frames`` exists in legacy but is unused — not invented here.
    """

    cfg = settings or AdaptiveSettings()
    out = df.copy().reset_index(drop=True)
    empty = StageDiagnostic(
        stage="adaptive_retract",
        rule_code=RULE_ADAPTIVE_RETRACT,
        detector="retract",
        n_rows_changed=0,
        detail="skipped (prob_retract or distance absent)",
    )
    if (
        "prob_retract" not in out.columns
        or "distance_nose_i_spine_j_norm" not in out.columns
    ):
        return out, empty

    arb_col = "arbitration_input_valid_retract"
    observed = (
        out[arb_col].astype(bool)
        if arb_col in out.columns
        else out["prob_retract"].notna()
    )

    engaged_dist = float(cfg.approach_engaged_distance) * float(
        cfg.retract_engaged_distance_scale
    )
    is_engaged = out["distance_nose_i_spine_j_norm"] < engaged_dist
    is_susceptible = is_engaged & _no_contact_or_sniff(out)

    prob_for_env = out["prob_retract"].astype(float).where(observed)
    prob_envelope = _directional_rolling_max(
        prob_for_env,
        out,
        window=int(cfg.window_frames),
        center=True,
    )

    y_strict = (out["prob_retract"].astype(float) >= float(cfg.strict_threshold)) & observed
    adaptive_mask = (
        is_susceptible
        & observed
        & (prob_envelope >= float(cfg.envelope_threshold))
    )
    y_final = (y_strict | adaptive_mask).astype(int)

    before = (
        pd.to_numeric(out.get("pred_retract", 0), errors="coerce").fillna(0).astype(int)
    )
    out["y_retract_smoothed"] = y_final
    if "pred_retract" in out.columns:
        out["pred_retract"] = y_final
    n_changed = int((before != y_final).sum())

    return out, StageDiagnostic(
        stage="adaptive_retract",
        rule_code=RULE_ADAPTIVE_RETRACT,
        detector="retract",
        n_rows_changed=n_changed,
        detail=(
            f"engaged_distance={engaged_dist}, "
            f"strict={cfg.strict_threshold}, envelope={cfg.envelope_threshold}"
        ),
    )
