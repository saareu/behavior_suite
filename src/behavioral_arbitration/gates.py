"""Initial Pipeline 100 scientific gates (Stages 3–4).

Hard-distance probability gates and postural approach/retract vetoes.
Formulas match ``legacy/pipelines/100_behavioral_arbiter.py`` exactly.
Evidence normalization (validity masking) is applied **before** these gates;
gates mutate only valid-evidence rows (conflict C2).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.config import (
    BehavioralArbitrationConfig,
    PosturalApproachSettings,
    PosturalRetractSettings,
)
from behavioral_arbitration.contracts import (
    PARALLEL_CHANNELS,
    RULE_GATE_HARD_DISTANCE,
    RULE_GATE_HARD_DISTANCE_PASS2,
    RULE_MERGE_S6,
    RULE_VETO_APPROACH,
    RULE_VETO_RETRACT,
)
from behavioral_arbitration.results import StageDiagnostic
from behavioral_arbitration.validity import mark_rule_vetoed

DIRECTIONAL_PAIR_COLS = ("exp_name", "focal_id", "partner_id")


def merge_s6_features(
    directional: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Left-join S6 feature columns onto the directional spine (Stage 2).

    Join keys: ``(exp_name, dense_frame_idx, focal_id, partner_id)``,
    ``how="left"``, ``validate="many_to_one"``. Does not create extra rows.
    """

    keys = ["exp_name", "dense_frame_idx", "focal_id", "partner_id"]
    for key in keys:
        if key not in directional.columns:
            raise ValueError(f"Directional table missing join key '{key}'.")
        if key not in features.columns:
            raise ValueError(f"S6 features missing join key '{key}'.")

    cols = [c for c in feature_columns if c in features.columns]
    if not cols:
        raise ValueError("No requested S6 feature columns present for merge.")

    feat = features.loc[:, list(keys) + cols].copy()
    if bool(feat.duplicated(subset=keys, keep=False).any()):
        raise ValueError(
            "S6 feature table contains duplicate directional keys for "
            "(exp_name, dense_frame_idx, focal_id, partner_id)."
        )

    n_before = len(directional)
    # Drop feature cols if already present to allow idempotent re-merge in tests.
    drop_existing = [c for c in cols if c in directional.columns]
    base = directional.drop(columns=drop_existing) if drop_existing else directional
    merged = base.merge(
        feat,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if len(merged) != n_before:
        raise ValueError(
            f"S6 feature merge changed row count ({n_before} → {len(merged)})."
        )

    diagnostic = StageDiagnostic(
        stage="merge_s6",
        rule_code=RULE_MERGE_S6,
        detector=None,
        n_rows_changed=0,
        detail=f"joined columns={cols}",
    )
    return merged, diagnostic


def apply_hard_distance_probability_gates(
    df: pd.DataFrame,
    config: BehavioralArbitrationConfig,
    *,
    pass_id: int = 1,
) -> tuple[pd.DataFrame, list[StageDiagnostic]]:
    """Hard-gate detector probs/preds when distance exceeds threshold.

    Pass 1 = Stage 3 (pre-hierarchy); pass 2 = Stage 9 (post-contact).

    Legacy: if distance present, non-null, and **>** threshold → ``prob=0``,
    ``pred=0``. NaN distances skip the gate. ``null`` threshold disables.

    S8: only rows with ``arbitration_input_valid_<det>`` are rewritten so
    unobserved sentinels are not converted into valid negatives.
    """

    sci = config.scientific
    gate_specs: list[tuple[str, str, float | None]] = [
        (
            "nose_to_nose",
            "distance_nose_i_nose_j_norm",
            sci.hard_gap_nose_to_nose_distance_norm,
        ),
        (
            "nose_to_body",
            "distance_nose_i_side_j_norm",
            sci.hard_gap_nose_to_body_distance_norm,
        ),
        (
            "nose_to_tail",
            "distance_nose_i_tail_j_norm",
            sci.hard_gap_nose_to_tail_distance_norm,
        ),
        (
            "side_by_side",
            "distance_spine_i_spine_j_norm",
            sci.hard_gap_side_by_side_distance_norm,
        ),
        (
            "contact",
            "distance_spine_i_spine_j_norm",
            sci.hard_gap_contact_spine_to_spine_distance_norm,
        ),
    ]

    out = df.copy().reset_index(drop=True)
    diagnostics: list[StageDiagnostic] = []
    stage_name = "hard_distance" if pass_id == 1 else "hard_distance_pass2"
    rule_code = (
        RULE_GATE_HARD_DISTANCE if pass_id == 1 else RULE_GATE_HARD_DISTANCE_PASS2
    )

    for detector, distance_col, max_dist in gate_specs:
        if max_dist is None:
            continue
        prob_col = f"prob_{detector}"
        if prob_col not in out.columns:
            continue
        if distance_col not in out.columns:
            continue

        arb_col = f"arbitration_input_valid_{detector}"
        gated_mask = out[distance_col].notna() & (out[distance_col] > float(max_dist))
        if arb_col in out.columns:
            gated_mask = gated_mask & out[arb_col].astype(bool)

        n_gated = int(gated_mask.sum())
        if n_gated == 0:
            continue

        out = mark_rule_vetoed(out, detector=detector, mask=gated_mask)
        diagnostics.append(
            StageDiagnostic(
                stage=stage_name,
                rule_code=rule_code,
                detector=detector,
                n_rows_changed=n_gated,
                detail=(
                    f"pass={pass_id}, threshold={float(max_dist):.3f}, "
                    f"col={distance_col}"
                ),
            )
        )

    return out, diagnostics


def _directional_rolling_max(
    series: pd.Series,
    df: pd.DataFrame,
    *,
    window: int,
    center: bool = False,
) -> pd.Series:
    """Apply rolling max chronologically within each directional pair."""

    required_cols = list(DIRECTIONAL_PAIR_COLS) + ["dense_frame_idx"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return series.rolling(window=window, min_periods=1, center=center).max()

    sorted_idx = df.sort_values(required_cols, kind="stable").index
    sorted_series = series.loc[sorted_idx]
    rolled = sorted_series.groupby(
        [df.loc[sorted_idx, c] for c in DIRECTIONAL_PAIR_COLS],
        sort=False,
    ).transform(
        lambda s: s.rolling(window=window, min_periods=1, center=center).max()
    )
    return rolled.reindex(series.index)


def apply_postural_approach_veto(
    df: pd.DataFrame,
    settings: PosturalApproachSettings | None = None,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Hard-gate approach for impossible postural states (Stage 4a).

    Legacy ``_apply_postural_approach_veto`` formulas unchanged.
    """

    cfg = settings or PosturalApproachSettings()
    out = df.copy().reset_index(drop=True)
    empty = StageDiagnostic(
        stage="postural_approach",
        rule_code=RULE_VETO_APPROACH,
        detector="approach",
        n_rows_changed=0,
        detail="skipped (prob_approach absent)",
    )
    if "prob_approach" not in out.columns:
        return out, empty

    # Missing columns: legacy .get defaults make flags false / safe.
    body = out.get(
        "body_contraction_rate_7",
        pd.Series(0.0, index=out.index),
    )
    behind = out.get("behind_score_bl_j", pd.Series(0.0, index=out.index))
    cos_b = out.get("cos_bearing_i_to_j", pd.Series(1.0, index=out.index))
    proj = out.get(
        "rolling_mean_projected_vel_i_to_j_60",
        pd.Series(0.0, index=out.index),
    )

    is_rearing = body > float(cfg.rearing_contraction_gt)
    is_partner_behind = (behind < float(cfg.behind_score_lt)) | (
        cos_b < float(cfg.cos_bearing_lt)
    )
    base_impossible = is_rearing & is_partner_behind
    smeared = _directional_rolling_max(
        base_impossible.astype(float),
        out,
        window=int(cfg.smear_frames),
    ).astype(bool)
    is_genuine = proj > float(cfg.escape_projected_vel_gt)
    is_vetoed = smeared & (~is_genuine)

    n_gated = int(is_vetoed.sum())
    if n_gated > 0:
        out = mark_rule_vetoed(out, detector="approach", mask=is_vetoed)

    return out, StageDiagnostic(
        stage="postural_approach",
        rule_code=RULE_VETO_APPROACH,
        detector="approach",
        n_rows_changed=n_gated,
        detail=None,
    )


def apply_postural_retract_veto(
    df: pd.DataFrame,
    settings: PosturalRetractSettings | None = None,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Hard-gate retract for impossible postural states (Stage 4b).

    Legacy ``_apply_postural_retract_veto`` formulas unchanged.
    """

    cfg = settings or PosturalRetractSettings()
    out = df.copy().reset_index(drop=True)
    empty = StageDiagnostic(
        stage="postural_retract",
        rule_code=RULE_VETO_RETRACT,
        detector="retract",
        n_rows_changed=0,
        detail="skipped (prob_retract absent)",
    )
    if "prob_retract" not in out.columns:
        return out, empty

    speed_i = out.get("rolling_speed_i_15", pd.Series(100.0, index=out.index))
    effort = out.get("focal_effort_ratio_15", pd.Series(1.0, index=out.index))
    speed_j = out.get("rolling_speed_j_15", pd.Series(0.0, index=out.index))
    body = out.get("body_contraction_rate_7", pd.Series(0.0, index=out.index))
    spine = out.get("spine_directed_vel_15_i", pd.Series(-100.0, index=out.index))
    combined = out.get(
        "rolling_mean_projected_vel_combined_i_to_j_7",
        pd.Series(0.0, index=out.index),
    )
    intent = out.get(
        "signed_kinematic_intent_nose_or_neck",
        pd.Series(0.0, index=out.index),
    )

    is_stationary = speed_i < float(cfg.stationary_speed_lt)
    is_passive = (effort < float(cfg.passive_effort_lt)) & (
        speed_j > float(cfg.passive_partner_speed_gt)
    )
    is_flinch = (body > float(cfg.flinch_contraction_gt)) & (
        spine > float(cfg.flinch_spine_vel_gt)
    )
    base_impossible = is_stationary | is_passive | is_flinch
    smeared = _directional_rolling_max(
        base_impossible.astype(float),
        out,
        window=int(cfg.smear_frames),
    ).astype(bool)
    is_genuine = (
        (spine < float(cfg.escape_spine_vel_lt))
        | (combined < float(cfg.escape_combined_vel_lt))
        | (intent < float(cfg.escape_intent_lt))
    )
    is_vetoed = smeared & (~is_genuine)

    n_gated = int(is_vetoed.sum())
    if n_gated > 0:
        out = mark_rule_vetoed(out, detector="retract", mask=is_vetoed)

    return out, StageDiagnostic(
        stage="postural_retract",
        rule_code=RULE_VETO_RETRACT,
        detector="retract",
        n_rows_changed=n_gated,
        detail=None,
    )


def assert_parallel_channels_preserved(frames: pd.DataFrame) -> None:
    """Contact / SBS remain present as parallel columns (not collapsed)."""

    for channel in PARALLEL_CHANNELS:
        if f"prob_{channel}" not in frames.columns:
            raise ValueError(f"Parallel channel prob_{channel} missing after gates.")
        if f"pred_{channel}" not in frames.columns:
            raise ValueError(f"Parallel channel pred_{channel} missing after gates.")
