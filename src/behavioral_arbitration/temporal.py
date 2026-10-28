"""Stages 10–21 — temporal gap/min-duration, stitch, rescue, reciprocity, hysteresis.

Migrates Pipeline 100 helpers with identical formulas. Adds invalid-evidence
safeguards (never rewrite validity / evidence / score_origin / s7_* columns;
Stage 10 skips gap spans containing ``arbitration_input_valid_<det>==False``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.config import BehavioralArbitrationConfig
from behavioral_arbitration.contracts import (
    EXCLUSIVE_ETHOGRAM_LABELS,
    LABEL_ENGAGED,
    LABEL_UNDEFINED,
    LABEL_UNRESOLVED,
    REQUIRED_DYADIC_DETECTORS,
    REQUIRED_FOCAL_DETECTORS,
    RESIDUAL_UNRESOLVED_INVALID,
    RULE_GAP_FILL,
    RULE_GATE_APPROACH_CLOSE,
    RULE_HYSTERESIS_ENGAGED,
    RULE_LABEL_SYNC,
    RULE_MIN_DURATION,
    RULE_MIN_MD_LABEL,
    RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
    RULE_RECIPROCITY_NOSE_TO_NOSE,
    RULE_RESCUE_DYNAMIC,
    RULE_STITCH_CONTEXTUAL,
    RULE_STITCH_DYNAMIC,
    RULE_STITCH_IDLE_RETRACT,
)
from behavioral_arbitration.finalize import finalize_behavior_labels
from behavioral_arbitration.hierarchy import (
    calculate_normalized_margin,
    resolve_threshold_for_row,
)
from behavioral_arbitration.results import StageDiagnostic

DIRECTIONAL_PAIR_COLS = ["exp_name", "focal_id", "partner_id"]
_SYNC_PRIORITY = (
    "nose_to_nose",
    "nose_to_body",
    "nose_to_tail",
    "approach",
    "retract",
    "mutually_disengaged",
    "engaged",
)
_RECON_STITCHED = "stitched"
_RECON_RESCUED = "rescued"
_RECON_RECIPROCITY = "reciprocity_fallback"
_RECON_HYSTERESIS = "hysteresis"
_RECON_GAP_FILLED = "gap_filled"
_RECON_MIN_DURATION = "min_duration"
_RECON_SYNCED = "synced"


def fill_gaps(seq: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill gaps (0-regions) of length <= max_gap with 1s (legacy formula)."""

    result = np.asarray(seq, dtype=int).copy()
    i = 0
    while i < len(result):
        if result[i] == 1:
            j = i
            while j < len(result) and result[j] == 1:
                j += 1
            if j < len(result) and result[j] == 0:
                gap_start = j
                while j < len(result) and result[j] == 0:
                    j += 1
                gap_len = j - gap_start
                if gap_len <= max_gap and j < len(result) and result[j] == 1:
                    result[gap_start:j] = 1
            i = j
        else:
            i += 1
    return result


def apply_min_duration(seq: np.ndarray, min_duration: int) -> np.ndarray:
    """Remove bouts (1-regions) shorter than min_duration (legacy formula)."""

    result = np.asarray(seq, dtype=int).copy()
    i = 0
    while i < len(result):
        if result[i] == 1:
            j = i
            while j < len(result) and result[j] == 1:
                j += 1
            if (j - i) < min_duration:
                result[i:j] = 0
            i = j
        else:
            i += 1
    return result


def _fill_gaps_skipping_invalid(
    seq: np.ndarray,
    max_gap: int,
    invalid_mask: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Legacy ``fill_gaps`` with optional skip when a gap contains invalid frames."""

    result = np.asarray(seq, dtype=int).copy()
    skipped = 0
    if max_gap <= 0:
        return result, skipped
    i = 0
    while i < len(result):
        if result[i] == 1:
            j = i
            while j < len(result) and result[j] == 1:
                j += 1
            if j < len(result) and result[j] == 0:
                gap_start = j
                while j < len(result) and result[j] == 0:
                    j += 1
                gap_len = j - gap_start
                if gap_len <= max_gap and j < len(result) and result[j] == 1:
                    if (
                        invalid_mask is not None
                        and bool(np.any(invalid_mask[gap_start:j]))
                    ):
                        skipped += 1
                    else:
                        result[gap_start:j] = 1
            i = j
        else:
            i += 1
    return result, skipped


def get_kinematic_intent_series(
    df: pd.DataFrame,
    default_value: float = 0.0,
) -> pd.Series:
    """Prefer ``signed_kinematic_intent_nose_or_neck``, else ``signed_kinematic_intent``."""

    if "signed_kinematic_intent_nose_or_neck" in df.columns:
        return df["signed_kinematic_intent_nose_or_neck"]
    if "signed_kinematic_intent" in df.columns:
        return df["signed_kinematic_intent"]
    return pd.Series(default_value, index=df.index, dtype=float)


def _ensure_label_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    if "label_reconstruction" not in df.columns:
        df = df.copy()
        df["label_reconstruction"] = ""
    return df


def _mark_reconstruction(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    tag: str,
) -> None:
    if "label_reconstruction" not in df.columns:
        df["label_reconstruction"] = ""
    idx = df.index[np.asarray(mask, dtype=bool)]
    if len(idx):
        df.loc[idx, "label_reconstruction"] = tag


def _split_working_thresholds(
    config: BehavioralArbitrationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split flat ``working_thresholds`` into focal vs dyadic (legacy lookup order)."""

    wt = dict(config.working_thresholds)
    focal = {k: wt[k] for k in REQUIRED_FOCAL_DETECTORS if k in wt}
    dyadic = {k: wt[k] for k in REQUIRED_DYADIC_DETECTORS if k in wt}
    return focal, dyadic


def _threshold_for_label(
    label: str,
    row: pd.Series,
    focal_thresholds: dict[str, Any],
    dyadic_thresholds: dict[str, Any],
) -> float:
    threshold_spec = focal_thresholds.get(label, dyadic_thresholds.get(label, 0.5))
    return resolve_threshold_for_row(threshold_spec, row, default=0.5)


def _mean_normalized_margin(
    group: pd.DataFrame,
    label: str,
    focal_thresholds: dict[str, Any],
    dyadic_thresholds: dict[str, Any],
) -> float:
    prob_col = f"prob_{label}"
    if prob_col not in group.columns:
        return float("nan")
    margins: list[float] = []
    for _, row in group.iterrows():
        prob = float(row.get(prob_col, 0.0))
        thresh = _threshold_for_label(label, row, focal_thresholds, dyadic_thresholds)
        margins.append(calculate_normalized_margin(prob, thresh))
    if not margins:
        return float("nan")
    return float(np.nanmean(margins))


def fill_small_gaps_and_filter_duration(
    df: pd.DataFrame,
    gap_frames: int = 6,
    min_duration_frames: int = 12,
    min_duration_overrides: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[StageDiagnostic]]:
    """Stage 10 — gap fill + min-duration on every ``pred_*`` with invalid safeguards."""

    out = _ensure_label_reconstruction(df.reset_index(drop=True).copy())
    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    empty = [
        StageDiagnostic(
            stage="gap_min_duration",
            rule_code=RULE_GAP_FILL,
            detector=None,
            n_rows_changed=0,
            detail="no pred_* columns",
        ),
        StageDiagnostic(
            stage="gap_min_duration",
            rule_code=RULE_MIN_DURATION,
            detector=None,
            n_rows_changed=0,
            detail="no pred_* columns",
        ),
    ]
    if not pred_cols:
        return out, empty

    # Normalize pred dtypes up front so group write-back does not trip pandas
    # FutureWarnings when fixtures use int8 / boolean-like columns.
    for pred_col in pred_cols:
        out[pred_col] = out[pred_col].fillna(0).astype(np.int64)

    if "exp_name" not in out.columns:
        out["exp_name"] = ""

    out["_row_id"] = (
        out["exp_name"].astype(str)
        + "_"
        + out["focal_id"].astype(str)
        + "_"
        + out["partner_id"].astype(str)
    )

    n_gap_changed = 0
    n_min_changed = 0
    n_gaps_skipped_invalid = 0
    duration_overrides = min_duration_overrides or {}

    for _, pair_group in out.groupby("_row_id", sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        orig_index = group.index.to_numpy()
        group = group.reset_index(drop=True)

        for pred_col in pred_cols:
            if pred_col not in group.columns:
                continue
            detector_name = pred_col.replace("pred_", "", 1)
            detector_min_duration = int(
                duration_overrides.get(detector_name, min_duration_frames)
            )
            seq = group[pred_col].fillna(0).astype(int).to_numpy().copy()
            arb_col = f"arbitration_input_valid_{detector_name}"
            invalid_mask: np.ndarray | None = None
            if arb_col in group.columns:
                valid = group[arb_col].fillna(False).astype(bool).to_numpy()
                invalid_mask = ~valid
                # Sentinel zeros must not form positive bouts.
                seq[invalid_mask] = 0

            before = seq.copy()
            filled, skipped = _fill_gaps_skipping_invalid(
                seq, int(gap_frames), invalid_mask
            )
            n_gaps_skipped_invalid += int(skipped)
            gap_filled_mask = (filled == 1) & (before == 0)
            n_gap_changed += int(np.sum(gap_filled_mask))

            filtered = apply_min_duration(filled, detector_min_duration)
            min_changed = filtered != filled
            n_min_changed += int(np.sum(min_changed))

            recon = group["label_reconstruction"].astype(str).to_numpy().copy()
            recon[gap_filled_mask] = _RECON_GAP_FILLED
            recon[min_changed] = _RECON_MIN_DURATION
            group["label_reconstruction"] = recon
            group[pred_col] = filtered.astype(np.int64, copy=False)

        for pred_col in pred_cols:
            out.loc[orig_index, pred_col] = (
                group[pred_col].to_numpy(dtype=np.int64, copy=False)
            )
        out.loc[orig_index, "label_reconstruction"] = group[
            "label_reconstruction"
        ].to_numpy()

    if "_row_id" in out.columns:
        out = out.drop(columns=["_row_id"])

    diagnostics = [
        StageDiagnostic(
            stage="gap_min_duration",
            rule_code=RULE_GAP_FILL,
            detector=None,
            n_rows_changed=n_gap_changed,
            detail=(
                f"gap_frames={gap_frames}; "
                f"gaps_skipped_invalid={n_gaps_skipped_invalid}"
            ),
        ),
        StageDiagnostic(
            stage="gap_min_duration",
            rule_code=RULE_MIN_DURATION,
            detector=None,
            n_rows_changed=n_min_changed,
            detail=f"min_duration_frames={min_duration_frames}",
        ),
    ]
    return out, diagnostics


def synchronize_behavior_label_from_preds(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 11 / 18 — rebuild ``behavior_label`` from exclusive ``pred_*`` priority."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    before = (
        out["behavior_label"].astype(str)
        if "behavior_label" in out.columns
        else pd.Series("", index=out.index)
    )

    conditions: list[pd.Series] = []
    labels: list[str] = []
    for label in _SYNC_PRIORITY:
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            conditions.append(out[pred_col].fillna(0).astype(int).eq(1))
            labels.append(label)

    if conditions:
        any_pred = conditions[0].copy()
        for cond in conditions[1:]:
            any_pred = any_pred | cond
        out["behavior_label"] = np.select(conditions, labels, default=LABEL_ENGAGED)
        # P3/P4: do not silently promote wholly invalid evidence to engaged via
        # the legacy sync default. Preserve unresolved / undefined until finalize.
        preserve = ~any_pred
        if "hierarchy_residual_kind" in out.columns:
            preserve = preserve & (
                out["hierarchy_residual_kind"]
                .astype(str)
                .eq(RESIDUAL_UNRESOLVED_INVALID)
                | before.isin([LABEL_UNRESOLVED, LABEL_UNDEFINED])
            )
        else:
            preserve = preserve & before.isin([LABEL_UNRESOLVED, LABEL_UNDEFINED])
        if preserve.any():
            out.loc[preserve, "behavior_label"] = before.loc[preserve].where(
                before.loc[preserve].isin([LABEL_UNRESOLVED, LABEL_UNDEFINED]),
                LABEL_UNRESOLVED,
            )

    if "pred_engaged" in out.columns:
        out["pred_engaged"] = out["behavior_label"].eq(LABEL_ENGAGED).astype(int)
    if "prob_engaged" in out.columns and "prob_mutually_disengaged" in out.columns:
        out["prob_engaged"] = np.where(
            out["behavior_label"].eq(LABEL_ENGAGED),
            1.0 - out["prob_mutually_disengaged"].fillna(0.0),
            out["prob_engaged"],
        )

    changed = out["behavior_label"].astype(str) != before.astype(str)
    _mark_reconstruction(out, changed, _RECON_SYNCED)
    return out, StageDiagnostic(
        stage="label_sync",
        rule_code=RULE_LABEL_SYNC,
        detector=None,
        n_rows_changed=int(changed.sum()),
        detail=None,
    )


def sync_pred_columns_from_behavior_label(df: pd.DataFrame) -> pd.DataFrame:
    """Update exclusive ``pred_*`` columns to match ``behavior_label``."""

    out = df.copy()
    for label in _SYNC_PRIORITY:
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            out[pred_col] = out["behavior_label"].eq(label).astype(int)

    if "prob_engaged" in out.columns and "prob_mutually_disengaged" in out.columns:
        out["prob_engaged"] = np.where(
            out["behavior_label"].eq(LABEL_ENGAGED),
            1.0 - out["prob_mutually_disengaged"].fillna(0.0),
            out["prob_engaged"],
        )
    return out


def stitch_contextual_engaged_bouts(
    df: pd.DataFrame,
    focal_thresholds: dict[str, Any],
    dyadic_thresholds: dict[str, Any],
    max_sniff_gap: int,
    max_dynamic_gap: int,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 12 — stitch brief engaged islands from surrounding context."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    if "behavior_label" not in out.columns:
        return out, StageDiagnostic(
            stage="stitch_contextual",
            rule_code=RULE_STITCH_CONTEXTUAL,
            detector=None,
            n_rows_changed=0,
            detail="behavior_label missing",
        )

    sniff_labels = ["nose_to_body", "nose_to_tail", "nose_to_nose"]
    dynamic_labels = ["approach", "retract"]
    label_sets = {"sniff": set(sniff_labels), "dynamic": set(dynamic_labels)}
    stitched_frames = 0

    for _, pair_group in out.groupby(DIRECTIONAL_PAIR_COLS, sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        group["_orig_index"] = group.index
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        bout_id = (group["behavior_label"] != group["behavior_label"].shift()).cumsum()
        bout_ids = list(dict.fromkeys(bout_id.tolist()))

        for i, current_bout_id in enumerate(bout_ids):
            current_mask = bout_id == current_bout_id
            current_bout = group.loc[current_mask].copy()
            if current_bout.empty:
                continue
            if current_bout["behavior_label"].iloc[0] != LABEL_ENGAGED:
                continue
            if (
                "cc_count" in current_bout.columns
                and not current_bout["cc_count"].fillna(0).astype(int).eq(1).all()
            ):
                continue

            bout_len = int(len(current_bout))
            prev_label = (
                group.loc[bout_id == bout_ids[i - 1], "behavior_label"].iloc[0]
                if i > 0
                else None
            )
            next_label = (
                group.loc[bout_id == bout_ids[i + 1], "behavior_label"].iloc[0]
                if i + 1 < len(bout_ids)
                else None
            )
            if prev_label is None or next_label is None:
                continue

            replacement_label = None
            prev_is_sniff = prev_label in label_sets["sniff"]
            next_is_sniff = next_label in label_sets["sniff"]
            prev_is_dynamic = prev_label in label_sets["dynamic"]
            next_is_dynamic = next_label in label_sets["dynamic"]

            if (
                prev_label == next_label
                and (
                    (prev_is_sniff and bout_len <= max_sniff_gap)
                    or (prev_is_dynamic and bout_len <= max_dynamic_gap)
                )
            ):
                replacement_label = prev_label
            elif prev_is_sniff and next_is_sniff and bout_len <= max_sniff_gap:
                prev_margin = _mean_normalized_margin(
                    current_bout, prev_label, focal_thresholds, dyadic_thresholds
                )
                next_margin = _mean_normalized_margin(
                    current_bout, next_label, focal_thresholds, dyadic_thresholds
                )
                if not np.isnan(prev_margin) and not np.isnan(next_margin):
                    replacement_label = (
                        prev_label if prev_margin >= next_margin else next_label
                    )
            elif (prev_is_sniff and next_is_dynamic) or (
                prev_is_dynamic and next_is_sniff
            ):
                replacement_label = None

            if replacement_label is None:
                continue

            group.loc[current_mask, "behavior_label"] = replacement_label
            group.loc[current_mask, "label_reconstruction"] = _RECON_STITCHED
            stitched_frames += int(bout_len)

        out.loc[group["_orig_index"], "behavior_label"] = group["behavior_label"].values
        out.loc[group["_orig_index"], "label_reconstruction"] = group[
            "label_reconstruction"
        ].values

    stitch_labels = set(sniff_labels + dynamic_labels + ["engaged", "mutually_disengaged"])
    for label in stitch_labels:
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            out[pred_col] = (out["behavior_label"] == label).astype(int)

    return out, StageDiagnostic(
        stage="stitch_contextual",
        rule_code=RULE_STITCH_CONTEXTUAL,
        detector=None,
        n_rows_changed=stitched_frames,
        detail=f"stitched_frames={stitched_frames}",
    )


def stitch_dynamic_transitions(
    df: pd.DataFrame,
    focal_thresholds: dict[str, Any],
    dyadic_thresholds: dict[str, Any],
    max_padding_frames: int,
    min_intent_stitch: float = 0.5,
    min_speed_stitch: float = 25.0,
    *,
    look_ahead_frames: int = 15,
    hard_speed: float = 50.0,
    hard_intent_factor: float = 2.0,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 13 — stitch short engaged gaps bounded by approach/retract (full cases)."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    if "behavior_label" not in out.columns or max_padding_frames <= 0:
        return out, StageDiagnostic(
            stage="stitch_dynamic",
            rule_code=RULE_STITCH_DYNAMIC,
            detector=None,
            n_rows_changed=0,
            detail="skipped",
        )

    dynamic_labels = {"approach", "retract"}
    contact_labels = {"nose_to_nose", "nose_to_body", "nose_to_tail"}
    stitched_frames = 0
    max_acceleration_frames = int(look_ahead_frames)

    for _, pair_group in out.groupby(DIRECTIONAL_PAIR_COLS, sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        group["_orig_index"] = group.index
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        bout_id = (group["behavior_label"] != group["behavior_label"].shift()).cumsum()
        bout_ids = list(dict.fromkeys(bout_id.tolist()))

        for i, current_bout_id in enumerate(bout_ids):
            current_mask = bout_id == current_bout_id
            current_bout = group.loc[current_mask]

            if current_bout.empty or current_bout["behavior_label"].iloc[0] != LABEL_ENGAGED:
                continue

            bout_len = int(len(current_bout))
            if bout_len > max_padding_frames:
                continue

            prev_label = (
                group.loc[bout_id == bout_ids[i - 1], "behavior_label"].iloc[0]
                if i > 0
                else None
            )
            next_label = (
                group.loc[bout_id == bout_ids[i + 1], "behavior_label"].iloc[0]
                if i + 1 < len(bout_ids)
                else None
            )

            replacement_label = None

            # Case 1: dynamic -> engaged -> dynamic
            if prev_label in dynamic_labels and next_label in dynamic_labels:
                replacement_label = prev_label
                if prev_label != next_label:
                    prev_margin = _mean_normalized_margin(
                        current_bout, prev_label, focal_thresholds, dyadic_thresholds
                    )
                    next_margin = _mean_normalized_margin(
                        current_bout, next_label, focal_thresholds, dyadic_thresholds
                    )
                    if not np.isnan(prev_margin) and not np.isnan(next_margin):
                        replacement_label = (
                            prev_label if prev_margin >= next_margin else next_label
                        )

            # Case 2: contact -> engaged -> dynamic
            elif prev_label in contact_labels and next_label in dynamic_labels:
                replacement_label = next_label

            # Case 3: dynamic -> engaged -> contact (turnaround)
            elif next_label in contact_labels:
                if prev_label == "approach":
                    replacement_label = "approach"
                elif prev_label == "retract":
                    intent_nose = current_bout.get(
                        "signed_kinematic_intent",
                        pd.Series([0.0] * bout_len, index=current_bout.index),
                    )
                    intent_neck = current_bout.get(
                        "signed_kinematic_intent_neck",
                        pd.Series([0.0] * bout_len, index=current_bout.index),
                    )
                    composite_intent = np.maximum(
                        np.asarray(intent_nose, dtype=float),
                        np.asarray(intent_neck, dtype=float),
                    )
                    forward_frames = np.where(composite_intent >= min_intent_stitch)[0]
                    if len(forward_frames) > 0:
                        trigger_rel_idx = int(forward_frames[0])
                        trigger_idx = current_bout.index[trigger_rel_idx]
                        approach_indices = current_bout.loc[trigger_idx:].index
                        group.loc[approach_indices, "behavior_label"] = "approach"
                        group.loc[approach_indices, "label_reconstruction"] = (
                            _RECON_STITCHED
                        )
                        stitched_frames += len(approach_indices)
                        continue
                    replacement_label = "retract"
                elif prev_label in contact_labels:
                    replacement_label = None
                else:
                    replacement_label = None

            # Case 4: retract -> engaged -> idle
            elif prev_label == "retract" and next_label == "mutually_disengaged":
                replacement_label = "mutually_disengaged"

            # Case 5: hit-and-run (approach -> engaged -> retract)
            elif prev_label == "approach" and next_label == "retract":
                if bout_len <= (max_padding_frames * 2):
                    prev_margin = _mean_normalized_margin(
                        current_bout, "approach", focal_thresholds, dyadic_thresholds
                    )
                    next_margin = _mean_normalized_margin(
                        current_bout, "retract", focal_thresholds, dyadic_thresholds
                    )
                    if not np.isnan(prev_margin) and not np.isnan(next_margin):
                        replacement_label = (
                            "approach" if prev_margin >= next_margin else "retract"
                        )
                    elif not np.isnan(prev_margin):
                        replacement_label = "approach"
                    elif not np.isnan(next_margin):
                        replacement_label = "retract"

            # Case 6: bait-and-switch (retract -> engaged -> approach)
            elif prev_label == "retract" and next_label == "approach":
                if bout_len <= (max_padding_frames * 2):
                    intent_nose = current_bout.get(
                        "signed_kinematic_intent",
                        pd.Series([0.0] * bout_len, index=current_bout.index),
                    )
                    intent_neck = current_bout.get(
                        "signed_kinematic_intent_neck",
                        pd.Series([0.0] * bout_len, index=current_bout.index),
                    )
                    composite_intent = (
                        np.asarray(intent_nose, dtype=float)
                        + np.asarray(intent_neck, dtype=float)
                    ) / 2.0
                    forward_frames = np.where(composite_intent >= 0.0)[0]
                    if len(forward_frames) > 0:
                        pivot_rel_idx = int(forward_frames[0])
                        pivot_idx = current_bout.index[pivot_rel_idx]
                        retract_indices = current_bout.loc[:pivot_idx].index[:-1]
                        approach_indices = current_bout.loc[pivot_idx:].index
                        if len(retract_indices) > 0:
                            group.loc[retract_indices, "behavior_label"] = "retract"
                            group.loc[retract_indices, "label_reconstruction"] = (
                                _RECON_STITCHED
                            )
                            stitched_frames += len(retract_indices)
                        if len(approach_indices) > 0:
                            group.loc[approach_indices, "behavior_label"] = "approach"
                            group.loc[approach_indices, "label_reconstruction"] = (
                                _RECON_STITCHED
                            )
                            stitched_frames += len(approach_indices)
                        continue
                    replacement_label = "retract"

            # Kinematic trigger points
            if replacement_label == "approach" and prev_label != "approach":
                intent_nose = current_bout.get(
                    "signed_kinematic_intent",
                    pd.Series([-100.0] * bout_len, index=current_bout.index),
                )
                intent_neck = current_bout.get(
                    "signed_kinematic_intent_neck",
                    pd.Series([-100.0] * bout_len, index=current_bout.index),
                )
                speed_col = np.asarray(
                    current_bout.get(
                        "rolling_speed_i_60",
                        pd.Series([0.0] * bout_len, index=current_bout.index),
                    ),
                    dtype=float,
                )
                composite_intent = np.maximum(
                    np.asarray(intent_nose, dtype=float),
                    np.asarray(intent_neck, dtype=float),
                )
                hard_thresh = max(1.0, float(min_intent_stitch) * float(hard_intent_factor))
                soft_hits = np.where(
                    (composite_intent >= min_intent_stitch)
                    & (speed_col >= min_speed_stitch)
                )[0]
                trigger_idx = None
                for soft_idx in soft_hits:
                    look_ahead_end = min(int(soft_idx) + max_acceleration_frames, bout_len)
                    future_intent = composite_intent[soft_idx:look_ahead_end]
                    future_speed = speed_col[soft_idx:look_ahead_end]
                    if np.any(
                        (future_intent >= hard_thresh)
                        & (future_speed >= float(hard_speed))
                    ):
                        trigger_idx = current_bout.index[int(soft_idx)]
                        break
                if trigger_idx is not None:
                    approach_indices = current_bout.loc[trigger_idx:].index
                    group.loc[approach_indices, "behavior_label"] = "approach"
                    group.loc[approach_indices, "label_reconstruction"] = _RECON_STITCHED
                    stitched_frames += len(approach_indices)

            elif replacement_label == "retract" and prev_label != "retract":
                intent_nose = current_bout.get(
                    "signed_kinematic_intent",
                    pd.Series([100.0] * bout_len, index=current_bout.index),
                )
                intent_neck = current_bout.get(
                    "signed_kinematic_intent_neck",
                    pd.Series([100.0] * bout_len, index=current_bout.index),
                )
                speed_col = np.asarray(
                    current_bout.get(
                        "rolling_speed_i_60",
                        pd.Series([0.0] * bout_len, index=current_bout.index),
                    ),
                    dtype=float,
                )
                composite_neg_intent = np.minimum(
                    np.asarray(intent_nose, dtype=float),
                    np.asarray(intent_neck, dtype=float),
                )
                hard_thresh_neg = min(
                    -1.0, -float(min_intent_stitch) * float(hard_intent_factor)
                )
                soft_hits = np.where(
                    (composite_neg_intent <= -min_intent_stitch)
                    & (speed_col >= min_speed_stitch)
                )[0]
                trigger_idx = None
                for soft_idx in soft_hits:
                    look_ahead_end = min(int(soft_idx) + max_acceleration_frames, bout_len)
                    future_intent = composite_neg_intent[soft_idx:look_ahead_end]
                    future_speed = speed_col[soft_idx:look_ahead_end]
                    if np.any(
                        (future_intent <= hard_thresh_neg)
                        & (future_speed >= float(hard_speed))
                    ):
                        trigger_idx = current_bout.index[int(soft_idx)]
                        break
                if trigger_idx is not None:
                    retract_indices = current_bout.loc[trigger_idx:].index
                    group.loc[retract_indices, "behavior_label"] = "retract"
                    group.loc[retract_indices, "label_reconstruction"] = _RECON_STITCHED
                    stitched_frames += len(retract_indices)

            elif replacement_label is not None:
                group.loc[current_mask, "behavior_label"] = replacement_label
                group.loc[current_mask, "label_reconstruction"] = _RECON_STITCHED
                stitched_frames += bout_len

        out.loc[group["_orig_index"], "behavior_label"] = group["behavior_label"].values
        out.loc[group["_orig_index"], "label_reconstruction"] = group[
            "label_reconstruction"
        ].values

    sync_labels = dynamic_labels.union(contact_labels).union(
        {"engaged", "mutually_disengaged"}
    )
    for label in sync_labels:
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            out[pred_col] = (out["behavior_label"] == label).astype(int)

    return out, StageDiagnostic(
        stage="stitch_dynamic",
        rule_code=RULE_STITCH_DYNAMIC,
        detector=None,
        n_rows_changed=stitched_frames,
        detail=f"stitched_frames={stitched_frames}",
    )


def stitch_idle_retract_idle_bouts(
    df: pd.DataFrame,
    max_retract_gap_frames: int,
    far_apart_distance: float = 3.5,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 14 — collapse short retract blips into mutually_disengaged."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    if "behavior_label" not in out.columns or max_retract_gap_frames <= 0:
        return out, StageDiagnostic(
            stage="stitch_idle_retract",
            rule_code=RULE_STITCH_IDLE_RETRACT,
            detector=None,
            n_rows_changed=0,
            detail="skipped",
        )

    stitched_frames = 0
    for _, pair_group in out.groupby(DIRECTIONAL_PAIR_COLS, sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        group["_orig_index"] = group.index
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        bout_id = (group["behavior_label"] != group["behavior_label"].shift()).cumsum()
        bout_ids = list(dict.fromkeys(bout_id.tolist()))

        for i, current_bout_id in enumerate(bout_ids):
            current_mask = bout_id == current_bout_id
            current_bout = group.loc[current_mask]
            if current_bout.empty or current_bout["behavior_label"].iloc[0] != "retract":
                continue

            bout_len = int(len(current_bout))
            if bout_len > max_retract_gap_frames:
                continue

            prev_label = (
                group.loc[bout_id == bout_ids[i - 1], "behavior_label"].iloc[0]
                if i > 0
                else None
            )
            next_label = (
                group.loc[bout_id == bout_ids[i + 1], "behavior_label"].iloc[0]
                if i + 1 < len(bout_ids)
                else None
            )

            if prev_label == "mutually_disengaged":
                is_short_blip = (bout_len <= max_retract_gap_frames) and (
                    next_label == "mutually_disengaged"
                )
                dist_col = current_bout.get(
                    "distance_spine_i_spine_j_norm",
                    pd.Series([0.0]),
                )
                is_far_apart = float(dist_col.min()) > float(far_apart_distance)
                if is_short_blip or is_far_apart:
                    group.loc[current_mask, "behavior_label"] = "mutually_disengaged"
                    group.loc[current_mask, "label_reconstruction"] = _RECON_STITCHED
                    stitched_frames += bout_len

        out.loc[group["_orig_index"], "behavior_label"] = group["behavior_label"].values
        out.loc[group["_orig_index"], "label_reconstruction"] = group[
            "label_reconstruction"
        ].values

    for label in ["retract", "mutually_disengaged"]:
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            out[pred_col] = (out["behavior_label"] == label).astype(int)

    return out, StageDiagnostic(
        stage="stitch_idle_retract",
        rule_code=RULE_STITCH_IDLE_RETRACT,
        detector=None,
        n_rows_changed=stitched_frames,
        detail=f"stitched_frames={stitched_frames}",
    )


def enforce_min_behavior_label_duration(
    df: pd.DataFrame,
    *,
    label_name: str,
    min_frames: int,
    replacement_label: str = LABEL_ENGAGED,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 15 — replace short ``behavior_label`` bouts with a fallback label."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    if "behavior_label" not in out.columns or min_frames <= 1:
        return out, StageDiagnostic(
            stage="min_md_label",
            rule_code=RULE_MIN_MD_LABEL,
            detector=label_name,
            n_rows_changed=0,
            detail="skipped",
        )

    replaced_frames = 0
    for _, pair_group in out.groupby(DIRECTIONAL_PAIR_COLS, sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        group["_orig_index"] = group.index
        group = group.reset_index(drop=True)

        bout_id = (group["behavior_label"] != group["behavior_label"].shift()).cumsum()
        for current_bout_id in bout_id.unique():
            mask = bout_id == current_bout_id
            bout = group.loc[mask]
            if bout.empty:
                continue
            if bout["behavior_label"].iloc[0] != label_name:
                continue
            if int(len(bout)) >= int(min_frames):
                continue
            group.loc[mask, "behavior_label"] = replacement_label
            group.loc[mask, "label_reconstruction"] = _RECON_MIN_DURATION
            replaced_frames += int(len(bout))

        out.loc[group["_orig_index"], "behavior_label"] = group["behavior_label"].values
        out.loc[group["_orig_index"], "label_reconstruction"] = group[
            "label_reconstruction"
        ].values

    return out, StageDiagnostic(
        stage="min_md_label",
        rule_code=RULE_MIN_MD_LABEL,
        detector=label_name,
        n_rows_changed=replaced_frames,
        detail=f"replaced_frames={replaced_frames}",
    )


def rescue_dynamic_arcs(
    df: pd.DataFrame,
    min_peak_intent: float = 1.0,
    min_peak_speed: float = 200.0,
    initiation_intent: float = 0.5,
    initiation_speed: float = 50.0,
    approach_prob_hint: float = 0.4,
    retract_prob_hint: float = 0.4,
    moving_intent_abs: float = 0.1,
    moving_smooth_frames: int = 12,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 16 — rescue high-momentum approach/retract arcs from engaged gaps."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    if "behavior_label" not in out.columns:
        return out, StageDiagnostic(
            stage="rescue_dynamic",
            rule_code=RULE_RESCUE_DYNAMIC,
            detector=None,
            n_rows_changed=0,
            detail="behavior_label missing",
        )

    rescued_app_frames = 0
    rescued_ret_frames = 0
    smooth_window = max(1, int(moving_smooth_frames))
    intent_abs = float(moving_intent_abs)

    for _, pair_group in out.groupby(DIRECTIONAL_PAIR_COLS, sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        group["_orig_index"] = group.index
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        bout_id = (group["behavior_label"] != group["behavior_label"].shift()).cumsum()
        bout_ids = list(dict.fromkeys(bout_id.tolist()))

        for i, current_bout_id in enumerate(bout_ids):
            current_mask = bout_id == current_bout_id
            current_bout = group.loc[current_mask]
            if current_bout.empty or current_bout["behavior_label"].iloc[0] != LABEL_ENGAGED:
                continue

            next_label = (
                group.loc[bout_id == bout_ids[i + 1], "behavior_label"].iloc[0]
                if i + 1 < len(bout_ids)
                else None
            )

            prob_app_col = current_bout.get(
                "prob_approach",
                pd.Series([0.0] * len(current_bout), index=current_bout.index),
            )
            prob_ret_col = current_bout.get(
                "prob_retract",
                pd.Series([0.0] * len(current_bout), index=current_bout.index),
            )
            max_app_prob = float(prob_app_col.max())
            max_ret_prob = float(prob_ret_col.max())

            is_approach_realm = False
            is_retract_realm = False
            if next_label == "approach":
                is_approach_realm = True
            elif next_label == "retract":
                is_retract_realm = True
            elif max_app_prob >= approach_prob_hint and max_app_prob >= max_ret_prob:
                is_approach_realm = True
            elif max_ret_prob >= retract_prob_hint and max_ret_prob > max_app_prob:
                is_retract_realm = True

            if not is_approach_realm and not is_retract_realm:
                continue

            intent_col = get_kinematic_intent_series(current_bout, default_value=0.0)
            speed_col = current_bout.get(
                "rolling_speed_i_60",
                pd.Series([0.0] * len(current_bout), index=current_bout.index),
            )

            if is_approach_realm:
                if (
                    float(intent_col.max()) > float(min_peak_intent)
                    and float(speed_col.max()) > float(min_peak_speed)
                ):
                    raw_moving = (intent_col > intent_abs) & (
                        speed_col > float(initiation_speed)
                    )
                    is_moving = (
                        raw_moving.rolling(
                            window=smooth_window, min_periods=1, center=True
                        )
                        .max()
                        .astype(bool)
                    )
                    contiguous_moving_mask = is_moving.iloc[::-1].cummin().iloc[::-1]
                    if contiguous_moving_mask.any():
                        app_indices = current_bout[contiguous_moving_mask].index
                        group.loc[app_indices, "behavior_label"] = "approach"
                        group.loc[app_indices, "label_reconstruction"] = _RECON_RESCUED
                        rescued_app_frames += len(app_indices)
                continue

            if is_retract_realm and (
                float(intent_col.min()) < -float(min_peak_intent)
                and float(speed_col.max()) > float(min_peak_speed)
            ):
                raw_moving = (intent_col < -intent_abs) & (
                    speed_col > float(initiation_speed)
                )
                is_moving = (
                    raw_moving.rolling(
                        window=smooth_window, min_periods=1, center=True
                    )
                    .max()
                    .astype(bool)
                )
                contiguous_moving_mask = is_moving.iloc[::-1].cummin().iloc[::-1]
                if contiguous_moving_mask.any():
                    ret_indices = current_bout[contiguous_moving_mask].index
                    group.loc[ret_indices, "behavior_label"] = "retract"
                    group.loc[ret_indices, "label_reconstruction"] = _RECON_RESCUED
                    rescued_ret_frames += len(ret_indices)

        out.loc[group["_orig_index"], "behavior_label"] = group["behavior_label"].values
        out.loc[group["_orig_index"], "label_reconstruction"] = group[
            "label_reconstruction"
        ].values

    for label in EXCLUSIVE_ETHOGRAM_LABELS:
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            out[pred_col] = (out["behavior_label"] == label).astype(int)

    n_changed = rescued_app_frames + rescued_ret_frames
    return out, StageDiagnostic(
        stage="rescue_dynamic",
        rule_code=RULE_RESCUE_DYNAMIC,
        detector=None,
        n_rows_changed=n_changed,
        detail=(
            f"rescued_approach={rescued_app_frames}; "
            f"rescued_retract={rescued_ret_frames}"
        ),
    )


def apply_approach_probability_gate(
    df: pd.DataFrame,
    *,
    nose_to_nose_max_dist: float | None,
    nose_to_body_max_dist: float | None,
    nose_to_tail_max_dist: float | None,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 17 — hard-gate approach when dyad is too close on NN/NB/NT distances.

    Zeros ``prob_approach`` / ``pred_approach`` on gated rows only. Does not
    clear validity masks or invent ``evidence_state`` valid_negative.
    """

    out = df.copy().reset_index(drop=True)
    empty = StageDiagnostic(
        stage="gate_approach_close",
        rule_code=RULE_GATE_APPROACH_CLOSE,
        detector="approach",
        n_rows_changed=0,
        detail="skipped (prob_approach absent or no rows gated)",
    )
    if "prob_approach" not in out.columns:
        return out, empty

    gate_specs = [
        ("distance_nose_i_nose_j_norm", nose_to_nose_max_dist),
        ("distance_nose_i_side_j_norm", nose_to_body_max_dist),
        ("distance_nose_i_tail_j_norm", nose_to_tail_max_dist),
    ]
    gated_mask = pd.Series(False, index=out.index)
    used_specs: list[str] = []
    for distance_col, threshold in gate_specs:
        if threshold is None or not np.isfinite(threshold) or distance_col not in out.columns:
            continue
        used_specs.append(f"{distance_col}<{float(threshold):.3f}")
        gated_mask = gated_mask | (
            out[distance_col].notna() & (out[distance_col] < float(threshold))
        )

    gated_rows = int(gated_mask.sum())
    if gated_rows == 0:
        return out, empty

    out.loc[gated_mask, "prob_approach"] = 0.0
    if "pred_approach" in out.columns:
        out.loc[gated_mask, "pred_approach"] = 0

    return out, StageDiagnostic(
        stage="gate_approach_close",
        rule_code=RULE_GATE_APPROACH_CLOSE,
        detector="approach",
        n_rows_changed=gated_rows,
        detail=f"criteria={', '.join(used_specs)}",
    )


def enforce_nose_to_nose_reciprocity(
    df: pd.DataFrame,
    *,
    fallback_label: str = LABEL_ENGAGED,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 19 — require both directions of a dyad to be nose_to_nose."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    key_cols = ["exp_name", "dense_frame_idx", "focal_id", "partner_id"]
    if any(c not in out.columns for c in key_cols + ["behavior_label"]):
        return out, StageDiagnostic(
            stage="reciprocity_nose_to_nose",
            rule_code=RULE_RECIPROCITY_NOSE_TO_NOSE,
            detector="nose_to_nose",
            n_rows_changed=0,
            detail="skipped",
        )

    is_nose_to_nose = out["behavior_label"].eq("nose_to_nose")
    if not bool(is_nose_to_nose.any()):
        return out, StageDiagnostic(
            stage="reciprocity_nose_to_nose",
            rule_code=RULE_RECIPROCITY_NOSE_TO_NOSE,
            detector="nose_to_nose",
            n_rows_changed=0,
            detail="no nose_to_nose frames",
        )

    nose_status = (
        out[key_cols]
        .assign(_is_nose_to_nose=is_nose_to_nose)
        .groupby(key_cols, sort=False)["_is_nose_to_nose"]
        .max()
    )
    reverse_index = pd.MultiIndex.from_arrays(
        [
            out["exp_name"],
            out["dense_frame_idx"],
            out["partner_id"],
            out["focal_id"],
        ],
        names=key_cols,
    )
    partner_is_nose_to_nose = (
        nose_status.reindex(reverse_index).fillna(False).to_numpy(dtype=bool, copy=False)
    )
    one_sided_mask = is_nose_to_nose & (~partner_is_nose_to_nose)
    replaced_frames = int(one_sided_mask.sum())
    if replaced_frames == 0:
        return out, StageDiagnostic(
            stage="reciprocity_nose_to_nose",
            rule_code=RULE_RECIPROCITY_NOSE_TO_NOSE,
            detector="nose_to_nose",
            n_rows_changed=0,
            detail="no one-sided frames",
        )

    out.loc[one_sided_mask, "behavior_label"] = fallback_label
    _mark_reconstruction(out, one_sided_mask, _RECON_RECIPROCITY)
    out = sync_pred_columns_from_behavior_label(out)
    return out, StageDiagnostic(
        stage="reciprocity_nose_to_nose",
        rule_code=RULE_RECIPROCITY_NOSE_TO_NOSE,
        detector="nose_to_nose",
        n_rows_changed=replaced_frames,
        detail=f"replaced_frames={replaced_frames}",
    )


def enforce_mutual_disengaged_reciprocity(
    df: pd.DataFrame,
    *,
    fallback_label: str = LABEL_ENGAGED,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 20 — require both directions of a dyad to be mutually_disengaged."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    key_cols = ["exp_name", "dense_frame_idx", "focal_id", "partner_id"]
    if any(c not in out.columns for c in key_cols + ["behavior_label"]):
        return out, StageDiagnostic(
            stage="reciprocity_mutually_disengaged",
            rule_code=RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
            detector="mutually_disengaged",
            n_rows_changed=0,
            detail="skipped",
        )

    is_mutually_disengaged = out["behavior_label"].eq("mutually_disengaged")
    if not bool(is_mutually_disengaged.any()):
        return out, StageDiagnostic(
            stage="reciprocity_mutually_disengaged",
            rule_code=RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
            detector="mutually_disengaged",
            n_rows_changed=0,
            detail="no mutually_disengaged frames",
        )

    mutual_status = (
        out[key_cols]
        .assign(_is_mutually_disengaged=is_mutually_disengaged)
        .groupby(key_cols, sort=False)["_is_mutually_disengaged"]
        .max()
    )
    reverse_index = pd.MultiIndex.from_arrays(
        [
            out["exp_name"],
            out["dense_frame_idx"],
            out["partner_id"],
            out["focal_id"],
        ],
        names=key_cols,
    )
    partner_is_mutually_disengaged = (
        mutual_status.reindex(reverse_index)
        .fillna(False)
        .to_numpy(dtype=bool, copy=False)
    )
    one_sided_mask = is_mutually_disengaged & (~partner_is_mutually_disengaged)
    replaced_frames = int(one_sided_mask.sum())
    if replaced_frames == 0:
        return out, StageDiagnostic(
            stage="reciprocity_mutually_disengaged",
            rule_code=RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
            detector="mutually_disengaged",
            n_rows_changed=0,
            detail="no one-sided frames",
        )

    out.loc[one_sided_mask, "behavior_label"] = fallback_label
    _mark_reconstruction(out, one_sided_mask, _RECON_RECIPROCITY)
    out = sync_pred_columns_from_behavior_label(out)
    return out, StageDiagnostic(
        stage="reciprocity_mutually_disengaged",
        rule_code=RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
        detector="mutually_disengaged",
        n_rows_changed=replaced_frames,
        detail=f"replaced_frames={replaced_frames}",
    )


def apply_final_engaged_hysteresis(
    df: pd.DataFrame,
    *,
    max_engaged_frames: int,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Stage 21 — close short engaged islands inside the same stable label."""

    out = _ensure_label_reconstruction(df.copy().reset_index(drop=True))
    if "behavior_label" not in out.columns or max_engaged_frames <= 0:
        return out, StageDiagnostic(
            stage="hysteresis_engaged",
            rule_code=RULE_HYSTERESIS_ENGAGED,
            detector=None,
            n_rows_changed=0,
            detail="skipped",
        )
    if any(c not in out.columns for c in DIRECTIONAL_PAIR_COLS + ["dense_frame_idx"]):
        return out, StageDiagnostic(
            stage="hysteresis_engaged",
            rule_code=RULE_HYSTERESIS_ENGAGED,
            detector=None,
            n_rows_changed=0,
            detail="missing pair columns",
        )

    replaced_frames = 0
    blocked_labels = {LABEL_ENGAGED, "mutually_disengaged"}

    for _, pair_group in out.groupby(DIRECTIONAL_PAIR_COLS, sort=False):
        group = pair_group.sort_values("dense_frame_idx").copy()
        group["_orig_index"] = group.index
        group = group.reset_index(drop=True)
        if group.empty:
            continue

        bout_id = (group["behavior_label"] != group["behavior_label"].shift()).cumsum()
        bout_ids = list(dict.fromkeys(bout_id.tolist()))

        for i, current_bout_id in enumerate(bout_ids):
            if i == 0 or i + 1 >= len(bout_ids):
                continue
            current_mask = bout_id == current_bout_id
            current_bout = group.loc[current_mask]
            if current_bout.empty or current_bout["behavior_label"].iloc[0] != LABEL_ENGAGED:
                continue
            if int(len(current_bout)) > int(max_engaged_frames):
                continue

            prev_label = group.loc[bout_id == bout_ids[i - 1], "behavior_label"].iloc[0]
            next_label = group.loc[bout_id == bout_ids[i + 1], "behavior_label"].iloc[0]
            if prev_label != next_label or prev_label in blocked_labels:
                continue

            group.loc[current_mask, "behavior_label"] = prev_label
            group.loc[current_mask, "label_reconstruction"] = _RECON_HYSTERESIS
            replaced_frames += int(len(current_bout))

        out.loc[group["_orig_index"], "behavior_label"] = group["behavior_label"].values
        out.loc[group["_orig_index"], "label_reconstruction"] = group[
            "label_reconstruction"
        ].values

    if replaced_frames > 0:
        out = sync_pred_columns_from_behavior_label(out)

    return out, StageDiagnostic(
        stage="hysteresis_engaged",
        rule_code=RULE_HYSTERESIS_ENGAGED,
        detector=None,
        n_rows_changed=replaced_frames,
        detail=f"replaced_frames={replaced_frames}",
    )


def apply_sprint3_temporal_stages(
    frames: pd.DataFrame,
    config: BehavioralArbitrationConfig,
) -> tuple[pd.DataFrame, list[StageDiagnostic], list[str]]:
    """Apply Stages 10–21 then finalize labels (exact Pipeline 100 order)."""

    diagnostics: list[StageDiagnostic] = []
    warnings: list[str] = []
    sci = config.scientific
    fps = float(config.fps)
    focal_thresholds, dyadic_thresholds = _split_working_thresholds(config)

    gap_frames = int(sci.temporal.gap_fill_seconds * fps)
    min_duration_frames = int(sci.temporal.min_bout_seconds * fps)
    min_md_frames = int(sci.min_mutually_disengaged_frames)

    out, gap_diags = fill_small_gaps_and_filter_duration(
        frames,
        gap_frames=gap_frames,
        min_duration_frames=min_duration_frames,
        min_duration_overrides={"mutually_disengaged": min_md_frames},
    )
    diagnostics.extend(gap_diags)

    # Mirror y_*_smoothed from preds when diagnostic columns exist.
    if "pred_approach" in out.columns and "y_approach_smoothed" in out.columns:
        out["y_approach_smoothed"] = out["pred_approach"].astype(int)
    if "pred_retract" in out.columns and "y_retract_smoothed" in out.columns:
        out["y_retract_smoothed"] = out["pred_retract"].astype(int)

    out, diag = synchronize_behavior_label_from_preds(out)
    diagnostics.append(diag)

    out, diag = stitch_contextual_engaged_bouts(
        out,
        focal_thresholds=focal_thresholds,
        dyadic_thresholds=dyadic_thresholds,
        max_sniff_gap=int(sci.max_sniff_engaged_gap_frames),
        max_dynamic_gap=int(sci.max_dynamic_engaged_gap_frames),
    )
    diagnostics.append(diag)

    out, diag = stitch_dynamic_transitions(
        out,
        focal_thresholds=focal_thresholds,
        dyadic_thresholds=dyadic_thresholds,
        max_padding_frames=int(sci.max_dynamic_padding_frames),
        min_intent_stitch=float(sci.min_kinematic_intent_stitch),
        min_speed_stitch=float(sci.min_rolling_speed_stitch),
        look_ahead_frames=int(sci.stitch_dynamic.look_ahead_frames),
        hard_speed=float(sci.stitch_dynamic.hard_speed),
        hard_intent_factor=float(sci.stitch_dynamic.hard_intent_factor),
    )
    diagnostics.append(diag)

    out, diag = stitch_idle_retract_idle_bouts(
        out,
        max_retract_gap_frames=int(sci.max_idle_retract_gap_frames),
        far_apart_distance=float(sci.far_apart_idle_distance),
    )
    diagnostics.append(diag)

    out, diag = enforce_min_behavior_label_duration(
        out,
        label_name="mutually_disengaged",
        min_frames=min_md_frames,
        replacement_label=LABEL_ENGAGED,
    )
    diagnostics.append(diag)

    # Sync only MD / engaged preds after Stage 15 (legacy).
    if "pred_mutually_disengaged" in out.columns:
        out["pred_mutually_disengaged"] = (
            out["behavior_label"].eq("mutually_disengaged").astype(int)
        )
    if "pred_engaged" in out.columns:
        out["pred_engaged"] = out["behavior_label"].eq(LABEL_ENGAGED).astype(int)

    out, diag = rescue_dynamic_arcs(
        out,
        min_peak_intent=float(sci.case4_max_kinematic_intent_stitch),
        min_peak_speed=float(sci.case4_max_rolling_speed_stitch),
        initiation_intent=float(sci.case4_initiation_kinematic_intent_stitch),
        initiation_speed=float(sci.case4_initiation_rolling_speed_stitch),
        approach_prob_hint=float(sci.rescue.approach_prob_hint),
        retract_prob_hint=float(sci.rescue.retract_prob_hint),
        moving_intent_abs=float(sci.rescue.moving_intent_abs),
        moving_smooth_frames=int(sci.rescue.moving_smooth_frames),
    )
    diagnostics.append(diag)

    out, diag = apply_approach_probability_gate(
        out,
        nose_to_nose_max_dist=sci.approach_hard_gap_nose_to_nose_distance_norm,
        nose_to_body_max_dist=sci.approach_hard_gap_nose_to_body_distance_norm,
        nose_to_tail_max_dist=sci.approach_hard_gap_nose_to_tail_distance_norm,
    )
    diagnostics.append(diag)

    out, diag = synchronize_behavior_label_from_preds(out)
    diagnostics.append(diag)

    out, diag = enforce_nose_to_nose_reciprocity(out, fallback_label=LABEL_ENGAGED)
    diagnostics.append(diag)

    out, diag = enforce_mutual_disengaged_reciprocity(out, fallback_label=LABEL_ENGAGED)
    diagnostics.append(diag)

    out, diag = apply_final_engaged_hysteresis(
        out,
        max_engaged_frames=int(sci.final_engaged_hysteresis_frames),
    )
    diagnostics.append(diag)

    out, diag = finalize_behavior_labels(out)
    diagnostics.append(diag)

    return out, diagnostics, warnings
