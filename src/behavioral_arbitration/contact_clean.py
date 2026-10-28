"""Stage 8 — contact optical assist / Schmitt / morphological close / min bout.

Migrates ``apply_sota_contact_arbitration`` + ``get_cleaned_bouts`` **without**
the intact/occluded remultiplexer (S7 fused ``prob_contact`` is authoritative).

Zero-gap policy (S8-B §3.2): ``max_gap_to_fill=0`` disables morphological
closing entirely. Do not clamp ``0→1``. Stage 10 gap fill remains separate
(Sprint 3).

Invalid / routing-sentinel contact evidence is excluded from optical boost and
Schmitt trips; sentinel zeros are never treated as confident negatives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.config import ContactPostProcessingSettings
from behavioral_arbitration.contracts import (
    RULE_CONTACT_MIN_DUR,
    RULE_CONTACT_MORPH,
    RULE_CONTACT_OPTICAL,
    RULE_CONTACT_SCHMITT,
)
from behavioral_arbitration.results import StageDiagnostic
from behavioral_arbitration.validity import mark_rule_vetoed


def _contact_config_dict(settings: ContactPostProcessingSettings) -> dict[str, Any]:
    return {
        "smoothing_window": max(1, int(settings.smoothing_window)),
        "prob_threshold_high": {
            "intact": float(settings.prob_threshold_high_intact),
            "occluded": float(settings.prob_threshold_high_occluded),
        },
        "prob_threshold_low": float(settings.prob_threshold_low),
        "optical_assist_min": float(settings.optical_assist_min),
        "max_gap_to_fill": int(settings.max_gap_to_fill),
        "min_bout_duration": max(1, int(settings.min_bout_duration)),
    }


def apply_contact_arbitration_no_remultiplex(
    df: pd.DataFrame,
    config: dict[str, Any] | ContactPostProcessingSettings,
) -> pd.DataFrame:
    """Per-direction contact cleaning without specialist remultiplex.

    Expects a single directional group sorted by ``dense_frame_idx``.
    """

    if isinstance(config, ContactPostProcessingSettings):
        cfg_raw = _contact_config_dict(config)
    else:
        cfg_raw = dict(config)

    out = df.copy()
    if "prob_contact" not in out.columns:
        return out

    # --- Intentionally omitted: remultiplex intact/occluded specialists ---

    is_occ = (
        out.get("is_total_occlusion", pd.Series(0, index=out.index))
        .fillna(0)
        .astype(int)
        == 1
    )

    thresh_high_cfg = cfg_raw.get(
        "prob_threshold_high", cfg_raw.get("prob_threshold", 0.43)
    )
    if isinstance(thresh_high_cfg, dict):
        t_intact = float(thresh_high_cfg.get("intact", 0.43))
        t_occ = float(thresh_high_cfg.get("occluded", 0.43))
    else:
        t_intact = float(thresh_high_cfg)
        t_occ = float(thresh_high_cfg)
    thresh_high_array = np.where(is_occ.to_numpy(), t_occ, t_intact)

    smoothing_window = max(1, int(cfg_raw.get("smoothing_window", 9)))
    prob_threshold_low = float(cfg_raw.get("prob_threshold_low", 0.15))
    optical_assist_min = float(cfg_raw.get("optical_assist_min", 0.15))
    max_gap_to_fill = int(cfg_raw.get("max_gap_to_fill", 0))
    min_bout_duration = max(1, int(cfg_raw.get("min_bout_duration", 30)))

    # Validity mask: never treat routing sentinels as observed zeros.
    if "arbitration_input_valid_contact" in out.columns:
        observed = out["arbitration_input_valid_contact"].astype(bool).to_numpy()
    elif "input_valid_contact" in out.columns:
        observed = out["input_valid_contact"].fillna(True).astype(bool).to_numpy()
    else:
        observed = np.ones(len(out), dtype=bool)

    raw_prob = pd.to_numeric(out["prob_contact"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    # Unobserved → NaN (not 0.0) so optical / Schmitt cannot trip on sentinels.
    work_prob = np.where(observed, raw_prob, np.nan)

    out["cc_count"] = pd.to_numeric(out.get("cc_count", np.nan), errors="coerce")
    cc = out["cc_count"].to_numpy()

    # 1. Optical assist (observed + cc_count==1 + prob >= min).
    boost_mask = (
        observed
        & (cc == 1)
        & np.isfinite(work_prob)
        & (work_prob >= optical_assist_min)
    )
    fused = work_prob.astype(np.float64, copy=True)
    fused[boost_mask] = np.maximum(fused[boost_mask], thresh_high_array[boost_mask])
    out["prob_contact_fused"] = fused
    out["contact_optical_boost"] = boost_mask.astype(int)

    # 2. Centered median smooth (NaNs preserved where window is all-NaN).
    fused_s = pd.Series(fused, index=out.index)
    smooth = fused_s.rolling(
        window=smoothing_window,
        center=True,
        min_periods=1,
    ).median()
    out["prob_contact_smooth"] = smooth

    # 3. Schmitt trigger — ignore unobserved frames for trip decisions.
    smooth_arr = smooth.to_numpy(dtype=np.float64)
    state = np.full(len(out), np.nan, dtype=np.float64)
    trip_high = (
        observed
        & np.isfinite(smooth_arr)
        & (smooth_arr >= thresh_high_array)
    )
    trip_low = (
        observed
        & np.isfinite(smooth_arr)
        & (smooth_arr < prob_threshold_low)
    )
    state[trip_high] = 1.0
    state[trip_low] = 0.0
    out["contact_schmitt_state"] = state
    is_contact_base = pd.Series(state, index=out.index).ffill().fillna(0.0)
    out["is_contact_base"] = is_contact_base

    # 4. Morphological closing — disabled when max_gap_to_fill == 0 (MVP policy).
    if max_gap_to_fill <= 0:
        is_contact_gapped = is_contact_base.copy()
        morph_applied = False
    else:
        w = int(max_gap_to_fill)
        is_contact_gapped = (
            is_contact_base.rolling(window=w, min_periods=1, center=True)
            .max()
            .rolling(window=w, min_periods=1, center=True)
            .min()
        )
        morph_applied = True
    out["is_contact_gapped"] = is_contact_gapped
    out["contact_morph_applied"] = int(morph_applied)

    # 5. Minimum bout duration on positive runs.
    bout_ids = (is_contact_gapped != is_contact_gapped.shift()).cumsum()
    bout_sizes = out.groupby(bout_ids)["is_contact_gapped"].transform("size")
    is_final = np.where(
        (is_contact_gapped.to_numpy() == 1) & (bout_sizes.to_numpy() >= min_bout_duration),
        1,
        0,
    )
    out["is_contact_final"] = is_final
    return out


def get_cleaned_contact_bouts(
    df: pd.DataFrame,
    settings: ContactPostProcessingSettings,
) -> tuple[pd.DataFrame, list[StageDiagnostic]]:
    """Apply contact cleaning per directional identity and write back columns."""

    out = df.copy().reset_index(drop=True)
    if "prob_contact" not in out.columns:
        return out, [
            StageDiagnostic(
                stage="contact_clean",
                rule_code=RULE_CONTACT_SCHMITT,
                detector="contact",
                n_rows_changed=0,
                detail="skipped (prob_contact absent)",
            )
        ]

    cfg = _contact_config_dict(settings)
    group_cols = ["exp_name", "focal_id", "partner_id"]
    for col in group_cols:
        if col not in out.columns:
            raise ValueError(f"Contact cleaning requires column '{col}'.")

    before_pred = (
        pd.to_numeric(out.get("pred_contact", 0), errors="coerce").fillna(0).astype(int)
    )
    before_prob = pd.to_numeric(out["prob_contact"], errors="coerce").to_numpy(
        dtype=np.float64
    )

    prob_out = before_prob.copy()
    pred_out = np.zeros(len(out), dtype=np.int8)
    optical_n = 0
    morph_n = 0
    schmitt_diag = np.zeros(len(out), dtype=np.float64)
    fused_out = np.full(len(out), np.nan, dtype=np.float64)
    smooth_out = np.full(len(out), np.nan, dtype=np.float64)

    for _, group_index in out.groupby(group_cols, sort=False).groups.items():
        pos = np.asarray(list(group_index), dtype=int)
        order = np.argsort(
            out.iloc[pos]["dense_frame_idx"].to_numpy(), kind="stable"
        )
        write_pos = pos[order]
        dyad_sorted = out.iloc[write_pos].copy()
        proc = apply_contact_arbitration_no_remultiplex(dyad_sorted, cfg)

        smooth_vals = pd.to_numeric(
            proc["prob_contact_smooth"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        # Write smoothed prob only where contact was observed; keep NaN for
        # unobserved so pass-2 gates / QC do not invent valid negatives.
        if "arbitration_input_valid_contact" in proc.columns:
            obs = proc["arbitration_input_valid_contact"].astype(bool).to_numpy()
        else:
            obs = np.isfinite(smooth_vals)
        written_prob = np.where(obs, np.nan_to_num(smooth_vals, nan=0.0), np.nan)
        final_pred = (
            pd.to_numeric(proc["is_contact_final"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int8)
        )
        # Unobserved frames stay pred=0 without claiming valid_negative.
        final_pred = np.where(obs, final_pred, 0).astype(np.int8)

        prob_out[write_pos] = written_prob
        pred_out[write_pos] = final_pred
        fused_out[write_pos] = pd.to_numeric(
            proc["prob_contact_fused"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        smooth_out[write_pos] = smooth_vals
        schmitt_diag[write_pos] = pd.to_numeric(
            proc["contact_schmitt_state"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        optical_n += int(proc["contact_optical_boost"].sum())
        morph_n += int(proc["contact_morph_applied"].iloc[0]) if len(proc) else 0

    out["prob_contact_fused"] = fused_out
    out["prob_contact_smooth"] = smooth_out
    out["contact_schmitt_state"] = schmitt_diag
    out["prob_contact"] = prob_out
    out["pred_contact"] = pred_out.astype(np.int8)

    # Stamp evidence for cleaned positives that were previously valid.
    changed_to_neg = (before_pred == 1) & (out["pred_contact"].astype(int) == 0)
    if "arbitration_input_valid_contact" in out.columns:
        # Min-bout / Schmitt cleaned former positives → rule_vetoed when still valid.
        out = mark_rule_vetoed(out, detector="contact", mask=changed_to_neg)

    n_pred_changed = int((before_pred != out["pred_contact"].astype(int)).sum())
    diagnostics = [
        StageDiagnostic(
            stage="contact_clean",
            rule_code=RULE_CONTACT_OPTICAL,
            detector="contact",
            n_rows_changed=optical_n,
            detail=f"optical_boost_frames={optical_n}",
        ),
        StageDiagnostic(
            stage="contact_clean",
            rule_code=RULE_CONTACT_SCHMITT,
            detector="contact",
            n_rows_changed=n_pred_changed,
            detail=f"pred_contact_changed={n_pred_changed}",
        ),
        StageDiagnostic(
            stage="contact_clean",
            rule_code=RULE_CONTACT_MORPH,
            detector="contact",
            n_rows_changed=morph_n,
            detail=(
                f"max_gap_to_fill={cfg['max_gap_to_fill']}; "
                "morphological_close="
                + ("enabled" if cfg["max_gap_to_fill"] > 0 else "disabled")
            ),
        ),
        StageDiagnostic(
            stage="contact_clean",
            rule_code=RULE_CONTACT_MIN_DUR,
            detector="contact",
            n_rows_changed=int(changed_to_neg.sum()),
            detail=f"min_bout_duration={cfg['min_bout_duration']}",
        ),
    ]
    return out, diagnostics
