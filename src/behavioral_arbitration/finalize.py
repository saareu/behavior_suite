"""Sprint 3 finalize — promote unresolved invalid evidence to ``undefined``.

Valid engaged residuals stay ``engaged`` (P4). Concrete exclusive hierarchy
labels are preserved. Does not invent ``pred_undefined``.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    EXCLUSIVE_ETHOGRAM_LABELS,
    HIERARCHY_DETECTORS,
    LABEL_ENGAGED,
    LABEL_UNDEFINED,
    LABEL_UNRESOLVED,
    RESIDUAL_ENGAGED_VALID,
    RESIDUAL_UNRESOLVED_INVALID,
    RULE_LABEL_FINALIZE,
    RULE_LABEL_UNDEFINED,
)
from behavioral_arbitration.results import StageDiagnostic

_CONCRETE_EXCLUSIVE = frozenset(EXCLUSIVE_ETHOGRAM_LABELS)


def finalize_behavior_labels(
    frames: pd.DataFrame,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Write final exclusive labels; promote unresolved invalid → undefined."""

    out = frames.copy().reset_index(drop=True)
    if "behavior_label" not in out.columns:
        out["behavior_label"] = LABEL_UNRESOLVED
    if "hierarchy_residual_kind" not in out.columns:
        out["hierarchy_residual_kind"] = ""
    if "undefined_reason" not in out.columns:
        out["undefined_reason"] = ""
    if "label_reconstruction" not in out.columns:
        out["label_reconstruction"] = ""

    labels = out["behavior_label"].astype(str)
    residual = out["hierarchy_residual_kind"].astype(str)

    hierarchy_pred_any = pd.Series(False, index=out.index)
    for det in HIERARCHY_DETECTORS:
        pred_col = f"pred_{det}"
        if pred_col in out.columns:
            hierarchy_pred_any |= out[pred_col].fillna(0).astype(int).eq(1)

    # P4: valid engaged residual never auto-undefined.
    keep_engaged_valid = residual.eq(RESIDUAL_ENGAGED_VALID)

    # Concrete exclusive labels (sniff / dynamic / MD / engaged) stay unless
    # residual still marks unresolved_invalid with no hierarchy winner pred
    # (Stage 11 sync may have defaulted unresolved → engaged).
    candidate = labels.eq(LABEL_UNRESOLVED) | residual.eq(RESIDUAL_UNRESOLVED_INVALID)
    to_undefined = candidate & (~hierarchy_pred_any) & (~keep_engaged_valid)

    # Still keep concrete winners that are not unresolved_invalid residuals.
    concrete_keep = labels.isin(_CONCRETE_EXCLUSIVE) & residual.ne(
        RESIDUAL_UNRESOLVED_INVALID
    )
    to_undefined = to_undefined & (~concrete_keep)

    n_undefined = int(to_undefined.sum())
    if n_undefined:
        out.loc[to_undefined, "behavior_label"] = LABEL_UNDEFINED
        out.loc[to_undefined, "undefined_reason"] = RESIDUAL_UNRESOLVED_INVALID

    # Sync exclusive pred_* from final label (no pred_undefined column).
    for label in (*HIERARCHY_DETECTORS, LABEL_ENGAGED):
        pred_col = f"pred_{label}"
        if pred_col in out.columns:
            out[pred_col] = out["behavior_label"].eq(label).astype(int)

    undefined_mask = out["behavior_label"].eq(LABEL_UNDEFINED)
    if undefined_mask.any():
        for det in HIERARCHY_DETECTORS:
            pred_col = f"pred_{det}"
            if pred_col in out.columns:
                out.loc[undefined_mask, pred_col] = 0
        if "pred_engaged" in out.columns:
            out.loc[undefined_mask, "pred_engaged"] = 0

    # Unchanged / empty reconstruction → selected after finalize.
    recon = out["label_reconstruction"].fillna("").astype(str)
    empty_recon = recon.str.len() == 0
    if empty_recon.any():
        out.loc[empty_recon, "label_reconstruction"] = "selected"

    rule = RULE_LABEL_UNDEFINED if n_undefined else RULE_LABEL_FINALIZE
    return out, StageDiagnostic(
        stage="finalize",
        rule_code=rule,
        detector=None,
        n_rows_changed=n_undefined,
        detail=f"undefined_frames={n_undefined}",
    )
