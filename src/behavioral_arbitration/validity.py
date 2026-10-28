"""S7 evidence normalization for arbitration (separate from scientific gates).

Distinguishes valid positives / valid negatives from routing sentinels and
missing detectors. A routing sentinel with exported probability 0 must never
become a valid observed negative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    EVIDENCE_MISSING_DETECTOR,
    EVIDENCE_RULE_VETOED,
    EVIDENCE_UNOBSERVED_INVALID_ROUTE,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_VALID_POSITIVE,
    REQUIRED_DYADIC_DETECTORS,
    REQUIRED_FOCAL_DETECTORS,
    RULE_VALIDITY_MASK,
)
from behavioral_arbitration.results import StageDiagnostic
from behavioral_inference.contracts import SCORE_ORIGIN_ROUTING_SENTINEL


def all_required_detectors() -> tuple[str, ...]:
    return tuple([*REQUIRED_DYADIC_DETECTORS, *REQUIRED_FOCAL_DETECTORS])


def classify_evidence_state(
    *,
    input_valid: bool | None,
    score_origin: str | None,
    pred: int | None,
    column_present: bool,
) -> str:
    """Classify one detector observation into an explicit evidence state."""

    if not column_present:
        return EVIDENCE_MISSING_DETECTOR
    if input_valid is False or score_origin == SCORE_ORIGIN_ROUTING_SENTINEL:
        return EVIDENCE_UNOBSERVED_INVALID_ROUTE
    pred_i = 0 if pred is None else int(pred)
    if pred_i == 1:
        return EVIDENCE_VALID_POSITIVE
    # Remaining observed cases (input_valid True/None; False already returned).
    return EVIDENCE_VALID_NEGATIVE


def normalize_arbitration_evidence(
    frames: pd.DataFrame,
    *,
    detectors: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Copy S7 provenance and build masked arbitration-input columns.

    For each detector ``d``:

    - ``s7_prob_d`` / ``s7_pred_d`` — frozen S7 intake copies
    - ``prob_d`` / ``pred_d`` — working arbitration values
    - ``arbitration_input_valid_d`` — bool
    - ``evidence_state_d`` — one of the EVIDENCE_* states

    Invalid / routing-sentinel rows: working ``prob_d`` set to NaN and
    ``pred_d`` to 0 **without** claiming a valid negative (state =
    ``unobserved_invalid_route``).
    """

    out = frames.copy()
    dets = list(detectors) if detectors is not None else list(all_required_detectors())
    n_masked = 0

    for det in dets:
        prob_col = f"prob_{det}"
        pred_col = f"pred_{det}"
        valid_col = f"input_valid_{det}"
        origin_col = f"score_origin_{det}"
        state_col = f"evidence_state_{det}"
        arb_valid_col = f"arbitration_input_valid_{det}"

        if prob_col not in out.columns:
            out[state_col] = EVIDENCE_MISSING_DETECTOR
            out[arb_valid_col] = False
            continue

        s7_prob = f"s7_prob_{det}"
        s7_pred = f"s7_pred_{det}"
        if s7_prob not in out.columns:
            out[s7_prob] = pd.to_numeric(out[prob_col], errors="coerce")
        if s7_pred not in out.columns:
            if pred_col in out.columns:
                out[s7_pred] = (
                    pd.to_numeric(out[pred_col], errors="coerce").fillna(0).astype(int)
                )
            else:
                out[s7_pred] = 0

        if valid_col in out.columns:
            valid_bool = out[valid_col].fillna(True).astype(bool)
            # Rows where input_valid was explicitly False.
            invalid_flag = out[valid_col].notna() & (~out[valid_col].astype(bool))
        else:
            valid_bool = pd.Series(True, index=out.index)
            invalid_flag = pd.Series(False, index=out.index)

        if origin_col in out.columns:
            sentinel = out[origin_col].astype(str) == SCORE_ORIGIN_ROUTING_SENTINEL
        else:
            sentinel = pd.Series(False, index=out.index)

        unobserved = invalid_flag | sentinel
        pred_vals = out[s7_pred].astype(int)
        states = np.where(
            unobserved,
            EVIDENCE_UNOBSERVED_INVALID_ROUTE,
            np.where(
                pred_vals == 1,
                EVIDENCE_VALID_POSITIVE,
                EVIDENCE_VALID_NEGATIVE,
            ),
        )
        # valid_bool unused except documenting that absent validity ⇒ observed.
        _ = valid_bool

        arb_valid = ~unobserved
        work_prob = out[s7_prob].astype(float).copy()
        work_pred = pred_vals.copy()
        work_prob.loc[unobserved] = np.nan
        work_pred.loc[unobserved] = 0
        n_masked += int(unobserved.sum())

        out[state_col] = states
        out[arb_valid_col] = arb_valid.to_numpy()
        out[prob_col] = work_prob
        out[pred_col] = work_pred.astype(int)

    diagnostic = StageDiagnostic(
        stage="validity",
        rule_code=RULE_VALIDITY_MASK,
        detector=None,
        n_rows_changed=n_masked,
        detail=f"masked unobserved arbitration inputs across detectors={dets}",
    )
    return out, diagnostic


def mark_rule_vetoed(
    frames: pd.DataFrame,
    *,
    detector: str,
    mask: pd.Series,
) -> pd.DataFrame:
    """Set vetoed valid rows to zero and stamp ``evidence_state`` as rule_vetoed."""

    out = frames.copy()
    state_col = f"evidence_state_{detector}"
    arb_valid_col = f"arbitration_input_valid_{detector}"
    prob_col = f"prob_{detector}"
    pred_col = f"pred_{detector}"

    if arb_valid_col in out.columns:
        eligible = mask.fillna(False) & out[arb_valid_col].astype(bool)
    else:
        eligible = mask.fillna(False)

    if prob_col in out.columns:
        out.loc[eligible, prob_col] = 0.0
    if pred_col in out.columns:
        out.loc[eligible, pred_col] = 0
    if state_col in out.columns:
        out.loc[eligible, state_col] = EVIDENCE_RULE_VETOED
    return out
