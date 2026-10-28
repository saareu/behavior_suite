"""Directional expansion of undirected dyadic predictions (Pipeline 100 Stage 1).

Source authority: ``_expand_dyadic_to_directional`` in
``legacy/pipelines/100_behavioral_arbiter.py``.

S8 addition: also expand / carry ``input_valid_*``, ``score_origin_*``,
``invalid_reason_*``, and ``threshold_*`` so validity is not lost.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    REQUIRED_DYADIC_DETECTORS,
    RULE_EXPAND_DYADIC,
    SYMMETRIC_DYADIC_DETECTORS,
    ExpansionError,
)
from behavioral_arbitration.results import StageDiagnostic


def _pair_key(frame: pd.DataFrame) -> pd.Series:
    """Unordered pair key ``(min_id, max_id)`` matching Pipeline 100."""

    lo = frame[["focal_id", "partner_id"]].min(axis=1).astype(int)
    hi = frame[["focal_id", "partner_id"]].max(axis=1).astype(int)
    return pd.Series(list(zip(lo, hi, strict=True)), index=frame.index, dtype=object)


def _dyadic_detector_names(dyadic_df: pd.DataFrame) -> list[str]:
    """Logical dyadic detectors present as ``prob_<name>`` (not specialists)."""

    known = set(REQUIRED_DYADIC_DETECTORS) | set(SYMMETRIC_DYADIC_DETECTORS)
    present: list[str] = []
    for det in list(REQUIRED_DYADIC_DETECTORS) + sorted(known - set(REQUIRED_DYADIC_DETECTORS)):
        if f"prob_{det}" in dyadic_df.columns and det not in present:
            present.append(det)
    return present


def _carry_columns_for_detector(dyadic_df: pd.DataFrame, detector: str) -> list[str]:
    cols: list[str] = []
    for prefix in (
        "prob_",
        "pred_",
        "input_valid_",
        "score_origin_",
        "invalid_reason_",
        "threshold_",
    ):
        col = f"{prefix}{detector}"
        if col in dyadic_df.columns:
            cols.append(col)
    raw = f"prob_{detector}_raw"
    if raw in dyadic_df.columns:
        cols.append(raw)
    return cols


def expand_dyadic_to_directional(
    dyadic_df: pd.DataFrame,
    focal_df: pd.DataFrame,
    *,
    required_dyadic: Sequence[str] = REQUIRED_DYADIC_DETECTORS,
) -> tuple[pd.DataFrame, StageDiagnostic]:
    """Left-merge undirected dyadic signals onto every focal direction.

    Failures
    --------
    - Duplicate undirected dyadic keys → hard fail
    - Duplicate directional focal keys → hard fail
    - Missing required dyadic detector column → hard fail
    - Left-merge miss for a required detector under K=2 → hard fail
      (does **not** legacy-fill 0 as a confident negative; conflict C1)
    """

    if dyadic_df.empty:
        raise ExpansionError("Dyadic predictions table is empty.")
    if focal_df.empty:
        raise ExpansionError("Focal predictions table is empty.")

    missing_req = [d for d in required_dyadic if f"prob_{d}" not in dyadic_df.columns]
    if missing_req:
        raise ExpansionError(
            "Dyadic wide view missing required detector probability columns: "
            + ", ".join(missing_req)
        )

    focal_copy = focal_df.copy()
    dyadic_copy = dyadic_df.copy()
    detectors = _dyadic_detector_names(dyadic_copy)
    for det in required_dyadic:
        if det not in detectors:
            detectors.append(det)

    for key in ("exp_name", "dense_frame_idx", "focal_id", "partner_id"):
        if key not in focal_copy.columns:
            raise ExpansionError(f"Focal table missing column '{key}'.")
        if key not in dyadic_copy.columns:
            raise ExpansionError(f"Dyadic table missing column '{key}'.")

    dyadic_copy["_pair"] = _pair_key(dyadic_copy)
    focal_copy["_pair"] = _pair_key(focal_copy)

    dyadic_key_cols = ["exp_name", "dense_frame_idx", "_pair"]
    if bool(dyadic_copy.duplicated(subset=dyadic_key_cols, keep=False).any()):
        raise ExpansionError(
            "Dyadic predictions contain duplicate undirected keys for "
            "(exp_name, dense_frame_idx, pair)."
        )

    focal_key_cols = ["exp_name", "dense_frame_idx", "focal_id", "partner_id"]
    if bool(focal_copy.duplicated(subset=focal_key_cols, keep=False).any()):
        raise ExpansionError(
            "Focal predictions contain duplicate directional keys for "
            "(exp_name, dense_frame_idx, focal_id, partner_id)."
        )

    carry: list[str] = []
    for det in detectors:
        for col in _carry_columns_for_detector(dyadic_copy, det):
            if col not in carry:
                carry.append(col)

    merge_cols = ["exp_name", "dense_frame_idx", "_pair", *carry]
    # Drop focal-side columns that would collide (should not exist yet).
    collide = [c for c in carry if c in focal_copy.columns]
    if collide:
        raise ExpansionError(
            "Focal table unexpectedly already contains dyadic columns: "
            + ", ".join(collide)
        )

    merged = focal_copy.merge(
        dyadic_copy.loc[:, merge_cols],
        on=["exp_name", "dense_frame_idx", "_pair"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.drop(columns=["_pair"])

    # Canonical unordered mouse_a / mouse_b from pair identities.
    merged["mouse_a"] = merged[["focal_id", "partner_id"]].min(axis=1).astype("int64")
    merged["mouse_b"] = merged[["focal_id", "partner_id"]].max(axis=1).astype("int64")

    # Conflict C1: required dyadic miss → hard fail (no silent 0 fill).
    for det in required_dyadic:
        prob_col = f"prob_{det}"
        if merged[prob_col].isna().any():
            n_miss = int(merged[prob_col].isna().sum())
            raise ExpansionError(
                f"Dyadic detector '{det}' missing for {n_miss} directional rows "
                "after expansion (K=2 requires a matching undirected pair; "
                "refusing legacy zero-fill as confident negative)."
            )

    # Non-required dyadic detectors (if any): still fill 0 only when not required.
    symmetric = set(SYMMETRIC_DYADIC_DETECTORS)
    for det in detectors:
        if det in required_dyadic:
            continue
        prob_col = f"prob_{det}"
        pred_col = f"pred_{det}"
        if prob_col in merged.columns:
            # Documented legacy fill for optional detectors only.
            _ = det in symmetric  # symmetry preserved by shared undirected source
            merged[prob_col] = merged[prob_col].fillna(0.0)
        if pred_col in merged.columns:
            merged[pred_col] = merged[pred_col].fillna(0).astype(int)

    merged = merged.sort_values(
        ["exp_name", "dense_frame_idx", "focal_id", "partner_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Uniqueness after expansion.
    if bool(
        merged.duplicated(
            subset=["dense_frame_idx", "focal_id", "partner_id"], keep=False
        ).any()
    ):
        raise ExpansionError(
            "Duplicate directional keys after expansion: "
            "(dense_frame_idx, focal_id, partner_id)."
        )

    diagnostic = StageDiagnostic(
        stage="expand",
        rule_code=RULE_EXPAND_DYADIC,
        detector=None,
        n_rows_changed=int(len(merged)),
        detail=(
            f"expanded {len(dyadic_copy)} undirected dyadic rows onto "
            f"{len(merged)} directional rows; detectors={list(detectors)}"
        ),
    )
    return merged, diagnostic


def assert_source_order_invariant(
    dyadic_a: pd.DataFrame,
    dyadic_b: pd.DataFrame,
    focal_df: pd.DataFrame,
) -> None:
    """Verify expansion is invariant to dyadic row order (same content)."""

    out_a, _ = expand_dyadic_to_directional(dyadic_a, focal_df)
    out_b, _ = expand_dyadic_to_directional(dyadic_b, focal_df)
    cols = [
        c
        for c in out_a.columns
        if c.startswith(("prob_", "pred_", "input_valid_", "score_origin_"))
        or c in ("dense_frame_idx", "focal_id", "partner_id", "mouse_a", "mouse_b")
    ]
    left = out_a.loc[:, cols].sort_values(
        ["dense_frame_idx", "focal_id", "partner_id"], kind="stable"
    ).reset_index(drop=True)
    right = out_b.loc[:, cols].sort_values(
        ["dense_frame_idx", "focal_id", "partner_id"], kind="stable"
    ).reset_index(drop=True)
    if not left.equals(right):
        raise ExpansionError(
            "Directional expansion depends on dyadic source row order."
        )
