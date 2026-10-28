"""Collapse finalized framewise states into exclusive + parallel bout tables.

Does **not** apply Pipeline 110 ``min_emit_bout_frames`` / ``min_emit_bout_seconds``
filters (P8). Inclusive endpoints match Pipeline 110 conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    BOUTS_SCHEMA_VERSION,
    LABEL_UNDEFINED,
    PublicationError,
)

BOUT_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "schema_version",
    "behavior_name",
    "channel",
    "identity_grain",
    "focal_id",
    "partner_id",
    "mouse_a_id",
    "mouse_b_id",
    "pair_id",
    "start_sample_idx",
    "end_sample_idx",
    "start_time_sec",
    "end_time_sec",
    "bout_duration_frames",
    "bout_duration_sec",
    "mean_prob",
    "max_prob",
    "median_prob",
    "frac_input_valid",
    "label_reconstruction_mode",
    "undefined_reason",
    "source_s7_run_id",
    "source_s6_run_id",
    "run_id",
    "exp_name",
)


def extract_behavior_bouts(
    states: pd.DataFrame,
    *,
    fps: float,
    source_s7_run_id: str,
    source_s6_run_id: str,
    run_id: str,
) -> pd.DataFrame:
    """Build exclusive ethogram + parallel contact/SBS bout rows from states."""

    if states is None or states.empty:
        raise PublicationError("Cannot extract bouts from an empty states table.")
    if fps <= 0:
        raise PublicationError(f"fps must be positive for bout timing; got {fps!r}.")

    work = states.copy().reset_index(drop=True)
    _ensure_identity_columns(work)

    rows: list[dict[str, Any]] = []
    rows.extend(
        _extract_exclusive_bouts(
            work,
            fps=float(fps),
            source_s7_run_id=source_s7_run_id,
            source_s6_run_id=source_s6_run_id,
            run_id=run_id,
        )
    )
    if "pred_contact" in work.columns:
        rows.extend(
            _extract_parallel_bouts(
                work,
                pred_col="pred_contact",
                prob_col="prob_contact",
                valid_col=_first_present(
                    work, ("arbitration_input_valid_contact", "input_valid_contact")
                ),
                channel="contact",
                behavior_name="contact",
                fps=float(fps),
                source_s7_run_id=source_s7_run_id,
                source_s6_run_id=source_s6_run_id,
                run_id=run_id,
            )
        )
    if "pred_side_by_side" in work.columns:
        rows.extend(
            _extract_parallel_bouts(
                work,
                pred_col="pred_side_by_side",
                prob_col="prob_side_by_side",
                valid_col=_first_present(
                    work,
                    (
                        "arbitration_input_valid_side_by_side",
                        "input_valid_side_by_side",
                    ),
                ),
                channel="side_by_side",
                behavior_name="side_by_side",
                fps=float(fps),
                source_s7_run_id=source_s7_run_id,
                source_s6_run_id=source_s6_run_id,
                run_id=run_id,
            )
        )

    if not rows:
        return _empty_bouts_frame()

    out = pd.DataFrame(rows)
    out["bout_id"] = [f"bout_{i:08d}" for i in range(len(out))]
    out["schema_version"] = BOUTS_SCHEMA_VERSION
    return out.loc[:, list(BOUT_COLUMNS)]


def _empty_bouts_frame() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=object) for col in BOUT_COLUMNS})


def _ensure_identity_columns(df: pd.DataFrame) -> None:
    required = ("dense_frame_idx", "focal_id", "partner_id")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PublicationError(
            "States table missing identity columns for bout extraction: "
            + ", ".join(missing)
        )
    if "behavior_label" not in df.columns:
        raise PublicationError("States table missing behavior_label for bout extraction.")
    if "exp_name" not in df.columns:
        df["exp_name"] = ""
    if "sample_idx" not in df.columns:
        df["sample_idx"] = df["dense_frame_idx"]
    if "time_sec" not in df.columns:
        df["time_sec"] = pd.NA
    if "mouse_a" not in df.columns or "mouse_b" not in df.columns:
        df["mouse_a"] = df[["focal_id", "partner_id"]].min(axis=1).astype("int64")
        df["mouse_b"] = df[["focal_id", "partner_id"]].max(axis=1).astype("int64")
    if "pair_id" not in df.columns:
        df["pair_id"] = (
            df["mouse_a"].astype(str) + "|" + df["mouse_b"].astype(str)
        )


def _extract_exclusive_bouts(
    frames: pd.DataFrame,
    *,
    fps: float,
    source_s7_run_id: str,
    source_s6_run_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    group_cols = ["exp_name", "focal_id", "partner_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in frames.groupby(group_cols, sort=True):
        exp_name, focal_id, partner_id = keys
        ordered = group.sort_values("dense_frame_idx", kind="stable").reset_index(
            drop=True
        )
        segments = _contiguous_segments(
            ordered,
            value_col="behavior_label",
            positive_only=False,
        )
        for start_i, end_i, label in segments:
            block = ordered.iloc[start_i : end_i + 1]
            behavior = str(label)
            prob_col = _exclusive_prob_column(behavior, ordered.columns)
            valid_col = _exclusive_valid_column(behavior, ordered.columns)
            rows.append(
                _bout_row(
                    block=block,
                    behavior_name=behavior,
                    channel="exclusive",
                    identity_grain="directional",
                    focal_id=int(focal_id),
                    partner_id=int(partner_id),
                    mouse_a_id=int(block["mouse_a"].iloc[0]),
                    mouse_b_id=int(block["mouse_b"].iloc[0]),
                    pair_id=str(block["pair_id"].iloc[0]),
                    exp_name=str(exp_name),
                    fps=fps,
                    prob_col=prob_col,
                    valid_col=valid_col,
                    source_s7_run_id=source_s7_run_id,
                    source_s6_run_id=source_s6_run_id,
                    run_id=run_id,
                )
            )
    return rows


def _extract_parallel_bouts(
    frames: pd.DataFrame,
    *,
    pred_col: str,
    prob_col: str,
    valid_col: str | None,
    channel: str,
    behavior_name: str,
    fps: float,
    source_s7_run_id: str,
    source_s6_run_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Collapse undirected pair frames so A→B / B→A do not double-count."""

    work = frames.copy()
    work["_mouse_a"] = work[["focal_id", "partner_id"]].min(axis=1).astype("int64")
    work["_mouse_b"] = work[["focal_id", "partner_id"]].max(axis=1).astype("int64")
    undirected = (
        work.sort_values(
            ["exp_name", "_mouse_a", "_mouse_b", "dense_frame_idx", "focal_id"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["exp_name", "_mouse_a", "_mouse_b", "dense_frame_idx"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    undirected["mouse_a"] = undirected["_mouse_a"]
    undirected["mouse_b"] = undirected["_mouse_b"]
    undirected["pair_id"] = (
        undirected["mouse_a"].astype(str) + "|" + undirected["mouse_b"].astype(str)
    )

    rows: list[dict[str, Any]] = []
    for keys, group in undirected.groupby(
        ["exp_name", "mouse_a", "mouse_b"], sort=True
    ):
        exp_name, mouse_a, mouse_b = keys
        ordered = group.sort_values("dense_frame_idx", kind="stable").reset_index(
            drop=True
        )
        segments = _contiguous_segments(
            ordered,
            value_col=pred_col,
            positive_only=True,
            positive_value=1,
        )
        for start_i, end_i, _value in segments:
            block = ordered.iloc[start_i : end_i + 1]
            rows.append(
                _bout_row(
                    block=block,
                    behavior_name=behavior_name,
                    channel=channel,
                    identity_grain="symmetric_pair",
                    focal_id=None,
                    partner_id=None,
                    mouse_a_id=int(mouse_a),
                    mouse_b_id=int(mouse_b),
                    pair_id=f"{int(mouse_a)}|{int(mouse_b)}",
                    exp_name=str(exp_name),
                    fps=fps,
                    prob_col=prob_col if prob_col in block.columns else None,
                    valid_col=valid_col if valid_col in block.columns else None,
                    source_s7_run_id=source_s7_run_id,
                    source_s6_run_id=source_s6_run_id,
                    run_id=run_id,
                )
            )
    return rows


def _contiguous_segments(
    ordered: pd.DataFrame,
    *,
    value_col: str,
    positive_only: bool,
    positive_value: Any = 1,
) -> list[tuple[int, int, Any]]:
    """Return inclusive index segments split on value / frame-gap discontinuities."""

    if ordered.empty:
        return []
    values = ordered[value_col].to_numpy()
    frames = pd.to_numeric(ordered["dense_frame_idx"], errors="coerce").to_numpy(
        dtype=float
    )
    segments: list[tuple[int, int, Any]] = []
    start = 0
    for i in range(1, len(ordered)):
        gap = frames[i] - frames[i - 1]
        value_changed = values[i] != values[i - 1]
        # NaN != NaN is True; treat both-NaN as same for exclusivity.
        if pd.isna(values[i]) and pd.isna(values[i - 1]):
            value_changed = False
        discontinuous = not np.isfinite(gap) or abs(gap - 1.0) > 1e-9
        if value_changed or discontinuous:
            segments.append((start, i - 1, values[start]))
            start = i
    segments.append((start, len(ordered) - 1, values[start]))

    if not positive_only:
        return [
            (s, e, (LABEL_UNDEFINED if pd.isna(v) else v))
            for s, e, v in segments
        ]

    kept: list[tuple[int, int, Any]] = []
    for s, e, v in segments:
        if pd.isna(v):
            continue
        try:
            if int(v) == int(positive_value):
                kept.append((s, e, v))
        except (TypeError, ValueError):
            continue
    return kept


def _bout_row(
    *,
    block: pd.DataFrame,
    behavior_name: str,
    channel: str,
    identity_grain: str,
    focal_id: int | None,
    partner_id: int | None,
    mouse_a_id: int,
    mouse_b_id: int,
    pair_id: str,
    exp_name: str,
    fps: float,
    prob_col: str | None,
    valid_col: str | None,
    source_s7_run_id: str,
    source_s6_run_id: str,
    run_id: str,
) -> dict[str, Any]:
    start_sample = int(
        pd.to_numeric(block["sample_idx"], errors="coerce").iloc[0]
    )
    end_sample = int(pd.to_numeric(block["sample_idx"], errors="coerce").iloc[-1])
    start_time = _optional_float(block["time_sec"].iloc[0])
    end_time = _optional_float(block["time_sec"].iloc[-1])
    n_frames = int(len(block))
    mean_prob, max_prob, median_prob = _prob_summaries(block, prob_col)
    frac_valid = _frac_input_valid(block, valid_col)
    recon_mode = _mode_or_empty(block.get("label_reconstruction"))
    undef_reason = _mode_or_empty(block.get("undefined_reason"))

    return {
        "bout_id": "",  # filled after concatenation
        "schema_version": BOUTS_SCHEMA_VERSION,
        "behavior_name": behavior_name,
        "channel": channel,
        "identity_grain": identity_grain,
        "focal_id": focal_id if focal_id is not None else pd.NA,
        "partner_id": partner_id if partner_id is not None else pd.NA,
        "mouse_a_id": mouse_a_id,
        "mouse_b_id": mouse_b_id,
        "pair_id": pair_id,
        "start_sample_idx": start_sample,
        "end_sample_idx": end_sample,
        "start_time_sec": start_time,
        "end_time_sec": end_time,
        "bout_duration_frames": n_frames,
        "bout_duration_sec": float(n_frames) / float(fps),
        "mean_prob": mean_prob,
        "max_prob": max_prob,
        "median_prob": median_prob,
        "frac_input_valid": frac_valid,
        "label_reconstruction_mode": recon_mode,
        "undefined_reason": undef_reason,
        "source_s7_run_id": source_s7_run_id,
        "source_s6_run_id": source_s6_run_id,
        "run_id": run_id,
        "exp_name": exp_name,
    }


def _exclusive_prob_column(behavior: str, columns: Sequence[str]) -> str | None:
    if behavior == LABEL_UNDEFINED:
        return None
    candidate = f"prob_{behavior}"
    return candidate if candidate in columns else None


def _exclusive_valid_column(behavior: str, columns: Sequence[str]) -> str | None:
    if behavior == LABEL_UNDEFINED:
        return None
    for name in (
        f"arbitration_input_valid_{behavior}",
        f"input_valid_{behavior}",
    ):
        if name in columns:
            return name
    return None


def _prob_summaries(
    block: pd.DataFrame, prob_col: str | None
) -> tuple[float, float, float]:
    if prob_col is None or prob_col not in block.columns:
        return (float("nan"), float("nan"), float("nan"))
    vals = pd.to_numeric(block[prob_col], errors="coerce")
    if vals.notna().sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(vals.mean()),
        float(vals.max()),
        float(vals.median()),
    )


def _frac_input_valid(block: pd.DataFrame, valid_col: str | None) -> float:
    if valid_col is None or valid_col not in block.columns:
        return float("nan")
    series = block[valid_col]
    if series.dtype == object:
        parsed = series.map(_as_bool_or_na)
        n = int(parsed.notna().sum())
        if n == 0:
            return float("nan")
        return float(parsed.fillna(False).astype(bool).sum()) / float(n)
    try:
        bools = series.astype(bool)
    except (TypeError, ValueError):
        return float("nan")
    return float(bools.sum()) / float(len(bools)) if len(bools) else float("nan")


def _as_bool_or_na(value: Any) -> bool | float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return float("nan")


def _mode_or_empty(series: pd.Series | None) -> str:
    if series is None:
        return ""
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned.str.len() > 0]
    if cleaned.empty:
        return ""
    modes = cleaned.mode(dropna=True)
    return str(modes.iloc[0]) if len(modes) else ""


def _optional_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _first_present(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None
