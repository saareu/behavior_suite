"""Compact ethogram summary rollups for Subsystem 09 packages.

Descriptive of packaged S8 content only — not a new ethogram. Framewise
``label_time`` statistics are read from S8 states. Parallel-channel coverage
is independent of exclusive labels. Unavailable framewise overlap stats are
**not** inferred from bout tables alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import (
    ETHOGRAM_SUMMARY_COLUMNS,
    ETHOGRAM_SUMMARY_SCHEMA_VERSION,
    SUMMARY_KIND_BOUT_COUNTS,
    SUMMARY_KIND_LABEL_TIME,
    SUMMARY_KIND_PARALLEL_OVERLAP,
    SUMMARY_KIND_SESSION_TIMING,
    TARGET_FPS,
    EthogramSummaryError,
)

# Inclusive frame-duration histogram edges for bout_counts (documented).
_DURATION_FRAME_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("frames_eq_1", 1, 1),
    ("frames_2_to_9", 2, 9),
    ("frames_10_to_11", 10, 11),
    ("frames_12_to_59", 12, 59),
    ("frames_60_to_119", 60, 119),
    ("frames_ge_120", 120, None),
)


def build_ethogram_summary(
    *,
    packaged_bouts: pd.DataFrame,
    states: pd.DataFrame | None,
    sync_index: pd.DataFrame,
    fps: float = TARGET_FPS,
    s8_qc_metrics: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the in-memory ethogram summary table (S9-B §1.4)."""

    if sync_index is None or sync_index.empty:
        raise EthogramSummaryError("sync_index is required for session_timing summary.")
    if abs(float(fps) - TARGET_FPS) > 1e-9:
        raise EthogramSummaryError(f"Ethogram summary requires fps={TARGET_FPS}.")

    rows: list[dict[str, Any]] = []
    rows.extend(_session_timing_rows(sync_index, fps=float(fps)))
    rows.extend(_bout_count_rows(packaged_bouts))
    rows.extend(
        _label_time_rows(
            states,
            n_directional_denominator=_directional_denominator(states, sync_index),
        )
    )
    rows.extend(
        _parallel_overlap_rows(
            packaged_bouts,
            states=states,
            s8_qc_metrics=s8_qc_metrics,
        )
    )

    if not rows:
        return _empty_summary_frame()
    out = pd.DataFrame(rows)
    out = out.loc[:, list(ETHOGRAM_SUMMARY_COLUMNS)]
    return out.sort_values(
        by=[
            "summary_kind",
            "channel",
            "behavior_name",
            "identity_key",
            "metric_name",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _row(
    *,
    summary_kind: str,
    metric_name: str,
    metric_value: float | int,
    channel: str = "*",
    behavior_name: str = "*",
    identity_key: str = "*",
    identity_grain: str = "session",
    denominator_name: str | None = None,
    denominator_value: float | int | None = None,
    unit: str = "count",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": ETHOGRAM_SUMMARY_SCHEMA_VERSION,
        "summary_kind": summary_kind,
        "channel": channel,
        "behavior_name": behavior_name,
        "identity_key": identity_key,
        "identity_grain": identity_grain,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "denominator_name": denominator_name,
        "denominator_value": (
            float(denominator_value) if denominator_value is not None else pd.NA
        ),
        "unit": unit,
        "notes": notes,
    }


def _session_timing_rows(sync_index: pd.DataFrame, *, fps: float) -> list[dict[str, Any]]:
    sample_idx = sync_index["sample_idx"].to_numpy(dtype=np.int64)
    time_sec = sync_index["time_sec"].to_numpy(dtype=np.float64)
    prepared = sync_index["prepared_frame_idx"].to_numpy(dtype=np.int64)
    n_dense = int(len(sync_index))
    n_prepared_unique = int(np.unique(prepared).shape[0])
    t0 = float(time_sec[0]) if n_dense else float("nan")
    t1 = float(time_sec[-1]) if n_dense else float("nan")
    notes = (
        "Behavioral clock is time_sec @ 120 Hz; prepared_frame_idx is video seek. "
        "Dense and prepared lengths are not interchangeable."
    )
    return [
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="n_samples",
            metric_value=n_dense,
            unit="count",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="sample_idx_min",
            metric_value=int(sample_idx.min()) if n_dense else -1,
            unit="index",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="sample_idx_max",
            metric_value=int(sample_idx.max()) if n_dense else -1,
            unit="index",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="t0_sec",
            metric_value=t0,
            unit="seconds",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="t1_sec",
            metric_value=t1,
            unit="seconds",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="fps",
            metric_value=fps,
            unit="Hz",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="n_prepared_unique",
            metric_value=n_prepared_unique,
            unit="count",
            notes=notes,
        ),
        _row(
            summary_kind=SUMMARY_KIND_SESSION_TIMING,
            metric_name="dense_ne_prepared_unique",
            metric_value=1.0 if n_dense != n_prepared_unique else 0.0,
            unit="bool",
            notes=notes,
        ),
    ]


def _bout_count_rows(bouts: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if bouts is None or bouts.empty:
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_BOUT_COUNTS,
                metric_name="n_bouts",
                metric_value=0,
                notes="Empty packaged bout table.",
            )
        )
        return rows

    rows.append(
        _row(
            summary_kind=SUMMARY_KIND_BOUT_COUNTS,
            metric_name="n_bouts",
            metric_value=int(len(bouts)),
            notes="Complete packaged S8 bout cardinality (unfiltered).",
        )
    )

    grouped = bouts.groupby(["channel", "behavior_name", "identity_grain"], dropna=False)
    for (channel, behavior, grain), block in grouped:
        ch = str(channel)
        beh = str(behavior)
        gr = str(grain)
        n = int(len(block))
        dur_frames = int(block["bout_duration_frames"].sum())
        dur_sec = float(block["bout_duration_sec"].sum())
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_BOUT_COUNTS,
                channel=ch,
                behavior_name=beh,
                identity_grain=gr,
                identity_key="*",
                metric_name="n_bouts",
                metric_value=n,
                unit="count",
            )
        )
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_BOUT_COUNTS,
                channel=ch,
                behavior_name=beh,
                identity_grain=gr,
                identity_key="*",
                metric_name="duration_frames_sum",
                metric_value=dur_frames,
                unit="frames",
                notes="Sum of inclusive bout_duration_frames (S8 authority).",
            )
        )
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_BOUT_COUNTS,
                channel=ch,
                behavior_name=beh,
                identity_grain=gr,
                identity_key="*",
                metric_name="duration_sec_sum",
                metric_value=dur_sec,
                unit="seconds",
                notes="Sum of bout_duration_sec = frames/120 (S8 authority).",
            )
        )
        for bucket_name, lo, hi in _DURATION_FRAME_BUCKETS:
            lo_i = int(lo) if lo is not None else 0
            mask = block["bout_duration_frames"] >= lo_i
            if hi is not None:
                mask = mask & (block["bout_duration_frames"] <= int(hi))
            rows.append(
                _row(
                    summary_kind=SUMMARY_KIND_BOUT_COUNTS,
                    channel=ch,
                    behavior_name=beh,
                    identity_grain=gr,
                    identity_key="*",
                    metric_name=f"duration_hist_{bucket_name}",
                    metric_value=int(mask.sum()),
                    unit="count",
                    notes=(
                        f"Inclusive frame bucket [{lo_i}, "
                        f"{hi if hi is not None else 'inf'}]."
                    ),
                )
            )
    return rows


def _directional_denominator(
    states: pd.DataFrame | None,
    sync_index: pd.DataFrame,
) -> int:
    if states is not None and not states.empty:
        return int(len(states))
    # Without states, label_time cannot be computed; session still has dense count.
    return int(len(sync_index))


def _label_time_rows(
    states: pd.DataFrame | None,
    *,
    n_directional_denominator: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    denom_name = "n_directional_state_rows"
    if states is None or states.empty or "behavior_label" not in getattr(states, "columns", []):
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_LABEL_TIME,
                channel="exclusive",
                metric_name="label_time_unavailable",
                metric_value=1.0,
                denominator_name=denom_name,
                denominator_value=n_directional_denominator,
                unit="bool",
                notes=(
                    "Framewise exclusive label coverage requires S8 states; "
                    "not inferred from bouts alone."
                ),
            )
        )
        return rows

    labels = states["behavior_label"].astype(str)
    denom = int(len(states))
    rows.append(
        _row(
            summary_kind=SUMMARY_KIND_LABEL_TIME,
            channel="exclusive",
            metric_name="n_directional_samples",
            metric_value=denom,
            denominator_name=denom_name,
            denominator_value=denom,
            unit="count",
            notes=(
                "Denominator for exclusive label proportions: all directional "
                "state rows (not unique dense samples)."
            ),
        )
    )
    counts = labels.value_counts(dropna=False)
    for behavior, count in counts.items():
        beh = "undefined" if pd.isna(behavior) else str(behavior)
        n = int(count)
        frac = float(n) / float(denom) if denom else 0.0
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_LABEL_TIME,
                channel="exclusive",
                behavior_name=beh,
                identity_grain="directional",
                identity_key="*",
                metric_name="n_frames",
                metric_value=n,
                denominator_name=denom_name,
                denominator_value=denom,
                unit="frames",
            )
        )
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_LABEL_TIME,
                channel="exclusive",
                behavior_name=beh,
                identity_grain="directional",
                identity_key="*",
                metric_name="fraction_of_directional_samples",
                metric_value=frac,
                denominator_name=denom_name,
                denominator_value=denom,
                unit="fraction",
            )
        )
    # Engaged / undefined explicit headline metrics.
    for special in ("engaged", "undefined"):
        n_special = int((labels == special).sum())
        rows.append(
            _row(
                summary_kind=SUMMARY_KIND_LABEL_TIME,
                channel="exclusive",
                behavior_name=special,
                identity_grain="directional",
                identity_key="*",
                metric_name="n_frames_special",
                metric_value=n_special,
                denominator_name=denom_name,
                denominator_value=denom,
                unit="frames",
                notes="Explicit engaged/undefined handling (may duplicate n_frames).",
            )
        )
    return rows


def _parallel_overlap_rows(
    bouts: pd.DataFrame,
    *,
    states: pd.DataFrame | None,
    s8_qc_metrics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    notes = (
        "Parallel contact/SBS coverage is independent of exclusive ethogram labels; "
        "not mutually exclusive with behavior_label."
    )
    for channel in ("contact", "side_by_side"):
        if bouts is not None and not bouts.empty and "channel" in bouts.columns:
            block = bouts[bouts["channel"].astype(str) == channel]
            rows.append(
                _row(
                    summary_kind=SUMMARY_KIND_PARALLEL_OVERLAP,
                    channel=channel,
                    behavior_name=channel,
                    identity_grain="symmetric_pair",
                    metric_name="n_bouts",
                    metric_value=int(len(block)),
                    unit="count",
                    notes=notes,
                )
            )
        else:
            rows.append(
                _row(
                    summary_kind=SUMMARY_KIND_PARALLEL_OVERLAP,
                    channel=channel,
                    behavior_name=channel,
                    identity_grain="symmetric_pair",
                    metric_name="n_bouts",
                    metric_value=0,
                    unit="count",
                    notes=notes,
                )
            )

        # Framewise parallel positives from states when present (not from bouts).
        pred_col = f"pred_{channel}"
        if states is not None and pred_col in states.columns:
            preds = pd.to_numeric(states[pred_col], errors="coerce").fillna(0).astype(int)
            n_pos = int(preds.eq(1).sum())
            denom = int(len(states))
            rows.append(
                _row(
                    summary_kind=SUMMARY_KIND_PARALLEL_OVERLAP,
                    channel=channel,
                    behavior_name=channel,
                    identity_grain="symmetric_pair",
                    metric_name="n_positive_state_rows",
                    metric_value=n_pos,
                    denominator_name="n_directional_state_rows",
                    denominator_value=denom,
                    unit="frames",
                    notes=notes + " Counted from S8 states pred_* columns.",
                )
            )
            rows.append(
                _row(
                    summary_kind=SUMMARY_KIND_PARALLEL_OVERLAP,
                    channel=channel,
                    behavior_name=channel,
                    identity_grain="symmetric_pair",
                    metric_name="positive_fraction_of_state_rows",
                    metric_value=float(n_pos) / float(denom) if denom else 0.0,
                    denominator_name="n_directional_state_rows",
                    denominator_value=denom,
                    unit="fraction",
                    notes=notes,
                )
            )
        else:
            rows.append(
                _row(
                    summary_kind=SUMMARY_KIND_PARALLEL_OVERLAP,
                    channel=channel,
                    behavior_name=channel,
                    identity_grain="symmetric_pair",
                    metric_name="framewise_parallel_unavailable",
                    metric_value=1.0,
                    unit="bool",
                    notes=(
                        notes
                        + " Framewise parallel stats unavailable; not inferred from bouts."
                    ),
                )
            )

        # Echo S8 QC parallel coverage when provided (no invention).
        if s8_qc_metrics is not None:
            coverage = s8_qc_metrics.get("parallel_coverage")
            if isinstance(coverage, Mapping) and channel in coverage:
                rows.append(
                    _row(
                        summary_kind=SUMMARY_KIND_PARALLEL_OVERLAP,
                        channel=channel,
                        behavior_name=channel,
                        identity_grain="symmetric_pair",
                        metric_name="s8_qc_parallel_coverage_present",
                        metric_value=1.0,
                        unit="bool",
                        notes=json.dumps(coverage.get(channel), sort_keys=True),
                    )
                )
    return rows


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in ETHOGRAM_SUMMARY_COLUMNS})
