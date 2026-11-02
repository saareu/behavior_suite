"""Build the legacy framewise table from already-resolved sources.

Formatting and I/O are handled by ``writer``. This module performs joins only —
no scientific relabeling, no fabricated coordinates, no invented frame indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from results_packaging.legacy_framewise_csv.contracts import (
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_POSE_NODE_TRACKS,
    LegacyFramewiseEligibilityError,
    LegacyFramewiseSourceError,
)


@dataclass(frozen=True, slots=True)
class LegacyFramewiseBuildResult:
    """In-memory framewise table plus join diagnostics for the manifest."""

    table: pd.DataFrame
    n_rows: int
    n_unique_original_frames: int
    n_duplicate_original_frame_rows: int
    n_missing_pose_coordinates: int
    n_empty_behavior_labels: int
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]


def build_legacy_framewise_table(
    *,
    sync_index: pd.DataFrame,
    behavior_states: pd.DataFrame,
    final_pose: pd.DataFrame,
) -> LegacyFramewiseBuildResult:
    """Assemble one output row per dense sync sample.

    Parameters
    ----------
    sync_index:
        S9 sync index; must include ``dense_frame_idx`` and
        ``raw_decode_frame_idx``.
    behavior_states:
        Framewise exclusive states with ``dense_frame_idx``, ``focal_id``,
        ``behavior_label``, and pair-level ``pred_side_by_side``.
    final_pose:
        S4 ``final_pose`` long table with ``sample_idx``, ``track``, ``node``,
        ``x``, ``y``.
    """

    sync, dense_diag = _normalize_sync(sync_index)
    labels = _pivot_behavior_labels(behavior_states, expected_dense=sync["dense_frame_idx"])
    pose = _pivot_pose(final_pose, expected_dense=sync["dense_frame_idx"])

    frame = sync.merge(labels, on="dense_frame_idx", how="left", validate="one_to_one")
    frame = frame.merge(pose, on="dense_frame_idx", how="left", validate="one_to_one")
    frame = frame.sort_values("dense_frame_idx", kind="mergesort").reset_index(drop=True)

    # Ensure all contract columns exist before projection.
    for col in LEGACY_FRAMEWISE_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA

    out = frame.loc[:, list(LEGACY_FRAMEWISE_COLUMNS)].copy()
    # Integer frame indices as nullable ints for writer.
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")
    out["dense_frame_idx"] = pd.to_numeric(out["dense_frame_idx"], errors="coerce").astype(
        "Int64"
    )
    if out["frame"].isna().any():
        raise LegacyFramewiseEligibilityError(
            "Unresolved original-frame mapping (NaN raw_decode_frame_idx) after join."
        )
    if out["dense_frame_idx"].isna().any():
        raise LegacyFramewiseSourceError("dense_frame_idx became null during join.")

    # Output cardinality must remain 1:1 with authoritative sync dense samples.
    # Never renumber or invent missing dense indices.
    out_dense = out["dense_frame_idx"].astype(np.int64).to_numpy()
    if len(out_dense) != int(dense_diag["n_sync_rows"]):
        raise LegacyFramewiseSourceError(
            "Output row count diverged from authoritative sync_index cardinality: "
            f"{len(out_dense)} vs {dense_diag['n_sync_rows']}."
        )
    if (
        int(out_dense.min()) != int(dense_diag["dense_min"])
        or int(out_dense.max()) != int(dense_diag["dense_max"])
        or int(pd.Series(out_dense).nunique()) != int(dense_diag["n_unique_dense"])
        or bool(pd.Series(out_dense).duplicated().any())
    ):
        raise LegacyFramewiseSourceError(
            "Output dense_frame_idx continuity/uniqueness violated after join."
        )
    expected = np.arange(
        int(dense_diag["dense_min"]),
        int(dense_diag["dense_max"]) + 1,
        dtype=np.int64,
    )
    if not np.array_equal(out_dense, expected):
        raise LegacyFramewiseSourceError(
            "Output dense_frame_idx is not the authoritative contiguous sequence "
            f"[{dense_diag['dense_min']}..{dense_diag['dense_max']}]."
        )

    # Side-by-side: keep 0/1 integers when present; leave missing as NA.
    sbs = pd.to_numeric(out["pred_side_by_side"], errors="coerce")
    out["pred_side_by_side"] = sbs.round().astype("Int64")

    warnings: list[str] = []
    empty_labels = 0
    for col in ("behaviour_label_focal_0", "behaviour_label_focal_1"):
        series = out[col]
        empty_labels += int(series.isna().sum())
        empty_labels += int(
            (~series.isna() & (series.astype(str).str.strip() == "")).sum()
        )
    if empty_labels:
        warnings.append(
            f"{empty_labels} empty behaviour_label cells (missing directional states "
            "preserved as empty fields; not substituted)."
        )

    pose_cols = [c for c in LEGACY_FRAMEWISE_COLUMNS if c.endswith("_x") or c.endswith("_y")]
    missing_pose = int(sum(out[c].isna().sum() for c in pose_cols))
    n_unique_orig = int(out["frame"].nunique(dropna=True))
    n_dup_orig = int(len(out) - n_unique_orig)
    frame_vals = out["frame"].astype(np.int64)
    trim_offset = int(frame_vals.min()) if len(frame_vals) else None

    return LegacyFramewiseBuildResult(
        table=out,
        n_rows=int(len(out)),
        n_unique_original_frames=n_unique_orig,
        n_duplicate_original_frame_rows=n_dup_orig,
        n_missing_pose_coordinates=missing_pose,
        n_empty_behavior_labels=empty_labels,
        warnings=tuple(warnings),
        diagnostics={
            "n_sync_rows": int(len(sync)),
            "coordinate_space": "prepared_video_pixels",
            "pose_join_key": "sample_idx==dense_frame_idx",
            "behavior_join_key": "dense_frame_idx",
            "original_frame_column": "raw_decode_frame_idx",
            "original_frame_indexing": "0_based",
            "original_frame_min": int(frame_vals.min()) if len(frame_vals) else None,
            "original_frame_max": int(frame_vals.max()) if len(frame_vals) else None,
            "nonzero_trim_offset": bool(trim_offset not in (None, 0)),
            "trim_offset_frames": trim_offset,
            "many_to_one_original_frames": n_dup_orig > 0,
            **dense_diag,
            "focal0_present_rows": int(out["behaviour_label_focal_0"].notna().sum()),
            "focal1_present_rows": int(out["behaviour_label_focal_1"].notna().sum()),
        },
    )


def _normalize_sync(sync_index: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if sync_index is None or sync_index.empty:
        raise LegacyFramewiseEligibilityError("sync_index is empty.")
    required = {"dense_frame_idx", "raw_decode_frame_idx"}
    missing = sorted(required - set(sync_index.columns))
    if missing:
        raise LegacyFramewiseEligibilityError(
            "sync_index missing original-frame mapping columns: "
            f"{missing}. raw_decode_frame_idx is required for legacy_framewise_v1 "
            "(do not invent dense+trim_start)."
        )
    work = sync_index.loc[:, ["dense_frame_idx", "raw_decode_frame_idx"]].copy()
    work["dense_frame_idx"] = pd.to_numeric(work["dense_frame_idx"], errors="coerce")
    work["raw_decode_frame_idx"] = pd.to_numeric(
        work["raw_decode_frame_idx"], errors="coerce"
    )
    if work["dense_frame_idx"].isna().any():
        raise LegacyFramewiseEligibilityError("sync_index has NaN dense_frame_idx.")
    if work["raw_decode_frame_idx"].isna().any():
        raise LegacyFramewiseEligibilityError(
            "Unresolved original-frame mapping: sync_index.raw_decode_frame_idx "
            "contains NaN. Export refused."
        )
    n_rows = int(len(work))
    n_unique = int(work["dense_frame_idx"].nunique(dropna=True))
    n_dups = int(work["dense_frame_idx"].duplicated().sum())
    dense_min = int(work["dense_frame_idx"].min())
    dense_max = int(work["dense_frame_idx"].max())
    expected = set(range(dense_min, dense_max + 1))
    actual = set(int(v) for v in work["dense_frame_idx"].astype(np.int64).tolist())
    missing_dense = sorted(expected - actual)
    extra_dense = sorted(actual - expected)
    if n_dups:
        raise LegacyFramewiseSourceError(
            f"sync_index dense_frame_idx must be unique (duplicates={n_dups})."
        )
    if missing_dense or extra_dense:
        raise LegacyFramewiseSourceError(
            "Authoritative sync_index dense_frame_idx is not contiguous without gaps: "
            f"min={dense_min}, max={dense_max}, n_rows={n_rows}, n_unique={n_unique}, "
            f"missing={len(missing_dense)}, extra={len(extra_dense)}; "
            f"missing_examples={missing_dense[:5]}, extra_examples={extra_dense[:5]}. "
            "Export refuses renumbering or silent fill."
        )
    if dense_min != 0:
        raise LegacyFramewiseSourceError(
            f"Authoritative dense_frame_idx must be 0-based; found min={dense_min}."
        )
    if n_rows != (dense_max - dense_min + 1) or n_unique != n_rows:
        raise LegacyFramewiseSourceError(
            "Authoritative dense_frame_idx cardinality mismatch: "
            f"n_rows={n_rows}, unique={n_unique}, span={dense_max - dense_min + 1}."
        )

    work = work.sort_values("dense_frame_idx", kind="mergesort").reset_index(drop=True)
    work["dense_frame_idx"] = work["dense_frame_idx"].astype(np.int64)
    work["frame"] = work["raw_decode_frame_idx"].astype(np.int64)
    if (work["frame"] < 0).any():
        raise LegacyFramewiseSourceError(
            "raw_decode_frame_idx contains negative values; original frames are 0-based."
        )
    diag = {
        "dense_min": dense_min,
        "dense_max": dense_max,
        "n_unique_dense": n_unique,
        "n_duplicate_dense": n_dups,
        "n_missing_dense": len(missing_dense),
        "dense_contiguous_from_zero": True,
        "n_sync_rows": n_rows,
    }
    return work.loc[:, ["dense_frame_idx", "frame"]], diag


def _pivot_behavior_labels(
    behavior_states: pd.DataFrame,
    *,
    expected_dense: pd.Series,
) -> pd.DataFrame:
    if behavior_states is None or behavior_states.empty:
        raise LegacyFramewiseSourceError("behavior_states table is empty.")
    required = {"dense_frame_idx", "focal_id", "behavior_label", "pred_side_by_side"}
    missing = sorted(required - set(behavior_states.columns))
    if missing:
        raise LegacyFramewiseSourceError(
            f"behavior_states missing required columns: {missing}"
        )

    work = behavior_states.loc[
        :, ["dense_frame_idx", "focal_id", "behavior_label", "pred_side_by_side"]
    ].copy()
    work["dense_frame_idx"] = pd.to_numeric(work["dense_frame_idx"], errors="coerce")
    work["focal_id"] = pd.to_numeric(work["focal_id"], errors="coerce")
    if work["dense_frame_idx"].isna().any() or work["focal_id"].isna().any():
        raise LegacyFramewiseSourceError(
            "behavior_states has NaN dense_frame_idx or focal_id."
        )
    work["dense_frame_idx"] = work["dense_frame_idx"].astype(np.int64)
    work["focal_id"] = work["focal_id"].astype(np.int64)

    # Pair-level SBS must agree across directional rows for the same dense sample.
    sbs = pd.to_numeric(work["pred_side_by_side"], errors="coerce")
    work["pred_side_by_side"] = sbs
    conflict = (
        work.groupby("dense_frame_idx")["pred_side_by_side"]
        .nunique(dropna=True)
        .gt(1)
    )
    if bool(conflict.any()):
        bad = conflict[conflict].index[:5].tolist()
        raise LegacyFramewiseSourceError(
            "pred_side_by_side disagrees across focal directions for dense_frame_idx "
            f"examples={bad}."
        )

    # Do not substitute missing/invalid/undefined labels.
    labels = work["behavior_label"]
    labels = labels.where(labels.notna(), other=pd.NA)
    work["behavior_label"] = labels.map(
        lambda v: pd.NA if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    )

    # Reject conflicting duplicate directional rows for the same dense sample.
    for focal in (0, 1):
        subset = work.loc[work["focal_id"] == focal]
        conflict = (
            subset.groupby("dense_frame_idx")["behavior_label"]
            .nunique(dropna=False)
            .gt(1)
        )
        if bool(conflict.any()):
            bad = conflict[conflict].index[:5].tolist()
            raise LegacyFramewiseSourceError(
                f"Conflicting behaviour labels for focal_id={focal} at "
                f"dense_frame_idx examples={bad}."
            )

    focal0 = (
        work.loc[work["focal_id"] == 0, ["dense_frame_idx", "behavior_label"]]
        .drop_duplicates("dense_frame_idx", keep="first")
        .rename(columns={"behavior_label": "behaviour_label_focal_0"})
    )
    focal1 = (
        work.loc[work["focal_id"] == 1, ["dense_frame_idx", "behavior_label"]]
        .drop_duplicates("dense_frame_idx", keep="first")
        .rename(columns={"behavior_label": "behaviour_label_focal_1"})
    )
    # States outside the authoritative sync dense grid are rejected (no silent drop
    # of misaligned science). Missing directions remain empty after the left join.
    state_dense = set(int(v) for v in work["dense_frame_idx"].unique().tolist())
    expected_set = set(int(v) for v in expected_dense.astype(np.int64).tolist())
    extra_state = sorted(state_dense - expected_set)
    if extra_state:
        raise LegacyFramewiseSourceError(
            "behavior_states contain dense_frame_idx values outside the "
            f"authoritative sync grid (extra={len(extra_state)}, "
            f"examples={extra_state[:5]})."
        )
    sbs_by_dense = (
        work.groupby("dense_frame_idx", sort=True)["pred_side_by_side"]
        .first()
        .rename("pred_side_by_side")
        .reset_index()
    )

    base = pd.DataFrame({"dense_frame_idx": expected_dense.astype(np.int64).to_numpy()})
    out = (
        base.merge(focal0, on="dense_frame_idx", how="left")
        .merge(focal1, on="dense_frame_idx", how="left")
        .merge(sbs_by_dense, on="dense_frame_idx", how="left")
    )
    return out


def _pivot_pose(
    final_pose: pd.DataFrame,
    *,
    expected_dense: pd.Series,
) -> pd.DataFrame:
    if final_pose is None or final_pose.empty:
        raise LegacyFramewiseSourceError("final_pose table is empty.")
    required = {"sample_idx", "track", "node", "x", "y"}
    missing = sorted(required - set(final_pose.columns))
    if missing:
        raise LegacyFramewiseSourceError(f"final_pose missing required columns: {missing}")

    work = final_pose.loc[:, ["sample_idx", "track", "node", "x", "y"]].copy()
    work["sample_idx"] = pd.to_numeric(work["sample_idx"], errors="coerce")
    work["track"] = pd.to_numeric(work["track"], errors="coerce")
    if work["sample_idx"].isna().any() or work["track"].isna().any():
        raise LegacyFramewiseSourceError("final_pose has NaN sample_idx or track.")
    work["sample_idx"] = work["sample_idx"].astype(np.int64)
    work["track"] = work["track"].astype(np.int64)
    work["node"] = work["node"].astype(str)

    wanted = {(node, track) for node, track in LEGACY_POSE_NODE_TRACKS}
    mask = [
        (str(node), int(track)) in wanted
        for node, track in zip(
            work["node"].to_numpy(), work["track"].to_numpy(), strict=True
        )
    ]
    work = work.loc[mask].copy()

    base = pd.DataFrame({"dense_frame_idx": expected_dense.astype(np.int64).to_numpy()})
    out = base.copy()
    for node, track in LEGACY_POSE_NODE_TRACKS:
        subset = work.loc[(work["node"] == node) & (work["track"] == track)]
        # One row per sample; if duplicates, keep first deterministically.
        subset = (
            subset.sort_values("sample_idx", kind="mergesort")
            .drop_duplicates("sample_idx", keep="first")
            .loc[:, ["sample_idx", "x", "y"]]
            .rename(
                columns={
                    "sample_idx": "dense_frame_idx",
                    "x": f"{node}_{track}_x",
                    "y": f"{node}_{track}_y",
                }
            )
        )
        out = out.merge(subset, on="dense_frame_idx", how="left", validate="one_to_one")

    # Never fabricate: leave missing as NA for the writer.
    return out
