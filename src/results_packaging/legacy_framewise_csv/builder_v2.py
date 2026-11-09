"""Build legacy_framewise_v2 tables on the S1 prepared-video frame domain.

One CSV row per prepared frame. Behavioral and S4 pose dense samples are
resolved independently via SyncMap + LAST_IN_PLATEAU (overlay timing semantics).
Missing coverage retains the row with empty fields from that source only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.sync import SyncMap
from results_packaging.legacy_framewise_csv.builder import (
    _pivot_behavior_labels,
    _pivot_pose,
)
from results_packaging.legacy_framewise_csv.contracts import (
    FRAME_ALIGNMENT_COLUMNS,
    LEGACY_FRAMEWISE_COLUMNS,
    PLATEAU_POLICY_LAST_IN_PLATEAU,
    LegacyFramewiseEligibilityError,
    LegacyFramewiseSourceError,
)


@dataclass(frozen=True, slots=True)
class LegacyFramewiseV2BuildResult:
    """Prepared-frame laboratory table plus alignment sidecar diagnostics."""

    table: pd.DataFrame
    alignment: pd.DataFrame
    n_rows: int
    n_prepared_frames: int
    n_behavior_covered: int
    n_pose_covered: int
    n_unique_original_frames: int
    n_duplicate_original_frame_rows: int
    n_missing_pose_coordinates: int
    n_empty_behavior_labels: int
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]


def build_legacy_framewise_v2_table(
    *,
    prepared_frame_idx: np.ndarray,
    raw_decode_frame_idx: np.ndarray,
    behavior_sync: SyncMap,
    pose_sync: SyncMap,
    behavior_states: pd.DataFrame,
    final_pose: pd.DataFrame,
) -> LegacyFramewiseV2BuildResult:
    """Assemble one output row per S1 prepared-video frame.

    ``frame`` is the authoritative S1 ``raw_decode_frame_idx``. CSV
    ``dense_frame_idx`` is the behavioral SyncMap LAST_IN_PLATEAU sample
    (nullable). Pose values use the independently resolved S4 SyncMap sample.
    """

    prep, raw = _normalize_prepared_universe(
        prepared_frame_idx=prepared_frame_idx,
        raw_decode_frame_idx=raw_decode_frame_idx,
    )
    n_prepared = int(len(prep))

    behavior_dense = prepared_to_last_in_plateau_dense(
        behavior_sync, n_prepared=n_prepared
    )
    pose_dense = prepared_to_last_in_plateau_dense(pose_sync, n_prepared=n_prepared)

    behavior_expected = pd.Series(
        behavior_sync.dense_frame_idx.astype(np.int64), name="dense_frame_idx"
    )
    pose_expected = pd.Series(
        pose_sync.dense_frame_idx.astype(np.int64), name="dense_frame_idx"
    )
    labels = _pivot_behavior_labels(
        behavior_states, expected_dense=behavior_expected
    ).set_index("dense_frame_idx")
    pose = _pivot_pose(final_pose, expected_dense=pose_expected).set_index(
        "dense_frame_idx"
    )

    label_cols = [
        "behaviour_label_focal_0",
        "behaviour_label_focal_1",
        "pred_side_by_side",
    ]
    pose_cols = [c for c in LEGACY_FRAMEWISE_COLUMNS if c.endswith("_x") or c.endswith("_y")]

    behavior_covered = behavior_dense >= 0
    pose_covered = pose_dense >= 0

    out = pd.DataFrame(index=np.arange(n_prepared, dtype=np.int64))
    out["frame"] = raw.astype(np.int64)
    out["dense_frame_idx"] = pd.Series(behavior_dense, dtype="Int64").where(
        behavior_covered, other=pd.NA
    )
    for col in label_cols:
        out[col] = pd.NA
    for col in pose_cols:
        out[col] = np.nan

    if bool(behavior_covered.any()):
        selected = behavior_dense[behavior_covered]
        mapped = labels.reindex(selected)
        for col in label_cols:
            out.loc[behavior_covered, col] = mapped[col].to_numpy()
    if bool(pose_covered.any()):
        selected_pose = pose_dense[pose_covered]
        mapped_pose = pose.reindex(selected_pose)
        for col in pose_cols:
            out.loc[pose_covered, col] = mapped_pose[col].to_numpy()

    for col in LEGACY_FRAMEWISE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    table = out.loc[:, list(LEGACY_FRAMEWISE_COLUMNS)].copy()
    table["frame"] = pd.to_numeric(table["frame"], errors="coerce").astype("Int64")
    table["dense_frame_idx"] = pd.to_numeric(
        table["dense_frame_idx"], errors="coerce"
    ).astype("Int64")
    if table["frame"].isna().any():
        raise LegacyFramewiseEligibilityError(
            "Unresolved original-frame mapping (NaN raw_decode_frame_idx) for a "
            "prepared frame."
        )
    sbs = pd.to_numeric(table["pred_side_by_side"], errors="coerce")
    table["pred_side_by_side"] = sbs.round().astype("Int64")

    alignment = pd.DataFrame(
        {
            "prepared_frame_idx": prep.astype(np.int64),
            "raw_decode_frame_idx": raw.astype(np.int64),
            "behavior_dense_frame_idx": pd.Series(behavior_dense, dtype="Int64").where(
                behavior_covered, other=pd.NA
            ),
            "pose_dense_frame_idx": pd.Series(pose_dense, dtype="Int64").where(
                pose_covered, other=pd.NA
            ),
            "behavior_coverage": behavior_covered.astype(bool),
            "pose_coverage": pose_covered.astype(bool),
        }
    )
    if list(alignment.columns) != list(FRAME_ALIGNMENT_COLUMNS):
        raise LegacyFramewiseSourceError(
            "Frame-alignment sidecar columns diverged from contract."
        )
    if int(len(alignment)) != n_prepared or int(len(table)) != n_prepared:
        raise LegacyFramewiseSourceError(
            "Prepared-frame CSV/sidecar cardinality mismatch with S1 universe: "
            f"table={len(table)}, alignment={len(alignment)}, n_prepared={n_prepared}."
        )
    if bool(alignment["prepared_frame_idx"].duplicated().any()):
        raise LegacyFramewiseSourceError(
            "Duplicate prepared_frame_idx in alignment sidecar."
        )

    warnings: list[str] = []
    empty_labels = 0
    for col in ("behaviour_label_focal_0", "behaviour_label_focal_1"):
        series = table[col]
        empty_labels += int(series.isna().sum())
        empty_labels += int(
            (~series.isna() & (series.astype(str).str.strip() == "")).sum()
        )
    n_behavior_covered = int(behavior_covered.sum())
    n_pose_covered = int(pose_covered.sum())
    n_behavior_missing = n_prepared - n_behavior_covered
    n_pose_missing = n_prepared - n_pose_covered
    if n_behavior_missing:
        warnings.append(
            f"{n_behavior_missing} prepared frames lack behavioral SyncMap coverage "
            "(labels left empty; rows retained)."
        )
    if n_pose_missing:
        warnings.append(
            f"{n_pose_missing} prepared frames lack independent S4 SyncMap coverage "
            "(pose left empty; rows retained)."
        )
    if empty_labels:
        warnings.append(
            f"{empty_labels} empty behaviour_label cells (missing directional states "
            "preserved as empty fields; not substituted)."
        )

    missing_pose = int(sum(table[c].isna().sum() for c in pose_cols))
    n_unique_orig = int(table["frame"].nunique(dropna=True))
    n_dup_orig = int(len(table) - n_unique_orig)
    frame_vals = table["frame"].astype(np.int64)
    trim_offset = int(frame_vals.min()) if len(frame_vals) else None

    behavior_audit = behavior_sync.audit_dict()
    pose_audit = pose_sync.audit_dict()
    return LegacyFramewiseV2BuildResult(
        table=table,
        alignment=alignment,
        n_rows=n_prepared,
        n_prepared_frames=n_prepared,
        n_behavior_covered=n_behavior_covered,
        n_pose_covered=n_pose_covered,
        n_unique_original_frames=n_unique_orig,
        n_duplicate_original_frame_rows=n_dup_orig,
        n_missing_pose_coordinates=missing_pose,
        n_empty_behavior_labels=empty_labels,
        warnings=tuple(warnings),
        diagnostics={
            "row_domain": "s1_prepared_video_frames",
            "n_prepared_frames": n_prepared,
            "n_behavior_covered": n_behavior_covered,
            "n_pose_covered": n_pose_covered,
            "n_behavior_missing": n_behavior_missing,
            "n_pose_missing": n_pose_missing,
            "coordinate_space": "prepared_video_pixels",
            "behavior_join_key": "behavior_sync_last_in_plateau_dense",
            "pose_join_key": "pose_sync_last_in_plateau_dense",
            "behavior_plateau_policy": PLATEAU_POLICY_LAST_IN_PLATEAU,
            "pose_plateau_policy": PLATEAU_POLICY_LAST_IN_PLATEAU,
            "original_frame_column": "s1_prepared_sync.raw_decode_frame_idx",
            "original_frame_indexing": "0_based",
            "original_frame_min": int(frame_vals.min()) if len(frame_vals) else None,
            "original_frame_max": int(frame_vals.max()) if len(frame_vals) else None,
            "nonzero_trim_offset": bool(trim_offset not in (None, 0)),
            "trim_offset_frames": trim_offset,
            "many_to_one_original_frames": n_dup_orig > 0,
            "behavior_sync": behavior_audit,
            "pose_sync": pose_audit,
            "independent_timing_authorities": True,
            "behavior_n_dense": int(behavior_sync.n_dense),
            "pose_n_dense": int(pose_sync.n_dense),
            "dense_indices_assumed_equal": False,
            "focal0_present_rows": int(table["behaviour_label_focal_0"].notna().sum()),
            "focal1_present_rows": int(table["behaviour_label_focal_1"].notna().sum()),
        },
    )


def prepared_to_last_in_plateau_dense(
    sync: SyncMap,
    *,
    n_prepared: int,
) -> np.ndarray:
    """Vectorized LAST_IN_PLATEAU map: prepared_frame_idx → dense sample_idx.

    Returns ``-1`` where the SyncMap has no dense sample for that prepared frame.
    Matches ``overlay.timing.resolve_prepared_frame_timing`` with
    ``PlateauPolicy.LAST_IN_PLATEAU`` (max sample_idx on the plateau).
    """

    if n_prepared < 0:
        raise LegacyFramewiseSourceError("n_prepared must be non-negative.")
    out = np.full(int(n_prepared), -1, dtype=np.int64)
    if sync.n_dense == 0:
        return out
    prep = sync.prepared_frame_idx.astype(np.int64, copy=False)
    sample = sync.sample_idx.astype(np.int64, copy=False)
    i = 0
    n = int(prep.shape[0])
    while i < n:
        p = int(prep[i])
        j = i + 1
        while j < n and int(prep[j]) == p:
            j += 1
        # sample_idx is ascending ⇒ last plateau member is chronological latest.
        if 0 <= p < n_prepared:
            out[p] = int(sample[j - 1])
        elif p >= n_prepared:
            raise LegacyFramewiseEligibilityError(
                f"SyncMap prepared_frame_idx {p} is outside the authoritative "
                f"S1 prepared-video universe [0, {n_prepared - 1}]."
            )
        i = j
    return out


def _normalize_prepared_universe(
    *,
    prepared_frame_idx: np.ndarray,
    raw_decode_frame_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if prepared_frame_idx is None or raw_decode_frame_idx is None:
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_sync prepared_frame_idx/raw_decode_frame_idx are required "
            "for legacy_framewise_v2."
        )
    prep = np.asarray(prepared_frame_idx, dtype=np.int64).reshape(-1)
    raw = np.asarray(raw_decode_frame_idx, dtype=np.int64).reshape(-1)
    if prep.shape[0] == 0:
        raise LegacyFramewiseEligibilityError("S1 prepared_sync has zero prepared frames.")
    if prep.shape[0] != raw.shape[0]:
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_sync prepared_frame_idx and raw_decode_frame_idx lengths "
            f"differ ({prep.shape[0]} vs {raw.shape[0]})."
        )
    expected = np.arange(prep.shape[0], dtype=np.int64)
    if not np.array_equal(prep, expected):
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_frame_idx must be the contiguous prepared-video universe "
            f"0..{prep.shape[0] - 1}; refusing renumbering."
        )
    if not np.all(np.isfinite(raw.astype(np.float64))):
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_sync.raw_decode_frame_idx contains non-finite values."
        )
    if bool((raw < 0).any()):
        raise LegacyFramewiseSourceError(
            "raw_decode_frame_idx contains negative values; original frames are 0-based."
        )
    return prep, raw
