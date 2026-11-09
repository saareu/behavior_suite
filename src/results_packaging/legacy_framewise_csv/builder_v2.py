"""Build legacy_framewise_v2 tables on the S1 prepared-video frame domain.

One CSV row per prepared frame. Behavioral labels come from the nearest
authoritative S8/S10 dense sample (earlier-on-tie). Pose coordinates come
directly from accepted S3 ``tracked_pose`` at the same prepared_frame_idx.
Missing coverage retains the row with empty fields from that source only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.sync import SyncMap
from results_packaging.contracts import TIME_ATOL_SEC
from results_packaging.legacy_framewise_csv.builder import _pivot_behavior_labels
from results_packaging.legacy_framewise_csv.contracts import (
    BEHAVIOR_MAPPING_PREPARED_NEAREST_DENSE,
    BEHAVIOR_TIE_BREAK_EARLIER,
    FRAME_ALIGNMENT_COLUMNS,
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
    LEGACY_POSE_NODE_TRACKS,
    LegacyFramewiseEligibilityError,
    LegacyFramewiseSourceError,
)

_S3_FRAME_CANDIDATES = ("prepared_frame_idx", "frame_idx", "frame")


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
    prepared_time_sec: np.ndarray,
    behavior_sync: SyncMap,
    behavior_states: pd.DataFrame,
    tracked_pose: pd.DataFrame,
    s3_run_id: str | None = None,
) -> LegacyFramewiseV2BuildResult:
    """Assemble one output row per S1 prepared-video frame.

    ``frame`` is the authoritative S1 ``raw_decode_frame_idx``. CSV
    ``dense_frame_idx`` is the behavioral prepared→nearest-dense sample
    (nullable). Pose values are accepted S3 tracked coordinates for the
    same ``prepared_frame_idx``.
    """

    prep, raw, prep_t = _normalize_prepared_universe(
        prepared_frame_idx=prepared_frame_idx,
        raw_decode_frame_idx=raw_decode_frame_idx,
        prepared_time_sec=prepared_time_sec,
    )
    n_prepared = int(len(prep))

    behavior_dense = prepared_to_nearest_dense(
        prepared_time_sec=prep_t,
        dense_time_sec=behavior_sync.time_sec,
        dense_sample_idx=behavior_sync.sample_idx,
    )
    pose_wide = _pivot_tracked_pose_by_prepared(
        tracked_pose, n_prepared=n_prepared
    )
    s3_available = pose_wide["_s3_pose_available"].to_numpy(dtype=bool)

    behavior_expected = pd.Series(
        behavior_sync.dense_frame_idx.astype(np.int64), name="dense_frame_idx"
    )
    labels = _pivot_behavior_labels(
        behavior_states, expected_dense=behavior_expected
    ).set_index("dense_frame_idx")

    label_cols = [
        "behaviour_label_focal_0",
        "behaviour_label_focal_1",
        "pred_side_by_side",
    ]
    pose_cols = [c for c in LEGACY_FRAMEWISE_COLUMNS if c.endswith("_x") or c.endswith("_y")]

    behavior_covered = behavior_dense >= 0

    out = pd.DataFrame(index=np.arange(n_prepared, dtype=np.int64))
    out["frame"] = raw.astype(np.int64)
    out["dense_frame_idx"] = pd.Series(behavior_dense, dtype="Int64").where(
        behavior_covered, other=pd.NA
    )
    for col in label_cols:
        out[col] = pd.NA
    for col in pose_cols:
        out[col] = pose_wide[col].to_numpy(dtype=np.float64)

    if bool(behavior_covered.any()):
        selected = behavior_dense[behavior_covered]
        mapped = labels.reindex(selected)
        for col in label_cols:
            out.loc[behavior_covered, col] = mapped[col].to_numpy()

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
            "behavior_coverage": behavior_covered.astype(bool),
            "s3_pose_available": s3_available.astype(bool),
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
    n_pose_covered = int(s3_available.sum())
    n_behavior_missing = n_prepared - n_behavior_covered
    n_pose_missing = n_prepared - n_pose_covered
    if n_behavior_missing:
        warnings.append(
            f"{n_behavior_missing} prepared frames lack behavioral nearest-dense "
            "coverage within source temporal bounds (labels left empty; rows retained)."
        )
    if n_pose_missing:
        warnings.append(
            f"{n_pose_missing} prepared frames lack accepted S3 tracked_pose "
            "observations (pose left empty; rows retained)."
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

    selected_dense = behavior_dense[behavior_covered]
    n_unique_behavior_dense = (
        int(np.unique(selected_dense).shape[0]) if selected_dense.size else 0
    )
    n_repeated_behavior_dense = int(selected_dense.size - n_unique_behavior_dense)

    behavior_audit = behavior_sync.audit_dict()
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
            "semantics_revision": LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
            "n_prepared_frames": n_prepared,
            "n_behavior_covered": n_behavior_covered,
            "n_pose_covered": n_pose_covered,
            "n_behavior_missing": n_behavior_missing,
            "n_pose_missing": n_pose_missing,
            "n_unique_behavior_dense_selected": n_unique_behavior_dense,
            "n_repeated_behavior_dense_selections": n_repeated_behavior_dense,
            "coordinate_space": "prepared_video_pixels",
            "pose_authority": "accepted_s3_tracked_pose",
            "s3_run_id": s3_run_id,
            "behavior_join_key": BEHAVIOR_MAPPING_PREPARED_NEAREST_DENSE,
            "pose_join_key": "prepared_frame_idx",
            "behavior_mapping_policy": BEHAVIOR_MAPPING_PREPARED_NEAREST_DENSE,
            "behavior_tie_break": BEHAVIOR_TIE_BREAK_EARLIER,
            "original_frame_column": "s1_prepared_sync.raw_decode_frame_idx",
            "original_frame_indexing": "0_based",
            "original_frame_min": int(frame_vals.min()) if len(frame_vals) else None,
            "original_frame_max": int(frame_vals.max()) if len(frame_vals) else None,
            "nonzero_trim_offset": bool(trim_offset not in (None, 0)),
            "trim_offset_frames": trim_offset,
            "many_to_one_original_frames": n_dup_orig > 0,
            "behavior_sync": behavior_audit,
            "independent_timing_authorities": True,
            "behavior_n_dense": int(behavior_sync.n_dense),
            "dense_indices_assumed_equal": False,
            "focal0_present_rows": int(table["behaviour_label_focal_0"].notna().sum()),
            "focal1_present_rows": int(table["behaviour_label_focal_1"].notna().sum()),
        },
    )


def prepared_to_nearest_dense(
    *,
    prepared_time_sec: np.ndarray,
    dense_time_sec: np.ndarray,
    dense_sample_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Map each prepared timestamp to the nearest behavioral dense sample.

    Exact ties select the earlier dense sample. Prepared times outside the
    inclusive dense temporal coverage ``[dense_t0, dense_tN]`` (within
    ``TIME_ATOL_SEC``) return ``-1`` (no extrapolation).
    """

    prep_t = np.asarray(prepared_time_sec, dtype=np.float64).reshape(-1)
    dense_t = np.asarray(dense_time_sec, dtype=np.float64).reshape(-1)
    if dense_sample_idx is None:
        samples = np.arange(dense_t.shape[0], dtype=np.int64)
    else:
        samples = np.asarray(dense_sample_idx, dtype=np.int64).reshape(-1)
    if prep_t.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if dense_t.shape[0] == 0:
        return np.full(prep_t.shape[0], -1, dtype=np.int64)
    if dense_t.shape[0] != samples.shape[0]:
        raise LegacyFramewiseSourceError(
            "dense_time_sec and dense_sample_idx lengths differ."
        )
    if not np.all(np.isfinite(prep_t)):
        raise LegacyFramewiseEligibilityError(
            "prepared_time_sec contains non-finite values."
        )
    if not np.all(np.isfinite(dense_t)):
        raise LegacyFramewiseEligibilityError(
            "behavioral dense time_sec contains non-finite values."
        )
    if dense_t.shape[0] > 1 and not np.all(np.diff(dense_t) > 0.0):
        raise LegacyFramewiseSourceError(
            "behavioral dense time_sec must be strictly increasing for "
            "prepared→nearest-dense mapping."
        )

    out = np.full(prep_t.shape[0], -1, dtype=np.int64)
    t0 = float(dense_t[0])
    t1 = float(dense_t[-1])
    in_coverage = (prep_t >= t0 - TIME_ATOL_SEC) & (prep_t <= t1 + TIME_ATOL_SEC)
    if not bool(in_coverage.any()):
        return out

    covered_i = np.flatnonzero(in_coverage)
    targets = prep_t[covered_i]
    left = np.searchsorted(dense_t, targets, side="right") - 1
    chosen_pos = np.empty(targets.shape[0], dtype=np.int64)
    for i, target in enumerate(targets):
        L = int(left[i])
        R = L + 1
        candidates: list[int] = []
        if 0 <= L < dense_t.shape[0]:
            candidates.append(L)
        if 0 <= R < dense_t.shape[0]:
            candidates.append(R)
        if not candidates:
            chosen_pos[i] = -1
            continue
        # Min abs residual; earlier position wins exact ties.
        best = min(
            candidates,
            key=lambda j: (abs(float(dense_t[j]) - float(target)), j),
        )
        chosen_pos[i] = int(best)

    valid = chosen_pos >= 0
    out[covered_i[valid]] = samples[chosen_pos[valid]]
    return out


def prepared_to_last_in_plateau_dense(
    sync: SyncMap,
    *,
    n_prepared: int,
) -> np.ndarray:
    """Historical LAST_IN_PLATEAU helper (dense→prepared inversion).

    Retained for regression comparison only. ``legacy_framewise_v2`` no longer
    uses this mapping for behavioral or pose joins.
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
        if 0 <= p < n_prepared:
            out[p] = int(sample[j - 1])
        elif p >= n_prepared:
            raise LegacyFramewiseEligibilityError(
                f"SyncMap prepared_frame_idx {p} is outside the authoritative "
                f"S1 prepared-video universe [0, {n_prepared - 1}]."
            )
        i = j
    return out


def _pivot_tracked_pose_by_prepared(
    tracked_pose: pd.DataFrame,
    *,
    n_prepared: int,
) -> pd.DataFrame:
    """Pivot accepted S3 tracked_pose onto the prepared-frame universe.

    Missing frames / nodes remain NaN. Never synthesizes coordinates.
    """

    if tracked_pose is None or tracked_pose.empty:
        raise LegacyFramewiseSourceError("accepted S3 tracked_pose table is empty.")
    required = {"track", "node", "x", "y"}
    missing = sorted(required - set(tracked_pose.columns))
    if missing:
        raise LegacyFramewiseSourceError(
            f"tracked_pose missing required columns: {missing}"
        )
    frame_col = _s3_frame_column(tracked_pose)
    work = tracked_pose.loc[:, [frame_col, "track", "node", "x", "y"]].copy()
    work["prepared_frame_idx"] = pd.to_numeric(work[frame_col], errors="coerce")
    if work["prepared_frame_idx"].isna().any():
        raise LegacyFramewiseSourceError(
            "tracked_pose has non-numeric prepared-frame identities."
        )
    work["prepared_frame_idx"] = work["prepared_frame_idx"].astype(np.int64)
    work["track"] = work["track"].map(_parse_track_index)
    if work["track"].isna().any():
        raise LegacyFramewiseSourceError(
            "tracked_pose has unparsable track identities."
        )
    work["track"] = work["track"].astype(np.int64)
    work["node"] = work["node"].astype(str)

    if bool((work["prepared_frame_idx"] < 0).any()):
        raise LegacyFramewiseSourceError(
            "tracked_pose prepared-frame identities contain negative values."
        )
    if bool((work["prepared_frame_idx"] >= n_prepared).any()):
        bad = sorted(
            {
                int(v)
                for v in work.loc[
                    work["prepared_frame_idx"] >= n_prepared, "prepared_frame_idx"
                ]
                .unique()
                .tolist()
            }
        )
        raise LegacyFramewiseEligibilityError(
            "Accepted S3 tracked_pose prepared-frame identities are incompatible "
            f"with the S1 prepared-video universe [0, {n_prepared - 1}] "
            f"(examples={bad[:5]})."
        )

    wanted = {(node, track) for node, track in LEGACY_POSE_NODE_TRACKS}
    present_nodes = set(str(v) for v in work["node"].unique().tolist())
    present_tracks = set(int(v) for v in work["track"].unique().tolist())
    required_nodes = {node for node, _track in LEGACY_POSE_NODE_TRACKS}
    missing_nodes = sorted(required_nodes - present_nodes)
    if missing_nodes:
        raise LegacyFramewiseSourceError(
            f"tracked_pose lacks required pose nodes for legacy export: {missing_nodes}"
        )
    if not {0, 1}.issubset(present_tracks):
        raise LegacyFramewiseSourceError(
            f"tracked_pose must include tracks 0 and 1; found {sorted(present_tracks)}"
        )
    present_pairs = {
        (str(node), int(track))
        for node, track in zip(
            work["node"].to_numpy(), work["track"].to_numpy(), strict=True
        )
    }
    missing_pairs = [
        f"{node}_{track}"
        for node, track in LEGACY_POSE_NODE_TRACKS
        if (node, track) not in present_pairs
    ]
    if missing_pairs:
        raise LegacyFramewiseSourceError(
            "tracked_pose lacks required node/track pairs for legacy export: "
            f"{missing_pairs}"
        )

    mask = [
        (str(node), int(track)) in wanted
        for node, track in zip(
            work["node"].to_numpy(), work["track"].to_numpy(), strict=True
        )
    ]
    work = work.loc[mask].copy()

    out = pd.DataFrame({"prepared_frame_idx": np.arange(n_prepared, dtype=np.int64)})
    for node, track in LEGACY_POSE_NODE_TRACKS:
        subset = work.loc[(work["node"] == node) & (work["track"] == track)]
        subset = (
            subset.sort_values("prepared_frame_idx", kind="mergesort")
            .drop_duplicates("prepared_frame_idx", keep="first")
            .loc[:, ["prepared_frame_idx", "x", "y"]]
            .rename(
                columns={
                    "x": f"{node}_{track}_x",
                    "y": f"{node}_{track}_y",
                }
            )
        )
        out = out.merge(
            subset, on="prepared_frame_idx", how="left", validate="one_to_one"
        )

    pose_cols = [c for c in LEGACY_FRAMEWISE_COLUMNS if c.endswith("_x") or c.endswith("_y")]
    observed_frames = set(
        int(v) for v in work["prepared_frame_idx"].unique().tolist()
    )
    out["_s3_pose_available"] = out["prepared_frame_idx"].map(
        lambda idx: int(idx) in observed_frames
    )
    # Ensure all pose columns exist even if somehow dropped.
    for col in pose_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _s3_frame_column(table: pd.DataFrame) -> str:
    for name in _S3_FRAME_CANDIDATES:
        if name in table.columns:
            return name
    raise LegacyFramewiseSourceError(
        "tracked_pose.parquet is missing prepared_frame_idx/frame_idx/frame."
    )


def _parse_track_index(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    lowered = text.lower()
    if lowered.startswith("track_"):
        suffix = text.split("_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _normalize_prepared_universe(
    *,
    prepared_frame_idx: np.ndarray,
    raw_decode_frame_idx: np.ndarray,
    prepared_time_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        prepared_frame_idx is None
        or raw_decode_frame_idx is None
        or prepared_time_sec is None
    ):
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_sync prepared_frame_idx/raw_decode_frame_idx/"
            "prepared_time_sec are required for legacy_framewise_v2."
        )
    prep = np.asarray(prepared_frame_idx, dtype=np.int64).reshape(-1)
    raw = np.asarray(raw_decode_frame_idx, dtype=np.int64).reshape(-1)
    times = np.asarray(prepared_time_sec, dtype=np.float64).reshape(-1)
    if prep.shape[0] == 0:
        raise LegacyFramewiseEligibilityError("S1 prepared_sync has zero prepared frames.")
    if prep.shape[0] != raw.shape[0] or prep.shape[0] != times.shape[0]:
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_sync prepared_frame_idx, raw_decode_frame_idx, and "
            f"prepared_time_sec lengths differ ({prep.shape[0]}, {raw.shape[0]}, "
            f"{times.shape[0]})."
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
    if not np.all(np.isfinite(times)):
        raise LegacyFramewiseEligibilityError(
            "S1 prepared_sync.prepared_time_sec contains non-finite values."
        )
    if bool((raw < 0).any()):
        raise LegacyFramewiseSourceError(
            "raw_decode_frame_idx contains negative values; original frames are 0-based."
        )
    return prep, raw, times
