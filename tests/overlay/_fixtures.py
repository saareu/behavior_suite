"""Synthetic fixtures for overlay C1 unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tests.behavioral_review._fixtures import build_hand_computed_sync

from pose_finalization.contracts import (
    PROVENANCE_IDENTITY_CORRECTED,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_NAMES,
    PROVENANCE_ORIGINAL,
    PROVENANCE_TEMPORAL_RESAMPLE,
)


def solid_frame(color: tuple[int, int, int] = (30, 40, 50), *, h: int = 48, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def write_s2_pose_parquet(path: Path, *, prepared_frames: tuple[int, ...] = (0, 1, 2)) -> Path:
    rows: list[dict[str, Any]] = []
    for prep in prepared_frames:
        for inst, track in ((0, None), (1, None)):
            for node, dx in (("nose", 0.0), ("ear", 3.0)):
                rows.append(
                    {
                        "schema_version": "pose_parquet_v1",
                        "frame_idx": prep,
                        "prepared_frame_idx": prep,
                        "instance_index": inst,
                        "track": track,
                        "node": node,
                        "x": 10.0 + prep + dx + inst * 20,
                        "y": 12.0 + inst * 5,
                        "score": 0.9,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def write_s3_tracked_pose_parquet(
    path: Path, *, prepared_frames: tuple[int, ...] = (0, 1, 2)
) -> Path:
    rows: list[dict[str, Any]] = []
    for prep in prepared_frames:
        for track in (0, 1):
            for node, dx in (("nose", 0.0), ("ear", 4.0)):
                rows.append(
                    {
                        "frame_idx": prep,
                        "track": track,
                        "node": node,
                        "x": 8.0 + prep + dx + track * 18,
                        "y": 9.0 + track * 6,
                        "score": 0.85,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def write_s4_final_pose_parquet(path: Path, *, n_samples: int = 12) -> Path:
    rows: list[dict[str, Any]] = []
    codes = (
        PROVENANCE_ORIGINAL,
        PROVENANCE_IDENTITY_CORRECTED,
        PROVENANCE_TEMPORAL_RESAMPLE,
        PROVENANCE_LINEAR_GAP_FILL,
        PROVENANCE_KALMAN_GAP_FILL,
        PROVENANCE_MISSING,
    )
    for sample in range(n_samples):
        for track in (0, 1):
            for node_i, node in enumerate(("nose", "ear")):
                prov = codes[(sample + track + node_i) % len(codes)]
                missing = prov == PROVENANCE_MISSING
                rows.append(
                    {
                        "schema_version": "final_pose_v1",
                        "sample_idx": sample,
                        "time_sec": sample / 120.0,
                        "track": track,
                        "node": node,
                        "x": np.nan if missing else 5.0 + sample + track * 10 + node_i,
                        "y": np.nan if missing else 6.0 + track * 4,
                        "x_pre_smooth": np.nan if missing else 5.0,
                        "y_pre_smooth": np.nan if missing else 6.0,
                        "point_score": None if missing or prov >= PROVENANCE_TEMPORAL_RESAMPLE else 0.8,
                        "instance_score": 0.7,
                        "provenance": prov,
                        "provenance_name": PROVENANCE_NAMES[prov],
                        "protected_manual_blank": False,
                        "was_smoothed": False,
                        "left_source_frame_idx": sample // 3,
                        "right_source_frame_idx": sample // 3,
                        "interpolation_weight": 0.0,
                        "is_exact_source_sample": True,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def write_behavior_states_parquet(
    path: Path,
    sync: pd.DataFrame,
    *,
    schema_version: str = "behavioral_arbitration_states_v1",
    label_a_to_b: str | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    n = int(len(sync))
    for i in range(n):
        t = float(sync.loc[i, "time_sec"])
        lab_a = label_a_to_b if label_a_to_b is not None else ("approach" if i < n // 2 else "engaged")
        lab_b = "retract" if i < n // 2 else "mutually_disengaged"
        # Leave one sample without a valid A→B label for invalid-state tests.
        if i == n - 1 and label_a_to_b is None:
            lab_a_val: Any = None
        else:
            lab_a_val = lab_a
        for focal, partner, lab in ((0, 1, lab_a_val), (1, 0, lab_b)):
            rows.append(
                {
                    "schema_version": schema_version,
                    "dense_frame_idx": i,
                    "sample_idx": i,
                    "time_sec": t,
                    "focal_id": focal,
                    "partner_id": partner,
                    "behavior_label": lab,
                    "label_reconstruction": "selected",
                    "undefined_reason": "",
                    "hierarchy_residual_kind": "",
                    "pred_contact": 1 if 2 <= i <= n - 3 else 0,
                    "pred_side_by_side": 1 if i >= n - 2 else 0,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def write_reviewed_states_parquet(path: Path, sync: pd.DataFrame) -> Path:
    """Reviewed states with deliberately different labels from automated S8."""

    return write_behavior_states_parquet(
        path,
        sync,
        schema_version="behavioral_review_states_v1",
        label_a_to_b="nose_to_body",
    )


def build_gap_sync_with_missing_prepared() -> pd.DataFrame:
    """Contiguous dense map that omits a prepared frame index (gap)."""

    sync = build_hand_computed_sync(n_prepared=4, plateau_repeats=2)
    # Remap prepared indices to 0,1,3,4 so prepared 2 is uncovered.
    prepared = sync["prepared_frame_idx"].to_numpy().copy()
    prepared = np.where(prepared >= 2, prepared + 1, prepared)
    sync = sync.copy()
    sync["prepared_frame_idx"] = prepared
    sync["prepared_time_sec"] = prepared.astype(np.float64) / 10.0
    sync["timing_residual_sec"] = sync["time_sec"] - sync["prepared_time_sec"]
    return sync
