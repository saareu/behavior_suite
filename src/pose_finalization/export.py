"""Authoritative long-form ``final_pose.parquet`` export for Subsystem 04.

Parquet is the only final pose format for the MVP. Unresolved coordinates remain
NaN rows. Pre-smoothing coordinates, provenance, SLEAP confidence, protected
manual blanks, smoothing flags, and source-frame mapping are retained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pose_finalization.contracts import (
    FINAL_POSE_COLUMNS,
    FINAL_POSE_SCHEMA_VERSION,
    PROVENANCE_NAMES,
    PoseFinalizationError,
)
from pose_finalization.dense_pose import DensePoseResult
from pose_finalization.smoothing import SmoothResult


def build_final_pose_table(
    dense: DensePoseResult,
    smooth: SmoothResult,
) -> pd.DataFrame:
    """Build the long-form final pose table (one row per sample/track/node)."""

    if dense.x.shape != smooth.x.shape:
        raise PoseFinalizationError("Dense and smoothed array shapes must match.")
    if dense.protected.shape != smooth.was_smoothed.shape:
        raise PoseFinalizationError("Protected and smoothed mask shapes must match.")

    n_samples, n_tracks, n_nodes = dense.x.shape
    expected_rows = n_samples * n_tracks * n_nodes
    timing = dense.timing_map

    if timing.target_sample_count != n_samples:
        raise PoseFinalizationError(
            "TimingMap sample count does not match dense pose sample axis."
        )

    sample_idx = np.repeat(np.arange(n_samples, dtype=np.int64), n_tracks * n_nodes)
    time_sec = np.repeat(
        np.asarray(timing.target_time_sec, dtype=np.float64),
        n_tracks * n_nodes,
    )
    track = np.tile(
        np.repeat(np.asarray(dense.track_ids, dtype=np.int64), n_nodes),
        n_samples,
    )
    node = np.tile(
        np.asarray(dense.node_names, dtype=object),
        n_samples * n_tracks,
    )

    left = np.repeat(
        np.asarray(dense.left_source_frame_idx, dtype=np.int64),
        n_tracks * n_nodes,
    )
    right = np.repeat(
        np.asarray(dense.right_source_frame_idx, dtype=np.int64),
        n_tracks * n_nodes,
    )
    weight = np.repeat(
        np.asarray(dense.interpolation_weight, dtype=np.float64),
        n_tracks * n_nodes,
    )
    exact = np.repeat(
        np.asarray(dense.is_exact_source_sample, dtype=bool),
        n_tracks * n_nodes,
    )

    provenance = dense.provenance.reshape(-1)
    provenance_name = np.asarray(
        [PROVENANCE_NAMES.get(int(code), "unknown") for code in provenance],
        dtype=object,
    )

    frame = pd.DataFrame(
        {
            "schema_version": FINAL_POSE_SCHEMA_VERSION,
            "sample_idx": sample_idx,
            "time_sec": time_sec,
            "track": track,
            "node": node,
            "x": smooth.x.reshape(-1),
            "y": smooth.y.reshape(-1),
            "x_pre_smooth": smooth.x_pre_smooth.reshape(-1),
            "y_pre_smooth": smooth.y_pre_smooth.reshape(-1),
            "point_score": dense.point_score.reshape(-1),
            "instance_score": dense.instance_score.reshape(-1),
            "provenance": provenance.astype(np.uint8),
            "provenance_name": provenance_name,
            "protected_manual_blank": dense.protected.reshape(-1),
            "was_smoothed": smooth.was_smoothed.reshape(-1),
            "left_source_frame_idx": left,
            "right_source_frame_idx": right,
            "interpolation_weight": weight,
            "is_exact_source_sample": exact,
        }
    )

    if len(frame) != expected_rows:
        raise PoseFinalizationError(
            f"final_pose row count {len(frame)} != expected {expected_rows}."
        )
    missing = [name for name in FINAL_POSE_COLUMNS if name not in frame.columns]
    if missing:
        raise PoseFinalizationError(
            "final_pose table is missing columns: " + ", ".join(missing)
        )
    return frame.loc[:, list(FINAL_POSE_COLUMNS)]


def write_final_pose_parquet(
    path: Path | str,
    table: pd.DataFrame,
) -> Path:
    """Write ``final_pose.parquet`` (creates parent directories as needed)."""

    out = Path(path).expanduser().resolve()
    if set(table.columns) != set(FINAL_POSE_COLUMNS) or list(table.columns) != list(
        FINAL_POSE_COLUMNS
    ):
        raise PoseFinalizationError(
            "final_pose table columns do not match the authoritative schema."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    return out


def load_final_pose_table(path: Path | str) -> pd.DataFrame:
    """Load a previously written final_pose.parquet (read-only helper)."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise PoseFinalizationError(f"final_pose.parquet is missing: {file_path}")
    try:
        frame = pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001 — surface as S4 error
        raise PoseFinalizationError(
            f"Could not read final_pose.parquet: {file_path}: {exc}"
        ) from exc
    missing = [name for name in FINAL_POSE_COLUMNS if name not in frame.columns]
    if missing:
        raise PoseFinalizationError(
            "final_pose.parquet is missing columns: " + ", ".join(missing)
        )
    return frame
