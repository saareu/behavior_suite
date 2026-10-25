"""Shared synthetic S4/S5 fixtures for Subsystem 06 Sprint 1 tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blob_features.contracts import (
    BLOB_FEATURES_COLUMNS,
    BLOB_FEATURES_DIRNAME,
    BLOB_FEATURES_FILENAME,
    BLOB_FEATURES_SCHEMA_VERSION,
)
from blob_features.contracts import (
    RUN_META_FILENAME as S5_RUN_META_FILENAME,
)
from pose_finalization.contracts import (
    FINAL_POSE_COLUMNS,
    FINAL_POSE_FILENAME,
    FINAL_POSE_SCHEMA_VERSION,
    POSE_FINALIZATION_DIRNAME,
    PROVENANCE_IDENTITY_CORRECTED,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_NAMES,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PROVENANCE_TEMPORAL_RESAMPLE,
)
from pose_finalization.contracts import (
    RUN_META_FILENAME as S4_RUN_META_FILENAME,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz

DEFAULT_NODES = ("nose", "neck", "spine_base", "tail_base", "headstage")
DEFAULT_TRACKS = (0, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_prepared_sync(path: Path, *, frame_count: int = 4, fps: float = 120.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(path, sync)
    return path


def build_final_pose_table(
    *,
    n_samples: int = 4,
    tracks: tuple[int, ...] = DEFAULT_TRACKS,
    nodes: tuple[str, ...] = DEFAULT_NODES,
    target_fps: float = 120.0,
    provenance_by_key: dict[tuple[int, int, str], int] | None = None,
    blank_keys: set[tuple[int, int, str]] | None = None,
    headstage_owner: int | None = 0,
    point_scores: dict[tuple[int, int, str], float] | None = None,
) -> pd.DataFrame:
    """Build a minimal long-form ``final_pose_v1`` table."""

    blank_keys = blank_keys or set()
    provenance_by_key = provenance_by_key or {}
    point_scores = point_scores or {}
    rows: list[dict[str, object]] = []
    for sample_idx in range(n_samples):
        time_sec = sample_idx / target_fps
        for track in tracks:
            for node in nodes:
                key = (sample_idx, track, node)
                is_blank = key in blank_keys
                if node == "headstage" and headstage_owner is not None and track != headstage_owner:
                    x = y = np.nan
                    prov = PROVENANCE_MISSING
                    score = np.nan
                    protected = False
                elif is_blank:
                    x = y = np.nan
                    prov = PROVENANCE_PROTECTED_MANUAL_BLANK
                    score = np.nan
                    protected = True
                else:
                    x = 10.0 * track + sample_idx + (0.1 if node == "nose" else 0.2)
                    y = 20.0 * track + 2.0 * sample_idx
                    prov = int(provenance_by_key.get(key, PROVENANCE_ORIGINAL))
                    score = float(point_scores.get(key, 0.91))
                    if (
                        prov
                        in (
                            PROVENANCE_LINEAR_GAP_FILL,
                            PROVENANCE_KALMAN_GAP_FILL,
                            PROVENANCE_TEMPORAL_RESAMPLE,
                        )
                        and key not in point_scores
                    ):
                        # Reconstructed samples do not fabricate confidence.
                        score = np.nan
                    protected = False
                rows.append(
                    {
                        "schema_version": FINAL_POSE_SCHEMA_VERSION,
                        "sample_idx": sample_idx,
                        "time_sec": time_sec,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                        "x_pre_smooth": x,
                        "y_pre_smooth": y,
                        "point_score": score,
                        "instance_score": 1.0 if np.isfinite(score) else np.nan,
                        "provenance": np.uint8(prov),
                        "provenance_name": PROVENANCE_NAMES.get(prov, "unknown"),
                        "protected_manual_blank": protected,
                        "was_smoothed": False,
                        "left_source_frame_idx": sample_idx,
                        "right_source_frame_idx": sample_idx,
                        "interpolation_weight": 0.0,
                        "is_exact_source_sample": True,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.loc[:, list(FINAL_POSE_COLUMNS)]


def build_blob_features_table(
    *,
    n_samples: int = 4,
    target_fps: float = 120.0,
    time_offset: float = 0.0,
) -> pd.DataFrame:
    """Build a minimal one-row-per-sample ``blob_features_v1`` table."""

    sample_idx = np.arange(n_samples, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "schema_version": BLOB_FEATURES_SCHEMA_VERSION,
            "sample_idx": sample_idx,
            "dense_frame_idx": sample_idx.copy(),
            "frame_idx": sample_idx.copy(),
            "time_sec": sample_idx.astype(np.float64) / target_fps + time_offset,
            "orig_frame_idx": sample_idx.copy(),
            "cc_count": np.full(n_samples, 2, dtype=np.int64),
            "cc_count_raw": np.full(n_samples, 2, dtype=np.int64),
            "min_inter_contour_dist": np.full(n_samples, 5.0, dtype=np.float64),
        }
    )
    for col in BLOB_FEATURES_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame.loc[:, list(BLOB_FEATURES_COLUMNS)]


def write_joint_session(
    root: Path,
    *,
    n_samples: int = 4,
    frame_count: int | None = None,
    s4_run_id: str = "pose_finalization__20261021T120000",
    s5_run_id: str = "blob_features__20261021T120100",
    sync_path: Path | None = None,
    s5_sync_path: Path | None = None,
    s5_sync_sha: str | None = None,
    final_pose: pd.DataFrame | None = None,
    blob_features: pd.DataFrame | None = None,
    s4_status: str = "completed",
    s5_status: str = "completed",
    s4_session_root: Path | None = None,
    s5_session_root: Path | None = None,
) -> dict[str, Path]:
    """Write a session tree with S4/S5 run metadata and parquet artifacts."""

    session = root / "session"
    session.mkdir(parents=True, exist_ok=True)
    frames = frame_count if frame_count is not None else n_samples
    sync = sync_path or write_prepared_sync(
        session / "preprocess" / "prepared_sync.npz",
        frame_count=frames,
    )
    sync_sha = sha256_file(sync)

    pose = final_pose if final_pose is not None else build_final_pose_table(n_samples=n_samples)
    blob = (
        blob_features
        if blob_features is not None
        else build_blob_features_table(n_samples=n_samples)
    )

    s4_dir = session / POSE_FINALIZATION_DIRNAME / s4_run_id
    s5_dir = session / BLOB_FEATURES_DIRNAME / s5_run_id
    s4_dir.mkdir(parents=True, exist_ok=True)
    s5_dir.mkdir(parents=True, exist_ok=True)

    pose_path = s4_dir / FINAL_POSE_FILENAME
    blob_path = s5_dir / BLOB_FEATURES_FILENAME
    pose.to_parquet(pose_path, index=False)
    blob.to_parquet(blob_path, index=False)

    s4_session = (s4_session_root or session).resolve()
    s5_session = (s5_session_root or session).resolve()
    recorded_s5_sync = (s5_sync_path or sync).resolve()

    s4_meta = {
        "schema_version": "pose_finalization_run_meta_v1",
        "run_id": s4_run_id,
        "status": s4_status,
        "created_at": "2026-10-21T12:00:00",
        "completed_at": "2026-10-21T12:01:00",
        "source_s3": {
            "run_dir": str(session / "tracking_correction" / "s3_run"),
            "run_id": "s3_run",
            "tracked_pose_sha256": "abc",
            "prepared_sync": str(sync.resolve()),
            "session_root": str(s4_session),
            "profile_id": "current_lab_two_mouse_headstage_v1",
        },
        "dense_summary": {
            "n_samples": int(n_samples),
            "n_tracks": 2,
            "n_nodes": 5,
            "track_ids": [0, 1],
            "node_names": list(DEFAULT_NODES),
            "target_fps": 120.0,
        },
    }
    s5_meta = {
        "schema_version": "blob_features_run_meta_v1",
        "run_id": s5_run_id,
        "status": s5_status,
        "created_at": "2026-10-21T12:01:00",
        "completed_at": "2026-10-21T12:02:00",
        "source_s1": {
            "session_root": str(s5_session),
            "prepared_sync": str(recorded_s5_sync),
            "prepared_sync_sha256": s5_sync_sha if s5_sync_sha is not None else sync_sha,
            "target_fps": 120.0,
        },
    }
    (s4_dir / S4_RUN_META_FILENAME).write_text(
        json.dumps(s4_meta, indent=2) + "\n", encoding="utf-8"
    )
    (s5_dir / S5_RUN_META_FILENAME).write_text(
        json.dumps(s5_meta, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "session_root": session,
        "prepared_sync": sync,
        "s4_run_dir": s4_dir,
        "s5_run_dir": s5_dir,
        "final_pose": pose_path,
        "blob_features": blob_path,
    }


# Re-export provenance constants for tests.
PROVENANCE = {
    "missing": PROVENANCE_MISSING,
    "original": PROVENANCE_ORIGINAL,
    "identity_corrected": PROVENANCE_IDENTITY_CORRECTED,
    "temporal_resample": PROVENANCE_TEMPORAL_RESAMPLE,
    "linear_gap_fill": PROVENANCE_LINEAR_GAP_FILL,
    "kalman_gap_fill": PROVENANCE_KALMAN_GAP_FILL,
    "protected_manual_blank": PROVENANCE_PROTECTED_MANUAL_BLANK,
}
