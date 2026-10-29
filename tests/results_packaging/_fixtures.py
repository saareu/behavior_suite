"""Synthetic fixtures for Subsystem 09 Sprint 1 tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from behavioral_arbitration.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    BEHAVIORAL_ARBITRATION_DIRNAME,
    BOUTS_SCHEMA_VERSION,
    QC_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATES_SCHEMA_VERSION,
    STATUS_COMPLETED,
)
from combined_features.contracts import COMBINED_FEATURES_DIRNAME
from combined_features.contracts import RUN_META_FILENAME as S6_RUN_META
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz

MOUSE_A = 1
MOUSE_B = 2
FPS = 120.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_prepared_sync(
    path: Path,
    *,
    frame_count: int,
    fps: float = 30.0,
    start_frame: int = 0,
) -> Path:
    """Write a small prepared_sync.npz with contiguous prepared frames."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=start_frame,
        end_frame_exclusive=start_frame + frame_count,
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=start_frame + frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(path, sync)
    return path


def make_states(
    *,
    n_dense: int,
    time_sec: np.ndarray | None = None,
    schema_version: str = STATES_SCHEMA_VERSION,
) -> pd.DataFrame:
    """Directional S8-like states (2 directions × n_dense)."""

    if time_sec is None:
        time_sec = np.arange(n_dense, dtype=np.float64) / FPS
    rows: list[dict[str, Any]] = []
    for i in range(n_dense):
        for focal, partner in ((MOUSE_A, MOUSE_B), (MOUSE_B, MOUSE_A)):
            rows.append(
                {
                    "schema_version": schema_version,
                    "exp_name": "synth",
                    "sample_idx": i,
                    "dense_frame_idx": i,
                    "time_sec": float(time_sec[i]),
                    "focal_id": focal,
                    "partner_id": partner,
                    "behavior_label": "engaged",
                    "pred_contact": 0,
                    "pred_side_by_side": 0,
                }
            )
    return pd.DataFrame(rows)


def make_bouts(
    *,
    intervals: list[tuple[int, int]],
    schema_version: str = BOUTS_SCHEMA_VERSION,
    run_id: str = "behavioral_arbitration__synth",
    s7_run_id: str = "behavioral_inference__synth",
    s6_run_id: str = "combined_features__synth",
    behavior_name: str = "engaged",
    channel: str = "exclusive",
    identity_grain: str = "directional",
    focal_id: int | None = MOUSE_A,
    partner_id: int | None = MOUSE_B,
    id_prefix: str = "synth_bout",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(intervals):
        dur = end - start + 1
        mouse_a = MOUSE_A
        mouse_b = MOUSE_B
        rows.append(
            {
                "bout_id": f"{id_prefix}_{i}",
                "schema_version": schema_version,
                "behavior_name": behavior_name,
                "channel": channel,
                "identity_grain": identity_grain,
                "focal_id": focal_id if focal_id is not None else pd.NA,
                "partner_id": partner_id if partner_id is not None else pd.NA,
                "mouse_a_id": mouse_a,
                "mouse_b_id": mouse_b,
                "pair_id": f"{mouse_a}|{mouse_b}",
                "start_sample_idx": start,
                "end_sample_idx": end,
                "start_time_sec": float(start) / FPS,
                "end_time_sec": float(end) / FPS,
                "bout_duration_frames": dur,
                "bout_duration_sec": float(dur) / FPS,
                "mean_prob": 0.5,
                "max_prob": 0.5,
                "median_prob": 0.5,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": None,
                "source_s7_run_id": s7_run_id,
                "source_s6_run_id": s6_run_id,
                "run_id": run_id,
                "exp_name": "synth",
            }
        )
    return pd.DataFrame(rows)


def make_multi_channel_bouts(
    *,
    run_id: str = "behavioral_arbitration__synth",
    s7_run_id: str = "behavioral_inference__synth",
    s6_run_id: str = "combined_features__synth",
) -> pd.DataFrame:
    """Deterministic mixed exclusive / contact / SBS bout table for Sprint 2."""

    exclusive_a = make_bouts(
        intervals=[(0, 0), (1, 5), (10, 21), (30, 41)],
        behavior_name="approach",
        channel="exclusive",
        identity_grain="directional",
        focal_id=MOUSE_A,
        partner_id=MOUSE_B,
        id_prefix="ex_ab",
        run_id=run_id,
        s7_run_id=s7_run_id,
        s6_run_id=s6_run_id,
    )
    exclusive_b = make_bouts(
        intervals=[(2, 4), (50, 50)],
        behavior_name="engaged",
        channel="exclusive",
        identity_grain="directional",
        focal_id=MOUSE_B,
        partner_id=MOUSE_A,
        id_prefix="ex_ba",
        run_id=run_id,
        s7_run_id=s7_run_id,
        s6_run_id=s6_run_id,
    )
    contact = make_bouts(
        intervals=[(8, 15), (40, 42)],
        behavior_name="contact",
        channel="contact",
        identity_grain="symmetric_pair",
        focal_id=None,
        partner_id=None,
        id_prefix="ct",
        run_id=run_id,
        s7_run_id=s7_run_id,
        s6_run_id=s6_run_id,
    )
    sbs = make_bouts(
        intervals=[(20, 25)],
        behavior_name="side_by_side",
        channel="side_by_side",
        identity_grain="symmetric_pair",
        focal_id=None,
        partner_id=None,
        id_prefix="sbs",
        run_id=run_id,
        s7_run_id=s7_run_id,
        s6_run_id=s6_run_id,
    )
    out = pd.concat([exclusive_a, exclusive_b, contact, sbs], ignore_index=True)
    # Assign S8-like sequential ids after concatenation (order-stable for fixture).
    out["bout_id"] = [f"bout_{i:08d}" for i in range(len(out))]
    return out


def build_packaging_session(
    root: Path,
    *,
    n_prepared: int = 5,
    prepared_fps: float = 30.0,
    n_dense: int | None = None,
    bout_intervals: list[tuple[int, int]] | None = None,
    include_video: bool = True,
    include_sync: bool = True,
    s8_status: str = STATUS_COMPLETED,
    tamper_sync_digest: bool = False,
) -> dict[str, Any]:
    """Build a minimal session tree for S9 intake / sync tests.

    When ``n_dense`` is omitted, uses the rebuilt 120 Hz length implied by the
    prepared sync (via S5 NN map).
    """

    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)

    sync_path = preprocess / "prepared_sync.npz"
    if include_sync:
        write_prepared_sync(sync_path, frame_count=n_prepared, fps=prepared_fps)
        sync_sha = sha256_file(sync_path)
    else:
        sync_sha = "missing"

    video_path = preprocess / "prepared_video.mp4"
    if include_video:
        video_path.write_bytes(b"fake-video-bytes")
        video_sha = sha256_file(video_path)
    else:
        video_sha = None

    # Resolve dense count from NN map when not provided.
    if n_dense is None and include_sync:
        from blob_features.timing import build_nearest_neighbor_timing_map_from_prepared_sync

        nn = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
        n_dense = int(nn.target_sample_count)
        time_sec = np.asarray(nn.target_time_sec, dtype=np.float64)
    else:
        n_dense = int(n_dense or 4)
        time_sec = np.arange(n_dense, dtype=np.float64) / FPS

    s6_run_id = "combined_features__synth"
    s6_dir = session / COMBINED_FEATURES_DIRNAME / s6_run_id
    s6_dir.mkdir(parents=True)
    s4_run_id = "pose_finalization__synth"
    s5_run_id = "blob_features__synth"
    recorded_sync_sha = "deadbeef" * 8 if tamper_sync_digest else sync_sha
    s6_meta = {
        "schema_version": "combined_features_run_meta_v1",
        "run_id": s6_run_id,
        "status": "completed",
        "session_root": str(session),
        "source_s4": {
            "run_id": s4_run_id,
            "run_dir": str(session / "pose_finalization" / s4_run_id),
            "prepared_sync": str(sync_path) if include_sync else str(sync_path),
            "prepared_sync_sha256": recorded_sync_sha if include_sync else None,
        },
        "source_s5": {
            "run_id": s5_run_id,
            "run_dir": str(session / "blob_features" / s5_run_id),
            "prepared_sync": str(sync_path) if include_sync else str(sync_path),
            "prepared_sync_sha256": recorded_sync_sha if include_sync else None,
            "prepared_video_sha256": video_sha,
            "source_s1": {
                "prepared_sync": str(sync_path) if include_sync else None,
                "prepared_sync_sha256": recorded_sync_sha if include_sync else None,
                "prepared_video": str(video_path) if include_video else None,
                "prepared_video_sha256": video_sha,
            },
        },
    }
    (s6_dir / S6_RUN_META).write_text(json.dumps(s6_meta), encoding="utf-8")

    s7_run_id = "behavioral_inference__synth"
    s7_dir = session / "behavioral_inference" / s7_run_id
    s7_dir.mkdir(parents=True)
    (s7_dir / "predictions.parquet").write_bytes(b"pred")
    pred_sha = sha256_file(s7_dir / "predictions.parquet")

    s8_run_id = "behavioral_arbitration__synth"
    s8_dir = session / BEHAVIORAL_ARBITRATION_DIRNAME / s8_run_id
    s8_dir.mkdir(parents=True)

    states = make_states(n_dense=n_dense, time_sec=time_sec)
    if bout_intervals is None:
        bout_intervals = [(0, min(1, n_dense - 1))]
    bouts = make_bouts(
        intervals=bout_intervals,
        run_id=s8_run_id,
        s7_run_id=s7_run_id,
        s6_run_id=s6_run_id,
    )
    states_path = s8_dir / BEHAVIOR_STATES_FILENAME
    bouts_path = s8_dir / BEHAVIOR_BOUTS_FILENAME
    states.to_parquet(states_path, index=False)
    bouts.to_parquet(bouts_path, index=False)
    (s8_dir / QC_FILENAME).write_text(json.dumps({"warning_count": 0}), encoding="utf-8")
    (s8_dir / SETTINGS_USED_FILENAME).write_text(
        yaml.safe_dump({"schema_version": "behavioral_arbitration_settings_v1"}),
        encoding="utf-8",
    )
    s8_meta = {
        "schema_version": "behavioral_arbitration_run_meta_v1",
        "run_id": s8_run_id,
        "status": s8_status,
        "session_root": str(session),
        "fps": FPS,
        "source_s7": {
            "run_id": s7_run_id,
            "run_dir": str(s7_dir),
            "predictions": str(s7_dir / "predictions.parquet"),
            "predictions_sha256": pred_sha,
        },
        "source_s6": {
            "run_id": s6_run_id,
            "run_dir": str(s6_dir),
            "dyadic_features": str(s6_dir / "dyadic_features_full.parquet"),
            "dyadic_features_sha256": "abc",
        },
    }
    (s8_dir / RUN_META_FILENAME).write_text(json.dumps(s8_meta), encoding="utf-8")

    return {
        "session_root": session,
        "s8_run_dir": s8_dir,
        "s8_run_id": s8_run_id,
        "s7_run_dir": s7_dir,
        "s7_run_id": s7_run_id,
        "s6_run_dir": s6_dir,
        "s6_run_id": s6_run_id,
        "prepared_sync": sync_path if include_sync else None,
        "prepared_video": video_path if include_video else None,
        "prepared_sync_sha256": sync_sha if include_sync else None,
        "n_dense": n_dense,
        "n_prepared": n_prepared,
        "states_path": states_path,
        "bouts_path": bouts_path,
        "time_sec": time_sec,
    }
