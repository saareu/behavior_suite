"""Fixtures for S9 media/CSV export tests (synthetic prepared video + mini package)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import yaml

from project.file_digest import sha256_file
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    ETHOGRAM_SUMMARY_FILENAME,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RESULTS_PACKAGE_DIRNAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    SYNC_INDEX_FILENAME,
    VERIFICATION_STRENGTH_SIZE_MTIME,
)
from results_packaging.media_csv_export.video import resolve_ffmpeg_executable


@contextmanager
def patch_export_package_currently_eligible() -> Iterator[None]:
    """Treat mini export packages as currently eligible for create/retry tests.

    Mini fixtures intentionally omit a full S8 eligibility chain; production
    MVP still requires current eligibility via ``validate_s9_run_for_downstream``.
    """

    def _ok(run_dir: Path | str) -> dict[str, Any]:
        path = Path(run_dir).resolve()
        return {
            "run_dir": path,
            "run_id": path.name,
            "status": STATUS_COMPLETED,
            "currently_eligible": True,
        }

    with patch(
        "results_packaging.media_csv_export.source.validate_s9_run_for_downstream",
        side_effect=_ok,
    ):
        yield


def write_synthetic_mp4(
    path: Path,
    *,
    n_frames: int = 60,
    fps: float = 30.0,
    width: int = 64,
    height: int = 48,
) -> Path:
    """Write a short synthetic MP4 with a known frame count via OpenCV→ffmpeg.

    Uses OpenCV to write an intermediate AVI/mp4v then remuxes with suite ffmpeg
    to libx264 yuv420p when possible; falls back to the OpenCV file if remux fails.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(raw), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {raw}")
    try:
        for i in range(int(n_frames)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (i % 255, (i * 3) % 255, (i * 7) % 255)
            cv2.putText(
                frame,
                str(i),
                (5, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()

    try:
        ffmpeg = resolve_ffmpeg_executable()
        cmd = [
            str(ffmpeg),
            "-y",
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{fps:.6f}",
            str(path),
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not path.is_file():
            # Fall back to copying raw if remux failed.
            path.write_bytes(raw.read_bytes())
    finally:
        if raw.exists():
            raw.unlink(missing_ok=True)
    return path


def count_video_frames(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return fps, n


def build_mini_export_package(
    root: Path,
    *,
    n_frames: int = 60,
    fps: float = 30.0,
    package_run_id: str = "results_package__export_synth",
) -> dict[str, Any]:
    """Create a session with a completed mini S9 package and real prepared video."""

    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    video_path = preprocess / "prepared_video.mp4"
    write_synthetic_mp4(video_path, n_frames=n_frames, fps=fps)
    probed_fps, probed_n = count_video_frames(video_path)
    assert probed_n > 0 and probed_fps > 0

    # Dense samples: 2:1 onto prepared (many-to-one).
    n_dense = int(probed_n) * 2
    sync = pd.DataFrame(
        {
            "schema_version": "results_package_sync_v1",
            "sample_idx": np.arange(n_dense, dtype=int),
            "dense_frame_idx": np.arange(n_dense, dtype=int),
            "time_sec": np.arange(n_dense, dtype=float) / 120.0,
            "prepared_frame_idx": np.repeat(np.arange(probed_n, dtype=int), 2),
            "prepared_time_sec": np.repeat(np.arange(probed_n, dtype=float) / probed_fps, 2),
            "is_exact_source_sample": True,
            "timing_residual_sec": 0.0,
        }
    )

    bouts = pd.DataFrame(
        [
            {
                "bout_id": f"{package_run_id}__bout_edge_start",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "engaged",
                "channel": "exclusive",
                "identity_grain": "directional",
                "focal_id": 0,
                "partner_id": 1,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 0,
                "end_sample_idx": 3,
                "start_time_sec": 0.0,
                "end_time_sec": 3 / 120,
                "bout_duration_frames": 4,
                "bout_duration_sec": 4 / 120,
                "mean_prob": 0.5,
                "max_prob": 0.5,
                "median_prob": 0.5,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": None,
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": "s8",
                "exp_name": "synth",
                "source_s8_bout_id": "s8_bout_edge_start",
                "source_s8_run_id": "s8",
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavior_bouts_v1",
                "prepared_frame_idx_start": 0,
                "prepared_frame_idx_end": 1,
                "prepared_time_sec_start": 0.0,
                "prepared_time_sec_end": 1 / probed_fps,
            },
            {
                "bout_id": f"{package_run_id}__bout_contact",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "contact",
                "channel": "contact",
                "identity_grain": "symmetric_pair",
                "focal_id": None,
                "partner_id": None,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 20,
                "end_sample_idx": 27,
                "start_time_sec": 20 / 120,
                "end_time_sec": 27 / 120,
                "bout_duration_frames": 8,
                "bout_duration_sec": 8 / 120,
                "mean_prob": 0.6,
                "max_prob": 0.6,
                "median_prob": 0.6,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": None,
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": "s8",
                "exp_name": "synth",
                "source_s8_bout_id": "s8_bout_contact",
                "source_s8_run_id": "s8",
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavior_bouts_v1",
                "prepared_frame_idx_start": 10,
                "prepared_frame_idx_end": 13,
                "prepared_time_sec_start": 10 / probed_fps,
                "prepared_time_sec_end": 13 / probed_fps,
            },
            {
                "bout_id": f"{package_run_id}__bout_edge_end",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "approach",
                "channel": "exclusive",
                "identity_grain": "directional",
                "focal_id": 1,
                "partner_id": 0,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": n_dense - 4,
                "end_sample_idx": n_dense - 1,
                "start_time_sec": (n_dense - 4) / 120,
                "end_time_sec": (n_dense - 1) / 120,
                "bout_duration_frames": 4,
                "bout_duration_sec": 4 / 120,
                "mean_prob": 0.7,
                "max_prob": 0.7,
                "median_prob": 0.7,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": None,
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": "s8",
                "exp_name": "synth",
                "source_s8_bout_id": "s8_bout_edge_end",
                "source_s8_run_id": "s8",
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavior_bouts_v1",
                "prepared_frame_idx_start": probed_n - 2,
                "prepared_frame_idx_end": probed_n - 1,
                "prepared_time_sec_start": (probed_n - 2) / probed_fps,
                "prepared_time_sec_end": (probed_n - 1) / probed_fps,
            },
        ]
    )

    package_dir = session / RESULTS_PACKAGE_DIRNAME / package_run_id
    package_dir.mkdir(parents=True)
    bouts.to_parquet(package_dir / BEHAVIOR_BOUTS_FILENAME, index=False)
    sync.to_parquet(package_dir / SYNC_INDEX_FILENAME, index=False)
    pd.DataFrame(
        [
            {
                "schema_version": "results_package_ethogram_summary_v1",
                "summary_kind": "bout_counts",
                "channel": "exclusive",
                "behavior_name": "engaged",
                "identity_key": "all",
                "identity_grain": "session",
                "metric_name": "n_bouts",
                "metric_value": float(len(bouts)),
                "denominator_name": None,
                "denominator_value": None,
                "unit": "count",
                "notes": None,
            }
        ]
    ).to_parquet(package_dir / ETHOGRAM_SUMMARY_FILENAME, index=False)

    bouts_sha = sha256_file(package_dir / BEHAVIOR_BOUTS_FILENAME)
    sync_sha = sha256_file(package_dir / SYNC_INDEX_FILENAME)
    video_fp = list(artifact_fingerprint(video_path))

    manifest = {
        "schema_version": "results_package_manifest_v1",
        "package_run_id": package_run_id,
        "status": STATUS_COMPLETED,
        "package_artifacts": [
            {
                "role": "behavior_bouts",
                "embedded": True,
                "relative_path": BEHAVIOR_BOUTS_FILENAME,
                "absolute_path": str(package_dir / BEHAVIOR_BOUTS_FILENAME),
                "sha256": bouts_sha,
                "verification_strength": "content_hash",
            },
            {
                "role": "sync_index",
                "embedded": True,
                "relative_path": SYNC_INDEX_FILENAME,
                "absolute_path": str(package_dir / SYNC_INDEX_FILENAME),
                "sha256": sync_sha,
                "verification_strength": "content_hash",
            },
        ],
    }
    (package_dir / PACKAGE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (package_dir / QC_FILENAME).write_text(
        json.dumps({"schema_version": "results_package_qc_v1", "structural_ok": True})
        + "\n",
        encoding="utf-8",
    )
    run_meta = {
        "schema_version": "results_package_run_meta_v1",
        "run_id": package_run_id,
        "status": STATUS_COMPLETED,
        "session_root": str(session),
        "source_s8": {
            "run_id": "s8",
            "run_dir": str(session / "behavioral_arbitration" / "s8"),
            "behavior_states": str(session / "behavioral_arbitration" / "s8" / "behavior_states.parquet"),
            "behavior_bouts": str(session / "behavioral_arbitration" / "s8" / "behavior_bouts.parquet"),
            "behavior_states_sha256": "a" * 64,
            "behavior_bouts_sha256": "b" * 64,
        },
        "source_s7": {"run_id": "s7", "run_dir": str(session / "behavioral_inference" / "s7")},
        "source_s6": {"run_id": "s6", "run_dir": str(session / "combined_features" / "s6")},
        "source_sync": {
            "path": str(preprocess / "prepared_sync.npz"),
            "sha256": "c" * 64,
        },
        "source_video": {
            "path": str(video_path),
            "sha256": None,
            "fingerprint": video_fp,
            "fingerprint_type": "size_mtime",
            "verification_strength": VERIFICATION_STRENGTH_SIZE_MTIME,
            "required": True,
        },
    }
    (package_dir / RUN_META_FILENAME).write_text(
        json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
    )
    (package_dir / SETTINGS_USED_FILENAME).write_text(
        yaml.safe_dump({"schema_version": "results_package_settings_v1"}),
        encoding="utf-8",
    )

    # Dummy upstream paths so package tree exists without full eligibility.
    (session / "behavioral_arbitration" / "s8").mkdir(parents=True, exist_ok=True)

    return {
        "session_root": session,
        "package_dir": package_dir,
        "package_run_id": package_run_id,
        "video_path": video_path,
        "video_fps": probed_fps,
        "video_n_frames": probed_n,
        "bouts": bouts,
        "bouts_sha256": bouts_sha,
        "sync_sha256": sync_sha,
    }
