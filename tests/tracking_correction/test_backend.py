"""Sprint 2: current-lab backend adapter and field adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction import (
    CURRENT_BACKEND_ID,
    CURRENT_PROFILE_ID,
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    run_tracking_correction,
)
from tracking_correction.contracts import (
    FINAL_TRACKED_POSE_FILENAME,
    MACHINE_CORRECTIONS_FILENAME,
    WORKING_TRACKED_POSE_FILENAME,
)
from tracking_correction.legacy_backend import (
    adapt_s2_pose_to_legacy_input,
    load_current_profile_config,
    run_current_lab_backend,
)

REQUIRED_NODES = ("nose", "neck", "spine_base", "tail_base", "headstage")
# Roughly lab-scale node offsets so skeleton-stretch rules are not constantly triggered.
_NODE_XY = {
    "nose": (0.0, 0.0),
    "neck": (30.0, -40.0),
    "spine_base": (55.0, -20.0),
    "tail_base": (90.0, -10.0),
    "headstage": (10.0, -55.0),
}
TIMESTAMP = "20260102T000000"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {path: (_sha256(path), path.stat().st_size) for path in paths if path.exists()}


def _write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (16, 16),
    )
    assert writer.isOpened()
    for frame_idx in range(frame_count):
        writer.write(np.full((16, 16, 3), frame_idx, dtype=np.uint8))
    writer.release()


def _write_s1(session_root: Path, *, frame_count: int = 12) -> dict[str, Path]:
    preprocess = session_root / "preprocess"
    preprocess.mkdir(parents=True)
    video = preprocess / "prepared_video.mp4"
    meta = preprocess / "prepare_meta.json"
    sync_path = preprocess / "prepared_sync.npz"
    _write_video(video, frame_count)
    meta.write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "opencv_reported_frame_count": frame_count,
                    "opencv_readable_frame_count": frame_count,
                    "frame_count_used_for_sleap": frame_count,
                    "fps_header": 20.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / 20.0,
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
    write_prepared_sync_npz(sync_path, sync)
    return {
        "session_root": session_root,
        "preprocess_dir": preprocess,
        "prepared_video": video,
        "prepare_meta": meta,
        "prepared_sync": sync_path,
    }


def _pose_rows(*, frame_count: int = 12, include_instance_score: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_idx in range(frame_count):
        for track_index, track in ((0, "track_0"), (1, "track_1")):
            origin_x = 140.0 + track_index * 180.0
            origin_y = 220.0 + track_index * 10.0
            for node in REQUIRED_NODES:
                # Keep headstage ownership on track 0 so Pipeline 20 transposition
                # does not fire on the synthetic fixture.
                if node == "headstage" and track_index == 1:
                    continue
                dx, dy = _NODE_XY[node]
                row: dict[str, object] = {
                    "frame_idx": frame_idx,
                    "prepared_frame_idx": frame_idx,
                    "video_index": 0,
                    "track": track,
                    "node": node,
                    "x": float(origin_x + dx),
                    "y": float(origin_y + dy),
                    "score": 0.91,
                }
                if include_instance_score:
                    row["instance_score"] = 0.8 + 0.1 * track_index
                rows.append(row)
    return rows


def _write_pose(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _completed_s2(tmp_path: Path, *, frame_count: int = 12) -> tuple[Path, dict[str, Path]]:
    session = tmp_path / "session"
    s1 = _write_s1(session, frame_count=frame_count)
    run_dir = session / "pose_inference" / "model__20260102T000000"
    run_dir.mkdir(parents=True)
    _write_pose(run_dir / "pose.parquet", _pose_rows(frame_count=frame_count))
    payload = {
        "schema_version": "pose_meta_v1",
        "run_id": run_dir.name,
        "status": "success",
        "dry_run": False,
        "input": {key: str(path.resolve()) for key, path in s1.items()},
    }
    (run_dir / "pose_meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return run_dir, s1


def test_adapt_s2_pose_maps_frame_idx_score_and_tracks() -> None:
    pose = pd.DataFrame(
        [
            {
                "frame_idx": 3,
                "prepared_frame_idx": 3,
                "track": "track_1",
                "node": "nose",
                "x": 1.0,
                "y": 2.0,
                "score": 0.7,
                "instance_score": 0.5,
            },
            {
                "frame_idx": 3,
                "prepared_frame_idx": 3,
                "track": "track_0",
                "node": "neck",
                "x": 4.0,
                "y": 5.0,
                "score": 0.8,
                "instance_score": 0.6,
            },
        ]
    )
    adapted = adapt_s2_pose_to_legacy_input(pose)
    assert "frame" in adapted.columns
    assert list(adapted["frame"]) == [3, 3]
    assert list(adapted["frame_idx"]) == [3, 3]
    assert list(adapted["track"]) == [1, 0]
    assert "point_score" in adapted.columns
    assert list(adapted["point_score"]) == [0.7, 0.8]
    assert list(adapted["score"]) == [0.7, 0.8]
    assert list(adapted["prepared_frame_idx"]) == [3, 3]


def test_profile_config_loads_validated_thresholds() -> None:
    config = load_current_profile_config()
    assert config["jump_threshold"] == 30
    assert config["proximity_threshold"] == 30
    assert config["frame_window"] == 10
    assert config["headstage_node"] == "headstage"
    assert config["fps"] == 120
    assert config["enforce_headstage"] is False


def test_backend_execution_on_synthetic_fixture(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path, frame_count=12)
    before = _snapshot(
        [
            run_dir / "pose.parquet",
            run_dir / "pose_meta.json",
            s1["prepared_video"],
            s1["prepare_meta"],
            s1["prepared_sync"],
        ]
    )
    out_dir = tmp_path / "backend_out"
    result = run_current_lab_backend(
        pose_parquet=run_dir / "pose.parquet",
        output_dir=out_dir,
        show_progress=False,
    )
    assert result.backend_id == CURRENT_BACKEND_ID
    assert result.profile_id == CURRENT_PROFILE_ID
    assert result.working_tracked_pose_path.is_file()
    assert result.machine_corrections_path.is_file()
    assert result.working_tracked_pose_path.name == WORKING_TRACKED_POSE_FILENAME
    assert result.machine_corrections_path.name == MACHINE_CORRECTIONS_FILENAME
    assert not (out_dir / FINAL_TRACKED_POSE_FILENAME).exists()

    working = pd.read_parquet(result.working_tracked_pose_path)
    assert "frame_idx" in working.columns
    assert "frame" in working.columns
    assert (working["frame_idx"] == working["frame"]).all()
    assert {"x_raw", "y_raw", "point_score", "point_score_raw"}.issubset(working.columns)
    assert set(working["track"].unique().tolist()) == {0, 1}
    assert "prepared_frame_idx" in working.columns
    assert (working["prepared_frame_idx"] == working["frame_idx"]).all()

    source = pd.read_parquet(run_dir / "pose.parquet")
    source_adapted = adapt_s2_pose_to_legacy_input(source)
    raw_check = working.merge(
        source_adapted[["frame", "track", "node", "x", "y", "score"]].rename(
            columns={"x": "x_src", "y": "y_src", "score": "score_src"}
        ),
        on=["frame", "track", "node"],
        how="inner",
    )
    assert len(raw_check) > 0
    assert np.allclose(raw_check["x_raw"], raw_check["x_src"])
    assert np.allclose(raw_check["y_raw"], raw_check["y_src"])
    assert np.allclose(raw_check["point_score_raw"], raw_check["score_src"])
    assert _snapshot(
        [
            run_dir / "pose.parquet",
            run_dir / "pose_meta.json",
            s1["prepared_video"],
            s1["prepare_meta"],
            s1["prepared_sync"],
        ]
    ) == before


def test_runner_live_path_records_completed_backend_without_final_pose(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path, frame_count=12)
    before = _snapshot(
        [
            run_dir / "pose.parquet",
            s1["prepared_video"],
            s1["prepare_meta"],
            s1["prepared_sync"],
        ]
    )
    result = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=run_dir,
            dry_run=False,
            timestamp=TIMESTAMP,
        )
    )
    assert result.status == "correction_complete"
    assert result.backend_status == "completed"
    assert result.working_tracked_pose_path is not None
    assert result.working_tracked_pose_path.is_file()
    assert not (result.run_dir / FINAL_TRACKED_POSE_FILENAME).exists()
    meta = json.loads(result.run_meta_path.read_text(encoding="utf-8"))
    assert meta["acceptance_state"] == "not_accepted"
    assert meta["backend_status"] == "completed"
    assert meta["correction_summary"]["total_corrections"] >= 0
    assert meta["outputs"]["tracked_pose_parquet"] is None
    settings = yaml.safe_load(result.settings_used_path.read_text(encoding="utf-8"))
    assert settings["corrector_invoked"] is True
    assert settings["profile_config"]["jump_threshold"] == 30
    assert _snapshot(
        [
            run_dir / "pose.parquet",
            s1["prepared_video"],
            s1["prepare_meta"],
            s1["prepared_sync"],
        ]
    ) == before


def test_backend_failure_propagates_and_records_failed_status(tmp_path: Path) -> None:
    run_dir, _s1 = _completed_s2(tmp_path, frame_count=12)
    with (
        patch(
            "tracking_correction.legacy_backend.process_video_minimal",
            side_effect=RuntimeError("synthetic corrector boom"),
        ),
        pytest.raises(TrackingCorrectionError, match="synthetic corrector boom"),
    ):
        run_tracking_correction(
            TrackingCorrectionRequest(
                s2_run_dir=run_dir,
                timestamp="20260102T010101",
            )
        )
    run_root = run_dir.parent.parent / "tracking_correction"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    meta = json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "correction_failed"
    assert meta["backend_status"] == "failed"
    assert meta["acceptance_state"] == "not_accepted"
    assert meta["outputs"]["tracked_pose_parquet"] is None
    assert not (run_dirs[0] / FINAL_TRACKED_POSE_FILENAME).exists()
    log = (run_dirs[0] / "processing_log.txt").read_text(encoding="utf-8")
    assert "backend_status: failed" in log
    assert "synthetic corrector boom" in log


def test_dry_run_still_skips_backend(tmp_path: Path) -> None:
    run_dir, _s1 = _completed_s2(tmp_path, frame_count=4)
    with patch("tracking_correction.runner.run_current_lab_backend") as mocked:
        result = run_tracking_correction(
            TrackingCorrectionRequest(
                s2_run_dir=run_dir,
                dry_run=True,
                timestamp="20260102T020202",
            )
        )
    mocked.assert_not_called()
    assert result.backend_status == "not_run"
    assert result.status == "dry_run_complete"
    assert not (result.run_dir / WORKING_TRACKED_POSE_FILENAME).exists()
