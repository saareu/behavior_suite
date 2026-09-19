"""S3 shell handoff validation. The current corrector is not invoked."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction import (
    CURRENT_BACKEND_ID,
    CURRENT_PROFILE_ID,
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    run_tracking_correction,
)
from tracking_correction.__main__ import app

REQUIRED_NODES = ("nose", "neck", "spine_base", "tail_base", "headstage")
_NODE_XY = {
    "nose": (0.0, 0.0),
    "neck": (30.0, -40.0),
    "spine_base": (55.0, -20.0),
    "tail_base": (90.0, -10.0),
    "headstage": (10.0, -55.0),
}
TIMESTAMP = "20260101T000000"


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


def _write_s1(session_root: Path, *, frame_count: int = 3) -> dict[str, Path]:
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


def _pose_rows(
    *,
    frame_indices: tuple[int, ...] = (0, 1, 2),
    prepared_frame_indices: tuple[int, ...] | None = None,
    include_score: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prepared = prepared_frame_indices or frame_indices
    if len(prepared) != len(frame_indices):
        raise AssertionError("test frame mappings must have the same length")
    for frame_idx, prepared_idx in zip(frame_indices, prepared, strict=True):
        for track_index, track in ((0, "track_0"), (1, "track_1")):
            origin_x = 140.0 + track_index * 180.0
            origin_y = 220.0
            for node in REQUIRED_NODES:
                if node == "headstage" and track_index == 1:
                    continue
                dx, dy = _NODE_XY[node]
                row: dict[str, object] = {
                    "frame_idx": frame_idx,
                    "prepared_frame_idx": prepared_idx,
                    "video_index": 0,
                    "track": track,
                    "node": node,
                    "x": float(origin_x + dx),
                    "y": float(origin_y + dy),
                }
                if include_score:
                    row["score"] = 0.9
                rows.append(row)
    return rows


def node_offset(node: str) -> int:
    return REQUIRED_NODES.index(node)


def _write_pose(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _completed_s2(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    status: str = "success",
    dry_run: bool = False,
    write_parquet: bool = True,
    include_video: bool = True,
) -> tuple[Path, dict[str, Path]]:
    session = tmp_path / "session"
    s1 = _write_s1(session)
    if not include_video:
        s1["prepared_video"].unlink()
    run_dir = session / "pose_inference" / "model__20260101T000000"
    run_dir.mkdir(parents=True)
    if write_parquet:
        _write_pose(run_dir / "pose.parquet", rows if rows is not None else _pose_rows())
    payload = {
        "schema_version": "pose_meta_v1",
        "run_id": run_dir.name,
        "status": status,
        "dry_run": dry_run,
        "input": {key: str(path.resolve()) for key, path in s1.items()},
    }
    (run_dir / "pose_meta.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir, s1


def _protected_inputs(run_dir: Path, s1: dict[str, Path]) -> list[Path]:
    return [
        run_dir / "pose.parquet",
        run_dir / "pose_meta.json",
        s1["prepared_video"],
        s1["prepare_meta"],
        s1["prepared_sync"],
    ]


def _run(
    run_dir: Path,
    *,
    dry_run: bool = False,
    output_root: Path | None = None,
) -> Any:
    return run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=run_dir,
            output_root=output_root,
            dry_run=dry_run,
            timestamp=TIMESTAMP,
        )
    )


def test_valid_s2_handoff_runs_backend_and_preserves_inputs(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path)
    before = _snapshot(_protected_inputs(run_dir, s1))
    s2_names = {path.name for path in run_dir.iterdir()}
    preprocess_names = {path.name for path in s1["preprocess_dir"].iterdir()}

    result = _run(run_dir)

    assert result.success is True
    assert result.status == "correction_complete"
    assert result.profile_id == CURRENT_PROFILE_ID
    assert result.backend_id == CURRENT_BACKEND_ID
    assert result.backend_status == "completed"
    assert result.run_dir == (
        s1["session_root"] / "tracking_correction" / f"{CURRENT_PROFILE_ID}__{TIMESTAMP}"
    )
    assert result.run_dir.is_dir()
    assert result.pose_parquet_path == (run_dir / "pose.parquet").resolve()
    assert result.prepared_video_path == s1["prepared_video"].resolve()
    assert result.working_tracked_pose_path is not None
    assert result.working_tracked_pose_path.is_file()
    assert result.machine_corrections_path is not None
    assert result.machine_corrections_path.is_file()
    assert not (result.run_dir / "tracked_pose.parquet").exists()
    assert s2_names == {path.name for path in run_dir.iterdir()}
    assert preprocess_names == {path.name for path in s1["preprocess_dir"].iterdir()}
    assert _snapshot(_protected_inputs(run_dir, s1)) == before

    meta = json.loads(result.run_meta_path.read_text(encoding="utf-8"))
    settings = yaml.safe_load(result.settings_used_path.read_text(encoding="utf-8"))
    log = result.processing_log_path.read_text(encoding="utf-8")
    assert meta["acceptance_state"] == "not_accepted"
    assert meta["backend_status"] == "completed"
    assert meta["provenance_source"] == "pose_meta"
    assert meta["timing"]["frame_count_used_for_sleap"] == 3
    assert meta["timing"]["fps_header"] == 20.0
    assert meta["outputs"]["tracked_pose_parquet"] is None
    assert meta["outputs"]["working_tracked_pose_parquet"] is not None
    assert settings["corrector_invoked"] is True
    assert settings["profile_id"] == CURRENT_PROFILE_ID
    assert settings["pose_parquet"] == str((run_dir / "pose.parquet").resolve())
    assert "backend_status: completed" in log
    assert "tracked_pose_parquet: not_generated" in log
    assert not result.run_dir.is_relative_to(run_dir)
    assert not result.run_dir.is_relative_to(s1["preprocess_dir"])


def test_missing_pose_parquet_is_rejected_without_creating_workspace(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path, write_parquet=False)
    before = _snapshot(_protected_inputs(run_dir, s1))

    with pytest.raises(TrackingCorrectionError, match="pose.parquet does not exist"):
        _run(run_dir)

    assert _snapshot(_protected_inputs(run_dir, s1)) == before
    assert not (s1["session_root"] / "tracking_correction").exists()


def test_unreadable_pose_parquet_is_rejected(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path, write_parquet=False)
    (run_dir / "pose.parquet").write_bytes(b"this is not a parquet file")
    before = _snapshot(_protected_inputs(run_dir, s1))

    with pytest.raises(TrackingCorrectionError, match="Could not read pose.parquet"):
        _run(run_dir)

    assert _snapshot(_protected_inputs(run_dir, s1)) == before
    assert not (s1["session_root"] / "tracking_correction").exists()


def test_missing_required_columns_are_rejected(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path, rows=_pose_rows(include_score=False))
    before = _snapshot(_protected_inputs(run_dir, s1))

    with pytest.raises(
        TrackingCorrectionError,
        match="pose.parquet is missing required columns: score",
    ):
        _run(run_dir)

    assert _snapshot(_protected_inputs(run_dir, s1)) == before
    assert not (s1["session_root"] / "tracking_correction").exists()


def test_missing_prepared_video_is_rejected(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path, include_video=False)
    before = _snapshot(_protected_inputs(run_dir, s1))

    with pytest.raises(TrackingCorrectionError, match="Prepared video does not exist"):
        _run(run_dir)

    assert _snapshot(_protected_inputs(run_dir, s1)) == before
    assert not (s1["session_root"] / "tracking_correction").exists()


def test_out_of_range_frame_indices_are_rejected(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(
        tmp_path,
        rows=_pose_rows(frame_indices=(0, 1, 99)),
    )
    before = _snapshot(_protected_inputs(run_dir, s1))

    with pytest.raises(
        TrackingCorrectionError,
        match="prepared-frame domain 0..2",
    ):
        _run(run_dir)

    assert _snapshot(_protected_inputs(run_dir, s1)) == before
    assert not (s1["session_root"] / "tracking_correction").exists()


def test_inconsistent_frame_indices_are_rejected(tmp_path: Path) -> None:
    rows = _pose_rows(
        frame_indices=(0, 1, 2),
        prepared_frame_indices=(2, 0, 1),
    )
    run_dir, s1 = _completed_s2(tmp_path, rows=rows)
    before = _snapshot(_protected_inputs(run_dir, s1))

    with pytest.raises(
        TrackingCorrectionError,
        match="Pose frame indices are inconsistent with the inherited S1/S2",
    ):
        _run(run_dir)

    assert _snapshot(_protected_inputs(run_dir, s1)) == before
    assert not (s1["session_root"] / "tracking_correction").exists()


def test_dry_run_writes_validation_records_and_preserves_inputs(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path)
    before = _snapshot(_protected_inputs(run_dir, s1))

    result = _run(run_dir, dry_run=True)

    assert result.status == "dry_run_complete"
    assert result.backend_status == "not_run"
    assert not (result.run_dir / "tracked_pose.parquet").exists()
    meta = json.loads(result.run_meta_path.read_text(encoding="utf-8"))
    assert meta["dry_run"] is True
    assert meta["status"] == "dry_run_complete"
    assert meta["backend_status"] == "not_run"
    assert _snapshot(_protected_inputs(run_dir, s1)) == before


def test_cli_dry_run_and_missing_parquet(tmp_path: Path) -> None:
    run_dir, s1 = _completed_s2(tmp_path)
    before = _snapshot(_protected_inputs(run_dir, s1))
    runner = CliRunner()

    dry_run = runner.invoke(
        app,
        ["run", "--s2-run", str(run_dir), "--dry-run"],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert "Status: dry_run_complete" in dry_run.output
    assert "Backend status: not_run" in dry_run.output
    assert _snapshot(_protected_inputs(run_dir, s1)) == before

    missing = tmp_path / "empty_s2"
    missing.mkdir()
    failed = runner.invoke(app, ["run", "--s2-run", str(missing)])
    assert failed.exit_code == 1
    assert "pose.parquet does not exist" in failed.output
