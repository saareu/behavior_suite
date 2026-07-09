import json
from pathlib import Path

import numpy as np
import pytest

from pose_inference.parquet_export import (
    PARQUET_COLUMNS,
    POSE_PARQUET_SCHEMA_VERSION,
    ParquetExportError,
    PoseFrameRecord,
    PoseInstanceRecord,
    PoseNodeRecord,
    export_pose_parquet,
    load_timing_lookup,
    pose_frames_to_rows,
)


def _preprocess_dir(tmp_path: Path, *, meta: dict[str, object] | None = None) -> Path:
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(meta or {"schema_version": "prepare_meta_v1"}) + "\n",
        encoding="utf-8",
    )
    return preprocess


def _pose_frames() -> list[PoseFrameRecord]:
    return [
        PoseFrameRecord(
            frame_idx=0,
            video_index=0,
            video_path="prepared_video.mp4",
            video_fps=20.0,
            instances=(
                PoseInstanceRecord(
                    instance_index=0,
                    track="track_0",
                    instance_score=0.91,
                    nodes=(
                        PoseNodeRecord(node="nose", x=1.0, y=2.0, score=0.8),
                        PoseNodeRecord(node="tail_base", x=None, y=None, score=None),
                    ),
                ),
                PoseInstanceRecord(
                    instance_index=1,
                    track=None,
                    instance_score=0.72,
                    nodes=(
                        PoseNodeRecord(node="nose", x=3.0, y=4.0, score=0.7),
                        PoseNodeRecord(node="tail_base", x=5.0, y=6.0, score=0.6),
                    ),
                ),
            ),
        ),
        PoseFrameRecord(
            frame_idx=1,
            video_index=0,
            video_path="prepared_video.mp4",
            instances=(
                PoseInstanceRecord(
                    instance_index=0,
                    track="track_0",
                    instance_score=0.88,
                    nodes=(
                        PoseNodeRecord(node="nose", x=7.0, y=8.0, score=0.9),
                        PoseNodeRecord(node="tail_base", x=9.0, y=10.0, score=0.85),
                    ),
                ),
            ),
        ),
    ]


def test_timing_vector_extracts_matching_prepared_sync_key(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path)
    np.savez(preprocess / "prepared_sync.npz", prepared_time_sec=np.array([0.0, 0.1, 0.2]))

    timing = load_timing_lookup(preprocess, frame_count=3)

    assert timing.source == "prepared_sync:prepared_time_sec"
    assert timing.time_for_frame(2) == pytest.approx(0.2)


def test_timing_falls_back_to_prepare_meta_fps(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path, meta={"prepared_video": {"fps": 20.0}})
    np.savez(preprocess / "prepared_sync.npz", unrelated=np.array([1.0, 2.0]))

    timing = load_timing_lookup(preprocess, frame_count=3)

    assert timing.source == "prepare_meta_fps"
    assert timing.time_for_frame(2) == pytest.approx(0.1)


def test_timing_unavailable_does_not_fail_export_rows(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path)

    timing = load_timing_lookup(preprocess, frame_count=2)
    rows = pose_frames_to_rows(_pose_frames(), timing)

    assert timing.source == "unavailable"
    assert rows[0]["time_sec"] is None
    assert rows[0]["timestamp_source"] == "unavailable"


def test_pose_rows_are_long_tidy_and_preserve_tracks_and_missing_nodes(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess_dir(tmp_path)
    np.savez(preprocess / "prepared_sync.npz", time_sec=np.array([0.0, 0.05]))
    timing = load_timing_lookup(preprocess, frame_count=2)

    rows = pose_frames_to_rows(_pose_frames(), timing)

    assert len(rows) == 6
    assert set(PARQUET_COLUMNS).issubset(rows[0])
    assert {row["schema_version"] for row in rows} == {POSE_PARQUET_SCHEMA_VERSION}
    assert rows[0]["track"] == "track_0"
    assert rows[0]["instance_score"] == pytest.approx(0.91)
    missing = rows[1]
    assert missing["node"] == "tail_base"
    assert missing["x"] is None
    assert missing["y"] is None
    assert missing["score"] is None
    assert missing["n_instances_in_frame"] == 2
    assert missing["n_valid_keypoints_in_instance"] == 1


def test_export_pose_parquet_uses_writer_without_real_sleap_or_parquet_engine(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess_dir(tmp_path)
    np.savez(preprocess / "prepared_sync.npz", timestamps_sec=np.array([0.0, 0.25]))
    pose_slp = tmp_path / "pose.slp"
    pose_slp.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    def writer(rows, output_path, columns):
        captured["rows"] = list(rows)
        captured["output_path"] = output_path
        captured["columns"] = tuple(columns)
        output_path.write_bytes(b"parquet")

    summary = export_pose_parquet(
        pose_slp_path=pose_slp,
        preprocess_dir=preprocess,
        output_path=tmp_path / "pose.parquet",
        labels_loader=lambda _path: _pose_frames(),
        parquet_writer=writer,
    )

    assert summary.status == "generated"
    assert summary.rows == 6
    assert summary.timestamp_sources == ("prepared_sync:timestamps_sec",)
    assert captured["output_path"] == tmp_path / "pose.parquet"
    assert "track_0" in {row["track"] for row in captured["rows"]}


def test_missing_pose_slp_fails_clearly(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path)

    with pytest.raises(ParquetExportError, match="pose.slp does not exist"):
        export_pose_parquet(
            pose_slp_path=tmp_path / "missing.slp",
            preprocess_dir=preprocess,
            output_path=tmp_path / "pose.parquet",
            labels_loader=lambda _path: _pose_frames(),
            parquet_writer=lambda _rows, _output, _columns: None,
        )
