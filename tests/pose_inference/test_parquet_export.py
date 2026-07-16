import json
import math
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
    TimingLookup,
    _instance_nodes,
    export_pose_parquet,
    load_timing_lookup,
    pose_frames_to_rows,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


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


def test_s1_prepared_sync_timing_and_frame_mapping_are_preserved(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path)
    sync = build_prepared_sync(
        prepared_frame_count=3,
        start_frame=10,
        end_frame_exclusive=13,
        fps_header=20.0,
        raw_fps_effective=30.0,
        raw_pts_time_sec=np.array([100.0, 100.5, 101.0]),
        raw_pts_status="valid",
        external_ttl_vector=np.arange(20, dtype=np.float64) * 0.1,
        external_time_status="valid",
        external_time_source="timing.mat",
        external_time_variable_name="camera_time",
        external_time_units="seconds",
        raw_frame_count_opencv_readable=20,
        prepared_frame_count_opencv_reported=3,
        prepared_frame_count_opencv_readable=3,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    frames = [
        PoseFrameRecord(
            frame_idx=1,
            original_frame_idx=1,
            source_video_frame_idx=1,
            instances=(
                PoseInstanceRecord(
                    instance_index=0,
                    nodes=(PoseNodeRecord(node="nose", x=1.0, y=2.0, score=0.8),),
                ),
            ),
        )
    ]

    timing = load_timing_lookup(preprocess, frame_count=3)
    rows = pose_frames_to_rows(frames, timing)

    row = rows[0]
    assert timing.status == "prepared_sync"
    assert row["time_sec"] == pytest.approx(1.1)
    assert row["timestamp_source"] == "prepared_sync:external_time_sec"
    assert row["prepared_frame_idx"] == 1
    assert row["raw_decode_frame_idx"] == 11
    assert row["original_frame_idx"] == 11
    assert row["source_video_frame_idx"] == 11
    assert row["prepared_time_sec"] == pytest.approx(0.05)
    assert row["raw_pts_time_sec"] == pytest.approx(100.5)
    assert row["external_time_sec"] == pytest.approx(1.1)
    assert row["raw_pts_status"] == "valid"
    assert row["external_time_status"] == "valid"
    assert row["external_time_source"] == "timing.mat"
    assert row["external_time_variable_name"] == "camera_time"
    assert row["external_time_units"] == "seconds"
    assert row["fps_header"] == pytest.approx(20.0)
    assert row["raw_fps_effective"] == pytest.approx(30.0)
    assert row["sync_frame_count_used_for_sleap"] == 3
    assert timing.to_metadata([1])["timestamp_sources"] == ["prepared_sync:external_time_sec"]


def test_s1_prepared_sync_uses_prepared_time_when_external_and_raw_pts_are_missing(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess_dir(tmp_path)
    sync = build_prepared_sync(
        prepared_frame_count=2,
        start_frame=5,
        end_frame_exclusive=7,
        fps_header=10.0,
        raw_fps_effective=10.0,
        raw_pts_time_sec=None,
        raw_pts_status="not_extracted",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=10,
        prepared_frame_count_opencv_reported=2,
        prepared_frame_count_opencv_readable=2,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)

    timing = load_timing_lookup(preprocess, frame_count=2)
    time_sec, source = timing.timestamp_for_frame(1)

    assert time_sec == pytest.approx(0.1)
    assert source == "prepared_sync:prepared_time_sec"
    assert timing.fields_for_frame(1)["external_time_sec"] is None
    assert timing.fields_for_frame(1)["raw_pts_time_sec"] is None


def test_timing_falls_back_to_prepare_meta_fps(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path, meta={"prepared_video": {"fps": 20.0}})
    np.savez(preprocess / "prepared_sync.npz", unrelated=np.array([1.0, 2.0]))

    timing = load_timing_lookup(preprocess, frame_count=3)

    assert timing.source == "prepare_meta_fps_fallback"
    assert timing.status == "fallback_fps"
    assert timing.timestamp_for_frame(2)[1] == "prepare_meta_fps_fallback"
    assert timing.time_for_frame(2) == pytest.approx(0.1)


def test_timing_unavailable_does_not_fail_export_rows(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path)

    timing = load_timing_lookup(preprocess, frame_count=2)
    rows = pose_frames_to_rows(_pose_frames(), timing)

    assert timing.source == "unavailable"
    assert timing.status == "unavailable"
    assert rows[0]["time_sec"] is None
    assert rows[0]["timestamp_source"] == "unavailable"
    assert rows[0]["raw_decode_frame_idx"] is None


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


def test_structured_predicted_points_array_exports_real_xy_score_and_names() -> None:
    point_dtype = [
        ("xy", "<f8", (2,)),
        ("score", "<f8"),
        ("visible", "?"),
        ("complete", "?"),
        ("name", "O"),
    ]
    points = np.array(
        [
            ([792.0, 75.6], 0.85, True, False, "nose"),
            ([758.0, 77.6], 0.83, True, False, "neck"),
            ([math.nan, math.nan], math.nan, False, False, "headstage"),
        ],
        dtype=point_dtype,
    )
    instance = type("PredictedInstance", (), {"points": points})()

    nodes = _instance_nodes(instance)
    rows = pose_frames_to_rows(
        [
            PoseFrameRecord(
                frame_idx=0,
                instances=(
                    PoseInstanceRecord(
                        instance_index=0,
                        nodes=nodes,
                        instance_score=0.91,
                    ),
                ),
            )
        ],
        TimingLookup(source="unavailable"),
    )

    assert [(node.node, node.x, node.y, node.score) for node in nodes[:2]] == [
        ("nose", 792.0, 75.6, 0.85),
        ("neck", 758.0, 77.6, 0.83),
    ]
    assert nodes[2].node == "headstage"
    assert nodes[2].x is not None and math.isnan(nodes[2].x)
    assert nodes[2].y is not None and math.isnan(nodes[2].y)
    assert nodes[2].score is not None and math.isnan(nodes[2].score)
    assert rows[0]["n_valid_keypoints_in_instance"] == 2
    assert rows[1]["n_valid_keypoints_in_instance"] == 2
    assert rows[2]["n_valid_keypoints_in_instance"] == 2
    assert rows[0]["x"] == pytest.approx(792.0)
    assert rows[0]["y"] == pytest.approx(75.6)
    assert rows[0]["score"] == pytest.approx(0.85)
    assert rows[0]["node"] == "nose"
    assert rows[2]["x"] is not None and math.isnan(rows[2]["x"])
    assert rows[2]["y"] is not None and math.isnan(rows[2]["y"])


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
    assert summary.timing["represented_frames"] == 2
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


def test_prediction_frame_mismatch_with_s1_contract_fails_export(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess_dir(tmp_path)
    sync = build_prepared_sync(
        prepared_frame_count=3,
        start_frame=0,
        end_frame_exclusive=3,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.array([0.0, 0.05, 0.1]),
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=3,
        prepared_frame_count_opencv_reported=3,
        prepared_frame_count_opencv_readable=3,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    pose_slp = tmp_path / "pose.slp"
    pose_slp.write_bytes(b"placeholder")

    with pytest.raises(ParquetExportError, match="authoritative S1"):
        export_pose_parquet(
            pose_slp_path=pose_slp,
            preprocess_dir=preprocess,
            output_path=tmp_path / "pose.parquet",
            labels_loader=lambda _path: [PoseFrameRecord(frame_idx=3, instances=())],
            parquet_writer=lambda _rows, _output, _columns: None,
        )


def test_sparse_prediction_frames_within_s1_range_are_reconciled(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess_dir(tmp_path)
    sync = build_prepared_sync(
        prepared_frame_count=3,
        start_frame=0,
        end_frame_exclusive=3,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.array([0.0, 0.05, 0.1]),
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=3,
        prepared_frame_count_opencv_reported=3,
        prepared_frame_count_opencv_readable=3,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    pose_slp = tmp_path / "pose.slp"
    pose_slp.write_bytes(b"placeholder")

    def writer(rows, output_path, _columns):
        assert {row["frame_idx"] for row in rows} == {0, 1}
        output_path.write_bytes(b"parquet")

    summary = export_pose_parquet(
        pose_slp_path=pose_slp,
        preprocess_dir=preprocess,
        output_path=tmp_path / "pose.parquet",
        labels_loader=lambda _path: _pose_frames(),
        parquet_writer=writer,
    )

    assert summary.status == "generated"
    assert summary.timing["represented_frames"] == 2


def test_inconsistent_source_prepared_frame_idx_is_rejected(tmp_path: Path) -> None:
    preprocess = _preprocess_dir(tmp_path)
    sync = build_prepared_sync(
        prepared_frame_count=3,
        start_frame=0,
        end_frame_exclusive=3,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.array([0.0, 0.05, 0.1]),
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=3,
        prepared_frame_count_opencv_reported=3,
        prepared_frame_count_opencv_readable=3,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    pose_slp = tmp_path / "pose.slp"
    pose_slp.write_bytes(b"placeholder")

    with pytest.raises(ParquetExportError, match="prepared_frame_idx"):
        export_pose_parquet(
            pose_slp_path=pose_slp,
            preprocess_dir=preprocess,
            output_path=tmp_path / "pose.parquet",
            labels_loader=lambda _path: [
                PoseFrameRecord(
                    frame_idx=1,
                    prepared_frame_idx=0,
                    instances=(),
                )
            ],
            parquet_writer=lambda _rows, _output, _columns: None,
        )


def test_equal_frame_indices_from_two_videos_reconcile_independently(
    tmp_path: Path,
) -> None:
    preprocess = _preprocess_dir(tmp_path)
    sync = build_prepared_sync(
        prepared_frame_count=1,
        start_frame=0,
        end_frame_exclusive=1,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.array([0.0]),
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=1,
        prepared_frame_count_opencv_reported=1,
        prepared_frame_count_opencv_readable=1,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    pose_slp = tmp_path / "pose.slp"
    pose_slp.write_bytes(b"placeholder")
    frames = [
        PoseFrameRecord(frame_idx=0, video_index=0, instances=()),
        PoseFrameRecord(frame_idx=0, video_index=1, instances=()),
    ]

    def writer(_rows, output_path, _columns):
        output_path.write_bytes(b"parquet")

    summary = export_pose_parquet(
        pose_slp_path=pose_slp,
        preprocess_dir=preprocess,
        output_path=tmp_path / "pose.parquet",
        labels_loader=lambda _path: frames,
        parquet_writer=writer,
    )

    assert summary.status == "generated"
    assert summary.timing["represented_frames"] == 2
