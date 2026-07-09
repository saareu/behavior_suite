from pathlib import Path

import pandas as pd
import pytest

from pose_inference.pose_qc import PoseQcError, compute_pose_qc_from_parquet


def _row(
    frame_idx: int,
    instance_index: int,
    node: str,
    x: float | None,
    y: float | None,
    *,
    score: float | None = 0.9,
    instance_score: float | None = 0.95,
    timestamp_source: str = "prepared_sync:time_sec",
) -> dict[str, object]:
    return {
        "frame_idx": frame_idx,
        "instance_index": instance_index,
        "node": node,
        "x": x,
        "y": y,
        "score": score,
        "instance_score": instance_score,
        "timestamp_source": timestamp_source,
    }


def _instance_rows(
    frame_idx: int,
    instance_index: int,
    *,
    nose: tuple[float | None, float | None],
    tail: tuple[float | None, float | None],
    score: float | None = 0.9,
) -> list[dict[str, object]]:
    return [
        _row(frame_idx, instance_index, "nose", nose[0], nose[1], score=score),
        _row(frame_idx, instance_index, "tail_base", tail[0], tail[1], score=score),
    ]


def _write_parquet(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "pose.parquet"
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    return path


def test_pose_qc_reports_perfect_two_animal_output(tmp_path: Path) -> None:
    rows = []
    for frame_idx in range(2):
        rows.extend(
            _instance_rows(
                frame_idx,
                0,
                nose=(0.0, 0.0),
                tail=(10.0, 0.0),
            )
        )
        rows.extend(
            _instance_rows(
                frame_idx,
                1,
                nose=(100.0, 100.0),
                tail=(110.0, 100.0),
            )
        )

    qc = compute_pose_qc_from_parquet(pose_parquet_path=_write_parquet(tmp_path, rows))
    metadata = qc.to_metadata()

    assert metadata["status"] == "computed"
    assert metadata["frames"]["total"] == 2
    assert metadata["frames"]["expected_count"]["count"] == 2
    assert metadata["frames"]["zero_animals"]["count"] == 0
    assert metadata["instances"]["valid"] == 4
    assert metadata["keypoints"]["missing_coordinates"]["count"] == 0
    assert metadata["keypoints"]["low_confidence"]["count"] == 0
    assert metadata["skeletons"]["partial_skeleton_instances"] == 0
    assert metadata["duplicate_candidate_risk"]["frames_flagged"] == 0
    assert metadata["implausible_geometry"]["frames_flagged"] == 0
    assert metadata["timestamp_sources"] == ["prepared_sync:time_sec"]
    assert metadata["warnings"] == []


def test_pose_qc_counts_missing_low_confidence_and_partial_skeletons(tmp_path: Path) -> None:
    rows = [
        *_instance_rows(0, 0, nose=(0.0, 0.0), tail=(10.0, 0.0), score=0.9),
        *_instance_rows(0, 1, nose=(20.0, 20.0), tail=(None, None), score=0.1),
    ]

    metadata = compute_pose_qc_from_parquet(
        pose_parquet_path=_write_parquet(tmp_path, rows),
        score_threshold=0.2,
    ).to_metadata()

    assert metadata["keypoints"]["missing_coordinates"]["count"] == 1
    assert metadata["keypoints"]["low_confidence"]["count"] == 2
    assert metadata["skeletons"]["partial_skeleton_instances"] == 1
    assert "missing keypoint coordinates are present" in metadata["warnings"]
    assert "low-confidence keypoints are present" in metadata["warnings"]


def test_pose_qc_counts_zero_fewer_and_more_than_expected_animals(tmp_path: Path) -> None:
    rows = [
        *_instance_rows(0, 0, nose=(0.0, 0.0), tail=(1.0, 0.0)),
        *_instance_rows(0, 1, nose=(10.0, 0.0), tail=(11.0, 0.0)),
        *_instance_rows(1, 0, nose=(0.0, 0.0), tail=(1.0, 0.0)),
        *_instance_rows(2, 0, nose=(None, None), tail=(None, None)),
        *_instance_rows(3, 0, nose=(0.0, 0.0), tail=(1.0, 0.0)),
        *_instance_rows(3, 1, nose=(10.0, 0.0), tail=(11.0, 0.0)),
        *_instance_rows(3, 2, nose=(20.0, 0.0), tail=(21.0, 0.0)),
    ]

    frames = compute_pose_qc_from_parquet(
        pose_parquet_path=_write_parquet(tmp_path, rows),
        expected_animals=2,
    ).to_metadata()["frames"]

    assert frames["zero_animals"]["count"] == 1
    assert frames["zero_animals"]["example_frame_indices"] == [2]
    assert frames["fewer_than_expected"]["count"] == 2
    assert frames["more_than_expected"]["count"] == 1
    assert frames["more_than_expected"]["example_frame_indices"] == [3]


def test_pose_qc_flags_duplicate_candidate_risk_for_extra_close_instances(
    tmp_path: Path,
) -> None:
    rows = [
        *_instance_rows(0, 0, nose=(0.0, 0.0), tail=(1.0, 0.0)),
        *_instance_rows(0, 1, nose=(50.0, 50.0), tail=(51.0, 50.0)),
        *_instance_rows(0, 2, nose=(53.0, 52.0), tail=(54.0, 52.0)),
    ]

    duplicate_risk = compute_pose_qc_from_parquet(
        pose_parquet_path=_write_parquet(tmp_path, rows),
        expected_animals=2,
        duplicate_distance_px=15.0,
    ).to_metadata()["duplicate_candidate_risk"]

    assert duplicate_risk["frames_with_extra_animals"] == 1
    assert duplicate_risk["frames_flagged"] == 1
    assert duplicate_risk["example_frame_indices"] == [0]
    assert duplicate_risk["examples"][0]["node"] == "nose"


def test_pose_qc_flags_implausible_geometry_outlier_conservatively(tmp_path: Path) -> None:
    rows = [
        *_instance_rows(0, 0, nose=(0.0, 0.0), tail=(10.0, 0.0)),
        *_instance_rows(1, 0, nose=(0.0, 0.0), tail=(10.0, 0.0)),
        *_instance_rows(2, 0, nose=(0.0, 0.0), tail=(1000.0, 0.0)),
    ]

    geometry = compute_pose_qc_from_parquet(
        pose_parquet_path=_write_parquet(tmp_path, rows),
        expected_animals=1,
    ).to_metadata()["implausible_geometry"]

    assert geometry["status"] == "computed"
    assert geometry["frames_flagged"] == 1
    assert geometry["instances_flagged"] == 1
    assert geometry["example_frame_indices"] == [2]


def test_pose_qc_tolerates_missing_optional_columns(tmp_path: Path) -> None:
    rows = [
        {"frame_idx": 0, "instance_index": 0, "node": "nose", "x": 0.0, "y": 0.0},
        {"frame_idx": 0, "instance_index": 1, "node": "nose", "x": 10.0, "y": 10.0},
    ]

    metadata = compute_pose_qc_from_parquet(
        pose_parquet_path=_write_parquet(tmp_path, rows),
        expected_animals=2,
    ).to_metadata()

    assert metadata["status"] == "computed"
    assert metadata["frames"]["expected_count"]["count"] == 1
    assert metadata["timestamp_sources"] == []
    assert metadata["instances"]["instance_score"]["status"] == "unavailable"


def test_pose_qc_fails_clearly_for_missing_required_columns(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path, [{"frame_idx": 0, "node": "nose", "x": 0.0, "y": 0.0}])

    with pytest.raises(PoseQcError, match="instance_index"):
        compute_pose_qc_from_parquet(pose_parquet_path=path)
