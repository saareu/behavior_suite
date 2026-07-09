from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pose_inference.overlay import OverlayGenerationError, generate_overlay_video


def _write_video(
    path: Path,
    *,
    frame_count: int = 5,
    width: int = 64,
    height: int = 64,
    fps: float = 12.0,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((height, width, 3), 20 + index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _video_metadata(path: Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    metadata = {
        "frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": capture.get(cv2.CAP_PROP_FPS),
    }
    capture.release()
    return metadata


def test_overlay_creates_mp4_and_preserves_video_metadata(tmp_path: Path) -> None:
    video = tmp_path / "prepared_video.mp4"
    parquet = tmp_path / "pose.parquet"
    output = tmp_path / "overlay.mp4"
    _write_video(video, frame_count=5, width=64, height=64, fps=12.0)
    _write_parquet(
        parquet,
        [
            {
                "frame_idx": 0,
                "instance_index": 0,
                "track": "track_0",
                "node": "nose",
                "x": 10.0,
                "y": 11.0,
            },
            {
                "frame_idx": 1,
                "instance_index": 1,
                "track": "track_1",
                "node": "nose",
                "x": 20.0,
                "y": 21.0,
            },
        ],
    )

    summary = generate_overlay_video(
        prepared_video_path=video,
        pose_parquet_path=parquet,
        output_path=output,
    )
    metadata = _video_metadata(output)

    assert summary.status == "generated"
    assert summary.frames_written == 5
    assert summary.width == 64
    assert summary.height == 64
    assert summary.codec == "mp4v"
    assert output.is_file()
    assert int(metadata["frames"]) == 5
    assert int(metadata["width"]) == 64
    assert int(metadata["height"]) == 64
    assert 10.0 <= metadata["fps"] <= 14.0


def test_overlay_skips_missing_coordinates_and_writes_frames_without_pose_rows(
    tmp_path: Path,
) -> None:
    video = tmp_path / "prepared_video.mp4"
    parquet = tmp_path / "pose.parquet"
    output = tmp_path / "overlay.mp4"
    _write_video(video, frame_count=3)
    _write_parquet(
        parquet,
        [
            {
                "frame_idx": 0,
                "instance_index": 0,
                "track": "track_0",
                "node": "nose",
                "x": None,
                "y": None,
            },
            {
                "frame_idx": 2,
                "instance_index": 0,
                "track": "track_0",
                "node": "tail_base",
                "x": 30.0,
                "y": 31.0,
            },
        ],
    )

    summary = generate_overlay_video(
        prepared_video_path=video,
        pose_parquet_path=parquet,
        output_path=output,
    )

    assert summary.status == "generated"
    assert summary.frames_written == 3
    assert int(_video_metadata(output)["frames"]) == 3


def test_overlay_uses_instance_coloring_when_track_column_is_absent(tmp_path: Path) -> None:
    video = tmp_path / "prepared_video.mp4"
    parquet = tmp_path / "pose.parquet"
    output = tmp_path / "overlay.mp4"
    _write_video(video, frame_count=3)
    _write_parquet(
        parquet,
        [
            {"frame_idx": 0, "instance_index": 0, "node": "nose", "x": 10.0, "y": 10.0},
            {"frame_idx": 1, "instance_index": 1, "node": "nose", "x": 20.0, "y": 20.0},
        ],
    )

    summary = generate_overlay_video(
        prepared_video_path=video,
        pose_parquet_path=parquet,
        output_path=output,
    )

    assert summary.status == "generated"
    assert output.is_file()


def test_overlay_fails_clearly_when_prepared_video_is_missing(tmp_path: Path) -> None:
    parquet = tmp_path / "pose.parquet"
    _write_parquet(
        parquet,
        [{"frame_idx": 0, "instance_index": 0, "node": "nose", "x": 10.0, "y": 10.0}],
    )

    with pytest.raises(OverlayGenerationError, match="prepared_video.mp4 does not exist"):
        generate_overlay_video(
            prepared_video_path=tmp_path / "missing.mp4",
            pose_parquet_path=parquet,
            output_path=tmp_path / "overlay.mp4",
        )
