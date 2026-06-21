from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess import video_probe
from preprocess.exceptions import VideoProbeError


def _write_tiny_video(path: Path, frame_count: int = 5) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI test fixture.")
    try:
        for index in range(frame_count):
            frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_opencv_reported_and_readable_frame_counts_are_distinct_and_correct(
    tmp_path: Path,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=5)

    assert video_probe.get_opencv_reported_frame_count(video_path) == 5
    assert video_probe.count_opencv_readable_frames(video_path) == 5


def test_probe_video_with_sequential_count(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)

    result = video_probe.probe_video(video_path, require_sequential_count=True)

    assert result.width == 32
    assert result.height == 24
    assert result.frame_count_opencv_reported == 4
    assert result.frame_count_opencv_readable == 4
    assert result.opencv_fps == pytest.approx(10.0)


def test_probe_video_skips_full_count_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi")

    def fail_if_called(_path: Path) -> int:
        raise AssertionError("sequential counter should not be called")

    monkeypatch.setattr(video_probe, "count_opencv_readable_frames", fail_if_called)

    result = video_probe.probe_video(video_path, require_sequential_count=False)

    assert result.frame_count_opencv_readable is None
    assert result.frame_count_opencv_reported == 5


def test_readable_video_remains_probeable_without_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi")
    monkeypatch.setattr(video_probe.shutil, "which", lambda _name: None)

    result = video_probe.probe_video(video_path, require_sequential_count=False)

    assert result.ffprobe_succeeded is False
    assert result.ffprobe_error is not None
    assert "not found" in result.ffprobe_error
    assert result.raw_fps_effective_method == "opencv_fps"


def test_unreadable_video_raises_domain_error(tmp_path: Path) -> None:
    unreadable_path = tmp_path / "broken.avi"
    unreadable_path.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoProbeError, match="OpenCV"):
        video_probe.probe_video(unreadable_path, require_sequential_count=False)
