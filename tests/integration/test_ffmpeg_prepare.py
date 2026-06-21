import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess.config import CanonicalResolutionConfig, PrepareConfig, PreprocessConfig
from preprocess.exceptions import VideoPreparationError
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.video_prepare import (
    resolve_ffmpeg_binary,
    resolve_ffprobe_binary,
    run_ffmpeg_prepare,
)


def _require_ffmpeg() -> tuple[Path, Path]:
    try:
        return resolve_ffmpeg_binary(), resolve_ffprobe_binary()
    except VideoPreparationError as exc:
        pytest.skip(str(exc))


def _write_test_video(path: Path, frame_count: int = 8) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI fixture.")
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), 30 + index * 10, dtype=np.uint8)
            cv2.rectangle(frame, (8, 4), (55, 43), (40, 180, 80), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _config(ffmpeg_binary: Path, ffprobe_binary: Path) -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=96,
                height=96,
            )
        ),
        encoding={
            "ffmpeg": {
                "ffmpeg_path": ffmpeg_binary,
                "ffprobe_path": ffprobe_binary,
            }
        },
    )


def _accepted_crop_plan(config: PreprocessConfig):
    plan = make_manual_crop_plan(
        raw_frame_shape=(48, 64),
        points_tl_tr_br_bl=np.array(
            [[8.0, 4.0], [55.0, 4.0], [55.0, 43.0], [8.0, 43.0]]
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )
    return plan.model_copy(update={"accepted_by_user": True})


def _count_readable_frames(path: Path) -> tuple[int, tuple[int, int], np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert frames
    height, width = frames[0].shape[:2]
    return len(frames), (width, height), frames[0]


@pytest.mark.integration
def test_ffmpeg_prepare_produces_trimmed_canonical_intermediate(
    tmp_path: Path,
) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    raw_path = _write_test_video(tmp_path / "raw.avi")
    output_path = tmp_path / "internal" / "rectified.mp4"
    config = _config(ffmpeg_binary, ffprobe_binary)
    crop_plan = _accepted_crop_plan(config)

    result = run_ffmpeg_prepare(
        raw_video_path=raw_path,
        intermediate_path=output_path,
        crop_plan=crop_plan,
        config=config,
        start_frame=2,
        end_frame_exclusive=6,
    )

    frame_count, size_wh, first_frame = _count_readable_frames(output_path)
    assert result.intermediate_path == output_path.resolve()
    assert result.expected_output_size_wh == (96, 96)
    assert result.return_code == 0
    assert output_path.is_file()
    assert size_wh == (96, 96)
    assert frame_count == 4
    assert float(first_frame[:6].mean()) < 5.0
    assert float(first_frame[20:76].mean()) > 20.0

    audio_probe = subprocess.run(
        [
            str(ffprobe_binary),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert audio_probe.returncode == 0
    assert audio_probe.stdout.strip() == ""


@pytest.mark.integration
def test_ffmpeg_prepare_failure_raises_domain_error(tmp_path: Path) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    raw_path = tmp_path / "broken.avi"
    raw_path.write_text("not a video", encoding="utf-8")
    config = _config(ffmpeg_binary, ffprobe_binary)
    crop_plan = _accepted_crop_plan(config)

    with pytest.raises(VideoPreparationError, match="ffmpeg preparation failed"):
        run_ffmpeg_prepare(
            raw_video_path=raw_path,
            intermediate_path=tmp_path / "internal" / "failed.mp4",
            crop_plan=crop_plan,
            config=config,
            start_frame=0,
            end_frame_exclusive=2,
        )
