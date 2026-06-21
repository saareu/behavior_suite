import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess import video_prepare
from preprocess.config import PreprocessConfig
from preprocess.exceptions import VideoPreparationError, VideoValidationError
from preprocess.validation import validate_prepared_video
from preprocess.video_prepare import (
    reencode_intermediate_with_opencv,
    resolve_ffprobe_binary,
)

_FRAME_COUNT = 6
_SIZE_WH = (64, 48)
_OUTPUT_FPS = 12.5


def _write_intermediate(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        _SIZE_WH,
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI fixture.")
    try:
        for index in range(_FRAME_COUNT):
            frame = np.full(
                (_SIZE_WH[1], _SIZE_WH[0], 3),
                20 + index * 35,
                dtype=np.uint8,
            )
            cv2.rectangle(
                frame,
                (3 + index * 5, 4),
                (8 + index * 5, 10),
                (255, 255, 255),
                thickness=-1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def _read_video(path: Path) -> tuple[list[np.ndarray], tuple[int, int], float]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    reported_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames, reported_size, reported_fps


@pytest.mark.integration
def test_stage_b_reencodes_validates_and_preserves_frame_order(
    tmp_path: Path,
) -> None:
    try:
        ffprobe_binary = resolve_ffprobe_binary()
    except VideoPreparationError as exc:
        pytest.skip(str(exc))
    intermediate_path = _write_intermediate(tmp_path / "intermediate.avi")
    prepared_path = tmp_path / "prepared.mp4"

    result = reencode_intermediate_with_opencv(
        intermediate_video_path=intermediate_path,
        prepared_video_path=prepared_path,
        output_fps=_OUTPUT_FPS,
        output_size_wh=_SIZE_WH,
        config=PreprocessConfig(),
    )

    input_frames, input_size, _input_fps = _read_video(intermediate_path)
    output_frames, output_size, output_fps = _read_video(prepared_path)
    validation = validate_prepared_video(
        prepared_path,
        expected_frame_count=len(input_frames),
        expected_size_wh=_SIZE_WH,
    )
    assert result.prepared_video_path == prepared_path.resolve()
    assert result.temporary_output_path.parent == prepared_path.parent.resolve()
    assert not result.temporary_output_path.exists()
    assert result.frames_written == len(input_frames) == _FRAME_COUNT
    assert result.output_size_wh == input_size == output_size == _SIZE_WH
    assert result.output_fps == pytest.approx(_OUTPUT_FPS)
    assert result.opencv_fourcc == "mp4v"
    assert result.validation_result.is_valid is True
    assert validation.is_valid is True
    assert validation.opencv_reported_frame_count == _FRAME_COUNT
    assert validation.opencv_readable_frame_count == _FRAME_COUNT
    assert output_fps == pytest.approx(_OUTPUT_FPS, abs=0.05)

    output_means = np.asarray([float(frame.mean()) for frame in output_frames])
    assert len(output_frames) == len(input_frames)
    assert np.all(np.diff(output_means) > 20.0)

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
            str(prepared_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert audio_probe.returncode == 0
    assert audio_probe.stdout.strip() == ""


@pytest.mark.integration
def test_stage_b_rejects_incorrect_intermediate_frame_dimensions(
    tmp_path: Path,
) -> None:
    intermediate_path = _write_intermediate(tmp_path / "intermediate.avi")
    prepared_path = tmp_path / "prepared.mp4"
    existing_output = b"existing validated output placeholder"
    prepared_path.write_bytes(existing_output)

    with pytest.raises(VideoPreparationError, match="does not resize frames"):
        reencode_intermediate_with_opencv(
            intermediate_video_path=intermediate_path,
            prepared_video_path=prepared_path,
            output_fps=_OUTPUT_FPS,
            output_size_wh=(66, 48),
            config=PreprocessConfig(),
        )

    assert prepared_path.read_bytes() == existing_output
    assert list(tmp_path.glob(".prepared.stage-b-*.mp4")) == []


@pytest.mark.integration
def test_stage_b_validation_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate_path = _write_intermediate(tmp_path / "intermediate.avi")
    prepared_path = tmp_path / "prepared.mp4"
    existing_output = b"existing validated output placeholder"
    prepared_path.write_bytes(existing_output)

    def reject_temporary_output(
        _path: Path,
        expected_frame_count: int,
        expected_size_wh: tuple[int, int],
    ) -> None:
        assert expected_frame_count == _FRAME_COUNT
        assert expected_size_wh == _SIZE_WH
        raise VideoValidationError("forced temporary-output validation failure")

    monkeypatch.setattr(
        video_prepare,
        "validate_prepared_video",
        reject_temporary_output,
    )

    with pytest.raises(VideoValidationError, match="forced temporary-output"):
        reencode_intermediate_with_opencv(
            intermediate_video_path=intermediate_path,
            prepared_video_path=prepared_path,
            output_fps=_OUTPUT_FPS,
            output_size_wh=_SIZE_WH,
            config=PreprocessConfig(),
        )

    assert prepared_path.read_bytes() == existing_output
    assert list(tmp_path.glob(".prepared.stage-b-*.mp4")) == []


class _ClosedVideoWriter:
    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        return None


@pytest.mark.integration
def test_stage_b_writer_failure_raises_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate_path = _write_intermediate(tmp_path / "intermediate.avi")
    prepared_path = tmp_path / "prepared.mp4"
    monkeypatch.setattr(
        video_prepare.cv2,
        "VideoWriter",
        lambda *_args, **_kwargs: _ClosedVideoWriter(),
    )

    with pytest.raises(VideoPreparationError, match="could not open VideoWriter"):
        reencode_intermediate_with_opencv(
            intermediate_video_path=intermediate_path,
            prepared_video_path=prepared_path,
            output_fps=_OUTPUT_FPS,
            output_size_wh=_SIZE_WH,
            config=PreprocessConfig(),
        )

    assert not prepared_path.exists()
    assert list(tmp_path.glob(".prepared.stage-b-*.mp4")) == []
