from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess import validation
from preprocess.config import PreprocessConfig
from preprocess.exceptions import VideoPreparationError, VideoValidationError
from preprocess.validation import (
    PreparedVideoValidationResult,
    validate_prepared_video,
)
from preprocess.video_prepare import reencode_intermediate_with_opencv


class _FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        reported_count: float = 3.0,
        size_wh: tuple[int, int] = (64, 48),
        fps: float = 10.0,
        readable_count: int = 3,
    ) -> None:
        self._opened = opened
        self._reported_count = reported_count
        self._size_wh = size_wh
        self._fps = fps
        self._frames = [
            np.zeros((size_wh[1], size_wh[0], 3), dtype=np.uint8)
            for _index in range(readable_count)
        ]
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_COUNT: self._reported_count,
            cv2.CAP_PROP_FRAME_WIDTH: float(self._size_wh[0]),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._size_wh[1]),
            cv2.CAP_PROP_FPS: self._fps,
        }
        return values[property_id]

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _prepared_path(tmp_path: Path) -> Path:
    path = tmp_path / "prepared.mp4"
    path.write_bytes(b"test fixture placeholder")
    return path


def _install_capture(
    monkeypatch: pytest.MonkeyPatch,
    capture: _FakeCapture,
) -> None:
    monkeypatch.setattr(validation.cv2, "VideoCapture", lambda _path: capture)


def _failure_result(exception: pytest.ExceptionInfo[VideoValidationError]) -> PreparedVideoValidationResult:
    result = exception.value.result
    assert isinstance(result, PreparedVideoValidationResult)
    return result


def test_valid_prepared_video_passes_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture()
    _install_capture(monkeypatch, capture)

    result = validate_prepared_video(
        _prepared_path(tmp_path),
        expected_frame_count=3,
        expected_size_wh=(64, 48),
    )

    assert result.is_valid is True
    assert result.errors == ()
    assert result.opencv_reported_frame_count == 3
    assert result.opencv_readable_frame_count == 3
    assert result.opencv_reported_size_wh == (64, 48)
    assert result.opencv_reported_fps == pytest.approx(10.0)
    assert capture.released is True


def test_missing_file_fails_validation(tmp_path: Path) -> None:
    with pytest.raises(VideoValidationError, match="does not exist") as exception:
        validate_prepared_video(
            tmp_path / "missing.mp4",
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )

    assert _failure_result(exception).is_valid is False


def test_unreadable_file_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture(opened=False)
    _install_capture(monkeypatch, capture)

    with pytest.raises(VideoValidationError, match="could not open") as exception:
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )

    assert _failure_result(exception).opencv_readable_frame_count == 0
    assert capture.released is True


def test_reported_readable_frame_mismatch_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_capture(
        monkeypatch,
        _FakeCapture(reported_count=3.0, readable_count=2),
    )

    with pytest.raises(
        VideoValidationError,
        match="does not match sequential readable count",
    ) as exception:
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=2,
            expected_size_wh=(64, 48),
        )

    result = _failure_result(exception)
    assert result.opencv_reported_frame_count == 3
    assert result.opencv_readable_frame_count == 2


def test_unavailable_reported_frame_count_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_capture(
        monkeypatch,
        _FakeCapture(reported_count=0.0, readable_count=3),
    )

    with pytest.raises(
        VideoValidationError,
        match="frame count must be available and positive",
    ) as exception:
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )

    assert _failure_result(exception).opencv_reported_frame_count == 0


def test_readable_expected_frame_mismatch_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_capture(
        monkeypatch,
        _FakeCapture(reported_count=2.0, readable_count=2),
    )

    with pytest.raises(
        VideoValidationError,
        match="does not match expected 3",
    ) as exception:
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )

    assert _failure_result(exception).opencv_readable_frame_count == 2


def test_wrong_dimensions_fail_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_capture(monkeypatch, _FakeCapture(size_wh=(80, 48)))

    with pytest.raises(VideoValidationError, match="does not match expected") as exception:
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )

    assert _failure_result(exception).opencv_reported_size_wh == (80, 48)


def test_odd_reported_dimensions_fail_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_capture(monkeypatch, _FakeCapture(size_wh=(63, 48)))

    with pytest.raises(VideoValidationError, match="must be even") as exception:
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )

    assert _failure_result(exception).opencv_reported_size_wh == (63, 48)


@pytest.mark.parametrize("expected_frame_count", [0, -1, True])
def test_invalid_expected_frame_count_fails(
    tmp_path: Path,
    expected_frame_count: int,
) -> None:
    with pytest.raises(VideoValidationError, match="positive integer"):
        validate_prepared_video(
            tmp_path / "unused.mp4",
            expected_frame_count=expected_frame_count,
            expected_size_wh=(64, 48),
        )


@pytest.mark.parametrize(
    "expected_size_wh",
    [(0, 48), (-2, 48), (63, 48), (64, 47)],
)
def test_invalid_expected_dimensions_fail(
    tmp_path: Path,
    expected_size_wh: tuple[int, int],
) -> None:
    with pytest.raises(VideoValidationError, match="must be (positive|even)"):
        validate_prepared_video(
            tmp_path / "unused.mp4",
            expected_frame_count=3,
            expected_size_wh=expected_size_wh,
        )


@pytest.mark.parametrize("reported_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_reported_fps_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_fps: float,
) -> None:
    _install_capture(monkeypatch, _FakeCapture(fps=reported_fps))

    with pytest.raises(VideoValidationError, match="FPS must be finite and positive"):
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        )


@pytest.mark.parametrize("output_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_stage_b_output_fps_fails(
    tmp_path: Path,
    output_fps: float,
) -> None:
    with pytest.raises(VideoPreparationError, match="FPS must be finite and positive"):
        reencode_intermediate_with_opencv(
            intermediate_video_path=tmp_path / "unused.mp4",
            prepared_video_path=tmp_path / "prepared.mp4",
            output_fps=output_fps,
            output_size_wh=(64, 48),
            config=PreprocessConfig(),
        )
