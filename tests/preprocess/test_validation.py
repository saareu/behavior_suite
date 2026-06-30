from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess import validation, video_prepare
from preprocess.config import MaskConfig, PreprocessConfig
from preprocess.exceptions import (
    PreprocessCancelledError,
    VideoPreparationError,
    VideoValidationError,
)
from preprocess.masking import apply_static_mask_to_frame, rasterize_static_mask
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
        frames: list[np.ndarray] | None = None,
    ) -> None:
        self._opened = opened
        self._reported_count = reported_count
        self._size_wh = size_wh
        self._fps = fps
        self._frames = frames or [
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


class _FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames_written = 0
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.array(frame, copy=True))
        self.frames_written += 1

    def release(self) -> None:
        self.released = True
        self.path.write_bytes(b"partial encoded video")


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

    progress = []
    result = validate_prepared_video(
        _prepared_path(tmp_path),
        expected_frame_count=3,
        expected_size_wh=(64, 48),
        progress_callback=progress.append,
    )

    assert result.is_valid is True
    assert result.errors == ()
    assert result.opencv_reported_frame_count == 3
    assert result.opencv_readable_frame_count == 3
    assert result.opencv_reported_size_wh == (64, 48)
    assert result.opencv_reported_fps == pytest.approx(10.0)
    assert capture.released is True
    assert [event.completed_units for event in progress] == [0, 3]
    assert all(event.total_units == 3 for event in progress)
    assert all(event.is_indeterminate is False for event in progress)


def test_validation_cancellation_releases_capture_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture(readable_count=5, reported_count=5)
    _install_capture(monkeypatch, capture)
    progress = []

    with pytest.raises(PreprocessCancelledError, match="validation"):
        validate_prepared_video(
            _prepared_path(tmp_path),
            expected_frame_count=5,
            expected_size_wh=(64, 48),
            progress_callback=progress.append,
            cancellation_requested=lambda: capture._index >= 2,
        )

    assert capture.released is True
    assert progress[-1].completed_units != 5


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


def test_disabled_or_empty_static_mask_leaves_frames_unchanged() -> None:
    frame = np.full((5, 6, 3), 100, dtype=np.uint8)

    disabled = apply_static_mask_to_frame(
        frame,
        MaskConfig(
            enabled=False,
            shapes=[{"type": "rectangle", "x": 1, "y": 1, "width": 2, "height": 2}],
        ),
    )
    empty = apply_static_mask_to_frame(frame, MaskConfig(enabled=True, shapes=[]))

    np.testing.assert_array_equal(disabled, frame)
    np.testing.assert_array_equal(empty, frame)


def test_rectangle_static_mask_blacks_exact_prepared_pixel_region() -> None:
    frame = np.full((5, 6, 3), 100, dtype=np.uint8)
    mask = MaskConfig(
        enabled=True,
        shapes=[{"type": "rectangle", "x": 1, "y": 2, "width": 3, "height": 2}],
    )

    masked = apply_static_mask_to_frame(frame, mask)

    assert np.all(masked[2:4, 1:4] == 0)
    assert np.all(masked[:2] == 100)
    assert np.all(masked[4] == 100)
    assert np.all(masked[2:4, :1] == 100)
    assert np.all(masked[2:4, 4:] == 100)


def test_polygon_and_overlapping_static_masks_are_black() -> None:
    mask = MaskConfig(
        enabled=True,
        shapes=[
            {"type": "polygon", "vertices": [(1, 1), (5, 1), (1, 5)]},
            {"type": "rectangle", "x": 2, "y": 2, "width": 3, "height": 2},
        ],
    )
    frame = np.full((7, 7), 200, dtype=np.uint8)

    raster = rasterize_static_mask(mask, (7, 7))
    masked = apply_static_mask_to_frame(frame, mask)

    assert raster[2, 2] == 255
    assert raster[3, 4] == 255
    assert masked[2, 2] == 0
    assert masked[3, 4] == 0
    assert masked[6, 6] == 200


def test_static_mask_applies_to_grayscale_and_bgr_without_shape_change() -> None:
    mask = MaskConfig(
        enabled=True,
        shapes=[{"type": "rectangle", "x": 0, "y": 0, "width": 2, "height": 2}],
    )
    gray = np.full((4, 5), 120, dtype=np.uint8)
    bgr = np.full((4, 5, 3), 120, dtype=np.uint8)

    masked_gray = apply_static_mask_to_frame(gray, mask)
    masked_bgr = apply_static_mask_to_frame(bgr, mask)

    assert masked_gray.shape == gray.shape
    assert masked_bgr.shape == bgr.shape
    assert np.all(masked_gray[:2, :2] == 0)
    assert np.all(masked_bgr[:2, :2, :] == 0)
    assert np.all(masked_gray[2:, 2:] == 120)
    assert np.all(masked_bgr[2:, 2:, :] == 120)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (
            {
                "enabled": True,
                "shapes": [{"type": "rectangle", "x": 5, "y": 0, "width": 2, "height": 2}],
            },
            "outside",
        ),
        (
            {
                "enabled": True,
                "shapes": [{"type": "polygon", "vertices": [(0, 0), (1, 1)]}],
            },
            "at least three",
        ),
        (
            {
                "enabled": True,
                "shapes": [{"type": "polygon", "vertices": [(0, 0), (1, 1), (2, 2)]}],
            },
            "nonzero area",
        ),
        (
            {
                "enabled": True,
                "shapes": [
                    {"type": "polygon", "vertices": [(0, 0), (0, 1), (1, 0), (0, 2)]}
                ],
            },
            "simple",
        ),
        (
            {
                "enabled": True,
                "shapes": [{"type": "polygon", "vertices": [(0, 0), (6, 1), (0, 3)]}],
            },
            "outside",
        ),
        (
            {
                "enabled": True,
                "shapes": [
                    {"type": "rectangle", "x": 0.5, "y": 0, "width": 2, "height": 2}
                ],
            },
            "integer",
        ),
    ],
)
def test_invalid_static_mask_geometry_fails_clearly(mask: dict, match: str) -> None:
    with pytest.raises((ValueError, VideoPreparationError), match=match):
        config = MaskConfig.model_validate(mask)
        rasterize_static_mask(config, (6, 6))


def test_stage_b_reports_monotonic_exact_frame_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate = tmp_path / "intermediate.mp4"
    intermediate.write_bytes(b"intermediate")
    capture = _FakeCapture(readable_count=3, reported_count=3)
    writers: list[_FakeWriter] = []
    monkeypatch.setattr(video_prepare.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        video_prepare.cv2,
        "VideoWriter",
        lambda path, *_args: writers.append(_FakeWriter(path)) or writers[-1],
    )
    monkeypatch.setattr(video_prepare.cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr(
        video_prepare,
        "validate_prepared_video",
        lambda *_args, **_kwargs: PreparedVideoValidationResult(
            is_valid=True,
            errors=(),
            opencv_reported_frame_count=3,
            opencv_readable_frame_count=3,
            opencv_reported_size_wh=(64, 48),
            opencv_reported_fps=10.0,
            expected_frame_count=3,
            expected_size_wh=(64, 48),
        ),
    )
    progress = []

    result = reencode_intermediate_with_opencv(
        intermediate_video_path=intermediate,
        prepared_video_path=tmp_path / "prepared.mp4",
        output_fps=10.0,
        output_size_wh=(64, 48),
        config=PreprocessConfig(),
        expected_frame_count=3,
        progress_callback=progress.append,
    )

    completed = [event.completed_units for event in progress]
    assert completed == sorted(completed)
    assert completed == [0, 3]
    assert all(event.total_units == 3 for event in progress)
    assert result.frames_written == 3


def test_stage_b_static_mask_writes_black_pixels_without_changing_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate = tmp_path / "intermediate.mp4"
    intermediate.write_bytes(b"intermediate")
    frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    capture = _FakeCapture(readable_count=2, reported_count=2, frames=[frame, frame.copy()])
    writers: list[_FakeWriter] = []
    monkeypatch.setattr(video_prepare.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        video_prepare.cv2,
        "VideoWriter",
        lambda path, *_args: writers.append(_FakeWriter(path)) or writers[-1],
    )
    monkeypatch.setattr(video_prepare.cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr(
        video_prepare,
        "validate_prepared_video",
        lambda *_args, **_kwargs: PreparedVideoValidationResult(
            is_valid=True,
            errors=(),
            opencv_reported_frame_count=2,
            opencv_readable_frame_count=2,
            opencv_reported_size_wh=(64, 48),
            opencv_reported_fps=10.0,
            expected_frame_count=2,
            expected_size_wh=(64, 48),
        ),
    )
    config = PreprocessConfig(
        mask=MaskConfig(
            enabled=True,
            shapes=[{"type": "rectangle", "x": 2, "y": 3, "width": 4, "height": 5}],
        )
    )

    result = reencode_intermediate_with_opencv(
        intermediate_video_path=intermediate,
        prepared_video_path=tmp_path / "prepared.mp4",
        output_fps=10.0,
        output_size_wh=(64, 48),
        config=config,
        expected_frame_count=2,
    )

    assert result.frames_written == 2
    assert len(writers) == 1
    assert len(writers[0].frames) == 2
    assert writers[0].frames[0].shape == (48, 64, 3)
    assert np.all(writers[0].frames[0][3:8, 2:6] == 0)
    assert np.all(writers[0].frames[0][:3] == 100)


def test_stage_b_cancellation_preserves_existing_prepared_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intermediate = tmp_path / "intermediate.mp4"
    intermediate.write_bytes(b"intermediate")
    prepared = tmp_path / "prepared.mp4"
    prepared.write_bytes(b"previous validated video")
    capture = _FakeCapture(readable_count=5, reported_count=5)
    monkeypatch.setattr(video_prepare.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        video_prepare.cv2,
        "VideoWriter",
        lambda path, *_args: _FakeWriter(path),
    )
    monkeypatch.setattr(video_prepare.cv2, "VideoWriter_fourcc", lambda *_args: 0)

    with pytest.raises(PreprocessCancelledError, match="Stage B"):
        reencode_intermediate_with_opencv(
            intermediate_video_path=intermediate,
            prepared_video_path=prepared,
            output_fps=10.0,
            output_size_wh=(64, 48),
            config=PreprocessConfig(),
            expected_frame_count=5,
            cancellation_requested=lambda: capture._index >= 2,
        )

    assert capture.released is True
    assert prepared.read_bytes() == b"previous validated video"
    assert list(tmp_path.glob(".prepared.stage-b-*.mp4")) == []
