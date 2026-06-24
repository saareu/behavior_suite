from pathlib import Path

import numpy as np
import pytest

from preprocess import background as background_module
from preprocess.background import (
    estimate_prepared_background,
    validate_background,
    write_background_png,
)
from preprocess.exceptions import BackgroundGenerationError, PreprocessCancelledError


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        opened: bool = True,
    ) -> None:
        self._frames = frames
        self._opened = opened
        self._index = 0
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _prepared_video_placeholder(tmp_path: Path) -> Path:
    path = tmp_path / "prepared.mp4"
    path.write_bytes(b"prepared video placeholder")
    return path


def _install_capture(
    monkeypatch: pytest.MonkeyPatch,
    capture: _FakeCapture,
) -> None:
    monkeypatch.setattr(
        background_module.cv2,
        "VideoCapture",
        lambda _path: capture,
    )


def _color_frame(value: int, size_wh: tuple[int, int] = (5, 4)) -> np.ndarray:
    return np.full((size_wh[1], size_wh[0], 3), value, dtype=np.uint8)


def test_valid_color_median_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture(
        [_color_frame(10), _color_frame(50), _color_frame(90)]
    )
    _install_capture(monkeypatch, capture)

    result = estimate_prepared_background(
        _prepared_video_placeholder(tmp_path),
        sample_every_n=1,
        max_samples=10,
    )

    np.testing.assert_array_equal(result.background, _color_frame(50))
    assert result.background.dtype == np.uint8
    assert result.background.shape == (4, 5, 3)
    assert result.output_dtype == "uint8"
    assert result.source_size_wh == (5, 4)
    assert result.method == "median"
    assert capture.released is True


def test_valid_grayscale_median_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        np.full((4, 5), value, dtype=np.uint8)
        for value in (20, 60, 100)
    ]
    _install_capture(monkeypatch, _FakeCapture(frames))

    result = estimate_prepared_background(
        _prepared_video_placeholder(tmp_path),
        sample_every_n=1,
        max_samples=3,
    )

    np.testing.assert_array_equal(result.background, frames[1])
    assert result.background.ndim == 2
    assert result.background.dtype == np.uint8


def test_sampled_indices_follow_every_n_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [_color_frame(index * 10) for index in range(7)]
    _install_capture(monkeypatch, _FakeCapture(frames))

    progress = []
    result = estimate_prepared_background(
        _prepared_video_placeholder(tmp_path),
        sample_every_n=2,
        max_samples=10,
        expected_frame_count=7,
        progress_callback=progress.append,
    )

    assert result.sampled_frame_indices == (0, 2, 4, 6)
    assert result.sampled_frame_count == 4
    np.testing.assert_array_equal(result.background, _color_frame(30))
    sample_events = [
        event for event in progress if event.completed_units is not None
    ]
    assert [event.completed_units for event in sample_events] == [0, 4, 4]
    assert all(event.total_units == 4 for event in sample_events)
    assert any(event.message == "Calculating median background" for event in progress)


def test_background_cancellation_releases_capture_without_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture([_color_frame(index) for index in range(8)])
    _install_capture(monkeypatch, capture)

    with pytest.raises(PreprocessCancelledError, match="background"):
        estimate_prepared_background(
            _prepared_video_placeholder(tmp_path),
            sample_every_n=1,
            max_samples=8,
            expected_frame_count=8,
            cancellation_requested=lambda: capture.read_count >= 2,
        )

    assert capture.released is True


def test_sample_limit_stops_sequential_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture([_color_frame(index * 10) for index in range(10)])
    _install_capture(monkeypatch, capture)

    result = estimate_prepared_background(
        _prepared_video_placeholder(tmp_path),
        sample_every_n=2,
        max_samples=2,
    )

    assert result.sampled_frame_indices == (0, 2)
    assert result.sampled_frame_count == 2
    assert capture.read_count == 3


@pytest.mark.parametrize(
    ("argument", "value", "match"),
    [
        ("sample_every_n", 0, "sample_every_n"),
        ("sample_every_n", -1, "sample_every_n"),
        ("max_samples", 0, "max_samples"),
        ("max_samples", -1, "max_samples"),
    ],
)
def test_invalid_sampling_arguments_fail(
    tmp_path: Path,
    argument: str,
    value: int,
    match: str,
) -> None:
    arguments = {
        "prepared_video_path": tmp_path / "unused.mp4",
        "sample_every_n": 1,
        "max_samples": 1,
    }
    arguments[argument] = value

    with pytest.raises(BackgroundGenerationError, match=match):
        estimate_prepared_background(**arguments)


def test_unsupported_method_fails(tmp_path: Path) -> None:
    with pytest.raises(BackgroundGenerationError, match="Unsupported"):
        estimate_prepared_background(
            tmp_path / "unused.mp4",
            sample_every_n=1,
            max_samples=1,
            method="mean",
        )


def test_missing_prepared_video_fails(tmp_path: Path) -> None:
    with pytest.raises(BackgroundGenerationError, match="does not exist"):
        estimate_prepared_background(
            tmp_path / "missing.mp4",
            sample_every_n=1,
            max_samples=1,
        )


def test_unreadable_prepared_video_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture([], opened=False)
    _install_capture(monkeypatch, capture)

    with pytest.raises(BackgroundGenerationError, match="could not open"):
        estimate_prepared_background(
            _prepared_video_placeholder(tmp_path),
            sample_every_n=1,
            max_samples=1,
        )

    assert capture.released is True


def test_no_readable_frames_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _FakeCapture([])
    _install_capture(monkeypatch, capture)

    with pytest.raises(BackgroundGenerationError, match="no readable frames"):
        estimate_prepared_background(
            _prepared_video_placeholder(tmp_path),
            sample_every_n=1,
            max_samples=1,
        )

    assert capture.released is True


def test_inconsistent_sampled_dimensions_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _color_frame(10, (5, 4)),
        _color_frame(20, (5, 4)),
        _color_frame(30, (6, 4)),
    ]
    _install_capture(monkeypatch, _FakeCapture(frames))

    with pytest.raises(BackgroundGenerationError, match="does not match"):
        estimate_prepared_background(
            _prepared_video_placeholder(tmp_path),
            sample_every_n=2,
            max_samples=2,
        )


def test_background_validation_rejects_wrong_dimensions() -> None:
    with pytest.raises(BackgroundGenerationError, match="does not match"):
        validate_background(
            np.zeros((4, 5), dtype=np.uint8),
            expected_size_wh=(6, 4),
        )


@pytest.mark.parametrize(
    "invalid_background",
    [
        np.zeros((4, 5), dtype=np.float32),
        np.zeros(5, dtype=np.uint8),
        np.zeros((4, 5, 2), dtype=np.uint8),
    ],
)
def test_background_validation_rejects_invalid_dtype_or_shape(
    invalid_background: np.ndarray,
) -> None:
    with pytest.raises(BackgroundGenerationError):
        validate_background(invalid_background, expected_size_wh=(5, 4))


def test_atomic_png_write_preserves_existing_destination_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "cropped_background.png"
    existing_content = b"existing validated background"
    destination.write_bytes(existing_content)
    monkeypatch.setattr(background_module.cv2, "imwrite", lambda *_args: False)

    with pytest.raises(BackgroundGenerationError, match="could not write"):
        write_background_png(_color_frame(50), destination)

    assert destination.read_bytes() == existing_content
    assert list(tmp_path.glob(".cropped_background.background-*.png")) == []


def test_png_reopen_failure_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "cropped_background.png"
    existing_content = b"existing validated background"
    destination.write_bytes(existing_content)
    monkeypatch.setattr(background_module.cv2, "imread", lambda *_args: None)

    with pytest.raises(BackgroundGenerationError, match="could not reopen"):
        write_background_png(_color_frame(50), destination)

    assert destination.read_bytes() == existing_content
    assert list(tmp_path.glob(".cropped_background.background-*.png")) == []
