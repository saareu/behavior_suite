from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess.background import (
    estimate_prepared_background,
    write_background_png,
)
from preprocess.validation import validate_prepared_video

_SIZE_WH = (64, 48)
_FPS = 10.0


def _write_prepared_style_video(
    path: Path,
    frame_values: tuple[int, ...],
    *,
    is_color: bool,
) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _FPS,
        _SIZE_WH,
        is_color,
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an mp4v integration fixture.")
    try:
        for value in frame_values:
            shape = (
                (_SIZE_WH[1], _SIZE_WH[0], 3)
                if is_color
                else (_SIZE_WH[1], _SIZE_WH[0])
            )
            writer.write(np.full(shape, value, dtype=np.uint8))
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(path))
    readable, _frame = capture.read()
    capture.release()
    if not readable:
        pytest.skip(
            "This OpenCV backend cannot decode its generated grayscale/color mp4v fixture."
        )
    return path


@pytest.mark.integration
def test_color_prepared_video_generates_expected_median_png(tmp_path: Path) -> None:
    frame_values = (20, 60, 100, 140, 180)
    prepared_path = _write_prepared_style_video(
        tmp_path / "prepared.mp4",
        frame_values,
        is_color=True,
    )
    validate_prepared_video(
        prepared_path,
        expected_frame_count=len(frame_values),
        expected_size_wh=_SIZE_WH,
    )

    result = estimate_prepared_background(
        prepared_path,
        sample_every_n=2,
        max_samples=3,
    )
    output_path = write_background_png(
        result.background,
        tmp_path / "cropped_background.png",
    )

    reopened = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert output_path.is_file()
    assert reopened is not None
    assert result.sampled_frame_indices == (0, 2, 4)
    assert result.sampled_frame_count == 3
    assert result.source_size_wh == _SIZE_WH
    assert result.background.shape == (_SIZE_WH[1], _SIZE_WH[0], 3)
    assert result.background.dtype == np.uint8
    assert float(result.background.mean()) == pytest.approx(100.0, abs=8.0)
    assert (reopened.shape[1], reopened.shape[0]) == _SIZE_WH


@pytest.mark.integration
def test_grayscale_prepared_video_behaves_consistently_with_backend(
    tmp_path: Path,
) -> None:
    frame_values = (30, 90, 150)
    prepared_path = _write_prepared_style_video(
        tmp_path / "prepared_grayscale.mp4",
        frame_values,
        is_color=False,
    )
    validate_prepared_video(
        prepared_path,
        expected_frame_count=len(frame_values),
        expected_size_wh=_SIZE_WH,
    )

    result = estimate_prepared_background(
        prepared_path,
        sample_every_n=1,
        max_samples=3,
    )
    output_path = write_background_png(
        result.background,
        tmp_path / "grayscale_background.png",
    )

    reopened = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert reopened is not None
    assert result.sampled_frame_indices == (0, 1, 2)
    assert result.sampled_frame_count == 3
    assert result.background.ndim in {2, 3}
    if result.background.ndim == 3:
        assert result.background.shape[2] == 3
    assert result.background.dtype == np.uint8
    assert float(result.background.mean()) == pytest.approx(90.0, abs=8.0)
    assert (reopened.shape[1], reopened.shape[0]) == _SIZE_WH
