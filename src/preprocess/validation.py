"""Strict OpenCV validation for final prepared videos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, ConfigDict

from preprocess.exceptions import VideoValidationError


class PreparedVideoValidationResult(BaseModel):
    """Measured final-video properties and strict validation outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_valid: bool
    errors: tuple[str, ...]
    opencv_reported_frame_count: int
    opencv_readable_frame_count: int
    opencv_reported_size_wh: tuple[int, int]
    opencv_reported_fps: float | None
    expected_frame_count: int
    expected_size_wh: tuple[int, int]


def _invalid_result(
    errors: list[str],
    *,
    reported_frame_count: int,
    readable_frame_count: int,
    reported_size_wh: tuple[int, int],
    reported_fps: float | None,
    expected_frame_count: int,
    expected_size_wh: tuple[int, int],
) -> VideoValidationError:
    result = PreparedVideoValidationResult(
        is_valid=False,
        errors=tuple(errors),
        opencv_reported_frame_count=reported_frame_count,
        opencv_readable_frame_count=readable_frame_count,
        opencv_reported_size_wh=reported_size_wh,
        opencv_reported_fps=reported_fps,
        expected_frame_count=expected_frame_count,
        expected_size_wh=expected_size_wh,
    )
    return VideoValidationError(
        "Prepared video validation failed: " + "; ".join(errors),
        result=result,
    )


def _validate_expected_inputs(
    expected_frame_count: int,
    expected_size_wh: tuple[int, int],
) -> tuple[int, tuple[int, int], list[str]]:
    errors: list[str] = []
    if (
        isinstance(expected_frame_count, bool)
        or not isinstance(expected_frame_count, int)
        or expected_frame_count <= 0
    ):
        errors.append("expected_frame_count must be a positive integer")
        normalized_frame_count = 0
    else:
        normalized_frame_count = expected_frame_count

    try:
        width, height = expected_size_wh
    except (TypeError, ValueError):
        errors.append("expected_size_wh must contain width and height")
        return normalized_frame_count, (0, 0), errors
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (width, height)
    ):
        errors.append("expected width and height must be integers")
        return normalized_frame_count, (0, 0), errors
    normalized_size = (width, height)
    if width <= 0 or height <= 0:
        errors.append("expected width and height must be positive")
    if width % 2 or height % 2:
        errors.append("expected width and height must be even")
    return normalized_frame_count, normalized_size, errors


def _positive_reported_fps(value: Any) -> float | None:
    try:
        fps = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return fps if math.isfinite(fps) and fps > 0 else None


def _positive_reported_integer(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(numeric) or numeric <= 0:
        return 0
    return int(numeric)


def validate_prepared_video(
    prepared_video_path: Path,
    expected_frame_count: int,
    expected_size_wh: tuple[int, int],
) -> PreparedVideoValidationResult:
    """Validate final prepared-video count, geometry, and FPS using OpenCV.

    Frames are counted only by sequential capture.read() calls until the first
    false result. Any invalid input, unreadable file, reported/readable
    mismatch, expected-count mismatch, geometry mismatch, odd dimension, or
    invalid FPS is a hard failure.

    Raises:
        VideoValidationError: If any strict validation gate fails. The
            exception's result attribute contains the populated invalid
            PreparedVideoValidationResult.
    """

    normalized_count, normalized_size, errors = _validate_expected_inputs(
        expected_frame_count,
        expected_size_wh,
    )
    if errors:
        raise _invalid_result(
            errors,
            reported_frame_count=0,
            readable_frame_count=0,
            reported_size_wh=(0, 0),
            reported_fps=None,
            expected_frame_count=normalized_count,
            expected_size_wh=normalized_size,
        )

    video_path = Path(prepared_video_path).expanduser()
    if not video_path.is_file():
        raise _invalid_result(
            [f"Prepared video does not exist: {video_path}"],
            reported_frame_count=0,
            readable_frame_count=0,
            reported_size_wh=(0, 0),
            reported_fps=None,
            expected_frame_count=normalized_count,
            expected_size_wh=normalized_size,
        )
    video_path = video_path.resolve()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise _invalid_result(
            [f"OpenCV could not open prepared video: {video_path}"],
            reported_frame_count=0,
            readable_frame_count=0,
            reported_size_wh=(0, 0),
            reported_fps=None,
            expected_frame_count=normalized_count,
            expected_size_wh=normalized_size,
        )

    reported_count = _positive_reported_integer(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_width = _positive_reported_integer(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = _positive_reported_integer(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_size = (reported_width, reported_height)
    reported_fps = _positive_reported_fps(capture.get(cv2.CAP_PROP_FPS))
    readable_count = 0
    decode_error: str | None = None
    try:
        while True:
            try:
                success, frame = capture.read()
            except cv2.error as exc:
                decode_error = f"OpenCV decode failed: {exc}"
                break
            if not success:
                break
            if frame is None:
                decode_error = "OpenCV reported a successful decode without a frame"
                break
            readable_count += 1
    finally:
        capture.release()

    if reported_count <= 0:
        errors.append("OpenCV-reported frame count must be available and positive")
    if reported_width <= 0 or reported_height <= 0:
        errors.append("OpenCV-reported width and height must be positive")
    elif reported_width % 2 or reported_height % 2:
        errors.append("OpenCV-reported width and height must be even")
    if reported_size != normalized_size:
        errors.append(
            f"OpenCV-reported size {reported_size} does not match expected {normalized_size}"
        )
    if reported_fps is None:
        errors.append("OpenCV-reported FPS must be finite and positive")
    if decode_error is not None:
        errors.append(decode_error)
    if readable_count <= 0:
        errors.append("OpenCV could not sequentially decode any prepared frames")
    if reported_count != readable_count:
        errors.append(
            "OpenCV-reported frame count "
            f"{reported_count} does not match sequential readable count {readable_count}"
        )
    if readable_count != normalized_count:
        errors.append(
            f"Sequential readable count {readable_count} does not match expected "
            f"{normalized_count}"
        )

    if errors:
        raise _invalid_result(
            errors,
            reported_frame_count=reported_count,
            readable_frame_count=readable_count,
            reported_size_wh=reported_size,
            reported_fps=reported_fps,
            expected_frame_count=normalized_count,
            expected_size_wh=normalized_size,
        )
    return PreparedVideoValidationResult(
        is_valid=True,
        errors=(),
        opencv_reported_frame_count=reported_count,
        opencv_readable_frame_count=readable_count,
        opencv_reported_size_wh=reported_size,
        opencv_reported_fps=reported_fps,
        expected_frame_count=normalized_count,
        expected_size_wh=normalized_size,
    )
