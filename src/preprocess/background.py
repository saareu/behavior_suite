"""Prepared-video median background estimation and safe PNG persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from preprocess.exceptions import BackgroundGenerationError

_SUPPORTED_METHOD = "median"
_PNG_CHANNEL_COUNTS = {1, 3, 4}


class BackgroundEstimateResult(BaseModel):
    """A background image and deterministic prepared-frame sampling details."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    background: np.ndarray = Field(repr=False)
    sampled_frame_indices: tuple[int, ...]
    sampled_frame_count: int = Field(gt=0)
    source_video_path: Path
    source_size_wh: tuple[int, int]
    method: str
    output_dtype: str


def _positive_sampling_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackgroundGenerationError(f"{name} must be a positive integer.")
    return value


def _validate_method(method: str) -> str:
    if method != _SUPPORTED_METHOD:
        raise BackgroundGenerationError(
            f"Unsupported background method '{method}'. Only median is supported."
        )
    return method


def _frame_shape(frame: np.ndarray, *, context: str) -> tuple[int, ...]:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise BackgroundGenerationError(f"{context} is not a non-empty NumPy array.")
    if frame.ndim == 2:
        height, width = frame.shape
    elif frame.ndim == 3 and frame.shape[2] in _PNG_CHANNEL_COUNTS:
        height, width = frame.shape[:2]
    else:
        raise BackgroundGenerationError(
            f"{context} must be grayscale or have 1, 3, or 4 color channels."
        )
    if width <= 0 or height <= 0:
        raise BackgroundGenerationError(f"{context} dimensions must be positive.")
    return tuple(int(dimension) for dimension in frame.shape)


def _size_wh(frame: np.ndarray) -> tuple[int, int]:
    return int(frame.shape[1]), int(frame.shape[0])


def validate_background(
    background: np.ndarray,
    expected_size_wh: tuple[int, int],
) -> None:
    """Validate background dtype, shape, and prepared-coordinate dimensions.

    Raises:
        BackgroundGenerationError: If the background is not a non-empty uint8
            grayscale/color array or its dimensions differ from the expected
            prepared-video dimensions.
    """

    try:
        expected_width, expected_height = expected_size_wh
    except (TypeError, ValueError) as exc:
        raise BackgroundGenerationError(
            "expected_size_wh must contain width and height."
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (expected_width, expected_height)
    ):
        raise BackgroundGenerationError(
            "Expected background width and height must be integers."
        )
    if expected_width <= 0 or expected_height <= 0:
        raise BackgroundGenerationError(
            "Expected background width and height must be positive."
        )

    _frame_shape(background, context="Background")
    if background.dtype != np.uint8:
        raise BackgroundGenerationError("Background dtype must be uint8.")
    actual_size = _size_wh(background)
    expected_size = (expected_width, expected_height)
    if actual_size != expected_size:
        raise BackgroundGenerationError(
            f"Background size {actual_size} does not match expected {expected_size}."
        )


def estimate_prepared_background(
    prepared_video_path: Path,
    sample_every_n: int,
    max_samples: int,
    method: str = "median",
) -> BackgroundEstimateResult:
    """Estimate a pixelwise median from sequential prepared-video samples.

    The first readable frame has index zero. Frames are decoded sequentially
    and accepted when their decode index is divisible by sample_every_n.
    Decoding stops at end of stream or immediately after max_samples accepted
    frames. Reported frame counts and random seeking are never used.

    Raises:
        BackgroundGenerationError: If arguments, source readability, sampled
            frame geometry, or median calculation is invalid.
    """

    sample_step = _positive_sampling_integer("sample_every_n", sample_every_n)
    sample_limit = _positive_sampling_integer("max_samples", max_samples)
    resolved_method = _validate_method(method)
    source_path = Path(prepared_video_path).expanduser()
    if not source_path.is_file():
        raise BackgroundGenerationError(
            f"Prepared video does not exist: {source_path}"
        )
    source_path = source_path.resolve()

    try:
        capture = cv2.VideoCapture(str(source_path))
    except cv2.error as exc:
        raise BackgroundGenerationError(
            f"OpenCV could not initialize prepared-video decoding: {exc}"
        ) from exc
    if not capture.isOpened():
        capture.release()
        raise BackgroundGenerationError(
            f"OpenCV could not open prepared video: {source_path}"
        )

    samples: list[np.ndarray] = []
    sampled_indices: list[int] = []
    decoded_index = 0
    sample_shape: tuple[int, ...] | None = None
    try:
        while len(samples) < sample_limit:
            try:
                success, frame = capture.read()
            except cv2.error as exc:
                raise BackgroundGenerationError(
                    f"OpenCV failed while decoding prepared video: {exc}"
                ) from exc
            if not success:
                break
            if frame is None:
                raise BackgroundGenerationError(
                    "OpenCV reported a successful prepared-video decode without a frame."
                )
            if decoded_index % sample_step == 0:
                current_shape = _frame_shape(
                    frame,
                    context=f"Sampled frame {decoded_index}",
                )
                if sample_shape is None:
                    sample_shape = current_shape
                elif current_shape != sample_shape:
                    raise BackgroundGenerationError(
                        f"Sampled frame {decoded_index} shape {current_shape} does not "
                        f"match first sampled frame shape {sample_shape}."
                    )
                samples.append(np.array(frame, copy=True))
                sampled_indices.append(decoded_index)
            decoded_index += 1
    finally:
        capture.release()

    if not samples or sample_shape is None:
        raise BackgroundGenerationError(
            "Prepared video contains no readable frames to sample."
        )
    try:
        stacked = np.stack(samples, axis=0)
        background = np.median(stacked, axis=0).astype(np.uint8)
    except (MemoryError, TypeError, ValueError) as exc:
        raise BackgroundGenerationError(
            f"Could not calculate prepared-video median background: {exc}"
        ) from exc
    source_size = _size_wh(samples[0])
    validate_background(background, source_size)
    background = np.array(background, copy=True)
    background.setflags(write=False)
    return BackgroundEstimateResult(
        background=background,
        sampled_frame_indices=tuple(sampled_indices),
        sampled_frame_count=len(samples),
        source_video_path=source_path,
        source_size_wh=source_size,
        method=resolved_method,
        output_dtype=str(background.dtype),
    )


def _create_temporary_png_path(destination: Path) -> Path:
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.background-",
            suffix=".png",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name).resolve()
        temporary_path.unlink()
    except OSError as exc:
        raise BackgroundGenerationError(
            f"Could not create temporary background PNG path: {exc}"
        ) from exc
    return temporary_path


def _remove_temporary_png(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise BackgroundGenerationError(
            f"Could not remove failed temporary background PNG {path}: {exc}"
        ) from exc


def write_background_png(
    background: np.ndarray,
    output_path: Path,
) -> Path:
    """Atomically write and reopen-validate a normal background PNG.

    The image is written to a same-directory temporary path. Any existing
    destination remains unchanged unless OpenCV can reopen the temporary PNG
    with the expected dimensions and uint8 grayscale/color representation.

    Raises:
        BackgroundGenerationError: If validation, writing, reopening,
            temporary cleanup, or atomic replacement fails.
    """

    if not isinstance(background, np.ndarray) or background.ndim < 2:
        raise BackgroundGenerationError(
            "Background must be a grayscale or color NumPy array."
        )
    expected_size = _size_wh(background)
    validate_background(background, expected_size)

    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise BackgroundGenerationError(
            "Background output path must use the .png suffix."
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
    except OSError as exc:
        raise BackgroundGenerationError(
            f"Could not create background output directory: {exc}"
        ) from exc

    temporary_path = _create_temporary_png_path(destination)
    try:
        try:
            written = cv2.imwrite(str(temporary_path), background)
        except cv2.error as exc:
            raise BackgroundGenerationError(
                f"OpenCV could not write temporary background PNG: {exc}"
            ) from exc
        if not written:
            raise BackgroundGenerationError(
                "OpenCV could not write temporary background PNG."
            )
        try:
            reopened = cv2.imread(str(temporary_path), cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise BackgroundGenerationError(
                f"OpenCV could not reopen temporary background PNG: {exc}"
            ) from exc
        if reopened is None:
            raise BackgroundGenerationError(
                "OpenCV could not reopen temporary background PNG."
            )
        validate_background(reopened, expected_size)
        try:
            temporary_path.replace(destination)
        except OSError as exc:
            raise BackgroundGenerationError(
                f"Could not atomically replace background PNG: {exc}"
            ) from exc
    except BackgroundGenerationError:
        _remove_temporary_png(temporary_path)
        raise
    return destination
