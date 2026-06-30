"""Static prepared-coordinate exclusion mask validation and rasterization."""

from __future__ import annotations

import cv2
import numpy as np

from preprocess.config import MaskConfig, MaskPolygonConfig, MaskRectangleConfig
from preprocess.exceptions import VideoPreparationError

__all__ = (
    "apply_static_mask_to_frame",
    "rasterize_static_mask",
    "serialize_static_mask",
    "validate_static_mask",
)


def validate_static_mask(
    mask: MaskConfig,
    prepared_size_wh: tuple[int, int],
) -> None:
    """Validate enabled mask geometry against final prepared-video dimensions.

    A disabled mask is a backward-compatible no-op. Its shape payload is still
    parsed by :class:`MaskConfig`, but prepared-size bounds are ignored because
    no pixels will be masked.
    """

    width, height = _validate_prepared_size(prepared_size_wh)
    if not mask.enabled:
        return
    for index, shape in enumerate(mask.shapes):
        if isinstance(shape, MaskRectangleConfig):
            if shape.x < 0 or shape.y < 0:
                raise VideoPreparationError(
                    f"Mask rectangle {index} origin must be inside prepared dimensions."
                )
            if shape.x + shape.width > width or shape.y + shape.height > height:
                raise VideoPreparationError(
                    f"Mask rectangle {index} extends outside prepared dimensions."
                )
        elif isinstance(shape, MaskPolygonConfig):
            for vertex_index, (x, y) in enumerate(shape.vertices):
                if x < 0 or y < 0 or x >= width or y >= height:
                    raise VideoPreparationError(
                        f"Mask polygon {index} vertex {vertex_index} lies outside "
                        "prepared dimensions."
                    )
        else:
            raise VideoPreparationError(f"Unsupported mask shape at index {index}.")


def rasterize_static_mask(
    mask: MaskConfig,
    prepared_size_wh: tuple[int, int],
) -> np.ndarray:
    """Return a deterministic uint8 mask image for prepared-pixel shapes."""

    width, height = _validate_prepared_size(prepared_size_wh)
    raster = np.zeros((height, width), dtype=np.uint8)
    if not mask.enabled or not mask.shapes:
        return raster
    validate_static_mask(mask, (width, height))
    for shape in mask.shapes:
        if isinstance(shape, MaskRectangleConfig):
            raster[
                shape.y : shape.y + shape.height,
                shape.x : shape.x + shape.width,
            ] = 255
        elif isinstance(shape, MaskPolygonConfig):
            points = np.asarray(shape.vertices, dtype=np.int32)
            cv2.fillPoly(raster, [points], color=255)
    return raster


def apply_static_mask_to_frame(frame: np.ndarray, mask: MaskConfig) -> np.ndarray:
    """Return one frame with enabled static mask regions filled black.

    The input may be grayscale or color. The output shape, dtype, frame count,
    timing, and coordinate transforms are unchanged by this visual operation.
    """

    array = np.asarray(frame)
    if array.ndim not in {2, 3} or array.size == 0:
        raise VideoPreparationError("Static mask input frame must be a non-empty image.")
    height, width = array.shape[:2]
    if not mask.enabled or not mask.shapes:
        return np.array(array, copy=True)
    validate_static_mask(mask, (width, height))
    raster = rasterize_static_mask(mask, (width, height))
    masked = np.array(array, copy=True)
    if masked.ndim == 2:
        masked[raster > 0] = 0
    else:
        masked[raster > 0, :] = 0
    return masked


def serialize_static_mask(mask: MaskConfig) -> dict[str, object]:
    """Return JSON-safe static mask metadata with shape order preserved."""

    return {
        "enabled": mask.enabled,
        "coordinate_space": mask.coordinate_space,
        "fill_value": mask.fill_value,
        "shapes": [shape.model_dump(mode="json") for shape in mask.shapes],
    }


def _validate_prepared_size(prepared_size_wh: tuple[int, int]) -> tuple[int, int]:
    try:
        width, height = prepared_size_wh
    except (TypeError, ValueError) as exc:
        raise VideoPreparationError(
            "prepared_size_wh must contain width and height."
        ) from exc
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height)):
        raise VideoPreparationError("Prepared mask dimensions must be integer pixels.")
    if width <= 0 or height <= 0:
        raise VideoPreparationError("Prepared mask dimensions must be positive.")
    return width, height
