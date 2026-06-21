"""Pure geometry for resolving detection pre-crops."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from preprocess.config import PreCropConfig
from preprocess.exceptions import PreCropError


class PreCropMode(StrEnum):
    """Supported detection pre-crop modes."""

    NONE = "none"
    VERTICAL_KEEP_LEFT = "vertical_keep_left"
    VERTICAL_KEEP_RIGHT = "vertical_keep_right"
    HORIZONTAL_KEEP_UPPER = "horizontal_keep_upper"
    HORIZONTAL_KEEP_LOWER = "horizontal_keep_lower"
    MANUAL_RECTANGLE = "manual_rectangle"


class PreCropROI(BaseModel):
    """An integer rectangular region in raw-frame coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def to_metadata_dict(self) -> dict[str, int]:
        """Return a JSON-safe x/y/width/height mapping."""

        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


class ResolvedPreCrop(BaseModel):
    """A concrete pre-crop and its raw-frame coordinate context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: PreCropMode
    roi: PreCropROI
    raw_size_wh: tuple[int, int]

    @model_validator(mode="after")
    def validate_roi_within_raw_frame(self) -> ResolvedPreCrop:
        raw_width, raw_height = self.raw_size_wh
        if raw_width <= 0 or raw_height <= 0:
            raise PreCropError("Raw frame width and height must be positive.")
        if self.roi.x + self.roi.width > raw_width:
            raise PreCropError("Resolved pre-crop exceeds the raw-frame width.")
        if self.roi.y + self.roi.height > raw_height:
            raise PreCropError("Resolved pre-crop exceeds the raw-frame height.")
        return self

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe pre-crop metadata including its final transform."""

        return serialize_pre_crop(self)


def _validate_raw_size(raw_size_wh: tuple[int, int]) -> tuple[int, int]:
    if len(raw_size_wh) != 2:
        raise PreCropError("Raw frame size must contain width and height.")
    raw_width, raw_height = raw_size_wh
    if (
        isinstance(raw_width, bool)
        or isinstance(raw_height, bool)
        or not isinstance(raw_width, int)
        or not isinstance(raw_height, int)
    ):
        raise PreCropError("Raw frame width and height must be integers.")
    if raw_width <= 0 or raw_height <= 0:
        raise PreCropError("Raw frame width and height must be positive.")
    return raw_width, raw_height


def _validate_manual_rectangle(
    rectangle: tuple[int, int, int, int],
    raw_width: int,
    raw_height: int,
) -> PreCropROI:
    if len(rectangle) != 4:
        raise PreCropError("Manual pre-crop rectangle must contain x, y, width, and height.")
    x, y, width, height = rectangle
    if any(isinstance(value, bool) or not isinstance(value, int) for value in rectangle):
        raise PreCropError("Manual pre-crop rectangle values must be integers.")
    if x < 0 or y < 0:
        raise PreCropError("Manual pre-crop x and y must be non-negative.")
    if width <= 0 or height <= 0:
        raise PreCropError("Manual pre-crop width and height must be positive.")
    if x + width > raw_width or y + height > raw_height:
        raise PreCropError("Manual pre-crop rectangle must lie within the raw frame.")
    return PreCropROI(x=x, y=y, width=width, height=height)


def resolve_pre_crop(
    config: PreCropConfig,
    raw_size_wh: tuple[int, int],
) -> ResolvedPreCrop:
    """Resolve an approved pre-crop mode to a concrete raw-frame ROI.

    Boundary coordinates use half-open image geometry. For example,
    vertical_keep_left with boundary_px=100 keeps x coordinates in [0, 100).

    Raises:
        PreCropError: If a boundary or rectangle is outside the raw frame or
            produces an empty ROI.
    """

    raw_width, raw_height = _validate_raw_size(raw_size_wh)
    mode = PreCropMode(config.mode)

    if mode is PreCropMode.NONE:
        roi = PreCropROI(x=0, y=0, width=raw_width, height=raw_height)
    elif mode is PreCropMode.MANUAL_RECTANGLE:
        if config.manual_rectangle is None:
            raise PreCropError("Manual rectangle geometry is required.")
        roi = _validate_manual_rectangle(
            config.manual_rectangle,
            raw_width,
            raw_height,
        )
    else:
        boundary = config.boundary_px
        if boundary is None:
            raise PreCropError("A boundary coordinate is required for this pre-crop mode.")
        if mode is PreCropMode.VERTICAL_KEEP_LEFT:
            if not 0 < boundary <= raw_width:
                raise PreCropError(
                    "vertical_keep_left boundary must be in the range (0, raw_width]."
                )
            roi = PreCropROI(x=0, y=0, width=boundary, height=raw_height)
        elif mode is PreCropMode.VERTICAL_KEEP_RIGHT:
            if not 0 <= boundary < raw_width:
                raise PreCropError(
                    "vertical_keep_right boundary must be in the range [0, raw_width)."
                )
            roi = PreCropROI(
                x=boundary,
                y=0,
                width=raw_width - boundary,
                height=raw_height,
            )
        elif mode is PreCropMode.HORIZONTAL_KEEP_UPPER:
            if not 0 < boundary <= raw_height:
                raise PreCropError(
                    "horizontal_keep_upper boundary must be in the range (0, raw_height]."
                )
            roi = PreCropROI(x=0, y=0, width=raw_width, height=boundary)
        else:
            if not 0 <= boundary < raw_height:
                raise PreCropError(
                    "horizontal_keep_lower boundary must be in the range [0, raw_height)."
                )
            roi = PreCropROI(
                x=0,
                y=boundary,
                width=raw_width,
                height=raw_height - boundary,
            )

    return ResolvedPreCrop(mode=mode, roi=roi, raw_size_wh=(raw_width, raw_height))


def build_pre_crop_translation_homography(
    roi: PreCropROI | ResolvedPreCrop,
) -> np.ndarray:
    """Build the raw-to-pre-crop translation homography for an ROI.

    The returned matrix maps raw coordinates (x, y) to (x - roi.x, y - roi.y).
    """

    concrete_roi = roi.roi if isinstance(roi, ResolvedPreCrop) else roi
    return np.array(
        [
            [1.0, 0.0, -float(concrete_roi.x)],
            [0.0, 1.0, -float(concrete_roi.y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def serialize_pre_crop(pre_crop: ResolvedPreCrop) -> dict[str, Any]:
    """Serialize a resolved pre-crop to JSON-safe metadata."""

    return {
        "enabled": pre_crop.mode is not PreCropMode.NONE,
        "mode": pre_crop.mode.value,
        "roi": pre_crop.roi.to_metadata_dict(),
        "raw_size_wh": [pre_crop.raw_size_wh[0], pre_crop.raw_size_wh[1]],
        "H_raw_to_pre_crop_3x3": build_pre_crop_translation_homography(pre_crop).tolist(),
        "defines_final_processing_region": True,
    }
