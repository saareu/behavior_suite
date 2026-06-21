"""Validated crop-plan and canonical-resolution geometry models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from preprocess.exceptions import CropPlanError
from preprocess.pre_crop import PreCropROI

HOMOGRAPHY_INVERSE_RTOL = 1e-6
HOMOGRAPHY_INVERSE_ATOL = 1e-6
QUAD_GEOMETRY_ATOL = 1e-9


class CropMode(StrEnum):
    """Origins supported by the shared CropPlan interface."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"


def _cross_z(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    vector_ab = point_b - point_a
    vector_ac = point_c - point_a
    return float(vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0])


def _point_on_segment(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point: np.ndarray,
) -> bool:
    return bool(
        min(point_a[0], point_b[0]) - QUAD_GEOMETRY_ATOL
        <= point[0]
        <= max(point_a[0], point_b[0]) + QUAD_GEOMETRY_ATOL
        and min(point_a[1], point_b[1]) - QUAD_GEOMETRY_ATOL
        <= point[1]
        <= max(point_a[1], point_b[1]) + QUAD_GEOMETRY_ATOL
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    orientation_1 = _cross_z(first_start, first_end, second_start)
    orientation_2 = _cross_z(first_start, first_end, second_end)
    orientation_3 = _cross_z(second_start, second_end, first_start)
    orientation_4 = _cross_z(second_start, second_end, first_end)

    if (
        orientation_1 * orientation_2 < -(QUAD_GEOMETRY_ATOL**2)
        and orientation_3 * orientation_4 < -(QUAD_GEOMETRY_ATOL**2)
    ):
        return True
    collinear_cases = (
        (orientation_1, first_start, first_end, second_start),
        (orientation_2, first_start, first_end, second_end),
        (orientation_3, second_start, second_end, first_start),
        (orientation_4, second_start, second_end, first_end),
    )
    return any(
        abs(orientation) <= QUAD_GEOMETRY_ATOL
        and _point_on_segment(segment_start, segment_end, point)
        for orientation, segment_start, segment_end, point in collinear_cases
    )


def _validate_quad_geometry(quad: np.ndarray) -> None:
    if _segments_intersect(quad[0], quad[1], quad[2], quad[3]) or _segments_intersect(
        quad[1], quad[2], quad[3], quad[0]
    ):
        raise CropPlanError("Crop quadrilateral is self-intersecting.")

    cross_products = np.array(
        [_cross_z(quad[index], quad[(index + 1) % 4], quad[(index + 2) % 4]) for index in range(4)]
    )
    if np.any(np.abs(cross_products) <= QUAD_GEOMETRY_ATOL):
        raise CropPlanError("Crop quadrilateral is degenerate.")
    if not np.all(cross_products > 0):
        if np.all(cross_products < 0):
            raise CropPlanError(
                "Crop quadrilateral must use top-left, top-right, bottom-right, bottom-left order."
            )
        raise CropPlanError("Crop quadrilateral must be convex.")

    top_center_y = float((quad[0, 1] + quad[1, 1]) / 2.0)
    bottom_center_y = float((quad[2, 1] + quad[3, 1]) / 2.0)
    left_center_x = float((quad[0, 0] + quad[3, 0]) / 2.0)
    right_center_x = float((quad[1, 0] + quad[2, 0]) / 2.0)
    if (
        top_center_y >= bottom_center_y - QUAD_GEOMETRY_ATOL
        or left_center_x >= right_center_x - QUAD_GEOMETRY_ATOL
    ):
        raise CropPlanError(
            "Crop quadrilateral must use top-left, top-right, bottom-right, bottom-left order."
        )


class CropPlan(BaseModel):
    """A shared, validated raw-frame-to-prepared-frame crop transformation.

    Homography inverse validation uses relative and absolute tolerances of
    1e-6 to accommodate composed float32/float64 image transforms.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    mode: CropMode
    pre_crop_roi: PreCropROI
    quad_raw_tl_tr_br_bl: np.ndarray
    H_raw_to_prepared_3x3: np.ndarray
    H_prepared_to_raw_3x3: np.ndarray
    prepared_size_wh: tuple[StrictInt, StrictInt]
    rotated_90: bool
    fit_score: float | None = None
    rim_density: float | None = None
    accepted_by_user: bool

    @field_validator("pre_crop_roi", mode="before")
    @classmethod
    def coerce_pre_crop_roi(cls, value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            if len(value) != 4:
                raise CropPlanError(
                    "pre_crop_roi must contain x, y, width, and height."
                )
            return {
                "x": value[0],
                "y": value[1],
                "width": value[2],
                "height": value[3],
            }
        return value

    @field_validator("quad_raw_tl_tr_br_bl", mode="before")
    @classmethod
    def validate_quad_array(cls, value: Any) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CropPlanError("Crop quadrilateral must be numeric.") from exc
        if array.shape != (4, 2):
            raise CropPlanError(
                "quad_raw_tl_tr_br_bl must contain exactly four (x, y) points."
            )
        if not np.all(np.isfinite(array)):
            raise CropPlanError("Crop quadrilateral must contain only finite values.")
        validated = np.array(array, dtype=np.float64, copy=True)
        _validate_quad_geometry(validated)
        validated.setflags(write=False)
        return validated

    @field_validator("H_raw_to_prepared_3x3", "H_prepared_to_raw_3x3", mode="before")
    @classmethod
    def validate_homography_array(cls, value: Any) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CropPlanError("Homography matrices must be numeric.") from exc
        if array.shape != (3, 3):
            raise CropPlanError("Homography matrices must have shape (3, 3).")
        if not np.all(np.isfinite(array)):
            raise CropPlanError("Homography matrices must contain only finite values.")
        validated = np.array(array, dtype=np.float64, copy=True)
        validated.setflags(write=False)
        return validated

    @model_validator(mode="after")
    def validate_crop_plan_geometry(self) -> CropPlan:
        width, height = self.prepared_size_wh
        if width <= 0 or height <= 0:
            raise CropPlanError("Prepared width and height must be positive.")
        if width % 2 or height % 2:
            raise CropPlanError("Prepared width and height must be even.")

        for field_name, value in (
            ("fit_score", self.fit_score),
            ("rim_density", self.rim_density),
        ):
            if value is not None and not math.isfinite(value):
                raise CropPlanError(f"{field_name} must be finite when provided.")

        identity = np.eye(3, dtype=np.float64)
        forward_then_inverse = self.H_prepared_to_raw_3x3 @ self.H_raw_to_prepared_3x3
        inverse_then_forward = self.H_raw_to_prepared_3x3 @ self.H_prepared_to_raw_3x3
        if not (
            np.allclose(
                forward_then_inverse,
                identity,
                rtol=HOMOGRAPHY_INVERSE_RTOL,
                atol=HOMOGRAPHY_INVERSE_ATOL,
            )
            and np.allclose(
                inverse_then_forward,
                identity,
                rtol=HOMOGRAPHY_INVERSE_RTOL,
                atol=HOMOGRAPHY_INVERSE_ATOL,
            )
        ):
            raise CropPlanError(
                "H_prepared_to_raw_3x3 must be the inverse of "
                "H_raw_to_prepared_3x3 within a 1e-6 tolerance."
            )
        return self

    @field_serializer(
        "quad_raw_tl_tr_br_bl",
        "H_raw_to_prepared_3x3",
        "H_prepared_to_raw_3x3",
        when_used="json",
    )
    def serialize_array(self, value: np.ndarray) -> list[list[float]]:
        return value.tolist()

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe CropPlan metadata with arrays as nested lists."""

        return self.model_dump(mode="json")

    @classmethod
    def from_metadata_dict(cls, metadata: Mapping[str, Any]) -> CropPlan:
        """Recreate and validate a CropPlan from JSON-compatible metadata."""

        return cls.model_validate(dict(metadata))


class CanonicalGeometry(BaseModel):
    """Uniform scaling and centered-padding geometry for canonical output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native_size_wh: tuple[int, int]
    uniform_scale: float
    scaled_size_wh: tuple[int, int]
    padding_left: int
    padding_right: int
    padding_top: int
    padding_bottom: int
    canonical_size_wh: tuple[int, int]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe canonical geometry metadata."""

        return self.model_dump(mode="json")


def _validate_dimension(name: str, value: int, *, require_even: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CropPlanError(f"{name} must be an integer.")
    if value <= 0:
        raise CropPlanError(f"{name} must be positive.")
    if require_even and value % 2:
        raise CropPlanError(f"{name} must be even.")
    return value


def compute_canonical_geometry(
    native_width: int,
    native_height: int,
    canonical_width: int,
    canonical_height: int,
) -> CanonicalGeometry:
    """Compute one uniform scale and centered black-padding geometry.

    Both native dimensions are multiplied by the same floating-point scale and
    rounded to the nearest integer pixel. No image processing library is used.

    Raises:
        CropPlanError: If native dimensions are invalid or canonical
            dimensions are not positive even integers.
    """

    native_width = _validate_dimension("native_width", native_width, require_even=False)
    native_height = _validate_dimension("native_height", native_height, require_even=False)
    canonical_width = _validate_dimension(
        "canonical_width",
        canonical_width,
        require_even=True,
    )
    canonical_height = _validate_dimension(
        "canonical_height",
        canonical_height,
        require_even=True,
    )

    uniform_scale = min(
        canonical_width / native_width,
        canonical_height / native_height,
    )
    scaled_width = max(1, min(canonical_width, int(round(native_width * uniform_scale))))
    scaled_height = max(1, min(canonical_height, int(round(native_height * uniform_scale))))

    horizontal_padding = canonical_width - scaled_width
    vertical_padding = canonical_height - scaled_height
    padding_left = horizontal_padding // 2
    padding_top = vertical_padding // 2

    return CanonicalGeometry(
        native_size_wh=(native_width, native_height),
        uniform_scale=uniform_scale,
        scaled_size_wh=(scaled_width, scaled_height),
        padding_left=padding_left,
        padding_right=horizontal_padding - padding_left,
        padding_top=padding_top,
        padding_bottom=vertical_padding - padding_top,
        canonical_size_wh=(canonical_width, canonical_height),
    )
