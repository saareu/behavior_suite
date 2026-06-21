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


def validate_quad_tl_tr_br_bl(value: Any) -> np.ndarray:
    """Validate and copy four finite corners in TL, TR, BR, BL order.

    The returned float64 array is read-only. Points are never reordered.

    Raises:
        CropPlanError: If the input is not a numeric ``(4, 2)`` array or the
            quadrilateral is degenerate, self-intersecting, non-convex, or in
            an order other than top-left, top-right, bottom-right, bottom-left.
    """

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
    native_size_wh: tuple[StrictInt, StrictInt] | None = None
    canonical_geometry: CanonicalGeometry | None = None
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
        return validate_quad_tl_tr_br_bl(value)

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

        if (self.native_size_wh is None) != (self.canonical_geometry is None):
            raise CropPlanError(
                "native_size_wh and canonical_geometry must be provided together."
            )
        if self.native_size_wh is not None and self.canonical_geometry is not None:
            native_width, native_height = self.native_size_wh
            if native_width <= 0 or native_height <= 0:
                raise CropPlanError("Native width and height must be positive.")
            if native_width % 2 or native_height % 2:
                raise CropPlanError("Native width and height must be even.")
            if self.canonical_geometry.native_size_wh != self.native_size_wh:
                raise CropPlanError(
                    "Canonical geometry native size must match native_size_wh."
                )
            if self.canonical_geometry.canonical_size_wh != self.prepared_size_wh:
                raise CropPlanError(
                    "Canonical geometry final size must match prepared_size_wh."
                )

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


CropPlan.model_rebuild()


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


def round_up_to_even_dimension(value: float) -> int:
    """Round a positive finite dimension upward to the next even integer.

    Raises:
        CropPlanError: If ``value`` is not numeric, finite, and positive.
    """

    if isinstance(value, bool):
        raise CropPlanError("Rectified dimensions must be numeric.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise CropPlanError("Rectified dimensions must be numeric.") from exc
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise CropPlanError("Rectified dimensions must be finite and positive.")
    return max(2, int(2 * math.ceil(numeric_value / 2.0)))


def compute_native_rectified_size(
    quad_tl_tr_br_bl: np.ndarray,
    padding_px: int = 0,
) -> tuple[int, int]:
    """Compute upward-rounded even rectified dimensions from opposite edges.

    ``padding_px`` is symmetric spatial padding around the source quadrilateral
    and is included before the final upward-even rounding.

    Raises:
        CropPlanError: If the quadrilateral or padding is invalid.
    """

    quad = validate_quad_tl_tr_br_bl(quad_tl_tr_br_bl)
    if isinstance(padding_px, bool) or not isinstance(padding_px, int):
        raise CropPlanError("Rectification padding must be an integer.")
    if padding_px < 0:
        raise CropPlanError("Rectification padding must be non-negative.")

    top_width = float(np.linalg.norm(quad[1] - quad[0]))
    bottom_width = float(np.linalg.norm(quad[2] - quad[3]))
    right_height = float(np.linalg.norm(quad[2] - quad[1]))
    left_height = float(np.linalg.norm(quad[3] - quad[0]))
    return (
        round_up_to_even_dimension(max(top_width, bottom_width) + 2 * padding_px),
        round_up_to_even_dimension(max(left_height, right_height) + 2 * padding_px),
    )


def build_clockwise_rotation_transform(
    native_size_wh: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], bool]:
    """Build the shared portrait-to-landscape clockwise rotation transform.

    Portrait geometry is rotated 90 degrees clockwise. Landscape and square
    geometry receives an identity transform.

    Raises:
        CropPlanError: If native dimensions are not positive even integers.
    """

    try:
        native_width, native_height = native_size_wh
    except (TypeError, ValueError) as exc:
        raise CropPlanError("Native size must contain width and height.") from exc
    native_width = _validate_dimension("native_width", native_width, require_even=True)
    native_height = _validate_dimension("native_height", native_height, require_even=True)
    if native_height <= native_width:
        return np.eye(3, dtype=np.float64), (native_width, native_height), False

    transform = np.array(
        [
            [0.0, -1.0, native_height - 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform, (native_height, native_width), True


def build_canonical_transform(
    native_size_wh: tuple[int, int],
    canonical_size_wh: tuple[int, int] | None,
) -> tuple[np.ndarray, CanonicalGeometry, tuple[int, int]]:
    """Build the shared native-to-final uniform scale and padding transform.

    A ``None`` canonical size disables canonical scaling and returns an identity
    transform with the native output dimensions and no padding.

    Raises:
        CropPlanError: If native or enabled canonical dimensions are invalid.
    """

    try:
        native_width, native_height = native_size_wh
    except (TypeError, ValueError) as exc:
        raise CropPlanError("Native size must contain width and height.") from exc
    if canonical_size_wh is None:
        canonical_width, canonical_height = native_width, native_height
    else:
        try:
            canonical_width, canonical_height = canonical_size_wh
        except (TypeError, ValueError) as exc:
            raise CropPlanError("Canonical size must contain width and height.") from exc

    geometry = compute_canonical_geometry(
        native_width,
        native_height,
        canonical_width,
        canonical_height,
    )
    if canonical_size_wh is None:
        return np.eye(3, dtype=np.float64), geometry, (native_width, native_height)

    transform = np.array(
        [
            [geometry.uniform_scale, 0.0, geometry.padding_left],
            [0.0, geometry.uniform_scale, geometry.padding_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform, geometry, geometry.canonical_size_wh
