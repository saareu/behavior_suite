"""Pure geometry for creating reviewed crop plans."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import ValidationError

from preprocess.config import CanonicalResolutionConfig
from preprocess.crop_plan import (
    PERSPECTIVE_REPROJECTION_ATOL_PX,
    CropMode,
    CropPlan,
    build_canonical_transform,
    build_clockwise_rotation_transform,
    build_perspective_homography,
    compute_native_rectified_size,
    validate_quad_tl_tr_br_bl,
)
from preprocess.exceptions import CropPlanError
from preprocess.pre_crop import PreCropROI, build_pre_crop_translation_homography


def _validate_raw_frame_shape(raw_frame_shape: tuple[int, int]) -> tuple[int, int]:
    try:
        dimension_count = len(raw_frame_shape)
    except TypeError as exc:
        raise CropPlanError(
            "raw_frame_shape must contain integer height and width values."
        ) from exc
    if dimension_count != 2:
        raise CropPlanError("raw_frame_shape must contain exactly height and width.")

    raw_height, raw_width = raw_frame_shape
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (raw_height, raw_width)
    ):
        raise CropPlanError("Raw frame height and width must be integers.")
    if raw_height <= 0 or raw_width <= 0:
        raise CropPlanError("Raw frame height and width must be positive.")
    return raw_height, raw_width


def _resolve_pre_crop_roi(
    pre_crop_roi: tuple[int, int, int, int] | None,
    raw_width: int,
    raw_height: int,
) -> PreCropROI:
    if pre_crop_roi is None:
        return PreCropROI(x=0, y=0, width=raw_width, height=raw_height)
    try:
        value_count = len(pre_crop_roi)
    except TypeError as exc:
        raise CropPlanError(
            "pre_crop_roi must contain integer x, y, width, and height values."
        ) from exc
    if value_count != 4:
        raise CropPlanError("pre_crop_roi must contain x, y, width, and height.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in pre_crop_roi):
        raise CropPlanError("pre_crop_roi values must be integers.")

    x, y, width, height = pre_crop_roi
    if x < 0 or y < 0:
        raise CropPlanError("pre_crop_roi x and y must be non-negative.")
    if width <= 0 or height <= 0:
        raise CropPlanError("pre_crop_roi width and height must be positive.")
    if x + width > raw_width or y + height > raw_height:
        raise CropPlanError("pre_crop_roi must lie within the raw frame.")
    return PreCropROI(x=x, y=y, width=width, height=height)


def _validate_points_within_region(
    points: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    region_name: str,
) -> None:
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    if not (
        np.all(x_coordinates >= x)
        and np.all(x_coordinates < x + width)
        and np.all(y_coordinates >= y)
        and np.all(y_coordinates < y + height)
    ):
        raise CropPlanError(
            f"Manual crop points must lie within the {region_name} bounds."
        )


def _validate_rectangle_xywh(
    rectangle_xywh: tuple[int, int, int, int],
    *,
    raw_width: int,
    raw_height: int,
    pre_crop_roi: PreCropROI,
) -> tuple[int, int, int, int]:
    try:
        value_count = len(rectangle_xywh)
    except TypeError as exc:
        raise CropPlanError(
            "Manual rectangle must contain integer x, y, width, and height values."
        ) from exc
    if value_count != 4:
        raise CropPlanError("Manual rectangle must contain x, y, width, and height.")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in rectangle_xywh
    ):
        raise CropPlanError("Manual rectangle values must be integers.")

    x, y, width, height = rectangle_xywh
    if x < 0 or y < 0:
        raise CropPlanError("Manual rectangle x and y must be non-negative.")
    if width <= 0 or height <= 0:
        raise CropPlanError("Manual rectangle width and height must be positive.")
    if width % 2 or height % 2:
        raise CropPlanError(
            "Manual rectangle width and height must be even for the current "
            "CropPlan preparation path."
        )
    if x + width > raw_width or y + height > raw_height:
        raise CropPlanError("Manual rectangle must lie within the raw frame.")
    if not (
        pre_crop_roi.x <= x
        and pre_crop_roi.y <= y
        and x + width <= pre_crop_roi.x + pre_crop_roi.width
        and y + height <= pre_crop_roi.y + pre_crop_roi.height
    ):
        raise CropPlanError(
            "Manual rectangle must lie within the selected pre-crop ROI."
        )
    return x, y, width, height


def _validate_full_frame_size(raw_width: int, raw_height: int) -> None:
    if raw_width % 2 or raw_height % 2:
        raise CropPlanError(
            "Full-frame raw width and height must be even for the current "
            "preparation path."
        )


def _build_rectification_geometry(
    points_in_pre_crop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], bool]:
    rectified_width, rectified_height = compute_native_rectified_size(
        points_in_pre_crop
    )

    destination = np.array(
        [
            [0.0, 0.0],
            [rectified_width - 1.0, 0.0],
            [rectified_width - 1.0, rectified_height - 1.0],
            [0.0, rectified_height - 1.0],
        ],
        dtype=np.float64,
    )
    points_float32 = points_in_pre_crop.astype(np.float32)
    destination_float32 = destination.astype(np.float32)
    legacy_homography = cv2.getPerspectiveTransform(
        points_float32,
        destination_float32,
    ).astype(np.float64)
    homogeneous_points = np.column_stack(
        [
            points_in_pre_crop,
            np.ones(points_in_pre_crop.shape[0], dtype=np.float64),
        ]
    )
    legacy_destination = (legacy_homography @ homogeneous_points.T).T
    legacy_destination = (
        legacy_destination[:, :2] / legacy_destination[:, 2, np.newaxis]
    )
    if np.allclose(
        legacy_destination,
        destination,
        rtol=0.0,
        atol=PERSPECTIVE_REPROJECTION_ATOL_PX,
    ):
        # Preserve the established automatic/manual numeric parity when the
        # legacy float32 solve already reproduces the authoritative points.
        homography_rectify = legacy_homography
    else:
        # Avoid storing float64 points with a homography fitted to materially
        # different float32 coordinates, which Stage A cannot reproduce.
        homography_rectify = build_perspective_homography(
            points_in_pre_crop,
            destination,
        )
    homography_rotate, output_size, rotated_90 = (
        build_clockwise_rotation_transform((rectified_width, rectified_height))
    )
    return homography_rectify, homography_rotate, output_size, rotated_90


def make_manual_crop_plan(
    raw_frame_shape: tuple[int, int],
    points_tl_tr_br_bl: np.ndarray,
    pre_crop_roi: tuple[int, int, int, int] | None,
    canonical_resolution: CanonicalResolutionConfig,
) -> CropPlan:
    """Create an unaccepted manual CropPlan from four raw-frame corners.

    ``raw_frame_shape`` uses NumPy order ``(height, width)``. Manual points use
    original raw-frame ``(x, y)`` coordinates in TL, TR, BR, BL order and are
    never reordered or clamped. A supplied pre-crop uses ``(x, y, width,
    height)`` and is composed into the final raw-to-prepared transform.

    Manual plans use ``None`` for both detector-only metrics, ``fit_score`` and
    ``rim_density``, and start with ``accepted_by_user=False``.

    Raises:
        CropPlanError: If frame, ROI, point, homography, canonical, or CropPlan
            geometry is invalid.
    """

    raw_height, raw_width = _validate_raw_frame_shape(raw_frame_shape)
    points = validate_quad_tl_tr_br_bl(points_tl_tr_br_bl)
    _validate_points_within_region(
        points,
        x=0,
        y=0,
        width=raw_width,
        height=raw_height,
        region_name="raw frame",
    )

    roi = _resolve_pre_crop_roi(pre_crop_roi, raw_width, raw_height)
    _validate_points_within_region(
        points,
        x=roi.x,
        y=roi.y,
        width=roi.width,
        height=roi.height,
        region_name="pre-crop ROI",
    )

    homography_raw_to_pre_crop = build_pre_crop_translation_homography(roi)
    points_in_pre_crop = points - np.array([roi.x, roi.y], dtype=np.float64)
    homography_rectify, homography_rotate, native_size_wh, rotated_90 = (
        _build_rectification_geometry(points_in_pre_crop)
    )
    canonical_size_wh = (
        (canonical_resolution.width, canonical_resolution.height)
        if canonical_resolution.enabled
        else None
    )
    homography_canonical, canonical_geometry, output_size_wh = (
        build_canonical_transform(
            native_size_wh,
            canonical_size_wh,
        )
    )

    homography_raw_to_prepared = (
        homography_canonical
        @ homography_rotate
        @ homography_rectify
        @ homography_raw_to_pre_crop
    )
    if not np.all(np.isfinite(homography_raw_to_prepared)):
        raise CropPlanError("Manual raw-to-prepared homography is non-finite.")
    try:
        homography_prepared_to_raw = np.linalg.inv(homography_raw_to_prepared)
    except np.linalg.LinAlgError as exc:
        raise CropPlanError("Manual raw-to-prepared homography is singular.") from exc
    if not np.all(np.isfinite(homography_prepared_to_raw)):
        raise CropPlanError("Manual prepared-to-raw homography is non-finite.")

    try:
        return CropPlan(
            mode=CropMode.MANUAL,
            pre_crop_roi=roi,
            quad_raw_tl_tr_br_bl=points,
            H_raw_to_prepared_3x3=homography_raw_to_prepared,
            H_prepared_to_raw_3x3=homography_prepared_to_raw,
            prepared_size_wh=output_size_wh,
            native_size_wh=native_size_wh,
            canonical_geometry=canonical_geometry,
            rotated_90=rotated_90,
            fit_score=None,
            rim_density=None,
            accepted_by_user=False,
        )
    except (CropPlanError, ValidationError) as exc:
        raise CropPlanError(f"Manual crop produced an invalid CropPlan: {exc}") from exc


def make_axis_aligned_rectangle_crop_plan(
    raw_frame_shape: tuple[int, int],
    rectangle_xywh: tuple[int, int, int, int],
    pre_crop_roi: tuple[int, int, int, int] | None,
    canonical_resolution: CanonicalResolutionConfig,
) -> CropPlan:
    """Create an unaccepted manual CropPlan from one raw-frame rectangle.

    ``raw_frame_shape`` uses NumPy order ``(height, width)``. The rectangle uses
    raw-frame half-open ``(x, y, width, height)`` coordinates. This helper
    deliberately does not call the four-corner manual helper, because that path
    preserves the historical portrait-to-landscape rotation behavior. Rectangle
    mode preserves source orientation by using an axis-aligned rectangle
    homography, no rotation transform, and the existing canonical uniform
    scale/pad stage when enabled.

    The current CropPlan/preparation contract requires even native dimensions;
    odd rectangle widths or heights are rejected clearly rather than silently
    changing the selected crop.

    Raises:
        CropPlanError: If frame, ROI, rectangle, homography, canonical, or
            CropPlan geometry is invalid.
    """

    raw_height, raw_width = _validate_raw_frame_shape(raw_frame_shape)
    roi = _resolve_pre_crop_roi(pre_crop_roi, raw_width, raw_height)
    x, y, width, height = _validate_rectangle_xywh(
        rectangle_xywh,
        raw_width=raw_width,
        raw_height=raw_height,
        pre_crop_roi=roi,
    )
    points = np.array(
        [
            [float(x), float(y)],
            [float(x + width - 1), float(y)],
            [float(x + width - 1), float(y + height - 1)],
            [float(x), float(y + height - 1)],
        ],
        dtype=np.float64,
    )
    points = validate_quad_tl_tr_br_bl(points)

    homography_raw_to_pre_crop = build_pre_crop_translation_homography(roi)
    points_in_pre_crop = points - np.array([roi.x, roi.y], dtype=np.float64)
    destination = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )
    homography_rectify = build_perspective_homography(
        points_in_pre_crop,
        destination,
    )
    canonical_size_wh = (
        (canonical_resolution.width, canonical_resolution.height)
        if canonical_resolution.enabled
        else None
    )
    native_size_wh = (width, height)
    homography_canonical, canonical_geometry, output_size_wh = (
        build_canonical_transform(native_size_wh, canonical_size_wh)
    )

    homography_raw_to_prepared = (
        homography_canonical @ homography_rectify @ homography_raw_to_pre_crop
    )
    if not np.all(np.isfinite(homography_raw_to_prepared)):
        raise CropPlanError("Manual rectangle raw-to-prepared homography is non-finite.")
    try:
        homography_prepared_to_raw = np.linalg.inv(homography_raw_to_prepared)
    except np.linalg.LinAlgError as exc:
        raise CropPlanError(
            "Manual rectangle raw-to-prepared homography is singular."
        ) from exc
    if not np.all(np.isfinite(homography_prepared_to_raw)):
        raise CropPlanError(
            "Manual rectangle prepared-to-raw homography is non-finite."
        )

    try:
        return CropPlan(
            mode=CropMode.MANUAL,
            pre_crop_roi=roi,
            quad_raw_tl_tr_br_bl=points,
            H_raw_to_prepared_3x3=homography_raw_to_prepared,
            H_prepared_to_raw_3x3=homography_prepared_to_raw,
            prepared_size_wh=output_size_wh,
            native_size_wh=native_size_wh,
            canonical_geometry=canonical_geometry,
            rotated_90=False,
            fit_score=None,
            rim_density=None,
            accepted_by_user=False,
        )
    except (CropPlanError, ValidationError) as exc:
        raise CropPlanError(
            f"Manual rectangle crop produced an invalid CropPlan: {exc}"
        ) from exc


def make_full_frame_crop_plan(
    raw_frame_shape: tuple[int, int],
    canonical_resolution: CanonicalResolutionConfig,
) -> CropPlan:
    """Create an unaccepted full-frame CropPlan from raw frame dimensions.

    ``raw_frame_shape`` uses NumPy order ``(height, width)``. The selected
    source region is the whole raw frame. No pre-crop, perspective crop, or
    automatic rotation is applied. Existing canonical uniform scale/pad
    geometry is composed through the shared canonical transform helper.

    Raises:
        CropPlanError: If frame dimensions, canonical geometry, or CropPlan
            validation fails.
    """

    raw_height, raw_width = _validate_raw_frame_shape(raw_frame_shape)
    _validate_full_frame_size(raw_width, raw_height)
    roi = PreCropROI(x=0, y=0, width=raw_width, height=raw_height)
    points = validate_quad_tl_tr_br_bl(
        np.array(
            [
                [0.0, 0.0],
                [float(raw_width - 1), 0.0],
                [float(raw_width - 1), float(raw_height - 1)],
                [0.0, float(raw_height - 1)],
            ],
            dtype=np.float64,
        )
    )
    native_size_wh = (raw_width, raw_height)
    canonical_size_wh = (
        (canonical_resolution.width, canonical_resolution.height)
        if canonical_resolution.enabled
        else None
    )
    homography_canonical, canonical_geometry, output_size_wh = (
        build_canonical_transform(native_size_wh, canonical_size_wh)
    )

    try:
        homography_prepared_to_raw = np.linalg.inv(homography_canonical)
    except np.linalg.LinAlgError as exc:
        raise CropPlanError("Full-frame canonical transform is singular.") from exc
    if not np.all(np.isfinite(homography_prepared_to_raw)):
        raise CropPlanError("Full-frame prepared-to-raw homography is non-finite.")

    try:
        return CropPlan(
            mode=CropMode.FULL_FRAME,
            pre_crop_roi=roi,
            quad_raw_tl_tr_br_bl=points,
            H_raw_to_prepared_3x3=homography_canonical,
            H_prepared_to_raw_3x3=homography_prepared_to_raw,
            prepared_size_wh=output_size_wh,
            native_size_wh=native_size_wh,
            canonical_geometry=canonical_geometry,
            rotated_90=False,
            fit_score=None,
            rim_density=None,
            accepted_by_user=False,
        )
    except (CropPlanError, ValidationError) as exc:
        raise CropPlanError(
            f"Full-frame crop produced an invalid CropPlan: {exc}"
        ) from exc
