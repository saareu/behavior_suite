"""Automatic cage detection migrated from the legacy preprocessing workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from preprocess.config import CageDetectConfig, PreprocessConfig
from preprocess.crop_plan import (
    CanonicalGeometry,
    CropMode,
    CropPlan,
    compute_canonical_geometry,
)
from preprocess.exceptions import CageDetectionError, CropPlanError
from preprocess.pre_crop import (
    PreCropROI,
    ResolvedPreCrop,
    build_pre_crop_translation_homography,
)


class CageDetectionResult(BaseModel):
    """A validated automatic CropPlan, preview background, and diagnostics."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    crop_plan: CropPlan
    cropped_background: np.ndarray
    detector_diagnostics: dict[str, object]

    @field_validator("cropped_background", mode="before")
    @classmethod
    def validate_cropped_background(cls, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != 2 or array.size == 0:
            raise CageDetectionError(
                "Detected cropped background must be a non-empty grayscale image."
            )
        if not np.all(np.isfinite(array)):
            raise CageDetectionError("Detected cropped background must contain finite values.")
        validated = np.array(array, dtype=np.uint8, copy=True)
        validated.setflags(write=False)
        return validated


@dataclass(frozen=True)
class _MedianBackground:
    image: np.ndarray
    sampled_frame_count: int


@dataclass(frozen=True)
class _DetectedRectangle:
    rectangle_in_search: tuple[tuple[float, float], tuple[float, float], float]
    quad_in_pre_crop: np.ndarray
    fit_score: float
    rim_density: float
    contour_area: float
    coarse_search_roi: tuple[int, int, int, int]


def _read_sampled_frames(video_path: Path, sample_step: int) -> list[np.ndarray]:
    if sample_step <= 0:
        raise CageDetectionError("sample_step must be positive.")
    if not video_path.is_file():
        raise CageDetectionError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise CageDetectionError(f"OpenCV could not open video for cage detection: {video_path}")

    frames: list[np.ndarray] = []
    try:
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_count > 0:
            for frame_index in range(0, reported_count, sample_step):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if success and frame is not None:
                    frames.append(frame)
        else:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame is not None and frame_index % sample_step == 0:
                    frames.append(frame)
                frame_index += 1
    finally:
        capture.release()
    return frames


def _build_median_background_with_count(
    video_path: Path,
    sample_step: int,
) -> _MedianBackground:
    frames = _read_sampled_frames(video_path, sample_step)
    if not frames:
        raise CageDetectionError(
            f"No usable frames were sampled for cage detection from: {video_path}"
        )

    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames):
        raise CageDetectionError("Sampled video frames do not have a consistent shape.")
    try:
        median = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    except (MemoryError, ValueError) as exc:
        raise CageDetectionError(f"Could not create median background: {exc}") from exc

    if median.ndim == 3:
        try:
            median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        except cv2.error as exc:
            raise CageDetectionError(
                f"Could not convert median background to grayscale: {exc}"
            ) from exc
    if median.ndim != 2 or median.size == 0:
        raise CageDetectionError("Median background is not a usable grayscale image.")
    return _MedianBackground(image=median, sampled_frame_count=len(frames))


def build_median_background(video_path: Path, sample_step: int) -> np.ndarray:
    """Build a grayscale median background from sampled video frames.

    Raises:
        CageDetectionError: If the video cannot be opened, no frames can be
            sampled, or a consistent median image cannot be produced.
    """

    return _build_median_background_with_count(
        Path(video_path).expanduser().resolve(),
        sample_step,
    ).image


def _estimate_coarse_search_roi(
    background: np.ndarray,
    detector_config: CageDetectConfig,
    roi_margin_px: int,
) -> tuple[int, int, int, int]:
    height, width = background.shape
    _, binary = cv2.threshold(
        background,
        detector_config.threshold,
        255,
        cv2.THRESH_BINARY,
    )
    edges = cv2.Canny(binary, 50, 150, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (detector_config.dilate_kernel_size, detector_config.dilate_kernel_size),
        ),
        iterations=1,
    )
    edges = cv2.erode(
        edges,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (detector_config.erode_kernel_size, detector_config.erode_kernel_size),
        ),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    usable_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= detector_config.minimum_contour_area
    ]
    if not usable_contours:
        raise CageDetectionError("No usable coarse cage region was found.")

    contour = max(usable_contours, key=cv2.contourArea)
    x, y, region_width, region_height = cv2.boundingRect(contour)
    center_x = x + region_width / 2.0
    center_y = y + region_height / 2.0
    expansion = 1.0 + detector_config.pre_crop_expansion_percent / 100.0
    expanded_width = region_width * expansion + 2 * roi_margin_px
    expanded_height = region_height * expansion + 2 * roi_margin_px

    x0 = max(0, int(np.floor(center_x - expanded_width / 2.0)))
    y0 = max(0, int(np.floor(center_y - expanded_height / 2.0)))
    x1 = min(width, int(np.ceil(center_x + expanded_width / 2.0)))
    y1 = min(height, int(np.ceil(center_y + expanded_height / 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise CageDetectionError("Coarse cage search ROI is empty after clamping.")
    return x0, y0, x1 - x0, y1 - y0


def _odd_kernel_from_fraction(
    image_shape: tuple[int, int],
    fraction: float,
    minimum: int = 3,
    maximum: int = 11,
) -> np.ndarray:
    size = int(round(min(image_shape) * fraction))
    size = max(minimum, min(maximum, size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _build_rim_edges(binary: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        _odd_kernel_from_fraction(binary.shape, 0.02),
        iterations=1,
    )
    gradient = cv2.morphologyEx(
        opened,
        cv2.MORPH_GRADIENT,
        _odd_kernel_from_fraction(binary.shape, 0.004),
        iterations=1,
    )
    gradient = cv2.dilate(
        gradient,
        _odd_kernel_from_fraction(binary.shape, 0.003),
        iterations=1,
    )
    return (gradient > 0).astype(np.uint8)


def _fit_score_for_rectangle(
    rim_edges: np.ndarray,
    rectangle: tuple[tuple[float, float], tuple[float, float], float],
    tolerance_px: int,
) -> float:
    perimeter = np.zeros(rim_edges.shape, dtype=np.uint8)
    box = cv2.boxPoints(rectangle).astype(np.int32)
    cv2.polylines(perimeter, [box], isClosed=True, color=255, thickness=1)
    tolerance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance_px + 1, 2 * tolerance_px + 1),
    )
    dilated_rim = cv2.dilate(rim_edges, tolerance_kernel, iterations=1) * 255
    matching_perimeter = cv2.bitwise_and(perimeter, dilated_rim)
    perimeter_pixels = int(np.count_nonzero(perimeter))
    if perimeter_pixels == 0:
        return 0.0
    return float(np.count_nonzero(matching_perimeter) / perimeter_pixels)


def _order_quad_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]
    indices = [
        int(np.argmin(sums)),
        int(np.argmin(differences)),
        int(np.argmax(sums)),
        int(np.argmax(differences)),
    ]
    if len(set(indices)) == 4:
        return np.asarray(points[indices], dtype=np.float64)

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    edge_a = ordered[1] - ordered[0]
    edge_b = ordered[2] - ordered[1]
    if edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0] < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return np.asarray(ordered, dtype=np.float64)


def _detect_rotated_rectangle(
    pre_cropped_background: np.ndarray,
    detector_config: CageDetectConfig,
    roi_margin_px: int,
) -> _DetectedRectangle:
    coarse_roi = _estimate_coarse_search_roi(
        pre_cropped_background,
        detector_config,
        roi_margin_px,
    )
    coarse_x, coarse_y, coarse_width, coarse_height = coarse_roi
    search_image = pre_cropped_background[
        coarse_y : coarse_y + coarse_height,
        coarse_x : coarse_x + coarse_width,
    ]
    _, binary = cv2.threshold(
        search_image,
        detector_config.threshold,
        255,
        cv2.THRESH_BINARY,
    )
    rim_edges = _build_rim_edges(binary)
    rim_density = float(rim_edges.mean())
    closed_edges = cv2.morphologyEx(
        rim_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                detector_config.rim_close_kernel_size,
                detector_config.rim_close_kernel_size,
            ),
        ),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours: list[np.ndarray] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (
            width >= detector_config.minimum_cage_width_fraction * coarse_width
            and height >= detector_config.minimum_cage_height_fraction * coarse_height
            and cv2.contourArea(contour) >= detector_config.minimum_contour_area
        ):
            valid_contours.append(contour)
    if not valid_contours:
        raise CageDetectionError(
            "No cage contour satisfied the configured size and area constraints."
        )

    best_contour = max(valid_contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(best_contour))
    rectangle_in_search = cv2.minAreaRect(best_contour.astype(np.float32))
    (_center_x, _center_y), (rectangle_width, rectangle_height), _angle = (
        rectangle_in_search
    )
    if rectangle_width <= 0 or rectangle_height <= 0:
        raise CageDetectionError("Detected cage rectangle has invalid dimensions.")

    fit_score = _fit_score_for_rectangle(
        rim_edges,
        rectangle_in_search,
        detector_config.fit_tolerance_px,
    )
    center, size, angle = rectangle_in_search
    rectangle_in_pre_crop = (
        (center[0] + coarse_x, center[1] + coarse_y),
        size,
        angle,
    )
    quad_in_pre_crop = _order_quad_tl_tr_br_bl(
        cv2.boxPoints(rectangle_in_pre_crop).astype(np.float64)
    )
    return _DetectedRectangle(
        rectangle_in_search=rectangle_in_search,
        quad_in_pre_crop=quad_in_pre_crop,
        fit_score=fit_score,
        rim_density=rim_density,
        contour_area=contour_area,
        coarse_search_roi=coarse_roi,
    )


def _make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _build_rectification_geometry(
    quad_in_pre_crop: np.ndarray,
    pad_px: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], bool]:
    top_width = float(np.linalg.norm(quad_in_pre_crop[1] - quad_in_pre_crop[0]))
    bottom_width = float(np.linalg.norm(quad_in_pre_crop[2] - quad_in_pre_crop[3]))
    right_height = float(np.linalg.norm(quad_in_pre_crop[2] - quad_in_pre_crop[1]))
    left_height = float(np.linalg.norm(quad_in_pre_crop[3] - quad_in_pre_crop[0]))
    rectified_width = _make_even(
        max(2 * pad_px + 2, int(round(max(top_width, bottom_width))) + 2 * pad_px)
    )
    rectified_height = _make_even(
        max(2 * pad_px + 2, int(round(max(left_height, right_height))) + 2 * pad_px)
    )
    destination = np.array(
        [
            [pad_px, pad_px],
            [rectified_width - 1 - pad_px, pad_px],
            [rectified_width - 1 - pad_px, rectified_height - 1 - pad_px],
            [pad_px, rectified_height - 1 - pad_px],
        ],
        dtype=np.float32,
    )
    homography_rectify = cv2.getPerspectiveTransform(
        quad_in_pre_crop.astype(np.float32),
        destination,
    ).astype(np.float64)

    rotated_90 = rectified_height > rectified_width
    if rotated_90:
        homography_rotate = np.array(
            [
                [0.0, -1.0, rectified_height - 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        output_size = (rectified_height, rectified_width)
    else:
        homography_rotate = np.eye(3, dtype=np.float64)
        output_size = (rectified_width, rectified_height)
    return homography_rectify, homography_rotate, output_size, rotated_90


def _build_canonical_transform(
    native_size_wh: tuple[int, int],
    config: PreprocessConfig,
) -> tuple[np.ndarray, CanonicalGeometry, tuple[int, int]]:
    native_width, native_height = native_size_wh
    canonical_config = config.prepare.canonical_resolution
    if canonical_config.enabled:
        geometry = compute_canonical_geometry(
            native_width,
            native_height,
            canonical_config.width,
            canonical_config.height,
        )
        transform = np.array(
            [
                [geometry.uniform_scale, 0.0, geometry.padding_left],
                [0.0, geometry.uniform_scale, geometry.padding_top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        output_size = geometry.canonical_size_wh
    else:
        geometry = compute_canonical_geometry(
            native_width,
            native_height,
            native_width,
            native_height,
        )
        transform = np.eye(3, dtype=np.float64)
        output_size = native_size_wh
    return transform, geometry, output_size


def _rectangle_diagnostics(
    detected: _DetectedRectangle,
    pre_crop: ResolvedPreCrop,
) -> dict[str, object]:
    center, size, angle = detected.rectangle_in_search
    coarse_x, coarse_y, _width, _height = detected.coarse_search_roi
    center_pre_crop = [center[0] + coarse_x, center[1] + coarse_y]
    return {
        "center_pre_crop_xy": center_pre_crop,
        "center_raw_xy": [
            center_pre_crop[0] + pre_crop.roi.x,
            center_pre_crop[1] + pre_crop.roi.y,
        ],
        "size_wh": [float(size[0]), float(size[1])],
        "angle_degrees": float(angle),
    }


def detect_cage_crop_plan(
    video_path: Path,
    config: PreprocessConfig,
    pre_crop: ResolvedPreCrop,
) -> CageDetectionResult:
    """Detect a cage and return an unaccepted automatic CropPlan.

    Detection operates inside the resolved pre-crop while composing the final
    homography from the original raw-frame coordinate system.

    Raises:
        CageDetectionError: If sampling, contour detection, transform creation,
            canonical geometry, CropPlan validation, or preview generation fails.
    """

    resolved_path = Path(video_path).expanduser().resolve()
    background_result = _build_median_background_with_count(
        resolved_path,
        config.cage_detect.sample_step,
    )
    background = background_result.image
    background_height, background_width = background.shape
    if pre_crop.raw_size_wh != (background_width, background_height):
        raise CageDetectionError(
            "Resolved pre-crop raw dimensions do not match the sampled video frames."
        )

    roi: PreCropROI = pre_crop.roi
    pre_cropped_background = background[
        roi.y : roi.y + roi.height,
        roi.x : roi.x + roi.width,
    ]
    if pre_cropped_background.shape != (roi.height, roi.width):
        raise CageDetectionError("Resolved pre-crop could not be applied to the background.")

    try:
        detected = _detect_rotated_rectangle(
            pre_cropped_background,
            config.cage_detect,
            config.prepare.roi_margin_px,
        )
        homography_rectify, homography_rotate, native_size_wh, rotated_90 = (
            _build_rectification_geometry(
                detected.quad_in_pre_crop,
                config.cage_detect.pad_px,
            )
        )
        homography_canonical, canonical_geometry, output_size_wh = (
            _build_canonical_transform(native_size_wh, config)
        )
    except cv2.error as exc:
        raise CageDetectionError(f"OpenCV cage geometry operation failed: {exc}") from exc
    except CropPlanError as exc:
        raise CageDetectionError(f"Canonical cage geometry is invalid: {exc}") from exc

    homography_raw_to_pre_crop = build_pre_crop_translation_homography(pre_crop)
    homography_pre_crop_to_prepared = (
        homography_canonical @ homography_rotate @ homography_rectify
    )
    homography_raw_to_prepared = (
        homography_pre_crop_to_prepared @ homography_raw_to_pre_crop
    )
    try:
        homography_prepared_to_raw = np.linalg.inv(homography_raw_to_prepared)
    except np.linalg.LinAlgError as exc:
        raise CageDetectionError("Detected cage homography is singular.") from exc
    if not np.all(np.isfinite(homography_prepared_to_raw)):
        raise CageDetectionError("Detected inverse cage homography is non-finite.")

    quad_raw = detected.quad_in_pre_crop + np.array(
        [pre_crop.roi.x, pre_crop.roi.y],
        dtype=np.float64,
    )
    try:
        crop_plan = CropPlan(
            mode=CropMode.AUTOMATIC,
            pre_crop_roi=pre_crop.roi,
            quad_raw_tl_tr_br_bl=quad_raw,
            H_raw_to_prepared_3x3=homography_raw_to_prepared,
            H_prepared_to_raw_3x3=homography_prepared_to_raw,
            prepared_size_wh=output_size_wh,
            rotated_90=rotated_90,
            fit_score=detected.fit_score,
            rim_density=detected.rim_density,
            accepted_by_user=False,
        )
    except (CropPlanError, ValidationError) as exc:
        raise CageDetectionError(f"Detected cage produced an invalid CropPlan: {exc}") from exc

    interpolation = (
        cv2.INTER_CUBIC
        if config.prepare.perspective_interpolation == "cubic"
        else cv2.INTER_LINEAR
    )
    try:
        cropped_background = cv2.warpPerspective(
            background,
            homography_raw_to_prepared,
            output_size_wh,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error as exc:
        raise CageDetectionError(f"Could not create detected cage preview: {exc}") from exc
    if cropped_background.shape != (output_size_wh[1], output_size_wh[0]):
        raise CageDetectionError("Detected cropped background has unexpected dimensions.")

    diagnostics: dict[str, object] = {
        "sampled_frame_count": background_result.sampled_frame_count,
        "background_shape": [background_height, background_width],
        "pre_crop_roi": pre_crop.roi.to_metadata_dict(),
        "coarse_search_roi_in_pre_crop": list(detected.coarse_search_roi),
        "detected_rotated_rectangle": _rectangle_diagnostics(detected, pre_crop),
        "detected_quad": quad_raw.tolist(),
        "fit_score": detected.fit_score,
        "rim_density": detected.rim_density,
        "contour_area": detected.contour_area,
        "output_size_before_canonical_scaling": list(native_size_wh),
        "canonical_scaling_info": {
            "enabled": config.prepare.canonical_resolution.enabled,
            **canonical_geometry.to_metadata_dict(),
        },
        "rotation_applied": rotated_90,
        "rotation_direction": "clockwise_90" if rotated_90 else "none",
        "H_raw_to_pre_crop_3x3": homography_raw_to_pre_crop.tolist(),
        "H_pre_crop_to_prepared_3x3": homography_pre_crop_to_prepared.tolist(),
        "detector_parameters": config.cage_detect.model_dump(mode="json"),
        "roi_margin_px": config.prepare.roi_margin_px,
    }
    return CageDetectionResult(
        crop_plan=crop_plan,
        cropped_background=cropped_background,
        detector_diagnostics=diagnostics,
    )
