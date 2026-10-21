"""Per-frame foreground segmentation adapted from legacy Pipeline 40.

Preserves the scientific sequence used by
``legacy/pipelines/40_extract_blob_features.py``:

1. Foreground subtraction (background minus frame)
2. Blur and thresholding (fixed / Otsu / Multi-Otsu)
3. Morphological cleanup (close/open, area filter, thickness prune)
4. Connected-component / contour extraction

Cable-artifact removal is intentionally omitted (legacy lab default ``false``;
``tools.remove_cable_artifact`` is also absent from this checkout).

This module is the shared core for interactive preview and full extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from scipy.spatial import cKDTree

from blob_features.config import BlobSegmentationSettings
from blob_features.contracts import BlobFeaturesError

_threshold_multiotsu: Callable[..., Any] | None
try:
    from skimage.filters import threshold_multiotsu as _imported_multiotsu

    _threshold_multiotsu = _imported_multiotsu
    _SKIMAGE_IMPORT_ERROR: BaseException | None = None
except ImportError as exc:  # pragma: no cover - exercised via require helper
    _threshold_multiotsu = None
    _SKIMAGE_IMPORT_ERROR = exc


@dataclass(frozen=True, slots=True)
class FrameSegmentationResult:
    """Binary mask plus legacy contour topology for one prepared-video frame."""

    mask: np.ndarray
    contours: tuple[np.ndarray, ...]
    cc_count_raw: int
    cc_count: float
    min_inter_contour_dist: float


@dataclass(frozen=True, slots=True)
class FrameSegmentationDiagnostics:
    """Diagnostic images from the same scientific mask path (preview only).

    Intermediate arrays are snapshots taken during ``_extract_robust_foreground_mask``;
    scientific calculations are unchanged. ``mask`` matches ``segment_frame``.
    """

    foreground_gray: np.ndarray
    binary_pre_morph: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class FrameSegmentationPreview:
    """Single-frame preview payload for GUI inspection (not temporal features)."""

    prepared_frame_idx: int
    settings: BlobSegmentationSettings
    topology: FrameSegmentationResult
    foreground_gray: np.ndarray
    binary_pre_morph: np.ndarray
    warnings: tuple[str, ...]

    @property
    def cc_count_raw(self) -> int:
        return self.topology.cc_count_raw

    @property
    def cc_count(self) -> float:
        return self.topology.cc_count

    @property
    def min_inter_contour_dist(self) -> float:
        return self.topology.min_inter_contour_dist

    @property
    def mask(self) -> np.ndarray:
        return self.topology.mask

    @property
    def contours(self) -> tuple[np.ndarray, ...]:
        return self.topology.contours


def segment_frame(
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
) -> FrameSegmentationResult:
    """Segment one prepared-video frame against its cropped background.

    Parameters
    ----------
    frame:
        BGR or grayscale prepared-video frame (uint8).
    background:
        Matching cropped background image (same HxW).
    settings:
        Validated segmentation settings (lab defaults unless overridden).
    """

    validated = settings.validated()
    _validate_frame_background(frame, background)
    mask = extract_foreground_mask(frame, background, validated)
    return extract_contour_topology(mask)


def segment_frame_with_diagnostics(
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
    *,
    prepared_frame_idx: int = 0,
) -> FrameSegmentationPreview:
    """Segment one frame and return diagnostic images + topology for GUI preview.

    Uses the same scientific core as ``segment_frame`` / full extraction. Does
    not compute source-sequence temporal features or gap fill.
    """

    validated = settings.validated()
    _validate_frame_background(frame, background)
    diagnostics = extract_foreground_mask_with_diagnostics(frame, background, validated)
    topology = extract_contour_topology(diagnostics.mask)
    return FrameSegmentationPreview(
        prepared_frame_idx=int(prepared_frame_idx),
        settings=validated,
        topology=topology,
        foreground_gray=diagnostics.foreground_gray,
        binary_pre_morph=diagnostics.binary_pre_morph,
        warnings=_preview_warnings(topology),
    )


def extract_foreground_mask(
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
) -> np.ndarray:
    """Return the robust binary foreground mask (mice=255, background=0)."""

    return extract_foreground_mask_with_diagnostics(frame, background, settings).mask


def extract_foreground_mask_with_diagnostics(
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
) -> FrameSegmentationDiagnostics:
    """Return final mask plus intermediate diagnostic images (same science)."""

    validated = settings.validated()
    return _extract_robust_foreground_mask(
        frame,
        background,
        threshold=validated.blob_threshold,
        kernel_size=validated.blob_kernel_size,
        min_area=validated.blob_min_area,
        blur_kernel_size=validated.blob_blur_kernel_size,
        tail_open_kernel_size=validated.blob_tail_open_kernel_size,
        max_area=validated.blob_max_area,
        prune_min_radius=validated.blob_prune_min_radius,
        prune_regrow_iters=validated.blob_prune_regrow_iters,
        prune_max_regrow_dist=validated.blob_prune_max_regrow_dist,
        post_prune_min_area=validated.blob_post_prune_min_area,
        post_prune_max_area=validated.blob_post_prune_max_area,
        final_smooth_kernel_size=validated.blob_final_smooth_kernel_size,
        abs_dark_threshold=validated.abs_dark_threshold,
    )


def _preview_warnings(topology: FrameSegmentationResult) -> tuple[str, ...]:
    warnings: list[str] = []
    if topology.cc_count_raw == 0:
        warnings.append("No connected components detected (cc_count_raw=0).")
    elif topology.cc_count_raw > 2:
        warnings.append(
            f"Fragmented detection: cc_count_raw={topology.cc_count_raw} "
            "(effective cc_count invalidated)."
        )
    return tuple(warnings)


def extract_contour_topology(mask_binary: np.ndarray) -> FrameSegmentationResult:
    """Extract legacy connected-component topology from a binary mask."""

    mask = np.asarray(mask_binary)
    if mask.ndim != 2:
        raise BlobFeaturesError(f"Binary mask must be 2-D; got shape {mask.shape}.")
    if mask.size == 0:
        raise BlobFeaturesError("Binary mask must not be empty.")

    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_tuple = tuple(contours)
    sorted_c = sorted(contour_tuple, key=cv2.contourArea, reverse=True)

    cc_count_raw = int(len(contour_tuple))
    cc_count: float = float(min(cc_count_raw, 2))
    min_inter = 0.0

    if cc_count >= 2:
        min_inter = _min_inter_contour_dist(sorted_c[0], sorted_c[1])

    # Fragmentation (more than two blobs): invalidate effective count / distance.
    if cc_count_raw > 2:
        cc_count = float("nan")
        min_inter = float("nan")

    return FrameSegmentationResult(
        mask=mask_u8,
        contours=contour_tuple,
        cc_count_raw=cc_count_raw,
        cc_count=cc_count,
        min_inter_contour_dist=float(min_inter),
    )


def _validate_frame_background(frame: np.ndarray, background: np.ndarray) -> None:
    frame_arr = np.asarray(frame)
    bg_arr = np.asarray(background)
    if frame_arr.size == 0 or bg_arr.size == 0:
        raise BlobFeaturesError("Frame and background must be non-empty arrays.")
    if frame_arr.ndim not in (2, 3) or bg_arr.ndim not in (2, 3):
        raise BlobFeaturesError(
            f"Unsupported frame/background ndim: frame={frame_arr.ndim}, "
            f"background={bg_arr.ndim}."
        )
    if frame_arr.shape[:2] != bg_arr.shape[:2]:
        raise BlobFeaturesError(
            f"Frame/background shape mismatch: frame={frame_arr.shape}, "
            f"background={bg_arr.shape}."
        )


def _require_skimage() -> Callable[..., Any]:
    if _SKIMAGE_IMPORT_ERROR is not None or _threshold_multiotsu is None:
        raise BlobFeaturesError(
            "Multi-Otsu thresholding (blob_threshold=-2) requires scikit-image. "
            "Install with: pip install scikit-image"
        ) from _SKIMAGE_IMPORT_ERROR
    return _threshold_multiotsu


def _odd(value: int, minimum: int = 3) -> int:
    """Return the nearest odd integer >= minimum (legacy helper)."""

    value = max(minimum, int(round(value)))
    return value if value % 2 == 1 else value + 1


def _filter_components_by_area(mask: np.ndarray, min_area: int, max_area: int = 0) -> np.ndarray:
    """Keep only connected components within the requested area range."""

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)

    lower = max(1, int(min_area))
    upper = int(max_area) if int(max_area) > 0 else np.iinfo(np.int32).max

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if lower <= area <= upper:
            filtered[labels == label] = 255

    return filtered


def _prune_thin_appendages_by_thickness(
    mask: np.ndarray,
    min_radius: int,
    regrow_iters: int,
    max_regrow_dist: int,
) -> np.ndarray:
    """Remove thin appendages by keeping a thick core and regrowing locally."""

    binary = (mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return mask

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    core = np.zeros_like(mask, dtype=np.uint8)
    core[dist >= float(min_radius)] = 255

    if not np.any(core):
        return mask

    core_binary = (core > 0).astype(np.uint8)
    inv_core = 1 - core_binary
    dist_from_core = cv2.distanceTransform(inv_core, cv2.DIST_L2, 5)

    allowed_region = np.zeros_like(mask, dtype=np.uint8)
    allowed_region[dist_from_core <= float(max_regrow_dist)] = 255

    current = core.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(0, int(regrow_iters))):
        dilated = cv2.dilate(current, kernel, iterations=1)
        current = cast(
            np.ndarray,
            cv2.bitwise_and(dilated, mask),
        ).astype(np.uint8)
        current = cast(
            np.ndarray,
            cv2.bitwise_and(current, allowed_region),
        ).astype(np.uint8)

    return current


def _extract_robust_foreground_mask(
    frame: np.ndarray,
    background: np.ndarray,
    threshold: int,
    kernel_size: int,
    min_area: int,
    blur_kernel_size: int,
    tail_open_kernel_size: int,
    max_area: int,
    prune_min_radius: int,
    prune_regrow_iters: int,
    prune_max_regrow_dist: int,
    post_prune_min_area: int,
    post_prune_max_area: int,
    final_smooth_kernel_size: int,
    abs_dark_threshold: int,
) -> FrameSegmentationDiagnostics:
    """Legacy robust foreground mask (mice=255, background=0)."""

    if frame.shape[:2] != background.shape[:2]:
        raise BlobFeaturesError(
            f"Frame/background shape mismatch: frame={frame.shape}, background={background.shape}"
        )

    working_background = background
    if frame.ndim != working_background.ndim:
        if frame.ndim == 3 and working_background.ndim == 2:
            working_background = cv2.cvtColor(working_background, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 2 and working_background.ndim == 3:
            working_background = cv2.cvtColor(working_background, cv2.COLOR_BGR2GRAY)

    diff = cv2.subtract(working_background, frame)
    gray = diff if diff.ndim == 2 else cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    if threshold >= 0:
        _, binary = cv2.threshold(blurred, int(threshold), 255, cv2.THRESH_BINARY)
    elif threshold == -1:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold == -2:
        multiotsu = _require_skimage()
        try:
            thresholds = multiotsu(blurred, classes=3)
            strict_body_threshold = thresholds[1]
            _, binary = cv2.threshold(blurred, strict_body_threshold, 255, cv2.THRESH_BINARY)
        except ValueError:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    frame_gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    abs_dark_mask = np.zeros_like(frame_gray, dtype=np.uint8)
    abs_dark_mask[frame_gray < abs_dark_threshold] = 255
    binary = cv2.bitwise_and(binary, abs_dark_mask)
    binary_pre_morph = binary.copy()

    k_close = max(1, int(kernel_size))
    if k_close % 2 == 0:
        k_close += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tail_open_size = _odd(max(1, int(tail_open_kernel_size)), minimum=1)
    tail_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tail_open_size, tail_open_size))

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, tail_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, open_kernel)

    mask = _filter_components_by_area(cleaned, min_area=min_area, max_area=max_area)
    mask = _prune_thin_appendages_by_thickness(
        mask,
        min_radius=max(1, int(prune_min_radius)),
        regrow_iters=max(0, int(prune_regrow_iters)),
        max_regrow_dist=max(0, int(prune_max_regrow_dist)),
    )
    mask = _filter_components_by_area(
        mask, min_area=post_prune_min_area, max_area=post_prune_max_area
    )

    smooth_size = _odd(max(1, int(final_smooth_kernel_size)), minimum=1)
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_size, smooth_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_kernel)
    return FrameSegmentationDiagnostics(
        foreground_gray=np.ascontiguousarray(gray, dtype=np.uint8),
        binary_pre_morph=np.ascontiguousarray(binary_pre_morph, dtype=np.uint8),
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
    )


def _contour_points(contour: np.ndarray) -> np.ndarray:
    return contour.reshape(-1, 2).astype(np.float32)


def _min_inter_contour_dist(c1: np.ndarray, c2: np.ndarray) -> float:
    pts1 = _contour_points(c1)
    pts2 = _contour_points(c2)
    if pts1.size == 0 or pts2.size == 0:
        return 0.0
    tree2 = cKDTree(pts2)
    dists, _ = tree2.query(pts1, k=1)
    return float(np.min(dists))
