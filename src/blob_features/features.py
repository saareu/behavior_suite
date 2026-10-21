"""Per-frame occlusion morphology and source-rate temporal feature processing.

Ports the scientific helpers from ``legacy/pipelines/40_extract_blob_features.py``.
All temporal calculations operate in prepared-video frame space before mapping
onto the authoritative 120 Hz target timeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from blob_features.config import BlobSegmentationSettings
from blob_features.contracts import (
    OCCLUSION_STATIC_FEATURE_COLS,
    OCCLUSION_TEMPORAL_FEATURE_COLS,
    BlobFeaturesError,
)
from blob_features.segmentation import FrameSegmentationResult, segment_frame
from blob_features.timing import NearestNeighborTimingMap

_peak_local_max: Callable[..., Any] | None
_skeletonize: Callable[..., Any] | None
_sk_label: Callable[..., Any] | None
try:
    from skimage.feature import peak_local_max as _imported_peak_local_max
    from skimage.measure import label as _imported_sk_label
    from skimage.morphology import skeletonize as _imported_skeletonize

    _peak_local_max = _imported_peak_local_max
    _skeletonize = _imported_skeletonize
    _sk_label = _imported_sk_label
    _SKIMAGE_IMPORT_ERROR: BaseException | None = None
except ImportError as exc:  # pragma: no cover - exercised via require helper
    _peak_local_max = None
    _skeletonize = None
    _sk_label = None
    _SKIMAGE_IMPORT_ERROR = exc

SOURCE_FEATURE_COLS = (
    "cc_count",
    "cc_count_raw",
    "min_inter_contour_dist",
    *OCCLUSION_STATIC_FEATURE_COLS,
)


@dataclass(frozen=True, slots=True)
class FrameFeatureResult:
    """Topology plus occlusion static features for one prepared-video frame."""

    mask: np.ndarray
    contours: tuple[np.ndarray, ...]
    cc_count_raw: int
    cc_count: float
    min_inter_contour_dist: float
    occlusion_static: Mapping[str, float]
    occlusion_static_error: bool

    def as_record(self, *, frame_idx: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "cc_count": float(self.cc_count),
            "cc_count_raw": int(self.cc_count_raw),
            "min_inter_contour_dist": float(self.min_inter_contour_dist),
        }
        for name in OCCLUSION_STATIC_FEATURE_COLS:
            record[name] = float(self.occlusion_static[name])
        return record


def compute_frame_features(
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
) -> FrameFeatureResult:
    """Segment one frame and compute topology + occlusion static features."""

    topology = segment_frame(frame, background, settings)
    return compute_frame_features_from_segmentation(topology)


def compute_frame_features_from_segmentation(
    topology: FrameSegmentationResult,
) -> FrameFeatureResult:
    """Compute occlusion static features from an existing segmentation result."""

    occlusion = {name: float("nan") for name in OCCLUSION_STATIC_FEATURE_COLS}
    occlusion_error = False
    if topology.cc_count == 1.0 and np.isfinite(topology.cc_count):
        try:
            occlusion = extract_occlusion_static_features(topology.mask)
        except Exception:
            occlusion_error = True
            occlusion = {name: float("nan") for name in OCCLUSION_STATIC_FEATURE_COLS}

    return FrameFeatureResult(
        mask=topology.mask,
        contours=topology.contours,
        cc_count_raw=topology.cc_count_raw,
        cc_count=topology.cc_count,
        min_inter_contour_dist=topology.min_inter_contour_dist,
        occlusion_static=occlusion,
        occlusion_static_error=occlusion_error,
    )


def extract_occlusion_static_features(mask_image: np.ndarray) -> dict[str, float]:
    """Extract static morphology features from one merged-blob mask."""

    binary_mask = _largest_connected_component(_binarize_mask(mask_image))
    smoothed_mask = _smooth_occlusion_mask(binary_mask)
    distance_info = _distance_transform_features(smoothed_mask)
    geometry_info = _geometry_features(smoothed_mask)
    skeleton_info = _skeleton_features(smoothed_mask)
    return {
        "ellipse_eccentricity": float(geometry_info["ellipse_eccentricity"]),
        "circularity": float(geometry_info["circularity"]),
        "solidity": float(geometry_info["solidity"]),
        "num_dt_peaks": float(distance_info["num_dt_peaks"]),
        "furthest_endpoint_distance_px": float(
            skeleton_info["furthest_endpoint_distance_px"]
        ),
    }


def impute_cc_count_and_min_inter_contour_dist(
    df: pd.DataFrame,
    *,
    max_gap_frames: int,
) -> pd.DataFrame:
    """Fill short internal NaN gaps only when flanking ``cc_count`` values match.

    Gaps between identical finite ``cc_count`` values are filled up to
    ``max_gap_frames``. Transitions between 1 and 2 remain NaN.
    ``min_inter_contour_dist`` is linearly interpolated over the same gaps.
    Operates in source prepared-video frame order.
    """

    if max_gap_frames <= 0:
        return df
    if "cc_count" not in df.columns or "min_inter_contour_dist" not in df.columns:
        return df

    out = df.copy()
    cc_count = pd.to_numeric(out["cc_count"], errors="coerce").astype(float)
    min_inter = pd.to_numeric(out["min_inter_contour_dist"], errors="coerce").astype(float)

    cc_values = cc_count.to_numpy(copy=True)
    dist_values = min_inter.to_numpy(copy=True)
    finite_idx = np.flatnonzero(np.isfinite(cc_values))
    if len(finite_idx) < 2:
        out["cc_count"] = cc_values
        out["min_inter_contour_dist"] = dist_values
        return out

    for left, right in zip(finite_idx[:-1], finite_idx[1:], strict=False):
        gap_len = int(right - left - 1)
        if gap_len <= 0 or gap_len > int(max_gap_frames):
            continue

        left_val = cc_values[left]
        right_val = cc_values[right]
        if not (np.isfinite(left_val) and np.isfinite(right_val)):
            continue
        if float(left_val) != float(right_val):
            continue

        cc_values[left + 1 : right] = left_val

        left_dist = dist_values[left]
        right_dist = dist_values[right]
        if np.isfinite(left_dist) and np.isfinite(right_dist):
            interp = np.linspace(
                float(left_dist), float(right_dist), gap_len + 2, dtype=np.float64
            )[1:-1]
            dist_values[left + 1 : right] = interp

    out["cc_count"] = cc_values
    out["min_inter_contour_dist"] = dist_values
    return out


def add_temporal_occlusion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bout-scoped rolling temporal features for merged-blob frames only.

    Windows (mean 5 / delta 5 / std 15) are counted in **source prepared-video
    frames**, not 120 Hz target samples. Non-occlusion rows remain NaN.
    """

    if "cc_count" not in df.columns:
        raise BlobFeaturesError(
            "add_temporal_occlusion_features requires a `cc_count` column."
        )

    out = df.copy()
    for col in OCCLUSION_TEMPORAL_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan

    occlusion_mask = out["cc_count"].eq(1)
    if not bool(occlusion_mask.any()):
        return out

    is_new_bout = occlusion_mask & ~occlusion_mask.shift(fill_value=False)
    out["bout_id"] = np.nan
    out.loc[occlusion_mask, "bout_id"] = (
        is_new_bout.cumsum().loc[occlusion_mask].astype(np.int64)
    )

    occlusion_slice = out.loc[
        occlusion_mask, ["bout_id", *OCCLUSION_STATIC_FEATURE_COLS]
    ].copy()
    grouped = occlusion_slice.groupby("bout_id", sort=False)

    for feature_name in OCCLUSION_STATIC_FEATURE_COLS:
        out.loc[occlusion_mask, f"{feature_name}_mean_5"] = grouped[feature_name].transform(
            lambda series: series.rolling(window=5, min_periods=1).mean()
        )
        out.loc[occlusion_mask, f"{feature_name}_delta_5"] = grouped[feature_name].transform(
            lambda series: series.diff(periods=5)
        )
        out.loc[occlusion_mask, f"{feature_name}_std_15"] = grouped[feature_name].transform(
            lambda series: series.rolling(window=15, min_periods=2).std()
        )

    return out.drop(columns=["bout_id"])


def map_source_features_to_target(
    df_source: pd.DataFrame,
    timing_map: NearestNeighborTimingMap,
) -> pd.DataFrame:
    """Nearest-neighbor map source-rate features onto the 120 Hz target timeline."""

    feature_cols = [
        col
        for col in (
            "cc_count",
            "cc_count_raw",
            "min_inter_contour_dist",
            *OCCLUSION_STATIC_FEATURE_COLS,
            *OCCLUSION_TEMPORAL_FEATURE_COLS,
        )
        if col in df_source.columns
    ]
    if df_source.empty:
        empty = {
            "sample_idx": np.asarray([], dtype=np.int64),
            "dense_frame_idx": np.asarray([], dtype=np.int64),
            "frame_idx": np.asarray([], dtype=np.int64),
            "time_sec": np.asarray([], dtype=np.float64),
            "orig_frame_idx": np.asarray([], dtype=np.int64),
        }
        for col in feature_cols:
            empty[col] = np.asarray([], dtype=np.float64)
        return pd.DataFrame(empty)

    if "frame_idx" not in df_source.columns:
        raise BlobFeaturesError(
            "Source feature table requires frame_idx (prepared-video index)."
        )

    df_indexed = (
        df_source.sort_values("frame_idx", kind="stable")
        .drop_duplicates(subset=["frame_idx"], keep="last")
        .set_index("frame_idx")
    )
    orig_idx = np.asarray(timing_map.orig_frame_idx, dtype=np.int64)
    feature_vals = df_indexed.reindex(orig_idx)[feature_cols].reset_index(drop=True)

    sample_idx = np.asarray(timing_map.target_frame_idx, dtype=np.int64)
    out = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "dense_frame_idx": sample_idx.copy(),
            "frame_idx": sample_idx.copy(),
            "time_sec": np.asarray(timing_map.target_time_sec, dtype=np.float64),
            "orig_frame_idx": orig_idx,
        }
    )
    for col in feature_cols:
        out[col] = feature_vals[col].to_numpy()
    return out


def build_source_feature_table(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Assemble ordered source-rate feature rows from per-frame records."""

    columns = ["frame_idx", *SOURCE_FEATURE_COLS]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(list(records))
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise BlobFeaturesError(
            "Source feature records missing columns: " + ", ".join(missing)
        )
    return frame.loc[:, columns].sort_values("frame_idx", kind="stable").reset_index(
        drop=True
    )


def apply_source_temporal_pipeline(
    df_source: pd.DataFrame,
    *,
    max_gap_frames: int,
) -> pd.DataFrame:
    """Legacy Pipeline 40 order: gap fill, then bout-scoped temporal features."""

    filled = impute_cc_count_and_min_inter_contour_dist(
        df_source,
        max_gap_frames=max_gap_frames,
    )
    return add_temporal_occlusion_features(filled)


def _require_skimage() -> None:
    if _SKIMAGE_IMPORT_ERROR is not None:
        raise BlobFeaturesError(
            "Occlusion morphology features require scikit-image. "
            "Install with: pip install scikit-image"
        ) from _SKIMAGE_IMPORT_ERROR


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(round(value)))
    return value if value % 2 == 1 else value + 1


def _coerce_grayscale(mask_image: np.ndarray) -> np.ndarray:
    array = np.asarray(mask_image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise BlobFeaturesError(f"Unsupported mask shape: {array.shape}")

    gray = np.nan_to_num(gray, nan=0.0)
    if gray.dtype == bool:
        return cast(np.ndarray, gray.astype(np.uint8) * 255)

    gray = gray.astype(np.float32)
    if gray.size == 0:
        return cast(np.ndarray, gray.astype(np.uint8))
    if gray.max(initial=0.0) <= 1.0:
        gray = gray * 255.0
    return cast(np.ndarray, np.clip(gray, 0, 255).astype(np.uint8))


def _binarize_mask(mask_image: np.ndarray) -> np.ndarray:
    gray = _coerce_grayscale(mask_image)
    if gray.size == 0 or gray.max(initial=0) == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    unique_values = np.unique(gray)
    if unique_values.size <= 2:
        return (gray > 0).astype(np.uint8)

    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def _largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if mask.size == 0 or mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return cast(np.ndarray, (labels == largest_label).astype(np.uint8))


def _largest_contour(binary_mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _adaptive_kernel(
    binary_mask: np.ndarray, fraction: float, minimum: int, maximum: int
) -> int:
    contour = _largest_contour(binary_mask)
    if contour is None:
        return _odd(minimum, minimum)

    _, _, width, height = cv2.boundingRect(contour)
    base = max(minimum, int(round(min(width, height) * fraction)))
    return min(maximum, _odd(base, minimum))


def _smooth_occlusion_mask(binary_mask: np.ndarray) -> np.ndarray:
    mask = _largest_connected_component(binary_mask)
    if mask.size == 0 or mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    mask_u8 = mask.astype(np.uint8) * 255
    blur_sigma = max(
        1.5, _adaptive_kernel(mask, fraction=0.03, minimum=3, maximum=9) / 2.5
    )
    blurred = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    _, hard = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_adaptive_kernel(mask, fraction=0.08, minimum=7, maximum=21),) * 2,
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_adaptive_kernel(mask, fraction=0.04, minimum=3, maximum=11),) * 2,
    )
    refine_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_adaptive_kernel(mask, fraction=0.05, minimum=3, maximum=15),) * 2,
    )

    smoothed = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, open_kernel, iterations=1)
    smoothed = cv2.dilate(smoothed, refine_kernel, iterations=1)
    smoothed = cv2.erode(smoothed, refine_kernel, iterations=1)
    return _largest_connected_component(smoothed > 0).astype(np.uint8)


def _adaptive_peak_distance(binary_mask: np.ndarray) -> int:
    contour = _largest_contour(binary_mask)
    if contour is None:
        return 5
    _, _, width, height = cv2.boundingRect(contour)
    return max(5, int(round(min(width, height) * 0.08)))


def _distance_transform_features(smoothed_mask: np.ndarray) -> dict[str, float]:
    _require_skimage()
    assert _peak_local_max is not None
    mask = (smoothed_mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return {"num_dt_peaks": float("nan")}

    distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak_coords = _peak_local_max(
        distance_transform,
        min_distance=_adaptive_peak_distance(mask),
        threshold_rel=0.25,
        labels=mask,
        exclude_border=False,
    )
    return {"num_dt_peaks": float(len(peak_coords))}


def _skeleton_neighbor_count(skeleton_mask: np.ndarray) -> np.ndarray:
    skeleton_u8 = (np.asarray(skeleton_mask) > 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    return (
        cv2.filter2D(skeleton_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        - skeleton_u8
    )


def _furthest_endpoint_pair(
    endpoint_coords_rc: np.ndarray,
) -> tuple[tuple[tuple[int, int], tuple[int, int]] | None, float]:
    if len(endpoint_coords_rc) < 2:
        return None, float("nan")

    max_distance = -1.0
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    for i in range(len(endpoint_coords_rc)):
        r1, c1 = endpoint_coords_rc[i]
        for j in range(i + 1, len(endpoint_coords_rc)):
            r2, c2 = endpoint_coords_rc[j]
            distance = float(np.hypot(c2 - c1, r2 - r1))
            if distance > max_distance:
                max_distance = distance
                best_pair = ((int(r1), int(c1)), (int(r2), int(c2)))

    return best_pair, max_distance if best_pair is not None else float("nan")


def _skeleton_features(smoothed_mask: np.ndarray) -> dict[str, float]:
    _require_skimage()
    assert _skeletonize is not None
    assert _sk_label is not None
    mask = smoothed_mask > 0
    if mask.sum() == 0:
        return {"furthest_endpoint_distance_px": float("nan")}

    skeleton = cast(np.ndarray, _skeletonize(mask))
    neighbor_count = _skeleton_neighbor_count(skeleton)
    branch_mask = skeleton & (neighbor_count >= 3)
    endpoint_mask = skeleton & (neighbor_count == 1)
    endpoint_coords = np.argwhere(endpoint_mask)

    # Trigger label import so missing skimage support fails here, not mid-pipeline.
    _ = int(_sk_label(branch_mask.astype(np.uint8), connectivity=2).max())
    _, furthest_endpoint_distance_px = _furthest_endpoint_pair(endpoint_coords)
    return {"furthest_endpoint_distance_px": float(furthest_endpoint_distance_px)}


def _geometry_features(smoothed_mask: np.ndarray) -> dict[str, float]:
    contour = _largest_contour(smoothed_mask)
    if contour is None or len(contour) < 3:
        return {
            "ellipse_eccentricity": float("nan"),
            "circularity": float("nan"),
            "solidity": float("nan"),
        }

    ellipse_eccentricity = float("nan")
    if len(contour) >= 5:
        try:
            (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            major_axis = max(axis_a, axis_b)
            minor_axis = min(axis_a, axis_b)
            ellipse_eccentricity = (
                float(major_axis / minor_axis) if minor_axis > 0 else float("nan")
            )
        except cv2.error:
            ellipse_eccentricity = float("nan")

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else float("nan")
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else float("nan")
    return {
        "ellipse_eccentricity": float(ellipse_eccentricity),
        "circularity": float(circularity),
        "solidity": float(solidity),
    }
