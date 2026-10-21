"""Per-frame segmentation core adapted from legacy Pipeline 40."""

from __future__ import annotations

import numpy as np
import pytest

from blob_features import (
    BlobFeaturesError,
    BlobSegmentationSettings,
    extract_contour_topology,
    segment_frame,
)


def _preview_settings(**overrides: int) -> BlobSegmentationSettings:
    """Lab defaults with area thresholds lowered for small synthetic fixtures."""

    values = {
        "blob_threshold": -1,  # Otsu; avoids Multi-Otsu edge cases on tiny images
        "blob_blur_kernel_size": 5,
        "blob_kernel_size": 5,
        "blob_tail_open_kernel_size": 3,
        "blob_min_area": 20,
        "blob_max_area": 0,
        "blob_prune_min_radius": 1,
        "blob_prune_regrow_iters": 1,
        "blob_prune_max_regrow_dist": 2,
        "blob_post_prune_min_area": 20,
        "blob_post_prune_max_area": 0,
        "blob_final_smooth_kernel_size": 3,
        "abs_dark_threshold": 250,  # allow dark synthetic mice on bright bg
    }
    values.update(overrides)
    return BlobSegmentationSettings(**values).validated()


def _two_dark_blobs_on_bright_bg(
    *,
    height: int = 80,
    width: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    background = np.full((height, width, 3), 220, dtype=np.uint8)
    frame = background.copy()
    # Two separated dark rectangles (mice).
    frame[20:45, 15:40] = 5
    frame[20:45, 75:100] = 5
    return frame, background


def test_detects_two_components() -> None:
    frame, background = _two_dark_blobs_on_bright_bg()
    result = segment_frame(frame, background, _preview_settings())
    assert result.mask.shape == frame.shape[:2]
    assert result.mask.dtype == np.uint8
    assert result.cc_count_raw == 2
    assert result.cc_count == pytest.approx(2.0)
    assert np.isfinite(result.min_inter_contour_dist)
    assert result.min_inter_contour_dist > 0.0
    assert len(result.contours) == 2


def test_single_component() -> None:
    background = np.full((60, 60, 3), 220, dtype=np.uint8)
    frame = background.copy()
    frame[15:45, 15:45] = 5
    result = segment_frame(frame, background, _preview_settings())
    assert result.cc_count_raw == 1
    assert result.cc_count == pytest.approx(1.0)
    assert result.min_inter_contour_dist == pytest.approx(0.0)


def test_fragmentation_invalidates_effective_count() -> None:
    background = np.full((80, 120, 3), 220, dtype=np.uint8)
    frame = background.copy()
    frame[10:25, 10:25] = 5
    frame[10:25, 50:65] = 5
    frame[10:25, 90:105] = 5
    result = segment_frame(frame, background, _preview_settings())
    assert result.cc_count_raw == 3
    assert np.isnan(result.cc_count)
    assert np.isnan(result.min_inter_contour_dist)


def test_multiotsu_threshold_path() -> None:
    frame, background = _two_dark_blobs_on_bright_bg()
    settings = _preview_settings(blob_threshold=-2)
    result = segment_frame(frame, background, settings)
    assert result.cc_count_raw >= 1
    assert result.mask.max() == 255


def test_fixed_threshold_path() -> None:
    frame, background = _two_dark_blobs_on_bright_bg()
    settings = _preview_settings(blob_threshold=30)
    result = segment_frame(frame, background, settings)
    assert result.cc_count_raw == 2


def test_shape_mismatch() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    background = np.full((40, 50, 3), 200, dtype=np.uint8)
    with pytest.raises(BlobFeaturesError, match="shape mismatch"):
        segment_frame(frame, background, _preview_settings())


def test_empty_arrays() -> None:
    frame = np.zeros((0, 0, 3), dtype=np.uint8)
    background = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(BlobFeaturesError, match="non-empty"):
        segment_frame(frame, background, _preview_settings())


def test_grayscale_frame_and_background() -> None:
    background = np.full((60, 60), 220, dtype=np.uint8)
    frame = background.copy()
    frame[20:40, 20:40] = 5
    result = segment_frame(frame, background, _preview_settings())
    assert result.cc_count_raw == 1


def test_extract_contour_topology_rejects_bad_mask() -> None:
    with pytest.raises(BlobFeaturesError, match="2-D"):
        extract_contour_topology(np.zeros((4, 4, 3), dtype=np.uint8))


def test_identical_frame_and_background_yields_empty_mask() -> None:
    image = np.full((50, 50, 3), 180, dtype=np.uint8)
    result = segment_frame(image, image, _preview_settings())
    assert result.cc_count_raw == 0
    assert result.cc_count == pytest.approx(0.0)
    assert int(result.mask.sum()) == 0
