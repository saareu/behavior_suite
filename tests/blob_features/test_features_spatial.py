"""Spatial occlusion morphology features (merged-blob frames)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from blob_features import (
    BlobSegmentationSettings,
    compute_frame_features,
    compute_frame_features_from_segmentation,
    extract_contour_topology,
    extract_occlusion_static_features,
)


def _merged_ellipse_mask(
    *,
    height: int = 120,
    width: int = 160,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (80, 60), (45, 22), 0, 0, 360, 255, thickness=-1)
    return mask


def test_occlusion_static_features_on_merged_mask() -> None:
    mask = _merged_ellipse_mask()
    features = extract_occlusion_static_features(mask)
    assert np.isfinite(features["ellipse_eccentricity"])
    assert features["ellipse_eccentricity"] > 1.0
    assert np.isfinite(features["circularity"])
    assert 0.0 < features["circularity"] <= 1.0 + 1e-6
    assert np.isfinite(features["solidity"])
    assert 0.0 < features["solidity"] <= 1.0 + 1e-6
    assert np.isfinite(features["num_dt_peaks"])
    assert features["num_dt_peaks"] >= 1.0
    assert np.isfinite(features["furthest_endpoint_distance_px"])
    assert features["furthest_endpoint_distance_px"] > 0.0


def test_occlusion_features_only_when_cc_count_is_one() -> None:
    background = np.full((100, 140, 3), 220, dtype=np.uint8)
    frame_one = background.copy()
    cv2.ellipse(frame_one, (70, 50), (30, 18), 0, 0, 360, (5, 5, 5), thickness=-1)
    settings = BlobSegmentationSettings(
        blob_threshold=-1,
        blob_blur_kernel_size=5,
        blob_kernel_size=5,
        blob_tail_open_kernel_size=3,
        blob_min_area=50,
        blob_max_area=0,
        blob_prune_min_radius=1,
        blob_prune_regrow_iters=1,
        blob_prune_max_regrow_dist=2,
        blob_post_prune_min_area=50,
        blob_post_prune_max_area=0,
        blob_final_smooth_kernel_size=3,
        abs_dark_threshold=250,
    ).validated()

    one = compute_frame_features(frame_one, background, settings)
    assert one.cc_count == pytest.approx(1.0)
    assert np.isfinite(one.occlusion_static["circularity"])

    frame_two = background.copy()
    frame_two[20:45, 15:40] = 5
    frame_two[20:45, 90:115] = 5
    two = compute_frame_features(frame_two, background, settings)
    assert two.cc_count == pytest.approx(2.0)
    for name, value in two.occlusion_static.items():
        assert not np.isfinite(value), name


def test_empty_mask_occlusion_is_nan() -> None:
    empty = np.zeros((40, 40), dtype=np.uint8)
    features = extract_occlusion_static_features(empty)
    for name, value in features.items():
        assert not np.isfinite(value), name


def test_fragmentation_keeps_occlusion_nan() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[10:25, 10:25] = 255
    mask[10:25, 50:65] = 255
    mask[10:25, 90:105] = 255
    topology = extract_contour_topology(mask)
    assert topology.cc_count_raw == 3
    assert not np.isfinite(topology.cc_count)
    result = compute_frame_features_from_segmentation(topology)
    for value in result.occlusion_static.values():
        assert not np.isfinite(value)
