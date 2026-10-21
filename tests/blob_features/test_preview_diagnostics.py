"""Preview diagnostics share the scientific segmentation core."""

from __future__ import annotations

import numpy as np
import pytest

from blob_features import (
    BlobSegmentationSettings,
    segment_frame,
    segment_frame_with_diagnostics,
)
from blob_features.preview import assert_preview_matches_batch_core


def _preview_settings(**overrides: int) -> BlobSegmentationSettings:
    values = {
        "blob_threshold": -1,
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
        "abs_dark_threshold": 250,
    }
    values.update(overrides)
    return BlobSegmentationSettings(**values).validated()


def _two_blobs() -> tuple[np.ndarray, np.ndarray]:
    background = np.full((80, 120, 3), 220, dtype=np.uint8)
    frame = background.copy()
    frame[20:45, 15:40] = 5
    frame[20:45, 75:100] = 5
    return frame, background


def test_preview_mask_matches_segment_frame() -> None:
    frame, background = _two_blobs()
    settings = _preview_settings()
    batch = segment_frame(frame, background, settings)
    preview = segment_frame_with_diagnostics(frame, background, settings)
    assert np.array_equal(batch.mask, preview.mask)
    assert batch.cc_count_raw == preview.cc_count_raw
    assert batch.cc_count == pytest.approx(preview.cc_count)
    assert preview.foreground_gray.shape == frame.shape[:2]
    assert preview.binary_pre_morph.shape == frame.shape[:2]
    assert_preview_matches_batch_core(frame, background, settings)


def test_preview_exposes_distinct_raw_and_effective_counts() -> None:
    background = np.full((80, 120, 3), 220, dtype=np.uint8)
    frame = background.copy()
    frame[10:25, 10:25] = 5
    frame[10:25, 50:65] = 5
    frame[10:25, 90:105] = 5
    preview = segment_frame_with_diagnostics(frame, background, _preview_settings())
    assert preview.cc_count_raw == 3
    assert np.isnan(preview.cc_count)
    assert any("Fragmented" in warning for warning in preview.warnings)
