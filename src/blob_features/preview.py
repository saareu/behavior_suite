"""Headless single-frame / multi-frame segmentation preview helpers.

GUI and batch paths share ``segment_frame_with_diagnostics`` so preview topology
matches full extraction. Temporal gap-fill and bout features are intentionally
omitted for isolated bookmarked frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from blob_features.config import BlobSegmentationSettings
from blob_features.contracts import BlobFeaturesError
from blob_features.segmentation import (
    FrameSegmentationPreview,
    segment_frame,
    segment_frame_with_diagnostics,
)
from preprocess.exceptions import VideoProbeError
from preprocess.video_probe import read_raw_frame_at_index


def read_prepared_video_frame(video_path: Path | str, prepared_frame_idx: int) -> np.ndarray:
    """Decode one prepared-video frame by prepared-frame index (not 120 Hz)."""

    path = Path(video_path)
    try:
        return read_raw_frame_at_index(path, int(prepared_frame_idx))
    except VideoProbeError as exc:
        raise BlobFeaturesError(
            f"Could not decode prepared-video frame {prepared_frame_idx}: {exc}"
        ) from exc


def load_cropped_background(background_path: Path | str) -> np.ndarray:
    path = Path(background_path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise BlobFeaturesError(f"Could not read cropped background: {path}")
    return image


def preview_prepared_frame(
    *,
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
    prepared_frame_idx: int,
) -> FrameSegmentationPreview:
    """Run the shared segmentation core for one prepared frame."""

    return segment_frame_with_diagnostics(
        frame,
        background,
        settings,
        prepared_frame_idx=prepared_frame_idx,
    )


def preview_prepared_frame_from_paths(
    *,
    prepared_video: Path | str,
    cropped_background: Path | str,
    settings: BlobSegmentationSettings,
    prepared_frame_idx: int,
) -> FrameSegmentationPreview:
    frame = read_prepared_video_frame(prepared_video, prepared_frame_idx)
    background = load_cropped_background(cropped_background)
    return preview_prepared_frame(
        frame=frame,
        background=background,
        settings=settings,
        prepared_frame_idx=prepared_frame_idx,
    )


def preview_prepared_frames(
    *,
    prepared_video: Path | str,
    cropped_background: Path | str,
    settings: BlobSegmentationSettings,
    prepared_frame_indices: Sequence[int],
) -> tuple[FrameSegmentationPreview, ...]:
    """Preview many prepared frames under one settings snapshot.

    Does not compute source-sequence temporal features from the isolated set.
    """

    validated = settings.validated()
    background = load_cropped_background(cropped_background)
    results: list[FrameSegmentationPreview] = []
    for idx in prepared_frame_indices:
        frame = read_prepared_video_frame(prepared_video, int(idx))
        results.append(
            preview_prepared_frame(
                frame=frame,
                background=background,
                settings=validated,
                prepared_frame_idx=int(idx),
            )
        )
    return tuple(results)


def draw_contour_overlay(
    frame: np.ndarray,
    contours: Sequence[np.ndarray],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a BGR copy of ``frame`` with contours overlaid."""

    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        canvas = frame.copy()
    elif frame.ndim == 3 and frame.shape[2] == 4:
        canvas = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        raise BlobFeaturesError(f"Unsupported frame shape for overlay: {frame.shape}")
    contour_list = [np.asarray(contour) for contour in contours]
    if contour_list:
        cv2.drawContours(canvas, contour_list, -1, color, int(thickness))
    return canvas


def assert_preview_matches_batch_core(
    frame: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
) -> None:
    """Raise AssertionError when preview diagnostics diverge from ``segment_frame``."""

    batch = segment_frame(frame, background, settings)
    preview = segment_frame_with_diagnostics(frame, background, settings)
    if not np.array_equal(batch.mask, preview.mask):
        raise AssertionError("Preview mask diverges from segment_frame mask.")
    if batch.cc_count_raw != preview.cc_count_raw:
        raise AssertionError("Preview cc_count_raw diverges from segment_frame.")
    if not (
        (np.isnan(batch.cc_count) and np.isnan(preview.cc_count))
        or batch.cc_count == preview.cc_count
    ):
        raise AssertionError("Preview cc_count diverges from segment_frame.")
