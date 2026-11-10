"""Local synthetic timing smoke for S5 cancel-poll overhead (not a lab claim)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from blob_features import (
    BlobExecutionSettings,
    BlobFeaturesConfig,
    BlobFeaturesRequest,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    run_blob_feature_extraction,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


def _soft_config() -> BlobFeaturesConfig:
    return BlobFeaturesConfig(
        segmentation=BlobSegmentationSettings(
            blob_threshold=-1,
            blob_blur_kernel_size=5,
            blob_kernel_size=5,
            blob_tail_open_kernel_size=3,
            blob_min_area=40,
            blob_max_area=0,
            blob_prune_min_radius=1,
            blob_prune_regrow_iters=1,
            blob_prune_max_regrow_dist=2,
            blob_post_prune_min_area=40,
            blob_post_prune_max_area=0,
            blob_final_smooth_kernel_size=3,
            abs_dark_threshold=250,
        ).validated(),
        temporal=BlobTemporalSettings(cc_count_gap_fill_max_gap=2).validated(),
        execution=BlobExecutionSettings(n_workers=1, max_in_flight=4).validated(),
    ).validated()


def _write_synthetic_s1(root: Path, *, frame_count: int = 30) -> Path:
    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    width, height = 120, 80
    background = np.full((height, width, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)
    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    assert writer.isOpened()
    for _ in range(frame_count):
        frame = background.copy()
        frame[20:45, 15:40] = 5
        frame[20:45, 75:100] = 5
        writer.write(frame)
    writer.release()
    meta: dict[str, Any] = {
        "schema_version": "prepare_meta_v1",
        "prepared_video": {
            "width": width,
            "height": height,
            "opencv_reported_frame_count": frame_count,
            "opencv_readable_frame_count": frame_count,
            "frame_count_used_for_sleap": frame_count,
        },
    }
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(meta) + "\n", encoding="utf-8"
    )
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / 20.0,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    return session


def main() -> None:
    root = Path(tempfile.mkdtemp())
    session = _write_synthetic_s1(root, frame_count=30)
    cancel_calls = {"n": 0}

    def cancel() -> bool:
        cancel_calls["n"] += 1
        return False

    # Warmup so the comparison is not dominated by first-touch imports/codecs.
    run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="perf",
            timestamp="20260101T199999",
        )
    )
    t0 = time.perf_counter()
    run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="perf",
            timestamp="20260101T200000",
        ),
        cancellation_requested=cancel,
    )
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="perf",
            timestamp="20260101T200001",
        )
    )
    t3 = time.perf_counter()
    print(f"with_cancel_poll_sec={t1 - t0:.4f} cancel_checks={cancel_calls['n']}")
    print(f"without_callback_sec={t3 - t2:.4f}")


if __name__ == "__main__":
    main()
