"""Frame-identity verification for media clip encoding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from results_packaging.media_csv_export.clip_bounds import compute_clip_bounds
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import SELECTION_BOUT_ID_LIST
from results_packaging.media_csv_export.runner import MediaExportRequest, run_media_export
from results_packaging.media_csv_export.video import encode_mp4_clip
from tests.results_packaging._export_fixtures import (
    build_mini_export_package,
    count_video_frames,
    patch_export_package_currently_eligible,
    write_synthetic_mp4,
)


def _decode_all(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _best_source_index(frame: np.ndarray, source_frames: list[np.ndarray]) -> int:
    best_i = -1
    best_d = 10**9
    target = frame.astype(np.int16)
    for index, src in enumerate(source_frames):
        diff = int(np.max(np.abs(target - src.astype(np.int16))))
        if diff < best_d:
            best_d = diff
            best_i = index
    return best_i


@pytest.mark.parametrize(
    ("start", "end", "pre", "post"),
    [
        (0, 0, 0.0, 0.0),  # first frame
        (89, 89, 0.0, 0.0),  # last frame
        (44, 45, 0.0, 0.0),  # middle
        (14, 15, 0.0, 0.0),  # non-keyframe-ish mid-GOP region
        (30, 30, 0.0, 0.0),  # single-frame bout
        (20, 22, 0.0, 0.0),  # short bout
        (40, 42, 0.1, 0.1),  # nonzero context padding
    ],
)
def test_frame_addressed_encode_preserves_source_identities(
    tmp_path: Path,
    start: int,
    end: int,
    pre: float,
    post: float,
) -> None:
    src = tmp_path / "source.mp4"
    write_synthetic_mp4(src, n_frames=90, fps=30.0)
    fps, n_frames = count_video_frames(src)
    assert n_frames == 90
    source_frames = _decode_all(src)
    assert len(source_frames) == n_frames

    bounds = compute_clip_bounds(
        prepared_frame_idx_start=start,
        prepared_frame_idx_end=end,
        source_fps=fps,
        source_n_frames=n_frames,
        pre_context_sec=pre,
        post_context_sec=post,
    )
    out = tmp_path / f"clip_{start}_{end}.mp4"
    encode_mp4_clip(source_video=src, output_path=out, bounds=bounds)

    encoded_fps, encoded_n = count_video_frames(out)
    assert encoded_n == bounds.clip_frame_count
    assert encoded_fps == pytest.approx(fps, rel=0.02)

    clip_frames = _decode_all(out)
    assert len(clip_frames) == bounds.clip_frame_count
    assert _best_source_index(clip_frames[0], source_frames) == bounds.clip_frame_idx_start
    assert _best_source_index(clip_frames[-1], source_frames) == bounds.clip_frame_idx_end


def test_media_job_clip_identities_and_context(tmp_path: Path) -> None:
    info = build_mini_export_package(tmp_path, n_frames=60, fps=30.0)
    source_frames = _decode_all(info["video_path"])
    bout_id = f"{info['package_run_id']}__bout_contact"
    with patch_export_package_currently_eligible():
        result = run_media_export(
            MediaExportRequest(
                session_root=info["session_root"],
                package_dir=info["package_dir"],
                timestamp="20260101T000030",
                selection=SelectionSpec(mode=SELECTION_BOUT_ID_LIST, bout_ids=(bout_id,)),
                pre_context_sec=0.1,
                post_context_sec=0.1,
            )
        )
    assert result.n_succeeded == 1
    row = pd.read_parquet(result.clip_index_path).iloc[0]
    clip_path = result.job_dir / "clips" / Path(str(row["clip_relpath"]))
    clip_frames = _decode_all(clip_path)
    assert len(clip_frames) == int(row["clip_frame_count"])
    assert _best_source_index(clip_frames[0], source_frames) == int(
        row["clip_frame_idx_start"]
    )
    assert _best_source_index(clip_frames[-1], source_frames) == int(
        row["clip_frame_idx_end"]
    )
    # Context padding applied in source fps; behavioral prepared span unchanged.
    assert int(row["prepared_frame_idx_start"]) == 10
    assert int(row["prepared_frame_idx_end"]) == 13
    assert int(row["clip_frame_idx_start"]) < int(row["prepared_frame_idx_start"])
    assert int(row["clip_frame_idx_end"]) > int(row["prepared_frame_idx_end"])
