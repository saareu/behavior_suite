"""S1 eligibility for Subsystem 05 (independent of S2–S4)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from blob_features import BlobFeaturesError, validate_s1_for_blob_features
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


def _write_eligible_s1(
    root: Path,
    *,
    frame_count: int = 4,
    fps: float = 20.0,
    size: tuple[int, int] = (32, 24),
    write_s2: bool = False,
) -> Path:
    """Create a minimal eligible preprocess tree. Optionally add empty S2 dir."""

    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    width, height = size

    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame_idx in range(frame_count):
        writer.write(np.full((height, width, 3), frame_idx % 256, dtype=np.uint8))
    writer.release()

    background = np.full((height, width, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)

    (preprocess / "prepare_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "width": width,
                    "height": height,
                    "opencv_reported_frame_count": frame_count,
                    "opencv_readable_frame_count": frame_count,
                    "frame_count_used_for_sleap": frame_count,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
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

    if write_s2:
        (session / "pose_inference" / "dummy_run").mkdir(parents=True)

    return session


def test_valid_s1_input(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    eligible = validate_s1_for_blob_features(session)
    assert eligible.frame_count == 4
    assert eligible.frame_width == 32
    assert eligible.frame_height == 24
    assert eligible.prepared_video.is_file()
    assert eligible.cropped_background.is_file()
    assert eligible.source_timing.frame_count == 4


def test_valid_from_preprocess_dir(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    eligible = validate_s1_for_blob_features(session / "preprocess")
    assert eligible.session_root == session.resolve()


def test_independent_of_s2_s4(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path, write_s2=False)
    # No pose_inference / tracking_correction / pose_finalization directories.
    assert not (session / "pose_inference").exists()
    assert not (session / "tracking_correction").exists()
    assert not (session / "pose_finalization").exists()
    eligible = validate_s1_for_blob_features(session)
    assert eligible.preprocess_dir.is_dir()

    # Presence of unrelated S2 artifacts must not be required or block S5.
    session_with_s2 = _write_eligible_s1(tmp_path / "with_s2", write_s2=True)
    eligible_with_s2 = validate_s1_for_blob_features(session_with_s2)
    assert eligible_with_s2.frame_count == 4


@pytest.mark.parametrize(
    "missing_name",
    [
        "prepared_video.mp4",
        "cropped_background.png",
        "prepared_sync.npz",
        "prepare_meta.json",
    ],
)
def test_missing_required_artifact(tmp_path: Path, missing_name: str) -> None:
    session = _write_eligible_s1(tmp_path)
    (session / "preprocess" / missing_name).unlink()
    with pytest.raises(BlobFeaturesError, match="Missing required"):
        validate_s1_for_blob_features(session)


def test_frame_count_mismatch(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path, frame_count=4)
    # Rewrite sync with a different authoritative count.
    sync = build_prepared_sync(
        prepared_frame_count=3,
        start_frame=0,
        end_frame_exclusive=3,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.arange(3, dtype=float) / 20.0,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=3,
        prepared_frame_count_opencv_reported=3,
        prepared_frame_count_opencv_readable=3,
    )
    write_prepared_sync_npz(session / "preprocess" / "prepared_sync.npz", sync)
    meta_path = session / "preprocess" / "prepare_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["prepared_video"]["frame_count_used_for_sleap"] = 3
    meta["prepared_video"]["opencv_reported_frame_count"] = 3
    meta["prepared_video"]["opencv_readable_frame_count"] = 3
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    with pytest.raises(BlobFeaturesError, match="frame count"):
        validate_s1_for_blob_features(session)


def test_background_size_mismatch(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path, size=(32, 24))
    bad_bg = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(session / "preprocess" / "cropped_background.png"), bad_bg)
    with pytest.raises(BlobFeaturesError, match="cropped_background.png size"):
        validate_s1_for_blob_features(session)


def test_unreadable_background(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    (session / "preprocess" / "cropped_background.png").write_bytes(b"not-a-png")
    with pytest.raises(BlobFeaturesError, match="cropped_background.png is unreadable"):
        validate_s1_for_blob_features(session)
