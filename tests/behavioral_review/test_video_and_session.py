"""Prepared-video access and read-only inspection session tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from behavioral_review.contracts import SessionError, VideoAccessError
from behavioral_review.session import open_inspection_session
from behavioral_review.video import PreparedFrameAccessor
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
    write_tiny_avi,
)


def test_first_last_and_non_keyframe_frame_access(tmp_path: Path) -> None:
    video = write_tiny_avi(tmp_path / "clip.avi", frame_count=6, fps=10.0)
    accessor = PreparedFrameAccessor(video, cache_size=2)

    first = accessor.read_frame(0)
    mid = accessor.read_frame(3)
    last = accessor.read_frame(5)
    assert first.prepared_frame_idx == 0
    assert mid.prepared_frame_idx == 3
    assert last.prepared_frame_idx == 5
    assert first.image_bgr.shape[:2] == (24, 32)
    # Frames are distinguishable by mean intensity.
    assert float(first.image_bgr.mean()) != float(mid.image_bgr.mean())
    assert float(mid.image_bgr.mean()) != float(last.image_bgr.mean())


def test_frame_cache_deterministic_and_bounded(tmp_path: Path) -> None:
    video = write_tiny_avi(tmp_path / "clip.avi", frame_count=5, fps=10.0)
    calls: list[int] = []

    def reader(path: Path, idx: int) -> np.ndarray:
        calls.append(idx)
        return np.full((24, 32, 3), idx, dtype=np.uint8)

    accessor = PreparedFrameAccessor(video, cache_size=2, frame_reader=reader)
    a1 = accessor.read_frame(1)
    a2 = accessor.read_frame(1)
    assert a1.from_cache is False
    assert a2.from_cache is True
    assert calls == [1]
    accessor.read_frame(2)
    accessor.read_frame(3)  # evicts 1
    accessor.read_frame(1)  # miss again
    assert calls == [1, 2, 3, 1]
    stats = accessor.cache_stats()
    assert stats["cached_frames"] == 2
    assert 1 in stats["cached_indices"]


def test_missing_video_accessor_errors(tmp_path: Path) -> None:
    with pytest.raises(VideoAccessError, match="unavailable"):
        PreparedFrameAccessor(tmp_path / "missing.avi")


def test_plateau_keeps_decoded_frame_without_redecode(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path, n_prepared=5, plateau_repeats=2)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(
            built["package_dir"],
            decode_initial_frame=True,
            video_cache_size=4,
        )
    assert session.last_decoded is not None
    prepared0 = session.playhead.prepared_frame_idx
    # Step within the same prepared plateau (samples 0 and 1 share prepared 0).
    session.seek_sample(1, decode=True)
    assert session.playhead.sample_idx == 1
    assert session.playhead.prepared_frame_idx == prepared0
    assert session.last_decoded.prepared_frame_idx == prepared0


def test_directional_and_symmetric_bout_identities(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(built["package_dir"])

    dir_id = f"{built['package_run_id']}__bout_dir_a"
    mapped = session.seek_bout_start(dir_id)
    assert session.selected_bout_id == dir_id
    assert session.selected_channel == "exclusive"
    assert session.selected_direction.focal_id == 0
    assert session.selected_direction.partner_id == 1
    assert mapped.start.sample_idx == 0
    assert mapped.end.sample_idx == 3

    contact_id = f"{built['package_run_id']}__bout_contact"
    session.seek_bout_end(contact_id)
    assert session.selected_channel == "contact"
    assert session.selected_direction.focal_id is None
    assert session.selected_direction.partner_id is None
    assert session.playhead.sample_idx == 7

    sbs_id = f"{built['package_run_id']}__bout_sbs"
    session.seek_bout_start(sbs_id)
    assert session.selected_channel == "side_by_side"
    assert session.selected_direction.focal_id is None

    reciprocal = f"{built['package_run_id']}__bout_dir_b"
    session.seek_bout_start(reciprocal)
    assert session.selected_direction.focal_id == 1
    assert session.selected_direction.partner_id == 0


def test_inclusive_bout_boundary_matches_sync(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(built["package_dir"])
    bout_id = f"{built['package_run_id']}__bout_contact"
    mapped = session.seek_bout_start(bout_id)
    sync_start = session.sync.mapping_at_sample(2)
    sync_end = session.sync.mapping_at_sample(7)
    assert mapped.prepared_frame_idx_start == sync_start.prepared_frame_idx
    assert mapped.prepared_frame_idx_end == sync_end.prepared_frame_idx


def test_seeking_does_not_create_review_decisions(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(built["package_dir"])
    snap = session.snapshot()
    assert snap["mode"] == "inspection"
    assert "accepted" not in snap
    assert "review_run_id" not in snap
    assert not (built["session_root"] / "behavioral_review").exists()


def test_unknown_bout_rejected(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(built["package_dir"])
    with pytest.raises(SessionError, match="Unknown bout_id"):
        session.seek_bout_start("does_not_exist")
