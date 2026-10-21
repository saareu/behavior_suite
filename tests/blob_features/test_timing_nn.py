"""Nearest-neighbor 120 Hz mapping for Subsystem 05."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from blob_features import (
    TARGET_FPS_DEFAULT,
    BlobFeaturesError,
    build_nearest_neighbor_timing_map,
    build_nearest_neighbor_timing_map_from_prepared_sync,
    map_target_nearest_source_frame,
)
from pose_finalization.timing import build_uniform_target_timeline
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sync(path: Path, *, frame_count: int, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    write_prepared_sync_npz(path, sync)
    return path


def test_target_timeline_matches_s4(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=8, fps=20.0)
    nn_map = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    source_times = np.arange(8, dtype=np.float64) / 20.0
    s4_targets = build_uniform_target_timeline(source_times, target_fps=TARGET_FPS_DEFAULT)
    assert np.array_equal(nn_map.target_time_sec, s4_targets)
    assert nn_map.target_fps == TARGET_FPS_DEFAULT
    assert nn_map.target_sample_count == 43


def test_exact_source_hits_at_120(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=5, fps=120.0)
    before = _sha256(sync_path)
    nn_map = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    assert np.all(nn_map.is_exact_source_sample)
    assert np.array_equal(nn_map.orig_frame_idx, np.arange(5))
    assert before == _sha256(sync_path)


def test_equidistant_tie_selects_earlier_frame() -> None:
    frames = np.arange(3, dtype=np.int64)
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    # Midpoint between frames 0 and 1 is equidistant.
    targets = np.array([0.05], dtype=np.float64)
    positions, exact = map_target_nearest_source_frame(
        source_time_sec=times,
        target_time_sec=targets,
    )
    assert not bool(exact[0])
    assert int(positions[0]) == 0  # earlier frame wins

    nn_map = build_nearest_neighbor_timing_map(
        source_frame_idx=frames,
        source_time_sec=times,
        target_time_sec=targets,
        target_fps=20.0,
        source_fps_header=10.0,
    )
    assert int(nn_map.orig_frame_idx[0]) == 0


def test_nearer_neighbor_not_tie() -> None:
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    targets = np.array([0.06], dtype=np.float64)  # closer to 0.1
    positions, _ = map_target_nearest_source_frame(
        source_time_sec=times,
        target_time_sec=targets,
    )
    assert int(positions[0]) == 1


def test_monotonicity_and_bounds(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=6, fps=30.0)
    nn_map = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    assert nn_map.orig_frame_idx.shape[0] == nn_map.target_time_sec.shape[0]
    assert int(nn_map.orig_frame_idx.min()) >= 0
    assert int(nn_map.orig_frame_idx.max()) < nn_map.source_frame_count
    assert np.all(np.diff(nn_map.orig_frame_idx) >= 0)
    assert int(nn_map.orig_frame_idx[0]) == 0
    assert int(nn_map.orig_frame_idx[-1]) == 5


def test_endpoint_behavior() -> None:
    frames = np.arange(4, dtype=np.int64)
    times = 2.5 + frames.astype(np.float64) / 40.0
    nn_map = build_nearest_neighbor_timing_map(
        source_frame_idx=frames,
        source_time_sec=times,
        target_fps=120.0,
        source_fps_header=40.0,
    )
    assert nn_map.target_time_sec[0] == pytest.approx(2.5)
    assert nn_map.target_time_sec[-1] == pytest.approx(float(times[-1]))
    assert int(nn_map.orig_frame_idx[0]) == 0
    assert int(nn_map.orig_frame_idx[-1]) == 3
    assert bool(nn_map.is_exact_source_sample[0])
    assert bool(nn_map.is_exact_source_sample[-1])


def test_deterministic_mapping(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=6, fps=30.0)
    first = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    second = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    assert np.array_equal(first.target_time_sec, second.target_time_sec)
    assert np.array_equal(first.orig_frame_idx, second.orig_frame_idx)
    assert np.array_equal(first.is_exact_source_sample, second.is_exact_source_sample)


def test_invalid_nonmonotonic_timestamps() -> None:
    times = np.array([0.0, 0.2, 0.1], dtype=np.float64)
    with pytest.raises(BlobFeaturesError, match="strictly increasing"):
        map_target_nearest_source_frame(
            source_time_sec=times,
            target_time_sec=np.array([0.0], dtype=np.float64),
        )


def test_mapping_bounds() -> None:
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    with pytest.raises(BlobFeaturesError, match="outside the authoritative"):
        map_target_nearest_source_frame(
            source_time_sec=times,
            target_time_sec=np.array([-0.01], dtype=np.float64),
        )
    with pytest.raises(BlobFeaturesError, match="outside the authoritative"):
        map_target_nearest_source_frame(
            source_time_sec=times,
            target_time_sec=np.array([0.0, 1.0], dtype=np.float64),
        )


def test_sync_bytes_unchanged(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=4, fps=60.0)
    before = _sha256(sync_path)
    build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    assert before == _sha256(sync_path)
