"""Authoritative S1 timing and uniform 120 Hz mapping for S4 Sprint 1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pose_finalization import (
    TARGET_FPS_DEFAULT,
    PoseFinalizationError,
    build_timing_map_from_prepared_sync,
    build_uniform_target_timeline,
    load_source_timing,
    map_target_to_source,
)
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


def test_source_fps_exactly_120(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=5, fps=120.0)
    before = _sha256(sync_path)

    timing_map = build_timing_map_from_prepared_sync(sync_path)
    assert timing_map.target_fps == TARGET_FPS_DEFAULT
    assert timing_map.target_sample_count == 5
    assert np.all(timing_map.is_exact_source_sample)
    assert np.allclose(timing_map.interpolation_weight, 0.0)
    assert np.array_equal(timing_map.left_source_frame_idx, np.arange(5))
    assert np.array_equal(timing_map.right_source_frame_idx, np.arange(5))
    assert before == _sha256(sync_path)


def test_source_fps_different_from_120(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=8, fps=20.0)
    timing_map = build_timing_map_from_prepared_sync(sync_path)

    assert timing_map.source_fps_header == pytest.approx(20.0)
    assert timing_map.target_sample_count == 43  # 0 .. 0.35 s at 120 Hz
    assert timing_map.target_time_sec[0] == pytest.approx(0.0)
    assert timing_map.target_time_sec[-1] == pytest.approx(0.35)
    # Endpoints coincide with observed frames; interior samples require derivation.
    assert bool(timing_map.is_exact_source_sample[0])
    assert bool(timing_map.is_exact_source_sample[-1])
    assert not bool(np.all(timing_map.is_exact_source_sample))
    interior = ~timing_map.is_exact_source_sample
    assert np.all(timing_map.interpolation_weight[interior] > 0.0)
    assert np.all(timing_map.interpolation_weight[interior] < 1.0)
    assert np.all(
        timing_map.left_source_frame_idx[interior]
        < timing_map.right_source_frame_idx[interior]
    )


def test_nonzero_source_time_origin() -> None:
    frames = np.arange(4, dtype=np.int64)
    times = 2.5 + frames.astype(np.float64) / 40.0
    targets = build_uniform_target_timeline(times, target_fps=120.0)
    timing_map = map_target_to_source(
        source_frame_idx=frames,
        source_time_sec=times,
        target_time_sec=targets,
        target_fps=120.0,
        source_fps_header=40.0,
    )
    assert timing_map.target_time_sec[0] == pytest.approx(2.5)
    assert timing_map.target_time_sec[-1] == pytest.approx(float(times[-1]))
    assert timing_map.target_sample_count >= 2


def test_deterministic_mapping(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=6, fps=30.0)
    first = build_timing_map_from_prepared_sync(sync_path)
    second = build_timing_map_from_prepared_sync(sync_path)
    assert np.array_equal(first.target_time_sec, second.target_time_sec)
    assert np.array_equal(first.left_source_frame_idx, second.left_source_frame_idx)
    assert np.array_equal(first.right_source_frame_idx, second.right_source_frame_idx)
    assert np.array_equal(first.interpolation_weight, second.interpolation_weight)
    assert np.array_equal(first.is_exact_source_sample, second.is_exact_source_sample)


def test_invalid_nonmonotonic_timestamps() -> None:
    frames = np.arange(3, dtype=np.int64)
    times = np.array([0.0, 0.2, 0.1], dtype=np.float64)
    with pytest.raises(PoseFinalizationError, match="strictly increasing"):
        build_uniform_target_timeline(times)
    with pytest.raises(PoseFinalizationError, match="strictly increasing"):
        map_target_to_source(
            source_frame_idx=frames,
            source_time_sec=times,
            target_time_sec=np.array([0.0], dtype=np.float64),
        )


def test_invalid_frame_counts() -> None:
    with pytest.raises(PoseFinalizationError, match="lengths do not match"):
        map_target_to_source(
            source_frame_idx=np.arange(3, dtype=np.int64),
            source_time_sec=np.array([0.0, 0.1], dtype=np.float64),
            target_time_sec=np.array([0.0], dtype=np.float64),
        )
    with pytest.raises(PoseFinalizationError, match=r"\[0, 1, ..., N - 1\]"):
        map_target_to_source(
            source_frame_idx=np.array([0, 2, 3], dtype=np.int64),
            source_time_sec=np.array([0.0, 0.1, 0.2], dtype=np.float64),
            target_time_sec=np.array([0.0], dtype=np.float64),
        )


def test_mapping_bounds() -> None:
    frames = np.arange(3, dtype=np.int64)
    times = frames.astype(np.float64) / 30.0
    with pytest.raises(PoseFinalizationError, match="outside the authoritative"):
        map_target_to_source(
            source_frame_idx=frames,
            source_time_sec=times,
            target_time_sec=np.array([-0.01, 0.0], dtype=np.float64),
        )
    with pytest.raises(PoseFinalizationError, match="outside the authoritative"):
        map_target_to_source(
            source_frame_idx=frames,
            source_time_sec=times,
            target_time_sec=np.array([0.0, 1.0], dtype=np.float64),
        )


def test_empty_or_malformed_timing(tmp_path: Path) -> None:
    with pytest.raises(PoseFinalizationError, match="must not be empty"):
        build_uniform_target_timeline(np.array([], dtype=np.float64))
    with pytest.raises(PoseFinalizationError, match="finite"):
        build_uniform_target_timeline(np.array([0.0, np.nan], dtype=np.float64))

    missing = tmp_path / "missing_prepared_sync.npz"
    with pytest.raises(PoseFinalizationError, match="Could not load"):
        load_source_timing(missing)

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, not_a_sync=np.array([1.0]))
    with pytest.raises(PoseFinalizationError, match="Could not load"):
        load_source_timing(bad)


def test_exact_vs_derived_distinction() -> None:
    frames = np.arange(3, dtype=np.int64)
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    targets = np.array([0.0, 0.05, 0.2], dtype=np.float64)
    timing_map = map_target_to_source(
        source_frame_idx=frames,
        source_time_sec=times,
        target_time_sec=targets,
        target_fps=20.0,
        source_fps_header=10.0,
    )
    assert bool(timing_map.is_exact_source_sample[0])
    assert not bool(timing_map.is_exact_source_sample[1])
    assert bool(timing_map.is_exact_source_sample[2])
    assert timing_map.interpolation_weight[1] == pytest.approx(0.5)
    assert timing_map.left_source_frame_idx[1] == 0
    assert timing_map.right_source_frame_idx[1] == 1


def test_load_source_timing_is_read_only(tmp_path: Path) -> None:
    sync_path = _write_sync(tmp_path / "prepared_sync.npz", frame_count=4, fps=60.0)
    before = _sha256(sync_path)
    source = load_source_timing(sync_path)
    assert source.frame_count == 4
    assert source.fps_header == pytest.approx(60.0)
    assert before == _sha256(sync_path)
