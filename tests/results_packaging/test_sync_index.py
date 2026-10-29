"""Hand-computed synchronization index tests for S9 Sprint 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from results_packaging.contracts import SYNC_SCHEMA_VERSION, SyncIndexError
from results_packaging.sync_index import (
    build_sync_index_from_prepared_sync,
    build_sync_index_from_source_arrays,
    validate_sync_index,
)
from tests.results_packaging._fixtures import write_prepared_sync


def _unique_dense(times: np.ndarray) -> pd.DataFrame:
    idx = np.arange(times.shape[0], dtype=np.int64)
    return pd.DataFrame(
        {
            "sample_idx": idx,
            "dense_frame_idx": idx,
            "time_sec": times.astype(np.float64),
        }
    )


def test_exact_nearest_neighbor_and_residual() -> None:
    # Prepared @ 10 Hz: 0.0, 0.1, 0.2
    # Target midpoint 0.06 → nearer to 0.1 (frame 1)
    source_frames = np.arange(3, dtype=np.int64)
    source_times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    targets = np.array([0.0, 0.06, 0.2], dtype=np.float64)
    result = build_sync_index_from_source_arrays(
        source_frame_idx=source_frames,
        source_time_sec=source_times,
        target_time_sec=targets,
        source_fps_header=10.0,
    )
    idx = result.sync_index
    assert list(idx["prepared_frame_idx"]) == [0, 1, 2]
    assert bool(idx.loc[0, "is_exact_source_sample"])
    assert not bool(idx.loc[1, "is_exact_source_sample"])
    assert bool(idx.loc[2, "is_exact_source_sample"])
    assert idx.loc[1, "timing_residual_sec"] == pytest.approx(0.06 - 0.1)
    assert idx.loc[0, "schema_version"] == SYNC_SCHEMA_VERSION


def test_earlier_frame_tie_break() -> None:
    source_times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    targets = np.array([0.05], dtype=np.float64)  # equidistant 0.0 vs 0.1
    result = build_sync_index_from_source_arrays(
        source_frame_idx=np.arange(3, dtype=np.int64),
        source_time_sec=source_times,
        target_time_sec=targets,
        source_fps_header=10.0,
    )
    assert int(result.sync_index.loc[0, "prepared_frame_idx"]) == 0


def test_first_and_last_frame_behavior() -> None:
    frames = np.arange(4, dtype=np.int64)
    times = 2.5 + frames.astype(np.float64) / 40.0
    # Use default uniform timeline construction via prepared-sync-like arrays.
    from blob_features.timing import build_nearest_neighbor_timing_map

    nn = build_nearest_neighbor_timing_map(
        source_frame_idx=frames,
        source_time_sec=times,
        target_fps=120.0,
        source_fps_header=40.0,
    )
    result = build_sync_index_from_source_arrays(
        source_frame_idx=frames,
        source_time_sec=times,
        target_time_sec=nn.target_time_sec,
        source_fps_header=40.0,
    )
    assert int(result.sync_index.iloc[0]["prepared_frame_idx"]) == 0
    assert int(result.sync_index.iloc[-1]["prepared_frame_idx"]) == 3
    assert bool(result.sync_index.iloc[0]["is_exact_source_sample"])
    assert bool(result.sync_index.iloc[-1]["is_exact_source_sample"])


def test_repeated_prepared_frame_mapping_and_dense_ne_prepared(tmp_path: Path) -> None:
    # 5 prepared frames @ 30 Hz → many 120 Hz dense samples (dense ≠ prepared).
    sync_path = write_prepared_sync(tmp_path / "prepared_sync.npz", frame_count=5, fps=30.0)
    from blob_features.timing import build_nearest_neighbor_timing_map_from_prepared_sync

    nn = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    unique = _unique_dense(nn.target_time_sec)
    result = build_sync_index_from_prepared_sync(sync_path, s8_unique_dense_times=unique)
    assert result.n_dense == int(nn.target_sample_count)
    assert result.n_prepared_source == 5
    assert result.n_dense != result.n_prepared_source
    assert result.n_prepared_unique <= result.n_prepared_source
    assert result.max_many_to_one >= 1
    # Many-to-one: at least one prepared frame reused when dense > prepared.
    assert result.n_dense > result.n_prepared_unique
    counts = result.sync_index["prepared_frame_idx"].value_counts()
    assert int(counts.max()) == result.max_many_to_one
    assert result.monotonic_non_decreasing
    validate_sync_index(result.sync_index)


def test_deterministic_ordering(tmp_path: Path) -> None:
    sync_path = write_prepared_sync(tmp_path / "prepared_sync.npz", frame_count=6, fps=30.0)
    from blob_features.timing import build_nearest_neighbor_timing_map_from_prepared_sync

    nn = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    unique = _unique_dense(nn.target_time_sec)
    first = build_sync_index_from_prepared_sync(sync_path, s8_unique_dense_times=unique)
    second = build_sync_index_from_prepared_sync(sync_path, s8_unique_dense_times=unique)
    assert list(first.sync_index["sample_idx"]) == list(second.sync_index["sample_idx"])
    assert list(first.sync_index["prepared_frame_idx"]) == list(
        second.sync_index["prepared_frame_idx"]
    )
    assert list(first.sync_index["sample_idx"]) == list(range(first.n_dense))


def test_out_of_range_targets_fail() -> None:
    source_times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    with pytest.raises(SyncIndexError, match="outside|Nearest-neighbor"):
        build_sync_index_from_source_arrays(
            source_frame_idx=np.arange(3, dtype=np.int64),
            source_time_sec=source_times,
            target_time_sec=np.array([-0.01, 0.0], dtype=np.float64),
            source_fps_header=10.0,
        )


def test_nonmonotonic_and_duplicate_source_times_fail() -> None:
    with pytest.raises(SyncIndexError, match="Nearest-neighbor|strictly increasing"):
        build_sync_index_from_source_arrays(
            source_frame_idx=np.arange(3, dtype=np.int64),
            source_time_sec=np.array([0.0, 0.2, 0.1], dtype=np.float64),
            target_time_sec=np.array([0.0], dtype=np.float64),
            source_fps_header=10.0,
        )
    with pytest.raises(SyncIndexError, match="Nearest-neighbor|strictly increasing"):
        build_sync_index_from_source_arrays(
            source_frame_idx=np.arange(3, dtype=np.int64),
            source_time_sec=np.array([0.0, 0.1, 0.1], dtype=np.float64),
            target_time_sec=np.array([0.0], dtype=np.float64),
            source_fps_header=10.0,
        )


def test_malformed_nonfinite_times_fail() -> None:
    with pytest.raises(SyncIndexError):
        build_sync_index_from_source_arrays(
            source_frame_idx=np.arange(2, dtype=np.int64),
            source_time_sec=np.array([0.0, np.nan], dtype=np.float64),
            target_time_sec=np.array([0.0], dtype=np.float64),
            source_fps_header=10.0,
        )


def test_s8_time_mismatch_fails(tmp_path: Path) -> None:
    sync_path = write_prepared_sync(tmp_path / "prepared_sync.npz", frame_count=4, fps=30.0)
    from blob_features.timing import build_nearest_neighbor_timing_map_from_prepared_sync

    nn = build_nearest_neighbor_timing_map_from_prepared_sync(sync_path)
    unique = _unique_dense(nn.target_time_sec)
    unique.loc[1, "time_sec"] = float(unique.loc[1, "time_sec"]) + 1e-6
    with pytest.raises(SyncIndexError, match="time_sec"):
        build_sync_index_from_prepared_sync(sync_path, s8_unique_dense_times=unique)
