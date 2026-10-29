"""Inclusive bout-boundary → prepared-frame mapping tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from results_packaging.bout_mapping import (
    map_bout_boundaries_to_prepared_frames,
    map_single_bout_boundary,
)
from results_packaging.contracts import BoutMappingError
from results_packaging.sync_index import build_sync_index_from_source_arrays
from tests.results_packaging._fixtures import make_bouts


def _sync_with_repeated_prepared() -> pd.DataFrame:
    # Dense samples 0..5 map to prepared frames [0,0,1,1,1,2] (many-to-one).
    source_frames = np.array([0, 1, 2], dtype=np.int64)
    source_times = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    # Hand-chosen targets that produce repeated prepared frames via NN.
    targets = np.array([0.0, 0.05, 0.2, 0.22, 0.28, 0.4], dtype=np.float64)
    result = build_sync_index_from_source_arrays(
        source_frame_idx=source_frames,
        source_time_sec=source_times,
        target_time_sec=targets,
        source_fps_header=5.0,
    )
    return result.sync_index


def test_inclusive_bout_boundary_mapping_preserves_lengths() -> None:
    sync = _sync_with_repeated_prepared()
    # prepared map for targets:
    # 0.0 → 0, 0.05 → 0 (tie-ish nearer/earlier), 0.2 → 1, 0.22 → 1, 0.28 → 1, 0.4 → 2
    assert list(sync["prepared_frame_idx"]) == [0, 0, 1, 1, 1, 2]

    bouts = make_bouts(intervals=[(0, 2), (2, 4), (5, 5)])
    mapped = map_bout_boundaries_to_prepared_frames(bouts, sync, fps=120.0)
    assert len(mapped) == 3
    assert list(mapped["start_sample_idx"]) == [0, 2, 5]
    assert list(mapped["end_sample_idx"]) == [2, 4, 5]
    assert list(mapped["bout_duration_frames"]) == [3, 3, 1]
    assert list(mapped["prepared_frame_idx_start"]) == [0, 1, 2]
    assert list(mapped["prepared_frame_idx_end"]) == [1, 1, 2]
    # Dense duration unchanged even when prepared start/end collapse.
    assert int(mapped.loc[1, "bout_duration_frames"]) == 3
    assert int(mapped.loc[1, "prepared_frame_idx_start"]) == int(
        mapped.loc[1, "prepared_frame_idx_end"]
    )


def test_single_bout_helper_and_invalid_indices() -> None:
    sync = _sync_with_repeated_prepared()
    mapped = map_single_bout_boundary(
        bout_id="x",
        start_sample_idx=1,
        end_sample_idx=3,
        sync_index=sync,
    )
    assert mapped.bout_duration_frames == 3
    assert mapped.prepared_frame_idx_start == 0
    assert mapped.prepared_frame_idx_end == 1

    with pytest.raises(BoutMappingError, match="outside sync domain"):
        map_single_bout_boundary(
            bout_id="bad",
            start_sample_idx=0,
            end_sample_idx=99,
            sync_index=sync,
        )
    with pytest.raises(BoutMappingError, match="start_sample_idx"):
        map_bout_boundaries_to_prepared_frames(
            pd.DataFrame(
                [
                    {
                        "bout_id": "bad",
                        "start_sample_idx": 3,
                        "end_sample_idx": 1,
                        "bout_duration_frames": -1,
                    }
                ]
            ),
            sync,
        )
