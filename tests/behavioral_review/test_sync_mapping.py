"""Dense↔prepared synchronization unit tests (hand-computed)."""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_review.contracts import SyncMappingError
from behavioral_review.sync import SyncMap
from tests.behavioral_review._fixtures import build_hand_computed_sync


def test_dense_to_prepared_plateau_mapping() -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=3)
    sync = SyncMap.from_sync_index(sync_df)

    assert sync.n_dense == 12
    # Samples 0,1,2 share prepared 0; 3,4,5 share prepared 1.
    assert sync.mapping_at_sample(0).prepared_frame_idx == 0
    assert sync.mapping_at_sample(2).prepared_frame_idx == 0
    assert sync.mapping_at_sample(3).prepared_frame_idx == 1
    plateau = sync.plateau_sample_indices(1)
    assert list(plateau) == [3, 4, 5]


def test_first_and_last_sample_seeking() -> None:
    sync_df = build_hand_computed_sync(n_prepared=5, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    first = sync.mapping_at_sample(0)
    last = sync.mapping_at_sample(sync.n_dense - 1)
    assert first.sample_idx == 0
    assert first.dense_frame_idx == 0
    assert first.prepared_frame_idx == 0
    assert last.sample_idx == sync.n_dense - 1
    assert last.prepared_frame_idx == 4


def test_out_of_range_sample_rejected() -> None:
    sync = SyncMap.from_sync_index(build_hand_computed_sync(n_prepared=3, plateau_repeats=2))
    with pytest.raises(SyncMappingError, match="out of range"):
        sync.mapping_at_sample(-1)
    with pytest.raises(SyncMappingError, match="out of range"):
        sync.mapping_at_sample(sync.n_dense)


def test_time_lookup_earlier_tie_break() -> None:
    sync_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    # Midpoint between sample 0 (0.0s) and sample 1 (1/120 s).
    mid = 0.5 / 120.0
    mapped = sync.mapping_at_time_sec(mid)
    assert mapped.sample_idx == 0


def test_prepared_frame_maps_to_nearest_dense_on_plateau() -> None:
    sync = SyncMap.from_sync_index(build_hand_computed_sync(n_prepared=3, plateau_repeats=2))
    mapped = sync.mapping_from_prepared_frame(1)
    # Plateau samples 2 (t=2/120) and 3 (t=3/120) share prepared 1 (t=0.1).
    # Nearest dense time wins (sample 3); earlier sample is only a tie-break.
    assert mapped.sample_idx == 3
    assert mapped.prepared_frame_idx == 1


def test_prepared_frame_earlier_sample_on_equal_residual() -> None:
    # Construct an explicit equal-residual plateau.
    import pandas as pd

    from results_packaging.contracts import TIME_ATOL_SEC

    sync_df = pd.DataFrame(
        {
            "schema_version": ["results_package_sync_v1"] * 2,
            "sample_idx": [10, 11],
            "dense_frame_idx": [10, 11],
            "time_sec": [1.0, 1.2],
            "prepared_frame_idx": [5, 5],
            "prepared_time_sec": [1.1, 1.1],
            "is_exact_source_sample": [False, False],
            "timing_residual_sec": [-0.1, 0.1],
        }
    )
    # Contiguous 0..N-1 required — rebuild as a 2-row contiguous map.
    sync_df["sample_idx"] = [0, 1]
    sync_df["dense_frame_idx"] = [0, 1]
    assert abs(abs(sync_df.loc[0, "timing_residual_sec"]) - abs(sync_df.loc[1, "timing_residual_sec"])) <= TIME_ATOL_SEC
    sync = SyncMap.from_sync_index(sync_df)
    mapped = sync.mapping_from_prepared_frame(5)
    assert mapped.sample_idx == 0


def test_inclusive_bout_boundary_seek() -> None:
    sync = SyncMap.from_sync_index(build_hand_computed_sync(n_prepared=5, plateau_repeats=2))
    bout = sync.bout_seek(start_sample_idx=2, end_sample_idx=5, bout_id="b1")
    assert bout.bout_duration_frames == 4
    assert bout.start.sample_idx == 2
    assert bout.end.sample_idx == 5
    assert bout.prepared_frame_idx_start == int(sync.prepared_frame_idx[2])
    assert bout.prepared_frame_idx_end == int(sync.prepared_frame_idx[5])


def test_invalid_sync_schema_rejected() -> None:
    sync_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    sync_df = sync_df.drop(columns=["timing_residual_sec"])
    with pytest.raises(SyncMappingError, match="missing required columns"):
        SyncMap.from_sync_index(sync_df)


def test_dense_not_interchanged_with_prepared() -> None:
    sync = SyncMap.from_sync_index(build_hand_computed_sync(n_prepared=4, plateau_repeats=2))
    audit = sync.audit_dict()
    assert audit["n_dense"] == 8
    assert audit["n_prepared_unique"] == 4
    assert audit["n_dense"] != audit["n_prepared_unique"]
    assert audit["max_many_to_one"] == 2
    # sample_idx 7 maps to prepared 3, not 7.
    assert sync.mapping_at_sample(7).prepared_frame_idx == 3


def test_behavioral_and_video_timestamps_exposed() -> None:
    sync = SyncMap.from_sync_index(build_hand_computed_sync(n_prepared=2, plateau_repeats=2))
    m = sync.mapping_at_sample(1)
    assert np.isfinite(m.time_sec)
    assert np.isfinite(m.prepared_time_sec)
    assert m.timing_residual_sec == pytest.approx(m.time_sec - m.prepared_time_sec)
