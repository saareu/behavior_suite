"""Timing / plateau resolution tests for overlay C1."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import build_gap_sync_with_missing_prepared

from behavioral_review.sync import SyncMap
from overlay.contracts import PlateauPolicy
from overlay.timing import resolve_prepared_frame_timing


def test_one_to_one_mapping() -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=4, plateau_repeats=1)
    )
    timing = resolve_prepared_frame_timing(sync, 2)
    assert timing.available
    assert timing.dense_sample_idx == 2
    assert timing.plateau_sample_indices == (2,)
    assert timing.plateau_policy is PlateauPolicy.LAST_IN_PLATEAU


def test_many_to_one_plateau_uses_last_dense_sample() -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=4, plateau_repeats=3)
    )
    # Prepared 1 ↔ samples 3,4,5 — last-in-plateau ⇒ 5.
    timing = resolve_prepared_frame_timing(
        sync, 1, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    assert timing.available
    assert list(timing.plateau_sample_indices) == [3, 4, 5]
    assert timing.dense_sample_idx == 5

    nearest = resolve_prepared_frame_timing(
        sync, 1, plateau_policy=PlateauPolicy.NEAREST_TIME
    )
    # Nearest-time may differ from last-in-plateau; document the distinction.
    assert nearest.available
    assert nearest.dense_sample_idx == sync.mapping_from_prepared_frame(1).sample_idx
    assert timing.dense_sample_idx == 5
    # For this fixture, nearest is also often the last member; force unequal case below.


def test_last_in_plateau_differs_from_nearest_time() -> None:
    # Plateau prepared=5 with times far from prepared_time on the last sample.
    sync_df = pd.DataFrame(
        {
            "schema_version": ["results_package_sync_v1"] * 3,
            "sample_idx": [0, 1, 2],
            "dense_frame_idx": [0, 1, 2],
            "time_sec": [0.90, 1.00, 1.40],
            "prepared_frame_idx": [5, 5, 5],
            "prepared_time_sec": [1.00, 1.00, 1.00],
            "is_exact_source_sample": [False, True, False],
            "timing_residual_sec": [-0.10, 0.00, 0.40],
        }
    )
    sync = SyncMap.from_sync_index(sync_df)
    last = resolve_prepared_frame_timing(
        sync, 5, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    nearest = resolve_prepared_frame_timing(
        sync, 5, plateau_policy=PlateauPolicy.NEAREST_TIME
    )
    assert last.dense_sample_idx == 2
    assert nearest.dense_sample_idx == 1
    assert last.dense_sample_idx != nearest.dense_sample_idx


def test_first_and_last_prepared_frames() -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=5, plateau_repeats=2)
    )
    first = resolve_prepared_frame_timing(sync, 0)
    last = resolve_prepared_frame_timing(sync, 4)
    assert first.dense_sample_idx == 1  # plateau (0,1) → last
    assert last.dense_sample_idx == 9  # plateau (8,9) → last


def test_missing_dense_coverage_unavailable() -> None:
    sync = SyncMap.from_sync_index(build_gap_sync_with_missing_prepared())
    timing = resolve_prepared_frame_timing(sync, 2)
    assert not timing.available
    assert timing.dense_sample_idx is None
    assert "not present" in (timing.reason or "").lower() or "prepared_frame_idx" in (
        timing.reason or ""
    )


def test_out_of_range_and_negative_indices() -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    )
    neg = resolve_prepared_frame_timing(sync, -1)
    assert not neg.available
    far = resolve_prepared_frame_timing(sync, 999)
    assert not far.available
    assert far.dense_sample_idx is None


def test_nonuniform_timestamps_still_last_in_plateau() -> None:
    sync_df = pd.DataFrame(
        {
            "schema_version": ["results_package_sync_v1"] * 4,
            "sample_idx": [0, 1, 2, 3],
            "dense_frame_idx": [0, 1, 2, 3],
            "time_sec": [0.0, 0.05, 0.20, 0.21],
            "prepared_frame_idx": [0, 0, 1, 1],
            "prepared_time_sec": [0.0, 0.0, 0.1, 0.1],
            "is_exact_source_sample": [True, False, False, False],
            "timing_residual_sec": [0.0, 0.05, 0.10, 0.11],
        }
    )
    sync = SyncMap.from_sync_index(sync_df)
    t0 = resolve_prepared_frame_timing(sync, 0)
    t1 = resolve_prepared_frame_timing(sync, 1)
    assert t0.dense_sample_idx == 1
    assert t1.dense_sample_idx == 3


def test_playhead_policy_rejected_offline() -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    )
    timing = resolve_prepared_frame_timing(
        sync, 0, plateau_policy=PlateauPolicy.PLAYHEAD
    )
    assert not timing.available
    assert "PLAYHEAD" in (timing.reason or "")


def test_invalid_prepared_frame_type() -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    )
    from overlay.contracts import OverlayTimingError

    with pytest.raises(OverlayTimingError):
        resolve_prepared_frame_timing(sync, 1.5)  # type: ignore[arg-type]


def test_no_independent_fps_rounding() -> None:
    """Dense choice must come from sync_index members, not fps*time rounding."""

    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=3, plateau_repeats=4, prepared_fps=10.0)
    )
    timing = resolve_prepared_frame_timing(sync, 2)
    assert timing.dense_sample_idx in timing.plateau_sample_indices
    # Naive prepared_fps rounding would not enumerate plateau members.
    naive = int(round(2 * (120.0 / 10.0)))
    assert naive not in timing.plateau_sample_indices or naive == timing.dense_sample_idx
    assert list(timing.plateau_sample_indices) == [8, 9, 10, 11]
    assert timing.dense_sample_idx == 11
