"""Source eligibility / switch rules and windowed behavioral adapter reads."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import write_behavior_states_parquet

from overlay import (
    BehaviorSourceKind,
    OverlaySourceError,
    make_s8_exclusive_a_to_b,
    resolve_overlay_qc_source,
)
from overlay.contracts import PlateauPolicy
from overlay.timing import PreparedFrameTiming


def test_automated_requires_package_dir() -> None:
    with pytest.raises(OverlaySourceError, match="package_dir"):
        resolve_overlay_qc_source(source_kind=BehaviorSourceKind.S8_AUTOMATED)


def test_automated_rejects_review_run_dir(tmp_path: Path) -> None:
    with pytest.raises(OverlaySourceError, match="review_run_dir"):
        resolve_overlay_qc_source(
            source_kind="automated",
            package_dir=tmp_path,
            review_run_dir=tmp_path / "review",
        )


def test_reviewed_requires_review_run_dir() -> None:
    with pytest.raises(OverlaySourceError, match="review_run_dir"):
        resolve_overlay_qc_source(source_kind=BehaviorSourceKind.S10_REVIEWED)


def test_working_draft_alias_rejected() -> None:
    with pytest.raises(OverlaySourceError, match="[Dd]raft|working"):
        resolve_overlay_qc_source(source_kind="working")


def test_windowed_adapter_loads_only_requested_dense_range(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=8, plateau_repeats=2)
    path = write_behavior_states_parquet(tmp_path / "states.parquet", sync_df)
    # Full load baseline.
    full = make_s8_exclusive_a_to_b(path)
    assert full.available
    assert full._by_sample is not None
    n_full = len(full._by_sample)

    windowed = make_s8_exclusive_a_to_b(path, dense_sample_range=(2, 5))
    assert windowed.available
    assert windowed._by_sample is not None
    assert set(windowed._by_sample.keys()) <= {2, 3, 4, 5}
    assert len(windowed._by_sample) < n_full

    timing = PreparedFrameTiming(
        prepared_frame_idx=1,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=3,
        plateau_sample_indices=(2, 3),
        mapping=None,
    )
    obs = windowed.observe(timing, prepared_frame_idx=1)
    assert obs.available
    assert obs.payload["dense_sample_idx"] == 3

    # Outside window → missing row (unavailable), not invented.
    timing_out = PreparedFrameTiming(
        prepared_frame_idx=0,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=0,
        plateau_sample_indices=(0, 1),
        mapping=None,
    )
    obs_out = windowed.observe(timing_out, prepared_frame_idx=0)
    assert obs_out.available is False


def test_resolve_automated_uses_open_review_source(tmp_path: Path) -> None:
    # Ensure resolve path calls package open + states requirement without hitting disk science.
    with patch(
        "overlay.source.open_review_source",
        side_effect=OverlaySourceError("package ineligible"),
    ), pytest.raises(OverlaySourceError, match="ineligible"):
        resolve_overlay_qc_source(
            source_kind=BehaviorSourceKind.S8_AUTOMATED,
            package_dir=tmp_path / "missing_pkg",
        )
