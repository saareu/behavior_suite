"""Adapter contract tests: S2/S3/S4 distinctions and automated vs reviewed."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import (
    write_behavior_states_parquet,
    write_reviewed_states_parquet,
    write_s2_pose_parquet,
    write_s3_tracked_pose_parquet,
    write_s4_final_pose_parquet,
)

from behavioral_review.sync import SyncMap
from overlay.adapters import (
    make_s2_inferred_pose_adapter,
    make_s3_tracked_pose_adapter,
    make_s5_segmentation_stub,
    make_s8_exclusive_a_to_b,
    make_s8_exclusive_b_to_a,
    make_s10_reviewed_exclusive_a_to_b,
)
from overlay.adapters.pose import S4FinalPoseAdapter
from overlay.contracts import BehaviorSourceKind, LayerId, OverlayAdapterError
from overlay.timing import resolve_prepared_frame_timing


def test_s2_s3_s4_source_distinctions(tmp_path: Path) -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    )
    s2_path = write_s2_pose_parquet(tmp_path / "s2" / "pose.parquet")
    s3_path = write_s3_tracked_pose_parquet(tmp_path / "s3" / "tracked_pose.parquet")
    s4_path = write_s4_final_pose_parquet(tmp_path / "s4" / "final_pose.parquet", n_samples=6)

    s2 = make_s2_inferred_pose_adapter(s2_path, run_id="pose_inference__t")
    s3 = make_s3_tracked_pose_adapter(s3_path, run_id="tracking_correction__t")
    s4 = S4FinalPoseAdapter(s4_path, run_id="pose_finalization__t")

    assert s2.layer_id is LayerId.S2_POSE_INFERRED
    assert s3.layer_id is LayerId.S3_POSE_TRACKED
    assert s4.layer_id is LayerId.S4_POSE_FINAL
    assert s2.source_identity.subsystem == "pose_inference"
    assert s3.source_identity.subsystem == "tracking_correction"
    assert s4.source_identity.subsystem == "pose_finalization"
    assert s2.source_identity.extra["legend"] == (
        "inferred pose (S2 predictions; provisional IDs)"
    )
    assert s3.source_identity.extra["legend"] == "S3 accepted tracked pose"

    timing = resolve_prepared_frame_timing(sync, 1)
    obs2 = s2.observe(timing, prepared_frame_idx=1)
    obs3 = s3.observe(timing, prepared_frame_idx=1)
    obs4 = s4.observe(timing, prepared_frame_idx=1)
    assert obs2.available and obs2.payload["legend"] == (
        "inferred pose (S2 predictions; provisional IDs)"
    )
    assert obs3.available and obs3.payload["legend"] == "S3 accepted tracked pose"
    assert obs4.available
    assert obs4.payload["domain"] == "dense"
    assert obs4.payload["dense_sample_idx"] == timing.dense_sample_idx
    # S4 payload keeps faithful provenance codes.
    for pt in obs4.payload["points"]:
        assert "provenance" in pt
        assert "provenance_name" in pt


def test_s2_does_not_fabricate_corrected_identities(tmp_path: Path) -> None:
    path = write_s2_pose_parquet(tmp_path / "pose.parquet", prepared_frames=(0,))
    adapter = make_s2_inferred_pose_adapter(path)
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=1, plateau_repeats=1)
    )
    timing = resolve_prepared_frame_timing(sync, 0)
    obs = adapter.observe(timing, prepared_frame_idx=0)
    assert obs.available
    # Tracks may be derived from instance_index for drawing only — no provenance.
    for pt in obs.payload["points"]:
        assert "provenance" not in pt
        assert "identity_corrected" not in pt


def test_s4_rejects_unknown_provenance(tmp_path: Path) -> None:
    import pandas as pd

    path = tmp_path / "final_pose.parquet"
    pd.DataFrame(
        [
            {
                "sample_idx": 0,
                "track": 0,
                "node": "nose",
                "x": 1.0,
                "y": 2.0,
                "provenance": 99,
                "provenance_name": "invented",
            }
        ]
    ).to_parquet(path, index=False)
    adapter = S4FinalPoseAdapter(path)
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=1, plateau_repeats=1)
    )
    timing = resolve_prepared_frame_timing(sync, 0)
    with pytest.raises(OverlayAdapterError, match="Unknown S4 provenance"):
        adapter.observe(timing, prepared_frame_idx=0)


def test_automated_versus_reviewed_behavioral_source(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    s8_path = write_behavior_states_parquet(tmp_path / "s8" / "behavior_states.parquet", sync_df)
    s10_path = write_reviewed_states_parquet(
        tmp_path / "s10" / "reviewed_behavior_states.parquet", sync_df
    )

    auto = make_s8_exclusive_a_to_b(s8_path, run_id="behavioral_arbitration__t")
    reviewed = make_s10_reviewed_exclusive_a_to_b(
        s10_path, run_id="behavioral_review__t"
    )
    assert auto.source_identity.subsystem == "behavioral_arbitration"
    assert reviewed.source_identity.subsystem == "behavioral_review"
    assert auto.source_identity.extra["behavior_source_kind"] == BehaviorSourceKind.S8_AUTOMATED.value
    assert (
        reviewed.source_identity.extra["behavior_source_kind"]
        == BehaviorSourceKind.S10_REVIEWED.value
    )

    timing = resolve_prepared_frame_timing(sync, 0)
    obs_auto = auto.observe(timing, prepared_frame_idx=0)
    obs_rev = reviewed.observe(timing, prepared_frame_idx=0)
    assert obs_auto.available and obs_rev.available
    assert obs_auto.payload["behavior_label"] != obs_rev.payload["behavior_label"]
    assert obs_auto.payload["behavior_source_kind"] == "s8_automated"
    assert obs_rev.payload["behavior_source_kind"] == "s10_reviewed"


def test_dual_direction_adapters_independent(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    path = write_behavior_states_parquet(tmp_path / "behavior_states.parquet", sync_df)
    a = make_s8_exclusive_a_to_b(path)
    b = make_s8_exclusive_b_to_a(path)
    timing = resolve_prepared_frame_timing(sync, 1)
    obs_a = a.observe(timing, prepared_frame_idx=1)
    obs_b = b.observe(timing, prepared_frame_idx=1)
    assert obs_a.available and obs_b.available
    assert obs_a.payload["focal_id"] == 0
    assert obs_b.payload["focal_id"] == 1
    assert obs_a.payload["behavior_label"] != obs_b.payload["behavior_label"]


def test_invalid_behavior_label_unavailable(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    path = write_behavior_states_parquet(tmp_path / "behavior_states.parquet", sync_df)
    adapter = make_s8_exclusive_a_to_b(path)
    # Last prepared plateau maps to last dense sample which has null A→B label.
    last_prep = int(sync.prepared_frame_idx.max())
    timing = resolve_prepared_frame_timing(sync, last_prep)
    obs = adapter.observe(timing, prepared_frame_idx=last_prep)
    assert not obs.available
    assert "invent" in (obs.reason or "").lower() or "null" in (obs.reason or "").lower()


def test_s5_stub_always_unavailable() -> None:
    stub = make_s5_segmentation_stub()
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=1, plateau_repeats=1)
    )
    timing = resolve_prepared_frame_timing(sync, 0)
    obs = stub.observe(timing, prepared_frame_idx=0)
    assert not stub.available
    assert not obs.available
    assert "C1" in (obs.reason or "")
