"""Compositor rendering, immutability, and layer-availability tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import (
    solid_frame,
    write_behavior_states_parquet,
    write_s2_pose_parquet,
)

from behavioral_review.sync import SyncMap
from overlay import (
    DrawConfig,
    LayerId,
    OverlayCompositor,
    collect_evidence,
    make_s2_inferred_pose_adapter,
    make_s8_exclusive_a_to_b,
    make_s8_exclusive_b_to_a,
)
from overlay.adapters import make_s5_segmentation_stub
from overlay.evidence import debug_counter_observation


def _adapters(tmp_path: Path, sync_df):
    s2 = make_s2_inferred_pose_adapter(
        write_s2_pose_parquet(tmp_path / "pose.parquet", prepared_frames=(0, 1, 2))
    )
    states = write_behavior_states_parquet(tmp_path / "states.parquet", sync_df)
    return {
        LayerId.S2_POSE_INFERRED: s2,
        LayerId.S8_EXCLUSIVE_A_TO_B: make_s8_exclusive_a_to_b(states),
        LayerId.S8_EXCLUSIVE_B_TO_A: make_s8_exclusive_b_to_a(states),
        LayerId.S5_SEGMENTATION: make_s5_segmentation_stub(),
        LayerId.DEBUG_FRAME_COUNTER: _DebugAdapter(),
    }


class _DebugAdapter:
    layer_id = LayerId.DEBUG_FRAME_COUNTER

    @property
    def source_identity(self):
        from overlay.adapters.base import SourceIdentity

        return SourceIdentity(subsystem="debug", run_id=None)

    @property
    def available(self) -> bool:
        return True

    def observe(self, timing, *, prepared_frame_idx: int):
        _ = prepared_frame_idx
        return debug_counter_observation(timing)


def test_source_array_immutability(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _adapters(tmp_path, sync_df)
    frame = solid_frame()
    original = frame.copy()
    evidence = collect_evidence(
        sync,
        1,
        adapters,
        selected_layers=[LayerId.S2_POSE_INFERRED, LayerId.DEBUG_FRAME_COUNTER],
    )
    composed = OverlayCompositor().render_frame(
        frame,
        prepared_frame_idx=1,
        evidence=evidence,
        selected_layers=[LayerId.S2_POSE_INFERRED, LayerId.DEBUG_FRAME_COUNTER],
        style=DrawConfig(),
    )
    assert np.array_equal(frame, original)
    assert composed.frame_bgr is not frame
    assert not np.array_equal(composed.frame_bgr, frame)


def test_deterministic_rendering(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _adapters(tmp_path, sync_df)
    layers = [
        LayerId.S2_POSE_INFERRED,
        LayerId.S8_EXCLUSIVE_A_TO_B,
        LayerId.S8_EXCLUSIVE_B_TO_A,
        LayerId.DEBUG_FRAME_COUNTER,
    ]
    evidence = collect_evidence(sync, 1, adapters, selected_layers=layers)
    comp = OverlayCompositor()
    a = comp.render_frame(
        solid_frame(),
        prepared_frame_idx=1,
        evidence=evidence,
        selected_layers=layers,
    )
    b = comp.render_frame(
        solid_frame(),
        prepared_frame_idx=1,
        evidence=evidence,
        selected_layers=layers,
    )
    assert np.array_equal(a.frame_bgr, b.frame_bgr)
    assert a.rendered_layers == b.rendered_layers
    assert a.dense_sample_idx == b.dense_sample_idx == evidence.timing.dense_sample_idx


def test_layer_availability_and_unavailable_badge(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _adapters(tmp_path, sync_df)
    layers = [LayerId.S2_POSE_INFERRED, LayerId.S5_SEGMENTATION]
    evidence = collect_evidence(sync, 0, adapters, selected_layers=layers)
    composed = OverlayCompositor().render_frame(
        solid_frame(),
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=layers,
        style=DrawConfig(unavailable_badge=True),
    )
    assert LayerId.S2_POSE_INFERRED in composed.rendered_layers
    assert LayerId.S5_SEGMENTATION in composed.unavailable_layers
    # Badge text burned in (blue-ish channel shift vs solid source).
    assert composed.notes
    assert any("s5.segmentation" in n for n in composed.notes)


def test_dual_direction_layers_both_render(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _adapters(tmp_path, sync_df)
    layers = [LayerId.S8_EXCLUSIVE_A_TO_B, LayerId.S8_EXCLUSIVE_B_TO_A]
    evidence = collect_evidence(sync, 0, adapters, selected_layers=layers)
    composed = OverlayCompositor().render_frame(
        solid_frame((10, 10, 10)),
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=layers,
    )
    assert set(composed.rendered_layers) == set(layers)
    assert composed.unavailable_layers == ()


def test_missing_pose_frame_marked_unavailable(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    # Pose only for frames 0..2; prepared 3 missing.
    adapters = _adapters(tmp_path, sync_df)
    layers = [LayerId.S2_POSE_INFERRED]
    evidence = collect_evidence(sync, 3, adapters, selected_layers=layers)
    composed = OverlayCompositor().render_frame(
        solid_frame(),
        prepared_frame_idx=3,
        evidence=evidence,
        selected_layers=layers,
    )
    assert LayerId.S2_POSE_INFERRED in composed.unavailable_layers
