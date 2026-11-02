"""Directional HUD, contact/SBS independence, and plateau golden-frame tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import solid_frame, write_behavior_states_parquet

from behavioral_review.sync import SyncMap
from overlay import (
    DrawConfig,
    HudDirectionMode,
    LayerId,
    OverlayCompositor,
    PlateauPolicy,
    collect_evidence,
    default_behavioral_qc_layers,
    layers_for_hud_mode,
    make_s8_contact,
    make_s8_exclusive_a_to_b,
    make_s8_exclusive_b_to_a,
    make_s8_sbs,
)
from overlay.evidence import debug_counter_observation
from overlay.timing import resolve_prepared_frame_timing


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


def _behavior_adapters(tmp_path: Path, sync_df):
    states = write_behavior_states_parquet(tmp_path / "states.parquet", sync_df)
    return {
        LayerId.S8_EXCLUSIVE_A_TO_B: make_s8_exclusive_a_to_b(states),
        LayerId.S8_EXCLUSIVE_B_TO_A: make_s8_exclusive_b_to_a(states),
        LayerId.S8_CONTACT: make_s8_contact(states),
        LayerId.S8_SBS: make_s8_sbs(states),
        LayerId.DEBUG_FRAME_COUNTER: _DebugAdapter(),
    }


def test_layers_for_hud_mode_directional() -> None:
    assert layers_for_hud_mode(HudDirectionMode.FOCAL_0_ONLY) == (
        LayerId.S8_EXCLUSIVE_A_TO_B,
    )
    assert layers_for_hud_mode(HudDirectionMode.FOCAL_1_ONLY) == (
        LayerId.S8_EXCLUSIVE_B_TO_A,
    )
    assert layers_for_hud_mode(HudDirectionMode.BOTH) == (
        LayerId.S8_EXCLUSIVE_A_TO_B,
        LayerId.S8_EXCLUSIVE_B_TO_A,
    )
    both = default_behavioral_qc_layers(hud_mode=HudDirectionMode.BOTH)
    assert LayerId.S8_EXCLUSIVE_A_TO_B in both
    assert LayerId.S8_EXCLUSIVE_B_TO_A in both
    focal0 = default_behavioral_qc_layers(hud_mode=HudDirectionMode.FOCAL_0_ONLY)
    assert LayerId.S8_EXCLUSIVE_A_TO_B in focal0
    assert LayerId.S8_EXCLUSIVE_B_TO_A not in focal0


def test_directional_hud_values_independent(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=3)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _behavior_adapters(tmp_path, sync_df)
    layers = [
        LayerId.S8_EXCLUSIVE_A_TO_B,
        LayerId.S8_EXCLUSIVE_B_TO_A,
    ]
    # Mid plateau prepared frame 1 → last dense sample in plateau.
    evidence = collect_evidence(sync, 1, adapters, selected_layers=layers)
    a = evidence.get(LayerId.S8_EXCLUSIVE_A_TO_B)
    b = evidence.get(LayerId.S8_EXCLUSIVE_B_TO_A)
    assert a is not None and a.available
    assert b is not None and b.available
    assert a.payload["behavior_label"] != b.payload["behavior_label"]
    assert a.payload["focal_id"] == 0 and a.payload["partner_id"] == 1
    assert b.payload["focal_id"] == 1 and b.payload["partner_id"] == 0
    assert a.payload["dense_sample_idx"] == b.payload["dense_sample_idx"]


def test_contact_sbs_independent_not_exclusive(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=3)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _behavior_adapters(tmp_path, sync_df)
    layers = [LayerId.S8_CONTACT, LayerId.S8_SBS]
    # Fixture: contact on for samples 2..n-3; SBS on for i >= n-2.
    # Last-in-plateau for prepared 0 is sample 2 → contact on, sbs off.
    evidence = collect_evidence(sync, 0, adapters, selected_layers=layers)
    contact = evidence.get(LayerId.S8_CONTACT)
    sbs = evidence.get(LayerId.S8_SBS)
    assert contact is not None and contact.available
    assert sbs is not None and sbs.available
    assert contact.payload["channel"] == "contact"
    assert sbs.payload["channel"] == "side_by_side"
    assert bool(contact.payload["active"]) is True
    assert bool(sbs.payload["active"]) is False

    composed = OverlayCompositor().render_frame(
        solid_frame(h=96, w=160),
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=layers,
        style=DrawConfig(),
    )
    # Both LEDs drawn; pixels near LED strip should differ from solid background.
    assert composed.frame_bgr.shape == (96, 160, 3)
    right = composed.frame_bgr[:, -30:, :]
    assert not np.array_equal(right, solid_frame(h=96, w=30))


def test_transition_plateau_last_in_plateau_golden(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=3)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _behavior_adapters(tmp_path, sync_df)
    # Prepared frame 1 plateau samples: 3,4,5 → last = 5.
    plateau = sync.plateau_sample_indices(1)
    timing = resolve_prepared_frame_timing(
        sync, 1, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    assert timing.available
    assert timing.dense_sample_idx == int(max(plateau))
    evidence = collect_evidence(
        sync,
        1,
        adapters,
        selected_layers=[LayerId.S8_EXCLUSIVE_A_TO_B],
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
    )
    obs = evidence.get(LayerId.S8_EXCLUSIVE_A_TO_B)
    assert obs is not None and obs.available
    assert obs.payload["dense_sample_idx"] == timing.dense_sample_idx
    # Label must match the selected dense sample exactly (fixture: approach for early).
    assert obs.payload["behavior_label"] == "approach"


def test_first_last_prepared_frame_labels(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    adapters = _behavior_adapters(tmp_path, sync_df)
    first_prep = int(sync.prepared_frame_idx.min())
    last_prep = int(sync.prepared_frame_idx.max())
    layers = [LayerId.S8_EXCLUSIVE_A_TO_B, LayerId.DEBUG_FRAME_COUNTER]
    first = collect_evidence(sync, first_prep, adapters, selected_layers=layers)
    last = collect_evidence(sync, last_prep, adapters, selected_layers=layers)
    assert first.timing.available and last.timing.available
    assert first.timing.dense_sample_idx == int(max(sync.plateau_sample_indices(first_prep)))
    assert last.timing.dense_sample_idx == int(max(sync.plateau_sample_indices(last_prep)))
    # Last sample in fixture has null A→B label → unavailable, distinguishable.
    last_obs = last.get(LayerId.S8_EXCLUSIVE_A_TO_B)
    assert last_obs is not None
    assert last_obs.available is False
    composed = OverlayCompositor().render_frame(
        solid_frame(h=72, w=128),
        prepared_frame_idx=last_prep,
        evidence=last,
        selected_layers=layers,
        style=DrawConfig(unavailable_badge=True),
    )
    # Unavailable exclusive row still painted (not silent omission).
    assert composed.unavailable_layers == (LayerId.S8_EXCLUSIVE_A_TO_B,)
    assert not np.array_equal(composed.frame_bgr, solid_frame(h=72, w=128))


def test_unknown_vs_active_contact_distinguishable(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    adapters = _behavior_adapters(tmp_path, sync_df)
    # Missing coverage prepared frame → contact unavailable.
    from tests.overlay._fixtures import build_gap_sync_with_missing_prepared

    gap = SyncMap.from_sync_index(build_gap_sync_with_missing_prepared())
    evidence = collect_evidence(
        gap, 2, adapters, selected_layers=[LayerId.S8_CONTACT]
    )
    obs = evidence.get(LayerId.S8_CONTACT)
    assert obs is not None
    assert obs.available is False
    composed = OverlayCompositor().render_frame(
        solid_frame(h=64, w=96),
        prepared_frame_idx=2,
        evidence=evidence,
        selected_layers=[LayerId.S8_CONTACT],
    )
    assert LayerId.S8_CONTACT in composed.unavailable_layers
