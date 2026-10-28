"""End-to-end Sprint 1 headless runner on synthetic session."""

from __future__ import annotations

from pathlib import Path

from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.runner import (
    BehavioralArbitrationRequest,
    run_behavioral_arbitration_sprint1,
)
from tests.behavioral_arbitration._fixtures import build_synthetic_session
from tests.behavioral_arbitration.test_intake import _patch_eligibility


def test_sprint1_runner_produces_directional_result(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path)
    patches = _patch_eligibility(info)
    config = load_behavioral_arbitration_config(None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = run_behavioral_arbitration_sprint1(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            )
        )

    assert result.n_directional_rows == result.n_frames * 2
    assert result.mouse_ids == (1, 2)
    assert "behavior_label" not in result.frames.columns
    assert "prob_contact" in result.frames.columns
    assert "prob_side_by_side" in result.frames.columns
    assert "distance_nose_i_nose_j_norm" in result.frames.columns
    assert "evidence_state_contact" in result.frames.columns
    assert not result.frames.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()
    stage_names = {d.stage for d in result.diagnostics}
    assert "expand" in stage_names
    assert "validity" in stage_names
    assert "merge_s6" in stage_names
    # hard_distance diagnostics appear only when rows are gated.
    assert "postural_approach" in stage_names
    assert "postural_retract" in stage_names
    meta = result.to_metadata()
    assert meta["source_s7_run_id"] == info["s7_run_id"]
    assert meta["source_s6_run_id"] == info["s6_run_id"]


def test_config_defaults_match_legacy_numeric_values() -> None:
    config = load_behavioral_arbitration_config(None)
    sci = config.scientific
    assert sci.hard_gap_nose_to_nose_distance_norm == 0.3
    assert sci.hard_gap_nose_to_body_distance_norm == 0.55
    assert sci.hard_gap_nose_to_tail_distance_norm == 0.45
    assert sci.hard_gap_side_by_side_distance_norm == 0.9
    assert sci.hard_gap_contact_spine_to_spine_distance_norm == 1.2
    assert sci.postural.approach.rearing_contraction_gt == 5.0
    assert sci.postural.retract.stationary_speed_lt == 15.0
    assert sci.contact_post_processing.max_gap_to_fill == 0
    assert sci.hierarchy.layer1_significance_floor == 0.05
    assert sci.temporal.gap_fill_seconds == 0.050
    assert config.execution.apply_pipeline_110_emit_filters is False
    assert config.require_exactly_two_mice is True
