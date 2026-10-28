"""End-to-end Sprint 3 runner — stages 10–21 + Sprint 2 handoff invariants."""

from __future__ import annotations

from pathlib import Path

from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.contracts import (
    LABEL_UNDEFINED,
    LABEL_UNRESOLVED,
    SPRINT3_RESULT_SCHEMA_VERSION,
)
from behavioral_arbitration.runner import (
    BehavioralArbitrationRequest,
    run_behavioral_arbitration_sprint2,
    run_behavioral_arbitration_sprint3,
)
from tests.behavioral_arbitration._fixtures import build_synthetic_session
from tests.behavioral_arbitration.test_export_qc_discovery import (
    _enter_all,
    _patch_publication,
)


def test_sprint3_e2e_diagnostics_and_handoff(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=12, invalid_contact_frames={1})
    config = load_behavioral_arbitration_config(None)
    with _enter_all(_patch_publication(info)):
        s2 = run_behavioral_arbitration_sprint2(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            )
        )
        result = run_behavioral_arbitration_sprint3(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            ),
            sprint2_result=s2,
        )

    assert result.schema_version == SPRINT3_RESULT_SCHEMA_VERSION
    assert result.n_directional_rows == result.n_frames * 2
    frames = result.frames
    assert not frames.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()

    # Sprint 2 handoff: unique keys, parallel channels preserved, s7_* retained.
    assert "behavior_label" in frames.columns
    assert "hierarchy_residual_kind" in frames.columns
    assert LABEL_UNRESOLVED not in set(frames["behavior_label"].astype(str))
    assert "pred_contact" in frames.columns
    assert "pred_side_by_side" in frames.columns
    assert "prob_contact" in frames.columns
    assert "s7_prob_contact" in frames.columns
    assert "s7_prob_approach" in frames.columns
    assert "evidence_state_contact" in frames.columns

    stages = {d.stage for d in result.diagnostics}
    for required in (
        "gap_min_duration",
        "label_sync",
        "stitch_contextual",
        "stitch_dynamic",
        "stitch_idle_retract",
        "min_md_label",
        "rescue_dynamic",
        "gate_approach_close",
        "reciprocity_nose_to_nose",
        "reciprocity_mutually_disengaged",
        "hysteresis_engaged",
        "finalize",
    ):
        assert required in stages, f"missing stage {required}"

    codes = {d.rule_code for d in result.diagnostics}
    for required_code in (
        "GAP_FILL",
        "MIN_DURATION",
        "LABEL_SYNC",
        "STITCH_CONTEXTUAL",
        "STITCH_DYNAMIC",
        "STITCH_IDLE_RETRACT",
        "MIN_MD_LABEL",
        "RESCUE_DYNAMIC",
        "GATE_APPROACH_CLOSE",
        "RECIPROCITY_NOSE_TO_NOSE",
        "RECIPROCITY_MUTUALLY_DISENGAGED",
        "HYSTERESIS_ENGAGED",
    ):
        assert required_code in codes, f"missing rule {required_code}"
    assert "LABEL_FINALIZE" in codes or "LABEL_UNDEFINED" in codes

    # Invalid contact frames remain unobserved (Sprint 2 invariant through Sprint 3).
    inv = frames[frames["dense_frame_idx"] == 1]
    assert (inv["arbitration_input_valid_contact"] == False).all()  # noqa: E712


def test_sprint3_unresolved_distinct_from_engaged(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=8)
    config = load_behavioral_arbitration_config(None)
    with _enter_all(_patch_publication(info)):
        result = run_behavioral_arbitration_sprint3(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            )
        )
    labels = set(result.frames["behavior_label"].astype(str))
    # Final vocabulary: concrete ethogram, engaged, and/or undefined — not both
    # unresolved and engaged conflated for the same residual policy.
    if LABEL_UNDEFINED in labels:
        undef = result.frames[
            result.frames["behavior_label"].astype(str) == LABEL_UNDEFINED
        ]
        assert (undef["undefined_reason"].astype(str).str.len() > 0).all()
    assert LABEL_UNRESOLVED not in labels or LABEL_UNDEFINED in labels
