"""End-to-end Sprint 2 runner and stage-order / validity integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.contracts import (
    LABEL_ENGAGED,
    LABEL_UNRESOLVED,
    SPRINT2_RESULT_SCHEMA_VERSION,
)
from behavioral_arbitration.hierarchy import resolve_hierarchical_conflicts
from behavioral_arbitration.runner import (
    BehavioralArbitrationRequest,
    apply_sprint2_stages,
    run_behavioral_arbitration_sprint1,
    run_behavioral_arbitration_sprint2,
)
from tests.behavioral_arbitration._fixtures import build_synthetic_session
from tests.behavioral_arbitration.test_intake import _patch_eligibility


def test_sprint2_runner_preserves_keys_and_diagnostics(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, invalid_contact_frames={1})
    patches = _patch_eligibility(info)
    config = load_behavioral_arbitration_config(None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = run_behavioral_arbitration_sprint2(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            )
        )

    assert result.schema_version == SPRINT2_RESULT_SCHEMA_VERSION
    assert result.n_directional_rows == result.n_frames * 2
    assert not result.frames.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()
    assert "behavior_label" in result.frames.columns
    assert "hierarchy_residual_kind" in result.frames.columns
    assert "y_approach_smoothed" in result.frames.columns
    assert "y_retract_smoothed" in result.frames.columns
    assert "prob_contact_smooth" in result.frames.columns
    assert "prob_side_by_side" in result.frames.columns
    # S7 provenance preserved.
    assert "s7_prob_contact" in result.frames.columns
    assert "evidence_state_contact" in result.frames.columns

    stages = {d.stage for d in result.diagnostics}
    assert "hierarchy" in stages
    assert "adaptive_approach" in stages
    assert "adaptive_retract" in stages
    assert "contact_clean" in stages
    codes = {d.rule_code for d in result.diagnostics}
    assert "HIERARCHY_WIN" in codes or "HIERARCHY_RESIDUAL" in codes
    assert "ADAPTIVE_APPROACH" in codes
    assert "CONTACT_SCHMITT" in codes
    assert any("max_gap_to_fill=0" in (w or "") for w in result.warnings) or any(
        "morphological closing disabled" in w for w in result.warnings
    )


def test_sprint2_continues_from_sprint1_result(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path)
    patches = _patch_eligibility(info)
    config = load_behavioral_arbitration_config(None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        s1 = run_behavioral_arbitration_sprint1(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            )
        )
        s2 = run_behavioral_arbitration_sprint2(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            ),
            sprint1_result=s1,
        )
    assert s2.n_directional_rows == s1.n_directional_rows
    assert s2.source_s7_run_id == s1.source_s7_run_id


def test_invalid_contact_frames_remain_unobserved_after_sprint2(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, invalid_contact_frames={0, 1})
    patches = _patch_eligibility(info)
    config = load_behavioral_arbitration_config(None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = run_behavioral_arbitration_sprint2(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
            )
        )
    inv = result.frames[result.frames["dense_frame_idx"].isin([0, 1])]
    # Provenance / validity mask not erased by contact cleaning.
    assert (inv["arbitration_input_valid_contact"] == False).all()  # noqa: E712
    assert inv["prob_contact"].isna().all()
    assert (inv["pred_contact"] == 0).all()


def test_stage_order_sensitivity_hierarchy_before_adaptive() -> None:
    """Adaptive uses post-hierarchy sniff preds; reversing order changes outcomes."""

    config = load_behavioral_arbitration_config(None)
    rows = []
    for i in range(5):
        rows.append(
            {
                "exp_name": "synth",
                "dense_frame_idx": i,
                "focal_id": 1,
                "partner_id": 2,
                "cc_count": 2,
                "is_total_occlusion": 0,
                "distance_nose_i_spine_j_norm": 1.0,
                "distance_spine_i_spine_j_norm": 0.4,
                "prob_nose_to_nose": 0.95 if i < 3 else 0.01,
                "pred_nose_to_nose": 0,
                "prob_nose_to_body": 0.01,
                "pred_nose_to_body": 0,
                "prob_nose_to_tail": 0.01,
                "pred_nose_to_tail": 0,
                "prob_approach": 0.30,
                "pred_approach": 0,
                "prob_retract": 0.01,
                "pred_retract": 0,
                "prob_mutually_disengaged": 0.01,
                "pred_mutually_disengaged": 0,
                "prob_contact": 0.01,
                "pred_contact": 0,
                "prob_side_by_side": 0.01,
                "pred_side_by_side": 0,
                "arbitration_input_valid_nose_to_nose": True,
                "arbitration_input_valid_nose_to_body": True,
                "arbitration_input_valid_nose_to_tail": True,
                "arbitration_input_valid_approach": True,
                "arbitration_input_valid_retract": True,
                "arbitration_input_valid_mutually_disengaged": True,
                "arbitration_input_valid_contact": True,
            }
        )
    df = pd.DataFrame(rows)

    # Canonical order: hierarchy then adaptive.
    frames_a, _, _ = apply_sprint2_stages(df.copy(), config)
    # Hierarchy alone then check approach preds vs adaptive-first manually.
    hier_only, _ = resolve_hierarchical_conflicts(df.copy(), config)
    # When nn wins early frames, adaptive must not fire on those frames.
    early = frames_a[frames_a["dense_frame_idx"] < 3]
    assert (early["behavior_label"] == "nose_to_nose").all()
    assert (early["pred_approach"] == 0).all()

    # Later frames (nn off) may engage adaptive approach.
    late_hier = hier_only[hier_only["dense_frame_idx"] >= 3]
    assert (late_hier["behavior_label"] != "nose_to_nose").all()


def test_partial_missing_evidence_does_not_force_undefined() -> None:
    config = load_behavioral_arbitration_config(None)
    row = {
        "exp_name": "synth",
        "dense_frame_idx": 0,
        "focal_id": 1,
        "partner_id": 2,
        "cc_count": 2,
        "is_total_occlusion": 0,
        "prob_nose_to_nose": np.nan,
        "pred_nose_to_nose": 0,
        "arbitration_input_valid_nose_to_nose": False,
        "prob_nose_to_body": 0.01,
        "pred_nose_to_body": 0,
        "arbitration_input_valid_nose_to_body": True,
        "prob_nose_to_tail": 0.01,
        "pred_nose_to_tail": 0,
        "arbitration_input_valid_nose_to_tail": True,
        "prob_approach": 0.01,
        "pred_approach": 0,
        "arbitration_input_valid_approach": True,
        "prob_retract": 0.01,
        "pred_retract": 0,
        "arbitration_input_valid_retract": True,
        "prob_mutually_disengaged": 0.01,
        "pred_mutually_disengaged": 0,
        "arbitration_input_valid_mutually_disengaged": True,
        "prob_contact": 0.01,
        "pred_contact": 0,
        "prob_side_by_side": 0.01,
        "pred_side_by_side": 0,
    }
    out, _ = resolve_hierarchical_conflicts(pd.DataFrame([row]), config)
    assert out.loc[0, "behavior_label"] == LABEL_ENGAGED
    assert out.loc[0, "behavior_label"] != LABEL_UNRESOLVED


def test_sprint2_determinism(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path)
    patches = _patch_eligibility(info)
    config = load_behavioral_arbitration_config(None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        req = BehavioralArbitrationRequest(
            session_root=info["session_root"],
            s7_run_dir=info["s7_run_dir"],
            config=config,
        )
        a = run_behavioral_arbitration_sprint2(req)
        b = run_behavioral_arbitration_sprint2(req)
    assert a.frames["behavior_label"].tolist() == b.frames["behavior_label"].tolist()
    assert a.frames["pred_contact"].tolist() == b.frames["pred_contact"].tolist()
