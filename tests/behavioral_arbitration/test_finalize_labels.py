"""Finalize — unresolved invalid → undefined; engaged_valid stays engaged."""

from __future__ import annotations

from behavioral_arbitration.contracts import (
    LABEL_ENGAGED,
    LABEL_UNDEFINED,
    LABEL_UNRESOLVED,
    RESIDUAL_ENGAGED_VALID,
    RESIDUAL_UNRESOLVED_INVALID,
)
from behavioral_arbitration.finalize import finalize_behavior_labels
from tests.behavioral_arbitration._fixtures import make_directional_timeline


def test_unresolved_invalid_becomes_undefined() -> None:
    df = make_directional_timeline([LABEL_UNRESOLVED] * 4)
    df["hierarchy_residual_kind"] = RESIDUAL_UNRESOLVED_INVALID
    for det in (
        "nose_to_nose",
        "nose_to_body",
        "nose_to_tail",
        "approach",
        "retract",
        "mutually_disengaged",
        "engaged",
    ):
        df[f"pred_{det}"] = 0
    out, diag = finalize_behavior_labels(df)
    assert (out["behavior_label"] == LABEL_UNDEFINED).all()
    assert (out["undefined_reason"] == RESIDUAL_UNRESOLVED_INVALID).all()
    assert diag.n_rows_changed == 4
    assert diag.rule_code == "LABEL_UNDEFINED"
    # Pred sync: exclusive preds stay zero (no pred_undefined).
    assert (out["pred_engaged"] == 0).all()
    assert (out["pred_nose_to_nose"] == 0).all()


def test_engaged_valid_stays_engaged() -> None:
    df = make_directional_timeline([LABEL_ENGAGED] * 3)
    df["hierarchy_residual_kind"] = RESIDUAL_ENGAGED_VALID
    df["pred_engaged"] = 1
    out, diag = finalize_behavior_labels(df)
    assert (out["behavior_label"] == LABEL_ENGAGED).all()
    assert (out["undefined_reason"] == "").all()
    assert diag.n_rows_changed == 0
    assert (out["pred_engaged"] == 1).all()


def test_all_negative_valid_stays_engaged_not_undefined() -> None:
    """Valid residual with no exclusive winner remains engaged (P4)."""

    df = make_directional_timeline([LABEL_ENGAGED] * 5)
    df["hierarchy_residual_kind"] = RESIDUAL_ENGAGED_VALID
    for det in (
        "nose_to_nose",
        "nose_to_body",
        "nose_to_tail",
        "approach",
        "retract",
        "mutually_disengaged",
    ):
        df[f"pred_{det}"] = 0
        df[f"prob_{det}"] = 0.01
    df["pred_engaged"] = 1
    out, _ = finalize_behavior_labels(df)
    assert (out["behavior_label"] == LABEL_ENGAGED).all()
    assert not (out["behavior_label"] == LABEL_UNDEFINED).any()


def test_undefined_reason_recorded_and_preds_cleared() -> None:
    df = make_directional_timeline(
        [LABEL_UNRESOLVED, LABEL_ENGAGED, LABEL_UNRESOLVED]
    )
    df["hierarchy_residual_kind"] = [
        RESIDUAL_UNRESOLVED_INVALID,
        RESIDUAL_ENGAGED_VALID,
        RESIDUAL_UNRESOLVED_INVALID,
    ]
    df.loc[0, "pred_approach"] = 0
    df.loc[1, "pred_engaged"] = 1
    df.loc[2, "pred_nose_to_body"] = 0
    out, _ = finalize_behavior_labels(df)
    assert out.loc[0, "behavior_label"] == LABEL_UNDEFINED
    assert out.loc[0, "undefined_reason"] == RESIDUAL_UNRESOLVED_INVALID
    assert out.loc[1, "behavior_label"] == LABEL_ENGAGED
    assert out.loc[1, "undefined_reason"] == ""
    assert out.loc[2, "behavior_label"] == LABEL_UNDEFINED
    # Undefined rows clear exclusive preds.
    assert out.loc[0, "pred_engaged"] == 0
    assert out.loc[0, "pred_approach"] == 0
    assert out.loc[1, "pred_engaged"] == 1
