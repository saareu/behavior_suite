"""Stages 19–21 — reciprocity and engaged hysteresis."""

from __future__ import annotations

import pandas as pd

from behavioral_arbitration.contracts import LABEL_ENGAGED
from behavioral_arbitration.temporal import (
    apply_final_engaged_hysteresis,
    enforce_mutual_disengaged_reciprocity,
    enforce_nose_to_nose_reciprocity,
)
from tests.behavioral_arbitration._fixtures import (
    MOUSE_A,
    MOUSE_B,
    make_directional_temporal_frame,
    make_directional_timeline,
)


def _both_directions(labels_ab: list[str], labels_ba: list[str]) -> pd.DataFrame:
    a = make_directional_timeline(labels_ab, focal=MOUSE_A, partner=MOUSE_B)
    b = make_directional_timeline(labels_ba, focal=MOUSE_B, partner=MOUSE_A)
    return pd.concat([a, b], ignore_index=True)


def test_nn_reciprocity_both_directions_unchanged() -> None:
    df = _both_directions(
        ["nose_to_nose"] * 4,
        ["nose_to_nose"] * 4,
    )
    # Preserve parallel preds to confirm reciprocity does not touch them.
    df["pred_contact"] = 1
    df["pred_side_by_side"] = 0
    out, diag = enforce_nose_to_nose_reciprocity(df, fallback_label=LABEL_ENGAGED)
    assert diag.n_rows_changed == 0
    assert (out["behavior_label"] == "nose_to_nose").all()
    assert (out["pred_contact"] == 1).all()


def test_nn_reciprocity_one_sided_becomes_engaged() -> None:
    df = _both_directions(
        ["nose_to_nose"] * 3,
        ["approach"] * 3,
    )
    df["pred_contact"] = [1, 0, 1, 1, 0, 1]
    out, diag = enforce_nose_to_nose_reciprocity(df, fallback_label=LABEL_ENGAGED)
    ab = out[(out["focal_id"] == MOUSE_A) & (out["partner_id"] == MOUSE_B)]
    assert (ab["behavior_label"] == LABEL_ENGAGED).all()
    assert diag.n_rows_changed == 3
    assert (ab["pred_nose_to_nose"] == 0).all()
    assert (ab["pred_engaged"] == 1).all()
    # Parallel contact/SBS preds unchanged by reciprocity.
    assert out["pred_contact"].tolist() == df["pred_contact"].tolist()


def test_md_reciprocity_one_sided_becomes_engaged() -> None:
    df = _both_directions(
        ["mutually_disengaged"] * 3,
        ["retract"] * 3,
    )
    out, diag = enforce_mutual_disengaged_reciprocity(
        df, fallback_label=LABEL_ENGAGED
    )
    ab = out[(out["focal_id"] == MOUSE_A) & (out["partner_id"] == MOUSE_B)]
    assert (ab["behavior_label"] == LABEL_ENGAGED).all()
    assert diag.n_rows_changed == 3


def test_engaged_hysteresis_merges_short_island() -> None:
    labels = (
        ["nose_to_body"] * 3
        + ["engaged"] * 2
        + ["nose_to_body"] * 3
    )
    df = make_directional_timeline(labels)
    out, diag = apply_final_engaged_hysteresis(df, max_engaged_frames=3)
    mid = out.iloc[3:5]
    assert (mid["behavior_label"] == "nose_to_body").all()
    assert (mid["label_reconstruction"] == "hysteresis").all()
    assert diag.n_rows_changed == 2


def test_hysteresis_skips_when_neighbors_differ() -> None:
    labels = ["nose_to_body"] * 2 + ["engaged"] * 2 + ["approach"] * 2
    df = make_directional_timeline(labels)
    out, diag = apply_final_engaged_hysteresis(df, max_engaged_frames=3)
    assert (out.iloc[2:4]["behavior_label"] == LABEL_ENGAGED).all()
    assert diag.n_rows_changed == 0


def test_reciprocity_does_not_alter_parallel_sbs() -> None:
    rows = []
    for fi in range(3):
        for focal, partner, label in (
            (MOUSE_A, MOUSE_B, "nose_to_nose"),
            (MOUSE_B, MOUSE_A, "approach"),
        ):
            row = make_directional_temporal_frame(
                frame=fi, focal=focal, partner=partner, behavior_label=label
            )
            row["pred_side_by_side"] = 1
            row["prob_side_by_side"] = 0.8
            rows.append(row)
    df = pd.DataFrame(rows)
    out, _ = enforce_nose_to_nose_reciprocity(df, fallback_label=LABEL_ENGAGED)
    assert (out["pred_side_by_side"] == 1).all()
    assert (out["prob_side_by_side"] == 0.8).all()
