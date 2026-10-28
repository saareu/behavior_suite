"""Stage 10 — gap fill / min-duration formulas and invalid-span safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from behavioral_arbitration.temporal import (
    apply_min_duration,
    fill_gaps,
    fill_small_gaps_and_filter_duration,
)
from tests.behavioral_arbitration._fixtures import make_directional_timeline


def test_fill_gaps_legacy_formula_and_zero_max() -> None:
    seq = np.array([1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
    filled = fill_gaps(seq, max_gap=2)
    assert filled.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    # gap of length 2 > max_gap=1 → no fill between first bout and middle.
    filled_tight = fill_gaps(seq, max_gap=1)
    assert filled_tight.tolist() == [1, 1, 0, 0, 1, 1, 1, 1]

    unchanged = fill_gaps(seq, max_gap=0)
    assert unchanged.tolist() == seq.tolist()


def test_apply_min_duration_removes_short_bouts() -> None:
    seq = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=int)
    filtered = apply_min_duration(seq, min_duration=3)
    assert filtered.tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    assert apply_min_duration(seq, min_duration=1).tolist() == seq.tolist()


def test_gap_frames_zero_no_fill_on_preds() -> None:
    labels = ["nose_to_body"] * 3 + ["engaged"] * 2 + ["nose_to_body"] * 3
    df = make_directional_timeline(labels)
    # Explicit positive island with a 2-frame zero gap on pred_nose_to_body.
    df["pred_nose_to_body"] = [1, 1, 1, 0, 0, 1, 1, 1]
    out, diags = fill_small_gaps_and_filter_duration(
        df,
        gap_frames=0,
        min_duration_frames=1,
    )
    assert out["pred_nose_to_body"].tolist() == [1, 1, 1, 0, 0, 1, 1, 1]
    gap_diag = next(d for d in diags if d.rule_code == "GAP_FILL")
    assert gap_diag.n_rows_changed == 0
    assert "gap_frames=0" in (gap_diag.detail or "")


def test_positive_gap_window_fills_and_is_deterministic() -> None:
    labels = ["nose_to_body"] * 3 + ["engaged"] * 2 + ["nose_to_body"] * 3
    df = make_directional_timeline(labels)
    df["pred_nose_to_body"] = [1, 1, 1, 0, 0, 1, 1, 1]
    out_a, diags_a = fill_small_gaps_and_filter_duration(
        df,
        gap_frames=2,
        min_duration_frames=1,
    )
    out_b, diags_b = fill_small_gaps_and_filter_duration(
        df,
        gap_frames=2,
        min_duration_frames=1,
    )
    assert out_a["pred_nose_to_body"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert out_a["pred_nose_to_body"].tolist() == out_b["pred_nose_to_body"].tolist()
    assert out_a["label_reconstruction"].tolist() == out_b["label_reconstruction"].tolist()
    assert diags_a[0].n_rows_changed == diags_b[0].n_rows_changed == 2
    assert (out_a["label_reconstruction"] == "gap_filled").sum() == 2


def test_invalid_span_skips_gap_fill() -> None:
    labels = ["nose_to_body"] * 3 + ["engaged"] * 2 + ["nose_to_body"] * 3
    df = make_directional_timeline(labels)
    df["pred_nose_to_body"] = [1, 1, 1, 0, 0, 1, 1, 1]
    df["arbitration_input_valid_nose_to_body"] = [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    out, diags = fill_small_gaps_and_filter_duration(
        df,
        gap_frames=2,
        min_duration_frames=1,
    )
    # Invalid frame forced to 0; gap containing invalid is not filled.
    assert out["pred_nose_to_body"].tolist() == [1, 1, 1, 0, 0, 1, 1, 1]
    gap_diag = next(d for d in diags if d.rule_code == "GAP_FILL")
    assert gap_diag.n_rows_changed == 0
    assert "gaps_skipped_invalid=1" in (gap_diag.detail or "")


def test_md_override_min_duration() -> None:
    # Short MD bout of 3 frames should survive default min=1 but fail override=5.
    labels = ["engaged"] * 2 + ["mutually_disengaged"] * 3 + ["engaged"] * 2
    df = make_directional_timeline(labels)
    df["pred_mutually_disengaged"] = [0, 0, 1, 1, 1, 0, 0]
    df["pred_engaged"] = [1, 1, 0, 0, 0, 1, 1]
    out, diags = fill_small_gaps_and_filter_duration(
        df,
        gap_frames=0,
        min_duration_frames=1,
        min_duration_overrides={"mutually_disengaged": 5},
    )
    assert out["pred_mutually_disengaged"].tolist() == [0, 0, 0, 0, 0, 0, 0]
    min_diag = next(d for d in diags if d.rule_code == "MIN_DURATION")
    assert min_diag.n_rows_changed == 3


def test_sequence_boundaries_do_not_cross_pairs() -> None:
    a = make_directional_timeline(
        ["nose_to_body", "nose_to_body", "engaged", "nose_to_body"],
        focal=1,
        partner=2,
        start_frame=0,
    )
    b = make_directional_timeline(
        ["nose_to_body", "engaged", "nose_to_body", "nose_to_body"],
        focal=2,
        partner=1,
        start_frame=0,
    )
    a["pred_nose_to_body"] = [1, 1, 0, 1]
    b["pred_nose_to_body"] = [1, 0, 1, 1]
    df = pd.concat([a, b], ignore_index=True)
    out, _ = fill_small_gaps_and_filter_duration(
        df,
        gap_frames=1,
        min_duration_frames=1,
    )
    a_out = out[(out["focal_id"] == 1) & (out["partner_id"] == 2)].sort_values(
        "dense_frame_idx"
    )
    b_out = out[(out["focal_id"] == 2) & (out["partner_id"] == 1)].sort_values(
        "dense_frame_idx"
    )
    assert a_out["pred_nose_to_body"].tolist() == [1, 1, 1, 1]
    assert b_out["pred_nose_to_body"].tolist() == [1, 1, 1, 1]
