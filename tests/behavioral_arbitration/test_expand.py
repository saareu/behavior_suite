"""Directional expansion cardinality, identity, and order invariance."""

from __future__ import annotations

import pandas as pd
import pytest

from behavioral_arbitration.contracts import ExpansionError
from behavioral_arbitration.expand import (
    assert_source_order_invariant,
    expand_dyadic_to_directional,
)
from tests.behavioral_arbitration._fixtures import (
    MOUSE_A,
    MOUSE_B,
    N_FRAMES,
    make_dyadic_wide,
    make_focal_wide,
)


def test_dyadic_expansion_cardinality() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    out, diag = expand_dyadic_to_directional(dyadic, focal)
    assert len(out) == N_FRAMES * 2
    assert diag.n_rows_changed == len(out)
    # Exactly two directions per frame.
    counts = out.groupby("dense_frame_idx").size()
    assert (counts == 2).all()


def test_directional_identity_canonical_mouse_a_b() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    out, _ = expand_dyadic_to_directional(dyadic, focal)
    assert (out["mouse_a"] == MOUSE_A).all()
    assert (out["mouse_b"] == MOUSE_B).all()
    # Both directions present.
    pairs = set(zip(out["focal_id"], out["partner_id"], strict=True))
    assert (MOUSE_A, MOUSE_B) in pairs
    assert (MOUSE_B, MOUSE_A) in pairs


def test_no_duplicate_directional_keys() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    out, _ = expand_dyadic_to_directional(dyadic, focal)
    assert not out.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()


def test_unordered_pair_source_order_invariance() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    shuffled = dyadic.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert_source_order_invariant(dyadic, shuffled, focal)


def test_symmetric_dyadic_same_in_both_directions() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    out, _ = expand_dyadic_to_directional(dyadic, focal)
    for _frame_idx, group in out.groupby("dense_frame_idx"):
        assert len(group) == 2
        for det in ("nose_to_nose", "contact", "side_by_side", "mutually_disengaged"):
            vals = group[f"prob_{det}"].tolist()
            assert vals[0] == vals[1]


def test_focal_predictions_remain_directional() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    out, _ = expand_dyadic_to_directional(dyadic, focal)
    # Approach positive only on A→B frame 1 in fixture.
    a_to_b = out[(out["focal_id"] == MOUSE_A) & (out["dense_frame_idx"] == 1)]
    b_to_a = out[(out["focal_id"] == MOUSE_B) & (out["dense_frame_idx"] == 1)]
    assert int(a_to_b["pred_approach"].iloc[0]) == 1
    assert int(b_to_a["pred_approach"].iloc[0]) == 0


def test_timestamp_alignment_preserved() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    out, _ = expand_dyadic_to_directional(dyadic, focal)
    merged = out.merge(
        focal[["dense_frame_idx", "focal_id", "partner_id", "time_sec"]],
        on=["dense_frame_idx", "focal_id", "partner_id"],
        suffixes=("", "_focal"),
    )
    assert (merged["time_sec"] == merged["time_sec_focal"]).all()


def test_missing_required_dyadic_hard_fails() -> None:
    dyadic = make_dyadic_wide()
    dyadic = dyadic.drop(columns=["prob_contact", "pred_contact"])
    focal = make_focal_wide()
    with pytest.raises(ExpansionError, match="missing required detector"):
        expand_dyadic_to_directional(dyadic, focal)


def test_merge_miss_hard_fails_not_zero_fill() -> None:
    dyadic = make_dyadic_wide(n_frames=2)
    # Drop one undirected frame so expansion misses.
    dyadic = dyadic[dyadic["dense_frame_idx"] != 1].reset_index(drop=True)
    focal = make_focal_wide(n_frames=2)
    with pytest.raises(ExpansionError, match="refusing legacy zero-fill"):
        expand_dyadic_to_directional(dyadic, focal)


def test_duplicate_dyadic_keys_fail() -> None:
    dyadic = make_dyadic_wide(n_frames=1)
    dyadic = pd.concat([dyadic, dyadic], ignore_index=True)
    focal = make_focal_wide(n_frames=1)
    with pytest.raises(ExpansionError, match="duplicate undirected"):
        expand_dyadic_to_directional(dyadic, focal)
