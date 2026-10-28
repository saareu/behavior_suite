"""Bout extraction — inclusive endpoints, splits, parallel channels."""

from __future__ import annotations

import pandas as pd

from behavioral_arbitration.bouts import extract_behavior_bouts
from tests.behavioral_arbitration._fixtures import (
    FPS,
    MOUSE_A,
    MOUSE_B,
    make_directional_temporal_frame,
    make_directional_timeline,
)


def test_inclusive_endpoints_and_duration_at_120hz() -> None:
    # Frames 10..14 inclusive → 5 frames → 5/120 sec.
    labels = ["nose_to_body"] * 5
    df = make_directional_timeline(labels, start_frame=10)
    bouts = extract_behavior_bouts(
        df,
        fps=FPS,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    exclusive = bouts[bouts["channel"] == "exclusive"]
    assert len(exclusive) == 1
    row = exclusive.iloc[0]
    assert int(row["start_sample_idx"]) == 10
    assert int(row["end_sample_idx"]) == 14
    assert int(row["bout_duration_frames"]) == 5
    assert abs(float(row["bout_duration_sec"]) - (5.0 / 120.0)) < 1e-12


def test_split_on_behavior_change() -> None:
    labels = ["approach"] * 3 + ["retract"] * 2 + ["approach"] * 2
    df = make_directional_timeline(labels)
    bouts = extract_behavior_bouts(
        df,
        fps=FPS,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    exclusive = bouts[bouts["channel"] == "exclusive"].reset_index(drop=True)
    assert list(exclusive["behavior_name"]) == ["approach", "retract", "approach"]
    assert list(exclusive["bout_duration_frames"]) == [3, 2, 2]


def test_split_on_frame_discontinuity() -> None:
    a = make_directional_timeline(["nose_to_tail"] * 3, start_frame=0)
    b = make_directional_timeline(["nose_to_tail"] * 2, start_frame=10)
    df = pd.concat([a, b], ignore_index=True)
    bouts = extract_behavior_bouts(
        df,
        fps=FPS,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    exclusive = bouts[bouts["channel"] == "exclusive"]
    assert len(exclusive) == 2
    assert list(exclusive["bout_duration_frames"]) == [3, 2]
    assert list(exclusive["start_sample_idx"]) == [0, 10]


def test_parallel_contact_sbs_independent_bouts() -> None:
    labels = ["engaged"] * 6
    df = make_directional_timeline(labels, focal=MOUSE_A, partner=MOUSE_B)
    reverse = make_directional_timeline(
        labels, focal=MOUSE_B, partner=MOUSE_A
    )
    df = pd.concat([df, reverse], ignore_index=True)
    # Contact positive frames 0-2; SBS positive frames 3-5 (undirected collapse).
    df["pred_contact"] = 0
    df["pred_side_by_side"] = 0
    df.loc[df["dense_frame_idx"].isin([0, 1, 2]), "pred_contact"] = 1
    df.loc[df["dense_frame_idx"].isin([3, 4, 5]), "pred_side_by_side"] = 1
    df["prob_contact"] = df["pred_contact"].astype(float) * 0.9
    df["prob_side_by_side"] = df["pred_side_by_side"].astype(float) * 0.8
    bouts = extract_behavior_bouts(
        df,
        fps=FPS,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    contact = bouts[bouts["channel"] == "contact"]
    sbs = bouts[bouts["channel"] == "side_by_side"]
    assert len(contact) == 1
    assert len(sbs) == 1
    assert int(contact.iloc[0]["bout_duration_frames"]) == 3
    assert int(sbs.iloc[0]["bout_duration_frames"]) == 3
    assert contact.iloc[0]["identity_grain"] == "symmetric_pair"
    assert pd.isna(contact.iloc[0]["focal_id"])


def test_directional_identity_change_splits() -> None:
    rows = []
    for fi in range(3):
        rows.append(
            make_directional_temporal_frame(
                frame=fi, focal=MOUSE_A, partner=MOUSE_B, behavior_label="approach"
            )
        )
    for fi in range(3):
        rows.append(
            make_directional_temporal_frame(
                frame=fi, focal=MOUSE_B, partner=MOUSE_A, behavior_label="approach"
            )
        )
    df = pd.DataFrame(rows)
    bouts = extract_behavior_bouts(
        df,
        fps=FPS,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    exclusive = bouts[bouts["channel"] == "exclusive"]
    assert len(exclusive) == 2
    pairs = set(
        zip(
            exclusive["focal_id"].astype(int),
            exclusive["partner_id"].astype(int),
            strict=True,
        )
    )
    assert pairs == {(MOUSE_A, MOUSE_B), (MOUSE_B, MOUSE_A)}
