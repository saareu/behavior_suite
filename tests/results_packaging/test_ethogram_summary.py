"""Tests for ethogram summary semantics and denominators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from results_packaging.bouts import package_behavior_bouts
from results_packaging.contracts import (
    SUMMARY_KIND_BOUT_COUNTS,
    SUMMARY_KIND_LABEL_TIME,
    SUMMARY_KIND_PARALLEL_OVERLAP,
    SUMMARY_KIND_SESSION_TIMING,
)
from results_packaging.ethogram_summary import build_ethogram_summary
from results_packaging.sync_index import build_sync_index_from_prepared_sync
from tests.results_packaging._fixtures import (
    build_packaging_session,
    make_multi_channel_bouts,
    make_states,
)


def _packaged_and_sync(tmp_path):
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    unique = pd.DataFrame(
        {
            "dense_frame_idx": np.arange(fixture["n_dense"], dtype=np.int64),
            "time_sec": np.asarray(fixture["time_sec"], dtype=np.float64),
        }
    )
    sync = build_sync_index_from_prepared_sync(
        fixture["prepared_sync"],
        s8_unique_dense_times=unique,
        target_fps=120.0,
    ).sync_index
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    packaged = package_behavior_bouts(
        s8, sync, source_s8_run_id="behavioral_arbitration__synth"
    )
    states = make_states(n_dense=fixture["n_dense"], time_sec=fixture["time_sec"])
    # Mix labels for summary coverage.
    states.loc[states.index[0::4], "behavior_label"] = "approach"
    states.loc[states.index[1::4], "behavior_label"] = "engaged"
    states.loc[states.index[2::4], "behavior_label"] = "undefined"
    states.loc[states.index[3::4], "behavior_label"] = "retract"
    states["pred_contact"] = 0
    states.loc[states.index[:10], "pred_contact"] = 1
    states["pred_side_by_side"] = 0
    states.loc[states.index[10:14], "pred_side_by_side"] = 1
    return packaged, sync, states, fixture


def test_summary_kinds_denominators_and_ordering(tmp_path) -> None:
    packaged, sync, states, _fixture = _packaged_and_sync(tmp_path)
    summary = build_ethogram_summary(
        packaged_bouts=packaged,
        states=states,
        sync_index=sync,
        fps=120.0,
    )
    kinds = set(summary["summary_kind"].astype(str))
    assert kinds == {
        SUMMARY_KIND_LABEL_TIME,
        SUMMARY_KIND_BOUT_COUNTS,
        SUMMARY_KIND_PARALLEL_OVERLAP,
        SUMMARY_KIND_SESSION_TIMING,
    }
    # Deterministic sort.
    assert summary.equals(
        summary.sort_values(
            by=[
                "summary_kind",
                "channel",
                "behavior_name",
                "identity_key",
                "metric_name",
            ],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    )

    label_rows = summary[summary["summary_kind"] == SUMMARY_KIND_LABEL_TIME]
    denom = int(len(states))
    n_frames = label_rows[label_rows["metric_name"] == "n_frames"]
    assert int(n_frames["denominator_value"].iloc[0]) == denom
    assert int(n_frames["metric_value"].sum()) == denom

    engaged = label_rows[
        (label_rows["behavior_name"] == "engaged")
        & (label_rows["metric_name"] == "n_frames_special")
    ]
    assert not engaged.empty
    undefined = label_rows[
        (label_rows["behavior_name"] == "undefined")
        & (label_rows["metric_name"] == "n_frames_special")
    ]
    assert not undefined.empty


def test_parallel_channel_independence_from_exclusive(tmp_path) -> None:
    packaged, sync, states, _fixture = _packaged_and_sync(tmp_path)
    summary = build_ethogram_summary(
        packaged_bouts=packaged,
        states=states,
        sync_index=sync,
    )
    parallel = summary[summary["summary_kind"] == SUMMARY_KIND_PARALLEL_OVERLAP]
    contact_pos = parallel[
        (parallel["channel"] == "contact")
        & (parallel["metric_name"] == "n_positive_state_rows")
    ]
    assert int(contact_pos["metric_value"].iloc[0]) == 10
    # Exclusive label fractions still sum using directional denominator, independent
    # of parallel positives.
    label_frac = summary[
        (summary["summary_kind"] == SUMMARY_KIND_LABEL_TIME)
        & (summary["metric_name"] == "fraction_of_directional_samples")
    ]
    assert float(label_frac["metric_value"].sum()) == pytest.approx(1.0)


def test_does_not_infer_framewise_from_bouts_alone(tmp_path) -> None:
    packaged, sync, _states, _fixture = _packaged_and_sync(tmp_path)
    summary = build_ethogram_summary(
        packaged_bouts=packaged,
        states=None,
        sync_index=sync,
    )
    label = summary[summary["summary_kind"] == SUMMARY_KIND_LABEL_TIME]
    assert (
        label["metric_name"] == "label_time_unavailable"
    ).any()
    parallel = summary[summary["summary_kind"] == SUMMARY_KIND_PARALLEL_OVERLAP]
    assert (
        parallel["metric_name"] == "framewise_parallel_unavailable"
    ).any()
    # Bout counts still available.
    assert (
        summary["summary_kind"] == SUMMARY_KIND_BOUT_COUNTS
    ).any()
