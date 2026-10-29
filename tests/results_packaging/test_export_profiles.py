"""Tests for optional review_emit_v1 export-profile selection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from results_packaging.bouts import package_behavior_bouts
from results_packaging.contracts import (
    REVIEW_EMIT_PROFILE_ID,
    ExportProfileError,
)
from results_packaging.export_profiles import (
    ReviewEmitV1Params,
    apply_review_emit_v1,
    review_emit_v1_params_from_mapping,
)
from results_packaging.sync_index import build_sync_index_from_prepared_sync
from tests.results_packaging._fixtures import (
    build_packaging_session,
    make_multi_channel_bouts,
)


def _packaged(tmp_path) -> pd.DataFrame:
    fixture = build_packaging_session(tmp_path, n_prepared=20, prepared_fps=30.0)
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
    return package_behavior_bouts(
        s8, sync, source_s8_run_id="behavioral_arbitration__synth"
    )


def test_default_params_and_120hz_floor_combination() -> None:
    params = ReviewEmitV1Params().validated()
    assert params.include_contact is False
    assert params.include_engaged is True
    assert params.include_side_by_side is True
    # max(10, ceil(0.10 * 120)) = max(10, 12) = 12
    assert params.min_required_frames() == max(
        10, int(math.ceil(0.10 * 120.0))
    )
    assert params.min_required_frames() == 12


def test_frame_and_seconds_floor_boundary_cases() -> None:
    # Seconds floor dominates when larger after 120 Hz conversion.
    p_sec = ReviewEmitV1Params(
        min_emit_bout_frames=5,
        min_emit_bout_seconds=0.10,
        target_fps=120.0,
    )
    assert p_sec.min_required_frames() == 12

    # Frame floor dominates when larger.
    p_frames = ReviewEmitV1Params(
        min_emit_bout_frames=20,
        min_emit_bout_seconds=0.05,
        target_fps=120.0,
    )
    assert p_frames.min_required_frames() == 20

    # Exact equality at boundary: duration == min_required is included.
    assert int(math.ceil(0.05 * 120.0)) == 6


def test_rejects_source_video_fps_parameter() -> None:
    with pytest.raises(ExportProfileError, match="source-video FPS"):
        review_emit_v1_params_from_mapping({"source_video_fps": 30.0})
    with pytest.raises(ExportProfileError, match="120"):
        ReviewEmitV1Params(target_fps=30.0).validated()


def test_review_emit_defaults_exclude_contact_keep_engaged(tmp_path) -> None:
    packaged = _packaged(tmp_path)
    result = apply_review_emit_v1(packaged)
    assert result.profile_id == REVIEW_EMIT_PROFILE_ID
    assert result.n_included + result.n_excluded == len(packaged)

    # Contact excluded by default.
    assert (result.bout_index["channel"] != "contact").all() or result.bout_index.empty
    contact_excluded = result.exclusion_table[
        result.exclusion_table["excluded_reason"] == "channel_contact_excluded"
    ]
    assert len(contact_excluded) == int((packaged["channel"] == "contact").sum())

    # Engaged exclusive bouts that pass the floor remain.
    engaged_src = packaged[
        (packaged["channel"] == "exclusive")
        & (packaged["behavior_name"] == "engaged")
        & (packaged["bout_duration_frames"] >= 12)
    ]
    engaged_idx = result.bout_index[
        (result.bout_index["channel"] == "exclusive")
        & (result.bout_index["behavior_name"] == "engaged")
    ]
    assert set(engaged_idx["source_s8_bout_id"]) == set(
        engaged_src["source_s8_bout_id"].astype(str)
    )


def test_emit_floor_boundary_and_no_boundary_rewrite(tmp_path) -> None:
    packaged = _packaged(tmp_path)
    # Force known durations around the floor.
    work = packaged.copy()
    # Keep one row at exactly min_required=12 and one at 11.
    idxs = work.index.tolist()
    work.loc[idxs[0], "bout_duration_frames"] = 12
    work.loc[idxs[0], "channel"] = "exclusive"
    work.loc[idxs[0], "behavior_name"] = "approach"
    work.loc[idxs[0], "start_sample_idx"] = 100
    work.loc[idxs[0], "end_sample_idx"] = 111
    work.loc[idxs[1], "bout_duration_frames"] = 11
    work.loc[idxs[1], "channel"] = "exclusive"
    work.loc[idxs[1], "behavior_name"] = "approach"
    work.loc[idxs[1], "start_sample_idx"] = 200
    work.loc[idxs[1], "end_sample_idx"] = 210

    result = apply_review_emit_v1(
        work,
        params=ReviewEmitV1Params(
            include_contact=True,
            include_side_by_side=True,
            include_engaged=True,
        ),
    )
    included_ids = set(result.bout_index["source_s8_bout_id"].astype(str))
    assert str(work.loc[idxs[0], "source_s8_bout_id"]) in included_ids
    assert str(work.loc[idxs[1], "source_s8_bout_id"]) not in included_ids

    # Included rows preserve dense endpoints.
    row = result.bout_index[
        result.bout_index["source_s8_bout_id"]
        == str(work.loc[idxs[0], "source_s8_bout_id"])
    ].iloc[0]
    assert int(row["start_sample_idx"]) == 100
    assert int(row["end_sample_idx"]) == 111
    assert int(row["bout_duration_frames"]) == 12


def test_exclusion_reason_accounting_priority(tmp_path) -> None:
    packaged = _packaged(tmp_path)
    # Contact bout that is also short: primary reason is channel_contact_excluded.
    contact = packaged[packaged["channel"] == "contact"].iloc[0].copy()
    contact["bout_duration_frames"] = 1
    tiny = pd.DataFrame([contact])
    result = apply_review_emit_v1(tiny)
    assert result.n_excluded == 1
    assert result.exclusion_table.iloc[0]["excluded_reason"] == "channel_contact_excluded"
    assert "below_emit_floor" in str(result.exclusion_table.iloc[0]["matched_reasons"])


def test_complete_package_unaffected_when_profile_not_selected(tmp_path) -> None:
    packaged = _packaged(tmp_path)
    # Applying profile returns a view; source table unchanged.
    before = packaged.copy()
    _ = apply_review_emit_v1(packaged)
    pd.testing.assert_frame_equal(packaged, before)


def test_source_bout_ids_retained_in_index(tmp_path) -> None:
    packaged = _packaged(tmp_path)
    result = apply_review_emit_v1(packaged)
    if result.bout_index.empty:
        pytest.skip("fixture produced empty review index")
    assert result.bout_index["source_s8_bout_id"].isin(
        packaged["source_s8_bout_id"].astype(str)
    ).all()
    assert result.bout_index["bout_id"].isin(packaged["bout_id"].astype(str)).all()
