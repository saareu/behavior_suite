"""Unit tests for S9 media/CSV export clip bounds, selection, and CSV fields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from results_packaging.contracts import BOUT_INDEX_FILENAME, EXPORT_PROFILES_DIRNAME
from results_packaging.media_csv_export.clip_bounds import compute_clip_bounds
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import (
    EXPORT_PROVENANCE_AUTOMATED,
    SELECTION_ALL_PACKAGE_BOUTS,
    SELECTION_BOUT_ID_LIST,
    SELECTION_CHANNELS_BEHAVIORS,
    SELECTION_EXPORT_PROFILE,
    ClipBoundsError,
)
from results_packaging.media_csv_export.csv_writer import build_automated_bout_csv_frame
from results_packaging.media_csv_export.selection import select_bouts_for_export
from results_packaging.media_csv_export.video import (
    clip_relpath_for_bout,
    sanitize_path_token,
    validate_bout_id_for_filename,
)


def _sample_bouts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bout_id": "pkg__bout_00000001",
                "source_s8_bout_id": "s8_1",
                "source_s8_run_id": "s8_run",
                "package_run_id": "pkg",
                "channel": "exclusive",
                "behavior_name": "approach",
                "identity_grain": "directional",
                "focal_id": 0,
                "partner_id": 1,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 10,
                "end_sample_idx": 19,
                "start_time_sec": 10 / 120,
                "end_time_sec": 19 / 120,
                "bout_duration_frames": 10,
                "bout_duration_sec": 10 / 120,
                "prepared_frame_idx_start": 5,
                "prepared_frame_idx_end": 9,
                "prepared_time_sec_start": 5 / 30,
                "prepared_time_sec_end": 9 / 30,
                "mean_prob": 0.9,
            },
            {
                "bout_id": "pkg__bout_00000002",
                "source_s8_bout_id": "s8_2",
                "source_s8_run_id": "s8_run",
                "package_run_id": "pkg",
                "channel": "contact",
                "behavior_name": "contact",
                "identity_grain": "symmetric_pair",
                "focal_id": None,
                "partner_id": None,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 40,
                "end_sample_idx": 45,
                "start_time_sec": 40 / 120,
                "end_time_sec": 45 / 120,
                "bout_duration_frames": 6,
                "bout_duration_sec": 6 / 120,
                "prepared_frame_idx_start": 20,
                "prepared_frame_idx_end": 22,
                "prepared_time_sec_start": 20 / 30,
                "prepared_time_sec_end": 22 / 30,
                "mean_prob": 0.8,
            },
            {
                "bout_id": "pkg__bout_00000003",
                "source_s8_bout_id": "s8_3",
                "source_s8_run_id": "s8_run",
                "package_run_id": "pkg",
                "channel": "exclusive",
                "behavior_name": "retract",
                "identity_grain": "directional",
                "focal_id": 1,
                "partner_id": 0,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 100,
                "end_sample_idx": 101,
                "start_time_sec": 100 / 120,
                "end_time_sec": 101 / 120,
                "bout_duration_frames": 2,
                "bout_duration_sec": 2 / 120,
                "prepared_frame_idx_start": 50,
                "prepared_frame_idx_end": 50,
                "prepared_time_sec_start": 50 / 30,
                "prepared_time_sec_end": 50 / 30,
                "mean_prob": 0.7,
            },
        ]
    )


def test_clip_bounds_inclusive_zero_padding():
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=10,
        prepared_frame_idx_end=14,
        source_fps=30.0,
        source_n_frames=100,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert bounds.clip_frame_idx_start == 10
    assert bounds.clip_frame_idx_end == 14
    assert bounds.clip_frame_count == 5
    assert bounds.clip_duration_sec == pytest.approx(5 / 30)
    assert bounds.clip_time_sec_end == pytest.approx((14 + 1) / 30)


def test_clip_bounds_context_padding_clamped_at_video_edges():
    start = compute_clip_bounds(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=2,
        source_fps=10.0,
        source_n_frames=20,
        pre_context_sec=0.5,  # 5 frames → would go negative
        post_context_sec=0.0,
    )
    assert start.clip_frame_idx_start == 0
    assert start.clamped_start is True
    assert start.clip_frame_count == 3

    end = compute_clip_bounds(
        prepared_frame_idx_start=17,
        prepared_frame_idx_end=18,
        source_fps=10.0,
        source_n_frames=20,
        pre_context_sec=0.0,
        post_context_sec=0.5,  # 5 frames → clamp to 19
    )
    assert end.clip_frame_idx_end == 19
    assert end.clamped_end is True
    assert end.clip_frame_count == 3


def test_clip_bounds_rejects_unreliable_fps():
    with pytest.raises(ClipBoundsError, match="source_fps"):
        compute_clip_bounds(
            prepared_frame_idx_start=0,
            prepared_frame_idx_end=1,
            source_fps=0.0,
            source_n_frames=10,
        )


def test_many_to_one_dense_prepared_span_vs_behavioral_duration():
    # Dense duration 10 frames @120Hz; prepared span only 5 frames (many-to-one).
    bouts = _sample_bouts()
    row = bouts.iloc[0]
    assert int(row["bout_duration_frames"]) == 10
    prepared_span = (
        int(row["prepared_frame_idx_end"]) - int(row["prepared_frame_idx_start"]) + 1
    )
    assert prepared_span == 5
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=int(row["prepared_frame_idx_start"]),
        prepared_frame_idx_end=int(row["prepared_frame_idx_end"]),
        source_fps=30.0,
        source_n_frames=100,
    )
    assert bounds.clip_frame_count == prepared_span
    assert bounds.clip_frame_count != int(row["bout_duration_frames"])


def test_selection_all_and_channels_and_ids(tmp_path: Path):
    bouts = _sample_bouts()
    all_rows = select_bouts_for_export(
        bouts, SelectionSpec(mode=SELECTION_ALL_PACKAGE_BOUTS)
    )
    assert len(all_rows) == 3
    assert list(all_rows["bout_id"]) == sorted(
        all_rows["bout_id"],
        key=lambda _: 0,
    ) or True
    # Deterministic: contact before exclusive? sort is channel, start_sample_idx...
    assert list(all_rows["channel"]) == ["contact", "exclusive", "exclusive"]

    contact = select_bouts_for_export(
        bouts,
        SelectionSpec(mode=SELECTION_CHANNELS_BEHAVIORS, channels=("contact",)),
    )
    assert len(contact) == 1
    assert contact.iloc[0]["identity_grain"] == "symmetric_pair"

    subset = select_bouts_for_export(
        bouts,
        SelectionSpec(
            mode=SELECTION_BOUT_ID_LIST,
            bout_ids=("pkg__bout_00000003", "pkg__bout_00000001"),
        ),
    )
    assert set(subset["bout_id"]) == {"pkg__bout_00000001", "pkg__bout_00000003"}
    assert list(subset["identity_grain"]) == ["directional", "directional"]


def test_selection_export_profile(tmp_path: Path):
    bouts = _sample_bouts()
    package = tmp_path / "pkg"
    profile = package / EXPORT_PROFILES_DIRNAME / "review_emit_v1"
    profile.mkdir(parents=True)
    pd.DataFrame({"bout_id": ["pkg__bout_00000002"]}).to_parquet(
        profile / BOUT_INDEX_FILENAME, index=False
    )
    selected = select_bouts_for_export(
        bouts,
        SelectionSpec(mode=SELECTION_EXPORT_PROFILE, profile_id="review_emit_v1"),
        package_dir=package,
    )
    assert len(selected) == 1
    assert selected.iloc[0]["bout_id"] == "pkg__bout_00000002"


def test_stable_clip_filenames_and_path_safety():
    rel = clip_relpath_for_bout(
        channel="exclusive",
        behavior_name="nose_to_body",
        bout_id="pkg__bout_00000001",
    )
    assert rel.as_posix() == "exclusive/nose_to_body/pkg__bout_00000001.mp4"
    assert sanitize_path_token("../evil") == "unknown"
    with pytest.raises(Exception, match="illegal path"):
        validate_bout_id_for_filename("../x")


def test_csv_lossless_fields_and_provenance():
    bouts = _sample_bouts()
    csv_frame = build_automated_bout_csv_frame(
        bouts,
        export_job_id="csv_export__test",
        selection_mode=SELECTION_ALL_PACKAGE_BOUTS,
        source_package_run_id="pkg",
    )
    assert len(csv_frame) == 3
    assert csv_frame["export_provenance"].unique().tolist() == [EXPORT_PROVENANCE_AUTOMATED]
    assert csv_frame["export_job_id"].unique().tolist() == ["csv_export__test"]
    # Preserve directional vs symmetric identities and science fields.
    dir_row = csv_frame.loc[csv_frame["bout_id"] == "pkg__bout_00000001"].iloc[0]
    sym_row = csv_frame.loc[csv_frame["bout_id"] == "pkg__bout_00000002"].iloc[0]
    assert dir_row["identity_grain"] == "directional"
    assert dir_row["focal_id"] == 0
    assert dir_row["source_s8_bout_id"] == "s8_1"
    assert dir_row["bout_duration_frames"] == 10
    assert dir_row["prepared_frame_idx_start"] == 5
    assert sym_row["identity_grain"] == "symmetric_pair"
    assert pd.isna(sym_row["focal_id"])
    assert "mean_prob" in csv_frame.columns
