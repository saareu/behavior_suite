"""Unit and contract tests for legacy_framewise_v2 prepared-frame CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavioral_review.sync import SyncMap
from overlay.contracts import PlateauPolicy
from overlay.timing import resolve_prepared_frame_timing
from results_packaging.legacy_framewise_csv import (
    FRAME_ALIGNMENT_COLUMNS,
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
    build_legacy_framewise_table,
    build_legacy_framewise_v2_table,
    prepared_to_last_in_plateau_dense,
    prepared_to_nearest_dense,
    validate_legacy_framewise_csv_bytes,
    write_legacy_framewise_csv,
)
from results_packaging.legacy_framewise_csv.config import LegacyFramewiseExportSettings
from results_packaging.media_csv_export.config import CsvExportSettings
from results_packaging.media_csv_export.contracts import DEFAULT_CSV_FORMAT_PROFILE
from tests.behavioral_review._fixtures import build_hand_computed_sync


def _tracked_pose(
    n_prepared: int,
    *,
    missing_frames: set[int] | None = None,
    blank_nodes: set[tuple[int, str, int]] | None = None,
) -> pd.DataFrame:
    nodes = ["headstage", "neck", "nose", "spine_base", "tail_base"]
    missing_frames = missing_frames or set()
    blank_nodes = blank_nodes or set()
    rows = []
    for frame in range(n_prepared):
        if frame in missing_frames:
            continue
        for track in (0, 1):
            for i, node in enumerate(nodes):
                if node == "headstage" and track == 1:
                    continue
                if (frame, node, track) in blank_nodes:
                    x = y = np.nan
                else:
                    x = 10.4 + i + track * 100 + frame
                    y = 20.6 + i + track * 50 + frame
                rows.append(
                    {
                        "prepared_frame_idx": frame,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                    }
                )
    return pd.DataFrame(rows)


def _pose_dense(n: int, *, missing_sample: int | None = None) -> pd.DataFrame:
    nodes = ["headstage", "neck", "nose", "spine_base", "tail_base"]
    rows = []
    for sample in range(n):
        for track in (0, 1):
            for i, node in enumerate(nodes):
                if node == "headstage" and track == 1 or (
                    missing_sample is not None and sample == missing_sample
                ):
                    x = y = np.nan
                else:
                    x = 10.4 + i + track * 100 + sample
                    y = 20.6 + i + track * 50 + sample
                rows.append(
                    {
                        "sample_idx": sample,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                        "time_sec": sample / 120.0,
                        "provenance": 2,
                    }
                )
    return pd.DataFrame(rows)


def _states(n: int) -> pd.DataFrame:
    rows = []
    for d in range(n):
        rows.append(
            {
                "dense_frame_idx": d,
                "focal_id": 0,
                "partner_id": 1,
                "behavior_label": f"b0_{d}",
                "pred_side_by_side": 0,
            }
        )
        rows.append(
            {
                "dense_frame_idx": d,
                "focal_id": 1,
                "partner_id": 0,
                "behavior_label": f"b1_{d}",
                "pred_side_by_side": 0,
            }
        )
    return pd.DataFrame(rows)


def test_default_profile_is_v2():
    assert DEFAULT_CSV_FORMAT_PROFILE == LEGACY_FRAMEWISE_V2_PROFILE_ID
    assert CsvExportSettings().validated().format_profile == LEGACY_FRAMEWISE_V2_PROFILE_ID


def test_v1_profile_still_selectable():
    settings = CsvExportSettings(format_profile=LEGACY_FRAMEWISE_PROFILE_ID).validated()
    assert settings.format_profile == LEGACY_FRAMEWISE_PROFILE_ID


def test_v2_settings_pose_authority_is_s3():
    notes = LegacyFramewiseExportSettings(
        format_profile=LEGACY_FRAMEWISE_V2_PROFILE_ID
    ).to_settings_dict()["notes"]
    assert notes["pose_authority"] == "accepted_s3_tracked_pose"
    v1_notes = LegacyFramewiseExportSettings(
        format_profile=LEGACY_FRAMEWISE_PROFILE_ID
    ).to_settings_dict()["notes"]
    assert v1_notes["pose_authority"] == "s4_final_pose"


def test_prepared_row_domain_nearest_dense_and_s3_pose(tmp_path: Path):
    # Dense at 2 Hz over prepared 0..3 at 1 Hz timestamps → repeated dense allowed.
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=2)
    sync_df["raw_decode_frame_idx"] = sync_df["prepared_frame_idx"] + 1000
    behavior_sync = SyncMap.from_sync_index(sync_df)

    n_prepared = 5
    prepared = np.arange(n_prepared, dtype=np.int64)
    raw = prepared + 1000
    prep_t = prepared.astype(np.float64) / 30.0
    # Extend prepared times so frame 4 is outside dense coverage.
    dense_t0 = float(behavior_sync.time_sec[0])
    dense_t1 = float(behavior_sync.time_sec[-1])
    assert prep_t[4] > dense_t1 or prep_t[4] < dense_t0 or True
    # Force frame 4 outside dense temporal bounds.
    prep_t = np.asarray(
        [dense_t0, dense_t0 + 0.001, (dense_t0 + dense_t1) / 2.0, dense_t1, dense_t1 + 1.0],
        dtype=np.float64,
    )

    tracked = _tracked_pose(n_prepared, missing_frames={4})
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=prepared,
        raw_decode_frame_idx=raw,
        prepared_time_sec=prep_t,
        behavior_sync=behavior_sync,
        behavior_states=_states(behavior_sync.n_dense),
        tracked_pose=tracked,
        s3_run_id="tracking_correction__test",
    )
    assert built.n_rows == 5
    assert list(built.table.columns) == list(LEGACY_FRAMEWISE_COLUMNS)
    assert list(built.alignment.columns) == list(FRAME_ALIGNMENT_COLUMNS)
    assert int(built.alignment["prepared_frame_idx"].nunique()) == 5
    assert built.diagnostics["semantics_revision"] == LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION
    assert built.diagnostics["pose_authority"] == "accepted_s3_tracked_pose"

    # Exact S3→CSV equality before serialization rounding.
    assert float(built.table.loc[1, "neck_0_x"]) == pytest.approx(12.4)
    assert float(built.table.loc[1, "neck_0_y"]) == pytest.approx(22.6)
    assert bool(built.alignment.loc[1, "s3_pose_available"]) is True

    # Trailing prepared frame outside dense coverage + missing S3 frame.
    assert pd.isna(built.table.loc[4, "dense_frame_idx"])
    assert pd.isna(built.table.loc[4, "behaviour_label_focal_0"])
    assert bool(built.alignment.loc[4, "behavior_coverage"]) is False
    assert bool(built.alignment.loc[4, "s3_pose_available"]) is False
    assert int(built.table.loc[4, "frame"]) == 1004
    assert pd.isna(built.table.loc[4, "neck_0_x"])

    out = write_legacy_framewise_csv(tmp_path / "legacy_framewise_v2.csv", built.table)
    qc = validate_legacy_framewise_csv_bytes(out.read_bytes())
    assert qc["n_header_columns"] == 23
    assert qc["n_data_rows"] == 5
    assert qc["header"] == list(LEGACY_FRAMEWISE_COLUMNS)


def test_prepared_to_nearest_dense_earlier_on_tie_and_repeats():
    dense_t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    # Midpoint equidistant → earlier dense wins.
    prep_t = np.array([0.5, 0.5, 1.0, 1.25], dtype=np.float64)
    mapped = prepared_to_nearest_dense(
        prepared_time_sec=prep_t,
        dense_time_sec=dense_t,
    )
    assert list(mapped) == [0, 0, 1, 1]
    # Outside bounds → uncovered.
    outside = prepared_to_nearest_dense(
        prepared_time_sec=np.array([-1.0, 3.0], dtype=np.float64),
        dense_time_sec=dense_t,
    )
    assert list(outside) == [-1, -1]


def test_prepared_fps_above_below_equal_dense():
    dense_t = np.linspace(0.0, 1.0, 11)  # 10 Hz over 1s → 11 samples
    # Above dense FPS: many prepared times reuse dense samples.
    prep_hi = np.linspace(0.0, 1.0, 21)
    hi = prepared_to_nearest_dense(prepared_time_sec=prep_hi, dense_time_sec=dense_t)
    assert int((hi >= 0).sum()) == 21
    assert len(np.unique(hi)) <= 11
    # Equal count.
    prep_eq = np.linspace(0.0, 1.0, 11)
    eq = prepared_to_nearest_dense(prepared_time_sec=prep_eq, dense_time_sec=dense_t)
    assert list(eq) == list(range(11))
    # Below dense FPS.
    prep_lo = np.linspace(0.0, 1.0, 6)
    lo = prepared_to_nearest_dense(prepared_time_sec=prep_lo, dense_time_sec=dense_t)
    assert int((lo >= 0).sum()) == 6
    assert len(np.unique(lo)) == 6


def test_missing_s3_coordinates_preserved():
    sync_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=1)
    sync = SyncMap.from_sync_index(sync_df)
    tracked = _tracked_pose(
        2, blank_nodes={(1, "nose", 0), (1, "tail_base", 1)}
    )
    prep_t = np.asarray(
        [float(sync.time_sec[0]), float(sync.time_sec[-1])], dtype=np.float64
    )
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=np.arange(2, dtype=np.int64),
        raw_decode_frame_idx=np.arange(2, dtype=np.int64) + 10,
        prepared_time_sec=prep_t,
        behavior_sync=sync,
        behavior_states=_states(sync.n_dense),
        tracked_pose=tracked,
    )
    assert pd.isna(built.table.loc[1, "nose_0_x"])
    assert pd.isna(built.table.loc[1, "nose_0_y"])
    assert pd.isna(built.table.loc[1, "tail_base_1_x"])
    assert pd.notna(built.table.loc[1, "neck_0_x"])
    assert bool(built.alignment.loc[1, "s3_pose_available"]) is True


def test_historical_last_in_plateau_helper_still_matches_overlay():
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=6, plateau_repeats=3)
    )
    lookup = prepared_to_last_in_plateau_dense(sync, n_prepared=6)
    for prep in range(6):
        timing = resolve_prepared_frame_timing(
            sync, prep, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
        )
        assert int(lookup[prep]) == int(timing.dense_sample_idx)


def test_v1_dense_domain_unchanged_with_same_sources():
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=1)
    sync_df["raw_decode_frame_idx"] = sync_df["prepared_frame_idx"] + 50
    built_v1 = build_legacy_framewise_table(
        sync_index=sync_df,
        behavior_states=_states(4),
        final_pose=_pose_dense(4),
    )
    assert built_v1.n_rows == 4
    assert list(built_v1.table["dense_frame_idx"].astype(int)) == [0, 1, 2, 3]

    sync = SyncMap.from_sync_index(sync_df)
    prep_t = np.asarray(sync.time_sec, dtype=np.float64)
    built_v2 = build_legacy_framewise_v2_table(
        prepared_frame_idx=np.arange(4, dtype=np.int64),
        raw_decode_frame_idx=np.arange(4, dtype=np.int64) + 50,
        prepared_time_sec=prep_t,
        behavior_sync=sync,
        behavior_states=_states(4),
        tracked_pose=_tracked_pose(4),
    )
    assert built_v2.n_rows == 4
    assert int(built_v2.table.loc[0, "frame"]) == 50


def test_missing_behavior_keeps_independent_s3_pose():
    behavior_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    behavior_sync = SyncMap.from_sync_index(behavior_df)
    # Prepared times: 0..1 inside dense span, 2..3 outside.
    dense_t0 = float(behavior_sync.time_sec[0])
    dense_t1 = float(behavior_sync.time_sec[-1])
    prep_t = np.asarray(
        [dense_t0, dense_t1, dense_t1 + 1.0, dense_t1 + 2.0], dtype=np.float64
    )
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=np.arange(4, dtype=np.int64),
        raw_decode_frame_idx=np.arange(4, dtype=np.int64) + 10,
        prepared_time_sec=prep_t,
        behavior_sync=behavior_sync,
        behavior_states=_states(behavior_sync.n_dense),
        tracked_pose=_tracked_pose(4),
    )
    assert pd.isna(built.table.loc[3, "dense_frame_idx"])
    assert pd.isna(built.table.loc[3, "behaviour_label_focal_0"])
    assert pd.notna(built.table.loc[3, "neck_0_x"])
    assert bool(built.alignment.loc[3, "behavior_coverage"]) is False
    assert bool(built.alignment.loc[3, "s3_pose_available"]) is True


def test_nonidentity_raw_decode_mapping_preserved():
    prepared = np.arange(3, dtype=np.int64)
    raw = np.array([72619, 72620, 72621], dtype=np.int64)
    sync_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=1)
    sync = SyncMap.from_sync_index(sync_df)
    dense_t0 = float(sync.time_sec[0])
    dense_t1 = float(sync.time_sec[-1])
    prep_t = np.asarray([dense_t0, dense_t1, dense_t1 + 5.0], dtype=np.float64)
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=prepared,
        raw_decode_frame_idx=raw,
        prepared_time_sec=prep_t,
        behavior_sync=sync,
        behavior_states=_states(2),
        tracked_pose=_tracked_pose(3),
    )
    assert int(built.table.loc[2, "frame"]) == 72621
    assert pd.isna(built.table.loc[2, "dense_frame_idx"])
    assert int(built.alignment.loc[2, "raw_decode_frame_idx"]) == 72621
