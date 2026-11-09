"""Unit and contract tests for legacy_framewise_v2 prepared-frame CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_review.sync import SyncMap
from overlay.contracts import PlateauPolicy
from overlay.timing import resolve_prepared_frame_timing
from results_packaging.legacy_framewise_csv import (
    FRAME_ALIGNMENT_COLUMNS,
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    build_legacy_framewise_table,
    build_legacy_framewise_v2_table,
    prepared_to_last_in_plateau_dense,
    validate_legacy_framewise_csv_bytes,
    write_legacy_framewise_csv,
)
from results_packaging.media_csv_export.config import CsvExportSettings
from results_packaging.media_csv_export.contracts import DEFAULT_CSV_FORMAT_PROFILE
from tests.behavioral_review._fixtures import build_hand_computed_sync


def _pose(n: int, *, missing_sample: int | None = None) -> pd.DataFrame:
    nodes = ["headstage", "neck", "nose", "spine_base", "tail_base"]
    rows = []
    for sample in range(n):
        for track in (0, 1):
            for i, node in enumerate(nodes):
                if node == "headstage" and track == 1 or missing_sample is not None and sample == missing_sample:
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


def test_prepared_row_domain_plateau_and_header(tmp_path: Path):
    # 4 prepared frames; dense plateaus of 2 → 8 dense samples covering 0..3.
    sync_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=2)
    sync_df["raw_decode_frame_idx"] = sync_df["prepared_frame_idx"] + 1000
    behavior_sync = SyncMap.from_sync_index(sync_df)
    # Pose uses a different dense count / map (3 prepared covered, 6 dense).
    pose_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    pose_sync = SyncMap.from_sync_index(pose_df)

    n_prepared = 5  # one uncovered trailing prepared frame (like E-session 36238)
    prepared = np.arange(n_prepared, dtype=np.int64)
    raw = prepared + 1000
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=prepared,
        raw_decode_frame_idx=raw,
        behavior_sync=behavior_sync,
        pose_sync=pose_sync,
        behavior_states=_states(behavior_sync.n_dense),
        final_pose=_pose(pose_sync.n_dense),
    )
    assert built.n_rows == 5
    assert list(built.table.columns) == list(LEGACY_FRAMEWISE_COLUMNS)
    assert list(built.alignment.columns) == list(FRAME_ALIGNMENT_COLUMNS)
    assert int(built.alignment["prepared_frame_idx"].nunique()) == 5
    # LAST_IN_PLATEAU for prepared 1 → dense 3 (plateau 2,3)
    assert int(built.table.loc[1, "dense_frame_idx"]) == 3
    timing = resolve_prepared_frame_timing(
        behavior_sync, 1, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    assert int(built.table.loc[1, "dense_frame_idx"]) == timing.dense_sample_idx
    assert built.table.loc[1, "behaviour_label_focal_0"] == "b0_3"
    # Independent pose map: prepared 1 → pose dense 3
    pose_timing = resolve_prepared_frame_timing(
        pose_sync, 1, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    assert int(built.alignment.loc[1, "pose_dense_frame_idx"]) == pose_timing.dense_sample_idx
    # Trailing prepared frame uncovered by both maps.
    assert pd.isna(built.table.loc[4, "dense_frame_idx"])
    assert pd.isna(built.table.loc[4, "behaviour_label_focal_0"])
    assert bool(built.alignment.loc[4, "behavior_coverage"]) is False
    assert bool(built.alignment.loc[4, "pose_coverage"]) is False
    assert int(built.table.loc[4, "frame"]) == 1004
    # Pose present only where pose sync covers.
    assert pd.notna(built.table.loc[0, "neck_0_x"])
    assert pd.isna(built.table.loc[4, "neck_0_x"])
    # Prepared 3 covered by behavior but not pose.
    assert bool(built.alignment.loc[3, "behavior_coverage"]) is True
    assert bool(built.alignment.loc[3, "pose_coverage"]) is False
    assert pd.isna(built.table.loc[3, "neck_0_x"])
    assert built.table.loc[3, "behaviour_label_focal_0"] == "b0_7"

    out = write_legacy_framewise_csv(tmp_path / "legacy_framewise_v2.csv", built.table)
    qc = validate_legacy_framewise_csv_bytes(out.read_bytes())
    assert qc["n_header_columns"] == 23
    assert qc["n_data_rows"] == 5
    assert qc["header"] == list(LEGACY_FRAMEWISE_COLUMNS)


def test_vectorized_last_in_plateau_matches_overlay_resolver():
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
        final_pose=_pose(4),
    )
    assert built_v1.n_rows == 4
    assert list(built_v1.table["dense_frame_idx"].astype(int)) == [0, 1, 2, 3]
    # v2 collapses to prepared domain (also 4 here, one-to-one).
    sync = SyncMap.from_sync_index(sync_df)
    built_v2 = build_legacy_framewise_v2_table(
        prepared_frame_idx=np.arange(4, dtype=np.int64),
        raw_decode_frame_idx=np.arange(4, dtype=np.int64) + 50,
        behavior_sync=sync,
        pose_sync=sync,
        behavior_states=_states(4),
        final_pose=_pose(4),
    )
    assert built_v2.n_rows == 4
    assert int(built_v2.table.loc[0, "frame"]) == 50


def test_missing_behavior_keeps_independent_pose():
    # Behavior covers prepared 0..1 only; pose covers 0..3.
    behavior_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    pose_df = build_hand_computed_sync(n_prepared=4, plateau_repeats=1)
    behavior_sync = SyncMap.from_sync_index(behavior_df)
    pose_sync = SyncMap.from_sync_index(pose_df)
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=np.arange(4, dtype=np.int64),
        raw_decode_frame_idx=np.arange(4, dtype=np.int64) + 10,
        behavior_sync=behavior_sync,
        pose_sync=pose_sync,
        behavior_states=_states(behavior_sync.n_dense),
        final_pose=_pose(pose_sync.n_dense),
    )
    assert pd.isna(built.table.loc[3, "dense_frame_idx"])
    assert pd.isna(built.table.loc[3, "behaviour_label_focal_0"])
    assert pd.notna(built.table.loc[3, "neck_0_x"])
    assert bool(built.alignment.loc[3, "behavior_coverage"]) is False
    assert bool(built.alignment.loc[3, "pose_coverage"]) is True


def test_nonidentity_raw_decode_mapping_preserved():
    prepared = np.arange(3, dtype=np.int64)
    raw = np.array([72619, 72620, 72621], dtype=np.int64)
    sync_df = build_hand_computed_sync(n_prepared=2, plateau_repeats=1)
    sync = SyncMap.from_sync_index(sync_df)
    built = build_legacy_framewise_v2_table(
        prepared_frame_idx=prepared,
        raw_decode_frame_idx=raw,
        behavior_sync=sync,
        pose_sync=sync,
        behavior_states=_states(2),
        final_pose=_pose(2),
    )
    assert int(built.table.loc[2, "frame"]) == 72621
    assert pd.isna(built.table.loc[2, "dense_frame_idx"])
    assert int(built.alignment.loc[2, "raw_decode_frame_idx"]) == 72621
