"""Sprint 2: dense 120 Hz construction, manual-blank protection, gap fill."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pose_finalization import (
    GAP_KALMAN_MAX_FRAMES_DEFAULT,
    GAP_LINEAR_MAX_FRAMES_DEFAULT,
    HEADSTAGE_NODE_NAME,
    PROVENANCE_IDENTITY_CORRECTED,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PROVENANCE_TEMPORAL_RESAMPLE,
    GapFillConfig,
    build_timing_map_from_prepared_sync,
    construct_dense_pose,
    load_manual_blank_intervals,
)
from pose_finalization.dense_pose import build_source_protected_mask
from pose_finalization.timing import map_target_to_source, build_uniform_target_timeline
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    MANUAL_ACTION_BLANK_NODE,
    MANUAL_ACTION_SWAP_IDENTITIES,
    MANUAL_ACTION_SWAP_NODE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sync(path: Path, *, frame_count: int, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(path, sync)
    return path


def _pose_rows(
    *,
    n_frames: int,
    nodes: tuple[str, ...] = ("nose", "neck", "headstage"),
    tracks: tuple[int, ...] = (0, 1),
    missing: set[tuple[int, int, str]] | None = None,
    blanked_nan: set[tuple[int, int, str]] | None = None,
    identity_frames: set[int] | None = None,
) -> pd.DataFrame:
    """Build a long-format table matching the accepted S3 schema subset."""

    missing = missing or set()
    blanked_nan = blanked_nan or set()
    identity_frames = identity_frames or set()
    rows: list[dict[str, object]] = []
    for frame_idx in range(n_frames):
        for track in tracks:
            for node in nodes:
                key = (frame_idx, track, node)
                if key in missing:
                    continue
                x = 10.0 * track + frame_idx + (0.5 if node == "nose" else 1.5)
                y = 20.0 * track + 2.0 * frame_idx + (0.25 if node == "neck" else 0.0)
                if key in blanked_nan:
                    x = float("nan")
                    y = float("nan")
                    point_score = 0.0
                    was_blanked = 1
                else:
                    point_score = 0.9
                    was_blanked = 0
                rows.append(
                    {
                        "frame": frame_idx,
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                        "point_score": point_score,
                        "instance_score": 1.5,
                        "x_raw": x,
                        "y_raw": y,
                        "point_score_raw": point_score,
                        "instance_score_raw": 1.5,
                        "was_track_swapped": np.uint8(1 if frame_idx in identity_frames else 0),
                        "was_node_blanked": np.uint8(was_blanked),
                        "time_sec": frame_idx / 120.0,
                        "timestamp_source": "prepared_sync:prepared_time_sec",
                        "video_index": 0,
                        "video_fps": 120.0,
                    }
                )
    return pd.DataFrame(rows)


def _timing_for_frames(n_frames: int, fps: float = 120.0):
    frames = np.arange(n_frames, dtype=np.int64)
    times = frames.astype(np.float64) / fps
    targets = build_uniform_target_timeline(times, target_fps=120.0)
    return map_target_to_source(
        source_frame_idx=frames,
        source_time_sec=times,
        target_time_sec=targets,
        target_fps=120.0,
        source_fps_header=fps,
    )


def test_dense_shape_track_node_preservation_and_source_mapping() -> None:
    n_frames = 5
    table = _pose_rows(n_frames=n_frames)
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(table, timing, apply_gap_fill=False)

    assert result.n_samples == timing.target_sample_count == n_frames
    assert list(result.track_ids) == [0, 1]
    assert list(result.node_names) == ["headstage", "neck", "nose"]
    assert result.x.shape == (n_frames, 2, 3)
    assert np.array_equal(result.left_source_frame_idx, timing.left_source_frame_idx)
    assert np.array_equal(result.right_source_frame_idx, timing.right_source_frame_idx)
    assert np.array_equal(result.interpolation_weight, timing.interpolation_weight)
    assert np.all(result.is_exact_source_sample)
    assert np.all(
        (result.provenance == PROVENANCE_ORIGINAL)
        | (result.provenance == PROVENANCE_MISSING)
    )


def test_source_fps_equal_to_120() -> None:
    table = _pose_rows(n_frames=4)
    timing = _timing_for_frames(4, fps=120.0)
    result = construct_dense_pose(table, timing, apply_gap_fill=False)
    assert result.n_samples == 4
    assert np.all(result.is_exact_source_sample)
    assert np.all(result.provenance == PROVENANCE_ORIGINAL)


def test_source_fps_different_from_120_exact_vs_interpolated(tmp_path: Path) -> None:
    n_frames = 5
    fps = 20.0
    sync = _write_sync(tmp_path / "prepared_sync.npz", frame_count=n_frames, fps=fps)
    timing = build_timing_map_from_prepared_sync(sync)
    table = _pose_rows(n_frames=n_frames)
    # Override motion to make interpolation checkable.
    for i, row in table.iterrows():
        table.at[i, "x"] = float(row["prepared_frame_idx"]) * 10.0
        table.at[i, "y"] = float(row["prepared_frame_idx"]) * 5.0

    result = construct_dense_pose(table, timing, apply_gap_fill=False)
    assert result.n_samples == timing.target_sample_count
    exact = result.is_exact_source_sample
    assert bool(np.any(exact))
    assert bool(np.any(~exact))

    nose_i = list(result.node_names).index("nose")
    # Exact samples carry original provenance and source confidence.
    exact_idx = int(np.flatnonzero(exact)[0])
    assert result.provenance[exact_idx, 0, nose_i] == PROVENANCE_ORIGINAL
    assert np.isfinite(result.point_score[exact_idx, 0, nose_i])

    # Derived samples are temporal resamples without fabricated confidence.
    derived_idx = int(np.flatnonzero(~exact)[0])
    assert result.provenance[derived_idx, 0, nose_i] == PROVENANCE_TEMPORAL_RESAMPLE
    assert np.isnan(result.point_score[derived_idx, 0, nose_i])
    assert np.isnan(result.instance_score[derived_idx, 0, nose_i])

    left = int(result.left_source_frame_idx[derived_idx])
    right = int(result.right_source_frame_idx[derived_idx])
    w = float(result.interpolation_weight[derived_idx])
    expected_x = (1.0 - w) * (left * 10.0) + w * (right * 10.0)
    assert result.x[derived_idx, 0, nose_i] == pytest.approx(expected_x)


def test_missing_source_observations_not_used_as_endpoints() -> None:
    n_frames = 5
    # Remove middle frame observations for nose/track 0 entirely (sparse missing).
    table = _pose_rows(
        n_frames=n_frames,
        missing={(2, 0, "nose")},
    )
    timing = _timing_for_frames(n_frames, fps=120.0)
    # Force a non-exact interior target between frames 1 and 2 by using 40 Hz source
    # mapped at 120 Hz would create derived samples; here keep 120 and verify
    # exact sample at frame 2 is missing.
    result = construct_dense_pose(table, timing, apply_gap_fill=False)
    nose_i = list(result.node_names).index("nose")
    assert result.provenance[2, 0, nose_i] == PROVENANCE_MISSING
    assert np.isnan(result.x[2, 0, nose_i])

    # Build a derived sample whose right endpoint is missing: source 20 Hz.
    frames = np.arange(n_frames, dtype=np.int64)
    times = frames.astype(np.float64) / 20.0
    targets = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2], dtype=np.float64)
    timing2 = map_target_to_source(
        source_frame_idx=frames,
        source_time_sec=times,
        target_time_sec=targets,
        target_fps=120.0,
        source_fps_header=20.0,
    )
    result2 = construct_dense_pose(table, timing2, apply_gap_fill=False)
    # Target at 0.075 s lies between source frames 1 (0.05) and 2 (0.10); frame 2 missing.
    idx = int(np.argmin(np.abs(timing2.target_time_sec - 0.075)))
    assert int(result2.left_source_frame_idx[idx]) == 1
    assert int(result2.right_source_frame_idx[idx]) == 2
    assert not bool(result2.is_exact_source_sample[idx])
    assert result2.provenance[idx, 0, nose_i] == PROVENANCE_MISSING
    assert np.isnan(result2.x[idx, 0, nose_i])


def test_linear_gap_filling() -> None:
    n_frames = 10
    table = _pose_rows(n_frames=n_frames, nodes=("nose",))
    # Clear an internal gap of 3 samples on the dense 120 Hz grid after construction
    # by removing source frames 3–5 for track 0 nose (exact 120 Hz → gap length 3).
    for frame_idx in (3, 4, 5):
        mask = (
            (table["prepared_frame_idx"] == frame_idx)
            & (table["track"] == 0)
            & (table["node"] == "nose")
        )
        table.loc[mask, ["x", "y"]] = np.nan

    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(
        table,
        timing,
        gap_fill_config=GapFillConfig(linear_max_frames=20, kalman_max_frames=140),
    )
    nose_i = 0
    assert result.provenance[3, 0, nose_i] == PROVENANCE_LINEAR_GAP_FILL
    assert result.provenance[4, 0, nose_i] == PROVENANCE_LINEAR_GAP_FILL
    assert result.provenance[5, 0, nose_i] == PROVENANCE_LINEAR_GAP_FILL
    # Linear between frames 2 and 6.
    assert result.x[4, 0, nose_i] == pytest.approx(
        0.5 * result.x[2, 0, nose_i] + 0.5 * result.x[6, 0, nose_i]
    )
    # Existing endpoints preserved.
    assert result.provenance[2, 0, nose_i] == PROVENANCE_ORIGINAL
    assert result.provenance[6, 0, nose_i] == PROVENANCE_ORIGINAL


def test_kalman_rts_gap_filling() -> None:
    n_frames = 80
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    gap_frames = set(range(10, 45))  # 35 frames → Kalman band
    for frame_idx in gap_frames:
        mask = table["prepared_frame_idx"] == frame_idx
        table.loc[mask, ["x", "y"]] = np.nan

    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(
        table,
        timing,
        gap_fill_config=GapFillConfig(linear_max_frames=20, kalman_max_frames=140),
    )
    assert result.provenance[20, 0, 0] == PROVENANCE_KALMAN_GAP_FILL
    assert np.isfinite(result.x[20, 0, 0])
    assert np.isfinite(result.y[20, 0, 0])
    # Endpoints unchanged.
    assert result.provenance[9, 0, 0] == PROVENANCE_ORIGINAL
    assert result.provenance[45, 0, 0] == PROVENANCE_ORIGINAL


def test_long_gaps_remain_unresolved() -> None:
    n_frames = 200
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    for frame_idx in range(10, 160):  # 150 > 140
        mask = table["prepared_frame_idx"] == frame_idx
        table.loc[mask, ["x", "y"]] = np.nan
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(table, timing)
    assert result.provenance[50, 0, 0] == PROVENANCE_MISSING
    assert np.isnan(result.x[50, 0, 0])


def test_leading_and_trailing_gaps_remain_nan() -> None:
    n_frames = 12
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    for frame_idx in (0, 1, 2, 9, 10, 11):
        mask = table["prepared_frame_idx"] == frame_idx
        table.loc[mask, ["x", "y"]] = np.nan
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(table, timing)
    assert np.all(np.isnan(result.x[0:3, 0, 0]))
    assert np.all(result.provenance[0:3, 0, 0] == PROVENANCE_MISSING)
    assert np.all(np.isnan(result.x[9:12, 0, 0]))
    assert np.all(result.provenance[9:12, 0, 0] == PROVENANCE_MISSING)


def test_protected_manual_blanks_boundaries_and_resampling(tmp_path: Path) -> None:
    n_frames = 8
    table = _pose_rows(n_frames=n_frames, nodes=("nose", "neck"))
    # Explicit manual blank at prepared frame 3, track 0, nose.
    corrections = tmp_path / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_BLANK_NODE,
                    "start_frame": 3,
                    "end_frame": 3,
                    "tracks": [0],
                    "track": 0,
                    "node": "nose",
                    "before": {"x": 1.0, "y": 2.0},
                    "after": {"x": None, "y": None},
                },
                {
                    "action": "swap_identities",
                    "start_frame": 0,
                    "end_frame": 2,
                    "tracks": [0, 1],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Also set was_node_blanked on an automatic-style blank that must NOT be protected.
    auto_mask = (
        (table["prepared_frame_idx"] == 5)
        & (table["track"] == 0)
        & (table["node"] == "neck")
    )
    table.loc[auto_mask, ["x", "y"]] = np.nan
    table.loc[auto_mask, "was_node_blanked"] = np.uint8(1)

    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(
        table,
        timing,
        manual_corrections_path=corrections,
    )
    nose_i = list(result.node_names).index("nose")
    neck_i = list(result.node_names).index("neck")

    assert bool(result.protected[3, 0, nose_i])
    assert result.provenance[3, 0, nose_i] == PROVENANCE_PROTECTED_MANUAL_BLANK
    assert np.isnan(result.x[3, 0, nose_i])

    # Gap fill must not reconstruct the protected sample.
    assert result.provenance[3, 0, nose_i] == PROVENANCE_PROTECTED_MANUAL_BLANK

    # Automatic was_node_blanked without manual record is ordinary missing → fillable.
    assert not bool(result.protected[5, 0, neck_i])
    assert result.provenance[5, 0, neck_i] == PROVENANCE_LINEAR_GAP_FILL
    assert np.isfinite(result.x[5, 0, neck_i])

    blanks = load_manual_blank_intervals(corrections)
    assert len(blanks) == 1
    assert blanks[0].node == "nose"

    # Resampling at non-120 source FPS: targets touching frame 3 stay protected.
    sync = _write_sync(tmp_path / "prepared_sync.npz", frame_count=n_frames, fps=40.0)
    timing40 = build_timing_map_from_prepared_sync(sync)
    result40 = construct_dense_pose(
        table,
        timing40,
        manual_corrections_path=corrections,
        apply_gap_fill=False,
    )
    touch = (
        (result40.left_source_frame_idx == 3) | (result40.right_source_frame_idx == 3)
    )
    assert np.all(result40.protected[touch, 0, nose_i])
    assert np.all(
        result40.provenance[touch, 0, nose_i] == PROVENANCE_PROTECTED_MANUAL_BLANK
    )


def test_headstage_excluded_from_imputation() -> None:
    n_frames = 12
    table = _pose_rows(n_frames=n_frames, nodes=("nose", HEADSTAGE_NODE_NAME), tracks=(0,))
    for frame_idx in (4, 5, 6):
        mask = table["prepared_frame_idx"] == frame_idx
        table.loc[mask, ["x", "y"]] = np.nan
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(table, timing)
    hs_i = list(result.node_names).index(HEADSTAGE_NODE_NAME)
    nose_i = list(result.node_names).index("nose")
    assert result.provenance[5, 0, hs_i] == PROVENANCE_MISSING
    assert np.isnan(result.x[5, 0, hs_i])
    assert result.provenance[5, 0, nose_i] == PROVENANCE_LINEAR_GAP_FILL
    assert np.isfinite(result.x[5, 0, nose_i])


def test_identity_preservation_and_identity_corrected_provenance() -> None:
    table = _pose_rows(n_frames=4, identity_frames={1})
    timing = _timing_for_frames(4, fps=120.0)
    result = construct_dense_pose(table, timing, apply_gap_fill=False)
    assert list(result.track_ids) == [0, 1]
    nose_i = list(result.node_names).index("nose")
    assert result.provenance[0, 0, nose_i] == PROVENANCE_ORIGINAL
    assert result.provenance[1, 0, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    # Coordinates for track 0/1 remain distinct.
    assert result.x[0, 0, nose_i] != result.x[0, 1, nose_i]


def test_original_confidence_preservation() -> None:
    table = _pose_rows(n_frames=3, nodes=("nose",), tracks=(0,))
    table.loc[table["prepared_frame_idx"] == 1, "point_score"] = 0.42
    table.loc[table["prepared_frame_idx"] == 1, "instance_score"] = 2.5
    timing = _timing_for_frames(3, fps=120.0)
    result = construct_dense_pose(table, timing, apply_gap_fill=False)
    assert result.point_score[1, 0, 0] == pytest.approx(0.42)
    assert result.instance_score[1, 0, 0] == pytest.approx(2.5)


def test_deterministic_outputs() -> None:
    table = _pose_rows(n_frames=30, nodes=("nose",), tracks=(0,))
    for frame_idx in range(5, 12):
        table.loc[table["prepared_frame_idx"] == frame_idx, ["x", "y"]] = np.nan
    for frame_idx in range(15, 28):
        table.loc[table["prepared_frame_idx"] == frame_idx, ["x", "y"]] = np.nan
    timing = _timing_for_frames(30, fps=120.0)
    a = construct_dense_pose(table, timing)
    b = construct_dense_pose(table, timing)
    assert np.array_equal(a.provenance, b.provenance)
    assert np.allclose(a.x, b.x, equal_nan=True)
    assert np.allclose(a.y, b.y, equal_nan=True)


def test_no_mutation_of_upstream_artifacts(tmp_path: Path) -> None:
    n_frames = 6
    sync = _write_sync(tmp_path / "preprocess" / "prepared_sync.npz", frame_count=n_frames, fps=120.0)
    pose_path = tmp_path / "tracking_correction" / "run" / "tracked_pose.parquet"
    pose_path.parent.mkdir(parents=True)
    table = _pose_rows(n_frames=n_frames)
    table.to_parquet(pose_path, index=False)
    corrections = pose_path.parent / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_BLANK_NODE,
                    "start_frame": 2,
                    "end_frame": 2,
                    "track": 0,
                    "tracks": [0],
                    "node": "nose",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = {
        pose_path: _sha256(pose_path),
        corrections: _sha256(corrections),
        sync: _sha256(sync),
    }
    timing = build_timing_map_from_prepared_sync(sync)
    construct_dense_pose(pose_path, timing)
    assert _sha256(pose_path) == before[pose_path]
    assert _sha256(corrections) == before[corrections]
    assert _sha256(sync) == before[sync]


def test_gap_thresholds_are_configurable() -> None:
    assert GAP_LINEAR_MAX_FRAMES_DEFAULT == 20
    assert GAP_KALMAN_MAX_FRAMES_DEFAULT == 140
    cfg = GapFillConfig(linear_max_frames=2, kalman_max_frames=5)
    n_frames = 20
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    for frame_idx in range(3, 7):  # gap length 4 → Kalman under custom cfg
        table.loc[table["prepared_frame_idx"] == frame_idx, ["x", "y"]] = np.nan
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(table, timing, gap_fill_config=cfg)
    assert result.provenance[4, 0, 0] == PROVENANCE_KALMAN_GAP_FILL


def test_integration_real_s3_schema_fixture(tmp_path: Path) -> None:
    """Representative integration-style fixture using the real S3 column set."""

    n_frames = 10
    fps = 119.048
    sync = _write_sync(tmp_path / "prepared_sync.npz", frame_count=n_frames, fps=fps)
    rows: list[dict[str, object]] = []
    nodes = ("nose", "neck", "spine_base", "tail_base", "headstage")
    for frame_idx in range(n_frames):
        for track in (0, 1):
            for node in nodes:
                if node == "headstage" and track == 1:
                    continue
                rows.append(
                    {
                        "frame": frame_idx,
                        "track": track,
                        "node": node,
                        "x": float(100 + track * 50 + frame_idx),
                        "y": float(200 + track * 30 + 2 * frame_idx),
                        "point_score": 0.8,
                        "instance_score": 2.0,
                        "x_raw": float(100 + track * 50 + frame_idx),
                        "y_raw": float(200 + track * 30 + 2 * frame_idx),
                        "point_score_raw": 0.8,
                        "instance_score_raw": 2.0,
                        "was_track_swapped": np.uint8(0),
                        "was_node_blanked": np.uint8(0),
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "original_frame_idx": frame_idx,
                        "source_video_frame_idx": frame_idx,
                        "time_sec": frame_idx / fps,
                        "timestamp_source": "prepared_sync:prepared_time_sec",
                        "video_index": 0,
                        "video_fps": fps,
                    }
                )
    pose_path = tmp_path / "tracked_pose.parquet"
    pd.DataFrame(rows).to_parquet(pose_path, index=False)
    corrections = tmp_path / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_BLANK_NODE,
                    "start_frame": 4,
                    "end_frame": 4,
                    "track": 0,
                    "tracks": [0],
                    "node": "nose",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    timing = build_timing_map_from_prepared_sync(sync)
    result = construct_dense_pose(pose_path, timing)
    assert result.n_tracks == 2
    assert "nose" in list(result.node_names)
    assert "headstage" in list(result.node_names)
    nose_i = list(result.node_names).index("nose")
    # Source FPS ≠ 120 → mix of exact and derived samples.
    assert bool(np.any(result.is_exact_source_sample))
    assert bool(np.any(~result.is_exact_source_sample))
    # Manual blank protected on the dense grid wherever frame 4 contributes.
    touch = (result.left_source_frame_idx == 4) | (result.right_source_frame_idx == 4)
    assert np.all(result.protected[touch, 0, nose_i])
    assert result.tracked_pose_path == pose_path.resolve()


def test_source_protected_mask_builder() -> None:
    from pose_finalization.dense_pose import ManualBlankInterval

    mask = build_source_protected_mask(
        n_source_frames=5,
        track_ids=[0, 1],
        node_names=["nose", "neck"],
        blanks=[
            ManualBlankInterval(start_frame=1, end_frame=2, track=0, node="nose"),
        ],
    )
    assert mask.shape == (5, 2, 2)
    assert bool(mask[1, 0, 0]) and bool(mask[2, 0, 0])
    assert not bool(mask[0, 0, 0])
    assert not bool(mask[1, 1, 0])


def test_manual_interval_identity_swap_provenance(tmp_path: Path) -> None:
    table = _pose_rows(n_frames=6, nodes=("nose", "neck"))
    corrections = tmp_path / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_SWAP_IDENTITIES,
                    "start_frame": 1,
                    "end_frame": 3,
                    "tracks": [0, 1],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timing = _timing_for_frames(6, fps=120.0)
    result = construct_dense_pose(
        table, timing, manual_corrections_path=corrections, apply_gap_fill=False
    )
    nose_i = list(result.node_names).index("nose")
    assert result.provenance[0, 0, nose_i] == PROVENANCE_ORIGINAL
    assert result.provenance[1, 0, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    assert result.provenance[2, 1, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    assert result.provenance[3, 0, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    assert result.provenance[4, 0, nose_i] == PROVENANCE_ORIGINAL
    # Coordinates unchanged by provenance labeling.
    assert np.isfinite(result.x[2, 0, nose_i])


def test_manual_single_node_swap_provenance(tmp_path: Path) -> None:
    table = _pose_rows(n_frames=4, nodes=("nose", "neck"))
    corrections = tmp_path / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_SWAP_NODE,
                    "start_frame": 2,
                    "end_frame": 2,
                    "tracks": [0, 1],
                    "node": "nose",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timing = _timing_for_frames(4, fps=120.0)
    result = construct_dense_pose(
        table, timing, manual_corrections_path=corrections, apply_gap_fill=False
    )
    nose_i = list(result.node_names).index("nose")
    neck_i = list(result.node_names).index("neck")
    assert result.provenance[2, 0, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    assert result.provenance[2, 1, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    assert result.provenance[2, 0, neck_i] == PROVENANCE_ORIGINAL
    assert result.provenance[1, 0, nose_i] == PROVENANCE_ORIGINAL


def test_manual_identity_undo_reset_and_overlapping_parity(tmp_path: Path) -> None:
    """S3 undo/reset drop records from the persisted stack; parity cancels double swaps."""

    table = _pose_rows(n_frames=5, nodes=("nose",))
    timing = _timing_for_frames(5, fps=120.0)
    nose_i = 0

    # Empty stack (reset / never edited) → no manual identity mark.
    empty = tmp_path / "empty_corrections.json"
    empty.write_text("[]\n", encoding="utf-8")
    result_empty = construct_dense_pose(
        table, timing, manual_corrections_path=empty, apply_gap_fill=False
    )
    assert np.all(result_empty.provenance[:, :, nose_i] == PROVENANCE_ORIGINAL)

    # Overlapping identical interval twice → even parity → not manually corrected.
    double = tmp_path / "double_swap.json"
    double.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_SWAP_IDENTITIES,
                    "start_frame": 1,
                    "end_frame": 2,
                    "tracks": [0, 1],
                },
                {
                    "action": MANUAL_ACTION_SWAP_IDENTITIES,
                    "start_frame": 1,
                    "end_frame": 2,
                    "tracks": [0, 1],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result_double = construct_dense_pose(
        table, timing, manual_corrections_path=double, apply_gap_fill=False
    )
    assert result_double.provenance[1, 0, nose_i] == PROVENANCE_ORIGINAL
    assert result_double.provenance[2, 1, nose_i] == PROVENANCE_ORIGINAL

    # Active single swap remains after an unrelated undone edit is absent from file.
    active = tmp_path / "active_after_undo.json"
    active.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_SWAP_IDENTITIES,
                    "start_frame": 3,
                    "end_frame": 3,
                    "tracks": [0, 1],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result_active = construct_dense_pose(
        table, timing, manual_corrections_path=active, apply_gap_fill=False
    )
    assert result_active.provenance[3, 0, nose_i] == PROVENANCE_IDENTITY_CORRECTED
    assert result_active.provenance[2, 0, nose_i] == PROVENANCE_ORIGINAL

    # Automatic was_track_swapped still combines with manual (OR).
    table2 = _pose_rows(n_frames=5, nodes=("nose",), identity_frames={0})
    result_auto = construct_dense_pose(
        table2, timing, manual_corrections_path=empty, apply_gap_fill=False
    )
    assert result_auto.provenance[0, 0, nose_i] == PROVENANCE_IDENTITY_CORRECTED


def test_protected_blank_splits_ordinary_missing_intervals(tmp_path: Path) -> None:
    n_frames = 12
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    # Ordinary missing around a protected blank at frame 5.
    for frame_idx in (3, 4, 6, 7):
        table.loc[table["prepared_frame_idx"] == frame_idx, ["x", "y"]] = np.nan
    corrections = tmp_path / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_BLANK_NODE,
                    "start_frame": 5,
                    "end_frame": 5,
                    "track": 0,
                    "tracks": [0],
                    "node": "nose",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(table, timing, manual_corrections_path=corrections)
    assert result.provenance[5, 0, 0] == PROVENANCE_PROTECTED_MANUAL_BLANK
    assert np.isnan(result.x[5, 0, 0])
    # Side gaps are not bridged across the protected sample.
    assert np.all(np.isnan(result.x[3:5, 0, 0]))
    assert np.all(result.provenance[3:5, 0, 0] == PROVENANCE_MISSING)
    assert np.all(np.isnan(result.x[6:8, 0, 0]))
    assert np.all(result.provenance[6:8, 0, 0] == PROVENANCE_MISSING)
    assert result.provenance[2, 0, 0] == PROVENANCE_ORIGINAL
    assert result.provenance[8, 0, 0] == PROVENANCE_ORIGINAL


def test_protected_blank_interrupts_kalman_length_gap(tmp_path: Path) -> None:
    n_frames = 80
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    for frame_idx in range(10, 50):
        table.loc[table["prepared_frame_idx"] == frame_idx, ["x", "y"]] = np.nan
    corrections = tmp_path / "manual_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {
                    "action": MANUAL_ACTION_BLANK_NODE,
                    "start_frame": 30,
                    "end_frame": 30,
                    "track": 0,
                    "tracks": [0],
                    "node": "nose",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    timing = _timing_for_frames(n_frames, fps=120.0)
    result = construct_dense_pose(
        table,
        timing,
        manual_corrections_path=corrections,
        gap_fill_config=GapFillConfig(linear_max_frames=20, kalman_max_frames=140),
    )
    assert result.provenance[30, 0, 0] == PROVENANCE_PROTECTED_MANUAL_BLANK
    assert np.isnan(result.x[30, 0, 0])
    # Without the blank, length 40 would Kalman-fill; with the blank, sides stay missing.
    assert result.provenance[20, 0, 0] == PROVENANCE_MISSING
    assert np.isnan(result.x[20, 0, 0])
    assert result.provenance[40, 0, 0] == PROVENANCE_MISSING
    assert np.isnan(result.x[40, 0, 0])


def test_resampling_across_missing_then_gap_fill_provenance() -> None:
    """Missing source frame is not temporally resampled; gap fill labels the fill."""

    n_frames = 5
    table = _pose_rows(n_frames=n_frames, nodes=("nose",), tracks=(0,))
    table.loc[table["prepared_frame_idx"] == 2, ["x", "y"]] = np.nan
    timing = _timing_for_frames(n_frames, fps=120.0)

    pre = construct_dense_pose(table, timing, apply_gap_fill=False)
    assert pre.provenance[2, 0, 0] == PROVENANCE_MISSING
    assert np.isnan(pre.x[2, 0, 0])
    assert pre.provenance[1, 0, 0] == PROVENANCE_ORIGINAL
    assert pre.provenance[3, 0, 0] == PROVENANCE_ORIGINAL

    filled = construct_dense_pose(table, timing, apply_gap_fill=True)
    assert filled.provenance[2, 0, 0] == PROVENANCE_LINEAR_GAP_FILL
    assert np.isfinite(filled.x[2, 0, 0])
    assert filled.provenance[2, 0, 0] != PROVENANCE_TEMPORAL_RESAMPLE
    assert filled.x[2, 0, 0] == pytest.approx(
        0.5 * filled.x[1, 0, 0] + 0.5 * filled.x[3, 0, 0]
    )
