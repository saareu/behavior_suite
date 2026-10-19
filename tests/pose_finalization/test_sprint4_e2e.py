"""Sprint 4 end-to-end validation and Sprint 3 coverage gaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pose_finalization import (
    ELIGIBLE_DOWNSTREAM,
    FINAL_POSE_FILENAME,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_TEMPORAL_RESAMPLE,
    PoseFinalizationError,
    PoseFinalizationRequest,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    SmoothConfig,
    apply_savgol_smoothing,
    is_s4_run_eligible_for_downstream,
    run_pose_finalization,
    summarize_pose_finalization_project,
    validate_s4_run_for_downstream,
)
from pose_finalization.contracts import RUN_META_FILENAME, SETTINGS_USED_FILENAME
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_NOT_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    FINAL_TRACKED_POSE_FILENAME,
    RUN_META_FILENAME as S3_RUN_META_FILENAME,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {
        path: (_sha256(path), path.stat().st_size)
        for path in paths
        if path.exists()
    }


def _write_sync(path: Path, *, frame_count: int, fps: float = 120.0) -> Path:
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
    nodes: tuple[str, ...] = ("nose", "neck"),
    tracks: tuple[int, ...] = (0, 1),
    gap_frames: range | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame_idx in range(n_frames):
        for track in tracks:
            for node in nodes:
                missing = gap_frames is not None and frame_idx in gap_frames
                x = np.nan if missing else 10.0 * track + frame_idx
                y = np.nan if missing else 20.0 * track + 2.0 * frame_idx
                rows.append(
                    {
                        "frame": frame_idx,
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                        "point_score": np.nan if missing else 0.9,
                        "instance_score": np.nan if missing else 1.5,
                        "was_track_swapped": np.uint8(0),
                        "was_node_blanked": np.uint8(0),
                        "time_sec": frame_idx / 120.0,
                        "timestamp_source": "prepared_sync:prepared_time_sec",
                        "video_index": 0,
                        "video_fps": 120.0,
                    }
                )
    return pd.DataFrame(rows)


def _write_accepted_session(
    session_root: Path,
    *,
    n_frames: int = 40,
    run_id: str = "current_lab_two_mouse_headstage_v1__20260101T000000",
    acceptance_state: str = ACCEPTANCE_ACCEPTED,
    gap_frames: range | None = None,
    corrupt_sha: bool = False,
) -> Path:
    sync = _write_sync(session_root / "preprocess" / "prepared_sync.npz", frame_count=n_frames)
    run_dir = session_root / "tracking_correction" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    table = _pose_rows(n_frames=n_frames, gap_frames=gap_frames)
    pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    table.to_parquet(pose_path, index=False)
    digest = _sha256(pose_path)
    if corrupt_sha:
        digest = "0" * 64
    meta = {
        "run_id": run_id,
        "status": "correction_complete",
        "backend_status": "completed",
        "acceptance_state": acceptance_state,
        "completed_at": "2026-01-01T00:00:00",
        "profile_id": "current_lab_two_mouse_headstage_v1",
        "acceptance": {
            "accepted_at": "2026-01-01T00:05:00",
            "tracked_pose_sha256": digest,
        },
        "input": {
            "session_root": str(session_root),
            "prepared_sync": str(sync),
        },
    }
    (run_dir / S3_RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def test_end_to_end_default_settings_and_settings_used(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=40, gap_frames=range(12, 18))
    upstream = [
        s3_run / FINAL_TRACKED_POSE_FILENAME,
        s3_run / S3_RUN_META_FILENAME,
        session / "preprocess" / "prepared_sync.npz",
    ]
    before = _snapshot(upstream)

    result = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20261019T100000")
    )
    assert result.success is True
    assert result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    assert result.final_pose_path is not None
    assert result.final_pose_path.is_file()

    settings = yaml.safe_load(result.settings_used_path.read_text(encoding="utf-8"))
    assert settings["gap_fill"]["linear_max_frames"] == 20
    assert settings["gap_fill"]["kalman_max_frames"] == 140
    assert settings["smoothing"]["enabled"] is True
    assert settings["smoothing"]["window_length"] == 11
    assert settings["smoothing"]["polyorder"] == 2
    assert _snapshot(upstream) == before


def test_end_to_end_modified_advanced_settings(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=50, gap_frames=range(10, 14))
    from pose_finalization import GapFillConfig

    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_run,
            timestamp="20261019T101000",
            gap_fill_config=GapFillConfig(linear_max_frames=5, kalman_max_frames=30),
            smooth_config=SmoothConfig(enabled=True, window_length=7, polyorder=2),
        )
    )
    settings = yaml.safe_load(result.settings_used_path.read_text(encoding="utf-8"))
    assert settings["gap_fill"]["linear_max_frames"] == 5
    assert settings["gap_fill"]["kalman_max_frames"] == 30
    assert settings["smoothing"]["window_length"] == 7


def test_reject_unaccepted_and_superseded_s3(tmp_path: Path) -> None:
    session = tmp_path / "session"
    unaccepted = _write_accepted_session(
        session,
        run_id="unaccepted__20260101T000000",
        acceptance_state=ACCEPTANCE_NOT_ACCEPTED,
    )
    with pytest.raises(PoseFinalizationError, match="not currently accepted"):
        run_pose_finalization(
            PoseFinalizationRequest(s3_run_dir=unaccepted, timestamp="20261019T102000")
        )

    superseded = _write_accepted_session(
        session,
        run_id="superseded__20260101T010000",
        acceptance_state=ACCEPTANCE_SUPERSEDED,
    )
    with pytest.raises(PoseFinalizationError, match="not currently accepted|superseded"):
        run_pose_finalization(
            PoseFinalizationRequest(s3_run_dir=superseded, timestamp="20261019T102100")
        )


def test_multiple_completed_runs_and_stale_classification(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=30)
    first = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20261019T103000")
    )
    second = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_run,
            timestamp="20261019T103100",
            smooth_config=SmoothConfig(enabled=False),
        )
    )
    assert first.run_dir != second.run_dir
    assert first.final_pose_path is not None and first.final_pose_path.is_file()
    assert second.final_pose_path is not None and second.final_pose_path.is_file()

    summary = summarize_pose_finalization_project(session)
    assert len(summary.completed_runs) == 2
    assert all(run.classification == ELIGIBLE_DOWNSTREAM for run in summary.completed_runs)

    meta_path = s3_run / S3_RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Prior S4 artifacts remain intact; they become stale for downstream use.
    assert first.final_pose_path.is_file()
    assert second.final_pose_path.is_file()
    stale = summarize_pose_finalization_project(session)
    assert all(run.classification == STALE_SOURCE for run in stale.completed_runs)
    assert is_s4_run_eligible_for_downstream(first.run_dir) is False


def test_cross_boundary_smoothing_independence() -> None:
    """Gap-separated segments must smooth independently (Sprint 3 coverage gap)."""

    n = 60
    x = np.zeros((n, 1, 1), dtype=np.float64)
    y = np.zeros((n, 1, 1), dtype=np.float64)
    protected = np.zeros((n, 1, 1), dtype=bool)
    t = np.arange(n, dtype=np.float64)
    x[:, 0, 0] = t
    y[:, 0, 0] = 2.0 * t
    # Unresolved gap separates two long segments.
    x[25:30, 0, 0] = np.nan
    y[25:30, 0, 0] = np.nan

    left_only = apply_savgol_smoothing(
        x=x.copy(), y=y.copy(), protected=protected.copy()
    )
    # Mutate the right segment after copying left values; left output must match.
    x_mut = x.copy()
    y_mut = y.copy()
    x_mut[30:, 0, 0] += 100.0
    y_mut[30:, 0, 0] += 50.0
    mutated = apply_savgol_smoothing(x=x_mut, y=y_mut, protected=protected.copy())

    assert np.allclose(left_only.x[:25, 0, 0], mutated.x[:25, 0, 0], equal_nan=True)
    assert np.allclose(left_only.y[:25, 0, 0], mutated.y[:25, 0, 0], equal_nan=True)
    assert not np.allclose(left_only.x[30:, 0, 0], mutated.x[30:, 0, 0])


def test_downstream_sha_corruption_detection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=25)
    result = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20261019T104000")
    )
    assert is_s4_run_eligible_for_downstream(result.run_dir) is True

    tracked = s3_run / FINAL_TRACKED_POSE_FILENAME
    table = pd.read_parquet(tracked)
    table.loc[0, "x"] = float(table.loc[0, "x"]) + 1.0
    table.to_parquet(tracked, index=False)

    assert is_s4_run_eligible_for_downstream(result.run_dir) is False
    with pytest.raises(PoseFinalizationError, match="SHA256|digest|accepted"):
        validate_s4_run_for_downstream(result.run_dir)
    # Completed S4 artifact itself is not rewritten by the staleness check.
    assert result.final_pose_path is not None
    assert result.final_pose_path.is_file()


def test_nan_confidence_on_reconstructed_rows(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=40, gap_frames=range(15, 20))
    result = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20261019T105000")
    )
    final = pd.read_parquet(result.final_pose_path)
    reconstructed = final[
        final["provenance"].isin([PROVENANCE_LINEAR_GAP_FILL, PROVENANCE_TEMPORAL_RESAMPLE])
    ]
    assert len(reconstructed) > 0
    assert reconstructed["point_score"].isna().all()
    assert reconstructed["instance_score"].isna().all()
    original = final[final["provenance_name"] == "original"]
    assert len(original) > 0
    assert original["point_score"].notna().all()


def test_exported_timing_source_mapping_consistency(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=30)
    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_run,
            timestamp="20261019T106000",
            smooth_config=SmoothConfig(enabled=False),
        )
    )
    final = pd.read_parquet(result.final_pose_path)
    from pose_finalization.timing import build_timing_map_from_prepared_sync

    timing = build_timing_map_from_prepared_sync(
        session / "preprocess" / "prepared_sync.npz"
    )
    # One representative track/node must mirror TimingMap arrays exactly.
    subset = final[(final["track"] == 0) & (final["node"] == "nose")].sort_values(
        "sample_idx"
    )
    assert len(subset) == timing.target_sample_count
    assert np.allclose(subset["time_sec"].to_numpy(), timing.target_time_sec)
    assert np.array_equal(
        subset["left_source_frame_idx"].to_numpy(), timing.left_source_frame_idx
    )
    assert np.array_equal(
        subset["right_source_frame_idx"].to_numpy(), timing.right_source_frame_idx
    )
    assert np.allclose(
        subset["interpolation_weight"].to_numpy(), timing.interpolation_weight
    )
    assert np.array_equal(
        subset["is_exact_source_sample"].to_numpy(), timing.is_exact_source_sample
    )


def test_output_overlap_protection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=20)
    with pytest.raises(PoseFinalizationError, match="must not overlap"):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3_run,
                timestamp="20261019T107000",
                output_root=session / "tracking_correction",
            )
        )
    assert not (session / "tracking_correction" / "pose_finalization__20261019T107000").exists()


def test_s3_superseded_during_processing_gui_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=25)
    import pose_finalization.runner as runner_mod
    from pose_finalization.accepted_s3 import validate_accepted_s3_run as real_validate

    calls = {"n": 0}

    def flipping_validate(path: Path | str):
        calls["n"] += 1
        accepted = real_validate(path)
        if calls["n"] >= 2:
            meta_path = Path(path) / S3_RUN_META_FILENAME
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            return real_validate(path)
        return accepted

    monkeypatch.setattr(runner_mod, "validate_accepted_s3_run", flipping_validate)
    with pytest.raises(PoseFinalizationError, match="not currently accepted|superseded"):
        run_pose_finalization(
            PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20261019T108000")
        )
    run_dir = session / "pose_finalization" / "pose_finalization__20261019T108000"
    meta = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert not (run_dir / FINAL_POSE_FILENAME).exists()
    assert (run_dir / SETTINGS_USED_FILENAME).is_file()
