"""Sprint 3: final Parquet export, QC semantics, and S4 runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pose_finalization import (
    FINAL_POSE_COLUMNS,
    FINAL_POSE_FILENAME,
    FINAL_POSE_SCHEMA_VERSION,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PoseFinalizationError,
    PoseFinalizationRequest,
    QcConfig,
    SmoothConfig,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    apply_savgol_smoothing,
    build_final_pose_table,
    construct_dense_pose,
    evaluate_finalization_qc,
    is_s4_run_eligible_for_downstream,
    run_pose_finalization,
    validate_s4_run_for_downstream,
)
from pose_finalization.contracts import RUN_META_FILENAME
from pose_finalization.timing import build_uniform_target_timeline, map_target_to_source
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    FINAL_TRACKED_POSE_FILENAME,
    MANUAL_ACTION_BLANK_NODE,
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
    noise: float = 0.0,
    jump_at: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for frame_idx in range(n_frames):
        for track in tracks:
            for node in nodes:
                x = 10.0 * track + frame_idx + (0.5 if node == "nose" else 1.5)
                y = 20.0 * track + 2.0 * frame_idx
                if noise:
                    x += float(rng.normal(0.0, noise))
                    y += float(rng.normal(0.0, noise))
                if jump_at is not None and frame_idx >= jump_at:
                    x += 200.0
                    y += 200.0
                rows.append(
                    {
                        "frame": frame_idx,
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                        "point_score": 0.9,
                        "instance_score": 1.5,
                        "was_track_swapped": np.uint8(0),
                        "was_node_blanked": np.uint8(0),
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


def _write_accepted_session(
    session_root: Path,
    *,
    n_frames: int = 40,
    run_id: str = "current_lab_two_mouse_headstage_v1__20260101T000000",
    noise: float = 0.3,
    blank: tuple[int, int, str] | None = None,
    jump_at: int | None = None,
) -> Path:
    sync = _write_sync(session_root / "preprocess" / "prepared_sync.npz", frame_count=n_frames)
    run_dir = session_root / "tracking_correction" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    table = _pose_rows(n_frames=n_frames, noise=noise, jump_at=jump_at)
    if blank is not None:
        frame_i, track, node = blank
        mask = (
            (table["prepared_frame_idx"] == frame_i)
            & (table["track"] == track)
            & (table["node"] == node)
        )
        table.loc[mask, "x"] = np.nan
        table.loc[mask, "y"] = np.nan
        (run_dir / "manual_corrections.json").write_text(
            json.dumps(
                [
                    {
                        "action": MANUAL_ACTION_BLANK_NODE,
                        "start_frame": frame_i,
                        "end_frame": frame_i,
                        "track": track,
                        "node": node,
                    }
                ]
            ),
            encoding="utf-8",
        )
    pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    table.to_parquet(pose_path, index=False)
    digest = _sha256(pose_path)
    meta = {
        "run_id": run_id,
        "status": "correction_complete",
        "backend_status": "completed",
        "acceptance_state": ACCEPTANCE_ACCEPTED,
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


def test_final_parquet_schema_row_count_and_nan_preservation(tmp_path: Path) -> None:
    n_frames = 30
    table = _pose_rows(n_frames=n_frames)
    # Force a missing observation.
    table.loc[
        (table["prepared_frame_idx"] == 5) & (table["track"] == 0) & (table["node"] == "nose"),
        ["x", "y"],
    ] = np.nan
    timing = _timing_for_frames(n_frames)
    dense = construct_dense_pose(table, timing, apply_gap_fill=False)
    smooth = apply_savgol_smoothing(
        x=dense.x, y=dense.y, protected=dense.protected, config=SmoothConfig(enabled=False)
    )
    final = build_final_pose_table(dense, smooth)

    assert list(final.columns) == list(FINAL_POSE_COLUMNS)
    assert len(final) == dense.n_samples * dense.n_tracks * dense.n_nodes
    assert set(final["schema_version"].unique()) == {FINAL_POSE_SCHEMA_VERSION}

    missing = final[
        (final["sample_idx"] == 5) & (final["track"] == 0) & (final["node"] == "nose")
    ]
    assert len(missing) == 1
    assert np.isnan(float(missing.iloc[0]["x"]))
    assert np.isnan(float(missing.iloc[0]["y"]))
    assert np.isnan(float(missing.iloc[0]["x_pre_smooth"]))
    assert bool(missing.iloc[0]["was_smoothed"]) is False


def test_original_coords_provenance_and_confidence_retained() -> None:
    n_frames = 40
    table = _pose_rows(n_frames=n_frames, noise=0.5)
    timing = _timing_for_frames(n_frames)
    dense = construct_dense_pose(table, timing, apply_gap_fill=True)
    prov_before = dense.provenance.copy()
    score_before = dense.point_score.copy()
    pre_x = dense.x.copy()
    pre_y = dense.y.copy()

    smooth = apply_savgol_smoothing(x=dense.x, y=dense.y, protected=dense.protected)
    final = build_final_pose_table(dense, smooth)

    assert np.array_equal(dense.provenance, prov_before)
    assert np.allclose(dense.point_score, score_before, equal_nan=True)
    assert np.allclose(smooth.x_pre_smooth, pre_x, equal_nan=True)
    assert np.allclose(smooth.y_pre_smooth, pre_y, equal_nan=True)
    assert np.allclose(
        final["point_score"].to_numpy(), dense.point_score.reshape(-1), equal_nan=True
    )
    assert np.array_equal(
        final["provenance"].to_numpy(), dense.provenance.reshape(-1)
    )


def test_qc_warnings_versus_structural_failures() -> None:
    n_frames = 40
    table = _pose_rows(n_frames=n_frames, noise=0.0, jump_at=20)
    timing = _timing_for_frames(n_frames)
    dense = construct_dense_pose(table, timing, apply_gap_fill=False)
    smooth = apply_savgol_smoothing(
        x=dense.x, y=dense.y, protected=dense.protected, config=SmoothConfig(enabled=False)
    )
    final = build_final_pose_table(dense, smooth)

    warned = evaluate_finalization_qc(
        dense=dense,
        smooth=smooth,
        final_table=final,
        config=QcConfig(jump_px_warn=50.0, missing_fraction_warn=0.99),
    )
    assert warned.structural_ok is True
    assert warned.has_warnings is True
    assert any("implausible" in w for w in warned.warnings)

    # Structural: corrupt protected blank preservation.
    bad_smooth = apply_savgol_smoothing(
        x=dense.x, y=dense.y, protected=dense.protected, config=SmoothConfig(enabled=False)
    )
    # Build a protected mask after the fact by mutating a copy path via evaluate inputs.
    # Simulate protected samples incorrectly carrying finite coordinates.
    from pose_finalization.smoothing import SmoothResult

    protected = np.zeros_like(dense.protected)
    protected[0, 0, 0] = True
    x = np.array(bad_smooth.x, copy=True)
    y = np.array(bad_smooth.y, copy=True)
    x[0, 0, 0] = 1.0
    y[0, 0, 0] = 2.0
    fake_dense = construct_dense_pose(table, timing, apply_gap_fill=False)
    # Overwrite protected/provenance on a mutable copy through object.__setattr__ workaround:
    # DensePoseResult is frozen; instead patch arrays used by QC via a tiny wrapper table.
    mutable_prot = np.array(fake_dense.protected, copy=True)
    mutable_prov = np.array(fake_dense.provenance, copy=True)
    mutable_prot[0, 0, 0] = True
    mutable_prov[0, 0, 0] = PROVENANCE_PROTECTED_MANUAL_BLANK
    object.__setattr__(fake_dense, "protected", mutable_prot)
    object.__setattr__(fake_dense, "provenance", mutable_prov)
    fake_smooth = SmoothResult(
        x=x,
        y=y,
        x_pre_smooth=np.array(x, copy=True),
        y_pre_smooth=np.array(y, copy=True),
        was_smoothed=np.zeros_like(x, dtype=bool),
        config=SmoothConfig(enabled=False),
    )
    failed = evaluate_finalization_qc(
        dense=fake_dense,
        smooth=fake_smooth,
        final_table=build_final_pose_table(fake_dense, fake_smooth),
    )
    assert failed.structural_ok is False
    assert any("protected" in f for f in failed.failures)


def test_end_to_end_runner_completed_and_upstream_non_mutation(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=40, noise=0.4)
    tracked = s3_run / FINAL_TRACKED_POSE_FILENAME
    meta = s3_run / S3_RUN_META_FILENAME
    sync = session / "preprocess" / "prepared_sync.npz"
    before = _snapshot([tracked, meta, sync])

    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_run,
            timestamp="20260101T120000",
            run_purpose="sprint3_test",
        )
    )
    assert result.success is True
    assert result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    assert result.final_pose_path is not None
    assert result.final_pose_path.is_file()
    assert (result.run_dir / RUN_META_FILENAME).is_file()
    assert (result.run_dir / "settings_used.yaml").is_file()
    assert (result.run_dir / "processing_log.txt").is_file()

    final = pd.read_parquet(result.final_pose_path)
    assert list(final.columns) == list(FINAL_POSE_COLUMNS)
    assert len(final) == 40 * 2 * 2
    assert _snapshot([tracked, meta, sync]) == before

    meta_payload = json.loads((result.run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta_payload["status"] == result.status
    assert meta_payload["source_s3"]["tracked_pose_sha256"] == _sha256(tracked)
    assert meta_payload["qc"] is not None
    assert meta_payload["settings"]["geometric_refinement"]["status"] == "deferred"


def test_runner_completed_with_warnings_for_jumps(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=40, noise=0.0, jump_at=20)
    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_run,
            timestamp="20260101T130000",
            qc_config=QcConfig(jump_px_warn=50.0, missing_fraction_warn=0.99),
            smooth_config=SmoothConfig(enabled=False),
        )
    )
    assert result.status == STATUS_COMPLETED_WITH_WARNINGS
    assert result.warnings
    meta = json.loads(result.run_meta_path.read_text(encoding="utf-8"))
    assert meta["qc"]["warning_count"] >= 1


def test_s3_acceptance_superseded_during_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=30, noise=0.2)
    tracked = s3_run / FINAL_TRACKED_POSE_FILENAME
    before = _snapshot([tracked, s3_run / S3_RUN_META_FILENAME])

    import pose_finalization.runner as runner_mod
    from pose_finalization.accepted_s3 import validate_accepted_s3_run as real_validate

    calls = {"n": 0}

    def flipping_validate(path: Path | str):
        calls["n"] += 1
        if calls["n"] >= 2:
            meta_path = Path(path) / S3_RUN_META_FILENAME
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return real_validate(path)

    monkeypatch.setattr(runner_mod, "validate_accepted_s3_run", flipping_validate)

    with pytest.raises(PoseFinalizationError, match="not currently accepted|superseded"):
        run_pose_finalization(
            PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20260101T140000")
        )

    run_dir = session / "pose_finalization" / "pose_finalization__20260101T140000"
    assert run_dir.is_dir()
    assert not (run_dir / FINAL_POSE_FILENAME).exists()
    meta = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == STATUS_FAILED
    assert meta["outputs"]["final_pose_parquet"] is None
    # Upstream S3 pose bytes unchanged; acceptance state was intentionally mutated.
    assert _sha256(tracked) == before[tracked][0]


def test_s4_run_stale_when_source_s3_superseded(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=30, noise=0.2)
    result = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20260101T150000")
    )
    assert is_s4_run_eligible_for_downstream(result.run_dir) is True

    meta_path = s3_run / S3_RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    assert is_s4_run_eligible_for_downstream(result.run_dir) is False
    with pytest.raises(PoseFinalizationError, match="not currently accepted"):
        validate_s4_run_for_downstream(result.run_dir)


def test_runner_failure_leaves_incomplete_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(session, n_frames=25, noise=0.1)

    import pose_finalization.runner as runner_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced export failure")

    monkeypatch.setattr(runner_mod, "write_final_pose_parquet", boom)

    with pytest.raises(PoseFinalizationError, match="forced export failure"):
        run_pose_finalization(
            PoseFinalizationRequest(s3_run_dir=s3_run, timestamp="20260101T160000")
        )

    run_dir = session / "pose_finalization" / "pose_finalization__20260101T160000"
    assert not (run_dir / FINAL_POSE_FILENAME).exists()
    meta = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == STATUS_FAILED
    assert "forced export failure" in meta["error"]


def test_protected_blank_preserved_through_runner(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3_run = _write_accepted_session(
        session,
        n_frames=40,
        noise=0.2,
        blank=(10, 0, "nose"),
    )
    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_run,
            timestamp="20260101T170000",
            smooth_config=SmoothConfig(enabled=True),
        )
    )
    final = pd.read_parquet(result.final_pose_path)
    rows = final[
        (final["sample_idx"] == 10) & (final["track"] == 0) & (final["node"] == "nose")
    ]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert bool(row["protected_manual_blank"]) is True
    assert int(row["provenance"]) == PROVENANCE_PROTECTED_MANUAL_BLANK
    assert np.isnan(float(row["x"]))
    assert np.isnan(float(row["y"]))
    assert bool(row["was_smoothed"]) is False
    assert int(final.loc[final["protected_manual_blank"], "provenance"].nunique()) == 1
    assert PROVENANCE_ORIGINAL in set(final["provenance"].tolist())
