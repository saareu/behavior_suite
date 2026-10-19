"""Accepted-S3 discovery and validation for Subsystem 04 Sprint 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pose_finalization import (
    PoseFinalizationError,
    find_latest_accepted_s3_run,
    list_accepted_s3_runs,
    resolve_accepted_s3_run,
    validate_accepted_s3_run,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_NOT_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    FINAL_TRACKED_POSE_FILENAME,
    RUN_META_FILENAME,
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


def _write_prepared_sync(path: Path, *, frame_count: int = 4, fps: float = 20.0) -> Path:
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


def _write_tracked_pose(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "frame": [0, 0],
            "frame_idx": [0, 0],
            "track": [0, 1],
            "node": ["nose", "nose"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "point_score": [0.9, 0.8],
        }
    )
    frame.to_parquet(path, index=False)
    return _sha256(path)


def _write_accepted_run(
    session_root: Path,
    *,
    run_id: str,
    acceptance_state: str = ACCEPTANCE_ACCEPTED,
    include_tracked_pose: bool = True,
    mutate_sha: bool = False,
    omit_acceptance_block: bool = False,
    malformed_meta: bool = False,
    completed_at: str | None = None,
) -> Path:
    preprocess = session_root / "preprocess"
    sync_path = _write_prepared_sync(preprocess / "prepared_sync.npz")
    run_dir = session_root / "tracking_correction" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tracked_sha = None
    tracked_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    if include_tracked_pose:
        tracked_sha = _write_tracked_pose(tracked_path)
        if mutate_sha:
            tracked_sha = "0" * 64

    if malformed_meta:
        (run_dir / RUN_META_FILENAME).write_text("{not-json\n", encoding="utf-8")
        return run_dir

    acceptance_block: dict[str, object] | None
    if omit_acceptance_block:
        acceptance_block = None
    elif acceptance_state == ACCEPTANCE_ACCEPTED:
        acceptance_block = {
            "accepted_at": "2026-03-01T12:00:00",
            "tracked_pose_sha256": tracked_sha,
            "working_tracked_pose_sha256": tracked_sha,
            "tracked_pose_parquet": str(tracked_path),
        }
    elif acceptance_state == ACCEPTANCE_SUPERSEDED:
        acceptance_block = {
            "state": ACCEPTANCE_SUPERSEDED,
            "reason": "working_pose_changed_after_acceptance",
        }
    else:
        acceptance_block = None

    meta = {
        "schema_version": "tracking_correction_run_meta_v1",
        "run_id": run_id,
        "status": "correction_complete",
        "backend_status": "completed",
        "dry_run": False,
        "acceptance_state": acceptance_state,
        "profile_id": "current_lab_two_mouse_headstage_v1",
        "created_at": "2026-03-01T11:00:00",
        "completed_at": completed_at or "2026-03-01T12:00:00",
        "input": {
            "session_root": str(session_root),
            "prepared_sync": str(sync_path),
        },
    }
    if acceptance_block is not None:
        meta["acceptance"] = acceptance_block
    (run_dir / RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    # Existing S3 discovery also looks for working artifacts for classification;
    # create minimal placeholders so unaccepted runs still appear in summaries.
    (run_dir / "working_tracked_pose.parquet").write_bytes(b"placeholder")
    (run_dir / "machine_corrections.json").write_text("{}\n", encoding="utf-8")
    return run_dir


def test_validate_accepted_valid_artifact(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(session, run_id="profile__20260301T120000")
    protected = _snapshot(
        [
            run_dir / RUN_META_FILENAME,
            run_dir / FINAL_TRACKED_POSE_FILENAME,
            session / "preprocess" / "prepared_sync.npz",
        ]
    )

    accepted = validate_accepted_s3_run(run_dir)
    assert accepted.run_dir == run_dir.resolve()
    assert accepted.acceptance_state == ACCEPTANCE_ACCEPTED
    assert accepted.tracked_pose_path.is_file()
    assert accepted.prepared_sync_path.is_file()
    assert accepted.tracked_pose_sha256 == _sha256(accepted.tracked_pose_path)
    assert protected == _snapshot(list(protected))


def test_reject_not_accepted(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        acceptance_state=ACCEPTANCE_NOT_ACCEPTED,
    )
    with pytest.raises(PoseFinalizationError, match="not currently accepted"):
        validate_accepted_s3_run(run_dir)


def test_reject_superseded(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        acceptance_state=ACCEPTANCE_SUPERSEDED,
    )
    with pytest.raises(PoseFinalizationError, match="not currently accepted"):
        validate_accepted_s3_run(run_dir)


def test_reject_missing_tracked_pose(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        include_tracked_pose=False,
    )
    with pytest.raises(PoseFinalizationError, match="tracked_pose.parquet is missing"):
        validate_accepted_s3_run(run_dir)


def test_reject_sha256_mismatch(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        mutate_sha=True,
    )
    with pytest.raises(PoseFinalizationError, match="SHA256"):
        validate_accepted_s3_run(run_dir)


def test_reject_missing_acceptance_block(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        omit_acceptance_block=True,
    )
    with pytest.raises(PoseFinalizationError, match="acceptance block"):
        validate_accepted_s3_run(run_dir)


def test_reject_malformed_metadata(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        malformed_meta=True,
    )
    with pytest.raises(PoseFinalizationError, match="malformed"):
        validate_accepted_s3_run(run_dir)


def test_reject_missing_run_meta(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = session / "tracking_correction" / "orphan"
    run_dir.mkdir(parents=True)
    with pytest.raises(PoseFinalizationError, match="run_meta.json is missing"):
        validate_accepted_s3_run(run_dir)


def test_multiple_candidate_runs_selects_latest(tmp_path: Path) -> None:
    session = tmp_path / "session"
    older = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        completed_at="2026-03-01T12:00:00",
    )
    newer = _write_accepted_run(
        session,
        run_id="profile__20260301T150000",
        completed_at="2026-03-01T15:00:00",
    )
    _write_accepted_run(
        session,
        run_id="profile__20260301T140000",
        acceptance_state=ACCEPTANCE_NOT_ACCEPTED,
        completed_at="2026-03-01T14:00:00",
    )

    accepted_runs = list_accepted_s3_runs(session)
    assert [run.run_dir for run in accepted_runs] == [newer.resolve(), older.resolve()]

    latest = find_latest_accepted_s3_run(session)
    assert latest is not None
    assert latest.run_dir == newer.resolve()

    resolved = resolve_accepted_s3_run(session)
    assert resolved.run_dir == newer.resolve()


def test_explicit_selected_run_overrides_latest(tmp_path: Path) -> None:
    session = tmp_path / "session"
    older = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        completed_at="2026-03-01T12:00:00",
    )
    _write_accepted_run(
        session,
        run_id="profile__20260301T150000",
        completed_at="2026-03-01T15:00:00",
    )

    resolved = resolve_accepted_s3_run(session, s3_run_dir=older)
    assert resolved.run_dir == older.resolve()


def test_complete_reviewable_without_acceptance_is_not_eligible(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(
        session,
        run_id="profile__20260301T120000",
        acceptance_state=ACCEPTANCE_NOT_ACCEPTED,
    )
    # Ensure discovery finds the reviewable-looking run but S4 still rejects it.
    assert find_latest_accepted_s3_run(session) is None
    with pytest.raises(PoseFinalizationError, match="No eligible accepted S3 run"):
        resolve_accepted_s3_run(session)
    with pytest.raises(PoseFinalizationError, match="not currently accepted"):
        validate_accepted_s3_run(run_dir)


def test_operations_are_read_only_for_s1_s3(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = _write_accepted_run(session, run_id="profile__20260301T120000")
    sync_path = session / "preprocess" / "prepared_sync.npz"
    meta_path = run_dir / RUN_META_FILENAME
    pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    before = _snapshot([sync_path, meta_path, pose_path])

    validate_accepted_s3_run(run_dir)
    list_accepted_s3_runs(session)
    resolve_accepted_s3_run(session, s3_run_dir=run_dir)

    assert before == _snapshot([sync_path, meta_path, pose_path])
