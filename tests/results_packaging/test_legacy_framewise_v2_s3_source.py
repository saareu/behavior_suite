"""Accepted-S3 source eligibility tests for legacy_framewise_v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from results_packaging.legacy_framewise_csv import (
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LegacyFramewiseSourceError,
)
from results_packaging.legacy_framewise_csv.source import (
    _load_and_validate_accepted_s3,
    inspect_package_s3_lineage,
)
from tracking_correction.contracts import FINAL_TRACKED_POSE_FILENAME, RUN_META_FILENAME


def _write_tracked_pose(path: Path, n_frames: int = 3) -> None:
    rows = []
    for frame in range(n_frames):
        for track in (0, 1):
            for node in ("headstage", "neck", "nose", "spine_base", "tail_base"):
                if node == "headstage" and track == 1:
                    continue
                rows.append(
                    {
                        "prepared_frame_idx": frame,
                        "track": track,
                        "node": node,
                        "x": float(frame + track),
                        "y": float(frame + 10 + track),
                    }
                )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _accepted_run(tmp_path: Path, *, accepted: bool = True) -> Path:
    session = tmp_path / "session"
    prep = session / "preprocess"
    prep.mkdir(parents=True)
    # Minimal prepared_sync.npz is not required for _load_and_validate_accepted_s3
    # session_root check when layout resolves from tracking_correction parent.
    run_dir = session / "tracking_correction" / "tracking_correction__t1"
    run_dir.mkdir(parents=True)
    pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    _write_tracked_pose(pose_path)
    from project.file_digest import sha256_file

    digest = sha256_file(pose_path)
    # Create a dummy prepared_sync so validate_accepted_s3_run can resolve it.
    np.savez(
        prep / "prepared_sync.npz",
        sync_schema_version=np.asarray("prepared_sync_v1"),
        prepared_frame_idx=np.arange(3, dtype=np.int64),
        raw_decode_frame_idx=np.arange(3, dtype=np.int64),
        prepared_time_sec=np.arange(3, dtype=np.float64) / 30.0,
        raw_pts_time_sec=np.full(3, np.nan),
        raw_pts_status=np.asarray("not_provided"),
        external_time_sec=np.full(3, np.nan),
        external_time_status=np.asarray("not_provided"),
        external_time_source=np.asarray(None),
        external_time_variable_name=np.asarray(None),
        external_time_units=np.asarray("unknown"),
        fps_header=np.asarray(30.0),
        raw_fps_effective=np.asarray(30.0),
        start_frame=np.asarray(0),
        end_frame_exclusive=np.asarray(3),
        raw_frame_count_opencv_readable=np.asarray(3),
        prepared_frame_count_opencv_reported=np.asarray(3),
        prepared_frame_count_opencv_readable=np.asarray(3),
        frame_count_used_for_sleap=np.asarray(3),
    )
    meta = {
        "run_id": run_dir.name,
        "acceptance_state": "accepted" if accepted else "complete_reviewable",
        "acceptance": {
            "tracked_pose_sha256": digest,
            "accepted_at": "2026-01-01T00:00:00Z",
        },
        "input": {"session_root": str(session)},
    }
    (run_dir / RUN_META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    return run_dir


def test_accepted_s3_loads_for_v2(tmp_path: Path):
    run_dir = _accepted_run(tmp_path, accepted=True)
    session = run_dir.parent.parent
    resolved = _load_and_validate_accepted_s3(
        root=session,
        run_dir=run_dir,
        expected_path=None,
        expected_sha256=None,
        expected_run_id=None,
        selection_mode="explicit",
    )
    assert resolved.s3_run_id == run_dir.name
    assert resolved.tracked_pose_sha256
    assert not resolved.table.empty


def test_rejects_unaccepted_s3_pose(tmp_path: Path):
    run_dir = _accepted_run(tmp_path, accepted=False)
    session = run_dir.parent.parent
    with pytest.raises(LegacyFramewiseSourceError, match="not eligible accepted"):
        _load_and_validate_accepted_s3(
            root=session,
            run_dir=run_dir,
            expected_path=None,
            expected_sha256=None,
            expected_run_id=None,
            selection_mode="explicit",
        )


def test_inspect_package_s3_lineage_fail_closed_without_pin(tmp_path: Path):
    pkg = tmp_path / "results_package__x"
    pkg.mkdir()
    (pkg / "package_manifest.json").write_text("{}", encoding="utf-8")
    (pkg / "run_meta.json").write_text("{}", encoding="utf-8")
    info = inspect_package_s3_lineage(pkg)
    assert info.available is False
    assert "unavailable" in info.reason.lower()


def test_v2_profile_constant():
    assert LEGACY_FRAMEWISE_V2_PROFILE_ID == "legacy_framewise_v2"
