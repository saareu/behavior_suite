"""Discovery of S3 runs under a project/session directory."""

from __future__ import annotations

import json
from pathlib import Path

from tests.tracking_correction.test_review import TIMESTAMP, _completed_s2

from tracking_correction.contracts import TrackingCorrectionRequest
from tracking_correction.run_discovery import (
    COMPLETE_REVIEWABLE,
    MISSING_REQUIRED_ARTIFACTS,
    summarize_tracking_correction_project,
)
from tracking_correction.runner import run_tracking_correction


def test_summarize_project_discovers_preprocess_pose_and_s3_runs(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    session_root = s1["session_root"]
    first = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=s2_run,
            dry_run=False,
            timestamp=TIMESTAMP,
        )
    )
    second = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=s2_run,
            dry_run=False,
            timestamp="20260301T150000",
        )
    )
    assert first.success and second.success

    incomplete = session_root / "tracking_correction" / "broken_run"
    incomplete.mkdir(parents=True)
    (incomplete / "run_meta.json").write_text(
        json.dumps({"status": "correction_failed", "backend_status": "failed", "dry_run": False})
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_tracking_correction_project(session_root)
    assert summary.session_root == session_root.resolve()
    assert summary.artifact_presence.preprocess is True
    assert summary.artifact_presence.pose_inference is True
    assert summary.artifact_presence.tracking_correction is True
    assert len(summary.runs) >= 3
    reviewable = summary.reviewable_runs
    assert {run.run_dir.resolve() for run in reviewable} == {
        first.run_dir.resolve(),
        second.run_dir.resolve(),
    }
    assert all(run.classification == COMPLETE_REVIEWABLE for run in reviewable)
    broken = next(run for run in summary.runs if run.run_dir.name == "broken_run")
    assert broken.classification == MISSING_REQUIRED_ARTIFACTS
