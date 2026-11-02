"""Overlay QC job discovery under results_exports/."""

from __future__ import annotations

import json
from pathlib import Path

from overlay.contracts import (
    OVERLAY_QC_JOB_PREFIX,
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_QC_RUN_META_FILENAME,
    OVERLAY_QC_S4_POSE_JOB_PREFIX,
    OVERLAY_QC_S4_POSE_VIDEO_FILENAME,
    OVERLAY_QC_VIDEO_FILENAME,
)
from overlay.discovery import (
    EXPORT_FAMILY_BEHAVIORAL,
    EXPORT_FAMILY_S4_POSE,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    summarize_overlay_qc_project,
)


def test_discovery_prefix_filter_and_completed_vs_incomplete(tmp_path: Path) -> None:
    session = tmp_path / "session"
    exports = session / "results_exports"
    exports.mkdir(parents=True)

    # Unrelated export families must be ignored.
    (exports / "media_export__x").mkdir()
    (exports / "pi_report__y").mkdir()
    (exports / "csv_export__z").mkdir()

    behavioral = exports / f"{OVERLAY_QC_JOB_PREFIX}done"
    behavioral.mkdir()
    (behavioral / OVERLAY_QC_VIDEO_FILENAME).write_bytes(b"fake")
    (behavioral / OVERLAY_QC_MANIFEST_FILENAME).write_text(
        json.dumps({"layers": ["s8.exclusive.A_to_B"], "output_frame_count": 10}),
        encoding="utf-8",
    )
    (behavioral / OVERLAY_QC_RUN_META_FILENAME).write_text(
        json.dumps(
            {
                "source_kind": "s8_automated",
                "package_run_id": "results_package__demo",
                "frames_written": 10,
                "created_at_utc": "2026-11-02T00:00:00+00:00",
                "plan": {
                    "selected_layers": ["s8.exclusive.A_to_B", "debug.frame_counter"],
                    "prepared_interval": {
                        "prepared_frame_idx_start_inclusive": 0,
                        "prepared_frame_idx_end_inclusive": 9,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    incomplete = exports / f"{OVERLAY_QC_S4_POSE_JOB_PREFIX}partial"
    incomplete.mkdir()
    (incomplete / OVERLAY_QC_RUN_META_FILENAME).write_text(
        json.dumps({"s4_run_id": "pose_finalization__x", "qc_status": "completed"}),
        encoding="utf-8",
    )
    # Missing video + manifest => incomplete even if meta claims completed.

    summary = summarize_overlay_qc_project(session)
    assert len(summary.jobs) == 2
    by_id = {job.job_id: job for job in summary.jobs}
    done = by_id[behavioral.name]
    assert done.export_family == EXPORT_FAMILY_BEHAVIORAL
    assert done.status == STATUS_COMPLETED
    assert done.is_completed is True
    assert done.package_run_id == "results_package__demo"
    assert done.layers == ("s8.exclusive.A_to_B", "debug.frame_counter")
    assert done.prepared_frame_idx_start == 0
    assert done.prepared_frame_idx_end == 9
    assert done.has_video is True

    partial = by_id[incomplete.name]
    assert partial.export_family == EXPORT_FAMILY_S4_POSE
    assert partial.status == STATUS_INCOMPLETE
    assert partial.is_completed is False
    assert partial.s4_run_id == "pose_finalization__x"
    assert (exports / "media_export__x").is_dir()
    assert all(not j.job_id.startswith("media_export") for j in summary.jobs)


def test_discovery_completed_s4_pose_job(tmp_path: Path) -> None:
    session = tmp_path / "session"
    job = session / "results_exports" / f"{OVERLAY_QC_S4_POSE_JOB_PREFIX}ok"
    job.mkdir(parents=True)
    (job / OVERLAY_QC_S4_POSE_VIDEO_FILENAME).write_bytes(b"mp4")
    (job / OVERLAY_QC_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (job / OVERLAY_QC_RUN_META_FILENAME).write_text(
        json.dumps(
            {
                "s4_run_id": "pose_finalization__ok",
                "frames_written": 30,
                "plan": {"selected_layers": ["s4.pose_final"]},
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_overlay_qc_project(session)
    assert len(summary.completed_jobs) == 1
    assert summary.completed_jobs[0].video_filename == OVERLAY_QC_S4_POSE_VIDEO_FILENAME
