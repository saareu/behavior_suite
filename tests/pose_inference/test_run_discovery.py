import json
from pathlib import Path
from typing import Any

import yaml

from pose_inference.run_discovery import (
    COMPLETE_REVIEWABLE,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    summarize_pose_inference_project,
)


def _write_s1_handoff(root: Path) -> None:
    preprocess = root / "preprocess"
    preprocess.mkdir()
    (preprocess / "prepared_video.mp4").write_bytes(b"video")
    (preprocess / "prepare_meta.json").write_text("{}\n", encoding="utf-8")
    (preprocess / "prepared_sync.npz").write_bytes(b"sync")


def _write_complete_run(
    root: Path,
    run_id: str = "model_a__20260709T010203",
    *,
    status: str = "success",
    dry_run: bool = False,
    write_pose_meta: bool = True,
    write_job_manifest: bool = True,
    missing_artifacts: set[str] | None = None,
    inference_config: dict[str, Any] | None = None,
    tracking_config: dict[str, Any] | None = None,
    parquet_timing: dict[str, Any] | None = None,
) -> Path:
    run_dir = root / "pose_inference" / run_id
    run_dir.mkdir(parents=True)
    missing = missing_artifacts or set()
    artifact_payloads = {
        "pose.slp": b"slp",
        "pose.parquet": b"parquet",
        "overlay.mp4": b"overlay",
        "processing_log.txt": (
            f"status: {status}\n"
            f"dry_run: {dry_run}\n"
            "parquet_status: generated\n"
            "pose_qc: computed\n"
            "overlay: generated\n"
            "sleap_provenance_status: available\n"
        ).encode(),
    }
    for filename, payload in artifact_payloads.items():
        if filename not in missing:
            (run_dir / filename).write_bytes(payload)

    if "settings_used.yaml" not in missing:
        (run_dir / "settings_used.yaml").write_text(
            yaml.safe_dump(
                {
                    "profile_id": "bottomup_default",
                    "profile_path": "configs/subsystem_02/sleapnn_default_profile.yaml",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    if write_pose_meta and "pose_meta.json" not in missing:
        (run_dir / "pose_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "created_at": "2026-07-09T01:02:03+03:00",
                    "status": status,
                    "dry_run": dry_run,
                    "model": {
                        "model_id": "model_a",
                        "model_path": "C:/models/model_a",
                    },
                    "settings": {
                        "profile_id": "bottomup_default",
                        "profile_path": "profile.yaml",
                    },
                    "artifact_status": {
                        "pose_slp": "generated",
                        "pose_parquet": "generated",
                        "overlay_mp4": "generated",
                    },
                    "parquet": {
                        "status": "generated",
                        "timing": parquet_timing or {"status": "prepared_sync"},
                    },
                    "pose_qc": {"status": "computed"},
                    "overlay": {"status": "generated"},
                    "sleap_provenance": {
                        "status": "available",
                        "inference_config": inference_config,
                        "tracking_config": tracking_config,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    if write_job_manifest and "job_manifest.yaml" not in missing:
        (run_dir / "job_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "run_id": run_id,
                    "created_at": "2026-07-09T01:02:03+03:00",
                    "completed_at": "2026-07-09T01:03:03+03:00",
                    "status": status,
                    "dry_run": dry_run,
                    "model": {
                        "model_id": "model_a",
                        "model_path": "C:/models/model_a",
                    },
                    "profile_path": "profile.yaml",
                    "artifact_status": {
                        "pose_slp": "generated",
                        "pose_parquet": "generated",
                        "overlay_mp4": "generated",
                    },
                    "qc_status": "computed",
                    "overlay_status": "generated",
                    "sleap_provenance_status": "available",
                    "effective_sleap_inference_config": inference_config,
                    "effective_sleap_tracking_config": tracking_config,
                    "parquet_timing": parquet_timing or {"status": "prepared_sync"},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return run_dir


def test_no_preprocess_dir_reports_s1_handoff_incomplete(tmp_path: Path) -> None:
    summary = summarize_pose_inference_project(tmp_path)

    assert summary.s1_handoff.is_complete is False
    assert summary.s1_handoff.missing_files == (
        "prepared_video.mp4",
        "prepare_meta.json",
        "prepared_sync.npz",
    )


def test_complete_s1_handoff_reports_complete(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)

    summary = summarize_pose_inference_project(tmp_path)

    assert summary.s1_handoff.is_complete is True
    assert summary.s1_handoff.artifact_presence == {
        "prepared_video": True,
        "prepare_meta": True,
        "prepared_sync": True,
    }


def test_no_pose_inference_dir_reports_zero_s2_runs(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)

    summary = summarize_pose_inference_project(tmp_path)

    assert summary.runs == ()


def test_complete_s2_run_is_classified_complete_reviewable(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)

    summary = summarize_pose_inference_project(tmp_path)

    assert len(summary.runs) == 1
    run = summary.runs[0]
    assert run.classification == COMPLETE_REVIEWABLE
    assert run.run_id == "model_a__20260709T010203"
    assert run.status == "success"
    assert run.parquet_status == "generated"
    assert run.pose_qc_status == "computed"
    assert run.overlay_status == "generated"
    assert run.sleap_provenance_status == "available"


def test_missing_required_artifact_is_classified_missing_required_artifacts(
    tmp_path: Path,
) -> None:
    _write_complete_run(tmp_path, missing_artifacts={"overlay.mp4"})

    run = summarize_pose_inference_project(tmp_path).runs[0]

    assert run.classification == MISSING_REQUIRED_ARTIFACTS
    assert run.missing_required_artifacts == ("overlay.mp4",)
    assert run.artifact_presence["overlay_mp4"] is False


def test_failed_metadata_status_is_classified_failed_or_incomplete(
    tmp_path: Path,
) -> None:
    _write_complete_run(tmp_path, status="export_failed")

    run = summarize_pose_inference_project(tmp_path).runs[0]

    assert run.classification == FAILED_OR_INCOMPLETE
    assert run.status == "export_failed"


def test_dry_run_metadata_status_is_classified_failed_or_incomplete(
    tmp_path: Path,
) -> None:
    _write_complete_run(tmp_path, status="dry_run_complete", dry_run=True)

    run = summarize_pose_inference_project(tmp_path).runs[0]

    assert run.classification == FAILED_OR_INCOMPLETE
    assert run.dry_run is True


def test_dry_run_missing_outputs_is_classified_failed_or_incomplete(
    tmp_path: Path,
) -> None:
    _write_complete_run(
        tmp_path,
        status="dry_run_complete",
        dry_run=True,
        missing_artifacts={"pose.slp", "pose.parquet", "overlay.mp4"},
    )

    run = summarize_pose_inference_project(tmp_path).runs[0]

    assert run.classification == FAILED_OR_INCOMPLETE
    assert run.missing_required_artifacts == (
        "pose.slp",
        "pose.parquet",
        "overlay.mp4",
    )


def test_corrupt_metadata_does_not_crash_discovery(tmp_path: Path) -> None:
    run_dir = _write_complete_run(tmp_path, write_pose_meta=False, write_job_manifest=False)
    (run_dir / "pose_meta.json").write_text("{not valid json", encoding="utf-8")
    (run_dir / "job_manifest.yaml").write_text("invalid: [yaml", encoding="utf-8")

    run = summarize_pose_inference_project(tmp_path).runs[0]

    assert run.classification == FAILED_OR_INCOMPLETE
    assert len(run.metadata_errors) == 2
    assert any("pose_meta.json" in error for error in run.metadata_errors)
    assert any("job_manifest.yaml" in error for error in run.metadata_errors)


def test_runs_are_sorted_newest_first_when_possible(tmp_path: Path) -> None:
    _write_complete_run(tmp_path, run_id="model_a__20260709T010203")
    _write_complete_run(tmp_path, run_id="model_a__20260710T010203")

    summary = summarize_pose_inference_project(tmp_path)

    assert [run.run_id for run in summary.runs] == [
        "model_a__20260710T010203",
        "model_a__20260709T010203",
    ]


def test_effective_sleap_configs_and_parquet_timing_are_copied(
    tmp_path: Path,
) -> None:
    inference_config = {"peak_threshold": 0.2, "batch_size": 4}
    tracking_config = {"tracking": True, "max_tracks": 2}
    parquet_timing = {
        "status": "prepared_sync",
        "source": "prepared_sync",
        "frames_with_time": 3,
    }
    _write_complete_run(
        tmp_path,
        inference_config=inference_config,
        tracking_config=tracking_config,
        parquet_timing=parquet_timing,
    )

    run = summarize_pose_inference_project(tmp_path).runs[0]

    assert run.effective_sleap_inference_config == inference_config
    assert run.effective_sleap_tracking_config == tracking_config
    assert run.parquet_timing == parquet_timing
    assert run.model_id == "model_a"
    assert run.profile_id == "bottomup_default"
