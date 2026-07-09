import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from pose_inference import (
    PoseInferenceError,
    PoseInferenceRequest,
    build_sleap_predict_command,
    load_inference_profile,
    run_pose_inference,
    validate_s1_handoff,
)
from pose_inference.__main__ import app
from pose_inference.parquet_export import ParquetExportError, ParquetExportSummary

DEFAULT_PROFILE = Path("configs/subsystem_02/sleapnn_default_profile.yaml")
FORBIDDEN_ARTIFACT_NAMES = (
    "pose_tracked.slp",
    "overlay_tracked.mp4",
    "tracking_qc.csv",
    "tracking_report.json",
    "track_identity_map.json",
)


def _write_s1_handoff(root: Path) -> Path:
    preprocess = root / "preprocess"
    preprocess.mkdir(parents=True)
    (preprocess / "prepared_video.mp4").write_bytes(b"")
    (preprocess / "prepare_meta.json").write_text(
        json.dumps({"schema_version": "prepare_meta_v1"}) + "\n",
        encoding="utf-8",
    )
    (preprocess / "prepared_sync.npz").write_bytes(b"")
    return preprocess


def _model_path(tmp_path: Path) -> Path:
    model = tmp_path / "2mice_bottomup_model"
    model.mkdir()
    return model


def test_s1_handoff_validation_accepts_session_root_and_preprocess_dir(
    tmp_path: Path,
) -> None:
    preprocess = _write_s1_handoff(tmp_path)

    from_session = validate_s1_handoff(tmp_path)
    from_preprocess = validate_s1_handoff(preprocess)

    assert from_session.session_root == tmp_path.resolve()
    assert from_session.preprocess_dir == preprocess.resolve()
    assert from_preprocess.session_root == tmp_path.resolve()
    assert from_preprocess.prepared_video.name == "prepared_video.mp4"


def test_s1_handoff_validation_reports_missing_files(tmp_path: Path) -> None:
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    (preprocess / "prepared_video.mp4").write_bytes(b"")

    with pytest.raises(PoseInferenceError, match="prepare_meta.json"):
        validate_s1_handoff(tmp_path)


def test_generated_command_writes_to_pose_slp_and_keeps_tracking_in_same_call(
    tmp_path: Path,
) -> None:
    _write_s1_handoff(tmp_path)
    handoff = validate_s1_handoff(tmp_path)
    profile = load_inference_profile(DEFAULT_PROFILE)
    pose_slp = tmp_path / "pose_inference" / "model__20260709T010203" / "pose.slp"

    command = build_sleap_predict_command(
        handoff=handoff,
        model_path=_model_path(tmp_path),
        profile=profile,
        pose_slp_path=pose_slp,
    )

    assert command[:2] == ("sleap-nn", "predict")
    assert "--data_path" in command
    assert "--model_paths" in command
    assert "--output_path" in command
    assert str(pose_slp.resolve()) in command
    assert "--tracking" in command
    assert "--max_tracks" in command
    assert "--candidates_method" in command
    assert "--tracking_window_size" in command
    assert "--features" in command
    assert "--output-path" not in command
    assert "--model-path" not in command
    assert "--max-tracks" not in command
    assert command.count("sleap-nn") == 1
    assert not any(name in " ".join(command) for name in FORBIDDEN_ARTIFACT_NAMES)


def test_min_instance_peaks_is_omitted_when_profile_value_is_null(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    handoff = validate_s1_handoff(tmp_path)
    profile = load_inference_profile(DEFAULT_PROFILE)
    assert profile["min_instance_peaks"] is None

    command = build_sleap_predict_command(
        handoff=handoff,
        model_path=_model_path(tmp_path),
        profile=profile,
        pose_slp_path=tmp_path / "pose.slp",
    )

    assert "--min_instance_peaks" not in command
    assert "--min-instance-peaks" not in command


def test_dry_run_creates_minimal_records_without_external_sleap(
    tmp_path: Path,
) -> None:
    _write_s1_handoff(tmp_path)
    called = False

    def unexpected_subprocess(**_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(args=[], returncode=99, stdout="", stderr="")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=_model_path(tmp_path),
            profile_path=DEFAULT_PROFILE,
            run_purpose="development",
            dry_run=True,
            timestamp="20260709T010203",
        ),
        subprocess_runner=unexpected_subprocess,
    )

    assert called is False
    assert result.success is True
    assert result.status == "dry_run_complete"
    assert result.run_dir == tmp_path / "pose_inference" / "2mice_bottomup_model__20260709T010203"
    assert result.pose_meta_path.is_file()
    assert result.settings_used_path.is_file()
    assert result.job_manifest_path.is_file()
    assert result.processing_log_path.is_file()
    assert not result.pose_slp_path.exists()
    assert not (result.run_dir / "pose.parquet").exists()
    assert not (result.run_dir / "overlay.mp4").exists()

    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    settings = yaml.safe_load(result.settings_used_path.read_text(encoding="utf-8"))
    assert pose_meta["artifact_status"]["pose_slp"] == "dry_run_not_generated"
    assert pose_meta["artifact_status"]["pose_parquet"] == "not_generated"
    assert pose_meta["artifact_status"]["overlay_mp4"] == "not_generated"
    assert pose_meta["pose_qc"]["status"] == "not_computed"
    assert manifest["dry_run"] is True
    assert manifest["outputs"]["pose_slp"].endswith("pose.slp")
    assert settings["tracking_enabled"] is True
    assert "--output_path" in settings["command"]


def test_output_root_override_and_run_directory_naming(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    output_root = tmp_path / "custom_outputs"

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path / "preprocess",
            model_path=_model_path(tmp_path),
            profile_path=DEFAULT_PROFILE,
            output_root=output_root,
            dry_run=True,
            timestamp="20260709T020304",
        )
    )

    assert result.run_id == "2mice_bottomup_model__20260709T020304"
    assert result.run_dir == output_root / "2mice_bottomup_model__20260709T020304"
    assert result.run_dir.is_dir()


def test_no_separate_tracking_artifacts_are_generated(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=_model_path(tmp_path),
            profile_path=DEFAULT_PROFILE,
            dry_run=True,
            timestamp="20260709T030405",
        )
    )

    generated_names = {path.name for path in result.run_dir.iterdir()}
    assert not generated_names.intersection(FORBIDDEN_ARTIFACT_NAMES)
    assert {
        "pose_meta.json",
        "settings_used.yaml",
        "job_manifest.yaml",
        "processing_log.txt",
    }.issubset(generated_names)


def test_runner_exports_parquet_after_successful_inference(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)
    export_calls: list[dict[str, Path]] = []

    def fake_subprocess(args, **kwargs):
        run_dir = Path(kwargs["cwd"])
        (run_dir / "pose.slp").write_bytes(b"slp")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    def fake_exporter(**kwargs):
        export_calls.append(dict(kwargs))
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"parquet")
        return ParquetExportSummary(
            status="generated",
            path=output_path,
            rows=12,
            timestamp_sources=("prepared_sync:time_sec",),
        )

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=model_path,
            profile_path=DEFAULT_PROFILE,
            dry_run=False,
            timestamp="20260709T040506",
        ),
        subprocess_runner=fake_subprocess,
        parquet_exporter=fake_exporter,
    )

    assert result.success is True
    assert result.status == "success"
    assert result.pose_slp_path.is_file()
    assert (result.run_dir / "pose.parquet").is_file()
    assert export_calls == [
        {
            "pose_slp_path": result.pose_slp_path,
            "preprocess_dir": tmp_path / "preprocess",
            "output_path": result.run_dir / "pose.parquet",
        }
    ]
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["artifact_status"]["pose_slp"] == "generated"
    assert pose_meta["artifact_status"]["pose_parquet"] == "generated"
    assert pose_meta["parquet"]["rows"] == 12
    assert pose_meta["parquet"]["timestamp_sources"] == ["prepared_sync:time_sec"]
    assert manifest["artifact_status"]["pose_parquet"] == "generated"
    assert "parquet_status: generated" in log_text


def test_runner_marks_export_failed_when_parquet_export_fails(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)

    def fake_subprocess(args, **kwargs):
        run_dir = Path(kwargs["cwd"])
        (run_dir / "pose.slp").write_bytes(b"slp")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    def failing_exporter(**_kwargs):
        raise ParquetExportError("forced export failure")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=model_path,
            profile_path=DEFAULT_PROFILE,
            dry_run=False,
            timestamp="20260709T050607",
        ),
        subprocess_runner=fake_subprocess,
        parquet_exporter=failing_exporter,
    )

    assert result.success is False
    assert result.status == "export_failed"
    assert result.pose_slp_path.is_file()
    assert not (result.run_dir / "pose.parquet").exists()
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["artifact_status"]["pose_slp"] == "generated"
    assert pose_meta["artifact_status"]["pose_parquet"] == "failed"
    assert pose_meta["parquet"]["status"] == "failed"
    assert "forced export failure" in pose_meta["parquet"]["error"]
    assert manifest["artifact_status"]["pose_parquet"] == "failed"
    assert "parquet_status: failed" in log_text
    assert "parquet_error: forced export failure" in log_text


def test_cli_dry_run_entrypoint(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run",
            "--session-root",
            str(tmp_path),
            "--model-path",
            str(model_path),
            "--profile",
            str(DEFAULT_PROFILE),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Status: dry_run_complete" in result.output
    assert "pose.slp" in result.output
    run_dirs = list((tmp_path / "pose_inference").glob("2mice_bottomup_model__*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "pose_meta.json").is_file()
    assert not (run_dirs[0] / "pose.slp").exists()
