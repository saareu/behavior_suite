import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import pose_inference.runner as runner_module
from pose_inference import (
    PoseInferenceError,
    PoseInferenceModelSpec,
    PoseInferenceRequest,
    build_sleap_predict_command,
    load_inference_profile,
    run_pose_inference,
    validate_s1_handoff,
)
from pose_inference.__main__ import app
from pose_inference.overlay import OverlayGenerationError, OverlaySummary
from pose_inference.parquet_export import ParquetExportError, ParquetExportSummary
from pose_inference.pose_qc import PoseQcError, PoseQcSummary
from pose_inference.runner import (
    SLEAP_PROVENANCE_NOTE,
    SleapProvenanceSummary,
    SleapRuntime,
    _model_bundle_id,
    _pose_qc_options,
    _query_sleap_version,
    _resolve_sleap_executable,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz

DEFAULT_PROFILE = Path("configs/subsystem_02/sleapnn_default_profile.yaml")
TOPDOWN_PROFILE = Path(
    "configs/subsystem_02/sleapnn_topdown_default_profile.yaml"
)
FORBIDDEN_ARTIFACT_NAMES = (
    "pose_tracked.slp",
    "overlay_tracked.mp4",
    "tracking_qc.csv",
    "tracking_report.json",
    "track_identity_map.json",
)


@pytest.fixture(autouse=True)
def stable_sleap_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SleapRuntime:
    executable = tmp_path / "runtime" / "sleap-nn.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"fake executable")
    runtime = SleapRuntime(
        executable_path=executable.resolve(),
        version="0.3.0",
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_sleap_runtime",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_sleap_executable",
        lambda *, explicit_path, profile: (
            Path(explicit_path).resolve()
            if explicit_path is not None
            else runtime.executable_path
        ),
    )
    return runtime


def _write_s1_handoff(
    root: Path,
    *,
    video_frame_count: int = 3,
    sync_frame_count: int | None = None,
) -> Path:
    preprocess = root / "preprocess"
    preprocess.mkdir(parents=True)
    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (16, 16),
    )
    assert writer.isOpened()
    for frame_idx in range(video_frame_count):
        writer.write(np.full((16, 16, 3), frame_idx, dtype=np.uint8))
    writer.release()
    authoritative_count = sync_frame_count or video_frame_count
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "opencv_reported_frame_count": authoritative_count,
                    "opencv_readable_frame_count": authoritative_count,
                    "frame_count_used_for_sleap": authoritative_count,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sync = build_prepared_sync(
        prepared_frame_count=authoritative_count,
        start_frame=0,
        end_frame_exclusive=authoritative_count,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.arange(authoritative_count, dtype=float) / 20.0,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=authoritative_count,
        prepared_frame_count_opencv_reported=authoritative_count,
        prepared_frame_count_opencv_readable=authoritative_count,
    )
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    return preprocess


def _model_path(tmp_path: Path) -> Path:
    model = tmp_path / "2mice_bottomup_model"
    model.mkdir()
    return model


def _role_model_path(tmp_path: Path, name: str, role: str) -> Path:
    model = tmp_path / name
    model.mkdir()
    (model / "training_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model_config": {
                    "head_configs": {
                        "centroid": {} if role == "centroid" else None,
                        "centered_instance": {}
                        if role == "centered_instance"
                        else None,
                        "bottomup": {} if role == "bottomup" else None,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return model


def _topdown_model_spec(tmp_path: Path) -> PoseInferenceModelSpec:
    return PoseInferenceModelSpec(
        inference_mode="topdown",
        centroid_model_path=_role_model_path(
            tmp_path, "centroid model with spaces", "centroid"
        ),
        centered_instance_model_path=_role_model_path(
            tmp_path, "centered instance model", "centered_instance"
        ),
    )


def test_explicit_sleap_executable_wins_over_sibling_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = tmp_path / "explicit" / "sleap-nn.exe"
    explicit.parent.mkdir()
    explicit.write_bytes(b"explicit")
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    sibling = python_dir / "sleap-nn.exe"
    sibling.write_bytes(b"sibling")
    path_match = tmp_path / "path" / "sleap-nn.exe"
    path_match.parent.mkdir()
    path_match.write_bytes(b"path")
    monkeypatch.setattr(runner_module.sys, "executable", str(python_dir / "python.exe"))
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: str(path_match))

    resolved = _resolve_sleap_executable(
        explicit_path=explicit,
        profile={"sleap_executable": str(path_match)},
    )

    assert resolved == explicit.resolve()


def test_profile_sleap_executable_is_used_when_cli_value_is_absent(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured" / "sleap-nn.exe"
    configured.parent.mkdir()
    configured.write_bytes(b"configured")

    resolved = _resolve_sleap_executable(
        explicit_path=None,
        profile={"sleap_executable": str(configured)},
    )

    assert resolved == configured.resolve()


def test_sibling_sleap_executable_is_selected_before_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    sibling = python_dir / "sleap-nn.exe"
    sibling.write_bytes(b"sibling")
    path_match = tmp_path / "path" / "sleap-nn.exe"
    path_match.parent.mkdir()
    path_match.write_bytes(b"path")
    monkeypatch.setattr(runner_module.sys, "executable", str(python_dir / "python.exe"))
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: str(path_match))

    resolved = _resolve_sleap_executable(explicit_path=None, profile={})

    assert resolved == sibling.resolve()


def test_path_sleap_executable_is_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    path_match = tmp_path / "path" / "sleap-nn.exe"
    path_match.parent.mkdir()
    path_match.write_bytes(b"path")
    monkeypatch.setattr(runner_module.sys, "executable", str(python_dir / "python.exe"))
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: str(path_match))

    resolved = _resolve_sleap_executable(explicit_path=None, profile={})

    assert resolved == path_match.resolve()


def test_missing_sleap_executable_fails_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module.sys,
        "executable",
        str(tmp_path / "python" / "python.exe"),
    )
    monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)

    with pytest.raises(PoseInferenceError, match="--sleap-executable"):
        _resolve_sleap_executable(explicit_path=None, profile={})


@pytest.mark.parametrize("reported_version", ["0.2.0", "0.4.0", "1.0.0"])
def test_incompatible_sleap_version_is_rejected(
    tmp_path: Path,
    reported_version: str,
) -> None:
    executable = tmp_path / "sleap-nn.exe"
    executable.write_bytes(b"runtime")

    def version_runner(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"sleap-nn {reported_version}\n",
            stderr="",
        )

    with pytest.raises(PoseInferenceError, match="requires SLEAP-NN 0.3.x"):
        _query_sleap_version(executable, version_runner=version_runner)


def test_sleap_version_03_is_accepted(tmp_path: Path) -> None:
    executable = tmp_path / "sleap-nn.exe"
    executable.write_bytes(b"runtime")

    def version_runner(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="sleap-nn 0.3.7\n",
            stderr="",
        )

    assert _query_sleap_version(executable, version_runner=version_runner) == "0.3.7"


def test_topdown_model_id_is_stable_bounded_and_always_hashed(tmp_path: Path) -> None:
    spec = PoseInferenceModelSpec(
        inference_mode="topdown",
        centroid_model_path=tmp_path / ("centroid_" + "a" * 100),
        centered_instance_model_path=tmp_path / ("instance_" + "b" * 100),
    )

    first = _model_bundle_id(spec)
    second = _model_bundle_id(spec)

    assert first == second
    assert first.startswith("topdown-centroid_")
    assert re.search(r"-[0-9a-f]{10}$", first)
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", first)
    assert len(first) <= 80


def test_topdown_model_id_hash_distinguishes_roles_and_parent_paths(
    tmp_path: Path,
) -> None:
    first = PoseInferenceModelSpec(
        inference_mode="topdown",
        centroid_model_path=tmp_path / "bundle_a" / "centroid",
        centered_instance_model_path=tmp_path / "bundle_a" / "instance",
    )
    same_names_different_parent = PoseInferenceModelSpec(
        inference_mode="topdown",
        centroid_model_path=tmp_path / "bundle_b" / "centroid",
        centered_instance_model_path=tmp_path / "bundle_b" / "instance",
    )
    swapped = PoseInferenceModelSpec(
        inference_mode="topdown",
        centroid_model_path=first.centered_instance_model_path,
        centered_instance_model_path=first.centroid_model_path,
    )

    first_id = _model_bundle_id(first)
    assert re.fullmatch(r"topdown-centroid-instance-[0-9a-f]{10}", first_id)
    assert first_id != _model_bundle_id(same_names_different_parent)
    assert first_id != _model_bundle_id(swapped)


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


def test_unreadable_s1_video_blocks_inference_submission(tmp_path: Path) -> None:
    preprocess = _write_s1_handoff(tmp_path)
    (preprocess / "prepared_video.mp4").write_bytes(b"not a video")
    called = False

    def unexpected_subprocess(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not be submitted")

    with pytest.raises(PoseInferenceError, match="prepared_video.mp4 is unreadable"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=_model_path(tmp_path),
                profile_path=DEFAULT_PROFILE,
            ),
            subprocess_runner=unexpected_subprocess,
        )

    assert called is False
    assert not (tmp_path / "pose_inference").exists()


def test_s1_frame_count_mismatch_blocks_inference_submission(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path, video_frame_count=3, sync_frame_count=4)
    called = False

    def unexpected_subprocess(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not be submitted")

    with pytest.raises(PoseInferenceError, match="frame count does not match"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=_model_path(tmp_path),
                profile_path=DEFAULT_PROFILE,
            ),
            subprocess_runner=unexpected_subprocess,
        )

    assert called is False
    assert not (tmp_path / "pose_inference").exists()


def test_s1_timing_array_mismatch_blocks_inference_submission(tmp_path: Path) -> None:
    preprocess = _write_s1_handoff(tmp_path)
    sync_path = preprocess / "prepared_sync.npz"
    with np.load(sync_path, allow_pickle=False) as archive:
        values = {key: np.array(archive[key], copy=True) for key in archive.files}
    values["prepared_time_sec"] = values["prepared_time_sec"][:-1]
    np.savez(sync_path, **values)
    called = False

    def unexpected_subprocess(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not be submitted")

    with pytest.raises(PoseInferenceError, match="prepared_sync.npz is unreadable or invalid"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=_model_path(tmp_path),
                profile_path=DEFAULT_PROFILE,
            ),
            subprocess_runner=unexpected_subprocess,
        )

    assert called is False
    assert not (tmp_path / "pose_inference").exists()


def test_generated_command_writes_to_pose_slp_and_keeps_tracking_in_same_call(
    tmp_path: Path,
    stable_sleap_runtime: SleapRuntime,
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

    assert command[:2] == (
        str(stable_sleap_runtime.executable_path),
        "predict",
    )
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
    assert command.count(str(stable_sleap_runtime.executable_path)) == 1
    assert not any(name in " ".join(command) for name in FORBIDDEN_ARTIFACT_NAMES)


def test_bottomup_command_tokens_remain_exact(
    tmp_path: Path,
    stable_sleap_runtime: SleapRuntime,
) -> None:
    _write_s1_handoff(tmp_path)
    handoff = validate_s1_handoff(tmp_path)
    model = _model_path(tmp_path)
    pose_slp = tmp_path / "pose.slp"

    command = build_sleap_predict_command(
        handoff=handoff,
        model_path=model,
        profile=load_inference_profile(DEFAULT_PROFILE),
        pose_slp_path=pose_slp,
    )

    assert command == (
        str(stable_sleap_runtime.executable_path),
        "predict",
        "--data_path",
        str(handoff.prepared_video.resolve()),
        "--model_paths",
        str(model.resolve()),
        "--output_path",
        str(pose_slp.resolve()),
        "--output_format",
        "slp",
        "--device",
        "cuda",
        "--batch_size",
        "4",
        "--max_instances",
        "4",
        "--peak_threshold",
        "0.05",
        "--integral_refinement",
        "integral",
        "--integral_patch_size",
        "5",
        "--max_edge_length_ratio",
        "0.25",
        "--dist_penalty_weight",
        "1.0",
        "--n_points",
        "10",
        "--min_line_scores",
        "0.25",
        "--tracking",
        "--candidates_method",
        "local_queues",
        "--max_tracks",
        "2",
        "--tracking_window_size",
        "240",
        "--min_new_track_points",
        "1",
        "--min_match_points",
        "1",
        "--track_matching_method",
        "hungarian",
        "--features",
        "keypoints",
        "--scoring_method",
        "oks",
        "--scoring_reduction",
        "robust_quantile",
        "--robust_best_instance",
        "0.95",
    )


def test_topdown_command_contains_both_role_paths_and_shared_settings(
    tmp_path: Path,
    stable_sleap_runtime: SleapRuntime,
) -> None:
    _write_s1_handoff(tmp_path)
    handoff = validate_s1_handoff(tmp_path)
    spec = _topdown_model_spec(tmp_path)

    command = build_sleap_predict_command(
        handoff=handoff,
        model_spec=spec,
        profile=load_inference_profile(TOPDOWN_PROFILE),
        pose_slp_path=tmp_path / "output with spaces" / "pose.slp",
    )

    model_flag_indices = [
        index for index, token in enumerate(command) if token == "--model_paths"
    ]
    assert command[0] == str(stable_sleap_runtime.executable_path)
    assert len(model_flag_indices) == 2
    assert command[model_flag_indices[0] + 1] == str(
        spec.centroid_model_path.resolve()
    )
    assert command[model_flag_indices[1] + 1] == str(
        spec.centered_instance_model_path.resolve()
    )
    assert "--peak_threshold" in command
    assert "--max_instances" in command
    assert "--tracking" in command
    assert "--max_tracks" in command
    assert "--max_edge_length_ratio" not in command


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


def test_legacy_profile_without_review_thresholds_uses_defaults(
    tmp_path: Path,
) -> None:
    profile = load_inference_profile(DEFAULT_PROFILE)
    for key in (
        "review_one_animal_fraction_threshold",
        "review_missing_keypoint_fraction_threshold",
        "review_frame_missing_keypoint_fraction_threshold",
    ):
        profile.pop(key)
    profile_path = tmp_path / "legacy_profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    options = _pose_qc_options(load_inference_profile(profile_path))

    assert options["review_one_animal_fraction_threshold"] == pytest.approx(0.9)
    assert options["review_missing_keypoint_fraction_threshold"] == pytest.approx(0.9)
    assert options["review_frame_missing_keypoint_fraction_threshold"] == pytest.approx(
        0.9
    )


def test_legacy_profile_without_inference_mode_remains_bottomup_compatible(
    tmp_path: Path,
) -> None:
    _write_s1_handoff(tmp_path)
    profile = load_inference_profile(DEFAULT_PROFILE)
    profile.pop("inference_mode")
    profile_path = tmp_path / "legacy_mode_profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=_model_path(tmp_path),
            profile_path=profile_path,
            dry_run=True,
            timestamp="20260716T020304",
        )
    )

    assert result.success is True
    assert result.status == "dry_run_complete"
    assert result.command.count("--model_paths") == 1


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
    assert pose_meta["overlay"]["status"] == "not_generated"
    assert pose_meta["pose_qc"]["status"] == "not_computed"
    assert pose_meta["sleap_provenance"]["status"] == "not_generated"
    assert pose_meta["sleap_provenance"]["inference_config"] is None
    assert manifest["dry_run"] is True
    assert manifest["overlay_status"] == "not_generated"
    assert manifest["sleap_provenance_status"] == "not_generated"
    assert manifest["effective_sleap_inference_config"] is None
    assert manifest["outputs"]["pose_slp"].endswith("pose.slp")
    assert settings["tracking_enabled"] is True
    assert "--output_path" in settings["command"]


def test_topdown_dry_run_records_bundle_and_does_not_launch_inference(
    tmp_path: Path,
    stable_sleap_runtime: SleapRuntime,
) -> None:
    _write_s1_handoff(tmp_path)
    spec = _topdown_model_spec(tmp_path)

    def unexpected_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run must not launch SLEAP-NN")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=None,
            model_spec=spec,
            profile_path=TOPDOWN_PROFILE,
            dry_run=True,
            timestamp="20260716T010203",
        ),
        subprocess_runner=unexpected_subprocess,
    )

    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    settings = yaml.safe_load(result.settings_used_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    for payload in (pose_meta, settings, manifest):
        assert payload["inference_mode"] == "topdown"
        assert payload["sleap_runtime"] == {
            "executable_path": str(stable_sleap_runtime.executable_path),
            "version": "0.3.0",
        }
        assert payload["model"]["centroid"]["model_path"] == str(
            spec.centroid_model_path.resolve()
        )
        assert payload["model"]["centered_instance"]["model_path"] == str(
            spec.centered_instance_model_path.resolve()
        )
        assert payload["model"]["centroid"]["model_id"]
        assert payload["model"]["centered_instance"]["model_id"]
    assert result.command.count("--model_paths") == 2
    assert result.run_id.startswith("topdown-")
    assert len(result.run_id.rsplit("__", maxsplit=1)[0]) <= 80
    assert "inference_mode: topdown" in log_text
    assert "centroid_model_path:" in log_text
    assert "centered_instance_model_path:" in log_text
    assert (
        f"sleap_executable_path: {stable_sleap_runtime.executable_path}"
        in log_text
    )
    assert "sleap_executable_version: 0.3.0" in log_text


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            PoseInferenceModelSpec(
                inference_mode="topdown",
                centered_instance_model_path=Path("centered"),
            ),
            "centroid_model_path",
        ),
        (
            PoseInferenceModelSpec(
                inference_mode="topdown",
                centroid_model_path=Path("centroid"),
            ),
            "centered_instance_model_path",
        ),
    ],
)
def test_incomplete_topdown_bundle_is_rejected_before_output_creation(
    tmp_path: Path,
    spec: PoseInferenceModelSpec,
    message: str,
) -> None:
    _write_s1_handoff(tmp_path)

    with pytest.raises(PoseInferenceError, match=message):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=None,
                model_spec=spec,
                profile_path=TOPDOWN_PROFILE,
                dry_run=True,
            )
        )

    assert not (tmp_path / "pose_inference").exists()


def test_topdown_same_component_path_is_rejected(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model = _role_model_path(tmp_path, "shared", "centroid")

    with pytest.raises(PoseInferenceError, match="must be distinct"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=None,
                model_spec=PoseInferenceModelSpec(
                    inference_mode="topdown",
                    centroid_model_path=model,
                    centered_instance_model_path=model,
                ),
                profile_path=TOPDOWN_PROFILE,
                dry_run=True,
            )
        )


def test_topdown_incompatible_model_role_is_rejected(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    centroid = _role_model_path(tmp_path, "not_named_by_role_a", "bottomup")
    centered = _role_model_path(
        tmp_path, "not_named_by_role_b", "centered_instance"
    )

    with pytest.raises(PoseInferenceError, match="identifies 'bottomup'"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=None,
                model_spec=PoseInferenceModelSpec(
                    inference_mode="topdown",
                    centroid_model_path=centroid,
                    centered_instance_model_path=centered,
                ),
                profile_path=TOPDOWN_PROFILE,
                dry_run=True,
            )
        )


def test_bottomup_rejects_model_identified_as_topdown_component(
    tmp_path: Path,
) -> None:
    _write_s1_handoff(tmp_path)
    centroid = _role_model_path(tmp_path, "centroid_only", "centroid")

    with pytest.raises(PoseInferenceError, match="incompatible 'centroid'"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=centroid,
                profile_path=DEFAULT_PROFILE,
                dry_run=True,
            )
        )


def test_topdown_unreconciled_model_role_is_rejected(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    unidentified = tmp_path / "unidentified"
    unidentified.mkdir()
    centered = _role_model_path(tmp_path, "instance", "centered_instance")

    with pytest.raises(PoseInferenceError, match="Could not identify"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=None,
                model_spec=PoseInferenceModelSpec(
                    inference_mode="topdown",
                    centroid_model_path=unidentified,
                    centered_instance_model_path=centered,
                ),
                profile_path=TOPDOWN_PROFILE,
                dry_run=True,
            )
        )


def test_profile_mode_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)

    with pytest.raises(PoseInferenceError, match="does not match"):
        run_pose_inference(
            PoseInferenceRequest(
                session_root=tmp_path,
                model_path=None,
                model_spec=_topdown_model_spec(tmp_path),
                profile_path=DEFAULT_PROFILE,
                dry_run=True,
            )
        )


def test_dry_run_preflight_reads_only_first_video_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_s1_handoff(tmp_path)

    class FirstFrameOnlyCapture:
        def __init__(self) -> None:
            self.read_calls = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _property: int) -> float:
            return 3.0

        def read(self):
            self.read_calls += 1
            if self.read_calls > 1:
                raise AssertionError("preflight must not sequentially decode the video")
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    capture = FirstFrameOnlyCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=_model_path(tmp_path),
            profile_path=DEFAULT_PROFILE,
            dry_run=True,
            timestamp="20260709T011203",
        )
    )

    assert result.success is True
    assert capture.read_calls == 1


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
    qc_calls: list[dict[str, Path]] = []
    overlay_calls: list[dict[str, Path]] = []
    provenance_calls: list[Path] = []

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
            timing={
                "status": "prepared_sync",
                "source": "prepared_sync",
                "timestamp_sources": ["prepared_sync:external_time_sec"],
                "represented_frames": 3,
                "frames_with_time": 3,
            },
        )

    def fake_pose_qc(**kwargs):
        qc_calls.append(dict(kwargs))
        return PoseQcSummary(
            status="computed",
            payload={
                "status": "computed",
                "schema_version": "pose_qc_v1",
                "outcome": "pass",
                "review_recommendation_reasons": [],
                "review_warning_count": 0,
                "flagged_intervals": {},
                "duplicate_candidate_risk": {"frames_flagged": 0},
                "implausible_geometry": {"frames_flagged": 0},
            },
        )

    def fake_overlay(**kwargs):
        overlay_calls.append(dict(kwargs))
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"overlay")
        return OverlaySummary(
            status="generated",
            path=output_path,
            frames_written=3,
            fps=20.0,
            width=64,
            height=64,
        )

    def fake_sleap_provenance(path: Path) -> SleapProvenanceSummary:
        provenance_calls.append(path)
        return SleapProvenanceSummary(
            status="available",
            inference_config={"peak_threshold": 0.2, "batch_size": 4},
            tracking_config={"tracking": True, "max_tracks": 2},
            device="cuda",
            system_info={"platform": "test"},
            model_paths=["model_a"],
            model_type="bottomup",
            source_file="prepared_video.mp4",
            frame_selection={"start": 0, "stop": 299},
            sleap_nn_version="0.3.0",
            sleap_io_version="0.2.0",
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
        pose_qc_computer=fake_pose_qc,
        overlay_generator=fake_overlay,
        sleap_provenance_reader=fake_sleap_provenance,
    )

    assert result.success is True
    assert result.status == "success"
    assert result.pose_slp_path.is_file()
    assert (result.run_dir / "pose.parquet").is_file()
    assert (result.run_dir / "overlay.mp4").is_file()
    assert export_calls == [
        {
            "pose_slp_path": result.pose_slp_path,
            "preprocess_dir": tmp_path / "preprocess",
            "output_path": result.run_dir / "pose.parquet",
        }
    ]
    assert qc_calls == [
        {
            "pose_parquet_path": result.run_dir / "pose.parquet",
            "expected_animals": 2,
            "review_one_animal_fraction_threshold": 0.9,
            "review_missing_keypoint_fraction_threshold": 0.9,
            "review_frame_missing_keypoint_fraction_threshold": 0.9,
        }
    ]
    assert provenance_calls == [result.pose_slp_path]
    assert overlay_calls == [
        {
            "prepared_video_path": tmp_path / "preprocess" / "prepared_video.mp4",
            "pose_parquet_path": result.run_dir / "pose.parquet",
            "output_path": result.run_dir / "overlay.mp4",
        }
    ]
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["artifact_status"]["pose_slp"] == "generated"
    assert pose_meta["artifact_status"]["pose_parquet"] == "generated"
    assert pose_meta["artifact_status"]["overlay_mp4"] == "generated"
    assert pose_meta["parquet"]["rows"] == 12
    assert pose_meta["parquet"]["timestamp_sources"] == ["prepared_sync:time_sec"]
    assert pose_meta["parquet"]["timing"]["status"] == "prepared_sync"
    assert pose_meta["parquet"]["timing"]["frames_with_time"] == 3
    assert pose_meta["pose_qc"]["status"] == "computed"
    assert pose_meta["pose_qc"]["outcome"] == "pass"
    assert pose_meta["overlay"]["status"] == "generated"
    assert pose_meta["overlay"]["frames_written"] == 3
    assert pose_meta["sleap_provenance"]["status"] == "available"
    assert pose_meta["sleap_provenance"]["source"] == "pose.slp labels.provenance"
    assert pose_meta["sleap_provenance"]["inference_config"] == {
        "peak_threshold": 0.2,
        "batch_size": 4,
    }
    assert pose_meta["sleap_provenance"]["tracking_config"] == {
        "tracking": True,
        "max_tracks": 2,
    }
    assert pose_meta["sleap_provenance"]["device"] == "cuda"
    assert pose_meta["sleap_provenance"]["sleap_nn_version"] == "0.3.0"
    assert pose_meta["sleap_provenance"]["sleap_io_version"] == "0.2.0"
    assert manifest["artifact_status"]["pose_parquet"] == "generated"
    assert manifest["artifact_status"]["overlay_mp4"] == "generated"
    assert manifest["qc_status"] == "computed"
    assert manifest["qc_outcome"] == "pass"
    assert manifest["qc_review_warning_count"] == 0
    assert manifest["overlay_status"] == "generated"
    assert manifest["sleap_provenance_status"] == "available"
    assert manifest["effective_sleap_inference_config"] == {
        "peak_threshold": 0.2,
        "batch_size": 4,
    }
    assert manifest["effective_sleap_tracking_config"] == {
        "tracking": True,
        "max_tracks": 2,
    }
    assert manifest["effective_sleap_provenance_source"] == "pose.slp labels.provenance"
    assert manifest["sleap_provenance_note"] == SLEAP_PROVENANCE_NOTE
    assert manifest["parquet_timing"]["status"] == "prepared_sync"
    assert manifest["parquet_timing"]["timestamp_sources"] == ["prepared_sync:external_time_sec"]
    assert "parquet_status: generated" in log_text
    assert "pose_qc: computed" in log_text
    assert "pose_qc_outcome: pass" in log_text
    assert "overlay: generated" in log_text
    assert "overlay_frames_written: 3" in log_text
    assert "sleap_provenance_status: available" in log_text
    assert SLEAP_PROVENANCE_NOTE in log_text


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
    assert pose_meta["artifact_status"]["overlay_mp4"] == "not_generated"
    assert pose_meta["parquet"]["status"] == "failed"
    assert pose_meta["pose_qc"]["outcome"] == "failed"
    assert "forced export failure" in pose_meta["parquet"]["error"]
    assert manifest["artifact_status"]["pose_parquet"] == "failed"
    assert manifest["qc_outcome"] == "failed"
    assert manifest["overlay_status"] == "not_generated"
    assert "parquet_status: failed" in log_text
    assert "parquet_error: forced export failure" in log_text
    assert "pose_qc_failure_reasons: parquet_export_failed" in log_text


def test_review_recommendation_does_not_mark_run_failed(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)

    def fake_subprocess(args, **kwargs):
        (Path(kwargs["cwd"]) / "pose.slp").write_bytes(b"slp")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    def fake_exporter(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"parquet")
        return ParquetExportSummary(status="generated", path=output_path, rows=4)

    def fake_pose_qc(**_kwargs):
        return PoseQcSummary(
            status="computed",
            payload={
                "status": "computed",
                "outcome": "review_recommended",
                "review_recommendation_reasons": ["one_detected_animal"],
                "review_warning_count": 1,
                "flagged_intervals": {
                    "one_detected_animal": [
                        {
                            "warning_type": "one_detected_animal",
                            "video_index": 0,
                            "start_frame": 0,
                            "end_frame": 2,
                            "frame_count": 3,
                            "time_span_sec": 0.1,
                        }
                    ]
                },
                "duplicate_candidate_risk": {"frames_flagged": 0},
                "implausible_geometry": {"frames_flagged": 0},
            },
        )

    def fake_overlay(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"overlay")
        return OverlaySummary(status="generated", path=output_path, frames_written=3)

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=_model_path(tmp_path),
            profile_path=DEFAULT_PROFILE,
            timestamp="20260709T051607",
        ),
        subprocess_runner=fake_subprocess,
        parquet_exporter=fake_exporter,
        pose_qc_computer=fake_pose_qc,
        overlay_generator=fake_overlay,
        sleap_provenance_reader=lambda _path: SleapProvenanceSummary(
            status="not_available"
        ),
    )

    assert result.success is True
    assert result.status == "success"
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["pose_qc"]["outcome"] == "review_recommended"
    assert manifest["qc_outcome"] == "review_recommended"
    assert manifest["qc_review_warning_count"] == 1
    assert manifest["qc_flagged_interval_count"] == 1
    assert "pose_qc_outcome: review_recommended" in log_text


def test_runner_records_sleap_provenance_failure_without_failing_run(
    tmp_path: Path,
) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)

    def fake_subprocess(args, **kwargs):
        run_dir = Path(kwargs["cwd"])
        (run_dir / "pose.slp").write_bytes(b"slp")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    def fake_exporter(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"parquet")
        return ParquetExportSummary(
            status="generated",
            path=output_path,
            rows=4,
            timestamp_sources=("prepared_sync:time_sec",),
        )

    def fake_pose_qc(**_kwargs):
        return PoseQcSummary(
            status="computed",
            payload={
                "status": "computed",
                "schema_version": "pose_qc_v1",
                "outcome": "pass",
                "duplicate_candidate_risk": {"frames_flagged": 0},
                "implausible_geometry": {"frames_flagged": 0},
            },
        )

    def fake_overlay(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"overlay")
        return OverlaySummary(status="generated", path=output_path, frames_written=1)

    def failing_sleap_provenance(_path: Path) -> SleapProvenanceSummary:
        raise RuntimeError("forced provenance failure")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=model_path,
            profile_path=DEFAULT_PROFILE,
            dry_run=False,
            timestamp="20260709T055607",
        ),
        subprocess_runner=fake_subprocess,
        parquet_exporter=fake_exporter,
        pose_qc_computer=fake_pose_qc,
        overlay_generator=fake_overlay,
        sleap_provenance_reader=failing_sleap_provenance,
    )

    assert result.success is True
    assert result.status == "success"
    assert result.pose_slp_path.is_file()
    assert (result.run_dir / "pose.parquet").is_file()
    assert (result.run_dir / "overlay.mp4").is_file()
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["sleap_provenance"]["status"] == "failed"
    assert "forced provenance failure" in pose_meta["sleap_provenance"]["error"]
    assert manifest["sleap_provenance_status"] == "failed"
    assert "forced provenance failure" in manifest["sleap_provenance_error"]
    assert manifest["effective_sleap_inference_config"] is None
    assert "sleap_provenance_status: failed" in log_text
    assert "sleap_provenance_error: forced provenance failure" in log_text


def test_runner_marks_qc_failed_when_pose_qc_fails(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)

    def fake_subprocess(args, **kwargs):
        run_dir = Path(kwargs["cwd"])
        (run_dir / "pose.slp").write_bytes(b"slp")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    def fake_exporter(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"parquet")
        return ParquetExportSummary(
            status="generated",
            path=output_path,
            rows=4,
            timestamp_sources=("prepared_sync:time_sec",),
        )

    def failing_pose_qc(**_kwargs):
        raise PoseQcError("forced QC failure")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=model_path,
            profile_path=DEFAULT_PROFILE,
            dry_run=False,
            timestamp="20260709T060708",
        ),
        subprocess_runner=fake_subprocess,
        parquet_exporter=fake_exporter,
        pose_qc_computer=failing_pose_qc,
    )

    assert result.success is False
    assert result.status == "qc_failed"
    assert result.pose_slp_path.is_file()
    assert (result.run_dir / "pose.parquet").is_file()
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["artifact_status"]["pose_parquet"] == "generated"
    assert pose_meta["artifact_status"]["overlay_mp4"] == "not_generated"
    assert pose_meta["parquet"]["status"] == "generated"
    assert pose_meta["pose_qc"]["status"] == "failed"
    assert pose_meta["pose_qc"]["outcome"] == "failed"
    assert "forced QC failure" in pose_meta["pose_qc"]["error"]
    assert manifest["qc_status"] == "failed"
    assert manifest["overlay_status"] == "not_generated"
    assert "pose_qc: failed" in log_text
    assert "pose_qc_error: forced QC failure" in log_text


def test_runner_marks_overlay_failed_when_overlay_generation_fails(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)

    def fake_subprocess(args, **kwargs):
        run_dir = Path(kwargs["cwd"])
        (run_dir / "pose.slp").write_bytes(b"slp")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    def fake_exporter(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"parquet")
        return ParquetExportSummary(
            status="generated",
            path=output_path,
            rows=4,
            timestamp_sources=("prepared_sync:time_sec",),
        )

    def fake_pose_qc(**_kwargs):
        return PoseQcSummary(
            status="computed",
            payload={
                "status": "computed",
                "schema_version": "pose_qc_v1",
                "outcome": "pass",
                "duplicate_candidate_risk": {"frames_flagged": 0},
                "implausible_geometry": {"frames_flagged": 0},
            },
        )

    def failing_overlay(**_kwargs):
        raise OverlayGenerationError("forced overlay failure")

    result = run_pose_inference(
        PoseInferenceRequest(
            session_root=tmp_path,
            model_path=model_path,
            profile_path=DEFAULT_PROFILE,
            dry_run=False,
            timestamp="20260709T070809",
        ),
        subprocess_runner=fake_subprocess,
        parquet_exporter=fake_exporter,
        pose_qc_computer=fake_pose_qc,
        overlay_generator=failing_overlay,
    )

    assert result.success is False
    assert result.status == "overlay_failed"
    assert result.pose_slp_path.is_file()
    assert (result.run_dir / "pose.parquet").is_file()
    assert not (result.run_dir / "overlay.mp4").exists()
    pose_meta = json.loads(result.pose_meta_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(result.job_manifest_path.read_text(encoding="utf-8"))
    log_text = result.processing_log_path.read_text(encoding="utf-8")
    assert pose_meta["artifact_status"]["pose_parquet"] == "generated"
    assert pose_meta["artifact_status"]["overlay_mp4"] == "failed"
    assert pose_meta["pose_qc"]["status"] == "computed"
    assert pose_meta["pose_qc"]["outcome"] == "failed"
    assert pose_meta["overlay"]["status"] == "failed"
    assert "forced overlay failure" in pose_meta["overlay"]["error"]
    assert manifest["qc_status"] == "computed"
    assert manifest["qc_outcome"] == "failed"
    assert manifest["overlay_status"] == "failed"
    assert "overlay: failed" in log_text
    assert "overlay_error: forced overlay failure" in log_text


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


def test_cli_explicit_bottomup_dry_run(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--session-root",
            str(tmp_path),
            "--inference-mode",
            "bottomup",
            "--model-path",
            str(model_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Status: dry_run_complete" in result.output


def test_cli_explicit_sleap_executable_is_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_s1_handoff(tmp_path)
    model_path = _model_path(tmp_path)
    explicit = tmp_path / "explicit" / "sleap-nn.exe"
    explicit.parent.mkdir()
    explicit.write_bytes(b"explicit")
    captured: dict[str, Path | None] = {}

    def capture_runtime(**kwargs: Any) -> SleapRuntime:
        captured["explicit_path"] = kwargs["explicit_path"]
        return SleapRuntime(explicit.resolve(), "0.3.2")

    monkeypatch.setattr(runner_module, "_resolve_sleap_runtime", capture_runtime)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--session-root",
            str(tmp_path),
            "--model-path",
            str(model_path),
            "--sleap-executable",
            str(explicit),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["explicit_path"] == explicit
    run_dir = next((tmp_path / "pose_inference").iterdir())
    settings = yaml.safe_load((run_dir / "settings_used.yaml").read_text())
    assert settings["command"][0] == str(explicit.resolve())


def test_cli_complete_topdown_pair_is_accepted(tmp_path: Path) -> None:
    _write_s1_handoff(tmp_path)
    spec = _topdown_model_spec(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--session-root",
            str(tmp_path),
            "--inference-mode",
            "topdown",
            "--centroid-model-path",
            str(spec.centroid_model_path),
            "--centered-instance-model-path",
            str(spec.centered_instance_model_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Status: dry_run_complete" in result.output
    run_dir = next((tmp_path / "pose_inference").iterdir())
    pose_meta = json.loads((run_dir / "pose_meta.json").read_text(encoding="utf-8"))
    assert pose_meta["inference_mode"] == "topdown"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--inference-mode", "topdown", "--centroid-model-path", "missing"],
        [
            "--inference-mode",
            "bottomup",
            "--model-path",
            "bottomup",
            "--centroid-model-path",
            "centroid",
        ],
        [
            "--inference-mode",
            "topdown",
            "--model-path",
            "bottomup",
            "--centroid-model-path",
            "centroid",
            "--centered-instance-model-path",
            "centered",
        ],
    ],
)
def test_cli_rejects_incomplete_or_ambiguous_model_flags(
    tmp_path: Path,
    extra_args: list[str],
) -> None:
    _write_s1_handoff(tmp_path)

    result = CliRunner().invoke(
        app,
        ["run", "--session-root", str(tmp_path), *extra_args, "--dry-run"],
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert not (tmp_path / "pose_inference").exists()
