import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from cli import preprocess as cli_module
from preprocess.config import CanonicalResolutionConfig, PrepareConfig, PreprocessConfig
from preprocess.crop_plan import CropMode
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.models import (
    PreprocessOutputs,
    PreprocessResult,
    TimingUnit,
    VideoProbeResult,
)
from project.service import ProjectService

runner = CliRunner()


def _project(tmp_path: Path, name: str = "CliProject") -> Path:
    return ProjectService().create_project(tmp_path, name).root_dir


def _config() -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=False,
                width=64,
                height=48,
            )
        )
    )


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(_config().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _crop_plan(*, accepted: bool, mode: CropMode = CropMode.AUTOMATIC):
    plan = make_manual_crop_plan(
        raw_frame_shape=(48, 64),
        points_tl_tr_br_bl=np.array([[0.0, 0.0], [63.0, 0.0], [63.0, 47.0], [0.0, 47.0]]),
        pre_crop_roi=None,
        canonical_resolution=_config().prepare.canonical_resolution,
    )
    return plan.model_copy(
        update={
            "mode": mode,
            "fit_score": 0.93 if mode is CropMode.AUTOMATIC else None,
            "rim_density": 0.18 if mode is CropMode.AUTOMATIC else None,
            "accepted_by_user": accepted,
        }
    )


def _probe(raw_path: Path) -> VideoProbeResult:
    return VideoProbeResult(
        source_path=raw_path.resolve(),
        width=64,
        height=48,
        codec="mjpeg",
        pixel_format="yuvj420p",
        duration_sec=1.0,
        avg_frame_rate="10/1",
        r_frame_rate="10/1",
        time_base="1/10",
        frame_count_ffprobe=10,
        frame_count_opencv_reported=10,
        frame_count_opencv_readable=None,
        opencv_fps=10.0,
        raw_fps_effective=10.0,
        raw_fps_effective_method="ffprobe_avg_frame_rate",
        pts_status="not_extracted",
        ffprobe_succeeded=True,
    )


def _run_arguments(
    project_dir: Path,
    raw_path: Path,
    config_path: Path,
    crop_plan_path: Path,
) -> list[str]:
    return [
        "preprocess",
        "run",
        "--project-dir",
        str(project_dir),
        "--raw-video",
        str(raw_path),
        "--config",
        str(config_path),
        "--crop-plan",
        str(crop_plan_path),
    ]


def _success_result(project_dir: Path) -> PreprocessResult:
    outputs = PreprocessOutputs.from_preprocess_dir(project_dir / "preprocess")
    return PreprocessResult(
        success=True,
        outputs=outputs,
        message="complete",
        warnings=[],
        elapsed_sec=0.5,
        fps_header=10.0,
        fps_header_source="ffprobe_avg_frame_rate",
        raw_fps_effective=10.0,
        raw_fps_effective_method="ffprobe_avg_frame_rate",
    )


def _install_fake_service(
    monkeypatch: pytest.MonkeyPatch,
    result: PreprocessResult,
    captured: dict[str, Any],
) -> None:
    class FakeService:
        def run(self, request: Any) -> PreprocessResult:
            captured["request"] = request
            return result

    monkeypatch.setattr(cli_module, "PreprocessService", FakeService)


def test_top_level_help_works() -> None:
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0
    assert "preprocess" in result.output


def test_preprocess_help_works() -> None:
    result = runner.invoke(cli_module.app, ["preprocess", "--help"])
    assert result.exit_code == 0
    assert "detect-crop" in result.output
    assert "accept-crop" in result.output
    assert "run" in result.output


def test_detect_crop_writes_unaccepted_json_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    config_path = _config_path(tmp_path)
    output_path = project_dir / "preprocess" / "detected.json"
    detected_plan = _crop_plan(accepted=False)
    monkeypatch.setattr(
        cli_module,
        "probe_video",
        lambda *_args, **_kwargs: _probe(raw_path),
    )
    monkeypatch.setattr(
        cli_module,
        "detect_cage_crop_plan",
        lambda *_args: SimpleNamespace(
            crop_plan=detected_plan,
            detector_diagnostics={"sampled_frame_count": 3},
        ),
    )

    result = runner.invoke(
        cli_module.app,
        [
            "preprocess",
            "detect-crop",
            "--project-dir",
            str(project_dir),
            "--raw-video",
            str(raw_path),
            "--config",
            str(config_path),
            "--output-crop-plan",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.is_file()
    loaded_plan, diagnostics = cli_module.load_crop_plan_document(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded_plan.accepted_by_user is False
    assert loaded_plan.mode is CropMode.AUTOMATIC
    assert loaded_plan.native_size_wh == detected_plan.native_size_wh
    assert loaded_plan.canonical_geometry == detected_plan.canonical_geometry
    assert diagnostics == {"sampled_frame_count": 3}
    assert payload["native_geometry"] == {"size_wh": [64, 48]}
    assert "native_size_wh" not in payload
    assert isinstance(payload["H_raw_to_prepared_3x3"], list)
    assert isinstance(payload["quad_raw_tl_tr_br_bl"], list)
    assert "Status: DETECTED, NOT ACCEPTED" in result.output
    assert "Crop acceptance: REQUIRED" in result.output
    assert "Output size: 64 x 48" in result.output


def test_accept_crop_creates_separate_file_without_modifying_input(
    tmp_path: Path,
) -> None:
    project_dir = _project(tmp_path)
    source = project_dir / "preprocess" / "detected.json"
    destination = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(
        source,
        _crop_plan(accepted=False),
        detector_diagnostics={"sampled_frame_count": 4},
    )
    original_bytes = source.read_bytes()

    result = runner.invoke(
        cli_module.app,
        [
            "preprocess",
            "accept-crop",
            "--crop-plan",
            str(source),
            "--output-crop-plan",
            str(destination),
        ],
    )

    assert result.exit_code == 0, result.output
    assert source.read_bytes() == original_bytes
    assert destination.is_file()
    accepted, diagnostics = cli_module.load_crop_plan_document(destination)
    assert accepted.accepted_by_user is True
    assert diagnostics == {"sampled_frame_count": 4}
    assert "Status: ACCEPTED" in result.output
    assert "ready for preprocessing" in result.output


def test_crop_plan_writer_refuses_to_replace_existing_file(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    destination = project_dir / "preprocess" / "plan.json"
    cli_module.write_crop_plan_json(destination, _crop_plan(accepted=False))
    original_bytes = destination.read_bytes()

    with pytest.raises(cli_module.CliInputError, match="refusing to replace"):
        cli_module.write_crop_plan_json(destination, _crop_plan(accepted=True))

    assert destination.read_bytes() == original_bytes


def test_run_rejects_unaccepted_crop_without_calling_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "unaccepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=False))
    called = False

    class UnexpectedService:
        def run(self, _request: Any) -> None:
            nonlocal called
            called = True

    monkeypatch.setattr(cli_module, "PreprocessService", UnexpectedService)
    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path),
    )
    assert result.exit_code == 1
    assert called is False
    assert "not accepted" in result.output
    assert "Status: SUCCESS" not in result.output


def test_run_constructs_request_calls_service_and_uses_no_ttl_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=True))
    captured: dict[str, Any] = {}
    _install_fake_service(monkeypatch, _success_result(project_dir), captured)

    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path)
        + ["--start-frame", "2", "--end-frame", "7"],
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.project_dir == project_dir
    assert request.raw_video_path == raw_path
    assert request.start_frame == 2
    assert request.end_frame_exclusive == 7
    assert request.crop_plan.accepted_by_user is True
    assert request.crop_accepted_by_user is True
    assert request.external_time_selection is None
    assert request.external_time_vector_seconds is None


def test_run_success_prints_all_six_official_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=True))
    captured: dict[str, Any] = {}
    _install_fake_service(monkeypatch, _success_result(project_dir), captured)

    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path),
    )

    assert result.exit_code == 0
    assert "Status: SUCCESS" in result.output
    for artifact_name in (
        "prepared_video.mp4",
        "prepare_meta.json",
        "prepared_sync.npz",
        "cropped_background.png",
        "settings_used.yaml",
        "processing_log.txt",
    ):
        assert artifact_name in result.output


def test_run_failure_exits_one_and_never_prints_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=True))
    failure = PreprocessResult(
        success=False,
        message="Preprocessing failed during Stage A: forced failure",
        warnings=["diagnostic warning"],
        elapsed_sec=0.2,
    )
    captured: dict[str, Any] = {}
    _install_fake_service(monkeypatch, failure, captured)

    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path),
    )

    assert result.exit_code == 1
    assert "Status: FAILED" in result.output
    assert "Stage A" in result.output
    assert "diagnostic warning" in result.output
    assert "Status: SUCCESS" not in result.output


def test_run_rejects_incomplete_external_time_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=True))
    vector_path = tmp_path / "time.npy"
    np.save(vector_path, np.arange(5, dtype=np.float64))
    captured: dict[str, Any] = {}
    _install_fake_service(monkeypatch, _success_result(project_dir), captured)

    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path)
        + ["--external-time-npy", str(vector_path)],
    )

    assert result.exit_code == 1
    assert "requires --external-time-npy" in result.output
    assert "request" not in captured


def test_run_rejects_invalid_external_time_npy_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=True))
    vector_path = tmp_path / "time.npy"
    np.save(vector_path, np.ones((2, 3), dtype=np.float64))
    captured: dict[str, Any] = {}
    _install_fake_service(monkeypatch, _success_result(project_dir), captured)

    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path)
        + [
            "--external-time-npy",
            str(vector_path),
            "--external-time-source",
            str(tmp_path / "timing.mat"),
            "--external-time-variable",
            "frame_times",
            "--external-time-units",
            "seconds",
        ],
    )

    assert result.exit_code == 1
    assert "one-dimensional" in result.output
    assert "request" not in captured


def test_run_converts_declared_temporal_units_before_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    plan_path = project_dir / "preprocess" / "accepted.json"
    cli_module.write_crop_plan_json(plan_path, _crop_plan(accepted=True))
    vector_path = tmp_path / "time.npy"
    np.save(vector_path, np.array([0.0, 100.0, 200.0], dtype=np.float64))
    captured: dict[str, Any] = {}
    _install_fake_service(monkeypatch, _success_result(project_dir), captured)

    result = runner.invoke(
        cli_module.app,
        _run_arguments(project_dir, raw_path, config_path, plan_path)
        + [
            "--external-time-npy",
            str(vector_path),
            "--external-time-source",
            str(tmp_path / "timing.mat"),
            "--external-time-variable",
            "frame_times",
            "--external-time-units",
            "milliseconds",
        ],
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.external_time_selection.declared_units is TimingUnit.MILLISECONDS
    assert request.external_time_selection.raw_vector_length == 3
    np.testing.assert_allclose(
        request.external_time_vector_seconds,
        np.array([0.0, 0.1, 0.2]),
    )


def test_invalid_pre_crop_arguments_fail_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    config_path = _config_path(tmp_path)
    output_path = project_dir / "preprocess" / "detected.json"
    monkeypatch.setattr(
        cli_module,
        "probe_video",
        lambda *_args, **_kwargs: _probe(raw_path),
    )

    result = runner.invoke(
        cli_module.app,
        [
            "preprocess",
            "detect-crop",
            "--project-dir",
            str(project_dir),
            "--raw-video",
            str(raw_path),
            "--config",
            str(config_path),
            "--output-crop-plan",
            str(output_path),
            "--pre-crop-mode",
            "vertical_keep_left",
        ],
    )

    assert result.exit_code == 1
    assert "requires --pre-crop-boundary" in result.output
    assert not output_path.exists()


def test_invalid_crop_plan_json_fails_clearly(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    invalid_path = project_dir / "preprocess" / "invalid.json"
    invalid_path.write_text("{not-json", encoding="utf-8")
    output_path = project_dir / "preprocess" / "accepted.json"

    result = runner.invoke(
        cli_module.app,
        [
            "preprocess",
            "accept-crop",
            "--crop-plan",
            str(invalid_path),
            "--output-crop-plan",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "malformed" in result.output
    assert not output_path.exists()


def test_unexpected_internal_failure_uses_exit_code_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    config_path = _config_path(tmp_path)
    output_path = project_dir / "preprocess" / "detected.json"
    monkeypatch.setattr(
        cli_module,
        "load_preprocess_config",
        lambda _path: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    result = runner.invoke(
        cli_module.app,
        [
            "preprocess",
            "detect-crop",
            "--project-dir",
            str(project_dir),
            "--raw-video",
            str(raw_path),
            "--config",
            str(config_path),
            "--output-crop-plan",
            str(output_path),
        ],
    )

    assert result.exit_code == 2
    assert "Internal error: unexpected" in result.output
    assert "Status: SUCCESS" not in result.output
