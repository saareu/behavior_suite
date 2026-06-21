from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from preprocess import service as service_module
from preprocess.background import BackgroundEstimateResult
from preprocess.config import (
    CanonicalResolutionConfig,
    DebugConfig,
    PrepareConfig,
    PreprocessConfig,
    TrimConfig,
)
from preprocess.crop_plan import CropMode
from preprocess.exceptions import PreprocessServiceError, VideoPreparationError
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.models import (
    ExternalTimeSelection,
    PreprocessRequest,
    SoftwareEnvironmentInfo,
    TimingUnit,
    VideoProbeResult,
)
from preprocess.service import PreprocessService, resolve_trim_range, select_output_fps
from preprocess.sync_writer import load_prepared_sync_npz
from preprocess.validation import PreparedVideoValidationResult
from project.service import ProjectService


def _project(tmp_path: Path) -> Path:
    return ProjectService().create_project(tmp_path, "ExampleProject").root_dir


def _config(
    *, debug: bool = False, start: int | None = None, end: int | None = None
) -> PreprocessConfig:
    return PreprocessConfig(
        trim=TrimConfig(start_frame=start, end_frame=end),
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(enabled=False, width=64, height=48)
        ),
        debug=DebugConfig(enabled=debug),
    )


def _plan(*, accepted: bool = True):
    plan = make_manual_crop_plan(
        raw_frame_shape=(48, 64),
        points_tl_tr_br_bl=np.array([[0.0, 0.0], [63.0, 0.0], [63.0, 47.0], [0.0, 47.0]]),
        pre_crop_roi=None,
        canonical_resolution=_config().prepare.canonical_resolution,
    )
    return plan.model_copy(update={"accepted_by_user": accepted})


def _probe(
    path: Path,
    *,
    readable: int | None = None,
    avg: str | None = "10/1",
    r: str | None = "12/1",
    opencv: float | None = 15.0,
) -> VideoProbeResult:
    return VideoProbeResult(
        source_path=path.resolve(),
        width=64,
        height=48,
        codec="mjpeg",
        pixel_format="yuvj420p",
        duration_sec=0.8,
        avg_frame_rate=avg,
        r_frame_rate=r,
        time_base="1/10",
        frame_count_ffprobe=8,
        frame_count_opencv_reported=8,
        frame_count_opencv_readable=readable,
        opencv_fps=opencv,
        raw_fps_effective=10.0 if avg == "10/1" else None,
        raw_fps_effective_method="ffprobe_avg_frame_rate" if avg == "10/1" else None,
        pts_status="not_extracted",
        ffprobe_succeeded=True,
    )


def _selection(units: TimingUnit, count: int = 8) -> ExternalTimeSelection:
    return ExternalTimeSelection(
        provided=True,
        source_path=Path("timing.mat"),
        selected_variable="ttl",
        declared_units=units,
        raw_vector_length=count,
        raw_video_frame_count_opencv_readable=count,
        is_numeric=True,
        is_one_dimensional=True,
        is_finite=True,
        is_monotonic_increasing=True,
        median_difference=0.1,
        estimated_fps=10.0 if units not in {TimingUnit.FRAMES, TimingUnit.UNKNOWN} else None,
        validation_status="valid",
    )


def _request(
    project_dir: Path,
    raw_path: Path,
    *,
    config: PreprocessConfig | None = None,
    plan: Any = None,
    **updates: Any,
) -> PreprocessRequest:
    values: dict[str, Any] = {
        "project_dir": project_dir,
        "raw_video_path": raw_path,
        "config": config or _config(),
        "start_frame": 0,
        "end_frame_exclusive": 4,
        "crop_plan": _plan() if plan is None else plan,
        "crop_accepted_by_user": True,
    }
    values.update(updates)
    return PreprocessRequest(**values)


def _install_probe(monkeypatch: pytest.MonkeyPatch, probe: VideoProbeResult) -> None:
    monkeypatch.setattr(service_module, "probe_video", lambda *_args, **_kwargs: probe)


def _install_fake_success_pipeline(
    monkeypatch: pytest.MonkeyPatch, validation: PreparedVideoValidationResult
) -> None:
    def fake_stage_a(**arguments: Any) -> SimpleNamespace:
        path = Path(arguments["intermediate_path"])
        path.write_bytes(b"stage-a")
        return SimpleNamespace(intermediate_path=path.resolve(), ffmpeg_command=[])

    def fake_stage_b(**arguments: Any) -> SimpleNamespace:
        Path(arguments["prepared_video_path"]).write_bytes(b"prepared")
        return SimpleNamespace(frames_written=validation.opencv_readable_frame_count)

    def fake_background(path: Path, **_arguments: Any) -> BackgroundEstimateResult:
        background = np.full((48, 64, 3), 40, dtype=np.uint8)
        background.setflags(write=False)
        return BackgroundEstimateResult(
            background=background,
            sampled_frame_indices=(0,),
            sampled_frame_count=1,
            source_video_path=Path(path).resolve(),
            source_size_wh=(64, 48),
            method="median",
            output_dtype="uint8",
        )

    monkeypatch.setattr(service_module, "run_ffmpeg_prepare", fake_stage_a)
    monkeypatch.setattr(service_module, "reencode_intermediate_with_opencv", fake_stage_b)
    monkeypatch.setattr(
        service_module, "validate_prepared_video", lambda *_args, **_kwargs: validation
    )
    monkeypatch.setattr(service_module, "estimate_prepared_background", fake_background)
    monkeypatch.setattr(
        service_module,
        "write_background_png",
        lambda _image, path: Path(path).write_bytes(b"png") or Path(path),
    )
    monkeypatch.setattr(
        service_module, "_software_environment", lambda **_kwargs: SoftwareEnvironmentInfo()
    )


def _validation(count: int = 4) -> PreparedVideoValidationResult:
    return PreparedVideoValidationResult(
        is_valid=True,
        errors=(),
        opencv_reported_frame_count=count,
        opencv_readable_frame_count=count,
        opencv_reported_size_wh=(64, 48),
        opencv_reported_fps=10.0,
        expected_frame_count=count,
        expected_size_wh=(64, 48),
    )


def test_service_rejects_missing_project(tmp_path: Path) -> None:
    request = PreprocessRequest(
        project_dir=tmp_path / "missing", raw_video_path=tmp_path / "raw.avi", config=_config()
    )
    result = PreprocessService().run(request)
    assert result.success is False
    assert "project validation" in result.message


def test_service_creates_and_reuses_internal_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))
    request = _request(project_dir, raw_path, plan=None, crop_plan=None)
    first = PreprocessService().run(request)
    second = PreprocessService().run(request)
    assert first.success is second.success is False
    assert (project_dir / "preprocess" / ".internal").is_dir()


def test_service_rejects_absent_crop_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))
    result = PreprocessService().run(_request(project_dir, raw_path, crop_plan=None))
    assert result.success is False
    assert "CropPlan is required" in result.message


@pytest.mark.parametrize("plan_accepted,request_accepted", [(False, True), (True, False)])
def test_service_rejects_unaccepted_crop_before_stage_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan_accepted: bool, request_accepted: bool
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))
    called = False

    def unexpected_stage_a(**_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(service_module, "run_ffmpeg_prepare", unexpected_stage_a)
    result = PreprocessService().run(
        _request(
            project_dir,
            raw_path,
            plan=_plan(accepted=plan_accepted),
            crop_accepted_by_user=request_accepted,
        )
    )
    assert result.success is False
    assert called is False
    assert not (project_dir / "preprocess" / "prepare_meta.json").exists()


def test_automatic_detection_is_never_auto_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))
    detected_plan = _plan(accepted=False).model_copy(update={"mode": CropMode.AUTOMATIC})
    monkeypatch.setattr(
        service_module,
        "detect_cage_crop_plan",
        lambda *_args: SimpleNamespace(
            crop_plan=detected_plan,
            detector_diagnostics={"detector": "fake"},
        ),
    )
    result = PreprocessService().run(
        _request(
            project_dir,
            raw_path,
            crop_plan=None,
            automatic_crop_requested=True,
        )
    )
    assert result.success is False
    assert result.crop_plan is not None
    assert result.crop_plan.accepted_by_user is False
    assert "accepted_by_user" in result.message


def test_resolve_trim_uses_request_then_config_then_defaults() -> None:
    config = _config(start=2, end=7)
    assert resolve_trim_range(
        request_start_frame=3, request_end_frame_exclusive=6, config=config
    ).model_dump() == {"start_frame": 3, "end_frame_exclusive": 6}
    assert resolve_trim_range(
        request_start_frame=None, request_end_frame_exclusive=None, config=config
    ).model_dump() == {"start_frame": 2, "end_frame_exclusive": 7}
    assert resolve_trim_range(
        request_start_frame=None, request_end_frame_exclusive=None, config=_config()
    ).model_dump() == {"start_frame": 0, "end_frame_exclusive": None}


@pytest.mark.parametrize(("start", "end"), [(-1, None), (3, 3), (4, 2)])
def test_resolve_trim_rejects_invalid_ranges(start: int, end: int | None) -> None:
    with pytest.raises(PreprocessServiceError):
        resolve_trim_range(
            request_start_frame=start, request_end_frame_exclusive=end, config=_config()
        )


@pytest.mark.parametrize(
    "vector",
    [
        None,
        np.array([0.0, 0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7]),
        np.array([0.0, 0.1, 0.2]),
        np.array([0.0, 0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]),
    ],
)
def test_service_rejects_invalid_temporal_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vector: np.ndarray | None
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path, readable=8))
    request = _request(
        project_dir,
        raw_path,
        external_time_selection=_selection(TimingUnit.SECONDS),
        external_time_vector_seconds=vector,
    )
    result = PreprocessService().run(request)
    assert result.success is False
    assert "external timing validation" in result.message


def test_fps_selection_prefers_external_and_records_raw_source(tmp_path: Path) -> None:
    fps = select_output_fps(_probe(tmp_path / "raw.avi"), external_fps_effective=20.0)
    assert fps.fps_header == pytest.approx(20.0)
    assert fps.fps_header_source == "external_time"
    assert fps.raw_fps_effective == pytest.approx(10.0)
    assert fps.external_fps_effective_method == "inverse_median_diff_seconds"


@pytest.mark.parametrize(
    ("avg", "r", "opencv", "expected", "source"),
    [
        ("10/1", "12/1", 15.0, 10.0, "ffprobe_avg_frame_rate"),
        ("0/0", "12/1", 15.0, 12.0, "ffprobe_r_frame_rate"),
        (None, None, 15.0, 15.0, "opencv_reported_fps"),
    ],
)
def test_fps_selection_falls_back_in_strict_order(
    tmp_path: Path,
    avg: str | None,
    r: str | None,
    opencv: float | None,
    expected: float,
    source: str,
) -> None:
    fps = select_output_fps(_probe(tmp_path / "raw.avi", avg=avg, r=r, opencv=opencv))
    assert fps.fps_header == pytest.approx(expected)
    assert fps.fps_header_source == source


def test_fps_selection_fails_without_valid_source(tmp_path: Path) -> None:
    with pytest.raises(PreprocessServiceError, match="No valid FPS source"):
        select_output_fps(_probe(tmp_path / "raw.avi", avg=None, r=None, opencv=None))


@pytest.mark.parametrize(("debug", "retained"), [(False, False), (True, True)])
def test_success_internal_cleanup_follows_debug_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, debug: bool, retained: bool
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))
    _install_fake_success_pipeline(monkeypatch, _validation())
    result = PreprocessService().run(_request(project_dir, raw_path, config=_config(debug=debug)))
    assert result.success is True
    assert bool(list((project_dir / "preprocess" / ".internal").glob("*.mp4"))) is retained


@pytest.mark.parametrize(("debug", "retained"), [(False, False), (True, True)])
def test_failure_internal_cleanup_follows_debug_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug: bool,
    retained: bool,
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))

    def fake_stage_a(**arguments: Any) -> SimpleNamespace:
        path = Path(arguments["intermediate_path"])
        path.write_bytes(b"stage-a")
        return SimpleNamespace(intermediate_path=path.resolve(), ffmpeg_command=[])

    def fail_stage_b(**_arguments: Any) -> None:
        raise VideoPreparationError("forced failure")

    monkeypatch.setattr(service_module, "run_ffmpeg_prepare", fake_stage_a)
    monkeypatch.setattr(service_module, "reencode_intermediate_with_opencv", fail_stage_b)
    result = PreprocessService().run(_request(project_dir, raw_path, config=_config(debug=debug)))
    assert result.success is False
    assert not (project_dir / "preprocess" / "prepare_meta.json").exists()
    assert bool(list((project_dir / "preprocess" / ".internal").glob("*.mp4"))) is retained


def test_no_ttl_success_marks_raw_pts_not_extracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path))
    _install_fake_success_pipeline(monkeypatch, _validation())
    result = PreprocessService().run(_request(project_dir, raw_path))
    assert result.success is True
    assert result.outputs is not None
    sync = load_prepared_sync_npz(result.outputs.prepared_sync_path)
    assert sync.external_time_status == "not_provided"
    assert sync.raw_pts_status == "not_extracted"
    assert np.all(np.isnan(sync.raw_pts_time_sec))


def test_open_ended_no_ttl_uses_completed_prepared_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path, readable=None))
    _install_fake_success_pipeline(monkeypatch, _validation())
    result = PreprocessService().run(_request(project_dir, raw_path, end_frame_exclusive=None))
    assert result.success is True
    assert result.outputs is not None
    sync = load_prepared_sync_npz(result.outputs.prepared_sync_path)
    assert sync.end_frame_exclusive is None
    assert sync.frame_count_used_for_sleap == 4


@pytest.mark.parametrize("units", [TimingUnit.FRAMES, TimingUnit.UNKNOWN])
def test_non_temporal_timing_is_not_convertible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, units: TimingUnit
) -> None:
    project_dir = _project(tmp_path)
    raw_path = tmp_path / "raw.avi"
    raw_path.write_bytes(b"raw")
    _install_probe(monkeypatch, _probe(raw_path, readable=8))
    _install_fake_success_pipeline(monkeypatch, _validation())
    request = _request(
        project_dir,
        raw_path,
        external_time_selection=_selection(units),
        external_time_vector_seconds=None,
    )
    result = PreprocessService().run(request)
    assert result.success is True
    assert result.outputs is not None
    sync = load_prepared_sync_npz(result.outputs.prepared_sync_path)
    assert sync.external_time_status == "not_convertible_to_seconds"
    assert np.all(np.isnan(sync.external_time_sec))
