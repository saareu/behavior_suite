from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from preprocess.config import CanonicalResolutionConfig, PrepareConfig, PreprocessConfig
from preprocess.crop_plan import CropPlan
from preprocess.ffmpeg_runtime import (
    ExecutableStatus,
    FFmpegCapabilityStatus,
    FFmpegRuntimePreflight,
)
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.models import (
    ExternalTimeSelection,
    OperationProgress,
    PreprocessExecutionContext,
    PreprocessOutputs,
    PreprocessRequest,
    PreprocessResult,
    TimingUnit,
    VideoProbeResult,
)
from preprocess.pre_crop import resolve_pre_crop
from preprocess.validation import PreparedVideoValidationResult
from project.models import Project
from ui.controllers.encode_settings_controller import EncodeSettingsController
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.run_preprocess_controller import (
    RunPreprocessController,
    RunPreprocessValidationError,
)
from ui.controllers.timing_controller import TimingController
from ui.pages.run_validate_page import RunValidatePage
from ui.state import PreprocessSetupState, TimingMode
from ui.tasks import (
    GuiTaskContext,
    GuiTaskProgressEvent,
    GuiTaskProgressTracker,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _config() -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=100,
                height=80,
            )
        )
    )


def _accepted_crop(config: PreprocessConfig) -> CropPlan:
    candidate = make_manual_crop_plan(
        raw_frame_shape=(80, 100),
        points_tl_tr_br_bl=np.asarray(
            ((10.0, 10.0), (90.0, 10.0), (90.0, 70.0), (10.0, 70.0))
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )
    return CropPlan.model_validate(
        {**candidate.model_dump(mode="python"), "accepted_by_user": True}
    )


def _probe(path: Path) -> VideoProbeResult:
    return VideoProbeResult(
        source_path=path,
        width=100,
        height=80,
        frame_count_opencv_reported=20,
        frame_count_opencv_readable=20,
        opencv_fps=10.0,
        raw_fps_effective=10.0,
        raw_fps_effective_method="opencv_fps",
        pts_status="not_extracted",
        ffprobe_succeeded=False,
    )


def _ready_state(tmp_path: Path) -> PreprocessSetupState:
    config = _config()
    raw_path = tmp_path / "raw.avi"
    preprocess_dir = tmp_path / "preprocess"
    preprocess_dir.mkdir(exist_ok=True)
    state = PreprocessSetupState(
        project=Project(root_dir=tmp_path, name="Project"),
        project_dir=tmp_path,
        preprocess_dir=preprocess_dir,
        raw_video_path=raw_path,
        raw_probe=_probe(raw_path),
        start_frame=2,
        end_frame_exclusive=18,
        pre_crop_config=config.pre_crop,
        resolved_pre_crop=resolve_pre_crop(config.pre_crop, (100, 80)),
        trim_pre_crop_valid=True,
    )
    state.set_loaded_preprocess_config(config)
    state.store_accepted_crop_plan(_accepted_crop(config))
    return state


def _selection(tmp_path: Path, units: TimingUnit) -> ExternalTimeSelection:
    temporal = units in {
        TimingUnit.SECONDS,
        TimingUnit.MILLISECONDS,
        TimingUnit.MICROSECONDS,
        TimingUnit.NANOSECONDS,
    }
    return ExternalTimeSelection(
        provided=True,
        source_path=tmp_path / "timing.mat",
        selected_variable="frame_times",
        declared_units=units,
        raw_vector_length=20,
        raw_video_frame_count_opencv_readable=20,
        is_numeric=True,
        is_one_dimensional=True,
        is_finite=True,
        is_monotonic_increasing=True,
        median_difference=0.1,
        estimated_fps=10.0 if temporal else None,
        validation_status="valid" if temporal else "not_convertible_to_seconds",
    )


def _success_result(state: PreprocessSetupState) -> PreprocessResult:
    assert state.preprocess_dir is not None
    outputs = PreprocessOutputs.from_preprocess_dir(state.preprocess_dir)
    validation = PreparedVideoValidationResult(
        is_valid=True,
        errors=(),
        opencv_reported_frame_count=16,
        opencv_readable_frame_count=16,
        opencv_reported_size_wh=(100, 80),
        opencv_reported_fps=10.0,
        expected_frame_count=16,
        expected_size_wh=(100, 80),
    )
    return PreprocessResult(
        success=True,
        outputs=outputs,
        prepared_validation=validation,
        message="ok",
        warnings=["example warning"],
        elapsed_sec=3.5,
        fps_header=10.0,
    )


def _supported_runtime(**_kwargs: Any) -> FFmpegRuntimePreflight:
    ffmpeg = ExecutableStatus(
        name="ffmpeg",
        path=Path(r"C:\conda\envs\behavior_suite_gui\Library\bin\ffmpeg.exe"),
        callable=True,
        banner="ffmpeg version 7.1.1",
        returncode=0,
    )
    ffprobe = ExecutableStatus(
        name="ffprobe",
        path=Path(r"C:\conda\envs\behavior_suite_gui\Library\bin\ffprobe.exe"),
        callable=True,
        banner="ffprobe version 7.1.1",
        returncode=0,
    )
    return FFmpegRuntimePreflight(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        capabilities=FFmpegCapabilityStatus(
            path=ffmpeg.path or Path("ffmpeg"),
            callable=True,
            fps_mode_supported=True,
            enc_time_base_demux_supported=True,
            returncode=0,
            modern_stage_timestamp_contract_supported=True,
            probe_command=("ffmpeg", "-enc_time_base", "demux", "-fps_mode", "passthrough"),
            probe_returncode=0,
        ),
        supported=True,
        errors=(),
    )


def _unsupported_runtime(**_kwargs: Any) -> FFmpegRuntimePreflight:
    ffmpeg = ExecutableStatus(
        name="ffmpeg",
        path=Path(r"C:\old\ffmpeg.exe"),
        callable=True,
        banner="ffmpeg version 4.3.1",
        returncode=0,
    )
    ffprobe = ExecutableStatus(
        name="ffprobe",
        path=Path(r"C:\old\ffprobe.exe"),
        callable=True,
        banner="ffprobe version 4.3.1",
        returncode=0,
    )
    return FFmpegRuntimePreflight(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        capabilities=FFmpegCapabilityStatus(
            path=ffmpeg.path or Path("ffmpeg"),
            callable=True,
            fps_mode_supported=False,
            enc_time_base_demux_supported=False,
            returncode=0,
            modern_stage_timestamp_contract_supported=False,
            probe_command=("ffmpeg", "-enc_time_base", "demux", "-fps_mode", "passthrough"),
            probe_returncode=1,
            probe_stderr="Unrecognized option 'fps_mode'.",
        ),
        supported=False,
        errors=("ffmpeg at C:\\old\\ffmpeg.exe does not support required option -fps_mode.",),
    )


class _FakeService:
    def __init__(self, result: PreprocessResult) -> None:
        self.result = result
        self.requests: list[PreprocessRequest] = []

    def run(
        self,
        request: PreprocessRequest,
        execution_context: PreprocessExecutionContext | None = None,
    ) -> PreprocessResult:
        self.requests.append(request)
        if execution_context is not None:
            execution_context.report(
                OperationProgress(
                    phase="Stage A: preparing video geometry",
                    message="Stage A running",
                    is_indeterminate=True,
                )
            )
        return self.result


class _ImmediateRunner:
    is_running = False

    def __init__(self) -> None:
        self.operations: list[Callable[..., Any]] = []

    def start_progressive(
        self,
        operation,
        *,
        task_name: str = "background task",
        context: GuiTaskContext,
        on_progress: Callable[[GuiTaskProgressEvent], None] | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        del task_name
        self.operations.append(operation)
        try:
            tracker = GuiTaskProgressTracker(context)
            value = operation(
                lambda progress: (
                    on_progress(tracker.build(progress))
                    if on_progress is not None
                    else None
                )
            )
        except BaseException as exc:
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(value)
        finally:
            if on_finished is not None:
                on_finished()


class _DeferredRunner:
    def __init__(self) -> None:
        self.is_running = False
        self.operation = None
        self.kwargs: dict[str, Any] = {}

    def start_progressive(self, operation, **kwargs: Any) -> None:
        self.is_running = True
        self.operation = operation
        self.kwargs = kwargs


def test_run_is_blocked_without_accepted_crop(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    state.accepted_crop_plan = None
    controller = RunPreprocessController(state)

    assert controller.can_run() is False
    with pytest.raises(RunPreprocessValidationError, match="accept"):
        controller.build_request()


def test_build_request_contains_current_gui_state(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    controller = RunPreprocessController(state)

    request = controller.build_request()

    assert request.project_dir == state.project_dir
    assert request.raw_video_path == state.raw_video_path
    assert isinstance(request.config, PreprocessConfig)
    assert request.start_frame == 2
    assert request.end_frame_exclusive == 18
    assert request.crop_plan is state.accepted_crop_plan
    assert request.crop_accepted_by_user is True
    assert request.automatic_crop_requested is False


def test_request_includes_valid_temporal_timing_fields(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    selection = _selection(tmp_path, TimingUnit.SECONDS)
    seconds = np.linspace(0.0, 1.9, 20)
    state.timing_mode = TimingMode.MATLAB
    state.timing_status = "valid"
    state.timing_valid = True
    state.external_time_selection = selection
    state.external_time_vector_seconds = seconds

    request = RunPreprocessController(state).build_request()

    assert request.external_time_selection == selection
    np.testing.assert_array_equal(request.external_time_vector_seconds, seconds)


@pytest.mark.parametrize("units", [TimingUnit.FRAMES, TimingUnit.UNKNOWN])
def test_request_includes_non_temporal_timing_without_seconds_vector(
    tmp_path: Path,
    units: TimingUnit,
) -> None:
    state = _ready_state(tmp_path)
    selection = _selection(tmp_path, units)
    state.timing_mode = TimingMode.MATLAB
    state.timing_status = "not_convertible_to_seconds"
    state.timing_valid = True
    state.external_time_selection = selection
    state.external_time_vector_seconds = None

    request = RunPreprocessController(state).build_request()

    assert request.external_time_selection == selection
    assert request.external_time_vector_seconds is None


def test_run_action_uses_task_runner_and_service(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    service = _FakeService(_success_result(state))
    runner = _ImmediateRunner()
    stages: list[str] = []
    controller = RunPreprocessController(
        state,
        service=service,
        ffmpeg_preflight=_supported_runtime,
    )

    request = controller.start_run(runner, on_stage=stages.append)

    assert len(runner.operations) == 1
    assert service.requests == [request]
    assert state.preprocess_result is service.result
    assert state.preprocess_task_running is False
    assert stages == [
        "Preparing preprocessing run",
        "Stage A: preparing video geometry",
    ]
    assert state.run_status == "Completed"


def test_duplicate_run_is_rejected_while_active(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    service = _FakeService(_success_result(state))
    runner = _DeferredRunner()
    controller = RunPreprocessController(
        state,
        service=service,
        ffmpeg_preflight=_supported_runtime,
    )
    controller.start_run(runner)

    with pytest.raises(RunPreprocessValidationError, match="already running"):
        controller.start_run(runner)

    assert service.requests == []


def test_run_start_blocks_after_failed_ffmpeg_preflight(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    service = _FakeService(_success_result(state))
    runner = _ImmediateRunner()
    controller = RunPreprocessController(
        state,
        service=service,
        ffmpeg_preflight=_unsupported_runtime,
    )

    with pytest.raises(RunPreprocessValidationError, match="-fps_mode"):
        controller.start_run(runner)

    assert runner.operations == []
    assert service.requests == []
    assert state.preprocess_task_running is False
    assert "-fps_mode" in (state.last_validation_error or "")


def test_success_summary_displays_all_six_official_outputs(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    result = _success_result(state)
    controller = RunPreprocessController(state)
    controller.apply_service_result(result)

    summary = controller.result_summary()

    assert summary.status == "SUCCESS"
    assert summary.artifacts_validated is True
    assert len(summary.official_outputs) == 6
    assert {path.name for _, path in summary.official_outputs} == {
        "prepared_video.mp4",
        "prepare_meta.json",
        "prepared_sync.npz",
        "cropped_background.png",
        "settings_used.yaml",
        "processing_log.txt",
    }
    assert summary.prepared_frame_count == 16
    assert summary.prepared_size_wh == (100, 80)
    assert summary.prepared_fps == 10.0
    assert summary.message == (
        "All official preprocessing artifacts were created and validated."
    )


def test_failure_summary_never_claims_success_or_valid_outputs(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    result = PreprocessResult(
        success=False,
        message="Output validation failed.",
        warnings=["invalid final frame count"],
        elapsed_sec=1.2,
    )
    controller = RunPreprocessController(state)
    controller.apply_service_result(result)

    summary = controller.result_summary()

    assert summary.status == "FAILED"
    assert "validation failed" in summary.message.lower()
    assert summary.artifacts_validated is False
    assert summary.official_outputs == ()


def test_cancelled_summary_is_distinct_from_failure(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    controller = RunPreprocessController(state)
    controller.apply_service_result(
        PreprocessResult(
            success=False,
            cancelled=True,
            message=(
                "Preprocessing cancelled. No incomplete outputs were marked as valid."
            ),
            elapsed_sec=1.5,
        )
    )

    summary = controller.result_summary()

    assert summary.status == "CANCELLED"
    assert summary.artifacts_validated is False
    assert summary.official_outputs == ()
    assert state.run_error_message is None


def test_stale_pipeline_progress_and_completion_are_ignored(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    service = _FakeService(_success_result(state))
    runner = _DeferredRunner()
    visible_progress: list[GuiTaskProgressEvent] = []
    controller = RunPreprocessController(
        state,
        service=service,
        ffmpeg_preflight=_supported_runtime,
    )
    controller.start_run(runner, on_progress=visible_progress.append)
    context = runner.kwargs["context"]
    assert isinstance(context, GuiTaskContext)
    tracker = GuiTaskProgressTracker(context)
    stale_event = tracker.build(
        OperationProgress(
            phase="Stage B: writing SLEAP-compatible video",
            message="1 / 16 frames written",
            completed_units=1,
            total_units=16,
            is_indeterminate=False,
        )
    )

    controller.task_coordinator.invalidate_inputs()
    state.raw_video_path = tmp_path / "replacement.avi"
    runner.kwargs["on_progress"](stale_event)
    runner.kwargs["on_success"](_success_result(state))

    assert context.cancellation_requested is True
    assert visible_progress == []
    assert state.preprocess_result is None


def test_run_page_shows_cancel_only_while_pipeline_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ui.controllers.run_preprocess_controller.preflight_ffmpeg_runtime",
        _supported_runtime,
    )
    state = _ready_state(tmp_path)
    page = RunValidatePage(PreprocessSetupController(state=state))

    page.refresh_from_state()
    assert page.cancel_button.isHidden() is True

    state.preprocess_task_running = True
    page.refresh_from_state()
    assert page.cancel_button.isHidden() is False
    assert page.cancel_button.isEnabled() is True

    state.preprocess_task_running = False
    page.refresh_from_state()
    assert page.cancel_button.isHidden() is True
    page.close()


def test_changing_timing_clears_result_but_preserves_crop(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    accepted = state.accepted_crop_plan
    state.preprocess_result = _success_result(state)

    TimingController(state).choose_matlab_timing()

    assert state.preprocess_result is None
    assert state.accepted_crop_plan is accepted


def test_non_geometry_encode_change_clears_result_but_preserves_crop(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    accepted = state.accepted_crop_plan
    state.preprocess_result = _success_result(state)

    EncodeSettingsController(state).update_settings(ffmpeg_crf=20)

    assert state.preprocess_result is None
    assert state.accepted_crop_plan is accepted


def test_geometry_encode_change_clears_crop_and_result(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    state.preprocess_result = _success_result(state)

    EncodeSettingsController(state).update_settings(canonical_height=100)

    assert state.preprocess_result is None
    assert state.accepted_crop_plan is None
    assert state.candidate_crop_plan is None


def test_open_preprocess_folder_uses_project_preprocess_path(tmp_path: Path) -> None:
    state = _ready_state(tmp_path)
    assert state.preprocess_dir is not None
    opened: list[Path] = []
    controller = RunPreprocessController(
        state,
        folder_opener=lambda path: opened.append(path) is None,
    )

    returned = controller.open_preprocess_folder()

    assert returned == state.preprocess_dir
    assert opened == [state.preprocess_dir]
