"""Headless construction and background dispatch of one preprocess service run."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from preprocess.config import PreprocessConfig, TrimConfig
from preprocess.exceptions import PreprocessError
from preprocess.models import PreprocessOutputs, PreprocessRequest, PreprocessResult
from preprocess.service import PreprocessService
from project.validation import ProjectError, validate_project_root
from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)
from ui.state import PreprocessSetupState, TimingMode, WorkflowStep
from ui.tasks import (
    GuiTaskContext,
    GuiTaskCoordinator,
    GuiTaskKind,
    TaskAlreadyRunningError,
)


class RunPreprocessValidationError(ValueError):
    """Expected user-facing final-review or run-dispatch failure."""


class _ServiceProtocol(Protocol):
    def run(self, request: PreprocessRequest) -> PreprocessResult: ...


class _TaskRunnerProtocol(Protocol):
    @property
    def is_running(self) -> bool: ...

    def start(
        self,
        operation: Callable[[], Any],
        *,
        task_name: str = "background task",
        context: GuiTaskContext | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None: ...


FolderOpener = Callable[[Path], bool | None]


@dataclass(frozen=True, slots=True)
class FinalReviewSummary:
    """Concise final inputs and official destinations shown before a run."""

    project_directory: Path
    raw_video_path: Path
    trim_range: str
    pre_crop_summary: str
    timing_summary: str
    crop_mode: str
    crop_output_dimensions: tuple[int, int]
    crop_accepted: bool
    canonical_resolution: str
    official_outputs: tuple[tuple[str, Path], ...]


@dataclass(frozen=True, slots=True)
class RunResultSummary:
    """Display-only projection of the typed PreprocessResult."""

    status: str
    message: str
    official_outputs: tuple[tuple[str, Path], ...]
    prepared_frame_count: int | None
    prepared_size_wh: tuple[int, int] | None
    prepared_fps: float | None
    elapsed_sec: float
    warnings: tuple[str, ...]
    processing_log_path: Path | None
    artifacts_validated: bool


def _official_output_rows(outputs: PreprocessOutputs) -> tuple[tuple[str, Path], ...]:
    return (
        ("Prepared video", outputs.prepared_video_path),
        ("Prepare metadata", outputs.prepare_meta_path),
        ("Prepared synchronization", outputs.prepared_sync_path),
        ("Cropped background", outputs.cropped_background_path),
        ("Settings used", outputs.settings_used_path),
        ("Processing log", outputs.processing_log_path),
    )


def open_folder_in_platform_explorer(path: Path) -> bool:
    """Open a directory with the platform's normal file explorer."""

    folder = Path(path)
    if sys.platform == "win32":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
    return True


class RunPreprocessController:
    """Build one typed request and invoke only the existing PreprocessService."""

    def __init__(
        self,
        state: PreprocessSetupState,
        *,
        service: _ServiceProtocol | None = None,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        task_coordinator: GuiTaskCoordinator | None = None,
    ) -> None:
        self.state = state
        self._service = service or PreprocessService()
        self._folder_opener = folder_opener
        self.task_coordinator = task_coordinator or GuiTaskCoordinator()
        self._started_monotonic: float | None = None

    def validate_preconditions(self) -> None:
        """Require every GUI setup and crop-acceptance gate before dispatch."""

        if self.state.preprocess_task_running:
            raise self._validation_error("A preprocessing task is already running.")
        if not self._project_is_valid():
            raise self._validation_error("A valid project is required before preprocessing.")
        try:
            PreprocessSetupController(state=self.state).validate_step(
                WorkflowStep.ENCODE_SETTINGS
            )
        except SetupValidationError as exc:
            raise self._validation_error(str(exc)) from exc
        if self.state.preprocess_config is None or not self.state.encode_settings_valid:
            raise self._validation_error("Current encode settings are invalid.")
        raw_path = self.state.raw_video_path
        raw_probe = self.state.raw_probe
        if (
            raw_path is None
            or raw_probe is None
            or raw_path.resolve() != raw_probe.source_path.resolve()
        ):
            raise self._validation_error(
                "The selected raw video must have a matching successful probe."
            )

    def can_run(self) -> bool:
        """Return whether the Run button may be enabled without mutating state."""

        accepted = self.state.accepted_crop_plan
        timing_ready = (
            self.state.timing_mode is TimingMode.NO_EXTERNAL
            and self.state.timing_valid
            and self.state.timing_status == "not_provided"
            and self.state.external_time_selection is None
            and self.state.external_time_vector_seconds is None
        ) or (
            self.state.timing_mode is TimingMode.MATLAB
            and self.state.timing_valid
            and self.state.external_time_selection is not None
            and self.state.timing_status
            in {"valid", "not_convertible_to_seconds"}
        )
        return bool(
            not self.state.preprocess_task_running
            and self._project_is_valid()
            and self.state.raw_video_path is not None
            and self.state.raw_probe is not None
            and self.state.raw_video_path.resolve()
            == self.state.raw_probe.source_path.resolve()
            and self.state.trim_pre_crop_valid
            and self.state.pre_crop_config is not None
            and self.state.resolved_pre_crop is not None
            and timing_ready
            and accepted is not None
            and accepted.accepted_by_user
            and self.state.preprocess_config is not None
            and self.state.encode_settings_valid
        )

    def build_request(self) -> PreprocessRequest:
        """Build the exact typed PreprocessRequest represented by current GUI state."""

        self.validate_preconditions()
        project_dir = self.state.project_dir
        raw_video_path = self.state.raw_video_path
        config = self.state.preprocess_config
        pre_crop_config = self.state.pre_crop_config
        crop_plan = self.state.accepted_crop_plan
        assert project_dir is not None
        assert raw_video_path is not None
        assert config is not None
        assert pre_crop_config is not None
        assert crop_plan is not None
        try:
            effective_config = PreprocessConfig.model_validate(
                {
                    **config.model_dump(mode="python"),
                    "trim": TrimConfig(
                        start_frame=self.state.start_frame,
                        end_frame=self.state.end_frame_exclusive,
                    ),
                    "pre_crop": pre_crop_config,
                }
            )
            return PreprocessRequest(
                project_dir=project_dir,
                raw_video_path=raw_video_path,
                config=effective_config,
                start_frame=self.state.start_frame,
                end_frame_exclusive=self.state.end_frame_exclusive,
                external_time_selection=self.state.external_time_selection,
                external_time_vector_seconds=self.state.external_time_vector_seconds,
                crop_plan=crop_plan,
                automatic_crop_requested=False,
                crop_accepted_by_user=True,
                detector_diagnostics=self.state.detector_diagnostics,
            )
        except (ValidationError, ValueError) as exc:
            raise self._validation_error(
                f"Could not build a valid preprocess request: {exc}"
            ) from exc

    def build_final_review(self) -> FinalReviewSummary:
        """Project current GUI decisions into the pre-run review fields."""

        self.validate_preconditions()
        project_dir = self.state.project_dir
        raw_path = self.state.raw_video_path
        pre_crop = self.state.resolved_pre_crop
        crop = self.state.accepted_crop_plan
        config = self.state.preprocess_config
        preprocess_dir = self.state.preprocess_dir
        assert project_dir is not None
        assert raw_path is not None
        assert pre_crop is not None
        assert crop is not None
        assert config is not None
        assert preprocess_dir is not None
        roi = pre_crop.roi
        trim_end = (
            "end of video"
            if self.state.end_frame_exclusive is None
            else str(self.state.end_frame_exclusive)
        )
        if self.state.timing_mode is TimingMode.NO_EXTERNAL:
            timing_summary = "No external timing"
        else:
            units = self.state.selected_timing_units
            timing_summary = (
                f"MATLAB: {self.state.selected_timing_variable} "
                f"({units.value if units is not None else 'units not selected'}, "
                f"{self.state.timing_status})"
            )
        canonical = config.prepare.canonical_resolution
        canonical_summary = (
            f"Enabled — {canonical.width} × {canonical.height}"
            if canonical.enabled
            else "Disabled — native accepted crop dimensions"
        )
        outputs = PreprocessOutputs.from_preprocess_dir(preprocess_dir)
        return FinalReviewSummary(
            project_directory=project_dir,
            raw_video_path=raw_path,
            trim_range=f"[{self.state.start_frame or 0}, {trim_end})",
            pre_crop_summary=(
                f"{pre_crop.mode.value}: ({roi.x}, {roi.y}, {roi.width}, {roi.height})"
            ),
            timing_summary=timing_summary,
            crop_mode=crop.mode.value,
            crop_output_dimensions=crop.prepared_size_wh,
            crop_accepted=crop.accepted_by_user,
            canonical_resolution=canonical_summary,
            official_outputs=_official_output_rows(outputs),
        )

    def start_run(
        self,
        task_runner: _TaskRunnerProtocol,
        *,
        on_complete: Callable[[PreprocessResult], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> PreprocessRequest:
        """Dispatch PreprocessService.run through the reusable GUI task runner."""

        if self.state.preprocess_task_running or task_runner.is_running:
            raise self._validation_error("A preprocessing task is already running.")
        self._set_stage("Preparing request", on_stage)
        request = self.build_request()
        try:
            context = self.task_coordinator.begin(
                task_kind=GuiTaskKind.PREPROCESS_RUN,
                project_dir=self.state.project_dir,
                raw_video_path=self.state.raw_video_path,
            )
        except TaskAlreadyRunningError as exc:
            raise self._validation_error(str(exc)) from exc
        self.state.preprocess_result = None
        self.state.run_error_message = None
        self.state.preprocess_task_running = True
        self.state.run_started_at = datetime.now(UTC)
        self.state.run_elapsed_sec = 0.0
        self._started_monotonic = time.perf_counter()
        self._set_stage("Running preprocessing pipeline", on_stage)
        def handle_success(value: object) -> None:
            if not self.task_coordinator.is_current(
                context,
                project_dir=self.state.project_dir,
                raw_video_path=self.state.raw_video_path,
            ):
                return
            try:
                if not isinstance(value, PreprocessResult):
                    raise TypeError("Preprocess task returned an unexpected result type.")
                self._set_stage("Validating outputs", on_stage)
                self.apply_service_result(value, mark_task_finished=False)
            except BaseException as exc:
                self.record_task_failure(exc, mark_task_finished=False)
                if on_error is not None:
                    on_error(exc)
                return
            if on_complete is not None:
                on_complete(value)

        def handle_error(exc: BaseException) -> None:
            if not self.task_coordinator.belongs_to_current_inputs(
                context,
                project_dir=self.state.project_dir,
                raw_video_path=self.state.raw_video_path,
            ):
                return
            self.record_task_failure(exc, mark_task_finished=False)
            if on_error is not None:
                on_error(exc)

        def handle_finished() -> None:
            self.task_coordinator.finish(context)
            self._finish_task()
            self._set_stage("Finished", on_stage)
            if on_finished is not None:
                on_finished()

        try:
            task_runner.start(
                lambda: self.execute_request(request),
                task_name="preprocessing pipeline",
                context=context,
                on_success=handle_success,
                on_error=handle_error,
                on_finished=handle_finished,
            )
        except BaseException as exc:
            self.task_coordinator.finish(context)
            self.record_task_failure(exc)
            if isinstance(exc, TaskAlreadyRunningError):
                raise self._validation_error(str(exc)) from exc
            raise
        return request

    def execute_request(self, request: PreprocessRequest) -> PreprocessResult:
        """Call the existing service without mutating GUI state (worker-safe)."""

        return self._service.run(request)

    def apply_service_result(
        self,
        result: PreprocessResult,
        *,
        mark_task_finished: bool = True,
    ) -> None:
        """Store a typed service result after it returns to the main thread."""

        if result.success and (
            result.outputs is None or result.prepared_validation is None
        ):
            raise RunPreprocessValidationError(
                "Successful preprocessing result is missing outputs or validation details."
            )
        self.state.preprocess_result = result
        self.state.run_error_message = None if result.success else result.message
        self.state.run_elapsed_sec = result.elapsed_sec
        self.state.run_status = "Finished"
        if mark_task_finished:
            self.state.preprocess_task_running = False
            self._started_monotonic = None

    def record_task_failure(
        self,
        exc: BaseException,
        *,
        mark_task_finished: bool = True,
    ) -> None:
        """Record a worker exception as readable state while preserving diagnostics."""

        if isinstance(exc, (PreprocessError, OSError, ValueError)):
            message = f"Preprocessing failed: {exc}"
        else:
            message = "An unexpected error occurred while running preprocessing."
            self.state.unexpected_error_detail = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        self.state.preprocess_result = None
        self.state.run_error_message = message
        self.state.run_status = "Finished"
        self.update_elapsed_time()
        if mark_task_finished:
            self.state.preprocess_task_running = False
            self._started_monotonic = None

    def update_elapsed_time(self) -> float:
        """Refresh elapsed wall time for the page's coarse activity display."""

        if self._started_monotonic is not None:
            self.state.run_elapsed_sec = max(
                0.0,
                time.perf_counter() - self._started_monotonic,
            )
        return self.state.run_elapsed_sec

    def result_summary(self) -> RunResultSummary:
        """Return success/failure display values without revalidating artifacts."""

        result = self.state.preprocess_result
        if result is None:
            log_path = self._existing_processing_log_path()
            return RunResultSummary(
                status="FAILED" if self.state.run_error_message else "NOT RUN",
                message=self.state.run_error_message or "Preprocessing has not run.",
                official_outputs=(),
                prepared_frame_count=None,
                prepared_size_wh=None,
                prepared_fps=None,
                elapsed_sec=self.state.run_elapsed_sec,
                warnings=(),
                processing_log_path=log_path,
                artifacts_validated=False,
            )
        if not result.success:
            return RunResultSummary(
                status="FAILED",
                message=result.message,
                official_outputs=(),
                prepared_frame_count=None,
                prepared_size_wh=None,
                prepared_fps=None,
                elapsed_sec=result.elapsed_sec,
                warnings=tuple(result.warnings),
                processing_log_path=self._existing_processing_log_path(),
                artifacts_validated=False,
            )
        assert result.outputs is not None
        assert result.prepared_validation is not None
        validation = result.prepared_validation
        return RunResultSummary(
            status="SUCCESS",
            message=(
                "All official preprocessing artifacts were created and validated."
            ),
            official_outputs=_official_output_rows(result.outputs),
            prepared_frame_count=validation.opencv_readable_frame_count,
            prepared_size_wh=validation.opencv_reported_size_wh,
            prepared_fps=validation.opencv_reported_fps or result.fps_header,
            elapsed_sec=result.elapsed_sec,
            warnings=tuple(result.warnings),
            processing_log_path=result.outputs.processing_log_path,
            artifacts_validated=True,
        )

    def can_open_preprocess_folder(self) -> bool:
        """Return whether the current project preprocess directory exists."""

        return bool(
            self.state.preprocess_dir is not None
            and self.state.preprocess_dir.is_dir()
        )

    def open_preprocess_folder(self) -> Path:
        """Open the project preprocess folder through an injectable platform helper."""

        folder = self.state.preprocess_dir
        if folder is None or not folder.is_dir():
            raise self._validation_error(
                "Cannot open output folder because the preprocess directory does not exist."
            )
        try:
            opened = self._folder_opener(folder)
        except (OSError, subprocess.SubprocessError) as exc:
            raise self._validation_error(
                f"Cannot open output folder: {exc}"
            ) from exc
        if opened is False:
            raise self._validation_error("Cannot open output folder.")
        return folder

    def _existing_processing_log_path(self) -> Path | None:
        if self.state.preprocess_dir is None:
            return None
        path = self.state.preprocess_dir / "processing_log.txt"
        return path if path.is_file() else None

    def _project_is_valid(self) -> bool:
        project = self.state.project
        project_dir = self.state.project_dir
        preprocess_dir = self.state.preprocess_dir
        if project is None or project_dir is None or preprocess_dir is None:
            return False
        try:
            validated = validate_project_root(project_dir, require_preprocess_dir=True)
        except (ProjectError, OSError, ValueError):
            return False
        return bool(
            validated == project.root_dir.resolve()
            and preprocess_dir.resolve() == validated / "preprocess"
        )

    def _finish_task(self) -> None:
        self.update_elapsed_time()
        self.state.preprocess_task_running = False
        self._started_monotonic = None

    def _set_stage(
        self,
        stage: str,
        callback: Callable[[str], None] | None = None,
    ) -> None:
        self.state.run_status = stage
        if callback is not None:
            callback(stage)

    def _validation_error(self, message: str) -> RunPreprocessValidationError:
        self.state.last_validation_error = message
        return RunPreprocessValidationError(message)
