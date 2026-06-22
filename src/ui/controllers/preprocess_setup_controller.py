"""Headless controller for project, video, trim, and pre-crop setup."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

from pydantic import ValidationError

from preprocess.config import (
    PreCropConfig,
    PreprocessConfig,
    TrimConfig,
    load_preprocess_config,
)
from preprocess.exceptions import PreprocessError
from preprocess.models import VideoProbeResult
from preprocess.pre_crop import PreCropMode, ResolvedPreCrop, resolve_pre_crop
from preprocess.service import resolve_trim_range
from preprocess.video_probe import probe_video
from project.models import Project
from project.paths import get_preprocess_dir
from project.service import ProjectService
from project.validation import ProjectError
from ui.state import PreprocessSetupState, TimingMode, WorkflowStep

ConfigLoader = Callable[[Path], PreprocessConfig]
PreCropResolver = Callable[[PreCropConfig, tuple[int, int]], ResolvedPreCrop]


class SetupValidationError(ValueError):
    """Expected user-facing setup validation failure."""


class PreprocessSetupController:
    """Coordinate setup state through existing typed core APIs."""

    def __init__(
        self,
        state: PreprocessSetupState | None = None,
        *,
        project_service: ProjectService | None = None,
        probe_function: Callable[..., VideoProbeResult] = probe_video,
        config_loader: ConfigLoader = load_preprocess_config,
        pre_crop_resolver: PreCropResolver = resolve_pre_crop,
    ) -> None:
        self.state = state or PreprocessSetupState()
        self._project_service = project_service or ProjectService()
        self._probe_function = probe_function
        self._config_loader = config_loader
        self._pre_crop_resolver = pre_crop_resolver

    @staticmethod
    def default_config_path() -> Path:
        """Return the repository default preprocessing YAML path."""

        return Path(__file__).resolve().parents[3] / "configs" / "preprocess_default.yaml"

    def load_startup_config(self, path: Path | None = None) -> PreprocessConfig:
        """Load an explicit/default YAML, or use typed defaults if unavailable."""

        selected_path = Path(path).expanduser() if path is not None else self.default_config_path()
        if selected_path.is_file():
            return self.load_config(selected_path)
        if path is not None:
            raise SetupValidationError(f"Configuration file does not exist: {selected_path}")
        config = PreprocessConfig()
        self._apply_config(config, None)
        return config

    def load_config(self, path: Path) -> PreprocessConfig:
        """Load a validated YAML configuration and update setup defaults."""

        try:
            config = self._config_loader(Path(path))
        except (PreprocessError, ValidationError, OSError, ValueError) as exc:
            self._validation_failure(f"Invalid preprocess configuration: {exc}", exc)
        self._apply_config(config, Path(path).expanduser().resolve())
        return config

    def _apply_config(self, config: PreprocessConfig, path: Path | None) -> None:
        self.state.set_loaded_preprocess_config(config)
        self.state.config_path = path
        self.state.start_frame = config.trim.start_frame or 0
        self.state.end_frame_exclusive = config.trim.end_frame
        self.state.pre_crop_config = config.pre_crop
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Configuration changed; crop review is required.")
        self.state.last_validation_error = None

    def create_project(self, parent_dir: Path, project_name: str) -> Project:
        """Create a project through ProjectService and reset downstream state."""

        try:
            project = self._project_service.create_project(parent_dir, project_name)
        except (ProjectError, OSError, ValueError) as exc:
            self._validation_failure(f"Could not create project: {exc}", exc)
        self._set_project(project)
        return project

    def open_project(self, project_dir: Path) -> Project:
        """Open a validated project and reset downstream state."""

        try:
            project = self._project_service.open_project(project_dir)
        except (ProjectError, OSError, ValueError) as exc:
            self._validation_failure(f"Could not open project: {exc}", exc)
        self._set_project(project)
        return project

    def _set_project(self, project: Project) -> None:
        self.state.project = project
        self.state.project_dir = project.root_dir
        self.state.preprocess_dir = get_preprocess_dir(project)
        self.state.raw_video_path = None
        self.state.raw_probe = None
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Project changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None

    def set_raw_video_path(self, path: Path) -> Path:
        """Store a selected raw-video path and invalidate downstream setup."""

        selected = Path(path).expanduser()
        self.state.raw_video_path = selected
        self.state.raw_probe = None
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Raw video changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None
        return selected

    def clear_raw_video_path(self) -> None:
        """Clear an edited raw-video path and all dependent workflow state."""

        self.state.raw_video_path = None
        self.state.raw_probe = None
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Raw video changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None

    def probe_raw_video(
        self,
        path: Path | None = None,
        *,
        require_sequential_count: bool = False,
    ) -> VideoProbeResult:
        """Probe the selected raw video with optional sequential counting."""

        selected = self.prepare_raw_video_probe(path)
        try:
            result = self.compute_raw_video_probe(selected, require_sequential_count)
        except (PreprocessError, OSError, ValueError) as exc:
            self._validation_failure(f"Could not probe raw video: {exc}", exc)
        return self.apply_raw_video_probe(result)

    def prepare_raw_video_probe(self, path: Path | None = None) -> Path:
        """Validate a raw-video probe request without doing blocking work."""

        if self.state.project is None:
            raise self._new_validation_error("Select or create a project first.")
        selected = Path(path).expanduser() if path is not None else self.state.raw_video_path
        if selected is None or not str(selected):
            raise self._new_validation_error("Select a raw video before probing.")
        self.state.invalidate_crop_review("Raw video probe changed; crop review is required.")
        return selected

    def compute_raw_video_probe(
        self,
        path: Path,
        require_sequential_count: bool,
    ) -> VideoProbeResult:
        """Run the core probe without mutating GUI state (worker-safe)."""

        return self._probe_function(
            path,
            require_sequential_count=require_sequential_count,
        )

    def apply_raw_video_probe(self, result: VideoProbeResult) -> VideoProbeResult:
        """Apply a completed raw-video probe on the GUI thread."""

        self.state.raw_video_path = result.source_path
        self.state.raw_probe = result
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Raw video probe changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None
        return result

    def record_raw_video_probe_failure(self, exc: BaseException) -> str:
        """Record a worker-delivered raw probe failure and return concise text."""

        self.state.raw_probe = None
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Raw video probe failed.")
        if isinstance(exc, (PreprocessError, OSError, ValueError)):
            message = f"Could not probe raw video: {exc}"
        else:
            self.record_unexpected_error(exc)
            message = "An unexpected error occurred while probing the raw video."
        self.state.last_validation_error = message
        return message

    def configure_trim_and_pre_crop(
        self,
        *,
        start_frame: int,
        end_frame_exclusive: int | None,
        mode: str | PreCropMode,
        boundary_px: int | None = None,
        rectangle_x: int | None = None,
        rectangle_y: int | None = None,
        rectangle_width: int | None = None,
        rectangle_height: int | None = None,
    ) -> ResolvedPreCrop:
        """Validate and store trim selection plus one resolved pre-crop ROI."""

        previous_selection = (
            self.state.start_frame,
            self.state.end_frame_exclusive,
            self.state.pre_crop_config,
            self.state.resolved_pre_crop,
        )
        config = self.state.preprocess_config
        raw_probe = self.state.raw_probe
        if config is None:
            raise self._new_validation_error("Load a preprocess configuration first.")
        if raw_probe is None:
            raise self._new_validation_error("Probe a raw video first.")
        try:
            trim = resolve_trim_range(
                request_start_frame=start_frame,
                request_end_frame_exclusive=end_frame_exclusive,
                config=config,
                raw_readable_frame_count=raw_probe.frame_count_opencv_readable,
            )
            pre_crop_config = self._build_pre_crop_config(
                mode=mode,
                boundary_px=boundary_px,
                rectangle_x=rectangle_x,
                rectangle_y=rectangle_y,
                rectangle_width=rectangle_width,
                rectangle_height=rectangle_height,
            )
            resolved = self._pre_crop_resolver(
                pre_crop_config,
                (raw_probe.width, raw_probe.height),
            )
        except (PreprocessError, ValidationError, ValueError) as exc:
            self.state.start_frame = start_frame
            self.state.end_frame_exclusive = end_frame_exclusive
            self.state.resolved_pre_crop = None
            self.state.trim_pre_crop_valid = False
            self.state.invalidate_crop_review(
                "Trim or pre-crop changed; crop review is required."
            )
            self._validation_failure(f"Invalid trim or pre-crop: {exc}", exc)
        self.state.start_frame = trim.start_frame
        self.state.end_frame_exclusive = trim.end_frame_exclusive
        self.state.pre_crop_config = pre_crop_config
        self.state.resolved_pre_crop = resolved
        self.state.trim_pre_crop_valid = True
        effective_config = PreprocessConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "trim": TrimConfig(
                    start_frame=trim.start_frame,
                    end_frame=trim.end_frame_exclusive,
                ),
                "pre_crop": pre_crop_config,
            }
        )
        self.state.store_preprocess_config(effective_config)
        current_selection = (
            self.state.start_frame,
            self.state.end_frame_exclusive,
            self.state.pre_crop_config,
            self.state.resolved_pre_crop,
        )
        if current_selection != previous_selection:
            self.state.invalidate_crop_review(
                "Trim or pre-crop changed; crop review is required."
            )
        self.state.last_validation_error = None
        return resolved

    @staticmethod
    def _build_pre_crop_config(
        *,
        mode: str | PreCropMode,
        boundary_px: int | None,
        rectangle_x: int | None,
        rectangle_y: int | None,
        rectangle_width: int | None,
        rectangle_height: int | None,
    ) -> PreCropConfig:
        try:
            resolved_mode = PreCropMode(mode)
        except ValueError as exc:
            raise SetupValidationError(f"Unsupported pre-crop mode: {mode}") from exc
        rectangle = (
            rectangle_x,
            rectangle_y,
            rectangle_width,
            rectangle_height,
        )
        if resolved_mode is PreCropMode.NONE:
            return PreCropConfig(enabled=False, mode=resolved_mode.value)
        if resolved_mode is PreCropMode.MANUAL_RECTANGLE:
            if any(value is None for value in rectangle):
                raise SetupValidationError("Manual rectangle requires x, y, width, and height.")
            return PreCropConfig(
                enabled=True,
                mode=resolved_mode.value,
                manual_rectangle=cast(tuple[int, int, int, int], rectangle),
            )
        if boundary_px is None:
            raise SetupValidationError(f"{resolved_mode.value} requires a boundary coordinate.")
        return PreCropConfig(
            enabled=True,
            mode=resolved_mode.value,
            boundary_px=boundary_px,
        )

    def validate_step(self, step: WorkflowStep | int) -> None:
        """Raise when setup state cannot advance from the requested step."""

        resolved_step = WorkflowStep(step)
        if self.state.project is None or self.state.project_dir is None:
            raise self._new_validation_error("A valid project is required.")
        if resolved_step is WorkflowStep.PROJECT:
            return
        if self.state.raw_video_path is None or self.state.raw_probe is None:
            raise self._new_validation_error("A successfully probed raw video is required.")
        if resolved_step is WorkflowStep.RAW_VIDEO:
            return
        if (
            not self.state.trim_pre_crop_valid
            or self.state.pre_crop_config is None
            or self.state.resolved_pre_crop is None
        ):
            raise self._new_validation_error(
                "A valid trim and resolved pre-crop are required."
            )
        if resolved_step is WorkflowStep.TRIM_PRE_CROP:
            return
        if not self.state.timing_valid:
            raise self._new_validation_error(
                "Complete and validate the selected external timing option."
            )
        if self.state.timing_mode is TimingMode.NO_EXTERNAL:
            if (
                self.state.timing_status != "not_provided"
                or self.state.external_time_selection is not None
                or self.state.external_time_vector_seconds is not None
            ):
                raise self._new_validation_error(
                    "No-external-timing state is inconsistent."
                )
        elif not (
            self.state.external_time_selection is not None
            and self.state.timing_status in {"valid", "not_convertible_to_seconds"}
        ):
            raise self._new_validation_error(
                "Complete and validate the selected external timing option."
            )
        if resolved_step is WorkflowStep.TIMING:
            return
        accepted = self.state.accepted_crop_plan
        if accepted is None or not accepted.accepted_by_user:
            raise self._new_validation_error(
                "Review the crop preview and explicitly accept the crop."
            )
        if resolved_step is WorkflowStep.CROP_REVIEW:
            return
        if self.state.preprocess_config is None or not self.state.encode_settings_valid:
            raise self._new_validation_error(
                "A valid typed Encode Settings configuration is required."
            )
        if resolved_step in {
            WorkflowStep.ENCODE_SETTINGS,
            WorkflowStep.RUN_VALIDATE,
        }:
            return
        raise self._new_validation_error("Not implemented in this GUI milestone.")

    def can_advance(self, step: WorkflowStep | int) -> bool:
        """Return whether the requested workflow step is currently valid."""

        try:
            self.validate_step(step)
        except SetupValidationError:
            return False
        return True

    def record_unexpected_error(self, exc: BaseException) -> None:
        """Preserve detailed unexpected-error text for future logging."""

        self.state.unexpected_error_detail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    def _reset_timing_state(self) -> None:
        self.state.timing_mode = TimingMode.NO_EXTERNAL
        self.state.timing_status = "not_provided"
        self.state.timing_valid = True
        self.state.external_time_mat_path = None
        self.state.mat_workspace_format = None
        self.state.mat_candidates.clear()
        self.state.selected_timing_variable = None
        self.state.selected_timing_units = None
        self.state.external_time_selection = None
        self.state.external_time_vector_seconds = None

    def _new_validation_error(self, message: str) -> SetupValidationError:
        self.state.last_validation_error = message
        return SetupValidationError(message)

    def _validation_failure(self, message: str, exc: BaseException) -> NoReturn:
        self.state.last_validation_error = message
        raise SetupValidationError(message) from exc
