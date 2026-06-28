"""Headless controller for project, video, trim, and pre-crop setup."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
from pydantic import ValidationError

from preprocess.config import (
    PreCropConfig,
    PreprocessConfig,
    TrimConfig,
    load_preprocess_config,
)
from preprocess.exceptions import PreprocessError, VideoProbeCancelledError
from preprocess.models import (
    OperationProgress,
    RawReadableCountProvenance,
    VideoProbeResult,
)
from preprocess.pre_crop import PreCropMode, ResolvedPreCrop, resolve_pre_crop
from preprocess.raw_probe_cache import get_raw_probe_cache_path
from preprocess.service import resolve_trim_range
from preprocess.video_probe import probe_video, read_raw_frame_at_index
from project.models import Project
from project.paths import PREPROCESS_DIRNAME, get_preprocess_dir
from project.service import ProjectService
from project.validation import ProjectError
from ui.project_hydration import ProjectHydrationResult, load_project_hydration
from ui.state import (
    PreprocessSetupState,
    RawReadableCountStatus,
    TimingMode,
    WorkflowStep,
)
from ui.tasks import (
    GuiTaskContext,
    GuiTaskCoordinator,
    GuiTaskKind,
    GuiTaskProgressEvent,
)

ConfigLoader = Callable[[Path], PreprocessConfig]
PreCropResolver = Callable[[PreCropConfig, tuple[int, int]], ResolvedPreCrop]
RawFrameReader = Callable[[Path, int], np.ndarray]

SUPPORTED_RAW_FRAME_JUMP_STEPS = (1, 10, 100, 1000)


def _same_video_source(first: Path, second: Path) -> bool:
    """Compare two source paths without requiring either path to exist."""

    return first.expanduser().resolve(strict=False) == second.expanduser().resolve(
        strict=False
    )


class SetupValidationError(ValueError):
    """Expected user-facing setup validation failure."""


@dataclass(frozen=True, slots=True)
class RawFrameReadRequest:
    """Immutable raw-frame read request captured for a worker thread."""

    raw_video_path: Path
    raw_decode_frame_idx: int


@dataclass(frozen=True, slots=True)
class RawFrameReadResult:
    """Ephemeral raw-frame read result; the frame is never stored in shared state."""

    raw_video_path: Path
    raw_decode_frame_idx: int
    frame: np.ndarray


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
        raw_frame_reader: RawFrameReader = read_raw_frame_at_index,
        task_coordinator: GuiTaskCoordinator | None = None,
    ) -> None:
        self.state = state or PreprocessSetupState()
        self._project_service = project_service or ProjectService()
        self._probe_function = probe_function
        self._config_loader = config_loader
        self._pre_crop_resolver = pre_crop_resolver
        self._raw_frame_reader = raw_frame_reader
        self.task_coordinator = task_coordinator or GuiTaskCoordinator()

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
        default_path = self.default_config_path().resolve(strict=False)
        if path is None or path.resolve(strict=False) == default_path:
            self.state.default_preprocess_config = PreprocessConfig.model_validate(
                config.model_dump(mode="python")
            )
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
        self.state.project_hydration_status = "new_project"
        self.state.project_hydration_message = (
            "New project created; no prior preprocessing metadata is expected."
        )
        return project

    def open_project(self, project_dir: Path) -> Project:
        """Open a validated project and reset downstream state."""

        try:
            project = self._project_service.open_project(project_dir)
        except (ProjectError, OSError, ValueError) as exc:
            self._validation_failure(f"Could not open project: {exc}", exc)
        self._set_project(project)
        self._apply_project_hydration(
            load_project_hydration(get_preprocess_dir(project))
        )
        return project

    def _set_project(self, project: Project) -> None:
        self.cancel_active_input_tasks()
        self.state.project = project
        self.state.project_dir = project.root_dir
        self.state.preprocess_dir = get_preprocess_dir(project)
        self.state.project_hydration_status = "not_checked"
        self.state.project_hydration_message = None
        self.state.prior_run_status = None
        self.state.raw_video_path = None
        self.state.raw_probe = None
        self.state.raw_readable_count_status = RawReadableCountStatus.NOT_COUNTED
        self.state.reset_raw_frame_navigation()
        config = self.state.preprocess_config
        self.state.start_frame = (
            (config.trim.start_frame or 0) if config is not None else None
        )
        self.state.end_frame_exclusive = (
            config.trim.end_frame if config is not None else None
        )
        self.state.pre_crop_config = config.pre_crop if config is not None else None
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Project changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None

    def _apply_project_hydration(self, result: ProjectHydrationResult) -> None:
        """Apply compact prior-run context without restoring crop acceptance."""

        self.state.project_hydration_status = result.status
        self.state.project_hydration_message = result.message
        self.state.prior_run_status = result.prior_run_status
        self.state.raw_video_path = result.raw_video_path
        self.state.raw_probe = result.raw_probe
        self.state.raw_readable_count_status = (
            RawReadableCountStatus.RECORDED_PRIOR_RUN
            if result.raw_count_recorded_by_prior_run
            else self._count_status_for_probe(result.raw_probe)
        )
        if result.start_frame is not None:
            self.state.start_frame = result.start_frame
        self.state.end_frame_exclusive = result.end_frame_exclusive
        if result.pre_crop_config is not None:
            self.state.pre_crop_config = result.pre_crop_config
        self.state.resolved_pre_crop = result.resolved_pre_crop
        self.state.trim_pre_crop_valid = (
            result.prior_run_status == "passed"
            and result.raw_probe is not None
            and result.start_frame is not None
            and result.pre_crop_config is not None
            and result.resolved_pre_crop is not None
        )
        config = self.state.preprocess_config
        if (
            config is not None
            and result.start_frame is not None
            and result.pre_crop_config is not None
        ):
            hydrated_config = PreprocessConfig.model_validate(
                {
                    **config.model_dump(mode="python"),
                    "trim": TrimConfig(
                        start_frame=result.start_frame,
                        end_frame=result.end_frame_exclusive,
                    ),
                    "pre_crop": result.pre_crop_config,
                }
            )
            self.state.store_preprocess_config(hydrated_config)
        self.state.candidate_crop_plan = None
        self.state.accepted_crop_plan = None
        self.state.crop_review_status = (
            "Prior run context loaded; crop review is required for a new run."
        )
        self.state.reset_raw_frame_navigation(
            0 if result.raw_probe is not None else None
        )
        self.state.last_validation_error = None

    def cancel_active_input_tasks(self) -> tuple[GuiTaskContext, ...]:
        """Cancel input-dependent work and invalidate all prior generations."""

        return self.task_coordinator.invalidate_inputs()

    def request_current_input_task_cancellation(
        self,
    ) -> tuple[GuiTaskContext, ...]:
        """Cancel current input-dependent work while retaining its generation."""

        return self.task_coordinator.request_input_cancellation()

    def cancel_all_tasks(self) -> tuple[GuiTaskContext, ...]:
        """Request cooperative cancellation during application shutdown."""

        return self.task_coordinator.request_all_cancellation()

    def begin_task(
        self,
        task_kind: GuiTaskKind,
        *,
        raw_video_path: Path | None = None,
    ) -> GuiTaskContext:
        """Create a task identity for the current project and raw video."""

        return self.task_coordinator.begin(
            task_kind=task_kind,
            project_dir=self.state.project_dir,
            raw_video_path=(
                raw_video_path
                if raw_video_path is not None
                else self.state.raw_video_path
            ),
        )

    def task_is_current(self, context: GuiTaskContext) -> bool:
        """Return whether a worker result still belongs to current inputs."""

        return self.task_coordinator.is_current(
            context,
            project_dir=self.state.project_dir,
            raw_video_path=self.state.raw_video_path,
        )

    def task_belongs_to_current_inputs(self, context: GuiTaskContext) -> bool:
        """Return whether a task generation/identity still matches current inputs."""

        return self.task_coordinator.belongs_to_current_inputs(
            context,
            project_dir=self.state.project_dir,
            raw_video_path=self.state.raw_video_path,
        )

    def task_progress_is_current(
        self,
        event: GuiTaskProgressEvent,
        context: GuiTaskContext,
    ) -> bool:
        """Return whether one task progress event may update current widgets."""

        return self.task_coordinator.progress_is_current(
            event,
            context,
            project_dir=self.state.project_dir,
            raw_video_path=self.state.raw_video_path,
        )

    def finish_task(self, context: GuiTaskContext) -> None:
        """Release a completed task identity."""

        self.task_coordinator.finish(context)

    def set_raw_video_path(self, path: Path) -> Path:
        """Store a selected raw-video path and invalidate downstream setup."""

        selected = Path(path).expanduser()
        self.cancel_active_input_tasks()
        self.state.raw_video_path = selected
        self.state.raw_probe = None
        self.state.raw_readable_count_status = RawReadableCountStatus.NOT_COUNTED
        self.state.reset_raw_frame_navigation()
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Raw video changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None
        return selected

    def clear_raw_video_path(self) -> None:
        """Clear an edited raw-video path and all dependent workflow state."""

        self.cancel_active_input_tasks()
        self.state.raw_video_path = None
        self.state.raw_probe = None
        self.state.raw_readable_count_status = RawReadableCountStatus.NOT_COUNTED
        self.state.reset_raw_frame_navigation()
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
        context: GuiTaskContext | None = None,
        progress_callback: Callable[[OperationProgress], None] | None = None,
        *,
        force_recount: bool = False,
    ) -> VideoProbeResult:
        """Run the core probe without mutating GUI state (worker-safe)."""

        arguments: dict[str, object] = {
            "require_sequential_count": require_sequential_count,
        }
        if context is not None:
            arguments["cancellation_requested"] = (
                context.is_cancellation_requested
            )
        if progress_callback is not None:
            arguments["progress_callback"] = progress_callback
        task_preprocess_dir = (
            context.project_dir / PREPROCESS_DIRNAME
            if context is not None and context.project_dir is not None
            else self.state.preprocess_dir
        )
        if self._probe_function is probe_video and task_preprocess_dir is not None:
            arguments["cache_path"] = get_raw_probe_cache_path(task_preprocess_dir)
            arguments["force_recount"] = force_recount
        return self._probe_function(path, **arguments)

    def apply_raw_video_probe(
        self,
        result: VideoProbeResult,
        *,
        context: GuiTaskContext | None = None,
    ) -> VideoProbeResult | None:
        """Apply a completed raw-video probe on the GUI thread."""

        if context is not None:
            if not self.task_belongs_to_current_inputs(context):
                return None
            if context.cancellation_requested:
                self.state.raw_readable_count_status = (
                    RawReadableCountStatus.COUNT_CANCELLED
                )
                self.state.last_validation_error = (
                    "Raw readable-frame count cancelled."
                )
                return None
        previous_probe = self.state.raw_probe
        previous_count_status = self.state.raw_readable_count_status
        count_was_preserved = False
        if (
            result.frame_count_opencv_readable is None
            and previous_probe is not None
            and previous_probe.frame_count_opencv_readable is not None
            and _same_video_source(previous_probe.source_path, result.source_path)
        ):
            count_was_preserved = True
            result = VideoProbeResult.model_validate(
                {
                    **result.model_dump(mode="python"),
                    "frame_count_opencv_readable": (
                        previous_probe.frame_count_opencv_readable
                    ),
                    "raw_readable_count_provenance": (
                        previous_probe.raw_readable_count_provenance
                    ),
                }
            )
        self.state.raw_video_path = result.source_path
        self.state.raw_probe = result
        if result.frame_count_opencv_readable is None:
            self.state.raw_readable_count_status = RawReadableCountStatus.NOT_COUNTED
        elif count_was_preserved:
            self.state.raw_readable_count_status = previous_count_status
        else:
            self.state.raw_readable_count_status = self._count_status_for_probe(result)
        self.state.reset_raw_frame_navigation(0)
        self.state.resolved_pre_crop = None
        self.state.trim_pre_crop_valid = False
        self.state.invalidate_crop_review("Raw video probe changed; crop review is required.")
        self._reset_timing_state()
        self.state.last_validation_error = None
        return result

    @staticmethod
    def _count_status_for_probe(
        probe: VideoProbeResult | None,
    ) -> RawReadableCountStatus:
        if probe is None or not probe.frame_count_opencv_readable:
            return RawReadableCountStatus.NOT_COUNTED
        if (
            probe.raw_readable_count_provenance
            is RawReadableCountProvenance.RECORDED_PRIOR_COMPLETED_RUN
        ):
            return RawReadableCountStatus.RECORDED_PRIOR_RUN
        if (
            probe.raw_readable_count_provenance
            is RawReadableCountProvenance.VALIDATED_REUSABLE_CACHE
        ):
            return RawReadableCountStatus.VALIDATED_REUSABLE_CACHE
        return RawReadableCountStatus.COUNTED_THIS_SESSION

    def record_raw_video_probe_failure(
        self,
        exc: BaseException,
        *,
        context: GuiTaskContext | None = None,
    ) -> str | None:
        """Record a worker-delivered raw probe failure and return concise text."""

        if context is not None and not self.task_belongs_to_current_inputs(context):
            return None
        if isinstance(exc, VideoProbeCancelledError):
            self.state.raw_readable_count_status = (
                RawReadableCountStatus.COUNT_CANCELLED
            )
            self.state.last_validation_error = "Raw readable-frame count cancelled."
            return self.state.last_validation_error
        recount_failed = (
            self.state.raw_readable_count_status
            is RawReadableCountStatus.RECOUNTING
            and self.state.raw_probe is not None
            and self.state.raw_probe.frame_count_opencv_readable is not None
            and self.state.raw_probe.frame_count_opencv_readable > 0
        )
        if recount_failed:
            self.state.raw_readable_count_status = (
                RawReadableCountStatus.RECOUNT_FAILED
            )
        else:
            self.state.raw_probe = None
            self.state.raw_readable_count_status = RawReadableCountStatus.NOT_COUNTED
            self.state.reset_raw_frame_navigation()
            self.state.resolved_pre_crop = None
            self.state.trim_pre_crop_valid = False
            self.state.invalidate_crop_review("Raw video probe failed.")
        if isinstance(exc, (PreprocessError, OSError, ValueError)):
            message = (
                "Readable raw-frame recount failed; the existing verified count "
                f"was retained: {exc}"
                if recount_failed
                else f"Could not probe raw video: {exc}"
            )
        else:
            self.record_unexpected_error(exc)
            message = "An unexpected error occurred while probing the raw video."
        self.state.last_validation_error = message
        return message

    def trusted_raw_frame_upper_bound(self) -> int | None:
        """Return a proven exclusive upper bound for raw-frame navigation."""

        probe = self.state.raw_probe
        if probe is None or probe.frame_count_opencv_readable is None:
            return None
        count = probe.frame_count_opencv_readable
        if count <= 0:
            return None
        if self.state.raw_readable_count_status in {
            RawReadableCountStatus.COUNTED_THIS_SESSION,
            RawReadableCountStatus.RECORDED_PRIOR_RUN,
            RawReadableCountStatus.VALIDATED_REUSABLE_CACHE,
        }:
            return count
        if probe.raw_readable_count_provenance is not None:
            return count
        return None

    def validate_raw_frame_index(self, raw_decode_frame_idx: int) -> int:
        """Validate one requested raw decode-frame index for navigation."""

        if self.state.raw_probe is None or self.state.raw_video_path is None:
            raise self._new_validation_error("Probe a raw video before frame navigation.")
        if (
            isinstance(raw_decode_frame_idx, bool)
            or not isinstance(raw_decode_frame_idx, int)
            or raw_decode_frame_idx < 0
        ):
            raise self._new_validation_error(
                "Raw frame index must be a non-negative integer."
            )
        upper_bound = self.trusted_raw_frame_upper_bound()
        if upper_bound is not None and raw_decode_frame_idx >= upper_bound:
            raise self._new_validation_error(
                "Raw frame index "
                f"{raw_decode_frame_idx:,} is outside the verified raw frame "
                f"range [0, {upper_bound:,})."
            )
        return raw_decode_frame_idx

    def first_raw_frame_index(self) -> int:
        """Return the first raw decode-frame index."""

        return self.validate_raw_frame_index(0)

    def previous_raw_frame_index(self) -> int:
        """Return the previous raw decode-frame index without silent clamping."""

        current = self._current_raw_frame_index_for_navigation()
        if current <= 0:
            raise self._new_validation_error("Already at the first raw frame.")
        return self.validate_raw_frame_index(current - 1)

    def next_raw_frame_index(self) -> int:
        """Return the next raw decode-frame index."""

        return self.validate_raw_frame_index(
            self._current_raw_frame_index_for_navigation() + 1
        )

    def raw_frame_index_after_step(self, *, direction: int, step: int) -> int:
        """Return the requested raw-frame index after one configured step."""

        if direction not in {-1, 1}:
            raise self._new_validation_error("Frame navigation direction must be -1 or 1.")
        if step not in SUPPORTED_RAW_FRAME_JUMP_STEPS:
            raise self._new_validation_error(
                "Raw frame jump step must be one of "
                f"{', '.join(str(value) for value in SUPPORTED_RAW_FRAME_JUMP_STEPS)}."
            )
        current = self._current_raw_frame_index_for_navigation()
        target = current + direction * step
        if target < 0:
            raise self._new_validation_error(
                "Raw frame jump would go before the first frame."
            )
        return self.validate_raw_frame_index(target)

    def set_raw_frame_navigation_step(self, step: int) -> int:
        """Store one supported visual-navigation jump step."""

        if step not in SUPPORTED_RAW_FRAME_JUMP_STEPS:
            raise self._new_validation_error(
                "Raw frame jump step must be one of "
                f"{', '.join(str(value) for value in SUPPORTED_RAW_FRAME_JUMP_STEPS)}."
            )
        self.state.raw_frame_navigation_step = step
        return step

    def begin_raw_frame_read(
        self,
        raw_decode_frame_idx: int,
    ) -> tuple[RawFrameReadRequest, GuiTaskContext]:
        """Capture a frame-read request and mark navigation busy."""

        index = self.validate_raw_frame_index(raw_decode_frame_idx)
        assert self.state.raw_video_path is not None
        context = self.begin_task(
            GuiTaskKind.RAW_FRAME_READ,
            raw_video_path=self.state.raw_video_path,
        )
        self.state.raw_frame_navigation_in_progress = True
        self.state.raw_frame_navigation_error = None
        return (
            RawFrameReadRequest(
                raw_video_path=self.state.raw_video_path,
                raw_decode_frame_idx=index,
            ),
            context,
        )

    def compute_raw_frame_read(
        self,
        request: RawFrameReadRequest,
        context: GuiTaskContext | None = None,
    ) -> RawFrameReadResult:
        """Read one raw frame without mutating shared GUI state."""

        if context is not None and context.is_cancellation_requested():
            raise ValueError("Raw frame read was cancelled.")
        frame = self._raw_frame_reader(
            request.raw_video_path,
            request.raw_decode_frame_idx,
        )
        array = np.asarray(frame)
        if array.ndim not in {2, 3} or array.size == 0:
            raise ValueError("Raw frame reader returned an invalid image array.")
        result_frame = np.array(array, copy=True)
        result_frame.setflags(write=False)
        return RawFrameReadResult(
            raw_video_path=request.raw_video_path,
            raw_decode_frame_idx=request.raw_decode_frame_idx,
            frame=result_frame,
        )

    def apply_raw_frame_read(
        self,
        result: RawFrameReadResult,
        *,
        context: GuiTaskContext | None = None,
    ) -> np.ndarray | None:
        """Apply a frame-read result only if it still belongs to current inputs."""

        if context is not None:
            if not self.task_belongs_to_current_inputs(context):
                return None
            if context.cancellation_requested:
                self.state.raw_frame_navigation_in_progress = False
                return None
        if self.state.raw_video_path is None or not _same_video_source(
            self.state.raw_video_path,
            result.raw_video_path,
        ):
            return None
        self.validate_raw_frame_index(result.raw_decode_frame_idx)
        probe = self.state.raw_probe
        if probe is not None and result.frame.shape[:2] != (probe.height, probe.width):
            raise self._new_validation_error(
                "Raw frame dimensions do not match the selected raw-video probe."
            )
        self.state.current_raw_frame_idx = result.raw_decode_frame_idx
        self.state.last_successfully_displayed_frame_idx = result.raw_decode_frame_idx
        self.state.raw_frame_navigation_error = None
        self.state.raw_frame_navigation_in_progress = False
        return np.array(result.frame, copy=True)

    def record_raw_frame_read_failure(
        self,
        exc: BaseException,
        *,
        context: GuiTaskContext | None = None,
        requested_frame_idx: int | None = None,
    ) -> str | None:
        """Record a failed frame read without changing the last valid frame index."""

        if context is not None:
            if not self.task_belongs_to_current_inputs(context):
                return None
            if context.cancellation_requested:
                self.state.raw_frame_navigation_in_progress = False
                return None
        prefix = (
            f"Could not load raw frame {requested_frame_idx:,}: "
            if requested_frame_idx is not None
            else "Could not load raw frame: "
        )
        message = prefix + str(exc)
        self.state.raw_frame_navigation_error = message
        self.state.raw_frame_navigation_in_progress = False
        self.state.last_validation_error = message
        return message

    def set_start_frame_to_current(self) -> ResolvedPreCrop:
        """Set inclusive start_frame to the currently displayed raw frame."""

        current = self._require_displayed_raw_frame_index()
        return self._configure_existing_pre_crop_with_trim(
            start_frame=current,
            end_frame_exclusive=self.state.end_frame_exclusive,
        )

    def set_end_frame_exclusive_to_current(self) -> ResolvedPreCrop:
        """Set exclusive end_frame to the currently displayed raw frame."""

        current = self._require_displayed_raw_frame_index()
        return self._configure_existing_pre_crop_with_trim(
            start_frame=self.state.start_frame or 0,
            end_frame_exclusive=current,
        )

    def set_end_frame_exclusive_to_current_plus_one(self) -> ResolvedPreCrop:
        """Set exclusive end_frame to one past the current raw frame."""

        current = self._require_displayed_raw_frame_index()
        return self._configure_existing_pre_crop_with_trim(
            start_frame=self.state.start_frame or 0,
            end_frame_exclusive=current + 1,
        )

    def clear_start_frame_selection(self) -> ResolvedPreCrop:
        """Clear direct start selection by restoring the default inclusive zero."""

        return self._configure_existing_pre_crop_with_trim(
            start_frame=0,
            end_frame_exclusive=self.state.end_frame_exclusive,
        )

    def clear_end_frame_selection(self) -> ResolvedPreCrop:
        """Clear the exclusive end-frame selection."""

        return self._configure_existing_pre_crop_with_trim(
            start_frame=self.state.start_frame or 0,
            end_frame_exclusive=None,
        )

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

        previous_trim_selection = (
            self.state.start_frame,
            self.state.end_frame_exclusive,
        )
        previous_pre_crop_selection = (
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
                "Pre-crop changed; crop review is required."
            )
            self._validation_failure(f"Invalid trim or pre-crop: {exc}", exc)
        pre_crop_selection = (pre_crop_config, resolved)
        pre_crop_changed = pre_crop_selection != previous_pre_crop_selection

        try:
            trim = resolve_trim_range(
                request_start_frame=start_frame,
                request_end_frame_exclusive=end_frame_exclusive,
                config=config,
                raw_readable_frame_count=raw_probe.frame_count_opencv_readable,
            )
        except (PreprocessError, ValidationError, ValueError) as exc:
            self.state.start_frame = start_frame
            self.state.end_frame_exclusive = end_frame_exclusive
            self.state.pre_crop_config = pre_crop_config
            self.state.resolved_pre_crop = resolved
            self.state.trim_pre_crop_valid = False
            self.state.invalidate_run_result("Configuration changed")
            if pre_crop_changed:
                self.state.invalidate_crop_review(
                    "Pre-crop changed; crop review is required."
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
        )
        trim_changed = current_selection != previous_trim_selection
        if pre_crop_changed:
            self.state.invalidate_crop_review("Pre-crop changed; crop review is required.")
        elif trim_changed:
            self.state.invalidate_run_result("Configuration changed")
        self.state.last_validation_error = None
        return resolved

    def _current_raw_frame_index_for_navigation(self) -> int:
        current = self.state.current_raw_frame_idx
        if current is None:
            current = self.state.last_successfully_displayed_frame_idx
        if current is None:
            current = 0
        return self.validate_raw_frame_index(current)

    def _require_displayed_raw_frame_index(self) -> int:
        current = self.state.last_successfully_displayed_frame_idx
        if current is None:
            current = self.state.current_raw_frame_idx
        if current is None:
            raise self._new_validation_error("Load a raw frame before setting trim.")
        return self.validate_raw_frame_index(current)

    def _configure_existing_pre_crop_with_trim(
        self,
        *,
        start_frame: int,
        end_frame_exclusive: int | None,
    ) -> ResolvedPreCrop:
        pre_crop = self.state.pre_crop_config
        mode = PreCropMode.NONE if pre_crop is None else PreCropMode(pre_crop.mode)
        rectangle = pre_crop.manual_rectangle if pre_crop is not None else None
        return self.configure_trim_and_pre_crop(
            start_frame=start_frame,
            end_frame_exclusive=end_frame_exclusive,
            mode=mode,
            boundary_px=pre_crop.boundary_px if pre_crop is not None else None,
            rectangle_x=rectangle[0] if rectangle is not None else None,
            rectangle_y=rectangle[1] if rectangle is not None else None,
            rectangle_width=rectangle[2] if rectangle is not None else None,
            rectangle_height=rectangle[3] if rectangle is not None else None,
        )

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
        self.state.timing_plausibility_assessment = None

    def _new_validation_error(self, message: str) -> SetupValidationError:
        self.state.last_validation_error = message
        return SetupValidationError(message)

    def _validation_failure(self, message: str, exc: BaseException) -> NoReturn:
        self.state.last_validation_error = message
        raise SetupValidationError(message) from exc
