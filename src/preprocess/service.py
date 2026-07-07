"""Typed orchestration for one complete preprocessing run."""

from __future__ import annotations

import json
import logging
import math
import platform
import shutil
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from preprocess.background import (
    estimate_prepared_background,
    validate_background,
    write_background_png,
)
from preprocess.cage_detection import detect_cage_crop_plan
from preprocess.config import PreprocessConfig
from preprocess.crop_plan import CropMode, CropPlan
from preprocess.exceptions import (
    PreprocessCancelledError,
    PreprocessError,
    PreprocessServiceError,
)
from preprocess.ffmpeg_runtime import (
    check_executable_version,
    ensure_supported_ffmpeg_runtime,
    resolve_ffprobe_binary,
)
from preprocess.logging_utils import create_processing_logger
from preprocess.metadata import (
    build_prepare_metadata,
    validate_prepare_metadata,
    write_prepare_meta_json,
    write_settings_used_yaml,
)
from preprocess.models import (
    ExternalTimeSelection,
    FPSHeaderSource,
    OperationProgress,
    PreprocessExecutionContext,
    PreprocessOutputs,
    PreprocessRequest,
    PreprocessResult,
    SoftwareEnvironmentInfo,
    TimingUnit,
    VideoProbeResult,
)
from preprocess.pre_crop import resolve_pre_crop
from preprocess.raw_probe_cache import get_raw_probe_cache_path
from preprocess.sync_writer import (
    PreparedSyncData,
    build_prepared_sync,
    load_prepared_sync_npz,
    validate_prepared_sync,
    write_prepared_sync_npz,
)
from preprocess.validation import PreparedVideoValidationResult, validate_prepared_video
from preprocess.video_prepare import (
    reencode_intermediate_with_opencv,
    run_ffmpeg_prepare,
)
from preprocess.video_probe import probe_video
from project.paths import get_preprocess_dir
from project.service import ProjectService
from project.validation import ProjectError

_TEMPORAL_UNITS = {
    TimingUnit.SECONDS,
    TimingUnit.MILLISECONDS,
    TimingUnit.MICROSECONDS,
    TimingUnit.NANOSECONDS,
}
_NON_TEMPORAL_UNITS = {TimingUnit.FRAMES, TimingUnit.UNKNOWN}
_EXTERNAL_FPS_METHOD = "inverse_median_diff_seconds"


class ResolvedTrim(BaseModel):
    """Resolved inclusive-start/exclusive-end decode-order selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    start_frame: int
    end_frame_exclusive: int | None

    @property
    def expected_frame_count(self) -> int | None:
        """Return the selected count when the exclusive end is known."""

        if self.end_frame_exclusive is None:
            return None
        return self.end_frame_exclusive - self.start_frame


class FPSSelection(BaseModel):
    """Output-header FPS and its raw/external provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fps_header: float
    fps_header_source: FPSHeaderSource
    raw_fps_effective: float | None
    raw_fps_effective_method: str | None
    external_fps_effective: float | None
    external_fps_effective_method: str | None


class _ExternalTimingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    status: str
    vector_seconds: np.ndarray | None
    units: TimingUnit
    source: str | None
    variable_name: str | None
    external_fps_effective: float | None
    external_fps_effective_method: str | None


def resolve_trim_range(
    *,
    request_start_frame: int | None,
    request_end_frame_exclusive: int | None,
    config: PreprocessConfig,
    raw_readable_frame_count: int | None = None,
) -> ResolvedTrim:
    """Resolve request/config/default trim precedence and validate the range.

    Raises:
        PreprocessServiceError: If the selected range or known raw bounds are invalid.
    """

    start = request_start_frame if request_start_frame is not None else config.trim.start_frame
    start = 0 if start is None else start
    end = (
        request_end_frame_exclusive
        if request_end_frame_exclusive is not None
        else config.trim.end_frame
    )
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise PreprocessServiceError("start_frame must be a non-negative integer.")
    if end is not None and (isinstance(end, bool) or not isinstance(end, int) or end <= start):
        raise PreprocessServiceError(
            "end_frame_exclusive must be an integer greater than start_frame."
        )
    if raw_readable_frame_count is not None:
        if raw_readable_frame_count <= 0:
            raise PreprocessServiceError(
                "Raw OpenCV-readable frame count must be positive when available."
            )
        if start >= raw_readable_frame_count:
            raise PreprocessServiceError("start_frame lies outside the raw readable frame range.")
        if end is not None and end > raw_readable_frame_count:
            raise PreprocessServiceError(
                "end_frame_exclusive lies outside the raw readable frame range."
            )
    return ResolvedTrim(start_frame=start, end_frame_exclusive=end)


def _rational_to_positive_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value)
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", maxsplit=1)
            numeric = float(numerator_text) / float(denominator_text)
        else:
            numeric = float(text)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _positive_float(value: float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def select_output_fps(
    raw_probe: VideoProbeResult, *, external_fps_effective: float | None = None
) -> FPSSelection:
    """Select the output FPS header using the required strict priority.

    Raises:
        PreprocessServiceError: If external FPS is invalid or no raw FPS source exists.
    """

    avg_fps = _rational_to_positive_float(raw_probe.avg_frame_rate)
    r_fps = _rational_to_positive_float(raw_probe.r_frame_rate)
    opencv_fps = _positive_float(raw_probe.opencv_fps)
    if avg_fps is not None:
        raw_fps, raw_method = avg_fps, "ffprobe_avg_frame_rate"
    elif r_fps is not None:
        raw_fps, raw_method = r_fps, "ffprobe_r_frame_rate"
    elif opencv_fps is not None:
        raw_fps, raw_method = opencv_fps, "opencv_reported_fps"
    else:
        raw_fps, raw_method = None, None
    if external_fps_effective is not None:
        external_fps = _positive_float(external_fps_effective)
        if external_fps is None:
            raise PreprocessServiceError("External effective FPS must be finite and positive.")
        return FPSSelection(
            fps_header=external_fps,
            fps_header_source="external_time",
            raw_fps_effective=raw_fps,
            raw_fps_effective_method=raw_method,
            external_fps_effective=external_fps,
            external_fps_effective_method=_EXTERNAL_FPS_METHOD,
        )
    if raw_fps is None or raw_method is None:
        raise PreprocessServiceError(
            "No valid FPS source exists in ffprobe avg_frame_rate, ffprobe r_frame_rate, or the raw OpenCV-reported FPS."
        )
    return FPSSelection(
        fps_header=raw_fps,
        fps_header_source=raw_method,
        raw_fps_effective=raw_fps,
        raw_fps_effective_method=raw_method,
        external_fps_effective=None,
        external_fps_effective_method=None,
    )


def _validate_selection_provenance(selection: ExternalTimeSelection, raw_count: int) -> TimingUnit:
    unit = selection.declared_units
    if unit is None:
        raise PreprocessServiceError("Provided external timing must declare its selected units.")
    if selection.validation_status != "valid" or not all(
        (
            selection.is_numeric,
            selection.is_one_dimensional,
            selection.is_finite,
            selection.is_monotonic_increasing,
        )
    ):
        raise PreprocessServiceError("External timing selection is not marked as valid.")
    if selection.raw_video_frame_count_opencv_readable != raw_count:
        raise PreprocessServiceError(
            "External timing selection raw-frame count differs from the current probe."
        )
    if selection.raw_vector_length != raw_count:
        raise PreprocessServiceError(
            "External timing vector length must equal the raw readable frame count."
        )
    return unit


def _validate_full_frame_plan_matches_probe(
    crop_plan: CropPlan,
    raw_probe: VideoProbeResult,
) -> None:
    if crop_plan.mode is not CropMode.FULL_FRAME:
        return
    expected_size = (raw_probe.width, raw_probe.height)
    roi = crop_plan.pre_crop_roi
    if (
        roi.x != 0
        or roi.y != 0
        or (roi.width, roi.height) != expected_size
        or crop_plan.native_size_wh != expected_size
    ):
        raise PreprocessServiceError(
            "Full-frame CropPlan must match the probed raw video dimensions."
        )


def _resolve_external_timing(
    selection: ExternalTimeSelection | None,
    vector_seconds: np.ndarray | None,
    raw_probe: VideoProbeResult,
) -> _ExternalTimingState:
    if selection is None or not selection.provided:
        if vector_seconds is not None:
            raise PreprocessServiceError(
                "external_time_vector_seconds requires an external time selection."
            )
        return _ExternalTimingState(
            status="not_provided",
            vector_seconds=None,
            units=TimingUnit.UNKNOWN,
            source=None,
            variable_name=None,
            external_fps_effective=None,
            external_fps_effective_method=None,
        )
    raw_count = raw_probe.frame_count_opencv_readable
    if raw_count is None:
        raise PreprocessServiceError(
            "Raw OpenCV-readable frame count is required with external timing."
        )
    unit = _validate_selection_provenance(selection, raw_count)
    source = str(selection.source_path) if selection.source_path is not None else None
    if unit in _NON_TEMPORAL_UNITS:
        if vector_seconds is not None:
            raise PreprocessServiceError(
                "Frames or unknown timing units must not supply a seconds vector."
            )
        return _ExternalTimingState(
            status="not_convertible_to_seconds",
            vector_seconds=None,
            units=unit,
            source=source,
            variable_name=selection.selected_variable,
            external_fps_effective=None,
            external_fps_effective_method=None,
        )
    if unit not in _TEMPORAL_UNITS:
        raise PreprocessServiceError(f"Unsupported external timing unit: {unit}")
    if vector_seconds is None:
        raise PreprocessServiceError(
            "Temporal external timing requires external_time_vector_seconds."
        )
    try:
        timing = np.asarray(vector_seconds)
    except (TypeError, ValueError) as exc:
        raise PreprocessServiceError(
            "External seconds timing must be a numeric one-dimensional array."
        ) from exc
    if (
        timing.ndim != 1
        or not np.issubdtype(timing.dtype, np.number)
        or np.issubdtype(timing.dtype, np.complexfloating)
        or np.issubdtype(timing.dtype, np.bool_)
    ):
        raise PreprocessServiceError(
            "External seconds timing must be a real numeric one-dimensional array."
        )
    timing = np.array(timing, dtype=np.float64, copy=True)
    if len(timing) != raw_count:
        raise PreprocessServiceError(
            "External seconds timing length must equal the raw readable frame count."
        )
    if not np.all(np.isfinite(timing)):
        raise PreprocessServiceError("External seconds timing contains non-finite values.")
    differences = np.diff(timing)
    if differences.size == 0 or not np.all(differences > 0):
        raise PreprocessServiceError(
            "External seconds timing must contain at least two strictly increasing values."
        )
    external_fps = 1.0 / float(np.median(differences))
    timing.setflags(write=False)
    return _ExternalTimingState(
        status="valid",
        vector_seconds=timing,
        units=unit,
        source=source,
        variable_name=selection.selected_variable,
        external_fps_effective=external_fps,
        external_fps_effective_method=_EXTERNAL_FPS_METHOD,
    )


def _build_sync(
    *,
    validation: PreparedVideoValidationResult,
    trim: ResolvedTrim,
    fps: FPSSelection,
    raw_probe: VideoProbeResult,
    timing: _ExternalTimingState,
) -> PreparedSyncData:
    sync = build_prepared_sync(
        prepared_frame_count=validation.opencv_readable_frame_count,
        start_frame=trim.start_frame,
        end_frame_exclusive=trim.end_frame_exclusive,
        fps_header=fps.fps_header,
        raw_fps_effective=fps.raw_fps_effective,
        raw_pts_time_sec=None,
        raw_pts_status="not_extracted",
        external_ttl_vector=timing.vector_seconds,
        external_time_status=timing.status,
        external_time_source=timing.source,
        external_time_variable_name=timing.variable_name,
        external_time_units=timing.units.value,
        raw_frame_count_opencv_readable=raw_probe.frame_count_opencv_readable,
        prepared_frame_count_opencv_reported=validation.opencv_reported_frame_count,
        prepared_frame_count_opencv_readable=validation.opencv_readable_frame_count,
    )
    if timing.status == "not_convertible_to_seconds":
        sync = sync.model_copy(
            update={
                "external_time_status": timing.status,
                "external_time_source": timing.source,
                "external_time_variable_name": timing.variable_name,
            }
        )
        validate_prepared_sync(sync)
    return sync


def _executable_version(executable: Path | None) -> str | None:
    if executable is None:
        return None
    status = check_executable_version(executable.name.split(".")[0], executable)
    return status.banner if status.callable else None


def _software_environment(
    *, ffmpeg_executable: Path | None, config: PreprocessConfig
) -> SoftwareEnvironmentInfo:
    try:
        application_version = version("behavior-suite")
    except PackageNotFoundError:
        application_version = None
    try:
        ffprobe_executable = resolve_ffprobe_binary(config.encoding.ffmpeg.ffprobe_path)
    except PreprocessError:
        ffprobe_executable = None
    return SoftwareEnvironmentInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
        ffmpeg_version=_executable_version(ffmpeg_executable),
        ffprobe_version=_executable_version(ffprobe_executable),
        application_version=application_version,
    )


def _close_logger(logger: logging.Logger | None) -> None:
    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _remove_file(path: Path, warnings: list[str]) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        warnings.append(f"Could not remove temporary file {path}: {exc}")


def _artifact_pairs(
    staged: PreprocessOutputs,
    official: PreprocessOutputs,
) -> tuple[tuple[Path, Path], ...]:
    return (
        (staged.prepared_video_path, official.prepared_video_path),
        (staged.prepare_meta_path, official.prepare_meta_path),
        (staged.prepared_sync_path, official.prepared_sync_path),
        (staged.cropped_background_path, official.cropped_background_path),
        (staged.settings_used_path, official.settings_used_path),
    )


def _commit_staged_outputs(
    staged: PreprocessOutputs,
    official: PreprocessOutputs,
) -> None:
    """Promote a validated five-artifact set with rollback on filesystem failure."""

    backup_dir = staged.preprocess_dir / ".previous"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    remove_backups = False
    try:
        for _staged_path, official_path in _artifact_pairs(staged, official):
            if official_path.exists():
                backup_path = backup_dir / official_path.name
                official_path.replace(backup_path)
                backups.append((backup_path, official_path))
        for staged_path, official_path in _artifact_pairs(staged, official):
            staged_path.replace(official_path)
            promoted.append(official_path)
        remove_backups = True
    except OSError as exc:
        rollback_errors: list[str] = []
        for promoted_path in promoted:
            try:
                promoted_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        for backup_path, official_path in backups:
            try:
                backup_path.replace(official_path)
            except OSError as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        remove_backups = not rollback_errors
        detail = f" Rollback errors: {rollback_errors}" if rollback_errors else ""
        raise PreprocessServiceError(
            f"Could not promote validated preprocessing artifacts: {exc}.{detail}"
        ) from exc
    finally:
        if remove_backups:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _remove_staging_dir(path: Path, warnings: list[str]) -> None:
    if (path / ".previous").exists():
        warnings.append(
            f"Pipeline staging retained for artifact rollback recovery: {path}"
        )
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        warnings.append(f"Could not remove pipeline staging directory {path}: {exc}")


class PreprocessService:
    """Coordinate existing preprocessing modules for one accepted crop run."""

    def run(
        self,
        request: PreprocessRequest,
        execution_context: PreprocessExecutionContext | None = None,
    ) -> PreprocessResult:
        """Execute one preprocess run and return success or an expected failure.

        Scientific operations remain delegated to their existing modules. A
        successful result is returned only after every official artifact has
        been written, reloaded where applicable, and validated.
        """

        execution = execution_context or PreprocessExecutionContext()
        started = time.perf_counter()
        warnings: list[str] = []
        logger: logging.Logger | None = None
        outputs: PreprocessOutputs | None = None
        staged_outputs: PreprocessOutputs | None = None
        staging_dir: Path | None = None
        internal_path: Path | None = None
        raw_probe: VideoProbeResult | None = None
        prepared_validation: PreparedVideoValidationResult | None = None
        crop_plan = request.crop_plan
        fps: FPSSelection | None = None
        current_stage = "project validation"
        succeeded = False

        def report(
            phase: str,
            message: str | None = None,
        ) -> None:
            execution.report(
                OperationProgress(
                    phase=phase,
                    message=message or phase,
                    completed_units=None,
                    total_units=None,
                    is_indeterminate=True,
                )
            )

        def raise_if_cancelled() -> None:
            if execution.is_cancellation_requested():
                raise PreprocessCancelledError(
                    f"Preprocessing cancelled during {current_stage}."
                )

        try:
            report("Preparing preprocessing run")
            raise_if_cancelled()
            project = ProjectService().open_project(request.project_dir)

            current_stage = "configuration validation"
            try:
                config = PreprocessConfig.model_validate(request.config.model_dump(mode="python"))
            except ValidationError as exc:
                raise PreprocessServiceError(f"Preprocess configuration is invalid: {exc}") from exc

            current_stage = "FFmpeg runtime preflight"
            ensure_supported_ffmpeg_runtime(
                ffmpeg_path=config.encoding.ffmpeg.ffmpeg_path,
                ffprobe_path=config.encoding.ffmpeg.ffprobe_path,
            )

            outputs = PreprocessOutputs.from_preprocess_dir(get_preprocess_dir(project))
            internal_dir = outputs.preprocess_dir / ".internal"
            internal_dir.mkdir(parents=True, exist_ok=True)
            staging_dir = internal_dir / f"pipeline_{uuid.uuid4().hex}"
            staging_dir.mkdir(parents=True, exist_ok=False)
            staged_outputs = PreprocessOutputs.from_preprocess_dir(staging_dir)
            logger = create_processing_logger(outputs.processing_log_path)
            logger.info("run started project=%s", project.name)
            logger.info("stage=config_validation")

            current_stage = "raw video probe"
            logger.info("stage=raw_video_probe")
            requires_raw_count = bool(
                request.external_time_selection is not None
                and request.external_time_selection.provided
            )
            raw_probe = probe_video(
                request.raw_video_path,
                require_sequential_count=requires_raw_count,
                cancellation_requested=execution.cancellation_requested,
                cache_path=get_raw_probe_cache_path(outputs.preprocess_dir),
                ffprobe_path=config.encoding.ffmpeg.ffprobe_path,
            )

            current_stage = "trim resolution"
            logger.info("stage=trim_validation")
            trim = resolve_trim_range(
                request_start_frame=request.start_frame,
                request_end_frame_exclusive=request.end_frame_exclusive,
                config=config,
                raw_readable_frame_count=raw_probe.frame_count_opencv_readable,
            )

            current_stage = "external timing validation"
            logger.info("stage=external_timing_validation")
            timing = _resolve_external_timing(
                request.external_time_selection, request.external_time_vector_seconds, raw_probe
            )
            fps = select_output_fps(raw_probe, external_fps_effective=timing.external_fps_effective)
            logger.info("fps selected value=%.12g source=%s", fps.fps_header, fps.fps_header_source)

            current_stage = "crop plan resolution"
            logger.info("stage=crop_plan_resolution")
            detector_diagnostics = request.detector_diagnostics
            if crop_plan is None and request.automatic_crop_requested:
                resolved_pre_crop = resolve_pre_crop(
                    config.pre_crop, (raw_probe.width, raw_probe.height)
                )
                detection = detect_cage_crop_plan(raw_probe.source_path, config, resolved_pre_crop)
                crop_plan = detection.crop_plan
                detector_diagnostics = detection.detector_diagnostics
            if crop_plan is None:
                raise PreprocessServiceError("A CropPlan is required before video processing.")

            current_stage = "crop acceptance validation"
            logger.info("stage=crop_acceptance_validation")
            if not crop_plan.accepted_by_user:
                raise PreprocessServiceError(
                    "CropPlan.accepted_by_user must be true before video processing."
                )
            if not request.crop_accepted_by_user:
                raise PreprocessServiceError("The request must explicitly confirm crop acceptance.")
            _validate_full_frame_plan_matches_probe(crop_plan, raw_probe)

            assert staged_outputs is not None
            raise_if_cancelled()
            expected_pipeline_count = trim.expected_frame_count
            if (
                expected_pipeline_count is None
                and raw_probe.frame_count_opencv_readable is not None
            ):
                expected_pipeline_count = (
                    raw_probe.frame_count_opencv_readable - trim.start_frame
                )
            suffix = config.encoding.ffmpeg.container
            base_internal_path = internal_dir / f"stage_a_intermediate.{suffix}"
            internal_path = (
                base_internal_path
                if not base_internal_path.exists()
                else internal_dir / f"stage_a_intermediate_{uuid.uuid4().hex}.{suffix}"
            )
            current_stage = "Stage A ffmpeg preparation"
            logger.info("stage=ffmpeg_stage_a output=%s", internal_path)
            report("Stage A: preparing video geometry")
            stage_a_result = run_ffmpeg_prepare(
                raw_video_path=raw_probe.source_path,
                intermediate_path=internal_path,
                crop_plan=crop_plan,
                config=config,
                start_frame=trim.start_frame,
                end_frame_exclusive=trim.end_frame_exclusive,
                expected_frame_count=expected_pipeline_count,
                progress_callback=execution.progress_callback,
                cancellation_requested=execution.cancellation_requested,
            )

            current_stage = "Stage B OpenCV re-encoding"
            raise_if_cancelled()
            expected_count = expected_pipeline_count or getattr(
                stage_a_result, "frames_processed", None
            )
            logger.info(
                "stage=opencv_stage_b output=%s", staged_outputs.prepared_video_path
            )
            report("Stage B: writing SLEAP-compatible video")
            stage_b_result = reencode_intermediate_with_opencv(
                intermediate_video_path=stage_a_result.intermediate_path,
                prepared_video_path=staged_outputs.prepared_video_path,
                output_fps=fps.fps_header,
                output_size_wh=crop_plan.prepared_size_wh,
                config=config,
                expected_frame_count=expected_count,
                progress_callback=execution.progress_callback,
                cancellation_requested=execution.cancellation_requested,
            )

            current_stage = "prepared video validation"
            raise_if_cancelled()
            logger.info("stage=prepared_video_validation")
            expected_count = (
                expected_pipeline_count
                if expected_pipeline_count is not None
                else stage_b_result.frames_written
            )
            report("Validating prepared video")
            prepared_validation = validate_prepared_video(
                staged_outputs.prepared_video_path,
                expected_frame_count=expected_count,
                expected_size_wh=crop_plan.prepared_size_wh,
                progress_callback=execution.progress_callback,
                cancellation_requested=execution.cancellation_requested,
            )
            if not prepared_validation.is_valid:
                raise PreprocessServiceError("Prepared video validation did not pass.")

            current_stage = "prepared sync writing"
            raise_if_cancelled()
            logger.info("stage=prepared_sync_write")
            report("Writing synchronization artifact")
            sync = _build_sync(
                validation=prepared_validation,
                trim=trim,
                fps=fps,
                raw_probe=raw_probe,
                timing=timing,
            )
            write_prepared_sync_npz(staged_outputs.prepared_sync_path, sync)

            current_stage = "background generation"
            raise_if_cancelled()
            logger.info("stage=background_generation")
            report("Generating cropped background")
            background_result = estimate_prepared_background(
                staged_outputs.prepared_video_path,
                sample_every_n=config.background.sample_every_n,
                max_samples=config.background.max_samples,
                method=config.background.method,
                expected_frame_count=prepared_validation.opencv_readable_frame_count,
                progress_callback=execution.progress_callback,
                cancellation_requested=execution.cancellation_requested,
            )
            validate_background(background_result.background, crop_plan.prepared_size_wh)
            raise_if_cancelled()
            write_background_png(
                background_result.background,
                staged_outputs.cropped_background_path,
            )
            background_result = background_result.model_copy(
                update={"source_video_path": outputs.prepared_video_path.resolve()}
            )

            current_stage = "metadata construction"
            raise_if_cancelled()
            logger.info("stage=metadata_build")
            report("Writing metadata and settings")
            ffmpeg_executable = (
                Path(stage_a_result.ffmpeg_command[0]) if stage_a_result.ffmpeg_command else None
            )
            software_environment = _software_environment(
                ffmpeg_executable=ffmpeg_executable, config=config
            )
            metadata = build_prepare_metadata(
                project=project,
                raw_probe=raw_probe,
                prepared_validation=prepared_validation,
                crop_plan=crop_plan,
                preprocess_config=config,
                external_time_selection=request.external_time_selection,
                sync_path=outputs.prepared_sync_path,
                background_result=background_result,
                outputs=outputs,
                start_frame=trim.start_frame,
                end_frame_exclusive=trim.end_frame_exclusive,
                software_environment=software_environment,
                detector_diagnostics=detector_diagnostics,
                fps_header=fps.fps_header,
                fps_header_source=fps.fps_header_source,
                raw_fps_effective=fps.raw_fps_effective,
                raw_fps_effective_method=fps.raw_fps_effective_method,
                external_fps_effective=fps.external_fps_effective,
                external_fps_effective_method=fps.external_fps_effective_method,
                external_time_status=timing.status,
            )
            validate_prepare_metadata(metadata)

            current_stage = "settings writing"
            logger.info("stage=settings_write")
            write_settings_used_yaml(staged_outputs.settings_used_path, config)

            current_stage = "metadata writing"
            logger.info("stage=metadata_write")
            write_prepare_meta_json(staged_outputs.prepare_meta_path, metadata)

            current_stage = "final artifact validation"
            raise_if_cancelled()
            logger.info("stage=final_artifact_validation")
            report("Validating final artifacts")
            staged_paths = (
                staged_outputs.prepared_video_path,
                staged_outputs.prepare_meta_path,
                staged_outputs.prepared_sync_path,
                staged_outputs.cropped_background_path,
                staged_outputs.settings_used_path,
                outputs.processing_log_path,
            )
            missing = [str(path) for path in staged_paths if not path.is_file()]
            if missing:
                raise PreprocessServiceError(
                    f"Successful run is missing official artifacts: {missing}"
                )
            reloaded_sync = load_prepared_sync_npz(staged_outputs.prepared_sync_path)
            if (
                reloaded_sync.frame_count_used_for_sleap
                != prepared_validation.opencv_readable_frame_count
            ):
                raise PreprocessServiceError(
                    "Reloaded sync frame count differs from prepared video."
                )
            try:
                with staged_outputs.prepare_meta_path.open(
                    "r", encoding="utf-8"
                ) as stream:
                    reloaded_metadata = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise PreprocessServiceError(f"Could not reload prepare metadata: {exc}") from exc
            if not isinstance(reloaded_metadata, dict):
                raise PreprocessServiceError("Reloaded prepare metadata must be a JSON object.")
            validate_prepare_metadata(reloaded_metadata)
            validate_background(
                background_result.background, prepared_validation.opencv_reported_size_wh
            )
            raise_if_cancelled()
            _commit_staged_outputs(staged_outputs, outputs)

            succeeded = True
            logger.info(
                "run completed status=success frames=%d",
                prepared_validation.opencv_readable_frame_count,
            )
            report("Completed")
            result = PreprocessResult(
                success=True,
                outputs=outputs,
                raw_probe=raw_probe,
                prepared_validation=prepared_validation,
                crop_plan=crop_plan,
                message="Preprocessing completed successfully.",
                warnings=warnings,
                elapsed_sec=time.perf_counter() - started,
                fps_header=fps.fps_header,
                fps_header_source=fps.fps_header_source,
                raw_fps_effective=fps.raw_fps_effective,
                raw_fps_effective_method=fps.raw_fps_effective_method,
                external_fps_effective=fps.external_fps_effective,
                external_fps_effective_method=fps.external_fps_effective_method,
            )
        except PreprocessCancelledError:
            if logger is not None:
                logger.info("run completed status=cancelled stage=%s", current_stage)
            report("Cancelled", "Preprocessing cancelled")
            result = PreprocessResult(
                success=False,
                cancelled=True,
                outputs=None,
                raw_probe=raw_probe,
                prepared_validation=prepared_validation,
                crop_plan=crop_plan,
                message=(
                    "Preprocessing cancelled. No incomplete outputs were marked "
                    "as valid."
                ),
                warnings=warnings,
                elapsed_sec=time.perf_counter() - started,
                fps_header=fps.fps_header if fps is not None else None,
                fps_header_source=fps.fps_header_source if fps is not None else None,
                raw_fps_effective=fps.raw_fps_effective if fps is not None else None,
                raw_fps_effective_method=(
                    fps.raw_fps_effective_method if fps is not None else None
                ),
                external_fps_effective=(
                    fps.external_fps_effective if fps is not None else None
                ),
                external_fps_effective_method=(
                    fps.external_fps_effective_method if fps is not None else None
                ),
            )
        except (PreprocessError, ProjectError, OSError) as exc:
            if logger is not None:
                logger.error("run failed stage=%s error=%s", current_stage, exc)
            report("Failed", f"Preprocessing failed during {current_stage}")
            result = PreprocessResult(
                success=False,
                outputs=None,
                raw_probe=raw_probe,
                prepared_validation=prepared_validation,
                crop_plan=crop_plan,
                message=f"Preprocessing failed during {current_stage}: {exc}",
                warnings=warnings,
                elapsed_sec=time.perf_counter() - started,
                fps_header=fps.fps_header if fps is not None else None,
                fps_header_source=fps.fps_header_source if fps is not None else None,
                raw_fps_effective=fps.raw_fps_effective if fps is not None else None,
                raw_fps_effective_method=fps.raw_fps_effective_method if fps is not None else None,
                external_fps_effective=fps.external_fps_effective if fps is not None else None,
                external_fps_effective_method=fps.external_fps_effective_method
                if fps is not None
                else None,
            )
        finally:
            debug_enabled = request.config.debug.enabled
            if internal_path is not None and not debug_enabled:
                _remove_file(internal_path, warnings)
            if staging_dir is not None and (succeeded or not debug_enabled):
                _remove_staging_dir(staging_dir, warnings)
            if logger is not None:
                if succeeded and debug_enabled:
                    logger.info("internal artifacts retained debug=true")
                elif internal_path is not None and not debug_enabled:
                    logger.info("internal artifacts cleaned debug=false")
            _close_logger(logger)

        result.warnings[:] = warnings
        result.elapsed_sec = time.perf_counter() - started
        return result
