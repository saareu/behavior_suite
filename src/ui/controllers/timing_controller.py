"""Headless controller for optional MATLAB external timing selection."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np

from preprocess.exceptions import PreprocessError
from preprocess.mat_sync_reader import (
    convert_timing_vector_to_seconds,
    get_numeric_vector,
    list_numeric_vectors,
    load_mat_workspace,
    validate_external_timing_vector,
)
from preprocess.models import (
    ExternalTimeSelection,
    MatVectorCandidate,
    MatWorkspace,
    TimingUnit,
    VideoProbeResult,
)
from preprocess.video_probe import probe_video
from ui.state import PreprocessSetupState, TimingMode

WorkspaceLoader = Callable[[Path], MatWorkspace]
CandidateLister = Callable[[MatWorkspace], list[MatVectorCandidate]]
VectorGetter = Callable[[MatWorkspace, str], np.ndarray]
TimingValidator = Callable[..., ExternalTimeSelection]
TimingConverter = Callable[[np.ndarray, str | TimingUnit], np.ndarray | None]


class TimingValidationError(ValueError):
    """Expected user-facing timing setup or validation failure."""


class TimingUnexpectedError(RuntimeError):
    """Concise wrapper for an unexpected controller-boundary failure."""


@dataclass(frozen=True, slots=True)
class TimingCandidateSummary:
    """Compact candidate row with current raw-count length comparison."""

    variable_name: str
    shape: tuple[int, ...]
    dtype: str
    length_after_squeeze: int
    first_value: float
    last_value: float
    median_difference: float | None
    estimated_fps: float | None
    length_matches_raw: bool | None


class TimingController:
    """Coordinate Timing-page state through existing MAT and probe APIs."""

    def __init__(
        self,
        state: PreprocessSetupState,
        *,
        workspace_loader: WorkspaceLoader = load_mat_workspace,
        candidate_lister: CandidateLister = list_numeric_vectors,
        vector_getter: VectorGetter = get_numeric_vector,
        timing_validator: TimingValidator = validate_external_timing_vector,
        timing_converter: TimingConverter = convert_timing_vector_to_seconds,
        probe_function: Callable[..., VideoProbeResult] = probe_video,
    ) -> None:
        self.state = state
        self._workspace_loader = workspace_loader
        self._candidate_lister = candidate_lister
        self._vector_getter = vector_getter
        self._timing_validator = timing_validator
        self._timing_converter = timing_converter
        self._probe_function = probe_function
        self._workspace: MatWorkspace | None = None

    def choose_no_external_timing(self) -> None:
        """Clear external timing and mark the no-timing path navigable."""

        self._workspace = None
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
        self.state.last_validation_error = None

    def choose_matlab_timing(self) -> None:
        """Select MATLAB timing mode and require explicit validation."""

        self.state.timing_mode = TimingMode.MATLAB
        self._invalidate_matlab_selection(status="pending")

    def set_mat_file_path(self, path: Path) -> Path:
        """Store a MATLAB path while clearing any prior workspace selection."""

        selected = Path(path).expanduser()
        self._workspace = None
        self.state.timing_mode = TimingMode.MATLAB
        self.state.external_time_mat_path = selected
        self.state.mat_workspace_format = None
        self.state.mat_candidates.clear()
        self.state.selected_timing_variable = None
        self.state.selected_timing_units = None
        self._invalidate_matlab_selection(status="pending")
        return selected

    def load_mat_file(self, path: Path | None = None) -> list[MatVectorCandidate]:
        """Load a MAT workspace and store only compact candidate summaries."""

        selected_path = (
            Path(path).expanduser()
            if path is not None
            else self.state.external_time_mat_path
        )
        if selected_path is None:
            raise self._new_validation_error("Select a MATLAB .mat file first.")
        self.set_mat_file_path(selected_path)
        try:
            workspace = self._workspace_loader(selected_path)
            candidates = self._candidate_lister(workspace)
        except (PreprocessError, OSError, ValueError) as exc:
            self._expected_failure(f"Could not load MATLAB timing file: {exc}", exc)
        except Exception as exc:
            self._unexpected_failure("loading MATLAB timing file", exc)
        if not candidates:
            raise self._new_validation_error(
                "MATLAB file contains no eligible numeric vector variables."
            )
        self._workspace = workspace
        self.state.external_time_mat_path = workspace.source_path
        self.state.mat_workspace_format = workspace.format
        self.state.mat_candidates = list(candidates)
        self.state.selected_timing_variable = None
        self.state.selected_timing_units = None
        self._invalidate_matlab_selection(status="pending")
        return list(candidates)

    def candidate_summaries(self) -> list[TimingCandidateSummary]:
        """Return display rows with length match against sequential raw count."""

        raw_count = (
            self.state.raw_probe.frame_count_opencv_readable
            if self.state.raw_probe is not None
            else None
        )
        return [
            TimingCandidateSummary(
                variable_name=candidate.variable_name,
                shape=candidate.shape,
                dtype=candidate.dtype,
                length_after_squeeze=candidate.length_after_squeeze,
                first_value=candidate.first_value,
                last_value=candidate.last_value,
                median_difference=candidate.median_difference,
                estimated_fps=candidate.estimated_fps,
                length_matches_raw=(
                    None
                    if raw_count is None
                    else candidate.length_after_squeeze == raw_count
                ),
            )
            for candidate in self.state.mat_candidates
        ]

    def select_candidate(self, variable_name: str) -> None:
        """Store one explicit candidate selection without auto-selecting."""

        if self._workspace is None:
            raise self._new_validation_error("Load MATLAB variables first.")
        available_names = {
            candidate.variable_name for candidate in self.state.mat_candidates
        }
        if variable_name not in available_names:
            raise self._new_validation_error(
                f"Selected MATLAB variable is not an eligible candidate: {variable_name}"
            )
        self.state.selected_timing_variable = variable_name
        self._invalidate_matlab_selection(status="pending", preserve_choices=True)

    def select_units(self, units: str | TimingUnit) -> TimingUnit:
        """Store one explicit supported timing-unit choice."""

        try:
            selected_units = TimingUnit(units)
        except ValueError as exc:
            raise self._new_validation_error(
                f"Unsupported timing units: {units}"
            ) from exc
        self.state.selected_timing_units = selected_units
        self._invalidate_matlab_selection(status="pending", preserve_choices=True)
        return selected_units

    def clear_units_selection(self) -> None:
        """Clear an explicit units choice and invalidate prior timing acceptance."""

        self.state.selected_timing_units = None
        self._invalidate_matlab_selection(status="pending", preserve_choices=True)

    def count_all_readable_raw_frames(self) -> int:
        """Request the existing probe API's full sequential readable count."""

        path = self.prepare_full_raw_frame_count()
        try:
            result = self.compute_full_raw_frame_count(path)
        except (PreprocessError, OSError, ValueError) as exc:
            self._expected_failure(f"Could not count readable raw frames: {exc}", exc)
        except Exception as exc:
            self._unexpected_failure("counting readable raw frames", exc)
        return self.apply_full_raw_frame_count(result)

    def prepare_full_raw_frame_count(self) -> Path:
        """Validate a sequential-count request before dispatching a worker."""

        if self.state.raw_video_path is None:
            raise self._new_validation_error("Select and probe a raw video first.")
        return self.state.raw_video_path

    def compute_full_raw_frame_count(self, path: Path) -> VideoProbeResult:
        """Run a full raw-video probe without mutating GUI state."""

        return self._probe_function(path, require_sequential_count=True)

    def apply_full_raw_frame_count(self, result: VideoProbeResult) -> int:
        """Apply a completed full raw-video count on the GUI thread."""

        readable_count = result.frame_count_opencv_readable
        if readable_count is None or readable_count <= 0:
            raise self._new_validation_error(
                "Full raw-video probe did not produce a positive readable frame count."
            )
        self.state.raw_probe = result
        self.state.raw_video_path = result.source_path
        if self.state.timing_mode is TimingMode.MATLAB:
            self._invalidate_matlab_selection(
                status="pending",
                preserve_choices=True,
            )
        self.state.last_validation_error = None
        return readable_count

    def record_full_raw_frame_count_failure(self, exc: BaseException) -> None:
        """Convert a worker-delivered failure into controller timing state."""

        if isinstance(exc, (PreprocessError, OSError, ValueError)):
            self.state.timing_valid = False
            self.state.timing_status = "invalid"
            self.state.external_time_selection = None
            self.state.external_time_vector_seconds = None
            self.state.last_validation_error = (
                f"Could not count readable raw frames: {exc}"
            )
            return
        self.state.timing_valid = False
        self.state.timing_status = "error"
        self.state.external_time_selection = None
        self.state.external_time_vector_seconds = None
        self.state.unexpected_error_detail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    def validate_selected_timing(self) -> ExternalTimeSelection:
        """Validate one explicit vector/units choice through core timing APIs."""

        if self.state.timing_mode is not TimingMode.MATLAB:
            raise self._new_validation_error("Choose MATLAB timing mode first.")
        if self._workspace is None:
            raise self._new_validation_error("Load MATLAB variables first.")
        variable_name = self.state.selected_timing_variable
        if variable_name is None:
            raise self._new_validation_error("Select a candidate timing variable.")
        units = self.state.selected_timing_units
        if units is None:
            raise self._new_validation_error("Select timing units.")
        raw_probe = self.state.raw_probe
        raw_count = (
            raw_probe.frame_count_opencv_readable
            if raw_probe is not None
            else None
        )
        if raw_count is None or raw_count <= 0:
            raise self._new_validation_error(
                "A full sequential raw readable-frame count is required."
            )
        self._invalidate_matlab_selection(status="validating", preserve_choices=True)
        try:
            vector = self._vector_getter(self._workspace, variable_name)
            selection = self._timing_validator(
                vector,
                raw_count,
                units,
                source_path=self.state.external_time_mat_path,
                selected_variable=variable_name,
            )
            vector_seconds = self._timing_converter(vector, units)
        except (PreprocessError, OSError, ValueError) as exc:
            self._expected_failure(f"External timing validation failed: {exc}", exc)
        except Exception as exc:
            self._unexpected_failure("validating external timing", exc)
        if vector_seconds is not None:
            vector_seconds = np.array(vector_seconds, dtype=np.float64, copy=True)
            vector_seconds.setflags(write=False)
            timing_status = "valid"
        else:
            timing_status = "not_convertible_to_seconds"
        self.state.external_time_selection = selection
        self.state.external_time_vector_seconds = vector_seconds
        self.state.timing_status = timing_status
        self.state.timing_valid = True
        self.state.last_validation_error = None
        return selection

    def validate_navigation(self) -> None:
        """Raise unless no timing or a fully validated MAT selection is ready."""

        if (
            self.state.timing_mode is TimingMode.NO_EXTERNAL
            and self.state.timing_valid
            and self.state.timing_status == "not_provided"
            and self.state.external_time_selection is None
            and self.state.external_time_vector_seconds is None
        ):
            return
        if (
            self.state.timing_mode is TimingMode.MATLAB
            and self.state.timing_valid
            and self.state.external_time_selection is not None
            and self.state.timing_status
            in {"valid", "not_convertible_to_seconds"}
        ):
            return
        raise self._new_validation_error(
            "Complete and validate the selected external timing option."
        )

    def can_advance(self) -> bool:
        """Return whether Timing-page state permits forward navigation."""

        try:
            self.validate_navigation()
        except TimingValidationError:
            return False
        return True

    def _invalidate_matlab_selection(
        self,
        *,
        status: str,
        preserve_choices: bool = False,
    ) -> None:
        self.state.timing_valid = False
        self.state.timing_status = status
        self.state.external_time_selection = None
        self.state.external_time_vector_seconds = None
        if not preserve_choices:
            self.state.selected_timing_variable = None
            self.state.selected_timing_units = None
        self.state.last_validation_error = None

    def _new_validation_error(self, message: str) -> TimingValidationError:
        self.state.timing_valid = False
        self.state.last_validation_error = message
        return TimingValidationError(message)

    def _expected_failure(self, message: str, exc: BaseException) -> NoReturn:
        self.state.timing_valid = False
        self.state.timing_status = "invalid"
        self.state.external_time_selection = None
        self.state.external_time_vector_seconds = None
        self.state.last_validation_error = message
        raise TimingValidationError(message) from exc

    def _unexpected_failure(self, context: str, exc: BaseException) -> NoReturn:
        self.state.timing_valid = False
        self.state.timing_status = "error"
        self.state.external_time_selection = None
        self.state.external_time_vector_seconds = None
        self.state.unexpected_error_detail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        raise TimingUnexpectedError(
            f"An unexpected error occurred while {context}."
        ) from exc
