"""Typed, lightweight shared state for preprocessing setup pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np

from preprocess.config import PreCropConfig, PreprocessConfig
from preprocess.crop_plan import CropPlan
from preprocess.models import (
    ExternalTimeSelection,
    MatVectorCandidate,
    PreprocessResult,
    TimingPlausibilityAssessment,
    TimingUnit,
    VideoProbeResult,
)
from preprocess.pre_crop import ResolvedPreCrop
from project.models import Project


class WorkflowStep(IntEnum):
    """Visible preprocessing workflow steps in desktop navigation order."""

    PROJECT = 0
    RAW_VIDEO = 1
    TRIM_PRE_CROP = 2
    TIMING = 3
    CROP_REVIEW = 4
    ENCODE_SETTINGS = 5
    RUN_VALIDATE = 6


WORKFLOW_STEP_LABELS = (
    "Project",
    "Raw Video",
    "Trim and Pre-Crop",
    "Timing",
    "Crop Review",
    "Encode Settings",
    "Run and Validate",
)


class TimingMode(StrEnum):
    """Explicit external-timing choices shown by the Timing page."""

    NO_EXTERNAL = "no_external_timing"
    MATLAB = "matlab_timing"


class RawReadableCountStatus(StrEnum):
    """Provenance or activity status of the exact sequential raw-frame count."""

    NOT_COUNTED = "not_counted"
    COUNTED_THIS_SESSION = "counted_this_session"
    RECORDED_PRIOR_RUN = "recorded_prior_run"
    VALIDATED_REUSABLE_CACHE = "validated_reusable_cache"
    COUNTING = "counting"
    RECOUNTING = "recounting"
    COUNT_FAILED = "count_failed"
    RECOUNT_FAILED = "recount_failed"
    COUNT_CANCELLED = "count_cancelled"


class CropReviewMode(StrEnum):
    """Crop-review choices available in the desktop workflow."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"


@dataclass(slots=True)
class PreprocessSetupState:
    """Small mutable view-model state without frames or open video handles."""

    project: Project | None = None
    project_dir: Path | None = None
    preprocess_dir: Path | None = None
    project_hydration_status: str = "not_checked"
    project_hydration_message: str | None = None
    prior_run_status: str | None = None
    raw_video_path: Path | None = None
    raw_probe: VideoProbeResult | None = None
    raw_readable_count_status: RawReadableCountStatus = (
        RawReadableCountStatus.NOT_COUNTED
    )
    preprocess_config: PreprocessConfig | None = None
    original_preprocess_config: PreprocessConfig | None = None
    config_dirty: bool = False
    encode_settings_valid: bool = True
    config_path: Path | None = None
    start_frame: int | None = None
    end_frame_exclusive: int | None = None
    pre_crop_config: PreCropConfig | None = None
    resolved_pre_crop: ResolvedPreCrop | None = None
    timing_mode: TimingMode = TimingMode.NO_EXTERNAL
    timing_status: str = "not_provided"
    timing_valid: bool = True
    external_time_mat_path: Path | None = None
    mat_workspace_format: str | None = None
    mat_candidates: list[MatVectorCandidate] = field(default_factory=list)
    selected_timing_variable: str | None = None
    selected_timing_units: TimingUnit | None = None
    external_time_selection: ExternalTimeSelection | None = None
    external_time_vector_seconds: np.ndarray | None = None
    timing_plausibility_assessment: TimingPlausibilityAssessment | None = None
    candidate_crop_plan: CropPlan | None = None
    accepted_crop_plan: CropPlan | None = None
    crop_mode: CropReviewMode = CropReviewMode.AUTOMATIC
    crop_review_status: str = "not_reviewed"
    detector_diagnostics: dict[str, object] | None = None
    crop_review_revision: int = 0
    preprocess_task_running: bool = False
    preprocess_result: PreprocessResult | None = None
    run_status: str = "Not started"
    run_error_message: str | None = None
    run_started_at: datetime | None = None
    run_elapsed_sec: float = 0.0
    trim_pre_crop_valid: bool = False
    current_step: WorkflowStep = WorkflowStep.PROJECT
    last_validation_error: str | None = None
    unexpected_error_detail: str | None = None

    def invalidate_crop_review(self, status: str = "not_reviewed") -> None:
        """Clear all crop decisions that depend on upstream setup state."""

        self.candidate_crop_plan = None
        self.accepted_crop_plan = None
        self.crop_review_status = status
        self.detector_diagnostics = None
        self.crop_review_revision += 1
        self.invalidate_run_result("Inputs changed")

    def store_accepted_crop_plan(self, crop_plan: CropPlan) -> None:
        """Store only a CropPlan carrying explicit user acceptance."""

        if not crop_plan.accepted_by_user:
            raise ValueError("accepted_crop_plan must be explicitly accepted by the user.")
        self.accepted_crop_plan = crop_plan

    def set_loaded_preprocess_config(self, config: PreprocessConfig) -> None:
        """Store independent validated current/original copies for this session."""

        payload = config.model_dump(mode="python")
        self.original_preprocess_config = PreprocessConfig.model_validate(payload)
        self.preprocess_config = PreprocessConfig.model_validate(payload)
        self.config_dirty = False
        self.encode_settings_valid = True
        self.invalidate_run_result("Configuration loaded")

    def store_preprocess_config(self, config: PreprocessConfig) -> None:
        """Store one validated GUI config copy and update its dirty state."""

        validated = PreprocessConfig.model_validate(config.model_dump(mode="python"))
        changed = self.preprocess_config is None or (
            self.preprocess_config.model_dump(mode="json")
            != validated.model_dump(mode="json")
        )
        self.preprocess_config = validated
        original = self.original_preprocess_config
        self.config_dirty = original is None or (
            original.model_dump(mode="json") != validated.model_dump(mode="json")
        )
        self.encode_settings_valid = True
        if changed:
            self.invalidate_run_result("Configuration changed")

    def invalidate_run_result(self, status: str = "Inputs changed") -> None:
        """Clear displayed run success without deleting any output artifacts."""

        self.preprocess_result = None
        self.run_error_message = None
        if not self.preprocess_task_running:
            self.run_status = status
            self.run_started_at = None
            self.run_elapsed_sec = 0.0
