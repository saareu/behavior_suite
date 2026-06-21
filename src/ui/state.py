"""Typed, lightweight shared state for preprocessing setup pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np

from preprocess.config import PreCropConfig, PreprocessConfig
from preprocess.models import (
    ExternalTimeSelection,
    MatVectorCandidate,
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


@dataclass(slots=True)
class PreprocessSetupState:
    """Small mutable view-model state without frames or open video handles."""

    project: Project | None = None
    project_dir: Path | None = None
    preprocess_dir: Path | None = None
    raw_video_path: Path | None = None
    raw_probe: VideoProbeResult | None = None
    preprocess_config: PreprocessConfig | None = None
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
    trim_pre_crop_valid: bool = False
    current_step: WorkflowStep = WorkflowStep.PROJECT
    last_validation_error: str | None = None
    unexpected_error_detail: str | None = None
