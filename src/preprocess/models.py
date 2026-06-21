"""Typed data models shared by first-sprint preprocessing modules."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from preprocess.config import PreprocessConfig
from preprocess.crop_plan import CropPlan
from project.models import Project


class TimingUnit(StrEnum):
    """Supported declarations for an external per-frame vector."""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    NANOSECONDS = "nanoseconds"
    FRAMES = "frames"
    UNKNOWN = "unknown"


class SoftwareEnvironmentInfo(BaseModel):
    """Version provenance supplied to preprocessing metadata generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str | None = None
    platform: str | None = None
    opencv_version: str | None = None
    numpy_version: str | None = None
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    application_version: str | None = None


class VideoProbeResult(BaseModel):
    """Metadata collected independently from ffprobe and OpenCV."""

    model_config = ConfigDict(extra="forbid")

    source_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec: str | None = None
    pixel_format: str | None = None
    duration_sec: float | None = Field(default=None, ge=0)
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    time_base: str | None = None
    frame_count_ffprobe: int | None = Field(default=None, ge=0)
    frame_count_opencv_reported: int = Field(ge=0)
    frame_count_opencv_readable: int | None = Field(default=None, ge=0)
    opencv_fps: float | None = Field(default=None, gt=0)
    raw_fps_effective: float | None = Field(default=None, gt=0)
    raw_fps_effective_method: str | None = None
    pts_status: str
    ffprobe_succeeded: bool
    ffprobe_error: str | None = None
    ffprobe_metadata: dict[str, Any] | None = None


class MatWorkspace(BaseModel):
    """Variables loaded from a MATLAB workspace."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    source_path: Path
    format: str
    variables: dict[str, Any]


class MatVectorCandidate(BaseModel):
    """Display metadata for a numeric vector in a MATLAB workspace."""

    model_config = ConfigDict(extra="forbid")

    variable_name: str
    shape: tuple[int, ...]
    dtype: str
    length_after_squeeze: int = Field(ge=1)
    first_value: float
    last_value: float
    median_difference: float | None = None
    estimated_fps: float | None = Field(default=None, gt=0)


class ExternalTimeSelection(BaseModel):
    """Validation metadata for a user-selected external timing vector."""

    model_config = ConfigDict(extra="forbid")

    provided: bool
    source_path: Path | None = None
    selected_variable: str | None = None
    declared_units: TimingUnit | None = None
    raw_vector_length: int = Field(ge=0)
    raw_video_frame_count_opencv_readable: int = Field(ge=0)
    is_numeric: bool
    is_one_dimensional: bool
    is_finite: bool
    is_monotonic_increasing: bool
    median_difference: float | None = None
    estimated_fps: float | None = Field(default=None, gt=0)
    validation_status: str


class PreprocessOutputs(BaseModel):
    """Official preprocessing artifact paths for a project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_dir: Path
    prepared_video_path: Path
    prepare_meta_path: Path
    prepared_sync_path: Path
    cropped_background_path: Path
    settings_used_path: Path
    processing_log_path: Path

    @classmethod
    def from_preprocess_dir(cls, preprocess_dir: Path) -> "PreprocessOutputs":
        """Build all official output paths below preprocess_dir."""

        root = Path(preprocess_dir)
        return cls(
            preprocess_dir=root,
            prepared_video_path=root / "prepared_video.mp4",
            prepare_meta_path=root / "prepare_meta.json",
            prepared_sync_path=root / "prepared_sync.npz",
            cropped_background_path=root / "cropped_background.png",
            settings_used_path=root / "settings_used.yaml",
            processing_log_path=root / "processing_log.txt",
        )


class PreprocessRequest(BaseModel):
    """First-sprint placeholder for a future end-to-end preprocess request."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    project: Project
    raw_video_path: Path
    config: PreprocessConfig
    start_frame: int = Field(default=0, ge=0)
    end_frame_exclusive: int | None = Field(default=None, ge=1)
    accepted_crop_plan: CropPlan | None = None
    external_time_selection: ExternalTimeSelection | None = None
    external_timing_vector: np.ndarray | None = None

    @model_validator(mode="after")
    def validate_frame_range(self) -> "PreprocessRequest":
        if self.end_frame_exclusive is not None and self.end_frame_exclusive <= self.start_frame:
            raise ValueError("end_frame_exclusive must be greater than start_frame.")
        return self


class PreprocessResult(BaseModel):
    """First-sprint placeholder for a future end-to-end preprocess result."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    outputs: PreprocessOutputs | None = None
    validation_result: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
