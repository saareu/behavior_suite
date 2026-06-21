"""Pydantic configuration models and YAML loading for preprocessing."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from preprocess.exceptions import PreprocessConfigError


class _StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrimConfig(_StrictConfigModel):
    """Decode-order frame range configuration."""

    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=1)
    end_frame_semantics: Literal["exclusive"] = "exclusive"

    @model_validator(mode="after")
    def validate_frame_range(self) -> "TrimConfig":
        start = self.start_frame or 0
        if self.end_frame is not None and self.end_frame <= start:
            raise ValueError("trim.end_frame must be greater than trim.start_frame.")
        return self


class TimingConfig(_StrictConfigModel):
    """Frame-identity and output-timing invariants."""

    mode: Literal["decode_order_cfr"] = "decode_order_cfr"
    allow_frame_resampling: Literal[False] = False
    require_one_to_one_frame_mapping: Literal[True] = True
    require_constant_output_fps_header: Literal[True] = True


class CanonicalResolutionConfig(_StrictConfigModel):
    """Aspect-ratio-preserving canonical output dimensions."""

    enabled: bool = True
    width: int = Field(default=928, gt=0)
    height: int = Field(default=528, gt=0)
    flags: Literal["linear", "cubic", "lanczos"] = "lanczos"

    @model_validator(mode="after")
    def validate_even_dimensions(self) -> "CanonicalResolutionConfig":
        if self.width % 2 or self.height % 2:
            raise ValueError("Canonical width and height must both be even.")
        return self


class PrepareConfig(_StrictConfigModel):
    """Spatial preparation settings used after crop acceptance."""

    roi_margin_px: int = Field(default=40, ge=0)
    perspective_interpolation: Literal["linear", "cubic"] = "cubic"
    canonical_resolution: CanonicalResolutionConfig = Field(
        default_factory=CanonicalResolutionConfig
    )


class PreCropConfig(_StrictConfigModel):
    """Optional detection pre-crop mode."""

    enabled: bool = False
    mode: Literal[
        "none",
        "vertical_keep_left",
        "vertical_keep_right",
        "horizontal_keep_upper",
        "horizontal_keep_lower",
        "manual_rectangle",
    ] = "none"

    @model_validator(mode="after")
    def validate_enabled_mode(self) -> "PreCropConfig":
        if self.enabled and self.mode == "none":
            raise ValueError("pre_crop.mode must be selected when pre_crop.enabled is true.")
        if not self.enabled and self.mode != "none":
            raise ValueError("pre_crop.mode must be 'none' when pre_crop.enabled is false.")
        return self


class CageDetectConfig(_StrictConfigModel):
    """Initial automatic cage detector settings."""

    sample_step: int = Field(default=500, gt=0)
    pad_px: int = Field(default=2, ge=0)
    threshold: int = Field(default=90, ge=0, le=255)


class MaskConfig(_StrictConfigModel):
    """Reserved static prepared-coordinate masking configuration."""

    enabled: bool = False
    coordinate_space: Literal["prepared_video"] = "prepared_video"
    fill_value_bgr: tuple[int, int, int] = (0, 0, 0)
    shapes: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_unimplemented_masking(self) -> "MaskConfig":
        if any(value < 0 or value > 255 for value in self.fill_value_bgr):
            raise ValueError("mask.fill_value_bgr entries must be between 0 and 255.")
        if self.enabled or self.shapes:
            raise ValueError("Static masking is not supported in the first development sprint.")
        return self


class FFmpegEncodingConfig(_StrictConfigModel):
    """Validated legacy-compatible ffmpeg encoding profile."""

    container: Literal["mp4"] = "mp4"
    codec: Literal["libx264"] = "libx264"
    pixel_format: Literal["yuv420p"] = "yuv420p"
    preset: Literal[
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ] = "veryfast"
    crf: int = Field(default=18, ge=0, le=51)
    bframes: Literal[0] = 0
    faststart: Literal[True] = True


class OpenCVEncodingConfig(_StrictConfigModel):
    """Validated legacy-compatible OpenCV encoding profile."""

    container: Literal["mp4"] = "mp4"
    fourcc: Literal["mp4v"] = "mp4v"


class EncodingConfig(_StrictConfigModel):
    """Two-stage video encoding settings."""

    ffmpeg: FFmpegEncodingConfig = Field(default_factory=FFmpegEncodingConfig)
    opencv: OpenCVEncodingConfig = Field(default_factory=OpenCVEncodingConfig)


class BackgroundConfig(_StrictConfigModel):
    """Final prepared-video background sampling settings."""

    method: Literal["median"] = "median"
    max_samples: int = Field(default=500, gt=0)
    sample_every_n: int = Field(default=50, gt=0)


class DebugConfig(_StrictConfigModel):
    """Debug artifact retention settings."""

    enabled: bool = False


class PreprocessConfig(_StrictConfigModel):
    """Complete, validated preprocessing configuration."""

    schema_version: Literal["preprocess_config_v1"] = "preprocess_config_v1"
    trim: TrimConfig = Field(default_factory=TrimConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    pre_crop: PreCropConfig = Field(default_factory=PreCropConfig)
    cage_detect: CageDetectConfig = Field(default_factory=CageDetectConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def load_preprocess_config(path: Path) -> PreprocessConfig:
    """Load and validate a preprocessing YAML file.

    Raises:
        PreprocessConfigError: If the file cannot be read, parsed, or does not
            contain a YAML mapping.
        pydantic.ValidationError: If any configuration value is invalid or an
            unsupported combination is requested.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise PreprocessConfigError(f"Preprocess config file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise PreprocessConfigError(f"Could not load preprocess config {config_path}: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise PreprocessConfigError("Preprocess config must contain a top-level YAML mapping.")
    return PreprocessConfig.model_validate(raw_config)
