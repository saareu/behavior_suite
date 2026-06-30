"""Pydantic configuration models and YAML loading for preprocessing."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    boundary_px: int | None = Field(default=None, ge=0)
    manual_rectangle: tuple[int, int, int, int] | None = None

    @model_validator(mode="after")
    def validate_enabled_mode(self) -> "PreCropConfig":
        if self.enabled and self.mode == "none":
            raise ValueError("pre_crop.mode must be selected when pre_crop.enabled is true.")
        if not self.enabled and self.mode != "none":
            raise ValueError("pre_crop.mode must be 'none' when pre_crop.enabled is false.")
        if self.mode == "none":
            if self.boundary_px is not None or self.manual_rectangle is not None:
                raise ValueError("pre_crop geometry must be absent when mode is 'none'.")
        elif self.mode == "manual_rectangle":
            if self.manual_rectangle is None:
                raise ValueError(
                    "pre_crop.manual_rectangle is required for manual_rectangle mode."
                )
            if self.boundary_px is not None:
                raise ValueError(
                    "pre_crop.boundary_px is not used for manual_rectangle mode."
                )
        else:
            if self.boundary_px is None:
                raise ValueError("pre_crop.boundary_px is required for boundary modes.")
            if self.manual_rectangle is not None:
                raise ValueError(
                    "pre_crop.manual_rectangle is only valid for manual_rectangle mode."
                )
        return self


class CageDetectConfig(_StrictConfigModel):
    """Initial automatic cage detector settings."""

    sample_step: int = Field(default=500, gt=0)
    pad_px: int = Field(default=2, ge=0)
    threshold: int = Field(default=90, ge=0, le=255)
    pre_crop_expansion_percent: float = Field(default=15.0, ge=0.0, le=200.0)
    dilate_kernel_size: int = Field(default=13, gt=0)
    erode_kernel_size: int = Field(default=7, gt=0)
    rim_close_kernel_size: int = Field(default=9, gt=0)
    minimum_cage_width_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    minimum_cage_height_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    minimum_contour_area: float = Field(default=100.0, gt=0.0)
    fit_tolerance_px: int = Field(default=6, ge=0)

    @field_validator(
        "dilate_kernel_size",
        "erode_kernel_size",
        "rim_close_kernel_size",
    )
    @classmethod
    def validate_odd_kernel_size(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("Detector morphology kernel sizes must be odd.")
        return value


class MaskRectangleConfig(_StrictConfigModel):
    """Axis-aligned static mask rectangle in prepared-pixel coordinates."""

    type: Literal["rectangle"] = "rectangle"
    x: int
    y: int
    width: int
    height: int

    @field_validator("x", "y", "width", "height", mode="before")
    @classmethod
    def validate_integer(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Rectangle mask coordinates must be integer pixels.")
        return value

    @model_validator(mode="after")
    def validate_positive_size(self) -> "MaskRectangleConfig":
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rectangle masks require positive width and height.")
        return self


def _polygon_signed_area(vertices: tuple[tuple[int, int], ...]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(vertices):
        x1, y1 = vertices[(index + 1) % len(vertices)]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def _orientation(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    return (value > 0) - (value < 0)


def _point_on_segment(
    point: tuple[int, int],
    segment_start: tuple[int, int],
    segment_end: tuple[int, int],
) -> bool:
    return (
        min(segment_start[0], segment_end[0])
        <= point[0]
        <= max(segment_start[0], segment_end[0])
        and min(segment_start[1], segment_end[1])
        <= point[1]
        <= max(segment_start[1], segment_end[1])
        and _orientation(segment_start, segment_end, point) == 0
    )


def _segments_intersect(
    first_start: tuple[int, int],
    first_end: tuple[int, int],
    second_start: tuple[int, int],
    second_end: tuple[int, int],
) -> bool:
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)
    if first_orientation != second_orientation and third_orientation != fourth_orientation:
        return True
    return (
        _point_on_segment(second_start, first_start, first_end)
        or _point_on_segment(second_end, first_start, first_end)
        or _point_on_segment(first_start, second_start, second_end)
        or _point_on_segment(first_end, second_start, second_end)
    )


def _polygon_is_simple(vertices: tuple[tuple[int, int], ...]) -> bool:
    edge_count = len(vertices)
    for first_index in range(edge_count):
        first_start = vertices[first_index]
        first_end = vertices[(first_index + 1) % edge_count]
        for second_index in range(first_index + 1, edge_count):
            if second_index in {
                first_index,
                (first_index + 1) % edge_count,
                (first_index - 1) % edge_count,
            }:
                continue
            second_start = vertices[second_index]
            second_end = vertices[(second_index + 1) % edge_count]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return False
    return True


class MaskPolygonConfig(_StrictConfigModel):
    """Simple polygon static mask in prepared-pixel coordinates."""

    type: Literal["polygon"] = "polygon"
    vertices: tuple[tuple[int, int], ...]

    @field_validator("vertices", mode="before")
    @classmethod
    def validate_vertices(cls, value: object) -> tuple[tuple[int, int], ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Polygon mask vertices must be a sequence.")
        vertices: list[tuple[int, int]] = []
        for vertex in value:
            if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
                raise ValueError("Each polygon mask vertex must contain x and y.")
            x, y = vertex
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, int)
                or not isinstance(y, int)
            ):
                raise ValueError("Polygon mask vertices must use integer pixels.")
            vertices.append((x, y))
        return tuple(vertices)

    @model_validator(mode="after")
    def validate_polygon(self) -> "MaskPolygonConfig":
        if len(self.vertices) < 3:
            raise ValueError("Polygon masks require at least three vertices.")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("Polygon masks must not repeat vertices.")
        if _polygon_signed_area(self.vertices) == 0:
            raise ValueError("Polygon masks must have nonzero area.")
        if not _polygon_is_simple(self.vertices):
            raise ValueError("Polygon masks must be simple and non-self-intersecting.")
        return self


MaskShapeConfig = Annotated[
    MaskRectangleConfig | MaskPolygonConfig,
    Field(discriminator="type"),
]


class MaskConfig(_StrictConfigModel):
    """Reserved static prepared-coordinate masking configuration."""

    enabled: bool = False
    coordinate_space: Literal["prepared_pixels"] = "prepared_pixels"
    fill_value: Literal["black"] = "black"
    shapes: list[MaskShapeConfig] = Field(default_factory=list)


class FFmpegEncodingConfig(_StrictConfigModel):
    """Validated legacy-compatible ffmpeg encoding profile."""

    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
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
