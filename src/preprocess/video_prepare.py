"""Stage A ffmpeg preparation with deterministic frame and crop geometry."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from preprocess.config import PreprocessConfig
from preprocess.crop_plan import (
    CanonicalGeometry,
    CropPlan,
    build_clockwise_rotation_transform,
)
from preprocess.exceptions import CropPlanError, VideoPreparationError

_BINARY_DIRECTORY_NAME = "bin"
_GEOMETRY_ATOL = 1e-5
_SCALE_FLAG_MAP = {
    "linear": "bilinear",
    "cubic": "bicubic",
    "lanczos": "lanczos",
}


class FiltergraphMetadata(BaseModel):
    """Typed geometry and ordering information for an ffmpeg filtergraph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stages: tuple[str, ...]
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int | None = Field(default=None, ge=1)
    pre_crop_roi_xywh: tuple[int, int, int, int]
    quad_pre_crop_tl_tr_br_bl: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    rectified_content_size_wh: tuple[int, int]
    rectification_padding_lrtb: tuple[int, int, int, int]
    pre_rotation_size_wh: tuple[int, int]
    native_size_wh: tuple[int, int]
    rotated_90: bool
    canonical_enabled: bool
    canonical_geometry: CanonicalGeometry
    expected_output_size_wh: tuple[int, int]
    perspective_interpolation: str


class IntermediatePrepareResult(BaseModel):
    """Result of one successful ffmpeg intermediate preparation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intermediate_path: Path
    ffmpeg_command: list[str]
    filtergraph: str
    filtergraph_metadata: FiltergraphMetadata
    expected_output_size_wh: tuple[int, int]
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int | None = Field(default=None, ge=1)
    return_code: int
    stdout: str
    stderr: str
    elapsed_sec: float = Field(ge=0.0)


def _bundled_binary_candidates(binary_name: str) -> tuple[Path, ...]:
    executable_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
    repository_root = Path(__file__).resolve().parents[2]
    candidates = [repository_root / _BINARY_DIRECTORY_NAME / executable_name]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        candidates.insert(
            0,
            Path(str(frozen_root)) / _BINARY_DIRECTORY_NAME / executable_name,
        )
    return tuple(candidates)


def _resolve_binary(binary_name: str, configured_path: Path | None) -> Path:
    if configured_path is not None:
        explicit_path = Path(configured_path).expanduser().resolve()
        if not explicit_path.is_file():
            raise VideoPreparationError(
                f"Configured {binary_name} executable does not exist: {explicit_path}"
            )
        return explicit_path

    for candidate in _bundled_binary_candidates(binary_name):
        if candidate.is_file():
            return candidate.resolve()

    discovered = shutil.which(binary_name)
    if discovered is None:
        raise VideoPreparationError(
            f"{binary_name} executable was not found in configured, bundled, or PATH locations."
        )
    return Path(discovered).expanduser().resolve()


def resolve_ffmpeg_binary(configured_path: Path | None = None) -> Path:
    """Resolve ffmpeg by explicit path, bundled location, then ``PATH``.

    Raises:
        VideoPreparationError: If an explicit path is invalid or no executable
            can be resolved.
    """

    return _resolve_binary("ffmpeg", configured_path)


def resolve_ffprobe_binary(configured_path: Path | None = None) -> Path:
    """Resolve ffprobe by explicit path, bundled location, then ``PATH``.

    Raises:
        VideoPreparationError: If an explicit path is invalid or no executable
            can be resolved.
    """

    return _resolve_binary("ffprobe", configured_path)


def _validate_trim(
    start_frame: int,
    end_frame_exclusive: int | None,
) -> None:
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise VideoPreparationError("start_frame must be an integer.")
    if start_frame < 0:
        raise VideoPreparationError("start_frame must be non-negative.")
    if end_frame_exclusive is not None:
        if isinstance(end_frame_exclusive, bool) or not isinstance(
            end_frame_exclusive,
            int,
        ):
            raise VideoPreparationError("end_frame_exclusive must be an integer.")
        if end_frame_exclusive <= start_frame:
            raise VideoPreparationError(
                "end_frame_exclusive must be greater than start_frame."
            )


def _transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    transformed = (homography @ homogeneous.T).T
    denominators = transformed[:, 2]
    if np.any(np.abs(denominators) <= np.finfo(np.float64).eps):
        raise VideoPreparationError("CropPlan maps a crop corner to infinity.")
    return transformed[:, :2] / denominators[:, np.newaxis]


def _integer_coordinate(name: str, value: float) -> int:
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=_GEOMETRY_ATOL):
        raise VideoPreparationError(
            f"CropPlan {name} coordinate must resolve to an integer pixel; got {value}."
        )
    return rounded


def _extract_rectification_geometry(
    crop_plan: CropPlan,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int, int, int],
    np.ndarray,
]:
    if crop_plan.native_size_wh is None or crop_plan.canonical_geometry is None:
        raise VideoPreparationError(
            "CropPlan lacks native_size_wh/canonical_geometry required by ffmpeg preparation."
        )

    native_size_wh = crop_plan.native_size_wh
    if crop_plan.rotated_90:
        pre_rotation_size_wh = (native_size_wh[1], native_size_wh[0])
    else:
        pre_rotation_size_wh = native_size_wh
    try:
        rotation, rotated_size_wh, rotated_90 = build_clockwise_rotation_transform(
            pre_rotation_size_wh
        )
    except CropPlanError as exc:
        raise VideoPreparationError(f"CropPlan native rotation is invalid: {exc}") from exc
    if rotated_90 != crop_plan.rotated_90 or rotated_size_wh != native_size_wh:
        raise VideoPreparationError(
            "CropPlan rotation flag and native dimensions are inconsistent."
        )

    canonical = crop_plan.canonical_geometry
    if not math.isfinite(canonical.uniform_scale) or canonical.uniform_scale <= 0:
        raise VideoPreparationError("CropPlan canonical scale must be finite and positive.")
    canonical_transform = np.array(
        [
            [canonical.uniform_scale, 0.0, canonical.padding_left],
            [0.0, canonical.uniform_scale, canonical.padding_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        rectification_homography = (
            np.linalg.inv(rotation)
            @ np.linalg.inv(canonical_transform)
            @ crop_plan.H_raw_to_prepared_3x3
        )
    except np.linalg.LinAlgError as exc:
        raise VideoPreparationError(
            "CropPlan stage geometry contains a singular transform."
        ) from exc

    destination = _transform_points(
        rectification_homography,
        crop_plan.quad_raw_tl_tr_br_bl,
    )
    left = float((destination[0, 0] + destination[3, 0]) / 2.0)
    right = float((destination[1, 0] + destination[2, 0]) / 2.0)
    top = float((destination[0, 1] + destination[1, 1]) / 2.0)
    bottom = float((destination[2, 1] + destination[3, 1]) / 2.0)
    expected_destination = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float64,
    )
    if not np.allclose(
        destination,
        expected_destination,
        rtol=0.0,
        atol=_GEOMETRY_ATOL,
    ):
        raise VideoPreparationError(
            "CropPlan pre-rotation quadrilateral is not an axis-aligned rectangle."
        )

    left_px = _integer_coordinate("rectified left", left)
    right_px = _integer_coordinate("rectified right", right)
    top_px = _integer_coordinate("rectified top", top)
    bottom_px = _integer_coordinate("rectified bottom", bottom)
    pre_width, pre_height = pre_rotation_size_wh
    padding = (
        left_px,
        pre_width - 1 - right_px,
        top_px,
        pre_height - 1 - bottom_px,
    )
    if any(value < 0 for value in padding):
        raise VideoPreparationError(
            "CropPlan rectified quadrilateral lies outside its native frame."
        )
    content_size_wh = (right_px - left_px + 1, bottom_px - top_px + 1)
    if content_size_wh[0] <= 0 or content_size_wh[1] <= 0:
        raise VideoPreparationError("CropPlan rectified content size must be positive.")
    return pre_rotation_size_wh, content_size_wh, padding, destination


def _format_number(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _quad_as_tuple(
    points: np.ndarray,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return (
        (float(points[0, 0]), float(points[0, 1])),
        (float(points[1, 0]), float(points[1, 1])),
        (float(points[2, 0]), float(points[2, 1])),
        (float(points[3, 0]), float(points[3, 1])),
    )


def _validate_canonical_contract(
    crop_plan: CropPlan,
    config: PreprocessConfig,
) -> CanonicalGeometry:
    canonical = crop_plan.canonical_geometry
    native_size_wh = crop_plan.native_size_wh
    if canonical is None or native_size_wh is None:
        raise VideoPreparationError("CropPlan lacks authoritative canonical geometry.")
    if canonical.native_size_wh != native_size_wh:
        raise VideoPreparationError("CropPlan canonical native size is inconsistent.")
    if canonical.canonical_size_wh != crop_plan.prepared_size_wh:
        raise VideoPreparationError("CropPlan canonical final size is inconsistent.")

    canonical_config = config.prepare.canonical_resolution
    if canonical_config.enabled:
        configured_size = (canonical_config.width, canonical_config.height)
        if canonical.canonical_size_wh != configured_size:
            raise VideoPreparationError(
                "CropPlan canonical dimensions do not match preprocessing configuration."
            )
    elif not (
        canonical.uniform_scale == 1.0
        and canonical.scaled_size_wh == native_size_wh
        and canonical.canonical_size_wh == native_size_wh
        and canonical.padding_left == 0
        and canonical.padding_right == 0
        and canonical.padding_top == 0
        and canonical.padding_bottom == 0
        and crop_plan.prepared_size_wh == native_size_wh
    ):
        raise VideoPreparationError(
            "Canonical-disabled CropPlan must use native dimensions without padding."
        )
    return canonical


def build_prepare_filtergraph(
    crop_plan: CropPlan,
    config: PreprocessConfig,
    *,
    start_frame: int = 0,
    end_frame_exclusive: int | None = None,
) -> tuple[str, FiltergraphMetadata]:
    """Build a frame-index trim and CropPlan-authoritative ffmpeg filtergraph.

    ``select`` preserves decode-order identity. ``setpts=PTS-STARTPTS`` only
    shifts timestamps so the intermediate starts at zero; it does not create,
    drop, interpolate, or reorder frames.

    Raises:
        VideoPreparationError: If trim, CropPlan decomposition, or config and
            CropPlan geometry are inconsistent.
    """

    _validate_trim(start_frame, end_frame_exclusive)
    canonical = _validate_canonical_contract(crop_plan, config)
    pre_rotation_size_wh, content_size_wh, rectification_padding, _destination = (
        _extract_rectification_geometry(crop_plan)
    )

    roi = crop_plan.pre_crop_roi
    quad_local = crop_plan.quad_raw_tl_tr_br_bl - np.array(
        [roi.x, roi.y],
        dtype=np.float64,
    )
    tl, tr, br, bl = quad_local
    filters: list[str] = []
    stages: list[str] = []
    if end_frame_exclusive is None:
        filters.append(f"select=gte(n\\,{start_frame})")
    else:
        filters.append(
            f"select=between(n\\,{start_frame}\\,{end_frame_exclusive - 1})"
        )
    stages.append("frame_index_select")
    filters.append("setpts=PTS-STARTPTS")
    stages.append("timestamp_origin_reset")

    filters.append(f"crop={roi.width}:{roi.height}:{roi.x}:{roi.y}")
    stages.append("pre_crop")
    interpolation = config.prepare.perspective_interpolation
    filters.append(
        "perspective="
        f"x0={_format_number(tl[0])}:y0={_format_number(tl[1])}:"
        f"x1={_format_number(tr[0])}:y1={_format_number(tr[1])}:"
        f"x2={_format_number(bl[0])}:y2={_format_number(bl[1])}:"
        f"x3={_format_number(br[0])}:y3={_format_number(br[1])}:"
        f"sense=source:interpolation={interpolation}"
    )
    stages.append("perspective_rectification")

    if (roi.width, roi.height) != content_size_wh:
        rectification_scale_flag = (
            "bicubic" if interpolation == "cubic" else "bilinear"
        )
        filters.append(
            f"scale={content_size_wh[0]}:{content_size_wh[1]}:"
            f"flags={rectification_scale_flag}"
        )
        stages.append("rectified_content_scale")

    pad_left, pad_right, pad_top, pad_bottom = rectification_padding
    if any(rectification_padding):
        filters.append(
            f"pad={pre_rotation_size_wh[0]}:{pre_rotation_size_wh[1]}:"
            f"{pad_left}:{pad_top}:color=black"
        )
        stages.append("rectification_padding")
    if (
        content_size_wh[0] + pad_left + pad_right != pre_rotation_size_wh[0]
        or content_size_wh[1] + pad_top + pad_bottom != pre_rotation_size_wh[1]
    ):
        raise VideoPreparationError(
            "CropPlan rectification content and padding do not fill native dimensions."
        )

    if crop_plan.rotated_90:
        filters.append("transpose=clock")
        stages.append("clockwise_rotation")
    filters.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
    stages.append("even_dimension_guard")

    canonical_config = config.prepare.canonical_resolution
    if canonical_config.enabled:
        scale_flag = _SCALE_FLAG_MAP[canonical_config.flags]
        scaled_width, scaled_height = canonical.scaled_size_wh
        filters.append(f"scale={scaled_width}:{scaled_height}:flags={scale_flag}")
        stages.append("canonical_uniform_scale")
        filters.append(
            f"pad={crop_plan.prepared_size_wh[0]}:{crop_plan.prepared_size_wh[1]}:"
            f"{canonical.padding_left}:{canonical.padding_top}:color=black"
        )
        stages.append("canonical_padding")
    filters.append("setsar=1")
    stages.append("square_sample_aspect_ratio")

    metadata = FiltergraphMetadata(
        stages=tuple(stages),
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        pre_crop_roi_xywh=(roi.x, roi.y, roi.width, roi.height),
        quad_pre_crop_tl_tr_br_bl=_quad_as_tuple(quad_local),
        rectified_content_size_wh=content_size_wh,
        rectification_padding_lrtb=rectification_padding,
        pre_rotation_size_wh=pre_rotation_size_wh,
        native_size_wh=crop_plan.native_size_wh,
        rotated_90=crop_plan.rotated_90,
        canonical_enabled=canonical_config.enabled,
        canonical_geometry=canonical,
        expected_output_size_wh=crop_plan.prepared_size_wh,
        perspective_interpolation=interpolation,
    )
    return ",".join(filters), metadata


def build_ffmpeg_prepare_command(
    raw_video_path: Path,
    intermediate_path: Path,
    crop_plan: CropPlan,
    config: PreprocessConfig,
    start_frame: int,
    end_frame_exclusive: int | None,
    *,
    ffmpeg_binary: Path | None = None,
) -> tuple[list[str], str, FiltergraphMetadata]:
    """Build the Stage A ffmpeg argument list without executing it.

    Raises:
        VideoPreparationError: If binary resolution or filtergraph validation
            fails.
    """

    binary = ffmpeg_binary or resolve_ffmpeg_binary(
        config.encoding.ffmpeg.ffmpeg_path
    )
    filtergraph, metadata = build_prepare_filtergraph(
        crop_plan,
        config,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
    )
    encoding = config.encoding.ffmpeg
    command = [
        str(binary),
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-fflags",
        "+genpts",
        "-i",
        str(Path(raw_video_path)),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filtergraph,
        "-c:v",
        encoding.codec,
        "-pix_fmt",
        encoding.pixel_format,
        "-preset",
        encoding.preset,
        "-crf",
        str(encoding.crf),
        "-bf",
        str(encoding.bframes),
        "-x264-params",
        f"bframes={encoding.bframes}",
        "-enc_time_base",
        "demux",
        "-fps_mode",
        "passthrough",
    ]
    if encoding.faststart:
        command.extend(["-movflags", "+faststart"])
    command.extend(["-f", encoding.container, str(Path(intermediate_path))])
    return command, filtergraph, metadata


def _error_excerpt(stderr: str | None, limit: int = 2000) -> str:
    detail = (stderr or "").strip() or "no diagnostic output"
    return detail[-limit:]


def run_ffmpeg_prepare(
    raw_video_path: Path,
    intermediate_path: Path,
    crop_plan: CropPlan,
    config: PreprocessConfig,
    start_frame: int,
    end_frame_exclusive: int | None,
) -> IntermediatePrepareResult:
    """Run ffmpeg Stage A and return its command, geometry, and diagnostics.

    The destination is an internal intermediate. Existing files are refused;
    overwrite mode is not enabled implicitly.

    Raises:
        VideoPreparationError: If input/output validation, crop acceptance,
            binary resolution, process execution, or ffmpeg itself fails.
    """

    raw_path = Path(raw_video_path).expanduser()
    if not raw_path.is_file():
        raise VideoPreparationError(f"Raw video does not exist: {raw_path}")
    raw_path = raw_path.resolve()
    output_path = Path(intermediate_path).expanduser()
    if output_path.exists():
        raise VideoPreparationError(
            f"Intermediate output already exists; refusing to overwrite: {output_path}"
        )
    if output_path.suffix.lower() != f".{config.encoding.ffmpeg.container}":
        raise VideoPreparationError(
            "Intermediate output suffix does not match configured ffmpeg container."
        )
    if raw_path == output_path.resolve():
        raise VideoPreparationError("Raw input and intermediate output must differ.")
    if not crop_plan.accepted_by_user:
        raise VideoPreparationError(
            "CropPlan must be explicitly accepted before ffmpeg preparation."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    command, filtergraph, metadata = build_ffmpeg_prepare_command(
        raw_path,
        output_path,
        crop_plan,
        config,
        start_frame,
        end_frame_exclusive,
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise VideoPreparationError(f"Could not execute ffmpeg: {exc}") from exc
    elapsed_sec = time.perf_counter() - started
    if completed.returncode != 0:
        rendered_command = subprocess.list2cmdline(command)
        raise VideoPreparationError(
            "ffmpeg preparation failed with exit code "
            f"{completed.returncode}. Command: {rendered_command}. "
            f"Error: {_error_excerpt(completed.stderr)}"
        )
    if not output_path.is_file():
        raise VideoPreparationError(
            "ffmpeg reported success but did not create the intermediate output."
        )

    return IntermediatePrepareResult(
        intermediate_path=output_path,
        ffmpeg_command=list(command),
        filtergraph=filtergraph,
        filtergraph_metadata=metadata,
        expected_output_size_wh=crop_plan.prepared_size_wh,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        return_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        elapsed_sec=elapsed_sec,
    )
