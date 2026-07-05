"""Two-stage video preparation with deterministic frames and crop geometry."""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from preprocess.config import PreprocessConfig
from preprocess.crop_plan import (
    PERSPECTIVE_REPROJECTION_ATOL_PX,
    CanonicalGeometry,
    CropPlan,
    build_clockwise_rotation_transform,
)
from preprocess.exceptions import (
    CropPlanError,
    FFmpegRuntimeError,
    PreprocessCancelledError,
    VideoPreparationError,
    VideoValidationError,
)
from preprocess.ffmpeg_runtime import (
    ensure_supported_ffmpeg_runtime,
)
from preprocess.ffmpeg_runtime import (
    resolve_ffmpeg_binary as _resolve_ffmpeg_binary,
)
from preprocess.ffmpeg_runtime import (
    resolve_ffprobe_binary as _resolve_ffprobe_binary,
)
from preprocess.masking import apply_static_mask_to_frame, validate_static_mask
from preprocess.models import OperationProgress
from preprocess.validation import (
    PreparedVideoValidationResult,
    validate_prepared_video,
)

# Stage A decomposes a composed float64 CropPlan transform. This local integer
# tolerance admits only floating-point solve/composition noise; it is not a
# tolerance on the user's raw quadrilateral or a license to alter crop geometry.
_INTEGER_COORDINATE_ATOL_PX = 1e-5
_SCALE_FLAG_MAP = {
    "linear": "bilinear",
    "cubic": "bicubic",
    "lanczos": "lanczos",
}
_PROGRESS_FRAME_INTERVAL = 250
_PROGRESS_TIME_INTERVAL_SEC = 0.25
_FFMPEG_TERMINATE_TIMEOUT_SEC = 2.0


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
    frames_processed: int | None = Field(default=None, ge=0)
    processed_video_time_sec: float | None = Field(default=None, ge=0.0)


class FinalReencodeResult(BaseModel):
    """Result of validated Stage B OpenCV final re-encoding.

    The temporary path records the same-directory staging name used before its
    successful atomic replacement of the prepared path; it no longer exists
    after a successful return.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temporary_output_path: Path
    prepared_video_path: Path
    frames_written: int = Field(gt=0)
    output_fps: float = Field(gt=0)
    output_size_wh: tuple[int, int]
    opencv_fourcc: str
    elapsed_sec: float = Field(ge=0.0)
    validation_result: PreparedVideoValidationResult


def resolve_ffmpeg_binary(configured_path: Path | None = None) -> Path:
    """Resolve ffmpeg through the shared runtime resolver.

    Raises:
        VideoPreparationError: If no executable can be resolved.
    """

    try:
        return _resolve_ffmpeg_binary(configured_path)
    except FFmpegRuntimeError as exc:
        raise VideoPreparationError(str(exc)) from exc


def resolve_ffprobe_binary(configured_path: Path | None = None) -> Path:
    """Resolve ffprobe through the shared runtime resolver.

    Raises:
        VideoPreparationError: If no executable can be resolved.
    """

    try:
        return _resolve_ffprobe_binary(configured_path)
    except FFmpegRuntimeError as exc:
        raise VideoPreparationError(str(exc)) from exc


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
    if not math.isclose(
        value,
        rounded,
        rel_tol=0.0,
        abs_tol=_INTEGER_COORDINATE_ATOL_PX,
    ):
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
    np.ndarray,
]:
    if crop_plan.native_size_wh is None or crop_plan.canonical_geometry is None:
        raise VideoPreparationError(
            "CropPlan lacks native_size_wh/canonical_geometry required by ffmpeg preparation."
        )

    native_size_wh = crop_plan.native_size_wh
    if crop_plan.rotated_90:
        pre_rotation_size_wh = (native_size_wh[1], native_size_wh[0])
        try:
            rotation, rotated_size_wh, rotated_90 = build_clockwise_rotation_transform(
                pre_rotation_size_wh
            )
        except CropPlanError as exc:
            raise VideoPreparationError(
                f"CropPlan native rotation is invalid: {exc}"
            ) from exc
        if not rotated_90 or rotated_size_wh != native_size_wh:
            raise VideoPreparationError(
                "CropPlan rotation flag and native dimensions are inconsistent."
            )
    else:
        pre_rotation_size_wh = native_size_wh
        rotation = np.eye(3, dtype=np.float64)

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
        atol=PERSPECTIVE_REPROJECTION_ATOL_PX,
    ):
        maximum_error = float(np.max(np.abs(destination - expected_destination)))
        raise VideoPreparationError(
            "CropPlan homography cannot be reproduced by the Stage A "
            "rectification path: transformed crop corners differ from the "
            f"expected rectified rectangle by up to {maximum_error:.9g} pixels."
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
    return (
        pre_rotation_size_wh,
        content_size_wh,
        padding,
        destination,
        rectification_homography,
    )


def _build_perspective_source_quad(
    crop_plan: CropPlan,
    rectification_homography: np.ndarray,
    content_size_wh: tuple[int, int],
    rectification_padding: tuple[int, int, int, int],
) -> np.ndarray:
    """Adapt CropPlan pixel centers to ffmpeg's W/H corner convention."""

    roi = crop_plan.pre_crop_roi
    content_width, content_height = content_size_wh
    input_width, input_height = roi.width, roi.height
    scale_x = content_width / input_width
    scale_y = content_height / input_height
    pixel_center_scale = np.array(
        [
            [scale_x, 0.0, scale_x / 2.0 - 0.5],
            [0.0, scale_y, scale_y / 2.0 - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    raw_from_local = np.array(
        [
            [1.0, 0.0, roi.x],
            [0.0, 1.0, roi.y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pad_left, _pad_right, pad_top, _pad_bottom = rectification_padding
    remove_padding = np.array(
        [
            [1.0, 0.0, -pad_left],
            [0.0, 1.0, -pad_top],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    desired_local_to_content = (
        remove_padding @ rectification_homography @ raw_from_local
    )
    try:
        desired_local_to_perspective = (
            np.linalg.inv(pixel_center_scale) @ desired_local_to_content
        )
        perspective_to_local = np.linalg.inv(desired_local_to_perspective)
    except np.linalg.LinAlgError as exc:
        raise VideoPreparationError(
            "CropPlan perspective stage contains a singular transform."
        ) from exc

    perspective_boundary = np.array(
        [
            [0.0, 0.0],
            [float(input_width), 0.0],
            [float(input_width), float(input_height)],
            [0.0, float(input_height)],
        ],
        dtype=np.float64,
    )
    source_quad = _transform_points(perspective_to_local, perspective_boundary)
    if not np.all(np.isfinite(source_quad)):
        raise VideoPreparationError(
            "CropPlan perspective source coordinates are non-finite."
        )
    return source_quad


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
    (
        pre_rotation_size_wh,
        content_size_wh,
        rectification_padding,
        _destination,
        rectification_homography,
    ) = _extract_rectification_geometry(crop_plan)

    roi = crop_plan.pre_crop_roi
    quad_local = crop_plan.quad_raw_tl_tr_br_bl - np.array(
        [roi.x, roi.y],
        dtype=np.float64,
    )
    perspective_source_quad = _build_perspective_source_quad(
        crop_plan,
        rectification_homography,
        content_size_wh,
        rectification_padding,
    )
    tl, tr, br, bl = perspective_source_quad
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
        "-progress",
        "pipe:1",
        "-nostats",
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


def _ffmpeg_time_seconds(progress: dict[str, str]) -> float | None:
    for key in ("out_time_us", "out_time_ms"):
        value = progress.get(key)
        if value is not None:
            try:
                microseconds = int(value)
            except ValueError:
                continue
            if microseconds >= 0:
                return microseconds / 1_000_000.0
    value = progress.get("out_time")
    if value is None:
        return None
    try:
        hours_text, minutes_text, seconds_text = value.split(":", maxsplit=2)
        seconds = (
            int(hours_text) * 3600
            + int(minutes_text) * 60
            + float(seconds_text)
        )
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _ffmpeg_frame_count(progress: dict[str, str]) -> int | None:
    value = progress.get("frame")
    if value is None:
        return None
    try:
        count = int(value)
    except ValueError:
        return None
    return count if count >= 0 else None


def _terminate_ffmpeg(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_FFMPEG_TERMINATE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_FFMPEG_TERMINATE_TIMEOUT_SEC)


def _drain_text_stream(
    stream_name: str,
    stream,
    events: queue.Queue[tuple[str, str | None]],
) -> None:
    try:
        for line in stream:
            events.put((stream_name, line))
    finally:
        events.put((stream_name, None))


def run_ffmpeg_prepare(
    raw_video_path: Path,
    intermediate_path: Path,
    crop_plan: CropPlan,
    config: PreprocessConfig,
    start_frame: int,
    end_frame_exclusive: int | None,
    *,
    expected_frame_count: int | None = None,
    progress_callback: Callable[[OperationProgress], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
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
    try:
        runtime = ensure_supported_ffmpeg_runtime(
            ffmpeg_path=config.encoding.ffmpeg.ffmpeg_path,
            ffprobe_path=config.encoding.ffmpeg.ffprobe_path,
        )
    except FFmpegRuntimeError as exc:
        raise VideoPreparationError(str(exc)) from exc
    assert runtime.ffmpeg.path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    command, filtergraph, metadata = build_ffmpeg_prepare_command(
        raw_path,
        output_path,
        crop_plan,
        config,
        start_frame,
        end_frame_exclusive,
        ffmpeg_binary=runtime.ffmpeg.path,
    )
    if expected_frame_count is not None and (
        isinstance(expected_frame_count, bool)
        or not isinstance(expected_frame_count, int)
        or expected_frame_count <= 0
    ):
        raise VideoPreparationError(
            "expected_frame_count must be a positive integer when supplied."
        )
    if cancellation_requested is not None and cancellation_requested():
        raise PreprocessCancelledError(
            "Preprocessing cancelled during Stage A ffmpeg preparation."
        )

    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    latest_progress: dict[str, str] = {}
    frames_processed: int | None = None
    processed_video_time_sec: float | None = None

    def emit() -> None:
        if progress_callback is None:
            return
        message_parts: list[str] = []
        if frames_processed is not None:
            if expected_frame_count is None:
                message_parts.append(f"{frames_processed:,} frames processed")
            else:
                message_parts.append(
                    f"{frames_processed:,} / {expected_frame_count:,} frames processed"
                )
        progress_callback(
            OperationProgress(
                phase="Stage A: preparing video geometry",
                message="; ".join(message_parts) or "ffmpeg is processing video geometry",
                completed_units=frames_processed,
                total_units=(
                    expected_frame_count if frames_processed is not None else None
                ),
                is_indeterminate=(
                    frames_processed is None or expected_frame_count is None
                ),
                processed_video_time_sec=processed_video_time_sec,
            )
        )

    emit()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise VideoPreparationError(f"Could not execute ffmpeg: {exc}") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    readers = (
        threading.Thread(
            target=_drain_text_stream,
            args=("stdout", process.stdout, events),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_text_stream,
            args=("stderr", process.stderr, events),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    closed_streams: set[str] = set()
    try:
        while process.poll() is None or len(closed_streams) < 2:
            if cancellation_requested is not None and cancellation_requested():
                _terminate_ffmpeg(process)
                raise PreprocessCancelledError(
                    "Preprocessing cancelled during Stage A ffmpeg preparation."
                )
            try:
                stream_name, line = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                closed_streams.add(stream_name)
                continue
            if stream_name == "stderr":
                stderr_lines.append(line)
                continue
            stdout_lines.append(line)
            text = line.strip()
            if "=" not in text:
                continue
            key, value = text.split("=", maxsplit=1)
            latest_progress[key] = value
            if key != "progress":
                continue
            frames_processed = _ffmpeg_frame_count(latest_progress)
            processed_video_time_sec = _ffmpeg_time_seconds(latest_progress)
            emit()
            latest_progress.clear()
    except BaseException:
        _terminate_ffmpeg(process)
        output_path.unlink(missing_ok=True)
        raise
    finally:
        for reader in readers:
            reader.join(timeout=1.0)
        process.stdout.close()
        process.stderr.close()

    elapsed_sec = time.perf_counter() - started
    return_code = process.wait()
    if return_code != 0:
        output_path.unlink(missing_ok=True)
        rendered_command = subprocess.list2cmdline(command)
        raise VideoPreparationError(
            "ffmpeg preparation failed with exit code "
            f"{return_code}. Command: {rendered_command}. "
            f"Error: {_error_excerpt(''.join(stderr_lines))}"
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
        return_code=return_code,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
        elapsed_sec=elapsed_sec,
        frames_processed=frames_processed,
        processed_video_time_sec=processed_video_time_sec,
    )


def _validate_stage_b_geometry(
    output_size_wh: tuple[int, int],
) -> tuple[int, int]:
    try:
        width, height = output_size_wh
    except (TypeError, ValueError) as exc:
        raise VideoPreparationError(
            "output_size_wh must contain width and height."
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (width, height)
    ):
        raise VideoPreparationError("Stage B output width and height must be integers.")
    if width <= 0 or height <= 0:
        raise VideoPreparationError("Stage B output width and height must be positive.")
    if width % 2 or height % 2:
        raise VideoPreparationError("Stage B output width and height must be even.")
    return width, height


def _validate_stage_b_fps(output_fps: float) -> float:
    if isinstance(output_fps, bool):
        raise VideoPreparationError("Stage B output FPS must be numeric.")
    try:
        fps = float(output_fps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VideoPreparationError("Stage B output FPS must be numeric.") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise VideoPreparationError("Stage B output FPS must be finite and positive.")
    return fps


def _create_temporary_video_path(prepared_path: Path) -> Path:
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{prepared_path.stem}.stage-b-",
            suffix=prepared_path.suffix,
            dir=prepared_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name).resolve()
        temporary_path.unlink()
    except OSError as exc:
        raise VideoPreparationError(
            f"Could not create Stage B temporary output path: {exc}"
        ) from exc
    return temporary_path


def _remove_temporary_video(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise VideoPreparationError(
            f"Could not remove failed Stage B temporary output {path}: {exc}"
        ) from exc


def reencode_intermediate_with_opencv(
    intermediate_video_path: Path,
    prepared_video_path: Path,
    output_fps: float,
    output_size_wh: tuple[int, int],
    config: PreprocessConfig,
    *,
    expected_frame_count: int | None = None,
    progress_callback: Callable[[OperationProgress], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> FinalReencodeResult:
    """Sequentially re-encode an ffmpeg intermediate through OpenCV.

    Exactly one output frame is written for each successful sequential input
    decode. No resize, frame-rate conversion, temporal resampling, or geometry
    transform is performed. An enabled static prepared-coordinate mask is
    applied after all Stage A spatial geometry and before final writing. The
    output is written beside the
    requested destination under a temporary name, strictly validated, and
    atomically moved into place only after validation succeeds.

    Raises:
        VideoPreparationError: If arguments, input, frame geometry, writer
            creation, encoding, temporary-file handling, or replacement fails.
        VideoValidationError: If the completed temporary video fails any
            strict prepared-video validation gate.
    """

    fps = _validate_stage_b_fps(output_fps)
    expected_size = _validate_stage_b_geometry(output_size_wh)
    try:
        validate_static_mask(config.mask, expected_size)
    except VideoPreparationError as exc:
        raise VideoPreparationError(f"Static mask configuration is invalid: {exc}") from exc
    if expected_frame_count is not None and (
        isinstance(expected_frame_count, bool)
        or not isinstance(expected_frame_count, int)
        or expected_frame_count <= 0
    ):
        raise VideoPreparationError(
            "expected_frame_count must be a positive integer when supplied."
        )
    if cancellation_requested is not None and cancellation_requested():
        raise PreprocessCancelledError(
            "Preprocessing cancelled during Stage B OpenCV re-encoding."
        )
    intermediate_path = Path(intermediate_video_path).expanduser()
    if not intermediate_path.is_file():
        raise VideoPreparationError(
            f"ffmpeg intermediate does not exist: {intermediate_path}"
        )
    intermediate_path = intermediate_path.resolve()

    prepared_path = Path(prepared_video_path).expanduser()
    encoding = config.encoding.opencv
    if prepared_path.suffix.lower() != f".{encoding.container}":
        raise VideoPreparationError(
            "Prepared output suffix does not match configured OpenCV container."
        )
    try:
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_path = prepared_path.resolve()
    except OSError as exc:
        raise VideoPreparationError(
            f"Could not create prepared-video output directory: {exc}"
        ) from exc
    if intermediate_path == prepared_path:
        raise VideoPreparationError(
            "ffmpeg intermediate and prepared-video output paths must differ."
        )

    temporary_path = _create_temporary_video_path(prepared_path)
    started = time.perf_counter()
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    last_emitted_count = 0
    last_emitted_at = time.perf_counter()

    def emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="Stage B: writing SLEAP-compatible video",
                    message=message,
                    completed_units=frames_written,
                    total_units=expected_frame_count,
                    is_indeterminate=expected_frame_count is None,
                )
            )

    if expected_frame_count is None:
        emit("0 frames written")
    else:
        emit(f"0 / {expected_frame_count:,} frames written")
    try:
        capture = cv2.VideoCapture(str(intermediate_path))
        if not capture.isOpened():
            raise VideoPreparationError(
                f"OpenCV could not open ffmpeg intermediate: {intermediate_path}"
            )

        fourcc = cv2.VideoWriter_fourcc(*encoding.fourcc)
        writer = cv2.VideoWriter(
            str(temporary_path),
            fourcc,
            fps,
            expected_size,
        )
        if not writer.isOpened():
            raise VideoPreparationError(
                "OpenCV could not open VideoWriter for the Stage B temporary output."
            )

        while True:
            if cancellation_requested is not None and cancellation_requested():
                raise PreprocessCancelledError(
                    "Preprocessing cancelled during Stage B OpenCV re-encoding."
                )
            try:
                success, frame = capture.read()
            except cv2.error as exc:
                raise VideoPreparationError(
                    f"OpenCV failed while decoding the ffmpeg intermediate: {exc}"
                ) from exc
            if not success:
                break
            if frame is None or frame.ndim < 2:
                raise VideoPreparationError(
                    "OpenCV reported a successful intermediate decode without a valid frame."
                )
            frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            if frame_size != expected_size:
                raise VideoPreparationError(
                    f"Intermediate frame {frames_written} has size {frame_size}; "
                    f"expected {expected_size}. Stage B does not resize frames."
                )
            try:
                frame = apply_static_mask_to_frame(frame, config.mask)
            except VideoPreparationError as exc:
                raise VideoPreparationError(
                    f"Could not apply static mask to prepared frame {frames_written}: {exc}"
                ) from exc
            try:
                writer.write(frame)
            except cv2.error as exc:
                raise VideoPreparationError(
                    f"OpenCV failed while writing prepared frame {frames_written}: {exc}"
                ) from exc
            frames_written += 1
            now = time.perf_counter()
            if (
                frames_written - last_emitted_count >= _PROGRESS_FRAME_INTERVAL
                or now - last_emitted_at >= _PROGRESS_TIME_INTERVAL_SEC
            ):
                if expected_frame_count is None:
                    message = f"{frames_written:,} frames written"
                else:
                    message = (
                        f"{frames_written:,} / {expected_frame_count:,} frames written"
                    )
                emit(message)
                last_emitted_count = frames_written
                last_emitted_at = now
    except cv2.error as exc:
        raise VideoPreparationError(
            f"OpenCV could not initialize or complete Stage B re-encoding: {exc}"
        ) from exc
    finally:
        failed = sys.exc_info()[0] is not None
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if failed:
            _remove_temporary_video(temporary_path)

    try:
        if frames_written <= 0:
            raise VideoPreparationError(
                "No frames were written during Stage B OpenCV re-encoding."
            )
        if (
            expected_frame_count is not None
            and frames_written != expected_frame_count
        ):
            raise VideoPreparationError(
                f"Stage B wrote {frames_written} frames; expected "
                f"{expected_frame_count}."
            )
        validation_arguments = {
            "expected_frame_count": frames_written,
            "expected_size_wh": expected_size,
        }
        if cancellation_requested is not None:
            validation_arguments["cancellation_requested"] = cancellation_requested
        validation_result = validate_prepared_video(
            temporary_path,
            **validation_arguments,
        )
        if cancellation_requested is not None and cancellation_requested():
            raise PreprocessCancelledError(
                "Preprocessing cancelled during Stage B OpenCV re-encoding."
            )
        try:
            temporary_path.replace(prepared_path)
        except OSError as exc:
            raise VideoPreparationError(
                f"Could not replace prepared-video output atomically: {exc}"
            ) from exc
    except (PreprocessCancelledError, VideoPreparationError, VideoValidationError):
        _remove_temporary_video(temporary_path)
        raise

    if expected_frame_count is None:
        emit(f"{frames_written:,} frames written")
    else:
        emit(f"{frames_written:,} / {expected_frame_count:,} frames written")

    return FinalReencodeResult(
        temporary_output_path=temporary_path,
        prepared_video_path=prepared_path,
        frames_written=frames_written,
        output_fps=fps,
        output_size_wh=expected_size,
        opencv_fourcc=encoding.fourcc,
        elapsed_sec=time.perf_counter() - started,
        validation_result=validation_result,
    )
