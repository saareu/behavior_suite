"""Nonfatal hydration of compact GUI state from authoritative run metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from preprocess.config import PreCropConfig
from preprocess.metadata import PREPARE_METADATA_SCHEMA_VERSION
from preprocess.models import VideoProbeResult
from preprocess.pre_crop import PreCropMode, PreCropROI, ResolvedPreCrop


@dataclass(frozen=True, slots=True)
class ProjectHydrationResult:
    """Compact prior-run fields safe to place in persistent GUI state."""

    status: str
    message: str
    prior_run_status: str | None = None
    raw_video_path: Path | None = None
    raw_probe: VideoProbeResult | None = None
    raw_count_recorded_by_prior_run: bool = False
    start_frame: int | None = None
    end_frame_exclusive: int | None = None
    pre_crop_config: PreCropConfig | None = None
    resolved_pre_crop: ResolvedPreCrop | None = None


def _integer(value: object, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _same_path(first: Path, second: Path) -> bool:
    return first.expanduser().resolve(strict=False) == second.expanduser().resolve(
        strict=False
    )


def _build_probe(
    raw_video: Mapping[str, object],
    source_path: Path,
    *,
    allow_recorded_count: bool,
) -> VideoProbeResult | None:
    readable_count = _integer(raw_video.get("frame_count_opencv_readable"), minimum=1)
    if not allow_recorded_count:
        readable_count = None
    try:
        return VideoProbeResult(
            source_path=source_path,
            width=raw_video.get("width"),
            height=raw_video.get("height"),
            codec=raw_video.get("codec"),
            pixel_format=raw_video.get("pixel_format"),
            duration_sec=raw_video.get("duration_sec"),
            avg_frame_rate=raw_video.get("avg_frame_rate"),
            r_frame_rate=raw_video.get("r_frame_rate"),
            time_base=raw_video.get("time_base"),
            frame_count_ffprobe=raw_video.get("frame_count_ffprobe"),
            frame_count_opencv_reported=raw_video.get(
                "frame_count_opencv_reported"
            ),
            frame_count_opencv_readable=readable_count,
            opencv_fps=None,
            raw_fps_effective=raw_video.get("fps_effective"),
            raw_fps_effective_method=raw_video.get("fps_effective_method"),
            pts_status=raw_video.get("pts_status", "not_extracted"),
            ffprobe_succeeded=any(
                raw_video.get(field) is not None
                for field in ("avg_frame_rate", "r_frame_rate", "codec")
            ),
        )
    except (ValidationError, TypeError, ValueError):
        return None


def _build_pre_crop(
    pre_crop: Mapping[str, object],
) -> tuple[PreCropConfig | None, ResolvedPreCrop | None]:
    mode_value = pre_crop.get("mode")
    roi_value = pre_crop.get("roi")
    raw_size_value = pre_crop.get("raw_size_wh")
    if (
        not isinstance(mode_value, str)
        or not isinstance(roi_value, Mapping)
        or not isinstance(raw_size_value, list | tuple)
        or len(raw_size_value) != 2
    ):
        return None, None
    try:
        mode = PreCropMode(mode_value)
        roi = PreCropROI.model_validate(roi_value)
        raw_size_wh = (int(raw_size_value[0]), int(raw_size_value[1]))
        if mode is PreCropMode.NONE:
            config = PreCropConfig(enabled=False, mode=mode.value)
        elif mode is PreCropMode.MANUAL_RECTANGLE:
            config = PreCropConfig(
                enabled=True,
                mode=mode.value,
                manual_rectangle=(roi.x, roi.y, roi.width, roi.height),
            )
        elif mode is PreCropMode.VERTICAL_KEEP_LEFT:
            config = PreCropConfig(
                enabled=True,
                mode=mode.value,
                boundary_px=roi.x + roi.width,
            )
        elif mode is PreCropMode.VERTICAL_KEEP_RIGHT:
            config = PreCropConfig(
                enabled=True,
                mode=mode.value,
                boundary_px=roi.x,
            )
        elif mode is PreCropMode.HORIZONTAL_KEEP_UPPER:
            config = PreCropConfig(
                enabled=True,
                mode=mode.value,
                boundary_px=roi.y + roi.height,
            )
        else:
            config = PreCropConfig(
                enabled=True,
                mode=mode.value,
                boundary_px=roi.y,
            )
        resolved = ResolvedPreCrop(mode=mode, roi=roi, raw_size_wh=raw_size_wh)
    except (ValidationError, TypeError, ValueError):
        return None, None
    return config, resolved


def load_project_hydration(preprocess_dir: Path) -> ProjectHydrationResult:
    """Read compact prior-run context without making project opening fail."""

    metadata_path = Path(preprocess_dir) / "prepare_meta.json"
    if not metadata_path.is_file():
        return ProjectHydrationResult(
            status="metadata_absent",
            message=(
                "Project opened. No prior prepare_meta.json was found; select and "
                "probe a raw video."
            ),
        )
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ProjectHydrationResult(
            status="metadata_invalid",
            message=f"Project opened, but prepare_meta.json could not be read: {exc}",
        )
    if not isinstance(metadata, Mapping):
        return ProjectHydrationResult(
            status="metadata_invalid",
            message="Project opened, but prepare_meta.json is not a JSON object.",
        )
    raw_video = metadata.get("raw_video")
    if not isinstance(raw_video, Mapping):
        return ProjectHydrationResult(
            status="metadata_incomplete",
            message=(
                "Project opened, but prepare_meta.json has no usable raw-video section."
            ),
        )
    source_value = raw_video.get("source_path")
    if not isinstance(source_value, str) or not source_value.strip():
        return ProjectHydrationResult(
            status="metadata_incomplete",
            message=(
                "Project opened, but prepare_meta.json has no usable raw-video path."
            ),
        )
    raw_video_path = Path(source_value).expanduser()
    validation = metadata.get("validation")
    prior_run_status = (
        validation.get("status") if isinstance(validation, Mapping) else None
    )
    completed_run = (
        metadata.get("schema_version") == PREPARE_METADATA_SCHEMA_VERSION
        and prior_run_status == "passed"
    )
    probe = _build_probe(
        raw_video,
        raw_video_path,
        allow_recorded_count=completed_run,
    )
    recorded_count = (
        completed_run
        and probe is not None
        and probe.frame_count_opencv_readable is not None
        and probe.frame_count_opencv_readable > 0
        and _same_path(raw_video_path, probe.source_path)
    )

    trim = metadata.get("trim")
    start_frame = None
    end_frame_exclusive = None
    if isinstance(trim, Mapping):
        start_frame = _integer(trim.get("start_frame"))
        end_value = trim.get("end_frame_exclusive")
        end_frame_exclusive = (
            None if end_value is None else _integer(end_value, minimum=1)
        )
    pre_crop_value = metadata.get("pre_crop")
    pre_crop_config, resolved_pre_crop = (
        _build_pre_crop(pre_crop_value)
        if isinstance(pre_crop_value, Mapping)
        else (None, None)
    )

    if probe is None:
        return ProjectHydrationResult(
            status="metadata_partial",
            message=(
                "Project opened and its raw-video path was restored, but prior probe "
                "details were incomplete; probe the video before continuing."
            ),
            prior_run_status=(
                str(prior_run_status) if prior_run_status is not None else None
            ),
            raw_video_path=raw_video_path,
            start_frame=start_frame,
            end_frame_exclusive=end_frame_exclusive,
            pre_crop_config=pre_crop_config,
            resolved_pre_crop=resolved_pre_crop,
        )
    count_message = (
        " Prior raw sequential count was restored with prior-run provenance."
        if recorded_count
        else " No prior raw sequential readable-frame count was recorded."
    )
    return ProjectHydrationResult(
        status="metadata_loaded",
        message="Project opened and prior preprocessing context was restored."
        + count_message,
        prior_run_status=(
            str(prior_run_status) if prior_run_status is not None else None
        ),
        raw_video_path=raw_video_path,
        raw_probe=probe,
        raw_count_recorded_by_prior_run=recorded_count,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        pre_crop_config=pre_crop_config,
        resolved_pre_crop=resolved_pre_crop,
    )
