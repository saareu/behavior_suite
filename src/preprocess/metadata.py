"""Authoritative preprocess metadata and accepted-settings artifact writers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml

from preprocess.background import BackgroundEstimateResult, validate_background
from preprocess.config import MaskConfig, PreprocessConfig
from preprocess.crop_plan import CropMode, CropPlan
from preprocess.exceptions import (
    BackgroundGenerationError,
    MetadataArtifactError,
    VideoPreparationError,
)
from preprocess.masking import serialize_static_mask, validate_static_mask
from preprocess.models import (
    ExternalTimeSelection,
    FPSHeaderSource,
    PreprocessOutputs,
    SoftwareEnvironmentInfo,
    VideoProbeResult,
)
from preprocess.sync_writer import PREPARED_SYNC_SCHEMA_VERSION
from preprocess.validation import PreparedVideoValidationResult
from project.models import Project

PREPARE_METADATA_SCHEMA_VERSION = "prepare_meta_v1"
PREPARE_METADATA_TOP_LEVEL_SECTIONS = (
    "schema_version",
    "project",
    "raw_video",
    "trim",
    "pre_crop",
    "external_time",
    "cage_detection",
    "manual_crop",
    "geometry",
    "encoding",
    "prepared_video",
    "background",
    "sync",
    "mask",
    "validation",
    "software_environment",
    "outputs",
)

_REQUIRED_NESTED_FIELDS: dict[str, set[str]] = {
    "project": {"project_name", "project_dir", "preprocess_dir"},
    "raw_video": {
        "source_path",
        "width",
        "height",
        "codec",
        "pixel_format",
        "duration_sec",
        "avg_frame_rate",
        "r_frame_rate",
        "time_base",
        "frame_count_ffprobe",
        "frame_count_opencv_reported",
        "frame_count_opencv_readable",
        "fps_effective",
        "fps_effective_method",
        "pts_status",
    },
    "trim": {
        "start_frame",
        "end_frame_exclusive",
        "end_frame_semantics",
        "expected_trimmed_frame_count",
    },
    "pre_crop": {"enabled", "mode", "roi", "raw_size_wh"},
    "external_time": {
        "provided",
        "source_path",
        "selected_variable",
        "selected_units",
        "raw_vector_length",
        "raw_video_frame_count_opencv_readable",
        "median_dt",
        "estimated_fps",
        "validation_status",
    },
    "cage_detection": {"used"},
    "manual_crop": {"used"},
    "geometry": {
        "mode",
        "pre_crop_roi",
        "quad_raw_tl_tr_br_bl",
        "H_raw_to_prepared_3x3",
        "H_prepared_to_raw_3x3",
        "prepared_size_wh",
        "rotated_90",
        "fit_score",
        "rim_density",
        "accepted_by_user",
        "native_geometry",
        "canonical_geometry",
    },
    "encoding": {
        "ffmpeg",
        "opencv",
        "output_fps_header",
        "fps_header_source",
        "raw_fps_effective",
        "raw_fps_effective_method",
        "external_fps_effective",
        "external_fps_effective_method",
    },
    "prepared_video": {
        "path",
        "width",
        "height",
        "fps_header",
        "opencv_reported_frame_count",
        "opencv_readable_frame_count",
        "frame_count_used_for_sleap",
    },
    "background": {
        "path",
        "method",
        "sampled_frame_count",
        "sampled_frame_indices",
        "source_video_path",
        "source_size_wh",
        "output_dtype",
    },
    "sync": {"path", "sync_schema_version", "frame_count_used_for_sleap"},
    "mask": {"enabled", "coordinate_space", "fill_value", "shapes"},
    "validation": {"status", "prepared_video", "background"},
    "software_environment": {
        "python_version",
        "platform",
        "opencv_version",
        "numpy_version",
        "ffmpeg_version",
        "ffprobe_version",
        "application_version",
    },
    "outputs": {
        "prepared_video_path",
        "prepare_meta_path",
        "prepared_sync_path",
        "cropped_background_path",
        "settings_used_path",
        "processing_log_path",
    },
}


def _path_text(path: Path) -> str:
    return str(Path(path).expanduser())


def _expected_trimmed_count(
    start_frame: int,
    end_frame_exclusive: int | None,
    completed_readable_count: int,
) -> int:
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise MetadataArtifactError("start_frame must be a non-negative integer.")
    if end_frame_exclusive is not None:
        if (
            isinstance(end_frame_exclusive, bool)
            or not isinstance(end_frame_exclusive, int)
            or end_frame_exclusive <= start_frame
        ):
            raise MetadataArtifactError(
                "end_frame_exclusive must be an integer greater than start_frame."
            )
        return end_frame_exclusive - start_frame
    if completed_readable_count <= 0:
        raise MetadataArtifactError("Completed prepared readable frame count must be positive.")
    return completed_readable_count


def _external_time_metadata(
    selection: ExternalTimeSelection | None,
    raw_probe: VideoProbeResult,
    validation_status: str | None,
    external_fps_effective: float | None,
) -> dict[str, object]:
    if selection is None or not selection.provided:
        return {
            "provided": False,
            "source_path": None,
            "selected_variable": None,
            "selected_units": None,
            "raw_vector_length": 0,
            "raw_video_frame_count_opencv_readable": (raw_probe.frame_count_opencv_readable),
            "median_dt": None,
            "estimated_fps": None,
            "validation_status": validation_status or "not_provided",
        }
    return {
        "provided": True,
        "source_path": (
            _path_text(selection.source_path) if selection.source_path is not None else None
        ),
        "selected_variable": selection.selected_variable,
        "selected_units": (
            selection.declared_units.value if selection.declared_units is not None else None
        ),
        "raw_vector_length": selection.raw_vector_length,
        "raw_video_frame_count_opencv_readable": (selection.raw_video_frame_count_opencv_readable),
        "median_dt": selection.median_difference,
        "estimated_fps": external_fps_effective,
        "validation_status": validation_status or selection.validation_status,
    }


def _geometry_metadata(
    crop_plan: CropPlan,
    preprocess_config: PreprocessConfig,
) -> dict[str, object]:
    serialized = crop_plan.to_metadata_dict()
    canonical = crop_plan.canonical_geometry
    return {
        "mode": crop_plan.mode.value,
        "pre_crop_roi": crop_plan.pre_crop_roi.to_metadata_dict(),
        "quad_raw_tl_tr_br_bl": serialized["quad_raw_tl_tr_br_bl"],
        "H_raw_to_prepared_3x3": serialized["H_raw_to_prepared_3x3"],
        "H_prepared_to_raw_3x3": serialized["H_prepared_to_raw_3x3"],
        "prepared_size_wh": list(crop_plan.prepared_size_wh),
        "rotated_90": crop_plan.rotated_90,
        "fit_score": crop_plan.fit_score,
        "rim_density": crop_plan.rim_density,
        "accepted_by_user": crop_plan.accepted_by_user,
        "native_geometry": {
            "size_wh": (
                list(crop_plan.native_size_wh) if crop_plan.native_size_wh is not None else None
            )
        },
        "canonical_geometry": (
            {
                "enabled": preprocess_config.prepare.canonical_resolution.enabled,
                **canonical.to_metadata_dict(),
            }
            if canonical is not None
            else None
        ),
    }


def build_prepare_metadata(
    *,
    project: Project,
    raw_probe: VideoProbeResult,
    prepared_validation: PreparedVideoValidationResult,
    crop_plan: CropPlan,
    preprocess_config: PreprocessConfig,
    external_time_selection: ExternalTimeSelection | None,
    sync_path: Path,
    background_result: BackgroundEstimateResult,
    outputs: PreprocessOutputs,
    start_frame: int,
    end_frame_exclusive: int | None,
    software_environment: SoftwareEnvironmentInfo,
    detector_diagnostics: dict[str, object] | None = None,
    fps_header: float | None = None,
    fps_header_source: FPSHeaderSource | None = None,
    raw_fps_effective: float | None = None,
    raw_fps_effective_method: str | None = None,
    external_fps_effective: float | None = None,
    external_fps_effective_method: str | None = None,
    external_time_status: str | None = None,
) -> dict[str, object]:
    """Build complete JSON-safe metadata from already-computed typed results.

    Raises:
        MetadataArtifactError: If supplied paths, counts, geometry, validation,
            or background provenance are inconsistent.
    """

    expected_count = _expected_trimmed_count(
        start_frame,
        end_frame_exclusive,
        prepared_validation.opencv_readable_frame_count,
    )
    sync_file = Path(sync_path).expanduser()
    if sync_file.resolve() != Path(outputs.prepared_sync_path).expanduser().resolve():
        raise MetadataArtifactError("sync_path must match outputs.prepared_sync_path.")
    prepared_path = Path(outputs.prepared_video_path).expanduser().resolve()
    if Path(background_result.source_video_path).expanduser().resolve() != prepared_path:
        raise MetadataArtifactError("Background source must be the final prepared video output.")
    try:
        validate_background(
            background_result.background,
            prepared_validation.opencv_reported_size_wh,
        )
    except BackgroundGenerationError as exc:
        raise MetadataArtifactError(f"Background result is invalid: {exc}") from exc

    prepared_width, prepared_height = prepared_validation.opencv_reported_size_wh
    output_fps = prepared_validation.opencv_reported_fps if fps_header is None else fps_header
    resolved_fps_source = fps_header_source
    if resolved_fps_source is None:
        raw_method = raw_probe.raw_fps_effective_method
        resolved_fps_source = "opencv_reported_fps" if raw_method == "opencv_fps" else raw_method
    if resolved_fps_source not in {
        "external_time",
        "ffprobe_avg_frame_rate",
        "ffprobe_r_frame_rate",
        "opencv_reported_fps",
    }:
        raise MetadataArtifactError("fps_header_source is missing or invalid.")
    resolved_raw_fps = (
        raw_probe.raw_fps_effective if raw_fps_effective is None else raw_fps_effective
    )
    resolved_raw_fps_method = (
        raw_probe.raw_fps_effective_method
        if raw_fps_effective_method is None
        else raw_fps_effective_method
    )
    validation_status = "passed" if prepared_validation.is_valid else "failed"
    geometry = _geometry_metadata(crop_plan, preprocess_config)
    is_automatic = crop_plan.mode is CropMode.AUTOMATIC
    ffmpeg_config = preprocess_config.encoding.ffmpeg
    opencv_config = preprocess_config.encoding.opencv
    metadata: dict[str, object] = {
        "schema_version": PREPARE_METADATA_SCHEMA_VERSION,
        "project": {
            "project_name": project.name,
            "project_dir": _path_text(project.root_dir),
            "preprocess_dir": _path_text(outputs.preprocess_dir),
        },
        "raw_video": {
            "source_path": _path_text(raw_probe.source_path),
            "width": raw_probe.width,
            "height": raw_probe.height,
            "codec": raw_probe.codec,
            "pixel_format": raw_probe.pixel_format,
            "duration_sec": raw_probe.duration_sec,
            "avg_frame_rate": raw_probe.avg_frame_rate,
            "r_frame_rate": raw_probe.r_frame_rate,
            "time_base": raw_probe.time_base,
            "frame_count_ffprobe": raw_probe.frame_count_ffprobe,
            "frame_count_opencv_reported": raw_probe.frame_count_opencv_reported,
            "frame_count_opencv_readable": raw_probe.frame_count_opencv_readable,
            "fps_effective": raw_probe.raw_fps_effective,
            "fps_effective_method": raw_probe.raw_fps_effective_method,
            "pts_status": raw_probe.pts_status,
        },
        "trim": {
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame_exclusive,
            "end_frame_semantics": "exclusive",
            "expected_trimmed_frame_count": expected_count,
        },
        "pre_crop": {
            "enabled": preprocess_config.pre_crop.enabled,
            "mode": preprocess_config.pre_crop.mode,
            "roi": crop_plan.pre_crop_roi.to_metadata_dict(),
            "raw_size_wh": [raw_probe.width, raw_probe.height],
            "defines_final_processing_region": True,
        },
        "external_time": _external_time_metadata(
            external_time_selection,
            raw_probe,
            external_time_status,
            external_fps_effective,
        ),
        "cage_detection": {
            "used": is_automatic,
            "fit_score": crop_plan.fit_score if is_automatic else None,
            "rim_density": crop_plan.rim_density if is_automatic else None,
            "detector_diagnostics": (
                detector_diagnostics if is_automatic and detector_diagnostics else {}
            ),
        },
        "manual_crop": {"used": not is_automatic},
        "geometry": geometry,
        "encoding": {
            "ffmpeg": {
                "container": ffmpeg_config.container,
                "codec": ffmpeg_config.codec,
                "pixel_format": ffmpeg_config.pixel_format,
                "preset": ffmpeg_config.preset,
                "crf": ffmpeg_config.crf,
                "bframes": ffmpeg_config.bframes,
                "faststart": ffmpeg_config.faststart,
            },
            "opencv": {
                "container": opencv_config.container,
                "fourcc": opencv_config.fourcc,
            },
            "output_fps_header": output_fps,
            "fps_header_source": resolved_fps_source,
            "raw_fps_effective": resolved_raw_fps,
            "raw_fps_effective_method": resolved_raw_fps_method,
            "external_fps_effective": external_fps_effective,
            "external_fps_effective_method": external_fps_effective_method,
        },
        "prepared_video": {
            "path": _path_text(outputs.prepared_video_path),
            "width": prepared_width,
            "height": prepared_height,
            "fps_header": output_fps,
            "opencv_reported_frame_count": (prepared_validation.opencv_reported_frame_count),
            "opencv_readable_frame_count": (prepared_validation.opencv_readable_frame_count),
            "frame_count_used_for_sleap": (prepared_validation.opencv_readable_frame_count),
        },
        "background": {
            "path": _path_text(outputs.cropped_background_path),
            "method": background_result.method,
            "sampled_frame_count": background_result.sampled_frame_count,
            "sampled_frame_indices": list(background_result.sampled_frame_indices),
            "source_video_path": _path_text(background_result.source_video_path),
            "source_size_wh": list(background_result.source_size_wh),
            "output_dtype": background_result.output_dtype,
        },
        "sync": {
            "path": _path_text(sync_file),
            "sync_schema_version": PREPARED_SYNC_SCHEMA_VERSION,
            "frame_count_used_for_sleap": (prepared_validation.opencv_readable_frame_count),
        },
        "mask": serialize_static_mask(preprocess_config.mask),
        "validation": {
            "status": validation_status,
            "prepared_video": prepared_validation.model_dump(mode="json"),
            "background": {
                "status": "passed",
                "expected_size_wh": [prepared_width, prepared_height],
                "actual_size_wh": list(background_result.source_size_wh),
            },
        },
        "software_environment": software_environment.model_dump(mode="json"),
        "outputs": {
            "prepared_video_path": _path_text(outputs.prepared_video_path),
            "prepare_meta_path": _path_text(outputs.prepare_meta_path),
            "prepared_sync_path": _path_text(outputs.prepared_sync_path),
            "cropped_background_path": _path_text(outputs.cropped_background_path),
            "settings_used_path": _path_text(outputs.settings_used_path),
            "processing_log_path": _path_text(outputs.processing_log_path),
        },
    }
    validate_prepare_metadata(metadata)
    return metadata


def _require_mapping(
    metadata: Mapping[str, object],
    section_name: str,
) -> Mapping[str, object]:
    value = metadata.get(section_name)
    if not isinstance(value, Mapping):
        raise MetadataArtifactError(f"Metadata section '{section_name}' must be an object.")
    return value


def _require_nested_fields(
    section_name: str,
    section: Mapping[str, object],
) -> None:
    missing = sorted(_REQUIRED_NESTED_FIELDS[section_name] - set(section))
    if missing:
        raise MetadataArtifactError(
            f"Metadata section '{section_name}' is missing required fields: {missing}."
        )


def _positive_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise MetadataArtifactError(f"{name} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetadataArtifactError(f"{name} must be numeric.") from exc
    if not math.isfinite(number) or number <= 0:
        raise MetadataArtifactError(f"{name} must be finite and positive.")
    return number


def _size_pair(name: str, value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MetadataArtifactError(f"{name} must contain width and height.")
    width, height = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (width, height)):
        raise MetadataArtifactError(f"{name} width and height must be integers.")
    if width <= 0 or height <= 0:
        raise MetadataArtifactError(f"{name} width and height must be positive.")
    return width, height


def validate_prepare_metadata(metadata: dict[str, object]) -> None:
    """Validate metadata structure, cross-section invariants, and JSON safety.

    Raises:
        MetadataArtifactError: If required content is absent, inconsistent,
            unsuccessful, or not strictly JSON serializable.
    """

    missing_top_level = sorted(set(PREPARE_METADATA_TOP_LEVEL_SECTIONS) - set(metadata))
    if missing_top_level:
        raise MetadataArtifactError(
            f"Metadata is missing required top-level sections: {missing_top_level}."
        )
    if metadata.get("schema_version") != PREPARE_METADATA_SCHEMA_VERSION:
        raise MetadataArtifactError("Metadata schema_version is missing or invalid.")

    sections: dict[str, Mapping[str, object]] = {}
    for section_name in _REQUIRED_NESTED_FIELDS:
        section = _require_mapping(metadata, section_name)
        _require_nested_fields(section_name, section)
        sections[section_name] = section

    prepared = sections["prepared_video"]
    readable_count = prepared["opencv_readable_frame_count"]
    reported_count = prepared["opencv_reported_frame_count"]
    used_count = prepared["frame_count_used_for_sleap"]
    if (
        isinstance(readable_count, bool)
        or not isinstance(readable_count, int)
        or readable_count <= 0
    ):
        raise MetadataArtifactError("prepared_video.opencv_readable_frame_count must be positive.")
    if reported_count != readable_count:
        raise MetadataArtifactError("Prepared reported and readable frame counts must match.")
    if used_count != readable_count:
        raise MetadataArtifactError(
            "prepared_video.frame_count_used_for_sleap must equal readable count."
        )
    prepared_size = _size_pair(
        "prepared_video size",
        [prepared["width"], prepared["height"]],
    )
    _positive_number("prepared_video.fps_header", prepared["fps_header"])

    geometry = sections["geometry"]
    if _size_pair("geometry.prepared_size_wh", geometry["prepared_size_wh"]) != prepared_size:
        raise MetadataArtifactError(
            "geometry.prepared_size_wh must match prepared-video dimensions."
        )
    if geometry["accepted_by_user"] is not True:
        raise MetadataArtifactError("Metadata geometry must be accepted by the user.")

    trim = sections["trim"]
    if trim["end_frame_semantics"] != "exclusive":
        raise MetadataArtifactError("trim.end_frame_semantics must be exclusive.")
    if trim["expected_trimmed_frame_count"] != readable_count:
        raise MetadataArtifactError("Trim expected frame count must match prepared readable count.")

    background = sections["background"]
    if _size_pair("background.source_size_wh", background["source_size_wh"]) != prepared_size:
        raise MetadataArtifactError("Background source size must match prepared-video dimensions.")
    if sections["sync"]["frame_count_used_for_sleap"] != readable_count:
        raise MetadataArtifactError(
            "Sync frame_count_used_for_sleap must match prepared readable count."
        )

    encoding = sections["encoding"]
    ffmpeg = encoding["ffmpeg"]
    opencv = encoding["opencv"]
    if not isinstance(ffmpeg, Mapping):
        raise MetadataArtifactError("encoding.ffmpeg must be an object.")
    missing_ffmpeg = sorted(
        {
            "container",
            "codec",
            "pixel_format",
            "preset",
            "crf",
            "bframes",
            "faststart",
        }
        - set(ffmpeg)
    )
    if missing_ffmpeg:
        raise MetadataArtifactError(
            f"encoding.ffmpeg is missing required fields: {missing_ffmpeg}."
        )
    if not isinstance(opencv, Mapping):
        raise MetadataArtifactError("encoding.opencv must be an object.")
    missing_opencv = sorted({"container", "fourcc"} - set(opencv))
    if missing_opencv:
        raise MetadataArtifactError(
            f"encoding.opencv is missing required fields: {missing_opencv}."
        )

    mode = geometry["mode"]
    cage_used = sections["cage_detection"]["used"]
    manual_used = sections["manual_crop"]["used"]
    if mode == CropMode.AUTOMATIC.value:
        missing_cage_fields = sorted(
            {"fit_score", "rim_density", "detector_diagnostics"} - set(sections["cage_detection"])
        )
        if missing_cage_fields:
            raise MetadataArtifactError(
                "Automatic cage_detection metadata is missing required fields: "
                f"{missing_cage_fields}."
            )
        if cage_used is not True or manual_used is not False:
            raise MetadataArtifactError(
                "Automatic geometry requires cage_detection used and manual_crop unused."
            )
    elif mode == CropMode.MANUAL.value:
        if manual_used is not True or cage_used is not False:
            raise MetadataArtifactError(
                "Manual geometry requires manual_crop used and cage_detection unused."
            )
    else:
        raise MetadataArtifactError(f"Unsupported geometry mode: {mode}")

    validation = sections["validation"]
    if validation["status"] not in {"passed", "failed"}:
        raise MetadataArtifactError("validation.status must be passed or failed.")
    if validation["status"] == "passed":
        prepared_summary = validation["prepared_video"]
        background_summary = validation["background"]
        if (
            not isinstance(prepared_summary, Mapping)
            or prepared_summary.get("is_valid") is not True
        ):
            raise MetadataArtifactError("Passed metadata requires a valid prepared-video summary.")
        if (
            not isinstance(background_summary, Mapping)
            or background_summary.get("status") != "passed"
        ):
            raise MetadataArtifactError("Passed metadata requires a passed background summary.")

    mask = sections["mask"]
    try:
        mask_config = MaskConfig.model_validate(mask)
        validate_static_mask(mask_config, prepared_size)
    except (ValueError, VideoPreparationError) as exc:
        raise MetadataArtifactError(f"Mask metadata is invalid: {exc}") from exc
    _positive_number(
        "encoding.output_fps_header",
        encoding["output_fps_header"],
    )
    if encoding["output_fps_header"] != prepared["fps_header"]:
        raise MetadataArtifactError(
            "encoding.output_fps_header must match prepared_video.fps_header."
        )
    if encoding["fps_header_source"] not in {
        "external_time",
        "ffprobe_avg_frame_rate",
        "ffprobe_r_frame_rate",
        "opencv_reported_fps",
    }:
        raise MetadataArtifactError("encoding.fps_header_source is invalid.")
    for field_name in ("raw_fps_effective", "external_fps_effective"):
        value = encoding[field_name]
        if value is not None:
            _positive_number(f"encoding.{field_name}", value)
    external = sections["external_time"]
    if external["validation_status"] not in {
        "valid",
        "not_provided",
        "not_convertible_to_seconds",
    }:
        raise MetadataArtifactError("external_time.validation_status is invalid.")
    if (
        encoding["fps_header_source"] == "external_time"
        and encoding["external_fps_effective"] != encoding["output_fps_header"]
    ):
        raise MetadataArtifactError(
            "External-time FPS must match the selected output FPS header."
        )
    outputs = sections["outputs"]
    for field_name in _REQUIRED_NESTED_FIELDS["outputs"]:
        value = outputs[field_name]
        if not isinstance(value, str) or not value.strip():
            raise MetadataArtifactError(
                f"outputs.{field_name} must contain an official artifact path."
            )

    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetadataArtifactError(
            f"Metadata contains non-JSON-serializable values: {exc}"
        ) from exc


def _create_temporary_path(destination: Path, marker: str) -> Path:
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.{marker}-",
            suffix=destination.suffix,
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name).resolve()
        temporary_path.unlink()
    except OSError as exc:
        raise MetadataArtifactError(f"Could not create temporary artifact path: {exc}") from exc
    return temporary_path


def _remove_temporary_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise MetadataArtifactError(
            f"Could not remove failed temporary artifact {path}: {exc}"
        ) from exc


def write_prepare_meta_json(
    path: Path,
    metadata: dict[str, object],
) -> Path:
    """Atomically write, reopen, and revalidate authoritative JSON metadata."""

    validate_prepare_metadata(metadata)
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".json":
        raise MetadataArtifactError("Prepare metadata path must use .json.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
    except OSError as exc:
        raise MetadataArtifactError(
            f"Could not create metadata destination directory: {exc}"
        ) from exc

    temporary_path = _create_temporary_path(destination, "metadata")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                metadata,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        with temporary_path.open("r", encoding="utf-8") as stream:
            reloaded = json.load(stream)
        if not isinstance(reloaded, dict):
            raise MetadataArtifactError("Reloaded prepare metadata is not a JSON object.")
        validate_prepare_metadata(reloaded)
        temporary_path.replace(destination)
    except MetadataArtifactError:
        _remove_temporary_path(temporary_path)
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _remove_temporary_path(temporary_path)
        raise MetadataArtifactError(f"Could not write prepare metadata JSON: {exc}") from exc
    return destination


def write_settings_used_yaml(
    path: Path,
    config: PreprocessConfig,
) -> Path:
    """Atomically write and parse-validate the accepted preprocess settings."""

    payload = config.model_dump(mode="json")
    destination = Path(path).expanduser()
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise MetadataArtifactError("Settings path must use .yaml or .yml.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
    except OSError as exc:
        raise MetadataArtifactError(
            f"Could not create settings destination directory: {exc}"
        ) from exc

    temporary_path = _create_temporary_path(destination, "settings")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                payload,
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
        with temporary_path.open("r", encoding="utf-8") as stream:
            reloaded = yaml.safe_load(stream)
        if not isinstance(reloaded, dict):
            raise MetadataArtifactError("Reloaded settings artifact is not a YAML mapping.")
        validated = PreprocessConfig.model_validate(reloaded)
        if validated.model_dump(mode="json") != payload:
            raise MetadataArtifactError("Reloaded settings differ from the accepted configuration.")
        temporary_path.replace(destination)
    except MetadataArtifactError:
        _remove_temporary_path(temporary_path)
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        _remove_temporary_path(temporary_path)
        raise MetadataArtifactError(f"Could not write settings YAML: {exc}") from exc
    return destination
