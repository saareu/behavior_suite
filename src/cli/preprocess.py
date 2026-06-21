"""Typer command-line interface for the preprocessing subsystem."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from pydantic import ValidationError

from preprocess.cage_detection import detect_cage_crop_plan
from preprocess.config import PreCropConfig, PreprocessConfig, load_preprocess_config
from preprocess.crop_plan import CropPlan
from preprocess.exceptions import PreprocessError
from preprocess.mat_sync_reader import (
    convert_timing_vector_to_seconds,
    validate_external_timing_vector,
)
from preprocess.models import (
    ExternalTimeSelection,
    PreprocessRequest,
    TimingUnit,
)
from preprocess.pre_crop import PreCropMode, resolve_pre_crop
from preprocess.service import PreprocessService, resolve_trim_range
from preprocess.video_probe import probe_video
from project.service import ProjectService
from project.validation import ProjectError

CROP_PLAN_SCHEMA_VERSION = "crop_plan_v1"

app = typer.Typer(
    name="behavior-suite",
    help="Scientific behavioral-video processing tools.",
    no_args_is_help=True,
)
preprocess_app = typer.Typer(
    help="Detect, accept, and run deterministic video preprocessing.",
    no_args_is_help=True,
)
app.add_typer(preprocess_app, name="preprocess")


class CliInputError(ValueError):
    """Expected command-line input or artifact validation failure."""


_EXPECTED_ERRORS = (
    CliInputError,
    PreprocessError,
    ProjectError,
    ValidationError,
    OSError,
)
_CROP_PLAN_FIELDS = {
    "schema_version",
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
}
_OPTIONAL_CROP_PLAN_FIELDS = {"detector_diagnostics"}


def _crop_plan_payload(
    crop_plan: CropPlan,
    detector_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    serialized = crop_plan.to_metadata_dict()
    native_geometry: dict[str, object] | None = None
    if crop_plan.native_size_wh is not None:
        native_geometry = {"size_wh": list(crop_plan.native_size_wh)}
    return {
        "schema_version": CROP_PLAN_SCHEMA_VERSION,
        "mode": serialized["mode"],
        "pre_crop_roi": serialized["pre_crop_roi"],
        "quad_raw_tl_tr_br_bl": serialized["quad_raw_tl_tr_br_bl"],
        "H_raw_to_prepared_3x3": serialized["H_raw_to_prepared_3x3"],
        "H_prepared_to_raw_3x3": serialized["H_prepared_to_raw_3x3"],
        "prepared_size_wh": serialized["prepared_size_wh"],
        "rotated_90": serialized["rotated_90"],
        "fit_score": serialized["fit_score"],
        "rim_density": serialized["rim_density"],
        "accepted_by_user": serialized["accepted_by_user"],
        "native_geometry": native_geometry,
        "canonical_geometry": serialized["canonical_geometry"],
        "detector_diagnostics": (
            dict(detector_diagnostics) if detector_diagnostics is not None else {}
        ),
    }


def _parse_crop_plan_payload(
    payload: Mapping[str, Any],
) -> tuple[CropPlan, dict[str, object]]:
    actual_fields = set(payload)
    missing = sorted(_CROP_PLAN_FIELDS - actual_fields)
    unexpected = sorted(actual_fields - _CROP_PLAN_FIELDS - _OPTIONAL_CROP_PLAN_FIELDS)
    if missing:
        raise CliInputError(f"Crop-plan JSON is missing required fields: {missing}.")
    if unexpected:
        raise CliInputError(f"Crop-plan JSON has unexpected fields: {unexpected}.")
    if payload["schema_version"] != CROP_PLAN_SCHEMA_VERSION:
        raise CliInputError("Crop-plan schema_version is missing or invalid.")

    native_geometry = payload["native_geometry"]
    if native_geometry is None:
        native_size_wh = None
    elif isinstance(native_geometry, Mapping) and set(native_geometry) == {"size_wh"}:
        native_size_wh = native_geometry["size_wh"]
    else:
        raise CliInputError("Crop-plan native_geometry must be null or contain only size_wh.")

    model_payload = {
        "mode": payload["mode"],
        "pre_crop_roi": payload["pre_crop_roi"],
        "quad_raw_tl_tr_br_bl": payload["quad_raw_tl_tr_br_bl"],
        "H_raw_to_prepared_3x3": payload["H_raw_to_prepared_3x3"],
        "H_prepared_to_raw_3x3": payload["H_prepared_to_raw_3x3"],
        "prepared_size_wh": payload["prepared_size_wh"],
        "native_size_wh": native_size_wh,
        "canonical_geometry": payload["canonical_geometry"],
        "rotated_90": payload["rotated_90"],
        "fit_score": payload["fit_score"],
        "rim_density": payload["rim_density"],
        "accepted_by_user": payload["accepted_by_user"],
    }
    crop_plan = CropPlan.from_metadata_dict(model_payload)
    raw_diagnostics = payload.get("detector_diagnostics", {})
    if not isinstance(raw_diagnostics, Mapping):
        raise CliInputError("detector_diagnostics must be a JSON object.")
    return crop_plan, dict(raw_diagnostics)


def load_crop_plan_document(
    path: Path,
) -> tuple[CropPlan, dict[str, object]]:
    """Load and strictly revalidate a versioned CropPlan JSON document."""

    crop_plan_path = Path(path).expanduser()
    if crop_plan_path.suffix.lower() != ".json":
        raise CliInputError("Crop-plan path must use the .json suffix.")
    if not crop_plan_path.is_file():
        raise CliInputError(f"Crop-plan JSON does not exist: {crop_plan_path}")
    try:
        with crop_plan_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise CliInputError(f"Crop-plan JSON is malformed: {crop_plan_path}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise CliInputError(f"Could not read crop-plan JSON {crop_plan_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CliInputError("Crop-plan JSON must contain a top-level object.")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CliInputError(f"Crop-plan JSON contains non-JSON-safe values: {exc}") from exc
    return _parse_crop_plan_payload(payload)


def load_crop_plan_json(path: Path) -> CropPlan:
    """Load and strictly revalidate a CropPlan from its JSON document."""

    crop_plan, _diagnostics = load_crop_plan_document(path)
    return crop_plan


def _validate_new_crop_plan_path(path: Path) -> Path:
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".json":
        raise CliInputError("Crop-plan output path must use the .json suffix.")
    if destination.exists():
        raise CliInputError(
            f"Crop-plan output already exists; refusing to replace it: {destination}"
        )
    if not destination.parent.is_dir():
        raise CliInputError(f"Crop-plan output directory does not exist: {destination.parent}")
    return destination.resolve()


def write_crop_plan_json(
    path: Path,
    crop_plan: CropPlan,
    *,
    detector_diagnostics: Mapping[str, object] | None = None,
) -> Path:
    """Safely write and reload a new, versioned CropPlan JSON document.

    Existing destinations are refused. The document is staged in the same
    directory, parsed, and fully revalidated before its final rename.
    """

    destination = _validate_new_crop_plan_path(path)
    payload = _crop_plan_payload(crop_plan, detector_diagnostics)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.crop-plan-",
            suffix=".json",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name).resolve()
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        loaded_plan, loaded_diagnostics = load_crop_plan_document(temporary_path)
        if _crop_plan_payload(loaded_plan, loaded_diagnostics) != payload:
            raise CliInputError("Reloaded crop-plan JSON differs from the requested document.")
        if destination.exists():
            raise CliInputError(
                f"Crop-plan output appeared during writing; refusing to replace it: {destination}"
            )
        temporary_path.replace(destination)
    except (TypeError, ValueError, OSError) as exc:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, CliInputError):
            raise
        raise CliInputError(f"Could not write crop-plan JSON: {exc}") from exc
    return destination


def _resolve_cli_pre_crop(
    config: PreprocessConfig,
    *,
    mode_text: str | None,
    boundary: int | None,
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
) -> PreCropConfig:
    rectangle_values = (x, y, width, height)
    if mode_text is None:
        if boundary is not None or any(value is not None for value in rectangle_values):
            raise CliInputError("Pre-crop geometry arguments require --pre-crop-mode.")
        return config.pre_crop
    try:
        mode = PreCropMode(mode_text)
    except ValueError as exc:
        accepted = ", ".join(item.value for item in PreCropMode)
        raise CliInputError(
            f"Unsupported pre-crop mode '{mode_text}'. Expected one of: {accepted}."
        ) from exc

    if mode is PreCropMode.NONE:
        if boundary is not None or any(value is not None for value in rectangle_values):
            raise CliInputError("Pre-crop mode none does not accept geometry arguments.")
        return PreCropConfig(enabled=False, mode=mode.value)
    if mode is PreCropMode.MANUAL_RECTANGLE:
        if boundary is not None:
            raise CliInputError("manual_rectangle does not accept --pre-crop-boundary.")
        if any(value is None for value in rectangle_values):
            raise CliInputError(
                "manual_rectangle requires --pre-crop-x, --pre-crop-y, "
                "--pre-crop-width, and --pre-crop-height."
            )
        return PreCropConfig(
            enabled=True,
            mode=mode.value,
            manual_rectangle=(x, y, width, height),
        )
    if boundary is None:
        raise CliInputError(f"{mode.value} requires --pre-crop-boundary.")
    if any(value is not None for value in rectangle_values):
        raise CliInputError("Directional pre-crop modes do not accept rectangle arguments.")
    return PreCropConfig(enabled=True, mode=mode.value, boundary_px=boundary)


def _load_external_timing_options(
    *,
    npy_path: Path | None,
    source_path: Path | None,
    variable_name: str | None,
    units_text: str | None,
) -> tuple[ExternalTimeSelection | None, np.ndarray | None]:
    supplied = (
        npy_path is not None,
        source_path is not None,
        variable_name is not None,
        units_text is not None,
    )
    if not any(supplied):
        return None, None
    if not all(supplied):
        raise CliInputError(
            "External timing requires --external-time-npy, --external-time-source, "
            "--external-time-variable, and --external-time-units together."
        )
    assert npy_path is not None
    assert source_path is not None
    assert variable_name is not None
    assert units_text is not None
    if not variable_name.strip():
        raise CliInputError("--external-time-variable must be non-empty.")
    try:
        unit = TimingUnit(units_text)
    except ValueError as exc:
        accepted = ", ".join(item.value for item in TimingUnit)
        raise CliInputError(
            f"Unsupported external timing units '{units_text}'. Expected: {accepted}."
        ) from exc

    vector_path = Path(npy_path).expanduser()
    if vector_path.suffix.lower() != ".npy":
        raise CliInputError("--external-time-npy must use the .npy suffix.")
    if not vector_path.is_file():
        raise CliInputError(f"External timing NPY does not exist: {vector_path}")
    try:
        vector = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise CliInputError(f"Could not load external timing NPY {vector_path}: {exc}") from exc
    if not isinstance(vector, np.ndarray) or vector.ndim != 1 or vector.size == 0:
        raise CliInputError("--external-time-npy must contain one non-empty one-dimensional array.")
    selection = validate_external_timing_vector(
        vector,
        int(vector.size),
        unit,
        source_path=source_path,
        selected_variable=variable_name,
    )
    vector_seconds = convert_timing_vector_to_seconds(vector, unit)
    return selection, vector_seconds


def _show_expected_error(exc: BaseException) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)


def _show_unexpected_error(exc: BaseException) -> None:
    typer.echo(f"Internal error: {exc}", err=True)
    raise typer.Exit(code=2)


def _score_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


@preprocess_app.command("detect-crop")
def detect_crop(
    project_dir: Annotated[Path, typer.Option("--project-dir", help="Existing project root.")],
    raw_video: Annotated[Path, typer.Option("--raw-video", help="Raw source video.")],
    config_path: Annotated[Path, typer.Option("--config", help="Preprocess YAML configuration.")],
    output_crop_plan: Annotated[
        Path, typer.Option("--output-crop-plan", help="New review JSON path.")
    ],
    start_frame: Annotated[int | None, typer.Option("--start-frame")] = None,
    end_frame: Annotated[
        int | None, typer.Option("--end-frame", help="Exclusive end frame.")
    ] = None,
    pre_crop_mode: Annotated[str | None, typer.Option("--pre-crop-mode")] = None,
    pre_crop_boundary: Annotated[int | None, typer.Option("--pre-crop-boundary")] = None,
    pre_crop_x: Annotated[int | None, typer.Option("--pre-crop-x")] = None,
    pre_crop_y: Annotated[int | None, typer.Option("--pre-crop-y")] = None,
    pre_crop_width: Annotated[int | None, typer.Option("--pre-crop-width")] = None,
    pre_crop_height: Annotated[int | None, typer.Option("--pre-crop-height")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show resolved input details.")
    ] = False,
) -> None:
    """Probe a video, detect an automatic crop, and save it unaccepted."""

    try:
        project = ProjectService().open_project(project_dir)
        destination = _validate_new_crop_plan_path(output_crop_plan)
        config = load_preprocess_config(config_path)
        raw_probe = probe_video(raw_video, require_sequential_count=False)
        trim = resolve_trim_range(
            request_start_frame=start_frame,
            request_end_frame_exclusive=end_frame,
            config=config,
        )
        pre_crop_config = _resolve_cli_pre_crop(
            config,
            mode_text=pre_crop_mode,
            boundary=pre_crop_boundary,
            x=pre_crop_x,
            y=pre_crop_y,
            width=pre_crop_width,
            height=pre_crop_height,
        )
        effective_config = config.model_copy(update={"pre_crop": pre_crop_config})
        resolved_pre_crop = resolve_pre_crop(
            pre_crop_config,
            (raw_probe.width, raw_probe.height),
        )
        detection = detect_cage_crop_plan(
            raw_probe.source_path,
            effective_config,
            resolved_pre_crop,
        )
        crop_plan = detection.crop_plan
        if crop_plan.accepted_by_user:
            raise RuntimeError("Automatic cage detection returned an accepted CropPlan.")
        saved_path = write_crop_plan_json(
            destination,
            crop_plan,
            detector_diagnostics=detection.detector_diagnostics,
        )

        typer.echo("Status: DETECTED, NOT ACCEPTED")
        typer.echo(f"Crop-plan JSON: {saved_path}")
        typer.echo(
            f"Output size: {crop_plan.prepared_size_wh[0]} x {crop_plan.prepared_size_wh[1]}"
        )
        typer.echo(f"Fit score: {_score_text(crop_plan.fit_score)}")
        typer.echo(f"Rim density: {_score_text(crop_plan.rim_density)}")
        typer.echo("Rotation: clockwise 90 degrees" if crop_plan.rotated_90 else "Rotation: none")
        typer.echo("Crop acceptance: REQUIRED before preprocessing.")
        if verbose:
            typer.echo(f"Project: {project.root_dir}")
            typer.echo(f"Raw size: {raw_probe.width} x {raw_probe.height}")
            typer.echo(f"Trim: start={trim.start_frame}, end_exclusive={trim.end_frame_exclusive}")
            typer.echo(f"Pre-crop ROI: {resolved_pre_crop.roi.model_dump()}")
    except _EXPECTED_ERRORS as exc:
        _show_expected_error(exc)
    except Exception as exc:
        _show_unexpected_error(exc)


@preprocess_app.command("accept-crop")
def accept_crop(
    crop_plan_path: Annotated[Path, typer.Option("--crop-plan", help="Unaccepted CropPlan JSON.")],
    output_crop_plan: Annotated[
        Path, typer.Option("--output-crop-plan", help="New accepted JSON path.")
    ],
) -> None:
    """Explicitly accept a reviewed CropPlan into a separate JSON file."""

    try:
        source = Path(crop_plan_path).expanduser().resolve()
        requested_destination = Path(output_crop_plan).expanduser().resolve()
        if source == requested_destination:
            raise CliInputError("Input and output crop-plan paths must differ.")
        crop_plan, detector_diagnostics = load_crop_plan_document(source)
        destination = _validate_new_crop_plan_path(output_crop_plan)
        accepted_payload = crop_plan.to_metadata_dict()
        accepted_payload["accepted_by_user"] = True
        accepted_plan = CropPlan.from_metadata_dict(accepted_payload)
        saved_path = write_crop_plan_json(
            destination,
            accepted_plan,
            detector_diagnostics=detector_diagnostics,
        )
        typer.echo("Status: ACCEPTED")
        typer.echo(f"Accepted crop-plan JSON: {saved_path}")
        typer.echo("The accepted CropPlan is ready for preprocessing.")
    except _EXPECTED_ERRORS as exc:
        _show_expected_error(exc)
    except Exception as exc:
        _show_unexpected_error(exc)


@preprocess_app.command("run")
def run_preprocess(
    project_dir: Annotated[Path, typer.Option("--project-dir", help="Existing project root.")],
    raw_video: Annotated[Path, typer.Option("--raw-video", help="Raw source video.")],
    config_path: Annotated[Path, typer.Option("--config", help="Preprocess YAML configuration.")],
    crop_plan_path: Annotated[Path, typer.Option("--crop-plan", help="Accepted CropPlan JSON.")],
    start_frame: Annotated[int | None, typer.Option("--start-frame")] = None,
    end_frame: Annotated[
        int | None, typer.Option("--end-frame", help="Exclusive end frame.")
    ] = None,
    external_time_npy: Annotated[Path | None, typer.Option("--external-time-npy")] = None,
    external_time_source: Annotated[Path | None, typer.Option("--external-time-source")] = None,
    external_time_variable: Annotated[str | None, typer.Option("--external-time-variable")] = None,
    external_time_units: Annotated[str | None, typer.Option("--external-time-units")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show FPS and elapsed details.")
    ] = False,
) -> None:
    """Run the complete preprocessing service with an accepted CropPlan."""

    try:
        project = ProjectService().open_project(project_dir)
        config = load_preprocess_config(config_path)
        crop_plan, detector_diagnostics = load_crop_plan_document(crop_plan_path)
        if not crop_plan.accepted_by_user:
            raise CliInputError("CropPlan is not accepted. Run preprocess accept-crop first.")
        external_selection, external_vector_seconds = _load_external_timing_options(
            npy_path=external_time_npy,
            source_path=external_time_source,
            variable_name=external_time_variable,
            units_text=external_time_units,
        )
        request = PreprocessRequest(
            project_dir=project.root_dir,
            raw_video_path=raw_video,
            config=config,
            start_frame=start_frame,
            end_frame_exclusive=end_frame,
            external_time_selection=external_selection,
            external_time_vector_seconds=external_vector_seconds,
            crop_plan=crop_plan,
            crop_accepted_by_user=True,
            detector_diagnostics=detector_diagnostics,
        )
        result = PreprocessService().run(request)
        if not result.success:
            typer.echo("Status: FAILED", err=True)
            typer.echo(f"Error: {result.message}", err=True)
            for warning in result.warnings:
                typer.echo(f"Warning: {warning}", err=True)
            log_path = project.root_dir / "preprocess" / "processing_log.txt"
            if log_path.is_file():
                typer.echo(f"Log: {log_path}", err=True)
            raise typer.Exit(code=1)
        if result.outputs is None:
            raise RuntimeError("PreprocessService reported success without official output paths.")

        outputs = result.outputs
        typer.echo("Status: SUCCESS")
        typer.echo(f"Prepared video: {outputs.prepared_video_path}")
        typer.echo(f"Sync artifact: {outputs.prepared_sync_path}")
        typer.echo(f"Background: {outputs.cropped_background_path}")
        typer.echo(f"Metadata: {outputs.prepare_meta_path}")
        typer.echo(f"Settings: {outputs.settings_used_path}")
        typer.echo(f"Log: {outputs.processing_log_path}")
        for warning in result.warnings:
            typer.echo(f"Warning: {warning}")
        if verbose:
            typer.echo(f"FPS header: {result.fps_header}")
            typer.echo(f"FPS source: {result.fps_header_source}")
            typer.echo(f"Elapsed seconds: {result.elapsed_sec:.3f}")
    except typer.Exit:
        raise
    except _EXPECTED_ERRORS as exc:
        _show_expected_error(exc)
    except Exception as exc:
        _show_unexpected_error(exc)


@app.command("gui")
def gui_command(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Optional preprocess YAML configuration."),
    ] = None,
) -> None:
    """Launch the desktop preprocessing setup application."""

    from ui.app import GUI_INSTALL_GUIDANCE, GuiDependencyError, launch_gui

    try:
        exit_code = launch_gui(config_path)
    except GuiDependencyError:
        typer.echo(GUI_INSTALL_GUIDANCE, err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        _show_unexpected_error(exc)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
