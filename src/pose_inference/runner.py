"""Initial Subsystem 02 SLEAP-NN inference runner.

This module intentionally avoids importing SLEAP or GPU libraries. The only
runtime boundary for real inference is the ``sleap-nn predict`` subprocess.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pose_inference.overlay import (
    DEFAULT_CODEC,
    OverlayGenerationError,
    OverlaySummary,
    generate_overlay_video,
)
from pose_inference.parquet_export import (
    POSE_PARQUET_SCHEMA_VERSION,
    ParquetExportError,
    ParquetExportSummary,
    export_pose_parquet,
)
from pose_inference.pose_qc import (
    DEFAULT_EXPECTED_ANIMALS,
    DEFAULT_REVIEW_FRAME_MISSING_KEYPOINT_FRACTION_THRESHOLD,
    DEFAULT_REVIEW_MISSING_KEYPOINT_FRACTION_THRESHOLD,
    DEFAULT_REVIEW_ONE_ANIMAL_FRACTION_THRESHOLD,
    POSE_QC_SCHEMA_VERSION,
    PoseQcError,
    PoseQcSummary,
    compute_pose_qc_from_parquet,
)

REQUIRED_S1_FILES = (
    "prepared_video.mp4",
    "prepare_meta.json",
    "prepared_sync.npz",
)
FORBIDDEN_STANDARD_ARTIFACT_NAMES = {
    "pose_tracked.slp",
    "overlay_tracked.mp4",
    "tracking_qc.csv",
    "tracking_report.json",
    "track_identity_map.json",
}
POSE_META_SCHEMA_VERSION = "pose_meta_v1"
JOB_MANIFEST_SCHEMA_VERSION = "pose_inference_job_manifest_v1"
SETTINGS_SCHEMA_VERSION = "pose_inference_settings_v1"
SLEAP_PROVENANCE_SOURCE = "pose.slp labels.provenance"
SLEAP_PROVENANCE_NOTE = (
    "SLEAP-NN checkpoint startup logs may report Predictor construction defaults. "
    "Effective prediction-stage settings are read from pose.slp labels.provenance."
)
SLEAP_PROVENANCE_FIELDS = {
    "inference_config": (
        "inference_config",
        "prediction_config",
        "predictor_config",
        "predict_config",
        "inference",
    ),
    "tracking_config": ("tracking_config", "tracker_config", "track_config", "tracking"),
    "device": ("device", "devices"),
    "system_info": ("system_info", "system", "environment"),
    "model_paths": ("model_paths", "model_path", "models"),
    "model_type": ("model_type", "model"),
    "source_file": ("source_file", "input_file", "data_path", "video_path"),
    "frame_selection": ("frame_selection", "frames", "frame_range"),
    "sleap_nn_version": ("sleap_nn_version", "sleap-nn_version", "sleap_nn"),
    "sleap_io_version": ("sleap_io_version", "sleap-io_version", "sleap_io"),
}


class PoseInferenceError(RuntimeError):
    """Expected Subsystem 02 input, dispatch, or artifact error."""


class SleapProvenanceError(RuntimeError):
    """Expected failure while reading SLEAP provenance from pose.slp."""


@dataclass(frozen=True)
class S1Handoff:
    """Resolved read-only Subsystem 01 handoff paths."""

    session_root: Path
    preprocess_dir: Path
    prepared_video: Path
    prepare_meta: Path
    prepared_sync: Path
    prepare_meta_payload: Mapping[str, Any]


@dataclass(frozen=True)
class PoseInferenceModelSpec:
    """Explicit SLEAP-NN checkpoint selection for one inference mode."""

    inference_mode: str
    bottomup_model_path: Path | None = None
    centroid_model_path: Path | None = None
    centered_instance_model_path: Path | None = None


@dataclass(frozen=True)
class PoseInferenceRequest:
    """Inputs for one Subsystem 02 inference or dry-run."""

    session_root: Path
    model_path: Path | None
    profile_path: Path
    output_root: Path | None = None
    run_purpose: str = "development"
    dry_run: bool = False
    timestamp: str | None = None
    model_spec: PoseInferenceModelSpec | None = None


@dataclass(frozen=True)
class RunPaths:
    """Resolved output paths for one run directory."""

    run_dir: Path
    pose_slp: Path
    pose_parquet: Path
    overlay_mp4: Path
    pose_meta: Path
    settings_used: Path
    job_manifest: Path
    processing_log: Path


@dataclass(frozen=True)
class PoseInferenceResult:
    """Summary returned by the Subsystem 02 runner."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    pose_slp_path: Path
    pose_meta_path: Path
    settings_used_path: Path
    job_manifest_path: Path
    processing_log_path: Path
    command: tuple[str, ...]
    returncode: int | None = None


@dataclass(frozen=True)
class SleapProvenanceSummary:
    """Compact effective SLEAP/SLEAP-NN provenance copied from pose.slp."""

    status: str
    source: str = SLEAP_PROVENANCE_SOURCE
    inference_config: Any = None
    tracking_config: Any = None
    device: Any = None
    system_info: Any = None
    model_paths: Any = None
    model_type: Any = None
    source_file: Any = None
    frame_selection: Any = None
    sleap_nn_version: str | None = None
    sleap_io_version: str | None = None
    error: str | None = None
    reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "inference_config": self.inference_config,
            "tracking_config": self.tracking_config,
            "device": self.device,
            "system_info": self.system_info,
            "model_paths": self.model_paths,
            "model_type": self.model_type,
            "source_file": self.source_file,
            "frame_selection": self.frame_selection,
            "sleap_nn_version": self.sleap_nn_version,
            "sleap_io_version": self.sleap_io_version,
            "error": self.error,
            "reason": self.reason,
        }


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]
ParquetExporter = Callable[..., ParquetExportSummary]
PoseQcComputer = Callable[..., PoseQcSummary]
OverlayGenerator = Callable[..., OverlaySummary]
SleapProvenanceReader = Callable[[Path], SleapProvenanceSummary]


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise PoseInferenceError(f"prepare_meta.json is malformed: {path}: {exc}") from exc
    except OSError as exc:
        raise PoseInferenceError(f"Could not read prepare_meta.json: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PoseInferenceError("prepare_meta.json must contain a top-level object.")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | bool) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError, AttributeError):
            pass
    return str(value)


def _provenance_mapping(provenance: Any) -> Mapping[str, Any]:
    if isinstance(provenance, Mapping):
        return provenance
    to_dict = getattr(provenance, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return {"value": provenance}


def _find_value(payload: Any, key_candidates: Sequence[str]) -> Any:
    if isinstance(payload, Mapping):
        for key in key_candidates:
            if key in payload:
                return payload[key]
        for value in payload.values():
            found = _find_value(value, key_candidates)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        for value in payload:
            found = _find_value(value, key_candidates)
            if found is not None:
                return found
    return None


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sleap_provenance_from_payload(
    provenance: Any,
    *,
    sleap_io_version: str | None,
) -> SleapProvenanceSummary:
    payload = _provenance_mapping(provenance)
    fields = {
        field_name: _jsonable(_find_value(payload, candidates))
        for field_name, candidates in SLEAP_PROVENANCE_FIELDS.items()
    }
    if fields["sleap_io_version"] is None:
        fields["sleap_io_version"] = sleap_io_version
    if fields["sleap_nn_version"] is None:
        fields["sleap_nn_version"] = _package_version("sleap-nn")
    return SleapProvenanceSummary(status="available", **fields)


def read_sleap_provenance(pose_slp_path: Path) -> SleapProvenanceSummary:
    """Read effective SLEAP prediction provenance from ``labels.provenance``."""

    try:
        import sleap_io  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SleapProvenanceError("sleap_io is required to read pose.slp provenance.") from exc

    path = Path(pose_slp_path)
    try:
        if hasattr(sleap_io, "load_slp"):
            labels = sleap_io.load_slp(path)
        elif hasattr(sleap_io, "load_file"):
            labels = sleap_io.load_file(path)
        else:
            raise SleapProvenanceError("Installed sleap_io does not expose load_slp/load_file.")
    except SleapProvenanceError:
        raise
    except Exception as exc:
        raise SleapProvenanceError(f"Could not read SLEAP provenance from pose.slp: {exc}") from exc

    provenance = getattr(labels, "provenance", None)
    if provenance is None:
        return SleapProvenanceSummary(
            status="not_available",
            reason="labels.provenance is missing",
        )
    return _sleap_provenance_from_payload(
        provenance,
        sleap_io_version=getattr(sleap_io, "__version__", None),
    )


def _resolve_preprocess_dir(session_root_or_preprocess: Path) -> tuple[Path, Path]:
    candidate = Path(session_root_or_preprocess).expanduser().resolve()
    if not candidate.exists():
        raise PoseInferenceError(f"Input path does not exist: {candidate}")
    if not candidate.is_dir():
        raise PoseInferenceError(f"Input path must be a directory: {candidate}")

    contains_handoff_files = any((candidate / name).exists() for name in REQUIRED_S1_FILES)
    if candidate.name.lower() == "preprocess" or contains_handoff_files:
        preprocess_dir = candidate
        session_root = candidate.parent
    else:
        session_root = candidate
        preprocess_dir = candidate / "preprocess"
    return session_root, preprocess_dir


def validate_s1_handoff(session_root_or_preprocess: Path) -> S1Handoff:
    """Validate required Subsystem 01 handoff artifacts and return resolved paths."""

    session_root, preprocess_dir = _resolve_preprocess_dir(session_root_or_preprocess)
    if not preprocess_dir.is_dir():
        raise PoseInferenceError(f"Preprocess directory does not exist: {preprocess_dir}")

    missing = [name for name in REQUIRED_S1_FILES if not (preprocess_dir / name).is_file()]
    if missing:
        raise PoseInferenceError(
            "Missing required Subsystem 01 handoff files: " + ", ".join(missing)
        )

    prepare_meta = preprocess_dir / "prepare_meta.json"
    return S1Handoff(
        session_root=session_root,
        preprocess_dir=preprocess_dir,
        prepared_video=preprocess_dir / "prepared_video.mp4",
        prepare_meta=prepare_meta,
        prepared_sync=preprocess_dir / "prepared_sync.npz",
        prepare_meta_payload=_load_json_object(prepare_meta),
    )


def _validate_prepared_video_against_sync(handoff: S1Handoff) -> None:
    try:
        from preprocess.sync_writer import SyncValidationError, load_prepared_sync_npz

        sync = load_prepared_sync_npz(handoff.prepared_sync)
    except (OSError, SyncValidationError) as exc:
        raise PoseInferenceError(
            f"Subsystem 01 prepared_sync.npz is unreadable or invalid: {exc}"
        ) from exc

    try:
        import cv2
    except ImportError as exc:
        raise PoseInferenceError(
            "OpenCV is required for prepared-video preflight validation."
        ) from exc

    capture = cv2.VideoCapture(str(handoff.prepared_video))
    if not capture.isOpened():
        capture.release()
        raise PoseInferenceError(
            f"Subsystem 01 prepared_video.mp4 is unreadable: {handoff.prepared_video}"
        )
    reported_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_count = (
        int(reported_count_value)
        if math.isfinite(reported_count_value) and reported_count_value > 0
        else None
    )
    try:
        success, frame = capture.read()
        if not success or frame is None:
            raise PoseInferenceError(
                "Subsystem 01 prepared_video.mp4 did not yield a readable first frame."
            )
    except cv2.error as exc:
        raise PoseInferenceError(
            f"Subsystem 01 prepared_video.mp4 could not be decoded: {exc}"
        ) from exc
    finally:
        capture.release()

    expected_count = int(sync.frame_count_used_for_sleap)
    if reported_count is not None and reported_count != expected_count:
        raise PoseInferenceError(
            "Prepared-video header frame count does not match the authoritative S1 "
            f"sync contract: reported={reported_count}, expected={expected_count}."
        )

    prepared_meta = handoff.prepare_meta_payload.get("prepared_video")
    if isinstance(prepared_meta, Mapping):
        for key in (
            "opencv_reported_frame_count",
            "opencv_readable_frame_count",
            "frame_count_used_for_sleap",
        ):
            value = prepared_meta.get(key)
            if value is not None and value != expected_count:
                raise PoseInferenceError(
                    f"prepare_meta.json {key} does not match the authoritative "
                    f"S1 sync frame count {expected_count}."
                )


def _validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise PoseInferenceError(f"Model path does not exist: {model_path}")
    try:
        if model_path.is_file():
            with model_path.open("rb") as stream:
                stream.read(1)
        elif model_path.is_dir():
            next(model_path.iterdir(), None)
        else:
            raise PoseInferenceError(f"Model path is not a file or directory: {model_path}")
    except OSError as exc:
        raise PoseInferenceError(f"Model path is unreadable: {model_path}: {exc}") from exc


def _normalize_inference_mode(value: str) -> str:
    mode = value.strip().lower().replace("-", "")
    if mode not in {"bottomup", "topdown"}:
        raise PoseInferenceError(
            "Inference mode must be either 'bottomup' or 'topdown'."
        )
    return mode


def _resolve_model_spec(request: PoseInferenceRequest) -> PoseInferenceModelSpec:
    if request.model_spec is not None and request.model_path is not None:
        raise PoseInferenceError(
            "Model selection is ambiguous: use either legacy model_path or model_spec, "
            "not both."
        )
    if request.model_spec is None:
        if request.model_path is None:
            raise PoseInferenceError("Bottom-up inference requires a model path.")
        spec = PoseInferenceModelSpec(
            inference_mode="bottomup",
            bottomup_model_path=request.model_path,
        )
    else:
        spec = request.model_spec

    mode = _normalize_inference_mode(spec.inference_mode)
    if mode == "bottomup":
        if spec.bottomup_model_path is None:
            raise PoseInferenceError("Bottom-up inference requires bottomup_model_path.")
        if spec.centroid_model_path is not None or spec.centered_instance_model_path is not None:
            raise PoseInferenceError(
                "Bottom-up inference cannot include top-down component model paths."
            )
        resolved = PoseInferenceModelSpec(
            inference_mode=mode,
            bottomup_model_path=Path(spec.bottomup_model_path).expanduser().resolve(),
        )
    else:
        if spec.bottomup_model_path is not None:
            raise PoseInferenceError(
                "Top-down inference cannot use the bottomup_model_path field."
            )
        missing = []
        if spec.centroid_model_path is None:
            missing.append("centroid_model_path")
        if spec.centered_instance_model_path is None:
            missing.append("centered_instance_model_path")
        if missing:
            raise PoseInferenceError(
                "Top-down inference requires both component model paths; missing: "
                + ", ".join(missing)
            )
        centroid = Path(spec.centroid_model_path).expanduser().resolve()
        centered_instance = Path(spec.centered_instance_model_path).expanduser().resolve()
        if centroid == centered_instance:
            raise PoseInferenceError(
                "Top-down centroid and centered-instance model paths must be distinct."
            )
        resolved = PoseInferenceModelSpec(
            inference_mode=mode,
            centroid_model_path=centroid,
            centered_instance_model_path=centered_instance,
        )

    for path in _model_paths(resolved):
        _validate_model_path(path)
    _validate_model_roles(resolved)
    return resolved


def _model_paths(spec: PoseInferenceModelSpec) -> tuple[Path, ...]:
    if spec.inference_mode == "bottomup":
        if spec.bottomup_model_path is None:
            raise PoseInferenceError("Bottom-up inference requires bottomup_model_path.")
        return (spec.bottomup_model_path,)
    if spec.centroid_model_path is None or spec.centered_instance_model_path is None:
        raise PoseInferenceError("Top-down inference requires both component model paths.")
    return (spec.centroid_model_path, spec.centered_instance_model_path)


def _load_model_role(model_path: Path) -> str | None:
    if model_path.is_dir():
        config_root = model_path
        config_path = next(
            (
                config_root / filename
                for filename in ("training_config.yaml", "training_config.json")
                if (config_root / filename).is_file()
            ),
            None,
        )
    elif model_path.is_file() and model_path.name in {
        "training_config.yaml",
        "training_config.json",
    }:
        config_path = model_path
    elif model_path.is_file():
        config_root = model_path.parent
        config_path = next(
            (
                config_root / filename
                for filename in ("training_config.yaml", "training_config.json")
                if (config_root / filename).is_file()
            ),
            None,
        )
    else:
        config_path = None
    if config_path is None:
        return None
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise PoseInferenceError(
            f"SLEAP model training configuration is unreadable: {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PoseInferenceError(
            f"SLEAP model training configuration must contain a mapping: {config_path}"
        )
    model_config = payload.get("model_config")
    head_configs = (
        model_config.get("head_configs") if isinstance(model_config, Mapping) else None
    )
    if not isinstance(head_configs, Mapping):
        return None
    active_roles = [str(key) for key, value in head_configs.items() if value is not None]
    if len(active_roles) != 1:
        return None
    return active_roles[0].strip().lower().replace("-", "_")


def _validate_model_roles(spec: PoseInferenceModelSpec) -> None:
    if spec.inference_mode == "bottomup":
        role = _load_model_role(_model_paths(spec)[0])
        if role is not None and role not in {"bottomup", "multi_class_bottomup"}:
            raise PoseInferenceError(
                "Bottom-up inference model metadata identifies an incompatible "
                f"'{role}' head."
            )
        return

    centroid, centered_instance = _model_paths(spec)
    observed = {
        "centroid": _load_model_role(centroid),
        "centered_instance": _load_model_role(centered_instance),
    }
    expected = {"centroid": "centroid", "centered_instance": "centered_instance"}
    for component, expected_role in expected.items():
        role = observed[component]
        if role is None:
            path = centroid if component == "centroid" else centered_instance
            raise PoseInferenceError(
                f"Could not identify the top-down {component} model role from "
                f"training_config.yaml or training_config.json: {path}"
            )
        if role != expected_role:
            raise PoseInferenceError(
                f"Top-down {component} model metadata identifies '{role}', expected "
                f"'{expected_role}'."
            )


def _validate_profile_mode(
    profile: Mapping[str, Any], spec: PoseInferenceModelSpec
) -> None:
    configured = profile.get("inference_mode", "bottomup")
    if not isinstance(configured, str):
        raise PoseInferenceError("Inference profile inference_mode must be text.")
    profile_mode = _normalize_inference_mode(configured)
    if profile_mode != spec.inference_mode:
        raise PoseInferenceError(
            f"Inference profile mode '{profile_mode}' does not match selected model "
            f"mode '{spec.inference_mode}'."
        )


def _model_identity(path: Path) -> str:
    return _sanitize_identifier(path.stem if path.is_file() else path.name)


def _component_model_identity(path: Path) -> str:
    if path.is_file() and path.name in {
        "best.ckpt",
        "training_config.yaml",
        "training_config.json",
    }:
        return _sanitize_identifier(path.parent.name)
    return _model_identity(path)


def _model_bundle_id(spec: PoseInferenceModelSpec) -> str:
    if spec.inference_mode == "bottomup":
        return _model_identity(_model_paths(spec)[0])
    centroid, centered_instance = _model_paths(spec)
    value = _sanitize_identifier(
        "topdown-"
        f"{_component_model_identity(centroid)}-"
        f"{_component_model_identity(centered_instance)}"
    )
    if len(value) <= 80:
        return value
    digest = hashlib.sha256(
        f"{_path_text(centroid)}\0{_path_text(centered_instance)}".encode()
    ).hexdigest()[:10]
    return f"{value[:69].rstrip('._-')}-{digest}"


def _model_metadata(spec: PoseInferenceModelSpec, model_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "inference_mode": spec.inference_mode,
        "model_id": model_id,
    }
    if spec.inference_mode == "bottomup":
        model_path = _model_paths(spec)[0]
        payload["model_path"] = _path_text(model_path)
        return payload
    centroid, centered_instance = _model_paths(spec)
    payload["centroid"] = {
        "model_id": _component_model_identity(centroid),
        "model_path": _path_text(centroid),
    }
    payload["centered_instance"] = {
        "model_id": _component_model_identity(centered_instance),
        "model_path": _path_text(centered_instance),
    }
    return payload


def _profile_fraction(profile: Mapping[str, Any], key: str, default: float) -> float:
    value = profile.get(key, default)
    if isinstance(value, bool):
        raise PoseInferenceError(f"Inference profile {key} must be numeric.")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PoseInferenceError(f"Inference profile {key} must be numeric.") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PoseInferenceError(f"Inference profile {key} must be between 0 and 1.")
    return threshold


def _pose_qc_options(profile: Mapping[str, Any]) -> dict[str, Any]:
    tracking = profile.get("tracking")
    configured_expected = profile.get("expected_animals")
    if configured_expected is None and isinstance(tracking, Mapping):
        configured_expected = tracking.get("max_tracks")
    expected_animals = (
        DEFAULT_EXPECTED_ANIMALS
        if configured_expected is None
        else configured_expected
    )
    if isinstance(expected_animals, bool) or not isinstance(expected_animals, int):
        raise PoseInferenceError("Inference profile expected animal count must be an integer.")
    if expected_animals < 1:
        raise PoseInferenceError("Inference profile expected animal count must be positive.")
    return {
        "expected_animals": expected_animals,
        "review_one_animal_fraction_threshold": _profile_fraction(
            profile,
            "review_one_animal_fraction_threshold",
            DEFAULT_REVIEW_ONE_ANIMAL_FRACTION_THRESHOLD,
        ),
        "review_missing_keypoint_fraction_threshold": _profile_fraction(
            profile,
            "review_missing_keypoint_fraction_threshold",
            DEFAULT_REVIEW_MISSING_KEYPOINT_FRACTION_THRESHOLD,
        ),
        "review_frame_missing_keypoint_fraction_threshold": _profile_fraction(
            profile,
            "review_frame_missing_keypoint_fraction_threshold",
            DEFAULT_REVIEW_FRAME_MISSING_KEYPOINT_FRACTION_THRESHOLD,
        ),
    }


def load_inference_profile(profile_path: Path) -> dict[str, Any]:
    """Load the default SLEAP-NN inference profile YAML."""

    path = Path(profile_path).expanduser().resolve()
    if not path.is_file():
        raise PoseInferenceError(f"Inference profile does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            profile = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise PoseInferenceError(f"Inference profile YAML is malformed: {path}: {exc}") from exc
    except OSError as exc:
        raise PoseInferenceError(f"Could not read inference profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise PoseInferenceError("Inference profile must contain a top-level mapping.")
    if "tracking" in profile and not isinstance(profile["tracking"], dict):
        raise PoseInferenceError("Inference profile tracking section must be a mapping.")
    if profile.get("output_format") not in {None, "slp"}:
        raise PoseInferenceError(
            "The current inference backend requires profile output_format=slp."
        )
    _pose_qc_options(profile)
    return profile


def _append_cli_option(command: list[str], option: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(option)
        return
    command.extend([option, str(value)])


def _profile_value(profile: Mapping[str, Any], key: str) -> Any:
    value = profile.get(key)
    return None if value == "" else value


def build_sleap_predict_command(
    *,
    handoff: S1Handoff,
    model_path: Path | None = None,
    model_spec: PoseInferenceModelSpec | None = None,
    profile: Mapping[str, Any],
    pose_slp_path: Path,
) -> tuple[str, ...]:
    """Build one ``sleap-nn predict`` command that writes exactly ``pose.slp``."""

    if model_spec is not None and model_path is not None:
        raise PoseInferenceError(
            "Command model selection is ambiguous: provide model_path or model_spec."
        )
    if model_spec is None:
        if model_path is None:
            raise PoseInferenceError("Bottom-up command requires model_path.")
        model_spec = PoseInferenceModelSpec(
            inference_mode="bottomup",
            bottomup_model_path=Path(model_path).expanduser().resolve(),
        )
    else:
        model_spec = PoseInferenceModelSpec(
            inference_mode=_normalize_inference_mode(model_spec.inference_mode),
            bottomup_model_path=model_spec.bottomup_model_path,
            centroid_model_path=model_spec.centroid_model_path,
            centered_instance_model_path=model_spec.centered_instance_model_path,
        )
    _validate_profile_mode(profile, model_spec)

    command = [
        "sleap-nn",
        "predict",
        "--data_path",
        _path_text(handoff.prepared_video),
    ]
    for selected_model_path in _model_paths(model_spec):
        command.extend(["--model_paths", _path_text(selected_model_path)])
    command.extend(["--output_path", _path_text(pose_slp_path)])
    for key, option in (
        ("output_format", "--output_format"),
        ("device", "--device"),
        ("batch_size", "--batch_size"),
        ("max_instances", "--max_instances"),
        ("peak_threshold", "--peak_threshold"),
        ("integral_refinement", "--integral_refinement"),
        ("integral_patch_size", "--integral_patch_size"),
        ("max_edge_length_ratio", "--max_edge_length_ratio"),
        ("dist_penalty_weight", "--dist_penalty_weight"),
        ("n_points", "--n_points"),
        ("min_line_scores", "--min_line_scores"),
        ("min_instance_peaks", "--min_instance_peaks"),
    ):
        _append_cli_option(command, option, _profile_value(profile, key))

    tracking = profile.get("tracking")
    if isinstance(tracking, Mapping) and tracking.get("enabled") is True:
        command.append("--tracking")
        for key, option in (
            ("candidates_method", "--candidates_method"),
            ("max_tracks", "--max_tracks"),
            ("tracking_window_size", "--tracking_window_size"),
            ("min_new_track_points", "--min_new_track_points"),
            ("min_match_points", "--min_match_points"),
            ("track_matching_method", "--track_matching_method"),
            ("features", "--features"),
            ("scoring_method", "--scoring_method"),
            ("scoring_reduction", "--scoring_reduction"),
            ("robust_best_instance", "--robust_best_instance"),
        ):
            _append_cli_option(command, option, tracking.get(key))

    return tuple(command)


def _resolve_run_paths(
    *,
    session_root: Path,
    model_id: str,
    timestamp: str,
    output_root: Path | None,
) -> tuple[str, RunPaths]:
    run_id = f"{model_id}__{timestamp}"
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / "pose_inference"
    )
    run_dir = root / run_id
    return run_id, RunPaths(
        run_dir=run_dir,
        pose_slp=run_dir / "pose.slp",
        pose_parquet=run_dir / "pose.parquet",
        overlay_mp4=run_dir / "overlay.mp4",
        pose_meta=run_dir / "pose_meta.json",
        settings_used=run_dir / "settings_used.yaml",
        job_manifest=run_dir / "job_manifest.yaml",
        processing_log=run_dir / "processing_log.txt",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_meta_fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": _path_text(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _artifact_status(
    *,
    dry_run: bool,
    pose_slp_exists: bool,
    pose_parquet_status: str = "not_generated",
    overlay_status: str = "not_generated",
) -> dict[str, str]:
    if dry_run:
        pose_status = "dry_run_not_generated"
    elif pose_slp_exists:
        pose_status = "generated"
    else:
        pose_status = "pending"
    return {
        "pose_slp": pose_status,
        "pose_parquet": pose_parquet_status,
        "overlay_mp4": overlay_status,
    }


def _not_generated_parquet_summary(path: Path) -> ParquetExportSummary:
    return ParquetExportSummary(
        status="not_generated",
        path=path,
        schema_version=POSE_PARQUET_SCHEMA_VERSION,
        rows=0,
        timestamp_sources=("unavailable",),
    )


def _failed_parquet_summary(path: Path, error: BaseException) -> ParquetExportSummary:
    return ParquetExportSummary(
        status="failed",
        path=path,
        schema_version=POSE_PARQUET_SCHEMA_VERSION,
        rows=0,
        timestamp_sources=("unavailable",),
        error=str(error),
    )


def _not_computed_pose_qc_summary(reason: str) -> PoseQcSummary:
    return PoseQcSummary(
        status="not_computed",
        payload={
            "status": "not_computed",
            "schema_version": POSE_QC_SCHEMA_VERSION,
            "reason": reason,
        },
    )


def _failed_pose_qc_summary(error: BaseException) -> PoseQcSummary:
    return PoseQcSummary(
        status="failed",
        payload={
            "status": "failed",
            "schema_version": POSE_QC_SCHEMA_VERSION,
            "outcome": "failed",
            "failure_reasons": ["pose_qc_computation_failed"],
            "error": str(error),
        },
        error=str(error),
    )


def _technical_failure_pose_qc_summary(
    summary: PoseQcSummary,
    reason: str,
) -> PoseQcSummary:
    payload = summary.to_metadata()
    failure_reasons = list(payload.get("failure_reasons", []))
    if reason not in failure_reasons:
        failure_reasons.append(reason)
    payload["outcome"] = "failed"
    payload["failure_reasons"] = failure_reasons
    return PoseQcSummary(status=summary.status, payload=payload, error=summary.error)


def _not_generated_overlay_summary(path: Path, reason: str) -> OverlaySummary:
    return OverlaySummary(
        status="not_generated",
        path=path,
        source="pose.parquet",
        codec=DEFAULT_CODEC,
        reason=reason,
    )


def _failed_overlay_summary(path: Path, error: BaseException) -> OverlaySummary:
    return OverlaySummary(
        status="failed",
        path=path,
        source="pose.parquet",
        codec=DEFAULT_CODEC,
        error=str(error),
    )


def _not_generated_sleap_provenance_summary(reason: str) -> SleapProvenanceSummary:
    return SleapProvenanceSummary(status="not_generated", reason=reason)


def _failed_sleap_provenance_summary(error: BaseException) -> SleapProvenanceSummary:
    return SleapProvenanceSummary(status="failed", error=str(error))


def _settings_payload(
    *,
    profile_path: Path,
    model_spec: PoseInferenceModelSpec,
    model_id: str,
    profile: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    tracking = profile.get("tracking") if isinstance(profile.get("tracking"), Mapping) else {}
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "inference_mode": model_spec.inference_mode,
        "model": _model_metadata(model_spec, model_id),
        "profile": {
            "profile_path": _path_text(profile_path),
            "profile_id": profile.get("profile_id"),
        },
        "inference": dict(profile),
        "tracking_enabled": bool(tracking.get("enabled")) if isinstance(tracking, Mapping) else False,
        "command": list(command),
    }


def _output_paths_payload(paths: RunPaths) -> dict[str, str]:
    return {
        "pose_slp": _path_text(paths.pose_slp),
        "pose_parquet": _path_text(paths.pose_parquet),
        "overlay_mp4": _path_text(paths.overlay_mp4),
        "pose_meta": _path_text(paths.pose_meta),
        "settings_used": _path_text(paths.settings_used),
        "job_manifest": _path_text(paths.job_manifest),
        "processing_log": _path_text(paths.processing_log),
    }


def _pose_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    handoff: S1Handoff,
    paths: RunPaths,
    model_spec: PoseInferenceModelSpec,
    model_id: str,
    profile_path: Path,
    profile: Mapping[str, Any],
    dry_run: bool,
    status: str,
    command: Sequence[str],
    returncode: int | None,
    parquet_summary: ParquetExportSummary,
    pose_qc_summary: PoseQcSummary,
    overlay_summary: OverlaySummary,
    sleap_provenance: SleapProvenanceSummary,
) -> dict[str, Any]:
    tracking = profile.get("tracking") if isinstance(profile.get("tracking"), Mapping) else {}
    return {
        "schema_version": POSE_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "status": status,
        "dry_run": dry_run,
        "inference_mode": model_spec.inference_mode,
        "input": {
            "session_root": _path_text(handoff.session_root),
            "preprocess_dir": _path_text(handoff.preprocess_dir),
            "prepared_video": _path_text(handoff.prepared_video),
            "prepare_meta": _path_text(handoff.prepare_meta),
            "prepared_sync": _path_text(handoff.prepared_sync),
        },
        "outputs": {
            "pose_slp": _path_text(paths.pose_slp),
            "pose_parquet": _path_text(paths.pose_parquet),
            "overlay_mp4": _path_text(paths.overlay_mp4),
        },
        "model": _model_metadata(model_spec, model_id),
        "settings": {
            "tracking_enabled": bool(tracking.get("enabled"))
            if isinstance(tracking, Mapping)
            else False,
            "profile_path": _path_text(profile_path),
            "profile_id": profile.get("profile_id"),
        },
        "command": list(command),
        "returncode": returncode,
        "pose_qc": pose_qc_summary.to_metadata(),
        "overlay": overlay_summary.to_metadata(),
        "parquet": parquet_summary.to_metadata(),
        "sleap_provenance": sleap_provenance.to_metadata(),
        "artifact_status": _artifact_status(
            dry_run=dry_run,
            pose_slp_exists=paths.pose_slp.is_file(),
            pose_parquet_status=parquet_summary.status,
            overlay_status=overlay_summary.status,
        ),
    }


def _manifest_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    handoff: S1Handoff,
    paths: RunPaths,
    model_spec: PoseInferenceModelSpec,
    model_id: str,
    profile_path: Path,
    command: Sequence[str],
    dry_run: bool,
    status: str,
    returncode: int | None,
    parquet_summary: ParquetExportSummary,
    pose_qc_summary: PoseQcSummary,
    overlay_summary: OverlaySummary,
    sleap_provenance: SleapProvenanceSummary,
) -> dict[str, Any]:
    provenance_metadata = sleap_provenance.to_metadata()
    parquet_metadata = parquet_summary.to_metadata()
    pose_qc_metadata = pose_qc_summary.to_metadata()
    flagged_intervals = pose_qc_metadata.get("flagged_intervals", {})
    flagged_interval_count = (
        sum(len(value) for value in flagged_intervals.values())
        if isinstance(flagged_intervals, Mapping)
        else 0
    )
    return {
        "schema_version": JOB_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "dry_run": dry_run,
        "inference_mode": model_spec.inference_mode,
        "input": {
            "session_root": _path_text(handoff.session_root),
            "preprocess_dir": _path_text(handoff.preprocess_dir),
            "prepared_video": {"path": _path_text(handoff.prepared_video), "sha256": None},
            "prepare_meta": _prepare_meta_fingerprint(handoff.prepare_meta),
            "prepared_sync": {"path": _path_text(handoff.prepared_sync), "sha256": None},
        },
        "outputs": _output_paths_payload(paths),
        "model": _model_metadata(model_spec, model_id),
        "profile_path": _path_text(profile_path),
        "command": list(command),
        "returncode": returncode,
        "qc_status": pose_qc_summary.status,
        "qc_outcome": pose_qc_metadata.get("outcome"),
        "qc_review_recommendation_reasons": pose_qc_metadata.get(
            "review_recommendation_reasons", []
        ),
        "qc_review_warning_count": pose_qc_metadata.get("review_warning_count", 0),
        "qc_flagged_interval_count": flagged_interval_count,
        "overlay_status": overlay_summary.status,
        "sleap_provenance_status": sleap_provenance.status,
        "sleap_provenance_error": sleap_provenance.error,
        "effective_sleap_inference_config": provenance_metadata.get("inference_config"),
        "effective_sleap_tracking_config": provenance_metadata.get("tracking_config"),
        "effective_sleap_provenance_source": SLEAP_PROVENANCE_SOURCE,
        "sleap_provenance_note": SLEAP_PROVENANCE_NOTE,
        "parquet_timing": parquet_metadata.get("timing"),
        "artifact_status": _artifact_status(
            dry_run=dry_run,
            pose_slp_exists=paths.pose_slp.is_file(),
            pose_parquet_status=parquet_summary.status,
            overlay_status=overlay_summary.status,
        ),
    }


def _write_processing_log(
    *,
    path: Path,
    created_at: str,
    completed_at: str,
    handoff: S1Handoff,
    paths: RunPaths,
    model_spec: PoseInferenceModelSpec,
    model_id: str,
    command: Sequence[str],
    dry_run: bool,
    status: str,
    returncode: int | None,
    parquet_summary: ParquetExportSummary,
    pose_qc_summary: PoseQcSummary,
    overlay_summary: OverlaySummary,
    sleap_provenance: SleapProvenanceSummary,
    stdout: str | None,
    stderr: str | None,
) -> None:
    pose_qc_metadata = pose_qc_summary.to_metadata()
    lines = [
        f"start_time: {created_at}",
        f"completed_time: {completed_at}",
        f"status: {status}",
        f"dry_run: {dry_run}",
        f"inference_mode: {model_spec.inference_mode}",
        f"model_id: {model_id}",
        f"preprocess_dir: {_path_text(handoff.preprocess_dir)}",
        f"prepared_video: {_path_text(handoff.prepared_video)}",
        f"prepare_meta: {_path_text(handoff.prepare_meta)}",
        f"prepared_sync: {_path_text(handoff.prepared_sync)}",
        f"run_dir: {_path_text(paths.run_dir)}",
        f"pose_slp: {_path_text(paths.pose_slp)}",
        "command:",
        "  " + " ".join(command),
        "command_json: " + json.dumps(list(command)),
        f"parquet_status: {parquet_summary.status}",
        f"pose_parquet: {parquet_summary.path.resolve()}",
        f"pose_qc: {pose_qc_summary.status}",
        f"pose_qc_outcome: {pose_qc_metadata.get('outcome', 'not_computed')}",
        f"overlay: {overlay_summary.status}",
        f"sleap_provenance_status: {sleap_provenance.status}",
        f"sleap_provenance_source: {sleap_provenance.source}",
        f"sleap_provenance_note: {SLEAP_PROVENANCE_NOTE}",
    ]
    if model_spec.inference_mode == "bottomup":
        lines.insert(6, f"model_path: {_path_text(_model_paths(model_spec)[0])}")
    else:
        centroid, centered_instance = _model_paths(model_spec)
        lines.insert(6, f"centroid_model_path: {_path_text(centroid)}")
        lines.insert(
            7,
            f"centered_instance_model_path: {_path_text(centered_instance)}",
        )
    if parquet_summary.error:
        lines.append(f"parquet_error: {parquet_summary.error}")
    reasons = pose_qc_metadata.get("review_recommendation_reasons", [])
    failure_reasons = pose_qc_metadata.get("failure_reasons", [])
    lines.append(
        "pose_qc_review_recommendation_reasons: "
        + (", ".join(str(reason) for reason in reasons) if reasons else "none")
    )
    lines.append(
        "pose_qc_failure_reasons: "
        + (
            ", ".join(str(reason) for reason in failure_reasons)
            if failure_reasons
            else "none"
        )
    )
    if pose_qc_summary.status == "computed":
        duplicate_risk = pose_qc_metadata.get("duplicate_candidate_risk", {})
        geometry = pose_qc_metadata.get("implausible_geometry", {})
        if isinstance(duplicate_risk, Mapping):
            lines.append(
                "pose_qc_duplicate_frames_flagged: "
                f"{duplicate_risk.get('frames_flagged', 0)}"
            )
        if isinstance(geometry, Mapping):
            lines.append(
                "pose_qc_implausible_geometry_frames_flagged: "
                f"{geometry.get('frames_flagged', 0)}"
            )
    if pose_qc_summary.error:
        lines.append(f"pose_qc_error: {pose_qc_summary.error}")
    if overlay_summary.frames_written:
        lines.append(f"overlay_frames_written: {overlay_summary.frames_written}")
    if overlay_summary.error:
        lines.append(f"overlay_error: {overlay_summary.error}")
    if sleap_provenance.error:
        lines.append(f"sleap_provenance_error: {sleap_provenance.error}")
    if returncode is not None:
        lines.append(f"subprocess_returncode: {returncode}")
    if stdout:
        lines.extend(["stdout:", stdout.rstrip()])
    if stderr:
        lines.extend(["stderr:", stderr.rstrip()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _validate_no_forbidden_outputs(paths: RunPaths) -> None:
    names = {path.name for path in paths.run_dir.iterdir()} if paths.run_dir.exists() else set()
    forbidden = sorted(names & FORBIDDEN_STANDARD_ARTIFACT_NAMES)
    if forbidden:
        raise PoseInferenceError(
            "Forbidden Subsystem 02 tracking artifacts were created: " + ", ".join(forbidden)
        )


def _validate_required_output_file(path: Path) -> None:
    if not path.is_file():
        raise PoseInferenceError(f"Required Subsystem 02 output is missing: {path.name}")
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise PoseInferenceError(
            f"Required Subsystem 02 output is unreadable: {path.name}: {exc}"
        ) from exc


def run_pose_inference(
    request: PoseInferenceRequest,
    *,
    subprocess_runner: SubprocessRunner = subprocess.run,
    parquet_exporter: ParquetExporter = export_pose_parquet,
    pose_qc_computer: PoseQcComputer = compute_pose_qc_from_parquet,
    overlay_generator: OverlayGenerator = generate_overlay_video,
    sleap_provenance_reader: SleapProvenanceReader = read_sleap_provenance,
) -> PoseInferenceResult:
    """Run or dry-run one Subsystem 02 SLEAP-NN inference request."""

    handoff = validate_s1_handoff(request.session_root)
    _validate_prepared_video_against_sync(handoff)
    profile_path = Path(request.profile_path).expanduser().resolve()
    profile = load_inference_profile(profile_path)
    pose_qc_options = _pose_qc_options(profile)
    model_spec = _resolve_model_spec(request)
    _validate_profile_mode(profile, model_spec)

    model_id = _model_bundle_id(model_spec)
    timestamp = request.timestamp or _now_timestamp()
    run_id, paths = _resolve_run_paths(
        session_root=handoff.session_root,
        model_id=model_id,
        timestamp=timestamp,
        output_root=request.output_root,
    )
    if paths.run_dir.exists():
        raise PoseInferenceError(f"Output run directory already exists: {paths.run_dir}")
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PoseInferenceError(
            f"Output location cannot be created or written: {paths.run_dir}: {exc}"
        ) from exc

    created_at = _iso_now()
    command = build_sleap_predict_command(
        handoff=handoff,
        model_spec=model_spec,
        profile=profile,
        pose_slp_path=paths.pose_slp,
    )

    stdout = None
    stderr = None
    returncode = None
    parquet_summary = _not_generated_parquet_summary(paths.pose_parquet)
    pose_qc_summary = _not_computed_pose_qc_summary("dry_run or parquet not generated")
    overlay_summary = _not_generated_overlay_summary(paths.overlay_mp4, "dry_run or parquet not generated")
    sleap_provenance = _not_generated_sleap_provenance_summary("dry_run or pose.slp not generated")
    if request.dry_run:
        status = "dry_run_complete"
        success = True
    else:
        completed = subprocess_runner(
            list(command),
            cwd=paths.run_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        if completed.returncode != 0:
            pose_qc_summary = _technical_failure_pose_qc_summary(
                pose_qc_summary,
                "inference_subprocess_failed",
            )
            status = "inference_failed"
            success = False
        elif paths.pose_slp.is_file():
            try:
                _validate_required_output_file(paths.pose_slp)
                sleap_provenance = sleap_provenance_reader(paths.pose_slp)
            except Exception as exc:
                sleap_provenance = _failed_sleap_provenance_summary(exc)
            try:
                parquet_summary = parquet_exporter(
                    pose_slp_path=paths.pose_slp,
                    preprocess_dir=handoff.preprocess_dir,
                    output_path=paths.pose_parquet,
                )
            except ParquetExportError as exc:
                parquet_summary = _failed_parquet_summary(paths.pose_parquet, exc)
                pose_qc_summary = _not_computed_pose_qc_summary("parquet export failed")
                pose_qc_summary = _technical_failure_pose_qc_summary(
                    pose_qc_summary,
                    "parquet_export_failed",
                )
                overlay_summary = _not_generated_overlay_summary(paths.overlay_mp4, "parquet export failed")
                status = "export_failed"
                success = False
            else:
                try:
                    _validate_required_output_file(paths.pose_parquet)
                    pose_qc_summary = pose_qc_computer(
                        pose_parquet_path=paths.pose_parquet,
                        **pose_qc_options,
                    )
                except PoseInferenceError as exc:
                    parquet_summary = _failed_parquet_summary(paths.pose_parquet, exc)
                    pose_qc_summary = _not_computed_pose_qc_summary(
                        "required pose.parquet validation failed"
                    )
                    pose_qc_summary = _technical_failure_pose_qc_summary(
                        pose_qc_summary,
                        "required_pose_parquet_missing_or_unreadable",
                    )
                    overlay_summary = _not_generated_overlay_summary(
                        paths.overlay_mp4,
                        "required pose.parquet validation failed",
                    )
                    status = "export_failed"
                    success = False
                except PoseQcError as exc:
                    pose_qc_summary = _failed_pose_qc_summary(exc)
                    overlay_summary = _not_generated_overlay_summary(paths.overlay_mp4, "pose-quality QC failed")
                    status = "qc_failed"
                    success = False
                else:
                    if pose_qc_summary.to_metadata().get("outcome") == "failed":
                        overlay_summary = _not_generated_overlay_summary(
                            paths.overlay_mp4,
                            "technical pose QC failed",
                        )
                        status = "qc_failed"
                        success = False
                    else:
                        try:
                            overlay_summary = overlay_generator(
                                prepared_video_path=handoff.prepared_video,
                                pose_parquet_path=paths.pose_parquet,
                                output_path=paths.overlay_mp4,
                            )
                            _validate_required_output_file(paths.overlay_mp4)
                        except (OverlayGenerationError, PoseInferenceError) as exc:
                            overlay_summary = _failed_overlay_summary(paths.overlay_mp4, exc)
                            pose_qc_summary = _technical_failure_pose_qc_summary(
                                pose_qc_summary,
                                "required_overlay_missing_unreadable_or_failed",
                            )
                            status = "overlay_failed"
                            success = False
                        else:
                            status = "success"
                            success = True
        else:
            pose_qc_summary = _technical_failure_pose_qc_summary(
                pose_qc_summary,
                "required_pose_slp_missing",
            )
            status = "validation_failed"
            success = False

    completed_at = _iso_now()
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            profile_path=profile_path,
            model_spec=model_spec,
            model_id=model_id,
            profile=profile,
            command=command,
        ),
    )
    _write_json(
        paths.pose_meta,
        _pose_meta_payload(
            run_id=run_id,
            run_purpose=request.run_purpose,
            created_at=created_at,
            handoff=handoff,
            paths=paths,
            model_spec=model_spec,
            model_id=model_id,
            profile_path=profile_path,
            profile=profile,
            dry_run=request.dry_run,
            status=status,
            command=command,
            returncode=returncode,
            parquet_summary=parquet_summary,
            pose_qc_summary=pose_qc_summary,
            overlay_summary=overlay_summary,
            sleap_provenance=sleap_provenance,
        ),
    )
    _write_yaml(
        paths.job_manifest,
        _manifest_payload(
            run_id=run_id,
            run_purpose=request.run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            handoff=handoff,
            paths=paths,
            model_spec=model_spec,
            model_id=model_id,
            profile_path=profile_path,
            command=command,
            dry_run=request.dry_run,
            status=status,
            returncode=returncode,
            parquet_summary=parquet_summary,
            pose_qc_summary=pose_qc_summary,
            overlay_summary=overlay_summary,
            sleap_provenance=sleap_provenance,
        ),
    )
    _write_processing_log(
        path=paths.processing_log,
        created_at=created_at,
        completed_at=completed_at,
        handoff=handoff,
        paths=paths,
        model_spec=model_spec,
        model_id=model_id,
        command=command,
        dry_run=request.dry_run,
        status=status,
        returncode=returncode,
        parquet_summary=parquet_summary,
        pose_qc_summary=pose_qc_summary,
        overlay_summary=overlay_summary,
        sleap_provenance=sleap_provenance,
        stdout=stdout,
        stderr=stderr,
    )
    for required_metadata in (
        paths.pose_meta,
        paths.settings_used,
        paths.job_manifest,
        paths.processing_log,
    ):
        _validate_required_output_file(required_metadata)
    _validate_no_forbidden_outputs(paths)

    return PoseInferenceResult(
        success=success,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        pose_slp_path=paths.pose_slp,
        pose_meta_path=paths.pose_meta,
        settings_used_path=paths.settings_used,
        job_manifest_path=paths.job_manifest,
        processing_log_path=paths.processing_log,
        command=command,
        returncode=returncode,
    )
