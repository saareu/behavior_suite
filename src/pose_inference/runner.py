"""Initial Subsystem 02 SLEAP-NN inference runner.

This module intentionally avoids importing SLEAP or GPU libraries. The only
runtime boundary for real inference is the ``sleap-nn predict`` subprocess.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

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


class PoseInferenceError(RuntimeError):
    """Expected Subsystem 02 input, dispatch, or artifact error."""


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
class PoseInferenceRequest:
    """Inputs for one Subsystem 02 inference or dry-run."""

    session_root: Path
    model_path: Path
    profile_path: Path
    output_root: Path | None = None
    run_purpose: str = "development"
    dry_run: bool = False
    timestamp: str | None = None


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


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


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
    model_path: Path,
    profile: Mapping[str, Any],
    pose_slp_path: Path,
) -> tuple[str, ...]:
    """Build one ``sleap-nn predict`` command that writes exactly ``pose.slp``."""

    command = [
        "sleap-nn",
        "predict",
        "--data_path",
        _path_text(handoff.prepared_video),
        "--model_paths",
        _path_text(model_path),
        "--output_path",
        _path_text(pose_slp_path),
    ]
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


def _artifact_status(*, dry_run: bool, pose_slp_exists: bool) -> dict[str, str]:
    if dry_run:
        pose_status = "dry_run_not_generated"
    elif pose_slp_exists:
        pose_status = "generated"
    else:
        pose_status = "pending"
    return {
        "pose_slp": pose_status,
        "pose_parquet": "not_generated",
        "overlay_mp4": "not_generated",
    }


def _settings_payload(
    *,
    profile_path: Path,
    model_path: Path,
    model_id: str,
    profile: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    tracking = profile.get("tracking") if isinstance(profile.get("tracking"), Mapping) else {}
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "model": {
            "model_id": model_id,
            "model_path": _path_text(model_path),
        },
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
    model_path: Path,
    model_id: str,
    profile_path: Path,
    profile: Mapping[str, Any],
    dry_run: bool,
    status: str,
    command: Sequence[str],
    returncode: int | None,
) -> dict[str, Any]:
    tracking = profile.get("tracking") if isinstance(profile.get("tracking"), Mapping) else {}
    return {
        "schema_version": POSE_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "status": status,
        "dry_run": dry_run,
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
        "model": {
            "model_path": _path_text(model_path),
            "model_id": model_id,
        },
        "settings": {
            "tracking_enabled": bool(tracking.get("enabled"))
            if isinstance(tracking, Mapping)
            else False,
            "profile_path": _path_text(profile_path),
            "profile_id": profile.get("profile_id"),
        },
        "command": list(command),
        "returncode": returncode,
        "pose_qc": {
            "status": "not_computed",
            "reason": "first implementation; parquet export and pose QC are not implemented yet",
        },
        "artifact_status": _artifact_status(
            dry_run=dry_run,
            pose_slp_exists=paths.pose_slp.is_file(),
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
    model_path: Path,
    model_id: str,
    profile_path: Path,
    command: Sequence[str],
    dry_run: bool,
    status: str,
    returncode: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": JOB_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "dry_run": dry_run,
        "input": {
            "session_root": _path_text(handoff.session_root),
            "preprocess_dir": _path_text(handoff.preprocess_dir),
            "prepared_video": {"path": _path_text(handoff.prepared_video), "sha256": None},
            "prepare_meta": _prepare_meta_fingerprint(handoff.prepare_meta),
            "prepared_sync": {"path": _path_text(handoff.prepared_sync), "sha256": None},
        },
        "outputs": _output_paths_payload(paths),
        "model": {
            "model_id": model_id,
            "model_path": _path_text(model_path),
        },
        "profile_path": _path_text(profile_path),
        "command": list(command),
        "returncode": returncode,
        "artifact_status": _artifact_status(
            dry_run=dry_run,
            pose_slp_exists=paths.pose_slp.is_file(),
        ),
    }


def _write_processing_log(
    *,
    path: Path,
    created_at: str,
    completed_at: str,
    handoff: S1Handoff,
    paths: RunPaths,
    command: Sequence[str],
    dry_run: bool,
    status: str,
    returncode: int | None,
    stdout: str | None,
    stderr: str | None,
) -> None:
    lines = [
        f"start_time: {created_at}",
        f"completed_time: {completed_at}",
        f"status: {status}",
        f"dry_run: {dry_run}",
        f"preprocess_dir: {_path_text(handoff.preprocess_dir)}",
        f"prepared_video: {_path_text(handoff.prepared_video)}",
        f"prepare_meta: {_path_text(handoff.prepare_meta)}",
        f"prepared_sync: {_path_text(handoff.prepared_sync)}",
        f"run_dir: {_path_text(paths.run_dir)}",
        f"pose_slp: {_path_text(paths.pose_slp)}",
        "command:",
        "  " + " ".join(command),
    ]
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


def run_pose_inference(
    request: PoseInferenceRequest,
    *,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> PoseInferenceResult:
    """Run or dry-run one Subsystem 02 SLEAP-NN inference request."""

    handoff = validate_s1_handoff(request.session_root)
    profile_path = Path(request.profile_path).expanduser().resolve()
    profile = load_inference_profile(profile_path)
    model_path = Path(request.model_path).expanduser().resolve()
    if not model_path.exists():
        raise PoseInferenceError(f"Model path does not exist: {model_path}")

    model_id = _sanitize_identifier(model_path.stem if model_path.is_file() else model_path.name)
    timestamp = request.timestamp or _now_timestamp()
    run_id, paths = _resolve_run_paths(
        session_root=handoff.session_root,
        model_id=model_id,
        timestamp=timestamp,
        output_root=request.output_root,
    )
    if paths.run_dir.exists():
        raise PoseInferenceError(f"Output run directory already exists: {paths.run_dir}")
    paths.run_dir.mkdir(parents=True, exist_ok=False)

    created_at = _iso_now()
    command = build_sleap_predict_command(
        handoff=handoff,
        model_path=model_path,
        profile=profile,
        pose_slp_path=paths.pose_slp,
    )

    stdout = None
    stderr = None
    returncode = None
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
            status = "inference_failed"
            success = False
        elif paths.pose_slp.is_file():
            status = "success"
            success = True
        else:
            status = "validation_failed"
            success = False

    completed_at = _iso_now()
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            profile_path=profile_path,
            model_path=model_path,
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
            model_path=model_path,
            model_id=model_id,
            profile_path=profile_path,
            profile=profile,
            dry_run=request.dry_run,
            status=status,
            command=command,
            returncode=returncode,
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
            model_path=model_path,
            model_id=model_id,
            profile_path=profile_path,
            command=command,
            dry_run=request.dry_run,
            status=status,
            returncode=returncode,
        ),
    )
    _write_processing_log(
        path=paths.processing_log,
        created_at=created_at,
        completed_at=completed_at,
        handoff=handoff,
        paths=paths,
        command=command,
        dry_run=request.dry_run,
        status=status,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
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
