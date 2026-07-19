"""Lightweight discovery of Subsystem 02 pose inference runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REQUIRED_S1_HANDOFF_FILES = (
    "prepared_video.mp4",
    "prepare_meta.json",
    "prepared_sync.npz",
)

REQUIRED_S2_ARTIFACTS = {
    "pose_slp": "pose.slp",
    "pose_parquet": "pose.parquet",
    "overlay_mp4": "overlay.mp4",
    "pose_meta": "pose_meta.json",
    "settings_used": "settings_used.yaml",
    "job_manifest": "job_manifest.yaml",
    "processing_log": "processing_log.txt",
}

COMPLETE_REVIEWABLE = "complete_reviewable"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"

_INCOMPLETE_STATUSES = {
    "canceled",
    "cancelled",
    "dry_run",
    "dry_run_complete",
    "incomplete",
    "not_generated",
    "pending",
    "running",
}


@dataclass(frozen=True)
class S1HandoffStatus:
    """Presence summary for the Subsystem 01 to Subsystem 02 handoff."""

    session_root: Path
    preprocess_dir: Path
    prepared_video: Path
    prepare_meta: Path
    prepared_sync: Path
    artifact_presence: Mapping[str, bool]
    missing_files: tuple[str, ...]
    is_complete: bool


@dataclass(frozen=True)
class PoseInferenceRunSummary:
    """Lightweight summary of one Subsystem 02 inference run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    artifact_presence: Mapping[str, bool]
    missing_required_artifacts: tuple[str, ...]
    model_id: str | None = None
    model_path: str | None = None
    inference_mode: str | None = None
    bottomup_model_id: str | None = None
    bottomup_model_path: str | None = None
    centroid_model_id: str | None = None
    centroid_model_path: str | None = None
    centered_instance_model_id: str | None = None
    centered_instance_model_path: str | None = None
    profile_id: str | None = None
    profile_path: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    parquet_status: str | None = None
    pose_qc_status: str | None = None
    pose_qc_outcome: str | None = None
    review_recommendation_reasons: tuple[str, ...] = ()
    review_warning_count: int | None = None
    flagged_intervals: Any = None
    overlay_status: str | None = None
    sleap_provenance_status: str | None = None
    effective_sleap_inference_config: Any = None
    effective_sleap_tracking_config: Any = None
    parquet_timing: Any = None
    dry_run: bool | None = None
    metadata_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoseInferenceProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    s1_handoff: S1HandoffStatus
    runs: tuple[PoseInferenceRunSummary, ...]


def summarize_pose_inference_project(session_root: Path | str) -> PoseInferenceProjectSummary:
    """Summarize S1 handoff status and existing S2 runs for a session root."""

    root = Path(session_root).expanduser().resolve()
    s1_handoff = _summarize_s1_handoff(root)
    run_root = root / "pose_inference"
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    return PoseInferenceProjectSummary(session_root=root, s1_handoff=s1_handoff, runs=runs)


def _summarize_s1_handoff(session_root: Path) -> S1HandoffStatus:
    preprocess_dir = session_root / "preprocess"
    paths = {
        "prepared_video": preprocess_dir / "prepared_video.mp4",
        "prepare_meta": preprocess_dir / "prepare_meta.json",
        "prepared_sync": preprocess_dir / "prepared_sync.npz",
    }
    artifact_presence = {key: path.is_file() for key, path in paths.items()}
    missing = tuple(
        path.name for key, path in paths.items() if not artifact_presence[key]
    )
    return S1HandoffStatus(
        session_root=session_root,
        preprocess_dir=preprocess_dir,
        prepared_video=paths["prepared_video"],
        prepare_meta=paths["prepare_meta"],
        prepared_sync=paths["prepared_sync"],
        artifact_presence=artifact_presence,
        missing_files=missing,
        is_complete=not missing,
    )


def _discover_runs(run_root: Path) -> list[PoseInferenceRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir()
    ]


def _summarize_run(run_dir: Path) -> PoseInferenceRunSummary:
    artifact_paths = {
        artifact_key: run_dir / filename
        for artifact_key, filename in REQUIRED_S2_ARTIFACTS.items()
    }
    artifact_presence = {
        artifact_key: path.is_file()
        for artifact_key, path in artifact_paths.items()
    }
    missing = tuple(
        filename
        for artifact_key, filename in REQUIRED_S2_ARTIFACTS.items()
        if not artifact_presence[artifact_key]
    )

    metadata_errors: list[str] = []
    job_manifest = _load_yaml_mapping(artifact_paths["job_manifest"], metadata_errors)
    pose_meta = _load_json_mapping(artifact_paths["pose_meta"], metadata_errors)
    settings_used = _load_yaml_mapping(artifact_paths["settings_used"], metadata_errors)
    processing_log = _load_processing_log(artifact_paths["processing_log"], metadata_errors)

    status = _first_text(
        _mapping_get(job_manifest, "status"),
        _mapping_get(pose_meta, "status"),
        processing_log.get("status"),
    )
    dry_run = _first_bool(
        _mapping_get(job_manifest, "dry_run"),
        _mapping_get(pose_meta, "dry_run"),
        processing_log.get("dry_run"),
    )
    parquet_status = _first_text(
        _mapping_get(job_manifest, "artifact_status", "pose_parquet"),
        _mapping_get(pose_meta, "artifact_status", "pose_parquet"),
        _mapping_get(pose_meta, "parquet", "status"),
        processing_log.get("parquet_status"),
    )
    pose_qc_status = _first_text(
        _mapping_get(job_manifest, "qc_status"),
        _mapping_get(pose_meta, "pose_qc", "status"),
        processing_log.get("pose_qc"),
    )
    pose_qc_outcome = _first_text(
        _mapping_get(job_manifest, "qc_outcome"),
        _mapping_get(pose_meta, "pose_qc", "outcome"),
        processing_log.get("pose_qc_outcome"),
    )
    review_recommendation_reasons = _text_tuple(
        _first_existing(
            _mapping_get(job_manifest, "qc_review_recommendation_reasons"),
            _mapping_get(pose_meta, "pose_qc", "review_recommendation_reasons"),
        )
    )
    review_warning_count = _optional_int(
        _first_existing(
            _mapping_get(job_manifest, "qc_review_warning_count"),
            _mapping_get(pose_meta, "pose_qc", "review_warning_count"),
        )
    )
    overlay_status = _first_text(
        _mapping_get(job_manifest, "overlay_status"),
        _mapping_get(pose_meta, "artifact_status", "overlay_mp4"),
        _mapping_get(pose_meta, "overlay", "status"),
        processing_log.get("overlay"),
    )
    sleap_provenance_status = _first_text(
        _mapping_get(job_manifest, "sleap_provenance_status"),
        _mapping_get(pose_meta, "sleap_provenance", "status"),
        processing_log.get("sleap_provenance_status"),
    )
    classification = _classify_run(
        missing=missing,
        status=status,
        dry_run=dry_run,
        metadata_errors=metadata_errors,
    )
    model_id = _first_text(
        _mapping_get(job_manifest, "model", "model_id"),
        _mapping_get(pose_meta, "model", "model_id"),
    )
    model_path = _first_text(
        _mapping_get(job_manifest, "model", "model_path"),
        _mapping_get(pose_meta, "model", "model_path"),
    )
    inference_mode = _first_text(
        _mapping_get(job_manifest, "inference_mode"),
        _mapping_get(pose_meta, "inference_mode"),
        _mapping_get(settings_used, "inference_mode"),
        _mapping_get(job_manifest, "model", "inference_mode"),
        _mapping_get(pose_meta, "model", "inference_mode"),
    )
    if inference_mode is None and model_path is not None:
        inference_mode = "bottomup"

    return PoseInferenceRunSummary(
        run_dir=run_dir.resolve(),
        run_id=run_dir.name,
        status=status,
        classification=classification,
        artifact_presence=artifact_presence,
        missing_required_artifacts=missing,
        model_id=model_id,
        model_path=model_path,
        inference_mode=inference_mode,
        bottomup_model_id=model_id if inference_mode == "bottomup" else None,
        bottomup_model_path=model_path if inference_mode == "bottomup" else None,
        centroid_model_id=_first_text(
            _mapping_get(job_manifest, "model", "centroid", "model_id"),
            _mapping_get(pose_meta, "model", "centroid", "model_id"),
        ),
        centroid_model_path=_first_text(
            _mapping_get(job_manifest, "model", "centroid", "model_path"),
            _mapping_get(pose_meta, "model", "centroid", "model_path"),
        ),
        centered_instance_model_id=_first_text(
            _mapping_get(job_manifest, "model", "centered_instance", "model_id"),
            _mapping_get(pose_meta, "model", "centered_instance", "model_id"),
        ),
        centered_instance_model_path=_first_text(
            _mapping_get(
                job_manifest, "model", "centered_instance", "model_path"
            ),
            _mapping_get(pose_meta, "model", "centered_instance", "model_path"),
        ),
        profile_id=_first_text(
            _mapping_get(pose_meta, "settings", "profile_id"),
            _mapping_get(settings_used, "profile_id"),
            _mapping_get(settings_used, "inference", "profile_id"),
        ),
        profile_path=_first_text(
            _mapping_get(job_manifest, "profile_path"),
            _mapping_get(pose_meta, "settings", "profile_path"),
            _mapping_get(settings_used, "profile_path"),
        ),
        created_at=_first_text(
            _mapping_get(job_manifest, "created_at"),
            _mapping_get(pose_meta, "created_at"),
            processing_log.get("start_time"),
        ),
        completed_at=_first_text(
            _mapping_get(job_manifest, "completed_at"),
            processing_log.get("completed_time"),
        ),
        parquet_status=parquet_status,
        pose_qc_status=pose_qc_status,
        pose_qc_outcome=pose_qc_outcome,
        review_recommendation_reasons=review_recommendation_reasons,
        review_warning_count=review_warning_count,
        flagged_intervals=_mapping_get(pose_meta, "pose_qc", "flagged_intervals"),
        overlay_status=overlay_status,
        sleap_provenance_status=sleap_provenance_status,
        effective_sleap_inference_config=_first_existing(
            _mapping_get(job_manifest, "effective_sleap_inference_config"),
            _mapping_get(pose_meta, "sleap_provenance", "inference_config"),
        ),
        effective_sleap_tracking_config=_first_existing(
            _mapping_get(job_manifest, "effective_sleap_tracking_config"),
            _mapping_get(pose_meta, "sleap_provenance", "tracking_config"),
        ),
        parquet_timing=_first_existing(
            _mapping_get(job_manifest, "parquet_timing"),
            _mapping_get(pose_meta, "parquet", "timing"),
        ),
        dry_run=dry_run,
        metadata_errors=tuple(metadata_errors),
    )


def _load_json_mapping(path: Path, errors: list[str]) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: {exc}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{path.name}: expected top-level object")
        return {}
    return payload


def _load_yaml_mapping(path: Path, errors: list[str]) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"{path.name}: {exc}")
        return {}
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"{path.name}: expected top-level object")
        return {}
    return payload


def _load_processing_log(path: Path, errors: list[str]) -> Mapping[str, str]:
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path.name}: {exc}")
        return {}
    payload: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip()
        if normalized:
            payload[normalized] = value.strip()
    return payload


def _mapping_get(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_existing(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_text(*values: Any) -> str | None:
    value = _first_existing(*values)
    if value is None:
        return None
    return str(value)


def _first_bool(*values: Any) -> bool | None:
    value = _first_existing(*values)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _classify_run(
    *,
    missing: tuple[str, ...],
    status: str | None,
    dry_run: bool | None,
    metadata_errors: list[str],
) -> str:
    if dry_run is True or _status_is_failed_or_incomplete(status):
        return FAILED_OR_INCOMPLETE
    if missing:
        return MISSING_REQUIRED_ARTIFACTS
    if metadata_errors:
        return FAILED_OR_INCOMPLETE
    return COMPLETE_REVIEWABLE


def _status_is_failed_or_incomplete(status: str | None) -> bool:
    if status is None:
        return False
    normalized = status.strip().lower()
    return (
        normalized in _INCOMPLETE_STATUSES
        or "failed" in normalized
        or "failure" in normalized
        or "error" in normalized
    )


def _run_sort_key(summary: PoseInferenceRunSummary) -> tuple[datetime, str]:
    for value in (summary.completed_at, summary.created_at):
        parsed = _parse_datetime(value)
        if parsed is not None:
            return (parsed, summary.run_id)
    parsed = _parse_run_id_timestamp(summary.run_id)
    if parsed is not None:
        return (parsed, summary.run_id)
    try:
        return (datetime.fromtimestamp(summary.run_dir.stat().st_mtime), summary.run_id)
    except OSError:
        return (datetime.min, summary.run_id)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _parse_run_id_timestamp(run_id: str) -> datetime | None:
    if "__" not in run_id:
        return None
    timestamp = run_id.rsplit("__", maxsplit=1)[-1]
    match = re.fullmatch(r"\d{8}T\d{6}", timestamp)
    if match is None:
        return None
    return datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
