"""Lightweight discovery of Subsystem 03 tracking-correction runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tracking_correction.contracts import (
    MACHINE_CORRECTIONS_FILENAME,
    RUN_META_FILENAME,
    STATUS_CORRECTION_COMPLETE,
    WORKING_TRACKED_POSE_FILENAME,
)

COMPLETE_REVIEWABLE = "complete_reviewable"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"


@dataclass(frozen=True, slots=True)
class SessionArtifactPresence:
    """Presence of top-level pipeline directories under a session root."""

    preprocess: bool
    pose_inference: bool
    tracking_correction: bool


@dataclass(frozen=True, slots=True)
class TrackingCorrectionRunSummary:
    """Lightweight summary of one Subsystem 03 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    backend_status: str | None
    dry_run: bool | None
    classification: str
    artifact_presence: Mapping[str, bool]
    missing_required_artifacts: tuple[str, ...]
    profile_id: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    metadata_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrackingCorrectionProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    artifact_presence: SessionArtifactPresence
    runs: tuple[TrackingCorrectionRunSummary, ...]

    @property
    def reviewable_runs(self) -> tuple[TrackingCorrectionRunSummary, ...]:
        return tuple(run for run in self.runs if run.classification == COMPLETE_REVIEWABLE)


def summarize_tracking_correction_project(
    session_root: Path | str,
) -> TrackingCorrectionProjectSummary:
    """Summarize pipeline directories and existing S3 runs for a session root."""

    root = Path(session_root).expanduser().resolve()
    presence = SessionArtifactPresence(
        preprocess=(root / "preprocess").is_dir(),
        pose_inference=(root / "pose_inference").is_dir(),
        tracking_correction=(root / "tracking_correction").is_dir(),
    )
    run_root = root / "tracking_correction"
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    return TrackingCorrectionProjectSummary(
        session_root=root,
        artifact_presence=presence,
        runs=runs,
    )


def is_completed_s3_run_dir(path: Path | str) -> bool:
    """Return True when ``path`` looks like a reviewable completed S3 run."""

    run_dir = Path(path).expanduser().resolve()
    if not run_dir.is_dir():
        return False
    summary = _summarize_run(run_dir)
    return summary.classification == COMPLETE_REVIEWABLE


def find_reviewable_s3_run_for_s2(
    session_root: Path | str,
    s2_run_dir: Path | str,
) -> Path | None:
    """Return the newest completed S3 run produced from ``s2_run_dir``, if any."""

    target = Path(s2_run_dir).expanduser().resolve()
    summary = summarize_tracking_correction_project(session_root)
    for run in summary.reviewable_runs:
        meta = _load_json_mapping(run.run_dir / RUN_META_FILENAME, [])
        input_block = meta.get("input")
        if not isinstance(input_block, Mapping):
            continue
        recorded = _optional_text(input_block.get("s2_run_dir"))
        if recorded is None:
            continue
        try:
            recorded_path = Path(recorded).expanduser().resolve()
        except OSError:
            continue
        if recorded_path == target:
            return run.run_dir
    return None


def _discover_runs(run_root: Path) -> list[TrackingCorrectionRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir()
    ]


def _summarize_run(run_dir: Path) -> TrackingCorrectionRunSummary:
    run_meta_path = run_dir / RUN_META_FILENAME
    working_path = run_dir / WORKING_TRACKED_POSE_FILENAME
    corrections_path = run_dir / MACHINE_CORRECTIONS_FILENAME
    artifact_presence = {
        "run_meta": run_meta_path.is_file(),
        "working_tracked_pose": working_path.is_file(),
        "machine_corrections": corrections_path.is_file(),
    }
    missing = tuple(
        name for name, present in artifact_presence.items() if not present
    )

    metadata_errors: list[str] = []
    meta = _load_json_mapping(run_meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))
    backend_status = _optional_text(meta.get("backend_status"))
    dry_run = meta.get("dry_run") if "dry_run" in meta else None
    if dry_run is not None and not isinstance(dry_run, bool):
        dry_run = bool(dry_run)

    classification = _classify_run(
        missing=missing,
        status=status,
        backend_status=backend_status,
        dry_run=dry_run if isinstance(dry_run, bool) else None,
        metadata_errors=metadata_errors,
    )
    return TrackingCorrectionRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        backend_status=backend_status,
        dry_run=dry_run if isinstance(dry_run, bool) else None,
        classification=classification,
        artifact_presence=artifact_presence,
        missing_required_artifacts=missing,
        profile_id=_optional_text(meta.get("profile_id")),
        created_at=_optional_text(meta.get("created_at")),
        completed_at=_optional_text(meta.get("completed_at")),
        metadata_errors=tuple(metadata_errors),
    )


def _classify_run(
    *,
    missing: tuple[str, ...],
    status: str | None,
    backend_status: str | None,
    dry_run: bool | None,
    metadata_errors: list[str],
) -> str:
    if dry_run is True:
        return FAILED_OR_INCOMPLETE
    if missing:
        return MISSING_REQUIRED_ARTIFACTS
    if metadata_errors:
        return FAILED_OR_INCOMPLETE
    if status != STATUS_CORRECTION_COMPLETE or backend_status != "completed":
        return FAILED_OR_INCOMPLETE
    return COMPLETE_REVIEWABLE


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


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _run_sort_key(summary: TrackingCorrectionRunSummary) -> tuple[datetime, str]:
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
