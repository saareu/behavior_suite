"""Discover completed Subsystem 04 pose-finalization runs for a session.

Completed S4 runs remain on disk when their source S3 acceptance later becomes
superseded; discovery distinguishes eligible vs stale without mutating them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pose_finalization.accepted_s3 import list_accepted_s3_runs, validate_accepted_s3_run
from pose_finalization.contracts import (
    FINAL_POSE_FILENAME,
    POSE_FINALIZATION_DIRNAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    PoseFinalizationError,
)
from pose_finalization.runner import is_s4_run_eligible_for_downstream

ELIGIBLE_DOWNSTREAM = "eligible"
STALE_SOURCE = "stale_source"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"


@dataclass(frozen=True, slots=True)
class PoseFinalizationRunSummary:
    """Lightweight summary of one Subsystem 04 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    source_s3_run_id: str | None
    source_s3_run_dir: Path | None
    final_pose_path: Path | None
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    metadata_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PoseFinalizationProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    pose_finalization_present: bool
    runs: tuple[PoseFinalizationRunSummary, ...]
    accepted_s3_count: int
    no_accepted_s3_reason: str | None

    @property
    def eligible_runs(self) -> tuple[PoseFinalizationRunSummary, ...]:
        return tuple(run for run in self.runs if run.classification == ELIGIBLE_DOWNSTREAM)

    @property
    def completed_runs(self) -> tuple[PoseFinalizationRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_pose_finalization_project(
    session_root: Path | str,
) -> PoseFinalizationProjectSummary:
    """Summarize existing S4 runs and accepted-S3 availability for a session."""

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise PoseFinalizationError(f"Session root does not exist: {root}")

    run_root = root / POSE_FINALIZATION_DIRNAME
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    accepted = list_accepted_s3_runs(root)
    reason = None if accepted else explain_missing_accepted_s3(root)
    return PoseFinalizationProjectSummary(
        session_root=root,
        pose_finalization_present=run_root.is_dir(),
        runs=runs,
        accepted_s3_count=len(accepted),
        no_accepted_s3_reason=reason,
    )


def explain_missing_accepted_s3(session_root: Path | str) -> str:
    """Return a human-readable explanation when no eligible accepted S3 exists."""

    root = Path(session_root).expanduser().resolve()
    tracking_root = root / "tracking_correction"
    if not tracking_root.is_dir():
        return (
            "No eligible accepted S3 result exists: this session has no "
            "tracking_correction/ directory. Accept a Subsystem 3 run before "
            "running pose finalization."
        )

    candidates = [path for path in tracking_root.iterdir() if path.is_dir()]
    if not candidates:
        return (
            "No eligible accepted S3 result exists: tracking_correction/ is empty. "
            "Complete and accept a Subsystem 3 run first."
        )

    reasons: list[str] = []
    for run_dir in sorted(candidates, key=lambda p: p.name, reverse=True):
        try:
            validate_accepted_s3_run(run_dir)
        except PoseFinalizationError as exc:
            reasons.append(f"{run_dir.name}: {exc}")
        else:
            # Should not happen when called after an empty accepted list, but
            # keep the message honest if a race occurs.
            return (
                "An accepted S3 run is now available. Refresh accepted-S3 discovery."
            )

    preview = "; ".join(reasons[:3])
    extra = "" if len(reasons) <= 3 else f" (+{len(reasons) - 3} more)"
    return (
        "No eligible accepted S3 result exists for pose finalization. "
        "S4 requires acceptance_state='accepted' with a matching tracked_pose "
        f"SHA256 and resolvable S1 prepared_sync.npz. Details: {preview}{extra}"
    )


def _discover_runs(run_root: Path) -> list[PoseFinalizationRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir()
    ]


def _summarize_run(run_dir: Path) -> PoseFinalizationRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    final_pose = run_dir / FINAL_POSE_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))
    source = meta.get("source_s3") if isinstance(meta.get("source_s3"), Mapping) else {}
    source_run_id = _optional_text(source.get("run_id")) if isinstance(source, Mapping) else None
    source_run_dir_text = (
        _optional_text(source.get("run_dir")) if isinstance(source, Mapping) else None
    )
    source_run_dir = Path(source_run_dir_text) if source_run_dir_text else None
    qc = meta.get("qc") if isinstance(meta.get("qc"), Mapping) else {}
    warning_count = 0
    if isinstance(qc, Mapping):
        raw = qc.get("warning_count")
        if isinstance(raw, int):
            warning_count = raw
        elif isinstance(qc.get("warnings"), list):
            warning_count = len(qc["warnings"])

    classification = _classify_run(
        status=status,
        final_pose_present=final_pose.is_file(),
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
    )
    return PoseFinalizationRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s3_run_id=source_run_id,
        source_s3_run_dir=source_run_dir.resolve() if source_run_dir is not None else None,
        final_pose_path=final_pose.resolve() if final_pose.is_file() else None,
        created_at=_optional_text(meta.get("created_at")),
        completed_at=_optional_text(meta.get("completed_at")),
        warning_count=warning_count,
        error_message=_optional_text(meta.get("error")),
        metadata_errors=tuple(metadata_errors),
    )


def _classify_run(
    *,
    status: str | None,
    final_pose_present: bool,
    metadata_errors: tuple[str, ...],
    run_dir: Path,
) -> str:
    if metadata_errors or status is None:
        return MISSING_REQUIRED_ARTIFACTS
    if status == STATUS_FAILED:
        return FAILED_OR_INCOMPLETE
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        return FAILED_OR_INCOMPLETE
    if not final_pose_present:
        return MISSING_REQUIRED_ARTIFACTS
    if is_s4_run_eligible_for_downstream(run_dir):
        return ELIGIBLE_DOWNSTREAM
    return STALE_SOURCE


def _run_sort_key(run: PoseFinalizationRunSummary) -> tuple[datetime, str]:
    for value in (run.completed_at, run.created_at):
        parsed = _parse_datetime(value)
        if parsed is not None:
            return (parsed, run.run_id)
    parsed = _parse_run_id_timestamp(run.run_id)
    if parsed is not None:
        return (parsed, run.run_id)
    try:
        return (datetime.fromtimestamp(run.run_dir.stat().st_mtime), run.run_id)
    except OSError:
        return (datetime.min, run.run_id)


def _load_json_mapping(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"unreadable {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{path.name} is not a JSON object")
        return {}
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


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
    if re.fullmatch(r"\d{8}T\d{6}", timestamp) is None:
        return None
    return datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
