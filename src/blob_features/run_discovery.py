"""Discover Subsystem 05 blob-feature runs for a session.

Completed S5 runs remain on disk when their recorded S1 source later changes
or is re-prepared; discovery distinguishes eligible vs stale without rewriting
historical completion status.

Downstream (S6) selection policy
--------------------------------
Prefer the newest currently eligible completed run
(``status in {completed, completed_with_warnings}`` and
``classification == eligible``), ordered by ``completed_at`` / ``created_at`` /
run-id timestamp (newest first). Do **not** auto-select a run for S6 unless it
is currently eligible. Failed, incomplete, structurally invalid, and
stale-source runs are never selected for downstream handoff.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from blob_features.contracts import (
    BLOB_FEATURES_DIRNAME,
    BLOB_FEATURES_FILENAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    BlobFeaturesError,
)
from blob_features.runner import is_s5_run_eligible_for_downstream

ELIGIBLE_DOWNSTREAM = "eligible"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STALE_SOURCE = "stale_source"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"


@dataclass(frozen=True, slots=True)
class BlobFeaturesRunSummary:
    """Lightweight summary of one Subsystem 05 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    source_s1_session_root: Path | None
    source_prepared_sync: Path | None
    blob_features_path: Path | None
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    metadata_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BlobFeaturesProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    blob_features_present: bool
    runs: tuple[BlobFeaturesRunSummary, ...]
    s1_eligible: bool
    no_s1_reason: str | None

    @property
    def eligible_runs(self) -> tuple[BlobFeaturesRunSummary, ...]:
        """Currently eligible for downstream (includes warning completions)."""

        return tuple(
            run
            for run in self.runs
            if run.classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
        )

    @property
    def completed_runs(self) -> tuple[BlobFeaturesRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_blob_features_project(
    session_root: Path | str,
) -> BlobFeaturesProjectSummary:
    """Summarize existing S5 runs and current S1 eligibility for a session."""

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise BlobFeaturesError(f"Session root does not exist: {root}")

    run_root = root / BLOB_FEATURES_DIRNAME
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    s1_eligible, reason = _assess_s1(root)
    return BlobFeaturesProjectSummary(
        session_root=root,
        blob_features_present=run_root.is_dir(),
        runs=runs,
        s1_eligible=s1_eligible,
        no_s1_reason=None if s1_eligible else reason,
    )


def find_latest_eligible_s5_run(
    session_root: Path | str,
) -> BlobFeaturesRunSummary | None:
    """Return the newest currently eligible completed S5 run, if any.

    This is the established selection policy for a future S6 handoff. Callers
    must not treat completed-but-stale runs as eligible.
    """

    summary = summarize_blob_features_project(session_root)
    eligible = summary.eligible_runs
    return eligible[0] if eligible else None


def _assess_s1(session_root: Path) -> tuple[bool, str | None]:
    from blob_features.s1_eligibility import validate_s1_for_blob_features

    try:
        validate_s1_for_blob_features(session_root)
    except BlobFeaturesError as exc:
        return False, str(exc)
    return True, None


def _discover_runs(run_root: Path) -> list[BlobFeaturesRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir() and not run_dir.name.startswith("_")
    ]


def _summarize_run(run_dir: Path) -> BlobFeaturesRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    parquet = run_dir / BLOB_FEATURES_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))
    source = meta.get("source_s1") if isinstance(meta.get("source_s1"), Mapping) else {}
    source_session = (
        _optional_text(source.get("session_root")) if isinstance(source, Mapping) else None
    )
    source_sync = (
        _optional_text(source.get("prepared_sync")) if isinstance(source, Mapping) else None
    )
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
        parquet_present=parquet.is_file(),
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
        warning_count=warning_count,
    )
    return BlobFeaturesRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s1_session_root=Path(source_session).resolve() if source_session else None,
        source_prepared_sync=Path(source_sync).resolve() if source_sync else None,
        blob_features_path=parquet.resolve() if parquet.is_file() else None,
        created_at=_optional_text(meta.get("created_at")),
        completed_at=_optional_text(meta.get("completed_at")),
        warning_count=warning_count,
        error_message=_optional_text(meta.get("error")),
        metadata_errors=tuple(metadata_errors),
    )


def _classify_run(
    *,
    status: str | None,
    parquet_present: bool,
    metadata_errors: tuple[str, ...],
    run_dir: Path,
    warning_count: int,
) -> str:
    if metadata_errors or status is None:
        return MISSING_REQUIRED_ARTIFACTS
    if status == STATUS_FAILED:
        return FAILED_OR_INCOMPLETE
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        return FAILED_OR_INCOMPLETE
    if not parquet_present:
        return MISSING_REQUIRED_ARTIFACTS
    if is_s5_run_eligible_for_downstream(run_dir):
        # Distinguish warning completions without rewriting historical status.
        if status == STATUS_COMPLETED_WITH_WARNINGS or warning_count > 0:
            return COMPLETED_WITH_WARNINGS
        return ELIGIBLE_DOWNSTREAM
    return STALE_SOURCE


def _run_sort_key(run: BlobFeaturesRunSummary) -> tuple[datetime, str]:
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
