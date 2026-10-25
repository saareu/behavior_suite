"""Discover Subsystem 06 combined-feature runs for a session.

Completed S6 runs remain on disk when their recorded S4/S5 sources later change;
discovery distinguishes eligible vs stale without rewriting historical
completion status.

Downstream selection policy
---------------------------
Prefer the newest currently eligible completed run
(``status in {completed, completed_with_warnings}`` and
``classification in {eligible, completed_with_warnings}``), ordered by
``completed_at`` / ``created_at`` / run-id timestamp (newest first). Do **not**
auto-select a stale run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from combined_features.contracts import (
    COMBINED_FEATURES_DIRNAME,
    COMPLETED_WITH_WARNINGS,
    DYADIC_FEATURES_FILENAME,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    MODEL_COMPAT_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    CombinedFeaturesError,
)
from combined_features.runner import is_s6_run_eligible_for_downstream


@dataclass(frozen=True, slots=True)
class CombinedFeaturesRunSummary:
    """Lightweight summary of one Subsystem 06 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    source_s4_run_id: str | None
    source_s5_run_id: str | None
    source_s4_run_dir: Path | None
    source_s5_run_dir: Path | None
    dyadic_features_path: Path | None
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    metadata_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CombinedFeaturesProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    combined_features_present: bool
    runs: tuple[CombinedFeaturesRunSummary, ...]
    joint_pair_available: bool
    no_pair_reason: str | None

    @property
    def eligible_runs(self) -> tuple[CombinedFeaturesRunSummary, ...]:
        """Currently eligible for downstream (includes warning completions)."""

        return tuple(
            run
            for run in self.runs
            if run.classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
        )

    @property
    def completed_runs(self) -> tuple[CombinedFeaturesRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_combined_features_project(
    session_root: Path | str,
) -> CombinedFeaturesProjectSummary:
    """Summarize existing S6 runs and current joint-pair availability."""

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise CombinedFeaturesError(f"Session root does not exist: {root}")

    run_root = root / COMBINED_FEATURES_DIRNAME
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    pair_ok, reason = _assess_joint_pair(root)
    return CombinedFeaturesProjectSummary(
        session_root=root,
        combined_features_present=run_root.is_dir(),
        runs=runs,
        joint_pair_available=pair_ok,
        no_pair_reason=None if pair_ok else reason,
    )


def find_latest_eligible_s6_run(
    session_root: Path | str,
) -> CombinedFeaturesRunSummary | None:
    """Return the newest currently eligible completed S6 run, if any.

    Callers must not treat completed-but-stale runs as eligible.
    """

    summary = summarize_combined_features_project(session_root)
    eligible = summary.eligible_runs
    return eligible[0] if eligible else None


def _assess_joint_pair(session_root: Path) -> tuple[bool, str | None]:
    from combined_features.upstream import select_default_joint_pair

    try:
        select_default_joint_pair(session_root)
    except CombinedFeaturesError as exc:
        return False, str(exc)
    return True, None


def _discover_runs(run_root: Path) -> list[CombinedFeaturesRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir() and not run_dir.name.startswith("_")
    ]


def _summarize_run(run_dir: Path) -> CombinedFeaturesRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    parquet = run_dir / DYADIC_FEATURES_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))
    source_s4 = (
        meta.get("source_s4") if isinstance(meta.get("source_s4"), Mapping) else {}
    )
    source_s5 = (
        meta.get("source_s5") if isinstance(meta.get("source_s5"), Mapping) else {}
    )
    s4_run_id = (
        _optional_text(source_s4.get("run_id")) if isinstance(source_s4, Mapping) else None
    )
    s5_run_id = (
        _optional_text(source_s5.get("run_id")) if isinstance(source_s5, Mapping) else None
    )
    s4_run_dir = (
        _optional_text(source_s4.get("run_dir")) if isinstance(source_s4, Mapping) else None
    )
    s5_run_dir = (
        _optional_text(source_s5.get("run_dir")) if isinstance(source_s5, Mapping) else None
    )
    qc = meta.get("qc") if isinstance(meta.get("qc"), Mapping) else {}
    warning_count = 0
    if isinstance(qc, Mapping):
        raw = qc.get("warning_count")
        if isinstance(raw, int):
            warning_count = raw
        elif isinstance(qc.get("warnings"), list):
            warning_count = len(qc["warnings"])

    # Soft-check companion artifacts for completed runs.
    if status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        for name in (QC_FILENAME, MODEL_COMPAT_FILENAME):
            if not (run_dir / name).is_file():
                metadata_errors.append(f"missing {name}")

    classification = _classify_run(
        status=status,
        parquet_present=parquet.is_file(),
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
        warning_count=warning_count,
    )
    return CombinedFeaturesRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s4_run_id=s4_run_id,
        source_s5_run_id=s5_run_id,
        source_s4_run_dir=Path(s4_run_dir).resolve() if s4_run_dir else None,
        source_s5_run_dir=Path(s5_run_dir).resolve() if s5_run_dir else None,
        dyadic_features_path=parquet.resolve() if parquet.is_file() else None,
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
        # Missing companion artifacts on an otherwise complete run still block
        # eligibility, but distinguish from total absence of meta.
        return MISSING_REQUIRED_ARTIFACTS
    if status == STATUS_FAILED:
        return FAILED_OR_INCOMPLETE
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        return FAILED_OR_INCOMPLETE
    if not parquet_present:
        return MISSING_REQUIRED_ARTIFACTS
    if is_s6_run_eligible_for_downstream(run_dir):
        if status == STATUS_COMPLETED_WITH_WARNINGS or warning_count > 0:
            return COMPLETED_WITH_WARNINGS
        return ELIGIBLE_DOWNSTREAM
    return STALE_SOURCE


def _run_sort_key(run: CombinedFeaturesRunSummary) -> tuple[datetime, str]:
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
