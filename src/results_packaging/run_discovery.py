"""Discover Subsystem 09 results-package runs for a session.

Completed S9 runs remain on disk when referenced upstream files later become
stale; discovery distinguishes historical completion status from current
downstream eligibility without rewriting ``run_meta.status``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    ETHOGRAM_SUMMARY_FILENAME,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RESULTS_PACKAGE_DIRNAME,
    RUN_META_FILENAME,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    SYNC_INDEX_FILENAME,
    ProvenanceError,
    PublicationError,
)


@dataclass(frozen=True, slots=True)
class ResultsPackageRunSummary:
    """Lightweight summary of one Subsystem 09 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    source_s8_run_id: str | None
    source_s8_run_dir: Path | None
    package_manifest_path: Path | None
    sync_index_path: Path | None
    behavior_bouts_path: Path | None
    ethogram_summary_path: Path | None
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    metadata_errors: tuple[str, ...] = ()
    staleness_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResultsPackageProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    results_package_present: bool
    runs: tuple[ResultsPackageRunSummary, ...]
    eligible_s8_available: bool
    no_s8_reason: str | None

    @property
    def eligible_runs(self) -> tuple[ResultsPackageRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
        )

    @property
    def completed_runs(self) -> tuple[ResultsPackageRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_results_package_project(
    session_root: Path | str,
    *,
    eligible_s8_available: bool | None = None,
    no_s8_reason: str | None = None,
) -> ResultsPackageProjectSummary:
    """Summarize existing S9 runs and current eligible-S8 availability."""

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise PublicationError(f"Session root does not exist: {root}")

    run_root = root / RESULTS_PACKAGE_DIRNAME
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    if eligible_s8_available is None:
        s8_ok, reason = _assess_eligible_s8(root)
    else:
        s8_ok = bool(eligible_s8_available)
        reason = None if s8_ok else (no_s8_reason or "no eligible S8 run")
    return ResultsPackageProjectSummary(
        session_root=root,
        results_package_present=run_root.is_dir(),
        runs=runs,
        eligible_s8_available=s8_ok,
        no_s8_reason=None if s8_ok else reason,
    )


def find_latest_eligible_s9_run(
    session_root: Path | str,
) -> ResultsPackageRunSummary | None:
    """Return the newest currently eligible completed S9 run, if any."""

    summary = summarize_results_package_project(session_root)
    eligible = summary.eligible_runs
    return eligible[0] if eligible else None


def list_completed_s9_runs(
    session_root: Path | str,
) -> tuple[ResultsPackageRunSummary, ...]:
    """Return historically completed S9 runs (may include currently stale)."""

    return summarize_results_package_project(session_root).completed_runs


def _assess_eligible_s8(session_root: Path) -> tuple[bool, str | None]:
    from results_packaging.contracts import IntakeError
    from results_packaging.eligibility import select_s8_run_for_packaging

    try:
        select_s8_run_for_packaging(session_root)
    except IntakeError as exc:
        return False, str(exc)
    return True, None


def _discover_runs(run_root: Path) -> list[ResultsPackageRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir() and not run_dir.name.startswith("_")
    ]


def _summarize_run(run_dir: Path) -> ResultsPackageRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    sync = run_dir / SYNC_INDEX_FILENAME
    bouts = run_dir / BEHAVIOR_BOUTS_FILENAME
    ethogram = run_dir / ETHOGRAM_SUMMARY_FILENAME
    manifest = run_dir / PACKAGE_MANIFEST_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))

    source_s8 = (
        meta.get("source_s8") if isinstance(meta.get("source_s8"), Mapping) else {}
    )
    s8_run_id = (
        _optional_text(source_s8.get("run_id")) if isinstance(source_s8, Mapping) else None
    )
    s8_run_dir = (
        _optional_text(source_s8.get("run_dir")) if isinstance(source_s8, Mapping) else None
    )

    qc = meta.get("qc") if isinstance(meta.get("qc"), Mapping) else {}
    warning_count = 0
    if isinstance(qc, Mapping):
        raw = qc.get("warning_count")
        if isinstance(raw, int):
            warning_count = raw
        elif isinstance(qc.get("warnings"), list):
            warning_count = len(qc["warnings"])

    if status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        for name in (
            QC_FILENAME,
            SYNC_INDEX_FILENAME,
            BEHAVIOR_BOUTS_FILENAME,
            ETHOGRAM_SUMMARY_FILENAME,
            PACKAGE_MANIFEST_FILENAME,
        ):
            if not (run_dir / name).is_file():
                metadata_errors.append(f"missing {name}")

    classification, staleness_reasons = _classify_run(
        status=status,
        core_present=all(
            p.is_file() for p in (sync, bouts, ethogram, manifest, run_dir / QC_FILENAME)
        ),
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
        warning_count=warning_count,
    )
    return ResultsPackageRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s8_run_id=s8_run_id,
        source_s8_run_dir=Path(s8_run_dir).resolve() if s8_run_dir else None,
        package_manifest_path=manifest.resolve() if manifest.is_file() else None,
        sync_index_path=sync.resolve() if sync.is_file() else None,
        behavior_bouts_path=bouts.resolve() if bouts.is_file() else None,
        ethogram_summary_path=ethogram.resolve() if ethogram.is_file() else None,
        created_at=_optional_text(meta.get("created_at")),
        completed_at=_optional_text(meta.get("completed_at")),
        warning_count=warning_count,
        error_message=_optional_text(meta.get("error")),
        metadata_errors=tuple(metadata_errors),
        staleness_reasons=staleness_reasons,
    )


def _classify_run(
    *,
    status: str | None,
    core_present: bool,
    metadata_errors: tuple[str, ...],
    run_dir: Path,
    warning_count: int,
) -> tuple[str, tuple[str, ...]]:
    if metadata_errors or status is None:
        return MISSING_REQUIRED_ARTIFACTS, ()
    if status == STATUS_FAILED:
        return FAILED_OR_INCOMPLETE, ()
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        return FAILED_OR_INCOMPLETE, ()
    if not core_present:
        return MISSING_REQUIRED_ARTIFACTS, ()

    # Lazy import avoids runner ↔ discovery circular imports at package load.
    from results_packaging.runner import (
        is_s9_run_eligible_for_downstream,
        validate_s9_run_for_downstream,
    )

    try:
        validate_s9_run_for_downstream(run_dir)
    except ProvenanceError as exc:
        return STALE_SOURCE, (str(exc),)
    except PublicationError as exc:
        message = str(exc)
        lowered = message.lower()
        if any(
            token in lowered
            for token in ("sha256", "eligible", "digest", "stale", "fingerprint")
        ):
            return STALE_SOURCE, (message,)
        return MISSING_REQUIRED_ARTIFACTS, (message,)
    except Exception as exc:  # noqa: BLE001 — discovery must not crash
        return MISSING_REQUIRED_ARTIFACTS, (str(exc),)

    if is_s9_run_eligible_for_downstream(run_dir):
        if status == STATUS_COMPLETED_WITH_WARNINGS or warning_count > 0:
            return COMPLETED_WITH_WARNINGS, ()
        return ELIGIBLE_DOWNSTREAM, ()
    return STALE_SOURCE, ()


def _run_sort_key(run: ResultsPackageRunSummary) -> tuple[datetime, str]:
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
