"""Discover Subsystem 08 behavioral-arbitration runs for a session.

Completed S8 runs remain on disk when their recorded S7/S6 sources later
change; discovery distinguishes eligible vs stale without rewriting historical
completion status.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from behavioral_arbitration.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    BEHAVIORAL_ARBITRATION_DIRNAME,
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    QC_FILENAME,
    RUN_META_FILENAME,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    PublicationError,
)
from behavioral_arbitration.runner import (
    is_s8_run_eligible_for_downstream,
    validate_s8_run_for_downstream,
)
from behavioral_inference.contracts import ProvenanceError
from behavioral_inference.run_discovery import find_latest_eligible_s7_run
from behavioral_inference.runner import is_s7_run_eligible_for_downstream


@dataclass(frozen=True, slots=True)
class BehavioralArbitrationRunSummary:
    """Lightweight summary of one Subsystem 08 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    source_s7_run_id: str | None
    source_s7_run_dir: Path | None
    source_s6_run_id: str | None
    source_s6_run_dir: Path | None
    behavior_states_path: Path | None
    behavior_bouts_path: Path | None
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    metadata_errors: tuple[str, ...] = ()
    staleness_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BehavioralArbitrationProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    behavioral_arbitration_present: bool
    runs: tuple[BehavioralArbitrationRunSummary, ...]
    eligible_s7_available: bool
    no_s7_reason: str | None

    @property
    def eligible_runs(self) -> tuple[BehavioralArbitrationRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
        )

    @property
    def completed_runs(self) -> tuple[BehavioralArbitrationRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_behavioral_arbitration_project(
    session_root: Path | str,
    *,
    eligible_s7_available: bool | None = None,
    no_s7_reason: str | None = None,
) -> BehavioralArbitrationProjectSummary:
    """Summarize existing S8 runs and current eligible-S7 availability.

    Callers that already assessed S7 options (e.g. GUI discovery) may pass
    ``eligible_s7_available`` to skip a redundant nested S7 eligibility probe.
    """

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise PublicationError(f"Session root does not exist: {root}")

    run_root = root / BEHAVIORAL_ARBITRATION_DIRNAME
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    if eligible_s7_available is None:
        s7_ok, reason = _assess_eligible_s7(root)
    else:
        s7_ok = bool(eligible_s7_available)
        reason = None if s7_ok else (no_s7_reason or "no eligible S7 run")
    return BehavioralArbitrationProjectSummary(
        session_root=root,
        behavioral_arbitration_present=run_root.is_dir(),
        runs=runs,
        eligible_s7_available=s7_ok,
        no_s7_reason=None if s7_ok else reason,
    )


def find_latest_eligible_s8_run(
    session_root: Path | str,
) -> BehavioralArbitrationRunSummary | None:
    """Return the newest currently eligible completed S8 run, if any."""

    summary = summarize_behavioral_arbitration_project(session_root)
    eligible = summary.eligible_runs
    return eligible[0] if eligible else None


def _assess_eligible_s7(session_root: Path) -> tuple[bool, str | None]:
    latest = find_latest_eligible_s7_run(session_root)
    if latest is None:
        return False, "no eligible S7 run"
    if not is_s7_run_eligible_for_downstream(latest.run_dir):
        return False, "latest S7 run is not eligible for downstream use"
    return True, None


def _discover_runs(run_root: Path) -> list[BehavioralArbitrationRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir() and not run_dir.name.startswith("_")
    ]


def _summarize_run(run_dir: Path) -> BehavioralArbitrationRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    states = run_dir / BEHAVIOR_STATES_FILENAME
    bouts = run_dir / BEHAVIOR_BOUTS_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))

    source_s7 = (
        meta.get("source_s7") if isinstance(meta.get("source_s7"), Mapping) else {}
    )
    source_s6 = (
        meta.get("source_s6") if isinstance(meta.get("source_s6"), Mapping) else {}
    )
    s7_run_id = (
        _optional_text(source_s7.get("run_id")) if isinstance(source_s7, Mapping) else None
    )
    s7_run_dir = (
        _optional_text(source_s7.get("run_dir")) if isinstance(source_s7, Mapping) else None
    )
    s6_run_id = (
        _optional_text(source_s6.get("run_id")) if isinstance(source_s6, Mapping) else None
    )
    s6_run_dir = (
        _optional_text(source_s6.get("run_dir")) if isinstance(source_s6, Mapping) else None
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
        for name in (QC_FILENAME, BEHAVIOR_STATES_FILENAME, BEHAVIOR_BOUTS_FILENAME):
            if not (run_dir / name).is_file():
                metadata_errors.append(f"missing {name}")

    classification, staleness_reasons = _classify_run(
        status=status,
        states_present=states.is_file(),
        bouts_present=bouts.is_file(),
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
        warning_count=warning_count,
    )
    return BehavioralArbitrationRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s7_run_id=s7_run_id,
        source_s7_run_dir=Path(s7_run_dir).resolve() if s7_run_dir else None,
        source_s6_run_id=s6_run_id,
        source_s6_run_dir=Path(s6_run_dir).resolve() if s6_run_dir else None,
        behavior_states_path=states.resolve() if states.is_file() else None,
        behavior_bouts_path=bouts.resolve() if bouts.is_file() else None,
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
    states_present: bool,
    bouts_present: bool,
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
    if not states_present or not bouts_present:
        return MISSING_REQUIRED_ARTIFACTS, ()

    try:
        validate_s8_run_for_downstream(run_dir)
    except ProvenanceError as exc:
        return STALE_SOURCE, (str(exc),)
    except PublicationError as exc:
        message = str(exc)
        if (
            "SHA256" in message
            or "eligible" in message.lower()
            or "digest" in message.lower()
            or "stale" in message.lower()
        ):
            return STALE_SOURCE, (message,)
        return MISSING_REQUIRED_ARTIFACTS, (message,)
    except Exception as exc:  # noqa: BLE001 — discovery must not crash
        return MISSING_REQUIRED_ARTIFACTS, (str(exc),)

    if is_s8_run_eligible_for_downstream(run_dir):
        if status == STATUS_COMPLETED_WITH_WARNINGS or warning_count > 0:
            return COMPLETED_WITH_WARNINGS, ()
        return ELIGIBLE_DOWNSTREAM, ()
    return STALE_SOURCE, ()


def _run_sort_key(run: BehavioralArbitrationRunSummary) -> tuple[datetime, str]:
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
