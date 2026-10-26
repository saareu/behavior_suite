"""Discover Subsystem 07 behavioral-inference runs for a session.

Completed S7 runs remain on disk when their recorded S6 sources or model
artifacts later change; discovery distinguishes eligible vs stale without
rewriting historical completion status.

Downstream selection policy
---------------------------
Prefer the newest currently eligible completed run
(``status in {completed, completed_with_warnings}`` and
``classification in {eligible, completed_with_warnings}``), ordered by
``completed_at`` / ``created_at`` / run-id timestamp (newest first). Do **not**
auto-select a stale run.

Detector subsets are allowed: summaries expose ``detectors_selected`` so S8
can decide whether required detectors are present.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from behavioral_inference.contracts import (
    BEHAVIORAL_INFERENCE_DIRNAME,
    COMPLETED_WITH_WARNINGS,
    DYADIC_WIDE_FILENAME,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    FOCAL_WIDE_FILENAME,
    INVALID_ARTIFACT,
    MISSING_REQUIRED_ARTIFACTS,
    MODEL_MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STALE_MODELS,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    ProvenanceError,
    PublicationError,
)
from behavioral_inference.runner import (
    assess_model_source_staleness,
    is_s7_run_eligible_for_downstream,
    validate_s7_run_for_downstream,
)
from combined_features.contracts import CombinedFeaturesError
from combined_features.run_discovery import (
    CombinedFeaturesRunSummary,
    find_latest_eligible_s6_run,
    summarize_combined_features_project,
)
from combined_features.runner import is_s6_run_eligible_for_downstream


@dataclass(frozen=True, slots=True)
class BehavioralInferenceRunSummary:
    """Lightweight summary of one Subsystem 07 run directory."""

    run_dir: Path
    run_id: str
    status: str | None
    classification: str
    source_s6_run_id: str | None
    source_s6_run_dir: Path | None
    predictions_path: Path | None
    dyadic_wide_path: Path | None
    focal_wide_path: Path | None
    detectors_dyadic: tuple[str, ...]
    detectors_focal: tuple[str, ...]
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    metadata_errors: tuple[str, ...] = ()
    staleness_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BehavioralInferenceProjectSummary:
    """Discovery result for one project/session root."""

    session_root: Path
    behavioral_inference_present: bool
    runs: tuple[BehavioralInferenceRunSummary, ...]
    eligible_s6_available: bool
    no_s6_reason: str | None

    @property
    def eligible_runs(self) -> tuple[BehavioralInferenceRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
        )

    @property
    def completed_runs(self) -> tuple[BehavioralInferenceRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_behavioral_inference_project(
    session_root: Path | str,
) -> BehavioralInferenceProjectSummary:
    """Summarize existing S7 runs and current eligible-S6 availability."""

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise PublicationError(f"Session root does not exist: {root}")

    run_root = root / BEHAVIORAL_INFERENCE_DIRNAME
    runs = tuple(sorted(_discover_runs(run_root), key=_run_sort_key, reverse=True))
    s6_ok, reason = _assess_eligible_s6(root)
    return BehavioralInferenceProjectSummary(
        session_root=root,
        behavioral_inference_present=run_root.is_dir(),
        runs=runs,
        eligible_s6_available=s6_ok,
        no_s6_reason=None if s6_ok else reason,
    )


def find_latest_eligible_s7_run(
    session_root: Path | str,
) -> BehavioralInferenceRunSummary | None:
    """Return the newest currently eligible completed S7 run, if any."""

    summary = summarize_behavioral_inference_project(session_root)
    eligible = summary.eligible_runs
    return eligible[0] if eligible else None


def find_latest_eligible_s6_for_s7(
    session_root: Path | str,
) -> CombinedFeaturesRunSummary | None:
    """Wrapper: newest eligible S6 run suitable as S7 input."""

    return find_latest_eligible_s6_run(session_root)


def validate_s7_run_summary(run_dir: Path | str) -> dict[str, Any]:
    """Public validate helper returning eligibility details or raising."""

    return validate_s7_run_for_downstream(run_dir)


def _assess_eligible_s6(session_root: Path) -> tuple[bool, str | None]:
    try:
        project = summarize_combined_features_project(session_root)
    except CombinedFeaturesError as exc:
        return False, str(exc)
    if not project.eligible_runs:
        return False, project.no_pair_reason or "no eligible S6 run"
    return True, None


def _discover_runs(run_root: Path) -> list[BehavioralInferenceRunSummary]:
    if not run_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in run_root.iterdir()
        if run_dir.is_dir() and not run_dir.name.startswith("_")
    ]


def _summarize_run(run_dir: Path) -> BehavioralInferenceRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    predictions = run_dir / PREDICTIONS_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    status = _optional_text(meta.get("status"))
    source_s6 = (
        meta.get("source_s6") if isinstance(meta.get("source_s6"), Mapping) else {}
    )
    s6_run_id = (
        _optional_text(source_s6.get("run_id")) if isinstance(source_s6, Mapping) else None
    )
    s6_run_dir = (
        _optional_text(source_s6.get("run_dir")) if isinstance(source_s6, Mapping) else None
    )
    detectors = (
        meta.get("detectors_selected")
        if isinstance(meta.get("detectors_selected"), Mapping)
        else {}
    )
    dyadic = tuple(
        str(x)
        for x in (
            detectors.get("dyadic", [])
            if isinstance(detectors, Mapping)
            else []
        )
        if str(x).strip()
    )
    focal = tuple(
        str(x)
        for x in (
            detectors.get("focal", [])
            if isinstance(detectors, Mapping)
            else []
        )
        if str(x).strip()
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
        for name in (QC_FILENAME, MODEL_MANIFEST_FILENAME):
            if not (run_dir / name).is_file():
                metadata_errors.append(f"missing {name}")

    classification, staleness_reasons = _classify_run(
        status=status,
        predictions_present=predictions.is_file(),
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
        warning_count=warning_count,
        source_s6_run_dir=Path(s6_run_dir).resolve() if s6_run_dir else None,
    )
    dyadic_wide = run_dir / DYADIC_WIDE_FILENAME
    focal_wide = run_dir / FOCAL_WIDE_FILENAME
    return BehavioralInferenceRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s6_run_id=s6_run_id,
        source_s6_run_dir=Path(s6_run_dir).resolve() if s6_run_dir else None,
        predictions_path=predictions.resolve() if predictions.is_file() else None,
        dyadic_wide_path=dyadic_wide.resolve() if dyadic_wide.is_file() else None,
        focal_wide_path=focal_wide.resolve() if focal_wide.is_file() else None,
        detectors_dyadic=dyadic,
        detectors_focal=focal,
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
    predictions_present: bool,
    metadata_errors: tuple[str, ...],
    run_dir: Path,
    warning_count: int,
    source_s6_run_dir: Path | None,
) -> tuple[str, tuple[str, ...]]:
    if metadata_errors or status is None:
        return MISSING_REQUIRED_ARTIFACTS, ()
    if status == STATUS_FAILED:
        return FAILED_OR_INCOMPLETE, ()
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        return FAILED_OR_INCOMPLETE, ()
    if not predictions_present:
        return MISSING_REQUIRED_ARTIFACTS, ()

    # Distinguish invalid artifacts from provenance staleness.
    try:
        validate_s7_run_for_downstream(run_dir)
    except ProvenanceError as exc:
        message = str(exc)
        # Model vs source staleness.
        model_info = assess_model_source_staleness(run_dir)
        if model_info.get("stale"):
            # If S6 is also bad, prefer reporting both; classify by primary cause.
            s6_stale = False
            if source_s6_run_dir is not None:
                s6_stale = not is_s6_run_eligible_for_downstream(source_s6_run_dir)
            if s6_stale and "model" not in message.lower():
                return STALE_SOURCE, (message,)
            if s6_stale:
                return STALE_SOURCE, (message, *model_info.get("reasons", []))
            return STALE_MODELS, tuple(model_info.get("reasons", []) or (message,))
        if source_s6_run_dir is not None and not is_s6_run_eligible_for_downstream(
            source_s6_run_dir
        ):
            return STALE_SOURCE, (message,)
        if "SHA256" in message or "eligible" in message.lower() or "S6" in message:
            return STALE_SOURCE, (message,)
        return INVALID_ARTIFACT, (message,)
    except PublicationError as exc:
        return INVALID_ARTIFACT, (str(exc),)
    except Exception as exc:  # noqa: BLE001 — discovery must not crash
        return INVALID_ARTIFACT, (str(exc),)

    if is_s7_run_eligible_for_downstream(run_dir):
        if status == STATUS_COMPLETED_WITH_WARNINGS or warning_count > 0:
            return COMPLETED_WITH_WARNINGS, ()
        return ELIGIBLE_DOWNSTREAM, ()
    return STALE_SOURCE, ()


def _run_sort_key(run: BehavioralInferenceRunSummary) -> tuple[datetime, str]:
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
