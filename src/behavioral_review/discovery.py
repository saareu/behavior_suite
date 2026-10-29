"""Discover, classify, and reopen immutable S10 reviewed runs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    BEHAVIORAL_REVIEW_DIRNAME,
    BEHAVIORAL_REVIEW_RUN_PREFIX,
    CLASS_COMPLETED_WITH_WARNINGS,
    CLASS_DRAFT,
    CLASS_ELIGIBLE,
    CLASS_FAILED_OR_INCOMPLETE,
    CLASS_MISSING_REQUIRED_ARTIFACTS,
    CLASS_STALE_PARENT,
    CLASS_SUPERSEDED,
    CORRECTION_EVENTS_HISTORY_FILENAME,
    CORRECTION_EVENTS_PARQUET_FILENAME,
    PUBLICATION_STATE_DRAFT,
    QC_FILENAME,
    REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS,
    REVIEW_BOUT_STATUS_PARQUET_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    REVIEWED_BEHAVIOR_BOUTS_FILENAME,
    REVIEWED_BEHAVIOR_STATES_FILENAME,
    RUN_META_FILENAME,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    WORKING_DIRNAME,
    WORKING_MANIFEST_FILENAME,
    ProvenanceError,
    PublicationError,
)
from project.file_digest import sha256_file
from results_packaging.export import load_json_object
from results_packaging.publication import (
    ProvenanceError as S9ProvenanceError,
)
from results_packaging.publication import (
    PublicationError as S9PublicationError,
)
from results_packaging.publication import (
    is_s9_run_eligible_for_downstream,
    validate_s9_run_for_downstream,
)


@dataclass(frozen=True, slots=True)
class ReviewedRunSummary:
    """Lightweight summary of one behavioral_review__* directory."""

    run_dir: Path
    review_run_id: str
    status: str | None
    publication_state: str | None
    classification: str
    parent_package_run_id: str | None
    parent_package_dir: Path | None
    source_s8_run_id: str | None
    source_type: str
    created_at: str | None
    completed_at: str | None
    warning_count: int
    error_message: str | None
    currently_eligible: bool
    historical_completed: bool
    superseded_by: str | None = None
    metadata_errors: tuple[str, ...] = ()
    staleness_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReviewedProjectSummary:
    """Discovery result for one session / review root."""

    session_root: Path | None
    review_root: Path
    runs: tuple[ReviewedRunSummary, ...]

    @property
    def eligible_runs(self) -> tuple[ReviewedRunSummary, ...]:
        return tuple(
            run
            for run in self.runs
            if run.classification in (CLASS_ELIGIBLE, CLASS_COMPLETED_WITH_WARNINGS)
            and run.currently_eligible
        )

    @property
    def completed_runs(self) -> tuple[ReviewedRunSummary, ...]:
        return tuple(run for run in self.runs if run.historical_completed)


def summarize_behavioral_review_project(
    session_root: Path | str | None = None,
    *,
    review_root: Path | str | None = None,
) -> ReviewedProjectSummary:
    """Summarize reviewed runs under a session or explicit review root."""

    if review_root is not None:
        root = Path(review_root).expanduser().resolve()
        session = Path(session_root).expanduser().resolve() if session_root else None
    elif session_root is not None:
        session = Path(session_root).expanduser().resolve()
        root = session / BEHAVIORAL_REVIEW_DIRNAME
    else:
        raise PublicationError("Pass session_root or review_root for discovery.")
    runs = tuple(sorted(_discover_runs(root), key=_run_sort_key, reverse=True))
    runs = _annotate_supersession(runs)
    return ReviewedProjectSummary(
        session_root=session,
        review_root=root,
        runs=runs,
    )


def list_completed_reviewed_runs(
    session_root: Path | str | None = None,
    *,
    review_root: Path | str | None = None,
) -> tuple[ReviewedRunSummary, ...]:
    """Historically completed reviewed runs (may include currently stale)."""

    return summarize_behavioral_review_project(
        session_root, review_root=review_root
    ).completed_runs


def find_latest_eligible_reviewed_run(
    session_root: Path | str | None = None,
    *,
    review_root: Path | str | None = None,
    parent_package_run_id: str | None = None,
) -> ReviewedRunSummary | None:
    """Newest currently eligible reviewed run (optional parent filter).

    Does **not** silently replace an explicitly requested run elsewhere; this
    helper is only for discovery UIs that ask for "latest eligible".
    """

    summary = summarize_behavioral_review_project(
        session_root, review_root=review_root
    )
    eligible = summary.eligible_runs
    if parent_package_run_id is not None:
        eligible = tuple(
            r for r in eligible if r.parent_package_run_id == parent_package_run_id
        )
    return eligible[0] if eligible else None


def is_s10_run_eligible_for_downstream(review_run_dir: Path | str) -> bool:
    try:
        validate_s10_run_for_downstream(review_run_dir)
    except (PublicationError, ProvenanceError):
        return False
    return True


def validate_s10_run_for_downstream(review_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed reviewed run for downstream analysis use.

    Historical ``run_meta.status`` is never rewritten. Eligibility becomes false
    when parent S9/S8/video pins drift or required artifacts are missing.
    Does not silently substitute a newer review version.
    """

    run_dir = Path(review_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise PublicationError(f"S10 review run directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise PublicationError(f"S10 run_meta.json is missing: {meta_path}")
    meta = load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise PublicationError(
            f"S10 review run is not completed (status={status or 'missing'!r})."
        )
    if not bool(meta.get("science_written", True)):
        raise PublicationError("S10 review run science_written is false.")

    missing = [
        name
        for name in REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS
        if not (run_dir / name).is_file()
    ]
    if missing:
        raise PublicationError(
            f"Completed S10 review run is missing required artifacts: {missing}."
        )

    manifest = load_json_object(run_dir / REVIEW_MANIFEST_FILENAME)
    parent_dir_text = _optional_text(
        manifest.get("parent_package_dir") or meta.get("parent_package_dir")
    )
    if parent_dir_text is None:
        raise PublicationError("S10 review manifest missing parent_package_dir.")
    parent_dir = Path(parent_dir_text)
    parent_run_id = _optional_text(
        manifest.get("parent_package_run_id") or meta.get("parent_package_run_id")
    )

    # Parent must still be currently eligible AND digests must match pins.
    try:
        if not is_s9_run_eligible_for_downstream(parent_dir):
            raise ProvenanceError(
                "Parent S9 package is no longer eligible for downstream use "
                f"({parent_dir})."
            )
        s9_validation = validate_s9_run_for_downstream(parent_dir)
    except (S9PublicationError, S9ProvenanceError) as exc:
        raise ProvenanceError(
            f"Parent S9 package failed current eligibility validation: {exc}"
        ) from exc

    live_manifest_sha = sha256_file(parent_dir / "package_manifest.json")
    live_sync_sha = sha256_file(parent_dir / "sync_index.parquet")
    live_bouts_sha = sha256_file(parent_dir / "behavior_bouts.parquet")
    _assert_pin(
        "parent_package_manifest_sha256",
        manifest.get("parent_package_manifest_sha256"),
        live_manifest_sha,
    )
    _assert_pin(
        "parent_sync_index_sha256",
        manifest.get("parent_sync_index_sha256"),
        live_sync_sha,
    )
    _assert_pin(
        "parent_behavior_bouts_sha256",
        manifest.get("parent_behavior_bouts_sha256"),
        live_bouts_sha,
    )
    if parent_run_id is not None and parent_dir.name != parent_run_id:
        raise ProvenanceError(
            "Parent package directory name no longer matches pinned "
            f"parent_package_run_id ({parent_dir.name!r} vs {parent_run_id!r})."
        )

    # Local artifact digest integrity vs manifest.
    artifact_digests = manifest.get("artifact_digests")
    if isinstance(artifact_digests, Mapping):
        path_map = {
            "correction_events": run_dir / CORRECTION_EVENTS_PARQUET_FILENAME,
            "correction_events_history": run_dir / CORRECTION_EVENTS_HISTORY_FILENAME,
            "reviewed_behavior_states": run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME,
            "reviewed_behavior_bouts": run_dir / REVIEWED_BEHAVIOR_BOUTS_FILENAME,
            "review_bout_status": run_dir / REVIEW_BOUT_STATUS_PARQUET_FILENAME,
            "review_manifest": run_dir / REVIEW_MANIFEST_FILENAME,
            "qc": run_dir / QC_FILENAME,
        }
        for role, recorded in artifact_digests.items():
            path = path_map.get(str(role))
            if path is None or not path.is_file():
                continue
            if str(role) in {"review_manifest", "qc"}:
                # Manifest digests themselves may be recorded before final write;
                # verify science tables strictly.
                continue
            live = sha256_file(path)
            if recorded is not None and str(recorded) != live:
                raise ProvenanceError(
                    f"S10 artifact digest mismatch for {role}: "
                    f"recorded={recorded!r} live={live!r}."
                )

    return {
        "review_run_dir": run_dir,
        "review_run_id": meta.get("review_run_id") or meta.get("run_id") or run_dir.name,
        "status": status,
        "source_type": SOURCE_TYPE_REVIEWED,
        "parent_package_run_id": parent_run_id,
        "parent_package_dir": parent_dir.resolve(),
        "source_s8_run_id": _optional_text(
            manifest.get("source_s8_run_id") or meta.get("source_s8_run_id")
        ),
        "publication_cursor_event_seq": manifest.get("publication_cursor_event_seq"),
        "s9_validation": s9_validation,
        "currently_eligible": True,
        "historical_status": status,
    }


def reopen_reviewed_run(review_run_dir: Path | str) -> dict[str, Any]:
    """Load an immutable reviewed run without recomputing science tables.

    Raises when the requested run is not currently eligible. Never silently
    replaces the requested path with a newer review version.
    """

    run_dir = Path(review_run_dir).expanduser().resolve()
    validated = validate_s10_run_for_downstream(run_dir)
    meta = load_json_object(run_dir / RUN_META_FILENAME)
    manifest = load_json_object(run_dir / REVIEW_MANIFEST_FILENAME)
    qc = load_json_object(run_dir / QC_FILENAME)
    return {
        "run_dir": run_dir,
        "review_run_id": validated["review_run_id"],
        "status": meta.get("status"),
        "source_type": SOURCE_TYPE_REVIEWED,
        "currently_eligible": True,
        "validation": validated,
        "run_meta": meta,
        "review_manifest": manifest,
        "qc": qc,
        "correction_events": pd.read_parquet(
            run_dir / CORRECTION_EVENTS_PARQUET_FILENAME
        ),
        "correction_events_history": pd.read_parquet(
            run_dir / CORRECTION_EVENTS_HISTORY_FILENAME
        ),
        "reviewed_behavior_states": pd.read_parquet(
            run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        ),
        "reviewed_behavior_bouts": pd.read_parquet(
            run_dir / REVIEWED_BEHAVIOR_BOUTS_FILENAME
        ),
        "review_bout_status": pd.read_parquet(
            run_dir / REVIEW_BOUT_STATUS_PARQUET_FILENAME
        ),
    }


def classify_analysis_source(source_type: str) -> str:
    """Normalize automated vs reviewed selection vocabulary for consumers."""

    value = str(source_type or "").strip().lower()
    if value in {SOURCE_TYPE_AUTOMATED, "s9", "results_package", "package"}:
        return SOURCE_TYPE_AUTOMATED
    if value in {SOURCE_TYPE_REVIEWED, "s10", "behavioral_review", "review"}:
        return SOURCE_TYPE_REVIEWED
    raise PublicationError(
        f"Unknown analysis source_type={source_type!r}; "
        f"expected {SOURCE_TYPE_AUTOMATED!r} or {SOURCE_TYPE_REVIEWED!r}."
    )


def _discover_runs(review_root: Path) -> list[ReviewedRunSummary]:
    if not review_root.is_dir():
        return []
    return [
        _summarize_run(run_dir)
        for run_dir in review_root.iterdir()
        if run_dir.is_dir()
        and not run_dir.name.startswith("_")
        and run_dir.name.startswith(f"{BEHAVIORAL_REVIEW_RUN_PREFIX}__")
    ]


def _summarize_run(run_dir: Path) -> ReviewedRunSummary:
    meta_path = run_dir / RUN_META_FILENAME
    working_manifest_path = run_dir / WORKING_DIRNAME / WORKING_MANIFEST_FILENAME
    metadata_errors: list[str] = []
    meta = _load_json_mapping(meta_path, metadata_errors)
    working = _load_json_mapping(working_manifest_path, metadata_errors)

    status = _optional_text(meta.get("status"))
    publication_state = _optional_text(
        meta.get("publication_state") or working.get("publication_state")
    )
    parent_package_run_id = _optional_text(
        meta.get("parent_package_run_id") or working.get("parent_package_run_id")
    )
    parent_dir_text = _optional_text(
        meta.get("parent_package_dir") or working.get("parent_package_dir")
    )
    parent_dir = Path(parent_dir_text) if parent_dir_text else None
    source_s8 = _optional_text(
        meta.get("source_s8_run_id") or working.get("source_s8_run_id")
    )
    created_at = _optional_text(
        meta.get("created_at_utc") or working.get("created_at_utc")
    )
    completed_at = _optional_text(meta.get("completed_at_utc"))
    error_message = _optional_text(meta.get("error_message"))

    qc = meta.get("qc") if isinstance(meta.get("qc"), Mapping) else {}
    warning_count = 0
    if isinstance(qc, Mapping):
        raw = qc.get("warning_count")
        if isinstance(raw, int):
            warning_count = raw
        elif isinstance(qc.get("warnings"), list):
            warning_count = len(qc["warnings"])

    historical_completed = status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    core_present = all(
        (run_dir / name).is_file() for name in REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS
    )
    if historical_completed and not core_present:
        for name in REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS:
            if not (run_dir / name).is_file():
                metadata_errors.append(f"missing {name}")

    classification, staleness, currently_eligible = _classify_run(
        status=status,
        publication_state=publication_state,
        core_present=core_present,
        metadata_errors=tuple(metadata_errors),
        run_dir=run_dir,
        parent_dir=parent_dir,
        historical_completed=historical_completed,
    )

    return ReviewedRunSummary(
        run_dir=run_dir,
        review_run_id=run_dir.name,
        status=status,
        publication_state=publication_state,
        classification=classification,
        parent_package_run_id=parent_package_run_id,
        parent_package_dir=parent_dir,
        source_s8_run_id=source_s8,
        source_type=SOURCE_TYPE_REVIEWED,
        created_at=created_at,
        completed_at=completed_at,
        warning_count=warning_count,
        error_message=error_message,
        currently_eligible=currently_eligible,
        historical_completed=historical_completed,
        metadata_errors=tuple(metadata_errors),
        staleness_reasons=tuple(staleness),
    )


def _classify_run(
    *,
    status: str | None,
    publication_state: str | None,
    core_present: bool,
    metadata_errors: tuple[str, ...],
    run_dir: Path,
    parent_dir: Path | None,
    historical_completed: bool,
) -> tuple[str, list[str], bool]:
    staleness: list[str] = []
    draft_like = publication_state == PUBLICATION_STATE_DRAFT or (
        status in (None, "running")
        and (run_dir / WORKING_DIRNAME).is_dir()
        and not core_present
    )
    if draft_like and not historical_completed:
        return CLASS_DRAFT, staleness, False

    if status == STATUS_FAILED or (
        publication_state == "failed" and not historical_completed
    ):
        return CLASS_FAILED_OR_INCOMPLETE, staleness, False

    if historical_completed and not core_present:
        return CLASS_MISSING_REQUIRED_ARTIFACTS, list(metadata_errors), False

    if historical_completed and core_present:
        # Check current parent eligibility without rewriting meta.
        if parent_dir is None or not parent_dir.is_dir():
            staleness.append("parent_package_dir missing")
            return CLASS_STALE_PARENT, staleness, False
        try:
            validate_s10_run_for_downstream(run_dir)
            cls = (
                CLASS_COMPLETED_WITH_WARNINGS
                if status == STATUS_COMPLETED_WITH_WARNINGS
                else CLASS_ELIGIBLE
            )
            return cls, staleness, True
        except ProvenanceError as exc:
            staleness.append(str(exc))
            return CLASS_STALE_PARENT, staleness, False
        except PublicationError as exc:
            staleness.append(str(exc))
            return CLASS_FAILED_OR_INCOMPLETE, staleness, False

    return CLASS_FAILED_OR_INCOMPLETE, staleness, False


def _annotate_supersession(
    runs: tuple[ReviewedRunSummary, ...],
) -> tuple[ReviewedRunSummary, ...]:
    """Mark older published reviews sharing a parent as superseded.

    Supersession is a discovery classification only — immutable run_meta.status
    is never rewritten. Explicit ``validate_s10_run_for_downstream`` /
    ``reopen_reviewed_run`` on an older path still succeed when parent pins
    match; discovery simply excludes superseded runs from ``eligible_runs``.
    """

    newest_eligible_by_parent: dict[str, ReviewedRunSummary] = {}
    for run in runs:
        if not run.currently_eligible or run.parent_package_run_id is None:
            continue
        if run.classification not in (CLASS_ELIGIBLE, CLASS_COMPLETED_WITH_WARNINGS):
            continue
        key = run.parent_package_run_id
        if key not in newest_eligible_by_parent:
            newest_eligible_by_parent[key] = run

    out: list[ReviewedRunSummary] = []
    for run in runs:
        parent = run.parent_package_run_id
        newest = newest_eligible_by_parent.get(parent) if parent else None
        if (
            newest is not None
            and run.historical_completed
            and run.review_run_id != newest.review_run_id
            and run.classification
            in (CLASS_ELIGIBLE, CLASS_COMPLETED_WITH_WARNINGS, CLASS_STALE_PARENT)
        ):
            out.append(
                ReviewedRunSummary(
                    run_dir=run.run_dir,
                    review_run_id=run.review_run_id,
                    status=run.status,
                    publication_state=run.publication_state,
                    classification=CLASS_SUPERSEDED,
                    parent_package_run_id=run.parent_package_run_id,
                    parent_package_dir=run.parent_package_dir,
                    source_s8_run_id=run.source_s8_run_id,
                    source_type=run.source_type,
                    created_at=run.created_at,
                    completed_at=run.completed_at,
                    warning_count=run.warning_count,
                    error_message=run.error_message,
                    # Explicit path reopen may still validate; discovery latest excludes.
                    currently_eligible=False,
                    historical_completed=run.historical_completed,
                    superseded_by=newest.review_run_id,
                    metadata_errors=run.metadata_errors,
                    staleness_reasons=run.staleness_reasons,
                )
            )
        else:
            out.append(run)
    return tuple(out)


def _assert_pin(label: str, recorded: Any, live: str) -> None:
    if recorded is None:
        raise ProvenanceError(f"S10 manifest missing pin for {label}.")
    if str(recorded) != live:
        raise ProvenanceError(
            f"S10 parent pin mismatch for {label}: recorded={recorded!r} live={live!r}."
        )


def _load_json_mapping(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"unreadable {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{path.name} is not an object")
        return {}
    return payload


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _run_sort_key(run: ReviewedRunSummary) -> tuple[Any, ...]:
    stamp = _parse_run_stamp(run.review_run_id)
    completed = _parse_iso(run.completed_at)
    created = _parse_iso(run.created_at)
    return (stamp, completed, created, run.review_run_id)


def _parse_run_stamp(run_id: str) -> datetime:
    match = re.search(r"(\d{8}T\d{6})", run_id)
    if not match:
        return datetime.min
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return datetime.min


def _parse_iso(value: str | None) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min
