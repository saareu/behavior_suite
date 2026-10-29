"""Atomic immutable reviewed-run publication lifecycle."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    CORRECTION_EVENTS_HISTORY_FILENAME,
    CORRECTION_EVENTS_PARQUET_FILENAME,
    PROCESSING_LOG_FILENAME,
    PUBLICATION_STATE_DRAFT,
    PUBLICATION_STATE_FAILED,
    PUBLICATION_STATE_PUBLISHED,
    QC_FILENAME,
    REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS,
    REVIEW_BOUT_STATUS_PARQUET_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    REVIEW_MANIFEST_SCHEMA_VERSION,
    REVIEW_RUN_META_SCHEMA_VERSION,
    REVIEW_SETTINGS_SCHEMA_VERSION,
    REVIEWED_BEHAVIOR_BOUTS_FILENAME,
    REVIEWED_BEHAVIOR_STATES_FILENAME,
    REVIEWER_NOTES_PARQUET_FILENAME,
    RUN_LIFECYCLE_IN_PROGRESS,
    RUN_LIFECYCLE_PUBLISHED_ACCEPTED,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    SOURCE_TYPE_REVIEWED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    WORKING_DIRNAME,
    WORKING_MANIFEST_FILENAME,
    PublicationError,
)
from behavioral_review.draft import WorkingDraft
from behavioral_review.editing import EditingSession
from behavioral_review.eligibility import (
    PublishEligibilityReport,
    assert_draft_publishable,
)
from behavioral_review.events import utc_now_iso
from behavioral_review.qc import ReviewQcSummary, assert_structural_qc, evaluate_review_qc
from behavioral_review.snapshot import PublicationSnapshot, build_publication_snapshot
from behavioral_review.source import (
    ReviewSourcePackage,
    assert_no_upstream_writes,
    open_review_source,
    snapshot_upstream_fingerprints,
)
from project.file_digest import sha256_file
from results_packaging.export import atomic_parquet_write, write_json, write_yaml


@dataclass
class ReviewedPublicationResult:
    """Outcome of an atomic S10 reviewed-run publication attempt."""

    success: bool
    status: str
    review_run_id: str
    review_run_dir: Path
    parent_package_run_id: str
    parent_package_dir: Path
    publication_state: str
    currently_eligible: bool
    classification: str | None
    source_type: str = SOURCE_TYPE_REVIEWED
    qc: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    science_written: bool = False
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def publish_reviewed_run(
    session: EditingSession,
    *,
    settings: dict[str, Any] | None = None,
) -> ReviewedPublicationResult:
    """Publish an immutable reviewed run from an open editing session draft.

    Requires editing mode, held writer lock, valid journal cursor, and a
    currently eligible parent S9 package whose digests still match draft pins.
    Does not mutate S8/S9 artifacts or require every bout to be accepted.
    """

    if session.draft is None:
        raise PublicationError(
            "No working draft open. Create or open a draft before publication."
        )
    draft = session.draft
    if not session.reviewer_id or not str(session.reviewer_id).strip():
        raise PublicationError("Valid reviewer identity is required for publication.")
    if session.baseline_states is None:
        session._load_baseline_states()  # noqa: SLF001 — shared session loader
    assert session.baseline_states is not None

    source = session.inspection.source
    upstream_snaps = snapshot_upstream_fingerprints(source)
    return _publish_from_draft(
        draft=draft,
        source=source,
        baseline_states=session.baseline_states,
        reviewer_id=str(session.reviewer_id),
        settings=settings,
        upstream_fingerprints=upstream_snaps,
    )


def publish_working_draft(
    draft: WorkingDraft,
    *,
    reviewer_id: str | None = None,
    settings: dict[str, Any] | None = None,
    require_writer_lock: bool = True,
) -> ReviewedPublicationResult:
    """Publish from an already-open working draft (headless entry point)."""

    reviewer = str(reviewer_id or draft.reviewer_id).strip()
    if not reviewer:
        raise PublicationError("Valid reviewer identity is required for publication.")
    source = open_review_source(draft.parent_package_dir, require_s8_states=True)
    if source.source_s8.states_path is None or not source.source_s8.states_path.is_file():
        raise PublicationError("Cannot publish without resolvable S8 behavior_states.")
    baseline = pd.read_parquet(source.source_s8.states_path)
    upstream_snaps = snapshot_upstream_fingerprints(source)
    # Ensure lock held when required.
    if require_writer_lock and not draft.lock.is_held:
        draft.lock.acquire(reviewer_id=reviewer)
    return _publish_from_draft(
        draft=draft,
        source=source,
        baseline_states=baseline,
        reviewer_id=reviewer,
        settings=settings,
        upstream_fingerprints=upstream_snaps,
        require_writer_lock=require_writer_lock,
    )


def _publish_from_draft(
    *,
    draft: WorkingDraft,
    source: ReviewSourcePackage,
    baseline_states: pd.DataFrame,
    reviewer_id: str,
    settings: dict[str, Any] | None,
    upstream_fingerprints: dict[Path, tuple[Any, ...]],
    require_writer_lock: bool = True,
) -> ReviewedPublicationResult:
    run_dir = draft.review_run_dir
    run_id = draft.review_run_id
    paths = _resolve_paths(run_dir)
    created_at = utc_now_iso()
    log_lines: list[str] = [
        f"[{created_at}] Beginning reviewed publication for {run_id}",
    ]

    eligibility = assert_draft_publishable(
        draft,
        source=source,
        require_writer_lock=require_writer_lock,
    )
    if not eligibility.eligible:
        log_lines.append("Publication blocked by eligibility gates.")
        for b in eligibility.blockers:
            log_lines.append(f"  BLOCKER: {b}")
        _append_processing_log(paths.processing_log, log_lines)
        eligibility.raise_if_blocked()

    settings_payload = _build_settings(
        draft=draft,
        source=source,
        reviewer_id=reviewer_id,
        settings=settings,
        eligibility=eligibility,
    )
    write_yaml(paths.settings_used, settings_payload)
    _write_run_meta(
        paths.run_meta,
        run_id=run_id,
        status=STATUS_RUNNING,
        publication_state=PUBLICATION_STATE_DRAFT,
        lifecycle_status=RUN_LIFECYCLE_IN_PROGRESS,
        created_at=created_at,
        completed_at=None,
        draft=draft,
        source=source,
        reviewer_id=reviewer_id,
        qc=None,
        error_message=None,
        science_written=False,
        settings=settings_payload,
    )

    snapshot: PublicationSnapshot | None = None
    qc: ReviewQcSummary | None = None
    science_written = False
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    artifact_paths: dict[str, Path] = {}
    currently_eligible = False
    classification: str | None = "failed_or_incomplete"

    try:
        log_lines.append("Validating parent pins and building publication snapshot…")
        snapshot = build_publication_snapshot(
            journal=draft.journal,
            baseline_states=baseline_states,
            source=source,
            review_run_id=run_id,
        )
        log_lines.append(
            f"Snapshot cursor={snapshot.publication_cursor_event_seq} "
            f"applied={snapshot.n_events_applied} history={snapshot.n_events_history}"
        )

        # Write science artifacts (atomic parquet).
        artifact_paths["correction_events"] = atomic_parquet_write(
            paths.correction_events, snapshot.applied_events_frame()
        )
        artifact_paths["correction_events_history"] = atomic_parquet_write(
            paths.correction_events_history, snapshot.history_events_frame()
        )
        artifact_paths["reviewed_behavior_states"] = atomic_parquet_write(
            paths.reviewed_states, snapshot.reviewed_states
        )
        artifact_paths["reviewed_behavior_bouts"] = atomic_parquet_write(
            paths.reviewed_bouts, snapshot.reviewed_bouts
        )
        status_frame = snapshot.status_frame()
        artifact_paths["review_bout_status"] = atomic_parquet_write(
            paths.review_bout_status, status_frame
        )
        notes = snapshot.notes_frame()
        if notes is not None and not notes.empty:
            artifact_paths["reviewer_notes"] = atomic_parquet_write(
                paths.reviewer_notes, notes
            )

        digests = {
            name: sha256_file(path) for name, path in artifact_paths.items()
        }
        qc = evaluate_review_qc(
            snapshot=snapshot,
            source=source,
            eligibility_warnings=list(eligibility.warnings),
            artifact_digests=digests,
        )
        assert_structural_qc(qc)

        # Revalidate source identity before sealing.
        log_lines.append("Revalidating parent identity before seal…")
        recheck = assert_draft_publishable(
            draft, source=open_review_source(source.package_dir, require_s8_states=True),
            require_writer_lock=require_writer_lock,
        )
        if not recheck.eligible:
            raise PublicationError(
                "Parent became stale during publication: "
                + "; ".join(recheck.blockers)
            )
        assert_no_upstream_writes(
            package_dir=source.package_dir,
            s8_run_dir=source.source_s8.run_dir,
            video_path=source.source_video.path,
            snapshots=upstream_fingerprints,
        )

        manifest = build_review_manifest(
            draft=draft,
            source=source,
            snapshot=snapshot,
            qc=qc,
            artifact_digests=digests,
            reviewer_id=reviewer_id,
            settings=settings_payload,
        )
        write_json(paths.review_manifest, manifest)
        write_json(paths.qc, qc.to_metadata())
        artifact_paths["review_manifest"] = paths.review_manifest
        artifact_paths["qc"] = paths.qc
        science_written = True

        if qc.has_warnings:
            status = STATUS_COMPLETED_WITH_WARNINGS
            warnings = tuple(qc.warnings)
        else:
            status = STATUS_COMPLETED
            warnings = ()
        success = True
        currently_eligible = True
        classification = (
            STATUS_COMPLETED_WITH_WARNINGS
            if status == STATUS_COMPLETED_WITH_WARNINGS
            else "eligible"
        )
        log_lines.append(f"Publication succeeded with status={status}.")

        # Seal working manifest publication_state without deleting journal.
        _seal_working_manifest(
            draft,
            publication_state=PUBLICATION_STATE_PUBLISHED,
            lifecycle_status=RUN_LIFECYCLE_PUBLISHED_ACCEPTED,
            published_at=utc_now_iso(),
            publication_cursor_event_seq=snapshot.publication_cursor_event_seq,
        )
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        currently_eligible = False
        classification = "failed_or_incomplete"
        science_written = False
        _unlink_science_artifacts(paths)
        artifact_paths = {}
        log_lines.append(f"Publication failed: {exc}")
        with contextlib.suppress(OSError, TypeError, ValueError):
            _seal_working_manifest(
                draft,
                publication_state=PUBLICATION_STATE_FAILED,
                lifecycle_status=RUN_LIFECYCLE_IN_PROGRESS,
                published_at=None,
                publication_cursor_event_seq=draft.journal.cursor.cursor_event_seq,
                error_message=error_message,
            )
        # Keep draft recoverable: restore draft publication_state for retry.
        with contextlib.suppress(OSError, TypeError, ValueError):
            _seal_working_manifest(
                draft,
                publication_state=PUBLICATION_STATE_DRAFT,
                lifecycle_status=RUN_LIFECYCLE_IN_PROGRESS,
                published_at=None,
                publication_cursor_event_seq=draft.journal.cursor.cursor_event_seq,
                error_message=error_message,
            )

    completed_at = utc_now_iso()
    publication_state = (
        PUBLICATION_STATE_PUBLISHED if success else PUBLICATION_STATE_FAILED
    )
    lifecycle = (
        RUN_LIFECYCLE_PUBLISHED_ACCEPTED if success else RUN_LIFECYCLE_IN_PROGRESS
    )
    _write_run_meta(
        paths.run_meta,
        run_id=run_id,
        status=status,
        publication_state=publication_state,
        lifecycle_status=lifecycle,
        created_at=created_at,
        completed_at=completed_at,
        draft=draft,
        source=source,
        reviewer_id=reviewer_id,
        qc=qc.to_metadata() if qc is not None else None,
        error_message=error_message,
        science_written=science_written,
        settings=settings_payload,
        snapshot=snapshot,
    )
    _append_processing_log(paths.processing_log, log_lines)
    artifact_paths["run_meta"] = paths.run_meta
    artifact_paths["settings_used"] = paths.settings_used
    artifact_paths["processing_log"] = paths.processing_log

    # Final upstream immutability check (even on failure).
    with contextlib.suppress(Exception):
        assert_no_upstream_writes(
            package_dir=source.package_dir,
            s8_run_dir=source.source_s8.run_dir,
            video_path=source.source_video.path,
            snapshots=upstream_fingerprints,
        )

    return ReviewedPublicationResult(
        success=success,
        status=status,
        review_run_id=run_id,
        review_run_dir=run_dir,
        parent_package_run_id=source.package_run_id,
        parent_package_dir=source.package_dir,
        publication_state=publication_state,
        currently_eligible=currently_eligible,
        classification=classification,
        qc=qc.to_metadata() if qc is not None else None,
        warnings=warnings,
        error_message=error_message,
        science_written=science_written,
        artifact_paths=artifact_paths,
        diagnostics={
            "publication_cursor_event_seq": (
                snapshot.publication_cursor_event_seq if snapshot else None
            ),
            "n_events_applied": snapshot.n_events_applied if snapshot else None,
            "n_events_history": snapshot.n_events_history if snapshot else None,
            "n_reviewed_bouts": (
                int(len(snapshot.reviewed_bouts)) if snapshot is not None else None
            ),
            "working_dir_intact": (run_dir / WORKING_DIRNAME).is_dir(),
            "journal_intact": (
                run_dir / WORKING_DIRNAME / "correction_events.jsonl"
            ).is_file(),
        },
    )


def build_review_manifest(
    *,
    draft: WorkingDraft,
    source: ReviewSourcePackage,
    snapshot: PublicationSnapshot,
    qc: ReviewQcSummary,
    artifact_digests: dict[str, str],
    reviewer_id: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "source_type": SOURCE_TYPE_REVIEWED,
        "review_run_id": draft.review_run_id,
        "created_at_utc": utc_now_iso(),
        "published_by_reviewer_id": reviewer_id,
        "fps": 120.0,
        "publication_cursor_event_seq": snapshot.publication_cursor_event_seq,
        "redo_high_water_seq": snapshot.redo_high_water_seq,
        "superseded_event_seqs": list(snapshot.superseded_event_seqs),
        "n_events_applied": snapshot.n_events_applied,
        "n_events_history": snapshot.n_events_history,
        "parent_package_run_id": source.package_run_id,
        "parent_package_dir": str(source.package_dir),
        "parent_package_manifest_sha256": source.package_manifest_sha256,
        "parent_sync_index_sha256": source.sync_index_sha256,
        "parent_behavior_bouts_sha256": source.behavior_bouts_sha256,
        "parent_package_status": source.package_status,
        "source_s8_run_id": source.source_s8.run_id,
        "source_s8_run_dir": (
            str(source.source_s8.run_dir) if source.source_s8.run_dir else None
        ),
        "source_s8_states_path": (
            str(source.source_s8.states_path) if source.source_s8.states_path else None
        ),
        "source_s8_states_sha256": source.source_s8.states_sha256_recorded,
        "source_s8_bouts_sha256": source.source_s8.bouts_sha256_recorded,
        "source_s7_run_id": source.source_s7_run_id,
        "source_s6_run_id": source.source_s6_run_id,
        "prepared_video_path": (
            str(source.source_video.path) if source.source_video.path else None
        ),
        "prepared_video_sha256": source.source_video.sha256,
        "mode": "reference_lightweight",
        "n_dense_samples": int(source.n_dense_samples),
        "n_reviewed_state_rows": int(len(snapshot.reviewed_states)),
        "n_reviewed_bouts": int(len(snapshot.reviewed_bouts)),
        "artifact_digests": artifact_digests,
        "qc": {
            "passed": qc.passed,
            "has_warnings": qc.has_warnings,
            "warning_count": len(qc.warnings),
            "blocker_count": len(qc.blockers),
        },
        "settings_schema_version": REVIEW_SETTINGS_SCHEMA_VERSION,
        "settings_keys": sorted(settings.keys()),
        "auto_accepted_on_open": False,
        "acceptance_required_for_publish": False,
    }


@dataclass(frozen=True, slots=True)
class _PublishPaths:
    run_dir: Path
    correction_events: Path
    correction_events_history: Path
    reviewed_states: Path
    reviewed_bouts: Path
    review_bout_status: Path
    reviewer_notes: Path
    review_manifest: Path
    qc: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path


def _resolve_paths(run_dir: Path) -> _PublishPaths:
    return _PublishPaths(
        run_dir=run_dir,
        correction_events=run_dir / CORRECTION_EVENTS_PARQUET_FILENAME,
        correction_events_history=run_dir / CORRECTION_EVENTS_HISTORY_FILENAME,
        reviewed_states=run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME,
        reviewed_bouts=run_dir / REVIEWED_BEHAVIOR_BOUTS_FILENAME,
        review_bout_status=run_dir / REVIEW_BOUT_STATUS_PARQUET_FILENAME,
        reviewer_notes=run_dir / REVIEWER_NOTES_PARQUET_FILENAME,
        review_manifest=run_dir / REVIEW_MANIFEST_FILENAME,
        qc=run_dir / QC_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
    )


def _unlink_science_artifacts(paths: _PublishPaths) -> None:
    for path in (
        paths.correction_events,
        paths.correction_events_history,
        paths.reviewed_states,
        paths.reviewed_bouts,
        paths.review_bout_status,
        paths.reviewer_notes,
        paths.review_manifest,
        paths.qc,
    ):
        with contextlib.suppress(OSError):
            if path.is_file():
                path.unlink()
    # Remove stray temp parquet fragments.
    for tmp in paths.run_dir.glob("*.parquet.tmp"):
        with contextlib.suppress(OSError):
            tmp.unlink()


def _build_settings(
    *,
    draft: WorkingDraft,
    source: ReviewSourcePackage,
    reviewer_id: str,
    settings: dict[str, Any] | None,
    eligibility: PublishEligibilityReport,
) -> dict[str, Any]:
    payload = {
        "schema_version": REVIEW_SETTINGS_SCHEMA_VERSION,
        "review_run_id": draft.review_run_id,
        "parent_package_run_id": source.package_run_id,
        "reviewer_id": reviewer_id,
        "fps": 120.0,
        "source_type": SOURCE_TYPE_REVIEWED,
        "acceptance_required_for_publish": False,
        "parent_eligible_at_publish": eligibility.parent_eligible,
        "parent_pins_match": eligibility.parent_pins_match,
    }
    if settings:
        payload["operator_settings"] = dict(settings)
    return payload


def _write_run_meta(
    path: Path,
    *,
    run_id: str,
    status: str,
    publication_state: str,
    lifecycle_status: str,
    created_at: str,
    completed_at: str | None,
    draft: WorkingDraft,
    source: ReviewSourcePackage,
    reviewer_id: str,
    qc: dict[str, Any] | None,
    error_message: str | None,
    science_written: bool,
    settings: dict[str, Any],
    snapshot: PublicationSnapshot | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": REVIEW_RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "review_run_id": run_id,
        "status": status,
        "publication_state": publication_state,
        "lifecycle_status": lifecycle_status,
        "source_type": SOURCE_TYPE_REVIEWED,
        "created_at_utc": created_at,
        "completed_at_utc": completed_at,
        "published_by_reviewer_id": reviewer_id,
        "parent_package_run_id": source.package_run_id,
        "parent_package_dir": str(source.package_dir),
        "source_s8_run_id": source.source_s8.run_id,
        "science_written": science_written,
        "error_message": error_message,
        "qc": qc,
        "settings_schema_version": settings.get("schema_version"),
        "working_dir": str(draft.working_dir),
        "required_artifacts": list(REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS),
    }
    if snapshot is not None:
        payload["publication_cursor_event_seq"] = snapshot.publication_cursor_event_seq
        payload["n_events_applied"] = snapshot.n_events_applied
        payload["n_events_history"] = snapshot.n_events_history
        payload["n_reviewed_bouts"] = int(len(snapshot.reviewed_bouts))
        payload["n_dense_samples"] = int(source.n_dense_samples)
    write_json(path, payload)


def _seal_working_manifest(
    draft: WorkingDraft,
    *,
    publication_state: str,
    lifecycle_status: str,
    published_at: str | None,
    publication_cursor_event_seq: int,
    error_message: str | None = None,
) -> None:
    manifest = dict(draft.working_manifest)
    manifest["publication_state"] = publication_state
    manifest["lifecycle_status"] = lifecycle_status
    manifest["publication_cursor_event_seq"] = int(publication_cursor_event_seq)
    if published_at is not None:
        manifest["published_at_utc"] = published_at
    if error_message is not None:
        manifest["last_publish_error"] = error_message
    path = draft.working_dir / WORKING_MANIFEST_FILENAME
    tmp = path.with_suffix(".json.tmp")
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    draft.working_manifest = manifest


def _append_processing_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(f"{line}\n")
        handle.write(f"[{stamp}] log flush\n")
        handle.flush()
        os.fsync(handle.fileno())
