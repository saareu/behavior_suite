"""Working draft directories under behavioral_review__*/working/."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from behavioral_review.contracts import (
    BEHAVIORAL_REVIEW_DIRNAME,
    BEHAVIORAL_REVIEW_RUN_PREFIX,
    CORRECTION_EVENTS_FILENAME,
    JOURNAL_CURSOR_FILENAME,
    LOCKS_DIRNAME,
    NOTES_FILENAME,
    PUBLICATION_STATE_DRAFT,
    REVIEW_BOUT_STATUS_FILENAME,
    RUN_LIFECYCLE_IN_PROGRESS,
    RUN_LIFECYCLE_NOT_STARTED,
    SNAPSHOTS_DIRNAME,
    WORKING_DIRNAME,
    WORKING_MANIFEST_FILENAME,
    WORKING_MANIFEST_SCHEMA_VERSION,
    DraftError,
)
from behavioral_review.events import utc_now_iso
from behavioral_review.journal import EventJournal, JournalCursor
from behavioral_review.locks import WriterLock
from behavioral_review.source import ReviewSourcePackage


def _next_review_run_id(review_root: Path, *, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    base = f"{BEHAVIORAL_REVIEW_RUN_PREFIX}__{stamp}"
    candidate = base
    suffix = 1
    while (review_root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


@dataclass
class WorkingDraft:
    """Durable in-progress review draft (not a published reviewed run)."""

    review_run_id: str
    review_run_dir: Path
    working_dir: Path
    reviewer_id: str
    parent_package_run_id: str
    parent_package_dir: Path
    working_manifest: dict[str, Any]
    journal: EventJournal
    lock: WriterLock
    warnings: list[str] = field(default_factory=list)

    @property
    def publication_state(self) -> str:
        return str(
            self.working_manifest.get("publication_state") or PUBLICATION_STATE_DRAFT
        )

    @property
    def lifecycle_status(self) -> str:
        if self.journal.events:
            return RUN_LIFECYCLE_IN_PROGRESS
        return str(
            self.working_manifest.get("lifecycle_status") or RUN_LIFECYCLE_NOT_STARTED
        )


def create_working_draft(
    source: ReviewSourcePackage,
    *,
    reviewer_id: str,
    review_root: Path | str | None = None,
    review_run_id: str | None = None,
    acquire_lock: bool = True,
) -> WorkingDraft:
    """Explicitly create a new draft. Never called by inspection-only open."""

    reviewer = str(reviewer_id).strip()
    if not reviewer:
        raise DraftError("reviewer_id is required to create a working draft.")
    if not source.s8_states_available or source.source_s8.states_path is None:
        raise DraftError(
            "Cannot create an editing draft without resolvable S8 behavior_states."
        )

    if review_root is None:
        if source.session_root is None:
            raise DraftError(
                "session_root unknown; pass review_root explicitly for draft create."
            )
        review_root_path = Path(source.session_root) / BEHAVIORAL_REVIEW_DIRNAME
    else:
        review_root_path = Path(review_root)
    review_root_path.mkdir(parents=True, exist_ok=True)

    run_id = review_run_id or _next_review_run_id(review_root_path)
    if not run_id.startswith(f"{BEHAVIORAL_REVIEW_RUN_PREFIX}__"):
        raise DraftError(
            f"review_run_id must start with {BEHAVIORAL_REVIEW_RUN_PREFIX}__; "
            f"got {run_id!r}."
        )
    run_dir = review_root_path / run_id
    if run_dir.exists():
        raise DraftError(f"Review run directory already exists: {run_dir}")

    working_dir = run_dir / WORKING_DIRNAME
    (working_dir / LOCKS_DIRNAME).mkdir(parents=True)
    (working_dir / SNAPSHOTS_DIRNAME).mkdir(parents=True)

    manifest = {
        "schema_version": WORKING_MANIFEST_SCHEMA_VERSION,
        "review_run_id": run_id,
        "publication_state": PUBLICATION_STATE_DRAFT,
        "lifecycle_status": RUN_LIFECYCLE_NOT_STARTED,
        "created_at_utc": utc_now_iso(),
        "created_by_reviewer_id": reviewer,
        "parent_package_run_id": source.package_run_id,
        "parent_package_dir": str(source.package_dir),
        "parent_package_manifest_sha256": source.package_manifest_sha256,
        "parent_sync_index_sha256": source.sync_index_sha256,
        "parent_behavior_bouts_sha256": source.behavior_bouts_sha256,
        "source_s8_run_id": source.source_s8.run_id,
        "source_s8_states_path": (
            str(source.source_s8.states_path) if source.source_s8.states_path else None
        ),
        "source_s8_states_sha256_recorded": source.source_s8.states_sha256_recorded,
        "source_s8_bouts_sha256_recorded": source.source_s8.bouts_sha256_recorded,
        "source_video_path": (
            str(source.source_video.path) if source.source_video.path else None
        ),
        "source_video_sha256": source.source_video.sha256,
        "n_dense_samples": int(source.n_dense_samples),
        "fps": 120.0,
        "auto_accepted_on_open": False,
    }
    _atomic_write_json(working_dir / WORKING_MANIFEST_FILENAME, manifest)
    # Empty journal + cursor.
    cursor = JournalCursor.empty()
    _atomic_write_json(working_dir / JOURNAL_CURSOR_FILENAME, cursor.to_dict())
    (working_dir / CORRECTION_EVENTS_FILENAME).write_text("", encoding="utf-8")
    (working_dir / REVIEW_BOUT_STATUS_FILENAME).write_text("", encoding="utf-8")
    (working_dir / NOTES_FILENAME).write_text("", encoding="utf-8")
    # Draft run meta (not published).
    _atomic_write_json(
        run_dir / "run_meta.json",
        {
            "schema_version": "behavioral_review_run_meta_v1",
            "review_run_id": run_id,
            "publication_state": PUBLICATION_STATE_DRAFT,
            "lifecycle_status": RUN_LIFECYCLE_NOT_STARTED,
            "parent_package_run_id": source.package_run_id,
            "created_at_utc": manifest["created_at_utc"],
            "created_by_reviewer_id": reviewer,
        },
    )

    journal = EventJournal.open(working_dir)
    lock = WriterLock(working_dir, review_run_id=run_id)
    if acquire_lock:
        lock.acquire(reviewer_id=reviewer)
    return WorkingDraft(
        review_run_id=run_id,
        review_run_dir=run_dir,
        working_dir=working_dir,
        reviewer_id=reviewer,
        parent_package_run_id=source.package_run_id,
        parent_package_dir=source.package_dir,
        working_manifest=manifest,
        journal=journal,
        lock=lock,
        warnings=list(journal.recovery_warnings),
    )


def open_working_draft(
    review_run_dir: Path | str,
    *,
    reviewer_id: str,
    expected_package_run_id: str | None = None,
    acquire_lock: bool = True,
    force_stale_lock: bool = False,
) -> WorkingDraft:
    """Reopen an existing draft and recover journal/cursor."""

    reviewer = str(reviewer_id).strip()
    if not reviewer:
        raise DraftError("reviewer_id is required to open a working draft for editing.")

    run_dir = Path(review_run_dir).resolve()
    working_dir = run_dir / WORKING_DIRNAME
    manifest_path = working_dir / WORKING_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DraftError(f"working_manifest.json missing under {working_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DraftError(f"Invalid working_manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DraftError("working_manifest.json must be an object.")

    run_id = str(manifest.get("review_run_id") or run_dir.name)
    parent_pkg = str(manifest.get("parent_package_run_id") or "")
    if expected_package_run_id is not None and parent_pkg != expected_package_run_id:
        raise DraftError(
            f"Draft parent package {parent_pkg!r} does not match expected "
            f"{expected_package_run_id!r}."
        )
    parent_dir_text = manifest.get("parent_package_dir")
    parent_dir = Path(str(parent_dir_text)) if parent_dir_text else Path()

    journal = EventJournal.open(working_dir)
    lock = WriterLock(working_dir, review_run_id=run_id)
    if acquire_lock:
        lock.acquire(reviewer_id=reviewer, force_stale=force_stale_lock)

    warnings = list(journal.recovery_warnings)
    return WorkingDraft(
        review_run_id=run_id,
        review_run_dir=run_dir,
        working_dir=working_dir,
        reviewer_id=reviewer,
        parent_package_run_id=parent_pkg,
        parent_package_dir=parent_dir,
        working_manifest=manifest,
        journal=journal,
        lock=lock,
        warnings=warnings,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
