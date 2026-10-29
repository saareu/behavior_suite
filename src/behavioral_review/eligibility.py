"""Publication eligibility gates and parent-pin validation (U12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from behavioral_review.contracts import (
    PUBLICATION_STATE_DRAFT,
    PUBLICATION_STATE_PUBLISHED,
    ProvenanceError,
    PublicationError,
)
from behavioral_review.draft import WorkingDraft
from behavioral_review.events import CorrectionEvent
from behavioral_review.journal import EventJournal
from behavioral_review.source import ReviewSourcePackage, open_review_source
from project.file_digest import sha256_file
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


@dataclass
class PublishEligibilityReport:
    """Result of publish-gate checks for one working draft."""

    eligible: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parent_eligible: bool = False
    parent_pins_match: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_blocked(self) -> None:
        if self.eligible:
            return
        msg = "; ".join(self.blockers) if self.blockers else "Publication blocked."
        # Prefer provenance error when parent drift is the cause.
        if any("stale" in b.lower() or "digest" in b.lower() for b in self.blockers):
            raise ProvenanceError(msg)
        raise PublicationError(msg)


def assert_draft_publishable(
    draft: WorkingDraft,
    *,
    source: ReviewSourcePackage | None = None,
    require_writer_lock: bool = True,
) -> PublishEligibilityReport:
    """Hard publish gate: identity, journal, lock, parent eligibility + pins."""

    blockers: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "review_run_id": draft.review_run_id,
        "parent_package_run_id": draft.parent_package_run_id,
    }

    if not draft.review_run_id or not str(draft.review_run_id).strip():
        blockers.append("Explicit review draft identity (review_run_id) is required.")
    pub_state = draft.publication_state
    if pub_state == PUBLICATION_STATE_PUBLISHED:
        blockers.append(
            f"Review run {draft.review_run_id!r} is already published; "
            "create a new draft for a new review version."
        )
    elif pub_state not in (PUBLICATION_STATE_DRAFT, "failed"):
        blockers.append(f"Unexpected publication_state={pub_state!r}.")

    reviewer = str(draft.reviewer_id or "").strip()
    if not reviewer:
        blockers.append("Valid reviewer identity is required for publication.")
    created_by = str(
        draft.working_manifest.get("created_by_reviewer_id") or ""
    ).strip()
    if not created_by:
        blockers.append("Working draft is missing created_by_reviewer_id metadata.")

    if require_writer_lock and not draft.lock.is_held:
        blockers.append("Writer lock is not held; unresolved writer conflict.")
    elif require_writer_lock:
        existing = draft.lock.read()
        if existing is not None and existing.holder_id != draft.lock.holder_id:
            blockers.append(
                "Unresolved writer conflict: lock held by another writer "
                f"(holder_id={existing.holder_id})."
            )

    journal = draft.journal
    journal_blockers = validate_journal_for_publish(journal)
    blockers.extend(journal_blockers)

    parent_dir = Path(draft.parent_package_dir)
    parent_eligible = False
    parent_pins_match = False
    resolved_source = source
    try:
        if resolved_source is None:
            resolved_source = open_review_source(parent_dir, require_s8_states=True)
        parent_eligible = bool(resolved_source.currently_eligible)
        if not parent_eligible:
            blockers.append(
                "Parent S9 package is not currently eligible for downstream use "
                f"({parent_dir}). Publishing with a stale parent is prohibited (U12)."
            )
        pin_report = verify_parent_pins(draft, resolved_source)
        parent_pins_match = pin_report["pins_match"]
        details["parent_pin_check"] = pin_report
        if not parent_pins_match:
            blockers.extend(pin_report["mismatches"])
        # Exact eligible parent identity — never silently substitute.
        if resolved_source.package_run_id != draft.parent_package_run_id:
            blockers.append(
                "Opened parent package_run_id does not match the draft pin "
                f"({resolved_source.package_run_id!r} vs "
                f"{draft.parent_package_run_id!r}). Silent substitution is forbidden."
            )
        warnings.extend(list(resolved_source.warnings))
        if resolved_source.package_status == "completed_with_warnings":
            warnings.append(
                "Parent S9 package historical status is completed_with_warnings."
            )
    except Exception as exc:  # noqa: BLE001 — classify known vs unexpected
        from behavioral_review.contracts import SourceError

        if isinstance(
            exc,
            (
                SourceError,
                PublicationError,
                ProvenanceError,
                S9PublicationError,
                S9ProvenanceError,
            ),
        ):
            blockers.append(
                "Parent S9 package failed eligibility/open validation "
                f"(stale or broken references): {exc}"
            )
        else:
            blockers.append(f"Parent package validation failed: {exc}")

    # Malformed corrections already surface via journal validation + event schema.
    for event in journal.active_events():
        malformed = _malformed_event_reason(event)
        if malformed:
            blockers.append(malformed)

    report = PublishEligibilityReport(
        eligible=not blockers,
        blockers=blockers,
        warnings=warnings,
        parent_eligible=parent_eligible,
        parent_pins_match=parent_pins_match,
        details=details,
    )
    return report


def validate_journal_for_publish(journal: EventJournal) -> list[str]:
    """Structural journal / cursor consistency checks."""

    blockers: list[str] = []
    if journal.recovery_warnings:
        # Recovery already truncated/clamped — still recoverable, but surface.
        # Incomplete trailing-line recovery is allowed; non-monotonic would have raised.
        pass
    events = journal.events
    cursor = journal.cursor
    max_seq = events[-1].event_seq if events else 0
    if cursor.cursor_event_seq < 0 or cursor.cursor_event_seq > max_seq:
        blockers.append(
            f"Active undo/redo cursor is inconsistent "
            f"(cursor={cursor.cursor_event_seq}, max={max_seq})."
        )
    if cursor.redo_high_water_seq < cursor.cursor_event_seq:
        blockers.append("redo_high_water_seq is below cursor_event_seq.")
    if cursor.redo_high_water_seq > max_seq:
        blockers.append(
            f"redo_high_water_seq {cursor.redo_high_water_seq} exceeds journal max."
        )
    last = 0
    for event in events:
        if event.event_seq <= last:
            blockers.append(
                f"Journal event_seq not strictly increasing at {event.event_seq}."
            )
            break
        if last and event.event_seq != last + 1:
            blockers.append(
                f"Journal event_seq gap: expected {last + 1}, got {event.event_seq}."
            )
            break
        last = event.event_seq
        if not str(event.reviewer_id or "").strip():
            blockers.append(
                f"Event {event.event_id} is missing reviewer_id."
            )
        if not str(event.event_id or "").strip():
            blockers.append(f"Event seq {event.event_seq} is missing event_id.")
    for seq in cursor.superseded_event_seqs:
        if seq < 1 or seq > max_seq:
            blockers.append(f"Superseded event_seq {seq} outside journal range.")
    return blockers


def verify_parent_pins(
    draft: WorkingDraft,
    source: ReviewSourcePackage,
) -> dict[str, Any]:
    """Compare draft-recorded digests to live parent / S8 / video identity."""

    manifest = draft.working_manifest
    mismatches: list[str] = []

    def _check(label: str, recorded: Any, live: Any) -> None:
        rec = None if recorded in (None, "") else str(recorded)
        liv = None if live in (None, "") else str(live)
        if rec is None:
            mismatches.append(f"Draft pin missing for {label}.")
            return
        if liv is None:
            mismatches.append(f"Live parent value missing for {label}.")
            return
        if rec != liv:
            mismatches.append(
                f"Parent pin mismatch for {label}: draft={rec!r} live={liv!r} "
                "(stale parent; publish blocked)."
            )

    _check(
        "parent_package_run_id",
        manifest.get("parent_package_run_id"),
        source.package_run_id,
    )
    _check(
        "parent_package_manifest_sha256",
        manifest.get("parent_package_manifest_sha256"),
        source.package_manifest_sha256,
    )
    _check(
        "parent_sync_index_sha256",
        manifest.get("parent_sync_index_sha256"),
        source.sync_index_sha256,
    )
    _check(
        "parent_behavior_bouts_sha256",
        manifest.get("parent_behavior_bouts_sha256"),
        source.behavior_bouts_sha256,
    )
    _check(
        "source_s8_run_id",
        manifest.get("source_s8_run_id"),
        source.source_s8.run_id,
    )
    _check(
        "source_s8_states_sha256_recorded",
        manifest.get("source_s8_states_sha256_recorded"),
        source.source_s8.states_sha256_recorded,
    )
    _check(
        "source_s8_bouts_sha256_recorded",
        manifest.get("source_s8_bouts_sha256_recorded"),
        source.source_s8.bouts_sha256_recorded,
    )

    # Live re-hash of pinned S8 states path when available.
    states_path = source.source_s8.states_path
    recorded_states = manifest.get("source_s8_states_sha256_recorded")
    if states_path is not None and states_path.is_file() and recorded_states:
        live_states = sha256_file(states_path)
        if live_states != str(recorded_states):
            mismatches.append(
                "Live S8 behavior_states digest drifted from draft pin "
                f"(draft={recorded_states!r} live={live_states!r})."
            )

    recorded_video = manifest.get("source_video_sha256")
    live_video = source.source_video.sha256
    if (
        (recorded_video not in (None, "") or live_video not in (None, ""))
        and str(recorded_video or "") != str(live_video or "")
    ):
        mismatches.append(
            "Prepared video digest drifted from draft pin "
            f"(draft={recorded_video!r} live={live_video!r}); "
            "silent video substitution is forbidden."
        )

    # Also confirm S9 validate still agrees with exact package dir.
    try:
        validation = validate_s9_run_for_downstream(source.package_dir)
        currently = is_s9_run_eligible_for_downstream(source.package_dir)
    except (S9PublicationError, S9ProvenanceError) as exc:
        mismatches.append(f"Parent S9 current eligibility failed: {exc}")
        validation = {}
        currently = False

    return {
        "pins_match": not mismatches,
        "mismatches": mismatches,
        "parent_currently_eligible": currently,
        "s9_validation": validation,
    }


def _malformed_event_reason(event: CorrectionEvent) -> str | None:
    import json

    from behavioral_review.contracts import (
        DECISION_STATUS_VALUES,
        KNOWN_OPS,
        OP_CLEAR_PARALLEL_PRED,
        OP_MERGE_ADJACENT_BOUTS,
        OP_SET_BOUT_REVIEW_STATUS,
        OP_SET_EXCLUSIVE_LABEL,
        OP_SET_PARALLEL_PRED,
        OP_SPLIT_BOUT_AT_SAMPLE,
        OP_TRIM_BOUT_BOUNDARY,
        STATUS_OPS,
    )

    if event.op not in KNOWN_OPS:
        return f"Unknown correction op {event.op!r} in event {event.event_id}."

    # Direct interval writes require inclusive endpoints on the event itself.
    if event.op in (
        OP_SET_EXCLUSIVE_LABEL,
        OP_SET_PARALLEL_PRED,
        OP_CLEAR_PARALLEL_PRED,
    ):
        if event.start_sample_idx is None or event.end_sample_idx is None:
            return (
                f"Framewise event {event.event_id} missing inclusive sample interval."
            )
        if int(event.end_sample_idx) < int(event.start_sample_idx):
            return f"Framewise event {event.event_id} has end < start."

    # Compiled bout geometry ops may carry endpoints in payload_json.
    if event.op in (
        OP_TRIM_BOUT_BOUNDARY,
        OP_SPLIT_BOUT_AT_SAMPLE,
        OP_MERGE_ADJACENT_BOUTS,
    ):
        if not event.target_bout_id and not event.payload_json:
            return (
                f"Compiled geometry event {event.event_id} missing target/payload."
            )
        if event.payload_json:
            try:
                payload = json.loads(event.payload_json)
            except json.JSONDecodeError:
                return f"Compiled geometry event {event.event_id} has invalid payload_json."
            if event.op == OP_TRIM_BOUT_BOUNDARY:
                if (
                    "new_start_sample_idx" not in payload
                    or "new_end_sample_idx" not in payload
                ):
                    return (
                        f"Trim event {event.event_id} missing new inclusive endpoints."
                    )
                if int(payload["new_end_sample_idx"]) < int(
                    payload["new_start_sample_idx"]
                ):
                    return f"Trim event {event.event_id} has end < start."

    if event.op in STATUS_OPS or event.op == OP_SET_BOUT_REVIEW_STATUS:
        if event.status not in DECISION_STATUS_VALUES:
            return (
                f"Review decision event {event.event_id} has invalid status "
                f"{event.status!r}."
            )
        if not event.target_bout_id:
            return f"Review decision event {event.event_id} missing target_bout_id."
    return None
