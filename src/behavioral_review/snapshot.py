"""Deterministic publication snapshot from a working draft + parent baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.apply import replay_active_correction_chain
from behavioral_review.contracts import (
    CORRECTION_EVENTS_HISTORY_SCHEMA_VERSION,
    CORRECTION_EVENTS_SCHEMA_VERSION,
    REVIEWED_STATES_SCHEMA_VERSION,
    PublicationError,
)
from behavioral_review.events import CorrectionEvent
from behavioral_review.journal import EventJournal, JournalCursor
from behavioral_review.source import ReviewSourcePackage
from behavioral_review.status import ReviewOverlayState, replay_review_overlay


@dataclass
class PublicationSnapshot:
    """Consistent reviewed snapshot at the saved journal cursor."""

    review_run_id: str
    parent_package_run_id: str
    publication_cursor_event_seq: int
    redo_high_water_seq: int
    superseded_event_seqs: list[int]
    active_events: list[CorrectionEvent]
    history_events: list[CorrectionEvent]
    reviewed_states: pd.DataFrame
    reviewed_bouts: pd.DataFrame
    overlay: ReviewOverlayState
    changed_intervals: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    n_events_applied: int = 0
    n_events_history: int = 0
    n_superseded: int = 0
    n_undone_beyond_cursor: int = 0

    def applied_events_frame(self) -> pd.DataFrame:
        rows = [e.to_dict() for e in self.active_events]
        if not rows:
            return _empty_events_frame(CORRECTION_EVENTS_SCHEMA_VERSION)
        frame = pd.DataFrame(rows)
        frame["schema_version"] = CORRECTION_EVENTS_SCHEMA_VERSION
        frame["applied"] = True
        return frame

    def history_events_frame(self) -> pd.DataFrame:
        superseded = set(self.superseded_event_seqs)
        cursor = int(self.publication_cursor_event_seq)
        rows: list[dict[str, Any]] = []
        for event in self.history_events:
            applied = (
                event.event_seq <= cursor and event.event_seq not in superseded
            )
            payload = event.to_dict()
            payload["schema_version"] = CORRECTION_EVENTS_HISTORY_SCHEMA_VERSION
            payload["applied"] = bool(applied)
            payload["superseded"] = bool(event.event_seq in superseded)
            payload["beyond_cursor"] = bool(event.event_seq > cursor)
            rows.append(payload)
        if not rows:
            return _empty_history_frame()
        return pd.DataFrame(rows)

    def status_frame(self) -> pd.DataFrame:
        import json

        frame = self.overlay.status_frame()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "schema_version",
                    "status_row_id",
                    "status_target_kind",
                    "source_s8_bout_id",
                    "reviewed_bout_id",
                    "package_bout_id",
                    "channel",
                    "interval_key_json",
                    "status",
                    "void_reason",
                    "reviewer_id",
                    "timestamp_utc",
                    "note_ref",
                    "event_id",
                    "event_seq",
                    "active",
                    "effective_status",
                    "original_status",
                ]
            )
        out = frame.copy()
        # Provenance: original decision vs effective (voided overrides).
        out["original_status"] = out["status"]
        out["effective_status"] = out["status"]
        # Parquet cannot store heterogeneous tuple/list cells — JSON-encode.
        if "interval_key" in out.columns:
            out["interval_key_json"] = [
                (
                    json.dumps(list(v), sort_keys=False)
                    if isinstance(v, (list, tuple))
                    else (None if v is None or (isinstance(v, float) and pd.isna(v)) else json.dumps(v))
                )
                for v in out["interval_key"].tolist()
            ]
            out = out.drop(columns=["interval_key"])
        return out

    def notes_frame(self) -> pd.DataFrame:
        return self.overlay.notes_frame()


def build_publication_snapshot(
    *,
    journal: EventJournal,
    baseline_states: pd.DataFrame,
    source: ReviewSourcePackage,
    review_run_id: str,
) -> PublicationSnapshot:
    """Replay active journal events against immutable automated baseline."""

    if baseline_states is None or baseline_states.empty:
        raise PublicationError("Cannot publish without baseline S8 behavior_states.")
    cursor = journal.cursor
    _assert_cursor_consistent(journal)

    active = journal.active_events()
    history = list(journal.events)
    superseded = list(cursor.superseded_event_seqs)
    beyond = sum(1 for e in history if e.event_seq > cursor.cursor_event_seq)

    working, reviewed, changed, n_applied = _replay_active_chain(
        baseline_states=baseline_states,
        active=active,
        source=source,
        review_run_id=review_run_id,
    )
    overlay = replay_review_overlay(
        active,
        automated_bouts=source.behavior_bouts,
        reviewed_bouts=reviewed,
    )
    stamped = stamp_reviewed_states(
        working,
        review_run_id=review_run_id,
        parent_package_run_id=source.package_run_id,
        source_s8_run_id=source.source_s8.run_id,
    )
    return PublicationSnapshot(
        review_run_id=review_run_id,
        parent_package_run_id=source.package_run_id,
        publication_cursor_event_seq=int(cursor.cursor_event_seq),
        redo_high_water_seq=int(cursor.redo_high_water_seq),
        superseded_event_seqs=superseded,
        active_events=active,
        history_events=history,
        reviewed_states=stamped,
        reviewed_bouts=reviewed,
        overlay=overlay,
        changed_intervals=changed,
        n_events_applied=n_applied,
        n_events_history=len(history),
        n_superseded=len(superseded),
        n_undone_beyond_cursor=beyond,
    )


def stamp_reviewed_states(
    working_states: pd.DataFrame,
    *,
    review_run_id: str,
    parent_package_run_id: str,
    source_s8_run_id: str | None,
) -> pd.DataFrame:
    """Stamp reviewed states with publication provenance columns."""

    out = working_states.copy()
    out["schema_version"] = REVIEWED_STATES_SCHEMA_VERSION
    out["review_run_id"] = review_run_id
    out["parent_package_run_id"] = parent_package_run_id
    out["source_s8_run_id"] = source_s8_run_id
    return out


def _replay_active_chain(
    *,
    baseline_states: pd.DataFrame,
    active: list[CorrectionEvent],
    source: ReviewSourcePackage,
    review_run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[tuple[int, int]]], int]:
    return replay_active_correction_chain(
        baseline_states,
        active,
        review_run_id=review_run_id,
        parent_package_run_id=source.package_run_id,
        source_s8_run_id=source.source_s8.run_id,
        source_s7_run_id=source.source_s7_run_id,
        source_s6_run_id=source.source_s6_run_id,
        automated_bouts=source.behavior_bouts,
        sync_index=source.sync_index,
    )


def _assert_cursor_consistent(journal: EventJournal) -> None:
    cursor: JournalCursor = journal.cursor
    max_seq = journal.events[-1].event_seq if journal.events else 0
    if cursor.cursor_event_seq < 0:
        raise PublicationError("Journal cursor_event_seq is negative.")
    if cursor.cursor_event_seq > max_seq:
        raise PublicationError(
            f"Journal cursor {cursor.cursor_event_seq} exceeds journal max {max_seq}."
        )
    if cursor.redo_high_water_seq < cursor.cursor_event_seq:
        raise PublicationError(
            "Journal redo_high_water_seq is below cursor_event_seq."
        )
    if cursor.redo_high_water_seq > max_seq:
        raise PublicationError(
            f"Journal redo_high_water_seq {cursor.redo_high_water_seq} "
            f"exceeds journal max {max_seq}."
        )
    for seq in cursor.superseded_event_seqs:
        if seq < 1 or seq > max_seq:
            raise PublicationError(
                f"Superseded event_seq {seq} is outside the journal range."
            )


def _empty_events_frame(schema: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": pd.Series(dtype=object),
            "event_id": pd.Series(dtype=object),
            "event_seq": pd.Series(dtype="int64"),
            "applied": pd.Series(dtype=bool),
        }
    )


def _empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": pd.Series(dtype=object),
            "event_id": pd.Series(dtype=object),
            "event_seq": pd.Series(dtype="int64"),
            "applied": pd.Series(dtype=bool),
            "superseded": pd.Series(dtype=bool),
            "beyond_cursor": pd.Series(dtype=bool),
        }
    )
