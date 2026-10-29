"""Independent bout review-status and notes overlays (status-only reject)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    BOUT_ID_KIND_PACKAGE,
    BOUT_ID_KIND_REVIEWED,
    BOUT_ID_KIND_SOURCE_S8,
    BOUT_STATUS_SCHEMA_VERSION,
    FRAMEWISE_OPS,
    GEOMETRY_CHANGING_OPS,
    OP_ADD_NOTE,
    OP_DELETE_NOTE,
    OP_EDIT_NOTE,
    OP_SET_BOUT_REVIEW_STATUS,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_TARGET_REVIEWED_BOUT,
    STATUS_TARGET_SOURCE_BOUT,
    STATUS_UNREVIEWED,
    STATUS_VOIDED,
    VOID_REASON_GEOMETRY_CHANGED,
)
from behavioral_review.events import CorrectionEvent
from behavioral_review.regenerate import interval_key_from_row


@dataclass
class StatusRecord:
    status_row_id: str
    status_target_kind: str
    source_s8_bout_id: str | None
    reviewed_bout_id: str | None
    package_bout_id: str | None
    channel: str | None
    interval_key: tuple[Any, ...] | None
    status: str
    void_reason: str | None
    reviewer_id: str
    timestamp_utc: str
    note_ref: str | None
    event_id: str
    event_seq: int
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BOUT_STATUS_SCHEMA_VERSION,
            "status_row_id": self.status_row_id,
            "status_target_kind": self.status_target_kind,
            "source_s8_bout_id": self.source_s8_bout_id,
            "reviewed_bout_id": self.reviewed_bout_id,
            "package_bout_id": self.package_bout_id,
            "channel": self.channel,
            "interval_key": list(self.interval_key) if self.interval_key else None,
            "status": self.status,
            "void_reason": self.void_reason,
            "reviewer_id": self.reviewer_id,
            "timestamp_utc": self.timestamp_utc,
            "note_ref": self.note_ref,
            "event_id": self.event_id,
            "event_seq": self.event_seq,
            "active": self.active,
        }


@dataclass
class NoteRecord:
    note_id: str
    target_bout_id: str | None
    target_bout_id_kind: str | None
    note_text: str
    reviewer_id: str
    timestamp_utc: str
    event_id: str
    event_seq: int
    deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "target_bout_id": self.target_bout_id,
            "target_bout_id_kind": self.target_bout_id_kind,
            "note_text": self.note_text,
            "reviewer_id": self.reviewer_id,
            "timestamp_utc": self.timestamp_utc,
            "event_id": self.event_id,
            "event_seq": self.event_seq,
            "deleted": self.deleted,
        }


@dataclass
class ReviewOverlayState:
    """Derived status/notes from active journal events (labels untouched)."""

    history: list[StatusRecord] = field(default_factory=list)
    current_by_target: dict[str, StatusRecord] = field(default_factory=dict)
    notes: dict[str, NoteRecord] = field(default_factory=dict)

    def status_for_source_bout(self, source_s8_bout_id: str) -> str:
        key = f"source:{source_s8_bout_id}"
        rec = self.current_by_target.get(key)
        if rec is None or not rec.active:
            return STATUS_UNREVIEWED
        return rec.status

    def status_frame(self) -> pd.DataFrame:
        rows = [r.to_dict() for r in self.history]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def notes_frame(self) -> pd.DataFrame:
        rows = [n.to_dict() for n in self.notes.values() if not n.deleted]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


def replay_review_overlay(
    events: list[CorrectionEvent],
    *,
    automated_bouts: pd.DataFrame | None = None,
    reviewed_bouts: pd.DataFrame | None = None,
) -> ReviewOverlayState:
    """Replay status/note events and void decisions after geometry changes."""

    state = ReviewOverlayState()
    auto_by_source = _index_bouts(automated_bouts, prefer_source=True)
    # Track last known interval keys for status targets.
    attached_intervals: dict[str, tuple[Any, ...]] = {}

    for event in events:
        if event.op == OP_SET_BOUT_REVIEW_STATUS:
            rec = _status_from_event(event, automated_bouts, reviewed_bouts)
            target_key = _target_key(rec)
            state.history.append(rec)
            state.current_by_target[target_key] = rec
            if rec.interval_key is not None:
                attached_intervals[target_key] = rec.interval_key
            elif rec.source_s8_bout_id and rec.source_s8_bout_id in auto_by_source:
                attached_intervals[target_key] = interval_key_from_row(
                    auto_by_source[rec.source_s8_bout_id]
                )
                rec.interval_key = attached_intervals[target_key]
        elif event.op == OP_ADD_NOTE:
            note_id = event.note_id or f"note_{event.event_id}"
            state.notes[note_id] = NoteRecord(
                note_id=note_id,
                target_bout_id=event.target_bout_id,
                target_bout_id_kind=event.target_bout_id_kind,
                note_text=str(event.note_text or ""),
                reviewer_id=event.reviewer_id,
                timestamp_utc=event.timestamp_utc,
                event_id=event.event_id,
                event_seq=event.event_seq,
            )
        elif event.op == OP_EDIT_NOTE:
            note_id = str(event.note_id)
            existing = state.notes.get(note_id)
            if existing is not None and not existing.deleted:
                existing.note_text = str(event.note_text or existing.note_text)
                existing.timestamp_utc = event.timestamp_utc
                existing.event_id = event.event_id
                existing.event_seq = event.event_seq
        elif event.op == OP_DELETE_NOTE:
            note_id = str(event.note_id)
            existing = state.notes.get(note_id)
            if existing is not None:
                existing.deleted = True
                existing.timestamp_utc = event.timestamp_utc
                existing.event_id = event.event_id
                existing.event_seq = event.event_seq
        elif event.op in GEOMETRY_CHANGING_OPS:
            _void_affected_statuses(
                state,
                event=event,
                attached_intervals=attached_intervals,
                automated_bouts=automated_bouts,
                reviewed_bouts=reviewed_bouts,
            )

    return state


def _void_affected_statuses(
    state: ReviewOverlayState,
    *,
    event: CorrectionEvent,
    attached_intervals: dict[str, tuple[Any, ...]],
    automated_bouts: pd.DataFrame | None,
    reviewed_bouts: pd.DataFrame | None,
) -> None:
    """Void accepted/rejected decisions whose interval identity was edited."""

    touched = _event_touch_mask(event)
    for target_key, rec in list(state.current_by_target.items()):
        if rec.status not in (STATUS_ACCEPTED, STATUS_REJECTED):
            continue
        if not rec.active:
            continue
        interval = attached_intervals.get(target_key) or rec.interval_key
        if interval is None:
            # Resolve from bout tables if possible.
            if rec.source_s8_bout_id and automated_bouts is not None:
                idx = _index_bouts(automated_bouts, prefer_source=True)
                if rec.source_s8_bout_id in idx:
                    interval = interval_key_from_row(idx[rec.source_s8_bout_id])
            elif rec.reviewed_bout_id and reviewed_bouts is not None:
                idx = _index_bouts(reviewed_bouts, prefer_source=False)
                if rec.reviewed_bout_id in idx:
                    interval = interval_key_from_row(idx[rec.reviewed_bout_id])
        if interval is None:
            continue
        if not _interval_overlaps_touch(interval, touched, event=event):
            continue
        voided = StatusRecord(
            status_row_id=f"void_{event.event_id}_{rec.status_row_id}",
            status_target_kind=rec.status_target_kind,
            source_s8_bout_id=rec.source_s8_bout_id,
            reviewed_bout_id=rec.reviewed_bout_id,
            package_bout_id=rec.package_bout_id,
            channel=rec.channel,
            interval_key=interval,
            status=STATUS_VOIDED,
            void_reason=VOID_REASON_GEOMETRY_CHANGED,
            reviewer_id=event.reviewer_id,
            timestamp_utc=event.timestamp_utc,
            note_ref=rec.note_ref,
            event_id=event.event_id,
            event_seq=event.event_seq,
            active=True,
        )
        # Prior decision retained in history; current becomes voided.
        rec.active = False
        state.history.append(voided)
        state.current_by_target[target_key] = voided


def _event_touch_mask(event: CorrectionEvent) -> dict[str, Any]:
    payload = {}
    if event.payload_json:
        try:
            raw = json.loads(event.payload_json)
            if isinstance(raw, dict):
                payload = raw
        except json.JSONDecodeError:
            payload = {}
    start = event.start_sample_idx
    end = event.end_sample_idx
    if start is None and "new_start_sample_idx" in payload:
        start = int(payload["new_start_sample_idx"])
        end = int(payload.get("new_end_sample_idx", start))
    if start is None and "at_sample_idx" in payload:
        start = int(payload["at_sample_idx"])
        end = start
    return {
        "channel": event.channel,
        "focal_id": event.focal_id,
        "partner_id": event.partner_id,
        "mouse_a_id": event.mouse_a_id,
        "mouse_b_id": event.mouse_b_id,
        "start": start,
        "end": end,
        "target_bout_id": event.target_bout_id,
        "payload": payload,
    }


def _interval_overlaps_touch(
    interval: tuple[Any, ...],
    touch: dict[str, Any],
    *,
    event: CorrectionEvent,
) -> bool:
    # interval: channel, grain, focal, partner, mouse_a, mouse_b, behavior, start, end
    channel, _grain, focal, partner, mouse_a, mouse_b, _beh, i_start, i_end = interval
    if (
        touch["target_bout_id"]
        and event.op in FRAMEWISE_OPS
        and event.target_bout_id
    ):
        # Bout-scoped compile ops always affect their target identity.
        return True
    if (
        touch["channel"] is not None
        and str(touch["channel"]) != str(channel)
        and event.channel is not None
    ):
        # Parallel vs exclusive isolation.
        return False
    if (
        touch["focal_id"] is not None
        and touch["partner_id"] is not None
        and focal is not None
        and partner is not None
        and (
            int(focal) != int(touch["focal_id"])
            or int(partner) != int(touch["partner_id"])
        )
    ):
        return False
    if (
        touch["mouse_a_id"] is not None
        and touch["mouse_b_id"] is not None
        and mouse_a is not None
        and mouse_b is not None
        and (
            int(mouse_a) != int(touch["mouse_a_id"])
            or int(mouse_b) != int(touch["mouse_b_id"])
        )
    ):
        return False
    if touch["start"] is None or touch["end"] is None:
        # Bout merge may only have payload ids — treat as affecting both.
        payload = touch.get("payload") or {}
        if "bout_id_a" in payload or "bout_id_b" in payload:
            return True
        return bool(event.target_bout_id)
    t0, t1 = int(touch["start"]), int(touch["end"])
    return not (int(i_end) < t0 or int(i_start) > t1)


def _status_from_event(
    event: CorrectionEvent,
    automated_bouts: pd.DataFrame | None,
    reviewed_bouts: pd.DataFrame | None,
) -> StatusRecord:
    kind = event.target_bout_id_kind or BOUT_ID_KIND_SOURCE_S8
    source_id = None
    reviewed_id = None
    package_id = None
    channel = event.channel
    interval = None
    if kind == BOUT_ID_KIND_SOURCE_S8:
        source_id = event.target_bout_id
        status_kind = STATUS_TARGET_SOURCE_BOUT
        if automated_bouts is not None:
            idx = _index_bouts(automated_bouts, prefer_source=True)
            if source_id in idx:
                row = idx[source_id]
                channel = str(row["channel"])
                interval = interval_key_from_row(row)
                if "bout_id" in row:
                    package_id = str(row["bout_id"])
    elif kind == BOUT_ID_KIND_REVIEWED:
        reviewed_id = event.target_bout_id
        status_kind = STATUS_TARGET_REVIEWED_BOUT
        if reviewed_bouts is not None:
            idx = _index_bouts(reviewed_bouts, prefer_source=False)
            if reviewed_id in idx:
                row = idx[reviewed_id]
                channel = str(row["channel"])
                interval = interval_key_from_row(row)
                if "source_s8_bout_id" in row and pd.notna(row["source_s8_bout_id"]):
                    source_id = str(row["source_s8_bout_id"])
    elif kind == BOUT_ID_KIND_PACKAGE:
        package_id = event.target_bout_id
        status_kind = STATUS_TARGET_SOURCE_BOUT
        if automated_bouts is not None and "bout_id" in automated_bouts.columns:
            hits = automated_bouts[
                automated_bouts["bout_id"].astype(str) == str(package_id)
            ]
            if len(hits) == 1:
                row = hits.iloc[0]
                channel = str(row["channel"])
                interval = interval_key_from_row(row)
                if "source_s8_bout_id" in row and pd.notna(row["source_s8_bout_id"]):
                    source_id = str(row["source_s8_bout_id"])
    else:
        status_kind = STATUS_TARGET_SOURCE_BOUT
        source_id = event.target_bout_id

    return StatusRecord(
        status_row_id=f"status_{event.event_id}",
        status_target_kind=status_kind,
        source_s8_bout_id=source_id,
        reviewed_bout_id=reviewed_id,
        package_bout_id=package_id,
        channel=channel,
        interval_key=interval,
        status=str(event.status),
        void_reason=event.void_reason,
        reviewer_id=event.reviewer_id,
        timestamp_utc=event.timestamp_utc,
        note_ref=None,
        event_id=event.event_id,
        event_seq=event.event_seq,
        active=True,
    )


def _target_key(rec: StatusRecord) -> str:
    if rec.source_s8_bout_id:
        return f"source:{rec.source_s8_bout_id}"
    if rec.reviewed_bout_id:
        return f"reviewed:{rec.reviewed_bout_id}"
    if rec.package_bout_id:
        return f"package:{rec.package_bout_id}"
    return f"row:{rec.status_row_id}"


def _index_bouts(
    bouts: pd.DataFrame | None, *, prefer_source: bool
) -> dict[str, pd.Series]:
    if bouts is None or bouts.empty:
        return {}
    out: dict[str, pd.Series] = {}
    for _, row in bouts.iterrows():
        if prefer_source and "source_s8_bout_id" in row and pd.notna(row["source_s8_bout_id"]):
            out[str(row["source_s8_bout_id"])] = row
        elif "bout_id" in row and pd.notna(row["bout_id"]):
            out[str(row["bout_id"])] = row
    return out
