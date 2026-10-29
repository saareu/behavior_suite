"""Append-only correction event journal with crash-safe cursor and undo/redo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from behavioral_review.contracts import (
    CORRECTION_EVENTS_FILENAME,
    JOURNAL_CURSOR_FILENAME,
    JOURNAL_CURSOR_SCHEMA_VERSION,
    JournalError,
)
from behavioral_review.events import CorrectionEvent


@dataclass
class JournalCursor:
    """Persisted undo/redo cursor over the durable journal."""

    schema_version: str
    cursor_event_seq: int
    redo_high_water_seq: int
    next_event_seq: int
    superseded_event_seqs: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cursor_event_seq": int(self.cursor_event_seq),
            "redo_high_water_seq": int(self.redo_high_water_seq),
            "next_event_seq": int(self.next_event_seq),
            "superseded_event_seqs": sorted(set(int(x) for x in self.superseded_event_seqs)),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> JournalCursor:
        return cls(
            schema_version=str(raw.get("schema_version") or JOURNAL_CURSOR_SCHEMA_VERSION),
            cursor_event_seq=int(raw.get("cursor_event_seq") or 0),
            redo_high_water_seq=int(raw.get("redo_high_water_seq") or 0),
            next_event_seq=int(raw.get("next_event_seq") or 1),
            superseded_event_seqs=[
                int(x) for x in (raw.get("superseded_event_seqs") or [])
            ],
        )

    @classmethod
    def empty(cls) -> JournalCursor:
        return cls(
            schema_version=JOURNAL_CURSOR_SCHEMA_VERSION,
            cursor_event_seq=0,
            redo_high_water_seq=0,
            next_event_seq=1,
            superseded_event_seqs=[],
        )


@dataclass
class EventJournal:
    """Durable JSONL journal under a working draft directory."""

    working_dir: Path
    events: list[CorrectionEvent] = field(default_factory=list)
    cursor: JournalCursor = field(default_factory=JournalCursor.empty)
    recovery_warnings: list[str] = field(default_factory=list)

    @property
    def events_path(self) -> Path:
        return self.working_dir / CORRECTION_EVENTS_FILENAME

    @property
    def cursor_path(self) -> Path:
        return self.working_dir / JOURNAL_CURSOR_FILENAME

    @classmethod
    def open(cls, working_dir: Path) -> EventJournal:
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        journal = cls(working_dir=working_dir)
        journal._load()
        return journal

    def _load(self) -> None:
        warnings: list[str] = []
        events: list[CorrectionEvent] = []
        if self.events_path.is_file():
            raw_text = self.events_path.read_text(encoding="utf-8")
            # Tolerate trailing partial line after crash mid-write.
            lines = raw_text.splitlines()
            truncated = False
            if raw_text and not raw_text.endswith("\n") and lines:
                # Last line may be incomplete — drop it after validation attempt.
                candidate = lines[-1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    lines = lines[:-1]
                    truncated = True
                    warnings.append(
                        "Truncated incomplete trailing journal line during recovery."
                    )
            last_seq = 0
            for i, line in enumerate(lines):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    if not isinstance(payload, dict):
                        raise JournalError("Journal line is not an object.")
                    event = CorrectionEvent.from_dict(payload)
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                    warnings.append(
                        f"Stopped journal replay at line {i + 1}: {exc}. "
                        "Trailing corrupt events discarded for recovery."
                    )
                    truncated = True
                    break
                if event.event_seq <= last_seq:
                    raise JournalError(
                        f"Non-monotonic event_seq in journal at line {i + 1} "
                        f"({event.event_seq} after {last_seq})."
                    )
                if last_seq and event.event_seq != last_seq + 1:
                    # Gaps after truncation are ok only if we already stopped; else fail.
                    raise JournalError(
                        f"Journal event_seq gap at line {i + 1}: "
                        f"expected {last_seq + 1}, got {event.event_seq}."
                    )
                events.append(event)
                last_seq = event.event_seq
            if truncated:
                # Rewrite truncated journal to last valid line (crash recovery).
                self._rewrite_valid_prefix(events)
        self.events = events

        if self.cursor_path.is_file():
            try:
                cursor_raw = json.loads(self.cursor_path.read_text(encoding="utf-8"))
                if not isinstance(cursor_raw, dict):
                    raise JournalError("journal_cursor.json must be an object.")
                cursor = JournalCursor.from_dict(cursor_raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                warnings.append(f"Invalid journal_cursor.json; rebuilding ({exc}).")
                cursor = self._rebuild_cursor_from_events(events)
                self._atomic_write_json(self.cursor_path, cursor.to_dict())
        else:
            cursor = self._rebuild_cursor_from_events(events)
            self._atomic_write_json(self.cursor_path, cursor.to_dict())

        max_seq = events[-1].event_seq if events else 0
        if cursor.cursor_event_seq > max_seq:
            warnings.append(
                f"Cursor {cursor.cursor_event_seq} beyond journal max {max_seq}; clamped."
            )
            cursor.cursor_event_seq = max_seq
        if cursor.redo_high_water_seq > max_seq:
            cursor.redo_high_water_seq = max_seq
        if cursor.next_event_seq <= max_seq:
            cursor.next_event_seq = max_seq + 1
        # Drop superseded refs that no longer exist.
        cursor.superseded_event_seqs = [
            s for s in cursor.superseded_event_seqs if 1 <= s <= max_seq
        ]
        self.cursor = cursor
        self.recovery_warnings = warnings
        if warnings:
            self._atomic_write_json(self.cursor_path, cursor.to_dict())

    @staticmethod
    def _rebuild_cursor_from_events(events: list[CorrectionEvent]) -> JournalCursor:
        max_seq = events[-1].event_seq if events else 0
        return JournalCursor(
            schema_version=JOURNAL_CURSOR_SCHEMA_VERSION,
            cursor_event_seq=max_seq,
            redo_high_water_seq=max_seq,
            next_event_seq=max_seq + 1,
            superseded_event_seqs=[],
        )

    def _rewrite_valid_prefix(self, events: list[CorrectionEvent]) -> None:
        tmp = self.events_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for event in events:
                handle.write(event.to_json_line())
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.events_path)

    def active_events(self) -> list[CorrectionEvent]:
        """Events in the active undo branch up to the cursor (stable ids)."""

        superseded = set(self.cursor.superseded_event_seqs)
        out: list[CorrectionEvent] = []
        for event in self.events:
            if event.event_seq > self.cursor.cursor_event_seq:
                break
            if event.event_seq in superseded:
                continue
            out.append(event)
        return out

    def event_is_active(self, event_seq: int) -> bool:
        if event_seq < 1 or event_seq > self.cursor.cursor_event_seq:
            return False
        return event_seq not in set(self.cursor.superseded_event_seqs)

    def can_undo(self) -> bool:
        return self.cursor.cursor_event_seq > 0 and bool(self.active_events())

    def can_redo(self) -> bool:
        return self.cursor.cursor_event_seq < self.cursor.redo_high_water_seq

    def append_events(self, events: list[CorrectionEvent]) -> JournalCursor:
        """Append events after truncating the logical redo branch if needed."""

        if not events:
            raise JournalError("append_events requires at least one event.")
        # New edit after undo: supersede redo side without rewriting history.
        if self.cursor.cursor_event_seq < self.cursor.redo_high_water_seq:
            for seq in range(
                self.cursor.cursor_event_seq + 1,
                self.cursor.redo_high_water_seq + 1,
            ):
                if seq not in self.cursor.superseded_event_seqs:
                    self.cursor.superseded_event_seqs.append(seq)

        expected = int(self.cursor.next_event_seq)
        for event in events:
            if int(event.event_seq) != expected:
                raise JournalError(
                    f"Event seq mismatch: expected {expected}, got {event.event_seq}."
                )
            expected += 1

        # Durable append: write all lines, fsync, then update cursor via temp→rename.
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            for event in events:
                handle.write(event.to_json_line())
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        self.events.extend(events)
        last_seq = events[-1].event_seq
        self.cursor.cursor_event_seq = last_seq
        self.cursor.redo_high_water_seq = last_seq
        self.cursor.next_event_seq = last_seq + 1
        self._atomic_write_json(self.cursor_path, self.cursor.to_dict())
        return self.cursor

    def undo_group(self) -> list[CorrectionEvent]:
        """Move cursor left by one undo_group_id; return undone events."""

        active = self.active_events()
        if not active:
            raise JournalError("Nothing to undo.")
        group_id = active[-1].undo_group_id
        undone: list[CorrectionEvent] = []
        idx = len(active) - 1
        while idx >= 0 and active[idx].undo_group_id == group_id:
            undone.append(active[idx])
            idx -= 1
        undone.reverse()
        new_cursor_seq = active[idx].event_seq if idx >= 0 else 0
        self.cursor.cursor_event_seq = new_cursor_seq
        # redo_high_water unchanged so redo remains possible.
        self._atomic_write_json(self.cursor_path, self.cursor.to_dict())
        return undone

    def redo_group(self) -> list[CorrectionEvent]:
        """Move cursor right by one undo_group_id within redo_high_water."""

        if not self.can_redo():
            raise JournalError("Nothing to redo.")
        superseded = set(self.cursor.superseded_event_seqs)
        candidates = [
            e
            for e in self.events
            if self.cursor.cursor_event_seq
            < e.event_seq
            <= self.cursor.redo_high_water_seq
            and e.event_seq not in superseded
        ]
        if not candidates:
            raise JournalError("Nothing to redo.")
        group_id = candidates[0].undo_group_id
        redone: list[CorrectionEvent] = []
        for event in candidates:
            if event.undo_group_id != group_id:
                break
            redone.append(event)
        self.cursor.cursor_event_seq = redone[-1].event_seq
        self._atomic_write_json(self.cursor_path, self.cursor.to_dict())
        return redone

    def allocate_seq_block(self, count: int) -> list[int]:
        start = int(self.cursor.next_event_seq)
        return list(range(start, start + int(count)))

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
