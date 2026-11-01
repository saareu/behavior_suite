"""Append-only correction event journal with crash-safe cursor and undo/redo."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from behavioral_review.contracts import (
    CORRECTION_EVENTS_FILENAME,
    JOURNAL_CURSOR_FILENAME,
    JOURNAL_CURSOR_SCHEMA_VERSION,
    JOURNAL_RECOVERY_BACKUPS_DIRNAME,
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
    """Durable JSONL journal under a working draft directory.

    Transaction commit boundary
    ---------------------------
    ``append_events`` fsyncs JSONL lines first, then atomically replaces
    ``journal_cursor.json``. The cursor is the commit record. After a successful
    append, ``redo_high_water_seq`` equals the last committed event seq. Undo
    moves only ``cursor_event_seq`` left while leaving ``redo_high_water_seq``
    unchanged, so legitimate redo history remains ``cursor < seq <= redo_high_water``.

    Incomplete appends can leave durable JSONL lines with ``event_seq`` greater
    than the still-old ``redo_high_water_seq``. Recovery discards only that
    uncommitted tail; committed events and redo history are preserved.
    """

    working_dir: Path
    events: list[CorrectionEvent] = field(default_factory=list)
    cursor: JournalCursor = field(default_factory=JournalCursor.empty)
    recovery_warnings: list[str] = field(default_factory=list)
    last_recovery_backup_dir: Path | None = None

    @property
    def events_path(self) -> Path:
        return self.working_dir / CORRECTION_EVENTS_FILENAME

    @property
    def cursor_path(self) -> Path:
        return self.working_dir / JOURNAL_CURSOR_FILENAME

    @property
    def recovery_backups_dir(self) -> Path:
        return self.working_dir / JOURNAL_RECOVERY_BACKUPS_DIRNAME

    @classmethod
    def open(cls, working_dir: Path) -> EventJournal:
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        journal = cls(working_dir=working_dir)
        journal._load()
        return journal

    def _load(self) -> None:
        warnings: list[str] = []
        events, truncated, line_warnings = self._parse_events_file()
        warnings.extend(line_warnings)

        cursor, cursor_source, cursor_warnings = self._resolve_cursor(events)
        warnings.extend(cursor_warnings)

        # Incomplete bulk/group append: durable events may have been fsynced
        # before the atomic cursor update. Events beyond redo_high_water_seq
        # are not on an intentional undo/redo branch — discard for
        # all-or-nothing recovery (cursor redo_high_water is the commit boundary).
        committed_high = int(cursor.redo_high_water_seq)
        orphan_count = sum(1 for e in events if int(e.event_seq) > committed_high)
        discarded_orphans = False
        if orphan_count:
            # Durable primary/backup cursor establishes the commit high-water.
            # Rebuilt-from-events cursors set redo_high_water=max_seq, so this
            # path is only reached when a durable cursor is available.
            warnings.append(
                f"Discarded {orphan_count} journal event(s) beyond "
                f"redo_high_water_seq={committed_high} after interrupted append."
            )
            events = [e for e in events if int(e.event_seq) <= committed_high]
            discarded_orphans = True

        max_seq = events[-1].event_seq if events else 0
        cursor_dirty = cursor_source in {"rebuilt_from_events", "recovery_backup"}
        if cursor.cursor_event_seq > max_seq:
            warnings.append(
                f"Cursor {cursor.cursor_event_seq} beyond journal max {max_seq}; clamped."
            )
            cursor.cursor_event_seq = max_seq
            cursor_dirty = True
        if cursor.redo_high_water_seq > max_seq:
            cursor.redo_high_water_seq = max_seq
            cursor_dirty = True
        if cursor.next_event_seq != max_seq + 1:
            cursor.next_event_seq = max_seq + 1
            cursor_dirty = True
        # Drop superseded refs that no longer exist.
        pruned_superseded = [
            s for s in cursor.superseded_event_seqs if 1 <= s <= max_seq
        ]
        if pruned_superseded != list(cursor.superseded_event_seqs):
            cursor.superseded_event_seqs = pruned_superseded
            cursor_dirty = True
        else:
            cursor.superseded_event_seqs = pruned_superseded

        needs_events_rewrite = bool(truncated) or discarded_orphans
        repairing = needs_events_rewrite or (
            cursor_dirty and cursor_source != "rebuilt_from_events"
        ) or (
            cursor_dirty
            and cursor_source == "rebuilt_from_events"
            and self.events_path.is_file()
        )
        if needs_events_rewrite or cursor_dirty:
            if repairing and (self.events_path.is_file() or self.cursor_path.is_file()):
                backup = self._preserve_recovery_backup(
                    reason="; ".join(warnings) if warnings else "journal_repair"
                )
                if backup is not None:
                    self.last_recovery_backup_dir = backup
                    warnings.append(f"Recovery backup preserved at {backup.name}.")
            if needs_events_rewrite:
                self._rewrite_valid_prefix(events)
            if cursor_dirty:
                self._atomic_write_json(self.cursor_path, cursor.to_dict())

        self.events = events
        self.cursor = cursor
        self.recovery_warnings = warnings

    def _parse_events_file(
        self,
    ) -> tuple[list[CorrectionEvent], bool, list[str]]:
        warnings: list[str] = []
        events: list[CorrectionEvent] = []
        truncated = False
        if not self.events_path.is_file():
            return events, truncated, warnings

        raw_text = self.events_path.read_text(encoding="utf-8")
        # Tolerate trailing partial line after crash mid-write.
        lines = raw_text.splitlines()
        if raw_text and not raw_text.endswith("\n") and lines:
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
                raise JournalError(
                    f"Journal event_seq gap at line {i + 1}: "
                    f"expected {last_seq + 1}, got {event.event_seq}."
                )
            events.append(event)
            last_seq = event.event_seq
        return events, truncated, warnings

    def _resolve_cursor(
        self, events: list[CorrectionEvent]
    ) -> tuple[JournalCursor, str, list[str]]:
        warnings: list[str] = []
        if self.cursor_path.is_file():
            try:
                cursor_raw = json.loads(self.cursor_path.read_text(encoding="utf-8"))
                if not isinstance(cursor_raw, dict):
                    raise JournalError("journal_cursor.json must be an object.")
                return JournalCursor.from_dict(cursor_raw), "primary", warnings
            except (json.JSONDecodeError, TypeError, ValueError, JournalError) as exc:
                warnings.append(
                    f"Invalid journal_cursor.json ({exc}); trying recovery backup."
                )

        backup_cursor = self._load_latest_backup_cursor()
        if backup_cursor is not None:
            warnings.append(
                "Restored journal cursor from immutable recovery backup "
                "(primary cursor missing or invalid)."
            )
            return backup_cursor, "recovery_backup", warnings

        warnings.append(
            "journal_cursor.json missing/invalid with no usable backup; "
            "rebuilding cursor from durable events."
        )
        return self._rebuild_cursor_from_events(events), "rebuilt_from_events", warnings

    def _load_latest_backup_cursor(self) -> JournalCursor | None:
        root = self.recovery_backups_dir
        if not root.is_dir():
            return None
        candidates = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        for folder in candidates:
            path = folder / JOURNAL_CURSOR_FILENAME
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return JournalCursor.from_dict(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return None

    def _preserve_recovery_backup(self, *, reason: str) -> Path | None:
        """Copy pre-repair journal/cursor into an immutable stamped directory."""

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        dest = self.recovery_backups_dir / f"recovery__{stamp}"
        # Avoid clobbering an existing stamp (immutable backups).
        suffix = 0
        while dest.exists():
            suffix += 1
            dest = self.recovery_backups_dir / f"recovery__{stamp}_{suffix}"
        dest.mkdir(parents=True, exist_ok=False)
        if self.events_path.is_file():
            shutil.copy2(self.events_path, dest / CORRECTION_EVENTS_FILENAME)
        if self.cursor_path.is_file():
            shutil.copy2(self.cursor_path, dest / JOURNAL_CURSOR_FILENAME)
        (dest / "recovery_reason.txt").write_text(
            reason.strip() + "\n", encoding="utf-8"
        )
        return dest

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
