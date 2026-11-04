"""Parse authoritative S3 correction journals for QC overlay indicators.

Uses only structured journal fields. Does not invent suspicious-frame markers,
does not read ``was_track_swapped`` as a per-node event, and does not parse
node names from free-text ``action`` strings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

AttributionKind = Literal["track_node", "track", "unlocalized"]
JournalOrigin = Literal["machine", "manual"]
JournalStatus = Literal["available", "absent", "empty", "malformed"]


@dataclass(frozen=True, slots=True)
class CorrectionEvent:
    """One documented automatic or manual correction for overlay indication."""

    origin: JournalOrigin
    action_type: str
    start_frame: int
    end_frame: int
    track: int | None
    node: str | None
    attribution: AttributionKind

    def covers(self, prepared_frame_idx: int) -> bool:
        idx = int(prepared_frame_idx)
        return int(self.start_frame) <= idx <= int(self.end_frame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "action_type": self.action_type,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "track": self.track,
            "node": self.node,
            "attribution": self.attribution,
        }


@dataclass(frozen=True, slots=True)
class CorrectionJournalIndex:
    """Frame-indexed correction events with honest availability disclosure."""

    origin: JournalOrigin
    status: JournalStatus
    path: Path | None
    events: tuple[CorrectionEvent, ...]
    disclosure: str
    digest_sha256: str | None = None

    def events_for_frame(self, prepared_frame_idx: int) -> tuple[CorrectionEvent, ...]:
        if self.status not in {"available", "empty"}:
            return ()
        idx = int(prepared_frame_idx)
        return tuple(event for event in self.events if event.covers(idx))

    def windowed(
        self, prepared_frame_range: tuple[int, int] | None
    ) -> CorrectionJournalIndex:
        if prepared_frame_range is None or self.status not in {"available", "empty"}:
            return self
        lo, hi = int(prepared_frame_range[0]), int(prepared_frame_range[1])
        kept = tuple(
            event
            for event in self.events
            if int(event.end_frame) >= lo and int(event.start_frame) <= hi
        )
        return CorrectionJournalIndex(
            origin=self.origin,
            status=self.status if kept or self.status == "empty" else self.status,
            path=self.path,
            events=kept,
            disclosure=self.disclosure,
            digest_sha256=self.digest_sha256,
        )

    def by_frame_dict(
        self, prepared_frame_range: tuple[int, int] | None = None
    ) -> dict[int, list[dict[str, Any]]]:
        index = self.windowed(prepared_frame_range)
        out: dict[int, list[dict[str, Any]]] = {}
        for event in index.events:
            for frame_idx in range(int(event.start_frame), int(event.end_frame) + 1):
                if prepared_frame_range is not None:
                    lo, hi = int(prepared_frame_range[0]), int(prepared_frame_range[1])
                    if frame_idx < lo or frame_idx > hi:
                        continue
                out.setdefault(frame_idx, []).append(event.to_dict())
        return out

    def to_disclosure(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "status": self.status,
            "path": None if self.path is None else str(self.path),
            "event_count": len(self.events),
            "digest_sha256": self.digest_sha256,
            "disclosure": self.disclosure,
            "attribution_rule": (
                "track_node only when structured track+node recorded; "
                "track when only structured track recorded; otherwise unlocalized. "
                "Never parse node from free-text action. Never invent events from "
                "was_track_swapped."
            ),
        }


def load_machine_correction_index(
    path: Path | str | None,
    *,
    digest_sha256: str | None = None,
) -> CorrectionJournalIndex:
    """Load ``machine_corrections.json`` (list of per-frame automatic events)."""

    if path is None:
        return CorrectionJournalIndex(
            origin="machine",
            status="absent",
            path=None,
            events=(),
            disclosure=(
                "machine_corrections.json path not provided; automatic correction "
                "indicators unavailable."
            ),
            digest_sha256=None,
        )
    journal_path = Path(path).expanduser().resolve()
    if not journal_path.is_file():
        return CorrectionJournalIndex(
            origin="machine",
            status="absent",
            path=journal_path,
            events=(),
            disclosure=(
                f"machine_corrections.json missing at {journal_path}; automatic "
                "correction indicators unavailable (not fabricated)."
            ),
            digest_sha256=None,
        )
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CorrectionJournalIndex(
            origin="machine",
            status="malformed",
            path=journal_path,
            events=(),
            disclosure=(
                f"machine_corrections.json unreadable/malformed at {journal_path}: "
                f"{exc}. Automatic correction indicators unavailable."
            ),
            digest_sha256=digest_sha256,
        )
    if not isinstance(payload, list):
        return CorrectionJournalIndex(
            origin="machine",
            status="malformed",
            path=journal_path,
            events=(),
            disclosure=(
                f"machine_corrections.json must be a list at {journal_path}; "
                "automatic correction indicators unavailable."
            ),
            digest_sha256=digest_sha256,
        )

    events: list[CorrectionEvent] = []
    for item in payload:
        parsed = _parse_machine_item(item)
        if parsed is not None:
            events.append(parsed)
    if not events:
        return CorrectionJournalIndex(
            origin="machine",
            status="empty",
            path=journal_path,
            events=(),
            disclosure=(
                "machine_corrections.json present but contains no usable frame "
                "events; no automatic indicators drawn."
            ),
            digest_sha256=digest_sha256,
        )
    return CorrectionJournalIndex(
        origin="machine",
        status="available",
        path=journal_path,
        events=tuple(events),
        disclosure=(
            f"Loaded {len(events)} automatic correction journal entries from "
            "machine_corrections.json."
        ),
        digest_sha256=digest_sha256,
    )


def load_manual_correction_index(
    path: Path | str | None,
    *,
    digest_sha256: str | None = None,
) -> CorrectionJournalIndex:
    """Load ``manual_corrections.json`` (list of inclusive-interval edit records)."""

    if path is None:
        return CorrectionJournalIndex(
            origin="manual",
            status="absent",
            path=None,
            events=(),
            disclosure=(
                "manual_corrections.json path not provided; manual correction "
                "indicators unavailable."
            ),
            digest_sha256=None,
        )
    journal_path = Path(path).expanduser().resolve()
    if not journal_path.is_file():
        return CorrectionJournalIndex(
            origin="manual",
            status="absent",
            path=journal_path,
            events=(),
            disclosure=(
                f"manual_corrections.json missing at {journal_path}; manual "
                "correction indicators unavailable (not fabricated)."
            ),
            digest_sha256=None,
        )
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CorrectionJournalIndex(
            origin="manual",
            status="malformed",
            path=journal_path,
            events=(),
            disclosure=(
                f"manual_corrections.json unreadable/malformed at {journal_path}: "
                f"{exc}. Manual correction indicators unavailable."
            ),
            digest_sha256=digest_sha256,
        )
    if not isinstance(payload, list):
        return CorrectionJournalIndex(
            origin="manual",
            status="malformed",
            path=journal_path,
            events=(),
            disclosure=(
                f"manual_corrections.json must be a list at {journal_path}; "
                "manual correction indicators unavailable."
            ),
            digest_sha256=digest_sha256,
        )

    events: list[CorrectionEvent] = []
    for item in payload:
        parsed = _parse_manual_item(item)
        if parsed is not None:
            events.append(parsed)
    if not events:
        return CorrectionJournalIndex(
            origin="manual",
            status="empty",
            path=journal_path,
            events=(),
            disclosure=(
                "manual_corrections.json present but contains no usable edit "
                "records; no manual indicators drawn."
            ),
            digest_sha256=digest_sha256,
        )
    return CorrectionJournalIndex(
        origin="manual",
        status="available",
        path=journal_path,
        events=tuple(events),
        disclosure=(
            f"Loaded {len(events)} manual correction journal entries from "
            "manual_corrections.json."
        ),
        digest_sha256=digest_sha256,
    )


def _parse_machine_item(item: object) -> CorrectionEvent | None:
    if not isinstance(item, Mapping):
        return None
    frame = _optional_int(item.get("frame"))
    if frame is None or frame < 0:
        return None
    action_type = str(item.get("type") or "unknown").strip() or "unknown"
    track = _optional_int(item.get("track"))
    # Structured node only — never scrape free-text action strings.
    node = _optional_node(item.get("node"))
    attribution = _attribution(track=track, node=node)
    return CorrectionEvent(
        origin="machine",
        action_type=action_type,
        start_frame=frame,
        end_frame=frame,
        track=track,
        node=node if attribution == "track_node" else None,
        attribution=attribution,
    )


def _parse_manual_item(item: object) -> CorrectionEvent | None:
    if not isinstance(item, Mapping):
        return None
    start = _optional_int(item.get("start_frame"))
    end = _optional_int(item.get("end_frame"))
    if start is None or end is None or start < 0 or end < start:
        return None
    action_type = str(item.get("action") or "manual_edit").strip() or "manual_edit"
    track = _optional_int(item.get("track"))
    if track is None:
        tracks = item.get("tracks")
        if isinstance(tracks, Sequence) and not isinstance(tracks, (str, bytes)):
            parsed_tracks = [
                value
                for value in (_optional_int(raw) for raw in tracks)
                if value is not None
            ]
            # Single-track lists can attribute; multi-track identity swaps stay
            # unlocalized unless a dedicated track field was recorded.
            if len(parsed_tracks) == 1:
                track = parsed_tracks[0]
    node = _optional_node(item.get("node"))
    attribution = _attribution(track=track, node=node)
    return CorrectionEvent(
        origin="manual",
        action_type=action_type,
        start_frame=start,
        end_frame=end,
        track=track,
        node=node if attribution == "track_node" else None,
        attribution=attribution,
    )


def _attribution(*, track: int | None, node: str | None) -> AttributionKind:
    if track is not None and node is not None:
        return "track_node"
    if track is not None:
        return "track"
    return "unlocalized"


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    return None


def _optional_node(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() == "nan":
        return None
    return text
