"""Correction / review event schema helpers (behavioral_review_correction_events_v1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from behavioral_review.contracts import (
    BOUT_ID_KIND_INTERVAL,
    BOUT_ID_KIND_PACKAGE,
    BOUT_ID_KIND_REVIEWED,
    BOUT_ID_KIND_SOURCE_S8,
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    CORRECTION_EVENTS_SCHEMA_VERSION,
    DECISION_STATUS_VALUES,
    EXCLUSIVE_ETHOGRAM_LABELS,
    IDENTITY_DIRECTIONAL,
    IDENTITY_SYMMETRIC_PAIR,
    KNOWN_OPS,
    OP_ADD_NOTE,
    OP_CLEAR_PARALLEL_PRED,
    OP_DELETE_NOTE,
    OP_EDIT_NOTE,
    OP_MERGE_ADJACENT_BOUTS,
    OP_SESSION_MARKER,
    OP_SET_BOUT_REVIEW_STATUS,
    OP_SET_EXCLUSIVE_LABEL,
    OP_SET_PARALLEL_PRED,
    OP_SPLIT_BOUT_AT_SAMPLE,
    OP_TRIM_BOUT_BOUNDARY,
    EditingError,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def new_event_id() -> str:
    return f"evt_{uuid4().hex}"


def new_undo_group_id() -> str:
    return f"ug_{uuid4().hex}"


def new_client_op_id() -> str:
    return f"cop_{uuid4().hex}"


def pair_id_for(mouse_a: int, mouse_b: int) -> str:
    a, b = (int(mouse_a), int(mouse_b)) if mouse_a <= mouse_b else (int(mouse_b), int(mouse_a))
    return f"{a}|{b}"


@dataclass(slots=True)
class CorrectionEvent:
    """One append-only journal event (framewise, status, note, or marker)."""

    schema_version: str
    event_id: str
    event_seq: int
    timestamp_utc: str
    reviewer_id: str
    op: str
    channel: str | None = None
    identity_grain: str | None = None
    focal_id: int | None = None
    partner_id: int | None = None
    mouse_a_id: int | None = None
    mouse_b_id: int | None = None
    pair_id: str | None = None
    start_sample_idx: int | None = None
    end_sample_idx: int | None = None
    label: str | None = None
    pred_value: int | None = None
    target_bout_id: str | None = None
    target_bout_id_kind: str | None = None
    note_text: str | None = None
    tags_json: str | None = None
    reciprocal: bool = False
    client_op_id: str | None = None
    undo_group_id: str | None = None
    payload_json: str | None = None
    status: str | None = None
    note_id: str | None = None
    void_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CorrectionEvent:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: raw.get(k) for k in known}
        payload["schema_version"] = str(
            payload.get("schema_version") or CORRECTION_EVENTS_SCHEMA_VERSION
        )
        payload["event_id"] = str(payload["event_id"])
        seq_raw = payload["event_seq"]
        if seq_raw is None:
            raise ValueError("event_seq is required")
        payload["event_seq"] = int(seq_raw)
        payload["timestamp_utc"] = str(payload["timestamp_utc"])
        payload["reviewer_id"] = str(payload["reviewer_id"])
        payload["op"] = str(payload["op"])
        payload["reciprocal"] = bool(payload.get("reciprocal") or False)
        for int_field in (
            "focal_id",
            "partner_id",
            "mouse_a_id",
            "mouse_b_id",
            "start_sample_idx",
            "end_sample_idx",
            "pred_value",
        ):
            val = payload.get(int_field)
            if val is None or val == "":
                payload[int_field] = None
            else:
                payload[int_field] = int(val)
        return cls(**payload)  # type: ignore[arg-type]


def validate_event(
    event: CorrectionEvent,
    *,
    n_dense: int,
    allow_reciprocal: bool = False,
) -> None:
    """Reject invalid identities, labels, channels, intervals, and ops."""

    if event.schema_version != CORRECTION_EVENTS_SCHEMA_VERSION:
        raise EditingError(
            f"Unsupported correction event schema {event.schema_version!r}."
        )
    if not event.event_id:
        raise EditingError("event_id is required.")
    if event.event_seq < 1:
        raise EditingError(f"event_seq must be >= 1; got {event.event_seq}.")
    if not str(event.reviewer_id).strip():
        raise EditingError("reviewer_id is required for write events.")
    if event.op not in KNOWN_OPS:
        raise EditingError(f"Unknown or deferred op {event.op!r}.")
    if event.reciprocal and not allow_reciprocal:
        raise EditingError(
            "Reciprocal correction ops are deferred (U2); reciprocal=True rejected."
        )
    if event.op == OP_SET_EXCLUSIVE_LABEL:
        _require_interval(event, n_dense=n_dense)
        _require_directional(event)
        if event.channel not in (None, CHANNEL_EXCLUSIVE):
            raise EditingError(
                f"set_exclusive_label requires channel exclusive; got {event.channel!r}."
            )
        if event.label not in EXCLUSIVE_ETHOGRAM_LABELS:
            raise EditingError(f"Invalid exclusive label {event.label!r}.")
        event.channel = CHANNEL_EXCLUSIVE
        event.identity_grain = IDENTITY_DIRECTIONAL
    elif event.op in (OP_SET_PARALLEL_PRED, OP_CLEAR_PARALLEL_PRED):
        _require_interval(event, n_dense=n_dense)
        _require_symmetric(event)
        if event.channel not in (CHANNEL_CONTACT, CHANNEL_SIDE_BY_SIDE):
            raise EditingError(
                f"Parallel ops require contact or side_by_side; got {event.channel!r}."
            )
        event.identity_grain = IDENTITY_SYMMETRIC_PAIR
        if event.op == OP_CLEAR_PARALLEL_PRED:
            event.pred_value = 0
        if event.pred_value not in (0, 1):
            raise EditingError(
                f"pred_value must be 0 or 1; got {event.pred_value!r}."
            )
    elif event.op == OP_TRIM_BOUT_BOUNDARY:
        _require_bout_target(event)
        payload = _parse_payload(event)
        if "new_start_sample_idx" not in payload or "new_end_sample_idx" not in payload:
            raise EditingError(
                "trim_bout_boundary requires new_start_sample_idx and "
                "new_end_sample_idx in payload_json."
            )
        ns = int(payload["new_start_sample_idx"])
        ne = int(payload["new_end_sample_idx"])
        if ne < ns:
            raise EditingError("trim new interval end < start.")
        if ns < 0 or ne >= n_dense:
            raise EditingError("trim new interval out of dense range.")
    elif event.op == OP_SPLIT_BOUT_AT_SAMPLE:
        _require_bout_target(event)
        payload = _parse_payload(event)
        if "at_sample_idx" not in payload:
            raise EditingError("split_bout_at_sample requires at_sample_idx.")
        at = int(payload["at_sample_idx"])
        if at < 0 or at >= n_dense:
            raise EditingError("split at_sample_idx out of dense range.")
    elif event.op == OP_MERGE_ADJACENT_BOUTS:
        payload = _parse_payload(event)
        if "bout_id_a" not in payload or "bout_id_b" not in payload:
            raise EditingError("merge_adjacent_bouts requires bout_id_a and bout_id_b.")
    elif event.op == OP_SET_BOUT_REVIEW_STATUS:
        _require_bout_target(event)
        if event.status not in DECISION_STATUS_VALUES:
            raise EditingError(
                f"Invalid review status {event.status!r}; expected one of "
                f"{sorted(DECISION_STATUS_VALUES)}."
            )
    elif event.op in (OP_ADD_NOTE, OP_EDIT_NOTE, OP_DELETE_NOTE):
        if (
            event.op == OP_ADD_NOTE
            and not (event.note_text or "").strip()
        ):
            raise EditingError("add_note requires note_text.")
        if event.op in (OP_EDIT_NOTE, OP_DELETE_NOTE) and not event.note_id:
            raise EditingError(f"{event.op} requires note_id.")
    elif event.op == OP_SESSION_MARKER:
        pass


def _require_interval(event: CorrectionEvent, *, n_dense: int) -> None:
    if event.start_sample_idx is None or event.end_sample_idx is None:
        raise EditingError(f"{event.op} requires inclusive start/end sample indices.")
    start = int(event.start_sample_idx)
    end = int(event.end_sample_idx)
    if end < start:
        raise EditingError(
            f"Illegal empty/inverted interval [{start}, {end}] (inclusive)."
        )
    if start < 0 or end >= n_dense:
        raise EditingError(
            f"Interval [{start}, {end}] outside dense range [0, {n_dense - 1}]."
        )


def _require_directional(event: CorrectionEvent) -> None:
    if event.focal_id is None or event.partner_id is None:
        raise EditingError(f"{event.op} requires focal_id and partner_id.")
    if int(event.focal_id) == int(event.partner_id):
        raise EditingError("focal_id and partner_id must differ.")
    a = min(int(event.focal_id), int(event.partner_id))
    b = max(int(event.focal_id), int(event.partner_id))
    event.mouse_a_id = a
    event.mouse_b_id = b
    event.pair_id = pair_id_for(a, b)


def _require_symmetric(event: CorrectionEvent) -> None:
    if event.mouse_a_id is None or event.mouse_b_id is None:
        if event.focal_id is not None and event.partner_id is not None:
            event.mouse_a_id = min(int(event.focal_id), int(event.partner_id))
            event.mouse_b_id = max(int(event.focal_id), int(event.partner_id))
        else:
            raise EditingError(f"{event.op} requires mouse_a_id and mouse_b_id.")
    a = int(event.mouse_a_id)
    b = int(event.mouse_b_id)
    if a == b:
        raise EditingError("mouse_a_id and mouse_b_id must differ.")
    if a > b:
        a, b = b, a
    event.mouse_a_id = a
    event.mouse_b_id = b
    event.pair_id = pair_id_for(a, b)
    # Parallel ops never invent directional focal/partner on bout rows.
    event.focal_id = None
    event.partner_id = None


def _require_bout_target(event: CorrectionEvent) -> None:
    if not event.target_bout_id:
        raise EditingError(f"{event.op} requires target_bout_id.")
    kind = event.target_bout_id_kind
    if kind is None:
        event.target_bout_id_kind = BOUT_ID_KIND_SOURCE_S8
        return
    if kind not in {
        BOUT_ID_KIND_SOURCE_S8,
        BOUT_ID_KIND_PACKAGE,
        BOUT_ID_KIND_REVIEWED,
        BOUT_ID_KIND_INTERVAL,
    }:
        raise EditingError(f"Invalid target_bout_id_kind {kind!r}.")


def _parse_payload(event: CorrectionEvent) -> dict[str, Any]:
    if not event.payload_json:
        return {}
    try:
        raw = json.loads(event.payload_json)
    except json.JSONDecodeError as exc:
        raise EditingError(f"Invalid payload_json: {exc}") from exc
    if not isinstance(raw, dict):
        raise EditingError("payload_json must be a JSON object.")
    return raw


def build_event(
    *,
    op: str,
    reviewer_id: str,
    event_seq: int,
    undo_group_id: str | None = None,
    client_op_id: str | None = None,
    **fields: Any,
) -> CorrectionEvent:
    event = CorrectionEvent(
        schema_version=CORRECTION_EVENTS_SCHEMA_VERSION,
        event_id=new_event_id(),
        event_seq=int(event_seq),
        timestamp_utc=utc_now_iso(),
        reviewer_id=str(reviewer_id).strip(),
        op=op,
        undo_group_id=undo_group_id or new_undo_group_id(),
        client_op_id=client_op_id or new_client_op_id(),
        **fields,
    )
    return event
