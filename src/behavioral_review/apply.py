"""Apply framewise correction events onto working reviewed states."""

from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    EXCLUSIVE_ETHOGRAM_LABELS,
    FRAMEWISE_OPS,
    LABEL_RECONSTRUCTION_MANUAL,
    OP_CLEAR_PARALLEL_PRED,
    OP_MERGE_ADJACENT_BOUTS,
    OP_SET_EXCLUSIVE_LABEL,
    OP_SET_PARALLEL_PRED,
    OP_SPLIT_BOUT_AT_SAMPLE,
    OP_TRIM_BOUT_BOUNDARY,
    EditingError,
)
from behavioral_review.events import CorrectionEvent, _parse_payload
from behavioral_review.regenerate import regenerate_reviewed_bouts

_EXCLUSIVE_PRED_LABELS = tuple(
    lab for lab in EXCLUSIVE_ETHOGRAM_LABELS if lab != "undefined"
)

# Bout geometry ops that resolve targets against a reviewed bout table.
_REVIEWED_LOOKUP_OPS = frozenset(
    {
        OP_TRIM_BOUT_BOUNDARY,
        OP_SPLIT_BOUT_AT_SAMPLE,
        OP_MERGE_ADJACENT_BOUTS,
    }
)


def replay_active_correction_chain(
    baseline_states: pd.DataFrame,
    active: list[CorrectionEvent],
    *,
    review_run_id: str,
    parent_package_run_id: str,
    source_s8_run_id: str | None,
    source_s7_run_id: str | None,
    source_s6_run_id: str | None,
    automated_bouts: pd.DataFrame | None,
    sync_index: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[tuple[int, int]]], int]:
    """Replay active journal events and regenerate reviewed bouts.

    Avoids redundant ``extract_behavior_bouts`` passes that do not change
    scientific outcomes:

    - empty / non-framewise chains regenerate once
    - framewise label/pred ops regenerate after mutation only
    - trim/split/merge still regenerate before (lookup) and after mutation
    """

    working = baseline_states.copy()
    reviewed: pd.DataFrame | None = None
    merged_changed: dict[str, list[tuple[int, int]]] = {}

    def _regen() -> pd.DataFrame:
        nonlocal reviewed
        reviewed = regenerate_reviewed_bouts(
            working,
            review_run_id=review_run_id,
            parent_package_run_id=parent_package_run_id,
            source_s8_run_id=source_s8_run_id,
            source_s7_run_id=source_s7_run_id,
            source_s6_run_id=source_s6_run_id,
            automated_bouts=automated_bouts,
            sync_index=sync_index,
        )
        return reviewed

    for event in active:
        if event.op in _REVIEWED_LOOKUP_OPS and reviewed is None:
            _regen()
        working, meta = apply_events_to_states(
            working,
            [event],
            automated_bouts=automated_bouts,
            reviewed_bouts_lookup=reviewed,
            copy_states=False,
        )
        for ch, spans in (meta.get("changed_intervals") or {}).items():
            merged_changed.setdefault(ch, []).extend(spans)
        if event.op in FRAMEWISE_OPS:
            _regen()

    if reviewed is None:
        _regen()
    assert reviewed is not None
    changed = {
        ch: merge_changed_intervals(spans) for ch, spans in merged_changed.items()
    }
    return working, reviewed, changed, len(active)


def apply_events_to_states(
    baseline_states: pd.DataFrame,
    events: list[CorrectionEvent],
    *,
    automated_bouts: pd.DataFrame | None = None,
    reviewed_bouts_lookup: pd.DataFrame | None = None,
    copy_states: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay events onto baseline/working states.

    When ``copy_states`` is True (default), starts from a copy of
    ``baseline_states``. When False, mutates ``baseline_states`` in place
    (used for incremental draft replay).

    Returns ``(working_states, apply_meta)`` where ``apply_meta`` includes
    changed sample intervals per channel for incremental consumers.
    """

    if baseline_states is None or baseline_states.empty:
        raise EditingError("Cannot apply corrections to empty baseline states.")
    work = baseline_states.copy() if copy_states else baseline_states
    if "dense_frame_idx" not in work.columns:
        raise EditingError("Baseline states missing dense_frame_idx.")
    if "sample_idx" not in work.columns:
        work["sample_idx"] = work["dense_frame_idx"]

    changed: dict[str, list[tuple[int, int]]] = {
        CHANNEL_EXCLUSIVE: [],
        CHANNEL_CONTACT: [],
        CHANNEL_SIDE_BY_SIDE: [],
    }
    bout_tables = {
        "automated": automated_bouts,
        "reviewed": reviewed_bouts_lookup,
    }

    for event in events:
        if event.op == OP_SET_EXCLUSIVE_LABEL:
            _apply_exclusive(work, event)
            changed[CHANNEL_EXCLUSIVE].append(
                (int(event.start_sample_idx), int(event.end_sample_idx))  # type: ignore[arg-type]
            )
        elif event.op in (OP_SET_PARALLEL_PRED, OP_CLEAR_PARALLEL_PRED):
            _apply_parallel(work, event)
            channel = str(event.channel)
            changed[channel].append(
                (int(event.start_sample_idx), int(event.end_sample_idx))  # type: ignore[arg-type]
            )
        elif event.op in (
            OP_TRIM_BOUT_BOUNDARY,
            OP_SPLIT_BOUT_AT_SAMPLE,
            OP_MERGE_ADJACENT_BOUTS,
        ):
            compiled = compile_bout_op_to_framewise(
                event,
                working_states=work,
                bout_tables=bout_tables,
            )
            for child in compiled:
                if child.op == OP_SET_EXCLUSIVE_LABEL:
                    _apply_exclusive(work, child)
                    changed[CHANNEL_EXCLUSIVE].append(
                        (int(child.start_sample_idx), int(child.end_sample_idx))  # type: ignore[arg-type]
                    )
                elif child.op in (OP_SET_PARALLEL_PRED, OP_CLEAR_PARALLEL_PRED):
                    _apply_parallel(work, child)
                    channel = str(child.channel)
                    changed[channel].append(
                        (int(child.start_sample_idx), int(child.end_sample_idx))  # type: ignore[arg-type]
                    )
                else:
                    raise EditingError(
                        f"Compiled bout op produced unsupported child op {child.op!r}."
                    )
        # Status / notes / markers do not mutate ethogram science.

    meta = {
        "changed_intervals": {
            ch: _merge_intervals(spans) for ch, spans in changed.items() if spans
        },
        "n_events_applied": len(events),
    }
    return work, meta


def compile_bout_op_to_framewise(
    event: CorrectionEvent,
    *,
    working_states: pd.DataFrame,
    bout_tables: dict[str, pd.DataFrame | None],
) -> list[CorrectionEvent]:
    """Deterministically expand trim/split/merge into framewise set ops."""

    payload = _parse_payload(event)
    if event.op == OP_TRIM_BOUT_BOUNDARY:
        return _compile_trim(event, payload, working_states, bout_tables)
    if event.op == OP_SPLIT_BOUT_AT_SAMPLE:
        return _compile_split(event, payload, working_states, bout_tables)
    if event.op == OP_MERGE_ADJACENT_BOUTS:
        return _compile_merge(event, payload, working_states, bout_tables)
    raise EditingError(f"Not a bout compile op: {event.op}")


def _resolve_bout_row(
    event: CorrectionEvent,
    bout_tables: dict[str, pd.DataFrame | None],
    *,
    bout_id: str | None = None,
    bout_id_kind: str | None = None,
) -> pd.Series:
    target = bout_id or event.target_bout_id
    kind = bout_id_kind or event.target_bout_id_kind or "source_s8_bout_id"
    if not target:
        raise EditingError("Bout op missing target bout id.")
    for table in (bout_tables.get("reviewed"), bout_tables.get("automated")):
        if table is None or table.empty:
            continue
        if kind == "source_s8_bout_id" and "source_s8_bout_id" in table.columns:
            hits = table[table["source_s8_bout_id"].astype(str) == str(target)]
            if len(hits) == 1:
                return hits.iloc[0]
        if kind == "package_bout_id" and "bout_id" in table.columns:
            hits = table[table["bout_id"].astype(str) == str(target)]
            if len(hits) == 1:
                return hits.iloc[0]
        if kind == "reviewed_bout_id" and "bout_id" in table.columns:
            hits = table[table["bout_id"].astype(str) == str(target)]
            if len(hits) == 1:
                return hits.iloc[0]
        # Fallback: try both id columns.
        for col in ("bout_id", "source_s8_bout_id"):
            if col in table.columns:
                hits = table[table[col].astype(str) == str(target)]
                if len(hits) == 1:
                    return hits.iloc[0]
    raise EditingError(f"Could not resolve bout target {target!r} ({kind}).")


def _compile_trim(
    event: CorrectionEvent,
    payload: dict[str, Any],
    states: pd.DataFrame,
    bout_tables: dict[str, pd.DataFrame | None],
) -> list[CorrectionEvent]:
    bout = _resolve_bout_row(event, bout_tables)
    channel = str(bout["channel"])
    old_start = int(bout["start_sample_idx"])
    old_end = int(bout["end_sample_idx"])
    new_start = int(payload["new_start_sample_idx"])
    new_end = int(payload["new_end_sample_idx"])
    if new_start < old_start or new_end > old_end:
        # Expansion: paint bout behavior into expanded region.
        behavior = str(bout["behavior_name"])
        children: list[CorrectionEvent] = []
        if new_start < old_start:
            children.append(
                _child_set_for_bout_channel(
                    event,
                    channel=channel,
                    bout=bout,
                    start=new_start,
                    end=old_start - 1,
                    label=behavior,
                    pred_value=1,
                )
            )
        if new_end > old_end:
            children.append(
                _child_set_for_bout_channel(
                    event,
                    channel=channel,
                    bout=bout,
                    start=old_end + 1,
                    end=new_end,
                    label=behavior,
                    pred_value=1,
                )
            )
        return children

    # Shrink: relabel removed shoulders from neighbor continuity.
    children = []
    if new_start > old_start:
        fill = payload.get("fill_label")
        fill_pred = payload.get("fill_pred_value", 0)
        if channel == CHANNEL_EXCLUSIVE:
            fill = fill or _neighbor_exclusive_label(
                states, bout=bout, sample_idx=old_start - 1, fallback="mutually_disengaged"
            )
            children.append(
                _child_set_for_bout_channel(
                    event,
                    channel=channel,
                    bout=bout,
                    start=old_start,
                    end=new_start - 1,
                    label=str(fill),
                    pred_value=None,
                )
            )
        else:
            children.append(
                _child_set_for_bout_channel(
                    event,
                    channel=channel,
                    bout=bout,
                    start=old_start,
                    end=new_start - 1,
                    label=None,
                    pred_value=int(fill_pred),
                )
            )
    if new_end < old_end:
        fill = payload.get("fill_label")
        fill_pred = payload.get("fill_pred_value", 0)
        if channel == CHANNEL_EXCLUSIVE:
            fill = fill or _neighbor_exclusive_label(
                states, bout=bout, sample_idx=old_end + 1, fallback="mutually_disengaged"
            )
            children.append(
                _child_set_for_bout_channel(
                    event,
                    channel=channel,
                    bout=bout,
                    start=new_end + 1,
                    end=old_end,
                    label=str(fill),
                    pred_value=None,
                )
            )
        else:
            children.append(
                _child_set_for_bout_channel(
                    event,
                    channel=channel,
                    bout=bout,
                    start=new_end + 1,
                    end=old_end,
                    label=None,
                    pred_value=int(fill_pred),
                )
            )
    if not children:
        # No geometry change — still legal no-op compile.
        return []
    return children


def _compile_split(
    event: CorrectionEvent,
    payload: dict[str, Any],
    states: pd.DataFrame,
    bout_tables: dict[str, pd.DataFrame | None],
) -> list[CorrectionEvent]:
    bout = _resolve_bout_row(event, bout_tables)
    channel = str(bout["channel"])
    start = int(bout["start_sample_idx"])
    end = int(bout["end_sample_idx"])
    at = int(payload["at_sample_idx"])
    if at <= start or at > end:
        raise EditingError(
            f"split at_sample_idx {at} must be inside ({start}, {end}] "
            "(right segment start)."
        )
    right_label = payload.get("right_label")
    right_pred = payload.get("right_pred_value")
    if channel == CHANNEL_EXCLUSIVE:
        if right_label is None:
            raise EditingError(
                "split_bout_at_sample for exclusive requires right_label "
                "(must differ from left to create a boundary)."
            )
        left_label = str(bout["behavior_name"])
        if str(right_label) == left_label:
            raise EditingError(
                "Exclusive split requires right_label different from the bout "
                "label so contiguous collapse yields two bouts."
            )
        return [
            _child_set_for_bout_channel(
                event,
                channel=channel,
                bout=bout,
                start=at,
                end=end,
                label=str(right_label),
                pred_value=None,
            )
        ]
    # Parallel: clear or set right side differently — default clear left? 
    # Split positive bout: set sample at-1 boundary by clearing `at` briefly...
    # Spec: force discontinuity — set right segment pred to right_pred (default 0)
    # which splits off the right if right becomes 0, or relabel if 1 with gap.
    # Practical: set [at, end] to right_pred_value (default 0 → ends bout early,
    # creating a trim-like split). To keep two positive bouts, set intervening
    # sample to 0.
    if right_pred is None:
        # Insert a one-sample zero at `at` to split two positive segments.
        if at == end:
            raise EditingError("Cannot split parallel bout at final sample without right_pred_value.")
        return [
            _child_set_for_bout_channel(
                event,
                channel=channel,
                bout=bout,
                start=at,
                end=at,
                label=None,
                pred_value=0,
            )
        ]
    return [
        _child_set_for_bout_channel(
            event,
            channel=channel,
            bout=bout,
            start=at,
            end=end,
            label=None,
            pred_value=int(right_pred),
        )
    ]


def _compile_merge(
    event: CorrectionEvent,
    payload: dict[str, Any],
    states: pd.DataFrame,
    bout_tables: dict[str, pd.DataFrame | None],
) -> list[CorrectionEvent]:
    bout_a = _resolve_bout_row(
        event,
        bout_tables,
        bout_id=str(payload["bout_id_a"]),
        bout_id_kind=str(payload.get("bout_id_kind_a") or event.target_bout_id_kind or "source_s8_bout_id"),
    )
    bout_b = _resolve_bout_row(
        event,
        bout_tables,
        bout_id=str(payload["bout_id_b"]),
        bout_id_kind=str(payload.get("bout_id_kind_b") or event.target_bout_id_kind or "source_s8_bout_id"),
    )
    if str(bout_a["channel"]) != str(bout_b["channel"]):
        raise EditingError("Cannot merge bouts from different channels.")
    channel = str(bout_a["channel"])
    if channel == CHANNEL_EXCLUSIVE:
        if int(bout_a["focal_id"]) != int(bout_b["focal_id"]) or int(
            bout_a["partner_id"]
        ) != int(bout_b["partner_id"]):
            raise EditingError("Exclusive merge requires the same direction.")
    else:
        if int(bout_a["mouse_a_id"]) != int(bout_b["mouse_a_id"]) or int(
            bout_a["mouse_b_id"]
        ) != int(bout_b["mouse_b_id"]):
            raise EditingError("Parallel merge requires the same symmetric pair.")

    start = min(int(bout_a["start_sample_idx"]), int(bout_b["start_sample_idx"]))
    end = max(int(bout_a["end_sample_idx"]), int(bout_b["end_sample_idx"]))
    # Ensure adjacency / small gap: gap samples also painted.
    gap_ok = abs(int(bout_a["start_sample_idx"]) - int(bout_b["end_sample_idx"])) <= 2 or abs(
        int(bout_b["start_sample_idx"]) - int(bout_a["end_sample_idx"])
    ) <= 2 or (
        min(int(bout_a["end_sample_idx"]), int(bout_b["end_sample_idx"])) + 1
        >= max(int(bout_a["start_sample_idx"]), int(bout_b["start_sample_idx"])) - 1
    )
    if not gap_ok and abs(start - end) > 0:
        # Still allow explicit merge across intervening samples (operator intent).
        pass
    shared_label = payload.get("label") or str(bout_a["behavior_name"])
    if channel == CHANNEL_EXCLUSIVE:
        return [
            _child_set_for_bout_channel(
                event,
                channel=channel,
                bout=bout_a,
                start=start,
                end=end,
                label=str(shared_label),
                pred_value=None,
            )
        ]
    return [
        _child_set_for_bout_channel(
            event,
            channel=channel,
            bout=bout_a,
            start=start,
            end=end,
            label=None,
            pred_value=1,
        )
    ]


def _child_set_for_bout_channel(
    parent: CorrectionEvent,
    *,
    channel: str,
    bout: pd.Series,
    start: int,
    end: int,
    label: str | None,
    pred_value: int | None,
) -> CorrectionEvent:
    if end < start:
        raise EditingError("Compiled child interval end < start.")
    if channel == CHANNEL_EXCLUSIVE:
        return CorrectionEvent(
            schema_version=parent.schema_version,
            event_id=f"{parent.event_id}_c{start}_{end}",
            event_seq=parent.event_seq,
            timestamp_utc=parent.timestamp_utc,
            reviewer_id=parent.reviewer_id,
            op=OP_SET_EXCLUSIVE_LABEL,
            channel=CHANNEL_EXCLUSIVE,
            identity_grain="directional",
            focal_id=int(bout["focal_id"]),
            partner_id=int(bout["partner_id"]),
            mouse_a_id=int(bout["mouse_a_id"]),
            mouse_b_id=int(bout["mouse_b_id"]),
            pair_id=str(bout["pair_id"]),
            start_sample_idx=int(start),
            end_sample_idx=int(end),
            label=str(label),
            client_op_id=parent.client_op_id,
            undo_group_id=parent.undo_group_id,
            reciprocal=False,
        )
    return CorrectionEvent(
        schema_version=parent.schema_version,
        event_id=f"{parent.event_id}_c{start}_{end}",
        event_seq=parent.event_seq,
        timestamp_utc=parent.timestamp_utc,
        reviewer_id=parent.reviewer_id,
        op=OP_SET_PARALLEL_PRED,
        channel=channel,
        identity_grain="symmetric_pair",
        mouse_a_id=int(bout["mouse_a_id"]),
        mouse_b_id=int(bout["mouse_b_id"]),
        pair_id=str(bout["pair_id"]),
        start_sample_idx=int(start),
        end_sample_idx=int(end),
        pred_value=int(pred_value if pred_value is not None else 0),
        client_op_id=parent.client_op_id,
        undo_group_id=parent.undo_group_id,
        reciprocal=False,
    )


def _neighbor_exclusive_label(
    states: pd.DataFrame,
    *,
    bout: pd.Series,
    sample_idx: int,
    fallback: str,
) -> str:
    focal = int(bout["focal_id"])
    partner = int(bout["partner_id"])
    mask = (
        (states["dense_frame_idx"] == int(sample_idx))
        & (states["focal_id"] == focal)
        & (states["partner_id"] == partner)
    )
    hits = states.loc[mask]
    if hits.empty:
        return fallback
    return str(hits.iloc[0]["behavior_label"])


def _apply_exclusive(states: pd.DataFrame, event: CorrectionEvent) -> None:
    focal = int(event.focal_id)  # type: ignore[arg-type]
    partner = int(event.partner_id)  # type: ignore[arg-type]
    start = int(event.start_sample_idx)  # type: ignore[arg-type]
    end = int(event.end_sample_idx)  # type: ignore[arg-type]
    label = str(event.label)
    mask = (
        (states["dense_frame_idx"] >= start)
        & (states["dense_frame_idx"] <= end)
        & (states["focal_id"] == focal)
        & (states["partner_id"] == partner)
    )
    if not bool(mask.any()):
        raise EditingError(
            f"No state rows for exclusive write "
            f"direction=({focal},{partner}) interval=[{start},{end}]."
        )
    states.loc[mask, "behavior_label"] = label
    if "label_reconstruction" in states.columns:
        states.loc[mask, "label_reconstruction"] = LABEL_RECONSTRUCTION_MANUAL
    if "undefined_reason" in states.columns:
        if label == "undefined":
            # Preserve existing reasons unless empty.
            empty = states.loc[mask, "undefined_reason"].fillna("").astype(str).str.len() == 0
            states.loc[mask & empty, "undefined_reason"] = "manual_review"
        else:
            states.loc[mask, "undefined_reason"] = ""
    _sync_exclusive_preds(states, mask, label)


def _sync_exclusive_preds(
    states: pd.DataFrame, mask: pd.Series, label: str
) -> None:
    """Keep exclusive pred_* columns consistent for display (not S7 rewrite)."""

    for lab in _EXCLUSIVE_PRED_LABELS:
        col = f"pred_{lab}"
        if col in states.columns:
            states.loc[mask, col] = 1 if lab == label else 0


def _apply_parallel(states: pd.DataFrame, event: CorrectionEvent) -> None:
    a = int(event.mouse_a_id)  # type: ignore[arg-type]
    b = int(event.mouse_b_id)  # type: ignore[arg-type]
    start = int(event.start_sample_idx)  # type: ignore[arg-type]
    end = int(event.end_sample_idx)  # type: ignore[arg-type]
    pred = int(event.pred_value)  # type: ignore[arg-type]
    channel = str(event.channel)
    col = "pred_contact" if channel == CHANNEL_CONTACT else "pred_side_by_side"
    if col not in states.columns:
        raise EditingError(f"Baseline states missing {col} for parallel edit.")
    # S8 states are directional rows carrying parallel preds — write both dirs.
    pair_mask = (
        (states["dense_frame_idx"] >= start)
        & (states["dense_frame_idx"] <= end)
        & (
            ((states["focal_id"] == a) & (states["partner_id"] == b))
            | ((states["focal_id"] == b) & (states["partner_id"] == a))
        )
    )
    if "mouse_a" in states.columns and "mouse_b" in states.columns:
        pair_mask = (
            (states["dense_frame_idx"] >= start)
            & (states["dense_frame_idx"] <= end)
            & (states["mouse_a"] == a)
            & (states["mouse_b"] == b)
        )
    if not bool(pair_mask.any()):
        raise EditingError(
            f"No state rows for parallel write pair=({a},{b}) interval=[{start},{end}]."
        )
    states.loc[pair_mask, col] = pred


def merge_changed_intervals(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted((int(s), int(e)) for s, e in spans)
    out = [ordered[0]]
    for start, end in ordered[1:]:
        ps, pe = out[-1]
        if start <= pe + 1:
            out[-1] = (ps, max(pe, end))
        else:
            out.append((start, end))
    return out


def _merge_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return merge_changed_intervals(spans)


def compare_labels(
    baseline: pd.DataFrame,
    reviewed: pd.DataFrame,
    *,
    start_sample_idx: int | None = None,
    end_sample_idx: int | None = None,
) -> pd.DataFrame:
    """Return rows where exclusive or parallel labels differ from automated."""

    cols = ["dense_frame_idx", "focal_id", "partner_id", "behavior_label"]
    for c in ("pred_contact", "pred_side_by_side"):
        if c in baseline.columns and c in reviewed.columns:
            cols.append(c)
    b = baseline.loc[:, [c for c in cols if c in baseline.columns]].copy()
    r = reviewed.loc[:, [c for c in cols if c in reviewed.columns]].copy()
    if start_sample_idx is not None:
        b = b[b["dense_frame_idx"] >= int(start_sample_idx)]
        r = r[r["dense_frame_idx"] >= int(start_sample_idx)]
    if end_sample_idx is not None:
        b = b[b["dense_frame_idx"] <= int(end_sample_idx)]
        r = r[r["dense_frame_idx"] <= int(end_sample_idx)]
    merged = b.merge(
        r,
        on=["dense_frame_idx", "focal_id", "partner_id"],
        suffixes=("_automated", "_reviewed"),
        how="inner",
    )
    differ = merged["behavior_label_automated"] != merged["behavior_label_reviewed"]
    if "pred_contact_automated" in merged.columns:
        differ = differ | (
            merged["pred_contact_automated"] != merged["pred_contact_reviewed"]
        )
    if "pred_side_by_side_automated" in merged.columns:
        differ = differ | (
            merged["pred_side_by_side_automated"]
            != merged["pred_side_by_side_reviewed"]
        )
    return merged.loc[differ].reset_index(drop=True)
