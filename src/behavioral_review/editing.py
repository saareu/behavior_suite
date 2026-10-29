"""Headless editing-session API: corrections, status, notes, undo/redo, drafts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.apply import (
    compare_labels,
    replay_active_correction_chain,
)
from behavioral_review.contracts import (
    BOUT_ID_KIND_SOURCE_S8,
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    DECISION_STATUS_VALUES,
    EDITING_SESSION_SCHEMA_VERSION,
    MODE_EDITING,
    MODE_INSPECTION,
    OP_ADD_NOTE,
    OP_CLEAR_PARALLEL_PRED,
    OP_MERGE_ADJACENT_BOUTS,
    OP_SESSION_MARKER,
    OP_SET_BOUT_REVIEW_STATUS,
    OP_SET_EXCLUSIVE_LABEL,
    OP_SET_PARALLEL_PRED,
    OP_SPLIT_BOUT_AT_SAMPLE,
    OP_TRIM_BOUT_BOUNDARY,
    RUN_LIFECYCLE_IN_PROGRESS,
    EditingError,
    SessionError,
)
from behavioral_review.draft import WorkingDraft, create_working_draft, open_working_draft
from behavioral_review.events import (
    CorrectionEvent,
    build_event,
    new_undo_group_id,
    validate_event,
)
from behavioral_review.session import InspectionSession, open_inspection_session
from behavioral_review.source import assert_no_upstream_writes, snapshot_upstream_fingerprints
from behavioral_review.status import ReviewOverlayState, replay_review_overlay
from behavioral_review.sync import PlayheadMapping


@dataclass
class EditingSession:
    """Inspection session + optional durable draft / edit mode.

    Seeking and evidence inspection never write journal events. Writes require
    editing mode, reviewer identity, an open draft, and the writer lock.
    """

    schema_version: str
    inspection: InspectionSession
    reviewer_id: str | None = None
    draft: WorkingDraft | None = None
    baseline_states: pd.DataFrame | None = None
    working_states: pd.DataFrame | None = None
    reviewed_bouts: pd.DataFrame | None = None
    overlay: ReviewOverlayState = field(default_factory=ReviewOverlayState)
    apply_meta: dict[str, Any] = field(default_factory=dict)
    _upstream_fingerprints: dict[Path, tuple[Any, ...]] = field(
        default_factory=dict, repr=False
    )
    _editing_enabled: bool = False

    @property
    def mode(self) -> str:
        return MODE_EDITING if self._editing_enabled else MODE_INSPECTION

    @property
    def package_run_id(self) -> str:
        return self.inspection.package_run_id

    @property
    def n_dense_samples(self) -> int:
        return self.inspection.n_dense_samples

    # --- mode / draft gates -------------------------------------------------

    def enter_editing_mode(self, reviewer_id: str) -> None:
        reviewer = str(reviewer_id).strip()
        if not reviewer:
            raise EditingError("reviewer_id is required to enter editing mode.")
        if not self.inspection.source.s8_states_available:
            raise EditingError(
                "Cannot enter editing mode without resolvable S8 behavior_states."
            )
        self.reviewer_id = reviewer
        self._editing_enabled = True
        self.inspection.mode = MODE_EDITING
        if self.baseline_states is None:
            self._load_baseline_states()
        if self.draft is not None:
            self._rebuild_working_projection()

    def exit_editing_mode(self, *, release_lock: bool = False) -> None:
        self._editing_enabled = False
        self.inspection.mode = MODE_INSPECTION
        if release_lock and self.draft is not None:
            self.draft.lock.release()

    def create_draft(
        self,
        *,
        review_root: Path | str | None = None,
        review_run_id: str | None = None,
    ) -> WorkingDraft:
        self._require_editing_gate(require_draft=False)
        assert self.reviewer_id is not None
        if self.draft is not None and self.draft.lock.is_held:
            raise EditingError("A working draft is already open on this session.")
        draft = create_working_draft(
            self.inspection.source,
            reviewer_id=self.reviewer_id,
            review_root=review_root,
            review_run_id=review_run_id,
            acquire_lock=True,
        )
        self.draft = draft
        self._append_session_marker("draft_created")
        self._rebuild_working_projection()
        return draft

    def open_draft(
        self,
        review_run_dir: Path | str,
        *,
        force_stale_lock: bool = False,
    ) -> WorkingDraft:
        self._require_editing_gate(require_draft=False)
        assert self.reviewer_id is not None
        draft = open_working_draft(
            review_run_dir,
            reviewer_id=self.reviewer_id,
            expected_package_run_id=self.package_run_id,
            acquire_lock=True,
            force_stale_lock=force_stale_lock,
        )
        self.draft = draft
        self._rebuild_working_projection()
        return draft

    def close_draft(self, *, release_lock: bool = True) -> None:
        if self.draft is not None and release_lock:
            self.draft.lock.release()
        self.draft = None
        self.working_states = (
            self.baseline_states.copy() if self.baseline_states is not None else None
        )
        self.reviewed_bouts = None
        self.overlay = ReviewOverlayState()

    # --- corrections --------------------------------------------------------

    def set_exclusive_label(
        self,
        *,
        focal_id: int,
        partner_id: int,
        start_sample_idx: int,
        end_sample_idx: int,
        label: str,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        return self._commit_events(
            [
                self._make_event(
                    OP_SET_EXCLUSIVE_LABEL,
                    channel=CHANNEL_EXCLUSIVE,
                    focal_id=int(focal_id),
                    partner_id=int(partner_id),
                    start_sample_idx=int(start_sample_idx),
                    end_sample_idx=int(end_sample_idx),
                    label=str(label),
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    def set_parallel_pred(
        self,
        *,
        channel: str,
        mouse_a_id: int,
        mouse_b_id: int,
        start_sample_idx: int,
        end_sample_idx: int,
        pred_value: int,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        if channel not in (CHANNEL_CONTACT, CHANNEL_SIDE_BY_SIDE):
            raise EditingError(
                f"Parallel channel must be contact or side_by_side; got {channel!r}."
            )
        op = OP_CLEAR_PARALLEL_PRED if int(pred_value) == 0 else OP_SET_PARALLEL_PRED
        return self._commit_events(
            [
                self._make_event(
                    op,
                    channel=channel,
                    mouse_a_id=int(mouse_a_id),
                    mouse_b_id=int(mouse_b_id),
                    start_sample_idx=int(start_sample_idx),
                    end_sample_idx=int(end_sample_idx),
                    pred_value=int(pred_value),
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    def trim_bout_boundary(
        self,
        *,
        target_bout_id: str,
        new_start_sample_idx: int,
        new_end_sample_idx: int,
        target_bout_id_kind: str = BOUT_ID_KIND_SOURCE_S8,
        fill_label: str | None = None,
        fill_pred_value: int | None = None,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        payload: dict[str, Any] = {
            "new_start_sample_idx": int(new_start_sample_idx),
            "new_end_sample_idx": int(new_end_sample_idx),
        }
        if fill_label is not None:
            payload["fill_label"] = fill_label
        if fill_pred_value is not None:
            payload["fill_pred_value"] = int(fill_pred_value)
        return self._commit_events(
            [
                self._make_event(
                    OP_TRIM_BOUT_BOUNDARY,
                    target_bout_id=str(target_bout_id),
                    target_bout_id_kind=target_bout_id_kind,
                    payload_json=json.dumps(payload, sort_keys=True),
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    def split_bout_at_sample(
        self,
        *,
        target_bout_id: str,
        at_sample_idx: int,
        right_label: str | None = None,
        right_pred_value: int | None = None,
        target_bout_id_kind: str = BOUT_ID_KIND_SOURCE_S8,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        payload: dict[str, Any] = {"at_sample_idx": int(at_sample_idx)}
        if right_label is not None:
            payload["right_label"] = right_label
        if right_pred_value is not None:
            payload["right_pred_value"] = int(right_pred_value)
        return self._commit_events(
            [
                self._make_event(
                    OP_SPLIT_BOUT_AT_SAMPLE,
                    target_bout_id=str(target_bout_id),
                    target_bout_id_kind=target_bout_id_kind,
                    payload_json=json.dumps(payload, sort_keys=True),
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    def merge_adjacent_bouts(
        self,
        *,
        bout_id_a: str,
        bout_id_b: str,
        label: str | None = None,
        bout_id_kind: str = BOUT_ID_KIND_SOURCE_S8,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        payload: dict[str, Any] = {
            "bout_id_a": str(bout_id_a),
            "bout_id_b": str(bout_id_b),
            "bout_id_kind_a": bout_id_kind,
            "bout_id_kind_b": bout_id_kind,
        }
        if label is not None:
            payload["label"] = label
        return self._commit_events(
            [
                self._make_event(
                    OP_MERGE_ADJACENT_BOUTS,
                    target_bout_id=str(bout_id_a),
                    target_bout_id_kind=bout_id_kind,
                    payload_json=json.dumps(payload, sort_keys=True),
                    label=label,
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    def set_bout_review_status(
        self,
        *,
        target_bout_id: str,
        status: str,
        target_bout_id_kind: str = BOUT_ID_KIND_SOURCE_S8,
        channel: str | None = None,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        if status not in DECISION_STATUS_VALUES:
            raise EditingError(
                f"Invalid status {status!r}; expected one of "
                f"{sorted(DECISION_STATUS_VALUES)}."
            )
        return self._commit_events(
            [
                self._make_event(
                    OP_SET_BOUT_REVIEW_STATUS,
                    target_bout_id=str(target_bout_id),
                    target_bout_id_kind=target_bout_id_kind,
                    status=str(status),
                    channel=channel,
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    def add_note(
        self,
        *,
        note_text: str,
        target_bout_id: str | None = None,
        target_bout_id_kind: str | None = None,
        note_id: str | None = None,
        client_op_id: str | None = None,
    ) -> CorrectionEvent:
        self._require_editing_gate(require_draft=True)
        return self._commit_events(
            [
                self._make_event(
                    OP_ADD_NOTE,
                    note_text=str(note_text),
                    target_bout_id=target_bout_id,
                    target_bout_id_kind=target_bout_id_kind,
                    note_id=note_id,
                    client_op_id=client_op_id,
                )
            ]
        )[0]

    # --- undo / redo --------------------------------------------------------

    def undo(self) -> list[CorrectionEvent]:
        self._require_editing_gate(require_draft=True)
        assert self.draft is not None
        undone = self.draft.journal.undo_group()
        self._rebuild_working_projection()
        self._assert_no_upstream_mutation()
        return undone

    def redo(self) -> list[CorrectionEvent]:
        self._require_editing_gate(require_draft=True)
        assert self.draft is not None
        redone = self.draft.journal.redo_group()
        self._rebuild_working_projection()
        self._assert_no_upstream_mutation()
        return redone

    # --- queries ------------------------------------------------------------

    def current_reviewed_states(self) -> pd.DataFrame:
        if self.working_states is None:
            self._load_baseline_states()
            assert self.working_states is not None
        return self.working_states

    def current_reviewed_bouts(self) -> pd.DataFrame:
        if self.reviewed_bouts is None:
            self._rebuild_working_projection()
        assert self.reviewed_bouts is not None
        return self.reviewed_bouts

    def current_statuses(self) -> pd.DataFrame:
        return self.overlay.status_frame()

    def current_notes(self) -> pd.DataFrame:
        return self.overlay.notes_frame()

    def changed_intervals(self) -> dict[str, list[tuple[int, int]]]:
        return dict(self.apply_meta.get("changed_intervals") or {})

    def compare_automated_versus_reviewed(
        self,
        *,
        start_sample_idx: int | None = None,
        end_sample_idx: int | None = None,
    ) -> pd.DataFrame:
        if self.baseline_states is None or self.working_states is None:
            self._rebuild_working_projection()
        assert self.baseline_states is not None
        assert self.working_states is not None
        return compare_labels(
            self.baseline_states,
            self.working_states,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
        )

    def seek_sample(
        self, sample_idx: int, *, decode: bool = False
    ) -> PlayheadMapping:
        return self.inspection.seek_sample(sample_idx, decode=decode)

    def publish_reviewed_run(
        self, *, settings: dict[str, Any] | None = None
    ) -> Any:
        """Atomically publish the open draft as an immutable reviewed run."""

        if not self._editing_enabled:
            raise EditingError(
                "Editing mode is not active. Call enter_editing_mode(reviewer_id) first."
            )
        if self.draft is None:
            raise EditingError(
                "No working draft open. Call create_draft() or open_draft() first."
            )
        from behavioral_review.publication import publish_reviewed_run

        # Writer-lock conflicts are reported by the publication eligibility gate.
        return publish_reviewed_run(self, settings=settings)

    def snapshot(self) -> dict[str, Any]:
        base = self.inspection.snapshot()
        base.update(
            {
                "editing_schema_version": self.schema_version,
                "mode": self.mode,
                "reviewer_id": self.reviewer_id,
                "draft_open": self.draft is not None,
                "review_run_id": self.draft.review_run_id if self.draft else None,
                "journal_cursor_event_seq": (
                    self.draft.journal.cursor.cursor_event_seq if self.draft else 0
                ),
                "redo_high_water_seq": (
                    self.draft.journal.cursor.redo_high_water_seq if self.draft else 0
                ),
                "n_active_events": (
                    len(self.draft.journal.active_events()) if self.draft else 0
                ),
                "can_undo": bool(self.draft and self.draft.journal.can_undo()),
                "can_redo": bool(self.draft and self.draft.journal.can_redo()),
                "changed_intervals": self.changed_intervals(),
                "n_reviewed_bouts": (
                    int(len(self.reviewed_bouts))
                    if self.reviewed_bouts is not None
                    else None
                ),
                "publication_state": (
                    self.draft.publication_state if self.draft else None
                ),
            }
        )
        return base

    # --- internals ----------------------------------------------------------

    def _require_editing_gate(self, *, require_draft: bool) -> None:
        if not self._editing_enabled:
            raise EditingError(
                "Editing mode is not active. Call enter_editing_mode(reviewer_id) first."
            )
        if not self.reviewer_id or not str(self.reviewer_id).strip():
            raise EditingError("Valid reviewer_id is required before writes.")
        if require_draft:
            if self.draft is None:
                raise EditingError(
                    "No working draft open. Call create_draft() or open_draft() first."
                )
            if not self.draft.lock.is_held:
                raise EditingError("Writer lock is not held for this draft.")

    def _make_event(self, op: str, **fields: Any) -> CorrectionEvent:
        assert self.reviewer_id is not None
        assert self.draft is not None
        seq = self.draft.journal.cursor.next_event_seq
        # Placeholder seq; _commit_events reassigns contiguous seqs for groups.
        return build_event(
            op=op,
            reviewer_id=self.reviewer_id,
            event_seq=seq,
            **fields,
        )

    def _commit_events(self, events: list[CorrectionEvent]) -> list[CorrectionEvent]:
        self._require_editing_gate(require_draft=True)
        assert self.draft is not None
        if not events:
            raise EditingError("No events to commit.")
        group = new_undo_group_id()
        seqs = self.draft.journal.allocate_seq_block(len(events))
        finalized: list[CorrectionEvent] = []
        for event, seq in zip(events, seqs, strict=True):
            event.event_seq = int(seq)
            event.undo_group_id = group
            validate_event(event, n_dense=self.n_dense_samples, allow_reciprocal=False)
            finalized.append(event)
        self.draft.journal.append_events(finalized)
        self.draft.working_manifest["lifecycle_status"] = RUN_LIFECYCLE_IN_PROGRESS
        self._rebuild_working_projection()
        self._assert_no_upstream_mutation()
        return finalized

    def _append_session_marker(self, marker: str) -> None:
        assert self.draft is not None
        assert self.reviewer_id is not None
        event = build_event(
            op=OP_SESSION_MARKER,
            reviewer_id=self.reviewer_id,
            event_seq=self.draft.journal.cursor.next_event_seq,
            payload_json=json.dumps({"marker": marker}, sort_keys=True),
        )
        validate_event(event, n_dense=self.n_dense_samples)
        self.draft.journal.append_events([event])

    def _load_baseline_states(self) -> None:
        path = self.inspection.source.source_s8.states_path
        if path is None or not path.is_file():
            raise EditingError("S8 behavior_states.parquet unavailable for editing.")
        self.baseline_states = pd.read_parquet(path)
        self.working_states = self.baseline_states.copy()
        self._upstream_fingerprints = snapshot_upstream_fingerprints(
            self.inspection.source
        )

    def _rebuild_working_projection(self) -> None:
        if self.baseline_states is None:
            self._load_baseline_states()
        assert self.baseline_states is not None
        active = self.draft.journal.active_events() if self.draft is not None else []
        source = self.inspection.source
        review_run_id = (
            self.draft.review_run_id if self.draft is not None else "ephemeral"
        )

        # Incremental replay: regenerate reviewed bouts after framewise events
        # (and before trim/split/merge) so compile ops can resolve targets.
        # Shared helper skips scientifically redundant regenerations.
        working, reviewed, changed, n_applied = replay_active_correction_chain(
            self.baseline_states,
            active,
            review_run_id=review_run_id,
            parent_package_run_id=source.package_run_id,
            source_s8_run_id=source.source_s8.run_id,
            source_s7_run_id=source.source_s7_run_id,
            source_s6_run_id=source.source_s6_run_id,
            automated_bouts=source.behavior_bouts,
            sync_index=source.sync_index,
        )

        self.working_states = working
        self.apply_meta = {
            "changed_intervals": changed,
            "n_events_applied": n_applied,
        }
        self.reviewed_bouts = reviewed
        self.overlay = replay_review_overlay(
            active,
            automated_bouts=source.behavior_bouts,
            reviewed_bouts=self.reviewed_bouts,
        )

    def _assert_no_upstream_mutation(self) -> None:
        source = self.inspection.source
        if not self._upstream_fingerprints:
            self._upstream_fingerprints = snapshot_upstream_fingerprints(source)
        assert_no_upstream_writes(
            package_dir=source.package_dir,
            s8_run_dir=source.source_s8.run_dir,
            video_path=source.source_video.path,
            snapshots=self._upstream_fingerprints,
        )


def open_editing_session(
    package_dir: Path | str,
    *,
    require_video: bool = False,
    require_s8_states: bool = True,
    initial_sample_idx: int = 0,
) -> EditingSession:
    """Open inspection-first session; editing requires explicit enter + draft."""

    inspection = open_inspection_session(
        package_dir,
        require_video=require_video,
        require_s8_states=require_s8_states,
        initial_sample_idx=initial_sample_idx,
        decode_initial_frame=False,
    )
    session = EditingSession(
        schema_version=EDITING_SESSION_SCHEMA_VERSION,
        inspection=inspection,
    )
    # Inspection-first: defer full baseline parquet load until enter_editing_mode
    # / draft create. Ethogram windows use EthogramAccessor column projection.
    del require_s8_states  # enforced by open_inspection_session / source open
    return session


def assert_inspection_cannot_write(session: EditingSession) -> None:
    """Helper for tests: inspection mode must reject writes."""

    if session.mode != MODE_INSPECTION:
        raise SessionError("Session is not in inspection mode.")
    try:
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="approach",
        )
    except EditingError:
        return
    raise SessionError("Inspection mode incorrectly allowed a write.")
