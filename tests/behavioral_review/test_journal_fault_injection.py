"""Fault-injection coverage for journal commit boundary and recovery."""

from __future__ import annotations

import os
from pathlib import Path

from behavioral_review.contracts import (
    BOUT_ID_KIND_SOURCE_S8,
    JOURNAL_RECOVERY_BACKUPS_DIRNAME,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.events import OP_SET_BOUT_REVIEW_STATUS, build_event
from behavioral_review.journal import EventJournal
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)


def _open_draft(tmp_path: Path, reviewer: str = "fault"):
    built = build_mini_review_package(tmp_path)
    review_root = tmp_path / "isolated_drafts"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"],
            require_video=False,
            require_s8_states=True,
        )
    session.enter_editing_mode(reviewer)
    draft = session.create_draft(review_root=review_root)
    return built, session, draft


def _status_events(
    *,
    reviewer: str,
    seqs: list[int],
    undo_group_id: str,
    targets: list[tuple[str, str]],
):
    events = []
    for i, seq in enumerate(seqs):
        bout_id, status = targets[i]
        events.append(
            build_event(
                op=OP_SET_BOUT_REVIEW_STATUS,
                reviewer_id=reviewer,
                event_seq=seq,
                target_bout_id=bout_id,
                target_bout_id_kind=BOUT_ID_KIND_SOURCE_S8,
                status=status,
                undo_group_id=undo_group_id,
                client_op_id=f"{undo_group_id}:{i}",
            )
        )
    return events


def _append_raw(journal: EventJournal, events) -> None:
    with journal.events_path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(event.to_json_line())
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_backup_immutable(working_dir: Path) -> Path:
    root = working_dir / JOURNAL_RECOVERY_BACKUPS_DIRNAME
    assert root.is_dir()
    backups = sorted(p for p in root.iterdir() if p.is_dir())
    assert backups
    latest = backups[-1]
    assert (latest / "correction_events.jsonl").is_file()
    assert (latest / "journal_cursor.json").is_file()
    assert (latest / "recovery_reason.txt").is_file()
    return latest


def test_interrupt_before_any_bulk_event(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    # Crash before writing any bulk lines: allocate only.
    seqs = draft.journal.allocate_seq_block(2)
    assert seqs == [committed_hw + 1, committed_hw + 2]

    recovered = EventJournal.open(draft.working_dir)
    assert recovered.recovery_warnings == []
    assert [e.event_seq for e in recovered.events] == list(range(1, committed_hw + 1))
    assert recovered.cursor.redo_high_water_seq == committed_hw
    assert recovered.cursor.next_event_seq == committed_hw + 1
    session.close_draft(release_lock=True)


def test_interrupt_after_one_fsynced_event(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    seqs = draft.journal.allocate_seq_block(2)
    orphans = _status_events(
        reviewer="fault",
        seqs=seqs[:1],
        undo_group_id="bulk-one",
        targets=[("bout_dir_b", STATUS_ACCEPTED)],
    )
    _append_raw(draft.journal, orphans)

    recovered = EventJournal.open(draft.working_dir)
    assert any("interrupted append" in w for w in recovered.recovery_warnings)
    assert [e.event_seq for e in recovered.events] == list(range(1, committed_hw + 1))
    assert recovered.cursor.cursor_event_seq == committed_hw
    assert recovered.cursor.next_event_seq == committed_hw + 1
    _assert_backup_immutable(draft.working_dir)
    session.close_draft(release_lock=True)


def test_interrupt_after_partial_bulk_before_cursor(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    seqs = draft.journal.allocate_seq_block(3)
    orphans = _status_events(
        reviewer="fault",
        seqs=seqs[:2],
        undo_group_id="bulk-partial",
        targets=[
            ("bout_dir_b", STATUS_ACCEPTED),
            ("bout_dir_a", STATUS_REJECTED),
        ],
    )
    _append_raw(draft.journal, orphans)

    recovered = EventJournal.open(draft.working_dir)
    assert len(recovered.events) == committed_hw
    assert recovered.cursor.redo_high_water_seq == committed_hw
    assert not recovered.can_redo()
    session.close_draft(release_lock=True)


def test_interrupt_after_all_bulk_events_before_cursor(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    seqs = draft.journal.allocate_seq_block(2)
    orphans = _status_events(
        reviewer="fault",
        seqs=seqs,
        undo_group_id="bulk-full",
        targets=[
            ("bout_dir_b", STATUS_ACCEPTED),
            ("bout_dir_a", STATUS_REJECTED),
        ],
    )
    _append_raw(draft.journal, orphans)

    recovered = EventJournal.open(draft.working_dir)
    assert [e.event_seq for e in recovered.events] == list(range(1, committed_hw + 1))
    assert recovered.cursor.next_event_seq == committed_hw + 1
    # Contiguous allocation after recovery; undo group remains atomic.
    session.close_draft(release_lock=True)
    with patch_s9_currently_eligible(
        source_s8_run_id=session.inspection.source.source_s8.run_id
    ):
        session2 = open_editing_session(
            session.inspection.source.package_dir,
            require_video=False,
            require_s8_states=True,
        )
    session2.enter_editing_mode("fault")
    session2.open_draft(draft.review_run_dir)
    events = session2.set_bout_review_statuses(
        [
            {"target_bout_id": "bout_dir_b", "status": STATUS_ACCEPTED},
            {"target_bout_id": "bout_dir_a", "status": STATUS_REJECTED},
        ],
        client_op_id="after-full-interrupt",
    )
    assert [e.event_seq for e in events] == [committed_hw + 1, committed_hw + 2]
    assert len({e.undo_group_id for e in events}) == 1
    session2.close_draft(release_lock=True)


def test_interrupt_after_successful_cursor_commit(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    events = session.set_bout_review_statuses(
        [
            {"target_bout_id": "bout_dir_a", "status": STATUS_ACCEPTED},
            {"target_bout_id": "bout_dir_b", "status": STATUS_REJECTED},
        ],
        client_op_id="committed-bulk",
    )
    assert len(events) == 2
    hw = draft.journal.cursor.redo_high_water_seq
    active_before = [e.event_seq for e in draft.journal.active_events()]
    recovered = EventJournal.open(draft.working_dir)
    assert recovered.recovery_warnings == []
    assert recovered.cursor.redo_high_water_seq == hw
    assert [e.event_seq for e in recovered.active_events()] == active_before
    assert recovered.cursor.cursor_event_seq == hw
    session.close_draft(release_lock=True)


def test_interrupt_with_legitimate_redo_history_preserved(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    session.set_bout_review_status(
        target_bout_id="bout_dir_b", status=STATUS_REJECTED
    )
    session.undo()
    assert draft.journal.can_redo()
    cursor_seq = draft.journal.cursor.cursor_event_seq
    redo_hw = draft.journal.cursor.redo_high_water_seq
    assert cursor_seq < redo_hw

    seqs = draft.journal.allocate_seq_block(2)
    orphans = _status_events(
        reviewer="fault",
        seqs=seqs,
        undo_group_id="after-undo-bulk",
        targets=[
            ("bout_dir_a", STATUS_REJECTED),
            ("bout_dir_b", STATUS_ACCEPTED),
        ],
    )
    _append_raw(draft.journal, orphans)

    recovered = EventJournal.open(draft.working_dir)
    assert recovered.cursor.cursor_event_seq == cursor_seq
    assert recovered.cursor.redo_high_water_seq == redo_hw
    assert recovered.can_redo()
    assert [e.event_seq for e in recovered.events] == list(range(1, redo_hw + 1))
    redone = recovered.redo_group()
    assert len(redone) == 1
    assert recovered.cursor.cursor_event_seq == redo_hw
    session.close_draft(release_lock=True)


def test_interrupt_during_branch_after_redo_then_new_correction(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    session.set_bout_review_status(
        target_bout_id="bout_dir_b", status=STATUS_REJECTED
    )
    session.undo()
    session.redo()
    assert not draft.journal.can_redo()
    hw_before = draft.journal.cursor.redo_high_water_seq

    # Simulate crash after undo + new correction lines before cursor commit.
    session.undo()
    seqs = draft.journal.allocate_seq_block(1)
    orphans = _status_events(
        reviewer="fault",
        seqs=seqs,
        undo_group_id="post-redo-correction",
        targets=[("bout_dir_a", STATUS_REJECTED)],
    )
    _append_raw(draft.journal, orphans)

    recovered = EventJournal.open(draft.working_dir)
    assert recovered.cursor.redo_high_water_seq == hw_before
    assert recovered.can_redo()
    assert all(e.event_seq <= hw_before for e in recovered.events)
    # Historical redo events remain present (not deleted).
    assert len(recovered.events) == hw_before
    session.close_draft(release_lock=True)


def test_recovery_repair_is_idempotent_and_backs_up(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    seqs = draft.journal.allocate_seq_block(2)
    orphans = _status_events(
        reviewer="fault",
        seqs=seqs,
        undo_group_id="idempotent-bulk",
        targets=[
            ("bout_dir_b", STATUS_ACCEPTED),
            ("bout_dir_a", STATUS_REJECTED),
        ],
    )
    _append_raw(draft.journal, orphans)

    first = EventJournal.open(draft.working_dir)
    backup = _assert_backup_immutable(draft.working_dir)
    first_events = [e.to_dict() for e in first.events]
    first_cursor = first.cursor.to_dict()
    assert any("interrupted append" in w for w in first.recovery_warnings)

    second = EventJournal.open(draft.working_dir)
    assert [e.to_dict() for e in second.events] == first_events
    assert second.cursor.to_dict() == first_cursor
    assert second.recovery_warnings == []
    # Prior backup directory remains untouched (immutable).
    assert backup.is_dir()
    bak_events = (backup / "correction_events.jsonl").read_text(encoding="utf-8")
    assert f'"event_seq":{committed_hw + 1}' in bak_events.replace(" ", "")
    session.close_draft(release_lock=True)


def test_corrupt_cursor_uses_backup_commit_boundary(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    # First create a recovery backup via an interrupted append repair.
    seqs = draft.journal.allocate_seq_block(1)
    _append_raw(
        draft.journal,
        _status_events(
            reviewer="fault",
            seqs=seqs,
            undo_group_id="seed-backup",
            targets=[("bout_dir_b", STATUS_ACCEPTED)],
        ),
    )
    repaired = EventJournal.open(draft.working_dir)
    assert repaired.cursor.redo_high_water_seq == committed_hw
    _assert_backup_immutable(draft.working_dir)

    # Interrupt again, then corrupt the primary cursor.
    seqs2 = repaired.allocate_seq_block(2)
    _append_raw(
        repaired,
        _status_events(
            reviewer="fault",
            seqs=seqs2,
            undo_group_id="after-backup",
            targets=[
                ("bout_dir_b", STATUS_ACCEPTED),
                ("bout_dir_a", STATUS_REJECTED),
            ],
        ),
    )
    repaired.cursor_path.write_text("{not-json", encoding="utf-8")

    recovered = EventJournal.open(draft.working_dir)
    assert any("recovery backup" in w for w in recovered.recovery_warnings)
    assert recovered.cursor.redo_high_water_seq == committed_hw
    assert [e.event_seq for e in recovered.events] == list(range(1, committed_hw + 1))
    session.close_draft(release_lock=True)


def test_partial_line_and_orphan_recovery_keeps_committed(tmp_path: Path) -> None:
    _, session, draft = _open_draft(tmp_path)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    committed_hw = draft.journal.cursor.redo_high_water_seq
    seqs = draft.journal.allocate_seq_block(1)
    _append_raw(
        draft.journal,
        _status_events(
            reviewer="fault",
            seqs=seqs,
            undo_group_id="orphan-then-partial",
            targets=[("bout_dir_b", STATUS_ACCEPTED)],
        ),
    )
    with draft.journal.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id":"evt_partial","event_seq":')
        handle.flush()
        os.fsync(handle.fileno())

    recovered = EventJournal.open(draft.working_dir)
    assert any("Truncated incomplete" in w for w in recovered.recovery_warnings)
    assert any("interrupted append" in w for w in recovered.recovery_warnings)
    assert [e.event_seq for e in recovered.events] == list(range(1, committed_hw + 1))
    assert all(e.event_id != "evt_partial" for e in recovered.events)
    session.close_draft(release_lock=True)
