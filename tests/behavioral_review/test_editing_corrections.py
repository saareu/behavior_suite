"""S10 Sprint 3 — framewise corrections, status, journal, undo/redo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_review.contracts import (
    CHANNEL_CONTACT,
    CHANNEL_SIDE_BY_SIDE,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_VOIDED,
    EditingError,
    LockError,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.journal import EventJournal
from behavioral_review.locks import WriterLock
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)


def _open_editor(tmp_path: Path, **kwargs):
    built = build_mini_review_package(tmp_path, **kwargs)
    review_root = tmp_path / "review_drafts_isolated"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"],
            require_video=False,
            require_s8_states=True,
        )
    return built, session, review_root


def _enter_with_draft(session, review_root: Path, reviewer: str = "reviewer_a"):
    session.enter_editing_mode(reviewer)
    draft = session.create_draft(review_root=review_root)
    return draft


def test_inspection_only_prohibits_writes(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    with pytest.raises(EditingError, match="Editing mode is not active"):
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="nose_to_body",
        )
    # Seeking does not create drafts or journal events.
    session.seek_sample(3)
    assert session.draft is None
    assert not (review_root.exists() and any(review_root.iterdir()))
    src = session.inspection.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path,
        snapshots=snaps,
    )


def test_reviewer_identity_required(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    with pytest.raises(EditingError, match="reviewer_id"):
        session.enter_editing_mode("   ")
    session.enter_editing_mode("alice")
    with pytest.raises(EditingError, match="No working draft"):
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="approach",
        )
    _enter_with_draft(session, review_root, reviewer="alice")


def test_exclusive_directional_correction_no_reciprocal(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=3,
        label="nose_to_nose",
    )
    states = session.current_reviewed_states()
    a = states[(states["focal_id"] == 0) & (states["partner_id"] == 1)]
    b = states[(states["focal_id"] == 1) & (states["partner_id"] == 0)]
    assert list(a.sort_values("dense_frame_idx")["behavior_label"].iloc[0:4]) == [
        "nose_to_nose"
    ] * 4
    # Opposite direction untouched (baseline retract on 0..3).
    assert list(b.sort_values("dense_frame_idx")["behavior_label"].iloc[0:4]) == [
        "retract"
    ] * 4
    src = session.inspection.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path,
        snapshots=snaps,
    )
    # Automated packaged bouts unchanged.
    auto = pd.read_parquet(built["package_dir"] / "behavior_bouts.parquet")
    assert len(auto) == len(built["bouts"])


def test_contact_and_sbs_independent(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    # Clear contact on [2,7]; SBS untouched.
    session.set_parallel_pred(
        channel=CHANNEL_CONTACT,
        mouse_a_id=0,
        mouse_b_id=1,
        start_sample_idx=2,
        end_sample_idx=7,
        pred_value=0,
    )
    states = session.current_reviewed_states()
    window = states[(states["dense_frame_idx"] >= 2) & (states["dense_frame_idx"] <= 7)]
    assert int(window["pred_contact"].sum()) == 0
    # SBS positives remain on last two samples of the mini package.
    sbs = states[states["dense_frame_idx"] >= 8]
    assert int(sbs["pred_side_by_side"].max()) == 1
    # Exclusive labels unchanged by contact edit.
    baseline = session.baseline_states
    assert list(states["behavior_label"]) == list(baseline["behavior_label"])


def test_inclusive_interval_boundaries_rejected_when_invalid(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    with pytest.raises(EditingError, match="empty/inverted|Illegal"):
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=5,
            end_sample_idx=4,
            label="approach",
        )
    with pytest.raises(EditingError, match="outside dense range"):
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=10_000,
            label="approach",
        )
    # Inclusive single-sample write.
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=5,
        end_sample_idx=5,
        label="nose_to_tail",
    )
    states = session.current_reviewed_states()
    hit = states[
        (states["focal_id"] == 0)
        & (states["partner_id"] == 1)
        & (states["dense_frame_idx"] == 5)
    ]
    assert str(hit.iloc[0]["behavior_label"]) == "nose_to_tail"


def test_bout_trim_split_merge_compilation(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    # Trim approach bout_dir_a [0,3] -> [1,3]
    session.trim_bout_boundary(
        target_bout_id="bout_dir_a",
        new_start_sample_idx=1,
        new_end_sample_idx=3,
        fill_label="mutually_disengaged",
    )
    bouts = session.current_reviewed_bouts()
    excl = bouts[
        (bouts["channel"] == "exclusive")
        & (bouts["focal_id"] == 0)
        & (bouts["partner_id"] == 1)
    ]
    approach = excl[excl["behavior_name"] == "approach"]
    assert len(approach) == 1
    assert int(approach.iloc[0]["start_sample_idx"]) == 1
    assert int(approach.iloc[0]["end_sample_idx"]) == 3
    assert int(approach.iloc[0]["bout_duration_frames"]) == 3
    assert float(approach.iloc[0]["bout_duration_sec"]) == pytest.approx(3 / 120.0)

    # Split remaining approach at sample 3 with right_label retract.
    session.split_bout_at_sample(
        target_bout_id=str(approach.iloc[0]["bout_id"]),
        at_sample_idx=3,
        right_label="retract",
        target_bout_id_kind="reviewed_bout_id",
    )
    bouts2 = session.current_reviewed_bouts()
    excl2 = bouts2[
        (bouts2["channel"] == "exclusive")
        & (bouts2["focal_id"] == 0)
        & (bouts2["partner_id"] == 1)
        & (bouts2["start_sample_idx"] >= 1)
        & (bouts2["end_sample_idx"] <= 3)
    ]
    names = list(excl2.sort_values("start_sample_idx")["behavior_name"])
    assert "approach" in names and "retract" in names

    # Merge the two small segments back by painting shared label.
    left = excl2.sort_values("start_sample_idx").iloc[0]
    right = excl2.sort_values("start_sample_idx").iloc[-1]
    session.merge_adjacent_bouts(
        bout_id_a=str(left["bout_id"]),
        bout_id_b=str(right["bout_id"]),
        label="approach",
        bout_id_kind="reviewed_bout_id",
    )
    merged = session.current_reviewed_bouts()
    row = merged[
        (merged["channel"] == "exclusive")
        & (merged["focal_id"] == 0)
        & (merged["behavior_name"] == "approach")
        & (merged["start_sample_idx"] == 1)
    ]
    assert len(row) == 1
    assert int(row.iloc[0]["end_sample_idx"]) == 3


def test_pure_s8_collapse_parity_on_unmodified_baseline(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    baseline = session.baseline_states.copy()
    if "exp_name" not in baseline.columns:
        baseline["exp_name"] = "synth"
    s8 = extract_behavior_bouts(
        baseline,
        fps=120.0,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    reviewed = session.current_reviewed_bouts()
    # Same geometry counts / inclusive endpoints / durations.
    assert len(reviewed) == len(s8)
    for ch in ("exclusive", "contact", "side_by_side"):
        a = s8[s8["channel"] == ch].sort_values(
            ["start_sample_idx", "behavior_name", "focal_id", "partner_id"]
        )
        b = reviewed[reviewed["channel"] == ch].sort_values(
            ["start_sample_idx", "behavior_name", "focal_id", "partner_id"]
        )
        assert list(a["start_sample_idx"]) == list(b["start_sample_idx"])
        assert list(a["end_sample_idx"]) == list(b["end_sample_idx"])
        assert list(a["bout_duration_frames"]) == list(b["bout_duration_frames"])
        assert list(a["behavior_name"]) == list(b["behavior_name"])


def test_reviewed_bout_identity_not_reused_when_geometry_changes(
    tmp_path: Path,
) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    before = session.current_reviewed_bouts()
    src = before[before["source_s8_bout_id"] == "bout_dir_a"]
    assert len(src) == 1
    old_id = str(src.iloc[0]["bout_id"])
    assert old_id.startswith("rb__")
    assert "bout_dir_a" not in old_id or old_id != "bout_dir_a"
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=1,
        label="nose_to_body",
    )
    after = session.current_reviewed_bouts()
    # Original exact lineage interval gone; new bouts get new ids.
    assert old_id not in set(after["bout_id"].astype(str))
    changed = after[
        (after["channel"] == "exclusive")
        & (after["focal_id"] == 0)
        & (after["start_sample_idx"] == 0)
    ]
    assert len(changed) >= 1
    assert pd.isna(changed.iloc[0]["source_s8_bout_id"]) or str(
        changed.iloc[0]["source_s8_bout_id"]
    ) != "bout_dir_a"


def test_status_only_rejection_and_notes_without_correction(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    before_labels = session.current_reviewed_states()["behavior_label"].tolist()
    session.set_bout_review_status(
        target_bout_id="bout_dir_a",
        status=STATUS_REJECTED,
    )
    session.add_note(
        note_text="timing looks early",
        target_bout_id="bout_dir_a",
        target_bout_id_kind="source_s8_bout_id",
    )
    after_labels = session.current_reviewed_states()["behavior_label"].tolist()
    assert before_labels == after_labels
    assert session.overlay.status_for_source_bout("bout_dir_a") == STATUS_REJECTED
    notes = session.current_notes()
    assert len(notes) == 1
    assert "timing" in str(notes.iloc[0]["note_text"])


def test_acceptance_voided_after_geometry_change_stable_unaffected(
    tmp_path: Path,
) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    session.set_bout_review_status(
        target_bout_id="bout_dir_b", status=STATUS_ACCEPTED
    )
    assert session.overlay.status_for_source_bout("bout_dir_a") == STATUS_ACCEPTED
    assert session.overlay.status_for_source_bout("bout_dir_b") == STATUS_ACCEPTED
    # Geometry change on direction 0→1 only.
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=3,
        label="nose_to_tail",
    )
    assert session.overlay.status_for_source_bout("bout_dir_a") == STATUS_VOIDED
    # Opposite-direction acceptance remains.
    assert session.overlay.status_for_source_bout("bout_dir_b") == STATUS_ACCEPTED
    hist = session.current_statuses()
    assert STATUS_ACCEPTED in set(hist["status"])
    assert STATUS_VOIDED in set(hist["status"])


def test_undo_redo_across_restart_and_new_edit_after_undo(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root, reviewer="alice")
    run_dir = draft.review_run_dir
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=0,
        label="nose_to_nose",
    )
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=1,
        end_sample_idx=1,
        label="nose_to_body",
    )
    assert session.draft.journal.can_undo()
    session.undo()
    states_mid = session.current_reviewed_states()
    mid = states_mid[
        (states_mid["focal_id"] == 0)
        & (states_mid["partner_id"] == 1)
        & (states_mid["dense_frame_idx"] == 1)
    ].iloc[0]["behavior_label"]
    assert str(mid) != "nose_to_body"

    # Restart: reopen draft and verify exact recovery at cursor.
    session.close_draft()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session2 = open_editing_session(
            built["package_dir"], require_video=False, require_s8_states=True
        )
    session2.enter_editing_mode("alice")
    session2.open_draft(run_dir)
    states_re = session2.current_reviewed_states()
    lab0 = states_re[
        (states_re["focal_id"] == 0)
        & (states_re["partner_id"] == 1)
        & (states_re["dense_frame_idx"] == 0)
    ].iloc[0]["behavior_label"]
    lab1 = states_re[
        (states_re["focal_id"] == 0)
        & (states_re["partner_id"] == 1)
        & (states_re["dense_frame_idx"] == 1)
    ].iloc[0]["behavior_label"]
    assert str(lab0) == "nose_to_nose"
    assert str(lab1) != "nose_to_body"
    assert session2.draft.journal.can_redo()
    session2.redo()
    lab1b = session2.current_reviewed_states()[
        (session2.current_reviewed_states()["focal_id"] == 0)
        & (session2.current_reviewed_states()["partner_id"] == 1)
        & (session2.current_reviewed_states()["dense_frame_idx"] == 1)
    ].iloc[0]["behavior_label"]
    assert str(lab1b) == "nose_to_body"

    # New edit after undo discards redo branch logically; ids remain stable.
    session2.undo()
    old_events = list(session2.draft.journal.events)
    old_ids = [e.event_id for e in old_events]
    session2.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=2,
        end_sample_idx=2,
        label="engaged",
    )
    # Historical events retained.
    assert [e.event_id for e in session2.draft.journal.events[: len(old_ids)]] == old_ids
    assert not session2.draft.journal.can_redo()
    # Superseded seqs recorded.
    assert session2.draft.journal.cursor.superseded_event_seqs
    lab2 = session2.current_reviewed_states()[
        (session2.current_reviewed_states()["focal_id"] == 0)
        & (session2.current_reviewed_states()["partner_id"] == 1)
        & (session2.current_reviewed_states()["dense_frame_idx"] == 2)
    ].iloc[0]["behavior_label"]
    assert str(lab2) == "engaged"


def test_journal_crash_recovery_truncates_partial_line(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root)
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=0,
        label="approach",
    )
    path = draft.journal.events_path
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id":"evt_partial","event_seq":')
        handle.flush()
        os.fsync(handle.fileno())
    # Corrupt cursor beyond journal.
    cursor_path = draft.journal.cursor_path
    cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
    cursor["cursor_event_seq"] = 999
    cursor_path.write_text(json.dumps(cursor), encoding="utf-8")

    recovered = EventJournal.open(draft.working_dir)
    assert recovered.recovery_warnings
    assert recovered.cursor.cursor_event_seq <= (
        recovered.events[-1].event_seq if recovered.events else 0
    )
    assert all(e.event_id != "evt_partial" for e in recovered.events)


def test_concurrent_writer_rejection_and_stale_lock_recovery(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root, reviewer="alice")
    # Second editor on same draft rejected.
    lock2 = WriterLock(draft.working_dir, review_run_id=draft.review_run_id)
    with pytest.raises(LockError, match="locked by another writer"):
        lock2.acquire(reviewer_id="bob")

    # Simulate stale lock: dead pid.
    lock_path = draft.lock.lock_path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["pid"] = 1_999_999_999  # unlikely live
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    session.draft.lock._held = False  # type: ignore[attr-defined]

    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session3 = open_editing_session(
            built["package_dir"], require_video=False, require_s8_states=True
        )
    session3.enter_editing_mode("bob")
    with pytest.raises(LockError, match="Stale writer lock"):
        session3.open_draft(draft.review_run_dir, force_stale_lock=False)
    session3.open_draft(draft.review_run_dir, force_stale_lock=True)
    assert session3.draft is not None
    assert session3.draft.lock.is_held


def test_deterministic_event_replay(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=2,
        label="nose_to_nose",
    )
    session.set_parallel_pred(
        channel=CHANNEL_SIDE_BY_SIDE,
        mouse_a_id=0,
        mouse_b_id=1,
        start_sample_idx=0,
        end_sample_idx=1,
        pred_value=1,
    )
    states_a = session.current_reviewed_states().copy()
    bouts_a = session.current_reviewed_bouts().copy()
    # Rebuild from journal only.
    session._rebuild_working_projection()
    states_b = session.current_reviewed_states()
    bouts_b = session.current_reviewed_bouts()
    assert list(states_a["behavior_label"]) == list(states_b["behavior_label"])
    assert list(states_a["pred_side_by_side"]) == list(states_b["pred_side_by_side"])
    assert list(bouts_a["bout_id"]) == list(bouts_b["bout_id"])
    assert list(bouts_a["start_sample_idx"]) == list(bouts_b["start_sample_idx"])


def test_compare_and_changed_intervals_api(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=4,
        end_sample_idx=6,
        label="approach",
    )
    changed = session.changed_intervals()
    assert "exclusive" in changed
    assert changed["exclusive"][0][0] <= 4
    diff = session.compare_automated_versus_reviewed(
        start_sample_idx=0, end_sample_idx=9
    )
    assert len(diff) > 0
    # Seek/inspect updated results without new events.
    n_events = len(session.draft.journal.events)
    session.seek_sample(5)
    assert len(session.draft.journal.events) == n_events
