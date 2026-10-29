"""S10 nonblocking edit-worker lifecycle: serialization, stale rejection, durability."""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)

from behavioral_review import STATUS_ACCEPTED
from behavioral_review.source import assert_no_upstream_writes, snapshot_upstream_fingerprints
from ui.controllers.behavioral_review_controller import (
    EDIT_OP_CREATE_DRAFT,
    EDIT_OP_SET_EXCLUSIVE_LABEL,
    BehavioralReviewController,
    BehavioralReviewUiError,
    EditRequestIdentity,
    EditSnapshot,
    S9PackageOption,
)
from ui.pages.behavioral_review_page import BehavioralReviewPage
from ui.tasks import GuiTaskKind

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _patch_s9_discovery_eligible(source_s8_run_id: str):
    def _ok(run_dir: Path | str) -> dict[str, Any]:
        path = Path(run_dir).resolve()
        return {
            "s9_run_dir": path,
            "status": "completed",
            "session_root": str(path.parent.parent),
            "source_s8_run_dir": path.parent.parent
            / "behavioral_arbitration"
            / source_s8_run_id,
            "source_s8_run_id": source_s8_run_id,
        }

    return (
        patch(
            "results_packaging.runner.validate_s9_run_for_downstream",
            side_effect=_ok,
        ),
        patch(
            "results_packaging.runner.is_s9_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "results_packaging.publication.validate_s9_run_for_downstream",
            side_effect=_ok,
            raising=False,
        ),
        patch(
            "results_packaging.publication.is_s9_run_eligible_for_downstream",
            return_value=True,
            raising=False,
        ),
    )


def _controller_with_package(tmp_path: Path):
    built = build_mini_review_package(tmp_path)
    controller = BehavioralReviewController()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]), ExitStack() as stack:
        for p in _patch_s9_discovery_eligible(built["s8_run_id"]):
            stack.enter_context(p)
        identity = controller.prepare_session(built["session_root"])
        snap = controller.collect_discovery_snapshot(identity)
        assert controller.apply_discovery_snapshot(snap)
        controller.state.s9_options = tuple(
            S9PackageOption(
                run_id=o.run_id,
                run_dir=o.run_dir,
                historical_status=o.historical_status,
                classification="eligible",
                currently_eligible=True,
                n_bouts=o.n_bouts,
                n_dense_samples=o.n_dense_samples,
                label=o.label,
                reasons=(),
            )
            for o in controller.state.s9_options
        )
        controller.select_s9_package(built["package_run_id"])
        open_id = controller.begin_open_automated()
        open_snap = controller.open_source_worker(open_id)
        assert controller.apply_open_snapshot(open_snap)
    return controller, built


def _enter_and_draft(controller: BehavioralReviewController, review_root: Path) -> None:
    controller.set_reviewer_id("edit_worker_tester")
    controller.enter_editing_mode()
    controller.create_draft(review_root=review_root)


def test_nonblocking_edit_launch_begin_returns_promptly(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    controller.set_reviewer_id("edit_worker_tester")
    controller.enter_editing_mode()
    identity = controller.begin_create_draft(review_root=tmp_path / "async_drafts")
    assert identity.op == EDIT_OP_CREATE_DRAFT
    assert controller.state.edit_running
    assert controller.state.active_edit_identity == identity
    # Heavy work remains in the worker callable.
    snapshot = controller.edit_worker(identity)
    assert isinstance(snapshot, EditSnapshot)
    assert snapshot.draft_open
    assert controller.apply_edit_snapshot(snapshot)
    assert not controller.state.edit_running
    assert controller.state.draft_open


def test_mutation_serialization_and_duplicate_prevention(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "ser_drafts")
    first = controller.begin_set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
    )
    assert controller.state.edit_running
    with pytest.raises(BehavioralReviewUiError, match="already running"):
        controller.begin_set_exclusive_label(
            start_sample_idx=1, end_sample_idx=1, label="approach"
        )
    with pytest.raises(BehavioralReviewUiError, match="already running"):
        controller.begin_undo()
    with pytest.raises(BehavioralReviewUiError, match="edit is in progress"):
        controller.begin_publish()
    with pytest.raises(BehavioralReviewUiError, match="edit is in progress"):
        controller.begin_open_automated()
    # Finish first mutation cleanly.
    snap = controller.edit_worker(first)
    assert controller.apply_edit_snapshot(snap)
    assert controller.state.can_undo


def test_publication_edit_mutual_exclusion(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "pub_mutex")
    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="retract"
    )
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        preview = controller.build_publish_preview()
        if preview.blockers:
            with patch.object(
                controller,
                "build_publish_preview",
                return_value=preview.__class__(
                    parent_package_run_id=preview.parent_package_run_id,
                    parent_package_dir=preview.parent_package_dir,
                    reviewer_id=preview.reviewer_id,
                    review_run_id=preview.review_run_id,
                    corrected_intervals_text=preview.corrected_intervals_text,
                    status_counts=preview.status_counts,
                    blockers=(),
                    warnings=preview.warnings,
                    eligible=True,
                ),
            ):
                pub_id = controller.begin_publish()
        else:
            pub_id = controller.begin_publish()
        assert controller.state.publish_running
        with pytest.raises(BehavioralReviewUiError, match="publication"):
            controller.begin_set_exclusive_label(
                start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
            )
        # Cancel publish flags without claiming success.
        assert controller.fail_publish(pub_id, "aborted for mutex test")


def test_stale_callback_rejection_keeps_journal(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "stale_drafts")
    identity = controller.begin_set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
    )
    snapshot = controller.edit_worker(identity)
    # Simulate generation bump / superseded request before GUI callback.
    controller.state.edit_generation += 1
    controller.state.active_edit_identity = EditRequestIdentity(
        request_id="other",
        generation=controller.state.edit_generation,
        session_root=identity.session_root,
        package_run_id=identity.package_run_id,
        review_run_id=identity.review_run_id,
        op=EDIT_OP_SET_EXCLUSIVE_LABEL,
    )
    assert controller.apply_edit_snapshot(snapshot) is False
    # Journal commit from the worker remains durable on the session.
    assert controller.state.editing_session is not None
    assert controller.state.editing_session.draft is not None
    assert controller.state.editing_session.draft.journal.can_undo()
    # Clear stuck synthetic active request for cleanup.
    controller.state.active_edit_identity = None
    controller.state.edit_running = False
    controller._pending_edit = None
    controller.refresh_after_stale_edit()
    assert controller.state.can_undo


def test_navigation_blocked_during_edit(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "nav_drafts")
    identity = controller.begin_undo() if controller.state.can_undo else None
    if identity is None:
        identity = controller.begin_set_exclusive_label(
            start_sample_idx=0, end_sample_idx=0, label="approach"
        )
    with pytest.raises(BehavioralReviewUiError, match="edit is in progress"):
        controller.close_session(release_lock=True)
    with pytest.raises(BehavioralReviewUiError, match="edit is in progress"):
        controller.prepare_session(tmp_path)
    snap = controller.edit_worker(identity)
    assert controller.apply_edit_snapshot(snap)


def test_journal_durability_and_undo_redo(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    snaps = snapshot_upstream_fingerprints(
        controller.state.editing_session.inspection.source  # type: ignore[union-attr]
    )
    _enter_and_draft(controller, tmp_path / "durable_drafts")
    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=1, label="nose_to_body"
    )
    assert controller.state.can_undo
    controller.undo()
    assert controller.state.can_redo
    controller.redo()
    assert controller.state.can_undo
    run_dir = controller.state.review_run_dir
    assert run_dir is not None
    journal = run_dir / "working" / "correction_events.jsonl"
    assert journal.is_file()
    assert journal.stat().st_size > 0
    src = controller.state.editing_session.inspection.source  # type: ignore[union-attr]
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path if src.source_video else None,
        snapshots=snaps,
    )
    del built


def test_crash_recovery_after_committed_edit(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    # Default review root under the session so discovery can find the draft.
    controller.set_reviewer_id("edit_worker_tester")
    controller.enter_editing_mode()
    controller.create_draft()
    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="approach"
    )
    run_id = controller.state.review_run_id
    run_dir = controller.state.review_run_dir
    assert run_id and run_dir
    controller.close_session(release_lock=True)

    controller2 = BehavioralReviewController()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]), ExitStack() as stack:
        for p in _patch_s9_discovery_eligible(built["s8_run_id"]):
            stack.enter_context(p)
        identity = controller2.prepare_session(built["session_root"])
        snap = controller2.collect_discovery_snapshot(identity)
        assert controller2.apply_discovery_snapshot(snap)
        controller2.state.s9_options = tuple(
            S9PackageOption(
                run_id=o.run_id,
                run_dir=o.run_dir,
                historical_status=o.historical_status,
                classification="eligible",
                currently_eligible=True,
                n_bouts=o.n_bouts,
                n_dense_samples=o.n_dense_samples,
                label=o.label,
                reasons=(),
            )
            for o in controller2.state.s9_options
        )
        drafts = [o for o in controller2.state.review_options if o.run_id == run_id]
        assert drafts
        controller2.select_review_run(run_id)
        controller2.set_reviewer_id("edit_worker_tester")
        open_id = controller2.begin_open_draft()
        open_snap = controller2.open_source_worker(
            open_id, reviewer_id="edit_worker_tester", force_stale_lock=True
        )
        assert open_snap.draft_resumed
        assert controller2.apply_open_snapshot(
            open_snap, reviewer_id="edit_worker_tester", force_stale_lock=True
        )
    assert controller2.state.draft_open
    assert controller2.state.can_undo or controller2.state.changed_intervals_text


def test_gui_refresh_after_correction_updates_flags(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "refresh_drafts")
    win_id = controller.begin_window_request(half_window=20)
    controller.apply_window_snapshot(controller.window_worker(win_id))
    exclusive = [r for r in controller.state.bout_rows if r["channel"] == "exclusive"]
    assert exclusive
    controller.select_bout(exclusive[0]["bout_id"])
    before = controller.state.changed_intervals_text
    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
    )
    assert controller.state.can_undo
    assert controller.state.changed_intervals_text != before or "exclusive" in (
        controller.state.changed_intervals_text or ""
    )
    controller.set_bout_status(STATUS_ACCEPTED)
    assert controller.state.status_counts.get(STATUS_ACCEPTED, 0) >= 1


def test_page_uses_edit_task_kind(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage(controller)
    page.refresh_from_state()
    assert hasattr(page, "edit_runner")
    assert GuiTaskKind.BEHAVIORAL_REVIEW_EDIT.value == "behavioral_review_edit"
    page.reviewer_edit.setText("page_async")
    page._reviewer_changed()
    page.controller.set_reviewer_id("page_async")
    # Drive begin/worker/apply without hanging on QThread joins in offscreen CI.
    identity = page.controller.begin_enter_editing()
    snap = page.controller.edit_worker(identity)
    assert page.controller.apply_edit_snapshot(snap)
    identity = page.controller.begin_create_draft(review_root=tmp_path / "page_async")
    snap = page.controller.edit_worker(identity)
    assert page.controller.apply_edit_snapshot(snap)
    page.refresh_from_state()
    assert "EDITING" in page.mode_banner.text().upper()
    assert page.controller.state.draft_open
    # Competing correction blocked while a synthetic active edit is held.
    held = page.controller.begin_set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
    )
    assert not page._guard_edit_idle("apply exclusive label")
    snap = page.controller.edit_worker(held)
    assert page.controller.apply_edit_snapshot(snap)
    page.controller.close_session(release_lock=True)
    del built


def test_fail_edit_before_commit_leaves_clean_state(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "fail_drafts")
    session = controller.state.editing_session
    assert session is not None and session.draft is not None
    cursor_before = session.draft.journal.cursor.cursor_event_seq
    identity = controller.begin_set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
    )
    assert controller.fail_edit(identity, "simulated pre-commit failure")
    assert not controller.state.edit_running
    assert controller.state.run_error == "simulated pre-commit failure"
    # Worker never ran, so the journal cursor is unchanged.
    assert session.draft.journal.cursor.cursor_event_seq == cursor_before


def test_window_blocked_during_edit(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    _enter_and_draft(controller, tmp_path / "win_block")
    identity = controller.begin_set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
    )
    with pytest.raises(BehavioralReviewUiError, match="edit is in progress"):
        controller.begin_window_request()
    snap = controller.edit_worker(identity)
    assert controller.apply_edit_snapshot(snap)
    # After completion, windows are allowed again.
    win_id = controller.begin_window_request(half_window=10)
    assert controller.apply_window_snapshot(controller.window_worker(win_id))
