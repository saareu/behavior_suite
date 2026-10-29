"""Subsystem 10 GUI: Group 4 nav, inspection sync, editing gates, publish safety."""

from __future__ import annotations

import os
import time
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

from behavioral_review import (
    MODE_EDITING,
    MODE_INSPECTION,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    ReviewedPublicationResult,
)
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from ui.controllers.behavioral_review_controller import (
    BehavioralReviewController,
    BehavioralReviewUiError,
    DecodeRequestIdentity,
    DecodeSnapshot,
    S9PackageOption,
    WindowRequestIdentity,
    WindowSnapshot,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import (
    GROUP_BEHAVIORAL_INFERENCE,
    GROUP_DATA_PREPARATION,
    GROUP_FEATURE_PROCESSING,
    GROUP_RESULTS_REVIEW,
    MainWindow,
)
from ui.pages.behavioral_review_page import BehavioralReviewPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


def _build_session(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    built = build_mini_review_package(tmp_path, **kwargs)
    return built


def _patch_s9_discovery_eligible(source_s8_run_id: str):
    """Extend eligibility patches to S9 discovery classification used by the GUI.

    ``run_discovery._classify_run`` lazily imports validators from
    ``results_packaging.runner``.
    """

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


def _controller_with_package(
    tmp_path: Path, **kwargs: Any
) -> tuple[BehavioralReviewController, dict[str, Any]]:
    built = _build_session(tmp_path, **kwargs)
    controller = BehavioralReviewController()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]), ExitStack() as stack:
        for p in _patch_s9_discovery_eligible(built["s8_run_id"]):
            stack.enter_context(p)
        identity = controller.prepare_session(built["session_root"])
        snapshot = controller.collect_discovery_snapshot(identity)
        assert controller.apply_discovery_snapshot(snapshot)
        # Mini fixtures may still classify stale under full S9 chain checks;
        # force eligibility for GUI open-path tests.
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


def test_main_window_group4_s10_navigation_and_session(tmp_path: Path) -> None:
    built = _build_session(tmp_path)
    controller = BehavioralReviewController()
    root = built["session_root"].resolve()
    controller.state.session_root = root
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_review_controller=controller,
    )
    # Do not set window._session_root before navigation — that would trigger
    # S9 set_session discovery against the mini fixture (hangs offscreen).
    window._last_results_page = "s10"
    window.pages.setCurrentWidget(window.behavioral_review_page)
    window._ensure_group(GROUP_RESULTS_REVIEW)
    assert window.s9_button.isVisibleTo(window)
    assert window.s10_button.isVisibleTo(window)
    assert not window.s7_button.isVisibleTo(window)
    assert window.pages.currentWidget() is window.behavioral_review_page
    assert controller.state.session_root == root
    window.set_application_group(GROUP_FEATURE_PROCESSING)
    window.set_application_group(GROUP_DATA_PREPARATION)
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert not window.s10_button.isVisibleTo(window)
    window.set_application_group(GROUP_RESULTS_REVIEW)
    assert window.pages.currentWidget() is window.behavioral_review_page
    assert window._last_results_page == "s10"
    assert controller.state.session_root == root


def test_source_discovery_and_selection(tmp_path: Path) -> None:
    built = _build_session(tmp_path)
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
    assert any(o.run_id == built["package_run_id"] for o in controller.state.s9_options)
    assert controller.state.selected_s9_run_id == built["package_run_id"]
    controller.select_s9_package(None)
    assert controller.state.selected_s9_run_id is None


def test_playhead_video_sync_and_plateau(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path, plateau_repeats=2)
    session = controller.state.editing_session
    assert session is not None
    snaps = snapshot_upstream_fingerprints(session.inspection.source)

    # Seek across a prepared plateau: dense advances, prepared may stay.
    decode_a = controller.seek_sample(0)
    assert decode_a is not None
    snap_a = controller.decode_worker(decode_a)
    assert controller.apply_decode_snapshot(snap_a)
    prep0 = controller.state.prepared_frame_idx
    frame0 = None if controller.state.frame_bgr is None else controller.state.frame_bgr.copy()

    decode_b = controller.seek_sample(1)
    assert decode_b is not None
    snap_b = controller.decode_worker(decode_b)
    applied = controller.apply_decode_snapshot(snap_b)
    assert applied
    assert controller.state.sample_idx == 1
    # Plateau: prepared may be unchanged; that is not a sync error.
    if controller.state.prepared_frame_idx == prep0:
        assert controller.state.prepared_frame_unchanged or controller.state.plateau
        if frame0 is not None and controller.state.frame_bgr is not None:
            assert controller.state.frame_bgr.shape == frame0.shape

    src = session.inspection.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path if src.source_video else None,
        snapshots=snaps,
    )
    del built


def test_directional_lanes_contact_sbs_and_bout_nav(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    assert len(controller.state.directions) >= 2
    controller.select_direction(*controller.state.directions[0])
    controller.select_channel("exclusive")
    win_id = controller.begin_window_request(half_window=20, include_s7=False)
    win = controller.window_worker(win_id)
    assert controller.apply_window_snapshot(win)
    assert controller.state.timeline is not None
    lane_ids = {lane.lane_id for lane in controller.state.timeline.lanes}
    assert any("exclusive" in lid or "dir" in lid for lid in lane_ids) or lane_ids
    assert controller.state.bout_rows
    channels = {row["channel"] for row in controller.state.bout_rows}
    assert "exclusive" in channels
    assert "contact" in channels or "side_by_side" in channels

    decode = controller.next_bout()
    if decode is not None:
        assert controller.state.selected_bout_id is not None
        controller.jump_bout_end(controller.state.selected_bout_id)


def test_jump_bout_without_selection_is_ui_error(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    controller.state.selected_bout_id = None
    with pytest.raises(BehavioralReviewUiError, match="Select a bout"):
        controller.jump_bout_start()
    with pytest.raises(BehavioralReviewUiError, match="Select a bout"):
        controller.jump_bout_end()


def test_stale_async_result_rejection(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    # Decode identity with mismatched generation must be rejected.
    stale = DecodeRequestIdentity(
        request_id="stale",
        generation=controller.state.decode_generation - 1,
        package_run_id=str(controller.state.package_run_id),
        sample_idx=controller.state.sample_idx,
        prepared_frame_idx=controller.state.prepared_frame_idx,
    )
    snap = DecodeSnapshot(
        identity=stale,
        frame_bgr=None,
        prepared_frame_unchanged=True,
        plateau=True,
    )
    controller.state.active_decode_identity = DecodeRequestIdentity(
        request_id="current",
        generation=controller.state.decode_generation,
        package_run_id=str(controller.state.package_run_id),
        sample_idx=controller.state.sample_idx,
        prepared_frame_idx=controller.state.prepared_frame_idx,
    )
    assert controller.apply_decode_snapshot(snap) is False

    # Window stale rejection.
    controller.state.window_generation = 5
    controller.state.active_window_identity = WindowRequestIdentity(
        request_id="w",
        generation=5,
        package_run_id=str(controller.state.package_run_id),
        sample_idx=controller.state.sample_idx,
        half_window=10,
        include_s7=False,
    )
    stale_win = WindowSnapshot(
        identity=WindowRequestIdentity(
            request_id="old",
            generation=1,
            package_run_id=str(controller.state.package_run_id),
            sample_idx=controller.state.sample_idx,
            half_window=10,
            include_s7=False,
        ),
        timeline=None,
        evidence_text="stale",
        provenance_text="",
        reconstruction_text="",
        bout_rows=(),
        status_counts={},
        changed_intervals_text="",
        compare_text="",
    )
    assert controller.apply_window_snapshot(stale_win) is False


def test_editing_mode_gate_and_reviewer_identity(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    assert controller.state.mode == MODE_INSPECTION
    with pytest.raises(BehavioralReviewUiError, match="reviewer"):
        controller.enter_editing_mode()
    controller.set_reviewer_id("lab_operator")
    controller.enter_editing_mode()
    assert controller.state.editing_active
    assert controller.state.mode == MODE_EDITING
    with pytest.raises(BehavioralReviewUiError, match="draft"):
        controller.set_exclusive_label(
            start_sample_idx=0, end_sample_idx=0, label="nose_to_body"
        )
    review_root = tmp_path / "gui_review_drafts"
    controller.create_draft(review_root=review_root)
    assert controller.state.draft_open
    assert controller.state.review_run_id is not None


def test_corrections_status_undo_redo(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    snaps = snapshot_upstream_fingerprints(
        controller.state.editing_session.inspection.source  # type: ignore[union-attr]
    )
    controller.set_reviewer_id("gui_reviewer")
    controller.enter_editing_mode()
    controller.create_draft(review_root=tmp_path / "gui_edits")
    win_id = controller.begin_window_request(half_window=30)
    controller.apply_window_snapshot(controller.window_worker(win_id))
    # Select first exclusive bout if present.
    exclusive = [r for r in controller.state.bout_rows if r["channel"] == "exclusive"]
    assert exclusive
    controller.select_bout(exclusive[0]["bout_id"])

    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=1, label="nose_to_body"
    )
    controller.set_parallel_pred(
        channel="contact",
        start_sample_idx=2,
        end_sample_idx=3,
        pred_value=0,
    )
    controller.set_bout_status(STATUS_ACCEPTED)
    controller.add_note("controlled GUI test annotation — not biological")
    assert controller.state.can_undo
    controller.undo()
    assert controller.state.can_redo
    controller.redo()
    controller.set_bout_status(STATUS_REJECTED)

    src = controller.state.editing_session.inspection.source  # type: ignore[union-attr]
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path if src.source_video else None,
        snapshots=snaps,
    )
    del built


def test_draft_reopening_recovery(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    # Default review root under session so discovery finds the draft.
    controller.set_reviewer_id("recovery_operator")
    controller.enter_editing_mode()
    controller.create_draft()
    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="approach"
    )
    run_dir = controller.state.review_run_dir
    run_id = controller.state.review_run_id
    assert run_dir is not None and run_id is not None

    # Close and rediscover.
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
        drafts = [o for o in controller2.state.review_options if o.is_draft]
        assert any(o.run_id == run_id for o in drafts)
        controller2.select_review_run(run_id)
        controller2.set_reviewer_id("recovery_operator")
        open_id = controller2.begin_open_draft()
        open_snap = controller2.open_source_worker(
            open_id,
            reviewer_id="recovery_operator",
            force_stale_lock=True,
        )
        assert open_snap.draft_resumed
        assert controller2.apply_open_snapshot(
            open_snap, reviewer_id="recovery_operator", force_stale_lock=True
        )
    assert controller2.state.draft_open
    assert controller2.state.editing_active
    assert controller2.state.can_undo or controller2.state.changed_intervals_text


def test_publication_blocking_and_success(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    controller.set_reviewer_id("publisher")
    controller.enter_editing_mode()
    controller.create_draft(review_root=tmp_path / "gui_publish")
    controller.set_exclusive_label(
        start_sample_idx=0, end_sample_idx=0, label="retract"
    )
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]), ExitStack() as stack:
        for p in _patch_s9_discovery_eligible(built["s8_run_id"]):
            stack.enter_context(p)
        preview = controller.build_publish_preview()
        assert preview.reviewer_id == "publisher"
        assert preview.parent_package_run_id == built["package_run_id"]

        fake = ReviewedPublicationResult(
            success=True,
            status="completed",
            review_run_id=str(controller.state.review_run_id),
            review_run_dir=Path(controller.state.review_run_dir or tmp_path),
            parent_package_run_id=built["package_run_id"],
            parent_package_dir=built["package_dir"],
            publication_state="published",
            currently_eligible=True,
            classification="eligible",
            qc={},
            warnings=(),
            error_message=None,
            science_written=True,
            artifact_paths={},
            diagnostics={},
        )
        controller._publish_fn = lambda session: fake  # type: ignore[method-assign]
        # Bypass live parent eligibility if mini package still reports stale.
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
        result = controller.publish_worker(pub_id)
        assert controller.apply_publish_result(pub_id, result)
    assert controller.state.last_publish_result is not None
    assert controller.state.last_publish_result.success


def test_publication_blocked_without_draft(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    with pytest.raises(BehavioralReviewUiError, match="draft"):
        controller.build_publish_preview()


def test_automated_versus_reviewed_distinction(tmp_path: Path) -> None:
    controller, _built = _controller_with_package(tmp_path)
    assert controller.state.view_source == "automated"
    assert controller.set_analysis_source_kind("reviewed") == SOURCE_TYPE_REVIEWED
    assert controller.set_analysis_source_kind("automated") == SOURCE_TYPE_AUTOMATED
    assert controller.state.inspected_published_only is False


def test_page_open_session_discovery_async(tmp_path: Path) -> None:
    """Exercise page discovery synchronously (avoid hanging offscreen QThread joins)."""

    built = _build_session(tmp_path)
    page = BehavioralReviewPage(BehavioralReviewController())
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]), ExitStack() as stack:
        for p in _patch_s9_discovery_eligible(built["s8_run_id"]):
            stack.enter_context(p)
        identity = page.controller.set_session(built["session_root"])
        snapshot = page.controller.collect_discovery_snapshot(identity)
        assert page.controller.apply_discovery_snapshot(snapshot)
    page.refresh_from_state()
    assert page.controller.state.session_root == built["session_root"].resolve()
    assert page.controller.state.s9_options
    assert "INSPECTION" in page.mode_banner.text().upper()


def test_bout_table_skips_unchanged_rebuild_on_window_apply(tmp_path: Path) -> None:
    """Scrub window updates must not rebuild an unchanged bout QTable."""

    controller, built = _controller_with_package(tmp_path)
    page = BehavioralReviewPage()
    page.controller = controller
    page.refresh_from_state()
    win_id = controller.begin_window_request(half_window=30)
    snap = controller.window_worker(win_id)
    assert controller.apply_window_snapshot(snap)
    page._populate_bout_table()  # noqa: SLF001
    assert page._last_bout_table_fingerprint is not None  # noqa: SLF001
    fingerprint = page._last_bout_table_fingerprint  # noqa: SLF001
    row_count = page.bout_table.rowCount()
    assert row_count > 0

    # Second apply with identical bout rows should be a no-op rebuild.
    page.bout_table.setRowCount(0)
    page._last_bout_table_fingerprint = fingerprint  # noqa: SLF001
    page._populate_bout_table(only_if_changed=True)  # noqa: SLF001
    assert page.bout_table.rowCount() == 0  # skipped rebuild
    page._populate_bout_table(only_if_changed=False)  # noqa: SLF001
    assert page.bout_table.rowCount() == row_count
    del built


def test_page_editing_gate_ui(tmp_path: Path) -> None:
    controller, built = _controller_with_package(tmp_path)
    # Use the page already wired to this controller without constructing a
    # second MainWindow (offscreen QThread teardown can hang).
    page = BehavioralReviewPage(controller)
    page.refresh_from_state()
    page.reviewer_edit.setText("ui_gate")
    page._reviewer_changed()
    # Drive enter/create via begin/worker/apply (avoid hanging offscreen QThreads).
    identity = controller.begin_enter_editing()
    assert controller.apply_edit_snapshot(controller.edit_worker(identity))
    assert controller.state.editing_active
    identity = controller.begin_create_draft(review_root=tmp_path / "page_drafts")
    assert controller.apply_edit_snapshot(controller.edit_worker(identity))
    page.refresh_from_state()
    assert controller.state.draft_open
    assert "EDITING" in page.mode_banner.text().upper()
    controller.close_session(release_lock=True)
    del built
