"""Product Quality Sprint 2: S3 tracking-review video responsiveness."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QThread
from tests.tracking_correction.test_review import _completed_s3
from tests.ui._responsiveness_support import (
    APPLICATION,
    SCHEDULE_BUDGET_SEC,
    Gate,
    count_event_loop_ticks,
    destroy_page,
    on_gui_thread,
    silence_modal_dialogs,
    wait_until,
)

from tracking_correction.review import PoseDisplaySource
from ui.controllers.tracking_review_controller import (
    FrameRenderSnapshot,
    FrameRequestIdentity,
    TrackingReviewController,
)
from ui.pages.tracking_review_page import TrackingReviewPage
from ui.tasks import GuiTaskKind
from ui.tracking_review_frame_decoder import TrackingReviewFrameDecoder


def _gated_reader(gate: Gate):
    def _read(_path: Path, frame_idx: int) -> np.ndarray:
        with gate.section(frame_idx):
            return np.full((16, 16, 3), int(frame_idx) % 256, dtype=np.uint8)

    return _read


def _wait_displayed(page: TrackingReviewPage, frame_idx: int) -> None:
    wait_until(
        lambda: (
            page.controller.displayed_frame_idx == frame_idx
            and not page.controller.frame_pending
            and page.video._frame is not None
        )
    )


def test_gui_task_kind_includes_tracking_review_frame() -> None:
    assert GuiTaskKind.TRACKING_REVIEW_FRAME.value == "tracking_review_frame"


def test_decode_executes_outside_gui_thread(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=4)
    gate = Gate()
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.show()

    started = time.perf_counter()
    page.open_run(s3_dir)
    assert time.perf_counter() - started < SCHEDULE_BUDGET_SEC
    assert page.has_running_task is True
    wait_until(gate.entered.is_set)
    assert count_event_loop_ticks() >= 3
    assert gate.never_on_gui_thread()
    assert on_gui_thread()

    gate.release()
    _wait_displayed(page, 0)
    assert page.has_running_task is False
    destroy_page(page)


def test_stable_videocapture_thread_ownership(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=6)
    thread_ids: list[int] = []
    lock = threading.Lock()

    def _reader(_path: Path, frame_idx: int) -> np.ndarray:
        with lock:
            thread_ids.append(threading.get_ident())
        assert QThread.currentThread() != APPLICATION.thread()
        return np.full((12, 12, 3), frame_idx % 256, dtype=np.uint8)

    controller = TrackingReviewController(frame_reader=_reader)
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    _wait_displayed(page, 0)
    page.controller.set_frame(1)
    page._refresh_view()
    _wait_displayed(page, 1)
    page.controller.set_frame(2)
    page._refresh_view()
    _wait_displayed(page, 2)

    assert thread_ids
    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != threading.get_ident()
    destroy_page(page)


def test_rapid_scrub_latest_wins_no_backlog(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=10)
    gate = Gate()
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    wait_until(gate.entered.is_set)
    # Hold the first decode so later scrub requests coalesce.
    for frame_idx in (3, 5, 7, 9):
        page.controller.set_frame(frame_idx)
        page._refresh_view()
        APPLICATION.processEvents()

    assert page.controller.session is not None
    assert page.controller.session.current_frame == 9
    assert page.controller.frame_pending is True
    # Labels track the selected frame immediately; pixels may still show older.
    assert "Prepared frame: 9" in page.frame_label.text()
    assert "Loading frame" in page.frame_pending_label.text()

    gate.release()
    _wait_displayed(page, 9)
    # First open (0) plus final scrub (9) — intermediate requests must not all run.
    assert gate.calls <= 3
    assert page.controller.displayed_frame_idx == 9
    destroy_page(page)


def test_playback_order_pause_and_seek(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=8)
    controller = TrackingReviewController(
        frame_reader=lambda _p, idx: np.full((8, 8, 3), idx % 256, dtype=np.uint8)
    )
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    _wait_displayed(page, 0)

    page._start_playback()
    assert page.controller.playback_active is True
    wait_until(lambda: (page.controller.session or None) is not None and (
        page.controller.session.current_frame >= 2  # type: ignore[union-attr]
    ))
    seen = page.controller.session.current_frame  # type: ignore[union-attr]
    page._stop_playback()
    assert page.controller.playback_active is False
    assert page.controller.session is not None
    paused_at = page.controller.session.current_frame
    assert paused_at >= 2

    page.controller.set_frame(1)
    page._refresh_view()
    _wait_displayed(page, 1)
    assert page.controller.displayed_frame_idx == 1

    page._start_playback()
    wait_until(
        lambda: page.controller.session is not None
        and page.controller.session.current_frame > 1
    )
    page._stop_playback()
    assert page.controller.session is not None
    assert page.controller.session.current_frame >= 1
    assert seen >= 2
    destroy_page(page)


def test_manual_edit_and_pose_source_invalidate_pending(
    tmp_path: Path, monkeypatch
) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=4)
    gate = Gate()
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    wait_until(gate.entered.is_set)
    first_revision = controller.edit_revision
    stale_identity = controller.active_frame_identity
    assert stale_identity is not None

    gate.release()
    _wait_displayed(page, 0)

    gate.hold()
    page.controller.set_frame(1)
    page._refresh_view()
    wait_until(gate.entered.is_set)
    pending_before_edit = controller.active_frame_identity
    assert pending_before_edit is not None

    controller.swap_identities(1, 1)
    assert controller.edit_revision == first_revision + 1
    page._refresh_view()
    assert controller.active_frame_identity is not None
    assert (
        controller.active_frame_identity.edit_revision
        == controller.edit_revision
    )

    stale = FrameRenderSnapshot(
        identity=pending_before_edit,
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        error=None,
    )
    assert controller.apply_frame_snapshot(stale) is False

    controller.set_pose_source(PoseDisplaySource.PROVISIONAL)
    assert controller.edit_revision >= first_revision + 2
    page._refresh_view()
    gate.release()
    wait_until(lambda: not controller.frame_pending)
    destroy_page(page)


def test_session_switch_rejects_stale_snapshot(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    first_dir, _s2a, _s1a = _completed_s3(tmp_path / "a", frame_count=4)
    second_dir, _s2b, _s1b = _completed_s3(tmp_path / "b", frame_count=4)
    controller = TrackingReviewController(
        frame_reader=lambda _p, idx: np.full((8, 8, 3), idx % 256, dtype=np.uint8)
    )
    page = TrackingReviewPage(controller)
    page.open_run(first_dir)
    _wait_displayed(page, 0)
    stale = controller.begin_frame_request()
    assert stale is not None
    first_generation = controller.session_generation

    page.open_run(second_dir)
    assert controller.session_generation == first_generation + 1
    assert controller.session is not None
    assert controller.session.run_dir.resolve() == second_dir.resolve()
    stale_snap = FrameRenderSnapshot(
        identity=stale.identity,
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    assert controller.apply_frame_snapshot(stale_snap) is False
    _wait_displayed(page, 0)
    destroy_page(page)


def test_decoder_failure_surfaces_without_modal(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=3)
    messages: list[str] = []

    def _boom(_path: Path, _frame_idx: int) -> np.ndarray:
        raise RuntimeError("synthetic decode failure")

    controller = TrackingReviewController(frame_reader=_boom)
    page = TrackingReviewPage(controller)
    page.status_message.connect(messages.append)
    page.open_run(s3_dir)
    wait_until(lambda: controller.last_frame_error is not None)
    assert "synthetic decode failure" in (controller.last_frame_error or "")
    assert page.frame_pending_label.text().startswith("Decode failed:")
    assert any("decode failed" in msg.lower() for msg in messages)
    destroy_page(page)


def test_safe_worker_shutdown(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=3)
    gate = Gate()
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    wait_until(gate.entered.is_set)
    assert page.has_running_task is True
    gate.release()
    wait_until(lambda: not page.has_running_task)
    finished = page._frame_decoder.shutdown(wait_ms=2_000)
    assert finished is True
    assert page._frame_decoder.is_running is False
    page.close()
    page.deleteLater()
    APPLICATION.processEvents()


def test_shutdown_timeout_retains_worker_until_completion(
    tmp_path: Path, monkeypatch
) -> None:
    """Soft wait False must not look like success; cleanup happens eventually."""

    from PySide6.QtGui import QCloseEvent

    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=3)
    gate = Gate()
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    wait_until(gate.entered.is_set)
    assert page.has_running_task is True

    timed_out = page.shutdown_frame_decoder(wait_ms=50)
    assert timed_out is False
    assert page._frame_decoder.is_shutting_down is True
    assert page._frame_decoder.is_running is True
    assert page.has_running_task is True

    # closeEvent must ignore while soft-timeout ownership remains. Cap the
    # cooperative wait so this assertion does not stall on the default 5 s.
    monkeypatch.setattr(
        page,
        "shutdown_frame_decoder",
        lambda *, wait_ms=5_000: page._frame_decoder.shutdown(
            wait_ms=min(50, int(wait_ms))
        ),
    )
    close_event = QCloseEvent()
    page.closeEvent(close_event)
    assert close_event.isAccepted() is False
    assert page._frame_decoder.is_running is True

    gate.release()
    wait_until(lambda: not page._frame_decoder.is_running, timeout_sec=10.0)
    assert page._frame_decoder.shutdown(wait_ms=2_000) is True
    assert page.has_running_task is False
    page.close()
    page.deleteLater()
    APPLICATION.processEvents()


def test_frame_dependent_edits_blocked_while_display_lags(
    tmp_path: Path, monkeypatch
) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=6)
    gate = Gate()
    messages: list[str] = []
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.status_message.connect(messages.append)
    page.open_run(s3_dir)
    wait_until(gate.entered.is_set)
    gate.release()
    _wait_displayed(page, 0)
    assert page.blank_node_button.isEnabled() is True

    calls_before = gate.calls
    gate.hold()
    page.controller.set_frame(4)
    page._refresh_view()
    wait_until(lambda: gate.calls > calls_before)
    assert page.controller.session is not None
    assert page.controller.session.current_frame == 4
    assert page.controller.displayed_frame_idx == 0
    assert page.controller.display_matches_playhead is False
    assert page.blank_node_button.isEnabled() is False
    assert page.swap_identities_button.isEnabled() is False
    assert page.use_current_start_button.isEnabled() is False
    assert "frame edits paused" in page.frame_pending_label.text()

    before_edits = page.controller.session.manual_edit_count
    page._blank_node()
    assert page.controller.session.manual_edit_count == before_edits
    assert any("unavailable until the selected prepared frame" in msg for msg in messages)

    gate.release()
    _wait_displayed(page, 4)
    assert page.controller.display_matches_playhead is True
    assert page.blank_node_button.isEnabled() is True
    destroy_page(page)


def test_pause_seek_and_late_snapshot_identity(tmp_path: Path, monkeypatch) -> None:
    silence_modal_dialogs(monkeypatch)
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=8)
    gate = Gate()
    controller = TrackingReviewController(frame_reader=_gated_reader(gate))
    page = TrackingReviewPage(controller)
    page.open_run(s3_dir)
    wait_until(gate.entered.is_set)
    gate.release()
    _wait_displayed(page, 0)

    calls_before = gate.calls
    gate.hold()
    page._start_playback()
    wait_until(
        lambda: page.controller.session is not None
        and page.controller.session.current_frame >= 1
        and gate.calls > calls_before
    )
    page._stop_playback()
    paused_at = page.controller.session.current_frame  # type: ignore[union-attr]
    assert page.controller.playback_active is False
    # Seek while a decode is still in flight.
    page.controller.set_frame(1)
    page._refresh_view()
    stale = controller.active_frame_identity
    assert stale is not None
    page.controller.set_frame(2)
    page._refresh_view()
    gate.release()
    _wait_displayed(page, 2)
    assert controller.displayed_frame_idx == 2
    assert controller.apply_frame_snapshot(
        FrameRenderSnapshot(
            identity=stale,
            image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        )
    ) is False
    assert paused_at >= 1
    destroy_page(page)


def test_apply_rejects_mismatched_identity(tmp_path: Path) -> None:
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=3)
    controller = TrackingReviewController(
        frame_reader=lambda _p, idx: np.zeros((4, 4, 3), dtype=np.uint8)
    )
    controller.open_run(s3_dir)
    request = controller.begin_frame_request()
    assert request is not None
    bad = FrameRenderSnapshot(
        identity=FrameRequestIdentity(
            request_id="other",
            frame_generation=request.identity.frame_generation,
            session_generation=request.identity.session_generation,
            run_id=request.identity.run_id,
            frame_idx=request.identity.frame_idx,
            pose_source=request.identity.pose_source,
            edit_revision=request.identity.edit_revision,
        ),
        image_bgr=np.ones((4, 4, 3), dtype=np.uint8),
    )
    assert controller.apply_frame_snapshot(bad) is False
    good = FrameRenderSnapshot(
        identity=request.identity,
        image_bgr=np.ones((4, 4, 3), dtype=np.uint8),
    )
    assert controller.apply_frame_snapshot(good) is True
    assert controller.displayed_frame_idx == 0


def test_sync_current_view_unchanged_for_tests(tmp_path: Path) -> None:
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=3)
    controller = TrackingReviewController(
        frame_reader=lambda _p, idx: np.full((8, 8, 3), idx % 256, dtype=np.uint8)
    )
    controller.open_run(s3_dir)
    view = controller.current_view()
    assert view.frame_idx == 0
    assert view.image_bgr.shape == (8, 8, 3)
    controller.swap_identities(0, 0)
    after = controller.current_view()
    assert after.manual_edit_count == 1
    assert after.frame_idx == 0


def test_decoder_pending_slot_is_single(tmp_path: Path) -> None:
    decoder = TrackingReviewFrameDecoder(
        frame_reader=lambda _p, idx: np.full((4, 4, 3), idx % 256, dtype=np.uint8),
        use_sequential_capture=False,
    )
    controller = TrackingReviewController(
        frame_reader=lambda _p, idx: np.full((4, 4, 3), idx % 256, dtype=np.uint8)
    )
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=5)
    controller.open_run(s3_dir)
    applied: list[int] = []

    def _on_snap(result: object) -> None:
        if isinstance(result, FrameRenderSnapshot) and controller.apply_frame_snapshot(
            result
        ):
            applied.append(result.identity.frame_idx)

    decoder.snapshot_ready.connect(_on_snap)
    for frame_idx in range(5):
        controller.set_frame(frame_idx)
        request = controller.begin_frame_request()
        assert request is not None
        decoder.submit(request)
    wait_until(lambda: controller.displayed_frame_idx == 4 and not decoder.is_running)
    assert applied[-1] == 4
    assert len(applied) < 5 or applied == sorted(applied)
    assert decoder.shutdown(wait_ms=2_000) is True


def test_correction_acceptance_still_works_via_controller(tmp_path: Path) -> None:
    s3_dir, _s2, _s1 = _completed_s3(tmp_path, frame_count=4)
    controller = TrackingReviewController(
        frame_reader=lambda _p, idx: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    controller.open_run(s3_dir)
    before = {
        (p.track, p.node): (p.x, p.y) for p in controller.current_view().pose_points
    }
    controller.swap_identities(0, 0)
    after = {
        (p.track, p.node): (p.x, p.y) for p in controller.current_view().pose_points
    }
    assert after[(0, "nose")] == pytest.approx(before[(1, "nose")])
    result = controller.accept_tracking()
    assert result.already_accepted is False
    assert result.tracked_pose_path.is_file()
