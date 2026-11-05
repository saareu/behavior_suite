"""S5 GUI responsiveness: asynchronous S1 open + run discovery, stale rejection."""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import pytest
from tests.ui._responsiveness_support import (
    APPLICATION,
    SCHEDULE_BUDGET_SEC,
    Gate,
    count_event_loop_ticks,
    destroy_page,
    silence_modal_dialogs,
    wait_until,
)
from tests.ui.test_blob_features_gui import _write_eligible_s1

from blob_features import (
    BlobFeaturesError,
    BlobFeaturesProjectSummary,
    BlobFeaturesRunSummary,
)
from blob_features.run_discovery import ELIGIBLE_DOWNSTREAM
from ui.controllers.blob_features_controller import (
    OPENING_SOURCE_MESSAGE,
    BlobFeaturesController,
    DiscoverySnapshot,
    read_prepared_video_frame,
)
from ui.pages.blob_features_page import BlobFeaturesPage

_ = APPLICATION

_MODULE = "ui.controllers.blob_features_controller"


@pytest.fixture(autouse=True)
def _no_blocking_qt_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    silence_modal_dialogs(monkeypatch)


def _gated_controller(gate: Gate) -> BlobFeaturesController:
    """Controller whose worker-side frame decode passes through ``gate``."""

    def reader(path: Path, index: int):
        with gate.section(Path(path)):
            return read_prepared_video_frame(path, index)

    return BlobFeaturesController(frame_reader=reader)


def _fake_run(session: Path, run_id: str, completed_at: str) -> BlobFeaturesRunSummary:
    return BlobFeaturesRunSummary(
        run_dir=session / "blob_features" / run_id,
        run_id=run_id,
        status="completed",
        classification=ELIGIBLE_DOWNSTREAM,
        source_s1_session_root=session,
        source_prepared_sync=None,
        blob_features_path=None,
        created_at=completed_at,
        completed_at=completed_at,
        warning_count=0,
        error_message=None,
    )


def _patch_runs(
    monkeypatch: pytest.MonkeyPatch, runs_box: dict[str, tuple[str, ...]]
) -> None:
    """Report fabricated S5 runs on top of the real (S1-derived) project summary."""

    import ui.controllers.blob_features_controller as module

    real = module.summarize_blob_features_project

    def summarize(root: Path | str) -> BlobFeaturesProjectSummary:
        summary = real(root)
        return dataclasses.replace(
            summary,
            blob_features_present=bool(runs_box["ids"]),
            runs=tuple(
                _fake_run(summary.session_root, run_id, f"2026-01-0{index + 1}T00:00:00")
                for index, run_id in reversed(list(enumerate(runs_box["ids"])))
            ),
        )

    monkeypatch.setattr(module, "summarize_blob_features_project", summarize)


def _gate_discovery(monkeypatch: pytest.MonkeyPatch, gate: Gate) -> None:
    import ui.controllers.blob_features_controller as module

    real = module.summarize_blob_features_project

    def summarize(root: Path | str):
        with gate.section(Path(root)):
            return real(root)

    monkeypatch.setattr(module, "summarize_blob_features_project", summarize)


def _source_fingerprint(session: Path) -> dict[str, tuple[int, int]]:
    preprocess = session / "preprocess"
    return {
        str(path.relative_to(session)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(preprocess.rglob("*"))
        if path.is_file()
    }


def _open(page: BlobFeaturesPage, session: Path) -> None:
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)


def test_open_and_discovery_run_outside_gui_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _write_eligible_s1(tmp_path)
    frame_gate = Gate(blocking=False)
    validate_gate = Gate(blocking=False)
    summarize_gate = Gate(blocking=False)

    import ui.controllers.blob_features_controller as module

    real_validate = module.validate_s1_for_blob_features
    real_summarize = module.summarize_blob_features_project

    def validate(root: Path):
        with validate_gate.section(root):
            return real_validate(root)

    def summarize(root: Path | str):
        with summarize_gate.section(Path(root)):
            return real_summarize(root)

    monkeypatch.setattr(module, "validate_s1_for_blob_features", validate)
    monkeypatch.setattr(module, "summarize_blob_features_project", summarize)
    controller = _gated_controller(frame_gate)
    page = BlobFeaturesPage(controller)

    _open(page, session)
    for gate in (frame_gate, validate_gate, summarize_gate):
        assert gate.calls == 1
        assert gate.never_on_gui_thread()
        assert threading.get_ident() not in gate.thread_ids
    assert controller.state.eligible is not None

    # Standalone run discovery is also a worker-only operation.
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)
    assert summarize_gate.calls == 2
    assert summarize_gate.never_on_gui_thread()
    destroy_page(page)


def test_gui_stays_responsive_during_slow_open(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    gate = Gate()
    controller = _gated_controller(gate)
    page = BlobFeaturesPage(controller)
    page.show()

    started = time.perf_counter()
    page.set_session(session)
    elapsed = time.perf_counter() - started

    assert elapsed < SCHEDULE_BUDGET_SEC
    # Open/loading state is visible immediately.
    assert controller.state.open_running is True
    assert page.has_running_task is True
    assert page.eligibility_label.text() == OPENING_SOURCE_MESSAGE
    assert page.frame_meta_label.text() == OPENING_SOURCE_MESSAGE
    assert page.progress.isHidden() is False
    assert page.run_button.isEnabled() is False
    # Session browse stays available so a newer session can supersede open.
    assert page.session_browse_button.isEnabled() is True
    assert page.open_session_button.isEnabled() is True
    wait_until(gate.entered.is_set)
    assert count_event_loop_ticks() >= 3
    assert controller.state.open_running is True
    assert controller.state.eligible is None

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.open_running is False
    assert controller.state.eligible is not None
    assert controller.state.current_frame_bgr is not None
    assert "Eligible S1" in page.eligibility_label.text()
    assert page.frame_view._frame is not None
    destroy_page(page)


def test_previous_session_frame_cleared_before_new_open_completes(
    tmp_path: Path,
) -> None:
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    gate = Gate(blocking=False)
    controller = _gated_controller(gate)
    page = BlobFeaturesPage(controller)
    _open(page, first)
    assert page.frame_view._frame is not None
    assert controller.state.current_frame_bgr is not None
    assert controller.state.background_bgr is not None

    gate.hold()
    page.set_session(second)

    state = controller.state
    assert state.session_root == second.resolve()
    assert state.open_running is True
    assert state.eligible is None
    assert state.source_identity is None
    assert state.current_frame_bgr is None
    assert state.background_bgr is None
    assert state.bookmarks is None
    assert state.summary is None
    assert state.last_single_preview is None
    assert state.preview_valid is False
    assert page.frame_view._frame is None
    assert page.eligibility_label.text() == OPENING_SOURCE_MESSAGE

    wait_until(lambda: gate.calls == 2)
    assert page.frame_view._frame is None  # still blocked: nothing stale re-shown
    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert state.eligible is not None
    assert state.eligible.session_root == second.resolve()
    assert page.frame_view._frame is not None
    destroy_page(page)


def test_session_change_rejects_stale_open_snapshot(tmp_path: Path) -> None:
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller = BlobFeaturesController()

    first_identity = controller.prepare_session(first)
    first_snapshot = controller.collect_open_snapshot(first_identity)
    second_identity = controller.prepare_session(second)
    assert second_identity.request_id != first_identity.request_id
    assert second_identity.generation > first_identity.generation

    assert controller.apply_open_snapshot(first_snapshot) is False
    assert controller.state.open_running is True
    assert controller.state.eligible is None
    assert controller.state.current_frame_bgr is None

    second_snapshot = controller.collect_open_snapshot(second_identity)
    assert controller.apply_open_snapshot(second_snapshot) is True
    assert controller.state.eligible is not None
    assert controller.state.eligible.session_root == second.resolve()
    # A late duplicate of the stale snapshot is still refused.
    assert controller.apply_open_snapshot(first_snapshot) is False
    assert controller.state.eligible.session_root == second.resolve()


def test_out_of_order_open_completion_cannot_overwrite_newer_results(
    tmp_path: Path,
) -> None:
    session = _write_eligible_s1(tmp_path)
    controller = BlobFeaturesController()

    older = controller.prepare_session(session)
    newer = controller.begin_open()  # same session, newer generation
    older_snapshot = controller.collect_open_snapshot(older)
    newer_snapshot = controller.collect_open_snapshot(newer)

    assert controller.apply_open_snapshot(newer_snapshot) is True
    assert controller.apply_open_snapshot(older_snapshot) is False
    assert controller.state.eligible is newer_snapshot.eligible
    assert controller.state.current_frame_bgr is newer_snapshot.current_frame_bgr


def test_stale_discovery_snapshot_rejected(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    controller = BlobFeaturesController()
    controller.set_session(session)

    older = controller.begin_discovery()
    newer = controller.begin_discovery()
    assert newer.generation > older.generation
    older_snapshot = controller.collect_discovery_snapshot(older)
    newer_snapshot = controller.collect_discovery_snapshot(newer)

    assert isinstance(older_snapshot, DiscoverySnapshot)
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.discovery_running is True
    assert controller.apply_discovery_snapshot(newer_snapshot) is True
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.summary is newer_snapshot.summary

    # A discovery snapshot cannot land while a (re)open is in flight.
    pending = controller.begin_discovery()
    pending_snapshot = controller.collect_discovery_snapshot(pending)
    controller.begin_open()
    assert controller.apply_discovery_snapshot(pending_snapshot) is False


def test_repeated_open_does_not_start_parallel_workers(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    gate = Gate()
    controller = _gated_controller(gate)
    page = BlobFeaturesPage(controller)

    for _ in range(4):
        page.set_session(session)
    assert controller.state.open_generation == 4
    assert page.open_runner.is_running is True

    wait_until(gate.entered.is_set)
    gate.release()
    wait_until(lambda: not page.has_running_task)

    assert gate.max_active == 1
    assert 1 <= gate.calls <= 2
    assert controller.state.open_generation == 4
    assert controller.state.eligible is not None
    destroy_page(page)


def test_repeated_refresh_does_not_start_parallel_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _write_eligible_s1(tmp_path)
    controller = BlobFeaturesController()
    page = BlobFeaturesPage(controller)
    _open(page, session)

    gate = Gate()
    _gate_discovery(monkeypatch, gate)
    generation = controller.state.discovery_generation
    page._refresh_discovery()
    page._refresh_discovery()
    page._refresh_discovery()

    assert controller.state.discovery_generation == generation + 1
    assert page.discovery_runner.is_running is True
    wait_until(gate.entered.is_set)
    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert gate.max_active == 1
    assert gate.calls == 1
    destroy_page(page)


def test_selected_s5_run_survives_same_session_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _write_eligible_s1(tmp_path)
    box = {"ids": ("blob_features__old", "blob_features__new")}
    _patch_runs(monkeypatch, box)
    controller = BlobFeaturesController()
    page = BlobFeaturesPage(controller)
    _open(page, session)
    assert controller.state.selected_s5_run_id == "blob_features__new"

    controller.select_s5_run("blob_features__old")
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    assert controller.state.selected_s5_run_id == "blob_features__old"
    selected = controller.selected_s5_run()
    assert selected is not None
    assert selected.run_id == "blob_features__old"
    destroy_page(page)


def test_invalidated_s5_selection_is_not_kept_silently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _write_eligible_s1(tmp_path)
    box = {"ids": ("blob_features__old", "blob_features__new")}
    _patch_runs(monkeypatch, box)
    controller = BlobFeaturesController()
    page = BlobFeaturesPage(controller)
    _open(page, session)
    controller.select_s5_run("blob_features__old")

    box["ids"] = ("blob_features__new",)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    ids = {run.run_id for run in (controller.state.summary.runs if controller.state.summary else ())}
    assert ids == {"blob_features__new"}
    assert controller.state.selected_s5_run_id != "blob_features__old"
    assert controller.state.selected_s5_run_id in ids
    assert page.runs_table.rowCount() == 1

    box["ids"] = ()
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.selected_s5_run_id is None
    destroy_page(page)


def test_open_worker_exception_is_surfaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _write_eligible_s1(tmp_path)

    def explode(_root: Path):
        raise RuntimeError("synthetic S5 open failure")

    monkeypatch.setattr(f"{_MODULE}.validate_s1_for_blob_features", explode)
    controller = BlobFeaturesController()
    page = BlobFeaturesPage(controller)
    seen: list[str] = []
    page.unexpected_error.connect(seen.append)

    _open(page, session)

    assert seen and "synthetic S5 open failure" in seen[0]
    assert controller.state.open_running is False
    assert controller.state.eligible is None
    assert controller.state.eligibility_error == "synthetic S5 open failure"
    assert "synthetic S5 open failure" in page.eligibility_label.text()
    destroy_page(page)


def test_ineligible_s1_is_reported_not_raised_in_worker(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    controller = BlobFeaturesController()
    page = BlobFeaturesPage(controller)
    messages: list[str] = []
    page.status_message.connect(messages.append)

    _open(page, empty)

    assert controller.state.open_running is False
    assert controller.state.eligible is None
    assert controller.state.eligibility_error
    assert controller.state.eligibility_error != OPENING_SOURCE_MESSAGE
    assert page.eligibility_label.text() == controller.state.eligibility_error
    assert page.run_button.isEnabled() is False
    assert any(controller.state.eligibility_error in message for message in messages)
    destroy_page(page)


def test_discovery_failure_during_open_and_refresh_is_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _write_eligible_s1(tmp_path)

    def broken(_root: Path | str):
        raise BlobFeaturesError("synthetic S5 scan failure")

    monkeypatch.setattr(f"{_MODULE}.summarize_blob_features_project", broken)
    controller = BlobFeaturesController()
    page = BlobFeaturesPage(controller)
    _open(page, session)

    # The S1 source opened, but the failed scan is reported (not swallowed).
    assert controller.state.eligible is not None
    assert controller.state.discovery_error == "synthetic S5 scan failure"
    assert "synthetic S5 scan failure" in page.error_label.text()

    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.discovery_running is False
    assert controller.state.discovery_error == "synthetic S5 scan failure"
    assert "Discovery failed" in page.eligibility_label.text()
    destroy_page(page)


def test_stale_failure_does_not_clobber_state(tmp_path: Path) -> None:
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller = BlobFeaturesController()
    stale_open = controller.prepare_session(first)
    controller.prepare_session(second)

    assert (
        controller.record_open_failure("late failure", expected_identity=stale_open)
        is False
    )
    assert controller.state.open_running is True
    assert controller.state.eligibility_error == OPENING_SOURCE_MESSAGE

    controller.set_session(second)
    stale_discovery = controller.begin_discovery()
    controller.begin_discovery()
    assert (
        controller.record_discovery_failure(
            "late failure", expected_identity=stale_discovery
        )
        is False
    )
    assert controller.state.discovery_error is None


def test_page_close_and_reopen_during_open_is_safe(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    gate = Gate()
    controller = _gated_controller(gate)
    page = BlobFeaturesPage(controller)
    page.show()
    page.set_session(session)
    wait_until(gate.entered.is_set)

    page.close()
    gate.release()
    wait_until(lambda: not page.has_running_task)
    destroy_page(page)

    reopened = BlobFeaturesPage(controller)
    reopened.show()
    reopened.set_session(session)
    wait_until(lambda: not reopened.has_running_task)
    assert controller.state.eligible is not None
    assert reopened.frame_view._frame is not None
    destroy_page(reopened)


def test_open_and_discovery_do_not_write_sources_or_runs(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    before = _source_fingerprint(session)
    page = BlobFeaturesPage(BlobFeaturesController())

    _open(page, session)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    assert _source_fingerprint(session) == before
    run_root = session / "blob_features"
    assert not run_root.is_dir() or not [
        child for child in run_root.iterdir() if child.name.startswith("blob_features__")
    ]
    destroy_page(page)
