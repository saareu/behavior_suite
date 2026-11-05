"""S4 GUI responsiveness: asynchronous session open/discovery, stale rejection."""

from __future__ import annotations

import json
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
from tests.ui.test_pose_finalization_gui import _write_accepted_session

from pose_finalization import (
    PoseFinalizationError,
    PoseFinalizationRequest,
    run_pose_finalization,
)
from tracking_correction.contracts import (
    ACCEPTANCE_SUPERSEDED,
)
from tracking_correction.contracts import (
    RUN_META_FILENAME as S3_RUN_META_FILENAME,
)
from ui.controllers.pose_finalization_controller import (
    DiscoverySnapshot,
    PoseFinalizationController,
)
from ui.pages.pose_finalization_page import PoseFinalizationPage

_ = APPLICATION

_MODULE = "ui.controllers.pose_finalization_controller"


@pytest.fixture(autouse=True)
def _no_blocking_qt_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    silence_modal_dialogs(monkeypatch)


def _session(tmp_path: Path, name: str, **kwargs: object) -> Path:
    root = tmp_path / name
    _write_accepted_session(root, n_frames=16, **kwargs)  # type: ignore[arg-type]
    return root.resolve()


def _install_gate(monkeypatch: pytest.MonkeyPatch, gate: Gate) -> None:
    """Wrap the worker-side collectors so every discovery call passes the gate."""

    import ui.controllers.pose_finalization_controller as module

    real_summarize = module.summarize_pose_finalization_project
    real_list = module.list_accepted_s3_runs

    def summarize(root: Path | str):
        with gate.section(Path(root)):
            return real_summarize(root)

    def list_runs(root: Path | str):
        # Same worker call as ``summarize``; only record the thread, do not block twice.
        gate.thread_ids.append(threading.get_ident())
        return real_list(root)

    monkeypatch.setattr(module, "summarize_pose_finalization_project", summarize)
    monkeypatch.setattr(module, "list_accepted_s3_runs", list_runs)


def _tree_fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_discovery_runs_outside_gui_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")
    gate = Gate(blocking=False)
    _install_gate(monkeypatch, gate)
    page = PoseFinalizationPage(PoseFinalizationController())

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert gate.calls == 1
    assert gate.never_on_gui_thread()
    assert threading.get_ident() not in gate.thread_ids
    assert page.controller.state.summary is not None
    assert page.controller.selected_accepted_s3() is not None
    destroy_page(page)


def test_gui_stays_responsive_during_slow_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.show()

    started = time.perf_counter()
    page.set_session(session)
    elapsed = time.perf_counter() - started

    assert elapsed < SCHEDULE_BUDGET_SEC
    assert controller.state.discovery_running is True
    assert page.has_running_task is True
    assert page.run_button.isEnabled() is False
    # Session browse stays available so a newer session can supersede discovery.
    assert page.session_browse_button.isEnabled() is True
    assert page.open_session_button.isEnabled() is True
    wait_until(gate.entered.is_set)
    assert count_event_loop_ticks() >= 3
    assert controller.state.discovery_running is True
    assert controller.state.summary is None

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.discovery_running is False
    assert controller.state.summary is not None
    assert page.run_button.isEnabled() is True
    destroy_page(page)


def test_session_change_rejects_stale_discovery_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _session(tmp_path, "first")
    second = _session(tmp_path, "second")
    controller = PoseFinalizationController()
    # Collect a genuine snapshot for the first session before it is superseded.
    first_identity = controller.prepare_session(first)
    first_snapshot = controller.collect_discovery_snapshot(first_identity)

    gate = Gate()
    _install_gate(monkeypatch, gate)
    page = PoseFinalizationPage(controller)
    page.set_session(second)
    second_identity = controller.state.active_discovery_identity
    assert second_identity is not None
    assert second_identity.request_id != first_identity.request_id
    assert second_identity.generation > first_identity.generation

    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert controller.state.summary is None
    assert controller.state.accepted_s3_runs == ()

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.session_root == second
    assert controller.state.summary is not None
    assert controller.state.summary.session_root == second
    # A late duplicate of the stale snapshot must still be refused after completion.
    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert isinstance(first_snapshot, DiscoverySnapshot)
    destroy_page(page)


def test_out_of_order_completion_cannot_overwrite_newer_results(
    tmp_path: Path,
) -> None:
    first = _session(tmp_path, "first")
    second = _session(tmp_path, "second")
    controller = PoseFinalizationController()

    first_identity = controller.prepare_session(first)
    first_snapshot = controller.collect_discovery_snapshot(first_identity)
    second_identity = controller.prepare_session(second)
    second_snapshot = controller.collect_discovery_snapshot(second_identity)

    assert controller.apply_discovery_snapshot(second_snapshot) is True
    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert controller.state.summary is second_snapshot.summary
    assert controller.state.session_root == second

    older = controller.begin_discovery()
    newer = controller.begin_discovery()
    older_snapshot = controller.collect_discovery_snapshot(older)
    newer_snapshot = controller.collect_discovery_snapshot(newer)
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.discovery_running is True
    assert controller.apply_discovery_snapshot(newer_snapshot) is True
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.summary is newer_snapshot.summary


def test_repeated_open_does_not_start_parallel_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)

    for _ in range(5):
        page.set_session(session)
    assert controller.state.discovery_generation == 5
    assert page.discovery_runner.is_running is True

    wait_until(gate.entered.is_set)
    gate.release()
    wait_until(lambda: not page.has_running_task)

    assert gate.max_active == 1
    assert 1 <= gate.calls <= 2
    assert controller.state.discovery_generation == 5
    assert controller.state.discovery_running is False
    destroy_page(page)


def test_refresh_is_refused_while_discovery_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    generation = controller.state.discovery_generation

    page._refresh_discovery()
    page._refresh_discovery()

    assert controller.state.discovery_generation == generation
    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert gate.max_active == 1
    destroy_page(page)


def test_accepted_s3_selection_persists_on_same_session_refresh(
    tmp_path: Path,
) -> None:
    session = _session(
        tmp_path,
        "s",
        run_id="older__20260101T000000",
        completed_at="2026-01-01T00:00:00",
        accepted_at="2026-01-01T00:05:00",
    )
    _write_accepted_session(
        session,
        run_id="newer__20260102T000000",
        completed_at="2026-01-02T00:00:00",
        accepted_at="2026-01-02T00:05:00",
        n_frames=16,
    )
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    assert controller.state.selected_s3_run_id == "newer__20260102T000000"

    controller.select_accepted_s3_run("older__20260101T000000")
    page._refresh_discovery()
    assert controller.state.discovery_running is True
    wait_until(lambda: not page.has_running_task)

    assert controller.state.selected_s3_run_id == "older__20260101T000000"
    selected = controller.selected_accepted_s3()
    assert selected is not None
    assert selected.run_id == "older__20260101T000000"
    destroy_page(page)


def test_invalidated_accepted_selection_is_not_kept(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        "s",
        run_id="older__20260101T000000",
        completed_at="2026-01-01T00:00:00",
        accepted_at="2026-01-01T00:05:00",
    )
    newer_dir = _write_accepted_session(
        session,
        run_id="newer__20260102T000000",
        completed_at="2026-01-02T00:00:00",
        accepted_at="2026-01-02T00:05:00",
        n_frames=16,
    )
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    controller.select_accepted_s3_run("newer__20260102T000000")

    meta_path = newer_dir / S3_RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["acceptance_state"] = ACCEPTANCE_SUPERSEDED
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    accepted_ids = {run.run_id for run in controller.state.accepted_s3_runs}
    assert "newer__20260102T000000" not in accepted_ids
    assert controller.state.selected_s3_run_id != "newer__20260102T000000"
    assert controller.state.selected_s3_run_id in accepted_ids
    destroy_page(page)


def test_selected_s4_run_survives_refresh(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    s3_dir = next((session / "tracking_correction").iterdir())
    result = run_pose_finalization(
        PoseFinalizationRequest(s3_run_dir=s3_dir, timestamp="20261019T201000")
    )
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    controller.select_s4_run(result.run_id)

    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    assert controller.state.selected_s4_run_id == result.run_id
    assert controller.selected_s4_run() is not None
    destroy_page(page)


def test_worker_errors_are_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")

    def broken(_root: Path | str):
        raise PoseFinalizationError("synthetic S4 scan failure")

    monkeypatch.setattr(f"{_MODULE}.summarize_pose_finalization_project", broken)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    messages: list[str] = []
    page.status_message.connect(messages.append)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert controller.state.discovery_running is False
    assert controller.state.run_error is not None
    assert "synthetic S4 scan failure" in controller.state.run_error
    assert any("S4 discovery failed" in message for message in messages)
    assert "synthetic S4 scan failure" in page.stage_label.text()
    destroy_page(page)


def test_unexpected_worker_exception_emits_unexpected_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")

    def broken(_root: Path | str):
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(f"{_MODULE}.summarize_pose_finalization_project", broken)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    seen: list[str] = []
    page.unexpected_error.connect(seen.append)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert seen and "synthetic unexpected failure" in seen[0]
    assert controller.state.run_error == "synthetic unexpected failure"
    assert controller.state.discovery_running is False
    destroy_page(page)


def test_stale_failure_does_not_clobber_state(tmp_path: Path) -> None:
    first = _session(tmp_path, "first")
    second = _session(tmp_path, "second")
    controller = PoseFinalizationController()
    stale_identity = controller.prepare_session(first)
    controller.prepare_session(second)

    assert (
        controller.record_discovery_failure("late failure", expected_identity=stale_identity)
        is False
    )
    assert controller.state.run_error is None
    assert controller.state.discovery_running is True


def test_page_close_and_reopen_during_discovery_is_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = PoseFinalizationController()
    page = PoseFinalizationPage(controller)
    page.show()
    page.set_session(session)
    wait_until(gate.entered.is_set)

    page.close()
    gate.release()
    wait_until(lambda: not page.has_running_task)
    destroy_page(page)

    reopened = PoseFinalizationPage(controller)
    reopened.show()
    reopened.set_session(session)
    wait_until(lambda: not reopened.has_running_task)
    assert controller.selected_accepted_s3() is not None
    assert reopened.run_button.isEnabled() is True
    destroy_page(reopened)


def test_discovery_does_not_write_session_artifacts(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    before = _tree_fingerprint(session)
    page = PoseFinalizationPage(PoseFinalizationController())

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    assert _tree_fingerprint(session) == before
    destroy_page(page)
