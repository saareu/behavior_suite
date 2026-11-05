"""S2 GUI responsiveness: asynchronous session open/discovery, stale rejection."""

from __future__ import annotations

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
from tests.ui.test_pose_inference_page import _run, _summary, _write_s1

from pose_inference.run_discovery import PoseInferenceProjectSummary
from ui.controllers.pose_inference_controller import (
    DiscoverySnapshot,
    PoseInferenceController,
    S1UiStatus,
)
from ui.pages.pose_inference_page import PoseInferencePage

_ = APPLICATION


@pytest.fixture(autouse=True)
def _no_blocking_qt_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    silence_modal_dialogs(monkeypatch)


def _session(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    _write_s1(root)
    return root.resolve()


def _gated_controller(
    gate: Gate, **kwargs: object
) -> PoseInferenceController:
    def discover(root: Path | str) -> PoseInferenceProjectSummary:
        with gate.section(Path(root)):
            return _summary(Path(root))

    return PoseInferenceController(discovery=discover, **kwargs)  # type: ignore[arg-type]


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
    discovery_gate = Gate(blocking=False)
    metadata_gate = Gate(blocking=False)

    import ui.controllers.pose_inference_controller as module

    real_validate = module.validate_s1_handoff

    def recording_validate(root: Path):
        with metadata_gate.section(root):
            return real_validate(root)

    monkeypatch.setattr(module, "validate_s1_handoff", recording_validate)
    controller = _gated_controller(discovery_gate)
    page = PoseInferencePage(controller)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert discovery_gate.calls == 1
    assert metadata_gate.calls == 1
    assert discovery_gate.never_on_gui_thread()
    assert metadata_gate.never_on_gui_thread()
    assert discovery_gate.thread_ids[0] != threading.get_ident()
    assert controller.state.s1_status is S1UiStatus.READY
    destroy_page(page)


def test_gui_stays_responsive_during_slow_discovery(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    controller = _gated_controller(gate)
    page = PoseInferencePage(controller)
    page.show()

    started = time.perf_counter()
    page.set_session(session)
    elapsed = time.perf_counter() - started

    assert elapsed < SCHEDULE_BUDGET_SEC
    assert controller.state.discovery_running is True
    assert page.has_running_task is True
    assert page.run_button.isEnabled() is False
    assert page.refresh_runs_button.isEnabled() is False
    # Session browse stays available so a newer session can supersede discovery.
    assert page.session_browse_button.isEnabled() is True
    assert page.open_session_button.isEnabled() is True
    assert "discovering" in page.s1_status_label.text().lower()
    wait_until(gate.entered.is_set)
    # The event loop keeps delivering timer events while the worker is blocked.
    assert count_event_loop_ticks() >= 3
    assert controller.state.discovery_running is True

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.discovery_running is False
    assert controller.state.s1_status is S1UiStatus.READY
    assert page.frame_count_label.text() == "12"
    assert page.session_path.text() == str(session)
    destroy_page(page)


def test_session_change_rejects_stale_discovery_snapshot(tmp_path: Path) -> None:
    first = _session(tmp_path, "first")
    second = _session(tmp_path, "second")
    gate = Gate()
    controller = _gated_controller(gate)
    page = PoseInferencePage(controller)

    page.set_session(first)
    first_identity = controller.state.active_discovery_identity
    assert first_identity is not None
    wait_until(gate.entered.is_set)

    page.set_session(second)
    second_identity = controller.state.active_discovery_identity
    assert second_identity is not None
    assert second_identity.request_id != first_identity.request_id
    assert second_identity.generation > first_identity.generation
    assert second_identity.session_root == second

    stale = DiscoverySnapshot(
        identity=first_identity,
        summary=_summary(first),
        s1_status=S1UiStatus.READY,
        s1_message="stale",
        prepared_video_path=first / "preprocess" / "prepared_video.mp4",
        prepared_frame_count=999,
        timing_status="stale",
    )
    assert controller.apply_discovery_snapshot(stale) is False
    assert controller.state.discovery_running is True
    assert controller.state.prepared_frame_count is None

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.session_root == second
    assert controller.state.summary is not None
    assert controller.state.summary.session_root == second
    assert controller.state.s1_message != "stale"
    # Every worker scanned the root it was started for; the final apply is the newest.
    assert gate.payloads[0] == first
    assert gate.payloads[-1] == second
    destroy_page(page)


def test_out_of_order_completion_cannot_overwrite_newer_results(
    tmp_path: Path,
) -> None:
    first = _session(tmp_path, "first")
    second = _session(tmp_path, "second")
    controller = PoseInferenceController(discovery=lambda root: _summary(Path(root)))

    first_identity = controller.prepare_session(first)
    first_snapshot = controller.collect_discovery_snapshot(first_identity)
    second_identity = controller.prepare_session(second)
    second_snapshot = controller.collect_discovery_snapshot(second_identity)

    # Newer result lands first; the older worker finishing late must be ignored.
    assert controller.apply_discovery_snapshot(second_snapshot) is True
    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert controller.state.summary is second_snapshot.summary
    assert controller.state.session_root == second

    # Same-session refreshes are ordered by identity as well.
    older = controller.begin_discovery()
    newer = controller.begin_discovery()
    older_snapshot = controller.collect_discovery_snapshot(older)
    newer_snapshot = controller.collect_discovery_snapshot(newer)
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.discovery_running is True
    assert controller.apply_discovery_snapshot(newer_snapshot) is True
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.summary is newer_snapshot.summary


def test_cleared_session_rejects_inflight_discovery(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    controller = _gated_controller(gate)
    page = PoseInferencePage(controller)

    page.set_session(session)
    wait_until(gate.entered.is_set)
    page.set_session("")
    assert controller.state.session_root is None
    assert controller.state.discovery_running is False

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.session_root is None
    assert controller.state.summary is None
    destroy_page(page)


def test_repeated_open_does_not_start_parallel_workers(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    controller = _gated_controller(gate)
    page = PoseInferencePage(controller)

    for _ in range(5):
        page.set_session(session)
    assert controller.state.discovery_generation == 5
    assert page.discovery_runner.is_running is True

    wait_until(gate.entered.is_set)
    gate.release()
    wait_until(lambda: not page.has_running_task)

    assert gate.max_active == 1
    # Coalesced: the first worker plus at most one restart for the latest request.
    assert 1 <= gate.calls <= 2
    assert controller.state.discovery_generation == 5
    assert controller.state.discovery_running is False
    destroy_page(page)


def test_refresh_is_refused_while_discovery_runs(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    controller = _gated_controller(gate)
    page = PoseInferencePage(controller)
    page.set_session(session)
    generation = controller.state.discovery_generation

    page._refresh_discovery()
    page._refresh_discovery()

    assert controller.state.discovery_generation == generation
    assert "cannot refresh run discovery" in page.error_label.text().lower()
    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert gate.max_active == 1
    destroy_page(page)


def test_selected_run_survives_same_session_refresh(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    newer = _run(session, "run-new")
    older = _run(session, "run-old")
    box = {"runs": (newer, older)}
    controller = PoseInferenceController(
        discovery=lambda root: _summary(Path(root), box["runs"]),
        s3_validator=lambda path: Path(path).resolve(strict=False),
    )
    page = PoseInferencePage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    assert page.runs_table.rowCount() == 2

    controller.select_run("run-old")
    page._refresh_discovery()
    assert controller.state.discovery_running is True
    wait_until(lambda: not page.has_running_task)

    assert controller.state.selected_run_id == "run-old"
    assert controller.selected_run is not None
    assert controller.selected_run.run_id == "run-old"
    destroy_page(page)


def test_invalidated_selection_is_reported_not_kept_silently(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, "s")
    newer = _run(session, "run-new")
    older = _run(session, "run-old")
    box = {"runs": (newer, older)}
    controller = PoseInferenceController(
        discovery=lambda root: _summary(Path(root), box["runs"]),
        s3_validator=lambda path: Path(path).resolve(strict=False),
    )
    page = PoseInferencePage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    controller.select_run("run-old")

    box["runs"] = (newer,)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    current_ids = {run.run_id for run in controller.runs}
    assert current_ids == {"run-new"}
    # Never left pointing at a run that no longer exists.
    assert controller.state.selected_run_id != "run-old"
    assert controller.state.selected_run_id in (None, *current_ids)
    assert "run-old" not in page.selected_summary.text()
    assert page.runs_table.rowCount() == 1
    destroy_page(page)


def test_worker_exception_is_surfaced_through_failure_state(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, "s")

    def explode(_root: Path | str) -> PoseInferenceProjectSummary:
        raise RuntimeError("synthetic discovery failure")

    controller = PoseInferenceController(discovery=explode)
    page = PoseInferencePage(controller)
    seen: list[str] = []
    page.unexpected_error.connect(seen.append)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert seen and "synthetic discovery failure" in seen[0]
    assert controller.state.discovery_running is False
    assert controller.state.s1_status is S1UiStatus.INVALID
    assert "synthetic discovery failure" in controller.state.s1_message
    assert "synthetic discovery failure" in page.s1_status_label.text()
    assert page.run_button.isEnabled() is False
    destroy_page(page)


def test_recoverable_discovery_error_is_visible_in_s1_status(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")

    def unreadable(_root: Path | str) -> PoseInferenceProjectSummary:
        raise OSError("disk went away")

    controller = PoseInferenceController(discovery=unreadable)
    page = PoseInferencePage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert controller.state.s1_status is S1UiStatus.INVALID
    assert "disk went away" in page.s1_status_label.text()
    destroy_page(page)


def test_stale_failure_does_not_clobber_state(tmp_path: Path) -> None:
    first = _session(tmp_path, "first")
    second = _session(tmp_path, "second")
    controller = PoseInferenceController(discovery=lambda root: _summary(Path(root)))
    stale_identity = controller.prepare_session(first)
    controller.prepare_session(second)

    assert (
        controller.record_discovery_failure("late failure", expected_identity=stale_identity)
        is False
    )
    assert controller.state.discovery_running is True
    assert "late failure" not in controller.state.s1_message


def test_page_close_and_reopen_during_discovery_is_safe(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    gate = Gate()
    controller = _gated_controller(gate)
    page = PoseInferencePage(controller)
    page.show()
    page.set_session(session)
    wait_until(gate.entered.is_set)

    page.close()  # hidden, worker still running
    gate.release()
    wait_until(lambda: not page.has_running_task)
    destroy_page(page)

    reopened = PoseInferencePage(controller)
    reopened.show()
    reopened.set_session(session)
    wait_until(lambda: not reopened.has_running_task)
    assert controller.state.s1_status is S1UiStatus.READY
    assert reopened.frame_count_label.text() == "12"
    destroy_page(reopened)


def test_discovery_does_not_write_session_artifacts(tmp_path: Path) -> None:
    session = _session(tmp_path, "s")
    _run(session, "run-new")
    before = _tree_fingerprint(session)
    controller = PoseInferenceController(
        discovery=lambda root: _summary(Path(root), ()),
    )
    page = PoseInferencePage(controller)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    assert _tree_fingerprint(session) == before
    destroy_page(page)
