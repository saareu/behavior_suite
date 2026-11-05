"""S6 GUI responsiveness: asynchronous S4/S5/S6 discovery, stale rejection."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest
from tests.combined_features.helpers import write_joint_session
from tests.ui._responsiveness_support import (
    APPLICATION,
    SCHEDULE_BUDGET_SEC,
    Gate,
    count_event_loop_ticks,
    destroy_page,
    silence_modal_dialogs,
    wait_until,
)
from tests.ui.test_combined_features_gui import _patch_upstream_eligible

from combined_features import CombinedFeaturesError
from ui.controllers.combined_features_controller import (
    CombinedFeaturesController,
    DiscoverySnapshot,
)
from ui.pages.combined_features_page import CombinedFeaturesPage

_ = APPLICATION

_MODULE = "ui.controllers.combined_features_controller"
_OLDER_S4 = "pose_finalization__20261021T120000"
_OLDER_S5 = "blob_features__20261021T120100"
_NEWER_S4 = "pose_finalization__20261021T130000"
_NEWER_S5 = "blob_features__20261021T130100"


@pytest.fixture(autouse=True)
def _no_blocking_qt_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    silence_modal_dialogs(monkeypatch)


@pytest.fixture(autouse=True)
def _upstream_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_eligible(monkeypatch)


def _session(tmp_path: Path, name: str = "one") -> Path:
    return write_joint_session(tmp_path / name, n_samples=4)["session_root"].resolve()


def _two_pair_session(tmp_path: Path) -> Path:
    paths = write_joint_session(
        tmp_path, s4_run_id=_OLDER_S4, s5_run_id=_OLDER_S5, n_samples=4
    )
    write_joint_session(
        tmp_path,
        s4_run_id=_NEWER_S4,
        s5_run_id=_NEWER_S5,
        sync_path=paths["prepared_sync"],
        n_samples=4,
    )
    return paths["session_root"].resolve()


def _install_gate(monkeypatch: pytest.MonkeyPatch, gate: Gate) -> None:
    """Pass the pair scan through ``gate`` and record every other worker call."""

    import ui.controllers.combined_features_controller as module

    real_pairs = module.list_eligible_joint_pairs
    real_s4 = module.summarize_pose_finalization_project
    real_s5 = module.summarize_blob_features_project
    real_s6 = module.summarize_combined_features_project

    def pairs(root: Path | str):
        with gate.section(Path(root)):
            return real_pairs(root)

    def recording(real):
        def call(root: Path | str):
            gate.thread_ids.append(threading.get_ident())
            gate.gui_thread_flags.append(
                threading.current_thread() is threading.main_thread()
            )
            return real(root)

        return call

    monkeypatch.setattr(module, "list_eligible_joint_pairs", pairs)
    monkeypatch.setattr(module, "summarize_pose_finalization_project", recording(real_s4))
    monkeypatch.setattr(module, "summarize_blob_features_project", recording(real_s5))
    monkeypatch.setattr(module, "summarize_combined_features_project", recording(real_s6))


def _fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _state_tuple(controller: CombinedFeaturesController) -> tuple[object, ...]:
    state = controller.state
    return (
        state.s4_options,
        state.s5_options,
        state.joint_pair_reports,
        state.summary,
        state.selected_s4_run_id,
        state.selected_s5_run_id,
        state.alignment,
        state.selected_s6_run_id,
    )


def test_discovery_runs_outside_gui_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)
    gate = Gate(blocking=False)
    _install_gate(monkeypatch, gate)
    page = CombinedFeaturesPage(CombinedFeaturesController())

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert gate.calls == 1
    assert gate.never_on_gui_thread()
    # Pair scan plus the three upstream summaries all ran on the same worker thread.
    assert len(gate.thread_ids) == 4
    assert threading.get_ident() not in gate.thread_ids
    assert len(set(gate.thread_ids)) == 1
    assert not any(gate.gui_thread_flags)
    assert page.controller.state.summary is not None
    assert page.controller.can_run_extraction() is True
    destroy_page(page)


def test_gui_stays_responsive_during_slow_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.show()

    started = time.perf_counter()
    page.set_session(session)
    elapsed = time.perf_counter() - started

    assert elapsed < SCHEDULE_BUDGET_SEC
    assert controller.state.discovery_running is True
    assert page.has_running_task is True
    assert page.run_button.isEnabled() is False
    assert page.refresh_button.isEnabled() is False
    assert controller.can_run_extraction() is False
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
    assert controller.can_run_extraction() is True
    assert page.run_button.isEnabled() is True
    destroy_page(page)


def test_session_change_rejects_stale_discovery_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _session(tmp_path, "one")
    second = _session(tmp_path, "two")
    controller = CombinedFeaturesController()
    first_identity = controller.prepare_session(first)
    first_snapshot = controller.collect_discovery_snapshot(first_identity)

    gate = Gate()
    _install_gate(monkeypatch, gate)
    page = CombinedFeaturesPage(controller)
    page.set_session(second)
    second_identity = controller.state.active_discovery_identity
    assert second_identity is not None
    assert second_identity.request_id != first_identity.request_id
    assert second_identity.generation > first_identity.generation

    before = _state_tuple(controller)
    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert _state_tuple(controller) == before  # nothing partially applied
    assert controller.state.discovery_running is True

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.session_root == second
    assert controller.state.summary is not None
    assert controller.state.summary.session_root == second
    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert controller.state.summary.session_root == second
    assert isinstance(first_snapshot, DiscoverySnapshot)
    destroy_page(page)


def test_out_of_order_completion_cannot_overwrite_newer_results(
    tmp_path: Path,
) -> None:
    first = _session(tmp_path, "one")
    second = _session(tmp_path, "two")
    controller = CombinedFeaturesController()

    first_identity = controller.prepare_session(first)
    first_snapshot = controller.collect_discovery_snapshot(first_identity)
    second_identity = controller.prepare_session(second)
    second_snapshot = controller.collect_discovery_snapshot(second_identity)

    assert controller.apply_discovery_snapshot(second_snapshot) is True
    applied = _state_tuple(controller)
    assert controller.apply_discovery_snapshot(first_snapshot) is False
    assert _state_tuple(controller) == applied
    assert controller.state.summary is second_snapshot.summary

    older = controller.begin_discovery()
    newer = controller.begin_discovery()
    older_snapshot = controller.collect_discovery_snapshot(older)
    newer_snapshot = controller.collect_discovery_snapshot(newer)
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.discovery_running is True
    assert controller.apply_discovery_snapshot(newer_snapshot) is True
    assert controller.apply_discovery_snapshot(older_snapshot) is False
    assert controller.state.summary is newer_snapshot.summary


def test_snapshot_is_applied_whole_with_compatible_pairing(tmp_path: Path) -> None:
    session = _session(tmp_path)
    controller = CombinedFeaturesController()
    identity = controller.prepare_session(session)
    assert controller.state.s4_options == ()
    assert controller.state.alignment is None
    assert controller.can_run_extraction() is False

    snapshot = controller.collect_discovery_snapshot(identity)
    assert controller.apply_discovery_snapshot(snapshot) is True

    state = controller.state
    assert state.s4_options == snapshot.s4_options
    assert state.s5_options == snapshot.s5_options
    assert state.joint_pair_reports == snapshot.joint_pair_reports
    assert state.summary is snapshot.summary
    s4 = controller.selected_s4_source()
    s5 = controller.selected_s5_source()
    assert s4 is not None and s5 is not None
    # The selected pair, its alignment, and the cached pair report agree.
    matching = [
        report
        for left, right, report in snapshot.joint_pair_reports
        if left.run_id == s4.run_id and right.run_id == s5.run_id
    ]
    assert matching and state.alignment is matching[0]
    assert state.alignment is not None and state.alignment.eligible is True
    assert controller.can_run_extraction() is True


def test_repeated_open_does_not_start_parallel_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)

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
    assert controller.can_run_extraction() is True
    destroy_page(page)


def test_repeated_refresh_coalesces_into_single_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.set_session(session)
    generation = controller.state.discovery_generation
    # The refresh button is disabled while discovery runs...
    assert page.refresh_button.isEnabled() is False
    wait_until(gate.entered.is_set)

    # ...and programmatic refreshes bump the generation without adding workers.
    page._refresh_discovery()
    page._refresh_discovery()
    assert controller.state.discovery_generation == generation + 2
    assert page.discovery_runner.is_running is True

    gate.release()
    wait_until(lambda: not page.has_running_task)
    assert gate.max_active == 1
    assert 1 <= gate.calls <= 2
    assert controller.state.discovery_running is False
    assert controller.can_run_extraction() is True
    destroy_page(page)


def test_manual_s4_s5_selection_survives_same_session_refresh(
    tmp_path: Path,
) -> None:
    session = _two_pair_session(tmp_path)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    # Default is the newest aligned pair.
    assert controller.state.selected_s4_run_id == _NEWER_S4

    controller.select_s4_source(_OLDER_S4, manual=True)
    controller.select_s5_source(_OLDER_S5, manual=True)
    page._refresh_discovery()
    assert controller.state.discovery_running is True
    wait_until(lambda: not page.has_running_task)

    assert controller.state.selected_s4_run_id == _OLDER_S4
    assert controller.state.selected_s5_run_id == _OLDER_S5
    assert controller.state.s4_selection_manual is True
    assert controller.state.s5_selection_manual is True
    assert controller.state.alignment is not None
    assert controller.state.alignment.eligible is True
    assert controller.can_run_extraction() is True

    # A same-session reopen (e.g. via shared-session navigation) keeps it too.
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    assert controller.state.selected_s4_run_id == _OLDER_S4
    assert controller.state.selected_s5_run_id == _OLDER_S5
    destroy_page(page)


def test_invalidated_manual_selection_is_reported_not_silently_kept(
    tmp_path: Path,
) -> None:
    session = _two_pair_session(tmp_path)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    controller.select_s4_source(_OLDER_S4, manual=True)
    controller.select_s5_source(_OLDER_S5, manual=True)

    shutil.rmtree(session / "pose_finalization" / _OLDER_S4)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    state = controller.state
    assert _OLDER_S4 not in {option.run_id for option in state.s4_options}
    assert state.selected_s4_run_id != _OLDER_S4
    assert state.s4_selection_manual is False
    # The still-valid manual S5 choice is not replaced behind the user's back...
    assert state.selected_s5_run_id == _OLDER_S5
    assert state.s5_selection_manual is True
    # ...and S6 refuses to run on the half-selected pair with a readable reason.
    assert controller.selected_s4_source() is None
    assert state.alignment is None
    assert controller.can_run_extraction() is False
    assert controller.cannot_run_reason()
    assert page.run_button.isEnabled() is False
    destroy_page(page)


def test_invalidated_manual_selection_reason_is_not_stale_discovering_text(
    tmp_path: Path,
) -> None:
    session = _two_pair_session(tmp_path)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    controller.select_s4_source(_OLDER_S4, manual=True)
    controller.select_s5_source(_OLDER_S5, manual=True)

    shutil.rmtree(session / "pose_finalization" / _OLDER_S4)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    reason = controller.cannot_run_reason()
    destroy_page(page)
    assert reason
    assert "discovering" not in reason.lower()


def test_selected_s6_run_survives_refresh(tmp_path: Path) -> None:
    session = _session(tmp_path)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    # No S6 run exists yet; selection must be empty rather than invented.
    assert controller.state.selected_s6_run_id is None
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)
    assert controller.state.selected_s6_run_id is None
    destroy_page(page)


def test_worker_errors_are_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)

    def broken(_root: Path | str):
        raise CombinedFeaturesError("synthetic S6 scan failure")

    monkeypatch.setattr(f"{_MODULE}.summarize_combined_features_project", broken)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    messages: list[str] = []
    page.status_message.connect(messages.append)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert controller.state.discovery_running is False
    assert controller.state.readiness_error == "synthetic S6 scan failure"
    assert any("S6 discovery failed" in message for message in messages)
    assert controller.can_run_extraction() is False
    assert page.run_button.isEnabled() is False
    destroy_page(page)


def test_unexpected_worker_exception_emits_unexpected_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)

    def broken(_root: Path | str):
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(f"{_MODULE}.summarize_blob_features_project", broken)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    seen: list[str] = []
    page.unexpected_error.connect(seen.append)

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)

    assert seen and "synthetic unexpected failure" in seen[0]
    assert controller.state.readiness_error == "synthetic unexpected failure"
    assert controller.state.discovery_running is False
    destroy_page(page)


def test_stale_failure_does_not_clobber_state(tmp_path: Path) -> None:
    first = _session(tmp_path, "one")
    second = _session(tmp_path, "two")
    controller = CombinedFeaturesController()
    stale_identity = controller.prepare_session(first)
    controller.prepare_session(second)

    assert (
        controller.record_discovery_failure("late failure", expected_identity=stale_identity)
        is False
    )
    assert controller.state.readiness_error != "late failure"
    assert controller.state.discovery_running is True


def test_page_close_and_reopen_during_discovery_is_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(tmp_path)
    gate = Gate()
    _install_gate(monkeypatch, gate)
    controller = CombinedFeaturesController()
    page = CombinedFeaturesPage(controller)
    page.show()
    page.set_session(session)
    wait_until(gate.entered.is_set)

    page.close()
    gate.release()
    wait_until(lambda: not page.has_running_task)
    destroy_page(page)

    reopened = CombinedFeaturesPage(controller)
    reopened.show()
    reopened.set_session(session)
    wait_until(lambda: not reopened.has_running_task)
    assert controller.can_run_extraction() is True
    assert reopened.run_button.isEnabled() is True
    destroy_page(reopened)


def test_discovery_does_not_write_session_artifacts(tmp_path: Path) -> None:
    session = _session(tmp_path)
    before = _fingerprint(session)
    page = CombinedFeaturesPage(CombinedFeaturesController())

    page.set_session(session)
    wait_until(lambda: not page.has_running_task)
    page._refresh_discovery()
    wait_until(lambda: not page.has_running_task)

    assert _fingerprint(session) == before
    assert not (session / "combined_features").exists()
    destroy_page(page)
