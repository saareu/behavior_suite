"""S8 GUI Cancel enablement, feedback, and MainWindow deferred-close tests."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from tests.behavioral_arbitration._fixtures import build_synthetic_session
from tests.ui.test_behavioral_arbitration_gui import _patch_s7_s6_ok

from behavioral_arbitration import (
    BehavioralArbitrationCancelledError,
    BehavioralArbitrationRequest,
    BehavioralArbitrationRunResult,
    run_behavioral_arbitration_publication,
)
from behavioral_arbitration.contracts import STATUS_CANCELED
from preprocess.models import OperationProgress
from ui.controllers.behavioral_arbitration_controller import (
    BehavioralArbitrationController,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.behavioral_arbitration_page import BehavioralArbitrationPage
from ui.tasks import GuiTaskContext, GuiTaskKind

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


@pytest.fixture
def silence_modals(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))


def _ready_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    runner: Any | None = None,
) -> tuple[BehavioralArbitrationPage, dict[str, Any]]:
    fixture = build_synthetic_session(tmp_path, n_frames=6)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller = (
        BehavioralArbitrationController(runner=runner)
        if runner is not None
        else BehavioralArbitrationController()
    )
    page = BehavioralArbitrationPage(controller)
    page.set_session(fixture["session_root"])
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    return page, fixture


def _gated_cancelable_runner(
    *,
    worker_started: threading.Event,
    hold_until_cancel: bool = True,
):
    def runner(
        request: BehavioralArbitrationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralArbitrationRunResult:
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="arbitration",
                    message="Running Pipeline 100 arbitration stages (Sprint 1–3)…",
                    completed_units=1,
                    total_units=6,
                    is_indeterminate=False,
                )
            )
        if hold_until_cancel:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if cancellation_requested is not None and cancellation_requested():
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("cancel was never requested")
        # Confirmed cancel path without needing full science cancel timing.
        if cancellation_requested is not None and cancellation_requested():
            raise BehavioralArbitrationCancelledError(
                "Behavioral arbitration cancelled before publication."
            )
        return run_behavioral_arbitration_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            publication_started=publication_started,
        )

    return runner


def test_cancel_button_enablement_and_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    worker_started = threading.Event()
    page, _fixture = _ready_page(
        monkeypatch,
        tmp_path,
        runner=_gated_cancelable_runner(worker_started=worker_started),
    )
    assert page.cancel_button.isEnabled() is False

    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)
    assert page.run_button.isEnabled() is False

    page.cancel_button.click()
    assert page.cancel_button.isEnabled() is False
    assert "Cancel requested" in page.stage_label.text()
    # Repeated Cancel clicks are harmless.
    page.cancel_button.click()
    page.cancel_button.click()

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED
    # A new run is permitted after worker completion.
    assert page.run_button.isEnabled() is True
    assert page.cancel_button.isEnabled() is False


def test_pre_bootstrap_cancel_does_not_claim_persisted_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancel before run-dir creation must not invent a persisted canceled path."""

    dialogs: list[tuple[str, str]] = []
    statuses: list[str] = []

    def _capture_information(parent, title, text, *args, **kwargs):  # noqa: ANN001
        del parent, args, kwargs
        dialogs.append((str(title), str(text)))
        return 0

    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(_capture_information)
    )
    for name in ("warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))

    def runner(
        request: BehavioralArbitrationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralArbitrationRunResult:
        del request, progress_callback, cancellation_requested, publication_started
        # Simulate confirmed cancel at checkpoint 1 (before run-dir creation).
        raise BehavioralArbitrationCancelledError(
            "Behavioral arbitration cancelled before publication."
        )

    page, fixture = _ready_page(monkeypatch, tmp_path, runner=runner)
    page.status_message.connect(statuses.append)
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)

    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED
    assert page.controller.state.last_result is None
    arb_root = fixture["session_root"] / "behavioral_arbitration"
    assert not arb_root.exists() or list(arb_root.iterdir()) == []
    assert dialogs
    body = dialogs[-1][1].lower()
    assert "kept for traceability" not in body
    assert "no science artifacts were published" in body
    assert "after run bootstrap" in body
    assert any("no science artifacts published" in msg.lower() for msg in statuses)
    # Subsequent run must remain available after a pre-bootstrap cancel.
    assert page.run_button.isEnabled() is True
    assert page.cancel_button.isEnabled() is False


def test_cancel_unavailable_in_publication_critical_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    entered = threading.Event()
    release = threading.Event()

    def runner(
        request: BehavioralArbitrationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralArbitrationRunResult:
        del cancellation_requested
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="publish",
                    message="Publishing immutable S8 artifacts…",
                    completed_units=5,
                    total_units=6,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        return run_behavioral_arbitration_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page, _fixture = _ready_page(monkeypatch, tmp_path, runner=runner)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )
    assert page.cancel_button.isEnabled() is False
    page._cancel_arbitration()
    assert "Publication in progress" in page.stage_label.text()
    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True


def test_mainwindow_close_during_publication_waits_without_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    fixture = build_synthetic_session(tmp_path, n_frames=6)
    _patch_s7_s6_ok(monkeypatch, fixture)
    entered = threading.Event()
    release = threading.Event()
    cancel_polled = {"requested": False}

    def runner(
        request: BehavioralArbitrationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralArbitrationRunResult:
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="publish",
                    message="Publishing immutable S8 artifacts…",
                    completed_units=5,
                    total_units=6,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        if cancellation_requested is not None:
            cancel_polled["requested"] = bool(cancellation_requested())
        return run_behavioral_arbitration_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    controller = BehavioralArbitrationController(runner=runner)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_arbitration_controller=controller,
    )
    window.open_behavioral_arbitration(fixture["session_root"])
    page = window.behavioral_arbitration_page
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s8_task is True
    assert page._arbitration_cancel_requested is False
    assert page.task_runner.is_running is True
    # Repeated close remains deferred and does not force-stop the worker.
    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted() is False
    assert page.task_runner.is_running is True

    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert cancel_polled["requested"] is False
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    # Successful publication must remain Completed*, never retroactively Canceled.
    assert page.controller.state.run_stage != STATUS_CANCELED
    assert page.stage_label.text() != "Canceled"
    assert "completed" in page.controller.state.run_stage


def test_mainwindow_deferred_close_requests_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    fixture = build_synthetic_session(tmp_path, n_frames=6)
    _patch_s7_s6_ok(monkeypatch, fixture)
    worker_started = threading.Event()
    controller = BehavioralArbitrationController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_arbitration_controller=controller,
    )
    window.open_behavioral_arbitration(fixture["session_root"])
    page = window.behavioral_arbitration_page
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s8_task is True
    assert page._arbitration_cancel_requested is True
    # Repeated close attempts remain idempotent.
    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted() is False
    assert window._close_after_s8_task is True

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"


def test_stale_arbitration_failure_rejected_for_superseded_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    fixture = build_synthetic_session(tmp_path, n_frames=4)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller = BehavioralArbitrationController()
    controller.set_session(fixture["session_root"])
    settings = controller.state.working_settings
    identity = controller.begin_arbitration_launch(settings)
    from dataclasses import replace

    controller.state.active_arbitration_identity = replace(
        identity, request_id="superseded-request"
    )
    assert (
        controller.record_failure(
            "canceled",
            expected_identity=identity,
            canceled=True,
        )
        is False
    )
    assert controller.state.run_stage != STATUS_CANCELED


def test_gui_task_context_publication_flag_defaults() -> None:
    context = GuiTaskContext(
        task_id="t",
        task_kind=GuiTaskKind.BEHAVIORAL_ARBITRATION,
        project_dir=None,
        raw_video_path=None,
        generation=0,
    )
    assert context.is_cancellation_requested() is False
    assert context.is_publication_critical_section() is False
    context.request_cancellation()
    context.request_cancellation()
    assert context.is_cancellation_requested() is True
    context.mark_publication_critical_section()
    assert context.is_publication_critical_section() is True


def test_cancel_after_task_completion_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    page, _fixture = _ready_page(monkeypatch, tmp_path)
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=120.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    completed_stage = page.stage_label.text()
    page._cancel_arbitration()
    assert page.stage_label.text() == completed_stage
    assert page.cancel_button.isEnabled() is False
    assert page.run_button.isEnabled() is True


def test_close_during_active_cancellation_keeps_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    fixture = build_synthetic_session(tmp_path, n_frames=6)
    _patch_s7_s6_ok(monkeypatch, fixture)
    worker_started = threading.Event()
    controller = BehavioralArbitrationController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_arbitration_controller=controller,
    )
    window.open_behavioral_arbitration(fixture["session_root"])
    page = window.behavioral_arbitration_page
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    page.cancel_button.click()
    assert page._arbitration_cancel_requested is True
    assert page.task_runner.is_running is True

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s8_task is True
    assert page.task_runner.is_running is True
    assert page.has_running_task is True

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
