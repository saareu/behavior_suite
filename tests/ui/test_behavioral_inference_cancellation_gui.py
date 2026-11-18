"""S7 GUI Cancel enablement, feedback, and MainWindow deferred-close tests."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from tests.ui.test_behavioral_inference_gui import _ready_controller

from behavioral_inference import (
    STATUS_CANCELED,
    BehavioralInferenceRequest,
    BehavioralInferenceRunResult,
    run_behavioral_inference_publication,
)
from behavioral_inference.config import DEFAULT_WORKING_THRESHOLDS
from preprocess.models import OperationProgress
from ui.controllers.behavioral_inference_controller import (
    BehavioralInferenceUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.behavioral_inference_page import BehavioralInferencePage
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


class _CancelAfterNFalsePolls:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


def _gated_cancelable_runner(
    *,
    worker_started: threading.Event,
    hold_until_cancel: bool = True,
):
    def runner(
        request: BehavioralInferenceRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralInferenceRunResult:
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="running_detector_inference",
                    message="Running detector inference: approach (0/2 detectors completed)",
                    completed_units=0,
                    total_units=2,
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
        cancel_after_bootstrap = _CancelAfterNFalsePolls(2)
        return run_behavioral_inference_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=cancel_after_bootstrap,
            publication_started=publication_started,
        )

    return runner


def _open_ready_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner=None,
) -> BehavioralInferencePage:
    controller, session, registry = _ready_controller(monkeypatch, tmp_path)
    if runner is not None:
        controller._runner = runner
    page = BehavioralInferencePage(controller)
    page.set_session(session)
    _wait_until(lambda: not page.has_running_task)
    page.controller.update_settings(
        BehavioralInferenceUiSettings(
            parity_mode="legacy_parity",
            model_registry_root=registry,
            selected_detectors_dyadic=("nose_to_nose",),
            selected_detectors_focal=("approach",),
            working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
        )
    )
    page.refresh_from_state()
    _wait_until(lambda: page.controller.can_run_inference())
    return page


def test_cancel_button_enablement_and_feedback(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    worker_started = threading.Event()
    page = _open_ready_page(
        tmp_path,
        monkeypatch,
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
    assert "current detector" in page.stage_label.text()
    page.cancel_button.click()
    page.cancel_button.click()

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED
    assert page.run_button.isEnabled() is True
    assert page.cancel_button.isEnabled() is False
    page.close()


def test_cancel_unavailable_in_publication_critical_section(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    entered = threading.Event()
    release = threading.Event()

    def runner(
        request: BehavioralInferenceRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralInferenceRunResult:
        del cancellation_requested
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="writing_artifacts",
                    message="Writing artifacts",
                    completed_units=1,
                    total_units=1,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        return run_behavioral_inference_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page = _open_ready_page(tmp_path, monkeypatch, runner=runner)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )
    assert page.cancel_button.isEnabled() is False
    page._cancel_inference()
    assert "Publication in progress" in page.stage_label.text()
    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    page.close()


def test_stale_progress_ignored_after_session_switch(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    hold = threading.Event()
    started = threading.Event()

    def runner(
        request: BehavioralInferenceRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralInferenceRunResult:
        del cancellation_requested, publication_started
        started.set()
        assert hold.wait(timeout=10.0)
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="running_detector_inference",
                    message="stale progress",
                    completed_units=0,
                    total_units=None,
                    is_indeterminate=True,
                )
            )
        return run_behavioral_inference_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page = _open_ready_page(tmp_path / "a", monkeypatch, runner=runner)
    session_a = page.controller.state.session_root
    page.run_button.click()
    _wait_until(lambda: started.is_set())
    _other_controller, other_session, _registry = _ready_controller(
        monkeypatch, tmp_path / "b"
    )
    page.set_session(other_session)
    assert page.controller.state.session_root == session_a
    hold.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    page.close()


def test_mainwindow_close_during_publication_waits_without_cancel(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    entered = threading.Event()
    release = threading.Event()
    cancel_polled = {"requested": False}

    def runner(
        request: BehavioralInferenceRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralInferenceRunResult:
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="writing_artifacts",
                    message="Writing artifacts",
                    completed_units=1,
                    total_units=1,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        if cancellation_requested is not None:
            cancel_polled["requested"] = bool(cancellation_requested())
        return run_behavioral_inference_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
            publication_started=None,
        )

    page = _open_ready_page(tmp_path, monkeypatch, runner=runner)
    window = MainWindow(PreprocessSetupController())
    monkeypatch.setattr(window, "behavioral_inference_page", page)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s7_task is True
    assert page._inference_cancel_requested is False
    assert page.task_runner.is_running is True
    assert page.cancel_button.isEnabled() is False
    assert "publishing" in page.request_cooperative_shutdown().lower()

    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted() is False
    assert page.task_runner.is_running is True

    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert cancel_polled["requested"] is False
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    assert page.controller.state.run_stage != STATUS_CANCELED
    page.close()
    window.close()


def test_mainwindow_defers_close_and_requests_s7_cancel(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    worker_started = threading.Event()
    page = _open_ready_page(
        tmp_path,
        monkeypatch,
        runner=_gated_cancelable_runner(worker_started=worker_started),
    )
    window = MainWindow(PreprocessSetupController())
    monkeypatch.setattr(window, "behavioral_inference_page", page)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert page._inference_cancel_requested is True
    assert "Cancel requested" in page.stage_label.text()

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    page.close()
    window.close()


def test_rerun_after_gui_cancel(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    clock = {"n": 0}

    def _fake_now() -> str:
        clock["n"] += 1
        return f"20260101T2{clock['n']:05d}"

    monkeypatch.setattr("behavioral_inference.runner._now_timestamp", _fake_now)
    worker_started = threading.Event()
    page = _open_ready_page(
        tmp_path,
        monkeypatch,
        runner=_gated_cancelable_runner(worker_started=worker_started),
    )
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    page.cancel_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"

    page.controller._runner = run_behavioral_inference_publication
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text().startswith("Finished:")
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.status in {"completed", "completed_with_warnings"}
    page.close()


def test_gui_task_context_publication_flag() -> None:
    context = GuiTaskContext(
        task_id="s7",
        task_kind=GuiTaskKind.BEHAVIORAL_INFERENCE,
        project_dir=None,
        raw_video_path=None,
        generation=0,
    )
    assert context.is_publication_critical_section() is False
    context.mark_publication_critical_section()
    assert context.is_publication_critical_section() is True
    context.request_cancellation()
    assert context.is_cancellation_requested() is True


def test_pre_bootstrap_cancel_message_reports_no_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from behavioral_inference import BehavioralInferenceCancelledError

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
        request: BehavioralInferenceRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralInferenceRunResult:
        del request, progress_callback, cancellation_requested, publication_started
        raise BehavioralInferenceCancelledError("Behavioral inference cancelled.")

    page = _open_ready_page(tmp_path, monkeypatch, runner=runner)
    page.status_message.connect(statuses.append)
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)

    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED
    session = page.controller.state.session_root
    assert session is not None
    root = session / "behavioral_inference"
    assert not root.exists() or list(root.iterdir()) == []
    assert dialogs
    body = dialogs[-1][1].lower()
    assert "before run bootstrap" in body
    assert "no s7 run directory was created" in body
    assert "kept for traceability" not in body
    assert any("before run bootstrap" in msg.lower() for msg in statuses)
    page.close()


def test_post_bootstrap_cancel_message_reports_retained_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    worker_started = threading.Event()
    page = _open_ready_page(
        tmp_path,
        monkeypatch,
        runner=_gated_cancelable_runner(worker_started=worker_started),
    )
    page.status_message.connect(statuses.append)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    page.cancel_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)

    assert page.stage_label.text() == "Canceled"
    session = page.controller.state.session_root
    assert session is not None
    runs = list((session / "behavioral_inference").iterdir())
    assert len(runs) == 1
    assert dialogs
    body = dialogs[-1][1]
    assert "Canceled run directory retained:" in body
    assert runs[0].name in body
    assert "Metadata/log were kept for traceability" in body
    assert any("retained" in msg.lower() for msg in statuses)
    page.close()


def test_soft_navigation_remains_available_during_s7_cancel_request(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui.main_window import GROUP_BEHAVIORAL_INFERENCE, GROUP_DATA_PREPARATION

    del silence_modals
    worker_started = threading.Event()
    controller, session, registry = _ready_controller(monkeypatch, tmp_path)
    controller._runner = _gated_cancelable_runner(worker_started=worker_started)
    window = MainWindow(
        PreprocessSetupController(),
        behavioral_inference_controller=controller,
    )
    window.open_behavioral_inference(session)
    page = window.behavioral_inference_page
    _wait_until(lambda: not page.has_running_task)
    page.controller.update_settings(
        BehavioralInferenceUiSettings(
            parity_mode="legacy_parity",
            model_registry_root=registry,
            selected_detectors_dyadic=("nose_to_nose",),
            selected_detectors_focal=("approach",),
            working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
        )
    )
    page.refresh_from_state()
    _wait_until(lambda: page.controller.can_run_inference())
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    page.cancel_button.click()
    assert page._inference_cancel_requested is True
    assert page.has_running_task is True
    window._update_navigation()
    assert window._hard_navigation_blocked() is False
    assert window.group_data_button.isEnabled() is True
    assert window.group_feature_button.isEnabled() is True
    assert window.s7_button.isEnabled() is True
    active_identity = page.controller.state.active_inference_identity
    window.set_application_group(GROUP_DATA_PREPARATION)
    assert page.controller.state.active_inference_identity is active_identity
    window.set_application_group(GROUP_BEHAVIORAL_INFERENCE)
    assert page.controller.state.active_inference_identity is active_identity
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    window.close()
