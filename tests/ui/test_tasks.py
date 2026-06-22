from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer

from preprocess.exceptions import CageDetectionError
from ui.tasks import GuiTaskRunner, TaskAlreadyRunningError


@pytest.fixture(scope="module")
def application() -> QCoreApplication:
    return QCoreApplication.instance() or QCoreApplication([])


def _wait_for_runner(runner: GuiTaskRunner, timeout_ms: int = 2_000) -> None:
    loop = QEventLoop()
    timed_out = False

    def timeout() -> None:
        nonlocal timed_out
        timed_out = True
        loop.quit()

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(timeout)
    runner.finished.connect(loop.quit)
    timer.start(timeout_ms)
    if runner.is_running:
        loop.exec()
    timer.stop()
    assert timed_out is False, "background task did not finish before the test timeout"


def test_task_wrapper_returns_success_to_main_thread_callback(
    application: QCoreApplication,
) -> None:
    runner = GuiTaskRunner()
    callbacks: list[tuple[object, QThread]] = []

    runner.start(
        lambda: 42,
        on_success=lambda result: callbacks.append(
            (result, QThread.currentThread())
        ),
    )
    _wait_for_runner(runner)

    assert callbacks == [(42, application.thread())]
    assert runner.is_running is False


def test_task_wrapper_forwards_expected_domain_errors(
    application: QCoreApplication,
) -> None:
    del application
    runner = GuiTaskRunner()
    errors: list[BaseException] = []

    def fail() -> None:
        raise CageDetectionError("expected detection failure")

    runner.start(fail, on_error=errors.append)
    _wait_for_runner(runner)

    assert len(errors) == 1
    assert isinstance(errors[0], CageDetectionError)
    assert str(errors[0]) == "expected detection failure"


def test_task_wrapper_rejects_duplicate_concurrent_runs(
    application: QCoreApplication,
) -> None:
    del application
    release = threading.Event()
    runner = GuiTaskRunner()
    runner.start(lambda: release.wait(timeout=1.0))

    with pytest.raises(TaskAlreadyRunningError, match="already running"):
        runner.start(lambda: None)

    release.set()
    _wait_for_runner(runner)
    assert runner.is_running is False
