from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer

from preprocess.exceptions import CageDetectionError
from ui.tasks import (
    GuiTaskCoordinator,
    GuiTaskKind,
    GuiTaskRunner,
    TaskAlreadyRunningError,
)


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


def test_task_coordinator_rejects_duplicate_count_identity(tmp_path) -> None:
    coordinator = GuiTaskCoordinator()
    project_dir = tmp_path / "project"
    raw_path = tmp_path / "raw.avi"
    first = coordinator.begin(
        task_kind=GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        project_dir=project_dir,
        raw_video_path=raw_path,
    )

    with pytest.raises(TaskAlreadyRunningError, match="already running"):
        coordinator.begin(
            task_kind=GuiTaskKind.RAW_SEQUENTIAL_COUNT,
            project_dir=project_dir,
            raw_video_path=raw_path,
        )

    assert first.task_id
    assert first.task_kind is GuiTaskKind.RAW_SEQUENTIAL_COUNT
    assert first.project_dir == project_dir
    assert first.raw_video_path == raw_path
    assert first.generation == coordinator.generation
    assert first.cancellation_requested is False


def test_input_generation_cancels_old_context_and_makes_result_stale(
    tmp_path,
) -> None:
    coordinator = GuiTaskCoordinator()
    project_dir = tmp_path / "project"
    raw_path = tmp_path / "raw.avi"
    context = coordinator.begin(
        task_kind=GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        project_dir=project_dir,
        raw_video_path=raw_path,
    )

    coordinator.invalidate_inputs()

    assert context.cancellation_requested is True
    assert coordinator.generation == context.generation + 1
    assert (
        coordinator.is_current(
            context,
            project_dir=project_dir,
            raw_video_path=raw_path,
        )
        is False
    )


def test_runner_exposes_context_and_forwards_cancellation_request(
    application: QCoreApplication,
    tmp_path,
) -> None:
    del application
    coordinator = GuiTaskCoordinator()
    context = coordinator.begin(
        task_kind=GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        project_dir=tmp_path,
        raw_video_path=tmp_path / "raw.avi",
    )
    started = threading.Event()
    runner = GuiTaskRunner()

    def wait_for_cancellation() -> bool:
        started.set()
        while not context.is_cancellation_requested():
            time.sleep(0.001)
        return True

    runner.start(wait_for_cancellation, context=context)
    assert started.wait(timeout=1.0)
    assert runner.active_context is context

    runner.request_cancellation()
    _wait_for_runner(runner)

    assert context.cancellation_requested is True
