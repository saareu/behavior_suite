from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer

from preprocess.exceptions import CageDetectionError
from preprocess.models import OperationProgress
from ui.tasks import (
    GuiTaskCoordinator,
    GuiTaskKind,
    GuiTaskProgressEvent,
    GuiTaskProgressTracker,
    GuiTaskRunner,
    TaskAlreadyRunningError,
    format_raw_count_progress,
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


def test_progressive_task_returns_typed_progress_to_main_thread(
    application: QCoreApplication,
    tmp_path,
) -> None:
    coordinator = GuiTaskCoordinator()
    context = coordinator.begin(
        task_kind=GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        project_dir=tmp_path,
        raw_video_path=tmp_path / "raw.avi",
    )
    runner = GuiTaskRunner()
    callbacks: list[tuple[GuiTaskProgressEvent, QThread]] = []

    def operation(report) -> int:
        report(
            OperationProgress(
                phase="counting",
                message="Counting sequentially readable frames",
                completed_units=250,
                total_units=1000,
                total_is_estimate=True,
                is_indeterminate=False,
            )
        )
        return 250

    runner.start_progressive(
        operation,
        context=context,
        on_progress=lambda event: callbacks.append(
            (event, QThread.currentThread())
        ),
    )
    _wait_for_runner(runner)

    assert len(callbacks) == 1
    event, callback_thread = callbacks[0]
    assert callback_thread is application.thread()
    assert event.task_id == context.task_id
    assert event.task_kind is GuiTaskKind.RAW_SEQUENTIAL_COUNT
    assert event.generation == context.generation
    assert event.completed_units == 250
    assert event.total_units == 1000
    assert event.total_is_estimate is True
    assert event.is_indeterminate is False


def test_progress_eta_is_hidden_until_rate_is_stable(tmp_path) -> None:
    context = GuiTaskCoordinator().begin(
        task_kind=GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        project_dir=tmp_path,
        raw_video_path=tmp_path / "raw.avi",
    )
    now = [0.0]
    tracker = GuiTaskProgressTracker(context, clock=lambda: now[0])

    def update(
        completed: int,
        *,
        total: int | None = 1000,
        total_is_estimate: bool = False,
        is_indeterminate: bool = False,
    ) -> OperationProgress:
        return OperationProgress(
            phase="counting",
            message="Counting",
            completed_units=completed,
            total_units=total,
            total_is_estimate=total_is_estimate,
            is_indeterminate=is_indeterminate,
        )

    initial = tracker.build(update(0))
    now[0] = 1.0
    second = tracker.build(update(100))
    now[0] = 2.0
    stable = tracker.build(update(200))
    now[0] = 3.0
    estimated_total = tracker.build(update(300, total_is_estimate=True))
    now[0] = 4.0
    unknown_total = tracker.build(
        update(400, total=None, is_indeterminate=True)
    )

    assert initial.estimated_remaining_sec is None
    assert second.estimated_remaining_sec is None
    assert stable.estimated_remaining_sec == pytest.approx(8.0)
    assert estimated_total.estimated_remaining_sec is None
    assert unknown_total.estimated_remaining_sec is None


def test_coordinator_rejects_stale_progress_generation(tmp_path) -> None:
    coordinator = GuiTaskCoordinator()
    context = coordinator.begin(
        task_kind=GuiTaskKind.CAGE_DETECTION,
        project_dir=tmp_path,
        raw_video_path=tmp_path / "raw.avi",
    )
    event = GuiTaskProgressTracker(context).build(
        OperationProgress(
            phase="Loading representative frames",
            message="Loading representative frames",
            is_indeterminate=True,
        )
    )

    coordinator.invalidate_inputs()

    assert context.cancellation_requested is True
    assert coordinator.progress_is_current(
        event,
        context,
        project_dir=tmp_path,
        raw_video_path=tmp_path / "raw.avi",
    ) is False


def test_current_generation_completion_can_refresh_visible_status_headlessly(
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
    runner = GuiTaskRunner()
    visible_status: list[str] = []

    def operation(report) -> int:
        report(
            OperationProgress(
                phase="complete",
                message="Sequential readable-frame count complete",
                completed_units=3,
                total_units=None,
                is_indeterminate=True,
            )
        )
        return 3

    def show_progress(event: GuiTaskProgressEvent) -> None:
        if coordinator.progress_is_current(
            event,
            context,
            project_dir=tmp_path,
            raw_video_path=tmp_path / "raw.avi",
        ):
            visible_status.append(format_raw_count_progress(event).splitlines()[0])

    runner.start_progressive(
        operation,
        context=context,
        on_progress=show_progress,
        on_success=lambda count: visible_status.append(
            f"Sequential count complete: {count} readable frames"
        ),
    )
    _wait_for_runner(runner)

    assert visible_status == [
        "Counting readable frames: 3 decoded",
        "Sequential count complete: 3 readable frames",
    ]
