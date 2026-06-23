"""Reusable Qt-safe execution of noticeable non-GUI operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, QThread, Signal, Slot


class TaskAlreadyRunningError(RuntimeError):
    """Raised when a runner is asked to start overlapping work."""


class GuiTaskKind(StrEnum):
    """Long-running GUI operation categories used for identity and cancellation."""

    RAW_PROBE = "raw_probe"
    RAW_SEQUENTIAL_COUNT = "raw_sequential_count"
    CAGE_DETECTION = "cage_detection"
    PREPROCESS_RUN = "preprocess_run"


@dataclass(slots=True)
class GuiTaskContext:
    """Stable task identity plus a cooperative cancellation flag."""

    task_id: str
    task_kind: GuiTaskKind
    project_dir: Path | None
    raw_video_path: Path | None
    generation: int
    cancellation_requested: bool = False

    def request_cancellation(self) -> None:
        """Request cooperative cancellation without force-stopping its thread."""

        self.cancellation_requested = True

    def is_cancellation_requested(self) -> bool:
        """Return a callback-compatible cancellation state."""

        return self.cancellation_requested


def _same_optional_path(first: Path | None, second: Path | None) -> bool:
    if first is None or second is None:
        return first is second
    return first.expanduser().resolve(strict=False) == second.expanduser().resolve(
        strict=False
    )


class GuiTaskCoordinator:
    """Coordinate task identity across page-local Qt task runners."""

    def __init__(self) -> None:
        self._generation = 0
        self._active: dict[str, GuiTaskContext] = {}

    @property
    def generation(self) -> int:
        """Return the current input generation."""

        return self._generation

    @property
    def active_contexts(self) -> tuple[GuiTaskContext, ...]:
        """Return a stable snapshot of registered active tasks."""

        return tuple(self._active.values())

    def begin(
        self,
        *,
        task_kind: GuiTaskKind,
        project_dir: Path | None,
        raw_video_path: Path | None,
    ) -> GuiTaskContext:
        """Register a task, rejecting a duplicate for the same active inputs."""

        for context in self._active.values():
            if (
                context.task_kind is task_kind
                and _same_optional_path(context.project_dir, project_dir)
                and _same_optional_path(context.raw_video_path, raw_video_path)
            ):
                raise TaskAlreadyRunningError(
                    f"A {task_kind.value.replace('_', ' ')} task is already running "
                    "for the selected project and raw video."
                )
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=task_kind,
            project_dir=project_dir,
            raw_video_path=raw_video_path,
            generation=self._generation,
        )
        self._active[context.task_id] = context
        return context

    def invalidate_inputs(self) -> tuple[GuiTaskContext, ...]:
        """Advance generation and cancel probe/count tasks tied to prior inputs."""

        self._generation += 1
        cancelled: list[GuiTaskContext] = []
        for context in self._active.values():
            if context.task_kind in {
                GuiTaskKind.RAW_PROBE,
                GuiTaskKind.RAW_SEQUENTIAL_COUNT,
            }:
                context.request_cancellation()
                cancelled.append(context)
        return tuple(cancelled)

    def request_input_cancellation(self) -> tuple[GuiTaskContext, ...]:
        """Cancel current-generation probe/count work without changing inputs."""

        cancelled: list[GuiTaskContext] = []
        for context in self._active.values():
            if context.task_kind in {
                GuiTaskKind.RAW_PROBE,
                GuiTaskKind.RAW_SEQUENTIAL_COUNT,
            }:
                context.request_cancellation()
                cancelled.append(context)
        return tuple(cancelled)

    def request_all_cancellation(self) -> tuple[GuiTaskContext, ...]:
        """Request cancellation for all tasks during application shutdown."""

        self._generation += 1
        for context in self._active.values():
            context.request_cancellation()
        return self.active_contexts

    def is_current(
        self,
        context: GuiTaskContext,
        *,
        project_dir: Path | None,
        raw_video_path: Path | None,
    ) -> bool:
        """Return whether a result still belongs to the current GUI inputs."""

        return (
            not context.cancellation_requested
            and self.belongs_to_current_inputs(
                context,
                project_dir=project_dir,
                raw_video_path=raw_video_path,
            )
        )

    def belongs_to_current_inputs(
        self,
        context: GuiTaskContext,
        *,
        project_dir: Path | None,
        raw_video_path: Path | None,
    ) -> bool:
        """Return whether identity/generation match, including cancelled tasks."""

        return (
            context.generation == self._generation
            and context.task_id in self._active
            and _same_optional_path(context.project_dir, project_dir)
            and _same_optional_path(context.raw_video_path, raw_video_path)
        )

    def finish(self, context: GuiTaskContext) -> None:
        """Forget a completed task without changing the input generation."""

        self._active.pop(context.task_id, None)


class _TaskWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self._operation = operation

    @Slot()
    def run(self) -> None:
        try:
            result = self._operation()
        except BaseException as exc:
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


class GuiTaskRunner(QObject):
    """Run one callable in a QThread and dispatch callbacks on the GUI thread."""

    started = Signal(str)
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _TaskWorker | None = None
        self._on_success: Callable[[Any], None] | None = None
        self._on_error: Callable[[BaseException], None] | None = None
        self._on_finished: Callable[[], None] | None = None
        self._context: GuiTaskContext | None = None

    @property
    def is_running(self) -> bool:
        """Return whether this runner currently owns an active task thread."""

        return self._thread is not None

    @property
    def active_context(self) -> GuiTaskContext | None:
        """Return the context associated with the current task, if any."""

        return self._context

    def request_cancellation(self) -> None:
        """Request cooperative cancellation of the current task."""

        if self._context is not None:
            self._context.request_cancellation()

    def start(
        self,
        operation: Callable[[], Any],
        *,
        task_name: str = "background task",
        context: GuiTaskContext | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        """Start one operation or reject an overlapping invocation."""

        if self.is_running:
            raise TaskAlreadyRunningError("A GUI background task is already running.")
        thread = QThread(self)
        worker = _TaskWorker(operation)
        worker.moveToThread(thread)
        self._thread = thread
        self._worker = worker
        self._on_success = on_success
        self._on_error = on_error
        self._on_finished = on_finished
        self._context = context

        thread.started.connect(worker.run)
        worker.succeeded.connect(self._dispatch_success)
        worker.failed.connect(self._dispatch_failure)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.started.emit(task_name)
        thread.start()

    @Slot(object)
    def _dispatch_success(self, result: object) -> None:
        self.succeeded.emit(result)
        if self._on_success is not None:
            self._on_success(result)

    @Slot(object)
    def _dispatch_failure(self, exc: object) -> None:
        error = exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
        self.failed.emit(error)
        if self._on_error is not None:
            self._on_error(error)

    @Slot()
    def _thread_finished(self) -> None:
        on_finished = self._on_finished
        self._thread = None
        self._worker = None
        self._on_success = None
        self._on_error = None
        self._on_finished = None
        self._context = None
        self.finished.emit()
        if on_finished is not None:
            on_finished()
