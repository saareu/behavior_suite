"""Reusable Qt-safe execution of noticeable non-GUI operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class TaskAlreadyRunningError(RuntimeError):
    """Raised when a runner is asked to start overlapping work."""


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

    @property
    def is_running(self) -> bool:
        """Return whether this runner currently owns an active task thread."""

        return self._thread is not None

    def start(
        self,
        operation: Callable[[], Any],
        *,
        task_name: str = "background task",
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
        self.finished.emit()
        if on_finished is not None:
            on_finished()
