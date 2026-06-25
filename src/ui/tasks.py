"""Reusable Qt-safe execution of noticeable non-GUI operations."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, QThread, Signal, Slot

from preprocess.models import OperationProgress


class TaskAlreadyRunningError(RuntimeError):
    """Raised when a runner is asked to start overlapping work."""


class GuiTaskKind(StrEnum):
    """Long-running GUI operation categories used for identity and cancellation."""

    RAW_PROBE = "raw_probe"
    RAW_SEQUENTIAL_COUNT = "raw_sequential_count"
    RAW_FRAME_READ = "raw_frame_read"
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


@dataclass(frozen=True, slots=True)
class GuiTaskProgressEvent:
    """One typed, task-identifiable progress update delivered on the GUI thread."""

    task_id: str
    task_kind: GuiTaskKind
    generation: int
    phase: str
    message: str
    completed_units: int | None
    total_units: int | None
    total_is_estimate: bool
    elapsed_sec: float
    estimated_remaining_sec: float | None
    is_indeterminate: bool
    cancellation_requested: bool
    processed_video_time_sec: float | None


class GuiTaskProgressTracker:
    """Add task identity, elapsed time, and conservative ETA to core progress."""

    _ETA_MIN_ELAPSED_SEC = 2.0
    _ETA_MIN_COMPLETED_UNITS = 100
    _ETA_RATE_INTERVALS = 2
    _ETA_MAX_RELATIVE_RATE_SPREAD = 0.25

    def __init__(
        self,
        context: GuiTaskContext,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._context = context
        self._clock = clock
        self._started_at = clock()
        self._samples: deque[tuple[float, int]] = deque(maxlen=5)
        self._phase: str | None = None
        self._phase_started_at_elapsed = 0.0

    def build(self, progress: OperationProgress) -> GuiTaskProgressEvent:
        """Build one GUI event without inventing progress unavailable from core."""

        elapsed = max(0.0, self._clock() - self._started_at)
        if progress.phase != self._phase:
            self._samples.clear()
            self._phase = progress.phase
            self._phase_started_at_elapsed = elapsed
        completed = progress.completed_units
        if completed is not None and (
            not self._samples or completed >= self._samples[-1][1]
        ):
            self._samples.append((elapsed, completed))
        eta = self._estimate_remaining(
            progress,
            elapsed,
            elapsed - self._phase_started_at_elapsed,
        )
        return GuiTaskProgressEvent(
            task_id=self._context.task_id,
            task_kind=self._context.task_kind,
            generation=self._context.generation,
            phase=progress.phase,
            message=progress.message,
            completed_units=completed,
            total_units=progress.total_units,
            total_is_estimate=progress.total_is_estimate,
            elapsed_sec=elapsed,
            estimated_remaining_sec=eta,
            is_indeterminate=progress.is_indeterminate,
            cancellation_requested=self._context.cancellation_requested,
            processed_video_time_sec=progress.processed_video_time_sec,
        )

    def _estimate_remaining(
        self,
        progress: OperationProgress,
        elapsed: float,
        phase_elapsed: float,
    ) -> float | None:
        completed = progress.completed_units
        total = progress.total_units
        if (
            completed is None
            or total is None
            or progress.total_is_estimate
            or progress.is_indeterminate
            or completed >= total
            or completed < self._ETA_MIN_COMPLETED_UNITS
            or phase_elapsed < self._ETA_MIN_ELAPSED_SEC
            or len(self._samples) < self._ETA_RATE_INTERVALS + 1
        ):
            return None
        rates: list[float] = []
        samples = tuple(self._samples)
        for previous, current in zip(samples, samples[1:], strict=False):
            delta_time = current[0] - previous[0]
            delta_units = current[1] - previous[1]
            if delta_time > 0 and delta_units > 0:
                rates.append(delta_units / delta_time)
        if len(rates) < self._ETA_RATE_INTERVALS:
            return None
        mean_rate = sum(rates) / len(rates)
        if not math.isfinite(mean_rate) or mean_rate <= 0:
            return None
        relative_spread = (max(rates) - min(rates)) / mean_rate
        if relative_spread > self._ETA_MAX_RELATIVE_RATE_SPREAD:
            return None
        estimate = (total - completed) / mean_rate
        return estimate if math.isfinite(estimate) and estimate >= 0 else None


def progress_matches_context(
    event: GuiTaskProgressEvent,
    context: GuiTaskContext,
) -> bool:
    """Return whether a progress event belongs to exactly one task generation."""

    return (
        event.task_id == context.task_id
        and event.task_kind is context.task_kind
        and event.generation == context.generation
    )


def format_task_duration(seconds: float) -> str:
    """Format elapsed/remaining seconds compactly for status text."""

    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def format_raw_count_progress(event: GuiTaskProgressEvent) -> str:
    """Render an honest sequential-count status without implying an exact total."""

    completed = event.completed_units or 0
    if event.total_units is None:
        first_line = f"Counting readable frames: {completed:,} decoded"
    else:
        qualifier = "approximately " if event.total_is_estimate else ""
        first_line = (
            f"Counting readable frames: {completed:,} decoded of "
            f"{qualifier}{event.total_units:,}"
        )
    lines = [first_line, f"Elapsed: {format_task_duration(event.elapsed_sec)}"]
    if event.estimated_remaining_sec is not None:
        lines.append(
            "Estimated remaining: "
            f"{format_task_duration(event.estimated_remaining_sec)}"
        )
    return "\n".join(lines)


def format_pipeline_progress(event: GuiTaskProgressEvent) -> str:
    """Render pipeline details without fabricating units, totals, or ETA."""

    lines: list[str] = []
    if event.message != event.phase:
        lines.append(event.message)
    if event.processed_video_time_sec is not None:
        lines.append(
            "Processed video time: "
            f"{format_task_duration(event.processed_video_time_sec)}"
        )
    lines.append(f"Elapsed: {format_task_duration(event.elapsed_sec)}")
    if event.estimated_remaining_sec is not None:
        lines.append(
            "Estimated remaining: "
            f"{format_task_duration(event.estimated_remaining_sec)}"
        )
    return "\n".join(lines)


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
        """Advance generation and cancel input-dependent tasks tied to prior inputs."""

        self._generation += 1
        cancelled: list[GuiTaskContext] = []
        for context in self._active.values():
            if context.task_kind in {
                GuiTaskKind.RAW_PROBE,
                GuiTaskKind.RAW_SEQUENTIAL_COUNT,
                GuiTaskKind.RAW_FRAME_READ,
                GuiTaskKind.CAGE_DETECTION,
                GuiTaskKind.PREPROCESS_RUN,
            }:
                context.request_cancellation()
                cancelled.append(context)
        return tuple(cancelled)

    def request_input_cancellation(self) -> tuple[GuiTaskContext, ...]:
        """Cancel current-generation input work without changing inputs."""

        cancelled: list[GuiTaskContext] = []
        for context in self._active.values():
            if context.task_kind in {
                GuiTaskKind.RAW_PROBE,
                GuiTaskKind.RAW_SEQUENTIAL_COUNT,
                GuiTaskKind.RAW_FRAME_READ,
                GuiTaskKind.CAGE_DETECTION,
                GuiTaskKind.PREPROCESS_RUN,
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

    def progress_is_current(
        self,
        event: GuiTaskProgressEvent,
        context: GuiTaskContext,
        *,
        project_dir: Path | None,
        raw_video_path: Path | None,
    ) -> bool:
        """Reject progress from a stale, cancelled, or different task generation."""

        return progress_matches_context(event, context) and self.is_current(
            context,
            project_dir=project_dir,
            raw_video_path=raw_video_path,
        )

    def finish(self, context: GuiTaskContext) -> None:
        """Forget a completed task without changing the input generation."""

        self._active.pop(context.task_id, None)


class _TaskWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(object)
    progressed = Signal(object)
    finished = Signal()

    def __init__(
        self,
        operation: Callable[..., Any],
        *,
        context: GuiTaskContext | None,
        progressive: bool,
    ) -> None:
        super().__init__()
        self._operation = operation
        self._progressive = progressive
        self._progress_tracker = (
            GuiTaskProgressTracker(context) if context is not None else None
        )

    def _report_progress(self, progress: OperationProgress) -> None:
        if self._progress_tracker is None:
            raise RuntimeError("Progress reporting requires a GUI task context.")
        validated = OperationProgress.model_validate(progress)
        self.progressed.emit(self._progress_tracker.build(validated))

    @Slot()
    def run(self) -> None:
        try:
            result = (
                self._operation(self._report_progress)
                if self._progressive
                else self._operation()
            )
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
    progressed = Signal(object)
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _TaskWorker | None = None
        self._on_success: Callable[[Any], None] | None = None
        self._on_error: Callable[[BaseException], None] | None = None
        self._on_progress: Callable[[GuiTaskProgressEvent], None] | None = None
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

        self._start(
            operation,
            progressive=False,
            task_name=task_name,
            context=context,
            on_success=on_success,
            on_error=on_error,
            on_finished=on_finished,
        )

    def start_progressive(
        self,
        operation: Callable[[Callable[[OperationProgress], None]], Any],
        *,
        task_name: str = "background task",
        context: GuiTaskContext,
        on_progress: Callable[[GuiTaskProgressEvent], None] | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        """Start an operation that emits typed progress from its worker thread."""

        self._start(
            operation,
            progressive=True,
            task_name=task_name,
            context=context,
            on_progress=on_progress,
            on_success=on_success,
            on_error=on_error,
            on_finished=on_finished,
        )

    def _start(
        self,
        operation: Callable[..., Any],
        *,
        progressive: bool,
        task_name: str,
        context: GuiTaskContext | None,
        on_progress: Callable[[GuiTaskProgressEvent], None] | None = None,
        on_success: Callable[[Any], None] | None,
        on_error: Callable[[BaseException], None] | None,
        on_finished: Callable[[], None] | None,
    ) -> None:
        """Configure the common QThread lifecycle for standard/progressive work."""

        if self.is_running:
            raise TaskAlreadyRunningError("A GUI background task is already running.")
        thread = QThread(self)
        worker = _TaskWorker(
            operation,
            context=context,
            progressive=progressive,
        )
        worker.moveToThread(thread)
        self._thread = thread
        self._worker = worker
        self._on_success = on_success
        self._on_error = on_error
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._context = context

        thread.started.connect(worker.run)
        worker.succeeded.connect(self._dispatch_success)
        worker.failed.connect(self._dispatch_failure)
        worker.progressed.connect(self._dispatch_progress)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.started.emit(task_name)
        thread.start()

    @Slot(object)
    def _dispatch_progress(self, event: object) -> None:
        if not isinstance(event, GuiTaskProgressEvent):
            return
        self.progressed.emit(event)
        if self._on_progress is not None:
            self._on_progress(event)

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
        self._on_progress = None
        self._on_finished = None
        self._context = None
        self.finished.emit()
        if on_finished is not None:
            on_finished()
