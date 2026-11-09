"""Shared helpers for S2/S4/S5/S6 asynchronous open/discovery responsiveness tests.

The helpers deliberately avoid sleeping inside production code paths: slow
discovery is simulated with a :class:`Gate` that a worker blocks on until the
test releases it, so assertions about "GUI stays responsive while the worker is
busy" are deterministic.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from PySide6.QtCore import QCoreApplication, QEvent, QThread, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APPLICATION = QApplication.instance() or QApplication([])

# A synchronous (regression) implementation would block here instead of returning,
# so keep the worker-side wait short enough that such a regression fails fast.
GATE_TIMEOUT_SEC = 8.0
# Generous upper bound for a page entry point that must only schedule work.
SCHEDULE_BUDGET_SEC = 1.0


def wait_until(predicate: Callable[[], bool], timeout_sec: float = 30.0) -> None:
    """Pump Qt events until ``predicate`` holds (same contract as the S8 tests)."""

    deadline = time.monotonic() + timeout_sec
    while not predicate():
        APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    APPLICATION.processEvents()


def count_event_loop_ticks(duration_sec: float = 0.15, interval_ms: int = 5) -> int:
    """Return how many QTimer ticks were delivered while pumping events."""

    ticks = [0]
    timer = QTimer()
    timer.setInterval(interval_ms)
    timer.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
    timer.start()
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        APPLICATION.processEvents()
        time.sleep(0.002)
    timer.stop()
    return ticks[0]


def on_gui_thread() -> bool:
    return QThread.currentThread() == APPLICATION.thread()


def silence_modal_dialogs(monkeypatch: Any) -> None:
    """Offscreen runs cannot dismiss modal ``QMessageBox`` dialogs."""

    for name in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))


def destroy_page(page: Any) -> None:
    """Close and delete a page once its workers are done, flushing deferred deletes."""

    has_running = getattr(page, "has_running_task", None)
    if callable(has_running):
        wait_until(lambda: not bool(has_running()), timeout_sec=120.0)
    page.close()
    page.deleteLater()
    APPLICATION.processEvents()
    QCoreApplication.sendPostedEvents(None, int(QEvent.Type.DeferredDelete))
    APPLICATION.processEvents()


class Gate:
    """Record worker-side calls and optionally block them until released."""

    def __init__(self, *, blocking: bool = True) -> None:
        self._release = threading.Event()
        if not blocking:
            self._release.set()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.payloads: list[Any] = []
        self.thread_ids: list[int] = []
        self.gui_thread_flags: list[bool] = []
        self.entered = threading.Event()

    @property
    def calls(self) -> int:
        return len(self.payloads)

    def release(self) -> None:
        self._release.set()

    def hold(self) -> None:
        self._release.clear()

    @contextmanager
    def section(self, payload: Any = None) -> Iterator[None]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.payloads.append(payload)
            self.thread_ids.append(threading.get_ident())
            self.gui_thread_flags.append(on_gui_thread())
        self.entered.set()
        try:
            self._release.wait(timeout=GATE_TIMEOUT_SEC)
            yield
        finally:
            with self._lock:
                self._active -= 1

    def never_on_gui_thread(self) -> bool:
        return bool(self.gui_thread_flags) and not any(self.gui_thread_flags)
