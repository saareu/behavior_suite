"""Minimal cooperative-cancellation contract shared across scientific runners.

Callers pass an optional ``cancellation_requested: Callable[[], bool] | None``.
When the callback is ``None``, behavior is unchanged. When present, a true return
value means the worker should stop at the next safe checkpoint by raising a
dedicated cancellation exception. False returns are ignored.

This module is intentionally Qt-free so S4–S8 runners can reuse it without
depending on ``GuiTaskRunner``.
"""

from __future__ import annotations

from collections.abc import Callable

CancellationCallback = Callable[[], bool]


class CooperativeCancellation(Exception):
    """Base exception for confirmed cooperative cancellation.

    Distinguishes intentional cancel from ordinary processing failure. Subsystems
    should subclass this (or raise it via ``raise_if_cancellation_requested``)
    so GUI layers can map the outcome to a canceled status only after the worker
    confirms termination.
    """


def is_cancellation_requested(
    cancellation_requested: CancellationCallback | None,
) -> bool:
    """Return True only when a callback is provided and currently reports True."""

    return cancellation_requested is not None and bool(cancellation_requested())


def raise_if_cancellation_requested(
    cancellation_requested: CancellationCallback | None,
    *,
    message: str = "Operation cancelled.",
    error: type[BaseException] | BaseException = CooperativeCancellation,
) -> None:
    """Raise when cooperative cancellation has been requested.

    No-op when ``cancellation_requested`` is ``None`` or returns a false value.
    Repeated requests remain harmless: each successful check raises the same
    dedicated cancellation error type.
    """

    if not is_cancellation_requested(cancellation_requested):
        return
    if isinstance(error, BaseException):
        raise error
    raise error(message)
