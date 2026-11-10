"""Small cross-subsystem runtime helpers (no Qt dependency)."""

from runtime.cancellation import (
    CancellationCallback,
    CooperativeCancellation,
    is_cancellation_requested,
    raise_if_cancellation_requested,
)

__all__ = [
    "CancellationCallback",
    "CooperativeCancellation",
    "is_cancellation_requested",
    "raise_if_cancellation_requested",
]
