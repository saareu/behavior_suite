"""Shared cooperative-cancellation contract tests."""

from __future__ import annotations

import pytest

from runtime.cancellation import (
    CooperativeCancellation,
    is_cancellation_requested,
    raise_if_cancellation_requested,
)


def test_no_callback_is_backward_compatible() -> None:
    assert is_cancellation_requested(None) is False
    raise_if_cancellation_requested(None)  # no-op


def test_false_callback_does_not_raise() -> None:
    assert is_cancellation_requested(lambda: False) is False
    raise_if_cancellation_requested(lambda: False)


def test_true_callback_raises_dedicated_exception() -> None:
    with pytest.raises(CooperativeCancellation, match="stop now"):
        raise_if_cancellation_requested(lambda: True, message="stop now")


def test_custom_error_type_is_raised() -> None:
    class DomainCancelled(CooperativeCancellation):
        pass

    with pytest.raises(DomainCancelled, match="domain stop"):
        raise_if_cancellation_requested(
            lambda: True,
            message="domain stop",
            error=DomainCancelled,
        )


def test_prebuilt_exception_instance_is_raised() -> None:
    exc = CooperativeCancellation("already built")
    with pytest.raises(CooperativeCancellation, match="already built"):
        raise_if_cancellation_requested(lambda: True, error=exc)


def test_repeated_true_checks_remain_harmless() -> None:
    for _ in range(3):
        with pytest.raises(CooperativeCancellation):
            raise_if_cancellation_requested(lambda: True, message="again")
