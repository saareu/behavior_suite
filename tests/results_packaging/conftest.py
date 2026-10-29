"""Pytest fixtures for Subsystem 09 Sprint 1."""

from __future__ import annotations

import pytest

from project.file_digest import clear_digest_cache
from project.validation_memo import clear_validation_memo


@pytest.fixture(autouse=True)
def _clear_process_caches() -> None:
    clear_digest_cache()
    clear_validation_memo()
    yield
    clear_digest_cache()
    clear_validation_memo()
