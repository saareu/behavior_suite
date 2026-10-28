"""Pytest fixtures for Subsystem 08 Sprint 1."""

from __future__ import annotations

import pytest

from project.file_digest import clear_digest_cache
from project.validation_memo import clear_validation_memo
from tests.behavioral_arbitration._fixtures import (
    build_synthetic_session,
    make_dyadic_wide,
    make_focal_wide,
    make_s6_features,
)


@pytest.fixture(autouse=True)
def _clear_process_caches() -> None:
    clear_digest_cache()
    clear_validation_memo()
    yield
    clear_digest_cache()
    clear_validation_memo()


@pytest.fixture
def synth_tables():
    return {
        "dyadic": make_dyadic_wide(),
        "focal": make_focal_wide(),
        "features": make_s6_features(),
    }


@pytest.fixture
def default_config():
    from behavioral_arbitration.config import load_behavioral_arbitration_config

    return load_behavioral_arbitration_config(None)


@pytest.fixture
def synthetic_session(tmp_path):
    return build_synthetic_session(tmp_path)
