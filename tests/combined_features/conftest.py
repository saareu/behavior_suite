"""Pytest fixtures for Subsystem 06 Sprint 1."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.combined_features.helpers import write_joint_session


@pytest.fixture
def joint_session(tmp_path: Path) -> dict[str, Path]:
    return write_joint_session(tmp_path)
