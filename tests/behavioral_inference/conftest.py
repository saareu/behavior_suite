"""Shared fixtures for Subsystem 07 Sprint 1 tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavioral_inference.loader import probe_runtime_dependencies

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_MODEL_ROOT = REPO_ROOT / "legacy" / "models"
S6_FEATURES = (
    REPO_ROOT
    / "test_cases"
    / "E_full_session_integration"
    / "combined_features"
    / "combined_features__20261025T173244"
    / "dyadic_features_full.parquet"
)


def _write_spec(path: Path, cols: list[str], detector_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "detector_name": detector_name,
                "feature_cols": cols,
                "selected_feature_cols": cols,
            }
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def mini_registry(tmp_path: Path) -> Path:
    """Synthetic registry with supported, obsolete, and incomplete bundles."""

    root = tmp_path / "models"
    dyadic = root / "dyadic" / "v1"
    focal = root / "focal" / "v1"

    _write_spec(
        dyadic / "contact_intact" / "feature_spec.json",
        ["feat_a", "feat_b"],
        "contact_intact",
    )
    (dyadic / "contact_intact" / "model.joblib").write_bytes(b"not-a-real-model")

    _write_spec(
        dyadic / "nose_to_nose" / "feature_spec.json",
        ["feat_a"],
        "nose_to_nose",
    )
    (dyadic / "nose_to_nose" / "model.joblib").write_bytes(b"not-a-real-model")

    _write_spec(
        dyadic / "contact" / "feature_spec.json",
        ["nose_any_refined"],
        "contact",
    )
    (dyadic / "contact" / "model.joblib").write_bytes(b"obsolete")

    _write_spec(
        focal / "approach" / "feature_spec.json",
        ["feat_a", "feat_c"],
        "approach",
    )
    (focal / "approach" / "model.joblib").write_bytes(b"not-a-real-model")

    # Unsupported JSON-expert style focal bundle (no joblib).
    _write_spec(
        focal / "nose_body" / "feature_spec.json",
        ["feat_a"],
        "nose_body",
    )
    (focal / "nose_body" / "open_model.json").write_text("{}", encoding="utf-8")

    # Incomplete active specialist (spec only).
    _write_spec(
        dyadic / "side_by_side" / "feature_spec.json",
        ["feat_a"],
        "side_by_side",
    )
    return root


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_a": [0.1, 0.2, np.nan],
            "feat_b": [1.0, 2.0, 3.0],
            "feat_c": [0.0, 1.0, 0.0],
            "other": [9, 9, 9],
        }
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "s7_model_smoke: loads trusted lab model.joblib bundles (requires .[s7] + local registry)",
    )


def lab_registry_available() -> bool:
    return (LAB_MODEL_ROOT / "dyadic" / "v1").is_dir() and (
        LAB_MODEL_ROOT / "focal" / "v1"
    ).is_dir()


def s7_deps_available() -> bool:
    return probe_runtime_dependencies().available
