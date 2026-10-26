"""Model loader tests (mocked estimators + dependency probes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from behavioral_inference import (
    ModelBundleInfo,
    ModelLoadError,
    OptionalDependencyError,
    discover_model_registry,
    load_model_bundle,
    probe_runtime_dependencies,
)
from behavioral_inference import loader as loader_mod
from combined_features import ModelRegistrySettings


class _FakeEstimator:
    def __init__(self, classes: list[Any], n_features: int = 2) -> None:
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = n_features

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        n = len(matrix)
        # Deliberately put positive mass in column matching classes_ order.
        out = np.zeros((n, len(self.classes_)), dtype=float)
        pos = list(self.classes_).index(1)
        out[:, pos] = 0.75
        out[:, 1 - pos] = 0.25
        return out


class _NoProba:
    classes_ = np.asarray([0, 1])


def _bundle(
    tmp_path: Path,
    *,
    name: str = "nose_to_nose",
    support: str = "supported",
    n_features: int = 2,
) -> ModelBundleInfo:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"payload")
    return ModelBundleInfo(
        detector_name=name,
        family="dyadic",
        model_version="v1",
        bundle_dir=tmp_path,
        feature_spec_path=tmp_path / "feature_spec.json",
        model_path=model_path,
        feature_cols=tuple(f"f{i}" for i in range(n_features)),
        n_features=n_features,
        model_sha256="abc",
        expected_sha256=None,
        sha256_matches_expected=None,
        serialization_format="joblib_xgboost",
        support=support,
        model_family="xgboost",
        train_meta=None,
        notes=(),
    )


def test_probe_runtime_dependencies_reports_versions() -> None:
    status = probe_runtime_dependencies()
    assert isinstance(status.available, bool)
    assert "xgboost" in status.versions
    assert "sklearn" in status.versions
    assert "joblib" in status.versions


def test_optional_dependencies_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find = __import__("importlib.util", fromlist=["find_spec"]).find_spec

    def fake_find(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name in {"xgboost", "sklearn"}:
            return None
        return real_find(name, package)

    monkeypatch.setattr("importlib.util.find_spec", fake_find)
    status = probe_runtime_dependencies()
    assert status.available is False
    assert "xgboost" in status.missing
    assert "sklearn" in status.missing


def test_load_refuses_obsolete(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, name="contact", support="obsolete")
    with pytest.raises(ModelLoadError, match="obsolete"):
        load_model_bundle(bundle)


def test_load_refuses_unsupported(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, name="nose_body", support="unsupported")
    with pytest.raises(ModelLoadError, match="unsupported"):
        load_model_bundle(bundle)


def test_load_missing_file(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    missing = bundle.model_path
    assert missing is not None
    missing.unlink()
    with pytest.raises(ModelLoadError, match="missing"):
        load_model_bundle(bundle)


def test_load_broken_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not probe_runtime_dependencies().available:
        pytest.skip("s7 optional dependencies not installed")

    bundle = _bundle(tmp_path)

    def boom(_path: Path) -> Any:
        raise ValueError("corrupt pickle")

    monkeypatch.setattr(loader_mod, "require_runtime_dependencies", lambda: {})
    import joblib

    monkeypatch.setattr(joblib, "load", boom)
    with pytest.raises(ModelLoadError, match="deserialize"):
        load_model_bundle(bundle)


def test_binary_class_order_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not probe_runtime_dependencies().available:
        pytest.skip("s7 optional dependencies not installed")

    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        loader_mod,
        "require_runtime_dependencies",
        lambda: {"xgboost": "3.2.0", "sklearn": "1.6.1", "joblib": "1.5.3"},
    )
    import joblib

    # classes_ = [1, 0] → positive index must be 0, not assumed 1.
    estimator = _FakeEstimator(classes=[1, 0], n_features=2)
    monkeypatch.setattr(joblib, "load", lambda _p: estimator)
    loaded = load_model_bundle(bundle)
    assert loaded.positive_class_index == 0
    assert loaded.classes == (1, 0)
    proba = loaded.positive_proba(np.zeros((3, 2)))
    assert proba.shape == (3,)
    assert np.allclose(proba, 0.75)


def test_rejects_missing_positive_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not probe_runtime_dependencies().available:
        pytest.skip("s7 optional dependencies not installed")

    bundle = _bundle(tmp_path)
    monkeypatch.setattr(loader_mod, "require_runtime_dependencies", lambda: {})
    import joblib

    monkeypatch.setattr(joblib, "load", lambda _p: _FakeEstimator(classes=["neg", "pos"]))
    with pytest.raises(ModelLoadError, match="positive class label"):
        load_model_bundle(bundle)


def test_rejects_estimator_without_predict_proba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not probe_runtime_dependencies().available:
        pytest.skip("s7 optional dependencies not installed")

    bundle = _bundle(tmp_path)
    monkeypatch.setattr(loader_mod, "require_runtime_dependencies", lambda: {})
    import joblib

    monkeypatch.setattr(joblib, "load", lambda _p: _NoProba())
    with pytest.raises(ModelLoadError, match="predict_proba"):
        load_model_bundle(bundle)


def test_require_runtime_dependencies_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        loader_mod,
        "probe_runtime_dependencies",
        lambda: loader_mod.RuntimeDependencyStatus(
            available=False,
            versions={"xgboost": None, "sklearn": None, "joblib": "1.5.3"},
            missing=("xgboost", "sklearn"),
            import_errors={},
        ),
    )
    bundle = _bundle(tmp_path)
    with pytest.raises(OptionalDependencyError, match=r"\.\[s2,s7\]|install_windows_gui"):
        load_model_bundle(bundle)


def test_broken_joblib_in_mini_registry(mini_registry: Path) -> None:
    if not probe_runtime_dependencies().available:
        pytest.skip("s7 optional dependencies not installed")
    snapshot = discover_model_registry(ModelRegistrySettings(root=mini_registry))
    intact = snapshot.by_name()["contact_intact"]
    with pytest.raises(ModelLoadError):
        load_model_bundle(intact)
