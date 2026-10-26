"""Safe loading of trusted local joblib / XGBoost classifier bundles."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from behavioral_inference.contracts import (
    BUNDLE_FALLBACK,
    BUNDLE_OBSOLETE,
    BUNDLE_SUPPORTED,
    BUNDLE_UNSUPPORTED,
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    S7_RUNTIME_DEPENDENCIES,
    S7_RUNTIME_INSTALL_HINT,
    SERIALIZATION_JOBLIB_XGBOOST,
    ModelLoadError,
    OptionalDependencyError,
)
from behavioral_inference.registry import ModelBundleInfo


@dataclass(frozen=True, slots=True)
class RuntimeDependencyStatus:
    """Presence / version report for optional S7 inference libraries."""

    available: bool
    versions: Mapping[str, str | None]
    missing: tuple[str, ...]
    import_errors: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded estimator with preserved identity and digest metadata."""

    bundle: ModelBundleInfo
    estimator: Any
    classes: tuple[Any, ...]
    positive_class_index: int
    positive_class_label: Any
    n_features_in: int | None
    library_versions: Mapping[str, str | None]

    @property
    def detector_name(self) -> str:
        return self.bundle.detector_name

    @property
    def model_sha256(self) -> str | None:
        return self.bundle.model_sha256

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Run ``predict_proba`` and return the full ``(n, n_classes)`` array."""

        proba = self.estimator.predict_proba(matrix)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim != 2:
            raise ModelLoadError(
                f"{self.detector_name}: predict_proba must return a 2-D array; "
                f"got shape {arr.shape}"
            )
        if arr.shape[1] <= self.positive_class_index:
            raise ModelLoadError(
                f"{self.detector_name}: predict_proba width {arr.shape[1]} is too "
                f"small for positive_class_index={self.positive_class_index}"
            )
        return arr

    def positive_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Return the positive-class probability column after class-order validation."""

        return self.predict_proba(matrix)[:, self.positive_class_index]


def probe_runtime_dependencies() -> RuntimeDependencyStatus:
    """Report whether the optional ``s7`` runtime libraries can be imported."""

    versions: dict[str, str | None] = {}
    missing: list[str] = []
    errors: dict[str, str] = {}
    # Import names vs PyPI names.
    import_names = {
        "xgboost": "xgboost",
        "sklearn": "sklearn",
        "joblib": "joblib",
    }
    for label in S7_RUNTIME_DEPENDENCIES:
        module_name = import_names[label]
        if importlib.util.find_spec(module_name) is None:
            missing.append(label)
            versions[label] = None
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(label)
            versions[label] = None
            errors[label] = f"{type(exc).__name__}: {exc}"
            continue
        versions[label] = getattr(module, "__version__", None)
    return RuntimeDependencyStatus(
        available=not missing,
        versions=versions,
        missing=tuple(missing),
        import_errors=errors,
    )


def require_runtime_dependencies() -> Mapping[str, str | None]:
    """Return imported library versions or raise ``OptionalDependencyError``."""

    status = probe_runtime_dependencies()
    if not status.available:
        missing = ", ".join(status.missing)
        raise OptionalDependencyError(
            "Subsystem 07 model loading requires optional dependencies "
            f"[{missing}]. {S7_RUNTIME_INSTALL_HINT}"
        )
    return status.versions


def _resolve_positive_class_index(estimator: Any, detector_name: str) -> tuple[tuple[Any, ...], int, Any]:
    classes_raw = getattr(estimator, "classes_", None)
    if classes_raw is None:
        raise ModelLoadError(
            f"{detector_name}: estimator has no classes_ attribute; "
            "cannot confirm binary positive-class order"
        )
    classes = tuple(classes_raw)
    if len(classes) != 2:
        raise ModelLoadError(
            f"{detector_name}: expected binary classifier with 2 classes; "
            f"got classes_={classes!r}"
        )
    # Prefer explicit positive label == 1; fall back to the second class only when
    # labels are exactly {0, 1} already covered, or when 1 is present.
    if POSITIVE_CLASS_LABEL in classes:
        positive_index = classes.index(POSITIVE_CLASS_LABEL)
        positive_label: Any = POSITIVE_CLASS_LABEL
    else:
        raise ModelLoadError(
            f"{detector_name}: positive class label {POSITIVE_CLASS_LABEL!r} not "
            f"found in classes_={classes!r}; refusing to assume column index 1"
        )
    if NEGATIVE_CLASS_LABEL not in classes:
        raise ModelLoadError(
            f"{detector_name}: expected negative class label "
            f"{NEGATIVE_CLASS_LABEL!r} in classes_={classes!r}"
        )
    return classes, positive_index, positive_label


def _assert_trusted_local_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ModelLoadError(f"Model artifact is missing: {resolved}")
    # Reject obvious remote / URL-like schemes; only local filesystem paths.
    if "://" in str(path):
        raise ModelLoadError(
            f"Refusing to load non-local model path: {path}. "
            "Only trusted local filesystem paths are allowed."
        )
    return resolved


def load_model_bundle(
    bundle: ModelBundleInfo,
    *,
    allow_fallback: bool = True,
) -> LoadedModel:
    """Load one discovered bundle from a trusted local path.

    Does not download models. Does not substitute another bundle when loading
    fails. Obsolete / unsupported bundles raise ``ModelLoadError``.
    """

    if bundle.support == BUNDLE_OBSOLETE:
        raise ModelLoadError(
            f"Refusing to load obsolete detector '{bundle.detector_name}'. "
            "Use contact_intact and contact_occluded specialists instead."
        )
    if bundle.support == BUNDLE_UNSUPPORTED:
        raise ModelLoadError(
            f"Refusing to load unsupported detector '{bundle.detector_name}' "
            f"(family={bundle.family})."
        )
    if bundle.support == BUNDLE_FALLBACK and not allow_fallback:
        raise ModelLoadError(
            f"Refusing to load fallback detector '{bundle.detector_name}' "
            "(allow_fallback=False)."
        )
    if bundle.support not in (BUNDLE_SUPPORTED, BUNDLE_FALLBACK):
        raise ModelLoadError(
            f"Detector '{bundle.detector_name}' is not loadable "
            f"(support={bundle.support})."
        )
    if bundle.model_path is None:
        raise ModelLoadError(
            f"{bundle.detector_name}: model artifact path is missing from registry metadata"
        )
    if bundle.serialization_format != SERIALIZATION_JOBLIB_XGBOOST:
        raise ModelLoadError(
            f"{bundle.detector_name}: unsupported serialization "
            f"{bundle.serialization_format!r}"
        )

    versions = require_runtime_dependencies()
    model_path = _assert_trusted_local_path(bundle.model_path)

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - covered by require_runtime_dependencies
        raise OptionalDependencyError(
            "joblib is required to load model.joblib bundles"
        ) from exc

    try:
        estimator = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 — surface serialization failures clearly
        raise ModelLoadError(
            f"{bundle.detector_name}: failed to deserialize {model_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(estimator, "predict_proba"):
        raise ModelLoadError(
            f"{bundle.detector_name}: loaded object "
            f"{type(estimator).__name__} does not support predict_proba"
        )

    classes, positive_index, positive_label = _resolve_positive_class_index(
        estimator, bundle.detector_name
    )
    n_features_in = getattr(estimator, "n_features_in_", None)
    if n_features_in is not None:
        try:
            n_features_in = int(n_features_in)
        except (TypeError, ValueError):
            n_features_in = None
    if (
        n_features_in is not None
        and bundle.n_features
        and n_features_in != bundle.n_features
    ):
        raise ModelLoadError(
            f"{bundle.detector_name}: estimator n_features_in_={n_features_in} "
            f"does not match feature_spec length {bundle.n_features}"
        )

    return LoadedModel(
        bundle=bundle,
        estimator=estimator,
        classes=classes,
        positive_class_index=positive_index,
        positive_class_label=positive_label,
        n_features_in=n_features_in,
        library_versions=versions,
    )


def load_models(
    bundles: tuple[ModelBundleInfo, ...] | list[ModelBundleInfo],
    *,
    allow_fallback: bool = True,
) -> tuple[LoadedModel, ...]:
    """Load each bundle; first failure aborts without substituting another model."""

    loaded: list[LoadedModel] = []
    for bundle in bundles:
        loaded.append(load_model_bundle(bundle, allow_fallback=allow_fallback))
    return tuple(loaded)
