"""Controlled real-model loading + predict_proba smoke (Sprint 1 acceptance)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    ACTIVE_DYADIC_BUNDLES,
    ACTIVE_FOCAL_BUNDLES,
    FOCAL_FALLBACK_BUNDLES,
    ModelLoadError,
    discover_model_registry,
    load_model_bundle,
    require_feature_matrix,
    validate_feature_matrix,
)
from behavioral_inference.loader import probe_runtime_dependencies
from combined_features import ModelRegistrySettings

from .conftest import LAB_MODEL_ROOT, S6_FEATURES, lab_registry_available, s7_deps_available

pytestmark = pytest.mark.s7_model_smoke


def _skip_if_unavailable() -> None:
    if not lab_registry_available():
        pytest.skip(f"Lab model registry not available at {LAB_MODEL_ROOT}")
    if not s7_deps_available():
        status = probe_runtime_dependencies()
        pytest.skip(
            "S7 optional dependencies unavailable "
            f"(missing={status.missing}). Install with: pip install -e \".[s7]\""
        )


def test_discover_active_lab_models() -> None:
    _skip_if_unavailable()
    snapshot = discover_model_registry(ModelRegistrySettings(root=LAB_MODEL_ROOT))
    assert snapshot.status in ("ok", "partial")
    names = {b.detector_name for b in snapshot.bundles}
    for name in ACTIVE_DYADIC_BUNDLES:
        assert name in names
    for name in ACTIVE_FOCAL_BUNDLES:
        assert name in names
    contact = snapshot.by_name()["contact"]
    assert contact.support == "obsolete"
    assert contact.is_loadable is False


def test_load_active_models_and_predict_proba() -> None:
    _skip_if_unavailable()
    snapshot = discover_model_registry(ModelRegistrySettings(root=LAB_MODEL_ROOT))
    loadable = [
        b
        for b in snapshot.bundles
        if b.detector_name in set(ACTIVE_DYADIC_BUNDLES + ACTIVE_FOCAL_BUNDLES)
        and b.is_loadable
    ]
    assert len(loadable) == len(ACTIVE_DYADIC_BUNDLES) + len(ACTIVE_FOCAL_BUNDLES)

    # Prefer a tiny S6 sample when present; otherwise synthetic finite rows.
    sample: pd.DataFrame | None = None
    if S6_FEATURES.is_file():
        sample = pd.read_parquet(S6_FEATURES).head(4)

    outcomes: list[dict[str, object]] = []
    for bundle in loadable:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_model_bundle(bundle)
        assert loaded.model_sha256 == bundle.model_sha256
        assert loaded.positive_class_label == 1
        assert loaded.classes == (0, 1)
        assert loaded.positive_class_index == 1

        feature_errors: tuple[str, ...] = ()
        if sample is not None:
            validation = validate_feature_matrix(sample, bundle.feature_cols)
            if validation.ok and validation.matrix is not None:
                matrix = validation.matrix
            else:
                # Fall back to NaN rows matching feature width when a column gap
                # appears (should not happen for active lab bundles vs E-session).
                matrix = np.full((2, bundle.n_features), np.nan, dtype=float)
                feature_errors = validation.errors
        else:
            matrix = np.full((2, bundle.n_features), np.nan, dtype=float)

        proba = loaded.predict_proba(matrix)
        pos = loaded.positive_proba(matrix)
        assert proba.shape == (len(matrix), 2)
        assert pos.shape == (len(matrix),)
        assert np.isfinite(proba).all()
        assert np.isfinite(pos).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        outcomes.append(
            {
                "detector": bundle.detector_name,
                "family": bundle.family,
                "n_features": bundle.n_features,
                "sha256": bundle.model_sha256,
                "warnings": [str(w.message) for w in caught],
                "feature_errors": list(feature_errors),
                "mean_pos": float(np.mean(pos)),
            }
        )

    assert len(outcomes) >= len(loadable)


def test_fallback_nose_to_body_loads() -> None:
    _skip_if_unavailable()
    snapshot = discover_model_registry(ModelRegistrySettings(root=LAB_MODEL_ROOT))
    fallback = snapshot.by_name().get("nose_to_body")
    if fallback is None:
        # Optional single-stage fallback; lab registries that keep only the
        # preferred two-stage specialists (operator cleanup) skip this probe.
        pytest.skip(
            "Optional focal fallback bundle nose_to_body not present under "
            f"{LAB_MODEL_ROOT}; two-stage specialists remain the preferred path"
        )
    assert fallback.support == "fallback"
    loaded = load_model_bundle(fallback, allow_fallback=True)
    matrix = require_feature_matrix(
        pd.DataFrame(
            {c: [0.0, 1.0] for c in fallback.feature_cols},
            columns=list(fallback.feature_cols),
        ),
        fallback.feature_cols,
    )
    pos = loaded.positive_proba(matrix)
    assert np.isfinite(pos).all()
    assert fallback.detector_name in FOCAL_FALLBACK_BUNDLES


def test_obsolete_contact_refuses_load() -> None:
    _skip_if_unavailable()
    snapshot = discover_model_registry(ModelRegistrySettings(root=LAB_MODEL_ROOT))
    contact = snapshot.by_name()["contact"]
    with pytest.raises(ModelLoadError, match="obsolete"):
        load_model_bundle(contact)
