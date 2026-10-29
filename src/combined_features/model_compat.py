"""Model registry discovery and feature-spec compatibility (no inference)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.config import ModelRegistrySettings
from combined_features.contracts import (
    MODEL_COMPAT_SCHEMA_VERSION,
    OBSOLETE_DYADIC_DETECTORS,
    CombinedFeaturesError,
)

_MODEL_FILE_CANDIDATES = (
    "model.joblib",
    "model.pkl",
    "booster.json",
    "model.json",
)


@dataclass(frozen=True, slots=True)
class DiscoveredDetector:
    """One detector bundle discovered under the configured registry root."""

    detector_name: str
    family: str
    bundle_dir: Path
    feature_spec_path: Path
    obsolete: bool = False


def discover_model_registry(
    settings: ModelRegistrySettings,
) -> tuple[str, tuple[DiscoveredDetector, ...], list[str]]:
    """Discover dyadic/focal detectors from an explicitly configured root.

    Returns ``(registry_status, detectors, notes)``. Never reads ``models/``
    unless that path is explicitly configured as ``settings.root``.
    """

    notes: list[str] = []
    validated = settings.validated()
    if validated.root is None:
        return "unavailable", (), ["model_registry.root is not configured"]
    root = validated.root
    if not root.is_dir():
        return (
            "unavailable",
            (),
            [f"model_registry.root is not a directory: {root}"],
        )

    detectors: list[DiscoveredDetector] = []
    dyadic_root = root / Path(validated.dyadic_subdir)
    focal_root = root / Path(validated.focal_subdir)

    if dyadic_root.is_dir():
        for child in sorted(dyadic_root.iterdir()):
            if not child.is_dir():
                continue
            spec = child / "feature_spec.json"
            if not spec.is_file():
                notes.append(f"dyadic bundle missing feature_spec.json: {child.name}")
                continue
            name = child.name
            obsolete = name in OBSOLETE_DYADIC_DETECTORS
            if obsolete and not validated.include_obsolete:
                # Still report obsolete contact explicitly as unsupported.
                if name == "contact":
                    detectors.append(
                        DiscoveredDetector(
                            detector_name=name,
                            family="dyadic",
                            bundle_dir=child.resolve(),
                            feature_spec_path=spec.resolve(),
                            obsolete=True,
                        )
                    )
                continue
            detectors.append(
                DiscoveredDetector(
                    detector_name=name,
                    family="dyadic",
                    bundle_dir=child.resolve(),
                    feature_spec_path=spec.resolve(),
                    obsolete=obsolete,
                )
            )
    else:
        notes.append(f"dyadic registry subdir missing: {dyadic_root}")

    if validated.include_focal:
        if focal_root.is_dir():
            for child in sorted(focal_root.iterdir()):
                if not child.is_dir():
                    continue
                spec = child / "feature_spec.json"
                if not spec.is_file():
                    notes.append(f"focal bundle missing feature_spec.json: {child.name}")
                    continue
                detectors.append(
                    DiscoveredDetector(
                        detector_name=child.name,
                        family="focal",
                        bundle_dir=child.resolve(),
                        feature_spec_path=spec.resolve(),
                        obsolete=False,
                    )
                )
        else:
            notes.append(f"focal registry subdir missing: {focal_root}")

    # Always surface obsolete contact even when the bundle directory is absent.
    if not any(d.detector_name == "contact" for d in detectors):
        detectors.append(
            DiscoveredDetector(
                detector_name="contact",
                family="dyadic",
                bundle_dir=dyadic_root / "contact",
                feature_spec_path=dyadic_root / "contact" / "feature_spec.json",
                obsolete=True,
            )
        )
        notes.append(
            "obsolete dyadic detector 'contact' is unsupported under S6 "
            "(nose_any_refined / unsuffixed nose_any_imputed)"
        )

    status = "ok" if detectors else "unavailable"
    if notes and status == "ok":
        status = "partial"
    return status, tuple(detectors), notes


def resolve_required_features(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve ordered required feature names from a feature_spec.json payload."""

    selected = spec.get("selected_feature_cols")
    if isinstance(selected, list) and selected:
        return tuple(str(x) for x in selected)

    feature_cols = spec.get("feature_cols")
    if isinstance(feature_cols, list) and feature_cols:
        return tuple(str(x) for x in feature_cols)

    names: list[str] = []
    for key in ("open_features", "occ_features", "blob_features"):
        values = spec.get(key)
        if isinstance(values, list):
            for item in values:
                text = str(item)
                if text not in names:
                    names.append(text)
    return tuple(names)


def load_feature_spec(path: Path | str) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise CombinedFeaturesError(f"feature_spec.json is missing: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CombinedFeaturesError(
            f"Could not load feature_spec.json: {file_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CombinedFeaturesError(f"feature_spec.json root must be an object: {file_path}")
    return payload


def _model_file_present(bundle_dir: Path, spec: Mapping[str, Any]) -> bool | None:
    """Return True/False when assessable; None when bundle dir is absent."""

    if not bundle_dir.is_dir():
        return None
    named: list[str] = []
    for key in ("model_file", "open_model_file", "occ_model_file"):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            named.append(value.strip())
    for name in named:
        if (bundle_dir / name).is_file():
            return True
    for name in _MODEL_FILE_CANDIDATES:
        if (bundle_dir / name).is_file():
            return True
    # Two-stage experts may only list json boosters; any named file missing → False
    if named:
        return all((bundle_dir / name).is_file() for name in named)
    # Spec-only bundle without binary is incomplete for inference but still checkable.
    return False


def evaluate_detector_compatibility(
    *,
    detector: DiscoveredDetector,
    features: pd.DataFrame,
    nan_warn_fraction: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one detector against an in-memory feature table (no prediction)."""

    notes: list[str] = []
    if detector.obsolete or detector.detector_name in OBSOLETE_DYADIC_DETECTORS:
        return {
            "detector_name": detector.detector_name,
            "family": detector.family,
            "bundle_dir": str(detector.bundle_dir),
            "status": "unsupported",
            "n_required": 0,
            "required_features": [],
            "available_required_features": [],
            "missing_columns": [],
            "dtype_warnings": [],
            "nan_fraction_warnings": [],
            "model_file_present": False,
            "notes": [
                "Obsolete standalone 'contact' detector is unsupported under S6. "
                "Active replacements: contact_intact + contact_occluded. "
                "Requires obsolete columns such as nose_any_refined and "
                "unsuffixed nose_any_imputed."
            ],
        }

    if not detector.feature_spec_path.is_file():
        return {
            "detector_name": detector.detector_name,
            "family": detector.family,
            "bundle_dir": str(detector.bundle_dir),
            "status": "bundle_incomplete",
            "n_required": 0,
            "required_features": [],
            "available_required_features": [],
            "missing_columns": [],
            "dtype_warnings": [],
            "nan_fraction_warnings": [],
            "model_file_present": False,
            "notes": ["feature_spec.json is missing"],
        }

    try:
        spec = load_feature_spec(detector.feature_spec_path)
    except CombinedFeaturesError as exc:
        return {
            "detector_name": detector.detector_name,
            "family": detector.family,
            "bundle_dir": str(detector.bundle_dir),
            "status": "invalid_spec",
            "n_required": 0,
            "required_features": [],
            "available_required_features": [],
            "missing_columns": [],
            "dtype_warnings": [],
            "nan_fraction_warnings": [],
            "model_file_present": False,
            "notes": [str(exc)],
        }

    required = resolve_required_features(spec)
    if not required:
        return {
            "detector_name": detector.detector_name,
            "family": detector.family,
            "bundle_dir": str(detector.bundle_dir),
            "status": "invalid_spec",
            "n_required": 0,
            "required_features": [],
            "available_required_features": [],
            "missing_columns": [],
            "dtype_warnings": [],
            "nan_fraction_warnings": [],
            "model_file_present": _model_file_present(detector.bundle_dir, spec),
            "notes": ["feature_spec resolves to zero required feature names"],
        }

    available = [c for c in required if c in features.columns]
    missing = [c for c in required if c not in features.columns]
    dtype_warnings: list[str] = []
    nan_warnings: list[str] = []
    for col in available:
        series = features[col]
        if not np.issubdtype(series.dtype, np.number) and series.dtype != object:
            # Non-numeric non-object is unusual for model inputs.
            dtype_warnings.append(f"{col}: unexpected dtype {series.dtype}")
        elif not np.issubdtype(series.dtype, np.number):
            dtype_warnings.append(f"{col}: non-numeric dtype {series.dtype}")
        if np.issubdtype(series.dtype, np.number) and nan_warn_fraction < 1.0:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            frac = float(1.0 - np.isfinite(values).mean()) if values.size else 1.0
            if frac > nan_warn_fraction:
                nan_warnings.append(f"{col}: nan/inf fraction={frac:.3f}")

    model_present = _model_file_present(detector.bundle_dir, spec)
    if model_present is False:
        notes.append("model binary / booster file not found (presence-only check)")
    elif model_present is None:
        notes.append("bundle directory missing; cannot assess model file presence")

    if missing:
        status = "incompatible"
    elif model_present is False:
        status = "bundle_incomplete"
    elif dtype_warnings:
        status = "compatible"
        notes.append("dtype warnings present; values may still be usable")
    else:
        status = "compatible"

    # Legitimate NaNs are never structural incompatibility.
    notes.append(
        "NaN feature values are treated as scientific missingness, not structural "
        "incompatibility."
    )

    return {
        "detector_name": detector.detector_name,
        "family": detector.family,
        "bundle_dir": str(detector.bundle_dir),
        "status": status,
        "n_required": len(required),
        "required_features": list(required),
        "available_required_features": available,
        "missing_columns": missing,
        "dtype_warnings": dtype_warnings,
        "nan_fraction_warnings": nan_warnings,
        "model_file_present": model_present,
        "notes": notes,
    }


def build_model_compat_report(
    *,
    features: pd.DataFrame,
    settings: ModelRegistrySettings,
    feature_artifact: str = "dyadic_features_full.parquet",
) -> dict[str, Any]:
    """Build the full ``model_compat.json`` payload."""

    registry_status, detectors, notes = discover_model_registry(settings)
    detector_reports = [
        evaluate_detector_compatibility(detector=det, features=features)
        for det in detectors
    ]
    return {
        "schema_version": MODEL_COMPAT_SCHEMA_VERSION,
        "registry_root": (
            str(settings.validated().root) if settings.validated().root is not None else None
        ),
        "registry_status": registry_status,
        "feature_artifact": feature_artifact,
        "n_feature_columns": int(len(features.columns)),
        "n_rows": int(len(features)),
        "detectors": detector_reports,
        "notes": notes,
    }


def write_model_compat_json(path: Path | str, report: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(report), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return out
