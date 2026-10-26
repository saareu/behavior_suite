"""Explicit model-registry discovery for Subsystem 07 (no automatic legacy/)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from behavioral_inference.contracts import (
    ACTIVE_DYADIC_BUNDLES,
    ACTIVE_FOCAL_BUNDLES,
    BUNDLE_FALLBACK,
    BUNDLE_INCOMPLETE,
    BUNDLE_OBSOLETE,
    BUNDLE_SUPPORTED,
    BUNDLE_UNSUPPORTED,
    EXPECTED_MODEL_SHA256,
    FEATURE_SPEC_FILENAME,
    FOCAL_FALLBACK_BUNDLES,
    MODEL_FILENAME,
    OBSOLETE_BUNDLES,
    REGISTRY_OK,
    REGISTRY_PARTIAL,
    REGISTRY_UNAVAILABLE,
    SERIALIZATION_JOBLIB_XGBOOST,
    UNSUPPORTED_FOCAL_BUNDLES,
    ModelRegistryError,
)
from combined_features.config import ModelRegistrySettings
from combined_features.model_compat import load_feature_spec, resolve_required_features
from project.file_digest import sha256_file


@dataclass(frozen=True, slots=True)
class ModelBundleInfo:
    """Metadata for one on-disk detector / specialist bundle (not loaded)."""

    detector_name: str
    family: str
    model_version: str
    bundle_dir: Path
    feature_spec_path: Path | None
    model_path: Path | None
    feature_cols: tuple[str, ...]
    n_features: int
    model_sha256: str | None
    expected_sha256: str | None
    sha256_matches_expected: bool | None
    serialization_format: str | None
    support: str
    model_family: str | None
    train_meta: Mapping[str, Any] | None
    notes: tuple[str, ...] = ()

    @property
    def is_loadable(self) -> bool:
        """True when the bundle is supported/fallback and has a joblib artifact."""

        return (
            self.support in (BUNDLE_SUPPORTED, BUNDLE_FALLBACK)
            and self.model_path is not None
            and self.model_path.is_file()
            and bool(self.feature_cols)
        )


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    """Result of discovering bundles under a configured registry root."""

    status: str
    settings: ModelRegistrySettings
    bundles: tuple[ModelBundleInfo, ...]
    notes: tuple[str, ...]

    def by_name(self, family: str | None = None) -> dict[str, ModelBundleInfo]:
        items = self.bundles
        if family is not None:
            items = tuple(b for b in items if b.family == family)
        return {b.detector_name: b for b in items}

    def loadable_bundles(self) -> tuple[ModelBundleInfo, ...]:
        return tuple(b for b in self.bundles if b.is_loadable)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _classify_bundle(*, family: str, name: str, has_model: bool) -> str:
    if family == "dyadic" and name in OBSOLETE_BUNDLES:
        return BUNDLE_OBSOLETE
    if family == "focal" and name in UNSUPPORTED_FOCAL_BUNDLES:
        return BUNDLE_UNSUPPORTED
    if family == "focal" and name in FOCAL_FALLBACK_BUNDLES:
        return BUNDLE_FALLBACK if has_model else BUNDLE_INCOMPLETE
    if family == "dyadic" and name in ACTIVE_DYADIC_BUNDLES:
        return BUNDLE_SUPPORTED if has_model else BUNDLE_INCOMPLETE
    if family == "focal" and name in ACTIVE_FOCAL_BUNDLES:
        return BUNDLE_SUPPORTED if has_model else BUNDLE_INCOMPLETE
    if has_model:
        # Unknown joblib bundle: discoverable but not in the active allow-list.
        return BUNDLE_UNSUPPORTED
    return BUNDLE_INCOMPLETE


def _inspect_bundle(
    *,
    family: str,
    model_version: str,
    bundle_dir: Path,
) -> ModelBundleInfo:
    name = bundle_dir.name
    notes: list[str] = []
    spec_path = bundle_dir / FEATURE_SPEC_FILENAME
    model_path = bundle_dir / MODEL_FILENAME
    has_model = model_path.is_file()
    support = _classify_bundle(family=family, name=name, has_model=has_model)

    feature_cols: tuple[str, ...] = ()
    resolved_spec: Path | None = None
    if spec_path.is_file():
        resolved_spec = spec_path.resolve()
        try:
            spec = load_feature_spec(resolved_spec)
            feature_cols = resolve_required_features(spec)
            if not feature_cols:
                notes.append("feature_spec resolves to zero feature columns")
        except Exception as exc:  # noqa: BLE001 — surface as note, keep discovery soft
            notes.append(f"invalid feature_spec.json: {exc}")
    else:
        notes.append(f"missing {FEATURE_SPEC_FILENAME}")

    model_sha: str | None = None
    serialization: str | None = None
    resolved_model: Path | None = None
    if has_model:
        resolved_model = model_path.resolve()
        model_sha = _sha256_file(resolved_model)
        serialization = SERIALIZATION_JOBLIB_XGBOOST
    else:
        notes.append(f"missing {MODEL_FILENAME}")

    expected_key = f"{family}/{name}"
    expected = EXPECTED_MODEL_SHA256.get(expected_key)
    sha_match: bool | None = None
    if model_sha is not None and expected is not None:
        sha_match = model_sha == expected
        if not sha_match:
            notes.append("model.joblib SHA-256 does not match S7-B lab inventory")

    train_meta = _read_json_optional(bundle_dir / "train_meta.json")
    model_family: str | None = None
    if isinstance(train_meta, Mapping):
        raw_family = train_meta.get("model_family")
        if isinstance(raw_family, str) and raw_family.strip():
            model_family = raw_family.strip()
    if model_family is None and serialization == SERIALIZATION_JOBLIB_XGBOOST:
        model_family = "xgboost"

    if support == BUNDLE_OBSOLETE:
        notes.append(
            "Obsolete standalone 'contact' detector is unsupported. "
            "Use contact_intact + contact_occluded specialists."
        )
    elif support == BUNDLE_UNSUPPORTED and name in UNSUPPORTED_FOCAL_BUNDLES:
        notes.append(
            "JSON two-stage expert architecture without model.joblib; "
            "unsupported for S7 MVP."
        )
    elif support == BUNDLE_FALLBACK:
        notes.append(
            "Single-stage nose_to_body fallback; preferred path uses "
            "nose_body_intact + nose_body_occ."
        )

    return ModelBundleInfo(
        detector_name=name,
        family=family,
        model_version=model_version,
        bundle_dir=bundle_dir.resolve(),
        feature_spec_path=resolved_spec,
        model_path=resolved_model,
        feature_cols=feature_cols,
        n_features=len(feature_cols),
        model_sha256=model_sha,
        expected_sha256=expected,
        sha256_matches_expected=sha_match,
        serialization_format=serialization,
        support=support,
        model_family=model_family,
        train_meta=train_meta,
        notes=tuple(notes),
    )


def _version_from_subdir(subdir: str) -> str:
    parts = Path(subdir).parts
    return parts[-1] if parts else "v1"


def discover_model_registry(
    settings: ModelRegistrySettings | None = None,
    *,
    root: Path | str | None = None,
) -> RegistrySnapshot:
    """Discover dyadic/focal bundles under an explicitly configured root.

    Never implies ``legacy/``. A missing / unset root yields
    ``status=unavailable`` without raising, so S1–S6 remain usable.
    """

    if settings is None:
        settings = ModelRegistrySettings(root=Path(root) if root is not None else None)
    elif root is not None:
        settings = replace(settings, root=Path(root))

    try:
        validated = settings.validated()
    except Exception as exc:  # combined_features raises CombinedFeaturesError
        raise ModelRegistryError(str(exc)) from exc

    notes: list[str] = []
    if validated.root is None:
        return RegistrySnapshot(
            status=REGISTRY_UNAVAILABLE,
            settings=validated,
            bundles=(),
            notes=("model_registry.root is not configured",),
        )
    if not validated.root.is_dir():
        return RegistrySnapshot(
            status=REGISTRY_UNAVAILABLE,
            settings=validated,
            bundles=(),
            notes=(f"model_registry.root is not a directory: {validated.root}",),
        )

    bundles: list[ModelBundleInfo] = []
    dyadic_root = validated.root / Path(validated.dyadic_subdir)
    focal_root = validated.root / Path(validated.focal_subdir)
    dyadic_version = _version_from_subdir(validated.dyadic_subdir)
    focal_version = _version_from_subdir(validated.focal_subdir)

    if dyadic_root.is_dir():
        for child in sorted(dyadic_root.iterdir()):
            if not child.is_dir():
                continue
            info = _inspect_bundle(
                family="dyadic",
                model_version=dyadic_version,
                bundle_dir=child,
            )
            # Always surface obsolete contact; skip other obsolete when excluded.
            if (
                info.support == BUNDLE_OBSOLETE
                and not validated.include_obsolete
                and info.detector_name != "contact"
            ):
                continue
            bundles.append(info)
    else:
        notes.append(f"dyadic registry subdir missing: {dyadic_root}")

    if validated.include_focal:
        if focal_root.is_dir():
            for child in sorted(focal_root.iterdir()):
                if not child.is_dir():
                    continue
                bundles.append(
                    _inspect_bundle(
                        family="focal",
                        model_version=focal_version,
                        bundle_dir=child,
                    )
                )
        else:
            notes.append(f"focal registry subdir missing: {focal_root}")

    # Guarantee obsolete contact appears even when the directory is absent.
    if not any(b.detector_name == "contact" and b.family == "dyadic" for b in bundles):
        contact_dir = dyadic_root / "contact"
        bundles.append(
            ModelBundleInfo(
                detector_name="contact",
                family="dyadic",
                model_version=dyadic_version,
                bundle_dir=contact_dir.resolve(),
                feature_spec_path=None,
                model_path=None,
                feature_cols=(),
                n_features=0,
                model_sha256=None,
                expected_sha256=EXPECTED_MODEL_SHA256.get("dyadic/contact"),
                sha256_matches_expected=None,
                serialization_format=None,
                support=BUNDLE_OBSOLETE,
                model_family=None,
                train_meta=None,
                notes=(
                    "Obsolete standalone 'contact' detector is unsupported. "
                    "Use contact_intact + contact_occluded specialists.",
                ),
            )
        )
        notes.append("obsolete dyadic detector 'contact' reported as unsupported")

    status = REGISTRY_OK if bundles else REGISTRY_UNAVAILABLE
    if notes and status == REGISTRY_OK:
        status = REGISTRY_PARTIAL
    # Soft incompleteness on otherwise discovered trees still counts as partial.
    if status == REGISTRY_OK and any(
        b.support == BUNDLE_INCOMPLETE for b in bundles if b.detector_name != "contact"
    ):
        status = REGISTRY_PARTIAL
        notes.append("one or more discovered bundles are incomplete")

    return RegistrySnapshot(
        status=status,
        settings=validated,
        bundles=tuple(bundles),
        notes=tuple(notes),
    )


def require_registry(
    settings: ModelRegistrySettings | None = None,
    *,
    root: Path | str | None = None,
) -> RegistrySnapshot:
    """Discover the registry or raise if unavailable."""

    snapshot = discover_model_registry(settings, root=root)
    if snapshot.status == REGISTRY_UNAVAILABLE:
        detail = "; ".join(snapshot.notes) if snapshot.notes else "unavailable"
        raise ModelRegistryError(f"Model registry unavailable: {detail}")
    return snapshot
