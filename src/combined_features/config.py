"""Subsystem 06 configuration: feature calculation + model registry settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from combined_features.contracts import (
    ADAPTER_POLICY_VERSION,
    DYADIC_FEATURES_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION,
    CombinedFeaturesError,
)


@dataclass(frozen=True, slots=True)
class ModelRegistrySettings:
    """Explicit path to dyadic/focal detector bundles (never implied from legacy/)."""

    root: Path | None = None
    dyadic_subdir: str = "dyadic/v1"
    focal_subdir: str = "focal/v1"
    include_obsolete: bool = False
    include_focal: bool = True

    def validated(self) -> ModelRegistrySettings:
        dyadic = self.dyadic_subdir.strip().replace("\\", "/")
        focal = self.focal_subdir.strip().replace("\\", "/")
        if not dyadic:
            raise CombinedFeaturesError("model_registry.dyadic_subdir must not be empty.")
        if not focal:
            raise CombinedFeaturesError("model_registry.focal_subdir must not be empty.")
        root = Path(self.root).expanduser().resolve() if self.root is not None else None
        return replace(
            self,
            root=root,
            dyadic_subdir=dyadic,
            focal_subdir=focal,
            include_obsolete=bool(self.include_obsolete),
            include_focal=bool(self.include_focal),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root) if self.root is not None else None,
            "dyadic_subdir": self.dyadic_subdir,
            "focal_subdir": self.focal_subdir,
            "include_obsolete": self.include_obsolete,
            "include_focal": self.include_focal,
        }


@dataclass(frozen=True, slots=True)
class CombinedFeaturesConfig:
    """Immutable S6 settings snapshot content."""

    smooth_win: int = 9
    diff_dt: int = 1
    iteration_name: str = "suite"
    exp_name: str | None = None
    model_registry: ModelRegistrySettings = ModelRegistrySettings()
    adapter_policy_version: str = ADAPTER_POLICY_VERSION
    dyadic_features_schema_version: str = DYADIC_FEATURES_SCHEMA_VERSION

    def validated(self) -> CombinedFeaturesConfig:
        smooth = int(self.smooth_win)
        diff = int(self.diff_dt)
        if smooth < 1 or smooth % 2 == 0:
            raise CombinedFeaturesError(
                f"smooth_win must be an odd integer >= 1; got {smooth}."
            )
        if diff < 1:
            raise CombinedFeaturesError(f"diff_dt must be >= 1; got {diff}.")
        iteration = self.iteration_name.strip()
        if not iteration:
            raise CombinedFeaturesError("iteration_name must not be empty.")
        exp = self.exp_name.strip() if isinstance(self.exp_name, str) else None
        if exp == "":
            exp = None
        return replace(
            self,
            smooth_win=smooth,
            diff_dt=diff,
            iteration_name=iteration,
            exp_name=exp,
            model_registry=self.model_registry.validated(),
            adapter_policy_version=str(self.adapter_policy_version),
            dyadic_features_schema_version=str(self.dyadic_features_schema_version),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "smooth_win": self.smooth_win,
            "diff_dt": self.diff_dt,
            "iteration_name": self.iteration_name,
            "exp_name": self.exp_name,
            "adapter_policy_version": self.adapter_policy_version,
            "dyadic_features_schema_version": self.dyadic_features_schema_version,
            "model_registry": self.model_registry.to_settings_dict(),
        }


def load_combined_features_config(path: Path | str | None) -> CombinedFeaturesConfig:
    """Load S6 settings from YAML, or return defaults when ``path`` is None."""

    if path is None:
        return CombinedFeaturesConfig().validated()
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise CombinedFeaturesError(f"S6 config YAML is missing: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise CombinedFeaturesError(f"Could not load S6 config: {file_path}: {exc}") from exc
    if payload is None:
        return CombinedFeaturesConfig().validated()
    if not isinstance(payload, Mapping):
        raise CombinedFeaturesError(f"S6 config root must be a mapping: {file_path}")
    return config_from_mapping(payload)


def config_from_mapping(payload: Mapping[str, Any]) -> CombinedFeaturesConfig:
    """Build a validated config from a settings mapping."""

    registry_raw = payload.get("model_registry", {})
    if registry_raw is None:
        registry_raw = {}
    if not isinstance(registry_raw, Mapping):
        raise CombinedFeaturesError("model_registry must be a mapping when provided.")
    root_raw = registry_raw.get("root")
    root: Path | None = (
        None if root_raw is None or root_raw == "" else Path(str(root_raw))
    )
    registry = ModelRegistrySettings(
        root=root,
        dyadic_subdir=str(registry_raw.get("dyadic_subdir", "dyadic/v1")),
        focal_subdir=str(registry_raw.get("focal_subdir", "focal/v1")),
        include_obsolete=bool(registry_raw.get("include_obsolete", False)),
        include_focal=bool(registry_raw.get("include_focal", True)),
    )
    return CombinedFeaturesConfig(
        smooth_win=int(payload.get("smooth_win", 9)),
        diff_dt=int(payload.get("diff_dt", 1)),
        iteration_name=str(payload.get("iteration_name", "suite")),
        exp_name=(
            None
            if payload.get("exp_name") in (None, "")
            else str(payload.get("exp_name"))
        ),
        model_registry=registry,
        adapter_policy_version=str(
            payload.get("adapter_policy_version", ADAPTER_POLICY_VERSION)
        ),
        dyadic_features_schema_version=str(
            payload.get("dyadic_features_schema_version", DYADIC_FEATURES_SCHEMA_VERSION)
        ),
    ).validated()


def default_config_dict() -> dict[str, Any]:
    """Return the default settings payload (for docs / GUI defaults)."""

    return asdict(CombinedFeaturesConfig().validated())
