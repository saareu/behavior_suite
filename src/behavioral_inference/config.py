"""Versioned behavioral-inference configuration and threshold resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from behavioral_inference.contracts import (
    SETTINGS_SCHEMA_VERSION,
    ThresholdConfigError,
)
from combined_features.config import ModelRegistrySettings

ThresholdSource = Literal["user_override", "inference_config"]


# Default working thresholds from legacy dyadic_inference.yaml + 90_focal_inference.yaml.
DEFAULT_WORKING_THRESHOLDS: dict[str, Any] = {
    "contact": {"intact": 0.9731, "occluded": 0.4295},
    "nose_to_nose": 0.0816,
    "mutually_disengaged": 0.0469,
    "side_by_side": 0.0095,
    "nose_to_body": {"intact": 0.6141, "occluded": 0.4825},
    "nose_to_tail": 0.6636,
    "approach": 0.3766,
    "retract": 0.4892,
}

# Map specialist / runtime bundle names onto nested YAML threshold keys.
_SPECIALIST_THRESHOLD_KEYS: dict[str, tuple[str, str]] = {
    "contact_intact": ("contact", "intact"),
    "contact_occluded": ("contact", "occluded"),
    "nose_body_intact": ("nose_to_body", "intact"),
    "nose_body_occ": ("nose_to_body", "occluded"),
}


@dataclass(frozen=True, slots=True)
class ResolvedThreshold:
    """Effective working threshold with an explicit provenance source."""

    detector_key: str
    value: float
    source: ThresholdSource


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    """Family-adapter flags for inference and artifact publication."""

    dyadic_collapse: str = "lower_id_as_focal"
    nose_to_body_rolling_median_enabled: bool = True
    nose_to_body_rolling_window: int = 5
    nose_to_body_rolling_min_periods: int = 1
    fail_fast_on_detector_error: bool = True
    allow_nose_to_body_single_stage_fallback: bool = True
    require_exactly_two_mice: bool = True
    emit_wide_views: bool = True

    def validated(self) -> AdapterSettings:
        collapse = str(self.dyadic_collapse).strip() or "lower_id_as_focal"
        if collapse != "lower_id_as_focal":
            raise ThresholdConfigError(
                f"Unsupported dyadic_collapse policy '{collapse}'. "
                "S7 MVP supports only 'lower_id_as_focal'."
            )
        window = int(self.nose_to_body_rolling_window)
        min_periods = int(self.nose_to_body_rolling_min_periods)
        if window < 1:
            raise ThresholdConfigError("nose_to_body rolling window must be >= 1")
        if min_periods < 1 or min_periods > window:
            raise ThresholdConfigError(
                "nose_to_body rolling min_periods must satisfy 1 <= min_periods <= window"
            )
        return replace(
            self,
            dyadic_collapse=collapse,
            nose_to_body_rolling_median_enabled=bool(
                self.nose_to_body_rolling_median_enabled
            ),
            nose_to_body_rolling_window=window,
            nose_to_body_rolling_min_periods=min_periods,
            fail_fast_on_detector_error=bool(self.fail_fast_on_detector_error),
            allow_nose_to_body_single_stage_fallback=bool(
                self.allow_nose_to_body_single_stage_fallback
            ),
            require_exactly_two_mice=bool(self.require_exactly_two_mice),
            emit_wide_views=bool(self.emit_wide_views),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "dyadic_collapse": self.dyadic_collapse,
            "nose_to_body_rolling_median": {
                "enabled": self.nose_to_body_rolling_median_enabled,
                "window": self.nose_to_body_rolling_window,
                "min_periods": self.nose_to_body_rolling_min_periods,
            },
            "fail_fast_on_detector_error": self.fail_fast_on_detector_error,
            "allow_nose_to_body_single_stage_fallback": (
                self.allow_nose_to_body_single_stage_fallback
            ),
            "require_exactly_two_mice": self.require_exactly_two_mice,
            "emit_wide_views": self.emit_wide_views,
        }


@dataclass(frozen=True, slots=True)
class BehavioralInferenceConfig:
    """Immutable inference settings (thresholds + registry + selection + adapters)."""

    schema_version: str = SETTINGS_SCHEMA_VERSION
    model_iteration: str = "v1"
    parity_mode: str = "legacy_parity"
    model_registry: ModelRegistrySettings = field(default_factory=ModelRegistrySettings)
    active_detectors_dyadic: tuple[str, ...] = (
        "contact",
        "nose_to_nose",
        "mutually_disengaged",
        "side_by_side",
    )
    active_detectors_focal: tuple[str, ...] = (
        "nose_to_body",
        "nose_to_tail",
        "approach",
        "retract",
    )
    working_thresholds: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_WORKING_THRESHOLDS)
    )
    user_threshold_overrides: Mapping[str, Any] = field(default_factory=dict)
    allow_default_threshold_0_5: bool = False
    adapters: AdapterSettings = field(default_factory=AdapterSettings)

    def validated(self) -> BehavioralInferenceConfig:
        schema = str(self.schema_version).strip() or SETTINGS_SCHEMA_VERSION
        iteration = str(self.model_iteration).strip()
        if not iteration:
            raise ThresholdConfigError("model_iteration must not be empty")
        parity = str(self.parity_mode).strip() or "legacy_parity"
        if parity not in ("legacy_parity", "raw_framewise"):
            raise ThresholdConfigError(
                f"Unsupported parity_mode '{parity}'. "
                "Expected 'legacy_parity' or 'raw_framewise'."
            )
        registry = self.model_registry.validated()
        dyadic = tuple(str(x).strip() for x in self.active_detectors_dyadic if str(x).strip())
        focal = tuple(str(x).strip() for x in self.active_detectors_focal if str(x).strip())
        thresholds = _freeze_mapping(self.working_thresholds)
        overrides = _freeze_mapping(self.user_threshold_overrides)
        adapters = self.adapters.validated()
        # raw_framewise disables the legacy nose-to-body smoother by policy.
        if parity == "raw_framewise" and adapters.nose_to_body_rolling_median_enabled:
            adapters = replace(adapters, nose_to_body_rolling_median_enabled=False)
        return replace(
            self,
            schema_version=schema,
            model_iteration=iteration,
            parity_mode=parity,
            model_registry=registry,
            active_detectors_dyadic=dyadic,
            active_detectors_focal=focal,
            working_thresholds=thresholds,
            user_threshold_overrides=overrides,
            allow_default_threshold_0_5=bool(self.allow_default_threshold_0_5),
            adapters=adapters,
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_iteration": self.model_iteration,
            "parity_mode": self.parity_mode,
            "model_registry": self.model_registry.to_settings_dict(),
            "active_detectors": {
                "dyadic": list(self.active_detectors_dyadic),
                "focal": list(self.active_detectors_focal),
            },
            "working_thresholds": _unfreeze_mapping(self.working_thresholds),
            "user_threshold_overrides": _unfreeze_mapping(self.user_threshold_overrides),
            "allow_default_threshold_0_5": self.allow_default_threshold_0_5,
            "adapters": self.adapters.to_settings_dict(),
        }


def _freeze_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            out[str(key)] = _freeze_mapping(value)
        else:
            out[str(key)] = value
    return out


def _unfreeze_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            out[str(key)] = _unfreeze_mapping(value)
        else:
            out[str(key)] = value
    return out


def _lookup_threshold_value(
    store: Mapping[str, Any],
    detector_key: str,
) -> float | None:
    """Resolve a scalar threshold from flat or nested YAML-shaped mappings."""

    if detector_key in store and not isinstance(store[detector_key], Mapping):
        try:
            return float(store[detector_key])
        except (TypeError, ValueError):
            return None

    nested = _SPECIALIST_THRESHOLD_KEYS.get(detector_key)
    if nested is not None:
        parent, child = nested
        parent_val = store.get(parent)
        if isinstance(parent_val, Mapping) and child in parent_val:
            try:
                return float(parent_val[child])
            except (TypeError, ValueError):
                return None

    # Allow dotted keys in overrides: "contact.intact"
    if "." in detector_key:
        parent, child = detector_key.split(".", 1)
        parent_val = store.get(parent)
        if isinstance(parent_val, Mapping) and child in parent_val:
            try:
                return float(parent_val[child])
            except (TypeError, ValueError):
                return None
        if detector_key in store and not isinstance(store[detector_key], Mapping):
            try:
                return float(store[detector_key])
            except (TypeError, ValueError):
                return None

    # Logical nested parents without specialist suffix are not scalars.
    return None


def resolve_working_threshold(
    detector_key: str,
    config: BehavioralInferenceConfig,
    *,
    user_overrides: Mapping[str, Any] | None = None,
) -> ResolvedThreshold:
    """Resolve a working threshold with binding precedence.

    1. Explicit user override
    2. Inference configuration ``working_thresholds``
    3. Hard error (never silently use ``0.5``, bundle ``threshold_working``,
       or other undocumented defaults — unless
       ``allow_default_threshold_0_5`` is enabled for legacy parity)
    """

    key = str(detector_key).strip()
    if not key:
        raise ThresholdConfigError("detector_key must not be empty")

    overrides = user_overrides if user_overrides is not None else config.user_threshold_overrides
    override_value = _lookup_threshold_value(overrides, key)
    if override_value is not None:
        return ResolvedThreshold(detector_key=key, value=override_value, source="user_override")

    config_value = _lookup_threshold_value(config.working_thresholds, key)
    if config_value is not None:
        return ResolvedThreshold(
            detector_key=key, value=config_value, source="inference_config"
        )

    if config.allow_default_threshold_0_5:
        # Documented legacy Pipeline 70/90 behavior — opt-in only.
        return ResolvedThreshold(detector_key=key, value=0.5, source="inference_config")

    raise ThresholdConfigError(
        f"No working threshold configured for '{key}'. Provide an entry in "
        "inference configuration working_thresholds or an explicit user override. "
        "Refusing to substitute 0.5 or bundle threshold_working."
    )


def default_suite_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "subsystem_07"
        / "behavioral_inference_default.yaml"
    )


def load_behavioral_inference_config(
    path: Path | str | None = None,
) -> BehavioralInferenceConfig:
    """Load inference settings from YAML, or suite defaults when ``path`` is None."""

    if path is None:
        suite = default_suite_config_path()
        if suite.is_file():
            return load_behavioral_inference_config(suite)
        return BehavioralInferenceConfig().validated()

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ThresholdConfigError(f"Inference config YAML is missing: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ThresholdConfigError(
            f"Could not load inference config: {file_path}: {exc}"
        ) from exc
    if payload is None:
        return BehavioralInferenceConfig().validated()
    if not isinstance(payload, Mapping):
        raise ThresholdConfigError(f"Inference config root must be a mapping: {file_path}")
    return config_from_mapping(payload)


def config_from_mapping(payload: Mapping[str, Any]) -> BehavioralInferenceConfig:
    registry_raw = payload.get("model_registry", {})
    if registry_raw is None:
        registry_raw = {}
    if not isinstance(registry_raw, Mapping):
        raise ThresholdConfigError("model_registry must be a mapping when provided")
    root_raw = registry_raw.get("root")
    root: Path | None = None if root_raw in (None, "") else Path(str(root_raw))
    registry = ModelRegistrySettings(
        root=root,
        dyadic_subdir=str(registry_raw.get("dyadic_subdir", "dyadic/v1")),
        focal_subdir=str(registry_raw.get("focal_subdir", "focal/v1")),
        include_obsolete=bool(registry_raw.get("include_obsolete", False)),
        include_focal=bool(registry_raw.get("include_focal", True)),
    )

    active_raw = payload.get("active_detectors", {})
    if active_raw is None:
        active_raw = {}
    if not isinstance(active_raw, Mapping):
        raise ThresholdConfigError("active_detectors must be a mapping when provided")
    dyadic = active_raw.get("dyadic")
    focal = active_raw.get("focal")

    thresholds_raw = payload.get("working_thresholds", DEFAULT_WORKING_THRESHOLDS)
    if thresholds_raw is None:
        thresholds_raw = {}
    if not isinstance(thresholds_raw, Mapping):
        raise ThresholdConfigError("working_thresholds must be a mapping")

    overrides_raw = payload.get("user_threshold_overrides", {})
    if overrides_raw is None:
        overrides_raw = {}
    if not isinstance(overrides_raw, Mapping):
        raise ThresholdConfigError("user_threshold_overrides must be a mapping")

    adapters = _adapters_from_mapping(payload.get("adapters"))

    return BehavioralInferenceConfig(
        schema_version=str(payload.get("schema_version", SETTINGS_SCHEMA_VERSION)),
        model_iteration=str(payload.get("model_iteration", "v1")),
        parity_mode=str(payload.get("parity_mode", "legacy_parity")),
        model_registry=registry,
        active_detectors_dyadic=tuple(
            str(x)
            for x in (
                dyadic
                if isinstance(dyadic, list)
                else BehavioralInferenceConfig().active_detectors_dyadic
            )
        ),
        active_detectors_focal=tuple(
            str(x)
            for x in (
                focal
                if isinstance(focal, list)
                else BehavioralInferenceConfig().active_detectors_focal
            )
        ),
        working_thresholds=dict(thresholds_raw),
        user_threshold_overrides=dict(overrides_raw),
        allow_default_threshold_0_5=bool(payload.get("allow_default_threshold_0_5", False)),
        adapters=adapters,
    ).validated()


def _adapters_from_mapping(raw: Any) -> AdapterSettings:
    if raw is None:
        return AdapterSettings()
    if not isinstance(raw, Mapping):
        raise ThresholdConfigError("adapters must be a mapping when provided")
    rolling_raw = raw.get("nose_to_body_rolling_median", {})
    if rolling_raw is None:
        rolling_raw = {}
    if not isinstance(rolling_raw, Mapping):
        raise ThresholdConfigError("adapters.nose_to_body_rolling_median must be a mapping")
    return AdapterSettings(
        dyadic_collapse=str(raw.get("dyadic_collapse", "lower_id_as_focal")),
        nose_to_body_rolling_median_enabled=bool(rolling_raw.get("enabled", True)),
        nose_to_body_rolling_window=int(rolling_raw.get("window", 5)),
        nose_to_body_rolling_min_periods=int(rolling_raw.get("min_periods", 1)),
        fail_fast_on_detector_error=bool(raw.get("fail_fast_on_detector_error", True)),
        allow_nose_to_body_single_stage_fallback=bool(
            raw.get("allow_nose_to_body_single_stage_fallback", True)
        ),
        require_exactly_two_mice=bool(raw.get("require_exactly_two_mice", True)),
        emit_wide_views=bool(raw.get("emit_wide_views", True)),
    )
