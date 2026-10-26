"""Threshold resolution precedence tests."""

from __future__ import annotations

import pytest

from behavioral_inference import (
    BehavioralInferenceConfig,
    ThresholdConfigError,
    load_behavioral_inference_config,
    resolve_working_threshold,
)


def test_default_yaml_loads() -> None:
    config = load_behavioral_inference_config()
    assert config.schema_version == "behavioral_inference_settings_v1"
    assert config.working_thresholds["nose_to_nose"] == 0.0816
    assert config.working_thresholds["contact"]["intact"] == 0.9731
    assert config.working_thresholds["nose_to_body"]["occluded"] == 0.4825


def test_config_threshold_source() -> None:
    config = BehavioralInferenceConfig().validated()
    resolved = resolve_working_threshold("nose_to_nose", config)
    assert resolved.value == 0.0816
    assert resolved.source == "inference_config"


def test_specialist_nested_thresholds() -> None:
    config = BehavioralInferenceConfig().validated()
    intact = resolve_working_threshold("contact_intact", config)
    occ = resolve_working_threshold("contact_occluded", config)
    assert intact.value == 0.9731
    assert occ.value == 0.4295
    nbi = resolve_working_threshold("nose_body_intact", config)
    nbo = resolve_working_threshold("nose_body_occ", config)
    assert nbi.value == 0.6141
    assert nbo.value == 0.4825


def test_user_override_precedence() -> None:
    config = BehavioralInferenceConfig(
        user_threshold_overrides={"nose_to_nose": 0.42}
    ).validated()
    resolved = resolve_working_threshold("nose_to_nose", config)
    assert resolved.value == 0.42
    assert resolved.source == "user_override"


def test_user_override_nested_specialist() -> None:
    config = BehavioralInferenceConfig(
        user_threshold_overrides={"contact": {"intact": 0.9}}
    ).validated()
    resolved = resolve_working_threshold("contact_intact", config)
    assert resolved.value == 0.9
    assert resolved.source == "user_override"


def test_missing_threshold_fails() -> None:
    config = BehavioralInferenceConfig(working_thresholds={}).validated()
    with pytest.raises(ThresholdConfigError, match="No working threshold"):
        resolve_working_threshold("nose_to_nose", config)


def test_never_uses_silent_half_by_default() -> None:
    config = BehavioralInferenceConfig(
        working_thresholds={"approach": 0.3766},
        allow_default_threshold_0_5=False,
    ).validated()
    with pytest.raises(ThresholdConfigError, match="0\\.5"):
        resolve_working_threshold("retract", config)


def test_parity_flag_allows_half() -> None:
    config = BehavioralInferenceConfig(
        working_thresholds={},
        allow_default_threshold_0_5=True,
    ).validated()
    resolved = resolve_working_threshold("retract", config)
    assert resolved.value == 0.5
