from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from preprocess.config import PreprocessConfig, load_preprocess_config
from preprocess.exceptions import PreprocessConfigError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "preprocess_default.yaml"


def _default_mapping() -> dict[str, Any]:
    loaded = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_default_config_loads_and_preserves_required_invariants() -> None:
    config = load_preprocess_config(DEFAULT_CONFIG_PATH)

    assert config.schema_version == "preprocess_config_v1"
    assert config.timing.allow_frame_resampling is False
    assert config.timing.require_one_to_one_frame_mapping is True
    assert config.prepare.canonical_resolution.width == 928
    assert config.prepare.canonical_resolution.height == 528
    assert config.encoding.ffmpeg.bframes == 0
    assert config.encoding.opencv.fourcc == "mp4v"


def test_invalid_canonical_dimension_is_rejected() -> None:
    values = _default_mapping()
    values["prepare"]["canonical_resolution"]["width"] = 927

    with pytest.raises(ValidationError, match="must both be even"):
        PreprocessConfig.model_validate(values)


def test_invalid_trim_range_is_rejected() -> None:
    values = _default_mapping()
    values["trim"]["start_frame"] = 10
    values["trim"]["end_frame"] = 10

    with pytest.raises(ValidationError, match="greater than"):
        PreprocessConfig.model_validate(values)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("timing", "allow_frame_resampling", True),
        ("timing", "require_one_to_one_frame_mapping", False),
        ("mask", "enabled", True),
    ],
)
def test_unsupported_configuration_combination_is_rejected(
    section: str,
    key: str,
    value: object,
) -> None:
    values = _default_mapping()
    values[section][key] = value

    with pytest.raises(ValidationError):
        PreprocessConfig.model_validate(values)


def test_pre_crop_enabled_state_must_match_mode() -> None:
    values = _default_mapping()
    values["pre_crop"]["enabled"] = False
    values["pre_crop"]["mode"] = "vertical_keep_left"

    with pytest.raises(ValidationError, match="must be 'none'"):
        PreprocessConfig.model_validate(values)


def test_missing_config_file_raises_clear_domain_error(tmp_path: Path) -> None:
    with pytest.raises(PreprocessConfigError, match="does not exist"):
        load_preprocess_config(tmp_path / "missing.yaml")
