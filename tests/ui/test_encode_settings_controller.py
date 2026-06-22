from pathlib import Path

import numpy as np
import pytest

from preprocess.config import CanonicalResolutionConfig, PrepareConfig, PreprocessConfig
from preprocess.crop_plan import CropPlan
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.models import PreprocessResult
from ui.controllers.encode_settings_controller import (
    EncodeSettingsController,
    EncodeSettingsValidationError,
)
from ui.state import PreprocessSetupState


def _config() -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=100,
                height=80,
            )
        )
    )


def _accepted_crop(config: PreprocessConfig) -> CropPlan:
    candidate = make_manual_crop_plan(
        raw_frame_shape=(80, 100),
        points_tl_tr_br_bl=np.asarray(
            ((10.0, 10.0), (90.0, 10.0), (90.0, 70.0), (10.0, 70.0))
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )
    return CropPlan.model_validate(
        {**candidate.model_dump(mode="python"), "accepted_by_user": True}
    )


def _state(_tmp_path: Path) -> PreprocessSetupState:
    state = PreprocessSetupState()
    config = _config()
    state.set_loaded_preprocess_config(config)
    state.store_accepted_crop_plan(_accepted_crop(config))
    return state


def _prior_result() -> PreprocessResult:
    return PreprocessResult(
        success=False,
        message="prior run",
        elapsed_sec=1.0,
    )


def test_valid_config_update_stores_typed_copy(tmp_path: Path) -> None:
    state = _state(tmp_path)
    original_current = state.preprocess_config
    controller = EncodeSettingsController(state)

    updated = controller.update_settings(ffmpeg_crf=19, debug_enabled=True)

    assert isinstance(updated, PreprocessConfig)
    assert updated is state.preprocess_config
    assert updated is not original_current
    assert updated.encoding.ffmpeg.crf == 19
    assert updated.debug.enabled is True
    assert state.config_dirty is True
    assert state.encode_settings_valid is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("canonical_width", 0), ("canonical_height", 79)],
)
def test_invalid_canonical_dimensions_are_rejected(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    state = _state(tmp_path)
    original = state.preprocess_config
    controller = EncodeSettingsController(state)

    with pytest.raises(EncodeSettingsValidationError, match="Invalid encode setting"):
        controller.update_settings(**{field: value})

    assert state.preprocess_config == original
    assert state.encode_settings_valid is False
    assert state.config_dirty is True


def test_geometry_change_clears_crop_and_prior_run_result(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.preprocess_result = _prior_result()
    controller = EncodeSettingsController(state)

    controller.update_settings(canonical_width=120)

    assert state.candidate_crop_plan is None
    assert state.accepted_crop_plan is None
    assert state.preprocess_result is None
    assert "crop review" in state.crop_review_status.lower()


def test_non_geometry_change_preserves_accepted_crop_and_clears_result(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    accepted = state.accepted_crop_plan
    state.preprocess_result = _prior_result()
    controller = EncodeSettingsController(state)

    controller.update_settings(
        ffmpeg_crf=17,
        ffmpeg_preset="slow",
        ffmpeg_faststart=True,
        opencv_fourcc="mp4v",
        debug_enabled=True,
    )

    assert state.accepted_crop_plan is accepted
    assert state.preprocess_result is None


def test_reset_restores_original_loaded_encode_config(tmp_path: Path) -> None:
    state = _state(tmp_path)
    original = state.original_preprocess_config
    controller = EncodeSettingsController(state)
    controller.update_settings(ffmpeg_crf=18, debug_enabled=True)

    restored = controller.reset_to_loaded_config()

    assert original is not None
    assert restored.encoding == original.encoding
    assert restored.prepare.canonical_resolution == (
        original.prepare.canonical_resolution
    )
    assert restored.debug == original.debug
    assert state.config_dirty is False


def test_encode_settings_blocks_next_when_config_is_invalid(tmp_path: Path) -> None:
    state = _state(tmp_path)
    controller = EncodeSettingsController(state)
    with pytest.raises(EncodeSettingsValidationError):
        controller.update_settings(canonical_width=101)

    assert controller.can_advance() is False
    with pytest.raises(EncodeSettingsValidationError, match="valid typed"):
        controller.validate_navigation()
