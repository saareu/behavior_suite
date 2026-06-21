import json

import numpy as np
import pytest

from preprocess.config import PreCropConfig
from preprocess.exceptions import PreCropError
from preprocess.pre_crop import (
    PreCropMode,
    build_pre_crop_translation_homography,
    resolve_pre_crop,
    serialize_pre_crop,
)


@pytest.mark.parametrize(
    ("config", "expected_mode", "expected_roi"),
    [
        (
            PreCropConfig(enabled=False, mode="none"),
            PreCropMode.NONE,
            (0, 0, 100, 80),
        ),
        (
            PreCropConfig(
                enabled=True,
                mode="vertical_keep_left",
                boundary_px=40,
            ),
            PreCropMode.VERTICAL_KEEP_LEFT,
            (0, 0, 40, 80),
        ),
        (
            PreCropConfig(
                enabled=True,
                mode="vertical_keep_right",
                boundary_px=40,
            ),
            PreCropMode.VERTICAL_KEEP_RIGHT,
            (40, 0, 60, 80),
        ),
        (
            PreCropConfig(
                enabled=True,
                mode="horizontal_keep_upper",
                boundary_px=30,
            ),
            PreCropMode.HORIZONTAL_KEEP_UPPER,
            (0, 0, 100, 30),
        ),
        (
            PreCropConfig(
                enabled=True,
                mode="horizontal_keep_lower",
                boundary_px=30,
            ),
            PreCropMode.HORIZONTAL_KEEP_LOWER,
            (0, 30, 100, 50),
        ),
        (
            PreCropConfig(
                enabled=True,
                mode="manual_rectangle",
                manual_rectangle=(10, 5, 40, 30),
            ),
            PreCropMode.MANUAL_RECTANGLE,
            (10, 5, 40, 30),
        ),
    ],
)
def test_resolve_all_pre_crop_modes(
    config: PreCropConfig,
    expected_mode: PreCropMode,
    expected_roi: tuple[int, int, int, int],
) -> None:
    resolved = resolve_pre_crop(config, raw_size_wh=(100, 80))

    assert resolved.mode is expected_mode
    assert (
        resolved.roi.x,
        resolved.roi.y,
        resolved.roi.width,
        resolved.roi.height,
    ) == expected_roi


def test_none_pre_crop_resolves_to_full_raw_frame() -> None:
    resolved = resolve_pre_crop(
        PreCropConfig(enabled=False, mode="none"),
        raw_size_wh=(1920, 1080),
    )

    assert resolved.roi.to_metadata_dict() == {
        "x": 0,
        "y": 0,
        "width": 1920,
        "height": 1080,
    }


@pytest.mark.parametrize(
    "config",
    [
        PreCropConfig(
            enabled=True,
            mode="vertical_keep_left",
            boundary_px=101,
        ),
        PreCropConfig(
            enabled=True,
            mode="vertical_keep_right",
            boundary_px=100,
        ),
        PreCropConfig(
            enabled=True,
            mode="horizontal_keep_upper",
            boundary_px=81,
        ),
        PreCropConfig(
            enabled=True,
            mode="horizontal_keep_lower",
            boundary_px=80,
        ),
    ],
)
def test_out_of_bounds_pre_crop_boundaries_are_rejected(
    config: PreCropConfig,
) -> None:
    with pytest.raises(PreCropError, match="boundary"):
        resolve_pre_crop(config, raw_size_wh=(100, 80))


@pytest.mark.parametrize(
    "rectangle",
    [
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (0, 0, 0, 10),
        (0, 0, 10, 0),
        (90, 0, 11, 10),
        (0, 70, 10, 11),
    ],
)
def test_invalid_manual_rectangle_is_rejected(
    rectangle: tuple[int, int, int, int],
) -> None:
    config = PreCropConfig(
        enabled=True,
        mode="manual_rectangle",
        manual_rectangle=rectangle,
    )

    with pytest.raises(PreCropError):
        resolve_pre_crop(config, raw_size_wh=(100, 80))


def test_pre_crop_translation_homography_maps_raw_origin_to_crop_origin() -> None:
    resolved = resolve_pre_crop(
        PreCropConfig(
            enabled=True,
            mode="manual_rectangle",
            manual_rectangle=(10, 5, 40, 30),
        ),
        raw_size_wh=(100, 80),
    )

    homography = build_pre_crop_translation_homography(resolved)
    translated = homography @ np.array([10.0, 5.0, 1.0])

    np.testing.assert_allclose(
        homography,
        [
            [1.0, 0.0, -10.0],
            [0.0, 1.0, -5.0],
            [0.0, 0.0, 1.0],
        ],
    )
    np.testing.assert_allclose(translated, [0.0, 0.0, 1.0])


def test_pre_crop_serialization_is_json_safe_and_records_final_region() -> None:
    resolved = resolve_pre_crop(
        PreCropConfig(
            enabled=True,
            mode="vertical_keep_right",
            boundary_px=25,
        ),
        raw_size_wh=(100, 80),
    )

    metadata = serialize_pre_crop(resolved)

    json.dumps(metadata)
    assert metadata["mode"] == "vertical_keep_right"
    assert metadata["roi"] == {"x": 25, "y": 0, "width": 75, "height": 80}
    assert metadata["defines_final_processing_region"] is True
