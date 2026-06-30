from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

import numpy as np
import pytest

from preprocess.config import MaskConfig
from preprocess.exceptions import VideoPreparationError
from preprocess.masking import (
    apply_static_mask_to_frame,
    rasterize_static_mask,
    serialize_static_mask,
    validate_static_mask,
)


def test_public_masking_api_exports_are_importable() -> None:
    module = importlib.import_module("preprocess.masking")
    expected_exports = {
        "apply_static_mask_to_frame",
        "rasterize_static_mask",
        "serialize_static_mask",
        "validate_static_mask",
    }

    assert expected_exports.issubset(module.__all__)
    for name in expected_exports:
        assert isinstance(getattr(module, name), Callable)

    assert apply_static_mask_to_frame is module.apply_static_mask_to_frame
    assert validate_static_mask is module.validate_static_mask


def test_preprocessing_and_cli_imports_do_not_import_qt() -> None:
    before = {name for name in sys.modules if name.startswith("PySide6")}

    for module_name in (
        "preprocess.masking",
        "preprocess.video_prepare",
        "preprocess.service",
        "cli.preprocess",
    ):
        importlib.import_module(module_name)

    after = {name for name in sys.modules if name.startswith("PySide6")}
    assert after == before


@pytest.mark.parametrize(
    "frame",
    [
        np.full((5, 6), 100, dtype=np.uint8),
        np.full((5, 6, 3), 100, dtype=np.uint8),
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        MaskConfig(enabled=False, shapes=[]),
        MaskConfig(
            enabled=False,
            shapes=[{"type": "rectangle", "x": 50, "y": 50, "width": 2, "height": 2}],
        ),
        MaskConfig(enabled=True, shapes=[]),
    ],
)
def test_disabled_or_empty_static_masks_are_no_ops(
    frame: np.ndarray,
    mask: MaskConfig,
) -> None:
    masked = apply_static_mask_to_frame(frame, mask)
    raster = rasterize_static_mask(mask, (frame.shape[1], frame.shape[0]))

    np.testing.assert_array_equal(masked, frame)
    assert not np.shares_memory(masked, frame)
    assert raster.shape == frame.shape[:2]
    assert raster.dtype == np.uint8
    assert np.count_nonzero(raster) == 0


def test_rectangle_static_mask_blacks_exact_half_open_region() -> None:
    frame = np.full((5, 6, 3), 100, dtype=np.uint8)
    mask = MaskConfig(
        enabled=True,
        shapes=[{"type": "rectangle", "x": 1, "y": 2, "width": 3, "height": 2}],
    )
    expected = np.array(frame, copy=True)
    expected[2:4, 1:4, :] = 0

    masked = apply_static_mask_to_frame(frame, mask)

    np.testing.assert_array_equal(masked, expected)
    np.testing.assert_array_equal(frame, np.full((5, 6, 3), 100, dtype=np.uint8))


def test_polygon_static_mask_rasterizes_and_blacks_selected_pixels() -> None:
    frame = np.full((6, 6), 200, dtype=np.uint8)
    mask = MaskConfig(
        enabled=True,
        shapes=[{"type": "polygon", "vertices": [(1, 1), (4, 1), (1, 4)]}],
    )

    raster = rasterize_static_mask(mask, (6, 6))
    masked = apply_static_mask_to_frame(frame, mask)

    assert raster[2, 2] == 255
    assert raster[0, 0] == 0
    assert masked[2, 2] == 0
    assert masked[0, 0] == 200


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "enabled": True,
                "shapes": [{"type": "rectangle", "x": 0.5, "y": 0, "width": 2, "height": 2}],
            },
            "integer",
        ),
        (
            {
                "enabled": True,
                "shapes": [{"type": "rectangle", "x": 0, "y": 0, "width": 0, "height": 2}],
            },
            "positive",
        ),
        (
            {
                "enabled": True,
                "shapes": [{"type": "polygon", "vertices": [(0, 0), (1, 1)]}],
            },
            "at least three",
        ),
        (
            {
                "enabled": True,
                "shapes": [{"type": "polygon", "vertices": [(0, 0), (1, 1), (2, 2)]}],
            },
            "nonzero area",
        ),
        (
            {
                "enabled": True,
                "shapes": [
                    {"type": "polygon", "vertices": [(0, 0), (3, 3), (0, 4), (4, 0)]}
                ],
            },
            "simple",
        ),
    ],
)
def test_invalid_static_mask_model_geometry_fails_clearly(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MaskConfig.model_validate(payload)


@pytest.mark.parametrize(
    "mask",
    [
        MaskConfig(
            enabled=True,
            shapes=[{"type": "rectangle", "x": 5, "y": 0, "width": 2, "height": 1}],
        ),
        MaskConfig(
            enabled=True,
            shapes=[{"type": "polygon", "vertices": [(0, 0), (6, 1), (0, 3)]}],
        ),
    ],
)
def test_enabled_static_mask_out_of_bounds_geometry_fails_clearly(mask: MaskConfig) -> None:
    with pytest.raises(VideoPreparationError, match="outside"):
        validate_static_mask(mask, (6, 6))


def test_static_mask_serialization_preserves_disabled_empty_compatibility() -> None:
    serialized = serialize_static_mask(MaskConfig())

    assert serialized == {
        "enabled": False,
        "coordinate_space": "prepared_pixels",
        "fill_value": "black",
        "shapes": [],
    }
