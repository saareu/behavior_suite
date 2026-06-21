import json
from typing import Any

import numpy as np
import pytest

from preprocess.crop_plan import (
    CropMode,
    CropPlan,
    compute_canonical_geometry,
)
from preprocess.exceptions import CropPlanError


def _make_crop_plan(**updates: Any) -> CropPlan:
    forward = np.array(
        [
            [1.0, 0.0, -10.0],
            [0.0, 1.0, -20.0],
            [0.0, 0.0, 1.0],
        ]
    )
    values: dict[str, Any] = {
        "mode": CropMode.AUTOMATIC,
        "pre_crop_roi": (0, 0, 640, 480),
        "quad_raw_tl_tr_br_bl": np.array(
            [
                [10.0, 20.0],
                [610.0, 20.0],
                [610.0, 460.0],
                [10.0, 460.0],
            ]
        ),
        "H_raw_to_prepared_3x3": forward,
        "H_prepared_to_raw_3x3": np.linalg.inv(forward),
        "prepared_size_wh": (600, 440),
        "rotated_90": False,
        "fit_score": 0.95,
        "rim_density": 0.25,
        "accepted_by_user": True,
    }
    values.update(updates)
    return CropPlan(**values)


def test_valid_crop_plan_creation() -> None:
    plan = _make_crop_plan()

    assert plan.mode is CropMode.AUTOMATIC
    assert plan.pre_crop_roi.width == 640
    assert plan.prepared_size_wh == (600, 440)
    assert isinstance(plan.H_raw_to_prepared_3x3, np.ndarray)
    assert plan.H_raw_to_prepared_3x3.flags.writeable is False


@pytest.mark.parametrize(
    "field_name",
    ["H_raw_to_prepared_3x3", "H_prepared_to_raw_3x3"],
)
def test_invalid_homography_dimensions_are_rejected(field_name: str) -> None:
    with pytest.raises(CropPlanError, match="shape"):
        _make_crop_plan(**{field_name: np.eye(2)})


def test_non_finite_homography_values_are_rejected() -> None:
    homography = np.eye(3)
    homography[0, 0] = np.nan

    with pytest.raises(CropPlanError, match="finite"):
        _make_crop_plan(H_raw_to_prepared_3x3=homography)


def test_invalid_inverse_homography_relationship_is_rejected() -> None:
    with pytest.raises(CropPlanError, match="must be the inverse"):
        _make_crop_plan(H_prepared_to_raw_3x3=np.eye(3))


@pytest.mark.parametrize(
    "prepared_size_wh",
    [(601, 440), (600, 441), (0, 440), (600, -2)],
)
def test_odd_or_invalid_output_dimensions_are_rejected(
    prepared_size_wh: tuple[int, int],
) -> None:
    with pytest.raises(CropPlanError):
        _make_crop_plan(prepared_size_wh=prepared_size_wh)


def test_incorrect_number_of_quad_points_is_rejected() -> None:
    with pytest.raises(CropPlanError, match="exactly four"):
        _make_crop_plan(quad_raw_tl_tr_br_bl=np.zeros((3, 2)))


def test_degenerate_quadrilateral_is_rejected() -> None:
    degenerate = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(CropPlanError, match="degenerate"):
        _make_crop_plan(quad_raw_tl_tr_br_bl=degenerate)


def test_self_intersecting_quadrilateral_is_rejected() -> None:
    bow_tie = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )

    with pytest.raises(CropPlanError, match="self-intersecting"):
        _make_crop_plan(quad_raw_tl_tr_br_bl=bow_tie)


def test_incorrect_quad_order_is_rejected() -> None:
    reversed_order = np.array(
        [
            [10.0, 20.0],
            [10.0, 460.0],
            [610.0, 460.0],
            [610.0, 20.0],
        ]
    )

    with pytest.raises(CropPlanError, match="top-left"):
        _make_crop_plan(quad_raw_tl_tr_br_bl=reversed_order)


def test_crop_plan_serialization_is_json_safe_and_round_trips() -> None:
    plan = _make_crop_plan()

    metadata = plan.to_metadata_dict()
    encoded = json.dumps(metadata)
    restored = CropPlan.from_metadata_dict(json.loads(encoded))

    assert metadata["mode"] == "automatic"
    assert metadata["pre_crop_roi"] == {
        "x": 0,
        "y": 0,
        "width": 640,
        "height": 480,
    }
    assert isinstance(metadata["H_raw_to_prepared_3x3"], list)
    np.testing.assert_allclose(
        restored.H_raw_to_prepared_3x3,
        plan.H_raw_to_prepared_3x3,
    )
    np.testing.assert_allclose(
        restored.quad_raw_tl_tr_br_bl,
        plan.quad_raw_tl_tr_br_bl,
    )


def test_canonical_geometry_with_same_aspect_ratio() -> None:
    geometry = compute_canonical_geometry(640, 480, 320, 240)

    assert geometry.uniform_scale == pytest.approx(0.5)
    assert geometry.scaled_size_wh == (320, 240)
    assert (
        geometry.padding_left,
        geometry.padding_right,
        geometry.padding_top,
        geometry.padding_bottom,
    ) == (0, 0, 0, 0)


def test_canonical_geometry_requiring_horizontal_padding() -> None:
    geometry = compute_canonical_geometry(100, 100, 200, 100)

    assert geometry.uniform_scale == pytest.approx(1.0)
    assert geometry.scaled_size_wh == (100, 100)
    assert (geometry.padding_left, geometry.padding_right) == (50, 50)
    assert (geometry.padding_top, geometry.padding_bottom) == (0, 0)


def test_canonical_geometry_requiring_vertical_padding() -> None:
    geometry = compute_canonical_geometry(200, 100, 200, 200)

    assert geometry.uniform_scale == pytest.approx(1.0)
    assert geometry.scaled_size_wh == (200, 100)
    assert (geometry.padding_left, geometry.padding_right) == (0, 0)
    assert (geometry.padding_top, geometry.padding_bottom) == (50, 50)


def test_canonical_geometry_uses_one_uniform_scale() -> None:
    geometry = compute_canonical_geometry(403, 211, 928, 528)
    scaled_width, scaled_height = geometry.scaled_size_wh

    assert scaled_width == round(403 * geometry.uniform_scale)
    assert scaled_height == round(211 * geometry.uniform_scale)
    assert scaled_width + geometry.padding_left + geometry.padding_right == 928
    assert scaled_height + geometry.padding_top + geometry.padding_bottom == 528


@pytest.mark.parametrize(
    ("canonical_width", "canonical_height"),
    [(0, 100), (100, 0), (101, 100), (100, 101)],
)
def test_invalid_canonical_dimensions_are_rejected(
    canonical_width: int,
    canonical_height: int,
) -> None:
    with pytest.raises(CropPlanError):
        compute_canonical_geometry(100, 100, canonical_width, canonical_height)
