import numpy as np
import pytest

from preprocess.config import CanonicalResolutionConfig
from preprocess.crop_plan import CropMode, CropPlan, compute_canonical_geometry
from preprocess.exceptions import CropPlanError
from preprocess.manual_crop import make_manual_crop_plan

RAW_FRAME_SHAPE = (480, 640)
WIDE_POINTS = np.array(
    [
        [100.0, 100.0],
        [500.0, 100.0],
        [500.0, 300.0],
        [100.0, 300.0],
    ]
)


def _canonical_config(
    *,
    enabled: bool = True,
    size_wh: tuple[int, int] = (320, 240),
) -> CanonicalResolutionConfig:
    return CanonicalResolutionConfig(
        enabled=enabled,
        width=size_wh[0],
        height=size_wh[1],
    )


def _make_plan(
    *,
    points: np.ndarray = WIDE_POINTS,
    pre_crop_roi: tuple[int, int, int, int] | None = None,
    canonical_resolution: CanonicalResolutionConfig | None = None,
) -> CropPlan:
    return make_manual_crop_plan(
        raw_frame_shape=RAW_FRAME_SHAPE,
        points_tl_tr_br_bl=points,
        pre_crop_roi=pre_crop_roi,
        canonical_resolution=canonical_resolution or _canonical_config(),
    )


def _transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
    transformed = (homography @ homogeneous_points.T).T
    return transformed[:, :2] / transformed[:, 2, np.newaxis]


def test_valid_manual_crop_returns_unaccepted_manual_crop_plan() -> None:
    plan = _make_plan()

    assert isinstance(plan, CropPlan)
    assert plan.mode is CropMode.MANUAL
    assert plan.accepted_by_user is False
    assert plan.prepared_size_wh == (320, 240)


def test_manual_crop_has_valid_forward_and_inverse_homographies() -> None:
    plan = _make_plan()

    np.testing.assert_allclose(
        plan.H_prepared_to_raw_3x3 @ plan.H_raw_to_prepared_3x3,
        np.eye(3),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.all(np.isfinite(plan.H_raw_to_prepared_3x3))
    assert np.all(np.isfinite(plan.H_prepared_to_raw_3x3))


def test_manual_points_remain_in_raw_coordinates_without_pre_crop() -> None:
    plan = _make_plan(canonical_resolution=_canonical_config(enabled=False))

    np.testing.assert_array_equal(plan.quad_raw_tl_tr_br_bl, WIDE_POINTS)
    assert plan.pre_crop_roi.to_metadata_dict() == {
        "x": 0,
        "y": 0,
        "width": 640,
        "height": 480,
    }
    np.testing.assert_allclose(
        _transform_points(plan.H_raw_to_prepared_3x3, WIDE_POINTS),
        [[0.0, 0.0], [399.0, 0.0], [399.0, 199.0], [0.0, 199.0]],
        atol=1e-5,
    )


def test_pre_crop_is_retained_and_translation_is_composed() -> None:
    plan = _make_plan(
        pre_crop_roi=(50, 50, 500, 300),
        canonical_resolution=_canonical_config(enabled=False),
    )

    assert plan.pre_crop_roi.to_metadata_dict() == {
        "x": 50,
        "y": 50,
        "width": 500,
        "height": 300,
    }
    np.testing.assert_array_equal(plan.quad_raw_tl_tr_br_bl, WIDE_POINTS)
    np.testing.assert_allclose(
        _transform_points(plan.H_raw_to_prepared_3x3, WIDE_POINTS),
        [[0.0, 0.0], [399.0, 0.0], [399.0, 199.0], [0.0, 199.0]],
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "raw_frame_shape",
    [(0, 640), (480, -1), (480,), (480.0, 640)],
)
def test_invalid_raw_frame_shape_fails(raw_frame_shape: tuple[int, int]) -> None:
    with pytest.raises(CropPlanError, match="frame|height|width"):
        make_manual_crop_plan(
            raw_frame_shape=raw_frame_shape,
            points_tl_tr_br_bl=WIDE_POINTS,
            pre_crop_roi=None,
            canonical_resolution=_canonical_config(),
        )


def test_out_of_bounds_raw_point_fails() -> None:
    points = WIDE_POINTS.copy()
    points[1, 0] = 640.0

    with pytest.raises(CropPlanError, match="raw frame"):
        _make_plan(points=points)


def test_point_outside_pre_crop_fails() -> None:
    with pytest.raises(CropPlanError, match="pre-crop ROI"):
        _make_plan(pre_crop_roi=(150, 50, 400, 300))


def test_wrong_number_of_points_fails() -> None:
    with pytest.raises(CropPlanError, match="exactly four"):
        _make_plan(points=np.zeros((3, 2)))


def test_non_finite_point_fails() -> None:
    points = WIDE_POINTS.copy()
    points[0, 0] = np.nan

    with pytest.raises(CropPlanError, match="finite"):
        _make_plan(points=points)


def test_degenerate_quadrilateral_fails() -> None:
    points = np.array(
        [[100.0, 100.0], [300.0, 100.0], [500.0, 100.0], [100.0, 300.0]]
    )

    with pytest.raises(CropPlanError, match="degenerate"):
        _make_plan(points=points)


def test_self_intersecting_quadrilateral_fails() -> None:
    points = np.array(
        [[100.0, 100.0], [500.0, 300.0], [500.0, 100.0], [100.0, 300.0]]
    )

    with pytest.raises(CropPlanError, match="self-intersecting"):
        _make_plan(points=points)


def test_incorrect_point_order_fails_without_reordering() -> None:
    points = WIDE_POINTS[[0, 3, 2, 1]]

    with pytest.raises(CropPlanError, match="top-left"):
        _make_plan(points=points)


def test_canonical_output_is_even_and_preserves_native_aspect_ratio() -> None:
    plan = _make_plan(canonical_resolution=_canonical_config(size_wh=(300, 300)))
    native_plan = _make_plan(canonical_resolution=_canonical_config(enabled=False))
    geometry = compute_canonical_geometry(400, 200, 300, 300)
    canonical_only_transform = (
        plan.H_raw_to_prepared_3x3 @ native_plan.H_prepared_to_raw_3x3
    )

    assert plan.prepared_size_wh == (300, 300)
    assert all(dimension % 2 == 0 for dimension in plan.prepared_size_wh)
    assert geometry.uniform_scale == pytest.approx(0.75)
    assert geometry.scaled_size_wh == (300, 150)
    assert geometry.padding_left == geometry.padding_right == 0
    assert geometry.padding_top == geometry.padding_bottom == 75
    np.testing.assert_allclose(
        canonical_only_transform,
        [[0.75, 0.0, 0.0], [0.0, 0.75, 75.0], [0.0, 0.0, 1.0]],
        atol=1e-8,
    )


def test_rotation_convention_matches_automatic_clockwise_rotation() -> None:
    vertical_points = np.array(
        [[200.0, 50.0], [300.0, 50.0], [300.0, 350.0], [200.0, 350.0]]
    )
    plan = _make_plan(
        points=vertical_points,
        canonical_resolution=_canonical_config(enabled=False),
    )

    assert plan.rotated_90 is True
    assert plan.prepared_size_wh == (300, 100)
    np.testing.assert_allclose(
        _transform_points(plan.H_raw_to_prepared_3x3, vertical_points),
        [[299.0, 0.0], [299.0, 99.0], [0.0, 99.0], [0.0, 0.0]],
        atol=1e-5,
    )


def test_manual_detector_metrics_are_none() -> None:
    plan = _make_plan()

    assert plan.fit_score is None
    assert plan.rim_density is None
