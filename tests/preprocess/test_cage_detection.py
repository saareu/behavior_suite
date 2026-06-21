from pathlib import Path
from typing import NoReturn

import cv2
import numpy as np
import pytest

from preprocess import cage_detection
from preprocess.cage_detection import (
    CageDetectionResult,
    build_median_background,
    detect_cage_crop_plan,
)
from preprocess.config import (
    CageDetectConfig,
    CanonicalResolutionConfig,
    PreCropConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.crop_plan import CropMode, compute_canonical_geometry
from preprocess.exceptions import CageDetectionError, CropPlanError
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.pre_crop import (
    build_pre_crop_translation_homography,
    resolve_pre_crop,
)


def _write_synthetic_video(
    path: Path,
    *,
    size_wh: tuple[int, int] = (200, 120),
    cage_rectangle: tuple[tuple[int, int], tuple[int, int]] | None = (
        (30, 20),
        (170, 100),
    ),
    cage_intensity: int = 200,
    frame_count: int = 3,
) -> Path:
    width, height = size_wh
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        size_wh,
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI test fixture.")
    try:
        for _index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if cage_rectangle is not None:
                cv2.rectangle(
                    frame,
                    cage_rectangle[0],
                    cage_rectangle[1],
                    (cage_intensity, cage_intensity, cage_intensity),
                    thickness=-1,
                )
            writer.write(frame)
    finally:
        writer.release()
    return path


def _detector_config(
    *,
    threshold: int = 100,
    canonical_size_wh: tuple[int, int] = (200, 120),
    canonical_enabled: bool = True,
    pad_px: int = 2,
) -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            roi_margin_px=0,
            canonical_resolution=CanonicalResolutionConfig(
                enabled=canonical_enabled,
                width=canonical_size_wh[0],
                height=canonical_size_wh[1],
            ),
        ),
        cage_detect=CageDetectConfig(
            sample_step=1,
            pad_px=pad_px,
            threshold=threshold,
            pre_crop_expansion_percent=5.0,
            dilate_kernel_size=5,
            erode_kernel_size=3,
            rim_close_kernel_size=5,
            minimum_cage_width_fraction=0.4,
            minimum_cage_height_fraction=0.4,
            minimum_contour_area=50.0,
            fit_tolerance_px=3,
        ),
    )


def _detect_full_frame(
    video_path: Path,
    config: PreprocessConfig,
) -> CageDetectionResult:
    pre_crop = resolve_pre_crop(config.pre_crop, raw_size_wh=(200, 120))
    return detect_cage_crop_plan(video_path, config, pre_crop)


def test_successful_automatic_detection_returns_unaccepted_crop_plan(
    tmp_path: Path,
) -> None:
    video_path = _write_synthetic_video(tmp_path / "cage.avi")
    config = _detector_config()

    result = _detect_full_frame(video_path, config)

    assert result.crop_plan.mode is CropMode.AUTOMATIC
    assert result.crop_plan.accepted_by_user is False
    assert result.crop_plan.fit_score is not None
    assert result.crop_plan.rim_density is not None
    assert all(dimension > 0 and dimension % 2 == 0 for dimension in result.crop_plan.prepared_size_wh)


def test_canonical_geometry_is_honored(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "cage.avi")
    config = _detector_config(canonical_size_wh=(240, 160))

    result = _detect_full_frame(video_path, config)

    assert result.crop_plan.prepared_size_wh == (240, 160)
    assert result.cropped_background.shape == (160, 240)
    canonical = result.detector_diagnostics["canonical_scaling_info"]
    assert isinstance(canonical, dict)
    assert canonical["enabled"] is True
    assert canonical["canonical_size_wh"] == [240, 160]


def test_auto_and_manual_crop_plans_share_final_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quad = np.array(
        [
            [20.0, 20.0],
            [120.1, 20.0],
            [120.1, 70.1],
            [20.0, 70.1],
        ],
        dtype=np.float64,
    )
    background = np.zeros((120, 200), dtype=np.uint8)
    detected = cage_detection._DetectedRectangle(
        rectangle_in_search=((70.05, 45.05), (100.1, 50.1), 0.0),
        quad_in_pre_crop=quad,
        fit_score=1.0,
        rim_density=0.1,
        contour_area=5015.01,
        coarse_search_roi=(0, 0, 200, 120),
    )
    monkeypatch.setattr(
        cage_detection,
        "_build_median_background_with_count",
        lambda _path, _step: cage_detection._MedianBackground(
            image=background,
            sampled_frame_count=1,
        ),
    )
    monkeypatch.setattr(
        cage_detection,
        "_detect_rotated_rectangle",
        lambda _background, _config, _margin: detected,
    )

    canonical_size_wh = (202, 120)
    native_config = _detector_config(
        canonical_size_wh=canonical_size_wh,
        canonical_enabled=False,
        pad_px=0,
    )
    canonical_config = _detector_config(
        canonical_size_wh=canonical_size_wh,
        canonical_enabled=True,
        pad_px=0,
    )
    pre_crop = resolve_pre_crop(native_config.pre_crop, raw_size_wh=(200, 120))
    automatic_native = detect_cage_crop_plan(
        Path("unused.avi"),
        native_config,
        pre_crop,
    )
    automatic_canonical = detect_cage_crop_plan(
        Path("unused.avi"),
        canonical_config,
        pre_crop,
    )
    manual_native = make_manual_crop_plan(
        raw_frame_shape=(120, 200),
        points_tl_tr_br_bl=quad,
        pre_crop_roi=None,
        canonical_resolution=native_config.prepare.canonical_resolution,
    )
    manual_canonical = make_manual_crop_plan(
        raw_frame_shape=(120, 200),
        points_tl_tr_br_bl=quad,
        pre_crop_roi=None,
        canonical_resolution=canonical_config.prepare.canonical_resolution,
    )

    assert automatic_native.crop_plan.prepared_size_wh == (102, 52)
    assert manual_native.prepared_size_wh == (102, 52)
    assert automatic_canonical.crop_plan.prepared_size_wh == canonical_size_wh
    assert manual_canonical.prepared_size_wh == canonical_size_wh
    np.testing.assert_allclose(
        automatic_native.crop_plan.H_raw_to_prepared_3x3,
        manual_native.H_raw_to_prepared_3x3,
    )
    np.testing.assert_allclose(
        automatic_canonical.crop_plan.H_raw_to_prepared_3x3,
        manual_canonical.H_raw_to_prepared_3x3,
    )
    np.testing.assert_allclose(
        automatic_canonical.crop_plan.H_prepared_to_raw_3x3,
        manual_canonical.H_prepared_to_raw_3x3,
    )

    expected_geometry = compute_canonical_geometry(102, 52, 202, 120)
    assert expected_geometry.scaled_size_wh == (202, 103)
    assert (
        expected_geometry.padding_left,
        expected_geometry.padding_right,
        expected_geometry.padding_top,
        expected_geometry.padding_bottom,
    ) == (0, 0, 8, 9)
    expected_canonical_transform = np.array(
        [
            [expected_geometry.uniform_scale, 0.0, 0.0],
            [0.0, expected_geometry.uniform_scale, 8.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        automatic_canonical.crop_plan.H_raw_to_prepared_3x3
        @ automatic_native.crop_plan.H_prepared_to_raw_3x3,
        expected_canonical_transform,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        manual_canonical.H_raw_to_prepared_3x3
        @ manual_native.H_prepared_to_raw_3x3,
        expected_canonical_transform,
        atol=1e-12,
    )

    disabled_geometry = automatic_native.detector_diagnostics[
        "canonical_scaling_info"
    ]
    assert isinstance(disabled_geometry, dict)
    assert disabled_geometry["uniform_scale"] == 1.0
    assert disabled_geometry["scaled_size_wh"] == [102, 52]
    assert (
        disabled_geometry["padding_left"],
        disabled_geometry["padding_right"],
        disabled_geometry["padding_top"],
        disabled_geometry["padding_bottom"],
    ) == (0, 0, 0, 0)


def test_pre_crop_is_carried_and_composed_into_raw_homography(
    tmp_path: Path,
) -> None:
    video_path = _write_synthetic_video(
        tmp_path / "offset-cage.avi",
        cage_rectangle=((80, 20), (180, 100)),
    )
    config = _detector_config()
    pre_crop = resolve_pre_crop(
        PreCropConfig(
            enabled=True,
            mode="manual_rectangle",
            manual_rectangle=(50, 0, 150, 120),
        ),
        raw_size_wh=(200, 120),
    )

    result = detect_cage_crop_plan(video_path, config, pre_crop)

    assert result.crop_plan.pre_crop_roi == pre_crop.roi
    assert np.all(result.crop_plan.quad_raw_tl_tr_br_bl[:, 0] >= pre_crop.roi.x)
    local_transform = np.asarray(
        result.detector_diagnostics["H_pre_crop_to_prepared_3x3"]
    )
    expected_raw_transform = (
        local_transform @ build_pre_crop_translation_homography(pre_crop)
    )
    np.testing.assert_allclose(
        result.crop_plan.H_raw_to_prepared_3x3,
        expected_raw_transform,
    )


def test_detector_configuration_changes_detection_behavior(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(
        tmp_path / "dim-cage.avi",
        cage_intensity=180,
    )

    successful = _detect_full_frame(video_path, _detector_config(threshold=100))

    assert successful.crop_plan.mode is CropMode.AUTOMATIC
    with pytest.raises(CageDetectionError, match="coarse cage region"):
        _detect_full_frame(video_path, _detector_config(threshold=250))


def test_no_sampled_frames_raises_cage_detection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cage_detection,
        "_read_sampled_frames",
        lambda _path, _step: [],
    )

    with pytest.raises(CageDetectionError, match="No usable frames"):
        build_median_background(Path("unused.avi"), sample_step=1)


def test_no_valid_cage_contour_raises_cage_detection_error(
    tmp_path: Path,
) -> None:
    video_path = _write_synthetic_video(
        tmp_path / "empty.avi",
        cage_rectangle=None,
    )

    with pytest.raises(CageDetectionError, match="coarse cage region"):
        _detect_full_frame(video_path, _detector_config())


def test_invalid_generated_crop_plan_becomes_detection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_synthetic_video(tmp_path / "cage.avi")

    def reject_crop_plan(**_values: object) -> NoReturn:
        raise CropPlanError("forced invalid plan")

    monkeypatch.setattr(cage_detection, "CropPlan", reject_crop_plan)

    with pytest.raises(CageDetectionError, match="invalid CropPlan"):
        _detect_full_frame(video_path, _detector_config())


def test_diagnostics_contain_required_detector_fields(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "cage.avi")

    diagnostics = _detect_full_frame(
        video_path,
        _detector_config(),
    ).detector_diagnostics

    required_fields = {
        "sampled_frame_count",
        "background_shape",
        "pre_crop_roi",
        "detected_rotated_rectangle",
        "detected_quad",
        "fit_score",
        "rim_density",
        "output_size_before_canonical_scaling",
        "canonical_scaling_info",
        "rotation_applied",
    }
    assert required_fields <= diagnostics.keys()
    assert diagnostics["sampled_frame_count"] == 3
    assert diagnostics["background_shape"] == [120, 200]
