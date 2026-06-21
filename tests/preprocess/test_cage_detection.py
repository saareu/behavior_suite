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
from preprocess.crop_plan import CropMode
from preprocess.exceptions import CageDetectionError, CropPlanError
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
) -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            roi_margin_px=0,
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=canonical_size_wh[0],
                height=canonical_size_wh[1],
            ),
        ),
        cage_detect=CageDetectConfig(
            sample_step=1,
            pad_px=2,
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
