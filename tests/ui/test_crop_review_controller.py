from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess.cage_detection import CageDetectionResult
from preprocess.config import (
    CageDetectConfig,
    CanonicalResolutionConfig,
    EncodingConfig,
    FFmpegEncodingConfig,
    MaskConfig,
    PrepareConfig,
    PreprocessConfig,
    TrimConfig,
)
from preprocess.crop_plan import CropMode, CropPlan
from preprocess.exceptions import CageDetectionError, CropPlanError, VideoProbeError
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.masking import apply_static_mask_to_frame
from preprocess.models import OperationProgress, VideoProbeResult
from preprocess.pre_crop import PreCropMode, resolve_pre_crop
from project.models import Project
from ui.controllers.crop_review_controller import (
    CropReviewController,
    CropReviewValidationError,
    build_crop_preview,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.state import CropReviewMode, PreprocessSetupState, WorkflowStep
from ui.widgets.video_frame_view import fit_image_target_rect

FRAME = np.full((80, 100, 3), 120, dtype=np.uint8)
POINTS = ((10.0, 10.0), (90.0, 10.0), (90.0, 70.0), (10.0, 70.0))


@pytest.fixture
def qt_application(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip("A non-GUI Qt application already exists.")
    return app


def _config() -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            roi_margin_px=4,
            perspective_interpolation="linear",
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=100,
                height=80,
            ),
        )
    )


def _probe(path: Path) -> VideoProbeResult:
    return VideoProbeResult(
        source_path=path,
        width=100,
        height=80,
        frame_count_opencv_reported=20,
        frame_count_opencv_readable=20,
        opencv_fps=10.0,
        raw_fps_effective=10.0,
        raw_fps_effective_method="opencv_fps",
        pts_status="not_extracted",
        ffprobe_succeeded=False,
    )


def _detector_defaults_config() -> PreprocessConfig:
    return PreprocessConfig(
        cage_detect=CageDetectConfig(
            sample_step=123,
            pad_px=7,
            threshold=77,
            pre_crop_expansion_percent=22.5,
            dilate_kernel_size=15,
            erode_kernel_size=11,
            rim_close_kernel_size=17,
            minimum_cage_width_fraction=0.43,
            minimum_cage_height_fraction=0.44,
            minimum_contour_area=321.5,
            fit_tolerance_px=9,
        )
    )


def _modified_detector_config() -> PreprocessConfig:
    return PreprocessConfig(
        trim=TrimConfig(start_frame=5, end_frame=15),
        prepare=PrepareConfig(
            roi_margin_px=9,
            perspective_interpolation="linear",
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=120,
                height=82,
            ),
        ),
        cage_detect=CageDetectConfig(
            sample_step=456,
            pad_px=3,
            threshold=155,
            pre_crop_expansion_percent=18.0,
            dilate_kernel_size=19,
            erode_kernel_size=13,
            rim_close_kernel_size=21,
            minimum_cage_width_fraction=0.52,
            minimum_cage_height_fraction=0.53,
            minimum_contour_area=654.5,
            fit_tolerance_px=12,
        ),
        encoding=EncodingConfig(
            ffmpeg=FFmpegEncodingConfig(
                container="mp4",
                codec="libx264",
                pixel_format="yuv420p",
                preset="slow",
                crf=23,
                bframes=0,
                faststart=True,
            )
        ),
    )


def _state(
    tmp_path: Path,
    *,
    start_frame: int = 5,
    config: PreprocessConfig | None = None,
    default_config: PreprocessConfig | None = None,
) -> PreprocessSetupState:
    raw_path = tmp_path / "raw.avi"
    config = config or _config()
    default_config = default_config or PreprocessConfig()
    pre_crop = resolve_pre_crop(config.pre_crop, (100, 80))
    return PreprocessSetupState(
        project=Project(root_dir=tmp_path, name="Project"),
        project_dir=tmp_path,
        preprocess_dir=tmp_path / "preprocess",
        raw_video_path=raw_path,
        raw_probe=_probe(raw_path),
        preprocess_config=config,
        original_preprocess_config=PreprocessConfig.model_validate(
            config.model_dump(mode="python")
        ),
        default_preprocess_config=PreprocessConfig.model_validate(
            default_config.model_dump(mode="python")
        ),
        start_frame=start_frame,
        end_frame_exclusive=config.trim.end_frame,
        pre_crop_config=config.pre_crop,
        resolved_pre_crop=pre_crop,
        trim_pre_crop_valid=True,
    )


def _automatic_plan(config: PreprocessConfig) -> CropPlan:
    manual = make_manual_crop_plan(
        raw_frame_shape=FRAME.shape[:2],
        points_tl_tr_br_bl=np.asarray(POINTS),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )
    metadata = manual.to_metadata_dict()
    metadata.update(
        mode=CropMode.AUTOMATIC.value,
        fit_score=0.91,
        rim_density=0.13,
        accepted_by_user=False,
    )
    return CropPlan.model_validate(metadata)


def _detector(_path: Path, config: PreprocessConfig, _pre_crop) -> CageDetectionResult:
    plan = _automatic_plan(config)
    return CageDetectionResult(
        crop_plan=plan,
        cropped_background=np.zeros((80, 100), dtype=np.uint8),
        detector_diagnostics={"fit_score": 0.91, "rim_density": 0.13},
    )


def _controller(
    tmp_path: Path,
    *,
    frame_reader=None,
    detector=_detector,
    manual_plan_builder=make_manual_crop_plan,
    preview_builder=build_crop_preview,
) -> CropReviewController:
    return CropReviewController(
        _state(tmp_path),
        detector=detector,
        frame_reader=frame_reader or (lambda _path, _index: FRAME.copy()),
        manual_plan_builder=manual_plan_builder,
        preview_builder=preview_builder,
    )


def _select_manual_candidate(controller: CropReviewController) -> CropPlan:
    controller.load_representative_frame()
    controller.set_crop_mode(CropReviewMode.MANUAL)
    plan = None
    for point in POINTS:
        plan = controller.add_manual_point(*point)
    assert plan is not None
    return plan


def test_representative_frame_uses_start_then_frame_zero_fallback(tmp_path: Path) -> None:
    calls: list[int] = []

    def reader(_path: Path, index: int) -> np.ndarray:
        calls.append(index)
        if index == 5:
            raise VideoProbeError("unreadable")
        return FRAME.copy()

    controller = _controller(tmp_path, frame_reader=reader)

    loaded = controller.load_representative_frame()

    assert calls == [5, 0]
    assert controller.representative_frame_index == 0
    np.testing.assert_array_equal(loaded, FRAME)


def test_automatic_detection_stores_unaccepted_candidate(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    plan = controller.run_automatic_detection()

    assert plan.accepted_by_user is False
    assert controller.state.candidate_crop_plan is plan
    assert controller.state.accepted_crop_plan is None
    assert controller.prepared_preview is not None
    assert controller.state.detector_diagnostics == {
        "fit_score": 0.91,
        "rim_density": 0.13,
    }


def test_automatic_detection_reports_all_ordered_gui_phases(tmp_path: Path) -> None:
    core_phases = (
        "Loading representative frames",
        "Estimating background",
        "Detecting cage contour",
        "Building CropPlan",
    )

    def detector(
        path,
        config,
        pre_crop,
        *,
        progress_callback,
        cancellation_requested,
    ) -> CageDetectionResult:
        del cancellation_requested
        for phase in core_phases:
            progress_callback(
                OperationProgress(
                    phase=phase,
                    message=phase,
                    is_indeterminate=True,
                )
            )
        return _detector(path, config, pre_crop)

    controller = _controller(tmp_path, detector=detector)
    request = controller.begin_automatic_detection()
    phases: list[str] = []

    result = controller.compute_automatic_detection(
        request,
        progress_callback=lambda progress: phases.append(progress.phase),
        cancellation_requested=lambda: False,
    )

    assert result.crop_plan.mode is CropMode.AUTOMATIC
    assert phases == [*core_phases, "Preparing preview"]


def test_automatic_detection_failure_never_reports_preview_or_completion(
    tmp_path: Path,
) -> None:
    def detector(
        _path,
        _config,
        _pre_crop,
        *,
        progress_callback,
        cancellation_requested,
    ) -> CageDetectionResult:
        del cancellation_requested
        progress_callback(
            OperationProgress(
                phase="Loading representative frames",
                message="Loading representative frames",
                is_indeterminate=True,
            )
        )
        raise CageDetectionError("no cage")

    controller = _controller(tmp_path, detector=detector)
    request = controller.begin_automatic_detection()
    phases: list[str] = []

    with pytest.raises(CageDetectionError, match="no cage"):
        controller.compute_automatic_detection(
            request,
            progress_callback=lambda progress: phases.append(progress.phase),
            cancellation_requested=lambda: False,
        )

    assert phases == ["Loading representative frames"]
    assert "Preparing preview" not in phases
    assert "Complete" not in phases


def test_automatic_failure_clears_stale_candidate_and_acceptance(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.run_automatic_detection()
    controller.accept_crop()

    def fail(_path, _config, _pre_crop):
        raise CageDetectionError("no cage")

    controller._detector = fail

    with pytest.raises(CropReviewValidationError, match="no cage"):
        controller.run_automatic_detection()

    assert controller.state.candidate_crop_plan is None
    assert controller.state.accepted_crop_plan is None
    assert controller.prepared_preview is None


def test_detector_setting_change_uses_typed_copy_and_invalidates_acceptance(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    old_config = controller.state.preprocess_config
    controller.run_automatic_detection()
    controller.accept_crop()

    updated = controller.update_detector_settings(sample_step=17, threshold=111)

    assert isinstance(updated, PreprocessConfig)
    assert updated is not old_config
    assert updated.cage_detect.sample_step == 17
    assert updated.cage_detect.threshold == 111
    assert controller.state.accepted_crop_plan is None
    assert controller.state.candidate_crop_plan is None


def test_reset_detector_settings_restores_loaded_defaults_without_hardcoded_values(
    tmp_path: Path,
) -> None:
    default_config = _detector_defaults_config()
    current_config = _modified_detector_config()
    state = _state(tmp_path, config=current_config, default_config=default_config)
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )

    result = controller.reset_detector_settings_to_defaults()

    assert result.changed_fields == tuple(CageDetectConfig.model_fields)
    assert state.preprocess_config is not None
    assert state.preprocess_config.cage_detect == default_config.cage_detect
    assert state.preprocess_config.cage_detect.sample_step == 123
    assert state.preprocess_config.cage_detect.threshold == 77


def test_detector_settings_modified_fields_report_differences(
    tmp_path: Path,
) -> None:
    default_config = _detector_defaults_config()
    current_config = PreprocessConfig(
        cage_detect=CageDetectConfig(
            **{
                **default_config.cage_detect.model_dump(mode="python"),
                "sample_step": 999,
                "threshold": 44,
            }
        )
    )
    state = _state(tmp_path, config=current_config, default_config=default_config)
    controller = CropReviewController(state)

    assert controller.detector_settings_modified_fields() == (
        "sample_step",
        "threshold",
    )


def test_detector_defaults_indicator_updates_in_headless_page(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.crop_review_page import CropReviewPage

    default_config = _detector_defaults_config()
    current_config = PreprocessConfig(
        cage_detect=CageDetectConfig(
            **{
                **default_config.cage_detect.model_dump(mode="python"),
                "sample_step": 999,
            }
        )
    )
    setup_controller = PreprocessSetupController(
        state=_state(tmp_path, config=current_config, default_config=default_config)
    )
    page = CropReviewPage(setup_controller)

    assert "Modified from defaults" in page.detector_defaults_status.text()
    assert "sample step" in page.detector_defaults_status.text()

    page._reset_detector_settings_to_defaults()

    assert page.detector_defaults_status.text() == "Detector settings use defaults."
    assert setup_controller.state.preprocess_config is not None
    assert setup_controller.state.preprocess_config.cage_detect == (
        default_config.cage_detect
    )


def test_reset_detector_settings_at_defaults_is_no_op(tmp_path: Path) -> None:
    default_config = _detector_defaults_config()
    state = _state(tmp_path, config=default_config, default_config=default_config)
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    controller.run_automatic_detection()
    controller.accept_crop()
    candidate = state.candidate_crop_plan
    accepted = state.accepted_crop_plan
    previous_revision = state.crop_review_revision

    result = controller.reset_detector_settings_to_defaults()

    assert result.already_at_defaults is True
    assert state.candidate_crop_plan is candidate
    assert state.accepted_crop_plan is accepted
    assert state.crop_review_revision == previous_revision


def test_reset_detector_settings_invalidates_automatic_candidate_when_changed(
    tmp_path: Path,
) -> None:
    default_config = _detector_defaults_config()
    state = _state(
        tmp_path,
        config=_modified_detector_config(),
        default_config=default_config,
    )
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    controller.run_automatic_detection()

    result = controller.reset_detector_settings_to_defaults()

    assert result.invalidated_candidate is True
    assert state.candidate_crop_plan is None
    assert state.accepted_crop_plan is None
    assert controller.prepared_preview is None
    assert state.crop_review_status == (
        "Detector settings reset to defaults; rerun automatic detection."
    )


def test_reset_detector_settings_invalidates_accepted_automatic_crop_when_changed(
    tmp_path: Path,
) -> None:
    default_config = _detector_defaults_config()
    state = _state(
        tmp_path,
        config=_modified_detector_config(),
        default_config=default_config,
    )
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    controller.run_automatic_detection()
    controller.accept_crop()

    result = controller.reset_detector_settings_to_defaults()

    assert result.invalidated_candidate is True
    assert result.invalidated_accepted is True
    assert state.candidate_crop_plan is None
    assert state.accepted_crop_plan is None


def test_reset_detector_settings_preserves_accepted_manual_crop(
    tmp_path: Path,
) -> None:
    default_config = _detector_defaults_config()
    state = _state(
        tmp_path,
        config=_modified_detector_config(),
        default_config=default_config,
    )
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    _select_manual_candidate(controller)
    accepted = controller.accept_crop()
    candidate = state.candidate_crop_plan

    result = controller.reset_detector_settings_to_defaults()

    assert result.preserved_manual_acceptance is True
    assert state.candidate_crop_plan is candidate
    assert state.accepted_crop_plan is accepted
    assert state.accepted_crop_plan.mode is CropMode.MANUAL


def test_reset_detector_settings_preserves_trim_pre_crop_timing_and_encoding(
    tmp_path: Path,
) -> None:
    default_config = _detector_defaults_config()
    current_config = _modified_detector_config()
    state = _state(tmp_path, config=current_config, default_config=default_config)
    before_trim = (state.start_frame, state.end_frame_exclusive)
    before_pre_crop = (state.pre_crop_config, state.resolved_pre_crop)
    before_timing = current_config.timing
    before_prepare = current_config.prepare
    before_encoding = current_config.encoding
    controller = CropReviewController(state)

    controller.reset_detector_settings_to_defaults()

    assert (state.start_frame, state.end_frame_exclusive) == before_trim
    assert (state.pre_crop_config, state.resolved_pre_crop) == before_pre_crop
    assert state.preprocess_config is not None
    assert state.preprocess_config.timing == before_timing
    assert state.preprocess_config.prepare == before_prepare
    assert state.preprocess_config.encoding == before_encoding


def test_static_mask_preview_uses_shared_helper_and_prepared_coordinates(
    tmp_path: Path,
) -> None:
    config = PreprocessConfig.model_validate(
        {
            **_config().model_dump(mode="python"),
            "mask": MaskConfig(
                enabled=True,
                shapes=[
                    {"type": "rectangle", "x": 0, "y": 0, "width": 6, "height": 4}
                ],
            ),
        }
    )
    state = _state(tmp_path, config=config)
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )

    candidate = controller.run_automatic_detection()
    preview = controller.prepared_preview
    unmasked = build_crop_preview(FRAME, candidate, config.prepare.perspective_interpolation)
    expected = apply_static_mask_to_frame(unmasked, config.mask)

    assert preview is not None
    np.testing.assert_array_equal(preview, expected)
    assert np.all(preview[:4, :6] == 0)


def test_static_mask_change_invalidates_run_but_preserves_crop_trim_precrop_and_timing(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    controller.run_automatic_detection()
    accepted = controller.accept_crop()
    candidate = state.candidate_crop_plan
    trim = (state.start_frame, state.end_frame_exclusive)
    pre_crop = (state.pre_crop_config, state.resolved_pre_crop)
    timing = (
        state.timing_mode,
        state.timing_status,
        state.external_time_selection,
        state.external_time_vector_seconds,
    )
    state.run_status = "Completed"

    changed = controller.add_static_mask_rectangle((1, 1, 4, 4))

    assert changed == 0
    assert state.candidate_crop_plan is candidate
    assert state.accepted_crop_plan is accepted
    assert state.accepted_crop_plan is not None
    assert state.accepted_crop_plan.accepted_by_user is True
    assert (state.start_frame, state.end_frame_exclusive) == trim
    assert (state.pre_crop_config, state.resolved_pre_crop) == pre_crop
    assert (
        state.timing_mode,
        state.timing_status,
        state.external_time_selection,
        state.external_time_vector_seconds,
    ) == timing
    assert state.run_status == "Configuration changed"


def test_no_op_static_mask_edit_does_not_invalidate_run_readiness(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    controller.run_automatic_detection()
    controller.add_static_mask_rectangle((1, 1, 4, 4))
    assert state.preprocess_config is not None
    shape = state.preprocess_config.mask.shapes[0]
    state.run_status = "Completed"

    changed = controller.replace_static_mask_shape(0, shape)

    assert changed is False
    assert state.run_status == "Completed"


def test_invalid_static_mask_rejected_in_crop_review_before_storage(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    controller.run_automatic_detection()

    with pytest.raises(CropReviewValidationError, match="Invalid static mask"):
        controller.add_static_mask_rectangle((99, 0, 2, 2))

    assert state.preprocess_config is not None
    assert state.preprocess_config.mask.shapes == []


def test_manual_collection_preserves_prescribed_order_and_rejects_wrong_order(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.load_representative_frame()
    controller.set_crop_mode(CropReviewMode.MANUAL)
    wrong_order = (POINTS[0], POINTS[3], POINTS[2], POINTS[1])

    for point in wrong_order[:3]:
        assert controller.add_manual_point(*point) is None
    with pytest.raises(CropReviewValidationError, match="top-left"):
        controller.add_manual_point(*wrong_order[3])

    assert controller.manual_points == wrong_order
    assert controller.state.candidate_crop_plan is None


def test_manual_crop_delegates_plan_construction_to_core_helper(tmp_path: Path) -> None:
    calls: list[np.ndarray] = []
    expected = make_manual_crop_plan(
        FRAME.shape[:2],
        np.asarray(POINTS),
        None,
        _config().prepare.canonical_resolution,
    )

    def builder(**kwargs) -> CropPlan:
        calls.append(np.array(kwargs["points_tl_tr_br_bl"], copy=True))
        return expected

    controller = _controller(tmp_path, manual_plan_builder=builder)

    plan = _select_manual_candidate(controller)

    assert plan.accepted_by_user is False
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.asarray(POINTS))


def test_invalid_manual_crop_does_not_create_candidate(tmp_path: Path) -> None:
    def invalid_builder(**_kwargs):
        raise CropPlanError("degenerate")

    controller = _controller(tmp_path, manual_plan_builder=invalid_builder)
    controller.load_representative_frame()
    controller.set_crop_mode(CropReviewMode.MANUAL)

    for point in POINTS[:3]:
        controller.add_manual_point(*point)
    with pytest.raises(CropReviewValidationError, match="degenerate"):
        controller.add_manual_point(*POINTS[3])

    assert controller.state.candidate_crop_plan is None
    assert controller.state.accepted_crop_plan is None


def test_candidate_cannot_advance_until_explicit_acceptance(tmp_path: Path) -> None:
    controller = _controller(tmp_path)

    controller.run_automatic_detection()

    assert controller.can_advance() is False


def test_accepting_valid_candidate_creates_distinct_accepted_plan(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    candidate = controller.run_automatic_detection()

    accepted = controller.accept_crop()

    assert accepted is controller.state.accepted_crop_plan
    assert accepted is not candidate
    assert accepted.accepted_by_user is True
    assert candidate.accepted_by_user is False
    assert controller.state.crop_review_status == (
        "Crop accepted. Ready for encode settings."
    )


def test_acceptance_requires_successful_matching_preview(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.candidate_crop_plan = _automatic_plan(state.preprocess_config)
    controller = CropReviewController(state)

    with pytest.raises(CropReviewValidationError, match="matching prepared"):
        controller.accept_crop()

    assert state.accepted_crop_plan is None


def test_editing_manual_points_clears_acceptance(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _select_manual_candidate(controller)
    controller.accept_crop()

    controller.clear_manual_points()

    assert controller.state.accepted_crop_plan is None
    assert controller.state.candidate_crop_plan is None
    assert controller.manual_points == ()


def test_editing_pre_crop_clears_acceptance(tmp_path: Path) -> None:
    state = _state(tmp_path)
    crop_controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    crop_controller.run_automatic_detection()
    crop_controller.accept_crop()
    setup = PreprocessSetupController(state=state)

    setup.configure_trim_and_pre_crop(
        start_frame=5,
        end_frame_exclusive=None,
        mode=PreCropMode.VERTICAL_KEEP_LEFT,
        boundary_px=95,
    )

    assert state.accepted_crop_plan is None
    assert state.candidate_crop_plan is None


def test_editing_trim_only_preserves_crop_review_acceptance(tmp_path: Path) -> None:
    state = _state(tmp_path)
    crop_controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    crop_controller.run_automatic_detection()
    crop_controller.accept_crop()
    setup = PreprocessSetupController(state=state)
    accepted = state.accepted_crop_plan
    candidate = state.candidate_crop_plan
    previous_revision = state.crop_review_revision

    setup.configure_trim_and_pre_crop(
        start_frame=7,
        end_frame_exclusive=None,
        mode=PreCropMode.NONE,
    )
    crop_controller.synchronize_upstream_context()

    assert state.accepted_crop_plan is accepted
    assert state.candidate_crop_plan is candidate
    assert state.crop_review_revision == previous_revision


def test_accepted_crop_enables_crop_review_next(tmp_path: Path) -> None:
    state = _state(tmp_path)
    crop_controller = CropReviewController(
        state,
        detector=_detector,
        frame_reader=lambda _path, _index: FRAME.copy(),
    )
    setup = PreprocessSetupController(state=state)
    crop_controller.run_automatic_detection()

    assert setup.can_advance(WorkflowStep.CROP_REVIEW) is False

    crop_controller.accept_crop()

    assert setup.can_advance(WorkflowStep.CROP_REVIEW) is True


def test_preview_uses_crop_plan_transform_and_output_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del tmp_path
    config = _config()
    plan = _automatic_plan(config)
    captured: dict[str, object] = {}

    def fake_warp(frame, transform, size, **kwargs):
        captured.update(frame=frame, transform=transform, size=size, kwargs=kwargs)
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, "warpPerspective", fake_warp)

    preview = build_crop_preview(FRAME, plan, "linear")

    assert captured["size"] == plan.prepared_size_wh
    np.testing.assert_array_equal(captured["transform"], plan.H_raw_to_prepared_3x3)
    assert preview.shape[:2] == (plan.prepared_size_wh[1], plan.prepared_size_wh[0])


def test_preview_blacks_pixels_outside_transformed_crop_footprint() -> None:
    config = _config()
    plan = _automatic_plan(config)
    preview = build_crop_preview(np.full_like(FRAME, 180), plan, "linear")
    canonical = plan.canonical_geometry

    assert canonical is not None
    assert canonical.padding_top > 0
    assert canonical.padding_bottom > 0
    assert np.count_nonzero(preview[: canonical.padding_top]) == 0
    assert np.count_nonzero(preview[-canonical.padding_bottom :]) == 0
    assert float(preview[canonical.padding_top + 3 : -canonical.padding_bottom - 3].mean()) > 100


def test_viewer_letterboxing_is_display_only_and_not_preview_pixels() -> None:
    preview_size_wh = (100, 80)

    target = fit_image_target_rect(preview_size_wh, (500, 300))

    assert target.width() / target.height() == pytest.approx(100 / 80)
    assert target.left() > 0
    assert target.right() < 500
    assert target.top() == pytest.approx(4.0)
    assert target.bottom() == pytest.approx(296.0)
    assert preview_size_wh == (100, 80)
