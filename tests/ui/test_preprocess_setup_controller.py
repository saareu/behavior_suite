import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from preprocess.config import PreprocessConfig, TrimConfig
from preprocess.models import (
    OperationProgress,
    RawReadableCountProvenance,
    TimingPlausibilityAssessment,
    TimingUnit,
    VideoProbeResult,
)
from preprocess.pre_crop import PreCropMode, PreCropROI
from project.service import ProjectService
from ui.controllers.preprocess_setup_controller import (
    SUPPORTED_RAW_FRAME_JUMP_STEPS,
    PreprocessSetupController,
    RawFrameReadResult,
    SetupValidationError,
)
from ui.state import RawReadableCountStatus, WorkflowStep
from ui.tasks import GuiTaskKind, GuiTaskProgressTracker


def _probe_result(path: Path, *, readable_count: int | None = None) -> VideoProbeResult:
    return VideoProbeResult(
        source_path=path.resolve(),
        width=100,
        height=80,
        codec="mjpeg",
        pixel_format="yuvj420p",
        duration_sec=2.0,
        avg_frame_rate="10/1",
        r_frame_rate="10/1",
        time_base="1/10",
        frame_count_ffprobe=20,
        frame_count_opencv_reported=20,
        frame_count_opencv_readable=readable_count,
        opencv_fps=10.0,
        raw_fps_effective=10.0,
        raw_fps_effective_method="ffprobe_avg_frame_rate",
        pts_status="not_extracted",
        ffprobe_succeeded=True,
    )


def _write_tiny_video(path: Path, frame_count: int = 4) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (100, 80),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI test fixture.")
    try:
        for index in range(frame_count):
            writer.write(np.full((80, 100, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _controller_with_probe(
    calls: list[bool] | None = None,
    raw_frame_reader=None,
) -> PreprocessSetupController:
    def fake_probe(
        path: Path,
        *,
        require_sequential_count: bool,
    ) -> VideoProbeResult:
        if calls is not None:
            calls.append(require_sequential_count)
        return _probe_result(
            Path(path),
            readable_count=20 if require_sequential_count else None,
        )

    kwargs = {"probe_function": fake_probe}
    if raw_frame_reader is not None:
        kwargs["raw_frame_reader"] = raw_frame_reader
    controller = PreprocessSetupController(**kwargs)
    controller.load_startup_config()
    return controller


def _ready_controller(
    tmp_path: Path,
    *,
    calls: list[bool] | None = None,
) -> PreprocessSetupController:
    controller = _controller_with_probe(calls)
    controller.create_project(tmp_path, "SetupProject")
    controller.probe_raw_video(tmp_path / "raw.avi")
    return controller


def _ready_controller_with_trusted_count(tmp_path: Path) -> PreprocessSetupController:
    controller = _controller_with_probe()
    controller.create_project(tmp_path, "TrustedCountProject")
    controller.probe_raw_video(tmp_path / "raw.avi", require_sequential_count=True)
    return controller


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


def _write_prepare_metadata(
    project_dir: Path,
    raw_path: Path,
    *,
    readable_count: int | None,
    start_frame: int = 3,
) -> Path:
    metadata = {
        "schema_version": "prepare_meta_v1",
        "raw_video": {
            "source_path": str(raw_path),
            "width": 100,
            "height": 80,
            "codec": "mjpeg",
            "pixel_format": "yuvj420p",
            "duration_sec": 2.0,
            "avg_frame_rate": "10/1",
            "r_frame_rate": "10/1",
            "time_base": "1/10",
            "frame_count_ffprobe": 20,
            "frame_count_opencv_reported": 20,
            "frame_count_opencv_readable": readable_count,
            "fps_effective": 10.0,
            "fps_effective_method": "ffprobe_avg_frame_rate",
            "pts_status": "not_extracted",
        },
        "trim": {
            "start_frame": start_frame,
            "end_frame_exclusive": 18,
            "end_frame_semantics": "exclusive",
            "expected_trimmed_frame_count": 15,
        },
        "pre_crop": {
            "enabled": False,
            "mode": "none",
            "roi": {"x": 0, "y": 0, "width": 100, "height": 80},
            "raw_size_wh": [100, 80],
        },
        "validation": {"status": "passed"},
    }
    path = project_dir / "preprocess" / "prepare_meta.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_create_project_updates_gui_state(tmp_path: Path) -> None:
    controller = _controller_with_probe()

    project = controller.create_project(tmp_path, "NewProject")

    assert controller.state.project == project
    assert controller.state.project_dir == tmp_path / "NewProject"
    assert controller.state.preprocess_dir == tmp_path / "NewProject" / "preprocess"
    assert controller.can_advance(WorkflowStep.PROJECT) is True


def test_open_project_updates_gui_state(tmp_path: Path) -> None:
    project = ProjectService().create_project(tmp_path, "ExistingProject")
    controller = _controller_with_probe()

    opened = controller.open_project(project.root_dir)

    assert opened == project
    assert controller.state.project_dir == project.root_dir
    assert controller.state.preprocess_dir == project.root_dir / "preprocess"


def test_open_completed_project_hydrates_raw_path_trim_and_pre_crop(
    tmp_path: Path,
) -> None:
    project = ProjectService().create_project(tmp_path, "HydratedProject")
    raw_path = tmp_path / "recorded_raw.avi"
    _write_prepare_metadata(project.root_dir, raw_path, readable_count=20)
    controller = _controller_with_probe()

    controller.open_project(project.root_dir)

    assert controller.state.raw_video_path == raw_path
    assert controller.state.start_frame == 3
    assert controller.state.end_frame_exclusive == 18
    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.mode == "none"
    assert controller.state.resolved_pre_crop is not None
    assert controller.state.accepted_crop_plan is None
    assert controller.state.prior_run_status == "passed"


def test_open_completed_project_restores_prior_run_raw_count_provenance(
    tmp_path: Path,
) -> None:
    project = ProjectService().create_project(tmp_path, "CountedProject")
    raw_path = _write_tiny_video(tmp_path / "recorded_raw.avi")
    counting_controller = PreprocessSetupController()
    counting_controller.load_startup_config()
    counting_controller.open_project(project.root_dir)
    counting_controller.probe_raw_video(raw_path, require_sequential_count=True)
    _write_prepare_metadata(project.root_dir, raw_path, readable_count=4)
    controller = _controller_with_probe()

    controller.open_project(project.root_dir)

    assert controller.state.raw_probe is not None
    assert controller.state.raw_probe.frame_count_opencv_readable == 4
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.RECORDED_PRIOR_RUN
    )


def test_completed_metadata_count_without_verified_fingerprint_is_not_restored(
    tmp_path: Path,
) -> None:
    project = ProjectService().create_project(tmp_path, "UnverifiedCountProject")
    raw_path = tmp_path / "recorded_raw.avi"
    _write_prepare_metadata(project.root_dir, raw_path, readable_count=20)
    controller = _controller_with_probe()

    controller.open_project(project.root_dir)

    assert controller.state.raw_probe is not None
    assert controller.state.raw_probe.frame_count_opencv_readable is None
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.NOT_COUNTED
    )


def test_reopened_gui_restores_matching_cache_with_distinct_provenance(
    tmp_path: Path,
) -> None:
    project = ProjectService().create_project(tmp_path, "CachedProbeProject")
    raw_path = _write_tiny_video(tmp_path / "cached_raw.avi")
    counting_controller = PreprocessSetupController()
    counting_controller.load_startup_config()
    counting_controller.open_project(project.root_dir)
    counting_controller.probe_raw_video(raw_path, require_sequential_count=True)
    reopened = PreprocessSetupController()
    reopened.load_startup_config()

    reopened.open_project(project.root_dir)

    assert reopened.state.raw_video_path == raw_path.resolve()
    assert reopened.state.raw_probe is not None
    assert reopened.state.raw_probe.frame_count_opencv_readable == 4
    assert reopened.state.raw_probe.raw_readable_count_provenance is (
        RawReadableCountProvenance.VALIDATED_REUSABLE_CACHE
    )
    assert reopened.state.raw_readable_count_status is (
        RawReadableCountStatus.VALIDATED_REUSABLE_CACHE
    )


def test_open_no_ttl_project_keeps_raw_sequential_count_not_measured(
    tmp_path: Path,
) -> None:
    project = ProjectService().create_project(tmp_path, "NoTtlProject")
    raw_path = tmp_path / "recorded_raw.avi"
    _write_prepare_metadata(project.root_dir, raw_path, readable_count=None)
    controller = _controller_with_probe()

    controller.open_project(project.root_dir)

    assert controller.state.raw_probe is not None
    assert controller.state.raw_probe.frame_count_opencv_readable is None
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.NOT_COUNTED
    )


@pytest.mark.parametrize("metadata_contents", [None, "{not-json"])
def test_missing_or_malformed_project_metadata_is_nonfatal(
    tmp_path: Path,
    metadata_contents: str | None,
) -> None:
    project = ProjectService().create_project(tmp_path, "NonfatalMetadataProject")
    if metadata_contents is not None:
        (project.root_dir / "preprocess" / "prepare_meta.json").write_text(
            metadata_contents,
            encoding="utf-8",
        )
    controller = _controller_with_probe()

    opened = controller.open_project(project.root_dir)

    assert opened == project
    assert controller.state.project is not None
    assert controller.state.project_hydration_status in {
        "metadata_absent",
        "metadata_invalid",
    }
    assert controller.state.project_hydration_message


def test_invalid_project_is_rejected(tmp_path: Path) -> None:
    controller = _controller_with_probe()

    with pytest.raises(SetupValidationError, match="Could not open project"):
        controller.open_project(tmp_path / "missing")

    assert controller.state.project is None
    assert controller.state.last_validation_error is not None


def test_valid_config_load_updates_gui_state(tmp_path: Path) -> None:
    config = PreprocessConfig(trim=TrimConfig(start_frame=3, end_frame=11))
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    controller = PreprocessSetupController()

    loaded = controller.load_config(config_path)

    assert loaded == config
    assert controller.state.preprocess_config == config
    assert controller.state.config_path == config_path.resolve()
    assert controller.state.start_frame == 3
    assert controller.state.end_frame_exclusive == 11


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("unknown_field: true\n", encoding="utf-8")
    controller = PreprocessSetupController()

    with pytest.raises(SetupValidationError, match="Invalid preprocess configuration"):
        controller.load_config(config_path)

    assert controller.state.preprocess_config is None


@pytest.mark.parametrize("require_sequential_count", [False, True])
def test_raw_video_probe_updates_state_and_forwards_count_choice(
    tmp_path: Path,
    require_sequential_count: bool,
) -> None:
    calls: list[bool] = []
    controller = _controller_with_probe(calls)
    controller.create_project(tmp_path, "ProbeProject")
    raw_path = tmp_path / "raw.avi"

    result = controller.probe_raw_video(
        raw_path,
        require_sequential_count=require_sequential_count,
    )

    assert calls == [require_sequential_count]
    assert controller.state.raw_video_path == raw_path.resolve()
    assert controller.state.raw_probe == result
    assert result.frame_count_opencv_readable == (20 if require_sequential_count else None)


def test_changing_raw_video_invalidates_same_session_readable_count(
    tmp_path: Path,
) -> None:
    controller = _controller_with_probe()
    controller.create_project(tmp_path, "CountInvalidationProject")
    first_path = tmp_path / "first.avi"
    controller.probe_raw_video(first_path, require_sequential_count=True)
    controller.state.timing_plausibility_assessment = TimingPlausibilityAssessment(
        selected_timing_units=TimingUnit.SECONDS,
        external_timing_estimated_fps=20.0,
        raw_video_nominal_fps=10.0,
        raw_video_nominal_fps_source="ffprobe_avg_frame_rate",
        symmetric_fps_mismatch_factor=2.0,
        warning_triggered=True,
    )

    controller.set_raw_video_path(tmp_path / "second.avi")

    assert controller.state.raw_probe is None
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.NOT_COUNTED
    )
    assert controller.state.timing_plausibility_assessment is None


def test_changing_raw_video_clears_cache_derived_count(tmp_path: Path) -> None:
    controller = _controller_with_probe()
    controller.create_project(tmp_path, "CachedCountInvalidationProject")
    first_path = tmp_path / "first.avi"
    controller.state.raw_video_path = first_path
    controller.state.raw_probe = _probe_result(
        first_path,
        readable_count=20,
    ).model_copy(
        update={
            "raw_readable_count_provenance": (
                RawReadableCountProvenance.VALIDATED_REUSABLE_CACHE
            )
        }
    )
    controller.state.raw_readable_count_status = (
        RawReadableCountStatus.VALIDATED_REUSABLE_CACHE
    )

    controller.set_raw_video_path(tmp_path / "second.avi")

    assert controller.state.raw_probe is None
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.NOT_COUNTED
    )


def test_probe_from_different_source_does_not_reuse_previous_count(
    tmp_path: Path,
) -> None:
    controller = _controller_with_probe()
    controller.create_project(tmp_path, "ProbeReplacementProject")
    controller.probe_raw_video(
        tmp_path / "first.avi", require_sequential_count=True
    )

    replacement = controller.apply_raw_video_probe(
        _probe_result(tmp_path / "second.avi", readable_count=None)
    )

    assert replacement.frame_count_opencv_readable is None
    assert controller.state.raw_video_path == (tmp_path / "second.avi").resolve()
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.NOT_COUNTED
    )


def test_same_source_metadata_reprobe_preserves_sequential_count(
    tmp_path: Path,
) -> None:
    controller = _controller_with_probe()
    controller.create_project(tmp_path, "SameSourceReprobeProject")
    raw_path = tmp_path / "raw.avi"
    controller.probe_raw_video(raw_path, require_sequential_count=True)

    replacement = controller.apply_raw_video_probe(
        _probe_result(raw_path, readable_count=None)
    )

    assert replacement.frame_count_opencv_readable == 20
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.COUNTED_THIS_SESSION
    )


def test_changing_raw_video_cancels_task_and_discards_stale_result(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    old_path = controller.state.raw_video_path
    assert old_path is not None
    context = controller.begin_task(
        GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        raw_video_path=old_path,
    )
    stale_result = _probe_result(old_path, readable_count=20)

    new_path = tmp_path / "new_raw.avi"
    controller.set_raw_video_path(new_path)
    applied = controller.apply_raw_video_probe(stale_result, context=context)

    assert context.cancellation_requested is True
    assert applied is None
    assert controller.state.raw_video_path == new_path
    assert controller.state.raw_probe is None


def test_changing_project_cancels_task_and_discards_stale_result(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    old_path = controller.state.raw_video_path
    assert old_path is not None
    context = controller.begin_task(
        GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        raw_video_path=old_path,
    )
    stale_result = _probe_result(old_path, readable_count=20)
    replacement = ProjectService().create_project(tmp_path, "ReplacementProject")

    controller.open_project(replacement.root_dir)
    applied = controller.apply_raw_video_probe(stale_result, context=context)

    assert context.cancellation_requested is True
    assert applied is None
    assert controller.state.project == replacement
    assert controller.state.raw_probe is None


def test_current_generation_count_result_updates_shared_probe_immediately(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    raw_path = controller.state.raw_video_path
    assert raw_path is not None
    context = controller.begin_task(
        GuiTaskKind.RAW_SEQUENTIAL_COUNT,
        raw_video_path=raw_path,
    )

    applied = controller.apply_raw_video_probe(
        _probe_result(raw_path, readable_count=20),
        context=context,
    )

    assert applied is not None
    assert controller.state.raw_probe is applied
    assert controller.state.raw_probe.frame_count_opencv_readable == 20


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, None), (5, 5), (7, 4)],
)
def test_invalid_trim_is_rejected(
    tmp_path: Path,
    start: int,
    end: int | None,
) -> None:
    controller = _ready_controller(tmp_path)

    with pytest.raises(SetupValidationError, match="Invalid trim"):
        controller.configure_trim_and_pre_crop(
            start_frame=start,
            end_frame_exclusive=end,
            mode=PreCropMode.NONE,
        )

    assert controller.state.trim_pre_crop_valid is False
    assert controller.state.resolved_pre_crop is not None
    assert controller.can_advance(WorkflowStep.TRIM_PRE_CROP) is False


def test_valid_trim_is_stored(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)

    controller.configure_trim_and_pre_crop(
        start_frame=2,
        end_frame_exclusive=9,
        mode=PreCropMode.NONE,
    )

    assert controller.state.start_frame == 2
    assert controller.state.end_frame_exclusive == 9
    assert controller.state.trim_pre_crop_valid is True
    assert controller.can_advance(WorkflowStep.TRIM_PRE_CROP) is True


@pytest.mark.parametrize(
    ("mode", "arguments", "expected_roi"),
    [
        (PreCropMode.NONE, {}, PreCropROI(x=0, y=0, width=100, height=80)),
        (
            PreCropMode.VERTICAL_KEEP_LEFT,
            {"boundary_px": 40},
            PreCropROI(x=0, y=0, width=40, height=80),
        ),
        (
            PreCropMode.VERTICAL_KEEP_RIGHT,
            {"boundary_px": 40},
            PreCropROI(x=40, y=0, width=60, height=80),
        ),
        (
            PreCropMode.HORIZONTAL_KEEP_UPPER,
            {"boundary_px": 30},
            PreCropROI(x=0, y=0, width=100, height=30),
        ),
        (
            PreCropMode.HORIZONTAL_KEEP_LOWER,
            {"boundary_px": 30},
            PreCropROI(x=0, y=30, width=100, height=50),
        ),
        (
            PreCropMode.MANUAL_RECTANGLE,
            {
                "rectangle_x": 10,
                "rectangle_y": 5,
                "rectangle_width": 50,
                "rectangle_height": 40,
            },
            PreCropROI(x=10, y=5, width=50, height=40),
        ),
    ],
)
def test_all_pre_crop_modes_map_to_core_models(
    tmp_path: Path,
    mode: PreCropMode,
    arguments: dict[str, Any],
    expected_roi: PreCropROI,
) -> None:
    controller = _ready_controller(tmp_path)

    resolved = controller.configure_trim_and_pre_crop(
        start_frame=0,
        end_frame_exclusive=None,
        mode=mode,
        **arguments,
    )

    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.mode == mode.value
    assert resolved.roi == expected_roi
    assert controller.state.resolved_pre_crop == resolved


def test_invalid_directional_boundary_is_rejected(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)

    with pytest.raises(SetupValidationError, match="Invalid trim or pre-crop"):
        controller.configure_trim_and_pre_crop(
            start_frame=0,
            end_frame_exclusive=None,
            mode=PreCropMode.VERTICAL_KEEP_LEFT,
            boundary_px=101,
        )

    assert controller.state.trim_pre_crop_valid is False


def test_invalid_manual_rectangle_is_rejected(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)

    with pytest.raises(SetupValidationError, match="Invalid trim or pre-crop"):
        controller.configure_trim_and_pre_crop(
            start_frame=0,
            end_frame_exclusive=None,
            mode=PreCropMode.MANUAL_RECTANGLE,
            rectangle_x=90,
            rectangle_y=5,
            rectangle_width=20,
            rectangle_height=40,
        )

    assert controller.state.resolved_pre_crop is None


def test_resolved_pre_crop_roi_is_stored_in_state(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)

    resolved = controller.configure_trim_and_pre_crop(
        start_frame=1,
        end_frame_exclusive=10,
        mode=PreCropMode.HORIZONTAL_KEEP_LOWER,
        boundary_px=25,
    )

    assert controller.state.resolved_pre_crop is resolved
    assert controller.state.resolved_pre_crop.roi == PreCropROI(
        x=0,
        y=25,
        width=100,
        height=55,
    )


@pytest.mark.parametrize(
    ("mode", "arguments", "expected_config", "expected_roi"),
    [
        (
            PreCropMode.NONE,
            {},
            {"enabled": False, "mode": "none"},
            PreCropROI(x=0, y=0, width=100, height=80),
        ),
        (
            PreCropMode.VERTICAL_KEEP_LEFT,
            {"boundary_px": 40},
            {"enabled": True, "mode": "vertical_keep_left", "boundary_px": 40},
            PreCropROI(x=0, y=0, width=40, height=80),
        ),
        (
            PreCropMode.VERTICAL_KEEP_RIGHT,
            {"boundary_px": 40},
            {"enabled": True, "mode": "vertical_keep_right", "boundary_px": 40},
            PreCropROI(x=40, y=0, width=60, height=80),
        ),
        (
            PreCropMode.HORIZONTAL_KEEP_UPPER,
            {"boundary_px": 30},
            {"enabled": True, "mode": "horizontal_keep_upper", "boundary_px": 30},
            PreCropROI(x=0, y=0, width=100, height=30),
        ),
        (
            PreCropMode.HORIZONTAL_KEEP_LOWER,
            {"boundary_px": 30},
            {"enabled": True, "mode": "horizontal_keep_lower", "boundary_px": 30},
            PreCropROI(x=0, y=30, width=100, height=50),
        ),
        (
            PreCropMode.MANUAL_RECTANGLE,
            {"rectangle": (10, 5, 50, 40)},
            {
                "enabled": True,
                "mode": "manual_rectangle",
                "manual_rectangle": (10, 5, 50, 40),
            },
            PreCropROI(x=10, y=5, width=50, height=40),
        ),
    ],
)
def test_visual_pre_crop_controls_produce_existing_pre_crop_config(
    tmp_path: Path,
    qt_application: object,
    mode: PreCropMode,
    arguments: dict[str, Any],
    expected_config: dict[str, object],
    expected_roi: PreCropROI,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)

    page._apply_pre_crop_controls(mode=mode, **arguments)

    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.model_dump(mode="python", exclude_none=True) == (
        expected_config
    )
    assert controller.state.resolved_pre_crop is not None
    assert controller.state.resolved_pre_crop.roi == expected_roi


def test_visual_split_edit_updates_numeric_boundary(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)

    page._visual_pre_crop_changed(PreCropMode.VERTICAL_KEEP_LEFT.value, 42)

    assert page.boundary.value() == 42
    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.mode == "vertical_keep_left"
    assert controller.state.pre_crop_config.boundary_px == 42
    assert controller.state.resolved_pre_crop is not None
    assert controller.state.resolved_pre_crop.roi == PreCropROI(
        x=0,
        y=0,
        width=42,
        height=80,
    )


def test_visual_manual_rectangle_edit_updates_numeric_config(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)

    page._visual_pre_crop_changed(PreCropMode.MANUAL_RECTANGLE.value, (5, 6, 40, 20))

    assert page.rectangle_inputs["x"].value() == 5
    assert page.rectangle_inputs["y"].value() == 6
    assert page.rectangle_inputs["width"].value() == 40
    assert page.rectangle_inputs["height"].value() == 20
    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.manual_rectangle == (5, 6, 40, 20)


def test_numeric_pre_crop_edits_update_visual_overlay(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)
    page.mode.setCurrentIndex(page.mode.findData(PreCropMode.MANUAL_RECTANGLE.value))
    page.rectangle_inputs["x"].setValue(5)
    page.rectangle_inputs["y"].setValue(6)
    page.rectangle_inputs["width"].setValue(40)
    page.rectangle_inputs["height"].setValue(20)

    overlay = page.frame_view.pre_crop_overlay

    assert overlay.mode == PreCropMode.MANUAL_RECTANGLE.value
    assert overlay.roi == (5, 6, 40, 20)


def test_manual_rectangle_visual_constraint_and_rejection() -> None:
    from ui.widgets.video_frame_view import constrain_manual_pre_crop_rectangle

    assert constrain_manual_pre_crop_rectangle((-5, 70, 50, 30), (100, 80)) == (
        0,
        50,
        50,
        30,
    )
    with pytest.raises(ValueError, match="positive size"):
        constrain_manual_pre_crop_rectangle((0, 0, 0, 10), (100, 80))
    with pytest.raises(ValueError, match="positive size"):
        constrain_manual_pre_crop_rectangle((0, 0, 10, -1), (100, 80))


def test_disable_and_reset_pre_crop_match_current_semantic_model(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)

    page._apply_pre_crop_controls(
        mode=PreCropMode.VERTICAL_KEEP_LEFT,
        boundary_px=30,
    )
    page._disable_pre_crop()

    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.enabled is False
    assert controller.state.pre_crop_config.mode == "none"
    assert controller.state.resolved_pre_crop is not None
    assert controller.state.resolved_pre_crop.roi == PreCropROI(
        x=0,
        y=0,
        width=100,
        height=80,
    )

    page._reset_pre_crop_to_full_frame()

    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.enabled is True
    assert controller.state.pre_crop_config.mode == "manual_rectangle"
    assert controller.state.pre_crop_config.manual_rectangle == (0, 0, 100, 80)
    assert controller.state.resolved_pre_crop is not None
    assert controller.state.resolved_pre_crop.roi == PreCropROI(
        x=0,
        y=0,
        width=100,
        height=80,
    )


def test_split_mode_reset_uses_active_full_span(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)

    page._apply_pre_crop_controls(
        mode=PreCropMode.VERTICAL_KEEP_LEFT,
        boundary_px=30,
    )
    page._reset_pre_crop_to_full_frame()

    assert controller.state.pre_crop_config is not None
    assert controller.state.pre_crop_config.mode == "vertical_keep_left"
    assert controller.state.pre_crop_config.boundary_px == 100
    assert controller.state.resolved_pre_crop is not None
    assert controller.state.resolved_pre_crop.roi == PreCropROI(
        x=0,
        y=0,
        width=100,
        height=80,
    )


def test_overlay_coordinate_conversion_excludes_viewer_letterboxing(
    qt_application: object,
) -> None:
    del qt_application
    from PySide6.QtCore import QPointF

    from ui.widgets.video_frame_view import VideoFrameView

    view = VideoFrameView()
    view.resize(300, 300)
    view.set_frame(np.zeros((80, 100, 3), dtype=np.uint8))
    target = view.image_target_rect()

    assert view.widget_to_frame_point(QPointF(target.left() - 1, target.center().y())) is None
    assert view.widget_to_frame_point(target.center()) == (50, 40)


def test_no_op_pre_crop_edit_does_not_invalidate_crop_review(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.configure_trim_and_pre_crop(
        start_frame=0,
        end_frame_exclusive=None,
        mode=PreCropMode.VERTICAL_KEEP_LEFT,
        boundary_px=40,
    )
    candidate = object()
    accepted = object()
    controller.state.candidate_crop_plan = candidate
    controller.state.accepted_crop_plan = accepted
    previous_revision = controller.state.crop_review_revision

    controller.configure_trim_and_pre_crop(
        start_frame=0,
        end_frame_exclusive=None,
        mode=PreCropMode.VERTICAL_KEEP_LEFT,
        boundary_px=40,
    )

    assert controller.state.candidate_crop_plan is candidate
    assert controller.state.accepted_crop_plan is accepted
    assert controller.state.crop_review_revision == previous_revision


def test_raw_video_change_clears_visual_pre_crop_overlay_roi(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)
    page._apply_pre_crop_controls(
        mode=PreCropMode.MANUAL_RECTANGLE,
        rectangle=(5, 5, 30, 20),
    )

    controller.set_raw_video_path(tmp_path / "replacement.avi")
    page.refresh_from_state()

    assert controller.state.resolved_pre_crop is None
    assert page.frame_view.pre_crop_overlay.roi is None


def test_state_cannot_advance_without_project() -> None:
    controller = _controller_with_probe()

    assert controller.can_advance(WorkflowStep.PROJECT) is False
    with pytest.raises(SetupValidationError, match="project"):
        controller.validate_step(WorkflowStep.PROJECT)


def test_state_cannot_advance_without_raw_video(tmp_path: Path) -> None:
    controller = _controller_with_probe()
    controller.create_project(tmp_path, "NoVideoProject")

    assert controller.can_advance(WorkflowStep.RAW_VIDEO) is False


def test_state_cannot_advance_with_invalid_trim_or_pre_crop(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    with pytest.raises(SetupValidationError):
        controller.configure_trim_and_pre_crop(
            start_frame=5,
            end_frame_exclusive=5,
            mode=PreCropMode.NONE,
        )

    assert controller.can_advance(WorkflowStep.TRIM_PRE_CROP) is False


def test_raw_probe_progress_is_forwarded_and_rejected_after_video_change(
    tmp_path: Path,
) -> None:
    received_callbacks = []

    def fake_probe(
        path: Path,
        *,
        require_sequential_count: bool,
        cancellation_requested,
        progress_callback,
    ) -> VideoProbeResult:
        assert require_sequential_count is True
        assert cancellation_requested() is False
        progress_callback(
            OperationProgress(
                phase="counting",
                message="Counting",
                completed_units=1,
                total_units=20,
                total_is_estimate=True,
                is_indeterminate=False,
            )
        )
        received_callbacks.append(progress_callback)
        return _probe_result(path, readable_count=20)

    controller = PreprocessSetupController(probe_function=fake_probe)
    controller.load_startup_config()
    controller.create_project(tmp_path, "ProgressProject")
    raw_path = tmp_path / "raw.avi"
    controller.set_raw_video_path(raw_path)
    context = controller.begin_task(GuiTaskKind.RAW_SEQUENTIAL_COUNT)
    progress: list[OperationProgress] = []

    controller.compute_raw_video_probe(
        raw_path,
        True,
        context,
        progress.append,
    )
    event = GuiTaskProgressTracker(context).build(progress[0])

    assert received_callbacks == [progress.append]
    assert controller.task_progress_is_current(event, context) is True

    controller.set_raw_video_path(tmp_path / "replacement.avi")

    assert controller.task_progress_is_current(event, context) is False


def test_raw_frame_index_initializes_after_raw_video_probe(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)

    assert controller.state.current_raw_frame_idx == 0
    assert controller.state.last_successfully_displayed_frame_idx is None
    assert controller.state.raw_frame_navigation_error is None
    assert controller.state.raw_frame_navigation_in_progress is False


def test_first_previous_and_next_navigation_compute_requested_index(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.state.current_raw_frame_idx = 5
    controller.state.last_successfully_displayed_frame_idx = 5

    assert controller.first_raw_frame_index() == 0
    assert controller.previous_raw_frame_index() == 4
    assert controller.next_raw_frame_index() == 6


@pytest.mark.parametrize("step", SUPPORTED_RAW_FRAME_JUMP_STEPS)
def test_jump_by_supported_step_computes_requested_index(
    tmp_path: Path,
    step: int,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.state.current_raw_frame_idx = 5_000
    controller.state.last_successfully_displayed_frame_idx = 5_000

    assert controller.set_raw_frame_navigation_step(step) == step
    assert controller.raw_frame_index_after_step(direction=1, step=step) == 5_000 + step
    assert controller.raw_frame_index_after_step(direction=-1, step=step) == 5_000 - step


def test_exact_raw_frame_jump_validates_nonnegative_integer(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)

    with pytest.raises(SetupValidationError, match="non-negative integer"):
        controller.validate_raw_frame_index(-1)


def test_out_of_range_raw_frame_request_is_rejected_when_count_is_trusted(
    tmp_path: Path,
) -> None:
    controller = _ready_controller_with_trusted_count(tmp_path)

    with pytest.raises(SetupValidationError, match=r"\[0, 20\)"):
        controller.validate_raw_frame_index(20)


def test_out_of_range_raw_frame_request_does_not_start_worker_when_count_is_trusted(
    tmp_path: Path,
) -> None:
    controller = _ready_controller_with_trusted_count(tmp_path)

    with pytest.raises(SetupValidationError, match=r"\[0, 20\)"):
        controller.begin_raw_frame_read(20)

    assert controller.task_coordinator.active_contexts == ()
    assert controller.state.raw_frame_navigation_in_progress is False


def test_unknown_raw_frame_count_allows_worker_to_attempt_navigation(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)

    request, context = controller.begin_raw_frame_read(45_716)

    assert request.raw_decode_frame_idx == 45_716
    assert controller.task_coordinator.active_contexts == (context,)
    controller.record_raw_frame_read_failure(
        OSError("synthetic stop"),
        context=context,
        requested_frame_idx=request.raw_decode_frame_idx,
    )
    controller.finish_task(context)


def test_failed_raw_frame_read_preserves_last_valid_displayed_index(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    raw_path = controller.state.raw_video_path
    assert raw_path is not None
    controller.state.current_raw_frame_idx = 5
    controller.state.last_successfully_displayed_frame_idx = 5
    _request, context = controller.begin_raw_frame_read(6)

    message = controller.record_raw_frame_read_failure(
        OSError("decode failed"),
        context=context,
        requested_frame_idx=6,
    )

    assert message == "Could not load raw frame 6: decode failed"
    assert controller.state.current_raw_frame_idx == 5
    assert controller.state.last_successfully_displayed_frame_idx == 5
    assert controller.state.raw_frame_navigation_in_progress is False


def test_raw_frame_scrubber_requires_verified_count(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller(tmp_path)
    page = TrimPreCropPage(controller)

    assert page.frame_scrubber.isEnabled() is False
    assert page.frame_scrubber.minimum() == 0
    assert page.frame_scrubber.maximum() == 0


def test_raw_frame_scrubber_defers_frame_read_until_release(
    tmp_path: Path,
    qt_application: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller_with_trusted_count(tmp_path)
    controller.state.current_raw_frame_idx = 5
    controller.state.last_successfully_displayed_frame_idx = 5
    page = TrimPreCropPage(controller)
    calls: list[int] = []
    monkeypatch.setattr(page, "_load_raw_frame", lambda index: calls.append(index))

    page.refresh_from_state()
    assert page.frame_scrubber.isEnabled() is True
    assert page.frame_scrubber.minimum() == 0
    assert page.frame_scrubber.maximum() == 19
    assert page.frame_scrubber.value() == 5
    assert page.raw_frame_status.text() == "Raw frame: 5 / 19"

    page.frame_scrubber.setValue(8)
    page._raw_frame_scrubber_moved(8)

    assert calls == []
    assert "8" in page.pending_frame_status.text()

    page._raw_frame_scrubber_released()

    assert calls == [8]


def test_raw_frame_scrubber_restores_last_valid_index_after_failed_read(
    tmp_path: Path,
    qt_application: object,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller_with_trusted_count(tmp_path)
    controller.state.current_raw_frame_idx = 5
    controller.state.last_successfully_displayed_frame_idx = 5
    page = TrimPreCropPage(controller)
    _request, context = controller.begin_raw_frame_read(8)
    page._pending_slider_frame_idx = 8
    page.frame_scrubber.setValue(8)

    page._raw_frame_read_failed(OSError("decode failed"), context, 8)
    page._raw_frame_read_finished(context)

    assert controller.state.current_raw_frame_idx == 5
    assert controller.state.last_successfully_displayed_frame_idx == 5
    assert page.frame_scrubber.value() == 5


def test_direct_jump_out_of_range_does_not_start_worker(
    tmp_path: Path,
    qt_application: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del qt_application
    from ui.pages.trim_precrop_page import TrimPreCropPage

    controller = _ready_controller_with_trusted_count(tmp_path)
    page = TrimPreCropPage(controller)
    calls: list[int] = []
    monkeypatch.setattr(page, "_load_raw_frame", lambda index: calls.append(index))

    page.exact_jump.setValue(20)
    page._jump_to_exact_frame()

    assert calls == []
    assert controller.task_coordinator.active_contexts == ()
    assert "[0, 20)" in page.error_label.text()


def test_changing_raw_video_resets_frame_navigation_state(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)
    controller.state.current_raw_frame_idx = 7
    controller.state.last_successfully_displayed_frame_idx = 7
    controller.state.raw_frame_navigation_error = "old error"
    controller.state.raw_frame_navigation_in_progress = True

    controller.set_raw_video_path(tmp_path / "replacement.avi")

    assert controller.state.current_raw_frame_idx is None
    assert controller.state.last_successfully_displayed_frame_idx is None
    assert controller.state.raw_frame_navigation_error is None
    assert controller.state.raw_frame_navigation_in_progress is False


def test_changing_project_resets_frame_navigation_state(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)
    controller.state.current_raw_frame_idx = 7
    controller.state.last_successfully_displayed_frame_idx = 7
    replacement = ProjectService().create_project(tmp_path, "FrameReplacementProject")

    controller.open_project(replacement.root_dir)

    assert controller.state.current_raw_frame_idx is None
    assert controller.state.last_successfully_displayed_frame_idx is None
    assert controller.state.raw_frame_navigation_error is None
    assert controller.state.raw_frame_navigation_in_progress is False


def test_stale_raw_frame_read_completion_cannot_update_new_video(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    old_path = controller.state.raw_video_path
    assert old_path is not None
    request, context = controller.begin_raw_frame_read(3)
    result = RawFrameReadResult(
        raw_video_path=request.raw_video_path,
        raw_decode_frame_idx=3,
        frame=np.zeros((80, 100, 3), dtype=np.uint8),
    )

    controller.set_raw_video_path(tmp_path / "new_raw.avi")
    applied = controller.apply_raw_frame_read(result, context=context)

    assert applied is None
    assert controller.state.current_raw_frame_idx is None
    assert controller.state.last_successfully_displayed_frame_idx is None


def test_set_start_to_current_uses_current_raw_decode_index(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.state.current_raw_frame_idx = 7
    controller.state.last_successfully_displayed_frame_idx = 7

    controller.set_start_frame_to_current()

    assert controller.state.start_frame == 7
    assert controller.state.trim_pre_crop_valid is True


def test_set_end_exclusive_to_current_uses_current_raw_decode_index(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.configure_trim_and_pre_crop(
        start_frame=3,
        end_frame_exclusive=None,
        mode=PreCropMode.NONE,
    )
    controller.state.current_raw_frame_idx = 7
    controller.state.last_successfully_displayed_frame_idx = 7

    controller.set_end_frame_exclusive_to_current()

    assert controller.state.start_frame == 3
    assert controller.state.end_frame_exclusive == 7
    assert controller.state.trim_pre_crop_valid is True


def test_set_end_exclusive_plus_one_uses_current_raw_decode_index_plus_one(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.configure_trim_and_pre_crop(
        start_frame=3,
        end_frame_exclusive=None,
        mode=PreCropMode.NONE,
    )
    controller.state.current_raw_frame_idx = 7
    controller.state.last_successfully_displayed_frame_idx = 7

    controller.set_end_frame_exclusive_to_current_plus_one()

    assert controller.state.end_frame_exclusive == 8


def test_invalid_visual_trim_range_is_rejected(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)
    controller.configure_trim_and_pre_crop(
        start_frame=10,
        end_frame_exclusive=None,
        mode=PreCropMode.NONE,
    )
    controller.state.current_raw_frame_idx = 5
    controller.state.last_successfully_displayed_frame_idx = 5

    with pytest.raises(SetupValidationError, match="Invalid trim"):
        controller.set_end_frame_exclusive_to_current()

    assert controller.state.trim_pre_crop_valid is False
    assert controller.state.end_frame_exclusive == 5


def test_visual_trim_change_preserves_crop_review_geometry(tmp_path: Path) -> None:
    controller = _ready_controller(tmp_path)
    controller.configure_trim_and_pre_crop(
        start_frame=0,
        end_frame_exclusive=None,
        mode=PreCropMode.NONE,
    )
    controller.state.current_raw_frame_idx = 7
    controller.state.last_successfully_displayed_frame_idx = 7
    candidate = object()
    accepted = object()
    controller.state.candidate_crop_plan = candidate
    controller.state.accepted_crop_plan = accepted
    previous_revision = controller.state.crop_review_revision

    controller.set_start_frame_to_current()

    assert controller.state.candidate_crop_plan is candidate
    assert controller.state.accepted_crop_plan is accepted
    assert controller.state.crop_review_revision == previous_revision


def test_pre_crop_change_still_invalidates_crop_review_geometry(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.configure_trim_and_pre_crop(
        start_frame=0,
        end_frame_exclusive=None,
        mode=PreCropMode.NONE,
    )
    controller.state.candidate_crop_plan = object()
    controller.state.accepted_crop_plan = object()
    previous_revision = controller.state.crop_review_revision

    controller.configure_trim_and_pre_crop(
        start_frame=0,
        end_frame_exclusive=None,
        mode=PreCropMode.VERTICAL_KEEP_LEFT,
        boundary_px=95,
    )

    assert controller.state.candidate_crop_plan is None
    assert controller.state.accepted_crop_plan is None
    assert controller.state.crop_review_revision > previous_revision


def test_direct_text_trim_entry_remains_functional_with_frame_navigation(
    tmp_path: Path,
) -> None:
    controller = _ready_controller(tmp_path)
    controller.state.current_raw_frame_idx = 12
    controller.state.last_successfully_displayed_frame_idx = 12

    controller.configure_trim_and_pre_crop(
        start_frame=2,
        end_frame_exclusive=9,
        mode=PreCropMode.NONE,
    )

    assert controller.state.start_frame == 2
    assert controller.state.end_frame_exclusive == 9
    assert controller.state.current_raw_frame_idx == 12
    assert controller.state.last_successfully_displayed_frame_idx == 12
