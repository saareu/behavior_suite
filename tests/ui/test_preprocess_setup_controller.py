from pathlib import Path
from typing import Any

import pytest
import yaml

from preprocess.config import PreprocessConfig, TrimConfig
from preprocess.models import VideoProbeResult
from preprocess.pre_crop import PreCropMode, PreCropROI
from project.service import ProjectService
from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)
from ui.state import WorkflowStep


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


def _controller_with_probe(
    calls: list[bool] | None = None,
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

    controller = PreprocessSetupController(probe_function=fake_probe)
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
    assert controller.state.resolved_pre_crop is None


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
