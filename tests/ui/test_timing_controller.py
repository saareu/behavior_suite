from pathlib import Path

import numpy as np
import pytest

from preprocess.models import (
    ExternalTimeSelection,
    MatWorkspace,
    TimingUnit,
    VideoProbeResult,
)
from ui.controllers.timing_controller import (
    TimingController,
    TimingValidationError,
)
from ui.state import PreprocessSetupState, RawReadableCountStatus, TimingMode


def _probe_result(path: Path, *, readable_count: int | None) -> VideoProbeResult:
    return VideoProbeResult(
        source_path=path.resolve(),
        width=100,
        height=80,
        codec="mjpeg",
        pixel_format="yuvj420p",
        duration_sec=0.4,
        avg_frame_rate="10/1",
        r_frame_rate="10/1",
        time_base="1/10",
        frame_count_ffprobe=4,
        frame_count_opencv_reported=4,
        frame_count_opencv_readable=readable_count,
        opencv_fps=10.0,
        raw_fps_effective=10.0,
        raw_fps_effective_method="ffprobe_avg_frame_rate",
        pts_status="not_extracted",
        ffprobe_succeeded=True,
    )


def _state(tmp_path: Path, *, readable_count: int | None = 4) -> PreprocessSetupState:
    raw_path = tmp_path / "raw.avi"
    return PreprocessSetupState(
        raw_video_path=raw_path,
        raw_probe=_probe_result(raw_path, readable_count=readable_count),
    )


def _workspace(
    tmp_path: Path,
    vector: np.ndarray | None = None,
) -> MatWorkspace:
    return MatWorkspace(
        source_path=tmp_path / "timing.mat",
        format="mat-v5",
        variables={
            "frame_times": (
                np.array([0.0, 0.1, 0.2, 0.3])
                if vector is None
                else vector
            ),
            "matrix": np.ones((2, 2)),
        },
    )


def _loaded_controller(
    tmp_path: Path,
    *,
    vector: np.ndarray | None = None,
    readable_count: int | None = 4,
) -> TimingController:
    workspace = _workspace(tmp_path, vector)
    controller = TimingController(
        _state(tmp_path, readable_count=readable_count),
        workspace_loader=lambda _path: workspace,
    )
    controller.choose_matlab_timing()
    controller.load_mat_file(workspace.source_path)
    return controller


def _select(
    controller: TimingController,
    units: TimingUnit = TimingUnit.SECONDS,
) -> None:
    controller.select_candidate("frame_times")
    controller.select_units(units)


def test_no_timing_mode_clears_state_and_permits_navigation(tmp_path: Path) -> None:
    controller = _loaded_controller(tmp_path)
    _select(controller)
    controller.validate_selected_timing()

    controller.choose_no_external_timing()

    assert controller.state.timing_mode is TimingMode.NO_EXTERNAL
    assert controller.state.timing_status == "not_provided"
    assert controller.state.external_time_selection is None
    assert controller.state.external_time_vector_seconds is None
    assert controller.state.mat_candidates == []
    assert controller.can_advance() is True


def test_mat_mode_initially_blocks_navigation(tmp_path: Path) -> None:
    controller = TimingController(_state(tmp_path))

    controller.choose_matlab_timing()

    assert controller.state.timing_mode is TimingMode.MATLAB
    assert controller.state.timing_status == "pending"
    assert controller.can_advance() is False


def test_mat_loading_populates_compact_candidate_summaries(tmp_path: Path) -> None:
    controller = _loaded_controller(tmp_path)

    summaries = controller.candidate_summaries()

    assert [row.variable_name for row in summaries] == ["frame_times"]
    assert summaries[0].shape == (4,)
    assert summaries[0].length_after_squeeze == 4
    assert summaries[0].length_matches_raw is True
    assert controller.state.mat_workspace_format == "mat-v5"
    assert controller.state.selected_timing_variable is None


def test_candidate_selection_is_required(tmp_path: Path) -> None:
    controller = _loaded_controller(tmp_path)
    controller.select_units(TimingUnit.SECONDS)

    with pytest.raises(TimingValidationError, match="candidate timing variable"):
        controller.validate_selected_timing()


def test_units_selection_is_required(tmp_path: Path) -> None:
    controller = _loaded_controller(tmp_path)
    controller.select_candidate("frame_times")

    with pytest.raises(TimingValidationError, match="Select timing units"):
        controller.validate_selected_timing()


def test_timing_validation_requires_raw_readable_frame_count(tmp_path: Path) -> None:
    controller = _loaded_controller(tmp_path, readable_count=None)
    _select(controller)

    assert controller.raw_frame_count_action() == "count"
    with pytest.raises(TimingValidationError, match="full sequential"):
        controller.validate_selected_timing()


def test_existing_same_session_count_is_reused_without_second_probe(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    probe_calls: list[Path] = []

    def unexpected_probe(
        path: Path, *, require_sequential_count: bool
    ) -> VideoProbeResult:
        probe_calls.append(path)
        return _probe_result(path, readable_count=4)

    controller = TimingController(
        _state(tmp_path),
        workspace_loader=lambda _path: workspace,
        probe_function=unexpected_probe,
    )
    controller.choose_matlab_timing()
    controller.load_mat_file(workspace.source_path)
    _select(controller)

    selection = controller.validate_selected_timing()

    assert selection.selected_variable == "frame_times"
    assert controller.reusable_raw_frame_count() == 4
    assert probe_calls == []
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.COUNTED_THIS_SESSION
    )


def test_existing_count_replaces_default_count_action_with_explicit_recount(
    tmp_path: Path,
) -> None:
    controller = TimingController(_state(tmp_path))

    assert controller.raw_frame_count_action() == "recount"
    with pytest.raises(TimingValidationError, match="already available"):
        controller.prepare_full_raw_frame_count()

    path = controller.prepare_full_raw_frame_count(force_recount=True)

    assert path == controller.state.raw_video_path
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.RECOUNTING
    )


def test_failed_diagnostic_recount_preserves_existing_count_and_no_timing_mode(
    tmp_path: Path,
) -> None:
    controller = TimingController(_state(tmp_path))
    controller.prepare_full_raw_frame_count(force_recount=True)

    controller.record_full_raw_frame_count_failure(OSError("decode failed"))

    assert controller.reusable_raw_frame_count() == 4
    assert (
        controller.state.raw_readable_count_status
        is RawReadableCountStatus.RECOUNT_FAILED
    )
    assert controller.state.timing_mode is TimingMode.NO_EXTERNAL
    assert controller.state.timing_valid is True
    assert controller.state.timing_status == "not_provided"


def test_full_raw_frame_count_updates_probe_and_requests_core_count(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, bool]] = []
    state = _state(tmp_path, readable_count=None)

    def fake_probe(path: Path, *, require_sequential_count: bool) -> VideoProbeResult:
        calls.append((path, require_sequential_count))
        return _probe_result(path, readable_count=4)

    controller = TimingController(state, probe_function=fake_probe)

    count = controller.count_all_readable_raw_frames()

    assert count == 4
    assert calls == [(state.raw_video_path, True)]
    assert state.raw_probe is not None
    assert state.raw_probe.frame_count_opencv_readable == 4
    assert controller.raw_frame_count_action() == "recount"
    assert (
        state.raw_readable_count_status
        is RawReadableCountStatus.COUNTED_THIS_SESSION
    )


def test_mismatched_probe_source_does_not_supply_timing_count(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.raw_probe = _probe_result(tmp_path / "other.avi", readable_count=4)
    controller = TimingController(state)

    assert controller.reusable_raw_frame_count() is None
    assert controller.raw_frame_count_action() == "count"
    assert (
        state.raw_readable_count_status is RawReadableCountStatus.NOT_COUNTED
    )


def test_valid_seconds_timing_stores_selection_and_seconds_vector(
    tmp_path: Path,
) -> None:
    controller = _loaded_controller(tmp_path)
    _select(controller)

    selection = controller.validate_selected_timing()

    assert isinstance(selection, ExternalTimeSelection)
    assert selection.selected_variable == "frame_times"
    assert selection.declared_units is TimingUnit.SECONDS
    np.testing.assert_allclose(
        controller.state.external_time_vector_seconds,
        [0.0, 0.1, 0.2, 0.3],
    )
    assert controller.state.timing_status == "valid"
    assert controller.can_advance() is True


def test_milliseconds_conversion_uses_core_helper(tmp_path: Path) -> None:
    controller = _loaded_controller(
        tmp_path,
        vector=np.array([0.0, 100.0, 200.0, 300.0]),
    )
    _select(controller, TimingUnit.MILLISECONDS)

    controller.validate_selected_timing()

    np.testing.assert_allclose(
        controller.state.external_time_vector_seconds,
        [0.0, 0.1, 0.2, 0.3],
    )


@pytest.mark.parametrize("units", [TimingUnit.FRAMES, TimingUnit.UNKNOWN])
def test_non_temporal_units_store_no_seconds_and_mark_not_convertible(
    tmp_path: Path,
    units: TimingUnit,
) -> None:
    controller = _loaded_controller(tmp_path)
    _select(controller, units)

    selection = controller.validate_selected_timing()

    assert selection.declared_units is units
    assert controller.state.external_time_vector_seconds is None
    assert controller.state.timing_status == "not_convertible_to_seconds"
    assert controller.can_advance() is True


def test_length_mismatch_is_rejected(tmp_path: Path) -> None:
    controller = _loaded_controller(
        tmp_path,
        vector=np.array([0.0, 0.1, 0.2]),
    )
    _select(controller)

    with pytest.raises(TimingValidationError, match="does not match"):
        controller.validate_selected_timing()


def test_non_monotonic_vector_is_rejected(tmp_path: Path) -> None:
    controller = _loaded_controller(
        tmp_path,
        vector=np.array([0.0, 0.1, 0.1, 0.3]),
    )
    _select(controller)

    with pytest.raises(TimingValidationError, match="strictly increasing"):
        controller.validate_selected_timing()


def test_non_finite_vector_is_rejected(tmp_path: Path) -> None:
    controller = _loaded_controller(
        tmp_path,
        vector=np.array([0.0, 0.1, np.nan, 0.3]),
    )
    _select(controller)

    with pytest.raises(TimingValidationError, match="non-finite"):
        controller.validate_selected_timing()


def test_invalid_mat_file_produces_clear_controller_error(tmp_path: Path) -> None:
    def invalid_loader(_path: Path) -> MatWorkspace:
        raise OSError("cannot read file")

    controller = TimingController(
        _state(tmp_path),
        workspace_loader=invalid_loader,
    )
    controller.choose_matlab_timing()

    with pytest.raises(TimingValidationError, match="Could not load.*cannot read"):
        controller.load_mat_file(tmp_path / "bad.mat")

    assert controller.state.timing_status == "invalid"


def test_state_transitions_are_headless(tmp_path: Path) -> None:
    controller = _loaded_controller(tmp_path)
    _select(controller)

    controller.validate_selected_timing()

    assert isinstance(controller.state, PreprocessSetupState)
    assert controller.state.timing_valid is True
