from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from preprocess.exceptions import ExternalTimingError
from preprocess.mat_sync_reader import (
    convert_timing_vector_to_seconds,
    get_numeric_vector,
    list_numeric_vectors,
    load_mat_workspace,
    validate_external_timing_vector,
)
from preprocess.models import TimingUnit


def test_load_mat_workspace_and_list_only_numeric_vectors(tmp_path: Path) -> None:
    mat_path = tmp_path / "timing.mat"
    savemat(
        mat_path,
        {
            "camera_ticks": np.array([[0.0, 0.1, 0.2, 0.3]]),
            "matrix": np.arange(6).reshape(2, 3),
            "label": "camera",
            "scalar": np.array([[7.0]]),
        },
    )

    workspace = load_mat_workspace(mat_path)
    candidates = list_numeric_vectors(workspace)

    assert workspace.format == "matlab_v7.2_or_earlier"
    assert [candidate.variable_name for candidate in candidates] == ["camera_ticks"]
    candidate = candidates[0]
    assert candidate.shape == (1, 4)
    assert candidate.length_after_squeeze == 4
    assert candidate.first_value == 0.0
    assert candidate.last_value == pytest.approx(0.3)
    assert candidate.median_difference == pytest.approx(0.1)
    assert candidate.estimated_fps == pytest.approx(10.0)
    np.testing.assert_allclose(
        get_numeric_vector(workspace, "camera_ticks"),
        [0.0, 0.1, 0.2, 0.3],
    )


def test_load_hdf5_mat_workspace(tmp_path: Path) -> None:
    mat_path = tmp_path / "timing-v73.mat"
    with h5py.File(mat_path, "w") as mat_file:
        group = mat_file.create_group("camera")
        group.create_dataset("ticks", data=np.array([[0.0, 0.5, 1.0]]))

    workspace = load_mat_workspace(mat_path)
    candidates = list_numeric_vectors(workspace)

    assert workspace.format == "matlab_v7.3_hdf5"
    assert [candidate.variable_name for candidate in candidates] == ["camera/ticks"]


def test_get_numeric_vector_rejects_non_vector(tmp_path: Path) -> None:
    mat_path = tmp_path / "matrix.mat"
    savemat(mat_path, {"matrix": np.arange(6).reshape(2, 3)})
    workspace = load_mat_workspace(mat_path)

    with pytest.raises(ExternalTimingError, match="not a non-empty vector"):
        get_numeric_vector(workspace, "matrix")


def test_validate_valid_temporal_vector() -> None:
    selection = validate_external_timing_vector(
        np.array([0.0, 100.0, 200.0, 300.0]),
        raw_frame_count_opencv_readable=4,
        declared_units="milliseconds",
        selected_variable="ticks",
    )

    assert selection.validation_status == "valid"
    assert selection.selected_variable == "ticks"
    assert selection.raw_vector_length == 4
    assert selection.declared_units is TimingUnit.MILLISECONDS
    assert selection.is_monotonic_increasing is True
    assert selection.median_difference == pytest.approx(100.0)
    assert selection.estimated_fps == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("units", "values", "expected"),
    [
        ("seconds", [0.0, 1.0], [0.0, 1.0]),
        ("milliseconds", [0.0, 1000.0], [0.0, 1.0]),
        ("microseconds", [0.0, 1_000_000.0], [0.0, 1.0]),
        ("nanoseconds", [0.0, 1_000_000_000.0], [0.0, 1.0]),
    ],
)
def test_convert_supported_temporal_units(
    units: str,
    values: list[float],
    expected: list[float],
) -> None:
    converted = convert_timing_vector_to_seconds(np.asarray(values), units)

    assert converted is not None
    np.testing.assert_allclose(converted, expected)


@pytest.mark.parametrize("units", ["frames", "unknown"])
def test_non_temporal_units_do_not_invent_seconds(units: str) -> None:
    converted = convert_timing_vector_to_seconds(np.array([0.0, 1.0]), units)
    selection = validate_external_timing_vector(
        np.array([0.0, 1.0]),
        raw_frame_count_opencv_readable=2,
        declared_units=units,
    )

    assert converted is None
    assert selection.estimated_fps is None


def test_timing_vector_length_must_exactly_match_readable_count() -> None:
    with pytest.raises(ExternalTimingError, match="does not match"):
        validate_external_timing_vector(
            np.array([0.0, 1.0, 2.0]),
            raw_frame_count_opencv_readable=4,
            declared_units="seconds",
        )


def test_non_finite_timing_vector_is_rejected() -> None:
    with pytest.raises(ExternalTimingError, match="non-finite"):
        validate_external_timing_vector(
            np.array([0.0, np.nan, 2.0]),
            raw_frame_count_opencv_readable=3,
            declared_units="seconds",
        )


@pytest.mark.parametrize(
    "vector",
    [
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
    ],
)
def test_non_monotonic_timing_vector_is_rejected(vector: np.ndarray) -> None:
    with pytest.raises(ExternalTimingError, match="strictly increasing"):
        validate_external_timing_vector(
            vector,
            raw_frame_count_opencv_readable=3,
            declared_units="seconds",
        )


def test_unsupported_timing_units_are_rejected() -> None:
    with pytest.raises(ExternalTimingError, match="Unsupported"):
        convert_timing_vector_to_seconds(np.array([0.0, 1.0]), "minutes")
