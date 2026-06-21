"""Generic MATLAB workspace loading and external timing validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

from preprocess.exceptions import ExternalTimingError
from preprocess.models import (
    ExternalTimeSelection,
    MatVectorCandidate,
    MatWorkspace,
    TimingUnit,
)

_TEMPORAL_UNIT_SCALE = {
    TimingUnit.SECONDS: 1.0,
    TimingUnit.MILLISECONDS: 1e-3,
    TimingUnit.MICROSECONDS: 1e-6,
    TimingUnit.NANOSECONDS: 1e-9,
}


def _is_real_numeric_array(value: Any) -> bool:
    try:
        array = np.asarray(value)
        return bool(
            np.issubdtype(array.dtype, np.number)
            and not np.issubdtype(array.dtype, np.complexfloating)
            and not np.issubdtype(array.dtype, np.bool_)
        )
    except (TypeError, ValueError):
        return False


def _read_hdf5_dataset(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset[()])
    fields = value.dtype.fields
    if fields is not None and {"real", "imag"}.issubset(fields):
        value = value["real"] + 1j * value["imag"]
    matlab_class = dataset.attrs.get("MATLAB_class")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="replace")
    if matlab_class == "char":
        value = value.astype("U1")
    return value


def _load_hdf5_workspace(path: Path) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    try:
        with h5py.File(path, "r") as mat_file:

            def collect_dataset(name: str, item: h5py.Group | h5py.Dataset) -> None:
                if isinstance(item, h5py.Dataset) and not name.startswith("#refs#"):
                    variables[name] = _read_hdf5_dataset(item)

            mat_file.visititems(collect_dataset)
    except (OSError, ValueError) as exc:
        raise ExternalTimingError(f"Could not load MATLAB v7.3 file {path}: {exc}") from exc
    return variables


def load_mat_workspace(path: Path) -> MatWorkspace:
    """Load a MATLAB workspace through SciPy or h5py as appropriate.

    MATLAB v7.2 and earlier files use scipy.io.loadmat. HDF5-based MATLAB
    v7.3 files use h5py. Internal MATLAB metadata keys are excluded.

    Raises:
        ExternalTimingError: If the file is absent or cannot be loaded.
    """

    mat_path = Path(path).expanduser()
    if not mat_path.is_file():
        raise ExternalTimingError(f"MATLAB file does not exist: {mat_path}")
    mat_path = mat_path.resolve()

    try:
        is_hdf5 = h5py.is_hdf5(mat_path)
    except OSError as exc:
        raise ExternalTimingError(f"Could not inspect MATLAB file {mat_path}: {exc}") from exc

    if is_hdf5:
        variables = _load_hdf5_workspace(mat_path)
        workspace_format = "matlab_v7.3_hdf5"
    else:
        try:
            loaded = loadmat(mat_path, struct_as_record=False, squeeze_me=False)
        except (OSError, ValueError, TypeError, NotImplementedError) as exc:
            raise ExternalTimingError(f"Could not load MATLAB file {mat_path}: {exc}") from exc
        variables = {name: value for name, value in loaded.items() if not name.startswith("__")}
        workspace_format = "matlab_v7.2_or_earlier"

    return MatWorkspace(
        source_path=mat_path,
        format=workspace_format,
        variables=variables,
    )


def list_numeric_vectors(workspace: MatWorkspace) -> list[MatVectorCandidate]:
    """List real numeric variables that are one-dimensional after squeeze.

    No variable is selected automatically and variable names do not affect
    eligibility.
    """

    candidates: list[MatVectorCandidate] = []
    for variable_name in sorted(workspace.variables):
        value = workspace.variables[variable_name]
        if not _is_real_numeric_array(value):
            continue
        array = np.asarray(value)
        squeezed = np.squeeze(array)
        if squeezed.ndim != 1 or squeezed.size == 0:
            continue

        numeric = squeezed.astype(np.float64, copy=False)
        differences = np.diff(numeric)
        median_difference: float | None = None
        estimated_fps: float | None = None
        if (
            differences.size > 0
            and np.all(np.isfinite(numeric))
            and np.all(differences > 0)
        ):
            median_difference = float(np.median(differences))
            if median_difference > 0:
                estimated_fps = 1.0 / median_difference

        candidates.append(
            MatVectorCandidate(
                variable_name=variable_name,
                shape=tuple(int(dimension) for dimension in array.shape),
                dtype=str(array.dtype),
                length_after_squeeze=int(squeezed.size),
                first_value=float(numeric[0]),
                last_value=float(numeric[-1]),
                median_difference=median_difference,
                estimated_fps=estimated_fps,
            )
        )
    return candidates


def get_numeric_vector(workspace: MatWorkspace, variable_name: str) -> np.ndarray:
    """Return a selected real numeric MATLAB variable as a 1D array.

    Raises:
        ExternalTimingError: If the variable is missing, non-numeric, empty,
            or not one-dimensional after squeeze.
    """

    if variable_name not in workspace.variables:
        raise ExternalTimingError(f"MATLAB variable was not found: {variable_name}")
    value = workspace.variables[variable_name]
    if not _is_real_numeric_array(value):
        raise ExternalTimingError(f"MATLAB variable is not a real numeric array: {variable_name}")
    vector = np.squeeze(np.asarray(value))
    if vector.ndim != 1 or vector.size == 0:
        raise ExternalTimingError(
            f"MATLAB variable is not a non-empty vector after squeeze: {variable_name}"
        )
    return np.array(vector, copy=True)


def _parse_timing_unit(declared_units: str | TimingUnit) -> TimingUnit:
    try:
        return TimingUnit(declared_units)
    except ValueError as exc:
        accepted = ", ".join(unit.value for unit in TimingUnit)
        raise ExternalTimingError(
            f"Unsupported external timing units '{declared_units}'. Expected one of: {accepted}."
        ) from exc


def convert_timing_vector_to_seconds(
    vector: np.ndarray,
    declared_units: str | TimingUnit,
) -> np.ndarray | None:
    """Convert a real numeric timing vector to seconds when units are temporal.

    The frames and unknown units deliberately return None. Unsupported units or
    invalid vector shapes raise ExternalTimingError.
    """

    unit = _parse_timing_unit(declared_units)
    if unit in {TimingUnit.FRAMES, TimingUnit.UNKNOWN}:
        return None
    if not _is_real_numeric_array(vector):
        raise ExternalTimingError("External timing vector must be a real numeric array.")
    squeezed = np.squeeze(np.asarray(vector))
    if squeezed.ndim != 1 or squeezed.size == 0:
        raise ExternalTimingError("External timing vector must be non-empty and 1D after squeeze.")
    return squeezed.astype(np.float64, copy=False) * _TEMPORAL_UNIT_SCALE[unit]


def validate_external_timing_vector(
    vector: np.ndarray,
    raw_frame_count_opencv_readable: int,
    declared_units: str | TimingUnit,
    *,
    source_path: Path | None = None,
    selected_variable: str | None = None,
) -> ExternalTimeSelection:
    """Validate a selected vector against the untrimmed readable frame count.

    Validation requires a real numeric, finite, strictly increasing vector
    whose length exactly matches raw_frame_count_opencv_readable.

    Raises:
        ExternalTimingError: If any hard external-timing validation fails.
    """

    unit = _parse_timing_unit(declared_units)
    if raw_frame_count_opencv_readable <= 0:
        raise ExternalTimingError("Raw OpenCV-readable frame count must be positive.")
    if not _is_real_numeric_array(vector):
        raise ExternalTimingError("External timing vector must be a real numeric array.")

    squeezed = np.squeeze(np.asarray(vector))
    if squeezed.ndim != 1 or squeezed.size == 0:
        raise ExternalTimingError("External timing vector must be non-empty and 1D after squeeze.")
    numeric = squeezed.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ExternalTimingError("External timing vector contains non-finite values.")
    if numeric.size != raw_frame_count_opencv_readable:
        raise ExternalTimingError(
            "External timing vector length "
            f"({numeric.size}) does not match raw OpenCV-readable frame count "
            f"({raw_frame_count_opencv_readable})."
        )

    differences = np.diff(numeric)
    if differences.size > 0 and not np.all(differences > 0):
        raise ExternalTimingError("External timing vector must be strictly increasing.")
    median_difference = float(np.median(differences)) if differences.size else None

    converted = convert_timing_vector_to_seconds(numeric, unit)
    estimated_fps: float | None = None
    if converted is not None and converted.size > 1:
        median_seconds = float(np.median(np.diff(converted)))
        if median_seconds > 0:
            estimated_fps = 1.0 / median_seconds

    return ExternalTimeSelection(
        provided=True,
        source_path=Path(source_path).resolve() if source_path is not None else None,
        selected_variable=selected_variable,
        declared_units=unit,
        raw_vector_length=int(numeric.size),
        raw_video_frame_count_opencv_readable=raw_frame_count_opencv_readable,
        is_numeric=True,
        is_one_dimensional=True,
        is_finite=True,
        is_monotonic_increasing=True,
        median_difference=median_difference,
        estimated_fps=estimated_fps,
        validation_status="valid",
    )
