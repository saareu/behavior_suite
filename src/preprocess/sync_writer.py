"""Deterministic prepared-frame synchronization data and NPZ persistence."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from preprocess.exceptions import SyncValidationError
from preprocess.models import TimingUnit

PREPARED_SYNC_SCHEMA_VERSION = "prepared_sync_v1"
PREPARED_SYNC_NPZ_KEYS = (
    "sync_schema_version",
    "prepared_frame_idx",
    "raw_decode_frame_idx",
    "prepared_time_sec",
    "raw_pts_time_sec",
    "raw_pts_status",
    "external_time_sec",
    "external_time_status",
    "external_time_source",
    "external_time_variable_name",
    "external_time_units",
    "fps_header",
    "raw_fps_effective",
    "start_frame",
    "end_frame_exclusive",
    "raw_frame_count_opencv_readable",
    "prepared_frame_count_opencv_reported",
    "prepared_frame_count_opencv_readable",
    "frame_count_used_for_sleap",
)
_EXTERNAL_TIME_STATUSES = {
    "valid",
    "not_provided",
    "not_convertible_to_seconds",
}
_NON_TEMPORAL_UNITS = {TimingUnit.FRAMES, TimingUnit.UNKNOWN}


class PreparedSyncData(BaseModel):
    """Typed frame mapping, timing arrays, and source-count provenance."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    sync_schema_version: str
    prepared_frame_idx: np.ndarray
    raw_decode_frame_idx: np.ndarray
    prepared_time_sec: np.ndarray
    raw_pts_time_sec: np.ndarray
    raw_pts_status: str
    external_time_sec: np.ndarray
    external_time_status: str
    external_time_source: str | None
    external_time_variable_name: str | None
    external_time_units: str
    fps_header: float
    raw_fps_effective: float | None
    start_frame: int
    end_frame_exclusive: int | None
    raw_frame_count_opencv_readable: int | None
    prepared_frame_count_opencv_reported: int
    prepared_frame_count_opencv_readable: int
    frame_count_used_for_sleap: int


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SyncValidationError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SyncValidationError(f"{name} must be a non-negative integer.")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise SyncValidationError(f"{name} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SyncValidationError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise SyncValidationError(f"{name} must be finite and positive.")
    return numeric


def _optional_positive_float(name: str, value: float | None) -> float | None:
    return None if value is None else _positive_float(name, value)


def _timing_unit(value: str) -> TimingUnit:
    try:
        return TimingUnit(value)
    except ValueError as exc:
        accepted = ", ".join(unit.value for unit in TimingUnit)
        raise SyncValidationError(
            f"external_time_units must be one of: {accepted}."
        ) from exc


def _real_vector(
    name: str,
    value: np.ndarray,
    *,
    finite: bool,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise SyncValidationError(f"{name} must be a numeric one-dimensional array.") from exc
    if (
        array.ndim != 1
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise SyncValidationError(f"{name} must be a real numeric one-dimensional array.")
    numeric = np.array(array, dtype=np.float64, copy=True)
    if finite and not np.all(np.isfinite(numeric)):
        raise SyncValidationError(f"{name} must contain only finite values.")
    numeric.setflags(write=False)
    return numeric


def _readonly_int_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _readonly_float_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _require_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise SyncValidationError(f"{name} must be a one-dimensional NumPy array.")
    if (
        not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        raise SyncValidationError(f"{name} must be a real numeric NumPy array.")
    return value


def validate_prepared_sync(sync: PreparedSyncData) -> None:
    """Validate all frame-count, mapping, timing, and status invariants.

    Raises:
        SyncValidationError: If any sync array or scalar violates the
            deterministic no-resampling contract.
    """

    if sync.sync_schema_version != PREPARED_SYNC_SCHEMA_VERSION:
        raise SyncValidationError(
            f"Unsupported sync schema version: {sync.sync_schema_version}"
        )
    readable_count = _positive_int(
        "prepared_frame_count_opencv_readable",
        sync.prepared_frame_count_opencv_readable,
    )
    reported_count = _positive_int(
        "prepared_frame_count_opencv_reported",
        sync.prepared_frame_count_opencv_reported,
    )
    if reported_count != readable_count:
        raise SyncValidationError(
            "Prepared OpenCV-reported frame count must equal readable frame count."
        )
    frame_count_used = _positive_int(
        "frame_count_used_for_sleap",
        sync.frame_count_used_for_sleap,
    )
    if frame_count_used != readable_count:
        raise SyncValidationError(
            "frame_count_used_for_sleap must equal the prepared readable frame count."
        )
    start_frame = _nonnegative_int("start_frame", sync.start_frame)
    fps_header = _positive_float("fps_header", sync.fps_header)
    _optional_positive_float("raw_fps_effective", sync.raw_fps_effective)

    arrays = {
        "prepared_frame_idx": _require_vector(
            "prepared_frame_idx",
            sync.prepared_frame_idx,
        ),
        "raw_decode_frame_idx": _require_vector(
            "raw_decode_frame_idx",
            sync.raw_decode_frame_idx,
        ),
        "prepared_time_sec": _require_vector(
            "prepared_time_sec",
            sync.prepared_time_sec,
        ),
        "raw_pts_time_sec": _require_vector(
            "raw_pts_time_sec",
            sync.raw_pts_time_sec,
        ),
        "external_time_sec": _require_vector(
            "external_time_sec",
            sync.external_time_sec,
        ),
    }
    for name, array in arrays.items():
        if len(array) != readable_count:
            raise SyncValidationError(
                f"{name} length {len(array)} does not match prepared readable "
                f"frame count {readable_count}."
            )
    for name in ("prepared_frame_idx", "raw_decode_frame_idx"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise SyncValidationError(f"{name} must use an integer dtype.")

    expected_prepared_idx = np.arange(readable_count, dtype=np.int64)
    if not np.array_equal(sync.prepared_frame_idx, expected_prepared_idx):
        raise SyncValidationError(
            "prepared_frame_idx must equal [0, 1, ..., N - 1]."
        )
    expected_raw_idx = start_frame + expected_prepared_idx
    if not np.array_equal(sync.raw_decode_frame_idx, expected_raw_idx):
        raise SyncValidationError(
            "raw_decode_frame_idx must equal start_frame + prepared_frame_idx."
        )
    expected_prepared_time = expected_prepared_idx.astype(np.float64) / fps_header
    if not np.array_equal(sync.prepared_time_sec, expected_prepared_time):
        raise SyncValidationError(
            "prepared_time_sec must equal prepared_frame_idx / fps_header."
        )
    if not np.all(np.isfinite(sync.prepared_time_sec)):
        raise SyncValidationError("prepared_time_sec must contain only finite values.")

    if sync.end_frame_exclusive is not None:
        end_frame = _positive_int(
            "end_frame_exclusive",
            sync.end_frame_exclusive,
        )
        if start_frame + readable_count != end_frame:
            raise SyncValidationError(
                "start_frame + prepared readable frame count must equal "
                "end_frame_exclusive."
            )

    raw_count = sync.raw_frame_count_opencv_readable
    if raw_count is not None:
        raw_count = _positive_int(
            "raw_frame_count_opencv_readable",
            raw_count,
        )
        if np.any(sync.raw_decode_frame_idx < 0) or np.any(
            sync.raw_decode_frame_idx >= raw_count
        ):
            raise SyncValidationError(
                "raw_decode_frame_idx contains an index outside the raw readable frame range."
            )

    if not isinstance(sync.raw_pts_status, str) or not sync.raw_pts_status:
        raise SyncValidationError("raw_pts_status must be a non-empty string.")
    unit = _timing_unit(sync.external_time_units)
    if sync.external_time_status not in _EXTERNAL_TIME_STATUSES:
        raise SyncValidationError(
            f"Unsupported external_time_status: {sync.external_time_status}"
        )
    if sync.external_time_status == "valid":
        if unit in _NON_TEMPORAL_UNITS:
            raise SyncValidationError(
                "Valid external timing must use temporal units."
            )
        if not np.all(np.isfinite(sync.external_time_sec)):
            raise SyncValidationError(
                "Valid external_time_sec must contain only finite values."
            )
        if readable_count > 1 and not np.all(np.diff(sync.external_time_sec) > 0):
            raise SyncValidationError(
                "Valid external_time_sec must be strictly increasing."
            )
    else:
        if not np.all(np.isnan(sync.external_time_sec)):
            raise SyncValidationError(
                "Unavailable external timing must be represented by NaN values."
            )
    if (
        sync.external_time_status == "not_convertible_to_seconds"
        and unit not in _NON_TEMPORAL_UNITS
    ):
        raise SyncValidationError(
            "not_convertible_to_seconds requires frames or unknown units."
        )
    if sync.external_time_status == "not_provided" and (
        sync.external_time_source is not None
        or sync.external_time_variable_name is not None
    ):
        raise SyncValidationError(
            "External timing source and variable must be null when timing is not provided."
        )


def build_prepared_sync(
    prepared_frame_count: int,
    start_frame: int,
    end_frame_exclusive: int | None,
    fps_header: float,
    raw_fps_effective: float | None,
    raw_pts_time_sec: np.ndarray | None,
    raw_pts_status: str,
    external_ttl_vector: np.ndarray | None,
    external_time_status: str,
    external_time_source: str | None,
    external_time_variable_name: str | None,
    external_time_units: str,
    raw_frame_count_opencv_readable: int | None,
    prepared_frame_count_opencv_reported: int,
    prepared_frame_count_opencv_readable: int,
) -> PreparedSyncData:
    """Build deterministic prepared-to-raw mapping and timing arrays.

    External temporal vectors must already be converted to seconds. Raw PTS
    arrays, when supplied, must already correspond one-to-one to prepared
    frames. This function never extracts, repairs, sorts, or infers timing.

    Raises:
        SyncValidationError: If counts, frame range, FPS, PTS, external timing,
            or generated sync invariants are invalid.
    """

    expected_count = _positive_int("prepared_frame_count", prepared_frame_count)
    readable_count = _positive_int(
        "prepared_frame_count_opencv_readable",
        prepared_frame_count_opencv_readable,
    )
    reported_count = _positive_int(
        "prepared_frame_count_opencv_reported",
        prepared_frame_count_opencv_reported,
    )
    if expected_count != readable_count:
        raise SyncValidationError(
            "prepared_frame_count must equal prepared OpenCV-readable frame count."
        )
    if reported_count != readable_count:
        raise SyncValidationError(
            "Prepared OpenCV-reported frame count must equal readable frame count."
        )
    start = _nonnegative_int("start_frame", start_frame)
    fps = _positive_float("fps_header", fps_header)
    raw_fps = _optional_positive_float("raw_fps_effective", raw_fps_effective)
    if end_frame_exclusive is not None:
        end_frame = _positive_int("end_frame_exclusive", end_frame_exclusive)
        if start + readable_count != end_frame:
            raise SyncValidationError(
                "start_frame + prepared readable frame count must equal "
                "end_frame_exclusive."
            )
    else:
        end_frame = None

    prepared_idx = _readonly_int_array(np.arange(readable_count, dtype=np.int64))
    raw_idx = _readonly_int_array(start + prepared_idx)
    prepared_time = _readonly_float_array(
        prepared_idx.astype(np.float64) / fps
    )
    if raw_pts_time_sec is None:
        raw_pts = _readonly_float_array(
            np.full(readable_count, np.nan, dtype=np.float64)
        )
    else:
        raw_pts = _real_vector(
            "raw_pts_time_sec",
            raw_pts_time_sec,
            finite=False,
        )
        if len(raw_pts) != readable_count:
            raise SyncValidationError(
                "raw_pts_time_sec length must equal prepared readable frame count."
            )
    if not isinstance(raw_pts_status, str) or not raw_pts_status:
        raise SyncValidationError("raw_pts_status must be a non-empty string.")

    if raw_frame_count_opencv_readable is None:
        raw_count = None
    else:
        raw_count = _positive_int(
            "raw_frame_count_opencv_readable",
            raw_frame_count_opencv_readable,
        )
        if np.any(raw_idx >= raw_count):
            raise SyncValidationError(
                "raw_decode_frame_idx contains an index outside the raw readable frame range."
            )

    unit = _timing_unit(external_time_units)
    if external_ttl_vector is None:
        external_time = _readonly_float_array(
            np.full(readable_count, np.nan, dtype=np.float64)
        )
        resolved_external_status = "not_provided"
        resolved_external_source = None
        resolved_external_variable = None
    else:
        external_vector = _real_vector(
            "external_ttl_vector",
            external_ttl_vector,
            finite=True,
        )
        if raw_count is None:
            raise SyncValidationError(
                "raw_frame_count_opencv_readable is required with external timing."
            )
        if len(external_vector) != raw_count:
            raise SyncValidationError(
                "external_ttl_vector length must equal raw readable frame count."
            )
        if len(external_vector) > 1 and not np.all(np.diff(external_vector) > 0):
            raise SyncValidationError(
                "external_ttl_vector must be strictly increasing."
            )
        if np.any(raw_idx < 0) or np.any(raw_idx >= len(external_vector)):
            raise SyncValidationError(
                "raw_decode_frame_idx is outside the external timing vector range."
            )
        if unit in _NON_TEMPORAL_UNITS:
            external_time = _readonly_float_array(
                np.full(readable_count, np.nan, dtype=np.float64)
            )
            resolved_external_status = "not_convertible_to_seconds"
        else:
            if external_time_status != "valid":
                raise SyncValidationError(
                    "Temporal external timing must have external_time_status='valid'."
                )
            external_time = _readonly_float_array(external_vector[raw_idx])
            resolved_external_status = "valid"
        resolved_external_source = external_time_source
        resolved_external_variable = external_time_variable_name

    sync = PreparedSyncData(
        sync_schema_version=PREPARED_SYNC_SCHEMA_VERSION,
        prepared_frame_idx=prepared_idx,
        raw_decode_frame_idx=raw_idx,
        prepared_time_sec=prepared_time,
        raw_pts_time_sec=raw_pts,
        raw_pts_status=raw_pts_status,
        external_time_sec=external_time,
        external_time_status=resolved_external_status,
        external_time_source=resolved_external_source,
        external_time_variable_name=resolved_external_variable,
        external_time_units=unit.value,
        fps_header=fps,
        raw_fps_effective=raw_fps,
        start_frame=start,
        end_frame_exclusive=end_frame,
        raw_frame_count_opencv_readable=raw_count,
        prepared_frame_count_opencv_reported=reported_count,
        prepared_frame_count_opencv_readable=readable_count,
        frame_count_used_for_sleap=readable_count,
    )
    validate_prepared_sync(sync)
    return sync


def _optional_array(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    """Encode null as an empty typed array so NPZ loading never needs pickle."""

    if value is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _sync_payload(sync: PreparedSyncData) -> dict[str, np.ndarray]:
    return {
        "sync_schema_version": np.asarray(sync.sync_schema_version),
        "prepared_frame_idx": np.asarray(sync.prepared_frame_idx, dtype=np.int64),
        "raw_decode_frame_idx": np.asarray(sync.raw_decode_frame_idx, dtype=np.int64),
        "prepared_time_sec": np.asarray(sync.prepared_time_sec, dtype=np.float64),
        "raw_pts_time_sec": np.asarray(sync.raw_pts_time_sec, dtype=np.float64),
        "raw_pts_status": np.asarray(sync.raw_pts_status),
        "external_time_sec": np.asarray(sync.external_time_sec, dtype=np.float64),
        "external_time_status": np.asarray(sync.external_time_status),
        "external_time_source": _optional_array(sync.external_time_source, np.dtype("U")),
        "external_time_variable_name": _optional_array(
            sync.external_time_variable_name,
            np.dtype("U"),
        ),
        "external_time_units": np.asarray(sync.external_time_units),
        "fps_header": np.asarray(sync.fps_header, dtype=np.float64),
        "raw_fps_effective": _optional_array(
            sync.raw_fps_effective,
            np.dtype(np.float64),
        ),
        "start_frame": np.asarray(sync.start_frame, dtype=np.int64),
        "end_frame_exclusive": _optional_array(
            sync.end_frame_exclusive,
            np.dtype(np.int64),
        ),
        "raw_frame_count_opencv_readable": _optional_array(
            sync.raw_frame_count_opencv_readable,
            np.dtype(np.int64),
        ),
        "prepared_frame_count_opencv_reported": np.asarray(
            sync.prepared_frame_count_opencv_reported,
            dtype=np.int64,
        ),
        "prepared_frame_count_opencv_readable": np.asarray(
            sync.prepared_frame_count_opencv_readable,
            dtype=np.int64,
        ),
        "frame_count_used_for_sleap": np.asarray(
            sync.frame_count_used_for_sleap,
            dtype=np.int64,
        ),
    }


def _required_scalar(name: str, array: np.ndarray) -> Any:
    if array.shape != ():
        raise SyncValidationError(f"NPZ field {name} must be a scalar.")
    return array.item()


def _optional_scalar(name: str, array: np.ndarray) -> Any | None:
    if array.size == 0:
        if array.ndim != 1:
            raise SyncValidationError(
                f"Null NPZ field {name} must be an empty one-dimensional array."
            )
        return None
    return _required_scalar(name, array)


def _load_sync_archive(path: Path) -> PreparedSyncData:
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual_keys = set(archive.files)
            expected_keys = set(PREPARED_SYNC_NPZ_KEYS)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                unexpected = sorted(actual_keys - expected_keys)
                raise SyncValidationError(
                    f"Invalid prepared sync NPZ keys; missing={missing}, "
                    f"unexpected={unexpected}."
                )
            values = {
                name: np.array(archive[name], copy=True)
                for name in PREPARED_SYNC_NPZ_KEYS
            }
    except SyncValidationError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise SyncValidationError(f"Could not load prepared sync NPZ {path}: {exc}") from exc

    try:
        sync = PreparedSyncData(
            sync_schema_version=str(
                _required_scalar("sync_schema_version", values["sync_schema_version"])
            ),
            prepared_frame_idx=_readonly_int_array(values["prepared_frame_idx"]),
            raw_decode_frame_idx=_readonly_int_array(values["raw_decode_frame_idx"]),
            prepared_time_sec=_readonly_float_array(values["prepared_time_sec"]),
            raw_pts_time_sec=_readonly_float_array(values["raw_pts_time_sec"]),
            raw_pts_status=str(
                _required_scalar("raw_pts_status", values["raw_pts_status"])
            ),
            external_time_sec=_readonly_float_array(values["external_time_sec"]),
            external_time_status=str(
                _required_scalar(
                    "external_time_status",
                    values["external_time_status"],
                )
            ),
            external_time_source=_optional_scalar(
                "external_time_source",
                values["external_time_source"],
            ),
            external_time_variable_name=_optional_scalar(
                "external_time_variable_name",
                values["external_time_variable_name"],
            ),
            external_time_units=str(
                _required_scalar("external_time_units", values["external_time_units"])
            ),
            fps_header=float(_required_scalar("fps_header", values["fps_header"])),
            raw_fps_effective=_optional_scalar(
                "raw_fps_effective",
                values["raw_fps_effective"],
            ),
            start_frame=int(_required_scalar("start_frame", values["start_frame"])),
            end_frame_exclusive=_optional_scalar(
                "end_frame_exclusive",
                values["end_frame_exclusive"],
            ),
            raw_frame_count_opencv_readable=_optional_scalar(
                "raw_frame_count_opencv_readable",
                values["raw_frame_count_opencv_readable"],
            ),
            prepared_frame_count_opencv_reported=int(
                _required_scalar(
                    "prepared_frame_count_opencv_reported",
                    values["prepared_frame_count_opencv_reported"],
                )
            ),
            prepared_frame_count_opencv_readable=int(
                _required_scalar(
                    "prepared_frame_count_opencv_readable",
                    values["prepared_frame_count_opencv_readable"],
                )
            ),
            frame_count_used_for_sleap=int(
                _required_scalar(
                    "frame_count_used_for_sleap",
                    values["frame_count_used_for_sleap"],
                )
            ),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise SyncValidationError(f"Prepared sync NPZ scalar metadata is invalid: {exc}") from exc
    validate_prepared_sync(sync)
    return sync


def load_prepared_sync_npz(path: Path) -> PreparedSyncData:
    """Load and strictly validate a prepared synchronization NPZ artifact.

    Raises:
        SyncValidationError: If the file is absent, unreadable, contains
            incorrect keys or unsafe object arrays, or violates sync invariants.
    """

    sync_path = Path(path).expanduser()
    if not sync_path.is_file():
        raise SyncValidationError(f"Prepared sync NPZ does not exist: {sync_path}")
    return _load_sync_archive(sync_path.resolve())


def _create_temporary_npz_path(destination: Path) -> Path:
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.sync-",
            suffix=".npz",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name).resolve()
        temporary_path.unlink()
    except OSError as exc:
        raise SyncValidationError(
            f"Could not create prepared sync temporary path: {exc}"
        ) from exc
    return temporary_path


def _remove_temporary_npz(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise SyncValidationError(
            f"Could not remove failed prepared sync temporary file {path}: {exc}"
        ) from exc


def write_prepared_sync_npz(path: Path, sync: PreparedSyncData) -> None:
    """Atomically write, reload, and validate a compressed prepared sync NPZ.

    The temporary file is created in the destination directory. Any existing
    destination remains untouched unless both in-memory and reloaded
    validation succeed.

    Raises:
        SyncValidationError: If validation, serialization, reload validation,
            temporary-file cleanup, or atomic replacement fails.
    """

    validate_prepared_sync(sync)
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        raise SyncValidationError("Prepared sync destination must use the .npz suffix.")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
    except OSError as exc:
        raise SyncValidationError(
            f"Could not create prepared sync destination directory: {exc}"
        ) from exc

    temporary_path = _create_temporary_npz_path(destination)
    try:
        np.savez_compressed(temporary_path, **_sync_payload(sync))
        load_prepared_sync_npz(temporary_path)
        temporary_path.replace(destination)
    except SyncValidationError:
        _remove_temporary_npz(temporary_path)
        raise
    except (OSError, TypeError, ValueError) as exc:
        _remove_temporary_npz(temporary_path)
        raise SyncValidationError(
            f"Could not write prepared sync NPZ {destination}: {exc}"
        ) from exc
