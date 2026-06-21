from pathlib import Path
from typing import Any

import numpy as np
import pytest

from preprocess import sync_writer
from preprocess.exceptions import SyncValidationError
from preprocess.sync_writer import (
    PREPARED_SYNC_NPZ_KEYS,
    PREPARED_SYNC_SCHEMA_VERSION,
    PreparedSyncData,
    build_prepared_sync,
    load_prepared_sync_npz,
    validate_prepared_sync,
    write_prepared_sync_npz,
)


def _build_sync(**overrides: Any) -> PreparedSyncData:
    arguments: dict[str, Any] = {
        "prepared_frame_count": 4,
        "start_frame": 2,
        "end_frame_exclusive": 6,
        "fps_header": 20.0,
        "raw_fps_effective": 30.0,
        "raw_pts_time_sec": None,
        "raw_pts_status": "extraction_failed",
        "external_ttl_vector": None,
        "external_time_status": "not_provided",
        "external_time_source": None,
        "external_time_variable_name": None,
        "external_time_units": "unknown",
        "raw_frame_count_opencv_readable": 10,
        "prepared_frame_count_opencv_reported": 4,
        "prepared_frame_count_opencv_readable": 4,
    }
    arguments.update(overrides)
    return build_prepared_sync(**arguments)


def test_valid_no_ttl_sync_creation_uses_nan_timing() -> None:
    sync = _build_sync()

    assert sync.sync_schema_version == PREPARED_SYNC_SCHEMA_VERSION
    assert sync.external_time_status == "not_provided"
    assert sync.external_time_source is None
    assert sync.external_time_variable_name is None
    assert np.all(np.isnan(sync.raw_pts_time_sec))
    assert np.all(np.isnan(sync.external_time_sec))
    assert sync.frame_count_used_for_sleap == 4


def test_valid_external_ttl_sync_creation_after_trim() -> None:
    external_timing_seconds = np.arange(10, dtype=np.float64) * 0.1

    sync = _build_sync(
        external_ttl_vector=external_timing_seconds,
        external_time_status="valid",
        external_time_source="timing.mat",
        external_time_variable_name="camera_ticks",
        external_time_units="seconds",
    )

    np.testing.assert_allclose(sync.external_time_sec, [0.2, 0.3, 0.4, 0.5])
    assert sync.external_time_status == "valid"
    assert sync.external_time_source == "timing.mat"
    assert sync.external_time_variable_name == "camera_ticks"


def test_prepared_and_raw_decode_frame_mapping_is_exact() -> None:
    sync = _build_sync()

    np.testing.assert_array_equal(sync.prepared_frame_idx, [0, 1, 2, 3])
    np.testing.assert_array_equal(sync.raw_decode_frame_idx, [2, 3, 4, 5])
    assert np.issubdtype(sync.prepared_frame_idx.dtype, np.integer)
    assert np.issubdtype(sync.raw_decode_frame_idx.dtype, np.integer)


def test_prepared_time_uses_fps_header() -> None:
    sync = _build_sync(fps_header=20.0)

    np.testing.assert_allclose(sync.prepared_time_sec, [0.0, 0.05, 0.1, 0.15])
    assert np.all(np.isfinite(sync.prepared_time_sec))


def test_supplied_raw_pts_must_correspond_to_prepared_frames() -> None:
    raw_pts = np.array([1.0, 1.04, np.nan, 1.12])
    sync = _build_sync(raw_pts_time_sec=raw_pts, raw_pts_status="partially_missing")

    np.testing.assert_array_equal(sync.raw_pts_time_sec, raw_pts)
    assert sync.raw_pts_status == "partially_missing"


@pytest.mark.parametrize("units", ["frames", "unknown"])
def test_non_temporal_external_units_do_not_invent_seconds(units: str) -> None:
    sync = _build_sync(
        external_ttl_vector=np.arange(10, dtype=np.float64),
        external_time_status="valid",
        external_time_source="timing.mat",
        external_time_variable_name="frame_counter",
        external_time_units=units,
    )

    assert sync.external_time_status == "not_convertible_to_seconds"
    assert np.all(np.isnan(sync.external_time_sec))


def test_external_ttl_vector_must_be_one_dimensional() -> None:
    with pytest.raises(SyncValidationError, match="one-dimensional"):
        _build_sync(
            external_ttl_vector=np.arange(10, dtype=np.float64).reshape(5, 2),
            external_time_status="valid",
            external_time_units="seconds",
        )


def test_external_ttl_length_must_match_raw_readable_count() -> None:
    with pytest.raises(SyncValidationError, match="length must equal"):
        _build_sync(
            external_ttl_vector=np.arange(9, dtype=np.float64),
            external_time_status="valid",
            external_time_units="seconds",
        )


def test_non_finite_external_ttl_fails() -> None:
    timing = np.arange(10, dtype=np.float64)
    timing[5] = np.nan

    with pytest.raises(SyncValidationError, match="finite"):
        _build_sync(
            external_ttl_vector=timing,
            external_time_status="valid",
            external_time_units="seconds",
        )


@pytest.mark.parametrize(
    "timing",
    [
        np.array([0.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        np.array([0.0, 1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    ],
)
def test_non_monotonic_external_ttl_fails(timing: np.ndarray) -> None:
    with pytest.raises(SyncValidationError, match="strictly increasing"):
        _build_sync(
            external_ttl_vector=timing,
            external_time_status="valid",
            external_time_units="seconds",
        )


def test_out_of_range_raw_decode_index_fails() -> None:
    with pytest.raises(SyncValidationError, match="outside the raw readable"):
        _build_sync(
            start_frame=3,
            end_frame_exclusive=7,
            raw_frame_count_opencv_readable=5,
        )


def test_prepared_reported_readable_mismatch_fails() -> None:
    with pytest.raises(SyncValidationError, match="reported frame count"):
        _build_sync(prepared_frame_count_opencv_reported=5)


def test_prepared_readable_expected_count_mismatch_fails() -> None:
    with pytest.raises(SyncValidationError, match="prepared_frame_count must equal"):
        _build_sync(prepared_frame_count=5)


def test_prepared_readable_end_frame_mismatch_fails() -> None:
    with pytest.raises(SyncValidationError, match="end_frame_exclusive"):
        _build_sync(end_frame_exclusive=7)


@pytest.mark.parametrize("fps_header", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fps_header_fails(fps_header: float) -> None:
    with pytest.raises(SyncValidationError, match="finite and positive"):
        _build_sync(fps_header=fps_header)


def test_raw_pts_length_mismatch_fails() -> None:
    with pytest.raises(SyncValidationError, match="raw_pts_time_sec length"):
        _build_sync(raw_pts_time_sec=np.array([0.0, 0.1, 0.2]))


def test_validation_rejects_frame_count_used_for_sleap_mismatch() -> None:
    invalid = _build_sync().model_copy(update={"frame_count_used_for_sleap": 3})

    with pytest.raises(SyncValidationError, match="frame_count_used_for_sleap"):
        validate_prepared_sync(invalid)


def test_validation_rejects_inconsistent_array_length() -> None:
    invalid = _build_sync().model_copy(
        update={"external_time_sec": np.full(3, np.nan)}
    )

    with pytest.raises(SyncValidationError, match="external_time_sec length"):
        validate_prepared_sync(invalid)


def test_validation_rejects_changed_decode_order_mapping() -> None:
    invalid = _build_sync().model_copy(
        update={"raw_decode_frame_idx": np.array([2, 3, 5, 6], dtype=np.int64)}
    )

    with pytest.raises(SyncValidationError, match="start_frame"):
        validate_prepared_sync(invalid)


def test_validation_rejects_invalid_marked_valid_external_time() -> None:
    valid = _build_sync(
        external_ttl_vector=np.arange(10, dtype=np.float64),
        external_time_status="valid",
        external_time_units="seconds",
    )
    invalid = valid.model_copy(
        update={"external_time_sec": np.array([2.0, 3.0, np.nan, 5.0])}
    )

    with pytest.raises(SyncValidationError, match="finite"):
        validate_prepared_sync(invalid)


def test_npz_round_trip_preserves_arrays_and_scalar_metadata(tmp_path: Path) -> None:
    sync = _build_sync(
        raw_pts_time_sec=np.array([0.2, 0.3, 0.4, 0.5]),
        raw_pts_status="valid",
        external_ttl_vector=np.arange(10, dtype=np.float64) * 0.1,
        external_time_status="valid",
        external_time_source="timing.mat",
        external_time_variable_name="camera_ticks",
        external_time_units="seconds",
    )
    path = tmp_path / "prepared_sync.npz"

    write_prepared_sync_npz(path, sync)
    loaded = load_prepared_sync_npz(path)

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(PREPARED_SYNC_NPZ_KEYS)
    np.testing.assert_array_equal(loaded.prepared_frame_idx, sync.prepared_frame_idx)
    np.testing.assert_array_equal(loaded.raw_decode_frame_idx, sync.raw_decode_frame_idx)
    np.testing.assert_allclose(loaded.prepared_time_sec, sync.prepared_time_sec)
    np.testing.assert_allclose(loaded.raw_pts_time_sec, sync.raw_pts_time_sec)
    np.testing.assert_allclose(loaded.external_time_sec, sync.external_time_sec)
    assert loaded.model_dump(exclude={
        "prepared_frame_idx",
        "raw_decode_frame_idx",
        "prepared_time_sec",
        "raw_pts_time_sec",
        "external_time_sec",
    }) == sync.model_dump(exclude={
        "prepared_frame_idx",
        "raw_decode_frame_idx",
        "prepared_time_sec",
        "raw_pts_time_sec",
        "external_time_sec",
    })


def test_npz_round_trip_preserves_null_optional_metadata(tmp_path: Path) -> None:
    sync = _build_sync(
        end_frame_exclusive=None,
        raw_frame_count_opencv_readable=None,
        raw_fps_effective=None,
    )
    path = tmp_path / "prepared_sync.npz"

    write_prepared_sync_npz(path, sync)
    loaded = load_prepared_sync_npz(path)

    assert loaded.end_frame_exclusive is None
    assert loaded.raw_frame_count_opencv_readable is None
    assert loaded.raw_fps_effective is None
    assert loaded.external_time_source is None
    assert loaded.external_time_variable_name is None


def test_atomic_write_failure_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "prepared_sync.npz"
    existing_content = b"existing validated sync artifact"
    destination.write_bytes(existing_content)

    def reject_reload(_path: Path) -> PreparedSyncData:
        raise SyncValidationError("forced reload validation failure")

    monkeypatch.setattr(sync_writer, "load_prepared_sync_npz", reject_reload)

    with pytest.raises(SyncValidationError, match="forced reload"):
        write_prepared_sync_npz(destination, _build_sync())

    assert destination.read_bytes() == existing_content
    assert list(tmp_path.glob(".prepared_sync.sync-*.npz")) == []
