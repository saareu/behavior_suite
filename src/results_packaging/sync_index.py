"""Build and validate the S9 ``sync_index`` (dense ↔ prepared mapping).

Authority: S1 ``prepared_sync.npz`` + S5 nearest-neighbor mapping
(``blob_features.timing``), earlier-frame tie-break, no silent clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from blob_features.contracts import BlobFeaturesError
from blob_features.timing import (
    NearestNeighborTimingMap,
    build_nearest_neighbor_timing_map,
    build_nearest_neighbor_timing_map_from_prepared_sync,
)
from preprocess.exceptions import SyncValidationError
from preprocess.sync_writer import load_prepared_sync_npz
from results_packaging.contracts import (
    SYNC_INDEX_REQUIRED_COLUMNS,
    SYNC_SCHEMA_VERSION,
    TARGET_FPS,
    TIME_ATOL_SEC,
    SyncIndexError,
)


@dataclass(frozen=True, slots=True)
class SyncIndexBuildResult:
    """In-memory synchronization index plus audit diagnostics."""

    sync_index: pd.DataFrame
    prepared_sync_path: Path
    n_dense: int
    n_prepared_source: int
    n_prepared_unique: int
    prepared_frame_idx_min: int
    prepared_frame_idx_max: int
    max_many_to_one: int
    n_exact_matches: int
    timing_residual_min: float
    timing_residual_max: float
    timing_residual_abs_max: float
    timing_residual_mean: float
    monotonic_non_decreasing: bool
    optional_raw_columns_present: bool
    warnings: tuple[str, ...] = ()

    def audit_dict(self) -> dict[str, Any]:
        return {
            "n_dense": self.n_dense,
            "n_prepared_source": self.n_prepared_source,
            "n_prepared_unique": self.n_prepared_unique,
            "prepared_frame_idx_min": self.prepared_frame_idx_min,
            "prepared_frame_idx_max": self.prepared_frame_idx_max,
            "max_many_to_one": self.max_many_to_one,
            "n_exact_matches": self.n_exact_matches,
            "timing_residual_min": self.timing_residual_min,
            "timing_residual_max": self.timing_residual_max,
            "timing_residual_abs_max": self.timing_residual_abs_max,
            "timing_residual_mean": self.timing_residual_mean,
            "monotonic_non_decreasing": self.monotonic_non_decreasing,
            "optional_raw_columns_present": self.optional_raw_columns_present,
            "warnings": list(self.warnings),
        }


def build_sync_index_from_prepared_sync(
    prepared_sync_path: Path | str,
    *,
    s8_unique_dense_times: pd.DataFrame,
    package_run_id: str | None = None,
    target_fps: float = TARGET_FPS,
) -> SyncIndexBuildResult:
    """Construct one sync row per S8 dense sample using S5 NN mapping.

    ``s8_unique_dense_times`` must contain unique ``dense_frame_idx`` /
    ``sample_idx`` / ``time_sec`` sorted ascending.
    """

    sync_path = Path(prepared_sync_path).expanduser().resolve()
    if not sync_path.is_file():
        raise SyncIndexError(f"Prepared sync missing: {sync_path}")

    unique = _normalize_unique_dense(s8_unique_dense_times)
    s8_times = unique["time_sec"].to_numpy(dtype=np.float64)
    s8_idx = unique["dense_frame_idx"].to_numpy(dtype=np.int64)

    try:
        nn_map = build_nearest_neighbor_timing_map_from_prepared_sync(
            sync_path, target_fps=float(target_fps)
        )
    except BlobFeaturesError as exc:
        raise SyncIndexError(f"Nearest-neighbor sync map failed: {exc}") from exc

    _assert_s8_times_match_nn_timeline(s8_idx=s8_idx, s8_times=s8_times, nn_map=nn_map)

    return _assemble_sync_index(
        nn_map=nn_map,
        prepared_sync_path=sync_path,
        package_run_id=package_run_id,
    )


def build_sync_index_from_source_arrays(
    *,
    source_frame_idx: np.ndarray,
    source_time_sec: np.ndarray,
    target_time_sec: np.ndarray,
    prepared_sync_path: Path | str | None = None,
    package_run_id: str | None = None,
    target_fps: float = TARGET_FPS,
    source_fps_header: float | None = None,
    raw_decode_frame_idx: np.ndarray | None = None,
    raw_pts_time_sec: np.ndarray | None = None,
) -> SyncIndexBuildResult:
    """Build a sync index from explicit arrays (unit tests / synthetic maps)."""

    try:
        nn_map = build_nearest_neighbor_timing_map(
            source_frame_idx=source_frame_idx,
            source_time_sec=source_time_sec,
            target_time_sec=target_time_sec,
            target_fps=float(target_fps),
            source_fps_header=source_fps_header,
            prepared_sync_path=(
                Path(prepared_sync_path).resolve() if prepared_sync_path else None
            ),
        )
    except BlobFeaturesError as exc:
        raise SyncIndexError(f"Nearest-neighbor sync map failed: {exc}") from exc

    return _assemble_sync_index(
        nn_map=nn_map,
        prepared_sync_path=(
            Path(prepared_sync_path).resolve()
            if prepared_sync_path is not None
            else Path(".")
        ),
        package_run_id=package_run_id,
        raw_decode_frame_idx=raw_decode_frame_idx,
        raw_pts_time_sec=raw_pts_time_sec,
        load_raw_from_sync=False,
    )


def validate_sync_index(sync_index: pd.DataFrame) -> None:
    """Structural validation of a sync_index table."""

    if sync_index is None or sync_index.empty:
        raise SyncIndexError("sync_index is empty.")
    missing = [c for c in SYNC_INDEX_REQUIRED_COLUMNS if c not in sync_index.columns]
    if missing:
        raise SyncIndexError(f"sync_index missing required columns: {missing}")

    sample = sync_index["sample_idx"].to_numpy(dtype=np.int64)
    dense = sync_index["dense_frame_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample, dense):
        raise SyncIndexError("sample_idx must equal dense_frame_idx.")
    if sync_index["sample_idx"].duplicated().any():
        raise SyncIndexError("sample_idx must be unique.")
    expected = np.arange(sample.shape[0], dtype=np.int64)
    if not np.array_equal(sample, expected):
        raise SyncIndexError("sample_idx must be contiguous 0 … N-1 in sorted order.")

    prepared = sync_index["prepared_frame_idx"].to_numpy(dtype=np.int64)
    if prepared.shape[0] > 1 and not np.all(np.diff(prepared) >= 0):
        raise SyncIndexError("prepared_frame_idx must be monotonically non-decreasing.")

    residuals = sync_index["timing_residual_sec"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(residuals)):
        raise SyncIndexError("timing_residual_sec must be finite.")

    exact = sync_index["is_exact_source_sample"].to_numpy(dtype=bool)
    prepared_times = sync_index["prepared_time_sec"].to_numpy(dtype=np.float64)
    times = sync_index["time_sec"].to_numpy(dtype=np.float64)
    recomputed_exact = np.abs(times - prepared_times) <= TIME_ATOL_SEC
    if not np.array_equal(exact, recomputed_exact):
        raise SyncIndexError(
            "is_exact_source_sample inconsistent with |time_sec - prepared_time_sec|."
        )
    residual_delta = np.abs(residuals - (times - prepared_times))
    if float(np.max(residual_delta)) > TIME_ATOL_SEC:
        raise SyncIndexError(
            "timing_residual_sec must equal time_sec - prepared_time_sec."
        )


def _assemble_sync_index(
    *,
    nn_map: NearestNeighborTimingMap,
    prepared_sync_path: Path,
    package_run_id: str | None,
    raw_decode_frame_idx: np.ndarray | None = None,
    raw_pts_time_sec: np.ndarray | None = None,
    load_raw_from_sync: bool = True,
) -> SyncIndexBuildResult:
    warnings: list[str] = []
    n = int(nn_map.target_sample_count)
    sample_idx = np.asarray(nn_map.target_frame_idx, dtype=np.int64)
    time_sec = np.asarray(nn_map.target_time_sec, dtype=np.float64)
    prepared_frame_idx = np.asarray(nn_map.orig_frame_idx, dtype=np.int64)
    prepared_time_sec = np.asarray(nn_map.source_time_sec, dtype=np.float64)
    is_exact = np.asarray(nn_map.is_exact_source_sample, dtype=bool)
    residual = time_sec - prepared_time_sec

    if n > 1 and not np.all(np.diff(prepared_frame_idx) >= 0):
        raise SyncIndexError("Nearest-neighbor prepared_frame_idx is non-monotonic.")

    frame = {
        "schema_version": np.full(n, SYNC_SCHEMA_VERSION, dtype=object),
        "sample_idx": sample_idx,
        "dense_frame_idx": sample_idx.copy(),
        "time_sec": time_sec,
        "prepared_frame_idx": prepared_frame_idx,
        "prepared_time_sec": prepared_time_sec,
        "is_exact_source_sample": is_exact,
        "timing_residual_sec": residual,
    }

    optional_raw = False
    raw_decode = raw_decode_frame_idx
    raw_pts = raw_pts_time_sec
    if load_raw_from_sync and prepared_sync_path.is_file():
        try:
            sync = load_prepared_sync_npz(prepared_sync_path)
            raw_decode = np.asarray(sync.raw_decode_frame_idx, dtype=np.int64)[
                prepared_frame_idx
            ]
            raw_pts = np.asarray(sync.raw_pts_time_sec, dtype=np.float64)[
                prepared_frame_idx
            ]
        except (SyncValidationError, OSError, IndexError, ValueError) as exc:
            warnings.append(f"Optional raw/decode columns unavailable: {exc}")
            raw_decode = None
            raw_pts = None

    if raw_decode is not None:
        if int(np.asarray(raw_decode).shape[0]) != n:
            raise SyncIndexError("raw_decode_frame_idx length must match dense samples.")
        frame["raw_decode_frame_idx"] = np.asarray(raw_decode, dtype=np.int64)
        optional_raw = True
    if raw_pts is not None:
        if int(np.asarray(raw_pts).shape[0]) != n:
            raise SyncIndexError("raw_pts_time_sec length must match dense samples.")
        frame["raw_pts_time_sec"] = np.asarray(raw_pts, dtype=np.float64)
        optional_raw = True
    if package_run_id is not None:
        frame["package_run_id"] = np.full(n, str(package_run_id), dtype=object)

    sync_index = pd.DataFrame(frame)
    # Deterministic sort / unique dense keys.
    sync_index = sync_index.sort_values("sample_idx", kind="mergesort").reset_index(
        drop=True
    )
    validate_sync_index(sync_index)

    unique_prepared = int(pd.unique(prepared_frame_idx).shape[0])
    counts = pd.Series(prepared_frame_idx).value_counts()
    max_many = int(counts.max()) if not counts.empty else 0
    if unique_prepared != n:
        # Expected when dense length differs from prepared length.
        pass

    return SyncIndexBuildResult(
        sync_index=sync_index,
        prepared_sync_path=prepared_sync_path,
        n_dense=n,
        n_prepared_source=int(nn_map.source_frame_count),
        n_prepared_unique=unique_prepared,
        prepared_frame_idx_min=int(prepared_frame_idx.min()) if n else -1,
        prepared_frame_idx_max=int(prepared_frame_idx.max()) if n else -1,
        max_many_to_one=max_many,
        n_exact_matches=int(np.count_nonzero(is_exact)),
        timing_residual_min=float(np.min(residual)) if n else 0.0,
        timing_residual_max=float(np.max(residual)) if n else 0.0,
        timing_residual_abs_max=float(np.max(np.abs(residual))) if n else 0.0,
        timing_residual_mean=float(np.mean(residual)) if n else 0.0,
        monotonic_non_decreasing=True,
        optional_raw_columns_present=optional_raw,
        warnings=tuple(warnings),
    )


def _normalize_unique_dense(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise SyncIndexError("S8 unique dense timeline is empty.")
    required = {"dense_frame_idx", "time_sec"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SyncIndexError(f"Unique dense timeline missing columns: {missing}")
    out = frame.copy()
    if "sample_idx" not in out.columns:
        out["sample_idx"] = out["dense_frame_idx"]
    out = out.sort_values("dense_frame_idx", kind="mergesort").reset_index(drop=True)
    if out["dense_frame_idx"].duplicated().any():
        raise SyncIndexError("Unique dense timeline has duplicate dense_frame_idx.")
    dense = out["dense_frame_idx"].to_numpy(dtype=np.int64)
    sample = out["sample_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(dense, sample):
        raise SyncIndexError("sample_idx must equal dense_frame_idx in unique timeline.")
    expected = np.arange(dense.shape[0], dtype=np.int64)
    if not np.array_equal(dense, expected):
        raise SyncIndexError("dense_frame_idx must be contiguous 0 … N-1.")
    times = out["time_sec"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise SyncIndexError("Unique dense time_sec must be finite.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise SyncIndexError("Unique dense time_sec must be strictly increasing.")
    return out


def _assert_s8_times_match_nn_timeline(
    *,
    s8_idx: np.ndarray,
    s8_times: np.ndarray,
    nn_map: NearestNeighborTimingMap,
) -> None:
    if int(s8_idx.shape[0]) != int(nn_map.target_sample_count):
        raise SyncIndexError(
            "S8 dense sample count does not match rebuilt 120 Hz target timeline "
            f"({s8_idx.shape[0]} != {nn_map.target_sample_count})."
        )
    if not np.array_equal(s8_idx, np.asarray(nn_map.target_frame_idx, dtype=np.int64)):
        raise SyncIndexError("S8 dense_frame_idx does not match rebuilt target indices.")
    deltas = np.abs(s8_times - np.asarray(nn_map.target_time_sec, dtype=np.float64))
    if float(np.max(deltas)) > TIME_ATOL_SEC:
        raise SyncIndexError(
            "S8 time_sec does not match rebuilt uniform target timeline within 1e-9 s."
        )
