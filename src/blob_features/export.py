"""``blob_features.parquet`` table construction and safe write helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from blob_features.contracts import (
    BLOB_FEATURES_COLUMNS,
    BLOB_FEATURES_SCHEMA_VERSION,
    BlobFeaturesError,
)
from blob_features.timing import NearestNeighborTimingMap


def build_blob_features_table(
    mapped_features: pd.DataFrame,
    timing_map: NearestNeighborTimingMap,
) -> pd.DataFrame:
    """Assemble the authoritative S5 parquet table from mapped feature rows."""

    if len(mapped_features) != timing_map.target_sample_count:
        raise BlobFeaturesError(
            "Mapped feature row count does not match the 120 Hz target sample count: "
            f"{len(mapped_features)} != {timing_map.target_sample_count}."
        )

    sample_idx = np.asarray(timing_map.target_frame_idx, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "schema_version": BLOB_FEATURES_SCHEMA_VERSION,
            "sample_idx": sample_idx,
            "dense_frame_idx": sample_idx.copy(),
            "frame_idx": sample_idx.copy(),
            "time_sec": np.asarray(timing_map.target_time_sec, dtype=np.float64),
            "orig_frame_idx": np.asarray(timing_map.orig_frame_idx, dtype=np.int64),
        }
    )

    for col in BLOB_FEATURES_COLUMNS:
        if col in (
            "schema_version",
            "sample_idx",
            "dense_frame_idx",
            "frame_idx",
            "time_sec",
            "orig_frame_idx",
        ):
            continue
        if col not in mapped_features.columns:
            raise BlobFeaturesError(
                f"Mapped features are missing required column '{col}'."
            )
        frame[col] = mapped_features[col].to_numpy()

    # Prefer timing-map values for index columns when mapped table also carries them.
    for name, values in (
        ("sample_idx", sample_idx),
        ("dense_frame_idx", sample_idx),
        ("frame_idx", sample_idx),
        ("time_sec", np.asarray(timing_map.target_time_sec, dtype=np.float64)),
        ("orig_frame_idx", np.asarray(timing_map.orig_frame_idx, dtype=np.int64)),
    ):
        if name in mapped_features.columns:
            mapped_vals = mapped_features[name].to_numpy()
            if not np.array_equal(mapped_vals, values):
                raise BlobFeaturesError(
                    f"Mapped '{name}' does not match the authoritative timing map."
                )

    missing = [name for name in BLOB_FEATURES_COLUMNS if name not in frame.columns]
    if missing:
        raise BlobFeaturesError(
            "blob_features table is missing columns: " + ", ".join(missing)
        )
    return frame.loc[:, list(BLOB_FEATURES_COLUMNS)]


def write_blob_features_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Write ``blob_features.parquet`` after schema validation (atomic replace)."""

    if list(table.columns) != list(BLOB_FEATURES_COLUMNS):
        raise BlobFeaturesError(
            "blob_features table columns do not match the authoritative schema."
        )
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        table.to_parquet(tmp, index=False)
        tmp.replace(out)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return out


def load_blob_features_table(path: Path | str) -> pd.DataFrame:
    """Load a previously written blob_features.parquet."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise BlobFeaturesError(f"blob_features.parquet is missing: {file_path}")
    try:
        frame = pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001 — surface as S5 error
        raise BlobFeaturesError(
            f"Could not read blob_features.parquet: {file_path}: {exc}"
        ) from exc
    missing = [name for name in BLOB_FEATURES_COLUMNS if name not in frame.columns]
    if missing:
        raise BlobFeaturesError(
            "blob_features.parquet is missing columns: " + ", ".join(missing)
        )
    return frame.loc[:, list(BLOB_FEATURES_COLUMNS)]
