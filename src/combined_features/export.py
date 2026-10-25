"""``dyadic_features_full.parquet`` table construction and safe write helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.contracts import (
    DYADIC_FEATURES_SCHEMA_VERSION,
    REQUIRED_IDENTITY_COLUMNS,
    CombinedFeaturesError,
)


def stamp_identity_columns(
    features: pd.DataFrame,
    *,
    session_root: Path | str,
    source_s4_run_id: str,
    source_s5_run_id: str,
) -> pd.DataFrame:
    """Attach required identity / provenance columns without changing science."""

    out = features.copy()
    n = len(out)
    out["schema_version"] = DYADIC_FEATURES_SCHEMA_VERSION
    out["session_root"] = str(Path(session_root).expanduser().resolve())
    out["source_s4_run_id"] = str(source_s4_run_id)
    out["source_s5_run_id"] = str(source_s5_run_id)

    if "dense_frame_idx" not in out.columns:
        raise CombinedFeaturesError("Feature table is missing dense_frame_idx.")
    if "sample_idx" not in out.columns:
        out["sample_idx"] = out["dense_frame_idx"].astype(np.int64)
    else:
        out["sample_idx"] = out["sample_idx"].astype(np.int64)
    out["dense_frame_idx"] = out["dense_frame_idx"].astype(np.int64)

    if "time_sec" not in out.columns:
        if "t_120" in out.columns:
            out["time_sec"] = out["t_120"].astype(np.float64)
        else:
            raise CombinedFeaturesError("Feature table is missing time_sec / t_120.")
    else:
        out["time_sec"] = out["time_sec"].astype(np.float64)

    if "orig_frame_idx" not in out.columns:
        raise CombinedFeaturesError(
            "Feature table is missing orig_frame_idx (must come from S5 via adapter)."
        )
    out["orig_frame_idx"] = out["orig_frame_idx"].astype(np.int64)

    for col in ("focal_id", "partner_id"):
        if col not in out.columns:
            raise CombinedFeaturesError(f"Feature table is missing required column '{col}'.")
        out[col] = out[col].astype(np.int64)

    if n == 0:
        raise CombinedFeaturesError("Cannot stamp identity columns on an empty feature table.")

    return out


def order_dyadic_feature_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Deterministic column order: required identity first, then remaining stable order."""

    missing = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in table.columns]
    if missing:
        raise CombinedFeaturesError(
            "dyadic features table is missing required identity columns: "
            + ", ".join(missing)
        )
    remainder = [c for c in table.columns if c not in REQUIRED_IDENTITY_COLUMNS]
    ordered = list(REQUIRED_IDENTITY_COLUMNS) + remainder
    return table.loc[:, ordered]


def build_dyadic_features_table(
    features: pd.DataFrame,
    *,
    session_root: Path | str,
    source_s4_run_id: str,
    source_s5_run_id: str,
) -> pd.DataFrame:
    """Stamp identity columns and enforce deterministic column ordering."""

    stamped = stamp_identity_columns(
        features,
        session_root=session_root,
        source_s4_run_id=source_s4_run_id,
        source_s5_run_id=source_s5_run_id,
    )
    return order_dyadic_feature_columns(stamped)


def write_dyadic_features_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Write ``dyadic_features_full.parquet`` after identity-schema validation (atomic)."""

    missing = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in table.columns]
    if missing:
        raise CombinedFeaturesError(
            "Refusing to publish parquet missing identity columns: "
            + ", ".join(missing)
        )
    if list(table.columns[: len(REQUIRED_IDENTITY_COLUMNS)]) != list(
        REQUIRED_IDENTITY_COLUMNS
    ):
        raise CombinedFeaturesError(
            "dyadic_features_full.parquet identity columns are not in contract order."
        )
    if table.empty:
        raise CombinedFeaturesError("Refusing to publish an empty dyadic features table.")

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


def load_dyadic_features_table(path: Path | str) -> pd.DataFrame:
    """Load a previously written dyadic_features_full.parquet."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise CombinedFeaturesError(f"dyadic_features_full.parquet is missing: {file_path}")
    try:
        frame = pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001 — surface as S6 error
        raise CombinedFeaturesError(
            f"Could not read dyadic_features_full.parquet: {file_path}: {exc}"
        ) from exc
    missing = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in frame.columns]
    if missing:
        raise CombinedFeaturesError(
            "dyadic_features_full.parquet is missing identity columns: "
            + ", ".join(missing)
        )
    return frame


def provenance_digest_block(
    *,
    path: Path,
    sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Small helper for run_meta source blocks."""

    payload: dict[str, Any] = {"path": str(Path(path).resolve())}
    if sha256 is not None:
        payload["sha256"] = sha256
    if extra:
        payload.update(dict(extra))
    return payload


def assert_unique_directional_keys(table: pd.DataFrame) -> None:
    """Hard-fail when directional keys collide."""

    key_cols: Sequence[str] = ("focal_id", "partner_id", "dense_frame_idx")
    missing = [c for c in key_cols if c not in table.columns]
    if missing:
        raise CombinedFeaturesError(
            "Cannot validate directional keys; missing columns: " + ", ".join(missing)
        )
    if table.duplicated(subset=list(key_cols)).any():
        raise CombinedFeaturesError(
            "Directional feature table has duplicate "
            "(focal_id, partner_id, dense_frame_idx) keys."
        )
