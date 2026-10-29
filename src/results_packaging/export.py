"""Atomic artifact writers for immutable Subsystem 09 package runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from results_packaging.contracts import PublicationError


def atomic_parquet_write(path: Path | str, table: pd.DataFrame) -> Path:
    """Write ``table`` to ``path`` via temporary file + atomic replace."""

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


def write_sync_index_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    if table is None or table.empty:
        raise PublicationError("Refusing to publish an empty sync_index table.")
    return atomic_parquet_write(path, table)


def write_behavior_bouts_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    if table is None:
        raise PublicationError("Refusing to publish a missing behavior_bouts table.")
    return atomic_parquet_write(path, table)


def write_ethogram_summary_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    if table is None:
        raise PublicationError("Refusing to publish a missing ethogram_summary table.")
    return atomic_parquet_write(path, table)


def write_bout_index_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    if table is None:
        raise PublicationError("Refusing to publish a missing bout_index table.")
    return atomic_parquet_write(path, table)


def write_exclusion_table_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    if table is None:
        raise PublicationError("Refusing to publish a missing exclusion_table.")
    return atomic_parquet_write(path, table)


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def write_yaml(path: Path | str, payload: dict[str, Any]) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def load_json_object(path: Path | str) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise PublicationError(f"JSON artifact missing: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Could not read JSON artifact {file_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicationError(f"JSON artifact is not an object: {file_path}")
    return payload
