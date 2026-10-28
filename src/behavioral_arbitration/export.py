"""Framewise state / bout / diagnostics table construction and atomic Parquet writes.

Scientific arbitration remains in runner / temporal stages. This module validates,
stamps provenance, and publishes artifacts for immutable S8 runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    EXCLUSIVE_ETHOGRAM_LABELS,
    PARALLEL_CHANNELS,
    STAGE_DIAGNOSTICS_SCHEMA_VERSION,
    STATES_SCHEMA_VERSION,
    PublicationError,
)
from behavioral_arbitration.results import StageDiagnostic

_IDENTITY_REQUIRED: tuple[str, ...] = (
    "dense_frame_idx",
    "focal_id",
    "partner_id",
    "behavior_label",
)

_IDENTITY_PREFERRED: tuple[str, ...] = (
    "sample_idx",
    "time_sec",
    "exp_name",
    "mouse_a",
    "mouse_b",
    "pair_id",
)

_META_STAMP: tuple[str, ...] = (
    "schema_version",
    "source_s7_run_id",
    "source_s6_run_id",
    "session_root",
    "run_id",
)

_TRACE_COLS: tuple[str, ...] = (
    "label_reconstruction",
    "undefined_reason",
    "hierarchy_residual_kind",
)


def build_behavior_states_table(
    frames: pd.DataFrame,
    *,
    session_root: Path | str,
    source_s7_run_id: str,
    source_s6_run_id: str,
    run_id: str,
) -> pd.DataFrame:
    """Stamp schema / provenance and ensure required identity columns for publish."""

    if frames is None or frames.empty:
        raise PublicationError("Refusing to build an empty behavior_states table.")

    out = frames.copy().reset_index(drop=True)
    missing = [c for c in _IDENTITY_REQUIRED if c not in out.columns]
    if missing:
        raise PublicationError(
            "behavior_states missing required columns: " + ", ".join(missing)
        )

    if "sample_idx" not in out.columns:
        out["sample_idx"] = out["dense_frame_idx"]
    if "time_sec" not in out.columns:
        out["time_sec"] = pd.NA
    if "exp_name" not in out.columns:
        out["exp_name"] = ""
    if "mouse_a" not in out.columns or "mouse_b" not in out.columns:
        out["mouse_a"] = out[["focal_id", "partner_id"]].min(axis=1).astype("int64")
        out["mouse_b"] = out[["focal_id", "partner_id"]].max(axis=1).astype("int64")
    if "pair_id" not in out.columns:
        out["pair_id"] = (
            out["mouse_a"].astype(str) + "|" + out["mouse_b"].astype(str)
        )
    for col in _TRACE_COLS:
        if col not in out.columns:
            out[col] = ""

    out["schema_version"] = STATES_SCHEMA_VERSION
    out["source_s7_run_id"] = str(source_s7_run_id)
    out["source_s6_run_id"] = str(source_s6_run_id)
    out["session_root"] = str(Path(session_root).expanduser().resolve())
    out["run_id"] = str(run_id)

    return _order_states_columns(out)


def _order_states_columns(table: pd.DataFrame) -> pd.DataFrame:
    ordered: list[str] = []
    for col in (*_META_STAMP, *_IDENTITY_PREFERRED, *_IDENTITY_REQUIRED):
        if col in table.columns and col not in ordered:
            ordered.append(col)
    for label in EXCLUSIVE_ETHOGRAM_LABELS:
        for prefix in ("pred_", "prob_"):
            name = f"{prefix}{label}"
            if name in table.columns and name not in ordered:
                ordered.append(name)
    for channel in PARALLEL_CHANNELS:
        for prefix in ("pred_", "prob_"):
            name = f"{prefix}{channel}"
            if name in table.columns and name not in ordered:
                ordered.append(name)
    for col in table.columns:
        if (
            col.startswith(
                (
                    "input_valid_",
                    "score_origin_",
                    "arbitration_input_valid_",
                    "invalid_reason_",
                    "threshold_",
                    "s7_",
                )
            )
            and col not in ordered
        ):
            ordered.append(col)
    for col in _TRACE_COLS:
        if col in table.columns and col not in ordered:
            ordered.append(col)
    remainder = [c for c in table.columns if c not in ordered]
    return table.loc[:, ordered + remainder]


def write_behavior_states_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Atomic write; refuses empty states tables."""

    if table.empty:
        raise PublicationError(f"Refusing to publish empty behavior_states parquet: {path}")
    return _atomic_parquet_write(path, table)


def write_behavior_bouts_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Atomic write for bout tables (empty allowed only when scientifically empty)."""

    # Empty exclusive+parallel is unusual after a valid states publish; still allow
    # writing an empty schema-shaped table so discovery can see the artifact.
    return _atomic_parquet_write(path, table)


def write_optional_diagnostics_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Atomic write for optional diagnostics (empty table allowed)."""

    return _atomic_parquet_write(path, table)


def build_arbitration_diagnostics_table(
    diagnostics: Sequence[StageDiagnostic] | None = None,
    frame_events: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compact parquet-ready diagnostics from stage summaries (+ optional events)."""

    rows: list[dict[str, Any]] = []
    for diag in diagnostics or ():
        payload = diag.to_dict() if isinstance(diag, StageDiagnostic) else dict(diag)
        rows.append(
            {
                "schema_version": STAGE_DIAGNOSTICS_SCHEMA_VERSION,
                "record_kind": "stage_summary",
                "stage": payload.get("stage"),
                "rule_code": payload.get("rule_code"),
                "detector": payload.get("detector"),
                "n_rows_changed": int(payload.get("n_rows_changed") or 0),
                "detail": payload.get("detail"),
                "dense_frame_idx": pd.NA,
                "focal_id": pd.NA,
                "partner_id": pd.NA,
                "before_label": pd.NA,
                "after_label": pd.NA,
            }
        )

    if frame_events is not None:
        if isinstance(frame_events, pd.DataFrame):
            event_rows = frame_events.to_dict(orient="records")
        elif isinstance(frame_events, Mapping):
            event_rows = [dict(frame_events)]
        else:
            event_rows = [dict(item) for item in frame_events]
        for event in event_rows:
            rows.append(
                {
                    "schema_version": STAGE_DIAGNOSTICS_SCHEMA_VERSION,
                    "record_kind": "frame_event",
                    "stage": event.get("stage"),
                    "rule_code": event.get("rule_code"),
                    "detector": event.get("detector"),
                    "n_rows_changed": int(event.get("n_rows_changed") or 1),
                    "detail": event.get("detail"),
                    "dense_frame_idx": event.get("dense_frame_idx", pd.NA),
                    "focal_id": event.get("focal_id", pd.NA),
                    "partner_id": event.get("partner_id", pd.NA),
                    "before_label": event.get("before_label", pd.NA),
                    "after_label": event.get("after_label", pd.NA),
                }
            )

    if not rows:
        return pd.DataFrame(
            {
                col: pd.Series(dtype=object)
                for col in (
                    "schema_version",
                    "record_kind",
                    "stage",
                    "rule_code",
                    "detector",
                    "n_rows_changed",
                    "detail",
                    "dense_frame_idx",
                    "focal_id",
                    "partner_id",
                    "before_label",
                    "after_label",
                )
            }
        )
    return pd.DataFrame(rows)


def _atomic_parquet_write(path: Path | str, table: pd.DataFrame) -> Path:
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


def load_behavior_states_parquet(path: Path | str) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise PublicationError(f"behavior_states parquet missing: {file_path}")
    try:
        return pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001
        raise PublicationError(
            f"Could not read behavior_states parquet: {file_path}: {exc}"
        ) from exc


def load_behavior_bouts_parquet(path: Path | str) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise PublicationError(f"behavior_bouts parquet missing: {file_path}")
    try:
        return pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001
        raise PublicationError(
            f"Could not read behavior_bouts parquet: {file_path}: {exc}"
        ) from exc
