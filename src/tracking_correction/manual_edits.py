"""Minimal manual tracking edits for the S3 review working pose.

Operates only on ``working_tracked_pose.parquet``. Does not call the automatic
corrector and does not mutate S1 or S2 artifacts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from tracking_correction.contracts import (
    AUTOMATIC_TRACKED_POSE_FILENAME,
    MANUAL_ACTION_BLANK_NODE,
    MANUAL_ACTION_SWAP_IDENTITIES,
    MANUAL_ACTION_SWAP_NODE,
    MANUAL_CORRECTIONS_FILENAME,
    WORKING_TRACKED_POSE_FILENAME,
    TrackingCorrectionError,
)

_TRACK_A = 0
_TRACK_B = 1
_TEMP_TRACK = -999


@dataclass(frozen=True, slots=True)
class ManualEditResult:
    """Outcome of one manual edit, undo, or reset."""

    action: str
    start_frame: int
    end_frame: int
    edit_count: int
    acceptance_invalidated: bool
    record: Mapping[str, Any] | None = None


def ensure_automatic_baseline(run_dir: Path, working_path: Path) -> Path:
    """Copy working pose to an immutable automatic baseline on first edit."""

    baseline = Path(run_dir) / AUTOMATIC_TRACKED_POSE_FILENAME
    if baseline.is_file():
        return baseline
    if not working_path.is_file():
        raise TrackingCorrectionError(
            f"Working tracked pose is missing: {working_path}"
        )
    try:
        shutil.copy2(working_path, baseline)
    except OSError as exc:
        raise TrackingCorrectionError(
            f"Could not create automatic baseline pose: {exc}"
        ) from exc
    return baseline


def load_manual_corrections(path: Path) -> list[dict[str, Any]]:
    """Load the manual edit stack; missing file means an empty stack."""

    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingCorrectionError(
            f"manual_corrections.json is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise TrackingCorrectionError(
            "manual_corrections.json must contain a list of edit records."
        )
    records: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            records.append(dict(item))
    return records


def write_manual_corrections(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(list(records), stream, ensure_ascii=False, indent=2, default=_json_default)
        stream.write("\n")


def apply_swap_identities(
    frame: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    """Swap track 0 ↔ track 1 ownership over an inclusive frame interval.

    Entire detection rows move by flipping their track label, so sparse
    ``(frame, track, node)`` sets and all associated pose/provenance columns
    travel with the detection. Missing nodes on one side are not invented.
    """

    start, end = _validate_interval(start_frame, end_frame)
    out = frame.copy()
    frame_col = _frame_column(out)
    track_values = _track_series(out)
    mask = (out[frame_col] >= start) & (out[frame_col] <= end)
    mask_a = mask & (track_values == _TRACK_A)
    mask_b = mask & (track_values == _TRACK_B)
    if not bool(mask_a.any()) and not bool(mask_b.any()):
        raise TrackingCorrectionError(
            f"No track 0/1 rows found in frames {start}–{end}."
        )
    # Flip ownership in place. Do not mutate other columns: provenance stays
    # attached to the detection row as the track label changes.
    out.loc[mask_a, "track"] = _TEMP_TRACK
    out.loc[mask_b, "track"] = _TRACK_A
    out.loc[out["track"] == _TEMP_TRACK, "track"] = _TRACK_B
    return _sort_pose(out)


def apply_swap_node(
    frame: pd.DataFrame,
    *,
    frame_idx: int,
    node: str,
) -> pd.DataFrame:
    """Swap or move one node between track 0 and track 1 at a single frame.

    If the node exists on both tracks, ownership is swapped. If it exists on
    only one track, the detection row is moved to the other track. Pose and
    provenance columns travel with the row; nothing is duplicated or invented.
    """

    index = _validate_frame(frame_idx)
    node_name = _validate_node(node)
    out = frame.copy()
    frame_col = _frame_column(out)
    track_values = _track_series(out)
    node_mask = (out[frame_col] == index) & (out["node"].astype(str) == node_name)
    idx_a = out.index[node_mask & (track_values == _TRACK_A)].tolist()
    idx_b = out.index[node_mask & (track_values == _TRACK_B)].tolist()
    if len(idx_a) > 1 or len(idx_b) > 1:
        raise TrackingCorrectionError(
            f"Node {node_name!r} has duplicate rows at frame {index}."
        )
    if not idx_a and not idx_b:
        raise TrackingCorrectionError(
            f"Node {node_name!r} is missing on both tracks at frame {index}."
        )
    if idx_a and idx_b:
        out.at[int(idx_a[0]), "track"] = _TRACK_B
        out.at[int(idx_b[0]), "track"] = _TRACK_A
    elif idx_a:
        out.at[int(idx_a[0]), "track"] = _TRACK_B
    else:
        out.at[int(idx_b[0]), "track"] = _TRACK_A
    return _sort_pose(out)


def apply_blank_node(
    frame: pd.DataFrame,
    *,
    frame_idx: int,
    track: int,
    node: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blank one node at the current frame; returns before-state for undo."""

    index = _validate_frame(frame_idx)
    track_idx = _validate_track(track)
    node_name = _validate_node(node)
    out = frame.copy()
    frame_col = _frame_column(out)
    track_values = _track_series(out)
    mask = (
        (out[frame_col] == index)
        & (track_values == track_idx)
        & (out["node"].astype(str) == node_name)
    )
    matches = out.index[mask].tolist()
    if len(matches) != 1:
        raise TrackingCorrectionError(
            f"Node {node_name!r} on track {track_idx} is missing at frame {index}."
        )
    row_idx = int(matches[0])
    before = _row_snapshot(out.loc[row_idx])
    for column in ("x", "y"):
        if column in out.columns:
            out.at[row_idx, column] = np.nan
    for score_col in ("point_score", "score"):
        if score_col in out.columns:
            out.at[row_idx, score_col] = 0.0
    if "was_node_blanked" in out.columns:
        out.at[row_idx, "was_node_blanked"] = np.uint8(1)
    return _sort_pose(out), before


def undo_edit(frame: pd.DataFrame, record: Mapping[str, Any]) -> pd.DataFrame:
    """Apply the inverse of one manual edit record."""

    action = str(record.get("action") or "")
    if action == MANUAL_ACTION_SWAP_IDENTITIES:
        return apply_swap_identities(
            frame,
            start_frame=int(record["start_frame"]),
            end_frame=int(record["end_frame"]),
        )
    if action == MANUAL_ACTION_SWAP_NODE:
        return apply_swap_node(
            frame,
            frame_idx=int(record["start_frame"]),
            node=str(record["node"]),
        )
    if action == MANUAL_ACTION_BLANK_NODE:
        before = record.get("before")
        if not isinstance(before, Mapping):
            raise TrackingCorrectionError(
                "blank_node undo record is missing before-state."
            )
        return _restore_row(frame, before)
    raise TrackingCorrectionError(f"Unknown manual edit action: {action!r}")


def restore_automatic_baseline(
    *,
    run_dir: Path,
    working_path: Path,
) -> Path:
    """Copy automatic baseline over the working pose."""

    baseline = Path(run_dir) / AUTOMATIC_TRACKED_POSE_FILENAME
    if not baseline.is_file():
        raise TrackingCorrectionError(
            "No automatic baseline exists; there are no manual edits to reset."
        )
    try:
        shutil.copy2(baseline, working_path)
    except OSError as exc:
        raise TrackingCorrectionError(
            f"Could not restore automatic baseline: {exc}"
        ) from exc
    return baseline


def read_working_pose(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise TrackingCorrectionError(
            f"Could not read working tracked pose: {exc}"
        ) from exc


def write_working_pose(path: Path, frame: pd.DataFrame) -> None:
    """Persist working pose via temp-file + atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, target)
    except Exception as exc:
        with suppress(OSError):
            if tmp.exists():
                tmp.unlink()
        raise TrackingCorrectionError(
            f"Could not write working tracked pose: {exc}"
        ) from exc


def frame_domain_signature(frame: pd.DataFrame) -> tuple[tuple[int, ...], int]:
    """Return sorted unique frame indices and row count for domain checks."""

    frame_col = _frame_column(frame)
    values = pd.to_numeric(frame[frame_col], errors="coerce")
    unique = tuple(sorted({int(v) for v in values.tolist() if pd.notna(v)}))
    return unique, int(len(frame))


def available_nodes(frame: pd.DataFrame) -> tuple[str, ...]:
    if "node" not in frame.columns:
        return ()
    names = {
        str(value).strip()
        for value in frame["node"].tolist()
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan"
    }
    return tuple(sorted(names))


def build_edit_record(
    *,
    action: str,
    start_frame: int,
    end_frame: int,
    tracks: Sequence[int],
    node: str | None = None,
    track: int | None = None,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "action": action,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "tracks": [int(t) for t in tracks],
        "timestamp": _iso_now(),
        "source": "user",
    }
    if node is not None:
        record["node"] = node
    if track is not None:
        record["track"] = int(track)
    if before is not None:
        record["before"] = dict(before)
    if after is not None:
        record["after"] = dict(after)
    return record


def manual_corrections_path(run_dir: Path) -> Path:
    return Path(run_dir) / MANUAL_CORRECTIONS_FILENAME


def automatic_baseline_path(run_dir: Path) -> Path:
    return Path(run_dir) / AUTOMATIC_TRACKED_POSE_FILENAME


def working_pose_path(run_dir: Path) -> Path:
    return Path(run_dir) / WORKING_TRACKED_POSE_FILENAME


def _validate_interval(start_frame: object, end_frame: object) -> tuple[int, int]:
    start = _validate_frame(start_frame)
    end = _validate_frame(end_frame)
    if end < start:
        raise TrackingCorrectionError(
            f"Interval end frame {end} is before start frame {start}."
        )
    return start, end


def _validate_frame(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingCorrectionError("Frame index must be an integer.")
    if value < 0:
        raise TrackingCorrectionError(f"Frame index {value} must be non-negative.")
    return value


def _validate_track(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingCorrectionError("Track index must be an integer.")
    if value not in (_TRACK_A, _TRACK_B):
        raise TrackingCorrectionError("Track index must be 0 or 1 for this MVP.")
    return value


def _validate_node(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrackingCorrectionError("Node name is required.")
    return value.strip()


def _frame_column(frame: pd.DataFrame) -> str:
    if "frame_idx" in frame.columns:
        return "frame_idx"
    if "frame" in frame.columns:
        return "frame"
    raise TrackingCorrectionError("Working pose is missing frame_idx/frame.")


def _track_series(frame: pd.DataFrame) -> pd.Series:
    if "track" not in frame.columns:
        raise TrackingCorrectionError("Working pose is missing track.")
    parsed = frame["track"].map(_parse_track_index)
    if parsed.isna().any():
        raise TrackingCorrectionError("Working pose has unparsable track labels.")
    return parsed.astype(int)


def _parse_track_index(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not float(value).is_integer():
            return None
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    lowered = text.lower()
    if lowered.startswith("track_"):
        suffix = text.split("_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _sort_pose(frame: pd.DataFrame) -> pd.DataFrame:
    frame_col = _frame_column(frame)
    sort_cols = [frame_col]
    if "track" in frame.columns:
        sort_cols.append("track")
    if "node" in frame.columns:
        sort_cols.append("node")
    return frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _row_snapshot(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        payload[str(key)] = _json_default(value)
    return payload


def _restore_row(frame: pd.DataFrame, before: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    frame_col = _frame_column(out)
    try:
        frame_idx = int(before[frame_col] if frame_col in before else before["frame"])
        track_idx = int(before["track"])
        node_name = str(before["node"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackingCorrectionError(
            "blank_node before-state is incomplete."
        ) from exc
    track_values = _track_series(out)
    mask = (
        (out[frame_col] == frame_idx)
        & (track_values == track_idx)
        & (out["node"].astype(str) == node_name)
    )
    matches = out.index[mask].tolist()
    if len(matches) != 1:
        raise TrackingCorrectionError(
            "Could not locate blanked node row to restore."
        )
    row_idx = int(matches[0])
    for column, value in before.items():
        if column not in out.columns:
            continue
        out.at[row_idx, column] = _from_json_value(value, dtype=out[column].dtype)
    return _sort_pose(out)


def _json_default(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, np.generic):
        as_python = value.item()
        if isinstance(as_python, float) and math.isnan(as_python):
            return None
        return as_python
    if pd.isna(value):
        return None
    return str(value)


def _from_json_value(value: object, *, dtype: Any | None = None) -> object:
    if value is None:
        if dtype is not None and pd.api.types.is_extension_array_dtype(dtype):
            return pd.NA
        return np.nan
    if dtype is None:
        return value
    try:
        if pd.api.types.is_bool_dtype(dtype):
            return bool(value)
        if pd.api.types.is_integer_dtype(dtype):
            if isinstance(value, bool) or not isinstance(value, int | float | str):
                raise TypeError(f"Cannot cast {value!r} to integer.")
            return int(value)
        if pd.api.types.is_float_dtype(dtype):
            if isinstance(value, bool) or not isinstance(value, int | float | str):
                raise TypeError(f"Cannot cast {value!r} to float.")
            number = float(value)
            cast_type = getattr(dtype, "type", None)
            return cast_type(number) if cast_type is not None else number
        if pd.api.types.is_string_dtype(dtype) or dtype is object:
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
