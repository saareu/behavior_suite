"""Serialize legacy framewise tables with reference-compatible formatting."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.legacy_framewise_csv.contracts import (
    LEGACY_FRAMEWISE_COLUMNS,
    LegacyFramewiseFormatError,
)


def format_coordinate(value: Any) -> str:
    """Format one pose coordinate for legacy CSV compatibility.

    Laboratory reference values are integer-valued floats rendered as ``N.0``.
    S4 stores sub-pixel floats in prepared-video pixel space; the ``.0`` suffix
    is therefore a **formatting convention** (round half-even to nearest
    integer, then one decimal place), not evidence that the scientific source
    was integer-typed. Missing / non-finite values become empty fields.

    This rounding is exclusive to legacy framewise CSV serialization (v1/v2) and
    must never be written back into S4 ``final_pose`` or other scientific stores.
    """

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{round(number):.1f}"


def format_legacy_framewise_rows(table: pd.DataFrame) -> list[list[str]]:
    """Convert the built table into string rows (no header)."""

    if list(table.columns) != list(LEGACY_FRAMEWISE_COLUMNS):
        raise LegacyFramewiseFormatError(
            "legacy framewise table columns must match the contract order exactly."
        )
    rows: list[list[str]] = []
    pose_cols = {
        c for c in LEGACY_FRAMEWISE_COLUMNS if c.endswith("_x") or c.endswith("_y")
    }
    for rec in table.itertuples(index=False, name=None):
        row: list[str] = []
        for col, value in zip(LEGACY_FRAMEWISE_COLUMNS, rec, strict=True):
            if col in ("frame", "dense_frame_idx", "pred_side_by_side"):
                # v2 permits nullable dense_frame_idx / SBS; v1 keeps these populated.
                try:
                    if value is None or pd.isna(value):
                        row.append("")
                        continue
                except (TypeError, ValueError):
                    if value is None or (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        row.append("")
                        continue
                row.append(str(int(value)))
            elif col in pose_cols:
                row.append(format_coordinate(value))
            else:
                # Behavior labels: empty when missing; otherwise exact text.
                if value is None:
                    row.append("")
                else:
                    try:
                        if pd.isna(value):
                            row.append("")
                        else:
                            row.append(str(value))
                    except (TypeError, ValueError):
                        row.append(str(value))
        rows.append(row)
    return rows


def write_legacy_framewise_csv(path: Path | str, table: pd.DataFrame) -> Path:
    """Write UTF-8 CSV with CRLF line endings and the exact 23-column header."""

    if list(table.columns) != list(LEGACY_FRAMEWISE_COLUMNS):
        raise LegacyFramewiseFormatError(
            "Refusing to write CSV with non-contract column order/names."
        )
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    rows = format_legacy_framewise_rows(table)
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(
                handle,
                delimiter=",",
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\r\n",
            )
            writer.writerow(list(LEGACY_FRAMEWISE_COLUMNS))
            writer.writerows(rows)
        tmp.replace(out)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return out


def validate_legacy_framewise_csv_bytes(payload: bytes) -> dict[str, Any]:
    """Validate serialized CSV against the format contract (header/CRLF/cols)."""

    if b"\r\n" not in payload:
        raise LegacyFramewiseFormatError("legacy framewise CSV must use CRLF endings.")
    text = payload.decode("utf-8")
    lines = text.split("\r\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        raise LegacyFramewiseFormatError("CSV is empty.")
    header = next(csv.reader([lines[0]]))
    if header != list(LEGACY_FRAMEWISE_COLUMNS):
        raise LegacyFramewiseFormatError(
            f"Header mismatch: got {header!r}, expected {list(LEGACY_FRAMEWISE_COLUMNS)!r}."
        )
    n_rows = 0
    for line in lines[1:]:
        row = next(csv.reader([line]))
        if len(row) != len(LEGACY_FRAMEWISE_COLUMNS):
            raise LegacyFramewiseFormatError(
                f"Row {n_rows} has {len(row)} columns; expected "
                f"{len(LEGACY_FRAMEWISE_COLUMNS)}."
            )
        n_rows += 1
    return {
        "n_header_columns": len(header),
        "n_data_rows": n_rows,
        "header": header,
        "lineterminator": "CRLF",
    }
