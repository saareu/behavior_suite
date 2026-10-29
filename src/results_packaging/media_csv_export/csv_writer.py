"""Automated bout-table CSV serialization (no science transforms)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.media_csv_export.contracts import (
    CSV_PROVENANCE_COLUMNS,
    EXPORT_PROVENANCE_AUTOMATED,
    ExportPublicationError,
)
from results_packaging.media_csv_export.selection import sort_export_bouts


def build_automated_bout_csv_frame(
    selected_bouts: pd.DataFrame,
    *,
    export_job_id: str,
    selection_mode: str,
    source_package_run_id: str,
) -> pd.DataFrame:
    """Copy selected package rows and append automated provenance columns."""

    if selected_bouts is None:
        raise ExportPublicationError("CSV export requires a bout table.")
    frame = sort_export_bouts(selected_bouts.copy())
    # Drop any colliding provenance columns then append contract order.
    for col in CSV_PROVENANCE_COLUMNS:
        if col in frame.columns:
            frame = frame.drop(columns=[col])
    frame["export_provenance"] = EXPORT_PROVENANCE_AUTOMATED
    frame["export_job_id"] = str(export_job_id)
    frame["selection_mode"] = str(selection_mode)
    frame["source_package_run_id"] = str(source_package_run_id)
    return frame


def write_automated_bout_csv(path: Path | str, table: pd.DataFrame) -> Path:
    """Write UTF-8 CSV with header; missing values as empty fields; atomic replace."""

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        table.to_csv(tmp, index=False, na_rep="", encoding="utf-8", lineterminator="\n")
        tmp.replace(out)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return out
