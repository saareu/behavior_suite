"""Bout selection for media/CSV export jobs (selection only; no science transforms)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import (
    BOUT_INDEX_FILENAME,
    EXPORT_PROFILES_DIRNAME,
)
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import (
    EXPORT_BOUTS_SORT_COLUMNS,
    SELECTION_ALL_PACKAGE_BOUTS,
    SELECTION_BOUT_ID_LIST,
    SELECTION_CHANNELS_BEHAVIORS,
    SELECTION_EXPORT_PROFILE,
    ExportSelectionError,
)


def sort_export_bouts(bouts: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row order for export tables."""

    if bouts is None or bouts.empty:
        return bouts.copy() if bouts is not None else pd.DataFrame()
    cols = [c for c in EXPORT_BOUTS_SORT_COLUMNS if c in bouts.columns]
    if not cols:
        return bouts.sort_index().reset_index(drop=True)
    return bouts.sort_values(cols, kind="mergesort").reset_index(drop=True)


def select_bouts_for_export(
    packaged_bouts: pd.DataFrame,
    selection: SelectionSpec,
    *,
    package_dir: Path | None = None,
) -> pd.DataFrame:
    """Return selected package bout rows without transforming science fields."""

    if packaged_bouts is None:
        raise ExportSelectionError("packaged behavior_bouts table is required.")
    if "bout_id" not in packaged_bouts.columns:
        raise ExportSelectionError("packaged bouts missing bout_id.")

    spec = selection.validated()
    frame = packaged_bouts.copy()

    if spec.mode == SELECTION_ALL_PACKAGE_BOUTS:
        selected = frame
    elif spec.mode == SELECTION_CHANNELS_BEHAVIORS:
        mask = pd.Series(True, index=frame.index)
        if spec.channels:
            if "channel" not in frame.columns:
                raise ExportSelectionError("packaged bouts missing channel.")
            mask &= frame["channel"].astype(str).isin(set(spec.channels))
        if spec.behavior_names:
            if "behavior_name" not in frame.columns:
                raise ExportSelectionError("packaged bouts missing behavior_name.")
            mask &= frame["behavior_name"].astype(str).isin(set(spec.behavior_names))
        selected = frame.loc[mask]
    elif spec.mode == SELECTION_EXPORT_PROFILE:
        if package_dir is None:
            raise ExportSelectionError(
                "export_profile selection requires package_dir."
            )
        assert spec.profile_id is not None
        index_path = (
            Path(package_dir)
            / EXPORT_PROFILES_DIRNAME
            / spec.profile_id
            / BOUT_INDEX_FILENAME
        )
        if not index_path.is_file():
            raise ExportSelectionError(
                f"Packaged export profile index missing: {index_path}"
            )
        index = pd.read_parquet(index_path)
        if "bout_id" not in index.columns:
            raise ExportSelectionError(
                f"Profile index missing bout_id: {index_path}"
            )
        ids = set(index["bout_id"].astype(str))
        selected = frame.loc[frame["bout_id"].astype(str).isin(ids)]
    elif spec.mode == SELECTION_BOUT_ID_LIST:
        wanted = set(spec.bout_ids)
        selected = frame.loc[frame["bout_id"].astype(str).isin(wanted)]
        found = set(selected["bout_id"].astype(str))
        missing = sorted(wanted - found)
        if missing:
            raise ExportSelectionError(
                "bout_id_list contains ids absent from package: "
                f"{missing[:10]}{'…' if len(missing) > 10 else ''}"
            )
    else:  # pragma: no cover - validated above
        raise ExportSelectionError(f"Unsupported selection mode: {spec.mode}")

    return sort_export_bouts(selected)
