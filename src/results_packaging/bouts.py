"""Lossless S8 bout packaging with prepared-frame navigation annotations.

The packaged table is an annotated copy of the complete S8 bout table. It does
**not** re-collapse, merge, split, filter, or rewrite scientific endpoints /
durations. Prepared-frame columns are seek aids only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.bouts import BOUT_COLUMNS as S8_BOUT_COLUMNS
from results_packaging.bout_mapping import map_bout_boundaries_to_prepared_frames
from results_packaging.contracts import (
    BOUT_NAVIGATION_COLUMNS,
    BOUTS_SCHEMA_VERSION,
    PACKAGE_BOUTS_SORT_COLUMNS,
    BoutPackagingError,
)


def build_package_bout_id(*, source_s8_run_id: str, source_s8_bout_id: str) -> str:
    """Build a package-local bout id that namespaces the stable S8 bout_id.

    S8 publishes unique ``bout_{i:08d}`` ids within a run. Packaging preserves
    that identifier as ``source_s8_bout_id`` and namespaces it with the S8 run
    id so package ids are globally unambiguous and stable across repeated
    packaging of the same S8 source (U-R5).
    """

    run_id = str(source_s8_run_id).strip()
    bout_id = str(source_s8_bout_id).strip()
    if not run_id:
        raise BoutPackagingError("source_s8_run_id is required for package bout_id.")
    if not bout_id:
        raise BoutPackagingError("source_s8_bout_id is required for package bout_id.")
    return f"{run_id}__{bout_id}"


def authoritative_bout_content_key(row: dict[str, Any] | pd.Series) -> str:
    """Deterministic content key distinguishing directional vs symmetric grain.

    Independent of DataFrame row order. Used for lossless multiset checks and
    documentation; does not replace ``source_s8_bout_id``.
    """

    channel = str(row["channel"])
    behavior = str(row["behavior_name"])
    grain = str(row["identity_grain"])
    start = int(row["start_sample_idx"])
    end = int(row["end_sample_idx"])
    if grain == "directional":
        focal_raw = row["focal_id"]
        partner_raw = row["partner_id"]
        if pd.isna(focal_raw) or pd.isna(partner_raw):
            raise BoutPackagingError(
                "Directional bout missing focal_id/partner_id for content key."
            )
        identity = f"f{int(focal_raw)}->p{int(partner_raw)}"
    elif grain == "symmetric_pair":
        mouse_a = int(row["mouse_a_id"])
        mouse_b = int(row["mouse_b_id"])
        identity = f"pair:{mouse_a}|{mouse_b}"
    else:
        raise BoutPackagingError(f"Unknown identity_grain {grain!r}.")
    return f"{channel}|{behavior}|{grain}|{identity}|{start}|{end}"


def package_behavior_bouts(
    s8_bouts: pd.DataFrame,
    sync_index: pd.DataFrame,
    *,
    source_s8_run_id: str,
    package_run_id: str | None = None,
    fps: float = 120.0,
) -> pd.DataFrame:
    """Return a lossless annotated copy of the complete S8 bout table.

    Preserves all authoritative S8 science columns and adds package stamps plus
    prepared-frame navigation annotations from ``sync_index``.
    """

    if s8_bouts is None:
        raise BoutPackagingError("S8 bouts table is required.")
    missing = [c for c in S8_BOUT_COLUMNS if c not in s8_bouts.columns]
    if missing:
        raise BoutPackagingError(f"S8 bouts missing required columns: {missing}")

    if s8_bouts.empty:
        return _empty_package_bouts_frame()

    if s8_bouts["bout_id"].isna().any():
        raise BoutPackagingError("S8 bouts contain null bout_id values.")
    source_ids = s8_bouts["bout_id"].astype(str)
    if int(source_ids.nunique()) != int(len(s8_bouts)):
        raise BoutPackagingError("S8 bout_id values are not unique within the source.")

    # Preserve every S8 science column; rename source bout_id for provenance.
    packaged = s8_bouts.copy().reset_index(drop=True)
    packaged["source_s8_bout_id"] = source_ids.to_numpy()
    packaged["source_s8_run_id"] = str(source_s8_run_id)
    packaged["package_run_id"] = (
        str(package_run_id) if package_run_id is not None else pd.NA
    )
    packaged["s8_bouts_schema_version"] = packaged["schema_version"].astype(str)
    packaged["schema_version"] = BOUTS_SCHEMA_VERSION
    packaged["bout_id"] = [
        build_package_bout_id(
            source_s8_run_id=str(source_s8_run_id),
            source_s8_bout_id=str(sid),
        )
        for sid in packaged["source_s8_bout_id"].tolist()
    ]

    mapped = map_bout_boundaries_to_prepared_frames(
        s8_bouts.assign(bout_id=source_ids),
        sync_index,
        fps=fps,
    )
    # Align navigation annotations by source bout_id (not incidental row order).
    nav = mapped.set_index("bout_id")[list(BOUT_NAVIGATION_COLUMNS)]
    for col in BOUT_NAVIGATION_COLUMNS:
        packaged[col] = packaged["source_s8_bout_id"].map(nav[col]).to_numpy()
    if packaged[list(BOUT_NAVIGATION_COLUMNS)].isna().any().any():
        raise BoutPackagingError(
            "Prepared-frame navigation annotations missing for one or more bouts."
        )

    # Scientific endpoints / durations must remain bit-equal to S8.
    _assert_lossless_science(s8_bouts, packaged)

    packaged = _stable_sort_package_bouts(packaged)
    if int(packaged["bout_id"].nunique()) != int(len(packaged)):
        raise BoutPackagingError("Package bout_id values are not unique.")
    return packaged.reset_index(drop=True)


def assert_lossless_bout_copy(
    s8_bouts: pd.DataFrame,
    packaged_bouts: pd.DataFrame,
) -> None:
    """Public invariant check used by tests and the Sprint 2 runner."""

    _assert_lossless_science(s8_bouts, packaged_bouts)


def _assert_lossless_science(
    s8_bouts: pd.DataFrame,
    packaged_bouts: pd.DataFrame,
) -> None:
    if int(len(packaged_bouts)) != int(len(s8_bouts)):
        raise BoutPackagingError(
            f"Packaged bout count {len(packaged_bouts)} != S8 count {len(s8_bouts)}."
        )

    s8_by_id = s8_bouts.assign(_sid=s8_bouts["bout_id"].astype(str)).set_index(
        "_sid", drop=True
    )
    pkg_by_id = packaged_bouts.set_index("source_s8_bout_id", drop=False)
    if set(s8_by_id.index) != set(pkg_by_id.index):
        raise BoutPackagingError(
            "Packaged bouts are not a 1:1 match to S8 source_s8_bout_id values."
        )

    science_compare = [
        "channel",
        "behavior_name",
        "identity_grain",
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
        "bout_duration_sec",
        "mouse_a_id",
        "mouse_b_id",
        "pair_id",
        "focal_id",
        "partner_id",
        "mean_prob",
        "max_prob",
        "median_prob",
        "frac_input_valid",
        "label_reconstruction_mode",
        "undefined_reason",
        "source_s7_run_id",
        "source_s6_run_id",
        "exp_name",
    ]
    for sid in s8_by_id.index:
        s8_row = s8_by_id.loc[sid]
        pkg_row = pkg_by_id.loc[sid]
        for col in science_compare:
            if col not in s8_bouts.columns:
                continue
            left = s8_row[col]
            right = pkg_row[col]
            if pd.isna(left) and pd.isna(right):
                continue
            if col in {
                "start_sample_idx",
                "end_sample_idx",
                "bout_duration_frames",
                "mouse_a_id",
                "mouse_b_id",
            }:
                if int(left) != int(right):
                    raise BoutPackagingError(
                        f"Packaged bout {sid!r} altered {col!r}."
                    )
            elif col in {
                "bout_duration_sec",
                "mean_prob",
                "max_prob",
                "median_prob",
                "frac_input_valid",
            }:
                if not np.isclose(
                    float(left),
                    float(right),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                ):
                    raise BoutPackagingError(
                        f"Packaged bout {sid!r} altered {col!r}."
                    )
            elif col in {"focal_id", "partner_id"}:
                if pd.isna(left) != pd.isna(right):
                    raise BoutPackagingError(
                        f"Packaged bout {sid!r} altered {col!r} nullability."
                    )
                if pd.notna(left) and int(left) != int(right):
                    raise BoutPackagingError(
                        f"Packaged bout {sid!r} altered {col!r}."
                    )
            else:
                if str(left) != str(right):
                    raise BoutPackagingError(
                        f"Packaged bout {sid!r} altered {col!r}."
                    )

        if str(pkg_row["identity_grain"]) == "symmetric_pair" and (
            pd.notna(pkg_row["focal_id"]) or pd.notna(pkg_row["partner_id"])
        ):
            raise BoutPackagingError(
                "Symmetric-pair packaged bout must keep null focal/partner."
            )
        if str(pkg_row["identity_grain"]) == "directional" and (
            pd.isna(pkg_row["focal_id"]) or pd.isna(pkg_row["partner_id"])
        ):
            raise BoutPackagingError(
                "Directional packaged bout missing focal_id/partner_id."
            )


def _stable_sort_package_bouts(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in PACKAGE_BOUTS_SORT_COLUMNS if c in df.columns]
    # Use na_position so null focal/partner sort deterministically for
    # symmetric channels without depending on source row order.
    return df.sort_values(
        by=sort_cols,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _empty_package_bouts_frame() -> pd.DataFrame:
    cols = (
        list(S8_BOUT_COLUMNS)
        + [
            "source_s8_bout_id",
            "source_s8_run_id",
            "package_run_id",
            "s8_bouts_schema_version",
        ]
        + list(BOUT_NAVIGATION_COLUMNS)
    )
    # bout_id / schema_version already in S8_BOUT_COLUMNS; stamps overwrite later.
    return pd.DataFrame({c: pd.Series(dtype=object) for c in dict.fromkeys(cols)})
