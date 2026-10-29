"""Pure reviewed-bout regeneration via S8 collapse (U11) + stable identities."""

from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_review.contracts import (
    CHANNEL_EXCLUSIVE,
    IDENTITY_DIRECTIONAL,
    REVIEW_TARGET_FPS,
    REVIEWED_BOUTS_SCHEMA_VERSION,
    EditingError,
)


def regenerate_reviewed_bouts(
    working_states: pd.DataFrame,
    *,
    review_run_id: str,
    parent_package_run_id: str,
    source_s8_run_id: str | None,
    source_s7_run_id: str | None = None,
    source_s6_run_id: str | None = None,
    automated_bouts: pd.DataFrame | None = None,
    sync_index: pd.DataFrame | None = None,
    fps: float = REVIEW_TARGET_FPS,
) -> pd.DataFrame:
    """Collapse working states with S8 ``extract_behavior_bouts`` (no arbitration)."""

    if working_states is None or working_states.empty:
        raise EditingError("Cannot regenerate bouts from empty working states.")
    work = working_states.copy()
    if "exp_name" not in work.columns:
        work["exp_name"] = parent_package_run_id

    raw = extract_behavior_bouts(
        work,
        fps=float(fps),
        source_s7_run_id=str(source_s7_run_id or ""),
        source_s6_run_id=str(source_s6_run_id or ""),
        run_id=str(source_s8_run_id or review_run_id),
    )
    return stamp_reviewed_bouts(
        raw,
        review_run_id=review_run_id,
        parent_package_run_id=parent_package_run_id,
        source_s8_run_id=source_s8_run_id,
        automated_bouts=automated_bouts,
        sync_index=sync_index,
    )


def stamp_reviewed_bouts(
    collapsed: pd.DataFrame,
    *,
    review_run_id: str,
    parent_package_run_id: str,
    source_s8_run_id: str | None,
    automated_bouts: pd.DataFrame | None = None,
    sync_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign deterministic reviewed bout ids and lineage to automated sources."""

    if collapsed is None or collapsed.empty:
        return _empty_reviewed_bouts()

    out = collapsed.copy().reset_index(drop=True)
    lineage_map = _build_lineage_index(automated_bouts)

    bout_ids: list[str] = []
    source_ids: list[Any] = []
    package_ids: list[Any] = []
    for i, row in out.iterrows():
        rid = deterministic_reviewed_bout_id(
            review_run_id=review_run_id,
            channel=str(row["channel"]),
            identity_grain=str(row["identity_grain"]),
            behavior_name=str(row["behavior_name"]),
            start_sample_idx=int(row["start_sample_idx"]),
            end_sample_idx=int(row["end_sample_idx"]),
            focal_id=_optional_int(row.get("focal_id")),
            partner_id=_optional_int(row.get("partner_id")),
            mouse_a_id=_optional_int(row.get("mouse_a_id")),
            mouse_b_id=_optional_int(row.get("mouse_b_id")),
            ordinal=int(i),
        )
        bout_ids.append(rid)
        key = interval_key_from_row(row)
        link = lineage_map.get(key)
        if link is None:
            source_ids.append(pd.NA)
            package_ids.append(pd.NA)
        else:
            source_ids.append(link.get("source_s8_bout_id", pd.NA))
            package_ids.append(link.get("bout_id", pd.NA))

    out["bout_id"] = bout_ids
    out["review_run_id"] = review_run_id
    out["parent_package_run_id"] = parent_package_run_id
    out["source_s8_bout_id"] = source_ids
    out["source_package_bout_id"] = package_ids
    out["source_s8_run_id"] = source_s8_run_id
    out["schema_version"] = REVIEWED_BOUTS_SCHEMA_VERSION

    if sync_index is not None and not sync_index.empty:
        sync = sync_index.set_index("sample_idx", drop=False)
        starts = []
        ends = []
        t_starts = []
        t_ends = []
        for _, row in out.iterrows():
            s = int(row["start_sample_idx"])
            e = int(row["end_sample_idx"])
            if s in sync.index and e in sync.index:
                starts.append(int(sync.loc[s, "prepared_frame_idx"]))
                ends.append(int(sync.loc[e, "prepared_frame_idx"]))
                t_starts.append(float(sync.loc[s, "prepared_time_sec"]))
                t_ends.append(float(sync.loc[e, "prepared_time_sec"]))
            else:
                starts.append(pd.NA)
                ends.append(pd.NA)
                t_starts.append(float("nan"))
                t_ends.append(float("nan"))
        out["prepared_frame_idx_start"] = starts
        out["prepared_frame_idx_end"] = ends
        out["prepared_time_sec_start"] = t_starts
        out["prepared_time_sec_end"] = t_ends

    return out


def deterministic_reviewed_bout_id(
    *,
    review_run_id: str,
    channel: str,
    identity_grain: str,
    behavior_name: str,
    start_sample_idx: int,
    end_sample_idx: int,
    focal_id: int | None,
    partner_id: int | None,
    mouse_a_id: int | None,
    mouse_b_id: int | None,
    ordinal: int,
) -> str:
    """Stable id for a reviewed bout geometry. Never reuses automated bout ids."""

    if identity_grain == IDENTITY_DIRECTIONAL or channel == CHANNEL_EXCLUSIVE:
        ident = f"f{focal_id}_p{partner_id}"
    else:
        ident = f"a{mouse_a_id}_b{mouse_b_id}"
    # Ordinal disambiguates rare identical keys within one collapse.
    return (
        f"rb__{review_run_id}__{channel}__{ident}__"
        f"{behavior_name}__{start_sample_idx}_{end_sample_idx}__{ordinal:04d}"
    )


def interval_key_from_row(row: pd.Series | dict[str, Any]) -> tuple[Any, ...]:
    channel = str(row["channel"])
    grain = str(row["identity_grain"])
    return (
        channel,
        grain,
        _optional_int(row.get("focal_id")),
        _optional_int(row.get("partner_id")),
        _optional_int(row.get("mouse_a_id")),
        _optional_int(row.get("mouse_b_id")),
        str(row["behavior_name"]),
        int(row["start_sample_idx"]),
        int(row["end_sample_idx"]),
    )


def _build_lineage_index(
    automated_bouts: pd.DataFrame | None,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if automated_bouts is None or automated_bouts.empty:
        return {}
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for _, row in automated_bouts.iterrows():
        key = interval_key_from_row(row)
        out[key] = {
            "source_s8_bout_id": (
                str(row["source_s8_bout_id"])
                if "source_s8_bout_id" in row and pd.notna(row["source_s8_bout_id"])
                else (
                    str(row["bout_id"])
                    if "bout_id" in row and pd.notna(row["bout_id"])
                    else None
                )
            ),
            "bout_id": str(row["bout_id"]) if "bout_id" in row else None,
        }
    return out


def _empty_reviewed_bouts() -> pd.DataFrame:
    cols = [
        "bout_id",
        "schema_version",
        "behavior_name",
        "channel",
        "identity_grain",
        "focal_id",
        "partner_id",
        "mouse_a_id",
        "mouse_b_id",
        "pair_id",
        "start_sample_idx",
        "end_sample_idx",
        "start_time_sec",
        "end_time_sec",
        "bout_duration_frames",
        "bout_duration_sec",
        "review_run_id",
        "parent_package_run_id",
        "source_s8_bout_id",
        "source_package_bout_id",
        "source_s8_run_id",
    ]
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
