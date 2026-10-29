"""Map inclusive S8 bout dense endpoints onto prepared-video frame indices.

Does not re-collapse, merge, split, or filter bouts. Prepared indices are seek
aids only; scientific duration remains S8 ``bout_duration_frames`` / 120 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import BoutMappingError


@dataclass(frozen=True, slots=True)
class BoutBoundaryMapping:
    """Prepared-frame seek annotations for one S8 bout (science unchanged)."""

    bout_id: str
    start_sample_idx: int
    end_sample_idx: int
    bout_duration_frames: int
    bout_duration_sec: float
    prepared_frame_idx_start: int
    prepared_frame_idx_end: int
    prepared_time_sec_start: float
    prepared_time_sec_end: float
    channel: str | None = None
    behavior_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bout_id": self.bout_id,
            "start_sample_idx": self.start_sample_idx,
            "end_sample_idx": self.end_sample_idx,
            "bout_duration_frames": self.bout_duration_frames,
            "bout_duration_sec": self.bout_duration_sec,
            "prepared_frame_idx_start": self.prepared_frame_idx_start,
            "prepared_frame_idx_end": self.prepared_frame_idx_end,
            "prepared_time_sec_start": self.prepared_time_sec_start,
            "prepared_time_sec_end": self.prepared_time_sec_end,
            "channel": self.channel,
            "behavior_name": self.behavior_name,
        }


def map_bout_boundaries_to_prepared_frames(
    bouts: pd.DataFrame,
    sync_index: pd.DataFrame,
    *,
    fps: float = 120.0,
) -> pd.DataFrame:
    """Return bout rows with prepared seek columns; preserve S8 lengths.

    Raises ``BoutMappingError`` for invalid source indices or missing sync rows.
    """

    if sync_index is None or sync_index.empty:
        raise BoutMappingError("sync_index is required for bout-boundary mapping.")
    required_sync = {
        "sample_idx",
        "prepared_frame_idx",
        "prepared_time_sec",
    }
    missing_sync = sorted(required_sync - set(sync_index.columns))
    if missing_sync:
        raise BoutMappingError(f"sync_index missing columns: {missing_sync}")

    if bouts is None:
        raise BoutMappingError("bouts table is required.")
    if bouts.empty:
        return _empty_mapping_frame()

    required_bouts = {
        "bout_id",
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
    }
    missing_bouts = sorted(required_bouts - set(bouts.columns))
    if missing_bouts:
        raise BoutMappingError(f"bouts missing columns: {missing_bouts}")

    lookup = (
        sync_index[["sample_idx", "prepared_frame_idx", "prepared_time_sec"]]
        .drop_duplicates(subset=["sample_idx"])
        .set_index("sample_idx")
    )
    n_dense = int(lookup.index.max()) + 1 if len(lookup) else 0
    if len(lookup) != n_dense:
        raise BoutMappingError(
            "sync_index sample_idx domain must be contiguous for bout mapping."
        )

    rows: list[dict[str, Any]] = []
    has_duration_sec = "bout_duration_sec" in bouts.columns
    has_channel = "channel" in bouts.columns
    has_behavior = "behavior_name" in bouts.columns
    for rec in bouts.to_dict(orient="records"):
        bout_id = str(rec["bout_id"])
        start = int(rec["start_sample_idx"])
        end = int(rec["end_sample_idx"])
        dur_frames = int(rec["bout_duration_frames"])
        if start > end:
            raise BoutMappingError(
                f"Invalid bout {bout_id!r}: start_sample_idx ({start}) > "
                f"end_sample_idx ({end})."
            )
        expected = end - start + 1
        if dur_frames != expected:
            raise BoutMappingError(
                f"Invalid bout {bout_id!r}: bout_duration_frames ({dur_frames}) "
                f"!= inclusive count ({expected})."
            )
        if start < 0 or end >= n_dense:
            raise BoutMappingError(
                f"Invalid bout {bout_id!r}: sample indices [{start}, {end}] "
                f"outside sync domain [0, {n_dense - 1}]."
            )
        try:
            start_prep = int(lookup.loc[start, "prepared_frame_idx"])
            end_prep = int(lookup.loc[end, "prepared_frame_idx"])
            start_ptime = float(lookup.loc[start, "prepared_time_sec"])
            end_ptime = float(lookup.loc[end, "prepared_time_sec"])
        except KeyError as exc:
            raise BoutMappingError(
                f"Bout {bout_id!r} references missing sync sample: {exc}"
            ) from exc

        if has_duration_sec and pd.notna(rec.get("bout_duration_sec")):
            dur_sec = float(rec["bout_duration_sec"])
        else:
            dur_sec = float(dur_frames) / float(fps)

        channel_val = rec.get("channel") if has_channel else None
        behavior_val = rec.get("behavior_name") if has_behavior else None
        rows.append(
            {
                "bout_id": bout_id,
                "channel": str(channel_val) if pd.notna(channel_val) else None,
                "behavior_name": (
                    str(behavior_val) if pd.notna(behavior_val) else None
                ),
                "start_sample_idx": start,
                "end_sample_idx": end,
                "bout_duration_frames": dur_frames,
                "bout_duration_sec": dur_sec,
                "prepared_frame_idx_start": start_prep,
                "prepared_frame_idx_end": end_prep,
                "prepared_time_sec_start": start_ptime,
                "prepared_time_sec_end": end_ptime,
            }
        )

    mapped = pd.DataFrame(rows)
    # Invariant: no change to S8 bout lengths.
    if not np.array_equal(
        mapped["bout_duration_frames"].to_numpy(dtype=np.int64),
        bouts["bout_duration_frames"].to_numpy(dtype=np.int64),
    ):
        raise BoutMappingError("Bout-boundary mapping altered bout_duration_frames.")
    if not np.array_equal(
        mapped["start_sample_idx"].to_numpy(dtype=np.int64),
        bouts["start_sample_idx"].to_numpy(dtype=np.int64),
    ) or not np.array_equal(
        mapped["end_sample_idx"].to_numpy(dtype=np.int64),
        bouts["end_sample_idx"].to_numpy(dtype=np.int64),
    ):
        raise BoutMappingError("Bout-boundary mapping altered dense start/end indices.")
    return mapped


def map_single_bout_boundary(
    *,
    bout_id: str,
    start_sample_idx: int,
    end_sample_idx: int,
    sync_index: pd.DataFrame,
    bout_duration_frames: int | None = None,
    fps: float = 120.0,
    channel: str | None = None,
    behavior_name: str | None = None,
) -> BoutBoundaryMapping:
    """Map one inclusive dense bout interval onto prepared frames."""

    dur = (
        int(bout_duration_frames)
        if bout_duration_frames is not None
        else int(end_sample_idx) - int(start_sample_idx) + 1
    )
    frame = pd.DataFrame(
        [
            {
                "bout_id": bout_id,
                "start_sample_idx": int(start_sample_idx),
                "end_sample_idx": int(end_sample_idx),
                "bout_duration_frames": dur,
                "channel": channel,
                "behavior_name": behavior_name,
            }
        ]
    )
    mapped = map_bout_boundaries_to_prepared_frames(frame, sync_index, fps=fps)
    row = mapped.iloc[0]
    return BoutBoundaryMapping(
        bout_id=str(row["bout_id"]),
        start_sample_idx=int(row["start_sample_idx"]),
        end_sample_idx=int(row["end_sample_idx"]),
        bout_duration_frames=int(row["bout_duration_frames"]),
        bout_duration_sec=float(row["bout_duration_sec"]),
        prepared_frame_idx_start=int(row["prepared_frame_idx_start"]),
        prepared_frame_idx_end=int(row["prepared_frame_idx_end"]),
        prepared_time_sec_start=float(row["prepared_time_sec_start"]),
        prepared_time_sec_end=float(row["prepared_time_sec_end"]),
        channel=row["channel"],
        behavior_name=row["behavior_name"],
    )


def _empty_mapping_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bout_id",
            "channel",
            "behavior_name",
            "start_sample_idx",
            "end_sample_idx",
            "bout_duration_frames",
            "bout_duration_sec",
            "prepared_frame_idx_start",
            "prepared_frame_idx_end",
            "prepared_time_sec_start",
            "prepared_time_sec_end",
        ]
    )
