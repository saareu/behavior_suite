"""Browse, filter, and navigate authoritative S9 packaged bouts (read-only).

Uses the full ``behavior_bouts.parquet`` table — never the optional
``review_emit_v1`` subset — as scientific inspection truth. Filters are
display/navigation only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    BOUT_BROWSER_SCHEMA_VERSION,
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    CHANNEL_VALUES,
    IDENTITY_DIRECTIONAL,
    IDENTITY_SYMMETRIC_PAIR,
    InspectionError,
)
from behavioral_review.sync import BoutSeekMapping, SyncMap


@dataclass(frozen=True, slots=True)
class BoutFilter:
    """Display-only bout filter (does not mutate packaged rows)."""

    channels: frozenset[str] | None = None
    behaviors: frozenset[str] | None = None
    focal_id: int | None = None
    partner_id: int | None = None
    mouse_a_id: int | None = None
    mouse_b_id: int | None = None
    pair_id: str | None = None
    identity_grain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": sorted(self.channels) if self.channels is not None else None,
            "behaviors": sorted(self.behaviors) if self.behaviors is not None else None,
            "focal_id": self.focal_id,
            "partner_id": self.partner_id,
            "mouse_a_id": self.mouse_a_id,
            "mouse_b_id": self.mouse_b_id,
            "pair_id": self.pair_id,
            "identity_grain": self.identity_grain,
        }


@dataclass(frozen=True, slots=True)
class BoutOverlay:
    """Inclusive dense bout span for timeline overlays + prepared seek aids."""

    bout_id: str
    source_s8_bout_id: str | None
    channel: str
    identity_grain: str
    behavior_name: str
    focal_id: int | None
    partner_id: int | None
    mouse_a_id: int | None
    mouse_b_id: int | None
    start_sample_idx: int
    end_sample_idx: int
    bout_duration_frames: int
    bout_duration_sec: float
    prepared_frame_idx_start: int
    prepared_frame_idx_end: int
    prepared_time_sec_start: float
    prepared_time_sec_end: float
    label_reconstruction_mode: str | None
    undefined_reason: str | None
    selected: bool

    @property
    def inclusive_endpoints(self) -> tuple[int, int]:
        return self.start_sample_idx, self.end_sample_idx


@dataclass(frozen=True, slots=True)
class BoutNavigationState:
    """Stable selection + active filter snapshot (UI state only)."""

    schema_version: str
    selected_bout_id: str | None
    active_filter: BoutFilter
    matching_bout_ids: tuple[str, ...]
    matching_count: int


class BoutBrowser:
    """In-memory navigation index over packaged S9 bouts."""

    def __init__(self, behavior_bouts: pd.DataFrame, sync: SyncMap) -> None:
        if behavior_bouts is None or behavior_bouts.empty:
            raise InspectionError("Packaged behavior_bouts is empty; cannot browse.")
        if "bout_id" not in behavior_bouts.columns:
            raise InspectionError("Packaged bouts missing bout_id.")
        self._sync = sync
        self._bouts = behavior_bouts.reset_index(drop=True).copy()
        self._bouts["bout_id"] = self._bouts["bout_id"].astype(str)
        duplicates = self._bouts["bout_id"][self._bouts["bout_id"].duplicated()].tolist()
        if duplicates:
            raise InspectionError(f"Duplicate packaged bout_id values: {duplicates[:5]}")
        self._by_id = {
            str(row.bout_id): row
            for row in self._bouts.itertuples(index=False)
        }
        # Precompute arrays for overlap queries.
        self._start = self._bouts["start_sample_idx"].to_numpy(dtype=np.int64)
        self._end = self._bouts["end_sample_idx"].to_numpy(dtype=np.int64)
        self._ids = self._bouts["bout_id"].astype(str).to_numpy()
        self._selected_bout_id: str | None = None
        self._active_filter = BoutFilter()

    @property
    def n_bouts(self) -> int:
        return int(len(self._bouts))

    @property
    def selected_bout_id(self) -> str | None:
        return self._selected_bout_id

    @property
    def active_filter(self) -> BoutFilter:
        return self._active_filter

    def set_filter(self, bout_filter: BoutFilter | None) -> BoutNavigationState:
        self._active_filter = bout_filter or BoutFilter()
        return self.navigation_state()

    def clear_filter(self) -> BoutNavigationState:
        return self.set_filter(None)

    def list_bouts(self, bout_filter: BoutFilter | None = None) -> pd.DataFrame:
        """Return matching bout rows (full science columns; filter is display-only)."""

        filt = bout_filter if bout_filter is not None else self._active_filter
        mask = self._filter_mask(filt)
        return self._bouts.loc[mask].reset_index(drop=True)

    def select(self, bout_id: str | None) -> pd.Series | None:
        if bout_id is None:
            self._selected_bout_id = None
            return None
        row = self.get(bout_id)
        self._selected_bout_id = str(bout_id)
        return row

    def get(self, bout_id: str) -> pd.Series:
        hits = self._bouts.loc[self._bouts["bout_id"] == str(bout_id)]
        if hits.empty:
            raise InspectionError(f"Unknown bout_id {bout_id!r}.")
        return hits.iloc[0]

    def overlay(
        self,
        bout_id: str | None = None,
        *,
        selected: bool | None = None,
    ) -> BoutOverlay:
        target = bout_id if bout_id is not None else self._selected_bout_id
        if target is None:
            raise InspectionError("No bout_id provided and no bout is selected.")
        row = self.get(target)
        is_selected = (
            bool(selected)
            if selected is not None
            else (self._selected_bout_id == str(target))
        )
        return _row_to_overlay(row, selected=is_selected)

    def overlays_for_ids(
        self,
        bout_ids: Sequence[str],
        *,
        selected_bout_id: str | None = None,
    ) -> tuple[BoutOverlay, ...]:
        sel = selected_bout_id if selected_bout_id is not None else self._selected_bout_id
        return tuple(
            self.overlay(bid, selected=(str(bid) == str(sel)) if sel is not None else False)
            for bid in bout_ids
        )

    def overlapping(
        self,
        start_sample_idx: int,
        end_sample_idx: int,
        *,
        bout_filter: BoutFilter | None = None,
    ) -> pd.DataFrame:
        """Bouts whose inclusive dense interval intersects ``[start, end]``."""

        start = int(start_sample_idx)
        end = int(end_sample_idx)
        if start > end:
            raise InspectionError(f"Invalid overlap window: start ({start}) > end ({end}).")
        filt = bout_filter if bout_filter is not None else self._active_filter
        mask = self._filter_mask(filt)
        # Inclusive interval overlap: start_i <= end and end_i >= start
        overlap = (self._start <= end) & (self._end >= start)
        return self._bouts.loc[mask & overlap].reset_index(drop=True)

    def previous_bout(
        self,
        bout_id: str | None = None,
        *,
        bout_filter: BoutFilter | None = None,
    ) -> str | None:
        ordered = self._ordered_matching_ids(bout_filter)
        if not ordered:
            return None
        current = bout_id if bout_id is not None else self._selected_bout_id
        if current is None:
            return ordered[-1]
        try:
            idx = ordered.index(str(current))
        except ValueError:
            return None
        return ordered[idx - 1] if idx > 0 else None

    def next_bout(
        self,
        bout_id: str | None = None,
        *,
        bout_filter: BoutFilter | None = None,
    ) -> str | None:
        ordered = self._ordered_matching_ids(bout_filter)
        if not ordered:
            return None
        current = bout_id if bout_id is not None else self._selected_bout_id
        if current is None:
            return ordered[0]
        try:
            idx = ordered.index(str(current))
        except ValueError:
            return None
        return ordered[idx + 1] if idx + 1 < len(ordered) else None

    def jump_to_start(self, bout_id: str | None = None) -> BoutSeekMapping:
        row = self.get(bout_id if bout_id is not None else self._require_selected())
        self._selected_bout_id = str(row["bout_id"])
        return self._seek_mapping(row)

    def jump_to_end(self, bout_id: str | None = None) -> BoutSeekMapping:
        return self.jump_to_start(bout_id)  # mapping carries both endpoints

    def jump_to_sample(
        self,
        sample_idx: int,
        *,
        bout_filter: BoutFilter | None = None,
        prefer_selected_channel: str | None = None,
    ) -> str | None:
        """Select a bout covering ``sample_idx`` (stable: earliest start, then bout_id)."""

        covering = self.overlapping(
            int(sample_idx), int(sample_idx), bout_filter=bout_filter
        )
        if covering.empty:
            return None
        if prefer_selected_channel is not None:
            channel_hits = covering.loc[covering["channel"] == prefer_selected_channel]
            if not channel_hits.empty:
                covering = channel_hits
        covering = covering.sort_values(
            ["start_sample_idx", "bout_id"], kind="mergesort"
        )
        bout_id = str(covering.iloc[0]["bout_id"])
        self._selected_bout_id = bout_id
        return bout_id

    def navigation_annotations(self, bout_id: str | None = None) -> BoutSeekMapping:
        row = self.get(bout_id if bout_id is not None else self._require_selected())
        return self._seek_mapping(row)

    def navigation_state(self) -> BoutNavigationState:
        ids = tuple(self._ordered_matching_ids(self._active_filter))
        return BoutNavigationState(
            schema_version=BOUT_BROWSER_SCHEMA_VERSION,
            selected_bout_id=self._selected_bout_id,
            active_filter=self._active_filter,
            matching_bout_ids=ids,
            matching_count=len(ids),
        )

    def channel_counts(self) -> dict[str, int]:
        counts = self._bouts["channel"].astype(str).value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def _seek_mapping(self, row: pd.Series) -> BoutSeekMapping:
        mapped = self._sync.bout_seek(
            start_sample_idx=int(row["start_sample_idx"]),
            end_sample_idx=int(row["end_sample_idx"]),
            bout_id=str(row["bout_id"]),
            bout_duration_frames=int(row["bout_duration_frames"]),
        )
        pkg_start = int(row["prepared_frame_idx_start"])
        pkg_end = int(row["prepared_frame_idx_end"])
        if (
            pkg_start != mapped.prepared_frame_idx_start
            or pkg_end != mapped.prepared_frame_idx_end
        ):
            raise InspectionError(
                f"Packaged prepared endpoints for bout {row['bout_id']!r} disagree "
                f"with sync_index "
                f"(packaged=[{pkg_start},{pkg_end}] "
                f"sync=[{mapped.prepared_frame_idx_start},"
                f"{mapped.prepared_frame_idx_end}])."
            )
        return mapped

    def _ordered_matching_ids(self, bout_filter: BoutFilter | None) -> list[str]:
        filt = bout_filter if bout_filter is not None else self._active_filter
        matched = self.list_bouts(filt)
        if matched.empty:
            return []
        ordered = matched.sort_values(
            ["start_sample_idx", "end_sample_idx", "bout_id"], kind="mergesort"
        )
        return [str(x) for x in ordered["bout_id"].tolist()]

    def _filter_mask(self, bout_filter: BoutFilter) -> np.ndarray:
        mask = np.ones(len(self._bouts), dtype=bool)
        if bout_filter.channels is not None:
            unknown = set(bout_filter.channels) - CHANNEL_VALUES
            if unknown:
                raise InspectionError(f"Unknown bout channels in filter: {sorted(unknown)}")
            mask &= self._bouts["channel"].astype(str).isin(bout_filter.channels).to_numpy()
        if bout_filter.behaviors is not None:
            mask &= (
                self._bouts["behavior_name"]
                .astype(str)
                .isin(bout_filter.behaviors)
                .to_numpy()
            )
        if bout_filter.identity_grain is not None:
            if bout_filter.identity_grain not in (
                IDENTITY_DIRECTIONAL,
                IDENTITY_SYMMETRIC_PAIR,
            ):
                raise InspectionError(
                    f"Unknown identity_grain {bout_filter.identity_grain!r}."
                )
            mask &= (
                self._bouts["identity_grain"].astype(str) == bout_filter.identity_grain
            ).to_numpy()
        if bout_filter.focal_id is not None:
            mask &= (
                pd.to_numeric(self._bouts["focal_id"], errors="coerce")
                == int(bout_filter.focal_id)
            ).to_numpy()
        if bout_filter.partner_id is not None:
            mask &= (
                pd.to_numeric(self._bouts["partner_id"], errors="coerce")
                == int(bout_filter.partner_id)
            ).to_numpy()
        if bout_filter.mouse_a_id is not None:
            mask &= (
                pd.to_numeric(self._bouts["mouse_a_id"], errors="coerce")
                == int(bout_filter.mouse_a_id)
            ).to_numpy()
        if bout_filter.mouse_b_id is not None:
            mask &= (
                pd.to_numeric(self._bouts["mouse_b_id"], errors="coerce")
                == int(bout_filter.mouse_b_id)
            ).to_numpy()
        if bout_filter.pair_id is not None:
            mask &= (self._bouts["pair_id"].astype(str) == str(bout_filter.pair_id)).to_numpy()
        return mask

    def _require_selected(self) -> str:
        if self._selected_bout_id is None:
            raise InspectionError("No bout is selected.")
        return self._selected_bout_id


def _row_to_overlay(row: pd.Series, *, selected: bool) -> BoutOverlay:
    channel = str(row["channel"])
    grain = str(row["identity_grain"])
    focal = row["focal_id"]
    partner = row["partner_id"]
    mouse_a = row["mouse_a_id"]
    mouse_b = row["mouse_b_id"]
    # Keep contact/SBS scientifically separate from exclusive directional.
    if channel in (CHANNEL_CONTACT, CHANNEL_SIDE_BY_SIDE):
        grain = IDENTITY_SYMMETRIC_PAIR
    elif channel == CHANNEL_EXCLUSIVE:
        grain = IDENTITY_DIRECTIONAL
    return BoutOverlay(
        bout_id=str(row["bout_id"]),
        source_s8_bout_id=(
            str(row["source_s8_bout_id"]) if pd.notna(row.get("source_s8_bout_id")) else None
        ),
        channel=channel,
        identity_grain=grain,
        behavior_name=str(row["behavior_name"]),
        focal_id=int(focal) if pd.notna(focal) else None,
        partner_id=int(partner) if pd.notna(partner) else None,
        mouse_a_id=int(mouse_a) if pd.notna(mouse_a) else None,
        mouse_b_id=int(mouse_b) if pd.notna(mouse_b) else None,
        start_sample_idx=int(row["start_sample_idx"]),
        end_sample_idx=int(row["end_sample_idx"]),
        bout_duration_frames=int(row["bout_duration_frames"]),
        bout_duration_sec=float(row["bout_duration_sec"]),
        prepared_frame_idx_start=int(row["prepared_frame_idx_start"]),
        prepared_frame_idx_end=int(row["prepared_frame_idx_end"]),
        prepared_time_sec_start=float(row["prepared_time_sec_start"]),
        prepared_time_sec_end=float(row["prepared_time_sec_end"]),
        label_reconstruction_mode=(
            str(row["label_reconstruction_mode"])
            if "label_reconstruction_mode" in row.index
            and pd.notna(row["label_reconstruction_mode"])
            else None
        ),
        undefined_reason=(
            str(row["undefined_reason"])
            if "undefined_reason" in row.index and pd.notna(row["undefined_reason"])
            else None
        ),
        selected=selected,
    )
