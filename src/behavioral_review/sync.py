"""Dense ↔ prepared synchronization using authoritative S9 ``sync_index``.

Does not rebuild S1/S5 nearest-neighbor maps. Dense ``sample_idx`` is the
behavioral authority; prepared indices are decode targets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import SyncMappingError
from results_packaging.contracts import (
    SYNC_INDEX_REQUIRED_COLUMNS,
    TARGET_FPS,
    TIME_ATOL_SEC,
)
from results_packaging.sync_index import validate_sync_index


@dataclass(frozen=True, slots=True)
class PlayheadMapping:
    """One dense sample mapped onto prepared-video coordinates."""

    sample_idx: int
    dense_frame_idx: int
    time_sec: float
    prepared_frame_idx: int
    prepared_time_sec: float
    is_exact_source_sample: bool
    timing_residual_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_idx": self.sample_idx,
            "dense_frame_idx": self.dense_frame_idx,
            "time_sec": self.time_sec,
            "prepared_frame_idx": self.prepared_frame_idx,
            "prepared_time_sec": self.prepared_time_sec,
            "is_exact_source_sample": self.is_exact_source_sample,
            "timing_residual_sec": self.timing_residual_sec,
        }


@dataclass(frozen=True, slots=True)
class BoutSeekMapping:
    """Inclusive bout endpoints mapped for seeking (science unchanged)."""

    bout_id: str | None
    start_sample_idx: int
    end_sample_idx: int
    bout_duration_frames: int
    start: PlayheadMapping
    end: PlayheadMapping

    @property
    def prepared_frame_idx_start(self) -> int:
        return self.start.prepared_frame_idx

    @property
    def prepared_frame_idx_end(self) -> int:
        return self.end.prepared_frame_idx


@dataclass(frozen=True, slots=True)
class SyncMap:
    """Validated dense↔prepared map backed by S9 ``sync_index`` arrays."""

    sync_index: pd.DataFrame
    n_dense: int
    sample_idx: np.ndarray
    dense_frame_idx: np.ndarray
    time_sec: np.ndarray
    prepared_frame_idx: np.ndarray
    prepared_time_sec: np.ndarray
    is_exact_source_sample: np.ndarray
    timing_residual_sec: np.ndarray
    fps: float = TARGET_FPS

    @classmethod
    def from_sync_index(
        cls,
        sync_index: pd.DataFrame,
        *,
        fps: float = TARGET_FPS,
    ) -> SyncMap:
        if sync_index is None or sync_index.empty:
            raise SyncMappingError("sync_index is empty.")
        missing = [c for c in SYNC_INDEX_REQUIRED_COLUMNS if c not in sync_index.columns]
        if missing:
            raise SyncMappingError(f"sync_index missing required columns: {missing}")
        try:
            validate_sync_index(sync_index)
        except Exception as exc:
            raise SyncMappingError(f"Invalid sync_index: {exc}") from exc

        ordered = sync_index.sort_values("sample_idx", kind="mergesort").reset_index(
            drop=True
        )
        return cls(
            sync_index=ordered,
            n_dense=int(len(ordered)),
            sample_idx=ordered["sample_idx"].to_numpy(dtype=np.int64),
            dense_frame_idx=ordered["dense_frame_idx"].to_numpy(dtype=np.int64),
            time_sec=ordered["time_sec"].to_numpy(dtype=np.float64),
            prepared_frame_idx=ordered["prepared_frame_idx"].to_numpy(dtype=np.int64),
            prepared_time_sec=ordered["prepared_time_sec"].to_numpy(dtype=np.float64),
            is_exact_source_sample=ordered["is_exact_source_sample"].to_numpy(dtype=bool),
            timing_residual_sec=ordered["timing_residual_sec"].to_numpy(dtype=np.float64),
            fps=float(fps),
        )

    def mapping_at_sample(self, sample_idx: int) -> PlayheadMapping:
        idx = _require_int(sample_idx, name="sample_idx")
        if idx < 0 or idx >= self.n_dense:
            raise SyncMappingError(
                f"sample_idx {idx} out of range [0, {self.n_dense - 1}]."
            )
        return self._mapping_at_row(idx)

    def mapping_at_dense_frame(self, dense_frame_idx: int) -> PlayheadMapping:
        # Contiguous identity: dense_frame_idx == sample_idx.
        return self.mapping_at_sample(dense_frame_idx)

    def mapping_at_time_sec(self, time_sec: float) -> PlayheadMapping:
        """Nearest dense sample by analysis time; earlier sample on ties."""

        if not np.isfinite(time_sec):
            raise SyncMappingError(f"time_sec must be finite; got {time_sec!r}.")
        times = self.time_sec
        # Insertion point for exact/nearest search.
        pos = int(np.searchsorted(times, float(time_sec), side="left"))
        candidates: list[int] = []
        if 0 <= pos < self.n_dense:
            candidates.append(pos)
        if pos - 1 >= 0:
            candidates.append(pos - 1)
        if not candidates:
            raise SyncMappingError("sync_index has no samples for time lookup.")
        # Choose minimum absolute residual; tie → earlier sample_idx.
        best = min(
            candidates,
            key=lambda i: (abs(float(times[i]) - float(time_sec)), i),
        )
        return self._mapping_at_row(best)

    def mapping_from_prepared_frame(self, prepared_frame_idx: int) -> PlayheadMapping:
        """Map a prepared frame onto a dense sample on its plateau.

        Selects the dense sample on the plateau whose ``time_sec`` is nearest
        to ``prepared_time_sec``; earlier ``sample_idx`` wins ties (S10-B §3.3).
        """

        prep = _require_int(prepared_frame_idx, name="prepared_frame_idx")
        hits = np.flatnonzero(self.prepared_frame_idx == prep)
        if hits.size == 0:
            raise SyncMappingError(
                f"prepared_frame_idx {prep} is not present in sync_index."
            )
        # All prepared_time_sec on a plateau should match; use first as target.
        target_time = float(self.prepared_time_sec[hits[0]])
        dense_times = self.time_sec[hits]
        residuals = np.abs(dense_times - target_time)
        min_res = float(np.min(residuals))
        # Earlier sample_idx wins ties (hits are ascending by sample_idx).
        equals = np.flatnonzero(np.abs(residuals - min_res) <= TIME_ATOL_SEC)
        chosen_local = int(equals[0])
        return self._mapping_at_row(int(hits[chosen_local]))

    def plateau_sample_indices(self, prepared_frame_idx: int) -> np.ndarray:
        """Return all dense ``sample_idx`` values sharing ``prepared_frame_idx``."""

        prep = _require_int(prepared_frame_idx, name="prepared_frame_idx")
        hits = np.flatnonzero(self.prepared_frame_idx == prep)
        if hits.size == 0:
            raise SyncMappingError(
                f"prepared_frame_idx {prep} is not present in sync_index."
            )
        return self.sample_idx[hits].copy()

    def bout_seek(
        self,
        *,
        start_sample_idx: int,
        end_sample_idx: int,
        bout_id: str | None = None,
        bout_duration_frames: int | None = None,
    ) -> BoutSeekMapping:
        """Seek mapping for inclusive dense bout endpoints."""

        start = _require_int(start_sample_idx, name="start_sample_idx")
        end = _require_int(end_sample_idx, name="end_sample_idx")
        if start > end:
            raise SyncMappingError(
                f"Invalid bout endpoints: start ({start}) > end ({end})."
            )
        expected = end - start + 1
        if bout_duration_frames is not None and int(bout_duration_frames) != expected:
            raise SyncMappingError(
                f"bout_duration_frames ({bout_duration_frames}) != inclusive "
                f"count ({expected})."
            )
        return BoutSeekMapping(
            bout_id=bout_id,
            start_sample_idx=start,
            end_sample_idx=end,
            bout_duration_frames=expected,
            start=self.mapping_at_sample(start),
            end=self.mapping_at_sample(end),
        )

    def audit_dict(self) -> dict[str, Any]:
        prepared = self.prepared_frame_idx
        unique = np.unique(prepared)
        return {
            "n_dense": self.n_dense,
            "n_prepared_unique": int(unique.shape[0]),
            "prepared_frame_idx_min": int(prepared.min()) if self.n_dense else None,
            "prepared_frame_idx_max": int(prepared.max()) if self.n_dense else None,
            "max_many_to_one": (
                int(np.max(np.unique(prepared, return_counts=True)[1]))
                if self.n_dense
                else 0
            ),
            "fps": self.fps,
        }

    def _mapping_at_row(self, row: int) -> PlayheadMapping:
        return PlayheadMapping(
            sample_idx=int(self.sample_idx[row]),
            dense_frame_idx=int(self.dense_frame_idx[row]),
            time_sec=float(self.time_sec[row]),
            prepared_frame_idx=int(self.prepared_frame_idx[row]),
            prepared_time_sec=float(self.prepared_time_sec[row]),
            is_exact_source_sample=bool(self.is_exact_source_sample[row]),
            timing_residual_sec=float(self.timing_residual_sec[row]),
        )


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SyncMappingError(f"{name} must be an integer; got {value!r}.")
    return int(value)
