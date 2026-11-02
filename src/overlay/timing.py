"""Prepared-frame ↔ dense-state timing resolution for overlays.

Uses authoritative ``SyncMap`` / ``sync_index`` mappings. Does not invent
dense↔prepared relationships via independent FPS rounding.

Plateau policy note
-------------------
``SyncMap.mapping_from_prepared_frame`` implements **nearest-time** lookup:
among dense samples sharing a prepared frame, pick the sample whose
``time_sec`` is nearest to plateau ``prepared_time_sec`` (earlier
``sample_idx`` on residual ties). That contract serves interactive seek.

Offline prepared-rate overlays (C1 default, approved P10) use
**last-in-plateau** aggregation: among the same plateau members, take the
chronologically latest dense ``sample_idx``. This is not the same as
nearest-time lookup and must be recorded in the overlay manifest.

When no dense sample maps to a prepared frame, behavioral state is
unavailable — do not invent a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from behavioral_review.contracts import SyncMappingError
from behavioral_review.sync import PlayheadMapping, SyncMap
from overlay.contracts import OverlayTimingError, PlateauPolicy


@dataclass(frozen=True, slots=True)
class PreparedFrameTiming:
    """Resolved prepared-frame → dense behavioral timing."""

    prepared_frame_idx: int
    plateau_policy: PlateauPolicy
    available: bool
    dense_sample_idx: int | None
    plateau_sample_indices: tuple[int, ...]
    mapping: PlayheadMapping | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared_frame_idx": self.prepared_frame_idx,
            "plateau_policy": self.plateau_policy.value,
            "available": self.available,
            "dense_sample_idx": self.dense_sample_idx,
            "plateau_sample_indices": list(self.plateau_sample_indices),
            "mapping": None if self.mapping is None else self.mapping.to_dict(),
            "reason": self.reason,
        }


def resolve_prepared_frame_timing(
    sync: SyncMap,
    prepared_frame_idx: int,
    *,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
) -> PreparedFrameTiming:
    """Resolve dense behavioral authority for one prepared video frame.

    Parameters
    ----------
    sync:
        Authoritative dense↔prepared ``SyncMap``.
    prepared_frame_idx:
        Index in prepared video space.
    plateau_policy:
        Aggregation policy for many-to-one plateaus. C1 offline default is
        ``LAST_IN_PLATEAU``.
    """

    if not isinstance(prepared_frame_idx, (int, np.integer)) or isinstance(
        prepared_frame_idx, bool
    ):
        raise OverlayTimingError(
            f"prepared_frame_idx must be an integer; got {prepared_frame_idx!r}."
        )
    prep = int(prepared_frame_idx)
    if prep < 0:
        return PreparedFrameTiming(
            prepared_frame_idx=prep,
            plateau_policy=plateau_policy,
            available=False,
            dense_sample_idx=None,
            plateau_sample_indices=(),
            mapping=None,
            reason="prepared_frame_idx out of range (negative).",
        )

    try:
        plateau = sync.plateau_sample_indices(prep)
    except SyncMappingError as exc:
        return PreparedFrameTiming(
            prepared_frame_idx=prep,
            plateau_policy=plateau_policy,
            available=False,
            dense_sample_idx=None,
            plateau_sample_indices=(),
            mapping=None,
            reason=str(exc),
        )

    plateau_tuple = tuple(int(x) for x in plateau.tolist())
    if not plateau_tuple:
        return PreparedFrameTiming(
            prepared_frame_idx=prep,
            plateau_policy=plateau_policy,
            available=False,
            dense_sample_idx=None,
            plateau_sample_indices=(),
            mapping=None,
            reason="No dense sample mapped to prepared frame.",
        )

    try:
        chosen = _choose_plateau_sample(
            sync,
            prepared_frame_idx=prep,
            plateau=plateau_tuple,
            plateau_policy=plateau_policy,
        )
        mapping = sync.mapping_at_sample(chosen)
    except (SyncMappingError, OverlayTimingError) as exc:
        return PreparedFrameTiming(
            prepared_frame_idx=prep,
            plateau_policy=plateau_policy,
            available=False,
            dense_sample_idx=None,
            plateau_sample_indices=plateau_tuple,
            mapping=None,
            reason=str(exc),
        )

    return PreparedFrameTiming(
        prepared_frame_idx=prep,
        plateau_policy=plateau_policy,
        available=True,
        dense_sample_idx=int(chosen),
        plateau_sample_indices=plateau_tuple,
        mapping=mapping,
        reason=None,
    )


def _choose_plateau_sample(
    sync: SyncMap,
    *,
    prepared_frame_idx: int,
    plateau: tuple[int, ...],
    plateau_policy: PlateauPolicy,
) -> int:
    if plateau_policy is PlateauPolicy.LAST_IN_PLATEAU:
        # Chronological latest: sample_idx ascending ⇒ last member.
        return int(max(plateau))
    if plateau_policy is PlateauPolicy.NEAREST_TIME:
        mapped = sync.mapping_from_prepared_frame(prepared_frame_idx)
        return int(mapped.sample_idx)
    if plateau_policy is PlateauPolicy.PLAYHEAD:
        raise OverlayTimingError(
            "PLAYHEAD plateau policy requires an explicit playhead dense sample "
            "(interactive C3); not used for offline C1 resolution."
        )
    raise OverlayTimingError(f"Unsupported plateau policy: {plateau_policy!r}.")


def timing_mapping_identity(sync: SyncMap) -> dict[str, Any]:
    """Stable identity summary for the authoritative timing map (no rehash)."""

    audit = sync.audit_dict()
    return {
        "authority": "results_package_sync_index",
        "n_dense": audit["n_dense"],
        "n_prepared_unique": audit["n_prepared_unique"],
        "prepared_frame_idx_min": audit["prepared_frame_idx_min"],
        "prepared_frame_idx_max": audit["prepared_frame_idx_max"],
        "max_many_to_one": audit["max_many_to_one"],
        "dense_fps": float(sync.fps),
    }
