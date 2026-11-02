"""Per-frame overlay context assembled from timing + selected evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from overlay.contracts import FrameDomain, LayerId, PlateauPolicy
from overlay.style import DrawConfig
from overlay.timing import PreparedFrameTiming


@dataclass(frozen=True, slots=True)
class OverlayObservation:
    """Geometry / labels / availability for one layer at one context."""

    layer_id: LayerId
    available: bool
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id.value,
            "available": self.available,
            "reason": self.reason,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class OverlayEvidenceBundle:
    """Resolved observations keyed by layer id for one prepared frame."""

    prepared_frame_idx: int
    timing: PreparedFrameTiming
    observations: dict[LayerId, OverlayObservation]

    def get(self, layer_id: LayerId) -> OverlayObservation | None:
        return self.observations.get(layer_id)

    def unavailable_layers(self, selected: tuple[LayerId, ...]) -> tuple[LayerId, ...]:
        out: list[LayerId] = []
        for layer in selected:
            obs = self.observations.get(layer)
            if obs is None or not obs.available:
                out.append(layer)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class OverlayContext:
    """Frame-local context passed to layer drawers."""

    prepared_frame_idx: int
    timing: PreparedFrameTiming
    frame_domain: FrameDomain
    plateau_policy: PlateauPolicy
    style: DrawConfig
    selected_layers: tuple[LayerId, ...]

    @property
    def dense_sample_idx(self) -> int | None:
        return self.timing.dense_sample_idx

    @property
    def behavioral_available(self) -> bool:
        return bool(self.timing.available and self.timing.dense_sample_idx is not None)
