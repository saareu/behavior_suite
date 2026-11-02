"""Layer drawing protocol for the shared compositor."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from overlay.context import OverlayContext, OverlayObservation
from overlay.contracts import LayerId


class OverlayLayer(Protocol):
    """Pure draw function over a BGR canvas copy (never mutates science)."""

    @property
    def layer_id(self) -> LayerId: ...

    def render(
        self,
        frame_bgr: np.ndarray,
        context: OverlayContext,
        observation: OverlayObservation,
    ) -> np.ndarray: ...
