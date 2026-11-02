"""Offline headless frame compositor (C1 foundation).

Accepts a prepared-domain frame, prepared frame index, resolved evidence,
explicitly selected layers, and drawing configuration. Returns a rendered
copy without modifying source arrays or scientific artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from overlay.context import OverlayContext, OverlayEvidenceBundle
from overlay.contracts import (
    FrameDomain,
    LayerId,
    OverlayContractError,
    OverlayRenderError,
    PlateauPolicy,
    validate_layer_selection,
)
from overlay.layers import default_layer_registry, draw_unavailable_badge
from overlay.layers.base import OverlayLayer
from overlay.style import DrawConfig


@dataclass(frozen=True, slots=True)
class ComposedFrame:
    """Rendered overlay frame plus composition metadata."""

    frame_bgr: np.ndarray
    prepared_frame_idx: int
    selected_layers: tuple[LayerId, ...]
    rendered_layers: tuple[LayerId, ...]
    unavailable_layers: tuple[LayerId, ...]
    timing_available: bool
    dense_sample_idx: int | None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared_frame_idx": self.prepared_frame_idx,
            "selected_layers": [x.value for x in self.selected_layers],
            "rendered_layers": [x.value for x in self.rendered_layers],
            "unavailable_layers": [x.value for x in self.unavailable_layers],
            "timing_available": self.timing_available,
            "dense_sample_idx": self.dense_sample_idx,
            "notes": list(self.notes),
            "frame_shape": list(self.frame_bgr.shape),
        }


@dataclass
class OverlayCompositor:
    """Ordered offline compositor over independently selectable layers."""

    layers: Mapping[LayerId, OverlayLayer] = field(
        default_factory=default_layer_registry
    )
    frame_domain: FrameDomain = FrameDomain.PREPARED
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        prepared_frame_idx: int,
        evidence: OverlayEvidenceBundle,
        selected_layers: Sequence[str | LayerId],
        style: DrawConfig | None = None,
    ) -> ComposedFrame:
        """Compose selected layers onto a copy of ``frame_bgr``.

        ``frame_bgr`` must be prepared-video pixels (HxWx3). Source array is
        never modified. Color convention matches existing OpenCV drawers (BGR).
        """

        if evidence.prepared_frame_idx != int(prepared_frame_idx):
            raise OverlayContractError(
                "Evidence prepared_frame_idx does not match compositor request."
            )
        selected = validate_layer_selection(list(selected_layers))
        style = style or DrawConfig()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise OverlayRenderError(
                f"Expected HxWx3 prepared frame; got shape {frame_bgr.shape}."
            )
        source_ptr = frame_bgr.__array_interface__["data"][0]
        canvas = np.ascontiguousarray(frame_bgr.copy())
        if canvas.dtype != np.uint8:
            canvas = canvas.astype(np.uint8, copy=False)

        if style.darken_alpha < 1.0:
            alpha = float(style.darken_alpha)
            canvas = cv2_add_weighted_black(canvas, alpha)

        context = OverlayContext(
            prepared_frame_idx=int(prepared_frame_idx),
            timing=evidence.timing,
            frame_domain=self.frame_domain,
            plateau_policy=self.plateau_policy,
            style=style,
            selected_layers=selected,
        )

        rendered: list[LayerId] = []
        unavailable: list[LayerId] = []
        notes: list[str] = []

        for layer_id in selected:
            obs = evidence.get(layer_id)
            drawer = self.layers.get(layer_id)
            if obs is None:
                unavailable.append(layer_id)
                if style.unavailable_badge:
                    draw_unavailable_badge(
                        canvas, layer_id, "Missing observation in evidence bundle."
                    )
                notes.append(f"{layer_id.value}: missing observation")
                continue
            if not obs.available:
                unavailable.append(layer_id)
                notes.append(f"{layer_id.value}: {obs.reason or 'unavailable'}")
                # Behavioral HUD drawers render unavailable/unknown in-place so
                # operators can distinguish missing labels from "no behavior".
                if drawer is not None and getattr(
                    drawer, "renders_when_unavailable", False
                ):
                    canvas = drawer.render(canvas, context, obs)
                elif style.unavailable_badge:
                    draw_unavailable_badge(canvas, layer_id, obs.reason)
                continue
            if drawer is None:
                unavailable.append(layer_id)
                if style.unavailable_badge:
                    draw_unavailable_badge(
                        canvas, layer_id, "No drawer registered for layer."
                    )
                notes.append(f"{layer_id.value}: no drawer")
                continue
            canvas = drawer.render(canvas, context, obs)
            rendered.append(layer_id)

        # Immutability guard: source buffer address unchanged / content intact.
        if frame_bgr.__array_interface__["data"][0] != source_ptr:
            raise OverlayRenderError("Source frame buffer identity changed during compose.")

        return ComposedFrame(
            frame_bgr=canvas,
            prepared_frame_idx=int(prepared_frame_idx),
            selected_layers=selected,
            rendered_layers=tuple(rendered),
            unavailable_layers=tuple(unavailable),
            timing_available=bool(evidence.timing.available),
            dense_sample_idx=evidence.timing.dense_sample_idx,
            notes=tuple(notes),
        )


def cv2_add_weighted_black(frame_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Darken toward black without importing cv2 at module import for tests."""

    import cv2

    black = np.zeros_like(frame_bgr)
    return cv2.addWeighted(frame_bgr, float(alpha), black, 1.0 - float(alpha), 0.0)
