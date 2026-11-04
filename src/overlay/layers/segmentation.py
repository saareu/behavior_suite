"""S5 blob-segmentation overlay drawer (C3-3 replay QC)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from blob_features.preview import draw_contour_overlay
from overlay.context import OverlayContext, OverlayObservation
from overlay.contracts import LayerId


class SegmentationLayer:
    """Draw S5 replay contours / optional translucent masks / scalar text."""

    def __init__(self, layer_id: LayerId = LayerId.S5_SEGMENTATION) -> None:
        self._layer_id = layer_id

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    def render(
        self,
        frame_bgr: np.ndarray,
        context: OverlayContext,
        observation: OverlayObservation,
    ) -> np.ndarray:
        if not observation.available:
            return frame_bgr

        style_extra = dict(context.style.extra)
        show_contours = bool(style_extra.get("show_contours", True))
        show_masks = bool(style_extra.get("show_masks", False))
        show_scalars = bool(style_extra.get("show_scalar_text", False))
        mask_opacity = float(style_extra.get("mask_opacity", 0.35))
        contour_thickness = int(style_extra.get("contour_thickness", 2))
        contour_color = _as_bgr(style_extra.get("contour_color"), default=(0, 255, 0))
        mask_color = _as_bgr(style_extra.get("mask_color"), default=(0, 180, 255))

        canvas = frame_bgr
        mask = observation.payload.get("mask")
        if show_masks and mask is not None:
            canvas = _draw_translucent_mask(
                canvas,
                np.asarray(mask),
                color=mask_color,
                opacity=mask_opacity,
            )

        contours_raw = observation.payload.get("contours") or ()
        contours = tuple(np.asarray(c) for c in contours_raw)
        if show_contours:
            canvas = draw_contour_overlay(
                canvas,
                contours,
                color=contour_color,
                thickness=contour_thickness,
            )

        if context.style.legend_text:
            _draw_legend(canvas, observation.payload)

        if show_scalars:
            _draw_scalar_text(canvas, observation.payload)

        return canvas


def _draw_translucent_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    if mask.ndim != 2 or mask.shape[:2] != frame_bgr.shape[:2]:
        return frame_bgr
    alpha = min(1.0, max(0.0, float(opacity)))
    if alpha <= 0.0:
        return frame_bgr
    overlay = frame_bgr.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)


def _draw_legend(frame_bgr: np.ndarray, payload: dict[str, Any]) -> None:
    legend = str(payload.get("legend") or "S5 segmentation replay QC")
    cv2.putText(
        frame_bgr,
        legend,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        legend,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 160, 20),
        1,
        lineType=cv2.LINE_AA,
    )


def _draw_scalar_text(frame_bgr: np.ndarray, payload: dict[str, Any]) -> None:
    cc_raw = payload.get("cc_count_raw")
    cc = payload.get("cc_count")
    dist = payload.get("min_inter_contour_dist")
    parts: list[str] = []
    if cc_raw is not None:
        parts.append(f"cc_raw={int(cc_raw)}")
    if cc is not None:
        try:
            cc_f = float(cc)
            parts.append("cc=nan" if cc_f != cc_f else f"cc={cc_f:g}")
        except (TypeError, ValueError):
            pass
    if dist is not None:
        try:
            dist_f = float(dist)
            parts.append(
                "min_dist=nan" if dist_f != dist_f else f"min_dist={dist_f:.1f}"
            )
        except (TypeError, ValueError):
            pass
    if not parts:
        return
    text = " · ".join(parts)
    cv2.putText(
        frame_bgr,
        text,
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        lineType=cv2.LINE_AA,
    )


def _as_bgr(value: object, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError):
            return default
    return default


__all__ = ["SegmentationLayer"]
