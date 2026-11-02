"""Minimal pose layer drawers for C1 (final styling deferred to C2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from overlay.context import OverlayContext, OverlayObservation
from overlay.contracts import LayerId
from tracking_correction.overlay_draw import draw_pose_overlay
from tracking_correction.review import PoseNodePoint


class PoseLayer:
    """Draw prepared- or dense-domain pose points via shared S3 drawer."""

    def __init__(self, layer_id: LayerId) -> None:
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
        points_raw = observation.payload.get("points") or []
        node_points = _to_node_points(points_raw, layer_id=self._layer_id)
        if not node_points:
            return frame_bgr
        edges: Sequence[tuple[str, str]] = ()
        if context.style.draw_skeleton:
            # Topology resolved by callers in C2; C1 keeps edges empty by default.
            edges = tuple(context.style.extra.get("skeleton_edges", ()))
        canvas = draw_pose_overlay(
            frame_bgr,
            node_points,
            skeleton_edges=edges,
            node_radius=context.style.node_radius,
            edge_thickness=context.style.edge_thickness,
        )
        if context.style.legend_text:
            legend = str(observation.payload.get("legend") or self._layer_id.value)
            _draw_legend(canvas, legend, y=20)
        return canvas


def _to_node_points(
    points_raw: Sequence[dict[str, Any]],
    *,
    layer_id: LayerId,
) -> list[PoseNodePoint]:
    out: list[PoseNodePoint] = []
    for item in points_raw:
        if layer_id is LayerId.S4_POSE_FINAL and not bool(item.get("drawable", True)):
            continue
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        # C1 draws drawable points with track color; S4 provenance styling → C2.
        out.append(
            PoseNodePoint(
                track=int(item["track"]),
                node=str(item["node"]),
                x=float(x),
                y=float(y),
            )
        )
    return out


def _draw_legend(frame_bgr: np.ndarray, text: str, *, y: int) -> None:
    cv2.putText(
        frame_bgr,
        text,
        (8, int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        lineType=cv2.LINE_AA,
    )
