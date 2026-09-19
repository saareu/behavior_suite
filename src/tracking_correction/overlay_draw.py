"""Interactive pose overlay helpers for S3 review (no review-video export)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from tracking_correction.review import TRACK_COLORS_BGR, PoseNodePoint


def draw_pose_overlay(
    frame: np.ndarray,
    points: Sequence[PoseNodePoint],
    *,
    skeleton_edges: Sequence[tuple[str, str]] = (),
    node_radius: int = 4,
    edge_thickness: int = 2,
) -> np.ndarray:
    """Return a BGR copy of ``frame`` with identity-colored pose overlays.

    Edges are drawn only from ``skeleton_edges``. An empty edge sequence draws
    nodes without connections (no guessed topology).
    """

    canvas = np.ascontiguousarray(frame.copy())
    by_track: dict[int, dict[str, PoseNodePoint]] = {}
    for point in points:
        by_track.setdefault(point.track, {})[point.node] = point

    for track, nodes in by_track.items():
        color = TRACK_COLORS_BGR.get(track, (200, 200, 200))
        for start_name, end_name in skeleton_edges:
            start = nodes.get(start_name)
            end = nodes.get(end_name)
            if start is None or end is None:
                continue
            cv2.line(
                canvas,
                (int(round(start.x)), int(round(start.y))),
                (int(round(end.x)), int(round(end.y))),
                color,
                edge_thickness,
                lineType=cv2.LINE_AA,
            )
        for point in nodes.values():
            cv2.circle(
                canvas,
                (int(round(point.x)), int(round(point.y))),
                node_radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
    return canvas
