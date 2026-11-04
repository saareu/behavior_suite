"""Pose layer drawers including C2-B S4 provenance and C3-1 S2 confidence styling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from overlay.context import OverlayContext, OverlayObservation
from overlay.contracts import (
    S2_SCORE_VISUAL_MAPPING_ID,
    S2_SCORE_VISUAL_MAPPING_SUMMARY,
    S2_SCORE_VISUAL_RANGE_MAX,
    S2_SCORE_VISUAL_RANGE_MIN,
    LayerId,
    S4ProvenanceVisualClass,
    s2_score_visual_intensity,
    s4_provenance_visual_class,
)
from tracking_correction.overlay_draw import draw_pose_overlay
from tracking_correction.review import TRACK_COLORS_BGR, PoseNodePoint

# High-contrast outlines so markers remain readable on bright and dark video.
_OUTLINE_BGR = (20, 20, 20)
_HALO_BGR = (245, 245, 245)


class PoseLayer:
    """Draw prepared- or dense-domain pose points.

    S2 uses provisional identity labels (+ optional confidence intensity).
    S3 uses the shared identity-colored drawer. S4 uses per-keypoint
    provenance visual classes while preserving mouse identity colors.
    """

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
        if self._layer_id is LayerId.S4_POSE_FINAL:
            return _render_s4_pose(
                frame_bgr,
                points_raw,
                context=context,
                legend=str(observation.payload.get("legend") or "final pose"),
            )
        if self._layer_id is LayerId.S2_POSE_INFERRED:
            return _render_s2_pose(
                frame_bgr,
                points_raw,
                context=context,
                legend=str(
                    observation.payload.get("legend")
                    or "inferred pose (S2 predictions; provisional IDs)"
                ),
            )

        node_points = _to_node_points(points_raw, layer_id=self._layer_id)
        if not node_points:
            return frame_bgr
        edges: Sequence[tuple[str, str]] = ()
        if context.style.draw_skeleton:
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
            _draw_simple_legend(canvas, legend, y=20)
        return canvas


def _render_s2_pose(
    frame_bgr: np.ndarray,
    points_raw: Sequence[Mapping[str, Any]],
    *,
    context: OverlayContext,
    legend: str,
) -> np.ndarray:
    canvas = np.ascontiguousarray(frame_bgr.copy())
    show_confidence = bool(context.style.extra.get("show_confidence", False))
    drawable = _s2_drawable_points(points_raw)
    if not drawable:
        if context.style.legend_text:
            _draw_s2_legend(
                canvas,
                legend=legend,
                tracks=(),
                show_confidence=show_confidence,
                skeleton_nodes_only=not bool(
                    context.style.extra.get("skeleton_edges")
                ),
            )
        return canvas

    by_track: dict[int, dict[str, dict[str, Any]]] = {}
    for point in drawable:
        by_track.setdefault(int(point["track"]), {})[str(point["node"])] = point

    edges: Sequence[tuple[str, str]] = ()
    if context.style.draw_skeleton:
        edges = tuple(context.style.extra.get("skeleton_edges", ()))

    radius = max(3, int(context.style.node_radius))
    edge_thickness = max(1, int(context.style.edge_thickness))

    for track, nodes in by_track.items():
        color = TRACK_COLORS_BGR.get(int(track), (200, 200, 200))
        for start_name, end_name in edges:
            start = nodes.get(start_name)
            end = nodes.get(end_name)
            if start is None or end is None:
                continue
            _draw_edge(
                canvas,
                start,
                end,
                color=color,
                thickness=edge_thickness,
            )

    for track, nodes in by_track.items():
        color = TRACK_COLORS_BGR.get(int(track), (200, 200, 200))
        for point in nodes.values():
            _draw_s2_keypoint(
                canvas,
                float(point["x"]),
                float(point["y"]),
                color=color,
                score=point.get("score"),
                show_confidence=show_confidence,
                radius=radius,
            )
        _draw_provisional_track_label(
            canvas, track=int(track), nodes=nodes, color=color
        )

    if context.style.legend_text:
        _draw_s2_legend(
            canvas,
            legend=legend,
            tracks=tuple(sorted(by_track)),
            show_confidence=show_confidence,
            skeleton_nodes_only=len(edges) == 0,
        )
    return canvas


def _s2_drawable_points(
    points_raw: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep every finite XY node; never filter by confidence score."""

    out: list[dict[str, Any]] = []
    for item in points_raw:
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        score = item.get("score")
        score_f: float | None = None
        if score is not None:
            try:
                candidate = float(score)
            except (TypeError, ValueError):
                candidate = float("nan")
            if np.isfinite(candidate):
                score_f = candidate
        out.append(
            {
                "track": int(item["track"]),
                "node": str(item["node"]),
                "x": float(x),
                "y": float(y),
                "score": score_f,
            }
        )
    return out


def _draw_s2_keypoint(
    canvas: np.ndarray,
    x: float,
    y: float,
    *,
    color: tuple[int, int, int],
    score: float | None,
    show_confidence: bool,
    radius: int,
) -> None:
    center = (int(round(x)), int(round(y)))
    # Always keep a strong identity outline so low/missing confidence stays legible.
    cv2.circle(canvas, center, radius + 1, _OUTLINE_BGR, 2, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius + 1, color, 1, lineType=cv2.LINE_AA)

    if not show_confidence:
        cv2.circle(canvas, center, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius, _HALO_BGR, 1, lineType=cv2.LINE_AA)
        return

    intensity = s2_score_visual_intensity(score)
    if intensity is None:
        # Missing confidence: hollow ring only (distinct from numerical zero fill).
        cv2.circle(canvas, center, radius, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius, _HALO_BGR, 1, lineType=cv2.LINE_AA)
        return

    fill = _blend_bgr(color, (25, 25, 25), float(intensity))
    cv2.circle(canvas, center, radius, fill, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius, _HALO_BGR, 1, lineType=cv2.LINE_AA)


def _blend_bgr(
    color: tuple[int, int, int],
    dark: tuple[int, int, int],
    intensity: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(intensity)))
    return (
        int(round(dark[0] + (color[0] - dark[0]) * t)),
        int(round(dark[1] + (color[1] - dark[1]) * t)),
        int(round(dark[2] + (color[2] - dark[2]) * t)),
    )


def _draw_provisional_track_label(
    canvas: np.ndarray,
    *,
    track: int,
    nodes: Mapping[str, Mapping[str, Any]],
    color: tuple[int, int, int],
) -> None:
    if not nodes:
        return
    preferred = nodes.get("nose") or nodes.get("neck") or next(iter(nodes.values()))
    x = int(round(float(preferred["x"])))
    y = int(round(float(preferred["y"]))) - 10
    # P{{id}} = provisional S2 prediction identity (not S3-accepted M{{id}}).
    label = f"P{track}"
    cv2.putText(
        canvas,
        label,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _OUTLINE_BGR,
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        lineType=cv2.LINE_AA,
    )


def _draw_s2_legend(
    canvas: np.ndarray,
    *,
    legend: str,
    tracks: Sequence[int],
    show_confidence: bool,
    skeleton_nodes_only: bool,
) -> None:
    x0, y0 = 8, 16
    cv2.putText(
        canvas,
        f"{legend}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        _HALO_BGR,
        1,
        lineType=cv2.LINE_AA,
    )
    row = y0 + 16
    cv2.putText(
        canvas,
        "S2 QC - provisional IDs (not S3-accepted)",
        (x0, row),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 120),
        1,
        lineType=cv2.LINE_AA,
    )
    row += 16
    if skeleton_nodes_only:
        cv2.putText(
            canvas,
            "nodes-only (no SLEAP topology)",
            (x0, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            lineType=cv2.LINE_AA,
        )
        row += 16
    if show_confidence:
        sample = (80, 180, 255)
        for label, score in (("score~0", 0.0), ("score~1", 1.0), ("missing", None)):
            _draw_s2_keypoint(
                canvas,
                float(x0 + 8),
                float(row - 3),
                color=sample,
                score=score,
                show_confidence=True,
                radius=4,
            )
            cv2.putText(
                canvas,
                label,
                (x0 + 20, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                _HALO_BGR,
                1,
                lineType=cv2.LINE_AA,
            )
            row += 15
        cv2.putText(
            canvas,
            (
                f"{S2_SCORE_VISUAL_MAPPING_ID}: fill intensity vs "
                f"[{S2_SCORE_VISUAL_RANGE_MIN:g},{S2_SCORE_VISUAL_RANGE_MAX:g}] "
                "(draw clamp only)"
            ),
            (x0, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (170, 170, 170),
            1,
            lineType=cv2.LINE_AA,
        )
        row += 14
        # Keep summary short on-frame; full text is in the manifest.
        _ = S2_SCORE_VISUAL_MAPPING_SUMMARY
    for track in tracks:
        color = TRACK_COLORS_BGR.get(int(track), (200, 200, 200))
        cv2.circle(canvas, (x0 + 8, row - 3), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"P{track} provisional",
            (x0 + 20, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            _HALO_BGR,
            1,
            lineType=cv2.LINE_AA,
        )
        row += 15


def _render_s4_pose(
    frame_bgr: np.ndarray,
    points_raw: Sequence[Mapping[str, Any]],
    *,
    context: OverlayContext,
    legend: str,
) -> np.ndarray:
    canvas = np.ascontiguousarray(frame_bgr.copy())
    drawable = _s4_drawable_points(points_raw)
    if not drawable:
        if context.style.legend_text:
            _draw_s4_legend(canvas, legend=legend, tracks=())
        return canvas

    by_track: dict[int, dict[str, dict[str, Any]]] = {}
    for point in drawable:
        by_track.setdefault(int(point["track"]), {})[str(point["node"])] = point

    edges: Sequence[tuple[str, str]] = ()
    if context.style.draw_skeleton:
        edges = tuple(context.style.extra.get("skeleton_edges", ()))

    radius = max(3, int(context.style.node_radius))
    edge_thickness = max(1, int(context.style.edge_thickness))

    # Edges first (only when both endpoints are drawable — never across blanks).
    for track, nodes in by_track.items():
        color = TRACK_COLORS_BGR.get(int(track), (200, 200, 200))
        for start_name, end_name in edges:
            start = nodes.get(start_name)
            end = nodes.get(end_name)
            if start is None or end is None:
                continue
            _draw_edge(
                canvas,
                start,
                end,
                color=color,
                thickness=edge_thickness,
            )

    for track, nodes in by_track.items():
        color = TRACK_COLORS_BGR.get(int(track), (200, 200, 200))
        for point in nodes.values():
            _draw_keypoint(
                canvas,
                float(point["x"]),
                float(point["y"]),
                color=color,
                visual_class=point["visual_class"],
                radius=radius,
            )
        _draw_track_label(canvas, track=int(track), nodes=nodes, color=color)

    if context.style.legend_text:
        _draw_s4_legend(canvas, legend=legend, tracks=tuple(sorted(by_track)))
    return canvas


def _s4_drawable_points(
    points_raw: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in points_raw:
        if not bool(item.get("drawable", False)):
            continue
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        prov = item.get("provenance")
        if prov is None:
            # Refuse to invent provenance; skip rather than guess a class.
            continue
        visual = item.get("visual_class")
        if visual is None:
            visual = s4_provenance_visual_class(int(prov)).value
        visual_class = (
            visual
            if isinstance(visual, S4ProvenanceVisualClass)
            else S4ProvenanceVisualClass(str(visual))
        )
        if visual_class is S4ProvenanceVisualClass.UNAVAILABLE:
            continue
        out.append(
            {
                "track": int(item["track"]),
                "node": str(item["node"]),
                "x": float(x),
                "y": float(y),
                "provenance": int(prov),
                "provenance_name": str(item.get("provenance_name") or ""),
                "visual_class": visual_class,
            }
        )
    return out


def _draw_edge(
    canvas: np.ndarray,
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    p0 = (int(round(float(start["x"]))), int(round(float(start["y"]))))
    p1 = (int(round(float(end["x"]))), int(round(float(end["y"]))))
    # Halo for contrast, then identity-colored stroke.
    cv2.line(canvas, p0, p1, _HALO_BGR, thickness + 2, lineType=cv2.LINE_AA)
    cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)


def _draw_keypoint(
    canvas: np.ndarray,
    x: float,
    y: float,
    *,
    color: tuple[int, int, int],
    visual_class: S4ProvenanceVisualClass,
    radius: int,
) -> None:
    center = (int(round(x)), int(round(y)))
    if visual_class is S4ProvenanceVisualClass.OBSERVED:
        # Filled disc = originally observed / identity-corrected measurement.
        cv2.circle(canvas, center, radius + 1, _OUTLINE_BGR, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius, _HALO_BGR, 1, lineType=cv2.LINE_AA)
        return
    if visual_class is S4ProvenanceVisualClass.TEMPORAL_RESAMPLE:
        # Hollow ring = temporal resampling (distinct from gap-fill).
        cv2.circle(canvas, center, radius + 2, _OUTLINE_BGR, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius + 1, _HALO_BGR, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius, color, 2, lineType=cv2.LINE_AA)
        return
    if visual_class is S4ProvenanceVisualClass.GAP_FILL:
        # Diamond = linear/kalman gap fill reconstruction.
        half = radius + 1
        diamond = np.array(
            [
                [center[0], center[1] - half],
                [center[0] + half, center[1]],
                [center[0], center[1] + half],
                [center[0] - half, center[1]],
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [diamond], True, _OUTLINE_BGR, 2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [diamond], True, color, 2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [diamond], True, _HALO_BGR, 1, lineType=cv2.LINE_AA)
        return
    # UNAVAILABLE must not be drawn as a valid keypoint.


def _draw_track_label(
    canvas: np.ndarray,
    *,
    track: int,
    nodes: Mapping[str, Mapping[str, Any]],
    color: tuple[int, int, int],
) -> None:
    xs = [float(p["x"]) for p in nodes.values()]
    ys = [float(p["y"]) for p in nodes.values()]
    if not xs:
        return
    # Prefer nose/neck when present for a stable identity badge.
    preferred = nodes.get("nose") or nodes.get("neck") or next(iter(nodes.values()))
    x = int(round(float(preferred["x"])))
    y = int(round(float(preferred["y"]))) - 10
    label = f"M{track}"
    cv2.putText(
        canvas,
        label,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _OUTLINE_BGR,
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        lineType=cv2.LINE_AA,
    )
    _ = xs, ys


def _draw_s4_legend(
    canvas: np.ndarray,
    *,
    legend: str,
    tracks: Sequence[int],
) -> None:
    x0, y0 = 8, 16
    cv2.putText(
        canvas,
        f"{legend} (S4 QC)",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _HALO_BGR,
        1,
        lineType=cv2.LINE_AA,
    )
    row = y0 + 18
    samples = (
        (S4ProvenanceVisualClass.OBSERVED, "observed"),
        (S4ProvenanceVisualClass.TEMPORAL_RESAMPLE, "temporal_resample"),
        (S4ProvenanceVisualClass.GAP_FILL, "gap_fill"),
    )
    sample_color = (80, 180, 255)
    for visual, label in samples:
        _draw_keypoint(
            canvas,
            float(x0 + 8),
            float(row - 3),
            color=sample_color,
            visual_class=visual,
            radius=4,
        )
        cv2.putText(
            canvas,
            label,
            (x0 + 20, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            _HALO_BGR,
            1,
            lineType=cv2.LINE_AA,
        )
        row += 16
    for track in tracks:
        color = TRACK_COLORS_BGR.get(int(track), (200, 200, 200))
        cv2.circle(canvas, (x0 + 8, row - 3), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"M{track} identity",
            (x0 + 20, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            _HALO_BGR,
            1,
            lineType=cv2.LINE_AA,
        )
        row += 16
    cv2.putText(
        canvas,
        "missing/blank: not drawn",
        (x0, row),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        lineType=cv2.LINE_AA,
    )


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
        out.append(
            PoseNodePoint(
                track=int(item["track"]),
                node=str(item["node"]),
                x=float(x),
                y=float(y),
            )
        )
    return out


def _draw_simple_legend(frame_bgr: np.ndarray, text: str, *, y: int) -> None:
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
