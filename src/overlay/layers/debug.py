"""Debug / unavailable markers for overlay composition."""

from __future__ import annotations

import cv2
import numpy as np

from overlay.context import OverlayContext, OverlayObservation
from overlay.contracts import LayerId


class FrameCounterLayer:
    """Prepared-frame / dense-sample / time counter (QC chrome)."""

    @property
    def layer_id(self) -> LayerId:
        return LayerId.DEBUG_FRAME_COUNTER

    def render(
        self,
        frame_bgr: np.ndarray,
        context: OverlayContext,
        observation: OverlayObservation,
    ) -> np.ndarray:
        _ = observation
        dense = context.dense_sample_idx
        parts = [f"prep={context.prepared_frame_idx}", f"dense={dense}"]
        fps = context.style.prepared_video_fps
        if context.style.show_prepared_time and fps is not None and float(fps) > 0:
            t_sec = float(context.prepared_frame_idx) / float(fps)
            parts.append(f"t={t_sec:.3f}s")
        if context.timing.mapping is not None:
            parts.append(f"map_t={float(context.timing.mapping.time_sec):.3f}s")
        text = " ".join(parts)
        cv2.putText(
            frame_bgr,
            text,
            (8, frame_bgr.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )
        return frame_bgr


def draw_unavailable_badge(frame_bgr: np.ndarray, layer_id: LayerId, reason: str | None) -> None:
    """Mark a selected but unavailable layer on-frame (no silent omission)."""

    text = f"{layer_id.value} unavailable"
    if reason:
        short = reason if len(reason) <= 48 else reason[:45] + "..."
        text = f"{text}: {short}"
    h = frame_bgr.shape[0]
    y = max(18, h - 28)
    cv2.putText(
        frame_bgr,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (40, 40, 220),
        1,
        lineType=cv2.LINE_AA,
    )
