"""Behavioral HUD drawers for C2-A QC (dual exclusive rows + parallel LEDs)."""

from __future__ import annotations

import cv2
import numpy as np

from overlay.context import OverlayContext, OverlayObservation
from overlay.contracts import LayerId

# BGR palette — display-only; labels remain exact recorded strings.
_BEHAVIOR_COLORS: dict[str, tuple[int, int, int]] = {
    "approach": (0, 165, 255),
    "retract": (0, 0, 200),
    "engaged": (0, 255, 255),
    "mutually_disengaged": (110, 110, 110),
    "nose_to_body": (0, 200, 0),
    "nose_to_nose": (150, 150, 255),
    "nose_to_tail": (100, 200, 255),
    "undefined": (40, 40, 40),
    "unresolved": (40, 40, 40),
}
_DEFAULT_COLOR = (90, 90, 90)
_UNAVAILABLE_COLOR = (40, 40, 180)
_UNKNOWN_COLOR = (30, 30, 30)
_ROW_BG = (28, 28, 28)
_LED_ON = (40, 220, 40)
_LED_OFF = (70, 70, 70)
_LED_UNKNOWN = (40, 40, 200)

# Fixed HUD geometry (prepared-frame pixels; not pose-anchored).
_HUD_LEFT = 8
_HUD_TOP = 8
_ROW_HEIGHT = 26
_ROW_GAP = 4
_PILL_PAD_X = 8
_PILL_PAD_Y = 5


class ExclusiveHudLayer:
    """Fixed directional exclusive rows; A→B and B→A remain independent layers."""

    renders_when_unavailable = True

    def __init__(self, layer_id: LayerId, *, stack_index: int = 0) -> None:
        self._layer_id = layer_id
        self._stack_index = int(stack_index)

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    def render(
        self,
        frame_bgr: np.ndarray,
        context: OverlayContext,
        observation: OverlayObservation,
    ) -> np.ndarray:
        if not context.style.exclusive_hud_enabled:
            return frame_bgr
        canvas = frame_bgr
        y = _HUD_TOP + self._stack_index * (_ROW_HEIGHT + _ROW_GAP)
        direction = _direction_glyph(observation)
        if not observation.available:
            text = f"{direction}  UNAVAILABLE"
            _draw_fixed_row(
                canvas,
                text,
                x=_HUD_LEFT,
                y=y,
                color=_UNAVAILABLE_COLOR,
                muted=True,
            )
            return canvas
        label = observation.payload.get("behavior_label")
        if label is None or str(label).strip() == "":
            text = f"{direction}  UNKNOWN"
            _draw_fixed_row(
                canvas,
                text,
                x=_HUD_LEFT,
                y=y,
                color=_UNKNOWN_COLOR,
                muted=True,
            )
            return canvas
        label_text = str(label)
        color = _BEHAVIOR_COLORS.get(label_text, _DEFAULT_COLOR)
        text = f"{direction}  {label_text}"
        _draw_fixed_row(canvas, text, x=_HUD_LEFT, y=y, color=color, muted=False)
        return canvas


class ParallelLedLayer:
    """Contact / SBS indicators (parallel channels; never exclusive ethogram labels)."""

    renders_when_unavailable = True

    def __init__(self, layer_id: LayerId, *, led_index: int) -> None:
        self._layer_id = layer_id
        self._led_index = int(led_index)

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    def render(
        self,
        frame_bgr: np.ndarray,
        context: OverlayContext,
        observation: OverlayObservation,
    ) -> np.ndarray:
        channel = str(observation.payload.get("channel") or self._layer_id.value)
        if channel == "contact" and not context.style.contact_led_enabled:
            return frame_bgr
        if channel == "side_by_side" and not context.style.sbs_led_enabled:
            return frame_bgr

        canvas = frame_bgr
        h, w = canvas.shape[:2]
        # Place LEDs on a fixed right-side strip, independent of exclusive rows.
        x = w - 18
        y = 18 + self._led_index * 34
        name = "CONTACT" if channel == "contact" else (
            "SBS" if channel in {"side_by_side", "sbs", LayerId.S8_SBS.value} else channel.upper()
        )

        if not observation.available or "active" not in observation.payload:
            color = _LED_UNKNOWN
            state = "?"
        else:
            active = bool(observation.payload.get("active"))
            color = _LED_ON if active else _LED_OFF
            state = "ON" if active else "OFF"

        cv2.circle(canvas, (x, y), 9, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 9, (220, 220, 220), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{name}:{state}",
            (x - 92, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (230, 230, 230),
            1,
            lineType=cv2.LINE_AA,
        )
        # Keep height reference so unused param lint stays quiet if shape changes.
        _ = h
        return canvas


def _direction_glyph(observation: OverlayObservation) -> str:
    focal = observation.payload.get("focal_id")
    partner = observation.payload.get("partner_id")
    if focal is None or partner is None:
        if observation.layer_id is LayerId.S8_EXCLUSIVE_A_TO_B:
            return "0->1"
        if observation.layer_id is LayerId.S8_EXCLUSIVE_B_TO_A:
            return "1->0"
        return "?->?"
    return f"{int(focal)}->{int(partner)}"


def _draw_fixed_row(
    frame_bgr: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    color: tuple[int, int, int],
    muted: bool,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x0 = int(x)
    y0 = int(y)
    x1 = int(x + tw + 2 * _PILL_PAD_X)
    y1 = int(y + th + 2 * _PILL_PAD_Y)
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), _ROW_BG, -1, lineType=cv2.LINE_AA)
    accent = color if not muted else _UNAVAILABLE_COLOR
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + 4, y1), accent, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        text,
        (x0 + _PILL_PAD_X, y0 + th + _PILL_PAD_Y - 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
