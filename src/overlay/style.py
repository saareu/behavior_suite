"""Drawing configuration for overlay composition (C1/C2-A styling)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from overlay.contracts import PIXEL_FORMAT_BGR, HudDirectionMode


@dataclass(frozen=True, slots=True)
class DrawConfig:
    """Explicit render settings pinned into the overlay manifest."""

    pixel_format: str = PIXEL_FORMAT_BGR
    darken_alpha: float = 1.0  # 1.0 = no darken; legacy used ~0.9
    node_radius: int = 4
    edge_thickness: int = 2
    draw_skeleton: bool = False
    unavailable_badge: bool = True
    legend_text: bool = True
    exclusive_hud_enabled: bool = True
    contact_led_enabled: bool = True
    sbs_led_enabled: bool = True
    hud_direction_mode: str = HudDirectionMode.BOTH.value
    show_prepared_time: bool = True
    prepared_video_fps: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_format": self.pixel_format,
            "darken_alpha": self.darken_alpha,
            "node_radius": self.node_radius,
            "edge_thickness": self.edge_thickness,
            "draw_skeleton": self.draw_skeleton,
            "unavailable_badge": self.unavailable_badge,
            "legend_text": self.legend_text,
            "exclusive_hud_enabled": self.exclusive_hud_enabled,
            "contact_led_enabled": self.contact_led_enabled,
            "sbs_led_enabled": self.sbs_led_enabled,
            "hud_direction_mode": self.hud_direction_mode,
            "show_prepared_time": self.show_prepared_time,
            "prepared_video_fps": self.prepared_video_fps,
            "extra": dict(self.extra),
        }
