"""Layer drawers for the shared overlay compositor."""

from overlay.contracts import LayerId
from overlay.layers.base import OverlayLayer
from overlay.layers.behavior import ExclusiveHudLayer, ParallelLedLayer
from overlay.layers.debug import FrameCounterLayer, draw_unavailable_badge
from overlay.layers.pose import PoseLayer
from overlay.layers.segmentation import SegmentationLayer


def default_layer_registry() -> dict[LayerId, OverlayLayer]:
    """Drawable layer registry (S5 segmentation replay shipped in C3-3)."""

    return {
        LayerId.S2_POSE_INFERRED: PoseLayer(LayerId.S2_POSE_INFERRED),
        LayerId.S3_POSE_TRACKED: PoseLayer(LayerId.S3_POSE_TRACKED),
        LayerId.S4_POSE_FINAL: PoseLayer(LayerId.S4_POSE_FINAL),
        LayerId.S5_SEGMENTATION: SegmentationLayer(),
        LayerId.S8_EXCLUSIVE_A_TO_B: ExclusiveHudLayer(
            LayerId.S8_EXCLUSIVE_A_TO_B, stack_index=0
        ),
        LayerId.S8_EXCLUSIVE_B_TO_A: ExclusiveHudLayer(
            LayerId.S8_EXCLUSIVE_B_TO_A, stack_index=1
        ),
        LayerId.S8_CONTACT: ParallelLedLayer(LayerId.S8_CONTACT, led_index=0),
        LayerId.S8_SBS: ParallelLedLayer(LayerId.S8_SBS, led_index=1),
        LayerId.DEBUG_FRAME_COUNTER: FrameCounterLayer(),
    }


__all__ = [
    "ExclusiveHudLayer",
    "FrameCounterLayer",
    "OverlayLayer",
    "ParallelLedLayer",
    "PoseLayer",
    "SegmentationLayer",
    "default_layer_registry",
    "draw_unavailable_badge",
]
