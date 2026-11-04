"""Read-only scientific source adapters for the overlay compositor."""

from overlay.adapters.base import (
    OverlayStageAdapter,
    SourceIdentity,
    UnavailableAdapter,
    behavior_source_kind_label,
)
from overlay.adapters.behavior import (
    DirectionalExclusiveAdapter,
    ParallelChannelAdapter,
    make_s5_segmentation_stub,
    make_s8_contact,
    make_s8_exclusive_a_to_b,
    make_s8_exclusive_b_to_a,
    make_s8_sbs,
    make_s10_reviewed_contact,
    make_s10_reviewed_exclusive_a_to_b,
    make_s10_reviewed_exclusive_b_to_a,
    make_s10_reviewed_sbs,
)
from overlay.adapters.pose import (
    PreparedPoseAdapter,
    S4FinalPoseAdapter,
    make_s2_inferred_pose_adapter,
    make_s3_tracked_pose_adapter,
    make_s4_final_pose_adapter,
)
from overlay.adapters.segmentation import (
    PreparedFrameReplayCache,
    S5SegmentationReplayAdapter,
    make_s5_segmentation_adapter,
)

__all__ = [
    "DirectionalExclusiveAdapter",
    "OverlayStageAdapter",
    "ParallelChannelAdapter",
    "PreparedFrameReplayCache",
    "PreparedPoseAdapter",
    "S4FinalPoseAdapter",
    "S5SegmentationReplayAdapter",
    "SourceIdentity",
    "UnavailableAdapter",
    "behavior_source_kind_label",
    "make_s2_inferred_pose_adapter",
    "make_s3_tracked_pose_adapter",
    "make_s4_final_pose_adapter",
    "make_s5_segmentation_adapter",
    "make_s5_segmentation_stub",
    "make_s8_contact",
    "make_s8_exclusive_a_to_b",
    "make_s8_exclusive_b_to_a",
    "make_s8_sbs",
    "make_s10_reviewed_contact",
    "make_s10_reviewed_exclusive_a_to_b",
    "make_s10_reviewed_exclusive_b_to_a",
    "make_s10_reviewed_sbs",
]
