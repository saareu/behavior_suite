"""Contracts for the legacy-compatible framewise CSV profile.

Profile ``legacy_framewise_v1`` matches the laboratory reference column names,
order, and formatting of historical framewise exports. Scientific values come
from pinned S8/S10 states, S9 ``sync_index``, and S4 ``final_pose`` — this
module never mutates upstream artifacts.
"""

from __future__ import annotations

# Shared profile ids live in media_csv_export.contracts (leaf module) to avoid
# an import cycle with media_csv_export.config.
from results_packaging.media_csv_export.contracts import (  # noqa: F401
    CSV_FORMAT_PROFILES,
    DEFAULT_CSV_FORMAT_PROFILE,
    LEGACY_FRAMEWISE_PROFILE_ID,
    PACKAGE_BOUTS_AUTOMATED_PROFILE_ID,
)

LEGACY_FRAMEWISE_PROFILE_VERSION = "1"
LEGACY_FRAMEWISE_CSV_FILENAME = "legacy_framewise_v1.csv"

BEHAVIOR_SOURCE_AUTOMATED = "automated_s8"
BEHAVIOR_SOURCE_REVIEWED = "published_s10"

LEGACY_FRAMEWISE_COLUMNS: tuple[str, ...] = (
    "frame",
    "dense_frame_idx",
    "behaviour_label_focal_0",
    "behaviour_label_focal_1",
    "pred_side_by_side",
    "headstage_0_x",
    "headstage_0_y",
    "neck_0_x",
    "neck_0_y",
    "neck_1_x",
    "neck_1_y",
    "nose_0_x",
    "nose_0_y",
    "nose_1_x",
    "nose_1_y",
    "spine_base_0_x",
    "spine_base_0_y",
    "spine_base_1_x",
    "spine_base_1_y",
    "tail_base_0_x",
    "tail_base_0_y",
    "tail_base_1_x",
    "tail_base_1_y",
)

# Pose nodes/tracks emitted by the reference schema (no headstage_1 columns).
LEGACY_POSE_NODE_TRACKS: tuple[tuple[str, int], ...] = (
    ("headstage", 0),
    ("neck", 0),
    ("neck", 1),
    ("nose", 0),
    ("nose", 1),
    ("spine_base", 0),
    ("spine_base", 1),
    ("tail_base", 0),
    ("tail_base", 1),
)

FRAMEWISE_EXPORT_RUN_PREFIX = "csv_export"
FRAMEWISE_JOB_KIND = "csv_export"

EXPORT_MANIFEST_SCHEMA_VERSION = "legacy_framewise_export_manifest_v1"
COORDINATE_SPACE = "prepared_video_pixels"
COORDINATE_FORMAT_POLICY = (
    "round_half_even_to_integer_then_one_decimal; "
    "empty_field_for_missing; "
    "reference_.0_is_formatting_convention_not_source_dtype; "
    "rounding_exclusive_to_legacy_framewise_v1_csv_serialization; "
    "does_not_mutate_s4_final_pose"
)
ORIGINAL_FRAME_INDEXING = "0_based_raw_decode_frame_idx"
DENSE_FRAME_INDEXING = "0_based_contiguous_dense_sample_idx"

# Indexing notes recorded in manifests (not CSV columns).
INDEXING_NOTES = (
    "frame = sync_index.raw_decode_frame_idx (S1 prepared_sync decode-order "
    "original-video index, 0-based). dense_frame_idx = sync_index.dense_frame_idx "
    "(0-based contiguous analysis grid). Legitimate many-to-one dense→original "
    "mappings are preserved. prepared_frame_idx is never substituted for frame."
)


class LegacyFramewiseError(ValueError):
    """Base error for legacy framewise CSV export."""


class LegacyFramewiseEligibilityError(LegacyFramewiseError):
    """Export blocked because a required mapping/source is unavailable."""


class LegacyFramewiseSourceError(LegacyFramewiseError):
    """Pinned source resolution or integrity failure."""


class LegacyFramewiseFormatError(LegacyFramewiseError):
    """CSV formatting / schema contract failure."""
