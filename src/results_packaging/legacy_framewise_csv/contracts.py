"""Contracts for the legacy-compatible framewise CSV profiles.

Profiles ``legacy_framewise_v1`` (dense-row domain) and ``legacy_framewise_v2``
(prepared-frame domain) share the laboratory 23-column header and formatting.

V1 scientific values come from pinned S8/S10 states, S9 SyncMap, and S4
``final_pose``.

V2 scientific values come from S1 prepared-video frames, accepted S3
``tracked_pose`` (prepared-frame coordinates), and S8/S10 behavioral evidence
mapped prepared→nearest-dense. This module never mutates upstream artifacts.
"""

from __future__ import annotations

# Shared profile ids live in media_csv_export.contracts (leaf module) to avoid
# an import cycle with media_csv_export.config.
from results_packaging.media_csv_export.contracts import (  # noqa: F401
    CSV_FORMAT_PROFILES,
    DEFAULT_CSV_FORMAT_PROFILE,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_PROFILE_IDS,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    PACKAGE_BOUTS_AUTOMATED_PROFILE_ID,
)

LEGACY_FRAMEWISE_PROFILE_VERSION = "1"
LEGACY_FRAMEWISE_V2_PROFILE_VERSION = "2"
# Distinguishes historical S4-based V2 jobs from prepared-frame S3 pose exports.
# Historical jobs remain immutable; new jobs record this revision in metadata.
LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION = "prepared_s3_pose_nearest_behavior_v1"
LEGACY_FRAMEWISE_CSV_FILENAME = "legacy_framewise_v1.csv"
LEGACY_FRAMEWISE_V2_CSV_FILENAME = "legacy_framewise_v2.csv"
LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME = (
    "legacy_framewise_v2_frame_alignment.parquet"
)

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

# Sidecar columns (not part of the laboratory CSV).
# Schema migration from historical V2 (S4 SyncMap) sidecars:
#   removed: pose_dense_frame_idx, pose_coverage
#   added:   s3_pose_available
# Historical export jobs are never rewritten; only new jobs use this schema.
FRAME_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "prepared_frame_idx",
    "raw_decode_frame_idx",
    "behavior_dense_frame_idx",
    "behavior_coverage",
    "s3_pose_available",
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
EXPORT_MANIFEST_SCHEMA_VERSION_V2 = "legacy_framewise_export_manifest_v2"
COORDINATE_SPACE = "prepared_video_pixels"
COORDINATE_FORMAT_POLICY = (
    "round_half_even_to_integer_then_one_decimal; "
    "empty_field_for_missing; "
    "reference_.0_is_formatting_convention_not_source_dtype; "
    "rounding_exclusive_to_legacy_framewise_csv_serialization; "
    "does_not_mutate_s4_final_pose"
)
COORDINATE_FORMAT_POLICY_V2 = (
    "round_half_even_to_integer_then_one_decimal; "
    "empty_field_for_missing; "
    "reference_.0_is_formatting_convention_not_source_dtype; "
    "rounding_exclusive_to_legacy_framewise_csv_serialization; "
    "coordinates_from_accepted_s3_tracked_pose_prepared_video_pixels; "
    "does_not_mutate_accepted_s3_tracked_pose; "
    "does_not_use_s4_final_pose"
)
ORIGINAL_FRAME_INDEXING = "0_based_raw_decode_frame_idx"
DENSE_FRAME_INDEXING = "0_based_contiguous_dense_sample_idx"
PREPARED_FRAME_INDEXING = "0_based_s1_prepared_video_frame_idx"
PLATEAU_POLICY_LAST_IN_PLATEAU = "last_in_plateau"
BEHAVIOR_MAPPING_PREPARED_NEAREST_DENSE = "prepared_to_nearest_dense"
BEHAVIOR_TIE_BREAK_EARLIER = "earlier_dense_sample_on_tie"

# Indexing notes recorded in manifests (not CSV columns).
INDEXING_NOTES = (
    "frame = sync_index.raw_decode_frame_idx (S1 prepared_sync decode-order "
    "original-video index, 0-based). dense_frame_idx = sync_index.dense_frame_idx "
    "(0-based contiguous analysis grid). Legitimate many-to-one dense→original "
    "mappings are preserved. prepared_frame_idx is never substituted for frame."
)
INDEXING_NOTES_V2 = (
    "Row domain = S1 prepared-video frames (prepared_frame_idx 0..n-1). "
    "frame = S1 prepared_sync.raw_decode_frame_idx for that prepared frame "
    "(0-based original-video decode index). dense_frame_idx = behavioral dense "
    "sample selected by prepared→nearest-dense on authoritative timestamps "
    "(earlier sample wins exact ties; nullable; noncontiguous; repeated dense "
    "selections allowed when prepared FPS exceeds dense FPS). Pose coordinates "
    "come from accepted S3 tracked_pose at the same prepared_frame_idx "
    "(prepared-video pixels; not S4 finalized pose). Missing coverage retains "
    "the prepared-frame row with empty fields from that source; never "
    "interpolate, forward-fill, or invent labels/coordinates."
)


def is_legacy_framewise_profile(profile: str | None) -> bool:
    """True for dense v1 or prepared-frame v2 laboratory CSV profiles."""

    return str(profile or "").strip() in LEGACY_FRAMEWISE_PROFILE_IDS


class LegacyFramewiseError(ValueError):
    """Base error for legacy framewise CSV export."""


class LegacyFramewiseEligibilityError(LegacyFramewiseError):
    """Export blocked because a required mapping/source is unavailable."""


class LegacyFramewiseSourceError(LegacyFramewiseError):
    """Pinned source resolution or integrity failure."""


class LegacyFramewiseFormatError(LegacyFramewiseError):
    """CSV formatting / schema contract failure."""
