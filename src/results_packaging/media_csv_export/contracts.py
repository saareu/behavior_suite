"""Frozen contracts for S9 media/CSV export jobs."""

from __future__ import annotations

from results_packaging.contracts import (
    PACKAGE_BOUTS_SORT_COLUMNS,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    TARGET_FPS,
)

# Re-export lifecycle constants used by export jobs.
STATUS_INCOMPLETE = "incomplete"

RESULTS_EXPORTS_DIRNAME = "results_exports"
MEDIA_EXPORT_RUN_PREFIX = "media_export"
CSV_EXPORT_RUN_PREFIX = "csv_export"

JOB_KIND_MEDIA = "media_export"
JOB_KIND_CSV = "csv_export"

EXPORT_SETTINGS_SCHEMA_VERSION = "results_export_settings_v1"
EXPORT_RUN_META_SCHEMA_VERSION = "results_export_run_meta_v1"
EXPORT_MANIFEST_SCHEMA_VERSION = "results_export_manifest_v1"
EXPORT_QC_SCHEMA_VERSION = "results_export_qc_v1"
CLIP_INDEX_SCHEMA_VERSION = "results_export_clip_index_v1"

EXPORT_JOB_MANIFEST_FILENAME = "export_job_manifest.json"
CLIP_INDEX_FILENAME = "clip_index.parquet"
FAILURES_JSONL_FILENAME = "failures.jsonl"
CSV_EXPORT_FILENAME = "behavior_bouts_automated.csv"
CLIPS_DIRNAME = "clips"

# Selection modes (exactly one per job).
SELECTION_ALL_PACKAGE_BOUTS = "all_package_bouts"
SELECTION_CHANNELS_BEHAVIORS = "channels_behaviors"
SELECTION_EXPORT_PROFILE = "export_profile"
SELECTION_BOUT_ID_LIST = "bout_id_list"
SELECTION_MODES = frozenset(
    {
        SELECTION_ALL_PACKAGE_BOUTS,
        SELECTION_CHANNELS_BEHAVIORS,
        SELECTION_EXPORT_PROFILE,
        SELECTION_BOUT_ID_LIST,
    }
)

# Encode statuses.
ENCODE_STATUS_SUCCEEDED = "succeeded"
ENCODE_STATUS_SKIPPED_EXISTS = "skipped_exists"
ENCODE_STATUS_FAILED = "failed"
ENCODE_STATUS_OMITTED_INVALID_BOUNDS = "omitted_invalid_bounds"
ENCODE_STATUS_PENDING = "pending"

EXPORT_PROVENANCE_AUTOMATED = "automated"

SOURCE_VIDEO_KIND_PREPARED = "prepared_video"
DEFAULT_CRF = 22
DEFAULT_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"

# Discovery classifications.
EXPORT_ELIGIBLE = "eligible"
EXPORT_INCOMPLETE = "incomplete"
EXPORT_FAILED_OR_INCOMPLETE = "failed_or_incomplete"
EXPORT_STALE_SOURCE = "stale_source"
EXPORT_MISSING_REQUIRED = "missing_required_artifacts"
EXPORT_COMPLETED_WITH_WARNINGS = "completed_with_warnings"

CSV_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "export_provenance",
    "export_job_id",
    "selection_mode",
    "source_package_run_id",
)

CLIP_INDEX_IDENTITY_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "source_s8_bout_id",
    "package_run_id",
    "source_s8_run_id",
    "channel",
    "behavior_name",
    "identity_grain",
    "focal_id",
    "partner_id",
    "mouse_a_id",
    "mouse_b_id",
    "pair_id",
)

CLIP_INDEX_BEHAVIORAL_COLUMNS: tuple[str, ...] = (
    "start_sample_idx",
    "end_sample_idx",
    "start_time_sec",
    "end_time_sec",
    "bout_duration_frames",
    "bout_duration_sec",
)

CLIP_INDEX_PREPARED_COLUMNS: tuple[str, ...] = (
    "prepared_frame_idx_start",
    "prepared_frame_idx_end",
    "prepared_time_sec_start",
    "prepared_time_sec_end",
)

CLIP_INDEX_CLIP_COLUMNS: tuple[str, ...] = (
    "clip_frame_idx_start",
    "clip_frame_idx_end",
    "clip_frame_count",
    "clip_time_sec_start",
    "clip_time_sec_end",
    "clip_duration_sec",
    "pre_context_sec",
    "post_context_sec",
)

CLIP_INDEX_MEDIA_COLUMNS: tuple[str, ...] = (
    "source_video_kind",
    "source_video_path",
    "source_fps",
    "source_n_frames",
    "clip_relpath",
    "clip_exists",
    "encode_status",
    "error",
)

CLIP_INDEX_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "export_job_id",
    "selection_mode",
    "crf",
    "codec",
    "schema_version",
)

CLIP_INDEX_COLUMNS: tuple[str, ...] = (
    *CLIP_INDEX_IDENTITY_COLUMNS,
    *CLIP_INDEX_BEHAVIORAL_COLUMNS,
    *CLIP_INDEX_PREPARED_COLUMNS,
    *CLIP_INDEX_CLIP_COLUMNS,
    *CLIP_INDEX_MEDIA_COLUMNS,
    *CLIP_INDEX_PROVENANCE_COLUMNS,
)

EXPORT_BOUTS_SORT_COLUMNS: tuple[str, ...] = (
    *PACKAGE_BOUTS_SORT_COLUMNS,
    "bout_id",
)

BEHAVIORAL_FPS = float(TARGET_FPS)


class ExportError(Exception):
    """Base error for media/CSV export jobs."""


class ExportConfigError(ExportError):
    """Invalid export settings."""


class ExportSourceError(ExportError):
    """Source package / video validation failed."""


class ClipBoundsError(ExportError):
    """Clip boundary / padding calculation failed."""


class ExportVideoError(ExportError):
    """Video probe or encode failure."""


class ExportPublicationError(ExportError):
    """Export job publication / retry / reopen failure."""


class ExportSelectionError(ExportError):
    """Bout selection resolution failed."""


# Silence unused re-exports for type checkers that only see assignments.
_ = (
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
)
