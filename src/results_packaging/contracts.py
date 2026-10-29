"""Subsystem 09 (results packaging) contracts for Sprint 1–3.

Defines schema stamps, sync-index / bout / summary columns, run layout
names, approved export-profile defaults, publication statuses, discovery
classifications, fingerprint semantics, and shared errors. S8 remains
the scientific authority for states and bouts.
"""

from __future__ import annotations

from behavioral_arbitration.bouts import BOUT_COLUMNS as S8_BOUT_COLUMNS
from behavioral_arbitration.contracts import (
    BOUTS_SCHEMA_VERSION as S8_BOUTS_SCHEMA_VERSION,
)
from behavioral_arbitration.contracts import (
    STATES_SCHEMA_VERSION as S8_STATES_SCHEMA_VERSION,
)
from behavioral_arbitration.contracts import (
    TARGET_FPS as S8_TARGET_FPS,
)

# Package / settings schema stamps.
SETTINGS_SCHEMA_VERSION = "results_package_settings_v1"
SYNC_SCHEMA_VERSION = "results_package_sync_v1"
BOUTS_SCHEMA_VERSION = "results_package_bouts_v1"
ETHOGRAM_SUMMARY_SCHEMA_VERSION = "results_package_ethogram_summary_v1"
MANIFEST_SCHEMA_VERSION = "results_package_manifest_v1"
RUN_META_SCHEMA_VERSION = "results_package_run_meta_v1"
QC_SCHEMA_VERSION = "results_package_qc_v1"
SPRINT1_RESULT_SCHEMA_VERSION = "results_package_sprint1_v1"
SPRINT2_RESULT_SCHEMA_VERSION = "results_package_sprint2_v1"
SPRINT3_RESULT_SCHEMA_VERSION = "results_package_sprint3_v1"
EXPORT_PROFILE_MANIFEST_SCHEMA_VERSION = "results_package_export_profile_v1"
DEPENDENCY_MANIFEST_SCHEMA_VERSION = "results_package_dependency_manifest_v1"
PACKAGE_VERSION = "results_package_v1"

# Expected upstream schema stamps.
EXPECTED_S8_STATES_SCHEMA_VERSION = S8_STATES_SCHEMA_VERSION
EXPECTED_S8_BOUTS_SCHEMA_VERSION = S8_BOUTS_SCHEMA_VERSION

# Package / run layout (immutable publish: Sprint 3).
RESULTS_PACKAGE_DIRNAME = "results_package"
RESULTS_PACKAGE_RUN_PREFIX = "results_package"
SYNC_INDEX_FILENAME = "sync_index.parquet"
BEHAVIOR_BOUTS_FILENAME = "behavior_bouts.parquet"
ETHOGRAM_SUMMARY_FILENAME = "ethogram_summary.parquet"
PACKAGE_MANIFEST_FILENAME = "package_manifest.json"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
QC_FILENAME = "qc.json"
PROCESSING_LOG_FILENAME = "processing_log.txt"
EXPORT_PROFILES_DIRNAME = "export_profiles"
BOUT_INDEX_FILENAME = "bout_index.parquet"
PROFILE_MANIFEST_FILENAME = "profile_manifest.json"
EXCLUSION_TABLE_FILENAME = "exclusion_table.parquet"

# Lifecycle statuses (historical; never rewritten when sources later go stale).
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"

# Discovery / current-eligibility classifications (distinct from historical status).
ELIGIBLE_DOWNSTREAM = "eligible"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STALE_SOURCE = "stale_source"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"

# Fingerprint / verification semantics (U-R6). Never equate size/mtime with SHA256.
FINGERPRINT_TYPE_SHA256 = "sha256"
FINGERPRINT_TYPE_SIZE_MTIME = "size_mtime"
VERIFICATION_STRENGTH_CONTENT_HASH = "content_hash"
VERIFICATION_STRENGTH_SIZE_MTIME = "size_mtime_fingerprint"
VERIFICATION_STRENGTH_PATH_ONLY = "path_only"

# Timeline.
TARGET_FPS = float(S8_TARGET_FPS)  # 120.0
TIME_ATOL_SEC = 1e-9
FPS_ATOL = 1e-6

# Sync construction methods.
SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC = "rebuild_from_prepared_sync"
SYNC_METHOD_COPY_VALIDATED_S6_MAP = "copy_validated_s6_map"

# Package modes.
PACKAGE_MODE_REFERENCE_LIGHTWEIGHT = "reference_lightweight"
PACKAGE_MODE_PORTABLE_EMBED = "portable_embed"

# Default analysis profile (always present conceptually; no filtering).
DEFAULT_PROFILE_ID = "analysis_full"
REVIEW_EMIT_PROFILE_ID = "review_emit_v1"

# Approved review_emit_v1 defaults (U-R1/U-R2). Applied only when the profile
# is explicitly selected (Sprint 2+). Never filters the default package.
REVIEW_EMIT_V1_INCLUDE_CONTACT = False
REVIEW_EMIT_V1_INCLUDE_ENGAGED = True
REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE = True
REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES = 10
REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS = 0.10
REVIEW_EMIT_V1_TARGET_FPS = TARGET_FPS

# Bout packaging sort (stable; independent of incidental source row order).
PACKAGE_BOUTS_SORT_COLUMNS: tuple[str, ...] = (
    "channel",
    "start_sample_idx",
    "behavior_name",
    "pair_id",
    "focal_id",
    "partner_id",
)

# Authoritative S8 bout content key fields (lossless multiset equality).
AUTHORITATIVE_BOUT_KEY_COLUMNS: tuple[str, ...] = (
    "source_s8_bout_id",
    "channel",
    "behavior_name",
    "start_sample_idx",
    "end_sample_idx",
    "identity_grain",
    "focal_id",
    "partner_id",
    "mouse_a_id",
    "mouse_b_id",
)

# Prepared-frame navigation annotations (seek aids; not scientific endpoints).
BOUT_NAVIGATION_COLUMNS: tuple[str, ...] = (
    "prepared_frame_idx_start",
    "prepared_frame_idx_end",
    "prepared_time_sec_start",
    "prepared_time_sec_end",
)

# Package stamp columns added to the lossless S8 bout copy.
PACKAGE_BOUT_STAMP_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "source_s8_bout_id",
    "source_s8_run_id",
    "package_run_id",
    "schema_version",
    "s8_bouts_schema_version",
)

# Ethogram summary kinds (S9-B §1.4).
SUMMARY_KIND_LABEL_TIME = "label_time"
SUMMARY_KIND_BOUT_COUNTS = "bout_counts"
SUMMARY_KIND_PARALLEL_OVERLAP = "parallel_overlap"
SUMMARY_KIND_SESSION_TIMING = "session_timing"
SUMMARY_KINDS: tuple[str, ...] = (
    SUMMARY_KIND_LABEL_TIME,
    SUMMARY_KIND_BOUT_COUNTS,
    SUMMARY_KIND_PARALLEL_OVERLAP,
    SUMMARY_KIND_SESSION_TIMING,
)

ETHOGRAM_SUMMARY_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "summary_kind",
    "channel",
    "behavior_name",
    "identity_key",
    "identity_grain",
    "metric_name",
    "metric_value",
    "denominator_name",
    "denominator_value",
    "unit",
    "notes",
)

# review_emit_v1 exclusion reason priority (first match is primary).
REVIEW_EMIT_EXCLUSION_REASON_PRIORITY: tuple[str, ...] = (
    "channel_contact_excluded",
    "channel_side_by_side_excluded",
    "behavior_engaged_excluded",
    "behavior_not_allowed",
    "channel_not_allowed",
    "below_emit_floor",
)

BOUT_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "bout_index_id",
    "bout_id",
    "source_s8_bout_id",
    "filter_profile_id",
    "behavior_name",
    "channel",
    "identity_grain",
    "focal_id",
    "partner_id",
    "mouse_a_id",
    "mouse_b_id",
    "pair_id",
    "start_sample_idx",
    "end_sample_idx",
    "start_time_sec",
    "end_time_sec",
    "bout_duration_frames",
    "bout_duration_sec",
    "prepared_frame_idx_start",
    "prepared_frame_idx_end",
    "prepared_time_sec_start",
    "prepared_time_sec_end",
    "min_required_frames",
)

# Required sync_index columns (Sprint 1 in-memory / future parquet).
SYNC_INDEX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "sample_idx",
    "dense_frame_idx",
    "time_sec",
    "prepared_frame_idx",
    "prepared_time_sec",
    "is_exact_source_sample",
    "timing_residual_sec",
)

# Optional sync columns when S1 metadata supports them.
SYNC_INDEX_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "raw_decode_frame_idx",
    "raw_pts_time_sec",
    "package_run_id",
)

# Required S8 state identity / timing fields for packaging intake.
REQUIRED_STATE_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "dense_frame_idx",
    "sample_idx",
    "time_sec",
    "focal_id",
    "partner_id",
)

# Required S8 bout fields for packaging intake / boundary mapping.
REQUIRED_BOUT_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "schema_version",
    "behavior_name",
    "channel",
    "identity_grain",
    "start_sample_idx",
    "end_sample_idx",
    "bout_duration_frames",
    "bout_duration_sec",
)

# Full S8 bout science columns preserved in the packaged table.
S8_BOUT_SCIENCE_COLUMNS: tuple[str, ...] = tuple(S8_BOUT_COLUMNS)


class ResultsPackagingError(Exception):
    """Base error for Subsystem 09."""


class ConfigError(ResultsPackagingError):
    """Invalid S9 configuration."""


class IntakeError(ResultsPackagingError):
    """S8 / sync / video intake or eligibility validation failed."""


class SyncIndexError(ResultsPackagingError):
    """Synchronization index construction or validation failed."""


class BoutMappingError(ResultsPackagingError):
    """Bout-boundary prepared-frame mapping failed."""


class BoutPackagingError(ResultsPackagingError):
    """Lossless bout packaging / identity construction failed."""


class EthogramSummaryError(ResultsPackagingError):
    """Ethogram summary construction failed."""


class ExportProfileError(ResultsPackagingError):
    """Optional export-profile transformation failed."""


class DependencyError(ResultsPackagingError):
    """Dependency / provenance identity resolution failed."""


class PublicationError(ResultsPackagingError):
    """Immutable S9 publication / reopen / eligibility failures."""


class ProvenanceError(ResultsPackagingError):
    """Source identity / digest drift detected before or after publication."""
