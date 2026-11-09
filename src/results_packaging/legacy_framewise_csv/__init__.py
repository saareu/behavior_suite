"""Legacy-compatible framewise CSV export (``legacy_framewise_v1`` / ``v2``).

Shared by S9 (automated S8 states) and S10 (published reviewed states). Formatting
is separated from source selection. Jobs are immutable under ``results_exports/``
and never mutate upstream scientific artifacts.
"""

from results_packaging.legacy_framewise_csv.builder import (
    LegacyFramewiseBuildResult,
    build_legacy_framewise_table,
)
from results_packaging.legacy_framewise_csv.builder_v2 import (
    LegacyFramewiseV2BuildResult,
    build_legacy_framewise_v2_table,
    prepared_to_last_in_plateau_dense,
    prepared_to_nearest_dense,
)
from results_packaging.legacy_framewise_csv.config import (
    LegacyFramewiseExportSettings,
    normalize_csv_format_profile,
)
from results_packaging.legacy_framewise_csv.contracts import (
    BEHAVIOR_SOURCE_AUTOMATED,
    BEHAVIOR_SOURCE_REVIEWED,
    CSV_FORMAT_PROFILES,
    DEFAULT_CSV_FORMAT_PROFILE,
    FRAME_ALIGNMENT_COLUMNS,
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_CSV_FILENAME,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_PROFILE_IDS,
    LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME,
    LEGACY_FRAMEWISE_V2_CSV_FILENAME,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
    PACKAGE_BOUTS_AUTOMATED_PROFILE_ID,
    LegacyFramewiseEligibilityError,
    LegacyFramewiseError,
    LegacyFramewiseFormatError,
    LegacyFramewiseSourceError,
    is_legacy_framewise_profile,
)
from results_packaging.legacy_framewise_csv.runner import (
    LegacyFramewiseExportRequest,
    run_legacy_framewise_csv_export,
)
from results_packaging.legacy_framewise_csv.source import (
    LegacyFramewiseSource,
    PoseLineageInfo,
    PoseRunCandidate,
    S3LineageInfo,
    S3RunCandidate,
    inspect_package_pose_lineage,
    inspect_package_s3_lineage,
    list_accepted_s3_run_candidates,
    list_pose_run_candidates,
    resolve_automated_legacy_framewise_source,
    resolve_reviewed_legacy_framewise_source,
)
from results_packaging.legacy_framewise_csv.writer import (
    format_coordinate,
    validate_legacy_framewise_csv_bytes,
    write_legacy_framewise_csv,
)

__all__ = [
    "BEHAVIOR_SOURCE_AUTOMATED",
    "BEHAVIOR_SOURCE_REVIEWED",
    "CSV_FORMAT_PROFILES",
    "DEFAULT_CSV_FORMAT_PROFILE",
    "FRAME_ALIGNMENT_COLUMNS",
    "LEGACY_FRAMEWISE_COLUMNS",
    "LEGACY_FRAMEWISE_CSV_FILENAME",
    "LEGACY_FRAMEWISE_PROFILE_ID",
    "LEGACY_FRAMEWISE_PROFILE_IDS",
    "LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME",
    "LEGACY_FRAMEWISE_V2_CSV_FILENAME",
    "LEGACY_FRAMEWISE_V2_PROFILE_ID",
    "LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION",
    "PACKAGE_BOUTS_AUTOMATED_PROFILE_ID",
    "LegacyFramewiseBuildResult",
    "LegacyFramewiseEligibilityError",
    "LegacyFramewiseError",
    "LegacyFramewiseExportRequest",
    "LegacyFramewiseExportSettings",
    "LegacyFramewiseFormatError",
    "LegacyFramewiseSource",
    "LegacyFramewiseSourceError",
    "LegacyFramewiseV2BuildResult",
    "PoseLineageInfo",
    "PoseRunCandidate",
    "S3LineageInfo",
    "S3RunCandidate",
    "build_legacy_framewise_table",
    "build_legacy_framewise_v2_table",
    "format_coordinate",
    "inspect_package_pose_lineage",
    "inspect_package_s3_lineage",
    "is_legacy_framewise_profile",
    "list_accepted_s3_run_candidates",
    "list_pose_run_candidates",
    "normalize_csv_format_profile",
    "prepared_to_last_in_plateau_dense",
    "prepared_to_nearest_dense",
    "resolve_automated_legacy_framewise_source",
    "resolve_reviewed_legacy_framewise_source",
    "run_legacy_framewise_csv_export",
    "validate_legacy_framewise_csv_bytes",
    "write_legacy_framewise_csv",
]
