"""Subsystem 09 — Results Packaging & Export (Sprint 1–3 headless APIs).

Sprint 1: headless S8 intake, sync-index construction, bout-boundary mapping,
dependency identities.

Sprint 2: lossless bout packaging, ethogram summaries, optional
``review_emit_v1`` export views.

Sprint 3: immutable publication, QC, package manifest, run discovery, and
downstream eligibility.

Sprint 4 GUI lives under ``ui.pages.results_packaging_page`` /
``ui.controllers.results_packaging_controller`` and calls these APIs. This
package does **not** host Qt widgets or implement S10 review.
"""

from results_packaging.bout_mapping import (
    BoutBoundaryMapping,
    map_bout_boundaries_to_prepared_frames,
    map_single_bout_boundary,
)
from results_packaging.bouts import (
    assert_lossless_bout_copy,
    authoritative_bout_content_key,
    build_package_bout_id,
    package_behavior_bouts,
)
from results_packaging.config import (
    ExecutionSettings,
    ResultsPackagingConfig,
    ReviewEmitV1Defaults,
    config_from_mapping,
    default_suite_config_path,
    load_results_packaging_config,
)
from results_packaging.contracts import (
    DEFAULT_PROFILE_ID,
    DEPENDENCY_MANIFEST_SCHEMA_VERSION,
    ELIGIBLE_DOWNSTREAM,
    EXPECTED_S8_BOUTS_SCHEMA_VERSION,
    EXPECTED_S8_STATES_SCHEMA_VERSION,
    FINGERPRINT_TYPE_SHA256,
    FINGERPRINT_TYPE_SIZE_MTIME,
    RESULTS_PACKAGE_DIRNAME,
    REVIEW_EMIT_PROFILE_ID,
    REVIEW_EMIT_V1_INCLUDE_CONTACT,
    REVIEW_EMIT_V1_INCLUDE_ENGAGED,
    REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES,
    REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS,
    REVIEW_EMIT_V1_TARGET_FPS,
    SETTINGS_SCHEMA_VERSION,
    SPRINT1_RESULT_SCHEMA_VERSION,
    SPRINT2_RESULT_SCHEMA_VERSION,
    SPRINT3_RESULT_SCHEMA_VERSION,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    SYNC_INDEX_FILENAME,
    SYNC_SCHEMA_VERSION,
    TARGET_FPS,
    VERIFICATION_STRENGTH_CONTENT_HASH,
    VERIFICATION_STRENGTH_SIZE_MTIME,
    BoutMappingError,
    BoutPackagingError,
    ConfigError,
    DependencyError,
    EthogramSummaryError,
    ExportProfileError,
    IntakeError,
    ProvenanceError,
    PublicationError,
    ResultsPackagingError,
    SyncIndexError,
)
from results_packaging.dependencies import (
    ArtifactRef,
    DependencyManifest,
    build_dependency_manifest,
    dependency_fingerprint_token,
)
from results_packaging.eligibility import (
    ResolvedSyncVideoSources,
    S9SourceValidationResult,
    select_s8_run_for_packaging,
    validate_s8_source_for_packaging,
)
from results_packaging.ethogram_summary import build_ethogram_summary
from results_packaging.export_profiles import (
    ExportProfileResult,
    ReviewEmitV1Params,
    apply_review_emit_v1,
    review_emit_v1_params_from_mapping,
)
from results_packaging.publication import (
    is_s9_run_eligible_for_downstream,
    reopen_results_package,
    run_publication_from_session,
    run_results_packaging_publication,
    validate_s9_run_for_downstream,
)
from results_packaging.qc import QcSummary, assert_structural_qc, evaluate_results_package_qc
from results_packaging.results import (
    ResultsPackageRunResult,
    Sprint1PackagingResult,
    Sprint2PackagingResult,
)
from results_packaging.run_discovery import (
    ResultsPackageProjectSummary,
    ResultsPackageRunSummary,
    find_latest_eligible_s9_run,
    list_completed_s9_runs,
    summarize_results_package_project,
)
from results_packaging.runner import (
    ResultsPackagingRequest,
    run_results_packaging_sprint1,
    run_results_packaging_sprint2,
    run_sprint1_from_session,
    run_sprint2_from_session,
)
from results_packaging.sync_index import (
    SyncIndexBuildResult,
    build_sync_index_from_prepared_sync,
    build_sync_index_from_source_arrays,
    validate_sync_index,
)

__all__ = [
    "DEFAULT_PROFILE_ID",
    "DEPENDENCY_MANIFEST_SCHEMA_VERSION",
    "ELIGIBLE_DOWNSTREAM",
    "EXPECTED_S8_BOUTS_SCHEMA_VERSION",
    "EXPECTED_S8_STATES_SCHEMA_VERSION",
    "FINGERPRINT_TYPE_SHA256",
    "FINGERPRINT_TYPE_SIZE_MTIME",
    "ArtifactRef",
    "BoutBoundaryMapping",
    "BoutMappingError",
    "BoutPackagingError",
    "ConfigError",
    "DependencyError",
    "DependencyManifest",
    "EthogramSummaryError",
    "ExecutionSettings",
    "ExportProfileError",
    "ExportProfileResult",
    "IntakeError",
    "ProvenanceError",
    "PublicationError",
    "QcSummary",
    "RESULTS_PACKAGE_DIRNAME",
    "REVIEW_EMIT_PROFILE_ID",
    "REVIEW_EMIT_V1_INCLUDE_CONTACT",
    "REVIEW_EMIT_V1_INCLUDE_ENGAGED",
    "REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES",
    "REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS",
    "REVIEW_EMIT_V1_TARGET_FPS",
    "ResolvedSyncVideoSources",
    "ResultsPackageProjectSummary",
    "ResultsPackageRunResult",
    "ResultsPackageRunSummary",
    "ResultsPackagingConfig",
    "ResultsPackagingError",
    "ResultsPackagingRequest",
    "ReviewEmitV1Defaults",
    "ReviewEmitV1Params",
    "S9SourceValidationResult",
    "SETTINGS_SCHEMA_VERSION",
    "SPRINT1_RESULT_SCHEMA_VERSION",
    "SPRINT2_RESULT_SCHEMA_VERSION",
    "SPRINT3_RESULT_SCHEMA_VERSION",
    "STATUS_COMPLETED",
    "STATUS_COMPLETED_WITH_WARNINGS",
    "STATUS_FAILED",
    "STALE_SOURCE",
    "SYNC_INDEX_FILENAME",
    "SYNC_SCHEMA_VERSION",
    "Sprint1PackagingResult",
    "Sprint2PackagingResult",
    "SyncIndexBuildResult",
    "SyncIndexError",
    "TARGET_FPS",
    "VERIFICATION_STRENGTH_CONTENT_HASH",
    "VERIFICATION_STRENGTH_SIZE_MTIME",
    "apply_review_emit_v1",
    "assert_lossless_bout_copy",
    "assert_structural_qc",
    "authoritative_bout_content_key",
    "build_dependency_manifest",
    "build_ethogram_summary",
    "build_package_bout_id",
    "build_sync_index_from_prepared_sync",
    "build_sync_index_from_source_arrays",
    "config_from_mapping",
    "default_suite_config_path",
    "dependency_fingerprint_token",
    "evaluate_results_package_qc",
    "find_latest_eligible_s9_run",
    "is_s9_run_eligible_for_downstream",
    "list_completed_s9_runs",
    "load_results_packaging_config",
    "map_bout_boundaries_to_prepared_frames",
    "map_single_bout_boundary",
    "package_behavior_bouts",
    "reopen_results_package",
    "review_emit_v1_params_from_mapping",
    "run_publication_from_session",
    "run_results_packaging_publication",
    "run_results_packaging_sprint1",
    "run_results_packaging_sprint2",
    "run_sprint1_from_session",
    "run_sprint2_from_session",
    "select_s8_run_for_packaging",
    "summarize_results_package_project",
    "validate_s8_source_for_packaging",
    "validate_s9_run_for_downstream",
    "validate_sync_index",
]
