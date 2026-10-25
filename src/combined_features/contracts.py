"""Subsystem 06 (combined pose & video features) contracts.

Defines package layout names, adapter policy stamps, joint-alignment tolerances,
legacy impute-stage codes, S6 run artifact names, and shared errors.
"""

from __future__ import annotations

from blob_features.contracts import (
    OCCLUSION_STATIC_FEATURE_COLS,
    OCCLUSION_TEMPORAL_FEATURE_COLS,
)
from pose_finalization.contracts import HEADSTAGE_NODE_NAME, TARGET_FPS_DEFAULT

# Package / run layout.
COMBINED_FEATURES_DIRNAME = "combined_features"
RUN_META_SCHEMA_VERSION = "combined_features_run_meta_v1"
SETTINGS_SCHEMA_VERSION = "combined_features_settings_v1"
DYADIC_FEATURES_SCHEMA_VERSION = "dyadic_features_full_v1"
QC_SCHEMA_VERSION = "combined_features_qc_v1"
MODEL_COMPAT_SCHEMA_VERSION = "combined_features_model_compat_v1"
ADAPTER_POLICY_VERSION = "combined_features_pose_adapter_v1"

DYADIC_FEATURES_FILENAME = "dyadic_features_full.parquet"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"
QC_FILENAME = "qc.json"
MODEL_COMPAT_FILENAME = "model_compat.json"

S6_RUN_ARTIFACT_FILENAMES = (
    DYADIC_FEATURES_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    MODEL_COMPAT_FILENAME,
)

# Required identity / timing columns on published dyadic_features_full.parquet.
REQUIRED_IDENTITY_COLUMNS = (
    "schema_version",
    "session_root",
    "focal_id",
    "partner_id",
    "sample_idx",
    "dense_frame_idx",
    "time_sec",
    "orig_frame_idx",
    "source_s4_run_id",
    "source_s5_run_id",
)

# S5 blob columns consumed by Pipeline 50 join.
# Mandatory: scientific measurements always produced by eligible S5 runs.
# Optional: none today — documented empty so callers do not invent columns.
MANDATORY_BLOB_FEATURE_COLS = (
    "cc_count",
    "cc_count_raw",
    "min_inter_contour_dist",
    *OCCLUSION_STATIC_FEATURE_COLS,
    *OCCLUSION_TEMPORAL_FEATURE_COLS,
)
OPTIONAL_BLOB_FEATURE_COLS: tuple[str, ...] = ()
BLOB_FEATURE_COLS = MANDATORY_BLOB_FEATURE_COLS + OPTIONAL_BLOB_FEATURE_COLS

# Active / obsolete detector names for registry discovery reporting.
ACTIVE_DYADIC_DETECTORS = (
    "contact_intact",
    "contact_occluded",
    "nose_to_nose",
    "side_by_side",
    "mutually_disengaged",
)
ACTIVE_FOCAL_DETECTORS = (
    "approach",
    "retract",
    "nose_body_intact",
    "nose_body_occ",
    "nose_to_body",
    "nose_to_tail",
    "nose_body",
    "nose_tail",
)
# Obsolete detectors always reported as unsupported.
OBSOLETE_DYADIC_DETECTORS = ("contact",)

# Discovery classifications (mirror S5).
ELIGIBLE_DOWNSTREAM = "eligible"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STALE_SOURCE = "stale_source"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"

# Joint published-grid tolerance (matches S5 QC index integrity).
JOINT_TIME_ATOL_SEC = 1e-12
JOINT_TARGET_FPS = float(TARGET_FPS_DEFAULT)

# Legacy Pipeline 30 / frame_features impute-stage encoding.
IMPUTE_STAGE_NONE = 0
IMPUTE_STAGE_LINEAR = 1
IMPUTE_STAGE_PCHIP = 2
IMPUTE_STAGE_KALMAN = 3

# Scientific nodes required after alias resolution (FF heading/body defaults).
REQUIRED_SCIENTIFIC_NODES = ("nose", "neck", "spine", "tail")

# Canonical ↔ alias pairs populated in the adapter node axis for FF compatibility.
NODE_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("spine", "spine_base"),
    ("tail", "tail_base"),
)

HEADSTAGE_NODE = HEADSTAGE_NODE_NAME

# Joint eligibility / alignment reason codes (stable for logs and GUI).
REASON_SESSION_MISMATCH = "session_mismatch"
REASON_SYNC_PATH_MISMATCH = "sync_path_mismatch"
REASON_SYNC_MISSING = "sync_missing"
REASON_SYNC_DIGEST_MISMATCH = "sync_digest_mismatch"
REASON_SYNC_DIGEST_UNAVAILABLE = "sync_digest_unavailable"
REASON_FPS_MISMATCH = "fps_mismatch"
REASON_SAMPLE_COUNT_MISMATCH = "sample_count_mismatch"
REASON_INDEX_INTEGRITY = "index_integrity"
REASON_TIME_GRID_MISMATCH = "time_grid_mismatch"
REASON_S4_STRUCTURE = "s4_structure"
REASON_S4_INELIGIBLE = "s4_ineligible"
REASON_S5_INELIGIBLE = "s5_ineligible"
REASON_MISSING_UPSTREAM = "missing_upstream"
REASON_LINEAGE_UNPROVABLE = "lineage_unprovable"

STATUS_NOT_STARTED = "not_started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"

# Implantation resolution outcomes (Sprint 1 adapter metadata).
IMPLANTATION_RESOLVED = "resolved"
IMPLANTATION_UNRESOLVED = "unresolved"
IMPLANTATION_ABSENT = "headstage_absent"
IMPLANTATION_AMBIGUOUS = "headstage_ambiguous"


class CombinedFeaturesError(RuntimeError):
    """Expected Subsystem 06 validation, alignment, or adapter error."""

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
