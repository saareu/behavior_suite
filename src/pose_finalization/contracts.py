"""Minimal Subsystem 04 (pose finalization) contracts.

These constants define the agreed MVP artifact names, required accepted-S3
inputs, the lab target timeline rate, dense-pose provenance codes, gap-fill
defaults, and lightweight run-status values for later runner integration.
Sprint 2 does not write S4 outputs.
"""

from __future__ import annotations

POSE_FINALIZATION_DIRNAME = "pose_finalization"
RUN_META_SCHEMA_VERSION = "pose_finalization_run_meta_v1"
SETTINGS_SCHEMA_VERSION = "pose_finalization_settings_v1"

FINAL_POSE_FILENAME = "final_pose.parquet"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"

# Agreed eventual S4 run layout (not written in Sprint 1/2):
#   pose_finalization/<run_id>/
#     final_pose.parquet
#     run_meta.json
#     settings_used.yaml
#     processing_log.txt
S4_RUN_ARTIFACT_FILENAMES = (
    FINAL_POSE_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    PROCESSING_LOG_FILENAME,
)

TARGET_FPS_DEFAULT = 120.0

# Required accepted-S3 handoff fields for later S4 runs.
REQUIRED_ACCEPTED_S3_INPUT_FIELDS = (
    "s3_run_dir",
    "tracked_pose_parquet",
    "prepared_sync",
    "session_root",
)

STATUS_NOT_STARTED = "not_started"
STATUS_TIMING_RESOLVED = "timing_resolved"
STATUS_DENSE_CONSTRUCTED = "dense_constructed"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Dense-pose sample provenance (uint8). Reconstructed samples never receive
# fabricated SLEAP confidence scores.
PROVENANCE_MISSING = 0
PROVENANCE_ORIGINAL = 1
PROVENANCE_IDENTITY_CORRECTED = 2
PROVENANCE_TEMPORAL_RESAMPLE = 3
PROVENANCE_LINEAR_GAP_FILL = 4
PROVENANCE_KALMAN_GAP_FILL = 5
PROVENANCE_PROTECTED_MANUAL_BLANK = 6

PROVENANCE_NAMES = {
    PROVENANCE_MISSING: "missing",
    PROVENANCE_ORIGINAL: "original",
    PROVENANCE_IDENTITY_CORRECTED: "identity_corrected",
    PROVENANCE_TEMPORAL_RESAMPLE: "temporal_resample",
    PROVENANCE_LINEAR_GAP_FILL: "linear_gap_fill",
    PROVENANCE_KALMAN_GAP_FILL: "kalman_gap_fill",
    PROVENANCE_PROTECTED_MANUAL_BLANK: "protected_manual_blank",
}

# Gap policy defaults (target-sample counts at TARGET_FPS_DEFAULT).
# Configurable for future GUI Advanced Settings.
GAP_LINEAR_MAX_FRAMES_DEFAULT = 20
GAP_KALMAN_MAX_FRAMES_DEFAULT = 140

# Node excluded from gap imputation (legacy Pipeline 30 behavior).
HEADSTAGE_NODE_NAME = "headstage"
DEFAULT_IMPUTATION_EXCLUDED_NODES = (HEADSTAGE_NODE_NAME,)


class PoseFinalizationError(RuntimeError):
    """Expected Subsystem 04 validation or timing error."""
