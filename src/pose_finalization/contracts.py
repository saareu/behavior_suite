"""Minimal Subsystem 04 (pose finalization) contracts.

These constants define the agreed MVP artifact names, required accepted-S3
inputs, the lab target timeline rate, dense-pose provenance codes, gap-fill
and SavGol defaults, final Parquet schema fields, QC thresholds, and
run-status values used by the Sprint 3 runner.
"""

from __future__ import annotations

POSE_FINALIZATION_DIRNAME = "pose_finalization"
RUN_META_SCHEMA_VERSION = "pose_finalization_run_meta_v1"
SETTINGS_SCHEMA_VERSION = "pose_finalization_settings_v1"
FINAL_POSE_SCHEMA_VERSION = "final_pose_v1"
QC_SCHEMA_VERSION = "pose_finalization_qc_v1"

FINAL_POSE_FILENAME = "final_pose.parquet"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"

# Agreed S4 run layout:
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

# Required accepted-S3 handoff fields for S4 runs.
REQUIRED_ACCEPTED_S3_INPUT_FIELDS = (
    "s3_run_dir",
    "tracked_pose_parquet",
    "prepared_sync",
    "session_root",
)

STATUS_NOT_STARTED = "not_started"
STATUS_TIMING_RESOLVED = "timing_resolved"
STATUS_DENSE_CONSTRUCTED = "dense_constructed"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
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

# Savitzky–Golay defaults (applied after temporal resampling and gap fill).
SAVGOL_ENABLED_DEFAULT = True
SAVGOL_WINDOW_LENGTH_DEFAULT = 11
SAVGOL_POLYORDER_DEFAULT = 2

# Lightweight QC warning thresholds (practical defaults; not scientific pass/fail).
QC_MISSING_FRACTION_WARN_DEFAULT = 0.25
QC_IMPUTED_FRACTION_WARN_DEFAULT = 0.50
QC_JUMP_PX_WARN_DEFAULT = 40.0
QC_SMOOTH_DISPLACEMENT_PX_WARN_DEFAULT = 5.0

# Node excluded from gap imputation (legacy Pipeline 30 behavior).
HEADSTAGE_NODE_NAME = "headstage"
DEFAULT_IMPUTATION_EXCLUDED_NODES = (HEADSTAGE_NODE_NAME,)

# Long-form final_pose.parquet columns (authoritative S4 numeric export).
FINAL_POSE_COLUMNS = (
    "schema_version",
    "sample_idx",
    "time_sec",
    "track",
    "node",
    "x",
    "y",
    "x_pre_smooth",
    "y_pre_smooth",
    "point_score",
    "instance_score",
    "provenance",
    "provenance_name",
    "protected_manual_blank",
    "was_smoothed",
    "left_source_frame_idx",
    "right_source_frame_idx",
    "interpolation_weight",
    "is_exact_source_sample",
)


class PoseFinalizationError(RuntimeError):
    """Expected Subsystem 04 validation or timing error."""
