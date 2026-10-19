"""Minimal Subsystem 04 (pose finalization) contracts for Sprint 1.

These constants define the agreed MVP artifact names, required accepted-S3
inputs, the lab target timeline rate, and lightweight run-status values for
later runner integration. Sprint 1 does not write S4 outputs.
"""

from __future__ import annotations

POSE_FINALIZATION_DIRNAME = "pose_finalization"
RUN_META_SCHEMA_VERSION = "pose_finalization_run_meta_v1"
SETTINGS_SCHEMA_VERSION = "pose_finalization_settings_v1"

FINAL_POSE_FILENAME = "final_pose.parquet"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"

# Agreed eventual S4 run layout (not written in Sprint 1):
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
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class PoseFinalizationError(RuntimeError):
    """Expected Subsystem 04 validation or timing error."""
