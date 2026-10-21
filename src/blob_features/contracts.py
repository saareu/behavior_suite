"""Subsystem 05 (blob feature extraction) contracts.

Defines artifact names, schema versions, frozen output column names, and the
shared error type. Column names align with legacy Pipeline 40 plus S1/S4
timing fields required for the Pipeline 50 join.
"""

from __future__ import annotations

# Package / run layout.
BLOB_FEATURES_DIRNAME = "blob_features"
RUN_META_SCHEMA_VERSION = "blob_features_run_meta_v1"
SETTINGS_SCHEMA_VERSION = "blob_features_settings_v1"
BLOB_FEATURES_SCHEMA_VERSION = "blob_features_v1"
QC_SCHEMA_VERSION = "blob_features_qc_v1"

BLOB_FEATURES_FILENAME = "blob_features.parquet"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"
QC_FILENAME = "qc.json"

# Agreed S5 run layout:
#   blob_features/<run_id>/
#     blob_features.parquet
#     run_meta.json
#     settings_used.yaml
#     processing_log.txt
#     qc.json
S5_RUN_ARTIFACT_FILENAMES = (
    BLOB_FEATURES_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
)

# Lab target clock (identical to S4). Timeline construction reuses
# pose_finalization.timing.build_uniform_target_timeline.
TARGET_FPS_DEFAULT = 120.0

# Required eligible-S1 files under session preprocess/ (stricter than S2 handoff).
REQUIRED_S1_FILES = (
    "prepared_video.mp4",
    "cropped_background.png",
    "prepared_sync.npz",
    "prepare_meta.json",
)

STATUS_NOT_STARTED = "not_started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"

# Legacy Pipeline 40 occlusion static columns (values filled in Sprint 2+).
OCCLUSION_STATIC_FEATURE_COLS = (
    "ellipse_eccentricity",
    "circularity",
    "solidity",
    "num_dt_peaks",
    "furthest_endpoint_distance_px",
)

OCCLUSION_TEMPORAL_FEATURE_COLS = tuple(
    f"{feature}_mean_5" for feature in OCCLUSION_STATIC_FEATURE_COLS
) + tuple(f"{feature}_delta_5" for feature in OCCLUSION_STATIC_FEATURE_COLS) + tuple(
    f"{feature}_std_15" for feature in OCCLUSION_STATIC_FEATURE_COLS
)

# Authoritative Parquet columns.
#
# Indexing for downstream legacy Pipeline 50:
#   - ``dense_frame_idx`` is the join key (preferred over ``frame_idx``).
#   - ``frame_idx`` equals ``dense_frame_idx`` after 120 Hz mapping (legacy alias).
#   - ``sample_idx`` equals ``dense_frame_idx`` (S4-aligned name).
#   - ``orig_frame_idx`` is the nearest prepared-video frame index.
#   - ``time_sec`` is the authoritative uniform 120 Hz target time.
BLOB_FEATURES_COLUMNS = (
    "schema_version",
    "sample_idx",
    "dense_frame_idx",
    "frame_idx",
    "time_sec",
    "orig_frame_idx",
    "cc_count",
    "cc_count_raw",
    "min_inter_contour_dist",
    *OCCLUSION_STATIC_FEATURE_COLS,
    *OCCLUSION_TEMPORAL_FEATURE_COLS,
)

# Temporal gap-fill default from legacy/configs/blob_features.yaml.
CC_COUNT_GAP_FILL_MAX_GAP_DEFAULT = 5

# Cable-artifact removal is excluded from the S5 MVP. The legacy lab YAML default
# is ``remove_cable_artifact: false``; S5 assumes that disabled state and does
# not expose cable parameters in active configuration or GUI scope.
CABLE_REMOVAL_EXCLUDED = True
CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT = False


class BlobFeaturesError(RuntimeError):
    """Expected Subsystem 05 validation, timing, config, or segmentation error."""
