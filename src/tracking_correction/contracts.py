"""Minimal Subsystem 03 shell contracts.

These types keep orchestration independent of the current corrector. They are
not a pose-schema redesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_PROFILE_ID = "current_lab_two_mouse_headstage_v1"
CURRENT_BACKEND_ID = "current_lab_corrector"
RUN_META_SCHEMA_VERSION = "tracking_correction_run_meta_v1"
SETTINGS_SCHEMA_VERSION = "tracking_correction_settings_v1"

WORKING_TRACKED_POSE_FILENAME = "working_tracked_pose.parquet"
MACHINE_CORRECTIONS_FILENAME = "machine_corrections.json"
FINAL_TRACKED_POSE_FILENAME = "tracked_pose.parquet"

# S2 pose.parquet is the authoritative numeric handoff. These are the columns
# the current corrector needs; they are not a new S3 output schema.
REQUIRED_POSE_COLUMNS = ("frame_idx", "track", "node", "x", "y", "score")
REQUIRED_PROFILE_NODES = ("nose", "neck", "spine_base", "tail_base", "headstage")
EXPECTED_TRACK_INDICES = (0, 1)

STATUS_HANDOFF_VALIDATED = "handoff_validated"
STATUS_DRY_RUN_COMPLETE = "dry_run_complete"
STATUS_CORRECTION_COMPLETE = "correction_complete"
STATUS_CORRECTION_FAILED = "correction_failed"

BACKEND_STATUS_NOT_RUN = "not_run"
BACKEND_STATUS_RUNNING = "running"
BACKEND_STATUS_COMPLETED = "completed"
BACKEND_STATUS_FAILED = "failed"

ACCEPTANCE_NOT_ACCEPTED = "not_accepted"


class TrackingCorrectionError(RuntimeError):
    """Expected Subsystem 03 handoff or workspace error."""


@dataclass(frozen=True)
class TrackingCorrectionRequest:
    """Inputs for one Subsystem 03 shell run or dry-run."""

    s2_run_dir: Path
    output_root: Path | None = None
    run_purpose: str = "development"
    dry_run: bool = False
    timestamp: str | None = None


@dataclass(frozen=True)
class BackendResult:
    """Compact result returned by a tracking backend adapter."""

    backend_id: str
    profile_id: str
    working_tracked_pose_path: Path
    machine_corrections_path: Path
    correction_summary: dict[str, Any]
    blanked_node_count: int
    swapped_frame_count: int
    input_row_count: int
    output_row_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackingCorrectionResult:
    """Summary returned after a validated S3 workspace is recorded."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    profile_id: str
    backend_id: str
    backend_status: str
    pose_parquet_path: Path
    prepared_video_path: Path
    working_tracked_pose_path: Path | None = None
    machine_corrections_path: Path | None = None
    correction_summary: dict[str, Any] | None = None
