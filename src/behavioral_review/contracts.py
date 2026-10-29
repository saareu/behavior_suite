"""Subsystem 10 (behavioral review) Sprint 1–4 contracts.

Source open, sync mapping, prepared-video access, read-only inspection,
framewise correction events, durable working drafts, review-status
vocabulary, and immutable reviewed-run publication / discovery.
"""

from __future__ import annotations

from behavioral_arbitration.contracts import (
    EXCLUSIVE_ETHOGRAM_LABELS as S8_EXCLUSIVE_ETHOGRAM_LABELS,
)
from behavioral_arbitration.contracts import LABEL_ENGAGED, LABEL_UNDEFINED
from behavioral_arbitration.contracts import (
    PARALLEL_CHANNELS as S8_PARALLEL_CHANNELS,
)
from behavioral_arbitration.contracts import TARGET_FPS as S8_TARGET_FPS
from results_packaging.contracts import (
    BOUT_NAVIGATION_COLUMNS,
    EXPECTED_S8_BOUTS_SCHEMA_VERSION,
    EXPECTED_S8_STATES_SCHEMA_VERSION,
)

# On-disk layout (working drafts created in Sprint 3; publish in Sprint 4).
BEHAVIORAL_REVIEW_DIRNAME = "behavioral_review"
BEHAVIORAL_REVIEW_RUN_PREFIX = "behavioral_review"
WORKING_DIRNAME = "working"
WORKING_MANIFEST_FILENAME = "working_manifest.json"
CORRECTION_EVENTS_FILENAME = "correction_events.jsonl"
JOURNAL_CURSOR_FILENAME = "journal_cursor.json"
REVIEW_BOUT_STATUS_FILENAME = "review_bout_status.jsonl"
NOTES_FILENAME = "notes.jsonl"
LOCKS_DIRNAME = "locks"
WRITER_LOCK_FILENAME = "writer.lock"
SNAPSHOTS_DIRNAME = "snapshots"

# Schema stamps.
INSPECTION_SESSION_SCHEMA_VERSION = "behavioral_review_inspection_session_v1"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "behavioral_review_source_snapshot_v1"
ETHOGRAM_WINDOW_SCHEMA_VERSION = "behavioral_review_ethogram_window_v1"
BOUT_BROWSER_SCHEMA_VERSION = "behavioral_review_bout_browser_v1"
EVIDENCE_WINDOW_SCHEMA_VERSION = "behavioral_review_evidence_window_v1"
TIMELINE_VIEW_SCHEMA_VERSION = "behavioral_review_timeline_view_v1"
CORRECTION_EVENTS_SCHEMA_VERSION = "behavioral_review_correction_events_v1"
JOURNAL_CURSOR_SCHEMA_VERSION = "behavioral_review_journal_cursor_v1"
WORKING_MANIFEST_SCHEMA_VERSION = "behavioral_review_working_manifest_v1"
BOUT_STATUS_SCHEMA_VERSION = "behavioral_review_bout_status_v1"
EDITING_SESSION_SCHEMA_VERSION = "behavioral_review_editing_session_v1"
REVIEWED_BOUTS_SCHEMA_VERSION = "behavioral_review_bouts_v1"
REVIEWED_STATES_SCHEMA_VERSION = "behavioral_review_states_v1"
REVIEW_MANIFEST_SCHEMA_VERSION = "behavioral_review_manifest_v1"
REVIEW_QC_SCHEMA_VERSION = "behavioral_review_qc_v1"
REVIEW_RUN_META_SCHEMA_VERSION = "behavioral_review_run_meta_v1"
REVIEW_SETTINGS_SCHEMA_VERSION = "behavioral_review_settings_v1"
CORRECTION_EVENTS_HISTORY_SCHEMA_VERSION = (
    "behavioral_review_correction_events_history_v1"
)

# Published artifact filenames (S10-B §7.1).
REVIEW_MANIFEST_FILENAME = "review_manifest.json"
CORRECTION_EVENTS_PARQUET_FILENAME = "correction_events.parquet"
CORRECTION_EVENTS_HISTORY_FILENAME = "correction_events_history.parquet"
REVIEWED_BEHAVIOR_STATES_FILENAME = "reviewed_behavior_states.parquet"
REVIEWED_BEHAVIOR_BOUTS_FILENAME = "reviewed_behavior_bouts.parquet"
REVIEW_BOUT_STATUS_PARQUET_FILENAME = "review_bout_status.parquet"
REVIEWER_NOTES_PARQUET_FILENAME = "reviewer_notes.parquet"
QC_FILENAME = "qc.json"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"

REQUIRED_PUBLISHED_SCIENCE_ARTIFACTS: tuple[str, ...] = (
    CORRECTION_EVENTS_PARQUET_FILENAME,
    CORRECTION_EVENTS_HISTORY_FILENAME,
    REVIEWED_BEHAVIOR_STATES_FILENAME,
    REVIEWED_BEHAVIOR_BOUTS_FILENAME,
    REVIEW_BOUT_STATUS_PARQUET_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    QC_FILENAME,
)

# Downstream source-type selection (automated vs reviewed).
SOURCE_TYPE_AUTOMATED = "automated"
SOURCE_TYPE_REVIEWED = "reviewed"

# Publication / run lifecycle statuses (historical; never rewritten for eligibility).
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"
PUBLICATION_STATE_PUBLISHED = "published"
PUBLICATION_STATE_FAILED = "failed"
RUN_LIFECYCLE_PUBLISHED_ACCEPTED = "published_accepted"
RUN_LIFECYCLE_SUPERSEDED = "superseded"

# Discovery classifications (current eligibility; distinct from historical status).
CLASS_ELIGIBLE = "eligible"
CLASS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
CLASS_STALE_PARENT = "stale_parent"
CLASS_DRAFT = "draft"
CLASS_FAILED_OR_INCOMPLETE = "failed_or_incomplete"
CLASS_SUPERSEDED = "superseded"
CLASS_MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"

# Analysis clock for reviewed bout durations (matches S8).
REVIEW_TARGET_FPS = float(S8_TARGET_FPS)  # 120.0
LABEL_RECONSTRUCTION_MANUAL = "manual_review"

EXPECTED_S8_STATES_SCHEMA = EXPECTED_S8_STATES_SCHEMA_VERSION
EXPECTED_S8_BOUTS_SCHEMA = EXPECTED_S8_BOUTS_SCHEMA_VERSION
EXPECTED_PACKAGE_BOUTS_SCHEMA = "results_package_bouts_v1"

# Channels / identity grains (match S8/S9).
CHANNEL_EXCLUSIVE = "exclusive"
CHANNEL_CONTACT = "contact"
CHANNEL_SIDE_BY_SIDE = "side_by_side"
CHANNEL_VALUES = frozenset({CHANNEL_EXCLUSIVE, CHANNEL_CONTACT, CHANNEL_SIDE_BY_SIDE})

IDENTITY_DIRECTIONAL = "directional"
IDENTITY_SYMMETRIC_PAIR = "symmetric_pair"

# Exclusive labels including published ``undefined`` (distinct from engaged).
EXCLUSIVE_ETHOGRAM_LABELS: tuple[str, ...] = (
    *S8_EXCLUSIVE_ETHOGRAM_LABELS,
    LABEL_UNDEFINED,
)
PARALLEL_CHANNELS: tuple[str, ...] = S8_PARALLEL_CHANNELS
LABEL_ENGAGED_VALUE = LABEL_ENGAGED
LABEL_UNDEFINED_VALUE = LABEL_UNDEFINED

# Evidence availability vocabulary (display only; not science rewrite).
EVIDENCE_AVAILABLE = "available"
EVIDENCE_MISSING = "missing"
EVIDENCE_UNAVAILABLE = "unavailable"
EVIDENCE_INVALID = "invalid"
EVIDENCE_VALID_NEGATIVE = "valid_negative"
EVIDENCE_VALID_POSITIVE = "valid_positive"
EVIDENCE_UNOBSERVED = "unobserved"

# Session modes.
MODE_INSPECTION = "inspection"
MODE_EDITING = "editing"

# Per-bout review status (U3: no uncertain in MVP).
STATUS_UNREVIEWED = "unreviewed"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_VOIDED = "voided"
BOUT_STATUS_VALUES = frozenset(
    {STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_VOIDED}
)
DECISION_STATUS_VALUES = frozenset(
    {STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED}
)

# Status attachment kinds.
STATUS_TARGET_SOURCE_BOUT = "source_bout"
STATUS_TARGET_REVIEWED_BOUT = "reviewed_bout"
STATUS_TARGET_INTERVAL_KEY = "interval_key"

# Bout id kinds for status/note targets.
BOUT_ID_KIND_SOURCE_S8 = "source_s8_bout_id"
BOUT_ID_KIND_PACKAGE = "package_bout_id"
BOUT_ID_KIND_REVIEWED = "reviewed_bout_id"
BOUT_ID_KIND_INTERVAL = "interval_key"

# Correction / review ops (U2 reciprocal deferred — not in active catalog).
OP_SET_EXCLUSIVE_LABEL = "set_exclusive_label"
OP_SET_PARALLEL_PRED = "set_parallel_pred"
OP_CLEAR_PARALLEL_PRED = "clear_parallel_pred"
OP_TRIM_BOUT_BOUNDARY = "trim_bout_boundary"
OP_SPLIT_BOUT_AT_SAMPLE = "split_bout_at_sample"
OP_MERGE_ADJACENT_BOUTS = "merge_adjacent_bouts"
OP_SET_BOUT_REVIEW_STATUS = "set_bout_review_status"
OP_ADD_NOTE = "add_note"
OP_EDIT_NOTE = "edit_note"
OP_DELETE_NOTE = "delete_note"
OP_SESSION_MARKER = "session_marker"

FRAMEWISE_OPS = frozenset(
    {
        OP_SET_EXCLUSIVE_LABEL,
        OP_SET_PARALLEL_PRED,
        OP_CLEAR_PARALLEL_PRED,
        OP_TRIM_BOUT_BOUNDARY,
        OP_SPLIT_BOUT_AT_SAMPLE,
        OP_MERGE_ADJACENT_BOUTS,
    }
)
GEOMETRY_CHANGING_OPS = frozenset(
    {
        OP_SET_EXCLUSIVE_LABEL,
        OP_SET_PARALLEL_PRED,
        OP_CLEAR_PARALLEL_PRED,
        OP_TRIM_BOUT_BOUNDARY,
        OP_SPLIT_BOUT_AT_SAMPLE,
        OP_MERGE_ADJACENT_BOUTS,
    }
)
STATUS_OPS = frozenset({OP_SET_BOUT_REVIEW_STATUS})
NOTE_OPS = frozenset({OP_ADD_NOTE, OP_EDIT_NOTE, OP_DELETE_NOTE})
KNOWN_OPS = frozenset(
    FRAMEWISE_OPS | STATUS_OPS | NOTE_OPS | {OP_SESSION_MARKER}
)

VOID_REASON_GEOMETRY_CHANGED = "geometry_changed"
RUN_LIFECYCLE_NOT_STARTED = "not_started"
RUN_LIFECYCLE_IN_PROGRESS = "in_progress"
PUBLICATION_STATE_DRAFT = "draft"

# Writer lock policy.
LOCK_STALE_SECONDS_DEFAULT = 3600.0

# Default window / LOD budgets (GUI may override).
DEFAULT_ETHOGRAM_HALF_WINDOW = 240  # ±2 s at 120 Hz
DEFAULT_EVIDENCE_HALF_WINDOW = 120  # ±1 s
DEFAULT_TIMELINE_MAX_SEGMENTS = 2_000

# Core S8 state columns for ethogram inspection (column-projected reads).
ETHOGRAM_IDENTITY_COLUMNS: tuple[str, ...] = (
    "dense_frame_idx",
    "sample_idx",
    "time_sec",
    "focal_id",
    "partner_id",
    "pair_id",
    "mouse_a",
    "mouse_b",
)
ETHOGRAM_LABEL_COLUMNS: tuple[str, ...] = (
    "behavior_label",
    "label_reconstruction",
    "undefined_reason",
    "hierarchy_residual_kind",
)
ETHOGRAM_PARALLEL_COLUMNS: tuple[str, ...] = (
    "pred_contact",
    "prob_contact",
    "pred_side_by_side",
    "prob_side_by_side",
)
ETHOGRAM_ECHO_VALIDITY_COLUMNS: tuple[str, ...] = (
    "input_valid_contact",
    "score_origin_contact",
    "arbitration_input_valid_contact",
    "input_valid_side_by_side",
    "score_origin_side_by_side",
    "arbitration_input_valid_side_by_side",
    "input_valid_nose_to_nose",
    "score_origin_nose_to_nose",
    "arbitration_input_valid_nose_to_nose",
    "input_valid_nose_to_body",
    "score_origin_nose_to_body",
    "arbitration_input_valid_nose_to_body",
    "input_valid_nose_to_tail",
    "score_origin_nose_to_tail",
    "arbitration_input_valid_nose_to_tail",
    "input_valid_approach",
    "score_origin_approach",
    "arbitration_input_valid_approach",
    "input_valid_retract",
    "score_origin_retract",
    "arbitration_input_valid_retract",
    "input_valid_mutually_disengaged",
    "score_origin_mutually_disengaged",
    "arbitration_input_valid_mutually_disengaged",
)
ETHOGRAM_ECHO_PROB_COLUMNS: tuple[str, ...] = (
    "prob_nose_to_nose",
    "prob_nose_to_body",
    "prob_nose_to_tail",
    "prob_approach",
    "prob_retract",
    "prob_mutually_disengaged",
    "prob_engaged",
    "pred_nose_to_nose",
    "pred_nose_to_body",
    "pred_nose_to_tail",
    "pred_approach",
    "pred_retract",
    "pred_mutually_disengaged",
    "pred_engaged",
)

# Required packaged bout science + seek columns for S10 open.
REQUIRED_PACKAGE_BOUT_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "source_s8_bout_id",
    "channel",
    "identity_grain",
    "behavior_name",
    "focal_id",
    "partner_id",
    "mouse_a_id",
    "mouse_b_id",
    "pair_id",
    "start_sample_idx",
    "end_sample_idx",
    "bout_duration_frames",
    "bout_duration_sec",
    "package_run_id",
    "s8_bouts_schema_version",
    *BOUT_NAVIGATION_COLUMNS,
)


class BehavioralReviewError(Exception):
    """Base error for Subsystem 10."""


class SourceError(BehavioralReviewError):
    """S9 package intake / eligibility / provenance failure."""


class SyncMappingError(BehavioralReviewError):
    """Dense ↔ prepared mapping failure."""


class VideoAccessError(BehavioralReviewError):
    """Prepared-video open / decode failure."""


class SessionError(BehavioralReviewError):
    """Read-only inspection session navigation failure."""


class InspectionError(BehavioralReviewError):
    """Ethogram / bout / evidence inspection failure (read-only layer)."""


class EditingError(BehavioralReviewError):
    """Edit-mode gate, correction validation, or apply failure."""


class JournalError(BehavioralReviewError):
    """Durable event journal / cursor / recovery failure."""


class LockError(BehavioralReviewError):
    """Single-writer lock conflict or stale-lock policy failure."""


class DraftError(BehavioralReviewError):
    """Working draft create / open / identity failure."""


class PublicationError(BehavioralReviewError):
    """Reviewed-run publication / eligibility / discovery failure."""


class ProvenanceError(PublicationError):
    """Parent package / source digest drift blocking publication or use."""
