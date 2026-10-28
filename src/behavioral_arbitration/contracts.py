"""Subsystem 08 (behavioral arbitration) contracts for Sprint 1–3.

Defines schema stamps, detector inventories, required S6 columns, evidence
states, run layout names, and shared errors. Sprint 2 adds hierarchy, adaptive
approach/retract, contact cleaning (no remultiplex), and pass-2 gates.
Sprint 3 adds temporal stitch / reciprocity / hysteresis rule codes and
publish schema stamps for states / bouts / QC / discovery.
"""

from __future__ import annotations

from behavioral_inference.contracts import (
    LOGICAL_DYADIC_DETECTORS,
    LOGICAL_FOCAL_DETECTORS,
    SCORE_ORIGIN_FUSION_ROUTED,
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
)
from combined_features.contracts import JOINT_TARGET_FPS

# Package / settings schema.
SETTINGS_SCHEMA_VERSION = "behavioral_arbitration_settings_v1"
INTAKE_RESULT_SCHEMA_VERSION = "behavioral_arbitration_intake_v1"
SPRINT1_RESULT_SCHEMA_VERSION = "behavioral_arbitration_sprint1_v1"
SPRINT2_RESULT_SCHEMA_VERSION = "behavioral_arbitration_sprint2_v1"
SPRINT3_RESULT_SCHEMA_VERSION = "behavioral_arbitration_sprint3_v1"
STAGE_DIAGNOSTICS_SCHEMA_VERSION = "behavioral_arbitration_stage_diag_v1"
STATES_SCHEMA_VERSION = "behavioral_arbitration_states_v1"
BOUTS_SCHEMA_VERSION = "behavioral_arbitration_bouts_v1"
RUN_META_SCHEMA_VERSION = "behavioral_arbitration_run_meta_v1"
QC_SCHEMA_VERSION = "behavioral_arbitration_qc_v1"

# Package / run layout (immutable publish reserved for later sprints).
BEHAVIORAL_ARBITRATION_DIRNAME = "behavioral_arbitration"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
QC_FILENAME = "qc.json"
PROCESSING_LOG_FILENAME = "processing_log.txt"
BEHAVIOR_STATES_FILENAME = "behavior_states.parquet"
BEHAVIOR_BOUTS_FILENAME = "behavior_bouts.parquet"
DIAGNOSTICS_FILENAME = "arbitration_diagnostics.parquet"

# Timeline.
TARGET_FPS = float(JOINT_TARGET_FPS)  # 120.0
FPS_ATOL = 1e-6

# Full-ethogram MVP detector coverage (Pipeline 100 + parallel SBS).
REQUIRED_DYADIC_DETECTORS: tuple[str, ...] = (
    "contact",
    "nose_to_nose",
    "mutually_disengaged",
    "side_by_side",
)
REQUIRED_FOCAL_DETECTORS: tuple[str, ...] = (
    "nose_to_body",
    "nose_to_tail",
    "approach",
    "retract",
)
# Sanity: required sets are subsets of S7 logical inventories.
assert set(REQUIRED_DYADIC_DETECTORS) <= set(LOGICAL_DYADIC_DETECTORS)
assert set(REQUIRED_FOCAL_DETECTORS) <= set(LOGICAL_FOCAL_DETECTORS)

SYMMETRIC_DYADIC_DETECTORS: tuple[str, ...] = (
    "nose_to_nose",
    "contact",
    "side_by_side",
    "mutually_disengaged",
)
PARALLEL_CHANNELS: tuple[str, ...] = ("contact", "side_by_side")
HIERARCHY_LAYER1_DETECTORS: tuple[str, ...] = (
    "nose_to_nose",
    "nose_to_body",
    "nose_to_tail",
)
HIERARCHY_LAYER2_DETECTORS: tuple[str, ...] = (
    "approach",
    "retract",
    "mutually_disengaged",
)
HIERARCHY_DETECTORS: tuple[str, ...] = (
    *HIERARCHY_LAYER1_DETECTORS,
    *HIERARCHY_LAYER2_DETECTORS,
)
EXCLUSIVE_ETHOGRAM_LABELS: tuple[str, ...] = (
    *HIERARCHY_DETECTORS,
    "engaged",
)
# Provisional Sprint-2 residual when all hierarchy evidence is unobserved (C3/P4).
# Sprint 3 finalize may promote unresolved invalid evidence to ``undefined``.
LABEL_ENGAGED = "engaged"
LABEL_UNRESOLVED = "unresolved"
LABEL_UNDEFINED = "undefined"
RESIDUAL_LAYER_WINNER = "layer_winner"
RESIDUAL_MD_BASELINE = "mutually_disengaged_baseline"
RESIDUAL_ENGAGED_VALID = "engaged_valid"
RESIDUAL_UNRESOLVED_INVALID = "unresolved_invalid_evidence"

# Run status lifecycle (mirror S5–S7).
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"

# Discovery classifications.
ELIGIBLE_DOWNSTREAM = "eligible"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STALE_SOURCE = "stale_source"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"

# Structural S6 columns (MVP hard-fail if absent).
REQUIRED_HARD_GAP_COLUMNS: tuple[str, ...] = (
    "distance_nose_i_nose_j_norm",
    "distance_nose_i_side_j_norm",
    "distance_nose_i_spine_j_norm",
    "distance_nose_i_tail_j_norm",
    "distance_spine_i_spine_j_norm",
    "cc_count",
    "is_total_occlusion",
)

# Recommended for postural vetoes (warn + skip dependent stage if absent).
RECOMMENDED_VETO_COLUMNS: tuple[str, ...] = (
    "body_contraction_rate_7",
    "behind_score_bl_j",
    "cos_bearing_i_to_j",
    "rolling_mean_intent_15",
    "rolling_mean_projected_vel_i_to_j_60",
    "rolling_speed_i_15",
    "rolling_speed_j_15",
    "focal_effort_ratio_15",
    "spine_directed_vel_15_i",
    "rolling_mean_projected_vel_combined_i_to_j_7",
)

# Recommended for later rescue / stitch (Sprint 2+; warn only in Sprint 1).
RECOMMENDED_RESCUE_COLUMNS: tuple[str, ...] = (
    "signed_kinematic_intent",
    "signed_kinematic_intent_nose_or_neck",
    "signed_kinematic_intent_neck",
    "rolling_speed_i_60",
)

FEATURE_JOIN_KEYS: tuple[str, ...] = (
    "exp_name",
    "dense_frame_idx",
    "focal_id",
    "partner_id",
)
DIRECTIONAL_IDENTITY_KEYS: tuple[str, ...] = (
    "dense_frame_idx",
    "focal_id",
    "partner_id",
)

# Per-detector evidence states (Sprint 1; no exclusive ethogram label yet).
EVIDENCE_VALID_POSITIVE = "valid_positive"
EVIDENCE_VALID_NEGATIVE = "valid_negative"
EVIDENCE_UNOBSERVED_INVALID_ROUTE = "unobserved_invalid_route"
EVIDENCE_MISSING_DETECTOR = "missing_detector"
EVIDENCE_RULE_VETOED = "rule_vetoed"

EVIDENCE_STATES: tuple[str, ...] = (
    EVIDENCE_VALID_POSITIVE,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_UNOBSERVED_INVALID_ROUTE,
    EVIDENCE_MISSING_DETECTOR,
    EVIDENCE_RULE_VETOED,
)

# Stage diagnostic rule codes.
RULE_EXPAND_DYADIC = "EXPAND_DYADIC"
RULE_MERGE_S6 = "MERGE_S6"
RULE_GATE_HARD_DISTANCE = "GATE_HARD_DISTANCE"
RULE_GATE_HARD_DISTANCE_PASS2 = "GATE_HARD_DISTANCE_PASS2"
RULE_VETO_APPROACH = "VETO_APPROACH"
RULE_VETO_RETRACT = "VETO_RETRACT"
RULE_VALIDITY_MASK = "VALIDITY_MASK"
RULE_HIERARCHY_WIN = "HIERARCHY_WIN"
RULE_HIERARCHY_DECAY = "DECAY_CARRY"
RULE_HIERARCHY_RESIDUAL = "HIERARCHY_RESIDUAL"
RULE_ADAPTIVE_APPROACH = "ADAPTIVE_APPROACH"
RULE_ADAPTIVE_RETRACT = "ADAPTIVE_RETRACT"
RULE_CONTACT_OPTICAL = "CONTACT_OPTICAL"
RULE_CONTACT_SCHMITT = "CONTACT_SCHMITT"
RULE_CONTACT_MORPH = "CONTACT_MORPH"
RULE_CONTACT_MIN_DUR = "CONTACT_MIN_DUR"
RULE_GAP_FILL = "GAP_FILL"
RULE_MIN_DURATION = "MIN_DURATION"
RULE_LABEL_SYNC = "LABEL_SYNC"
RULE_STITCH_CONTEXTUAL = "STITCH_CONTEXTUAL"
RULE_STITCH_DYNAMIC = "STITCH_DYNAMIC"
RULE_STITCH_IDLE_RETRACT = "STITCH_IDLE_RETRACT"
RULE_MIN_MD_LABEL = "MIN_MD_LABEL"
RULE_RESCUE_DYNAMIC = "RESCUE_DYNAMIC"
RULE_GATE_APPROACH_CLOSE = "GATE_APPROACH_CLOSE"
RULE_RECIPROCITY_NOSE_TO_NOSE = "RECIPROCITY_NOSE_TO_NOSE"
RULE_RECIPROCITY_MUTUALLY_DISENGAGED = "RECIPROCITY_MUTUALLY_DISENGAGED"
RULE_HYSTERESIS_ENGAGED = "HYSTERESIS_ENGAGED"
RULE_LABEL_FINALIZE = "LABEL_FINALIZE"
RULE_LABEL_UNDEFINED = "LABEL_UNDEFINED"

# Re-export S7 score origins for validity checks.
SCORE_ORIGINS = (
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
    SCORE_ORIGIN_FUSION_ROUTED,
)


class BehavioralArbitrationError(Exception):
    """Base error for Subsystem 08."""


class IntakeError(BehavioralArbitrationError):
    """S7/S6 joint intake or lineage validation failed."""


class ExpansionError(BehavioralArbitrationError):
    """Directional expansion or uniqueness failure."""


class GateError(BehavioralArbitrationError):
    """Scientific gate configuration or structural failure."""


class ConfigError(BehavioralArbitrationError):
    """Invalid S8 configuration."""


class HierarchyError(BehavioralArbitrationError):
    """Hierarchy / decay adjudication failure."""


class ContactCleanError(BehavioralArbitrationError):
    """Contact post-inference cleaning failure."""


class TemporalError(BehavioralArbitrationError):
    """Temporal stitch / reciprocity / hysteresis failure."""


class PublicationError(BehavioralArbitrationError):
    """Immutable S8 publish / artifact write failure."""
