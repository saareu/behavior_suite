"""Subsystem 08 — Behavioral Arbitration and Bout Generation (Sprint 1–4).

Sprint 1: S7/S6 joint intake, validity normalization, directional expansion,
S6 feature merge, hard-distance gates, and postural vetoes.

Sprint 2: hierarchy + decay, adaptive approach/retract, contact cleaning
(without remultiplex), and pass-2 hard-distance gates.

Sprint 3: temporal stitch / reciprocity / hysteresis / finalize, bout
collapse, QC, discovery, and immutable Parquet publish.

Sprint 4: Group 3 GUI orchestration reuses these headless APIs (science stays here).
"""

from behavioral_arbitration.adaptive import (
    apply_adaptive_approach_thresholding,
    apply_adaptive_retract_thresholding,
)
from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_arbitration.config import (
    BehavioralArbitrationConfig,
    ExecutionSettings,
    ScientificSettings,
    config_from_mapping,
    default_suite_config_path,
    load_behavioral_arbitration_config,
)
from behavioral_arbitration.contact_clean import (
    apply_contact_arbitration_no_remultiplex,
    get_cleaned_contact_bouts,
)
from behavioral_arbitration.contracts import (
    BEHAVIORAL_ARBITRATION_DIRNAME,
    BOUTS_SCHEMA_VERSION,
    EVIDENCE_MISSING_DETECTOR,
    EVIDENCE_RULE_VETOED,
    EVIDENCE_UNOBSERVED_INVALID_ROUTE,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_VALID_POSITIVE,
    LABEL_ENGAGED,
    LABEL_UNDEFINED,
    LABEL_UNRESOLVED,
    PARALLEL_CHANNELS,
    QC_SCHEMA_VERSION,
    REQUIRED_DYADIC_DETECTORS,
    REQUIRED_FOCAL_DETECTORS,
    REQUIRED_HARD_GAP_COLUMNS,
    SETTINGS_SCHEMA_VERSION,
    SPRINT1_RESULT_SCHEMA_VERSION,
    SPRINT2_RESULT_SCHEMA_VERSION,
    SPRINT3_RESULT_SCHEMA_VERSION,
    STATES_SCHEMA_VERSION,
    TARGET_FPS,
    BehavioralArbitrationError,
    ConfigError,
    ContactCleanError,
    ExpansionError,
    GateError,
    HierarchyError,
    IntakeError,
    PublicationError,
    TemporalError,
)
from behavioral_arbitration.expand import expand_dyadic_to_directional
from behavioral_arbitration.export import (
    build_arbitration_diagnostics_table,
    build_behavior_states_table,
    write_behavior_bouts_parquet,
    write_behavior_states_parquet,
    write_optional_diagnostics_parquet,
)
from behavioral_arbitration.gates import (
    apply_hard_distance_probability_gates,
    apply_postural_approach_veto,
    apply_postural_retract_veto,
    merge_s6_features,
)
from behavioral_arbitration.hierarchy import (
    calculate_normalized_margin,
    resolve_hierarchical_conflicts,
)
from behavioral_arbitration.intake import (
    JointIntakeBundle,
    S7S8ReadinessAssessment,
    assess_s7_for_s8,
    find_latest_s8_ready_s7_run,
    list_s7_options_for_s8,
    resolve_linked_s6_run,
    select_eligible_s7_run,
    validate_joint_s7_s6_for_s8,
)
from behavioral_arbitration.qc import (
    QcSummary,
    assert_structural_qc,
    evaluate_behavioral_arbitration_qc,
)
from behavioral_arbitration.results import (
    BehavioralArbitrationRunResult,
    Sprint1ArbitrationResult,
    Sprint2ArbitrationResult,
    Sprint3ArbitrationResult,
    StageDiagnostic,
)
from behavioral_arbitration.run_discovery import (
    BehavioralArbitrationProjectSummary,
    BehavioralArbitrationRunSummary,
    find_latest_eligible_s8_run,
    summarize_behavioral_arbitration_project,
)
from behavioral_arbitration.runner import (
    BehavioralArbitrationRequest,
    apply_sprint2_stages,
    apply_sprint3_stages,
    is_s8_run_eligible_for_downstream,
    run_behavioral_arbitration_publication,
    run_behavioral_arbitration_sprint1,
    run_behavioral_arbitration_sprint2,
    run_behavioral_arbitration_sprint3,
    run_sprint1_from_session,
    run_sprint2_from_session,
    run_sprint3_from_session,
    validate_s8_run_for_downstream,
)
from behavioral_arbitration.validity import (
    classify_evidence_state,
    normalize_arbitration_evidence,
)

__all__ = [
    "BEHAVIORAL_ARBITRATION_DIRNAME",
    "BOUTS_SCHEMA_VERSION",
    "BehavioralArbitrationConfig",
    "BehavioralArbitrationError",
    "BehavioralArbitrationProjectSummary",
    "BehavioralArbitrationRequest",
    "BehavioralArbitrationRunResult",
    "BehavioralArbitrationRunSummary",
    "ConfigError",
    "ContactCleanError",
    "EVIDENCE_MISSING_DETECTOR",
    "EVIDENCE_RULE_VETOED",
    "EVIDENCE_UNOBSERVED_INVALID_ROUTE",
    "EVIDENCE_VALID_NEGATIVE",
    "EVIDENCE_VALID_POSITIVE",
    "ExecutionSettings",
    "ExpansionError",
    "GateError",
    "HierarchyError",
    "IntakeError",
    "JointIntakeBundle",
    "LABEL_ENGAGED",
    "LABEL_UNDEFINED",
    "LABEL_UNRESOLVED",
    "PARALLEL_CHANNELS",
    "PublicationError",
    "QC_SCHEMA_VERSION",
    "QcSummary",
    "REQUIRED_DYADIC_DETECTORS",
    "REQUIRED_FOCAL_DETECTORS",
    "REQUIRED_HARD_GAP_COLUMNS",
    "SETTINGS_SCHEMA_VERSION",
    "SPRINT1_RESULT_SCHEMA_VERSION",
    "SPRINT2_RESULT_SCHEMA_VERSION",
    "SPRINT3_RESULT_SCHEMA_VERSION",
    "STATES_SCHEMA_VERSION",
    "S7S8ReadinessAssessment",
    "ScientificSettings",
    "Sprint1ArbitrationResult",
    "Sprint2ArbitrationResult",
    "Sprint3ArbitrationResult",
    "StageDiagnostic",
    "TARGET_FPS",
    "TemporalError",
    "apply_adaptive_approach_thresholding",
    "apply_adaptive_retract_thresholding",
    "apply_contact_arbitration_no_remultiplex",
    "apply_hard_distance_probability_gates",
    "apply_postural_approach_veto",
    "apply_postural_retract_veto",
    "apply_sprint2_stages",
    "apply_sprint3_stages",
    "assert_structural_qc",
    "assess_s7_for_s8",
    "build_arbitration_diagnostics_table",
    "build_behavior_states_table",
    "calculate_normalized_margin",
    "classify_evidence_state",
    "config_from_mapping",
    "default_suite_config_path",
    "evaluate_behavioral_arbitration_qc",
    "expand_dyadic_to_directional",
    "extract_behavior_bouts",
    "find_latest_eligible_s8_run",
    "find_latest_s8_ready_s7_run",
    "get_cleaned_contact_bouts",
    "is_s8_run_eligible_for_downstream",
    "list_s7_options_for_s8",
    "load_behavioral_arbitration_config",
    "merge_s6_features",
    "normalize_arbitration_evidence",
    "resolve_hierarchical_conflicts",
    "resolve_linked_s6_run",
    "run_behavioral_arbitration_publication",
    "run_behavioral_arbitration_sprint1",
    "run_behavioral_arbitration_sprint2",
    "run_behavioral_arbitration_sprint3",
    "run_sprint1_from_session",
    "run_sprint2_from_session",
    "run_sprint3_from_session",
    "select_eligible_s7_run",
    "summarize_behavioral_arbitration_project",
    "validate_joint_s7_s6_for_s8",
    "validate_s8_run_for_downstream",
    "write_behavior_bouts_parquet",
    "write_behavior_states_parquet",
    "write_optional_diagnostics_parquet",
]
