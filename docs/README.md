# behavior_suite Documentation

## Subsystem status

| Subsystem | Status | Downstream boundary |
| --- | --- | --- |
| 01 — Video Preprocessing | Functionally closed / maintenance | Hands prepared video + timing to S2 / S5 |
| 02 — Pose Inference and Technical QC | MVP finalized / validated | Hands completed `pose.parquet` run to S3 |
| 03 — Tracking Correction and Verification | MVP implemented / validated | Hands accepted `tracked_pose.parquet` to S4 |
| 04 — Pose Finalization | MVP complete / validated | Hands `final_pose.parquet` to later analysis |
| 05 — Video / Blob Feature Extraction | MVP complete / lab-validated | Consumes eligible S1; feeds S6 Combined Features |
| 06 - Combined Features | MVP complete / lab-validated | Eligible S4 + S5 -> `dyadic_features_full.parquet` + model-compat report |
| 07 — Behavioral Inference | MVP complete / lab-validated | Consumes eligible S6; publishes predictions for S8 |
| 08 — Behavioral Arbitration | MVP complete / lab-validated | Consumes eligible S7 (+ linked S6 features); Group 3 beside S7 |
| 09 — Results Packaging & Export | Core MVP lab-accepted; optional media/CSV export under design (S9-E1) | Group 4 packaging; consumes eligible S8; S10 later |
| 10 — Behavioral Review & Editing | Proposed / not started | Consumes S9 packages; optional manual review — not started |

## Subsystem 01 — Video Preprocessing

- [Functional Specification](subsystem_01/preprocessing.md)
- [Status and Roadmap](subsystem_01/status_and_roadmap.md)
- [Geometry Modes](subsystem_01/design/geometry_modes.md)
- [Issue Logs](subsystem_01/issue_logs/)
- [Historical Archive](subsystem_01/archive/)

## Subsystem 02 — SLEAP-NN Inference (MVP Finalized)

- [Final MVP Scope, Completion Status, and Roadmap][s2-mvp]
- [Minimal Pose Inference Contract][s2-minimal-contract]
- [Backend Pose Inference Contract][s2-backend-contract]
- [Backend Inference Acceptance-Test Specification][s2-backend-acceptance]
- [Full GPU MVP Acceptance Evidence][s2-gpu-acceptance]

## Subsystem 03 — Tracking Correction (MVP Implemented / Validated)

- [Final MVP Scope, Completion Status, and Roadmap][s3-mvp]
- [Minimal MVP Implementation Contract][s3-minimal-contract]
- [Tracking Correction Vision (Gate 1 history)][s3-vision]
- [Tracking Workflow and Review Design (Gate 2 history)][s3-workflow]
- [Assumptions and Current-Profile Constraints][s3-assumptions]
- [Legacy Corrector Integration Architecture (Gate 4 history)][s3-architecture]
- [Current Corrector Behavioral Inventory][s3-inventory]
- [Correction Rule Catalog][s3-rules]
- [Golden Regression Plan][s3-golden]

## Subsystem 04 — Pose Finalization (MVP Complete / Validated)

- [MVP Scope and Roadmap](subsystem_04/mvp_scope_and_roadmap.md)

## Subsystem 05 — Video / Blob Feature Extraction (MVP Complete / Lab-Validated)

- [Final MVP Scope, Completion Status, and Roadmap](subsystem_05/mvp_scope_and_roadmap.md)
- [Sprint 0 Repository Preflight and Integration Audit](subsystem_05/sprint_0_repository_audit.md)
- [Sprint 1 Backend Foundation](subsystem_05/sprint_1_backend_foundation.md)
- [Sprint 2 Complete Extraction Backend](subsystem_05/sprint_2_complete_extraction.md)
- [Sprint 3 Interactive Video Segmentation GUI](subsystem_05/sprint_3_interactive_gui.md)
- [Sprint 4 Full GUI Integration and MVP Acceptance](subsystem_05/sprint_4_full_gui_integration.md)

## Subsystem 06 - Combined Pose & Video Feature Generation (MVP Complete / Lab-Validated)

S6 MVP is complete and lab-validated. Sprints 1-4 delivered headless joint
S4/S5 processing plus Group 2 GUI. Operator laboratory GUI acceptance passed
on `E_full_session_integration` (2026-10-25). S6 does not train models or run
inference. Subsystem 07 Behavioral Inference is MVP complete / lab-validated.

- [Final MVP Scope, Completion Status, and Roadmap](subsystem_06/mvp_scope_and_roadmap.md)
- [Architecture Audit (S6-A)](subsystem_06/architecture_audit.md)
- [Implementation Contract (S6-B)](subsystem_06/implementation_contract.md)
- [Sprint 1 - Input Validation and S4 Pose Adapter](subsystem_06/sprint_1_input_adapter.md)
- [Sprint 2 - Feature Calculation Migration](subsystem_06/sprint_2_feature_migration.md)
- [Sprint 3 - Export, QC, Model Compatibility, Run Discovery](subsystem_06/sprint_3_export_compatibility.md)
- [Sprint 4 - GUI Integration and Laboratory Acceptance](subsystem_06/sprint_4_gui_integration.md)
- [Laboratory Acceptance Checklist](subsystem_06/laboratory_acceptance_checklist.md)

S6 consumes currently eligible S4 + S5 runs, aligns on the shared 120 Hz
timeline, adapts `final_pose.parquet` into the legacy Pipeline 50 /
`frame_features` input contract, and emits a complete directional feature
artifact plus a separate model-compat report (no inference).

## Subsystem 07 — Behavioral Inference (MVP Complete / Lab-Validated)

S7 MVP is complete and lab-validated. Sprints 1–4 delivered registry, unified
dyadic/focal inference, immutable artifacts/QC/discovery, and Group 3 GUI.
Operator laboratory GUI acceptance passed on `E_full_session_integration`
(2026-10-26) after install/registry/responsiveness remediation and obsolete
focal-directory cleanup. Detector bundles ship under tracked `models/`. S8
arbitration remains out of S7 scope; S8 is MVP complete / lab-validated in Group 3.

- [Final MVP Scope, Completion Status, and Roadmap](subsystem_07/mvp_scope_and_roadmap.md)
- [Architecture Audit (S7-A)](subsystem_07/architecture_audit.md)
- [Implementation Contract (S7-B)](subsystem_07/implementation_contract.md)
- [Sprint 1 — Model Registry and Runtime Foundation](subsystem_07/sprint_1_model_runtime.md)
- [Sprint 2 — Unified Dyadic and Focal Inference](subsystem_07/sprint_2_inference_engine.md)
- [Sprint 3 — Prediction Artifacts, QC, and Run Discovery](subsystem_07/sprint_3_artifacts_qc_discovery.md)
- [Sprint 4 — GUI Integration and Laboratory Acceptance](subsystem_07/sprint_4_gui_integration.md)
- [Laboratory Acceptance Checklist](subsystem_07/laboratory_acceptance_checklist.md)

S7 consumes eligible S6 `dyadic_features_full.parquet` tables, scores configured
dyadic/focal detector bundles, and publishes canonical + wide prediction
artifacts for downstream S8. Cross-model arbitration and temporal bout
post-processing remain Subsystem 08 (legacy Pipeline 100).

## Subsystem 08 — Behavioral Arbitration and Bout Generation (MVP Complete / Lab-Validated)

S8 MVP is complete and lab-validated. Sprints 1–4 delivered headless Pipeline 100
arbitration (intake through temporal/bouts/publish), Group 3 GUI beside S7,
responsiveness remediation, scientific E-session audit (**47/47**), and operator
laboratory GUI acceptance on `E_full_session_integration` (2026-10-28; S8 run
`behavioral_arbitration__20261028T133953`). Documented deviations from legacy
Pipeline 100 (invalid-evidence handling, `undefined`, no contact remultiplex)
are preserved. Manual bout correction stays downstream (S10 / legacy Pipeline 110
review). Group 4 packaging is delivered under Subsystem 09.

- [Final MVP Scope, Completion Status, and Roadmap](subsystem_08/mvp_scope_and_roadmap.md)
- [Architecture Audit (S8-A)](subsystem_08/architecture_audit.md)
- [Implementation Contract (S8-B)](subsystem_08/implementation_contract.md)
- [Sprint 1 — Source Intake, Directional Expansion and Initial Gates](subsystem_08/sprint_1_intake_gates.md)
- [Sprint 2 — Hierarchy, Adaptive Movement and Contact Arbitration](subsystem_08/sprint_2_arbitration_contact.md)
- [Sprint 3 — Temporal Rules, Bouts and Publication](subsystem_08/sprint_3_temporal_bouts_publication.md)
- [Sprint 4 — GUI Integration and Laboratory Acceptance](subsystem_08/sprint_4_gui_integration.md)
- [Laboratory Acceptance Checklist](subsystem_08/laboratory_acceptance_checklist.md)
- [Scientific Output Audit (E-session)](subsystem_08/scientific_output_audit_e_session.md)

S8 consumes eligible S7 prediction runs (plus the linked S6 feature columns for
gates / vetoes / optical assist / stitch rescue) and publishes immutable ethogram
states + bouts without modifying S7 outputs.

## Subsystem 09 — Results Packaging & Export (Core MVP Lab-Accepted)

S9-A/S9-B design is accepted. **Sprint 1** delivered headless S8 intake and
sync-index construction. **Sprint 2** added lossless bout packaging, ethogram
summaries, and optional `review_emit_v1`. **Sprint 3** publishes immutable S9
runs with QC, package manifests, discovery, and current eligibility. **Sprint 4**
adds Group 4 GUI (packaging-only; no S10). **Operator laboratory acceptance for
the core packaging MVP has passed.** Optional media (MP4) and automated CSV
export are under architectural design (S9-E1) before final S9 closure — stop for
review of that audit before implementation.

- [Architecture Audit (S9-A)](subsystem_09/architecture_audit.md)
- [Implementation Contract (S9-B)](subsystem_09/implementation_contract.md)
- [Sprint 1 — Input Validation and Synchronization Index](subsystem_09/sprint_1_intake_sync.md)
- [Sprint 2 — Bout Packaging, Summaries and Export Profiles](subsystem_09/sprint_2_bouts_summaries_exports.md)
- [Sprint 3 — Immutable Publication, QC and Run Discovery](subsystem_09/sprint_3_publication_qc_discovery.md)
- [Sprint 4 — GUI Integration and Laboratory Acceptance](subsystem_09/sprint_4_gui_integration.md)
- [Laboratory Acceptance Checklist](subsystem_09/laboratory_acceptance_checklist.md)
- [Media and CSV Export Architecture Audit (S9-E1)](subsystem_09/media_csv_export_architecture.md)

Group 4 conceptual separation: S9 packages eligible S8 outputs (plus sync /
provenance); S10 consumes S9 packages for interactive review and optional
versioned corrections (unimplemented). Media/CSV exports are optional derivatives
of existing packages and must not mutate authoritative S8/S9 artifacts.

## General

- [AI-Assisted Development Guide](general/development/ai_coding_guide.md)

[s2-mvp]: subsystem_02/mvp_scope_and_roadmap.md
[s2-minimal-contract]: subsystem_02/minimal_pose_inference_contract.md
[s2-backend-contract]: subsystem_02/sleap_inference_specification.md
[s2-backend-acceptance]: subsystem_02/acceptance_test_specification.md
[s2-gpu-acceptance]: subsystem_02/evidence/gpu_mvp_acceptance_v030.md
[s3-mvp]: subsystem_03/mvp_scope_and_roadmap.md
[s3-minimal-contract]: subsystem_03/minimal_implementation_contract.md
[s3-vision]: subsystem_03/tracking_correction_vision.md
[s3-workflow]: subsystem_03/tracking_workflow_and_review_design.md
[s3-assumptions]: subsystem_03/assumptions.md
[s3-architecture]: subsystem_03/legacy_corrector_integration_architecture.md
[s3-inventory]: subsystem_03/current_corrector_behavioral_inventory.md
[s3-rules]: subsystem_03/correction_rule_catalog.md
[s3-golden]: subsystem_03/golden_regression_plan.md
