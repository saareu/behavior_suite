# behavior_suite Documentation

## Subsystem status

| Subsystem | Status | Downstream boundary |
| --- | --- | --- |
| 01 — Video Preprocessing | Functionally closed / maintenance | Hands prepared video + timing to S2 / S5 |
| 02 — Pose Inference and Technical QC | MVP finalized / validated | Hands completed `pose.parquet` run to S3 |
| 03 — Tracking Correction and Verification | MVP implemented / validated | Hands accepted `tracked_pose.parquet` to S4 |
| 04 — Pose Finalization | MVP complete / validated | Hands `final_pose.parquet` to later analysis |
| 05 — Video / Blob Feature Extraction | MVP complete / lab-validated | Consumes eligible S1; feeds upcoming S6 Combined Features |
| 06 — Combined Features | Not started (upcoming) | Will consume eligible S5 (+ S4 timelines) |

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

## Subsystem 06 — Combined Features (Upcoming)

Not started. S6 will consume currently eligible S5 runs and join on the shared
120 Hz timeline with S4-compatible pose features. See the S5 roadmap handoff
boundary for selection policy.

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
