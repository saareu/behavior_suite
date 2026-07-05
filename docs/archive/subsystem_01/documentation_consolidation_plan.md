# Documentation Consolidation Plan

> **Archive notice:** This historical document is retained for traceability. It is not the current source of truth. See `docs/subsystem_01/preprocessing.md`, `docs/subsystem_01/status_and_roadmap.md`, and `docs/subsystem_01/design/geometry_modes.md`.

**Status date:** 2026-07-05  
**Scope:** Audit tracked documentation, classify current sources of truth, and propose a low-risk consolidation plan.  
**Non-action:** No documents were moved, renamed, archived, or deleted in this milestone.

The untracked local `docs/issue_logs/` folder was intentionally not inspected and is not treated as a current source of truth. It should be reviewed later using the issue-log integration plan below.

---

## 1. Complete inventory of current tracked docs

The tracked documentation inventory at the start of this audit was:

| Document | Current classification |
| --- | --- |
| `README.md` | Active repository overview, supported install/launch entry point, current status links |
| `docs/preprocess_subsystem_spec_v1.md` | Active canonical Subsystem 01 functional specification |
| `docs/preprocess_v2_implementation_plan.md` | Active current V2 milestone plan with historical milestone context |
| `docs/preprocess_v2_6_geometry_modes_design.md` | Active design reference for future spatial geometry modes |
| `docs/ai_coding_guide.md` | Repository-wide development guidance |
| `docs/preprocess_implementation_plan_v1.md` | Historical/superseded implementation plan |
| `docs/preprocess_v1_field_test_audit.md` | Historical/release snapshot and field-test evidence |
| `docs/preprocess_v1_field_test_defect_triage.md` | Historical/release snapshot and corrective-patch analysis |
| `docs/preprocess_v1_release_readiness.md` | Historical/release snapshot and release-readiness checklist |
| `docs/preprocess_v2_requirements_draft.md` | Superseded draft requirements, largely overlapped by the V2 implementation plan |

New documents created by this milestone:

| Document | Intended classification |
| --- | --- |
| `docs/subsystem_01/status_and_roadmap.md` | Active current status and roadmap for Subsystem 01 |
| `docs/documentation_consolidation_plan.md` | Active consolidation plan until the documentation migration is completed |

---

## 2. Classification for each doc

### Active canonical specification

- `docs/preprocess_subsystem_spec_v1.md`

This remains the authoritative functional specification for Subsystem 01 scientific invariants, artifact contracts, timing rules, frame mapping, validation gates, and failure behavior.

### Active current roadmap/status

- `docs/subsystem_01/status_and_roadmap.md`
- `docs/preprocess_v2_implementation_plan.md`

The new status document should become the compact current-status source. The V2 implementation plan remains useful for milestone history and pending V2 design context until its active items are migrated or closed.

### Active design reference

- `docs/preprocess_v2_6_geometry_modes_design.md`

This is the active forward-looking reference for final spatial geometry modes and authoritative transform descriptions.

### Development guidance

- `docs/ai_coding_guide.md`

This is repository-wide development guidance. It is not a Subsystem 01 functional specification.

### Historical/release snapshot

- `docs/preprocess_v1_field_test_audit.md`
- `docs/preprocess_v1_field_test_defect_triage.md`
- `docs/preprocess_v1_release_readiness.md`

These preserve important evidence, triage decisions, and release-readiness context, but should not be treated as the live roadmap.

### Superseded plan

- `docs/preprocess_implementation_plan_v1.md`
- `docs/preprocess_v2_requirements_draft.md`

These are useful historical planning artifacts. Their durable requirements should be preserved in the canonical spec or status document before any future archival move.

### Redundant/overlapping

- `README.md` overlaps with install/status sections in active docs.
- `docs/preprocess_v2_implementation_plan.md` overlaps with `docs/preprocess_v2_requirements_draft.md`.
- The three v1 field/release docs overlap with each other on field-test evidence, corrective patches, and deferred V2 items.
- `docs/preprocess_subsystem_spec_v1.md` and `docs/preprocess_v2_6_geometry_modes_design.md` both discuss geometry; the v2.6 design clarifies that CropPlan is not necessarily the only future geometry authority.

---

## 3. Overlap/redundancy notes

The current documentation is fragmented because implementation plans, release snapshots, and forward-looking design notes accumulated during rapid Subsystem 01 development.

Main redundancy patterns:

- installation and launch instructions appear in README and runtime milestone notes;
- v1 implementation sequencing remains prominent even though implementation has moved past v1;
- V2 requirements draft and V2 implementation plan cover many of the same features;
- field-test audit, defect triage, and release readiness all describe related real-data evidence and corrective work;
- geometry concepts are split across the v1 spec and the v2.6 design document.

The consolidation should preserve evidence and decisions, not discard history.

---

## 4. Proposed active documentation hierarchy

Recommended future structure:

```text
README.md
    short overview
    supported installation/launch
    current status
    links to active docs

docs/
    subsystem_01/preprocessing.md
        canonical functional specification

    subsystem_01/status_and_roadmap.md
        implemented / validated / deferred / planned

    design/
        active forward-looking design documents

    development/
        repository-wide coding/development guidance

    archive/
        superseded plans, release snapshots, historical design notes
```

This milestone does not create those folders and does not move files.

---

## 5. Proposed archive candidates

Likely archive candidates after review:

- `docs/preprocess_implementation_plan_v1.md`
- `docs/preprocess_v1_field_test_audit.md`
- `docs/preprocess_v1_field_test_defect_triage.md`
- `docs/preprocess_v1_release_readiness.md`
- `docs/preprocess_v2_requirements_draft.md`

Conditional archive candidate:

- `docs/preprocess_v2_implementation_plan.md`, after remaining active roadmap content is migrated into `docs/subsystem_01/status_and_roadmap.md` or an active design document.

These files should not be moved until the user reviews this plan.

---

## 6. Documents that should remain canonical

In the proposed structure:

- `README.md` remains the repository entry point, not a full specification.
- A future `docs/subsystem_01/preprocessing.md` should become the single canonical Subsystem 01 functional spec, derived from `docs/preprocess_subsystem_spec_v1.md` plus accepted current clarifications.
- `docs/subsystem_01/status_and_roadmap.md` should remain the current Subsystem 01 status/roadmap document.
- `docs/preprocess_v2_6_geometry_modes_design.md` or its future location under `docs/design/` should remain the active geometry design reference until its decisions are folded into the canonical spec.
- `docs/ai_coding_guide.md` or its future location under `docs/development/` should remain repository-wide development guidance.

---

## 7. Migration plan with low-risk ordered steps

1. Review and approve this consolidation plan.
2. Review `docs/subsystem_01/status_and_roadmap.md` for factual accuracy.
3. Create a future `docs/subsystem_01/preprocessing.md` by consolidating:
   - `docs/preprocess_subsystem_spec_v1.md`;
   - accepted static-mask clarifications;
   - accepted geometry-mode transform rule from `docs/preprocess_v2_6_geometry_modes_design.md`;
   - current contiguous-trim and SLEAP handoff closure rules.
4. Update README links to point only to the active spec, status/roadmap, design, and development guidance.
5. Update `docs/ai_coding_guide.md` links after the active spec path is finalized.
6. Move approved historical documents into `docs/archive/` in a separate reviewed documentation-only change.
7. Move active design references into `docs/design/` only after links are updated.
8. Move development guidance into `docs/development/` only after links are updated.
9. Run `git diff --check` and a link/path review after each migration step.

---

## 8. Issue-log integration plan for later

The untracked local `docs/issue_logs/` folder should be reviewed later. Each issue should be classified as:

- resolved;
- current limitation;
- durable design decision;
- future roadmap item;
- reproducible open bug.

Only after that review should selected issue-log content be migrated into active docs, archive notes, or an issue tracker. The folder should not be treated as a current source of truth during this milestone.

---

## 9. No moves/deletions in this milestone

This milestone only creates an audit and proposed consolidation plan. No tracked document was moved, renamed, archived, or deleted.
