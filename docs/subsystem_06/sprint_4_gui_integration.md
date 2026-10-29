# Subsystem 06 — Sprint 4 GUI Integration and Laboratory Acceptance Prep

**Status:** Implemented and **lab-validated**. Automated Sprint 4 suite green; operator laboratory GUI acceptance passed on `E_full_session_integration` (real S3 accept -> S4 -> jointly aligned S5 -> S6 GUI extraction).  
**Date:** 2026-10-25  
**Authority:** Approved Sprint 4 scope + S6-B contract + S5 navigation remediation policy + operator laboratory checklist.  
**Depends on:** Sprints 1-3 (headless backend).  

See also: [Implementation Contract](implementation_contract.md), [Architecture Audit](architecture_audit.md).

---

## What landed

| Area | Change |
| --- | --- |
| Group 2 navigation | `MainWindow` enables **S6 — Combined Features** beside S5; remembers last Feature Processing page (S5 vs S6) |
| Page / controller | `CombinedFeaturesPage` + `CombinedFeaturesController` |
| Source selection | Eligible S4 × S5 combos, default newest aligned pair, manual selection preserved |
| Alignment UI | Joint alignment status, sample count, reason codes, compatible-pair list |
| Extraction | `run_combined_feature_extraction` via `GuiTaskRunner.start_progressive` with immutable settings snapshot |
| Model registry | Optional path selector; dyadic/focal discovery; obsolete `contact` shown unsupported; no inference |
| QC / results | Historical status vs current eligibility; sources; samples/rows/tracks; implantation; compat summary |
| Run discovery | `summarize_combined_features_project` table; reopen without recalculation |
| Task safety | Soft nav lock matching S5; S2/S4 hard lock unchanged; no Cancel button |
| Tests | `tests/ui/test_combined_features_gui.py` (18 cases) + S5 nav regression updates |
| Docs | This note; contract / README / roadmap status updates |

---

## GUI workflow and controls

1. Open **Group 2 — Feature Processing** → **Subsystem 6 — Combined Features**.
2. Browse/open a session (shared `MainWindow` session root with S1–S5).
3. Inspect eligible S4 and S5 runs; confirm alignment readiness.
4. Optionally adjust `smooth_win` / `diff_dt` / labels and an optional model-registry root.
5. Review model-compat discovery (schema/availability only).
6. **Run Combined Feature Extraction** (non-cancellable).
7. Inspect QC + compatibility; reopen prior S6 runs from the discovery table.

Scientific formulas are not duplicated in GUI code; the page only orchestrates `combined_features.*`.

---

## Source-selection and alignment behavior

| Policy | Behavior |
| --- | --- |
| Default | Newest eligible S4 + newest eligible S5 **only if** `select_default_joint_pair` / joint alignment succeeds |
| Manual | User combo changes set `*_selection_manual`; refresh does **not** replace them |
| Missing upstream | Clear messaging when S4 and/or S5 eligible sets are empty |
| Incompatible pair | Shown with reason code; **Run** disabled; `build_request` rejects before extraction |
| Alternatives | Passing pairs listed from `list_eligible_joint_pairs` |
| Independence | Independent S4/S5 eligibility never implies joint alignment |

---

## Background execution and task safety

Applied the S5 remediation policy to S6:

**Allowed while S6 extraction runs**

- Switch Group 1 ↔ Group 2
- Open S1–S5 pages (S2/S4 must not themselves be running)
- Return to S6 and observe progress / final status

**Blocked while S6 task runs**

- Starting another S6 extraction
- Changing / browsing / clearing the S6 session
- Changing S4/S5 source selection or settings / registry
- Application close waits for the non-cancellable task

**Unchanged**

- S2/S4 hard-lock all navigation
- No Cancel button (runner has no cooperative cancel)

Returning to Feature Processing restores the last S5/S6 page via `_last_feature_page`.

---

## QC and model compatibility presentation

QC panel distinguishes:

- **Historical status** (immutable completion string)
- **Current eligibility** (live discovery classification; stale is never the default downstream pick)

Also shows S4/S5 sources, output path, 120 Hz sample count, directional rows, tracks, implantation resolution, structural failures, missingness / non-finite warnings, and a model-compat summary.

Model registry UI:

- Absent registry → `registry_unavailable`; extraction still allowed
- Present → discover dyadic/focal specs; obsolete `contact` → `unsupported`
- After a completed parquet exists → full `build_model_compat_report`
- Explicit copy: feature-schema compatibility ≠ artifact presence ≠ inference readiness
- Registry / model selection does not alter scientific feature calculations

---

## Completed-run discovery

Uses Sprint 3 APIs:

- `summarize_combined_features_project`
- Classifications: eligible / completed_with_warnings / stale_source / failed_or_incomplete / missing_required_artifacts
- Default inspection selection prefers newest **currently eligible** run; never prefers stale as downstream default
- Reopen loads `qc.json` / `model_compat.json` without recomputing features

---

## Automated tests and results

| Suite | Result |
| --- | --- |
| `tests/ui/test_combined_features_gui.py` | **18 passed** |
| S5 nav regressions (`test_group_navigation_*`, `test_main_window_exposes_s5_navigation`) | **passed** (S6 button now enabled) |
| `tests/combined_features/` | **passed** (headless suite unchanged in intent) |
| Ruff (Sprint 4 GUI files) | **passed** |
| Mypy (`combined_features_controller.py`) | **passed**; page/main_window PySide6 import-untyped warnings match existing UI modules |

Covered behaviors: navigation + session share, missing S4/S5, compatible/incompatible pairs, manual selection, readiness, full launch, immutable settings, progress/errors, duplicate prevention, safe nav during extraction, reopen, historical vs current eligibility, registry absent/present, dyadic/focal/obsolete display, unknown implantation warnings, S5/S6 coexistence.

---

## Laboratory / fixture validation evidence

Distinguish three evidence classes:

| Class | What it proves | Status |
| --- | --- | --- |
| Synthetic / unit tests | Contracts, formulas, GUI wiring, discovery | Automated suite green (Sprint 4) |
| Fixture inspection (pre-lab) | `E_full_session_integration` initially had eligible S5 only; no S4 | Historical blocker only |
| **Operator laboratory GUI acceptance** | Full checklist on a real session with eligible S4+S5 | **Passed 2026-10-25** |

### Operator-confirmed laboratory run (authoritative for MVP close)

Session: `C:\behavior_suite\test_cases\E_full_session_integration`  
Operator path: accept S3 in GUI -> run S4 -> open S6 -> extract (run_purpose=`gui`).

| Field | Recorded value |
| --- | --- |
| S6 run_id | `combined_features__20261025T173244` |
| S4 source | `pose_finalization__20261025T173206` |
| S5 source | `blob_features__20261021T152749` |
| Shared prepared_sync SHA-256 | `39f8d3b00663a1c9367303bc4154df698a82d283e995b8f51a1d838ecbdb044f` |
| Alignment | Eligible; sample_count **36528** @ 120 Hz; reason_code null |
| Status | `completed` (`created_at` 2026-10-25T17:32:44+02:00 -> `completed_at` 2026-10-25T17:33:26+02:00) |
| Tracks / nodes | tracks `[0, 1]`; 7 nodes including `headstage` |
| Output dimensions | **36528** samples x **2** directional rows = **73056** rows; **483** feature columns |
| Implantation | `resolved`, implanted_track_id `0` (unique finite headstage owner) |
| QC | `structural_ok=true`; warning_count **0**; failure_count **0** |
| Model compatibility | `registry_status=unavailable` (`model_registry.root` null; 0 detectors) — extraction allowed without registry |
| Current discovery | Classification **eligible**; `is_s6_run_eligible_for_downstream=True` |

Artifacts written under `combined_features/combined_features__20261025T173244/` include `dyadic_features_full.parquet`, `run_meta.json`, `qc.json`, `model_compat.json`, `settings_used.yaml`, and `processing_log.txt`. Laboratory video/pose/Parquet outputs remain local (not committed).

### Earlier fixture inspection (not acceptance)

Before the operator completed S3 acceptance and S4, the same session had eligible S5 (`blob_features__20261021T152749`) and no `pose_finalization/` tree. That inspection correctly **did not** claim laboratory acceptance.

---

## Known limitations

- No model inference or training UI
- No Cancel for long extraction
- Model registry is optional; detector bundles ship under tracked `models/` (SLEAP media remains gitignored)
- Working settings are not yet persisted under `combined_features/_gui/` (defaults + per-session widgets only)
- Numerical regression vs historical Pipeline 50 caches remains out of scope
- DataFrame fragmentation PerformanceWarnings from vendored FF/P50 ports (formulas unchanged)

---

## MVP acceptance

Laboratory MVP acceptance is **closed**. Operator checklist:
[`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md).
Final status: [`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).

