# Subsystem 07 — Sprint 4 GUI Integration and Laboratory Acceptance

**Status:** Implemented and **lab-validated**. Automated Sprint 4 suite green; operator laboratory GUI acceptance passed on `E_full_session_integration` after install/registry/responsiveness remediation and obsolete focal-directory cleanup (all nine interactive checks).  
**Date:** 2026-10-26  
**Authority:** Sprint 4 scope + S7-B contract + S5/S6 soft-navigation policy + operator laboratory checklist.  
**Depends on:** Sprints 1–3 (headless registry, engine, artifacts/QC/discovery).

See also: [Implementation Contract](implementation_contract.md), [Architecture Audit](architecture_audit.md), [MVP Scope](mvp_scope_and_roadmap.md), [Laboratory Acceptance Checklist](laboratory_acceptance_checklist.md).

---

## What landed

| Area | Change |
| --- | --- |
| Group 3 navigation | `MainWindow` adds **Group 3 — Behavioral Inference** with **S7 — Behavioral Inference**; remembers last Feature (S5/S6) and Behavior (S7) pages; S1–S6 unchanged |
| Page / controller | `BehavioralInferencePage` + `BehavioralInferenceController` |
| S6 source selection | Eligible S6 combo; default newest eligible; manual selection preserved on refresh; blocks stale/incomplete |
| Model registry | **Required** trusted local registry path; discovery without loading estimators |
| Detector selection | Unified dyadic + focal logical detectors; specialists nested under parents; unsupported/obsolete identified but not executable |
| Thresholds / parity | Effective thresholds + sources; user overrides; `legacy_parity` (default) vs `raw_framewise`; settings frozen in request |
| Readiness | S6 eligibility, two-mouse MVP, registry, deps, selected detector bundles, thresholds |
| Execution | `run_behavioral_inference_publication` via `GuiTaskRunner` / `GuiTaskKind.BEHAVIORAL_INFERENCE`; no Cancel |
| QC / discovery | Historical status vs current eligibility (incl. `stale_models`); prediction positives by detector; S8 handoff note |
| Tests | `tests/ui/test_behavioral_inference_gui.py` (20 cases) + S6 GUI regression green |
| Docs | This note; contract / README status updates |

---

## GUI workflow and controls

1. Open **Group 3 — Behavioral Inference** → **Subsystem 7 — Behavioral Inference**.
2. Browse/open a session (shared `MainWindow` session root with S1–S6).
3. Confirm the eligible S6 run (default newest; manual preserved).
4. Set the trusted model-registry root; review runtime deps and detector readiness.
5. Select any supported dyadic/focal subset (or Select all compatible).
6. Review thresholds / parity; optional overrides; Apply settings.
7. **Run Behavioral Inference** (non-cancellable; settings snapshotted).
8. Inspect QC / positives / invalid-route counts; reopen prior S7 runs without re-scoring.

Scientific formulas are not duplicated in GUI code; the page orchestrates `behavioral_inference.*`.

---

## Registry, detectors, and thresholds

| Policy | Behavior |
| --- | --- |
| Registry | Required for Run; absent → blocked (unlike optional S6 registry) |
| Discovery | `discover_model_registry` only (no `joblib.load` to populate lists) |
| Logical detectors | Dyadic: contact, nose_to_nose, side_by_side, mutually_disengaged; Focal: nose_to_body, nose_to_tail, approach, retract |
| Specialists | Shown under contact / nose_to_body; not separate selectable behaviors |
| Missing selected model | Detector stays selected and marked not-ready (does not disappear) |
| Threshold precedence | User override → inference config → fail |
| Parity | Default `legacy_parity`; `raw_framewise` disables nose-to-body rolling median |

---

## Readiness and execution

Before Run: eligible S6, exactly two tracks (MVP default), registry available, `[s7]` runtime deps, selected detectors loadable by discovery, thresholds resolved, ≥1 detector.

Execution uses an immutable `BehavioralInferenceRequest` + `InferenceRequestIdentity`. Soft navigation (Groups 1–3) remains enabled during S7; session/source/registry/detector mutations are page-locked. S2/S4 hard locks unchanged. Duplicate start rejected. Close waits for non-cancellable S7 work.

---

## QC, discovery, and S8 handoff

QC panel shows historical status, current eligibility, S6 source, registry/manifest, sample/frame counts, per-detector positives (dyadic pair-level vs focal directional), invalid-routing / non-finite warnings, thresholds/parity echo, artifact paths.

Completed-run table exposes eligible / completed_with_warnings / stale_source / stale_models / failed_or_incomplete / missing_required_artifacts / invalid_artifact. Reopening loads published QC/manifest without substituting new binaries or thresholds.

S8 handoff rows distinguish **S7 artifact eligibility** from whether a particular S8 workflow’s required detectors/views are present. Subsets are valid S7 runs. **S8 is not implemented.**

---

## Automated tests and static checks

| Command | Result |
| --- | --- |
| `pytest tests/ui/test_behavioral_inference_gui.py` | 20 passed |
| `pytest tests/behavioral_inference tests/ui/test_behavioral_inference_gui.py tests/ui/test_combined_features_gui.py` | 126 passed |
| `ruff check` on S7 GUI touched files | clean |
| `mypy` on new controller (non-PySide stubs) | clean for S7-specific typing |

Coverage includes Group 3 nav + S1–S6 regressions, S6 source preservation, ineligible S6, registry/detector/missing-model status, threshold override/reset, smoothing mode, readiness, immutable settings, duplicate-task prevention, soft nav during inference, progress/failure, QC, reopen, S8 view-availability note.

---

## Laboratory validation evidence

**Session:** `test_cases/E_full_session_integration`  
**S6 input:** `combined_features__20261025T173244` (classification `eligible`, 36528 samples, 2 tracks)  
**Registry:** `legacy/models` (active dyadic/focal bundles; obsolete conflicting focal directories removed by operator)  
**Final operator GUI run:** `behavioral_inference__20261026T163224` (`run_purpose=gui`)  
**Prior headless lab run (same science path; still eligible):** `behavioral_inference__20261026T150000` via `run_behavioral_inference_publication`  
**Config:** suite defaults — all active dyadic + focal detectors, `legacy_parity`

| Check | Result |
| --- | --- |
| Eligible S6 selection (controller) | Defaulted to eligible S6 above |
| Detector availability | contact, nose_to_nose, mutually_disengaged, side_by_side, nose_to_body, nose_to_tail, approach, retract — all ready |
| Thresholds / smoothing | Config thresholds; `legacy_parity` |
| Full inference | `success=True`, `status=completed`, 0 warnings |
| Prediction cardinalities | Dyadic logical detectors: **36528** rows each (1/frame); Focal: **73056** rows each (2 directions/frame) |
| Wide views | `dyadic_predictions_wide.parquet` 36528 rows; `focal_predictions_wide.parquet` 73056 rows |
| Invalid-route / non-finite | 0 / 0 |
| Published artifacts | predictions, specialist, wide views, qc, model_manifest, run_meta, settings_used |
| Reopen / eligibility | Final GUI run + prior headless run both currently `eligible`; S8 handoff note shown |
| Skipped detectors | None (full active set) |
| Operator nine-check GUI acceptance | **Passed 2026-10-26** — see [`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md) |

Positive counts (logical detectors only; specialists not counted as additional final detections):

| Detector | Grain | n_positive | fraction |
| --- | --- | --- | --- |
| contact | dyadic | 16424 | 0.450 |
| nose_to_nose | dyadic | 1747 | 0.048 |
| mutually_disengaged | dyadic | 12996 | 0.356 |
| side_by_side | dyadic | 803 | 0.022 |
| nose_to_body | focal | 9040 | 0.124 |
| nose_to_tail | focal | 3940 | 0.054 |
| approach | focal | 7550 | 0.103 |
| retract | focal | 6955 | 0.095 |

---

## Installation and registry (post–Sprint 4 remediation)

- Standard `scripts\install_windows_gui.bat` installs `.[s2,s7]` into `behavior_suite_gui` and validates xgboost/sklearn/joblib imports.
- Runtime installed ≠ models ready: configure a trusted registry root in the GUI; the path persists via QSettings (`BehaviorSuite` / `s7/last_model_registry_root`).
- Discovery / SHA-256 eligibility runs on a background `GuiTaskRunner` with digest caching; stale async results are rejected when the session changes mid-scan.
- Install hints use PowerShell-safe `'.[s2,s7]'` (not double-quoted `.[s7]`).

## Known limitations (non-MVP follow-ups)

- S8 arbitration / bout generation remain out of scope.
- Model binaries stay lab-local (not committed); fresh clones must configure a registry path once (persisted thereafter).
- Two-mouse MVP remains the default gate (`require_exactly_two_mice=true`).
- No cooperative cancellation; unsupported JSON-expert focal bundles stay non-executable.
- See [`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md) for the full non-MVP list.

---

## Exit criteria

Operator-capable Group 3 GUI ships; headless Sprint 3 runner is the sole long task; lab E-session full-detector inference published and reopenable; operator nine-check GUI acceptance passed; no S8 claims. **S7 MVP is closed.** Do not start S8 in this change set.
