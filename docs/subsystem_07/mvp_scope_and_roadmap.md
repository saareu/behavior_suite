# Subsystem 07 — MVP Scope, Completion Status, and Roadmap

**Status:** MVP complete / lab-validated (Sprints 1–4 + operator laboratory GUI acceptance). Automated tests green. Operator laboratory acceptance passed after install/registry/responsiveness remediation and focal obsolete-directory cleanup.  
**Date:** 2026-10-26  
**Authority:** S7-A audit + S7-B contract + Sprints 1–4 implementation + operator laboratory checklist on `E_full_session_integration`.

---

## Final subsystem architecture

```text
MainWindow
  Group 1 — Data Preparation → S1–S4
  Group 2 — Feature Processing → S5 + S6
  Group 3 — Behavioral Inference → S7
       BehavioralInferencePage
            BehavioralInferenceController
            GuiTaskRunner (non-cancellable inference)
            behavioral_inference.* (headless scientific core)
```

| Layer | Modules |
| --- | --- |
| Registry / load / thresholds | `registry`, `loader`, `config`, `validate` |
| Features / adapters | `features`, `adapters/dyadic`, `adapters/focal` |
| Engine / predict | `engine`, `predict` |
| Export / QC / discovery | `export`, `qc`, `results`, `run_discovery`, `runner` |
| GUI persistence | `gui_persistence` (QSettings registry path) |
| UI | `BehavioralInferenceController`, `BehavioralInferencePage`, `MainWindow` Group 3 |

S7 consumes eligible S6 `dyadic_features_full.parquet` and trusted local model bundles. It does **not** train models, arbitrate detectors, or generate ethogram bouts (Subsystem 08).

---

## Navigation behavior

- Group 3 exposes S7 only; last Behavior page is remembered.
- Session root is shared with S1–S6.
- **S2 / S4 tasks** hard-block all navigation.
- **S7 inference** soft-locks like S5/S6: Groups 1–3 remain reachable; duplicate Run and session/registry/detector mutation stay blocked; no Cancel.
- Default S6 source is newest currently eligible; manual eligible selection is preserved on refresh.

---

## User workflow

1. Open Group 3 — Behavioral Inference → Subsystem 7 — Behavioral Inference.
2. Open a session with a currently eligible S6 run (exactly two tracks under MVP default).
3. Configure the trusted model-registry root (persisted across launches); confirm runtime deps.
4. Select any supported dyadic/focal detector subset; review thresholds / `legacy_parity`.
5. **Run Behavioral Inference** (immutable settings snapshot).
6. Inspect QC + positives; reopen prior S7 runs from discovery.

Scientific formulas are not duplicated in GUI code.

---

## Input / readiness

| Requirement | Rule |
| --- | --- |
| Eligible S6 | Completed `dyadic_features_full.parquet`; currently eligible for downstream |
| Two-mouse MVP | `require_exactly_two_mice=true` by default |
| Model registry | **Required** trusted local root; discovery without loading estimators for lists |
| Runtime deps | `[s7]` extras (`xgboost`, `scikit-learn`, `joblib`) in `behavior_suite_gui` |
| Thresholds | Explicit config (or user override) for every selected detector/stage |
| Detectors | ≥1 selected; all required active bundles discoverable and loadable |

---

## Output and provenance contracts

```text
<session_root>/behavioral_inference/<run_id>/
  predictions.parquet
  specialist_predictions.parquet
  dyadic_predictions_wide.parquet
  focal_predictions_wide.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
  qc.json
  model_manifest.json
```

Statuses: `running` → `completed` | `completed_with_warnings` | `failed`.

Discovery classifications: `eligible`, `completed_with_warnings`, `stale_source`, `stale_models`, `failed_or_incomplete`, `missing_required_artifacts`, `invalid_artifact`. Digest-authoritative model checks: missing or changed `model.joblib` / feature-spec digests → `stale_models`; identical digests after registry relocation remain eligible.

Downstream default prefers newest currently eligible run; never prefers stale.

---

## Laboratory acceptance evidence

Operator-confirmed GUI acceptance on 2026-10-26 (post-remediation; not synthetic):

| Field | Recorded value |
| --- | --- |
| Session | `test_cases/E_full_session_integration` |
| Final GUI run | `behavioral_inference__20261026T163224` (`run_purpose=gui`) |
| Prior headless lab run | `behavioral_inference__20261026T150000` (same science path; still eligible) |
| S6 source | `combined_features__20261025T173244` |
| Registry | `legacy/models` (obsolete conflicting focal directories removed by operator) |
| Frames / prediction rows | 36528 / 438336 |
| QC | structural_ok; 0 warnings; 0 failures; 0 invalid routes |
| Discovery | currently `eligible` for both completed runs |

Full checklist: [`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md). Sprint 4 detail: [`sprint_4_gui_integration.md`](sprint_4_gui_integration.md).

---

## Known limitations (non-MVP follow-ups)

1. No Subsystem 08 arbitration, reciprocity, hard-gap gates, or bout stitching.
2. No cooperative cancellation for inference.
3. Model binaries stay lab-local; suite does not ship `legacy/models`.
4. Two-mouse gate remains the MVP default (`require_exactly_two_mice=true`).
5. Unsupported JSON-expert focal bundles (`nose_body` / `nose_tail`) stay non-executable.
6. Optional continue-on-error across detectors is not the MVP default (fail-fast).
7. Working non-registry settings are not yet persisted under `behavioral_inference/_gui/`.
8. Product default for new non-parity runs may later prefer `raw_framewise`; MVP acceptance used `legacy_parity`.

---

## Automated test results (MVP closure)

| Command | Result |
| --- | --- |
| `pytest tests/behavioral_inference tests/ui/test_behavioral_inference_gui.py tests/ui/test_combined_features_gui.py tests/test_windows_gui_runtime_scripts.py` | **144 passed, 1 skipped** (optional `nose_to_body` fallback absent after lab cleanup) |
| `ruff check` on S7 package / touched UI / scripts / tests | **All checks passed** |
| `mypy -p behavioral_inference` + controller / file_digest / S6 handoff modules | **Clean** |

---

## Sprint history

| Sprint | Deliverable | Status |
| --- | --- | --- |
| S7-A / audit | Architecture audit | Accepted |
| S7-B | Implementation contract | Binding; MVP closed |
| Sprint 1 | Model registry and runtime foundation | Done |
| Sprint 2 | Unified dyadic and focal inference | Done |
| Sprint 3 | Prediction artifacts, QC, run discovery | Done |
| Sprint 4 | GUI integration + lab acceptance | Done / lab-validated |
| Post–Sprint 4 remediation | Installer `[s7]`, registry persistence, async discovery, PowerShell-safe hints | Done / operator-accepted |

---

## Design references

- [Architecture Audit (S7-A)](architecture_audit.md)
- [Implementation Contract (S7-B)](implementation_contract.md)
- [Sprint 1](sprint_1_model_runtime.md) · [Sprint 2](sprint_2_inference_engine.md) · [Sprint 3](sprint_3_artifacts_qc_discovery.md) · [Sprint 4](sprint_4_gui_integration.md)
- [Laboratory Acceptance Checklist](laboratory_acceptance_checklist.md)
