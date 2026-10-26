# Subsystem 06 - MVP Scope, Completion Status, and Roadmap

**Status:** MVP complete / lab-validated (Sprints 1-4 + laboratory GUI acceptance). Automated tests green. Operator laboratory acceptance passed.
**Date:** 2026-10-25
**Authority:** S6-A audit + S6-B contract + Sprints 1-4 implementation + operator laboratory checklist on `E_full_session_integration`.

---

## Final subsystem architecture

```text
MainWindow
  Group 1 - Data Preparation -> S1-S4
  Group 2 - Feature Processing -> S5 + S6
       CombinedFeaturesPage
            CombinedFeaturesController
            GuiTaskRunner (progressive extraction; non-cancellable)
            combined_features.* (headless scientific core)
```

| Layer | Modules |
| --- | --- |
| Upstream joint eligibility | `upstream` (S4+S5 pair validation, 120 Hz alignment) |
| Pose adapter | `pose_adapter` (S4 long-form -> FF tensors / quality) |
| Feature computation | `frame_features`, `pipeline50_post`, `processing` (vendored formulas) |
| Export / QC / compat | `export`, `qc`, `model_compat` |
| Runner / discovery | `runner`, `run_discovery` |
| Config / contracts | `config`, `contracts` |
| UI | `CombinedFeaturesController`, `CombinedFeaturesPage`, `MainWindow` Group 2 |

S4 `final_pose.parquet` and S5 `blob_features.parquet` are read-only inputs. S6 does **not** train models or run inference.

---

## Navigation behavior

- Group 2 exposes S5 and S6; last Feature Processing page is remembered.
- Session root is shared with S1-S5.
- **S2 / S4 tasks** hard-block all navigation.
- **S6 extraction** soft-locks like S5: Group 1 <-> Group 2 and other pages remain reachable; duplicate Run and session mutation stay blocked; no Cancel.
- Default S4/S5 pair only when joint alignment succeeds; manual selections are preserved on refresh.

---

## User workflow

1. Open Group 2 - Feature Processing -> Subsystem 6 - Combined Features.
2. Open a session with currently eligible S4 and S5 runs that jointly align.
3. Confirm alignment (sample count @ 120 Hz); optionally set model-registry root.
4. **Run Combined Feature Extraction** (immutable settings snapshot).
5. Inspect QC + model-compat summary; reopen prior S6 runs from discovery.

Scientific formulas are not duplicated in GUI code.

---

## Input / readiness

| Requirement | Rule |
| --- | --- |
| Eligible S4 | Completed `final_pose.parquet`; currently eligible for downstream |
| Eligible S5 | Completed `blob_features.parquet`; currently eligible for downstream |
| Joint alignment | Same session, shared prepared_sync identity, matching 120 Hz sample count / time grid |
| Model registry | Optional; absence yields `registry_unavailable` and does not block extraction |

Independent S4/S5 eligibility never implies joint alignment.

---

## Output and provenance contracts

```text
<session_root>/combined_features/<run_id>/
  dyadic_features_full.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
  qc.json
  model_compat.json
```

Statuses: `running` -> `completed` | `completed_with_warnings` | `failed`.

Discovery classifications: `eligible`, `completed_with_warnings`, `stale_source`, `failed_or_incomplete`, `missing_required_artifacts`. Downstream default prefers newest currently eligible run; never prefers stale.

---

## Laboratory acceptance evidence

Operator-confirmed GUI acceptance on 2026-10-25 (not synthetic; not the earlier S5-only fixture inspection):

| Field | Recorded value |
| --- | --- |
| Session | `test_cases/E_full_session_integration` |
| S6 run | `combined_features__20261025T173244` |
| S4 / S5 | `pose_finalization__20261025T173206` / `blob_features__20261021T152749` |
| Samples / rows / columns | 36528 / 73056 / 483 |
| QC | structural_ok; 0 warnings; 0 failures |
| Model compat | registry unavailable (root null) |
| Discovery | currently `eligible` |

Full checklist: [`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md). Sprint 4 detail: [`sprint_4_gui_integration.md`](sprint_4_gui_integration.md).

---

## Known limitations (non-MVP follow-ups)

1. No model inference or training UI (intentional MVP boundary).
2. No cooperative cancellation for extraction.
3. Optional external model registry; suite does not ship `legacy/models`.
4. Working settings not yet persisted under `combined_features/_gui/`.
5. No numerical regression against historical Pipeline 50 training caches.
6. Vendored FF/P50 ports emit DataFrame fragmentation PerformanceWarnings (formulas unchanged).
7. Subsystem 07 (behavior classification / inference) is MVP complete / lab-validated; S8 arbitration remains downstream.

---

## Automated test results (MVP closure)

| Command | Result |
| --- | --- |
| `pytest tests/combined_features tests/ui/test_combined_features_gui.py tests/ui/test_blob_features_gui.py` | **97 passed** |
| `ruff check` on S6 / touched UI / S4 discovery modules | **All checks passed** |
| `mypy -p combined_features` and controller | **Clean** |

---

## Sprint history

| Sprint | Deliverable | Status |
| --- | --- | --- |
| S6-A / audit | Architecture audit | Accepted |
| S6-B | Implementation contract | Binding; MVP closed |
| Sprint 1 | Input validation + pose adapter | Done |
| Sprint 2 | Feature calculation migration | Done |
| Sprint 3 | Export, QC, model_compat, discovery, runner | Done |
| Sprint 4 | GUI integration + lab acceptance | Done / lab-validated |

---

## Design references

- [Architecture Audit (S6-A)](architecture_audit.md)
- [Implementation Contract (S6-B)](implementation_contract.md)
- [Sprint 1](sprint_1_input_adapter.md) · [Sprint 2](sprint_2_feature_migration.md) · [Sprint 3](sprint_3_export_compatibility.md) · [Sprint 4](sprint_4_gui_integration.md)
