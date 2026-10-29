# Subsystem 08 — MVP Scope, Completion Status, and Roadmap

**Status:** MVP complete / lab-validated (Sprints 1–4 + scientific E-session audit + GUI responsiveness remediation + operator laboratory GUI acceptance).  
**Date:** 2026-10-28  
**Authority:** S8-A audit + S8-B contract + Sprints 1–4 implementation + scientific output audit on `behavioral_arbitration__20261028T133953` + operator laboratory checklist on `E_full_session_integration`.

---

## Final subsystem architecture

```text
MainWindow
  Group 1 — Data Preparation → S1–S4
  Group 2 — Feature Processing → S5 + S6
  Group 3 — Behavioral Inference → S7 + S8
       BehavioralInferencePage (S7)
       BehavioralArbitrationPage (S8)
            BehavioralArbitrationController
            GuiTaskRunner (non-cancellable arbitration / discovery)
            behavioral_arbitration.* (headless scientific core)
```

| Layer | Modules |
| --- | --- |
| Config / contracts / intake | `config`, `contracts`, `intake`, `expand`, `validity` |
| Gates / hierarchy / adaptive / contact | `gates`, `hierarchy`, `adaptive`, `contact_clean` |
| Temporal / finalize / bouts | `temporal`, `finalize`, `bouts` |
| Export / QC / discovery | `export`, `qc`, `results`, `run_discovery`, `runner` |
| UI | `BehavioralArbitrationController`, `BehavioralArbitrationPage`, `MainWindow` Group 3 |

S8 consumes eligible S7 predictions plus the **exact** linked S6 feature columns. It does **not** train models, re-run S7 classifiers, remultiplex contact specialists, overwrite S7 artifacts, package review clips, or host manual bout correction. Group 4 Results/Review is delivered as Subsystems 09–10 (see `docs/subsystem_09/` and `docs/subsystem_10/`).

---

## Navigation behavior

- Group 3 exposes **S7 and S8**; last Behavior page (`s7`/`s8`) is remembered.
- Session root is shared with S1–S7.
- **S2 / S4 tasks** hard-block all navigation.
- **S8 arbitration** soft-locks like S5–S7: Groups 1–3 remain reachable; duplicate Run and session/source/settings mutation stay blocked; no Cancel.
- Default S7 source is newest **S8-ready** (full ethogram + linked eligible S6); manual selection is preserved on refresh.

---

## User workflow

1. Open Group 3 — Behavioral Inference → Subsystem 8 — Behavioral Arbitration.
2. Open a session with a currently S8-ready S7 run and its exact linked S6.
3. Review scientific settings (Stage 8 morph close vs Stage 10 gap fill labeled); Apply settings.
4. **Run Behavioral Arbitration** (immutable settings snapshot; launch accepted immediately).
5. Inspect exclusive ethogram vs parallel contact/SBS, bout summaries, QC/provenance; reopen prior S8 runs.

Scientific formulas are not duplicated in GUI code.

---

## Input / readiness

| Requirement | Rule |
| --- | --- |
| Eligible S7 | Completed(+warnings); currently eligible for downstream; full ethogram detector coverage |
| Linked S6 | Exact S7-recorded run only — never newest-S6 substitution |
| Two-mouse MVP | K=2 directional expansion |
| Joint alignment | `dense_frame_idx` / lineage / required hard-gap + optical columns |
| Validity | Routing sentinels never treated as confident negatives |

---

## Output and provenance contracts

```text
<session_root>/behavioral_arbitration/<run_id>/
  behavior_states.parquet
  behavior_bouts.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
  qc.json
  # optional: arbitration_diagnostics.parquet
```

Statuses: `running` → `completed` | `completed_with_warnings` | `failed`.

Discovery classifications: `eligible`, `completed_with_warnings`, `stale_source`, `failed_or_incomplete`, `missing_required_artifacts`, `invalid_artifact`. Downstream default prefers newest currently eligible run; never prefers stale.

---

## Laboratory acceptance evidence

Operator-confirmed GUI acceptance on 2026-10-28 (post responsiveness remediation; not synthetic):

| Field | Recorded value |
| --- | --- |
| Session | `test_cases/E_full_session_integration` |
| S7 source | `behavioral_inference__20261026T163224` |
| Linked S6 | `combined_features__20261025T173244` |
| S8 lab-validation run | `behavioral_arbitration__20261028T133953` (`run_purpose=lab_validation`) |
| Sibling GUI run | `behavioral_arbitration__20261028T155722` (same scientific counts) |
| Synchronized samples / directional rows | 36,528 / 73,056 |
| Bouts | 609 total (538 exclusive, 60 contact, 11 side-by-side) |
| Scientific consistency audit | **47/47 PASS** ([scientific_output_audit_e_session.md](scientific_output_audit_e_session.md)) |
| QC | `structural_ok`; 1 warning (intentional Stage 8 morph-close disable, `max_gap_to_fill=0`); 0 failures; currently downstream-eligible |
| Responsiveness | Cold discovery ~2.1 s (was ~98 s); Run GUI accept ~0.3 ms; digests/provenance retained via shared cache + validation memo |

Full checklist: [`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md). Sprint 4 detail: [`sprint_4_gui_integration.md`](sprint_4_gui_integration.md).

This acceptance records operator interactive GUI sign-off and algorithmic/schema consistency on the published E-session artifacts. It does **not** claim independent video ground-truth validation of ethogram accuracy.

### Documented scientific deviations from legacy Pipeline 100

| Topic | MVP stance |
| --- | --- |
| Invalid / routing-sentinel evidence | Masked as unobserved; never treated as confident negatives |
| `undefined` label | Published only when evidence cannot support the legacy residual; valid residual remains `engaged` |
| Contact remultiplex | Removed — trust S7 fused `prob_contact` / `pred_contact`; retain optical/Schmitt/morph/min-bout cleaning |
| Stage 8 morph close | `max_gap_to_fill=0` disables morphological closing (YAML-0 policy); Stage 10 gap fill remains independent |

Arbitration formulas and numeric thresholds were **not** changed for MVP closure.

---

## Known limitations (non-MVP follow-ups)

1. Group 4 packaging/review is out of S8 scope (delivered as S9+S10; see subsystem 09/10 docs).
2. No cooperative cancellation for arbitration.
3. No overlay video or manual bout editing inside S8.
4. Two-mouse (K=2) gate remains the MVP default; K>2 out of scope.
5. Reduced / partial-detector ethogram profile remains unresolved (full coverage hard-fail only).
6. Optional long-form `arbitration_diagnostics.parquet` defaults off.
7. No independent video ground-truth ethogram validation in this MVP closure.
8. Canonical-only intake (drop wide-view dependency) deferred; dual intake remains.

---

## Automated test results (MVP closure)

| Command | Result |
| --- | --- |
| `pytest tests/behavioral_arbitration tests/ui/test_behavioral_arbitration_gui.py tests/ui/test_behavioral_arbitration_responsiveness.py` | **117 passed** |
| Related S7/S6 + GUI regression suites | See final verification report |
| `ruff check` on S8 package / touched UI / runners / tests | **All checks passed** |
| `mypy` on S8 package / controller / validation_memo / related runners | See final verification report (pandas stub gaps are environment/stubs, not new S8 logic failures) |

---

## Sprint history

| Sprint | Deliverable | Status |
| --- | --- | --- |
| S8-A / audit | Architecture audit | Accepted |
| S8-B | Implementation contract | Binding; MVP closed |
| Sprint 1 | Intake, expansion, validity, gates, vetoes | Done |
| Sprint 2 | Hierarchy, adaptive, contact cleaning | Done |
| Sprint 3 | Temporal rules, bouts, QC, publish, discovery | Done |
| Sprint 4 | Group 3 GUI + lab extraction | Done |
| Post–Sprint 4 remediation | Discovery/run responsiveness; shared digests + validation memo | Done / operator-accepted |
| Scientific audit | E-session 47/47 consistency | Done |
| Operator GUI acceptance | Laboratory checklist | Done / lab-validated |

---

## Design references

- [Architecture Audit (S8-A)](architecture_audit.md)
- [Implementation Contract (S8-B)](implementation_contract.md)
- [Sprint 1](sprint_1_intake_gates.md) · [Sprint 2](sprint_2_arbitration_contact.md) · [Sprint 3](sprint_3_temporal_bouts_publication.md) · [Sprint 4](sprint_4_gui_integration.md)
- [Laboratory Acceptance Checklist](laboratory_acceptance_checklist.md)
- [Scientific Output Audit (E-session)](scientific_output_audit_e_session.md)
