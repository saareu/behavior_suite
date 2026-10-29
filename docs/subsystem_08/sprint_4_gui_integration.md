# Subsystem 08 — Sprint 4 GUI Integration and Laboratory Acceptance

**Status:** Implemented and **lab-validated**. Automated suite green; headless E-session lab extraction verified (47/47 scientific consistency); GUI responsiveness remediation applied; **operator interactive GUI acceptance signed off 2026-10-28**.  
**Date:** 2026-10-28  
**Authority:** Sprint 4 scope + S8-B contract §9–§10 + S7 Group 3 soft-navigation policy + operator laboratory checklist.  
**Depends on:** Sprints 1–3 (headless Pipeline 100 → publish/discovery).

See also: [Implementation Contract](implementation_contract.md), [MVP Scope and Roadmap](mvp_scope_and_roadmap.md), [Architecture Audit](architecture_audit.md), [Laboratory Acceptance Checklist](laboratory_acceptance_checklist.md), [Scientific Output Audit (E-session)](scientific_output_audit_e_session.md).

---

## What landed

| Area | Change |
| --- | --- |
| Group 3 navigation | `MainWindow` adds **S8 — Behavioral Arbitration** beside S7; remembers `_last_behavior_page` (`s7`/`s8`); shared session; soft nav during S8; S2/S4 hard locks unchanged. Group 4 Results/Review is delivered separately as S9+S10. |
| Page / controller | `BehavioralArbitrationPage` + `BehavioralArbitrationController` |
| S7 source selection | Assessed S7 combo via `list_s7_options_for_s8` / `assess_s7_for_s8`; default newest **S8-ready**; manual selection preserved; exact linked S6 shown |
| Scientific config | Frozen suite YAML defaults; Stage 8 morph close vs Stage 10 gap fill distinguished; immutable request snapshot |
| Execution | Progressive `GuiTaskRunner`; coarse stage progress; no Cancel; nonblocking launch accept |
| Results / QC | Exclusive ethogram separate from parallel contact/SBS; engaged vs undefined; bout/QC/provenance panels |
| Discovery | Async discovery with generation/stale rejection; completed S8 history; historical status vs current eligibility |
| Responsiveness | Shared digest + validation memo; N+1 discovery fixed; Run launch off main thread; post-success discovery async; upstream S4/S5 fingerprints keep staleness correct under memoization |
| Tests | GUI + responsiveness + E-session consistency suites |
| Docs | This note; operator checklist; scientific output audit; MVP roadmap |

---

## GUI responsiveness remediation (post Sprint-4 operator lag)

### Root causes

1. **Session open / discovery lag**  
   - `list_s7_options_for_s8` re-summarized the entire S7 project **per candidate** (N+1).  
   - `select_eligible_s7_run(explicit path)` also rescanned the whole S7 project.  
   - Every S8 history classify nested `validate_s8 → validate_s7 → validate_s6 → validate_s5`, and S5 `_sha256` bypassed the shared digest cache (full re-hash of blob/video artifacts).  
   - Joint readiness loaded large Parquets for every S7 candidate on each discovery.

2. **Run Arbitration click lag**  
   - `build_request()` ran full `assess_s7_for_s8` (joint Parquet + digests) on the **Qt main thread** before starting the worker.  
   - `record_success()` called synchronous `refresh_discovery()` on the main thread after publish.

Digests and provenance checks were **not** removed. Remediation keeps them and avoids repeating unchanged work.

### Fixes

| Change | Effect |
| --- | --- |
| S5 `_sha256` → `project.file_digest.sha256_file` | Nested S5/S6/S7 validates reuse cached digests |
| `project.validation_memo` for S6/S7/S8 validate + S8 assess | Warm rediscovery skips identical nested walks; fingerprint includes upstream artifacts (including nested S4/S5) so digest mutations still invalidate |
| Single S7 project summarize in `list_s7_options_for_s8`; light explicit-run summary | Removes N+1 discovery |
| `begin_arbitration_launch` + `execute_arbitration_launch` | GUI accepts Run immediately; joint revalidation runs on the worker |
| Post-success discovery via `discovery_runner` | No sync freeze after publish |
| Stale discovery / arbitration identity checks | Newer session selection cannot be overwritten by older async results |

### Measured wall-clock (E_full_session_integration, same machine)

| Operation | Before | After |
| --- | --- | --- |
| `list_s7_options_for_s8` (cold) | ~56.5 s | ~2.1 s |
| `summarize_behavioral_arbitration_project` (cold, after S7 warm) | ~41.7 s | ~0.04 s |
| Discovery-like total (cold) | ~98 s | ~2.1 s |
| Discovery-like total (warm) | (same order) | ~0.33 s |
| `begin_arbitration_launch` (GUI-thread accept) | N/A (blocked in `build_request`) | **~0.3 ms** |
| Worker revalidate after warm memo | was on GUI thread (~seconds) | ~2.5 ms (warm) / still seconds cold first assess |

GUI blocking time for Run is now the cheap accept path; backend arbitration duration remains separate progressive work.

---

## GUI workflow and controls

1. Open **Group 3 — Behavioral Inference** → **Subsystem 8 — Behavioral Arbitration**.
2. Browse/open a session (shared `MainWindow` session root with S1–S7).
3. Confirm an **S8-ready** S7 run (default newest S8-ready; manual preserved on refresh).
4. Inspect linked S6 id/path/alignment and detector / feature readiness reasons.
5. Review scientific settings (Stage 8 vs Stage 10 clearly labeled); Apply settings.
6. **Run Behavioral Arbitration** — button disables immediately (“Launching…”); settings snapshotted; cancellation unsupported.
7. Soft-navigate to other groups during execution; return to S8 for progress/status.
8. Inspect ethogram, bout, and QC/provenance panels; reopen prior S8 runs without re-arbitrating.

Scientific formulas are not duplicated in GUI code; the page orchestrates `behavioral_arbitration.*`.

---

## Input selection and readiness

| Policy | Behavior |
| --- | --- |
| S7 eligibility ≠ S8 readiness | Full ethogram + linked S6 + alignment + K=2 via `validate_joint_s7_s6_for_s8` |
| Default selection | Newest **S8-ready** S7 |
| Manual selection | Preserved across discovery refresh when still listed |
| Linked S6 | Exact S7-recorded run only — never newest-S6 substitution |
| Ineligibility | Explicit reasons (missing detectors, S6 columns, lineage, K≠2, etc.) |

---

## Execution / progress / task safety

- Immediate launch accept via `begin_arbitration_launch`; worker revalidates then publishes.
- Immutable `BehavioralArbitrationRequest` + `ArbitrationRequestIdentity`.
- Coarse phase progress: intake (incl. revalidation) → arbitration → states → bouts → QC → publish (no fabricated fine percentages).
- Duplicate start rejected; source/settings mutations page-locked while busy.
- Soft group navigation remains enabled; S2/S4 hard locks unchanged.
- Close waits for non-cancellable S8 work; partial science never marked eligible.
- Post-success S7/S8 rediscovery is asynchronous; stale discovery snapshots are rejected.

---

## Automated tests and static checks

| Command | Result |
| --- | --- |
| `pytest tests/behavioral_arbitration tests/ui/test_behavioral_arbitration_gui.py tests/ui/test_behavioral_arbitration_responsiveness.py` | **117 passed** |
| Related S7 GUI + S6 regression (incl. stale-source) | Green after upstream-fingerprint memo fix |
| `ruff check` on remediation-touched files | **clean** |
| `mypy` on S8 packages / controller / `validation_memo` | pandas stub `import-untyped` on intake/runner is an environment/stubs gap (pre-existing when `pandas-stubs` absent); no new S8 logic type errors identified |

Coverage additions: async discovery + stale rejection, nonblocking Run accept, duplicate launch prevention, E-session framewise/bout/provenance consistency (`test_e_session_consistency.py`).

---

## Real-session extraction evidence

**Session:** `test_cases/E_full_session_integration`  
**S7:** `behavioral_inference__20261026T163224`  
**S6:** `combined_features__20261025T173244`  
**S8:** `behavioral_arbitration__20261028T133953` (`lab_validation`)  
**Sibling GUI run:** `behavioral_arbitration__20261028T155722` (same scientific counts)  
**Status:** `completed_with_warnings` — sole warning is intentional Stage 8 morph-close disable (`max_gap_to_fill=0`); **not** treated as harmless without inspection.  
**Downstream eligibility:** currently eligible; digests match live S7/S6 artifacts.

Counts: 36,528 synchronized samples; 73,056 directional frame-state rows; 609 bouts (538 exclusive, 60 contact, 11 side-by-side).

Full tabular audit (label proportions, bout distributions, transition windows, PASS/FAIL tally **47/47**): [`scientific_output_audit_e_session.md`](scientific_output_audit_e_session.md).

Arbitration formulas were **not** changed for this remediation or MVP closure. This evidence is algorithmic/schema consistency — **not** independent video ground-truth validation.

---

## Operator acceptance

See [`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md). Operator signed off the remediated GUI on 2026-10-28. S8 is **MVP complete / lab-validated**.

Retained architecture: S8 stays in **Group 3** beside S7. Group 4 Results/Review is delivered as Subsystems 09–10 (see subsystem 09/10 docs).

---

## Out of scope (unchanged)

Manual bout editing and video overlays **inside S8**, model inference/retraining, threshold optimization, and new arbitration rules. Packaging and interactive review live in Group 4 (S9/S10), not in S8.
