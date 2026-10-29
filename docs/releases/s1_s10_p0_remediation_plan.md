# S1–S10 MVP — P0 Remediation Execution Plan

**Authority:** [`s1_s10_mvp_readiness.md`](s1_s10_mvp_readiness.md)  
**Date:** 2026-10-29  
**Scope:** Outstanding **P0** blockers only. No P1/P2 work, no scientific changes, no feature additions, no production-code edits until this plan is approved.  
**Stop condition:** Return the plan for approval before modifying application/runtime code. Documentation and operator checklist execution are the intended P0 surface.

**Execution status:** **P0 complete (2026-10-29)** — docs/operator checklist updates only; no scientific/GUI production-code changes; no model binaries or laboratory recordings added to Git. Final verification accepted; included in internal release **`v0.1.0`**.

---

## Summary

Four P0 items blocked a defensible internal release. None required algorithm or GUI-feature work.

| ID | Blocker | Class | Production code? | Status |
| --- | --- | --- | --- | --- |
| **P0-1** | Operator-facing documentation honesty | **Confirmed defect** (stale / conflicting docs) | No — docs only | **Resolved** |
| **P0-2** | Install prerequisite sheet | **Confirmed gap** (prereqs exist but are scattered / incomplete for operators) | No — docs only | **Resolved** |
| **P0-3** | S9 operator laboratory acceptance | **Unverified risk / process gate** — implementation present; interactive sign-off absent | No — operator execution + checklist updates | **Resolved (Pass)** |
| **P0-4** | S10 operator laboratory acceptance | **Unverified risk / process gate** — same | No — operator execution + checklist updates | **Resolved (Pass)** |

**Not P0 (explicitly excluded here):** missing one-click continues after S4→S10, doctor not checking `sleap-nn`, LFS training-video clone size, CI absence, Pipeline 50 numerical parity, fresh full-length S1→S10 recording.

---

## Recommended execution order

```text
P0-1  Documentation honesty          ✅
  ↓
P0-2  Install prerequisite sheet     ✅
  ↓
P0-3  S9 operator checklist          ✅ Pass (operator confirmed)
  ↓
P0-4  S10 operator checklist         ✅ Pass (operator confirmed)
  ↓
Release acceptance per readiness §10 — **accepted; `v0.1.0`**
```

**Dependencies:**

- **P0-2** may merge with **P0-1** in one documentation edit set.  
- **P0-3** depended on **P0-1** fixing S9 checklist architecture / item 1 (obsolete “no S10 page” criterion removed).  
- **P0-4** depended on an eligible S9 package (E-session `results_package__20261028T181714`).  
- **P0-3** and **P0-4** did not require production-code changes.

---

## P0-1 — Operator-facing documentation honesty

### Class

**Confirmed defect.** Multiple tracked docs contradicted `main` @ `985de69` (Group 4 S9+S10 exist in `src/ui/main_window.py`).

### Resolution evidence (2026-10-29)

| File | Change |
| --- | --- |
| `README.md` | Groups 1–4 / S1–S10; S9–S10 lab-validated; workflow + boundary updated; install-prereq link |
| `docs/README.md` | Status table + S9/S10 sections match signed acceptance |
| `docs/subsystem_08/mvp_scope_and_roadmap.md` | Group 4 delivered as S9+S10; known-limitation #1 retargeted |
| `docs/subsystem_08/sprint_4_gui_integration.md` | Removed “Group 4 not implemented / has not started” |
| `docs/subsystem_08/implementation_contract.md` | Follow-on / stop point updated |
| `docs/subsystem_08/laboratory_acceptance_checklist.md` | Architecture + notes: S9+S10 delivered |
| `docs/subsystem_09/laboratory_acceptance_checklist.md` | Architecture allows S9+S10; item 1 pass criteria no longer require absence of S10 |
| Relevant S9/S10 status docs | Sprint/contract/export GUI status aligned with Pass |

**Actual application structure recorded:**

- Group 1: S1–S4  
- Group 2: S5–S6  
- Group 3: S7–S8  
- Group 4: S9–S10  

No GUI navigation or production code changed.

### Verification

- Operator-facing paths no longer claim “Group 4 has not started” or that S10 must be absent from Group 4.  
- S9 checklist item 1 requires Group 4 to show **S9 and S10** side by side while keeping S9 packaging-only.

---

## P0-2 — Install prerequisite sheet

### Class

**Confirmed gap.** Installer/launch scripts work for the GUI env, but a fresh machine still needs external pieces that are easy to miss.

### Resolution evidence (2026-10-29)

- Created [`installation_prerequisites.md`](installation_prerequisites.md).  
- Linked from root `README.md` Supported Windows workflow.  
- Content verified against `environment-gui.yml`, `pyproject.toml` (`[s2]` / `[s7]`), `scripts/install_windows_gui.bat`, `scripts/launch_windows_gui.bat`, and S2 SLEAP resolution docs.  
- Corrected `environment-gui.yml` header comment to `.[s2,s7]` (comment only).  
- No installer behavior changes; no model binaries committed.

Covers: supported Windows/Conda env; Python deps; external `sleap-nn` 0.3.x; lab SLEAP inference models; S7 `models/` registry; historical `legacy/models` compatibility; FFmpeg; session inputs; writable outputs; optional Git LFS; verification/troubleshooting.

---

## P0-3 — S9 operator laboratory acceptance

### Class

**Unverified risk / process gate** → **Pass recorded**.

### Resolution evidence (2026-10-29)

Operator explicitly confirmed completion of the S9 laboratory acceptance checklist.

Recorded in `docs/subsystem_09/laboratory_acceptance_checklist.md`:

- Core packaging acceptance (items 1–10) — **Pass**  
- CSV/media export acceptance (E1–E7 and concise O1–O6) — **Pass**  
- Session / S8 / S9: `E_full_session_integration` / `behavioral_arbitration__20261028T133953` / `results_package__20261028T181714`  
- Notes: package bytes unchanged by export; no ethogram/video bout-edit UI on S9; **no biological ground-truth or numerical legacy-parity claim**

Supporting status updates: `docs/subsystem_09/sprint_4_gui_integration.md`, `media_csv_export_gui.md`, `implementation_contract.md`, `docs/README.md`.

---

## P0-4 — S10 operator laboratory acceptance

### Class

**Unverified risk / process gate** → **Pass recorded**.

### Resolution evidence (2026-10-29)

Operator explicitly confirmed completion of the S10 laboratory acceptance checklist.

Recorded in `docs/subsystem_10/laboratory_acceptance_checklist.md`:

- Synchronized inspection (sections A–C) — **Pass**  
- Controlled editing, undo/redo (section D) — **Pass**  
- Publication / immutability (section E, including E6 S8/S9 fingerprints unchanged) — **Pass**  
- Controlled synthetic edits are **not** genuine biological annotations  
- **No biological ground-truth or numerical legacy-parity claim**

Supporting status updates: `docs/subsystem_10/sprint_5_gui_integration.md`, `implementation_contract.md`, architecture audit post-audit note, `docs/README.md`.

---

## Confirmed defects vs unverified risks

| Item | Confirmed? | Evidence / outcome |
| --- | --- | --- |
| Stale root README / S8 “no Group 4” / S9 checklist “no S10” | **Yes — defect** | **Fixed** in P0-1 |
| `docs/README.md` “S9 lab-accepted” vs unsigned checklist | **Yes — defect** | **Fixed**; checklist now signed Pass |
| Missing consolidated install prerequisites | **Yes — gap** | **Fixed** via `installation_prerequisites.md` |
| S9 GUI/packaging/export broken | **No — unverified** → operator **Pass** | Checklist signed 2026-10-29 |
| S10 GUI/review/publish broken | **No — unverified** → operator **Pass** | Checklist signed 2026-10-29 |
| Pipeline cannot complete without extra continue buttons | **No — not a blocker** | Manual Group/Subsystem navigation + shared session suffice |

---

## Explicit non-goals (do not pull into this plan)

- Production Python/UI feature or science changes.  
- Enhancing `behavior-suite doctor` for `sleap-nn`/registry (P1).  
- Fresh full-length S1→S10 recording (P1 optional smoke).  
- LFS training-video cleanup / deleting `models/SLEAP/New folder/...` (P2).  
- CI, cooperative cancel, K>2, navigation conveniences after S4.

---

## Done criteria (P0 complete)

1. ✅ P0-1 docs updated and consistent on S9/S10 presence and validation wording.  
2. ✅ P0-2 prerequisite sheet visible from README (`docs/releases/installation_prerequisites.md`).  
3. ✅ P0-3 S9 checklist signed **Pass** (core + O1–O6).  
4. ✅ P0-4 S10 checklist signed **Pass** (critical C/D/E at minimum; A–E recorded).  
5. ✅ No production-code changes; no operator-check defect requiring a code fix.

---

## Remaining P1 / P2 (not blocking P0)

See readiness §8:

- **P1:** fresh short-clip S1→S10 smoke; LFS clone burden documentation/ops; doctor does not check `sleap-nn`/registry; no CI; Pipeline 50 / ethogram ground-truth parity unverified; historical `legacy/models` eligibility footgun.  
- **P2:** cooperative cancel; K>2; S8 overlay; S10 reviewed re-export / reciprocal / `uncertain`; LFS cleanup; richer doctor; auto S3→S4 navigation polish.

---

## Approval / stop gate

**P0 complete.** Final release verification (2026-10-29) found no new P0 blocker. Release documentation and annotated tag **`v0.1.0`** proceed under the user’s release authorization. No production-code changes were required for P0.
