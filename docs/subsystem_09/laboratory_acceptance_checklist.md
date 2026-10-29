# Subsystem 09 — Laboratory GUI Acceptance Checklist

**Status:** **Passed — S9 lab-validated** (operator acceptance of core packaging + concise CSV/media export, 2026-10-29).  
**Authority:** Sprint 4 GUI integration + operator execution of this checklist on a real session.  
**Session:** `test_cases/E_full_session_integration` with S8 `behavioral_arbitration__20261028T133953` and historical S9 `results_package__20261028T181714` (609 authoritative bouts; optional review 514/95; sync 36 528 rows).

This checklist is distinct from automated unit/GUI tests and from headless Sprint 1–3 publication evidence. Acceptance records interactive GUI usability for packaging and optional exports; it does **not** claim S10 review capability, biological ground-truth ethogram accuracy, or unperformed numerical legacy parity.

---

## Distinctions

| Layer | What it proves |
| --- | --- |
| Automated GUI/unit tests | Wiring, readiness gates, task safety, summary presentation, responsiveness regressions |
| Headless E-session publication (Sprints 1–3) + Sprint 4 probe | Packaging correctness and controller responsiveness on real lab artifacts |
| **This operator checklist** | Interactive GUI usability and laboratory acceptance |

---

## Architecture (retained)

- **Group 4 — Behavioral Results & Review:** S9 Results Packaging & Export and S10 Behavioral Review & Editing both present (separate pages).
- S9 is packaging-only: no ethogram timeline, video playback, or bout editing mixed into this page.

---

## Operator checklist (GUI)

| # | Check | Pass criteria | Result |
| --- | --- | --- | --- |
| 1 | Open Group 4 / S9 | Group 4 shows S9 and S10 side by side; S9 loads without multi-minute freeze; discovery shows indeterminate progress | ☑ |
| 2 | Select the E-session | Session path binds; shared session works when returning from Groups 1–3 | ☑ |
| 3 | Source readiness | Eligible S8 discovered; readiness / provenance / linked S7–S6 visible; latest S9-ready defaulted or manual pick preserved | ☑ |
| 4 | Review package / profile settings | Full package always all S8 bouts; optional `review_emit_v1` clearly additional; floors @ 120 Hz; hash video optional/expensive; lightweight ≠ video archive | ☑ |
| 5 | Run Packaging | Run disables immediately (“Launching…”); progress/status update; no Cancel; GUI remains responsive while validation/publish run in background | ☑ |
| 6 | Navigate away and back while processing | Soft nav to other groups works; S9 progress restored on return; S2/S4 hard locks still hold | ☑ |
| 7 | Inspect result summary | New unique run id; 609 authoritative bouts when packaging the audited S8; optional profile counts separate; sync rows 36 528 when applicable | ☑ |
| 8 | Inspect QC and dependencies | Structural / warnings / information separated; informational profile omissions not treated as failures; dependency notes if stale/missing | ☑ |
| 9 | Reopen a completed package | Select `results_package__20261028T181714` (or newest eligible); summaries/QC reload without reprocessing | ☑ |
| 10 | Confirm eligibility and S10 boundary | Historical status ≠ current eligibility; confirm no ethogram/video/bout-edit UI accidentally mixed into S9 | ☑ |

### Optional export extension (S9-E3) — operator

See also [media_csv_export_gui.md](media_csv_export_gui.md).

| # | Check | Pass criteria | Result |
| --- | --- | --- | --- |
| E1 | Eligible vs stale package selection | Export CSV / Bout Videos enable only for currently eligible S9 packages; stale packages show an explicit block (no silent substitute) | ☑ |
| E2 | Export CSV | Preview shows row count + automated provenance; launch accepts immediately; job appears under `results_exports/`; package bytes unchanged | ☑ |
| E3 | Export Bout Videos | Selection + context defaults 0; expected clip count shown; progress uses bout counters (no fabricated %); representative subset only for smoke | ☑ |
| E4 | Job history | Export jobs table lists job id, source package, type, historical status, current eligibility, counts, failures, output path | ☑ |
| E5 | Incomplete media retry | Retry incomplete/failed media jobs without overwriting successful clips; completed jobs immutable | ☑ |
| E6 | Task safety | Packaging and export cannot run concurrently; settings/source locked while busy; navigation remains soft-responsive; stale callbacks ignored | ☑ |
| E7 | Ordinary packaging | Core S9 packaging path still works when an eligible S8 source is available | ☑ |

**E-session note (S9-E4, 2026-10-29):** S7 model paths recorded as `legacy/models/...` were restored via a local directory junction to the validated artifacts under `models/` (digests match). Package `results_package__20261028T181714` is again **currently eligible** for new CSV/media exports.

### Concise export acceptance (operator — interactive)

Use eligible package `results_package__20261028T181714` on the E-session. Do **not** regenerate all 609 videos.

| # | Action | Pass criteria | Result |
| --- | --- | --- | --- |
| O1 | Select eligible S9 package | Package appears currently eligible; Export CSV / Bout Videos enabled | ☑ |
| O2 | Export full automated CSV | Job completes; CSV has **609** rows; provenance columns present; package bytes unchanged | ☑ |
| O3 | Export small MP4 batch | Select a few bouts / channels (not all 609); clips open in a player at source FPS (~119) | ☑ |
| O4 | Open exported files | CSV opens in spreadsheet/editor; MP4s play; clip names/ids match selected bouts | ☑ |
| O5 | Inspect job status / manifest / provenance | Jobs table shows completed status, source package id, output path; reopen job shows provenance | ☑ |
| O6 | Retry / error presentation | Completed media job refuses overwrite; incomplete/failed job offers retry; errors are explicit (no silent substitute) | ☑ |

### Responsiveness notes for the operator

Automated probe (2026-10-28): GUI-thread session prepare ~0.3 ms; cold discovery ~9 s background; warm rediscovery ~2 s background; Run Packaging accept &lt;1 ms; multi-second publish is expected background work and separate from GUI blocking.

---

## Sign-off

| Field | Value |
| --- | --- |
| Operator | Laboratory operator (explicit confirmation for P0 remediation) |
| Date | 2026-10-29 |
| Session / S8 / S9 run IDs | `E_full_session_integration` / `behavioral_arbitration__20261028T133953` / `results_package__20261028T181714` (+ export jobs under `results_exports/`) |
| Outcome | ☑ Pass — S9 lab-validated (core packaging + CSV/media export) |
| Blockers / notes | Package bytes unchanged by CSV/media export. No ethogram/video bout-edit UI on S9. No biological ground-truth or legacy numerical-parity claim. |
