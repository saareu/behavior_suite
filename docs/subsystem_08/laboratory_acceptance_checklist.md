# Subsystem 08 — Laboratory GUI Acceptance Checklist

**Status:** **Passed — S8 lab-validated** (operator acceptance of the remediated GUI, 2026-10-28).  
**Authority:** Sprint 4 GUI integration + operator execution of this checklist on a real session.  
**Session:** `test_cases/E_full_session_integration` with S7 `behavioral_inference__20261026T163224`, linked S6 `combined_features__20261025T173244`, and S8 `behavioral_arbitration__20261028T133953`.

This checklist is distinct from automated unit/GUI tests and from the scientific audit of `behavioral_arbitration__20261028T133953` ([scientific_output_audit_e_session.md](scientific_output_audit_e_session.md)). Acceptance records interactive GUI usability plus the already-audited headless publication; it does **not** claim independent video ground-truth ethogram validation.

---

## Distinctions

| Layer | What it proves |
| --- | --- |
| Automated GUI/unit tests | Wiring, readiness gates, task safety, summary presentation, responsiveness regressions |
| Headless E-session publication + scientific audit | Pipeline 100 → publish correctness on real lab artifacts (**47/47**) |
| **This operator checklist** | Interactive GUI usability and laboratory acceptance |

---

## Architecture (retained)

- **Group 3 — Behavioral Inference:** S7 Behavioral Inference + S8 Behavioral Arbitration.
- **Group 4 — Results and Review:** delivered as Subsystem 09 (S9 packaging/export) and Subsystem 10 (S10 behavioral review); see subsystem 09/10 docs.

---

## Operator checklist (GUI) — ten interactive checks

| # | Check | Pass criteria | Result |
| --- | --- | --- | --- |
| 1 | Open S8 under Group 3 | Group 3 shows S7 and S8; S8 page loads without multi-minute freeze; discovery shows indeterminate progress | ☑ |
| 2 | Select an eligible / S8-ready S7 input | Newest S8-ready defaulted or manual pick preserved; reasons shown if not ready | ☑ |
| 3 | Inspect linked S6 readiness | Exact linked S6 id/path; alignment/readiness visible; no silent S6 substitution | ☑ |
| 4 | Review scientific configuration | Defaults load; Stage 8 morph close vs Stage 10 gap fill labeled; units visible | ☑ |
| 5 | Launch arbitration | Run button disables immediately (“Launching…”); progress/status update; no Cancel control; GUI remains responsive while validation/publish runs in background | ☑ |
| 6 | Navigate between groups during execution | Soft nav works; S8 progress restored on return; S2/S4 hard locks still hold | ☑ |
| 7 | Inspect state and bout summaries | Exclusive ethogram separate from parallel contact/SBS; engaged ≠ undefined; compare to audit counts if using the E-session | ☑ |
| 8 | Inspect QC / undefined / reconstruction | Structural QC, warnings (morph-close policy), provenance/digests visible; do not dismiss `completed_with_warnings` without reading warnings | ☑ |
| 9 | Reopen the run | Select completed run; summaries reload without rerunning; UI stays responsive | ☑ |
| 10 | Confirm current downstream eligibility | Historical status distinct from current eligibility classification | ☑ |

### Responsiveness notes for the operator

After remediation, cold discovery on the E-session completes in a few seconds (not ~1–2 minutes). Warm rediscovery feels near-instant. Clicking **Run Arbitration** returns control immediately with launching status; multi-minute backend arbitration time is expected and separate from GUI blocking.

---

## Sign-off

| Field | Value |
| --- | --- |
| Operator | Laboratory operator (final MVP closure approval) |
| Date | 2026-10-28 |
| Session / S7 / S6 / S8 run IDs | `E_full_session_integration` / `behavioral_inference__20261026T163224` / `combined_features__20261025T173244` / `behavioral_arbitration__20261028T133953` |
| Outcome | ☑ Pass — S8 lab-validated |
| Blockers / notes | Sole QC warning is intentional Stage 8 morph-close disable (`max_gap_to_fill=0`). Scientific audit 47/47. No independent video ground-truth claim. Group 4 delivered as S9+S10 (outside S8 scope). |
