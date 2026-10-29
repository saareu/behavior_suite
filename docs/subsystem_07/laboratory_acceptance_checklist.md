# Subsystem 07 — Laboratory Acceptance Checklist

**Status:** **Completed** — operator laboratory GUI acceptance passed.  
**Date:** 2026-10-26  
**Authority:** Sprint 4 GUI + post-Sprint 4 install/registry/responsiveness remediation + operator execution of this checklist on a real session.

This record is distinct from synthetic/unit tests and from the earlier Sprint 4 headless E-session publication (`behavioral_inference__20261026T150000`).

---

## Preflight — session used

| Check | Result |
| --- | --- |
| Session | `C:\behavior_suite\test_cases\E_full_session_integration` |
| Eligible S6 | `combined_features__20261025T173244` (36528 samples, 2 tracks) |
| Registry | `C:\behavior_suite\models` |
| Focal conflict resolution | Operator removed obsolete conflicting focal model directories; active specialists remain (`nose_body_intact` / `nose_body_occ`, `nose_to_tail`, `approach`, `retract`) |
| Runtime | `behavior_suite_gui` with `.[s2,s7]` via `scripts\install_windows_gui.bat` |

---

## Operator checklist (GUI) — nine interactive checks

| Step | Outcome |
| --- | --- |
| 1. Open Group 3 / Subsystem 7 on the eligible session | Pass |
| 2. Confirm eligible S6 source (default newest; no stale) | Pass — `combined_features__20261025T173244` |
| 3. Configure trusted registry; confirm runtime deps ready | Pass — registry persisted; xgboost/sklearn/joblib present |
| 4. Select supported dyadic + focal detectors (full active set) | Pass — contact, nose_to_nose, mutually_disengaged, side_by_side, nose_to_body, nose_to_tail, approach, retract |
| 5. Review thresholds / `legacy_parity` readiness | Pass — config thresholds; ready to run |
| 6. Run Behavioral Inference from the GUI | Pass — `behavioral_inference__20261026T163224`, `run_purpose=gui` |
| 7. Inspect QC / positives / invalid-route counts | Pass — structural_ok; 0 warnings/failures; 0 invalid |
| 8. Reopen completed run; confirm current eligibility | Pass — discovery classification `eligible` |
| 9. Confirm soft navigation / S1–S6 remain functional | Pass — Groups 1–3 soft-nav during/after S7; S2/S4 hard locks unchanged |

---

## Recorded run evidence (from artifacts)

| Field | Value |
| --- | --- |
| Final acceptance S7 run_id | `behavioral_inference__20261026T163224` |
| Status | `completed` |
| Created / completed | 2026-10-26T16:32:24+02:00 / 2026-10-26T16:32:34+02:00 |
| Prediction rows / frames | 438336 / 36528 |
| Wide views | dyadic 36528 rows; focal 73056 rows |
| Invalid-route / non-finite | 0 / 0 |
| Prior headless lab run (still eligible) | `behavioral_inference__20261026T150000` |

Positive counts (logical detectors; from final GUI QC `detector_positive_stats`):

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

Do not invent additional metrics beyond these recorded artifacts.

---

## Historical run-discovery policy (verified at closure)

| Case | Expected classification | Verified |
| --- | --- | --- |
| Recorded `model.joblib` digest unchanged | remains `eligible` | Yes — both lab runs |
| Recorded `model.joblib` deleted or missing | `stale_models` | Yes — live probe + unit test |
| Recorded digest changed (tamper / substitute) | `stale_models` | Yes — unit test |
| Same digest after registry relocation | not stale (digest-authoritative) | Yes — unit test |

---

## Suggested registry path (this workstation)

`C:\behavior_suite\models`  
(Required for Run; detector bundles are tracked under `models/`. Operators may still browse to an alternate registry; the path persists via QSettings.)
