# Subsystem 06 - Laboratory Acceptance Checklist

**Status:** **Completed** — operator laboratory GUI acceptance passed.  
**Date:** 2026-10-25  
**Authority:** Sprint 4 GUI + operator execution of this checklist on a real session.

This record is distinct from synthetic/unit tests and from the earlier fixture
inspection that found eligible S5 without eligible S4.

---

## Preflight - session used

| Check | Result |
| --- | --- |
| Session | `C:\behavior_suite\test_cases\E_full_session_integration` |
| Eligible S4 | `pose_finalization__20261025T173206` (after operator S3 accept + S4 run) |
| Eligible S5 | `blob_features__20261021T152749` |
| Shared S1 sync | prepared_sync SHA-256 `39f8d3b00663a1c9367303bc4154df698a82d283e995b8f51a1d838ecbdb044f` |
| Joint timeline | Aligned; **36528** samples @ 120 Hz |

---

## Operator checklist (GUI) — outcomes

| Step | Outcome |
| --- | --- |
| 1. Open eligible real session | Pass — Group 2 / Subsystem 6 |
| 2. Select jointly compatible S4 and S5 | Pass — S4 `pose_finalization__20261025T173206` + S5 `blob_features__20261021T152749` |
| 3. Inspect successful alignment | Pass — eligible; sample_count 36528; reason_code null |
| 4. Run S6 extraction | Pass — `combined_features__20261025T173244`, run_purpose=`gui` |
| 5. Inspect output row counts / identities | Pass — 73056 rows = 2 x 36528; tracks `[0, 1]` |
| 6. Inspect QC and model compatibility | Pass — structural_ok; 0 warnings/failures; registry unavailable (allowed) |
| 7. Reopen completed run | Pass — discovery classification `eligible` |
| 8. Confirm downstream eligibility | Pass — `is_s6_run_eligible_for_downstream=True` |
| 9. Verify S1–S5 navigation remains functional | Pass — soft S6 nav; S2/S4 hard locks unchanged |

---

## Recorded run evidence (from artifacts)

| Field | Value |
| --- | --- |
| S6 run_id | `combined_features__20261025T173244` |
| Status | `completed` |
| Created / completed | 2026-10-25T17:32:44+02:00 / 2026-10-25T17:33:26+02:00 |
| Feature columns | 483 |
| Implantation | resolved, track 0 |
| Model compat | `registry_status=unavailable` (root null) |

Do not invent additional metrics beyond these recorded artifacts.

---

## Suggested registry path (this workstation)

`C:\behavior_suite\models`  
(Optional — missing registry must not block extraction; this acceptance run used no registry.)

---

## Historical fixture blocker (resolved)

| Session | Prior blocker | Resolution |
| --- | --- | --- |
| `E_full_session_integration` | No `pose_finalization/`; S3 `acceptance_state=not_accepted` | Operator accepted S3, ran S4, then S6 |

