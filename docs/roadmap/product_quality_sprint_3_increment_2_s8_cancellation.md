# Product Quality Sprint 3 — Increment 2: S8 Cooperative Cancellation

**Status:** Implemented and integrated on `main`  
**Date:** 2026-11-10  
**Baseline:** `3292f8b015a8f17defd209ae9dd51c098c61ef34` (Increment 1 / S5 cancellation)  
**Scope:** Subsystem 08 Behavioral Arbitration only. S4/S6/S7 unchanged. Sprint 3 is **not** marked complete.

---

## Shared cancellation architecture reused

Increment 2 reuses the Qt-free shared contract from Increment 1:

| Piece | Location |
| --- | --- |
| `CancellationCallback` / `raise_if_cancellation_requested` | `src/runtime/cancellation.py` |
| `GuiTaskContext` cancel + publication-critical flags | `src/ui/tasks.py` |
| `GuiTaskRunner.request_cancellation()` | `src/ui/tasks.py` |

No second cancellation framework was introduced. Scientific arbitration modules remain Qt-free.

---

## S8 cancellation checkpoints

Worker-side polls in `run_behavioral_arbitration_publication` at real stage boundaries:

1. Before expensive joint S7/S6 source loading/validation (pre-intake).
2. After intake / run bootstrap (`STATUS_RUNNING` meta written), before arbitration.
3. Between Sprint 1 → Sprint 2 and Sprint 2 → Sprint 3 wrappers (publication calls the wrappers separately; scientifically identical to `run_behavioral_arbitration_sprint3` alone).
4. After arbitration, before `build_behavior_states_table`.
5. After states, before `extract_behavior_bouts`.
6. After bouts, before QC evaluation.
7. After upstream revalidation, immediately before publication (final gate).

No per-frame polling. No arbitrary splits of scientific formulas. Optional
`cancellation_requested` / `publication_started` kwargs; callers without them
behave unchanged.

---

## Publication gate

Final cancel check occurs immediately after `_revalidate_sources_before_publish`.

Once publication begins:

1. `in_publication = True`
2. `publication_started()` → GUI marks noninterruptible critical section
3. Required states / bouts / optional diagnostics / QC writes proceed with **no** further cancel polls
4. Terminal status is completed / completed_with_warnings / failed only

A cancel request after the gate never retroactively yields `status=canceled`.

---

## Canceled status and S9 eligibility

- New `STATUS_CANCELED = "canceled"`
- New `BehavioralArbitrationCancelledError` (distinct from ordinary failure)
- On confirmed cancel: persist `status=canceled`, keep run dir + meta/settings/log, unlink unpublished science artifacts
- Prior completed S8 runs remain untouched and eligible
- Discovery continues to group non-completed statuses (including canceled) as `failed_or_incomplete`
- `validate_s8_run_for_downstream` / S9 packaging still require completed\* only → canceled S8 is not S9-eligible

Ordinary failures still return `success=False` with `status=failed` (existing S8 return semantics). Confirmed cancel re-raises after persist so the GUI can distinguish outcomes.

---

## GUI Cancel and shutdown

S8 page mirrors the S5 pattern:

- Cancel enabled only during active scientific arbitration, before first request, and outside the publication critical section
- Inline progress distinguishes Running vs Cancel requested (stage-level wording)
- Canceled confirmation comes only from the worker (`BehavioralArbitrationCancelledError`)
- Canceled explained as downstream-ineligible; discovery refreshes; rerun allowed after stop
- Existing request-identity stale-result rejection preserved; cancel does not retarget work

MainWindow close:

- If S8 is cancellable → `request_cooperative_shutdown()`
- If already publishing → wait for real completed/failed
- Defer close until worker stops; no QThread force termination; repeated close idempotent
- S5 shutdown behavior preserved

---

## Tests and GUI evidence

Focused:

- `tests/behavioral_arbitration/test_s8_cancellation.py`
- `tests/ui/test_behavioral_arbitration_cancellation_gui.py`
- Existing S5 cancel suites remain the Increment 1 regression gate

Regressions run green in this increment (179 passed):

- S8 unit/export/discovery (excluding lab E-session live-eligibility check that
  currently fails independently because linked S7 is no longer eligible)
- S8 GUI + responsiveness + cancellation GUI
- `tests/ui/test_tasks.py`
- S5 cancellation + GUI cancellation
- shared `tests/runtime/test_cancellation.py`
- S9 `tests/results_packaging/test_eligibility.py`

Static:

- Ruff clean on touched modules
- Mypy clean on `runtime` + S8 contracts/discovery (`--follow-imports=skip`)

Visible GUI:

- Method: instrumented slow worker + real cancel-persist path
  (`scripts/_pq_sprint3_s8_cancel_visible_gui_smoke.py`) because synthetic S8
  publication can finish before a reliable human Cancel click.
- Evidence: `docs/roadmap/_pq_sprint3_s8_cancel_visible_gui_evidence.txt` (PASS)

Windows/Qt harness note (separate from assertions):

- Combined offscreen S8 GUI / S9 eligibility pytest runs printed all assertions
  passed, then could still end with a nonzero Windows/Qt process abort at
  interpreter/`atexit` teardown (same pre-existing harness class as earlier
  PQ suites). Treat assertion outcomes as the verification signal; do **not**
  represent that process exit as a fully clean run.

---

## Performance / latency limitations

- Cancel polling is a handful of stage-boundary callback checks (O(1) phases), not an extra scientific pass.
- Individual Sprint wrappers, state/bout construction, QC, and native parquet writes are not interruptible mid-call; cancel takes effect at the next safe boundary.
- No sampling-rate, threshold, or algorithm changes.

---

## Known limitations

1. S4 / S6 / S7 scientific runners are still not wired to the shared contract.
2. Mid-stage arbitration math inside a Sprint wrapper cannot stop until that wrapper returns.
3. Canceled runs share discovery label `failed_or_incomplete` (no distinct canceled chip yet).
4. Discovery / open tasks soft-wait on app close (not cancel-driven).
5. Pre-intake cancel (before run-dir creation) raises without a canceled run directory.
   GUI cancel copy covers both cases and does not claim a persisted run path when
   bootstrap never occurred.
6. Lab E-session live eligibility
   (`tests/behavioral_arbitration/test_e_session_consistency.py::test_provenance_digests_and_eligibility`)
   currently fails independently because the linked S7 is no longer downstream-eligible.
   Recorded as a fixture limitation; not repaired in this increment.

---

## Architectural acceptance / integration

Architectural design review and final safety verification passed
(pre-bootstrap messaging, publication gate, shutdown deferral, scientific
identity, S9 ineligibility, S5 regression). Integrated on `main` as Increment 2.

**Stop before Increment 3 / S4 cancellation. Do not wire S6/S7 cancellation or
mark Sprint 3 complete in this increment.**
