# Product Quality Sprint 3 — Increment 3: S4 Cooperative Cancellation

**Status:** Integrated on `main` (Product Quality Sprint 3 Increment 3 closed).  
**Date:** 2026-11-10  
**Baseline:** `b93e0ab86f622c80ea0ca633a51a2d426b947dae` (Increment 2 / S8 cancellation)  
**Scope:** Subsystem 04 Pose Finalization only. S6/S7 cancellation unchanged.
Sprint 3 is **not** marked complete.

---

## Shared cancellation architecture reused

Increment 3 reuses the Qt-free shared contract from Increment 1:

| Piece | Location |
| --- | --- |
| `CancellationCallback` / `raise_if_cancellation_requested` | `src/runtime/cancellation.py` |
| `GuiTaskContext` cancel + publication-critical flags | `src/ui/tasks.py` |
| `GuiTaskRunner.request_cancellation()` / `start_progressive` | `src/ui/tasks.py` |
| `GuiTaskKind.POSE_FINALIZATION` | `src/ui/tasks.py` |

No second cancellation framework was introduced. Scientific S4 modules remain
Qt-free.

---

## Actual S4 stage / checkpoint map

Worker-side polls in `run_pose_finalization` at real stage boundaries:

| # | When | Notes |
| --- | --- | --- |
| 0 | Before accepted-S3 validation / run bootstrap | Pre-run-dir cancel raises without inventing a path |
| 1 | After `STATUS_RUNNING` bootstrap, before timing | Before expensive processing work |
| 2 | After timing map construction | Source/timing validation complete |
| 3 | Immediately before dense construction | Dense call loads `tracked_pose.parquet` |
| 4 | After dense construction (`apply_gap_fill=False`) | Existing public API seam |
| 5 | After `fill_coordinate_gaps` / rebuilt dense result | Reconstruction complete |
| 6 | After Savitzky–Golay smoothing | |
| 7 | After QC evaluation (+ structural assert) | |
| 8 | After S3 revalidation, immediately before publication | Final gate |

Dense construction and gap fill remain the existing algorithms. The runner uses
the already-public `apply_gap_fill=False` + `fill_coordinate_gaps` seam so a
checkpoint can sit between those observable stages without splitting science
internals. No per-sample cancellation polling.

### Uninterruptible stages / cancellation latency

Once a stage call begins, cancel is honored only at the next boundary:

- `construct_dense_pose(...)` (source load + temporal resampling)
- `fill_coordinate_gaps(...)` (linear + Kalman/RTS reconstruction)
- `apply_savgol_smoothing(...)`
- `evaluate_finalization_qc(...)` / table build
- Parquet write inside the publication critical section

Observed synthetic-fixture latency for dense+gap+smooth+QC is typically
sub-second on small fixtures; larger sessions inherit native algorithm latency
until the next checkpoint.

---

## Progress reporting

Optional coarse `progress_callback` emits stage transitions only:

- Validating inputs
- Preparing timing
- Building dense pose
- Reconstructing gaps
- Smoothing
- Computing QC
- Revalidating sources
- Writing artifacts
- Completed

No invented percentage denominators. While Cancel is pending, the GUI keeps the
“Cancel requested — waiting for a safe processing checkpoint.” message even if
an uninterruptible stage is still running.

---

## Publication gate and race protection

Final cancel check occurs immediately after accepted-S3 revalidation.

Once publication begins:

1. `in_publication = True`
2. `publication_started()` → GUI marks noninterruptible critical section
3. `final_pose.parquet` write + terminal status persistence proceed with **no**
   further cancel polls
4. Terminal status is completed / completed_with_warnings / failed only

A cancel request after the gate never retroactively yields `status=canceled`.
Worker outcome determines disk status.

---

## Canceled status and S6 eligibility

- New `STATUS_CANCELED = "canceled"`
- New `PoseFinalizationCancelledError` (distinct from ordinary failure)
- On confirmed cancel after bootstrap: persist `status=canceled`, keep run dir +
  meta/settings/log, unlink unpublished `final_pose.parquet`
- Prior completed S4 runs remain untouched and eligible
- Discovery continues to classify non-completed statuses (including canceled) as
  `failed_or_incomplete`
- `validate_s4_run_for_downstream` / S6 still require completed\* only → canceled
  S4 is not S6-eligible

Ordinary failures still raise `PoseFinalizationError` with `status=failed`.
Confirmed cancel re-raises after persist so the GUI can distinguish outcomes.
Pre-bootstrap cancel raises without a canceled run directory.

---

## GUI Cancel and hard-navigation lock

S4 page mirrors the S5/S8 Cancel pattern:

- Cancel enabled only during active scientific finalization, before first
  request, and outside the publication critical section
- Disabled immediately after Cancel is requested
- Unavailable once publication begins
- Run / source-changing controls stay locked while processing
- Inline feedback: “Cancel requested — waiting for a safe processing checkpoint.”
- Confirmed cancel shows **Canceled** only after worker confirmation
- Discovery refreshes; canceled runs explained as downstream-ineligible; rerun
  allowed after stop
- No modal confirmation for ordinary cancel; no global Cancel toolbar

Hard navigation lock (existing S2/S4 behavior) is preserved for the entire
worker lifetime:

- Cancel requested does **not** release navigation
- Lock releases only after worker completion + task cleanup
- Session change / another S4 launch / opening another tracking source remain
  blocked while the unfinished worker is active

---

## Shutdown lifecycle

`MainWindow.closeEvent` for an active S4 scientific run:

1. Ignores the close event (idempotent repeated attempts)
2. Calls `PoseFinalizationPage.request_cooperative_shutdown()`
3. Requests cooperative cancel when still before the publication gate
4. If already publishing, waits for real completion/failure
5. Keeps page/controller/thread alive until worker finished
6. No `QThread.terminate`

S2 hard-lock and S3 decoder shutdown guarantees are unchanged. S5/S8 shutdown
semantics are unchanged.

---

## Tests and GUI evidence

Focused:

- `tests/pose_finalization/test_s4_cancellation.py`
- `tests/ui/test_pose_finalization_cancellation_gui.py`

Regression gates retained:

- Existing S4 GUI / export-QC suites
- S5 cancellation + shared runtime cancellation
- S8 cancellation

Static:

- Ruff on touched modules
- Mypy on `runtime` + S4 contracts/runner (`--follow-imports=skip`)

Visible GUI:

- Method: instrumented slow worker + real cancel-persist path
  (`scripts/_pq_sprint3_s4_cancel_visible_gui_smoke.py`)
- Evidence: `docs/roadmap/_pq_sprint3_s4_cancel_visible_gui_evidence.txt`

Windows/Qt harness note (same class as Increments 1–2): combined offscreen GUI
pytest runs may abort at interpreter teardown after assertions pass. Treat
assertion outcomes as the verification signal.

---

## Scientific invariance

Deterministic fixtures compare:

- No cancellation callback
- Callback always returning `False`

Verified identical final pose values, provenance/missingness/reconstruction
flags, confidence fields, timing/frame indices, QC payload, source lineage, and
artifact schema. Dense 120 Hz contract preserved. No accepted-S3 mutation, no
CSV V2 changes, no S6/S7 algorithm changes.

---

## Known limitations

1. S6 / S7 scientific runners are still not wired to the shared contract.
2. Mid-stage dense/gap/smooth/QC native work cannot stop until that call returns.
3. Canceled runs share discovery label `failed_or_incomplete` (no distinct
   canceled chip yet).
4. Pre-bootstrap cancel (before run-dir creation) raises without a canceled run
   directory; GUI copy covers both cases.
5. Sprint 3 is not complete until remaining increments (if any) and architectural
   acceptance are finished.

---

## Architectural acceptance / integration

Architectural design review and final safety verification passed
(checkpoint cancellation before publication, noninterruptible publication gate,
worker-outcome status, S6 ineligibility for canceled runs, hard-navigation lock
until worker stop, deferred MainWindow shutdown, scientific invariance, S5/S8
unchanged). Integrated on `main` as Increment 3.

**Stop before Increment 4. Do not wire S6/S7 cancellation or mark Sprint 3
complete in this increment.**
