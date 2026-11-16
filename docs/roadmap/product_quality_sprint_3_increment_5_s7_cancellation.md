# Product Quality Sprint 3 — Increment 5: S7 Cooperative Cancellation

**Status:** Integrated on `main` (Product Quality Sprint 3 Increment 5 closed).  
**Date:** 2026-11-16  
**Baseline:** `e5d5fed12d15dfb3d73906afb5862d2939beac99` (Increment 4 / S6 cancellation)  
**Scope:** Subsystem 07 Behavioral Inference only. S4/S5/S6/S8 cancellation unchanged.  
Sprint 3 is **not** marked complete.

---

## Shared cancellation architecture reused

Increment 5 reuses the Qt-free shared contract from Increment 1:

| Piece | Location |
| --- | --- |
| `CancellationCallback` / `raise_if_cancellation_requested` | `src/runtime/cancellation.py` |
| `GuiTaskContext` cancel + publication-critical flags | `src/ui/tasks.py` |
| `GuiTaskRunner.request_cancellation()` / `start_progressive` | `src/ui/tasks.py` |
| `GuiTaskKind.BEHAVIORAL_INFERENCE` | `src/ui/tasks.py` |

No second cancellation framework was introduced. Scientific S7 modules remain
Qt-free.

---

## Actual S7 inference execution architecture

`run_behavioral_inference_publication` remains the sole publication runner:

1. Validate purpose / resolve config and eligible S6
2. Load S6 `dyadic_features_full` + digest
3. Create immutable run directory; persist `STATUS_RUNNING` meta/settings
4. Call `run_behavioral_inference` (in-memory engine)
5. Validate predictions; build canonical / specialist / optional wide tables
6. Build model manifest + QC; assert structural QC
7. Revalidate S6 digests + model digests
8. Enter publication critical section; write parquet/QC/manifest
9. Persist terminal status

Engine / adapter execution order is unchanged:

- Dyadic logical detectors first (caller order), then focal (caller order)
- Each logical detector is scored sequentially
- Contact / nose-to-body may issue multiple specialist `predict_proba` batches
- Scoring remains full-row batch calls (no per-sample polling)

---

## Actual cancellation checkpoints

Worker-side polls at real stage boundaries:

| # | When | Notes |
| --- | --- | --- |
| 0 | Before S6 resolve / feature load | Pre-run-dir cancel raises without inventing a path |
| 1 | Immediately before run-dir creation | Confirmed cancel still invents no path |
| 2 | After `STATUS_RUNNING` bootstrap | Before model load / inference |
| 3 | Before registry discovery / model loading | Inside engine |
| 4 | Before each required model bundle load | Inside `load_models_for_selection` |
| 5 | Before starting detector inference | After models are ready |
| 6 | Before each logical detector | Between detectors; preserves order |
| 7 | Before each blocking `predict_proba` | Via `score_positive_probabilities` |
| 8 | Between dyadic and focal families | Engine family boundary |
| 9 | After all detectors complete | Before result assembly |
| 10 | After inference returns to publication runner | Before table construction |
| 11 | After prediction tables / manifest prep | Before QC |
| 12 | After QC | Before source/model revalidation |
| 13 | Final gate after revalidation | Immediately before publication |

### Uninterruptible stages / active-detector limitations

Once a detector’s `predict_proba` call begins, cancel is honored only after that
call returns. Active detectors are **not** interrupted mid-probability. Cancel
latency is therefore bounded by the current detector’s batch scoring time (and,
for dual-specialist detectors, by any specialist call already in flight).

Publication artifact writes are noninterruptible after the final gate.

---

## Shared cancellation interface

```text
run_behavioral_inference_publication(
    request,
    *,
    progress_callback=None,
    cancellation_requested: Callable[[], bool] | None = None,
    publication_started: Callable[[], None] | None = None,
) -> BehavioralInferenceRunResult
```

- No callback → existing behavior.
- Callback returning `False` → existing behavior.
- Confirmed cancel raises `BehavioralInferenceCancelledError` (subclass of
  `BehavioralInferenceError`) after the worker stops at a safe checkpoint.
- Repeated requests are harmless.
- Ordinary exceptions remain failures (`STATUS_FAILED`).
- No forced thread termination; no Qt dependency in science modules.

Engine/adapters accept the same optional callback without changing scientific
formulas, detector order, adapters, or prediction schemas.

---

## Progress reporting

Optional coarse `progress_callback` emits stage transitions only:

- Validating S6 source and models (`validating_sources`)
- Loading S6 features (`loading_inputs`)
- Loading models (`loading_models`)
- Preparing inference (`preparing_inference`)
- Running detector inference (`running_detector_inference`) with current
  detector name and completed/total detector counts
- Constructing prediction tables (`constructing_prediction_tables`)
- Computing QC (`computing_qc`)
- Revalidating sources (`revalidating_sources`)
- Writing artifacts (`writing_artifacts`)
- Completed (`completed`)

No invented percentage denominators beyond real detector counts. While Cancel is
pending, the GUI keeps:

“Cancel requested — waiting for the current detector to finish.”

---

## Publication boundary

Final ordering:

1. Finish scientific inference (all selected detectors).
2. Prepare prediction tables, manifest, and QC.
3. Revalidate S6 source and model digests.
4. Final cooperative cancellation check.
5. Enter publication critical section (`publication_started()` / GUI context mark).
6. Write scientific Parquet/QC/manifest artifacts.
7. Persist actual terminal status.

Once step 5 is committed, cancellation is ignored; the run completes or fails
normally and is never retroactively marked canceled.

---

## Canceled status and S8 eligibility

- `STATUS_CANCELED = "canceled"` in `behavioral_inference.contracts`.
- On confirmed cancel after run creation: persist `status=canceled`, keep
  meta/settings/log, remove unpublished prediction/QC/manifest artifacts, retain
  the run directory.
- On cancel before run creation: raise confirmed cancel; no invented run path.
- Discovery keeps canceled under outward `failed_or_incomplete`.
- `is_s7_run_eligible_for_downstream` / `validate_s7_run_for_downstream` still
  require `completed` / `completed_with_warnings`.
- S8 `find_latest_eligible_s7_run` therefore excludes canceled runs without
  weakening completed/completed_with_warnings eligibility.

---

## GUI Cancel behavior

`BehavioralInferencePage`:

- Inline Cancel available only while a cancellable scientific job is active.
- Disabled after cancel is requested; unavailable during publication.
- Run / source / registry / detector controls remain protected while busy.
- Pending copy: “Cancel requested — waiting for the current detector to finish.”
- “Canceled” only after worker confirmation; explains S8 ineligibility.
- Discovery refresh after task completion; rerun permitted once the worker ends.
- Existing request-identity / stale-result protection preserved.
- No modal confirmation dialogs for Cancel.

---

## Shutdown behavior

`MainWindow.closeEvent` now calls
`behavioral_inference_page.request_cooperative_shutdown()` for active S7 science
jobs (same pattern as S4/S5/S6/S8):

- Request cancel before publication when possible.
- Defer close until the worker terminates.
- If a detector is running, wait until it reaches a safe checkpoint.
- If publication has started, allow publication to finish.
- Preserve QThread ownership; never force-terminate.
- Repeated close requests remain idempotent.

S3 decoder and S4/S5/S6/S8 shutdown behavior are preserved.

---

## Scientific invariance

Deterministic fake-registry fixtures compare:

- No cancellation callback
- Callback always returning `False`

Verified identical detector execution order, predicted probabilities,
predictions, focal identities, dense frame indices, prediction masks, QC
structural fields, and model-manifest detector/specialist lineage. S6 artifacts
and model files are not modified. No prediction batch-size or scoring-formula
changes.

---

## Validation evidence

### Targeted suites (2026-11-16)

- `tests/behavioral_inference/test_s7_cancellation.py` — **24 passed**
- `tests/ui/test_behavioral_inference_cancellation_gui.py` — **7 passed**
- Broader S7 GUI / discovery / runtime / S4/S5/S8 cancel regression —
  **111 passed** (process exit may show known offscreen Qt `0xC0000409` abort
  after green assertions)
- Ruff on touched modules — clean
- Targeted Mypy (`--follow-imports=skip`) on S7 cancel surfaces: no newly
  introduced diagnostics beyond pre-existing `no-any-return` /
  `var-annotated` notes already present on baseline `predict.py` /
  adapters / runner helpers

Visible GUI:

- Method: **instrumented slow worker + real cancel-persist path**
  (`scripts/_pq_sprint3_s7_cancel_visible_gui_smoke.py`)
- Evidence: `docs/roadmap/_pq_sprint3_s7_cancel_visible_gui_evidence.txt`
- Confirmed: Cancel enablement during active detector feedback, pending
  “current detector” message, worker-confirmed `Canceled`, S8 exclusion of
  canceled run, prior completed remains eligible, successful rerun, deferred
  application close while worker finishes
- Explicit inference mode: instrumented worker (not live lab `predict_proba`)

Performance (synthetic n_frames=8 fake-model fixture, 3 repeats after warmup):

- No cancellation callback mean ≈ 0.089 s
- Callback always returning False mean ≈ 0.092 s
- No meaningful additional overhead; no extra model loads, inference passes,
  batch-size changes, or prediction-matrix copies for cancellation
- Active `predict_proba` calls are **not** interruptible

---

## Known limitations

- An active detector’s `predict_proba` must return before cancel is honored.
- Dual-specialist detectors (contact / nose-to-body) may finish an already-started
  specialist score before cancel takes effect.
- No per-sample cancellation polling.
- Sprint 3 is not complete until cross-subsystem closure work is authorized.

---

## Architectural acceptance / integration

Architectural design review and final safety verification passed
(checkpoint cancellation before publication, empty-specialist-mask cancel
boundaries, noninterruptible publication gate, worker-outcome status,
S8 ineligibility for canceled runs with prior completed runs still eligible,
soft-nav + Cancel/MainWindow cooperative shutdown, scientific invariance of
no-callback vs always-False outputs, S4/S5/S6/S8 cancellation unchanged,
detector models/features/thresholds/schemas unchanged). Integrated on `main`
as Increment 5.

Final safety confirmation retained:

- Cancellation exceptions propagate through engine, adapters, and prediction helpers.
- Empty specialist masks cannot bypass cancellation checks.
- Active `predict_proba` calls finish before cancellation is honored.
- Partially accumulated detector predictions cannot be published as success.
- Ordinary model failures remain failures.
- Completed predictions remain scientifically identical with a false cancellation callback.
- Canceled S7 runs remain S8-ineligible; previously completed S7 runs remain eligible.
- Publication is noninterruptible after the final cancellation checkpoint.
- Application shutdown cannot destroy a live S7 worker.
- Existing S4/S5/S6/S8 cancellation remains unchanged.

Do not mark Sprint 3 complete. Do not begin cross-subsystem Sprint 3 closure
in this increment. Do not change scientific models, feature calculations,
thresholds, schemas, CSV V2 exports, or Hala cohort data.
