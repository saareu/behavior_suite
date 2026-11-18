# Product Quality Sprint 3 — Increment 4: S6 Cooperative Cancellation

**Status:** Integrated on `main` (Product Quality Sprint 3 Increment 4 closed).  
**Product Quality Sprint 3:** Complete and Accepted — see
[`product_quality_sprint_3_closure_review.md`](product_quality_sprint_3_closure_review.md).  
**Date:** 2026-11-15  
**Baseline:** `d875014f58bf79c34633080cae591cbfe66932a9` (Increment 3 / S4 cancellation)  
**Scope:** Subsystem 06 Combined Feature Extraction only. S7 cancellation unchanged in this increment.

---

## Shared cancellation architecture reused

Increment 4 reuses the Qt-free shared contract from Increment 1:

| Piece | Location |
| --- | --- |
| `CancellationCallback` / `raise_if_cancellation_requested` | `src/runtime/cancellation.py` |
| `GuiTaskContext` cancel + publication-critical flags | `src/ui/tasks.py` |
| `GuiTaskRunner.request_cancellation()` / `start_progressive` | `src/ui/tasks.py` |
| `GuiTaskKind.COMBINED_FEATURES_EXTRACTION` | `src/ui/tasks.py` |

No second cancellation framework was introduced. Scientific S6 modules remain
Qt-free.

---

## Actual S6 stage / checkpoint map

Worker-side polls in `run_combined_feature_extraction` and
`compute_combined_features_*` at real stage boundaries:

| # | When | Notes |
| --- | --- | --- |
| 0 | Before S4/S5 pair resolve / run bootstrap | Pre-run-dir cancel raises without inventing a path |
| 1 | After `STATUS_RUNNING` bootstrap | Before adaptation / feature construction |
| 2 | Before pose adaptation | Inside `compute_combined_features_from_pair` |
| 3 | After pose adaptation | Before frame-feature construction |
| 4 | Before `build_frame_features` | Inside `compute_combined_features_from_adapted` |
| 5 | After `build_frame_features` | Before Pipeline50 post |
| 6 | Before `apply_pipeline50_post` | |
| 7 | After `apply_pipeline50_post` | |
| 8 | After compute returns (table stamp) | Runner resumes |
| 9 | After QC + model-compat preparation | |
| 10 | After S4/S5 lineage revalidation, immediately before publication | Final gate |

No cancellation polling inside forward-fill loops, feature-vector calculations,
or individual samples. No unnecessary large-array copies were introduced for
cancellation.

### Uninterruptible stages / cancellation latency

Once a stage call begins, cancel is honored only at the next boundary:

- `validate_joint_s4_s5_pair` / source parquet loads (pair resolve + revalidation)
- `adapt_final_pose_to_frame_feature_input`
- `build_frame_features` (longest scientific stage on large sessions)
- `apply_pipeline50_post` (blob join, echoes, occlusion, implantation)
- QC / model-compat report construction
- Parquet + companion writes inside the publication critical section

`build_frame_features` remains the longest noninterruptible stage until return.

---

## Shared cancellation interface

```text
run_combined_feature_extraction(
    request,
    *,
    progress_callback=None,
    cancellation_requested: Callable[[], bool] | None = None,
    publication_started: Callable[[], None] | None = None,
) -> CombinedFeaturesResult
```

- No callback → existing behavior.
- Callback returning `False` → existing behavior.
- Confirmed cancel raises `CombinedFeaturesCancelledError` (subclass of
  `CombinedFeaturesError`) after the worker stops at a safe checkpoint.
- Repeated requests are harmless.
- Ordinary exceptions remain failures (`STATUS_FAILED`).
- No forced thread termination; no Qt dependency in science modules.

---

## Progress reporting

Optional coarse `progress_callback` emits stage transitions only:

- Validating S4/S5 sources (`validating_sources`)
- Loading inputs (`loading_inputs`)
- Adapting sources (`adapting_sources`)
- Computing combined features (`computing_features`)
- Postprocessing (`postprocessing`)
- Computing QC (`computing_qc`)
- Revalidating source lineage (`revalidating_sources`)
- Writing artifacts (`writing_artifacts`)
- Completed (`completed`)

No invented percentage denominators. While Cancel is pending, the GUI keeps the
“Cancel requested — waiting for a safe processing checkpoint.” message even if
an uninterruptible stage is still running.

---

## Publication boundary

Final ordering:

1. Complete scientific computation (adapt → FF → Pipeline50).
2. Complete QC and model-compat preparation.
3. Revalidate S4 and S5 source lineage.
4. Final cooperative cancellation check.
5. Enter publication critical section (`publication_started()` / GUI context mark).
6. Write parquet + QC + model-compat (existing per-file atomic writes preserved).
7. Persist actual terminal status.

Once step 5 is committed, cancellation is ignored; the run completes or fails
normally and is never retroactively marked canceled.

---

## Canceled status and S7 eligibility

- `STATUS_CANCELED = "canceled"` in `combined_features.contracts`.
- On confirmed cancel after run creation: persist `status=canceled`, keep
  meta/settings/log, remove unpublished science outputs, retain the run
  directory.
- On cancel before run creation: raise confirmed cancel; no invented run path.
- Discovery keeps canceled under outward `failed_or_incomplete`.
- `is_s6_run_eligible_for_downstream` / `validate_s6_run_for_downstream` still
  require `completed` / `completed_with_warnings`.
- S7 `find_latest_eligible_s6_for_s7` therefore excludes canceled runs without
  weakening completed/completed_with_warnings eligibility.

---

## GUI Cancel behavior

`CombinedFeaturesPage`:

- Inline Cancel available only while a cancellable scientific job is active.
- Disabled after cancel is requested; unavailable during publication.
- Run / source-mutating controls remain protected while busy.
- Pending copy: “Cancel requested — waiting for a safe processing checkpoint.”
- “Canceled” only after worker confirmation; explains S7 ineligibility.
- Discovery refresh after task completion; rerun permitted once the worker ends.
- Existing request-identity / stale-result protection preserved.
- No modal confirmation dialogs for Cancel.

---

## Shutdown behavior

`MainWindow.closeEvent` now calls
`combined_features_page.request_cooperative_shutdown()` for active S6 science
jobs (same pattern as S4/S5/S8):

- Request cancel before publication when possible.
- Defer close until the worker terminates.
- If publication has started, allow publication to finish.
- Preserve QThread ownership; never force-terminate.
- Repeated close requests remain idempotent.

S7 shutdown remains wait-only in this increment.

---

## Scientific invariance

Deterministic joint S4/S5 fixtures compare:

- No cancellation callback
- Callback always returning `False`

Verified identical combined feature values, frame indices, timing/alignment,
QC, model-compat, and source run identities. S4/S5 science artifacts are not
modified. The 120 Hz joint timing contract is unchanged.

---

## Validation evidence

### Final verification pass (2026-11-15, Increment 4 closure checks)

1. **Processing scientific invariance** (`processing.py` checkpoints):
   - Added/executed `test_processing_checkpoints_scientific_invariance` comparing
     `compute_combined_features` with no callback vs always-`False`.
   - Verified identical directional + frame-feature tables (incl. forward-fill /
     temporal echo columns), Pipeline50 blob join columns, frame alignment
     metadata, adapted pose tensors, and implantation attrs.
   - Exception propagation: injected FF / Pipeline50 failures remain
     `CombinedFeaturesError` (not rewritten as canceled); cancel before FF skips
     science stages.

2. **Cancellation / publication races**:
   - Final pre-publication checkpoint cancel → `status=canceled`, no science
     artifacts.
   - Cancel immediately after publication gate / during parquet write → completed
     run retained; never retroactively canceled; S7-eligible.
   - GUI: Cancel unavailable in publication; MainWindow close during publication
     defers close, does **not** request cancel, preserves QThread ownership until
     worker finishes successfully.
   - MainWindow close during pre-publication processing requests cooperative
     cancel and shows pending checkpoint feedback.

3. **Mypy `no-any-return` vs baseline `d875014`** (detached temp worktree):
   - Current and baseline each report **exactly 3** `no-any-return` diagnostics
     in `src/combined_features/runner.py`, same symbols:
     - `validate_s6_run_for_downstream` → `return memoized_validation(...)`
     - `_settings_payload` → `return payload`
     - `_sha256` → `return sha256_file(path)`
   - Line numbers shifted only because Increment 4 inserted cancel/progress code.
   - **No newly introduced** `no-any-return` (or cancel-related) typing failures.

Targeted suites (executed this verification pass):

- `tests/combined_features/test_s6_cancellation.py` — **28 passed**
- `tests/ui/test_combined_features_cancellation_gui.py` — **7 passed** (incl.
  close-during-publication)
- Combined with export/processing/S6 GUI + S4/S5/S8 cancel unit/GUI + runtime
  cancel — **121 passed** (process exit may show known offscreen Qt
  `0xC0000409` abort after green assertions)
- Ruff on touched modules — clean
- Mypy on `runtime` + S6 contracts/processing/runner (`--follow-imports=skip`) —
  three pre-existing `no-any-return` only

Visible GUI:

- Method: instrumented slow worker + real cancel-persist path
  (`scripts/_pq_sprint3_s6_cancel_visible_gui_smoke.py`)
- Evidence: `docs/roadmap/_pq_sprint3_s6_cancel_visible_gui_evidence.txt`
- Confirmed: Cancel enablement, pending checkpoint feedback, worker-confirmed
  Canceled, S7 exclusion of canceled run, prior completed remains eligible,
  deferred application close while processing

Performance (synthetic n_samples=8 joint fixture, 3 repeats after warmup):

- No cancellation callback mean ≈ 0.340 s
- Callback always returning False mean ≈ 0.338 s
- No meaningful additional overhead; no extra full-source reads or large-array
  copies introduced for cancellation

---

## Known limitations

- `build_frame_features` and `apply_pipeline50_post` remain noninterruptible
  until return; cancel latency is bounded by those stage durations.
- No per-sample / forward-fill loop polling in this increment.
- S7 cooperative cancellation is intentionally deferred.
- S7 cancellation and formal Sprint 3 acceptance are recorded in later
  increment / closure docs (`product_quality_sprint_3_closure_review.md`).

---

## Architectural acceptance / integration

Architectural design review and final safety verification passed
(checkpoint cancellation before publication, noninterruptible publication gate,
worker-outcome status, S7 ineligibility for canceled runs with prior completed
runs still eligible, soft-nav + Cancel/MainWindow cooperative shutdown,
scientific invariance of no-callback vs always-False outputs, S4/S5/S8
cancellation unchanged, S6 feature definitions/timing/schemas unchanged).
Integrated on `main` as Increment 4.

Final safety confirmation retained:

- No-callback and always-False-callback scientific outputs are identical.
- Cancellation exceptions propagate correctly.
- Canceled S6 runs remain S7-ineligible; previously completed S6 runs remain eligible.
- Publication is noninterruptible after the final cancellation checkpoint.
- App closure cannot destroy a live S6 worker.
- S4/S5/S8 cancellation behavior remains unchanged.
- S6 feature definitions, timing, and schemas remain unchanged.

Static diagnostics retained vs baseline `d875014`:

- Exactly three pre-existing Mypy `no-any-return` diagnostics in
  `src/combined_features/runner.py` (`validate_s6_run_for_downstream`,
  `_settings_payload`, `_sha256`); no newly introduced typing failures.

Windows/Qt harness note (same class as Increments 1–3): combined offscreen GUI
pytest runs may abort at interpreter teardown (`0xC0000409`) after assertions
pass. Treat assertion outcomes as the verification signal; keep that abort
separate from green test results.

**Increment 4 stop condition (historical):** stop before Increment 5 / S7
cancellation inside this increment. Formal Sprint 3 completion is recorded in
`product_quality_sprint_3_closure_review.md`.
