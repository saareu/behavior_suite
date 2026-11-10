# Product Quality Sprint 3 — Increment 1: S5 Cooperative Cancellation

**Status:** Integrated on `main` (Product Quality Sprint 3 Increment 1 closed).  
**Date:** 2026-11-09 (verification); 2026-11-10 (integration)  
**Baseline:** `216eb431d275f837b0d9ff7d5e4f417409fdabcf`  
**Scope:** Shared cancellation contract + complete cooperative cancellation for
S5 scientific blob-feature extraction only. S4/S6/S7/S8 cancellation is out of
scope for this increment.

---

## Objective

Add a minimal reusable cooperative-cancellation contract and wire it through
Subsystem 05 full extraction so the GUI and application shutdown can request a
clean stop without force-terminating `QThread`s, without partial parquet
publication, and without changing scientific algorithms or schemas.

---

## Shared cancellation contract

| Piece | Location | Notes |
| --- | --- | --- |
| `CancellationCallback` | `src/runtime/cancellation.py` | `Callable[[], bool]` |
| `CooperativeCancellation` | same | Base exception, distinguishable from ordinary failure |
| `raise_if_cancellation_requested` / `is_cancellation_requested` | same | Optional callback; `None` or false → no-op |
| `GuiTaskContext` flags | `src/ui/tasks.py` | Existing `cancellation_requested`; added `in_publication_critical_section` |
| S5 exception / status | `blob_features.contracts` | `BlobFeaturesCancelledError`, `STATUS_CANCELED = "canceled"` |

Design rules:

- Optional callback preserves backward compatibility.
- GUI must not report **Canceled** until the worker confirms
  `BlobFeaturesCancelledError`.
- Repeated cancel requests are harmless.
- Non-cancel exceptions remain **Failed**.
- No `QThread.terminate`.
- Runtime helper is Qt-free for later S4/S6/S7/S8 reuse.

---

## S5 cancellation checkpoints

Safe poll points in `run_blob_feature_extraction` /
`_extract_source_features`:

1. Before expensive source opening (before timing/background/video open work).
2. After `VideoCapture` opens and before the frame loop.
3. Once per frame, immediately before the next `cap.read()`.
4. After extraction returns and before temporal/QC postprocessing.
5. After S1 revalidation and immediately before publication.

Progress UI events remain throttled independently of per-frame cancel polls.
A blocking OpenCV `read()` may finish the current native decode before the next
checkpoint.

---

## Publication critical section

After checkpoint 5 passes:

1. `publication_started()` is invoked (GUI marks
   `GuiTaskContext.in_publication_critical_section`).
2. Progress phase becomes `writing`.
3. Parquet / QC / metadata finalization runs without further cancel checks.

If cancel was requested after the gate:

- Successful publication → **Completed** (or completed_with_warnings).
- Genuine write/validation error → **Failed**.
- Never retroactively **Canceled**.

---

## Status transition rules

| Outcome | Terminal `run_meta.status` | Science artifacts |
| --- | --- | --- |
| Success | `completed` / `completed_with_warnings` | Published |
| Ordinary error | `failed` | Unpublished parquet/QC removed |
| Confirmed cancel before gate | `canceled` | Unpublished parquet/QC removed; run dir kept |
| Cancel request after gate | completed or failed as above | Normal publication rules |

Canceled runs:

- Preserve settings + processing log / cancel reason.
- Are never downstream-eligible
  (`validate_s5_run_for_downstream` requires completed status).
- Discover as `failed_or_incomplete` (existing outward classification retained).
- Do not rewrite prior completed runs.

---

## GUI behavior

On the S5 processing page:

- **Cancel** available only while scientific extraction is cancellable.
- Disabled after cancel is requested.
- Unavailable after publication critical section begins.
- Run / source-mutating controls stay locked while busy.
- Inline feedback: “Cancel requested — stopping after the current frame.”
- Confirmed cancel shows **Canceled**, refreshes discovery, explains
  non-eligibility, and allows a new run.
- No modal confirmation for ordinary cancel.
- No global cancellation toolbar.

---

## Shutdown lifecycle

`MainWindow.closeEvent` for an active S5 scientific run:

1. Ignores the close event (idempotent repeated attempts).
2. Calls `BlobFeaturesPage.request_cooperative_shutdown()`.
3. Requests cooperative cancel when still before the publication gate.
4. If already publishing, waits for real completion/failure.
5. Keeps page/controller/thread alive until worker finished.
6. Defers close via existing `_close_after_s5_task` + `task_finished`.

Open/discovery S5 tasks still soft-wait (no force kill). S4/S6/S7/S8 shutdown
behavior is unchanged in this increment.

---

## Test evidence

Focused:

- `tests/runtime/test_cancellation.py`
- `tests/blob_features/test_s5_cancellation.py`
- `tests/ui/test_blob_features_cancellation_gui.py`

Regressions run green in this increment:

- S5 export/QC/runner, run discovery, S1 eligibility
- S5 GUI + discovery responsiveness
- `tests/ui/test_tasks.py`

Static:

- Ruff clean on touched modules
- Mypy clean on `runtime` + S5 contracts/runner/discovery

Visible GUI validation:

- Automated Qt offscreen suite (`tests/ui/test_blob_features_cancellation_gui.py`).
- Visible native Windows smoke via
  `scripts/_pq_sprint3_s5_cancel_visible_gui_smoke.py` against a disposable
  synthetic eligible session (`tmp_pq_sprint3_s5_cancel_gui/`). Evidence:
  `docs/roadmap/_pq_sprint3_s5_cancel_visible_gui_evidence.txt`.

---

## Performance comparison

Synthetic 30-frame fixture via
`scripts/_pq_sprint3_s5_cancel_perf_smoke.py` (local, not a lab NAS claim):

| Mode | Wall time (after warmup) | Cancel polls |
| --- | --- | --- |
| With false cancel callback | 0.0763 s | 35 |
| With no callback | 0.0749 s | 0 |

Delta is within normal local noise on a tiny fixture. Cancel polling is
O(frames)+phase checks only; no extra full-video pass; progress throttling
unchanged; no threshold optimization.

---

## Remaining limitations

1. S4 / S6 / S7 / S8 scientific runners are not yet wired to the shared contract.
2. In-flight OpenCV `VideoCapture.read()` is not interruptible mid-call.
3. Canceled runs share the discovery label `failed_or_incomplete` (no distinct
   canceled discovery chip yet).
4. Preview / open / discovery tasks are not cancel-driven on app close (still
   soft-wait).
5. Disposable synthetic visible smoke was executed; optional lab-NAS session
   cancel smoke remains available before broader Sprint 3 closure if desired.

---

## Changed files (implementation)

| File | Role |
| --- | --- |
| `src/runtime/cancellation.py` | Shared cancel contract |
| `src/runtime/__init__.py` | Package export |
| `src/blob_features/contracts.py` | `STATUS_CANCELED`, `BlobFeaturesCancelledError` |
| `src/blob_features/runner.py` | Checkpoints, publication gate, cleanup |
| `src/blob_features/run_discovery.py` | Canceled eligibility docs/classification note |
| `src/blob_features/__init__.py` | Public exports |
| `src/ui/tasks.py` | Publication-critical flag on context/progress |
| `src/ui/controllers/blob_features_controller.py` | Cancel + publication callbacks |
| `src/ui/pages/blob_features_page.py` | Cancel UI + shutdown helper |
| `src/ui/main_window.py` | S5 cooperative close |
| `tests/runtime/test_cancellation.py` | Shared helper tests |
| `tests/blob_features/test_s5_cancellation.py` | S5 checkpoint/gate/eligibility |
| `tests/ui/test_blob_features_cancellation_gui.py` | GUI + MainWindow close |
| `pyproject.toml` | Mypy package `runtime` |
| `docs/roadmap/product_quality_sprint_3_increment_1_s5_cancellation.md` | This document |
| `docs/subsystem_05/mvp_scope_and_roadmap.md` | Cancellation status pointer |

---

## Architectural acceptance / final verification

Architectural design review passed. Final verification on 2026-11-09 confirmed:

- Publication-boundary cancel/complete exclusivity (worker authoritative).
- Visible Windows Cancel + deferred close with no QThread destruction warnings.
- Canceled-run metadata retained; not S6-eligible; prior completed runs remain
  eligible.
- Focused cancel/GUI/task regressions green; Ruff/Mypy clean on touched scope.

**Stop before Increment 2 / S4–S8 cancellation. No further cancellation wiring
in this increment.**
