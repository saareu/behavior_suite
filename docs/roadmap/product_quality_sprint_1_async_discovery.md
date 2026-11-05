# Product Quality Sprint 1 — Asynchronous Discovery and Source Opening

**Status:** Implemented and verified (Sprint 1 final verification).  
**Date:** 2026-11-05  
**Baseline:** `d2f7f6b9532173feca5bef7cd8cb04734434643e`  
**Scope:** S2, S4, S5, S6 discovery / S5 source-opening responsiveness only.

---

## Objective

Remove avoidable GUI-thread blocking from discovery and S5 source opening while
preserving scientific eligibility, digest checks, and source-selection semantics.

---

## Discovery worker architecture

Reuse of the established S7–S10 pattern (no new thread-pool framework):

| Layer | Mechanism |
| --- | --- |
| Identity | Frozen `DiscoveryRequestIdentity` / `OpenRequestIdentity` with `request_id`, `generation`, `session_root` |
| Snapshot | Frozen `DiscoverySnapshot` / `OpenSnapshot` built off the GUI thread |
| Controller | `prepare_session` → `begin_*` → `collect_*_snapshot` (worker) → `apply_*_snapshot` / `record_*_failure` |
| Runner | Dedicated `GuiTaskRunner` per page (`discovery_runner`; S5 also `open_runner`) |
| Task kinds | `POSE_INFERENCE_DISCOVERY`, `POSE_FINALIZATION_DISCOVERY`, `BLOB_FEATURES_OPEN`, `BLOB_FEATURES_DISCOVERY`, `COMBINED_FEATURES_DISCOVERY` |
| Serialization | One in-flight worker per runner; newer requests bump generation; finished handler restarts for the latest identity |

Workers receive immutable snapshots of session/request identity, never touch Qt
widgets, and never mutate controller state. GUI updates happen only on the GUI
thread inside `apply_*` / page callbacks.

Synchronous `controller.set_session` / `refresh_discovery` remain available for
unit tests (prepare + collect + apply on the calling thread). Pages always use
the async path.

---

## Request / session identity rules

1. `prepare_session` / `begin_*` bumps a monotonic generation and stores
   `active_*_identity`.
2. `apply_*` succeeds only when `request_id`, `generation`, and resolved
   `session_root` still match.
3. An older snapshot cannot overwrite a newer one.
4. Same-session refresh preserves currently selected eligible runs under the
   pre-existing reconcile rules.
5. Real session changes clear prior selections / frames before the worker starts
   (S5 clears frames/thumbnails immediately so stale imagery cannot appear).

---

## Loading and error states

| State | Feedback |
| --- | --- |
| Discovering / opening | Indeterminate progress + stage/status text; no modal |
| Available sources | Existing combo/table population after apply |
| No eligible sources | Existing empty/ineligible messages |
| Validation / worker error | `record_*_failure` → status label; modal only for unexpected/actionable UI errors |
| Refresh completed | Status message; progress hidden when idle |

Actions that require resolved data stay disabled while discovery/open runs;
entire pages are not frozen beyond the existing busy patterns. Session browse
remains available so a newer session can supersede in-flight discovery.

---

## Operations moved off the GUI thread

| Subsystem | Former GUI-thread work | Measured baseline cost (E-session) |
| --- | --- | --- |
| S2 | `summarize_pose_inference_project`, S1 metadata `validate_s1_handoff` | ~41 ms cold summarize |
| S4 | `summarize_pose_finalization_project`, `list_accepted_s3_runs` (incl. SHA) | ~46 ms cold summarize |
| S5 open | `validate_s1_for_blob_features`, `S1SourceIdentity.from_eligible` digests, background load, bookmarks, frame 0, embedded discovery | ~335 ms digests + ~45 ms validate + ~19 ms frame0 + ~122 ms discovery |
| S6 | S4+S5+S6 summarizers + `list_eligible_joint_pairs` | **~1876 ms** GUI bundle |

Digest and eligibility checks are unchanged.

---

## Remaining limitations

- Cooperative cancellation only; stale discard is the safety net when workers
  cannot be interrupted mid-hash.
- Destroying a page while its discovery/open `QThread` is still running is unsafe
  (same as S7–S10); MainWindow close waits on `has_running_task`.
- S2/S4 hard-navigation locks briefly while discovery runs because
  `has_running_task` includes discovery.
- S4 `select_accepted_s3_run` / pre-dispatch revalidation still hashes on the
  GUI thread (selection-time, not discovery).
- S5 `build_request` still re-validates/re-hashes S1 on extraction start.
- No new global caches or thread pools.

---

## Test evidence

Focused responsiveness suites:

- `tests/ui/test_pose_inference_discovery_responsiveness.py`
- `tests/ui/test_pose_finalization_discovery_responsiveness.py`
- `tests/ui/test_blob_features_discovery_responsiveness.py`
- `tests/ui/test_combined_features_discovery_responsiveness.py`
- Shared helpers: `tests/ui/_responsiveness_support.py`

Existing S2/S4/S5/S6/workspace GUI tests updated to wait for async completion.
S7–S10 discovery behavior is unchanged by design (no edits to those runners).

---

## Performance measurements

**Session:** `test_cases/E_full_session_integration`  
**Mode:** offscreen Qt (`QT_QPA_PLATFORM=offscreen`) via
`scripts/_pq_sprint1_gui_discovery_audit.py`  
**Visible Windows GUI operator pass:** not run in this environment (record
separately when a desktop display is available).

### Baseline (pre-change, GUI-thread blocking)

From `tmp_audit_discovery_latency.py` / `tmp_audit_s5_open_cost.py` (audit era):

| Path | Cold cost on GUI thread |
| --- | --- |
| S2 summarize | ~41 ms |
| S4 summarize | ~46 ms |
| S5 digests + open | ~335 ms digests (+ validate/frame) |
| S6 discovery bundle | **~1876 ms** |

`page.set_session` blocked for the full scientific duration.

### After Sprint 1 (offscreen)

| Page | Cold `set_session` return | Cold until results | Warm return | Warm until results | Event-loop ticks while waiting |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 | 1.7 ms | 1862 ms | 3.4 ms | 1694 ms | 2 / 3 |
| S4 | 2.4 ms | 102 ms | 2.6 ms | 104 ms | 5 / 4 |
| S5 | 2.4 ms | 1440 ms | 3.0 ms | 954 ms | 88 / 53 |
| S6 | 12.0 ms | 3719 ms | 3.6 ms | 3168 ms | 127 / 91 |

Re-measured scientific-only costs (unchanged algorithms; wall clock varies):

| Op | Cold | Warm |
| --- | ---: | ---: |
| S2 summarize | 10.5 ms | 9.9 ms |
| S4 summarize | 32.4 ms | 31.7 ms |
| S5 summarize | 84.1 ms | 100.5 ms |
| S6 summarize | 1091.5 ms | 668.0 ms |
| S5 `validate_s1` | 41.1 ms | — |
| S5 digests | 375.2 ms | — |
| S5 frame0 | 18.7 ms | — |
| S6 controller bundle | 1350.3 ms | — |

**Interpretation:** `set_session` returns in single-digit/low-teen milliseconds; the
GUI event loop is no longer blocked for the full discovery/open duration.
Total scientific wall time is intentionally similar (S6/S5 remain I/O and
digest-heavy). S5/S6 show sustained timer ticks while workers run. Session-switch
under load is covered by responsiveness tests (stale snapshot rejection +
restart-on-finish).

**S2 ~1.86 s “until results” breakdown (final verification):** the scientific
worker path is not the bottleneck. Phased offscreen timing on the E-session
showed roughly:

| Phase | Cost |
| --- | ---: |
| Worker schedule / QThread start | ~4 ms |
| `collect_discovery_snapshot` (summarize + `validate_s1_handoff`) | ~30 ms |
| Signal delivery to GUI success handler | ~4 ms |
| GUI `refresh_from_state` → `_render_selected_run` | **~0.8–2.7 s** |
| Harness poll overhead after idle | ~8 ms |

Almost all of the wall time is **post-apply GUI rendering**: auto-selecting the
first discovered run calls `selected_run_s3_unavailable_reason()`, which runs
authoritative S3 handoff validation (`_s3_validator`) on the GUI thread
(~0.75–0.9 s warm-cache, higher on cold disk). That selection-time validation
pre-existed Sprint 1; the audit’s “until idle” metric includes it inside the
success callback’s `processEvents`, which also explains the very low S2 timer
tick counts (the GUI thread is blocked during that render, not during worker
discovery). This is a real first-paint delay after discovery completes, not an
async-discovery scheduling defect and not harness-only overhead. Moving S3
eligibility off the GUI thread is out of Sprint 1 scope.

Objective: eliminate UI blocking during discovery/open, not necessarily shorten
total scientific validation duration or selection-time S3 validation.
