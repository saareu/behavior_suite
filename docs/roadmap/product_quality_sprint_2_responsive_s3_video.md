# Product Quality Sprint 2 — Responsive S3 Video Review

**Status:** Architecturally accepted and integrated on `main`.  
**Date:** 2026-11-09  
**Baseline:** `5fb37a1a5a213158115431771a54f659f704956b`  
**Scope:** S3 Tracking Review decode + pose-overlay composition responsiveness only.

---

## Objective

Move S3 prepared-video decoding and pose-overlay composition off the Qt GUI
thread so scrubbing, playback, and review controls stay interactive, without
changing tracking results, review/edit semantics, pose schema, or scientific
sources.

---

## Implemented worker architecture

### Layers

| Layer | Mechanism |
| --- | --- |
| Identity | Frozen `FrameRequestIdentity` (`request_id`, `frame_generation`, `session_generation`, `run_id`, `frame_idx`, `pose_source`, `edit_revision`) |
| Snapshot in | Frozen `FrameRenderRequest`: video path + immutable `pose_points` + `skeleton_edges` |
| Snapshot out | Frozen `FrameRenderSnapshot`: identity + `image_bgr` or `error` |
| Controller | Playhead/edit on GUI; `begin_frame_request` → worker render → `apply_frame_snapshot` |
| Decoder | Dedicated persistent `TrackingReviewFrameDecoder` (not short-lived `GuiTaskRunner` threads) |
| Task kind | `GuiTaskKind.TRACKING_REVIEW_FRAME` reserved for identity/docs; page uses `has_running_task` |

### Why not `GuiTaskRunner` for each frame?

`GuiTaskRunner` creates a new `QThread` per task. OpenCV `VideoCapture` is not
safe across unrelated short-lived threads, and sequential NAS playback depends on
keeping one capture warm. Sprint 2 therefore uses a **dedicated decoder thread**
with the same latest-wins coalesce semantics as S10, while preserving the
existing sequential-capture algorithm inside that worker.

### Decoder thread ownership

- `cv2.VideoCapture` is created, read, seeked, and released **only** on the
  decoder thread (`_DecoderEngine`).
- GUI never holds a live capture after this change.
- `queue_request` is thread-safe so rapid scrub can overwrite pending work even
  while OpenCV is blocked inside a seek/read.
- Only one pending request is retained (no unbounded backlog).
- No raw-frame LRU was added.

### Frame request / stale-result rejection

1. Scrub / step / play tick / source change / edit refresh updates cheap metadata
   on the GUI thread, then calls `begin_frame_request()`.
2. Pose points and skeleton edges are snapshotted on the GUI thread; the worker
   never reads `TrackingReviewSession` or widgets.
3. `apply_frame_snapshot` accepts only when `request_id`, `frame_generation`,
   `session_generation`, `run_id`, `frame_idx`, `pose_source`, and `edit_revision`
   still match.
4. Labels always show the **selected** prepared frame; pixels may lag with a
   lightweight “Loading frame…” indicator. An old pixmap is never treated as the
   newly selected frame.

### Scrub / playback behavior

- Rapid scrub: latest requested frame wins; obsolete pending requests are dropped.
- Playback keeps the existing 8 ms wall-clock timer and
  `PlaybackTimingState` skip semantics (scientific playhead).
- Decode requests coalesce; slow NAS decode does not freeze controls.
- Display may lag the playhead during fast playback; that is display catch-up,
  not a new scientific frame-skipping policy.
- Pause stops the timer immediately; in-flight snapshots still must match identity.

### Edit / session invalidation

- Manual edits, resets, and pose-source changes call `invalidate_composed_frames()`
  (`edit_revision++`), then request a fresh compose of the current frame.
- `open_run` / session switches bump `session_generation` and release the worker
  capture.
- Previously rendered overlays cannot apply after an edit/source/session change.
- Correction history and acceptance behavior are unchanged. No automatic
  acceptance was introduced.

### Shutdown lifecycle

- `TrackingReviewPage.has_running_task` covers decoder busy + controller pending
  + soft-timeout “still shutting down” ownership.
- `MainWindow.closeEvent` soft-waits on S3 frame work (like S10), then calls
  `review_page.shutdown_frame_decoder()` before accepting close.
- Soft shutdown wait default: 5 s. A stalled NAS seek may outlive that limit;
  the thread is **not** force-terminated.
- Soft timeout returns False and is **not** treated as successful completion:
  application close stays deferred; worker/VideoCapture ownership remains valid
  until the thread finishes and emits `finished`.
- Hard remaining limit: OpenCV may hold the worker until the in-flight seek
  returns; document and wait, do not kill the thread.

### UI feedback

- Unobtrusive `frame_pending_label` under the viewer for pending / decode failure.
- Transport and scrubber stay enabled while decoding.
- Frame-dependent manual edits (`blank` / `swap` / “use current as start|end”)
  stay disabled until `displayed_frame_idx == playhead` and not pending, so edits
  cannot silently target a lagging visible frame.
- Undo/reset/accept remain available (history / run-level actions).
- No modal dialogs for normal seeking (failures use status text).

### Synchronous helper preserved

`TrackingReviewController.current_view()` remains a **synchronous** headless /
unit-test helper that composes on the calling thread via the injected frame
reader. The GUI page uses only the async begin/apply path.

---

## Baseline vs after (E-session)

Session: `test_cases/E_full_session_integration`  
Run: `current_lab_two_mouse_headstage_v1__20260919T181948`  
Video: NAS UNC prepared MP4  

### Before (sync GUI-thread `current_view`, 2026-11-05)

| Path | Result |
| --- | --- |
| Cold `current_view()` frame 0 | **517.5 ms** (GUI blocked) |
| Sequential frames 1–30 | **avg 91.0 ms** (GUI blocked) |
| Random 15 seeks | **avg 4849.6 ms** (GUI blocked; up to 18.1 s) |
| Warm sequential capture only | **avg 66.3 ms** |
| Overlay ×20 warm | **avg 0.7 ms** |

### After (async request-to-display, 2026-11-09 offscreen audit)

Script: `scripts/_pq_sprint2_s3_video_responsiveness_audit.py`

| Path | Result |
| --- | --- |
| Cold request→display frame 0 | **113.4 ms** (GUI not blocked) |
| Sequential request→display 1–30 | **avg 9.5 ms** (warm capture path) |
| Seek frame 5000 request→display | **2.50 s** |
| GUI timer ticks in first 250 ms of that seek | **49** (loop stayed live) |
| Random seeks request→display (n=10) | **avg 4228.7 ms** (73–13600) |
| Playback displayed-index catch-up in 2.0 s | **238 index delta → 119.0 /s** |
| tracemalloc peak | **116.4 MiB** |
| Cleanup | ok |

### Clarifying the reported “119 FPS”

That figure is **not** decoded frames/sec and **not** distinct images presented/sec.

| Metric | Meaning in the audit harness |
| --- | --- |
| Scientific playhead advance/sec | `session.current_frame` delta / wall window via `PlaybackTimingState` |
| Decode completions/sec | Worker `snapshot_ready` emissions (includes stale/coalesced work) |
| Distinct images presented/sec | Count of accepted `apply_frame_snapshot` updates that call `set_frame` |
| Frames skipped/coalesced | Playhead can advance multiple indices while only the latest pending decode runs |
| Displayed-index catch-up/sec | `(displayed_after - displayed_before) / window` — **this is what 119 measured** |
| Timer ticks | 8 ms playback poll / GUI responsiveness probe only; never frame throughput |

So “238 frames → 119 FPS” means the last successfully displayed frame index advanced by ~238 over 2 s after catch-up, while many intermediate indices were skipped by latest-wins coalesce. Actual decode and distinct present rates are lower and must be reported separately.

Interpretation:

- Random NAS seek cost remains a usability limitation (~seconds).
- The GUI stays interactive during those seeks (primary Sprint 2 goal).
- Scientific playhead still follows wall-clock `PlaybackTimingState`; display may lag.
- Sequential numbers are warmer/faster than the Nov-5 baseline; treat them as
  same-session after measurements, not a claim of new decode algorithms.

---

## Visible GUI validation

Script: `scripts/_pq_sprint2_visible_gui_smoke.py`  
Evidence: `docs/roadmap/_pq_sprint2_visible_gui_evidence.txt`

| Check | Result |
| --- | --- |
| Qt platform | `windows` (visible) |
| Open S3 run / frame 0 | displayed |
| Rapid scrub to distant frames | selected frame 100 displayed |
| Step forward | displayed |
| Play ~1.5 s / pause | playhead advanced; pixels catch up asynchronously |
| Reversible swap identities + undo on frame 0 | edit_revision advanced then undo restored workflow |
| App/page cleanup | ok |

Session switch while decoding and full MainWindow close-during-seek were covered
by automated lifecycle tests; the visible smoke focused on scrub/play/edit on the
E-session run without accepting tracking.

---

## Tests and static checks

Focused suite: `tests/ui/test_tracking_review_video_responsiveness.py`

Coverage exercised:

1. Decode off GUI thread  
2. Stable worker-thread ownership for reads  
3. Rapid scrub latest-wins / bounded pending  
4–6. Playback order, pause, seek-after-pause  
7–8. Manual edit + pose-source invalidation  
9. Session/run switch stale rejection  
10. Decoder failure surfacing (no seek modal)  
11. Safe worker shutdown  
12. Single pending slot  
13. Frame/overlay apply identity sync  
14–15. Correction/acceptance + sync `current_view` immutability helpers  

Regressions run green: S3 review, manual edits, sprint4 page load, S2→S3 handoff,
S3 overlay entry button.  

Static:

- Ruff clean on touched modules  
- Mypy clean on controller; decoder/page clean with `--disable-error-code=import-untyped`
  (PySide6 stubs; `ui` is outside default mypy packages)

---

## Known limitations / remaining risks

1. **NAS random seek latency** is unchanged in magnitude; responsiveness ≠ faster
   OpenCV seeks. Caching/re-encoding still require measurement + review.
2. Soft shutdown wait (5 s) cannot abort an in-flight OpenCV seek.
3. During fast playback, displayed pixels may lag the scientific playhead until
   the latest coalesced request completes.
4. No shared video-player unification; S10 still uses `PreparedFrameAccessor`.
5. No raw-frame LRU yet (explicit non-goal for this increment).

---

## Changed files (implementation)

| File | Role |
| --- | --- |
| `src/ui/tracking_review_frame_decoder.py` | New persistent decoder worker |
| `src/ui/controllers/tracking_review_controller.py` | Identity/begin/apply; capture removed from GUI |
| `src/ui/pages/tracking_review_page.py` | Async refresh, pending chrome, `has_running_task` |
| `src/ui/tasks.py` | `GuiTaskKind.TRACKING_REVIEW_FRAME` |
| `src/ui/main_window.py` | S3 close wait + decoder shutdown |
| `tests/ui/test_tracking_review_video_responsiveness.py` | New focused suite |
| `tests/ui/_responsiveness_support.py` | Wait for idle before destroy |
| `tests/ui/test_s2_s3_handoff.py` / sprint4 / overlay GUI | Safe teardown |
| `scripts/_pq_sprint2_s3_video_responsiveness_audit.py` | Offscreen timing audit |
| `scripts/_pq_sprint2_visible_gui_smoke.py` | Visible Windows smoke |
| `docs/roadmap/product_quality_sprint_2_responsive_s3_video.md` | This document |
| `docs/subsystem_03/mvp_scope_and_roadmap.md` | Short pointer |

---

## Architectural acceptance

Accepted for final integration on `main`: dedicated decoder-thread + immutable
frame-request design for S3 review, including the post-review safety
corrections below. Sprint 2 meets the responsiveness goal without scientific
schema/algorithm changes and without introducing a shared player or raw-frame
cache.

Post-design-review safety corrections included in this integration:

1. Soft shutdown timeout retains worker ownership and defers app/page close.
2. Frame-dependent edits stay disabled until displayed pixels match the playhead.
3. Audit/docs clarify that “119 FPS” was displayed-index catch-up, not decode/present rate.

Product Quality Sprint 2 is closed. Stop before Product Quality Sprint 3.
