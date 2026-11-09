# Product Quality Sprint 2 — Responsive S3 Video Review

**Status:** Proposed — awaiting architectural review (not implemented).  
**Date:** 2026-11-05  
**Baseline:** `7e43216b358272235bcb0485125379d293ea04eb`  
**Scope:** S3 Tracking Review decode + pose-overlay composition responsiveness only.

---

## Objective

Move S3 prepared-video decoding and pose-overlay composition off the Qt GUI
thread so scrubbing, playback, and review controls stay interactive, without
changing tracking results, review/edit semantics, pose schema, or scientific
sources.

---

## Audit findings (current GUI-thread path)

### Call chain

```text
TrackingReviewPage._refresh_view()          # GUI thread
  → TrackingReviewController.current_view() # GUI thread
      → _read_frame() / VideoCapture        # decode / seek
      → session.pose_points_for_frame()     # O(1) overlay-index lookup
      → draw_pose_overlay()                 # OpenCV compose
  → PoseVideoView.set_frame() + labels      # paint
```

Every scrubber tick, frame step, playback advance, pose-source change, and
manual edit ends in synchronous `_refresh_view()`.

### What runs on the GUI thread today

| Operation | Thread | Cost class |
| --- | --- | --- |
| `set_frame` / playback wall-clock math | GUI | cheap |
| `pose_points_for_frame` | GUI | ~0 ms (dict lookup) |
| Sequential `VideoCapture.read` | GUI | ~66 ms/frame (E-session NAS) |
| Random seek via `read_raw_frame_at_index` | GUI | ~0.2–18 s (E-session NAS) |
| `draw_pose_overlay` | GUI | ~0.4–0.7 ms |
| `PoseVideoView` paint | GUI | small vs decode |
| Manual edit mutate + overlay-index patch | GUI | existing path; keep |
| Working-pose parquet persist | background thread already | keep |

### Mutable state dependencies

- Playhead: `TrackingReviewSession.current_frame`
- Display source: `pose_source` (corrected vs provisional)
- Corrected overlay index: `_corrected_by_frame` (patched on manual edits)
- Acceptance / episode list / manual-edit stack
- Controller-owned `cv2.VideoCapture` (`_capture`, `_capture_path`, `_capture_next_idx`)

OpenCV capture handles are not thread-safe and must not be shared across threads.

### Existing patterns to reuse (do not invent a third video stack)

| Pattern | Location | Reuse for S3 |
| --- | --- | --- |
| Latest-wins decode runner | S10 `decode_runner` + `DecodeRequestIdentity` | Yes — same coalesce model |
| `GuiTaskRunner` one-in-flight | `ui.tasks` | Yes — dedicated page runner |
| `PreparedFrameAccessor` | S10 raw LRU + `read_raw_frame_at_index` | Optional for raw LRU only; insufficient alone for sequential playback |
| S3 sequential capture | `TrackingReviewController._read_frame` | Yes — move ownership into the frame worker |
| Pose overlay | `tracking_correction.overlay_draw.draw_pose_overlay` | Unchanged API |

Sprint 1 discovery async is orthogonal; S3 review has no discovery runner today.

### Baseline measurements (E-session, 2026-11-05)

Session: `test_cases/E_full_session_integration`  
Run: `current_lab_two_mouse_headstage_v1__20260919T181948`  
Video: NAS UNC prepared MP4 (`36239` frames @ ~119 Hz)

| Path | Result |
| --- | --- |
| Cold `current_view()` frame 0 | **517.5 ms** |
| Sequential frames 1–30 via `current_view()` | **avg 91.0 ms** (37–179) |
| Random 15 seeks via `current_view()` | **avg 4849.6 ms** (346–18085) |
| Mid-frame breakdown (frame 5000) | decode **196.2 ms**, pose points **~0 ms**, overlay **0.4 ms** |
| Warm sequential capture only | **avg 66.3 ms** |
| Overlay ×20 warm | **avg 0.7 ms** |

**Conclusion:** blocking cost is almost entirely decode/seek. Overlay is cheap but
should still run off-GUI with decode so one worker returns a ready BGR image.
Random scrub on the NAS path currently freezes the GUI for seconds per tick.

---

## Proposed worker architecture

### Layers

| Layer | Mechanism |
| --- | --- |
| Identity | Frozen `FrameRequestIdentity` (`request_id`, `frame_generation`, `session_generation`, `run_id`, `frame_idx`, `pose_source`, `edit_revision`) |
| Snapshot in | Frozen request payload: video path, frame index, immutable `pose_points`, `skeleton_edges` |
| Snapshot out | Frozen `FrameRenderSnapshot`: identity + `image_bgr` (or failure) |
| Controller | Playhead/edit on GUI; `begin_frame_request` → worker render → `apply_frame_snapshot` |
| Runner | Page-owned `GuiTaskRunner` (`frame_runner`) + `GuiTaskKind.TRACKING_REVIEW_FRAME` |
| Serialization | One in-flight worker; newer requests bump generation; `on_finished` restarts for latest identity |

### Decoder ownership (recommended)

**Worker-exclusive sequential capture**, relocating the existing
`TrackingReviewController._read_frame` logic — not a new abstraction and not a
full switch to open-per-frame `PreparedFrameAccessor`.

Rationale:

1. Sequential capture is ~66 ms/frame; random `read_raw_frame_at_index` on this
   NAS video averages multi-second seeks. Playback needs the sequential path.
2. `PreparedFrameAccessor` is the right LRU idea for raw revisits, but its default
   reader is random-access and would regress continuous playback if used alone.
3. Compatible reuse: keep S1 `read_raw_frame_at_index` for non-sequential jumps
   (already true today); optionally add a small raw-frame LRU *inside* the
   worker-owned helper without publishing a second public video API.

Rules:

- Capture open/read/seek/release only on the frame worker thread.
- GUI thread never holds a live `VideoCapture` after this change.
- Session change / page teardown / app exit: bump `session_generation`, wait for
  runner idle (or schedule close on finished), then release capture.
- Add `TrackingReviewPage.has_running_task` and include it in `MainWindow.closeEvent`
  (today S3 is omitted from the close wait list).

### Pose snapshot rule (thread safety)

`pose_points_for_frame` is already an O(1) lookup. On the GUI thread, when
issuing a frame request:

1. Read immutable `tuple[PoseNodePoint, ...]` + `skeleton_edges` for the target
   frame / source.
2. Pass those into the worker.
3. Worker never reads `TrackingReviewSession`, widgets, or working-pose tables.

Edits continue to patch `_corrected_by_frame` on the GUI thread (existing code).
After an edit, bump `edit_revision` and re-issue a frame request for the current
playhead.

### Stale-result / scrub coalesce semantics

Mirror S10 decode:

1. Every scrub/step/play tick / source change / edit refresh calls
   `begin_frame_request()` → new `request_id` + bumped `frame_generation`.
2. If `frame_runner.is_running`: do **not** queue; store as `active_*` only.
3. On worker finished: if a newer active identity exists, start that one only.
4. `apply_frame_snapshot` accepts iff `request_id`, `frame_generation`,
   `session_generation`, `run_id`, `frame_idx`, `pose_source`, and
   `edit_revision` still match current controller intent.
5. Prefer the most recent requested frame; never apply an older generation after
   a newer one has been requested.
6. Playback remains wall-clock scheduled on the GUI timer (`PlaybackTimingState`);
   only composition moves async. Showing frame N after the playhead has moved to
   N+k is rejected by identity mismatch.

### Cache / edit invalidation

| Cache | Invalidate when |
| --- | --- |
| Worker sequential capture position | Non-sequential jump; session/run change |
| Optional raw LRU | Session/run/video path change (not on pose edits) |
| Composed overlay image | Never reuse across `edit_revision` or `pose_source` changes |
| Pose point tables | Existing edit/reset paths only |

Never display a composed frame whose `edit_revision` is older than the session’s
current revision.

### UI feedback (minimal)

- Keep scrubber, transport, and edit controls enabled while a frame is pending
  (GUI stays interactive).
- Unobtrusive pending indicator (status text or light progress on the video
  chrome) — no modal during playback/scrub.
- Metadata labels (frame index, corrections, acceptance) update immediately when
  the playhead changes; pixels catch up when the snapshot applies.
- Preserve synchronous `current_view()` for headless unit tests (compose on the
  calling thread), as Sprint 1 did for discovery collect/apply.

### Explicit non-goals

- No tracking algorithm / corrector changes
- No pose schema or scientific source changes
- No shared video-player rewrite / new public video framework
- No S2 validation refactor
- No S4–S8 science-task cancellation changes
- No S3 review UI redesign

---

## Proposed page flow

```text
User scrub / step / play tick / edit / source change
        ↓
Update playhead + cheap labels on GUI thread
        ↓
begin_frame_request()  (snapshot pose points + identity)
        ↓
GuiTaskRunner frame worker (if idle; else coalesce)
        ↓
decode (worker-owned capture) + draw_pose_overlay
        ↓
apply_frame_snapshot() if identity still current
        ↓
PoseVideoView.set_frame(image_bgr)
```

---

## Test plan (when implementing)

Focused suite (proposed): `tests/ui/test_tracking_review_video_responsiveness.py`
plus helpers in `tests/ui/_responsiveness_support.py`.

Coverage:

- Decode/render off GUI thread
- Rapid scrub with stale-frame rejection
- Playback ordering / seek-during-playback / pause-resume
- Session/run switch while decode pending
- Manual edit → revision bump → refresh
- Cache/composed-frame invalidation after edits
- Missing frame / decoder failure surfacing
- Worker/resource cleanup; no unsafe shutdown
- Pose/artifact immutability except explicit edits

Regressions: existing S3 review, manual edits, workspace, overlay-QC GUI tests.
Static: Ruff + targeted Mypy on touched modules.

---

## Performance plan (when implementing)

Repeat the baseline script with async path instrumentation:

- seek/decode time (worker)
- overlay time (worker)
- GUI present/repaint time
- perceived scrub latency (time-to-interactive vs time-to-final-pixels)
- cold vs warm; playback; cache/memory growth

Visible Windows GUI pass: rapid scrub + short playback on the same E-session run.
Document remaining jitter / frame-order anomalies.

---

## Open decisions for review

1. **Decoder:** approve worker-owned sequential capture (recommended) vs
   PreparedFrameAccessor-only (simpler, likely worse playback on NAS).
2. **Optional raw LRU** inside the worker helper (size 8, S10-like) — yes/no.
3. **Pending chrome:** status-bar text only vs small indeterminate indicator on
   the viewer.
4. **`current_view()`:** keep as synchronous test/API helper that still
   decodes inline, while the page uses async `begin_frame_request` /
   `apply_frame_snapshot` exclusively.
5. **Close/navigation:** treat in-flight S3 frame work like S10 soft-busy
   (wait on close; allow session browse to supersede via generation bump).

---

## Implementation sequence (after approval)

1. Add identity/snapshot types + controller begin/apply/worker APIs; relocate
   capture ownership off the GUI thread.
2. Wire `TrackingReviewPage` to `frame_runner` with latest-wins coalesce and
   minimal pending feedback; add `has_running_task` + MainWindow close wait.
3. Edit/source/session invalidation via `edit_revision` / `session_generation`.
4. Responsiveness tests + existing S3 regressions.
5. E-session timing audit + visible GUI note.
6. Finalize this roadmap doc and a short note in
   `docs/subsystem_03/mvp_scope_and_roadmap.md`.

---

## Required report fields (pre-implementation)

| Item | Status |
| --- | --- |
| 1. Files changed | None yet (design only) |
| 2. Original S3 blocking paths | Documented above |
| 3. Worker/decoder ownership design | Proposed above — needs approval |
| 4. Frame request / stale-result semantics | Proposed (S10 latest-wins) |
| 5. Edit/cache invalidation | `edit_revision` + no stale composed frames |
| 6. Before/after performance | Before only (table above); after TBD |
| 7. Visible GUI evidence | Not run (awaiting implementation) |
| 8. Tests/static checks | Not run |
| 9. Remaining defects / acceptance | N/A until implementation |

**Recommendation:** Approve the worker-owned sequential-capture + S10-style
latest-wins `GuiTaskRunner` design, then implement. Do not commit/push until
after implementation review.
