# Subsystem 05 — Sprint 3 Interactive Video Segmentation GUI

**Status:** Implemented (GUI + diagnostic preview API). Awaiting review.
**Date:** 2026-10-21
**Authority:** Approved Sprint 3 scope + Sprint 0–2 docs + existing S1–S4 GUI patterns.

---

## What landed

| Area | Module / file | Role |
| --- | --- | --- |
| Diagnostic preview API | `blob_features.segmentation` / `preview` | Same scientific core as extraction; returns foreground / pre-morph / mask + topology |
| Bookmarks | `blob_features.bookmarks` | Lightweight persistence under `blob_features/_gui/` |
| Controller | `ui.controllers.blob_features_controller` | Eligibility, frame nav, settings snapshots, async preview identity |
| Page | `ui.pages.blob_features_page` | Interactive workspace + explicit Preview Frame |
| Entry point | `ui.main_window` | Flat S5 button (no Feature Processing group yet) |

No S6, no cable removal, no group-navigation redesign, no S1–S4 behavior changes beyond shared shell wiring for S5.

---

## GUI architecture and workflow

```text
MainWindow (flat S1–S5 buttons)
  └─ BlobFeaturesPage
       ├─ BlobFeaturesController (headless decisions / state)
       ├─ GuiTaskRunner (non-blocking preview)
       └─ VideoFrameView (prepared-frame display)
```

### Workflow

1. Open a project/session with eligible S1 (`prepared_video.mp4`, `cropped_background.png`, `prepared_sync.npz`, `prepare_meta.json`). S2–S4 are not required.
2. Browse prepared-video frames by **prepared-frame index** (timestamp from S1 sync). No silent 120 Hz resampling of the preview stream.
3. Adjust segmentation settings (suite YAML defaults). Changes invalidate preview snapshots but never overwrite a recorded extraction-settings snapshot.
4. Click **Preview Frame** (explicit). Worker calls `segment_frame_with_diagnostics` → same mask/topology as `segment_frame` / runner.
5. Inspect Original / Background / Foreground / Binary mask / Contours overlay plus `cc_count_raw`, effective `cc_count`, `min_inter_contour_dist`, warnings.
6. Optionally bookmark frames, then **Preview Bookmarks** under one settings snapshot (no temporal gap-fill on the isolated set).

### Settings state

| Snapshot | Meaning |
| --- | --- |
| Working settings | Live GUI values |
| Last preview settings | Snapshot used for the last accepted preview |
| Last extraction settings | Optional record of a completed extraction (not invalidated by working edits) |

Stale async results are rejected via request id + session path + settings fingerprint.

### Bookmark persistence

`\<session_root\>/blob_features/_gui/segmentation_bookmarks.json`

Diagnostic only; associated with the selected S1 source paths.

### Sprint 4 integration prerequisites

1. Introduce Group 1 (Data Preparation: S1–S4) vs Group 2 (Feature Processing: S5, later S6) without changing S5 page internals.
2. Move the flat S5 button into Group 2; keep stack widget + shared `_session_root`.
3. Extend navigation blocking already wired for S5 preview tasks.
4. Optionally discover completed S5 runs / staleness (out of Sprint 3 scope).
5. Wire full extraction launch from the same working-settings snapshot (runner already exists from Sprint 2).

---

## Backend checks performed

| Check | Result |
| --- | --- |
| Preview vs batch core | `segment_frame_with_diagnostics` reuses `_extract_robust_foreground_mask` / `extract_contour_topology`; masks and counts match `segment_frame` |
| Gap-fill max-gap + boundaries | Existing behavior preserved; leading/trailing NaNs not filled; over-long gaps skipped |
| `cc_count_raw` vs gap fill | Gap fill only writes `cc_count` and `min_inter_contour_dist`; raw counts unchanged (tests added) |

No scientific calculation changes beyond exposing intermediate diagnostic images already computed in the mask path.

---

## Known limitations

1. Flat S5 nav only — Feature Processing group is Sprint 4.
2. Full extraction is not launched from the GUI yet (preview only).
3. Preview decode is per-request (no frame cache).
4. Cable removal remains excluded.
5. Manual mask correction / mandatory acceptance workflow not implemented (out of scope).

---

## Manual validation remaining

1. Open a real lab session with eligible S1; scrub frames and confirm timestamps match prepared sync.
2. Preview Frame on contact / occlusion / fragmented moments; confirm overlays and diagnostics.
3. Change a slider, confirm preview marked stale until explicit Preview Frame.
4. Bookmark several frames, Preview Bookmarks, navigate results without losing working settings.
5. Switch sessions mid-preview and confirm stale results are discarded.
6. Confirm S1–S4 navigation still works and is blocked while an S5 preview runs.
