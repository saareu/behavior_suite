# Subsystem 05 — Sprint 4 Full GUI Integration and MVP Acceptance

**Status:** Implemented + laboratory remediation applied + laboratory GUI acceptance passed. S5 MVP closed.
**Date:** 2026-10-21
**Authority:** Approved Sprint 4 scope + Sprint 0–3 docs + repository conventions (especially S4) + laboratory acceptance feedback remediation + final lab GUI acceptance.

See also: [MVP Scope and Roadmap](mvp_scope_and_roadmap.md).

---

## What landed

| Area | Change |
| --- | --- |
| Group navigation | `MainWindow` Group 1 (S1–S4) / Group 2 (S5 + disabled S6) |
| Full extraction GUI | `BlobFeaturesPage` + controller → `run_blob_feature_extraction` via `GuiTaskRunner.start_progressive` |
| Run discovery | `blob_features.run_discovery` with eligible / warnings / stale / failed / invalid |
| Persistence | Working settings + bookmarks keyed by S1 digests under `blob_features/_gui/` |
| Progress | Prepared-video frames processed via `OperationProgress` |
| Docs | MVP roadmap + this sprint note |

---

## Laboratory remediation (post Sprint 4 delivery)

Observed lab blockers and fixes:

| Issue | Root cause | Fix |
| --- | --- | --- |
| Preview Bookmarks appeared single-frame / jumped | Batch success path forced navigation to bookmark index 0; no per-frame cache restore | Process all bookmarks under one settings snapshot; cache every success; preserve selected frame; progressive batch progress + per-frame errors |
| Processed views silently showed original | `_display_current_view` fell back to `current_frame_bgr` | Explicit **Preview required for this frame** for Foreground / Mask / Contours when cache miss |
| Preview lost when changing frames | Only `last_single_preview` retained | Bounded in-memory cache keyed by S1 identity (video+background+sync digests), frame index, settings fingerprint |
| Confusing dual min-area controls | One in common, one in Advanced | Both under **Blob Area Filtering** with stage labels + tooltips; values remain distinct |
| Defaults still 3100 | Suite/dataclass defaults lagged lab tuning | Scientific defaults → `blob_min_area=2900`, `blob_post_prune_min_area=2900` (saved working settings / completed runs untouched) |
| Could not open S1–S4 during S5 extraction | `MainWindow` hard-locked all navigation on any S5 task | S2/S4 keep hard lock; S5 allows group/page navigation while protecting session mutation and duplicate runs |

### Task / navigation safety policy (current)

**Allowed while S5 preview or extraction runs**

- Switch Group 1 ↔ Group 2
- Open S1–S4 pages (S2/S4 must not themselves be running)
- Return to S5 and observe progress / final status
- Frame navigation and view-mode changes during **extraction** (not during an active preview worker)

**Blocked while S5 task runs**

- Starting another S5 preview or extraction
- Changing / browsing / clearing the S5 session source
- Resetting settings / refreshing discovery (page busy lock)
- Application close still waits for the non-cancellable task

**Unchanged hard locks**

- Any S2 or S4 running task still disables all group and subsystem navigation

### Completed-run preview (follow-up)

Full extraction stores numerical parquet + immutable `settings_used.yaml`; it does **not** precompute masks/overlays for every frame.

- Preview with **current working settings** is supported now (on-demand).
- Preview with a **selected completed run’s immutable settings** is feasible (load `settings_used.yaml` → temporary preview snapshot) without new segmentation code, but was **not** implemented in this remediation to avoid expanding the working-settings vs run-settings UX. Track as a small follow-up.

### Source identity / background

`S1SourceIdentity` and downstream eligibility now include `cropped_background_sha256`. Background-only changes invalidate preview caches and mark completed runs stale when the recorded digest no longer matches.

---

## Cancellation limitation

The extraction runner does not support cooperative cancellation. Closing the app waits for the task to finish. Partial parquet is removed on failure and never published as completed. A Cancel button was intentionally omitted.

---

## Manual acceptance

Final laboratory GUI acceptance passed. Evidence and the seven closed checks live in
`mvp_scope_and_roadmap.md` (batch bookmark previews, preview-required states,
cache restoration, settings invalidation, area filtering, navigation during
extraction, and persistence across application restart).
