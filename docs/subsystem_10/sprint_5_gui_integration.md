# Subsystem 10 — Sprint 5: Integrated Review GUI and Laboratory Validation

**Status:** Implemented (Group 4 GUI + tests + operator checklist). Stop for architectural review / operator lab acceptance.  
**Date:** 2026-10-29  
**Authority:** Accepted S10-B (`implementation_contract.md`) + Sprint 1–4 headless APIs.  
**Non-scope:** Reviewed CSV/media exports (U6), reciprocal editing (U2), `uncertain` (U3), S1–S9 science mutation, commit/push.

---

## 1. Deliverables

| Area | Implementation |
| --- | --- |
| Group 4 navigation | `s10_button` beside S9; `_last_results_page` restore; soft-nav (not S2/S4 hard lock) |
| Page / controller | `behavioral_review_page.py`, `behavioral_review_controller.py` |
| Timeline widget | `ethogram_timeline_view.py` (dense playhead authority) |
| Video | `PoseVideoView` + async `PreparedFrameAccessor` decode |
| Task kinds | `BEHAVIORAL_REVIEW_{DISCOVERY,OPEN,DECODE,WINDOW,EDIT,PUBLISH}` |
| Docs | This report + `laboratory_acceptance_checklist.md` |
| Tests | `tests/ui/test_behavioral_review_gui.py`, `tests/ui/test_behavioral_review_edit_worker.py`, `tests/behavioral_review/test_e_session_sprint5.py` |

---

## 2. GUI architecture

```text
MainWindow (Group 4)
  ├─ ResultsPackagingPage (S9) — packaging / export only
  └─ BehavioralReviewPage (S10)
       ├─ BehavioralReviewController  (UI state + stale identities)
       │    └─ EditingSession / InspectionSession (headless science)
       ├─ GuiTaskRunner ×6 (discovery, open, decode, window, edit, publish)
       ├─ PoseVideoView
       └─ EthogramTimelineView
```

- All corrections/status/notes/publish call `EditingSession` — no Qt-side science.
- Discovery uses S9 `summarize_results_package_project` + S10 `summarize_behavioral_review_project`.
- Opening a package never creates drafts, accepts bouts, or publishes (R10).
- Mutations use `begin_*` → `edit_worker` (`BEHAVIORAL_REVIEW_EDIT`) → `apply_edit_snapshot`.

---

## 3. Inspection and synchronization

- Dense `sample_idx` is playhead authority; prepared decode uses `sync_index.prepared_frame_idx`.
- Dual clock labels: analysis time/sample vs prepared frame/time.
- Plateaus: dense advances; pixmap kept when prepared index unchanged (not a sync error).
- Dual exclusive ethogram lanes + independent contact/SBS lanes via `timeline_view`.
- Bout list filters are display-only; prev/next/start/end navigation use packaged seeks.
- S8 echoed evidence by default; optional S7 window when requested.
- Explicit automated vs reviewed analysis-source picker (never silent auto-pick).

---

## 4. Editing interactions

Default = inspection. Editing requires:

1. Non-empty reviewer identity  
2. Explicit **Enter editing mode**  
3. **Create draft** (or resume draft with lock)  

Exposed ops (all via controller → `EditingSession`):

- Directional exclusive label correction  
- Contact / SBS parallel pred correction  
- Bout trim / split (merge available via headless API)  
- Accept / reject / unreviewed (reject status-only)  
- Notes  
- Undo / redo  
- Draft recovery after restart  

Geometry-changing edits void affected prior accept/reject decisions (backend).  
Published reviewed runs open inspection-only; new edits require a new working draft.

---

## 5. Background task safety

| Worker | Work | Stale rejection |
| --- | --- | --- |
| Discovery | S9 + S10 project summarize | `DiscoveryRequestIdentity` |
| Open | `open_editing_session` (+ draft resume) | `OpenSourceIdentity` |
| Decode | prepared-frame decode | `DecodeRequestIdentity` (sample + prepared + package) |
| Window | ethogram/evidence/bouts | `WindowRequestIdentity` |
| Edit | draft create / corrections / undo-redo / status / notes | `EditRequestIdentity` |
| Publish | `publish_reviewed_run` | `PublishRequestIdentity` |

### Edit-worker lifecycle

1. GUI thread: `begin_*` validates gates, captures kwargs, sets `edit_running`, returns identity promptly.
2. Worker thread: `edit_worker` calls the existing headless `EditingSession` API (append-only journal + projection rebuild). One active mutation per working draft.
3. GUI thread: `apply_edit_snapshot` rejects stale session/source/draft/request identities. A rejected UI callback never rolls back a durable journal commit.
4. Mutual exclusion: edit blocks competing edits, undo/redo, publication, and source/draft changes. Scrub/play remain usable; window refresh is deferred until the mutation finishes.
5. On return to the original draft after navigation, refresh from the authoritative working state (`refresh_after_stale_edit` / reopen draft).

No video hashing on playback poll timers. Playback advances dense samples; decode coalesces to latest prepared index.

---

## 6. Publication and reopening

**Publish Reviewed Results** shows preview: parent package, reviewer, corrected intervals, accept/reject/unreviewed counts, QC/eligibility blockers, output version id.

Hard block when parent stale, draft/cursor invalid, writer conflict, or structural QC fails (U12).

Discovery lists drafts, eligible/completed reviewed runs, superseded classifications. Historical `run_meta.status` is never rewritten. Inspect published without mutation.

---

## 7. Tests and static checks

| Suite | Focus |
| --- | --- |
| `tests/ui/test_behavioral_review_gui.py` | Group 4 nav, session, discovery, sync/plateau, lanes, bout nav, stale rejection, edit gates, corrections, undo/redo, draft reopen, publish block/success, automated vs reviewed |
| `tests/ui/test_behavioral_review_edit_worker.py` | Nonblocking edit launch, mutation serialization, duplicate prevention, stale callback rejection, navigation during edit, publish/edit exclusion, journal durability, undo/redo, crash recovery, GUI refresh |
| `tests/behavioral_review/test_e_session_sprint5.py` | E-session counts (36528 / 609), sync, controlled edit+publish+reopen |
| Existing S10-1…4 | Regression |
| S9 GUI | Updated Group 4 assertion to expect `s10_button` |

Run: focused S10 + UI behavioral review, relevant S9/S8, ruff, mypy (see Sprint 5 validation log in final report).

---

## 8. E-session validation

Package: `results_package__20261028T181714`

| Check | Expectation |
| --- | --- |
| Dense samples | 36 528 |
| Automated bouts | 609 |
| Directions | Both exclusive directions present |
| Contact/SBS | Independent channels in bout table / lanes |
| Sync | Dual clocks; plateau keep-frame |
| Controlled draft | Isolated under `tmp_s10_sprint5_review_gui/` |
| Upstream | S8/S9 fingerprints unchanged |

Controlled edits are labeled non-biological.

---

## 9. Performance notes

- Windowed ethogram/evidence (not full S7 load).  
- Frame LRU cache on prepared accessor.  
- Timeline LOD via headless `TimelineView`.  
- Soft scrub budget asserted in E-session test (< 90 s for seven seeks+windows).
- Edit mutations no longer rebuild projections on the Qt GUI thread.

### Before / after (E-session `results_package__20261028T181714`)

| Operation | Before (GUI-thread stall) | After (GUI begin / worker) |
| --- | --- | --- |
| Create draft + projection | ~1.8 s on GUI thread | begin returns promptly; rebuild in `BEHAVIORAL_REVIEW_EDIT` worker |
| Exclusive / contact-SBS correction | ~1–2 s on GUI thread | begin prompt; projection in worker |
| Undo / redo | ~1–2 s on GUI thread | begin prompt; projection in worker |
| Bout trim/split / status | GUI-blocking rebuild | same worker path |
| Seek / scrub / decode / window | already nonblocking (prior remediation) | unchanged |

Measure with `scripts/_s10_gui_responsiveness_audit.py`. Exact workstation numbers go in the laboratory checklist.

---

## 10. Outstanding operator acceptance

S10 is **not** marked lab-validated until the operator completes:

[`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md)

---

## 11. Stop point

Sprint 5 implementation is merged for operator laboratory acceptance. Operator lab acceptance remains open — do not mark S10 lab-validated until the checklist passes.
