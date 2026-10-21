# Subsystem 05 — MVP Scope, Completion Status, and Roadmap

**Status:** MVP complete / lab-validated (Sprints 0–4 + laboratory remediation). Automated tests green. Final laboratory GUI acceptance passed.
**Date:** 2026-10-21
**Authority:** Repository implementation + Sprint 0–3 docs + Sprint 4 delivery + lab remediation + laboratory GUI acceptance.

---

## Final subsystem architecture

```text
MainWindow
  ├─ Group 1 — Data Preparation → S1–S4 (unchanged pages/stack indices)
  └─ Group 2 — Feature Processing → S5 (S6 button visible, disabled)
       └─ BlobFeaturesPage
            ├─ BlobFeaturesController (+ bounded preview cache)
            ├─ GuiTaskRunner (preview + progressive extraction)
            └─ blob_features.* (headless scientific core)
```

| Layer | Modules |
| --- | --- |
| Eligibility / timing | `s1_eligibility`, `timing` (S4-compatible 120 Hz grid) |
| Segmentation / features | `segmentation`, `features` |
| Export / QC / runner | `export`, `qc`, `runner` |
| Run discovery | `run_discovery` |
| GUI persistence | `bookmarks`, `gui_persistence` |
| UI | `BlobFeaturesController`, `BlobFeaturesPage`, `MainWindow` groups |

Upstream S1 artifacts are read-only. S2–S4 are never required for S5 readiness.

Scientific suite defaults (active): `blob_min_area = 2900`, `blob_post_prune_min_area = 2900`. Existing completed runs and explicit saved working settings are not rewritten when defaults change.

---

## Navigation behavior

- Two mutually exclusive group selectors reuse the existing `QStackedWidget`.
- Session root (`MainWindow._session_root`) is shared across both groups.
- Switching groups shows/hides subsystem buttons; pages are not destroyed.
- S5 is reachable whenever eligible S1 exists, independent of S2–S4.
- S6 is displayed as **coming later** and remains disabled (no functional page).
- **S2 / S4 tasks** still hard-block all group and subsystem navigation.
- **S5 tasks** do **not** hard-block Group 1 ↔ Group 2 or S1–S4 navigation. Extraction continues; progress remains available on return. Session mutation and duplicate S5 tasks remain page-locked.

---

## User workflow

1. Open Group 2 — Feature Processing (or open S5 with a shared session).
2. Select an eligible S1 session (`prepared_video.mp4`, `cropped_background.png`, `prepared_sync.npz`, `prepare_meta.json`).
3. Adjust segmentation / temporal / execution settings (suite YAML defaults).
4. Navigate prepared-video frames; **Preview Frame** / **Preview Bookmarks**.
5. **Run Full Extraction** — validates S1, freezes an immutable settings snapshot, creates `blob_features/blob_features__<timestamp>/`.
6. Observe prepared-frame progress and final status (may leave and return to Feature Processing while extraction runs).
7. Inspect QC summary and reopen prior runs from the discovery table.

Changing working settings never rewrites `settings_used.yaml` of an existing run.

Preview notes:

- Original / Background views display directly.
- Foreground / Binary Mask / Contours require a valid cached preview for the current frame; otherwise the UI shows **Preview required for this frame** (never a silent original-frame substitute).
- Preview Bookmarks processes every bookmark under one immutable settings snapshot, caches successes, and does not jump the selected frame.
- Full extraction does **not** store per-frame masks/overlays; on-demand preview remains available afterward using current working settings. Preview using a completed run’s immutable settings is a documented follow-up.

---

## Input / readiness

| Requirement | Rule |
| --- | --- |
| Eligible S1 | Required (stricter than S2 handoff: includes background PNG) |
| S2 / S3 / S4 | Not required |
| Target timeline | Fixed 120 Hz via `pose_finalization.timing` |

Source identity for GUI persistence / preview cache / eligibility includes prepared-video, prepared-sync, and cropped-background digests.

---

## Output and provenance contracts

```text
<session_root>/blob_features/<run_id>/
  blob_features.parquet
  run_meta.json          # source_s1 digests + status + QC summary
  settings_used.yaml     # immutable snapshot for this run
  processing_log.txt
  qc.json
```

Parquet columns (`BLOB_FEATURES_COLUMNS`): `sample_idx`, `dense_frame_idx`, `frame_idx`, `time_sec`, `orig_frame_idx`, plus legacy topology / morphology / temporal occlusion features. `cc_count_raw` is never modified by gap filling.

Statuses: `running` → `completed` | `completed_with_warnings` | `failed`. Partial parquet is deleted on failure and never marked completed.

---

## Run discovery / eligibility policy

Classifications (discovery only; historical `status` is not rewritten):

| Classification | Meaning |
| --- | --- |
| `eligible` | Completed, parquet present, recorded S1 digests still match |
| `completed_with_warnings` | Same as eligible, but QC warnings / warning status |
| `stale_source` | Completed historically, but S1 path/digest no longer matches |
| `failed_or_incomplete` | Failed, running, or non-completed |
| `missing_required_artifacts` | Missing / unreadable meta or parquet |

**Future S6 selection policy:** choose the newest currently eligible run (`eligible` or `completed_with_warnings`), ordered by `completed_at` / `created_at` / run-id timestamp. Never auto-select stale, failed, incomplete, or structurally invalid runs. (`find_latest_eligible_s5_run` implements this; S6 does not consume it yet.)

---

## QC and error behavior

- Structural QC failures block parquet publication.
- Quality warnings allow `completed_with_warnings`.
- GUI shows QC rows (status, output, classification, row counts, warnings/failures).
- Errors surface through `status_message` / `unexpected_error` and existing logging.
- Session change while an S5 task is active is blocked; stale async results are still rejected if source/settings diverge before accept.
- Application close waits for non-cancellable S5 work (same pattern as S2/S4).

**Cancellation:** not supported safely in the runner. No Cancel button. Documented for a later cooperative-cancellation improvement.

---

## Known limitations (non-MVP follow-ups)

1. No cooperative cancellation for full extraction.
2. Preview using a selected completed run’s immutable settings is not yet exposed in the GUI.
3. Prepared-video decode for preview remains per-request (results are cached; frames are not pre-decoded) — optional cache/performance improvements remain open.
4. Cable-artifact removal excluded.
5. No manual mask editing / mandatory acceptance workflow.
6. S6 Combined Features not implemented (next stage).
7. No numerical regression against legacy training datasets in this MVP.

---

## Automated test results

Executed for S5 MVP closure:

| Command | Result |
| --- | --- |
| `pytest tests/blob_features tests/ui/test_blob_features_gui.py` | **93 passed** |
| Selected S4 / pose UI / segmentation regressions | **32 passed** |
| `ruff check` on touched S5/UI modules | **All checks passed** |
| `mypy -p blob_features` | **Clean** (project-wide `mypy` may still report pre-existing errors outside S5) |

---

## Laboratory fixture verification (`E_full_session_integration`)

Inspected read-only (artifacts not modified; no new full extraction launched on the fixture for closure):

| Check | Result |
| --- | --- |
| Eligible S1 | Yes — `prepared_video.mp4`, `cropped_background.png`, `prepared_sync.npz`, `prepare_meta.json` |
| Frame geometry | 36239 prepared frames · 928×528 |
| Existing S5 run | `blob_features__20261021T152749` · `status=completed` · `classification=eligible` |
| Digests | Video / background / sync digests match `run_meta.source_s1` |
| `find_latest_eligible_s5_run` | Returns that run |
| Parquet | **36,528** samples at **120 Hz** · schema includes required columns |
| QC | `structural_ok=true`, `warning_count=0` |
| Settings snapshot | `blob_min_area` / `blob_post_prune_min_area` = 2900 |

---

## Final laboratory GUI acceptance (passed)

Representative laboratory session — all seven final manual acceptance checks **passed**:

| # | Check | Result |
| --- | --- | --- |
| 1 | Batch bookmark previews (all bookmarks under one settings snapshot; selection unchanged) | **Passed** |
| 2 | Preview-required states for Foreground / Mask / Contours when uncached | **Passed** |
| 3 | Cache restoration across frame navigation after successful previews | **Passed** |
| 4 | Settings invalidation until Preview Frame / Preview Bookmarks | **Passed** |
| 5 | Blob Area Filtering labels/tooltips and defaults 2900 / 2900 | **Passed** |
| 6 | Navigation to Group 1 / S1–S4 during S5 extraction; progress available on return | **Passed** |
| 7 | Persistence of working settings and bookmarks across application restart | **Passed** |

Supporting lab flows also confirmed: load eligible S1 without S2–S4; full extraction progress and QC summary; reopen completed runs with immutable `settings_used.yaml`; S1–S4 unaffected.

**Status:** S5 MVP is closed. Laboratory GUI acceptance of remediated Sprint 4 flows is complete.

---

## Sprint 4 remediation outcomes (retained)

| Issue | Fix delivered |
| --- | --- |
| Preview Bookmarks single-frame / jumped | Batch under one settings snapshot; cache every success; preserve selection |
| Processed views fell back to original | Explicit **Preview required for this frame** |
| Preview lost on frame change | Bounded in-memory cache keyed by S1 identity + frame + settings fingerprint |
| Confusing dual min-area controls | Both under **Blob Area Filtering** with stage labels + tooltips |
| Defaults still 3100 | Scientific defaults → 2900 / 2900 (saved settings / completed runs untouched) |
| Could not open S1–S4 during S5 extraction | S5 allows group/page navigation; session mutation and duplicate runs remain locked |

---

## Future S6 handoff boundary

S6 Combined Features (not in this MVP; upcoming stage) should:

- Discover currently eligible S5 runs only (`find_latest_eligible_s5_run` policy).
- Join on `dense_frame_idx` / 120 Hz `time_sec` with S4-compatible timelines.
- Treat stale S5 completions as historical artifacts, not live inputs.

Out of scope for S5 follow-ups unless defects are demonstrated: new segmentation algorithms, cable removal, model inference, legacy training-set numerical regression.

---

## Sprint history

| Sprint | Outcome |
| --- | --- |
| 0 | Repository audit / integration plan |
| 1 | Backend foundation (eligibility, timing, segmentation core) |
| 2 | Complete extraction (parquet, QC, runner) |
| 3 | Interactive segmentation GUI |
| 4 | Group navigation + full extraction GUI + discovery/persistence |
| 4 rem. | Lab blocker remediation: preview cache/batch, presentation, defaults/UI, navigation safety |
| Closure | Laboratory GUI acceptance passed; S5 MVP marked complete / lab-validated |
