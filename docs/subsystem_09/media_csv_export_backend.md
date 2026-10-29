# Subsystem 09 — Media / CSV Export Backend (S9-E2)

**Status:** Headless backend implemented. Stop for architectural review before GUI (S9-E3).  
**Contracts:** [`media_csv_export_contract.md`](media_csv_export_contract.md) (frozen) · [`media_csv_export_architecture.md`](media_csv_export_architecture.md) (accepted audit).  
**Non-goals in this slice:** GUI controls, GIF, framewise CSV, reviewed CSV, S10, S1–S9 scientific changes, commit/push.

---

## 1. Backend modules and APIs

Package: `src/results_packaging/media_csv_export/`

| Module | Role |
| --- | --- |
| `contracts.py` | Schema stamps, statuses, clip-index columns, errors |
| `config.py` | `MediaExportSettings`, `CsvExportSettings`, `SelectionSpec` |
| `source.py` | Validate completed S9 package + prepared video for export |
| `selection.py` | Bout selection (all / channels+behaviors / packaged profile / bout ids) |
| `clip_bounds.py` | Inclusive prepared clip windows + context pad + clamp |
| `video.py` | OpenCV probe, ffmpeg resolve/encode (atomic `.mp4.tmp` + `-f mp4`) |
| `clip_index.py` | Pending clip-index construction |
| `csv_writer.py` | Automated bout CSV + provenance columns |
| `publication.py` | Immutable job dirs, encode loop, same-job retry, CSV publish |
| `discovery.py` | Scan `results_exports/`, reopen, staleness classification |
| `runner.py` | Headless entry: `run_media_export`, `retry_media_export`, `run_csv_export` |
| `results.py` | Result dataclasses |

Primary APIs:

```python
from results_packaging.media_csv_export import (
    MediaExportRequest,
    CsvExportRequest,
    SelectionSpec,
    run_media_export,
    retry_media_export,
    run_csv_export,
    summarize_export_project,
    reopen_export_job,
)
```

Jobs write under `{session_root}/results_exports/` only. Completed S9 packages are never mutated.

---

## 2. Export contracts (summary)

See the freeze document for full schemas. Highlights:

- Job IDs: `media_export__{YYYYMMDDTHHMMSS}`, `csv_export__{…}`
- Media artifacts: `clip_index.parquet`, `clips/{channel}/{behavior}/{bout_id}.mp4`, manifest/QC/meta
- CSV artifact: `behavior_bouts_automated.csv` + provenance columns
- Behavioral timebase remains **120 Hz**; clip padding/playback use **source video fps**
- Unreliable video fps/frame count → hard fail (no invented timing)

---

## 3. Job lifecycle and retry model

**Chosen model: same-job resume for recoverable media batches.**

| Status | Meaning |
| --- | --- |
| `running` | In progress |
| `incomplete` | Batch finished with failed/pending clips — **not** fully completed |
| `completed` | All selected clips succeeded (or empty selection) — immutable |
| `completed_with_warnings` | Operator `accept_partial=True` with failures frozen — immutable |
| `failed` | Preflight/hard failure without usable success set |

`retry_media_export(job_dir)`:

1. Allowed only for `incomplete` / `failed` / interrupted `running`
2. Re-validates source package digests + video
3. Re-encodes failed/pending/missing only
4. Never overwrites existing successful MP4 files (`skipped_exists`)

Completed jobs require a **new** job id for further exports.

CSV jobs are single-artifact publishes (success → `completed`, else `failed` without leaving a fake-complete CSV).

---

## 4. Video encoding and timing semantics

- Source: package `run_meta.source_video.path` (prepared video)
- Codec: `libx264`, CRF default 22, `yuv420p`, no audio
- ffmpeg: `-ss` / `-t` from inclusive prepared clip window; `-r source_fps`
- `clip_frame_count = end - start + 1`; `duration = count / source_fps`
- Context pad uses `round(sec * source_fps)` then clamps to `[0, n_frames-1]`
- Dense behavioral endpoints are recorded unchanged; padding does not edit science
- Atomic write: encode to `*.mp4.tmp` with explicit `-f mp4`, then replace

---

## 5. CSV schema and provenance

- Default: full packaged `behavior_bouts.parquet` column set
- Appended: `export_provenance=automated`, `export_job_id`, `selection_mode`, `source_package_run_id`
- UTF-8, header row, missing values as empty fields, deterministic sort
- Optional selection modes subset rows only — no bout transforms
- Framewise / reviewed CSV deferred

---

## 6. Provenance and eligibility

Preflight records and checks:

- Package run id/path; digests of bouts, sync, manifest
- Linked S8/S7/S6 ids from package meta
- Prepared video path + fingerprint/SHA when recorded
- Selection snapshot and counts

**MVP (S9-E3):** new create/retry exports require a currently eligible S9 package. Do not silently export from `stale_source` packages or substitute another package/video. Historical job discovery/reopen still surfaces completed exports when sources later go stale. Explicit archival export from stale packages is deferred. Missing prepared video blocks media.

Discovery classifications distinguish historical job status from current source staleness.

---

## 7. Test results

| Suite | Result |
| --- | --- |
| `tests/results_packaging/test_media_csv_export_unit.py` | **8 passed** |
| `tests/results_packaging/test_media_csv_export_jobs.py` | **8 passed** (synthetic MP4 via OpenCV/ffmpeg) |
| `tests/results_packaging` (non E-session) | **67 passed** |
| E-session packaging smoke tests | **13 errors** (pre-E3) — environmental: S8→S7 ineligible because legacy model paths under `legacy/models/...` are missing; tests now skip when S8 is not currently eligible (not an export-code defect) |
| Ruff (`media_csv_export` + new tests) | **clean** |
| Mypy (`media_csv_export`) | **clean** |

Covered: selection modes, filenames/path safety, inclusive bounds, edge clamping, many-to-one dense/prepared, directional vs symmetric, CSV lossless + provenance, encoder failure + tmp cleanup, same-job retry, stale digest rejection, immutable outputs, no S1–S9 writes, discovery/reopen.

---

## 8. E-session validation

Accepted package: `results_package__20261028T181714`

| Job | Result |
| --- | --- |
| `csv_export__20261029T153000` | **609/609** automated rows; provenance columns present; package bytes unchanged |
| `media_export__20261029T153001` | **4/4** representative clips (video-start, video-end, short contact, SBS; start bout also covers short directional) with 0.1 s context |

Encoded frame counts matched planned inclusive counts exactly; playback fps matched source prepared fps (~119.048). Report: `test_cases/E_full_session_integration/results_exports/_s9e2_validation_report.json`. Reproducible helper: `scripts/_s9e2_e_session_validate.py`.

Upstream package currently reports stale S8/S7 eligibility (missing legacy model artifacts). Under S9-E3 MVP policy, **new** GUI/headless exports from that package are blocked until eligibility is restored; prior S9-E2 jobs remain inspectable under `results_exports/`.

---

## 9. GUI integration prerequisites (S9-E3)

1. Add Group 4 page controls distinct from packaging: Export CSV / Export Videos.
2. Require selected completed S9 package before enabling export actions.
3. Selection UI: all / channels+behaviors / existing packaged profile / explicit ids.
4. Media settings: CRF, pre/post context, accept_partial, fail_fast.
5. New `GuiTaskKind`s (e.g. `RESULTS_MEDIA_EXPORT`, `RESULTS_CSV_EXPORT`) with mutual exclusion vs packaging task.
6. Progress binding to `OperationProgress` (`done/total` + bout id).
7. Discovery table for `results_exports/` jobs; open folder; show incomplete + retry affordance.
8. Do not gate packaging success or S10 readiness on export jobs.
9. Surface stale-source warnings without rewriting package status.

---

## 10. Open issues / follow-ups

- Overlay encode sources deferred (prepared-only MVP).
- Parallel ffmpeg workers deferred.
- Per-clip SHA256 hashing off by default.
- Encode uses frame-addressed ffmpeg `trim=start_frame:end_frame` with post-encode frame-count/fps verification (no silent boundary approximation).
- E-session upstream S7 model paths must be restored before packaging smoke or new exports against historical packages become eligible again.
- GIF / framewise CSV / reviewed CSV remain out of scope.
