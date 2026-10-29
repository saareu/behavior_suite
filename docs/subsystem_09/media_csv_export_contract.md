# Subsystem 09 — Media / CSV Export Contract (S9-E2 Freeze)

**Status:** Frozen for headless implementation (S9-E2).  
**Authority:** Accepted architecture `media_csv_export_architecture.md`.  
**Non-goals:** GUI, GIF, framewise CSV, reviewed CSV, S10, mutation of S1–S9 science artifacts.

This document freezes identity, schemas, timing, lifecycle, and failure semantics. Implementation must match these rules.

---

## 1. Export-job identity and directory layout

### 1.1 Roots

| Root | Role |
| --- | --- |
| `{session_root}/results_package/results_package__{ts}/` | Authoritative S9 package (unchanged; never written by export) |
| `{session_root}/results_exports/` | Sibling root for optional export jobs |

### 1.2 Job kinds and IDs

| Kind | Directory prefix | Example |
| --- | --- | --- |
| Media (MP4) | `media_export__` | `media_export__20261029T153000` |
| Automated CSV | `csv_export__` | `csv_export__20261029T153001` |

- `export_job_id` == directory basename.
- New jobs always receive a fresh timestamped ID. Never reuse a completed job directory.
- Jobs never write into an existing completed S9 package directory.

### 1.3 Media job layout

```text
{session_root}/results_exports/media_export__{ts}/
  run_meta.json
  settings_used.yaml
  export_job_manifest.json
  qc.json
  processing_log.txt
  clip_index.parquet
  failures.jsonl                 # append-friendly per-clip failures
  clips/
    {sanitized_channel}/
      {sanitized_behavior_name}/
        {package_bout_id}.mp4
```

Clip files are finalized via temp path + atomic replace (`*.mp4.tmp` → `*.mp4`).

### 1.4 CSV job layout

```text
{session_root}/results_exports/csv_export__{ts}/
  run_meta.json
  settings_used.yaml
  export_job_manifest.json
  qc.json
  processing_log.txt
  behavior_bouts_automated.csv
```

### 1.5 Schema stamps

| Artifact | Schema version |
| --- | --- |
| Export settings | `results_export_settings_v1` |
| Export run meta | `results_export_run_meta_v1` |
| Export job manifest | `results_export_manifest_v1` |
| Export QC | `results_export_qc_v1` |
| Clip index | `results_export_clip_index_v1` |
| CSV companion provenance | embedded in CSV columns + manifest |

---

## 2. MP4 encoding policy and dependencies

| Setting | Frozen value |
| --- | --- |
| Container | MP4 |
| Video codec | `libx264` |
| Pixel format | `yuv420p` |
| CRF | configurable; default `22` |
| Audio | none |
| Playback rate flag | `-r <source_fps>` where `source_fps` is the **prepared-video / encode-source fps** |
| Seek / duration | Frame-addressed ffmpeg `trim=start_frame:end_frame` (inclusive prepared indices → exclusive end_frame) + `-frames:v clip_frame_count`; post-encode probe must match planned count and source fps |
| Executable | PATH `ffmpeg`, else suite `preprocess.ffmpeg_runtime.resolve_ffmpeg_binary` |
| GIF | out of scope |

**Dependency preflight (media):** ffmpeg callable; OpenCV probe of source video yields **positive fps** and **positive frame count**. If metadata cannot support a reliable clip timebase, the job **fails explicitly** (status `failed`) — never invent fps or frame counts.

**Encode source (MVP):** package `run_meta.source_video.path` (prepared video). Index space = prepared frames. Overlay sources deferred.

**Do not** copy the session prepared video into the export job.

---

## 3. Bout selection and profile semantics

Exactly one `selection_mode` per job:

| Mode | Behavior |
| --- | --- |
| `all_package_bouts` | Full packaged `behavior_bouts.parquet` |
| `channels_behaviors` | Allow-lists on `channels` and/or `behavior_names` (selection only) |
| `export_profile` | Rows whose `bout_id` appear in an **already packaged** profile index (e.g. `export_profiles/review_emit_v1/bout_index.parquet`). Profile must exist; floors are **not** recomputed |
| `bout_id_list` | Explicit package `bout_id` list |

Rules:

- Selection never re-collapses, re-arbitrates, or applies new emit floors.
- Preserve directional (`identity_grain=directional`, focal/partner) and symmetric (`symmetric_pair`, pair ids) identities from package rows.
- Deterministic row order: package sort key `(channel, start_sample_idx, behavior_name, pair_id, focal_id, partner_id)` then `bout_id`.
- Zero selected bouts → job may complete as empty (`completed`) with explicit QC note; no silent pretence of clips.

---

## 4. Context padding, clipping, and intervals

Three intervals are recorded separately:

| Interval | Definition | Mutable by padding? |
| --- | --- | --- |
| Behavioral | Inclusive dense `[start_sample_idx, end_sample_idx]` @ **120 Hz** from package | **No** |
| Prepared bout | Inclusive `[prepared_frame_idx_start, prepared_frame_idx_end]` from package | **No** |
| Exported clip | Inclusive prepared frames after pad + clamp | **Yes** (pad only) |

Defaults: `pre_context_sec = 0.0`, `post_context_sec = 0.0`.

Padding uses **source video fps** (prepared fps), not 120 Hz:

```text
pre_frames  = round(pre_context_sec  * source_fps)
post_frames = round(post_context_sec * source_fps)
clip_start  = max(0, prepared_frame_idx_start - pre_frames)
clip_end    = min(n_frames - 1, prepared_frame_idx_end + post_frames)
```

Context extends the **video** interval only. Behavioral dense endpoints and prepared-bout endpoints in the clip index remain the package values.

---

## 5. Dense ↔ prepared mapping

- Dense science timebase: **120 Hz** (`TARGET_FPS` from S8/S9).
- Prepared navigation columns on package bouts are seek aids (many dense samples may map to one prepared frame).
- Media encode targets **prepared** inclusive frames.
- Export does **not** recompute prepared endpoints unless validating against sync; default authority is package columns.
- Clip index always exposes both dense behavioral fields and prepared/clip fields. Never rename dense indices as `*_prepared_idx`.

---

## 6. Inclusive endpoints and actual exported counts

| Quantity | Formula |
| --- | --- |
| Behavioral frame count | `end_sample_idx - start_sample_idx + 1` (package; authoritative) |
| Prepared bout span | `prepared_frame_idx_end - prepared_frame_idx_start + 1` (navigation; may be ≤ dense count) |
| Clip frame count | `clip_frame_idx_end - clip_frame_idx_start + 1` |
| `clip_time_sec_start` | `clip_frame_idx_start / source_fps` |
| `clip_duration_sec` | `clip_frame_count / source_fps` |
| `clip_time_sec_end` | `clip_time_sec_start + clip_duration_sec` (exclusive-end time for media; equals `(clip_end + 1) / fps`) |

ffmpeg `-t` uses `clip_duration_sec`. Playback rate is source fps.

---

## 7. Source bout identities

Every clip-index / CSV row preserves:

- Package `bout_id`
- `source_s8_bout_id`, `source_s8_run_id`, `package_run_id`
- `channel`, `behavior_name`, `identity_grain`
- `focal_id`, `partner_id`, `mouse_a_id`, `mouse_b_id`, `pair_id`

Filenames use package `bout_id` under sanitized channel/behavior directories. Sanitized path tokens are recorded as `clip_relpath`; identity columns keep unsanitized scientific ids. Path traversal via bout ids is rejected (`..`, separators, absolute paths).

---

## 8. Clip index schema (`clip_index.parquet`)

One row per **attempted** selected bout. Column groups:

**Identity:** `bout_id`, `source_s8_bout_id`, `package_run_id`, `source_s8_run_id`, `channel`, `behavior_name`, `identity_grain`, `focal_id`, `partner_id`, `mouse_a_id`, `mouse_b_id`, `pair_id`

**Behavioral:** `start_sample_idx`, `end_sample_idx`, `start_time_sec`, `end_time_sec`, `bout_duration_frames`, `bout_duration_sec`

**Prepared bout:** `prepared_frame_idx_start`, `prepared_frame_idx_end`, `prepared_time_sec_start`, `prepared_time_sec_end`

**Exported clip:** `clip_frame_idx_start`, `clip_frame_idx_end`, `clip_frame_count`, `clip_time_sec_start`, `clip_time_sec_end`, `clip_duration_sec`, `pre_context_sec`, `post_context_sec`

**Media / status:** `source_video_kind`, `source_video_path`, `source_fps`, `source_n_frames`, `clip_relpath`, `clip_exists`, `encode_status`, `error`

**Provenance:** `export_job_id`, `selection_mode`, `crf`, `codec`, `schema_version`

`encode_status` ∈ {`succeeded`, `skipped_exists`, `failed`, `omitted_invalid_bounds`, `pending`}.

---

## 9. CSV schema and null representation

### 9.1 Default content

Export the complete packaged `behavior_bouts.parquet` table (all science + navigation + package stamp columns), plus export provenance columns.

Optional selection modes mirror §3 (subset of rows only; no new transformations).

### 9.2 Provenance columns (always appended)

| Column | Value |
| --- | --- |
| `export_provenance` | `automated` |
| `export_job_id` | job id |
| `selection_mode` | mode string |
| `source_package_run_id` | package run id |

### 9.3 Encoding

| Rule | Value |
| --- | --- |
| Encoding | UTF-8 with BOM optional **off**; plain UTF-8 |
| Header | yes, first row |
| Column order | stable: package column order as read, then provenance columns in the order above |
| Row order | same deterministic sort as §3 |
| Missing values | empty field (pandas default `na_rep=""`) |
| Booleans / numbers | pandas default string forms |
| Filename | `behavior_bouts_automated.csv` |

No framewise CSV. No reviewed labels.

---

## 10. Manifest, settings, QC, status

### 10.1 Lifecycle statuses

| Status | Meaning |
| --- | --- |
| `running` | Job directory created; work in progress |
| `incomplete` | Encode/CSV batch finished with recoverable unfinished/failed units; **not** fully completed |
| `completed` | All selected units succeeded (or empty selection explicitly accepted) |
| `completed_with_warnings` | Operator-accepted terminal partial success (`accept_partial=True`) or non-fatal QC warnings with zero pending units |
| `failed` | Preflight/provenance failure, or unrecoverable hard error before/without a usable partial ledger |

Historical status is written at finalize and not rewritten when sources later go stale (discovery surfaces staleness separately).

### 10.2 Settings snapshot (`settings_used.yaml`)

Captures: `job_kind`, source package id/path/digests, selection mode + parameters, media encode settings (crf, context, fail_fast, accept_partial), output root, schema versions.

### 10.3 `export_job_manifest.json`

Records: job identity/status/timestamps; source package digests; linked S8/S7/S6 ids from package meta; source video path + fingerprint; selection block; counts (`n_selected`, `n_succeeded`, `n_failed`, …); artifact inventory (CSV / clip_index hashed; individual MP4 hashing off by default — size/mtime listing).

### 10.4 `qc.json`

Structural checks, counts, warnings/failures, timing conventions note (`behavioral_fps=120`, `clip_playback_fps=source_fps`).

---

## 11. Partial-failure and retry semantics (media)

**Chosen model: same-job resume for recoverable jobs.**

| Rule | Detail |
| --- | --- |
| Per-clip ledger | `clip_index.parquet` + `failures.jsonl` |
| Failed clip | Does not delete or invalidate already successful MP4 files |
| Atomic finalize | Each MP4 written to `.tmp` then replaced |
| Batch terminal | All success → `completed`; any pending/failed remaining → `incomplete` (unless `accept_partial` → `completed_with_warnings`) |
| Retry | `retry_media_export(job_dir)` allowed only when status ∈ {`incomplete`, `failed`} with an existing clip ledger |
| Retry preflight | Re-validate source package digests, video reachability, and that settings selection/encode parameters match the job snapshot |
| Retry scope | Re-attempt rows with `encode_status` in {`failed`, `pending`, `omitted_invalid_bounds`} or `succeeded`/`skipped_exists` whose file is missing |
| Immutability | Never overwrite an existing successful MP4; `skipped_exists` if file present and status was success |
| Completed jobs | `completed` / `completed_with_warnings` are immutable; further exports require a **new** job id |

CSV jobs are atomic single-artifact publishes (no per-row resume). Failure leaves status `failed` without a completed CSV, or writes CSV only on success.

---

## 12. Provenance and eligibility

### 12.1 Export source validation

Before create/retry:

1. Package `status` ∈ {`completed`, `completed_with_warnings`}.
2. Required package artifacts present (`behavior_bouts.parquet`, `sync_index.parquet`, `package_manifest.json`, `run_meta.json`, `qc.json`).
3. Recorded package digests for bouts/sync/manifest match live files.
4. Media: prepared video path exists; probe fps/n_frames OK; fingerprint/digest check when recorded.
5. **MVP eligibility:** new create/retry exports require a **currently eligible** S9 package (`classification` ∈ {`eligible`, `completed_with_warnings`}). Do **not** silently export from `stale_source` packages, and do **not** substitute another package or video. Historical job discovery/reopen remains available for inspection when sources later go stale. Explicit archival export from stale packages is **deferred**.
6. Missing prepared video → block media export. Never rewrite S9 historical completion status.

### 12.2 Discovery classifications

Scan `results_exports/`. Classifications analogous to S9: eligible / completed_with_warnings / incomplete / failed_or_incomplete / stale_source / missing_required_artifacts.

---

## 13. Timing convention summary

| Concept | Timebase |
| --- | --- |
| Behavioral bout duration / dense indices | 120 Hz |
| Context padding → frame offsets | Source (prepared) video fps |
| Clip playback / `-r` / duration | Source (prepared) video fps |
| Unreliable source fps or frame count | Hard fail |

---

## 14. Performance / safety constraints

- Headless APIs suitable for background GUI tasks.
- Stream ffmpeg; do not load whole videos into memory.
- Sequential encode MVP; frame-addressed `trim=start_frame:end_frame` per clip with post-encode count/fps verification. Future: optional shared demuxer.
- Progress callback: `{done}/{total}` + current `bout_id`.
- Validate output paths; reject traversal.
- Do not change S9 package discovery or publication behavior.
