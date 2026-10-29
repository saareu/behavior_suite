# Subsystem 09 — Media and CSV Export Architecture Audit (S9-E1)

**Status:** Discovery milestone (documentation only). Stop for architectural review before implementation.  
**Date:** 2026-10-29  
**Scope:** Inventory legacy Pipeline 110 media generation; propose optional MP4 and automated CSV export operations over **existing eligible S9 packages**; recommend immutable export-job layout; define GUI boundary and S10 handoff. **No code, clips, CSV files, commits, or pushes in this sprint.**  
**Authority:** Local git-ignored `legacy/pipelines/110_bout_browser_packager.py` + `legacy/configs/bout_browser.yaml`; tracked S9 `src/results_packaging/`; Group 4 GUI under `src/ui/`; E-session package `results_package__20261028T183759` (609 bouts; prepared video referenced, not copied).

**Preconditions (operator-stated):** S9 core MVP has passed laboratory acceptance. This extension is optional before final S9 closure and must not rewrite scientific packaging contracts.

**Legend:**  
- **Fact** — observed in repository / local legacy / published S9 artifacts.  
- **Recommendation** — design proposal pending review; not binding.

---

## 0. Product boundary for this extension

| Operation | Role | Mutates S8 / S9 science? |
| --- | --- | --- |
| **S9 packaging** (already shipped) | Immutable analysis package: sync index, full bouts, summaries, optional `review_emit_v1` index | No (lossless annotated copy + references) |
| **S9 media export** (proposed, optional) | Batch MP4 clips from an eligible package + prepared video | No — writes a separate export job |
| **S9 CSV export** (proposed, optional) | Automated bout-table CSV from an eligible package | No — writes a separate export job |
| **S10 reviewed CSV / media** | After manual review | Out of scope here |

**Fact / product rule:** Media generation must never become a prerequisite for packaging eligibility or for S10 review. S10 opens packages via sync index + prepared-video seek whether or not clips exist.

---

## 1. Legacy Pipeline 110 media inventory

### 1.1 Purpose and entry points (fact)

Pipeline 110 (`legacy/pipelines/110_bout_browser_packager.py`, ~1481 lines) converts **Pipeline 100 framewise** predictions into a review bout index plus optional MP4 clips and GIF previews. Entry: `run_bout_browser_packager()` → per-experiment `export_experiment()`. Config: `legacy/configs/bout_browser.yaml`.

It is **not** an S9 package consumer. It reloads predictions, re-collapses contiguous positives into bouts, applies emit floors, then optionally extracts media. Scientific packaging and media generation are fused into one pipeline — a coupling S9 deliberately broke.

### 1.2 Libraries and media workflow (fact)

| Concern | Implementation |
| --- | --- |
| Video probe | OpenCV `cv2.VideoCapture` (`_probe_video`) — width, height, fps, `CAP_PROP_FRAME_COUNT` |
| Clip / GIF encode | External `ffmpeg` via `subprocess` (`_ffmpeg_executable`: PATH first, else `imageio_ffmpeg.get_ffmpeg_exe()`) |
| MP4 codec | `libx264`, `-crf` (default 22), `-pix_fmt yuv420p`, optional `-r` source fps |
| GIF | ffmpeg filtergraph: `fps`, `scale`, `palettegen`/`paletteuse` (Bayer dither); write `.tmp.gif` then `_replace_with_retries` |
| Progress | `tqdm` per clip / preview |
| Atomicity | GIFs use temp + `os.replace` with PermissionError retries; MP4 written directly to final path (`-y`) |

**MP4 command shape (fact):**

```text
ffmpeg -y -ss <clip_time_sec_start> -t <clip_frame_count / source_fps>
       -i <source_video>
       -c:v libx264 -crf <crf> -pix_fmt yuv420p [-r <source_fps>]
       <clips/{presentation_label}/{bout_id}.mp4>
```

Seek uses **time** (`-ss` / `-t`), not frame-accurate `-vf select`. Frame indices in the manifest are converted to seconds before ffmpeg.

### 1.3 Video input, seeking, and clip boundaries (fact)

**Source selection.** `source_video_priority` (default): `tracks_overlay` → `prepared_video` → `overlay_imputed` → `pose_overlay`. First existing candidate wins. OpenCV probe must yield positive fps and frame count or the run fails.

**Timeline maps.**

1. Dense (120 Hz imputed) indices from predictions: `dense_frame_idx`.
2. Dense → prepared: NPZ `dense_pose_120_imputed_npz` key `orig_frame_idx` (`_load_dense_imputed_map`).
3. Prepared → raw/original: `raw_frame_idx = trim.start_frame + prepared_frame_idx` from prepared-video metadata (`_load_timing_mapping`).

**Index-kind branching for clip endpoints** (`_build_manifest`):

| Source kind | Clip frame indices | Time base |
| --- | --- | --- |
| Dense-indexed (`overlay_imputed`, `behavioral_overlay`) | Dense start/end ± pad, clamped to `[0, n_frames-1]` | `source_video.fps` (may be 120) |
| Prepared-indexed (`tracks_overlay`, `prepared_video`, `pose_overlay`) | Mapped prepared start/end ± pad | `source_video.fps` |

**Padding.** `clip_padding_seconds` (YAML default `0.0`) → `pad_frames = round(pad_sec * source_video.fps)` applied in the **source index space** chosen above.

**Inclusive endpoints (critical).**

- Bout dense interval is inclusive: `start_idx`…`end_idx`; `bout_frame_count = len(segment)`.
- Clip inclusive source frames: `clip_start_*`…`clip_end_*`; `clip_frame_count = end - start + 1`.
- ffmpeg duration uses half-open time: `clip_time_sec_end = (clip_end_idx + 1) / fps` conceptually via `duration = clip_frame_count / fps` after `-ss clip_time_sec_start`.
- Legacy also stores `prepared_time_sec_end = (prepared_end_idx + 1) / source_fps` for the **unpadded** bout mapped prepared endpoints — i.e. exclusive-end time semantics for media, not “timestamp of last inclusive sample.”

### 1.4 Dense ↔ prepared conversion and naming trap (fact)

| Stage | What indices actually are |
| --- | --- |
| `_build_bouts` columns `start_prepared_idx` / `end_prepared_idx` | **Dense** `dense_frame_idx` min/max (misnamed) |
| Internal `prepared_start_idx` / `prepared_end_idx` | True prepared frames via `orig_frame_idx[dense]` |
| Manifest `start_prepared_idx` / `end_prepared_idx` | Still the **dense** values |
| Manifest `clip_start_prepared_idx` / `clip_end_prepared_idx` | Source-timeline clip bounds (dense **or** prepared depending on video kind) |

**Fact:** Dense and prepared lengths differ on the E-session (dense 0…36527 vs prepared/orig up to ~36237). Many dense samples can share one prepared frame. S9 already fixed naming (`start_sample_idx` / `end_sample_idx` vs `prepared_frame_idx_*`) and must not reintroduce the legacy misnomer into export jobs.

### 1.5 Clip naming, metadata, output structure (fact)

Output root: `reports/bout_browser/{exp}/{iteration}/`.

| Artifact | Content |
| --- | --- |
| `bout_index.parquet` | Full `MANIFEST_COLUMNS` (ids, identities, dense-as-misnamed-prepared, clip bounds, times, paths, existence flags) |
| `summary.json` | Counts, config snapshot, clip/preview errors, validation status |
| `source_manifest.json` | Source video probe + candidates + timing provenance |
| `clips/{presentation_label}/{bout_id}.mp4` | Optional |
| `gif_previews/{presentation_label}/{bout_id}.gif` | Optional |

**Bout id:** `{exp}__{presentation_label}_{dense_start}_{dense_end}` (not S8/S9 `bout_*` ids).

### 1.6 Bout selection / filtering (fact)

Before media:

1. Frame-bound filter vs dense map / prepared length (any drop → hard fail).
2. Label allow-lists (symmetric / focal; contact off by default; engaged optionally forced into focal).
3. Optional SBS supplemental rows from `pred_side_by_side`.
4. Contiguous collapse per presentation identity.
5. **Emit floors:** `min_required = max(min_emit_bout_frames, ceil(min_emit_bout_seconds * source_video.fps))` — uses **source video fps**, not forced 120 Hz.

**Deliberate S9 difference already shipped:** `review_emit_v1` floors use explicit **120 Hz** and never mutate the full bout table. Media export must consume S9 package / profile rows, not re-collapse Pipeline 100 predictions.

### 1.7 Performance, storage, failure handling (fact)

| Topic | Legacy behavior |
| --- | --- |
| Throughput | Sequential ffmpeg per bout; tqdm only |
| Overwrite | `clips.overwrite` / `previews.overwrite` (default false) — existing files skipped and marked exists |
| Fail policy | `fail_on_error` (clips default **true**, previews default **false**); errors listed in `summary.json` then optional raise after writing manifests |
| Storage | Full-session clip trees under `reports/`; source video never copied into export root |
| Retry | GIF replace retries only; no per-clip encode retry; no resume ledger beyond “file exists && !overwrite” |

### 1.8 Reusable vs incompatible pieces

**Reusable ideas / patterns (recommendation to re-encode, not import legacy):**

- ffmpeg availability resolution (PATH / imageio-ffmpeg).
- Inclusive source-frame count → duration for `-t`.
- Per-bout error list + summary QC without aborting mid-write of the index.
- Skip-if-exists for resumable batch extraction.
- Reference source video in place (do not duplicate 417 MB prepared MP4).

**Do not reuse as-is:**

- Prediction reload / re-bout / emit floors inside the media job.
- Misnamed `*_prepared_idx` dense columns.
- Default preference for tracks overlay over prepared video for S9/S10 navigation.
- GIF as a required sibling of MP4.
- Bout ids based on presentation label + dense endpoints instead of S8/S9 ids.
- Seconds→frames using source-video fps for **behavioral** selection (S9 already fixed this for `review_emit_v1`).

**Prefer tracked S9 modules for export implementation:** `bout_mapping.py`, `sync_index` / package bouts, `export_profiles.py` (selection views), `eligibility` / `run_discovery`, publication lifecycle patterns, Group 4 `GuiTaskRunner`.

---

## 2. Proposed MP4 export contract

### 2.1 Operation model (recommendation)

**Name:** `media_export` (optional S9 extension).  
**Input:** One **currently eligible** completed S9 package (`package_manifest.json` + embedded bouts/sync + `run_meta.source_video`).  
**Output:** A new **immutable export-job directory** (see §4) containing clips + clip index + job QC.  
**Does not:** edit S8/S9 Parquet/JSON; re-arbitrate; re-collapse; apply new scientific thresholds.

Primary format: **MP4** (`libx264`). **GIF deferred** for initial implementation (see §2.7).

### 2.2 Bout selection modes (recommendation)

Exactly one selection mode per job (snapshot in settings):

| Mode | Source rows |
| --- | --- |
| `all_package_bouts` | Full packaged `behavior_bouts.parquet` |
| `channels_behaviors` | Subset by explicit `channels` and/or `behavior_names` allow-lists (selection only) |
| `export_profile` | Existing packaged profile index (e.g. `review_emit_v1/bout_index.parquet`) — **require the profile already exists in the package**; do not recompute floors inside the media job unless an explicit future “ad-hoc profile” is approved |
| `bout_id_list` | Explicit package `bout_id` list (operator/API) |

Preserve for every exported clip:

- Package `bout_id`
- `source_s8_bout_id`
- `source_s8_run_id` / `package_run_id`
- `channel`, `behavior_name`, `identity_grain`, focal/partner or pair ids

### 2.3 Behavioral interval vs exported clip interval (recommendation)

**Support configurable pre/post context.** Default both to `0.0` seconds (legacy default; exact bout mapping).

| Interval | Definition | Authority |
| --- | --- | --- |
| **Behavioral interval** | Inclusive dense `[start_sample_idx, end_sample_idx]` @ 120 Hz; durations from package | Unchanged S8/S9 science |
| **Mapped prepared bout span** | Inclusive `[prepared_frame_idx_start, prepared_frame_idx_end]` from package navigation columns (or recomputed identically via `map_bout_boundaries_to_prepared_frames`) | Seek aids only |
| **Exported clip interval** | Inclusive prepared-video frames after context pad + clamp to `[0, n_prepared_frames-1]` | **What was actually encoded** — must be recorded even if padding is zero |

**Why not “exact dense frames” as the encode domain?** Prepared video is the authoritative playback file for S9/S10. Dense samples are many-to-one onto prepared frames (E-session QC already notes multiplicity ≤ 2). Encoding must target prepared (or explicitly chosen overlay) frames, while still citing dense science endpoints.

**Context application (recommendation):** convert `pre_context_sec` / `post_context_sec` using **source video fps** (prepared fps), not 120 Hz:

```text
clip_start = max(0, prepared_frame_idx_start - round(pre_context_sec * source_fps))
clip_end   = min(n_frames - 1, prepared_frame_idx_end + round(post_context_sec * source_fps))
duration_sec = (clip_end - clip_start + 1) / source_fps
start_sec    = clip_start / source_fps   # or prepared_time at clip_start if PTS map exists
```

Distinguish in the clip index:

- `behavior_*` columns (dense science)
- `prepared_bout_*` columns (unpadded mapped bout)
- `clip_*` columns (actual exported inclusive frames + times + duration)

### 2.4 Source video policy (recommendation)

| Priority | Rule |
| --- | --- |
| Default | Package `run_meta.source_video.path` — S1 `prepared_video.mp4` referenced by the eligible package |
| Optional override | Explicit path + kind in job settings (e.g. tracks overlay) **only if** frame index space is declared (`prepared` vs `dense`) and validated against clip bounds |
| Never | Copy full session video into the export job |

Validate before encode: file exists; probe fps/n_frames; optional digest check against package-recorded video fingerprint when present.

### 2.5 Clip naming and layout (recommendation)

```text
clips/
  {channel}/
    {sanitized_behavior_name}/
      {package_bout_id}.mp4
```

- One file per selected bout; filename uses **package `bout_id`** (already globally unique within a package).
- Sanitize path tokens; keep `bout_id` intact in the clip-index column even if filesystem sanitization differs (record both `clip_relpath` and ids).

### 2.6 Clip index schema (recommendation)

`clip_index.parquet` — one row per **attempted** bout (success and failure), minimum fields:

| Group | Columns |
| --- | --- |
| Identity | `bout_id`, `source_s8_bout_id`, `package_run_id`, `source_s8_run_id`, `channel`, `behavior_name`, `identity_grain`, focal/partner/pair fields |
| Behavioral | `start_sample_idx`, `end_sample_idx`, `start_time_sec`, `end_time_sec`, `bout_duration_frames`, `bout_duration_sec` |
| Prepared bout | `prepared_frame_idx_start`, `prepared_frame_idx_end`, `prepared_time_sec_start`, `prepared_time_sec_end` |
| Exported clip | `clip_frame_idx_start`, `clip_frame_idx_end`, `clip_frame_count`, `clip_time_sec_start`, `clip_time_sec_end`, `clip_duration_sec`, `pre_context_sec`, `post_context_sec` |
| Media | `source_video_kind`, `source_video_path`, `source_fps`, `source_n_frames`, `clip_relpath`, `clip_exists`, `encode_status`, `error` |
| Provenance | `export_job_id`, `selection_mode`, `crf`, `codec` |

### 2.7 GIF decision (recommendation)

**Defer GIFs** from the initial media-export MVP.

Justification: MP4 covers offline sharing and S10 optional clip open; dual encode roughly doubles wall time and disk; legacy already treated GIF failures as soft; S10 can synthesize previews lazily if needed. Revisit only if operators require lightweight email/browser previews without video players.

### 2.8 Batch scale, partial failure, retry (recommendation)

Assume **hundreds–thousands** of clips (E-session full table ≈ 609; review_emit ≈ 514; larger sessions possible).

| Concern | Policy |
| --- | --- |
| Execution | Background GUI/CLI task; progress = `{done}/{total}` bouts + current bout id |
| Parallelism | Start **sequential** (match legacy predictability); optional worker pool later behind a setting |
| Partial failure | Continue after encode errors; record per-bout `encode_status` ∈ {`succeeded`,`skipped_exists`,`failed`,`omitted_invalid_bounds`} |
| Job status | `completed` (all succeeded/skipped), `completed_with_warnings` (any failures/skips of interest), `failed` (preflight/provenance failure before encode, or fail-fast setting) |
| Retry | New export job **or** resume mode on same incomplete job directory with `overwrite=false` (skip existing good clips) + optional `retry_failed_only=true` |
| Cancellation | Follow S9 packaging precedent initially (no cancel mid-publish); if cancel is added later, never mark job `completed` with a partial silent tree — leave `failed`/`running` cleared to failed |

---

## 3. Proposed CSV export contract

### 3.1 Operation model (recommendation)

**Name:** `csv_export` (optional S9 extension).  
**Input:** Eligible S9 package.  
**Output:** Immutable CSV export-job directory (may share the export-job architecture with media, different `job_kind`).  
**Default:** Full automated bout table.  
**Optional:** Explicit filtered profile / channel-behavior selection (same modes as §2.2, selection only).

### 3.2 Column contract (recommendation)

Export a flat CSV derived **only** from packaged bout rows (+ optional profile index join for inclusion flag). Required semantics:

| Requirement | Columns / notes |
| --- | --- |
| Exact identities | `bout_id`, `source_s8_bout_id`, `source_s8_run_id`, `package_run_id` |
| Behavior / channel | `behavior_name`, `channel` |
| Directional semantics | `identity_grain`, `focal_id`, `partner_id`, `mouse_a_id`, `mouse_b_id`, `pair_id` |
| Dense boundaries | `start_sample_idx`, `end_sample_idx`, `start_time_sec`, `end_time_sec`, `bout_duration_frames`, `bout_duration_sec` |
| Prepared navigation | `prepared_frame_idx_start/end`, `prepared_time_sec_start/end` |
| Provenance | `export_provenance = automated`, `export_job_id`, `selection_mode`, `source_package_run_id` |
| Optional science extras | Prob summaries (`mean_prob`, …), `undefined_reason`, etc. — include by default for analysis fidelity |

**Do not:** invent new filters; rewrite durations; emit “reviewed” labels; modify Parquet sources.

Default filename: `behavior_bouts_automated.csv` (UTF-8, header row). Companion `csv_export_manifest.json` lists row count, column list, package digests, selection snapshot.

### 3.3 Framewise CSV (recommendation)

**Defer framewise CSV from this MVP.**

| Option | Verdict |
| --- | --- |
| Bout-level CSV | **In scope** — small, matches analysis/review tables, mirrors package authority |
| Framewise states CSV | **Deferred** — E-session states ~21 MB / 73k rows; duplicates S8 reference; slow to open in Excel; S10/notebooks can read Parquet by reference |

If later needed, publish as a separate job kind (`csv_framewise_export`) referencing S8 states path + package sync, still without copying into the S9 package directory.

### 3.4 Reviewed CSV boundary (fact / product)

CSV after manual correction belongs to **S10 reviewed-output publication**, with reviewed provenance — not this automated extension.

---

## 4. Export-job architecture and provenance

### 4.1 Option comparison

| Criterion | **A. Artifacts inside original S9 package** | **B. Independent immutable export-job dirs** |
| --- | --- | --- |
| Source immutability | Weak — mutates completed package trees; conflicts with “never rewrite completed runs” | **Strong** — package remains byte-stable |
| Repeatability | Hard — second export overwrites or forks unclear versions inside one run id | **Strong** — each job has its own id + settings snapshot |
| Discoverability | Convenient co-location | Requires job discovery beside package discovery (acceptable; same session root) |
| Failure recovery | Risk of half-written clips inside a “completed” package | **Clean** — failed/partial jobs isolated; package eligibility unchanged |
| Storage management | Couples clip lifetime to package lifetime | Jobs can be deleted independently |

### 4.2 Recommendation

**Choose B — independent immutable export-job directories** that **reference** an existing S9 package (and its prepared video) by path + digests.

Do not nest mutable `clips/` under a completed `results_package__*`. Optional convenience: package discovery UI may list child export jobs by matching `source_package_run_id`.

### 4.3 Directory layout (recommendation)

```text
{session_root}/
  results_package/
    results_package__{ts}/          # unchanged authoritative package
  results_exports/                  # NEW sibling root
    media_export__{ts}/
      run_meta.json
      settings_used.yaml
      export_job_manifest.json
      qc.json
      processing_log.txt
      clip_index.parquet
      failures.jsonl                # optional append-friendly detail
      clips/...
    csv_export__{ts}/
      run_meta.json
      settings_used.yaml
      export_job_manifest.json
      qc.json
      processing_log.txt
      behavior_bouts_automated.csv
```

Naming mirrors S9 run prefixes (`media_export__YYYYMMDDTHHMMSS`, `csv_export__…`).

### 4.4 Input / output contracts (recommendation)

**Inputs (validated preflight):**

1. `source_package_run_id` + `package_dir` with `status` completed / completed_with_warnings and current eligibility classification acceptable for export (define policy: allow `completed_with_warnings`; block `stale_source` / `failed` unless operator override).
2. Digests of `behavior_bouts.parquet`, `sync_index.parquet`, `package_manifest.json` identity fields match reopen checks.
3. Prepared video path reachable; fps/n_frames probe OK (media jobs).
4. Selection mode resolves to ≥0 bouts (0 bouts → completed empty job with explicit QC note, not silent success pretending clips exist).

**Outputs:** job artifacts above; never rewrite package files.

### 4.5 Settings snapshot (recommendation)

`settings_used.yaml` captures: job_kind, source package id/paths/digests, selection mode + parameters, media encode settings (crf, context, overwrite, fail_fast), output root, schema versions. Same spirit as S9 packaging settings — export reproducibility without mutating science configs.

### 4.6 Provenance and dependency validation (recommendation)

`export_job_manifest.json` should record:

- `job_kind`, `export_job_id`, `created_at`, `status`
- `source_package` {run_id, path, bouts_sha256, sync_sha256, manifest schema}
- `source_s8` ids from package meta (informational)
- `source_video` path + fingerprint/sha when available
- Artifact inventory with relative paths + hashes for CSV / clip_index (individual MP4 hashing optional/expensive — default size/mtime listing + count; optional `--hash-clips`)
- `selection` block and row counts (`n_selected`, `n_succeeded`, `n_failed`)

### 4.7 Job status, QC, discovery, duplicates (recommendation)

| Topic | Policy |
| --- | --- |
| Lifecycle | `running` → validate → write under job dir → `completed` / `completed_with_warnings` / `failed` (atomic finalize of meta/QC like S5–S9) |
| Discovery | Scan `results_exports/` ; classify by meta status + whether source package still eligible/present |
| Reopen | Load manifest + clip_index/CSV; verify source package digests; surface stale warnings without deleting outputs |
| Duplicate jobs | Always allow a **new** timestamped job; never silently overwrite another job dir |
| Overwrite inside a job | Only for resume of the **same** `running`/`failed` job with explicit resume flag; completed jobs are immutable |
| Storage | Document expected size ≈ Σ clip durations × bitrate; provide delete-job utility later; never auto-delete packages when deleting exports |

---

## 5. GUI responsibilities (S9 page extension)

Keep **Packaging** and **Export** as distinct operations on the existing Group 4 S9 page (no S10 playback/editor).

### 5.1 Proposed UI blocks (recommendation)

| Control | Behavior |
| --- | --- |
| Package context | Selected completed S9 run (from existing discovery table) — required before export actions enable |
| **Export Videos** | Opens/enables media-export settings; starts `GuiTaskKind` e.g. `RESULTS_MEDIA_EXPORT` |
| **Export CSV** | Starts `RESULTS_CSV_EXPORT` (fast; still background for consistency) |
| Selection | Radio/mode: all bouts / channels+behaviors / existing package profile (`review_emit_v1` if present) |
| Media settings | CRF, pre/post context seconds, overwrite/resume, fail-fast toggle |
| Output directory | Default `{session}/results_exports/`; optional browse override for the exports root |
| Progress | Determinate bout counter for media; indeterminate or quick determinate for CSV |
| Completed exports | Table under package: job id, kind, status, succeeded/failed counts, open folder |
| Failures | Show QC warnings + link to `failures.jsonl` / filter clip_index `encode_status==failed` |

### 5.2 Explicit non-goals on S9 page (fact / retained)

No video playback, ethogram timeline, bout editing, or reviewed-CSV publication. Scope label on the page should state exports are optional derivatives of an existing package.

### 5.3 Task isolation (recommendation)

- Packaging task and export tasks must not run concurrently on the same controller (extend existing `TaskAlreadyRunningError` pattern).
- Soft navigation away mid-export should restore progress like packaging.
- Export must not be required to mark packaging successful.

---

## 6. S10 handoff

| S10 need | How this extension helps | Fallback if absent |
| --- | --- | --- |
| Open exported clips | Resolve latest eligible `media_export` job for package + `clip_index` row by `bout_id` | — |
| Seek prepared video | Always available via package `prepared_frame_idx_*` + `source_video.path` | **Primary path** — clips optional |
| Stable bout identity | Package / S8 bout ids preserved in clip_index and CSV | Same ids in package Parquet |
| Reviewed exports | S10 publishes separate reviewed artifacts referencing S9 package id (+ optional media job id) | Automated CSV remains `provenance=automated` |

**Hard rule:** S10 review readiness = eligible S9 package + prepared video (+ sync). Media export jobs are convenience caches, never gates.

---

## 7. Performance and storage considerations

| Factor | E-session scale (fact) | Implication |
| --- | --- | --- |
| Bout counts | 609 full / ~514 review_emit | Media jobs minutes–tens of minutes sequential |
| Prepared video | ~417 MB | Reference only |
| Clip storage | Dominates exports root | Prefer profile/channel selection for routine shares; full-bout export is explicit |
| CSV | ≪1 MB for bout table | Cheap; safe default full table |
| Hashing every MP4 | Expensive | Optional; default hash clip_index + CSV + settings |
| Many-to-one prepared map | Dense duration ≠ prepared frame span | Clip index must expose both; do not “fix” duration from prepared span |

---

## 8. Open decisions

| ID | Question | Options | Lean |
| --- | --- | --- | --- |
| E1 | Default media selection | All bouts vs require explicit profile | Require explicit mode in GUI (default UI = `review_emit_v1` if present else all) to avoid accidental 609-clip jobs |
| E2 | Context padding defaults | Always 0 vs small fixed pad (e.g. 0.25 s) | **0.0** with optional pad |
| E3 | Overlay videos as encode source | Prepared-only MVP vs priority list | **Prepared-only** MVP; overlays later |
| E4 | Frame-accurate ffmpeg | `-ss` time (legacy) vs select/filter by frame | Start legacy-compatible time slice; document non-sample-accurate risk; revisit if QC demands |
| E5 | Export eligibility on `stale_source` packages | Block vs allow historical export | **MVP: block** new create/retry exports from stale packages; preserve historical job inspection; defer explicit archival export |
| E6 | Shared vs separate export roots for media/CSV | Single `results_exports/` | Single root, `job_kind` prefix |
| E7 | Ad-hoc filters without packaged profile | Disallow vs ephemeral selection | Allow channel/behavior allow-lists; forbid inventing new emit floors in export |
| E8 | GIF / preview later | Keep deferred vs schedule S9-E3 | Deferred |
| E9 | Parallel ffmpeg workers | Off vs N workers | Off for MVP |
| E10 | Clip SHA256 | Off vs on | Off by default |

---

## 9. Bounded implementation roadmap

Documentation-only stop here. Suggested implementation slices **after** acceptance of this audit:

| Slice | Deliverable | Validation |
| --- | --- | --- |
| **S9-E2a** | Headless `csv_export` job writer + discovery helpers | Round-trip row counts vs package; provenance columns; no Parquet mutation tests |
| **S9-E2b** | Headless `media_export` preflight + clip_index planning (no encode) | Boundary/context math unit tests; inclusive/prepared collapse cases |
| **S9-E2c** | ffmpeg MP4 encode + partial failure/resume | Small fixture video; fail-one-bout job → `completed_with_warnings` |
| **S9-E3** | GUI: Export CSV / Export Videos on S9 page + task kinds | Responsiveness; concurrent-task guard; lab checklist addendum |
| **S9-E4** | S10 consumption notes only (or thin open-clip helper API) | Still no S10 UI requirement for S9 closure |

**Out of roadmap unless reopened:** GIF batch, framewise CSV, portable embed of video, scientific threshold changes, edits to existing packages.

---

## 10. Deliverable checklist

| # | Item | Location |
| --- | --- | --- |
| 1 | Legacy Pipeline 110 media inventory | §1 |
| 2 | Proposed MP4 export contract | §2 |
| 3 | Proposed CSV export contract | §3 |
| 4 | Export-job architecture and provenance | §4 (**recommend B**) |
| 5 | GUI responsibilities | §5 |
| 6 | Performance and storage | §7 |
| 7 | Open decisions | §8 |
| 8 | Bounded implementation roadmap | §9 |
| — | S10 handoff | §6 |

**Stop for architectural review.** Do not implement code, generate clips/CSV, commit, or push until this audit is accepted.
