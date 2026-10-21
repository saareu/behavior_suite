# Subsystem 05 — Sprint 0 Repository Preflight and Integration Audit

**Status:** Investigation complete (documentation only). No S5 production code.
**Date:** 2026-10-20
**Authority:** Current repository source and active docs. Older archive docs are
not treated as implementation truth when they conflict with code.

---

## 1. Existing architecture findings

### 1.1 Repository layout (observed)

| Area | Location |
| --- | --- |
| Project lifecycle / paths | `src/project/` (`paths.py`, `service.py`, `models.py`, `validation.py`) |
| S1 preprocessing | `src/preprocess/` |
| S2 pose inference | `src/pose_inference/` |
| S3 tracking correction | `src/tracking_correction/` |
| S4 pose finalization | `src/pose_finalization/` |
| GUI | `src/ui/` (`main_window.py`, pages, controllers, `tasks.py`) |
| CLI entry | `src/cli/preprocess.py` (`behavior-suite`) |
| Tests | `tests/<package>/`, `tests/ui/`, `tests/integration/` |
| Configs | `configs/preprocess_default.yaml`, `configs/subsystem_02/` |
| In-repo legacy references | `legacy/cage_cropper_reference.py`, `legacy/prepare_reference.py` only |
| Docs index | `docs/README.md` |

`pyproject.toml` already packages `pose_finalization*` under `src/`. The AI
coding guide package tree (`docs/general/development/ai_coding_guide.md` §2)
is outdated: it omits `pose_finalization` and does not yet mention S5.

### 1.2 Session artifact layout (observed)

```text
<session_root>/
  preprocess/                 # S1 official outputs (single directory, not run-id)
  pose_inference/<run_id>/    # S2 timestamped runs
  tracking_correction/<run_id>/  # S3 timestamped runs
  pose_finalization/<run_id>/ # S4 timestamped runs
```

S1 writes a single official `preprocess/` tree
(`preprocess.models.PreprocessOutputs.from_preprocess_dir`). S2–S4 use unique
timestamped run directories and never overwrite prior completed runs.

### 1.3 S1 prepared artifacts, timing, readiness

| Concern | Authority |
| --- | --- |
| Official artifact set | `docs/subsystem_01/preprocessing.md` §10; `PreprocessOutputs` |
| Sync schema / validation | `preprocess.sync_writer` (`PREPARED_SYNC_SCHEMA_VERSION = prepared_sync_v1`) |
| Sync build | `preprocess.sync_writer.build_prepared_sync` |
| Sync load | `preprocess.sync_writer.load_prepared_sync_npz` |
| Orchestration | `preprocess.service` (stages, validates, promotes) |
| Logging helper | `preprocess.logging_utils.create_processing_logger` |
| Background PNG | `preprocess.background` → `cropped_background.png` |

S1 official files:

```text
prepared_video.mp4
prepare_meta.json
prepared_sync.npz
cropped_background.png
settings_used.yaml
processing_log.txt
```

S2/S3 handoff currently requires only three S1 files
(`prepared_video.mp4`, `prepare_meta.json`, `prepared_sync.npz`):

- Discovery: `pose_inference.run_discovery.REQUIRED_S1_HANDOFF_FILES`
- Validation: `pose_inference.runner.validate_s1_handoff` / `REQUIRED_S1_FILES`

**Implication for S5:** approved design also needs `cropped_background.png`.
That is stricter than today’s S2 handoff. S5 should define its own eligibility
helper rather than reusing S2’s three-file set unchanged.

### 1.4 S2–S4 navigation and upstream handoffs

| Step | Mechanism | Key symbols |
| --- | --- | --- |
| S1 → S2 | Auto after successful prepare; shared `_session_root` | `MainWindow._s1_completed` → `open_pose_inference` |
| S2 → S3 | Explicit handoff of selected completed S2 run | `PoseInferencePage.s3_handoff_requested` → `MainWindow._show_s3_from_s2` |
| S3 → S4 | Optional continue after acceptance; S4 rediscovers accepted S3 | `MainWindow._s3_accepted`; `open_pose_finalization` |
| Flat nav | Four top-level buttons | `MainWindow` S1–S4 buttons + `QStackedWidget` |

There is **no application group concept** yet. Navigation is a single row of
subsystem buttons (`src/ui/main_window.py`).

Upstream validation patterns:

- S2: `validate_s1_handoff`, `_validate_prepared_video_against_sync`
- S3: `tracking_correction.runner._validate_handoff` / `validate_s2_handoff`
- S4: `pose_finalization.accepted_s3.validate_accepted_s3_run` (acceptance +
  SHA256 + `prepared_sync.npz`)

### 1.5 S4 source integrity, metadata, status

| Concern | Location |
| --- | --- |
| Accepted-S3 contract | `pose_finalization.accepted_s3` |
| Runner / failure cleanup | `pose_finalization.runner.run_pose_finalization` |
| Downstream eligibility / stale | `is_s4_run_eligible_for_downstream`, `validate_s4_run_for_downstream` |
| Completed-run discovery | `pose_finalization.run_discovery.summarize_pose_finalization_project` |
| Status values | `pose_finalization.contracts` (`completed`, `completed_with_warnings`, `failed`, …) |
| GUI controller | `ui.controllers.pose_finalization_controller.PoseFinalizationController` |
| GUI page | `ui.pages.pose_finalization_page.PoseFinalizationPage` |

Observed S4 policies worth mirroring for S5:

1. Unique timestamped run directory; never overwrite prior runs.
2. Upstream artifacts remain read-only.
3. On failure, remove incomplete primary numeric artifact (`final_pose.parquet`);
   keep `run_meta.json` / log with `status=failed`.
4. Revalidate upstream eligibility immediately before marking completed.
5. Completed discovery classifies `eligible` vs `stale_source` without rewriting
   prior run folders.
6. No cooperative cancellation for long runs (GUI stays responsive; blocks
   duplicate launch / navigation while active).

### 1.6 GUI task execution and progress

| Component | Role |
| --- | --- |
| `ui.tasks.GuiTaskRunner` | One callable on a `QThread`; GUI-thread callbacks |
| `GuiTaskContext` / `GuiTaskKind` | Identity + cooperative cancel (S1 kinds only today) |
| `GuiTaskProgressTracker` | ETA from `preprocess.models.OperationProgress` |
| S2/S4 pages | Own `GuiTaskRunner`; block nav via `has_running_task` |

S1 uses typed `GuiTaskKind` and cancellation. S2 and S4 use `GuiTaskRunner`
without a safe cancel hook (`MainWindow.closeEvent` waits for completion).

### 1.7 Completed-run discovery and supersession

| Subsystem | Discovery | Supersession / staleness |
| --- | --- | --- |
| S2 | `summarize_pose_inference_project` | Classification from artifacts/status; no S1 “acceptance” |
| S3 | `summarize_tracking_correction_project` | `acceptance_state`: `not_accepted` / `accepted` / `superseded` (`tracking_correction.contracts`, `review.py`) |
| S4 | `summarize_pose_finalization_project` | Rechecks recorded accepted S3 SHA; `eligible` / `stale_source` |

S5 has no acceptance step in the approved design (similar to S4). Staleness
should likely key off **S1 preprocess identity** (paths + digests), not S3
acceptance. Exact digest fields are an unresolved decision (see §7).

### 1.8 Shared configuration and logging

- S1 defaults: `configs/preprocess_default.yaml` + `preprocess.config`
- S2 profiles: `configs/subsystem_02/*.yaml`
- S3 profile: `tracking_correction/backends/current_lab/profile.yaml`
- S4 settings: dataclass defaults → `settings_used.yaml` (no separate configs/)
- Processing logs: S1 uses `create_processing_logger`; S4 writes a structured
  text summary via `_write_processing_log` (does not reuse the S1 logger helper)

### 1.9 Test conventions

- Layout mirrors packages: `tests/pose_finalization/test_timing.py`, etc.
- UI tests under `tests/ui/`
- Integration marker: `@pytest.mark.integration`
- `pythonpath = ["src"]` in `pyproject.toml`
- Focused unit tests for contracts/timing before e2e GUI tests (S4 pattern)

---

## 2. S1 timing and artifact contract

### 2.1 Authoritative arrays

From `prepared_sync.npz` / `PreparedSyncData` (`preprocess.sync_writer`):

| Array / field | Role for S5 |
| --- | --- |
| `prepared_frame_idx` | Authoritative prepared-frame index; must be `[0..N-1]` |
| `prepared_time_sec` | **Authoritative source clock** for lab 120 Hz mapping |
| `fps_header` | Defines `prepared_time_sec = prepared_frame_idx / fps_header` |
| `prepared_frame_count_opencv_readable` / `frame_count_used_for_sleap` | Authoritative frame count `N` |
| `raw_decode_frame_idx` | Traceability to raw decode order (not S5 sampling clock) |
| `raw_pts_time_sec` | Diagnostic only |
| `external_time_sec` | Optional external timeline; **not** S4/S5 sampling clock |

S4 explicitly loads `prepared_time_sec` + `prepared_frame_idx` via
`pose_finalization.timing.load_source_timing` and rejects non-monotonic or
non-contiguous index arrays.

### 2.2 Absolute vs relative

`prepared_time_sec` is **relative to the prepared timeline origin**:

```text
prepared_time_sec[i] = prepared_frame_idx[i] / fps_header
```

Validated in `validate_prepared_sync`. For standard S1 builds this starts at
`0.0`. It is **not** wall-clock absolute time. Optional `external_time_sec` may
carry absolute/session times but is not the S4/S5 target-mapping authority.

### 2.3 Prepared-video frame mapping

```text
prepared_frame_idx i  ↔  prepared_video frame i
raw_decode_frame_idx = start_frame + prepared_frame_idx
trim interval: [start_frame, end_frame_exclusive)
```

No silent drop/duplicate/reorder/resample of prepared frames (S1 contract).

### 2.4 Exact 120 Hz timeline construction (existing, reusable)

Implemented in `pose_finalization.timing`:

1. `t0 = prepared_time_sec[0]`, `t1 = prepared_time_sec[-1]`
2. `dt = 1 / target_fps` with `target_fps = 120.0` (`TARGET_FPS_DEFAULT`)
3. `sample_count = floor((t1 - t0) * target_fps + 1e-9) + 1`
4. `target_time_sec[k] = t0 + k * dt`, then clip any FP overshoot past `t1`
5. Endpoints included when they land on the grid; last sample is ≤ `t1`

S4 then maps each target sample to **left/right source frames + interpolation
weight** (`map_target_to_source`) for pose coordinate interpolation.

### 2.5 Start/end and rounding rules

| Rule | Observation |
| --- | --- |
| Source start | First prepared time (`prepared_time_sec[0]`, normally 0) |
| Source end | Last prepared time inclusive as upper bound |
| Sample count | Floor formula above + atol `1e-9` |
| Exact coincidence | `|t_target - t_source| ≤ 1e-9` → exact source sample |
| Endpoint | Target timeline must stay within `[t0, t1]` |

Covered by `tests/pose_finalization/test_timing.py`.

### 2.6 Proposed exact S5 mapping contract (aligned with S1/S4)

**Shared with S4 (must not diverge):**

1. Load timing only via `load_prepared_sync_npz` / `load_source_timing`.
2. Build `target_time_sec` only via `build_uniform_target_timeline(..., target_fps=120.0)`.
3. Emit S5 rows on `(sample_idx, time_sec)` where `time_sec == target_time_sec[sample_idx]`.
4. Record provenance of source S1 sync path (and recommended content digests).

**S5-specific (approved design; differs from S4 pose interpolation):**

5. Features are computed in **prepared-video frame space** at integer
   `prepared_frame_idx`.
6. Map each `target_time_sec[k]` → one prepared frame by **nearest neighbor** in
   time:

```text
source_frame_idx[k] = argmin_i |prepared_time_sec[i] - target_time_sec[k]|
```

7. Do **not** linearly interpolate blob/segmentation features between frames.
8. When a target coincides with a source time within `1e-9`, that frame is used
   (same exactness notion as S4).

**Reusable helpers:** `load_source_timing`, `build_uniform_target_timeline`.
Optionally derive NN indices from an S4 `TimingMap` (left/right + weight →
choose nearer side) **or** add a small pure function
`map_target_nearest_source_frame(...)` next to S4 timing / under S5 that shares
the same timeline builder. Prefer sharing the timeline builder over duplicating
the floor/atol rules.

### 2.7 Ambiguities (do not invent silently)

| Topic | Status |
| --- | --- |
| Equidistant NN ties (`|t-tL| == |t-tR|`) | Legacy uses `np.round(t * source_fps)` (IEEE half-to-even at `*.5`). Approved S5 should keep **S1 `prepared_time_sec` authority** and pick an explicit tie rule. Prefer documenting half-to-even vs left-frame before Sprint 2 port. |
| Whether to store both NN frame and S4 left/right/weight columns in S5 parquet | Unresolved; legacy exports `orig_frame_idx` only. Recommend keeping that column (rename mapping clear in meta). |
| Staleness when `preprocess/` is overwritten by a new S1 run | Unresolved policy (see §7) |
| Use of `external_time_sec` for export columns | Not required for mapping; optional display only |

---

## 3. Legacy Pipeline 40 processing and parameter inventory

### 3.1 Location (resolved)

| Asset | Path |
| --- | --- |
| Pipeline script | `legacy/pipelines/40_extract_blob_features.py` |
| Lab config (authoritative defaults) | `legacy/configs/blob_features.yaml` |
| Global FPS fallback | `legacy/configs/global.yaml` (`fps: 120`) |
| Related S6 helper (not S5) | `legacy/src/dynamic_train/frame_features.py` |
| Related S6 pipeline | `legacy/pipelines/50_dyadic_features.py` |

Entry point: `run_blob_features()` → config merge
`configs/blob_features.yaml` over `configs/global.yaml` → per-experiment
`_run_blob_features_for_experiment`.

**Missing companion in this checkout:** `tools.remove_cable_artifact` is
imported by Pipeline 40, but `legacy/tools/` currently only contains
`extract_pose_csv.py`. Cable removal cannot be executed from this tree until
that module is restored or reimplemented from another lab copy.

### 3.2 Processing trace (verified from source)

| Stage | Function(s) | Notes |
| --- | --- | --- |
| Video / background load | `_run_blob_features_for_experiment` | OpenCV `VideoCapture` on prepared mask/video artifact; `cv2.imread` background |
| Optional cable removal | `_prepare_frame_for_blob_features` → `remove_cable_artifact` | Skipped when `remove_cable_artifact: false` (lab YAML default) |
| Foreground extraction | `_extract_robust_foreground_mask` | `cv2.subtract(background, frame)`; Gaussian blur; threshold; abs-dark AND |
| Threshold selection | same | `>=0` fixed; `-1` Otsu; `-2` 3-class Multi-Otsu (skimage) using upper split `thresholds[1]` |
| Morphological cleanup | same | CLOSE (`blob_kernel_size`) → OPEN 3×3 → OPEN tail kernel → CLOSE 3×3; area filter; thickness prune; post-prune area filter; final CLOSE smooth |
| Blob / component detection | `_process_frame` | External contours; `cc_count_raw`; `cc_count = min(raw, 2)` or NaN if `raw > 2`; `min_inter_contour_dist` via `cKDTree` between two largest contours |
| Occlusion morphology | `_extract_occlusion_static_features` (when `cc_count == 1`) | Smooth merged mask; ellipse eccentricity, circularity, solidity; DT peaks; skeleton furthest-endpoint distance (skimage) |
| Short gap fill | `_impute_cc_count_and_min_inter_contour_dist` | Fill internal `cc_count` NaNs only when flanking values match; linear interp of `min_inter_contour_dist` on same gaps; **not** Pipeline 30 pose imputation |
| Temporal occlusion features | `_add_temporal_occlusion_features` | Bout-scoped (`cc_count == 1` runs): rolling mean 5, diff 5, std 15 per static feature |
| 120 Hz resampling | `_resample_blob_features_to_target_fps` | Nearest neighbor via `round(t_target * source_fps)`; see timing conflict §3.5 |
| Parquet export | `df.to_parquet` | Artifact type `blob_features` via legacy path utils |

Parallelism: frames read sequentially; mask built on reader thread; `_process_frame`
submitted to `ThreadPoolExecutor` (`n_workers` / `max_in_flight`).

### 3.3 Output schema (legacy)

Source-rate rows (before resample) carry:

`frame_idx`, `cc_count`, `cc_count_raw`, `min_inter_contour_dist`, plus
`OCCLUSION_STATIC_FEATURE_COLS`.

After temporal features + resample, exported columns:

| Column | Meaning |
| --- | --- |
| `dense_frame_idx` | Target timeline index (0…N_target−1) |
| `frame_idx` | Same as `dense_frame_idx` after resample |
| `orig_frame_idx` | Nearest source prepared-frame index |
| `cc_count` | Effective count capped at 2; NaN if fragmented (`raw > 2`) unless gap-filled |
| `cc_count_raw` | Raw component count |
| `min_inter_contour_dist` | Min distance between two largest contours; 0 if &lt;2; NaN if fragmented |
| `ellipse_eccentricity`, `circularity`, `solidity` | Static occlusion geometry (`cc_count == 1` only) |
| `num_dt_peaks`, `furthest_endpoint_distance_px` | Static occlusion morphology |
| `*_mean_5`, `*_delta_5`, `*_std_15` | Temporal occlusion features for each of the five static fields |

### 3.4 Parameter inventory

**Authority for lab defaults:** `legacy/configs/blob_features.yaml` (not the
weaker `.get(...)` fallbacks inside `_run_blob_features_for_experiment`).

Affects: **P** = single-frame preview / segmentation, **T** = temporal post-pass,
**E** = execution/runtime only, **R** = resampling / timeline.

| Parameter | Lab YAML default | Code `.get` fallback | Affects | Validation / notes |
| --- | --- | --- | --- | --- |
| `blob_threshold` | `-2` | `-1` | P | `>=0` fixed; `-1` Otsu; `-2` Multi-Otsu (needs skimage) |
| `blob_blur_kernel_size` | `9` | `5` | P | Coerced `>=1`; used as Gaussian `(k,k)` (should be odd) |
| `blob_kernel_size` | `27` | `5` | P | CLOSE ellipse; even values bumped +1 at use site |
| `blob_tail_open_kernel_size` | `7` | `7` | P | Forced odd via `_odd` |
| `blob_min_area` | `3100` | `300` | P | `>=1` |
| `blob_max_area` | `0` | `0` | P | `0` = no upper bound |
| `blob_prune_min_radius` | `10` | `8` | P | Thickness core threshold |
| `blob_prune_regrow_iters` | `7` | `6` | P | `>=0` |
| `blob_prune_max_regrow_dist` | `8` | `8` | P | `>=0` |
| `blob_post_prune_min_area` | `3100` | `= blob_min_area` | P | `>=1` |
| `blob_post_prune_max_area` | `0` | `= blob_max_area` | P | `0` = no upper bound |
| `blob_final_smooth_kernel_size` | `3` | `3` | P | Odd ellipse CLOSE |
| `abs_dark_threshold` | `10` | `10` | P | Foreground must also be darker than this in original frame; `0` disables meaningfully (all pass) |
| `remove_cable_artifact` | `false` | `True` | P | YAML disables; code fallback would enable |
| `cable_tophat_kernel_size` | `7` | `5` | P | Only if cable removal on |
| `cable_open_kernel_size` | `3` | `3` | P | |
| `cable_close_kernel_size` | `5` | `5` | P | |
| `cable_dilate_kernel_size` | `5` | `3` | P | |
| `cable_min_area` | `3` | `3` | P | |
| `cable_max_area` | `4000` | `4000` | P | |
| `cable_inpaint_radius` | `2` | `2` | P | |
| `cc_count_gap_fill_max_gap` | `5` | `5` | T | `0` disables; source-rate only, before temporal/resample |
| `target_fps` | *(from global `fps: 120`)* | global/`fallback_target_fps` | R | Must be `> 0`; S5 MVP should fix `120.0` like S4 |
| `n_workers` | `16` | `cpu_count - 1` | E | `>=1` |
| `max_in_flight` | `128` | `max(16, n_workers*8)` | E | `>=1`; memory bound |

Hard-coded occlusion morph constants (not YAML): adaptive Gaussian/close/open
kernels from blob size; DT `threshold_rel=0.25`; rolling windows 5 / 15.

### 3.5 Legacy timing vs approved S1/S4 contract

Legacy `_resample_blob_features_to_target_fps`:

```text
t_last = (n_source - 1) / source_fps
n_target = floor(t_last * target_fps) + 1
t_target[k] = k / target_fps
orig_idx[k] = clip(round(t_target[k] * source_fps), 0, n_source - 1)
```

`source_fps` comes from `get_fps_effective(video)` / `CAP_PROP_FPS`, **not**
from `prepared_sync.npz`.

Approved S5 + existing S4:

```text
prepared_time_sec from prepared_sync.npz
target_time_sec = build_uniform_target_timeline(...)
orig_idx = nearest prepared_frame_idx in prepared_time_sec
```

When `prepared_time_sec[i] = i / fps_header` and `source_fps == fps_header ==
target_fps`, counts usually match; midpoint ties may still differ (`np.round`
half-to-even vs left-argmin). **S5 must follow S1/S4 timing authority**, not
legacy video-header FPS probing. No numerical regression against Pipeline 40
is required; structural columns (`orig_frame_idx`, features) remain.

### 3.6 Reusable pure functions / preview sharing

Safe to reuse as pure cores (after de-legacy-path cleanup):

| Function | Preview? | Full run? |
| --- | --- | --- |
| `_prepare_frame_for_blob_features` | yes | yes |
| `_extract_robust_foreground_mask` | yes | yes |
| `_process_frame` (topology + static occlusion) | yes (mask overlay + scalars) | yes |
| `_impute_cc_count_and_min_inter_contour_dist` | no | yes |
| `_add_temporal_occlusion_features` | no | yes |
| `_resample_blob_features_to_target_fps` | no | replace with S1-aligned NN mapper |

**Conclusion:** single-frame GUI preview **can** call the same core segmentation
(`_extract_robust_foreground_mask`) and static feature path as full extraction.
Temporal features, gap fill, and resampling remain batch-only.

Occlusion static features and Multi-Otsu need **scikit-image** (optional
dependency today; not in `behavior_suite` `pyproject.toml`).

---

## 4. Proposed S5 module architecture

Prefer existing package patterns (`pose_finalization` is the closest analogue:
S1-timed export, run dirs, QC, discovery, GUI runner). Avoid new frameworks.

### 4.1 Package layout

```text
src/blob_features/                 # or video_features/ — recommend blob_features
  __init__.py
  contracts.py                     # names, statuses, schema version, defaults
  s1_eligibility.py                # validate eligible S1 for S5
  timing.py                        # NN map onto shared uniform 120 Hz timeline
  config.py                        # frozen settings dataclass (+ YAML IO)
  segmentation.py                  # pure per-frame mask/contour (from legacy)
  features.py                      # per-frame + temporal feature calc (from legacy)
  export.py                        # parquet writer / schema
  qc.py                            # structural + operational QC
  runner.py                        # end-to-end run + failure cleanup
  run_discovery.py                 # list completed / stale S5 runs

src/ui/controllers/blob_features_controller.py
src/ui/pages/blob_features_page.py

tests/blob_features/...
tests/ui/test_blob_features_*.py

docs/subsystem_05/                 # this audit + future MVP doc

configs/subsystem_05/              # only if defaults benefit from YAML profiles;
                                   # otherwise dataclass defaults like S4
```

Also update `pyproject.toml` `packages.find.include`, coverage `source`, and
ruff/mypy first-party lists when implementation starts (not in Sprint 0).

### 4.2 Input readiness and provenance

Recommended `validate_s1_for_blob_features(session_root) -> EligibleS1Input`:

Required files under `preprocess/`:

- `prepared_video.mp4`
- `cropped_background.png`
- `prepared_sync.npz` (loadable via `load_prepared_sync_npz`)
- `prepare_meta.json`

Recommended readiness checks (mirror S2 where useful):

- Sync validates; `prepared_frame_idx` / times OK
- Video opens; reported frame count matches `frame_count_used_for_sleap`
- Background PNG readable; size matches prepared frame size from meta/video

Provenance to record in `run_meta.json`:

- `session_root`, preprocess paths
- SHA256 of video, background, sync (and optionally prepare_meta)
- Sync schema version, `fps_header`, frame count
- Settings actually used
- Software / package version

### 4.3 Processing / preview boundaries

| Boundary | Responsibility |
| --- | --- |
| `segmentation.segment_frame(...)` | Pure: frame + background + params → mask/contours |
| `features.compute_frame_features(...)` | Pure per-frame metrics |
| `features.compute_temporal_features(...)` | Sequence-only; not used in single-frame preview |
| `timing.map_nn(...)` | Target clock → prepared frame indices |
| `runner.run_blob_feature_extraction(...)` | I/O, logging, QC, artifact policy |
| GUI preview | Calls `segment_frame` only; no run directory required |
| GUI bookmarks | UI state / optional sidecar; not scientific output |

### 4.4 Configuration

Follow S4: frozen dataclass defaults + `settings_used.yaml` written per run.
Expose interactive segmentation knobs in Advanced Settings once legacy defaults
are known. Do not invent scientific defaults in Sprint 0–1 beyond placeholders
clearly marked `TBD_FROM_LEGACY`.

### 4.5 Output schema and artifact placement

Proposed (pattern-aligned; columns from legacy §3.3):

```text
<session_root>/blob_features/<run_id>/
  blob_features.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

Run id pattern: `blob_features__YYYYMMDDTHHMMSS` (mirrors S2/S3/S4).

Parquet should preserve legacy scientific columns and add S1/S4-aligned timing:

- `schema_version`
- `sample_idx` / `time_sec` (uniform 120 Hz from S1 `prepared_time_sec`)
- `orig_frame_idx` (NN prepared-frame index; legacy name retained)
- legacy feature columns from §3.3 (`cc_count`, occlusion fields, …)
- optional aliases: keep `dense_frame_idx`/`frame_idx` only if downstream S6
  still expects those exact names (confirm against Pipeline 50 before freeze)

### 4.6 Run metadata, logging, QC

Mirror S4 statuses: `running` → `completed` | `completed_with_warnings` | `failed`.
QC summary embedded in `run_meta.json`. Processing log as UTF-8 text summary
(S4 style) or `create_processing_logger` (S1 style); prefer one pattern and
stick to it—recommend S4-style summary for export subsystems.

### 4.7 Safe interruption / failure

- Block duplicate launches while a run is active (`GuiTaskRunner` + controller flag).
- No cooperative cancel in MVP unless segmentation loop can check a flag cheaply
  (same limitation class as S2/S4 is acceptable).
- On failure: do not leave a completed parquet; write failed meta/log.
- Revalidate S1 eligibility (paths + digests) immediately before completion if
  staleness policy requires it.

### 4.8 Test locations

```text
tests/blob_features/test_s1_eligibility.py
tests/blob_features/test_timing_nn.py
tests/blob_features/test_segmentation.py      # once legacy ported
tests/blob_features/test_export_qc_runner.py
tests/ui/test_blob_features_gui.py           # later sprint
```

Reuse `tests/pose_finalization/test_timing.py` ideas; assert S5 target times
**equal** S4 `build_uniform_target_timeline` outputs for the same sync.

---

## 5. GUI / group-navigation integration plan

### 5.1 Current state

`MainWindow` hosts a flat S1–S4 button row and a four-page `QStackedWidget`
(wizard, pose page, S3 shell, S4 shell). Session root is shared. Navigation is
blocked while S2 or S4 tasks run.

### 5.2 Desired groups

| Group | Subsystems |
| --- | --- |
| Group 1 — Data Preparation | S1, S2, S3, S4 |
| Group 2 — Feature Processing | S5, S6 |

S5 must be reachable whenever eligible S1 artifacts exist (independent of
S2–S4).

### 5.3 Smallest safe navigation change (plan only — do not implement in Sprint 0)

Recommended incremental approach:

1. **Add a group selector** (two mutually exclusive controls) above or beside
   the subsystem button row: `Data Preparation` | `Feature Processing`.
2. **Keep existing S1–S4 widgets and stack indices unchanged** when Group 1 is
   active (preserve current behavior and tests).
3. When Group 2 is selected, show S5 (and later S6) buttons; hide or disable
   S1–S4 buttons without destroying widgets/state.
4. Add `BlobFeaturesPage` (+ optional shell like S4) to the stack.
5. Share `_session_root`; on opening S5, call eligibility discovery (analogous
   to `PoseFinalizationPage.set_session`).
6. Extend `_navigation_blocked` / `_update_navigation` for S5 running tasks.
7. Optional: after S1 completion, keep today’s auto-open S2 behavior (Group 1).
   Do **not** auto-jump to S5.

This avoids a full rewrite of `PreprocessWizard` or S2–S4 controllers.

### 5.4 Reuse for S5 UI

| Need | Reuse |
| --- | --- |
| Background execution | `GuiTaskRunner` |
| Session browse | Same `QFileDialog` pattern as S3/S4 shells |
| Frame display | `ui.widgets.video_frame_view.VideoFrameView` (extend with mask overlay later) |
| Status line | `status_message` → `MainWindow.statusBar` |
| Settings persistence | Optional `QSettings` for last paths only; scientific params in run YAML |
| Eligibility messaging | Follow S4’s explicit “why unavailable” copy |

S5 UI is more interactive than S4 (preview, bookmarks). Backend must remain
headless-testable; GUI must not own scientific algorithms
(`ai_coding_guide.md`).

---

## 6. Reusable infrastructure

| Capability | Reuse as-is |
| --- | --- |
| Sync load/validate | `preprocess.sync_writer` |
| Uniform 120 Hz timeline | `pose_finalization.timing.build_uniform_target_timeline` |
| Source timing load | `pose_finalization.timing.load_source_timing` |
| Target FPS constant | `pose_finalization.contracts.TARGET_FPS_DEFAULT` (or re-export from S5 contracts to avoid cross-subsystem coupling—decision in Sprint 1) |
| Gui task runner | `ui.tasks.GuiTaskRunner` |
| Video frame widget | `ui.widgets.video_frame_view` |
| Project preprocess path | `project.paths.get_preprocess_dir` / `PreprocessOutputs` |
| Run discovery patterns | S2/S4 `summarize_*_project` style |
| Failure/status policy | S4 runner conventions |
| Parquet stack | pandas + pyarrow (already depended) |
| OpenCV | already depended for video/background |

Avoid depending on S2–S4 **outputs** for S5 eligibility. Avoid importing S3
review/corrector code.

---

## 7. Risks, incompatibilities, and unresolved decisions

### 7.1 Conflicts with the approved S5 design

| Topic | Assessment |
| --- | --- |
| S5 independent of S2–S4 | **Compatible** with current architecture; nav must not force S2 first |
| Need eligible S1 including background | **Compatible**, but stricter than S2’s three-file handoff |
| Interactive preview + bookmarks | **New vs S4**; no existing bookmark subsystem |
| Nearest-neighbor 120 Hz mapping | **Compatible** with shared timeline; **differs** from S4 pose interpolation (intentional) |
| Legacy-compatible blob schema | **Mostly resolved** — columns inventoried in §3.3; confirm S6 name expectations |
| Retain legacy algorithms/defaults | **Resolved for planning** — use YAML lab defaults; port scientific functions unchanged |
| Legacy FPS-header resample vs S1 sync | **Design conflict** — S5 must prefer S1/S4 timing (§3.5), not `get_fps_effective` |
| Code `.get` defaults vs YAML | Prefer YAML; document divergence so Advanced Settings match lab |
| `remove_cable_artifact` module missing | Cable path blocked in this checkout until tool restored |
| scikit-image not in suite deps | Required for `blob_threshold=-2` and occlusion features |
| No numerical regression vs old pipeline | Compatible with structural/schema/timing tests |

### 7.2 Other risks

1. **Missing legacy Pipeline 40** — hard blocker for scientific port and
   parameter UI.
2. **S1 single `preprocess/` directory** — a re-prepare overwrites official S1
   artifacts; S5 staleness policy must be defined (digest compare vs assume
   operator discipline).
3. **Doc drift** — `ai_coding_guide.md` package map and root `README.md`
   “Active documentation” list omit S5; update when S5 docs land.
4. **Cross-package timing ownership** — S5 importing `pose_finalization.timing`
   couples Feature Processing to Pose Finalization. Alternatives: (a) accept
   the import for Sprint 1, (b) later extract `src/timing/` shared module.
   Prefer (a) initially to avoid speculative refactor.
5. **Group navigation regressions** — highest GUI risk; keep Group 1 path
   behavior bitwise-compatible and cover with existing UI tests.
6. **Preview performance** — full-res segmentation per scrub may need caching;
   defer until legacy cost is known.

### 7.3 Unresolved decisions (explicit)

1. ~~Path to Pipeline 40~~ — resolved: `legacy/pipelines/40_extract_blob_features.py`.
2. Restore or reimplement `tools.remove_cable_artifact` before enabling cable cleanup.
3. NN midpoint tie-break (`np.round` half-even vs left frame) under S1 times.
4. Whether exported parquet must keep exact legacy index column names for S6.
5. S5 package name: `blob_features` vs `video_features` vs `feature_extraction`.
6. Whether S5 completion requires digest-stable S1 revalidation.
7. Whether bookmarks persist inside the run directory or are UI-only.
8. Whether to extract shared timing to a neutral package before S6.
9. When to add `scikit-image` to suite dependencies (required for full legacy parity).

---

## 8. Proposed Sprint 1 scope and acceptance tests

### 8.1 Sprint 1 goals (implementation)

1. Add `src/blob_features/` skeleton: `contracts`, `s1_eligibility`, `timing`
   (NN mapping), `config` dataclass seeded from **YAML lab defaults** (§3.4).
2. Wire package into `pyproject.toml` packaging/test path (no GUI yet).
3. Implement `validate_s1_for_blob_features` including background + sync load.
4. Implement NN mapping that **reuses** `build_uniform_target_timeline` and
   asserts identical `target_time_sec` to S4 for the same sync; record
   `orig_frame_idx` via nearest `prepared_time_sec` (not video-header FPS).
5. Freeze output column list from §3.3 in `contracts.py` (values still empty
   until segmentation port).
6. Unit tests only; no MainWindow group navigation yet.

**Out of scope for Sprint 1:** segmentation/morphology port, cable tool,
temporal features, full parquet writer, GUI preview, group navigation, S6,
Pipeline 30 imputation, scikit-image dependency addition (unless needed for
imported stubs).

### 8.2 Sprint 1 acceptance tests

| ID | Test |
| --- | --- |
| T1 | Missing any required S1 artifact → clear eligibility error |
| T2 | Valid sync + video + background → `EligibleS1Input` populated |
| T3 | Frame-count mismatch video vs sync → hard failure |
| T4 | `target_time_sec` matches `pose_finalization.timing.build_uniform_target_timeline` |
| T5 | NN mapping deterministic; exact source hits when times coincide |
| T6 | Equidistant tie-break follows the decided rule (once approved) |
| T7 | S1 sync file bytes unchanged after timing helpers (read-only) |

### 8.3 Prerequisite for Sprint 2+

Pipeline 40 source and YAML defaults are now in-repo. Sprint 2 scientific port
should also restore `remove_cable_artifact` (or keep cable disabled) and decide
`scikit-image` packaging before enabling Multi-Otsu / occlusion morphology.

---

## 9. Files inspected (Sprint 0)

**Docs:** `README.md`, `docs/README.md`, `docs/subsystem_01/preprocessing.md`,
`docs/subsystem_01/status_and_roadmap.md`, `docs/subsystem_02/mvp_scope_and_roadmap.md`,
`docs/subsystem_04/mvp_scope_and_roadmap.md`, `docs/general/development/ai_coding_guide.md`

**S1:** `src/preprocess/sync_writer.py`, `models.py`, `service.py`,
`logging_utils.py`, `background.py`, `metadata.py`

**S2:** `src/pose_inference/run_discovery.py`, `runner.py` (handoff/validation)

**S3:** `src/tracking_correction/contracts.py`, `runner.py`, `review.py` (acceptance)

**S4:** `src/pose_finalization/{timing,contracts,accepted_s3,run_discovery,runner,__init__}.py`,
`tests/pose_finalization/test_timing.py`

**GUI:** `src/ui/main_window.py`, `tasks.py`,
`controllers/pose_finalization_controller.py`, `pages/pose_finalization_page.py`,
`widgets/video_frame_view.py`

**Project / packaging:** `src/project/paths.py`, `pyproject.toml`, `configs/*`,
`legacy/*`

**Legacy Pipeline 40:** `legacy/pipelines/40_extract_blob_features.py`,
`legacy/configs/blob_features.yaml`, `legacy/configs/global.yaml`; related
`legacy/pipelines/50_dyadic_features.py`,
`legacy/src/dynamic_train/frame_features.py`.
