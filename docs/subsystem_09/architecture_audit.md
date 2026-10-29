# Subsystem 09 — Results Packaging & Export

## Architecture Audit (S9-A)

**Status:** Discovery milestone (documentation only). **Accepted.** Binding decisions are formalized in the S9-B implementation contract.  
**Date:** 2026-10-28  
**Scope:** Inventory legacy Pipeline 110 packaging / bout-browser exports, map S1–S8 handoff contracts, propose an S9 package schema, video↔120 Hz synchronization contract, S10 consumption hooks, eligibility/provenance, and a bounded sprint roadmap. **No S9 or S10 production code in this sprint.**  
**Authority:** Local git-ignored `legacy/` tree (Pipeline 110 + bout review app), tracked S8 package `src/behavioral_arbitration/`, S7 `src/behavioral_inference/`, S6 `src/combined_features/`, S5 timing `src/blob_features/timing.py`, S4 timing `src/pose_finalization/timing.py`, S1 sync loaders under `src/preprocess/`, Group 3 GUI patterns under `src/ui/`, and E-session artifacts under `test_cases/E_full_session_integration/`.

**Follow-on:** [Implementation Contract (S9-B)](implementation_contract.md). Stop for architectural review of S9-B before any implementation. Do not implement S9, modify S1–S8 science, redesign GUI, commit, or push until S9-B is accepted.

**Legend used below:**  
- **Fact** — observed in repository / local legacy / E-session artifacts.  
- **Recommendation** — design proposal pending review; not binding.

---

## 0. Preconditions and inspection method

### 0.1 Upstream closure (repository-confirmed)

| Check | Result |
| --- | --- |
| S7 | MVP complete / lab-validated; publishes canonical + wide predictions with validity / provenance |
| S8 | Headless arbitration + immutable `behavior_states.parquet` / `behavior_bouts.parquet` / QC / discovery implemented under `src/behavioral_arbitration/` |
| S8 science boundary | Explicitly forbids Pipeline 110 emit floors inside S8 (`execution.apply_pipeline_110_emit_filters` must remain `false`) |
| S9 / S10 code | Absent — no `src/results_packaging/` (or equivalent); no Group 4 pages |
| Group 4 UI | Not present in `src/ui/main_window.py` navigation (Groups 1–3 only through S8) |

### 0.2 Local legacy availability (critical)

Pipeline 110 and the bout review app live under git-ignored `legacy/` on this workstation (`.git/info/exclude` contains `/legacy/`). **Do not treat `legacy/` as a production runtime dependency for CI or other clones.** Semantics required by S9 must be re-encoded into tracked docs/code after S9-B.

| Path | Role |
| --- | --- |
| `legacy/pipelines/110_bout_browser_packager.py` | Bout index / clip / GIF packaging from Pipeline 100 framewise predictions (~1481 lines) |
| `legacy/configs/bout_browser.yaml` | Emit floors, label sets, clip/GIF options, source-video priority |
| `legacy/src/bout_review_app/` | Streamlit manual bout review consumer of Pipeline 110 exports |
| `legacy/src/video/behavioral_overlay.py` | Optional ethogram overlay (Pipeline 100 / review presentation; not S9 science) |

**Smoke check performed (read-only):** traced `run_bout_browser_packager()` / `export_experiment()` stages, config defaults, manifest schema, timing maps, and review-app contracts; compared against S8 E-session states/bouts and S1/S5/S6 timeline fields. No Pipeline 110 execution and no S9 implementation.

### 0.3 Group 4 conceptual separation (approved product boundary)

| Subsystem | Role |
| --- | --- |
| **S9 — Results Packaging & Export** | Automated assembly, validation, synchronization indexing, and immutable packaging of eligible behavioral outputs for inspection, export, and analysis |
| **S10 — Behavioral Review & Editing** | Interactive video / timeline / bout review with optional manual correction and versioned reviewed outputs |

**Fact / product rule:** Manual review remains optional. Automated S9 packages may be used directly for downstream analysis with automated provenance clearly identified. S10 should consume S9 packages rather than reconstructing all upstream outputs independently.

---

## 1. Legacy Pipeline 110 source and stage inventory

### 1.1 Purpose (fact)

Pipeline 110 converts **Pipeline 100 framewise** `final_social_predictions.parquet` into a **review-ready bout manifest** plus optional **MP4 clips** and **GIF previews**. It does **not** re-run arbitration science. It **does**:

1. Filter labels for presentation.
2. Re-collapse contiguous positive frames into bouts.
3. Apply **emit floors** that drop short bouts from the review index.
4. Map dense prediction indices onto prepared / original video indices.
5. Extract media clips for the Streamlit bout review app.

Entry point: `run_bout_browser_packager()` → per-experiment `export_experiment()`.

### 1.2 Required Pipeline 100 / timing inputs (fact)

| Input | Path pattern (legacy) | Required fields / notes |
| --- | --- | --- |
| Framewise predictions | `data/inference/combined/{model_iteration}/{exp}/final_social_predictions.parquet` | `REQUIRED_PREDICTION_COLUMNS`: `exp_name`, `dense_frame_idx`, `focal_id`, `partner_id`, `behavior_label`; optional `pred_side_by_side` |
| Prepared video metadata | `path_utils` artifact `prepared_video_meta` | Trim `start_frame`, prepared/raw fps, prepared frame count |
| Dense→prepared map | `dense_pose_120_imputed_npz` key `orig_frame_idx` | Dense sample → prepared frame index |
| Source video | Priority list (default: tracks overlay → prepared → imputed overlay → pose overlay) | Probed via OpenCV for fps / n_frames |
| Config | `configs/bout_browser.yaml` + `configs/global.yaml` | Experiments, iteration names, emit floors, clip/GIF flags |

Nulls or duplicate directional keys in required prediction columns fail hard.

### 1.3 Execution flow and source-file mapping (fact)

| Stage | Function(s) | What it does |
| --- | --- | --- |
| Config load | `_load_config`, `_load_yaml`, `_global_experiment_value` | Resolve emit floors, label sets, clip/GIF, implanted-id role mapping |
| Load predictions | `_load_predictions` | Column/null/duplicate validation; keep required + optional SBS column |
| Resolve video | `_resolve_source_video`, `_candidate_source_path`, `_probe_video` | First existing priority candidate |
| Timing | `_load_timing_mapping` | prepared meta → `raw_frame_idx = start_frame + prepared_idx`; `original_time_sec = raw / original_fps` |
| Dense map | `_load_dense_imputed_map` | NPZ `orig_frame_idx` dense→prepared |
| Bound filter | `_filter_predictions_to_frame_bounds` | Drop dense frames beyond reviewable length (legacy fails if any dropped) |
| Presentation rows | `_prepare_presentation_rows`, `_active_label_sets` | Label allow-lists; optional SBS supplemental rows; symmetric collapse; directional presentation labels + I/NI roles |
| Bout collapse | `_build_bouts` | Contiguous segments per presentation identity; **emit floors** |
| Manifest | `_build_manifest`, `_validate_manifest` | Video mapping fields, bout ids, non-overlap checks |
| Clips | `_extract_clips`, `_ffmpeg_executable` | ffmpeg MP4 slice by source-frame indices |
| Previews | `_extract_previews` | GIF generation |
| Publish | `_summary_payload`, `_source_manifest_payload` | `bout_index.parquet`, `summary.json`, `source_manifest.json` under `reports/bout_browser/{exp}/{iteration}` |

### 1.4 Label / identity / bout assumptions (fact)

| Topic | Legacy behavior |
| --- | --- |
| Temporal sampling | Predictions indexed by `dense_frame_idx` (120 Hz dense / imputed timeline). **Not** interchangeable with prepared-video frame indices. |
| Symmetric labels | Default: `nose_to_nose`, `side_by_side`, `contact`, `mutually_disengaged`. Collapsed to unordered `(mouse_a_id, mouse_b_id)`; focal/partner nulled. |
| Focal / directional | Default: `nose_to_body`, `nose_to_tail`, `approach`, `retract`. Keep direction; `presentation_label = label_focalpartner`. |
| `engaged` | Forced out of symmetric set; optionally added to focal set (`include_engaged_labels: true` in current YAML). |
| `contact` | Default YAML `include_contact_labels: false` — contact excluded from review emit unless enabled. |
| Supplemental SBS | If `side_by_side` active and `pred_side_by_side==1`, rows are **added** even when exclusive `behavior_label` differs. |
| Roles | `implanted_id` (default 0) → `focal_role` / `partner_role` ∈ {`I`,`NI`}. |
| Bout endpoints | Inclusive dense indices; contiguous when `diff(dense_frame_idx)==1`. |
| Emit floors | `min_required_frames = max(min_emit_bout_frames, ceil(min_emit_bout_seconds * fps))`. YAML: **10 frames** and **0.10 s**. |
| Bout id | `{exp}__{presentation_label}_{start}_{end}` (uses dense start/end). |

**Important naming fact:** Columns `start_prepared_idx` / `end_prepared_idx` in the bout collapse and manifest **store dense_frame_idx values**, not prepared-video indices. True prepared indices are computed internally (`prepared_start_idx` / `prepared_end_idx`) for timing and clip seeking but are **not** published under distinct column names. S9 must not inherit this ambiguity.

### 1.5 Outputs (fact)

| Artifact | Path | Grain |
| --- | --- | --- |
| `bout_index.parquet` | `reports/bout_browser/{exp}/{iteration}/` | One row per emitted bout (after floors) |
| `summary.json` | same | Counts, dropped labels, clip/preview errors, config snapshot |
| `source_manifest.json` | same | Source video + timing provenance |
| `clips/**/*.mp4` | optional | Per-bout media |
| `gif_previews/**/*.gif` | optional | Per-bout previews |

Downstream consumer: `legacy/src/bout_review_app/` (`scan_exports`, `load_export`, annotation store under `reports/bout_browser_reviews/`). Review annotations are **outside** Pipeline 110 itself (S10 territory).

### 1.6 What Pipeline 110 deliberately does *not* do (fact)

- Does not re-score models or re-run Pipeline 100 arbitration.
- Does not publish a full unfiltered scientific bout table alongside the emit-filtered index.
- Does not package S7 probability tables or S8 rule diagnostics.
- Does not provide a framewise synchronized analysis table for offline stats (only bout-level review index + clips).

---

## 2. Upstream dependency map (S1–S8 → S9)

### 2.1 Artifact classes (recommendation grounded in facts)

| Class | Artifacts | S9 role |
| --- | --- | --- |
| **1. Required source** | Eligible S8 run: `behavior_states.parquet`, `behavior_bouts.parquet`, `run_meta.json`, `settings_used.yaml`, `qc.json` | Scientific contents of the package |
| **1. Required source** | Linked S1 `prepared_sync.npz` + `prepared_video.mp4` (paths/digests via S8→S7→S6→S5/S4 lineage) | Authoritative video timing + playback reference |
| **1. Required source** | Deterministic dense→prepared index map (from S5/S6 `orig_frame_idx` or rebuilt identically from `prepared_sync`) | Sync index for S10 and exporters |
| **2. Optional contextual** | S8 `arbitration_diagnostics.parquet` (if published); compact S8 QC counters already in package | Rule inspection without full re-arbitration |
| **2. Optional contextual** | Selected S7 fields (validity / score_origin / probs for detectors under review) via **reference** or thin join extract | Probability inspection |
| **3. S10 advanced only** | Full S6 `dyadic_features_full.parquet`; full S7 wide/canonical/specialist tables; pose overlays; blob masks | Keep as immutable upstream references |
| **4. Reference, do not copy** | Prepared video (~417 MB on E-session), S6 features (~225 MB), full S7 prediction set (~17 MB combined), pose overlays | Manifest paths + digests; open in place |

### 2.2 Exact inputs S9 genuinely requires (recommendation)

**Minimum to build an analysis-ready automated package:**

1. One **currently eligible** S8 run (states + bouts + meta/QC/settings).
2. Provenance pointers to the recorded S7 and S6 runs (already in S8 `run_meta.json`).
3. S1 prepared video + prepared sync (or equivalent lineage-validated paths).
4. A validated **sample_idx/dense_frame_idx → prepared_frame_idx (`orig_frame_idx`) → time** mapping table.

**Not required to copy into every package:** full S6 feature matrix; full S7 probability tables; clip/GIF media (optional export profile); S4 pose coordinates.

### 2.3 Reusable utilities (fact — prefer reuse over rewrite)

| Utility | Location | Reuse in S9 |
| --- | --- | --- |
| S8 discovery / eligibility | `behavioral_arbitration.run_discovery`, `validate_s8_run_for_downstream` | Primary eligibility gate |
| S7 / S6 eligibility | `behavioral_inference.*`, `combined_features.*` | Nested lineage checks via S8 meta |
| Digests / memoized validation | `project.file_digest`, `project.validation_memo` | Provenance without GUI rehash storms |
| Atomic publish pattern | S5–S8 runners (`running` → validate → atomic write → `completed` / `failed`) | Immutable S9 run directories |
| Nearest-neighbor timing | `blob_features.timing` (+ S4 `pose_finalization.timing` for source load) | Canonical dense↔prepared map |
| S1 sync loader | `preprocess.sync_writer.load_prepared_sync_npz` | Authoritative prepared timeline |

### 2.4 E-session scale snapshot (fact)

For `behavioral_arbitration__20261028T164423` on `E_full_session_integration`:

| Artifact | Size / shape |
| --- | --- |
| S8 `behavior_states.parquet` | ~21 MB; 73 056 rows × 147 cols; 36 528 dense samples × 2 directions |
| S8 `behavior_bouts.parquet` | ~51 KB; 609 bouts (538 exclusive, 60 contact, 11 SBS); min exclusive duration **1 frame** |
| Exclusive bouts with duration &lt; 10 | 31 (would be dropped by Pipeline 110 emit floor if applied) |
| Dense vs prepared | `dense_frame_idx` 0…36527; S6 `orig_frame_idx` 0…36237 — **not identical** |
| Prepared video | ~417 MB |
| S6 features | ~225 MB |
| Session duration | `time_sec` ≈ 0 … 304.39 s @ 120 Hz |

---

## 3. S8 → S9 scientific boundary

### 3.1 Principles (recommendation; aligns with accepted S8 P8)

1. S8 owns **scientific** exclusive labels, parallel contact/SBS channels, and bout collapse after arbitration.
2. S9 must **not** silently reinterpret states, re-arbitrate, or re-collapse bouts as a new scientific truth.
3. Any emit-floor / label-subset / presentation collapse is an **export view** that must leave the complete S8 labels/bouts accessible.
4. No new scientific thresholds in S9.

### 3.2 Pipeline 110 operation classification

| Pipeline 110 operation | Class | Disposition |
| --- | --- | --- |
| Load/validate prediction columns; duplicate-key checks | **A** packaging/validation | S9 eligibility / intake |
| Frame-bound filtering vs video/timing length | **A** (integrity) | S9 sync validation (fail or warn explicitly; do not silently drop science) |
| Label allow-list filtering (`include_contact`, engaged policy) | **B** presentation filter | Optional **export profile**; retain full S8 ethogram in package |
| Supplemental SBS rows from `pred_side_by_side` | **B** presentation (parallel channel already scientific in S8) | Prefer S8 parallel bouts/channel columns; do not invent a second exclusive label stream |
| Symmetric pair collapse + null focal/partner | **B** presentation | Optional review index view; keep directional S8 rows as truth |
| Directional `presentation_label` / I–NI roles | **A/B** indexing metadata | Optional profile; role mapping config must be provenance-stamped |
| **Re-collapse bouts from framewise labels** | **C** if treated as scientific replacement; **B** if filtered view only | **Do not replace S8 bouts.** Consume S8 `behavior_bouts.parquet`. Optional filtered index for review. |
| **`min_emit_bout_frames` / `min_emit_bout_seconds`** | **B** presentation filter with scientific *impact on what reviewers see* | **Export profile only.** Default analysis package keeps all S8 bouts. Document floors in profile provenance. |
| Dense→prepared→original mapping | **A** sync indexing | Core S9 (modernize onto S1 `prepared_sync` + S5 NN map) |
| Source-video priority / probing | **A/D** mixed | Prefer **prepared_video** as S10 default; overlays optional references |
| ffmpeg clip extraction | **A** media packaging | Optional heavy export profile / S10 convenience — not required for analysis packages |
| GIF previews | **A** presentation | Optional; S10 can generate lazily |
| `summary.json` / `source_manifest.json` | **A** QC/provenance | S9 `run_meta` / `package_manifest` |
| Review annotations / corrected labels | **D** for S9 | Belongs to **S10** versioned reviewed outputs |
| HTML presentation galleries | **D/B** | Out of S9 MVP; optional later exporter |

Class key: **A** pure packaging; **B** presentation filter preserving underlying science; **C** scientifically meaningful modification; **D** obsolete / inconsistent with S1–S8.

### 3.3 Emit floors — explicit recommendation

**Fact:** S8 publishes short exclusive bouts (E-session min 1 frame) because Pipeline 110 floors are forbidden in S8.  
**Fact:** Pipeline 110 YAML floors are 10 frames / 0.10 s and would drop 31 exclusive E-session bouts.  
**Recommendation:** S9 default package includes **all** S8 bouts. A named export profile (e.g. `review_emit_v1`) may produce a **filtered bout index** for human review without deleting or rewriting the full bout table. Never introduce new floors beyond an explicitly approved profile.

---

## 4. Proposed S9 package schema

### 4.1 Package layout (recommendation)

Immutable run directory under the session root:

```text
{session_root}/results_package/{run_id}/
  run_meta.json
  settings_used.yaml
  package_manifest.json
  qc.json
  processing_log.txt
  sync_index.parquet              # dense/sample → prepared (+ times)
  behavior_states_ref.json        # or thin parquet view — see §4.2
  behavior_bouts.parquet          # full S8 bouts (+ package stamps)
  bout_index.parquet              # optional profile: presentation/filtered index
  ethogram_summary.parquet        # cached rollups for S10/analysis
  # optional profiles:
  # clips/...
  # previews/...
```

`run_id` pattern recommendation: `results_package__{UTC_timestamp}` (mirror S5–S8).

### 4.2 Artifact contracts (recommendation)

#### `sync_index.parquet` (derived, small, required)

| Item | Spec |
| --- | --- |
| Grain | One row per `sample_idx` / `dense_frame_idx` on the 120 Hz target timeline |
| Required columns | `sample_idx`, `dense_frame_idx`, `time_sec`, `prepared_frame_idx` (= modern `orig_frame_idx`), `prepared_time_sec`, `is_exact_source_sample` (bool) |
| Optional | `raw_frame_idx`, `original_time_sec` if trim/raw metadata available without inventing a new sync model |
| Source | Rebuild from S1 `prepared_sync.npz` via `blob_features.timing` **or** copy unique `(dense_frame_idx, orig_frame_idx, time_sec)` from linked S6 (must hash-match) |
| Consumer | S10 seek; exporters; offline alignment |

#### `behavior_bouts.parquet` (copied from S8 + stamps)

| Item | Spec |
| --- | --- |
| Grain | S8 bout grain (`channel` ∈ exclusive/contact/side_by_side) |
| Required columns | Full S8 `BOUT_COLUMNS` plus `source_s8_run_id`, `package_run_id`, `schema_version` (S9 stamp) |
| Transformation | **Copy**; no duration filtering; no label rewriting |
| Consumer | Analysis; S10 bout navigation |

#### Framewise states policy (recommendation)

| Option | Pros | Cons |
| --- | --- | --- |
| **Reference-only** to S8 `behavior_states.parquet` | Avoids ~21 MB+ duplication; single scientific authority | S10 depends on S8 path remaining readable |
| **Copy full states** | Package self-contained for archival | Storage duplication; drift risk if copy diverges |
| **Thin derived view** | e.g. identity + `behavior_label` + parallel preds + key probs only | Still a derivation; must stay bit-compatible with S8 labels |

**Recommendation:** Default = **immutable reference** to the S8 states path + digest in `package_manifest.json`, plus optional profile `embed_states=true` for portable archives. Analysis that needs only ethogram labels can read the referenced S8 file; do not rebuild labels.

#### `bout_index.parquet` (optional export profile)

Presentation/review index analogous to Pipeline 110 `bout_index.parquet`, but:

- Built **from S8 bouts** (not re-segmented from scratch as scientific truth).
- May apply profile filters (emit floors, include/exclude contact/engaged, symmetric presentation collapse).
- Must include clear columns: `start_sample_idx` / `end_sample_idx` (dense), `prepared_frame_idx_start` / `prepared_frame_idx_end`, times, `filter_profile_id`, `included_by_emit_floor` (bool) or simply omit filtered rows **while** full bouts remain in `behavior_bouts.parquet`.

#### `package_manifest.json` (required)

Lists every required upstream artifact path, digest, schema version, export profile id, and whether media was generated. Intended as the S10 open-package entrypoint.

#### `ethogram_summary.parquet` / summary blocks in `qc.json` (derived)

Cached label % time, bout counts/duration histograms, parallel overlap counts — avoid S10 recomputing full-table scans on open.

#### S7 probabilities (recommendation)

**Do not** copy the complete S7 probability tables into the default package. Record S7 run id + digests; S10 loads probabilities **by reference** for inspection. Optional profile may extract a narrow column subset for offline notebooks.

#### S8 rule reconstruction (recommendation)

Keep S8 bout fields (`label_reconstruction_mode`, etc.) and QC counters in-package. Full per-frame diagnostics remain optional upstream reference (`arbitration_diagnostics.parquet` when present).

### 4.3 Default export format (recommendation)

- **Primary structured tables:** Parquet.
- **Provenance / manifests:** JSON (+ YAML settings snapshot).
- **Video:** reference path to S1 prepared video by default (not copied).
- **Clips/GIFs:** opt-in profile only.

---

## 5. Video synchronization contract

### 5.1 Timeline vocabulary (fact)

| Timeline | Authority | Index name |
| --- | --- | --- |
| Prepared video frames | S1 `prepared_sync.npz` (`prepared_frame_idx`, `prepared_time_sec`) + `prepared_video.mp4` | Prepared frame index for OpenCV/ffmpeg seek on prepared video |
| Uniform 120 Hz target | S4/S5 construction from prepared times; shared by S6/S7/S8 | `sample_idx` ≡ `dense_frame_idx`; `time_sec` |
| Prepared frame nearest to each target sample | S5 nearest-neighbor map; echoed as S6 `orig_frame_idx` | `prepared_frame_idx` / `orig_frame_idx` |
| Original/raw camera frames | Legacy Pipeline 110 used prepared meta trim formula; modern S1 sync is prepared-timeline authoritative | Only if trim/raw metadata is already present — **do not invent a new camera model** |

**Fact:** On the E-session, dense length (36 528) ≠ prepared frame span reflected by `orig_frame_idx` max (36 237). Indices are **not** interchangeable.

### 5.2 Deterministic mapping S8 sample/bout → prepared video frame (recommendation)

For sample `i` with `time_sec[i]` on the 120 Hz grid:

1. **Source of truth for prepared times:** S1 `prepared_sync.npz`.
2. **Mapping rule:** nearest-neighbor in time — reuse `blob_features.timing.map_target_nearest_source_frame`.
3. **Tie-break:** equidistant left/right → **earlier (left)** prepared sample (existing S5 rule).
4. **Exact match:** `|t_target - t_source| ≤ 1e-9` s ⇒ exact; `is_exact_source_sample=True`.
5. **Boundaries:** targets outside `[prepared_time_sec[0], prepared_time_sec[-1]]` are invalid — packaging must fail validation rather than clamp silently.
6. **Bout display frame:** for bout `[start_sample_idx, end_sample_idx]` inclusive, S10 should seek to `prepared_frame_idx[start_sample_idx]` for bout start and `prepared_frame_idx[end_sample_idx]` for bout end; mid-bout scrub uses the sync index per sample.
7. **Missing frames / discontinuities:** prepared timeline may have non-uniform source spacing; NN map yields monotonically non-decreasing `orig_frame_idx` (S5 QC). S9 should re-validate monotonicity and bounds against prepared video frame count.
8. **Precision:** time comparisons at 1e-9 s absolute tolerance (S4/S5); display precision may round for UI but stored indices remain int64.

### 5.3 Precompute vs derive (recommendation)

**Precompute `sync_index.parquet` in S9** (cheap relative to video; ~36k rows on E-session). S10 must open packages without rebuilding timing maps or re-reading S6 features solely for sync.

Legacy Pipeline 110’s imputed NPZ + prepared-meta formula is a **legacy path**. Modern S9 should prefer S1 sync + S5 NN semantics already used by S5/S6.

### 5.4 Playback timestamp source of truth (recommendation)

| Use | Clock |
| --- | --- |
| Behavioral analysis / timelines | S8/S6 `time_sec` on 120 Hz grid |
| Prepared-video seek | `prepared_frame_idx` → frame in `prepared_video.mp4` |
| Wall-clock labels in UI | `prepared_time_sec` and/or `time_sec` with explicit labeling — never mix without naming |

---

## 6. S10 consumption requirements (audit only)

| S10 need | Package in S9? | Upstream reference? |
| --- | --- | --- |
| Video playback / seek | Sync index + prepared video path/digest | Prepared video file in place |
| Synchronized behavioral timelines | Reference or embed states; ethogram summary | Full S8 states |
| Clickable bout navigation | Full `behavior_bouts` + optional `bout_index` | — |
| Directional comparison | Directional identity columns from S8 | — |
| Contact / SBS parallel tracks | S8 parallel preds/bouts | — |
| S7 probs / validity | Manifest pointers (+ optional thin extract) | S7 run directory |
| S8 rule diagnostics | QC + bout reconstruction fields | Optional diagnostics parquet |
| Manual corrections | **No** (S10 publishes versioned reviewed outputs) | Reviews reference S9 package id |
| Versioned reviewed publication | S9 package id is parent provenance | S10 review store |

**Recommendation:** S10 opens an S9 `package_manifest.json`, trusts packaged digests, and streams large tables lazily. Clip generation can remain lazy in S10 if the analysis package omitted media.

---

## 7. Eligibility and provenance

### 7.1 Proposed S9 eligibility requirements (recommendation)

A session may launch S9 packaging when **all** hold:

1. `find_latest_eligible_s8_run` (or explicit eligible picker) returns a run with classification `eligible` or `completed_with_warnings` as allowed by policy.
2. `validate_s8_run_for_downstream(s8_run_dir)` succeeds (S7/S6 digests + nested S4/S5 fingerprint invalidation already implemented).
3. S8 states/bouts schemas match expected stamps (`behavioral_arbitration_states_v1`, `behavioral_arbitration_bouts_v1` or approved successors).
4. Directional key uniqueness on states; bout inclusive endpoints consistent with fps.
5. Timeline alignment: `sample_idx == dense_frame_idx`; `fps == 120` within existing atol; sync map length matches unique sample count.
6. Prepared video + `prepared_sync.npz` exist and match lineage digests recorded through S6/S5/S4 meta (hash prepared_sync; optionally hash video once at package time, cache in S9 meta).
7. Export profile id + settings hash recorded in `settings_used.yaml`.

### 7.2 Historical completion vs current eligibility (fact pattern to preserve)

Mirror S8: never rewrite historical `status` on disk. Discovery distinguishes:

| Classification | Meaning |
| --- | --- |
| `eligible` | Completed package; upstream digests still match |
| `completed_with_warnings` | Usable with documented warnings |
| `stale_source` | Package intact but S8/S7/S6/video digests drifted |
| `failed_or_incomplete` / `missing_required_artifacts` | Not downstream-ready |

### 7.3 GUI hashing guidance (recommendation)

- Eligibility checks should use `memoized_validation` + `artifact_fingerprint` / recorded SHA256 from meta.
- Do **not** rehash prepared video or S6 features on every GUI poll.
- Expensive packaging (sync build, optional clips) runs in a background `GuiTaskKind` equivalent — same pattern as S7/S8.

---

## 8. Performance assessment

| Risk | E-session signal | Mitigation (recommendation) |
| --- | --- | --- |
| Full-table joins S6×S7×S8 | S6 73k×483; S8 states 73k×147 | Package references; join only thin sync + bout tables |
| Video indexing | 417 MB prepared MP4 | Reference + frame seek; optional clip profile offline |
| Large Parquet reads | States ~21 MB acceptable; S6 225 MB not for default open | Column projection / lazy read in S10 |
| Timeline rendering | 36k samples × 2 directions | Use summaries + windowed queries; do not plot all probs by default |
| Cached summaries | Ethogram/bout histograms tiny | Build once in S9 |
| Discovery / eligibility | S8 already memoizes | Extend same fingerprint pattern to S9 |
| Package duplication | Copying states+video would dominate disk | Reference-by-default |
| Clip extraction | Pipeline 110 ffmpeg per bout | Optional; defer to profile / S10 lazy |

**Recommendation:** S9 background task produces a complete navigable package; S10 open path reads manifest + sync index + bouts (+ summaries) without re-running packaging.

---

## 9. Unresolved architectural / scientific decisions

| ID | Question | Evidence-based recommendation | Needs explicit approval? |
| --- | --- | --- | --- |
| U1 | Copy full S8 framewise states into every package vs reference? | **Reference by default**; optional embed profile | Yes |
| U2 | Retain Pipeline 110 emit floors? | **Only as named review export profile**; never default analysis filter | Yes (profile thresholds) |
| U3 | Default export format? | Parquet + JSON/YAML; video by reference | Low risk — confirm |
| U4 | Include complete S7 probability tables? | **No** by default; reference S7 run | Yes |
| U5 | Prepared video copy vs reference for S10? | **Reference** S1 prepared video path | Yes |
| U6 | Multiple export profiles? | **Yes** (`analysis_full`, `review_emit_v1`, optional `portable_embed`) | Yes |
| U7 | Minimum contract for manual review? | Sync index + full bouts + states reference + video path + provenance; clips optional | Yes |
| U8 | Symmetric presentation collapse in default package? | Keep directional S8 truth; collapse only in review index profile | Yes |
| U9 | Contact inclusion in review index? | Legacy default excluded contact; S8 publishes contact bouts — decide lab default for review profile | **Scientific/product** |
| U10 | Engaged inclusion in review index? | Legacy included engaged; confirm lab preference | **Scientific/product** |
| U11 | Raw/original frame columns required? | Only if existing S1 metadata supports them without new sync science | Yes |
| U12 | Should S9 rebuild bouts from states “to match 110”? | **No** — would be class C relative to S8 | Confirm |

---

## 10. Recommended implementation roadmap

Adjust relative to Pipeline 110: media extraction is optional and should not block analysis packaging.

### S9-1 — Contracts, source validation, synchronization indexing

| | |
| --- | --- |
| **Scope** | Package schemas; eligibility against S8; build/validate `sync_index.parquet`; package manifest stubs; no media |
| **Output** | Headless validation + sync index writer; docs S9-B draft alignment |
| **Key tests** | E-session sync length/monotonicity; dense≠prepared assertion; eligibility stale vs eligible; schema stamps |
| **Scientific approvals** | U1, U5, U11 mapping authority (S1+S5 NN) |
| **Acceptance** | Given eligible S8 + S1 sync/video, produces validated sync index and manifest without modifying S8 |

### S9-2 — Packaging transformations and export profiles

| | |
| --- | --- |
| **Scope** | Copy/stamp full S8 bouts; states reference/embed policy; ethogram summaries; optional `bout_index` profiles (emit floors / label subsets / presentation collapse) **without** mutating full bouts |
| **Output** | Analysis package + review profile index |
| **Key tests** | Full bout count preserved; filtered index counts match floors; no label reinterpretation vs S8; profile provenance stamped |
| **Scientific approvals** | U2, U6, U8, U9, U10, U12 |
| **Acceptance** | Package usable for analysis with zero floors; review profile explicitly documented |

### S9-3 — Immutable publication, QC, discovery

| | |
| --- | --- |
| **Scope** | Atomic run dirs; QC; discovery classifications; digest/memoization; optional clip/GIF profile behind flag |
| **Output** | Discoverable S9 runs; stale detection when S8/video digests change |
| **Key tests** | Publish lifecycle; stale after S8 replace; GUI-safe memoized eligibility |
| **Scientific approvals** | Clip profile necessity (media vs lazy S10) |
| **Acceptance** | Completed packages reopen without recompute; eligibility ≠ historical status |

### S9-4 — GUI integration and laboratory validation

| | |
| --- | --- |
| **Scope** | Group 4 S9 page; task runner; open package summary; E-session lab checklist; **no S10 editor** |
| **Output** | Operator-packaged E-session result; acceptance checklist |
| **Key tests** | GUI smoke; responsiveness (no video hash on poll); package open &lt; interactive budget |
| **Scientific approvals** | Lab sign-off on default vs review profiles |
| **Acceptance** | Lab packages E-session; opens package metadata/timelines without S10 |

**Stop after each sprint documentation/implementation gate as directed; this audit does not begin S9-1.**

---

## 11. Architectural recommendations (summary)

1. **Treat Pipeline 110 as a packaging/review ancestor, not the scientific authority.** S8 already superseded Pipeline 100 for states/bouts; S9 should package S8, not recreate Pipeline 110’s re-collapse as truth.
2. **Default S9 package = sync index + full S8 bouts + S8/S7/S6/S1 provenance + ethogram summaries + video/sync references.**
3. **Emit floors and presentation collapses belong in optional export profiles**, with full unfiltered S8 bouts always retained.
4. **Prefer modern S1 `prepared_sync` + S5 nearest-neighbor mapping** over legacy imputed-NPZ naming; fix the legacy `*_prepared_idx` misnomer with explicit dense vs prepared columns.
5. **Do not copy prepared video or full S6/S7 tables by default.**
6. **S10 consumes S9 packages;** manual annotations are S10 outputs that point back to an S9 package id.
7. **Reuse S8 eligibility, digest memoization, and atomic publish patterns** rather than inventing a parallel provenance system.

---

## 12. Document index (this audit)

1. Legacy Pipeline 110 source and stage inventory — §1  
2. Upstream dependency map — §2  
3. S8→S9 scientific boundary — §3  
4. Proposed S9 package schema — §4  
5. Video synchronization contract — §5  
6. S10 consumption requirements — §6  
7. Eligibility and provenance — §7  
8. Performance assessment — §8  
9. Unresolved decisions — §9  
10. Recommended implementation roadmap — §10  

---

_End of S9-A architecture audit. Accepted; binding detail continues in [`implementation_contract.md`](implementation_contract.md) (S9-B). Documentation only; awaiting architectural review of S9-B before implementation._
