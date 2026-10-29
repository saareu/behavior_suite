# Subsystem 09 — Results Packaging & Export

## Implementation Contract (S9-B)

**Status:** Binding design contract. **S9-B accepted; Sprint 1–4 implemented** (headless + Group 4 GUI). U-R1–U-R8 resolved. **Operator laboratory acceptance pending** (automated Sprint 4 evidence recorded).  
**Date:** 2026-10-28 (Sprint 4 GUI integration)  
**Authority:** Accepted S9-A audit (`architecture_audit.md`) + source inspection of:

- S8 package `src/behavioral_arbitration/` (states / bouts / QC / discovery / `validate_s8_run_for_downstream`)
- S7 `src/behavioral_inference/` prediction + provenance contracts
- S6 `src/combined_features/` (`orig_frame_idx`, lineage to S4/S5/`prepared_sync`)
- S5 `src/blob_features/timing.py` nearest-neighbor map
- S4 `src/pose_finalization/timing.py` uniform 120 Hz target timeline
- S1 `src/preprocess/sync_writer.py` (`prepared_sync_v1`)
- Local git-ignored `legacy/pipelines/110_bout_browser_packager.py` + `legacy/configs/bout_browser.yaml`
- E-session artifacts under `test_cases/E_full_session_integration/` (S8 run `behavioral_arbitration__20261028T133953` / S9 `results_package__20261028T181714`)
- Group 3/4 GUI patterns under `src/ui/`

**Follow-on:** Sprint 4 delivered Group 4 S9 packaging GUI. Stop for architectural review. Do **not** implement S10. Do not alter S1–S8 science. Do not commit/push unless explicitly requested.

S9 assembles an immutable, analysis-ready package from one currently eligible S8 run plus validated synchronization and provenance references. It does **not** re-arbitrate behavior, re-collapse scientific bouts, train models, host ethogram/video review, or publish versioned manual corrections (S10).

---

## 0. Frozen scope and scientific boundaries

### 0.1 Scientific responsibility (frozen)

| Responsibility | In S9? | Notes |
| --- | --- | --- |
| Source validation (eligible S8 + lineage) | **Yes** | Structural / provenance only |
| Temporal and identity indexing (`sync_index`) | **Yes** | Derived from S1 + S5 NN policy |
| Results packaging (full S8 bouts + summaries) | **Yes** | Copy/stamp; no science rewrite |
| Optional export-specific views | **Yes** | Explicit profiles; not default science |
| Summaries and QC | **Yes** | Structural + packaging QC |
| Immutable publication and discovery | **Yes** | Mirror S5–S8 lifecycle |
| New behavioral arbitration | **No** | S8 owns states / exclusive labels / parallel channels |
| Silent re-collapse, merge, split, or rewrite of S8 bouts | **No** | Forbidden |
| Dropping short S8 bouts from the default package | **No** | Full S8 bout collection retained |
| Interactive review / versioned corrections | **No** | S10 |
| Materializing full S6 feature or full S7 probability tables | **No** | Reference by default |
| Threshold optimization / new scientific floors | **No** | Forbidden |

### 0.2 Binding decisions (S9-B)

| ID | Decision |
| --- | --- |
| P1 | **S8 is the scientific authority** for exclusive ethogram states, parallel contact / SBS, and bout endpoints. S9 packages; it does not reinterpret. |
| P2 | **Default package retains every authoritative S8 bout**, including 1-frame exclusive bouts (E-session has 31 exclusive bouts with `bout_duration_frames < 10`). |
| P3 | **`behavior_bouts.parquet` in the S9 package is a lossless annotated copy** of the complete S8 bout table (plus package stamps). It is **not** a filtered Pipeline-110-style index. |
| P4 | **`bout_index.parquet` is optional and profile-scoped** — a derived navigation / review index that **references** package bout IDs. It never replaces the full bout table. |
| P5 | **Pipeline 110 emit floors** (`min_emit_bout_frames` / `min_emit_bout_seconds`) exist **only** in named export profiles. They are presentation filters, not scientific truth. |
| P6 | **Any operation that changes bout boundaries** (merge/split/pad/resegment) is a **derived transformation** requiring later explicit approval. It is **out of MVP** export profiles. |
| P7 | **Framewise S8 states are referenced by digest/path by default**; optional portable embed is a named profile. |
| P8 | **Prepared video and full S6/S7 tables are referenced**, not copied, in the default lightweight package. |
| P9 | **Synchronization authority** is S1 `prepared_sync.npz` + S5 nearest-neighbor mapping (earlier-frame tie-break). Do not treat S8 120 Hz samples as prepared-video frames. |
| P10 | **No silent fallback** to another S8/S7/S6 run or another video when a referenced file is missing, moved, modified, or stale. |
| P11 | **Historical `status` on disk is never rewritten**; discovery separates completion from current eligibility. |
| P12 | **S9 GUI is packaging-only** (Group 4). No ethogram viewer, video playback, or manual editing. |
| P13 | **Expensive validation / hashing never blocks the Qt main thread** (reuse S7/S8 `GuiTaskKind` + memoization patterns). |

### 0.3 Conflicts and disagreements (do not silently resolve)

| ID | Topic | Observation | Contract stance |
| --- | --- | --- | --- |
| D1 | Pipeline 110 re-collapses framewise labels into bouts | Legacy `_build_bouts` rebuilds contiguous segments from presentation rows | **S9 consumes S8 `behavior_bouts.parquet`**; never replace S8 bouts with a re-collapse as scientific truth |
| D2 | Legacy column misnomer `start_prepared_idx` / `end_prepared_idx` | Store **dense** indices, not prepared-video indices (`110_bout_browser_packager.py` `_build_bouts` / `_build_manifest`) | S9 uses explicit `start_sample_idx` / `end_sample_idx` (dense) and separate `prepared_frame_idx_*` columns |
| D3 | Emit-floor fps mixing | Legacy: `min_required_frames = max(min_emit_bout_frames, ceil(min_emit_bout_seconds * source_video.fps))` while counting **dense** segment length | Documented legacy fact. S9 profile `review_emit_v1` uses **explicit 120 Hz target fps** for the seconds criterion (see §4.3 / §5). This is a **deliberate difference from legacy Pipeline 110**, which multiplied seconds by **source_video.fps**. Numerical equivalence of “10 frames ≡ 0.10 s” is **not** assumed when source fps ≠ 120 |
| D4 | Dense vs prepared length | E-session: 36 528 dense samples (`0…36527`); S6 `orig_frame_idx` span `0…36237` (36 238 unique prepared frames) | Indices are **not** interchangeable; sync validation must assert this class of mismatch is expected |
| D5 | Legacy default excludes contact from review emit | YAML `include_contact_labels: false`; S8 publishes contact bouts | Default S9 package **always includes** all S8 contact/SBS bouts. Optional `review_emit_v1` defaults to **`include_contact=false`** (**U-R1 resolved**) — only when that profile is explicitly selected |
| D6 | Legacy engaged presentation | YAML `include_engaged_labels: true`; engaged is an exclusive S8 label | Default package always includes engaged exclusive bouts. Optional `review_emit_v1` defaults to **`include_engaged=true`** (**U-R2 resolved**) — only when that profile is explicitly selected |
| D7 | Supplemental SBS from `pred_side_by_side` into exclusive presentation stream | Pipeline 110 can add SBS rows even when exclusive `behavior_label` differs | S8 already publishes parallel SBS bouts. S9 must **not** invent a second exclusive SBS stream |
| D8 | Source-video priority (tracks overlay first) | Pipeline 110 clip seeking prefers overlays | S9/S10 default playback reference is **S1 prepared_video**; overlays remain optional references |

Unresolved items that still need lab approval are listed in §11.

---

## 1. Exact artifact schemas

### 1.1 Immutable run layout

```text
{session_root}/results_package/results_package__YYYYMMDDTHHMMSS/
  sync_index.parquet
  behavior_bouts.parquet          # lossless full S8 bout table + stamps
  ethogram_summary.parquet        # compact cached rollups
  package_manifest.json
  settings_used.yaml
  qc.json
  run_meta.json
  processing_log.txt
  # optional profile outputs (never replace the files above):
  # export_profiles/<profile_id>/bout_index.parquet
  # export_profiles/<profile_id>/profile_manifest.json
  # export_profiles/<profile_id>/clips/...          # deferred / optional
  # export_profiles/<profile_id>/gif_previews/...   # deferred / optional
  # portable/behavior_states.parquet                # embed profile only
```

| Item | Rule |
| --- | --- |
| Package dirname | `results_package` |
| Run id | `results_package__{UTC_or_local_timestamp}` matching S5–S8 convention used by the session |
| Schema stamps | `results_package_sync_v1`, `results_package_bouts_v1`, `results_package_ethogram_summary_v1`, `results_package_manifest_v1`, `results_package_run_meta_v1`, `results_package_qc_v1`, `results_package_settings_v1` |
| Atomic publish | Write under `running` → validate → promote to `completed` / `completed_with_warnings` / `failed` (same pattern as S8) |

### 1.2 `sync_index.parquet` (required)

| Item | Spec |
| --- | --- |
| **Filename** | `sync_index.parquet` |
| **Row grain** | One row per 120 Hz target sample |
| **Primary key** | `sample_idx` (unique); must equal `dense_frame_idx` |
| **Sort** | Ascending `sample_idx` (stable) |
| **Scientific role** | Deterministic dense↔prepared map for analysis / S10 seek |
| **Execution role** | Built once at package time; S10 must not rebuild timing solely to open a package |
| **Consumer** | S10 seek; exporters; offline alignment audits |

#### Required columns

| Column | Type | Null? | Source / derivation |
| --- | --- | --- | --- |
| `schema_version` | str | no | `results_package_sync_v1` |
| `sample_idx` | int64 | no | 120 Hz index; `0 … N-1` contiguous |
| `dense_frame_idx` | int64 | no | Identical to `sample_idx` (S6/S7/S8 contract) |
| `time_sec` | float64 | no | Uniform target time from S4/S5 construction (`build_uniform_target_timeline`); must match S8 unique `(dense_frame_idx → time_sec)` within `1e-9` s |
| `prepared_frame_idx` | int64 | no | S5 NN map = modern `orig_frame_idx`; index into prepared video / `prepared_sync.prepared_frame_idx` |
| `prepared_time_sec` | float64 | no | `prepared_sync.prepared_time_sec[prepared_frame_idx]` (mapped source time) |
| `is_exact_source_sample` | bool | no | True when `|time_sec - prepared_time_sec| ≤ 1e-9` |
| `timing_residual_sec` | float64 | no | `time_sec - prepared_time_sec` (signed) |

#### Optional columns (include when S1 metadata supports without new sync science)

| Column | Type | Null? | Source |
| --- | --- | --- | --- |
| `raw_decode_frame_idx` | int64 | yes | `prepared_sync.raw_decode_frame_idx[prepared_frame_idx]` (= `start_frame + prepared_frame_idx`) |
| `raw_pts_time_sec` | float64 | yes | `prepared_sync.raw_pts_time_sec[prepared_frame_idx]` when status permits |
| `package_run_id` | str | no | Stamp |

#### Construction rule (binding)

Prefer rebuild via:

1. Resolve lineage `prepared_sync.npz` from S8 → S7 → S6 → S4/S5 meta (see §6).
2. `build_nearest_neighbor_timing_map_from_prepared_sync(path)` (`blob_features.timing`).
3. Cross-check against unique S8 `(dense_frame_idx, time_sec)` and, when available, unique S6 `(dense_frame_idx, orig_frame_idx, time_sec)`.

If rebuilding and copying from S6, **hash-match** `orig_frame_idx` / times; mismatch is a structural blocker. Do not invent a third sync model.

### 1.3 `behavior_bouts.parquet` (required; lossless full bout table)

| Item | Spec |
| --- | --- |
| **Filename** | `behavior_bouts.parquet` |
| **Row grain** | Identical to S8 bout grain: one row per S8 bout (`channel` ∈ `exclusive` \| `contact` \| `side_by_side`) |
| **Primary key** | `bout_id` unique within package; also retain `source_s8_bout_id` |
| **Relationship to S8** | **Lossless annotated copy** of the complete S8 bout table. Not a separate scientific index. Not filtered by emit floors. |
| **Sort** | Stable: `channel`, `start_sample_idx`, `behavior_name`, `pair_id`, `focal_id`, `partner_id` |
| **Consumer** | Analysis; S10 bout navigation; provenance parent for profile indexes |

#### Required columns

All S8 `BOUT_COLUMNS` from `behavioral_arbitration.bouts.BOUT_COLUMNS`, plus package stamps:

| Column | Type | Null? | Notes |
| --- | --- | --- | --- |
| `bout_id` | str | no | Package-local id (may equal or namespace S8 id; must be unique) |
| `source_s8_bout_id` | str | no | Exact S8 `bout_id` string |
| `source_s8_run_id` | str | no | From S8 meta / bout `run_id` |
| `package_run_id` | str | no | This S9 run |
| `schema_version` | str | no | `results_package_bouts_v1` (package stamp; also retain S8 bout schema in provenance) |
| `s8_bouts_schema_version` | str | no | Copied S8 `schema_version` (`behavioral_arbitration_bouts_v1`) |
| `behavior_name` | str | no | Ethogram or parallel name |
| `channel` | str | no | `exclusive` \| `contact` \| `side_by_side` |
| `identity_grain` | str | no | `directional` \| `symmetric_pair` |
| `focal_id` | int64 | **yes** | Required for directional exclusive; **null** for symmetric_pair parallel bouts |
| `partner_id` | int64 | **yes** | Same nullability rule |
| `mouse_a_id` | int64 | no | `min(ids)` |
| `mouse_b_id` | int64 | no | `max(ids)` |
| `pair_id` | str | no | `"{mouse_a}|{mouse_b}"` |
| `start_sample_idx` | int64 | no | Inclusive dense/sample index |
| `end_sample_idx` | int64 | no | Inclusive dense/sample index |
| `start_time_sec` | float64 | yes | S8: `time_sec` at start sample |
| `end_time_sec` | float64 | yes | S8: `time_sec` at **last inclusive** sample (not exclusive end) |
| `bout_duration_frames` | int64 | no | Inclusive sample count |
| `bout_duration_sec` | float64 | no | `bout_duration_frames / fps` with `fps=120` |
| `mean_prob` / `max_prob` / `median_prob` | float64 | yes | As published by S8 |
| `frac_input_valid` | float64 | yes | As published by S8 |
| `label_reconstruction_mode` | str | yes | Evidence / reconstruction provenance |
| `undefined_reason` | str | yes | When applicable |
| `source_s7_run_id` | str | no | Copied |
| `source_s6_run_id` | str | no | Copied |
| `exp_name` | str | yes | Compat |

#### Optional annotation columns (derived navigation; do not alter bout science)

| Column | Type | Null? | Derivation |
| --- | --- | --- | --- |
| `prepared_frame_idx_start` | int64 | no | `sync_index.prepared_frame_idx[start_sample_idx]` |
| `prepared_frame_idx_end` | int64 | no | `sync_index.prepared_frame_idx[end_sample_idx]` |
| `prepared_time_sec_start` | float64 | no | Mapped prepared time at start |
| `prepared_time_sec_end` | float64 | no | Mapped prepared time at end sample |

These prepared columns are **seek aids**. They must not be used to redefine `bout_duration_*`.

**Row-count invariant:** `len(package.behavior_bouts) == len(s8.behavior_bouts)` and multiset equality on `(source_s8_bout_id, channel, behavior_name, start_sample_idx, end_sample_idx, identity_grain, focal_id, partner_id, mouse_a_id, mouse_b_id)`.

### 1.4 `ethogram_summary.parquet` (required; compact behavioral summary)

| Item | Spec |
| --- | --- |
| **Filename** | `ethogram_summary.parquet` |
| **Purpose** | Cached rollups so S10/analysis avoid full-table scans on open |
| **Scientific role** | Descriptive summary of packaged S8 content — not a new ethogram |
| **Consumer** | Package open summary; S10 overview panels |

Recommended grain: one row per `(summary_kind, channel, behavior_name, identity_key)` where `identity_key` may be `pair_id` or `focal_id→partner_id` or session-total (`*`).

Minimum required summary kinds (may be multiple tables concatenated with a `summary_kind` column, or separate logical sections inside one parquet):

| `summary_kind` | Metrics |
| --- | --- |
| `label_time` | Exclusive label frame counts and `%` of directional samples |
| `bout_counts` | Bout counts per channel/behavior; duration histograms (frames + seconds) |
| `parallel_overlap` | Contact/SBS bout counts; overlap-with-exclusive counters from S8 QC when available |
| `session_timing` | `n_samples`, `t0`, `t1`, `fps`, dense vs prepared unique-frame counts |

Also embed headline duplicates in `qc.json` for discovery without reading parquet.

### 1.5 `package_manifest.json` (required)

S10 open-package entrypoint.

| Block | Contents |
| --- | --- |
| Identity | `package_run_id`, `schema_version`, `created_at`, `session_root` |
| Profile | `default_profile_id` (`analysis_full`), list of generated export profile ids |
| Package artifacts | Relative paths + SHA256 + schema versions for every required file |
| Upstream references | See §6 (`source_references`) |
| States policy | `behavior_states_mode`: `reference` \| `embedded`; path + SHA256 |
| Video / sync | Prepared video path + digest (if recorded); prepared_sync path + digest |
| Portability | `package_mode`: `reference_lightweight` \| `portable_embed` |
| Dependency snapshot | Eligibility classification at publish time |

### 1.6 `settings_used.yaml` (required)

Frozen packaging settings snapshot:

- `schema_version: results_package_settings_v1`
- Selected S8 run id / path
- Export profiles requested + parameters (units explicit)
- States embed flag
- Media generation flags (default false for MVP)
- Sync construction method (`rebuild_from_prepared_sync` \| `copy_validated_s6_map`)
- Implanted-id / role mapping **only if** a presentation profile requests it

### 1.7 `qc.json` (required)

| Block | Contents |
| --- | --- |
| Structural | Artifact presence; key uniqueness; sync monotonicity; bout count parity vs S8 |
| Sync audit | `n_dense`, `n_prepared_unique`, `max_many_to_one`, residual stats, exact-match count |
| Bout audit | Per-channel counts; short-bout counts; null focal rules for symmetric channels |
| Provenance | Digests compared; eligibility snapshot |
| Warnings | Nonblocking issues (§7) |
| Profile QC | Per-profile inclusion/exclusion counts |

### 1.8 `run_meta.json` (required)

Mirror S5–S8:

| Field | Meaning |
| --- | --- |
| `run_id`, `status`, timestamps, `session_root` | Lifecycle |
| `source_s8` | run id/dir, states/bouts paths + SHA256, eligibility at publish |
| `source_s7` / `source_s6` | Echoed from S8 meta (+ re-validated digests) |
| `source_sync` / `source_video` | Prepared sync + prepared video references |
| `outputs` | Artifact paths |
| `fps` | `120.0` |
| `package_mode` | `reference_lightweight` \| `portable_embed` |
| `qc_summary` | Headline counters |

Statuses: `running` → `completed` \| `completed_with_warnings` \| `failed`.

### 1.9 `processing_log.txt` (required)

Human-readable stage progress: intake, sync build, bout copy/stamp, summary, optional profiles, publish. No scientific reinterpretation narrative presented as fact.

### 1.10 Explicitly out of default package

| Artifact | Disposition |
| --- | --- |
| Full S8 `behavior_states.parquet` | Reference (default) or embed profile |
| Full S7 prediction tables | Reference only |
| Full S6 `dyadic_features_full.parquet` | Reference only |
| `arbitration_diagnostics.parquet` | Optional upstream reference |
| Clips / GIFs | Optional deferred profile / S10 lazy |
| Prepared video bytes | Reference only (default) |

---

## 2. Synchronization equations and index conventions

### 2.1 Timeline vocabulary (frozen)

| Timeline | Index | Authority | Begins at |
| --- | --- | --- | --- |
| Prepared video frames | `prepared_frame_idx` | S1 `prepared_sync.npz` + `prepared_video.mp4` | **0** … `N_prep-1` contiguous |
| Uniform 120 Hz target | `sample_idx` ≡ `dense_frame_idx` | S4/S5 `build_uniform_target_timeline` from prepared times; shared by S6/S7/S8 | **0** … `N_dense-1` contiguous |
| NN map | `prepared_frame_idx` / legacy name `orig_frame_idx` | S5 `map_target_nearest_source_frame` | Values in `0 … N_prep-1`, monotonically non-decreasing |
| Raw/decode camera frames | `raw_decode_frame_idx` | S1: `start_frame + prepared_frame_idx` when present | Not assumed identical to prepared |

**Binding:** S8 120 Hz samples are **not** prepared-video frames. E-session proof: `N_dense=36528`, unique prepared mapped frames `36238` (`orig_frame_idx` max `36237`).

### 2.2 Timestamp anchoring

| Quantity | Equation / rule |
| --- | --- |
| Prepared times | `prepared_time_sec[p] = prepared_frame_idx[p] / fps_header` (S1 validation) |
| Target grid | `time_sec[i] = t0 + i / 120`, with `t0 = prepared_time_sec[0]`, ending at or before `prepared_time_sec[-1]` (`pose_finalization.timing.build_uniform_target_timeline`) |
| S8 times | Must match this grid per unique `dense_frame_idx` within `1e-9` s |
| Analysis clock | S8/S6 `time_sec` |
| Video seek clock | `prepared_frame_idx` into `prepared_video.mp4` |

### 2.3 Frame-selection equation (nearest neighbor)

For each target sample time `t = time_sec[i]`:

1. Require `prepared_time_sec[0] ≤ t ≤ prepared_time_sec[-1]` within `1e-9` (else **fail** — no silent clamp).
2. Find left neighbor via `searchsorted(..., side="right") - 1`.
3. If `|t - prepared_time_sec[j]| ≤ 1e-9` for a neighbor → exact match; `is_exact_source_sample=True`.
4. Else choose nearer of left/right; **ties (`dist_left == dist_right`) select earlier (left)** — `blob_features.timing.map_target_nearest_source_frame`.
5. `prepared_frame_idx[i] = prepared_sync.prepared_frame_idx[chosen_position]` (identity `[0..N-1]`).
6. `timing_residual_sec[i] = time_sec[i] - prepared_time_sec_mapped[i]`.

### 2.4 Boundary clipping and missing data

| Case | Rule |
| --- | --- |
| Target outside prepared time range | Structural **blocker** |
| Non-finite prepared or target times | Structural **blocker** |
| Non-strictly-increasing prepared times | Structural **blocker** (S1/S5 already require this) |
| Non-monotonic NN `prepared_frame_idx` | Structural **blocker** |
| Missing prepared_sync / video | Structural **blocker** (no alternate video fallback) |
| Final video frame | Last target may map to last prepared frame; many trailing dense samples may share it (many-to-one). Valid. |
| Bout endpoints | Inclusive dense indices → seek prepared frames via sync rows at those samples; do not convert duration into prepared-frame counts |

### 2.5 Many-to-one mappings

Multiple dense samples may share one `prepared_frame_idx`. Expected when `N_dense > N_prep`. QC must report:

- `n_dense`, `n_prepared_unique`
- max multiplicity
- fraction of non-exact mappings

Never expand/contract bout lengths to “unique prepared frames.”

### 2.6 Bout start/end → video frames

For bout `[start_sample_idx, end_sample_idx]` inclusive:

| Need | Mapping |
| --- | --- |
| Bout start seek | `prepared_frame_idx_start = sync_index.prepared_frame_idx[start_sample_idx]` |
| Bout end seek | `prepared_frame_idx_end = sync_index.prepared_frame_idx[end_sample_idx]` |
| Mid-bout scrub | Per-sample sync lookup |
| Scientific duration | Still `bout_duration_frames / 120` from S8 |

Prepared start/end may be equal even when dense duration > 1 (shared prepared frame). That is a display fact, not a bout-length rewrite.

### 2.7 E-session audit tests (required)

On `E_full_session_integration` with S8 `behavioral_arbitration__20261028T164423` (or successor eligible run):

1. `N_dense == 36528`; `sample_idx == dense_frame_idx == 0…36527`.
2. Unique `prepared_frame_idx` count == 36238; max == 36237; **≠** dense length.
3. Sync length equals unique S8 sample count.
4. Monotonic non-decreasing `prepared_frame_idx`.
5. Residuals finite; exact-match flag consistent with `1e-9`.
6. S8 unique times match sync `time_sec`.
7. Optional: rebuilt map equals S6 unique `(dense_frame_idx, orig_frame_idx)`.

---

## 3. Full bout preservation rules

### 3.1 Inclusive endpoints and duration (S8 authority)

From `behavioral_arbitration.bouts`:

| Quantity | Definition |
| --- | --- |
| Endpoints | Inclusive `start_sample_idx`, `end_sample_idx` |
| Contiguity | Segments break when `dense_frame_idx` gap ≠ 1 or label/pred changes |
| `bout_duration_frames` | Inclusive sample count (`len(block)`) |
| `bout_duration_sec` | `bout_duration_frames / fps` with `fps=120` |
| `start_time_sec` / `end_time_sec` | `time_sec` at first / last inclusive sample |

**Do not** redefine duration as `end_time_sec - start_time_sec` (that equals `(frames-1)/fps` for uniform grids).

### 3.2 Channel semantics in the package

| Channel | `identity_grain` | Focal/partner | Pair identity | Behavior |
| --- | --- | --- | --- | --- |
| Exclusive directional ethogram | `directional` | Required ints | `pair_id` still recorded | `behavior_name` = exclusive label (`nose_to_nose`, `nose_to_body`, …, `engaged`, `undefined`, …) |
| Contact | `symmetric_pair` | **null** | `mouse_a_id`/`mouse_b_id`/`pair_id` | `behavior_name=contact` |
| Side-by-side | `symmetric_pair` | **null** | same | `behavior_name=side_by_side` |

**Do not invent a focal direction for symmetric parallel channels.**

Exclusive labels that are scientifically reciprocal (e.g. `nose_to_nose`, `mutually_disengaged`) still appear as **directional exclusive rows** in S8 (both directions). S9 preserves that representation. Optional review profiles may add a presentation collapse (§5) without mutating `behavior_bouts.parquet`.

### 3.3 Provenance fields retained

Copy S8 evidence/reconstruction fields: `label_reconstruction_mode`, `undefined_reason`, probability summaries, `frac_input_valid`, `source_s7_run_id`, `source_s6_run_id`.

### 3.4 Navigation indexes must not change science

Adding `prepared_frame_idx_*`, presentation labels, roles, or filter flags in profile tables must leave:

- bout sample endpoints
- durations
- channel membership
- identity_grain nullability rules

unchanged in `behavior_bouts.parquet`.

### 3.5 Reconstruction consistency check (QC)

Nonblocking or blocking per severity:

- Every exclusive bout’s `[start,end]` samples should agree with contiguous equal `behavior_label` on referenced S8 states for that directional key (spot-check or full check in background validation).
- Parallel bouts agree with contiguous `pred_contact` / `pred_side_by_side` == 1 on undirected pair grain.

Hard fail if systematic mismatch implies wrong S8 source selection; do not “fix” by rewriting bouts.

---

## 4. Optional export-profile semantics

### 4.1 Mechanism

| Item | Rule |
| --- | --- |
| Default profile | `analysis_full` — no bout filtering; full `behavior_bouts.parquet` is the analysis table |
| Profile outputs | Written under `export_profiles/<profile_id>/` only |
| Provenance | Every emitted row references `source_s8_bout_id` / package `bout_id` |
| Lossless? | Default package yes; filtered profiles **no** (selection), unless profile only adds columns |
| Boundary edits | **Out of MVP**; classify as derived transformation needing approval |

### 4.2 Legacy Pipeline 110 emit-floor fact

From `_build_bouts` + YAML:

```text
min_frames_from_seconds = ceil(min_emit_bout_seconds * source_video.fps)
min_required_frames     = max(min_emit_bout_frames, min_frames_from_seconds)
keep bout iff dense_segment_length >= min_required_frames
```

YAML defaults: `min_emit_bout_frames: 10`, `min_emit_bout_seconds: 0.10`.

**Interpretation (fact):** criteria are **combined via `max`**, not alternatives. They are **not** guaranteed numerically equivalent. Legacy multiplies seconds by **source video fps**, while counting **dense** frames — a documented inconsistency (D3).

### 4.3 Profile catalog (MVP)

#### `analysis_full` (default; always present)

| Field | Value |
| --- | --- |
| Parameters | None beyond source S8 selection |
| Bout ops | None (identity) |
| Outputs | Core package artifacts only |
| Lossless | **Yes** |
| Compatibility | Analysis / S10 parent |

#### `review_emit_v1` (optional; Pipeline-110-inspired selection)

| Field | Contract |
| --- | --- |
| Purpose | Human-review index analogous to legacy `bout_index.parquet` without replacing science |
| Required parameters | `min_emit_bout_frames` (int, dense samples @ 120 Hz); `min_emit_bout_seconds` (float seconds); `target_fps` **fixed 120.0** for seconds→frames; `include_contact` (bool); `include_engaged` (bool); `include_side_by_side` (bool); optional `channels` / `behaviors` allow-list; optional `directions` filter for directional rows |
| Selection equation | `min_required = max(min_emit_bout_frames, ceil(min_emit_bout_seconds * 120))`; keep package bout if `bout_duration_frames >= min_required` **and** label/channel policy passes |
| Transform? | **Selection only** in MVP — no boundary rewrite, no resegmentation |
| Output | `export_profiles/review_emit_v1/bout_index.parquet` + `profile_manifest.json` |
| Provenance columns | `bout_id`, `source_s8_bout_id`, `filter_profile_id`, `excluded_reason` omitted (absent rows are excluded); inclusion implied by presence |
| Lossless | **No** (filtered view) |
| Compatibility limits | Not bit-identical to Pipeline 110 when legacy used `source_video.fps ≠ 120`, when 110 re-collapsed labels, when supplemental SBS rows were injected into exclusive presentation, or when `start_prepared_idx` naming differed |
| Defaults for lab review | **Resolved (U-R1/U-R2):** `include_contact=false`, `include_engaged=true`, floors `min_emit_bout_frames=10`, `min_emit_bout_seconds=0.10`, `target_fps=120.0`. These defaults apply **only** when `review_emit_v1` is explicitly selected. They never change authoritative S8 bout boundaries. Seconds→frame conversion uses **120 Hz** (not Pipeline 110 `source_video.fps`) — deliberate legacy difference (D3) |

#### `presentation_symmetric_collapse_v1` (optional; columns-only / view)

| Field | Contract |
| --- | --- |
| Purpose | Review convenience: collapse selected reciprocal exclusive labels to unordered pair presentation |
| Bout ops | MVP: emit **view rows** referencing underlying directional bout ids; **do not** merge endpoints |
| Lossless w.r.t. science table | Full table unchanged |
| Boundary merge of reciprocal directions | **Derived transformation — not MVP** |

#### `portable_embed_v1` (optional)

| Field | Contract |
| --- | --- |
| Purpose | Self-contained archive: embed `behavior_states.parquet` (+ optional narrow S7 extracts) |
| Still references | Prepared video may remain external unless a future media-embed profile is approved |
| Lossless science | Yes for embedded states copy if digests match |

#### Media profiles (`clips_v1`, `gif_v1`)

Deferred behind explicit approval. Prefer S10 lazy clip generation for MVP. If added later: ffmpeg slices by **prepared** indices for prepared video; never silently switch source video kinds.

### 4.4 `bout_index.parquet` schema (profile artifact)

| Item | Spec |
| --- | --- |
| Grain | One row per **included** bout after profile selection |
| PK | `bout_index_id` or (`filter_profile_id`, `bout_id`) |
| Required | `bout_id`, `source_s8_bout_id`, `filter_profile_id`, channel/behavior/identity columns copied from package bouts, dense start/end, duration, prepared start/end, times |
| Optional presentation | `presentation_label`, `direction_label`, `focal_role`/`partner_role` (I/NI), `implanted_id` config stamp |
| Must not | Omit the full `behavior_bouts.parquet`; rewrite durations; null out evidence fields on the scientific table |

---

## 5. Source references and dependency behavior

### 5.1 Reference manifest (`package_manifest.json` / `run_meta.source_*`)

Must record:

| Reference | Fields |
| --- | --- |
| S8 run | `run_id`, `run_dir` (relative + absolute-at-publish), states/bouts/qc/meta paths, SHA256, schema versions, historical status, eligibility-at-publish |
| S7 run | From S8 meta; predictions + wide digests |
| S6 run | Features path + SHA256; nested S4/S5 prepared_sync digests |
| Prepared sync | Path + SHA256 + `prepared_sync_v1` |
| Prepared video | Path + SHA256 when available from S1/S5 provenance (`prepared_video_sha256`); size/timing metadata optional |
| Sync map method | rebuild vs validated S6 copy |
| Package mode | lightweight vs portable |

### 5.2 Lightweight vs portable

| Mode | Contents | Use |
| --- | --- | --- |
| `reference_lightweight` (default) | sync_index + full bouts + summaries + manifests; large binaries by path/digest | Lab NAS workflows; S10 |
| `portable_embed` | Above + embedded states (+ optional thin extracts) | Offline handoff without S8 directory |

Neither mode silently substitutes alternate upstream runs.

### 5.3 Dependency failure matrix

| Situation | Package behavior |
| --- | --- |
| Referenced file moved but digest reachable via updated path policy | **No silent path guess.** Operator must repair references / re-package. Discovery → missing/stale |
| Missing S8/S7/S6/video/sync | Cannot publish new package; existing package → `missing_required_artifacts` / ineligible |
| Modified file (digest mismatch) | `stale_source`; do not open as eligible analysis parent |
| Unavailable NAS | Soft failure at access time; historical status unchanged; current eligibility false |
| Present but stale vs provenance | `stale_source` |
| Operator selects different eligible S8 | Creates a **new** package run — never mutates old package |

### 5.4 Eligibility reuse

Launch gating:

1. `find_latest_eligible_s8_run` or explicit picker.
2. `validate_s8_run_for_downstream(s8_run_dir)` succeeds.
3. Resolve prepared_sync via S6/S5/S4 lineage recorded in S8→S7→S6 meta; verify digests.
4. Confirm prepared video exists for packaging that needs seek references (required for S10-ready packages).

---

## 6. QC and current eligibility

### 6.1 Structural blockers (fail publish / ineligible)

- Missing or ineligible S8 inputs (status not completed(+warnings); failed downstream validation)
- Broken S7/S6 lineage or digest mismatch
- Missing prepared_sync or unresolved prepared video when required by package mode
- Invalid timeline mapping (length mismatch, out-of-range targets, non-monotonic map, non-finite times)
- Duplicate or nonmonotonic sync keys; `sample_idx != dense_frame_idx`
- Unmapped sample ranges (targets outside prepared times)
- Bout table not 1:1 with S8 authoritative bouts (count/key mismatch)
- Bouts inconsistent with final states at a level indicating wrong source (hard)
- Missing required package artifacts after write
- Non-finite or invalid timing values in sync or bout times

### 6.2 Nonblocking warnings (`completed_with_warnings`)

- Optional raw/decode columns unavailable
- Optional profile omitted / skipped by config
- High many-to-one multiplicity (informational)
- S8 itself was `completed_with_warnings` (echo)
- Review profile excludes short bouts (record counts)
- Prepared video digest missing historically but file present (hash-on-publish once; warn if only path known)
- Spot-check reconstruction discrepancies below hard threshold (define in implementation as count-based warn)

### 6.3 Explicitly not QC gates

- No arbitrary scientific quality thresholds (e.g. minimum ethogram purity)
- No new emit floors in default package
- No “prefer longer bouts” heuristics

### 6.4 Discovery classifications

| Classification | Meaning |
| --- | --- |
| `eligible` | Completed(+warnings) package; upstream digests still match; references resolvable |
| `completed_with_warnings` | Usable; warnings documented |
| `stale_source` | Package intact; S8/S7/S6/sync/video digests drifted or upstream ineligible |
| `failed_or_incomplete` | Publish failed / running |
| `missing_required_artifacts` | Core files absent |

**Never rewrite** historical `run_meta.status` when eligibility changes.

---

## 7. S10 consumption contract (minimum)

S10 is not designed here beyond the data-access surface.

### 7.1 Open path

1. Load `package_manifest.json`.
2. Verify package digests for local artifacts.
3. Load `sync_index.parquet`, `behavior_bouts.parquet`, `ethogram_summary.parquet`, `qc.json`.
4. Resolve prepared video via manifest reference (fail clearly if missing/stale).
5. Resolve S8 states path (or embedded states) for framewise display.
6. Resolve S7/S8 diagnostic references lazily as needed.

### 7.2 Operation matrix

| S10 need | Materialized in S9? | Lazy upstream? |
| --- | --- | --- |
| Load one package | Manifest + core tables | — |
| Resolve prepared video | Path/digest in manifest | Open video file in place |
| Seek to S8 sample | `sync_index` | — |
| Seek to bout | `behavior_bouts` + sync annotations | — |
| Display both mouse directions | Directional exclusive bouts + referenced states | States columns projected |
| Parallel contact/SBS | Package bouts `channel∈{contact,side_by_side}` | Optional state preds |
| Selected S7 probability windows | Manifest S7 pointers | Windowed parquet read |
| S8 diagnostic windows | QC + bout reconstruction fields | Optional diagnostics parquet |
| Correction layer | **No writes into S9/S8** | S10 versioned store references `package_run_id` |

### 7.3 Correction boundary

S10 creates additive reviewed outputs pointing at `package_run_id` + `source_s8_bout_id`. S9 packages remain immutable.

---

## 8. GUI and execution boundary

### 8.1 Placement

Group 4 — Behavioral Results & Review navigation in `main_window.py` (**Sprint 4**):

- Group 4 button alongside Groups 1–3
- Page: **Subsystem 09 — Results Packaging & Export**
- S10 page reserved later; not in S9 MVP (no empty interactive S10 page)

### 8.2 Page capabilities

| Support | Yes |
| --- | --- |
| Source S8 selection (eligible + explicit) | Yes |
| Input readiness / lineage summary | Yes |
| Package/export settings (profiles, embed flag) | Yes |
| Background execution | Yes (`GuiTaskKind` packaging + discovery) |
| Progress and warnings | Yes |
| Output summary (counts, sync audit, paths) | Yes |
| Completed-package discovery and reopen | Yes |

| Must not include | |
| --- | --- |
| Ethogram viewer | |
| Video playback | |
| Manual bout editing | |

### 8.3 Responsiveness rules (from S7/S8 lessons)

- Discovery/eligibility via `memoized_validation` + artifact fingerprints
- Do **not** rehash prepared video or S6 features on GUI poll timers
- Sync build, digest compute, optional media in worker threads
- UI reads cached summaries; heavy parquet scans stay off the main thread

---

## 9. Sprint roadmap

### S9-1 — Input contracts, validation, sync-index construction

| | |
| --- | --- |
| **Scope** | Package schemas/constants; S8 eligibility intake; lineage resolve for prepared_sync/video; build/validate `sync_index.parquet`; stub `package_manifest` / meta; no media; no profiles |
| **Deliverables** | `src/results_packaging/` skeleton; headless sync writer; unit/integration tests; this contract followed |
| **Dependencies** | Accepted S9-B; reusable `blob_features.timing`, S8 discovery APIs, `project.file_digest` / `validation_memo` |
| **Tests** | E-session dense≠prepared assertions; monotonicity; residual/exact flags; eligibility stale vs eligible; schema stamps; no S8 mutation |
| **Acceptance** | Given eligible S8 + resolvable S1 sync, produces validated `sync_index` + manifest stub without modifying S1–S8 science |

### S9-2 — Full bout packaging, summaries, optional export profiles

| | |
| --- | --- |
| **Scope** | Lossless `behavior_bouts` copy/stamp + prepared seek annotations; `ethogram_summary`; `review_emit_v1` filtered `bout_index` (selection only); settings snapshot |
| **Deliverables** | Analysis package writer; profile emitter; tests for 1:1 bout preservation and floor counts |
| **Dependencies** | S9-1 |
| **Tests** | E-session 609 bouts preserved; 31 short exclusive retained in full table; review profile drops exactly those failing `max(10, ceil(0.10*120))` when enabled; no label rewrite; symmetric channels keep null focal |
| **Acceptance** | Default package scientifically complete; profile provenance stamped; no boundary transforms |

### S9-3 — Immutable publication, provenance, QC, discovery

| | |
| --- | --- |
| **Scope** | Atomic run dirs; full QC; discovery classifications; digest memoization; portable embed / clip/GIF **deferred** (U-R3/U-R4) |
| **Deliverables** | Publish/discover APIs; stale detection when S8/video/sync digests change |
| **Dependencies** | S9-2 |
| **Tests** | Lifecycle; stale after sync replace; fingerprint semantics; historical status preserved; E-session publish |
| **Acceptance** | **Met (2026-10-28).** Completed packages reopen without recompute; eligibility ≠ historical status |

### S9-4 — GUI integration and E-session laboratory validation

| | |
| --- | --- |
| **Scope** | Group 4 S9 page; task runner; discovery/reopen; lab checklist on E-session; **no S10** |
| **Deliverables** | GUI page/controller; laboratory acceptance checklist doc; Sprint 4 report |
| **Dependencies** | S9-3 |
| **Tests** | GUI smoke; responsiveness; no video hash on poll; packaging in background; package open summary within interactive budget |
| **Acceptance** | **Automated met (2026-10-28).** Group 4 S9 GUI packages E-session and opens package metadata/summaries without S10. **Operator interactive checklist still pending** — S9 is not lab-validated until that sign-off. |

Scope note: media extraction is deferred unless review reopens it after S9-3; it must not block analysis packaging. See [sprint_4_gui_integration.md](sprint_4_gui_integration.md) and [laboratory_acceptance_checklist.md](laboratory_acceptance_checklist.md).

---

## 10. Document map

1. Frozen scope and scientific boundaries — §0  
2. Exact artifact schemas — §1  
3. Synchronization equations and index conventions — §2  
4. Full bout preservation rules — §3  
5. Optional export-profile semantics — §4  
6. Source references and dependency behavior — §5  
7. QC and current eligibility — §6  
8. S10 consumption contract — §7  
9. GUI and execution boundary — §8  
10. Sprint roadmap — §9  
11. Unresolved decisions — §11  

---

## 11. Explicit unresolved / resolved decisions

| ID | Question | Decision / recommendation | Status |
| --- | --- | --- | --- |
| U-R1 | Default `review_emit_v1` `include_contact` | **`false`** for the optional review index; default analysis package still retains all S8 contact bouts | **Resolved (Sprint 1 record)** — profile not implemented in Sprint 1 |
| U-R2 | Default `review_emit_v1` `include_engaged` | **`true`** for the optional review index; default analysis package retains engaged | **Resolved (Sprint 1 record)** — profile not implemented in Sprint 1 |
| U-R3 | Ship `portable_embed_v1` in MVP or post-MVP? | **Deferred beyond MVP.** Default package remains `reference_lightweight` (not self-contained). | **Resolved (Sprint 3)** |
| U-R4 | Clips/GIFs in S9 vs S10 lazy | **Deferred.** S10 will initially use lazy prepared-video access; S9 does not generate clips/GIFs. | **Resolved (Sprint 3)** |
| U-R5 | Exact package-local `bout_id` scheme | Preserve S8 `bout_id` as `source_s8_bout_id`; package `bout_id = {source_s8_run_id}__{source_s8_bout_id}` | **Resolved (Sprint 2)** — retained |
| U-R6 | Require prepared-video SHA256 at package time always? | **Reuse trusted existing content hashes.** Use clearly labeled **size/mtime** fingerprints when content hashes are unavailable. Optional full video SHA256 (`hash_prepared_video=true`) may run off the GUI thread and must **not** become an implicit expensive operation. **Never silently equate size/mtime with SHA256.** | **Resolved (Sprint 3)** |
| U-R7 | Presentation symmetric collapse profile in MVP? | **Do not implement** additional symmetric collapse in S9 MVP. | **Resolved (Sprint 3)** |
| U-R8 | Hard-fail vs warn on limited bout↔state reconstruction mismatches | **Publish reconstruction counts** and scientifically meaningful warnings **without arbitrary quality thresholds.** Hard-fail only when mismatches imply wrong-source selection (e.g. all exclusive bouts disagree with referenced states). | **Resolved (Sprint 3)** |

### 11.3 Sprint 3 publication / fingerprint addendum

1. Immutable runs use `{session_root}/results_package/results_package__YYYYMMDDTHHMMSS/`.
2. Required artifacts: `sync_index.parquet`, `behavior_bouts.parquet`, `ethogram_summary.parquet`, `package_manifest.json`, `settings_used.yaml`, `qc.json`, `run_meta.json`, `processing_log.txt`.
3. Optional profiles write under `export_profiles/<profile_id>/` only; default package does **not** create top-level `bout_index.parquet`.
4. Publication failure deletes science artifacts (tables/manifest/qc/profiles) and preserves `run_meta` / `settings_used` / `processing_log` with `status=failed`.
5. Historical `run_meta.status` is never rewritten when upstream references later go stale; discovery uses a separate classification.
6. Manifest distinguishes embedded science digests (`sha256` / `content_hash`) from lifecycle path records and external references; size/mtime fingerprints are explicitly labeled and are not SHA256 verification.

### 11.2 Sprint 2 identity / profile addendum

1. S8 published `bout_id` values are retained verbatim as `source_s8_bout_id`.
2. Package-local `bout_id` namespaces the S8 id with `source_s8_run_id` for global uniqueness and stable re-packaging.
3. `review_emit_v1` is implemented as an optional selection-only view; default package remains unfiltered (`analysis_full`).
4. Seconds→dense-frame emit floors use **120 Hz** only; source-video FPS parameters are rejected.
5. When multiple omission conditions apply, the primary `excluded_reason` is the first match in the frozen priority list; all matches are recorded in `matched_reasons`.

### 11.1 Binding profile-default addendum (Sprint 1)

1. The default S9 package (`analysis_full`) **always retains all S8 bouts**, including contact, engaged, and short-duration events.
2. Optional `review_emit_v1` defaults: `include_contact=false`, `include_engaged=true`, floors `10` frames / `0.10` s at **`target_fps=120.0`**.
3. Profile defaults apply **only** when the optional profile is explicitly selected.
4. **No profile may change authoritative S8 bout boundaries** (selection/annotation only in MVP).
5. Seconds→dense-frame conversion for emit floors uses **120 Hz**, a deliberate difference from legacy Pipeline 110’s use of `source_video.fps` while counting dense frames (D3).

---

## 12. Evidence index (repository)

| Topic | Evidence |
| --- | --- |
| S8 bout columns / inclusive duration | `src/behavioral_arbitration/bouts.py` (`BOUT_COLUMNS`, `_bout_row`) |
| S8 no emit floors | S8-B P8; `bouts.py` module docstring |
| S8 discovery / downstream validation | `runner.validate_s8_run_for_downstream`, `run_discovery.py` |
| S5 NN + left tie-break | `src/blob_features/timing.py` |
| S1 prepared index/time | `src/preprocess/sync_writer.py` (`prepared_frame_idx=[0..N-1]`, `prepared_time_sec=idx/fps_header`, `raw_decode=start+idx`) |
| S6 `orig_frame_idx` | `combined_features` contracts / pose adapter |
| Legacy emit floor combo | `legacy/pipelines/110_bout_browser_packager.py` `_build_bouts`; `legacy/configs/bout_browser.yaml` |
| Legacy dense-in-`prepared_idx` misnomer | `_build_bouts` stores dense; `_build_manifest` maps via `orig_frame_idx` |
| E-session scale | S8 run `behavioral_arbitration__20261028T164423`: 36 528 dense; 609 bouts; 31 exclusive `<10` frames; prepared map max 36 237 |

---

_End of S9-B implementation contract. Documentation only; stop for architectural review before S9-1._
