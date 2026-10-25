# Subsystem 06 — Combined Pose & Video Feature Generation

## Architecture Audit (S6-A)

**Status:** Accepted as discovery milestone (S6-A). Binding implementation contracts are in [`implementation_contract.md`](implementation_contract.md) (S6-B).  
**Date:** 2026-10-21  
**Scope:** Documentation and design only. No S6 implementation in this sprint.  
**Authority:** Repository source under `src/pose_finalization`, `src/blob_features`, `src/ui`, plus local legacy Pipeline 50 / `frame_features` / model bundles.

---

## 0. Preconditions verified

### 0.1 Subsystem 05 closure

| Check | Result |
| --- | --- |
| HEAD commit | `b0d54f8` — *Add Subsystem 05 blob feature extraction MVP and mark it lab-validated.* |
| Branch | `main` tracking `origin/main` |
| Docs | `docs/subsystem_05/mvp_scope_and_roadmap.md` marks S5 **MVP complete / lab-validated** |
| GUI | Group 2 Feature Processing hosts S5; S6 button present and disabled |

S5 scientific behavior must remain unchanged. S6 consumes completed eligible S5 artifacts only.

### 0.2 Local legacy availability (critical)

Pipeline 50 and helpers live under `legacy/` on this workstation:

| Path | Role |
| --- | --- |
| `legacy/pipelines/50_dyadic_features.py` | Orchestrator (blob join + post-FF features) |
| `legacy/src/dynamic_train/frame_features.py` | Core directional pose features (`build_frame_features`) |
| `legacy/src/io/path_utils.py` | Legacy artifact path templates |
| `legacy/pipelines/30_impute.py` | Obsolete upstream of FF (not to be restored) |
| `legacy/models/dyadic/v1/*` | Dyadic detector bundles + `feature_spec.json` |
| `legacy/models/focal/v1/*` | Focal detector bundles + `feature_spec.json` |

**Git note:** `.git/info/exclude` contains `/legacy/`, so this tree is **local-only** and not part of the tracked suite package. S6 implementation must either vendor required modules into `src/` or explicitly bring selected legacy paths into version control. Do not assume CI or other clones have these files.

Suite-root equivalents are **missing**:

- `src/dynamic_train/` — absent
- `src/io/path_utils.py` — absent

---

## 1. Task 1 — Upstream S4 / S5 artifact contracts

### 1.1 Discovery interfaces (prefer these)

| Concern | API | File |
| --- | --- | --- |
| S4 project summary | `summarize_pose_finalization_project(session_root)` | `src/pose_finalization/run_discovery.py` |
| S4 eligible filter | `PoseFinalizationProjectSummary.eligible_runs` | same |
| S4 eligibility bool | `is_s4_run_eligible_for_downstream(s4_run_dir)` | `src/pose_finalization/runner.py` |
| S4 validate | `validate_s4_run_for_downstream(s4_run_dir)` | same |
| S5 project summary | `summarize_blob_features_project(session_root)` | `src/blob_features/run_discovery.py` |
| S5 latest eligible | `find_latest_eligible_s5_run(session_root)` | same |
| S5 eligibility bool | `is_s5_run_eligible_for_downstream(s5_run_dir)` | `src/blob_features/runner.py` |

There is **no** `find_latest_eligible_s4_run` yet. Newest-first ordering is already applied by `_run_sort_key` (`completed_at` → `created_at` → run-id timestamp → mtime). S6 should add a thin helper mirroring S5, or call `eligible_runs[0]`.

### 1.2 S4 eligibility and staleness

`validate_s4_run_for_downstream` requires:

1. Run directory exists; `run_meta.json` present
2. `status ∈ {completed, completed_with_warnings}`
3. `final_pose.parquet` present
4. `source_s3` mapping with `run_dir`
5. Current `validate_accepted_s3_run(source_s3.run_dir)` still succeeds
6. If recorded, `tracked_pose_sha256` matches the current accepted digest

**Not checked today:**

- `prepared_sync` path equality vs recorded path
- `prepared_sync` SHA256 (S4 never stores one)
- S1 video / background digests
- Parquet grid integrity

S1 is referenced only nested under `source_s3.prepared_sync` (+ `session_root`). There is no top-level `source_s1` block.

### 1.3 S5 eligibility and staleness

`validate_s5_run_for_downstream` requires:

1. Completed status + `blob_features.parquet`
2. `source_s1.session_root` present
3. Re-run `validate_s1_for_blob_features(session_root)`
4. Recorded `prepared_sync` path resolve-equals current eligible sync
5. Recorded SHA256s match for sync / video / background when present

S5 is **stricter** on S1 identity than S4. Historical `status` is never rewritten when a run becomes stale.

### 1.4 Parquet schemas and indexing

#### S4 `final_pose.parquet` (`final_pose_v1`)

Source: `pose_finalization.contracts.FINAL_POSE_COLUMNS`, built by `export.build_final_pose_table`.

**Row grain:** long-form — one row per `(sample_idx, track, node)`.

| Column | Role |
| --- | --- |
| `sample_idx` | `int64`, `0 .. N-1` on the 120 Hz grid |
| `time_sec` | `float64`, uniform target clock |
| `track` | `int64` track ID |
| `node` | string node name |
| `x`, `y` | post-smooth coordinates (NaN if unresolved) |
| `x_pre_smooth`, `y_pre_smooth` | pre-SavGol |
| `point_score`, `instance_score` | SLEAP confidence; **not fabricated** for reconstructed samples |
| `provenance` / `provenance_name` | uint8 codes (see §3) |
| `protected_manual_blank` | bool |
| `was_smoothed` | bool |
| `left_source_frame_idx`, `right_source_frame_idx`, `interpolation_weight`, `is_exact_source_sample` | source-frame mapping |

#### S5 `blob_features.parquet` (`blob_features_v1`)

Source: `blob_features.contracts.BLOB_FEATURES_COLUMNS`, built by `export.build_blob_features_table`.

**Row grain:** one row per 120 Hz target sample.

| Column | Role |
| --- | --- |
| `sample_idx` | S4-aligned name; equals target index |
| `dense_frame_idx` | Preferred Pipeline 50 join key; equals `sample_idx` |
| `frame_idx` | Legacy alias; equals `dense_frame_idx` |
| `time_sec` | Same uniform 120 Hz clock constructor as S4 |
| `orig_frame_idx` | Nearest prepared-video frame |
| Blob metrics | `cc_count`, `cc_count_raw`, `min_inter_contour_dist`, occlusion static + temporal columns |

### 1.5 Shared 120 Hz clock vs mapping difference

Both subsystems call `pose_finalization.timing.build_uniform_target_timeline` from the same `prepared_sync.npz` arrays.

| Subsystem | Sample mapping onto prepared video |
| --- | --- |
| S4 | Linear interpolation between flanking source frames (`build_timing_map_from_prepared_sync`) |
| S5 | Nearest-neighbor source frame (`build_nearest_neighbor_timing_map_from_prepared_sync`) |

For the **same sync file**, `sample_idx` / `time_sec` grids are designed to match. Coordinate values and blob features intentionally sample video differently; that is expected.

**There is no joint S4↔S5 grid validator today.** S6 must add one.

### 1.6 Upstream source-selection and timing policy (recommended)

Default auto-selection for a session:

1. **S5:** `find_latest_eligible_s5_run(session_root)` — already encoded.
2. **S4:** newest `eligible_runs[0]` from `summarize_pose_finalization_project` (add `find_latest_eligible_s4_run` for symmetry).
3. **Alignment gate (new):** both runs must share the same session root and the same `prepared_sync` identity:
   - Resolve-equal sync paths (`source_s3.prepared_sync` vs `source_s1.prepared_sync`)
   - Prefer comparing S5’s `prepared_sync_sha256` against a newly computed SHA of the sync path recorded on S4 (S4 does not store the digest)
   - Assert equal `target_sample_count` / `time_sec` sequences (`atol` consistent with S5 QC, e.g. `1e-12`)
4. Never auto-select stale, failed, incomplete, or structurally invalid runs.
5. GUI may allow manual override only among currently eligible + alignment-passing pairs, with an explicit warning if digests differ.

**Risk:** S4 can remain “eligible” after S1 re-prepare if accepted S3 + tracked_pose SHA still validate, while S5 goes stale. The joint sync-identity check is mandatory for S6.

---

## 2. Task 2 — Legacy Pipeline 50 calculation inventory

### 2.1 End-to-end sequence

Entry: `legacy/pipelines/50_dyadic_features.py` → `run_dyadic_feature_producer()`.

`_repo_root` for this script is `legacy/` (parent of `pipelines/`).

| Step | Function | Produces |
| --- | --- | --- |
| 1 | `_load_experiments_and_iteration()` | Reads `legacy/configs/global.yaml` |
| 2 | `build_frame_features(exp, iteration)` | Directional pose feature table |
| 3 | `_attach_blob_features(...)` | Left-join blob columns on `dense_frame_idx` |
| 4 | `_add_geometric_echo_features(...)` | Lagged distance anchors + `rolling_sum_cc_count_60` |
| 5 | `add_head_sweep_features(...)` | Nose/neck projected velocities, intent, radials, rollings |
| 6 | `add_total_occlusion_flag(...)` | `is_focal_lost` / `is_partner_lost` / `is_total_occlusion` |
| 7 | `add_last_known_good_features(...)` | Echo columns (ffill limit 60 under triggers) |
| 8 | `add_is_implanted_feature(...)` | `is_implanted` (ID == 0) |
| 9 | `to_parquet(...)` | `data/dyadic_features/{exp}/{iter}/features_full.parquet` |

Helpers used only inside head-sweep: `_signed_kinematic_intent_series`, `_interpolate_short_gaps`.

### 2.2 Pose input contract of `build_frame_features`

File: `legacy/src/dynamic_train/frame_features.py` (`build_frame_features`, ~L995+).

**Required Pipeline 30 artifacts (obsolete for suite path; adapter replaces them):**

| Artifact | Keys / content |
| --- | --- |
| `dense_pose_120_imputed.npz` | `x`, `y`, `t_120`, `orig_frame_idx`, `impute_stage`, `is_imputed`, `was_refined`, `point_score` — shapes `(n_tracks, n_nodes, T)` |
| `impute_meta.json` | `tracks`, `nodes` |

**Impute stage encoding:**

| Code | Meaning |
| --- | --- |
| 0 | none |
| 1 | linear |
| 2 | PCHIP |
| 3 | Kalman |

**Directional rows:** all ordered pairs `(focal_id, partner_id)` with `focal ≠ partner`; one row per dense frame. Identity columns include `exp_name`, `iteration_name`, `focal_id`, `partner_id`, `dense_frame_idx`, `orig_frame_idx`, `t_120`.

**Quality flags produced by FF (consumed by Pipeline 50 and models):**

- Per-node stage bits: `nose_i_is_pchip`, `nose_i_is_kf`, … (same for neck/spine/tail × i/j)
- `nose_any_imputed_i/j` = **PCHIP OR Kalman only** (linear does **not** count)
- Same rule for `neck/spine/tail_any_imputed_{i,j}`
- `nose_any_refined` = `nose_i_was_refined or nose_j_was_refined`
- `nose_i_sigma` / `nose_j_sigma` from `point_score` via `sigma = 1/score - 1`

Node aliases: `spine_base`↔`spine`, `tail_base`↔`tail` (resolved inside FF).

### 2.3 Features calculated in Pipeline 50 itself

#### Blob join (`_attach_blob_features`) — join only

`BLOB_FEATURE_COLS` matches S5 export columns (static + temporal occlusion set). Prefer `dense_frame_idx`; fallback `frame_idx`. `validate="many_to_one"` after de-duplicating blob key. Missing → NaN.

#### Geometric echo (`_add_geometric_echo_features`)

- `distance_nose_i_spine_j_norm_shift_{10,30}`
- `distance_spine_i_nose_j_norm_shift_{10,30}`
- `distance_nose_i_tail_j_norm_shift_10`
- `distance_tail_i_nose_j_norm_shift_10`
- `rolling_sum_cc_count_60`

#### Head sweep (`add_head_sweep_features`)

Large set of nose/neck velocities, unit directions, projected velocities (i→j and j→i), weighted combined projected velocities (imputation-aware), kinetic energy / max windows, signed kinematic intent family, radial velocity diffs of normalized distances, and multiple rolling means. Uses `*_any_imputed_*` for weighting (`1 - 0.5 * flag`). Combined series gap-filled with `_interpolate_short_gaps(..., max_gap=30)`.

#### Occlusion / echo / implanted

- `is_focal_lost`, `is_partner_lost`, `is_total_occlusion`
- Echo columns registered in `add_last_known_good_features` (occlusion-triggered and imputation-triggered)
- `is_implanted`

### 2.4 Output

| Artifact | Legacy path |
| --- | --- |
| Full feature cache | `data/dyadic_features/{exp}/{iter}/features_full.parquet` (`dyadic_features_full`) |

No separate schema file; model `feature_spec.json` files are the consumer contracts.

### 2.5 Classification for S6 (formulas unchanged)

| Block | Class | Notes |
| --- | --- | --- |
| `build_frame_features` core geometry/kinematics | **Needs input adaptation** | Replace NPZ loader with S4 pose adapter; keep formulas |
| Blob join | **Needs input adaptation** | Session `blob_features/<run_id>/` instead of flat `data/blob_features/...` |
| Geometric shifts / rolling cc | **Directly reusable** | Pure DataFrame ops |
| Head-sweep / intent | **Directly reusable** (logic) after adapted columns exist | |
| Occlusion + last-known-good echoes | **Directly reusable** | |
| `is_implanted` (ID == 0) | **Design decision** | Hardcoded implanted-mouse convention |
| Imputation / PCHIP / refined flags | **Needs input adaptation + design decision** | Map S4 provenance; do not restore P30 |
| Exp/iteration config via `global.yaml` | **Design decision** | Suite is session-centric |
| Pipeline 30 NPZ production | **Out of scope** | Explicitly not restored |

---

## 3. Task 3 — Proposed S4 → feature pose adapter mapping

Goal: present an in-memory (or temporary) structure that `build_frame_features`-equivalent code can consume **without** changing feature formulas, and **without** fabricating unavailable legacy provenance.

### 3.1 Coordinates and nodes

| Legacy NPZ / FF field | S4 `final_pose.parquet` | Mapping |
| --- | --- | --- |
| `x[track, node, t]`, `y[...]` | `x`, `y` pivoted on `(sample_idx, track, node)` | Use **post-smooth** `x`/`y` as the scientific coordinates (matches final export intent) |
| `t_120[t]` | `time_sec` (unique per `sample_idx`) | `t_120[sample_idx] = time_sec` |
| `orig_frame_idx[t]` | Prefer nearest of `left_source_frame_idx` when `is_exact_source_sample`, else NN via weight | Or join S5 `orig_frame_idx` on `sample_idx` for consistency with blob join |
| Node names | `node` strings from S3/S4 skeleton | Alias map: `tail`↔`tail_base`, `spine`↔`spine_base` as FF already does |
| Headstage | Present when skeleton includes `headstage` | S4 excludes headstage from gap fill (`HEADSTAGE_NODE_NAME`); missing headstage coords remain NaN — preserve |

**Do not** feed `x_pre_smooth` / `y_pre_smooth` into feature formulas unless a reviewed decision says otherwise. Default: post-smooth only; retain pre-smooth in adapter metadata for QC.

### 3.2 Track IDs and focal / partner identities

| Legacy | S4 | Mapping |
| --- | --- | --- |
| `tracks` from `impute_meta.json` | Distinct `track` values | Integer track IDs as-is |
| `focal_id` / `partner_id` | Derived | All ordered pairs `i≠j` from available tracks (same as FF) |
| `_i` / `_j` suffixes | — | Focal = i, partner = j on each directional row |

### 3.3 Timestamps and sample indices

| Legacy | S4 / S5 | Mapping |
| --- | --- | --- |
| Dense time index | `sample_idx` | Identity |
| `dense_frame_idx` in FF/P50 rows | `sample_idx` | Identity (required for blob join) |
| `t_120` | `time_sec` | Identity |

### 3.4 Point confidence

| Legacy | S4 | Mapping |
| --- | --- | --- |
| `point_score` | `point_score` | Pass through; NaN where S4 left it NaN (reconstructed samples) |
| `nose_*_sigma` | Derived | Keep FF formula `sigma = 1/score - 1` only when `score > 0` and finite; else NaN |

### 3.5 Imputation / reconstruction provenance (no fabrication)

S4 provenance codes (`pose_finalization.contracts`):

| Code | Name |
| --- | --- |
| 0 | missing |
| 1 | original |
| 2 | identity_corrected |
| 3 | temporal_resample |
| 4 | linear_gap_fill |
| 5 | kalman_gap_fill |
| 6 | protected_manual_blank |

**Proposed adapter outputs for FF quality block:**

| Legacy flag | Proposed S4-derived rule | Notes |
| --- | --- | --- |
| `*_is_kf` | `1` iff provenance == `PROVENANCE_KALMAN_GAP_FILL` | Direct analogue |
| `*_is_pchip` | **Always 0** | PCHIP not produced by S4; obsolete for new models |
| `*_any_imputed_{i,j}` | `*_is_pchip OR *_is_kf` → effectively **Kalman-only** | Matches legacy definition (linear excluded) |
| `is_imputed` (raw) | Optional: gap-fill ∈ {linear, kalman} or resample | Not required by active contact_intact/occluded specs |
| `was_refined` / `nose_any_refined` | **Always 0 / False** | S4 `geometric_refinement` is deferred/disabled in runner settings — do **not** invent refinement |

**Unresolved scientific decisions (see §8):**

1. Should `PROVENANCE_TEMPORAL_RESAMPLE` count toward any quality flag? Legacy had no equivalent bit.
2. Should `PROVENANCE_LINEAR_GAP_FILL` ever contribute to `*_any_imputed_*`? Legacy linear did **not**.
3. Should identity-corrected originals (`provenance=2`) be treated as original for quality?
4. Protected manual blanks: coords NaN; flags should not claim “imputed”.

### 3.6 Smoothing-related information

| S4 field | Adapter use |
| --- | --- |
| `was_smoothed` | QC / metadata only; not a legacy FF input |
| `x_pre_smooth`, `y_pre_smooth` | QC / optional debug export |
| SavGol settings in `run_meta` / `settings_used` | Provenance record on S6 run |

### 3.7 Missing-value semantics

Unresolved coordinates remain NaN in S4. Adapter must preserve NaNs so FF missingness / occlusion heuristics behave as in legacy. Do not fill with zeros.

---

## 4. Task 4 — Model-feature compatibility matrix

### 4.1 Bundle inventory (local disk)

All paths under `legacy/models/`.

#### Active dyadic design (agreed contact split)

| Detector | `n_selected` | Status | Notes |
| --- | --- | --- | --- |
| `contact_intact` | 139 | **Active** | Uses `nose_any_imputed_i/j`, `nose_i/j_is_kf`, `nose_i/j_sigma` |
| `contact_occluded` | 34 | **Active** | Blob + echo heavy; no nose imputation flags |
| `nose_to_nose` | 44 | **Active** | Uses `nose_any_imputed_i/j` |
| `side_by_side` | 26 | **Active** | Mostly geometry + `cc_count` |
| `mutually_disengaged` | 79 | **Active** | Consumes Pipeline 50 intent / projected-vel cols |

#### Archived / unsupported dyadic

| Detector | Status | Blocking issues |
| --- | --- | --- |
| `contact` | **Obsolete** (superseded by intact/occluded) | Requires `nose_any_refined`, unsuffixed `nose_any_imputed` — agreed removals |

#### Focal bundles (present locally; S6 should generate features they need, but inference is out of scope)

| Detector | `n` features | Notes |
| --- | --- | --- |
| `approach` | 71 | Includes `nose_any_imputed_i`, `rolling_mean_intent_15` |
| `retract` | 39 | Includes `nose_any_imputed_i` |
| `nose_body_intact` | 109 | |
| `nose_body_occ` | 26 | |
| `nose_to_body` | 49 | |
| `nose_to_tail` | 133 | Echo-imputed distance features + `tail_any_imputed_j` |
| `nose_body`, `nose_tail` | 0 | Empty specs — unsupported placeholders |

**Retrained suite-packaged model specs:** not present under tracked `src/` or `configs/`. Compatibility checking must read the local `legacy/models/**/feature_spec.json` paths (or a future vendored copy). Report: **no non-legacy model archive is checked into the suite package today.**

### 4.2 Agreed obsolete feature policy

| Feature | Policy |
| --- | --- |
| `nose_any_refined` | Remove from supported model set |
| `nose_any_imputed` (unsuffixed) | Remove |
| PCHIP-specific columns (`*_is_pchip`) | Do not require for active models; adapter may emit zeros |
| `nose_any_imputed_i`, `nose_any_imputed_j` | **Retain** where required |

### 4.3 Detector matrix (active dyadic)

Legend for **Stage**: `FF` = `frame_features.build_frame_features`; `P50` = Pipeline 50 post-process; `S5` = blob parquet join.

#### `contact_intact` (139) — directional rows

| Feature group | Examples | Stage | Dtype (expected) | Directional? | Ambiguity |
| --- | --- | --- | --- | --- | --- |
| Pairwise distances | `distance_nose_i_*_j_norm`, headstage distances | FF | float64 | Yes (i/j) | Node alias `tail`/`spine` |
| Contact summaries | `distance_nose_i_partner_body_min_norm`, `contact_asymmetry_*`, `nose_contact_best_norm`, … | FF | float64 | Yes | |
| Headings | `cos_heading_*`, `mutual_facing`, `heading_to_partner_*` | FF | float64 | Yes | |
| Kinematics | `projected_vel_i_to_j`, `vel_radial_*`, rollings | FF | float64 | Yes | |
| Shifts | `distance_*_shift_{10,30}` | P50 | float64 | Yes | |
| Echoes | `*_echo`, `speed_*_echo` | P50 | float64 | Yes | Needs occlusion/impute triggers |
| Quality | `nose_i/j_sigma`, `nose_any_imputed_i/j`, `nose_i/j_is_kf` | FF←adapter | float / uint8 | Yes | **S4 provenance mapping** |
| Blob | `cc_count`, occlusion static/temporal, `solidity` | S5→P50 | float/int | Undirected (same on both directions) | Join on `dense_frame_idx` |

Missing vs S4 alone: all quality flags require adapter. Missing vs S5 alone: all pose features.

#### `contact_occluded` (34)

| Feature group | Stage | Notes |
| --- | --- | --- |
| Full blob static + temporal set including `solidity_*` | S5 | Intact without solidities still uses most blob cols |
| Echo pose features | P50 | Depends on FF pose + occlusion flag |
| Long-window rollings (`rolling_min_dist_*_360`) | FF | |

No `nose_any_*` dependence — more robust to quality-mapping decisions.

#### `nose_to_nose` (44)

Requires FF geometry + side-distance features + `nose_any_imputed_i/j` + limited blob (`cc_count`, `min_inter_contour_dist`).

#### `side_by_side` (26)

FF body-axis / offset features + `cc_count` / `min_inter_contour_dist`. No imputation flags.

#### `mutually_disengaged` (79)

FF distances/headings + **P50 head-sweep intent columns** (`signed_kinematic_intent*`, `projected_vel_nose/neck_*`) + limited blob. Soft dependency on imputation weights inside head-sweep (defaults to weight 1.0 if flags absent — but semantics change).

### 4.4 Compatibility verdict summary

| Detector | Compatible with S4+S5+adapted FF+P50? | Blockers |
| --- | --- | --- |
| `contact_intact` | **Conditional** | Provenance→`nose_any_imputed_*` / `*_is_kf` policy; node naming |
| `contact_occluded` | **Likely yes** | Echo triggers need occlusion + FF pose |
| `nose_to_nose` | **Conditional** | Same imputed_i/j mapping |
| `side_by_side` | **Likely yes** | |
| `mutually_disengaged` | **Conditional** | Head-sweep + intent; imputation weighting |
| `contact` (old) | **Unsupported** | Obsolete refined/unsuffixed imputed features |
| Focal active set | **Feature-complete goal** for artifact; inference later | Same adapter issues; empty specs unsupported |

S6 should emit the **complete directional feature artifact**, then run a **separate model compatibility checker** that loads selected `feature_spec.json` files and reports missing / dtype / NaN-fraction issues — without running inference.

---

## 5. Task 5 — Proposed S6 architecture

### 5.1 Package structure

```text
src/combined_features/          # proposed name (alt: dyadic_features)
  __init__.py
  contracts.py                  # schema versions, artifact names, status enums
  run_discovery.py              # S6 run discovery + stale classification
  upstream.py                   # select/validate eligible S4+S5 pair + grid align
  pose_adapter.py               # final_pose.parquet → FF input contract
  frame_features_bridge.py      # wrap/vendor build_frame_features (no formula edits)
  pipeline50_post.py            # blob join, geometric echo, head sweep, echoes, implanted
  export.py                     # directional features parquet writer
  qc.py                         # grid align, missingness, join coverage
  model_compat.py               # feature_spec checker (no inference)
  runner.py                     # end-to-end session run
  config.py                     # YAML defaults (120 Hz fixed, pair policy)
  gui_persistence.py            # optional bookmarks / last-used pair

src/ui/controllers/combined_features_controller.py
src/ui/pages/combined_features_page.py
```

Vendor strategy for FF: copy or thin-wrap `legacy/src/dynamic_train/frame_features.py` into the suite package so CI does not depend on excluded `legacy/`. Pipeline 50 post-process functions can move similarly. **Do not change formulas** during port; only replace I/O boundaries.

### 5.2 Input eligibility and alignment

```text
session_root
  → find_latest_eligible_s4_run (new)
  → find_latest_eligible_s5_run (existing)
  → assert same session_root
  → assert prepared_sync path (+ SHA vs S5 digest)
  → assert time_sec / sample_idx grids match
  → load final_pose.parquet + blob_features.parquet
```

### 5.3 Pose adapter interface (sketch)

```text
adapt_final_pose_to_frame_feature_input(final_pose_df, *, run_meta) -> AdaptedPose
  AdaptedPose:
    x, y: (n_tracks, n_nodes, T)
    t_120, orig_frame_idx
    point_score
    impute_stage-equivalent arrays OR explicit flag arrays
    was_refined: zeros
    tracks, nodes
    provenance_map metadata for QC
```

### 5.4 Legacy calculation integration

1. Adapter → FF core (directional table)
2. Attach S5 blob columns via existing join semantics (`dense_frame_idx`)
3. Run Pipeline 50 post-processors in the same order as `run_dyadic_feature_producer`
4. Write complete feature parquet + run_meta + qc + settings_used

### 5.5 Output contract (proposed)

Run layout:

```text
session_root/combined_features/<run_id>/
  combined_features.parquet      # directional rows (complete feature artifact)
  run_meta.json
  settings_used.yaml
  processing_log.txt
  qc.json
  model_compat.json              # separate from feature generation
```

**Minimum identity columns:** `session_id`/`exp_name` (decision), `focal_id`, `partner_id`, `sample_idx`, `dense_frame_idx`, `time_sec`, plus full feature set.

**Provenance in `run_meta`:** source S4 run id/dir + final_pose digest; source S5 run id/dir + blob digest; prepared_sync path + SHA; adapter policy version; FF/P50 code version stamps.

### 5.6 Model compatibility checker

- Input: feature parquet + list of model bundle directories
- For each `feature_spec.json`: verify column presence, basic dtypes, optional NaN fraction warnings
- Mark obsolete detectors (`contact`, empty focal specs) as unsupported
- **No** model load, **no** inference, **no** training

### 5.7 Test strategy

| Layer | Tests |
| --- | --- |
| Upstream selection | Eligible vs stale S4/S5; sync mismatch rejects pair |
| Grid align | Identical `time_sec` from shared sync; deliberate mismatch fails |
| Pose adapter | Pivot correctness; provenance→flag mapping unit table; headstage NaN preserved |
| Blob join | many-to-one on `dense_frame_idx`; row count invariant |
| Smoke on lab session | Produce artifact; compat report for `contact_intact` / `contact_occluded` |
| Explicit non-goals | No numerical regression vs legacy caches in S6-A/MVP unless later approved |

### 5.8 GUI boundary

Reuse Group 2 — Feature Processing (`MainWindow`):

- Enable S6 button when session has eligible aligned S4+S5 (or always visible with disabled reason, matching S5 pattern)
- Page: select/confirm S4 run, S5 run, show alignment status, run generation, show QC + model_compat summary, reopen prior S6 runs
- Share `MainWindow._session_root`
- Follow S5 navigation safety: long tasks should not hard-block Group 1 unless required; mirror established patterns

S6 does **not** implement behavior inference UI.

---

## 6. Task 6 — Documentation index

This file is the S6-A authority. Index entry added in `docs/README.md`.

---

## 7. Risks

1. **`legacy/` excluded from git** — port/vendor required before implementable CI.
2. **S4 sync staleness weaker than S5** — joint alignment mandatory.
3. **Quality-flag semantics** — Kalman-only `*_any_imputed_*` vs broader “any gap fill” changes model inputs.
4. **No `was_refined` in S4** — obsolete `contact` detector cannot be supported honestly.
5. **Node naming drift** (`tail` vs `tail_base`) — must exercise adapter aliases against real lab skeletons.
6. **FF size** (~3.6k LOC) — port risk; keep I/O adapter thin.
7. **Focal empty specs** — do not present as active.
8. **`is_implanted` ID==0** — lab-specific; confirm for suite sessions.

---

## 8. Decisions requiring review

1. **Package name:** `combined_features` vs `dyadic_features`.
2. **Vendor plan** for `frame_features.py` + Pipeline 50 post functions (copy into `src/` vs track selected `legacy/` paths).
3. **Provenance→`nose_any_imputed_*` policy** (recommended: Kalman-only, matching legacy; confirm).
4. **Treatment of `temporal_resample` and `linear_gap_fill` for quality flags.**
5. **Coordinates:** post-smooth only vs optional pre-smooth mode.
6. **`orig_frame_idx` source:** S4 mapping vs S5 nearest-neighbor column.
7. **`is_implanted` rule** retention.
8. **Default model set** for compat UI: active dyadic five only, or include focal.
9. **Whether to add `prepared_sync_sha256` to future S4 run_meta** (hardening; optional S4 change — not required if S6 computes digest at pair validation).
10. **Add `find_latest_eligible_s4_run`** to `pose_finalization.run_discovery` (small non-scientific API).

---

## 9. Implementation roadmap (post-approval)

| Phase | Deliverable | Status (2026-10-25) |
| --- | --- | --- |
| S6-0 | Approve §8 decisions; vendor FF/P50 modules into tracked tree | Done (Sprints 1–2) |
| S6-1 | Contracts, upstream pair validation, pose adapter, unit tests | Done (Sprint 1) |
| S6-2 | Headless runner: FF + blob join + P50 post + export/QC | Done (Sprints 2–3) |
| S6-3 | `model_compat` checker against active dyadic specs | Done (Sprint 3) |
| S6-4 | GUI page under Feature Processing + discovery | Done (Sprint 4) |
| S6-5 | Lab acceptance on real S4+S5 session (artifact + compat report) | Done (operator GUI, 2026-10-25) |

**Out of scope through S6 MVP:** model inference, training, Pipeline 30 restoration, new feature definitions, mandatory numerical regression vs legacy caches.

---

## Appendix A — Key file index

| Area | Paths |
| --- | --- |
| S4 contracts / export / discovery / eligibility | `src/pose_finalization/{contracts,export,run_discovery,runner,timing,accepted_s3}.py` |
| S5 contracts / export / discovery / eligibility | `src/blob_features/{contracts,export,run_discovery,runner,timing,s1_eligibility}.py` |
| GUI Feature Processing | `src/ui/main_window.py`, `src/ui/pages/blob_features_page.py`, `src/ui/pages/combined_features_page.py`, `src/ui/controllers/combined_features_controller.py` |
| Legacy P50 | `legacy/pipelines/50_dyadic_features.py` |
| Legacy FF | `legacy/src/dynamic_train/frame_features.py` |
| Legacy paths | `legacy/src/io/path_utils.py` |
| Dyadic models | `legacy/models/dyadic/v1/*/feature_spec.json` |
| Focal models | `legacy/models/focal/v1/*/feature_spec.json` |
| S5 handoff notes | `docs/subsystem_05/mvp_scope_and_roadmap.md` (§ Future S6 handoff boundary) |

## Appendix B — S5 / S4 closure commits (context)

| Subsystem | Commit | Message |
| --- | --- | --- |
| S5 | `b0d54f8` | Add Subsystem 05 blob feature extraction MVP and mark it lab-validated. |
| S4 | `f90ac3d` | Mark Subsystem 04 MVP complete after lab GUI validation |
