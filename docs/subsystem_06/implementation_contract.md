# Subsystem 06 — Combined Pose & Video Feature Generation

## Implementation Contract (S6-B)

**Status:** Binding design contract. S6 MVP **complete / lab-validated** (Sprints 1-4 + operator laboratory GUI acceptance). See [`sprint_1_input_adapter.md`](sprint_1_input_adapter.md), [`sprint_2_feature_migration.md`](sprint_2_feature_migration.md), [`sprint_3_export_compatibility.md`](sprint_3_export_compatibility.md), [`sprint_4_gui_integration.md`](sprint_4_gui_integration.md), [`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).  
**Date:** 2026-10-25  
**Authority:** Accepted S6-A audit (`architecture_audit.md`) + implemented S6 source + operator-confirmed laboratory acceptance on `E_full_session_integration`.  
**Scope:** Full S6 design. Sprint 4 delivers §8 GUI integration on top of Sprint 3 export/QC/compat/discovery/runner.

S6 generates a **complete directional feature artifact** and a **separate model-compatibility report**. S6 does **not** train models or run inference.

---

## 0. Binding decisions from S6-A (accepted)

| Decision | Contract stance |
| --- | --- |
| No Pipeline 30 restoration | S4 `final_pose.parquet` is the only pose source |
| No new feature formulas | Port legacy FF + Pipeline 50 math unchanged |
| Contact design | Active: `contact_intact` + `contact_occluded`; obsolete: `contact` |
| Drop for supported set | `nose_any_refined`, unsuffixed `nose_any_imputed`, PCHIP-as-truth |
| Retain | `nose_any_imputed_i`, `nose_any_imputed_j` (Kalman-only under S4) |
| Coordinates | Final **post-smoothing** `x`/`y` |
| Package | Prefer `src/combined_features/` (name confirmed in §8 if contested) |
| `legacy/` git-excluded | Must vendor required code into tracked `src/` before CI-ready implementation |

---

## 1. S4-to-feature adapter contract

### 1.1 Purpose

Convert long-form S4 `final_pose.parquet` (+ `run_meta.json`) into the **pose tensor / quality arrays** historically loaded from Pipeline 30 `dense_pose_120_imputed.npz` + `impute_meta.json`, so that `build_frame_features` (or its vendored equivalent) can run without formula changes.

Adapter module (planned): `src/combined_features/pose_adapter.py`.

### 1.2 Input artifacts

| Artifact | Required fields |
| --- | --- |
| `final_pose.parquet` | All of `pose_finalization.contracts.FINAL_POSE_COLUMNS` |
| `run_meta.json` | `source_s3`, `dense_summary` (when present), `status` |

### 1.3 Track and node normalization

| Step | Rule |
| --- | --- |
| Tracks | Distinct integer `track` values from parquet; sort ascending for stable tensor axis order; record order in adapter metadata |
| Track ID coerce | Match FF `_normalize_track_id`: bytes→str→int when possible |
| Nodes | Distinct `node` strings; preserve skeleton order from first occurrence per track or from `dense_summary.node_names` when available |
| Required scientific nodes | At least resolvable `nose`, `neck`, and `tail`/`tail_base`, plus `spine`/`spine_base` for heading/body-length defaults |
| Aliases (FF-compatible) | `spine` ↔ `spine_base`; `tail` ↔ `tail_base`. Adapter must populate both names in the node index consumed by FF |
| Headstage | If node `headstage` present, include it; S4 never gap-fills headstage (`HEADSTAGE_NODE_NAME`). Missing headstage → omit node (distances involving headstage become NaN via FF missingness) |
| Implanted identity | Not encoded in S4 nodes. Downstream `is_implanted` remains Pipeline 50 rule: numeric focal/track id `== 0` (see §8) |

**Failure:** missing required nodes after alias resolution → hard error (do not invent nodes).

### 1.4 120 Hz indexing and timestamps

| Legacy NPZ / FF field | S4 source | Adapter rule |
| --- | --- | --- |
| Time axis `T` | Unique `sample_idx` | `T = n_unique(sample_idx)`; must equal `max(sample_idx)+1` with contiguous `0..T-1` |
| `t_120[t]` | `time_sec` at that `sample_idx` | One value per sample; reject conflicting `time_sec` for same `sample_idx` |
| Dense index in feature rows | — | `dense_frame_idx := sample_idx` |
| `orig_frame_idx[t]` | **Contract:** copy S5 `orig_frame_idx` aligned on `sample_idx` after joint validation | Ensures blob join and FF share the same prepared-frame index. Do **not** invent from S4 left/right interpolation unless S5 is unavailable (that path is a hard failure under joint eligibility) |

### 1.5 Coordinate selection

| Field | Source | Rule |
| --- | --- | --- |
| `x[track, node, t]`, `y[...]` | S4 `x`, `y` | **Post-smoothing** coordinates only |
| Pre-smooth | `x_pre_smooth`, `y_pre_smooth` | Retained only in optional QC/debug side channel — **never** fed into FF formulas |
| Shape | — | `(n_tracks, n_nodes, T)` float64 |
| Missing | NaN in S4 | Preserve NaN; do not zero-fill |

Pivot: one long-form row → one tensor cell keyed by `(track, node, sample_idx)`. Duplicate keys → hard error.

### 1.6 Point score handling

| Field | Rule |
| --- | --- |
| `point_score[track, node, t]` | Pass through S4 `point_score` |
| Fabrication | **Forbidden.** Reconstructed / gap-filled samples often have NaN scores in S4; leave NaN |
| `instance_score` | Not required by FF; may be ignored or stored in adapter metadata |
| Downstream `nose_*_sigma` | Computed inside FF: `sigma = 1/score - 1` iff `score > 0` and finite; else NaN |

### 1.7 Missing coordinates and protected blanks

| S4 condition | Adapter / FF behavior |
| --- | --- |
| `x`/`y` NaN | Missing observation; FF distances/kinematics NaN-propagate |
| `protected_manual_blank == True` | Coordinates remain NaN; provenance should be `protected_manual_blank` |
| Do not | Treat blanks as imputed, refined, or Kalman |

Preserve a parallel `protected_manual_blank[track, node, t]` bool array in adapter metadata for QC (not a legacy FF NPZ key).

### 1.8 S4 reconstruction provenance → legacy quality indicators

S4 codes (`pose_finalization.contracts`):

| Code | Name |
| --- | --- |
| 0 | `missing` |
| 1 | `original` |
| 2 | `identity_corrected` |
| 3 | `temporal_resample` |
| 4 | `linear_gap_fill` |
| 5 | `kalman_gap_fill` |
| 6 | `protected_manual_blank` |

#### Authoritative quality mapping (binding)

| Legacy FF output | Adapter derivation from S4 `provenance` | Notes |
| --- | --- | --- |
| `*_is_kf` (nose/neck/spine/tail × i/j later) | `1` iff provenance == **`PROVENANCE_KALMAN_GAP_FILL` (5)** at that track/node/sample | Verified Kalman reconstruction only |
| `*_is_pchip` | **Always `0`** | PCHIP does not exist in S4; must not be falsely represented |
| `impute_stage` (if still constructed for FF internals) | `3` if Kalman; `1` if linear_gap_fill; `0` otherwise; **never `2` (PCHIP)** | Prefer emitting flag arrays directly if FF is refactored at I/O boundary only |
| `is_imputed` (raw bool, if needed) | `True` for provenance ∈ {`linear_gap_fill`, `kalman_gap_fill`} only | Linear is “imputed” for diagnostics but **not** for `*_any_imputed_*` |
| `was_refined` | **Always `False` / `0`** | S4 geometric refinement is deferred/disabled; smoothing ≠ refinement; identity correction ≠ refinement |
| `nose_any_refined` | Always `False` | Obsolete for active models; still emit if FF computes it from zeros |
| `nose_any_imputed_i/j` | `nose_*_is_pchip OR nose_*_is_kf` → **Kalman-only under S4** | Preserves legacy PCHIP-or-Kalman definition |
| `neck/spine/tail_any_imputed_{i,j}` | Same rule on that node | Required for head-sweep weights and `nose_to_tail` (`tail_any_imputed_j`) |

#### Explicit non-equivalences

| Do **not** map to imputed / kf / refined |
| --- |
| `temporal_resample` (3) |
| `linear_gap_fill` (4) → not `*_any_imputed_*`, not `*_is_kf` |
| `identity_corrected` (2) |
| `was_smoothed == True` |
| Manual blank / missing |

#### Preserve S4 provenance separately

Adapter **must** retain `provenance[track, node, t]` (uint8) and optionally `provenance_name` in metadata / optional side parquet columns for QC. Feature formulas continue to consume only the legacy-compatible flags above.

### 1.9 Field-level mapping table (summary)

| Legacy expectation | S4 field(s) | Mapping |
| --- | --- | --- |
| `x`, `y` | `x`, `y` | Post-smooth pivot |
| `t_120` | `time_sec` | Per `sample_idx` |
| `orig_frame_idx` | S5 `orig_frame_idx` | After alignment |
| `point_score` | `point_score` | Pass-through; no fabrication |
| `impute_stage` / kf / pchip | `provenance` | §1.8 |
| `is_imputed` | `provenance` | linear∪kalman only |
| `was_refined` | — | Forced false |
| `tracks`, `nodes` | `track`, `node` (+ aliases) | §1.3 |
| Pre-smooth / `was_smoothed` / left-right / weight | corresponding columns | Metadata/QC only |

### 1.10 Non-contact model impact of dropping PCHIP / refinement

Inspected all local `feature_spec.json` under `models/{dyadic,focal}/v1`.

| Detector class | Requires `*_is_pchip`? | Requires `nose_any_refined` / `*_was_refined`? | Requires unsuffixed `nose_any_imputed`? |
| --- | --- | --- | --- |
| Active dyadic (`contact_intact`, `contact_occluded`, `nose_to_nose`, `side_by_side`, `mutually_disengaged`) | **No** | **No** | **No** |
| Active focal (all non-empty / two-stage specs) | **No** | **No** | **No** |
| Obsolete `contact` | No | **Yes** (`nose_any_refined`) | **Yes** (`nose_any_imputed`) |

**Incompatibility to document, not paper over:**

| Bundle | Status | Exact missing semantics under S6 adapter |
| --- | --- | --- |
| `dyadic/v1/contact` | **Unsupported / obsolete** | Needs `nose_any_refined` and unsuffixed `nose_any_imputed`. Producer historically never wrote unsuffixed `nose_any_imputed` either. Compat checker must mark **incompatible**; do not synthesize columns to “pass” the check |

Active models that **do** require Kalman-aware / imputed-i/j fields (must be produced correctly):

| Feature | Consumers |
| --- | --- |
| `nose_any_imputed_i` / `nose_any_imputed_j` | `contact_intact`, `nose_to_nose`; focal `approach`, `retract`; `nose_any_imputed_i` also `nose_to_tail` |
| `nose_i_is_kf` / `nose_j_is_kf` | `contact_intact` |
| `tail_any_imputed_j` | `nose_to_tail` |
| `*_echo_imputed` | `nose_to_tail` (Pipeline 50 last-known-good) |
| `nose_i_sigma` / `nose_j_sigma` | `contact_intact`, focal `nose_body` / `nose_tail` open features, etc. |

---

## 2. Joint S4 / S5 eligibility and alignment

### 2.1 Evidence available for shared S1 (no false claims)

**S4 does not store `prepared_sync_sha256`.** Sync identity on S4 is path-only under `run_meta.source_s3`.

| Side | Field | Use |
| --- | --- | --- |
| S4 | `source_s3.session_root` | Session identity |
| S4 | `source_s3.prepared_sync` | Recorded sync **path** |
| S4 | `source_s3.tracked_pose_sha256` | S3 pose staleness only (not S1) |
| S5 | `source_s1.session_root` | Session identity |
| S5 | `source_s1.prepared_sync` | Recorded sync path |
| S5 | `source_s1.prepared_sync_sha256` | Stored digest |
| Live | `sha256(Path(S4.source_s3.prepared_sync))` | Compare to S5 digest at validation time |

**S4 change not required.** Optional future hardening (add sync SHA to S4 `run_meta`) is deferred unless review demands it (§8).

### 2.2 Selection procedure (default)

```text
1. session_root must exist
2. s5 = find_latest_eligible_s5_run(session_root)
      # classifications: eligible | completed_with_warnings; newest-first
3. s4 = find_latest_eligible_s4_run(session_root)   # NEW thin helper
      # == summarize_pose_finalization_project(...).eligible_runs[0]
4. validate_s4_run_for_downstream(s4.run_dir)
5. validate_s5_run_for_downstream(s5.run_dir)
6. joint_align(s4, s5)  # §2.3
7. on failure of default pair: do not silently pick another pair in headless mode;
   GUI may search other eligible×eligible pairs that pass joint_align
```

Never auto-select `stale_source`, `failed_or_incomplete`, or `missing_required_artifacts`.

### 2.3 `joint_align` validation (exact)

| Check | Pass criterion | Fail |
| --- | --- | --- |
| Session | `Path(s4.source_s3.session_root).resolve() == Path(s5.source_s1.session_root).resolve() == session_root.resolve()` | `session_mismatch` |
| Sync path | Resolve-equal `prepared_sync` paths | `sync_path_mismatch` |
| Sync file exists | Path is a file | `sync_missing` |
| Sync digest | `sha256(s4_sync_path).lower() == s5.source_s1.prepared_sync_sha256.lower()` | `sync_digest_mismatch` |
| Target FPS | Both imply 120 Hz (`TARGET_FPS_DEFAULT`); reject recorded overrides ≠ 120 | `fps_mismatch` |
| Sample count | `n_unique(s4.sample_idx) == len(s5)` and equals expected timeline length from sync | `sample_count_mismatch` |
| Indices | S5 `sample_idx` == `dense_frame_idx` == `frame_idx` == `0..N-1`; S4 unique samples same | `index_integrity` |
| Timestamps | `np.allclose(time_s4, time_s5, rtol=0.0, atol=1e-12)` | `time_grid_mismatch` |

**Timestamp tolerance:** `atol=1e-12`, `rtol=0`, matching S5 QC (`blob_features.qc._check_index_integrity`).  
(Timing construction uses `1e-9` for source-mapping coincidence; that is **not** the joint published-grid tolerance.)

**Not required to match:** S4 left/right source frames vs S5 `orig_frame_idx` (interp vs nearest-neighbor). After alignment, FF/`orig_frame_idx` uses **S5** values (§1.4).

### 2.4 Staleness / supersession

| Event | S4 discovery | S5 discovery | S6 |
| --- | --- | --- | --- |
| New accepted S3 | Prior S4 → `stale_source` | Unaffected | Cannot use stale S4 |
| S1 re-prepare (sync SHA changes) | S4 may still look eligible (no sync SHA) | S5 → `stale_source` | Joint digest check fails even if S4 still “eligible” |
| Historical status strings | Never rewritten | Never rewritten | Same for S6 completions |

### 2.5 Multiple eligible runs

- Default: newest eligible each side + joint gate.
- If newest pair fails alignment: headless → **fail with explicit error** listing why; GUI → show matrix of eligible S4×S5 with pass/fail reasons; user may select any passing pair.
- Prefer digest-aligned pairs over independently newest when they diverge.

### 2.6 Failure when alignment cannot be established

Hard-fail the S6 run (`status=failed`); write `run_meta` + `processing_log` with structured reason code from §2.3; do **not** write `dyadic_features_full.parquet` as completed.

---

## 3. Feature calculation inventory

### 3.1 Pipeline order (binding)

Exact order from `legacy/pipelines/50_dyadic_features.py` → `run_dyadic_feature_producer`:

| Step | Function | Role |
| --- | --- | --- |
| 1 | `build_frame_features` (vendored) | Pose → directional features |
| 2 | `_attach_blob_features` | Left-join S5 blob columns on `dense_frame_idx` |
| 3 | `_add_geometric_echo_features` | Lagged distances + `rolling_sum_cc_count_60` |
| 4 | `add_head_sweep_features` | Nose/neck projected velocities, intent, radials |
| 5 | `add_total_occlusion_flag` | Lost-track / total occlusion flags |
| 6 | `add_last_known_good_features` | Echo / echo_imputed carry-forward (limit 60) |
| 7 | `add_is_implanted_feature` | `is_implanted` |

Do not reorder. Do not alter formulas.

### 3.2 Modules to migrate into tracked production code

| Legacy path | Production destination (proposed) | Notes |
| --- | --- | --- |
| `legacy/src/dynamic_train/frame_features.py` | `src/combined_features/frame_features.py` (or package sub-module) | Replace NPZ I/O with adapter-fed arrays; **no formula edits** |
| Pipeline 50 helpers in `50_dyadic_features.py` | `src/combined_features/pipeline50_post.py` | Blob join adapted to S5 run paths |
| — | `src/combined_features/pose_adapter.py` | New |
| — | `src/combined_features/upstream.py` | New joint eligibility |
| — | `src/combined_features/model_compat.py` | New checker |
| — | `src/combined_features/{contracts,export,qc,runner,run_discovery,config}.py` | New, mirror S4/S5 patterns |
| `legacy/src/io/path_utils.py` | **Do not** bring flat `data/...` layout as primary I/O | Session-centric suite paths only |

**Not migrated:** `30_impute.py`, PCHIP codepaths as producers, training/inference pipelines 60–90.

### 3.3 Dependency matrix (by producer stage)

Column legend: **Kind** = `pose` / `blob` / `quality` / `temporal` / `meta`. **Dir** = directional (`i`/`j`) vs `sym` (same on both directions of a dyad sample) vs `undir` (blob).

#### A. Identity / index (adapter + FF)

| Feature / field | Producer | Raw inputs | Kind | Dir | Consumers (representative) | Missingness | Migration risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `focal_id`, `partner_id` | FF pair expand | S4 tracks | meta | dir | all | n/a | Low |
| `dense_frame_idx`, `sample_idx`, `t_120`/`time_sec` | adapter/FF | S4 sample grid | meta | undir | join, all | fail if broken | Low if §2 holds |
| `orig_frame_idx` | S5 via align | S5 parquet | meta | undir | FF | n/a | Must come from S5 |
| `dyad_body_scale` | FF | body refs | pose | dir | many norms | NaN if scale invalid | Node alias risk |
| `exp_name` / session id | S6 runner | session_root | meta | undir | grouping | — | Map session→name policy §8 |

#### B. Coordinates (FF from adapter tensors)

| Feature | Producer | Inputs | Kind | Dir | Consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nose/neck/spine/tail/headstage_{x,y}_{i,j}` (+ base aliases) | FF | post-smooth x/y | pose | dir | geometry/kinematics | NaN propagate | Node names |

#### C. FF pose geometry / kinematics / headings / rollings

| Group | Producer | Inputs | Kind | Dir | Primary consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pairwise `distance_*_norm` | FF | coords + scale | pose | dir | contact_*, nose_*, side_by_side, mutually_disengaged, focals | NaN | Low |
| Contact summaries (`*_body_*`, asymmetry, gaps) | FF | distances | pose | dir | contact_intact, focals | NaN | Low |
| Headings / bearings / side-by-side offsets | FF | coords | pose | dir | most dyadic + focal | NaN (some `nan_to_num` exceptions in FF) | Preserve exceptions |
| Speeds / projected_vel / vel_radial_* / accel | FF | coords + t | pose+temporal | dir | contact_intact, mutually_disengaged, focals | NaN | Low |
| Rolling windows 15/30/60/360 | FF | base series | temporal | dir | many | edge min_periods | Window params frozen |
| Morphology (`compactness_*`, lengths) | FF | coords | pose | dir | side_by_side, focals | NaN | Low |

Authoritative exhaustive FF column set: producer assignments in vendored `frame_features.py` (hundreds of columns). S6 export retains the full FF output plus P50 additions (complete artifact), not a model-subset projection.

#### D. Quality flags (FF ← adapter)

| Feature | Producer | Inputs | Kind | Dir | Consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nose_i/j_is_kf` | FF←adapter | provenance==kalman | quality | dir | contact_intact | 0/1 | **Must be Kalman-only** |
| `nose_i/j_is_pchip` | FF←adapter | always 0 | quality | dir | none active | 0 | Do not invent PCHIP |
| `nose_any_imputed_i/j` | FF | pchip∨kf → kf-only | quality | dir | contact_intact, nose_to_nose, approach, retract, nose_to_tail | 0/1 | Definition binding |
| `neck/spine/tail_any_imputed_*` | FF | same | quality | dir | head-sweep weights; `tail_any_imputed_j`→nose_to_tail | 0/1 | Same |
| `nose_i/j_sigma` | FF | point_score | quality | dir | contact_intact, nose_body, nose_tail, … | NaN if no score | No fabricated scores |
| `nose_any_refined`, `*_was_refined` | FF | forced false | quality | dir | **obsolete contact only** | 0 | Unsupported bundle |
| `frac_primitive_distance_missing` et al. | FF | distance finiteness | quality | dir | contact_intact | derived | Low |

#### E. Blob join (Pipeline 50 step 2 / S5)

| Feature | Producer | Inputs | Kind | Dir | Consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cc_count`, `cc_count_raw`, `min_inter_contour_dist` | S5→join | blob parquet | blob | undir | most detectors; routing for two-stage focals | NaN if join miss | Join key integrity |
| Occlusion static + temporal (`ellipse_*`, `circularity*`, `solidity*`, `num_dt_peaks*`, `furthest_*`) | S5→join | blob parquet | blob | undir | contact_intact, contact_occluded, … | NaN | Column name parity with S5 |

Join contract: `left_on=dense_frame_idx`, prefer right key `dense_frame_idx`, `how=left`, `validate=many_to_one`, de-dupe blob key; row count must not change.

#### F. Pipeline 50 geometric shifts (step 3)

| Feature | Producer | Inputs | Kind | Dir | Consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `distance_*_shift_{10,30}` (listed in P50) | P50 | FF distances | temporal | dir | contact_intact, contact (obs), nose_body | NaN at series start | Low |
| `rolling_sum_cc_count_60` | P50 | `cc_count` | temporal+blob | undir | obsolete contact; available generally | 0-filled per P50 | Low |

#### G. Head-sweep / intent (step 4)

| Feature group | Producer | Inputs | Kind | Dir | Consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `projected_vel_nose/neck_*`, combined, j_to_i mirrors | P50 | coords + `*_any_imputed_*` weights | pose+quality | dir | mutually_disengaged; focals (intent) | NaN + gap-fill max 30 on combined | Weight policy depends on quality flags |
| `signed_kinematic_intent*` / `rolling_mean_intent_15` | P50 | projected vel + energy | temporal | dir | mutually_disengaged, approach, … | NaN | Preserve hysteresis params |
| `vel_radial_*` diffs of norm distances | P50 | FF distances | temporal | dir | mutually_disengaged | NaN | Low |

#### H. Occlusion + echoes + implanted (steps 5–7)

| Feature | Producer | Inputs | Kind | Dir | Consumers | Missing | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `is_focal_lost`, `is_partner_lost`, `is_total_occlusion` | P50 | coord NaN patterns | quality | dir | echo triggers | 0/1 | Heuristic |
| Occlusion echoes (`*_echo`) | P50 | FF + occlusion | temporal | dir | contact_occluded, contact_intact | ffill limit 60 | Trigger correctness |
| `*_echo_imputed` | P50 | FF + `nose_any_imputed_i` / `tail_any_imputed_j` | temporal+quality | dir | nose_to_tail | ffill limit 60 | Needs correct imputed flags |
| `is_implanted` | P50 | focal/track id==0 | meta | dir | optional / lab | 0 if unknown | **Lab convention** |

### 3.4 Model × quality/blob sensitivity (active set)

| Detector | `nose_any_imputed_i/j` | `*_is_kf` | Blob-heavy | P50 intent/echo |
| --- | --- | --- | --- | --- |
| contact_intact | Y | Y | Y | shifts + some echoes |
| contact_occluded | N | N | Y | occlusion echoes |
| nose_to_nose | Y | N | light | N |
| side_by_side | N | N | light | N |
| mutually_disengaged | N | N | light | **intent / projected nose-neck** |
| approach / retract | i only | N | mix | intent |
| nose_to_tail | i + `tail_any_imputed_j` | N | mix | **echo_imputed** |
| nose_body / nose_tail (two-stage) | N | N | routing on `cc_count` | shifts; uses sigma |

---

## 4. Model bundle contract

### 4.1 Packaging reality (binding honesty)

| Location | Status |
| --- | --- |
| `models/**` detector bundles | **Tracked** in the suite package (SLEAP `*.mp4` / `*.slp` remain gitignored) |
| Tracked suite package today | **No** production-packaged dyadic/focal model directory |

S6 MVP **must not** claim models are shipped with the suite unless/until a tracked registry path is added. Compat checking reads an **explicit configured directory** that may point at a local lab path.

### 4.2 Configured model registry

Proposed settings key (YAML / GUI):

```yaml
model_registry:
  root: null   # required path; e.g. lab checkout of models or future package data
  dyadic_subdir: "dyadic/v1"
  focal_subdir: "focal/v1"
  include_obsolete: false
```

If `root` is missing/unreadable → compatibility report status `registry_unavailable` (features may still generate).

### 4.3 Discovery rules

| Kind | Discovery |
| --- | --- |
| Dyadic | Each immediate child of `{root}/dyadic/v1/` containing `feature_spec.json` |
| Focal | Each immediate child of `{root}/focal/v1/` containing `feature_spec.json` |
| Spec load | Prefer `selected_feature_cols` if non-empty list; else `feature_cols` list; else union of `open_features` + `occ_features` (+ any `blob_features` list) for two-stage experts |

### 4.4 Active vs obsolete

| Class | Detectors | Compat default |
| --- | --- | --- |
| **Active dyadic** | `contact_intact`, `contact_occluded`, `nose_to_nose`, `side_by_side`, `mutually_disengaged` | Check when registry present |
| **Obsolete dyadic** | `contact` | Always `unsupported` with reasons (`nose_any_refined`, `nose_any_imputed`) |
| **Active focal** | `approach`, `retract`, `nose_body_intact`, `nose_body_occ`, `nose_to_body`, `nose_to_tail`, `nose_body`, `nose_tail` (two-stage) | Optional check group |
| **Malformed** | Spec with zero resolvable feature names | `unsupported` / `invalid_spec` |

### 4.5 Validation performed by `model_compat` (no inference)

For each selected detector:

1. Load `feature_spec.json`
2. Resolve ordered feature name list (§4.3)
3. Assert every name ∈ feature parquet columns
4. Soft-check dtypes (numeric for continuous; 0/1-compatible for flags)
5. Optional: NaN fraction warnings per column
6. Record `model.joblib` / booster file **presence only** (exists / missing) — do **not** load
7. Emit structured result

### 4.6 Compatibility report structure (`model_compat.json`)

```json
{
  "schema_version": "combined_features_model_compat_v1",
  "registry_root": "<path or null>",
  "registry_status": "ok | unavailable | partial",
  "feature_artifact": "dyadic_features_full.parquet",
  "n_feature_columns": 0,
  "n_rows": 0,
  "detectors": [
    {
      "detector_name": "contact_intact",
      "family": "dyadic",
      "bundle_dir": "<path>",
      "status": "compatible | incompatible | unsupported | invalid_spec | bundle_incomplete",
      "n_required": 139,
      "missing_columns": [],
      "dtype_warnings": [],
      "nan_fraction_warnings": [],
      "model_file_present": true,
      "notes": []
    }
  ]
}
```

Obsolete `contact` → `unsupported` with explicit note listing required obsolete columns.

---

## 5. S6 output and provenance

### 5.1 Run layout (session-centric)

```text
{session_root}/combined_features/{run_id}/
  dyadic_features_full.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
  qc.json
  model_compat.json
```

| Constant (proposed) | Value |
| --- | --- |
| Dirname | `combined_features` |
| Primary parquet | `dyadic_features_full.parquet` |
| Schema version | `dyadic_features_full_v1` |
| Run meta schema | `combined_features_run_meta_v1` |
| Statuses | mirror S5: `not_started`, `running`, `completed`, `completed_with_warnings`, `failed` |

`run_id` pattern: `combined_features__YYYYMMDDTHHMMSS` (consistent with S5).

### 5.2 `dyadic_features_full.parquet` schema

**Not** a small fixed column list — it is the **complete** directional feature table:

1. Identity / timing columns (normalized names below)
2. Full FF feature set
3. Blob columns from §3.3.E
4. All Pipeline 50 post columns from steps 3–7

#### Required identity / timing columns (normalized)

| Column | Dtype | Rule |
| --- | --- | --- |
| `schema_version` | str | `dyadic_features_full_v1` |
| `session_root` | str | resolved path |
| `focal_id` | int64 | track id |
| `partner_id` | int64 | track id ≠ focal |
| `sample_idx` | int64 | 0..N-1 |
| `dense_frame_idx` | int64 | equals `sample_idx` |
| `time_sec` | float64 | aligned grid |
| `orig_frame_idx` | int64 | from S5 |
| `source_s4_run_id` | str | provenance denorm (optional but recommended) |
| `source_s5_run_id` | str | optional denorm |

Legacy `exp_name` / `iteration_name` may be filled for compatibility (`exp_name` ← session folder name or configured label; `iteration_name` ← `"suite"` or settings) — finalize in §8.

#### Unique directional row key

`(focal_id, partner_id, dense_frame_idx)` unique; enforce in QC.

#### Expected rows per 120 Hz sample

For `K` tracks: **`K * (K - 1)`** directional rows per sample (all ordered pairs).  
Typical dyad lab (`K=2`): **2 rows per sample**.

### 5.3 Blob merge contract

- Source: selected eligible S5 `blob_features.parquet`
- Keys: `dense_frame_idx` (required on both after adapter/FF)
- Columns: exact `BLOB_FEATURE_COLS` list from Pipeline 50 / S5 contracts
- Semantics: left join; many-to-one; no row inflation; missing blob → NaN columns + QC warning

### 5.4 `run_meta.json` provenance (minimum)

| Block | Contents |
| --- | --- |
| Identity | `run_id`, `created_at`, `completed_at`, `status`, `session_root` |
| `source_s4` | `run_dir`, `run_id`, `final_pose` path, **SHA256 of parquet**, `prepared_sync` path, **live SHA256 of sync at validation**, `tracked_pose_sha256` copied from S4 meta |
| `source_s5` | `run_dir`, `run_id`, `blob_features` path + SHA256, full `prepared_sync` path + SHA256, video/bg digests copied from S5 `source_s1` when present |
| `alignment` | sample_count, target_fps, time_atol, checks passed |
| `adapter` | policy version, kalman/pchip/refined rules stamp, node/track lists |
| `outputs` | paths to parquet, qc, model_compat, settings, log |
| `qc` | embedded summary (also standalone `qc.json`) |

### 5.5 `settings_used.yaml`

Immutable snapshot: upstream run ids, adapter policy, FF defaults (`smooth_win=9`, `diff_dt=1`, body/heading node tuples), Pipeline 50 constants, model registry root used, feature schema version.

### 5.6 QC statistics (`qc.json`)

| Check | Content |
| --- | --- |
| Alignment | echo of joint_align results |
| Row counts | `N_samples`, `N_tracks`, `N_rows`, expected `N_samples * K * (K-1)` |
| Key uniqueness | directional key |
| Index aliases | `sample_idx` vs `dense_frame_idx` |
| Blob coverage | fraction of rows with non-null `cc_count` |
| Quality rates | fraction Kalman / any_imputed / missing nose per track |
| Warnings | non-fatal issues → may yield `completed_with_warnings` |

### 5.7 Completed-run eligibility for downstream (future S7+)

Mirror S5 pattern:

| Classification | Rule |
| --- | --- |
| `eligible` | `completed`/`completed_with_warnings` + parquet present + recorded S4/S5 digests still match live eligible upstream + sync SHA still matches |
| `stale_source` | Upstream superseded or digest drift |
| `failed_or_incomplete` / `missing_required_artifacts` | as named |

API: `summarize_combined_features_project`, `is_s6_run_eligible_for_downstream`, `find_latest_eligible_s6_run`.

---

## 6. Implementation roadmap

### Sprint 1 — Input contracts and S4 adapter

**Depends on:** S6-B approval.  
**Deliverables:**

- `combined_features.contracts` + `upstream` joint eligibility (no S4 schema change)
- `find_latest_eligible_s4_run` helper (pose_finalization discovery)
- `pose_adapter` with unit tests for pivot, provenance→flags, aliasing, NaN/blank behavior
- Headless align failure modes

**Acceptance:** Given fixtures or lab snippets, adapter produces tensors/flags matching §1.8; joint_align enforces path+live SHA+`atol=1e-12`.

### Sprint 2 — Legacy feature-calculation migration

**Depends on:** Sprint 1.  
**Deliverables:**

- Vendored `frame_features` behind adapter input (formulas unchanged)
- Pipeline 50 post modules ported (order preserved)
- Session-centric I/O (no dependence on excluded `legacy/` at runtime)

**Acceptance:** Directional frame table builds for a known 2-track session segment; column families from FF+P50 present; no PCHIP truth bits; Kalman flags only on provenance 5.

### Sprint 3 — Blob merge, final export, model compatibility

**Depends on:** Sprint 2.  
**Deliverables:**

- Full runner writing `dyadic_features_full.parquet`, `run_meta`, `settings_used`, `qc.json`
- Blob join against eligible S5
- `model_compat.json` against configured registry (graceful if registry absent)
- Discovery + downstream eligibility stubs

**Acceptance:** Complete artifact for lab session; compat reports `contact_intact` / `contact_occluded` when registry pointed at local models; obsolete `contact` marked unsupported; no inference.

### Sprint 4 — GUI integration and laboratory validation

**Depends on:** Sprint 3.  
**Status:** Implemented (automated). See [`sprint_4_gui_integration.md`](sprint_4_gui_integration.md).  
**Deliverables:**

- Enable S6 under Group 2 Feature Processing — **done** (`CombinedFeaturesPage` / controller; `MainWindow` soft-nav policy aligned with S5)
- Page: select/confirm S4+S5, show alignment, run, QC + compat summary, reopen runs — **done**
- Lab GUI acceptance checklist — **prepared**; fixture `E_full_session_integration` blocked (eligible S5 present, eligible S4 absent / S3 not accepted)

**Acceptance:** Lab operator produces S6 run from eligible S4+S5 without CLI; S1–S5 behavior unchanged.  
**Remaining for MVP close:** jointly aligned lab S4+S5 session + operator GUI evidence + architectural review.

### Explicit non-goals (all sprints)

Training, inference, Pipeline 30, new features, mandatory numerical regression vs legacy caches (unless later approved), claiming git-excluded models are packaged.

---

## 7. Unresolved decisions (require review sign-off)

| ID | Topic | Options / recommendation |
| --- | --- | --- |
| D1 | Package dirname / import | Recommend `combined_features` / `dyadic_features_full.parquet` as herein |
| D2 | `exp_name` compatibility field | Session folder name vs free-text label |
| D3 | `is_implanted` (id==0) | **Resolved:** headstage-only; resolved → `0.0`/`1.0`; unresolved → `NaN` (never encode uncertainty as `0`). |
| D4 | Default compat scope | Dyadic-active only vs dyadic+focal |
| D5 | Optional S4 `prepared_sync_sha256` | **Not required**; optional hardening later |
| D6 | Headless multi-pair search | Fail vs search all eligible pairs |
| D7 | Whether to emit always-zero `*_is_pchip` / `nose_any_refined` columns | Emit for FF parity vs omit and let compat mark obsolete-only gaps |
| D8 | Tracked model registry in-repo | Defer; configure external path for MVP |

---

## 8. GUI boundary (contract)

- Host: existing Group 2 — Feature Processing (`MainWindow`)
- S6 control enabled; page + controller: `ui.pages.combined_features_page`, `ui.controllers.combined_features_controller`
- Shared `session_root`; S5 soft-navigation safety patterns applied to S6 (S2/S4 hard lock unchanged)
- Optional model-registry path; compatibility display only — **no inference UI**
- Default S4/S5 pair only when joint alignment succeeds; manual selections are not silently replaced

---

## Appendix A — Active dyadic feature counts

| Detector | n (`selected_feature_cols`) |
| --- | --- |
| contact_intact | 139 |
| contact_occluded | 34 |
| nose_to_nose | 44 |
| side_by_side | 26 |
| mutually_disengaged | 79 |
| contact (obsolete) | 84 |

Exact name lists: the corresponding `feature_spec.json` files (local reference).

## Appendix B — Key source references

| Concern | Path |
| --- | --- |
| S4 columns / provenance | `src/pose_finalization/contracts.py`, `export.py` |
| S4 eligibility | `src/pose_finalization/runner.py` |
| S5 columns / QC atol | `src/blob_features/contracts.py`, `qc.py` |
| S5 eligibility / latest | `src/blob_features/runner.py`, `run_discovery.py` |
| Legacy FF | `legacy/src/dynamic_train/frame_features.py` |
| Legacy P50 | `legacy/pipelines/50_dyadic_features.py` |
| S6-A audit | `docs/subsystem_06/architecture_audit.md` |
