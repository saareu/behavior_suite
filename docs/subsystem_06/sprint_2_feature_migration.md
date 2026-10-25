# Subsystem 06 — Sprint 2: Legacy Feature Calculation Migration

**Status:** Implemented (awaiting architectural review).  
**Date:** 2026-10-25  
**Depends on:** Sprint 1 (`sprint_1_input_adapter.md`) provisionally accepted.  
**Out of scope:** GUI, model registry/compat, Parquet run publication, discovery, training/inference.

---

## Migrated modules / functions

| Production module | Legacy source | Role |
| --- | --- | --- |
| `src/combined_features/frame_features.py` | `legacy/src/dynamic_train/frame_features.py` | Base directional pose features |
| `src/combined_features/pipeline50_post.py` | `legacy/pipelines/50_dyadic_features.py` helpers | Blob join + post steps |
| `src/combined_features/processing.py` | (new) | Headless in-memory orchestration |

### `frame_features.py`

- `build_frame_features(adapted: AdaptedPose, ...)` — I/O boundary only changed.
- All kinematic / geometry / quality / rolling formulas preserved.
- Removed: NPZ/`impute_meta` loader, `path_utils`, CLI / debug writers.

### `pipeline50_post.py`

- `attach_blob_features` (was `_attach_blob_features`) — joins an in-memory S5 table
- `_add_geometric_echo_features`
- `add_head_sweep_features` (+ `_signed_kinematic_intent_series`, `_interpolate_short_gaps`)
- `add_total_occlusion_flag`
- `add_last_known_good_features`
- `add_is_implanted_feature` — now takes `ImplantationResolution`
- `apply_pipeline50_post` — ordered composition

### `processing.py`

- `compute_combined_features` / `compute_combined_features_from_pair` / `compute_combined_features_from_adapted`
- Returns `CombinedFeaturesComputation` (DataFrame + adapter + warnings + metadata)

---

## Feature calculation order

1. Sprint 1 joint validate (when using run dirs) → `AdaptedPose`
2. `build_frame_features(AdaptedPose)`
3. Add `sample_idx` / `time_sec` aliases (`dense_frame_idx` / `t_120`)
4. `attach_blob_features` on `dense_frame_idx`
5. Geometric shifts + `rolling_sum_cc_count_60`
6. Head-sweep / intent
7. Total-occlusion flags
8. Last-known-good / echo features
9. `is_implanted` from resolved implantation

---

## Adapter integration (Sprint 1 verification)

| Check | Result |
| --- | --- |
| Array shapes `(n_tracks, n_nodes, T)` | OK via `AdaptedPose.to_frame_feature_arrays()` |
| Node aliases | Both names present; coordinates identical (one landmark, two labels). FF resolves by name — no double-counting of anatomy |
| Track ordering | Tensor axis sorted; `focal_id`/`partner_id` use real track IDs |
| S5 `orig_frame_idx` | Passed through adapter into FF rows |
| Provenance categories | Distinct in `impute_stage` / `is_kf` / `is_imputed` |
| Unresolved implantation | Never defaults to track `0` |

**Adapter code changes in Sprint 2:** none required. Consumption of `AdaptedPose.implantation` replaces legacy FF `argmax(headstage counts)` and P50 `id==0`.

---

## Deviations from legacy behavior

| Topic | Legacy | Sprint 2 | Reason |
| --- | --- | --- | --- |
| FF input | NPZ + `impute_meta.json` | `AdaptedPose` | No Pipeline 30 |
| Implantation in FF | `argmax` finite headstage counts (always picks a track if any) | Sprint 1 unique-owner only; else unresolved | Avoid inventing identity when ambiguous |
| Robust nose↔headstage fallback | Hardcoded dyad `(0,1)` / `(1,0)` | Any row involving resolved implanted track id | Generalize without assuming id 0 |
| `is_implanted` | `focal_id == 0` | `focal_id == implanted_track_id` iff resolved; else **NaN** + warning (Sprint 3 review) | Sprint 1 D3 / Sprint 3 |
| Blob join source | Flat `data/blob_features/...` path | In-memory S5 parquet table | Session-centric suite |
| Blob missing keys | Soft NaN fill | Hard fail on missing/duplicate join keys (default) | Safer for S6 |
| `exp_name` / `iteration_name` | Config experiments | Session folder name + `"suite"` | Suite session model |
| Runtime `legacy/` | Required | **Forbidden** | Tracked package only |

Obsolete columns still emitted when FF computes them (`nose_any_refined`, `*_is_pchip` zeros) for chain parity; they are not active model requirements.

---

## Implantation and quality-field semantics

### Quality

- `*_is_kf` ← S4 Kalman only  
- `*_is_pchip` ← always 0  
- `*_any_imputed_*` ← pchip ∨ kf → Kalman-only under S4  
- `was_refined` / `nose_any_refined` ← false  
- Point scores pass through; never synthesized  

### Implantation

- Resolved → headstage geometry block + robust fallbacks + `is_implanted`  
- Unresolved → headstage-dependent FF blocks remain NaN where legacy gated on implanted id; ordinary pose features preserved; `is_implanted=NaN` with explicit warning / attrs (Sprint 3)  

---

## Generated table structure

- One row per ordered `(focal_id, partner_id)` × `dense_frame_idx` (120 Hz)  
- For `K` tracks: `K*(K-1)` rows per sample  
- Identity: `focal_id`, `partner_id`, `dense_frame_idx`, `sample_idx`, `time_sec`/`t_120`, `orig_frame_idx`  
- Contents: full FF feature set + S5 blob columns + P50 post columns  
- Unique key: `(focal_id, partner_id, dense_frame_idx)`  

No Parquet publication in this sprint.

---

## Tests

`tests/combined_features/test_processing.py` covers aliases, directional i/j, implantation, Kalman-only imputed flags, geometry smoke, blob cardinality, occlusion/echo presence, blanks, full in-memory chain, and representative active-model column names.

Also retain Sprint 1 eligibility/adapter tests.

---

## Limitations

- No numerical regression vs full-session legacy caches  
- Pandas fragmentation warnings from legacy column-by-column inserts (unchanged formulas)  
- Headstage targeting cosine still uses legacy `nan_to_num(..., nan=0.0)` where that formula existed  
- Model registry / `feature_spec` checker deferred to Sprint 3  

---

## Sprint 3 prerequisites

1. Session-centric run layout + `dyadic_features_full.parquet` export  
2. `run_meta` / `settings_used` / `qc.json` writers  
3. `model_compat` checker against configured registry  
4. Run discovery + downstream eligibility stubs  
5. Optional: quiet fragmentation / copy-once DataFrame hygiene without formula changes  
