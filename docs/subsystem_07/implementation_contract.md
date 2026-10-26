# Subsystem 07 — Behavioral Inference

## Implementation Contract (S7-B)

**Status:** Binding design contract. S7 MVP **complete / lab-validated** (Sprints 1–4 + operator laboratory GUI acceptance). See [`sprint_4_gui_integration.md`](sprint_4_gui_integration.md), [`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md), [`laboratory_acceptance_checklist.md`](laboratory_acceptance_checklist.md).  
**Date:** 2026-10-26  
**Authority:** Accepted S7-A audit (`architecture_audit.md`) + source inspection of legacy Pipelines 70 / 90 / 100, YAML configs, local model bundles, S6 handoff APIs, and existing GUI task/navigation architecture.  
**Architectural decision (accepted):** one S7 Behavioral Inference subsystem and one Group 3 GUI page, containing **separate dyadic and focal model adapters**.

S7 consumes an eligible S6 `dyadic_features_full.parquet`, scores configured dyadic/focal detector bundles, and publishes framewise predictions + QC/provenance. Cross-model arbitration, bout cleaning, and ethogram packaging remain **Subsystem 08** (legacy Pipeline 100).

---

## 0. Binding decisions frozen in S7-B

| Topic | Binding stance |
| --- | --- |
| Subsystem shape | One package + one GUI page; dyadic adapter + focal adapter |
| Dyadic directionality | Collapse to unordered pair **before** scoring; **no** directional aggregation / averaging |
| Focal directionality | Keep both `i→j` and `j→i`; no collapse |
| Threshold authority | Versioned **inference configuration** (YAML) is runtime authority; never silently substitute bundle `threshold_working` |
| Contact fusion | Keep dual-specialist routing + fused `contact` inside the dyadic adapter |
| Nose-to-body | Two-stage specialists preferred; fused probability; **legacy 5-frame rolling median** retained in the focal adapter under `legacy_parity` (MVP default) |
| Canonical predictions | Long-form `predictions.parquet` is scientific source of truth |
| S8 handoff | Also emit Pipeline-100-shaped wide views by default so S8 can recover legacy inputs |
| Failures | Missing required columns / missing required bundles → hard fail for that detector; **no invented specialist fallbacks** beyond documented legacy single-stage `nose_to_body` |
| Model binaries | Lab-local registry root; track manifests / hashes / specs, not joblib binaries by default |
| S8 boundary | No hierarchical conflict resolution, hard-gap gates, reciprocity, or bout stitching in S7 |

---

## 1. Dyadic inference semantics (frozen)

### 1.1 Source of truth

Legacy: `legacy/pipelines/70_dyadic_inference.py` collapse block (sort → `mouse_a`/`mouse_b` → `drop_duplicates(..., keep="first")`).

### 1.2 Exact directional-row selection (collapse rule)

Given directional S6 rows with keys `(exp_name, dense_frame_idx, focal_id, partner_id)`:

1. Require non-null `dense_frame_idx` (legacy raises on NaN).
2. Stable-sort by `(exp_name, dense_frame_idx, focal_id, partner_id)`.
3. Define unordered pair IDs:
   - `mouse_a = min(focal_id, partner_id)`
   - `mouse_b = max(focal_id, partner_id)`
4. Collapse with `drop_duplicates(subset=[exp_name, dense_frame_idx, mouse_a, mouse_b], keep="first")`.
5. Drop helper columns `mouse_a` / `mouse_b`.
6. Score **only** the surviving rows. Do **not** average, max, or otherwise fuse the discarded direction’s feature vector.

#### Canonical equivalent (binding for S7)

For comparable integer track IDs (lab default), step 4 is equivalent to:

> For each `(exp_name, dense_frame_idx, mouse_a, mouse_b)`, keep the unique directional row where  
> `focal_id == mouse_a` and `partner_id == mouse_b`  
> (lower ID as focal; higher ID as partner).

S7 **must** implement this canonical selection explicitly (not rely on accidental row order). The sort+`keep="first"` form remains the parity reference against Pipeline 70.

#### Two-mouse cardinality

For `K=2` animals: S6 emits `2 × N_frames` directional rows; after collapse S7 emits **`N_frames`** dyadic prediction entities per undirected detector.

#### Explicit non-goals

| Forbidden | Reason |
| --- | --- |
| Score both directions then average / max probs | Changes scientific output vs Pipeline 70 |
| Rebuild symmetric feature vectors | Not what 70 does |
| Publish two undirected rows per pair | Breaks Pipeline 100 uniqueness check |

### 1.3 Pair-level vs directionally dependent outputs

| Detector / output | Published grain | True pair-level? | Directional dependence remaining after collapse |
| --- | --- | --- | --- |
| `nose_to_nose` | Undirected pair | **Published** as pair-level | Scored on **one** directional feature vector (`*_i`/`*_j` from kept row). Not feature-symmetric by construction. |
| `side_by_side` | Undirected pair | Same | Same |
| `mutually_disengaged` | Undirected pair | Same | Same |
| `contact_intact` / `contact_occluded` specialists | Undirected pair | Same | Same; additionally gated by `is_total_occlusion` on the kept row |
| Fused `contact` | Undirected pair | Same | Same |
| Routing flag `is_total_occlusion` | Used at pair grain | **Effectively pair-symmetric** under S6 definition `(focal_lost \| partner_lost)` — swapping i/j preserves the OR | Value should match across both directions of a pair; S7 still reads it from the **kept** row only |

**S8 implication:** Pipeline 100 expands undirected dyadic probs onto both focal directions via `(min_id, max_id)` pair keys and treats `nose_to_nose`, `contact`, `side_by_side`, `mutually_disengaged` as symmetric. S7 must therefore guarantee **at most one** undirected row per `(frame, mouse_a, mouse_b)`.

### 1.4 Dyadic temporal operations

**None.** No probability smoothing in the dyadic adapter.

### 1.5 Active dyadic detector set (initial config)

Logical config list (matches `legacy/configs/dyadic_inference.yaml`):

| Config name | Runtime bundles | Notes |
| --- | --- | --- |
| `contact` | expands → `contact_intact` + `contact_occluded` | Dual-model; fused outputs required |
| `nose_to_nose` | `nose_to_nose` | |
| `mutually_disengaged` | `mutually_disengaged` | |
| `side_by_side` | `side_by_side` | |

Obsolete standalone `contact` joblib remains **unsupported** (`OBSOLETE_DYADIC_DETECTORS`). Selecting logical `contact` must load the two specialists, never the obsolete bundle.

---

## 2. Focal inference semantics (frozen)

### 2.1 Source of truth

Legacy: `legacy/pipelines/90_focal_inference.py`.

### 2.2 Focal / partner row selection

| Rule | Contract |
| --- | --- |
| Grain | Directional: every S6 row `(focal_id, partner_id, dense_frame_idx)` with `focal_id ≠ partner_id` |
| Collapse | **Forbidden** |
| Metadata | Require non-null `exp_name` (or S7 session label), `focal_id`, `partner_id`, `dense_frame_idx` |
| Soft check | For `K=2`, expect `≈ 2 × n_unique_frames` rows; warn if unequal (legacy prints a note; do not auto-repair) |

Approach / retract / nose-to-body / nose-to-tail are asymmetric; scoring both directions is correct and is **not** double-counting.

### 2.3 Model-specific feature preparation

Shared with dyadic:

1. Load `feature_spec.json` → ordered `feature_cols` (prefer non-empty `selected_feature_cols` only if S7 registry discovery mirrors S6; Pipeline 90 uses `feature_cols`).
2. Missing required columns → **hard error** for that detector.
3. If blob helpers `cc_count` / `min_inter_contour_dist` absent → warn and create NaN columns (defensive; eligible S6 already publishes them).
4. Assemble `X = features_df[feature_cols]` in exact training order.
5. No imputation, clipping, or dtype coercion beyond pandas subsetting.
6. Call `model.predict_proba(X)`; positive class = column index `1`. Require shape `(n, 2)` (Pipeline 90 validates; Pipeline 70 assumes).
7. NaNs/Infs in features are passed through; XGBoost handles missing natively. Do not drop rows.

### 2.4 Nose-to-body two-stage path (preferred)

When **both** `nose_body_intact` and `nose_body_occ` bundles load:

1. Require `is_total_occlusion` present (hard error if missing).
2. Initialize fused `prob_nose_to_body = 0.0` for all rows.
3. `mask_intact = is_total_occlusion.eq(0)`; `mask_occ = is_total_occlusion.eq(1)`.
4. Score only masked rows with the matching specialist; write into fused `prob_nose_to_body`.
5. Write specialist **binary** columns from **raw** specialist probabilities:
   - `pred_nose_body_intact` on intact mask (`prob >= intact_threshold`)
   - `pred_nose_body_occ` on occluded mask (`prob >= occ_threshold`)
6. Apply five-frame rolling median (§2.5) to fused `prob_nose_to_body`.
7. Re-binarize **smoothed** fused probs with stage-specific thresholds on the **original** masks → `pred_nose_to_body`.

Rows where `is_total_occlusion` is neither 0 nor 1 (including NaN) remain at fused prob `0` / pred `0`. **Do not** invent a third specialist.

### 2.5 Five-frame nose-to-body rolling median (binding)

| Parameter | Legacy value | Binding |
| --- | --- | --- |
| Window | `5` | Fixed in MVP parity mode |
| `min_periods` | `1` | Fixed |
| Grouping keys | `(exp_name, focal_id, partner_id)` | Per directional dyad stream |
| Time order | Sort by `(exp_name, focal_id, partner_id, dense_frame_idx)` before transform | Required |
| Operation | `rolling(...).median()` | Median, not mean |
| Target series | Fused `prob_nose_to_body` **after** specialist stitching | Only this series |
| Specialist raw probs | Not published by Pipeline 90 | S7 **must** also publish pre-smooth diagnostics (§3.5) without changing parity outputs |

#### Edge behavior

- Leading frames in each directional group use whatever history exists (`min_periods=1`): frame 0 → median of 1 value; frame 1 → median of 2; … until window fills.
- No center-alignment / backward window: pandas default trailing window.
- Groups are independent; no cross-partner leakage.

#### NaNs during smoothing

- Pandas `Series.rolling(...).median()` defaults to skipping NaNs inside the window (`skipna=True` semantics of the reduction).
- If every value in the window is NaN, the smoothed value is NaN.
- S7 must **not** fill NaNs with 0 before smoothing (legacy does not).
- After smoothing, binarize with `prob >= threshold`; NaN comparisons yield False → `pred=0` under pandas/numpy rules when assigned via boolean mask (document and cover with a unit test).

#### Threshold vs smooth order

| Stage | When |
| --- | --- |
| Specialist `pred_nose_body_*` | **Before** smoothing, from raw specialist probs |
| Fused `pred_nose_to_body` | **After** smoothing, from smoothed fused probs |

#### What is smoothed

| Series | Smoothed? |
| --- | --- |
| Fused / routed `prob_nose_to_body` | **Yes** (legacy) |
| Specialist probabilities | N/A in legacy (not stored); S7 diagnostics keep **unsmoothed** specialist probs |
| Other focal detectors (`approach`, `retract`, `nose_to_tail`) | **No** |

### 2.6 Single-stage `nose_to_body` fallback (legacy-only allowed fallback)

If either `nose_body_intact` or `nose_body_occ` is missing, Pipeline 90 falls through to loading bundle `nose_to_body` and scores all rows with **no** rolling median.

S7 binding:

- Allowed only when two-stage bundles are incomplete **and** operator explicitly selects single-stage / legacy fallback (or parity mode detects the same missing-bundle condition Pipeline 90 used).
- Must record `nose_to_body_mode: single_stage_fallback` in `model_plan` / QC.
- Must **not** invent other fallbacks (e.g. do not synthesize specialists from `nose_body` JSON experts).

### 2.7 Active focal detector set (initial config)

Matches `legacy/configs/90_focal_inference.yaml`:

| Config name | Runtime bundles | Notes |
| --- | --- | --- |
| `nose_to_body` | prefers `nose_body_intact` + `nose_body_occ` | YAML `two_stage_routing` block is documentation; Pipeline 90 hardcodes those names |
| `nose_to_tail` | `nose_to_tail` | Directional; no smooth |
| `approach` | `approach` | Directional; no smooth |
| `retract` | `retract` | Directional; no smooth |

On-disk but **not** in active YAML: standalone `nose_to_body` (fallback), `nose_body` / `nose_tail` JSON experts (no joblib; unsupported for S7 MVP).

### 2.8 Ambiguities preserved (not silently “fixed”)

| Ambiguity | Legacy fact | S7 stance |
| --- | --- | --- |
| Dyadic contact NaN occlusion → treat as 0 (`fillna(0)`); focal nose-to-body uses `.eq` without fillna | Inconsistent | Preserve family-specific behavior |
| YAML `two_stage_routing` unused by Pipeline 90 | Names hardcoded | S7 config may declare routing explicitly; must match these bundle names for parity |
| Only specialist **preds** published for nose-to-body, not specialist **probs** | Legacy gap | S7 publishes both (diagnostics) while keeping fused parity columns |
| Smoothing overwrites fused prob in-place | Raw fused lost | S7 stores `prob_nose_to_body_raw` + smoothed `prob_nose_to_body` |
| Single-stage path skips smoothing | Legacy | Document; do not “helpfully” smooth |

---

## 3. Specialist routing and fusion (frozen)

### 3.1 Contact intact / occluded routing (dyadic)

Gate column: `is_total_occlusion` (required when either contact specialist is active).

| Specialist | Score when | Else |
| --- | --- | --- |
| `contact_intact` | `to_numeric(...).fillna(0).astype(int).eq(0)` | `prob=0`, then threshold → usually `pred=0` |
| `contact_occluded` | `...eq(1)` | `prob=0` |

Missing `is_total_occlusion` → **hard error** (no invented default mask).

Invalid/missing specialist bundle when logical `contact` selected → **hard error**. Do not fall back to obsolete `contact`.

### 3.2 Fused contact probability / prediction

Exact Pipeline 70 algorithm:

1. If `prob_contact_intact` exists:  
   `prob_contact = to_numeric(prob_contact_intact).fillna(0.0)`  
   `pred_contact = to_numeric(pred_contact_intact).fillna(0).astype(int)` (when pred intact exists).
2. If fused contact, occluded specialist, and `is_total_occlusion` all present:  
   where occlusion mask (`eq(1)` with same fillna-0 coercion), overwrite  
   `prob_contact ← prob_contact_occluded` and `pred_contact ← pred_contact_occluded`.

This is **hard routing**, not a weighted blend. Thresholding already occurred per specialist before fusion; fusion copies specialist probs/preds.

### 3.3 Nose-to-body intact / occluded routing (focal)

See §2.4–§2.6. Summary:

- Hard split on `is_total_occlusion` ∈ {0,1} without fillna.
- Stitch into fused probability, then smooth, then re-threshold with stage thresholds.
- No weighted fusion.

### 3.4 Masks, gates, and source columns

| Column / gate | Used by | Role in S7 |
| --- | --- | --- |
| `is_total_occlusion` | Contact specialists; nose-to-body two-stage | Routing mask only |
| `cc_count`, `min_inter_contour_dist` | Feature specs (esp. occluded specialists) | Features, not S7 gates |
| Bundle `thresholds.json` | Not read by 70/90 | Provenance / training echo only |
| S8 optical assist / hard distance gates | Pipeline 100 | **Out of scope** for S7 |

### 3.5 Threshold application (inference)

| Family | Rule |
| --- | --- |
| Ordinary detector | `pred = 1` iff `prob_positive >= working_threshold` |
| Contact specialists | Same, on routed subset; excluded rows have `prob=0` then compared |
| Nose-to-body specialists | Raw-prob threshold → specialist preds |
| Fused nose-to-body | Smoothed-prob threshold with stage-specific thresholds on original masks |
| Default if detector absent from config | `0.5` (legacy) — S7 MVP should instead **fail** if an active detector lacks an explicit threshold entry (stricter; document as deliberate hardening). Parity mode may keep `0.5` default behind a flag. |

**Recommended MVP hardening:** require explicit thresholds for every selected detector/stage in inference config; do not silently use `0.5`.

### 3.6 Intermediate probabilities to publish for debugging

| Name | Required | Notes |
| --- | --- | --- |
| `prob_contact_intact`, `pred_contact_intact` | Yes | Legacy |
| `prob_contact_occluded`, `pred_contact_occluded` | Yes | Legacy; S8 may re-multiplex |
| `prob_contact`, `pred_contact` | Yes | Fused logical |
| `prob_nose_body_intact`, `prob_nose_body_occ` | Yes (S7 addition) | Raw specialist probs (unsmoothed) |
| `pred_nose_body_intact`, `pred_nose_body_occ` | Yes | Legacy |
| `prob_nose_to_body_raw` | Yes (S7 addition) | Pre-smooth fused |
| `prob_nose_to_body`, `pred_nose_to_body` | Yes | Post-smooth fused (parity column names) |
| `is_total_occlusion` | Copy onto focal outputs when present | Legacy focal copy |

Do not invent optical-assist or arbiter fused columns (`prob_contact_fused`, etc.) — those are S8.

---

## 4. Threshold and model contract

### 4.1 Versioned inference configuration

Proposed tracked config (initial values = legacy YAML):

```text
configs/subsystem_07/behavioral_inference_default.yaml
```

Conceptual schema (binding fields):

```yaml
schema_version: behavioral_inference_settings_v1
model_iteration: v1          # registry version segment
parity_mode: legacy_parity   # legacy_parity | raw_framewise

model_registry:
  root: null                 # operator-configured; reuse S6 ModelRegistrySettings shape
  dyadic_subdir: dyadic/v1
  focal_subdir: focal/v1
  include_obsolete: false

active_detectors:
  dyadic: [contact, nose_to_nose, mutually_disengaged, side_by_side]
  focal: [nose_to_body, nose_to_tail, approach, retract]

working_thresholds:          # AUTHORITATIVE at runtime
  contact:
    intact: 0.9731
    occluded: 0.4295
  nose_to_nose: 0.0816
  mutually_disengaged: 0.0469
  side_by_side: 0.0095
  nose_to_body:
    intact: 0.6141
    occluded: 0.4825
  nose_to_tail: 0.6636
  approach: 0.3766
  retract: 0.4892

adapters:
  dyadic_collapse: lower_id_as_focal   # §1.2
  nose_to_body_rolling_median:
    enabled: true          # true under legacy_parity; false under raw_framewise
    window: 5
    min_periods: 1
  emit_wide_views: true
  fail_fast_on_detector_error: true
  allow_nose_to_body_single_stage_fallback: true  # legacy-compatible only

user_threshold_overrides: {}  # filled only when operator overrides in GUI
```

### 4.2 Threshold inventory (initial = legacy)

| Detector / stage | Working threshold | Bundle `threshold_working` | Bundle / metrics F1 | Runtime source |
| --- | --- | --- | --- | --- |
| `contact_intact` | **0.9731** | 0.2 | ≈0.9615 (`thresholds.json`) | Config nested `contact.intact` |
| `contact_occluded` | **0.4295** | 0.2 | ≈0.4295 | Config nested `contact.occluded` |
| `nose_to_nose` | **0.0816** | 0.2 | ≈0.0816 | Config |
| `mutually_disengaged` | **0.0469** | 0.2 | ≈0.0469 | Config |
| `side_by_side` | **0.0095** | 0.2 | ≈0.0095 | Config |
| `nose_to_body` intact | **0.6141** | *(no thresholds.json)* | ≈0.6141 (`metrics.json`) | Config nested |
| `nose_to_body` occluded | **0.4825** | — | ≈0.4825 | Config nested |
| `nose_to_tail` | **0.6636** | — | ≈0.6636 | Config |
| `approach` | **0.3766** | — | ≈0.3766 | Config |
| `retract` | **0.4892** | — | ≈0.4892 | Config |

**Precedence (binding):**

1. Explicit user override recorded for this run (`user_threshold_overrides` / GUI advanced)  
2. Inference configuration `working_thresholds`  
3. **Hard fail** if still unresolved (MVP)  
4. Bundle `threshold_f1` / `threshold_working` → **never auto-applied**; may be shown in UI as reference only  

Parity note: Pipeline 70/90 defaulted missing thresholds to `0.5`. S7 documents that behavior under `parity_mode=legacy_parity` + `allow_default_threshold_0_5=true` if exact script parity is required; default recommended product setting is fail-closed.

### 4.3 Model name, family, API

| Field | Contract |
| --- | --- |
| Family | `xgboost` classifiers exposed via sklearn-compatible API |
| Load | `joblib.load(model.joblib)` |
| Predict | `predict_proba(X)[:, 1]` |
| Feature order | Exact `feature_spec.json` / `feature_cols` order |
| Positive class | Index 1 |

### 4.4 Executable model inventory (local lab, 2026-10-26)

#### Completeness (files)

| Bundle | `#features` | `model.joblib` | `feature_spec.json` | Active? | SHA-256 of `model.joblib` |
| --- | --- | --- | --- | --- | --- |
| `dyadic/v1/contact_intact` | 139 | yes | yes | Yes | `ffd2aeb7bc8f8b5582c85738f9192015e8f97cbe304f281fb31fa70ac19533f2` |
| `dyadic/v1/contact_occluded` | 34 | yes | yes | Yes | `1301ef479558a9e5031349c5a0dabf3cfae29e6724fc8567871af2ca30cd1256` |
| `dyadic/v1/nose_to_nose` | 44 | yes | yes | Yes | `84ae82becf4abc908d4fb8ecb72fc95ba322121d66acc997a79cdc419d30e2fa` |
| `dyadic/v1/mutually_disengaged` | 79 | yes | yes | Yes | `5c241be8ad687cbd320f408225b44e0ad2f675de5769ff33fa79957770c49f95` |
| `dyadic/v1/side_by_side` | 26 | yes | yes | Yes | `b8b3f65de40528be05bf9d606c9fd73e3bbcb1720b502fb8a562ac0fda9877c7` |
| `dyadic/v1/contact` | 84 | yes | yes | **Obsolete / unsupported** | `6a2bc6ed4c76ba651b0c0633763fa119c96572c9a29796b3955cb2c95ec722e4` |
| `focal/v1/nose_body_intact` | 109 | yes | yes | Yes (via `nose_to_body`) | `d96e78645caecb3e4875925f2d73daba5f4316e08e79e446fb4a7f7423c9d65c` |
| `focal/v1/nose_body_occ` | 26 | yes | yes | Yes (via `nose_to_body`) | `318f33f032bf28c6bd9d32ad78197eb728529804d388e491d61ff9272d268d2d` |
| `focal/v1/nose_to_tail` | 133 | yes | yes | Yes | `6150e208e7edf4613c3b46751a3527419dea6e05307b7af6d804fd28ac6fab74` |
| `focal/v1/approach` | 71 | yes | yes | Yes | `3878b56b008d93855ae1944e35e6cd15a74953505b4074f39b0015c2e663f094` |
| `focal/v1/retract` | 39 | yes | yes | Yes | `33817b07a60310cc03b52361b35108aa334938d95e79bd319b10dba9ddff778b` |
| `focal/v1/nose_to_body` | 49 | yes | yes | Fallback only | `ceb650551d575218a7859132ab46a9e569fc2edffab8b3b079f33fcc54ebacf5` |
| `focal/v1/nose_body` | — | **no** | yes (+ JSON experts) | Unsupported MVP | — |
| `focal/v1/nose_tail` | — | **no** | yes (+ JSON experts) | Unsupported MVP | — |

#### Executability vs schema compatibility

| Check | Result (this workstation / default suite env) |
| --- | --- |
| S6 `model_compat` column presence (prior S7-A smoke) | Active joblib detectors `compatible` vs E-session features |
| Bundle file completeness | Active set complete (§ above) |
| `joblib.load` + `predict_proba` in **default** suite environment | **Not executable** — `xgboost` and `scikit-learn` are **not** installed / not declared in `pyproject.toml` |
| `joblib` present | Yes (`1.5.3` observed in smoke), but not declared as a project dependency |

**Verdict:** Active bundles are **complete on disk** and **schema-compatible** with S6 features, but they are **not yet runtime-executable** inside the tracked suite environment until an `inference`/`s7` optional extra pins and installs compatible `xgboost`, `scikit-learn`, and `joblib`. Controlled unpickle/`predict_proba` smoke under those pins is a **Sprint 1 prerequisite**, not claimed done here.

`train_meta.json` files inspected do **not** record reliable `xgboost_version` / `sklearn_version` fields — library pins remain an unresolved lab measurement (§9).

### 4.5 Required runtime dependencies (to add)

| Package | Role | Declared in suite today? |
| --- | --- | --- |
| `xgboost` | Unpickle + predict | **No** |
| `scikit-learn` | Often required for sklearn-API wrappers / unpickle graph | **No** |
| `joblib` | `model.joblib` load | Present transitively in env; **not** declared |
| `pandas` / `numpy` / `pyarrow` / `PyYAML` | Features + artifacts | Yes |

Propose optional extra, e.g. `[project.optional-dependencies] s7 = [...]`, with versions pinned after the controlled load smoke.

### 4.6 NaN / Inf treatment

| Case | Rule |
| --- | --- |
| NaN features | Pass through to `predict_proba`; QC records fractions |
| Inf features | Pass through; QC warn (do not invent clips) |
| Missing required feature columns | Fail detector |
| Missing optional blob helpers | NaN fill columns + warn |
| Router-excluded rows | Forced `prob=0` (contact specialists; nose-to-body non-matching masks) |

### 4.7 User override rules

- GUI may display config thresholds and allow advanced overrides.
- Overrides apply only to the current run and must be written into `settings_used.yaml` + `model_plan.json` / `model_manifest.json`.
- Overrides never rewrite on-disk bundle `thresholds.json`.
- Clearing overrides reverts to inference configuration values.

### 4.8 Model metadata and SHA-256 provenance

Each run must record, per loaded bundle:

- `family`, `detector_name` / `specialist_name`, `model_version` (`v1`)
- Absolute / resolved `bundle_dir`
- `model_sha256` (full digest of `model.joblib`)
- Feature-spec path + ordered feature count
- Threshold used + threshold source (`config` | `user_override`)
- Library versions actually imported (`xgboost`, `sklearn`, `joblib`, python)

Registry manifest (tracked, small) should list expected SHA-256 values for lab verification without committing binaries.

---

## 5. Prediction artifacts

### 5.1 Run layout

```text
{session_root}/behavioral_inference/behavioral_inference__YYYYMMDDTHHMMSS/
  predictions.parquet                 # canonical long-form
  dyadic_predictions_wide.parquet     # Pipeline 70-shaped view (default on)
  focal_predictions_wide.parquet      # Pipeline 90-shaped view (default on)
  specialist_diagnostics.parquet      # optional dense debug; may be embedded in long-form instead
  run_meta.json
  settings_used.yaml
  qc.json
  model_manifest.json                 # aka model_plan + digests
  processing_log.txt
```

Schema versions (proposed):

| Artifact | `schema_version` |
| --- | --- |
| Long-form predictions | `behavioral_predictions_v1` |
| Run meta | `behavioral_inference_run_meta_v1` |
| QC | `behavioral_inference_qc_v1` |
| Model manifest | `behavioral_inference_model_manifest_v1` |
| Settings | `behavioral_inference_settings_v1` |

### 5.2 Canonical `predictions.parquet` (long-form)

One row per `(sample, entity, detector_or_specialist_output)`:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | str | `behavioral_predictions_v1` |
| `sample_idx` | int64 | From S6 (`== dense_frame_idx` under S6 contract) |
| `dense_frame_idx` | int64 | Shared 120 Hz index |
| `time_sec` | float64 | From S6 |
| `prediction_grain` | str | `undirected_pair` \| `directional` |
| `mouse_a` / `mouse_b` | int64 nullable | Required when `undirected_pair`; `a < b` |
| `focal_id` / `partner_id` | int64 nullable | Required when `directional`; for undirected rows store collapse survivors for provenance |
| `family` | str | `dyadic` \| `focal` |
| `detector_name` | str | Logical name (`contact`, `approach`, `nose_to_body`, …) |
| `specialist_name` | str nullable | `contact_intact`, `nose_body_occ`, …; null for ordinary single-model detectors |
| `output_role` | str | `specialist` \| `logical_fused` \| `primary` |
| `model_version` | str | e.g. `v1` |
| `model_bundle_dir` | str | Resolved path |
| `model_sha256` | str nullable | Binary digest |
| `prob_positive_raw` | float64 | Probability **before** S7 smoothing (equals scored prob for unsmoothed detectors) |
| `prob_positive` | float64 | Effective probability after S7 adapter post-process (smoothed nose-to-body; else = raw) |
| `threshold` | float64 | Threshold used for `pred` |
| `pred` | int8 | `0/1` from effective `prob_positive >= threshold` (or forced 0 when invalid) |
| `input_valid` | bool | False when router excluded / unscored |
| `invalid_reason` | str nullable | e.g. `occlusion_mask_excluded`, `non_matching_occlusion_state` |
| `source_s6_run_id` | str | Provenance |
| `session_root` | str | Provenance |

#### Unique prediction keys

| Grain | Unique key |
| --- | --- |
| Undirected | `(dense_frame_idx, mouse_a, mouse_b, detector_name, specialist_name, output_role)` |
| Directional | `(dense_frame_idx, focal_id, partner_id, detector_name, specialist_name, output_role)` |

#### Output cardinality (K=2)

| Output | Rows / frame |
| --- | --- |
| Each undirected dyadic primary / fused | 1 |
| Each contact specialist | 1 (same undirected keys; `input_valid` may be false on excluded mask) |
| Each focal primary / fused | 2 |
| Each nose-to-body specialist | 2 (masked validity) |

### 5.3 Score semantics: raw vs effective

| Detector class | `prob_positive_raw` | `prob_positive` (effective) | `pred` from |
| --- | --- | --- | --- |
| Ordinary dyadic / focal | `predict_proba`[:,1] | Same | Effective |
| Contact specialists | Scored or forced 0 | Same | Effective |
| Fused contact | Routed copy of specialists | Same (no S7 smooth) | Effective |
| Nose-to-body specialists | Specialist `predict_proba` | Same (unsmoothed) | Specialist threshold on raw |
| Fused nose-to-body | Pre-smooth stitch | Post-smooth median | Stage threshold on **effective** |

S8 must consume **effective** probs for parity with Pipeline 90’s published `prob_nose_to_body`, and may also read raw diagnostics.

### 5.4 Wide views (S8 / Pipeline 100 recovery)

#### `dyadic_predictions_wide.parquet`

Keys: `exp_name` (compat label), `focal_id`, `partner_id`, `dense_frame_idx` of **collapse survivors**, plus:

- `prob_*` / `pred_*` for each selected dyadic detector
- Contact extras: intact, occluded, fused

Grain: one row per `(frame, unordered pair)` — undirected uniqueness required.

#### `focal_predictions_wide.parquet`

Keys: `exp_name`, `focal_id`, `partner_id`, `dense_frame_idx`, optional `is_total_occlusion`, plus all `prob_*` / `pred_*` including nose-to-body specialist preds and fused (smoothed) columns. Prefer also emitting `prob_nose_to_body_raw` and specialist probs for S8 debugging without breaking Pipeline 100 column discovery (100 looks up `prob_<name>` / `pred_<name>` prefixes).

### 5.5 `run_meta.json` (minimum)

| Block | Contents |
| --- | --- |
| Identity | `run_id`, timestamps, `status`, `session_root`, schema version |
| `source_s6` | run id/dir, parquet path + SHA256, eligibility snapshot, S4/S5 digests echoed from S6 meta |
| `registry` | root, subdirs, include flags |
| `detectors_selected` | dyadic + focal lists after expansion |
| `parity_mode` / adapter flags | collapse rule, rolling median on/off |
| `libraries` | xgboost / sklearn / joblib / python versions |
| `outputs` | artifact paths |
| `qc_summary` | embedded headline stats |

Statuses mirror upstream: `not_started`, `running`, `completed`, `completed_with_warnings`, `failed`.

### 5.6 `settings_used.yaml`

Immutable snapshot of inference config + resolved registry root + user overrides + selected S6 run id + adapter flags.

### 5.7 `qc.json`

Per detector / specialist:

- `n_rows_expected`, `n_scored`, `n_input_invalid`, `n_positive`, positive rate
- mean / median effective prob; NaN feature fraction sampled or exact
- routing coverage (`n_intact`, `n_occluded`) for dual-model detectors
- grain checks (undirected uniqueness; directional `K*(K-1)` expectation)

Run-level:

- S6 eligibility at start and pre-publish
- registry readiness
- partial failure list (if continue mode ever enabled)
- warnings only (never silent science changes)

### 5.8 `model_manifest.json`

Selected detectors → bundle paths, SHA-256, feature counts, thresholds, threshold sources, routing mode, load status.

### 5.9 `processing_log.txt`

Human-readable progress mirroring Pipeline 70/90 printouts (detector positive counts, routing counts, paths).

### 5.10 Missing-input / selected-subset / partial-failure policies

| Case | Policy |
| --- | --- |
| Missing required feature columns for a selected detector | Fail that detector; with `fail_fast_on_detector_error=true` (default) fail the run |
| Selected subset of detectors | Score only selected; wide views contain only those columns; long-form only those rows |
| Load / predict failure | Same as missing columns |
| S6 becomes stale mid-run | Fail before publish |
| Dual-model logical detector with one specialist missing | Contact: fail. Nose-to-body: only documented single-stage fallback (§2.6) |
| Continue-on-error mode | Optional advanced; must mark `completed_with_warnings` and list failed detectors in QC — **not** MVP default |

### 5.11 Source / model staleness

Downstream eligibility for S8 (and GUI reopen):

| Classification | Rule |
| --- | --- |
| `eligible` | completed(+warnings) + artifacts present + recorded S6 digests still match live eligible S6 + selected model SHA-256 still match registry files |
| `stale_source` | S6 superseded or digest drift |
| `stale_models` | Bundle SHA drift vs manifest |
| `failed_or_incomplete` / `missing_required_artifacts` | as named |

Reuse S6 discovery patterns: `summarize_*_project`, `find_latest_eligible_*`, `is_*_eligible_for_downstream`.

### 5.12 Downstream S8 requirements (binding handoff)

S8 (Pipeline 100 today) must be able to:

1. Load directional focal wide preds as the spine.
2. Expand undirected dyadic wide preds onto both directions via `(min(focal,partner), max(...))` without duplicate undirected keys.
3. See fused `prob_contact` / `pred_contact` **and** preferably `prob_contact_intact` / `prob_contact_occluded` (100 re-multiplexes if occluded specialist present).
4. See `prob_nose_to_body` / `pred_nose_to_body` with **legacy smoothed** semantics when parity artifacts are requested.
5. Retain detector identities as column suffixes (`prob_<detector>`), including specialists.
6. Preserve directional semantics for focal detectors (no collapse).

S7 must not perform Pipeline 100’s hierarchical conflict resolution, reciprocity enforcement, hard-gap gates, optical-assist fusion, or bout stitching.

---

## 6. S7 GUI contract (Group 3 — implemented in Sprint 4)

**Implemented:** see `sprint_4_gui_integration.md`.

### 6.1 Navigation

- **Group 3 — Behavioral Inference** with single page: Subsystem 7 — Behavioral Inference (dyadic + focal together).
- S8 remains absent/disabled.
- `GuiTaskKind.BEHAVIORAL_INFERENCE` for `GuiTaskRunner`.

### 6.2 Required page capabilities

1. Session root / project context (same pattern as S6).
2. Eligible S6 run selection (`find_latest_eligible_s6_for_s7` + manual eligible picker).
3. Source eligibility / staleness reporting (reuse S6 classifications).
4. Model registry root configuration (`ModelRegistrySettings`; **required** for Run).
5. Configured model registry browser / readiness (discovery without unnecessary loads; definitive checks before execution).
6. Unified detector selection: dyadic + focal checklists.
7. Threshold / settings visibility; advanced overrides only.
8. Detector availability table (logical detectors + unsupported/obsolete identification).
9. Execute one inference task covering selected detectors; progress via task runner.
10. Model-specific warnings/errors panel.
11. Result summaries (positives, rates, invalid counts, routing coverage).
12. Completed S7 run discovery, reopen, and QC display.

### 6.3 Must not

- Train models
- Run S8 / arbiter
- Edit S1–S6 scientific outputs
- Auto-copy model binaries into the repo

---

## 7. Proposed backend package shape (for later implementation)

```text
src/behavioral_inference/
  contracts.py
  config.py
  registry.py          # extend S6 discovery; load + digest + executable probe
  thresholds.py
  features.py
  adapters/
    dyadic.py          # §1 + contact routing
    focal.py           # §2 + nose_to_body routing/smooth
  predict.py
  export.py            # long-form + wide views
  qc.py
  runner.py
  run_discovery.py
```

GUI (Sprint 4):

```text
src/ui/controllers/behavioral_inference_controller.py
src/ui/pages/behavioral_inference_page.py
```

---

## 8. Implementation roadmap (bounded sprints)

### Sprint S7-1 — Model loading, registry, runtime compatibility

**Prerequisites**

- S7-B accepted
- Lab decision on optional-extra pins **or** willingness to discover pins via controlled smoke in this sprint
- Operator-configured registry root available (local `legacy/models` or equivalent)

**Deliverables**

- Package skeleton + contracts/config
- Registry discovery extending S6 `ModelRegistrySettings`
- Bundle load validation: files present, SHA-256, `joblib.load`, `predict_proba` shape probe on tiny matrix
- Threshold resolution module (config authority)
- Feature matrix validation helpers
- `pyproject.toml` optional `s7` extra once pins known
- Headless unit tests with fixtures / tiny mocks (no GUI)

**Exit criteria:** Active bundles load and score a synthetic 2-row matrix under declared deps; obsolete `contact` rejected; missing columns detected.

### Sprint S7-2 — Dyadic/focal inference and specialist routing

**Prerequisites**

- S7-1 complete
- Eligible S6 E-session (or equivalent) artifact available for parity

**Deliverables**

- Dyadic adapter: §1.2 collapse + contact routing/fusion
- Focal adapter: directional pass-through + nose-to-body route/smooth
- Shared predict/binarize
- Parity tests vs legacy wide outputs on a frozen snippet (and/or full E-session when feasible)
- `legacy_parity` vs `raw_framewise` flag behavior for nose-to-body

**Exit criteria:** Bitwise or documented-tolerance match to Pipeline 70/90 wide preds for active detectors under parity mode.

### Sprint S7-3 — Artifact publication, QC, discovery

**Status:** Implemented (see `sprint_3_artifacts_qc_discovery.md`).

**Prerequisites**

- S7-2 scientific path stable

**Deliverables**

- Long-form `predictions.parquet` + default wide views
- `run_meta.json`, `settings_used.yaml`, `qc.json`, `model_manifest.json`, `processing_log.txt`
- Run discovery + eligibility/staleness vs S6 and model SHA
- Partial-failure policy enforcement (fail-fast default)

**Exit criteria:** Fresh S7 run is discoverable; S8 can reconstruct Pipeline 100 inputs from wide views; stale S6 cannot be selected.

### Sprint S7-4 — Unified GUI and real-session laboratory validation

**Status:** Implemented (see `sprint_4_gui_integration.md`).

**Prerequisites**

- S7-3 headless complete
- GUI extra (`PySide6`) available
- Lab registry + E-session (or production session) ready

**Deliverables**

- Group 3 navigation + S7 page + controller + `GuiTaskKind`
- End-to-end operator acceptance checklist
- Document laboratory evidence

**Exit criteria:** Operator runs dyadic+focal inference from GUI on a real eligible S6 session; QC and summaries reviewed; no S8 claims.

### Cross-cutting prerequisites before each sprint

| Before | Must be true |
| --- | --- |
| S7-1 | Inference dependency strategy agreed; registry path known |
| S7-2 | Executable models under pins; S6 eligibility APIs unchanged |
| S7-3 | Parity semantics signed off (esp. smoothing + collapse) |
| S7-4 | Headless artifacts frozen enough for GUI wiring |

Parallel track: **S8-A** audit may start after S7-3 artifact schemas are accepted, but must not block S7-4.

---

## 9. Unresolved scientific / engineering decisions

These remain open after S7-B freezes; they must not be silently decided in code:

1. **Exact `xgboost` / `scikit-learn` / `joblib` pins** that successfully unpickle lab bundles (blocked on missing deps in default env; measure in S7-1).
2. **Whether product default** for new (non-parity) runs disables nose-to-body rolling median (`raw_framewise`) while MVP acceptance still uses `legacy_parity`.
3. **Fail-closed thresholds** (recommended) vs Pipeline 70/90 silent `0.5` default for missing keys.
4. **Multi-animal K>2** validation of collapse + focal cardinality (lab today is two-mouse; rule is defined generally but not validated).
5. **Whether to commit feature_spec / metrics excerpts** into tracked `configs/subsystem_07/` vs hash-only manifest.
6. **Binary distribution** policy (lab path only vs optional release assets / LFS) — still lab-local by default.
7. **Tolerance policy** for floating parity vs legacy (bitwise vs `allclose`) once deps differ slightly from training environment.
8. **Group 3 placeholder** visibility before S7-4 vs hide until ready.

---

## 10. Files inspected (checklist)

**Legacy**

- `legacy/pipelines/70_dyadic_inference.py`
- `legacy/pipelines/90_focal_inference.py`
- `legacy/pipelines/100_behavioral_arbiter.py` (`_expand_dyadic_to_directional`, contact multiplexer boundary)
- `legacy/configs/dyadic_inference.yaml`
- `legacy/configs/90_focal_inference.yaml`
- `legacy/models/dyadic/v1/*/feature_spec.json`, `thresholds.json`, `train_meta.json`, `metrics.json`, `model.joblib` presence + SHA-256
- `legacy/models/focal/v1/*/feature_spec.json`, `metrics.json`, joblib / JSON expert presence + SHA-256

**Tracked suite**

- `docs/subsystem_07/architecture_audit.md` (S7-A)
- `docs/subsystem_06/implementation_contract.md` (artifact / eligibility patterns)
- `src/combined_features/{config,contracts,model_compat,run_discovery,runner,pipeline50_post,qc}.py`
- `src/ui/{main_window,tasks}.py`, `src/ui/pages/combined_features_page.py`
- `pyproject.toml` (dependency gap)

**Read-only smokes**

- Bundle inventory + SHA-256 digests
- Attempted `joblib.load` / `predict_proba` → failed without `xgboost`/`sklearn` in default env (documents executability gap)

---

## 11. Stop point

This document is the binding S7-B implementation contract for architectural review. **No production inference package, GUI page, model copy, commit, or push is included in this gate.**
