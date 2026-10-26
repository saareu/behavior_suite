# Subsystem 07 — Behavioral Inference

## Architecture Audit (S7-A)

**Status:** Discovery milestone (documentation only). **Superseded for binding decisions by** [`implementation_contract.md`](implementation_contract.md) **(S7-B).**  
**Date:** 2026-10-26  
**Scope:** Inspect legacy Pipelines 70 / 90 / 100 contracts, local model bundles, and S6 handoff APIs; propose a unified S7 inference architecture. **No S7 implementation in this sprint.**  
**Authority:** Local `legacy/` tree (git-excluded), tracked `src/combined_features/` + `src/ui/`, and the E-session S6 artifact under `test_cases/E_full_session_integration/`.

**Follow-on:** S7-B binding contract — [`implementation_contract.md`](implementation_contract.md).

---

## 0. Preconditions and inspection method

### 0.1 Upstream closure

| Check | Result |
| --- | --- |
| S6 docs | `docs/subsystem_06/mvp_scope_and_roadmap.md` marks S6 **MVP complete / lab-validated** |
| S6 package | `src/combined_features/` publishes `dyadic_features_full.parquet` + `model_compat.json` |
| S6 GUI | Group 2 hosts Combined Features; S6 does **not** run inference |
| S7 code | Absent — no `src/behavioral_inference/` (or equivalent) yet |
| Group 3 UI | Absent — only Group 1 (Data Preparation) and Group 2 (Feature Processing) exist |

### 0.2 Local legacy availability (critical)

Pipeline 70 / 90 / 100 and model binaries live under git-ignored `legacy/` on this workstation (`.git/info/exclude` contains `/legacy/`). **Do not treat `legacy/` as a production runtime dependency for CI or other clones.**

| Path | Role |
| --- | --- |
| `legacy/pipelines/70_dyadic_inference.py` | Dyadic binary detector inference |
| `legacy/pipelines/90_focal_inference.py` | Focal / asymmetric detector inference |
| `legacy/pipelines/100_behavioral_arbiter.py` | Downstream temporal consensus / arbitration (**S8 territory**) |
| `legacy/configs/dyadic_inference.yaml` | Active dyadic detectors + working thresholds |
| `legacy/configs/90_focal_inference.yaml` | Active focal detectors + working thresholds |
| `legacy/configs/behavioral_arbiter.yaml` | Arbiter / post-processing config |
| `legacy/models/dyadic/v1/*` | Dyadic detector bundles |
| `legacy/models/focal/v1/*` | Focal detector bundles |
| `legacy/src/io/path_utils.py` | Legacy artifact path templates |
| `legacy/src/dynamic_train/train_binary_detector.py` | Training path (out of S7 scope) |
| `legacy/src/dynamic_train/threshold_utils.py` | Training-time threshold helpers (not used by 70/90 at runtime) |

**Smoke check performed (read-only):** pointed S6 `build_model_compat_report` at `legacy/models` against the E-session `dyadic_features_full.parquet`. No `predict_proba` execution; no model loading beyond presence/spec parsing already done by S6.

---

## 1. Legacy dependency map

```text
S6 dyadic_features_full.parquet (directional; K*(K-1) rows/frame)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Pipeline 70 (dyadic)           Pipeline 90 (focal)
  • collapse to unordered pair   • keep both directions
  • joblib XGB predict_proba     • joblib XGB predict_proba
  • config working_thresholds    • config working_thresholds
  • contact intact/occ routing   • nose_to_body intact/occ routing
  • fused prob_contact           • optional rolling-median on
                                   nose_to_body probs (w=5)
        │                              │
        ▼                              ▼
 dyadic_predictions.parquet     focal_predictions.parquet
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Pipeline 100 (arbiter)  ← NOT S7
          • expand dyadic → directional
          • hard-gap gates, postural vetoes
          • hierarchical conflict resolution
          • bout stitching / hysteresis / ethogram
```

Inference entry points are self-contained scripts: model load + feature subset + `predict_proba` live **inside** Pipelines 70/90. There is no shared `src/inference/` helper today. Training helpers under `legacy/src/dynamic_train/` are not imported by 70/90 at inference time.

---

## 2. Dyadic inference audit (Pipeline 70)

### 2.1 Active detectors (config)

From `legacy/configs/dyadic_inference.yaml`:

| Config name | Runtime models | Notes |
| --- | --- | --- |
| `contact` | expands → `contact_intact` + `contact_occluded` | Logical detector; fused `prob_contact` / `pred_contact` |
| `nose_to_nose` | `nose_to_nose` | Symmetric semantics |
| `mutually_disengaged` | `mutually_disengaged` | Symmetric semantics |
| `side_by_side` | `side_by_side` | Symmetric semantics |

Obsolete standalone `contact` bundle still exists on disk but is **unsupported** under S6 contracts (`OBSOLETE_DYADIC_DETECTORS`). Pipeline 70 still understands the logical name `contact` as an expansion alias.

### 2.2 Model format and loading

| Item | Contract |
| --- | --- |
| Path | `models/dyadic/{model_iteration}/{detector}/` |
| Binary | `model.joblib` via `joblib.load` |
| Spec | `feature_spec.json` → `feature_cols` (ordered list) |
| Family | XGBoost classifiers (sklearn API); `xgboost` imported for unpickle compatibility |
| API | `model.predict_proba(X)` → positive class = column index `1` |

### 2.3 Feature order, missing columns, NaNs

1. Exact column order from `feature_spec.feature_cols` (also mirrored as `selected_feature_cols`).
2. Missing required columns → **hard error**.
3. Missing blob helpers `cc_count` / `min_inter_contour_dist` → warn and fill NaN (defensive; S6 already publishes them).
4. No explicit NaN/Inf row drop, imputation, or clip before `predict_proba`. NaNs are passed through; XGBoost handles missing values natively.
5. No dtype coercion beyond pandas subsetting.

### 2.4 Thresholds

| Source | Role in Pipeline 70 |
| --- | --- |
| `configs/dyadic_inference.yaml` → `working_thresholds` | **Authoritative at runtime** |
| Bundle `thresholds.json` / `train_meta.json` | **Not read** by Pipeline 70 |
| Default if detector absent from config | `0.5` |

Production YAML values (and corresponding bundle `threshold_f1` where available):

| Detector | Working threshold used by 70 |
| --- | --- |
| `contact_intact` | `0.9731` (from nested `contact.intact`) |
| `contact_occluded` | `0.4295` (from nested `contact.occluded`) |
| `nose_to_nose` | `0.0816` |
| `mutually_disengaged` | `0.0469` |
| `side_by_side` | `0.0095` |

**Important mismatch:** every dyadic `thresholds.json` stores `threshold_working: 0.2` with policy `manual_working`, while the inference YAML uses the F1-optimal (or specialist) values above. S7 must treat **run config / registry threshold policy** as source of truth and not silently trust bundle `threshold_working`.

Binarization: `pred = 1` iff `prob_positive >= threshold`.

### 2.5 Directionality, collapse, and double-counting

S6 emits **both** directional rows per unordered pair (`focal_id ≠ partner_id`). For two animals that is exactly `2 × n_frames`.

Pipeline 70 **collapses** before scoring:

1. Sort by `(exp_name, dense_frame_idx, focal_id, partner_id)` (stable).
2. Derive `mouse_a = min(focal, partner)`, `mouse_b = max(...)`.
3. `drop_duplicates(..., keep='first')` on `(exp_name, dense_frame_idx, mouse_a, mouse_b)`.

**Verdict on double-counting:** evaluating **both** directional rows for undirected dyadic detectors **would double-count** the same interaction at each frame if results were later treated as independent positives or Bout-level events. Collapse is therefore required for undirected dyadic semantics.

**Caveat:** collapse keeps the **first** directional feature vector after sort (typically the lower `focal_id` direction). Features remain directionally labeled (`*_i` / `*_j`). This is **not** an average of both directions and is **not** a guarantee of feature symmetry. Pair-level predictions are stored with the surviving `focal_id`/`partner_id` from that kept row; Pipeline 100 later re-broadcasts them to both directions via undirected pair keys.

### 2.6 Contact dual-model routing (still inside inference)

| Model | Scored when |
| --- | --- |
| `contact_intact` | `is_total_occlusion == 0` (else prob forced `0`) |
| `contact_occluded` | `is_total_occlusion == 1` (else prob forced `0`) |

Then fused outputs:

- Start from intact probs/preds as `prob_contact` / `pred_contact`.
- Where occluded, overwrite with occluded specialist outputs.

This fusion is **per-model routing**, not cross-behavior arbitration. It is reasonable to keep in S7 as a dyadic-family adapter, provided raw specialist columns remain available for S8.

### 2.7 Temporal operations

**None** in Pipeline 70 beyond upstream features already containing rolling/shift columns. No probability smoothing in 70.

### 2.8 Output schema (legacy)

Path: `data/inference/dyadic/{model_iteration}/{exp_name}/dyadic_predictions.parquet`

| Columns | Content |
| --- | --- |
| Keys | `exp_name`, `focal_id`, `partner_id`, `dense_frame_idx` |
| Per detector | `prob_{name}`, `pred_{name}` |
| Contact extras | `prob_contact_intact`, `pred_contact_intact`, `prob_contact_occluded`, `pred_contact_occluded`, fused `prob_contact`, `pred_contact` |
| Grain | One row per `(frame, unordered pair)` after collapse |
| Sort | `(exp_name, dense_frame_idx, focal_id, partner_id)` |

No `sample_idx`, `time_sec`, model version, or threshold columns are written today.

---

## 3. Focal inference audit (Pipeline 90)

### 3.1 Active detectors (config)

From `legacy/configs/90_focal_inference.yaml`:

| Config name | Runtime models | Notes |
| --- | --- | --- |
| `nose_to_body` | prefers `nose_body_intact` + `nose_body_occ` when both exist; else single `nose_to_body` | Two-stage routing on `is_total_occlusion` |
| `nose_to_tail` | `nose_to_tail` | Directional |
| `approach` | `approach` | Directional |
| `retract` | `retract` | Directional |

Also present on disk but **not** in the active YAML list:

| Bundle | Status |
| --- | --- |
| `nose_to_body` (standalone joblib) | Fallback / legacy single-stage |
| `nose_body`, `nose_tail` | Older two-stage JSON expert specs (`architecture: two_stage_expert`); no `model.joblib`; not used by Pipeline 90’s active path |

### 3.2 Focal / partner identity

- Requires metadata: `exp_name`, `focal_id`, `partner_id`, `dense_frame_idx` (no NaNs allowed in these keys).
- **No dyadic collapse.** Both `i→j` and `j→i` rows are scored independently.
- Scientific meaning: approach / retract / nose-to-body / nose-to-tail are **asymmetric**. Keeping both directions is correct and is **not** double-counting.

Pipeline 90 asserts a soft expectation of ~`2 × n_unique_frames` rows for two animals and only prints a note if that equality fails.

### 3.3 Model loading and preprocessing

Same joblib + ordered `feature_cols` pattern as dyadic. Missing columns → hard error. Blob helper NaN fill identical to 70.

`nose_to_body` two-stage:

1. Score intact rows with `nose_body_intact`; occluded rows with `nose_body_occ`.
2. Write intermediate specialist preds (`pred_nose_body_intact` / `pred_nose_body_occ`) when those masks fire.
3. **Temporal operation:** within each `(exp_name, focal_id, partner_id)`, apply `rolling(window=5, min_periods=1).median()` to `prob_nose_to_body`, then re-binarize with stage-specific thresholds.

No other detector is smoothed in Pipeline 90.

### 3.4 Thresholds

| Source | Role |
| --- | --- |
| `configs/90_focal_inference.yaml` → `working_thresholds` | **Authoritative** |
| Bundle `thresholds.json` | **Absent** for all focal joblib bundles inspected |
| Bundle `metrics.json` → `threshold_f1` | Training record only; not read by 90 |
| Default | `0.5` via `_resolve_threshold` |

YAML values:

| Detector | Threshold |
| --- | --- |
| `nose_to_body` intact | `0.6141` |
| `nose_to_body` occluded | `0.4825` |
| `nose_to_tail` | `0.6636` |
| `approach` | `0.3766` |
| `retract` | `0.4892` |

These match focal `metrics.json` `threshold_f1` values where present.

### 3.5 Output schema (legacy)

Path: `data/inference/focal/{model_iteration}/{exp_name}/focal_predictions.parquet`

| Columns | Content |
| --- | --- |
| Keys | `exp_name`, `focal_id`, `partner_id`, `dense_frame_idx` |
| Optional | `is_total_occlusion` (copied when present) |
| Per detector | `prob_{name}`, `pred_{name}` |
| Nose-to-body extras | specialist preds + stitched `prob_nose_to_body` / `pred_nose_to_body` |
| Grain | One row per directional `(frame, focal, partner)` |

### 3.6 Shared vs family-specific logic

| Shared (true common core) | Family-specific |
| --- | --- |
| Registry discovery + bundle load (`joblib` + `feature_spec.json`) | Dyadic unordered-pair collapse |
| Ordered feature matrix assembly | Focal directional retention |
| Missing-column validation | Contact / nose-to-body occlusion routers |
| `predict_proba`[:, 1] extraction | Fused logical detector columns (`contact`, stitched `nose_to_body`) |
| Threshold resolution + binarization | Focal `nose_to_body` rolling-median (legacy; migrate decision) |
| Blob helper NaN fill | Arbiter-side vetoes / conflict resolution (**S8**, not S7) |
| Per-detector progress / QC summaries | Threshold nesting (`intact`/`occluded`) schemas |

---

## 4. Model registry and versions

### 4.1 Inventory (local `legacy/models`)

#### Dyadic `v1` (all have `model.joblib` + `feature_spec.json` + `thresholds.json` + `train_meta.json` + `metrics.json`)

| Detector | `#features` | `model_family` | Active? |
| --- | --- | --- | --- |
| `contact_intact` | 139 | xgboost | Yes (via `contact`) |
| `contact_occluded` | 34 | xgboost | Yes (via `contact`) |
| `nose_to_nose` | 44 | xgboost | Yes |
| `side_by_side` | 26 | xgboost | Yes |
| `mutually_disengaged` | 79 | xgboost | Yes |
| `contact` | 84 | xgboost | **Obsolete** (requires `nose_any_refined`, unsuffixed `nose_any_imputed`) |

#### Focal `v1`

| Detector | `#features` | Binary | Active? |
| --- | --- | --- | --- |
| `approach` | 71 | `model.joblib` | Yes |
| `retract` | 39 | `model.joblib` | Yes |
| `nose_to_tail` | 133 | `model.joblib` | Yes |
| `nose_body_intact` | 109 | `model.joblib` | Yes (via `nose_to_body`) |
| `nose_body_occ` | 26 | `model.joblib` | Yes (via `nose_to_body`) |
| `nose_to_body` | 49 | `model.joblib` | Fallback / legacy single-stage |
| `nose_body` | open/occ JSON experts | no joblib | Legacy alternate architecture |
| `nose_tail` | open/occ JSON experts | no joblib | Legacy alternate architecture |

### 4.2 Serialization / library dependencies

| Need | Status in suite package |
| --- | --- |
| `joblib` | **Not** declared in `pyproject.toml` |
| `xgboost` | **Not** declared |
| `scikit-learn` | **Not** declared (often required transitively for sklearn-API XGB wrappers) |
| `pandas` / `pyarrow` / `numpy` | Present |

S7 must add an explicit optional extra (e.g. `s7` / `inference`) with pinned compatible versions. Bundle metadata currently lacks reliable `xgboost_version` / `sklearn_version` fields in the inspected `train_meta.json` files — another registry gap.

### 4.3 Bundled preprocessing

- Feature engineering is **entirely upstream** (S6 / Pipeline 50 / FF). Bundles do not embed scalers or imputers beyond what the pickled estimator contains.
- Inference-time extras: occlusion masks; focal `nose_to_body` rolling median.

### 4.4 Proposed tracked registry contract (no automatic binary copy)

Do **not** commit large `model.joblib` binaries by default. Track a **registry manifest** instead:

```text
configs/subsystem_07/model_registry.example.yaml
# or docs + schema only in MVP

model_registry:
  root: null   # operator-configured absolute/lab path
  dyadic_subdir: dyadic/v1
  focal_subdir: focal/v1
  include_obsolete: false

detectors:   # optional explicit allow-list / versions
  - family: dyadic
    name: contact_intact
    version: v1
    required_files: [model.joblib, feature_spec.json]
    threshold_policy: config_or_bundle_f1
```

Recommended tracked artifacts (small, reviewable):

| Tracked | Purpose |
| --- | --- |
| `feature_spec.json` copies or hashes | Column contracts |
| Threshold policy YAML | Runtime decisions |
| Manifest of detector names / versions / SHA256 of binaries | Reproducibility without binaries |
| Optional `metrics.json` excerpts | Lab documentation |

Untracked / lab-local:

| Local-only | Purpose |
| --- | --- |
| `model.joblib` / booster JSON | Weight binaries |
| `val_predictions.parquet` | Training diagnostics |

Reuse and extend S6’s `ModelRegistrySettings` (`root`, `dyadic_subdir`, `focal_subdir`, `include_focal`, `include_obsolete`) rather than inventing a second root key. S7 adds: threshold resolution, model load validation, active-detector selection, and binary digest recording into run meta.

---

## 5. S6 handoff

### 5.1 Published artifact

| Item | Contract |
| --- | --- |
| File | `{session}/combined_features/combined_features__*/dyadic_features_full.parquet` |
| Schema | `dyadic_features_full_v1` |
| Grain | Directional: one row per `(focal_id, partner_id, dense_frame_idx)` |
| Identity | `schema_version`, `session_root`, `focal_id`, `partner_id`, `sample_idx`, `dense_frame_idx`, `time_sec`, `orig_frame_idx`, `source_s4_run_id`, `source_s5_run_id` |
| E-session check | 73 056 rows = 36 528 frames × 2 directions; 483 columns; `is_total_occlusion` present |

### 5.2 Discovery / eligibility APIs (reuse)

| API | Module |
| --- | --- |
| `summarize_combined_features_project(session_root)` | `combined_features.run_discovery` |
| `find_latest_eligible_s6_run(session_root)` | same |
| `is_s6_run_eligible_for_downstream(run_dir)` | `combined_features.runner` |
| `validate_s6_run_for_downstream(run_dir)` | same |

Eligibility re-validates completed status, parquet presence, and recorded S4/S5/sync digests against the live joint-eligible pair. Stale runs must never be selected.

### 5.3 `model_compat.json`

S6 writes a **presence / column / dtype** report only. It:

- does **not** load estimators or call `predict_proba`;
- does **not** resolve working thresholds;
- does **not** verify routing columns beyond ordinary feature presence;
- may be `registry_status=unavailable` when `model_registry.root` is unset (common; E-session artifact is in this state).

**Read-only smoke (2026-10-26):** with `model_registry.root=legacy/models`, all active dyadic/focal joblib detectors reported `status=compatible` and `missing_columns=[]` against the E-session feature table. Obsolete `contact` correctly `unsupported`.

### 5.4 Is S6 compatibility enough to execute models?

**No.** Sufficient to know “columns exist for these specs,” insufficient for execution. S7 still needs:

1. Re-run (or cache) registry discovery against the **operator’s current** registry root.
2. Confirm `model.joblib` (or declared booster) loads under the declared library versions.
3. Confirm `predict_proba` shape `(n, 2)`.
4. Resolve thresholds (config > policy > bundle F1 > hard fail).
5. Confirm routing columns (`is_total_occlusion`) when dual-model detectors are selected.
6. Confirm S6 run still `eligible` at execution time (staleness).
7. Optionally re-check NaN/Inf fractions as warnings, not structural blockers (S6 policy already treats scientific NaNs as non-blocking).

---

## 6. Unified prediction schema (proposal)

Preserve **distinct grains**. Do not invent a fake direction for undirected dyadic results.

### 6.1 Canonical long-form table: `predictions.parquet`

One row per `(sample, entity, detector)`:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | str | e.g. `behavioral_predictions_v1` |
| `sample_idx` | int64 | S6 sample index |
| `dense_frame_idx` | int64 | Same grid as S6 |
| `time_sec` | float64 | Shared 120 Hz clock |
| `prediction_grain` | enum | `undirected_pair` \| `directional` |
| `mouse_a` / `mouse_b` | int64 nullable | Undirected pair IDs (`a < b`); required when grain=`undirected_pair` |
| `focal_id` / `partner_id` | int64 nullable | Directional IDs; required when grain=`directional`; for undirected rows may store collapse survivors for provenance only |
| `family` | enum | `dyadic` \| `focal` |
| `detector_name` | str | e.g. `nose_to_nose`, `approach`, `contact` (logical) |
| `specialist_name` | str nullable | e.g. `contact_intact` when routed |
| `model_version` | str | e.g. `v1` |
| `model_bundle_dir` | str | Resolved path used |
| `model_sha256` | str nullable | Binary digest |
| `prob_positive` | float64 | Raw positive-class probability **before** S8 gates |
| `threshold` | float64 | Decision threshold applied for `pred` |
| `pred` | int8 | `0/1` from `prob >= threshold` (or 0 when masked invalid) |
| `input_valid` | bool | False when routing excluded the row / features unusable |
| `invalid_reason` | str nullable | e.g. `occlusion_mask_excluded`, `missing_features` |
| `source_s6_run_id` | str | Provenance |
| `session_root` | str | Provenance |

**Cardinality rules:**

| Family | Rows per frame (2 animals) |
| --- | --- |
| Each undirected dyadic detector | 1 (`prediction_grain=undirected_pair`) |
| Each focal detector | 2 (`prediction_grain=directional`) |
| Logical fused detectors (`contact`, stitched `nose_to_body`) | Same grain as their family; specialists retained as additional rows or columns |

### 6.2 Companion wide views (optional export)

For Pipeline 100 parity / debugging, S7 may also emit:

- `dyadic_predictions_wide.parquet` (collapsed keys + `prob_*` / `pred_*`)
- `focal_predictions_wide.parquet` (directional keys + `prob_*` / `pred_*`)

These are **views**, not a second scientific source of truth. The long-form table is canonical.

### 6.3 Invalid / unknown input treatment

| Case | Proposal |
| --- | --- |
| Missing required feature columns | Fail the detector (or whole run if configured strict) |
| NaN features | Allow; score; record QC NaN fractions; do not invent fills |
| Occlusion router excluded rows | `prob_positive=0`, `pred=0`, `input_valid=False`, reason set |
| Registry / load failure | Fail that detector; continue others only if multi-detector partial mode enabled |

### 6.4 Legacy migration decisions (explicit)

| Topic | Legacy behavior | Decision needed |
| --- | --- | --- |
| Dyadic collapse policy | `keep='first'` after sort | Keep for parity **or** replace with max-prob / feature-symmetric aggregation |
| `nose_to_body` rolling median | Inside Pipeline 90 | Keep as focal adapter flag **or** move to S8 (preferred for “raw framewise” purity) |
| Threshold authority | YAML overrides stale bundle `threshold_working=0.2` | Config/manifest policy wins; document bundle fields as training echoes |
| Fused `prob_contact` | Written by 70 | Keep as dyadic logical output **and** retain specialists |
| Wide vs long | Wide only | Adopt long-form canonical + optional wide views |
| `exp_name` key | Required legacy string | Prefer `session_root` + S6 run id; map `exp_name` only for legacy export |

---

## 7. S7 versus S8 boundary

### 7.1 Belongs in S7 (classifier inference)

- S6 run selection + eligibility validation
- Model registry configuration and bundle load
- Feature matrix assembly in training order
- Per-detector `predict_proba`
- Per-detector thresholding to binary preds
- Family adapters: dyadic collapse; occlusion specialists; optional documented focal probability smoother **if retained**
- Prediction artifacts, QC, run discovery, provenance
- Model-specific warnings/errors and readiness summaries

### 7.2 Belongs in S8 (post-processing / arbiter) — Pipeline 100 today

- Expanding undirected dyadic preds onto directional focal structure for joint ethograms
- Hard distance probability gates
- Postural approach/retract vetoes
- Hierarchical cross-behavior conflict resolution
- Adaptive re-thresholding
- Contact bout cleaning / optical assist post-processing
- Gap fill, min-duration filters, engaged/idle stitching
- Mutual reciprocity enforcement (`mutually_disengaged`, `nose_to_nose`)
- Final engaged hysteresis
- Overlay video / ethogram packaging

**Do not silently migrate cross-model arbitration into S7.**

### 7.3 Borderline item

`nose_to_body` 5-frame rolling median is temporal post-processing embedded in focal inference. Recommendation: S7 default emits **raw** specialist + stitched probs; expose legacy smoothing as an explicit adapter flag defaulting **off** for new runs (or on for parity mode). Final choice is an unresolved design decision (§10).

---

## 8. Proposed S7 backend architecture

```text
src/behavioral_inference/          # proposed package name
  contracts.py                     # schema versions, grains, statuses
  config.py                        # detectors, thresholds, registry, parity flags
  registry.py                      # extend S6 discovery; load + digest bundles
  thresholds.py                    # resolve working thresholds
  features.py                      # ordered matrix + validation
  adapters/
    dyadic.py                      # collapse + contact routing + fuse
    focal.py                       # directional pass-through + nose_to_body route
  predict.py                       # shared predict_proba / binarize
  export.py                        # long-form + optional wide views
  qc.py                            # counts, NaN fractions, positive rates
  runner.py                        # end-to-end S6 → predictions
  run_discovery.py                 # completed S7 runs + staleness vs S6
```

GUI:

```text
src/ui/controllers/behavioral_inference_controller.py
src/ui/pages/behavioral_inference_page.py
GuiTaskKind.BEHAVIORAL_INFERENCE
```

Main window: add **Group 3 — Behavioral Inference** hosting a single S7 page (dyadic + focal together). S8 remains future/disabled.

---

## 9. Proposed run artifacts and QC

```text
{session_root}/behavioral_inference/behavioral_inference__YYYYMMDDTHHMMSS/
  predictions.parquet              # canonical long-form
  dyadic_predictions_wide.parquet  # optional legacy-shaped view
  focal_predictions_wide.parquet   # optional legacy-shaped view
  run_meta.json                    # sources, registry, digests, library versions
  settings_used.yaml
  processing_log.txt
  qc.json                          # per-detector summaries + readiness
  model_plan.json                  # selected detectors, thresholds, routing
```

Suggested QC contents:

- Per detector: n_scored, n_positive, positive rate, mean/median prob, NaN feature fraction
- Grain checks: undirected uniqueness; directional `K*(K-1)` expectation
- Registry readiness vs selected set
- S6 eligibility snapshot at start and pre-publish
- Warnings for dual-model routing coverage and missing optional specialists

---

## 10. GUI proposal (Group 3 — one page)

**Page:** Subsystem 7 — Behavioral Inference  

**Must support:**

1. Session root / project context (same pattern as S6)
2. Eligible S6 run selection (`find_latest_eligible_s6_run` + manual eligible picker)
3. Source staleness / eligibility reporting (reuse S6 classifications)
4. Model registry root configuration (persist like S6)
5. Unified detector picker: dyadic + focal checklists on one screen
6. Feature/model readiness table (extend S6 compat + loadable + threshold resolved)
7. Threshold display / optional override (advanced; default from policy)
8. Run full inference (one task covering selected detectors)
9. Progress via `GuiTaskRunner` (new `GuiTaskKind`)
10. Model-specific warnings/errors panel
11. Prediction summaries (positives / rates / invalid counts)
12. Completed S7 run discovery and reopen

**Must not:** train models; run arbiter/S8; edit S1–S6 science.

---

## 11. Unresolved design decisions

1. **Dyadic collapse policy:** keep `keep='first'` for parity, or score both directions and aggregate (max / mean / symmetric-feature rebuild)?
2. **`nose_to_body` rolling median:** S7 parity flag vs move entirely to S8?
3. **Threshold source hierarchy** when YAML, `threshold_f1`, and `threshold_working` disagree (already true for dyadic bundles).
4. **Package binaries:** lab-local only vs optional tracked LFS / release assets?
5. **Partial multi-detector runs:** fail-fast vs continue-on-detector-error?
6. **Canonical table only vs also emit Pipeline 100-compatible wide files by default?**
7. **Library pinning:** which `xgboost` / `joblib` / `scikit-learn` versions match the existing pickles? (Requires a controlled load smoke in implementation sprint.)
8. **Logical detector naming:** publish fused `contact` / stitched `nose_to_body` as first-class detectors, specialists-only, or both?
9. **Group 3 navigation:** enable only after S7 MVP, or show disabled placeholder earlier?

---

## 12. Suggested implementation roadmap

| Sprint | Deliverable |
| --- | --- |
| **S7-A** | This architecture audit (done) |
| **S7-B** | Binding implementation contract: schemas, threshold policy, S7/S8 boundary, registry manifest format ([`implementation_contract.md`](implementation_contract.md), done) |
| **S7-1** | Package skeleton + registry/threshold/feature validation (no GUI); optional dependency extra |
| **S7-2** | Shared predictor + dyadic/focal adapters; write canonical predictions + QC; headless E-session parity tests vs legacy wide outputs |
| **S7-3** | Run discovery, staleness vs S6, model_plan, settings snapshots |
| **S7-4** | Group 3 GUI page + task integration + laboratory acceptance |
| **S8-A** | Separate audit for Pipeline 100 arbiter (consumes S7 artifacts only) |

Acceptance intent for S7 MVP: reproduce Pipeline 70/90 scientific outputs (within documented parity flags) from an eligible S6 run + configured registry, without performing S8 arbitration.

---

## 13. Files inspected (checklist)

**Legacy pipelines / configs**

- `legacy/pipelines/70_dyadic_inference.py`
- `legacy/pipelines/90_focal_inference.py`
- `legacy/pipelines/100_behavioral_arbiter.py` (boundary only)
- `legacy/configs/dyadic_inference.yaml`
- `legacy/configs/90_focal_inference.yaml`
- `legacy/configs/behavioral_arbiter.yaml`
- `legacy/src/io/path_utils.py` (artifact templates)

**Model bundles**

- All `legacy/models/dyadic/v1/*/feature_spec.json`, `thresholds.json`, `train_meta.json`, `metrics.json`
- All `legacy/models/focal/v1/*/feature_spec.json`, `metrics.json`, and present binaries / JSON experts

**Tracked suite**

- `src/combined_features/{contracts,config,model_compat,runner,run_discovery,export}.py`
- `configs/subsystem_06/combined_features_default.yaml`
- `src/ui/{main_window,tasks}.py`, `src/ui/pages/combined_features_page.py`
- `docs/subsystem_06/{architecture_audit,implementation_contract,sprint_3_export_compatibility,README path via docs/README}.md`
- `pyproject.toml` (dependency gap for xgboost/joblib)
- E-session `dyadic_features_full.parquet` + `model_compat.json`

**Read-only smoke**

- `combined_features.model_compat.build_model_compat_report` against `legacy/models` + E-session features (no prediction).
