# Subsystem 07 — Sprint 2: Unified Dyadic and Focal Inference

**Status:** Implemented (headless; no GUI / no Parquet publication).  
**Date:** 2026-10-26  
**Authority:** Accepted S7-A audit + S7-B implementation contract + Sprint 1 runtime.

This sprint delivers the scientific inference path that reproduces Pipeline 70
(dyadic) and Pipeline 90 (focal) framewise outputs as an in-memory result.
Artifact publication, run discovery, and GUI remain Sprint 3 / 4.

---

## 1. Shared versus family-specific modules

```text
src/behavioral_inference/
  predict.py                 # shared: feature matrix → positive proba → binarize
  engine.py                  # shared entry: run_behavioral_inference(...)
  results.py                 # in-memory result + long-form column contract
  adapters/
    dyadic.py                # unordered-pair collapse + contact routing
    focal.py                 # directional pass-through + nose_to_body route/smooth
  # Sprint 1 retained:
  registry.py / loader.py / features.py / config.py / contracts.py
```

| Concern | Module |
| --- | --- |
| Ordered feature validation, Inf reject, NaN keep | `features.py` + `predict.py` |
| Trusted load + positive-class index | `loader.py` |
| Threshold precedence | `config.py` |
| Dyadic collapse + contact fusion | `adapters/dyadic.py` |
| Focal directions + nose-to-body smoother | `adapters/focal.py` |
| Detector selection / fail-fast orchestration | `engine.py` |

**Shared entry point:** `behavioral_inference.run_behavioral_inference`.

---

## 2. Detector inventory

### Dyadic (logical config names)

| Logical | Bundles | Grain |
| --- | --- | --- |
| `contact` | `contact_intact` + `contact_occluded` → fused `contact` | undirected pair |
| `nose_to_nose` | `nose_to_nose` | undirected pair |
| `mutually_disengaged` | `mutually_disengaged` | undirected pair |
| `side_by_side` | `side_by_side` | undirected pair |

Obsolete standalone `contact` joblib remains unsupported.

### Focal

| Logical | Bundles | Grain |
| --- | --- | --- |
| `nose_to_body` | prefers `nose_body_intact` + `nose_body_occ`; optional single-stage `nose_to_body` | directional |
| `nose_to_tail` | `nose_to_tail` | directional |
| `approach` | `approach` | directional |
| `retract` | `retract` | directional |

---

## 3. Canonical dyadic pair policy

For each `(exp_name, dense_frame_idx, mouse_a, mouse_b)`:

1. Stable-sort by `(exp_name, dense_frame_idx, focal_id, partner_id)`.
2. `mouse_a = min(focal_id, partner_id)`, `mouse_b = max(...)`.
3. Keep the unique row where `focal_id == mouse_a` and `partner_id == mouse_b`.
4. Score **only** that directional feature vector — no averaging / max fusion.

S7 MVP requires exactly two tracked mice (`adapters.require_exactly_two_mice`,
default true). Cardinality for K=2: **N_frames** undirected rows per dyadic
detector.

The classifier itself is **not** claimed symmetric; pair-level publication is a
policy on which directional vector is scored.

---

## 4. Specialist routing

### Contact (Pipeline 70)

- Gate: `is_total_occlusion` with **NaN→0** coercion (`to_numeric → fillna(0) → int`).
- Intact specialist when value `== 0`; occluded when `== 1`.
- Excluded rows: specialist `prob=0`, then threshold.
- Fuse: copy intact → overwrite where occluded.
- States neither 0 nor 1 after coercion: no invented prediction; fused stays 0 /
  `input_valid=False`.
- Missing occlusion column or missing specialist → **hard fail**.

### Nose-to-body (Pipeline 90)

- Gate: exact `.eq(0)` / `.eq(1)` **without** fillna.
- Stitch specialist probs into fused raw; non-{0,1} (incl. NaN) remain 0.
- Specialist binary preds from **raw** probs before smoothing.
- Fused binary preds from **smoothed** probs with stage thresholds.
- Single-stage fallback only when specialists incomplete **and**
  `allow_nose_to_body_single_stage_fallback=true` (no smoothing).

---

## 5. Focal smoothing

Legacy five-frame median (MVP default under `parity_mode=legacy_parity`):

| Parameter | Value |
| --- | --- |
| Window | 5 |
| `min_periods` | 1 |
| Grouping | `(exp_name, focal_id, partner_id)` |
| Sort before transform | `(exp_name, focal_id, partner_id, dense_frame_idx)` |
| Target | fused `prob_nose_to_body` only |

`parity_mode=raw_framewise` forces the smoother off. Leading frames use partial
windows. All-NaN windows remain NaN (no pre-fill with 0).

---

## 6. Threshold and smoothing semantics

**Precedence (binding):**

1. Explicit user override  
2. Inference YAML `working_thresholds`  
3. Hard fail  

Never auto-apply `0.5` or bundle `threshold_working` (unless
`allow_default_threshold_0_5=true` for documented legacy experiments).

Each detector/specialist records `(value, source)` in result metadata.
Selected detector subsets do not alter probabilities of unrelated selected
detectors.

---

## 7. Prediction and diagnostics schemas

In-memory `BehavioralInferenceResult`:

| Field | Content |
| --- | --- |
| `predictions` | Canonical long-form final outputs |
| `diagnostics` | Specialist / intermediate rows |
| `selected_detectors_*` | Requested logical detectors |
| `detector_meta` | Per-detector identities, digests, thresholds, routing |
| `thresholds` / `model_identities` | Convenience maps |
| `warnings` / `routing_diagnostics` | Soft notes + routing coverage |
| `metadata` | Cardinality, adapter flags, library versions |

Long-form columns include:

`dense_frame_idx`, `time_sec`, `detector_name`, `prediction_grain`, `pair_id`,
`mouse_a`, `mouse_b`, `focal_id`, `partner_id`, `prob_raw`, `prob_effective`,
`prediction`, `threshold`, `threshold_source`, `input_valid`, plus family /
`specialist_name` / `output_role` / model provenance.

Rules:

- Undirected rows: `pair_id` / `mouse_a` / `mouse_b` set; **no fabricated**
  `focal_id` / `partner_id`.
- Directional rows: focal/partner set; pair fields null.
- Fused `contact`: `prob_raw` is null (multi-classifier fusion);
  `prob_effective` holds the routed probability.
- Fused `nose_to_body`: `prob_raw` = pre-smooth stitch; `prob_effective` =
  post-smooth (or equal when smoothing off).

**No Parquet publication in this sprint.**

### Cardinality (K=2)

| Output | Rows / frame |
| --- | --- |
| Each undirected dyadic primary / fused | 1 |
| Each contact specialist (diagnostics) | 1 |
| Each focal primary / fused | 2 |
| Each nose-to-body specialist (diagnostics) | 2 |

---

## 8. Parity results

Focused parity tests (`tests/behavioral_inference/test_parity_legacy.py`,
marker `s7_model_smoke`) compare S7 against an independent Pipeline 70/90
formula mirror on the same S6 snippet, models, and thresholds:

| Check | Tolerance |
| --- | --- |
| Probabilities | `rtol=1e-6`, `atol=1e-8` |
| Binary predictions | Exact |
| Canonical pair / directional keys | Exact |

Covered: `nose_to_nose`, fused `contact` routing, two-stage `nose_to_body`
(with rolling median), and `approach` directional keys.

**Blocker note:** Full Pipeline 70/90 script re-execution against the E-session
is not required for Sprint 2 exit. Legacy scripts write under
`data/inference/...`, which is not present as a frozen gold artifact in this
repo. The formula-mirror tests use the same lab `model.joblib` files and the
E-session `dyadic_features_full.parquet` snippet when available; environments
without `models/` or `[s7]` skip those tests.

---

## 9. Differences from legacy

| Topic | Legacy | S7 Sprint 2 |
| --- | --- | --- |
| Missing threshold | Default `0.5` | Hard fail (opt-in `0.5` flag) |
| Inf features | Passed through | Rejected (Sprint 1 hardening) |
| Nose-to-body specialist probs | Not stored | Published in diagnostics |
| Pre-smooth fused nose-to-body | Overwritten in place | Stored as `prob_raw` |
| Output shape | Wide parquet per family | Long-form in-memory (+ diagnostics) |
| Artifact publish | Always wrote parquet | Deferred to Sprint 3 |
| Failures | Per-script abort | Fail-fast; selected detector never silently omitted |
| K≠2 | Soft note (focal) | Hard fail by default for MVP |

Family-specific occlusion NaN handling is **preserved** (contact fillna vs
nose-to-body `.eq`).

---

## 10. Sprint 3 prerequisites

1. Publish canonical `predictions.parquet` + optional wide views for S8.
2. Write `run_meta.json`, `settings_used.yaml`, `qc.json`, `model_manifest.json`,
   `processing_log.txt`.
3. Run discovery + eligibility/staleness vs S6 digests and model SHA-256.
4. Pre-publish eligibility re-check; fail before publish if S6 goes stale.
5. Keep fail-fast default; document any continue-on-error advanced mode.
6. Do **not** implement S8 arbitration, GUI, or training.

---

## 11. Automated coverage

| Area | Tests |
| --- | --- |
| Collapse / row-order invariance | `test_dyadic_collapse.py` |
| Contact routing / NaN / invalid occlusion | `test_contact_routing.py` |
| Focal directions / smoother / fallback | `test_focal_inference.py` |
| Engine selection / Inf / NaN / schema / determinism | `test_engine.py` |
| Lab formula parity | `test_parity_legacy.py` (`s7_model_smoke`) |
| Sprint 1 retained | registry / loader / features / thresholds |
