# Subsystem 07 — Sprint 3: Prediction Artifacts, QC, and Run Discovery

**Status:** Implemented (headless). GUI delivered in Sprint 4; no S8 arbitration.  
**Date:** 2026-10-26  
**Authority:** Accepted S7-B implementation contract + Sprint 2 inference engine.

This sprint publishes validated Sprint 2 predictions as immutable run artifacts,
adds QC / model provenance / source staleness, and exposes S6-style run
discovery. Scientific inference formulas are unchanged.

---

## 1. Review-point resolutions (Sprint 2)

### Invalid contact routing

| Requirement | Resolution |
| --- | --- |
| Preserve `input_valid=False` for unhandled occlusion states | Unchanged in adapter; validation refuses to publish if violated |
| Distinguish legacy zero sentinel from observed model probability | Export-time `score_origin`: `routing_sentinel` vs `model_observed` vs `fusion_routed` (contact fused) |
| Validity not lost in wide views | Dyadic wide includes `input_valid_contact`, `score_origin_contact`, and specialist validity columns |

No formula changes — only export annotations + regressions.

### Invalid nose-to-body routing

| Requirement | Resolution |
| --- | --- |
| Preserve pre-smooth vs smoothed probability | Canonical `prob_raw` / `prob_effective`; wide `prob_nose_to_body_raw` / `prob_nose_to_body` |
| Legacy binary on invalid occlusion frames | `prediction=0`, `input_valid=False`, `invalid_reason=non_matching_occlusion_state` |
| No blind re-thresholding of invalid scores | Wide `input_valid_nose_to_body` + `score_origin_nose_to_body=routing_sentinel`; S8 must gate on validity |

Smoothing may pull a non-zero **effective** probability onto an invalid frame from
neighbors; that value is **not** an observed model score for that frame.

---

## 2. Run layout

```text
{session_root}/behavioral_inference/behavioral_inference__YYYYMMDDTHHMMSS/
  predictions.parquet
  specialist_predictions.parquet
  dyadic_predictions_wide.parquet
  focal_predictions_wide.parquet
  run_meta.json
  settings_used.yaml
  model_manifest.json
  qc.json
  processing_log.txt
```

Atomic publication mirrors S5/S6: science parquets via `.tmp` + `replace`; on
failure, science artifacts are unlinked and `status=failed` is persisted.
Partial results are never marked completed / eligible.

---

## 3. Published artifact schemas

### `predictions.parquet` (canonical)

Sprint 2 long-form columns plus:

- `score_origin` — `model_observed` \| `routing_sentinel` \| `fusion_routed`
- `source_s6_run_id`, `session_root`

Schema stamp: `behavioral_predictions_v1`.

### `specialist_predictions.parquet`

Same column contract as canonical, containing specialist / diagnostic rows
(contact specialists, nose-to-body specialists). May be empty (typed schema)
when no dual-model detectors were selected.

### Wide views

**Dyadic** (`dyadic_predictions_wide_v1`): one row per
`(dense_frame_idx, mouse_a, mouse_b)`. Collapse survivors set
`focal_id=mouse_a`, `partner_id=mouse_b` for Pipeline 100 pair expansion.
Columns: `prob_<det>`, `pred_<det>`, `input_valid_<det>`, `score_origin_<det>`,
plus contact specialist `prob_/pred_/input_valid_contact_*`.

**Focal** (`focal_predictions_wide_v1`): one row per
`(dense_frame_idx, focal_id, partner_id)`. Includes
`prob_nose_to_body` (smoothed effective), `prob_nose_to_body_raw`, validity,
and specialist nose-body columns when selected.

---

## 4. Canonical → wide transformation

1. Validate canonical keys / cardinality / grains / validity.
2. Annotate `score_origin` (export-only).
3. Pivot primary/fused rows into `prob_*` / `pred_*` columns.
4. Join specialist diagnostics when present.
5. Assert every wide key is traceable to canonical rows; refuse duplicate keys.
6. Never fabricate a second undirected direction for dyadic detectors.

---

## 5. QC and model manifest

### `qc.json` (`behavioral_inference_qc_v1`)

Structural failures block publication. Metrics include sample/frame counts,
per-detector prediction counts and positive fractions, invalid-route counts,
NaN/non-finite probability counts, thresholds, routing diagnostics, source
provenance, and detector coverage (explicit subsets for S8).

Ordinary warnings → `completed_with_warnings` and **remain eligible**.

### `model_manifest.json` (`behavioral_inference_model_manifest_v1`)

Selected detectors, specialist bundles, resolved paths, model SHA-256,
feature-spec digests, library versions, effective thresholds + sources,
smoothing / collapse policy, settings echo. **Does not copy model binaries.**

---

## 6. Source / model staleness

| Event | Historical `status` | Downstream eligibility |
| --- | --- | --- |
| S6 digest / lineage becomes ineligible | unchanged | `false` → classification `stale_source` |
| Model `model.joblib` missing or digest mismatch | unchanged | `false` → `stale_models` |
| Feature-spec digest mismatch (when recorded) | unchanged | `stale_models` |
| Registry root moved but recorded bundle paths still hold **identical digests** | unchanged | still eligible (digests authoritative) |
| Different model version substituted (digest change) | unchanged | stale; never silently accepted |

Immutable `settings_used.yaml` is the run’s config snapshot; external default
YAML drift does not stale a completed run.

Pre-publish: revalidate S6 eligibility + feature SHA and model digests; fail
before writing science artifacts if either drifted mid-run.

---

## 7. Run discovery and eligibility

APIs (analogous to S6):

- `summarize_behavioral_inference_project`
- `validate_s7_run_for_downstream` / `is_s7_run_eligible_for_downstream`
- `find_latest_eligible_s7_run`

Classifications:

`eligible` \| `completed_with_warnings` \| `stale_source` \| `stale_models` \|
`failed_or_incomplete` \| `missing_required_artifacts` \| `invalid_artifact`

Detector subsets are allowed; summaries expose `detectors_dyadic` /
`detectors_focal` so S8 can require specific coverage.

---

## 8. Headless runner

`run_behavioral_inference_publication(BehavioralInferenceRequest)`:

1. Select / validate eligible S6  
2. Freeze settings into run dir (`status=running`)  
3. Load only required trusted models  
4. Execute Sprint 2 `run_behavioral_inference` (in memory)  
5. Validate + wide views + QC + manifest  
6. Revalidate S6 / models  
7. Atomic publish  
8. Record `completed` / `completed_with_warnings` / `failed`

---

## 9. Exact S8 input requirements

S8 (legacy Pipeline 100 recovery) must be able to:

1. Load **focal wide** as the directional spine.
2. Expand **dyadic wide** onto both directions via `(min,max)` pair keys without
   duplicate undirected rows.
3. Consume fused `prob_contact` / `pred_contact` and preferably specialist
   contact columns.
4. Consume smoothed `prob_nose_to_body` / `pred_nose_to_body` under parity.
5. **Gate on `input_valid_*` / `score_origin_*`** — do not re-threshold invalid
   routing sentinels or smoothed effective values on invalid nose-to-body frames.
6. Respect detector coverage in run meta / QC before assuming a full detector set.
7. **Not** expect S7 to perform hierarchical conflict resolution, reciprocity,
   hard-gap gates, optical-assist fusion, or bout stitching.

---

## 10. Unresolved migration issues

1. Exact lab `xgboost` / `sklearn` / `joblib` pin confirmation on every workstation.  
2. Product default for new runs: `legacy_parity` vs `raw_framewise` smoother.  
3. Whether S8 should prefer long-form canonical over wide views long-term.  
4. Multi-animal K>2 cardinality validation beyond MVP.  
5. Binary distribution policy (still lab-local).  
6. Tolerance policy if future library pins change float parity vs Pipeline 70/90.

---

## 11. Sprint 4 prerequisites

1. Headless artifacts + discovery accepted in architectural review.  
2. GUI extra (`PySide6`) available.  
3. Group 3 page: S6 picker, registry root, detector selection, thresholds,
   execute task, QC reopen.  
4. Lab registry + eligible S6 session for operator acceptance.  
5. No S8 implementation in Sprint 4.

---

## 12. Automated coverage

| Area | Tests |
| --- | --- |
| Invalid-route sentinels / wide validity | `test_invalid_route_export.py` |
| Canonical ↔ wide / subset / uniqueness | `test_export_wide_views.py` |
| Publication / staleness / discovery / determinism | `test_runner_discovery.py` |
| Sprint 1–2 retained | existing `tests/behavioral_inference/*` |
