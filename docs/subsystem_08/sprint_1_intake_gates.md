# Subsystem 08 — Sprint 1: Source Intake, Directional Expansion and Initial Gates

**Status:** Implemented (headless; no hierarchy / publish / GUI).  
**Date:** 2026-10-26  
**Authority:** Accepted S8-A audit + S8-B implementation contract (MVP policy §0.1–0.2).

This sprint validates linked S7/S6 inputs, expands undirected dyadic predictions to directional rows, normalizes S7 validity evidence, merges required S6 geometric features, and applies Pipeline 100 hard-distance gates plus postural approach/retract vetoes. It does **not** implement the remaining 21-stage arbiter, bout tables, Parquet publish, or GUI.

---

## 1. Modules and APIs

```text
src/behavioral_arbitration/
  __init__.py       # public Sprint 1 API
  contracts.py      # schema stamps, detector sets, evidence states, errors
  config.py         # scientific vs execution settings
  intake.py         # joint S7/S6 eligibility, lineage, readiness
  expand.py         # Stage 1 dyadic → directional expansion
  validity.py       # evidence normalization (not a Pipeline 100 stage)
  gates.py          # Stage 2 merge + Stage 3–4 hard-distance / postural vetoes
  results.py        # Sprint1ArbitrationResult + StageDiagnostic
  runner.py         # headless orchestration

configs/subsystem_08/
  behavioral_arbitration_default.yaml
```

Primary entry points:

| API | Role |
| --- | --- |
| `validate_joint_s7_s6_for_s8(...)` | Joint intake / S8 readiness |
| `expand_dyadic_to_directional(...)` | Stage 1 |
| `normalize_arbitration_evidence(...)` | Validity masking |
| `merge_s6_features(...)` | Stage 2 |
| `apply_hard_distance_probability_gates(...)` | Stage 3 |
| `apply_postural_approach_veto(...)` / `apply_postural_retract_veto(...)` | Stage 4 |
| `run_behavioral_arbitration_sprint1(request)` | Full Sprint 1 pipeline → in-memory result |

---

## 2. Intake and validation requirements

Joint checks (hard fail unless noted):

1. Select eligible S7 (`completed` / `completed_with_warnings` + downstream eligibility).
2. Resolve the **exact** S6 run recorded in S7 `source_s6` — never substitute newest S6.
3. Revalidate S7 and linked S6 for downstream eligibility.
4. Verify S7 lineage digests against recorded S6 features.
5. Confirm matching 120 Hz `dense_frame_idx` timeline (`sample_idx == dense_frame_idx`).
6. Verify undirected / directional key uniqueness.
7. Require exactly two mice.
8. Require full Pipeline-100 detector set on wide views.
9. Require structural hard-gap / optical S6 columns.
10. Reject stale, mismatched, missing, or structurally invalid inputs.

**S7 eligibility ≠ S8 readiness:** overall S7 eligibility can hold with a detector subset; S8 full-ethogram profile still hard-fails missing detectors or validity columns.

---

## 3. Detector coverage policy

Required dyadic: `contact`, `nose_to_nose`, `mutually_disengaged`, `side_by_side`.  
Required focal: `nose_to_body`, `nose_to_tail`, `approach`, `retract`.

Partial arbitration is out of MVP (policy P1 / conflict C5).

---

## 4. Directional expansion

Matches `_expand_dyadic_to_directional`:

- Undirected join key `(exp_name, dense_frame_idx, (min_id, max_id))`.
- Left-merge onto every focal direction (`validate="many_to_one"`).
- Symmetric dyadic detectors receive identical values in both directions.
- Focal detectors remain directional (not collapsed).
- Canonical `mouse_a` / `mouse_b` = min/max of focal/partner.
- Unique `(dense_frame_idx, focal_id, partner_id)` after expansion.
- Source-order invariant.

**Deviation (C1):** missing required dyadic after merge → **hard fail** (no legacy zero-fill as confident negative). Validity / score_origin / threshold columns are carried through expansion.

---

## 5. Validity normalization

Evidence states per detector:

| State | Meaning |
| --- | --- |
| `valid_positive` | `input_valid` / observed + `pred=1` |
| `valid_negative` | observed + `pred=0` (including genuine prob=0) |
| `unobserved_invalid_route` | `input_valid=False` or `score_origin=routing_sentinel` |
| `missing_detector` | required column absent (intake should already fail) |
| `rule_vetoed` | scientific gate/veto zeroed a previously valid row |

Working `prob_*` for unobserved routes → NaN; `s7_prob_*` / `s7_pred_*` preserve provenance. A routing sentinel with exported probability 0 is **never** a valid negative.

---

## 6. Initial-gate formulas

### Hard-distance (Stage 3)

If distance present, non-null, and **>** threshold → zero working `prob_*`/`pred_*` for that detector (valid-evidence rows only):

| Detector | Column | Default |
| --- | --- | --- |
| nose_to_nose | `distance_nose_i_nose_j_norm` | 0.3 |
| nose_to_body | `distance_nose_i_side_j_norm` | 0.55 |
| nose_to_tail | `distance_nose_i_tail_j_norm` | 0.45 |
| side_by_side | `distance_spine_i_spine_j_norm` | 0.9 |
| contact | `distance_spine_i_spine_j_norm` | 1.2 |

NaN distance → skip that row. `null` threshold → disable gate.

### Postural approach veto (Stage 4a)

Legacy formula unchanged: rearing (`body_contraction_rate_7 > 5`) **and** partner behind (`behind_score_bl_j < -0.5` **or** `cos_bearing_i_to_j < -0.2`); 15-frame directional rolling max; escape if `rolling_mean_projected_vel_i_to_j_60 > 1`.

### Postural retract veto (Stage 4b)

Legacy formula unchanged: stationary / passive-receiver / flinch flags; 15-frame smear; escape via spine / combined projected / intent velocities.

Contact and side-by-side remain parallel channels (not exclusive ethogram members).

---

## 7. Invalid-input handling

| Case | Sprint 1 behavior |
| --- | --- |
| Routing sentinel | Masked to unobserved before gates; gates do not rewrite to valid-negative zeros (C2) |
| Valid negative | Eligible for gates |
| Missing required dyadic after expand | Hard fail (C1) |
| Missing required hard-gap columns | Hard fail (C4) |
| Missing recommended veto columns | Warning; legacy `.get` defaults keep vetoes safe |
| Hierarchy residual / `engaged` / `undefined` | Deferred to Sprint 2; bound by P4 |

---

## 8. Scientific deviations / unresolved ambiguities

Documented in S8-B §0.2 (C1–C5). Sprint 1 does not silently resolve them.

Additional notes:

- Sprint 1 requires dual wide views (`emit_wide_views`); canonical-only rebuild deferred (U7).
- Contact Stage-8 morphological close / Stage-10 gap fill remain separate (P6); not implemented yet.
- Pipeline 110 emit floors remain forbidden inside S8 (P8).

---

## 9. Tests and results

```text
tests/behavioral_arbitration/
  test_intake.py
  test_expand.py
  test_validity.py
  test_gates.py
  test_runner_sprint1.py
```

| Suite | Result |
| --- | --- |
| `pytest tests/behavioral_arbitration` | **30 passed** |
| Ruff (`src/behavioral_arbitration`, `tests/behavioral_arbitration`) | clean |
| Mypy (`src/behavioral_arbitration`) | clean |
| Targeted S7 export / invalid-route / discovery + S6 export QC | **29 passed** |

Coverage includes: joint lineage, stale linked S6 (no newest substitution), missing detectors/columns, K=2, expansion cardinality/identity/order invariance, valid-negative vs sentinel, hard-distance + postural vetoes, NaN distance skip, parallel contact/SBS preservation, no duplicate directional keys.

---

## 10. Headless output contract (Sprint 1)

`Sprint1ArbitrationResult` (in-memory only):

- Validated directional frame rows (`frames`)
- S7 detector probs/preds + validity + `s7_*` provenance + evidence states
- Joined S6 geometric / veto features
- Initial-gate results on working `prob_*`/`pred_*`
- Stage-level `diagnostics`
- Source/run identity + warnings + `settings_used`

No `behavior_label`, bouts, or Parquet publish.

---

## 11. Sprint 2 prerequisites

Addressed in [`sprint_2_arbitration_contact.md`](sprint_2_arbitration_contact.md):

- Hierarchy + decay (Stage 5) consuming masked arbitration inputs
- Adaptive approach/retract (Stages 6–7)
- Contact optical / Schmitt / morphological close **without** remultiplex (Stage 8) + pass-2 hard-distance (Stage 9)
- Apply P4 residual policy for valid vs undefined evidence at label write-back
- Keep Stage 10 gap fill independent of Stage 8 zero-gap morphology
- Intermediate `frames` schema from Sprint 1 remains the Stage 5 input spine
