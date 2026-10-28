# Subsystem 08 — Sprint 2: Hierarchy, Adaptive Movement and Contact Arbitration

**Status:** Implemented (headless; no Stage 10+ temporal / publish / GUI).  
**Date:** 2026-10-28  
**Authority:** Accepted S8-A audit + S8-B implementation contract (MVP policy §0.1–0.2, stages §1.5–1.7 / §2–§3).

Sprint 2 continues from the Sprint 1 directional frame spine and migrates Pipeline 100 Stages 5–9: hierarchical conflict resolution with decay, adaptive approach/retract re-binarization, contact post-inference cleaning **without** specialist remultiplex, and pass-2 hard-distance gates. It does **not** implement Stage 10 gap fill, bout stitching, reciprocity, hysteresis, Parquet publish, or GUI.

---

## 1. Exact stages migrated

| Stage | Pipeline 100 name | Production module / function |
| --- | --- | --- |
| 5 | Hierarchical conflict resolution + decay | `hierarchy.resolve_hierarchical_conflicts` ← `_resolve_hierarchical_conflicts` |
| 6 | Adaptive approach thresholding | `adaptive.apply_adaptive_approach_thresholding` ← `_apply_adaptive_approach_thresholding` |
| 7 | Adaptive retract thresholding | `adaptive.apply_adaptive_retract_thresholding` ← `_apply_adaptive_retract_thresholding` |
| 8 | Contact optical / Schmitt / morph / min bout | `contact_clean.apply_contact_arbitration_no_remultiplex` + `get_cleaned_contact_bouts` ← `apply_sota_contact_arbitration` / `get_cleaned_bouts` **minus remultiplex** |
| 9 | Hard-distance gates (pass 2) | `gates.apply_hard_distance_probability_gates(..., pass_id=2)` ← second `_apply_hard_distance_probability_gates` call |

**Not migrated (Sprint 3+):** Stage 10 gap fill / min duration; Stages 11–21 sync / stitch / rescue / reciprocity / hysteresis; bout tables; immutable publish; GUI.

Sprint 1 already owns Stages 1–4 (expand, merge, hard-distance pass 1, postural vetoes) plus validity normalization.

Headless entry points:

| API | Role |
| --- | --- |
| `run_behavioral_arbitration_sprint1(request)` | Stages 1–4 |
| `apply_sprint2_stages(frames, config)` | Stages 5–9 on a Sprint 1 frame table |
| `run_behavioral_arbitration_sprint2(request, sprint1_result=...)` | Full Sprint 1+2 → `Sprint2ArbitrationResult` |

---

## 2. Source-to-production mapping

| Legacy helper | Production |
| --- | --- |
| `_calculate_normalized_margin` | `hierarchy.calculate_normalized_margin` |
| `_resolve_threshold_value` / `_resolve_threshold_for_row` | `hierarchy.resolve_threshold_value` / `resolve_threshold_for_row` |
| `_directional_rolling_max` | `gates._directional_rolling_max` (shared) |
| `_prediction_flag` | `adaptive._prediction_flag` |
| Contact remultiplex block | **Intentionally omitted** (S7 fused contact trusted) |
| `max(1, max_gap_to_fill)` clamp | **Replaced** by MVP zero-gap policy (§3 below) |

---

## 3. Hierarchy and adaptive rules

### Hierarchy (Stage 5)

Per `(exp_name, focal_id, partner_id)` ascending `dense_frame_idx`:

1. Soft approach penalty while `cc_count==1` (`approach_contact_penalty`, default **0.0** — no-op).
2. Decay memory: when streak ≥ `conf_frames` (**5**) and winner matches detector,  
   `eff = max(rank_prob, winner_prob * decay_rate)` with YAML `decay_rate` **0.5**.  
   Sniff memory clears when `cc_count==2`. Engaged / unresolved do not become decay winners.
3. Margin `S = (P - T) / (1 - T)` if `P ≥ T` else 0; `T` from `working_thresholds` (occlusion-aware intact/occluded dicts).
4. Layer 1 (`nose_to_nose`, `nose_to_body`, `nose_to_tail`): requires `S > layer1_significance_floor` (**0.05**). Strict `>` → earlier detector wins ties.
5. Else Layer 2 (`approach`, `retract`, `mutually_disengaged`): max `S`.
6. Else residual (see §6).

Writes exclusive hierarchy `pred_*`, optional decayed winner `prob_*`, provisional `behavior_label`, `hierarchy_residual_kind`, `hierarchy_layer`, `hierarchy_decay_carry`. **Does not** touch `contact` / `side_by_side`.

### Adaptive (Stages 6–7)

| | Approach | Retract |
| --- | --- | --- |
| Engaged distance | `distance_nose_i_spine_j_norm < 3.0` | `< 3.0 * 1.5` |
| Strict | `prob >= 0.5` | same |
| Envelope | directional centered rolling max, **15** frames, trip ≥ **0.25** | same |
| Susceptible | engaged **and** no active `pred_contact` / sniff preds | same |
| Write | `y_approach_smoothed` → `pred_approach` | `y_retract_smoothed` → `pred_retract` |

Uses contact/sniff preds **after** hierarchy and **before** contact cleaning (legacy order). Does not rewrite `behavior_label` (Stage 11 sync is Sprint 3). Invalid / NaN probs never trip strict or envelope thresholds. Legacy unused `direction_window_frames` is not invented.

---

## 4. Contact-processing boundaries

### In scope (Stage 8)

1. Optical assist: `cc_count==1` and observed `prob_contact ≥ optical_assist_min` (0.30) → boost fused score to occlusion-aware high threshold (intact **0.9731** / occluded **0.4295**).
2. Centered rolling median smooth (`smoothing_window` **3**).
3. Schmitt trigger: high / low (**0.4**) with forward-fill; initial fill 0.
4. Morphological closing (see zero-gap policy).
5. Contact-specific `min_bout_duration` (**6** frames).

Write-back: working `prob_contact` ← smooth (observed rows); `pred_contact` ← final mask. Diagnostic columns: `prob_contact_fused`, `prob_contact_smooth`, `contact_schmitt_state`, `contact_optical_boost`.

### Out of scope / forbidden

- Remultiplex of `prob_contact_intact` / `prob_contact_occluded` into fused contact (S7 owns fusion).
- Stage 10 general gap fill on all `pred_*`.
- Folding contact into exclusive ethogram.

### Zero-gap policy (binding)

| Config | Behavior |
| --- | --- |
| `max_gap_to_fill: 0` (default) | **Disable** contact morphological closing (identity; no `0→1` clamp) |
| `max_gap_to_fill ≥ 1` | Legacy morphological close window `W` |
| Stage 10 | Separate; reserved for Sprint 3 (`gap_fill_seconds=0.050` → 6 frames) |

Contact and side-by-side remain parallel channels after all Sprint 2 stages.

### Pass 2 (Stage 9)

Same hard-distance formulas as Stage 3, re-applied after contact cleaning so rewritten contact scores respect distance gates. Invalid-route rows remain masked (not rewritten to valid negatives).

---

## 5. Configuration values (legacy parity)

Consumed from `configs/subsystem_08/behavioral_arbitration_default.yaml` (no new hardcoded scientific defaults in production paths):

| Key | Value |
| --- | --- |
| `decay_rate` | 0.5 |
| `conf_frames` | 5 |
| `engaged_min_threshold` | 0.0469 |
| `hierarchy.layer1_significance_floor` | 0.05 |
| `hierarchy.approach_contact_penalty` | 0.0 |
| `adaptive.*` | 3.0 / 1.5 / 0.5 / 0.25 / 15 |
| `post_processing.contact.*` | high intact/occluded 0.9731/0.4295; low 0.4; optical 0.30; gap 0; min bout 6 |
| `working_thresholds` | S7 echoes (NN 0.0816, NB intact/occluded, …) |
| Hard-gap distances | unchanged from Sprint 1 |

---

## 6. Invalid-evidence behavior

Carried from Sprint 1 evidence states through Stages 5–9:

| State | Hierarchy | Adaptive | Contact |
| --- | --- | --- | --- |
| Valid positive / negative | Eligible for margins / residual | Eligible | Eligible |
| `unobserved_invalid_route` | Margin contribution off (NaN, not 0-fill); never wins via sentinel zero | Envelope/strict ignore | No optical boost; no Schmitt trip; working `prob_contact` stays NaN |
| `rule_vetoed` | Treated as observed zeroed evidence | Uses post-veto preds | Pass-2 / min-bout may stamp further veto |
| Missing detector | Intake already hard-fails full profile | — | — |

`s7_prob_*` / `s7_pred_*` provenance columns are preserved. Temporal decay may bridge a confirmed winner across a transient gap for ranking, but does **not** erase `arbitration_input_valid_*` / `evidence_state_*` masks.

**Conservative conflict note:** legacy hierarchy `float(prob)` with implicit 0 for missing columns would treat sentinels as confident negatives. S8 intentionally diverges (P3/C2). Numerical parity vs Pipeline 100 on invalid-route frames is therefore expected to differ; formula parity on fully valid evidence is the migration target.

---

## 7. Residual / undefined policy (C3 / P4)

Resolved at hierarchy write-back:

| Condition | `behavior_label` | `hierarchy_residual_kind` |
| --- | --- | --- |
| Layer 1 or 2 winner | winning detector | `layer_winner` |
| No winner; MD observed and `prob_MD ≥ engaged_min_threshold` | `mutually_disengaged` | `mutually_disengaged_baseline` |
| No winner; **any** hierarchy detector has valid evidence | `engaged` | `engaged_valid` |
| No winner; **all** hierarchy detectors unobserved | `unresolved` (provisional) | `unresolved_invalid_evidence` |

- Valid frames with no winning candidate keep legacy **`engaged`** (not auto-`undefined`).
- Wholly invalid evidence is **not** silently converted to `engaged`.
- Final publish label `undefined` (U2) remains deferred; Sprint 3+ write-back may promote provisional `unresolved` or keep QC-only counting. No new scientific fallback threshold was introduced.

---

## 8. Diagnostic schema

`StageDiagnostic(stage, rule_code, detector, n_rows_changed, detail)` aggregates in `Sprint2ArbitrationResult.diagnostics`.

| Rule code | Stage |
| --- | --- |
| `HIERARCHY_WIN` / `DECAY_CARRY` / `HIERARCHY_RESIDUAL` | hierarchy |
| `ADAPTIVE_APPROACH` / `ADAPTIVE_RETRACT` | adaptive |
| `CONTACT_OPTICAL` / `CONTACT_SCHMITT` / `CONTACT_MORPH` / `CONTACT_MIN_DUR` | contact |
| `GATE_HARD_DISTANCE_PASS2` | pass 2 |

Frame-level fields for explainability: `behavior_label`, `hierarchy_residual_kind`, `hierarchy_layer`, `hierarchy_decay_carry`, `y_approach_smoothed`, `y_retract_smoothed`, `prob_contact_fused`, `prob_contact_smooth`, `contact_schmitt_state`, `contact_optical_boost`, plus Sprint 1 evidence / `s7_*` columns.

A behavior change can be attributed by inspecting which rule code / residual kind / adaptive or contact diagnostic fired for that directional key.

Warnings include the zero-gap morphological-disable note when `max_gap_to_fill=0`.

---

## 9. Numerical / parity test findings

```text
tests/behavioral_arbitration/
  test_hierarchy.py
  test_adaptive.py
  test_contact_clean.py
  test_runner_sprint2.py
  (+ Sprint 1 suites retained)
```

| Check | Result |
| --- | --- |
| `pytest tests/behavioral_arbitration` | **57 passed** |
| Ruff (`src/behavioral_arbitration`, `tests/behavioral_arbitration`) | clean |
| Mypy (`src/behavioral_arbitration`) | clean |
| Targeted S7 + S6 suites | see validation report |

Representative numerical examples covered:

- Margin: `S=(0.9-0.0816)/(1-0.0816)` exact.
- Decay: after 5 confirmed frames, `eff = max(0.01, 0.90*0.5) = 0.45` with `cc_count=1`; sniff decay blocked when `cc_count=2`.
- Optical assist boost to intact high **0.9731**.
- Schmitt hold between high/low; morph fill with `W=3` vs identity at `W=0`.
- Adaptive envelope on A→B only when directional distance is engaged; retract wider window `4.5`.

**Intentional parity differences vs Pipeline 100** (validity hardening, not formula bugs):

1. Invalid-route probs are NaN-masked (legacy treated exported 0 as numeric).
2. Contact morphological `max_gap_to_fill=0` disables close (legacy code clamped to ≥1; YAML-0 runs were observationally identity — MVP matches YAML intent).
3. No contact remultiplex (S7 fusion trusted).
4. Provisional `unresolved` residual for wholly invalid hierarchy evidence (legacy always `engaged` / MD).

---

## 10. Sprint 3 prerequisites

- Stage 10 `_fill_small_gaps_and_filter_duration` (independent of Stage 8 zero-gap morph).
- Stage 11 label sync from `pred_*` (adaptive/contact may diverge from provisional `behavior_label`).
- Stages 12–16 stitch / rescue; Stage 17 approach close gate; Stage 18 re-sync.
- Stages 19–21 reciprocity + engaged hysteresis.
- Finalize provisional `unresolved` → publish policy (engaged vs undefined vs QC-only).
- Bout collapse + immutable `behavior_states` / `behavior_bouts` / QC / run_meta publish.
- Optional arbitration diagnostics parquet; GUI remains Sprint 4.

Intermediate handoff: `Sprint2ArbitrationResult.frames` retains directional keys, S7 evidence, hierarchy/adaptive/contact working columns, and stage diagnostics.
