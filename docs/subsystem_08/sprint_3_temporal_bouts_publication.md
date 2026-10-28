# Subsystem 08 — Sprint 3: Temporal Rules, Bouts and Publication

**Status:** Implemented (headless; no GUI).  
**Date:** 2026-10-28  
**Authority:** Accepted S8-A audit + S8-B implementation contract (MVP policy §0.1–0.2, stages §1.8–1.11 / §7–§9).

Sprint 3 continues from the Sprint 2 directional frame spine and migrates Pipeline 100 Stages 10–21, then finalizes exclusive labels, collapses bouts, runs structural QC, and publishes immutable S8 artifacts. It does **not** implement Group 3 GUI, Pipeline 110 emit floors, overlay video, or manual bout correction.

---

## 1. Remaining migrated stage functions (10–21)

| Stage | Pipeline 100 name | Production module / function |
| --- | --- | --- |
| 10 | Gap fill + min duration on all `pred_*` | `temporal.fill_small_gaps_and_filter_duration` ← `_fill_small_gaps_and_filter_duration` (+ invalid-span skip) |
| 11 | Synchronize `behavior_label` from `pred_*` | `temporal.synchronize_behavior_label_from_preds` |
| 12 | Contextual engaged bout stitching | `temporal.stitch_contextual_engaged_bouts` |
| 13 | Dynamic transition stitching | `temporal.stitch_dynamic_transitions` |
| 14 | Idle–retract–idle collapse | `temporal.stitch_idle_retract_idle_bouts` |
| 15 | Min MD label duration | `temporal.enforce_min_behavior_label_duration` |
| 16 | Rescue dynamic arcs | `temporal.rescue_dynamic_arcs` |
| 17 | Approach close-distance hard gate | `temporal.apply_approach_probability_gate` |
| 18 | Re-sync label from preds | `temporal.synchronize_behavior_label_from_preds` (second call) |
| 19 | Nose-to-nose reciprocity | `temporal.enforce_nose_to_nose_reciprocity` |
| 20 | Mutually-disengaged reciprocity | `temporal.enforce_mutual_disengaged_reciprocity` |
| 21 | Final engaged hysteresis | `temporal.apply_final_engaged_hysteresis` |
| — | Finalize unresolved → undefined | `finalize.finalize_behavior_labels` |
| — | Bout collapse | `bouts.extract_behavior_bouts` |
| — | States / diagnostics export | `export.build_behavior_states_table` / `build_arbitration_diagnostics_table` |
| — | Structural QC | `qc.evaluate_behavioral_arbitration_qc` |
| — | Immutable publish | `runner.run_behavioral_arbitration_publication` |
| — | Run discovery | `run_discovery.summarize_behavioral_arbitration_project` |

Helpers preserved with legacy formulas: `fill_gaps`, `apply_min_duration`, `get_kinematic_intent_series`.

Headless entry points:

| API | Role |
| --- | --- |
| `apply_sprint3_stages(frames, config)` | Stages 10–21 + finalize on a Sprint 2 frame table |
| `run_behavioral_arbitration_sprint3(request, sprint2_result=...)` | In-memory Sprint 1–3 → `Sprint3ArbitrationResult` |
| `run_behavioral_arbitration_publication(request)` | Full pipeline + atomic Parquet / meta / QC publish |

Orchestration order is exactly Pipeline 100: gap/min → sync → contextual stitch → dynamic stitch → idle-retract → min MD label → rescue → approach close gate → sync → NN reciprocity → MD reciprocity → hysteresis → finalize.

---

## 2. Rule ordering and semantics

### Stage 10 — gap / min-duration

- Applied per directional pair (`exp_name`, `focal_id`, `partner_id`) sorted by `dense_frame_idx`.
- `gap_frames = int(gap_fill_seconds * fps)` (default **0.050 × 120 → 6**).
- `min_duration_frames = int(min_bout_seconds * fps)` (default **0.100 × 120 → 12**).
- Override: `mutually_disengaged` uses `min_mutually_disengaged_frames` (**120**).
- `gap_frames=0` → no fill (identity for gap step). Positive windows fill interior 0-runs of length ≤ max only when bounded by 1s on both sides.
- Invalid safeguard: if any frame in a candidate gap has `arbitration_input_valid_<det>==False`, that gap is **not** filled; invalid frames are forced to pred 0 so they cannot seed positive bouts.
- Does **not** rewrite validity masks, evidence states, score origins, or `s7_*` columns.

### Stages 11 / 18 — label sync

Priority: `nose_to_nose` → `nose_to_body` → `nose_to_tail` → `approach` → `retract` → `mutually_disengaged` → `engaged`.  
Default when no exclusive pred is active is `engaged`, except rows still marked `unresolved` / `undefined` or `hierarchy_residual_kind=unresolved_invalid_evidence` with no hierarchy winner — those are preserved until finalize (P3/P4).

### Stages 12–14 — stitch

| Stage | Trigger | Replacement |
| --- | --- | --- |
| 12 | Short `engaged` island (`cc_count==1` when present) between sniff or same dynamic labels | Neighbor sniff / dynamic label (margin tie-break when sniff labels differ) |
| 13 | Short `engaged` between approach/retract / contact / MD contexts | Case table (dynamic↔dynamic, contact→dynamic, turnaround, idle after retract, hit-and-run, bait-and-switch) with kinematic trigger points |
| 14 | Short `retract` blip between MD (or far-apart) | `mutually_disengaged` |

### Stages 15–17

- Stage 15 replaces short MD **labels** with `engaged` (independent of Stage 10 MD pred override).
- Stage 16 rescues high-momentum approach/retract arcs from engaged gaps using kinematic peaks + probability hints.
- Stage 17 zeros `prob_approach` / `pred_approach` when NN/NB/NT distances are below configured hard gaps; does **not** invent valid-negative evidence.

### Stages 19–21 — reciprocity / hysteresis

- NN and MD require both directions of a dyad on the same frame; one-sided → fallback `engaged`.
- Parallel `contact` / `side_by_side` preds are **not** rewritten by reciprocity.
- Hysteresis merges short `engaged` islands when previous and next labels are equal and neither is `engaged` / `mutually_disengaged`.

### Finalize

See §4.

---

## 3. Invalid-evidence handling

Carried from Sprint 1–2 through temporal stages:

| State | Gap fill (10) | Sync / stitch / rescue | Reciprocity / hysteresis | Finalize |
| --- | --- | --- | --- | --- |
| Valid positive / negative | Eligible | Eligible | Eligible | Concrete or engaged |
| `unobserved_invalid_route` | Gap skip if span contains invalid; pred forced 0 | Never promoted via sentinel zero | May still rewrite exclusive labels from surrounding context | Unresolved invalid → `undefined` |
| `rule_vetoed` | Treated as observed zeroed pred | Uses post-veto preds | Same | Same as valid zeros |
| Missing detector | Intake already hard-fails | — | — | — |

Temporal stages never clear `arbitration_input_valid_*`, `evidence_state_*`, `score_origin_*`, or `s7_prob_*` / `s7_pred_*`.

---

## 4. Final label vocabulary (engaged vs undefined)

| Condition | Final `behavior_label` | Notes |
| --- | --- | --- |
| Concrete exclusive winner | detector name | Sniff / dynamic / MD / engaged |
| `hierarchy_residual_kind=engaged_valid` (incl. all-negative valid) | `engaged` | **P4** — never auto-`undefined` |
| Provisional `unresolved` / `unresolved_invalid_evidence` with no hierarchy pred winner | `undefined` | **U2 resolved for MVP**; `undefined_reason` recorded |
| One-sided reciprocity fallback | `engaged` | Before finalize |

- There is **no** `pred_undefined` column; exclusive `pred_*` are zeroed on undefined rows.
- `unresolved` is provisional only; published states should not retain it after finalize.

---

## 5. Bout construction

`bouts.extract_behavior_bouts` collapses finalized states into:

| Channel | Identity grain | Source |
| --- | --- | --- |
| `exclusive` | directional (`focal_id`, `partner_id`) | Contiguous `behavior_label` runs |
| `contact` | symmetric pair | Positive `pred_contact` (undirected collapse) |
| `side_by_side` | symmetric pair | Positive `pred_side_by_side` (undirected collapse) |

Rules:

- Inclusive endpoints (`start_sample_idx` / `end_sample_idx`).
- `bout_duration_frames = n_rows`; `bout_duration_sec = n_frames / fps` (120 Hz).
- Split on behavior / pred change **or** `dense_frame_idx` discontinuity (gap ≠ 1).
- Directional identity change (A→B vs B→A) yields separate exclusive bouts.
- Parallel contact/SBS emit independent bout rows (**U3 resolved for MVP**).
- **No** Pipeline 110 `min_emit_bout_frames` / `min_emit_bout_seconds` filters (P8).

---

## 6. Artifact schemas

Immutable run folder: `{session}/behavioral_arbitration/behavioral_arbitration__{timestamp}/`

| Artifact | Schema stamp | Role |
| --- | --- | --- |
| `behavior_states.parquet` | `behavioral_arbitration_states_v1` | Directional framewise ethogram + parallel channels + evidence / s7 provenance |
| `behavior_bouts.parquet` | `behavioral_arbitration_bouts_v1` | Exclusive + parallel bout table |
| `qc.json` | `behavioral_arbitration_qc_v1` | Structural OK + warnings + metrics |
| `run_meta.json` | `behavioral_arbitration_run_meta_v1` | Status, provenance digests, settings pointer |
| `settings_used.yaml` | settings v1 snapshot | Exact config used |
| `processing_log.txt` | text | Operator-facing log |
| `arbitration_diagnostics.parquet` | `behavioral_arbitration_stage_diag_v1` | Optional (`emit_arbitration_diagnostics`) |

Publish is atomic (temp → replace). Duplicate timestamp / existing run dir is rejected. On failure after science write attempts, science artifacts are unlinked; `run_meta` records `failed`.

---

## 7. QC and discovery

### QC (structural vs warning)

Structural failures block publication (`PublicationError`): empty states, missing detector preds, directional key duplicates, K=2 cardinality mismatch when `n_frames` supplied, missing provenance blocks, infinite probs, non-positive bout durations, ineligible linked S7/S6 flags.

Warnings do **not** block: empty bout table note, missing optional parallel column notes, residual unresolved counts, eligibility flag omitted (when not False).

### Discovery

`summarize_behavioral_arbitration_project` / `find_latest_eligible_s8_run` classify completed runs as:

| Classification | Meaning |
| --- | --- |
| `eligible` / `completed_with_warnings` | Completed + science present + recorded S7/S6 digests still match |
| `stale_source` | Digests / linked S7 eligibility changed since publish |
| `failed_or_incomplete` / `missing_required_artifacts` | Non-completed or incomplete folder |

Historical `status` on disk is never rewritten when sources go stale.

---

## 8. Parity notes / intentional deviations

1. **Invalid-span gap skip** — Pipeline 100 filled numeric zeros indiscriminately; S8 skips gaps containing `arbitration_input_valid_*=False` (P3/C2).
2. **Unresolved → undefined** — legacy residual was always `engaged` / MD; S8 publishes `undefined` for wholly invalid evidence (U2/P4).
3. **Sync preserve** — Stage 11/18 does not silently promote unresolved invalid rows to `engaged` via the legacy default.
4. **Contact morph zero-gap** — Stage 8 still disables morphological close at `max_gap_to_fill=0`; Stage 10 gap fill remains independent (P6).
5. **No contact remultiplex** — S7 fused contact trusted (Sprint 2 carry-forward).
6. **No Pipeline 110 emit floors** inside S8 (P8).
7. **Parallel contact/SBS bouts emitted** in MVP (U3).

Formula parity target remains fully valid evidence under documented config defaults (`gap_fill_seconds=0.050`, stitch / rescue / hysteresis YAML values unchanged from Pipeline 100 promotions).

---

## 9. Sprint 4 GUI prerequisites

Before Group 3 GUI / lab acceptance:

- Headless publication path stable on E-session eligible S7 + linked S6.
- Discovery eligible vs stale behavior verified against digest mutation.
- Bout browser / review remains **downstream** (Pipeline 110 / future review) — S8 page should launch / inspect runs, not edit bouts.
- GUI must never overwrite S7 or S6 artifacts; reuse S7 page patterns (immutable request, progress, eligibility banners).
- Optional diagnostics parquet toggle exposed as a non-scientific execution setting.
- Overlay video and manual correction remain out of MVP (P7 / U5).

Architectural stop: **stop after S8-3 for review** before GUI implementation (see implementation contract §14).

---

## 10. Tests

```text
tests/behavioral_arbitration/
  test_temporal_gap_min.py
  test_temporal_stitch_rescue.py
  test_temporal_reciprocity_hysteresis.py
  test_finalize_labels.py
  test_bouts.py
  test_export_qc_discovery.py
  test_runner_sprint3.py
  (+ Sprint 1–2 suites retained)
```
