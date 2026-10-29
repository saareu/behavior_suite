# Subsystem 08 — Behavioral Arbitration and Bout Generation

## Implementation Contract (S8-B)

**Status:** Binding design contract. **S8 MVP complete / lab-validated** (Sprints 1–4 + scientific E-session audit 47/47 + GUI responsiveness remediation + operator laboratory GUI acceptance on 2026-10-28).  
**Date:** 2026-10-26 (MVP policy appendix recorded with S8-1; S8-3 temporal/publish recorded 2026-10-28; S8-4 GUI + lab closure recorded 2026-10-28)  
**Authority:** Accepted S8-A audit (`architecture_audit.md`) + source inspection of local `legacy/pipelines/100_behavioral_arbiter.py`, `legacy/configs/behavioral_arbiter.yaml`, Pipeline 110 consumers, S7 prediction contracts (`src/behavioral_inference/`), S6 feature contracts (`src/combined_features/`), and Group 3 GUI patterns.  
**Follow-on:** Group 4 Results/Review is delivered as Subsystems 09–10 (packaging + review). S8 itself still does not package clips or host manual bout editing.

S8 converts eligible S7 predictions (+ linked S6 feature columns) into a directional exclusive ethogram, parallel contact / side-by-side channels, and bout tables. It does **not** train models, re-run classifiers, re-fuse S7 contact specialists, overwrite S7 artifacts, package review clips, or host manual bout correction.

---

## 0. Binding decisions frozen in S8-B

| Topic | Binding stance |
| --- | --- |
| Subsystem shape | One package `behavioral_arbitration` + one Group 3 GUI page beside S7 |
| Rule authority | Preserve Pipeline 100 formulas and stage order unless an explicit reviewed decision below authorizes a change |
| Contact fusion | **Trust S7 fused** `prob_contact` / `pred_contact`; **do not remultiplex** specialists (§2) |
| Contact post-inference | Retain optical assist, Schmitt trigger, morphological gap, min-bout, and post-contact hard-distance re-gate |
| Zero-gap policy | Document legacy YAML/code discrepancy; MVP encodes **disable contact morphological close** as the observed YAML-0 execution (§3) |
| Validity | Gate on S7 `input_valid_*` / `score_origin_*`; never treat routing sentinels as confident negatives (§5) |
| Nose-to-body | Consume S7 **effective** `prob_nose_to_body`; **no** re-smoothing or re-thresholding of S7 inference |
| Hierarchy thresholds | Use frozen S8 config echoes of S7 working thresholds for S-score margins only; do not rewrite S7 files |
| Ethogram exclusivity | One `behavior_label` per `(dense_frame_idx, focal_id, partner_id)`; contact / SBS remain parallel (§6) |
| Outputs | Immutable S8 run folder; primary `behavior_states.parquet` + `behavior_bouts.parquet`; do not overwrite S7 |
| Manual review | Downstream of S8 (legacy Pipeline 110 / future review subsystem) |
| Overlay video | **Out of MVP**; optional later parity with Pipeline 100 |
| K animals | MVP assumes K=2 (same as current S7 `require_exactly_two_mice`) |

### 0.1 MVP policy decisions (S8-1; preserve legacy semantics separately)

These decisions govern S8 MVP implementation. Where they diverge from exact frozen Pipeline 100 formulas, the **original legacy semantics** remain documented in §1–§3 and are **not** silently rewritten. Conflicts are listed in §0.2 — do not resolve them without lab review.

| ID | Decision | Legacy semantics (preserved as reference) |
| --- | --- | --- |
| P1 | **Full required detector coverage for MVP; no partial arbitration.** Missing any Pipeline-100 detector → hard fail. | Legacy may proceed with whatever `prob_*` columns exist and fill missing with 0. |
| P2 | **Exactly two tracked mice (K=2).** | Legacy assumes a single undirected pair after collapse; no K>2 path. |
| P3 | **Explicitly distinguish unobserved inputs from valid negative predictions.** Routing sentinels (`input_valid=False`, `score_origin=routing_sentinel`) are never treated as confident negatives merely because exported probability is 0. | Legacy has no `input_valid` / `score_origin` policy; zero is always a numeric score. |
| P4 | **Undefined / unavailable evidence state** is permitted **only** when available evidence cannot support the legacy interpretation. It must **not** replace the legacy `engaged` fallback for **valid** inputs. Hierarchy residual for valid evidence remains `engaged` (§1.5 / §6.2). | Legacy residual is always `engaged` (or MD when MD prob ≥ threshold); no undefined label. |
| P5 | **Contact and side-by-side remain parallel channels** (not exclusive ethogram members). | Same as Pipeline 100. |
| P6 | **Preserve separate contact morphological closing (Stage 8) and Stage 10 gap filling.** Zero-gap YAML policy (§3) applies only to Stage 8 morphology; Stage 10 remains independent. | Same separation in Pipeline 100; YAML-0 + clamp discrepancy documented in §3. |
| P7 | **No manual correction or visualization overlay** in S8 MVP. | Overlay optional in Pipeline 100; bout correction is Pipeline 110 / review app. |
| P8 | **No downstream-only Pipeline 110 emission filters** inside S8 (`min_emit_bout_frames` / `min_emit_bout_seconds`). | Pipeline 110 applies emit floors; Pipeline 100 does not. |
| P9 | **Promote relevant hardcoded scientific constants to configuration** without changing their numeric values (§4.2). | Constants live in `100_behavioral_arbiter.py` body. |

### 0.2 Conflicts with frozen legacy formulas (do not silently resolve)

| Conflict | Legacy formula / behavior | MVP policy | Sprint 1 stance |
| --- | --- | --- | --- |
| C1 — Missing dyadic after left-merge | Fill `prob_*=0.0`, `pred_*=0` for all dyadic detectors (§1.1) | P3: do not equate unobserved miss to confident negative; P2: K=2 pair must exist | **Hard fail** when a required dyadic detector is missing for a K=2 directional frame after expansion. Do not zero-fill as observed. |
| C2 — Invalid route zeros in gates / later rules | Distance gates and hierarchy operate on raw `prob_*` including sentinel zeros | P3: routing sentinel ≠ valid negative | **Evidence normalization first:** arbitration-input probs for invalid routes are masked (NaN / `arbitration_input_valid=False`). Initial hard-distance and postural gates mutate **only** valid-evidence rows. Invalid rows stay unobserved (not rewritten to valid-negative zeros). |
| C3 — Undefined vs engaged residual | Baseline always picks MD or `engaged` (§1.5) | P4: undefined only when evidence cannot support legacy interpretation; never replace engaged for valid inputs | Sprint 1 does **not** publish `behavior_label`. Record per-detector evidence state (`valid_positive` / `valid_negative` / `unobserved_invalid_route` / `missing_detector` / `rule_vetoed`). Hierarchy residual policy deferred to S8-2 with P4 binding. |
| C4 — Missing required hard-gap columns | Warn and skip hard-gap merge / gates | S8-B §1.2 / §5.4: missing **required** hard-gap / optical columns → hard fail | **Hard fail** (intentional structural strengthening). Recommended veto/kinematic columns: warn + skip dependent vetoes. |
| C5 — Partial detector runs | May run on subset of columns | P1: full coverage only | **Hard fail** (U1 remains closed for MVP). |

---

## 1. Frozen Pipeline 100 execution order (stage contracts)

Units assume laboratory **120 Hz** unless noted. Frame windows are integers; millisecond defaults convert as `int(seconds * fps)`.

Source of truth for orchestration: `legacy/pipelines/100_behavioral_arbiter.py` → `run_behavioral_arbiter()` (approx. lines 1939–2409). Nested helpers cited per stage.

```text
[Intake]   Load configs + S7 wide (or canonical) predictions + linked S6 features
[1]        Expand undirected dyadic → directional spine
[2]        Merge matching S6 feature samples
[3]        Hard-distance gates (pass 1)
[4]        Postural approach veto → postural retract veto
[5]        Hierarchical conflict resolution + decay → behavior_label
[6]        Adaptive approach thresholding
[7]        Adaptive retract thresholding
[8]        Contact optical assist + Schmitt + morphological gap + min bout
[9]        Hard-distance gates (pass 2; contact may have been rewritten)
[10]       Gap fill + minimum-duration on all pred_*
[11]       Synchronize behavior_label from pred_* priority
[12]       Contextual engaged bout stitching
[13]       Dynamic transition stitching
[14]       Idle–retract–idle stitching
[15]       Enforce min mutually_disengaged duration on behavior_label
[16]       Rescue dynamic arcs (approach/retract)
[17]       Approach close-distance hard gate
[18]       Re-synchronize behavior_label from preds
[19]       Nose-to-nose reciprocity
[20]       Mutually-disengaged reciprocity
[21]       Final engaged hysteresis
[Publish]  Ethogram summary + immutable S8 artifacts (+ optional later overlay)
```

Stages 1–10 below map to the user’s ten scientific blocks; stages 11–21 are the nested temporal / reciprocity / hysteresis steps inside those blocks and must remain ordered exactly as Pipeline 100.

### 1.1 Stage 1 — Expand undirected dyadic predictions into directional records

| Field | Contract |
| --- | --- |
| Source | `_expand_dyadic_to_directional` (`100_behavioral_arbiter.py` ~421–517) |
| Inputs | Focal directional spine (`focal_id`, `partner_id`, `dense_frame_idx`, …); undirected dyadic rows with collapse-survivor IDs (`focal_id=mouse_a`, `partner_id=mouse_b`) |
| Keys | Undirected join: `(exp_name, dense_frame_idx, (min(focal,partner), max(...)))` |
| Calculation | Left-merge dyadic `prob_*`/`pred_*` onto every focal direction sharing the unordered pair; `validate="many_to_one"` |
| Failures | Duplicate undirected dyadic keys → **hard fail**; duplicate directional focal keys → **hard fail** |
| Missing dyadic | Fill `prob_*=0.0`, `pred_*=0` for both symmetric and non-symmetric dyadic detectors |
| Symmetric detectors | `nose_to_nose`, `contact`, `side_by_side`, `mutually_disengaged` — both directions receive the **same** signal |
| Intermediate | Drop helper `_pair`; sort `(exp_name, dense_frame_idx, focal_id, partner_id)` |
| Effect | Directional table is the arbitration grain for all later stages |

**S8 intake addition (binding):** also expand / carry `input_valid_*`, `score_origin_*`, `invalid_reason_*`, and S7 `threshold_*` echoes for every dyadic detector so validity is not lost at expansion (§5).

### 1.2 Stage 2 — Merge matching S6 feature samples

| Field | Contract |
| --- | --- |
| Sources | `_load_hard_gap_distance_features`, `_load_rescue_kinematic_features`, `_load_focal_veto_features` (~289–404, ~69–111); orchestration merges ~2084–2147 |
| Join | `(exp_name, dense_frame_idx, focal_id, partner_id)`, `how="left"`, `validate="many_to_one"` |
| Required column families | See §5.4 |
| Missing artifact | Legacy: warn and skip that family (gates/vetoes/stitch degrade). **S8 MVP structural policy:** missing **required** hard-gap / optical columns → **hard fail**; missing recommended veto/kinematic columns → scientific warning + documented skip of dependent stages |
| Intermediate | Feature columns appear on the directional spine; no prediction rewrite yet |
| Effect | Enables distance gates, postural vetoes, optical assist, stitching, rescue |

### 1.3 Stage 3 — Hard-distance gates (and pass 2 after contact)

| Field | Contract |
| --- | --- |
| Source | `_apply_hard_distance_probability_gates` (~567–619) |
| Rule | If distance column present, non-null, and **>** threshold → `prob_*=0`, `pred_*=0` |
| Config (`behavioral_arbiter.yaml`) | Defaults below; `null` disables that gate |

| Detector | Distance column | Default threshold (`*_norm`) |
| --- | --- | --- |
| `nose_to_nose` | `distance_nose_i_nose_j_norm` | 0.3 |
| `nose_to_body` | `distance_nose_i_side_j_norm` | 0.55 |
| `nose_to_tail` | `distance_nose_i_tail_j_norm` | 0.45 |
| `side_by_side` | `distance_spine_i_spine_j_norm` | 0.9 |
| `contact` | `distance_spine_i_spine_j_norm` | 1.2 |

| Missing distance col / NaN distance | Skip that gate for those rows (no invented distance) |
| Ordering | Pass 1 before hierarchy; **pass 2 after contact cleaning** (legacy comment: contact may have been remultiplexed / rewritten) |
| Effect | Far frames cannot win hierarchy or survive as positives for gated detectors |

### 1.4 Stage 4 — Postural vetoes

#### Approach veto — `_apply_postural_approach_veto` (~189–229)

| Item | Value |
| --- | --- |
| Base impossible | `body_contraction_rate_7 > 5.0` **and** (`behind_score_bl_j < -0.5` **or** `cos_bearing_i_to_j < -0.2`) |
| Smear | Directional rolling max, window **15 frames**, not centered |
| Escape | `rolling_mean_projected_vel_i_to_j_60 > 1.0` |
| Effect | Vetoed → `prob_approach=0`, `pred_approach=0` |
| Missing features | `.get(...)` defaults make flags false / safe; veto does nothing for missing columns |

#### Retract veto — `_apply_postural_retract_veto` (~147–186)

| Item | Value |
| --- | --- |
| Stationary | `rolling_speed_i_15 < 15.0` (default series 100 if missing → not stationary) |
| Passive receiver | `focal_effort_ratio_15 < 0.20` **and** `rolling_speed_j_15 > 80.0` |
| Flinch | `body_contraction_rate_7 > 3.0` **and** `spine_directed_vel_15_i > -5.0` |
| Smear | Directional rolling max, window **15 frames** |
| Escape | `spine_directed_vel_15_i < -1.0` **or** `rolling_mean_projected_vel_combined_i_to_j_7 < -1.0` **or** `signed_kinematic_intent_nose_or_neck < -1.0` |
| Effect | Vetoed → `prob_retract=0`, `pred_retract=0` |

**Ambiguity note:** `rolling_mean_projected_vel_combined_i_to_j_7` is used in the escape hatch but is **not** listed in `FOCAL_VETO_COLUMNS`. It is produced by S6 Pipeline-50 post (`pipeline50_post.py`). S8 must merge it with veto features when present (§4 inventory).

All postural cutoffs are currently **hardcoded** (not YAML). Initial S8 config keys must copy these numeric values unchanged (§4).

### 1.5 Stage 5 — Behavioral hierarchy and decay

| Field | Contract |
| --- | --- |
| Source | `_resolve_hierarchical_conflicts` (~817–936) |
| Chronology | Per `(exp_name, focal_id, partner_id)`, ascending `dense_frame_idx` |
| Layer 1 | `nose_to_nose`, `nose_to_body`, `nose_to_tail` |
| Layer 2 | `approach`, `retract`, `mutually_disengaged` |
| Decay | `eff = max(rank_prob, winner_prob * decay_rate)` when streak ≥ `conf_frames` and winner matches detector |
| Config | `decay_rate` default YAML **0.5** (overrides module `DECAY_RATE=0.96`); `conf_frames` default **5**; `engaged_min_threshold` default **0.0469** |
| Soft approach penalty | `APPROACH_CONTACT_PENALTY = 0.0` while `cc_count==1` — **currently a no-op**; preserve constant |
| Sniff memory clear | If `cc_count==2` and prior winner is a sniff → clear memory |
| Margin | `S = (P - T) / (1 - T)` if `P ≥ T` else 0; `T` from focal/dyadic working thresholds (occlusion-aware dicts via `is_total_occlusion`) |
| Layer-1 floor | Winner requires `S > 0.05` (hardcoded significance floor) |
| Layer-2 | Max `S` among Layer 2 (floor 0) |
| Baseline | If no winner: `mutually_disengaged` if `prob_mutually_disengaged ≥ engaged_min_threshold`, else `engaged` |
| Write-back | Exclusive `pred_*` for hierarchy detectors; winner’s `prob_*` may be replaced by **effective** (decayed) prob; `behavior_label = winner`; `prob_engaged = 1 - prob_mutually_disengaged` when engaged |
| Memory | Engaged does **not** become a decay winner (`winner=None`) |
| Not in hierarchy | `contact`, `side_by_side` — preds untouched by this stage |
| Invalid inputs | Legacy ignores validity. **S8:** for detectors with `input_valid=False`, force rank/raw prob used in margins to **unobserved** treatment (§5.5) before computing `S` |

### 1.6 Stage 6–7 — Adaptive approach / retract

| | Approach `_apply_adaptive_approach_thresholding` (~1003–1039) | Retract `_apply_adaptive_retract_thresholding` (~1042–1080) |
| --- | --- | --- |
| Engaged distance | `distance_nose_i_spine_j_norm < 3.0` | `< 3.0 * 1.5` |
| Strict thresh | `prob >= 0.5` | same |
| Adaptive envelope | directional centered rolling max, **15 frames**, trip ≥ **0.25** | same |
| Susceptible | engaged distance **and** no active `pred_contact` / sniff preds | same |
| Write | `y_approach_smoothed`; overwrite `pred_approach` | `y_retract_smoothed`; overwrite `pred_retract` |
| Note | Uses contact/sniff preds **before** contact cleaning | `direction_window_frames=5` parameter exists but is **unused** in body — do not invent use |

These are **S8 arbitration re-binarizations**, not S7 working thresholds. Do not mutate S7 artifacts.

### 1.7 Stage 8 — Contact optical assistance and Schmitt-trigger processing

| Field | Contract |
| --- | --- |
| Sources | `get_cleaned_bouts` (~772–814) → per directional group `apply_sota_contact_arbitration` (~664–769) |
| Grain | Process each `(exp_name, focal_id, partner_id)` sorted by `dense_frame_idx` |
| Config | `behavioral_arbiter.post_processing.contact` |

| Sub-step | Formula / default |
| --- | --- |
| **(Legacy) Multiplexer** | If `prob_contact_occluded` present: `prob_contact = where(is_total_occlusion, occluded, intact_or_fused)` — **removed from S8 science path**; see §2 |
| Optical assist | If `cc_count==1` and `prob_contact ≥ optical_assist_min` (0.30): boost fused score to ≥ high threshold for that occlusion state |
| High threshold | intact **0.9731** / occluded **0.4295** (array by `is_total_occlusion`) |
| Low threshold | **0.4** |
| Smoothing | Centered rolling **median**, window **3** frames (`max(1, …)`) |
| Schmitt | State = 1 if smooth ≥ high; 0 if smooth < low; else forward-fill; initial fill 0 |
| Morphological gap | Centered rolling max then min with `window = max_gap_to_fill` after clamp (§3) |
| Min bout | Drop positive runs with length `< min_bout_duration` (**6** frames) |
| Write-back | `prob_contact ← prob_contact_smooth`; `pred_contact ← is_contact_final` |
| Does not | Touch `behavior_label` or hierarchy `pred_*` |

### 1.8 Stage 9–10 — Gap filling and minimum-duration enforcement

| Field | Contract |
| --- | --- |
| Source | `_fill_small_gaps_and_filter_duration` (~939–1000) using `_fill_gaps` / `_apply_min_duration` (~1864–1910) |
| Scope | **Every** `pred_*` column per directional pair chronological sequence |
| Gap fill | Fill 0-runs of length **≤** `gap_frames` that sit between 1-runs; `gap_frames = int(0.050 * fps)` → **6** @ 120 Hz |
| Min duration | Drop 1-runs shorter than `min_duration_frames = int(0.100 * fps)` → **12** @ 120 Hz |
| Override | `mutually_disengaged` → `min_mutually_disengaged_frames` (**120** frames = 1.0 s @ 120 Hz) |
| Semantics | Gap parameter = length of the **missing (0) run** between positives, in **frames**, not total transition length including positives |
| Effect | Cleans all detector binary channels; then Stage 11 rebuilds exclusive label |

After gap/min-duration, legacy copies `pred_approach`/`pred_retract` into `y_*_smoothed` if those diagnostic columns exist.

### 1.9 Stage 11–18 — Bout stitching and rescue (ordered)

#### 11. Label sync — `_synchronize_behavior_label_from_preds` (~1618–1651)

First true `pred_*` wins:

1. `nose_to_nose` → 2. `nose_to_body` → 3. `nose_to_tail` → 4. `approach` → 5. `retract` → 6. `mutually_disengaged` → 7. `engaged`  
Default if none: `engaged`. **Ignores** `pred_contact` / `pred_side_by_side`.

#### 12. Contextual engaged stitch — `_stitch_contextual_engaged_bouts` (~1083–1193)

| Condition | Action |
| --- | --- |
| Engaged island with **all** `cc_count==1`, length ≤ limits, bounded by labels | Reassign island to surrounding sniff/dynamic label |
| Same sniff both sides | ≤ `max_sniff_engaged_gap_frames` (**180**) |
| Same dynamic both sides | ≤ `max_dynamic_engaged_gap_frames` (**120**) |
| Different sniffs | Choose higher mean normalized margin on island probs |
| Sniff vs dynamic | **No** stitch |

#### 13. Dynamic transition stitch — `_stitch_dynamic_transitions` (~1306–1549)

Cases for short engaged gaps (≤ `max_dynamic_padding_frames` **120**, with some cases allowing `*2`). Uses kinematic intent/speed triggers (`min_kinematic_intent_stitch` **0.5**, `min_rolling_speed_stitch` **25**, hard look-ahead **15** frames, hard speed **50**). Syncs hierarchy `pred_*` from resulting labels.

#### 14. Idle–retract–idle — `_stitch_idle_retract_idle_bouts` (~1552–1615)

Short retract (≤ `max_idle_retract_gap_frames` **120**) after MD, if short-blip surrounded by MD **or** `min(distance_spine_i_spine_j_norm) > far_apart_idle_distance` (**3.5**) → label `mutually_disengaged`.

#### 15. Min MD on label — `_enforce_min_behavior_label_duration` (~1654–1689)

Short `mutually_disengaged` bouts on `behavior_label` → `engaged` (min **120** frames). Sync MD/engaged preds.

#### 16. Rescue dynamic arcs — `_rescue_dynamic_arcs` (~1196–1303)

Engaged islands preceding approach/retract (or high approach/retract prob hints **0.4**) with peak intent/speed (`case4_*` YAML defaults: peak intent **1.0**, peak speed **200**, initiation intent **0.5**, initiation speed **50**) backfill contiguous moving frames into approach/retract. Mutual exclusivity of approach vs retract realms.

#### 17. Approach close gate — `_apply_approach_probability_gate` (~622–661)

If any of (distance **<** threshold): force approach off.

| Distance | Default |
| --- | --- |
| `distance_nose_i_nose_j_norm` | 0.15 |
| `distance_nose_i_side_j_norm` | 0.2 |
| `distance_nose_i_tail_j_norm` | 0.15 |

#### 18. Re-sync label from preds

Same priority as Stage 11.

### 1.10 Stage 19–21 — Reciprocity and engaged hysteresis

#### 19. Nose-to-nose reciprocity — `_enforce_nose_to_nose_reciprocity` (~1766–1810)

If direction A→B is `nose_to_nose` but reverse is not → replace A→B with `engaged`. Sync preds.

#### 20. Mutually-disengaged reciprocity — `_enforce_mutual_disengaged_reciprocity` (~1719–1763)

Same pattern for `mutually_disengaged` → fallback `engaged`.

#### 21. Final engaged hysteresis — `_apply_final_engaged_hysteresis` (~1813–1861)

Engaged islands of length ≤ `final_engaged_hysteresis_frames` (**12**) bounded by the **same** non-blocked label (`blocked = {engaged, mutually_disengaged}`) merge into that label.

---

## 2. Contact fusion boundary

### 2.1 Binding decision

| Operation | S7 | S8 |
| --- | --- | --- |
| Specialist scoring + hard routing into fused `contact` | **Owns** (dyadic adapter) | **Forbidden** to re-run or re-fuse |
| Optical assist / Schmitt / morphological gap / min bout | Out of scope | **Retain** (post-inference cleaning) |
| Hard-distance gate on contact | Out of scope | **Retain** (passes 1 and 2) |
| Remultiplex `intact`/`occluded` into `prob_contact` when occluded column present | Already done in S7 fusion | **Do not remultiplex** |

### 2.2 Legacy remultiplex semantics (observed)

Pipeline 100 `apply_sota_contact_arbitration` step 0:

```text
if prob_contact_occluded present:
  intact_source = prob_contact_intact if present else prob_contact
  prob_contact = where(is_total_occlusion, prob_contact_occluded, intact_source)
```

S7 already publishes the same hard-routed fused `prob_contact` / `pred_contact` (S7-B §3.1–3.2). Re-running remultiplex on S7 wide outputs that also expose specialist columns is **redundant for valid rows** and **dangerous for invalid routing sentinels** (specialist off-mask zeros / NaNs can be reintroduced as if observed).

### 2.3 Exact S8 changes to avoid double fusion

1. Intake fused `prob_contact` / `pred_contact` from S7 as the contact inference result.  
2. Skip multiplexer block entirely.  
3. Run optical assist → median smooth → Schmitt → morphological gap → min bout on that fused stream.  
4. Preserve specialist columns + `input_valid_contact_*` / `score_origin_contact_*` as **provenance only** (diagnostics / QC), not as alternate science inputs to rewrite fused contact.  
5. Optical assist and Schmitt **must** ignore frames where `input_valid_contact=False` (treat as unavailable; do not boost or trip from sentinel zeros) — see §5.5.  
6. Pass-2 hard-distance gate remains after contact cleaning (still scientifically appropriate).

### 2.4 Provenance columns to retain

| Column family | Role |
| --- | --- |
| `prob_contact`, `pred_contact` | S7 fused intake; then S8-cleaned versions in published states |
| `prob_contact_intact` / `_occluded` (+ preds) | Provenance only |
| `input_valid_contact`, `score_origin_contact`, specialist validity | Validity / sentinel provenance |
| Optional diagnostics | `prob_contact_fused`, `prob_contact_smooth`, Schmitt state — prefer compact rule codes / optional diagnostics artifact (§7.4) rather than widening the primary table unbounded |

---

## 3. Zero-gap discrepancy

### 3.1 Original execution behavior (do not silently reinterpret)

| Item | Observation |
| --- | --- |
| YAML | `post_processing.contact.max_gap_to_fill: 0` |
| Code clamp | `max_gap_to_fill = max(1, int(config.get("max_gap_to_fill", 90)))` in `apply_sota_contact_arbitration` |
| Imposing function | That clamp — **not** `_fill_gaps` |
| Affected channel | **Contact only** at Stage 8 (morphological close). Other detectors are unaffected by this knob |
| Morphology | `rolling(window=W, center=True).max().rolling(window=W).min()` on Schmitt binary |
| W=1 effect | Identity transform → **no morphological gap fill** |
| Parameter meaning | Despite the name “max_gap_to_fill”, contact-stage implementation is a **morphological closing window in frames**, not the explicit “0-run length ≤ G” semantics of `_fill_gaps` |
| Later reconnect | Stage 10 `_fill_small_gaps_and_filter_duration` still applies to `pred_contact` with **6-frame** gap fill and **12-frame** min duration. Stitch/rescue stages operate on `behavior_label` / hierarchy preds and can still reconnect exclusive ethogram bouts; they do **not** rewrite `pred_contact` |

Therefore YAML `0` and a literal morphological window of `1` are **observationally identical** under current code: contact-stage morphological fill is a no-op; contact positives may still be gap-filled later by Stage 10.

Changing the clamp to honor `0` as “skip morphological close” would **not** alter historical outputs relative to today’s YAML-0 runs. Changing it to fill ≥1 missing frame via `_fill_gaps`-style logic **would** alter behavior and is **not** authorized without lab review.

### 3.2 Proposed MVP policy (explicit)

| Policy | Detail |
| --- | --- |
| Config | Keep key `post_processing.contact.max_gap_to_fill` with default **0** |
| Semantics | `0` means **disable contact-stage morphological closing** (parity with observed YAML-0 execution) |
| `W ≥ 1` | Enable morphological close with window `W` (legacy formula) |
| Do not | Auto-promote `0→1` with different scientific meaning; do not reinterpret `0` as “fill one missing frame” via `_fill_gaps` |
| Stage 10 | Remains independent (`gap_frames=6`, `min_duration_frames=12`) for all `pred_*` including contact |
| Rationale | Matches laboratory YAML intent (“no contact morphological fill”) and observed runtime; avoids inventing a one-frame fill that neither YAML nor morphology currently perform |

Document both **original execution** (§3.1) and **MVP policy** (§3.2) in `settings_used.yaml` comments / `qc.json` notes when implemented.

---

## 4. Hardcoded scientific parameter inventory

### 4.1 Already YAML-configured (keep; initial values = legacy)

| Key | Default | Unit / meaning |
| --- | --- | --- |
| `decay_rate` | 0.5 | dimensionless |
| `conf_frames` | 5 | frames |
| `engaged_min_threshold` | 0.0469 | probability |
| `hard_gap_*_distance_norm` | see §1.3 | normalized distance |
| `approach_hard_gap_*_distance_norm` | 0.15 / 0.2 / 0.15 | normalized distance |
| `max_sniff_engaged_gap_frames` | 180 | frames |
| `max_dynamic_engaged_gap_frames` | 120 | frames |
| `max_dynamic_padding_frames` | 120 | frames |
| `min_kinematic_intent_stitch` | 0.5 | intent units |
| `min_rolling_speed_stitch` | 25.0 | speed units |
| `case4_max_kinematic_intent_stitch` | 1.0 | intent |
| `case4_max_rolling_speed_stitch` | 200.0 | speed |
| `case4_initiation_kinematic_intent_stitch` | 0.5 | intent |
| `case4_initiation_rolling_speed_stitch` | 50.0 | speed |
| `max_idle_retract_gap_frames` | 120 | frames |
| `far_apart_idle_distance` | 3.5 | normalized spine–spine |
| `min_mutually_disengaged_frames` | 120 | frames |
| `final_engaged_hysteresis_frames` | 12 | frames |
| `post_processing.contact.*` | see §1.7 / §3 | frames / probs |

Hierarchy S-score thresholds: echo S7 `working_thresholds` into S8 `settings_used` (not silent bundle thresholds).

### 4.2 Hardcoded today — propose config keys (initial values identical)

| Proposed key | Legacy value | Stage |
| --- | --- | --- |
| `hierarchy.layer1_significance_floor` | 0.05 | hierarchy |
| `hierarchy.approach_contact_penalty` | 0.0 | hierarchy |
| `temporal.gap_fill_seconds` | 0.050 | gap fill |
| `temporal.min_bout_seconds` | 0.100 | min duration |
| `postural.approach.rearing_contraction_gt` | 5.0 | approach veto |
| `postural.approach.behind_score_lt` | -0.5 | approach veto |
| `postural.approach.cos_bearing_lt` | -0.2 | approach veto |
| `postural.approach.smear_frames` | 15 | approach veto |
| `postural.approach.escape_projected_vel_gt` | 1.0 | approach veto |
| `postural.retract.stationary_speed_lt` | 15.0 | retract veto |
| `postural.retract.passive_effort_lt` | 0.20 | retract veto |
| `postural.retract.passive_partner_speed_gt` | 80.0 | retract veto |
| `postural.retract.flinch_contraction_gt` | 3.0 | retract veto |
| `postural.retract.flinch_spine_vel_gt` | -5.0 | retract veto |
| `postural.retract.smear_frames` | 15 | retract veto |
| `postural.retract.escape_spine_vel_lt` | -1.0 | retract veto |
| `postural.retract.escape_combined_vel_lt` | -1.0 | retract veto |
| `postural.retract.escape_intent_lt` | -1.0 | retract veto |
| `adaptive.approach.engaged_distance` | 3.0 | adaptive |
| `adaptive.retract.engaged_distance_scale` | 1.5 | adaptive |
| `adaptive.strict_threshold` | 0.5 | adaptive |
| `adaptive.envelope_threshold` | 0.25 | adaptive |
| `adaptive.window_frames` | 15 | adaptive |
| `rescue.approach_prob_hint` | 0.4 | rescue |
| `rescue.retract_prob_hint` | 0.4 | rescue |
| `rescue.moving_intent_abs` | 0.1 | rescue |
| `rescue.moving_smooth_frames` | 12 | rescue |
| `stitch.dynamic.look_ahead_frames` | 15 | dynamic stitch |
| `stitch.dynamic.hard_speed` | 50 | dynamic stitch |
| `stitch.dynamic.hard_intent_factor` | 2.0 | dynamic stitch (`max(1, min_intent*2)`) |

Module `DECAY_RATE = 0.96` is **not** scientific authority when YAML is present; do not expose as competing default. Document as dead code relative to lab YAML.

### 4.3 Scientific vs execution / performance

| Class | Examples |
| --- | --- |
| Scientific | All thresholds, windows, distances, hierarchy floors, contact Schmitt/optical, stitch/rescue/reciprocity/hysteresis |
| Execution / performance | Overlay on/off, chunk sizes, logging verbosity, optional diagnostics artifact flag, parallel group processing |
| Out of S8 MVP science | Overlay video composition; Pipeline 110 clip CRF / GIF settings |

---

## 5. S7 input validity and S6 alignment

### 5.1 Eligible S7 input

| Requirement | Rule |
| --- | --- |
| Status | `completed` or `completed_with_warnings` |
| Discovery | `is_s7_run_eligible_for_downstream` / `find_latest_eligible_s7_run` |
| Artifacts | Canonical `predictions.parquet` **or** both wide views when `emit_wide_views=true`; prefer dual intake with validity from wide/canonical flags |
| Staleness | Refuse `stale_source` / `stale_models` / missing required artifacts |
| Detector coverage (full ethogram MVP) | Dyadic: `contact`, `nose_to_nose`, `mutually_disengaged` (+ `side_by_side` if parallel SBS desired). Focal: `nose_to_body`, `nose_to_tail`, `approach`, `retract` |
| Partial detectors | **Hard fail** for full-ethogram profile; optional future reduced profile is unresolved (§9) |

### 5.2 Eligible linked S6 input

| Requirement | Rule |
| --- | --- |
| Link | S7 `run_meta.source_s6.run_id` / digests |
| Eligibility | Linked S6 still `is_s6_run_eligible_for_downstream` |
| Artifact | `dyadic_features_full.parquet` |
| Lineage | Matching `prepared_sync` / S4 / S5 digests echoed through S7 meta |
| Sample grid | `dense_frame_idx == sample_idx`; directional keys align with focal wide |

### 5.3 Directional expansion and uniqueness

| Check | Rule |
| --- | --- |
| Dyadic undirected uniqueness | At most one row per `(dense_frame_idx, mouse_a, mouse_b)` |
| Focal directional uniqueness | At most one row per `(dense_frame_idx, focal_id, partner_id)` |
| Expansion | Same `(min,max)` pair key as Pipeline 100 |
| Duplicate directional keys after merge | Hard fail |
| K=2 cardinality | Expect ≈ `2 × N_frames` directional rows after expansion |

### 5.4 Required S6 feature columns

**Structural (MVP hard-fail if absent):**

- `distance_nose_i_nose_j_norm`, `distance_nose_i_side_j_norm`, `distance_nose_i_spine_j_norm`, `distance_nose_i_tail_j_norm`, `distance_spine_i_spine_j_norm`
- `cc_count`, `is_total_occlusion`

**Scientific-recommended (warn + skip dependent stages if absent):**

- Veto: `body_contraction_rate_7`, `behind_score_bl_j`, `cos_bearing_i_to_j`, `rolling_mean_intent_15`, `rolling_mean_projected_vel_i_to_j_60`, `rolling_speed_i_15`, `rolling_speed_j_15`, `focal_effort_ratio_15`, `spine_directed_vel_15_i`, `rolling_mean_projected_vel_combined_i_to_j_7`
- Rescue / stitch: `signed_kinematic_intent`, `signed_kinematic_intent_nose_or_neck`, `signed_kinematic_intent_neck`, `rolling_speed_i_60`

### 5.5 Invalid / missing / negative treatment (binding)

Legacy Pipeline 100 has **no** `input_valid` policy. S8 must not invent confidence from sentinels.

| Case | Appearance (S7) | S8 treatment |
| --- | --- | --- |
| `input_valid=False`, `score_origin=routing_sentinel` (contact specialist / fused invalid) | prob often **0**, pred 0 | **Exclude** from optical assist, Schmitt trips, hierarchy margins, and adaptive envelopes as observed scores; do not treat as confident negative |
| Nose-to-body invalid | pred 0; effective may be non-zero after neighbor smooth | Trust flags; use effective **only if** `input_valid_nose_to_body=True`; otherwise unobserved |
| Genuine negative | `input_valid=True`, low prob, pred 0 | Valid negative — eligible for all rules |
| Missing detector column | Absent | Fail coverage check for full profile |
| Legitimate feature NaN | e.g. distance NaN | Skip distance gate for that row; do not coerce to 0 distance |
| Missing dyadic merge | left-join miss | Legacy filled 0; **S8:** if pair should exist under K=2, count as structural warning/fail; do not silently equate to confident negative without QC |
| Focal nose-to-body already smoothed | `prob_nose_to_body` under `legacy_parity` | Consume as-is; **no** second median |
| S7 preds for hierarchy detectors | Present | Hierarchy **recomputes** exclusive preds from probs + S-scores (legacy); S7 preds are intake diagnostics only for those detectors |
| Adaptive 0.5 / 0.25 | Hardcoded | S8-owned arbitration; not S7 inference thresholds |

**Conservative unobserved encoding (testable):** for invalid rows, set arbitration-input probability used by hierarchy/optical/adaptive to NaN-or-masked and force that detector’s contribution to margin/boost **off**, while leaving published provenance columns intact. Prefer explicit `arbitration_input_valid_*` mirrors in QC counts over silent zero fills.

### 5.6 What S8 must not do

- Re-run S7 models or specialist fusion  
- Re-apply S7 working thresholds to rewrite S7 files  
- Double-smooth nose-to-body  
- Blindly remultiplex contact specialists  
- Interpret invalid zero as confident negative  
- Overwrite S7 run directories  

---

## 6. Behavioral semantics

### 6.1 Ethogram label names (exclusive)

| Label | Family | Notes |
| --- | --- | --- |
| `nose_to_nose` | Layer 1 | Reciprocity enforced |
| `nose_to_body` | Layer 1 | Directional |
| `nose_to_tail` | Layer 1 | Directional |
| `approach` | Layer 2 | Directional |
| `retract` | Layer 2 | Directional |
| `mutually_disengaged` | Layer 2 / baseline | Reciprocity enforced |
| `engaged` | Baseline / fallback | Derived; not a detector model |

### 6.2 Priority / exclusivity

- After hierarchy and all sync stages: **exactly one** of the exclusive labels per `(dense_frame_idx, focal_id, partner_id)`.
- Sync priority order: §1.9 Stage 11.
- Undefined legacy init value `"none"` must not survive publish; final default is `engaged` if no pred wins.

### 6.3 Parallel channels (non-exclusive)

| Channel | May be positive while exclusive label is… | Reciprocity |
| --- | --- | --- |
| `pred_contact` / cleaned `prob_contact` | any exclusive label | Symmetric by expansion; not hierarchy |
| `pred_side_by_side` | any exclusive label | Symmetric by expansion; hard-gap gated; not hierarchy |

### 6.4 Directional identity

- Focal behaviors keep `focal_id → partner_id`.
- Dyadic signals expand to both directions with identical values.
- Across directions, labels may differ except where reciprocity forces both sides for `nose_to_nose` and `mutually_disengaged`.
- Across unordered pairs / dyads: independent (no global exclusivity).

### 6.5 Reciprocity

| Label | Rule | Fallback |
| --- | --- | --- |
| `nose_to_nose` | Require both directions | one-sided → `engaged` |
| `mutually_disengaged` | Require both directions | one-sided → `engaged` |
| Other exclusive labels | No forced reciprocity | — |
| Contact / SBS | Symmetric by construction; no one-sided repair stage | — |

### 6.6 Multiple simultaneous channels

Allowed: exclusive `behavior_label` **plus** any combination of parallel contact / SBS positives.  
Forbidden: two exclusive ethogram labels on the same directional frame key.

### 6.7 Unresolved / invalid frames

- After arbitration publish, every directional frame has an exclusive label: hierarchy winners, legacy `engaged` residual for **valid** evidence, or `undefined` when evidence cannot support the legacy interpretation (U2 / P4).
- Invalid S7 inputs alone are never treated as confident negatives. Provisional Sprint-2 `unresolved` is promoted to published `undefined` with `undefined_reason` when no hierarchy winner remains; valid all-negative frames stay `engaged`.

---

## 7. Output schemas

Do **not** modify S7 directories. Publish a new immutable run:

```text
{session_root}/behavioral_arbitration/behavioral_arbitration__YYYYMMDDTHHMMSS/
  behavior_states.parquet
  behavior_bouts.parquet
  run_meta.json
  settings_used.yaml
  qc.json
  processing_log.txt
  # optional behind config flag:
  # arbitration_diagnostics.parquet
```

Schema stamps: `behavioral_arbitration_states_v1`, `behavioral_arbitration_bouts_v1`, `behavioral_arbitration_run_meta_v1`, `behavioral_arbitration_qc_v1`, `behavioral_arbitration_settings_v1`.

### 7.1 `behavior_states.parquet` (primary framewise)

Grain: one row per `(dense_frame_idx, focal_id, partner_id)`.

| Group | Columns |
| --- | --- |
| Identity / time | `schema_version`, `sample_idx`, `dense_frame_idx`, `time_sec`, `exp_name` (compat), `focal_id`, `partner_id`, `mouse_a`, `mouse_b`, `pair_id` |
| Exclusive ethogram | `behavior_label`; `pred_*` / `prob_*` for hierarchy labels including `engaged` |
| Parallel | `pred_contact`, `prob_contact`; `pred_side_by_side`, `prob_side_by_side` when configured |
| Validity echoes | `input_valid_<det>`, `score_origin_<det>` for detectors used |
| Provenance | `source_s7_run_id`, `source_s6_run_id`, `session_root` |
| Compact rule trace | `rule_flags` bitfield **or** short `last_rule_code` — prefer counters in QC + optional diagnostics over an excessively wide frame table |

Missingness: publish NaN only for truly missing features/probs; exclusive preds are 0/1; `behavior_label` always defined at publish.

### 7.2 `behavior_bouts.parquet`

Collapse contiguous identical exclusive labels (and separately optional parallel-channel bouts if configured) per identity grain.

| Field | Convention |
| --- | --- |
| `bout_id` | Stable within run |
| `behavior_name` / `base_behavior_label` | Ethogram or parallel channel name |
| `identity_grain` | `symmetric_pair` \| `directional` |
| IDs | `focal_id`, `partner_id`, `mouse_a_id`, `mouse_b_id` |
| `start_sample_idx`, `end_sample_idx` | **Inclusive** dense/sample indices (Pipeline 110 style) |
| `start_time_sec`, `end_time_sec` | From S6/S7 timeline |
| `bout_duration_frames` | Inclusive count (`end - start + 1` when contiguous) |
| `bout_duration_sec` | `frames / fps` |
| Summaries | `mean_prob`, `max_prob`, `median_prob`, `frac_input_valid` where defined |
| `arbitration_summary` | Compact list/JSON of rule families that touched the bout |
| `source_s7_run_id` | Provenance |

**MVP emit floors:** publish all scientific bouts meeting internal min-duration already enforced; **do not** silently apply Pipeline 110 `min_emit_bout_frames: 10` / `min_emit_bout_seconds: 0.10` inside S8 — leave packaging filters to downstream review (§9).

Overlapping representation: exclusive bouts never overlap on the same directional key; contact/SBS bout tables (if emitted) may overlap exclusive bouts in time.

### 7.3 Run meta / settings / log

Mirror S5–S7 immutable conventions:

- `run_meta.json`: run id, status lifecycle (`running` → `completed` / `completed_with_warnings` / `failed`), S7/S6 digests, detector coverage, schema versions, artifact paths, fps  
- `settings_used.yaml`: frozen arbiter config + threshold echoes + zero-gap policy note + contact-fusion policy  
- `processing_log.txt`: human-readable stage progress and counters  

### 7.4 Optional `arbitration_diagnostics.parquet`

**Recommendation:** **yes, optional** (default off). Long-form stage events (`frame`, direction, `rule_code`, before/after label) preserve rule-level traceability without forcing dozens of stage columns into `behavior_states.parquet`. Always publish aggregate rule-stage change counts in `qc.json`.

### 7.5 Downstream Pipeline 110 compatibility

Framewise view must be able to supply:

`exp_name`, `dense_frame_idx`, `focal_id`, `partner_id`, `behavior_label`

Optional: `pred_side_by_side` for supplemental symmetric emit. Clip packaging / video mapping remain outside S8 MVP.

---

## 8. QC and provenance

### 8.1 Structural blockers (fail run)

- Ineligible / stale S7 or linked S6  
- Missing required detector coverage for selected profile  
- Duplicate undirected or directional keys  
- S6/S7 sample-grid or lineage mismatch  
- Missing required hard-gap / optical feature columns  
- Failure to write required artifacts atomically  

### 8.2 Scientific warnings (completed_with_warnings)

- Missing recommended veto/kinematic columns (dependent stages skipped)  
- High invalid-input fractions  
- Reciprocity repair spikes  
- Unusual ethogram / bout duration distributions  
- Contact/SBS parallel coverage anomalies  

### 8.3 Required QC content

| Block | Metrics |
| --- | --- |
| Coverage | Detector presence; frame × direction counts vs expected K=2 |
| Validity | `n_input_invalid_*`, routing-sentinel exclusions, feature-merge miss rates |
| Ethogram | Label frequencies / % time; published `undefined` counts + reasons (P4; not expected to be zero when invalid evidence exists) |
| Bouts | Counts, duration histograms (frames + seconds) |
| Parallel | Contact / SBS positive rates and overlap with exclusive labels |
| Reciprocity | One-sided repair counts for N–N and MD |
| Rule stages | Change counts per family (`GATE_*`, `VETO_*`, `HIERARCHY_*`, `CONTACT_*`, `GAP_FILL`, `STITCH_*`, `RESCUE_*`, `RECIPROCITY_*`, `HYSTERESIS_*`) |
| Provenance | S7/S6 digests, settings hash, code/config versions, contact-fusion policy, zero-gap policy |

### 8.4 Run lifecycle / staleness

Reuse S5–S7 patterns:

| Classification | Meaning |
| --- | --- |
| `eligible` | completed(+warnings) + artifacts + S7/S6 digests still match live eligible sources |
| `stale_source` | S7 or linked S6 superseded / digest drift |
| `failed_or_incomplete` / `missing_required_artifacts` | as named |

Never mark partial science as eligible.

---

## 9. GUI contract (implemented in S8-4)

One **Subsystem 08 — Behavioral Arbitration** page in **Group 3** beside S7
(`BehavioralArbitrationPage` + `BehavioralArbitrationController`).

| Panel | Behavior |
| --- | --- |
| Session | Same session-root pattern as S6/S7 |
| Eligible S7 | Newest **S8-ready** default + manual picker; detector coverage / joint readiness via `assess_s7_for_s8` |
| S6 readiness | Exact linked S6 only; report required vs recommended feature columns / alignment |
| Config | Load `configs/subsystem_08/*.yaml`; stage-grouped scientific settings; Stage 8 vs Stage 10 gap labels; immutable snapshot at run |
| Launch | Headless `GuiTaskKind.BEHAVIORAL_ARBITRATION` (+ discovery task) via `GuiTaskRunner` |
| Status | Coarse phase progress; no Cancel |
| QC / summaries | Exclusive ethogram vs parallel contact/SBS; engaged vs undefined; bout / reciprocity / rule-stage / provenance |
| Runs | Discover / reopen S8 runs; historical status vs current eligibility |

Controller assembles an immutable request; science stays in `behavioral_arbitration`. Manual review / correction remain downstream — not on this page.

---

## 10. Implementation roadmap

| Sprint | Scope | Dependencies | Testing |
| --- | --- | --- | --- |
| **S8-B** | This contract (done when accepted) | Accepted S8-A | Doc review only |
| **S8-1** | Package skeleton; S7/S6 intake; validity gates; directional expansion; hard-distance gates; feature merge; eligibility/discovery stubs | Eligible S7+S6 fixtures (E-session) | Unit tests: expansion uniqueness, validity masking, gate fires, lineage mismatch hard-fail |
| **S8-2** | Postural vetoes; hierarchy+decay; adaptive approach/retract; contact cleaning **without** remultiplex; pass-2 gates | S8-1; frozen hardcoded→config defaults | Parity vs legacy intermediate preds where fixtures exist; contact no-double-fusion tests; zero-gap morphological policy tests |
| **S8-3** (implemented) | Gap/min-duration; sync; stitch/rescue; reciprocity; hysteresis; finalize `undefined`; `behavior_states` / `behavior_bouts`; QC/provenance publish; discovery | S8-2 | Temporal golden snippets; bout inclusive endpoints; Pipeline 110 column view smoke; QC counters; publish/discovery eligibility |
| **S8-4** (implemented; lab-validated) | Group 3 GUI + task integration + headless E-session extraction + responsiveness remediation | S8-3; PySide6 patterns from S7 | GUI readiness tests green; operator checklist passed 2026-10-28; no S7 mutation |

**MVP acceptance intent:** from an eligible full-detector S7 run + linked S6 features, reproduce Pipeline 100 ethogram / bout semantics under documented parity flags (S7 contact fusion trusted; YAML-0 contact morphological close disabled), without modifying S7 artifacts and without implementing manual bout review.

---

## 11. Unresolved decisions (explicit)

| ID | Topic | Default until reviewed |
| --- | --- | --- |
| U1 | Reduced ethogram mode for partial detector S7 runs | Full-profile hard-fail only |
| U2 | Publish `behavior_label=undefined` for invalid-heavy frames | **Resolved for MVP (S8-3):** publish `undefined` for unresolved invalid evidence (`hierarchy_residual_kind=unresolved_invalid_evidence` / provisional `unresolved` with no hierarchy winner). Bound by P4: valid residual remains `engaged` (including all-negative valid). `undefined_reason` recorded; no `pred_undefined`. |
| U3 | Emit parallel contact/SBS as bout rows in MVP | **Resolved for MVP (S8-3):** exclusive ethogram bouts required; parallel contact / side-by-side bout rows are emitted (`channel=contact` / `side_by_side`, symmetric-pair grain). |
| U4 | Apply Pipeline 110 min-emit floors inside S8 | No — downstream packaging |
| U5 | Overlay video in S8 | Deferred |
| U6 | K>2 animals | Out of MVP |
| U7 | Long-term canonical-only intake (drop wide dependency) | Dual intake in MVP |
| U8 | Promote all §4.2 keys in v1 YAML vs code-locked constants with documented values | Prefer YAML with identical initials for lab tunability |
| U9 | Whether Stage 10 gap fill on `pred_contact` should be independently configurable from hierarchy preds | Keep legacy “all pred_*” behavior unless lab requests split |

---

## 12. Proposed package layout (implementation preview)

```text
src/behavioral_arbitration/
  contracts.py
  config.py
  intake.py          # S7/S6 load, validity, eligibility
  expand.py
  gates.py           # hard-distance + postural vetoes
  hierarchy.py
  adaptive.py
  contact_clean.py   # no remultiplex
  temporal.py        # gap/min, stitch, rescue, reciprocity, hysteresis
  bouts.py
  export.py
  qc.py
  runner.py
  run_discovery.py
configs/subsystem_08/
  behavioral_arbitration_default.yaml
```

Headless entry: `run_behavioral_arbitration_publication(request)` with atomic publish lifecycle matching S6/S7.

---

## 13. Files inspected (checklist)

**Legacy**

- `legacy/pipelines/100_behavioral_arbiter.py`
- `legacy/configs/behavioral_arbiter.yaml`
- `legacy/configs/dyadic_inference.yaml`
- `legacy/configs/90_focal_inference.yaml`
- `legacy/pipelines/110_bout_browser_packager.py`
- `legacy/configs/bout_browser.yaml`

**S7 / S6 / GUI / docs**

- `docs/subsystem_08/architecture_audit.md`
- `docs/subsystem_07/implementation_contract.md` (§3 contact fusion, §5.12 S8 handoff)
- `docs/subsystem_07/sprint_3_artifacts_qc_discovery.md` (§9 S8 inputs)
- `docs/subsystem_06/implementation_contract.md`
- `src/behavioral_inference/export.py`, `validate.py`, `contracts.py`, `run_discovery.py`
- `src/combined_features/pipeline50_post.py` (combined projected velocity)
- `src/ui/main_window.py`, `pages/behavioral_inference_page.py`, `tasks.py`
- E-session S7 wide artifacts under `test_cases/E_full_session_integration/behavioral_inference/`

---

## 14. Stop point / MVP closure

S8-B documentation is accepted. S8-1…S8-4 are implemented. Headless E-session extraction, scientific consistency audit (**47/47**), GUI responsiveness remediation, and **operator interactive GUI acceptance** are complete. S8 MVP is **lab-validated** and resides in **Group 3 beside S7**. Overlay video and manual bout editing remain out of S8 scope. Group 4 packaging/review is delivered as S9+S10.
