# Subsystem 08 — Behavioral Arbitration and Bout Generation

## Architecture Audit (S8-A)

**Status:** Accepted discovery milestone (documentation only). Binding decisions live in S8-B.  
**Date:** 2026-10-26  
**Scope:** Inventory legacy Pipeline 100 arbitration / bout post-processing, map S7 prediction handoff contracts, propose S8 outputs / provenance / Group 3 GUI placement. **No S8 production code in this sprint.**  
**Authority:** Local `legacy/` tree (git-excluded), tracked S7 package `src/behavioral_inference/`, S6 feature contracts, and Group 3 GUI patterns under `src/ui/`.

**Follow-on:** Binding implementation contract — [`implementation_contract.md`](implementation_contract.md) (S8-B). Do not implement S8 until S8-B is accepted.

---

## 0. Preconditions and inspection method

### 0.1 Upstream closure

| Check | Result |
| --- | --- |
| S7 docs | `docs/subsystem_07/` marks S7 Sprint 4 GUI + lab validation; S8 explicitly out of S7 scope |
| S7 package | `src/behavioral_inference/` publishes canonical + wide prediction artifacts with `input_valid` / `score_origin` |
| S7 GUI | Group 3 hosts Behavioral Inference only; soft-nav copy states S8 is not run |
| S8 code | Absent — no `src/behavioral_arbitration/` (or equivalent) yet |
| S8 configs | Absent — no `configs/subsystem_08/` yet |
| Downstream review | Legacy Pipeline 110 + Streamlit bout review app remain separate |

### 0.2 Local legacy availability (critical)

Pipeline 100 / 110 and configs live under git-ignored `legacy/` on this workstation (`.git/info/exclude` contains `/legacy/`). **Do not treat `legacy/` as a production runtime dependency for CI or other clones.** Ported semantics must be encoded into tracked S8 docs/code after S8-B.

| Path | Role |
| --- | --- |
| `legacy/pipelines/100_behavioral_arbiter.py` | Full temporal consensus / arbitration engine (~2413 lines) |
| `legacy/configs/behavioral_arbiter.yaml` | Tunable arbiter / contact post-processing knobs |
| `legacy/configs/dyadic_inference.yaml` | Upstream dyadic working thresholds consumed by hierarchy |
| `legacy/configs/90_focal_inference.yaml` | Upstream focal working thresholds consumed by hierarchy |
| `legacy/configs/global.yaml` | Single `exp_name`, feature `iteration_name`, `fps` |
| `legacy/pipelines/110_bout_browser_packager.py` | Bout index / clip packaging from Pipeline 100 output |
| `legacy/configs/bout_browser.yaml` | Bout emit thresholds, label sets, clip options |
| `legacy/src/bout_review_app/` | Manual bout review UI (downstream of 110) |
| `legacy/src/video/behavioral_overlay.py` | Optional ethogram overlay video |

**Smoke check performed (read-only):** traced `run_behavioral_arbiter()` stage order, config defaults, and S7 wide-view column contracts against E-session S7 artifacts. No arbiter execution and no S8 implementation.

---

## 1. Legacy Pipeline 100 algorithm inventory

### 1.1 Scientific role

Pipeline 100 converts **framewise detector probabilities / preds** from Pipelines 70 (dyadic) and 90 (focal) into:

1. A **directional framewise ethogram** (`behavior_label` + mutually exclusive `pred_*` for hierarchy detectors).
2. **Parallel contact / side-by-side** prediction columns that are cleaned but **not** members of the exclusive ethogram priority.
3. Session ethogram percentages and optional overlay video.

It does **not** train models, re-run classifiers, or package review clips (Pipeline 110).

### 1.2 Inputs (legacy)

| Artifact | Path pattern | Grain |
| --- | --- | --- |
| Dyadic predictions | `data/inference/dyadic/{dyadic_iteration}/{exp}/dyadic_predictions.parquet` | Undirected pair (`focal_id`/`partner_id` = collapse survivors) |
| Focal predictions | `data/inference/focal/{focal_iteration}/{exp}/focal_predictions.parquet` | Directional (`focal_id`, `partner_id`) |
| Feature table(s) | `dyadic_features_full` (preferred) / fallbacks | Directional distances, `cc_count`, occlusion, kinematics, postural veto features |

Required prediction keys: `exp_name`, `dense_frame_idx`, `focal_id`, `partner_id`. NaNs in those keys fail hard.

### 1.3 Configuration authority

| Source | Authority |
| --- | --- |
| `behavioral_arbiter.yaml` → `behavioral_arbiter:` | Primary arbiter knobs (decay, hard gaps, stitch limits, contact post-processing, overlay toggle) |
| `dyadic_inference.yaml` → `working_thresholds` | Hierarchy / stitching S-score thresholds for dyadic detectors |
| `90_focal_inference.yaml` → `working_thresholds` | Hierarchy / stitching S-score thresholds for focal detectors |
| `global.yaml` | Single experiment name, feature iteration, fps (default 120) |
| Hardcoded constants in `100_behavioral_arbiter.py` | Adaptive approach/retract thresholds, postural veto cutoffs, gap/min-duration defaults (50 ms / 100 ms), approach-contact penalty (=0.0), Layer-1 significance floor (0.05) |

**Note:** module constant `DECAY_RATE = 0.96` is overridden by YAML `decay_rate: 0.5` when present.

### 1.4 Active detectors and identity semantics

| Detector | Family | Identity | In exclusive `behavior_label`? |
| --- | --- | --- | --- |
| `nose_to_nose` | Dyadic | Symmetric; expanded to both directions; reciprocity enforced | Yes (Layer 1) |
| `nose_to_body` | Focal | Directional | Yes (Layer 1) |
| `nose_to_tail` | Focal | Directional | Yes (Layer 1) |
| `approach` | Focal | Directional | Yes (Layer 2) |
| `retract` | Focal | Directional | Yes (Layer 2) |
| `mutually_disengaged` | Dyadic | Symmetric; reciprocity enforced | Yes (Layer 2 + baseline) |
| `engaged` | Derived | Directional fallback when no specific winner | Yes (baseline) |
| `contact` | Dyadic | Symmetric; cleaned separately | **No** — parallel `pred_contact` |
| `side_by_side` | Dyadic | Symmetric; hard-gap gated | **No** — parallel `pred_side_by_side` |

### 1.5 Feature dependencies (beyond S7 predictions)

Pipeline 100 merges directional features from the feature iteration (legacy Pipeline 50 / modern S6):

| Feature set | Columns (representative) | Used by |
| --- | --- | --- |
| Hard-gap / optical | `distance_nose_i_nose_j_norm`, `distance_nose_i_side_j_norm`, `distance_nose_i_spine_j_norm`, `distance_nose_i_tail_j_norm`, `distance_spine_i_spine_j_norm`, `cc_count`, `is_total_occlusion` | Distance gates, contact multiplexer / optical assist, sniff-memory separation |
| Rescue kinematics | `signed_kinematic_intent`, `signed_kinematic_intent_nose_or_neck`, `signed_kinematic_intent_neck`, `rolling_speed_i_60` | Dynamic stitching / arc rescue |
| Postural veto | `body_contraction_rate_7`, `behind_score_bl_j`, `cos_bearing_i_to_j`, `rolling_mean_intent_15`, `rolling_mean_projected_vel_i_to_j_60`, `rolling_speed_i_15`, `rolling_speed_j_15`, `focal_effort_ratio_15`, `spine_directed_vel_15_i` | Approach / retract postural vetoes |

Missing feature artifacts degrade gracefully (warnings + skip that gate), except duplicate directional keys which fail hard.

### 1.6 Output (legacy)

| Artifact | Path |
| --- | --- |
| Framewise ethogram | `data/inference/combined/{dyadic_iteration}/{exp}/final_social_predictions.parquet` |
| Optional overlay | `reports/behavioral_overlay/{dyadic_iteration}/{exp}/behavioral_overlay.mp4` |

Primary consumer: Pipeline 110 (`REQUIRED_PREDICTION_COLUMNS` = `exp_name`, `dense_frame_idx`, `focal_id`, `partner_id`, `behavior_label`).

---

## 2. Exact rule and temporal-processing order

Observed orchestration inside `run_behavioral_arbiter()` (plus nested helpers). Units assume **120 Hz** unless noted.

```text
[1] Load configs (global, dyadic thresholds, focal thresholds, arbiter YAML)
[2] Load dyadic + focal prediction parquets
[3] Harmonize
    3a. Expand undirected dyadic → directional spine via (min_id, max_id) pair keys
    3b. Merge hard-gap / rescue / postural-veto feature columns
    3c. Hard-distance probability gates (far → prob/pred = 0)
    3d. Postural approach veto
    3e. Postural retract veto
[4] Resolve / re-threshold / contact-clean
    4a. Hierarchical conflict resolution + decay memory → behavior_label
    4b. Adaptive approach thresholding
    4c. Adaptive retract thresholding
    4d. Contact-only cleaned bouts (optical assist + Schmitt + gap + min duration)
    4e. Re-apply hard-distance gates (contact may have been remultiplexed)
[5] Biological filtering / stitching / reciprocity
    5a. Per-detector gap fill (50 ms) + min duration (100 ms; MD override)
    5b. Synchronize behavior_label from pred_* priority
    5c. Contextual engaged bout stitching
    5d. Dynamic transition stitching
    5e. Idle–retract–idle stitching
    5f. Enforce min mutually_disengaged duration on behavior_label
    5g. Rescue dynamic arcs (approach/retract)
    5h. Approach close-distance hard gate (too close → approach = 0)
    5i. Re-synchronize behavior_label from preds
    5j. Nose-to-nose reciprocity
    5k. Mutually-disengaged reciprocity
    5l. Final engaged hysteresis
[6] Ethogram summary + write final_social_predictions.parquet (+ optional overlay)
```

### 2.1 Stage details (thresholds, units, notes)

#### A. Dyadic → directional expansion (`_expand_dyadic_to_directional`)

- Join key: `(exp_name, dense_frame_idx, (min(focal,partner), max(...)))`.
- Duplicate undirected dyadic keys **fail**.
- Missing dyadic probs/preds fill **0** for both symmetric and non-symmetric detectors.
- Both directions receive the **same** dyadic signal for symmetric detectors.

#### B. Hard-distance gates (`_apply_hard_distance_probability_gates`)

If distance **>** threshold → set `prob_*=0`, `pred_*=0`.

| Detector | Distance column | Default threshold (`*_norm`) |
| --- | --- | --- |
| `nose_to_nose` | `distance_nose_i_nose_j_norm` | 0.3 |
| `nose_to_body` | `distance_nose_i_side_j_norm` | 0.55 |
| `nose_to_tail` | `distance_nose_i_tail_j_norm` | 0.45 |
| `side_by_side` | `distance_spine_i_spine_j_norm` | 0.9 |
| `contact` | `distance_spine_i_spine_j_norm` | 1.2 |

`null` disables a gate. Applied twice (pre-hierarchy and post-contact cleaning).

#### C. Postural vetoes (hardcoded feature cutoffs)

| Veto | Core conditions (simplified) | Smear | Escape hatch |
| --- | --- | --- | --- |
| Approach | rearing (`body_contraction_rate_7 > 5`) **and** partner behind | 15-frame directional rolling max | `rolling_mean_projected_vel_i_to_j_60 > 1` |
| Retract | stationary / passive-receiver / flinch flags | 15-frame directional rolling max | genuine escape velocities / negative intent |

Vetoed rows: `prob_*=0`, `pred_*=0`.

#### D. Hierarchical conflict resolution (`_resolve_hierarchical_conflicts`)

Per directional pair, chronologically:

1. Build effective probs with optional **decay memory** of previous winner:  
   `eff = max(raw, winner_prob * decay_rate)` when streak ≥ `conf_frames` (defaults: decay 0.5, conf 5).  
   Sniff memory clears when `cc_count == 2` (separated).
2. Soft approach penalty while `cc_count == 1`: currently `APPROACH_CONTACT_PENALTY = 0.0` (no-op).
3. Normalized margin  
   `S = (P - T) / (1 - T)` if `P ≥ T`, else 0. Thresholds from focal/dyadic working configs (occlusion-aware dicts supported).
4. **Layer 1** (sniffs): `nose_to_nose`, `nose_to_body`, `nose_to_tail` — winner needs `S > 0.05`.
5. Else **Layer 2** (dynamics): `approach`, `retract`, `mutually_disengaged` — max `S`.
6. Else baseline: if `prob_mutually_disengaged ≥ engaged_min_threshold` (0.0469) → `mutually_disengaged`, else → `engaged`.
7. Write exclusive `pred_*` for hierarchy detectors; set `behavior_label`.

`contact` / `side_by_side` are **not** hierarchy winners.

#### E. Adaptive approach / retract thresholding

Hardcoded (not YAML):

| Detector | Engaged distance | Strict thresh | Adaptive envelope | Window |
| --- | --- | --- | --- | --- |
| Approach | `distance_nose_i_spine_j_norm < 3.0` | 0.5 | rolling-max ≥ 0.25 | 15 frames centered |
| Retract | `< 3.0 * 1.5` | 0.5 | rolling-max ≥ 0.25 | 15 frames centered |

Only when no contact / sniff preds are active. Updates `pred_approach` / `pred_retract` (and `y_*_smoothed` diagnostics).

#### F. Contact-only post-processing (`apply_sota_contact_arbitration` / `get_cleaned_bouts`)

Per directional group, YAML `post_processing.contact`:

| Knob | Lab default | Meaning |
| --- | --- | --- |
| Multiplexer | if `prob_contact_occluded` present | Remultiplex intact/occluded specialist probs by `is_total_occlusion` into `prob_contact` |
| `prob_threshold_high` | intact 0.9731 / occluded 0.4295 | Schmitt high trip |
| `prob_threshold_low` | 0.4 | Schmitt low trip |
| `optical_assist_min` | 0.30 | If `cc_count==1` and prob ≥ min, boost fused prob to high threshold |
| `smoothing_window` | 3 | Centered median on fused contact |
| `max_gap_to_fill` | YAML 0 → **code forces ≥ 1** | Morphological close on contact mask |
| `min_bout_duration` | 6 frames (~50 ms @ 120 Hz) | Drop short contact positives |

Writes back smoothed `prob_contact` and cleaned `pred_contact`. Does **not** overwrite `behavior_label`.

#### G. Gap fill + min duration (`_fill_small_gaps_and_filter_duration`)

Applied to **all** `pred_*` columns per directional pair:

| Parameter | Default |
| --- | --- |
| Gap fill | `0.050 * fps` → **6 frames** (50 ms) |
| Min bout | `0.100 * fps` → **12 frames** (100 ms) |
| Override | `mutually_disengaged` → `min_mutually_disengaged_frames` (**120** frames = 1.0 s) |

#### H. Label synchronization priority (`_synchronize_behavior_label_from_preds`)

First true `pred_*` wins:

1. `nose_to_nose`
2. `nose_to_body`
3. `nose_to_tail`
4. `approach`
5. `retract`
6. `mutually_disengaged`
7. `engaged`  
Default if none: `engaged`.

`contact` / `side_by_side` are ignored by this rebuild.

#### I. Contextual / dynamic / idle stitching

| Stage | What it does | Key limits (YAML) |
| --- | --- | --- |
| Contextual engaged stitch | Fill short `engaged` islands bounded by same sniff/dynamic labels when `cc_count==1` for the island | sniff gap ≤ 180 f; dynamic gap ≤ 120 f |
| Dynamic transition stitch | Cases for approach/retract/contact/idle transitions; kinematic trigger points | padding ≤ 120 f; intent ≥ 0.5; speed ≥ 25 |
| Idle–retract–idle | Collapse short retracts fully surrounded by MD when not far apart | retract gap ≤ 120 f; far apart distance 3.5 |
| Min MD duration (label) | Short MD bouts → `engaged` | 120 frames |
| Rescue dynamic arcs | Backfill high-momentum approach/retract into preceding engaged | case4 peak/init intent & speed |
| Approach close gate | If NN/NB/NT distance **too close**, force approach off | NN 0.15 / NB 0.2 / NT 0.15 |
| N–N reciprocity | One-sided `nose_to_nose` → `engaged` | pair-symmetric |
| MD reciprocity | One-sided `mutually_disengaged` → `engaged` | pair-symmetric |
| Final engaged hysteresis | Tiny engaged islands bounded by same non-idle label merge into that label | ≤ 12 frames |

### 2.2 Overlap / exclusivity semantics (legacy truth)

| Scope | Rule |
| --- | --- |
| Exclusive ethogram | **One** `behavior_label` per `(exp_name, dense_frame_idx, focal_id, partner_id)` |
| Hierarchy preds | Mutually exclusive among Layer1/Layer2/`engaged` after sync |
| Contact | Independent parallel channel; may be positive while label is e.g. `nose_to_body` or `engaged` |
| Side-by-side | Independent parallel channel (Pipeline 110 can emit as supplemental symmetric label) |
| Across directions | Directional labels may differ (A→B approach vs B→A retract). Exceptions: reciprocity forces both sides for `nose_to_nose` and `mutually_disengaged` |
| Across pairs | Independent (no global exclusivity across different dyads) |

---

## 3. S7-to-S8 compatibility matrix

### 3.1 Artifact mapping

| Pipeline 100 expectation | S7 artifact | Mapping notes |
| --- | --- | --- |
| `dyadic_predictions.parquet` | `dyadic_predictions_wide.parquet` | One undirected row per `(dense_frame_idx, mouse_a, mouse_b)`; survivors expose `focal_id=mouse_a`, `partner_id=mouse_b` for `(min,max)` expansion |
| `focal_predictions.parquet` | `focal_predictions_wide.parquet` | Directional spine; includes `is_total_occlusion`, fused + specialist nose-to-body columns |
| Combined long scientific table | `predictions.parquet` (`behavioral_predictions_v1`) | Preferred long-term source of truth; joinable grain + validity |
| Contact specialists | `specialist_predictions.parquet` + wide specialist columns | Optional remultiplex; prefer gating on `input_valid_*` |
| Feature merges | Eligible **S6** `dyadic_features_full.parquet` via S7 `source_s6_run_id` | Hard gaps / vetoes / optical assist / kinematics |
| Working thresholds | S7 `settings_used.yaml` / `model_manifest.json` / per-row `threshold` | Do **not** silently re-threshold S7 effective probs for ordinary detectors |

### 3.2 Column-level intake rules

| Need | Use | Avoid |
| --- | --- | --- |
| Contact logical score | `prob_contact` / `pred_contact` when `input_valid_contact` (or canonical fused row valid) | Treating `routing_sentinel` zeros as confident negatives without the flag |
| Contact remultiplex | Prefer S7 already-fused contact; if remultiplexing, use specialist probs **only where** `input_valid_*=True` | Blind `np.where(occlusion, occ, intact)` on sentinel zeros |
| Nose-to-body parity | `prob_nose_to_body` (= S7 `prob_effective`, may be 5-frame median under `legacy_parity`) | Re-smoothing already-smoothed probs; using invalid smoothed neighbors as observed scores |
| Nose-to-body diagnostics | `prob_nose_to_body_raw`, specialist columns | Overwriting S7 artifacts |
| Ordinary detectors | `prob_*` / `pred_*` + thresholds from S7 provenance | Inventing new thresholds unless S8 config explicitly overrides for **arbitration-only** adaptive stages |
| Invalid routes | Gate on `input_valid_*` and `score_origin_* ∈ {routing_sentinel}` | Interpreting fill `0` / `NaN` as model-observed negatives |

### 3.3 Invalid routed predictions (non-negotiable)

S7 contracts already encode:

| Case | Appearance | S8 duty |
| --- | --- | --- |
| Contact specialist off-mask | prob **0**, pred 0, `input_valid=False`, `score_origin=routing_sentinel` | Exclude from remultiplex / optical logic as observed score |
| Nose-body specialist off-mask | prob **NaN**, pred 0, same flags | Same |
| Fused invalid (rare) | pred 0; effective may be non-zero after nose-body smoothing | Trust flags, not raw magnitude |
| Ordinary detectors | Always valid in current S7 | Still carry flags for schema stability |

**Do not blindly re-threshold S7 smoothed probabilities.** Adaptive approach/retract stages in Pipeline 100 re-binarize using hardcoded 0.5 / 0.25 envelopes — those are **S8 arbitration rules**, distinct from S7 working thresholds. Document them as S8-owned; do not mutate S7 files.

### 3.4 Coverage / eligibility

S7 allows detector subsets. S8 must refuse or degrade when required detectors / features are missing:

| Minimum for full ethogram parity | Detectors |
| --- | --- |
| Dyadic | `contact`, `nose_to_nose`, `mutually_disengaged` (+ `side_by_side` if parallel channel desired) |
| Focal | `nose_to_body`, `nose_to_tail`, `approach`, `retract` |
| Features | hard-gap distances + `cc_count` + `is_total_occlusion`; veto/kinematic columns strongly recommended |

Reuse S7 discovery classifications (`eligible`, `stale_source`, `stale_models`, …) and add S8-specific coverage checks.

### 3.5 What S7 must not / does not provide

S7 intentionally omits: undirected→directional expansion for ethograms, hard-gap gates, postural vetoes, hierarchy, adaptive re-thresholding, contact optical-assist cleaning, gap/min-duration stitching, reciprocity, engaged hysteresis, bout tables, overlay packaging.

---

## 4. Behavioral identity and overlap semantics (proposed S8)

Preserve legacy semantics unless an explicit S8-B decision changes them.

| Behavior | Grain | Overlap with exclusive label | Reciprocity |
| --- | --- | --- | --- |
| Exclusive ethogram labels | Directional pair | Mutually exclusive among themselves | N–N and MD force both directions |
| `contact` | Directional rows carrying symmetric signal | May overlap exclusive label | Symmetric by expansion |
| `side_by_side` | Same | May overlap exclusive label | Symmetric by expansion |
| Across dyads | Independent | No global exclusivity | N/A |

**Proposal:** publish both:

1. **Exclusive framewise ethogram** (`behavior_label` + hierarchy `pred_*`).
2. **Parallel channels** (`pred_contact`, optionally `pred_side_by_side`) with clear schema documentation that they are non-exclusive.

Bout collapse should be identity-aware:

- Symmetric bouts: collapse on unordered pair + label (Pipeline 110 style).
- Focal bouts: keep direction (`focal_id` → `partner_id`).

---

## 5. Proposed output schemas

Do **not** modify or overwrite S7 run directories. Write a new immutable S8 run folder.

### 5.1 Proposed run layout

```text
{session_root}/behavioral_arbitration/behavioral_arbitration__YYYYMMDDTHHMMSS/
  framewise_states.parquet          # directional exclusive ethogram + parallel channels
  bouts.parquet                     # bout-level detections
  arbitration_decisions.parquet     # optional long-form rule diagnostics (or embedded columns)
  ethogram_summary.json             # % time / counts
  run_meta.json                     # S7/S6 provenance, digests, schema versions
  settings_used.yaml                # frozen arbiter config + threshold echoes
  qc.json                           # structural + scientific QC
  processing_log.txt
  # optional later:
  # overlay.mp4                     # ethogram video (parity with Pipeline 100)
```

Suggested schema stamps: `behavioral_arbitration_framewise_v1`, `behavioral_arbitration_bouts_v1`, `behavioral_arbitration_qc_v1`, `behavioral_arbitration_run_meta_v1`.

### 5.2 Framewise contract (`framewise_states.parquet`)

| Column group | Fields |
| --- | --- |
| Keys | `schema_version`, `sample_idx`, `dense_frame_idx`, `time_sec`, `focal_id`, `partner_id`, `pair_id` / `mouse_a` / `mouse_b` |
| Exclusive state | `behavior_label`, `pred_<hierarchy...>`, `prob_<hierarchy...>` (post-arbitration effective where applicable) |
| Parallel channels | `pred_contact`, `prob_contact` (+ optional SBS); never overwrite exclusive label |
| Supporting S7 echoes | Selected pre-arbitration probs/preds **or** foreign keys into S7 rows |
| Validity / missingness | `input_valid_<det>` echoes, `feature_missing_*` flags, `score_origin_*` echoes |
| Arbitration | `arbitration_stage` / decision codes (or join to decisions table) |
| Provenance | `source_s7_run_id`, `source_s6_run_id`, `session_root` |

Grain: one row per `(dense_frame_idx, focal_id, partner_id)` — same as Pipeline 100 combined table.

### 5.3 Bout contract (`bouts.parquet`)

Align with Pipeline 110 scientific needs (clip packaging can remain downstream):

| Field | Meaning |
| --- | --- |
| `bout_id` | Stable id within run |
| `behavior_name` / `base_behavior_label` | Scientific label |
| `identity_grain` | `symmetric_pair` \| `directional` |
| `focal_id`, `partner_id`, `mouse_a_id`, `mouse_b_id` | Identity |
| `start_sample_idx`, `end_sample_idx` | Inclusive dense / sample indices |
| `start_time_sec`, `end_time_sec` | From S6/S7 timeline |
| `bout_duration_frames`, `bout_duration_sec` | Inclusive duration |
| `mean_prob`, `max_prob`, `median_prob` | Summaries over supporting frames (where defined) |
| `frac_input_valid` | Missingness / routing coverage inside bout |
| `arbitration_summary` | Compact code list / JSON of applied rule families |
| `source_s7_run_id` | Provenance |

Pipeline 110 additionally needs video mapping fields; those can stay in a future packaging subsystem rather than blocking S8 MVP.

### 5.4 QC contract (minimum)

- Structural: grain uniqueness; exclusive label integrity; detector coverage vs config.
- Scientific: positive rates / ethogram %; reciprocity repair counts; gate hit counts; stitch/rescue frame counts; contact cleaned bout stats.
- Validity: counts of routing-sentinel frames excluded; feature-merge miss rates.
- Provenance: S7 eligibility snapshot, S6 run id, settings hash, code/config versions.

### 5.5 Downstream Pipeline 110 interface (boundary)

Pipeline 110 requires framewise:

`exp_name`, `dense_frame_idx`, `focal_id`, `partner_id`, `behavior_label`

Optional supplemental: `pred_side_by_side` for symmetric presentation. Bout emit floors in `bout_browser.yaml`: `min_emit_bout_frames: 10`, `min_emit_bout_seconds: 0.10`.

S8 should remain able to emit a **Pipeline-110-compatible view** without owning clip extraction or manual review.

---

## 6. Provenance and QC design

### 6.1 Principles

1. S7 artifacts are **immutable inputs**; S8 never rewrites them.
2. Every final frame/bout must trace to: S7 run id, detector scores used, feature columns used, and ordered S8 rule decisions.
3. Prefer **reproducible rule codes** over free-text narratives.
4. Keep diagnostic verbosity configurable (summary counters always; per-frame decision log optional).

### 6.2 Proposed decision vocabulary (illustrative)

| Code family | Examples |
| --- | --- |
| `GATE_HARD_DISTANCE` | detector, column, threshold, fired |
| `VETO_POSTURAL_APPROACH` / `VETO_POSTURAL_RETRACT` | fired + escape hatch |
| `HIERARCHY_WIN` | winner, layer, margins snapshot |
| `DECAY_CARRY` | prior winner, streak, decayed prob |
| `ADAPTIVE_APPROACH` / `ADAPTIVE_RETRACT` | envelope trip |
| `CONTACT_OPTICAL` / `CONTACT_SCHMITT` / `CONTACT_MIN_DUR` | contact pipeline |
| `GAP_FILL` / `MIN_DURATION` | detector, frames |
| `STITCH_*` / `RESCUE_*` / `RECIPROCITY_*` / `HYSTERESIS_ENGAGED` | temporal repairs |

### 6.3 Traceability pattern

- Framewise: store `source_prediction_keys` (S7 grain + detector) and `rule_codes[]` / bitflags for stages that changed the state.
- Bout-level: aggregate rule codes + mean/max supporting probs + validity fraction.
- Run-level: `settings_used.yaml` + counter map in `qc.json` (frames gated, stitched, rescued, reciprocity-replaced).

### 6.4 Explainability without bloat

Default publish: framewise states + bout table + QC counters.  
Advanced/debug: optional `arbitration_decisions.parquet` (long-form stage events) behind a config flag — analogous to S7 specialist separation.

---

## 7. Architecture and GUI proposal

### 7.1 Backend package (proposed)

```text
src/behavioral_arbitration/          # proposed name
  contracts.py                       # schemas, statuses, rule codes
  config.py                          # arbiter YAML + threshold policy
  intake.py                          # S7 wide/canonical + S6 feature merge; validity gates
  expand.py                          # dyadic → directional
  gates.py                           # hard-distance + postural vetoes
  hierarchy.py                       # conflict resolution + decay memory
  contact_clean.py                   # optical assist / Schmitt / min bout
  temporal.py                        # gap/min-duration, stitches, reciprocity, hysteresis
  bouts.py                           # collapse framewise → bouts
  export.py                          # immutable run artifacts
  qc.py
  runner.py / run_discovery.py
```

Headless entry: `run_behavioral_arbitration_publication(request)` mirroring S6/S7 publication lifecycle (`running` → validate → atomic publish → `completed` / `failed`).

### 7.2 Group 3 GUI (documentation only — do not implement yet)

Place a dedicated **Subsystem 08 — Behavioral Arbitration** page in **Group 3** beside S7.

| Panel | Behavior |
| --- | --- |
| Session / project | Same session-root pattern as S6/S7 |
| Eligible S7 source | `find_latest_eligible_s7_run` + manual eligible picker; show detector coverage |
| Feature readiness | Resolve linked S6 run; report hard-gap / veto / kinematic column presence |
| Arbitration config | Load default `configs/subsystem_08/*.yaml`; show thresholds / stitch limits; advanced overrides |
| Run | Headless task via new `GuiTaskKind.BEHAVIORAL_ARBITRATION` |
| Progress / log | `GuiTaskRunner` pattern |
| QC | Ethogram summary, gate/stitch counters, validity exclusions |
| Completed runs | Discover S8 runs; reopen; mark stale if S7/S6 superseded |

Controller should assemble an immutable request object; **science stays in the package**, not the page.

### 7.3 Manual bout correction — placement decision

**Recommendation: keep manual bout correction / review outside S8.**

| Concern | Why separate |
| --- | --- |
| Legacy precedent | Pipeline 100 = automation; Pipeline 110 + `bout_review_app` = packaging + human review |
| Product boundary | S8 should publish reproducible machine states/bouts; human edits need their own provenance + non-destructive overlays |
| GUI complexity | Correction UIs need video sync, annotation stores, and export packs unrelated to arbitration science |

Treat review/correction as a **downstream subsystem** (future S9 / Bout Review) that consumes S8 `bouts.parquet` + framewise states. S8 may later emit packaging hooks, but should not own annotation editing in MVP.

---

## 8. Unresolved scientific decisions

These require explicit lab/product choices in S8-B (defaults below preserve legacy parity):

1. **Contact remultiplex vs trust S7 fusion:** Pipeline 100 remultiplexes specialists when present and can diverge from S7 fused `prob_contact`. Prefer S7 fusion + cleaning-only, or keep remultiplex for parity?
2. **`max_gap_to_fill: 0` vs code `max(1, …)`:** YAML says 0; runtime forces ≥1. Confirm intended lab behavior and encode deliberately.
3. **Adaptive approach/retract hardcoded thresholds (0.5 / 0.25 / distance 3.0):** promote to YAML or freeze as parity constants?
4. **Postural veto cutoffs:** currently hardcoded; promote to config or leave code-locked?
5. **`decay_rate` 0.5 (YAML) vs 0.96 (module default):** confirm YAML is scientific authority.
6. **Exclusive ethogram vs parallel contact:** keep dual-channel semantics, or eventually fold contact into hierarchy?
7. **Nose-to-body smoothing ownership:** S7 already may smooth under `legacy_parity`. S8 must not double-smooth; confirm intake always uses S7 effective column for parity mode.
8. **Partial detector runs:** fail-fast vs reduced ethogram modes?
9. **K>2 animals:** reciprocity / pair expansion beyond dyads needs explicit policy (legacy focused on K=2).
10. **Overlay video in S8 MVP or defer?**
11. **Canonical long-form vs wide-view intake preference long-term:** Sprint guidance allows either; recommend dual intake with validity from canonical/wide flags.
12. **Bout emit floors:** inherit Pipeline 110 `min_emit_bout_*` inside S8 bout export, or leave filtering to review packaging?

---

## 9. Suggested implementation roadmap

| Sprint | Deliverable |
| --- | --- |
| **S8-A** | This architecture audit (done) |
| **S8-B** | Binding implementation contract: schemas, rule order, validity gates, eligibility, provenance codes, config surface |
| **S8-1** | Package skeleton + S7/S6 intake + expansion + hard gates (headless; no GUI) |
| **S8-2** | Hierarchy, adaptive stages, contact cleaning — parity tests vs legacy intermediate preds where fixtures exist |
| **S8-3** | Temporal stitch/reciprocity/hysteresis + bout collapse + QC/provenance publish |
| **S8-4** | Group 3 GUI page + task integration + laboratory acceptance on E-session (S7 → S8) |

**Acceptance intent for S8 MVP:** from an eligible full-detector S7 run + linked S6 features, reproduce Pipeline 100 scientific ethogram / bout semantics within documented parity flags, without modifying S7 artifacts and without implementing manual bout review.

---

## 10. Files inspected (checklist)

**Legacy**

- `legacy/pipelines/100_behavioral_arbiter.py`
- `legacy/configs/behavioral_arbiter.yaml`
- `legacy/configs/dyadic_inference.yaml`
- `legacy/configs/90_focal_inference.yaml`
- `legacy/pipelines/110_bout_browser_packager.py`
- `legacy/configs/bout_browser.yaml`
- `legacy/src/bout_review_app/` (boundary)
- `legacy/src/video/behavioral_overlay.py` (boundary)

**S7 / S6 / GUI**

- `docs/subsystem_07/architecture_audit.md` (§7 S7/S8 boundary)
- `docs/subsystem_07/implementation_contract.md` (§5.12 handoff)
- `docs/subsystem_07/sprint_3_artifacts_qc_discovery.md` (§9 S8 input requirements)
- `docs/subsystem_07/sprint_4_gui_integration.md`
- `src/behavioral_inference/contracts.py`
- `src/behavioral_inference/export.py`
- `src/behavioral_inference/results.py`
- `src/behavioral_inference/validate.py`
- `src/behavioral_inference/qc.py`
- `src/behavioral_inference/run_discovery.py`
- `configs/subsystem_07/behavioral_inference_default.yaml`
- `src/ui/main_window.py` (Group 3)
- `src/ui/pages/behavioral_inference_page.py`
- `src/ui/controllers/behavioral_inference_controller.py`
- `src/combined_features/` feature columns used by gates (spot-check)
- E-session S7 run under `test_cases/E_full_session_integration/behavioral_inference/`

---

## 11. Stop point

S8-A is documentation only. Binding schemas, rule order, validity gates, and sprint plan are in [`implementation_contract.md`](implementation_contract.md) (S8-B). **Do not implement** production arbitration / GUI code until S8-B is accepted.
