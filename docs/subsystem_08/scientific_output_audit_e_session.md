# S8 Scientific Output Audit — E Full Session

**Generated:** 2026-10-28  
**Mode:** read-only inspection of published artifacts (no result files modified).  
**Context:** Post Sprint-4 GUI responsiveness remediation; arbitration formulas unchanged.  
**MVP closure:** Preserved as laboratory scientific-acceptance evidence for S8 MVP complete / lab-validated (operator GUI acceptance 2026-10-28). **47/47 PASS**. Not an independent video ground-truth validation.

## Artifact roots

| Layer | Path |
| --- | --- |
| S8 | C:\behavior_suite\test_cases\E_full_session_integration\behavioral_arbitration\behavioral_arbitration__20261028T133953 |
| S7 | C:\behavior_suite\test_cases\E_full_session_integration\behavioral_inference\behavioral_inference__20261026T163224 |
| S6 | C:\behavior_suite\test_cases\E_full_session_integration\combined_features\combined_features__20261025T173244 |

## A. Framewise states

### A.1 Row counts and key uniqueness

| Metric | Observed | Expected | Result |
| --- | --- | --- | --- |
| Unique dense_frame_idx (synchronized samples) | 36528 | 36528 | **PASS** — got 36528 |
| Directional rows | 73056 | 73056 | **PASS** — got 73056 |
| QC metrics.n_state_rows | 73056 | 73056 | **PASS** |

- Unique `(dense_frame_idx, focal_id, partner_id)` keys: **73056**; duplicate key rows: **0** — **PASS**

### A.2 Exclusive label integrity

| Check | Observed | Result |
| --- | --- | --- |
| Invalid vocabulary labels | (none) | **PASS** |
| Exactly one exclusive pred_* == 1 per row | 73056/73056 (zero=0, multi=0) | **PASS** |
| behavior_label matches its pred_* flag | match=73056, mismatch=0 | **PASS** |
| engaged distinct from undefined | n_engaged=16574, n_undefined=0 | **PASS** |
| n_undefined / n_unresolved | 0 / 0 | **PASS** — expected for this completed ethogram session |

- Non-empty `undefined_reason` values: **0** (empty on all rows when n_undefined=0).

### A.3 Time alignment

| Metric | Value | Result |
| --- | --- | --- |
| max \|time_sec − dense_frame_idx/120\| | 5.68434e-14 | **PASS** — max_dt=5.684341886080802e-14 |
| rows with time misalignment > 1e-9 | 0 | **PASS** |
| TARGET_FPS / settings fps | 120.0 / 120.0 | **PASS** |

### A.4 S7 validity and score_origin preservation

| Detector | All sample cols present | n_null input_valid | n_null score_origin | score_origin values | Result |
| --- | --- | --- | --- | --- | --- |
| contact | True | 0 | 0 | fusion_routed | **PASS** — ok |
| nose_to_nose | True | 0 | 0 | model_observed | **PASS** — ok |
| mutually_disengaged | True | 0 | 0 | model_observed | **PASS** — ok |
| side_by_side | True | 0 | 0 | model_observed | **PASS** — ok |
| nose_to_body | True | 0 | 0 | model_observed | **PASS** — ok |
| nose_to_tail | True | 0 | 0 | model_observed | **PASS** — ok |
| approach | True | 0 | 0 | model_observed | **PASS** — ok |
| retract | True | 0 | 0 | model_observed | **PASS** — ok |

Sample columns checked per detector: `input_valid_*`, `score_origin_*`, `s7_pred_*`, `s7_prob_*` (plus `invalid_reason_*` / `threshold_*` where published).

### A.5 Parallel contact / SBS independence from exclusive labels

- Exclusive `behavior_label` never equals a parallel channel name: **0** such rows — **PASS**
- Contact positives: **27796** (38.05%); SBS positives: **1588** (2.17%).

Contact × exclusive label:

| behavior_label | contact=0 | contact=1 |
| --- | --- | --- |
| approach | 10055 | 917 |
| engaged | 7413 | 9161 |
| mutually_disengaged | 18200 | 304 |
| nose_to_body | 832 | 8028 |
| nose_to_nose | 626 | 2470 |
| nose_to_tail | 492 | 3335 |
| retract | 7642 | 3581 |

SBS × exclusive label:

| behavior_label | sbs=0 | sbs=1 |
| --- | --- | --- |
| approach | 10941 | 31 |
| engaged | 16195 | 379 |
| mutually_disengaged | 18504 | 0 |
| nose_to_body | 8444 | 416 |
| nose_to_nose | 2440 | 656 |
| nose_to_tail | 3827 | 0 |
| retract | 11117 | 106 |

- Parallel contact positives under `mutually_disengaged` exclusive: **304** (allowed — channels are independent). **PASS** — structural independence confirmed

### A.6 Label counts / proportions by direction (focal→partner)

#### Direction 0→1 (n=36528)

| label | count | proportion |
| --- | --- | --- |
| nose_to_nose | 1548 | 4.24% |
| nose_to_body | 7270 | 19.90% |
| nose_to_tail | 3827 | 10.48% |
| approach | 8219 | 22.50% |
| retract | 2493 | 6.82% |
| mutually_disengaged | 9252 | 25.33% |
| engaged | 3919 | 10.73% |
| undefined | 0 | 0.00% |
| unresolved | 0 | 0.00% |

- Parallel: contact+=13898 (38.05%), SBS+=794 (2.17%).

#### Direction 1→0 (n=36528)

| label | count | proportion |
| --- | --- | --- |
| nose_to_nose | 1548 | 4.24% |
| nose_to_body | 1590 | 4.35% |
| nose_to_tail | 0 | 0.00% |
| approach | 2753 | 7.54% |
| retract | 8730 | 23.90% |
| mutually_disengaged | 9252 | 25.33% |
| engaged | 12655 | 34.64% |
| undefined | 0 | 0.00% |
| unresolved | 0 | 0.00% |

- Parallel: contact+=13898 (38.05%), SBS+=794 (2.17%).

## B. Bouts

### B.1 Published bout inventory

| channel | n_bouts | QC by_channel |
| --- | --- | --- |
| exclusive | 538 | 538 |
| contact | 60 | 60 |
| side_by_side | 11 | 11 |
- Total published bouts: **609** (QC n_bouts=609) — **PASS**

### B.2 Inclusive start/end, n_frames, duration_sec

| Check | OK | Bad | Result |
| --- | --- | --- | --- |
| bout_duration_frames == end_sample_idx − start_sample_idx + 1 | 609 | 0 | **PASS** |
| bout_duration_sec == n_frames / 120 | 609 | 0 | **PASS** |

### B.3 Bout ↔ framewise state correspondence

- Exclusive bout spans match contiguous identical `behavior_label`: **538/538** — **PASS**
- Parallel contact/SBS bout spans match contiguous positive channel frames: **71/71** — **PASS**

### B.4 Overlap / duplicate checks

- Overlapping exclusive bouts within same ordered direction: **0** — **PASS**
- Duplicate / non-symmetric parallel bout issues: **0** — **PASS**

### B.5 Splitting at transitions / discontinuities

- Adjacent exclusive bouts with identical label (should have been merged): **0** — **PASS**
- `n_undefined=0` so no undefined-boundary bout splits in this run (exclusive bouts still split on label transitions and frame discontinuities).

### B.6 Bout counts and duration distributions

| channel | behavior | n | min_frames | median_frames | mean_frames | max_frames | total_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contact | contact | 60 | 12 | 143.5 | 231.633 | 1439 | 115.817 |
| exclusive | approach | 108 | 5 | 60 | 101.593 | 711 | 91.4333 |
| exclusive | engaged | 139 | 1 | 99 | 119.237 | 645 | 138.117 |
| exclusive | mutually_disengaged | 38 | 18 | 398 | 486.947 | 1630 | 154.2 |
| exclusive | nose_to_body | 82 | 12 | 73 | 108.049 | 639 | 73.8333 |
| exclusive | nose_to_nose | 38 | 15 | 82 | 81.4737 | 172 | 25.8 |
| exclusive | nose_to_tail | 53 | 9 | 42 | 72.2075 | 544 | 31.8917 |
| exclusive | retract | 80 | 9 | 95 | 140.287 | 583 | 93.525 |
| side_by_side | side_by_side | 11 | 14 | 50 | 72.1818 | 179 | 6.61667 |

### B.7 Independent reconstruction via `extract_behavior_bouts`

| Metric | Published | Reconstructed | Result |
| --- | --- | --- | --- |
| n_bouts | 609 | 609 | **PASS** |
| Content match (excl. bout_id) | — | exact | **PASS** — exact |

## C. QC and provenance

### C.1 Warnings (interpreted; not assumed harmless)

- QC `structural_ok`: **True**; `warning_count`: **1**; `failure_count`: **0**; run_meta.status: **completed_with_warnings**.

| Warning text | Severity class | Interpretation |
| --- | --- | --- |
| contact morphological closing disabled (max_gap_to_fill=0; Stage 10 gap fill remains independent). | expected_config | Expected structural / configuration warning — Stage 8 contact morphological closing intentionally disabled (`max_gap_to_fill=0`); Stage 10 exclusive-label gap fill remains independent. Legitimate configuration missingness of morph-close, not data corruption. |

- Failures list empty — **PASS**

**Note:** Status `completed_with_warnings` is **not** treated as harmless a priori. The sole warning here documents intentional Stage-8 morph-close disablement (`max_gap_to_fill=0`), which matches `settings_used.yaml` / run settings snapshot. It does not indicate undefined labels, digest mismatch, or structural publish failure.

### C.2 Current S8 downstream eligibility

| API | Result | Detail |
| --- | --- | --- |
| validate_s8_run_for_downstream | OK | returned provenance payload |
| is_s8_run_eligible_for_downstream | True | **PASS** |

- Validate payload keys: `['dyadic_features_sha256', 'predictions_sha256', 's8_run_dir', 'session_root', 'source_s6_run_dir', 'source_s6_run_id', 'source_s7_run_dir', 'source_s7_run_id', 'status']`

### C.3 S7/S6 linkage and live digests

| Artifact | Recorded SHA256 (prefix) | Live SHA256 (prefix) | Path match | Result |
| --- | --- | --- | --- | --- |
| S7 predictions | ad24272724d3… | ad24272724d3… | True | **PASS** — sha+path |
| S7 dyadic_wide | 60bb0b3c11c5… | 60bb0b3c11c5… | True | **PASS** — sha+path |
| S7 focal_wide | c2dcbc576910… | c2dcbc576910… | True | **PASS** — sha+path |
| S6 dyadic_features | 66b67152bf44… | 66b67152bf44… | True | **PASS** — sha+path |

| Field | Value |
| --- | --- |
| S7 run_id | behavioral_inference__20261026T163224 |
| S7 run_dir | C:\behavior_suite\test_cases\E_full_session_integration\behavioral_inference\behavioral_inference__20261026T163224 |
| S7 eligible_at_publish | True |
| S6 run_id | combined_features__20261025T173244 |
| S6 run_dir | C:\behavior_suite\test_cases\E_full_session_integration\combined_features\combined_features__20261025T173244 |
| S6 eligible_at_publish | True |
| Linked paths == audit targets | **PASS** |

### C.4 Stage completion, reciprocity, reconstruction metrics

- Diagnostic stages recorded: **38**; stage_rule_codes_missing: **(none)** — **PASS**

Observed stage rule codes:

`EXPAND_DYADIC`, `VALIDITY_MASK`, `MERGE_S6`, `GATE_HARD_DISTANCE`, `VETO_APPROACH`, `VETO_RETRACT`, `HIERARCHY_WIN`, `DECAY_CARRY`, `HIERARCHY_RESIDUAL`, `ADAPTIVE_APPROACH`, `ADAPTIVE_RETRACT`, `CONTACT_OPTICAL`, `CONTACT_SCHMITT`, `CONTACT_MORPH`, `CONTACT_MIN_DUR`, `GATE_HARD_DISTANCE_PASS2`, `GAP_FILL`, `MIN_DURATION`, `LABEL_SYNC`, `STITCH_CONTEXTUAL`, `STITCH_DYNAMIC`, `STITCH_IDLE_RETRACT`, `MIN_MD_LABEL`, `RESCUE_DYNAMIC`, `GATE_APPROACH_CLOSE`, `RECIPROCITY_NOSE_TO_NOSE`, `RECIPROCITY_MUTUALLY_DISENGAGED`, `HYSTERESIS_ENGAGED`, `LABEL_FINALIZE`

| Reciprocity repair | Count |
| --- | --- |
| nose_to_nose | 428 |
| mutually_disengaged | 2718 |

Framewise `label_reconstruction` counts (stage-level reconstruction modes):

| mode | n_frames | fraction |
| --- | --- | --- |
| selected | 53317 | 72.98% |
| stitched | 8428 | 11.54% |
| synced | 4878 | 6.68% |
| reciprocity_fallback | 3146 | 4.31% |
| min_duration | 1190 | 1.63% |
| gap_filled | 1138 | 1.56% |
| rescued | 881 | 1.21% |
| hysteresis | 78 | 0.11% |

Selected QC `rule_change_counts` (temporal / finalize-adjacent):

| rule | n_changes |
| --- | --- |
| GAP_FILL | 3227 |
| STITCH_CONTEXTUAL | 2413 |
| STITCH_DYNAMIC | 5572 |
| STITCH_IDLE_RETRACT | 1351 |
| RESCUE_DYNAMIC | 881 |
| RECIPROCITY_NOSE_TO_NOSE | 428 |
| RECIPROCITY_MUTUALLY_DISENGAGED | 2718 |
| HYSTERESIS_ENGAGED | 78 |
| MIN_DURATION | 4696 |
| LABEL_SYNC | 7648 |

| QC metric | Value | States cross-check | Result |
| --- | --- | --- | --- |
| n_undefined | 0 | 0 | **PASS** |
| n_unresolved | 0 | 0 | **PASS** |
| label_counts sum | 73056 | 73056 | **PASS** |

### C.5 Run metadata and settings snapshot highlights

| Field | Value |
| --- | --- |
| run_id | behavioral_arbitration__20261028T133953 |
| run_purpose | lab_validation |
| status | completed_with_warnings |
| created_at | 2026-10-28T15:40:09+02:00 |
| completed_at | 2026-10-28T15:41:15+02:00 |
| fps | 120 |
| detector_coverage_profile | full_ethogram |
| contact.max_gap_to_fill | 0 |
| temporal.gap_fill_seconds | 0.05 |
| temporal.min_bout_seconds | 0.1 |
| final_engaged_hysteresis_frames | 12 |
| settings_used.yaml schema | behavioral_arbitration_settings_v1 |

Processing log (verbatim):

```
run_id: behavioral_arbitration__20261028T133953
run_purpose: lab_validation
created_at: 2026-10-28T15:40:09+02:00
completed_at: 2026-10-28T15:41:15+02:00
status: completed_with_warnings
session_root: C:\behavior_suite\test_cases\E_full_session_integration
run_dir: C:\behavior_suite\test_cases\E_full_session_integration\behavioral_arbitration\behavioral_arbitration__20261028T133953
science_written: True
source_s7_run_id: behavioral_inference__20261026T163224
source_s6_run_id: combined_features__20261025T173244
exp_name: E_full_session_integration
n_frames: 36528
n_directional_rows: 73056
n_diagnostics: 38
qc_structural_ok: True
qc_warning_count: 1
qc_failure_count: 0
```

## D. Scientific transition audit

Representative frames comparing **S7 evidence columns** (`s7_pred_*`, `s7_prob_*`, `input_valid_*`, `score_origin_*`) to **S8 final** `behavior_label` / parallel channels / `label_reconstruction`. This checks algorithmic consistency with documented arbitration rules — **not** video ground-truth accuracy.

### D.1 Exclusive label transition

Boundary between two exclusive ethogram labels. S8 should show a hard label switch; S7 multi-detector scores may remain simultaneously elevated. Hierarchy / decay / sync modes in `label_reconstruction` explain which rule produced the winner.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2256 | 0 | 1 | approach | selected | 0 | 0 | 0 | 6.364e-06 | 0 | 0.3102 | 1 | 0.9464 | 0 | 0.002783 | 0 | 3.544e-05 | 1 | 0.9999 | True | model_observed |
| 2257 | 0 | 1 | approach | selected | 0 | 0 | 0 | 7.141e-06 | 0 | 0.3372 | 1 | 0.8851 | 0 | 0.003092 | 0 | 3.81e-05 | 1 | 0.9999 | True | model_observed |
| 2258 | 0 | 1 | approach | selected | 0 | 0 | 0 | 6.906e-06 | 0 | 0.3717 | 1 | 0.7196 | 0 | 0.003579 | 0 | 3.526e-05 | 1 | 0.9999 | True | model_observed |
| 2259 | 0 | 1 | nose_to_body | selected | 0 | 0 | 0 | 6.57e-06 | 1 | 0.8412 | 1 | 0.5698 | 0 | 0.003635 | 0 | 3.867e-05 | 1 | 0.9999 | True | model_observed |
| 2260 | 0 | 1 | nose_to_body | selected | 0 | 0 | 0 | 6.703e-06 | 1 | 0.8533 | 1 | 0.5851 | 0 | 0.004641 | 0 | 4.171e-05 | 1 | 0.9999 | True | model_observed |
| 2261 | 0 | 1 | nose_to_body | selected | 0 | 0 | 0 | 1.037e-05 | 1 | 0.8566 | 1 | 0.5035 | 0 | 0.004721 | 0 | 3.436e-05 | 1 | 1 | True | model_observed |
| 2262 | 0 | 1 | nose_to_body | gap_filled | 0 | 0 | 0 | 1.305e-05 | 1 | 0.8566 | 1 | 0.4258 | 0 | 0.004054 | 0 | 4.367e-05 | 1 | 1 | True | model_observed |

### D.2 Contact onset

Parallel `pred_contact` 0→1 while exclusive `behavior_label` continues on the ethogram axis (not renamed to contact). Compare `s7_pred_contact`/`s7_prob_contact` to cleaned `pred_contact` (optical/Schmitt/min-dur may differ from raw S7).

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37 | 0 | 1 | nose_to_tail | selected | 0 | 0 | 0 | 1.953e-06 | 0 | 0.2434 | 0 | 0.05623 | 0 | 0.001482 | 0 | 0.0001042 | 1 | 1 | True | model_observed |
| 38 | 0 | 1 | nose_to_tail | selected | 0 | 0 | 0 | 1.953e-06 | 0 | 0.2467 | 0 | 0.04953 | 0 | 0.00155 | 0 | 9.097e-05 | 1 | 1 | True | model_observed |
| 39 | 0 | 1 | nose_to_tail | selected | 0 | 0 | 0 | 1.868e-06 | 0 | 0.2467 | 0 | 0.04029 | 0 | 0.001654 | 0 | 9.333e-05 | 1 | 1 | True | model_observed |
| 40 | 0 | 1 | nose_to_tail | selected | 1 | 0 | 0 | 1.868e-06 | 0 | 0.2434 | 0 | 0.04535 | 0 | 0.001506 | 0 | 8.802e-05 | 1 | 1 | True | model_observed |
| 41 | 0 | 1 | nose_to_tail | selected | 1 | 0 | 0 | 1.868e-06 | 0 | 0.2467 | 0 | 0.04529 | 0 | 0.001603 | 0 | 9.98e-05 | 1 | 1 | True | model_observed |
| 42 | 0 | 1 | nose_to_tail | selected | 1 | 0 | 0 | 1.731e-06 | 0 | 0.2598 | 0 | 0.01556 | 0 | 0.001607 | 0 | 0.0001022 | 1 | 1 | True | model_observed |
| 43 | 0 | 1 | nose_to_tail | selected | 1 | 0 | 0 | 1.792e-06 | 0 | 0.2638 | 0 | 0.02368 | 0 | 0.001704 | 0 | 0.0001053 | 1 | 1 | True | model_observed |

### D.3 Contact offset

Parallel contact 1→0. Exclusive labels should not be forced to change solely due to contact channel offset.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 615 | 0 | 1 | nose_to_tail | selected | 1 | 0 | 0 | 1.802e-06 | 0 | 0.3515 | 0 | 0.0009646 | 0 | 0.001856 | 0 | 0.0001676 | 1 | 1 | True | model_observed |
| 616 | 0 | 1 | nose_to_tail | selected | 1 | 0 | 0 | 1.944e-06 | 0 | 0.3515 | 0 | 0.001482 | 0 | 0.002028 | 0 | 0.0001429 | 1 | 1 | True | model_observed |
| 617 | 0 | 1 | engaged | selected | 1 | 0 | 0 | 2.03e-06 | 0 | 0.3515 | 0 | 0.001965 | 0 | 0.003333 | 0 | 0.0001311 | 1 | 1 | True | model_observed |
| 618 | 0 | 1 | engaged | selected | 0 | 0 | 0 | 2.016e-06 | 0 | 0.3349 | 0 | 0.001965 | 0 | 0.003097 | 0 | 0.0001537 | 1 | 1 | True | model_observed |
| 619 | 0 | 1 | engaged | selected | 0 | 0 | 0 | 2.379e-06 | 0 | 0.3065 | 0 | 0.002674 | 0 | 0.00283 | 0 | 0.0001435 | 1 | 1 | True | model_observed |
| 620 | 0 | 1 | engaged | selected | 0 | 0 | 0 | 2.379e-06 | 0 | 0.2827 | 0 | 0.00325 | 0 | 0.002444 | 0 | 0.0001163 | 1 | 1 | True | model_observed |
| 621 | 0 | 1 | engaged | selected | 0 | 0 | 0 | 2.08e-06 | 0 | 0.252 | 0 | 0.006819 | 0 | 0.003313 | 0 | 0.0001347 | 1 | 1 | True | model_observed |

### D.4 SBS onset

Parallel `pred_side_by_side` onset; exclusive label remains independent.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8703 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 0.002868 | 1 | 0.9832 | 0 | 0.01847 | 0 | 0.003961 | 0 | 8.438e-05 | 1 | 1 | True | model_observed |
| 8704 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 0.005165 | 1 | 0.9832 | 0 | 0.01467 | 0 | 0.001702 | 0 | 7.126e-05 | 1 | 1 | True | model_observed |
| 8705 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 0.006296 | 1 | 0.9832 | 0 | 0.01515 | 0 | 0.001443 | 0 | 7.842e-05 | 1 | 1 | True | model_observed |
| 8706 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 0.006571 | 1 | 0.9819 | 0 | 0.01489 | 0 | 0.001324 | 0 | 7.862e-05 | 1 | 1 | True | model_observed |
| 8707 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 0.009152 | 1 | 0.9819 | 0 | 0.01719 | 0 | 0.001513 | 0 | 7.764e-05 | 1 | 1 | True | model_observed |
| 8708 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 0.006093 | 1 | 0.9819 | 0 | 0.0162 | 0 | 0.001584 | 0 | 7.287e-05 | 1 | 1 | True | model_observed |
| 8709 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 0.006439 | 1 | 0.9819 | 0 | 0.01631 | 0 | 0.001693 | 0 | 7.862e-05 | 1 | 1 | True | model_observed |

### D.5 SBS offset

Parallel SBS offset.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8882 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 5.264e-05 | 1 | 0.9012 | 0 | 0.0009486 | 0 | 0.002218 | 0 | 6.114e-05 | 1 | 1 | True | model_observed |
| 8883 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 9.821e-05 | 1 | 0.9012 | 0 | 0.00106 | 0 | 0.002124 | 0 | 0.0002064 | 1 | 1 | True | model_observed |
| 8884 | 0 | 1 | nose_to_body | selected | 1 | 1 | 0 | 8.676e-05 | 1 | 0.9012 | 0 | 0.001056 | 0 | 0.002042 | 0 | 0.0002579 | 1 | 1 | True | model_observed |
| 8885 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 8.972e-05 | 1 | 0.8152 | 0 | 0.001065 | 0 | 0.002249 | 0 | 0.0002199 | 1 | 1 | True | model_observed |
| 8886 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 4.336e-05 | 1 | 0.8152 | 0 | 0.00107 | 0 | 0.001948 | 0 | 6.81e-05 | 1 | 1 | True | model_observed |
| 8887 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 8.682e-05 | 1 | 0.8152 | 0 | 0.001105 | 0 | 0.001919 | 0 | 0.0003171 | 1 | 1 | True | model_observed |
| 8888 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 8.682e-05 | 1 | 0.8152 | 0 | 0.0009574 | 0 | 0.002857 | 0 | 0.0003142 | 1 | 1 | True | model_observed |

### D.6 Engaged / undefined boundaries

- **n_undefined = 0** in this publish; no engaged↔undefined boundary exists. n_engaged = 16574. Engaged is used as residual/valid ethogram state, distinct from undefined.
### D.6b Engaged boundary (substitute)

Substitute case: exclusive transition involving `engaged` (not undefined).

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 627 | 0 | 1 | engaged | synced | 0 | 0 | 0 | 3.871e-06 | 0 | 0.2515 | 0 | 0.05975 | 0 | 0.00402 | 0 | 0.0001218 | 1 | 0.9999 | True | model_observed |
| 628 | 0 | 1 | engaged | synced | 0 | 0 | 0 | 4.034e-06 | 0 | 0.1581 | 0 | 0.08184 | 0 | 0.003617 | 0 | 0.000153 | 1 | 0.9999 | True | model_observed |
| 629 | 0 | 1 | engaged | synced | 0 | 0 | 0 | 4.034e-06 | 0 | 0.1414 | 0 | 0.08249 | 0 | 0.004679 | 0 | 0.000137 | 1 | 0.9999 | True | model_observed |
| 630 | 0 | 1 | approach | synced | 0 | 0 | 0 | 4.138e-06 | 0 | 0.1193 | 0 | 0.02858 | 0 | 0.003923 | 0 | 0.0001483 | 0 | 0.02632 | True | model_observed |
| 631 | 0 | 1 | approach | synced | 0 | 0 | 0 | 4.565e-06 | 0 | 0.06164 | 0 | 0.06252 | 0 | 0.004077 | 0 | 0.000139 | 0 | 0.01626 | True | model_observed |
| 632 | 0 | 1 | approach | synced | 0 | 0 | 0 | 4.565e-06 | 0 | 0.005278 | 0 | 0.07967 | 0 | 0.004253 | 0 | 0.0001498 | 0 | 0.01846 | True | model_observed |
| 633 | 0 | 1 | approach | synced | 0 | 0 | 0 | 4.565e-06 | 0 | 0.00496 | 0 | 0.2088 | 0 | 0.003364 | 0 | 0.000129 | 0 | 0.02219 | True | model_observed |

### D.7 Gap fill (`label_reconstruction=gap_filled`)

Stage 10 short exclusive gaps filled; S7 evidence on filled frames may be negative/weak while S8 label continues prior bout — consistent with gap_fill_seconds rule. Global frames with this mode: **1138**.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 0 | retract | synced | 0 | 0 | 0 | 9.559e-06 | 0 | 0.001678 | 0 | 0.001035 | 0 | 0.02435 | 0 | 3.24e-05 | 1 | 0.9999 | True | model_observed |
| 3 | 1 | 0 | retract | synced | 0 | 0 | 0 | 9.464e-06 | 0 | 0.00173 | 0 | 0.0007461 | 0 | 0.02876 | 0 | 2.583e-05 | 1 | 0.9999 | True | model_observed |
| 4 | 1 | 0 | retract | synced | 0 | 0 | 0 | 9.531e-06 | 0 | 0.001693 | 0 | 0.0008196 | 0 | 0.0315 | 0 | 2.915e-05 | 1 | 0.9999 | True | model_observed |
| 5 | 1 | 0 | retract | gap_filled | 0 | 0 | 0 | 7.244e-06 | 0 | 0.001782 | 0 | 0.0006943 | 1 | 0.9705 | 0 | 3.073e-05 | 1 | 0.9999 | True | model_observed |
| 6 | 1 | 0 | retract | synced | 0 | 0 | 0 | 4.72e-06 | 0 | 0.001811 | 0 | 0.0007636 | 0 | 0.09821 | 0 | 3.063e-05 | 1 | 0.9999 | True | model_observed |
| 7 | 1 | 0 | retract | gap_filled | 0 | 0 | 0 | 4.144e-06 | 0 | 0.001812 | 0 | 0.0007526 | 1 | 0.9742 | 0 | 3.162e-05 | 1 | 0.9999 | True | model_observed |
| 8 | 1 | 0 | retract | synced | 0 | 0 | 0 | 3.757e-06 | 0 | 0.001953 | 0 | 0.0007048 | 0 | 0.1554 | 0 | 2.826e-05 | 1 | 0.9999 | True | model_observed |
| 9 | 1 | 0 | retract | gap_filled | 0 | 0 | 0 | 2.543e-06 | 0 | 0.002052 | 0 | 0.0006627 | 1 | 0.9741 | 0 | 3.162e-05 | 1 | 0.9999 | True | model_observed |

### D.8 Bout stitching (`label_reconstruction=stitched`)

Contextual/dynamic/idle-retract stitch modes collapse into `stitched` reconstruction tag; compare flanking S7 approach/retract evidence. Global frames with this mode: **8428**.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 591 | 0 | 1 | nose_to_body | selected | 1 | 0 | 0 | 2.121e-06 | 1 | 0.6926 | 0 | 0.001073 | 0 | 0.07991 | 0 | 0.0002193 | 1 | 1 | True | model_observed |
| 592 | 0 | 1 | nose_to_body | gap_filled | 1 | 0 | 0 | 1.986e-06 | 1 | 0.6618 | 0 | 0.001231 | 0 | 0.1037 | 0 | 0.0002368 | 1 | 1 | True | model_observed |
| 593 | 0 | 1 | nose_to_body | gap_filled | 1 | 0 | 0 | 1.986e-06 | 1 | 0.6618 | 0 | 0.001461 | 0 | 0.004395 | 0 | 0.0002007 | 1 | 1 | True | model_observed |
| 594 | 0 | 1 | nose_to_tail | stitched | 1 | 0 | 0 | 1.986e-06 | 0 | 0.5687 | 0 | 0.001168 | 0 | 0.003417 | 0 | 0.0001958 | 1 | 1 | True | model_observed |
| 595 | 0 | 1 | nose_to_tail | stitched | 1 | 0 | 0 | 1.771e-06 | 0 | 0.5687 | 0 | 0.001207 | 0 | 0.003613 | 0 | 0.0001791 | 1 | 1 | True | model_observed |
| 596 | 0 | 1 | nose_to_tail | stitched | 1 | 0 | 0 | 1.771e-06 | 0 | 0.5687 | 0 | 0.00118 | 0 | 0.003535 | 0 | 0.0001837 | 1 | 1 | True | model_observed |
| 597 | 0 | 1 | nose_to_tail | stitched | 1 | 0 | 0 | 1.875e-06 | 0 | 0.5563 | 0 | 0.001424 | 0 | 0.0005485 | 0 | 0.0001424 | 1 | 1 | True | model_observed |
| 598 | 0 | 1 | nose_to_tail | stitched | 1 | 0 | 0 | 1.875e-06 | 0 | 0.5467 | 0 | 0.001637 | 0 | 0.0007862 | 0 | 0.0001411 | 1 | 1 | True | model_observed |

### D.9 Dynamic rescue (`label_reconstruction=rescued`)

RESCUE_DYNAMIC may promote approach/retract from kinematics when raw S7 was weak. Global frames with this mode: **881**.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5020 | 1 | 0 | engaged | selected | 1 | 0 | 0 | 0.0002023 | 0 | 0.004441 | 0 | 0.001513 | 0 | 0.003868 | 0 | 2.449e-05 | 1 | 1 | True | model_observed |
| 5021 | 1 | 0 | engaged | selected | 1 | 0 | 0 | 0.0001758 | 0 | 0.004395 | 0 | 0.001639 | 0 | 0.0005971 | 0 | 2.49e-05 | 1 | 1 | True | model_observed |
| 5022 | 1 | 0 | engaged | selected | 1 | 0 | 0 | 0.000187 | 0 | 0.004277 | 0 | 0.001316 | 0 | 0.0004239 | 0 | 2.348e-05 | 1 | 1 | True | model_observed |
| 5023 | 1 | 0 | retract | rescued | 1 | 0 | 0 | 0.0001963 | 0 | 0.004263 | 0 | 0.001298 | 0 | 0.000415 | 0 | 2.376e-05 | 1 | 1 | True | model_observed |
| 5024 | 1 | 0 | retract | rescued | 1 | 0 | 0 | 0.000183 | 0 | 0.004085 | 0 | 0.0009121 | 0 | 0.0005321 | 0 | 2.376e-05 | 1 | 1 | True | model_observed |
| 5025 | 1 | 0 | retract | rescued | 1 | 0 | 0 | 0.0002002 | 0 | 0.003872 | 0 | 0.0008148 | 0 | 0.0005154 | 0 | 2.376e-05 | 1 | 1 | True | model_observed |
| 5026 | 1 | 0 | retract | rescued | 1 | 0 | 0 | 0.000156 | 0 | 0.003007 | 0 | 0.001269 | 0 | 0.0006189 | 0 | 2.472e-05 | 1 | 1 | True | model_observed |
| 5027 | 1 | 0 | retract | rescued | 1 | 0 | 0 | 0.0001687 | 0 | 0.003007 | 0 | 0.001576 | 0 | 0.0005073 | 0 | 2.562e-05 | 1 | 1 | True | model_observed |

### D.10 Reciprocity (`label_reconstruction=reciprocity_fallback`)

Nose-to-nose / mutually_disengaged reciprocity repairs one direction from the partner direction; expect asymmetric S7 evidence with synchronized S8 labels. Global frames with this mode: **3146**.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1710 | 1 | 0 | retract | synced | 0 | 0 | 0 | 1.18e-05 | 0 | 0.0007408 | 0 | 0.0009505 | 0 | 0.05375 | 0 | 0.01248 | 0 | 6.965e-06 | True | model_observed |
| 1711 | 1 | 0 | retract | synced | 0 | 0 | 0 | 1.18e-05 | 0 | 0.0007408 | 0 | 0.000956 | 0 | 0.03085 | 0 | 0.01153 | 0 | 6.889e-06 | True | model_observed |
| 1712 | 1 | 0 | retract | synced | 0 | 0 | 0 | 1.18e-05 | 0 | 0.0007408 | 0 | 0.001059 | 0 | 0.06069 | 0 | 0.01369 | 0 | 7.021e-06 | True | model_observed |
| 1713 | 1 | 0 | engaged | reciprocity_fallback | 0 | 0 | 0 | 1.083e-05 | 0 | 0.0007408 | 0 | 0.000913 | 0 | 0.03299 | 0 | 0.01595 | 0 | 7.167e-06 | True | model_observed |
| 1714 | 1 | 0 | engaged | reciprocity_fallback | 0 | 0 | 0 | 1.186e-05 | 0 | 0.0007373 | 0 | 0.0009229 | 0 | 0.02595 | 0 | 0.01707 | 0 | 7.167e-06 | True | model_observed |
| 1715 | 1 | 0 | engaged | reciprocity_fallback | 0 | 0 | 0 | 1.186e-05 | 0 | 0.0007002 | 0 | 0.0008657 | 0 | 0.007926 | 0 | 0.02005 | 0 | 6.977e-06 | True | model_observed |
| 1716 | 1 | 0 | engaged | reciprocity_fallback | 0 | 0 | 0 | 1.124e-05 | 0 | 0.0006935 | 0 | 0.0008844 | 0 | 0.005069 | 0 | 0.01779 | 0 | 7.266e-06 | True | model_observed |
| 1717 | 1 | 0 | engaged | reciprocity_fallback | 0 | 0 | 0 | 1.124e-05 | 0 | 0.0006935 | 0 | 0.001183 | 0 | 0.0048 | 0 | 0.0131 | 0 | 7.395e-06 | True | model_observed |

### D.11 Engaged hysteresis (`label_reconstruction=hysteresis`)

HYSTERESIS_ENGAGED holds engaged through brief interruptions (final_engaged_hysteresis_frames=12). Global frames with this mode: **78**.

| dense_frame_idx | focal_id | partner_id | behavior_label | label_reconstruction | pred_contact | pred_side_by_side | s7_pred_nose_to_nose | s7_prob_nose_to_nose | s7_pred_nose_to_body | s7_prob_nose_to_body | s7_pred_approach | s7_prob_approach | s7_pred_retract | s7_prob_retract | s7_pred_mutually_disengaged | s7_prob_mutually_disengaged | s7_pred_contact | s7_prob_contact | input_valid_approach | score_origin_approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5160 | 0 | 1 | approach | selected | 0 | 0 | 0 | 3.202e-06 | 0 | 0.08159 | 1 | 0.7679 | 0 | 0.001366 | 0 | 1.035e-05 | 1 | 0.9998 | True | model_observed |
| 5161 | 0 | 1 | approach | selected | 0 | 0 | 0 | 2.347e-06 | 0 | 0.2541 | 1 | 0.7905 | 0 | 0.001725 | 0 | 9.203e-06 | 1 | 0.9999 | True | model_observed |
| 5162 | 0 | 1 | approach | selected | 0 | 0 | 0 | 3.161e-06 | 0 | 0.4399 | 1 | 0.6171 | 0 | 0.00141 | 0 | 9.859e-06 | 1 | 0.9999 | True | model_observed |
| 5163 | 0 | 1 | approach | hysteresis | 0 | 0 | 0 | 2.431e-06 | 0 | 0.4653 | 1 | 0.6356 | 0 | 0.001471 | 0 | 1.013e-05 | 1 | 0.9998 | True | model_observed |
| 5164 | 0 | 1 | approach | hysteresis | 0 | 0 | 0 | 2.375e-06 | 0 | 0.5814 | 0 | 0.2316 | 0 | 0.001338 | 0 | 1.078e-05 | 1 | 0.9998 | True | model_observed |
| 5165 | 0 | 1 | approach | hysteresis | 0 | 0 | 0 | 2.375e-06 | 0 | 0.5871 | 0 | 0.2122 | 0 | 0.00126 | 0 | 8.597e-06 | 1 | 0.9998 | True | model_observed |
| 5166 | 0 | 1 | approach | hysteresis | 0 | 0 | 0 | 2.216e-06 | 1 | 0.818 | 0 | 0.1996 | 0 | 0.001165 | 0 | 9.209e-06 | 1 | 0.9998 | True | model_observed |
| 5167 | 0 | 1 | approach | stitched | 0 | 0 | 0 | 2.483e-06 | 1 | 0.818 | 0 | 0.2957 | 0 | 0.001354 | 0 | 8.66e-06 | 1 | 0.9998 | True | model_observed |

### D.12 Algorithmic consistency summary

- Exclusive vocabulary restricted to ['nose_to_nose', 'nose_to_body', 'nose_to_tail', 'approach', 'retract', 'mutually_disengaged', 'engaged']; parallel channels never appear as exclusive labels.
- Contact/SBS edges coexist with diverse exclusive labels (independence).
- Reconstruction modes present: selected, stitched, synced, reciprocity_fallback, min_duration, gap_filled, rescued, hysteresis — matches QC temporal rule activity (gap fill / stitch / rescue / reciprocity / hysteresis).
- n_undefined=0; LABEL_UNDEFINED rule code observed stages omit LABEL_UNDEFINED changes (LABEL_FINALIZE change count 0).

- Sampled reciprocity_fallback rows with partner present: **50** — **PASS**
- Overall D audit: algorithmic consistency with documented hierarchy/temporal/parallel-channel rules — **PASS** — no rule contradictions found in sampled windows

## Check tally

| Section | Check | Status |
| --- | --- | --- |
| A | n_unique_frames==36528 | PASS |
| A | n_directional_rows==73056 | PASS |
| A | qc.n_state_rows matches states | PASS |
| A | unique directional keys | PASS |
| A | label vocabulary | PASS |
| A | exactly one exclusive pred | PASS |
| A | label↔pred consistency | PASS |
| A | engaged≠undefined | PASS |
| A | no undefined/unresolved rows | PASS |
| A | time_sec alignment | PASS |
| A | no time misaligned rows | PASS |
| A | fps==120 | PASS |
| A | S7 sample cols present (contact) | PASS |
| A | S7 sample cols present (nose_to_nose) | PASS |
| A | S7 sample cols present (mutually_disengaged) | PASS |
| A | S7 sample cols present (side_by_side) | PASS |
| A | S7 sample cols present (nose_to_body) | PASS |
| A | S7 sample cols present (nose_to_tail) | PASS |
| A | S7 sample cols present (approach) | PASS |
| A | S7 sample cols present (retract) | PASS |
| A | exclusive≠parallel names | PASS |
| A | parallel channels independent | PASS |
| A | both directions present with equal rows | PASS |
| B | bout count matches QC | PASS |
| B | inclusive n_frames | PASS |
| B | duration_sec formula | PASS |
| B | exclusive bouts match states | PASS |
| B | parallel bouts match states | PASS |
| B | no exclusive overlap | PASS |
| B | no duplicate symmetric contact/SBS | PASS |
| B | splits only at transitions/gaps | PASS |
| B | recon row count | PASS |
| B | extract_behavior_bouts exact match | PASS |
| C | no QC failures | PASS |
| C | S8 currently eligible downstream | PASS |
| C | digest match S7 predictions | PASS |
| C | digest match S7 dyadic_wide | PASS |
| C | digest match S7 focal_wide | PASS |
| C | digest match S6 dyadic_features | PASS |
| C | S7/S6 path linkage | PASS |
| C | all expected stage codes observed | PASS |
| C | QC n_undefined matches | PASS |
| C | QC n_unresolved matches | PASS |
| C | QC label_counts sum to n_rows | PASS |
| D | undefined boundary absent (n_undefined=0) | PASS |
| D | reciprocity samples have partner rows | PASS |
| D | transition audit consistent (algorithmic) | PASS |

**PASS:** 47  |  **FAIL:** 0  |  **Total:** 47

## Residual risks

1. **No video ground-truth check** — this audit validates internal algorithmic/schema consistency only; ethogram scientific accuracy vs. annotated video is out of scope.
2. **`completed_with_warnings`** remains the historical status because Stage-8 contact morphological closing is disabled (`max_gap_to_fill=0`). Downstream eligibility is currently true, but consumers must understand contact cleaning omits morph-close (Stage-10 exclusive gap fill still applied).
3. **Parallel contact under mutually_disengaged** occurs (expected independence) but can confuse naive consumers that assume exclusive labels encode contact.
4. **Reconstruction modes are many** (stitched/synced/gap_filled/rescued/reciprocity/hysteresis/min_duration); bout-level `label_reconstruction_mode` is a mode statistic over the span, not a full provenance trace per frame.
5. **Short exclusive bouts (min 1 frame)** exist because Pipeline-110 emit filters are off (`apply_pipeline_110_emit_filters=false`); analysis consumers needing min-duration filters must apply them themselves.
6. **S7 routing_sentinel cells** are numerous (`n_routing_sentinel_cells=73056` in QC); validity masks treat these as non-observed evidence — residual risk if a downstream tool misreads sentinel scores as calibrated probabilities.

---

_End of scratch audit. PASS=47, FAIL=0. Sole QC warning classified as expected_config (morph-close disabled)._
