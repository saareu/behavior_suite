# Subsystem 10 — Behavioral Review & Editing

## Architecture Audit (S10-A)

**Status:** Discovery milestone (documentation only). **S10-A accepted as architecture input for S10-B.**  
**Date:** 2026-10-29  
**Scope:** Inventory reusable GUI / video / review infrastructure; map S9 → S10 consumption contracts (with S8/S7 evidence); propose synchronized inspection, manual-review, review-state, and immutable reviewed-output architectures; identify performance risks, scientific safeguards, unresolved decisions, and a bounded sprint roadmap. **No S10 production code in this sprint. No S9 changes, commits, or pushes.**  
**Authority:** Tracked S9 package `src/results_packaging/` + `docs/subsystem_09/` (accepted S9-A/S9-B); S8 `src/behavioral_arbitration/`; S7 `src/behavioral_inference/`; S3 `src/tracking_correction/` + tracking review UI; S1 `src/preprocess/` seek/sync; Group 1–4 GUI under `src/ui/`; local git-ignored `legacy/` Pipeline 110 + `bout_review_app` (workstation-only); E-session scale facts already recorded in S9 docs.

**Post-audit status (2026-10-29):** S10 was subsequently implemented (Sprints 1–5) and operator laboratory acceptance passed. S9 core packaging and CSV/media export are also operator lab-validated. Section 0.1 below remains the discovery-time snapshot.

**Follow-on:** Binding schemas, U1–U15 resolutions, and sprint gates live in **[S10-B Implementation Contract](implementation_contract.md)**.

**Legend used below:**  
- **Fact** — observed in repository / local legacy / accepted S9–S8 contracts.  
- **Recommendation** — design proposal pending review; not binding.

---

## 0. Preconditions and inspection method

### 0.1 Upstream closure (repository-confirmed)

| Check | Result |
| --- | --- |
| S7 | MVP complete / lab-validated; canonical + wide predictions with `input_valid` / `score_origin` / probs |
| S8 | MVP complete / lab-validated; immutable `behavior_states.parquet` + `behavior_bouts.parquet`; exclusive ethogram + parallel contact/SBS; inclusive dense bout endpoints |
| S9 | Core packaging MVP lab-accepted; `sync_index` / full bouts / ethogram summary / manifest / discovery / eligibility; optional `review_emit_v1`; S10 consumption surface frozen in S9-B §7 |
| S9 media/CSV | Optional export backend + GUI exist; final operator acceptance of the extension may still be open — **does not block S10 architecture** |
| S10 code / docs | Absent before this audit — no `src/behavioral_review/` (or equivalent); no ethogram/video review page |
| Group 4 UI | Hosts S9 packaging only; `main_window.py` explicitly reserves S10 (“Does not host ethogram/video review or bout editing”) |

### 0.2 Inspection method (read-only)

Traced:

1. S9 open path, sync equations, bout seek annotations, eligibility APIs (`publication.py`, `sync_index.py`, `contracts.py`, S9-B §7).
2. S8 states/bouts/finalize semantics (`contracts.py`, `bouts.py`, `finalize.py`, S8-B).
3. S7 prediction columns and validity (`results.py`, `export.py`).
4. Existing viewers: S3 tracking review, S1 trim navigator, S5 blob browser, shared `VideoFrameView` / `read_raw_frame_at_index`.
5. S3 acceptance / undo / working-vs-baseline persistence.
6. Legacy Pipeline 110 + Streamlit bout review app (git-ignored `legacy/` on this workstation only).

No S10 implementation, no Pipeline 110 execution, no mutation of packages.

### 0.3 Architectural objective (product boundary)

| Responsibility | Owner |
| --- | --- |
| Automated ethogram + bouts | S8 (immutable) |
| Packaging, sync index, summaries, optional export profiles / media | S9 (immutable packages) |
| Interactive synchronized inspection | **S10** |
| Optional manual correction + versioned reviewed publication | **S10** |
| Re-arbitration / re-packaging of S8/S9 science | **Forbidden in S10** |

**Fact / product rule (from S9-A §0.3):** Manual review remains optional. Automated S9 packages remain independently usable for analysis with automated provenance. Opening a package is **not** acceptance of content.

---

## 1. Existing reusable infrastructure

### 1.1 What can be reused (fact)

| Asset | Path | Reuse role | Frame / science confidence |
| --- | --- | --- | --- |
| Still-frame viewers | `src/ui/widgets/video_frame_view.py`, `pose_video_view.py` | Display decoded BGR/RGB frames | Temporal N/A |
| Verified frame seek | `src/preprocess/video_probe.py` → `read_raw_frame_at_index` | Exact 0-based decode-order seek with position verify or bounded sequential fallback | **High** for identity seeks |
| Prepared wrappers | `tracking_review_controller.read_prepared_frame`, `blob_features.preview.read_prepared_video_frame` | Thin prepared-video adapters | **High** when index = prepared order |
| Async GUI tasks | `src/ui/tasks.py` (`GuiTaskRunner`, `GuiTaskKind`) | Background discovery, eligibility, heavy I/O | Pattern to extend for S10 |
| S1 trim navigation | `trim_precrop_page.py` + preprocess controller | Scrub/step/jump **off** GUI thread | Best async seek precedent |
| S3 tracking review shell | `tracking_review_page.py` + `tracking_review_controller.py` + `tracking_correction/review.py` | Play/pause, scrub, ±1, jump, speeds, episode list, accept UX | UX shell **Medium–High**; continuous play accuracy **Medium** (see §7) |
| S5 prepared scrub | `blob_features_page.py` | Prepared-frame scrub + sync timestamp (no play) | High prepared index; sync GUI decode |
| S9 package APIs | `find_latest_eligible_s9_run`, `validate_s9_run_for_downstream`, `reopen_results_package`, `summarize_results_package_project` | Open eligible packages without recomputing science | Binding for intake |
| S9 sync + bout seeks | `sync_index.parquet`, bout `prepared_frame_idx_start/end` | Dense↔prepared mapping authority | Binding |
| S3 persistence pattern | `working_*` / automatic baseline / `manual_corrections.json` / `not_accepted`→`accepted`→`superseded` / undo stack | Semantic model for review lifecycle | Emulate; do not copy pose-edit ops |
| S1 atomic accept | crop-plan temp→validate→rename | Prefer for new S10 writes over full JSON rewrite | Pattern |
| Legacy review UX ideas | `legacy/src/bout_review_app/` | Filterable queue, verdicts, notes, reviewer_id, progress | Ideas only — not production GUI |

**Fact:** No `QMediaPlayer` / Qt Multimedia continuous player exists in tracked Python sources. All interactive video is OpenCV decode → still pixmap.

### 1.2 What must not be assumed reusable without verification (fact)

| Assumption | Reality |
| --- | --- |
| “Reuse S3 player as ethogram player” | S3 advances by **prepared** fps × wall clock; ethogram truth is **120 Hz dense**. No dual-clock playhead. GUI-thread decode during play. Post-jump sequential `CAP_PROP_POS_FRAMES` is **not** verified like `read_raw_frame_at_index`. |
| “Reuse Pipeline 110 / Streamlit as S10” | Streamlit is not Group 4 Qt; legacy re-collapses presentation rows; emit floors as science; misnamed `start_prepared_idx` stores dense indices; no bout boundary editor. |
| Dense index == video frame | E-session: 36 528 dense samples vs 36 238 unique prepared frames. |
| Opening S9 package embeds S7/S8 states | Default `reference_lightweight` packages **reference** states/probs/video by path+digest. |
| `review_emit_v1` is scientific bout truth | Selection-only navigation index; full `behavior_bouts.parquet` remains authoritative automated science. |

### 1.3 Missing capabilities (fact — gaps)

1. Ethogram / bout timeline widget synced to a playhead.
2. Dual-clock scrubber (dense `sample_idx`/`time_sec` ↔ prepared `prepared_frame_idx`/`prepared_time_sec`).
3. Package-open interactive review session (Group 4 page reserved, unimplemented).
4. Windowed Parquet projection helpers dedicated to scrubbing S8 states / S7 probs.
5. Correction event store + reviewed framewise/bout publication schemas.
6. Undo/**redo** for behavioral corrections (S3 has undo only).
7. Bout split / merge / boundary trim editor (absent in modern UI and legacy Streamlit app).
8. Async continuous decode / frame cache suitable for long-session ethogram scrub.
9. Explicit inspection vs editing mode vocabulary for behavioral review.

### 1.4 Legacy lessons (fact + recommendation)

| Legacy behavior | Lesson for S10 |
| --- | --- |
| Verdict / optional `corrected_label` / error tags on bout IDs | Useful **presentation** of acceptance; insufficient as sole science model if boundaries must change |
| Annotations outside packaging tree | **Preserve** — additive store, never mutate S9 |
| Timing issues as tags (`timing_too_short`) without editing endpoints | Too weak if lab requires boundary correction; keep tags as optional metadata |
| Clip/GIF as review medium | Optional convenience; **authority is synchronized prepared video + dense ethogram**, not pre-cut clips |
| Pipeline 110 re-collapse + emit floors | **Forbidden** as scientific rewrite; optional navigation filters already owned by S9 profiles |

---

## 2. S9 / S8 / S7 input contracts

### 2.1 S10 open path (fact — S9-B §7)

1. Load `package_manifest.json`.
2. Verify package digests for local package artifacts.
3. Load `sync_index.parquet`, `behavior_bouts.parquet`, `ethogram_summary.parquet`, `qc.json`.
4. Resolve prepared video via manifest path/digest — **fail clearly** if missing/stale (no silent alternate).
5. Resolve S8 `behavior_states.parquet` by reference (or embedded profile) for framewise display.
6. Resolve S7 / S8 diagnostic references **lazily** as inspection windows require.

Entry APIs (fact): `validate_s9_run_for_downstream`, `reopen_results_package`, `find_latest_eligible_s9_run`. Historical `status` on disk is never current-eligibility truth.

### 2.2 Package layout consumed (fact)

```text
{session_root}/results_package/results_package__YYYYMMDDTHHMMSS/
  sync_index.parquet
  behavior_bouts.parquet
  ethogram_summary.parquet
  package_manifest.json
  settings_used.yaml
  qc.json
  run_meta.json
  processing_log.txt
  export_profiles/<profile_id>/   # optional; e.g. review_emit_v1
```

Default mode `reference_lightweight` does **not** embed full S8 states, S7 tables, S6 features, or video bytes.

### 2.3 `sync_index.parquet` (fact — binding)

| Column | Role |
| --- | --- |
| `sample_idx` | Primary key; contiguous `0…N-1` |
| `dense_frame_idx` | Must equal `sample_idx` |
| `time_sec` | Uniform 120 Hz analysis clock |
| `prepared_frame_idx` | NN map into prepared video |
| `prepared_time_sec` | Prepared clock at that frame |
| `is_exact_source_sample` | Exact time match ≤ 1e-9 s |
| `timing_residual_sec` | `time_sec - prepared_time_sec` |

Mapping: S1 `prepared_sync.npz` + S5 nearest-neighbor, **earlier-frame tie-break**. Do not rebuild sync solely to open a package. Do not clamp outside prepared time range.

### 2.4 Automated bouts (fact)

Lossless stamped copy of **all** S8 bouts:

- Inclusive `start_sample_idx` / `end_sample_idx`; `bout_duration_frames` = inclusive count; duration_sec = frames / 120.
- Channels: `exclusive` \| `contact` \| `side_by_side`.
- Identity: exclusive → directional (`focal_id`/`partner_id`); parallel → `symmetric_pair` (focal/partner null).
- Seek aids (not science): `prepared_frame_idx_start/end`, `prepared_time_sec_start/end`.
- Provenance: `bout_id`, `source_s8_bout_id`, `source_s8_run_id`, `package_run_id`.

### 2.5 S8 ethogram authority (fact)

Framewise grain: one row per `(dense_frame_idx, focal_id, partner_id)`.

**Exclusive labels:** `nose_to_nose`, `nose_to_body`, `nose_to_tail`, `approach`, `retract`, `mutually_disengaged`, `engaged`, plus published `undefined` when evidence cannot support legacy residual.

**Parallel (non-exclusive):** `pred_contact` / `prob_contact`, `pred_side_by_side` / `prob_side_by_side` — may be positive alongside any exclusive label.

Provenance fields of review interest: `label_reconstruction`, `undefined_reason`, `hierarchy_residual_kind`, validity / `score_origin_*` echoes, bout `label_reconstruction_mode`.

| Concept | Meaning S10 must preserve |
| --- | --- |
| `engaged` | Valid residual (not “unknown”) |
| `undefined` | Unavailable evidence / unresolved invalid — not engaged |
| Invalid S7 zeros | Routing sentinels, **not** confident negatives |

### 2.6 S7 evidence (fact)

Canonical long-form `predictions.parquet` (+ wide companions). Implemented column names: `prob_raw`, `prob_effective`, `prediction`, `input_valid`, `invalid_reason`, `score_origin` ∈ {`model_observed`, `routing_sentinel`, `fusion_routed`}.

S9 does not copy full S7 tables by default. S10 loads **windowed** projections via manifest pointers for inspection only. S10 must **not** re-score models or re-arbitrate.

### 2.7 Optional navigation profile (fact)

`export_profiles/review_emit_v1/` — filtered `bout_index.parquet` + `exclusion_table.parquet`. Defaults: exclude contact, include engaged, floors 10 frames / 0.10 s at **120 Hz**. Presentation aid only; never replaces full packaged bouts.

---

## 3. Synchronized inspection contract

### 3.1 Display surfaces (recommendation)

One interactive workspace with linked panels (not separate apps):

| Surface | Content |
| --- | --- |
| Video pane | S1 `prepared_video.mp4` still/play frames (default). Optional overlays are references only — never silent source switch (S9 D8). |
| Exclusive ethogram | **Two directional tracks** (A→B and B→A) for the active pair, color-coded exclusive labels including `engaged` / `undefined`. |
| Parallel tracks | Independent **contact** and **side-by-side** lanes (symmetric_pair), not folded into exclusive. |
| Bout overlay | Selected bout span(s) with inclusive dense endpoints; channel-aware highlighting. |
| Evidence pane | Windowed S7 `prob_effective` (+ threshold markers) with `input_valid` / `score_origin` badges; invalid shown as **unobserved/invalid**, never as confident 0. |
| Provenance pane | S8 `label_reconstruction` / `undefined_reason` / bout reconstruction mode for the playhead sample or selected bout. |
| Summary strip | Package `ethogram_summary` + review progress (see §5) — not a second science authority. |

Inspection must work with **zero** corrections applied.

### 3.2 Clocks and seeking (recommendation; grounded in S9 facts)

| Concern | Rule |
| --- | --- |
| Analysis / ethogram playhead | Dense `sample_idx` + `time_sec` (120 Hz) is the **behavioral authority**. |
| Video decode index | `sync_index.prepared_frame_idx[sample_idx]` → `read_raw_frame_at_index(prepared_video, prepared_frame_idx)`. |
| Bout jump start/end | Use packaged `prepared_frame_idx_start/end` (equivalent to sync lookup at inclusive endpoints). |
| Duration display | Always `bout_duration_frames / 120` (or sample count); never `end_time - start_time` alone; never unique-prepared-frame counts. |
| UI labels | Explicitly name clocks (“analysis time” vs “prepared video time”). |

**Identity seeks** always route through the verified helper (or an equivalent verified sequential capture). Do not treat unverified `CAP_PROP_POS_FRAMES` jumps as scientifically authoritative.

### 3.3 Many dense samples → one prepared frame (recommendation)

**Fact:** NN mapping yields plateaus — multiple consecutive `sample_idx` share one `prepared_frame_idx`.

| Policy | Specification |
| --- | --- |
| Playhead authority | Advance ethogram by **dense sample** (or dense-time), not by unique prepared frames. |
| Video update | When `prepared_frame_idx` is unchanged across playhead moves, **keep the same decoded frame** and still advance ethogram/evidence cursors. |
| Scrubber UI | Primary scrubber = dense timeline; secondary read-only prepared-frame indicator. |
| Click-on-video | Map click time via prepared clock → nearest dense sample(s); if a plateau, select the dense sample nearest in `time_sec` (document tie-break: earlier sample). |
| Bout boundaries on plateaus | Inclusive dense endpoints remain correct even when start and end share a prepared frame (short bouts / low motion). Viewer must not “expand” or “collapse” science to unique frames. |
| Playback scheduling | Prefer scheduling ticks on dense time; decode only when prepared index changes (cache last frame). |

### 3.4 Timeline navigation and selection (recommendation)

- Click lane → set playhead sample; select bout if click falls inside a bout interval for that channel/identity.
- Prev/next bout navigates within the **active filter set** (full science table by default; optional `review_emit_v1` or user filters for navigation only).
- Filters that hide short/contact bouts are **view filters**, logged in session UI state — they must not rewrite labels or drop rows from the authoritative automated table.
- Dual-direction exclusive selection: selecting a reciprocal bout may optionally reveal the opposite-direction sibling without merging scientific rows.

### 3.5 What inspection explicitly does not do (recommendation)

- No S8 hierarchy re-run, gap-fill, or contact remultiplex.
- No silent smoothing / majority filters on displayed labels.
- No substitution of clip MP4s for prepared-video authority when both exist.
- No auto-mark reviewed on open or on scrub.

---

## 4. Manual review model

### 4.1 Candidate operations (recommendation)

| Operation | Exclusive directional | Contact / SBS independent | Notes |
| --- | --- | --- | --- |
| Relabel interval | Yes — set exclusive label over `[s0,s1]` inclusive for one direction | Yes — set pred/presence over interval for that channel | Must not auto-write the other exclusive direction unless user opts into explicit reciprocal edit |
| Adjust bout start/end | Derived from framewise edit (preferred) | Same | See authoritative model below |
| Split / merge adjacent | Derived after framewise continuity change | Same | |
| Accept / reject bout | Review-status overlay; reject ≠ delete automated row | Same | Rejected remain visible with status |
| Reviewer notes / tags | Attach to bout_id and/or correction event | Same | |
| Undo / redo | Required for edit mode | Same | Stronger than S3 (redo needed) |

### 4.2 Two editing philosophies (audit)

| Model | Pros | Cons |
| --- | --- | --- |
| **A. Framewise-authoritative** — edits mutate reviewed framewise states; bouts regenerated by deterministic contiguous collapse (S8 bout rules: split on label/pred change or dense gap ≠ 1; inclusive endpoints; duration = count) | Matches S8 science; split/merge/trim are consequences; single consistency invariant; evidence panes stay aligned to samples | Requires loading/editing state windows; careful channel separation |
| **B. Bout-record-authoritative** — edits mutate bout table rows directly | Matches legacy verdict UX; fewer rows touched | Easy to create gaps/overlaps/illegal exclusive multi-labels; framewise evidence desyncs; regeneration ambiguity |

### 4.3 Authoritative recommendation: **Model A (framewise-authoritative)**

**Recommendation:** Corrections operate primarily on **reviewed framewise channel states**. Reviewed bouts are **deterministically regenerated** from those states using the published S8 bout-collapse contract (contiguous blocks; inclusive endpoints; channel-specific identity grain). Automated S8/S9 artifacts remain untouched.

Channel rules:

1. **Exclusive directional channel** — at most one exclusive label per `(sample_idx, focal_id, partner_id)`. Edits write that cell (or interval). Opposite direction is independent unless the operator applies an explicit reciprocal tool.
2. **Contact channel** — independent binary/presence (+ optional reviewed prob display remains original S7/S8 evidence). Symmetric pair grain; do not invent focal/partner.
3. **Side-by-side channel** — same independence as contact. Never promote SBS into the exclusive ethogram via viewer logic.
4. **Original evidence** — S7 probs and S8 automated states/bouts always remain readable side-by-side with reviewed overlays; corrections never overwrite source Parquets.

Bout-level UI (trim handles, split, merge, accept/reject) is a **convenience projection** that compiles into framewise correction events before bout regeneration.

### 4.4 Correction-event compilation (recommendation)

Each user gesture becomes one or more append-only correction events, for example:

- `set_exclusive_label(interval, focal, partner, label)`
- `set_parallel_pred(channel, interval, value)`
- `set_bout_review_status(bout_id, accepted|rejected|unreviewed)`
- `add_note(...)`

Working reviewed states = apply event log to a copy of automated S8 states (baseline). Regenerated reviewed bouts = collapse(working states). Undo/redo = move event-log cursor; reapply.

**Reject semantics (recommendation):** `rejected` is a review-status flag on a bout identity (stable `source_s8_bout_id` / derived reviewed bout id). It does **not** delete automated history. Downstream consumers that ask for “accepted reviewed only” exclude rejected; analysis using automated packages ignores S10 status entirely.

---

## 5. Review-state contract

### 5.1 Modes (recommendation)

| Mode | Capabilities |
| --- | --- |
| **Inspection** (default on open) | Play, scrub, timelines, evidence, bout jump, filters, notes **read-only**. No correction events. |
| **Editing** (explicit toggle) | Enables correction tools; requires reviewer identity; records edit session metadata. |

Opening a package, scrubbing, or applying navigation filters **never** implies acceptance of automated content.

### 5.2 Status vocabulary (recommendation)

Per package review run (and optionally per bout):

| Status | Meaning |
| --- | --- |
| `not_started` | Review run created or package opened; no accept publish |
| `in_progress` | Edits and/or bout statuses recorded; not published as accepted reviewed output |
| `published_accepted` | Immutable reviewed package published for downstream use |
| `superseded` | Newer review publish exists for same parent S9 package (or same review lineage) |

Per-bout review flags (overlay, not S9 columns): `unreviewed` (default), `accepted`, `rejected`, optional `uncertain` (legacy parity — **U-decision** whether to keep).

Progress metrics (UI/QC): counts of bouts by channel/status; fraction of exclusive time touched by corrections; last reviewer / timestamp — derived, not science.

### 5.3 Correction history and reviewer metadata (recommendation)

- Append-only event log with `event_id`, `timestamp`, `reviewer_id`, `op`, payload, `schema_version`.
- Session header: `review_run_id`, `parent_package_run_id`, `parent_s8_run_id`, tool versions, settings hash.
- Emulate S3 separation of **automatic baseline** vs **manual events**; prefer atomic writes (temp→rename) over unprotected rewrite.

### 5.4 Optionality (fact + recommendation)

Automated S9 packages remain first-class analysis inputs. S10 is optional. Downstream tools must select **automated package** vs **reviewed package** explicitly (see §6.4). Absence of S10 output must not block analysis pipelines that already consume S9.

---

## 6. Immutable output architecture

### 6.1 Non-goals (fact / boundary)

S10 **must not** modify:

- S8 run directories or Parquets  
- S9 package directories or digests  
- S7 prediction tables  

Corrections are additive products that **reference** `package_run_id` (+ `source_s8_run_id` / digests).

### 6.2 Proposed reviewed-run layout (recommendation)

```text
{session_root}/behavioral_review/behavioral_review__YYYYMMDDTHHMMSS/
  review_manifest.json            # parent package identity, digests, mode, schema stamps
  settings_used.yaml
  run_meta.json
  processing_log.txt
  qc.json
  correction_events.parquet       # append-only compiled event table (published snapshot)
  reviewed_behavior_states.parquet
  reviewed_behavior_bouts.parquet
  review_bout_status.parquet      # accepted/rejected/unreviewed + notes linkage
  reviewer_sessions.json          # optional session rollup
  # working/ (non-published, mutable while in_progress) — implementation detail
```

Naming is illustrative; S10-B should freeze stamps analogous to other subsystems (`behavioral_review_*_v1`).

### 6.3 Artifact roles (recommendation)

| Artifact | Role |
| --- | --- |
| `correction_events` | Audit-grade history; enough to rebuild reviewed states from automated baseline |
| `reviewed_behavior_states` | Framewise exclusive + parallel channels after applying events |
| `reviewed_behavior_bouts` | Deterministic collapse of reviewed states; includes linkage to `source_s8_bout_id` where intervals still correspond |
| `review_bout_status` | Acceptance overlay; rejected/excluded handling without deleting rows |
| `review_manifest` | Parent `package_run_id`, S8/S7/S1 digests at review time, schema versions, publication stamp |
| Automated originals | Still opened via parent S9 package — never copied as “new truth” without provenance |

### 6.4 How reviewed outputs differ from automated results (recommendation)

| Aspect | Automated (S9/S8) | Reviewed (S10) |
| --- | --- | --- |
| Authority | Machine arbitration | Human-corrected overlay + regenerated bouts |
| Provenance | `source_s7` / arbitration QC | Parent package + correction events + reviewer ids |
| Eligibility | `validate_s9_run_for_downstream` | Separate `validate_s10_run_for_downstream` (future) |
| Default analysis | Use S9 unless user selects reviewed | Explicit opt-in |

### 6.5 Downstream export selection (recommendation)

- S9 media/CSV exporters today target **automated** packages. Extending them to prefer reviewed bouts is a **later explicit decision** (do not silently swap).
- Recommendation for post-S10: exporters accept an explicit `source_kind ∈ {automated_s9, reviewed_s10}` with digest pins; never auto-pick “latest reviewed” without UI confirmation.

### 6.6 Atomic publication and versioning (recommendation)

Mirror S5–S9 lifecycle:

1. Mutable working review session under a run directory.  
2. Publish copies validated tables into immutable stamp; write manifest last (or temp dir → atomic rename).  
3. Never rewrite historical published review status; mark prior publish `superseded` via discovery classification when a newer accepted review shares lineage.  
4. Fail closed if parent S9 digests drifted (`stale_source`) unless operator explicitly acknowledges — default: block publish.

---

## 7. Performance

### 7.1 Scale signals (fact from S9 audit / E-session)

| Asset | Approximate scale |
| --- | --- |
| Dense samples | ~36.5k |
| Directional state rows | ~73k (pair × directions) |
| Prepared video | ~417 MB; ~36.2k frames |
| S8 states Parquet | ~21 MB (acceptable to map carefully; still project columns) |
| S6 features | ~225 MB — **never** default-open for review |
| Full S7 tables | Reference only; windowed reads |

### 7.2 Risks and mitigations (recommendation)

| Risk | Mitigation |
| --- | --- |
| GUI-thread OpenCV decode (S3 play pattern) | Decode + heavy Parquet on `GuiTaskRunner` / worker; cache last prepared frame; only decode on prepared-index change |
| Random access on long MP4 | Prefer verified seek; maintain sequential capture during forward play; avoid reopen-from-zero on every tick |
| Ethogram render of full session | Multi-resolution timeline (overview pyramid + detail window); do not draw 73k rectangles naively each paint |
| Parquet I/O | Column projection; predicate pushdown / row-group window reads around playhead; avoid reconstructing S7→S9 chain |
| Source discovery / eligibility hashing | Background discovery runner; memoized validation; do not rehash video every poll |
| Correction on long recordings | Event log + sparse interval application; regenerate bouts for touched channels/identities only when possible |
| Full dependency rebuild temptation | **Forbidden on GUI thread and on package open** — trust S9 sync_index + referenced paths |

### 7.3 Playback architecture sketch (recommendation)

1. **Dense clock** drives UI.  
2. Frame cache keyed by `prepared_frame_idx`.  
3. Async decode requests coalesced (keep latest requested index).  
4. Ethogram detail window loads state rows for `[playhead ± W]` plus bout index in memory.  
5. S7 evidence fetched only when evidence pane visible and window changes.

Do **not** ship continuous HW decode as MVP unless lab performance fails; design interfaces so a later decoder backend can replace OpenCV without changing sync contracts.

---

## 8. Scientific safeguards

| Safeguard | Requirement |
| --- | --- |
| Directional identities | Exclusive rows remain `(focal_id, partner_id)`; no silent symmetric collapse of science tables |
| Exclusive ethogram semantics | One exclusive label per directional sample; hierarchy not re-run in viewer |
| Independent contact/SBS | Parallel lanes; never invent exclusive SBS from `pred_side_by_side` |
| Original model evidence | S7 probs/validity always available as read-only evidence; invalid ≠ negative |
| `engaged` vs `undefined` | Distinct labels; UI legend must not synonymize |
| Invalid/unobserved provenance | Surface `score_origin` / `undefined_reason`; do not paint sentinels as confident zeros |
| Inclusive bout boundaries | Preserve inclusive dense endpoints; duration = frame count / 120 |
| Automated alongside corrected | Parent package immutable; reviewed outputs additive |
| No silent filtering as science | View filters / `review_emit_v1` cannot rewrite packaged automated bouts |
| No viewer-side smoothing | Display raw reviewed or automated labels; any temporal cleanup is an explicit correction event |

---

## 9. Unresolved architectural / scientific decisions

These require **explicit approval before implementation** (defaults suggested for discussion only):

| ID | Decision | Options / notes | Suggested default for S10-B discussion |
| --- | --- | --- | --- |
| U1 | Authoritative edit model | Framewise (A) vs bout-record (B) | **A — framewise** (§4.3) |
| U2 | Reciprocal exclusive edits | Independent only vs optional “apply to both directions” tool | Independent by default; explicit reciprocal tool allowed |
| U3 | `uncertain` verdict | Keep legacy 4-way vs accepted/rejected/unreviewed only | Defer; MVP can omit `uncertain` |
| U4 | Working-session storage | Sidecar under session vs only in-memory until publish | Persist working under review run dir (crash-safe) |
| U5 | Embed vs reference states in S9 for review | Rely on reference_lightweight vs require embed profile for S10-ready packages | Allow reference; fail clearly if S8 path stale; optional “S10-ready” packaging checklist |
| U6 | Reviewed export into S9 media/CSV | Out of S10 MVP vs simultaneous | **Out of S10 MVP**; explicit later extension |
| U7 | Overlay video as alternate view | Offer vs prepared-only MVP | Prepared-only MVP; overlays optional later |
| U8 | Presentation collapse of reciprocal bouts | UI convenience vs forbidden | Optional UI grouping; science tables stay directional |
| U9 | Role mapping (I/NI) | Show implanted roles in UI | Optional presentation; not scientific bout rewrite |
| U10 | Minimum evidence columns in MVP | Full S7 window vs S8 echoed probs only | Prefer S8 echoed `prob_*`/`input_valid_*` first; lazy S7 for deep dive |
| U11 | Bout regeneration parity details | Exact S8 `bouts.py` collapse vs documented subset | **Call same pure collapse function** or shared library extracted carefully without changing S8 outputs |
| U12 | Eligibility when parent S9 becomes stale mid-review | Block publish vs warn-and-allow | Block publish by default |
| U13 | Multi-pair / K>2 | Dyad-only MVP vs general | Dyad workflows first; schema should not preclude K>2 |
| U14 | Redo stack depth / branching | Linear undo-redo vs DAG | Linear undo-redo MVP |
| U15 | Whether reject auto-clears exclusive labels | Status-only vs also zero interval | **Status-only**; label clear is separate explicit edit |

---

## 10. Recommended implementation roadmap

Adjustments vs the brief: Sprint 1 absorbs dual-clock seek verification and package-open contracts before ethogram chrome; Sprint 3 owns the framewise editing model end-to-end; publication is Sprint 4; GUI polish + lab acceptance is Sprint 5. No S9 science changes in any sprint.

### S10-1 — Source contracts and synchronized video navigation

| | |
| --- | --- |
| **Scope** | S10-B contract freeze for intake; open eligible S9 packages; load sync_index + bout seeks; prepared-video identity seek; dense playhead with many-to-one prepared plateaus; inspection-only session skeleton (headless + minimal UI probe allowed) |
| **Out** | Ethogram painting, corrections, publish |
| **Exit** | Seek demos on E-session: bout start/end prepared frames match sync_index; plateau policy documented and tested; stale video/package fails closed |

### S10-2 — Ethogram, bout and evidence inspection

| | |
| --- | --- |
| **Scope** | Dual directional exclusive lanes; contact + SBS lanes; bout overlay/navigation; windowed S8 states; provenance fields; optional S7 window evidence; view filters without science mutation |
| **Out** | Correction writes, accept publish |
| **Exit** | Full inspection workspace on E-session package without edits; performance acceptable for scrub/play |

### S10-3 — Manual correction and review history

| | |
| --- | --- |
| **Scope** | Explicit edit mode; framewise correction events; undo/redo; bout UI compiling to events; accept/reject/notes; working reviewed states + deterministic bout regeneration; channel separation rules |
| **Out** | Immutable downstream publication (may snapshot working artifacts only) |
| **Exit** | Golden tests: interval relabel → expected bout split/merge; exclusive vs parallel isolation; automated package untouched |

### S10-4 — Reviewed-output publication and QC

| | |
| --- | --- |
| **Scope** | Atomic publish layout; manifests; digests; QC; discovery/eligibility; superseded handling; parent stale detection |
| **Out** | Deep GUI theming |
| **Exit** | Published review reopened by path; validates; does not modify S8/S9; automated vs reviewed selection documented |

### S10-5 — GUI integration and laboratory validation

| | |
| --- | --- |
| **Scope** | Group 4 S10 page beside S9; task runners; lab checklist on `E_full_session_integration`; inspection-only and edit+publish operator paths |
| **Out** | S9 media/CSV auto-switch to reviewed (unless U6 reopened) |
| **Exit** | Operator laboratory acceptance; documentation status update |

---

## 11. Architectural recommendations (summary)

1. **Reuse** verified prepared-frame seek, still viewers, S9 package APIs/sync_index, S3 acceptance semantics, and async task patterns — **not** S3 play-loop decode or Pipeline 110 Streamlit as production review.  
2. **Consume** eligible S9 packages per S9-B §7; treat S8 as ethogram authority and S7 as evidence-by-reference.  
3. **Synchronize** with dense playhead authority and prepared decode via `sync_index`; handle prepared plateaus without collapsing science.  
4. **Edit** via framewise-authoritative correction events with deterministic bout regeneration; keep contact/SBS independent.  
5. **Default** to inspection-only; explicit editing; opening ≠ acceptance; automated packages remain valid alone.  
6. **Publish** additive immutable reviewed runs referencing parent `package_run_id`; never mutate S8/S9.  
7. **Guard** performance with async decode, timeline pyramid, windowed Parquet, and no GUI-thread dependency rebuild.  
8. **Preserve** engaged≠undefined, inclusive endpoints, directional identities, invalid provenance, and original evidence beside corrections.  
9. **Resolve U1–U15** in S10-B before coding corrections/publication.  
10. **Implement** in five bounded sprints (§10); stop here for architectural review.

---

## 12. Document index (this audit)

| § | Content |
| --- | --- |
| 0 | Preconditions / objective |
| 1 | Reusable infrastructure and gaps |
| 2 | S9/S8/S7 input contracts |
| 3 | Synchronized inspection design |
| 4 | Manual editing model |
| 5 | Review status and audit history |
| 6 | Reviewed-output schemas and versioning |
| 7 | Performance risks |
| 8 | Scientific safeguards |
| 9 | Unresolved decisions |
| 10 | Implementation roadmap |
| 11 | Summary recommendations |

### Deliverable checklist (task §10)

1. Existing reusable infrastructure — §1  
2. S9/S8/S7 input contracts — §2  
3. Video/timeline synchronization design — §3  
4. Proposed editing model — §4  
5. Review status and audit history — §5  
6. Reviewed-output schemas — §6  
7. Performance risks — §7  
8. Scientific safeguards — §8  
9. Unresolved architectural decisions — §9  
10. Recommended implementation roadmap — §10  

---

## 13. Stop point

Documentation only. **S10-A audit complete.**

Do not implement S10 production code from this audit alone. Binding follow-on: **[S10-B Implementation Contract](implementation_contract.md)** (U1–U15 decisions, correction/review schemas, working persistence, publication, sprints). Stop for architectural review of S10-B before coding.

_End of S10-A architecture audit._
