# Post-v0.1.1 — S10 Review Efficiency, Visualization, and PI Reporting

**Status:** **Gallery A2 accepted** (working-draft gallery + single/bulk status; journal durability closure). A1 remains accepted. PI report / overlays remain deferred.  
**Date:** 2026-11-01 (**D0 closed**; **Gallery A1** accepted; **Gallery A2** accepted — see also [`d0_workspace_architecture.md`](d0_workspace_architecture.md))  
**Baseline:** Behavior Suite **v0.1.1** released; S1–S10 MVP lab-validated; D0 workspace MVP closed at `e81d530`; A1 base `0d4b237`.  
**Non-scope of remaining follow-ons:** PI HTML report, overlay generation, subsession science, S8 schema changes.

**Legend**

| Mark | Meaning |
| --- | --- |
| **Fact** | Observed in current repository / contracts / local git-ignored `legacy/` |
| **Recommendation** | Proposed design pending operator approval; not binding |
| **Reuse** | Existing component that should be extended, not replaced |
| **New** | Infrastructure that does not exist today |

---

## 0. Executive summary

v0.1.1 already provides authoritative packaging (S9), framewise-authoritative review (S10), optional eager MP4/CSV export jobs, dual-clock ethogram inspection, and digest-pinned publication. What it does **not** yet provide is an efficient **behavior-grouped bout gallery**, a **PI-facing offline HTML report**, a **shared overlay rendering layer**, or a coherent **application workspace context**.

Recommended order (dependency-aware):

1. **D0 — Shared workspace architecture** (revised; see [`d0_workspace_architecture.md`](d0_workspace_architecture.md)): catalog + `WorkspaceContext` + migration; optional subsessions in D0-C.  
2. **A — S10 grouped bout gallery + exclusive-label approval UX** (primary lab bottleneck; session write-back / D0-A at least in parallel).  
3. **B — PI-facing immutable HTML report bundle** (consumes gallery preview contracts; pins digests independently of GUI context).  
4. **C — Shared overlay infrastructure + selected QC artifacts** (after A/B contracts; do not ship every exporter this phase).

Defer model-training / configuration UIs until after S10 review UX and scientific validation of gallery/report workflows.

**D0 revision note (2026-11-01):** The thin “session write-back only” sketch in earlier drafts is **insufficient**. Approved direction is Project → Session → [Optional Subsession] → Subsystem Runs with a reference catalog (no science rewrite). Implementation is split into **D0-A…D0-D**; stop for architectural review before coding.

---

## 1. Current architecture inventory

### 1.1 Subsystem map (fact)

```text
S1 prepared_video + prepared_sync
        │
S2 pose → S3 tracking → S4 final_pose
S5 blob_features ──┐
                   ├→ S6 combined → S7 inference → S8 arbitration (exclusive + contact/SBS)
                   │                                      │
                   │                                      ▼
                   │                               S9 results_package__*
                   │                                      │
                   │                         optional results_exports/
                   │                         (media_export / csv_export)
                   │                                      │
                   └──────────────────────────────► S10 behavioral_review__*
                                                      (inspection / draft / publish)
```

Group 4 navigation: S9 packaging/export page beside S10 review page (`MainWindow` Group 4).

### 1.2 S9 packaging and media/CSV export (fact)

| Concern | Location | Behavior |
| --- | --- | --- |
| Science package | `src/results_packaging/` | Immutable `results_package__*`: `sync_index.parquet`, `behavior_bouts.parquet`, `ethogram_summary.parquet`, manifests, QC |
| Media/CSV jobs | `src/results_packaging/media_csv_export/` | Separate trees under `results_exports/`; never rewrite package science |
| Bout MP4s | `encode_mp4_clip` | Eager sequential ffmpeg; frame-trim (`trim=start_frame:end_frame`); libx264; prepared video source |
| GIF | — | Explicitly **out of scope** for S9 MVP (deferred U-R4 / E8) |
| Clip index | `clip_index.parquet` | Dual timing: dense behavioral endpoints + prepared seek + padded clip frames |
| Layout | `clips/{channel}/{behavior}/{package_bout_id}.mp4` | Stable ids; resume via ledger + `skipped_exists` |
| CSV | `behavior_bouts_automated.csv` | `export_provenance=automated`; pins package digests |
| Reviewed export | — | **Frozen deferred (S10 U6)** — exporters remain automated-package-only |

**Reuse for gallery/report:** `behavior_bouts.parquet`, `clip_index` + existing MP4s when a media job exists, `ethogram_summary.parquet`, package/export manifests for provenance footers, selection modes (`channels_behaviors`, `bout_id_list`).

**Not reusable as gallery engine:** eager full-package encode jobs. Gallery must be **lazy / on-demand**, not “export hundreds of GIFs first.”

### 1.3 S10 ethogram, bout navigation, editing (fact)

| Layer | Path | Key types |
| --- | --- | --- |
| Headless | `src/behavioral_review/` | `InspectionSession`, `EditingSession`, `EthogramAccessor`, `BoutBrowser`, `SyncMap`, `PreparedFrameAccessor`, journal/publish |
| GUI page | `src/ui/pages/behavioral_review_page.py` | Inspection + edit + publish workspace |
| Timeline | `src/ui/widgets/ethogram_timeline_view.py` | Dual exclusive lanes + contact/SBS |
| Video pane | `src/ui/widgets/pose_video_view.py` | Displays prepared frames (no behavior burn-in today) |
| Workers | Six `GuiTaskRunner`s | discovery / open / decode / window / edit / publish |

**Bout table columns (fact):** `bout_id`, `channel`, `behavior_name`, `focal_id`, `partner_id`, `start_sample_idx`, `end_sample_idx`, `status`.

**Edit ops (fact):** set exclusive label; set/clear contact or SBS pred; trim; split at playhead; accept/reject/unreviewed; notes; undo/redo. Headless also has `merge_adjacent_bouts` (no dedicated GUI button). Reciprocal exclusive tool **rejected** (U2 frozen).

**Scientific clock (fact):** dense 120 Hz playhead is authority; prepared decode via `sync_index.prepared_frame_idx`; plateau policy skips re-decode when prepared index unchanged.

### 1.4 Bout status and journal semantics (fact)

**Review status vocabulary:** `unreviewed` | `accepted` | `rejected` | `voided`  
(`uncertain` deferred — U3.)

**Binding rules (S10-B):**

| Rule | Meaning |
| --- | --- |
| R1 | Framewise-authoritative corrections; bouts regenerated via S8 `extract_behavior_bouts` only |
| R7 | Label/interval corrections, review status, and notes are **independent** stores |
| R8 / U15 | Reject is **status-only** — never clears labels |
| R9 | Geometry-changing ops void prior accept/reject (`void_reason=geometry_changed`) via **interval-overlap** matching (see R9 note below) |
| R10 | Opening a package never auto-accepts or auto-publishes |
| R15 | Append-only journal + linear undo/redo; fsync; cursor is the commit boundary; durable under `behavioral_review__*/working/` |

**R9 observed voiding behavior (fact — do not change in A2 closure):**

Geometry-changing corrections void active `accepted`/`rejected` decisions whose stored interval **overlaps** the edit’s touch mask on the same identity/channel (inclusive dense endpoints; opposite exclusive directions are isolated). Bout-scoped compile ops that carry `target_bout_id` void that target even without a numeric interval compare. This is broader than “only the edited bout id”: a relabel/trim/split can void another decision whose interval overlaps the touched span. Acceptance is never silently inherited onto regenerated bout identities after voiding.

**Follow-on:** a **targeted R9 audit** after A2 release is warranted (precision of overlap vs identity-only voiding; operator messaging when collateral voids occur). **Do not change R9 invalidation rules in this closure.**

**Channels (fact):**

- `exclusive` — directional; at most one label per `(sample_idx, focal_id, partner_id)`
- `contact`, `side_by_side` — independent `symmetric_pair` parallel channels

**Exclusive vocabulary (fact):** `nose_to_nose`, `nose_to_body`, `nose_to_tail`, `approach`, `retract`, `mutually_disengaged`, `engaged`, `undefined`.

### 1.5 Predicted labels vs corrected labels vs review decisions (recommendation — preserve fact distinctions)

| Concept | Authority | Storage | Gallery display |
| --- | --- | --- | --- |
| **Predicted (automated) label** | S8 → S9 `behavior_bouts` / referenced states | Immutable package artifacts | Baseline badge / “auto” column |
| **Corrected label** | Working or published reviewed states/bouts after framewise ops | Journal → `reviewed_behavior_states/bouts` | Effective label; mark `label_reconstruction=manual_review` when touched |
| **Review decision** | Operator accept/reject/unreviewed(/voided) | `review_bout_status` overlay | Status chip; **orthogonal** to label text |

Gallery and PI report must never collapse “accepted” into “label is ground truth,” nor treat missing accept as rejection. Contact/SBS decisions are optional overlays and must not be implied by exclusive acceptance (see §3).

### 1.6 Video decode, prepared-frame mapping, caching (fact)

| Piece | Role |
| --- | --- |
| `preprocess.video_probe.read_raw_frame_at_index` | Verified seek authority |
| S9 `sync_index.parquet` | Published dense↔prepared map; audits `max_many_to_one` |
| `behavioral_review.sync.SyncMap` | Playhead / bout seek mapping |
| `PreparedFrameAccessor` | Thread-safe LRU (default 8) + plateau skip |
| S5 `PreviewResultCache` | LRU ≤64 keyed by frame + settings fingerprint (contour overlays) |
| Bout duration | Inclusive dense count / 120 Hz; prepared indices are seek aids only |
| Clip padding | Uses **prepared fps**, never 120 Hz (`clip_bounds`) |

**Many-to-one:** multiple dense samples can share one prepared frame (E-session: dense length ≠ unique prepared span). Indices are not interchangeable.

### 1.7 GUI background workers (fact)

Shared: `src/ui/tasks.py` — `GuiTaskRunner`, `GuiTaskCoordinator`, cooperative cancel.

Typed kinds already cover S9 (`RESULTS_*`) and S10 (`BEHAVIORAL_REVIEW_*`). S3 remains UI-thread decode + `QTimer` (gap if overlay QC needs async decode). Gallery should follow the **S10 multi-runner + request-id stale rejection** pattern, not S3’s UI-thread loop.

### 1.8 Provenance and publication contracts (fact)

| Artifact | Pins |
| --- | --- |
| S9 package | S8/S7/S6 run ids + digests; prepared video path/digest; artifact SHA256s |
| S9 export job | Source package digests + video fingerprint + selection + clip listing |
| S10 review | `parent_package_run_id` + digests; `source_type=reviewed`; `auto_accepted_on_open=false` |
| Discovery | Classifications separate from historical `run_meta.status`; stale parent blocks publish (U12) |

Downstream must choose `SOURCE_TYPE_AUTOMATED` vs `SOURCE_TYPE_REVIEWED` explicitly — auto-picking latest reviewed without confirmation is forbidden.

### 1.9 Overlay capabilities today (fact)

| Subsystem | Overlay | Notes |
| --- | --- | --- |
| S1 | Crop / mask Qt paint | Geometry, not pose/behavior |
| S2 | `generate_overlay_video` → `overlay.mp4` | Nodes from `pose.parquet`; sequential prepared read; mp4v |
| S3 | Live `draw_pose_overlay` | Skeleton edges; no review-video export |
| S4 | None on video | |
| S5 | `draw_contour_overlay` | Preview modes + cache |
| S6–S8 | None | S8 has `generate_overlay_video: false` **config flag only** — no producer |
| S9 | Plain prepared clips | No ethogram/pose burn-in |
| S10 | Timeline/status overlays | Video pane is plain prepared frames |

### 1.10 Legacy Pipeline 100 / 110 (fact — local `legacy/`, git-excluded)

| Legacy piece | Finding |
| --- | --- |
| Pipeline 100 `VideoComposer` (`legacy/src/video/behavioral_overlay.py`) | Dense-loop QC exporter: darken → skeletons → focal HUD + contact/SBS LEDs → write at **target 120 fps**; uses imputed dense pose NPZ + predictions |
| S8 morph-close | Legacy Stage 8 contact morph; modern MVP disables via `max_gap_to_fill=0` |
| Pipeline 110 | Fused re-collapse + emit floors + MP4/GIF; source priority `tracks_overlay` → `prepared_video` → …; time-based ffmpeg seek |
| Modern S9 | Broke science/media coupling; frame-accurate trim; prepared-first; no GIF |

**Reuse as reference only:** HUD layout, plateau copy of prepared frames, layer ordering. **Do not** port dense-fps writer semantics or Pipeline 110 emit floors into S10 science.

### 1.11 Project / session / run discovery (fact)

| Concern | Today |
| --- | --- |
| Identity | Project root ≡ session root (directory with `preprocess/`) |
| Shared hint | `MainWindow._session_root` in-memory only |
| Persistence | QSettings stores S2/S7 tool paths — **not** last session / recent projects |
| Propagation gap | Page-local Browse on S2/S4–S10 does **not** write back `_session_root`; navigating away can reset to the remembered root |
| Run selection | Per-controller (e.g. S10 `selected_s9_run_id`) — correct scientifically |
| Safety | Soft-nav S5–S10; hard-lock S2/S4; discovery generation IDs; S10 closes draft lock on session change |
| Subsession | **None** — S1 is a singleton under `preprocess/`; no multi-recording child model |

**Revised target (recommendation):** see [`d0_workspace_architecture.md`](d0_workspace_architecture.md) — catalog-referenced Project → Session → optional Subsession → subsystem runs; existing trees referenced, not rewritten.

### 1.12 Reuse vs new infrastructure (summary)

| Capability | Verdict |
| --- | --- |
| Bout identity, filters, seek, timeline | **Reuse** S10 `BoutBrowser` / `BoutFilter` / timeline payloads |
| Framewise edits + journal + publish | **Reuse** unchanged; gallery actions call existing edit/status APIs |
| Eager S9 MP4 jobs | **Reuse as optional warm cache**; never required for gallery open |
| Lazy multi-preview decode | **New** preview scheduler + shared decode pool / disk preview cache |
| Behavior-grouped gallery UI | **New** S10 (or Group 4) surface on top of existing session |
| PI HTML bundle | **New** exporter; consume bout + preview + provenance contracts |
| Overlay compositor | **New** shared read-only layer API; adapt S2/S3/S5/legacy patterns |
| App workspace context | **New** `WorkspaceContext` + analysis-target resolver (not merely `_session_root`) |
| Workspace catalog | **New** sidecar JSON under `.behavior_suite/` — references paths/runs; **not** a science DB |
| Project database | **Not justified** |

---

## 2. Behavior-grouped bout gallery — interaction design

### 2.1 Goals (recommendation)

Enable efficient visual QC and exclusive-label approval by showing **all exclusive bouts grouped by effective behavioral label**, with multi-preview looping, outlier spotting, single-bout correction, bulk review actions, and deep-link into the existing S10 timeline/editor — without eager full-video GIF encoding.

### 2.2 Placement and source binding

| Decision | Recommendation |
| --- | --- |
| Host page | **S10** primary surface (same package/draft/publish lifecycle). Optional later “Results gallery” entry that opens S10 bound to a package |
| Default source | Opened S9 package (automated). If a working draft or published review is active, gallery shows **effective** labels/status from the review overlay |
| Binding | Digest-pinned package/run identity — same open path as S10-B §2.1 |
| Media prerequisite | **None.** Previews decode from prepared video via sync map; existing `media_export` MP4s are opportunistic hits |

### 2.3 Information architecture

```text
┌─ Filters ─────────────────────────────────────────────────────────┐
│ Focal mouse ▾ │ Label group ▾ │ Status ▾ │ Duration ▾ │ Sort ▾   │
└───────────────────────────────────────────────────────────────────┘
┌─ Label group: approach (n=42)  review: 12✓ 5✗ 25? ───────────────┐
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ...  (virtualized grid)              │
│ │vid │ │vid │ │vid │ │vid │                                      │
│ │meta│ │meta│ │meta│ │meta│                                      │
│ └────┘ └────┘ └────┘ └────┘                                      │
└───────────────────────────────────────────────────────────────────┘
┌─ Inspector (selected bout) ── deep-link: Open in timeline/editor ─┐
```

**Default grouping key:** effective exclusive `behavior_name` (corrected if review overlay present, else automated).  
**Secondary facets:** `focal_id`, `partner_id`, direction, review status, duration, `label_reconstruction`.

Contact and SBS appear as **optional secondary galleries / filters**, not as default exclusive groups (see §3).

### 2.4 Card contents (per bout)

Visible without opening inspector:

- Looping video preview (prepared-domain clip with configurable pad)
- Bout id (package or reviewed id as appropriate)
- Focal → partner
- Duration (dense frames and seconds @ 120 Hz)
- Effective exclusive label
- Review status chip (`unreviewed` / `accepted` / `rejected` / `voided`)
- Optional: “corrected” indicator when effective label ≠ automated baseline

### 2.5 Preview policy (lazy, not GIF-eager)

| Setting | Default recommendation | Notes |
| --- | --- | --- |
| Context pad | 0.25–0.75 s prepared-fps | Reuse `clip_bounds` semantics |
| Preview window | Bout span + pad, clamped | Same three-interval model as S9 clip_index |
| Max concurrent looping players | Small (e.g. 4–8 visible) | Pause off-screen; virtualize grid |
| Decode path | Async workers + LRU frame/clip cache | Extend `PreparedFrameAccessor` / S10 decode worker pattern |
| Disk cache | Optional short MP4/WebM under session cache dir keyed by `(video_digest, prepared_start, prepared_end, pad, scale)` | Not science; deletable |
| GIF | **Off** unless operator explicitly requests (gallery or PI export option) | Avoid hundreds of palette encodes |
| Existing S9 MP4 | If `encode_status=succeeded` for bout → use as preview source | Opportunistic |

**Hard rule:** Opening the gallery must not enqueue full-package eager GIF/MP4 generation.

### 2.6 Outlier detection within a label group (recommendation)

Presentation aids only — **not** scientific reclassification:

| Signal | Use |
| --- | --- |
| Duration vs group median / IQR | Flag short/long outliers |
| Automated vs corrected disagreement | Highlight relabeled bouts |
| High `undefined` / reconstruction flags | Flag weak evidence |
| Prob echoes from S8 (optional) | Sort “least confident accepted” |
| Reciprocal sibling mismatch (UI only) | Surface A→B vs B→A disagreement without merging rows (U8) |

Outlier badges never auto-reject or auto-relabel.

### 2.7 Single-bout inspection and correction

- Selecting a card seeks the existing ethogram timeline + video playhead (reuse `BoutSeekMapping`).
- “Open in editor” focuses the current S10 right-rail tools; requires explicit edit mode + `reviewer_id` for mutations (R6).
- Corrections remain framewise-authoritative; gallery refreshes effective label/status after journal apply (existing edit worker).
- Geometry edits void prior status (R9) — gallery must show voided → unreviewed transition clearly.

### 2.8 Selective and bulk review actions

| Action | Scope | Rules |
| --- | --- | --- |
| Accept / Reject / Unreviewed | Selected cards or filtered exclusive subset | Status-only APIs; no label mutation |
| Accept all visible in group | Exclusive label group ∩ current filters ∩ `unreviewed` | Confirm dialog; write per-bout status events; never imply contact/SBS acceptance |
| Reject all outliers | Explicit outlier set only | Confirm; status-only |
| Relabel selection | Edit mode | Compiles to `set_exclusive_label` over each bout’s dense interval; voids prior status |

Bulk actions must go through the durable journal (same as single edits). No silent in-memory-only verdicts.

### 2.9 S10 timeline/editor integration

| Gallery event | Existing target |
| --- | --- |
| Select bout | `seek_bout_start` / table selection |
| Double-click | Focus timeline segment + scrubber |
| Edit | Existing edit controls / workers |
| Publish | Existing publish gate (U12 stale parent block) |

Do not fork a second editing engine for the gallery.

### 2.10 Performance sketch

1. Build in-memory bout index from package (+ optional status overlay) — already cheap relative to video.  
2. Virtualized UI: only mount players for visible cells.  
3. Shared decode pool with request tokens (stale rejection like S10).  
4. Prefer scaled preview (e.g. ≤480 px width) for grid; full-res in inspector.  
5. Plateau-aware decode; never treat dense index as prepared frame.  
6. Optional background warm-up for next N cards in scroll direction only.

### 2.11 Data contracts for gallery (recommendation)

Headless API sketch (names illustrative):

```text
GallerySource =
  automated package | working draft | published review
  + pinned digests + prepared video identity

GalleryQuery =
  channels (default {exclusive})
  focal_ids, labels, statuses, duration_range
  group_by = exclusive_label | ...
  sort = duration | time | status | outlier_score

GalleryCard =
  bout_id, channel, focal_id, partner_id
  automated_label, effective_label, corrected: bool
  start/end sample_idx + prepared_frame_idx_*
  duration_samples, duration_sec
  status (effective), void_reason?
  preview_plan (prepared start/end/pad) | cache_key | optional existing_mp4_path
  outlier_flags[]
```

Science tables remain authoritative; gallery query results are **views**.

---

## 3. Review scope — exclusive-first approval

### 3.1 Default workflow (recommendation)

1. Operator reviews **exclusive** ethogram bouts in the grouped gallery.  
2. Default approval actions apply only to `channel=exclusive`.  
3. Session/package may be published with exclusive decisions incomplete or complete — existing publish rules already allow publish without every bout accepted; **do not** invent auto-accept.  
4. Contact and SBS remain visible (timeline + optional gallery mode) for inspection and explicit per-bout/per-interval decisions when the operator chooses.

### 3.2 What must not happen

| Forbidden | Why |
| --- | --- |
| Alter S8 schemas / arbitration / parallel predictions | User constraint; science freeze |
| Implicitly mark contact/SBS “accepted” when exclusive accepted | Parallel channels are independent (R3) |
| Treat excluded channels as reviewed-by-omission | Would invent provenance |
| Fold SBS/contact into exclusive groups as if ethogram labels | Violates channel grain |
| Use Pipeline 110 emit floors as science filters | Navigation-only (`review_emit_v1`) |

### 3.3 Explicit optional approval for contact / SBS

Recommendation: separate filter mode or tab — “Parallel channels” — with the same status API but **no** bulk “accept all exclusive” bleed-through. QC summary should report exclusive vs parallel review coverage separately (extend S10 QC summaries; do not change S8 outputs).

### 3.4 Label vs decision reminder

Exclusive **acceptance** means “operator endorses this bout interval’s effective exclusive label for PI consumption,” not “model was correct at prediction time.” Corrected labels can be accepted; rejected bouts retain labels (U15) and should be countable in reports as rejected, not deleted.

---

## 4. PI-facing export — immutable HTML report bundle

### 4.1 Product goal (recommendation)

Ship a **portable, immutable folder** a PI can open in a normal browser (double-click `index.html`) with no server and no Python runtime. It presents the same visual browsing experience as the gallery: grouped previews, filters, bout metadata, review status, and provenance.

### 4.2 Bundle layout (recommendation)

```text
pi_report__{timestamp}/
  index.html                 # self-contained app shell
  app.js / app.css           # no CDN required (vendored)
  data/
    manifest.json            # provenance + digests + source_type
    bouts.jsonl|parquet→json # bout cards + group indices
    ethogram_summary.json    # optional rollups
    review_status.json       # if reviewed source
  media/
    previews/{bout_id}.mp4   # browser-friendly, generated for included set
    # optional gifs/{bout_id}.gif  only if explicitly requested
  qc/
    report_qc.json
  README.txt                 # how to open offline
```

**Immutability:** write once under a timestamped directory; never mutate after finalize. Manifest lists SHA256 of data + media files.

### 4.3 Source selection

| Mode | When | Contents |
| --- | --- | --- |
| **Automated S9** | Default quick share / pre-review | Effective label = automated; status column absent or all `n/a` / `unreviewed` |
| **Selected S10 publication** | Post-review PI handoff | Effective labels + review status + correction summary counts; pin `behavioral_review__*` + parent package digests |

Recommendation: GUI explicit chooser — “Export PI report from: [S9 package ▾] | [S10 published review ▾]”. Working drafts are **not** exportable as PI reports (draft ≠ published). Reopening U6 for “reviewed media via S9 exporters” is **not** required if the PI report is a new S10/Group-4 exporter that pins reviewed publications directly.

### 4.4 Media strategy (video-first)

| Approach | Role |
| --- | --- |
| Primary | Per-bout **MP4** (or browser-safe H.264) previews with pad; generate only for bouts included in the report filter |
| Reuse | Copy/link from existing S9 `media_export` clips when digests/bounds match |
| GIF | Optional checkbox “Also generate GIFs” — off by default |
| Lazy in-app vs export | GUI gallery stays lazy; **export** materializes the filtered preview set into the bundle so offline viewing works |

Browser constraints: prefer `yuv420p` + `+faststart` fragmented/moov-at-front MP4 for `file://` playback testing on lab browsers; document supported browsers in README.txt.

### 4.5 HTML UX parity with gallery

- Group by exclusive label; filter by mouse / label / status  
- Grid of looping `<video>` elements (respect browser autoplay policies: muted loop)  
- Bout metadata + status chips  
- Provenance footer: session id, package/review run ids, digests, export timestamp, software version  
- Read-only: no editing in the HTML bundle

### 4.6 Contracts (recommendation)

```text
pi_report_manifest_v1:
  report_id, created_utc, behavior_suite_version
  source_type: automated | reviewed
  source_package_run_id + sha256s
  source_review_run_id? + publication_cursor?
  prepared_video_identity (path recorded; video NOT required at view time)
  selection: filters, included_bout_ids, preview_pad_sec, gif_requested: bool
  media_policy: video_previews | video_plus_gif
  artifact_digests{}
```

**QC gates:** every included bout has metadata row; every preview file exists and probes to expected frame count (±tolerance policy aligned with S9); manifest digests match; no draft sources.

### 4.7 Generation pipeline (recommendation)

New module (illustrative): `src/behavioral_review/pi_report/` or `src/results_packaging/pi_report/` owned by Group 4, calling:

1. Source validate (S9 eligibility or S10 published eligibility)  
2. Bout view build (gallery query contract)  
3. Preview materialize (ffmpeg; reuse `encode_mp4_clip` / clip_bounds)  
4. Static site write + atomic finalize  
5. Discovery entry under session `pi_reports/` (or `results_exports/pi_report__*`)

Does **not** mutate S8/S9 science trees.

---

## 5. Overlay infrastructure

### 5.1 Design goal (recommendation)

Introduce a **reusable, read-only overlay compositor** that can render selected layers onto prepared frames (live GUI and/or offline QC MP4), with explicit artifact provenance — without implementing every QC exporter in this phase.

### 5.2 Layer catalog

| Layer id | Source artifact | Domain | Notes |
| --- | --- | --- | --- |
| `s2.pose_inferred` | S2 `pose.parquet` | Prepared | Nodes; adapt `pose_inference.overlay` |
| `s3.pose_tracked` | Accepted S3 `tracked_pose.parquet` | Prepared | Reuse `draw_pose_overlay` + skeleton |
| `s4.pose_final` | S4 `final_pose.parquet` | Dense → prepared via sync | Show reconstruction / gap-fill indicators as annotations, not silent geometry |
| `s5.blobs` | S5 contours / mask (optional) | Prepared | Reuse `draw_contour_overlay`; optional QC only |
| `s8.exclusive_label` | S8/S9/S10 effective states | Dense → prepared | HUD text/color; plateau-stable |
| `s8.contact` / `s8.sbs` | Parallel preds | Dense → prepared | LED-style indicators (legacy VideoComposer reference) |
| `perspective.focal_A` / `focal_B` | Directional rows | Dense | Separate HUD perspectives; never merge directions |

### 5.3 Timing contract (binding)

1. Video pixels are always addressed in **prepared** frame indices.  
2. Behavioral labels are sampled on **dense** indices, then mapped with `sync_index` / `SyncMap`.  
3. When many dense samples map to one prepared frame, define a deterministic HUD policy (recommendation: **playhead dense sample** for interactive; for offline  prepared-rate export, use **last dense sample in plateau** or explicit “dense-rate export” mode — **operator decision**, see §8).  
4. Do not write overlays at 120 fps unless explicitly requested as a dense QC product; default QC overlays should match prepared fps to avoid huge files and timing confusion.  
5. Read-only: overlay rendering never writes back into pose/behavior tables.

### 5.4 Rendering API sketch (recommendation)

```text
OverlayContext(prepared_frame_idx, dense_sample_idx?, sync_map, artifacts[])
OverlayLayer.render(frame_bgr, context) -> frame_bgr
OverlayCompositor(layers[], style) 
  .render_frame(...)
  .export_mp4(plan)   # later selected exporters
```

Provenance sidecar for any exported overlay video:

```text
overlay_artifact_manifest_v1:
  layers[], source run ids + digests, sync_digest,
  frame_domain: prepared | dense_export,
  fps, codec, settings_fingerprint
```

### 5.5 Legacy findings applied

| Legacy VideoComposer idea | Modern stance |
| --- | --- |
| Skeleton + focal HUD + contact/SBS LEDs | Reuse as UX reference for `s8.*` layers |
| Dense 120 fps writer | Optional advanced mode only; not default |
| Darken / contrast mask | Optional style preset |
| Pipeline 110 overlay source priority | Gallery/PI default remains prepared; overlays are explicit opt-in (reopen U7 carefully) |
| mp4v vs libx264 | Prefer ffmpeg libx264 for distributable QC, consistent with S9 clips |

### 5.6 Phase-C delivery bound (recommendation)

Implement **shared compositor + 1–2 selected artifacts** this phase, for example:

1. Live optional pose layer in S10 inspector (S3/S4) — reopen U7 narrowly.  
2. Optional offline “exclusive HUD on prepared clips” for PI/debug — not full-session ethogram MP4.

Defer: full-session Pipeline-100-parity ethogram MP4; S5 blob QC movies; automatic overlay-as-S9-clip-source priority chain.

---

## 6. Shared project / session context (revised → D0)

**Authority for D0 detail:** [`d0_workspace_architecture.md`](d0_workspace_architecture.md). This section is a summary only.

### 6.1 Problem (fact)

`MainWindow._session_root` is a thin hint. Page-local Browse does not update it; cross-page navigation can desynchronize S9/S10 from the operator’s mental model. QSettings does not remember the last session. There is no Project → Session nesting, no subsession model, and no durable registration of existing lab folders. Selected runs correctly remain per-page, but gallery/PI workflows need a trustworthy **analysis target** plus explicitly pinned package/review identities.

### 6.2 Target model (recommendation)

```text
Project → Session → [Optional Subsession] → Subsystem Runs
WorkspaceContext:
  active project / session / optional subsession
  analysis_target_id + discovery_root
  signals + remember-only recent list
  # NOT: selected run ids, ethogram buffers, drafts
```

Rules (binding intent):

1. Page Browse/Open that changes analysis target **emits** shared context signals (fixes desync).  
2. Soft-nav / hard-lock preserved; never retarget an in-flight job.  
3. Changing session/subsession closes S10 draft locks when leaving the prior target; does not rewrite journals.  
4. Opening bouts / gallery / PI export always goes through S9/S10 package or publication identity + eligibility digests.  
5. Catalog references existing results; migration is idempotent and rollback = delete/rebuild catalog.  
6. Persist recent paths/ids only — never selected science runs in QSettings.  
7. Do **not** assume subsessions can reuse a parent’s S1 outputs (`preprocess_binding` is explicit).

### 6.3 Must this precede gallery?

| Gallery scope | Context needed first? |
| --- | --- |
| Strictly inside already-opened S10 session/package | **No** — can prototype A against current controller state |
| Cross-page / “open session → gallery” / multi-entry Group 4 | **Yes** — D0-A (+ session-level D0-D write-back) before declaring done |
| Multi-recording / temporal subsession targets | **Yes** — D0-C (operator W3 scope) |

**Recommendation:** implement **D0-A** in parallel with A; complete session-level GUI integration (D0-D) before trusting cross-page gallery; schedule D0-B migration early so lab folders register cleanly; trail D0-C unless multi-recording is in the same acceptance wave.

Do not globalize selected S8/S9 run ids.

---

## 7. Roadmap increments

### Increment A — S10 grouped bout gallery and approval UX

| Item | Content |
| --- | --- |
| **Reusable** | `BoutBrowser`, `BoutFilter`, `SyncMap`, `PreparedFrameAccessor`, ethogram timeline, edit/status/journal APIs, `GuiTaskRunner` decode pattern, optional S9 MP4 cache |
| **New** | Gallery query/index, virtualized multi-preview UI, preview scheduler/cache, outlier presentation aids, bulk status actions UX, exclusive-first approval chrome |
| **Data contracts** | `GallerySource` / `GalleryQuery` / `GalleryCard` (§2.11); no S8 schema changes; status via existing `set_bout_review_status` |
| **Scientific risks** | Bulk accept misinterpreted as ground truth; accidental parallel-channel implication; reciprocal grouping mistaken for merged science; emit floors applied as filters silently |
| **Performance risks** | Too many simultaneous decoders; UI-thread decode regression; cache growth; scroll storms |
| **Acceptance criteria** | Group exclusive bouts by effective label; filter focal/label/status; ≥4 concurrent looping previews without eager GIF job; card shows id/duration/status; select → timeline seek; single + bulk accept/reject status-only; edits durable in journal; contact/SBS not auto-accepted |
| **Tests** | Headless gallery query grouping; plateau/preview_plan bounds; bulk status journal events; GUI worker stale-token tests; no media-job prerequisite test |
| **Dependencies** | S10 MVP (done). D0 recommended in parallel. Not blocked on B/C |

#### Gallery A1 implementation record (2026-11-01) — **accepted**

**Shipped (A1 read-only browser only):**

| Concern | Implementation |
| --- | --- |
| Headless index | `src/behavioral_review/gallery.py` — `GalleryIndex` / `GalleryCard` / `GalleryQuery`; exclusive-only; group by effective label |
| Sources | Explicit `automated` (S9 package bouts) vs `published` (S10 reviewed bouts + status parquet). **Never merged** into one row set; review status is not silently transferred onto automated cards |
| Preview decode | `src/behavioral_review/gallery_preview.py` — `GalleryPreviewScheduler` over `PreparedFrameAccessor` + `SyncMap`/`clip_bounds`; lazy visible-only; no eager GIF/MP4 |
| GUI | S10 page Timeline \| Gallery toggle (`QStackedWidget`); `src/ui/widgets/bout_gallery_view.py`; filters for mouse / label / status / source |
| Deep-link | Double-click / open → dense playhead + direction + timeline highlight via `select_gallery_card` (no corrections, no draft creation) |
| Tests | `tests/behavioral_review/test_gallery_a1.py`; `tests/ui/test_behavioral_review_gallery_radio.py`; `tests/ui/test_behavioral_review_gallery_lifecycle.py` |

**Closure audit — Qt teardown / lifecycle (2026-11-01):**

| Finding | Classification | Evidence / action |
| --- | --- | --- |
| FIX-GALLERY-RADIO | Fixed earlier | Gallery radio `toggled` wired; click-path tests green |
| False-visible all cards under `QScrollArea` | **Gallery A1 lifecycle defect (fixed)** | Expanding preview + stretch + sync emit before layout settled marked ~538 E-session cards visible → decode thrash (`dropped_queue_count` thousands). Minimal fix: Preferred size policy, no trailing stretch, deferred visibility emit, reject collapsed widgets, page `shutdown_gallery_surface()` + scheduler closed-worker reject |
| Windows process abort after green assertions | **Pre-existing offscreen Qt limitation** (not a remaining unexplained A1 defect) | Same class as v0.1.1 note: monolithic/offscreen UI can abort at interpreter/Qt atexit (`0xC0000409`) after tests print passed. E-session smokes with live gallery previews + ordered shutdown printed `FUNCTIONAL_OK` and exited **0**; combined suites may still abort intermittently at process exit without failing assertions |
| E-session ordered teardown smoke | Pass | 538 exclusive cards; ~3 viewport-visible; scheduler `n_visible≈3`; `dropped_queue_count=0`; process exit code **0** |

A1 is accepted for read-only gallery use. Do **not** treat intermittent post-assertion Qt atexit aborts as A1 product regressions unless a new in-session crash reproduces with non-zero functional failure.

**Performance limits (A1 defaults, validated by focused tests):**

| Limit | Value | Notes |
| --- | --- | --- |
| Max concurrent preview decodes | **4** (`GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS`) | Chosen via `measure_preview_concurrency_capacity` helper + bounded-queue tests; not an unbounded pool |
| Max queued preview requests | **16** | Scroll storms drop oldest pending |
| Preview pad | **0.5 s** prepared-fps (`GALLERY_DEFAULT_PREVIEW_PAD_SEC`) | Reuses `compute_clip_bounds`; clamps at video edges |
| Decode domain | Prepared frames only | Dense endpoints remain authority; plateau skip via `read_frame_if_changed` |
| Source switch | Generation + `source_token` invalidate | Stale callbacks rejected; caches cleared |

**Source semantics (binding for A1):**

1. Automated gallery rows use packaged `behavior_bouts` identities (`package_bout_id` = `bout_id`); status defaults to `unreviewed`.
2. Published gallery rows use reviewed bout identities (`reviewed_bout_id`); lineage fields may point at package/S8 ids for deep-link and corrected badges only.
3. Opening Gallery never creates a draft, never appends journal events, and never requires a media_export job.

**Explicitly deferred to A2 / later (not in A1):**

- Bulk accept / reject / unreviewed for visible groups
- Single-card status actions from the gallery surface
- Outlier badges / sort-by-outlier
- Working-draft gallery overlay as a third source kind
- PI report export, overlay burn-in, GIF generation
- Contact/SBS secondary gallery mode

#### Gallery A2 implementation record (2026-11-01) — **accepted**

**Laboratory acceptance:** functional E-session acceptance passed (`tmp_s10_gallery_a2_lab/a2_acceptance_report.json`, 44/44). Journal durability defect (interrupted bulk JSONL append before cursor commit) remediated and fault-injection verified before closure.

**Shipped (A2 working-draft review decisions):**

| Concern | Implementation |
| --- | --- |
| Working-draft source | `GALLERY_SOURCE_WORKING` + `build_gallery_from_working_draft` over current regenerated exclusive bouts + live status overlay |
| Source kinds | Explicit `automated` \| `working_draft` \| `published` — never merged into one row set |
| Single Accept/Reject | Gallery actions call `EditingSession.set_bout_review_status` via existing edit worker (`EDIT_OP_SET_STATUS`) |
| Explicit multi-select | Ctrl+click additive selection; selected count shown; filters never imply selection |
| Bulk Accept/Reject | `EditingSession.set_bout_review_statuses` commits one undo group; exclusive working-draft cards only; no Accept All |
| Contact/SBS | Excluded from default gallery bulk/single eligibility (`channel=exclusive` only) |
| Editing entry | Inspection/gallery open never creates a draft; review actions explain enter-editing + draft workflow |
| Correct-and-return | Open in timeline/editor → edit worker → `refresh_gallery_after_mutation` rebuilds cards; stale selections pruned |
| Journal / undo | Reuses append-only journal; bulk = one `undo_group_id`; no independent write path |
| Journal commit boundary | Cursor atomic replace after event fsync; recovery discards only events beyond `redo_high_water_seq`; legitimate redo preserved; repairs write immutable `working/recovery_backups/` |
| Concurrency | Mutations only through background edit worker; GUI rebuilds gallery cards from already-projected tables |
| Tests | `tests/behavioral_review/test_gallery_a2.py`; `tests/ui/test_behavioral_review_gallery_a2.py`; `tests/behavioral_review/test_journal_fault_injection.py` |

**Working-draft gallery semantics (binding for A2):**

1. Cards use **current effective exclusive labels** and **regenerated bout geometry** from the open draft projection — never stale automated intervals after a correction.
2. Review status chips come from the live overlay (`unreviewed` / `accepted` / `rejected` / `voided`).
3. Status-only Accept/Reject never alters predictions, labels, or geometry (R7/R8/U15).
4. Geometry-changing corrections void prior accept/reject (`void_reason=geometry_changed`, R9); acceptance is not silently carried onto new bout identities.
5. Automated and published gallery sources remain **read-only**.

**Undo/redo behavior:**

- Single gallery status → one journal event / one undo step.
- Bulk gallery status → N events sharing one `undo_group_id` → one undo/redo restores the whole bulk.
- Undo/redo run on the edit worker and refresh the working gallery view.

**Performance notes (A2):**

| Measurement | Result |
| --- | --- |
| Synthetic E-session regroup (~538 exclusive cards) | Card regroup/status refresh **&lt; 250 ms** in focused unit test |
| Preview concurrency | Unchanged A1 defaults (max 4 concurrent / 16 queued) |
| Broad decode optimization | Not undertaken (no confirmed gallery decode regression in A2) |

**Known limitations:**

- No outlier badges / sort-by-outlier.
- No blanket Accept All / Accept visible group (explicit selection required).
- Contact/SBS have no secondary gallery mode yet.
- Working-draft gallery requires an already-open draft; switching to Working without a draft is blocked with guidance.
- Bout cards without lineage after large geometry rewrites resolve status via reviewed id / interval key when available; operators should re-review voided intervals.
- Intermittent offscreen Qt atexit aborts remain a pre-existing test-harness class (same as A1).

**Laboratory acceptance criteria (A2):**

1. Open automated gallery without creating a draft.
2. Enter editing + create/open draft; switch gallery source to Working draft.
3. Accept / Reject one exclusive card; confirm status chip and journal undo.
4. Explicitly multi-select several exclusive cards; Bulk Accept; one Undo reverts all.
5. Open a card in the timeline editor, relabel/trim, return to Gallery — geometry/labels refresh; prior acceptance does not silently stick to new identities.
6. Restart app / reopen draft — working gallery statuses persist.
7. Confirm contact/SBS are not bulk-accepted by exclusive gallery actions.
8. Confirm no S8/S9 science artifact mutation.

**Still deferred after A2:**

- Outlier badges / sort-by-outlier
- PI report export, overlay burn-in, GIF generation
- Contact/SBS secondary gallery mode
- Accept-all-visible-in-group convenience (explicitly rejected in favor of selection-only bulk)

### Increment B — PI-facing report export

| Item | Content |
| --- | --- |
| **Reusable** | Gallery card schema, `clip_bounds` / `encode_mp4_clip`, package & review manifests, ethogram summary |
| **New** | Static HTML/JS bundle writer, `pi_report_manifest_v1`, export job discovery, optional GIF flag path |
| **Data contracts** | §4.6 manifest; `source_type` automated\|reviewed; draft export forbidden |
| **Scientific risks** | Exporting drafts; mixing automated labels with reviewed status; omitting digests; GIFs treated as authority |
| **Performance risks** | Encoding huge unfiltered bout sets; UI freeze without worker; disk fill |
| **Acceptance criteria** | Offline `index.html` opens with filters + grouped muted looping videos; provenance visible; works without Python; GIF only if requested; reviewed vs automated explicitly labeled |
| **Tests** | Manifest digest round-trip; fixture tiny bundle playwright/smoke; reject draft source; media probe counts |
| **Dependencies** | A’s card/preview_plan contract (soft). Can prototype HTML against JSON fixtures before GUI gallery polish |

### Increment C — Shared overlay infrastructure and selected QC artifacts

| Item | Content |
| --- | --- |
| **Reusable** | S2/S3/S5 draw helpers, sync map, S10 video pane, ffmpeg encode patterns, legacy VideoComposer as reference |
| **New** | `OverlayLayer` / `OverlayCompositor`, provenance sidecar, 1–2 chosen exporters or live layers |
| **Data contracts** | `overlay_artifact_manifest_v1`; read-only rendering; explicit layer list |
| **Scientific risks** | Dense/prepared confusion in HUD; presenting reconstructed S4 points as raw measurements; directional perspective mix-ups |
| **Performance risks** | Full-session 120 fps exports; live multi-layer cost on GUI thread |
| **Acceptance criteria** | Compositor unit tests for plateau mapping; one live optional layer **or** one offline QC artifact with digest pin; no S8 science mutation; S8 `generate_overlay_video` flag either wired honestly or remains clearly unused |
| **Tests** | Layer golden frames (tiny); timing plateau tests; manifest pin tests |
| **Dependencies** | Sync/provenance clarity from A; not required for exclusive approval MVP. U7 reopen for S10 live overlay |

### Increment D — Shared workspace (revised; split D0-A…D0-D)

Full design: [`d0_workspace_architecture.md`](d0_workspace_architecture.md).

| Slice | Content |
| --- | --- |
| **D0-A** | Core `workspace_catalog_v1` types + `WorkspaceContext` + analysis-target resolver; MainWindow stops owning ad-hoc `_session_root` |
| **D0-B** | Scan/register existing session folders; detect runs via existing discovery; duplicates / missing / relocated; operator review; idempotent rescan; catalog rollback |
| **D0-C** | Optional subsessions (`temporal_segment` / `separate_recording`); explicit `preprocess_binding`; independent `subsessions/{slug}/` trees; no silent S1 reuse |
| **D0-D** | S1–S10 Browse write-back, status-bar identity, soft/hard nav vs analysis target, S10 draft/publish integrity, lab acceptance |

| Item | Content |
| --- | --- |
| **Reusable** | `ProjectService`, soft/hard nav, per-subsystem `run_discovery`, S10 draft lock close, request-id stale rejection |
| **New** | `src/workspace/*`, catalog sidecar, registration UX, optional subsession paths, optional new-write `workspace_ref_v1` |
| **Data contracts** | Catalog references paths/runs; science trees immutable; no global run-id store; no DB |
| **Scientific risks** | Silent provenance reassignment on context switch (mitigate with guards); shared-parent mistaken for independent lineage |
| **Acceptance criteria** | See D0 doc §7; session-level: Browse updates shared context across S9/S10; mid-task retarget blocked; existing E-session opens; catalog delete does not touch science |
| **Dependencies** | D0-A before cross-page gallery trust; D0-B for migration; D0-C only if multi-recording/segments scheduled; gallery/PI must not embed GUI-only context in science manifests |
| **Operator gates** | **W1–W7** in D0 doc §10 (project nesting, incomplete sessions, shared-parent science, auto-open, S1 versioning, run_meta ref, slug policy) |

### Proposed implementation order

```text
D0-A (schema + WorkspaceContext)
   ├─► D0-B (register / migrate existing sessions)
   ├─► D0-D session-level GUI integration  ─┬─► A (gallery + exclusive approval UX)
   │                                         ├─► B (PI HTML) [after GalleryCard/preview_plan]
   └─► D0-C (optional subsessions; may trail)┘

C (overlay compositor + selected artifacts) after A stabilizes inspector integration;
full ethogram QC movie exporters remain later.
```

Defer: model-training UIs, active learning, reciprocal tool (U2), `uncertain` (U3), unless explicitly reopened with lab criteria.

---

## 8. Open scientific / product decisions requiring approval

| ID | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| **P1** | Default PI report source | Automated S9 vs require reviewed S10 | Allow both with explicit chooser; label clearly |
| **P2** | Exclusive acceptance meaning | “Endorsed for analysis” vs “pixel-perfect GT” | Endorsed for analysis; document non-GT (matches S10 acceptance language) |
| **P3** | Bulk accept policy | Visible group / filtered unreviewed only / never bulk | Filtered unreviewed exclusive only + confirm |
| **P4** | Contact/SBS in PI report | Omit / appendix appendix / optional section | Optional section; default exclusive-only |
| **P5** | Reopen U6 vs new PI exporter | Extend S9 exporters for reviewed / new report module | **New PI report module** pinning reviewed pubs; leave S9 automated exporters unchanged |
| **P6** | Reopen U3 `uncertain` | Keep deferred / add for gallery triage | Keep deferred until lab defines criteria |
| **P7** | Reopen U2 reciprocal tool | Keep deferred / add for gallery sibling edits | Keep deferred; UI may **link** siblings only (U8) |
| **P8** | Gallery preview pad defaults | 0 / 0.25 / 0.75 s | 0.5 s starting point; user-configurable |
| **P9** | Offline overlay export frame domain | Prepared-fps vs dense 120 fps | Default prepared-fps; dense export opt-in |
| **P10** | Plateau HUD label policy | Playhead sample / last in plateau / majority in plateau | Interactive: playhead; offline prepared: last-in-plateau |
| **P11** | Auto-open last session | On / off / remember-only list | Remember-only list; no forced auto-open |
| **P12** | GIF in PI bundle | Never / explicit opt-in | Explicit opt-in only |
| **P13** | Whether gallery may warm-write preview cache under session | Yes cache / memory-only | Yes, deletable cache with digest keys |
| **P14** | Accept-all exclusive ⇒ publish readiness gate? | Soft coverage meter only / hard gate | Soft meter only (preserve R10 / no forced accept) |

**D0-related decisions** (do not silently decide): **W1–W7** in [`d0_workspace_architecture.md`](d0_workspace_architecture.md) §10 — project/session nesting, incomplete-path listing, shared-parent science scope, launch restore, S1 versioning, optional `workspace_ref_v1`, subsession path policy. P1–P14 matrix with D0/gallery/PI blocking notes is in that document §9.

---

## 9. Deliverable checklist (this document)

| # | Section | Status |
| --- | --- | --- |
| 1 | Current architecture inventory | §1 |
| 2 | Recommended gallery interaction design | §2 |
| 3 | PI report format | §4 |
| 4 | Overlay architecture and legacy findings | §5 + §1.10 |
| 5 | Shared project-context recommendation | §6 → revised; detail in `d0_workspace_architecture.md` |
| 6 | Proposed implementation order | §7 (D0 split A–D) |
| 7 | Open scientific/product decisions | §8 + D0 **W1–W7** |

---

## 10. Stop point

**D0:** shared workspace MVP closed (see [`d0_workspace_architecture.md`](d0_workspace_architecture.md)).  
**Gallery A1:** **accepted** — read-only behavior-grouped bout browser against the shared analysis target, including FIX-GALLERY-RADIO and teardown/visibility lifecycle closure above.

**Gallery A2:** **accepted** — working-draft source + single/bulk status-only actions through existing journal / edit-worker APIs; interrupted-append journal recovery closed with fault-injection coverage. No release tag from this acceptance.

**Still deferred:** PI report (B), overlays (C), subsession science processing, outlier aids, contact/SBS secondary gallery. Targeted **R9 overlap-voiding audit** recommended as a follow-on (rules unchanged here).

**Next authorized increment:** **B** (PI HTML) and/or outlier polish using the A1/A2 `GalleryCard` / preview_plan contracts; keep S8/S9 science schemas frozen.

_End of post-v0.1.1 S10 review and visualization architecture proposal._
