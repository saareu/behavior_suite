# Post-v0.1.1 — Overlay Architecture (C0 audit → C1 foundation → C2-A/C2-B QC export)

**Status:** **C3-3 accepted** (S5 Blob Segmentation QC Video). Builds on C1 + C2-A + C2-B + C2-C + C3-0 + C3-1 + C3-2.
**C3-3 base commit:** `6695c76d389a0a555aec6f6f00f6432009907057`
**C3-3 implementation date:** 2026-11-04
**C3-3 final acceptance date:** 2026-11-04
**C3-2 accepted:** 2026-11-04 (S3 Tracking Correction QC Video)
**C3-1 accepted:** 2026-11-04 (S2 Pose Inference QC Video)
**C3-0 accepted:** 2026-11-04 (S8 Behavioral QC Export Accessibility)
**C2-C base commit:** `f84ddd1714046fee477a18a8de805068a629bbdd`
**C2-C final acceptance date:** 2026-11-02
**C2-B final acceptance date:** 2026-11-02  
**C2-A / C1 final acceptance date:** 2026-11-02  
**C0 date:** 2026-11-01  
**C2-B base commit:** `a1da1e170297619b338a1d63f0c8239f5cbf5ea0` (`origin/main` — C1/C2-A + mainline CSV)  
**Companion:** [`post_v011_s10_review_and_visualization.md`](post_v011_s10_review_and_visualization.md) (Increment B closed; overlays → this document)  
**Non-scope (still):** full-session QC exporter, PI/gallery overlay integration, live overlays, S9/S10 overlay-media integration, scientific algorithm / schema mutation. S5 contour **QC replay** shipped in C3-3; S5/S9/S10 media burn-in remains deferred.  
**O4 closed:** S8 `execution.generate_overlay_video` is **deprecated and unsupported** (parsed for settings compatibility; runtime unchanged; never produced video).

**Legend**

| Mark | Meaning |
| --- | --- |
| **Fact** | Observed in current repository / contracts / local git-ignored `legacy/` |
| **Recommendation** | Proposed design pending operator approval; not binding |
| **Reuse** | Existing component that should be extended, not replaced |
| **New** | Infrastructure that does not exist today |

---

## 0. Executive summary

Modern S1–S10 already expose enough timing, pose, contour-preview, and ethogram contracts to design a shared **read-only overlay compositor**, but **no shared compositor exists**. Overlay drawing today is subsystem-local (S1 Qt geometry, S2 `overlay.mp4`, S3 live pose, S5 preview contours). S9 clips and S10/gallery/PI video remain naked prepared frames by design.

Legacy Pipeline 100 `VideoComposer` is a useful UX reference (darken → skeletons → exclusive HUD → counter; plateau frame copy) but must **not** be ported blindly: it hardcodes direction `0→1`, writes at dense 120 fps with OpenCV `mp4v` (docstring falsely claims H.264), and never actually draws contact/SBS LEDs despite helper methods.

**Recommended Increment C shape:** C1 contracts + compositor core + adapters; C2 S2/S3/S4/S8 layers + offline prepared-rate QC clip; C3 optional live S10 layer and/or full-session QC export. Keep existing naked-video exports unmodified by default.

---

## 1. Modern overlay-data inventory (S1–S10)

### 1.1 Cross-cutting domains (**fact**)

| Domain | Authority | Notes |
| --- | --- | --- |
| Raw / source pixels | Pre-S1 capture / cage ROI | Pose after S1 is **not** stored in raw space |
| Prepared pixels | S1 `prepared_video.mp4` | All S2+ pose `x,y` and S5 contours |
| Prepared frame index | S1 `prepared_sync.npz` | `prepared_frame_idx`, `raw_decode_frame_idx`, `prepared_time_sec` |
| Dense 120 Hz | S4+ joint grid | `sample_idx` ≡ `dense_frame_idx`; `TARGET_FPS_DEFAULT = 120.0` |
| Published map | S9 `sync_index.parquet` | Dense ↔ prepared; audits `max_many_to_one` |
| Interactive seek | S10 `SyncMap` | Playhead dense authority; prepared decode via map |

**Binding:** dense samples and prepared frames are **not** interchangeable. Many dense samples may share one prepared frame (plateau).

**Coordinate transform (**fact**):** `CropPlan.H_raw_to_prepared_3x3` / `H_prepared_to_raw_3x3` in S1 `prepare_meta.json`. Downstream overlays paint in **prepared** coordinates onto prepared video unless an explicit raw-source product is approved later.

### 1.2 Per-subsystem inventory

#### S1 — Preprocess

| Topic | Fact |
| --- | --- |
| Available data | `prepared_video.mp4`, `prepared_sync.npz`, `prepare_meta.json`, crop/mask geometry, `cropped_background.png` |
| Coords | Prepared pixels; raw↔prepared homographies in meta |
| Time | Prepared frame domain only |
| Artifact path | `{session}/preprocess/` |
| Viz code | Qt crop/mask in `src/ui/widgets/video_frame_view.py` |
| Limits | No pose/behavior; geometry overlays are not a BGR compositor API |

#### S2 — Pose inference

| Topic | Fact |
| --- | --- |
| Available data | `pose.parquet` long-form: `frame_idx`/`prepared_frame_idx`, `node`, `x`, `y`, `score`, `instance_score`, `track`/`instance_index`; `pose.slp`; meta |
| Coords | Prepared |
| Time | Prepared frame domain (not 120 Hz) |
| Artifact path | `{session}/pose_inference/<run_id>/` |
| Viz code | **Reuse** `pose_inference.overlay.generate_overlay_video` / `_draw_pose_rows` → `overlay.mp4` (nodes, identity colors, OpenCV `mp4v`) |
| Limits | No skeleton edges; no confidence styling; tracks may be provisional; codec differs from S9 `libx264` |

#### S3 — Tracking correction

| Topic | Fact |
| --- | --- |
| Available data | Accepted `tracked_pose.parquet` (same numeric handoff as S2: `frame_idx, track, node, x, y, score`); working/automatic tables; correction journals |
| Coords | Prepared |
| Time | Prepared frames |
| Artifact path | `{session}/tracking_correction/<run_id>/` |
| Viz code | **Reuse** `tracking_correction.overlay_draw.draw_pose_overlay` (nodes + optional `skeleton_edges`) via live S3 GUI |
| Limits | No review-video export; no per-point reconstruction categories in the table (identity/blank live in corrections + S4) |

#### S4 — Final pose

| Topic | Fact |
| --- | --- |
| Available data | `final_pose.parquet` dense long-form including **per-keypoint** `provenance` / `provenance_name`, pre-smooth coords, SLEAP scores where retained, source-frame indices, interpolation weight |
| Provenance codes (verified; do not invent others) | `0 missing`, `1 original`, `2 identity_corrected`, `3 temporal_resample`, `4 linear_gap_fill`, `5 kalman_gap_fill`, `6 protected_manual_blank` (`src/pose_finalization/contracts.py`) |
| Confidence honesty | SLEAP `point_score` retained for measured/identity points; **not fabricated** for resample/gap-fill |
| Coords | Prepared pixels on dense timeline |
| Time | Dense 120 Hz; `left_source_frame_idx` / `right_source_frame_idx` are prepared-domain source indices |
| Artifact path | `{session}/pose_finalization/<run_id>/` |
| Viz code | **None** on video (UI metrics only) |
| Limits | Drawing on prepared video requires dense→prepared mapping; S6 `is_imputed` maps only linear∪kalman (not `temporal_resample`) — overlay styling must not silently conflate these |

#### S5 — Blob segmentation / features

| Topic | Fact |
| --- | --- |
| Available data (parquet) | Scalar features only (`cc_count`, distances, shape metrics, temporal windows) + timing columns |
| Contours | Runtime `cv2.findContours` on `FrameSegmentationResult`; **not stored** in parquet |
| Coords | Prepared |
| Time | Dense table with `orig_frame_idx` ≈ nearest prepared; preview uses prepared frame index |
| Artifact path | `{session}/blob_features/<run_id>/` |
| Viz code | **Reuse** `blob_features.preview.draw_contour_overlay`; GUI modes + `PreviewResultCache` (LRU ≤64) |
| Limits | Historical contour overlay requires **re-segmentation** with settings fingerprint; no full-run contour movie artifact |

#### S6 — Combined features

| Topic | Fact |
| --- | --- |
| Available data | Dense dyadic feature parquet (`focal_id`, `partner_id`, kinematics, blob join cols, imputation quality flags) |
| Coords / time | Features derived from prepared-space pose on dense grid |
| Viz code | **None** |
| Limits | Not a pose overlay source; load S4/S5/S8 for drawing. Visually meaningful only as optional debug badges (imputation flags), not primary layers |

#### S7 — Behavioral inference

| Topic | Fact |
| --- | --- |
| Available data | `predictions.parquet` / specialist / wide tables: detector probs, `prediction`, validity, directional IDs |
| Time | Dense |
| Viz code | **None** (S10 can window evidence tables) |
| Limits | Not final exclusive ethogram; contact specialists ≠ arbitration outcomes. Probability burn-in is optional later QC, not C1 |

#### S8 — Behavioral arbitration

| Topic | Fact |
| --- | --- |
| Exclusive states | Directional rows `(dense_frame_idx, focal_id, partner_id)` + `behavior_label` / `pred_*` / `prob_*` |
| Directions | A→B and B→A are **separate** rows; ethogram lanes `exclusive:{focal}->{partner}` |
| Contact / SBS | Parallel channels (`pred_contact`, `pred_side_by_side`); bouts use `identity_grain="symmetric_pair"` with undirected pair IDs |
| Time | Dense only |
| Artifact path | `{session}/behavioral_arbitration/<run_id>/` |
| Viz code | **None**. Config `execution.generate_overlay_video: false` is parsed but **no producer exists** |
| Limits | Overlay must use final arbitration states (or S10 effective reviewed states when selected), never invent labels |

#### S9 — Results packaging / media export

| Topic | Fact |
| --- | --- |
| Science | `sync_index.parquet`, `behavior_bouts.parquet`, ethogram summary, manifests |
| Media | `encode_mp4_clip` — ffmpeg `libx264`, prepared-frame trim, naked clips under `results_exports/media_export__*` |
| Dual clocks | `clip_index` carries dense endpoints + prepared seek + padded clip frames at **prepared fps** |
| Viz | Plain MP4 only |
| Limits | Reviewed re-export frozen (U6). Overlay must not silently become default clip source |

#### S10 — Review / gallery / PI report

| Topic | Fact |
| --- | --- |
| Data | Inspection over package + optional review status/label overlays; gallery exclusive-first; PI offline HTML + naked MP4s |
| Time | Dense playhead authority; `PreparedFrameAccessor` plateau skip |
| Video pane | `PoseVideoView` displays pre-composited BGR; **no** behavior burn-in today |
| Timeline | `EthogramTimelineView` — dual exclusive + contact/SBS (not video overlay) |
| Limits | U7 prepared-only MVP for video source; shared compositor is **New** |

### 1.3 Existing draw / encode helpers (**Reuse**)

| Helper | Path | Role |
| --- | --- | --- |
| S2 overlay writer | `src/pose_inference/overlay.py` | Sequential prepared nodes → mp4v |
| S3 pose draw | `src/tracking_correction/overlay_draw.py` | Nodes + skeleton |
| S5 contours | `src/blob_features/preview.py` | Contour overlay |
| S9 clip encode | `src/results_packaging/media_csv_export/video.py` | ffmpeg libx264 trim |
| Sync | `src/behavioral_review/sync.py` | `SyncMap` plateau / seek |
| Frame cache | `src/behavioral_review/video.py` | Prepared LRU |
| Tasks | `src/ui/tasks.py` | Background runners + cancel |

**Missing (**fact**):** shared `OverlayLayer` / `OverlayCompositor`, S4 provenance styling, S8 HUD drawers, overlay provenance manifest, honest wiring of S8 `generate_overlay_video`.

---

## 2. Legacy Pipeline 100 findings

**Primary source (**fact**):** local git-excluded `legacy/src/video/behavioral_overlay.py` (`VideoComposer`).  
**Orchestration:** `legacy/pipelines/100_behavioral_arbiter.py` optional step after ethogram write.  
**Tests:** none found for `VideoComposer`.

### 2.1 What it actually does

| Concern | Legacy behavior |
| --- | --- |
| Source video | Prepared video (`prepared_video.mp4`) |
| Pose | Dense imputed NPZ (`dense_pose_120_imputed.npz`) |
| Labels | In-memory final social predictions table |
| Output | `reports/behavioral_overlay/.../behavioral_overlay.mp4` |
| Loop | One written frame per **dense** sample at `target_fps` (typically 120) |
| Plateau | Copy previous prepared pixels when `orig_frame_idx` unchanged; re-draw overlays |
| Render order | Darken (`alpha=0.9`) → skeletons mice 0/1 → focal exclusive HUD → frame counter |
| Encode | OpenCV `VideoWriter` FourCC **`mp4v`** (docstring incorrectly says H.264); **no ffmpeg** |
| Identity colors | Hardcoded cyan/magenta; `% 2`; no ID glyphs; no implanted/NI role |
| Focal labels | Top-center colored pill from direction **hardcoded `focal_id=0, partner_id=1` only** |
| Partner perspective | **Never rendered** |
| Contact/SBS | `_draw_context_hud` **defined but never called** — LEDs are dead code |
| Headstage | Ignored (4-node skeleton) |

**Correction to earlier roadmap wording:** §1.10 of the S10 visualization doc previously summarized “focal HUD + contact/SBS LEDs” as shipped behavior. Code audit shows only the exclusive HUD is drawn.

### 2.2 How focal behavioral labels are presented

1. Prefer `behavior_label` string if present and not `none`/`nan`.  
2. Else first true `pred_*` among exclusive vocabulary.  
3. Else display `"disengaged"`; rename `mutually_disengaged` → `"disengaged"`.  
4. Draw filled rectangle at `(width//2, 40)` with `BEHAVIOR_COLORS[label]` + white `FONT_HERSHEY_SIMPLEX` text.  
5. `neck_pos` parameter exists but is always `None` — HUD is screen-fixed, not pose-anchored.

### 2.3 Compatible modern output (**recommendation**)

Preserve UX cues worth keeping; reject scientific/layout failures:

| Keep as reference | Do not copy blindly |
| --- | --- |
| Darken → skeleton → exclusive pill → optional LEDs → counter | Default 120 fps full-session writer |
| Plateau pixel reuse | Hardcoded `0→1` direction |
| Behavior color map / short display names | `mp4v` labeled as H.264 |
| Dense science authority for labels | OpenCV random seek as primary long-export path |
| Optional partner perspective as **explicit second mode or second HUD** | Silent dead features documented as shipped |
| | Coupling overlay into arbitration publish |

**Proposed compatible product:** prepared-rate (default) QC video with independently selectable layers; exclusive HUD driven by selected directional perspective (`A→B` default, `B→A` optional, optional dual HUD). Contact/SBS as separate optional LED layer. Encode via ffmpeg `libx264` + provenance sidecar.

---

## 3. Shared compositor architecture (**recommendation**)

### 3.1 Separation of concerns

```text
┌──────────────────────────────┐
│ Export orchestration (GUI /  │
│ jobs / PI optional path)     │
└──────────────┬───────────────┘
               │ read-only plans
┌──────────────▼───────────────┐
│ Timing alignment             │
│ SyncMap / sync_index /       │
│ plateau policy / bout bounds │
└──────────────┬───────────────┘
               │ OverlayContext
┌──────────────▼───────────────┐
│ Scientific data adapters     │
│ (S2/S3/S4/S5/S8/S9/S10)      │
│ load tables; never recompute │
│ science algorithms           │
└──────────────┬───────────────┘
               │ layer observations
┌──────────────▼───────────────┐
│ Drawing / rendering          │
│ OverlayLayer.render(BGR)     │
│ OverlayCompositor compose    │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Video encoding               │
│ Prefer S9 ffmpeg libx264;    │
│ streaming frame pipe         │
└──────────────────────────────┘
```

**Hard rule:** renderer never modifies or reruns scientific processing. Contour layers may **replay** stored S5 settings to re-segment for draw (preview pattern), but must not rewrite S5 artifacts.

### 3.2 Candidate modules (**New** under something like `src/overlay/` or `src/visualization/overlay/`)

| Module | Responsibility |
| --- | --- |
| `contracts.py` | Layer ids, `overlay_artifact_manifest_v1`, frame-domain enums, unavailable markers |
| `context.py` | `OverlayContext(prepared_frame_idx, dense_sample_idx, sync_map, style, provenance refs)` |
| `adapters/*` | Read-only loaders for S2/S3/S4/S5/S8/S9/S10 effective states |
| `layers/*` | Pure draw functions per layer |
| `compositor.py` | Ordered compose; skip unavailable layers with explicit watermark/badge |
| `timing.py` | Plateau policy helpers wrapping `SyncMap` |
| `export.py` | Streaming export plan → ffmpeg; cancel hooks |
| `style.py` | Colors, fonts, darken preset (optional) |

### 3.3 Layer catalog (independently selectable)

| Layer id | Adapter source | Draw notes | Unavailable when |
| --- | --- | --- | --- |
| `s2.pose_inferred` | S2 `pose.parquet` | Nodes; label HUD “inferred pose” | No S2 run / missing frame rows |
| `s3.pose_tracked` | Accepted S3 tracked pose | Skeleton; label “tracking-corrected identities” | No accepted S3 |
| `s4.pose_final` | S4 `final_pose` + sync | Provenance styling; never fabricate categories | No S4 / missing sync |
| `s5.segmentation` | Re-segment with S5 settings | Contours; QC only | Missing settings fingerprint / video |
| `s8.exclusive.A_to_B` | Final (or reviewed) exclusive row | Top-center or near-mouse pill | Missing directional row |
| `s8.exclusive.B_to_A` | Opposite exclusive row | Independent toggle | Missing opposite row |
| `s8.contact` | Parallel contact pred | LED; not exclusive ethogram | Channel absent |
| `s8.sbs` | Parallel SBS pred | LED; separate from contact | Channel absent |
| `debug.frame_counter` | Context indices | Dense/prepared/time | Optional |

S6/S7 are **not** default C layers. Optional later debug badges only.

### 3.4 API sketch

```text
OverlayObservation  # geometry / labels / availability for one layer at one context
OverlayLayer.render(frame_bgr, context, observation) -> frame_bgr
OverlayCompositor(layers[], style).render_frame(frame_bgr, context) -> ComposedFrame
OverlayExporter.export_mp4(plan, compositor) -> OverlayArtifactRef
```

Manifest (`overlay_artifact_manifest_v1`):

```text
layers[], source run ids + digests, sync_digest,
frame_domain: prepared | dense_export,
plateau_policy, perspectives[],
fps, codec, settings_fingerprint, unavailable_layers[]
```

---

## 4. Timing and coordinate contracts (**binding recommendations**)

### 4.1 Index domains

| Index | Meaning | Authority |
| --- | --- | --- |
| `raw_decode_frame_idx` | Source/decode index before/at S1 | S1 sync |
| `prepared_frame_idx` | Frame in `prepared_video.mp4` | S1 / S9 sync |
| `sample_idx` / `dense_frame_idx` | Dense 120 Hz analysis sample | S4–S10 / S9 sync_index |

**Never** treat dense index as a prepared seek index.

### 4.2 Source → prepared coordinates

- Pose and contours after S1: already prepared pixels.  
- If painting on raw video ever approved: apply `H_prepared_to_raw_3x3` (or inverse of raw→prepared) from `prepare_meta.json`; pin digest of meta in manifest.  
- Default C products: prepared video only (preserves U7 spirit).

### 4.3 Dense → prepared mapping

- Use S9 `sync_index` / S10 `SyncMap.mapping_at_sample` for interactive playhead.  
- `SyncMap.mapping_from_prepared_frame`: nearest dense `time_sec` to plateau `prepared_time_sec`; **earlier `sample_idx` wins ties** (existing S10-B contract).  
- `plateau_sample_indices(prepared_frame_idx)` enumerates many-to-one members.

### 4.4 Plateau HUD policy (**approved for C1**)

| Mode | Policy |
| --- | --- |
| Interactive live | Use **playhead** dense sample (already authoritative) — C3 |
| Offline prepared-rate export | Use **last dense sample in plateau** (deterministic; matches “state as of this displayed frame’s end”) — **C1 default** |
| Offline dense-rate export (opt-in) | One output frame per dense sample; may reuse prepared pixels across plateau — deferred |

Document policy id in manifest. Do not silently majority-vote labels.  
**Note:** `SyncMap.mapping_from_prepared_frame` nearest-time lookup ≠ last-in-plateau aggregation; both are available in `overlay.timing`, with offline default = last-in-plateau.

### 4.5 Missing / invalid observations

- Missing pose coords / `provenance=missing` / protected blank: do not draw fabricated points; optional hollow marker only if explicitly styled as missing.  
- Invalid S7/S8 inputs: show layer **unavailable** or last valid exclusive policy only if science already defined it — compositor must follow published states, not invent fill.  
- Adapter failure → layer marked unavailable in compose result + manifest; optional on-frame badge `"s4.pose_final unavailable"`.

### 4.6 Bout / video boundaries

- Inclusive dense bout endpoints remain science authority.  
- Prepared seek aids from `SyncMap.bout_seek` / clip_index.  
- Clip padding uses **prepared fps**, never 120 Hz (S9 fact).  
- Overlay export of a bout: iterate prepared frames in padded clip range (default) or dense samples (opt-in); both pin sync digest.

### 4.7 Provenance / determinism

- Tie-breaks: reuse SyncMap earlier-sample ties; plateau offline last-in-plateau if approved.  
- Every exported overlay pins: layer list, perspectives, sync digest, source run digests, plateau policy, frame domain, codec/fps.

---

## 5. Stage-specific scientific honesty

| Stage | Overlay honesty rules |
| --- | --- |
| S2 | On-frame / legend text: **“inferred pose”**; do not imply identity-final |
| S3 | Legend: **“tracking-corrected identities”**; skeleton from accepted tracks |
| S4 | Distinguish observed (`original` / `identity_corrected`) from reconstructed (`temporal_resample`, `linear_gap_fill`, `kalman_gap_fill`); never fabricate confidence for imputed points; do not invent provenance categories |
| S5 | Contours are QC replay, not stored truth polygons; mark settings fingerprint |
| S8 exclusive | Exact final arbitration `behavior_label` (or explicit reviewed effective source); directions independently selectable; never merge A→B and B→A into one banner without dual presentation |
| Contact / SBS | Separate parallel indicators; never imply exclusive membership |
| Unavailable | Explicit unavailable marking — no silent omission that looks like “no behavior” |

**S4 styling note:** S6 `is_imputed` excludes `temporal_resample`. Overlay legend should either (a) three visual classes (observed / temporal_resample / gap_fill) or (b) operator-approved mapping — do not silently call temporal_resample “observed.”

---

## 6. S5 feasibility

| Question | Answer |
| --- | --- |
| Can contours be drawn today? | **Yes** at preview time via re-segmentation + `draw_contour_overlay` |
| Are polygons in parquet? | **No** |
| Full-session contour movie? | Not present; expensive; defer past C1 |
| C recommendation | Optional `s5.segmentation` layer for live/QC clips only; require settings fingerprint match; mark unavailable if replay unsafe |

---

## 7. Export integration plan

| Consumer | Default today | Overlay integration (**recommendation**) |
| --- | --- | --- |
| S9 media export bout MP4s | Naked prepared clips | **Remain unmodified by default.** Optional separate job `media_export_overlay__*` or layer plan flag — never rewrite existing clip trees |
| S10 gallery preview | Naked prepared decode | Optional live compose in preview worker later (C3); cache keys must include layer fingerprint |
| PI HTML report | Naked MP4s in bundle | Default stays naked. Optional “include overlay clips” checkbox pins overlay digests separately |
| Full-session QC video | Absent (legacy only) | New optional exporter under results_exports; prepared-rate default; dense-rate advanced |
| S2 `overlay.mp4` | Existing nodes-only product | May later call shared drawer; keep path/compat until migrated |
| S3 live review | Live draw | Migrate draw call sites to shared layer when ready |
| S8 `generate_overlay_video` | Config flag unused | Wire honestly to optional exporter **or** remove/rename flag — do not leave dishonest |

**Avoid:** forcing overlays into existing naked-video exports; Pipeline 110 overlay-source priority chains; making PI/gallery require overlays.

---

## 8. Performance architecture (**recommendation**)

| Concern | Approach |
| --- | --- |
| Streaming render | Decode prepared frame → compose layers → write/pipe; do not hold full session RGB |
| Background | `GuiTaskRunner` / coordinator pattern (S9/S10/PI); request-id stale rejection |
| Memory | Bound pose row caches by plateau/window; reuse `PreparedFrameAccessor` LRU; S5 preview LRU pattern |
| Reuse structures | Pre-index parquet by frame/sample; skeleton edge list once; color tables once |
| Cancellation | Cooperative between frames/clips; document in-flight ffmpeg limits (known PI/S9 constraint) |
| Long sessions | Prefer prepared-rate; chunked progress; avoid default 120 fps full session |
| Windows FFmpeg | Prefer existing `encode_mp4_clip` / preprocess ffmpeg runtime; careful temp→final replace (known WinError 5 under load) |
| Live GUI | Do not block UI thread (S3 QTimer gap); compose off UI or cheap single layer only |

Prefer existing supported video infrastructure (S9 ffmpeg, S1 probe/read helpers) over new OpenCV writers for distributable artifacts.

---

## 9. Proposed implementation increments

### C1 — Contracts, timing, compositor core, adapters (no full exporters)

| Item | Content |
| --- | --- |
| Files / components | `src/overlay/` (or `src/visualization/overlay/`): `contracts`, `context`, `timing`, `compositor`, adapters stubs for S2/S3/S4/S8; thin wrappers reusing `draw_pose_overlay` |
| Data contracts | Layer ids; `overlay_artifact_manifest_v1` schema; unavailable observation type; plateau policy enum |
| Tests | Sync plateau mapping golden tests; compositor no-op/unavailable badge; adapter schema guards; read-only (temp dirs unchanged) |
| Scientific risks | Dense/prepared confusion — mitigated by SyncMap-only mapping |
| Performance risks | Low (unit-scale) |
| Acceptance | Unit tests green; no science mutation; docs list open decisions still pending for C2 exporters |
| Dependencies | Existing SyncMap; operator lean on P9/P10 preferred |

### C2 — Honest layers + offline prepared-rate QC clip exporter

| Item | Content |
| --- | --- |
| Files / components | Layers: `s2`, `s3`, `s4` provenance style, `s8.exclusive` dual perspective, `s8.contact`/`s8.sbs`; `export.py` streaming ffmpeg; optional CLI/GUI QC action under results_exports |
| Data contracts | Manifest pins; perspective modes; S4 visual class map (approved) |
| Tests | Golden tiny frames per layer; plateau last-sample HUD; dual-direction independence; encode smoke; unavailable layers |
| Scientific risks | Mislabeling reconstructed pose; merging directions; treating contact as exclusive |
| Performance risks | Bout-length encode OK; forbid accidental full-session 120 fps as default |
| Acceptance | Offline QC artifact for a bout or short span with digest pin; naked S9 export unchanged; S8 flag wired or clearly deprecated |
| Dependencies | C1; P9/P10 resolved; S4 styling decision |

### C3 — Live S10 optional layer and/or full-session prepared QC

| Item | Content |
| --- | --- |
| Files / components | S10 video pane compose path; gallery preview optional overlay; full-session prepared-rate exporter; optional S5 layer |
| Data contracts | Live context includes playhead dense sample; cache keys include layer fingerprint |
| Tests | Live plateau skip still works; cancel; memory bound; PI optional overlay checkbox isolation |
| Scientific risks | Reopening U7 for live overlay source clarity; reviewed vs automated label source confusion |
| Performance risks | GUI jank; long-session IO; S5 re-segment cost |
| Acceptance | Optional live pose **or** exclusive HUD in S10 without changing default naked decode; full-session export optional and cancellable |
| Dependencies | C2; narrow U7 reopen if live overlay ships; gallery cache policy if previews overlay |

### Operator decisions — C1 resolutions and remaining

**Resolved for C1 (approved defaults):**

| ID | Resolution |
| --- | --- |
| **P9** | Video output operates in **prepared-video** frame space and FPS. Dense 120 Hz remains the authoritative behavioral timeline. |
| **P10** | Prepared-frame plateaus use the **last mapped dense sample** (chronological max `sample_idx`). No dense coverage ⇒ behavioral state unavailable (do not invent labels). |
| **O5** | Overlays remain **opt-in**. |
| **O6** | **No S5** contour generation in C1. |
| **O8** | Automated S8 and published reviewed S10 source identities stay **separate** (pinned in observations + manifest). |
| Live | **No live overlays** in C1. |
| S8 flag | **No implicit behavior** from unused `execution.generate_overlay_video`. |

Still open for C2 (styling / export product): **O1**, **O2**, **O3**, **O4**, **O7**, and careful **U7** reopen if live overlay ships in C3.

---

## 10. Open decisions requiring operator approval

| ID | Decision | Options | Recommendation | C1 status |
| --- | --- | --- | --- | --- |
| **P9** | Offline overlay export frame domain | Prepared-fps vs dense 120 fps | Default prepared-fps; dense opt-in | **Resolved:** prepared default |
| **P10** | Plateau HUD label policy | Playhead / last-in-plateau / majority | Interactive playhead; offline prepared last-in-plateau | **Resolved:** last-in-plateau offline |
| **O1** | C2 first artifact | Bout QC overlay clip vs live S10 layer vs both | Offline bout/span QC first (safer performance) | Open (C2) |
| **O2** | Partner perspective UX | Toggle single HUD vs dual HUD always-available | Toggle + optional dual strip; never merge | Open (C2); dual layers exist |
| **O3** | S4 `temporal_resample` styling | Group with observed / with imputed / third class | Third class (honest; matches S6 distinction) | Open (C2); codes faithful in C1 |
| **O4** | S8 `generate_overlay_video` flag | Wire to exporter / remove / rename | Wire to optional exporter or deprecate explicitly | **Closed in C2-B:** deprecated/unsupported; runtime unchanged |
| **O5** | Overlay in PI / gallery defaults | Off / opt-in / on | Off by default; explicit opt-in | **Resolved:** opt-in |
| **O6** | S5 in C scope | Defer / C3 optional | Defer full movies; optional layer in C3 only | **Resolved:** deferred |
| **O7** | Near-mouse vs top-center exclusive HUD | Legacy top-center / pose-anchored / both | Start top-center (legacy familiarity); pose-anchored later | Open (C2); C1 uses top-center stub |
| **O8** | Reviewed vs automated labels for overlays | Package automated only / allow reviewed effective | Allow explicit source chooser; pin in manifest | **Resolved:** explicit separate sources |

Carried from S10 roadmap: careful **U7** reopen for live overlay (C3).

---

## 11. Documentation / roadmap linkage

- This document is the Increment **C** architecture authority for overlays (C0 audit + C1 implementation record).  
- [`post_v011_s10_review_and_visualization.md`](post_v011_s10_review_and_visualization.md): Increment B marked closed; Increment C points here.  
- Production package: `src/overlay/` (see §13).

---

## 12. C0 stop point (historical)

**C0 complete as architecture/source audit** (2026-11-01). Operator approved C1 defaults (P9/P10/O5/O6/O8 + no live / no S5 / no S8 flag wiring) and authorized C1 implementation.

---

## 13. Increment C1 — implemented foundation (2026-11-02)

### 13.1 Package layout

```text
src/overlay/
  contracts.py      # LayerId, FrameDomain, PlateauPolicy, S4 provenance mirrors
  timing.py         # prepared→dense resolution (last-in-plateau default)
  context.py        # OverlayObservation / OverlayEvidenceBundle / OverlayContext
  style.py          # DrawConfig
  evidence.py       # collect_evidence
  compositor.py     # OverlayCompositor.render_frame (offline, copy-only)
  provenance.py     # overlay_artifact_manifest_v1
  adapters/         # S2/S3/S4/S8/S10 read-only; S5 stub
  layers/           # minimal pose / exclusive HUD / LEDs / debug
tests/overlay/      # timing, adapters, compositor, provenance, E-session timing
```

Registered in `pyproject.toml` (setuptools / ruff first-party / mypy / coverage).

### 13.2 Contracts implemented

| Contract | Detail |
| --- | --- |
| Layer ids | `s2.pose_inferred`, `s3.pose_tracked`, `s4.pose_final`, `s5.segmentation` (stub), `s8.exclusive.A_to_B`, `s8.exclusive.B_to_A`, `s8.contact`, `s8.sbs`, `debug.frame_counter` |
| Plateau policy | `last_in_plateau` (offline default), `nearest_time` (SyncMap contrast), `playhead` (rejected offline) |
| Frame domain | `prepared` default; `dense_export` reserved |
| Behavior sources | `s8_automated` vs `s10_reviewed` kept distinct |
| S4 provenance | Codes 0–6 only; unknown codes raise; no fabricated categories |
| Manifest | `overlay_artifact_manifest_v1` — source runs/digests, prepared-video identity, timing mapping identity, plateau policy, layers, renderer version, output domain/FPS, render settings |
| Opt-in | Empty layer selection rejected; no S8 flag side effects |

### 13.3 Timing / plateau

`resolve_prepared_frame_timing(sync, prepared_frame_idx, plateau_policy=LAST_IN_PLATEAU)` wraps authoritative `SyncMap` / `sync_index` only.

- **Last-in-plateau:** `max(plateau_sample_indices)` chronological latest.  
- **Nearest-time:** delegates to `SyncMap.mapping_from_prepared_frame` (documented difference; not offline HUD default).  
- Missing / out-of-range prepared indices → `available=False` (no invented dense sample).  
- Does **not** use independent FPS rounding in place of sync mappings.

### 13.4 Adapter coverage (C1)

| Adapter | Status |
| --- | --- |
| S2 inferred pose | Implemented (prepared-domain; legend “inferred pose”) |
| S3 tracked pose | Implemented (prepared-domain; legend “tracking-corrected identities”) |
| S4 final pose | Implemented (dense-domain via timing; faithful provenance) |
| S8 exclusive A→B / B→A | Implemented (automated source kind) |
| S8 contact / SBS | Implemented (parallel channels) |
| S10 reviewed exclusive | Implemented as distinct source kind (same layer id when selected; identity pinned) |
| S5 segmentation | Explicit unavailable stub |

### 13.5 Rendering capabilities (C1)

- Offline single-frame compose; source array immutability.  
- Minimal pose nodes via reused `tracking_correction.overlay_draw.draw_pose_overlay`.  
- Minimal dual exclusive HUD + contact/SBS LED placeholders (final styling deferred).  
- Unavailable-layer badge.  
- Debug prepared/dense counter.  
- **Not in C1:** full-video export, ffmpeg encode, GUI/live path, S5 contours, final S4 observed/reconstructed styles.

### 13.6 Deliberate deferrals

- S5 contour generation / full-session contour movies.  
- Live S10 compose / gallery / PI overlay paths.  
- Streaming `export.py` / bout QC MP4 (C2).  
- Final S4 visual class map and behavior color system.  
- Wiring or removing S8 `generate_overlay_video`.  
- Dense-rate export path (opt-in later).

### 13.7 C2 prerequisites

1. Approve **O1** (bout/span QC clip as first export artifact).  
2. Approve **O3** S4 visual class mapping (recommend third class for `temporal_resample`).  
3. Approve **O2**/**O7** HUD UX (dual strip vs toggle; top-center first).  
4. Decide **O4** for the unused S8 overlay flag (wire vs deprecate).  
5. Implement `export.py` streaming prepared-rate ffmpeg encode reusing S9 helpers; pin full manifest.  
6. Flesh layer styling (S4 provenance classes, dual-direction HUD layout, contact/SBS LEDs).  
7. Golden tiny-frame tests per layer + encode smoke; keep naked S9 exports unmodified.  
8. No GUI integration required for C2 offline QC; keep overlays opt-in.

### 13.8 Stop point (historical)

**C1 complete as shared compositor foundation.** Operator authorized C2-A offline behavioral QC export next.

---

## 14. Increment C2-A — S8 behavioral QC video export (2026-11-02)

### 14.1 Operator-visible product

Offline prepared-rate behavioral QC MP4 under:

```text
{session}/results_exports/overlay_qc_behavioral__{timestamp}/
  behavioral_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
```

- Default short clip (explicit prepared interval or bout/span); not full-session.
- H.264 / yuv420p via ffmpeg pipe; prepared-video FPS (no 120 fps duplication).
- Never overwrites a prior QC job directory / artifact.
- Naked S9 media exports and PI-report media remain unmodified.

### 14.2 Source selection and eligibility

| Source | Gate | Notes |
| --- | --- | --- |
| Automated S8 via eligible S9 package | `open_review_source(..., require_s8_states=True)` + current S9 downstream eligibility | Explicit `package_dir`; refuse `review_run_dir` merge |
| Published S10 reviewed | `assert_not_draft_source` + `validate_s10_run_for_downstream` | Uses `reviewed_behavior_states.parquet`; identity distinct from S8 |
| Working drafts | **Rejected** | No draft/working aliases |

Pinned digests (package / sync / bouts / video / states) are revalidated before final publication. Scientific trees are read-only.

### 14.3 Export scope and frame boundaries

- Explicit inclusive prepared interval `[start, end]` (`boundary_convention=inclusive_inclusive`).
- Explicit bout/span via `SyncMap.bout_seek` (optional packaged prepared-endpoint cross-check).
- Optional prepared-FPS context pad via existing S9 `compute_clip_bounds`.
- Output frame count = inclusive prepared count; encode verifies frame count + fps.

### 14.4 Behavioral HUD (C2-A)

Fixed HUD (not pose-anchored):

| Element | Presentation |
| --- | --- |
| Exclusive `0→1` | Top-left row 0; exact recorded `behavior_label` |
| Exclusive `1→0` | Top-left row 1; independent |
| Contact | Top-right LED `CONTACT:ON/OFF/?` (parallel channel) |
| SBS | Top-right LED `SBS:ON/OFF/?` (parallel; not mutually exclusive with contact) |
| Counter | Bottom `prep=… dense=… t=…s map_t=…s` |

Directional modes: `focal_0_only` / `focal_1_only` / `both` (default).  
Unavailable / unknown / invalid remain distinguishable (row `UNAVAILABLE` / `UNKNOWN`; LED `?`).  
Contact/SBS are never drawn as exclusive ethogram labels.

### 14.5 Timing

- Frame domain: prepared video.
- Authority: S9 `sync_index` / `SyncMap`.
- Plateau policy: **`LAST_IN_PLATEAU`** (offline default).
- Output FPS: prepared-video FPS.
- Each prepared frame’s HUD label is traced to the selected dense sample (manifest notes first/last; counter shows dense idx).

### 14.6 Video encode architecture

Streaming pipeline in `src/overlay/export.py`:

1. Seek/open prepared video at clip start (random access with sequential fallback).
2. Per prepared frame: decode → `collect_evidence` → `OverlayCompositor.render_frame` → write BGR24 bytes to ffmpeg stdin.
3. ffmpeg: `rawvideo bgr24` → `libx264` / `yuv420p` / CFR at source FPS; stderr drained on a thread (avoid pipe deadlock).
4. Staging dir + atomic replace; cooperative cancel between frames; failure cleans staging / unpublished job dir.
5. Behavior-only adapters with optional dense-window parquet filters (no pose tables for behavioral-only QC).

### 14.7 Manifest contract (`overlay_artifact_manifest_v1`)

C2-A pins (in core fields + `extra`):

- Source kind / package / review identities + digests + eligibility recheck stamp
- Prepared-video identity (path, sha256, fps, frame_count)
- Sync mapping identity + plateau policy
- Inclusive prepared frame range + plan (HUD mode, layers, bout ids)
- Output FPS, frame count, codec/crf/pixel_format, video sha256
- Renderer version `overlay_renderer_c2a_v1`
- Unavailable layer union across frames

### 14.8 Files added/updated (C2-A)

```text
src/overlay/source.py
src/overlay/plan.py
src/overlay/export.py
src/overlay/contracts.py          # HudDirectionMode, export constants, C2A layers
src/overlay/style.py              # HUD mode / prepared FPS hooks
src/overlay/layers/behavior.py    # fixed dual-row HUD + independent LEDs
src/overlay/layers/debug.py       # prepared time counter
src/overlay/adapters/behavior.py  # windowed reads; S10 reviewed factories
src/overlay/compositor.py         # render unavailable via HUD drawers
tests/overlay/test_hud_behavior.py
tests/overlay/test_export_qc.py
tests/overlay/test_source_and_windowing.py
scripts/_overlay_c2a_e_session_qc_smoke.py
```

### 14.9 E-session evidence

Automated S8 source on `results_package__20261028T181714` (local review run is draft → not used).  
Short QC clip published under session `results_exports/overlay_qc_behavioral__*`; inspected frames show dual exclusive rows, independent CONTACT/SBS, and prep/dense/time counters. Upstream S8/S9 fingerprints unchanged.

### 14.10 Decisions resolved by C2-A

| ID | Resolution |
| --- | --- |
| **O1** | First artifact = offline bout/span / prepared-interval QC clip |
| **O2** | Dual directional rows; modes focal0 / focal1 / both |
| **O7** | Fixed top-left exclusive HUD (not pose-anchored) for C2-A |

Still deferred: **O3** S4 styling, **O4** S8 `generate_overlay_video` flag wiring, GUI/PI/gallery, full-session exporter, dense-rate export.

### 14.11 C2-B / C3 prerequisites (historical at C2-A close)

1. Optional S4 provenance-styled pose layer on prepared-rate QC (approve **O3**) — **done in §16**.  
2. Decide **O4** (wire vs deprecate unused S8 overlay flag) — still open.  
3. Optional GUI action to launch QC export (still opt-in).  
4. Full-session prepared-rate exporter with stronger cancel / chunk progress (C3).  
5. Live S10 compose path only after U7 reopen if needed.

### 14.12 Stop point (historical)

**C2-A complete as first offline behavioral QC MP4 exporter.** C2-B S4 pose QC follows in §16.

---

## 15. C1 + C2-A final acceptance and Git checkpoint (2026-11-02)

### 15.1 Visual acceptance — E-session C2-A QC

Artifact:

```text
test_cases/E_full_session_integration/results_exports/overlay_qc_behavioral__20261102T152728/
  behavioral_qc.mp4
  overlay_artifact_manifest.json
```

| Check | Result |
| --- | --- |
| Dual exclusive labels legible | **Pass** — sampled frames show `0->1 engaged` and `1->0 approach` with high-contrast fixed HUD rows |
| CONTACT/SBS ON / OFF / unavailable distinguishable | **Pass with note** — this bout clip is CONTACT/SBS **OFF** throughout dense `[14748,14808]`; text `CONTACT:OFF` / `SBS:OFF` + dark LED. ON / `?` / exclusive `UNAVAILABLE` covered by compositor unit tests (`test_hud_behavior.py`) and distinct draw paths (`ON`/`OFF`/`?` text + green/gray/red LEDs) |
| Labels match authoritative S8 rows | **Pass** — dense 14748 / 14778 / 14808: focal0 `engaged`, focal1 `approach`, `pred_contact=0`, `pred_side_by_side=0` |
| Counters match sampled frames | **Pass** — frame0 `prep=14631 dense=14748 t=122.900s`; mid `prep=14661 dense=14778`; last `prep=14690 dense=14808` |
| Plateau policy | **Pass** — manifest `plateau_policy=last_in_plateau`; first/last timing notes pin dense samples |
| HUD concealment | **Pass** — HUD confined to top-left / top-right / bottom bar; animal interaction zone unobstructed |
| Encode vs manifest | **Pass** — ffprobe `nb_frames=60`, `r_frame_rate=14881/125` (=119.048), duration≈0.504s; matches manifest `frame_count=60`, `output_fps=119.048`, `libx264`/`yuv420p` |
| Targeted HUD styling changes | **None** — no rendering defects requiring C2-A code changes |

### 15.2 C1 foundation status

**Accepted.** Shared `src/overlay/` contracts, SyncMap timing (`LAST_IN_PLATEAU` offline default), read-only adapters (S2/S3/S4/S8/S10), compositor core, provenance manifest scaffold, and unit/E-session timing tests remain in place. No science mutation; overlays opt-in.

### 15.3 C2-A behavioral QC export status

**Accepted.** Offline prepared-rate exporter publishes `overlay_qc_behavioral__*` with `behavioral_qc.mp4` + `overlay_artifact_manifest_v1`. Dual exclusive HUD + independent CONTACT/SBS LEDs + debug counter. Naked S9 / PI media paths unchanged. S8 `generate_overlay_video` later **deprecated in C2-B** (O4 closed).

### 15.4 Remaining C2-B prerequisites (resolved by §16)

1. Approve **O3** S4 provenance visual classes — **resolved in C2-B** (third class for `temporal_resample`).  
2. Decide **O4** — wire vs deprecate unused S8 `generate_overlay_video` — **closed in C2-B** (deprecated/unsupported; runtime unchanged).  
3. Optional S4 provenance-styled pose layer on prepared-rate QC — **shipped in C2-B**.  
4. Optional GUI launch for QC export (still opt-in; no GUI in C2-B).  
5. Keep C3 (full-session exporter / live S10) deferred until U7 reopen if needed.

### 15.5 Integration baseline and strategy

| Item | Value |
| --- | --- |
| Overlay feature base | `a6572cc` (local checkout where C1/C2-A was developed) |
| `origin/main` at acceptance | `f91f458` — merge of legacy `legacy_framewise_v1` CSV exporter (PR #1; commits `ee1f0c5` + merge) |
| Delta on main since overlay base | CSV exporter + S4 lineage picker only (no overlay package) |
| Preferred integration | **Clean worktree** from `origin/main`, merge/cherry-pick overlay feature branch; do **not** rebase the active C1/C2-A development checkout in place |
| Preserve on merge | D0 workspace; S10 Gallery A1/A2; PI report exporter; legacy framewise CSV; existing S9 naked media export; all C1/C2-A overlay functionality |
| Expected conflict surface | Low — overlay adds `src/overlay/`, `tests/overlay/`, roadmap docs, and `pyproject.toml` package/tooling entries; CSV work touches results packaging / CSV paths. Watch `pyproject.toml` only |
| Push / merge to main | **Blocked** pending architectural review of this checkpoint |

### 15.6 Stop point

**Local overlay feature branch committed; combined regressions run in integration worktree when prepared.** Do not push or merge until architectural review. No GUI / PI / gallery overlay / S4 provenance / full-session exporter / science changes in this checkpoint.

_End of C1+C2-A final acceptance record (C0 audit + C1 + C2-A retained above)._

---

## 16. Increment C2-B — S4 finalized-pose QC video (2026-11-02)

### 16.1 Operator-visible product

Offline prepared-rate S4 finalized-pose QC MP4 under:

```text
{session}/results_exports/overlay_qc_s4_pose__{timestamp}/
  s4_pose_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

- Opt-in only; explicit S4 run directory required (never auto-picks newest).
- H.264 / yuv420p via the shared C2-A streaming ffmpeg pipe; prepared-video FPS.
- Immutable job directories; digest revalidation before publication.
- Naked S9 / PI / C2-A behavioral exports unchanged.

### 16.2 Provenance visual map (O3 resolved)

Exact S4 codes from `pose_finalization.contracts` / overlay mirrors:

| Code | `provenance_name` | Visual class | Draw style |
| --- | --- | --- | --- |
| 0 | `missing` | `unavailable` | **Not drawn** |
| 1 | `original` | `observed` | Filled disc (mouse identity color) |
| 2 | `identity_corrected` | `observed` | Filled disc (mouse identity color) |
| 3 | `temporal_resample` | `temporal_resample` | Hollow ring (third class; not conflated with gap-fill) |
| 4 | `linear_gap_fill` | `gap_fill` | Diamond outline |
| 5 | `kalman_gap_fill` | `gap_fill` | Diamond outline |
| 6 | `protected_manual_blank` | `unavailable` | **Not drawn** |

- Per-keypoint styling; mouse identity via track color + `M{track}` label.
- Original provenance codes retained in observation payload + manifest.
- Unknown codes raise; no fabricated confidence / categories.
- Skeleton edges only between drawable endpoints; topology from S2 `pose.slp` when `sleap_io` can read it — otherwise nodes-only (never invent edges).

### 16.3 Source eligibility and timing

| Pin | Authority |
| --- | --- |
| S4 run identity | Explicit `s4_run_dir`; `validate_s4_run_for_downstream` |
| S4 artifact digest | `final_pose.parquet` sha256 |
| S1 prepared video | `{session}/preprocess/prepared_video.mp4` + sha256 + probe |
| Timing source | S1 `prepared_sync.npz` → NN dense map validated against S4 dense `sample_idx`/`time_sec` → `SyncMap` |
| S3 provenance | Recorded `source_s3` + live acceptance / tracked_pose digest |

- Plateau policy: **`LAST_IN_PLATEAU`**.
- Frame domain: prepared video FPS (no artificial 120 fps duplication).
- Reject missing / incompatible / digest-mismatched sources.

### 16.4 Export architecture

Reuses C2-A streaming path in `src/overlay/export.py` (`export_s4_pose_qc_clip`):

1. Open/pin S4 source (`source_s4.py`).
2. Plan inclusive prepared interval (`plan_s4_prepared_interval`).
3. Windowed S4 adapter + debug counter → compositor → ffmpeg raw BGR24 pipe.
4. Staging + atomic publish; cancel cleans staging / unpublished job dir.
5. Manifest `overlay_artifact_manifest_v1` with renderer `overlay_renderer_c2b_v1`.

### 16.5 Files added/updated (C2-B)

```text
src/overlay/contracts.py              # O3 map, C2-B constants, default_s4_pose_qc_layers
src/overlay/adapters/pose.py          # visual_class payload; dense window; factory
src/overlay/layers/pose.py            # per-keypoint S4 provenance drawing + legend
src/overlay/source_s4.py              # explicit S4 QC source resolve + integrity
src/overlay/plan.py                   # S4PoseQcExportPlan / plan_s4_prepared_interval
src/overlay/export.py                 # export_s4_pose_qc_clip (+ sidecars)
src/overlay/provenance.py             # optional renderer_version
src/overlay/__init__.py / adapters/__init__.py
tests/overlay/test_s4_provenance_style.py
tests/overlay/test_export_s4_pose_qc.py
scripts/_overlay_c2b_e_session_s4_pose_qc_smoke.py
docs/roadmap/post_v011_overlay_architecture.md
```

### 16.6 Laboratory evidence (E-session)

Explicit eligible run: `pose_finalization__20261025T173206`.

Published artifact:

```text
test_cases/E_full_session_integration/results_exports/overlay_qc_s4_pose__20261102T185333/
  s4_pose_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

Clip: prepared inclusive `[971, 1000]` (30 frames @ 119.048 fps).  
Inspected frames (evidence under `tmp_overlay_c2b_e_session_s4_pose_qc/`):

| Clip frame | prep | dense (LAST_IN_PLATEAU) | Recorded S4 codes | Visual check |
| --- | --- | --- | --- | --- |
| 0 | 971 | 979 | `missing`, `temporal_resample` | Hollow rings for drawable points; M1 `headstage` missing not drawn |
| 15 | 986 | 994 | `missing`, `temporal_resample`, `kalman_gap_fill` | Rings + diamond (M0 nose gap-fill) |
| 29 | 1000 | 1008 | `missing`, `identity_corrected`, `kalman_gap_fill` | Filled observed discs + diamond nose |

**Absent from this clip (not visually validated here):** `original` (1), `linear_gap_fill` (4), `protected_manual_blank` (6).  
Unit tests cover those codes’ mapping and non-drawing rules.

**Skeleton note:** this environment lacks `sleap_io`, so edges resolved to nodes-only (`skeleton_source=unresolved:…/pose.slp`). Honest; no invented topology. Upstream S4/S1 digests unchanged.

### 16.7 Decisions resolved by C2-B

| ID | Resolution |
| --- | --- |
| **O3** | Three visual classes: observed / temporal_resample / gap_fill; unavailable not drawn |
| **O4** | S8 `generate_overlay_video` **deprecated and unsupported** — documented in config / YAML / overlay package docstring; runtime parse/default unchanged; no producer wired |

Still deferred after C2-B: PI/gallery overlay, full-session exporter, dense-rate export, S5 contours.  
**GUI overlay launch:** shipped in §17 (C2-C).

### 16.8 Topology (optional `sleap_io`)

- Skeleton edges load from S2 `pose.slp` via `tracking_correction.skeleton.resolve_review_skeleton_edges` only when readable.
- **`sleap_io` remains optional** for overlay QC and is **not** a hard GUI-launch dependency for C2-B.
- When topology is unavailable/unreadable: **nodes-only** mode with explicit disclosure in processing log, manifest (`skeleton_mode`, `skeleton_nodes_only`, `skeleton_disclosure`), and source provenance — **never invent edges**.
- Focused tests: `tests/overlay/test_s4_skeleton_topology.py` (injected real topology edge drawing + nodes-only disclosure).

### 16.9 Final acceptance evidence (2026-11-02)

#### Provenance mapping (accepted)

| Code | Name | Visual |
| --- | --- | --- |
| 0 | missing | not drawn |
| 1 | original | filled disc (observed) |
| 2 | identity_corrected | filled disc (observed) |
| 3 | temporal_resample | hollow ring |
| 4 | linear_gap_fill | diamond |
| 5 | kalman_gap_fill | diamond |
| 6 | protected_manual_blank | not drawn |

#### Performance breakdown (E-session short clip prep `[971,1000]`, 30 frames)

Instrumenting `export_s4_pose_qc_clip` + source open on 2026-11-02 (job `overlay_qc_s4_pose__20261102T191230`):

| Phase | Seconds |
| --- | --- |
| Source resolve/open (+ hashing / sync build) | 0.510 |
| Source validation and hashing (export recheck) | 0.016 |
| S4 pose loading | 0.068 |
| Timing resolution (per-frame evidence) | 0.034 |
| Frame composition | 0.096 |
| Prepared video decode | 0.032 |
| FFmpeg encoding (stdin write + close) | 0.147 |
| Sidecar publication | 0.014 |

Dominant cost for this short clip is **source open** (S4 eligibility + sync index + digests), not per-frame composition/encode. No localized performance defect requiring a patch; no broad optimization in C2-B closure.

#### Real-session visual evidence

Prior lab clip under `overlay_qc_s4_pose__20261102T185333` + frames in `tmp_overlay_c2b_e_session_s4_pose_qc/`:

- Filled discs = observed; hollow rings = temporal_resample; diamonds = gap_fill.
- Missing/protected not drawn; per-keypoint classes coexist on one mouse; M0/M1 distinguishable; legend readable.
- Nodes-only on this host (`sleap_io` absent); no misleading edges.
- **Validated only via unit tests (absent from short clip):** `original` (1), `linear_gap_fill` (4), `protected_manual_blank` (6). Topology edge drawing validated via injected-topology fixture (not this host’s missing `sleap_io`).

#### Acceptance status

**C2-B scientifically and architecturally accepted.** Offline prepared-rate S4 pose QC exporter closed. Upstream S4/S1 digests unchanged.

#### Deferred

GUI overlay integration, full-session export, PI/gallery overlay, S5 overlays, dense-rate export.

### 16.10 Stop point

**C2-B final acceptance complete.** Commit C2-B code/tests/docs and push to `origin/main` (no release tag). No GUI / PI / gallery / S5 / full-session / scientific algorithm changes.

_End of C2-B record._

---

## 17. Increment C2-C — QC Overlay Export GUI (2026-11-02)

### 17.1 Operator-visible workflow

Shared modal **QC Overlay Video Export** dialog (PI-report style), opened from:

| Page | Button | Prefill |
| --- | --- | --- |
| S4 Pose Finalization | `Export QC Overlay Video…` | `export_family=s4_pose`, selected eligible S4 run |
| S8 Behavioral Arbitration | `Export QC Overlay Video…` | `export_family=behavioral` (locked); eligible S9 package(s) that pin the selected S8 run (see §18 C3-0) |
| S9 Results Packaging | `Export QC Overlay Video…` (separate from CSV/media) | `export_family=behavioral`, selected eligible S9 package |
| S10 Behavioral Review | `Open QC Overlay Export…` | behavioral; package + optional published review; optional gallery bout |

Dialog capabilities:

- Explicit export family (S4 pose vs behavioral); opener may lock family.
- Explicit source selection (never auto-picks newest science run).
- Prepared-frame inclusive interval **or** packaged exclusive bout span (behavioral).
- Behavioral HUD mode (`both` default), optional contact/SBS LEDs, optional frame/time counter.
- S4 optional skeleton draw + counter.
- Pre/post prepared-FPS context padding.
- Progress + cancel; job discovery table; open folder / open MP4.
- Immutable job dirs under `{session}/results_exports/overlay_qc_*__{timestamp}/`.

Naked S9 CSV/media and S10 PI-report workflows are unchanged (separate runners/kinds/tables).

### 17.2 Supported sources and settings

| Family | Eligible sources | Rejected |
| --- | --- | --- |
| S4 pose QC | Explicit completed/eligible S4 run via `resolve_overlay_s4_qc_source` | Missing/ineligible S4; digest mismatch; auto-newest |
| Behavioral QC | Automated S8 via eligible S9 package; published S10 reviewed | Working drafts; silent automated+review merge; auto-newest |

Settings snapshotted into `OverlayQcRequestIdentity` at launch (immutable for the worker). Backend planners remain `plan_prepared_interval` / `plan_bout_span` / `plan_s4_prepared_interval`. Encoding reuses C2-A/C2-B streaming ffmpeg only (no second encoder).

### 17.3 Background execution and navigation safety

- Task kind: `GuiTaskKind.OVERLAY_QC_EXPORT`.
- Dialog owns a dedicated `GuiTaskRunner` (does not share S9 packaging/CSV/media or S10 PI runners).
- Resolve / plan / hash / encode run on the worker thread; progress bridged from exporter `dict` → `OperationProgress`.
- Cooperative cancel via existing exporter `cancellation_requested` → `OverlayExportCancelled`.
- Non-overlapping: controller `task_running` + runner `TaskAlreadyRunningError`.
- Session retarget blocked while export runs; stale `request_id` / session-root mismatches reject success/failure UI updates.
- Incomplete exports are not treated as completed (discovery requires video + manifest).

### 17.4 Discovery

`overlay.discovery.summarize_overlay_qc_project` scans only:

- `overlay_qc_behavioral__*`
- `overlay_qc_s4_pose__*`

Ignores `media_export__*`, `csv_export__*`, `pi_report__*`. Prior jobs are never overwritten.

### 17.5 Files added/updated (C2-C)

```text
src/overlay/discovery.py
src/overlay/__init__.py
src/ui/tasks.py                              # OVERLAY_QC_EXPORT
src/ui/controllers/overlay_qc_export_support.py
src/ui/controllers/overlay_qc_export_controller.py
src/ui/dialogs/__init__.py
src/ui/dialogs/overlay_qc_export_dialog.py
src/ui/pages/pose_finalization_page.py       # entry button
src/ui/pages/results_packaging_page.py       # entry button
src/ui/pages/behavioral_review_page.py       # entry button
tests/overlay/test_discovery.py
tests/ui/test_overlay_qc_export_gui.py
scripts/_overlay_c2c_e_session_gui_smoke.py
docs/roadmap/post_v011_overlay_architecture.md
```

### 17.6 Known limitations

- No full-session export.
- No S5 contours; no live S10/PI/gallery overlay burn-in. (S2/S3 offline QC exporters shipped in C3-1/C3-2.)
- Bout combo lists exclusive packaged bouts only (first 250 by dense start); dense endpoints remain editable; S10 preferred bouts outside the window are injected into the combo.
- Published S10 reviewed export requires an explicit published review (working drafts rejected). Live E-session review tree remains draft-only; published S10 validation uses the isolated PI B2 session copy.
- Topology / `sleap_io` remains optional (nodes-only disclosure unchanged from C2-B).
- Dialog blocks close while an export is running (cancel first).
- GUI controller always publishes under the **bound session** `results_exports/` (never the package manifest's embedded `session_root`) so isolated package copies cannot leak QC jobs into the lab E tree (D0).

### 17.7 Lab acceptance (E-session provisional, 2026-11-02)

Controller path identical to the dialog worker (`OverlayQcExportController.execute_export_launch`):

| Check | Result |
| --- | --- |
| S4 pose QC | **Pass** — `overlay_qc_s4_pose__20261102T193624` (30 frames, H.264, 119.048 fps) |
| S8 automated behavioral QC | **Pass** — `overlay_qc_behavioral__20261102T193625` from `results_package__20261028T181714` (30 frames) |
| Source selection / draft gate | **Pass** — draft reviewed source rejected |
| Discovery | **Pass** — prior + new jobs listed; media/PI prefixes ignored |
| Upstream digests | **Pass** — S4 `final_pose` + S9 `package_manifest` unchanged |
| Published S10 | Deferred on live E-session (draft only) → covered in §17.9 via isolated published review |

Evidence: `tmp_overlay_c2c_e_session_gui/C2C_GUI_SMOKE_REPORT.json`.

### 17.8 Final acceptance (2026-11-02) — published S10 + visible GUI + bout coverage

**Bound:** no science / PI / gallery / full-session / S5 / new overlay layers. No commit/push in this pass.

#### Published S10 (isolated session)

Source: `tmp_pi_report_b2_final_acceptance/isolated_session` with published `behavioral_review__20261101T201442` + in-session package copy `results_package__20261028T181714` (D0: junctions rejected; package copied so resolve stays in-session). Prepared video remains a read-only lab E reference.

Shared dialog/controller path (`OverlayQcExportController` ≡ dialog worker):

| Check | Result |
| --- | --- |
| Select published reviewed source | **Pass** — dialog prefills `s10_reviewed` + review run |
| Short behavioral QC clip | **Pass** — `overlay_qc_behavioral__20261102T195200` under **isolated** `results_exports/` (30 frames) |
| HUD / states identity | **Pass** — `run_meta.source_kind=s10_reviewed`; manifest `source_runs` pin `reviewed_behavior_states.parquet` + published review run id |
| Draft rejection | **Pass** — working draft `behavioral_review__20261101T203010` rejected |
| Source package / reviewed artifacts unchanged | **Pass** — package + reviewed digests identical before/after; lab E package digests unchanged |
| D0 containment | **Pass** after fix — exports publish under bound session, not package manifest `session_root` |

Evidence: `tmp_overlay_c2c_final_acceptance/PUBLISHED_S10_REPORT.json` (+ extracted HUD frames).

#### Visible Windows GUI

Qt platform `windows` (not offscreen). Scripted interactive dialog exercise with screenshots:

| Check | Result |
| --- | --- |
| S4 entry → pose export mode | **Pass** |
| S9 entry → automated behavioral mode | **Pass** |
| S10 entry → reviewed source/bout defaults | **Pass** |
| Invalid selection feedback | **Pass** (missing S4 / missing published review) |
| Export without GUI freeze + progress events | **Pass** |
| Close-while-running blocked | **Pass** |
| Open folder / open MP4 hooks | **Pass** |
| Discovery table + reopen dialog | **Pass** |
| Session retarget blocked while running | **Pass** |

Evidence: `tmp_overlay_c2c_final_acceptance/VISIBLE_GUI_REPORT.json` + `gui_*.png`.

#### Bout-selection coverage

| Check | Result |
| --- | --- |
| 250-item limit communicated | **Pass** — dialog `bout_limit_label` |
| Excluded (rejected) bout via prepared interval | **Pass** — package bout `...bout_00000009` → `overlay_qc_behavioral__20261102T195201` |
| S10 preferred bout outside first 250 | **Pass** — injected into combo with correct dense bounds |
| Identity / focal / frame bounds | **Pass** — packaged bout id + dense/prepared endpoints preserved |

Targeted UX fix shipped for the outside-250 / preferred-bout blocker (inject + label). S10 opener also forwards selected bout dense/prepared endpoints.

#### Navigation / cancel / teardown

Covered by GUI unit tests + visible acceptance: stale callback rejection, cancel outcome rows, session retarget block, dialog reopen, close-while-running guard, no source/artifact writes on science trees. No new Windows Qt lifecycle regression observed in this pass (no abnormal Qt process exits recorded in the targeted suites).

#### Regression / static

| Suite | Result |
| --- | --- |
| `tests/overlay` + packaging/pose/overlay GUI + tasks | **121 passed** |
| S4 pose QC + behavioral QC export + discovery + arbitration GUI | **31 passed** |
| `tests/ui/test_overlay_qc_export_gui.py` | **12 passed** (includes outside-250 + output_root pin) |
| Broader S10 `tests/behavioral_review` sweep | **276 passed, 1 failed** — `test_e_session_controlled_edit_publish_reopen` (pre-existing D0 draft-under-`review_root` vs lab SESSION guard; **not** C2-C overlay code) |
| Ruff (C2-C touched files) | **Pass** |
| Mypy (`overlay_qc_export_controller`, support, `discovery`) | **Pass**; dialog module only reports pre-existing PySide6 `import-untyped` stubs |

### 17.9 Required fixes found during final acceptance

1. **D0 output containment (blocker):** controller now passes `output_root = {bound_session}/results_exports` into exporters so isolated package copies whose manifests still embed lab E `session_root` cannot leak QC jobs into the lab tree.
2. **Bout combo UX (blocker):** communicate the 250 exclusive-bout cap; inject/prefill preferred bouts outside that window (by id or dense interval); S10 opener forwards selected bout endpoints.

Probe artifact note: one pre-fix acceptance attempt briefly wrote two overlay jobs under lab E `results_exports/` before the containment fix; science digests were unchanged. Subsequent acceptance exports are under the isolated session only.

### 17.10 Stop point

**C2-C final acceptance complete for architectural review.** Recommendation: **accept C2-C for commit** after operator architectural review. No commit/push in this pass unless operator requests. No science / PI / gallery / full-session / S5 changes.

Evidence root: `tmp_overlay_c2c_final_acceptance/`.

_End of C2-C record._

## 18. Increment C3-0 — S8 Behavioral QC Export Accessibility (2026-11-04)

### 18.1 Scope

Make the existing shared **QC Overlay Video Export** dialog reachable from the S8 Behavioral Arbitration page. **No** new exporter, renderer, scientific algorithm, source schema, or overlay layer. **No** automatic overlay generation when S8 arbitration completes. Naked S9 CSV/media defaults and S10 PI reports are untouched.

### 18.2 Operator-visible behavior

| Item | Behavior |
| --- | --- |
| Button | `Export QC Overlay Video…` on the S8 run row (beside Open S8 run folder) |
| Family | `export_family=behavioral`, `lock_family=True` |
| Source lock | `lock_source_kind=True`, `locked_source_kind=s8_automated` — S8 entry does **not** offer published S10 as an alternative; shared dialog/controller only (no separate exporter). |
| Source contract | Automated S8 evidence only via a **currently eligible S9 package** that pins the selected S8 run (`source_s8_run_id`). Never a raw S8 directory. |
| Prefill | Prefill package only when exactly one eligible S9 package pins the selected S8. Otherwise require explicit eligible package selection. |
| Unrelated sources | Do not list S9 packages that pin a different S8; do not prefill reviewed S10 (`review_*` / `review_options` empty). |
| No eligible S9 | Block with an explicit message that packaging in S9 is required — do not bypass eligibility. |
| HUD defaults | Both focal directions; CONTACT and SBS LEDs optional and on by default (shared dialog defaults). |
| S9/S10 entries | Unchanged: source-kind selection remains available from Results Packaging and Behavioral Review openers. |

Helper: `plan_s8_behavioral_overlay_launch` in `overlay_qc_export_support.py`. Page opener discovers via `summarize_results_package_project(...).eligible_runs`.

### 18.3 Safety preserved from C2-C

Existing eligibility, explicit source identity, pinned digests, bound-session `results_exports/` root, background execution/cancellation, workspace retarget protection, immutable export publication, draft rejection, and S8-automated vs S10-reviewed separation remain enforced by the shared controller/exporters.

### 18.4 Files touched (C3-0)

```text
src/ui/controllers/overlay_qc_export_support.py   # plan_s8_behavioral_overlay_launch + lock_source_kind
src/ui/dialogs/overlay_qc_export_dialog.py        # honor lock_source_kind (hide S10 option)
src/ui/pages/behavioral_arbitration_page.py       # entry button + opener
tests/ui/test_overlay_qc_export_gui.py            # S8 launch / source lock / S9·S10 retain
docs/roadmap/post_v011_overlay_architecture.md    # this section
```

### 18.5 Acceptance evidence (targeted)

| Check | Expected |
| --- | --- |
| S8 button visible + launches shared dialog | Pass (GUI unit) |
| Behavioral family locked; session bound | Pass |
| Unambiguous S9 prefill / ambiguous explicit selection | Pass |
| No eligible S9 → clear dependency message; dialog not opened | Pass |
| S8 entry cannot switch to reviewed S10 (`lock_source_kind`) | Pass |
| S8 package filtering by selected S8 remains correct | Pass |
| S9/S10 retain source-selection behavior | Pass |
| HUD both + CONTACT/SBS defaults propagate | Pass |
| S4/S9/S10 entry buttons unchanged | Pass |
| Overlay / S8 / S9 / workspace GUI regressions + Ruff + targeted Mypy | Recorded in closure report |

### 18.6 Closure

**C3-0 accepted** after architectural review + bounded GUI source-lock correction. Commit/push authorized on this closure pass. Stop before C3-1. Deferred C3 live/full-session work remains out of scope.

_End of C3-0 record._

---

## 19. Increment C3-1 — S2 Pose Inference QC Video (2026-11-04)

### 19.1 Operator-visible product

Offline prepared-rate S2 inferred-pose QC MP4 under:

```text
{session}/results_exports/overlay_qc_s2_pose__{timestamp}/
  s2_pose_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

- Opt-in only; explicit eligible S2 run required (never auto-picks newest).
- H.264 / yuv420p via the shared C2-A/C2-B streaming ffmpeg pipe; prepared-video FPS.
- Immutable job directories; digest revalidation before publication.
- Existing S2 `overlay.mp4` generation path **unchanged**.
- Naked S9 / PI / S4 / behavioral QC exports unchanged.

### 19.2 Source eligibility and provenance contract

| Pin | Authority |
| --- | --- |
| S2 eligibility | Explicit `s2_run_dir`; `tracking_correction.runner.validate_s2_handoff` (completed S2 + authoritative `pose.parquet` + resolvable S1 prepared video/timing). **Not** S3 acceptance. |
| S2 artifact digest | `pose.parquet` sha256 |
| S1 prepared video | Resolved from S2 `pose_meta.json` input paths (or `pose_inference/` layout fallback) + sha256 + probe |
| S1 timing | `prepared_sync.npz` → **prepared-identity SyncMap** (1:1 prepared↔sample at prepared FPS). **No** S4 dense 120 Hz NN duplication. |
| Skeleton | S2 `pose.slp` via `resolve_review_skeleton_edges` only when readable |

Reject incomplete / dry-run / mismatched frame-domain / digest-mismatched sources. Never silently select the newest S2 run.

### 19.3 Pose, skeleton, and confidence rendering

| Concern | Design |
| --- | --- |
| Nodes | All finite XY keypoints from S2 `pose.parquet` (prepared-frame domain) |
| Identity | Provisional `P{id}` labels + track colors; legend states **S2 predictions, not S3-accepted** |
| Track labels | Parse integer / `track_N` SLEAP labels; fall back to `instance_index` |
| Skeleton | Recorded SLEAP topology only; otherwise **nodes-only** with manifest/log disclosure (never invent edges) |
| Confidence (optional) | Mapping id `s2_score_fill_intensity_v1`: fill intensity scales with SLEAP `score` assuming `[0,1]`; values outside clamped **for drawing only**; originals preserved. Missing/NaN → hollow ring (distinct from numerical zero). No confidence thresholding / no node dropping for viz. Identity-colored outline remains legible. |

Adapter / layer reuse: `make_s2_inferred_pose_adapter` + shared `PoseLayer` S2 path; compositor / ffmpeg encode unchanged.

### 19.4 Timing and export behavior

- Frame domain: prepared video indices + prepared FPS.
- Inclusive prepared interval `[start, end]`; optional prepared-FPS context padding.
- Cooperative cancel; staging + atomic publish; integrity recheck before publish.
- Renderer version: `overlay_renderer_c3_1_v1`.
- Debug counter uses prepared-identity sample idx (equal to prepared frame); notes disclose “not dense 120 Hz”.

### 19.5 GUI integration

| Item | Behavior |
| --- | --- |
| Shared dialog family | `export_family=s2_pose` |
| S2 Pose Inference page | `Export QC Overlay Video…` beside Open overlay |
| Prefill | Prefill selected eligible S2 when unambiguous; otherwise explicit selection among eligible options |
| Options | Skeleton draw; optional confidence styling; frame/time counter; prepared interval + pad |
| Safety | Bound-session `results_exports/` (D0); background runner; cancel; session retarget block |
| Other families | S3/S4/S8/S9/S10 entry behaviors unchanged |

### 19.6 Laboratory acceptance evidence (E-session)

Explicit eligible run: `bottomup__20260709T195039`.

Published artifact:

```text
test_cases/E_full_session_integration/results_exports/overlay_qc_s2_pose__20261104T144105/
  s2_pose_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

Clip: prepared inclusive `[971, 1000]` (30 frames @ 119.048 fps).  
Evidence: `tmp_overlay_c31_e_session_s2_pose_qc/` (+ `C31_S2_POSE_QC_SMOKE_REPORT.json`, sampled frames).

| Check | Result |
| --- | --- |
| Keypoints on prepared video | **Pass** — nodes drawn in prepared coordinates |
| Provisional identities distinguishable | **Pass** — `P0`/`P1` in clip; manifest pins provisional semantics |
| Skeleton | **Nodes-only** on this host (`sleap_io` absent); disclosure recorded; no invented edges. Topology edge drawing covered by fixtures |
| Confidence styling | **Pass** — enabled; scores in clip within `[0.179, 0.940]`; mapping disclosed. Missing-score / numerical-zero / >1 clamp covered by unit fixtures (not present in this short clip) |
| Frame count / FPS vs manifest | **Pass** — 30 frames, 119.048 fps, libx264/yuv420p |
| Upstream digests | **Pass** — `pose.parquet` sha256 unchanged; existing `overlay.mp4` still present |

### 19.7 Confidence visual acceptance (`s2_score_fill_intensity_v1`)

Final closure pixel fixture (score **0**, low+, mid, **1**, **missing**, out-of-range **1.5**) confirmed:

| Criterion | Result |
| --- | --- |
| Numeric confidence 0 remains drawable | **Pass** — filled disc at minimum intensity + identity outline/halo |
| Missing confidence visually distinct | **Pass** — hollow ring; center stays background (≠ numerical zero fill) |
| Low confidence does not drop markers | **Pass** — no score filtering; low+ still filled + outlined |
| Scores outside `[0,1]` display-clamped only | **Pass** — 1.5 renders identical to 1.0; payload keeps 1.5 |
| Original confidence values unchanged | **Pass** — adapter/payload preserve originals including 0 and >1 |
| Legend + manifest describe mapping | **Pass** — on-frame mapping id + summary; manifest `confidence` disclosure |
| Visual output deterministic | **Pass** — identical re-renders |

Drawing/legend mapping required no visual correction on this closure pass.
Minimum closure fixes only: pixel-fixture regression; S2 legend-string adapter test update;
`state.summary` eligibility lookup; `np.isfinite` typing in drawable-score parsing.
No confidence-computation or scientific-filtering changes.

### 19.8 Semantics pins (accepted)

- **Provisional identity:** `P{id}` labels + track colors; legend states S2 predictions, **not** S3-accepted `M{id}`.
- **Nodes-only fallback:** when `pose.slp` topology is unreadable / `sleap_io` absent → draw nodes only; never invent edges; disclose in legend/manifest/log.
- **Performance limitation:** export loads full `pose.parquet` then windows the adapter — can be heavy for very long sessions.

### 19.9 Files touched (C3-1)

```text
src/overlay/source_s2.py
src/overlay/export_s2.py
src/overlay/contracts.py
src/overlay/plan.py
src/overlay/adapters/pose.py
src/overlay/layers/pose.py
src/overlay/discovery.py
src/overlay/__init__.py
src/ui/controllers/overlay_qc_export_support.py
src/ui/controllers/overlay_qc_export_controller.py
src/ui/dialogs/overlay_qc_export_dialog.py
src/ui/pages/pose_inference_page.py
tests/overlay/test_export_s2_pose_qc.py
tests/overlay/test_s2_confidence_identity_skeleton.py
tests/ui/test_overlay_qc_export_gui.py
scripts/_overlay_c31_e_session_s2_pose_qc_smoke.py
docs/roadmap/post_v011_overlay_architecture.md
```

### 19.10 Remaining limitations / risks

- Full-session / dense-rate export still deferred (C3 live/full-session out of scope).
- Topology requires optional `sleap_io`; otherwise honest nodes-only.
- Loading full `pose.parquet` then windowing can be heavy for very long sessions.
- Existing S2 `overlay.mp4` generation path remains the S2 runner product (unchanged by this QC family).

### 19.11 Closure

**C3-1 accepted** after architectural review + final confidence-visualization verification. Commit/push authorized on this closure pass. Stop before C3-2. No S3/S5/S9-media/S10 overlay-media / science or artifact-schema changes.

_End of C3-1 record._

---

## 20. Increment C3-2 — S3 Tracking Correction QC Video (2026-11-04)

### 20.1 Operator-visible product

Offline prepared-rate S3 accepted tracked-pose QC MP4 under:

```text
{session}/results_exports/overlay_qc_s3_pose__{timestamp}/
  s3_pose_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

- Opt-in only; explicit **accepted** S3 run required (never auto-picks newest).
- H.264 / yuv420p via the shared C2-A/C2-B streaming ffmpeg pipe; prepared-video FPS.
- Immutable job directories; digest revalidation before publication.
- S3 working-pose visualization and Accept Tracking workflow **unchanged**.
- Naked S9 / PI / S2 / S4 / behavioral QC exports unchanged.

### 20.2 Accepted-run source contract

| Pin | Authority |
| --- | --- |
| S3 eligibility | Explicit `s3_run_dir`; `pose_finalization.accepted_s3.validate_accepted_s3_run` (`acceptance_state == accepted` + recorded `tracked_pose.parquet` digest match). **Not** `complete_reviewable` / working-only. |
| S3 artifact digest | `tracked_pose.parquet` sha256 |
| S1 prepared video/timing | Prefer `{session}/preprocess/` layout; else recorded S3 `run_meta.json` input paths. Prepared-identity SyncMap (1:1) — **no** dense 120 Hz duplication. |
| S2 lineage | Recorded `input.s2_run_dir` / `pose_parquet` when present (for skeleton topology via `pose.slp`). Missing topology → nodes-only disclosure. |
| Journals | Authoritative `machine_corrections.json` / `manual_corrections.json` only; digests pinned when present |

Reject unaccepted / superseded / working-only / mismatched lineage / digest-mismatched sources. Never silently select the newest S3 run.

### 20.3 Journal schema findings (binding)

**`machine_corrections.json`** — list of automatic events:

| Field | Role |
| --- | --- |
| `frame` | Prepared-frame index (single-frame event) |
| `type` | Action type (`node_overlap`, `identity_swap`, `headstage_reassigned`, …) |
| `track` | Optional track attribution |
| `node` | Optional structured node (rare); **never** scraped from free-text `action` |
| `action` | Optional human-readable text (display/debug only) |

**`manual_corrections.json`** — list of inclusive edit records:

| Field | Role |
| --- | --- |
| `action` | `swap_identities` / `swap_node` / `blank_node` |
| `start_frame` / `end_frame` | Inclusive prepared-frame interval |
| `track` | Optional dedicated track (e.g. `blank_node`) |
| `tracks` | Involved tracks; multi-track alone does **not** force per-track attribution |
| `node` | Present for `swap_node` / `blank_node` |
| `source` | `"user"` |

**Attribution rules:** `track_node` only when structured track+node recorded; `track` when only structured track; otherwise **unlocalized**. Frame-wide parquet `was_track_swapped` is **never** treated as a per-node event and never invents suspicious-frame markers. Absent/malformed/empty journals are disclosed; events are never fabricated.

### 20.3a Event-to-frame semantics (binding)

| Origin | Frame coverage | Renderer rule |
| --- | --- | --- |
| Machine | Exact recorded `frame` only (`start_frame == end_frame == frame`) | Draw on that prepared frame only; **never** extend into neighbors |
| Manual | Inclusive `[start_frame, end_frame]` | Draw on every prepared frame in the closed interval; nothing on `start-1` or `end+1` |
| Event identity | One journal record → one drawable event | **No silent merge/dedup** of identical `(type, track, node, action)` rows; **no fabricated** events |
| Attribution locality | Structured fields only | Unlocalized events → top-center pill; never imply a localized correction without structured track/node |
| Missing attributed geometry | `track_node` whose node is absent from accepted pose (e.g. blanked NaNs) | Unlocalized pill — **never** fall back to another node on the same track |

`machine_event_draws_in_clip` / `manual_event_draws_in_clip` count **per-frame event applications** (one journal event spanning N frames contributes N draws).

### 20.4 Pose / event visualization

| Concern | Design |
| --- | --- |
| Pose | Accepted `tracked_pose.parquet` coordinates via `make_s3_tracked_pose_adapter` + shared `PoseLayer` S3 path |
| Identity | `M{id}` labels + track colors; legend **“S3 accepted tracked pose”** |
| Skeleton | Recorded S2 `pose.slp` topology when readable; otherwise nodes-only with disclosure |
| AUTO events | Orange (`BGR 0,140,255`) indicators; action type shown; localized diamond when track/node attribution + accepted-pose geometry available; else top-center unlocalized pill |
| MAN events | Teal (`BGR 200,180,40`) indicators; same attribution rules over inclusive intervals |
| Placement | Uses accepted-pose geometry for display placement only — never geometry invented from correction metadata |
| Overlap | Multiple events on one frame remain separate drawables; unlocalized pills stack vertically; localized diamonds use attributed anchors. Same-anchor journal duplicates may visually coincide but are not merged |

### 20.5 Timing / export

- Frame domain: prepared video indices + prepared FPS.
- Inclusive prepared interval `[start, end]`; optional prepared-FPS context padding.
- Cooperative cancel; staging + atomic publish; integrity recheck before publish.
- Renderer version: `overlay_renderer_c3_2_v1`.
- Debug counter uses prepared-identity sample idx (equal to prepared frame).

### 20.6 GUI

| Item | Behavior |
| --- | --- |
| Shared dialog family | `export_family=s3_pose` |
| S3 Tracking Review page | `Export QC Overlay Video…` beside Accept Tracking |
| Prefill | Prefill selected accepted S3 when eligible; otherwise explicit selection among accepted options |
| Options | Skeleton; optional machine indicators; optional manual indicators; frame/time counter; prepared interval + pad |
| Unaccepted | Export blocked with actionable explanation (Accept Tracking / select accepted run) |
| Safety | Bound-session `results_exports/` (D0); background runner; cancel; session retarget block |

### 20.7 Laboratory evidence (E-session)

Explicit accepted run: `current_lab_two_mouse_headstage_v1__20260919T181948`.

Published artifact:

```text
test_cases/E_full_session_integration/results_exports/overlay_qc_s3_pose__20261104T153251/
  s3_pose_qc.mp4
  overlay_artifact_manifest.json
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

Clip: prepared inclusive `[971, 1000]` (30 frames @ 119.048 fps).  
Evidence: `tmp_overlay_c32_e_session_s3_pose_qc/` (+ `C32_S3_POSE_QC_SMOKE_REPORT.json`, sampled frames).

| Check | Result |
| --- | --- |
| M0/M1 identities on accepted pose | **Pass** — orange/blue nodes + `M0`/`M1` labels |
| AUTO indicators match journal | **Pass** — see §20.7a event-count interpretation |
| MAN indicators | **Empty on this run** (`manual_corrections.json` = `[]`); disclosed on-frame + manifest. Manual inclusive-interval pixels validated by controlled fixture (§20.7b) |
| Frame counters / FPS | **Pass** — 30 frames, 119.048 fps, libx264/yuv420p; counters show prep=dense prepared-identity |
| Nodes-only disclosure | **Pass** — `sleap_io` absent on host; no invented edges |
| Upstream digests | **Pass** — S1 prepared video/sync + S2 pose + S3 `tracked_pose.parquet` + journals unchanged |

### 20.7a Real-session machine event-count interpretation (90 draws)

Published clip `overlay_qc_s3_pose__20261104T153251` reports `machine_event_draws_in_clip = 90` across 30 exported prepared frames `[971, 1000]`.

| Quantity | Value | Meaning |
| --- | --- | --- |
| Unique machine journal records in interval | **90** | Exact rows in `machine_corrections.json` with `frame ∈ [971, 1000]` |
| Distinct prepared frames containing corrections | **30** | Every exported frame has journaled events |
| Events per frame | **3** | `1× identity_swap` (unlocalized) + `2× headstage_reassigned` (track=1) |
| Rendered event indicators | **90 draws** | One draw per journal record per covered frame |

**Why 90 is expected (not a defect):**

1. Machine events are single-frame only; the renderer uses exact recorded `frame` values and does not extend into neighbors.
2. The source journal itself contains exact-duplicate `headstage_reassigned` rows (same `frame`/`type`/`track`/`action`) — a journal pattern across ~16k frames in this run, not exporter duplication.
3. The renderer preserves each journal identity (no silent merge/dedup), so both `headstage_reassigned` rows are drawn.
4. Unlocalized `identity_swap` uses a top pill (no invented localization). Track-attributed `headstage_reassigned` uses accepted-pose track anchors only when structured `track` is present.
5. Overlapping same-anchor duplicates may visually coincide; unlocalized vs track-attributed remain distinguishable (pill vs diamond).

### 20.7b Manual-event boundary fixture (pixel-level)

Controlled test `test_manual_inclusive_interval_pixel_boundaries_and_styling`:

- Inclusive multi-frame `blank_node` (`track_node` on neck) over `[10, 14]`.
- Overlapping `swap_identities` (unlocalized) over `[12, 13]`.
- Additional unlocalized `review_note` over `[10, 14]`.
- Machine single-frame `node_overlap` at frame `12` only.

Probed frames: before (`9`), first (`10`), intermediate (`12`), last (`14`), after (`15`).

| Confirmation | Result |
| --- | --- |
| No off-by-one | No pixels before/after; indicators present on first/last inclusive |
| Multiple events distinguishable | Unlocalized pills (top band) vs neck-localized diamond vs machine track diamond |
| Manual vs machine styling | Teal manual vs orange machine both present and separable on overlap frame |
| No inferred pose/track attribution | Unlocalized events never get diamonds; localization uses accepted-pose geometry only |

### 20.8 Files touched (C3-2)

```text
src/overlay/corrections_s3.py
src/overlay/source_s3.py
src/overlay/export_s3.py
src/overlay/contracts.py
src/overlay/plan.py
src/overlay/adapters/pose.py
src/overlay/layers/pose.py
src/overlay/discovery.py
src/overlay/__init__.py
src/ui/controllers/overlay_qc_export_support.py
src/ui/controllers/overlay_qc_export_controller.py
src/ui/dialogs/overlay_qc_export_dialog.py
src/ui/pages/tracking_review_page.py
tests/overlay/test_export_s3_pose_qc.py
tests/overlay/test_s3_correction_events.py
tests/overlay/test_adapters.py
tests/ui/test_overlay_qc_export_gui.py
docs/roadmap/post_v011_overlay_architecture.md
```

(Smoke script / generated MP4 / PNG / tmp lab outputs remain local and are **not** committed.)

### 20.9 Remaining limitations / risks

- Full-session / dense-rate export still deferred (C3-3+).
- Topology requires optional `sleap_io`; otherwise honest nodes-only (E-session validation path).
- Loading full `tracked_pose.parquet` + full machine journal then windowing can be heavy for very long sessions / large journals (performance limitation; clip export remains the supported path).
- E-session accepted run has empty manual journal; manual visuals validated by fixtures.
- Same-anchor journal duplicates (common in this machine journal) may visually coincide; identities are preserved, not merged.
- S5 / S9-media / S10 overlay-media integration remain out of scope.

### 20.10 Closure

**C3-2 accepted** after provisional architectural review + final machine-event count verification + manual inclusive-interval pixel fixture. Commit/push authorized on this closure pass. Stop before C3-3. No S5 / S9-media / S10 overlay-media / science or artifact-schema changes.

_End of C3-2 record._

---

## 21. Increment C3-3 — S5 Blob Segmentation QC Video (2026-11-04)

### 21.1 Operator-visible product

Offline prepared-rate QC MP4 that re-draws S5 blob contours by **replaying** the recorded segmentation algorithm on the pinned prepared video + static cropped background, using the completed run’s immutable `settings_used.yaml`.

- Export family: `s5_segmentation`
- Job dir: `results_exports/overlay_qc_s5_segmentation__{timestamp}/`
- Video: `s5_segmentation_qc.mp4`
- Renderer: `overlay_renderer_c3_3_v1`
- Layers: `s5.segmentation` (+ optional `debug.frame_counter`)
- Contours on by default; optional translucent masks and scalar text (`cc_count_raw` / `cc_count` / `min_inter_contour_dist`)
- Clear on-frame legend: **S5 segmentation replay QC**

### 21.2 Replay source contract

| Pin | Authority |
| --- | --- |
| Explicit eligible S5 run | `validate_s5_run_for_downstream` (`completed` / `completed_with_warnings` + live S1 digests) |
| Settings | Run `settings_used.yaml` → `BlobSegmentationSettings` (never GUI `working_settings.json`) |
| Settings fingerprint | SHA256 over validated segmentation knobs only |
| S1 prepared video / sync / background | Live digests must match `run_meta.source_s1` |
| Parquet | Lineage/immutability only — **never** contour source |
| Implementation ids | `blob_features.segmentation.segment_frame` + `blob_features.preview.draw_contour_overlay` |

**Reject:** incomplete runs, missing settings/background, digest mismatches, fingerprint mismatches, cross-session mixes. **Never** auto-select newest. If exact replay cannot be established → explicit unavailable (no approximate contours).

**Persisted vs recomputed:** parquet stores scalars only. Masks/contours are **not** persisted and must be recomputed per prepared frame. Processing is per-frame independent (static background; no adaptive/session-stateful segmentation).

### 21.3 Geometry semantics

- Contours = recorded-algorithm external topology from `segment_frame` (same science as batch/preview).
- Optional masks = translucent binary mask overlay (not a second algorithm).
- Optional scalar text = live topology scalars from the same `segment_frame` result (faithful to drawn contours; not gap-filled parquet values).
- No invented blob identities, no S3/S4 track substitution, no contours inferred from parquet.

### 21.4 Prepared-frame timing

- Encoding domain = S1 prepared video FPS / inclusive prepared interval.
- SyncMap = prepared-identity 1:1 map for compositor reuse (**not** dense 120 Hz).
- Dense parquet `frame_idx` is never used as a video seek index.
- Export loop feeds the just-decoded prepared frame into a replay cache so drawn pixels match encoded frames.

### 21.5 Export family and GUI

- Shared compositor + streaming FFmpeg pipe + progress/cancel + atomic publish + integrity revalidation + manifest + discovery.
- GUI: shared QC Overlay Export dialog family `s5_segmentation`; entry on S5 Blob Features page (`Export QC Overlay Video…`).
- Workspace/session safety unchanged (bound session `results_exports/`).

### 21.6 Replay dependency requirements

| Requirement | Behavior |
| --- | --- |
| Recorded settings | Immutable run `settings_used.yaml` only (never GUI `working_settings.json`) |
| S1 digests | Live prepared video / sync / cropped background must match `run_meta.source_s1` |
| Implementation | Always `blob_features.segmentation.segment_frame` (no second algorithm; no parquet→contour) |
| Multi-Otsu (`blob_threshold == -2`) | Requires `scikit-image`; resolve/integrity **fail closed** if missing (no approximate threshold) |
| Runtime disclosure | Manifest / settings sidecar / processing log record current host `python` / OpenCV / NumPy / skimage versions |
| Historical environment | **Not recorded** in S5 `run_meta.json`; QC **does not claim** historical environment equivalence |

### 21.7 Reproducibility limitations

- Contours are **QC replay**, not stored truth polygons.
- Residual OpenCV seek-vs-sequential decode risk is mitigated for export by frame-cache; draw chrome (legend / scalar text / darken / frame counter) is intentionally excluded from geometry equivalence.
- Multi-Otsu requires a working `scikit-image` on the QC host; original extraction library versions were never persisted, so bit-identical historical environment replay is not asserted.
- Temporal gap-fill / dense parquet features do not affect drawn contours.
- Full-session / dense-rate export still deferred.
- Nodes / track identities are **never** inferred from segmentation (no S3/S4 substitution; parquet scalars are lineage only).

### 21.8 Laboratory validation (E-session)

- Run: `blob_features__20261021T152749` (eligible completed; Multi-Otsu `blob_threshold=-2`).
- Clip: prepared `[971, 1000]` → 30 frames @ ~119.05 fps.
- Job: `overlay_qc_s5_segmentation__20261104T163310` (lab smoke).
- Upstream digests unchanged (parquet, settings, background, prepared video).
- Contour availability: full (30/30 frames with contours; `cc_count_raw=2` on sampled frames).
- Preview vs export green-contour overlap OK on sampled frames 971 / 986 / 1000 (`all_comparisons_ok=true`).
- Evidence: `tmp_overlay_c33_e_session_s5_segmentation_qc/` + smoke script `scripts/_overlay_c33_e_session_s5_segmentation_qc_smoke.py` (local; not committed).

### 21.8a Isolated contour-equivalence (final acceptance)

Independent of legends, labels, frame counters, scalar text, and other QC decorations:

| Item | Result |
| --- | --- |
| Frames evaluated | Prepared `971, 976, 981, 986, 991, 996, 1000` (7 frames) |
| Method | Same prepared BGR (via replay cache) + pinned `settings_used` + pinned `cropped_background` → `segment_frame` vs `S5SegmentationReplayAdapter.observe`; compare with `np.array_equal` on every contour array and on the binary mask |
| Contour geometry | **Exact match** on all 7 frames |
| Masks | **Exact match** on all 7 frames |
| Differences | None |
| Identities | No `track` / blob-id keys in observation payload; `no_invented_blob_identities=true` |
| Upstream digests | Unchanged after equivalence probe |

Evidence JSON (local): `tmp_overlay_c33_e_session_s5_segmentation_qc/C33_ISOLATED_CONTOUR_EQUIVALENCE.json`.

### 21.9 Remaining performance limitations

- Per-frame re-segmentation (Multi-Otsu + morphology) dominates wall time; short prepared-rate clips remain the supported path.
- Full-session / dense-rate QC export is still deferred.
- Loading/decoding long prepared videos for wide intervals remains I/O heavy (same shared FFmpeg/OpenCV path as prior QC families).

### 21.10 Files touched (C3-3)

```text
src/overlay/source_s5.py
src/overlay/export_s5.py
src/overlay/adapters/segmentation.py
src/overlay/layers/segmentation.py
src/overlay/contracts.py
src/overlay/plan.py
src/overlay/discovery.py
src/overlay/__init__.py
src/overlay/adapters/__init__.py
src/overlay/layers/__init__.py
src/ui/controllers/overlay_qc_export_support.py
src/ui/controllers/overlay_qc_export_controller.py
src/ui/dialogs/overlay_qc_export_dialog.py
src/ui/pages/blob_features_page.py
tests/overlay/test_export_s5_segmentation_qc.py
tests/overlay/test_adapters.py
tests/ui/test_overlay_qc_export_gui.py
docs/roadmap/post_v011_overlay_architecture.md
```

(Smoke script / generated MP4 / PNG / tmp lab outputs remain local and are **not** committed.)

### 21.11 Closure

**C3-3 accepted** after provisional architectural review + replay-dependency disclosure hardening + isolated contour-equivalence on E-session frames. Commit/push authorized on this closure pass. Stop before C3-4. No mixed overlays, S9/S10 media integration, full-session exports, or scientific / artifact-schema changes.

_End of C3-3 record._
