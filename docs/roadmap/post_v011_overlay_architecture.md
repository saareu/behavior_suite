# Post-v0.1.1 — Overlay Architecture (C0 audit → C1 foundation → C2-A/C2-B QC export)

**Status:** **C2-B accepted and closed** (S4 provenance-styled pose QC video). Builds on C1 + C2-A at `origin/main`.  
**C2-B final acceptance date:** 2026-11-02  
**C2-A / C1 final acceptance date:** 2026-11-02  
**C0 date:** 2026-11-01  
**C2-B base commit:** `a1da1e170297619b338a1d63f0c8239f5cbf5ea0` (`origin/main` — C1/C2-A + mainline CSV)  
**Companion:** [`post_v011_s10_review_and_visualization.md`](post_v011_s10_review_and_visualization.md) (Increment B closed; overlays → this document)  
**Non-scope (still):** GUI overlay integration, full-session QC exporter, PI/gallery overlay integration, S5 contour generation, live overlays, scientific algorithm / schema mutation.  
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

Still deferred: GUI overlay launch/integration, PI/gallery overlay, full-session exporter, dense-rate export, S5 contours.

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
