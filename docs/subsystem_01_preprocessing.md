# Subsystem 01: Video Preprocessing

- **Status:** Functionally closed and entering maintenance
- **Subsystem:** Raw behavioral-video preparation for SLEAP-compatible input
- **Current source of truth:** This document, `docs/subsystem_01_status_and_roadmap.md`, and `docs/design/subsystem_01_geometry_modes.md`

Subsystem 01 transforms raw behavioral videos into validated prepared videos while preserving frame identity, timing traceability, accepted spatial geometry, preprocessing settings, and artifact provenance.

It validates prepared-video compatibility and frame-domain integrity.

It does not validate pose quality, SLEAP model accuracy, tracking quality, instance counts, confidence scores, coordinate exports, inference results, or SLEAP output-row structure. Those belong to the future SLEAP subsystem.

---

## 1. Normal user workflow

The normal GUI workflow is intentionally simple:

```text
Choose video
→ inspect / trim / optional pre-crop
→ detect cage automatically or select Manual ROI
→ review geometry
→ optional static mask
→ prepare
```

Internal concepts such as source facts, detector diagnostics, camera context, and future detector presets are developer/design concepts. They must not become mandatory user-facing steps in the ordinary workflow.

---

## 2. Current responsibility boundary

Subsystem 01 is responsible for:

- reading and validating raw-video properties needed for preprocessing;
- preserving raw decode-order frame identity;
- selecting one contiguous raw-frame interval;
- optionally using external timing vectors;
- accepting spatial geometry before final processing;
- producing a prepared video readable by OpenCV and SLEAP video reading;
- producing official synchronization, metadata, background, settings, and log artifacts;
- failing explicitly when validation gates do not pass.

Subsystem 01 is not responsible for:

- SLEAP model inference;
- pose or tracking quality;
- instance count correctness;
- confidence-score interpretation;
- SLEAP coordinate export structure;
- downstream behavioral analysis.

The completed read-only SLEAP handoff check showed:

```text
expected_frame_count = 5716
opencv_readable_frame_count = 5716
opencv_reported_frame_count = 5716
prepared size = 928 x 528
SLEAP video shape = (5716, 528, 928, 3)
SLEAP-reported frame count = 5716
frame 0 read successfully
frame 2858 read successfully
frame 5715 read successfully
```

Therefore:

```text
prepared-video readable-frame count
=
SLEAP-readable frame count
=
5716
```

---

## 3. Source facts and raw-video inspection

Raw-video inspection records source facts such as:

- raw width and height;
- aspect ratio;
- FPS candidates;
- reported frame count;
- sequential OpenCV-readable frame count;
- container and codec facts where useful;
- source fingerprint data used for safe cache/reuse decisions.

Source facts are not scene understanding. They cannot determine cage occupancy or identify the true ROI.

Scene characterization requires image content, such as representative frames, background imagery, possible cage contours, candidate ROI geometry, border/contour evidence, and detection confidence. Those are produced by detector/manual-review workflows, not by raw probing alone.

Full sequential readable-frame counting is used where required for trustworthy frame-count validation. Cached or prior counts may be used only when tied to a proven selected source identity.

---

## 4. Frame identity, trim, and timing

Raw decode frame index is the primary frame identity. Raw PTS is diagnostic only and may be missing, duplicated, non-monotonic, or otherwise untrusted.

Current supported trim is one contiguous half-open interval:

```text
[start_frame, end_frame_exclusive)
```

Rules:

- `start_frame` is included.
- `end_frame_exclusive` is excluded.
- `start_frame` must be nonnegative.
- when both are set, `end_frame_exclusive` must be greater than `start_frame`.
- non-consecutive/discontinuous trimming is not implemented.

Default frame mapping is:

```text
prepared_frame_idx → raw_decode_frame_idx
raw_decode_frame_idx = start_frame + prepared_frame_idx
```

Default processing must not silently drop, duplicate, interpolate, reorder, or resample frames.

Optional external MATLAB timing uses a user-selected numeric vector from a `.mat` workspace. The selected vector must be one-dimensional after squeeze, finite, monotonically increasing, and exactly match the original untrimmed raw-video readable-frame count. Temporal units are converted to seconds and mapped by raw decode frame index.

Automatic timing-variable selection is deferred.

---

## 5. Spatial geometry and CropPlan behavior

Current implemented final spatial geometry uses the perspective `CropPlan` workflow:

```text
raw video
→ optional pre-crop
→ automatic cage detection or manual four corners
→ accepted CropPlan
→ rectification / rotation / canonical scale-pad
→ prepared video
```

Current behavior:

- automatic cage detection produces a candidate `CropPlan`;
- manual ROI currently uses manual four-corner geometry and also produces a `CropPlan`;
- the user must explicitly accept geometry before final processing;
- accepted geometry is stored as the authoritative spatial transform in metadata;
- prepared video remains rectangular;
- canonical output sizing preserves aspect ratio through uniform scaling and padding.

Future geometry modes are designed separately in `docs/design/subsystem_01_geometry_modes.md`. They include identity, axis-aligned pre-crop-only, current perspective CropPlan, and future composed geometry. Those future modes are not implemented unless explicitly stated in a later scoped implementation.

Future simple manual source-aligned rectangle crop with optional manual rotation is deferred. Current documentation must not imply that existing CropPlan behavior has changed.

---

## 6. Visual pre-crop behavior

Current pre-crop is an optional source-region selection that affects final processing. It is not merely a detector hint.

When enabled, pre-crop excludes pixels outside the selected region from final prepared output and is included in the accepted raw-to-prepared spatial transform.

Visual and numeric pre-crop controls are input methods for the same typed core pre-crop configuration. GUI widgets must not own independent scientific geometry rules; they must pass accepted values through core validation.

Changing spatial geometry, including effective pre-crop geometry, invalidates accepted geometry review. Changing only the trim interval for an unchanged raw video changes which frames are processed, not how each selected frame is transformed.

Future pre-crop and simple-rectangle crop improvements are maintenance/roadmap items. They must still feed typed core configuration and validation paths rather than duplicating scientific validation in GUI widgets.

---

## 7. Prepared-video coordinate system and static masks

Prepared-video coordinates are pixel coordinates in the final prepared frame after crop, rectification, rotation, uniform scaling, and padding.

Static exclusion masks are implemented in prepared-video coordinates:

- supported shapes: axis-aligned rectangles and simple polygons;
- fill value: black;
- mask is applied identically to every prepared frame;
- disabled or empty mask is a no-op;
- mask is applied after current spatial transform and before final video encoding/background generation;
- no separate mask-image artifact is required.

Static rectangle and polygon masks have been tested through the real GUI and preprocessing workflow, verified in prepared output, and checked against prepared-video/background behavior.

Dynamic/keyframed masks are deferred. They would require frame-indexed geometry, interpolation or visibility rules, metadata representation, preview behavior, validation, and background-generation decisions.

---

## 8. Video preparation, staging, and promotion guarantees

The current pipeline uses a legacy-compatible two-stage preparation design:

```text
Stage A: ffmpeg transformation and intermediate encode
Stage B: OpenCV sequential final re-encode
```

Stage A performs the accepted trim/spatial transform work. Stage B sequentially reads the intermediate and writes the final prepared video, applying any static prepared-coordinate mask.

The system must stage intermediate and final artifacts safely:

- partial outputs must not be promoted as official success;
- cancellation must preserve previously validated official outputs where applicable;
- failed validation must stop promotion;
- stale task results must not update current GUI/project state;
- no hidden fallback may bypass a failed validation gate.

Long-running GUI work uses task/generation identity so stale progress or completion events from an old project/video context are discarded.

---

## 9. Prepared-video validation

Final prepared videos must pass strict validation:

```text
OpenCV reported frame count
=
OpenCV sequentially readable frame count
=
expected trimmed frame count
```

Validation also checks final prepared width and height against expected dimensions and requires even dimensions for the supported encoding path.

Any mismatch is a hard failure. The legacy `prepared_count - 1` workaround is not allowed.

Subsystem 01 closure additionally verified that SLEAP video reading sees the same frame domain for a prepared video:

```text
prepared-video readable-frame count
=
SLEAP-readable frame count
```

This is a compatibility/frame-domain check only, not inference validation.

---

## 10. Official artifacts

A successful preprocessing run produces:

```text
ProjectName/
└── preprocess/
    ├── prepared_video.mp4
    ├── prepare_meta.json
    ├── prepared_sync.npz
    ├── cropped_background.png
    ├── settings_used.yaml
    └── processing_log.txt
```

Artifact meanings:

| Artifact | Meaning |
| --- | --- |
| `prepared_video.mp4` | Final SLEAP-compatible prepared video |
| `prepare_meta.json` | Authoritative run metadata, geometry, timing, validation, mask, software, and provenance |
| `prepared_sync.npz` | Frame-level raw-to-prepared identity and timing arrays |
| `cropped_background.png` | Background generated from the final prepared video |
| `settings_used.yaml` | Accepted settings used for the run |
| `processing_log.txt` | Processing commands, validation summaries, warnings, and failures |

Intermediate files belong under internal/debug locations and are not official outputs.

Official artifact names and schema meanings must not be changed without an explicit specification update.

---

## 11. Supported installation/runtime

The supported Windows lab/developer workflow is:

```bat
git pull
scripts\install_windows_gui.bat
scripts\launch_windows_gui.bat
```

The supported runtime currently uses Python 3.12, FFmpeg 7.1.1 from conda-forge, PySide6 6.11.1 from conda-forge, and editable installation of the current checkout.

The Windows runtime has been field-tested on two machines, including repair of a mixed pip/Conda PySide6 environment, successful FFmpeg doctor/preflight, GUI launch, and a real preprocessing run that produced a prepared video.

Packaged cross-platform desktop releases are deferred release-engineering work.

---

## 12. Historical implementation decisions retained

The following decisions remain binding unless a future specification changes them:

- raw decode-order identity is authoritative;
- raw PTS remains diagnostic only;
- valid external temporal timing is preferred for timing/FPS when available;
- default processing is no-resampling;
- geometry must be explicitly accepted before final processing;
- final prepared video must be strictly validated;
- official artifacts are fixed;
- scientific processing logic belongs in core modules, not GUI widgets.

Historical plans, audits, and release snapshots are preserved under `docs/archive/` for traceability. They are not the current source of truth.

---

## 13. Future/deferred work

Future maintenance and roadmap items include:

- detector robustness and diagnostics;
- optional reusable detector presets;
- normalized resolution-aware detector defaults;
- manual-first ROI workflow improvements;
- future simple rectangle crop with optional manual rotation;
- segmented non-consecutive trimming;
- dynamic/keyframed masks;
- batch preprocessing;
- packaged cross-platform releases.

These are not blockers to the current functional closure.
