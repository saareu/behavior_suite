# Subsystem 01 Status and Roadmap

**Subsystem:** Video preprocessing  
**Status date:** 2026-07-05  
**Purpose:** Compact current-status document for the current Subsystem 01 milestone, maintenance status, and deferred roadmap items.

This document is a status and roadmap companion to the canonical functional specification in `docs/subsystem_01/preprocessing.md`. It does not replace the scientific invariants in that specification.

Active related documentation:

- `docs/subsystem_01/preprocessing.md`
- `docs/subsystem_01/design/geometry_modes.md`
- `docs/development/ai_coding_guide.md`

---

## 1. Purpose and scope

Subsystem 01 prepares raw behavioral videos for SLEAP by producing validated prepared videos, synchronization arrays, metadata, settings, background images, and logs.

The scope remains narrow:

- preserve raw decode-order frame identity;
- preserve deterministic raw-to-prepared frame mapping in the default no-resampling mode;
- validate prepared videos strictly;
- record timing, geometry, preprocessing settings, and provenance;
- hand off a prepared video that SLEAP can read frame-for-frame.

This subsystem does not evaluate SLEAP pose quality, tracking quality, pose confidence, downstream pose rows, behavior features, or behavioral classifications.

Current functional status:

```text
Functionally closed and entering maintenance.
```

This does not forbid future feature work. It means the current Subsystem 01 functional implementation milestone has passed its required compatibility checks, and future work should be treated as separate scoped roadmap items.

---

## 2. What is implemented and field-tested

The current implementation supports the ordinary preprocessing workflow:

```text
Choose video
→ inspect / trim / optionally pre-crop
→ choose Detect cage automatically, Manual ROI, or Full frame — no crop
→ review accepted geometry
→ optionally add static exclusion masks
→ prepare video
```

The following are implemented:

- project creation/opening and official preprocess artifact locations;
- raw-video probing and full sequential readable-frame counting;
- one contiguous trim interval: `[start_frame, end_frame_exclusive)`;
- optional external MATLAB timing-vector selection and validation;
- optional pre-crop before final spatial processing;
- automatic cage detection with adjustable detector settings;
- manual ROI route with four-corner and axis-aligned rectangle selection;
- Full frame — no crop route for preparing the whole raw frame without
  automatic detection, manual geometry, pre-crop, perspective crop, or
  automatic rotation;
- live crop-content width/height display for valid manual geometry;
- explicit crop/geometry acceptance before final processing;
- two-stage video preparation with ffmpeg followed by OpenCV final re-encode;
- prepared-coordinate static exclusion masks filled with black;
- background generation from the final prepared video;
- `prepared_sync.npz`, `prepare_meta.json`, `settings_used.yaml`, and processing log generation;
- strict OpenCV prepared-video validation;
- GUI workflow through Run and Validate;
- responsive long-running tasks, cancellation, stale task-result suppression, raw-frame trim navigation, detector reset-to-defaults, and Windows GUI installer hardening.

Field-test evidence recorded in the existing v1 documents includes two successful full-video GUI workflows:

- automatic cage detection with external timing;
- manual four-corner crop without external timing.

Additional completed field-tested checks:

- External timing has already been successfully used in a real prepared project. Timing-variable auto-selection remains outside Subsystem 01 scope.
- Static prepared-coordinate exclusion masks have been tested through the real GUI and preprocessing workflow. Rectangle and polygon masks were verified in prepared output, and prepared-video/background behavior was checked.
- The supported Windows runtime, installer, and launcher have been tested on two different Windows machines. A mixed pip/Conda PySide6 environment was repaired successfully, FFmpeg 7.1.1 doctor/preflight passed, the GUI launched successfully, and a real preprocessing run produced a prepared video on the second machine.
- The SLEAP read-only handoff check passed without running model inference. Prepared-video metadata recorded expected/readable/reported frame counts of 5716 with prepared size 928 × 528. SLEAP reported video shape `(5716, 528, 928, 3)`, reported 5716 frames, and successfully read frame 0, frame 2858, and frame 5715. Therefore:

  ```text
  prepared-video readable-frame count
  =
  SLEAP-readable frame count
  =
  5716
  ```

---

## 3. Remaining maintenance and roadmap follow-up items

The current functional implementation milestone is closed. Remaining follow-up work is not a blocker to functional closure:

1. Review the local untracked issue log later and integrate only durable items into active documentation.
2. Future detector robustness, presets, diagnostics, normalized defaults, and manual-first ROI improvements.
3. Future segmented non-consecutive trimming, dynamic masks, batch preprocessing, and packaged cross-platform releases.

The completed SLEAP handoff check was a prepared-video compatibility/frame-domain check only. Subsystem 01 still does not claim validation of pose quality, model accuracy, tracking quality, instance counts, confidence scores, coordinate exports, SLEAP inference results, or SLEAP output-row structure.

---

## 4. Completed narrow SLEAP handoff/frame-count closure requirement

Subsystem 01 required one read-only SLEAP handoff check:

```text
prepared-video readable-frame count
=
frame count SLEAP reads from that prepared video
```

This check has passed:

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

This is intentionally narrow. It does not include validating:

- pose quality;
- tracking quality;
- confidence values;
- exported pose rows;
- identity assignment;
- downstream behavioral features.

Pose inference and technical pose QC belong to the finalized Subsystem 02 MVP.
Identity, tracking correctness, and scientific usability belong to downstream
analysis subsystems.

---

## 5. Current supported installation/runtime workflow

The supported Windows lab/developer runtime workflow is:

```bat
git pull
scripts\install_windows_gui.bat
scripts\launch_windows_gui.bat
```

The supported runtime currently uses:

```text
Python 3.12
FFmpeg 7.1.1 from conda-forge
PySide6 6.11.1 from conda-forge
pip for editable behavior_suite installation
```

This remains the supported workflow for now. Future release engineering should move toward packaged desktop distributions, Windows first and then relevant Linux/macOS support, so regular application launch does not require Git, Conda, or terminal commands.

No packaging framework is selected by this document, and this document does not replace the current installer with `uv` or another developer tool.

---

## 6. Current normal user workflow

The normal GUI workflow should remain simple and user-facing:

```text
Choose video
→ inspect / trim / optionally pre-crop
→ choose Detect cage automatically, Manual ROI, or Full frame — no crop
→ review accepted geometry
→ optionally add static exclusion masks
→ prepare video
```

Internal concepts such as source facts, scene characterization, camera context, detector presets, and future profile eligibility must not become mandatory user-facing steps in the ordinary workflow.

---

## 7. Detector preset / camera robustness future design

### Source facts versus scene characterization

Source facts are facts available from probing or container-level inspection:

- raw width and height;
- aspect ratio;
- FPS;
- reported frame count;
- sequential readable-frame count;
- container and codec facts where useful.

Scene characterization requires image content:

- representative frame/background content;
- possible cage contours;
- candidate ROI geometry;
- candidate location and occupancy within the source frame;
- border or contour evidence;
- detection confidence and diagnostics.

Raw probing alone cannot know cage occupancy or identify the true ROI. ROI occupancy or candidate plausibility can only be computed after image-based detection produces a candidate, or after the user accepts a manual ROI.

### Normal videos

For normal videos, users should be able to:

- use default detection;
- adjust local detector settings such as threshold;
- rerun detection;
- choose Manual ROI.

Manual ROI is a first-class valid route, not merely an emergency fallback after failed automatic detection.

### Optional future detector presets

For recurring difficult or new camera/cage setups, a future optional workflow may be:

```text
Create or refine detector preset
```

A detector preset should be created from representative frames plus either an accepted automatic candidate or an accepted manual ROI.

It may store reusable normalized context such as:

- normalized detection pre-crop;
- expected ROI location and size ranges;
- expected ROI occupancy ranges;
- contour-size fractions;
- morphology scaling reference;
- source-frame reference dimensions;
- source aspect-ratio expectations;
- detector padding/margins in normalized or scale-derived form.

A detector preset is only a starting point. It must not lock the user into parameters. Threshold is especially per-video/per-lighting and should remain easily adjustable locally; it is not a resolution-normalized quantity.

Profile or preset recommendation must not be claimed from FFmpeg/video probe facts alone.

### Detector outcomes and diagnostics

Future detector behavior should distinguish:

- high-confidence automatic candidate;
- low-confidence candidate requiring review;
- inconclusive/no reliable candidate;
- manual-only route.

An inconclusive result must not imply that detection is impossible. A threshold, morphology, pre-crop, or margin adjustment may yield a good new candidate.

Future diagnostics should help users understand and tune results through views such as:

- representative frame;
- detection pre-crop;
- thresholded/edge image;
- candidate contours;
- accepted/rejected candidate overlays;
- concise rejection reasons.

---

## 8. Manual ROI and geometry decisions

Current manual geometry supports the existing four-corner CropPlan workflow and a manual axis-aligned rectangle route. Both produce accepted `CropPlan` geometry before final preprocessing. The rectangle route preserves source orientation and does not request the historical automatic 90° portrait-to-landscape rotation.

Future geometry distinctions should continue to be documented and implemented carefully:

### Manual axis-aligned rectangle

- simple crop in source coordinates;
- may later support explicit user-selected rotation.

### Manual quadrilateral CropPlan

- perspective rectification;
- does not imply an additional automatic 90°/180° rotation;
- orientation follows the selected quadrilateral/corner-order contract.

### Final prepared frame

The final prepared video frame remains rectangular.

### Full frame — no crop

Full frame — no crop is implemented as an explicit geometry-review route. It
uses the whole raw frame, requires pre-crop to be disabled, performs no
automatic detector invocation, uses no manual geometry, applies no perspective
crop, and records `rotated_90: false`.

Static masks remain a separate optional prepared-coordinate operation.
Canonical scale/pad remains available and may make the prepared output size
different from the full-frame content size; canonical scale/pad is not
cropping.

### Irregular exclusions

Irregular excluded regions use prepared-coordinate polygon masks, not a non-rectangular prepared-video format.

The broader final-spatial-geometry design is documented in `docs/subsystem_01/design/geometry_modes.md`.

---

## 9. Static-mask status and future dynamic-mask deferral

Static prepared-coordinate exclusion masks are implemented as a Subsystem 01 feature:

- coordinates are in final prepared-video pixels;
- rectangles and simple polygons are supported;
- masks are applied after current spatial transform and before final video encoding;
- masked pixels are black;
- disabled or empty masks are no-ops;
- mask geometry is represented in metadata and settings;
- no separate mask-image artifact is required.

Dynamic masks are intentionally deferred. A dynamic mask means prepared-coordinate mask geometry or visibility that changes over time. Dynamic masks are more complex because they require frame-indexed geometry, optional interpolation rules, metadata representation, preview behavior, validation, and background-generation decisions.

---

## 10. Trimming status and future segmented trimming

Current supported trim remains:

```text
one contiguous [start_frame, end_frame_exclusive) interval
```

Future non-consecutive/discontinuous trimming is deferred.

When implemented, discontinuous trimming must not silently concatenate non-consecutive raw intervals into one fake continuous video. The expected future direction is:

- segment manifest;
- separate prepared video per retained segment;
- explicit raw-frame mapping per segment;
- preserved discontinuity boundaries;
- no silent time compression.

---

## 11. Explicitly deferred features

The following are intentionally deferred, not missing bugs:

- automatic timing-variable selection;
- batch preprocessing;
- non-consecutive/discontinuous trimming;
- dynamic/keyframed masks;
- image-sequence export;
- interactive post-crop rotation;
- packaged cross-platform desktop releases;
- developer installation through `uv` or another future developer tool.

---

## 12. Recommended implementation order after functional closure

Subsystem 01 is functionally closed and entering maintenance. Recommended future work should remain separately scoped:

1. Review the local untracked issue log later and integrate only durable items.
2. Treat future feature work as separate scoped roadmap items, including detector robustness and diagnostics, optional reusable detector presets, normalized resolution-aware detector defaults, manual-first ROI workflow, optional manual crop rotation, segmented non-consecutive trimming, dynamic/keyframed masks, batch preprocessing, and packaged cross-platform releases.

---

## 13. Real-data acceptance matrix

| Check | Status | Input | Expected result | Scope boundary |
| --- | --- | --- | --- | --- |
| Automatic ROI + external timing | Completed/field-tested | Representative full video with valid timing vector | Six official artifacts, strict prepared-video validation, external timing recorded | Does not validate SLEAP poses |
| Manual ROI + no external timing | Completed/field-tested | Representative full video | Six official artifacts, strict prepared-video validation, fallback FPS source recorded | Does not validate tracking |
| Static mask | Completed/field-tested | Real GUI and preprocessing workflow with rectangle and polygon masks | Masked pixels verified in prepared output; prepared video and background behavior checked | Does not implement dynamic masks |
| Windows runtime | Completed/field-tested | Two Windows machines, including a previously mixed pip/Conda PySide6 environment | Mixed environment repaired, FFmpeg 7.1.1 doctor/preflight passed, GUI launched, and a real preprocessing run produced a prepared video on the second machine | Does not create packaged release |
| SLEAP handoff | Completed/field-tested | Prepared video from accepted workflow | Expected/readable/SLEAP frame count = 5716; prepared size = 928 × 528; SLEAP read frame 0, frame 2858, and frame 5715 | Does not evaluate pose quality, model accuracy, tracking, instance counts, confidence values, coordinate exports, inference results, or output-row structure |
| Regression tests | Ongoing development practice | Small known inputs, generated frames, mocked FFmpeg output, or small fixtures | Expected outputs/errors/behavior retained permanently | Does not require large real lab videos |

Regression tests mean:

```text
small known inputs
→ expected outputs/errors/behavior
→ retained permanently so future changes do not break verified behavior
```
