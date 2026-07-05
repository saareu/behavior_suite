# Preprocess Subsystem Specification v1

## 1. Document Control

**Subsystem:** Video Preprocessing
**Specification Version:** v1.0
**Status:** Approved implementation baseline
**Intended System:** Scientific behavioral video processing suite for SLEAP-based pose estimation and later downstream analysis
**Primary Output:** SLEAP-compatible prepared video with deterministic mapping to source-video frame identity
**Prepared For:** Engelhard Lab behavioral video processing workflow

---

## 2. Purpose

The Preprocess Subsystem prepares raw behavioral videos for SLEAP pose estimation while preserving deterministic frame identity, scientifically traceable timing, and reproducible transformation metadata.

The subsystem shall allow a user to:

1. Create or open a project.
2. Select a raw video.
3. Optionally select an external MATLAB `.mat` timing file.
4. Select the appropriate timing vector manually.
5. Define a frame range.
6. Optionally apply a pre-crop.
7. Detect the cage automatically or define it manually.
8. Review and accept the final crop.
9. Generate a SLEAP-compatible prepared video.
10. Generate synchronization, background, metadata, settings, and log artifacts.

The prepared video shall preserve a deterministic one-to-one relationship between prepared frame indices and selected raw-video decode-order frame indices unless a future validated mode explicitly allows temporal resampling.

---

## 3. Scope

### 3.1 In Scope

The Preprocess Subsystem v1 shall support:

1. Shared project creation and project loading.
2. Raw video selection and probing.
3. Start-frame and end-frame selection.
4. Optional external timing selection from a generic MATLAB `.mat` workspace.
5. Manual selection of a timing variable from the MATLAB workspace.
6. Validation of the selected timing vector against the original untrimmed raw-video frame space.
7. Optional detection pre-crop.
8. Automatic cage detection.
9. Detector setting adjustment and repeated detection attempts.
10. Manual four-corner cage crop fallback.
11. User acceptance of crop geometry before final processing.
12. Legacy-compatible two-stage video preparation:

    * ffmpeg transformation and intermediate encoding;
    * OpenCV final re-encoding for SLEAP compatibility.
13. Configurable ffmpeg and OpenCV codec settings.
14. Aspect-ratio-preserving canonical resolution.
15. Strict prepared-video frame-count validation using OpenCV.
16. Background generation from the final prepared video.
17. Synchronization artifact generation.
18. Metadata generation.
19. Settings artifact generation.
20. Structured logging.
21. Hard failure on frame-count ambiguity, invalid TTL selection, invalid crop geometry, or invalid final output.

### 3.2 Explicitly Out of Scope for v1

The following are deferred:

1. SLEAP tracking implementation.
2. Pose QC and pose-fixing implementation.
3. Blob extraction and behavioral clustering implementation.
4. Image-sequence export.
5. Interactive rotated ROI adjustment after initial manual four-corner crop.
6. Batch preprocessing.
7. Automatic timing-vector variable-name recognition.
8. Automatic synchronization of unrelated timing vectors to video frames.
9. Copying raw video or timing files into the project directory.
10. Checksumming or archival of source files.
11. Alternative validated processing recipes beyond the default legacy-compatible profile.
12. Dynamic masks that vary across frames.

### 3.3 Reserved Future Capability: Static Prepared-Coordinate Masking

Static masking is approved as a future preprocessing capability but is not required for v1 implementation.

When implemented, the mask shall:

1. Be defined in final prepared-video coordinates.
2. Be applied after crop, rectification, rotation, canonical scaling, and padding.
3. Be applied identically to every prepared frame.
4. Fill masked pixels with black.
5. Be applied before background estimation.
6. Cause the generated background to contain the black masked region naturally.
7. Store mask geometry in `prepare_meta.json`.
8. Not require a separate mask-image artifact.

### 3.4 Current Closure Clarifications

This specification remains the canonical source for Subsystem 01 scientific invariants and artifact contracts. Current implementation status, closure checks, and deferred roadmap items are summarized in `docs/subsystem_01_status_and_roadmap.md`.

The ordinary user workflow should remain simple:

```text
Choose video
→ inspect / trim / optionally pre-crop
→ choose Detect cage automatically or Manual ROI
→ review accepted geometry
→ optionally add static exclusion masks
→ prepare video
```

Internal detector-design concepts such as source facts, scene characterization, detector presets, camera context, or profile eligibility are not mandatory user-facing workflow steps.

Source facts include raw width and height, aspect ratio, FPS, reported frame count, sequential readable-frame count, and useful container/codec facts. Scene characterization requires image content, such as representative frames, background images, possible cage contours, candidate ROI geometry, candidate occupancy, border/contour evidence, and detection confidence/diagnostics. Raw probing alone cannot know cage occupancy or identify the true ROI.

Manual ROI remains a first-class route. Future geometry work may distinguish manual axis-aligned rectangles, manual quadrilateral CropPlans, and composed geometry, but the final prepared video frame remains rectangular. Irregular excluded regions belong in prepared-coordinate polygon masks, not in a non-rectangular prepared-video format.

Subsystem 01 closure requires a narrow SLEAP handoff check:

```text
prepared-video readable-frame count
=
frame count SLEAP reads from that prepared video
```

This check does not include pose quality, tracking quality, confidence values, exported pose rows, or other SLEAP result semantics.

Current trim support remains one contiguous `[start_frame, end_frame_exclusive)` interval. Future discontinuous trimming must preserve segment boundaries explicitly instead of silently concatenating non-consecutive raw intervals into one fake continuous video.

---

## 4. Definitions

### 4.1 Raw Video

The original user-selected behavioral camera video.

### 4.2 Raw Decode Frame Index

The frame index assigned according to sequential decode order from the raw video.

Raw decode frame index is the primary source of source-frame identity.

### 4.3 Prepared Video

The final cropped, rectified, optionally padded, and re-encoded video intended for SLEAP processing.

### 4.4 Prepared Frame Index

The zero-based frame index in the prepared video.

### 4.5 Raw PTS

A timestamp associated with a raw-video frame by the video container or decoder.

Raw PTS may be missing, duplicated, non-monotonic, corrupted, or otherwise untrusted.

### 4.6 External Timing Vector

A user-selected numeric vector loaded from a MATLAB `.mat` workspace. It contains one value per original untrimmed raw-video frame.

The user declares the units of the vector.

When units represent time, each vector value represents an absolute acquisition time for the corresponding raw frame.

### 4.7 CropPlan

An internal geometry object representing the transformation from raw frame coordinates to prepared-video coordinates.

A CropPlan may be created by automatic cage detection or manual four-corner crop selection.

### 4.8 Detection Pre-Crop

An optional coarse crop applied before cage detection and included in the final raw-to-prepared transform.

Detection pre-crop is not merely a detector hint. When enabled, pixels outside the selected region are excluded from final preprocessing.

### 4.9 Canonical Resolution

A user-configurable final output resolution. The native rectified crop shall be uniformly scaled to fit within this resolution and padded as required. Non-uniform scaling is prohibited.

### 4.10 Frame Resampling

Any operation that duplicates, drops, interpolates, temporally reorders, or otherwise changes the one-to-one correspondence between selected raw frames and prepared frames.

Frame resampling is prohibited by default.

---

## 5. Design Principles

### 5.1 Frame Identity Principle

The primary mapping shall be:

```text
prepared_frame_idx → raw_decode_frame_idx
```

For the default no-resampling mode:

```text
raw_decode_frame_idx = start_frame + prepared_frame_idx
```

### 5.2 External Timing Priority Principle

Timing sources shall be prioritized as follows:

```text
1. external_time_sec, when valid
2. raw_pts_time_sec, when valid and external timing is absent
3. prepared_time_sec, as a playback-domain fallback only
```

### 5.3 PTS Robustness Principle

Raw PTS shall be treated as diagnostic metadata and shall not define frame identity.

Failure to extract usable raw PTS shall not cause preprocessing to fail when decode-order mapping and final-video validation are successful.

### 5.4 No Silent Repair Principle

The subsystem shall not silently accept:

```text
frame-count mismatch
unreadable prepared frames
invalid timing-vector length
invalid timing-vector values
unaccepted crop geometry
invalid output dimensions
failed metadata validation
```

### 5.5 User Approval Principle

Final video processing shall not begin until the user explicitly accepts the crop geometry.

### 5.6 Metadata Authority Principle

`prepare_meta.json` shall be the single authoritative metadata file for the preprocessing run.

### 5.7 Geometry Preservation Principle

Prepared-video processing shall preserve cage and animal geometry.

No non-uniform x/y scaling is permitted.

---

## 6. User Workflow

The v1 workflow shall be:

```text
Open application
↓
Create project or open existing project
↓
Select raw video
↓
Probe raw video
↓
Optional: select MATLAB .mat timing file
    ↓
    Display workspace variables
    ↓
    User selects timing variable manually
    ↓
    User declares timing-vector units
    ↓
    Validate selected vector
↓
Choose start frame and end frame
↓
Optional: define detection pre-crop
↓
Find cage
↓
Display one cropped and rotated preview frame
↓
User chooses:
    Accept crop
    Change detector settings and retry
    Manual four-corner crop
↓
Optional: adjust advanced settings
↓
Run preprocessing
↓
Validate final prepared video
↓
Write official artifacts
```

---

## 7. Project and Repository Boundaries

### 7.1 Shared Project Module

Project creation, project loading, project validation, and project-path management are shared system capabilities.

They shall reside outside the preprocess module.

The preprocess subsystem shall receive an already-created or opened project object.

### 7.2 Repository Layout

The repository root is named `behavior_suite`.

The recommended source layout is:

```text
behavior_suite/
├── pyproject.toml
├── README.md
├── configs/
│   └── preprocess_default.yaml
│
├── docs/
│   ├── preprocess_subsystem_spec_v1.md
│   ├── preprocess_implementation_plan_v1.md
│   └── ai_coding_guide.md
│
├── legacy/
│   ├── prepare_reference.py
│   └── cage_cropper_reference.py
│
├── src/
│   ├── project/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── paths.py
│   │   ├── service.py
│   │   └── validation.py
│   │
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── video_probe.py
│   │   ├── mat_sync_reader.py
│   │   ├── crop_plan.py
│   │   ├── pre_crop.py
│   │   ├── cage_detection.py
│   │   ├── manual_crop.py
│   │   ├── masking.py
│   │   ├── video_prepare.py
│   │   ├── background.py
│   │   ├── sync_writer.py
│   │   ├── metadata.py
│   │   ├── validation.py
│   │   ├── logging_utils.py
│   │   └── service.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   │
│   └── ui/
│       └── ...
│
└── tests/
    ├── project/
    ├── preprocess/
    └── integration/
```

The repository shall not contain an unnecessary nested package directory such as:

```text
src/behavior_suite/
```

---

## 8. Official Artifact Directory

The official preprocessing artifact directory shall be:

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

The subsystem may create temporary files internally.

Internal artifacts shall be written only under:

```text
ProjectName/preprocess/.internal/
ProjectName/preprocess/debug/
```

Internal files shall be removed after successful processing unless debug mode is enabled.

---

## 9. Official Outputs

### 9.1 prepared_video.mp4

The final SLEAP-compatible prepared video.

It shall:

1. Be readable by OpenCV.
2. Have a constant FPS header.
3. Preserve one-to-one mapping to selected raw decode frames.
4. Contain no audio.
5. Have even width and height.
6. Preserve intended cage and animal geometry.
7. Use uniform scaling only.
8. Match configured canonical resolution when canonical resolution is enabled.
9. Pass strict OpenCV reported-versus-readable frame validation.

### 9.2 prepare_meta.json

The authoritative metadata artifact.

It shall contain at minimum:

```text
schema_version
project
raw_video
trim
pre_crop
external_time
cage_detection
manual_crop
geometry
encoding
prepared_video
background
sync
mask
validation
software_environment
outputs
```

The `mask` section shall be present even when masking is disabled.

### 9.3 prepared_sync.npz

The frame-level synchronization artifact.

It shall contain:

```text
sync_schema_version

prepared_frame_idx
raw_decode_frame_idx

prepared_time_sec
raw_pts_time_sec
raw_pts_status

external_time_sec
external_time_status
external_time_source
external_time_variable_name
external_time_units

fps_header
fps_header_source
raw_fps_effective
raw_fps_effective_method
external_fps_effective
external_fps_effective_method

start_frame
end_frame_exclusive

raw_frame_count_opencv_readable
prepared_frame_count_opencv_reported
prepared_frame_count_opencv_readable
frame_count_used_for_sleap
```

The following invariant shall hold:

```text
len(prepared_frame_idx)
=
len(raw_decode_frame_idx)
=
len(prepared_time_sec)
=
len(raw_pts_time_sec)
=
len(external_time_sec)
=
prepared_frame_count_opencv_readable
```

### 9.4 cropped_background.png

A median background image estimated from the final prepared video.

It shall:

1. Be generated after final video preparation.
2. Have identical width and height to `prepared_video.mp4`.
3. Reflect any applied static prepared-coordinate mask.
4. Be suitable for future blob and background-based analysis.

### 9.5 settings_used.yaml

The settings artifact shall record the exact accepted preprocessing settings.

It shall include:

```text
trim settings
timing settings
pre-crop settings
cage-detector settings
canonical-resolution settings
mask settings
background settings
ffmpeg settings
OpenCV encoding settings
debug settings
```

The CropPlan and accepted crop geometry shall not be stored in `settings_used.yaml`. They belong in `prepare_meta.json`.

### 9.6 processing_log.txt

The processing log shall include:

```text
software version
dependency versions
input paths
probe results
TTL selection and validation
crop mode
crop acceptance
settings used
ffmpeg command
OpenCV encoding information
output paths
validation outcomes
warnings
errors
runtime
```

---

## 10. Raw Video Handling

### 10.1 Raw Video Selection

The user shall select a raw video file from disk.

The original source path shall be recorded in `prepare_meta.json`.

The raw video shall not be copied into the project directory in v1.

### 10.2 Raw Video Probe

The subsystem shall record, where available:

```text
source_path
width
height
codec
pixel_format
duration_sec
avg_frame_rate
r_frame_rate
time_base
frame_count_ffprobe
frame_count_opencv_reported
frame_count_opencv_readable
opencv_fps
raw_fps_effective
raw_fps_effective_method
pts_status
```

### 10.3 Raw Frame Count Policy

A full sequential OpenCV-readable raw-frame count shall be required when external timing is uploaded.

The selected timing vector must correspond to the original untrimmed raw-video frame space.

When no external timing is uploaded, a full raw sequential frame count is not required solely for timing validation.

### 10.4 Trim Validation

The subsystem shall validate:

```text
0 <= start_frame < raw_video_frame_count
```

When `end_frame` is specified:

```text
start_frame < end_frame <= raw_video_frame_count
```

`end_frame` shall be exclusive.

Selected frames are:

```text
[start_frame, end_frame)
```

If `end_frame` is null:

```text
[start_frame, raw_video_frame_count)
```

---

## 11. External MATLAB Timing Support

### 11.1 Supported Input

Version 1 shall support a generic MATLAB `.mat` workspace.

The reader shall support:

```text
MATLAB v7.2 and earlier
MATLAB v7.3 / HDF5-based files
```

### 11.2 Workspace Variable Display

The application shall display available numeric-vector candidates.

For each candidate, the application shall show:

```text
variable name
shape
class/type
length after squeeze
first value
last value
median difference
estimated FPS where meaningful
length-match result
```

### 11.3 User Selection

The user shall manually select a timing variable.

The system shall not automatically infer the intended variable from its name in v1.

### 11.4 Meaning and Units

The selected vector represents one timestamp/value per raw frame.

The user shall declare the vector units.

When the selected units are time units, values represent absolute acquisition times.

Supported units are:

```text
seconds
milliseconds
microseconds
nanoseconds
frames
unknown
```

### 11.5 Validation

A selected timing vector shall pass all of the following:

1. It is numeric.
2. It becomes one-dimensional after squeeze.
3. It contains finite values.
4. It is monotonically increasing.
5. Its length exactly equals the original untrimmed raw-video OpenCV-readable frame count.

Hard rule:

```text
len(selected_ttl_vector) == raw_frame_count_opencv_readable
```

If this rule fails, the application shall reject the timing selection.

### 11.6 External Time Conversion

For temporal units, values shall be converted to seconds.

```text
seconds       → multiply by 1
milliseconds  → multiply by 1e-3
microseconds  → multiply by 1e-6
nanoseconds   → multiply by 1e-9
```

For `frames` or `unknown`:

```text
external_time_sec = NaN array
external_time_status = "not_convertible_to_seconds"
```

The system shall not invent or infer timestamps for undefined units.

### 11.7 External Timing Mapping

For valid temporal timing vectors:

```text
external_time_sec = selected_timing_vector_in_seconds[raw_decode_frame_idx]
```

### 11.8 Timing File Storage

The timing `.mat` file shall not be copied into the project directory in v1.

Its original source path, selected variable, declared units, validation state, and timing statistics shall be recorded in `prepare_meta.json`.

---

## 12. FPS Header Selection

### 12.1 FPS Source Priority

The output FPS header shall be selected in this order:

```text
1. Valid external temporal timing vector
2. ffprobe avg_frame_rate
3. ffprobe r_frame_rate
4. OpenCV-reported FPS
5. Fail if no valid FPS can be resolved
```

### 12.2 External Timing FPS

When a valid temporal external timing vector is available:

```text
external_fps_effective = 1 / median(diff(external_time_sec))
```

This is the preferred source of output FPS because it represents measured camera timing.

### 12.3 Safe FPS Representation

After selecting the source FPS, the subsystem shall apply the validated safe-FPS representation logic from the legacy pipeline before passing FPS to OpenCV.

The safe-FPS logic shall preserve the effective camera-FPS domain while ensuring the FPS is representable by the selected OpenCV encoding pathway.

### 12.4 Metadata

The subsystem shall record:

```text
raw_fps_effective
raw_fps_effective_method
external_fps_effective
external_fps_effective_method
fps_header
fps_header_source
```

### 12.5 Prepared Playback Time

Prepared playback time shall be:

```text
prepared_time_sec = prepared_frame_idx / fps_header
```

This is a playback-domain fallback and shall not override valid external timing.

---

## 13. Detection Pre-Crop

### 13.1 Purpose

Detection pre-crop reduces irrelevant image regions before automatic cage detection.

### 13.2 Supported Modes

The subsystem shall support:

```text
none
vertical_keep_left
vertical_keep_right
horizontal_keep_upper
horizontal_keep_lower
manual_rectangle
```

### 13.3 Final Processing Effect

Detection pre-crop is part of final preprocessing.

Pixels outside the accepted pre-crop are excluded from final processing.

The pre-crop transform shall be incorporated into the final raw-to-prepared transform.

### 13.4 Metadata

The accepted pre-crop shall be recorded in `prepare_meta.json` and `settings_used.yaml`.

---

## 14. Cage Detection and Crop Approval

### 14.1 Automatic Cage Detection

Automatic cage detection shall produce a CropPlan.

The CropPlan shall include:

```text
mode
pre_crop_roi
quad_raw_tl_tr_br_bl
H_raw_to_prepared_3x3
H_prepared_to_raw_3x3
prepared_size_wh
rotated_90
fit_score
rim_density
accepted_by_user
```

### 14.2 Detector Settings

The user shall be able to modify detector settings and retry automatic cage detection.

Basic exposed settings shall include:

```text
sample_step
pad_px
threshold
roi_margin_px
perspective_interpolation
canonical-resolution enabled state
canonical width
canonical height
```

Advanced settings may include:

```text
pre-crop expansion percentage
dilate kernel size
erode kernel size
rim close kernel
minimum cage width fraction
minimum cage height fraction
minimum contour area
fit tolerance
```

### 14.3 Preview

The application shall display one cropped and rotated preview frame.

The preview exists for user review only and is not an official required artifact.

### 14.4 User Acceptance

Final preprocessing shall not proceed until the user accepts the crop.

---

## 15. Manual Crop

### 15.1 Manual Crop v1 Method

Manual crop shall use four selected cage corners in this order:

```text
top-left
top-right
bottom-right
bottom-left
```

### 15.2 Common CropPlan Interface

Both crop paths shall return the same internal type:

```text
automatic crop → CropPlan
manual crop → CropPlan
```

The video preparation engine shall consume CropPlan only and shall not branch based on crop origin.

### 15.3 Deferred Enhancement

Interactive rotation and ROI-alignment adjustment are deferred beyond v1.

---

## 16. Encoding and Video Preparation

### 16.1 Default Legacy-Compatible Recipe

The default v1 preparation recipe shall remain compatible with the existing validated pipeline:

```text
Stage A: ffmpeg
- apply trim
- apply detection pre-crop, when enabled
- apply cage crop and perspective transformation
- apply rotation, when required
- apply even-dimension crop
- apply uniform canonical scaling and padding, when enabled
- encode intermediate video

Stage B: OpenCV
- sequentially read the ffmpeg intermediate
- apply static prepared-coordinate mask when enabled
- write final prepared video
```

### 16.2 ffmpeg Settings

ffmpeg settings shall be configurable and recorded in `settings_used.yaml`.

The default legacy-compatible ffmpeg profile shall include:

```text
container: mp4
codec: libx264
pixel format: yuv420p
preset: veryfast
CRF: 18
B-frames: disabled
faststart: enabled
audio: disabled
```

### 16.3 OpenCV Settings

OpenCV final re-encoding settings shall be configurable and recorded in `settings_used.yaml`.

The default legacy-compatible OpenCV profile shall include:

```text
container: mp4
fourcc: mp4v
FPS header: resolved safe FPS
audio: absent
```

### 16.4 Allowed Codec Configuration

The interface may expose ffmpeg and OpenCV codec settings.

The implementation shall reject unsupported or unvalidated codec/container combinations before processing.

The default legacy-compatible codec profile shall remain the primary supported configuration in v1.

### 16.5 No Frame Resampling

Default timing configuration shall be:

```text
mode: decode_order_cfr
allow_frame_resampling: false
require_one_to_one_frame_mapping: true
require_constant_output_fps_header: true
```

### 16.6 Canonical Resolution

Default canonical resolution shall be:

```text
928 × 528
```

The user may alter it in advanced settings.

The implementation shall require:

```text
width > 0
height > 0
width is even
height is even
```

### 16.7 Geometry Preservation

When canonical resolution is enabled:

1. The native rectified image shall be scaled uniformly.
2. The scaled image shall fit inside the configured canonical resolution.
3. Remaining space shall be center-padded with black.
4. Independent horizontal and vertical stretching is prohibited.
5. Metadata shall record uniform scale, scaled dimensions, and padding offsets.

### 16.8 Intermediates

Intermediate files may be generated under `.internal`.

They shall be deleted after successful processing unless debug mode is enabled.

---

## 17. Synchronization Model

### 17.1 Frame Identity Layer

The primary frame mapping shall be:

```text
prepared_frame_idx → raw_decode_frame_idx
```

In default mode:

```text
raw_decode_frame_idx = start_frame + prepared_frame_idx
```

### 17.2 Raw PTS Layer

Raw PTS shall be stored when extractable.

Allowed raw PTS statuses include:

```text
valid
missing
non_monotonic
duplicated
partially_missing
untrusted
extraction_failed
```

If extraction fails:

```text
raw_pts_time_sec = NaN array
raw_pts_status = "extraction_failed"
```

Raw PTS extraction failure shall not independently fail preprocessing.

### 17.3 External Timing Layer

When valid temporal timing is supplied:

```text
prepared_frame_idx
→ raw_decode_frame_idx
→ external_time_sec
```

Where:

```text
external_time_sec = selected_timing_vector_in_seconds[raw_decode_frame_idx]
```

---

## 18. Validation Gates

A preprocessing run shall succeed only if every applicable hard validation gate passes.

### 18.1 Raw Video Validation

Hard failures:

1. Raw video cannot be opened.
2. A required raw OpenCV-readable frame count cannot be determined.
3. Start frame is invalid.
4. End frame is invalid.
5. Selected trim is empty.

### 18.2 External Timing Validation

When external timing is selected, hard failures include:

1. MATLAB file cannot be loaded.
2. Selected variable is non-numeric.
3. Selected variable is not one-dimensional after squeeze.
4. Selected variable contains non-finite values.
5. Selected variable is not monotonically increasing.
6. Selected vector length does not exactly equal raw untrimmed OpenCV-readable frame count.
7. Declared temporal unit is unsupported.

### 18.3 Crop Validation

Hard failures:

1. User has not accepted crop geometry.
2. CropPlan cannot be generated.
3. Required homography cannot be computed.
4. Prepared output dimensions are invalid.

### 18.4 Prepared Video Validation

Hard failures:

1. Prepared video cannot be opened by OpenCV.
2. Prepared OpenCV-reported frame count differs from sequentially readable frame count.
3. Prepared readable frame count differs from expected trimmed-frame count.
4. Prepared dimensions differ from recorded metadata.
5. Prepared dimensions are not even.
6. Prepared dimensions differ from configured canonical resolution when canonical resolution is enabled.

### 18.5 Sync Validation

Hard failures:

1. Frame-level arrays in `prepared_sync.npz` have inconsistent lengths.
2. Sync array length differs from prepared OpenCV-readable frame count.
3. Valid external timing array length differs from prepared OpenCV-readable frame count.
4. `raw_decode_frame_idx` differs from `start_frame + prepared_frame_idx` in default mode.

### 18.6 Background Validation

Hard failures:

1. Background image cannot be generated.
2. Background dimensions differ from prepared-video dimensions.

### 18.7 Metadata Validation

Hard failures:

1. `prepare_meta.json` fails schema validation.
2. Required output paths are absent.
3. Validation status is absent or not final.

---

## 19. Failure Behavior

When any hard validation gate fails:

1. The run shall be marked failed.
2. The project shall not be presented as successfully preprocessed.
3. A clear error shall be displayed to the user.
4. The error shall be written to `processing_log.txt`.
5. Partial artifacts shall not be treated as valid official outputs.
6. Intermediate files may remain under `.internal` for debugging or may be cleaned up according to configuration.

The system shall not apply a hidden workaround such as subtracting one frame from the prepared-video frame count.

---

## 20. Configuration

### 20.1 Default Configuration

The default configuration shall preserve the legacy-compatible recipe.

```yaml
schema_version: preprocess_config_v1

trim:
  start_frame: null
  end_frame: null
  end_frame_semantics: exclusive

timing:
  mode: decode_order_cfr
  allow_frame_resampling: false
  require_one_to_one_frame_mapping: true
  require_constant_output_fps_header: true

prepare:
  roi_margin_px: 40
  perspective_interpolation: cubic

  canonical_resolution:
    enabled: true
    width: 928
    height: 528
    flags: lanczos

pre_crop:
  enabled: false
  mode: none

cage_detect:
  sample_step: 500
  pad_px: 2
  threshold: 90

mask:
  enabled: false
  coordinate_space: prepared_video
  fill_value_bgr: [0, 0, 0]
  shapes: []

encoding:
  ffmpeg:
    container: mp4
    codec: libx264
    pixel_format: yuv420p
    preset: veryfast
    crf: 18
    bframes: 0
    faststart: true

  opencv:
    container: mp4
    fourcc: mp4v

background:
  method: median
  max_samples: 500
  sample_every_n: 50

debug:
  enabled: false
```

### 20.2 User-Editable Settings

The application shall permit adjustment of:

```text
start frame
end frame
pre-crop geometry
detector settings
manual crop points
canonical resolution
ffmpeg encoding settings
OpenCV encoding settings
background settings
mask settings when masking is implemented
debug mode
```

---

## 21. Architecture Requirements

### 21.1 Core and GUI Separation

Scientific processing logic shall reside in the core modules.

The GUI shall only:

```text
collect user inputs
display frames and previews
invoke core services
display progress
display validation results
display errors
```

The GUI shall not contain scientific transformation, timing, synchronization, or validation logic.

### 21.2 Shared Core Service

The same preprocess core engine shall be callable from:

```text
GUI application
CLI command
containerized environment
```

### 21.3 Legacy Code Reuse

Existing working code shall be retained as behavioral reference during migration.

Relevant legacy functionality shall be refactored into dedicated modules rather than copied into a single monolithic service.

Legacy code shall remain under:

```text
legacy/
```

until equivalent behavior has been verified.

---

## 22. Dependency Strategy

The subsystem shall support:

```text
A. Bundled desktop dependencies
B. Containerized reproducible processing
```

### 22.1 Desktop Application

The desktop application shall use bundled or managed ffmpeg and ffprobe binaries.

It shall not depend on an arbitrary system installation of ffmpeg.

### 22.2 Containerized Environment

A reproducible containerized environment shall pin at minimum:

```text
Python
OpenCV
NumPy
SciPy
h5py
ffmpeg
PyYAML
pydantic
```

---

## 23. Logging and Traceability

Every official output shall be traceable to:

```text
raw-video source path
timing-file source path, when provided
selected timing variable
timing units
start frame
end frame
pre-crop
crop geometry
encoding settings
software version
dependency versions
validation status
```

Frame-level downstream artifacts shall be able to map back to:

```text
prepared_frame_idx
raw_decode_frame_idx
external_time_sec, when valid
```

---

## 24. Final Design Laws

The Preprocess Subsystem shall obey the following laws:

1. `prepare_meta.json` is the single metadata authority.
2. `settings_used.yaml` records accepted processing settings.
3. Crop geometry and CropPlan belong in `prepare_meta.json`.
4. Decode order is the primary source of frame identity.
5. Raw PTS is diagnostic and may be invalid.
6. External timing is preferred when valid and convertible to seconds.
7. Timing-vector length must exactly equal original untrimmed raw OpenCV-readable frame count.
8. Prepared frame index must map deterministically to raw decode frame index.
9. No temporal resampling is allowed by default.
10. The output FPS header shall prefer valid external timing-derived FPS.
11. Prepared OpenCV-reported frame count must equal sequential OpenCV-readable frame count.
12. Prepared frame count must equal expected trimmed frame count.
13. OpenCV frame-count mismatch is a hard failure.
14. The user must accept crop geometry before final processing.
15. Automatic and manual crop must produce the same CropPlan interface.
16. Canonical resizing must preserve aspect ratio.
17. Detection pre-crop permanently defines the processed source region.
18. Intermediate files are not official artifacts.
19. `cropped_background.png` is the only required visual image artifact.
20. Scientific traceability is mandatory for every official output.
