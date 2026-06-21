# AI Coding Guide

## 1. Purpose

This guide defines mandatory rules for AI-assisted development in the `behavior_suite` repository.

Before modifying code, the coding assistant shall read:

```text
docs/preprocess_subsystem_spec_v1.md
docs/preprocess_implementation_plan_v1.md
docs/ai_coding_guide.md
```

When documents conflict, priority is:

```text
1. preprocess_subsystem_spec_v1.md
2. preprocess_implementation_plan_v1.md
3. ai_coding_guide.md
```

Do not change the specification or implementation plan silently to justify a code change.

---

## 2. Repository Structure

The repository root is `behavior_suite`.

Use this package structure:

```text
src/
├── project/
├── preprocess/
├── cli/
└── ui/
```

Do not create:

```text
src/behavior_suite/
```

Shared project lifecycle logic belongs in:

```text
src/project/
```

Preprocessing-specific logic belongs in:

```text
src/preprocess/
```

The GUI belongs in:

```text
src/ui/
```

The GUI must not contain scientific processing logic.

---

## 3. Required Working Method

For every task:

1. Read the relevant specification and implementation-plan sections.
2. Identify the affected module and its tests.
3. Make the smallest change that satisfies the requirement.
4. Add or update tests for changed core behavior.
5. Run the relevant tests.
6. Report changed files, tests run, and unresolved risks.

Do not refactor unrelated modules while implementing a focused task.

Do not replace working scientific behavior with a different design unless the specification is updated first.

---

## 4. Preprocess Scientific Invariants

The following rules are non-negotiable.

1. `prepare_meta.json` is the authoritative preprocessing metadata artifact.
2. `settings_used.yaml` stores accepted run settings.
3. CropPlan geometry belongs in `prepare_meta.json`, not `settings_used.yaml`.
4. Raw decode order is the primary source of source-frame identity.
5. Raw PTS is diagnostic only and may be invalid.
6. Valid external timing is preferred over raw PTS.
7. A selected external timing vector must exactly match the original untrimmed raw-video OpenCV-readable frame count.
8. Default processing must preserve a one-to-one mapping:

```text
prepared_frame_idx → raw_decode_frame_idx
```

9. Default processing must not resample, drop, duplicate, interpolate, or reorder frames.
10. A crop must be explicitly accepted before final preprocessing.
11. Auto crop and manual crop must return the same `CropPlan` interface.
12. Canonical sizing must preserve aspect ratio through uniform scaling and padding only.
13. Detection pre-crop permanently defines the processed source region.
14. The final prepared video must pass strict OpenCV frame validation.
15. No silent fallback may bypass a failed validation gate.

---

## 5. Frame Mapping Rules

For default processing:

```python
raw_decode_frame_idx = start_frame + prepared_frame_idx
```

The final prepared video frame count must equal:

```python
expected_frame_count = end_frame_exclusive - start_frame
```

or, when `end_frame_exclusive` is absent:

```python
expected_frame_count = raw_frame_count - start_frame
```

Never apply a hidden adjustment such as:

```python
usable_count = prepared_count - 1
```

If the prepared video contains a mismatch between reported and sequentially readable frames, preprocessing must fail.

---

## 6. Video Validation Rules

The final prepared video must always be validated with OpenCV.

Validation must:

1. Open the final prepared video.
2. Read width and height.
3. Read OpenCV-reported frame count.
4. Sequentially decode frames from beginning to end.
5. Count actual readable frames.
6. Require:

```text
opencv_reported_frame_count == opencv_readable_frame_count
```

7. Require:

```text
opencv_readable_frame_count == expected_trimmed_frame_count
```

8. Require final width and height to match expected output dimensions.
9. Require final width and height to be even.

Any failure is a hard failure.

---

## 7. External MATLAB Timing Rules

For generic `.mat` timing input:

1. Do not automatically infer the intended timing variable by variable name.
2. List numeric vectors for user selection.
3. Require the user to declare timing units.
4. Validate selected vector:

```text
numeric
one-dimensional after squeeze
finite
monotonically increasing
exact length match to raw untrimmed OpenCV-readable frame count
```

5. Supported units:

```text
seconds
milliseconds
microseconds
nanoseconds
frames
unknown
```

6. Convert temporal units to seconds.
7. Do not invent timing for `frames` or `unknown`.
8. For undefined timing units:

```text
external_time_sec = NaN array
external_time_status = "not_convertible_to_seconds"
```

9. For valid temporal timing:

```python
external_time_sec = selected_timing_vector_in_seconds[raw_decode_frame_idx]
```

---

## 8. FPS Header Rules

Resolve final FPS header in this order:

```text
1. Valid external temporal timing
2. ffprobe avg_frame_rate
3. ffprobe r_frame_rate
4. OpenCV-reported FPS
5. Fail
```

For valid external temporal timing:

```python
external_fps_effective = 1 / median(diff(external_time_sec))
```

Apply the validated legacy safe-FPS representation before writing through OpenCV.

Record all resolved FPS values and their source methods in metadata.

---

## 9. Encoding Rules

Preserve the legacy-compatible two-stage recipe unless the specification changes.

```text
Stage A: ffmpeg
- trim
- optional pre-crop
- crop/perspective transform
- rotation
- uniform canonical scaling and padding
- intermediate encode

Stage B: OpenCV
- sequential intermediate read
- optional static prepared-coordinate mask
- final prepared-video write
```

Default settings:

```text
ffmpeg container: mp4
ffmpeg codec: libx264
ffmpeg pixel format: yuv420p
ffmpeg preset: veryfast
ffmpeg CRF: 18
ffmpeg B-frames: disabled

OpenCV container: mp4
OpenCV fourcc: mp4v
```

Codec or container changes must be validated before processing. Do not expose arbitrary untested combinations as accepted settings.

---

## 10. Masking Rules

Static masking is architecturally reserved but is not mandatory in the first implementation phase.

When implemented:

1. Mask coordinates are in final prepared-video space.
2. Mask is applied after crop, rectification, rotation, scaling, and padding.
3. Mask is applied identically to every frame.
4. Masked pixels are black.
5. Background is estimated from the masked final prepared video.
6. Do not create a separate mask image artifact.
7. Store mask geometry in `prepare_meta.json`.

---

## 11. Artifact Rules

Official preprocess artifacts are:

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

Do not rename, remove, or add official artifacts without updating both preprocessing documents.

Intermediate files belong only in:

```text
ProjectName/preprocess/.internal/
ProjectName/preprocess/debug/
```

Delete intermediates after a successful run unless debug mode is enabled.

---

## 12. Error Handling Rules

Use explicit domain exceptions.

Recommended hierarchy:

```text
PreprocessError
├── VideoProbeError
├── ExternalTimingError
├── CropPlanError
├── CageDetectionError
├── VideoPreparationError
├── VideoValidationError
├── SyncArtifactError
└── MetadataError
```

Do not:

```text
catch broad exceptions and continue
replace missing values with guesses
silently lower validation standards
return partial output as successful
```

Errors must be logged and returned clearly to the CLI or GUI.

---

## 13. Testing Rules

Every core behavior change requires relevant tests.

Required test categories:

```text
unit tests
integration tests
real-data regression tests after production pipeline exists
```

At minimum, protect tests for:

```text
project creation/opening
config validation
OpenCV readable-frame counting
prepared-video validation
MAT file loading
TTL length mismatch rejection
timing-unit conversion
CropPlan validation
manual crop
automatic crop
ffmpeg filtergraph generation
OpenCV re-encoding
sync array generation
metadata schema validation
background dimensions
```

Do not begin GUI work before core engine and CLI tests pass.

---

## 14. Code Quality Rules

Use:

```text
typed Python
Pydantic for external config and metadata validation
pathlib.Path instead of raw path strings
structured logging
small focused modules
clear public interfaces
docstrings on public functions
explicit return types
```

Avoid:

```text
global state
hard-coded user paths
monolithic service functions
GUI-owned scientific logic
hidden filesystem side effects
untyped dictionaries for core data models
```

---

## 15. Legacy Code Rules

The files in `legacy/` are behavioral references during migration.

Use them to preserve proven logic, especially:

```text
ffmpeg transformation behavior
safe-FPS behavior
OpenCV final re-encoding behavior
cage detection behavior
background estimation behavior
```

Do not copy entire legacy scripts into new production modules unchanged.

Migrate functionality into focused modules, then compare behavior on real data before retiring legacy code.

Do not preserve legacy frame-count workarounds that conflict with the current specification.

---

## 16. Change Summary Format

After completing a coding task, provide a concise summary:

```text
Changed:
- file path: purpose

Tests:
- test command or test files run
- result

Validation:
- requirements checked
- known limitations or risks
```

If tests cannot be run, state exactly why.

---

## 17. Stop Conditions

Stop and request a specification update instead of guessing when a task would require:

```text
changing official artifacts
adding frame resampling
weakening frame validation
changing timing-vector validation
changing frame identity semantics
changing crop-coordinate semantics
changing default encoding recipe
moving scientific logic into the GUI
```

For ordinary implementation details that do not change these decisions, proceed using the approved documents.
