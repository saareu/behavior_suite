# Preprocess Subsystem Implementation Plan v1

> **Archive notice:** This historical document is retained for traceability. It is not the current source of truth. See `docs/subsystem_01/preprocessing.md`, `docs/subsystem_01/status_and_roadmap.md`, and `docs/subsystem_01/design/geometry_modes.md`.

## 1. Purpose

This document defines how to implement the Video Preprocess Subsystem described in `preprocess_subsystem_spec_v1.md`.

It is the implementation-oriented source of truth for human and AI-assisted development. All preprocessing code, tests, CLI behavior, and GUI behavior shall conform to this document and the approved subsystem specification.

This plan prioritizes:

```text
scientific traceability
deterministic frame identity
strict validation
reuse of validated legacy processing behavior
maintainable modular code
core-engine implementation before GUI development
```

---

## 2. Implementation Strategy

Implementation shall proceed in controlled milestones.

The first deliverable is not a GUI. It is a tested core engine and CLI capable of producing valid official artifacts from a raw video and an accepted CropPlan.

Implementation order:

```text
1. Repository and dependency setup
2. Shared project module
3. Configuration and typed data models
4. Video probing
5. MATLAB timing-file reader
6. CropPlan model and geometry utilities
7. Legacy cage detector migration
8. Manual four-corner crop generation
9. ffmpeg preparation stage
10. OpenCV final re-encode stage
11. Prepared-video validation
12. Synchronization artifact writer
13. Background generation
14. Metadata and settings writer
15. End-to-end preprocess service
16. CLI
17. Minimal GUI
```

The GUI shall only use the same core service exposed to the CLI.

---

## 3. Non-Negotiable Implementation Rules

The implementation shall obey all of the following:

1. `prepare_meta.json` is the authoritative metadata artifact.
2. `settings_used.yaml` records the accepted settings used for the run.
3. Crop geometry and CropPlan belong in `prepare_meta.json`, not `settings_used.yaml`.
4. Raw decode order is the primary source of source-frame identity.
5. Raw PTS is diagnostic and may be invalid.
6. External timing is preferred when valid and convertible to seconds.
7. A selected timing vector must exactly match the original untrimmed raw-video OpenCV-readable frame count.
8. Prepared frame index must map deterministically to raw decode frame index.
9. No temporal frame resampling is allowed by default.
10. The output FPS header shall prefer a valid external-timing-derived effective FPS.
11. Prepared OpenCV-reported frame count must equal sequential OpenCV-readable frame count.
12. Prepared readable frame count must equal expected trimmed frame count.
13. OpenCV frame-count mismatch is a hard failure.
14. The user must accept crop geometry before final processing.
15. Automatic and manual crop must produce the same internal `CropPlan` interface.
16. Canonical resizing must preserve aspect ratio.
17. Detection pre-crop permanently defines the processed source region.
18. Intermediate files are not official artifacts.
19. `cropped_background.png` is the only required visual image artifact.
20. No silent fallback may bypass a validation failure.

---

## 4. Repository Structure

The repository root is `behavior_suite`.

Use this structure:

```text
behavior_suite/
├── pyproject.toml
├── README.md
│
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
│   │   ├── models.py
│   │   ├── exceptions.py
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

Do not create a redundant nested package path such as:

```text
src/behavior_suite/
```

---

## 5. Dependencies

### 5.1 Required Core Dependencies

Use pinned versions through `pyproject.toml` and a lock file.

Required packages:

```text
numpy
opencv-python
scipy
h5py
pydantic
PyYAML
pytest
```

Recommended additional packages:

```text
rich
typer
platformdirs
pytest-cov
```

### 5.2 Video Dependencies

The application shall use managed or bundled binaries for:

```text
ffmpeg
ffprobe
```

The desktop application shall not depend on arbitrary user PATH configuration.

The implementation shall resolve ffmpeg and ffprobe through a dedicated utility or configuration layer.

### 5.3 GUI Dependencies

Do not add GUI dependencies until the core engine and CLI pass validation.

When GUI work begins, use:

```text
PySide6
```

---

## 6. Legacy Code Migration Policy

Existing working code is a behavioral reference and shall be reused.

The legacy scripts shall not be copied unchanged into one monolithic module.

Instead, migrate functions into focused modules while preserving their tested behavior.

Suggested migration map:

```text
Legacy functionality                             Destination module
---------------------------------------------------------------------------
probe_video_metadata()                           preprocess/video_probe.py
_ffps_effective_from_ffprobe()                  preprocess/video_probe.py
extract_raw_pts_time_for_trim()                 preprocess/video_probe.py
make_plan()                                      preprocess/cage_detection.py
CropPlan dataclass                              preprocess/crop_plan.py
run_initial_left_crop_ffmpeg()                  preprocess/video_prepare.py
_build_filtergraph()                            preprocess/video_prepare.py
run_prepare_ffmpeg()                            preprocess/video_prepare.py
reencode_prepared_opencv()                      preprocess/video_prepare.py
estimate_background_prepared()                  preprocess/background.py
_compute_raw_to_prepared_homography()           preprocess/crop_plan.py
_qc_prepared_video()                            preprocess/validation.py
sync NPZ construction                           preprocess/sync_writer.py
prepare metadata JSON construction              preprocess/metadata.py
```

The legacy scripts shall remain under `legacy/` until the refactored implementation has been shown to produce equivalent valid outputs on representative real videos.

### 6.1 Important Legacy Behavior to Preserve

The following behavior shall remain the default implementation profile:

```text
ffmpeg transformation stage
ffmpeg crop/perspective/rotation behavior
uniform canonical scaling and padding
OpenCV final video re-encode
safe-FPS representation logic
background estimation after prepared-video creation
```

### 6.2 Legacy Behavior Not to Preserve

The new implementation shall not preserve this workaround:

```text
usable_frame_count = prepared_frame_count - 1
```

A mismatch between OpenCV-reported and sequentially readable prepared frames is a hard failure.

---

## 7. Shared Project Module

The project module is shared across all future subsystems.

### 7.1 Responsibilities

`src/project/` shall provide:

```text
project creation
project opening
project root validation
project naming validation
subsystem artifact-directory paths
project manifest support
shared path utilities
```

### 7.2 Initial Interfaces

```python
from pathlib import Path

class Project:
    root_dir: Path
    name: str

class ProjectService:
    def create_project(self, parent_dir: Path, project_name: str) -> Project:
        ...

    def open_project(self, project_dir: Path) -> Project:
        ...

def get_preprocess_dir(project: Project) -> Path:
    ...
```

The preprocess service shall receive a `Project` instance and use the project module to resolve its artifact directory.

---

## 8. Core Data Models

Use Pydantic models for external configuration, artifact metadata, and validation results.

Use NumPy arrays inside computational code. Convert arrays to JSON-compatible lists only when serializing metadata.

### 8.1 PreprocessConfig

Implement in:

```text
src/preprocess/config.py
```

Required sections:

```text
trim
timing
prepare
pre_crop
cage_detect
mask
encoding
background
debug
```

### 8.2 VideoProbeResult

Implement in:

```text
src/preprocess/models.py
```

Required fields:

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

### 8.3 ExternalTimeSelection

Required fields:

```text
provided
source_path
selected_variable
declared_units
raw_vector_length
raw_video_frame_count_opencv_readable
is_numeric
is_one_dimensional
is_finite
is_monotonic_increasing
median_difference
estimated_fps
validation_status
```

### 8.4 CropPlan

Implement in:

```text
src/preprocess/crop_plan.py
```

Required fields:

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

### 8.5 PreprocessRequest

Required fields:

```text
project
raw_video_path
config
start_frame
end_frame_exclusive
accepted_crop_plan
external_time_selection
external_timing_vector
```

### 8.6 PreprocessResult

Required fields:

```text
success
outputs
validation_result
warnings
errors
```

### 8.7 PreprocessOutputs

Required fields:

```text
preprocess_dir
prepared_video_path
prepare_meta_path
prepared_sync_path
cropped_background_path
settings_used_path
processing_log_path
```

---

## 9. Default Configuration

Create:

```text
configs/preprocess_default.yaml
```

Use this baseline:

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

The implementation shall reject unsupported or internally inconsistent configuration combinations before processing starts.

---

## 10. Milestone Plan

## Milestone 1 — Repository Skeleton, Project Module, and Config

Implement:

```text
src/project/models.py
src/project/paths.py
src/project/service.py
src/project/validation.py

src/preprocess/config.py
src/preprocess/models.py
src/preprocess/exceptions.py

configs/preprocess_default.yaml
```

Acceptance criteria:

```text
Project can be created.
Existing project can be opened.
ProjectName/preprocess/ path can be resolved.
Default config loads.
Invalid config fails with clear validation errors.
```

Tests:

```text
tests/project/test_service.py
tests/project/test_paths.py
tests/preprocess/test_config.py
```

---

## Milestone 2 — Video Probe

Implement:

```text
src/preprocess/video_probe.py
```

Required functions:

```python
from pathlib import Path

def probe_video_ffprobe(path: Path) -> dict:
    ...

def get_opencv_reported_frame_count(path: Path) -> int:
    ...

def count_opencv_readable_frames(path: Path) -> int:
    ...

def probe_video(path: Path, require_sequential_count: bool) -> VideoProbeResult:
    ...
```

Requirements:

```text
ffprobe metadata shall be recorded when available.
OpenCV sequential readable count is mandatory only when requested.
OpenCV frame count is mandatory for final prepared-video validation.
Raw full sequential counting is mandatory when external timing is uploaded.
```

Acceptance criteria:

```text
Clean video probe succeeds.
Missing ffprobe data does not crash a readable video probe.
Unreadable video fails clearly.
Sequential count returns actual decoded frame count.
```

Tests:

```text
tests/preprocess/test_video_probe.py
```

---

## Milestone 3 — MATLAB Timing Reader

Implement:

```text
src/preprocess/mat_sync_reader.py
```

Required interfaces:

```python
from pathlib import Path

def load_mat_workspace(path: Path) -> MatWorkspace:
    ...

def list_numeric_vectors(workspace: MatWorkspace) -> list[MatVectorCandidate]:
    ...

def get_numeric_vector(
    workspace: MatWorkspace,
    variable_name: str,
) -> np.ndarray:
    ...

def validate_external_timing_vector(
    vector: np.ndarray,
    raw_frame_count_opencv_readable: int,
    declared_units: str,
) -> ExternalTimeSelection:
    ...

def convert_timing_vector_to_seconds(
    vector: np.ndarray,
    declared_units: str,
) -> np.ndarray | None:
    ...
```

Requirements:

```text
Support MAT v7.2 and earlier through scipy.
Support MAT v7.3/HDF5 through h5py.
List numeric vector candidates without automatic name selection.
Require exact selected-vector length match to raw readable frame count.
Reject non-finite or non-monotonic vectors.
```

Accepted units:

```text
seconds
milliseconds
microseconds
nanoseconds
frames
unknown
```

Rules:

```text
Time units are converted to seconds.
frames and unknown do not produce external_time_sec.
frames and unknown may be retained as selected metadata but cannot provide timing-derived FPS.
```

Acceptance criteria:

```text
MAT workspace can be loaded.
Numeric vectors are listed.
Correct vector validates.
Length mismatch fails.
Non-monotonic vector fails.
Time-unit conversion is correct.
```

Tests:

```text
tests/preprocess/test_mat_sync_reader.py
```

---

## Milestone 4 — CropPlan and Geometry Utilities

Implement:

```text
src/preprocess/crop_plan.py
src/preprocess/pre_crop.py
```

Required responsibilities:

```text
represent CropPlan
validate 3×3 homography matrices
serialize CropPlan to JSON-compatible metadata
validate quad point ordering
compute inverse transform
apply pre-crop geometry
validate even dimensions
compute uniform scale and padding
```

Acceptance criteria:

```text
Valid CropPlan serializes and deserializes.
Invalid matrix dimensions fail.
Homography inverse is valid.
Uniform scaling preserves aspect ratio.
Padding is recorded correctly.
```

Tests:

```text
tests/preprocess/test_crop_plan.py
tests/preprocess/test_pre_crop.py
```

---

## Milestone 5 — Automatic Cage Detection Migration

Implement:

```text
src/preprocess/cage_detection.py
```

Responsibilities:

```text
migrate existing cage detector behavior
move configurable values out of hard-coded code
produce CropPlan
support optional pre-crop
report fit score and rim density
```

Requirements:

```text
Existing detector algorithm remains behavioral reference.
Detector configuration comes from PreprocessConfig.
Detection failure raises explicit domain exception.
```

Acceptance criteria:

```text
Existing successful detector case creates CropPlan.
Detector configuration changes affect run behavior.
Failure is clear and does not create invalid CropPlan.
```

Tests:

```text
tests/preprocess/test_cage_detection.py
```

---

## Milestone 6 — Manual Four-Corner Crop

Implement:

```text
src/preprocess/manual_crop.py
```

Required function:

```python
def make_manual_crop_plan(
    raw_frame_shape: tuple[int, int],
    points_tl_tr_br_bl: np.ndarray,
    pre_crop_roi: tuple[int, int, int, int] | None,
    canonical_resolution: CanonicalResolutionConfig,
) -> CropPlan:
    ...
```

Requirements:

```text
Input point order is top-left, top-right, bottom-right, bottom-left.
Manual crop returns same CropPlan model used by automatic crop.
Manual crop supports canonical uniform scaling and padding.
```

Acceptance criteria:

```text
Four valid corners create valid CropPlan.
Invalid corner count fails.
Self-intersecting or degenerate quads fail.
Manual and auto CropPlans can use the same preparation service.
```

Tests:

```text
tests/preprocess/test_manual_crop.py
```

---

## Milestone 7 — ffmpeg Preparation Stage

Implement:

```text
src/preprocess/video_prepare.py
```

Required responsibilities:

```text
build deterministic ffmpeg filtergraph
apply trim
apply optional pre-crop
apply ROI crop
apply perspective correction
apply rotation
force even dimensions
apply uniform canonical scaling and black padding
write intermediate video under .internal
```

Requirements:

```text
Default ffmpeg behavior must match legacy pipeline.
No temporal resampling filters may be introduced in default mode.
Audio must be removed.
All ffmpeg commands must be logged.
```

Required key functions:

```python
def resolve_ffmpeg_binary() -> Path:
    ...

def build_prepare_filtergraph(...) -> tuple[str, FiltergraphMetadata]:
    ...

def run_ffmpeg_prepare(...) -> IntermediatePrepareResult:
    ...
```

Acceptance criteria:

```text
Intermediate video is generated.
Expected dimensions are obtained.
Transform metadata is available.
ffmpeg command is recorded.
```

Tests:

```text
tests/preprocess/test_video_prepare_filtergraph.py
tests/integration/test_ffmpeg_prepare.py
```

---

## Milestone 8 — OpenCV Final Re-encode and Static Mask Hook

Implement in:

```text
src/preprocess/video_prepare.py
src/preprocess/masking.py
```

Required responsibilities:

```text
sequentially read ffmpeg intermediate
write final prepared video using configured OpenCV fourcc
use resolved safe FPS header
apply static prepared-coordinate mask when enabled
write every frame exactly once
count frames written
```

Default v1 behavior:

```text
masking disabled
OpenCV MP4 output
fourcc mp4v
```

Masking architecture requirements:

```text
Mask is defined in prepared-video coordinates.
Mask is applied after crop and scaling.
Mask is applied to every frame before output.
Mask fill is black.
Mask is not a required v1 implemented feature.
```

Required key function:

```python
def reencode_prepared_opencv(
    intermediate_path: Path,
    final_path: Path,
    fps_header: float,
    encoding_config: OpenCVEncodingConfig,
    mask: StaticMask | None,
) -> int:
    ...
```

Acceptance criteria:

```text
Output can be created.
Writer-reported frame count is recorded.
Every intermediate frame is written exactly once.
No audio is produced.
```

Tests:

```text
tests/preprocess/test_opencv_reencode.py
tests/preprocess/test_masking.py
```

---

## Milestone 9 — Prepared Video Validation

Implement:

```text
src/preprocess/validation.py
```

Required function:

```python
def validate_prepared_video(
    prepared_video_path: Path,
    expected_frame_count: int,
    expected_size_wh: tuple[int, int],
) -> PreparedVideoValidationResult:
    ...
```

Required validation steps:

```text
open video with OpenCV
read width and height
read OpenCV-reported frame count
sequentially read every frame
count actual readable frames
verify reported count equals readable count
verify readable count equals expected trimmed count
verify dimensions match expected dimensions
verify dimensions are even
```

Hard failure rule:

```text
If OpenCV reports one more frame than it can sequentially read,
the run fails.
```

Acceptance criteria:

```text
Valid video passes.
Reported/readable mismatch fails.
Wrong dimensions fail.
Wrong expected count fails.
```

Tests:

```text
tests/preprocess/test_validation.py
tests/integration/test_prepared_video_validation.py
```

---

## Milestone 10 — Synchronization Artifact Writer

Implement:

```text
src/preprocess/sync_writer.py
```

Required function:

```python
def write_prepared_sync_npz(
    path: Path,
    prepared_frame_count_opencv_readable: int,
    prepared_frame_count_opencv_reported: int,
    raw_frame_count_opencv_readable: int | None,
    start_frame: int,
    end_frame_exclusive: int | None,
    fps_header: float,
    fps_header_source: str,
    raw_fps_effective: float | None,
    raw_fps_effective_method: str | None,
    external_fps_effective: float | None,
    external_fps_effective_method: str | None,
    raw_pts_time_sec: np.ndarray | None,
    raw_pts_status: str,
    external_timing_seconds: np.ndarray | None,
    external_time_status: str,
    external_time_source: str | None,
    external_time_variable_name: str | None,
    external_time_units: str | None,
) -> None:
    ...
```

Required generated arrays:

```text
prepared_frame_idx
raw_decode_frame_idx
prepared_time_sec
raw_pts_time_sec
external_time_sec
```

Core mapping:

```python
prepared_frame_idx = np.arange(prepared_frame_count_opencv_readable)
raw_decode_frame_idx = start_frame + prepared_frame_idx
prepared_time_sec = prepared_frame_idx / fps_header
```

External timing rule:

```python
external_time_sec = external_timing_seconds[raw_decode_frame_idx]
```

when valid temporal external timing exists.

No external timing rule:

```python
external_time_sec = np.full(prepared_frame_count_opencv_readable, np.nan)
```

Acceptance criteria:

```text
All frame-level arrays have equal length.
Raw decode mapping is correct.
Trimmed external time indexing is correct.
No-TTL case stores NaN external_time_sec.
```

Tests:

```text
tests/preprocess/test_sync_writer.py
```

---

## Milestone 11 — Background Generation

Implement:

```text
src/preprocess/background.py
```

Required function:

```python
def estimate_prepared_background(
    prepared_video_path: Path,
    sample_every_n: int,
    max_samples: int,
    method: str,
) -> np.ndarray:
    ...
```

Requirements:

```text
Input is final prepared video.
Default method is grayscale median.
Background is generated after any enabled static masking.
Output dimensions match prepared video dimensions.
```

Acceptance criteria:

```text
Background PNG is created.
Dimensions match prepared video.
No frame is used from an intermediate video.
```

Tests:

```text
tests/preprocess/test_background.py
```

---

## Milestone 12 — Metadata and Settings Writers

Implement:

```text
src/preprocess/metadata.py
```

Required functions:

```python
def write_settings_used_yaml(
    path: Path,
    config: PreprocessConfig,
) -> None:
    ...

def write_prepare_meta_json(
    path: Path,
    ...
) -> None:
    ...
```

### 12.1 settings_used.yaml

Must contain accepted settings only:

```text
trim settings
timing settings
pre-crop settings
cage-detector settings
canonical-resolution settings
mask settings
background settings
ffmpeg settings
OpenCV settings
debug settings
```

Do not store CropPlan geometry here.

### 12.2 prepare_meta.json

Must record:

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

Acceptance criteria:

```text
YAML is readable.
JSON is schema-valid.
CropPlan geometry is stored in metadata.
Settings and metadata do not duplicate roles unnecessarily.
```

Tests:

```text
tests/preprocess/test_metadata.py
```

---

## Milestone 13 — Preprocess Service

Implement:

```text
src/preprocess/service.py
```

Main API:

```python
class PreprocessService:
    def run(self, request: PreprocessRequest) -> PreprocessResult:
        ...
```

Execution flow:

```text
1. Resolve project preprocess directory.
2. Initialize structured log.
3. Load and validate config.
4. Probe raw video.
5. Validate external timing when selected.
6. Validate trim settings.
7. Receive accepted CropPlan.
8. Resolve FPS header.
9. Run ffmpeg transformation stage.
10. Run OpenCV final re-encode stage.
11. Validate final prepared video.
12. Build prepared_sync.npz.
13. Generate cropped_background.png.
14. Write settings_used.yaml.
15. Write prepare_meta.json.
16. Delete intermediates when successful and debug is disabled.
17. Return PreprocessResult.
```

Failure flow:

```text
1. Capture exception.
2. Write failure details to processing_log.txt.
3. Do not mark output as successful.
4. Do not write valid final metadata state.
5. Keep or clean intermediates according to debug configuration.
6. Return failed PreprocessResult.
```

Acceptance criteria:

```text
A successful run writes all official artifacts.
A failed run does not appear successful.
All hard validation failures terminate processing.
```

Tests:

```text
tests/integration/test_preprocess_service.py
```

---

## Milestone 14 — CLI

Implement:

```text
src/cli/preprocess.py
```

Recommended CLI framework:

```text
Typer
```

Initial command:

```bash
behavior-suite preprocess \
  --project-dir /path/to/ProjectName \
  --raw-video /path/to/raw_video.avi \
  --config configs/preprocess_default.yaml \
  --start-frame 34038 \
  --end-frame 120000
```

The first CLI may support:

```text
automatic cage detection
saved CropPlan JSON input
explicit crop configuration
external MAT file selection by variable name
timing units declaration
```

Example:

```bash
behavior-suite preprocess \
  --project-dir /path/to/ProjectName \
  --raw-video /path/to/video.avi \
  --mat-timing /path/to/sync.mat \
  --mat-variable frame_times_up \
  --timing-units seconds \
  --start-frame 34038 \
  --config configs/preprocess_default.yaml
```

Acceptance criteria:

```text
CLI can create final artifacts.
CLI returns nonzero exit status on failure.
CLI logs clear validation failures.
```

---

## Milestone 15 — Minimal GUI

Begin GUI work only after the CLI and core engine are validated on real data.

The minimal GUI shall contain:

```text
Project open/create page
Raw-video selection page
Raw-video information panel
MAT timing-file selection page
MAT variable selection panel
Frame trim page
Detection pre-crop page
Cage detection and retry page
Manual four-corner crop page
Crop acceptance page
Advanced settings page
Run and validation page
```

The GUI shall not perform scientific processing directly.

It shall construct a `PreprocessRequest`, invoke `PreprocessService`, and render the result.

---

## 11. Validation-First Development

Each milestone shall include tests before proceeding.

Do not begin GUI development until these are implemented and tested:

```text
project creation/opening
config validation
raw video probe
MAT reader
TTL length-match rejection
CropPlan serialization
automatic crop
manual crop
ffmpeg prepare stage
OpenCV final re-encode
prepared video reported/readable validation
sync artifact writing
metadata writing
background generation
end-to-end CLI run
```

---

## 12. Test Strategy

### 12.1 Unit Tests

Unit tests shall cover:

```text
config validation
project paths
video probe parsing
OpenCV frame counting
MAT loading
MAT vector validation
unit conversion
CropPlan validation
pre-crop geometry
manual crop geometry
filtergraph construction
safe-FPS selection
sync array construction
metadata schema validation
background dimensions
```

### 12.2 Integration Tests

Integration tests shall cover:

```text
clean CFR video
video with unavailable or broken PTS
video using external timing
TTL vector length mismatch
video requiring vertical pre-crop
video requiring horizontal pre-crop
automatic cage-detection success
manual-crop fallback
prepared reported/readable frame mismatch
canonical scaling with padding
full preprocess service run
```

### 12.3 Real Data Regression Testing

After production implementation is established, create a separate test-data manifest and regression dataset.

This is deferred until the core production pathway exists.

---

## 13. Coding Standards

All core implementation shall use:

```text
typed Python
explicit public interfaces
Pydantic validation for external data
structured logging
custom domain exceptions
no silent fallback
no hidden path assumptions
unit tests for core logic
integration tests for video behavior
```

Every public function shall provide:

```text
type annotations
docstring
explicit expected behavior
explicit error behavior
```

Recommended exception hierarchy:

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

---

## 14. AI-Assisted Development Rules

Any AI coding assistant modifying this repository shall follow these rules:

1. Read `docs/preprocess_subsystem_spec_v1.md` before changing preprocess code.
2. Read this implementation plan before changing module boundaries or milestones.
3. Do not rename official artifacts without updating both documents.
4. Do not add temporal resampling unless the specification is explicitly revised.
5. Do not weaken strict TTL length validation.
6. Do not use raw PTS as the primary frame identity source.
7. Do not add silent frame-count workarounds.
8. Do not continue after a hard validation failure.
9. Do not place scientific transformation or validation logic in the GUI.
10. Do not expose internal intermediate files as official artifacts.
11. Add or update tests whenever core behavior changes.
12. Preserve the legacy default recipe until a replacement is formally validated.

---

## 15. First Development Sprint

The first implementation sprint shall create a working foundation without GUI work.

### 15.1 Scope

Implement:

```text
src/project/models.py
src/project/paths.py
src/project/service.py
src/project/validation.py

src/preprocess/config.py
src/preprocess/models.py
src/preprocess/exceptions.py
src/preprocess/video_probe.py
src/preprocess/mat_sync_reader.py

configs/preprocess_default.yaml

tests/project/test_service.py
tests/project/test_paths.py
tests/preprocess/test_config.py
tests/preprocess/test_video_probe.py
tests/preprocess/test_mat_sync_reader.py
```

### 15.2 First Sprint Deliverable

The first working command or script shall be able to:

```text
1. Create or open a project.
2. Load and validate the default preprocess config.
3. Probe a raw video.
4. Display raw-video metadata.
5. Optionally load a MATLAB .mat file.
6. List numeric vector variables from the MAT workspace.
7. Select a timing vector by name.
8. Validate exact timing-vector length against raw OpenCV-readable frame count.
9. Convert valid temporal timing vectors to seconds.
10. Report clear failures.
```

### 15.3 Explicitly Not Included in First Sprint

```text
GUI
automatic cage detection
manual crop
ffmpeg processing
OpenCV final re-encode
sync artifact writing
background generation
metadata writing
```

---

## 16. Completion Criteria for Implementation Phase 1

The preprocess implementation phase is complete when:

```text
1. Core service can process a real video end to end.
2. Default legacy-compatible recipe is used.
3. Final prepared video passes strict OpenCV frame validation.
4. prepared_sync.npz has correct deterministic frame mapping.
5. Valid external timing maps correctly after trim.
6. Invalid external timing is rejected.
7. CropPlan can originate from auto or manual crop.
8. cropped_background.png matches final prepared dimensions.
9. settings_used.yaml records accepted settings.
10. prepare_meta.json validates and records all required provenance.
11. CLI runs successfully.
12. All critical tests pass.
```
