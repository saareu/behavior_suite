# Subsystem 02 — SLEAP-NN v0.3 Inference Specification

## 1. Purpose

Subsystem 02 performs SLEAP-NN v0.3 pose inference on validated prepared videos produced by Subsystem 01.

Its purpose is to transform a prepared video into:

* a native SLEAP pose artifact;
* a time-aligned numeric pose table;
* an optional visual overlay;
* reproducible metadata, settings, logs, and validation records.

Subsystem 02 shall support two-mouse bottom-up and top-down SLEAP inference profiles.

Bottom-up inference is the initial default model family.

Subsystem 02 is designed to preserve useful raw SLEAP output for later quality control and identity-resolution work. It shall not attempt to solve final animal identity or behavior classification.

---

## 2. Scope

### 2.1 Included responsibilities

Subsystem 02 shall provide:

* validation of official Subsystem 01 input artifacts;
* registered SLEAP model selection;
* model-family-specific inference profiles;
* SLEAP-NN v0.3 inference through `sleap-nn predict`;
* SLEAP internal provisional tracking;
* native `.slp` output generation;
* numeric pose export to `pose.parquet`;
* timing attachment from Subsystem 01;
* optional SLEAP-native overlay generation;
* execution dispatch and output collection;
* runtime, model, settings, and command provenance;
* artifact validation;
* structured success, warning, and failure reporting.

### 2.2 Excluded responsibilities

Subsystem 02 shall not perform:

* custom candidate scoring;
* custom candidate selection;
* custom track merging;
* custom track splitting;
* final biological identity assignment;
* forced `mouse_1` / `mouse_2` slots;
* manual proofreading;
* interpolation of missing nodes;
* behavioral event detection;
* feature extraction;
* downstream analysis.

Those tasks belong to later subsystems, especially pose tracking and quality control.

---

## 3. Subsystem 01 Protection

### 3.1 Non-regression law

Subsystem 02 implementation shall not alter, replace, or change the behavior of the completed Subsystem 01 preprocessing workflow.

Subsystem 01 remains a protected, independently functioning subsystem.

Subsystem 02 shall not modify:

```text
src/preprocess/
src/ui/controllers/
src/ui/pages/
src/ui/widgets/
src/ui/main_window.py
src/ui/app.py
src/cli/preprocess.py
scripts/launch_windows_gui.bat
scripts/install_windows_gui.bat
```

Subsystem 02 shall not change:

* preprocessing algorithms;
* preprocessing artifact names;
* preprocessing artifact schemas;
* preprocessing GUI controls;
* preprocessing navigation;
* preprocessing startup behavior;
* existing preprocessing entry points;
* existing preprocessing test behavior.

### 3.2 Dependency direction

Dependency flow shall remain one-way:

```text
Subsystem 01 artifacts
        ↓
Subsystem 02 validation and inference
        ↓
Subsystem 02 pose artifacts
```

Subsystem 01 shall not import, invoke, inspect, or depend on Subsystem 02.

Subsystem 02 may read the official Subsystem 01 artifacts in read-only mode.

### 3.3 Subsystem 02 write boundary

Subsystem 02 shall not write into `preprocess/`.

It shall write only under:

```text
pose_inference/<model-id>__<timestamp>/
```

### 3.4 Runtime isolation

The existing Subsystem 01 GUI environment and the SLEAP-NN v0.3 environment shall remain separable.

```text
behavior_suite_gui
    Existing GUI environment for Subsystem 01.

behavior_suite_sleap_v030
    Dedicated SLEAP-NN v0.3 inference environment.
```

Launching Subsystem 01 shall not require:

* SLEAP-NN;
* PyTorch CUDA;
* SLEAP-NN model files;
* SLEAP-NN-specific runtime dependencies.

Subsystem 02 shall run inference through a controlled process boundary.

---

## 4. Core Design Laws

1. Subsystem 02 shall consume only validated official Subsystem 01 outputs.

2. Subsystem 02 shall target SLEAP-NN v0.3 only.

3. Subsystem 02 shall use `sleap-nn predict` as its inference interface.

4. `pose.slp` shall remain the native authoritative SLEAP result.

5. Subsystem 02 shall preserve every SLEAP instance retained by the selected SLEAP inference settings.

6. Subsystem 02 shall not apply wrapper-side candidate deletion after inference.

7. SLEAP track labels shall be treated as provisional tracking labels, not final animal identities.

8. Subsystem 02 shall not assign custom labels such as `mouse_1`, `mouse_2`, `animal_slot`, `selected`, or `final_identity`.

9. Timing shall be copied from Subsystem 01. Subsystem 02 shall not estimate, replace, or independently reconstruct timing.

10. Missing nodes shall remain missing. Subsystem 02 shall not interpolate them.

11. A run shall be marked successful only after the required outputs pass validation.

12. An overlay failure shall not invalidate valid pose inference and numeric export. It shall create a `success_with_warning` result.

---

## 5. Input Contract

Subsystem 02 requires a validated Subsystem 01 output directory containing:

```text
ProjectName/
└── preprocess/
    ├── prepared_video.mp4
    ├── prepare_meta.json
    ├── prepared_sync.npz
    └── settings_used.yaml
```

### 5.1 Required input information

Subsystem 02 shall read:

```text
prepared video path
prepared frame count
prepared width
prepared height
prepared FPS
prepared_frame_idx
raw_decode_frame_idx
prepared_time_sec
raw_pts_time_sec
external_time_sec
external timing status
preprocessing provenance
```

### 5.2 Frame identity

The authoritative frame mapping remains:

```text
pose_frame_idx = prepared_frame_idx
```

Subsystem 02 shall not create a separate frame axis.

### 5.3 Timing contract

All exported timing columns shall originate from:

```text
preprocess/prepared_sync.npz
```

The join key shall be:

```text
prepared_frame_idx
```

Subsystem 02 shall not derive time from inference speed, overlay timing, SLEAP timestamps, or independently estimated FPS.

---

## 6. Runtime Contract

```yaml
runtime_profile:
  profile_id: sleapnn_v030_cuda130_v1

  sleap_nn_version: "0.3.0"
  sleap_io_version: ">=0.8.0,<0.9.0"

  inference_command: "sleap-nn predict"
  default_device: "cuda"
```

### 6.1 Supported devices

```text
cuda
cuda:0
cpu
auto
```

The default device shall be:

```yaml
device: cuda
```

### 6.2 CUDA failure behavior

If CUDA is explicitly requested but unavailable, Subsystem 02 shall not silently fall back to CPU.

The request shall terminate with:

```yaml
status: dispatch_failed
reason: requested_cuda_unavailable
```

The user may then explicitly select another valid device or another execution provider.

---

## 7. Execution Lifecycle

Subsystem 02 owns the full inference request lifecycle:

```text
create inference request
↓
validate Subsystem 01 artifacts
↓
resolve model and profile
↓
resolve settings and user overrides
↓
validate runtime capabilities
↓
select execution provider
↓
dispatch inference
↓
obtain native output
↓
export pose.parquet
↓
generate overlay when enabled
↓
validate outputs
↓
record metadata and final status
```

### 7.1 Final statuses

```text
success
success_with_warning
dispatch_failed
inference_failed
export_failed
validation_failed
cancelled
```

Example:

```text
pose.slp valid
pose.parquet valid
overlay generation fails
↓
status = success_with_warning
overlay_status = failed
```

---

## 8. Execution Providers

Subsystem 02 shall use an execution-provider abstraction.

```text
ExecutionProvider
├── LocalCudaProvider
├── LocalCpuProvider
└── AthenaProvider
```

Each provider shall return:

```text
execution status
stdout log
stderr log
runtime metadata
native output location
failure reason, when applicable
```

Subsystem 02 remains responsible for scientific and artifact validation regardless of where inference runs.

---

## 9. Model Registry

Model files shall be organized under:

```text
models/
└── SLEAP/
    ├── bottomup/
    │   └── <model-id>/
    │       ├── best.ckpt
    │       ├── training_config.yaml
    │       ├── model_manifest.yaml
    │       └── inference_defaults.yaml
    │
    └── topdown/
        └── <model-id>/
            ├── centroid/
            │   ├── best.ckpt
            │   └── training_config.yaml
            │
            ├── centered_instance/
            │   ├── best.ckpt
            │   └── training_config.yaml
            │
            ├── model_manifest.yaml
            └── inference_defaults.yaml
```

### 9.1 Required model-manifest fields

```yaml
model_id:
model_version:
model_family:

expected_animal_count: 2

skeleton_id:
node_names:
edges:

camera_view:
implant_context:

training_data_description:
training_video_geometry:
training_preprocessing:

required_runtime_profile:
default_inference_profile:
```

The following fields are mandatory:

```text
training_data_description
training_video_geometry
training_preprocessing
```

They may initially be concise, but they shall not remain empty.

---

## 10. Skeleton Contract

The current mixed implanted/non-implanted two-mouse skeleton shall be identified as:

```text
2mice_upview_implanted_nonimplanted_5node_v1
```

### 10.1 Node names

```text
nose
neck
spine_base
tail_base
headstage
```

### 10.2 Skeleton edges

```text
nose → neck
neck → spine_base
spine_base → tail_base
neck → headstage
```

### 10.3 Mixed implant interpretation

The skeleton contains five possible nodes.

```text
Implanted mouse
    May have all five nodes.

Non-implanted mouse
    May legitimately have no headstage node.
```

A missing headstage prediction for the non-implanted mouse shall not be treated as a model error or skeleton mismatch.

---

## 11. Inference Profile System

Subsystem 02 shall initially support:

```text
two-mouse bottom-up inference
two-mouse top-down inference
```

Bottom-up shall be the initial default model family.

Each inference profile shall define:

```text
model-family defaults
basic user-editable settings
advanced user-editable settings
runtime requirements
parameter validation rules
model-family applicability
resolved execution settings
```

The user interface shall provide:

```text
Basic settings
    Focused settings appropriate for ordinary inference.

Advanced settings
    Explicit advanced overrides with validation and warnings.
```

The UI shall not expose raw shell commands for editing.

---

## 12. Capability Registry

Subsystem 02 shall use a capability registry to validate the exact installed SLEAP-NN v0.3 interface.

At runtime validation, the wrapper shall capture:

```text
sleap-nn --version
sleap-nn predict --help
Python executable path
sleap-nn package path
sleap-io package path
```

The capability record shall include:

```text
supported CLI option names
option types
option defaults
allowed enum values
model-family applicability
command serialization rules
```

The wrapper shall reject:

```text
unknown parameter names
unsupported parameters
incorrect parameter types
invalid enum values
parameters incompatible with selected model family
parameters incompatible with selected execution provider
```

The wrapper shall not accept arbitrary shell fragments as additional arguments.

---

## 13. Request and Settings Contracts

### 13.1 Inference request

The user-facing request shall contain only explicit user choices and overrides.

Example:

```yaml
schema_version: pose_inference_request_v1

project_root: Z:\Projects\ExampleProject

input:
  preprocess_dir: Z:\Projects\ExampleProject\preprocess

model:
  model_id: 2mice_upview_implanted_nonimplanted_5node_v1_bottomup
  model_family: bottomup

profile:
  profile_id: 2mice_bottomup_default_v1

execution:
  provider: local_cuda
  requested_device: cuda

overrides:
  batch_size: null
  max_instances: null
  peak_threshold: null
  advanced_cli_overrides: {}
```

### 13.2 Resolved settings

After validation, Subsystem 02 shall write a fully resolved settings record:

```text
settings_used.yaml
```

This record shall include:

```text
selected model
selected profile
runtime version
device
resolved paths
all effective inference parameters
derived tracking-window frame count
validated CLI command
output locations
```

---

## 14. Common Inference Settings

These settings are shared across model families.

```yaml
runtime:
  device: cuda
  batch_size: 8

prediction:
  max_instances: 6
  min_instance_peaks: 1
  integral_refinement:
  integral_patch_size: 5

tracking:
  enabled: true
  candidates_method: local_queues
  max_tracks: 2

  tracking_window_sec: 2.0
  track_matching_method: hungarian

  min_new_track_points: 1
  min_match_points: 1

  features: keypoints
  scoring_method: oks
  scoring_reduction:
  robust_best_instance:

  use_flow: false
  use_kalman: false

  of_img_scale: 1.0
  of_window_size: 21
  of_max_levels: 3

  post_connect_single_breaks: true

  tracking_target_instance_count: 0
  tracking_pre_cull_to_target: 0
  tracking_pre_cull_iou_threshold: 0

  tracking_clean_instance_count: 0
  tracking_clean_iou_threshold: 0
```

### 14.1 Tracking-window resolution

Profiles shall store tracking memory in seconds:

```yaml
tracking_window_sec: 2.0
```

Before execution, Subsystem 02 shall resolve:

```text
tracking_window_size =
round(tracking_window_sec × prepared_video_fps)
```

The resolved frame count shall be written to:

```text
settings_used.yaml
pose_meta.json
job_manifest.yaml
```

---

## 15. Candidate Preservation and Provisional Tracking

The expected animal count is:

```yaml
expected_animal_count: 2
```

This does not mean only two candidate pose instances shall exist or be retained per frame.

The following concepts are distinct:

```text
real animals
predicted pose instances
provisional SLEAP tracks
final biological identities
```

### 15.1 Candidate retention

`max_instances` controls the maximum number of raw pose candidates retained per frame.

The default for both bottom-up and top-down profiles shall be:

```yaml
max_instances: 6
```

A frame may contain:

```text
zero instances
one instance
two instances
more than two instances
```

Additional instances may represent:

```text
false positives
duplicate detections
partial detections
fragmented poses
occluded real detections
recovery candidates useful for later QC
```

Subsystem 02 shall preserve all candidates retained by SLEAP.

### 15.2 Track limiting

```yaml
max_tracks: 2
```

instructs SLEAP to limit the creation of provisional tracks.

It does not mean:

```text
only two pose candidates may be retained
every candidate must receive a provisional track label
provisional track labels equal final animal identity
```

When:

```yaml
max_instances: 6
max_tracks: 2
```

a candidate may be retained without a provisional track label.

That candidate shall remain represented in:

```text
pose.slp
pose.parquet
```

with null or missing provisional tracking fields when SLEAP did not assign a track.

### 15.3 Non-destructive tracking policy

Subsystem 02 shall use SLEAP internal tracking controls:

```yaml
candidates_method: local_queues
max_tracks: 2
min_new_track_points: 1
min_match_points: 1
track_matching_method: hungarian
post_connect_single_breaks: true
```

The one-point thresholds are intentional.

A real mouse may be strongly occluded and only one visible node may be available, including in mixed implanted/non-implanted sessions.

### 15.4 No destructive cleanup by default

Subsystem 02 shall not enable SLEAP settings that intentionally reduce candidate counts by default.

```yaml
tracking_target_instance_count: 0
tracking_pre_cull_to_target: 0
tracking_clean_instance_count: 0
```

This preserves extra candidates for later tracking and QC.

---

## 16. Bottom-Up Default Profile

```yaml
profile_id: 2mice_bottomup_default_v1
model_family: bottomup

runtime:
  device: cuda
  batch_size: 8

prediction:
  max_instances: 6
  peak_threshold: 0.05
  min_instance_peaks: 1
  integral_refinement: null
  integral_patch_size: 5

tracking:
  enabled: true
  candidates_method: local_queues
  max_tracks: 2

  tracking_window_sec: 2.0
  track_matching_method: hungarian

  min_new_track_points: 1
  min_match_points: 1

  features: keypoints
  scoring_method: oks
  scoring_reduction: robust_quantile
  robust_best_instance: 0.85

  use_flow: false
  use_kalman: false

  of_img_scale: 1.0
  of_window_size: 21
  of_max_levels: 3

  post_connect_single_breaks: true

  tracking_target_instance_count: 0
  tracking_pre_cull_to_target: 0
  tracking_clean_instance_count: 0

filtering:
  filter_overlapping: false
  filter_min_visible_nodes: 0
  filter_min_visible_node_fraction: 0.0
  filter_min_mean_node_score: 0.0
  filter_min_instance_score: 0.0

outputs:
  write_native_slp: true
  write_pose_parquet: true
  create_overlay: true
  overlay_required_for_success: false
```

### 16.1 Bottom-up advanced settings

The following settings shall be available only in Advanced settings until targeted experiments justify a profile-level default change:

```yaml
max_edge_length_ratio:
dist_penalty_weight:
n_points:
min_line_scores:
queue_maxsize:
```

---

## 17. Top-Down Initial Profile

```yaml
profile_id: 2mice_topdown_default_v1
model_family: topdown

runtime:
  device: cuda
  batch_size: 24

topdown:
  centroid_peak_threshold: 0.10
  centered_instance_peak_threshold: 0.45
  anchor_part: tail_base

prediction:
  max_instances: 6
  min_instance_peaks: 1
  integral_refinement: integral
  integral_patch_size: 5

tracking:
  enabled: true
  candidates_method: local_queues
  max_tracks: 2

  tracking_window_sec: 2.0
  track_matching_method: hungarian

  min_new_track_points: 1
  min_match_points: 1

  features: keypoints
  scoring_method: oks
  scoring_reduction: mean
  robust_best_instance: 1.0

  use_flow: false
  use_kalman: false

  of_img_scale: 1.0
  of_window_size: 21
  of_max_levels: 3

  post_connect_single_breaks: true

  tracking_target_instance_count: 0
  tracking_pre_cull_to_target: 0
  tracking_clean_instance_count: 0

filtering:
  filter_overlapping: true
  filter_overlapping_method: iou
  filter_overlapping_threshold: 0.8

  filter_min_visible_nodes: 0
  filter_min_visible_node_fraction: 0.0
  filter_min_mean_node_score: 0.0
  filter_min_instance_score: 0.0

outputs:
  write_native_slp: true
  write_pose_parquet: true
  create_overlay: true
  overlay_required_for_success: false
```

### 17.1 Top-down threshold serialization

The scientific settings shall remain explicit:

```yaml
centroid_peak_threshold: 0.10
centered_instance_peak_threshold: 0.45
```

The exact installed SLEAP-NN v0.3 CLI serialization shall be confirmed during acceptance testing and recorded in the capability registry.

The profile shall not be marked executable until the capability registry confirms that the pinned CLI supports independent top-down threshold control.

---

## 18. Native Output Structure

Each successful or partially successful run shall write under:

```text
ProjectName/
└── pose_inference/
    └── <model-id>__<timestamp>/
        ├── pose.slp
        ├── pose.parquet
        ├── overlay.mp4
        ├── pose_meta.json
        ├── settings_used.yaml
        ├── job_manifest.yaml
        └── processing_log.txt
```

### 18.1 Artifact roles

```text
pose.slp
    Native authoritative SLEAP result.

pose.parquet
    Default time-aligned numeric pose artifact.

overlay.mp4
    Optional visual QC artifact, enabled by default.

pose_meta.json
    Single authoritative metadata, summary, and validation artifact.

settings_used.yaml
    Fully resolved settings used for the run.

job_manifest.yaml
    Immutable request and execution provenance.

processing_log.txt
    Human-readable execution and validation log.
```

---

## 19. Native SLEAP Output Contract

`pose.slp` shall remain the authoritative SLEAP result.

Subsystem 02 shall preserve:

```text
videos
skeletons
labeled frames
predicted instances
point coordinates
point scores
point visibility
point completeness metadata
instance scores
provisional track assignments
tracking scores
SLEAP provenance
```

The final `.slp` schema shall be frozen only after the pinned SLEAP-NN v0.3 and sleap-io v0.8.x smoke test is complete.

---

## 20. pose.parquet Contract

`pose.parquet` shall be the single default numeric output format.

NPZ may be added later as an optional compatibility export, but shall not be a default v1 artifact.

### 20.1 Row layout

The table shall contain one row per:

```text
prepared frame
× retained predicted instance
× skeleton node
```

The export shall not force two-animal slots.

### 20.2 Required columns

```text
prepared_frame_idx
raw_decode_frame_idx

prepared_time_sec
raw_pts_time_sec
external_time_sec

instance_index_in_frame
sleap_track_name

instance_score
tracking_score
instance_n_visible
instance_centroid_x_px
instance_centroid_y_px

node_index
node_name

x_px
y_px
node_score
node_visible
node_complete
```

### 20.3 Point interpretation

```text
x_px, y_px
    Pixel coordinates in prepared-video coordinates.

node_score
    SLEAP node confidence score.

node_visible
    Whether the node has usable coordinates.

node_complete
    Native SLEAP annotation-state metadata.
    Preserved for compatibility and provenance.
    Not a prediction-confidence score.
    Not an analysis-quality filter.
```

### 20.4 Missing nodes

Missing nodes shall remain missing:

```text
x_px = NaN
y_px = NaN
node_score = NaN
node_visible = false
node_complete = false
```

### 20.5 Parquet implementation

The recommended initial implementation is:

```yaml
parquet:
  engine: pyarrow
  compression: zstd
  compression_level: 3
  use_dictionary: true
  schema_version: pose_parquet_v1
```

---

## 21. pose_meta.json Contract

`pose_meta.json` shall merge metadata, numerical summary, provenance, and validation information.

Required top-level sections:

```text
schema_version
run
input_preprocess
model
runtime
execution_provider
resolved_settings
resolved_cli_command
tracking
outputs
summary
validation
warnings
```

### 21.1 Required summary information

```text
requested_frame_count
processed_frame_count
first_frame_idx
last_frame_idx

total_predicted_instances
minimum_instances_per_frame
maximum_instances_per_frame
mean_instances_per_frame

frames_with_zero_instances
frames_with_one_instance
frames_with_two_instances
frames_with_more_than_two_instances

provisional_track_count
instances_with_track_assignment
instances_without_track_assignment

per_node_visible_count
per_node_missing_count
per_node_score_summary

instance_score_summary
tracking_score_summary

overlay_status
validation_status
```

---

## 22. Overlay Policy

The preferred overlay renderer shall be the native renderer available in the pinned SLEAP-NN v0.3 / sleap-io v0.8.x environment.

Overlay generation shall be:

```text
enabled by default
optional for run success
validated independently
```

The overlay should show:

```text
prepared-video frame
predicted node markers
skeleton edges
provisional SLEAP track labels when available
frame index
```

The overlay shall not visually imply that a provisional SLEAP track is a final biological identity.

The exact renderer entry point, output format, and rendering options shall be confirmed during acceptance testing.

---

## 23. Validation Requirements

### 23.1 Input validation

Hard failures:

```text
required preprocessing artifact is missing
prepared video cannot be opened
prepared_sync.npz cannot be loaded
prepared frame count is inconsistent
required timing fields are absent
```

### 23.2 Runtime validation

Hard failures:

```text
SLEAP-NN version does not match pinned v0.3 requirement
sleap-io version is outside supported range
sleap-nn predict is unavailable
requested device is unavailable
required model component is missing
requested parameter is unsupported
requested parameter has invalid type or value
```

### 23.3 Native SLEAP validation

Hard failures:

```text
pose.slp cannot be loaded
no video record exists
no skeleton exists
skeleton differs from selected model manifest
frame index lies outside prepared-video range
duplicate output frame index exists
frame identity cannot be joined to prepared timing
```

Valid but reportable outcomes:

```text
zero predicted instances in a frame
one predicted instance in a frame
more than two predicted instances in a frame
untracked retained instance
missing node coordinates
missing headstage node on non-implanted mouse
```

### 23.4 Parquet validation

Hard failures:

```text
pose.parquet cannot be written
required columns are missing
timing join fails
frame index lies outside prepared-video range
node name differs from selected skeleton
row count is inconsistent with native retained instance-node records
```

### 23.5 Candidate-preservation audit

Every run shall report:

```text
raw_instance_count
tracked_instance_count
untracked_instance_count
unique_track_count
```

The audit shall confirm:

```text
no wrapper-side postprocessing removed a retained SLEAP candidate
all retained native SLEAP instances were exported to pose.parquet
any SLEAP-native filtering or cleanup is visible in resolved settings
```

---

## 24. User Interface Architecture

Subsystem 02 shall initially use a separate inference entry point.

```text
scripts/launch_pose_inference.bat
```

The initial interface shall remain independent from the Subsystem 01 preprocessing GUI.

```text
PoseInferenceWindow
├── Project / Preprocess Input
├── Model and Profile Selection
├── Basic Inference Settings
├── Advanced Settings
├── Runtime / Execution Provider
├── Run Summary
└── Output and Validation Results
```

### 24.1 Basic settings

```text
model family
registered model
profile
execution provider
device
max instances
overlay enabled
```

### 24.2 Advanced settings

```text
batch-size override
peak thresholds
tracking window in seconds
max tracks
min new track points
min match points
overlap filtering
optical flow
Kalman tracking
validated advanced SLEAP-NN parameters
```

### 24.3 Batch-size policy

Initial behavior:

```text
Basic GUI
    Uses the profile batch-size default and displays the resolved value.

Advanced GUI
    Allows a validated manual batch-size override.

Future benchmark tool
    Tests safe candidate batch sizes and recommends a value.
    It does not automatically change global defaults without user confirmation.
```

---

## 25. Future Unified Launcher

A unified launcher shall be created only after Subsystem 02 has independently passed its acceptance tests.

The unified launcher shall be a shell and navigator, not a rewrite of either subsystem.

```text
SuiteLauncher
├── Video Preprocessing
│   └── Existing Subsystem 01 workflow
│
├── Pose Inference
│   └── Independent Subsystem 02 workflow
│
└── Future modules
    ├── Pose QC
    ├── Behavior Features
    └── Analysis and Reporting
```

The existing Subsystem 01 launcher shall remain available during and after unified-launcher development.

---

## 26. Run Identifier

Run directories shall use:

```text
<model-id>__<timestamp>
```

Example:

```text
2mice_bottomup_implanted_nonimplanted_5node_v1__20260705T143522
```

The full local timestamp and timezone shall be stored in `pose_meta.json`.

---

## 27. Implementation Readiness

This specification is approved as the design contract for Subsystem 02.

Implementation may begin with:

```text
documentation
contracts
model registry
profile loading
request validation
runtime capability capture
CLI command generation
artifact export
artifact validation
```

Production inference defaults shall remain subject to the acceptance-test specification.

The following require empirical confirmation under the pinned SLEAP-NN v0.3 runtime before they are frozen as production defaults:

```text
exact top-down threshold CLI serialization
native overlay renderer and output options
bottom-up batch-size default
top-down batch-size default
top-down overlap-filtering default
optical-flow benefit
Kalman-tracking benefit and compatibility with candidate preservation
```
