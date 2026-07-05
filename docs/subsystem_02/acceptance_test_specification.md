## 24. SLEAP-NN v0.3 Acceptance-Test Specification

### 24.1 Purpose

This acceptance-test specification defines the evidence required before Subsystem 02 is declared ready for production implementation.

The tests shall establish that the pinned SLEAP-NN v0.3 environment can:

```text
run the selected registered models
produce valid native SLEAP output
preserve candidate detections as intended
apply provisional SLEAP tracking as intended
export time-aligned pose.parquet
generate a usable native overlay
operate safely on the selected GPU
```

These tests shall also determine the remaining empirical profile defaults.

---

### 24.2 Test Principles

All acceptance tests shall follow these rules:

```text
1. Use prepared videos produced by Subsystem 01.

2. Do not alter Subsystem 01 artifacts.

3. Use a dedicated Subsystem 02 test-run directory.

4. Save every resolved setting, command, output, log, and result summary.

5. Test one variable at a time whenever possible.

6. Compare settings on the same exact frame ranges.

7. Do not select a default based only on speed.
   Candidate preservation and visual plausibility are required.

8. A test failure shall produce an interpretable failure record,
   not merely a console error.
```

---

### 24.3 Test Environment Capture

Before any inference experiment, capture the exact runtime environment.

Required evidence artifacts:

```text
docs/evidence/sleapnn_v030/
├── sleapnn_version.txt
├── sleapnn_predict_help.txt
├── sleapnn_system.txt
├── python_runtime.txt
├── installed_package_paths.txt
├── cli_capability_catalog.yaml
└── environment_validation.json
```

Required captured values:

```text
sleap-nn version
sleap-io version
Python version
PyTorch version
CUDA version
cuDNN version
GPU name
GPU total memory
operating system
sleap-nn executable path
Python executable path
sleap-nn package path
sleap-io package path
```

Environment validation shall fail when:

```text
sleap-nn is not version 0.3.x
sleap-io is outside the v0.3-supported version range
the selected CUDA device is unavailable
sleap-nn predict is unavailable
required CLI options are unavailable
```

---

### 24.4 Test Input Set

Acceptance tests shall use a small, fixed set of representative prepared-video clips.

Each clip shall preserve the original frame identity from Subsystem 01.

```text
TestClipSet/
├── A_clear_separation/
│   └── 300-frame prepared-video segment
│
├── B_social_proximity/
│   └── 300-frame prepared-video segment
│
├── C_strong_occlusion/
│   └── 300-frame prepared-video segment
│
└── D_motion_transition/
    └── 300-frame prepared-video segment
```

### 24.4.1 Required Clip Characteristics

```text
A_clear_separation
    Two clearly separated animals.
    Used to assess baseline detection and track initialization.

B_social_proximity
    Animals close together but still distinguishable.
    Used to assess duplicate detections, instance overlap, and track stability.

C_strong_occlusion
    One animal partially obscures the other.
    Used to assess one-node recovery, track continuity, and candidate preservation.

D_motion_transition
    Animals move rapidly, cross paths, or change direction.
    Used to assess tracking continuity and possible value of optical flow or Kalman tracking.
```

The same clips shall be used for all profile comparisons.

---

### 24.5 Required Test Run Structure

Each test run shall be stored independently:

```text
test_runs/
└── subsystem02_v030/
    └── <test-id>__<timestamp>/
        ├── inference_request.yaml
        ├── settings_used.yaml
        ├── command.txt
        ├── stdout.log
        ├── stderr.log
        ├── pose.slp
        ├── pose.parquet
        ├── overlay.mp4
        ├── pose_meta.json
        ├── test_result.json
        └── review_notes.md
```

Suggested test identifier format:

```text
<model-family>__<test-clip>__<variable-name>__<value>
```

Examples:

```text
bottomup__C_strong_occlusion__batch_size__16
topdown__B_social_proximity__overlap_filter__enabled
bottomup__D_motion_transition__tracking_mode__flow
```

---

### 24.6 Baseline Bottom-Up Smoke Test

#### Goal

Verify that the registered bottom-up model can run under the pinned v0.3 environment and produce valid Subsystem 02 outputs.

#### Initial settings

```yaml
model_family: bottomup

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
  min_new_track_points: 1
  min_match_points: 1
  track_matching_method: hungarian
  post_connect_single_breaks: true

  tracking_target_instance_count: 0
  tracking_pre_cull_to_target: 0
  tracking_clean_instance_count: 0
```

#### Minimum test

```text
Input:
    A_clear_separation, 300 frames.

Expected:
    Native pose.slp is written.
    pose.parquet is written.
    Timing is joined for every exported frame.
    Overlay is attempted.
    No candidates are removed by wrapper-side logic.
```

#### Pass criteria

```text
1. The model loads without checkpoint incompatibility errors.

2. The prediction run completes successfully on the requested device.

3. pose.slp loads through the pinned sleap-io version.

4. Exported frame indices are within the prepared-video range.

5. Every exported frame has timing joined from prepared_sync.npz.

6. pose.parquet contains all required fields.

7. Every retained instance in pose.slp is represented in pose.parquet.

8. Every node record in pose.parquet matches the selected skeleton.

9. The overlay is created, or failure is recorded as a non-fatal warning.

10. Existing Subsystem 01 tests and GUI launch smoke test remain unaffected.
```

---

### 24.7 Top-Down CLI Capability Test

#### Goal

Determine the exact `sleap-nn predict` v0.3 command syntax for top-down inference.

The wrapper shall not assume top-down threshold argument names from package source alone.

#### Required checks

```text
1. Confirm whether `sleap-nn predict --help` exposes:
   - a centroid-specific threshold option;
   - a centered-instance/keypoint threshold option;
   - repeated model-path support;
   - anchor-part support.

2. Confirm the required order of:
   - centroid model path;
   - centered-instance model path.

3. Confirm whether top-down inference accepts:
   - two model directories;
   - two checkpoint paths;
   - mixed directory/checkpoint inputs.

4. Confirm which threshold is written to output provenance.

5. Confirm that the output skeleton matches the model manifest.
```

#### Top-down threshold target

The scientific profile shall remain:

```yaml
topdown:
  centroid_peak_threshold: 0.10
  centered_instance_peak_threshold: 0.45
```

The acceptance test shall establish the exact v0.3 CLI serialization.

#### Pass criteria

```text
1. Both top-down model components are accepted by the pinned CLI.

2. The two threshold values can be passed independently.

3. The produced pose.slp is valid.

4. Resolved metadata records both threshold values separately.

5. The output can be exported to pose.parquet without losing track,
   instance, node, or timing information.
```

If independent threshold control cannot be confirmed through the pinned v0.3 CLI, the top-down profile shall remain marked:

```text
status = not_yet_executable
reason = separate_threshold_serialization_unverified
```

---

### 24.8 Native Overlay Test

#### Goal

Determine the official SLEAP-native overlay mechanism for the pinned v0.3 environment.

#### Required checks

```text
1. Identify the exact rendering entry point.

2. Confirm whether it writes MP4 directly.

3. Confirm which input is required:
   - pose.slp only;
   - pose.slp plus reachable prepared video;
   - another rendering configuration.

4. Confirm whether the renderer works with non-embedded source-video references.

5. Confirm visual availability of:
   - node markers;
   - skeleton edges;
   - frame position;
   - provisional track labels;
   - missing-node behavior.

6. Measure output resolution, FPS, frame count, and duration.
```

#### Overlay validation

```text
overlay frame count must equal the processed-frame count
overlay dimensions must match prepared-video dimensions unless explicitly documented
overlay timing must correspond to the prepared-video frame sequence
overlay shall be readable by OpenCV or ffprobe validation
```

#### Pass criteria

```text
1. The renderer can create an inspectable overlay for a 300-frame test.

2. Predicted nodes and skeleton edges are visible.

3. No silent mismatch exists between overlay frames and pose-frame indices.

4. The renderer does not require modifying the prepared video.

5. Failure to render is clearly separated from inference failure.
```

---

### 24.9 Batch-Size Benchmark Test

#### Goal

Determine safe, fast initial batch-size defaults for the actual GPU, model family, and prepared-video geometry.

#### Initial batch-size candidates

```text
4
8
12
16
24
32
```

Additional candidates may be added only after the previous candidate completes successfully.

#### Test design

For each model family:

```text
1. Use the same 300-frame test clip.

2. Keep every setting identical except batch size.

3. Run each candidate twice.

4. Record performance and validation results.

5. Stop increasing batch size after the first CUDA out-of-memory failure.
```

#### Required metrics

```text
batch_size
run duration
prediction duration
tracking duration
frames per second
peak GPU memory
GPU utilization
CPU utilization
system RAM
successful completion
CUDA out-of-memory occurrence
pose.slp validation status
pose.parquet validation status
overlay validation status
```

#### Selection rule

The selected profile default shall be:

```text
the largest batch size that completes two consecutive runs,
does not exceed 85% of available GPU memory,
does not cause unstable throughput,
and produces valid output artifacts.
```

The user-facing batch-size policy shall be:

```text
Basic GUI:
    use profile default.

Advanced GUI:
    allow validated manual override.

Future benchmark tool:
    recommend a batch size but never apply it globally without user confirmation.
```

---

### 24.10 Top-Down Overlap-Filtering Experiment

#### Goal

Determine whether top-down overlap filtering removes duplicates without removing useful detections during close contact or occlusion.

#### Configurations

```yaml
Condition A:
  filter_overlapping: false

Condition B:
  filter_overlapping: true
  filter_overlapping_method: iou
  filter_overlapping_threshold: 0.8
```

Potential later condition:

```yaml
Condition C:
  filter_overlapping: true
  filter_overlapping_method: oks
  filter_overlapping_threshold: <validated value>
```

#### Required clips

```text
A_clear_separation
B_social_proximity
C_strong_occlusion
```

#### Required comparison metrics

```text
total retained instances
instances per frame
frames with zero instances
frames with one instance
frames with two instances
frames with more than two instances
number of untracked candidates
number of provisional tracks
number of candidates removed by filtering
manual visual review of selected overlap frames
```

#### Decision rule

Enable overlap filtering by default only when:

```text
it reduces obvious duplicate detections
and
it does not remove a plausible true-animal candidate
during the manually reviewed close-contact and occlusion frames.
```

Candidate preservation takes priority over cosmetic reduction of extra detections.

---

### 24.11 Optical-Flow Tracking Experiment

#### Goal

Determine whether SLEAP optical-flow tracking improves provisional continuity enough to justify its computational cost and complexity.

#### Configurations

```yaml
Baseline:
  use_flow: false

OpticalFlow:
  use_flow: true
  of_img_scale: 1.0
  of_window_size: 21
  of_max_levels: 3
```

#### Required clips

```text
C_strong_occlusion
D_motion_transition
```

#### Required comparison metrics

```text
total provisional track count
new-track events
track breaks
track re-connections
frames with no assigned track
frames with multiple competing candidates
track-assignment continuity
runtime
GPU memory
visual review of identity continuity
```

#### Decision rule

Enable optical flow by default only when it:

```text
reduces track fragmentation or obvious identity instability
without reducing retained candidate instances
without creating visually implausible motion associations
and without unacceptable runtime cost.
```

Otherwise:

```yaml
use_flow: false
```

remains the default.

---

### 24.12 Kalman Tracking Experiment

#### Goal

Determine whether SLEAP Kalman tracking improves provisional continuity while preserving raw candidates.

#### Important constraint

Kalman tracking may require or derive a target instance count.

Therefore, this test must explicitly determine whether its use:

```text
only changes provisional track assignment
or
also culls, removes, or suppresses retained candidates.
```

#### Configurations

```yaml
Baseline:
  use_kalman: false

KalmanCentroid:
  use_kalman: true
  kf_track_features: centroid
  kf_init_frame_count: 10
  kf_reset_gap_size: 5

KalmanKeypoints:
  use_kalman: true
  kf_track_features: keypoints
  kf_init_frame_count: 10
  kf_reset_gap_size: 5
```

#### Required clips

```text
C_strong_occlusion
D_motion_transition
```

#### Required comparison metrics

```text
retained instance count before and after tracking
per-frame retained candidate count
number of candidates lost
number of candidates made untracked
provisional track continuity
new-track events
track breaks
runtime
GPU memory
manual visual review
```

#### Decision rule

Kalman tracking shall not be enabled by default unless all conditions hold:

```text
1. It improves track continuity or identity stability.

2. It does not cull or erase useful retained candidates.

3. Its required target-count behavior is compatible with the
   preservation-first contract.

4. It does not introduce visually implausible extrapolated associations.

5. Its runtime cost is acceptable.
```

Until then:

```yaml
use_kalman: false
```

shall remain the default.

---

### 24.13 Candidate Preservation Audit

Every acceptance test shall include a preservation audit.

For every processed frame, calculate:

```text
raw_instance_count
tracked_instance_count
untracked_instance_count
unique_track_count
```

Required summary fields:

```text
frames_with_zero_instances
frames_with_one_instance
frames_with_two_instances
frames_with_more_than_two_instances

instances_with_track_assignment
instances_without_track_assignment

maximum_instances_in_frame
maximum_unique_tracks_in_frame
total_unique_provisional_tracks
```

The audit shall verify:

```text
No wrapper-side postprocessing removed a retained SLEAP candidate.

Any candidate removed by SLEAP-native filtering or cleanup is visible in
the resolved settings and test comparison.

No candidate is silently dropped during pose.slp → pose.parquet export.
```

---

### 24.14 Subsystem 01 Non-Regression Test

After each implementation milestone that adds Subsystem 02 code:

```text
1. Run the existing Subsystem 01 automated tests.

2. Run the existing Subsystem 01 GUI smoke test.

3. Confirm the existing preprocessing launcher still opens.

4. Confirm preprocessing still completes a representative test video.

5. Confirm no SLEAP-NN dependency is needed to launch Subsystem 01.
```

Failure of this non-regression test blocks merging the Subsystem 02 change.

---

### 24.15 Acceptance-Test Result Format

Each test shall emit a machine-readable result:

```yaml
schema_version: subsystem02_acceptance_test_result_v1

test_id:
status:

runtime:
  gpu:
  gpu_memory_total_mb:
  batch_size:
  device:

input:
  test_clip_id:
  prepared_frame_count:
  prepared_fps:

model:
  model_id:
  model_family:

settings:
  resolved_settings_path:

performance:
  total_runtime_sec:
  prediction_runtime_sec:
  tracking_runtime_sec:
  fps:
  peak_gpu_memory_mb:

outputs:
  pose_slp:
    path:
    valid:
  pose_parquet:
    path:
    valid:
  overlay:
    path:
    attempted:
    valid:

preservation_audit:
  total_instances:
  tracked_instances:
  untracked_instances:
  frames_with_more_than_two_instances:
  total_unique_provisional_tracks:

validation:
  passed_checks:
  warnings:
  failures:

decision:
  recommendation:
  rationale:
```

---

### 24.16 Completion Criteria

Subsystem 02 v0.3 design shall move from:

```text
design_complete_pending_runtime_validation
```

to:

```text
implementation_ready
```

only when all of the following are complete:

```text
1. The exact v0.3 runtime environment is captured.

2. The v0.3 CLI capability catalog is generated.

3. Bottom-up smoke test passes.

4. Top-down command and threshold serialization are verified.

5. Native overlay behavior is verified.

6. pose.slp and pose.parquet validation pass.

7. Candidate-preservation audit passes.

8. Batch-size default is selected for the target GPU.

9. Initial decisions on overlap filtering, optical flow, and Kalman tracking
   are recorded.

10. Subsystem 01 non-regression tests pass.
```
