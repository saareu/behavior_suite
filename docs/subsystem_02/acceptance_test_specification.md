# Subsystem 02 — Backend Inference Acceptance-Test Specification

## 1. Purpose

This document defines acceptance evidence for the Subsystem 02 backend
SLEAP/SLEAP-NN inference implementation.

Acceptance focuses on reproducible inference, the minimal artifact contract,
pose coverage, pose-quality QC, overlay generation, Parquet export, and
preservation of the Subsystem 01 timing/frame contract.

This is not the full Subsystem 02 MVP acceptance specification. The full MVP
also requires UI-based inference and review, main UI integration, existing-run
review, and downstream run selection. See
[`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).

Identity-stable final tracking is not a blocking acceptance criterion for
Subsystem 02.

## 2. Acceptance Scope

Backend acceptance tests must prove that the Subsystem 02 backend can:

- consume Subsystem 01 prepared outputs without modifying them;
- run the selected default SLEAP/SLEAP-NN inference profile reproducibly;
- create the minimal output directory;
- store SLEAP/SLEAP-NN tracking inside `pose.slp` when tracking is enabled;
- export `pose.parquet` with pose, frame indices, S1 timing, and frame-level
  metadata;
- generate `overlay.mp4` from `pose.slp`;
- write `pose_meta.json`, `settings_used.yaml`, `job_manifest.yaml`, and
  `processing_log.txt`;
- report technical pose-inference QC without claiming final biological identity
  continuity or final session usability.

The current provisional SLEAP tracking settings are accepted as good enough for
Subsystem 02 development. Parameter optimization is postponed to a later guided
workflow.

The backend supports bottom-up inference from one model and top-down inference
from a centroid plus centered-instance bundle. Bottom-up has real GPU smoke
evidence. A SLEAP-NN 0.3.0 top-down smoke also completed inference, Parquet
export, technical QC, overlay generation, and run discovery with a successful
`complete_reviewable` result and QC outcome `pass`.
See [`evidence/topdown_gpu_smoke_v030.md`](evidence/topdown_gpu_smoke_v030.md).

Acceptance exercises these public command forms (with optional `--profile` and
`--dry-run` as appropriate):

```powershell
python -m pose_inference run --session-root SESSION `
  --inference-mode bottomup --model-path BOTTOMUP_MODEL

python -m pose_inference run --session-root SESSION `
  --inference-mode topdown --centroid-model-path CENTROID_MODEL `
  --centered-instance-model-path CENTERED_INSTANCE_MODEL
```

Both modes must produce the same locked artifacts and traverse the same S1
timing, Parquet, technical-QC, diagnostic-findings, review-recommendation, and
overlay path. Provisional tracks remain S2 output preparation and are not final
identity validation.

## 3. Required Inputs

Each acceptance case must start from Subsystem 01 prepared outputs:

```text
preprocess/
├── prepared_video.mp4
├── prepare_meta.json
└── prepared_sync.npz
```

The tests must verify that Subsystem 01 remains the source of truth for:

- prepared frame index;
- raw decode frame index;
- timing;
- prepared video dimensions and FPS;
- crop/prepared geometry metadata.

## 4. Expected Output Contract

Each successful or warning-success run must write:

```text
pose_inference/{model-id}__{timestamp}/
├── pose.slp
├── pose.parquet
├── overlay.mp4
├── pose_meta.json
├── settings_used.yaml
├── job_manifest.yaml
└── processing_log.txt
```

The following are not accepted as required standard outputs:

- `pose_tracked.slp`;
- `overlay_tracked.mp4`;
- `tracking_qc.csv`;
- `tracking_report.json`;
- `track_identity_map.json`.

Tracking, when enabled, must be inside `pose.slp`.

## 5. Acceptance Clip Set

Acceptance cases A-D are short frozen prepared-video clips for detector and
provisional tracking behavior.

```text
A_clear_separation
    Short frozen clip with two clearly separated animals.

B_social_proximity
    Short frozen clip with close contact but distinguishable animals.

C_strong_occlusion
    Short frozen clip with partial occlusion and reduced visible keypoints.

D_motion_transition
    Short frozen clip with faster motion, crossing, or direction change.
```

Acceptance case E is a full-session integration case:

```text
E_full_session_integration
    Full-session Subsystem 1-to-Subsystem 2 integration case.
```

A-D are used to validate reproducible inference behavior on fixed short clips.
E validates full-session input handling, output writing, timing preservation,
and practical runtime behavior.

## 6. Required Checks for A-D

Each A-D clip must check:

- required S1 input artifacts are present and read-only;
- inference completes with the established default profile;
- `pose.slp` is written and loadable;
- tracking is present in `pose.slp` when enabled by the profile;
- `pose.parquet` is written and validates against `pose.slp`;
- all exported rows retain `prepared_frame_idx`;
- S1 timing columns are joined when available;
- frame references remain inside the prepared-video frame range;
- required overlay generation succeeds and writes a readable artifact;
- pose-quality QC is written to `pose_meta.json`;
- `settings_used.yaml`, `job_manifest.yaml`, and `processing_log.txt` are
  written.

## 7. Required Checks for E_full_session_integration

The full-session integration case must check:

- Subsystem 02 can consume the full-session Subsystem 01 prepared video;
- output directory naming is deterministic and collision-safe;
- long-run logs are written;
- `pose.slp` and `pose.parquet` validate for the full processed frame range;
- S1 frame identity and timing are preserved through the export;
- pose-quality QC summarizes the full run;
- no generated outputs are written into `preprocess/`;
- no separate tracking artifacts are required.

## 8. pose.parquet Acceptance Criteria

`pose.parquet` must include, at minimum:

- frame index;
- video index/name;
- track when present;
- instance index;
- node name/index;
- node coordinates;
- node scores;
- instance score when available;
- S1 timing columns from `prepared_sync.npz` and `prepare_meta.json` when
  available;
- relevant frame-level metadata needed downstream.

The export must preserve retained instances from `pose.slp`. Missing keypoints
must remain missing rather than being interpolated.

## 9. Technical Pose-Inference QC Acceptance Criteria

`pose_meta.json` must include a pose-quality QC summary covering, when
available:

- animal count coverage;
- frames with zero animals;
- frames with fewer than expected animals;
- frames with extra animals;
- missing keypoint rates;
- low-confidence keypoint rates;
- partial skeleton frequency;
- duplicate candidate risk;
- implausible geometry flags;
- tracked and untracked instance counts when tracks are present.

This QC section must not include pipeline success/provenance as pose-quality
metrics. Pipeline success and provenance belong in `job_manifest.yaml` and
`processing_log.txt`.

QC retains `status: computed` and separately records outcome `pass`,
`review_recommended`, or `failed`. Acceptance must prove that:

- zero represented prepared frames containing at least one finite x/y pose
  point produces `failed`;
- at least one such frame avoids that hard-failure condition;
- a moderate exactly-one-animal fraction does not recommend review;
- an exactly-one-animal fraction at or above `0.90` recommends review only for
  the configured two-animal workflow;
- a moderate missing-keypoint fraction does not recommend review;
- a missing-keypoint fraction at or above `0.90` recommends review;
- `review_recommended` remains a successful, S3-eligible run;
- triggered warnings include at most 10 longest-first contiguous intervals,
  generated independently per video, using inclusive frame bounds and timing
  when available; `time_span_sec` is the end timestamp minus the start timestamp
  and may be zero for a one-frame interval.

These are conservative technical-review thresholds, not scientific-validity
criteria. No MVP triggers are added for partial-skeleton runs, extended
low-confidence periods, unexpected animal-count distributions, identity
switches, or tracking continuity. Final biological identity, tracking
correctness, and final session usability are S3 responsibilities. No persistent
S2 user-review/final-usability record or elaborate S2 review screen is required.

## 10. Overlay Acceptance Criteria

`overlay.mp4` must be generated from `pose.slp`.

If tracks are present, the overlay may color by track. Track coloring must be
treated as provisional SLEAP/SLEAP-NN tracking, not final biological identity.

Overlay validation should check:

- readable video file;
- expected frame count when available;
- prepared-video dimensions unless explicitly documented otherwise;
- visible skeleton/node rendering on sampled frames;
- a hard post-run failure if the required overlay is missing, unreadable, or
  cannot be generated.

## 11. Preflight and Post-Run Validation Acceptance

Before subprocess submission, acceptance tests must cover missing or unreadable
S1 inputs, invalid S1 timing-array lengths or prepared-frame mappings,
prepared-video/S1 frame-count disagreement, missing model/profile inputs, and
an output location that cannot be created or written. Top-down cases must also
cover incomplete bundles, duplicate component paths, structurally incompatible
model roles, and profile/mode conflicts. A valid S1 handoff must
still permit normal command execution and dry-run behavior.

Post-run tests must cover subprocess failure, missing/unreadable required S2
artifacts, prediction/S1 frame mismatch, invalid exported frame/timing mapping,
and the exact zero-finite-pose-frame hard-failure definition. Missing S2
outputs are post-run failures, not preflight failures.

## 12. Settings and Provenance Acceptance Criteria

`settings_used.yaml` must record the actual SLEAP/SLEAP-NN parameters used.

`job_manifest.yaml` must record:

- input artifact paths and fingerprints;
- output artifact paths and fingerprints;
- model identity and model artifact fingerprints;
- runtime profile;
- invocation provenance;
- final run status.

`processing_log.txt` must record runtime logs, warnings, errors, and validation
messages.

## 13. Non-Regression Requirement

Backend Subsystem 02 acceptance must not require changes to Subsystem 01
behavior.

Before merging implementation changes, the existing Subsystem 01 tests and GUI
launch path must remain valid. SLEAP/SLEAP-NN dependencies must not be required
to launch or use Subsystem 01 preprocessing.

Full Subsystem 02 MVP acceptance additionally requires:

- UI-based inference and review;
- main UI launch/navigation integration;
- opening existing completed Subsystem 02 runs for review, reuse, rerun, and
  downstream selection;
- a Subsystem 01 completion-to-Subsystem 02 transition.

## 14. Pass/Fail Summary

Subsystem 02 passes initial acceptance when:

1. A-D short frozen clips produce valid minimal artifacts.
2. E full-session integration produces valid minimal artifacts.
3. `pose.parquet` preserves S1 frame/timing contract.
4. `pose_meta.json` reports pose-quality QC.
5. Required `overlay.mp4` is created and readable.
6. Tracking, when enabled, is stored inside `pose.slp`.
7. No separate tracking artifacts are required.
8. Identity-stable final tracking is not treated as a blocker.
