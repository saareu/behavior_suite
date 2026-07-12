# Subsystem 02 — Backend Inference Acceptance-Test Specification

## 1. Purpose

This document defines acceptance evidence for the Subsystem 02 backend
SLEAP/SLEAP-NN inference implementation.

Acceptance focuses on reproducible inference, the minimal artifact contract,
pose coverage, pose-quality QC, overlay generation, Parquet export, and
preservation of the Subsystem 01 timing/frame contract.

This is not the full Subsystem 02 MVP acceptance specification. The full MVP
also requires UI-based inference and review, top-down model support, main UI
integration, existing-run review, and downstream run selection. See
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
- report pose-quality QC without claiming final biological identity continuity.

The current provisional SLEAP tracking settings are accepted as good enough for
Subsystem 02 development. Parameter optimization is postponed to a later guided
workflow.

The current validated backend path is bottom-up inference. Top-down support is
required for the full Subsystem 02 MVP but is not yet covered by this backend
acceptance set unless a later implementation task explicitly extends it.

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
pose_inference/<model-id>__<timestamp>/
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
- overlay generation is attempted and either succeeds or records a non-fatal
  warning;
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

## 9. Pose-Quality QC Acceptance Criteria

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

Final biological identity-continuity evaluation is excluded from required
Subsystem 02 QC.

## 10. Overlay Acceptance Criteria

`overlay.mp4` must be generated from `pose.slp`.

If tracks are present, the overlay may color by track. Track coloring must be
treated as provisional SLEAP/SLEAP-NN tracking, not final biological identity.

Overlay validation should check:

- readable video file;
- expected frame count when available;
- prepared-video dimensions unless explicitly documented otherwise;
- visible skeleton/node rendering on sampled frames;
- clear warning record if overlay generation fails while inference and Parquet
  export are valid.

## 11. Settings and Provenance Acceptance Criteria

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

## 12. Non-Regression Requirement

Backend Subsystem 02 acceptance must not require changes to Subsystem 01
behavior.

Before merging implementation changes, the existing Subsystem 01 tests and GUI
launch path must remain valid. SLEAP/SLEAP-NN dependencies must not be required
to launch or use Subsystem 01 preprocessing.

Full Subsystem 02 MVP acceptance additionally requires:

- UI-based inference and review;
- top-down model support;
- main UI launch/navigation integration;
- opening existing completed Subsystem 02 runs for review, reuse, rerun, and
  downstream selection;
- a Subsystem 01 completion-to-Subsystem 02 transition.

## 13. Pass/Fail Summary

Subsystem 02 passes initial acceptance when:

1. A-D short frozen clips produce valid minimal artifacts.
2. E full-session integration produces valid minimal artifacts.
3. `pose.parquet` preserves S1 frame/timing contract.
4. `pose_meta.json` reports pose-quality QC.
5. `overlay.mp4` is created or failure is recorded as an allowed warning.
6. Tracking, when enabled, is stored inside `pose.slp`.
7. No separate tracking artifacts are required.
8. Identity-stable final tracking is not treated as a blocker.
