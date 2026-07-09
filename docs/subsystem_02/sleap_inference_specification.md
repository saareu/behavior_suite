# Subsystem 02 — SLEAP-NN Inference Specification

## 1. Purpose

Subsystem 02 runs SLEAP/SLEAP-NN pose inference on validated videos prepared by
Subsystem 01.

The purpose of Subsystem 02 is to create one native pose result, one
analysis-ready pose table, one optional visual overlay, and the minimal
metadata/provenance needed to reproduce and audit the run.

Subsystem 02 does not change preprocessing behavior. Subsystem 01 remains the
source of truth for frame identity, timing, crop geometry, prepared-video
dimensions, and preprocessing provenance.

## 2. Scope

Subsystem 02 is responsible for:

- validating required Subsystem 01 prepared outputs;
- resolving a registered SLEAP/SLEAP-NN model and one established default
  inference profile;
- running SLEAP/SLEAP-NN inference;
- enabling SLEAP/SLEAP-NN tracking inside the inference call when the profile
  enables tracking;
- writing native `pose.slp`;
- exporting `pose.parquet`;
- generating `overlay.mp4` from `pose.slp`;
- writing run metadata, settings, manifest, and logs;
- reporting pose-inference quality summaries.

Subsystem 02 is not responsible for:

- modifying Subsystem 01 artifacts;
- redefining frame timing or crop geometry;
- custom candidate selection after inference;
- separate tracking post-processing artifacts;
- final long-term biological identity continuity;
- parameter optimization or guided hyperparameter search;
- behavior classification or downstream biological analysis.

Parameter optimization is postponed to a later guided workflow. The initial
implementation uses one established default inference profile.

## 3. Required Inputs

Subsystem 02 consumes a completed Subsystem 01 output directory:

```text
preprocess/
├── prepared_video.mp4
├── prepare_meta.json
└── prepared_sync.npz
```

`prepared_video.mp4` is the inference video. `prepare_meta.json` provides
machine-readable preprocessing provenance and prepared-video metadata.
`prepared_sync.npz` provides the authoritative prepared-frame-to-raw-frame and
timing mapping.

Subsystem 02 may read additional Subsystem 01 artifacts when useful for
diagnostics, but these three files are the required input contract.

## 4. Subsystem 01 Contract

Subsystem 01 remains authoritative for:

- prepared frame count;
- prepared frame index;
- raw decode frame index;
- S1 timing columns;
- prepared-video size and FPS;
- crop geometry and canonical scale/pad metadata;
- preprocessing settings and provenance.

Subsystem 02 shall not write into `preprocess/`.

Subsystem 02 shall not estimate, replace, or independently reconstruct timing.
Timing exported by Subsystem 02 must come from `prepared_sync.npz` and
`prepare_meta.json` when available.

The join key for timing is:

```text
prepared_frame_idx
```

## 5. Output Directory

Each run writes one minimal output directory:

```text
pose_inference/
└── <model-id>__<timestamp>/
    ├── pose.slp
    ├── pose.parquet
    ├── overlay.mp4
    ├── pose_meta.json
    ├── settings_used.yaml
    ├── job_manifest.yaml
    └── processing_log.txt
```

No other standard artifacts are part of the locked Subsystem 02 contract.

The following are not standard outputs:

- `pose_tracked.slp`;
- `overlay_tracked.mp4`;
- `tracking_qc.csv`;
- `tracking_report.json`;
- `track_identity_map.json`.

Tracking, when enabled, is performed inside the SLEAP/SLEAP-NN inference call.
Track assignments are stored inside the single `pose.slp` output and exported
to `pose.parquet` when present.

## 6. Artifact Roles

### `pose.slp`

`pose.slp` is the single native SLEAP/SLEAP-NN output and the authoritative pose
artifact.

It contains:

- SLEAP video references;
- skeleton definitions;
- labeled/predicted frames;
- predicted instances;
- node coordinates and scores;
- instance scores when available;
- SLEAP/SLEAP-NN track assignments when tracking is enabled and assigned.

Subsystem 02 does not create a separate tracked `.slp`.

### `pose.parquet`

`pose.parquet` is the analysis-ready pose table. It is the default numeric
output for downstream analysis.

The table should include:

- `prepared_frame_idx`;
- `raw_decode_frame_idx`;
- `video_index`;
- `video_name`;
- `track` when present;
- `instance_index`;
- `node_index`;
- `node_name`;
- `x_px`;
- `y_px`;
- `node_score`;
- `instance_score` when available;
- S1 timing columns from `prepared_sync.npz` and `prepare_meta.json` when
  available;
- relevant frame-level metadata needed downstream, such as prepared video FPS,
  prepared frame count, raw frame count, trim information, and crop/prepared
  geometry identifiers.

The table must preserve all retained instances and nodes represented in
`pose.slp`. Missing coordinates remain missing; Subsystem 02 does not
interpolate nodes.

The exact Parquet schema should be versioned in metadata as
`pose_parquet_v1`.

### `overlay.mp4`

`overlay.mp4` is generated from `pose.slp` and the prepared video.

The overlay is a visual review aid. If tracks are present in `pose.slp`, the
overlay may color by track. Track colors must not imply final biological
identity continuity.

Overlay generation failure may be reported as a warning when `pose.slp`,
`pose.parquet`, and required metadata are valid.

### `pose_meta.json`

`pose_meta.json` contains machine-readable run metadata and pose-quality QC
summary.

The pose-quality QC section is limited to pose inference quality. Pipeline
success, dispatch provenance, and file provenance belong in
`job_manifest.yaml` and `processing_log.txt`.

### `settings_used.yaml`

`settings_used.yaml` records the actual SLEAP/SLEAP-NN parameters used for the
run, including:

- model id;
- runtime profile id;
- execution provider and device;
- inference profile;
- tracking enabled/disabled;
- all effective SLEAP/SLEAP-NN inference parameters.

### `job_manifest.yaml`

`job_manifest.yaml` records the input/output contract and provenance:

- Subsystem 01 input artifact paths and fingerprints;
- output artifact paths and fingerprints;
- model path and model metadata fingerprint;
- command or structured invocation record;
- run start/end timestamps;
- run status and warning/failure summary.

### `processing_log.txt`

`processing_log.txt` records runtime logs, command output, warnings, errors,
and validation messages useful for debugging and audit.

## 7. Default Inference Profile

The initial implementation uses one established default inference profile.

The profile may enable SLEAP/SLEAP-NN tracking. When enabled, tracking is run
inside the inference call and written into `pose.slp`.

The profile is not a parameter-optimization workflow. Changing thresholds,
batch size, tracking methods, or model-family-specific options belongs to a
later guided workflow unless a narrow implementation task explicitly updates
the default profile.

## 8. Tracking Scope

Subsystem 02 treats SLEAP/SLEAP-NN tracks as provisional inference output.

Tracks are useful for visual inspection and downstream analysis, but they are
not final biological identities. Final long-term identity continuity is outside
the required Subsystem 02 scope.

Subsystem 02 must not define separate tracking artifacts or identity maps as
required outputs.

## 9. pose.parquet Frame and Timing Contract

Every pose row must retain the Subsystem 01 frame contract:

```text
pose frame = prepared_frame_idx
```

Timing columns should be joined from `prepared_sync.npz` and `prepare_meta.json`
when available. The required join key is `prepared_frame_idx`.

Recommended timing/frame columns:

- `prepared_frame_idx`;
- `raw_decode_frame_idx`;
- `prepared_time_sec`;
- `raw_pts_time_sec`;
- `external_time_sec`;
- `external_time_available`;
- `external_time_source`;
- `prepared_video_fps`;
- `prepared_frame_count`;
- `raw_video_frame_count_opencv_readable`.

If a timing source is absent in Subsystem 01 metadata, the corresponding column
may be null, but the absence must be explicit and machine-readable.

## 10. Pose-Quality QC Scope

The user-facing QC summary in `pose_meta.json` is about pose inference quality
only.

It should include, when available:

- requested and processed frame counts;
- expected animal count;
- animal-count coverage by frame;
- frames with zero animals;
- frames with fewer than expected animals;
- frames with expected animal count;
- frames with extra animals;
- per-node missing keypoint rates;
- per-node low-confidence keypoint rates;
- partial skeleton frequency;
- instance score summaries;
- node score summaries;
- duplicate candidate risk;
- implausible geometry flags;
- tracked and untracked instance counts when tracks are present.

Implausible geometry flags may include impossible body lengths, extreme
inter-node distances, severe skeleton self-crossing, or coordinates outside the
prepared frame when those checks are implemented.

The required Subsystem 02 QC does not evaluate final biological identity
continuity. Identity-stable final tracking is a later workflow.

Pipeline success/provenance fields are not part of pose-quality QC. They belong
in `job_manifest.yaml` and `processing_log.txt`.

## 11. Validation Requirements

Hard validation failures include:

- missing required Subsystem 01 inputs;
- unreadable `prepared_video.mp4`;
- unreadable or inconsistent `prepare_meta.json`;
- unreadable or inconsistent `prepared_sync.npz`;
- inference failure;
- missing or unreadable `pose.slp`;
- failure to export required `pose.parquet`;
- `pose.parquet` row/frame references outside the prepared-frame range;
- inability to join required frame identity data;
- missing required output metadata files.

Reportable but not automatically fatal pose-quality outcomes include:

- zero detected animals in a frame;
- fewer than expected animals;
- extra candidate animals;
- missing keypoints;
- low-confidence keypoints;
- untracked instances;
- overlay generation failure when required pose outputs are valid.

## 12. Completion Criteria

A Subsystem 02 run is complete when:

1. Required Subsystem 01 inputs are validated.
2. `pose.slp` is produced and can be loaded.
3. `pose.parquet` is produced and validates against `pose.slp` and S1 frame
   identity.
4. `pose_meta.json` contains pose-quality QC.
5. `settings_used.yaml` records actual inference parameters.
6. `job_manifest.yaml` records input/output provenance.
7. `processing_log.txt` records runtime logs.
8. `overlay.mp4` is produced, or its failure is recorded as a non-fatal warning
   when allowed.

