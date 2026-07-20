# Subsystem 02 — Backend Pose Inference Contract

## 1. Purpose

This document defines the backend pose inference contract for Subsystem 02. It
covers how the backend consumes completed Subsystem 01 outputs, runs
SLEAP/SLEAP-NN inference, and writes the locked pose inference artifact set.

This document is not the full Subsystem 02 MVP scope. The PySide6 application
now provides the UI-based inference/review workflow, existing-run review,
Subsystem 01 completion-to-Subsystem 02 transition, and main UI integration.
See
[`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).

The backend supports bottom-up SLEAP/SLEAP-NN inference and a top-down bundle
containing centroid and centered-instance checkpoints. Both modes completed the
first full real-GPU S2 MVP acceptance workflow using SLEAP-NN 0.3.0 and
`sleap-io` 0.8.0. The complete artifact set, provenance extraction, S1 timing
propagation, technical QC, discovery, UI selection, and S3 handoff were
verified with QC outcome `pass`.
See
[`evidence/gpu_mvp_acceptance_v030.md`](evidence/gpu_mvp_acceptance_v030.md).

The backend creates one native pose result, one analysis-ready pose table, one
visual overlay, and the minimal metadata/provenance needed to reproduce and
audit the run. It does not change preprocessing behavior. Subsystem 01 remains
the source of truth for frame identity, timing, crop geometry, prepared-video
dimensions, and preprocessing provenance.

## 2. Scope

The backend pose inference contract is responsible for:

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
- reporting technical pose-inference QC and non-blocking review recommendations.

The backend pose inference contract is not responsible for:

- modifying Subsystem 01 artifacts;
- redefining frame timing or crop geometry;
- custom candidate selection after inference;
- separate tracking post-processing artifacts;
- final long-term biological identity continuity;
- final tracking/identity correctness or final session usability;
- a persistent S2 user-review or final-usability decision;
- parameter optimization or guided hyperparameter search;
- behavior classification or downstream biological analysis;
- final identity review, scientific-usability approval, or S3 processing.

Parameter optimization is postponed to a later guided workflow. The
implementation uses one explicit default profile per supported inference mode.

### Desktop integration

The main PySide6 application contains persistent S1 and S2 screens. Successful
S1 completion opens S2 with the same project/session but does not start
inference. A user can also enter S2 through subsystem navigation or browse an
existing session directly. Back navigation preserves the selected session.

S2 calls `summarize_pose_inference_project(...)` to populate newest-first run
rows and the technical-details panel without opening large pose artifacts. A
selected technically complete run with QC outcome `pass` or
`review_recommended` may be represented as a transient S3 input containing the
session/run and locked artifact paths. This is navigation input selection, not
identity correctness or final usability approval. The real-GPU MVP acceptance
workflow verified this handoff from a selected completed S2 run into the S3
interface.

New runs use the typed `PoseInferenceRequest` and `PoseInferenceModelSpec` APIs.
Only basic mode/field presence is checked in the UI; backend preflight remains
authoritative for the S1 contract, model roles, profile/mode agreement, output
location, and SLEAP-NN 0.3.x runtime. The full backend call runs on the existing
Qt worker thread. The backend reports validating, inference, Parquet export,
QC, overlay, and terminal stage transitions plus the run directory, but no
reliable percentage. Because it provides no safe cancellation hook or live
subprocess-log streaming, the UI uses indeterminate activity, exposes the final
log after return, disables navigation while active, and does not offer
cancellation.

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

### Model specification and CLI

Model selection never relies on an undocumented positional meaning. A
bottom-up specification contains `inference_mode: bottomup` and one model
path. A top-down specification contains `inference_mode: topdown`, one centroid
model path, and one centered-instance model path. Top-down preflight requires
both paths to exist and be readable, requires distinct resolved paths, and
reads `training_config.yaml` or `training_config.json` to reconcile the active
`model_config.head_configs` role with the requested component. Missing or
unreconcilable roles and profile/mode conflicts fail before SLEAP submission.

Behavior Suite CLI examples are:

```powershell
python -m pose_inference run --session-root SESSION `
  --inference-mode bottomup --model-path BOTTOMUP_MODEL

python -m pose_inference run --session-root SESSION `
  --inference-mode topdown `
  --centroid-model-path CENTROID_MODEL `
  --centered-instance-model-path CENTERED_INSTANCE_MODEL
```

For compatibility, omitting `--inference-mode` while supplying only
`--model-path` selects bottom-up. Ambiguous bottom-up/top-down flag mixtures
and incomplete top-down bundles are rejected. If `--profile` is omitted, the
mode selects `sleapnn_default_profile.yaml` or
`sleapnn_topdown_default_profile.yaml`.

`--sleap-executable PATH` may explicitly select the runtime and takes
precedence over a profile `sleap_executable` value, a runtime beside the active
Python executable, and PATH. The resolved executable must report SLEAP-NN
0.3.x; other interfaces fail preflight before a run directory is created.

The project’s captured SLEAP-NN 0.3.0 `predict --help` evidence documents a
repeatable `--model_paths` option. The inspected SLEAP-NN checkpoint predictor
implementation loads each training config, identifies its active head, and
selects paths by `centroid` and `centered_instance` role, so semantic model
order does not matter. Behavior Suite nevertheless emits centroid first and
centered-instance second for deterministic audit logs. The generated command
shapes are:

The separately configured developer runtime inspected during implementation is
SLEAP-NN 0.2.0. In that version, the checkpoint-oriented equivalent is exposed
as `sleap-nn track`, while `predict` has an export-runtime interface. Behavior
Suite deliberately retains the already GPU-validated project command builder
for the captured 0.3.x `predict` interface and rejects incompatible 0.2.x
executables during preflight. The external executable is resolved in this
order: an explicit CLI/profile path, a `sleap-nn.exe` or `sleap-nn` sibling of
the active Python executable, then PATH. Its absolute path and queried version
are recorded in run metadata and logs.

Inference execution and artifact post-processing have separate dependency
boundaries. The resolved external `sleap-nn` 0.3.x executable performs model
inference and writes `pose.slp`. Parquet export and effective provenance
extraction run in the Behavior Suite GUI Python process, which therefore
provides the pinned `sleap-io` 0.8.0 runtime dependency. The resulting table is
then consumed by the shared technical-QC and overlay artifact pipeline. The
Windows one-click installer installs the repository's `s2` optional dependency
and validates these imports before reporting a usable GUI runtime.

```text
SLEAP_EXECUTABLE predict --data_path PREPARED_VIDEO --model_paths BOTTOMUP_MODEL \
  --output_path POSE_SLP --output_format slp --device cuda --batch_size 4 \
  --max_instances 4 --peak_threshold 0.05 --integral_refinement integral \
  --integral_patch_size 5 --max_edge_length_ratio 0.25 \
  --dist_penalty_weight 1.0 --n_points 10 --min_line_scores 0.25 \
  --tracking --candidates_method local_queues --max_tracks 2 \
  --tracking_window_size 240 --min_new_track_points 1 --min_match_points 1 \
  --track_matching_method hungarian --features keypoints \
  --scoring_method oks --scoring_reduction robust_quantile \
  --robust_best_instance 0.95

SLEAP_EXECUTABLE predict --data_path PREPARED_VIDEO --model_paths CENTROID_MODEL \
  --model_paths CENTERED_INSTANCE_MODEL --output_path POSE_SLP \
  --output_format slp --device cuda --batch_size 4 --max_instances 4 \
  --peak_threshold 0.05 --integral_refinement integral \
  --integral_patch_size 5 --tracking --candidates_method local_queues \
  --max_tracks 2 --tracking_window_size 240 --min_new_track_points 1 \
  --min_match_points 1 --track_matching_method hungarian \
  --features keypoints --scoring_method oks \
  --scoring_reduction robust_quantile --robust_best_instance 0.95
```

Device, batch size, `max_instances`, peak/integral settings, and tracking are
shared by both modes in the verified CLI. The CLI also exposes top-down
facilities such as centroid-stage peak overrides and crop sizing, but the
default profile does not set those optional values. Bottom-up PAF grouping
options remain exclusive to the bottom-up profile. `max_instances` bounds
top-down instance consideration, while tracking operates on the resulting
predicted instances inside the same inference call.

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
pose_inference/{model-id}__{timestamp}/
├── pose.slp
├── pose.parquet
├── overlay.mp4
├── pose_meta.json
├── settings_used.yaml
├── job_manifest.yaml
└── processing_log.txt
```

No other standard artifacts are part of the locked Subsystem 02 contract.

Top-down run IDs use
`topdown-{centroid-name}-{centered-instance-name}-{path-hash}`. Names are
sanitized and bounded, and the deterministic hash always incorporates both
resolved component paths in role order. Existing legacy run names remain
discoverable.

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

`overlay.mp4` is a required S2 output. A missing, unreadable, or failed overlay
is a post-run technical failure under the locked artifact contract.

### `pose_meta.json`

`pose_meta.json` contains machine-readable run metadata and a technical
pose-inference QC summary.

It records `inference_mode`. Bottom-up retains the legacy `model_id` and
`model_path`; top-down retains a stable bundle id plus separate centroid and
centered-instance ids and paths.

The technical QC section is limited to inference/artifact integrity and
conservative review indicators. Pipeline success, dispatch provenance, and file
provenance belong in `job_manifest.yaml` and `processing_log.txt`.

When available, `pose_meta.json` should also include compact effective SLEAP
provenance copied from `pose.slp` `labels.provenance`.

### `settings_used.yaml`

`settings_used.yaml` records the actual SLEAP/SLEAP-NN parameters used for the
run, including:

- model id;
- inference mode and separate component identities for top-down bundles;
- runtime profile id;
- execution provider and device;
- inference profile;
- tracking enabled/disabled;
- all effective SLEAP/SLEAP-NN inference parameters.

Raw SLEAP/SLEAP-NN startup logs may report Predictor construction defaults for
checkpoint models. Effective prediction-stage settings should be read from
`pose.slp` `labels.provenance` after inference completes.

### `job_manifest.yaml`

`job_manifest.yaml` records the input/output contract and provenance:

- Subsystem 01 input artifact paths and fingerprints;
- output artifact paths and fingerprints;
- model path or separate top-down component paths and model metadata
  fingerprints;
- command or structured invocation record;
- effective SLEAP inference and tracking provenance copied from `pose.slp`
  `labels.provenance` when available;
- run start/end timestamps;
- run status and warning/failure summary.

### `processing_log.txt`

`processing_log.txt` records runtime logs, command output, warnings, errors,
and validation messages useful for debugging and audit.

Raw stdout/stderr is diagnostic only. The effective SLEAP prediction-stage
settings source is `pose.slp` `labels.provenance` when available.
That provenance is copied without inferring settings from startup text and may
include the effective model paths/model type, inference configuration, tracking
configuration, device/system information, and SLEAP-NN/SLEAP-IO versions for
the two-stage run.

## 7. Default Inference Profiles

The implementation has explicit bottom-up and top-down default profiles. Shared
device, batching, maximum-instance, expected-animal, technical-QC, and
provisional-tracking settings are aligned. The bottom-up profile alone contains
bottom-up PAF grouping options. Each profile declares `inference_mode`, and a
conflict with the selected model specification is a preflight failure.

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

## 10. Technical Pose-Inference QC Scope

The QC summary in `pose_meta.json` is technical pose-inference QC. It retains
the processing status (for example, `status: computed`) and records a separate
outcome:

```text
pass | review_recommended | failed
```

`review_recommended` is non-blocking: it does not make the run unsuccessful
and does not prevent the output from being passed to S3.

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

The MVP review recommendation has exactly two triggers:

1. For the configured two-animal workflow, the fraction of represented frames
   with exactly one detected animal is at least the configured threshold.
2. The fraction of pose rows whose x/y coordinate pair is not finite is at
   least the configured threshold.

Other informational pose observations are preserved in the
`diagnostic_findings` list. They do not determine `review_recommended`; the MVP
review scope is represented separately by `review_recommendation_reasons`,
`review_warnings`, and `flagged_intervals`.

The global defaults are `0.90` for both metrics. These are conservative
technical-review thresholds, not scientific-validity criteria. Sparse
detections, many one-animal frames below the extreme threshold, high but
sub-threshold missing-keypoint rates, partial-skeleton runs, low-confidence
periods, unexpected animal-count distributions, identity switches, and
tracking continuity do not independently trigger an MVP review recommendation
or hard failure.

The profile fields are:

```yaml
expected_animals: 2
review_one_animal_fraction_threshold: 0.90
review_missing_keypoint_fraction_threshold: 0.90
review_frame_missing_keypoint_fraction_threshold: 0.90
```

For each triggered warning, metadata retains at most the 10 longest contiguous
prepared-frame intervals, sorted longest-first. Interval `start_frame` and
`end_frame` are inclusive. Frame timestamps are copied when available;
`time_span_sec` is the end-frame timestamp minus the start-frame timestamp.
It is a timestamp span rather than an inclusive media duration, so a one-frame
interval may have a zero span. Intervals are generated independently per
`video_index` and include that index in each record.
The per-frame high-missing interval threshold also defaults to `0.90`, and a
frame is included when its missing-keypoint fraction is at least
that threshold. These intervals guide optional overlay review; no clips or
elaborate S2 QC dashboard are required.

A hard QC failure occurs when:

```text
number of represented prepared frames containing at least one finite x/y pose point == 0
```

The required Subsystem 02 QC does not decide final biological identity,
tracking correctness, or final session usability. It validates inference
execution and artifact integrity, detects extreme abnormal failures, and may
recommend review; it does not replace tracking validation, identity
verification, or scientific-usability assessment. Those are S3
responsibilities.

Pipeline success/provenance fields are not part of technical pose-inference QC.
They belong in `job_manifest.yaml` and `processing_log.txt`.

## 11. Validation Requirements

Pre-submission validation occurs before the inference subprocess is submitted.
It requires readable `prepared_video.mp4`, `prepare_meta.json`, and
`prepared_sync.npz`; validates the S1 sync array lengths and prepared-frame
mapping; opens the prepared video and requires one readable first frame; and,
when OpenCV reports a positive header frame count, reconciles that header count
with authoritative S1 metadata. S2 does not repeat S1's sequential full-video
decode. Preflight also validates model-bundle completeness, readability,
distinct top-down component paths, structurally identifiable model roles, and
profile/mode agreement, then confirms that the output run directory can be
created. These are preflight failures. Missing S2 outputs are
not called preflight failures because they are expected to be absent before
inference.

Authoritative S1 contract parsing and validation currently occurs in more than
one S2 layer and may be centralized in a later refactor.

Post-run technical failures include:

- inference subprocess failure;
- an expected required S2 output that is missing or unreadable;
- prediction frame indices/count that cannot be reconciled with the
  authoritative S1 prepared-frame contract;
- invalid frame/timing mapping introduced in `pose.parquet`;
- zero represented prepared frames containing at least one finite x/y pose
  point.

Per-frame zero animals, sparse detections, moderate one-animal fractions,
moderate missing-keypoint fractions, partial skeletons, and low-confidence
points remain diagnostic observations rather than hard failures.

## 12. Completion Criteria

A Subsystem 02 run is complete when:

1. Required Subsystem 01 inputs are validated.
2. `pose.slp` is produced and can be loaded.
3. `pose.parquet` is produced and validates against `pose.slp` and S1 frame
   identity.
4. `pose_meta.json` contains technical pose-inference QC.
5. `settings_used.yaml` records actual inference parameters.
6. `job_manifest.yaml` records input/output provenance.
7. `processing_log.txt` records runtime logs.
8. Required `overlay.mp4` is produced and readable.

The integrated S2 MVP workflow additionally requires UI-based submission and
review, run discovery and selection, and a downstream S3 handoff interface.
These were verified for both bottom-up and top-down runs in the recorded full
GPU acceptance. UI-assisted model/parameter optimization, expanded pose-quality
review tools, and richer QC visualization remain future enhancements, not
completion requirements for this contract.
