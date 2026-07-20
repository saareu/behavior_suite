# Subsystem 02 — Final MVP Scope, Closure Status, and Roadmap

**Status:** MVP finalized and closed to scope expansion. Future changes require
a separately scoped post-MVP task.

## MVP Definition

Subsystem 02 is the pose-inference and technical-QC subsystem. It integrates
with completed Subsystem 01 preprocessing, runs SLEAP/SLEAP-NN pose inference,
and determines whether the technical output can be passed to Subsystem 03.
Final tracking/identity correctness and final session usability are Subsystem
03 responsibilities.

The finalized MVP supports both bottom-up and top-down SLEAP/SLEAP-NN models.
It is reachable from the main UI launch point and supports both new inference
runs and technical inspection of existing completed Subsystem 02 runs.

## Subsystem 02 MVP Completion Status

Subsystem 02 MVP implementation and acceptance are complete. The validation
environment was the supported Windows PySide6 application, an external
SLEAP-NN 0.3.0 GPU inference runtime, and `sleap-io` 0.8.0 in the Behavior Suite
GUI Python environment. No machine-specific path is part of the contract.

Closure evidence confirms:

- GPU acceptance completed;
- bottom-up acceptance completed;
- top-down centroid plus centered-instance acceptance completed;
- the one-click Windows installation completed successfully after the S2
  `sleap-io` dependency fix;
- `pose.slp`, `pose.parquet`, `overlay.mp4`, and all required
  metadata/provenance artifacts were generated successfully;
- S1 frame/timing propagation, technical QC, run discovery, UI run selection,
  and S3 handoff selection completed successfully.

Both accepted inference modes produced QC outcome `pass`.

See
[`evidence/gpu_mvp_acceptance_v030.md`](evidence/gpu_mvp_acceptance_v030.md)
for the recorded acceptance evidence. This finalized status is limited to
inference, artifact integrity, technical QC, and downstream handoff selection;
it is not a claim of final tracking, identity, or scientific usability.

Finalized MVP scope:

- S1-to-S2 workflow integration;
- UI-based pose inference workflow;
- bottom-up SLEAP-NN inference;
- top-down centroid plus centered-instance inference;
- SLEAP-NN 0.3.x execution-interface support;
- generation of `pose.slp`, `pose.parquet`, `overlay.mp4`, and the required
  metadata/provenance artifacts;
- preservation of the S1 timing and frame mapping in pose outputs;
- technical pose QC;
- run discovery;
- S3 handoff selection.

Supporting implementation details include:

- effective SLEAP provenance capture from `pose.slp` `labels.provenance`;
- pre-submission validation of the S1 frame/sync contract;
- post-run technical-QC outcomes and bounded review intervals;
- main-application S1/S2 navigation with preserved session context;
- automatic transition to the same-session S2 screen after successful S1
  completion, without automatic inference submission;
- existing-session browsing and metadata-only run listing/details;
- bottom-up and top-down profile-driven configuration and asynchronous
  submission;
- overlay/folder actions, settings-copy rerun support, and transient technically
  complete run selection for the S3 handoff interface;
- QSettings convenience persistence for existing model/profile/runtime paths;
- validation-only authoritative backend preflight, which reports the S1
  handoff, model/profile consistency, resolved SLEAP-NN executable/version, and
  prospective command without creating a run directory or launching inference;
- generation-token protection for asynchronous progress/results and action-time
  revalidation of the selected S3 handoff artifacts and QC outcome.

Deferred non-blocking enhancements:

- percentage progress and live subprocess-log streaming beyond the implemented
  validating/inference/export/QC/overlay stage callbacks;
- safe active-subprocess cancellation. Cancellation remains omitted because the
  backend still executes one blocking pipeline call and force termination
  would not reliably preserve terminal metadata/logging.

The broader future-feature list below records additional post-MVP improvements.
None are missing MVP requirements.

## Finalized MVP Workflow

Subsystem 02 MVP combines UI-based inference and technical pose-inference QC.
It does not create a persistent user-review decision or final-usability record.
The implemented UI exposes run metadata, review intervals, and overlay/folder
actions for technical inspection; expanded pose-review tooling and richer QC
visualization are deferred and are not MVP requirements.

The workflow has two supported entry modes:

1. Continue from a newly completed Subsystem 01 preprocessing run.
2. Open an existing project or session that already contains completed
   Subsystem 02 inference runs for technical inspection, reuse, rerun, or
   downstream selection.

The main UI exposes Subsystem 02 from the same launch/navigation surface as
Subsystem 01; S2 is not a hidden command-line-only workflow.

The implemented desktop workspace uses run discovery metadata only for listing
and selection; it does not load `pose.slp` or `pose.parquet`. It shows S1
handoff/frame/timing context, existing-run status/artifacts/QC/review intervals,
profile-driven bottom-up or top-down configuration, and a copyable technical
details panel. The authoritative backend preflight and full run execute in the
existing Qt worker abstraction with coarse stage callbacks and an indeterminate
progress indicator.

`review_recommended` is displayed as successful and optionally reviewable, not
failed. Both `pass` and `review_recommended` technically complete runs can form
a transient S3 navigation handoff. Failed/incomplete/missing-artifact runs
cannot. No permanent downstream-approval artifact is written.

## Backend Contract

The backend contract is defined in
[`sleap_inference_specification.md`](sleap_inference_specification.md). The
required output directory remains:

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

Subsystem 01 remains the source of truth for frame identity, timing, crop
geometry, prepared-video dimensions, and preprocessing provenance. Subsystem 02
must preserve that contract into its artifacts.

## SLEAP Provenance

Raw SLEAP/SLEAP-NN startup stdout may report construction-stage defaults for
checkpoint models. Effective prediction-stage settings should be read from
`pose.slp` `labels.provenance` after inference completes.

Raw stdout/stderr is diagnostic only. Subsystem 02 metadata should use
`labels.provenance` as the effective SLEAP provenance source when available.

## Technical-QC Policy

Preflight validation occurs before inference submission and validates the
readable S1 handoff, the prepared-video frame count, authoritative sync arrays
and prepared-frame mapping, model/profile inputs, runtime compatibility, and
prospective command. It creates no run directory. Run execution creates the
output directory and reports an output-location failure before launching the
inference subprocess. Missing S2 outputs are post-run failures, not preflight
failures.

Post-run technical validation is distinct from pose-quality review warnings.
Technical failures, including zero represented prepared frames containing at
least one finite x/y point, produce QC outcome `failed`. Valid output produces
`pass` unless one of two deliberately conservative technical-review thresholds
is reached:

- fraction of represented frames with exactly one detected animal, for the
  configured two-animal workflow;
- fraction of pose rows with a missing x/y coordinate pair.

Both global thresholds default to `0.90`. A `review_recommended` outcome is
non-blocking, does not make a successful run fail, and does not prevent S3
handoff. Bounded, longest-first contiguous frame intervals identify where each
trigger is concentrated and can guide optional overlay seeking.

This technical QC validates inference execution and artifact integrity, detects
extreme abnormal failures, and may recommend review. It does not perform final
tracking validation, identity verification, or scientific-usability assessment.

## Intentional Non-Scope

The following are downstream responsibilities, not Subsystem 02 MVP blockers or
limitations:

- elaborate pose-review workspace;
- model-optimization UI;
- active learning;
- pose correction;
- identity verification;
- tracking correction;
- final pose processing;
- final biological identity assignment;
- final tracking verification;
- implanted/partner mouse assignment;
- identity-switch correction;
- imputation;
- pose smoothing/finalization;
- final-analysis-ready trajectory generation;
- behavior-ready feature extraction.

## Future Non-Blocking Features

These features may improve later workflows but are not MVP blockers:

- UI-assisted model and parameter optimization;
- expanded pose-quality review tools;
- richer QC visualization.
