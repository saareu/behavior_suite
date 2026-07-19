# Subsystem 02 — MVP Scope and Roadmap

## MVP Definition

Subsystem 02 is the pose-inference and technical-QC subsystem. It integrates
with completed Subsystem 01 preprocessing, runs SLEAP/SLEAP-NN pose inference,
and determines whether the technical output can be passed to Subsystem 03.
Final tracking/identity correctness and final session usability are Subsystem
03 responsibilities.

The MVP must support both bottom-up and top-down SLEAP/SLEAP-NN models. It must
be reachable from the main UI launch point and must support both new inference
runs and review of existing completed Subsystem 02 runs.

## Current Status

Subsystem 02 is under active MVP development. The implementation covers both
bottom-up and top-down backend inference paths and the minimal backend artifact
contract. Bottom-up has passed a real GPU smoke test; top-down has passed a
real GPU smoke test using a centroid plus centered-instance bundle
and SLEAP-NN 0.3.0. The full MVP is not complete yet.

Implemented backend pieces:

- bottom-up backend inference path;
- top-down centroid plus centered-instance model-bundle path;
- minimal artifact generation;
- `pose.parquet` export;
- `overlay.mp4` generation;
- pose-quality QC summary;
- Subsystem 01 timing/frame metadata preservation;
- effective SLEAP provenance capture from `pose.slp` `labels.provenance`.
- pre-submission validation of the S1 frame/sync contract;
- post-run technical-QC outcomes and bounded review intervals;
- existing-run discovery.

Not-yet-complete MVP pieces:

- Subsystem 02 UI workspace;
- main UI launch and navigation;
- Subsystem 01 completion to Subsystem 02 transition;
- existing-run review, reuse, rerun, and downstream selection UI.

## Required MVP Workflow

Subsystem 02 MVP combines inference and technical pose-inference QC. It does
not create a persistent user-review decision or final-usability record. A
future UI may expose the overlay and flagged intervals for optional inspection,
but a dedicated elaborate S2 review screen is not required for MVP.

The workflow has two required entry modes:

1. Continue from a newly completed Subsystem 01 preprocessing run.
2. Open an existing project or session that already contains completed
   Subsystem 02 inference runs for review, reuse, rerun, or downstream
   selection.

The main UI must expose Subsystem 02 from the same launch/navigation surface as
Subsystem 01. Subsystem 02 should not be a separate hidden command-line-only
workflow for the MVP.

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
and prepared-frame mapping, model/profile inputs, and writable output
location. Missing S2 outputs are post-run failures, not preflight failures.

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

## Intentional Non-Scope

The following are downstream responsibilities, not Subsystem 02 MVP blockers or
limitations:

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
- expanded pose-quality review tools.
