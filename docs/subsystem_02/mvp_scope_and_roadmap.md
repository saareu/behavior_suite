# Subsystem 02 — MVP Scope and Roadmap

## MVP Definition

Subsystem 02 is the pose inference and review subsystem. The MVP is a
UI-based workflow that integrates with completed Subsystem 01 preprocessing,
runs SLEAP/SLEAP-NN pose inference, supports review of generated outputs, and
records enough validation and provenance for downstream subsystems.

The MVP must support both bottom-up and top-down SLEAP/SLEAP-NN models. It must
be reachable from the main UI launch point and must support both new inference
runs and review of existing completed Subsystem 02 runs.

## Current Status

Subsystem 02 is under active MVP development. The validated implementation
currently covers the bottom-up backend inference path and the minimal backend
artifact contract. The full MVP is not complete yet.

Implemented backend pieces:

- bottom-up backend inference path;
- minimal artifact generation;
- `pose.parquet` export;
- `overlay.mp4` generation;
- pose-quality QC summary;
- Subsystem 01 timing/frame metadata preservation;
- effective SLEAP provenance capture from `pose.slp` `labels.provenance`.

Not-yet-complete MVP pieces:

- top-down model support;
- Subsystem 02 UI workspace;
- main UI launch and navigation;
- Subsystem 01 completion to Subsystem 02 transition;
- existing Subsystem 02 run discovery, review, reuse, rerun, and downstream
  selection.

## Required MVP Workflow

Subsystem 02 MVP must combine inference, review, and S2-level usability
validation in one workflow. A user should be able to select a prepared
Subsystem 01 result, run pose inference, inspect the output, review quality
signals, and choose an inference run for downstream use without leaving the
main application flow.

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
