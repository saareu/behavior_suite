# Subsystem 02 Minimal Pose Inference Contract

## MVP Acceptance Status

The S2 MVP implementation has completed its first full real-GPU acceptance
workflow. Bottom-up inference and top-down inference with a centroid plus
centered-instance bundle both ran through the UI using SLEAP-NN 0.3.0 and
`sleap-io` 0.8.0. Both modes generated the complete locked artifact set,
preserved S1 timing, extracted SLEAP provenance, computed technical QC with
outcome `pass`, and were discovered in S2. Completed runs were selected in the
UI, and a selected completed run was handed to S3. See
[`evidence/gpu_mvp_acceptance_v030.md`](evidence/gpu_mvp_acceptance_v030.md).

This is technical MVP acceptance, not a determination of final identity,
tracking usability, or scientific usability.

## Inputs

Subsystem 02 consumes the completed Subsystem 01 prepared outputs:

```text
preprocess/prepared_video.mp4
preprocess/prepare_meta.json
preprocess/prepared_sync.npz
```

Subsystem 01 remains the source of truth for frame indices, timing, crop
geometry, prepared-video metadata, and preprocessing provenance.

The inference model selection is explicit. Bottom-up uses one model path.
Top-down uses a complete bundle containing distinct centroid and
centered-instance model paths. Both modes use the same output contract and the
same S1 timing, Parquet, technical-QC, and overlay pipeline.

## Outputs

Each Subsystem 02 run writes:

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

## Locked Decisions

- `pose.slp` is the only native SLEAP/SLEAP-NN output.
- Tracking, when enabled, is performed inside the SLEAP/SLEAP-NN inference call.
- Track assignments are stored inside `pose.slp`.
- No separate tracking artifacts are required.
- `pose.parquet` is the analysis-ready table integrating pose, frame indices,
  S1 timing, and relevant frame-level metadata.
- `overlay.mp4` is generated from `pose.slp`; it may color by track when tracks
  are present.
- `pose_meta.json` contains technical pose-inference QC, including a separate
  `pass`, `review_recommended`, or `failed` outcome, not pipeline provenance.
- Technical QC validates inference execution and artifact integrity, detects
  extreme abnormal failures, and may recommend review. It does not replace
  tracking validation, identity verification, or scientific-usability
  assessment.
- `settings_used.yaml` records actual inference parameters.
- Model metadata records `inference_mode`; top-down metadata retains separate
  centroid and centered-instance paths and stable component identifiers.
- Run metadata records the absolute external SLEAP-NN executable and its
  queried version. The supported execution interface is currently SLEAP-NN
  0.3.x `predict`.
- `job_manifest.yaml` records input/output contract and provenance.
- `processing_log.txt` records runtime logs.

## Out of Scope

- Final long-term biological identity continuity.
- Separate tracked `.slp` files.
- Tracking reports, tracking QC CSVs, or identity maps as standard outputs.
- Parameter optimization and guided profile tuning.
- Expanded pose-quality review tools and richer QC visualization.
- Any change to Subsystem 01 preprocessing behavior.

Subsystem 02 decides technical completion and S3 handoff eligibility. A
`review_recommended` result is non-blocking. Final tracking/identity correctness
and final session usability are Subsystem 03 responsibilities; Subsystem 02
does not persist those decisions.
