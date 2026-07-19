# Subsystem 02 Minimal Pose Inference Contract

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
- Any change to Subsystem 01 preprocessing behavior.

Subsystem 02 decides technical completion and S3 handoff eligibility. A
`review_recommended` result is non-blocking. Final tracking/identity correctness
and final session usability are Subsystem 03 responsibilities; Subsystem 02
does not persist those decisions.
