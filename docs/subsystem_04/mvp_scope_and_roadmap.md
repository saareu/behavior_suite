# Subsystem 04 — Pose Finalization MVP Scope and Roadmap

**Status:** Sprint 1 implemented (contracts, accepted-S3 discovery, timing map).
Full MVP processing, export, and GUI are not yet implemented.

## Agreed S4 boundary

Subsystem 04 consumes an **explicitly accepted** Subsystem 03 result and
produces an analysis-oriented dense pose timeline on a uniform lab clock.

| Owns | Does not own |
| --- | --- |
| Accepted-S3 validation for handoff | Tracking identity reassignment |
| Authoritative S1 timing → 120 Hz mapping | Pose-model training |
| Dense pose construction / gap fill | S3 review or acceptance UX |
| Geometric refinement / smoothing | Mutation of S1/S2/S3 artifacts |
| Final QC of finalized pose | Behavior classification |

S3 `complete_reviewable` means automatic correction finished and review is
possible. It is **never** equivalent to acceptance for S4.

## Input contract

Primary numeric input:

```text
<session_root>/tracking_correction/<s3_run_id>/tracked_pose.parquet
```

Eligibility requires all of:

1. `run_meta.json` with `acceptance_state == "accepted"`;
2. present `tracked_pose.parquet`;
3. recorded acceptance SHA256 matching the on-disk accepted pose;
4. resolvable authoritative S1 `preprocess/prepared_sync.npz`.

Rejected: missing, `not_accepted`, `superseded`, SHA mismatch, malformed meta,
or reviewable-but-unaccepted runs.

Authoritative timing source:

```text
<session_root>/preprocess/prepared_sync.npz
```

using `prepared_time_sec` and `prepared_frame_idx` (not FPS-nearest video
indexing, and not Pipeline 30-style mapping).

## Output contract (agreed; not written in Sprint 1)

```text
<session_root>/pose_finalization/<run_id>/
  final_pose.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

Parquet is the only final pose format for the MVP. No optional NPZ outputs.

## Processing sequence (Gates 1–4)

```text
Accepted S3 tracked_pose.parquet
    + S1 prepared_sync.npz
    ↓
Validate acceptance / SHA / paths
    ↓
Build uniform 120 Hz target timeline from S1 prepared times
    ↓
Map each target sample → left/right source frames + weights
    ↓
(later) Dense pose / gap fill / refinement / smoothing / QC
    ↓
Write final_pose.parquet + run metadata/log
```

Sprint 1 implements through the timing-mapping step only.

## Completion semantics

- S4 MVP completion is structural/process completion with warnings recorded in
  run metadata (S2-style), **not** a second user acceptance gate.
- Upstream S1/S2/S3 artifacts remain read-only.

## Sprint 1 scope and status

**Implemented**

- `src/pose_finalization/contracts.py` — artifact names, target FPS default
  (`120.0`), required input field names, minimal status constants.
- `src/pose_finalization/accepted_s3.py` — accepted-S3 validate/discover/resolve
  helpers using S3 `is_currently_accepted` and acceptance SHA metadata.
- `src/pose_finalization/timing.py` — load S1 timing; uniform 120 Hz timeline;
  deterministic left/right indices, interpolation weights, and exact-vs-derived
  flags.
- Focused unit tests under `tests/pose_finalization/`.

**Not implemented yet**

- Full S4 runner / CLI / GUI
- Coordinate interpolation / dense pose construction
- Gap filling, Kalman/RTS, geometric refinement, Savitzky–Golay
- Final QC and `final_pose.parquet` export
- Implanted-nose special imputation

## Remaining implementation stages

1. **Sprint 2+:** dense pose construction on the mapped 120 Hz timeline, gap
   handling, and provenance flags.
2. **Later MVP stages:** refinement, smoothing, QC summaries, runner writing the
   agreed run directory, then GUI handoff from accepted S3.
3. **Post-MVP:** additional profiles/backends only if required by lab needs.

## Timing notes

- Target rate default is exactly `120.0` Hz.
- Source recordings may have `fps_header` near but not equal to 120 (common in
  real sessions); mapping uses S1 `prepared_time_sec`, not an assumed 120 Hz
  source clock.
- Valid S1 `prepared_sync` timelines start at time 0 by S1 contract. The mapping
  helpers also accept a nonzero origin for general monotonic arrays; that case
  is not produced by current S1 validation.
