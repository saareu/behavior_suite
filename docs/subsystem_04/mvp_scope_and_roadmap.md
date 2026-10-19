# Subsystem 04 — Pose Finalization MVP Scope and Roadmap

**Status:** Sprint 2 implemented (dense 120 Hz construction, manual-blank
protection, configurable gap fill, provenance). Final export, SavGol smoothing,
QC, runner, and GUI are not yet implemented.

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

Manual blank protection source (Sprint 2):

```text
<session_root>/tracking_correction/<s3_run_id>/manual_corrections.json
```

Only records with `action == "blank_node"` are protected. The S3 parquet flag
`was_node_blanked` is **not** treated as manual protection because automatic
correction also sets it (explicit blanked-node tracking and implicit
raw-finite / final-NaN detection).

## Output contract (agreed; not written in Sprint 1–2)

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
    + manual_corrections.json (blank_node only)
    ↓
Validate acceptance / SHA / paths
    ↓
Build uniform 120 Hz target timeline from S1 prepared times
    ↓
Map each target sample → left/right source frames + weights
    ↓
Construct dense (sample, track, node) coordinates
    ↓
Propagate protected manual blanks; temporal resample; gap fill
    ↓
(later) SavGol / refinement / QC / final_pose.parquet export
```

Sprint 2 implements dense construction through gap fill in memory only.

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

## Sprint 2 scope and status

**Implemented**

- `src/pose_finalization/dense_pose.py` — load accepted tracked pose; build dense
  `(sample, track, node)` arrays from the Sprint 1 `TimingMap`; preserve track
  IDs and SLEAP node names; map with actual S1 timestamps; retain unresolved
  samples as NaN; never extrapolate beyond observed source times; never mutate
  upstream files.
- Temporal resampling provenance:
  - exact source transfer → `original` or `identity_corrected`;
  - derived sample between two valid endpoints → `temporal_resample`;
  - missing / non-interpolable → `missing`.
  Missing source observations are never used as interpolation endpoints.
  Temporally resampled samples do **not** receive fabricated SLEAP confidence.
- Identity-corrected provenance combines:
  - automatic S3 `was_track_swapped` on the accepted pose row; and
  - active manual stack records in `manual_corrections.json`
    (`swap_identities` over an interval; `swap_node` for one node on both
    tracks at one frame).
  S3 undo/reset rewrite that file to the current stack only, so undone edits
  are ignored. Overlapping manual identity swaps use parity (even coverage
  cancels). Coordinates and track labels are never rewritten by S4.
- Manual blank protection:
  - authoritative intervals from `manual_corrections.json` `blank_node` records;
  - mask built on the prepared-frame domain and propagated conservatively to
    120 Hz (any contributing left/right source blank protects the target);
  - protected samples remain NaN through temporal resampling and gap fill;
  - `was_node_blanked` alone does not protect.
- `src/pose_finalization/gap_fill.py` — configurable internal gap fill:

  | Gap length at 120 Hz | Method |
  | --- | --- |
  | 1–20 frames | Linear interpolation |
  | 21–140 frames | Constant-velocity Kalman + RTS |
  | >140 frames | Remain NaN |
  | Leading/trailing gaps | Remain NaN |
  | Protected/manual-blank intervals | Remain NaN |

  Thresholds live in `GapFillConfig` (`linear_max_frames`, `kalman_max_frames`)
  for future GUI Advanced Settings. Headstage is excluded from imputation.
  Existing valid coordinates are preserved; no identity reassignment; no
  implanted-nose special model.
- In-memory provenance codes and `x_pre_impute` / `y_pre_impute` retained for
  later SavGol / export stages.
- Focused unit + schema-fixture tests in `tests/pose_finalization/test_dense_pose.py`.

**Not implemented yet**

- Savitzky–Golay smoothing
- Final `final_pose.parquet` export / QC / completion semantics
- S4 GUI or runner integration
- New acceptance workflow
- Special nose/headstage imputation model
- Identity correction (owned by S3)
- NPZ output
- General plugin/profile architecture

## Remaining implementation stages

1. **Later MVP stages:** SavGol / geometric refinement, QC summaries, runner
   writing the agreed run directory, then GUI handoff from accepted S3.
2. **Post-MVP:** additional profiles/backends only if required by lab needs.

## Timing notes

- Target rate default is exactly `120.0` Hz.
- Source recordings may have `fps_header` near but not equal to 120 (common in
  real sessions); mapping uses S1 `prepared_time_sec`, not an assumed 120 Hz
  source clock.
- Valid S1 `prepared_sync` timelines start at time 0 by S1 contract. The mapping
  helpers also accept a nonzero origin for general monotonic arrays; that case
  is not produced by current S1 validation.

## Manual-blanking notes (Sprint 2)

- S3 `apply_blank_node` blanks a single `(frame, track, node)` and appends a
  `blank_node` record with `start_frame == end_frame`.
- Automatic corrector blanking also sets `was_node_blanked` and may omit or NaN
  coordinates without a manual record. S4 treats those as ordinary missing
  observations (fillable when gap policy allows).
- Conservative policy: if either timing-map endpoint is a manually blanked
  source sample, the 120 Hz target is protected and left NaN.

## Provenance limitations (Sprint 2)

- Automatic `was_track_swapped` is frame-granular in S3 finalize: a single-node
  automatic swap marks every node/track row at that frame.
- Temporal resamples are always labeled `temporal_resample`, even when one or
  both source endpoints were identity-corrected.
- Manual identity provenance follows the accepted active stack + automatic
  flag only; S4 does not reconstruct undo history beyond what remains on disk.
