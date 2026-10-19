# Subsystem 04 — Pose Finalization MVP Scope and Roadmap

**Status:** Sprint 3 implemented (SavGol smoothing, final Parquet export,
automated QC, end-to-end runner). GUI handoff and manual S4 review remain
out of scope.

## Agreed S4 boundary

Subsystem 04 consumes an **explicitly accepted** Subsystem 03 result and
produces an analysis-oriented dense pose timeline on a uniform lab clock.

| Owns | Does not own |
| --- | --- |
| Accepted-S3 validation for handoff | Tracking identity reassignment |
| Authoritative S1 timing → 120 Hz mapping | Pose-model training |
| Dense pose construction / gap fill | S3 review or acceptance UX |
| SavGol smoothing / final QC / export | Mutation of S1/S2/S3 artifacts |
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

Manual blank protection source:

```text
<session_root>/tracking_correction/<s3_run_id>/manual_corrections.json
```

Only records with `action == "blank_node"` are protected. The S3 parquet flag
`was_node_blanked` is **not** treated as manual protection because automatic
correction also sets it.

## Output contract

```text
<session_root>/pose_finalization/<run_id>/
  final_pose.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
```

Parquet is the only final pose format for the MVP. No NPZ outputs.

### `final_pose.parquet` schema (`final_pose_v1`)

Long-form: one row per `(sample_idx, track, node)`.

| Field | Meaning |
| --- | --- |
| `schema_version` | Always `final_pose_v1` |
| `sample_idx` | Dense 120 Hz target sample index |
| `time_sec` | Target timestamp on the uniform lab clock |
| `track` | Track identity (integer) |
| `node` | Skeleton node name |
| `x`, `y` | Final coordinates after gap fill + optional SavGol |
| `x_pre_smooth`, `y_pre_smooth` | Coordinates after gap fill, before smoothing |
| `point_score`, `instance_score` | Original SLEAP confidence where applicable; NaN for derived samples |
| `provenance` | Uint8 provenance code (see below) |
| `provenance_name` | Human-readable provenance label |
| `protected_manual_blank` | True when the sample is a protected manual blank |
| `was_smoothed` | True when SavGol modified that coordinate |
| `left_source_frame_idx` | Left prepared-frame index used by the timing map |
| `right_source_frame_idx` | Right prepared-frame index used by the timing map |
| `interpolation_weight` | Fractional weight from left toward right |
| `is_exact_source_sample` | True when the target time coincides with a source frame |

Unresolved coordinates remain NaN rows (they are not dropped).

## Processing sequence

```text
Accepted S3 tracked_pose.parquet
    + S1 prepared_sync.npz
    + manual_corrections.json (blank_node / identity provenance)
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
Savitzky–Golay smoothing (optional; default enabled)
    ↓
Automated QC (structural failures vs quality warnings)
    ↓
Revalidate S3 acceptance immediately before completion
    ↓
Write final_pose.parquet + run_meta.json + settings_used.yaml + processing_log.txt
```

Geometric refinement is **deferred** for the current-lab MVP: no sufficiently
justified general-purpose method exists in-repo, and the legacy implanted-nose
model is explicitly out of scope.

## Smoothing policy (Savitzky–Golay)

Defaults:

- Enabled
- Window length: 11 frames
- Polynomial order: 2

Rules:

1. Applied after temporal resampling and gap filling.
2. Smooths X and Y independently per track/node.
3. Processes independent continuous valid segments only.
4. Never smooths across unresolved NaN gaps, protected manual blanks, or
   segment discontinuities.
5. Short segments shorter than the configured window are left unchanged
   (no invented coordinates).
6. Pre-smoothing coordinates are retained (`x_pre_smooth` / `y_pre_smooth`).
7. Source confidence and imputation provenance are preserved unchanged.
8. Track identities and timing are never altered.
9. Window length and polynomial order are configurable; smoothing can be
   disabled.

## QC and completion semantics

QC distinguishes:

- **Structural failures** — prevent successful completion (timing/schema
  integrity, protected-blank violations, invalid partial coordinates, etc.).
- **Quality warnings** — recorded in `run_meta.json` without blocking
  completion (high missing/imputation fractions, implausible jumps, large
  smoothing displacements). Thresholds are practical defaults, **not**
  scientifically validated pass/fail claims.

Run statuses:

| Status | Meaning |
| --- | --- |
| `completed` | Structural QC passed; no quality warnings |
| `completed_with_warnings` | Structural QC passed; quality warnings recorded |
| `failed` | Validation, processing, structural QC, or acceptance revalidation failed |

Rules:

- Upstream S1/S2/S3 artifacts remain read-only.
- Partial/invalid artifacts are never marked completed; on failure,
  `final_pose.parquet` is absent/removed.
- S3 acceptance is revalidated immediately before completion. If the source
  becomes `superseded` or the SHA changes mid-run, the S4 run fails.
- A previously completed S4 run is **not** eligible for downstream use when
  its recorded source S3 acceptance is superseded or SHA-mismatched
  (`is_s4_run_eligible_for_downstream` / `validate_s4_run_for_downstream`).

QC summary lives inside `run_meta.json` (no separate QC JSON artifact).

## Sprint 1–2 status (retained)

Sprint 1: contracts, accepted-S3 discovery/validation, authoritative timing map.

Sprint 2: dense 120 Hz construction, manual-blank protection, configurable
linear/Kalman gap fill, provenance codes, identity-corrected provenance from
automatic + active manual stack.

### Provenance codes

| Code | Name |
| --- | --- |
| 0 | `missing` |
| 1 | `original` |
| 2 | `identity_corrected` |
| 3 | `temporal_resample` |
| 4 | `linear_gap_fill` |
| 5 | `kalman_gap_fill` |
| 6 | `protected_manual_blank` |

### Gap-fill defaults at 120 Hz

| Gap length | Method |
| --- | --- |
| 1–20 frames | Linear interpolation |
| 21–140 frames | Constant-velocity Kalman + RTS |
| >140 frames | Remain NaN |
| Leading/trailing gaps | Remain NaN |
| Protected/manual-blank intervals | Remain NaN |

Headstage is excluded from imputation.

## Sprint 3 scope and status

**Implemented**

- `src/pose_finalization/smoothing.py` — segment-wise SavGol.
- `src/pose_finalization/export.py` — authoritative long-form
  `final_pose.parquet`.
- `src/pose_finalization/qc.py` — structural vs warning QC.
- `src/pose_finalization/runner.py` — end-to-end backend runner and
  downstream eligibility helpers.
- Focused tests under `tests/pose_finalization/`.

**Not implemented (explicit non-scope)**

- GUI integration
- Manual S4 review or acceptance
- Identity correction (owned by S3)
- Special implanted-nose imputation
- NPZ export
- General model-based pose refinement / geometric refinement
- New plugin/profile frameworks

## Known limitations

- Geometric refinement is deferred; final coordinates are gap-filled and
  optionally SavGol-smoothed only.
- QC thresholds are operational heuristics for the current lab MVP.
- Temporal resamples are labeled `temporal_resample` even when endpoints were
  identity-corrected.
- Automatic `was_track_swapped` remains frame-granular as produced by S3.
- S4 does not reconstruct S3 undo history beyond the active
  `manual_corrections.json` stack.
- No CLI/GUI scaffolding in Sprint 3; call `run_pose_finalization` from Python.

## Timing notes

- Target rate default is exactly `120.0` Hz.
- Source recordings may have `fps_header` near but not equal to 120; mapping
  uses S1 `prepared_time_sec`.
- Valid S1 `prepared_sync` timelines start at time 0 by S1 contract.
