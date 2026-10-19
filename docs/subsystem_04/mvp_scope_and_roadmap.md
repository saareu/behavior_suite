# Subsystem 04 — Pose Finalization MVP Scope and Roadmap

**Status:** MVP complete / validated (Sprints 1–4). GUI integration, Advanced
Settings, completed-run discovery, automated tests, and a real laboratory
session GUI run are done.

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
| Compact GUI run + QC summary | S4 visual review / user acceptance |

S3 `complete_reviewable` means automatic correction finished and review is
possible. It is **never** equivalent to acceptance for S4.

## GUI workflow

1. Select a project/session (shared navigation with S1–S3, or browse from S4).
2. S4 automatically selects the latest eligible accepted S3 run; choose another
   accepted run when several exist.
3. Optionally open Advanced Settings (defaults are preserved if left closed).
4. Click **Run Finalization** (background `GuiTaskRunner`; duplicate launches
   blocked while a run is active).
5. Inspect processing status (`completed`, `completed_with_warnings`, or
   `failed`).
6. View the compact QC summary and open the S4 run folder for
   `final_pose.parquet`.

There is no S4 visual review workspace and no S4 user-acceptance step.
After S3 acceptance, the main window can continue to Subsystem 4; S4 still
requires an eligible accepted S3 result and revalidates before completion.

When no eligible accepted S3 result exists, the GUI explains why (missing
`tracking_correction/`, unaccepted/superseded runs, SHA mismatch, or missing
`prepared_sync.npz`). Users never pick Parquet or timing files manually.

## Advanced Settings

Exposed parameters (backend defaults when Advanced Settings are not opened):

| Group | Parameter | Default |
| --- | --- | --- |
| Gap filling | Linear maximum gap | 20 frames |
| Gap filling | Kalman maximum gap | 140 frames |
| Gap filling | Kalman process / measurement variance | `10.0` / `1.0` |
| Smoothing | Enabled | yes |
| Smoothing | SavGol window | 11 frames (odd) |
| Smoothing | Polynomial order | 2 |
| Output sampling | Target FPS | 120 Hz (fixed for MVP) |

Settings are validated before execution (incompatible gap thresholds, even
windows, polyorder ≥ window, non-positive Kalman variances, non-120 Hz target).
Actual values are written to `settings_used.yaml`.

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

Parquet is the only final pose format for the MVP. No NPZ outputs. Each run
uses a unique timestamped directory; reruns never overwrite prior S4 results.

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

Geometric refinement remains **deferred** for the current-lab MVP.

## Completion and staleness

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
- Completed S4 discovery (`summarize_pose_finalization_project`) classifies
  runs as `eligible` or `stale_source` by rechecking the recorded accepted S3
  source. Prior S4 directories are never rewritten merely because upstream
  acceptance later changes.
- Downstream helpers: `is_s4_run_eligible_for_downstream` /
  `validate_s4_run_for_downstream`.

QC summary lives inside `run_meta.json` (no separate QC JSON artifact).

## Provenance codes

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

## Sprint status

| Sprint | Status |
| --- | --- |
| 1 — Contracts, accepted-S3 discovery, timing map | Done |
| 2 — Dense pose, blank protection, gap fill | Done |
| 3 — SavGol, export, QC, runner | Done |
| 4 — GUI, Advanced Settings, discovery, e2e, lab GUI | Done (MVP closed) |

### Sprint 4 validation coverage

Focused tests cover:

- Accepted S3 → S1 timing → dense 120 Hz → gap fill → SavGol → QC →
  `final_pose.parquet`
- GUI session/project discovery and automatic accepted-S3 selection
- Default and modified Advanced Settings persistence in `settings_used.yaml`
- Missing / unaccepted / superseded S3 rejection with clear GUI messaging
- S3 superseded during processing → failed run (no completed parquet)
- Completed S4 becoming stale after upstream supersession / SHA corruption
  without mutating prior S4 artifacts
- Multiple completed runs; upstream non-mutation; processing-error recovery
- Sprint 3 coverage gaps: cross-boundary smoothing independence; downstream
  SHA corruption detection; NaN confidence on reconstructed rows; exported
  timing/source mapping consistency; output-overlap protection

### Laboratory GUI session validation

A full laboratory session derived from
`test_cases/E_full_session_integration` was exercised through the real
`MainWindow` → Subsystem 4 page → **Run Finalization** path (sandbox copy only;
experimental fixture acceptance state left `not_accepted`).

| Item | Result |
| --- | --- |
| Source session | E full-session integration (~36 239 prepared frames, 2 tracks, 5 nodes) |
| GUI path | Auto-selected accepted S3 → background `GuiTaskRunner` → QC panel |
| Outcome | `completed_with_warnings` |
| Output | `final_pose.parquet` written under sandbox `pose_finalization/<run_id>/` |
| QC summary | Missingness ~15%; reconstructed ~6%; SavGol enabled; warnings recorded |
| Fixture safety | Original `test_cases/` S3 run remained `not_accepted` / no `tracked_pose.parquet` |
| Defects found | None |

## Known limitations / remaining post-MVP risks

- Geometric refinement is deferred; final coordinates are gap-filled and
  optionally SavGol-smoothed only.
- QC thresholds are operational heuristics for the current lab MVP.
- Target FPS is fixed at 120 Hz for the authoritative timing contract.
- Temporal resamples are labeled `temporal_resample` even when endpoints were
  identity-corrected.
- Automatic `was_track_swapped` remains frame-granular as produced by S3.
- S4 does not reconstruct S3 undo history beyond the active
  `manual_corrections.json` stack.
- No NPZ export; no mandatory visual QC; no S4 acceptance step.
- Pose finalization has no cooperative cancellation hook (same class of
  limitation as S2 inference): the GUI stays responsive and blocks duplicate
  launches / navigation while a run is active.

## Timing notes

- Target rate default is exactly `120.0` Hz.
- Source recordings may have `fps_header` near but not equal to 120; mapping
  uses S1 `prepared_time_sec`.
- Valid S1 `prepared_sync` timelines start at time 0 by S1 contract.
