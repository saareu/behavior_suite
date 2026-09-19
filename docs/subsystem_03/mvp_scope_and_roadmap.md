# Subsystem 03 — Final MVP Scope, Closure Status, and Roadmap

**Status:** MVP implemented / validated. Future changes require a separately
scoped post-MVP task.

## MVP Definition

Subsystem 03 is the tracking-correction and verification subsystem. It consumes
a selected completed Subsystem 02 pose-inference run, applies automatic
current-lab tracking correction, provides a video-centered review workspace
with optional manual corrections, and writes an explicitly accepted
`tracked_pose.parquet`.

S3 owns tracking identity correction, tracking-related pose repair, review,
manual correction, and acceptance. Later pose-finalization owns interpolation,
imputation, resampling, and smoothing. S3 does not train the pose model.

## Subsystem 03 MVP Completion Status

Subsystem 03 MVP implementation and validation are complete for the supported
Windows PySide6 workflow and the current-lab automatic backend.

Closure evidence confirms:

- Gate 3 golden baseline reproduced with zero pose/tracking divergence;
- real GUI S2→S3 flow tested;
- real review and acceptance tested;
- manual-edit smoke testing performed;
- full repository test suite passed at MVP closure.

This finalized status is limited to tracking correction, review, manual repair,
and acceptance under the current-lab profile. It is not a claim of universal
tracking across arbitrary laboratories, nor of final pose smoothing or
analysis-ready trajectories.

## Current S3 input

S3 consumes a completed S2 run directory. The primary numeric pose input is:

```text
pose.parquet
```

Associated S1 prepared video and timing are resolved through the S2→S3 pipeline
handoff (`pose_meta.json` and S1 `preprocess/` artifacts referenced by the S2
run). S3 does not reconstruct timing independently.

Default run placement is project-root-aware:

```text
<session_root>/tracking_correction/<profile_id>__<timestamp>/
```

The GUI launches S3 from a selected completed S2 run. Users may also open an
existing project/session and discover completed S3 runs under
`tracking_correction/`. Direct selection of an S3 run directory remains an
advanced/CLI path, not the primary MVP workflow.

## Current automatic backend

- Profile: `current_lab_two_mouse_headstage_v1`
- Backend: current validated legacy corrector integrated behind the S3 adapter
  boundary (`legacy_backend.py` / `backends/current_lab/`)
- Behavior preserved rather than refactored for MVP
- Current assumptions: two mice, one implanted/headstage animal, current
  skeleton/node set
- No interpolation, smoothing, resampling, or model training in S3

## Current S3 workflow

```text
S2 completed run
    ↓
Run Subsystem 3 (GUI busy/progress indication for long-running work)
    ↓
Automatic current-lab tracking correction
    ↓
S3 video-centered review
    ↓
Optional manual corrections + undo/reset
    ↓
Explicit acceptance
    ↓
tracked_pose.parquet
```

Implemented review capabilities:

- project/session directory discovery of completed S3 runs;
- S2 provisional vs S3 corrected overlay comparison;
- SLEAP skeleton topology sourced from S2 `pose.slp`, with nodes-only fallback;
- authoritative S1 timing used for playback when available;
- stable track-color legend (track 0 = implanted/headstage in the current profile);
- machine-correction episode prev/next navigation (compact, not the main list);
- visible manual-correction list sourced from `manual_corrections.json`;
- manual operations: swap identities (interval), swap node (frame), blank node
  (frame);
- undo last manual edit and reset to automatic baseline;
- acceptance with superseded/re-acceptance after further edits.

Training-candidate marking and per-event deny/reversal of automatic corrections
are deferred post-MVP items; they are not missing blockers for the implemented
workflow above.

## Current S3 outputs

Distinguish clearly between automatic working state, immutable baseline, logs,
and the accepted final:

| Artifact | Role |
| --- | --- |
| `working_tracked_pose.parquet` | Automatic (and later manually edited) working result |
| `automatic_tracked_pose.parquet` | Immutable automatic baseline created when manual editing begins |
| `machine_corrections.json` | Automatic correction provenance log |
| `manual_corrections.json` | User manual-edit stack / provenance |
| `tracked_pose.parquet` | Final accepted tracked pose (written only on Accept Tracking) |
| `run_meta.json` | Run metadata, including acceptance state |
| `settings_used.yaml` | Settings/profile snapshot |
| `processing_log.txt` | Processing log |

Automatic correction alone does **not** write or select `tracked_pose.parquet`.

## Acceptance semantics

1. Automatic correction produces a working result, not acceptance.
2. The user visually reviews the corrected tracking on the prepared video.
3. Explicit Accept Tracking writes/selects `tracked_pose.parquet`.
4. A manual edit after acceptance changes state to `superseded`.
5. Re-acceptance is required after further edits before the accepted file is
   current again.

## Stage boundary

| Owns | Does not own |
| --- | --- |
| Tracking identity correction | Interpolation / imputation |
| Tracking-related pose repair | Resampling / smoothing |
| Review and manual correction | Pose-model training |
| Explicit acceptance of `tracked_pose.parquet` | Behavior classification / feature extraction |

Downstream pose-finalization is the next subsystem boundary after accepted
tracked pose.

## Post-MVP roadmap (non-blocking)

These may improve later workflows but are not missing MVP requirements:

- training-candidate marking UI;
- richer suspicious-region surfacing;
- selective undo of a chosen automatic episode;
- additional tracking profiles/backends;
- algorithmic corrector cleanup after the behavior-preserving MVP integration.

## Related design history

Gate 1–4 decision documents under `docs/subsystem_03/` remain design history.
Where they conflict with this document on implementation status or exact
filenames, this MVP closure document is authoritative for the implemented
system.
