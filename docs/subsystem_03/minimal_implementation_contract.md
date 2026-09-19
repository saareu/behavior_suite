# Subsystem 03 — Minimal MVP Implementation Contract

**Status:** MVP implemented / validated. See
[`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md) for the authoritative
closure summary, filenames, and stage boundary. This document remains the Gate 4
implementation-boundary design record; where status or filenames conflict, the
MVP closure document wins.

**Scope level:** MVP implementation boundary and integration contract  
**Primary objective:** Deliver an end-to-end usable S3 quickly by integrating the current validated corrector with minimal behavioral modification.

## 1. Implementation strategy

For the MVP, S3 did **not** begin with a major rewrite of the existing tracking corrector.

The current corrector is treated as the first validated S3 tracking backend. The MVP priority is to make the complete S3 workflow operational around it:

```text
completed S2 run
    ↓
S3 input validation / adapter
    ↓
current tracking corrector
    ↓
S3-normalized corrected pose + correction provenance
    ↓
review workspace
    ↓
optional manual edits
    ↓
user acceptance
    ↓
selected tracked pose
```

The current algorithm may be wrapped, lightly adapted, or relocated as needed for integration, but its correction behavior should not be redesigned during the first MVP implementation.

## 2. MVP principle

The MVP prioritizes:

1. end-to-end usability;
2. preservation of the current corrector's effective behavior;
3. clean separation between S3 orchestration and the correction backend;
4. preservation of S2 source information and S3 correction provenance;
5. a usable review-and-accept workflow;
6. minimal implementation risk and minimal unnecessary refactoring.

Internal cleanup that is not required to achieve these goals is deferred.

## 3. Inputs

S3 operates on a selected completed S2 run.

The primary numeric input is:

```text
pose.parquet
```

The associated prepared video is required for review.

S3 may read supporting S2/S1 artifacts when needed for validation, timing preservation, provenance, or display, including:

- `pose_meta.json`;
- `pose.slp`;
- S2 overlay output;
- S1 preparation metadata and sync data referenced by the selected S2 run.

The implementation must preserve the S1/S2 frame and timing domain. S3 does not reconstruct timing independently.

Default outputs are project-root-aware under:

```text
<session_root>/tracking_correction/<profile_id>__<timestamp>/
```

## 4. Primary output

The intended primary S3 pose artifact remains:

```text
tracked_pose.parquet
```

For the MVP, `tracked_pose.parquet` is written only on explicit Accept Tracking.
It is not created by automatic correction alone. The automatic/manual working
state lives in `working_tracked_pose.parquet`.

## 5. Supporting MVP information

S3 must retain enough information to support:

- correction provenance;
- automatic correction review;
- manual correction provenance;
- user acceptance state;
- future debugging and regression validation.

Implemented supporting artifacts:

```text
working_tracked_pose.parquet
automatic_tracked_pose.parquet   # immutable baseline created on first manual edit
machine_corrections.json
manual_corrections.json
run_meta.json
settings_used.yaml
processing_log.txt
```

Training-candidate marking remains a deferred post-MVP item.

## 6. Automatic correction execution

Automatic correction is always performed before review.

The first MVP backend is the current laboratory corrector. The backend may retain its existing:

- sequential frame processing;
- stateful use of already-corrected history;
- rule order;
- current thresholds;
- current track/headstage assumptions;
- current correction actions;
- current suspicious/unresolved behavior.

No algorithmic improvement is required as a prerequisite for integration.

## 7. Legacy-corrector compatibility

The current corrector may continue to use its existing internal data model and helper classes during the MVP.

S3 must isolate backend-specific assumptions behind an adapter boundary so that the rest of S3 does not depend directly on legacy implementation details.

The adapter is responsible for:

- preparing S2 data in the form expected by the corrector;
- passing the appropriate current profile/configuration;
- invoking the current corrector;
- collecting corrected pose state and raw correction records;
- converting the result into S3-owned outputs;
- exposing backend warnings/suspicious frames to the review layer.

## 8. Review contract

After automatic correction, S3 opens the corrected result for user review.

The user reviews the overall stability of tracking across the video.

The user is **not** required to approve every automatic correction individually.

The review workspace must support:

- normal video navigation and frame stepping;
- viewing corrected identity-colored pose;
- comparing S2 provisional vs S3 corrected overlay;
- lightweight prev/next navigation of automatic machine-correction episodes;
- a visible list of manual user corrections from `manual_corrections.json`;
- manual tracking/pose correction where necessary;
- undo/reset of manual edits;
- final acceptance of the tracked result.

The accepted object is the complete tracked pose result, not the correction log.
The main visible list is the manual-correction list, not the raw machine log.

## 9. Manual-edit contract

The MVP review workspace should support sufficiently powerful manual intervention to repair remaining tracking problems.

Implemented MVP operations:

- swapping identities over a frame interval;
- swapping one node between tracks at a frame;
- blanking an invalid node on a track at a frame;
- undoing the last manual change;
- resetting all manual edits to the automatic baseline.

The workspace is not intended to become a general-purpose SLEAP labeling environment.

## 10. Suspicious unresolved cases

The current corrector can identify situations where a failure may have occurred but no automatic correction is applied.

S3 should preserve and expose these as review aids where available.

They should be treated as:

```text
suspicious / unresolved review candidates
```

rather than automatic failures or mandatory per-event approval tasks.

## 11. Tracking profiles

The MVP may expose only one active profile:

```text
current_lab_two_mouse_headstage_v1
```

The user does not need a profile-selection UI for the first MVP.

However, S3 orchestration should refer to the correction backend/profile through an explicit internal boundary so that future profiles or tracking engines can be introduced without redesigning the review workflow.

## 12. Refactoring policy for MVP

The following are explicitly **not prerequisites** for the MVP:

- rewriting the current correction algorithm;
- deduplicating all existing rules;
- replacing all magic values with a new configuration system;
- converting every correction record to a new typed model;
- redesigning the current data container;
- replacing the sequential algorithm with a global optimizer;
- achieving a publication-perfect correction-engine architecture before integration.

Small changes are permitted when required to:

- connect S2 inputs;
- preserve provenance;
- expose suspicious regions;
- support review/manual correction;
- remove hard failures that block the MVP;
- test the backend safely.

Any behavioral change beyond that should be deferred or separately justified.

## 13. Out of scope for Gate 4 MVP implementation

S3 MVP does not add:

- interpolation;
- trajectory imputation;
- smoothing;
- pose resampling;
- dense-pose generation;
- model retraining;
- automatic active-learning execution;
- behavior classification;
- replacement of the current corrector with a new tracking algorithm.

## 14. Acceptance of Gate 4

Gate 4 is approved when the team agrees that:

1. the current corrector will be integrated before major refactoring;
2. the backend will be isolated behind an S3 adapter boundary;
3. S3 orchestration owns input validation, output normalization, review state, and acceptance;
4. automatic correction happens before review;
5. review is global by default with optional event-level intervention;
6. manual correction is supported without turning S3 into a full labeling application;
7. suspicious unresolved cases are preserved for review;
8. later backend refactoring remains possible without blocking the MVP.

These Gate 4 criteria were satisfied by the implemented MVP; see
[`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).
