# Subsystem 03 — Gate 4 Review Checklist

**Document role:** Design-history checklist for Gate 4 integration approval.  
**Implementation status:** MVP implemented / validated — see
[`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).

Use this checklist to approve the MVP integration architecture before implementation begins.

## A. MVP strategy

- [ ] The current corrector should be integrated first rather than substantially refactored first.
- [ ] End-to-end usability has priority over correction-engine cleanup for the MVP.
- [ ] Algorithmic improvement is not a prerequisite for S3 MVP delivery.

## B. Backend isolation

- [ ] The current corrector is treated as the first S3 backend/profile.
- [ ] S3 orchestration does not depend directly on the backend's internal heuristics.
- [ ] A small adapter boundary prepares inputs, invokes the current corrector, and normalizes outputs.
- [ ] Future tracking profiles/backends can replace the current backend without redesigning the overall review workflow.

## C. Current behavior

- [ ] Sequential/stateful current-corrector behavior may remain unchanged for MVP.
- [ ] Current thresholds and special cases may remain unchanged.
- [ ] Existing correction records may remain internally verbose.
- [ ] Suspicious/unresolved cases should be preserved when the backend exposes them.

## D. S3 orchestration

- [ ] S3 consumes a selected completed S2 run.
- [ ] S3 never modifies S1/S2 source artifacts.
- [ ] Automatic correction runs before review.
- [ ] The associated prepared video is available to the review workspace.
- [ ] User acceptance is owned by S3, not by the correction backend.

## E. Outputs/provenance

- [ ] `tracked_pose.parquet` remains the intended primary S3 pose artifact.
- [ ] Original/raw pose information should be retained sufficiently to trace S3 changes.
- [ ] Automatic and manual corrections remain distinguishable.
- [ ] Proven historical flags such as swapped/blanked information may be reused rather than redesigned immediately.
- [ ] Gate 4 does not require a large new artifact family.

## F. Review workflow

- [ ] User review happens after automatic correction.
- [ ] Acceptance means the user judges tracking stable across the video.
- [ ] Per-correction approval is not required.
- [ ] User may inspect/deny/modify selected correction events when warranted.
- [ ] Suspicious regions can be used as navigation aids.

## G. Manual correction

- [ ] Manual editing should be sufficiently powerful to repair remaining tracking errors.
- [ ] Iterative edit → review → edit → accept is supported conceptually.
- [ ] Manual edits apply to an S3 working copy, never directly to S2 output.
- [ ] S3 does not become a general-purpose labeling application.

## H. Deferred work

Confirm that the MVP does **not** require first:

- [ ] a full correction-engine rewrite;
- [ ] a complete typed-event redesign;
- [ ] removal of duplicate/legacy rules;
- [ ] replacement of all configuration values;
- [ ] non-sequential/global tracking;
- [ ] interpolation or smoothing;
- [ ] model retraining;
- [ ] a multi-profile selection UI.

## I. Implementation order

- [ ] Sprint 1: S3 shell + S2 validation.
- [ ] Sprint 2: current corrector adapter + regression check.
- [ ] Sprint 3: corrected review workspace.
- [ ] Sprint 4: manual corrections + training candidate marking.
- [ ] Sprint 5: full end-to-end acceptance validation.

## Approval record

**Decision:** ☐ Approved ☐ Approved with edits ☐ Not approved  
**Reviewer:**  
**Date:**  

### Required edits or comments

1. 
2. 
3. 
