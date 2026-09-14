# Subsystem 03 — Gate 2 Review Checklist

Use this checklist to approve or revise the S3 workflow and review philosophy before the current corrector is audited and refactored.

## A. Processing order

- [ ] Automatic correction always runs before user review.
- [ ] The automatic result is a working corrected state, not automatically the final accepted result.
- [ ] Manual edits can be made after automatic correction without requiring immediate finalization.

## B. Acceptance model

- [ ] The user accepts the **overall tracked pose**, not each automatic correction.
- [ ] Acceptance means the user has reviewed the result sufficiently to confirm tracking stability across the video.
- [ ] The user does not need to approve every frame/node/event individually.

## C. Review philosophy

- [ ] Review is primarily video-centered.
- [ ] Whole-session stability review is the default workflow.
- [ ] Selected corrections and suspicious regions can be inspected directly.
- [ ] Detailed machine correction records do not automatically become separate user tasks.

## D. Automatic correction events

- [ ] Automatically corrected events are retained for provenance but do not require routine per-event confirmation.
- [ ] The user may inspect a selected automatic correction when warranted.
- [ ] The user should be able to keep, deny/revert, or replace a selected automatic correction conceptually, with safe semantics to be designed later.

## E. Suspicious unresolved regions

- [ ] The corrector may identify suspicious regions where evidence is insufficient for safe automatic correction.
- [ ] Suspicious does not mean confirmed error.
- [ ] Such regions are useful navigation/review targets.

## F. Manual correction scope

- [ ] The MVP should allow broad tracking correction rather than only marking bad regions.
- [ ] Track/identity reassignment and blanking are in scope.
- [ ] Node-level correction may be available when required to restore tracking.
- [ ] S3 is not intended to become a general SLEAP pose-labeling environment.
- [ ] Interpolation, smoothing, and general pose completion remain out of scope.

## G. Iteration

- [ ] The user can edit, continue reviewing, undo/revise, and make additional corrections before final acceptance.
- [ ] A manual correction does not automatically require rerunning the entire automatic tracker.
- [ ] The exact interaction between manual edits and sequential dependencies is deferred to later architecture design.

## H. Current corrector migration

- [ ] The current pipeline is treated as a behavioral reference, not copied unchanged as the new architecture.
- [ ] A behavioral inventory and golden regression examples come before the rewrite.
- [ ] The initial rewrite preserves validated rule behavior before algorithmic simplification/generalization.

## I. Sequentiality

- [ ] Sequential/stateful behavior is preserved during the current-backend refactor.
- [ ] Sequentiality is not a permanent S3 requirement.
- [ ] Future backends may use non-sequential methods.

## J. Profiles

- [ ] The architecture may support multiple future tracking profiles/backends.
- [ ] Only the current-lab profile is required for MVP.
- [ ] A user-facing profile selector is not required for MVP.

## K. Training candidates

- [ ] The user can mark valuable frames/short intervals for future model training.
- [ ] Context and reason/category should be retained.
- [ ] S3 does not execute model training or active learning.

## L. Deferred decisions

Confirm that Gate 2 does **not** yet lock:

- [ ] parquet/correction schemas;
- [ ] exact event/status enums;
- [ ] UI framework/layout;
- [ ] exact manual-edit gestures;
- [ ] undo/replay implementation;
- [ ] correction aggregation algorithm;
- [ ] internal classes/modules;
- [ ] quantitative acceptance thresholds;
- [ ] alternative backend implementation.

## Approval record

**Decision:** ☐ Approved ☐ Approved with edits ☐ Not approved  
**Reviewer:**  
**Date:**  

### Required edits or comments

1. 
2. 
3. 
