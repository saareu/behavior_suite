# Subsystem 03 — Gate 1 Review Checklist

Use this checklist to approve or revise the S3 product vision before workflow, UI, schema, or implementation design begins.

## A. Subsystem identity

- [ ] S3 is correctly defined as an **automatic tracking-correction subsystem with a review workspace**.
- [ ] S3 is not described as only a passive tracking-QC or review tool.
- [ ] `tracked_pose.parquet` is acceptable as the intended primary output name.

## B. Boundary with S2

- [ ] S2 tracks are treated as provisional inference output.
- [ ] S2 pose QC may inform S3 but is not duplicated or treated as confirmed tracking error.
- [ ] Final identity continuity, implanted/partner assignment, and identity-switch correction belong to S3.
- [ ] S3 preserves the S1/S2 prepared-frame and timing contract.

## C. MVP ambition

- [ ] For validated current-lab experiments, S3 should aim to correct the great majority of tracking failures automatically.
- [ ] High precision and low user effort are the primary MVP priorities.
- [ ] The wording does not imply guaranteed perfect or universal tracking.

## D. Current-lab profile

- [ ] The MVP may specialize in two mice, one identifiable headstage, and the current skeleton/video setup.
- [ ] This specialization is treated as a supported profile rather than the permanent S3 definition.
- [ ] The architecture remains open to future tracking methods and experiment profiles.

## E. Pose fixing

- [ ] S3 may perform pose edits required for tracking correction, including track swaps, node moves, and blanking.
- [ ] S3 does not perform general interpolation, imputation, smoothing, or trajectory completion.
- [ ] Limited profile-specific coordinate corrections may remain when required to preserve validated current behavior.

## F. Review and user effort

- [ ] The user must be able to visually review and explicitly accept the S3 result.
- [ ] The user is not required to inspect every frame, track, node, or correction record.
- [ ] Detailed frame-level logs must not automatically become thousands of UI review tasks.
- [ ] The exact digestible UI/event presentation is intentionally deferred to Gate 2.

## G. Training candidates

- [ ] The MVP must let the user mark valuable frames or short intervals for future SLEAP training.
- [ ] Neighboring-frame context and a reason/category should be retained.
- [ ] Automatic candidate suggestion is optional for the initial MVP.
- [ ] Model retraining and active-learning execution remain outside S3.

## H. Refactoring policy

- [ ] The existing corrector should be rewritten in a cleaner structure rather than committed unchanged as legacy production code.
- [ ] The rewrite must initially preserve validated behavior, unusual thresholds, rule ordering, and special cases.
- [ ] Duplicate-looking conditions are not removed without regression evidence.
- [ ] Configuration replaces magic numbers where practical without sacrificing efficacy.

## I. Sequentiality

- [ ] Sequential/stateful processing is a compatibility constraint of the current MVP corrector refactor.
- [ ] Sequentiality is not locked as a permanent principle of S3.
- [ ] Future backends may be non-sequential, global, bidirectional, learned, or externally integrated.

## J. Frame policy and execution environment

- [ ] S3 MVP performs no resampling.
- [ ] S3 operates in prepared-video frame space.
- [ ] Future thresholds should become time-aware where practical.
- [ ] Local-PC execution is sufficient for the MVP.
- [ ] No HPC-oriented job manifest is assumed at Gate 1.

## K. Deferred decisions

Confirm that Gate 1 does **not** yet lock:

- [ ] the full S3 artifact set;
- [ ] `tracked_pose.parquet` columns;
- [ ] correction-record fields;
- [ ] review-status fields;
- [ ] correction-episode aggregation rules;
- [ ] manual UI actions;
- [ ] internal class/module structure;
- [ ] quantitative acceptance thresholds;
- [ ] the future alternative tracking backend.

## Approval record

**Decision:** ☐ Approved ☐ Approved with edits ☐ Not approved  
**Reviewer:**  
**Date:**  

### Required edits or comments

1. 
2. 
3. 
