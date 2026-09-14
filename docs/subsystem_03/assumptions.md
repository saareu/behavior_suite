# Subsystem 03 — Assumptions and Current-Profile Constraints

**Status:** Gate 2 review draft  
**Purpose:** Make current MVP assumptions explicit so that they do not silently become permanent S3 architecture.

## 1. Scope of this document

This document separates:

- assumptions required by the current-lab MVP tracking profile;
- compatibility constraints inherited from the current correction algorithm;
- broader S3 principles that should remain valid when future profiles are introduced.

An item appearing here does not automatically become a permanent requirement of Subsystem 03.

## 2. Current-lab MVP assumptions

The first supported tracking profile assumes:

- exactly two experimental animals are the primary tracking targets;
- one animal has a visually identifiable headstage/implant cue;
- the pose skeleton is known and includes the nodes expected by the current correction logic;
- the current experiment/video geometry is sufficiently similar to the validated development data for the existing correction rules to remain meaningful;
- input pose data comes from a completed S2 run;
- review uses the associated prepared video;
- frame identity and timing have already been established by S1 and preserved through S2;
- S3 works in prepared-video frame space and does not resample the data.

These assumptions define the first profile, not all future S3 use cases.

## 3. Identity assumptions

For the current profile:

- the headstage is a strong experiment-specific identity cue for the implanted mouse;
- the second animal can be interpreted relative to that implanted identity;
- temporal continuity and profile-specific geometric/pose rules can be used to preserve identity through time;
- provisional S2 track labels are not assumed to represent final biological identities.

Future profiles may use different identity cues or may only promise stable anonymous animal identities rather than implanted/partner roles.

## 4. Pose assumptions

The current corrector relies on a known skeleton and on profile-specific relationships among nodes.

The current implementation contains logic involving nodes such as:

- headstage;
- neck;
- nose;
- spine/base-body landmarks;
- tail/base landmarks.

The exact required-node contract will be established during the behavioral audit rather than frozen in Gate 2.

A future S3 profile may use a different skeleton.

## 5. Sequential compatibility constraint

The current correction algorithm is stateful and frame-order dependent.

For the initial behavior-preserving refactor:

- frames must be processed in the order expected by the validated algorithm;
- corrections applied to earlier frames must be visible to later decisions;
- rule ordering must initially be preserved;
- caches/state affected by corrections must be invalidated or recomputed as required by the current behavior;
- apparently redundant rules must not be removed until regression evidence shows equivalence.

This sequentiality is **not** a permanent architectural requirement of S3.

Future profiles/backends may use global optimization, bidirectional context, learned identity models, IDTracker.ai, or other non-sequential methods.

## 6. Conservative-correction assumption

The current-lab profile is designed around conservative intervention.

The MVP assumes that:

- an automatic correction should be applied only when supported strongly enough by validated rules;
- uncertain situations may remain unresolved and be surfaced for review;
- avoiding a harmful correction is preferable to forcing a speculative identity assignment;
- difficult pose-model failures may be better captured for future retraining than "repaired" through unsafe tracking logic.

This principle may remain useful beyond the current profile, but its implementation may differ across future backends.

## 7. Review assumptions

The MVP assumes that:

- the user can visually assess identity stability from the corrected overlay/video;
- stable track colors and the visible headstage make whole-session review practical;
- the user does not need to validate each automatic correction individually;
- selected corrections and suspicious regions should remain inspectable;
- final acceptance is a user judgment that tracking is stable across the reviewed video.

The exact degree of review required for future profiles may differ.

## 8. Manual-edit assumptions

The MVP assumes that automatic correction will solve most routine cases but not every possible case.

Therefore the review workspace should permit manual intervention at the level necessary to restore tracking, potentially including:

- identity swaps/reassignment;
- detection removal/blanking;
- node reassignment;
- node removal;
- limited node-position correction when required for tracking.

This does not imply that S3 becomes a general-purpose pose-labeling environment.

## 9. Data-preservation assumptions

S3 must preserve the ability to distinguish original S2 information from changes introduced by S3.

The MVP therefore assumes that:

- S1 and S2 artifacts remain immutable inputs to S3;
- S3 creates a new corrected working/output representation rather than overwriting S2 pose data;
- automatic and user-applied corrections must be provenance-recoverable;
- frame indices and timing must remain aligned with the prepared-video frame domain.

The exact storage mechanism is deferred to the contract-design gate.

## 10. Out-of-scope assumptions

The MVP assumes the following are handled elsewhere and must not be silently introduced into S3 correction logic:

- general interpolation;
- trajectory gap filling;
- Kalman/geometric imputation as a pose-finalization step;
- smoothing;
- resampling/dense-grid construction;
- general pose reconstruction through long occlusion;
- behavior classification;
- SLEAP/model training or optimization.

Limited profile-specific coordinate correction already required by the validated tracker may be preserved during refactoring, but it must not expand into general imputation.

## 11. Execution-environment assumption

The S3 MVP is expected to run locally on a normal lab PC.

No HPC dispatch or GPU requirement is assumed for the initial tracking-correction subsystem unless later behavioral audit or implementation evidence demonstrates a need.

Future alternative tracking backends may have different compute requirements.

## 12. Profile/generalization assumption

S3 should not encode current-lab assumptions globally if they can be scoped to a tracking profile.

Conceptually, the first profile may be named similarly to:

```text
current_lab_two_mouse_headstage_v1
```

Future profiles may differ in:

- animal count;
- skeleton;
- identity cues;
- camera geometry;
- correction rules;
- tracking algorithm;
- compute requirements;
- sequential versus non-sequential processing.

The profile abstraction itself is a design direction. Its final implementation is deferred.

## 13. Assumptions requiring validation during Gate 3

The behavioral audit must verify rather than merely assume:

- the exact skeleton node names relied upon by every current correction rule;
- all thresholds and whether each is pixel-, frame-, or time-dependent;
- which rules depend on the headstage cue;
- which rules depend on fixed two-animal indexing;
- all correction action types currently performed;
- all caches and state that are invalidated after corrections;
- the exact definition and production of suspicious/no-correction cases;
- which coordinate modifications are truly tracking corrections versus legacy imputation-like behavior;
- all pre-processing steps performed before the sequential corrector;
- all post-processing/finalization steps performed before saving corrected tracks;
- representative known-success and known-failure intervals for regression testing.

These findings will determine the Gate 3 behavioral inventory and Gate 4 internal contract.
