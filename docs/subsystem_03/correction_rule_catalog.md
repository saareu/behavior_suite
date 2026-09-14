# Subsystem 03 — Current Correction Rule Catalog

**Status:** Gate 3 review draft  
**Purpose:** Catalog the rule families and mutation semantics of the current-lab corrector before refactoring.

This document intentionally catalogs rule **families** rather than assigning permanent S3 rule IDs. Stable rule IDs belong to the later implementation contract.

## 1. Headstage ownership rules

### H1 — Correct ownership already present

Condition: headstage observed on track 0 and not track 1.  
Action: no mutation.

### H2 — Headstage present only on track 1

Condition: headstage observed on track 1 and absent on track 0.  
Action: copy the observed headstage to track 0, blank track 1 headstage.  
Legacy log type: `headstage_reassigned`.

### H3 — Duplicate headstage detections

Condition: both tracks contain a headstage detection.  
Action: use continuity with the prior track-0 headstage to decide whether to swap the node, then blank track 1's duplicate.  
Legacy log type: `headstage_duplicate_removed`.

### H4 — Recover headstage from original input

Condition: neither corrected track currently contains the headstage but the original input has an observed headstage for the frame.  
Action: assign an original observed headstage to track 0 and remove any exact duplicate on track 1.  
Legacy log type: `headstage_assigned`.

## 2. Whole-track identity rules

### I1 — Cross-frame identity swap

Evidence family:

- both tracks show abnormal jumps or cross-track continuity;
- current nodes fit the opposite track's previous/predicted location better than their own;
- proximity and overlap constraints support the swap.

Action: swap track 0 and track 1 for the current frame.  
Legacy log type: `identity_swap`.

Important: the current code contains many subconditions for this family. The first rewrite should preserve those subconditions and their order rather than collapsing them into a single generalized score.

## 3. Partial-track extraction/reassignment rules

### P1 — Selected node swap

Condition: one or more nodes on the current track match the other track's historical position with very tight spatial agreement.  
Action: swap only the implicated node(s).

### P2 — Extract-and-blank

Condition: a track contains a mixture of nodes that appear to belong to the opposite animal plus invalid remainder.  
Action: extract/reassign supported node detections and blank invalid residual detections.

These rules are especially sensitive to rule ordering and corrected history.

## 4. Single-node jump rules

### J1 — Large isolated node displacement with reliable predecessor

Condition: a node moves implausibly relative to its previous location while surrounding evidence indicates the track identity itself should remain unchanged.  
Action: set current node coordinates to a reliable preceding position.  
Legacy log type: `single_node_jump`.

### J2 — Large isolated node displacement without reliable predecessor

Condition: same as J1, but the preceding position is unavailable or itself considered corrected/unreliable.  
Action: blank the node rather than propagate a possibly false coordinate.  
Legacy log type: `single_node_jump`.

## 5. Intra-track overlap rules

### O1 — Implausible node overlap with reliable predecessor

Condition: node geometry/overlap indicates a duplicated or misplaced node while a reliable preceding position exists.  
Action: restore previous reliable position.  
Legacy log type: `intra_node_overlap`.

### O2 — Implausible node overlap without reliable predecessor

Action: blank the suspect node.  
Legacy log type: `intra_node_overlap`.

Anti-chaining behavior is important: a previous coordinate that appears to have been carried forward may be rejected as the source for another carry-forward.

## 6. Skeleton stretch rules

### S1 — Anatomical stretch with identifiable suspect node and reliable predecessor

Condition: configured anatomical distances exceed profile-specific plausibility radii and the handler identifies a likely offending node.  
Action: restore the preceding reliable position.  
Legacy log type: `skeleton_stretch`.

### S2 — Anatomical stretch without reliable predecessor

Action: blank the suspect node.  
Legacy log type: `skeleton_stretch`.

## 7. Orientation rules

### R1 — Orientation inconsistency with reliable predecessor

Condition: relative skeleton geometry/cosine tests identify a node whose placement is inconsistent with the expected body orientation.  
Action: restore previous reliable position.  
Legacy log type: `orientation`.

### R2 — Orientation inconsistency without reliable predecessor

Action: blank the suspect node.  
Legacy log type: `orientation`.

## 8. Suspicious/no-correction rules

### U1 — Both tracks jump but swap evidence is insufficient

Condition: both tracks show abnormal motion, but cross-track, geometry, continuity, or proximity evidence does not satisfy a conservative correction branch.  
Action: preserve data unchanged and surface the frame/interval as suspicious for review.

This family is especially important for S3 because conservative non-action is a valid algorithm outcome, not a failure to implement a correction.

Additional no-op/suspicious branches must be enumerated directly from the legacy source during implementation migration and covered by tests.

## 9. Mutation semantics to preserve

The refactor should distinguish the following conceptual actions even if the legacy code currently performs them inline:

```text
SWAP_TRACKS
SWAP_NODE
REASSIGN_NODE
BLANK_NODE
BLANK_INSTANCE_OR_TRACK
EXTRACT_AND_BLANK
RESTORE_PREVIOUS_OBSERVED_POSITION
NO_CHANGE_SUSPICIOUS
```

These are conceptual behavior categories for audit purposes, not yet the final action schema.

## 10. Parameter sources

### Configured current-profile values

Examples include:

```text
proximity_threshold = 30 px
overlap_dist = 6 px
spine_overlap_dist = 15 px
overlap_jump_dist = 10 px
medium_proximity_threshold = 12 px
jump_threshold = 30 px/frame at 120 FPS baseline
frame_window = 10 frames at 120 FPS baseline
proximate_jump_thresh = 1 px
orientation_cosine_threshold = 0.866
minimum_separation_distance = 23 px
orientation_flip_threshold = 1.57 rad
```

### Anatomical radii

```text
headstage_nose = 95 px
nose_neck = 95 px
neck_spine = 95 px
spine_tail = 105 px
headstage_neck = 95 px
neck_tail = 140 px
```

### Hard-coded rule thresholds

The current jump handler also contains direct thresholds such as 2, 3, 8, 9, 60 px and ratio/cosine tolerances in individual branches. These values must be inventoried during the rewrite and given semantic names before any attempt is made to generalize them.

## 11. Rule-order constraint

The rule catalog is not commutative. The same frame can produce different downstream behavior depending on which correction occurs first because:

- mutations alter the current dense state;
- some branches return immediately after a correction;
- later stages operate on already-corrected current-frame data;
- future frames use corrected history.

Therefore, preserving rule **order** is part of behavior preservation for the current profile.

## 12. Logging versus biological events

A correction record is not necessarily a unique biological event.

The uploaded example contains long contiguous sequences of `identity_swap` and repeated headstage records across many frames. The S3 UI must later aggregate these for review, but Gate 3 regression should still preserve the underlying coordinate/identity result rather than requiring exact equality of noisy duplicate log counts.
