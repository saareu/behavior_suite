# Subsystem 03 — Current Corrector Behavioral Inventory

**Status:** Gate 3 review draft  
**Purpose:** Capture the behavior that must be understood and regression-protected before refactoring the current tracking corrector.

## 1. Scope

This document inventories the behavior of the current Pipeline 20 tracking-correction stack. It is descriptive, not prescriptive: it records what the current implementation does, including awkward or highly specific behavior, so the S3 rewrite does not accidentally remove validated logic.

The current stack is composed of:

- Pipeline 20 orchestration;
- `id_corrector_minimal.py`;
- `TrackingData` mutable state container;
- `CorrectionParams` parameter adapter;
- `track_ids.yaml` active profile configuration;
- `finalize_tracks_df_for_save()` output finalizer.

The current implementation should be treated as a behavioral reference and regression oracle, not as the target architecture.

## 2. Processing model

The corrector is sequential and stateful. Frames are processed in order, and corrections applied to an earlier frame can alter the data used to interpret later frames.

The current frame loop performs the following broad sequence:

1. headstage ownership/removal handling;
2. jump/identity correction once enough history is available;
3. headstage handling again after jump correction;
4. skeleton stretch validation;
5. orientation validation;
6. cache invalidation whenever the current frame changes.

The duplication of headstage handling inside the same frame is current behavior and must not be removed merely because it appears redundant.

The jump handler uses cached trajectory, stable-node, and track-validity calculations. Position/stable-node caches are explicitly cleared after mutations so subsequent decisions use corrected state.

## 3. Mutable tracking state

`TrackingData` converts the long pose table to dense arrays shaped approximately as:

```text
[frame, track, node]
```

for `x`, `y`, and score.

Important behavior:

- string tracks such as `track_0` are converted to integer indices;
- the current implementation derives the number of tracks from the maximum observed track index;
- preferred node order is `nose`, `neck`, `spine_base`, `tail_base`, `headstage`, followed by any remaining nodes;
- the original dataframe is retained as `df_orig`;
- non-spatial extra columns are retained for later merge-back;
- explicit node blanking is recorded in `blanked_nodes`;
- any track or node swap marks the frame in `swapped_frames`.

Core mutation primitives currently include:

```text
blank_frame(frame, track)
blank_node(frame, track, node)
swap_tracks(frame)
swap_node(frame, node)
```

These mutation semantics are part of the behavioral baseline even if the rewrite later exposes them through typed correction actions.

## 4. Parameter model

`CorrectionParams.from_config()` treats the active configuration as a 120 FPS baseline.

The current code scales selected motion-related parameters according to the video FPS:

- `jump_threshold`;
- `proximity_threshold`;
- `proximate_jump_thresh`;
- `frame_window`.

Anatomical/spatial thresholds such as overlap distances, plausibility radii, and headstage-to-neck constraints are not scaled.

The active configuration currently specifies, among other values:

```text
proximity_threshold: 30
overlap_dist: 6
spine_overlap_dist: 15
overlap_jump_dist: 10
medium_proximity_threshold: 12
jump_threshold: 30
frame_window: 10
proximate_jump_thresh: 1
orientation_cosine_threshold: 0.866
minimum_separation_distance: 23
orientation_flip_threshold: 1.57
```

Skeleton plausibility radii are defined for:

```text
headstage_nose
nose_neck
neck_spine
spine_tail
headstage_neck
neck_tail
```

The active profile assumes `headstage` is the biological identity cue and that implanted identity corresponds to track 0 in the current-lab profile.

## 5. Headstage ownership behavior

The current headstage handler enforces the working assumption that the implanted animal belongs to track 0.

Observed behaviors include:

- if only track 0 has a headstage, leave the frame unchanged;
- if only track 1 has a headstage, copy the observed headstage into track 0 and blank it from track 1;
- if both tracks have a headstage, optionally swap the headstage node based on continuity with track 0's previous headstage, then blank the duplicate from track 1;
- if neither corrected track currently has the headstage but the original dataframe contained it, recover an original observed headstage into track 0;
- if a duplicate remains at the same coordinates on the other track, blank it.

This logic relies on both corrected state and preserved original input.

## 6. Jump/identity correction behavior

The jump handler is the main identity-continuity engine.

It estimates track position from visible/stable non-headstage nodes, predicts trajectory from a recent frame window, and computes jump magnitudes relative to predicted or previous positions.

It also evaluates:

- cross-track continuity;
- same-track continuity;
- node overlap with previous-frame observations;
- predicted position distances;
- orientation consistency;
- current visible-node counts;
- cross-visible separation;
- special-case geometry around tail, spine, neck, nose, and headstage.

The implementation contains many highly specific combinations and fixed thresholds, including several direct pixel comparisons that are not all routed through YAML configuration.

Current mutation families inside jump handling include:

- whole-track identity swap;
- selected node swap;
- extraction of nodes from one track followed by blanking invalid remnants;
- single-node carry-forward from the previous frame;
- single-node blanking;
- overlap cleanup;
- suspicious detection with no automatic correction.

The rule ordering matters because the handler returns after some successful corrections and because subsequent frames see already-corrected history.

## 7. Stable-node and trajectory behavior

A node is considered stable based on visibility over the recent frame window. The current implementation uses a relatively permissive minimum visibility count for position calculations and a stricter variant for orientation calculations.

The headstage is excluded from trajectory/stable-node position estimation.

If no stable nodes are available, currently visible non-headstage nodes may be used as a fallback.

Trajectory prediction uses average velocity over available historical positions. Jump magnitude is then measured against the predicted next position when possible, otherwise against the previous position.

## 8. Tracking-related coordinate carry-forward

The current corrector sometimes sets a suspicious node's current coordinates equal to its previous-frame coordinates.

This occurs in several tracking-repair contexts, including single-node jump, skeleton stretch, overlap, and orientation repair.

A key conservative safeguard is the anti-chaining behavior: when the previous position appears itself to be a carried-forward/corrected value, the code may blank the current node instead of continuing to propagate the same coordinate indefinitely.

For S3 design purposes this behavior is classified as a **profile-specific tracking correction**, not as general interpolation. It must remain regression-protected during the first refactor.

## 9. Skeleton stretch validation

The current corrector checks configured anatomical distances between selected node pairs.

When a node is identified as the likely source of an implausible skeleton stretch, the current implementation may:

- replace the current node position with the preceding reliable position; or
- blank the node when a reliable predecessor is unavailable or already corrected.

The correction log uses `skeleton_stretch` records for these actions.

## 10. Orientation validation

The current orientation handler uses relative skeleton geometry/cosine tests to identify likely node misassignments.

Suspect nodes can be repaired by carrying forward a reliable preceding position; otherwise they are blanked.

The same anti-chaining idea is used: if the previous position appears already corrected, the handler avoids blindly propagating it.

The correction log uses `orientation` records.

## 11. Suspicious but uncorrected states

The current algorithm contains cases where abnormal motion is detected but the evidence is insufficient for a safe automatic correction. Gate 2 already established that these should become review candidates rather than forced corrections.

During refactoring, every current branch that intentionally declines to mutate data after detecting a suspicious condition should be identified and preserved as an explicit behavioral case. The first rewrite must not silently convert these branches into aggressive corrections.

## 12. Correction logging behavior

The legacy correction list is frame-level and verbose. Records generally contain:

```text
frame
track (when applicable)
type
action
```

Observed correction types include at least:

```text
identity_swap
headstage_reassigned
headstage_duplicate_removed
headstage_assigned
single_node_jump
skeleton_stretch
intra_node_overlap
orientation
```

The log is sorted after processing by frame, then track where available, then type.

The example full-session `corrections.json` demonstrates that repeated frame-level records can represent a much smaller number of biologically meaningful episodes. Therefore the legacy correction count must not be treated as a user review count.

## 13. Output conversion behavior

After correction, dense arrays are converted back to a long dataframe.

The current code attempts to preserve extra input columns by merging them from the original dataframe on `(frame, track, node)`.

`finalize_tracks_df_for_save()` then standardizes the persisted tracking table by:

1. dropping redundant `score`;
2. ensuring instance-score columns exist;
3. computing `was_node_blanked` from explicit blanking and raw-vs-final coordinate differences;
4. computing frame-level `was_track_swapped`;
5. converting coordinate/score fields to `float32` and flags to `uint8`;
6. sorting stably by `(frame, track, node)`;
7. emitting a fixed output column order.

The historical accepted `tracks.parquet` schema contains:

```text
frame
track
node
x
y
point_score
instance_score
x_raw
y_raw
point_score_raw
instance_score_raw
was_track_swapped
was_node_blanked
```

The uploaded accepted example contains 304,994 persisted rows according to its parquet metadata.

The corresponding uploaded input `pose.parquet` has the historical schema:

```text
frame
track
node
x
y
score
instance_score
```

These files are valuable golden artifacts, but their historical schema is not automatically the future S3 schema; S2's current contract must remain authoritative when the new S3 handoff is finalized.

## 14. Behaviors that must be regression-protected

The first S3 rewrite must preserve, unless a deliberate later change is approved:

- sequential frame ordering;
- corrected-history dependence;
- rule ordering within the current profile;
- headstage ownership semantics;
- whole-track swap behavior;
- node swap/blank behavior;
- extraction-and-blank behavior;
- anti-chaining safeguards;
- current suspicious-but-uncorrected decisions;
- current unusual hard-coded thresholds and special cases;
- provenance of raw versus corrected coordinates;
- correction event generation at a semantically equivalent level;
- preservation of input confidence/provenance information required downstream.

## 15. Known cleanup opportunities that are not yet behavior changes

The Gate 4 architecture may safely improve structure around these behaviors without changing their decisions:

- centralize mutations through typed actions;
- separate rule evaluation from data mutation;
- name currently hard-coded thresholds;
- make suspicious/no-op outcomes explicit;
- prevent duplicate correction records where duplicates are logging artifacts rather than distinct mutations;
- isolate profile-specific assumptions;
- make caches and invalidation rules explicit;
- add deterministic unit and golden regression tests.

Any change that alters actual corrected coordinates, track identity, blanking, or suspicious decisions must be evaluated against the golden regression set rather than treated as mere cleanup.
