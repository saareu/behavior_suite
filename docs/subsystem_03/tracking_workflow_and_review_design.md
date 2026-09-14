# Subsystem 03 — Tracking Workflow and Review Design

**Status:** Gate 2 review draft  
**Scope level:** User workflow, review philosophy, and correction-workspace behavior  
**Depends on:** Approved Gate 1 S3 product vision  

## 1. Purpose of Gate 2

This document defines how Subsystem 03 should behave from the user's point of view and how the automatic-correction and review phases relate to one another.

Gate 2 does **not** define the final file schemas, internal class structure, UI framework, correction-record schema, or implementation details. Those decisions belong to later gates.

The approved Gate 1 boundary remains authoritative:

- S2 produces provisional pose/tracking output.
- S3 owns identity continuity and tracking correction.
- S3 is an automatic correction subsystem with a review workspace, not merely a QC viewer.
- S3 preserves original S2 information and correction provenance.
- The MVP targets the current two-mouse/headstage laboratory profile without permanently locking S3 to that profile.
- General interpolation, smoothing, resampling, and model training are outside S3 MVP.

## 2. Primary S3 workflow

For the MVP, automatic correction always runs before user review.

```text
Select completed S2 run
        ↓
Run automatic tracking correction
        ↓
Create corrected working result
        ↓
Open tracking review workspace
        ↓
User reviews tracking stability across the video
        ↓
Optional interventions
  ├─ inspect suspicious regions
  ├─ inspect selected automatic correction events
  ├─ accept / deny / modify a selected correction event when warranted
  ├─ apply manual tracking or tracking-related pose fixes
  └─ mark valuable frames/intervals for future pose-model training
        ↓
Continue review and editing as needed
        ↓
User confirms tracking is stable across the video
        ↓
Accept final tracked pose
```

The automatic correction result is therefore a **working corrected state**, not the final accepted result by itself.

## 3. Meaning of acceptance

S3 acceptance applies to the **overall tracking result**, not to every individual automatic correction.

The user accepts the tracked pose when they have reviewed the result sufficiently to confirm that tracking appears stable and biologically consistent across the video.

Acceptance does **not** require:

- approving every automatic correction individually;
- reviewing every frame one by one;
- reviewing every node individually;
- clearing every machine-generated correction record;
- confirming every suspicious region if the user judges broader review sufficient.

The review workspace must make it possible to inspect individual events when desired, but event-level approval is not the default workflow.

## 4. Review philosophy

### 4.1 Review is video-centered

The primary review object is the corrected tracking overlaid on the prepared video.

The user should be able to evaluate identity continuity directly by watching and navigating the video, using stable identity colors and the visual cues available in the experiment, especially the implanted/headstage mouse in the current MVP profile.

The workspace should support efficient whole-session review rather than forcing event-by-event processing.

### 4.2 Review should minimize user effort

The user should be able to:

- play through the video;
- scrub rapidly through the session;
- step frame by frame when needed;
- jump directly to suspicious locations or selected corrections;
- compare the working corrected result with the provisional S2 result when useful;
- apply manual corrections without leaving the review context.

Exact controls and visual layout are intentionally deferred until a later UI-design step.

### 4.3 Machine detail must not become user burden

The correction engine may produce many frame-level changes. A sustained identity problem may generate corrections across many consecutive frames even though the user experiences it as one tracking episode.

Detailed records should remain available for provenance, testing, and debugging, but they must not automatically become thousands of separate review tasks.

The user-facing system should summarize or navigate machine activity in a digestible form. The exact aggregation strategy is not fixed in Gate 2.

## 5. Review-relevant event classes

Gate 2 distinguishes three broad concepts. These are workflow concepts, not final schema enums.

### 5.1 Automatically corrected event

The automatic corrector identifies a sufficiently supported tracking problem and modifies the working tracking result.

The user does not need to confirm the event individually by default.

The event should remain inspectable so that the user can deny or modify it when warranted.

### 5.2 Suspicious unresolved region

The corrector identifies evidence that something may be wrong but does not have sufficient confidence to make a safe automatic correction.

Examples include current-code situations in which both tracks appear to jump but the rule set cannot establish a sufficiently supported correction.

These regions are useful review targets because they represent places where conservative automatic correction deliberately declined to act.

A suspicious region is **not** equivalent to a confirmed tracking error.

### 5.3 User-applied correction

The user modifies the working tracking result after review.

User-applied changes should remain distinguishable from automatic changes in provenance, but the exact recording format is deferred.

## 6. Selected-event inspection

Although global review is the default, the user should be able to select a particular automatic correction or suspicious event and inspect the relevant video context.

For a selected automatic correction, the user should be able to conceptually:

- keep/accept the automatic result;
- deny/revert the correction;
- replace it with a manual correction.

This capability is optional per event. The system must not require the user to make one of these decisions for every automatic correction.

The exact semantics of reverting an event whose downstream state depends on earlier sequential corrections will be designed later with the correction-engine architecture. Gate 2 only establishes that the user must be able to challenge or modify a selected automatic decision in a safe, understandable way.

## 7. Manual correction philosophy

The S3 workspace should provide broad enough editing capability to recover tracking when automatic correction is insufficient.

The MVP therefore follows the spirit of a **full tracking-correction editor**, while maintaining a clear boundary: S3 is not intended to replace a general SLEAP pose-labeling application.

Manual capabilities may include, where required for tracking correction:

- swapping animal identities;
- reassigning a pose/track;
- correcting identity from a chosen point or interval;
- blanking an invalid detection;
- selecting the valid detection when duplicates exist;
- reassigning a node between animals;
- blanking an invalid node;
- moving/correcting a node when necessary to restore valid tracking;
- undoing or revising a previous manual correction.

The exact MVP action set, interaction gestures, and constraints will be defined after the current corrector behavior is audited.

Manual editing should remain focused on restoring tracking correctness and identity continuity. General interpolation, smoothing, gap filling, dense reconstruction, or broad pose annotation remain outside S3.

## 8. Iterative working state

The user should be able to modify the corrected result and continue reviewing before final acceptance.

Conceptually:

```text
automatic corrected state
        ↓
user edit
        ↓
updated working state
        ↓
continued review
        ↓
additional edit / undo / inspection
        ↓
final acceptance
```

A manual edit does not automatically imply rerunning the entire automatic correction algorithm.

Later architecture work must define how manual changes interact safely with sequential correction dependencies, undo behavior, and event provenance.

## 9. Automatic correction philosophy

Automatic correction remains the main engine of S3.

For the current-lab MVP, the goal is to perform almost all routine tracking correction automatically so that the user's main task is verification rather than repair.

The corrector should remain conservative:

- act when the evidence is strong enough;
- avoid speculative corrections;
- surface suspicious unresolved regions when useful;
- preserve enough provenance to understand what changed;
- leave the user free to override a selected automatic decision.

The initial implementation should prioritize preserving the efficacy of the current working correction logic over algorithmic simplification.

## 10. Current-corrector migration principle

The current tracking pipeline is a behavioral reference, not the desired S3 architecture.

The migration strategy is:

```text
current working implementation
        ↓
behavioral inventory
        ↓
golden regression examples
        ↓
clean S3 correction architecture
        ↓
behavior-preserving rewrite
        ↓
regression validation
        ↓
future algorithmic improvement
```

The current code should not simply be copied into the new subsystem unchanged.

At the same time, its validated rules, rule ordering, special cases, state dependencies, and effective thresholds must not be casually rewritten or removed during cleanup.

## 11. Current-profile sequential compatibility

The current corrector is sequential and stateful: corrections applied at an earlier frame can change the interpretation of later frames.

The initial refactor must preserve this behavior so that code restructuring does not reduce tracking efficacy.

This is a compatibility requirement of the first current-lab tracking backend only.

S3 itself is not permanently defined as sequential. Future backends may use global, bidirectional, learned, or otherwise non-sequential tracking methods.

## 12. Tracking profiles

S3 should conceptually support a tracking-profile abstraction so that experiment-specific assumptions and correction approaches do not become permanent global assumptions.

The MVP needs only one active current-lab profile, conceptually similar to:

```text
current_lab_two_mouse_headstage_v1
```

The MVP does **not** require a user-facing profile selector or the ability to rerun the same session interactively through multiple profiles.

Future profiles or backends may be added without changing the overall S3 workflow.

## 13. Training-candidate workflow

During review, the user should be able to mark a frame or short interval as valuable for future SLEAP/model improvement.

This is especially relevant when:

- the underlying pose prediction is too poor for safe tracking correction;
- a hard occlusion or interaction exposes a model failure;
- the user makes a difficult manual correction;
- a suspicious interval would be informative for future labeling.

The saved candidate should retain neighboring context and a reason/category.

S3 records candidates only. Training, dataset construction, active-learning execution, and model optimization remain outside S3.

## 14. What Gate 2 intentionally does not lock

Gate 2 does not yet define:

- `tracked_pose.parquet` columns;
- correction JSON/schema fields;
- exact event/status enums;
- correction-episode aggregation rules;
- UI framework or visual layout;
- exact playback controls;
- exact manual-edit gestures;
- undo implementation;
- how denial of an earlier sequential correction propagates downstream;
- internal class/module architecture;
- parameter configuration format;
- quantitative acceptance thresholds;
- alternative tracking backend implementation.

These decisions should be made only after the current correction behavior is fully audited.

## 15. Gate 2 completion criterion

Gate 2 is complete when the user approves:

1. automatic correction always precedes review;
2. the corrected result is a working state until user acceptance;
3. acceptance means the user confirms tracking stability across the video;
4. the user does not approve every correction by default;
5. selected automatic events may be inspected, denied, or modified;
6. suspicious unresolved regions are useful review targets but are not confirmed errors;
7. manual editing should be powerful enough to restore tracking, including node-level fixes when warranted;
8. the workspace remains a tracking-correction environment rather than a general pose-labeling system;
9. user edits can be iterative before final acceptance;
10. the current correction behavior will be preserved through structured refactoring rather than copied blindly;
11. sequential processing is preserved only as a compatibility constraint of the initial backend;
12. future tracking profiles/backends remain possible;
13. training-candidate marking belongs in S3, while model training does not.
