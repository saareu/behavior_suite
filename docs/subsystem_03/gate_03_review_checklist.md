# Subsystem 03 — Gate 3 Review Checklist

**Document role:** Design-history checklist for Gate 3 regression approval.  
**Implementation status:** MVP implemented / validated — Gate 3 golden baseline
reproduced with zero pose/tracking divergence; see
[`mvp_scope_and_roadmap.md`](mvp_scope_and_roadmap.md).

## Behavioral inventory

- [ ] The current corrector is accurately described as sequential and stateful.
- [ ] Rule ordering and corrected-history dependence are treated as behavior that must initially be preserved.
- [ ] The second headstage pass is preserved as current behavior rather than deleted as apparent duplication.
- [ ] Cache invalidation after mutations is recognized as behaviorally important.
- [ ] Tracking-related coordinate carry-forward is distinguished from general interpolation.
- [ ] Anti-chaining safeguards are explicitly preserved.
- [ ] Suspicious/no-correction states are treated as valid conservative outcomes.

## Current profile assumptions

- [ ] Track 0 represents the implanted/headstage animal in the current profile.
- [ ] Two-track/headstage assumptions are profile-specific, not permanent S3 architecture.
- [ ] Current configured pixel and temporal thresholds are preserved for the first rewrite.
- [ ] Hidden hard-coded thresholds are inventoried before cleanup/generalization.

## Correction rule catalog

- [ ] Whole-track swaps are represented.
- [ ] Node swaps/reassignments are represented.
- [ ] Blank-node/blank-track behavior is represented.
- [ ] Extract-and-blank behavior is represented.
- [ ] Headstage reassignment/duplicate handling is represented.
- [ ] Single-node jump repair is represented.
- [ ] Skeleton stretch repair is represented.
- [ ] Orientation repair is represented.
- [ ] Suspicious but unchanged outcomes are represented.

## Golden regression plan

- [ ] Uploaded `pose.parquet` is accepted as a baseline input example.
- [ ] Uploaded `tracks.parquet` is accepted as its corrected-output baseline.
- [ ] Uploaded `corrections.json` is accepted as a behavior-discovery/provenance reference, not an exact UI-event oracle.
- [ ] Short real intervals will be extracted around representative correction families.
- [ ] At least one suspicious/no-correction interval will be included.
- [ ] At least one anti-chaining case will be included.
- [ ] At least one sequential-dependence case will be included.
- [ ] Full-session regression compares corrected coordinates/identity rather than only correction counts.

## Output preservation

- [ ] Raw versus corrected coordinates remain distinguishable.
- [ ] Node blanking provenance remains available.
- [ ] Track-swap provenance remains available.
- [ ] Confidence/provenance information is preserved where required by the current S2→S3 contract.
- [ ] Historical finalizer behavior is converted into normal automated tests during implementation.

## Gate decision

**Decision:** ☐ Approved  ☐ Approved with edits  ☐ Not approved  
**Reviewer:**  
**Date:**  

### Required edits/comments

1.
2.
3.
