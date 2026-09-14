# Subsystem 03 — Golden Regression Plan

**Status:** Gate 3 review draft  
**Purpose:** Define the evidence needed to prove that the first S3 rewrite preserves current-lab correction behavior before algorithmic improvements begin.

## 1. Principle

The current implementation is not assumed to be architecturally correct or universally optimal. It is, however, the validated working baseline for current-lab experiments.

The first S3 rewrite therefore passes Gate 3/4 regression only when it reproduces the accepted tracking outcome on selected golden cases, except for explicitly approved differences.

The regression target is the **corrected tracking result**, not exact legacy log verbosity.

## 2. Golden artifacts already available

The current review package includes a matched example set:

```text
pose.parquet        # legacy/current Pipeline 20 input example
tracks.parquet      # accepted corrected output example
corrections.json    # legacy correction log
track_ids.yaml      # active parameter profile
```

These should be retained outside normal repo source if they are large, or represented in the repo by a manifest/checksum plus small extracted regression fixtures.

The accepted `tracks.parquet` parquet metadata reports 304,994 rows and the fixed historical finalization schema including raw coordinates and correction flags.

## 3. Regression layers

### Layer A — Unit rule fixtures

Create small synthetic or hand-extracted frame windows for individual behavior families:

- correct headstage ownership;
- headstage on wrong track;
- duplicate headstage;
- full identity swap;
- partial-node reassignment;
- extract-and-blank;
- single-node jump with reliable previous observation;
- single-node jump with anti-chaining blank;
- skeleton stretch repair;
- orientation repair;
- suspicious both-track jump with no correction.

Each fixture should test both the mutation and relevant provenance/suspicious outcome.

### Layer B — Real short intervals

Extract short windows around representative correction episodes from accepted full sessions.

Recommended first windows from the uploaded legacy correction log include examples around:

- the early identity-swap/headstage sequence beginning near frame 382;
- single-node jump and skeleton-stretch events near frames 583 and 615–616;
- identity/jump/overlap activity around frames 619–688;
- large node-jump/stretch behavior around frames 717–718;
- overlap anti-chaining around frames 943–1009;
- node-jump/stretch behavior around frames 1177–1196;
- later sustained identity-swap episodes beginning near frame 1813.

The exact final window boundaries should be selected after visual confirmation against the associated video.

### Layer C — Full-session golden regression

Run the refactored current-lab profile on the same `pose.parquet` used to produce the accepted `tracks.parquet`.

Compare the new output to the accepted output at the level of:

- track assignment;
- node coordinates;
- node presence/blanking;
- raw-coordinate preservation;
- track-swap provenance;
- node-blank provenance;
- stable row/key coverage.

## 4. Primary comparison metrics

The full-session comparison should report at minimum:

```text
rows/keys in baseline and rewrite
missing or extra (frame, track, node) keys
x/y equality or tolerance agreement
raw x/y preservation
point-score preservation
instance-score preservation
was_track_swapped agreement
was_node_blanked agreement
frames with any output divergence
first divergent frame
contiguous divergence intervals
```

Exact numerical tolerance should be zero or near-zero for deterministic coordinate-copy/swap operations unless a dtype conversion explains a small float representation difference.

## 5. Correction-log comparison policy

Legacy `corrections.json` is useful for discovering behavior but should not be the primary equality target.

The rewrite may legitimately change logging representation while preserving tracking output. Therefore compare logs at two levels:

1. **semantic action coverage** — did the same meaningful correction families occur at the same frames/intervals?;
2. **raw record counts** — diagnostic only, because the legacy log may contain duplicate or per-frame repeated records.

A reduction in duplicate log entries is acceptable only if corrected pose/tracking output and correction provenance remain equivalent.

## 6. Suspicious/no-correction regression

The golden set must include frames where the current algorithm intentionally detects an abnormal state but does not mutate the data.

For these cases, the rewrite should preserve two properties:

- no speculative correction is introduced;
- the condition remains discoverable as a review candidate if the new architecture exposes suspicious outcomes explicitly.

## 7. Sequential-dependence regression

At least one fixture must demonstrate that changing an earlier frame alters later interpretation.

This test exists specifically to prevent a future cleanup from accidentally replacing the current profile with independent per-frame evaluation.

The fixture should verify that:

- a frame is corrected;
- caches/state are invalidated;
- a subsequent frame uses corrected rather than original history;
- output changes if the earlier mutation is omitted.

## 8. Headstage identity regression

At least one real interval must verify that:

- track 0 retains implanted/headstage identity;
- wrong-track headstage detections are reassigned or removed as in the baseline;
- duplicate headstage detections do not create two biological implanted identities;
- recovery from original input behaves equivalently where exercised.

## 9. Anti-chaining regression

Include at least one sequence where the legacy algorithm:

1. restores a node from the previous frame;
2. encounters another bad frame immediately after;
3. avoids indefinitely copying the already-corrected coordinate and blanks instead when appropriate.

This is a high-value scientific safeguard and should have an explicit test.

## 10. Output-finalization regression

The historical finalizer behavior should be tested separately from the correction algorithm.

Tests should verify:

- stable sort by `(frame, track, node)`;
- expected dtypes;
- raw coordinate columns preserved;
- implicit blanking detected when raw coordinates exist but final coordinates are NaN;
- explicit blanked-node markers honored;
- frame-level swapped flag set for swap frames;
- required output columns present.

The historical helper already contains a small self-test; the S3 rewrite should convert these expectations into normal automated tests rather than relying on an executable helper self-test.

## 11. Golden data storage strategy

Do not commit large full-session parquet/video artifacts to the main repository unless repository policy explicitly allows it.

Preferred strategy:

```text
repo:
  docs/subsystem_03/evidence/golden_cases.yaml
  tests/tracking_correction/fixtures/<small windows>

external/local run storage:
  full pose.parquet
  accepted tracks.parquet
  source video
  full corrections.json
```

The manifest should record paths or logical IDs plus hashes/checksums so the exact full-session baseline can be identified reproducibly.

## 12. Gate 3 acceptance criteria

Gate 3 is complete when:

1. current behavior has been inventoried at rule-family level;
2. stateful/sequential dependencies are documented;
3. active configuration and hidden hard-coded thresholds are acknowledged;
4. accepted input/output/log artifacts are identified;
5. representative short windows are selected for each major behavior family;
6. a full-session baseline is designated;
7. suspicious/no-correction cases are included;
8. anti-chaining behavior is explicitly covered;
9. the regression comparison plan is approved;
10. no refactor has yet changed the baseline algorithm.

## 13. What happens after Gate 3

After approval, Gate 4 can define the clean internal architecture and output contract needed to implement the same behavior safely.

Only after the behavior-preserving rewrite passes golden regression should we begin deliberate algorithm simplification, threshold redesign, generalized profiles, or alternative tracking backends.
