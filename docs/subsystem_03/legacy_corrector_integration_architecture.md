# Subsystem 03 — Legacy Corrector Integration Architecture

**Status:** Gate 4 review draft  
**Purpose:** Define the minimum architecture needed to integrate the current corrector safely without first rewriting it.

## 1. Architecture objective

The MVP should separate two concerns:

```text
S3 product workflow
    versus
current tracking algorithm implementation
```

The current tracking algorithm is treated as a backend behind a narrow integration boundary.

The rest of S3 should not need to know how the backend internally detects jumps, headstage ownership, skeleton stretch, orientation failures, or other legacy rule details.

## 2. Proposed minimal package structure

```text
src/tracking_correction/
    __init__.py
    runner.py
    contracts.py
    legacy_backend.py
    review/
        __init__.py
        workspace.py
```

This is intentionally small.

The exact filenames may change during implementation if repository conventions require it, but the responsibility split should remain.

## 3. `runner.py`

`runner.py` owns S3 orchestration.

Responsibilities:

- select/validate a completed S2 run;
- locate the S2 pose input and associated prepared video;
- identify the active tracking profile/backend;
- create the S3 working/run directory;
- invoke the backend adapter;
- persist normalized S3 outputs;
- launch or hand off to review;
- record acceptance/failure state;
- never modify S1 or S2 source artifacts.

`runner.py` should not contain tracking heuristics.

## 4. `legacy_backend.py`

`legacy_backend.py` isolates the current corrector from the rest of S3.

Responsibilities:

1. load/prepare S2 `pose.parquet` for the current corrector;
2. normalize only compatibility differences that are required for the legacy code;
3. load the current correction configuration/profile;
4. invoke the existing correction pipeline with minimal behavioral changes;
5. collect:
   - corrected pose;
   - raw correction records;
   - blanked-node information;
   - swapped-frame information;
   - suspicious/unresolved frame information where available;
6. adapt legacy output into S3-owned working output;
7. return a compact backend result to `runner.py`.

The adapter should not redesign correction logic.

## 5. `contracts.py`

For the MVP, `contracts.py` should stay minimal.

It may define only lightweight internal structures needed to keep orchestration independent from the backend, for example conceptual results such as:

```text
BackendResult
    corrected_pose
    correction_records
    suspicious_regions
    warnings
    backend_id
    profile_id
```

This does **not** require an elaborate domain model or complete schema redesign.

The purpose is merely to prevent `runner.py` and review code from importing internal corrector classes directly.

## 6. Existing corrector location

Two acceptable MVP strategies are allowed:

### Strategy A — Keep current modules largely where they are

`legacy_backend.py` imports them and acts as the compatibility boundary.

This is the lowest-risk approach if relocating files would create unnecessary churn.

### Strategy B — Move the current modules under an explicit legacy/profile folder

For example:

```text
src/tracking_correction/backends/current_lab/
    id_corrector_minimal.py
    tracking_data.py
    correction_params.py
```

This is acceptable only if the move is mechanical and does not trigger a broader rewrite.

For MVP speed, Strategy A is preferred unless repository organization makes it awkward.

## 7. Historical finalization behavior

The existing pipeline's useful output-preservation behavior should be reused where practical rather than redesigned immediately.

In particular, the historical output already preserves concepts such as:

- corrected coordinates;
- raw/original coordinates;
- point/instance confidence;
- track-swapped flags;
- node-blanked flags;
- stable frame/track/node ordering.

These concepts are valuable for S3 provenance and regression testing.

Gate 4 does not require preserving the historical output schema byte-for-byte, but the MVP should avoid discarding proven provenance information.

## 8. Review-data preparation

The review layer should receive a corrected working pose plus compact navigation aids.

Raw correction records may remain detailed internally.

Before presentation, S3 may derive lightweight summaries such as:

- suspicious frames/regions;
- frames where automatic track swaps occurred;
- frames with node blanking or reassignment;
- manually edited frames;
- user-marked training candidates.

These are navigation aids, not mandatory review tasks.

## 9. Manual editing architecture

Manual edits should be applied to an S3-owned working copy of the corrected result.

They should not mutate the original S2 input.

The MVP should support iterative editing:

```text
automatic corrected result
        ↓
manual edit
        ↓
updated working result
        ↓
continue review
        ↓
additional edit / undo / accept
```

The exact persistence model for undo/history may start simple, but manual changes must remain distinguishable from automatic backend changes.

## 10. Acceptance state

The S3 result becomes selected/final-for-S3 only after the user explicitly confirms that tracking is stable across the video.

Acceptance belongs to S3 orchestration/review state, not to the legacy corrector.

The backend itself should not decide that a session is accepted.

## 11. Error handling

S3 should distinguish at minimum:

- invalid/missing S2 input;
- backend execution failure;
- backend completed with warnings/suspicious cases;
- review not yet accepted;
- accepted S3 result.

A suspicious region is not the same as backend failure.

## 12. MVP implementation sequence

### Sprint 1 — S3 shell

Implement:

- package/CLI entry point;
- S2 input validation;
- local S3 run directory;
- current-profile selection;
- no review UI yet.

### Sprint 2 — Legacy backend adapter

Implement:

- compatibility loading;
- invocation of the current corrector;
- output normalization;
- preservation of raw/corrected provenance;
- smoke/regression tests against the Gate 3 baseline.

### Sprint 3 — Corrected overlay/review workspace

Implement:

- corrected pose rendering;
- video playback/scrubbing/frame stepping;
- identity-stable colors;
- suspicious-region navigation;
- overall acceptance action.

### Sprint 4 — Manual correction tools

Implement the smallest complete useful editor for:

- identity swap/reassignment;
- blank/remove invalid pose data;
- node-level correction where needed;
- undo/revise;
- mark training candidates.

### Sprint 5 — End-to-end acceptance

Validate:

- S2 → automatic S3 correction;
- review;
- manual fixes;
- acceptance;
- persisted `tracked_pose.parquet`;
- preserved provenance;
- no changes to S1/S2 artifacts.

## 13. Refactor trigger after MVP

Refactoring the backend should be considered only after:

- the end-to-end S3 workflow is operational;
- the Gate 3 regression baseline is reproducible;
- major UX gaps are understood;
- performance bottlenecks or maintenance pain are observed in actual use.

At that point, the adapter boundary established here allows the backend to be replaced incrementally without redesigning the user workflow.
