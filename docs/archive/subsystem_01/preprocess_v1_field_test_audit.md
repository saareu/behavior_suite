# Preprocess Subsystem v1 Field-Test Audit

> **Archive notice:** This historical document is retained for traceability. It is not the current source of truth. See `docs/subsystem_01_preprocessing.md`, `docs/subsystem_01_status_and_roadmap.md`, and `docs/design/subsystem_01_geometry_modes.md`.

**Audit date:** 2026-06-22  
**Repository revision inspected:** `62e0014` (`Complete preprocess GUI workflow`)  
**Audit type:** Documentation-only implementation and first-field-test review

## 1. Purpose and scope

This audit records the implemented state of the Preprocess Subsystem after its
first successful real-video GUI field test. It compares the approved v1 design,
current source, current tests, recent repository history, and the retained field
test artifacts.

This document does not change v1 behavior, approve v2 behavior, or treat a
candidate improvement as implemented. Field-test observations are identified as
observations; possible causes that have not been reproduced are identified as
hypotheses.

## 2. Current v1 implementation inventory

The repository currently implements:

- Shared project creation, opening, validation, and preprocess-directory paths.
- Strict typed preprocessing configuration, including trim, pre-crop, detector,
  canonical geometry, the validated encoding profile, background, and debug
  settings.
- Raw-video probing through OpenCV and ffprobe, with optional full sequential
  OpenCV-readable frame counting.
- MATLAB v7.2-and-earlier and v7.3 workspace loading, numeric-vector listing,
  explicit variable and unit selection, exact length validation, and temporal
  unit conversion.
- A shared validated `CropPlan` for automatic cage detection and manual
  four-corner crop.
- Detection pre-crop modes in raw-frame coordinates.
- Uniform canonical scaling and centered black padding.
- Two-stage video preparation: ffmpeg Stage A followed by sequential OpenCV
  Stage B re-encoding.
- Strict prepared-video validation, prepared synchronization writing, prepared
  background generation, authoritative metadata, accepted settings, and logging.
- A single `PreprocessService.run()` orchestration path used by the CLI and GUI.
- CLI commands for detecting, explicitly accepting, and running with a
  versioned `CropPlan` document.
- A seven-page GUI: Project, Raw Video, Trim and Pre-Crop, Timing, Crop Review,
  Encode Settings, and Run and Validate.
- Reusable GUI background tasks for long probes, cage detection, and the full
  preprocessing service run.

The following are not implemented:

- Static masking; configuration explicitly rejects enabled masking.
- Raw PTS extraction; synchronization records `not_extracted` and NaN values.
- Persistent raw-video probe or readable-count caching.
- Visual trim navigation or a visual pre-crop picker.
- Timing-unit plausibility confirmation and writer-profile FPS preflight.
- Service-level progress callbacks, frame progress, or ETA.
- Batch processing, SLEAP inference, or downstream analysis.

## 3. Implemented workflow

The current GUI workflow is:

1. Create or open a validated project with a `preprocess` directory.
2. Select a raw video and probe it. A full sequential OpenCV-readable frame
   count is optional on this page.
3. Enter the raw decode frame range using `[start, end)` semantics and configure
   a typed pre-crop.
4. Choose no external timing or load a MATLAB workspace, select one candidate,
   declare units, obtain a full sequential OpenCV-readable frame count when
   required, and validate the selected external timing.
5. Generate an automatic or manual crop candidate, review a prepared preview,
   and explicitly accept the `CropPlan`.
6. Review or edit the supported typed canonical and encoding settings.
7. Review inputs and official output paths, then run `PreprocessService` in a
   background task.
8. The service reopens the project, validates configuration, probes the raw
   video, validates trim and external timing, resolves FPS, enforces crop
   acceptance, performs Stage A and Stage B, validates the prepared video,
   writes synchronization and background artifacts, writes settings and
   metadata, reloads key artifacts, and verifies all official paths.
9. The GUI reports success only from a successful typed service result.

The CLI uses the same core service but has a separate explicit
detect-crop/accept-crop/run workflow.

## 4. Current official artifacts

A successful run produces exactly:

| Artifact | Current role |
| --- | --- |
| `prepared_video.mp4` | Final OpenCV-readable prepared video for SLEAP |
| `prepare_meta.json` | Authoritative run metadata, geometry, timing, validation, provenance, and output paths |
| `prepared_sync.npz` | Deterministic prepared frame index to raw decode frame index mapping and timing arrays |
| `cropped_background.png` | Median background derived from the final prepared video |
| `settings_used.yaml` | Validated accepted configuration used by the run; no `CropPlan` geometry |
| `processing_log.txt` | Append-mode stage, FPS-selection, success, warning, and failure log |

`.internal` and `debug` content is not official. The service removes its Stage A
intermediate after a successful run when debug mode is disabled.

## 5. Current safety guarantees

Confirmed implemented guarantees include:

- Raw decode order is the primary frame identity.
- Default processing uses no frame resampling.
- The mapping is `raw_decode_frame_idx = start_frame + prepared_frame_idx`.
- External timing length must exactly equal the original untrimmed full
  sequential OpenCV-readable frame count.
- Non-finite, non-numeric, non-vector, non-monotonic, and length-mismatched
  external timing is rejected.
- `frames` and `unknown` units do not invent seconds.
- A `CropPlan` and separate request confirmation must both state explicit crop
  acceptance before Stage A begins.
- Pre-crop geometry remains in raw-frame coordinates and participates in the
  final transform.
- Canonical resolution uses one uniform scale and centered padding.
- Stage B reads and writes sequentially and does not resize or resample frames.
- Prepared OpenCV-reported and sequentially readable counts must match each
  other and the expected trimmed count.
- Prepared dimensions must match the accepted `CropPlan` and be even.
- Sync arrays must have consistent lengths and deterministic frame mapping.
- Background dimensions must match the prepared video.
- Metadata must validate, be strict JSON, and record accepted geometry.
- Key artifacts use same-directory temporary files and atomic replacement.
- A failed service result is not presented as successful, and new partial
  official outputs are cleaned where the service owns them.

There is no current guarantee that every finite positive FPS is representable
by the selected Stage B writer profile. The field test exposed this gap.

## 6. Test and validation status

Repository evidence includes 260 Python test functions across project, core,
CLI, integration, and GUI test modules. Parametrization expands this inventory.
The most recent full-suite run immediately before this audit collected 335 tests
and reported `335 passed`; repository-wide Ruff and `git diff --check` also
passed. This documentation-only task did not rerun the test suite.

The tests cover, among other behavior:

- Project and configuration validation.
- OpenCV probing and sequential readable counts.
- MATLAB loading, timing conversion, monotonicity, and exact length rejection.
- `CropPlan`, pre-crop, canonical geometry, automatic crop, and manual crop.
- ffmpeg filtergraph geometry and Stage A integration.
- OpenCV Stage B sequencing, failure handling, and atomic output preservation.
- Strict prepared-video validation.
- Sync, background, metadata, settings, logging, service, and CLI behavior.
- GUI setup, Timing, Crop Review, Encode Settings, Run/Validate, task threading,
  invalidation, and duplicate-run prevention without requiring an active display.

Not yet covered by an approved real-video regression protocol are persistent
probe caching, timing-unit mismatch warnings, writer-profile FPS preflight,
direct GUI-preview versus final-frame parity, captured MAT warnings, and
service-level progress callbacks.

## 7. Real-world field-test summary

One real video was processed through the completed GUI workflow on 2026-06-22.
The first run used an incorrect external timing unit. The selected FPS was:

```text
119402.985075, source=external_time
```

Stage A completed, but Stage B could not open its temporary OpenCV
`VideoWriter`. The observed backend diagnostic included:

```text
timebase 1000/119403 not supported by MPEG 4 standard
```

The timing units were corrected without changing frame-identity rules. The
second run selected approximately `119.402985075` FPS from external timing and
completed successfully.

This is one successful acceptance observation, not yet a representative
real-video regression suite.

## 8. Successful field-test outcome

The retained field-test directory contains all six official artifacts. Its
authoritative metadata records:

```text
schema_version: prepare_meta_v1
validation.status: passed
prepared OpenCV-reported frame count: 45716
prepared sequential readable frame count: 45716
prepared dimensions: 928 × 528
prepared FPS: approximately 119.402985075
FPS source: external_time
selected timing units: seconds
external timing status: valid
```

The processing log reaches `final_artifact_validation` and records
`run completed status=success frames=45716`. This confirms that the corrected
field-test run produced and validated the full official artifact set.

## 9. Timing-unit failure analysis

### Confirmed facts

- The selected timing vector passed numeric, shape, finiteness, monotonicity,
  and exact raw-readable-count validation under the declared unit.
- Unit conversion produced an external timing median interval corresponding to
  approximately `119402.985075` FPS.
- The service accepts any external effective FPS that is finite and positive.
- Stage A ran before the Stage B writer rejected the selected FPS/timebase.
- The service returned failure and did not claim successful artifacts.
- Correcting the unit produced approximately `119.402985075` FPS and success.

### Interpretation

The observed failure was writer initialization caused by an implausible and
unrepresentable output FPS. It was not a raw decode frame index or prepared
frame index mapping error. Decode-order mapping remained the primary identity,
and no resampling occurred.

The wrong unit was not automatically detectable from vector validity alone.
The missing controls are a converted-unit preview, a comparison with raw nominal
video FPS, explicit confirmation of an unusual mismatch, and a writer-profile
representability gate before lengthy Stage A processing.

The v1 specification states that safe-FPS representation should ensure the FPS
is representable by the selected OpenCV pathway. Current source only validates
that Stage B FPS is finite and positive. This is a confirmed specification-to-
implementation gap.

## 10. MAT duplicate-variable warning analysis

### Confirmed field observation

SciPy emitted a nonfatal warning concerning duplicate variable name `"None"`
while loading the field-test MAT workspace. A valid intended vector could still
be selected and used successfully.

### Confirmed implementation behavior

- SciPy `loadmat` warnings are not captured.
- Such warnings can therefore appear only on the launching console rather than
  as structured GUI warnings or processing-log entries.
- Workspace keys beginning with `__` are excluded.
- Numeric non-empty one-dimensional values are eligible candidates regardless
  of variable name; there is no explicit unusable `None`/duplicate-name filter.
- Validation is applied to the selected vector, so unrelated non-candidate
  workspace entries do not by themselves invalidate a good selection.

### Hypothesis requiring reproduction

The warning likely originates from duplicate or malformed variable naming in
the MAT workspace as interpreted by SciPy. The audit does not establish whether
SciPy retained a key named `None`, replaced one duplicate with another, or
whether either value was numerically eligible. A controlled fixture is required
before specifying exact parser behavior.

## 11. UX and GUI gaps discovered

Confirmed current gaps are:

- The Raw Video page explicitly selects and probes a path but does not display
  video frames or provide go-to-frame/step controls.
- Trim uses numeric inputs only; there is no representative video viewer or
  current-frame action for inclusive start and exclusive end.
- Pre-crop uses numeric boundary or rectangle controls only; there is no visual
  boundary or rectangle picker.
- Timing shows candidate raw numeric median difference and an unconverted
  candidate FPS. It does not show the selected unit's converted median interval,
  converted timing FPS, raw nominal FPS, or a strong mismatch warning.
- Detector settings have no distinct actions for restoring typed defaults and
  restoring configuration-loaded values.
- MAT warnings are not surfaced in the GUI.

The field report stated that a full count performed on Screen 2 was repeated on
the Timing screen. Current source retains and uses an in-memory
`raw_probe.frame_count_opencv_readable` for Timing summaries and validation.
However, the Timing page leaves `Count All Readable Raw Frames Now` enabled and
the controller permits another count request even when that valid same-session
count exists. Reuse is therefore partially working: the value is not lost, but
the GUI still offers and can execute a redundant count. The count is also not
persistent, has no source fingerprint, and is not reused by
`PreprocessService`, which performs a fresh full probe whenever external timing
is provided. Persistent cross-workflow reuse remains a v2 requirement.

## 12. Preview-versus-final-output discrepancy

### Confirmed field observation

The GUI crop preview appeared to contain extra vertical area that was not
visible in the final prepared video.

### Current code evidence

The GUI preview applies the accepted candidate's exact
`H_raw_to_prepared_3x3` through OpenCV `warpPerspective` at
`CropPlan.prepared_size_wh`. Stage A separately decomposes the same `CropPlan`
into ffmpeg pre-crop, perspective, scaling, rotation, even-dimension, canonical
scale, and padding operations. The preview widget also scales the resulting
image for display.

### Audit conclusion

The observation is real, but its cause is not established. Plausible hypotheses
are display-widget letterboxing/scaling, OpenCV-versus-ffmpeg pixel-center or
perspective convention differences, or an error in Stage A transform
decomposition. The final output's appearance does not prove which component is
wrong. A direct regression comparison using the same raw decode frame index and
accepted `CropPlan` is required.

## 13. Progress-reporting limitations

The GUI currently reports only:

```text
Preparing request
Running preprocessing pipeline
Validating outputs
Finished
```

It also reports elapsed wall time and an indeterminate activity indicator. The
service writes more detailed stages to `processing_log.txt`, but it exposes no
progress callback to the GUI. Stage A ffmpeg progress, Stage B processed frames,
background processed frames, expected counts, throughput, percentage, and ETA
are not available interactively. No fake percentage or ETA is shown.

## 14. Items explicitly deferred from v1

The approved v1 scope and current implementation defer:

- SLEAP inference and tracking.
- Pose QC/correction and downstream behavioral analysis.
- Batch preprocessing.
- Image-sequence export.
- Automatic timing-variable selection by name.
- Automatic synchronization or repair of unrelated timing vectors.
- Frame resampling.
- Raw source copying, checksumming, and archival.
- Interactive rotated-ROI adjustment after manual crop.
- Alternative unvalidated processing recipes.
- Static and dynamic masking; static masking is architecturally reserved only.
- Raw PTS extraction in the current implementation.

## 15. Known limitations and risk classification

| Limitation | Evidence status | Risk |
| --- | --- | --- |
| No timing-unit plausibility preview or confirmation | Confirmed | High: a wrong but structurally valid unit can select an implausible playback/timing FPS |
| No writer-profile FPS representability preflight before Stage A | Confirmed and field-triggered | High operational cost; failure is loud, but occurs after lengthy work |
| Preview/final visual discrepancy | Confirmed observation; cause unconfirmed | High until parity is demonstrated because crop review is an explicit scientific approval gate |
| No persistent probe/readable-count cache; service re-counts with external timing | Confirmed | Medium usability and runtime cost; low direct scientific risk if fresh counts remain authoritative |
| Screen-2-to-Timing repeat count report | Partially working: stored value is reused, but the page and controller permit a redundant recount | Medium usability and runtime cost |
| MAT warnings are console-only and `None`/duplicate candidates are not explicitly excluded | Confirmed implementation gap; exact workspace behavior unconfirmed | Medium observability and selection clarity |
| No visual trim/pre-crop navigation | Confirmed | Medium usability and user-entry risk; core validation remains active |
| No detector reset actions | Confirmed | Medium reproducibility/usability; current typed changes still invalidate crop acceptance |
| Coarse progress only | Confirmed | Medium usability, low scientific risk |
| ffmpeg/ffprobe may fall back to system `PATH` | Confirmed documentation/implementation drift | Medium reproducibility risk |
| Raw PTS remains `not_extracted` | Confirmed deferred behavior | Low for decode-order identity; diagnostic capability is absent |
| One real-video success is not a regression dataset | Confirmed | High release-confidence limitation |

## 16. Recommendation

Freeze v1 scientific and artifact behavior at the current revision. Do not
patch the observed issues ad hoc in the field-tested path. First execute and
record an approved acceptance-test protocol across representative real videos,
including the successful field case and controlled failure cases. Then review
and approve `preprocess_v2_requirements_draft.md` before creating a v2
implementation plan.

This sequence preserves the validated v1 design laws while allowing v2 to
address timing safety, writer preflight, preview parity, persistent measurement
reuse, and workflow usability as explicit reviewed requirements.
