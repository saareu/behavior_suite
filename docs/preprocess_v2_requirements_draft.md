# Preprocess Subsystem v2 Requirements Draft

**Status:** Draft for review; not approved  
**Document type:** Requirements only; not an implementation plan  
**Evidence source:** v1 implementation audit and first real-video GUI field test

## 1. Purpose

This draft defines candidate v2 outcomes without prescribing code structure or
an implementation sequence. No item is approved until reviewed against the v1
field-test audit and acceptance-test evidence.

All v2 work must preserve these v1 design laws:

- Raw decode order is the primary frame identity.
- `raw_decode_frame_idx = start_frame + prepared_frame_idx` in the default mode.
- No frame resampling, dropping, duplication, interpolation, or reordering.
- No silent repair or guessed timing units.
- External timing length must exactly equal the original untrimmed full
  sequential OpenCV-readable frame count.
- Crop processing requires explicit acceptance of a validated `CropPlan`.
- `prepare_meta.json` remains authoritative.
- Official artifact names and roles remain unchanged unless separately approved.

## 2. Priority summary

### P0 — Scientific correctness and safety

- D. Timing-unit safety preview and explicit unusual-mismatch confirmation.
- E. Stage-B FPS safety and early writer-profile representability validation.
- G. GUI preview and final prepared-output parity.
- H. Structured MAT warning handling and candidate hygiene.
- J. Real-video acceptance and regression testing.

### P1 — Essential workflow usability

- A. Persistent raw-video probe cache shared by GUI and CLI.
- B. Visual raw decode frame navigation and trim selection.
- C. Visual pre-crop picker using existing raw-coordinate semantics.
- F. Distinct detector-settings reset actions.

### P2 — Quality of life and observability

- I. Structured service progress, measured frame progress, and honest ETA.

### Deferred / out of scope

- Frame resampling or automatic frame/timing repair.
- Automatic timing-unit changes or guesses.
- Automatic timing-variable selection by variable name.
- New official artifacts or metadata-schema changes without separate approval.
- Masking, batch preprocessing, raw PTS extraction, SLEAP inference, pose QC,
  and downstream behavioral analysis unless independently scoped and approved.

## 3. P0 — Scientific correctness and safety

### D. Timing-unit safety preview

#### Problem

The selected timing vector can be structurally valid under an incorrect declared
unit. Current candidate rows show the raw numeric median difference and a raw
`1 / median_difference` value, but do not show the selected unit's converted
interval, converted external timing FPS, raw nominal video FPS, or their
mismatch. The field test selected an implausible FPS of approximately 119403
before correction to approximately 119.403.

#### Desired behavior

Before timing validation is accepted, the interface shall display:

- Raw timing median interval in the vector's stored units.
- Converted median interval in seconds for temporal units.
- Estimated external timing FPS after unit conversion.
- Raw nominal video FPS and its source.
- A prominent warning when the converted external timing FPS is implausibly
  different from raw nominal video FPS.

The system shall not automatically change units. An unusual mismatch shall
require explicit user confirmation and a concise user-entered or selected
rationale. The confirmation and rationale shall be logged and represented in
authoritative run provenance under an approved schema revision if required.

#### Scientific or usability rationale

Unit selection is an experimental interpretation, not a value the software may
guess. Showing the conversion makes the consequence reviewable while preserving
user authority and the no-silent-repair law. Comparison with raw nominal FPS is
a safety signal, not a replacement timing source.

#### Acceptance criteria

- Each temporal unit produces the correct converted interval and FPS preview.
- `frames` and `unknown` clearly state that no seconds/FPS conversion exists.
- Raw nominal FPS and its provenance are visible beside external timing FPS.
- An approved mismatch policy produces a strong warning for the field-test
  wrong-unit case and no warning for the corrected case.
- The warning does not change units or timing data.
- Unusual mismatch cannot be accepted without explicit confirmation and a
  recorded rationale.
- Normal agreement requires no exceptional confirmation.
- Tests cover seconds, milliseconds, microseconds, nanoseconds, frames,
  unknown, missing raw FPS, boundary cases, and confirmation persistence.

#### Dependencies

- An approved definition of unusual or implausible FPS mismatch.
- Reliable raw nominal FPS provenance from the current probe.
- Review of whether confirmation provenance requires a metadata schema revision.
- Requirement E, so a plausible comparison is not confused with writer support.

### E. Stage-B FPS safety

#### Problem

Current Stage B validation requires FPS to be finite and positive but does not
establish that it is representable by the selected OpenCV/container/backend
profile. The field-test wrong-unit value passed service timing validation,
completed lengthy Stage A processing, and then failed when Stage B could not
open its writer.

#### Desired behavior

Before lengthy Stage A processing, the selected output FPS shall be validated as:

- Numeric, finite, and positive.
- Compatible with the selected validated OpenCV/container/codec profile.
- Representable by the underlying writer/timebase pathway used for the run.

An impossible FPS shall fail early with a readable explanation that identifies
the selected value, writer profile, and likely causes such as incorrect external
timing units. The system shall not repair FPS by resampling frames or silently
substituting a different timing source.

#### Scientific or usability rationale

Early failure avoids expensive work and makes a unit mistake diagnosable. It
must not weaken external timing priority or alter the deterministic frame map.

#### Acceptance criteria

- The field-test value near 119403 FPS is rejected before Stage A begins for the
  v1 MP4/mp4v writer profile.
- The corrected value near 119.403 FPS passes preflight for that profile.
- Non-finite, zero, negative, overflow, and unsupported timebase values fail
  with domain-level messages.
- A failure identifies likely unit-selection error without claiming certainty.
- No frame is decoded or encoded by Stage A after a failed FPS preflight.
- No fallback changes units, selects raw FPS, or introduces resampling.
- Tests cover supported boundary values and backend/profile-specific rejection.

#### Dependencies

- A documented capability contract for each supported writer profile.
- A deterministic preflight method that reflects the deployed OpenCV/backend
  environment.
- Requirement D for user-facing timing interpretation.

### G. Preview/output parity

#### Problem

The field-test GUI preview appeared to contain extra vertical area that was not
present in the final prepared video. Current preview and Stage A use the same
accepted `CropPlan` but execute geometry through different rendering pathways.
The discrepancy may be display scaling or image-geometry divergence; the cause
is not confirmed.

#### Desired behavior

The prepared preview shall represent the exact accepted `CropPlan` geometry that
will be used for final processing. Display scaling, widget letterboxing, and
background chrome shall be visually distinguishable from pixels in the preview
image. The final prepared frame corresponding to the same raw decode frame index
shall match preview geometry within an approved codec/interpolation tolerance.

#### Scientific or usability rationale

Explicit crop acceptance is a scientific safety gate. A user cannot give
meaningful acceptance if the preview is visually ambiguous or differs from the
prepared result.

#### Acceptance criteria

- Preview pixel dimensions exactly equal `CropPlan.prepared_size_wh`.
- Preview uses the accepted `H_raw_to_prepared_3x3` and approved interpolation.
- The UI clearly marks any non-image display margins or letterboxing.
- A regression test renders a known raw decode frame index through preview and
  final preparation paths and verifies dimensions, transformed landmarks, crop
  boundaries, rotation, scale, and padding.
- Pixel comparisons use a documented tolerance for codec and interpolation
  effects while geometry comparisons remain strict.
- Tests include horizontal and vertical canonical padding, rotation, automatic
  crop, manual crop, and each pre-crop orientation.
- The field-test discrepancy is either reproduced and fixed or documented as a
  display-only misunderstanding with evidence.

#### Dependencies

- A representative-frame test fixture with known landmarks.
- A defined OpenCV-versus-ffmpeg comparison tolerance.
- Requirement J for real-video acceptance evidence.

### H. MAT warning handling

#### Problem

SciPy can emit nonfatal warnings while parsing a MAT workspace, including the
observed duplicate variable name `"None"` warning. Current loading does not
capture warnings, so they may appear only in the launching console. Candidate
filtering does not explicitly exclude unusable `None`/duplicate-name entries.

#### Desired behavior

MAT parser warnings shall be captured as structured warning records. Relevant
warnings shall be displayed in the GUI and written to diagnostic logs. Unusable
`None`, malformed-name, or ambiguous duplicate candidates shall not be offered
for selection. A valid selected vector shall remain usable when unrelated
workspace entries are malformed and do not compromise the selected value.

#### Scientific or usability rationale

Warnings must be visible and traceable without converting unrelated workspace
noise into a hard failure. Candidate selection must not present ambiguous data.

#### Acceptance criteria

- A fixture reproduces the duplicate/`None` warning behavior for the supported
  SciPy version.
- Captured warnings appear in the Timing UI and an appropriate diagnostic log.
- Warning text is concise for users while full technical text remains available.
- Unusable `None` and ambiguous duplicate candidates are excluded.
- A valid independent selected vector still validates and runs.
- Warnings that directly undermine the selected vector block acceptance with a
  clear reason.
- HDF5 MAT loading receives equivalent structured warning/error treatment.

#### Dependencies

- Reproducible MAT fixtures for SciPy and HDF5 pathways.
- A reviewed candidate-name and duplicate-resolution policy.
- A logging destination available before the preprocessing run starts.

### J. Acceptance and regression testing

#### Problem

The repository has strong unit and synthetic integration coverage but only one
documented successful real-video GUI field case. The observed timing failure,
MAT warning, and preview discrepancy are not represented by approved fixtures or
an acceptance protocol.

#### Desired behavior

Define a repeatable real-video acceptance-test protocol and maintain regression
fixtures for safety-critical v2 behavior. Acceptance evidence shall distinguish
scientific invariants, visual review, performance observations, warnings, and
environment details.

#### Scientific or usability rationale

Synthetic tests cannot fully characterize deployed codecs, OpenCV backends, MAT
parser behavior, large-video runtimes, or human crop review. Release confidence
requires repeatable evidence without weakening deterministic checks.

#### Acceptance criteria

- The protocol records source fingerprint, environment, config, selected raw
  decode frame range, external timing provenance, accepted `CropPlan`, output
  validation, and artifact checks.
- It verifies all six official artifacts and treats `prepare_meta.json` as
  authoritative.
- It verifies exact prepared/raw index mapping and exact external timing length.
- It includes cases for cached and newly measured raw counts, timing-unit
  mismatch and confirmation, unrepresentable FPS rejection before Stage A,
  preview/final parity, captured MAT warnings, and progress callbacks.
- It includes no-external-timing, temporal external timing, frames/unknown
  timing, automatic crop, manual crop, pre-crop, canonical padding, and failure
  cases.
- Fixtures are licensed, privacy-reviewed, versioned, and small enough for their
  intended test tier, or referenced through a controlled test-data manifest.
- Pass/fail criteria and allowed image tolerances are approved before release.

#### Dependencies

- Approved representative datasets and data-handling policy.
- Requirements A, D, E, G, H, and I.
- A recorded dependency/backend matrix for ffmpeg, OpenCV, SciPy, and h5py.

## 4. P1 — Essential workflow usability

### A. Persistent raw-video probe cache

#### Problem

A full sequential OpenCV-readable frame count can take several minutes. The GUI
can reuse the result in memory, but there is no persistent cache shared across
sessions or callers, and the service performs a new full probe when external
timing is supplied.

#### Desired behavior

GUI and CLI shall use a shared persistent raw-video probe cache. A cached full
sequential OpenCV-readable frame count may be stored only after a successful
decode from the first frame through end of stream. The source fingerprint shall
include:

- Resolved source path.
- File size.
- File modification time.

The cache shall invalidate when any fingerprint field changes. The UI and CLI
shall clearly state whether a result was newly measured or loaded from cache.
The cache is operational data, not an official scientific output artifact.

#### Scientific or usability rationale

Safe reuse avoids repeated long decodes while binding the measurement to a
specific source-file state. Failed or partial counts must never become trusted.

#### Acceptance criteria

- A successful full decode stores the readable count and fingerprint.
- A failed, cancelled, interrupted, or partial decode stores no reusable count.
- An exact fingerprint match returns the cached count to GUI and CLI and labels
  it `cached`.
- A cache miss performs a new full decode and labels it `newly measured`.
- Path, size, or modification-time change invalidates the entry.
- A changed raw-video selection cannot inherit another file's count.
- Corrupt or incompatible cache data is ignored with a warning and remeasured.
- `prepared_sync.npz`, `prepare_meta.json`, and official artifact enumeration do
  not treat the cache as an official output.
- Concurrency does not expose partially written cache records.

#### Dependencies

- An approved non-official cache location and lifecycle policy.
- Source fingerprint normalization rules across supported platforms.
- Integration with service, GUI, and CLI probe entry points.

### B. Visual frame navigation and trim selection

#### Problem

Trim selection is currently numeric only. Users cannot visually inspect a raw
decode frame before entering inclusive start or exclusive end indices.

#### Desired behavior

Provide a representative raw-video viewer with:

- Current raw decode frame index.
- Go-to-frame control.
- Single-frame and approved larger-step controls.
- Set inclusive start frame from the current raw decode frame index.
- Set end-exclusive frame from the current raw decode frame index.
- Clear persistent display of `[start, end)` semantics and selected count when
  the required source count is known.

Navigation shall address raw decode frame indices. A time-based slider may be a
visual convenience only if it resolves through explicit decode navigation; it
shall not infer scientific frame identity from timestamps or duration.

#### Scientific or usability rationale

Visual selection reduces transcription errors while preserving decode-order
identity and exclusive-end semantics.

#### Acceptance criteria

- Go-to-frame displays the requested raw decode frame or reports it unreadable.
- Step controls update the displayed raw decode frame index deterministically.
- Set-start records the displayed index as inclusive start.
- Set-end-exclusive records the displayed index as exclusive end and clearly
  states that the displayed frame is excluded.
- Invalid or empty `[start, end)` ranges remain blocked by existing core rules.
- Changing trim clears crop acceptance and prior run-result display.
- No operation derives raw decode frame identity from PTS, playback time, or a
  percentage position.
- Tests cover first/last boundaries, unknown counts, failed seeks, and
  inclusive/exclusive off-by-one behavior.

#### Dependencies

- Reliable representative-frame decoding by raw decode frame index.
- Requirement A when exact upper bounds or selected counts require a full count.
- Existing trim validation and GUI invalidation rules.

### C. Visual pre-crop picker

#### Problem

Directional pre-crop boundaries and manual rectangles are entered numerically,
without visual feedback on a raw frame.

#### Desired behavior

The raw-frame viewer shall allow visual selection of:

- Vertical keep-left and keep-right boundaries.
- Horizontal keep-upper and keep-lower boundaries.
- A manual rectangle.

The picker shall produce values consumed by the existing validated pre-crop
logic. Coordinates shall remain raw-frame coordinates with current half-open
boundary/rectangle semantics. The picker shall not introduce a second geometry
interpretation.

#### Scientific or usability rationale

Visual selection improves usability while keeping the approved pre-crop as a
permanent part of final raw-to-prepared geometry.

#### Acceptance criteria

- The visual overlay and numeric values remain synchronized in both directions.
- Every mode resolves through the existing core pre-crop validation behavior.
- Coordinates are clamped only for display interaction; invalid requested
  geometry is not silently repaired when committed.
- The resolved ROI is shown in raw-frame `(x, y, width, height)` coordinates.
- Changing any pre-crop value clears crop candidates, accepted crop, and prior
  run-result display.
- Regression tests cover all modes, frame edges, manual rectangle bounds, and
  raw-to-pre-crop transform metadata.

#### Dependencies

- Requirement B's raw-frame viewer and coordinate mapping.
- Existing pre-crop core validation and invalidation behavior.

### F. Detector settings reset

#### Problem

Detector settings can be edited and retried, but there is no clear way to
restore library defaults or the values loaded from the session configuration.

#### Desired behavior

Provide two distinct actions:

1. Restore detector defaults defined by the current validated configuration
   schema/version.
2. Restore detector settings loaded at the start of the current configuration
   session.

The UI shall explain which source each action uses and show whether current
values differ from the loaded configuration.

#### Scientific or usability rationale

Distinct reset sources make retries reproducible and prevent ambiguity between
software defaults and experiment-specific loaded settings.

#### Acceptance criteria

- Restore defaults yields a typed configuration equal to current schema
  defaults for detector-related fields.
- Restore loaded yields a typed configuration equal to session-loaded
  detector-related fields.
- The two actions remain distinct when loaded settings differ from defaults.
- Neither action silently writes the source YAML file.
- Any effective detector-setting change clears candidate crop, accepted crop,
  and prior run result and requires detection/review again.
- Invalid reset source data fails visibly without mutating the active typed
  configuration.

#### Dependencies

- Retention of both typed schema defaults and typed session-loaded settings.
- Existing detector config validation and crop invalidation policy.

## 5. P2 — Quality of life and observability

### I. Structured progress and ETA

#### Problem

The GUI shows one indeterminate pipeline stage for most of a run. Detailed
service stages exist only in the processing log; frame counts and throughput are
not reported interactively.

#### Desired behavior

The service shall expose a structured progress-event interface usable by GUI,
CLI, and future callers. Events shall support:

- Current stage and message.
- Processed and expected frames when both are known.
- Elapsed time.
- Stage A ffmpeg progress.
- Stage B sequential frame progress.
- Background-generation progress.
- Completion or failure state.

Percentage shall be shown only when a valid numerator and denominator are
available. ETA shall be shown only when computed from measured current-run
throughput and shall be labeled estimated. Unknown totals shall remain
indeterminate.

#### Scientific or usability rationale

Long field runs require actionable observability, but invented percentages or
ETA would mislead users and undermine trust.

#### Acceptance criteria

- Progress events are typed/structured and do not require worker code to mutate
  GUI widgets.
- Stage A reports backend progress only from parseable ffmpeg evidence.
- Stage B reports frames processed and expected when the expected trim count is
  known.
- Background generation reports decoded/sampled progress with an honest total
  only when known.
- Elapsed time is monotonic for a run.
- ETA appears only after sufficient measured throughput, updates from actual
  progress, is labeled estimated, and disappears when estimation is invalid.
- No stage emits a fabricated percentage.
- Callback absence preserves current service behavior.
- Tests cover event order, known and unknown totals, callback failure isolation,
  cancellation/failure terminal events, and ETA suppression.

#### Dependencies

- Stable stage definitions across the service and its subprocess/decoder work.
- Parseable ffmpeg progress configuration.
- Known expected frame counts from trim/probe data where available.
- GUI task-thread delivery back to the main thread.

## 6. Cross-cutting release constraints

- v2 shall not rename or add official artifacts as part of these requirements.
- A cache or GUI session file shall not be described as scientific output.
- Timing warnings and writer safety shall not change selected units or timing
  sources automatically.
- Progress reporting shall not change processing order or scientific outputs.
- Visual tools shall produce inputs for existing validated core behavior rather
  than duplicate scientific validation in the GUI.
- Any required metadata schema change must be separately reviewed for backward
  compatibility before implementation planning.
- An implementation plan shall be written only after this requirements draft is
  reviewed, revised, and approved.
