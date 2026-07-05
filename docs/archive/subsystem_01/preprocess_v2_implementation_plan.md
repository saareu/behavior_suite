# Preprocess Subsystem Implementation Plan v2

> **Archive notice:** This historical document is retained for traceability. It is not the current source of truth. See `docs/subsystem_01_preprocessing.md`, `docs/subsystem_01_status_and_roadmap.md`, and `docs/design/subsystem_01_geometry_modes.md`.

**Status:** Proposed implementation plan; requires review before implementation  
**Planning baseline:** `preprocess-v1-field-tested` (`9074a1d`)  
**Document date:** 2026-06-23

## Current closure note

This document remains useful milestone history and design context, but the compact current status, closure checks, and deferred roadmap for Subsystem 01 are now summarized in `docs/subsystem_01_status_and_roadmap.md`.

The normal user workflow remains:

```text
Choose video
→ inspect / trim / optionally pre-crop
→ choose Detect cage automatically or Manual ROI
→ review accepted geometry
→ optionally add static exclusion masks
→ prepare video
```

Future detector presets and scene characterization are optional developer/design concepts, not mandatory user-facing steps. Detector presets must be derived from representative frames plus an accepted automatic candidate or accepted manual ROI; they must not be recommended from FFmpeg/video probe facts alone.

The V2.6 geometry design is broader than visual pre-crop. It covers future final spatial geometry modes and the requirement that every prepared video retain one authoritative raw-to-prepared and prepared-to-raw transform description.

## 1. V2 scope and relationship to the v1 tag

Preprocess v2 is an incremental extension of the field-tested v1 subsystem. The
annotated tag `preprocess-v1-field-tested` is the immutable comparison baseline.
V2 work shall begin on a new branch from that tag or its descendant and shall
not move, rewrite, or amend the tag.

The v1 scientific and artifact contracts remain binding unless a separately
approved specification revision explicitly changes them. In particular:

- Raw sequential decode order remains the primary frame identity.
- The default mapping remains
  `raw_decode_frame_idx = start_frame + prepared_frame_idx`.
- External timing retains exact untrimmed readable-frame-count validation.
- No frame resampling, dropping, duplication, interpolation, or reordering is
  introduced.
- For the current V1/V2 perspective workflow, a validated, explicitly accepted
  `CropPlan` remains mandatory. V2.6 broadens the long-term design language so
  future final spatial geometry modes may use another explicit raw-to-prepared
  transform authority, but no current `CropPlan` behavior is changed by this
  plan.
- Pre-crop remains part of the final raw-to-prepared transform.
- The six official artifact names and roles remain unchanged.
- Core services remain authoritative; the GUI remains an orchestration and
  display layer.

This plan covers six candidate improvements: timing-unit plausibility warning,
visual trim navigation, visual pre-crop selection plus final spatial geometry
mode design, detector-default reset, better progress and cancellation, and an
optional reusable raw-probe cache.
None is present in the v1 tag merely because it is described here.

## 2. Explicit non-goals

The following are outside this plan:

- Temporal resampling or any change to frame-identity semantics.
- Relaxing external timing numeric, shape, finiteness, monotonicity, or exact
  length validation.
- Automatic timing-unit inference, repair, or selection changes.
- Blocking a run solely because external and raw FPS differ strongly.
- Silent FPS substitution or a fallback to another timing source.
- A new writer-profile FPS representability algorithm. Writer failure remains
  a separate technical hard failure when the requested FPS cannot be encoded.
- Changes to `CropPlan` meaning, point ordering, coordinate conventions, or
  acceptance rules.
- New preprocessing algorithms, crop-detection algorithms, codecs, containers,
  masks, batch processing, raw PTS extraction, or SLEAP inference.
- New or renamed official artifacts, or changes to the v1 metadata, sync, or
  settings schemas.
- Storing video frames, open capture handles, or full timing arrays in
  persistent GUI state.
- Treating a cache entry as valid because its file path matches.
- Fabricated progress percentages or ETA without measurable work and a usable
  denominator.

## 3. Prioritized roadmap

| Candidate | Priority | Rationale |
| --- | --- | --- |
| 1. Timing-unit plausibility warning | **P0 — scientific safety / workflow correctness** | The first field test demonstrated that a structurally valid but incorrectly declared timing unit can imply an extreme FPS. A prominent warning improves the user's scientific interpretation without changing data or blocking unusual experiments. |
| 2. Visual trim navigation | **P1 — major usability improvement** | Visual decode-frame selection removes error-prone transcription while preserving the existing numeric `[start, end)` contract. |
| 3. Visual pre-crop and spatial geometry modes | **P1 — major usability improvement / design correction** | Visual boundaries and rectangles make a permanent geometry decision reviewable. V2.6 also defines future final spatial geometry modes so `CropPlan` remains valid for perspective workflows without becoming the only possible geometry authority. |
| 4. Detector settings reset | **P2 — quality-of-life improvement** | Typed defaults already exist and current editing is valid. A clear reset improves reproducibility and recovery from experiments but does not address a current scientific failure. |
| 5. Better progress and cancellation UX | **P1 — major usability improvement** | Long counts and processing stages currently have safe worker ownership but limited observability and incomplete user-directed cancellation. Honest stage progress and cleanup materially improve long-run operation. |
| 6. Reusable raw probe cache | **P1 — major usability improvement** | Full sequential counts are expensive and repeated across restarts. Safe reuse is valuable, but conservative fingerprinting and provenance are required so usability never overrides source identity. |

P0 work is completed first. P1 milestones then establish execution observability
and source identity before adding visual selection. The independent P2 reset is
kept narrow and follows the larger workflow changes.

## 4. Milestone breakdown

### Milestone V2.1 — Timing plausibility assessment and warning (P0)

Add a pure typed assessment that consumes the existing validated timing result
and raw probe FPS. For finite positive values, calculate:

```text
symmetric_mismatch_factor = max(external_fps / raw_fps,
                                raw_fps / external_fps)
```

The initial warning threshold is `>= 2.0`. The warning shall show declared
units, raw median interval, converted interval when temporal, external FPS, raw
nominal FPS and provenance, and the mismatch factor. It shall state that
incorrect units may prevent video encoding.

The assessment shall be unavailable, not guessed, for `frames`, `unknown`, a
missing raw FPS, or invalid/non-finite inputs. Recompute it when the selected
variable, units, or raw probe changes. The existing timing validator remains the
only hard timing gate.

Acceptance conditions:

- The field wrong-unit case near 119403 FPS versus 119.1 FPS warns.
- The corrected case near 119.403 FPS does not warn.
- Exactly 2.0 warns; a value below 2.0 does not.
- The comparison is symmetric.
- Warning presence never changes units, arrays, validation state, navigation,
  FPS selection, or frame mapping.
- The user is not required to confirm or justify an unusual FPS.

### Milestone V2.2 — Progress protocol and short-operation cancellation (P1)

Introduce a core-owned, UI-independent progress event contract and connect it
to the existing Qt-safe task runner. First instrument:

- Raw sequential readable-frame counting.
- Automatic cage-detection frame sampling and computation stages.

Progress callbacks execute in worker code but reach widgets only through queued
Qt signals. Events are throttled by frame count or elapsed time to avoid making
processing slower. Existing callers remain valid when no observer is supplied.

Add visible Cancel actions for active count and cage-detection tasks. A cancelled
count stores no count or cache record. A cancelled detection creates no
candidate and cannot disturb an already accepted crop belonging to another
input generation.

### Milestone V2.3 — Pipeline progress, cancellation, and cleanup (P1)

Extend the same protocol through `PreprocessService.run()` and these stages:

- Stage A ffmpeg preparation.
- Stage B OpenCV re-encode.
- Prepared-video validation.
- Background generation.

Execution-only hooks shall be supplied separately from `PreprocessRequest`; a
callable, cancellation token, or Qt object shall not become scientific request
data. The service API should remain backward compatible, for example through an
optional typed execution context.

Stage A shall use ffmpeg's machine-readable progress stream and a managed
`Popen` lifecycle. Cancellation first requests graceful termination and then
uses a bounded forced termination only if needed. Per-frame OpenCV loops check
cancellation between reads/writes.

Cancellation is a distinct terminal state, not success and not an unexplained
failure. New partial official outputs and temporary files are removed by
default, pre-existing official artifacts are preserved, and the processing log
records cancellation. With debug enabled, an incomplete internal may remain
only below `.internal`, clearly non-official and never validated or advertised
as usable.

### Milestone V2.4 — Optional fingerprinted raw-probe cache (P1)

Add a non-official cache below the per-user application cache directory resolved
with `platformdirs`. The cache is optional; disabling it always falls back to a
fresh measurement. It is not stored in or enumerated with project artifacts.

Each cache entry shall be based on a typed source fingerprint containing at
minimum:

```text
absolute resolved path
file size in bytes
last-modified time with the highest available precision
decoded video width and height
```

A bounded fast content fingerprint over stable source regions is recommended by
default. If recorded, it must match. The entry also records cache schema,
completed sequential readable count, measurement time, and decoder/application
identity sufficient to conservatively invalidate incompatible records.

Compute the fingerprint before and after a full count; store the entry only if
the count completed successfully and the fingerprint did not change. Cache
writes use same-directory temporary files and atomic replacement. Corrupt,
partial, incompatible, or unverifiable entries are ignored with a warning.

The workflow shall distinguish these states:

1. `current_session_verified` — a full count completed for the current source
   during this process.
2. `prior_completed_run` — authoritative metadata records a count from a prior
   successful run, but it is not automatically a cache hit.
3. `validated_reusable_cache` — the current fingerprint matches a completed
   cache entry under the active validation policy.
4. `not_measured` — no currently trusted count is available.

Trust order is current-session measurement, then validated cache. A prior-run
count remains useful provenance, but shall satisfy a current exact-count
precondition only when supported by a matching validated cache entry or a new
measurement. This avoids treating the v1 metadata path alone as source identity.

Both GUI and service/CLI probing must use the same cache API so one caller does
not silently recount after another accepted a validated entry. Cancellation or
failure never creates a reusable record. A source change during lookup or use
invalidates the result.

### Milestone V2.5 — Raw decode-frame navigator and visual trim (P1)

Add a reusable raw-frame navigation controller and viewer to the existing Trim
and Pre-Crop page. It shall support:

```text
First
Previous
Next
Jump to frame
Jump backward/forward by configurable step
Set Start from Current
Set End Exclusive from Current
```

The page always displays the current **raw decode-frame index**. Setting the end
from the displayed frame means that displayed frame is excluded. The selected
range is rendered as `[start_frame, end_frame_exclusive)` and shows its frame
count when both bounds are known.

Initial navigation shall prefer correctness over random-seek speed:

- Forward navigation decodes sequentially and counts successful reads.
- Backward navigation or a backward jump reopens the source and decodes from
  frame zero to the requested index.
- A backend random seek may be added later only after it is verified against
  sequential decode identity on supported backends; it is not assumed exact in
  this milestone.
- EOF and unreadable targets produce clear errors and do not silently select a
  different index.

Frame decoding runs in the task infrastructure with stale-result protection.
Only the current index and navigation status enter shared GUI state. The current
frame and capture handle are page/controller-owned ephemeral resources and are
released on source/project change, cancellation, or shutdown.

Changing the displayed frame alone changes no scientific input. The existing
typed trim path and invalidation rules run only when Start or End Exclusive is
set or edited.

### Milestone V2.6 — Visual pre-crop and final spatial geometry modes (P1)

The detailed design is recorded in
`docs/preprocess_v2_6_geometry_modes_design.md`. The current perspective
`CropPlan` workflow remains valid:

```text
raw video
→ optional pre-crop
→ automatic cage detection or manual four corners
→ perspective CropPlan
→ rectification / rotation / canonical scale-pad
→ prepared video
```

V2.6 corrects the long-term geometry model: `CropPlan` is the authority for the
current perspective mode, but every prepared video should ultimately retain one
authoritative transform description sufficient to map prepared coordinates back
to raw-video coordinates without hidden GUI state. Future final spatial modes
may include:

- `identity`
- `axis_aligned_pre_crop_only`
- `perspective_crop_plan`
- `composed_geometry`

The first V2.6 implementation should remain narrow. It should extend the V2.5
raw-frame viewer with a dedicated pre-crop overlay supporting:

- Vertical boundary with Keep Left and Keep Right.
- Horizontal boundary with Keep Upper and Keep Lower.
- Dragged manual rectangle.
- Clear/Reset to no pre-crop.
- A visible retained-region overlay and numeric ROI summary.

Overlay interactions produce only candidate numeric inputs. They are converted
to a typed `PreCropConfig` and passed through the existing
`resolve_pre_crop()` path. The GUI shall not reproduce its validation or
raw-to-pre-crop transform logic.

Boundary lines represent half-open image edges: Keep Left retains `[0, x)`,
Keep Right retains `[x, width)`, Keep Upper retains `[0, y)`, and Keep Lower
retains `[y, height)`. A manual rectangle is stored as integer
`(x, y, width, height)` in raw-frame coordinates. Mapping tests must exclude Qt
letterboxing from image coordinates and cover the right/bottom exclusive edge.

Numeric controls and overlay stay synchronized in both directions. Clear/Reset
creates the existing disabled/`none` config and full-frame resolved ROI.
Changing only `start_frame` or `end_frame_exclusive` must not invalidate
selected spatial geometry for an unchanged raw video. Effective spatial
geometry changes still invalidate accepted geometry review.

The first implementation must not introduce a new `GeometryPlan`, change
current `CropPlan` behavior, alter metadata or sync schemas, or implement
identity, pre-crop-only, or composed-geometry processing.

### Milestone V2.7 — Detector settings reset to defaults (P2)

Add `Reset detector settings to defaults` to Crop Review. Reset these fields to
their current typed model defaults:

```text
CageDetectConfig.sample_step
CageDetectConfig.pad_px
CageDetectConfig.threshold
CageDetectConfig.pre_crop_expansion_percent
CageDetectConfig.dilate_kernel_size
CageDetectConfig.erode_kernel_size
CageDetectConfig.rim_close_kernel_size
CageDetectConfig.minimum_cage_width_fraction
CageDetectConfig.minimum_cage_height_fraction
CageDetectConfig.minimum_contour_area
CageDetectConfig.fit_tolerance_px
PrepareConfig.roi_margin_px
PrepareConfig.perspective_interpolation
```

Canonical enabled/width/height are output-geometry settings and are explicitly
not reset by this action; Encode Settings already owns their loaded-config
reset. The page should visually separate these controls from detector defaults.

Build a new validated `PreprocessConfig` copy; never mutate a raw dictionary or
write the loaded YAML. If any reset field changes, clear both candidate and
accepted `CropPlan`, clear the prior run result, return to automatic mode, and
require detection and acceptance again. A no-op reset preserves the current
candidate/acceptance.

Display a text-backed `Modified from defaults` indicator at group level and an
accessible marker or tooltip on changed fields. Do not communicate the state by
color alone. Enable Reset only when at least one reset-owned field differs from
typed defaults.

### Milestone V2.8 — Cross-feature acceptance and release hardening

Run all automated suites and the real-data protocol in Section 10. Confirm that
v1 behavior is unchanged when every v2 option is unused. Record dependency and
backend versions, cache policy, progress capabilities, and cancellation results.
Do not tag v2 while any P0 regression or artifact/frame-identity discrepancy is
open.

## 5. Dependencies between milestones

```text
V2.1 Timing warning ───────────────────────────────────────────┐
                                                               │
V2.2 Progress protocol + count/detection cancellation ─┐       │
                                                       ├─ V2.3 Pipeline progress/cancellation
                                                       │
                                                       └─ V2.4 Probe cache

V2.2 task/progress foundation ── V2.5 Visual trim ── V2.6 Visual pre-crop + geometry modes

V2.7 Detector reset (independent after v2 state conventions stabilize)

All milestones ───────────────────────────────────────── V2.8 Acceptance
```

V2.1 has no cache or visual-navigation dependency and is the recommended first
implementation milestone. V2.4 follows cancellation instrumentation so failed
or cancelled counts cannot be persisted. V2.6 reuses V2.5's viewer and exact
coordinate mapping, while documenting future non-perspective geometry modes.
V2.7 is technically independent but follows shared GUI state changes to avoid
rework.

## 6. Core-data-model changes

Add only in-memory or non-official typed models; do not revise official artifact
schemas.

### Timing assessment

Introduce a frozen `TimingPlausibilityAssessment` with:

```text
declared_units
raw_median_interval
converted_interval_sec
external_fps
raw_fps
raw_fps_source
symmetric_mismatch_factor
warning_triggered
availability_reason
```

The helper belongs in core timing code because unit conversion and comparison
must not be reimplemented by Qt widgets.

### Progress and cancellation

Introduce typed `ProgressEvent`, `ProgressStage`, `ProgressUnit`, and
`ProgressTotalKind` models. An event should contain stage, activity message,
completed units, optional total units, unit, whether the denominator is exact or
estimated, and monotonic elapsed time. Percentage is derived only when a total
exists and is explicitly labeled exact or estimated.

Use a lightweight execution context containing an optional progress callback
and cancellation predicate. Keep this context outside `PreprocessRequest`.
Introduce a domain-level cancellation exception and an in-memory cancelled
result indicator so callers can distinguish cancellation from failure without
claiming success.

### Probe evidence and cache

Introduce frozen typed models for `RawSourceFingerprint`,
`RawReadableCountEvidence`, `RawReadableCountProvenance`, and a versioned
`RawProbeCacheEntry`. The cache store validates entries through these models.
Count provenance may be exposed through `VideoProbeResult` or a companion probe
outcome, but must not be serialized into official artifacts unless a separate
schema change is approved.

### Frame navigation

Use an ephemeral `RawDecodeFrame` result containing the requested/decoded index,
source identity, and frame array. It must not be retained in
`PreprocessSetupState`. No change to `CropPlan`, `ResolvedPreCrop`, or
`PreprocessRequest` scientific fields is required.

## 7. GUI-state changes

Extend lightweight GUI state with only compact values such as:

```text
timing_plausibility_assessment
raw_count_evidence / raw_count_provenance
current_raw_decode_frame_index
frame_navigation_status
frame_navigation_step
latest_progress_event
progress_cancel_requested
detector_fields_modified_from_defaults
```

Do not store frames, capture handles, subprocesses, cache contents, or large
arrays in persistent GUI state. Page/controller objects own those resources.

Required invalidation behavior:

- A timing warning is display-only and does not invalidate timing acceptance.
- Changing timing selection still clears the prior run result but not crop
  acceptance.
- Navigating to another display frame changes no workflow input.
- Changing only trim changes which frames are processed and clears prior run
  output, but must not invalidate accepted spatial geometry for an unchanged
  raw video.
- Changing effective spatial geometry, including pre-crop geometry, manual
  four-corner points, automatic detection output, geometry-affecting output
  settings, or raw video identity, clears accepted geometry and prior run
  output.
- Loading a count from validated cache invalidates a previously validated MAT
  selection if its exact-count dependency changes.
- Detector reset invalidates crop/run state only when reset changes an effective
  field.
- Progress from stale task generations is discarded by the existing task
  coordinator.

## 8. Core-engine changes

### Timing

Add one pure assessment helper beside the existing timing converter/validator.
It may use `ExternalTimeSelection.estimated_fps` after validation and
`VideoProbeResult.raw_fps_effective` plus provenance. It shall not select FPS or
modify validation outcomes.

### Raw frame access

Factor exact display-frame decoding out of page code into a read-only core
helper/session. Sequential decoding defines the index. The helper performs no
trim, crop, resampling, or output writing.

### Progress/cancellation hooks

Add optional hooks to the existing functions rather than copying algorithms:

```text
count_opencv_readable_frames
detect_cage_crop_plan and frame sampling
run_ffmpeg_prepare
reencode_intermediate_with_opencv
validate_prepared_video
estimate_prepared_background
PreprocessService.run
```

Callbacks must be optional, bounded in frequency, and unable to mutate core
state. Callback failures are isolated and logged or surfaced without silently
changing scientific output. Cancellation checks occur at safe loop/subprocess
boundaries.

### Progress capability table

| Operation | Progress source | Total | Percent/ETA policy | Cancellation and partial state |
| --- | --- | --- | --- | --- |
| Raw sequential count | Successful sequential `capture.read()` calls | Unknown on first measurement; a prior value is reference evidence, not an authoritative denominator | Show frames counted. No exact percent or ETA on first count. A diagnostic recount may show progress against prior evidence, explicitly non-authoritative. | Check before each read. Store no partial count/cache entry. |
| Automatic cage detection | Sample positions attempted and major compute stages | Exact planned sample positions only when a verified readable count is supplied; otherwise unknown | Exact percent only with a verified denominator. ETA only after sufficient measured sample throughput; hide during unbounded compute phases. | Check between samples and compute phases. No candidate is applied on cancellation. |
| Stage A ffmpeg | Parsed `-progress` frame/out-time events | Exact selected-frame count when trim end or verified raw count resolves it; otherwise unknown | Exact frame percent only with exact total. ETA is estimated from current-run speed after warm-up and hidden on stalls, missing totals, or non-monotonic progress. | Terminate managed subprocess, remove partial intermediate by default. |
| Stage B OpenCV | Frames decoded and written | Exact when Stage A/trim yields an expected count; otherwise unknown until completion | Exact percent only with exact total. ETA uses a stable current-run throughput window and is hidden when total is unknown. | Check each frame. Remove temporary output; preserve pre-existing official output. |
| Prepared validation | Frames sequentially decoded | Exact expected prepared count from Stage B | Exact percent. ETA may be estimated after a minimum sample window. | Check each frame. Cancellation never marks the video valid; service cleanup applies. |
| Background generation | Prepared frames decoded and samples accepted | Exact planned decode extent from validated prepared count, sample step, and sample limit | Exact decode percent; ETA only from measured decode throughput. Median calculation is an indeterminate activity stage. | Check each read and before median calculation. No PNG is committed on cancellation. |

### Cache store

Implement cache lookup/write in a dedicated preprocess module using
`platformdirs`, Pydantic validation, atomic replacement, and conservative
miss-on-error behavior. Probe callers inject or opt into the cache; scientific
functions must remain usable with no cache. Recompute and compare the source
fingerprint before trusting an entry. Never use path-only lookup as a validity
decision.

No official writer, artifact enumerator, metadata validator, or sync writer is
changed by the cache.

## 9. Test strategy

All existing v1 tests remain mandatory. New tests shall avoid real laboratory
data and active displays unless explicitly in the acceptance tier.

### Unit and headless controller tests

- Timing assessment: all temporal units, `frames`, `unknown`, missing FPS,
  non-finite inputs, symmetric ratios, 2.0 boundary, field wrong/correct unit
  cases, and non-blocking state behavior.
- Progress models: exact/estimated/unknown totals, monotonic elapsed time,
  throttling, callback failure isolation, and no percentage without a total.
- Cancellation: each per-frame loop stops cooperatively and reports cancelled,
  never success or a partial count.
- Fingerprints/cache: resolved paths, size/mtime/dimension/content changes,
  decoder/schema mismatch, corrupt records, atomic-write failure, source change
  during count, cancellation, and provenance precedence.
- Frame navigation: first/previous/next, forward/backward step, jump, EOF,
  unreadable target, stale result rejection, and unique synthetic per-frame
  markers proving decode-index identity.
- Trim: inclusive start, exclusive end, empty/off-by-one ranges, open-ended end,
  and no invalidation from display-only navigation.
- Pre-crop overlay: all directional modes, rectangle drag directions,
  half-open edges, Qt letterboxing exclusion, numeric/visual synchronization,
  clear/reset, and existing core-resolution errors.
- Detector reset: exact reset-owned fields, canonical exclusion, typed defaults,
  config-dirty behavior, no disk write, no-op preservation, changed-value crop
  invalidation, and accessible modified indicators.

### Integration tests

- Parse deterministic fake ffmpeg progress before using a real ffmpeg fixture.
- Cancel Stage A and prove the subprocess terminates and partial internal output
  is handled by policy.
- Cancel Stage B, validation, and background loops and prove temporary/new
  outputs are removed while pre-existing official artifacts survive.
- Complete an observed service run and verify progress event order without
  changing any output, frame count, geometry, or sync mapping.
- Persist a completed raw count, create a fresh application/controller instance,
  reuse it only on a full fingerprint match, and force a new count for every
  mismatch class.
- Select trim and pre-crop visually on a synthetic indexed video, run the real
  service, and verify `prepared_sync.npz`, metadata ROI, prepared dimensions,
  and visible bounds.

### GUI tests

Use headless Qt tests/fakes for signals, button enablement, warning text,
progress rendering, Cancel behavior, modified-default indicators, and overlay
mapping. Workers and platform operations are mocked where possible. Widgets
must never be mutated from worker threads.

## 10. Real-data acceptance criteria

Run acceptance on controlled copies outside the repository and retain the v1
automatic/external-timing and manual/no-external-timing cases as comparisons.

1. With all v2 options unused, both v1 workflows still create the same six
   official artifacts and pass strict validation with unchanged frame mapping,
   crop geometry, and artifact schemas.
2. The known wrong timing unit displays the prominent non-blocking warning and
   the corrected unit removes it. Continuing with the warning does not alter
   units or mapping; any unsupported writer FPS fails normally and visibly.
3. Visual navigation displays verified raw decode frames at first, interior,
   trim-start, trim-end-exclusive, and last valid indices. Prepared sync maps
   the selected range exactly.
4. In the first perspective-mode implementation, each visual pre-crop mode
   produces the expected retained raw region, accepted `CropPlan`, final video
   geometry, and metadata transform.
5. Detector reset restores the documented fields, leaves canonical settings
   unchanged, and requires re-detection only after an effective change.
6. Progress is observed for every listed long stage. Percent and ETA appear
   only under the documented denominator rules.
7. Cancellation is exercised during count, detection, Stage A, Stage B,
   validation, and background generation. No cancelled operation is reported
   successful and no partial output is treated as scientifically usable.
8. An unchanged source reuses a validated count after application restart. Path,
   size, mtime, dimensions, content fingerprint, or decoder-policy changes each
   cause a safe miss and new measurement. Prior-run metadata without matching
   fingerprint evidence is visibly distinct from a validated cache hit.
9. Record OS, Python, OpenCV/backend, ffmpeg/ffprobe, SciPy, h5py, configuration,
   source fingerprints, event summaries, runtimes, warnings, and final artifact
   validation.

## 11. Risks and scientific safety constraints

| Risk | Required control |
| --- | --- |
| Plausibility warning is mistaken for validation | Label it warning-only; retain existing validation and navigation rules; never auto-change units. |
| Warning threshold creates false positives for unusual experiments | Keep it non-blocking, show both values/provenance, and test boundary behavior. |
| Random access displays the wrong raw frame | Initial implementation uses sequential decode identity; random seek is prohibited until backend parity is demonstrated. |
| Overlay/display coordinates drift because of letterboxing or pixel-edge semantics | Use shared mapping helpers, half-open edge tests, and core-resolved ROI display. |
| GUI duplicates pre-crop or timing science | GUI submits typed values to existing core converters/validators/resolvers only. |
| Progress events slow processing or arrive on the wrong thread | Throttle events, use queued signals, benchmark overhead, and prohibit worker widget mutation. |
| ETA misleads users | Require measurable throughput and an exact/declared denominator; hide ETA during unknown or unstable work. |
| Cancellation corrupts or replaces valid artifacts | Continue temporary/atomic output patterns, preserve pre-existing artifacts, and test cancellation at every stage. |
| Cancelled count becomes trusted | Persist only successful full counts with unchanged before/after fingerprints. |
| Source file changes while cached | Compare a multi-field fingerprint at lookup and relevant execution boundaries; default to cache miss. |
| Network timestamps are coarse or preserved during replacement | Prefer nanosecond metadata plus bounded content fingerprint and decoder identity; offer cache disablement. |
| Cache becomes an unofficial source of scientific truth | Treat it as operational evidence only; fresh measurement remains available; never enumerate it as an official artifact. |
| Typed defaults drift across versions | Derive reset values from current Pydantic defaults and cover them with schema-version-specific tests. |

Any implementation proposal that requires resampling, relaxed timing length,
changed `CropPlan` meaning, changed artifact names, or GUI-owned scientific logic
must stop for specification review rather than extending this plan implicitly.

## 12. Suggested implementation order

1. Branch from the tagged v1 baseline and keep the tag untouched.
2. Implement V2.1 timing plausibility assessment and warning.
3. Implement V2.2 progress contracts and count/detection cancellation.
4. Implement V2.3 service/pipeline progress, cancellation, and cleanup.
5. Implement V2.4 fingerprinted optional cache using the completed-count and
   cancellation contracts.
6. Implement V2.5 exact raw-frame navigation and visual trim.
7. Implement V2.6 visual pre-crop and geometry-mode design on the same
   viewer/mapping layer, keeping current perspective `CropPlan` processing
   unchanged.
8. Implement V2.7 detector-default reset.
9. Complete V2.8 automated and real-data acceptance before considering a v2
   release tag.

V2.1 is the recommended first implementation milestone: it addresses the only
candidate classified P0, reuses existing validated timing/probe values, has a
small testable surface, and cannot change scientific output when implemented as
specified.

## 13. Proposed Git commit boundaries

Keep commits reviewable and avoid mixing scientific core, GUI wiring, cache,
and acceptance evidence.

1. `Add typed timing plausibility assessment`
   - Core model/helper and unit tests only.
2. `Show non-blocking timing FPS mismatch warning`
   - Timing controller/page/state and headless GUI tests.
3. `Add typed preprocessing progress events`
   - Progress/cancellation contracts and task-runner signal bridge.
4. `Report and cancel raw count and cage detection`
   - Count/detection hooks, GUI Cancel actions, cleanup and tests.
5. `Report Stage A preprocessing progress and cancellation`
   - Managed ffmpeg progress parsing/termination and integration tests.
6. `Report and cancel OpenCV validation and background stages`
   - Stage B/validation/background hooks, service result state, cleanup tests.
7. `Add versioned raw source fingerprint and probe cache`
   - Cache models/store, atomic persistence, corruption/invalidation tests.
8. `Reuse validated raw counts across callers and restarts`
   - Probe/service/CLI/GUI integration and provenance UI tests.
9. `Add exact raw decode-frame navigation controller`
   - Read-only decoder/session and headless navigation tests.
10. `Add visual trim navigation workflow`
    - Viewer controls, task integration, `[start, end)` UI tests.
11. `Add visual pre-crop overlays`
    - Boundary/rectangle interaction, typed core delegation, mapping tests, and
      geometry-mode design documentation. Do not add non-`CropPlan` processing
      modes in this commit.
12. `Add detector settings default reset`
    - Controller/page changes and exact invalidation/default tests.
13. `Document v2 acceptance evidence`
    - Acceptance records only after automated and real-data gates pass.

Each commit must pass its targeted tests, Ruff, and `git diff --check`. Milestone
completion requires the full suite. No commit may modify or recreate the
`preprocess-v1-field-tested` tag.
