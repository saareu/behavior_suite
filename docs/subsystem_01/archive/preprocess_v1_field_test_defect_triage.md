# Preprocess v1 Field-Test Defect Triage

> **Archive notice:** This historical document is retained for traceability. It is not the current source of truth. See `docs/subsystem_01/preprocessing.md`, `docs/subsystem_01/status_and_roadmap.md`, and `docs/subsystem_01/design/geometry_modes.md`.

## 1. Scope and evidence

This audit covers four field-test findings against the current preprocessing
implementation:

1. a manual four-corner `CropPlan` rejected during Stage A geometry extraction;
2. an apparently repeated full sequential raw-frame count on the Timing page;
3. possible prepared-preview versus encoded-output geometry mismatch; and
4. design of a timing-unit/FPS mismatch warning.

The conclusions are based on the current GUI state, page, controller, worker,
crop geometry, ffmpeg preparation, service, model, and exception paths, plus the
existing unit and integration tests named in the field-test audit request. The
approved v1 specification, v1 implementation plan, AI coding guide, README,
current field-test audit, and v2 requirements draft were also reviewed. Git
status was inspected without cleaning, staging, restoring, or changing existing
workspace material.

This is a source audit, not an implementation change. No production code, test,
configuration, GUI behavior, approved v1 specification, or artifact contract is
changed here.

## 2. Confirmed field-test facts

- A manual-crop run reached Stage A filtergraph construction and failed with
  `CropPlan pre-rotation quadrilateral is not an axis-aligned rectangle.`
- Manual crop is specified and implemented as an arbitrary valid TL, TR, BR, BL
  quadrilateral, not as an axis-aligned raw-frame rectangle.
- A full sequential count completed on Raw Video can be retained in the shared
  in-memory `raw_probe` and is consumed by Timing validation.
- The Timing page nevertheless continues to display an enabled `Count All
  Readable Raw Frames Now` button after such a count. The controller also allows
  that redundant operation to be dispatched.
- The GUI preview applies the composed candidate `CropPlan` homography directly.
  Stage A instead decomposes that homography into ffmpeg crop, perspective,
  scale, padding, rotation, and canonical stages.
- The dark area outside the image in the Qt viewer is display letterboxing. It
  is not part of the preview array. Whether the reported difference in visible
  tray/floor content occurs inside the image remains unconfirmed.
- Timing candidates expose an unconverted median difference and raw
  `1 / median_difference`. After declared units are validated, the core timing
  result and converted vector provide enough information to show converted
  interval and FPS. Raw nominal FPS and its provenance are already in the video
  probe.

## 3. Root-cause analysis for each issue

### A. Manual crop Stage A failure

#### Exact enforcement point

`preprocess.video_prepare._extract_rectification_geometry()` removes canonical
scaling/padding and rotation from `CropPlan.H_raw_to_prepared_3x3`, transforms
the four stored raw crop points with the recovered rectification homography,
and builds an ideal rectangle from averaged left, right, top, and bottom values.
It then requires:

```python
np.allclose(
    destination,
    expected_destination,
    rtol=0.0,
    atol=_GEOMETRY_ATOL,
)
```

where `_GEOMETRY_ATOL` is `1e-5`. Failure raises the reported error. The next
checks require each recovered rectangle coordinate to be an integer within the
same fixed absolute tolerance.

This does **not** require the user's raw quadrilateral itself to be axis-aligned.
It requires the raw quadrilateral, after the stored rectification homography is
recovered from the composed plan, to land on an axis-aligned integer-pixel
destination with extremely small absolute error.

#### Why manual plans can fail while automatic plans proceed

`manual_crop.make_manual_crop_plan()` preserves the validated GUI click points
as float64 values in `quad_raw_tl_tr_br_bl`. Its rectification helper, however,
casts local click points to float32 before calling
`cv2.getPerspectiveTransform()`, and uses a float32 destination. The resulting
homography was therefore fitted to float32-quantized points, while Stage A later
tests it against the original float64 click points. Subpixel manual clicks,
especially at larger coordinates or with a nonzero pre-crop origin, can produce
a residual greater than `1e-5` even though the quadrilateral and homography are
scientifically valid.

Automatic geometry originates in `cv2.minAreaRect()` and `cv2.boxPoints()`.
Those box coordinates are already float32-derived before being stored as
float64, and the automatic rectification helper fits its homography to the same
float32 coordinates. It therefore usually avoids the manual stored-point versus
fitted-point precision split. This is an implementation-path distinction, not a
valid scientific reason to accept automatic geometry and reject equivalent
manual geometry.

#### CropPlan contract

The rejection contradicts the effective v1 contract. `CropPlan` accepts a
finite, convex, nondegenerate quadrilateral in TL, TR, BR, BL order, validates
the forward/inverse homographies, even prepared dimensions, and canonical
geometry consistency. It does not require raw points to be integer-valued or
independently require their recovered destination to satisfy a fixed `1e-5`
axis/integer test. Manual crop documentation explicitly preserves arbitrary raw
four-corner points without clamping or reordering.

#### Is Stage A unnecessarily reconstructing geometry?

Partly. Stage A cannot pass the OpenCV homography directly to ffmpeg's
`perspective` filter because that filter is expressed with source corners and a
different W/H boundary convention. Decomposition and coordinate adaptation are
therefore necessary in the current ffmpeg pipeline. In particular,
`_build_perspective_source_quad()` intentionally converts CropPlan pixel-center
geometry to ffmpeg boundary coordinates.

The brittle part is reconstructing intermediate rectangle coordinates from the
composed homography and then imposing stricter fixed-absolute precision and
integer conditions than the authoritative `CropPlan` contract. Stage A is
consuming CropPlan geometry, but it also adds a second, numerically inconsistent
acceptance gate.

#### Classification

**v1 blocking defect.** An explicitly accepted, valid manual arbitrary
quadrilateral can be prevented from entering Stage A.

### B. Same-session sequential readable-frame count reuse

#### State and controller flow

- `PreprocessSetupState.raw_probe` is shared by all wizard pages.
- A completed Raw Video probe is assigned directly to `state.raw_probe` by
  `PreprocessSetupController.apply_raw_video_probe()`.
- Its `frame_count_opencv_readable` remains inside that probe.
- The subsequent timing reset clears timing selection fields only; it does not
  clear or replace `raw_probe`.
- The wizard constructs one shared setup controller and one persistent set of
  pages, so normal navigation does not replace the state.
- `TimingController.candidate_summaries()` uses the stored readable count for
  length-match display.
- `TimingController.validate_selected_timing()` checks the same stored count and
  does not request another probe when it is positive.

The data path is therefore working in-memory. The page/action path is not.
`TimingPage._refresh_raw_count()` updates only a label. It does not disable or
hide the count button when the current probe already contains a positive full
count. `_set_busy(False)` enables the button unconditionally, and
`TimingController.prepare_full_raw_frame_count()` checks only that a raw path is
selected. If the user follows the still-active prompt, the count is repeated.

#### Classification

Reuse is **partially working**. Stored count reuse for Timing summaries and
validation works; the current GUI workflow misleadingly offers and permits a
redundant recount. This is a **v1 usability defect**, not merely a lack of
cross-session persistence.

### C. Preview/output parity for automatic and manual CropPlans

#### Preview path

`ui.controllers.crop_review_controller.build_crop_preview()` calls
`cv2.warpPerspective()` once with the candidate plan's
`H_raw_to_prepared_3x3`, `prepared_size_wh`, configured interpolation, and a
constant black border. Because the homography is composed by the automatic and
manual plan builders, this single operation includes:

- pre-crop translation;
- perspective rectification into native geometry;
- any required 90-degree clockwise rotation;
- canonical uniform scaling;
- canonical padding and its left/top offsets.

There is no additional crop margin, expansion, scale, or pad in the preview
builder. It does not separately model ffmpeg W/H boundary coordinates or apply
Stage A's pixel-center correction; it relies on OpenCV's direct homography
sampling semantics.

`VideoFrameView` fits the completed preview array inside the widget while
preserving aspect ratio and paints the unused widget area dark gray. Those
viewer margins can look like black padding, but they do not change preview
pixels or `prepared_size_wh`.

#### Stage A path

Stage A uses the same accepted CropPlan but follows a different rendering path:
frame-index select, pre-crop, ffmpeg perspective source corners, optional
rectified-content scale, rectification padding, clockwise rotation, even-size
guard, canonical uniform scale, and canonical padding. The perspective corners
are derived through a specific pixel-center-to-ffmpeg-boundary conversion.

Thus the two paths are intended to represent the same geometry, but they do not
execute the same sequence or sampler. Small interpolation/codec differences are
expected; a difference in visible crop bounds, rotation, or padding is not.

#### Existing test coverage and conclusion

The current GUI unit test proves only that preview construction passes the
plan's exact homography and output size to OpenCV. The Stage A spatial integration
test checks marker locations, rotation/canonical geometry, and padding for an
arbitrary integer-coordinate manual plan. No test generates the GUI preview and
Stage A frame from the same source frame and accepted plan. There is no such
automatic-plan parity test either.

The reported visible-content mismatch is therefore **still unconfirmed from
source inspection**. Source inspection neither proves it nor disproves it. The
different transform/sampling paths make a direct regression comparison
mandatory. Classification: **unconfirmed hypothesis**.

### D. Timing-unit mismatch warning design

#### Values already available

The Timing page/controller already receive:

- selected `TimingUnit` in `state.selected_timing_units`;
- candidate raw `median_difference` and raw `estimated_fps` in
  `MatVectorCandidate`/`TimingCandidateSummary`;
- validated `ExternalTimeSelection`, including the declared units, raw median
  difference, and external FPS after unit interpretation;
- `external_time_vector_seconds` for seconds-convertible units;
- raw nominal FPS in `raw_probe.raw_fps_effective`, with
  `raw_fps_effective_method` as provenance and `opencv_fps` as available probe
  data.

For temporal units, converted median seconds can be obtained through the
existing core validator/converter path; external FPS is already calculated by
that path. A symmetric mismatch factor is:

```text
max(external_fps / raw_fps, raw_fps / external_fps)
```

for finite positive values. The GUI should not independently reinterpret or
validate timing arrays.

#### Minimal later v2 change

Add a small typed timing-preview result and headless controller method that runs
the selected candidate and units through the existing vector getter,
`validate_external_timing_vector()`, and conversion helper without committing or
changing the user's selection. Render the returned raw interval, declared unit,
converted seconds interval, external FPS, raw FPS/provenance, and mismatch
factor on the Timing page. Recompute after candidate or unit changes. For
`frames` or `unknown`, state that seconds/FPS conversion is unavailable. If raw
nominal FPS is absent, show the converted timing values and state that comparison
is unavailable.

Recommended prominent-warning trigger: both FPS values are finite and positive
and the symmetric mismatch factor is at least `2.0`. This conservative threshold
catches the field case near 1000x without making ordinary nominal/probe rounding
a warning. The threshold should be a named, tested GUI-policy constant, not a
scientific correction rule.

Recommended text:

> Warning: External timing implies {external_fps} FPS after converting
> {selected_units}, while the raw video reports {raw_fps} FPS
> ({raw_fps_source}); mismatch {factor}x. Verify the selected timing units.
> Incorrect units may prevent video encoding. Continuing will not change units
> or frame mapping.

This warning must never auto-change units, alter frame mapping, or block the user
solely because the FPS is unusual. A later writer-profile preflight may still
hard-fail before Stage A when the selected FPS is technically unsupported by the
actual writer. Classification: **v2 enhancement**.

The current v2 draft's requirement for explicit unusual-mismatch confirmation
and recorded rationale is superseded by the latest field-test decision: warning
only. That draft should be corrected in a separately authorized requirements
revision; this triage does not edit or create a v2 implementation plan.

## 4. Classification of each issue

| Issue | Classification | Current status |
| --- | --- | --- |
| A. Manual crop rejected by Stage A decomposition | v1 blocking defect | Confirmed root cause |
| B. Same-session readable-count reuse UX | v1 usability defect | Partially working |
| C. Preview/final visible-content parity | unconfirmed hypothesis | Direct parity test missing |
| D. Timing-unit/FPS mismatch warning | v2 enhancement | Values/helpers exist; UI absent |

## 5. Current source-code behavior

- Manual and automatic crop builders both produce the shared typed `CropPlan`,
  but manual plans can retain higher-precision raw points than the points used
  to fit their homography. Stage A then applies a stricter recovered-rectangle
  check than `CropPlan` validation.
- Raw Video and Timing share one `raw_probe`. Timing reads its completed
  sequential count, but neither the page nor recount-preparation method protects
  against dispatching the same expensive operation again.
- Crop preview is a one-pass OpenCV rendering of the composed homography. Stage A
  is a decomposed ffmpeg rendering with explicit pixel-center/boundary
  adaptation. Their intended geometry is shared, but end-to-end parity is not
  tested.
- Timing already has the selected units, vector statistics, converted timing
  result, and probed raw FPS needed for a warning. It currently renders only
  candidate raw statistics and a concise validation result, with no converted
  comparison or mismatch factor.

## 6. Minimal v1 corrective patch plan

### Manual geometry precision and contract

1. In `manual_crop._build_rectification_geometry()`, fit the homography using
   the same float64 local points retained by the authoritative plan and a
   float64 destination. Do not round, clamp, reorder, or replace the user's raw
   points.
2. Make Stage A's recovered-rectangle numerical check scale-aware and consistent
   with the CropPlan transform contract. Retain rejection of materially
   non-rectifying or non-finite geometry; do not simply remove validation.
3. Keep the ffmpeg pixel-center/boundary adaptation and all artifact, frame
   identity, rotation, canonical, and accepted-crop contracts unchanged.

This is smaller and safer than changing the CropPlan schema or replacing the
Stage A pipeline. The patch must be validated against the exact field CropPlan
before release.

### Same-session count reuse UX

1. Add a headless controller predicate that recognizes a positive sequential
   count belonging to the currently selected raw path.
2. When it is present, display `Already counted in this session: {count}` and
   disable or hide the recount button.
3. Guard `prepare_full_raw_frame_count()` against redundant dispatch so page
   refresh/busy-state transitions cannot accidentally re-enable a repeat count.
4. Do not add disk caching, fingerprints, or service-level count reuse in this
   v1 patch.

No v1 source patch is justified for the preview observation until the parity
test reproduces a geometric difference. The timing warning remains deferred to
v2.

## 7. Deferred v2 requirements

- Timing-unit consequence preview and the prominent, nonblocking mismatch
  warning described above.
- Early writer-profile FPS representability preflight. This is a separate
  technical hard-fail and must not be conflated with plausibility warning policy.
- Persistent, source-identity-aware raw probe/readable-count caching across
  sessions and optional safe service reuse.
- Any broader preview architecture change, but only if the parity regression
  confirms a geometric mismatch that cannot be corrected narrowly.
- The other reviewed v2 workflow, observability, MAT warning, progress, and
  real-video acceptance requirements remain governed by an approved future
  requirements revision.

## 8. Required regression tests

### A. Manual Stage A defect

Add these tests with non-integer points, a nonzero pre-crop origin, and
coordinates large enough to exercise float32 quantization:

- `tests/preprocess/test_manual_crop.py`: build a manual plan and transform its
  exact stored raw quadrilateral through the stored homography. Assert that it
  reaches the intended rectified/rotated/canonical geometry within the approved
  scale-aware tolerance, while the stored raw points remain unchanged.
- `tests/preprocess/test_video_prepare_filtergraph.py`: pass the accepted form of
  that plan to `build_prepare_filtergraph()`. Assert no false axis/integer
  rejection and verify recovered content dimensions, rectification padding,
  rotation stage, and canonical stages.
- `tests/integration/test_ffmpeg_prepare.py`: encode one synthetic frame with
  labeled corner/interior markers through Stage A using the fractional manual
  plan. Assert output dimensions, rotation, padding, visible bounds, and marker
  positions within the existing codec/interpolation tolerance.
- `tests/integration/test_preprocess_service.py`: run the accepted fractional
  manual plan through `PreprocessService` and assert successful creation and
  validation of the six official artifacts. This protects the field-visible
  orchestration path, not just filtergraph construction.

Retain negative tests proving materially inconsistent CropPlans still fail.

### B. Same-session count reuse

Add a headless `tests/ui/test_timing_controller.py` regression that initializes
state with a current raw path and a matching `raw_probe` containing a positive
sequential readable count. Use a fake probe function that records calls (or
fails if called), load/select a valid MATLAB candidate and units, and assert:

- candidate length matching uses the stored count;
- timing validation succeeds without a probe call;
- the controller reports that the current count is reusable; and
- explicit recount preparation is rejected or short-circuited without worker
  dispatch.

Add the complementary mismatched-source-path case, which must not reuse the
count. A lightweight page test may additionally assert that the button is
disabled and the existing-count message is shown, but the controller regression
is the required non-display test.

### C. Automatic and manual preview parity

Add an integration regression parameterized over one accepted automatic plan
and one accepted manual arbitrary four-corner plan. For each case:

1. create one synthetic raw frame containing uniquely colored corner, edge, and
   orientation markers;
2. use that exact raw frame and exact accepted CropPlan to generate the GUI
   preview;
3. encode the same frame through Stage A (or compare the final prepared frame if
   Stage B is intentionally included); and
4. compare the decoded output with codec-aware tolerances.

Assert exact output dimensions and parity of marker locations, clockwise
rotation, canonical padding offsets, and visible crop bounds. Separate ordinary
interpolation/compression error from a systematic coordinate offset. Include a
viewer mapping assertion that letterboxing stays outside `image_target_rect()`
and cannot be mistaken for image padding.

### D. Timing warning

Deferred v2 controller tests must cover seconds, milliseconds, microseconds,
nanoseconds, frames, unknown, missing raw FPS, threshold boundaries, the field
case near `119403` versus `119.1`, and the corrected near-match. Assert the
warning never changes units, timing arrays, frame mapping, or navigation
validity. Writer-profile tests must separately assert early hard failure only
for technically unrepresentable output FPS.

## 9. Explicit non-goals

- No new preprocessing, crop-detection, manual-crop, video-geometry, or timing
  algorithm in this audit.
- No CropPlan, metadata, sync, settings, logging, or artifact schema change.
- No frame resampling, dropping, duplication, interpolation, reordering, or
  altered frame identity.
- No crop auto-acceptance or relaxation of explicit user acceptance.
- No change to official artifact names or output roles.
- No persistent cache as a v1 defect fix.
- No automatic timing-unit inference or repair.
- No hard plausibility block based solely on unusual FPS.
- No batch processing, masking, SLEAP inference, or raw PTS extraction.
- No v2 implementation plan.

## 10. Recommended sequencing before the v1 release tag

1. Implement the narrow manual precision/Stage A contract patch and its unit and
   integration regressions. Validate the exact field CropPlan.
2. Implement the same-session count action guard and headless controller
   regression.
3. Add and run the automatic/manual preview-versus-Stage-A parity regression.
   If it passes, document the field observation as viewer/interpolation-related
   or otherwise unreproduced. If it fails geometrically, triage and correct the
   smallest shared-coordinate defect before release.
4. Run the complete repository quality gates and the reviewed real-video
   acceptance protocol, including automatic and manual runs.
5. Review generated metadata, sync, settings, logs, backgrounds, frame mapping,
   dimensions, FPS, and all six official artifacts.
6. Create the v1 release tag only after the two confirmed v1 defects are fixed,
   preview parity is demonstrated, and no blocking regression remains.
7. Revise and approve v2 requirements separately, including the latest
   warning-only FPS mismatch decision, before beginning v2 implementation.
