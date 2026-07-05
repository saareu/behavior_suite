# Subsystem 01 Design: Final Spatial Geometry Modes

**Status:** Active future-facing design reference; no production behavior is changed by this document.
**Relationship:** Complements `docs/subsystem_01_preprocessing.md` while preserving the approved artifact, frame identity, timing, and validation contracts until a separate implementation patch changes them explicitly.

## 1. Purpose and relationship to the current Subsystem 01 contract

V1 implements one final spatial workflow:

```text
raw video
→ optional pre-crop
→ automatic cage detection or manual four corners
→ perspective CropPlan
→ rectification / rotation / canonical scale-pad
→ prepared video
```

Raw decode-frame navigation and visual trim selection do not change spatial
processing. Future visual pre-crop work and future geometry modes must fit into
the same scientific contract.

The current perspective `CropPlan` workflow remains valid and should remain the
first fully supported processing mode. The correction in this document is that
`CropPlan` should not be described as the only possible long-term spatial
authority. Future runs may use an identity transform, an axis-aligned pre-crop
as final geometry, the current perspective `CropPlan`, or an explicit composed
geometry chain.

## 2. Current behavior and limitations

Current core behavior:

- `ResolvedPreCrop` resolves the typed pre-crop setting into one raw-coordinate
  integer ROI.
- `serialize_pre_crop()` records the ROI and an axis-aligned raw-to-pre-crop
  translation homography.
- Automatic detection and manual four-corner selection produce a validated
  `CropPlan`.
- Stage A ffmpeg preparation requires an explicitly accepted `CropPlan`.
- Stage A decomposes the `CropPlan` into trim, pre-crop, perspective
  rectification, optional rotation, even-dimension guard, and optional canonical
  uniform scale/pad.
- Stage B re-encodes the intermediate sequentially without spatial resizing.
- The GUI stores current/candidate/accepted crop state as `CropPlan` values.

Limitations:

- A run cannot currently complete without a `CropPlan`.
- Pre-crop is always treated as input to a later perspective crop, not as a
  possible final geometry.
- Identity and pre-crop-only workflows have no accepted-geometry state,
  metadata representation, or service path.
- Future transforms such as masks, axis-aligned crops, perspective transforms,
  rotation, scaling, and padding are not represented as an ordered chain.

## 3. Final spatial geometry mode taxonomy

The long-term taxonomy should distinguish the final spatial geometry mode from
frame trim and timing.

### `identity`

The selected raw frames pass through spatially unchanged. No crop, homography,
rotation, scale, or padding is applied. If existing output constraints require
even dimensions or a supported encoder size, identity mode must either validate
that the raw dimensions already satisfy those constraints or fail explicitly.

### `axis_aligned_pre_crop_only`

The user-selected pre-crop is the final spatial geometry. The prepared frame is
the retained raw rectangle or split-line region. No cage detection, manual four
corners, perspective homography, or rotation is required.

If canonical output settings are later allowed with this mode, the uniform
scale/pad transform is part of the authoritative transform description.
Otherwise, prepared size is the retained ROI size.

### `perspective_crop_plan`

The current V1/V2 perspective workflow. Automatic cage detection or manual
four-corner selection produces a validated, accepted `CropPlan`. The `CropPlan`
stores raw-to-prepared and prepared-to-raw homographies and the output size.

### `composed_geometry`

A future explicit ordered transform chain. It may include axis-aligned crop,
perspective transform, rotation, uniform scale, padding, masking, or other
validated spatial steps. The chain must be represented explicitly rather than
reconstructed from hidden GUI state or inferred from ad hoc metadata fields.

## 4. Authoritative transform contract

Every prepared video must retain one authoritative raw-to-prepared and
prepared-to-raw transform description.

The transform description must be sufficient to map prepared coordinates back
to raw-video coordinates without relying on hidden GUI state.

The representation may differ by geometry mode:

- `identity`: identity transform plus raw/prepared size equality.
- `axis_aligned_pre_crop_only`: translation/crop transform, and optional
  explicit uniform scale/pad if canonical sizing is enabled.
- `perspective_crop_plan`: current `CropPlan` homographies and geometry fields.
- `composed_geometry`: an ordered list of validated transform steps with an
  explicit inverse or a provably invertible inverse mapping where applicable.

This contract does not require introducing a new `GeometryPlan` model in the
first future geometry implementation. It defines the target rule for future schema work.

## 5. Coordinate-system definitions

All spatial modes must name their coordinate systems explicitly.

- **Raw frame coordinates:** pixel coordinates in the decoded raw image. Origin
  is top-left. `x` increases rightward; `y` increases downward.
- **Raw decode-frame index:** sequential raw frame identity. This is temporal
  identity, not a spatial coordinate.
- **Pre-crop coordinates:** local coordinates inside a retained raw ROI. For an
  ROI `(x, y, width, height)`, pre-crop coordinate `(0, 0)` corresponds to raw
  `(x, y)`.
- **Perspective/native coordinates:** the rectified content coordinate system
  before optional rotation and canonical scale/pad.
- **Prepared coordinates:** final prepared-video pixel coordinates after all
  spatial transforms in the selected geometry mode.
- **GUI display coordinates:** widget coordinates after scaling/letterboxing.
  These are never scientific coordinates and must be converted to raw image
  coordinates before validation.

Rectangular regions use half-open image geometry:

```text
[x, x + width) × [y, y + height)
```

Split-line pre-crop modes use the same half-open convention:

```text
Keep Left:  [0, boundary_x)
Keep Right: [boundary_x, raw_width)
Keep Upper: [0, boundary_y)
Keep Lower: [boundary_y, raw_height)
```

## 6. Raw-to-prepared and prepared-to-raw requirements

For every successful run:

1. The selected raw decode frames remain determined by trim only.
2. Spatial transforms are applied identically to every selected frame unless a
   future mode explicitly permits frame-varying geometry and documents it.
3. The transform description records source and target sizes for every step.
4. Prepared-to-raw mapping is available for downstream coordinate projection.
5. The mapping does not depend on GUI widget size, current display zoom,
   transient overlays, or page-local objects.
6. The final prepared video dimensions match the authoritative transform target.
7. The transform metadata is validated before official artifacts are promoted.

The frame mapping remains separate:

```text
raw_decode_frame_idx = start_frame + prepared_frame_idx
```

in the default no-resampling mode. Geometry mode changes must not change this
mapping.

## 7. Pre-crop-only mode design

`axis_aligned_pre_crop_only` treats `ResolvedPreCrop` as the accepted final
geometry.

Required transform data:

- raw source size;
- retained ROI in raw coordinates;
- prepared output size;
- raw-to-prepared mapping;
- prepared-to-raw mapping;
- optional canonical scale/pad if canonical sizing applies.

For crop-only without canonical sizing:

```text
raw_to_prepared(x, y) = (x - roi.x, y - roi.y)
prepared_to_raw(u, v) = (u + roi.x, v + roi.y)
prepared_size = (roi.width, roi.height)
```

No homography, cage quadrilateral, automatic detector output, manual
four-corner selection, or rotation is required. The user still must explicitly
accept the spatial geometry before processing, because pixels outside the ROI
are permanently excluded from prepared output.

The first visual pre-crop or geometry implementation should not implement this processing mode unless
metadata and service support are explicitly added in a later scoped patch.

## 8. Perspective CropPlan mode design

`perspective_crop_plan` is the current supported mode.

Current authority:

- accepted `CropPlan`;
- `pre_crop_roi`;
- `quad_raw_tl_tr_br_bl`;
- `H_raw_to_prepared_3x3`;
- `H_prepared_to_raw_3x3`;
- `native_size_wh`;
- `canonical_geometry`;
- `prepared_size_wh`;
- `rotated_90`;
- acceptance flag and crop mode provenance.

The perspective mode remains valid and should continue to use the existing
automatic detection and manual four-corner review workflow. Future visual
pre-crop must feed the existing `PreCropConfig` and `resolve_pre_crop()` path,
then require a new perspective crop review only when the effective pre-crop
geometry changes.

## 9. Future composed-geometry design

`composed_geometry` should be an explicit ordered transform chain. A future
schema may represent steps such as:

```text
raw_decode_frame
→ axis_aligned_crop
→ perspective_rectification
→ rotation
→ even_dimension_guard
→ canonical_uniform_scale
→ canonical_padding
→ prepared_coordinate_mask
→ prepared_frame
```

Each step should record:

- step type and version;
- input coordinate system and size;
- output coordinate system and size;
- parameters;
- raw-to-step or step-to-step transform;
- inverse mapping when meaningful;
- validation status.

Composed geometry must not be reconstructed from ffmpeg filtergraph strings
alone. The filtergraph may implement the transform, but metadata must remain
the scientific authority.

## 10. GUI workflow implications

Future visual pre-crop should reuse the raw-frame viewer and mapping layer.

Expected first UI behavior:

- draw vertical/horizontal split-line candidates;
- draw manual rectangle candidates;
- show retained-region overlay and numeric ROI;
- synchronize overlay edits with numeric controls;
- pass resulting numbers through `PreCropConfig` and `resolve_pre_crop()`;
- display core validation errors clearly;
- avoid duplicating core geometry validation in Qt widgets.

Trim and geometry must be separated:

```text
trim range changes → change only which frames are processed
spatial geometry changes → change how each selected frame is transformed
```

Therefore, changing only `start_frame` or `end_frame_exclusive` must not
invalidate accepted spatial geometry for an unchanged raw video. Changing
pre-crop geometry, manual points, automatic detection output,
geometry-affecting output settings, or raw video identity must invalidate
accepted geometry.

Future non-perspective modes should still include an explicit geometry review
or acceptance step, even if they do not use the current Crop Review page.

## 11. Validation requirements

Common validation:

- raw video dimensions are known and positive;
- requested ROI or transform lies within the raw frame where required;
- output dimensions are positive and satisfy current encoder constraints;
- transform parameters are finite;
- prepared-to-raw mapping is defined for every prepared-coordinate pixel center
  or documented for non-invertible operations such as masks;
- no hidden GUI state is required to reproduce the transform;
- metadata transform target size matches prepared-video validation size.

Mode-specific validation:

- `identity`: raw and prepared spatial dimensions match exactly, unless a
  later explicit scale/pad step is added under another mode.
- `axis_aligned_pre_crop_only`: ROI is non-empty, inside raw bounds, and maps
  round trip by translation or explicit scale/pad.
- `perspective_crop_plan`: existing `CropPlan` validation remains binding.
- `composed_geometry`: every step validates locally and the composed chain
  validates end-to-end.

## 12. Metadata and synchronization implications

No metadata schema, artifact format, or `prepared_sync.npz` change is made by
this design document.

Future implementation will require a separately reviewed metadata revision
before non-`CropPlan` modes can produce official artifacts. That revision should
record:

- final spatial geometry mode;
- authoritative transform representation;
- raw and prepared coordinate systems;
- source and target sizes;
- accepted geometry review provenance;
- backward-compatible interpretation of existing `CropPlan` metadata.

`prepared_sync.npz` remains the frame identity and timing artifact. Spatial
geometry belongs in `prepare_meta.json`, not in `prepared_sync.npz`.

`settings_used.yaml` may record accepted settings, but not hidden geometry
state. Authoritative accepted geometry belongs in metadata.

## 13. Backward compatibility constraints

Until a schema revision is approved:

- existing V1/V2 perspective runs remain interpreted as
  `perspective_crop_plan`;
- existing `CropPlan` metadata meaning must not change;
- official artifact names remain unchanged;
- frame mapping remains unchanged;
- external timing validation remains unchanged;
- prepared-video validation remains unchanged;
- current CLI and GUI paths requiring an accepted `CropPlan` remain valid.

Older metadata without an explicit geometry mode should be treated as the
current perspective mode when a valid accepted `CropPlan` is present.

## 14. Explicit non-goals for the first visual pre-crop / geometry implementation

The first implementation should not:

- introduce a new `GeometryPlan` model;
- change `CropPlan` fields or validation;
- change pre-crop core behavior;
- change `prepare_meta.json` schema;
- change `prepared_sync.npz`;
- change frame mapping;
- implement pre-crop-only processing;
- implement identity processing;
- implement composed-geometry processing;
- add rotation adjustment, playback, or timeline rendering;
- change codec, container, or encoding behavior;
- move scientific geometry validation into GUI widgets.

## 15. Recommended implementation milestones

1. **Documentation and contract alignment.** Land this design and adjust the V2
   plan language so future work does not assume `CropPlan` is the only possible
   geometry authority.
2. **Viewer-coordinate mapping tests.** Protect raw-image coordinate conversion
   from widget letterboxing, scaling, and right/bottom half-open edges.
3. **Visual pre-crop overlay for existing perspective mode.** Add split-line
   and rectangle overlays that write typed pre-crop numeric inputs only.
4. **Numeric/visual synchronization.** Keep direct numeric controls and overlay
   candidates synchronized in both directions.
5. **Invalidation correction checks.** Confirm trim-only edits preserve
   accepted spatial geometry, while pre-crop geometry edits invalidate it.
6. **Real-data perspective-mode acceptance.** Verify visual pre-crop still
   feeds the existing perspective `CropPlan` workflow and produces unchanged
   artifact semantics.
7. **Future schema proposal.** Separately design metadata for identity,
   pre-crop-only, and composed geometry before implementing those modes.

## 16. Future test strategy and real-data acceptance criteria

Headless/unit tests:

- ROI resolution for every pre-crop mode;
- overlay-to-raw coordinate mapping with letterboxing excluded;
- split-line half-open edge semantics;
- rectangle drag directions and clamping/rejection behavior;
- numeric-to-overlay and overlay-to-numeric synchronization;
- trim-only change preserves accepted geometry;
- pre-crop geometry change invalidates accepted geometry;
- raw video change invalidates accepted geometry;
- current perspective `CropPlan` behavior remains unchanged.

Integration tests:

- visual pre-crop values feed the same `resolve_pre_crop()` core path as direct
  numeric entry;
- automatic and manual perspective crops generated after visual pre-crop match
  existing `CropPlan` validation rules;
- prepared video, metadata, background, and sync artifacts remain schema- and
  frame-compatible with V1/V2.5.

Real-data acceptance:

- run a known automatic perspective workflow with no visual pre-crop change and
  confirm no artifact or frame-mapping regression;
- run visual split-line pre-crop followed by automatic detection and explicit
  crop acceptance;
- run visual rectangle pre-crop followed by manual four-corner crop and
  explicit crop acceptance;
- verify the retained raw region, accepted perspective geometry, prepared
  dimensions, metadata, background dimensions, and sync frame count;
- record OS, OpenCV/backend, ffmpeg/ffprobe, configuration, selected ROI,
  accepted `CropPlan`, output dimensions, and validation results.
