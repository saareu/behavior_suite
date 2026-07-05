# Preprocess Subsystem v1 Release Readiness

> **Archive notice:** This historical document is retained for traceability. It is not the current source of truth. See `docs/subsystem_01_preprocessing.md`, `docs/subsystem_01_status_and_roadmap.md`, and `docs/design/subsystem_01_geometry_modes.md`.

**Readiness review date:** 2026-06-23  
**Status:** Release candidate pending commit and final acceptance

## 1. Release candidate scope

This release candidate covers the implemented v1 Video Preprocessing subsystem:
the shared project layer, typed configuration and core service, CLI workflow,
seven-page desktop GUI, strict validation, and the six official preprocessing
artifacts. It includes only the narrow corrective patches verified after field
testing.

This is not a v2 release. Timing-unit warnings, writer-profile FPS preflight,
persistent cross-session probe caching, visual trim/pre-crop navigation,
fine-grained service progress, structured MAT warnings, masking, batch
processing, raw PTS extraction, and SLEAP inference remain outside this release.

## 2. Confirmed implemented v1 workflow

The implemented desktop workflow is:

```text
Project
→ Raw Video
→ Trim and Pre-Crop
→ Timing
→ Crop Review
→ Encode Settings
→ Run and Validate
```

Automatic and manual crop paths both produce the same validated `CropPlan`.
Crop acceptance is explicit. The GUI and CLI invoke the existing core
`PreprocessService`; scientific processing and validation are not duplicated in
the GUI.

## 3. Scientific safety guarantees

The release candidate retains these v1 guarantees:

- Raw sequential decode order remains the primary frame identity.
- The default mapping is
  `raw_decode_frame_idx = start_frame + prepared_frame_idx`.
- Frames are not dropped, duplicated, reordered, or temporally resampled.
- External timing must be numeric, finite, monotonic, one-dimensional after
  squeeze, and exactly match the original untrimmed readable-frame count.
- `frames` and `unknown` timing units do not invent seconds.
- A validated crop and explicit user acceptance are required before processing.
- Canonical geometry uses uniform scaling and centered padding only.
- The final video must pass strict OpenCV reported/readable/expected frame-count
  and dimension validation.
- `prepare_meta.json` remains the authoritative processing record.
- Failures are not presented as successful or scientifically usable partial
  outputs.

## 4. Official artifacts

A successful run produces exactly:

```text
prepared_video.mp4
prepare_meta.json
prepared_sync.npz
cropped_background.png
settings_used.yaml
processing_log.txt
```

Internal and debug files are not official outputs.

## 5. Real-data acceptance evidence

Two full-video GUI workflows have completed successfully:

1. Automatic cage detection with external timing produced all six official
   artifacts and passed final validation. The retained metadata records 45,716
   prepared frames, 928 × 528 output, and approximately 119.403 FPS from
   external timing.
2. Manual four-corner crop without external timing produced all six official
   artifacts and passed final validation. The retained metadata records 45,716
   prepared frames, 928 × 528 output, and approximately 119.106914 FPS.

These local acceptance projects are evidence, not repository source material.
They must not be included in a release commit or tag.

## 6. Corrective patches verified after field testing

The following narrow corrections were verified after field testing:

- Manual fractional four-corner `CropPlan` geometry now reaches Stage A without
  the false precision rejection while materially inconsistent geometry remains
  rejected.
- GUI preview and Stage A output geometry have automatic and manual parity
  regression coverage, including viewer letterboxing separation.
- A same-session completed sequential readable-frame count is reused and a
  redundant Timing-page recount is protected.
- Reopened projects hydrate completed preprocessing state from authoritative
  artifacts.
- Long GUI operations support cooperative cancellation, and stale task results
  are not applied after upstream state changes.
- GUI shutdown and navigation protect active task lifecycle boundaries.

No approved v1 scientific rule, artifact contract, or frame-identity rule was
changed by these corrective patches.

## 7. Test and quality-gate status

The release-hygiene validation completed on 2026-06-23 with:

```text
pytest: 367 passed
ruff check src tests: passed
git diff --check (staged and unstaged): passed
```

The same gates must pass once more on the final clean working tree after the
logical commits are created and before tagging.

## 8. Known limitations explicitly deferred to v2

- Timing-unit consequence preview and a prominent nonblocking FPS mismatch
  warning.
- Early writer-profile FPS representability preflight.
- Persistent, source-fingerprinted raw probe/readable-count caching shared
  across sessions and callers.
- Visual raw-frame trim navigation and visual pre-crop selection.
- Service-level detailed progress callbacks and evidence-based ETA.
- Structured MAT parser warnings and duplicate/unusable candidate handling.
- Static masking, batch preprocessing, raw PTS extraction, and SLEAP inference.

These limitations must not be described as implemented v1 features.

## 9. Local-data and Git hygiene requirements before tagging

- The 63 tracked Python bytecode artifacts listed in Appendix A must be removed
  from the Git index without deleting local files.
- `.gitignore` must continue to exclude Python caches, test/coverage caches, and
  temporary/partial files without broadly excluding scientific data formats.
- `4050/` and `4050_no_ttl/` must remain unstaged. Before final cleanup, move
  local field-test projects outside the repository or adopt a separately
  reviewed, explicitly ignored local-runs convention.
- `docs/issue_logs/` must be reviewed separately and must not be deleted or
  accidentally included in the release.
- YAML configurations, JSON metadata, NPZ synchronization artifacts, PNG
  backgrounds, and source files are intentionally not ignored globally.

## 10. Recommended final commit sequence

1. Commit `.gitignore`, `.gitattributes`, this readiness document, and the
   index-only removal of tracked generated bytecode. This clears the deliberate
   hygiene-only index changes before staging corrective patches.
2. Commit the manual `CropPlan` Stage A corrective source and regression tests.
3. Retain the preview/output parity correction already present in tracked
   commit `fbfc7be`; no additional parity files remain uncommitted.
4. Commit same-session count-reuse and GUI task-lifecycle source/tests.
5. Run the full checks on the resulting clean tree and perform the final short
   user acceptance checks. Do not combine local field-test projects or
   unreviewed issue logs with these commits.

## 11. Recommended release-tag criteria

v1 is a release candidate only after all of the following are true:

- Real corrective-patch source/test changes are committed.
- Generated bytecode is removed from tracking.
- Local field-test folders are not accidentally staged.
- Final checks pass on a clean working tree.
- The user performs final short acceptance checks.
- All six artifacts are confirmed for both automatic/external-timing and
  manual/no-external-timing workflows.
- No blocking scientific, validation, artifact, or GUI lifecycle regression is
  open.

Only then should an annotated v1 release tag be considered. This hygiene pass
does not create a tag.

## Appendix A. Generated artifacts removed from Git tracking

The release-hygiene pass removes these tracked generated files from the Git
index while preserving their local working copies:

- `src/preprocess/__pycache__/__init__.cpython-312.pyc`
- `src/preprocess/__pycache__/config.cpython-312.pyc`
- `src/preprocess/__pycache__/exceptions.cpython-312.pyc`
- `src/preprocess/__pycache__/mat_sync_reader.cpython-312.pyc`
- `src/preprocess/__pycache__/models.cpython-312.pyc`
- `src/preprocess/__pycache__/video_probe.cpython-312.pyc`
- `src/project/__pycache__/__init__.cpython-312.pyc`
- `src/project/__pycache__/models.cpython-312.pyc`
- `src/project/__pycache__/paths.cpython-312.pyc`
- `src/project/__pycache__/service.cpython-312.pyc`
- `src/project/__pycache__/validation.cpython-312.pyc`
- `src/ui/__pycache__/__init__.cpython-312.pyc`
- `src/ui/__pycache__/app.cpython-312.pyc`
- `src/ui/__pycache__/main_window.cpython-312.pyc`
- `src/ui/__pycache__/preprocess_wizard.cpython-312.pyc`
- `src/ui/__pycache__/state.cpython-312.pyc`
- `src/ui/__pycache__/tasks.cpython-312.pyc`
- `src/ui/controllers/__pycache__/__init__.cpython-312.pyc`
- `src/ui/controllers/__pycache__/crop_review_controller.cpython-312.pyc`
- `src/ui/controllers/__pycache__/encode_settings_controller.cpython-312.pyc`
- `src/ui/controllers/__pycache__/preprocess_setup_controller.cpython-312.pyc`
- `src/ui/controllers/__pycache__/run_preprocess_controller.cpython-312.pyc`
- `src/ui/controllers/__pycache__/timing_controller.cpython-312.pyc`
- `src/ui/pages/__pycache__/__init__.cpython-312.pyc`
- `src/ui/pages/__pycache__/crop_review_page.cpython-312.pyc`
- `src/ui/pages/__pycache__/encode_settings_page.cpython-312.pyc`
- `src/ui/pages/__pycache__/project_page.cpython-312.pyc`
- `src/ui/pages/__pycache__/raw_video_page.cpython-312.pyc`
- `src/ui/pages/__pycache__/run_validate_page.cpython-312.pyc`
- `src/ui/pages/__pycache__/timing_page.cpython-312.pyc`
- `src/ui/pages/__pycache__/trim_precrop_page.cpython-312.pyc`
- `src/ui/widgets/__pycache__/__init__.cpython-312.pyc`
- `src/ui/widgets/__pycache__/crop_overlay_view.cpython-312.pyc`
- `src/ui/widgets/__pycache__/video_frame_view.cpython-312.pyc`
- `tests/__pycache__/__init__.cpython-312.pyc`
- `tests/integration/__pycache__/__init__.cpython-312.pyc`
- `tests/integration/__pycache__/test_ffmpeg_prepare.cpython-312-pytest-8.4.2.pyc`
- `tests/integration/__pycache__/test_prepared_video_validation.cpython-312-pytest-8.4.2.pyc`
- `tests/integration/__pycache__/test_preprocess_service.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/__init__.cpython-312.pyc`
- `tests/preprocess/__pycache__/test_background.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_cage_detection.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_config.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_crop_plan.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_manual_crop.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_masking.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_mat_sync_reader.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_metadata.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_opencv_reencode.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_pre_crop.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_sync_writer.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_validation.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_video_prepare_filtergraph.cpython-312-pytest-8.4.2.pyc`
- `tests/preprocess/__pycache__/test_video_probe.cpython-312-pytest-8.4.2.pyc`
- `tests/project/__pycache__/__init__.cpython-312.pyc`
- `tests/project/__pycache__/test_paths.cpython-312-pytest-8.4.2.pyc`
- `tests/project/__pycache__/test_service.cpython-312-pytest-8.4.2.pyc`
- `tests/ui/__pycache__/test_crop_review_controller.cpython-312-pytest-8.4.2.pyc`
- `tests/ui/__pycache__/test_encode_settings_controller.cpython-312-pytest-8.4.2.pyc`
- `tests/ui/__pycache__/test_preprocess_setup_controller.cpython-312-pytest-8.4.2.pyc`
- `tests/ui/__pycache__/test_run_preprocess_controller.cpython-312-pytest-8.4.2.pyc`
- `tests/ui/__pycache__/test_tasks.cpython-312-pytest-8.4.2.pyc`
- `tests/ui/__pycache__/test_timing_controller.cpython-312-pytest-8.4.2.pyc`
