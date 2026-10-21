# Subsystem 05 — Sprint 1 Backend Foundation

**Status:** Implemented (backend only). Awaiting review before Sprint 2.
**Date:** 2026-10-20
**Authority:** Approved Sprint 1 scope + `sprint_0_repository_audit.md`.

---

## What landed

| Area | Module | Role |
| --- | --- | --- |
| Contracts | `blob_features.contracts` | Artifact names, statuses, frozen Parquet columns, errors |
| S1 eligibility | `blob_features.s1_eligibility` | Validate eligible S1 independently of S2–S4 |
| Timing | `blob_features.timing` | Uniform 120 Hz timeline + nearest-neighbor frame map |
| Config | `blob_features.config` | Segmentation vs execution settings; lab YAML defaults |
| Segmentation | `blob_features.segmentation` | Per-frame mask + contour topology (legacy science) |
| Suite YAML | `configs/subsystem_05/blob_features_default.yaml` | Faithful lab defaults without cable keys |

No GUI, runner, parquet export, temporal features, or S1–S4 behavior changes.

---

## Implemented contracts

### Eligible S1 input

Required under `<session_root>/preprocess/` (or a preprocess directory itself):

- `prepared_video.mp4`
- `cropped_background.png`
- `prepared_sync.npz` (via `load_prepared_sync_npz` / `load_source_timing`)
- `prepare_meta.json`

Checks:

- Sync loads and source timing is contiguous / monotonic.
- Video opens; header frame count matches `frame_count_used_for_sleap`.
- Background PNG readable; HxW matches the first prepared-video frame.
- Optional `prepare_meta.json` width/height and frame-count fields, when present, must agree.

S2–S4 directories are never consulted. Eligibility succeeds with only S1 present.

### Timing (authoritative 120 Hz → prepared frame)

Reuses:

- `pose_finalization.timing.load_source_timing`
- `pose_finalization.timing.build_uniform_target_timeline`
- `TARGET_FPS_DEFAULT = 120.0`

S5 mapping (differs from S4 pose interpolation):

```text
source_frame_idx[k] = argmin_i |prepared_time_sec[i] - target_time_sec[k]|
```

**Tie-break:** when left and right neighbors are equidistant, select the
**earlier** (left) source frame.

Exact coincidence uses the same absolute tolerance as S4 (`1e-9` s).

Validated: target length, frame bounds, monotonic non-decreasing
`orig_frame_idx`, and endpoint behavior (start at `t0` → first frame; end on
`t1` → last frame).

### Configuration

| Block | Contents |
| --- | --- |
| `BlobSegmentationSettings` | Lab scientific parameters (threshold, blur, morphology, areas, prune, abs-dark) |
| `BlobExecutionSettings` | `n_workers`, `max_in_flight` only |

Defaults match `legacy/configs/blob_features.yaml`. Suite copy:
`configs/subsystem_05/blob_features_default.yaml`.

**Cable exclusion:** `remove_cable_artifact` and all `cable_*` keys are out of
active S5 configuration / GUI scope. This assumes the legacy lab default
`remove_cable_artifact: false`. Enabling cable removal in a loaded YAML is
rejected. The legacy helper `tools.remove_cable_artifact` is also missing from
this checkout.

Deferred (Sprint 2+): `cc_count_gap_fill_max_gap`, full-video runner, temporal
occlusion features, parquet export.

### Segmentation core

`segment_frame(frame, background, settings) -> FrameSegmentationResult`

Scientific sequence (legacy Pipeline 40):

1. Foreground subtraction (`cv2.subtract(background, frame)`)
2. Gaussian blur + threshold (`>=0` fixed, `-1` Otsu, `-2` Multi-Otsu)
3. Absolute-dark AND, morphological cleanup, area filters, thickness prune
4. Connected-component / contour topology (`cc_count`, `cc_count_raw`,
   `min_inter_contour_dist`)

Same core is intended for interactive preview and full extraction later.
Occlusion static morphology (`ellipse_eccentricity`, DT peaks, skeleton, …)
remains Sprint 2 (`features` module); column names are already frozen in
`BLOB_FEATURES_COLUMNS`.

---

## Dependencies

| Package | Why |
| --- | --- |
| `scikit-image>=0.22,<1` | Multi-Otsu (`blob_threshold=-2`, lab default) |

Already present and reused: OpenCV, NumPy, SciPy (`cKDTree`), PyYAML.

---

## Known limitations

1. No full-video extraction, temporal features, gap fill, or Parquet writer.
2. No S5 GUI / group navigation.
3. Cable-artifact removal excluded (and legacy tool missing).
4. Occlusion static features not yet ported (column names reserved).
5. S5 imports `pose_finalization.timing` for shared timeline construction
   (accepted Sprint 1 coupling; optional later extract to a neutral package).
6. Staleness / S1 digest revalidation policy still unresolved (Sprint 0 §7).

---

## Remaining Sprint 2 prerequisites

1. Port occlusion static feature helpers (skimage peak/skeleton path).
2. Implement source-rate gap fill (`cc_count_gap_fill_max_gap`).
3. Full-video runner: read prepared frames by NN map, parallel topology, QC,
   `settings_used.yaml`, `run_meta.json`, failure cleanup (S4-style).
4. Parquet export with frozen `BLOB_FEATURES_COLUMNS` (+ decide legacy
   `dense_frame_idx` / `frame_idx` aliases for S6 if required).
5. Temporal occlusion features (bout-scoped rolling stats).
6. Decide bookmark / GUI preview wiring (backend core already callable).
7. Confirm whether cable cleanup stays permanently disabled or is restored
   from another lab copy before any enablement.

---

## Tests

```text
tests/blob_features/test_s1_eligibility.py
tests/blob_features/test_timing_nn.py
tests/blob_features/test_config.py
tests/blob_features/test_segmentation.py
```
