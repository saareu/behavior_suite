# Subsystem 05 — Sprint 2 Complete Video Feature Extraction Backend

**Status:** Implemented (backend only). Awaiting review.
**Date:** 2026-10-21
**Authority:** Approved Sprint 2 scope + Sprint 0/1 docs + legacy Pipeline 40.

---

## What landed

| Area | Module | Role |
| --- | --- | --- |
| Spatial occlusion | `blob_features.features` | Ellipse eccentricity, circularity, solidity, DT peaks, skeleton endpoints |
| Temporal processing | `blob_features.features` | Source-rate gap fill + bout-scoped rolling mean/delta/std |
| Export | `blob_features.export` | Model-compatible `blob_features.parquet` writer |
| QC | `blob_features.qc` | Structural + operational QC (`qc.json`) |
| Runner | `blob_features.runner` | Headless full-video extraction with S4-style artifacts |

No GUI, bookmarks, group navigation, S6, cable removal, or S1–S4 behavior changes.

---

## Sprint 1 verification

| Concern | Finding |
| --- | --- |
| Scientific config use | All active segmentation parameters are applied in `segmentation._extract_robust_foreground_mask`. No silent miswiring. |
| Timing | NN map uses `prepared_time_sec` + S4 `build_uniform_target_timeline`; left equidistant tie-break. Correct. |
| Raw vs effective `cc_count` | `cc_count = min(raw, 2)` then NaN when `raw > 2`. Matches legacy. |

**Concrete Sprint 1 gap closed in Sprint 2:** `cc_count_gap_fill_max_gap` was deferred; it is now an active `BlobTemporalSettings` field loaded from suite/legacy YAML. Schema aliases `dense_frame_idx` / `frame_idx` were added for Pipeline 50 join compatibility (alongside `sample_idx` / `time_sec`).

---

## Processing order (preserved)

1. Decode prepared video frame `i` in order `[0..N-1]`
2. Segment + topology + occlusion static features (when `cc_count == 1`)
3. Short-gap `cc_count` / `min_inter_contour_dist` imputation (**source rate**)
4. Bout-scoped temporal occlusion features (**source-frame windows**, not 120 Hz)
5. Nearest-neighbor map onto authoritative 120 Hz timeline
6. Structural QC → atomic parquet publish

---

## Output schema

Columns: see `BLOB_FEATURES_COLUMNS` in `blob_features.contracts`.

Pipeline 50 join: prefers `dense_frame_idx`, falls back to `frame_idx`. Both equal `sample_idx` after mapping. `orig_frame_idx` is the NN prepared-video frame.

---

## Known limitations

1. Sequential decode only (execution knobs recorded, parallelism deferred).
2. No GUI / bookmarks / group navigation.
3. Cable-artifact removal remains excluded.
4. Numerical equivalence vs entire legacy Pipeline 40 is not an MVP acceptance requirement.
5. S1 digest staleness policy beyond completion-time revalidation remains open (Sprint 0 §7).
