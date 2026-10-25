# Subsystem 06 — Sprint 1: Joint Input Validation and S4 Pose Adapter

**Status:** Provisionally accepted; consumed by Sprint 2.  
**Date:** 2026-10-21  
**Scope:** S4/S5 joint eligibility, source alignment, and S4→legacy pose adaptation.  
**Out of scope:** Feature formulas, Pipeline 50 postprocessing, blob merge, export, GUI, models.

---

## Deliverables

| Module | Role |
| --- | --- |
| `src/combined_features/contracts.py` | Package names, tolerances, impute-stage codes, reason codes |
| `src/combined_features/upstream.py` | Discover / revalidate / jointly align eligible S4+S5 |
| `src/combined_features/pose_adapter.py` | Long-form `final_pose.parquet` → FF tensors + implantation |
| `pose_finalization.run_discovery.find_latest_eligible_s4_run` | Symmetric newest-eligible S4 helper |

---

## Joint source-eligibility checks (exact)

`validate_joint_s4_s5_pair` / `evaluate_joint_alignment`:

1. **Upstream revalidation** — `validate_s4_run_for_downstream`, `validate_s5_run_for_downstream`
2. **Session** — resolve-equal `source_s3.session_root` == `source_s1.session_root` == requested session
3. **Sync path** — resolve-equal prepared_sync paths on S4 and S5
4. **Sync exists** — path is a file
5. **Sync digest** — `sha256(S4.prepared_sync)` equals S5 `source_s1.prepared_sync_sha256` (live hash; S4 does not store sync SHA)
6. **Target FPS** — lab 120 Hz (`JOINT_TARGET_FPS`); reject recorded overrides ≠ 120
7. **Sample count** — unique S4 samples == S5 rows == timeline length from sync
8. **Index integrity** — contiguous unique `[0 .. N-1]`; S5 `sample_idx` == `dense_frame_idx` == `frame_idx`
9. **Timestamps** — `np.allclose(time_s4, time_s5, rtol=0, atol=1e-12)`
10. **S4 structure** — unique `(sample_idx, track, node)`; consistent `time_sec` per sample

Filename/timestamp coincidence alone is never accepted as lineage proof. Missing provenance → `lineage_unprovable`.

Headless default: newest eligible each side + joint gate; alignment failure does **not** silently search other pairs (`select_default_joint_pair`).

---

## Pose adapter interface

```text
adapt_final_pose_to_frame_feature_input(
    final_pose: DataFrame,          # final_pose_v1 long-form
    *,
    orig_frame_idx: ndarray[int64], # from S5, length T
    run_meta: Mapping | None = None # optional dense_summary.node_names order
) -> AdaptedPose
```

`AdaptedPose` fields (FF I/O surface via `to_frame_feature_arrays` / `to_impute_meta`):

| Field | Shape / type | Source |
| --- | --- | --- |
| `x`, `y` | `(n_tracks, n_nodes, T)` float64 | S4 post-smooth |
| `t_120` | `(T,)` | S4 `time_sec` |
| `orig_frame_idx` | `(T,)` | **S5** only |
| `point_score` | same as x | S4 pass-through; never fabricated |
| `impute_stage` | int8 | provenance map |
| `is_imputed` | bool | linear ∪ kalman |
| `is_kf` / `is_pchip` / `was_refined` | bool | Kalman-only / always false / always false |
| `provenance` | uint8 | full S4 codes for QC |
| `protected_manual_blank` | bool | S4 |
| `tracks`, `nodes` | tuples | sorted tracks; alias-expanded nodes |
| `implantation` | `ImplantationResolution` | headstage evidence |

---

## Quality / provenance mapping

| Legacy flag | Rule |
| --- | --- |
| `*_is_kf` | provenance == `kalman_gap_fill` (5) |
| `*_is_pchip` | always false |
| `*_any_imputed_*` (Sprint 2 FF) | pchip ∨ kf → **Kalman-only** under S4 |
| `impute_stage` | 3=Kalman, 1=linear, else 0; never PCHIP(2) |
| `is_imputed` | linear ∪ kalman only |
| `was_refined` | always false |
| Not mapped to imputed/kf | temporal_resample, identity_corrected, smoothing, blanks, missing |

---

## Track and implantation handling

- Tracks: integer-coerced, sorted ascending; directional pairs = all ordered `i≠j`.
- Nodes: `spine`↔`spine_base`, `tail`↔`tail_base` both present in the tensor axis when either appears.
- **Implantation (binding Sprint 1 decision for D3):** do **not** assume track `0`.

### Deterministic resolution

Count finite headstage `(x,y)` observations per track (excluding missing / protected blanks):

| Evidence | `status` | `implanted_track_id` |
| --- | --- | --- |
| Exactly one track has ≥1 finite headstage | `resolved` | that track id |
| No finite headstage | `headstage_absent` | `None` |
| ≥2 tracks have finite headstage | `headstage_ambiguous` | `None` |

Profile `implanted_track: 0` is **ignored** as authority.

### Sprint 2 consumption

- If `implantation.status == "resolved"`: set `is_implanted = (focal_id == implanted_track_id)`.
- Otherwise: preserve uncertainty — emit `is_implanted=NaN` with an explicit QC warning that identity is unresolved (Sprint 3). **Never** fall back to `focal_id == 0` or encode uncertainty as `0`.

---

## Timing validation

- Unique contiguous `sample_idx`
- Matching S4/S5 counts
- `atol=1e-12`, `rtol=0` on `time_sec`
- No truncation / padding / interpolation in the adapter
- S5 `orig_frame_idx` preserved as the prepared-frame map for FF

---

## Deviations from S6-B design contract

| Topic | Contract | Sprint 1 | Reason |
| --- | --- | --- | --- |
| D3 `is_implanted` id==0 | Listed unresolved; legacy used id==0 | Headstage-only resolution; no id==0 default | User Sprint 1 requirement: do not assume track 0 |
| Package modules | Listed many future modules | Only `contracts`, `upstream`, `pose_adapter` | “Do not create empty future modules” |
| S4 `prepared_sync_sha256` | Optional hardening | Not added to S4 | Contract D5: not required; live hash at join time |
| Full runner / export | Sprint 2–3 | Deferred | Sprint 1 scope |

---

## Remaining dependencies for Sprint 2

1. Vendor `legacy/.../frame_features.py` behind `AdaptedPose.to_frame_feature_arrays()`
2. Port Pipeline 50 post steps (order preserved), consuming implantation metadata as above
3. Session-centric I/O (no runtime dependence on git-excluded `legacy/`)
4. Wire `select_default_joint_pair` → adapter → FF into a headless runner stub
