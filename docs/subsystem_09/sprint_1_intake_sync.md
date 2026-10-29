# Subsystem 09 — Sprint 1: Input Validation and Synchronization Index

**Status:** Implemented (headless). Stop for architectural review before Sprint 2.  
**Date:** 2026-10-28  
**Scope:** S8 intake contracts, eligibility validation, in-memory `sync_index` construction, bout-boundary prepared-frame mapping, dependency identity snapshot. **No** immutable S9 publication, export profiles, GUI, or S10.

Authoritative design: [architecture_audit.md](architecture_audit.md) (S9-A), [implementation_contract.md](implementation_contract.md) (S9-B). U-R1/U-R2 recorded in S9-B §11.1.

---

## 1. Modules and APIs

Package: `src/results_packaging/`

| Module | Role |
| --- | --- |
| `contracts.py` | Schema stamps, sync columns, approved `review_emit_v1` defaults, errors |
| `config.py` | `ResultsPackagingConfig` / YAML loader (`configs/subsystem_09/results_packaging_default.yaml`) |
| `eligibility.py` | S8 selection + structural/provenance validation; sync/video lineage resolve |
| `sync_index.py` | Rebuild dense↔prepared map via S5 NN; validate sync table |
| `bout_mapping.py` | Inclusive S8 bout → prepared seek indices (science unchanged) |
| `dependencies.py` | In-memory dependency manifest + fingerprint token |
| `results.py` | `Sprint1PackagingResult` |
| `runner.py` | `run_results_packaging_sprint1` / `run_sprint1_from_session` |

### Primary entry point

```python
from results_packaging import (
    ResultsPackagingRequest,
    run_results_packaging_sprint1,
)

result = run_results_packaging_sprint1(
    ResultsPackagingRequest(
        session_root=session_root,
        s8_run_dir=explicit_s8_or_None,  # None → latest eligible
    )
)
# result.sync.sync_index          # DataFrame
# result.bout_boundary_map        # DataFrame
# result.dependencies.to_dict()   # provenance snapshot
# result.to_metadata()["published"] == False
```

---

## 2. Validation rules (Sprint 1)

`validate_s8_source_for_packaging` requires:

1. Explicit S8 run **or** `find_latest_eligible_s8_run`.
2. Current eligibility via S8 `validate_s8_run_for_downstream` / `is_s8_run_eligible_for_downstream` (historical `status` not rewritten).
3. States/bouts schema stamps `behavioral_arbitration_states_v1` / `behavioral_arbitration_bouts_v1`.
4. Exact linked S7/S6 identities from S8 meta, cross-checked against downstream validation payload (no silent substitution).
5. Required S8 files: states, bouts, `qc.json`, `run_meta.json`, `settings_used.yaml`.
6. Resolvable `prepared_sync.npz` + `prepared_video.mp4` from S6→S4/S5/S1 lineage (video required by default).
7. Unique directional keys `(dense_frame_idx, focal_id, partner_id)`; `sample_idx == dense_frame_idx`.
8. Unique `bout_id`; consistent `source_s7_run_id` / `source_s6_run_id` / `run_id` when present.
9. Required time/index fields present and finite; contiguous dense `0…N-1`; bout inclusive duration `end-start+1`.
10. Prepared-sync live SHA256 must match recorded lineage digest; missing/moved files fail (no alternate path guess).

No arbitrary scientific QC thresholds are introduced.

---

## 3. Source identity handling

| Identity | Source |
| --- | --- |
| S8 run | Explicit path or latest eligible discovery summary |
| S7 / S6 | Exact `run_id` / `run_dir` from S8 `run_meta` + digest revalidation via S8 APIs |
| Prepared sync | S6 `source_s4`/`source_s5` paths; S4/S5 path agreement required |
| Prepared video | S5 `source_s1.prepared_video` (fallback: `{session}/preprocess/prepared_video.mp4`) |
| Digests | `project.file_digest.sha256_file`; validation memoized with `artifact_fingerprint` / `run_dir_fingerprint` |

Large prepared-video hashing is **off** by default (`hash_prepared_video=false`) so GUI-era callers can avoid hashing ~400 MB videos on the main thread.

---

## 4. Timing / index equations

| Quantity | Rule |
| --- | --- |
| Target grid | `time_sec[i] = t0 + i/120`, `t0 = prepared_time_sec[0]`, end ≤ last prepared time (S4/S5) |
| Mapping | S5 `map_target_nearest_source_frame` / `build_nearest_neighbor_timing_map_from_prepared_sync` |
| Exact match | `|t - prepared_time| ≤ 1e-9` → `is_exact_source_sample=True` |
| Tie-break | Equidistant left/right → **earlier (left)** prepared frame |
| Residual | `timing_residual_sec = time_sec - prepared_time_sec` (signed) |
| Out of range | Structural failure (no silent clamp) |
| Keys | `sample_idx == dense_frame_idx`; unique; sorted ascending |
| Prepared column | `prepared_frame_idx` ≡ modern `orig_frame_idx` (not interchangeable with dense index) |

Optional when S1 supports: `raw_decode_frame_idx`, `raw_pts_time_sec` mapped through the chosen prepared index.

---

## 5. Boundary behavior

- Inclusive S8 bout `[start_sample_idx, end_sample_idx]`.
- Seek aids: `prepared_frame_idx_start/end = sync_index.prepared_frame_idx[start/end]`.
- Scientific duration remains `bout_duration_frames / 120` from S8.
- Many-to-one dense→prepared is valid; prepared start may equal prepared end while dense length > 1.
- Invalid / out-of-domain bout indices → `BoutMappingError`.
- No re-collapse, merge, split, or emit filtering in Sprint 1.

---

## 6. Hand-computed synchronization examples

### 6.1 Nearer neighbor (not a tie)

Prepared times `{0.0, 0.1, 0.2}`; target `0.06` → frame **1** (closer to 0.1). Residual `0.06 - 0.1 = -0.04`.

### 6.2 Earlier-frame tie

Target `0.05` equidistant to `0.0` and `0.1` → frame **0**.

### 6.3 Repeated prepared frames / dense ≠ prepared

Prepared 5 frames @ 30 Hz → 17 dense samples @ 120 Hz. Multiple dense rows share one `prepared_frame_idx`. Sync length equals dense count, not prepared count.

### 6.4 Inclusive bout over shared prepared frame

Dense bout `[2, 4]` with prepared map `[…, 1, 1, 1, …]` → prepared start=end=`1`, while `bout_duration_frames` remains `3`.

---

## 7. E-session smoke-test results

Session: `test_cases/E_full_session_integration`  
S8 run: `behavioral_arbitration__20261028T133953`

| Check | Observed (run `behavioral_arbitration__20261028T133953`) |
| --- | --- |
| Dense samples | **36 528** (`0…36527`) |
| Sync rows | **36 528** (one per dense sample) |
| Prepared source frame count | **36 239** |
| Unique mapped prepared frames | **36 238** (`prepared_frame_idx` min **0**, max **36 237**) |
| Max many-to-one multiplicity | **2** |
| Exact matches (`≤1e-9`) | **290** |
| Residual min / max / abs max | ≈ `-0.004200` / `+0.004133` / `0.004200` s |
| Residual mean | ≈ `-3.35e-5` s |
| Dense ≠ prepared | Asserted (`36528 ≠ 36238`) |
| Bout membership | **609** (full S8 table) |
| Monotonic prepared map | Yes (non-decreasing) |
| Out-of-domain targets | None (build succeeds) |
| S8 historical status | `completed_with_warnings` (eligibility still current) |
| Upstream mutation | None (states/bouts/sync digests unchanged; no `results_package/` write) |

See `tests/results_packaging/test_e_session_sync_smoke.py`.

---

## 8. Approved export-profile decisions (recorded only)

| Decision | Value |
| --- | --- |
| Default package | Retains **all** S8 bouts (contact, engaged, short events) |
| `review_emit_v1` defaults | `include_contact=false`, `include_engaged=true`, floors `10` / `0.10` s |
| Seconds→frames | **`target_fps=120`** (deliberate difference from Pipeline 110 `source_video.fps`) |
| Profile effect on boundaries | **Forbidden** — selection/annotation only in later sprints |
| Sprint 1 | Profile **not implemented** |

---

## 9. Known limitations

- No immutable S9 run publication / discovery.
- No packaged `sync_index.parquet` write (in-memory only).
- No bout table stamping / ethogram summary / `bout_index` profiles.
- No GUI; no video playback; no S10.
- Prepared-video digest hashing optional and off by default.
- S6 map copy path (`copy_validated_s6_map`) not implemented; rebuild-from-prepared-sync only.

---

## 10. Sprint 2 prerequisites

1. Architectural review acceptance of Sprint 1 intake/sync behavior and E-session audit numbers.
2. Implement lossless `behavior_bouts.parquet` copy/stamp + prepared seek annotations.
3. Implement `ethogram_summary` + default `analysis_full` package writer.
4. Implement optional `review_emit_v1` **selection-only** filtered index using approved defaults.
5. Confirm remaining open items U-R3…U-R8 as needed for summaries / embed / media.

---

## 11. Tests and static checks (Sprint 1 gate)

| Suite | Command / result |
| --- | --- |
| S9 unit + E-smoke | `pytest tests/results_packaging` — 25 passed |
| S5 NN regression | `pytest tests/blob_features/test_timing_nn.py` |
| S8 discovery / E consistency | targeted `tests/behavioral_arbitration/...` |
| Ruff | `ruff check src/results_packaging tests/results_packaging` |
| Mypy | `mypy src/results_packaging` |
