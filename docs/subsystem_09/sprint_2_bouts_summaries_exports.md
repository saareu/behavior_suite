# Subsystem 09 — Sprint 2: Bout Packaging, Summaries and Export Profiles

**Status:** Implemented (headless). Stop for architectural review before Sprint 3.  
**Date:** 2026-10-28  
**Scope:** Lossless annotated S8 bout packaging, ethogram summaries, optional `review_emit_v1` selection view, extended in-memory Sprint 2 result. **No** immutable S9 publication, GUI, or S10.

Authoritative design: [architecture_audit.md](architecture_audit.md) (S9-A), [implementation_contract.md](implementation_contract.md) (S9-B), [sprint_1_intake_sync.md](sprint_1_intake_sync.md).

---

## 1. Modules and entry points

| Module | Role |
| --- | --- |
| `bouts.py` | Lossless package bout table + stable IDs + prepared navigation |
| `ethogram_summary.py` | Cached `label_time` / `bout_counts` / `parallel_overlap` / `session_timing` |
| `export_profiles.py` | Optional `review_emit_v1` selection-only bout index |
| `runner.py` | `run_results_packaging_sprint2` / `run_sprint2_from_session` |
| `results.py` | `Sprint2PackagingResult` |
| Sprint 1 modules | Unchanged intake / sync / bout-boundary mapping (reused) |

### Primary entry point

```python
from results_packaging import (
    ResultsPackagingRequest,
    ResultsPackagingConfig,
    ExecutionSettings,
    run_results_packaging_sprint2,
)

result = run_results_packaging_sprint2(
    ResultsPackagingRequest(
        session_root=session_root,
        s8_run_dir=explicit_s8_or_None,
        # Optional: explicitly select review_emit_v1
        config=ResultsPackagingConfig(
            execution=ExecutionSettings(
                selected_export_profiles=("review_emit_v1",),
            )
        ),
    )
)
# result.sync.sync_index
# result.behavior_bouts          # full lossless package table
# result.ethogram_summary
# result.export_profiles.get("review_emit_v1")  # only if selected
# result.to_metadata()["published"] == False
```

Default config leaves `selected_export_profiles` empty — `review_emit_v1` is **never** auto-applied.

---

## 2. Full bout schema and identity

### 2.1 Lossless copy semantics

`behavior_bouts` (in-memory; parquet name reserved for Sprint 3) is a **lossless annotated copy** of the complete S8 bout table:

- Every S8 `BOUT_COLUMNS` science field is preserved.
- No re-collapse, merge, split, emit filtering, or duration rewrite.
- Row-count invariant: `len(package) == len(s8)`.
- Multiset equality on authoritative keys:
  `(source_s8_bout_id, channel, behavior_name, start_sample_idx, end_sample_idx, identity_grain, focal_id, partner_id, mouse_a_id, mouse_b_id)`.

### 2.2 Identity (U-R5 resolved for Sprint 2)

| Field | Rule |
| --- | --- |
| `source_s8_bout_id` | Exact S8 `bout_id` (`bout_{i:08d}` within the published S8 run) |
| `bout_id` (package) | `{source_s8_run_id}__{source_s8_bout_id}` |
| Content key (audit) | `authoritative_bout_content_key`: distinguishes directional `f→p` from symmetric `pair:a\|b` without depending on DataFrame row order |

Package ids are globally unambiguous within a package, stable across repeated packaging of the same S8 source, and independent of incidental input row order (table is stably sorted by `channel`, `start_sample_idx`, `behavior_name`, `pair_id`, `focal_id`, `partner_id`).

### 2.3 Package stamps

| Column | Value |
| --- | --- |
| `schema_version` | `results_package_bouts_v1` |
| `s8_bouts_schema_version` | Copied S8 stamp (`behavioral_arbitration_bouts_v1`) |
| `source_s8_run_id` | S8 run id |
| `package_run_id` | Optional Sprint 2 probe id (publication reserved for Sprint 3) |

---

## 3. Prepared-frame navigation fields

Added as **seek aids only** (do not redefine scientific duration):

| Column | Derivation |
| --- | --- |
| `prepared_frame_idx_start` | `sync_index.prepared_frame_idx[start_sample_idx]` |
| `prepared_frame_idx_end` | `sync_index.prepared_frame_idx[end_sample_idx]` |
| `prepared_time_sec_start` / `_end` | Mapped prepared times at those samples |

Guarantees:

- Dense `start_sample_idx` / `end_sample_idx` remain authoritative.
- Inclusive endpoints and `bout_duration_frames` / `bout_duration_sec` unchanged.
- Repeated prepared-frame mappings allowed (many-to-one).
- Prepared start may equal prepared end while dense duration > 1.
- Timing residuals remain on the sync index (not discarded/rounded away).
- Video seek clock (`prepared_*`) stays distinct from behavioral clock (`time_sec` @ 120 Hz).

---

## 4. Ethogram summary

In-memory table matching future `ethogram_summary.parquet` (`results_package_ethogram_summary_v1`).

Long-form grain: one row per
`(summary_kind, channel, behavior_name, identity_key, metric_name)`.

| `summary_kind` | Content |
| --- | --- |
| `label_time` | Exclusive `behavior_label` frame counts and fractions of **directional state rows**; explicit engaged/undefined metrics |
| `bout_counts` | Per-channel/behavior bout counts, duration sums, inclusive frame-duration histogram buckets |
| `parallel_overlap` | Contact/SBS bout counts; framewise `pred_*` positives from states when available; S8 QC echo when present |
| `session_timing` | `n_samples`, `t0`/`t1`, `fps=120`, unique prepared-frame count, dense≠prepared flag |

Denominator policy:

- Exclusive label proportions use `n_directional_state_rows` (= `len(states)`).
- Parallel coverage is **not** mutually exclusive with ethogram labels.
- Framewise stats are **not** inferred from bouts alone when states are unavailable (`label_time_unavailable` / `framewise_parallel_unavailable` markers).

---

## 5. Optional export profile `review_emit_v1`

Selection-only derived view. Never mutates the complete package table.

### 5.1 Defaults (U-R1/U-R2; apply only when explicitly selected)

| Parameter | Default |
| --- | --- |
| `include_contact` | `false` |
| `include_engaged` | `true` |
| `include_side_by_side` | `true` |
| `min_emit_bout_frames` | `10` |
| `min_emit_bout_seconds` | `0.10` |
| `target_fps` | `120.0` (fixed) |

### 5.2 Duration selection

```text
min_required_frames = max(min_emit_bout_frames, ceil(min_emit_bout_seconds * 120))
keep iff bout_duration_frames >= min_required_frames AND channel/behavior policy passes
```

With defaults: `max(10, ceil(12.0)) = 12`.

Deliberate difference from Pipeline 110 (D3): seconds use **120 Hz**, not `source_video.fps`. Source-video FPS parameters are **rejected**. Profile is **not** labeled numerically identical to Pipeline 110.

### 5.3 Exclusion reason semantics

Primary `excluded_reason` = first match in priority order:

1. `channel_contact_excluded`
2. `channel_side_by_side_excluded`
3. `behavior_engaged_excluded`
4. `behavior_not_allowed`
5. `channel_not_allowed`
6. `below_emit_floor`

All matching reasons are also recorded pipe-separated in `matched_reasons`. Absent `bout_index` rows are excluded; inclusion is implied by presence. `source_s8_bout_id` / package `bout_id` are retained on included rows.

---

## 6. E-session validation findings

Session: `test_cases/E_full_session_integration`  
S8 run: `behavioral_arbitration__20261028T133953`

| Check | Observed |
| --- | --- |
| Sync rows | **36 528** |
| Authoritative S8 / packaged bouts | **609** |
| Exclusive / contact / SBS | **538 / 60 / 11** |
| Exclusive duration &lt; 10 (retained in full package) | **31** |
| Dense endpoints / durations vs S8 | Full-table match (lossless) |
| Navigation annotations | Valid; within prepared bounds |
| Upstream mutation | None (states/bouts/sync digests unchanged; no `results_package/` write) |

### Measured `review_emit_v1` (defaults; explicitly selected)

| Metric | Measured |
| --- | --- |
| `min_required_frames` | **12** |
| Included | **514** |
| Excluded | **95** |
| `channel_contact_excluded` | **60** |
| `below_emit_floor` | **35** |
| Contact in review index | **0** |

The 31 exclusive bouts shorter than 10 frames remain in the complete package; they are absent from the review index under the 12-frame floor (along with additional exclusive bouts of length 10–11).

---

## 7. Remaining U-R decisions (Sprint 2 review)

| ID | Original meaning | Sprint 2 needs resolution? | Recommendation / status |
| --- | --- | --- | --- |
| **U-R3** | Ship `portable_embed_v1` in MVP or post-MVP? | **No** for Sprint 2 | Keep **post-MVP**. Reference-lightweight mode covers lab packaging; embed belongs with Sprint 3 portability if needed. |
| **U-R4** | Clips/GIFs in S9 vs S10 lazy | **No** | Prefer **S10 lazy** for MVP; do not block Sprint 3 publication on media. |
| **U-R5** | Exact package-local `bout_id` scheme | **Yes — resolved in Sprint 2** | Preserve S8 `bout_id` as `source_s8_bout_id`; package `bout_id = {s8_run_id}__{source_s8_bout_id}`. |
| **U-R6** | Require prepared-video SHA256 at package time always? | **No** (Sprint 3 publish concern) | Hash once at publish when practical; memoize; warn if only path known. Sprint 2 keeps `hash_prepared_video=false` default. |
| **U-R7** | Presentation symmetric collapse profile in MVP? | **No** | Keep optional / later. Default package remains directional S8 truth. |
| **U-R8** | Hard-fail vs warn on limited bout↔state reconstruction mismatches | **No** for Sprint 2 packaging | Recommend **warn + counts** unless mismatch rate implies wrong S8 source (hard fail). Full reconstruction audit can land with Sprint 3 QC. |

No unresolved item blocked Sprint 2. No new scientific thresholds were introduced.

---

## 8. Sprint 3 publication prerequisites

1. Architectural review acceptance of Sprint 2 bout/summary/profile behavior and E-session measured counts.
2. Atomic run directory lifecycle (`running` → validate → `completed` / warnings / `failed`).
3. Write `sync_index.parquet`, `behavior_bouts.parquet`, `ethogram_summary.parquet`, manifests, `qc.json`, `run_meta.json`, `settings_used.yaml`, `processing_log.txt`.
4. Optional profile artifacts under `export_profiles/<profile_id>/`.
5. Discovery classifications + digest memoization; stale detection when S8/video/sync digests change.
6. Decide U-R6 hashing policy at publish time; optionally portable embed (U-R3) behind explicit flag.
7. No GUI (Sprint 4); no S10; no media required.

---

## 9. Tests and static checks (Sprint 2 gate)

| Suite | Result |
| --- | --- |
| `pytest tests/results_packaging` | **50 passed** |
| S8 bouts + export/QC discovery + S5 NN | **21 passed** |
| Ruff `src/results_packaging tests/results_packaging` | Clean |
| Mypy `src/results_packaging` | Clean |

---

_End of S9 Sprint 2 documentation. Stop for architectural review before Sprint 3._
