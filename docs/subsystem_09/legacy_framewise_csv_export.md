# Legacy framewise CSV export (`legacy_framewise_v1`)

**Status:** Implemented for S9 automated and S10 published reviewed sources.  
**Non-goals:** Scientific algorithm changes; inventing original-frame indices; transforming pose into raw-video coordinates; exporting unpublished S10 drafts; mutating S4 pose.

## Profile

| Item | Value |
| --- | --- |
| Profile id | `legacy_framewise_v1` |
| Default for new CSV jobs | No — default is now `legacy_framewise_v2` (prepared-frame). v1 remains explicitly selectable. |
| Preserved prior profile | `package_bouts_automated` (bout table CSV); also see `legacy_framewise_csv_export_v2.md` |
| Output file | `results_exports/csv_export__{ts}/legacy_framewise_v1.csv` |
| Line endings | CRLF |
| Columns | Exactly 23 (see contract below) |

## Column → source mapping

| Column | Authority |
| --- | --- |
| `frame` | S9 `sync_index.raw_decode_frame_idx` (S1 `prepared_sync.raw_decode_frame_idx` via prepared NN map). **0-based** original-video decode index. |
| `dense_frame_idx` | S9 `sync_index.dense_frame_idx` (**0-based** contiguous dense grid). |
| `behaviour_label_focal_0` | Framewise exclusive `behavior_label` where `focal_id==0` (S8 automated or published S10 reviewed states). |
| `behaviour_label_focal_1` | Same for `focal_id==1`. |
| `pred_side_by_side` | Pair-level `pred_side_by_side` (0/1) from the same states table; directions must agree. |
| `headstage_0_x/y`, `neck_*`, `nose_*`, `spine_base_*`, `tail_base_*` | S4 `final_pose.parquet` (`sample_idx` ≡ dense), prepared-video pixel space. No `headstage_1_*` columns (matches reference). |

Missing pose coordinates and missing directional labels are emitted as **empty fields** (never fabricated / substituted).

## Domains (do not conflate)

- **Frame domain:** `frame` is an original-video raw-decode index (0-based). Nonzero trim offsets and many-to-one dense→original mappings are legitimate.
- **Coordinate domain:** pose `x`/`y` remain prepared-video pixels from S4. Export never maps them into raw-video coordinates.

## Coordinate formatting

S4 stores sub-pixel floats. The laboratory reference renders populated coordinates as integer-valued floats (`N.0`). Policy: round half-even to nearest integer, then format with one decimal place. This is a **formatting convention exclusive to `legacy_framewise_v1` CSV serialization** and must not alter S4 `final_pose` or other scientific stores.

## S4 pose source integrity

1. Prefer the exact S6 lineage pin (`combined_features` `run_meta.source_s4`).
2. When lineage is missing, require an **explicit** `pose_run_dir`.
3. Validate session root, sample timeline (`sample_idx` ≡ sync `dense_frame_idx`), and required track/node pairs.
4. Record selected S4 run id, path, digest, and `selection_mode` (`lineage_s6` | `explicit`).
5. Reject ambiguous, incompatible, or digest-mismatched sources.
6. **Never** fall back to “latest eligible S4”.

### GUI (S9 Results Packaging)

- Default: automatic exact-lineage resolution (status label only).
- When lineage is unavailable for `legacy_framewise_v1`, show a minimal S4 run
  picker listing completed S4 identities for the active session.
- Export is blocked until the operator explicitly selects a compatible run.
- The picker is hidden for `package_bouts_automated` and when lineage is present.
- Conflicting explicit selection vs recorded lineage is rejected with an
  actionable error. Selected run id + digest are written into the export
  manifest (`pose_source`).

## Dense-frame contract

Authoritative `sync_index.dense_frame_idx` must be unique, 0-based, and contiguous (`0..n-1`). Export emits exactly one row per sync sample. Gaps, duplicates, renumbering, or silent fills block the export.

## Eligibility

Export refuses when:

- `sync_index` lacks `raw_decode_frame_idx` or contains NaN mappings
- dense indices violate uniqueness/contiguity/0-based contract
- S8 / published S10 states or S4 `final_pose` cannot be resolved
- S4 lineage is missing and no explicit pose run is provided
- explicit pose disagrees with lineage, session, timeline, or schema
- S10 source is a draft / unpublished review
- behavior states contain dense indices outside the sync grid or conflicting directional labels

## Manifest metadata (not CSV columns)

`export_job_manifest.json` records profile name/version, source run identities, pinned digests, original-frame mapping provenance, pose selection mode + alignment, row count, dense continuity diagnostics, and format validation.
