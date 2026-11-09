# Legacy framewise CSV export (`legacy_framewise_v2`)

**Status:** Implemented for S9 automated and S10 published reviewed sources.  
**Default:** New CSV export jobs use `legacy_framewise_v2`.  
**Preserved:** `legacy_framewise_v1` (dense-row) and `package_bouts_automated` remain explicitly selectable. Historical jobs are never rewritten.

## Profile

| Item | Value |
| --- | --- |
| Profile id | `legacy_framewise_v2` |
| Row domain | S1 prepared-video frames (`0..n-1`) |
| Output CSV | `results_exports/csv_export__{ts}/legacy_framewise_v2.csv` |
| Alignment sidecar | `legacy_framewise_v2_frame_alignment.parquet` |
| Laboratory header | Exact same 23 columns / CRLF / coord formatting as v1 |

## Column semantics (CSV)

| Column | Authority |
| --- | --- |
| `frame` | S1 `prepared_sync.raw_decode_frame_idx[prepared_frame_idx]` |
| `dense_frame_idx` | Behavioral SyncMap + `LAST_IN_PLATEAU` (nullable / noncontiguous / repeated allowed) |
| Behavior labels / SBS | Selected behavioral dense sample only |
| Pose coordinates | Independently selected S4 SyncMap + `LAST_IN_PLATEAU` sample |

Prepared-frame identity is the row domain; it is not substituted into the `frame` column.

## Independent timing

- Behavior: package `sync_index` → `SyncMap` → overlay `LAST_IN_PLATEAU`.
- Pose: S1 `prepared_sync` + S4 dense timeline → independent SyncMap → `LAST_IN_PLATEAU`.
- Behavioral and S4 dense indices are never assumed equal.
- Fail closed when prepared universe or either timing authority cannot be established.

## Missing coverage

Retain every prepared-frame row. Leave uncovered source fields empty. Never forward-fill, interpolate, or invent labels/coordinates.

## Sidecar (not CSV columns)

`prepared_frame_idx`, `raw_decode_frame_idx`, `behavior_dense_frame_idx`, `pose_dense_frame_idx`, `behavior_coverage`, `pose_coverage` — one row per prepared frame. Filename, digest, mapping authorities, timing policies, and source provenance are recorded in `export_job_manifest.json`.
