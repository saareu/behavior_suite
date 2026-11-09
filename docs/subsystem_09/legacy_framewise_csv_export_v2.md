# Legacy framewise CSV export (`legacy_framewise_v2`)

**Status:** Implemented for S9 automated and S10 published reviewed sources.  
**Default:** New CSV export jobs use `legacy_framewise_v2`.  
**Preserved:** `legacy_framewise_v1` (dense-row / S4) and `package_bouts_automated` remain explicitly selectable. Historical jobs are never rewritten.

## Profile

| Item | Value |
| --- | --- |
| Profile id | `legacy_framewise_v2` |
| Semantics revision | `prepared_s3_pose_nearest_behavior_v1` |
| Row domain | S1 prepared-video frames (`0..n-1`) |
| Output CSV | `results_exports/csv_export__{ts}/legacy_framewise_v2.csv` |
| Alignment sidecar | `legacy_framewise_v2_frame_alignment.parquet` |
| Laboratory header | Exact same 23 columns / CRLF / coord formatting as v1 |

Historical V2 jobs that used S4 SyncMap pose remain immutable. New jobs record `semantics_revision` so S3-based exports are distinguishable.

## Column semantics (CSV)

| Column | Authority |
| --- | --- |
| `frame` | S1 `prepared_sync.raw_decode_frame_idx[prepared_frame_idx]` |
| `dense_frame_idx` | Behavioral prepared→nearest-dense sample (nullable / noncontiguous / repeated allowed) |
| Behavior labels / SBS | Selected behavioral dense sample only |
| Pose coordinates | Accepted S3 `tracked_pose` at the same `prepared_frame_idx` (prepared-video pixels; **not** S4 finalized pose) |

Prepared-frame identity is the row domain; it is not substituted into the `frame` column.

## Independent timing

- Behavior: package `sync_index` timestamps → prepared→nearest-dense (earlier dense sample on exact ties; no extrapolation outside dense temporal coverage).
- Pose: accepted S3 tracked pose joined directly on `prepared_frame_idx` (no resampling, gap fill, smoothing, or S4 substitution).
- Behavioral dense index and S3 prepared-frame pose are independent authorities.
- Fail closed when prepared universe, accepted S3 identity, or behavioral timing authority cannot be established.

## Missing coverage

Retain every prepared-frame row. Leave uncovered source fields empty. Never forward-fill, interpolate, or invent labels/coordinates. Genuine missing S3 keypoints remain empty.

## Sidecar (not CSV columns)

`prepared_frame_idx`, `raw_decode_frame_idx`, `behavior_dense_frame_idx`, `behavior_coverage`, `s3_pose_available` — one row per prepared frame.

**Sidecar schema migration:** historical V2 sidecars used `pose_dense_frame_idx` / `pose_coverage` for independent S4 SyncMap mapping. New jobs replace those with `s3_pose_available`. Historical files are never rewritten.

Filename, digest, mapping authorities, timing policies, S3 run identity / tracked-pose digest, and source provenance are recorded in `export_job_manifest.json`.
