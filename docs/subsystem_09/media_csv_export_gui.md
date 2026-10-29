# S9 Media / CSV Export — GUI Integration (S9-E3)

**Status:** Implemented for architectural / operator review. **Not lab-validated** until the operator completes interactive acceptance.

**Scope:** Group 4 S9 page gains distinct **Export CSV** and **Export Bout Videos** actions over existing eligible packages, plus export-job discovery and incomplete media retry. Core packaging MVP behavior is preserved.

## 1. Operator actions

### Export CSV

1. Open an E-session / project on the S9 page; wait for discovery.
2. Select a **currently eligible** completed S9 package in the packages table.
3. Choose selection: all package bouts, channels/behaviors, or an existing packaged profile (`review_emit_v1` when present).
4. Confirm preview text shows selected row count and automated provenance (`export_provenance=automated`, source package id, selection mode).
5. Click **Export CSV**. The UI must accept immediately; validation and write run in `GuiTaskRunner` (`RESULTS_CSV_EXPORT`).
6. On success, note output under `{session}/results_exports/csv_export__…` and status in the export jobs table.

### Export Bout Videos

1. Select a currently eligible S9 package.
2. Choose selection (prefer a profile/channel subset for smoke tests — do **not** regenerate all 609 clips casually).
3. Set `pre_context_sec` / `post_context_sec` (default **0**).
4. Confirm expected clip count in the preview.
5. Click **Export Bout Videos** (`RESULTS_MEDIA_EXPORT`). Progress shows completed/total bout units when the worker reports them — no fabricated percentages.
6. Review succeeded/failed counts; open the export folder for clips + `clip_index.parquet`.

### Incomplete media retry

1. Select an export job with historical status `incomplete` or `failed`.
2. Ensure the source S9 package is **still currently eligible**.
3. Click **Retry incomplete media job**. Same-job resume must not overwrite successful clips or completed jobs.
4. Distinguish incomplete / failed / completed in the export jobs table (status + classification columns).

## 2. Eligibility policy (MVP)

| Action | Policy |
| --- | --- |
| New CSV / media export | Requires currently eligible S9 package |
| Media retry | Same eligibility gate; digest match vs job snapshot |
| Stale package selected | Export buttons blocked with explicit reason; no silent substitution |
| Historical jobs | Always discoverable/reopenable for inspection |
| Archival export from stale packages | Deferred |
| S9 package status on disk | Never rewritten |

## 3. Task safety

- Packaging and export share the long-running task runner; concurrent packaging/export is rejected.
- Settings, S8 source, S9 package selection, and export controls are disabled while a task runs.
- Expensive source validation and video work run on the worker thread only.
- Soft navigation away mid-job is allowed; session switch while busy is blocked.
- Stale callbacks after session/package change are ignored via `ExportRequestIdentity`.

## 4. Laboratory smoke (operator)

Package under test: `results_package__20261028T181714`.

**Environmental note (S9-E4, 2026-10-29):** eligibility restored. Recorded S7 paths under `legacy/models/...` resolve again via a local junction to validated `models/` artifacts (SHA-256 match). Package is currently eligible; new CSV/media exports are unblocked. Prior S9-E2/E4 jobs under `results_exports/` remain inspectable.

Operator interactive verification (still required — see laboratory checklist):

- [ ] Full automated CSV: 609 rows + provenance columns
- [ ] Representative MP4 selection (not all 609) with correct clip bounds and source playback FPS
- [ ] Completed-job discovery + reopen
- [ ] CSV and media provenance
- [ ] No modification of the source S9 package
- [ ] Ordinary S9 packaging still works when an eligible S8 source exists
- [ ] Retry/error presentation for incomplete media jobs

**Do not mark the extension lab-validated until the operator completes interactive acceptance.**

## 5. Non-goals (retained)

No video playback, ethogram editing, reviewed CSV, GIF batch, or S10 UI in this slice.
