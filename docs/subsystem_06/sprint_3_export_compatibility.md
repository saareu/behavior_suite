# Subsystem 06 — Sprint 3: Export, QC, Model Compatibility, Run Discovery

**Status:** Implemented (pending architectural review)  
**Date:** 2026-10-21  
**Depends on:** Sprint 1 (joint eligibility + pose adapter), Sprint 2 (FF + Pipeline 50 in-memory)

## Objective

Deliver reproducible, immutable S6 run artifacts with structural QC, upstream
provenance, model compatibility reports, and completed-run discovery. No GUI,
inference, or training.

## Sprint 2 review resolutions

### Unknown implantation → NaN

| State | `is_implanted` |
| --- | --- |
| Resolved, focal == implanted track | `1.0` |
| Resolved, focal != implanted track | `0.0` (confirmed non-implanted) |
| Unresolved / absent / ambiguous | `NaN` |

Uncertainty is never encoded as `0`. QC records `implantation_status`,
`implanted_track_id`, and reason. Headstage-dependent geometry still does not
fabricate observations when evidence is missing.

### Mandatory vs optional S5 blob columns

| Class | Columns |
| --- | --- |
| **Mandatory** | `cc_count`, `cc_count_raw`, `min_inter_contour_dist`, all S5 occlusion static + temporal columns |
| **Optional** | empty today (documented so callers do not invent fills) |

Missing mandatory columns → hard `CombinedFeaturesError`. Legitimate measurement
NaNs inside present columns are preserved.

## Artifacts

```text
{session_root}/combined_features/combined_features__YYYYMMDDTHHMMSS/
  dyadic_features_full.parquet
  run_meta.json
  settings_used.yaml
  processing_log.txt
  qc.json
  model_compat.json
```

Publication is atomic for the parquet (write `.tmp` then replace). On failure the
parquet / qc / model_compat files are removed and status is `failed`. Historical
completion status is never rewritten when upstream later becomes stale.

## Runner lifecycle

1. Resolve and validate the selected S4/S5 pair (or default newest eligible pair).
2. Create immutable run directory + `settings_used.yaml` snapshot (`status=running`).
3. Compute complete in-memory features (`compute_combined_features_from_pair`).
4. Stamp identity columns and evaluate structural QC.
5. Build `model_compat.json` (no inference; registry may be unavailable).
6. Revalidate joint S4/S5 digests immediately before publish.
7. Write parquet + qc + model_compat; set `completed` or `completed_with_warnings`.

Scientific computation remains separate from file publication.

## QC design

Structural errors (block publication):

- Required identity columns and order
- Unique `(focal_id, partner_id, dense_frame_idx)`
- Expected `K × (K−1)` directional rows per sample
- Matching `sample_idx` / `dense_frame_idx`
- Presence of mandatory S5 blob columns on the output table

Ordinary warnings (do **not** auto-ineligible a structurally valid run):

- Feature-family missingness / Inf counts
- Unresolved implantation (NaN policy)
- Computation warnings from the adapter / Pipeline 50 path

No arbitrary biological quality thresholds.

## Model registry and compatibility

Configured via `model_registry.root` (YAML / `CombinedFeaturesConfig`). The
git-ignored `legacy/` tree is **never** required at runtime unless explicitly
pointed at.

Discovery:

- `{root}/dyadic/v1/*/feature_spec.json`
- `{root}/focal/v1/*/feature_spec.json` (when `include_focal`)

Per detector the report records required feature order, availability, missing
columns, dtype warnings, model-file presence (not loaded), and status:

`compatible | incompatible | unsupported | invalid_spec | bundle_incomplete`

Obsolete standalone `contact` is always `unsupported`. Missing registry →
`registry_status=unavailable`; the complete feature table is still generated.

## Run discovery and staleness

APIs:

- `summarize_combined_features_project`
- `is_s6_run_eligible_for_downstream` / `validate_s6_run_for_downstream`
- `find_latest_eligible_s6_run`

Classifications mirror S5: `eligible`, `completed_with_warnings`, `stale_source`,
`failed_or_incomplete`, `missing_required_artifacts`. Eligibility rechecks the
recorded S4 final_pose digest, S5 blob digest, and prepared_sync SHA against the
live joint-eligible pair. Stale runs are never selected.

## Modules added / updated

| Module | Role |
| --- | --- |
| `contracts.py` | Artifact names, mandatory blob cols, discovery classes |
| `config.py` | Settings + model registry |
| `export.py` | Identity stamp, ordered columns, atomic parquet |
| `qc.py` | Structural vs warning QC |
| `model_compat.py` | Registry discovery + feature_spec checker |
| `runner.py` | Headless lifecycle + downstream validation |
| `run_discovery.py` | Project summary + latest eligible helper |
| `pipeline50_post.py` | NaN implantation; mandatory blob attach |
| `configs/subsystem_06/combined_features_default.yaml` | Default settings |

## Tests

Covered under `tests/combined_features/`:

- Unknown vs confirmed non-implanted
- Mandatory S5 columns / preserved measurement NaNs
- Output schema + directional uniqueness
- Atomic publication; failed run without eligible partial parquet
- Source digests; stale-run detection; completed-with-warnings eligibility
- Model registry / dyadic+focal specs / unsupported contact / dtype gaps
- Deterministic republish; synthetic headless E2E

## GUI prerequisites (Sprint 4)

- Group 2 Feature Processing page: select/confirm S4+S5, show alignment
- Run / reopen S6 runs; surface QC + model_compat summaries
- Configure model registry root in the UI
- No inference UI

## Deviations

| Topic | Stance |
| --- | --- |
| `is_implanted` unknown | NaN (Sprint 2 review), not `0` |
| Optional blob columns | Explicitly empty tuple; no silent manufacture |
| Model binaries | Presence-only; never packaged by this sprint |
| Numerical regression vs historic caches | Out of scope |
