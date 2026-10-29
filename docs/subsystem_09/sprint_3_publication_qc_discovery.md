# Subsystem 09 — Sprint 3: Immutable Publication, QC and Run Discovery

**Status:** Implemented (headless). Sprint 4 GUI follows in [sprint_4_gui_integration.md](sprint_4_gui_integration.md).  
**Date:** 2026-10-28  
**Scope:** Atomic S9 run publication, package manifest, structural/scientific QC, source revalidation, run discovery, and current downstream eligibility. **No** GUI, S10, clips/GIFs, or portable embed in this sprint.

Authoritative design: [architecture_audit.md](architecture_audit.md) (S9-A), [implementation_contract.md](implementation_contract.md) (S9-B), [sprint_1_intake_sync.md](sprint_1_intake_sync.md), [sprint_2_bouts_summaries_exports.md](sprint_2_bouts_summaries_exports.md).

---

## 1. Modules and entry points

| Module | Role |
| --- | --- |
| `publication.py` | Immutable publish lifecycle, revalidation, reopen, eligibility |
| `export.py` | Atomic parquet / JSON / YAML writers |
| `qc.py` | Structural blockers + scientific/informational QC |
| `manifest.py` | `package_manifest.json` + labeled fingerprint records |
| `run_discovery.py` | Latest eligible / completed listing / classifications |
| Sprint 1–2 modules | Unchanged science processing (reused) |

### Primary entry point

```python
from results_packaging import (
    ResultsPackagingRequest,
    ResultsPackagingConfig,
    ExecutionSettings,
    run_results_packaging_publication,
    find_latest_eligible_s9_run,
    is_s9_run_eligible_for_downstream,
    reopen_results_package,
)

result = run_results_packaging_publication(
    ResultsPackagingRequest(
        session_root=session_root,
        s8_run_dir=explicit_s8_or_None,
        config=ResultsPackagingConfig(
            execution=ExecutionSettings(
                selected_export_profiles=("review_emit_v1",),  # optional
                publish_run=True,
            )
        ),
    )
)
# result.run_dir, result.status, result.currently_eligible
```

Optional progress callbacks accept `preprocess.models.OperationProgress` (Sprint 4-ready; no science effect).

---

## 2. Publication lifecycle and failure semantics

Layout:

```text
{session_root}/results_package/results_package__YYYYMMDDTHHMMSS/
  sync_index.parquet
  behavior_bouts.parquet
  ethogram_summary.parquet
  package_manifest.json
  settings_used.yaml
  qc.json
  run_meta.json
  processing_log.txt
  export_profiles/<profile_id>/...   # optional only
```

Lifecycle:

1. Preflight validate current S8 + sync/video readiness; capture digests/fingerprints.
2. Create unique run dir (`exist_ok=False`; never overwrite).
3. Write `running` `run_meta` + `settings_used`.
4. Execute Sprint 1–2 packaging in memory.
5. Evaluate QC; structural failures abort before science write.
6. Revalidate source identities (reject mid-run mutation / replacement).
7. Atomically write science tables + QC + optional profiles + manifest.
8. Persist terminal `completed` / `completed_with_warnings` / `failed`.

On failure:

- Delete science outputs (`sync_index`, `behavior_bouts`, `ethogram_summary`, `package_manifest`, `qc`, `export_profiles/`).
- Preserve `run_meta.json`, `settings_used.yaml`, `processing_log.txt` with `status=failed` and `science_written=false`.
- Partial science is never discoverable as eligible.

Protected overlap rejection prevents nesting under S1–S8 trees.

---

## 3. Manifest and fingerprint semantics (U-R6)

`package_manifest.json` records:

- Package / schema versions and `package_run_id`
- Exact source S8 run + S7/S6 lineage
- Embedded science artifact paths + **SHA256 content hashes**
- External references (S8 states/bouts, prepared sync/video) with path + digest/fingerprint
- Timeline properties, row counts, full-bout preservation flag
- Selected optional export profiles
- Explicit note that `reference_lightweight` is **not** self-contained (U-R3)

Fingerprint policy:

| Kind | `fingerprint_type` | `verification_strength` | Meaning |
| --- | --- | --- | --- |
| Trusted/computed SHA256 | `sha256` | `content_hash` | Authoritative content identity |
| Size/mtime only | `size_mtime` | `size_mtime_fingerprint` | Replacement heuristic; **not** SHA256 |
| Lifecycle records | `size_mtime` | `lifecycle_path` | `run_meta` / `settings_used` / manifest / log — rewritten at finalize; not used as science digests |

Optional `execution.hash_prepared_video=true` computes full video SHA256 off the publication path. Default reuses trusted recorded digests when present; otherwise records labeled size/mtime. GUI polls must not imply expensive hashing.

---

## 4. QC classifications

| Class | Effect |
| --- | --- |
| Structural blockers | Fail publish / ineligible |
| Nonblocking warnings | `completed_with_warnings` (still eligible if sources remain valid) |
| Informational | Configuration / many-to-one notes; not status-driving alone |

Structural examples: ineligible/stale S8, broken lineage, missing sync/video, invalid sync cardinality, duplicate keys, nonmonotonic prepared map, bout count/key mismatch, authoritative field drift, missing required artifacts.

Scientific/informational examples: channel counts, short-bout retention counts, engaged/undefined coverage, reconstruction mismatch counts (U-R8), many-to-one mapping, optional profile omission accounting, inherited S8 warnings.

Reconstruction policy (U-R8): report counts; warn on limited mismatches; hard-fail only when all checked exclusive bouts disagree with referenced states (wrong-source signal). No new biological thresholds.

---

## 5. Discovery and eligibility APIs

| API | Behavior |
| --- | --- |
| `summarize_results_package_project` | All runs + classifications |
| `find_latest_eligible_s9_run` | Newest eligible / completed_with_warnings classification |
| `list_completed_s9_runs` | Historical completed(+warnings), may include currently stale |
| `validate_s9_run_for_downstream` | Memoized digest/fingerprint validation |
| `is_s9_run_eligible_for_downstream` | Boolean wrapper |
| `reopen_results_package` | Load existing package without recompute |

Classifications: `eligible`, `completed_with_warnings`, `stale_source`, `failed_or_incomplete`, `missing_required_artifacts`.

**Historical `run_meta.status` is never rewritten** when upstream files later become stale. A completed package with broken external references is **not** currently eligible for S10.

---

## 6. Resolved policy decisions (Sprint 3)

| ID | Resolution |
| --- | --- |
| U-R3 | Portable self-contained packages deferred beyond MVP |
| U-R4 | Clips/GIFs deferred; S10 lazy prepared-video access |
| U-R5 | Retain namespaced S8 bout identity (Sprint 2) |
| U-R6 | Reuse trusted hashes; labeled size/mtime when unavailable; optional video SHA256 never implicit |
| U-R7 | No additional symmetric collapse in S9 MVP |
| U-R8 | Reconstruction counts + warnings; no arbitrary quality thresholds |

---

## 7. E-session publication evidence

Source session: `E_full_session_integration`  
Source S8: `behavioral_arbitration__20261028T133953`  
Published S9 (representative): `results_package__20261028T181714`

| Check | Result |
| --- | --- |
| Sync rows | 36 528 |
| Authoritative bouts | 609 (538 exclusive, 60 contact, 11 SBS) |
| Dense boundaries / durations | Preserved vs S8 (id-aligned) |
| Prepared navigation fields | Present; seek aids only |
| `review_emit_v1` | 514 included / 95 excluded |
| Status | `completed_with_warnings` (structural_ok=true) |
| Current eligibility | True at publish; reopen succeeds |
| Upstream S8 mutation | None |

---

## 8. Tests and static checks

Targeted Sprint 3 tests cover: successful publish, unique run dirs, failure cleanup, mid-processing source change, missing video, fingerprint semantics, profile isolation, historical status vs staleness, reopen, determinism, no S1–S8 writes, and E-session publication.

Run:

```text
pytest tests/results_packaging/ -q
ruff check src/results_packaging tests/results_packaging
mypy src/results_packaging
```

Plus relevant S8/S5 regressions as available in the suite.

---

## 9. Sprint 4 prerequisites

1. Group 4 GUI page for packaging settings, progress, and discovery/reopen.
2. Wire `GuiTaskKind` / background workers; never hash prepared video on poll timers.
3. Surface QC warnings vs informational notes distinctly.
4. Laboratory checklist on E-session package open within interactive budgets.
5. No S10 ethogram viewer / video playback / editing in S9 GUI.

---

_End of Sprint 3 documentation. Stop for architectural review before Sprint 4._
