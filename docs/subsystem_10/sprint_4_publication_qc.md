# Subsystem 10 — Sprint 4: Reviewed Publication, QC and Eligibility

**Status:** Implemented (headless publication + discovery). Stop for architectural review before Sprint 5.  
**Date:** 2026-10-29  
**Scope:** Atomic immutable `behavioral_review__*` publication, structural QC, run discovery, current eligibility, supersession classification, automated vs reviewed source-type metadata.  
**Non-scope:** Full Qt GUI, reviewed CSV/media exports, reciprocal correction tools, biological acceptance thresholds, S1–S9 science mutation.

Deferred MVP items remain frozen: **U2** reciprocal tools; **U3** `uncertain`; **U6** reviewed exports; **U12** stale-parent publish block (enforced); **U15** reject status-only.

---

## 1. Publication entry point

Package: `src/behavioral_review/` (extends Sprint 1–3).

| Module | Role |
| --- | --- |
| `publication.py` | `publish_reviewed_run` / `publish_working_draft` — atomic publish lifecycle |
| `snapshot.py` | Cursor-resolved active-event replay + stamped reviewed tables |
| `eligibility.py` | Publish gates (parent pins, journal, lock, reviewer) |
| `qc.py` | Structural blockers + nonblocking review summaries |
| `discovery.py` | Discovery, `validate_s10_run_for_downstream`, reopen, supersession |

### Primary entry points

```text
session.publish_reviewed_run()                 # EditingSession with open draft + lock
publish_working_draft(draft, reviewer_id=...)  # headless draft path
validate_s10_run_for_downstream(run_dir)
reopen_reviewed_run(run_dir)
summarize_behavioral_review_project(review_root=...)
find_latest_eligible_reviewed_run(...)
classify_analysis_source("automated"|"reviewed")
```

Publication requires:

1. Explicit draft identity (`behavioral_review__*`)
2. Valid reviewer identity + review metadata
3. Recoverable, internally consistent event journal
4. Valid active undo/redo cursor
5. Exact eligible S9 parent with matching recorded digests
6. Held writer lock (no unresolved conflict)
7. No malformed corrections / review decisions

**Parent stale → block publish (U12).** No silent substitution of a newer S9 run or source video. Full bout acceptance is **not** required.

---

## 2. Snapshot and cursor semantics

At publication:

1. Acquire / require draft writer lock  
2. Validate parent eligibility + pins  
3. Resolve active events at the saved cursor (`active_events()`, excluding superseded redo-side seqs)  
4. Replay corrections onto immutable automated baseline states  
5. Regenerate reviewed bouts via pure S8 `extract_behavior_bouts` (U11)  
6. Resolve review statuses / voided decisions  
7. Produce a consistent snapshot  
8. Revalidate source identity  
9. Publish immutable outputs  

| Store | Contents |
| --- | --- |
| `correction_events.parquet` | **Applied** events only (`journal` active at cursor) |
| `correction_events_history.parquet` | **Full** append-only history with `applied` / `superseded` / `beyond_cursor` flags |
| Working `correction_events.jsonl` | Retained unchanged for audit / recovery |

Undone corrections beyond the cursor and superseded redo-side events are **not** included in reviewed states/bouts.

---

## 3. Reviewed artifact schemas

Published under the existing draft run directory (no overwrite of prior distinct runs):

```text
behavioral_review__YYYYMMDDTHHMMSS/
  review_manifest.json
  settings_used.yaml
  run_meta.json
  processing_log.txt
  qc.json
  correction_events.parquet
  correction_events_history.parquet
  reviewed_behavior_states.parquet
  reviewed_behavior_bouts.parquet
  review_bout_status.parquet
  reviewer_notes.parquet          # optional
  working/                        # sealed but retained
```

**Schema stamps (S10-B §7.1):**

| Artifact | Stamp |
| --- | --- |
| Manifest | `behavioral_review_manifest_v1` |
| Correction events | `behavioral_review_correction_events_v1` |
| History | `behavioral_review_correction_events_history_v1` |
| Reviewed states | `behavioral_review_states_v1` |
| Reviewed bouts | `behavioral_review_bouts_v1` |
| Bout status | `behavioral_review_bout_status_v1` |
| QC | `behavioral_review_qc_v1` |
| Run meta | `behavioral_review_run_meta_v1` |
| Settings | `behavioral_review_settings_v1` |

Preserved scientific grain: dense sample/time, ordered focal/partner identity, exclusive labels, independent contact/SBS, inclusive bout endpoints (`duration = frames/120`), original provenance links, reviewer identity + correction history.

Do **not** rerun S8 arbitration or apply `review_emit_v1` floors to scientific reviewed bouts.

---

## 4. Review-status handling

Statuses: `unreviewed` | `accepted` | `rejected` | `voided` (no `uncertain`).

- Rejection is **status-only** (labels unchanged).  
- Geometry-changing edits void affected earlier decisions (`void_reason=geometry_changed`) without transferring acceptance.  
- Published status table reports `original_status` and `effective_status` plus JSON interval keys.  
- Unchanged reviewed intervals may retain valid decisions as in Sprint 3 / S10-B §5.4.

---

## 5. QC

### Structural blockers (fail publish)

- Source identity / eligibility / pin drift (U12)  
- Journal integrity + active cursor consistency  
- Directional state cardinality / uniqueness  
- Valid exclusive label vocabulary  
- Contact/SBS binary semantics  
- Valid reviewed bout identifiers  
- Bout↔state reconstruction via pure collapse  
- Inclusive frame counts / durations  
- Review-status referential integrity  
- No upstream source modifications  
- Artifact write + digest integrity  

### Nonblocking warnings

- Parent `completed_with_warnings`  
- High unreviewed fraction  
- Voided statuses after edits  
- Optional S7 unavailable  

No biological acceptance thresholds. No mandatory 100% bout acceptance.

---

## 6. Publication lifecycle

Mirrors S5–S9 atomic conventions:

1. Validate draft + parent  
2. Write `settings_used.yaml` + `run_meta` (`running`)  
3. Build snapshot; write science parquet/json  
4. Structural QC; revalidate parent  
5. Seal `qc.json` + `review_manifest.json`  
6. Promote `run_meta.status` → `completed` / `completed_with_warnings`  
7. On failure: delete science artifacts; retain `run_meta` / log / settings; restore working draft to recoverable `draft` state  

Unique immutable run IDs; no overwrite of prior reviewed runs; no half-published science eligible for downstream analysis.

---

## 7. Discovery and eligibility

| API | Behavior |
| --- | --- |
| `summarize_behavioral_review_project` | Classify runs under a review root |
| `list_completed_reviewed_runs` | Historical completed (may be stale) |
| `find_latest_eligible_reviewed_run` | Newest currently eligible (optional parent filter) |
| `validate_s10_run_for_downstream` | Explicit-path eligibility; never rewrites historical status |
| `reopen_reviewed_run` | Load immutable artifacts for an explicit path |
| `classify_analysis_source` | `automated` ↔ S9; `reviewed` ↔ S10 |

### Classifications

| Class | Meaning |
| --- | --- |
| `eligible` / `completed_with_warnings` | Published; parent pins still match |
| `stale_parent` | Review intact; parent drifted |
| `draft` | Working review not published |
| `failed_or_incomplete` | Publish failed / incomplete |
| `superseded` | Newer currently eligible publish shares parent lineage (informational) |
| `missing_required_artifacts` | Core files absent |

**Supersession:** publishing a new version does **not** rewrite an old immutable run. Discovery may classify older runs as `superseded` and exclude them from `eligible_runs` / latest selection, but an explicitly requested older path still validates and reopens when parent pins match.

**Automated vs reviewed:** downstream consumers must deliberately select `source_type=automated` (S9) or `source_type=reviewed` (S10). Review is not mandatory for automated analysis. S9 packages are never modified.

---

## 8. E-session validation

Package: `results_package__20261028T181714`  
Isolated publish root: `tmp_s10_sprint4_review_publications/isolated_e_session_controlled_publish/`  

`tests/behavioral_review/test_e_session_sprint4.py` verifies (when the package is currently eligible):

- Controlled synthetic edits + review decisions (not biological annotations)  
- Unique S10 reviewed run publication  
- 36 528 dense samples × 2 directional rows  
- Valid reviewed bout reconstruction / inclusive durations  
- Original S9 package + S8 states/bouts unchanged  
- Active-event replay + complete audit history  
- Review statuses consistent with changes  
- Current downstream eligibility + successful reopen  

---

## 9. Synthetic test coverage

`tests/behavioral_review/test_publication_qc_discovery.py` covers:

- Publication from a valid draft; no-correction publish  
- Active cursor replay after undo; superseded redo excluded from states  
- Complete history retained  
- Parent staleness blocking; invalid reviewer/journal; writer-lock conflicts  
- Corrected directional + parallel states; pure bout-collapse parity  
- Status-only rejection; voided decisions after geometry changes  
- Reviewed bout identity/lineage  
- Atomic failure cleanup + draft recovery; immutable no-overwrite  
- Discovery, reopen, historical vs current eligibility, supersession  
- No upstream modifications; no working-draft corruption  

---

## 10. Sprint 5 GUI prerequisites

1. Group 4 S10 page beside S9 (inspection workspace + explicit edit mode).  
2. Wire `GuiTaskRunner` for eligibility / publish / decode (no video hash on poll timers).  
3. Operator confirmation for “latest reviewed” selection (never silent auto-pick).  
4. Clear automated vs reviewed source picker for downstream analysis.  
5. Surface QC blockers/warnings and stale-parent publish blocks in UI.  
6. Lab checklist on `E_full_session_integration` (inspection-only and edit+publish paths).  
7. Still deferred: reviewed CSV/media via S9 exporters (U6), reciprocal tool (U2), `uncertain` (U3).

---

## 11. Stop point

Sprint 4 complete for architectural review. Do not start Sprint 5 coding until this report is accepted. Do not commit/push from this task unless explicitly requested.

_End of S10 Sprint 4 publication/QC report._
