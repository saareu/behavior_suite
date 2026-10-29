# Subsystem 10 — Sprint 3: Framewise Corrections, Review Status and Recoverable Working Drafts

**Status:** Implemented (headless editing engine). Stop for architectural review before Sprint 4.  
**Date:** 2026-10-29  
**Scope:** Explicit edit mode, framewise correction events, pure S8 bout regeneration, independent review status/notes, durable working drafts with single-writer lock, linear undo/redo across restart.  
**Non-scope:** Reviewed-run publication, Group 4 Qt GUI, reciprocal editing tool, `uncertain` status, S8/S9 science mutation.

Deferred MVP items remain frozen: **U2** reciprocal tools; **U3** `uncertain`; **U6** reviewed exports; **U12** stale-parent publish block; **U15** reject status-only.

---

## 1. Editing APIs

Package: `src/behavioral_review/` (extends Sprint 1–2).

| Module | Role |
| --- | --- |
| `editing.py` | `EditingSession` / `open_editing_session` — headless edit API |
| `events.py` | `CorrectionEvent` schema + validation |
| `journal.py` | Append-only JSONL journal + cursor / undo-redo |
| `locks.py` | Single-writer lock + stale recovery |
| `draft.py` | Create/open `behavioral_review__*/working/` |
| `apply.py` | Framewise apply + trim/split/merge compile |
| `regenerate.py` | Pure S8 collapse + reviewed bout identities |
| `status.py` | Status/notes overlay + geometry voiding |

### Primary entry points

```text
open_editing_session(package_dir) -> EditingSession   # inspection-first
session.enter_editing_mode(reviewer_id)
session.create_draft(review_root=...) / open_draft(run_dir)
session.set_exclusive_label(...)
session.set_parallel_pred(channel=contact|side_by_side, ...)
session.trim_bout_boundary(...) / split_bout_at_sample(...) / merge_adjacent_bouts(...)
session.set_bout_review_status(...) / add_note(...)
session.undo() / redo()
session.current_reviewed_states() / current_reviewed_bouts() / current_statuses()
session.changed_intervals() / compare_automated_versus_reviewed(...)
session.seek_sample(...)   # never writes journal events
```

### Editing-mode gate

1. Opening an S9 package remains **inspection-only**.
2. Writes require: `enter_editing_mode(reviewer_id)` + open draft + held writer lock.
3. Seeking, filtering, and evidence inspection never create journal events or imply acceptance.
4. No writes to S8/S9 inputs (fingerprint-guarded after each commit).

---

## 2. Event schemas

**Stamp:** `behavioral_review_correction_events_v1` (JSONL under `working/correction_events.jsonl`).

Core fields follow S10-B §4.4: `event_id`, `event_seq`, `timestamp_utc`, `reviewer_id`, `op`, channel/identity, inclusive `start_sample_idx`/`end_sample_idx`, payloads, `undo_group_id`, `client_op_id`.

### Op catalog (implemented)

| Op | Framewise? |
| --- | --- |
| `set_exclusive_label` | Yes |
| `set_parallel_pred` / `clear_parallel_pred` | Yes |
| `trim_bout_boundary` | Compiled → framewise |
| `split_bout_at_sample` | Compiled → framewise |
| `merge_adjacent_bouts` | Compiled → framewise |
| `set_bout_review_status` | No (status store) |
| `add_note` / `edit_note` / `delete_note` | No |
| `session_marker` | No |

`set_exclusive_label_reciprocal` remains **deferred (U2)** — `reciprocal=True` is rejected.

---

## 3. Framewise correction behavior

- Inclusive dense intervals; `end < start` and out-of-range samples rejected (no silent clamp).
- Exclusive: one label per `(sample_idx, focal_id, partner_id)`; opposite direction untouched.
- Contact and SBS: independent symmetric-pair channels; both directional state rows’ parallel preds updated consistently.
- No implicit reciprocal edits, arbitration, smoothing, or scientific threshold changes.
- Touched exclusive samples set `label_reconstruction="manual_review"`.
- Original automated evidence remains readable via the parent S9/S8 baseline.

---

## 4. Bout regeneration

Reviewed bouts are regenerated with S8 `extract_behavior_bouts` only (U11) — no hierarchy/temporal/contact cleaning.

Preserved:

- Directional vs symmetric identity grain
- Inclusive start/end sample indices
- `bout_duration_sec = bout_duration_frames / 120`
- Channel membership

Incremental replay regenerates after each framewise event so trim/split/merge can resolve `reviewed_bout_id` targets deterministically.

---

## 5. Identity and lineage

Reviewed bout ids are deterministic and **never** reuse automated bout ids:

```text
rb__{review_run_id}__{channel}__{identity}__{behavior}__{start}_{end}__{ordinal:04d}
```

`source_s8_bout_id` is set only when the reviewed interval key still exactly matches an automated bout; otherwise null. Geometrically changed bouts therefore lose automated lineage and prior acceptances are voided (below).

---

## 6. Review status

Statuses: `unreviewed` | `accepted` | `rejected` | `voided` (no `uncertain`).

- Rejection is **status-only** — labels unchanged (U15).
- Notes are independent of label corrections.
- Decisions attach to stable bout identities (`source_s8_bout_id` / reviewed bout id).
- Geometry-changing edits void affected `accepted`/`rejected` decisions (`void_reason=geometry_changed`) while retaining them in audit history.
- Acceptance is **not** transferred to split/merged/boundary-modified children.
- Unaffected approvals on other identities remain stable.

---

## 7. Journal and recovery

Working layout:

```text
{review_root}/behavioral_review__{ts}/
  run_meta.json
  working/
    working_manifest.json      # parent package pins, reviewer, draft state
    correction_events.jsonl    # append-only
    journal_cursor.json        # cursor + redo_high_water + superseded seqs
    review_bout_status.jsonl   # reserved companion
    notes.jsonl
    locks/writer.lock
    snapshots/
```

Write path: append event line(s) → fsync → atomic cursor temp→rename.

Crash recovery:

- Truncate incomplete trailing JSONL line
- Clamp cursor to last valid event
- Rebuild cursor when corrupt
- Operator-visible `recovery_warnings`

Single-writer lock per draft; concurrent writers rejected. Stale locks (dead PID) require explicit `force_stale_lock=True` and leave an audit break marker. Multiple reviewers use **separate** drafts.

---

## 8. Undo/redo

Linear undo/redo over `undo_group_id` units:

- Survives application restart (cursor + journal persisted)
- Replay of `active_events()` reconstructs identical working states/statuses
- New edits after undo mark the redo-side seqs as **superseded** without rewriting historical lines; event ids remain stable
- History distinguishes active vs superseded via cursor metadata

---

## 9. E-session test evidence

Package: `results_package__20261028T181714`  
Isolated draft root: `tmp_s10_sprint3_review_drafts/isolated_e_session_controlled_edits/` (not the production session `behavioral_review/` tree).

Controlled edits exercised by `tests/behavioral_review/test_e_session_sprint3.py`:

1. Directional exclusive label change  
2. Inclusive bout boundary trim  
3. Independent contact/SBS correction  
4. Accept/reject status  
5. Undo/redo  
6. Draft reopen with exact cursor/state recovery  

**Invariant:** all **609** automated packaged bouts remain byte-identical in the source S9 package; upstream fingerprints unchanged. No reviewed-run publication performed.

---

## 10. Synthetic test coverage

`tests/behavioral_review/test_editing_corrections.py` covers:

- Inspection-only write prohibition  
- Reviewer identity / draft gates  
- Exclusive directional corrections; no reciprocal propagation  
- Contact/SBS independence  
- Inclusive interval validation  
- Trim/split/merge compilation  
- Pure S8 collapse parity on unmodified baseline  
- Reviewed-bout identity / lineage  
- Status-only rejection; notes without correction  
- Acceptance voiding after geometry change; stable unaffected approvals  
- Undo/redo across restart; new edit after undo  
- Journal crash recovery  
- Concurrent-writer rejection; stale-lock recovery  
- Deterministic replay; no S8/S9 writes  

---

## 11. Sprint 4 prerequisites

1. Atomic immutable publish layout (`correction_events.parquet`, reviewed states/bouts, status, manifest, QC).  
2. Parent digest / eligibility stale → **block publish** (U12).  
3. Discovery classifications (`eligible`, `draft`, `stale_parent`, `superseded`, …).  
4. `validate_s10_run_for_downstream`.  
5. Still no Group 4 full GUI (Sprint 5) and no reviewed CSV/media via S9 exporters (U6).

---

## 12. Stop point

Sprint 3 complete for architectural review. Do not start Sprint 4 coding until this report is accepted. Do not commit/push from this task unless explicitly requested.

_End of S10 Sprint 3 corrections/history report._
