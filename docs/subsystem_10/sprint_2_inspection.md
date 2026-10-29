# Subsystem 10 — Sprint 2: Ethogram, Bout and Evidence Inspection

**Status:** Implemented (headless data layer). Stop for architectural review before Sprint 3.  
**Date:** 2026-10-29  
**Scope:** Read-only behavioral inspection over eligible S9 packages — windowed S8 ethogram lanes, full packaged bout browsing/filtering, lazy S7/S8 evidence, visualization payloads.  
**Non-scope:** Manual correction, review accept/reject, draft persistence, reviewed publication, full Qt viewer, S9/S8/S7 science changes.

S10-B frozen deferred items remain: **U2** reciprocal tools; **U3** `uncertain`; **U6** reviewed exports; **U12** stale-parent publish block; **U15** reject status-only.

---

## 1. Modules and APIs

Package: `src/behavioral_review/` (extends Sprint 1).

| Module | Role |
| --- | --- |
| `ethogram.py` | `EthogramAccessor` / `EthogramWindow` — column-projected S8 states windows |
| `bouts.py` | `BoutBrowser` / `BoutFilter` / `BoutOverlay` — full packaged bout navigation |
| `evidence.py` | `EvidenceAccessor` — S8 echoed probs + lazy S7 windows + arbitration provenance |
| `timeline.py` | `TimelineView` / LOD segments / probability traces / playhead sync info |
| `session.py` | Extended `InspectionSession` — one dense playhead drives all inspection surfaces |
| `source.py` | Adds `source_s7_run_dir`, `s8_states_available` |
| `contracts.py` | Inspection schema stamps, evidence vocabulary, ethogram column sets |

### Primary entry points

```text
open_inspection_session(package_dir, *, require_s8_states=True, ...)
    -> InspectionSession

session.ethogram_window(half_window=240 | start/end) -> EthogramWindow
session.overlapping_bouts(...) -> DataFrame
session.set_bout_filter(BoutFilter(...))  # display only
session.previous_bout() / next_bout()
session.s8_echoed_evidence(...) -> S8EchoEvidenceWindow
session.s7_evidence(...) -> S7EvidenceWindow
session.provenance_at_playhead() / provenance_for_selected_bout()
session.timeline_view(...) -> TimelineView
```

Seeking, filtering, and evidence reads never modify science artifacts or create review decisions.

---

## 2. Shared inspection state

`InspectionSession` keeps **one authoritative dense playhead** (`PlayheadMapping`):

- `sample_idx` / `time_sec` (behavioral authority)
- `prepared_frame_idx` / `prepared_time_sec` (video decode target)
- Selected channel, direction `(focal_id, partner_id)`, bout id
- Active bout filter (UI state only)
- Ethogram / bout / evidence accessors bound at open

All inspection panels derive from this state. Plateau policy from Sprint 1 is unchanged: ethogram advances by dense sample even when prepared index is unchanged.

---

## 3. Timeline / bout / evidence schemas

### Ethogram (`behavioral_review_ethogram_window_v1`)

| Surface | Semantics |
| --- | --- |
| Exclusive lanes | One track per `(focal_id, partner_id)`; labels include `engaged` and `undefined` as distinct values |
| Contact lane | Independent `symmetric_pair` track (`pred_contact` / `prob_contact`) |
| SBS lane | Independent `symmetric_pair` track (`pred_side_by_side` / `prob_side_by_side`) |
| Reconstruction | `label_reconstruction`, `undefined_reason`, `hierarchy_residual_kind` |
| Evidence badges | Invalid / unobserved / valid positive / valid negative (never paint routing sentinels as confident 0) |

Window bounds are inclusive dense `[start_sample_idx, end_sample_idx]`. Reads project only required S8 columns; no bout re-collapse; no arbitration re-run.

### Bouts (`behavioral_review_bout_browser_v1`)

Authoritative source: full S9 `behavior_bouts.parquet` (not `review_emit_v1`).

Supports listing, filter by channel/behavior/direction, stable `bout_id` selection, inclusive start/end seek, overlap queries, previous/next within the active filter set, and prepared-video navigation annotations. Contact/SBS remain separate from exclusive directional bouts. Filters are display-only.

### Evidence (`behavioral_review_evidence_window_v1`)

| Path | Policy |
| --- | --- |
| Default | S8 echoed `prob_*` / `pred_*` / `input_valid_*` / `score_origin_*` (U10) |
| Deep dive | Lazy windowed S7 `predictions.parquet` (`prob_effective`, threshold, prediction, validity, specialist/fusion provenance) |
| Provenance | Final label, candidate preds, reconstruction fields, contact/SBS channel values |
| Diagnostics | Optional `arbitration_diagnostics.parquet` / JSON — **explicit unavailable** when absent (never invent rule explanations) |

Missing evidence ≠ valid negative. Invalid / routing_sentinel ≠ confident 0.

### Timeline view (`behavioral_review_timeline_view_v1`)

GUI-ready payloads without Qt:

- Windowed run-length segments per lane
- Directional + parallel lane metadata
- Selected-bout overlays (inclusive endpoints)
- Probability/validity traces
- Playhead sync info (including plateau membership)

LOD/downsampling may merge paint segments or stride probability traces for long timelines. Flag `scientific_values_unchanged=True` is mandatory — LOD never rewrites ethogram values, bout boundaries, or future edit precision. Dense accessors remain authoritative.

---

## 4. Read-only semantics

| Action | Effect |
| --- | --- |
| Open package | Inspection session only |
| Seek / filter / evidence | In-memory only |
| Upstream S7/S8/S9 | Fingerprint-guarded; no writes |
| Review drafts | Not created |
| Acceptance | Never implied |

---

## 5. Lazy-loading strategy

| Asset | Load policy |
| --- | --- |
| `sync_index`, packaged bouts, summary | Eager at open (Sprint 1) |
| S8 `behavior_states` | Windowed + column-projected; small window cache |
| S7 predictions | Only when `s7_evidence` requested; filter by sample window + detectors |
| S8 stage diagnostics | Lazy; unavailable when file absent |
| Prepared video | Decode on demand (Sprint 1) |
| S6 features | Never opened |

APIs are synchronous headless functions suitable for later `GuiTaskRunner` / cancellation wrappers (Sprint 5). No GUI-thread disk I/O is implied by this package.

---

## 6. Performance measurements (E-session)

Package: `results_package__20261028T181714` on workstation (2026-10-29).

| Operation | Timing | Notes |
| --- | --- | --- |
| `open_inspection_session` (states required) | ~1.14 s | Includes eligibility + sync/bouts load |
| Ethogram window ±240 samples | ~0.044 s | 2 directions; 54 projected columns |
| Cold ethogram window `[0,480]` | ~0.033 s | |
| S8 echoed evidence ±120 | ~0.189 s | ~4.3k detector points |
| Lazy S7 window ±60 (`approach`,`contact`) | ~0.039 s | 121 rows |
| Timeline view ±240 | ~0.368 s | 7 RLE segments; LOD=full |
| Peak traced memory (inspection path) | ~38 MB | `tracemalloc` peak during script |

Windowed paths avoid full S7 table materialization on open and avoid loading all S8 columns for small timeline windows.

---

## 7. E-session findings

| Check | Result |
| --- | --- |
| Dense samples | 36 528 |
| Full packaged bouts | 609 (`exclusive` 538, `contact` 60, `side_by_side` 11) |
| Directional ethogram tracks | Both directions present |
| Contact / SBS | Independent parallel tracks |
| Representative bout prev/next | Passes under exclusive filter |
| Lazy S7 | Available via package `source_s7.run_dir` |
| S8 reconstruction | `label_reconstruction` / residuals readable from states |
| Stage diagnostics file | **Unavailable** on this S8 run (`arbitration_diagnostics.parquet` absent) — explicit unavailable state |
| Upstream writes / drafts | None |
| Engaged vs undefined | Distinct in synthetic tests; this E-session states have no `undefined` rows (empty `undefined_reason`) |

Tests: `tests/behavioral_review/test_e_session_sprint2.py` (skip if package not currently eligible).

---

## 8. Synthetic test coverage

Under `tests/behavioral_review/`:

| Module | Coverage |
| --- | --- |
| `test_ethogram_and_bouts.py` | Window bounds/time alignment; dual exclusive tracks; contact/SBS independence; engaged≠undefined; reconstruction; filters; prev/next; inclusive overlays; plateaus; read-only |
| `test_evidence_and_timeline.py` | S8 echo + lazy S7; invalid vs valid-negative; missing diagnostics; LOD without science change; no upstream writes |
| `test_e_session_sprint2.py` | Scale, lanes, navigation, evidence, timings, no writes |

---

## 9. Sprint 3 prerequisites

1. Explicit edit-mode toggle + required `reviewer_id`.  
2. Append-only correction event journal + apply engine (framewise Model A).  
3. Undo/redo over journal cursor; working dir under `behavioral_review__*`.  
4. Bout UI gestures compiling to framewise events; status/notes stores independent (U15).  
5. Void acceptance on geometry change (R9).  
6. Collapse via S8 pure `extract_behavior_bouts` only (U11).  
7. Still no immutable publication (Sprint 4) and no full Group 4 viewer (Sprint 5).  
8. Async GUI task integration for windowed ethogram/evidence fetches + stale-request tokens.

---

## 10. Stop point

Sprint 2 complete for architectural review. Do not start Sprint 3 coding until this report is accepted. Do not commit/push from this task unless explicitly requested.

_End of S10 Sprint 2 inspection report._
