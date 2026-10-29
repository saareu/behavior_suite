# Subsystem 10 — Sprint 1: Source Contracts, Synchronization and Video Seeking

**Status:** Implemented (headless). Accepted; Sprint 2 follows.  
**Date:** 2026-10-29  
**Scope:** Eligible S9 package intake, dense↔prepared sync mapping, prepared-video frame access, read-only inspection session.  
**Non-scope:** Ethogram UI, playback GUI, S7 evidence windows, corrections, accept/reject, journals, publication, S9 mutation.

S10-B remaining MVP decisions frozen for coding: **U2** reciprocal tools deferred; **U3** `uncertain` deferred; **U6** reviewed CSV/media exports deferred; **U12** stale-parent publish prohibited; **U15** reject status-only. Independent directional correction semantics retained.

---

## 1. Modules and APIs

Package: `src/behavioral_review/` (registered in `pyproject.toml`).

| Module | Role |
| --- | --- |
| `contracts.py` | Errors, channel/identity constants, schema stamps, required bout columns |
| `source.py` | `open_review_source`, `ReviewSourcePackage`, upstream no-write snapshots |
| `sync.py` | `SyncMap`, `PlayheadMapping`, `BoutSeekMapping` over S9 `sync_index` |
| `video.py` | `PreparedFrameAccessor` / `DecodedFrame` (verified seek + small LRU) |
| `session.py` | `open_inspection_session`, `InspectionSession` read-only playhead state |
| `__init__.py` | Sprint 1 public re-exports only |

No empty modules for later sprints.

### Primary entry points

```text
open_review_source(package_dir, *, require_video=True, require_s8_states=False)
    -> ReviewSourcePackage

open_inspection_session(package_dir, *, require_video=True, ...)
    -> InspectionSession

SyncMap.from_sync_index(sync_index) -> SyncMap
PreparedFrameAccessor(video_path).read_frame(prepared_frame_idx) -> DecodedFrame
```

---

## 2. Source validation behavior

1. Accept an **explicit** S9 package directory identity.
2. Require historical status ∈ `{completed, completed_with_warnings}`.
3. Require current eligibility via `validate_s9_run_for_downstream` (historical status ≠ eligibility).
4. Verify package artifact digests against `package_manifest.json` when recorded.
5. Load and structurally validate `sync_index.parquet` (`validate_sync_index`).
6. Validate packaged bout schema + required seek columns; check `s8_bouts_schema_version`.
7. Resolve exact S8 states/bouts path+digest pins from `run_meta` (no alternate S8 run).
8. Resolve prepared video path+digest/fingerprint from `run_meta.source_video` (no substitute).
9. Opening is **inspection only** — no `behavioral_review__*` draft, no acceptance.

Failure modes raise `SourceError` (wrapping eligibility `PublicationError` / `ProvenanceError`).

---

## 3. Synchronization semantics

- Authority: published S9 `sync_index.parquet` (no S1/S5 recomputation).
- Dense `sample_idx` / `dense_frame_idx` / `time_sec` are behavioral authority.
- Video decode index = `prepared_frame_idx[sample_idx]`.
- Many-to-one plateaus supported; playhead advances by dense sample even when prepared index is unchanged.
- Prepared→dense click mapping: nearest `time_sec` on the plateau; **earlier `sample_idx` on ties**.
- Bout seeking uses inclusive `[start_sample_idx, end_sample_idx]`; packaged prepared endpoints must agree with sync lookup.
- Malformed / out-of-range mappings raise `SyncMappingError`.

---

## 4. Video access

- Decode via `preprocess.video_probe.read_raw_frame_at_index` (verified seek / bounded sequential fallback).
- API is frame-index based; suitable for `GuiTaskRunner` / worker threads (no Qt in this package).
- Small LRU cache keyed by `prepared_frame_idx` (default size 8); no full-video load.
- Plateau policy: `read_frame_if_changed` skips decode when prepared index unchanged.
- Missing video / digest mismatch / decoder failures → `VideoAccessError` (explicit; no source substitution).

---

## 5. Read-only session state

`InspectionSession` holds:

- Exact `package_run_id` / package dir / eligibility + warnings  
- Dense playhead (`PlayheadMapping` with both clocks)  
- Selected channel, direction `(focal_id, partner_id)`, bout id  
- Optional `PreparedFrameAccessor` + last decoded frame  

Seeking / selection does **not** modify behavioral results, create review decisions, or write journals.

---

## 6. Synthetic test evidence

Under `tests/behavioral_review/`:

| Test module | Coverage |
| --- | --- |
| `test_sync_mapping.py` | Plateaus, first/last, time tie-break, bout inclusivity, invalid schema, dense≠prepared |
| `test_source_open.py` | Eligible / stale / missing packages; broken video; digest mismatch; invalid sync; no upstream writes; no package substitution |
| `test_video_and_session.py` | First/mid/last decode; cache behavior; plateau keep-frame; directional vs symmetric bout identities; read-only open |

Fixtures: `tests/behavioral_review/_fixtures.py` (hand-computed sync + tiny MJPG AVI).

---

## 7. E-session findings

Target package: `test_cases/E_full_session_integration/results_package/results_package__20261028T181714`.

| Check | Expected |
| --- | --- |
| Dense samples | 36 528 |
| Unique prepared frames | 36 238 |
| Packaged bouts | 609 |
| Source S8 | `behavioral_arbitration__20261028T133953` |
| First/middle/last seek | Decode mapped prepared frames |
| Directional / contact / SBS bout seek | Inclusive endpoints agree with sync |
| Upstream writes | None; no review draft created |

Test: `tests/behavioral_review/test_e_session_sprint1.py`. If the package is not **currently** eligible, the suite **skips** and reports the eligibility error (missing/stale upstream dependencies) — checks are not weakened.

---

## 8. Sprint 2 prerequisites

1. Windowed S8 states projection around the playhead (states already resolved/pinned in Sprint 1).  
2. Dual directional exclusive + contact/SBS ethogram lanes (GUI).  
3. Bout overlay / navigation filters (`review_emit_v1` as navigation only).  
4. Lazy S7 evidence windows (optional deep dive).  
5. Async decode integration with Group 4 `GuiTaskRunner` (API already worker-safe).  
6. Still no corrections, journals, or publication.

---

## 9. Stop point

Sprint 1 complete for architectural review. Do not start Sprint 2 coding until this report is accepted. Do not commit/push from this task unless explicitly requested.
