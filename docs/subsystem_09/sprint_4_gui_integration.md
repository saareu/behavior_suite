# Subsystem 09 — Sprint 4: GUI Integration and Laboratory Validation

**Status:** Implemented (GUI + automated validation). **Operator laboratory acceptance pending** — see [laboratory_acceptance_checklist.md](laboratory_acceptance_checklist.md). Stop for architectural review.  
**Date:** 2026-10-28  
**Scope:** Group 4 navigation; S9 packaging page/controller; async discovery; nonblocking Run Packaging via `GuiTaskRunner`; package summary / QC / reopen; E-session probe evidence. **No S10**, no ethogram/video viewer, no bout editing.

Authoritative design: [architecture_audit.md](architecture_audit.md) (S9-A), [implementation_contract.md](implementation_contract.md) (S9-B), Sprints 1–3 docs.

---

## 1. GUI architecture and navigation

| Item | Implementation |
| --- | --- |
| Group 4 | `GROUP_RESULTS_REVIEW` — **Behavioral Results & Review** |
| Page | **S9 — Results Packaging & Export** (`ResultsPackagingPage`) |
| S10 | Not present (no empty interactive page) |
| Groups 1–3 | Unchanged (S1–S4 / S5–S6 / S7–S8) |
| Session | Shared `MainWindow._session_root`; opening S9 binds/refreshes that session |
| Hard locks | S2/S4 still hard-block navigation |
| Soft background | S5–S9 allow leaving and returning while work continues |

Files:

- `src/ui/main_window.py` — Group 4 button, S9 button, open/close/task wiring
- `src/ui/pages/results_packaging_page.py`
- `src/ui/controllers/results_packaging_controller.py`
- `src/ui/tasks.py` — `GuiTaskKind.RESULTS_PACKAGING` / `RESULTS_PACKAGING_DISCOVERY`

Headless packaging science remains exclusively in `src/results_packaging/` (GUI never duplicates sync/bout/QC logic).

---

## 2. Source selection and readiness

On open / session change:

1. GUI thread: `prepare_session` → immediate return with discovering state.
2. Worker: `collect_discovery_snapshot` lists S8 runs, assesses S9 readiness via memoized `validate_s8_source_for_packaging`, summarizes S9 packages.
3. GUI thread: `apply_discovery_snapshot` only if request id + generation + session still current.

Behavior:

- Defaults to the **latest S9-ready** S8 source.
- Manual S8 selection is preserved across refresh when still listed.
- Stale discovery results after session switch are rejected.
- Prepared-video SHA256 is **not** hashed on discovery polls (`hash_prepared_video` defaults false; worker-only when enabled).

---

## 3. Settings interface

| Area | Controls |
| --- | --- |
| Scientific / profile | Optional `review_emit_v1`; include contact/engaged/SBS; emit floors (frames + seconds @ fixed 120 Hz) |
| Execution | `require_prepared_video`; optional `hash_prepared_video` |
| Fixed / read-only | `analysis_full` default package; `reference_lightweight` mode |

UI copy clarifies:

- Default package always includes all S8 bouts.
- Optional profile is an additional filtered view.
- Selecting a profile never modifies the full package.
- Lightweight package is not a self-contained video archive.

Settings are snapshotted into `PackagingRequestIdentity` at launch.

---

## 4. Task lifecycle

1. **Run Packaging** → immediate `begin_packaging_launch` (no full revalidation on GUI thread).
2. Worker → `execute_packaging_launch` (revalidate S8 + `run_results_packaging_publication` with progress).
3. Success/failure recorded with identity gates; post-completion discovery refresh is async.
4. Duplicate launch blocked; source/settings locked while busy.
5. No Cancel control; no fake percentages (indeterminate or real `OperationProgress` units).

---

## 5. Package summary, QC, reopen

Summary distinguishes:

- Full authoritative bout counts vs optional profile include/exclude.
- Historical completion status vs current eligibility.
- Sync-row count, channel counts, artifact paths, prepared-video / sync references.

QC categories:

- **Structural errors** (`failures`)
- **Warnings** (scientifically meaningful)
- **Information** (expected configuration / residuals / profile omissions)

Completed packages listed newest-first; selecting reopens metadata/QC without reprocessing (JSON reads only on GUI thread).

---

## 6. Automated tests

| File | Coverage |
| --- | --- |
| `tests/ui/test_results_packaging_gui.py` | Group 4 nav, shared session, settings, snapshot immutability, duplicate prevention, nav during run, QC/reopen, historical vs eligibility, S1–S8 group regression |
| `tests/ui/test_results_packaging_responsiveness.py` | Async discovery, stale rejection, nonblocking launch, immediate Run Packaging |

Backend S9 suite and S7/S8 GUI regressions remain green (see §8).

---

## 7. E-session GUI probe evidence (2026-10-28)

Session: `test_cases/E_full_session_integration`  
Historical package: `results_package__20261028T181714` (source S8 `behavioral_arbitration__20261028T133953`)  
Probe script: `tmp_s9_gui_e_session_probe.py` → `_sprint4_gui_probe_report.json`

| Check | Result |
| --- | --- |
| Discover eligible S8 | Pass (multiple S9-ready S8; default newest ready) |
| Open historical `…181714` | Pass |
| Authoritative bout count | **609** (538 exclusive / 60 contact / 11 SBS) |
| Optional review export | **514 included / 95 excluded** |
| Sync rows | **36 528** |
| Inherited warnings + informational | Present and separated |
| Current eligibility | `completed_with_warnings` (current) |
| New unique GUI launch | `results_package__20261028T183941` from S8 `…133953` |
| S8 / prior S9 unmodified | Confirmed |

### Measured responsiveness (probe; separate GUI-blocking from background)

| Metric | Value |
| --- | --- |
| GUI-thread `prepare_session` | **0.34 ms** |
| Cold discovery (worker) | **~8.8 s** |
| Warm rediscovery (worker) | **~1.8 s** |
| Run-button accept (`begin_packaging_launch`) | **0.19 ms** |
| Background publish | **~2.7 s** |
| Post-run summary rows | **0.48 ms** |

Cold discovery time is background computation, not GUI freeze. Page open returns control immediately with discovering state.

---

## 8. Static checks

| Check | Outcome |
| --- | --- |
| Ruff (S9 GUI + tests) | Pass |
| Pytest S9 GUI + responsiveness | 15 passed |
| Pytest S9 backend + S7/S8 GUI regressions | 102 passed (combined with prior suite run) |
| Mypy on `ui.*` files | **Environment limitation:** PySide6 lacks stubs / `py.typed`; project `mypy` packages list does not include `ui` (same constraint as S7/S8 GUI). Not treated as an S9 code defect. Headless `results_packaging` remains in the mypy package set. |

---

## 9. Outstanding operator acceptance and blockers

- Interactive checklist in [laboratory_acceptance_checklist.md](laboratory_acceptance_checklist.md) is **not** signed off.
- Do **not** mark S9 lab-validated until the operator completes that checklist.
- Remaining MVP blockers outside Sprint 4: S10 (deferred), portable embed / clips / GIFs (U-R3/U-R4), additional export profiles.

---

_End of Sprint 4 documentation. Stop for architectural review._
