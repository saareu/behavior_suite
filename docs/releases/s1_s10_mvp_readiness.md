# Behavior Suite — S1–S10 MVP Release Readiness

**Audit date:** 2026-10-29  
**Final verification:** 2026-10-29 (this document)  
**Checkout inspected (pre-release baseline):** `main` @ `985de69` (= `origin/main`)  
**Release intent:** annotated tag `v0.1.0` after docs-only P0 remediation + verification evidence  
**Authority:** repository code, configs, tests, subsystem docs, and local (gitignored) E-session artifacts — not documentation claims alone.  
**Scope of this document:** release audit + final verification evidence. No scientific changes or feature work.

---

## Verdict

**Internal S1–S10 MVP release accepted for `v0.1.0` (2026-10-29).**  
P0 remediation complete. Operator laboratory acceptance recorded for **S1–S10** (S9/S10 signed 2026-10-29). Fresh-machine prerequisites: [`installation_prerequisites.md`](installation_prerequisites.md). Release notes: [`v0.1.0.md`](v0.1.0.md).

No unresolved P0 blockers. Remaining items are **P1/P2 limitations**, not ship blockers.

---

## 1. Release scope

| In scope for internal MVP | Out of scope (do not block MVP) |
| --- | --- |
| Windows GUI workflow Groups 1–4 covering S1–S10 | Numerical/biological parity vs every legacy Pipeline cache |
| Headless runners + immutable run publication under session roots | Model training / SLEAP weight packaging in Git |
| Lab install via `scripts\install_windows_gui.bat` + launch script | Cooperative cancellation, K>2, S10 reviewed media re-export |
| Operator use on local/NAS session directories with writable outputs | CI/CD pipeline, non-Windows platforms |
| Documented limitations and known legacy deviations | Cleanup of gitignored `tmp_*` / `test_cases` working trees |

**Primary goal:** ship S1–S10 as soon as safely possible for internal lab use.

---

## 2. Actual repository state

### Git

| Item | Observed |
| --- | --- |
| Branch | `main` tracking `origin/main` |
| HEAD | `985de69` — *Add Subsystem 10 behavioral review beside S9 with nonblocking GUI edits.* |
| Sync | `origin/main...HEAD` = **0 / 0** (push of S9 `1ddca7f` and S10 `985de69` **succeeded**) |
| Working tree | **Clean** — no staged/unstaged/untracked paths reported |
| Prior feature branch | `feature/s2-topdown-backend` exists locally/remotely; not required for MVP |

### S1–S10 implementations: committed vs local-only

| Subsystem | On `origin/main`? | Notes |
| --- | --- | --- |
| S1–S8 | Yes | Prior MVP commits; S8 closure `ed283b1` + later fixes |
| S9 | Yes | `1ddca7f` — `src/results_packaging/`, Group 4 GUI, media/CSV export |
| S10 | Yes | `985de69` — `src/behavioral_review/`, Group 4 beside S9 |
| Local-only science | No | No uncommitted subsystem code |
| Local lab data | Yes (gitignored) | `test_cases/`, `tmp_s*/`, `acceptance_tests/` present on this workstation only |

### Documentation / operator acceptance (post–P0 remediation)

| Item | Status |
| --- | --- |
| Root `README.md` | **Updated** — Groups 1–4 / S1–S10; S9–S10 lab-validated; link to install prerequisites |
| `docs/README.md` | **Updated** — S9/S10 status matches signed checklists |
| S8 MVP roadmap / contracts | **Updated** — Group 4 described as delivered S9+S10 (outside S8 scope) |
| S9 lab checklist | **Pass** 2026-10-29 (core packaging + concise CSV/media export O1–O6) |
| S9 media/CSV export interactive checks | **Pass** (same checklist / `media_csv_export_gui.md`) |
| S10 lab checklist | **Pass** 2026-10-29 (inspection + controlled edit/undo-redo + publication) |
| Install prerequisites sheet | [`installation_prerequisites.md`](installation_prerequisites.md) |
| S1–S8 operator / lab validation docs | Recorded as passed in subsystem MVP docs (treat as **reported acceptance**, not re-executed in this audit) |

### Temporary scripts, generated data, accidental artifacts

| Path | Tracked? | Assessment |
| --- | --- | --- |
| `scripts/_s9e2_e_session_validate.py`, `scripts/_s10_gui_responsiveness_audit.py` | **Yes** | Probe/audit harnesses committed with S10; not part of operator install path |
| `tmp_s8_gui_probe/`, `tmp_s9_gui_e_session_probe.py`, `tmp_s10_*` | **No** (gitignore `/tmp_s*/`) | Local probe trees; safe to leave / delete locally |
| `test_cases/E_full_session_integration/` | **No** | Full lab session used by many tests/checklists; required for local regression, not for Git |
| `acceptance_tests/` | **No** | Historical S2 SLEAP acceptance dumps |
| `models/SLEAP/New folder/merged_all copy.slp` | **Yes (LFS)** | Accidental duplicate naming; low scientific value |
| `models/SLEAP/training_data/*.mp4` | **Yes (LFS)** | ~6+ GB training videos — not required to *run* S2 inference |
| Cache dirs (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`) | Ignored | Local only |

### Large / sensitive Git tracking

| Asset class | Tracking | Risk |
| --- | --- | --- |
| S7 detector bundles `models/dyadic/**`, `models/focal/**` (`*.joblib` ~0.3–0.5 MB each + JSON/parquet) | Regular Git | Acceptable size; **required** for S7 without external copy if using repo `models/` |
| SLEAP `.slp` + training `.mp4` under `models/SLEAP/` | Git LFS | Large clone cost; training MP4s are **not** S2 runtime weights |
| SLEAP **inference model directories** (trained networks) | **Not in repo** | Operators must supply lab-local paths in S2 UI |
| Session videos / run parquets under `test_cases/` | Ignored | Correct |

**Do not add further model binaries or research session data to Git for this release.**

### Previously authorized commits / pushes

Succeeded: HEAD matches `origin/main` at S10 commit `985de69`. No pending local commits.

---

## 3. S1–S10 capability matrix

Legend for **Lab validation**:  
**Reported** = subsystem docs/checklists claim operator pass.  
**Verified here** = this audit re-ran the GUI or full scientific pipeline (it did **not**).  
**Automated** = pytest coverage exists in-repo.

### Summary table

| SS | Entry points | Upstream | Produced | GUI | Lab validation | Tests (approx.) | External deps | Release blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S1** | `preprocess.service`; CLI `behavior-suite`; GUI wizard pages | Raw video (+ optional sync) | `preprocess/`: `prepared_video.mp4`, `prepare_meta.json`, `prepared_sync.npz`, `cropped_background.png`, `settings_used.yaml`, `processing_log.txt` | Group 1 | Reported closed / maintenance | `tests/preprocess/*`, `tests/integration/*` | Conda FFmpeg 7.1.1, OpenCV | None for MVP if doctor passes |
| **S2** | `pose_inference.runner`; `PoseInferencePage` | Eligible S1 handoff (video+meta+sync) | `pose_inference/<run>/`: `pose.slp`, `pose.parquet`, `overlay.mp4`, `pose_meta.json`, `settings_used.yaml`, `job_manifest.yaml`, `processing_log.txt` | Group 1; auto-open after S1 | Reported GPU MVP pass (both modes) | `tests/pose_inference/*`, UI tests | **External `sleap-nn` 0.3.x**, `sleap-io` 0.8.0, CUDA GPU practical, **lab SLEAP models** | Fresh machine without `sleap-nn`/models cannot run inference |
| **S3** | `tracking_correction.runner`; CLI `python -m tracking_correction`; `TrackingReviewPage` | Completed S2 run (`pose.parquet` + video via handoff) | `tracking_correction/<run>/`: `working_tracked_pose.parquet`, corrections JSON, accepted `tracked_pose.parquet` | Group 1; S2 handoff button; **Continue to Pose Finalization** after accept | Reported MVP validated | `tests/tracking_correction/*`, UI handoff tests | Prepared video + S2 pose | None beyond upstream |
| **S4** | `pose_finalization.runner`; `PoseFinalizationPage` (no dedicated CLI `__main__`) | Explicitly accepted S3 | `pose_finalization/<run>/`: `final_pose.parquet`, `run_meta.json` (**QC embedded**; no separate `qc.json`), settings, log | Group 1; manual **Continue to Pose Finalization** (not auto-started) | Reported lab GUI session | `tests/pose_finalization/*` | S1 timing + accepted S3 | Convenience only |
| **S5** | `blob_features.runner`; `BlobFeaturesPage` | Eligible S1 (stricter: + background) | `blob_features/<run>/`: `blob_features.parquet` + meta/QC | Group 2 | Reported lab-validated | `tests/blob_features/*`, UI | S1 prepared video/bg/sync | Independent of S2–S4 |
| **S6** | `combined_features.runner`; `CombinedFeaturesPage` | Jointly eligible S4 + S5 | `combined_features/<run>/`: `dyadic_features_full.parquet` + QC/compat | Group 2 | Reported E-session GUI pass | `tests/combined_features/*` | Aligned S4/S5 | Pipeline 50 numerical parity **unverified** (documented) |
| **S7** | `behavioral_inference.runner`; `BehavioralInferencePage` | Eligible S6 | `behavioral_inference/<run>/`: `predictions.parquet`, specialist + wide views, `model_manifest.json`, QC… | Group 3 | Reported lab-validated (2026-10-26) | `tests/behavioral_inference/*` | `[s7]` xgboost/sklearn/joblib; **registry root** → `models/` | Unconfigured registry blocks run; historical `legacy/models` paths need remap/junction |
| **S8** | `behavioral_arbitration.runner`; `BehavioralArbitrationPage` | S8-ready S7 + **exact** linked S6 | `behavioral_arbitration/<run>/`: `behavior_states.parquet`, `behavior_bouts.parquet`, QC… | Group 3 | Reported lab-validated + 47/47 consistency audit | `tests/behavioral_arbitration/*` | Full-ethogram S7 coverage | Documented Pipeline 100 deviations (not blockers) |
| **S9** | `results_packaging.runner` + `media_csv_export`; `ResultsPackagingPage` | Eligible S8 (+ prepared video for media) | `results_package/<run>/` sync/bouts/summary/manifest/QC; optional `results_exports/` CSV/MP4 jobs | Group 4 | **Operator Pass 2026-10-29** (core + CSV/media) | `tests/results_packaging/*`, UI | FFmpeg for bout videos | None for MVP after P0 |
| **S10** | `behavioral_review.*`; `BehavioralReviewPage` | Eligible S9 package | Working drafts under `behavioral_review/working/`; published `behavioral_review/<run>/` reviewed states/bouts + manifests | Group 4 beside S9 | **Operator Pass 2026-10-29** (inspect + edit + publish) | `tests/behavioral_review/*`, UI | Prepared video decode via sync | None for MVP after P0 |

### Distinguishing reported vs verified

- This audit **verified Git presence, module layout, contracts, GUI wiring, install scripts, and test inventory**.
- It did **not** re-execute GPU S2, full S1→S10 on a new recording, or interactive operator checklists.
- E-session on this workstation contains published runs through S10 (`behavioral_review__20261029T194225`, multiple `results_package__*`, exports) — evidence of **prior** processing, not a single continuous fresh GUI acceptance in this audit.

---

## 4. End-to-end workflow (new laboratory recording)

```text
Raw video
  → S1 Prepare (GUI) → preprocess artifacts
  → [auto] S2 workspace (inference not auto-started)
  → S2 configure SLEAP models + run → select completed run → S3 handoff
  → S3 auto-correct → review → optional edits → Accept tracked_pose
  → [manual] open S4 → finalize → final_pose
  → [parallel/manual] S5 from same S1 → blob_features
  → S6 combine eligible S4+S5 → dyadic_features_full
  → S7 set registry → infer → predictions
  → S8 arbitrate (exact linked S6) → states + bouts
  → S9 package (+ optional CSV / bout videos)
  → S10 inspect → optional draft corrections → Publish reviewed run
```

### Gaps (blocker vs convenience)

| Finding | Class |
| --- | --- |
| S1→S2 auto-navigation; S2→S3 handoff button | Working handoffs |
| S3 accept enables **Continue to Pose Finalization** (manual click; does not auto-run S4) | Convenience |
| No one-click continue after S4→S5 / S5→S6 / S6→S7 / S7→S8 / S8→S9 / S9→S10 | Convenience |
| S5 independent of S2–S4; S6 requires both S4+S5 eligible and aligned | By design |
| S7 requires one-time registry browse (`models/` suggested; QSettings persist; `legacy/models`→`models` remap in `gui_persistence`) | Setup (document) |
| S8 refuses silent newest-S6 substitution; uses S7-linked S6 only | Correct / safe |
| S9/S10 soft-nav; S2/S4 hard-lock | By design |
| No Cancel on long S5–S10 jobs | Documented limitation |
| Failed runs leave incomplete dirs; discovery classifies failed/incomplete/stale | Recoverable via new run (immutable publish) |
| Historical E-session S7 manifests with `legacy/models/...` can mark S9 **stale** until digests resolvable (lab used junction 2026-10-29) | Ops footgun for **old** runs; new runs with `models/` OK |
| Stages **can** run from documented upstream artifacts alone once deps exist | No missing science stage |
| Continuous **fresh** recording through S10 in one sitting | **Not evidenced** as a single accepted lab run (E-session stages span July–Oct dates) | Validation gap (P1 if checklists signed on E-session only) |

**Manual / undocumented setup for a fresh machine:** install Conda; run installer; install/resolve **external `sleap-nn` 0.3.x**; obtain **SLEAP inference models** outside Git; point S7 at `models/`; ensure session root writable (local or verified NAS).

---

## 5. Installation readiness

| Check | Result |
| --- | --- |
| Python / Conda | `environment-gui.yml`: Python 3.12, FFmpeg 7.1.1, PySide6 6.11.1; recreate via `scripts\install_windows_gui.bat` |
| Editable install | `pip install -e ".[s2,s7]"` (installer); `environment-gui.yml` header comment matches `[s2,s7]` |
| GUI launch | `scripts\launch_windows_gui.bat` → `behavior-suite doctor` then `behavior-suite gui` |
| Doctor | FFmpeg/runtime preflight only — **does not** verify `sleap-nn`, GPU, or S7 model registry |
| Post-install validator | `scripts\validate_windows_gui_runtime.py` checks PySide6, sleap-io, S7 imports, key app modules |
| SLEAP inference | **Not** installed by suite installer; must be resolvable (`sleap-nn` beside env or on PATH); suite uses `sleap-io` for post-process |
| S7 registry | YAML `model_registry.root: null`; GUI persistence + suggest `repo/models`; expected SHA256 digests in `behavioral_inference.contracts` |
| Video / FFmpeg | Conda-forge FFmpeg in GUI env; S9 media export + S1 prepare depend on it |
| NAS / paths | S2 treats `\\` paths as network and requires explicit verification; prefer writable local or known-good NAS session roots |
| Writable outputs | All stages write under session project tree; need create permission |
| Fresh lab machine | **Can** install and launch GUI after Conda + installer; **cannot** complete S2 without external SLEAP stack/models; **cannot** complete S7 without registry; LFS pull of training MP4s optional for inference |
| pytest in GUI env | Not installed by default (`[dev]` extra) — expected |

---

## 6. Legacy-parallel assessment

Approximate stage map (legacy tree present on this workstation under `legacy/`; often gitignored on other clones — **not** a runtime dependency):

| Legacy | Suite | Relationship |
| --- | --- | --- |
| `00_prepare` | S1 | Equivalent preprocessing role; modern sync/geometry contract |
| `10_pose` (SLEAP) | S2 | Equivalent inference role; suite standardizes artifacts + technical QC |
| `20_fix_track_ids` | S3 | Current-lab corrector backend; explicit accept gate |
| `30_impute` / imputed NPZ | **Not restored**; role partly replaced by **S4** | Dense 120 Hz finalization from accepted S3 + S1 sync; legacy imputed-NPZ path abandoned |
| `40_extract_blob_features` | S5 | Feature extraction on shared 120 Hz grid |
| `50_dyadic_features` | S6 | Vendored formulas; **numerical regression vs training caches deferred** |
| `60/80` train | — | **Deferred** (out of MVP) |
| `70` dyadic + `90` focal inference | S7 | Joblib XGB bundles; `legacy_parity` mode used for lab acceptance |
| `100_behavioral_arbiter` | S8 | Ported arbitration with **documented deviations** (invalid evidence, `undefined`, no contact remultiplex; Stage 8 morph-close disabled at `max_gap_to_fill=0`) |
| `110_bout_browser_packager` | S9 | Packaging ancestor; S8 is scientific authority; emit floors only as optional `review_emit_v1` profile |
| Streamlit `bout_review_app` | S10 | Replaced by Qt Group 4 review; no reciprocal/`uncertain` in MVP |

**Do not claim numerical or biological parity** where evidence is absent (esp. Pipeline 50 caches, ethogram ground truth, full Pipeline 100 bit-identity).

---

## 7. Release test plan (smallest credible suite)

### A. Automated regression (cached / fixture where designed)

Run from a **dev** environment with `pytest` (not required inside `behavior_suite_gui`):

```text
pytest tests/preprocess tests/pose_inference tests/tracking_correction \
  tests/pose_finalization tests/blob_features tests/combined_features \
  tests/behavioral_inference tests/behavioral_arbitration \
  tests/results_packaging tests/behavioral_review tests/ui \
  tests/project tests/cli
```

- Mark **integration** tests separately when FFmpeg/video fixtures available: `tests/integration`.
- Distinguish: most S7–S10 E-session tests **reuse** gitignored `test_cases/E_full_session_integration` (cached upstream) vs tests that synthesize tiny frames in-process.
- Expect skips/fails if `test_cases/` absent on a clean clone — record that as environment limitation, not product failure.

### B. Fresh integrated S1→S10 (lab)

1. New short session (or trimmed clip) on writable disk.
2. S1 prepare → S2 inference (lab SLEAP models) → S3 accept → S4 → S5 → S6 → S7 (`models/`) → S8 → S9 package → S10 inspect.
3. Confirm each stage discovers the intended upstream run (no silent stale substitute).

If full GPU time is scarce: do **B on a short clip**, and use **C** for full-length packaging/review scale.

### C. Failure / recovery / immutability

1. Induce one failed S7/S8/S9 (e.g. stop mid-run or bad registry) → discovery shows failed/incomplete; prior completed runs unchanged.
2. Publish S9; mutate nothing in S8 tree; confirm fingerprints.
3. S10 draft edit + publish; confirm S8/S9 bytes unchanged; superseded vs current classification visible.

### D. One controlled review correction + publication

Execute S10 checklist sections D–E on E-session or B’s S9 package: identity → edit mode → draft → one exclusive (and/or contact) correction → publish → reopen immutable reviewed run.

### E. External model / video dependencies

1. `sleap-nn` resolvable; S2 preflight fails clearly when missing.
2. S7 registry points at `models/`; digests match `EXPECTED_MODEL_SHA256` where enforced.
3. S9 bout-video export small subset plays; CSV row count matches selection.

### F. Clean installation

On a machine or clean env: `install_windows_gui.bat` → validator → doctor → `launch_windows_gui.bat` opens Groups 1–4.

### G. Operator GUI smoke

Signed **Pass** (2026-10-29):

- `docs/subsystem_09/laboratory_acceptance_checklist.md` (core + concise export O1–O6)
- `docs/subsystem_10/laboratory_acceptance_checklist.md` (critical C/D/E)

---

## 8. Findings by priority

### P0 — Must fix / complete before defensible internal release

**Resolved 2026-10-29** (see [`s1_s10_p0_remediation_plan.md`](s1_s10_p0_remediation_plan.md)):

1. ~~S9 operator laboratory acceptance~~ — **Pass** (core + CSV/media export).  
2. ~~S10 operator laboratory acceptance~~ — **Pass** (inspection + controlled edit + publish).  
3. ~~Operator-facing documentation honesty~~ — README / docs index / S8–S10 status aligned with Groups 1–4.  
4. ~~Install prerequisite sheet~~ — [`installation_prerequisites.md`](installation_prerequisites.md) linked from root README.

### P1 — Important; ship with documented limitations

1. No single continuous **fresh** full-length S1→S10 lab acceptance recorded (E-session is multi-date staged). Prefer at least one short-clip fresh path (test plan B).  
2. ~6 GB LFS training videos + accidental `models/SLEAP/New folder/...` burden clones; document “LFS skip / not required for inference.”  
3. `behavior-suite doctor` does not check `sleap-nn` or S7 registry — operators can launch a GUI that fails only at S2/S7.  
4. Tracked probe scripts under `scripts/_s9*` / `_s10*` — clarify non-operator.  
5. No CI workflow in-repo — regressions rely on local pytest.  
6. Pipeline 50 / full ethogram ground-truth parity unverified — already documented; keep as limitation.  
7. Historical `legacy/models` path eligibility footgun for old S7/S9 trees — document remap/junction for archival sessions.

### P2 — Defer after MVP

1. Cooperative cancellation; K>2; S8 overlay video; S10 reviewed CSV/media export; reciprocal editing; `uncertain` status.  
2. Remove or relocate SLEAP training MP4s from default clone story; delete accidental LFS duplicate.  
3. Numerical Pipeline 50 regression harness; richer doctor checks; CI.  
4. Auto S3→S4 navigation; S5 cable removal; GUI settings persistence polish called out in S5–S8 roadmaps.

---

## 9. Shortest viable remediation sequence

**P0 steps 1–4 completed 2026-10-29.** Remaining before ship branding:

5. **Optional but recommended:** one short-clip fresh S1→S10 smoke (test plan B) — P1.  
6. **Ship** internal MVP after final release-verification review; keep §8 P1/P2 as documented limitations.  
7. **Do not** block on LFS video cleanup, CI, or parity campaigns.

---

## 10. Exact release acceptance criteria

Internal S1–S10 MVP may be declared **accepted** only when all of the following are true:

1. `main` remains green/clean relative to intended release commit; S1–S10 packages present under `src/` as on `985de69` (or successor that only remediates P0 docs/ops).  
2. Root README and `docs/README.md` agree on S9/S10 presence and validation status.  
3. S9 laboratory checklist signed **Pass** (core packaging + concise export O1–O6).  
4. S10 laboratory checklist signed **Pass** for critical inspection, one controlled edit path, and publication/immutability.  
5. Installer + launch validated on at least one lab Windows machine (`doctor` + GUI Groups 1–4 visible).  
6. Written prerequisites acknowledge: external `sleap-nn` 0.3.x, lab SLEAP inference models, S7 `models/` registry, FFmpeg via Conda env.  
7. Release notes list P1 limitations (no full-length fresh E2E required if short-clip smoke **or** signed E-session S9/S10 checklists are accepted as scale evidence).  
8. No new research data or large weight dumps added to Git for the release.

**Fail** the release if S9/S10 checklists fail, if S2 cannot be documented as runnable with lab SLEAP assets, or if documentation still tells operators Group 4 does not exist.

---

## 11. Capability detail appendix (artifacts / config anchors)

| SS | Package | Default config | Primary GUI |
| --- | --- | --- | --- |
| S1 | `src/preprocess/` | `configs/preprocess_default.yaml` | Wizard pages under `src/ui/pages/` |
| S2 | `src/pose_inference/` | `configs/subsystem_02/*.yaml` | `pose_inference_page.py` |
| S3 | `src/tracking_correction/` | backend YAML under package | `tracking_review_page.py` |
| S4 | `src/pose_finalization/` | (runner defaults / Advanced Settings) | `pose_finalization_page.py` |
| S5 | `src/blob_features/` | `configs/subsystem_05/blob_features_default.yaml` | `blob_features_page.py` |
| S6 | `src/combined_features/` | `configs/subsystem_06/combined_features_default.yaml` | `combined_features_page.py` |
| S7 | `src/behavioral_inference/` | `configs/subsystem_07/behavioral_inference_default.yaml` | `behavioral_inference_page.py` |
| S8 | `src/behavioral_arbitration/` | `configs/subsystem_08/behavioral_arbitration_default.yaml` | `behavioral_arbitration_page.py` |
| S9 | `src/results_packaging/` | `configs/subsystem_09/results_packaging_default.yaml` | `results_packaging_page.py` |
| S10 | `src/behavioral_review/` | (settings via review session) | `behavioral_review_page.py` |

Navigation shell: `src/ui/main_window.py` — Groups 1–4, S1–S10 buttons.

Automated test inventory: **~140** `tests/**/test_*.py` files across preprocess through behavioral_review + UI (no `.github` CI workflows found).

---

## 12. Final verification evidence (2026-10-29)

### Repository

| Check | Result |
| --- | --- |
| Branch | `main` up to date with `origin/main` at `985de69` before release commit |
| S1–S10 packages tracked under `src/` | Confirmed (`preprocess` … `behavioral_review`) |
| Pending release surface | Docs/operator acceptance + `environment-gui.yml` header comment + `docs/releases/*` only — **no** scientific/GUI production-code edits |
| Excluded from release | `test_cases/`, `tmp_*` probes, model weight dumps, generated exports |

### Automated regression

Command (dev/base env; `--import-mode=importlib` required because duplicate test basenames exist across subsystem dirs):

```text
pytest tests/preprocess tests/pose_inference tests/tracking_correction \
  tests/pose_finalization tests/blob_features tests/combined_features \
  tests/behavioral_inference tests/behavioral_arbitration \
  tests/results_packaging tests/behavioral_review tests/ui \
  tests/project tests/cli --import-mode=importlib
```

| Outcome | Count / detail |
| --- | --- |
| Passed | **1450** |
| Skipped | **1** — optional S7 `nose_to_body` focal fallback absent (documented) |
| Failed | **3** — explained; **not** treated as new scientific/product regressions (below) |
| Duration | ~5 minutes |

**Explained failures (do not weaken tests; do not block `v0.1.0`):**

1. `tests/pose_inference/test_gitignore.py` — expects obsolete pattern `test_cases/**/preprocess/`; committed `.gitignore` already uses broader `/test_cases/` (since 2026-10-26). Stale assertion vs intentional ignore policy.
2. `tests/behavioral_review/test_e_session_sprint1.py::test_e_session_no_upstream_writes` and `…sprint2::test_e_session_sprint2_no_writes` — assert E-session has **no** published `behavioral_review__*` runs. After signed S10 acceptance, published run `behavioral_review__20261029T194225` exists. Upstream immutability helpers ran before that assertion; failure is **lab-state pollution from successful S10 publish**, not a new write-path regression.

**Ruff** (`ruff check src tests`): **8** findings, all in pre-existing S4 / pose-finalization UI/test import-sort and style debt on `985de69` source (no release-diff Python).  
**Mypy** (project `packages` list): **43** errors across 15 files — pre-existing S1/S2/CLI/pandas-stubs debt on baseline; S5–S10 packages historically validated focused. Not introduced by this docs release.

### Installation / dependency smoke

| Check | Result |
| --- | --- |
| `behavior-suite doctor` in `behavior_suite_gui` | Pass — FFmpeg 7.1.1 / runtime supported |
| `scripts/validate_windows_gui_runtime.py` | Pass — PySide6 6.11.1, sleap-io 0.8.0, S7 xgboost/sklearn/joblib, app modules |
| Full recreate via `install_windows_gui.bat` | **Not re-run** this pass (env already present); prior installer path remains canonical |
| External SLEAP-NN | Resolvable `sleap-nn` **0.3.0** with `predict` (`~/.local/bin`); conda `sleap` / `sleap-nn-env` still ship older track-only builds — operators must use the lab 0.3.x executable |
| S7 registry artifacts | `models/dyadic/v1` and `models/focal/v1` present |
| Offscreen GUI nav smoke | Groups **1–4** and Subsystems **1–10** buttons present |

### E-session integration evidence (gitignored local)

Session `test_cases/E_full_session_integration` contains published/completed trees for S1–S10 (multi-date staged, not one continuous fresh clip):

- S1 `preprocess/` … S8 `behavioral_arbitration__20261028T133953` (lab-validated)
- S9 `results_package__20261028T181714` (+ exports under `results_exports/`)
- S10 `behavioral_review__20261029T194225`

**Fresh short-clip S1→S10 was not executed in this verification.** Internal MVP acceptance relies on automated regression + signed S9/S10 operator checklists + staged E-session artifacts (acceptance criterion §10 item 7).

### Acceptance criteria (§10) checklist

1. S1–S10 on `main` — **Yes**  
2. README / docs index honest on S9/S10 — **Yes** (P0 docs)  
3. S9 checklist Pass — **Yes**  
4. S10 checklist Pass — **Yes**  
5. Installer/doctor/GUI Groups 1–4 — **Yes** (smoke; full recreate deferred)  
6. Prerequisites sheet — **Yes**  
7. Release notes list P1 limits; fresh full-length E2E not required — **Yes** ([`v0.1.0.md`](v0.1.0.md))  
8. No new research data / large weights added — **Yes**

---

## 13. Stop point

> **`v0.1.0` internal laboratory MVP may ship** after the release commit + annotated tag.  
> **Do not** open a feature sprint for P1/P2 items as release blockers.
