# behavior_suite

Scientific behavioral-video processing suite for preparing mouse behavior
videos for SLEAP pose estimation and future downstream behavioral analysis.

**Subsystem 01 — Video Preprocessing** is functionally closed and in
maintenance. It prepares raw videos into validated, SLEAP-compatible prepared
videos while preserving raw decode-order frame identity, timing traceability,
accepted spatial geometry, and processing provenance.

**Subsystem 02 — Pose Inference and Technical QC** is finalized at MVP scope.
Its accepted PySide6 workflow covers S1-to-S2 navigation, bottom-up and
top-down centroid plus centered-instance inference, the standardized pose
artifact set, preserved S1 frame/timing mapping, technical QC, run discovery,
completed-run selection, and S3 handoff selection. Both inference modes ran on
a real GPU with SLEAP-NN 0.3.0 and `sleap-io` 0.8.0 and produced QC outcome
`pass`. The one-click Windows installation was also validated after the S2
runtime dependency fix.

**Subsystem 03 — Tracking Correction and Verification** is MVP implemented /
validated. From a completed S2 run, the GUI can run automatic current-lab
tracking correction, open video-centered review with optional manual
corrections, and explicitly accept `tracked_pose.parquet`. S3 owns tracking
identity correction and acceptance. S3 does not train the pose model.

**Subsystem 04 — Pose Finalization** is MVP complete / validated. From an
explicitly accepted S3 result, the GUI discovers the latest eligible accepted
run, runs dense 120 Hz finalization in the background (gap fill, optional
SavGol, automated QC), and writes `final_pose.parquet` under
`pose_finalization/<run_id>/`. A full laboratory session was validated through
the GUI. There is no S4 visual review or acceptance step.

**Subsystem 05 — Video / Blob Feature Extraction** is MVP complete /
lab-validated. From an eligible S1 session (independent of S2–S4), Group 2
exposes interactive segmentation preview, bookmarks, full extraction to
`blob_features.parquet` on the 120 Hz grid, run discovery, QC summary, and
GUI persistence. Laboratory GUI acceptance passed after Sprint 4 remediation
(batch bookmark previews, preview-required states, cache restoration, settings
invalidation, area filtering, navigation during extraction, and restart
persistence). Documented non-MVP follow-ups remain: cooperative cancellation,
completed-run settings preview, optional cache/performance work, and cable
removal. Subsystem 06 Combined Features is MVP complete / lab-validated.


**Subsystem 06 - Combined Pose & Video Feature Generation** is MVP complete / lab-validated. From jointly aligned eligible S4 `final_pose.parquet` and S5 `blob_features.parquet`, Group 2 exposes source selection, alignment status, background extraction to `dyadic_features_full.parquet`, QC, optional model-compatibility reporting, and run discovery. Operator laboratory GUI acceptance passed on session `E_full_session_integration` (S4 `pose_finalization__20261025T173206` + S5 `blob_features__20261021T152749` -> S6 `combined_features__20261025T173244`: 36528 samples, 73056 directional rows, structural QC OK). S6 does not train models or run inference. Documented non-MVP follow-ups remain: cooperative cancellation, GUI settings persistence, optional packaged model registry, and numerical regression vs historical Pipeline 50 caches.

**Subsystem 07 - Behavioral Inference** is MVP complete / lab-validated. From an eligible S6 `dyadic_features_full.parquet`, Group 3 exposes registry configuration, unified dyadic/focal detector selection, thresholds/`legacy_parity`, background inference to canonical + wide prediction artifacts, QC, and run discovery (including `stale_models` when recorded model digests are missing or changed). Operator laboratory GUI acceptance passed on session `E_full_session_integration` (2026-10-26; final GUI run `behavioral_inference__20261026T163224`). The standard Windows installer installs S2+S7 runtime libraries into `behavior_suite_gui`. Trusted model binaries remain lab-local and are configured once in the S7 registry path (persisted across launches). Documented non-MVP follow-ups remain: S8 arbitration/bouts, cooperative cancellation, multi-animal beyond two tracks, and shipping packaged models. S8 has not started.

S1–S7 MVP progress does not make the suite a complete scientific-analysis
pipeline. Elaborate model-optimization UI and active learning remain future or
downstream responsibilities.

---

## Supported Windows workflow

The supported Windows lab/developer workflow is:

```bat
git pull
scripts\install_windows_gui.bat
scripts\launch_windows_gui.bat
```

The installer recreates the dedicated `behavior_suite_gui` Conda environment
with Python 3.12, FFmpeg 7.1.1 from conda-forge, PySide6 6.11.1 from
conda-forge, and this repository installed in editable mode with the S2 and S7
runtime extras (`sleap-io`, plus `xgboost` / `scikit-learn` / `joblib` for
behavioral inference). Model binaries are **not** installed by the script; open
Subsystem 7 and browse to a trusted local model-registry root (saved for later
launches). The installer does not require PowerShell activation/profile setup
and does not modify the user's base environment.

If you must install S7 libraries manually into the GUI env, use a PowerShell-safe
form (avoid double-quoted `.[s7]`, which expands `$7`):

```powershell
conda run -n behavior_suite_gui python -m pip install -e '.[s2,s7]'
```

S2 uses two deliberately separate runtime components. The resolved external
`sleap-nn` 0.3.x executable performs model inference and writes `pose.slp`.
The Behavior Suite GUI Python environment provides `sleap-io` 0.8.0, which is
required to read that file for Parquet export and provenance extraction before
the shared QC and overlay artifact pipeline completes.

The installer reconciles this dedicated environment by recreating it from the
pinned Conda definition before installing the checkout. This intentionally
avoids `pip uninstall` for Qt packages: mixed or damaged PySide6 installations,
including installations whose pip `RECORD` metadata is missing, recover
automatically on the next installer run. Project/session data is outside this
dedicated environment and is not removed. Post-install checks start Python,
import the pinned PySide6 runtime and application dependencies, run `pip check`,
validate `sleap-io` and the S2 artifact modules, and run `behavior-suite doctor`.

Manual runtime check:

```bat
conda run -n behavior_suite_gui behavior-suite doctor
```

---

## User-facing preprocessing workflow

The GUI workflow is:

```text
Choose video
→ inspect / trim / optional pre-crop
→ detect cage automatically or select Manual ROI
→ review geometry
→ optional static mask
→ prepare
```

Successful preprocessing writes the official artifacts under the project
`preprocess/` directory:

```text
prepared_video.mp4
prepare_meta.json
prepared_sync.npz
cropped_background.png
settings_used.yaml
processing_log.txt
```

After a successful S1 run, the desktop application opens the same session in
the Subsystem 2 workspace without starting inference automatically. The
application-level navigation can also open S2 for an already selected project,
and S2 itself can browse an existing project/session.

In S2, users can review discovery metadata for prior runs, open available
overlays/run folders, copy discoverable settings, or configure a new bottom-up
model or top-down centroid + centered-instance bundle. Advanced device, batch,
animal-count, and tracking settings remain profile-driven. The backend performs
the authoritative handoff/model/profile/runtime preflight in a background
worker, so the UI stays responsive and displays honest indeterminate activity.

Both QC `pass` and non-blocking `review_recommended` runs that are technically
complete can be selected as intended S3 input. This selection is not identity
approval or a final scientific-usability judgment. From a selected completed S2
run, the GUI can launch Subsystem 3 (automatic correction → review → optional
manual edits → explicit acceptance).

After an accepted S3 result, Subsystem 4 finalizes dense pose on the
authoritative S1 120 Hz clock. Advanced Settings expose gap-fill and SavGol
parameters; defaults are used when left unchanged. Completed S4 runs are
discovered as eligible or stale by rechecking the recorded accepted S3 source.

Independently of S2–S4, Subsystem 5 (Group 2 — Feature Processing) loads an
eligible S1 session, previews segmentation, and runs full blob-feature
extraction. Completed S5 runs are discovered as eligible or stale by rechecking
recorded S1 digests. Subsystem 6 combines eligible S4+S5 into
`dyadic_features_full.parquet`. Subsystem 7 (Group 3) scores configured
dyadic/focal detectors from an eligible S6 run and publishes prediction
artifacts for future Subsystem 8.

---

## Active documentation

Current active documentation:

- `docs/README.md` — active documentation index
- `docs/subsystem_01/preprocessing.md` — canonical Subsystem 01 functional
  specification
- `docs/subsystem_01/status_and_roadmap.md` — current status, field-tested
  evidence, and future roadmap items
- `docs/subsystem_01/design/geometry_modes.md` — active future-facing spatial
  geometry design reference
- `docs/subsystem_02/mvp_scope_and_roadmap.md` — finalized Subsystem 02 MVP
  scope, completion status, and post-MVP roadmap
- `docs/subsystem_02/sleap_inference_specification.md` — Subsystem 02 backend
  pose inference contract
- `docs/subsystem_02/acceptance_test_specification.md` — Subsystem 02 backend
  inference acceptance-test specification
- `docs/subsystem_02/evidence/gpu_mvp_acceptance_v030.md` — recorded full S2
  MVP GPU acceptance evidence for both inference modes and S3 handoff
- `docs/subsystem_03/mvp_scope_and_roadmap.md` — finalized Subsystem 03 MVP
  scope, completion status, artifact contract, and stage boundary
- `docs/subsystem_03/minimal_implementation_contract.md` — S3 MVP implementation
  boundary (design history with implemented-status note)
- `docs/subsystem_04/mvp_scope_and_roadmap.md` — finalized Subsystem 04 MVP
  scope, completion status, and post-MVP roadmap
- `docs/subsystem_05/mvp_scope_and_roadmap.md` — finalized Subsystem 05 MVP
  scope, laboratory acceptance evidence, and S6 handoff boundary
- `docs/subsystem_06/mvp_scope_and_roadmap.md` — finalized Subsystem 06 MVP
  scope, laboratory acceptance evidence, and non-MVP follow-ups
- `docs/subsystem_07/mvp_scope_and_roadmap.md` — finalized Subsystem 07 MVP
  scope, laboratory acceptance evidence, and S8 boundary
- `docs/general/development/ai_coding_guide.md` — repository-wide AI-assisted
  development guidance

Historical plans, audits, release snapshots, and superseded design drafts are
preserved under `docs/subsystem_01/archive/`. Gate 1–4 S3 design documents under
`docs/subsystem_03/` are retained as design history; the S3 MVP closure document
is authoritative for implemented status and filenames.

---

## Subsystem boundary

Subsystem 01 validates prepared-video compatibility and frame-domain integrity.
It does not validate pose quality, SLEAP model accuracy, tracking quality,
instance counts, confidence scores, coordinate exports, inference results, or
SLEAP output-row structure.

Subsystem 02 is the pose-inference and technical-QC subsystem. Its MVP is
finalized for both the bottom-up path and the top-down centroid plus
centered-instance bundle under the same minimal artifact contract. The
validated workflow includes S1 integration, UI-based inference, standardized
artifacts, provenance, technical QC, overlay generation, run discovery,
completed-run selection, and S3 handoff selection. Post-MVP enhancements are
not missing MVP acceptance requirements.

Subsystem 03 is the tracking-correction and verification subsystem. Its MVP is
implemented and validated for automatic current-lab correction, video review,
manual correction, and explicit acceptance of `tracked_pose.parquet`. S2
technical QC does not replace tracking validation, identity verification, or
final scientific-usability assessment.

Subsystem 04 is the pose-finalization subsystem. Its MVP consumes accepted S3
`tracked_pose.parquet`, builds dense 120 Hz pose from S1 timing, applies gap
fill and optional SavGol smoothing, and exports `final_pose.parquet` with
automated QC.

Subsystem 05 is the video / blob feature-extraction subsystem. Its MVP
consumes eligible S1 prepared video and background, extracts morphology and
temporal occlusion features onto the shared 120 Hz timeline, and writes
`blob_features.parquet` with automated QC. Combined pose+blob features (S6) and
behavioral inference (S7) are implemented and lab-validated; cross-model
arbitration and bout packaging remain Subsystem 08.

---

## Development

Before making repository changes, read the applicable documentation:

- `docs/subsystem_01/preprocessing.md`
- `docs/subsystem_01/status_and_roadmap.md`
- `docs/subsystem_02/mvp_scope_and_roadmap.md` — required when changing or
  implementing Subsystem 02
- `docs/subsystem_02/sleap_inference_specification.md` — required when
  changing or implementing the Subsystem 02 backend
- `docs/subsystem_02/acceptance_test_specification.md` — required when
  changing backend inference acceptance behavior
- `docs/subsystem_03/mvp_scope_and_roadmap.md` — required when changing or
  implementing Subsystem 03
- `docs/subsystem_04/mvp_scope_and_roadmap.md` — required when changing or
  implementing Subsystem 04
- `docs/subsystem_05/mvp_scope_and_roadmap.md` — required when changing or
  implementing Subsystem 05
- `docs/general/development/ai_coding_guide.md`

Keep changes scoped, preserve scientific invariants, and update tests for any
implementation behavior change.
- `docs/subsystem_06/mvp_scope_and_roadmap.md` — required when changing or
  implementing Subsystem 06
- `docs/subsystem_07/mvp_scope_and_roadmap.md` — required when changing or
  implementing Subsystem 07
