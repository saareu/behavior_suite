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

This closure does not make S2 a complete scientific-analysis pipeline. An
elaborate pose-review workspace, model-optimization UI, active learning, pose
correction, identity verification, tracking correction, and final pose
processing are not part of the MVP. They remain future or downstream
responsibilities; S3 owns identity/tracking correctness and final scientific
usability.

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
conda-forge, and this repository installed in editable mode with its S2 runtime
dependencies. It does not
require PowerShell activation/profile setup and does not modify the user's base
environment.

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
approval or a final scientific-usability judgment; S3 is currently represented
by a clear downstream handoff interface. The full GPU acceptance workflow
verified this S3 handoff from a selected completed run.

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
- `docs/general/development/ai_coding_guide.md` — repository-wide AI-assisted
  development guidance

Historical plans, audits, release snapshots, and superseded design drafts are
preserved under `docs/subsystem_01/archive/`. They are retained for
traceability but are not the current source of truth.

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

Final biological identity assignment, tracking verification, implanted/partner
mouse assignment, identity-switch correction, imputation, pose
smoothing/finalization, behavior-ready feature extraction, and final
trajectory generation are downstream responsibilities. S2 technical QC does
not replace tracking validation, identity verification, or final scientific-
usability assessment.

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
- `docs/general/development/ai_coding_guide.md`

Keep changes scoped, preserve scientific invariants, and update tests for any
implementation behavior change.
