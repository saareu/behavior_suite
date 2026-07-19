# behavior_suite

Scientific behavioral-video processing suite for preparing mouse behavior
videos for SLEAP pose estimation and future downstream behavioral analysis.

The current completed subsystem is **Subsystem 01 — Video Preprocessing**. It
prepares raw videos into validated, SLEAP-compatible prepared videos while
preserving raw decode-order frame identity, timing traceability, accepted
spatial geometry, and processing provenance.

Subsystem 01 is functionally closed and entering maintenance.

**Subsystem 02 — SLEAP-NN Inference and Review** is under active MVP
development. The bottom-up backend inference path is GPU-validated and the
top-down centroid plus centered-instance path has passed a SLEAP-NN 0.3.0 GPU
smoke test. The PySide6 desktop application now includes S1-to-S2 navigation,
existing-run technical review, bottom-up/top-down submission, and transient S3
input selection. Final field acceptance, safe cancellation, and live
subprocess-log streaming remain future work.

---

## Supported Windows workflow

The supported Windows lab/developer workflow is:

```bat
git pull
scripts\install_windows_gui.bat
scripts\launch_windows_gui.bat
```

The installer creates or updates the `behavior_suite_gui` Conda environment
with Python 3.12, FFmpeg 7.1.1 from conda-forge, PySide6 6.11.1 from
conda-forge, and this repository installed in editable mode. It does not
require PowerShell activation/profile setup and does not modify the user's base
environment.

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
by a clear placeholder only.

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
- `docs/subsystem_02/mvp_scope_and_roadmap.md` — current Subsystem 02 MVP
  scope, status, and roadmap
- `docs/subsystem_02/sleap_inference_specification.md` — Subsystem 02 backend
  pose inference contract
- `docs/subsystem_02/acceptance_test_specification.md` — Subsystem 02 backend
  inference acceptance-test specification
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

Subsystem 02 is the active pose-inference and review subsystem. Its current
validated implementation covers the bottom-up backend inference path and the
GPU-smoke-tested top-down model-bundle path under the same minimal artifact
contract. Its MVP desktop workflow and main-application navigation are now
implemented. Full MVP release status still depends on field acceptance and any
later backend progress/cancellation work chosen for release.

Final biological identity assignment, tracking verification, implanted/partner
mouse assignment, identity-switch correction, imputation, pose
smoothing/finalization, behavior-ready feature extraction, and final
trajectory generation are downstream responsibilities.

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
