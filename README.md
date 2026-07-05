# behavior_suite

Scientific behavioral-video processing suite for preparing mouse behavior videos for SLEAP pose estimation and future downstream behavioral analysis.

The current completed subsystem is **Subsystem 01 — Video Preprocessing**. It prepares raw videos into validated, SLEAP-compatible prepared videos while preserving raw decode-order frame identity, timing traceability, accepted spatial geometry, and processing provenance.

Subsystem 01 is functionally closed and entering maintenance.

---

## Supported Windows workflow

The supported Windows lab/developer workflow is:

```bat
git pull
scripts\install_windows_gui.bat
scripts\launch_windows_gui.bat
```

The installer creates or updates the `behavior_suite_gui` Conda environment with Python 3.12, FFmpeg 7.1.1 from conda-forge, PySide6 6.11.1 from conda-forge, and this repository installed in editable mode. It does not require PowerShell activation/profile setup and does not modify the user's base environment.

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

Successful preprocessing writes the official artifacts under the project `preprocess/` directory:

```text
prepared_video.mp4
prepare_meta.json
prepared_sync.npz
cropped_background.png
settings_used.yaml
processing_log.txt
```

---

## Active documentation

Current active documentation:

- `docs/README.md` — active documentation index
- `docs/subsystem_01/preprocessing.md` — canonical Subsystem 01 functional specification
- `docs/subsystem_01/status_and_roadmap.md` — current status, field-tested evidence, and future roadmap items
- `docs/subsystem_01/design/geometry_modes.md` — active future-facing spatial geometry design reference
- `docs/subsystem_02/sleap_inference_specification.md` — Subsystem 02 inference specification
- `docs/subsystem_02/acceptance_test_specification.md` — Subsystem 02 acceptance-test specification
- `docs/general/development/ai_coding_guide.md` — repository-wide AI-assisted development guidance

Historical plans, audits, release snapshots, and superseded design drafts are preserved under `docs/subsystem_01/archive/`. They are retained for traceability but are not the current source of truth.

---

## Subsystem boundary

Subsystem 01 validates prepared-video compatibility and frame-domain integrity. It does not validate pose quality, SLEAP model accuracy, tracking quality, instance counts, confidence scores, coordinate exports, inference results, or SLEAP output-row structure.

Future subsystems may cover SLEAP inference and tracking, pose quality control, behavioral feature extraction, visualization, and reporting.

---

## Development

Before making repository changes, read the applicable documentation:

- `docs/subsystem_01/preprocessing.md`
- `docs/subsystem_01/status_and_roadmap.md`
- `docs/subsystem_02/sleap_inference_specification.md` — required when changing or implementing Subsystem 02
- `docs/general/development/ai_coding_guide.md`

Keep changes scoped, preserve scientific invariants, and update tests for any implementation behavior change.
