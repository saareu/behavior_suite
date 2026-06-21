# behavior_suite

Scientific behavioral-video processing suite for preparing mouse behavior videos for SLEAP pose estimation and future downstream behavioral analysis.

The first implemented subsystem is **Video Preprocessing**. Its purpose is to transform raw behavioral videos into validated, SLEAP-compatible prepared videos while preserving deterministic frame identity, timing traceability, crop geometry, and processing provenance.

---

## Current Status

This repository is under active development.

Current development focus:

```text
Subsystem 01 — Video Preprocessing
```

Planned future subsystems include:

```text
SLEAP inference and tracking
Pose quality control and correction
Behavioral feature extraction
Blob and trajectory analysis
Behavior classification and clustering
Visualization and reporting
```

---

## Core Design Goals

The preprocessing pipeline is designed around several scientific requirements:

* Preserve deterministic mapping from each prepared frame to its original raw-video decode-order frame.
* Avoid frame resampling by default.
* Support external timing vectors from MATLAB `.mat` files.
* Use external timing as the preferred experimental timing source when it is valid.
* Treat raw video PTS as diagnostic information only.
* Validate that the final prepared video can be read sequentially by OpenCV.
* Fail loudly on frame-count mismatch rather than silently applying workarounds.
* Preserve animal geometry through aspect-ratio-preserving scaling and padding.
* Record all meaningful settings, crop geometry, timing choices, and validation results.

---

## Repository Structure

```text
behavior_suite/
├── configs/
│   └── preprocess_default.yaml
│
├── docs/
│   ├── preprocess_subsystem_spec_v1.md
│   ├── preprocess_implementation_plan_v1.md
│   └── ai_coding_guide.md
│
├── legacy/
│   ├── prepare_reference.py
│   └── cage_cropper_reference.py
│
├── src/
│   ├── project/
│   │   ├── models.py
│   │   ├── paths.py
│   │   ├── service.py
│   │   └── validation.py
│   │
│   ├── preprocess/
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── video_probe.py
│   │   ├── mat_sync_reader.py
│   │   ├── crop_plan.py
│   │   ├── pre_crop.py
│   │   ├── cage_detection.py
│   │   ├── manual_crop.py
│   │   ├── masking.py
│   │   ├── video_prepare.py
│   │   ├── background.py
│   │   ├── sync_writer.py
│   │   ├── metadata.py
│   │   ├── validation.py
│   │   └── service.py
│   │
│   ├── cli/
│   │   └── preprocess.py
│   │
│   └── ui/
│
└── tests/
    ├── project/
    ├── preprocess/
    └── integration/
```

The `project` module is shared infrastructure and is intentionally separate from `preprocess`.

---

## Command-Line Interface

Install the project and inspect the available preprocessing commands:

```powershell
pip install -e ".[dev]"

behavior-suite --help
behavior-suite preprocess --help
```

Automatic cropping uses an explicit three-phase workflow:

```powershell
behavior-suite preprocess detect-crop --project-dir ProjectName --raw-video raw.avi --config configs/preprocess_default.yaml --output-crop-plan ProjectName/preprocess/detected_crop_plan.json

behavior-suite preprocess accept-crop --crop-plan ProjectName/preprocess/detected_crop_plan.json --output-crop-plan ProjectName/preprocess/accepted_crop_plan.json

behavior-suite preprocess run --project-dir ProjectName --raw-video raw.avi --config configs/preprocess_default.yaml --crop-plan ProjectName/preprocess/accepted_crop_plan.json
```

Crop detection never accepts a crop automatically. Review the detected JSON,
create a separate accepted JSON with `accept-crop`, and pass only that accepted
plan to `run`.

### Desktop GUI setup

```powershell
pip install -e ".[dev,gui]"

behavior-suite gui
```

The desktop GUI covers project creation/opening, raw-video probing, trim
selection, and typed pre-crop configuration. The Timing page supports optional
MATLAB `.mat` timing-vector selection with strict validation against the raw
sequential readable-frame count. Crop review, encoding, and final execution
remain visible workflow placeholders.

---

## Preprocess Subsystem

The Preprocess Subsystem supports the following workflow:

```text
Create or open a project
↓
Select raw video
↓
Probe video metadata
↓
Optionally select MATLAB timing file
↓
Select and validate timing vector
↓
Select frame range
↓
Optionally define pre-crop
↓
Detect cage or use manual four-corner crop
↓
Accept crop
↓
Run preprocessing
↓
Validate final prepared video
↓
Write official artifacts
```

The authoritative design document is:

```text
docs/preprocess_subsystem_spec_v1.md
```

The implementation sequence is defined in:

```text
docs/preprocess_implementation_plan_v1.md
```

---

## Preprocess Outputs

A successful preprocessing run produces:

```text
ProjectName/
└── preprocess/
    ├── prepared_video.mp4
    ├── prepare_meta.json
    ├── prepared_sync.npz
    ├── cropped_background.png
    ├── settings_used.yaml
    └── processing_log.txt
```

### Artifact Overview

| Artifact                 | Purpose                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| `prepared_video.mp4`     | Final SLEAP-compatible video                                             |
| `prepare_meta.json`      | Authoritative run metadata, geometry, timing, validation, and provenance |
| `prepared_sync.npz`      | Frame-level mapping and timing arrays                                    |
| `cropped_background.png` | Median background from the final prepared video                          |
| `settings_used.yaml`     | Accepted processing settings used for the run                            |
| `processing_log.txt`     | Processing commands, validation results, warnings, and errors            |

Intermediate files may be created under:

```text
ProjectName/preprocess/.internal/
ProjectName/preprocess/debug/
```

They are not official outputs.

---

## Scientific Frame-Mapping Rule

In the default no-resampling mode:

```text
prepared_frame_idx → raw_decode_frame_idx
```

with:

```python
raw_decode_frame_idx = start_frame + prepared_frame_idx
```

The pipeline must not silently drop, duplicate, interpolate, or reorder frames.

The final prepared video must satisfy:

```text
OpenCV reported frame count
=
OpenCV sequentially readable frame count
=
expected trimmed frame count
```

Any mismatch is a hard failure.

---

## External Timing Support

The preprocessing subsystem can optionally load a generic MATLAB `.mat` file containing one timing value per original raw-video frame.

The user selects the intended variable manually.

For a timing vector to be accepted, it must be:

```text
Numeric
One-dimensional after squeeze
Finite
Monotonically increasing
Exactly the same length as the original untrimmed raw video
```

Supported timing units:

```text
seconds
milliseconds
microseconds
nanoseconds
frames
unknown
```

When temporal units are used, values are converted to seconds and stored as `external_time_sec` in `prepared_sync.npz`.

---

## Installation

Installation instructions will be added when the initial core engine is implemented.

The intended environment includes:

```text
Python
NumPy
OpenCV
SciPy
h5py
Pydantic
PyYAML
ffmpeg
ffprobe
pytest
```

The desktop application will use managed or bundled `ffmpeg` and `ffprobe` binaries rather than relying on arbitrary system PATH settings.

---

## Development Workflow

Before modifying preprocessing code, read:

```text
docs/preprocess_subsystem_spec_v1.md
docs/preprocess_implementation_plan_v1.md
docs/ai_coding_guide.md
```

The coding rules in `docs/ai_coding_guide.md` are mandatory for AI-assisted development.

The initial implementation order is:

```text
1. Shared project module
2. Configuration models
3. Video probe
4. MATLAB timing reader
5. CropPlan and geometry
6. Cage detection
7. Manual crop
8. ffmpeg preparation
9. OpenCV final re-encode
10. Prepared-video validation
11. Sync and metadata artifacts
12. Background generation
13. CLI
14. GUI
```

---

## Legacy Code

The `legacy/` directory contains existing scripts used as a behavioral reference during migration.

They are useful for preserving validated behavior, including:

```text
ffmpeg preprocessing
safe FPS handling
OpenCV final re-encoding
cage detection
background estimation
```

Legacy code should be refactored into focused modules rather than copied unchanged into the new implementation.

One legacy behavior must not be retained:

```text
prepared_frame_count - 1 workaround
```

The new pipeline must fail whenever final video frame availability is ambiguous.

---

## Testing

The project uses:

```text
Unit tests
Integration tests
Future real-data regression tests
```

Run tests with:

```powershell
pytest
```

Once the package configuration is added, targeted tests can be run with:

```powershell
pytest tests/preprocess/test_video_probe.py
pytest tests/preprocess/test_mat_sync_reader.py
pytest tests/preprocess/test_validation.py
```

---

## Contributing Rules

When modifying the codebase:

1. Keep changes scoped to the requested feature.
2. Preserve scientific invariants unless the specification is formally updated.
3. Add or update tests for changed core behavior.
4. Do not weaken validation rules.
5. Do not put scientific processing logic in the GUI.
6. Do not rename official artifacts without updating the documentation.
7. Do not add hidden fallbacks or guessed timing values.
8. Record unresolved limitations clearly.

---

## License

License information will be added before external distribution.
