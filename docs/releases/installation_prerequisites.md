# Fresh laboratory machine — installation prerequisites

**Authority:** `environment-gui.yml`, `pyproject.toml`, `scripts/install_windows_gui.bat`, `scripts/launch_windows_gui.bat`, Subsystem 02 SLEAP resolution contract.  
**Date:** 2026-10-29  
**Audience:** Operators installing Behavior Suite for internal Windows lab use (S1–S10 GUI).

This sheet consolidates external dependencies the Windows installer does **not** fully provision. It does not change installer behavior.

---

## 1. Supported environment

| Item | Requirement |
| --- | --- |
| OS | Windows lab/developer workstation (supported install/launch path is the `.bat` workflow below) |
| Conda | Miniforge or Anaconda on `PATH` ([conda-forge download](https://conda-forge.org/download/) recommended by the installer) |
| Python runtime | Dedicated Conda env `behavior_suite_gui` from `environment-gui.yml`: **Python 3.12**, **FFmpeg 7.1.1**, **PySide6 6.11.1** (conda-forge) |
| Package extras | Editable install `.[s2,s7]` — `sleap-io==0.8.0` (S2) plus `xgboost` / `scikit-learn` / `joblib` (S7) |

Do not install into the user’s base Conda environment. The installer recreates `behavior_suite_gui` from the pinned YAML.

---

## 2. Install and launch (canonical)

From the repository root:

```bat
git pull
scripts\install_windows_gui.bat
scripts\launch_windows_gui.bat
```

Manual runtime check (same doctor call the launcher runs):

```bat
conda run -n behavior_suite_gui behavior-suite doctor
```

If S7 libraries must be reinstalled manually into the GUI env, use a PowerShell-safe form (avoid double-quoted `.[s7]`, which expands `$7`):

```powershell
conda run -n behavior_suite_gui python -m pip install -e '.[s2,s7]'
```

Post-install validation also runs `scripts\validate_windows_gui_runtime.py` (PySide6, `sleap-io`, S7 imports, key application modules) and `pip check`.

---

## 3. What the installer provides vs what it does not

| Provided by installer | Not installed / not verified by installer or doctor |
| --- | --- |
| Conda env recreate + editable suite with `[s2,s7]` | External **`sleap-nn` 0.3.x** executable |
| FFmpeg 7.1.1 in the GUI env | Lab **SLEAP inference model directories** (trained networks) |
| PySide6 6.11.1 (conda-forge) | GPU/CUDA stack for SLEAP inference |
| `sleap-io` 0.8.0 for reading `pose.slp` | Automatic S7 registry path (operator browses once in GUI) |
| S7 runtime libs (`xgboost` / `scikit-learn` / `joblib`) | Session videos, project trees, or laboratory recordings |

`behavior-suite doctor` checks FFmpeg/runtime preflight only. It does **not** verify `sleap-nn`, GPU, or the S7 model registry.

---

## 4. External SLEAP-NN 0.3.x (Subsystem 02)

- The suite does **not** install SLEAP-NN.
- Inference requires a resolvable **`sleap-nn` 0.3.x** executable. Preflight rejects incompatible 0.2.x builds.
- Resolution order (recorded in run metadata): explicit CLI/profile path → `sleap-nn.exe` / `sleap-nn` beside the active Python executable → `PATH`.
- External `sleap-nn` writes `pose.slp`; the GUI env’s `sleap-io` 0.8.0 reads that file for Parquet export and provenance.
- A CUDA-capable GPU is practical for laboratory inference; without `sleap-nn` and lab models, the GUI can launch but S2 inference cannot complete.

**Do not invent alternate SLEAP install commands here.** Follow your laboratory’s validated SLEAP-NN 0.3.x provisioning, then confirm the executable is resolvable under the rules above.

---

## 5. SLEAP inference models vs Git LFS training assets

| Asset | Location | Required to run S2 inference? |
| --- | --- | --- |
| Trained SLEAP **inference** model directories | Lab-local paths selected in the S2 GUI | **Yes** |
| SLEAP training videos / labels under `models/SLEAP/` (often Git LFS) | Tracked optional clone payload | **No** |

Training MP4s under `models/SLEAP/training_data/` are not required to run inference. Skipping a large LFS pull is acceptable for runtime use.

---

## 6. Subsystem 07 model registry

- Trusted detector bundles ship under the repository `models/` tree: `models/dyadic/v1/…` and `models/focal/v1/…`.
- In Subsystem 7, browse once to a trusted registry root (typically the repo `models/` directory). The path is persisted across launches.
- Model binaries are **not** installed by `install_windows_gui.bat`.

### Historical path compatibility

Older S7 runs may record model paths under `legacy/models/...`. Those digests must still resolve for downstream eligibility (S8/S9). Laboratory practice on archival E-session data used a local directory junction from `legacy/models` to validated `models/` artifacts (matching SHA-256). Prefer new runs with registry root `models/`.

---

## 7. Video decoding and FFmpeg

- FFmpeg **7.1.1** comes from conda-forge via `environment-gui.yml` into `behavior_suite_gui`.
- S1 prepare and S9 bout-video export depend on that env FFmpeg.
- Prefer decoding/writing on a writable local disk. Network/`\\` session paths may require explicit verification (S2 treats UNC paths as network).

---

## 8. Session / video / synchronization inputs

A typical laboratory session root must provide (as each stage completes):

| Stage | Inputs needed |
| --- | --- |
| S1 | Raw behavior video (+ optional sync); writable project/session directory |
| S2 | Eligible S1 prepared video + meta + sync; resolvable `sleap-nn` + inference models |
| S3–S4 | Completed/accepted upstream pose artifacts under the same session |
| S5 | Eligible S1 (prepared video + background + sync) |
| S6 | Jointly eligible S4 + S5 on the shared 120 Hz timeline |
| S7 | Eligible S6 + configured model registry |
| S8 | S8-ready S7 + **exact** linked S6 (no silent newest-S6 substitution) |
| S9 | Eligible S8 (+ prepared video when exporting bout MP4s) |
| S10 | Eligible S9 package (+ prepared video via package sync index) |

All stages write under the session/project tree. Ensure create permission on the chosen root.

**Do not commit** laboratory recordings, generated run trees, or additional large model weights to Git for this release.

---

## 9. Optional Git LFS

If you need SLEAP training media tracked under `models/SLEAP/`, use Git LFS per your clone policy. LFS pull is optional for ordinary S1–S10 inference and packaging workflows.

---

## 10. Basic verification and troubleshooting

| Symptom | Check |
| --- | --- |
| Installer fails at Conda | Install Miniforge/Anaconda; ensure `conda` is on `PATH` |
| Launch fails missing env | Re-run `scripts\install_windows_gui.bat` |
| Doctor fails | Repair with the installer; do not mix damaged pip PySide6 into the dedicated env |
| GUI opens but S2 inference fails | Confirm `sleap-nn` 0.3.x resolves; confirm lab inference model paths in S2 |
| S7 Run blocked | Browse registry root to `models/` (dyadic/focal v1 present) |
| Old S9 package stale / exports blocked | Resolve historical `legacy/models` digests (junction/remap) or package from a new S7/S8 with `models/` |
| Bout videos fail | Confirm FFmpeg in `behavior_suite_gui`; use a currently eligible S9 package; smoke-test a small clip subset |

Smoke after install: launch GUI and confirm Groups **1–4** (S1–S10) are visible. Full GPU S2 is not required to confirm installation of the suite GUI itself.

---

## Related documents

- Root [`README.md`](../../README.md) — supported Windows workflow
- [`s1_s10_mvp_readiness.md`](s1_s10_mvp_readiness.md) — release audit and limitations
- [`s1_s10_p0_remediation_plan.md`](s1_s10_p0_remediation_plan.md) — P0 remediation tracking
