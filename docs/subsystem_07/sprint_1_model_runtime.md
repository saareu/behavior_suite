# Subsystem 07 — Sprint 1: Model Registry and Runtime Foundation

**Status:** Implemented (headless; no GUI / no full Pipeline 70/90).  
**Date:** 2026-10-26  
**Authority:** Accepted S7-A audit + S7-B implementation contract.

This sprint establishes that active trained models can be discovered, validated,
loaded, and executed safely under an optional dependency extra. It does **not**
implement dyadic/focal adapters, specialist routing, smoothing, prediction
artifacts, or GUI.

---

## 1. Package structure

```text
src/behavioral_inference/
  __init__.py          # public Sprint 1 API
  contracts.py         # schema stamps, inventories, errors, expected digests
  registry.py          # explicit root discovery + SHA-256 + support classification
  loader.py            # trusted-path joblib load + class-order + predict_proba
  features.py          # ordered feature-matrix validation (NaN keep / Inf reject)
  config.py            # versioned inference YAML + threshold precedence

configs/subsystem_07/
  behavioral_inference_default.yaml
```

Out of scope (intentionally absent): adapters, predict/export/qc/runner,
run discovery, GUI pages.

---

## 2. Optional dependencies

Install inference runtime libraries without forcing them onto S1–S6:

```bat
scripts\install_windows_gui.bat
```

Manual PowerShell-safe form (avoid double-quoted `.[s7]`, which expands `$7`):

```powershell
conda run -n behavior_suite_gui python -m pip install -e '.[s2,s7]'
```

| Package | Constraint | Verified version (this workstation) |
| --- | --- | --- |
| `xgboost` | `>=3.2.0,<3.3` | **3.2.0** |
| `scikit-learn` | `>=1.6.1,<1.7` | **1.6.1** |
| `joblib` | `>=1.3,<2` | **1.5.3** (already present transitively) |

**How pins were chosen (not guessed from newest PyPI):**

1. Default suite env lacked `xgboost` / `sklearn` (documented S7-B gap).
2. Lab `model.joblib` payloads are `xgboost.sklearn.XGBClassifier` pickles.
3. Embedded booster serialization version array reads as `[3, 2, 0]`.
4. Controlled load smoke succeeded for **all** active dyadic/focal joblib
   bundles under `xgboost==3.2.0` + `scikit-learn==1.6.1` + `joblib==1.5.3`.
5. Bundle `train_meta.json` files do **not** record library versions.

Python used for verification: **3.12.11** (conda-forge).

Missing optional deps produce `OptionalDependencyError` with an install hint;
registry discovery and threshold config remain importable without `[s7]`.

---

## 3. Registry contract

Reuse S6 `ModelRegistrySettings` shape:

| Field | Default | Notes |
| --- | --- | --- |
| `root` | `null` | Operator-configured; **never** implied from `legacy/` |
| `dyadic_subdir` | `dyadic/v1` | |
| `focal_subdir` | `focal/v1` | |
| `include_obsolete` | `false` | Obsolete `contact` still reported as unsupported |
| `include_focal` | `true` | |

`discover_model_registry(...)` returns `RegistrySnapshot`:

| `status` | Meaning |
| --- | --- |
| `unavailable` | Root unset or not a directory — **non-fatal** for S1–S6 |
| `ok` / `partial` | Bundles discovered (partial when notes / incompleteness) |

Per bundle (`ModelBundleInfo`), discovery records:

- detector name, family (`dyadic` / `focal`), model version segment
- `bundle_dir`, `model.joblib` path, `feature_spec.json` path
- ordered `feature_cols` / `n_features`
- SHA-256 of `model.joblib`, optional match vs S7-B inventory digests
- serialization label `joblib_xgboost`
- support class: `supported` | `fallback` | `obsolete` | `unsupported` | `incomplete`
- train_meta excerpt when present
- notes

**Support inventory (lab layout):**

| Bundle | Support |
| --- | --- |
| `contact_intact`, `contact_occluded`, `nose_to_nose`, `mutually_disengaged`, `side_by_side` | supported |
| `nose_body_intact`, `nose_body_occ`, `nose_to_tail`, `approach`, `retract` | supported |
| `nose_to_body` | fallback (single-stage) |
| standalone `contact` | **obsolete / unsupported** |
| `nose_body`, `nose_tail` (JSON experts, no joblib) | unsupported |

Discovery does **not** load estimators into memory.

---

## 4. Model-loading behavior

`load_model_bundle(bundle)`:

1. Refuses obsolete / unsupported / incomplete bundles (no silent substitute).
2. Requires optional `[s7]` libraries.
3. Loads only resolved local filesystem paths (`model.joblib`).
4. Requires `predict_proba`.
5. Validates binary `classes_` contains `{0, 1}` and resolves the positive-class
   **index** from label `1` (does not assume column `[:, 1]` when order differs).
6. Preserves bundle identity + SHA-256 on `LoadedModel`.
7. Surfaces clear errors for missing files and broken serialization.

`LoadedModel.positive_proba(X)` returns the validated positive-class column.

---

## 5. Feature-validation contract

`validate_feature_matrix(features, feature_cols)` / `require_feature_matrix(...)`:

| Check | Behavior |
| --- | --- |
| Exact names | Missing columns → error |
| Exact order | Matrix columns follow `feature_cols` order |
| Duplicates | Duplicate spec names → error |
| Numeric dtype | Non-numeric → error (safe coercion attempted once) |
| Shape / row count | Must match `(n_rows, n_features)`; `min_rows` default 1 |
| NaN | **Preserved** (XGBoost native missing); optional hard reject |
| Inf | **Rejected** (not clipped / replaced) — Sprint 1 hardening |

Distinction from S6 `model_compat`: S6 reports schema presence/dtype warnings
only; S7 validation gates **runtime executability**.

No inference-time imputation.

---

## 6. Threshold configuration

Tracked defaults: `configs/subsystem_07/behavioral_inference_default.yaml`
(values = legacy `dyadic_inference.yaml` + `90_focal_inference.yaml`).

**Precedence (binding):**

1. Explicit user override (`user_threshold_overrides`)
2. Inference configuration `working_thresholds`
3. **Hard fail** if unresolved

Never silently substitutes `0.5`, bundle `threshold_working`, or other
undocumented defaults. Opt-in `allow_default_threshold_0_5=true` exists only for
documented Pipeline 70/90 parity experiments.

Specialist keys resolve nested YAML:

- `contact_intact` / `contact_occluded` ← `contact.intact` / `contact.occluded`
- `nose_body_intact` / `nose_body_occ` ← `nose_to_body.intact` / `nose_to_body.occluded`

`ResolvedThreshold` records `(detector_key, value, source)`.

Routing and smoothing are **not** implemented in Sprint 1.

---

## 7. Models successfully loaded (lab smoke)

Registry root used: workstation `legacy/models` (explicitly configured; not an
implicit runtime dependency for other clones).

| Bundle | Features | `classes_` | `predict_proba` |
| --- | --- | --- | --- |
| dyadic `contact_intact` | 139 | `[0, 1]` | OK |
| dyadic `contact_occluded` | 34 | `[0, 1]` | OK |
| dyadic `nose_to_nose` | 44 | `[0, 1]` | OK |
| dyadic `mutually_disengaged` | 79 | `[0, 1]` | OK |
| dyadic `side_by_side` | 26 | `[0, 1]` | OK |
| focal `nose_body_intact` | 109 | `[0, 1]` | OK |
| focal `nose_body_occ` | 26 | `[0, 1]` | OK |
| focal `nose_to_tail` | 133 | `[0, 1]` | OK |
| focal `approach` | 71 | `[0, 1]` | OK |
| focal `retract` | 39 | `[0, 1]` | OK |
| focal `nose_to_body` (fallback) | 49 | `[0, 1]` | OK |

Obsolete `contact` correctly refused. SHA-256 digests matched the S7-B inventory.

Feature sample: E-session
`test_cases/E_full_session_integration/.../dyadic_features_full.parquet`
(head rows) when present; otherwise NaN matrices of the correct width.

Finite probabilities observed for all successful smoke predictions; no runtime
warnings recorded during the controlled load.

Automated coverage: `tests/behavioral_inference/` (registry, loader, features,
thresholds, real-model smoke marker `s7_model_smoke`).

---

## 8. Remaining Sprint 2 prerequisites

1. Dyadic adapter: canonical unordered-pair collapse (lower ID as focal).
2. Focal adapter: directional pass-through (no collapse).
3. Contact dual-specialist routing + fused `contact` outputs.
4. Nose-to-body two-stage routing + optional 5-frame rolling median under
   `legacy_parity`.
5. Shared predict / binarize using resolved thresholds.
6. Parity tests vs Pipeline 70/90 wide predictions on a frozen snippet /
   E-session.
7. Still out of Sprint 2: artifact publication (Sprint 3), GUI (Sprint 4), S8.

---

## 9. Environment / compatibility limitations

- Model binaries remain lab-local; not committed.
- CI clones without a configured registry skip real-model smoke via
  `pytest.skip` when `legacy/models` is absent.
- Environments without `[s7]` skip loader paths that need unpickle; unit tests
  that mock `joblib.load` still run when deps are present.
- Library pins are verified against current lab pickles; retrained bundles may
  require re-measuring compatible `xgboost` / `sklearn` versions.
