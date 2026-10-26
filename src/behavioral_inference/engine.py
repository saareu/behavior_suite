"""Shared headless inference entry point for Subsystem 07 Sprint 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.adapters.dyadic import run_dyadic_inference
from behavioral_inference.adapters.focal import run_focal_inference
from behavioral_inference.config import BehavioralInferenceConfig
from behavioral_inference.contracts import (
    CONTACT_SPECIALISTS,
    FAMILY_DYADIC,
    FAMILY_FOCAL,
    LOGICAL_DYADIC_DETECTORS,
    LOGICAL_FOCAL_DETECTORS,
    NOSE_TO_BODY_SPECIALISTS,
    STANDALONE_DYADIC_DETECTORS,
    BehavioralInferenceError,
    DetectorSelectionError,
    InferenceError,
    ModelRegistryError,
)
from behavioral_inference.loader import LoadedModel, load_model_bundle
from behavioral_inference.predict import ensure_blob_helper_columns
from behavioral_inference.registry import RegistrySnapshot, discover_model_registry
from behavioral_inference.results import (
    BehavioralInferenceResult,
    DetectorRunMeta,
    build_inference_result,
    empty_predictions_frame,
)


def _unique_mouse_ids(features_df: pd.DataFrame) -> set[int]:
    mice = set(pd.to_numeric(features_df["focal_id"], errors="coerce").dropna().astype(int))
    mice |= set(pd.to_numeric(features_df["partner_id"], errors="coerce").dropna().astype(int))
    return mice


def _resolve_selections(
    config: BehavioralInferenceConfig,
    *,
    dyadic: Sequence[str] | None,
    focal: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dyadic_sel = (
        tuple(config.active_detectors_dyadic)
        if dyadic is None
        else tuple(str(x).strip() for x in dyadic if str(x).strip())
    )
    focal_sel = (
        tuple(config.active_detectors_focal)
        if focal is None
        else tuple(str(x).strip() for x in focal if str(x).strip())
    )
    bad_d = [n for n in dyadic_sel if n not in LOGICAL_DYADIC_DETECTORS]
    bad_f = [n for n in focal_sel if n not in LOGICAL_FOCAL_DETECTORS]
    if bad_d or bad_f:
        raise DetectorSelectionError(
            f"Unknown detectors in selection: dyadic={bad_d}, focal={bad_f}"
        )
    if not dyadic_sel and not focal_sel:
        raise DetectorSelectionError(
            "No detectors selected for inference; refusing an empty successful run"
        )
    return dyadic_sel, focal_sel


def required_bundle_names(
    dyadic_selected: Sequence[str],
    focal_selected: Sequence[str],
    *,
    nose_to_body_mode: str,
) -> tuple[tuple[str, str], ...]:
    """Return ``(family, bundle_name)`` pairs required for the selection."""

    needed: list[tuple[str, str]] = []
    for name in dyadic_selected:
        if name == "contact":
            for specialist in CONTACT_SPECIALISTS:
                needed.append((FAMILY_DYADIC, specialist))
        elif name in STANDALONE_DYADIC_DETECTORS:
            needed.append((FAMILY_DYADIC, name))
        else:
            raise DetectorSelectionError(f"Unknown dyadic detector '{name}'")

    for name in focal_selected:
        if name == "nose_to_body":
            if nose_to_body_mode == "two_stage":
                for specialist in NOSE_TO_BODY_SPECIALISTS:
                    needed.append((FAMILY_FOCAL, specialist))
            elif nose_to_body_mode == "single_stage_fallback":
                needed.append((FAMILY_FOCAL, "nose_to_body"))
            else:
                raise InferenceError(f"Unknown nose_to_body_mode '{nose_to_body_mode}'")
        elif name in ("nose_to_tail", "approach", "retract"):
            needed.append((FAMILY_FOCAL, name))
        else:
            raise DetectorSelectionError(f"Unknown focal detector '{name}'")

    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for item in needed:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def resolve_nose_to_body_mode(
    snapshot: RegistrySnapshot,
    *,
    allow_fallback: bool,
) -> str:
    by_name = snapshot.by_name()
    specialists_ok = all(
        s in by_name and by_name[s].is_loadable for s in NOSE_TO_BODY_SPECIALISTS
    )
    if specialists_ok:
        return "two_stage"
    if allow_fallback:
        fb = by_name.get("nose_to_body")
        if fb is not None and fb.is_loadable:
            return "single_stage_fallback"
    raise InferenceError(
        "nose_to_body requires nose_body_intact + nose_body_occ specialists, "
        "or an allowed loadable single-stage nose_to_body fallback bundle"
    )


def load_models_for_selection(
    snapshot: RegistrySnapshot,
    dyadic_selected: Sequence[str],
    focal_selected: Sequence[str],
    *,
    allow_nose_to_body_fallback: bool,
) -> dict[str, LoadedModel]:
    """Load only the bundles required by the selected detectors (fail-fast)."""

    by_name = snapshot.by_name()
    ntb_mode = "two_stage"
    if "nose_to_body" in focal_selected:
        ntb_mode = resolve_nose_to_body_mode(
            snapshot, allow_fallback=allow_nose_to_body_fallback
        )
    needed = required_bundle_names(
        dyadic_selected,
        focal_selected,
        nose_to_body_mode=ntb_mode,
    )

    loaded: dict[str, LoadedModel] = {}
    for family, name in needed:
        info = by_name.get(name)
        if info is None or info.family != family:
            raise InferenceError(
                f"Required {family} model bundle '{name}' not found in registry. "
                "Selected detector cannot be silently omitted."
            )
        loaded[name] = load_model_bundle(
            info, allow_fallback=(name == "nose_to_body")
        )
    return loaded


def run_behavioral_inference(
    features_df: pd.DataFrame,
    config: BehavioralInferenceConfig,
    *,
    registry: RegistrySnapshot | None = None,
    models: Mapping[str, LoadedModel] | None = None,
    selected_detectors_dyadic: Sequence[str] | None = None,
    selected_detectors_focal: Sequence[str] | None = None,
) -> BehavioralInferenceResult:
    """Run unified dyadic + focal inference and return an in-memory result.

    Parameters
    ----------
    features_df:
        Eligible S6 ``dyadic_features_full`` table (directional).
    config:
        Validated inference configuration (thresholds, adapters, registry).
    registry:
        Optional pre-discovered registry snapshot. Discovered from
        ``config.model_registry`` when models are not supplied.
    models:
        Optional pre-loaded model map keyed by bundle name. When omitted,
        only bundles required by the selection are loaded.
    selected_detectors_dyadic / selected_detectors_focal:
        Optional subset overrides. A selected detector is never silently
        omitted on success.

    Notes
    -----
    Does **not** publish Parquet artifacts by itself — use
    ``run_behavioral_inference_publication`` for Sprint 3 immutable runs.
    Default fail-fast for required model / configuration / feature failures.
    """

    config = config.validated()
    dyadic_sel, focal_sel = _resolve_selections(
        config,
        dyadic=selected_detectors_dyadic,
        focal=selected_detectors_focal,
    )

    if "focal_id" not in features_df.columns or "partner_id" not in features_df.columns:
        raise InferenceError("Feature table missing focal_id / partner_id")

    mice = _unique_mouse_ids(features_df)
    if config.adapters.require_exactly_two_mice and len(mice) != 2:
        raise InferenceError(
            f"S7 MVP requires exactly two tracked mice; found {sorted(mice)}. "
            "Set adapters.require_exactly_two_mice=false only for an explicitly "
            "documented validation exception."
        )

    warnings: list[str] = []
    work = ensure_blob_helper_columns(features_df, warnings=warnings)

    loaded: dict[str, LoadedModel]
    if models is not None:
        loaded = dict(models)
        # Verify every required bundle is present — no silent omission.
        snapshot_for_plan = registry
        if snapshot_for_plan is None:
            # Minimal check against selection without rediscovery when models given.
            required_names: list[str] = []
            if "contact" in dyadic_sel:
                required_names.extend(CONTACT_SPECIALISTS)
            for name in dyadic_sel:
                if name in STANDALONE_DYADIC_DETECTORS:
                    required_names.append(name)
            if "nose_to_body" in focal_sel:
                if all(s in loaded for s in NOSE_TO_BODY_SPECIALISTS):
                    required_names.extend(NOSE_TO_BODY_SPECIALISTS)
                elif "nose_to_body" in loaded:
                    required_names.append("nose_to_body")
                else:
                    required_names.extend(NOSE_TO_BODY_SPECIALISTS)
            for name in focal_sel:
                if name in ("nose_to_tail", "approach", "retract"):
                    required_names.append(name)
            missing = [n for n in required_names if n not in loaded]
            if missing:
                raise InferenceError(
                    f"Preloaded models missing required bundles {missing}; "
                    "refusing silent omission of selected detectors"
                )
    else:
        snapshot = registry
        if snapshot is None:
            snapshot = discover_model_registry(config.model_registry)
        if snapshot.status == "unavailable":
            raise ModelRegistryError(
                "Model registry unavailable; configure model_registry.root explicitly"
            )
        loaded = load_models_for_selection(
            snapshot,
            dyadic_sel,
            focal_sel,
            allow_nose_to_body_fallback=config.adapters.allow_nose_to_body_single_stage_fallback,
        )

    dyadic_preds = empty_predictions_frame()
    dyadic_diags = empty_predictions_frame()
    focal_preds = empty_predictions_frame()
    focal_diags = empty_predictions_frame()
    metas: list[DetectorRunMeta] = []
    routing: dict[str, Any] = {}

    try:
        if dyadic_sel:
            dyadic_preds, dyadic_diags, dyadic_metas, dyadic_warn = run_dyadic_inference(
                work,
                loaded,
                config,
                selected_detectors=dyadic_sel,
            )
            warnings.extend(dyadic_warn)
            metas.extend(dyadic_metas)
        if focal_sel:
            focal_preds, focal_diags, focal_metas, focal_warn, focal_route = run_focal_inference(
                work,
                loaded,
                config,
                selected_detectors=focal_sel,
            )
            warnings.extend(focal_warn)
            metas.extend(focal_metas)
            routing.update(focal_route)
    except BehavioralInferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        if config.adapters.fail_fast_on_detector_error:
            raise InferenceError(f"Inference failed: {type(exc).__name__}: {exc}") from exc
        raise

    frames_to_concat = [f for f in (dyadic_preds, focal_preds) if len(f)]
    predictions = (
        pd.concat(frames_to_concat, ignore_index=True)
        if frames_to_concat
        else empty_predictions_frame()
    )
    diag_to_concat = [f for f in (dyadic_diags, focal_diags) if len(f)]
    diagnostics = (
        pd.concat(diag_to_concat, ignore_index=True)
        if diag_to_concat
        else empty_predictions_frame()
    )

    # Cardinality bookkeeping for metadata / Sprint 3 handoff.
    n_frames = int(work["dense_frame_idx"].nunique()) if "dense_frame_idx" in work.columns else 0
    metadata = {
        "parity_mode": config.parity_mode,
        "model_iteration": config.model_iteration,
        "n_input_rows": int(len(features_df)),
        "n_unique_frames": n_frames,
        "n_mice": len(mice),
        "mouse_ids": sorted(mice),
        "dyadic_rows_per_detector": n_frames,  # K=2 undirected
        "focal_rows_per_detector": n_frames * 2 if len(mice) == 2 else None,
        "adapters": config.adapters.to_settings_dict(),
        "library_versions": next(iter(loaded.values())).library_versions if loaded else {},
        "fail_fast_on_detector_error": config.adapters.fail_fast_on_detector_error,
    }
    if "contact" in dyadic_sel:
        routing.setdefault(
            "contact",
            {
                "mode": "occlusion_gated_overwrite",
                "specialists": list(CONTACT_SPECIALISTS),
            },
        )

    return build_inference_result(
        predictions=predictions,
        diagnostics=diagnostics,
        selected_detectors_dyadic=dyadic_sel,
        selected_detectors_focal=focal_sel,
        detector_meta=metas,
        warnings=warnings,
        routing_diagnostics=routing,
        metadata=metadata,
    )
