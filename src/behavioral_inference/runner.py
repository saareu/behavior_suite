"""Subsystem 07 end-to-end headless runner: eligible S6 → published predictions.

Pipeline:
  select/validate eligible S6
  → freeze config + create immutable run directory
  → load required trusted models
  → run Sprint 2 inference engine (in memory)
  → validate predictions + derive wide views
  → QC + model manifest + provenance
  → revalidate S6 / model digests
  → publish artifacts atomically
  → record completion or failure

Partial science artifacts are never marked completed / eligible.
Scientific inference remains independent of file publication.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from behavioral_inference.config import (
    BehavioralInferenceConfig,
    load_behavioral_inference_config,
)
from behavioral_inference.contracts import (
    BEHAVIORAL_INFERENCE_DIRNAME,
    DYADIC_WIDE_FILENAME,
    FOCAL_WIDE_FILENAME,
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    SPECIALIST_PREDICTIONS_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    ProvenanceError,
    PublicationError,
)
from behavioral_inference.engine import run_behavioral_inference
from behavioral_inference.export import (
    build_canonical_predictions_table,
    build_dyadic_predictions_wide,
    build_focal_predictions_wide,
    build_model_manifest,
    build_specialist_predictions_table,
    collect_feature_spec_digests,
    write_optional_predictions_parquet,
    write_predictions_parquet,
)
from behavioral_inference.qc import (
    QcSummary,
    assert_structural_qc,
    evaluate_behavioral_inference_qc,
)
from behavioral_inference.results import BehavioralInferenceResult
from behavioral_inference.validate import annotate_score_origin, validate_canonical_predictions
from combined_features.contracts import DYADIC_FEATURES_FILENAME
from combined_features.export import load_dyadic_features_table
from combined_features.runner import (
    is_s6_run_eligible_for_downstream,
    validate_s6_run_for_downstream,
)
from preprocess.models import OperationProgress
from project.file_digest import sha256_file

ProgressCallback = Callable[[OperationProgress], None]


@dataclass(frozen=True, slots=True)
class BehavioralInferenceRequest:
    """Inputs for one Subsystem 07 behavioral-inference run."""

    session_root: Path | None = None
    s6_run_dir: Path | None = None
    output_root: Path | None = None
    config: BehavioralInferenceConfig | None = None
    config_path: Path | None = None
    selected_detectors_dyadic: Sequence[str] | None = None
    selected_detectors_focal: Sequence[str] | None = None
    run_purpose: str = "development"
    timestamp: str | None = None
    exp_name: str | None = None


@dataclass(frozen=True, slots=True)
class BehavioralInferenceRunResult:
    """Summary returned after an S7 run directory is recorded."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    qc_path: Path
    model_manifest_path: Path
    predictions_path: Path | None
    specialist_predictions_path: Path | None
    dyadic_wide_path: Path | None
    focal_wide_path: Path | None
    session_root: Path
    source_s6_run_id: str | None
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    qc: Mapping[str, Any] | None = None
    model_manifest: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _RunPaths:
    run_dir: Path
    predictions: Path
    specialist_predictions: Path
    dyadic_wide: Path
    focal_wide: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path
    qc: Path
    model_manifest: Path


def run_behavioral_inference_publication(
    request: BehavioralInferenceRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> BehavioralInferenceRunResult:
    """Run Sprint 3 publication pipeline over the Sprint 2 inference engine."""

    purpose = request.run_purpose.strip()
    if not purpose:
        raise PublicationError("run_purpose must not be empty.")

    config = _resolve_config(request)
    session_root, s6_run_dir, s6_info = _resolve_s6(request)
    features_path = s6_run_dir / DYADIC_FEATURES_FILENAME
    features_df = load_dyadic_features_table(features_path)
    features_sha = _sha256(features_path)

    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    run_id = f"behavioral_inference__{timestamp}"
    paths = _resolve_run_paths(
        session_root=session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _report_progress(
        progress_callback,
        phase="starting",
        message="Validating S6 source and creating S7 run directory",
        completed_units=0,
        total_units=None,
        is_indeterminate=True,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            session_root / "preprocess",
            session_root / "pose_inference",
            session_root / "tracking_correction",
            session_root / "pose_finalization",
            session_root / "blob_features",
            session_root / "combined_features",
            s6_run_dir,
            features_path,
        ),
    )
    if paths.run_dir.exists():
        raise PublicationError(f"S7 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PublicationError(
            f"S7 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    s6_run_id = str(s6_info.get("s6_run_id") or s6_run_dir.name)
    provenance = _s6_provenance(
        s6_run_dir=s6_run_dir,
        s6_run_id=s6_run_id,
        features_path=features_path,
        features_sha256=features_sha,
        s6_info=s6_info,
        session_root=session_root,
    )
    settings_snapshot = _settings_payload(
        config=config,
        status=STATUS_RUNNING,
        s6_run_id=s6_run_id,
        selected_dyadic=request.selected_detectors_dyadic,
        selected_focal=request.selected_detectors_focal,
    )
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=session_root,
            paths=paths,
            config=config,
            provenance=provenance,
            result=None,
            qc_summary=None,
            model_manifest=None,
            error_message=None,
            science_written=False,
            settings_snapshot=settings_snapshot,
        ),
    )
    _write_yaml(paths.settings_used, settings_snapshot)

    inference_result: BehavioralInferenceResult | None = None
    qc_summary: QcSummary | None = None
    model_manifest: dict[str, Any] | None = None
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    science_written = False
    predictions_table: pd.DataFrame | None = None
    specialist_table: pd.DataFrame | None = None
    dyadic_wide: pd.DataFrame | None = None
    focal_wide: pd.DataFrame | None = None

    try:
        _report_progress(
            progress_callback,
            phase="inference",
            message="Running behavioral inference engine",
            completed_units=0,
            total_units=None,
            is_indeterminate=True,
        )
        inference_result = run_behavioral_inference(
            features_df,
            config,
            selected_detectors_dyadic=request.selected_detectors_dyadic,
            selected_detectors_focal=request.selected_detectors_focal,
        )

        validation_metrics = validate_canonical_predictions(
            inference_result,
            n_unique_frames=int(inference_result.metadata.get("n_unique_frames") or 0)
            or None,
            require_exactly_two_mice=config.adapters.require_exactly_two_mice,
        )
        predictions_table = build_canonical_predictions_table(
            inference_result,
            session_root=session_root,
            source_s6_run_id=s6_run_id,
            n_unique_frames=validation_metrics.get("n_unique_frames"),
            require_exactly_two_mice=config.adapters.require_exactly_two_mice,
        )
        specialist_table = build_specialist_predictions_table(
            inference_result,
            session_root=session_root,
            source_s6_run_id=s6_run_id,
        )

        annotated_preds = annotate_score_origin(inference_result.predictions)
        annotated_diags = (
            annotate_score_origin(inference_result.diagnostics)
            if not inference_result.diagnostics.empty
            else inference_result.diagnostics
        )
        exp_name = request.exp_name or _infer_exp_name(features_df, session_root)

        if config.adapters.emit_wide_views:
            if inference_result.selected_detectors_dyadic:
                dyadic_wide = build_dyadic_predictions_wide(
                    predictions=annotated_preds,
                    diagnostics=annotated_diags,
                    selected_detectors=inference_result.selected_detectors_dyadic,
                    exp_name=exp_name,
                )
            if inference_result.selected_detectors_focal:
                focal_wide = build_focal_predictions_wide(
                    predictions=annotated_preds,
                    diagnostics=annotated_diags,
                    selected_detectors=inference_result.selected_detectors_focal,
                    features_df=features_df,
                    exp_name=exp_name,
                )

        feature_digests = collect_feature_spec_digests(inference_result)
        registry_root = (
            str(config.model_registry.root.resolve())
            if config.model_registry.root is not None
            else None
        )
        model_manifest = build_model_manifest(
            inference_result,
            config_settings=settings_snapshot,
            registry_root=registry_root,
            feature_spec_digests=feature_digests,
        )

        qc_summary = evaluate_behavioral_inference_qc(
            result=inference_result,
            predictions=predictions_table,
            specialist_predictions=specialist_table,
            dyadic_wide=dyadic_wide,
            focal_wide=focal_wide,
            validation_metrics=validation_metrics,
            source_provenance=provenance,
            model_manifest=model_manifest,
            source_warnings=(),
        )
        assert_structural_qc(qc_summary)

        # Revalidate S6 eligibility + digests immediately before publish.
        _revalidate_s6_before_publish(
            s6_run_dir=s6_run_dir,
            recorded_features_sha=features_sha,
            features_path=features_path,
        )
        _revalidate_models_before_publish(model_manifest)

        _report_progress(
            progress_callback,
            phase="writing",
            message="Writing prediction artifacts and QC",
            completed_units=1,
            total_units=1,
            is_indeterminate=False,
        )
        write_predictions_parquet(paths.predictions, predictions_table)
        write_optional_predictions_parquet(paths.specialist_predictions, specialist_table)
        if dyadic_wide is not None and not dyadic_wide.empty:
            write_predictions_parquet(paths.dyadic_wide, dyadic_wide)
        if focal_wide is not None and not focal_wide.empty:
            write_predictions_parquet(paths.focal_wide, focal_wide)
        _write_json(paths.qc, qc_summary.to_metadata())
        _write_json(paths.model_manifest, model_manifest)
        science_written = True

        if qc_summary.has_warnings:
            status = STATUS_COMPLETED_WITH_WARNINGS
            warnings = qc_summary.warnings
        else:
            status = STATUS_COMPLETED
            warnings = ()
        success = True
        _report_progress(
            progress_callback,
            phase="completed",
            message=f"Behavioral inference finished ({status})",
            completed_units=1,
            total_units=1,
            is_indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        _unlink_science_artifacts(paths)
        science_written = False

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        session_root=session_root,
        config=config,
        provenance=provenance,
        result=inference_result,
        qc_summary=qc_summary,
        model_manifest=model_manifest,
        error_message=error_message,
        science_written=science_written,
        settings_snapshot=settings_snapshot,
        dyadic_wide_written=bool(dyadic_wide is not None and not dyadic_wide.empty and science_written),
        focal_wide_written=bool(focal_wide is not None and not focal_wide.empty and science_written),
    )

    if not success:
        raise PublicationError(error_message or "Behavioral inference publication failed.")

    return BehavioralInferenceRunResult(
        success=True,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        qc_path=paths.qc,
        model_manifest_path=paths.model_manifest,
        predictions_path=paths.predictions if science_written else None,
        specialist_predictions_path=(
            paths.specialist_predictions if science_written else None
        ),
        dyadic_wide_path=(
            paths.dyadic_wide
            if science_written and paths.dyadic_wide.is_file()
            else None
        ),
        focal_wide_path=(
            paths.focal_wide
            if science_written and paths.focal_wide.is_file()
            else None
        ),
        session_root=session_root,
        source_s6_run_id=s6_run_id,
        warnings=warnings,
        error_message=None,
        qc=qc_summary.to_metadata() if qc_summary is not None else None,
        model_manifest=model_manifest,
    )


def is_s7_run_eligible_for_downstream(s7_run_dir: Path | str) -> bool:
    """Return True when a completed S7 run still matches recorded S6 + models."""

    try:
        validate_s7_run_for_downstream(s7_run_dir)
    except (PublicationError, ProvenanceError):
        return False
    return True


def validate_s7_run_for_downstream(s7_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed S7 run for downstream use (staleness-aware).

    Historical ``status`` on disk is never rewritten. Eligibility becomes
    false when the recorded S6 source or model digests no longer match.
    """

    run_dir = Path(s7_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ProvenanceError(f"S7 run directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise ProvenanceError(f"S7 run_meta.json is missing: {meta_path}")
    meta = _load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise ProvenanceError(
            f"S7 run is not completed (status={status or 'missing'!r})."
        )

    predictions = run_dir / PREDICTIONS_FILENAME
    if not predictions.is_file():
        raise ProvenanceError(
            f"Completed S7 run is missing {PREDICTIONS_FILENAME}: {predictions}"
        )

    source_s6 = meta.get("source_s6")
    if not isinstance(source_s6, Mapping):
        raise ProvenanceError("S7 run_meta.json is missing source_s6 provenance.")

    s6_run_dir_text = _optional_text(source_s6.get("run_dir"))
    if s6_run_dir_text is None:
        raise ProvenanceError("S7 run_meta.json source_s6.run_dir is missing.")
    s6_run_dir = Path(s6_run_dir_text)
    if not is_s6_run_eligible_for_downstream(s6_run_dir):
        raise ProvenanceError(
            "S7 source S6 run is no longer eligible for downstream use."
        )

    recorded_feat_sha = _optional_text(source_s6.get("dyadic_features_sha256"))
    features_path = Path(
        _optional_text(source_s6.get("dyadic_features"))
        or str(s6_run_dir / DYADIC_FEATURES_FILENAME)
    )
    if recorded_feat_sha is not None:
        if not features_path.is_file():
            raise ProvenanceError(
                f"Recorded S6 dyadic features missing: {features_path}"
            )
        current = _sha256(features_path)
        if current.lower() != recorded_feat_sha.lower():
            raise ProvenanceError(
                "S7 source S6 dyadic_features SHA256 no longer matches the recorded digest."
            )

    # Also re-check S6 validate for nested S4/S5 lineage.
    validate_s6_run_for_downstream(s6_run_dir)

    model_staleness = assess_model_source_staleness(run_dir)
    if model_staleness["stale"]:
        raise ProvenanceError(
            "S7 model artifacts are stale or missing vs model_manifest.json: "
            + "; ".join(model_staleness["reasons"])
        )

    return {
        "s7_run_dir": run_dir,
        "status": status,
        "session_root": _optional_text(meta.get("session_root")),
        "source_s6_run_dir": s6_run_dir.resolve(),
        "source_s6_run_id": _optional_text(source_s6.get("run_id")),
        "dyadic_features_sha256": recorded_feat_sha,
        "model_staleness": model_staleness,
        "detector_coverage": (
            meta.get("detectors_selected")
            if isinstance(meta.get("detectors_selected"), Mapping)
            else {}
        ),
    }


def assess_model_source_staleness(s7_run_dir: Path | str) -> dict[str, Any]:
    """Detect missing/changed model binaries, feature specs, or frozen settings.

    Policy when a model registry is *moved* but identical artifact digests remain
    available at the paths recorded in ``model_manifest.json`` (or resolvable via
    the same SHA-256 content): the run is **not** stale — digests are authoritative,
    absolute registry roots are not. Silent substitution of a different model
    version (digest mismatch) is always stale.
    """

    run_dir = Path(s7_run_dir).expanduser().resolve()
    reasons: list[str] = []
    checked: list[dict[str, Any]] = []

    manifest_path = run_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return {
            "stale": True,
            "reasons": [f"missing {MODEL_MANIFEST_FILENAME}"],
            "checked": [],
        }
    manifest = _load_json_object(manifest_path)
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        return {
            "stale": True,
            "reasons": ["model_manifest.bundles missing or invalid"],
            "checked": [],
        }

    for item in bundles:
        if not isinstance(item, Mapping):
            continue
        name = item.get("specialist_name") or item.get("detector_name")
        recorded_sha = _optional_text(item.get("model_sha256"))
        bundle_dir_text = _optional_text(item.get("model_bundle_dir"))
        entry: dict[str, Any] = {
            "name": name,
            "recorded_sha256": recorded_sha,
            "bundle_dir": bundle_dir_text,
        }
        if recorded_sha is None:
            # Fused logical outputs may omit a single model sha — skip.
            if item.get("output_role") in ("logical_fused",):
                entry["status"] = "skipped_fused"
                checked.append(entry)
                continue
            reasons.append(f"{name}: missing recorded model_sha256")
            entry["status"] = "missing_digest"
            checked.append(entry)
            continue
        if bundle_dir_text is None:
            reasons.append(f"{name}: missing model_bundle_dir")
            entry["status"] = "missing_path"
            checked.append(entry)
            continue
        model_path = Path(bundle_dir_text) / MODEL_FILENAME
        if not model_path.is_file():
            reasons.append(f"{name}: model.joblib missing at {model_path}")
            entry["status"] = "missing_file"
            checked.append(entry)
            continue
        current = _sha256(model_path)
        entry["current_sha256"] = current
        if current.lower() != recorded_sha.lower():
            reasons.append(
                f"{name}: model SHA256 changed ({recorded_sha[:12]}… → {current[:12]}…)"
            )
            entry["status"] = "digest_mismatch"
        else:
            entry["status"] = "ok"
            # Feature spec digest when recorded.
            recorded_feat = _optional_text(item.get("feature_spec_sha256"))
            if recorded_feat is not None:
                feat_path = Path(bundle_dir_text) / "feature_spec.json"
                if not feat_path.is_file():
                    reasons.append(f"{name}: feature_spec.json missing")
                    entry["status"] = "missing_feature_spec"
                else:
                    feat_sha = _sha256(feat_path)
                    entry["feature_spec_sha256"] = feat_sha
                    if feat_sha.lower() != recorded_feat.lower():
                        reasons.append(f"{name}: feature_spec.json SHA256 changed")
                        entry["status"] = "feature_spec_mismatch"
        checked.append(entry)

    # Frozen settings snapshot must remain readable (immutable); content drift of
    # the external default YAML is irrelevant because settings_used.yaml is the
    # run dependency. We only flag if settings_used is missing.
    settings_path = run_dir / SETTINGS_USED_FILENAME
    if not settings_path.is_file():
        reasons.append(f"missing immutable {SETTINGS_USED_FILENAME}")

    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "checked": checked,
        "policy": (
            "Digests are authoritative. A moved registry root with identical "
            "model.joblib / feature_spec.json digests at the recorded bundle "
            "paths (or rewritten paths with matching digests) is not stale. "
            "Digest mismatch is always stale; never silently substitute another "
            "model version."
        ),
    }


def _resolve_s6(
    request: BehavioralInferenceRequest,
) -> tuple[Path, Path, dict[str, Any]]:
    if request.session_root is None and request.s6_run_dir is None:
        raise PublicationError("session_root or s6_run_dir is required.")

    if request.s6_run_dir is not None:
        s6_run_dir = Path(request.s6_run_dir).expanduser().resolve()
        info = validate_s6_run_for_downstream(s6_run_dir)
        if request.session_root is not None:
            session_root = Path(request.session_root).expanduser().resolve()
        else:
            session_root = Path(str(info["session_root"])).resolve()
        # Attach run id from meta when available.
        meta = _load_json_object(s6_run_dir / "run_meta.json")
        info = dict(info)
        info["s6_run_id"] = _optional_text(meta.get("run_id")) or s6_run_dir.name
        return session_root, s6_run_dir, info

    if request.session_root is None:
        raise PublicationError("session_root is required when s6_run_dir is omitted.")
    session_root = Path(request.session_root).expanduser().resolve()
    from behavioral_inference.run_discovery import find_latest_eligible_s6_for_s7

    summary = find_latest_eligible_s6_for_s7(session_root)
    if summary is None:
        raise ProvenanceError(
            f"No currently eligible S6 run found under {session_root}"
        )
    s6_run_dir = summary.run_dir
    info = validate_s6_run_for_downstream(s6_run_dir)
    info = dict(info)
    info["s6_run_id"] = summary.run_id
    return session_root, s6_run_dir, info


def _resolve_config(request: BehavioralInferenceRequest) -> BehavioralInferenceConfig:
    if request.config is not None:
        return request.config.validated()
    if request.config_path is not None:
        return load_behavioral_inference_config(request.config_path)
    return load_behavioral_inference_config(None)


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / BEHAVIORAL_INFERENCE_DIRNAME
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        predictions=run_dir / PREDICTIONS_FILENAME,
        specialist_predictions=run_dir / SPECIALIST_PREDICTIONS_FILENAME,
        dyadic_wide=run_dir / DYADIC_WIDE_FILENAME,
        focal_wide=run_dir / FOCAL_WIDE_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
        qc=run_dir / QC_FILENAME,
        model_manifest=run_dir / MODEL_MANIFEST_FILENAME,
    )


def _s6_provenance(
    *,
    s6_run_dir: Path,
    s6_run_id: str,
    features_path: Path,
    features_sha256: str,
    s6_info: Mapping[str, Any],
    session_root: Path,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    meta_path = s6_run_dir / "run_meta.json"
    if meta_path.is_file():
        with suppress(OSError, json.JSONDecodeError, PublicationError):
            meta = _load_json_object(meta_path)
    source_s4 = meta.get("source_s4") if isinstance(meta.get("source_s4"), Mapping) else {}
    source_s5 = meta.get("source_s5") if isinstance(meta.get("source_s5"), Mapping) else {}
    return {
        "run_dir": _path_text(s6_run_dir),
        "run_id": s6_run_id,
        "session_root": _path_text(session_root),
        "dyadic_features": _path_text(features_path),
        "dyadic_features_sha256": features_sha256,
        "eligibility_at_start": True,
        "s6_status": _optional_text(meta.get("status")),
        "source_s4": dict(source_s4) if isinstance(source_s4, Mapping) else {},
        "source_s5": dict(source_s5) if isinstance(source_s5, Mapping) else {},
        "prepared_sync_sha256": s6_info.get("prepared_sync_sha256"),
        "final_pose_sha256": s6_info.get("final_pose_sha256"),
        "blob_features_sha256": s6_info.get("blob_features_sha256"),
    }


def _revalidate_s6_before_publish(
    *,
    s6_run_dir: Path,
    recorded_features_sha: str,
    features_path: Path,
) -> None:
    if not is_s6_run_eligible_for_downstream(s6_run_dir):
        raise ProvenanceError(
            "S6 source became ineligible before S7 publish; refusing partial publish."
        )
    current = _sha256(features_path)
    if current.lower() != recorded_features_sha.lower():
        raise ProvenanceError(
            "S6 dyadic_features SHA256 changed during S7 processing."
        )
    validate_s6_run_for_downstream(s6_run_dir)


def _revalidate_models_before_publish(manifest: Mapping[str, Any]) -> None:
    """Ensure recorded model digests still match files about to be claimed."""

    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        raise ProvenanceError("model_manifest.bundles missing before publish")
    for item in bundles:
        if not isinstance(item, Mapping):
            continue
        if item.get("output_role") == "logical_fused" and item.get("model_sha256") is None:
            continue
        recorded = _optional_text(item.get("model_sha256"))
        bundle_dir = _optional_text(item.get("model_bundle_dir"))
        name = item.get("specialist_name") or item.get("detector_name")
        if recorded is None or bundle_dir is None:
            raise ProvenanceError(
                f"Cannot publish: bundle '{name}' missing sha256 or path"
            )
        model_path = Path(bundle_dir) / MODEL_FILENAME
        if not model_path.is_file():
            raise ProvenanceError(
                f"Cannot publish: model missing for '{name}' at {model_path}"
            )
        if _sha256(model_path).lower() != recorded.lower():
            raise ProvenanceError(
                f"Cannot publish: model digest changed for '{name}' during run"
            )


def _unlink_science_artifacts(paths: _RunPaths) -> None:
    for path in (
        paths.predictions,
        paths.specialist_predictions,
        paths.dyadic_wide,
        paths.focal_wide,
        paths.qc,
        paths.model_manifest,
    ):
        if path.exists():
            with suppress(OSError):
                path.unlink()


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    session_root: Path,
    config: BehavioralInferenceConfig,
    provenance: Mapping[str, Any],
    result: BehavioralInferenceResult | None,
    qc_summary: QcSummary | None,
    model_manifest: Mapping[str, Any] | None,
    error_message: str | None,
    science_written: bool,
    settings_snapshot: Mapping[str, Any],
    dyadic_wide_written: bool,
    focal_wide_written: bool,
) -> None:
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            session_root=session_root,
            paths=paths,
            config=config,
            provenance=provenance,
            result=result,
            qc_summary=qc_summary,
            model_manifest=model_manifest,
            error_message=error_message,
            science_written=science_written,
            settings_snapshot=settings_snapshot,
            dyadic_wide_written=dyadic_wide_written,
            focal_wide_written=focal_wide_written,
        ),
    )
    snapshot = dict(settings_snapshot)
    snapshot["status"] = status
    _write_yaml(paths.settings_used, snapshot)
    if qc_summary is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        _write_json(paths.qc, qc_summary.to_metadata())
    if model_manifest is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        _write_json(paths.model_manifest, model_manifest)
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        session_root=session_root,
        paths=paths,
        config=config,
        provenance=provenance,
        result=result,
        qc_summary=qc_summary,
        model_manifest=model_manifest,
        error_message=error_message,
        science_written=science_written,
    )


def _run_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    session_root: Path,
    paths: _RunPaths,
    config: BehavioralInferenceConfig,
    provenance: Mapping[str, Any],
    result: BehavioralInferenceResult | None,
    qc_summary: QcSummary | None,
    model_manifest: Mapping[str, Any] | None,
    error_message: str | None,
    science_written: bool,
    settings_snapshot: Mapping[str, Any],
    dyadic_wide_written: bool = False,
    focal_wide_written: bool = False,
) -> dict[str, Any]:
    detectors_selected = {
        "dyadic": list(
            result.selected_detectors_dyadic
            if result is not None
            else settings_snapshot.get("active_detectors", {}).get("dyadic", [])
            if isinstance(settings_snapshot.get("active_detectors"), Mapping)
            else list(config.active_detectors_dyadic)
        ),
        "focal": list(
            result.selected_detectors_focal
            if result is not None
            else settings_snapshot.get("active_detectors", {}).get("focal", [])
            if isinstance(settings_snapshot.get("active_detectors"), Mapping)
            else list(config.active_detectors_focal)
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "session_root": _path_text(session_root),
        "source_s6": dict(provenance),
        "registry": config.model_registry.to_settings_dict(),
        "detectors_selected": detectors_selected,
        "parity_mode": config.parity_mode,
        "adapters": config.adapters.to_settings_dict(),
        "libraries": (
            dict(result.metadata.get("library_versions", {})) if result is not None else {}
        ),
        "settings": dict(settings_snapshot),
        "outputs": {
            "predictions_parquet": (
                _path_text(paths.predictions) if science_written else None
            ),
            "specialist_predictions_parquet": (
                _path_text(paths.specialist_predictions) if science_written else None
            ),
            "dyadic_predictions_wide_parquet": (
                _path_text(paths.dyadic_wide) if dyadic_wide_written else None
            ),
            "focal_predictions_wide_parquet": (
                _path_text(paths.focal_wide) if focal_wide_written else None
            ),
            "settings_used": _path_text(paths.settings_used),
            "processing_log": _path_text(paths.processing_log),
            "qc": _path_text(paths.qc),
            "model_manifest": _path_text(paths.model_manifest),
        },
        "qc": qc_summary.to_metadata() if qc_summary is not None else None,
        "model_manifest_summary": (
            {
                "n_bundles": len(model_manifest.get("bundles", [])),
                "registry_root": model_manifest.get("registry_root"),
            }
            if model_manifest is not None
            else None
        ),
    }
    if result is not None:
        payload["inference_summary"] = result.to_summary_dict()
    if error_message is not None:
        payload["error"] = error_message
    return payload


def _settings_payload(
    *,
    config: BehavioralInferenceConfig,
    status: str,
    s6_run_id: str,
    selected_dyadic: Sequence[str] | None,
    selected_focal: Sequence[str] | None,
) -> dict[str, Any]:
    payload = config.to_settings_dict()
    payload["status"] = status
    payload["source_s6_run_id"] = s6_run_id
    if selected_dyadic is not None:
        payload.setdefault("active_detectors", {})
        if isinstance(payload["active_detectors"], dict):
            payload["active_detectors"]["dyadic"] = list(selected_dyadic)
        payload["selected_detectors_dyadic_override"] = list(selected_dyadic)
    if selected_focal is not None:
        payload.setdefault("active_detectors", {})
        if isinstance(payload["active_detectors"], dict):
            payload["active_detectors"]["focal"] = list(selected_focal)
        payload["selected_detectors_focal_override"] = list(selected_focal)
    return payload


def _write_processing_log(
    *,
    path: Path,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    session_root: Path,
    paths: _RunPaths,
    config: BehavioralInferenceConfig,
    provenance: Mapping[str, Any],
    result: BehavioralInferenceResult | None,
    qc_summary: QcSummary | None,
    model_manifest: Mapping[str, Any] | None,
    error_message: str | None,
    science_written: bool,
) -> None:
    lines = [
        "Subsystem 07 behavioral inference",
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"created_at: {created_at}",
        f"completed_at: {completed_at}",
        f"status: {status}",
        f"session_root: {_path_text(session_root)}",
        f"source_s6_run_id: {provenance.get('run_id')}",
        f"source_s6_run_dir: {provenance.get('run_dir')}",
        f"dyadic_features_sha256: {provenance.get('dyadic_features_sha256')}",
        f"parity_mode: {config.parity_mode}",
        (
            "model_registry.root: "
            + (
                str(config.model_registry.root)
                if config.model_registry.root is not None
                else "null"
            )
        ),
    ]
    if result is not None:
        lines.append(
            "selected_dyadic: " + ",".join(result.selected_detectors_dyadic)
        )
        lines.append(
            "selected_focal: " + ",".join(result.selected_detectors_focal)
        )
        lines.append(f"n_prediction_rows: {len(result.predictions)}")
        lines.append(f"n_diagnostic_rows: {len(result.diagnostics)}")
        for warning in result.warnings:
            lines.append(f"inference.warning: {warning}")
        for meta in result.detector_meta:
            key = meta.specialist_name or meta.detector_name
            lines.append(
                f"detector[{key}]: n_rows={meta.n_rows} n_scored={meta.n_scored} "
                f"n_positive={meta.n_positive} threshold={meta.threshold} "
                f"source={meta.threshold_source}"
            )
    if qc_summary is not None:
        lines.append(f"qc.structural_ok: {qc_summary.structural_ok}")
        lines.append(f"qc.warning_count: {len(qc_summary.warnings)}")
        for warning in qc_summary.warnings:
            lines.append(f"qc.warning: {warning}")
        for failure in qc_summary.failures:
            lines.append(f"qc.failure: {failure}")
    if model_manifest is not None:
        lines.append(
            f"model_manifest.n_bundles: {len(model_manifest.get('bundles', []))}"
        )
    lines.append(
        "predictions_parquet: "
        + (_path_text(paths.predictions) if science_written else "not_written")
    )
    if error_message:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reject_output_overlap(run_dir: Path, protected: Sequence[Path]) -> None:
    resolved_run = run_dir.resolve()
    for protected_path in protected:
        resolved_protected = Path(protected_path).resolve()
        if (
            resolved_run == resolved_protected
            or _is_relative_to(resolved_run, resolved_protected)
            or _is_relative_to(resolved_protected, resolved_run)
        ):
            raise PublicationError(
                "S7 run directory must not overlap S1–S6 artifacts: "
                f"{resolved_run}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _infer_exp_name(features_df: pd.DataFrame, session_root: Path) -> str:
    if "exp_name" in features_df.columns and len(features_df):
        value = features_df["exp_name"].iloc[0]
        if value is not None and str(value).strip():
            return str(value)
    return session_root.name


def _report_progress(
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed_units: int | None,
    total_units: int | None,
    is_indeterminate: bool,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        OperationProgress(
            phase=phase,
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            total_is_estimate=False,
            is_indeterminate=is_indeterminate,
            processed_video_time_sec=None,
        )
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Could not load JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicationError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value.strip())
    if not cleaned:
        raise PublicationError("timestamp is empty after sanitization.")
    return cleaned


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
