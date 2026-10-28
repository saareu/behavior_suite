"""Subsystem 06 end-to-end headless runner: eligible S4+S5 → dyadic features.

Pipeline:
  resolve/validate joint S4/S5 pair
  → create immutable run directory + settings snapshot
  → compute complete in-memory S6 features
  → structural QC + model compatibility
  → revalidate upstream provenance
  → publish artifacts safely
  → record completion or failure

Partial or structurally invalid parquet outputs are never marked completed.
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

import yaml

from combined_features.config import (
    CombinedFeaturesConfig,
    load_combined_features_config,
)
from combined_features.contracts import (
    COMBINED_FEATURES_DIRNAME,
    DYADIC_FEATURES_FILENAME,
    DYADIC_FEATURES_SCHEMA_VERSION,
    MODEL_COMPAT_FILENAME,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    CombinedFeaturesError,
)
from combined_features.export import build_dyadic_features_table, write_dyadic_features_parquet
from combined_features.model_compat import (
    build_model_compat_report,
    write_model_compat_json,
)
from combined_features.processing import (
    CombinedFeaturesComputation,
    compute_combined_features_from_pair,
)
from combined_features.qc import QcSummary, assert_structural_qc, evaluate_combined_features_qc
from combined_features.upstream import JointEligiblePair, validate_joint_s4_s5_pair
from preprocess.models import OperationProgress
from project.file_digest import sha256_file
from project.validation_memo import (
    artifact_fingerprint,
    memoized_validation,
    run_dir_fingerprint,
)

ProgressCallback = Callable[[OperationProgress], None]


@dataclass(frozen=True, slots=True)
class CombinedFeaturesRequest:
    """Inputs for one Subsystem 06 combined-feature run."""

    session_root: Path | None = None
    s4_run_dir: Path | None = None
    s5_run_dir: Path | None = None
    output_root: Path | None = None
    config: CombinedFeaturesConfig | None = None
    config_path: Path | None = None
    run_purpose: str = "development"
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class CombinedFeaturesResult:
    """Summary returned after an S6 run directory is recorded."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    qc_path: Path
    model_compat_path: Path
    dyadic_features_path: Path | None
    session_root: Path
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    qc: Mapping[str, Any] | None = None
    model_compat: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _RunPaths:
    run_dir: Path
    dyadic_features: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path
    qc: Path
    model_compat: Path


def run_combined_feature_extraction(
    request: CombinedFeaturesRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> CombinedFeaturesResult:
    """Run the Sprint 3 combined-feature pipeline and persist S6 artifacts."""

    purpose = request.run_purpose.strip()
    if not purpose:
        raise CombinedFeaturesError("run_purpose must not be empty.")

    pair = _resolve_pair(request)
    config = _resolve_config(request)
    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    run_id = f"combined_features__{timestamp}"
    paths = _resolve_run_paths(
        session_root=pair.session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _report_progress(
        progress_callback,
        phase="starting",
        message="Validating S4/S5 pair and creating run directory",
        completed_units=0,
        total_units=None,
        is_indeterminate=True,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            pair.session_root / "preprocess",
            pair.session_root / "pose_inference",
            pair.session_root / "tracking_correction",
            pair.session_root / "pose_finalization",
            pair.session_root / "blob_features",
            pair.s4_run_dir,
            pair.s5_run_dir,
            pair.final_pose_path,
            pair.blob_features_path,
            pair.prepared_sync,
        ),
    )
    if paths.run_dir.exists():
        raise CombinedFeaturesError(f"S6 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise CombinedFeaturesError(
            f"S6 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    provenance = _pair_provenance(pair)
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            pair=pair,
            paths=paths,
            config=config,
            provenance=provenance,
            computation=None,
            qc_summary=None,
            model_compat=None,
            error_message=None,
            dyadic_features_written=False,
        ),
    )
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            config=config,
            status=STATUS_RUNNING,
            s4_run_id=pair.s4_run_id,
            s5_run_id=pair.s5_run_id,
        ),
    )

    computation: CombinedFeaturesComputation | None = None
    qc_summary: QcSummary | None = None
    model_compat: dict[str, Any] | None = None
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    dyadic_features_written = False

    try:
        _report_progress(
            progress_callback,
            phase="computing",
            message="Computing combined pose and video features",
            completed_units=0,
            total_units=None,
            is_indeterminate=True,
        )
        computation = compute_combined_features_from_pair(
            pair,
            exp_name=config.exp_name,
            iteration_name=config.iteration_name,
            smooth_win=config.smooth_win,
            diff_dt=config.diff_dt,
        )
        table = build_dyadic_features_table(
            computation.features,
            session_root=pair.session_root,
            source_s4_run_id=pair.s4_run_id,
            source_s5_run_id=pair.s5_run_id,
        )
        implantation = computation.adapted_pose.implantation
        implant_reason = computation.features.attrs.get("implantation_reason")
        qc_summary = evaluate_combined_features_qc(
            table=table,
            n_tracks=computation.adapted_pose.n_tracks,
            n_samples=computation.adapted_pose.n_samples,
            tracks=computation.adapted_pose.tracks,
            nodes=computation.adapted_pose.nodes,
            implantation_status=implantation.status,
            implantation_reason=(
                str(implant_reason) if implant_reason is not None else None
            ),
            implanted_track_id=implantation.implanted_track_id,
            alignment=pair.alignment.to_metadata(),
            source_warnings=computation.warnings,
        )
        assert_structural_qc(qc_summary)

        model_compat = build_model_compat_report(
            features=table,
            settings=config.model_registry,
            feature_artifact=DYADIC_FEATURES_FILENAME,
        )

        # Revalidate joint S4/S5 eligibility immediately before publishing.
        revalidated = validate_joint_s4_s5_pair(
            session_root=pair.session_root,
            s4_run_dir=pair.s4_run_dir,
            s5_run_dir=pair.s5_run_dir,
        )
        if revalidated.prepared_sync.resolve() != pair.prepared_sync.resolve():
            raise CombinedFeaturesError(
                "Prepared sync path changed during S6 processing."
            )
        if revalidated.prepared_sync_sha256 != pair.prepared_sync_sha256:
            raise CombinedFeaturesError(
                "Prepared sync SHA256 changed during S6 processing."
            )
        if _sha256(revalidated.final_pose_path) != provenance["source_s4"][
            "final_pose_sha256"
        ]:
            raise CombinedFeaturesError(
                "S4 final_pose.parquet changed during S6 processing (SHA256 mismatch)."
            )
        if _sha256(revalidated.blob_features_path) != provenance["source_s5"][
            "blob_features_sha256"
        ]:
            raise CombinedFeaturesError(
                "S5 blob_features.parquet changed during S6 processing (SHA256 mismatch)."
            )

        _report_progress(
            progress_callback,
            phase="writing",
            message="Writing dyadic features and QC artifacts",
            completed_units=1,
            total_units=1,
            is_indeterminate=False,
        )
        write_dyadic_features_parquet(paths.dyadic_features, table)
        dyadic_features_written = True
        _write_json(paths.qc, qc_summary.to_metadata())
        write_model_compat_json(paths.model_compat, model_compat)

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
            message=f"Combined features finished ({status})",
            completed_units=1,
            total_units=1,
            is_indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        if paths.dyadic_features.exists():
            with suppress(OSError):
                paths.dyadic_features.unlink()
        if paths.qc.exists():
            with suppress(OSError):
                paths.qc.unlink()
        if paths.model_compat.exists():
            with suppress(OSError):
                paths.model_compat.unlink()
        dyadic_features_written = False

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        pair=pair,
        config=config,
        provenance=provenance,
        computation=computation,
        qc_summary=qc_summary,
        model_compat=model_compat,
        error_message=error_message,
        dyadic_features_written=dyadic_features_written,
    )

    if not success:
        raise CombinedFeaturesError(error_message or "Combined feature extraction failed.")

    return CombinedFeaturesResult(
        success=True,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        qc_path=paths.qc,
        model_compat_path=paths.model_compat,
        dyadic_features_path=paths.dyadic_features if dyadic_features_written else None,
        session_root=pair.session_root,
        warnings=warnings,
        error_message=None,
        qc=qc_summary.to_metadata() if qc_summary is not None else None,
        model_compat=model_compat,
    )


def is_s6_run_eligible_for_downstream(s6_run_dir: Path | str) -> bool:
    """Return True when a completed S6 run still matches its recorded S4/S5 sources."""

    try:
        validate_s6_run_for_downstream(s6_run_dir)
    except CombinedFeaturesError:
        return False
    return True


def validate_s6_run_for_downstream(s6_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed S6 run for downstream use (staleness-aware)."""

    run_dir = Path(s6_run_dir).expanduser().resolve()
    fingerprint: tuple[Any, ...] = run_dir_fingerprint(
        run_dir, (DYADIC_FEATURES_FILENAME,)
    )
    # Invalidate when linked S4/S5 artifacts change even if S6 outputs do not.
    meta_path = run_dir / RUN_META_FILENAME
    if meta_path.is_file():
        try:
            meta = _load_json_object(meta_path)
            upstream: list[Any] = []
            source_s4 = meta.get("source_s4")
            source_s5 = meta.get("source_s5")
            if isinstance(source_s4, Mapping):
                upstream.append(artifact_fingerprint(source_s4.get("final_pose")))
                upstream.append(artifact_fingerprint(source_s4.get("prepared_sync")))
            if isinstance(source_s5, Mapping):
                upstream.append(artifact_fingerprint(source_s5.get("blob_features")))
                upstream.append(artifact_fingerprint(source_s5.get("prepared_sync")))
            if upstream:
                fingerprint = (*fingerprint, *upstream)
        except (CombinedFeaturesError, OSError, TypeError, ValueError):
            fingerprint = (*fingerprint, "upstream_unreadable")
    return memoized_validation(
        kind="validate_s6_run_for_downstream",
        fingerprint=fingerprint,
        compute=lambda: _validate_s6_run_for_downstream_uncached(run_dir),
    )


def _validate_s6_run_for_downstream_uncached(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise CombinedFeaturesError(f"S6 run directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise CombinedFeaturesError(f"S6 run_meta.json is missing: {meta_path}")
    meta = _load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise CombinedFeaturesError(
            f"S6 run is not completed (status={status or 'missing'!r})."
        )

    parquet = run_dir / DYADIC_FEATURES_FILENAME
    if not parquet.is_file():
        raise CombinedFeaturesError(
            f"Completed S6 run is missing {DYADIC_FEATURES_FILENAME}: {parquet}"
        )

    source_s4 = meta.get("source_s4")
    source_s5 = meta.get("source_s5")
    if not isinstance(source_s4, Mapping) or not isinstance(source_s5, Mapping):
        raise CombinedFeaturesError(
            "S6 run_meta.json is missing source_s4 / source_s5 provenance."
        )

    session_root_text = _optional_text(meta.get("session_root"))
    if session_root_text is None:
        session_root_text = _optional_text(source_s4.get("session_root"))
    if session_root_text is None:
        raise CombinedFeaturesError("S6 run_meta.json session_root is missing.")

    s4_run_dir = _optional_text(source_s4.get("run_dir"))
    s5_run_dir = _optional_text(source_s5.get("run_dir"))
    if s4_run_dir is None or s5_run_dir is None:
        raise CombinedFeaturesError(
            "S6 run_meta.json is missing source_s4.run_dir or source_s5.run_dir."
        )

    pair = validate_joint_s4_s5_pair(
        session_root=session_root_text,
        s4_run_dir=s4_run_dir,
        s5_run_dir=s5_run_dir,
    )

    recorded_pose_sha = _optional_text(source_s4.get("final_pose_sha256"))
    if recorded_pose_sha is not None:
        current = _sha256(pair.final_pose_path)
        if current.lower() != recorded_pose_sha.lower():
            raise CombinedFeaturesError(
                "S6 source S4 final_pose SHA256 no longer matches the recorded digest."
            )
    recorded_blob_sha = _optional_text(source_s5.get("blob_features_sha256"))
    if recorded_blob_sha is not None:
        current = _sha256(pair.blob_features_path)
        if current.lower() != recorded_blob_sha.lower():
            raise CombinedFeaturesError(
                "S6 source S5 blob_features SHA256 no longer matches the recorded digest."
            )
    recorded_sync_sha = _optional_text(source_s4.get("prepared_sync_sha256"))
    if recorded_sync_sha is None:
        recorded_sync_sha = _optional_text(source_s5.get("prepared_sync_sha256"))
    if (
        recorded_sync_sha is not None
        and pair.prepared_sync_sha256.lower() != recorded_sync_sha.lower()
    ):
        raise CombinedFeaturesError(
            "S6 prepared_sync SHA256 no longer matches the recorded digest."
        )

    return {
        "s6_run_dir": run_dir,
        "status": status,
        "session_root": pair.session_root,
        "s4_run_dir": pair.s4_run_dir,
        "s5_run_dir": pair.s5_run_dir,
        "prepared_sync_sha256": pair.prepared_sync_sha256,
        "final_pose_sha256": _sha256(pair.final_pose_path),
        "blob_features_sha256": _sha256(pair.blob_features_path),
    }


def _resolve_pair(request: CombinedFeaturesRequest) -> JointEligiblePair:
    if request.session_root is None:
        raise CombinedFeaturesError("session_root is required for combined feature extraction.")
    if request.s4_run_dir is None or request.s5_run_dir is None:
        # Allow default selection when both omitted.
        if request.s4_run_dir is None and request.s5_run_dir is None:
            from combined_features.upstream import select_default_joint_pair

            return select_default_joint_pair(request.session_root)
        raise CombinedFeaturesError(
            "Provide both s4_run_dir and s5_run_dir, or omit both to select defaults."
        )
    return validate_joint_s4_s5_pair(
        session_root=request.session_root,
        s4_run_dir=request.s4_run_dir,
        s5_run_dir=request.s5_run_dir,
    )


def _resolve_config(request: CombinedFeaturesRequest) -> CombinedFeaturesConfig:
    if request.config is not None:
        return request.config.validated()
    if request.config_path is not None:
        return load_combined_features_config(request.config_path)
    return load_combined_features_config(None)


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / COMBINED_FEATURES_DIRNAME
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        dyadic_features=run_dir / DYADIC_FEATURES_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
        qc=run_dir / QC_FILENAME,
        model_compat=run_dir / MODEL_COMPAT_FILENAME,
    )


def _reject_output_overlap(run_dir: Path, protected: Sequence[Path]) -> None:
    resolved_run = run_dir.resolve()
    for protected_path in protected:
        resolved_protected = Path(protected_path).resolve()
        if (
            resolved_run == resolved_protected
            or _is_relative_to(resolved_run, resolved_protected)
            or _is_relative_to(resolved_protected, resolved_run)
        ):
            raise CombinedFeaturesError(
                "S6 run directory must not overlap S1–S5 artifacts: "
                f"{resolved_run}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _pair_provenance(pair: JointEligiblePair) -> dict[str, Any]:
    source_s1 = pair.s5_meta.get("source_s1") if isinstance(pair.s5_meta, Mapping) else {}
    source_s3 = pair.s4_meta.get("source_s3") if isinstance(pair.s4_meta, Mapping) else {}
    if not isinstance(source_s1, Mapping):
        source_s1 = {}
    if not isinstance(source_s3, Mapping):
        source_s3 = {}
    return {
        "source_s4": {
            "run_dir": _path_text(pair.s4_run_dir),
            "run_id": pair.s4_run_id,
            "session_root": _path_text(pair.session_root),
            "final_pose": _path_text(pair.final_pose_path),
            "final_pose_sha256": _sha256(pair.final_pose_path),
            "prepared_sync": _path_text(pair.prepared_sync),
            "prepared_sync_sha256": pair.prepared_sync_sha256,
            "tracked_pose_sha256": _optional_text(source_s3.get("tracked_pose_sha256")),
            "source_s3": dict(source_s3),
        },
        "source_s5": {
            "run_dir": _path_text(pair.s5_run_dir),
            "run_id": pair.s5_run_id,
            "session_root": _path_text(pair.session_root),
            "blob_features": _path_text(pair.blob_features_path),
            "blob_features_sha256": _sha256(pair.blob_features_path),
            "prepared_sync": _optional_text(source_s1.get("prepared_sync"))
            or _path_text(pair.prepared_sync),
            "prepared_sync_sha256": _optional_text(source_s1.get("prepared_sync_sha256"))
            or pair.prepared_sync_sha256,
            "prepared_video_sha256": _optional_text(source_s1.get("prepared_video_sha256")),
            "cropped_background_sha256": _optional_text(
                source_s1.get("cropped_background_sha256")
            ),
            "source_s1": dict(source_s1),
        },
        "alignment": pair.alignment.to_metadata(),
        "sample_count": int(pair.sample_count),
        "dyadic_features_schema_version": DYADIC_FEATURES_SCHEMA_VERSION,
    }


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    pair: JointEligiblePair,
    config: CombinedFeaturesConfig,
    provenance: Mapping[str, Any],
    computation: CombinedFeaturesComputation | None,
    qc_summary: QcSummary | None,
    model_compat: Mapping[str, Any] | None,
    error_message: str | None,
    dyadic_features_written: bool,
) -> None:
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            pair=pair,
            paths=paths,
            config=config,
            provenance=provenance,
            computation=computation,
            qc_summary=qc_summary,
            model_compat=model_compat,
            error_message=error_message,
            dyadic_features_written=dyadic_features_written,
        ),
    )
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            config=config,
            status=status,
            s4_run_id=pair.s4_run_id,
            s5_run_id=pair.s5_run_id,
        ),
    )
    if qc_summary is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        _write_json(paths.qc, qc_summary.to_metadata())
    if model_compat is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        write_model_compat_json(paths.model_compat, model_compat)
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        pair=pair,
        paths=paths,
        config=config,
        computation=computation,
        qc_summary=qc_summary,
        model_compat=model_compat,
        error_message=error_message,
        dyadic_features_written=dyadic_features_written,
    )


def _run_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    pair: JointEligiblePair,
    paths: _RunPaths,
    config: CombinedFeaturesConfig,
    provenance: Mapping[str, Any],
    computation: CombinedFeaturesComputation | None,
    qc_summary: QcSummary | None,
    model_compat: Mapping[str, Any] | None,
    error_message: str | None,
    dyadic_features_written: bool,
) -> dict[str, Any]:
    adapter_meta: dict[str, Any] | None = None
    if computation is not None:
        adapter_meta = {
            "adapter_policy_version": computation.adapted_pose.adapter_policy_version,
            "implantation": computation.adapted_pose.implantation.to_metadata(),
            "tracks": list(computation.adapted_pose.tracks),
            "nodes": list(computation.adapted_pose.nodes),
            "n_samples": computation.adapted_pose.n_samples,
            "n_tracks": computation.adapted_pose.n_tracks,
            "n_directional_rows": computation.n_rows,
            "computation_metadata": dict(computation.metadata),
            "warnings": list(computation.warnings),
        }
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "session_root": _path_text(pair.session_root),
        "source_s4": dict(provenance["source_s4"]),
        "source_s5": dict(provenance["source_s5"]),
        "alignment": dict(provenance["alignment"]),
        "adapter": adapter_meta,
        "settings": config.to_settings_dict(),
        "outputs": {
            "dyadic_features_parquet": (
                _path_text(paths.dyadic_features) if dyadic_features_written else None
            ),
            "settings_used": _path_text(paths.settings_used),
            "processing_log": _path_text(paths.processing_log),
            "qc": _path_text(paths.qc),
            "model_compat": _path_text(paths.model_compat),
        },
        "qc": qc_summary.to_metadata() if qc_summary is not None else None,
        "model_compat_summary": (
            {
                "registry_status": model_compat.get("registry_status"),
                "n_detectors": len(model_compat.get("detectors", [])),
                "detector_statuses": {
                    str(item.get("detector_name")): item.get("status")
                    for item in model_compat.get("detectors", [])
                    if isinstance(item, Mapping)
                },
            }
            if model_compat is not None
            else None
        ),
        "timing_summary": {
            "sample_count": int(pair.sample_count),
            "target_fps": float(pair.alignment.target_fps),
            "row_count": computation.n_rows if computation is not None else None,
        },
    }
    if error_message is not None:
        payload["error"] = error_message
    return payload


def _settings_payload(
    *,
    config: CombinedFeaturesConfig,
    status: str,
    s4_run_id: str,
    s5_run_id: str,
) -> dict[str, Any]:
    payload = config.to_settings_dict()
    payload["status"] = status
    payload["source_s4_run_id"] = s4_run_id
    payload["source_s5_run_id"] = s5_run_id
    return payload


def _write_processing_log(
    *,
    path: Path,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    pair: JointEligiblePair,
    paths: _RunPaths,
    config: CombinedFeaturesConfig,
    computation: CombinedFeaturesComputation | None,
    qc_summary: QcSummary | None,
    model_compat: Mapping[str, Any] | None,
    error_message: str | None,
    dyadic_features_written: bool,
) -> None:
    lines = [
        "Subsystem 06 combined pose & video features",
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"created_at: {created_at}",
        f"completed_at: {completed_at}",
        f"status: {status}",
        f"session_root: {_path_text(pair.session_root)}",
        f"s4_run_id: {pair.s4_run_id}",
        f"s5_run_id: {pair.s5_run_id}",
        f"prepared_sync: {_path_text(pair.prepared_sync)}",
        f"prepared_sync_sha256: {pair.prepared_sync_sha256}",
        f"sample_count: {pair.sample_count}",
        f"smooth_win: {config.smooth_win}",
        f"diff_dt: {config.diff_dt}",
        f"adapter_policy_version: {config.adapter_policy_version}",
        f"dyadic_features_schema_version: {config.dyadic_features_schema_version}",
        (
            "model_registry.root: "
            + (
                str(config.model_registry.root)
                if config.model_registry.root is not None
                else "null"
            )
        ),
    ]
    if computation is not None:
        lines.extend(
            [
                f"n_tracks: {computation.adapted_pose.n_tracks}",
                f"n_nodes: {computation.adapted_pose.n_nodes}",
                f"n_directional_rows: {computation.n_rows}",
                f"implantation_status: {computation.adapted_pose.implantation.status}",
            ]
        )
        for warning in computation.warnings:
            lines.append(f"computation.warning: {warning}")
    if qc_summary is not None:
        lines.append(f"qc.structural_ok: {qc_summary.structural_ok}")
        lines.append(f"qc.warning_count: {len(qc_summary.warnings)}")
        for warning in qc_summary.warnings:
            lines.append(f"qc.warning: {warning}")
        for failure in qc_summary.failures:
            lines.append(f"qc.failure: {failure}")
    if model_compat is not None:
        lines.append(f"model_compat.registry_status: {model_compat.get('registry_status')}")
        lines.append(
            "model_compat.n_detectors: "
            f"{len(model_compat.get('detectors', []))}"
        )
    lines.append(
        "dyadic_features_parquet: "
        + (_path_text(paths.dyadic_features) if dyadic_features_written else "not_written")
    )
    if error_message:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        raise CombinedFeaturesError(f"Could not load JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CombinedFeaturesError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value.strip())
    if not cleaned:
        raise CombinedFeaturesError("timestamp is empty after sanitization.")
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
