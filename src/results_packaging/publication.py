"""Immutable Subsystem 09 publication lifecycle and downstream eligibility.

Mirrors S5–S8 conventions: unique run ids, running → publish → completed /
completed_with_warnings / failed, atomic parquet writes, science-artifact
cleanup on failure while preserving diagnostics, and historical status that
is never rewritten when sources later go stale.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import BEHAVIORAL_ARBITRATION_DIRNAME
from behavioral_inference.contracts import BEHAVIORAL_INFERENCE_DIRNAME
from combined_features.contracts import COMBINED_FEATURES_DIRNAME
from preprocess.models import OperationProgress
from project.file_digest import sha256_file
from project.validation_memo import (
    artifact_fingerprint,
    memoized_validation,
    run_dir_fingerprint,
)
from results_packaging.config import ResultsPackagingConfig
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BOUT_INDEX_FILENAME,
    ELIGIBLE_DOWNSTREAM,
    ETHOGRAM_SUMMARY_FILENAME,
    EXCLUSION_TABLE_FILENAME,
    EXPORT_PROFILES_DIRNAME,
    FINGERPRINT_TYPE_SHA256,
    PACKAGE_MANIFEST_FILENAME,
    PROCESSING_LOG_FILENAME,
    PROFILE_MANIFEST_FILENAME,
    QC_FILENAME,
    RESULTS_PACKAGE_DIRNAME,
    RESULTS_PACKAGE_RUN_PREFIX,
    REVIEW_EMIT_PROFILE_ID,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    SYNC_INDEX_FILENAME,
    TARGET_FPS,
    VERIFICATION_STRENGTH_CONTENT_HASH,
    VERIFICATION_STRENGTH_SIZE_MTIME,
    ProvenanceError,
    PublicationError,
)
from results_packaging.eligibility import validate_s8_source_for_packaging
from results_packaging.export import (
    load_json_object,
    write_behavior_bouts_parquet,
    write_bout_index_parquet,
    write_ethogram_summary_parquet,
    write_exclusion_table_parquet,
    write_json,
    write_sync_index_parquet,
    write_yaml,
)
from results_packaging.export_profiles import ExportProfileResult
from results_packaging.manifest import (
    build_package_manifest,
    compute_packaged_bout_checksum,
)
from results_packaging.qc import QcSummary, assert_structural_qc, evaluate_results_package_qc
from results_packaging.results import ResultsPackageRunResult, Sprint2PackagingResult

ProgressCallback = Callable[[OperationProgress], None]


@dataclass(frozen=True, slots=True)
class _RunPaths:
    run_dir: Path
    sync_index: Path
    behavior_bouts: Path
    ethogram_summary: Path
    package_manifest: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path
    qc: Path
    export_profiles_root: Path


def run_results_packaging_publication(
    request: Any,
    progress_callback: ProgressCallback | None = None,
) -> ResultsPackageRunResult:
    """Validate, package, QC, revalidate, and atomically publish an S9 run."""

    from results_packaging.runner import run_results_packaging_sprint2

    session_root = Path(request.session_root).expanduser().resolve()
    if not session_root.is_dir():
        raise PublicationError(f"Session root does not exist: {session_root}")

    total_phases = 7
    _report_progress(
        progress_callback,
        phase="validate",
        message="Validating S8 packaging readiness…",
        completed_units=1,
        total_units=total_phases,
        is_indeterminate=False,
    )

    config_probe = _resolve_publish_config(request)
    preflight = validate_s8_source_for_packaging(
        session_root,
        s8_run_dir=request.s8_run_dir,
        require_prepared_video=config_probe.execution.require_prepared_video,
    )

    recorded = {
        "s8_states_sha256": preflight.states_sha256,
        "s8_bouts_sha256": preflight.bouts_sha256,
        "prepared_sync_sha256": preflight.sync_video.prepared_sync_sha256,
        "prepared_video_path": (
            str(preflight.sync_video.prepared_video_path)
            if preflight.sync_video.prepared_video_path is not None
            else None
        ),
        "prepared_video_fingerprint": (
            artifact_fingerprint(preflight.sync_video.prepared_video_path)
            if preflight.sync_video.prepared_video_path is not None
            else None
        ),
        "prepared_video_sha256_recorded": preflight.sync_video.prepared_video_sha256_recorded,
    }

    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    run_id = request.package_run_id or f"{RESULTS_PACKAGE_RUN_PREFIX}__{timestamp}"
    if (
        request.package_run_id is None
        and not str(run_id).startswith(f"{RESULTS_PACKAGE_RUN_PREFIX}__")
    ):
        run_id = f"{RESULTS_PACKAGE_RUN_PREFIX}__{timestamp}"
    paths = _resolve_run_paths(
        session_root=session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            session_root / "preprocess",
            session_root / "pose_inference",
            session_root / "tracking_correction",
            session_root / "pose_finalization",
            session_root / "blob_features",
            session_root / COMBINED_FEATURES_DIRNAME,
            session_root / BEHAVIORAL_INFERENCE_DIRNAME,
            session_root / BEHAVIORAL_ARBITRATION_DIRNAME,
            preflight.s8_run_dir,
            preflight.source_s7_run_dir,
            preflight.source_s6_run_dir,
            preflight.states_path,
            preflight.bouts_path,
            preflight.sync_video.prepared_sync_path,
            *(
                (preflight.sync_video.prepared_video_path,)
                if preflight.sync_video.prepared_video_path is not None
                else ()
            ),
        ),
    )
    if paths.run_dir.exists():
        raise PublicationError(f"S9 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PublicationError(
            f"S9 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    settings_snapshot = config_probe.to_settings_dict()
    settings_snapshot["selected_s8_run_dir"] = str(preflight.s8_run_dir)
    settings_snapshot["package_run_id"] = run_id
    settings_snapshot["execution"] = {
        **settings_snapshot.get("execution", {}),
        "publish_run": True,
    }
    provenance = _build_provenance(preflight, recorded=recorded, settings=settings_snapshot)
    write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=request.run_purpose,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=session_root,
            paths=paths,
            provenance=provenance,
            qc_summary=None,
            error_message=None,
            science_written=False,
            settings_snapshot=settings_snapshot,
            export_profile_dirs={},
        ),
    )
    write_yaml(paths.settings_used, {**settings_snapshot, "status": STATUS_RUNNING})

    sprint2: Sprint2PackagingResult | None = None
    qc_summary: QcSummary | None = None
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    science_written = False
    export_profile_dirs: dict[str, Path] = {}
    currently_eligible = False
    classification: str | None = None

    try:
        _report_progress(
            progress_callback,
            phase="package",
            message="Building sync index, bouts, summaries, and profiles…",
            completed_units=2,
            total_units=total_phases,
            is_indeterminate=False,
        )
        sprint2 = run_results_packaging_sprint2(
            replace(
                request,
                package_run_id=run_id,
                s8_run_dir=preflight.s8_run_dir,
                build_package_tables=True,
                config=config_probe,
            )
        )

        _report_progress(
            progress_callback,
            phase="qc",
            message="Evaluating structural and scientific QC…",
            completed_units=3,
            total_units=total_phases,
            is_indeterminate=False,
        )
        s8_bouts = pd.read_parquet(preflight.bouts_path)
        states = pd.read_parquet(preflight.states_path)
        qc_summary = evaluate_results_package_qc(
            sync=sprint2.sync,
            packaged_bouts=sprint2.behavior_bouts,
            ethogram_summary=sprint2.ethogram_summary,
            source_s8_bouts=s8_bouts,
            states=states,
            export_profiles=sprint2.export_profiles,
            source_warnings=sprint2.source.warnings,
            dependency_notes=sprint2.dependencies.notes,
            source_s8_eligible=sprint2.source.s8_eligible,
            source_s8_status=sprint2.source.s8_status,
            prepared_video_present=preflight.sync_video.prepared_video_path is not None
            and preflight.sync_video.prepared_video_path.is_file(),
            require_prepared_video=config_probe.execution.require_prepared_video,
            package_mode=config_probe.execution.package_mode,
            full_bout_preservation=True,
        )
        assert_structural_qc(qc_summary)

        _report_progress(
            progress_callback,
            phase="revalidate",
            message="Revalidating source identity before publish…",
            completed_units=4,
            total_units=total_phases,
            is_indeterminate=False,
        )
        _revalidate_sources_before_publish(
            session_root=session_root,
            s8_run_dir=preflight.s8_run_dir,
            require_prepared_video=config_probe.execution.require_prepared_video,
            recorded=recorded,
        )

        _report_progress(
            progress_callback,
            phase="publish",
            message="Publishing immutable S9 artifacts…",
            completed_units=5,
            total_units=total_phases,
            is_indeterminate=False,
        )
        write_sync_index_parquet(paths.sync_index, sprint2.sync.sync_index)
        write_behavior_bouts_parquet(paths.behavior_bouts, sprint2.behavior_bouts)
        write_ethogram_summary_parquet(paths.ethogram_summary, sprint2.ethogram_summary)
        write_json(paths.qc, qc_summary.to_metadata())
        export_profile_dirs = _write_export_profiles(
            paths.export_profiles_root, sprint2.export_profiles
        )

        bout_checksum = compute_packaged_bout_checksum(sprint2.behavior_bouts)
        profile_paths = {
            profile_id: {
                "bout_index": paths.export_profiles_root / profile_id / BOUT_INDEX_FILENAME,
                "profile_manifest": (
                    paths.export_profiles_root / profile_id / PROFILE_MANIFEST_FILENAME
                ),
                "exclusion_table": (
                    paths.export_profiles_root / profile_id / EXCLUSION_TABLE_FILENAME
                ),
            }
            for profile_id in export_profile_dirs
        }
        manifest = build_package_manifest(
            package_run_id=run_id,
            created_at=created_at,
            session_root=session_root,
            run_dir=paths.run_dir,
            dependencies=sprint2.dependencies,
            qc=qc_summary,
            settings_used=sprint2.settings_used,
            export_profiles=sprint2.export_profiles,
            packaged_bouts_checksum=bout_checksum,
            n_sync_rows=int(len(sprint2.sync.sync_index)),
            n_bouts=int(len(sprint2.behavior_bouts)),
            n_ethogram_rows=int(len(sprint2.ethogram_summary)),
            source_s8_states_path=preflight.states_path,
            source_s8_bouts_path=preflight.bouts_path,
            source_s8_states_sha256=preflight.states_sha256,
            source_s8_bouts_sha256=preflight.bouts_sha256,
            source_s8_run_id=preflight.s8_run_id,
            source_s8_run_dir=preflight.s8_run_dir,
            source_s8_status=preflight.s8_status,
            source_s7_run_id=preflight.source_s7_run_id,
            source_s7_run_dir=preflight.source_s7_run_dir,
            source_s6_run_id=preflight.source_s6_run_id,
            source_s6_run_dir=preflight.source_s6_run_dir,
            sync_construction_method=config_probe.execution.sync_construction_method,
            package_mode=config_probe.execution.package_mode,
            selected_export_profiles=config_probe.execution.selected_export_profiles,
            profile_artifact_paths=profile_paths,
        )
        write_json(paths.package_manifest, manifest)
        science_written = True

        if qc_summary.has_warnings:
            status = STATUS_COMPLETED_WITH_WARNINGS
            warnings = qc_summary.warnings
        else:
            status = STATUS_COMPLETED
            warnings = ()
        success = True
        currently_eligible = True
        classification = (
            STATUS_COMPLETED_WITH_WARNINGS
            if status == STATUS_COMPLETED_WITH_WARNINGS
            else ELIGIBLE_DOWNSTREAM
        )
        _report_progress(
            progress_callback,
            phase="complete",
            message=f"Results packaging {status}.",
            completed_units=total_phases,
            total_units=total_phases,
            is_indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        _unlink_science_artifacts(paths)
        science_written = False
        export_profile_dirs = {}
        currently_eligible = False
        classification = "failed_or_incomplete"
        _report_progress(
            progress_callback,
            phase="failed",
            message=f"Results packaging failed: {exc}",
            completed_units=None,
            total_units=total_phases,
            is_indeterminate=True,
        )

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=request.run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        session_root=session_root,
        provenance=provenance,
        sprint2=sprint2,
        qc_summary=qc_summary,
        error_message=error_message,
        science_written=science_written,
        settings_snapshot=settings_snapshot,
        export_profile_dirs=export_profile_dirs,
    )

    return ResultsPackageRunResult(
        success=success,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        session_root=session_root,
        source_s8_run_id=preflight.s8_run_id,
        source_s8_run_dir=preflight.s8_run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        qc_path=paths.qc if science_written else None,
        package_manifest_path=paths.package_manifest if science_written else None,
        sync_index_path=paths.sync_index if science_written else None,
        behavior_bouts_path=paths.behavior_bouts if science_written else None,
        ethogram_summary_path=paths.ethogram_summary if science_written else None,
        export_profile_dirs=export_profile_dirs,
        warnings=warnings,
        error_message=error_message,
        qc=qc_summary.to_metadata() if qc_summary is not None else None,
        currently_eligible=currently_eligible,
        classification=classification,
        diagnostics={
            "science_written": science_written,
            "n_bouts": (
                int(len(sprint2.behavior_bouts)) if sprint2 is not None else None
            ),
            "n_sync_rows": (
                int(len(sprint2.sync.sync_index)) if sprint2 is not None else None
            ),
        },
    )


def run_publication_from_session(
    session_root: Path | str,
    *,
    s8_run_dir: Path | str | None = None,
    config: ResultsPackagingConfig | None = None,
    timestamp: str | None = None,
    selected_export_profiles: tuple[str, ...] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ResultsPackageRunResult:
    """Convenience wrapper for immutable S9 publication."""

    from results_packaging.runner import ResultsPackagingRequest

    resolved = config or ResultsPackagingConfig()
    if selected_export_profiles is not None:
        resolved = replace(
            resolved,
            execution=replace(
                resolved.execution,
                selected_export_profiles=tuple(selected_export_profiles),
                publish_run=True,
            ),
        )
    else:
        resolved = replace(
            resolved,
            execution=replace(resolved.execution, publish_run=True),
        )
    return run_results_packaging_publication(
        ResultsPackagingRequest(
            session_root=Path(session_root),
            s8_run_dir=Path(s8_run_dir) if s8_run_dir is not None else None,
            config=resolved.validated(),
            timestamp=timestamp,
            build_package_tables=True,
        ),
        progress_callback=progress_callback,
    )


def is_s9_run_eligible_for_downstream(s9_run_dir: Path | str) -> bool:
    """Return True when a completed S9 run still matches recorded dependencies."""

    try:
        validate_s9_run_for_downstream(s9_run_dir)
    except (PublicationError, ProvenanceError):
        return False
    return True


def validate_s9_run_for_downstream(s9_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed S9 run for S10 / analysis use (staleness-aware).

    Historical ``status`` on disk is never rewritten. Eligibility becomes false
    when recorded S8/sync/video digests no longer match or references break.
    """

    run_dir = Path(s9_run_dir).expanduser().resolve()
    fingerprint: tuple[Any, ...] = run_dir_fingerprint(
        run_dir,
        (
            SYNC_INDEX_FILENAME,
            BEHAVIOR_BOUTS_FILENAME,
            ETHOGRAM_SUMMARY_FILENAME,
            PACKAGE_MANIFEST_FILENAME,
            QC_FILENAME,
        ),
    )
    meta_path = run_dir / RUN_META_FILENAME
    if meta_path.is_file():
        try:
            meta = load_json_object(meta_path)
            upstream: list[Any] = []
            source_s8 = meta.get("source_s8")
            source_sync = meta.get("source_sync")
            source_video = meta.get("source_video")
            if isinstance(source_s8, Mapping):
                s8_text = _optional_text(source_s8.get("run_dir"))
                if s8_text is not None:
                    upstream.append(
                        run_dir_fingerprint(
                            Path(s8_text),
                            (
                                "behavior_states.parquet",
                                "behavior_bouts.parquet",
                                "qc.json",
                            ),
                        )
                    )
                upstream.append(artifact_fingerprint(source_s8.get("behavior_states")))
                upstream.append(artifact_fingerprint(source_s8.get("behavior_bouts")))
            if isinstance(source_sync, Mapping):
                upstream.append(artifact_fingerprint(source_sync.get("path")))
            if isinstance(source_video, Mapping):
                upstream.append(artifact_fingerprint(source_video.get("path")))
            if upstream:
                fingerprint = (*fingerprint, *upstream)
        except (PublicationError, OSError, TypeError, ValueError):
            fingerprint = (*fingerprint, "upstream_unreadable")
    return memoized_validation(
        kind="validate_s9_run_for_downstream",
        fingerprint=fingerprint,
        compute=lambda: _validate_s9_run_for_downstream_uncached(run_dir),
    )


def reopen_results_package(s9_run_dir: Path | str) -> dict[str, Any]:
    """Load an existing package without recomputing science tables."""

    run_dir = Path(s9_run_dir).expanduser().resolve()
    validated = validate_s9_run_for_downstream(run_dir)
    meta = load_json_object(run_dir / RUN_META_FILENAME)
    manifest = load_json_object(run_dir / PACKAGE_MANIFEST_FILENAME)
    qc = load_json_object(run_dir / QC_FILENAME)
    return {
        "run_dir": run_dir,
        "run_id": meta.get("run_id"),
        "status": meta.get("status"),
        "currently_eligible": True,
        "validation": validated,
        "run_meta": meta,
        "package_manifest": manifest,
        "qc": qc,
        "sync_index": pd.read_parquet(run_dir / SYNC_INDEX_FILENAME),
        "behavior_bouts": pd.read_parquet(run_dir / BEHAVIOR_BOUTS_FILENAME),
        "ethogram_summary": pd.read_parquet(run_dir / ETHOGRAM_SUMMARY_FILENAME),
    }


def _validate_s9_run_for_downstream_uncached(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise PublicationError(f"S9 run directory does not exist: {run_dir}")
    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise PublicationError(f"S9 run_meta.json is missing: {meta_path}")
    meta = load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise PublicationError(
            f"S9 run is not completed (status={status or 'missing'!r})."
        )

    for name in (
        SYNC_INDEX_FILENAME,
        BEHAVIOR_BOUTS_FILENAME,
        ETHOGRAM_SUMMARY_FILENAME,
        PACKAGE_MANIFEST_FILENAME,
        QC_FILENAME,
    ):
        if not (run_dir / name).is_file():
            raise PublicationError(f"Completed S9 run is missing {name}")

    source_s8 = meta.get("source_s8")
    if not isinstance(source_s8, Mapping):
        raise PublicationError("S9 run_meta.json is missing source_s8 provenance.")
    s8_run_dir_text = _optional_text(source_s8.get("run_dir"))
    if s8_run_dir_text is None:
        raise PublicationError("S9 run_meta.json source_s8.run_dir is missing.")
    s8_run_dir = Path(s8_run_dir_text)

    # Nested S8 eligibility + exact digest compare.
    from behavioral_arbitration.runner import (
        is_s8_run_eligible_for_downstream,
        validate_s8_run_for_downstream,
    )

    if not is_s8_run_eligible_for_downstream(s8_run_dir):
        raise ProvenanceError(
            "S9 source S8 run is no longer eligible for downstream use."
        )
    validate_s8_run_for_downstream(s8_run_dir)

    states_path = Path(
        _optional_text(source_s8.get("behavior_states"))
        or str(s8_run_dir / "behavior_states.parquet")
    )
    bouts_path = Path(
        _optional_text(source_s8.get("behavior_bouts"))
        or str(s8_run_dir / "behavior_bouts.parquet")
    )
    _assert_recorded_digest(
        label="S8 behavior_states",
        recorded=_optional_text(source_s8.get("behavior_states_sha256")),
        path=states_path,
    )
    _assert_recorded_digest(
        label="S8 behavior_bouts",
        recorded=_optional_text(source_s8.get("behavior_bouts_sha256")),
        path=bouts_path,
    )

    source_sync = meta.get("source_sync")
    if isinstance(source_sync, Mapping):
        sync_path_text = _optional_text(source_sync.get("path"))
        if sync_path_text is None:
            raise PublicationError("S9 source_sync.path is missing.")
        sync_path = Path(sync_path_text)
        if not sync_path.is_file():
            raise ProvenanceError(f"Prepared sync reference missing: {sync_path}")
        recorded_sync = _optional_text(source_sync.get("sha256"))
        if recorded_sync is not None:
            _assert_recorded_digest(
                label="prepared_sync",
                recorded=recorded_sync,
                path=sync_path,
            )

    source_video = meta.get("source_video")
    if isinstance(source_video, Mapping):
        video_path_text = _optional_text(source_video.get("path"))
        required = bool(source_video.get("required", True))
        if video_path_text is None and required:
            raise PublicationError("S9 source_video.path is missing.")
        if video_path_text is not None:
            video_path = Path(video_path_text)
            if not video_path.is_file():
                raise ProvenanceError(f"Prepared video reference missing: {video_path}")
            strength = _optional_text(source_video.get("verification_strength"))
            recorded_video = _optional_text(source_video.get("sha256"))
            if (
                strength == VERIFICATION_STRENGTH_CONTENT_HASH
                and recorded_video is not None
            ):
                _assert_recorded_digest(
                    label="prepared_video",
                    recorded=recorded_video,
                    path=video_path,
                )
            else:
                # Size/mtime fingerprint check when content hash was not asserted.
                recorded_fp = source_video.get("fingerprint")
                live_fp = list(artifact_fingerprint(video_path))
                if recorded_fp is not None and list(recorded_fp) != live_fp:
                    raise ProvenanceError(
                        "Prepared video size/mtime fingerprint no longer matches "
                        "the fingerprint recorded by S9 "
                        f"(type={FINGERPRINT_TYPE_SHA256 if recorded_video else 'size_mtime'}; "
                        f"strength={strength or VERIFICATION_STRENGTH_SIZE_MTIME}). "
                        "This is not silent SHA256 equivalence."
                    )

    # Embedded science artifact digests from manifest (skip lifecycle records).
    manifest_path = run_dir / PACKAGE_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = load_json_object(manifest_path)
        for entry in manifest.get("package_artifacts", []):
            if not isinstance(entry, Mapping):
                continue
            if not entry.get("embedded"):
                continue
            if entry.get("verification_strength") == "lifecycle_path":
                continue
            if entry.get("sha256") is None:
                continue
            abs_path = entry.get("absolute_path")
            rel = entry.get("relative_path")
            path = Path(str(abs_path)) if abs_path else (run_dir / str(rel or ""))
            _assert_recorded_digest(
                label=f"package:{entry.get('role')}",
                recorded=_optional_text(entry.get("sha256")),
                path=path,
            )

    return {
        "s9_run_dir": run_dir,
        "status": status,
        "session_root": _optional_text(meta.get("session_root")),
        "source_s8_run_dir": s8_run_dir.resolve(),
        "source_s8_run_id": _optional_text(source_s8.get("run_id")),
        "behavior_states_sha256": _optional_text(
            source_s8.get("behavior_states_sha256")
        ),
        "behavior_bouts_sha256": _optional_text(source_s8.get("behavior_bouts_sha256")),
    }


def _write_export_profiles(
    root: Path,
    profiles: Mapping[str, ExportProfileResult],
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    if not profiles:
        return written
    root.mkdir(parents=True, exist_ok=True)
    for profile_id, result in profiles.items():
        profile_dir = root / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        write_bout_index_parquet(profile_dir / BOUT_INDEX_FILENAME, result.bout_index)
        write_exclusion_table_parquet(
            profile_dir / EXCLUSION_TABLE_FILENAME, result.exclusion_table
        )
        write_json(
            profile_dir / PROFILE_MANIFEST_FILENAME,
            {
                **result.to_manifest_dict(),
                "published": True,
                "artifact_paths": {
                    "bout_index": str(profile_dir / BOUT_INDEX_FILENAME),
                    "exclusion_table": str(profile_dir / EXCLUSION_TABLE_FILENAME),
                },
            },
        )
        written[profile_id] = profile_dir
    return written


def _revalidate_sources_before_publish(
    *,
    session_root: Path,
    s8_run_dir: Path,
    require_prepared_video: bool,
    recorded: Mapping[str, Any],
) -> None:
    live = validate_s8_source_for_packaging(
        session_root,
        s8_run_dir=s8_run_dir,
        require_prepared_video=require_prepared_video,
    )
    if live.states_sha256.lower() != str(recorded["s8_states_sha256"]).lower():
        raise ProvenanceError("S8 behavior_states SHA256 changed during S9 processing.")
    if live.bouts_sha256.lower() != str(recorded["s8_bouts_sha256"]).lower():
        raise ProvenanceError("S8 behavior_bouts SHA256 changed during S9 processing.")
    if (
        live.sync_video.prepared_sync_sha256.lower()
        != str(recorded["prepared_sync_sha256"]).lower()
    ):
        raise ProvenanceError("Prepared sync SHA256 changed during S9 processing.")

    recorded_video_path = recorded.get("prepared_video_path")
    if recorded_video_path is not None:
        live_video = live.sync_video.prepared_video_path
        if live_video is None or str(live_video.resolve()) != str(
            Path(str(recorded_video_path)).resolve()
        ):
            raise ProvenanceError(
                "Prepared video path changed or disappeared during S9 processing."
            )
        live_fp = artifact_fingerprint(live_video)
        if live_fp != recorded.get("prepared_video_fingerprint"):
            raise ProvenanceError(
                "Prepared video size/mtime fingerprint changed during S9 processing "
                "(not equated with SHA256)."
            )


def _build_provenance(
    source: Any,
    *,
    recorded: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    video_path = source.sync_video.prepared_video_path
    video_sha = source.sync_video.prepared_video_sha256_recorded
    if video_path is not None and settings.get("execution", {}).get("hash_prepared_video"):
        video_sha = sha256_file(video_path)
        strength = VERIFICATION_STRENGTH_CONTENT_HASH
        fp_type = FINGERPRINT_TYPE_SHA256
    elif video_sha is not None:
        strength = VERIFICATION_STRENGTH_CONTENT_HASH
        fp_type = FINGERPRINT_TYPE_SHA256
    else:
        strength = VERIFICATION_STRENGTH_SIZE_MTIME
        fp_type = "size_mtime"
    return {
        "session_root": str(source.session_root),
        "settings": dict(settings),
        "source_s8": {
            "run_id": source.s8_run_id,
            "run_dir": str(source.s8_run_dir),
            "status_at_publish": source.s8_status,
            "classification_at_publish": source.s8_classification,
            "eligible_at_publish": source.s8_eligible,
            "behavior_states": str(source.states_path),
            "behavior_states_sha256": source.states_sha256,
            "behavior_bouts": str(source.bouts_path),
            "behavior_bouts_sha256": source.bouts_sha256,
        },
        "source_s7": {
            "run_id": source.source_s7_run_id,
            "run_dir": str(source.source_s7_run_dir),
        },
        "source_s6": {
            "run_id": source.source_s6_run_id,
            "run_dir": str(source.source_s6_run_dir),
        },
        "source_sync": {
            "path": str(source.sync_video.prepared_sync_path),
            "sha256": source.sync_video.prepared_sync_sha256,
            "fingerprint_type": FINGERPRINT_TYPE_SHA256,
            "verification_strength": VERIFICATION_STRENGTH_CONTENT_HASH,
        },
        "source_video": {
            "path": str(video_path) if video_path is not None else None,
            "sha256": video_sha,
            "fingerprint": (
                list(recorded.get("prepared_video_fingerprint") or ())
                if video_path is not None
                else None
            ),
            "fingerprint_type": fp_type,
            "verification_strength": strength,
            "required": True,
            "size_mtime_is_not_sha256": strength != VERIFICATION_STRENGTH_CONTENT_HASH
            or video_sha is None,
        },
    }


def _unlink_science_artifacts(paths: _RunPaths) -> None:
    for path in (
        paths.sync_index,
        paths.behavior_bouts,
        paths.ethogram_summary,
        paths.package_manifest,
        paths.qc,
    ):
        if path.exists():
            with suppress(OSError):
                path.unlink()
    if paths.export_profiles_root.exists():
        with suppress(OSError):
            shutil.rmtree(paths.export_profiles_root)
    # Clean leftover parquet.tmp files.
    for tmp in paths.run_dir.glob("*.parquet.tmp"):
        with suppress(OSError):
            tmp.unlink()


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    session_root: Path,
    provenance: Mapping[str, Any],
    sprint2: Sprint2PackagingResult | None,
    qc_summary: QcSummary | None,
    error_message: str | None,
    science_written: bool,
    settings_snapshot: Mapping[str, Any],
    export_profile_dirs: Mapping[str, Path],
) -> None:
    write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            session_root=session_root,
            paths=paths,
            provenance=provenance,
            qc_summary=qc_summary,
            error_message=error_message,
            science_written=science_written,
            settings_snapshot=settings_snapshot,
            export_profile_dirs=export_profile_dirs,
            sprint2=sprint2,
        ),
    )
    snapshot = dict(settings_snapshot)
    snapshot["status"] = status
    write_yaml(paths.settings_used, snapshot)
    if qc_summary is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        write_json(paths.qc, qc_summary.to_metadata())
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        session_root=session_root,
        paths=paths,
        provenance=provenance,
        sprint2=sprint2,
        qc_summary=qc_summary,
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
    provenance: Mapping[str, Any],
    qc_summary: QcSummary | None,
    error_message: str | None,
    science_written: bool,
    settings_snapshot: Mapping[str, Any],
    export_profile_dirs: Mapping[str, Path],
    sprint2: Sprint2PackagingResult | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "session_root": str(session_root),
        "fps": TARGET_FPS,
        "package_mode": settings_snapshot.get("execution", {}).get(
            "package_mode", "reference_lightweight"
        ),
        "source_s8": dict(provenance.get("source_s8", {})),
        "source_s7": dict(provenance.get("source_s7", {})),
        "source_s6": dict(provenance.get("source_s6", {})),
        "source_sync": dict(provenance.get("source_sync", {})),
        "source_video": dict(provenance.get("source_video", {})),
        "settings": dict(settings_snapshot),
        "outputs": {
            "sync_index_parquet": str(paths.sync_index) if science_written else None,
            "behavior_bouts_parquet": (
                str(paths.behavior_bouts) if science_written else None
            ),
            "ethogram_summary_parquet": (
                str(paths.ethogram_summary) if science_written else None
            ),
            "package_manifest": (
                str(paths.package_manifest) if science_written else None
            ),
            "settings_used": str(paths.settings_used),
            "processing_log": str(paths.processing_log),
            "qc": str(paths.qc) if science_written else None,
            "export_profiles": {k: str(v) for k, v in export_profile_dirs.items()},
        },
        "qc": qc_summary.to_metadata() if qc_summary is not None else None,
        "qc_summary": (
            {
                "warning_count": len(qc_summary.warnings),
                "failure_count": len(qc_summary.failures),
                "structural_ok": qc_summary.structural_ok,
                "n_bouts": (
                    int(len(sprint2.behavior_bouts)) if sprint2 is not None else None
                ),
                "n_sync_rows": (
                    int(len(sprint2.sync.sync_index)) if sprint2 is not None else None
                ),
            }
            if qc_summary is not None
            else None
        ),
        "error": error_message,
        "science_written": science_written,
    }
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
    provenance: Mapping[str, Any],
    sprint2: Sprint2PackagingResult | None,
    qc_summary: QcSummary | None,
    error_message: str | None,
    science_written: bool,
) -> None:
    lines = [
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"created_at: {created_at}",
        f"completed_at: {completed_at}",
        f"status: {status}",
        f"session_root: {session_root}",
        f"run_dir: {paths.run_dir}",
        f"science_written: {science_written}",
        f"source_s8_run_id: {(provenance.get('source_s8') or {}).get('run_id')}",
        f"source_s7_run_id: {(provenance.get('source_s7') or {}).get('run_id')}",
        f"source_s6_run_id: {(provenance.get('source_s6') or {}).get('run_id')}",
    ]
    if sprint2 is not None:
        lines.extend(
            [
                f"n_sync_rows: {len(sprint2.sync.sync_index)}",
                f"n_bouts: {len(sprint2.behavior_bouts)}",
                f"n_ethogram_rows: {len(sprint2.ethogram_summary)}",
                f"export_profiles: {sorted(sprint2.export_profiles)}",
            ]
        )
    if qc_summary is not None:
        lines.extend(
            [
                f"qc_structural_ok: {qc_summary.structural_ok}",
                f"qc_warning_count: {len(qc_summary.warnings)}",
                f"qc_failure_count: {len(qc_summary.failures)}",
                f"qc_informational_count: {len(qc_summary.informational)}",
            ]
        )
    if error_message:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (session_root / RESULTS_PACKAGE_DIRNAME)
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        sync_index=run_dir / SYNC_INDEX_FILENAME,
        behavior_bouts=run_dir / BEHAVIOR_BOUTS_FILENAME,
        ethogram_summary=run_dir / ETHOGRAM_SUMMARY_FILENAME,
        package_manifest=run_dir / PACKAGE_MANIFEST_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
        qc=run_dir / QC_FILENAME,
        export_profiles_root=run_dir / EXPORT_PROFILES_DIRNAME,
    )


def _reject_output_overlap(run_dir: Path, protected: tuple[Path, ...]) -> None:
    resolved_run = run_dir.resolve()
    for protected_path in protected:
        try:
            resolved_protected = Path(protected_path).resolve()
        except OSError:
            continue
        if (
            resolved_run == resolved_protected
            or _is_relative_to(resolved_run, resolved_protected)
            or _is_relative_to(resolved_protected, resolved_run)
        ):
            raise PublicationError(
                "S9 run directory must not overlap S1–S8 artifacts: "
                f"{resolved_run} vs {resolved_protected}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _resolve_publish_config(request: Any) -> ResultsPackagingConfig:
    from results_packaging.config import config_from_mapping, load_results_packaging_config
    from results_packaging.contracts import IntakeError

    if request.config is not None and request.settings_overrides:
        raise IntakeError("Pass either config or settings_overrides, not both.")
    if request.config is not None:
        cfg = request.config.validated()
    elif request.settings_overrides is not None:
        cfg = config_from_mapping(dict(request.settings_overrides))
    else:
        cfg = load_results_packaging_config(None)
    resolved: ResultsPackagingConfig = replace(
        cfg,
        execution=replace(cfg.execution, publish_run=True),
    )
    return resolved


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


def _assert_recorded_digest(
    *,
    label: str,
    recorded: str | None,
    path: Path,
    required: bool = True,
) -> None:
    if recorded is None:
        if required:
            raise ProvenanceError(f"S9 provenance missing recorded digest for {label}.")
        return
    if not path.is_file():
        raise ProvenanceError(f"Recorded {label} artifact missing: {path}")
    current = sha256_file(path)
    if current.lower() != recorded.lower():
        raise ProvenanceError(
            f"{label} SHA256 no longer matches the digest recorded by S9."
        )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^0-9T]", "", str(value).strip())
    if re.fullmatch(r"\d{8}T\d{6}", cleaned) is None:
        raise PublicationError(f"Invalid S9 run timestamp: {value!r}")
    return cleaned


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# Silence unused import in type checkers when profile constant is referenced indirectly.
_ = REVIEW_EMIT_PROFILE_ID
