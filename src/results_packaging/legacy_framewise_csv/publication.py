"""Immutable legacy framewise CSV export jobs under results_exports/."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from preprocess.models import OperationProgress
from project.file_digest import sha256_file
from results_packaging.contracts import (
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from results_packaging.export import write_json, write_yaml
from results_packaging.legacy_framewise_csv.builder import build_legacy_framewise_table
from results_packaging.legacy_framewise_csv.builder_v2 import (
    build_legacy_framewise_v2_table,
)
from results_packaging.legacy_framewise_csv.contracts import (
    COORDINATE_FORMAT_POLICY,
    COORDINATE_SPACE,
    DENSE_FRAME_INDEXING,
    EXPORT_MANIFEST_SCHEMA_VERSION,
    EXPORT_MANIFEST_SCHEMA_VERSION_V2,
    FRAME_ALIGNMENT_COLUMNS,
    INDEXING_NOTES,
    INDEXING_NOTES_V2,
    LEGACY_FRAMEWISE_CSV_FILENAME,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_PROFILE_VERSION,
    LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME,
    LEGACY_FRAMEWISE_V2_CSV_FILENAME,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_PROFILE_VERSION,
    ORIGINAL_FRAME_INDEXING,
    PLATEAU_POLICY_LAST_IN_PLATEAU,
    PREPARED_FRAME_INDEXING,
    LegacyFramewiseError,
    is_legacy_framewise_profile,
)
from results_packaging.legacy_framewise_csv.source import LegacyFramewiseSource
from results_packaging.legacy_framewise_csv.writer import (
    validate_legacy_framewise_csv_bytes,
    write_legacy_framewise_csv,
)
from results_packaging.media_csv_export.contracts import (
    CSV_EXPORT_RUN_PREFIX,
    EXPORT_JOB_MANIFEST_FILENAME,
    EXPORT_QC_SCHEMA_VERSION,
    EXPORT_RUN_META_SCHEMA_VERSION,
    JOB_KIND_CSV,
    RESULTS_EXPORTS_DIRNAME,
    ExportPublicationError,
)
from results_packaging.media_csv_export.results import CsvExportJobResult

ProgressCallback = Callable[[OperationProgress], None]

_PROTECTED_SESSION_SUBDIRS = (
    "preprocess",
    "pose_inference",
    "tracking_correction",
    "pose_finalization",
    "blob_features",
    "combined_features",
    "behavioral_inference",
    "behavioral_arbitration",
    "behavioral_review",
    "results_package",
)


@dataclass(frozen=True, slots=True)
class _Paths:
    job_dir: Path
    run_meta: Path
    settings_used: Path
    manifest: Path
    qc: Path
    processing_log: Path
    csv_path: Path
    alignment_path: Path | None


def publish_legacy_framewise_csv(
    *,
    source: LegacyFramewiseSource,
    settings_snapshot: dict[str, Any],
    format_profile: str = LEGACY_FRAMEWISE_PROFILE_ID,
    export_job_id: str | None = None,
    timestamp: str | None = None,
    output_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CsvExportJobResult:
    """Write one immutable legacy framewise CSV export job (v1 or v2)."""

    profile = str(format_profile or source.format_profile).strip()
    if not is_legacy_framewise_profile(profile):
        raise LegacyFramewiseError(f"Unsupported format_profile {profile!r}.")
    if profile == LEGACY_FRAMEWISE_V2_PROFILE_ID:
        return _publish_v2(
            source=source,
            settings_snapshot=settings_snapshot,
            export_job_id=export_job_id,
            timestamp=timestamp,
            output_root=output_root,
            progress_callback=progress_callback,
        )
    return _publish_v1(
        source=source,
        settings_snapshot=settings_snapshot,
        export_job_id=export_job_id,
        timestamp=timestamp,
        output_root=output_root,
        progress_callback=progress_callback,
    )


def _publish_v1(
    *,
    source: LegacyFramewiseSource,
    settings_snapshot: dict[str, Any],
    export_job_id: str | None,
    timestamp: str | None,
    output_root: Path | None,
    progress_callback: ProgressCallback | None,
) -> CsvExportJobResult:
    root = source.session_root
    ts = _sanitize_timestamp(timestamp or _now_timestamp())
    job_id = export_job_id or f"{CSV_EXPORT_RUN_PREFIX}__{ts}"
    if not str(job_id).startswith(f"{CSV_EXPORT_RUN_PREFIX}__"):
        job_id = f"{CSV_EXPORT_RUN_PREFIX}__{ts}"
    paths = _paths(
        root,
        job_id,
        output_root=output_root,
        csv_filename=LEGACY_FRAMEWISE_CSV_FILENAME,
        alignment_filename=None,
    )
    _reject_output_overlap(paths.job_dir, root, source.package_dir)
    if source.review_run_dir is not None:
        _reject_output_overlap(paths.job_dir, root, source.review_run_dir)
    if paths.job_dir.exists():
        raise ExportPublicationError(f"CSV export job already exists: {paths.job_dir}")

    created_at = _iso_now()
    warnings = list(source.warnings)
    snapshot = {
        **settings_snapshot,
        "format_profile": LEGACY_FRAMEWISE_PROFILE_ID,
        "format_profile_version": LEGACY_FRAMEWISE_PROFILE_VERSION,
        "behavior_source_kind": source.behavior_source_kind,
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_review_run_id": source.review_run_id,
        "source_review_run_dir": (
            str(source.review_run_dir) if source.review_run_dir else None
        ),
        "export_job_id": job_id,
        "pinned_digests": dict(source.digests),
        "pose_run_id": source.pose_run_id,
        "pose_path": str(source.final_pose_path),
        "pose_selection_mode": source.pose_selection_mode,
        "pose_alignment": dict(source.pose_alignment),
        "original_frame_mapping_provenance": dict(
            source.original_frame_mapping_provenance
        ),
    }

    paths.job_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(paths.settings_used, {**snapshot, "status": STATUS_RUNNING})
    write_json(
        paths.run_meta,
        _run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            source=source,
            settings_snapshot=snapshot,
            paths=paths,
            n_rows=None,
            error_message=None,
            warnings=warnings,
            format_profile=LEGACY_FRAMEWISE_PROFILE_ID,
        ),
    )

    status = STATUS_FAILED
    error_message: str | None = None
    csv_written = False
    n_rows = 0
    build_diag: dict[str, Any] = {}
    try:
        _report(
            progress_callback,
            phase="build",
            message="Building legacy_framewise_v1 table…",
            completed_units=1,
            total_units=3,
            indeterminate=False,
        )
        built = build_legacy_framewise_table(
            sync_index=source.sync_index,
            behavior_states=source.behavior_states,
            final_pose=source.final_pose,
        )
        warnings.extend(built.warnings)
        build_diag = {
            **built.diagnostics,
            "n_unique_original_frames": built.n_unique_original_frames,
            "n_duplicate_original_frame_rows": built.n_duplicate_original_frame_rows,
            "n_missing_pose_coordinates": built.n_missing_pose_coordinates,
            "n_empty_behavior_labels": built.n_empty_behavior_labels,
        }
        _report(
            progress_callback,
            phase="write",
            message="Writing legacy_framewise_v1.csv…",
            completed_units=2,
            total_units=3,
            indeterminate=False,
        )
        write_legacy_framewise_csv(paths.csv_path, built.table)
        payload = paths.csv_path.read_bytes()
        format_qc = validate_legacy_framewise_csv_bytes(payload)
        if int(format_qc["n_data_rows"]) != int(built.n_rows):
            raise LegacyFramewiseError(
                "Written CSV row count does not match built table."
            )
        csv_written = True
        n_rows = int(built.n_rows)
        status = STATUS_COMPLETED
        _report(
            progress_callback,
            phase="complete",
            message=f"legacy_framewise_v1 export completed ({n_rows} rows).",
            completed_units=3,
            total_units=3,
            indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        error_message = str(exc)
        if paths.csv_path.exists():
            with contextlib.suppress(OSError):
                paths.csv_path.unlink()
        _report(
            progress_callback,
            phase="failed",
            message=f"legacy_framewise_v1 export failed: {exc}",
            completed_units=None,
            total_units=3,
            indeterminate=True,
        )

    completed_at = _iso_now()
    csv_sha = sha256_file(paths.csv_path) if csv_written and paths.csv_path.is_file() else None
    qc = {
        "schema_version": EXPORT_QC_SCHEMA_VERSION,
        "status": status,
        "format_profile": LEGACY_FRAMEWISE_PROFILE_ID,
        "n_rows": n_rows,
        "csv_sha256": csv_sha,
        "warnings": list(warnings),
        "error_message": error_message,
        "build": build_diag,
        "no_upstream_science_writes": True,
    }
    manifest = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "export_manifest_schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "job_kind": JOB_KIND_CSV,
        "export_job_id": job_id,
        "format_profile": LEGACY_FRAMEWISE_PROFILE_ID,
        "format_profile_version": LEGACY_FRAMEWISE_PROFILE_VERSION,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "behavior_source_kind": source.behavior_source_kind,
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_review_run_id": source.review_run_id,
        "source_review_run_dir": (
            str(source.review_run_dir) if source.review_run_dir else None
        ),
        "pose_source": {
            "run_id": source.pose_run_id,
            "run_dir": str(source.pose_run_dir) if source.pose_run_dir else None,
            "path": str(source.final_pose_path),
            "sha256": source.final_pose_sha256,
            "selection_mode": source.pose_selection_mode,
            "alignment": dict(source.pose_alignment),
            "coordinate_space": COORDINATE_SPACE,
            "coordinate_format_policy": COORDINATE_FORMAT_POLICY,
            "rounding_scope": "legacy_framewise_csv_export_serialization_only",
            "scientific_pose_unchanged": True,
        },
        "original_frame_mapping": {
            **source.original_frame_mapping_provenance,
            "indexing": ORIGINAL_FRAME_INDEXING,
            "dense_indexing": DENSE_FRAME_INDEXING,
            "notes": INDEXING_NOTES,
            "sync_index_path": str(source.sync_index_path),
            "sync_index_sha256": source.sync_index_sha256,
        },
        "pinned_source_digests": dict(source.digests),
        "behavior_states": {
            "path": str(source.behavior_states_path),
            "sha256": source.behavior_states_sha256,
        },
        "artifacts": {
            "csv": LEGACY_FRAMEWISE_CSV_FILENAME,
            "csv_sha256": csv_sha,
            "n_rows": n_rows,
        },
        "format_validation": {
            "n_columns": 23,
            "lineterminator": "CRLF",
            "header_contract": LEGACY_FRAMEWISE_PROFILE_ID,
        },
        "warnings": list(warnings),
        "error_message": error_message,
        "build": build_diag,
        "settings": snapshot,
    }
    write_json(paths.qc, qc)
    write_json(paths.manifest, manifest)
    write_json(
        paths.run_meta,
        _run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            source=source,
            settings_snapshot=snapshot,
            paths=paths,
            n_rows=n_rows if csv_written else None,
            error_message=error_message,
            warnings=warnings,
            format_profile=LEGACY_FRAMEWISE_PROFILE_ID,
        ),
    )
    write_yaml(paths.settings_used, {**snapshot, "status": status})
    paths.processing_log.write_text(
        "\n".join(
            [
                f"export_job_id: {job_id}",
                f"job_kind: {JOB_KIND_CSV}",
                f"format_profile: {LEGACY_FRAMEWISE_PROFILE_ID}",
                f"behavior_source_kind: {source.behavior_source_kind}",
                f"status: {status}",
                f"n_rows: {n_rows}",
                f"source_package_run_id: {source.package_run_id}",
                f"source_review_run_id: {source.review_run_id}",
                f"pose_run_id: {source.pose_run_id}",
                f"error: {error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return CsvExportJobResult(
        success=status == STATUS_COMPLETED,
        status=status,
        export_job_id=job_id,
        job_dir=paths.job_dir,
        session_root=root,
        source_package_run_id=source.package_run_id,
        source_package_dir=source.package_dir,
        n_rows=n_rows,
        csv_path=paths.csv_path if csv_written else None,
        export_job_manifest_path=paths.manifest,
        qc_path=paths.qc,
        warnings=tuple(warnings),
        error_message=error_message,
        diagnostics={
            "format_profile": LEGACY_FRAMEWISE_PROFILE_ID,
            "behavior_source_kind": source.behavior_source_kind,
            "csv_written": csv_written,
            **build_diag,
        },
    )


def _publish_v2(
    *,
    source: LegacyFramewiseSource,
    settings_snapshot: dict[str, Any],
    export_job_id: str | None,
    timestamp: str | None,
    output_root: Path | None,
    progress_callback: ProgressCallback | None,
) -> CsvExportJobResult:
    if (
        source.prepared_frame_idx is None
        or source.raw_decode_frame_idx is None
        or source.behavior_sync is None
        or source.pose_sync is None
        or source.prepared_sync_path is None
        or source.n_prepared_frames is None
    ):
        raise LegacyFramewiseError(
            "legacy_framewise_v2 source is missing prepared-frame timing authorities "
            "(prepared_sync / behavior SyncMap / independent S4 SyncMap)."
        )

    root = source.session_root
    ts = _sanitize_timestamp(timestamp or _now_timestamp())
    job_id = export_job_id or f"{CSV_EXPORT_RUN_PREFIX}__{ts}"
    if not str(job_id).startswith(f"{CSV_EXPORT_RUN_PREFIX}__"):
        job_id = f"{CSV_EXPORT_RUN_PREFIX}__{ts}"
    paths = _paths(
        root,
        job_id,
        output_root=output_root,
        csv_filename=LEGACY_FRAMEWISE_V2_CSV_FILENAME,
        alignment_filename=LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME,
    )
    assert paths.alignment_path is not None
    _reject_output_overlap(paths.job_dir, root, source.package_dir)
    if source.review_run_dir is not None:
        _reject_output_overlap(paths.job_dir, root, source.review_run_dir)
    if paths.job_dir.exists():
        raise ExportPublicationError(f"CSV export job already exists: {paths.job_dir}")

    created_at = _iso_now()
    warnings = list(source.warnings)
    snapshot = {
        **settings_snapshot,
        "format_profile": LEGACY_FRAMEWISE_V2_PROFILE_ID,
        "format_profile_version": LEGACY_FRAMEWISE_V2_PROFILE_VERSION,
        "timing_contract": dict(source.timing_contract or {}),
        "behavior_source_kind": source.behavior_source_kind,
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_review_run_id": source.review_run_id,
        "source_review_run_dir": (
            str(source.review_run_dir) if source.review_run_dir else None
        ),
        "export_job_id": job_id,
        "pinned_digests": dict(source.digests),
        "pose_run_id": source.pose_run_id,
        "pose_path": str(source.final_pose_path),
        "pose_selection_mode": source.pose_selection_mode,
        "pose_alignment": dict(source.pose_alignment),
        "original_frame_mapping_provenance": dict(
            source.original_frame_mapping_provenance
        ),
    }

    paths.job_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(paths.settings_used, {**snapshot, "status": STATUS_RUNNING})
    write_json(
        paths.run_meta,
        _run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            source=source,
            settings_snapshot=snapshot,
            paths=paths,
            n_rows=None,
            error_message=None,
            warnings=warnings,
            format_profile=LEGACY_FRAMEWISE_V2_PROFILE_ID,
        ),
    )

    status = STATUS_FAILED
    error_message: str | None = None
    csv_written = False
    alignment_written = False
    n_rows = 0
    build_diag: dict[str, Any] = {}
    alignment_sha: str | None = None
    try:
        _report(
            progress_callback,
            phase="build",
            message="Building legacy_framewise_v2 prepared-frame table…",
            completed_units=1,
            total_units=4,
            indeterminate=False,
        )
        built = build_legacy_framewise_v2_table(
            prepared_frame_idx=source.prepared_frame_idx,
            raw_decode_frame_idx=source.raw_decode_frame_idx,
            behavior_sync=source.behavior_sync,
            pose_sync=source.pose_sync,
            behavior_states=source.behavior_states,
            final_pose=source.final_pose,
        )
        warnings.extend(built.warnings)
        build_diag = {
            **built.diagnostics,
            "n_unique_original_frames": built.n_unique_original_frames,
            "n_duplicate_original_frame_rows": built.n_duplicate_original_frame_rows,
            "n_missing_pose_coordinates": built.n_missing_pose_coordinates,
            "n_empty_behavior_labels": built.n_empty_behavior_labels,
            "n_behavior_covered": built.n_behavior_covered,
            "n_pose_covered": built.n_pose_covered,
        }
        _report(
            progress_callback,
            phase="write_csv",
            message="Writing legacy_framewise_v2.csv…",
            completed_units=2,
            total_units=4,
            indeterminate=False,
        )
        write_legacy_framewise_csv(paths.csv_path, built.table)
        payload = paths.csv_path.read_bytes()
        format_qc = validate_legacy_framewise_csv_bytes(payload)
        if int(format_qc["n_data_rows"]) != int(built.n_rows):
            raise LegacyFramewiseError(
                "Written CSV row count does not match built table."
            )
        csv_written = True
        n_rows = int(built.n_rows)
        _report(
            progress_callback,
            phase="write_alignment",
            message="Writing frame-alignment sidecar…",
            completed_units=3,
            total_units=4,
            indeterminate=False,
        )
        built.alignment.to_parquet(paths.alignment_path, index=False)
        if int(len(built.alignment)) != n_rows:
            raise LegacyFramewiseError(
                "Alignment sidecar row count does not match prepared-frame CSV."
            )
        alignment_sha = sha256_file(paths.alignment_path)
        alignment_written = True
        status = STATUS_COMPLETED
        _report(
            progress_callback,
            phase="complete",
            message=(
                f"legacy_framewise_v2 export completed ({n_rows} prepared-frame rows)."
            ),
            completed_units=4,
            total_units=4,
            indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        error_message = str(exc)
        if paths.csv_path.exists():
            with contextlib.suppress(OSError):
                paths.csv_path.unlink()
        if paths.alignment_path.exists():
            with contextlib.suppress(OSError):
                paths.alignment_path.unlink()
        _report(
            progress_callback,
            phase="failed",
            message=f"legacy_framewise_v2 export failed: {exc}",
            completed_units=None,
            total_units=4,
            indeterminate=True,
        )

    completed_at = _iso_now()
    csv_sha = sha256_file(paths.csv_path) if csv_written and paths.csv_path.is_file() else None
    if alignment_written and paths.alignment_path.is_file() and alignment_sha is None:
        alignment_sha = sha256_file(paths.alignment_path)

    frame_alignment_block = {
        "filename": LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME,
        "path": str(paths.alignment_path),
        "sha256": alignment_sha,
        "n_rows": n_rows if alignment_written else 0,
        "columns": list(FRAME_ALIGNMENT_COLUMNS),
        "mapping_authorities": {
            "prepared_frames": "s1_prepared_sync",
            "raw_decode_frame_idx": "s1_prepared_sync.raw_decode_frame_idx",
            "behavior_dense_frame_idx": (
                "results_package_sync_index + LAST_IN_PLATEAU"
            ),
            "pose_dense_frame_idx": (
                "s1_prepared_sync_nn_map_validated_against_s4_dense + LAST_IN_PLATEAU"
            ),
        },
        "timing_policies": {
            "behavior_plateau_policy": PLATEAU_POLICY_LAST_IN_PLATEAU,
            "pose_plateau_policy": PLATEAU_POLICY_LAST_IN_PLATEAU,
            "overlay_timing_semantics_reused": True,
        },
        "source_provenance": {
            "prepared_sync_path": str(source.prepared_sync_path),
            "prepared_sync_sha256": source.prepared_sync_sha256,
            "behavior_sync_index_path": str(source.sync_index_path),
            "behavior_sync_index_sha256": source.sync_index_sha256,
            "pose_run_id": source.pose_run_id,
            "pose_path": str(source.final_pose_path),
            "pose_sha256": source.final_pose_sha256,
            "pose_sync_build_audit": dict(source.pose_sync_build_audit or {}),
            "timing_contract": dict(source.timing_contract or {}),
        },
    }

    qc = {
        "schema_version": EXPORT_QC_SCHEMA_VERSION,
        "status": status,
        "format_profile": LEGACY_FRAMEWISE_V2_PROFILE_ID,
        "n_rows": n_rows,
        "csv_sha256": csv_sha,
        "frame_alignment_sha256": alignment_sha,
        "warnings": list(warnings),
        "error_message": error_message,
        "build": build_diag,
        "no_upstream_science_writes": True,
    }
    manifest = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION_V2,
        "export_manifest_schema_version": EXPORT_MANIFEST_SCHEMA_VERSION_V2,
        "job_kind": JOB_KIND_CSV,
        "export_job_id": job_id,
        "format_profile": LEGACY_FRAMEWISE_V2_PROFILE_ID,
        "format_profile_version": LEGACY_FRAMEWISE_V2_PROFILE_VERSION,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "behavior_source_kind": source.behavior_source_kind,
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_review_run_id": source.review_run_id,
        "source_review_run_dir": (
            str(source.review_run_dir) if source.review_run_dir else None
        ),
        "timing_contract": dict(source.timing_contract or {}),
        "pose_source": {
            "run_id": source.pose_run_id,
            "run_dir": str(source.pose_run_dir) if source.pose_run_dir else None,
            "path": str(source.final_pose_path),
            "sha256": source.final_pose_sha256,
            "selection_mode": source.pose_selection_mode,
            "alignment": dict(source.pose_alignment),
            "coordinate_space": COORDINATE_SPACE,
            "coordinate_format_policy": COORDINATE_FORMAT_POLICY,
            "rounding_scope": "legacy_framewise_csv_export_serialization_only",
            "scientific_pose_unchanged": True,
            "independent_timing_authority": (
                "s1_prepared_sync_nn_map_validated_against_s4_dense"
            ),
        },
        "original_frame_mapping": {
            **source.original_frame_mapping_provenance,
            "indexing": ORIGINAL_FRAME_INDEXING,
            "prepared_indexing": PREPARED_FRAME_INDEXING,
            "dense_indexing": (
                "nullable_behavioral_last_in_plateau_dense_sample_idx"
            ),
            "notes": INDEXING_NOTES_V2,
            "prepared_sync_path": str(source.prepared_sync_path),
            "prepared_sync_sha256": source.prepared_sync_sha256,
            "sync_index_path": str(source.sync_index_path),
            "sync_index_sha256": source.sync_index_sha256,
        },
        "frame_alignment": frame_alignment_block,
        "pinned_source_digests": dict(source.digests),
        "behavior_states": {
            "path": str(source.behavior_states_path),
            "sha256": source.behavior_states_sha256,
        },
        "artifacts": {
            "csv": LEGACY_FRAMEWISE_V2_CSV_FILENAME,
            "csv_sha256": csv_sha,
            "n_rows": n_rows,
            "frame_alignment": LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME,
            "frame_alignment_sha256": alignment_sha,
        },
        "format_validation": {
            "n_columns": 23,
            "lineterminator": "CRLF",
            "header_contract": LEGACY_FRAMEWISE_V2_PROFILE_ID,
            "row_domain": "s1_prepared_video_frames",
        },
        "warnings": list(warnings),
        "error_message": error_message,
        "build": build_diag,
        "settings": snapshot,
    }
    write_json(paths.qc, qc)
    write_json(paths.manifest, manifest)
    write_json(
        paths.run_meta,
        _run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            source=source,
            settings_snapshot=snapshot,
            paths=paths,
            n_rows=n_rows if csv_written else None,
            error_message=error_message,
            warnings=warnings,
            format_profile=LEGACY_FRAMEWISE_V2_PROFILE_ID,
        ),
    )
    write_yaml(paths.settings_used, {**snapshot, "status": status})
    paths.processing_log.write_text(
        "\n".join(
            [
                f"export_job_id: {job_id}",
                f"job_kind: {JOB_KIND_CSV}",
                f"format_profile: {LEGACY_FRAMEWISE_V2_PROFILE_ID}",
                f"behavior_source_kind: {source.behavior_source_kind}",
                f"status: {status}",
                f"n_rows: {n_rows}",
                f"n_prepared_frames: {source.n_prepared_frames}",
                f"source_package_run_id: {source.package_run_id}",
                f"source_review_run_id: {source.review_run_id}",
                f"pose_run_id: {source.pose_run_id}",
                f"frame_alignment: {LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME}",
                f"error: {error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return CsvExportJobResult(
        success=status == STATUS_COMPLETED,
        status=status,
        export_job_id=job_id,
        job_dir=paths.job_dir,
        session_root=root,
        source_package_run_id=source.package_run_id,
        source_package_dir=source.package_dir,
        n_rows=n_rows,
        csv_path=paths.csv_path if csv_written else None,
        export_job_manifest_path=paths.manifest,
        qc_path=paths.qc,
        warnings=tuple(warnings),
        error_message=error_message,
        diagnostics={
            "format_profile": LEGACY_FRAMEWISE_V2_PROFILE_ID,
            "behavior_source_kind": source.behavior_source_kind,
            "csv_written": csv_written,
            "alignment_written": alignment_written,
            "frame_alignment_sha256": alignment_sha,
            **build_diag,
        },
    )


def _paths(
    session_root: Path,
    job_id: str,
    *,
    output_root: Path | None,
    csv_filename: str,
    alignment_filename: str | None,
) -> _Paths:
    exports_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / RESULTS_EXPORTS_DIRNAME
    )
    job_dir = exports_root / job_id
    return _Paths(
        job_dir=job_dir,
        run_meta=job_dir / RUN_META_FILENAME,
        settings_used=job_dir / SETTINGS_USED_FILENAME,
        manifest=job_dir / EXPORT_JOB_MANIFEST_FILENAME,
        qc=job_dir / QC_FILENAME,
        processing_log=job_dir / PROCESSING_LOG_FILENAME,
        csv_path=job_dir / csv_filename,
        alignment_path=(
            job_dir / alignment_filename if alignment_filename is not None else None
        ),
    )


def _run_meta(
    *,
    job_id: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    source: LegacyFramewiseSource,
    settings_snapshot: dict[str, Any],
    paths: _Paths,
    n_rows: int | None,
    error_message: str | None,
    warnings: list[str],
    format_profile: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_RUN_META_SCHEMA_VERSION,
        "export_job_id": job_id,
        "job_kind": JOB_KIND_CSV,
        "format_profile": format_profile,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "session_root": str(source.session_root),
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_review_run_id": source.review_run_id,
        "behavior_source_kind": source.behavior_source_kind,
        "n_rows": n_rows,
        "csv_path": str(paths.csv_path),
        "settings": settings_snapshot,
        "warnings": list(warnings),
        "error_message": error_message,
    }


def _reject_output_overlap(job_dir: Path, session_root: Path, other: Path) -> None:
    resolved_job = job_dir.resolve()
    resolved_other = other.resolve()
    if resolved_job == resolved_other or resolved_other in resolved_job.parents:
        raise ExportPublicationError(
            "Export job directory overlaps a protected science path."
        )
    for name in _PROTECTED_SESSION_SUBDIRS:
        protected = (session_root / name).resolve()
        if resolved_job == protected or protected in resolved_job.parents:
            raise ExportPublicationError(
                f"Refusing to write export job under protected session path {name}/."
            )


def _report(
    callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed_units: int | None,
    total_units: int | None,
    indeterminate: bool,
) -> None:
    if callback is None:
        return
    callback(
        OperationProgress(
            phase=phase,
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            total_is_estimate=False,
            is_indeterminate=indeterminate,
            processed_video_time_sec=None,
        )
    )


def _now_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _sanitize_timestamp(value: str) -> str:
    text = "".join(ch for ch in str(value) if ch.isalnum() or ch in {"T", "_", "-"})
    return text or _now_timestamp()


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
