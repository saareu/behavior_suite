"""Immutable media/CSV export-job publication, encode loop, and same-job retry."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from preprocess.models import OperationProgress
from project.file_digest import sha256_file
from results_packaging.contracts import (
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from results_packaging.export import atomic_parquet_write, write_json, write_yaml
from results_packaging.media_csv_export.clip_index import (
    build_pending_clip_index,
    clip_bounds_from_index_row,
)
from results_packaging.media_csv_export.config import (
    CsvExportSettings,
    MediaExportSettings,
    selection_from_mapping,
)
from results_packaging.media_csv_export.contracts import (
    BEHAVIORAL_FPS,
    CLIP_INDEX_FILENAME,
    CLIPS_DIRNAME,
    CSV_EXPORT_FILENAME,
    CSV_EXPORT_RUN_PREFIX,
    ENCODE_STATUS_FAILED,
    ENCODE_STATUS_OMITTED_INVALID_BOUNDS,
    ENCODE_STATUS_PENDING,
    ENCODE_STATUS_SKIPPED_EXISTS,
    ENCODE_STATUS_SUCCEEDED,
    EXPORT_JOB_MANIFEST_FILENAME,
    EXPORT_MANIFEST_SCHEMA_VERSION,
    EXPORT_QC_SCHEMA_VERSION,
    EXPORT_RUN_META_SCHEMA_VERSION,
    FAILURES_JSONL_FILENAME,
    JOB_KIND_CSV,
    JOB_KIND_MEDIA,
    MEDIA_EXPORT_RUN_PREFIX,
    RESULTS_EXPORTS_DIRNAME,
    STATUS_INCOMPLETE,
    ExportPublicationError,
    ExportSourceError,
    ExportVideoError,
)
from results_packaging.media_csv_export.csv_writer import (
    build_automated_bout_csv_frame,
    write_automated_bout_csv,
)
from results_packaging.media_csv_export.results import (
    CsvExportJobResult,
    MediaExportJobResult,
)
from results_packaging.media_csv_export.selection import select_bouts_for_export
from results_packaging.media_csv_export.source import (
    ExportSourcePackage,
    assert_source_unchanged,
    validate_package_for_export,
)
from results_packaging.media_csv_export.video import encode_mp4_clip

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
    "results_package",
)


@dataclass(frozen=True, slots=True)
class _MediaPaths:
    job_dir: Path
    run_meta: Path
    settings_used: Path
    manifest: Path
    qc: Path
    processing_log: Path
    clip_index: Path
    failures: Path
    clips_root: Path


@dataclass(frozen=True, slots=True)
class _CsvPaths:
    job_dir: Path
    run_meta: Path
    settings_used: Path
    manifest: Path
    qc: Path
    processing_log: Path
    csv_path: Path


def publish_media_export(
    *,
    session_root: Path,
    source: ExportSourcePackage,
    settings: MediaExportSettings,
    export_job_id: str | None = None,
    timestamp: str | None = None,
    output_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    resume_job_dir: Path | None = None,
) -> MediaExportJobResult:
    """Create or resume a media export job under results_exports/."""

    cfg = settings.validated()
    root = Path(session_root).expanduser().resolve()
    if resume_job_dir is not None:
        return _resume_media_export(
            session_root=root,
            source=source,
            settings=cfg,
            job_dir=Path(resume_job_dir).expanduser().resolve(),
            progress_callback=progress_callback,
        )

    ts = _sanitize_timestamp(timestamp or _now_timestamp())
    job_id = export_job_id or f"{MEDIA_EXPORT_RUN_PREFIX}__{ts}"
    if not str(job_id).startswith(f"{MEDIA_EXPORT_RUN_PREFIX}__"):
        job_id = f"{MEDIA_EXPORT_RUN_PREFIX}__{ts}"
    paths = _media_paths(root, job_id, output_root=output_root)
    _reject_output_overlap(paths.job_dir, root, source.package_dir)
    if paths.job_dir.exists():
        raise ExportPublicationError(f"Media export job already exists: {paths.job_dir}")

    if source.video_probe is None or source.source_video_path is None:
        raise ExportPublicationError("Media export requires a probed prepared video.")

    selected = select_bouts_for_export(
        source.behavior_bouts,
        cfg.selection,
        package_dir=source.package_dir,
    )
    created_at = _iso_now()
    paths.job_dir.mkdir(parents=True, exist_ok=False)
    settings_snapshot = {
        **cfg.to_settings_dict(),
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_package_bouts_sha256": source.behavior_bouts_sha256,
        "source_package_sync_sha256": source.sync_index_sha256,
        "source_package_manifest_sha256": source.package_manifest_sha256,
        "export_job_id": job_id,
    }
    warnings: list[str] = []
    if source.stale_warning:
        warnings.append(source.stale_warning)

    write_yaml(paths.settings_used, {**settings_snapshot, "status": STATUS_RUNNING})
    write_json(
        paths.run_meta,
        _media_run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=root,
            source=source,
            settings_snapshot=settings_snapshot,
            paths=paths,
            counts=None,
            error_message=None,
            warnings=warnings,
        ),
    )

    clip_index = build_pending_clip_index(
        selected,
        export_job_id=job_id,
        selection_mode=cfg.selection.mode,
        source_video_kind=source.source_video_kind,
        source_video_path=str(source.source_video_path),
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        pre_context_sec=cfg.pre_context_sec,
        post_context_sec=cfg.post_context_sec,
        crf=cfg.crf,
        codec=cfg.codec,
    )
    atomic_parquet_write(paths.clip_index, clip_index)
    paths.clips_root.mkdir(parents=True, exist_ok=True)

    return _encode_media_batch(
        session_root=root,
        source=source,
        settings=cfg,
        settings_snapshot=settings_snapshot,
        paths=paths,
        job_id=job_id,
        created_at=created_at,
        clip_index=clip_index,
        warnings=warnings,
        progress_callback=progress_callback,
        retry=False,
    )


def retry_media_export_job(
    job_dir: Path | str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> MediaExportJobResult:
    """Resume an incomplete/failed media job without overwriting successful clips."""

    path = Path(job_dir).expanduser().resolve()
    meta = _load_json(path / RUN_META_FILENAME)
    status = meta.get("status")
    if status not in (STATUS_INCOMPLETE, STATUS_FAILED, STATUS_RUNNING):
        raise ExportPublicationError(
            "Media export retry is only allowed for incomplete/failed/running jobs; "
            f"got status={status!r}. Completed jobs are immutable — create a new job."
        )
    settings_raw = _load_yaml(path / SETTINGS_USED_FILENAME)
    settings = MediaExportSettings(
        schema_version=str(settings_raw.get("schema_version", "")),
        job_kind=str(settings_raw.get("job_kind", JOB_KIND_MEDIA)),
        selection=selection_from_mapping(settings_raw.get("selection")),
        pre_context_sec=float(settings_raw.get("pre_context_sec", 0.0)),
        post_context_sec=float(settings_raw.get("post_context_sec", 0.0)),
        crf=int(settings_raw.get("crf", 22)),
        codec=str(settings_raw.get("codec", "libx264")),
        pixel_format=str(settings_raw.get("pixel_format", "yuv420p")),
        source_video_kind=str(settings_raw.get("source_video_kind", "prepared_video")),
        fail_fast=bool(settings_raw.get("fail_fast", False)),
        accept_partial=bool(settings_raw.get("accept_partial", False)),
        hash_clips=bool(settings_raw.get("hash_clips", False)),
    ).validated()

    package_dir = Path(str(settings_raw["source_package_dir"]))
    # Same MVP eligibility gate as new media exports (no silent stale retry).
    source = validate_package_for_export(
        package_dir, require_video=True, allow_stale_source=False
    )
    recorded_bouts = str(settings_raw.get("source_package_bouts_sha256", ""))
    if recorded_bouts and recorded_bouts.lower() != source.behavior_bouts_sha256.lower():
        raise ExportSourceError(
            "Cannot retry: source package behavior_bouts digest no longer matches "
            "the digest recorded in this export job."
        )
    session_root = Path(str(meta.get("session_root") or source.session_root or path.parent.parent))
    return publish_media_export(
        session_root=session_root,
        source=source,
        settings=settings,
        progress_callback=progress_callback,
        resume_job_dir=path,
    )


def publish_csv_export(
    *,
    session_root: Path,
    source: ExportSourcePackage,
    settings: CsvExportSettings,
    export_job_id: str | None = None,
    timestamp: str | None = None,
    output_root: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CsvExportJobResult:
    """Publish an automated bout CSV export job."""

    cfg = settings.validated()
    root = Path(session_root).expanduser().resolve()
    ts = _sanitize_timestamp(timestamp or _now_timestamp())
    job_id = export_job_id or f"{CSV_EXPORT_RUN_PREFIX}__{ts}"
    if not str(job_id).startswith(f"{CSV_EXPORT_RUN_PREFIX}__"):
        job_id = f"{CSV_EXPORT_RUN_PREFIX}__{ts}"
    paths = _csv_paths(root, job_id, output_root=output_root)
    _reject_output_overlap(paths.job_dir, root, source.package_dir)
    if paths.job_dir.exists():
        raise ExportPublicationError(f"CSV export job already exists: {paths.job_dir}")

    _report(
        progress_callback,
        phase="validate",
        message="Selecting bouts for CSV export…",
        completed_units=0,
        total_units=3,
        indeterminate=False,
    )
    selected = select_bouts_for_export(
        source.behavior_bouts,
        cfg.selection,
        package_dir=source.package_dir,
    )
    created_at = _iso_now()
    paths.job_dir.mkdir(parents=True, exist_ok=False)
    settings_snapshot = {
        **cfg.to_settings_dict(),
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_package_bouts_sha256": source.behavior_bouts_sha256,
        "source_package_sync_sha256": source.sync_index_sha256,
        "source_package_manifest_sha256": source.package_manifest_sha256,
        "export_job_id": job_id,
    }
    warnings: list[str] = []
    if source.stale_warning:
        warnings.append(source.stale_warning)

    write_yaml(paths.settings_used, {**settings_snapshot, "status": STATUS_RUNNING})
    write_json(
        paths.run_meta,
        _csv_run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=root,
            source=source,
            settings_snapshot=settings_snapshot,
            paths=paths,
            n_rows=None,
            error_message=None,
            warnings=warnings,
        ),
    )

    status = STATUS_FAILED
    error_message: str | None = None
    csv_written = False
    n_rows = 0
    try:
        _report(
            progress_callback,
            phase="export",
            message="Writing automated bout CSV…",
            completed_units=1,
            total_units=3,
            indeterminate=False,
        )
        table = build_automated_bout_csv_frame(
            selected,
            export_job_id=job_id,
            selection_mode=cfg.selection.mode,
            source_package_run_id=source.package_run_id,
        )
        write_automated_bout_csv(paths.csv_path, table)
        csv_written = True
        n_rows = int(len(table))
        assert_source_unchanged(source, require_video=False)
        status = STATUS_COMPLETED
        _report(
            progress_callback,
            phase="complete",
            message=f"CSV export completed ({n_rows} rows).",
            completed_units=3,
            total_units=3,
            indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        error_message = str(exc)
        if paths.csv_path.exists():
            with suppress(OSError):
                paths.csv_path.unlink()
        _report(
            progress_callback,
            phase="failed",
            message=f"CSV export failed: {exc}",
            completed_units=None,
            total_units=3,
            indeterminate=True,
        )

    completed_at = _iso_now()
    qc = _csv_qc(
        status=status,
        n_rows=n_rows,
        warnings=warnings,
        error_message=error_message,
        source=source,
        selection_mode=cfg.selection.mode,
    )
    manifest = _csv_manifest(
        job_id=job_id,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        source=source,
        settings_snapshot=settings_snapshot,
        paths=paths,
        n_rows=n_rows,
        csv_written=csv_written,
        warnings=warnings,
        error_message=error_message,
    )
    write_json(paths.qc, qc)
    write_json(paths.manifest, manifest)
    write_json(
        paths.run_meta,
        _csv_run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            session_root=root,
            source=source,
            settings_snapshot=settings_snapshot,
            paths=paths,
            n_rows=n_rows if csv_written else None,
            error_message=error_message,
            warnings=warnings,
        ),
    )
    write_yaml(paths.settings_used, {**settings_snapshot, "status": status})
    paths.processing_log.write_text(
        "\n".join(
            [
                f"export_job_id: {job_id}",
                f"job_kind: {JOB_KIND_CSV}",
                f"status: {status}",
                f"n_rows: {n_rows}",
                f"source_package_run_id: {source.package_run_id}",
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
        diagnostics={"csv_written": csv_written},
    )


def _resume_media_export(
    *,
    session_root: Path,
    source: ExportSourcePackage,
    settings: MediaExportSettings,
    job_dir: Path,
    progress_callback: ProgressCallback | None,
) -> MediaExportJobResult:
    paths = _media_paths_from_job_dir(job_dir)
    if not paths.clip_index.is_file():
        raise ExportPublicationError(
            f"Cannot retry media export without clip_index.parquet: {paths.clip_index}"
        )
    clip_index = pd.read_parquet(paths.clip_index)
    settings_snapshot = {
        **settings.to_settings_dict(),
        "source_package_run_id": source.package_run_id,
        "source_package_dir": str(source.package_dir),
        "source_package_bouts_sha256": source.behavior_bouts_sha256,
        "source_package_sync_sha256": source.sync_index_sha256,
        "source_package_manifest_sha256": source.package_manifest_sha256,
        "export_job_id": job_dir.name,
    }
    meta = _load_json(paths.run_meta)
    created_at = str(meta.get("created_at") or _iso_now())
    warnings: list[str] = []
    if source.stale_warning:
        warnings.append(source.stale_warning)
    write_json(
        paths.run_meta,
        _media_run_meta(
            job_id=job_dir.name,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=session_root,
            source=source,
            settings_snapshot=settings_snapshot,
            paths=paths,
            counts=None,
            error_message=None,
            warnings=warnings,
        ),
    )
    return _encode_media_batch(
        session_root=session_root,
        source=source,
        settings=settings,
        settings_snapshot=settings_snapshot,
        paths=paths,
        job_id=job_dir.name,
        created_at=created_at,
        clip_index=clip_index,
        warnings=warnings,
        progress_callback=progress_callback,
        retry=True,
    )


def _encode_media_batch(
    *,
    session_root: Path,
    source: ExportSourcePackage,
    settings: MediaExportSettings,
    settings_snapshot: Mapping[str, Any],
    paths: _MediaPaths,
    job_id: str,
    created_at: str,
    clip_index: pd.DataFrame,
    warnings: list[str],
    progress_callback: ProgressCallback | None,
    retry: bool,
) -> MediaExportJobResult:
    assert source.source_video_path is not None
    total = int(len(clip_index))
    hard_error: str | None = None
    try:
        for i, idx in enumerate(list(clip_index.index)):
            row = clip_index.loc[idx]
            bout_id = str(row["bout_id"])
            status = str(row["encode_status"])
            rel = str(row["clip_relpath"])
            out_path = paths.clips_root / Path(rel)
            _report(
                progress_callback,
                phase="encode",
                message=f"Encoding clip {i + 1}/{total}: {bout_id}",
                completed_units=i,
                total_units=max(total, 1),
                indeterminate=False,
            )

            if status == ENCODE_STATUS_OMITTED_INVALID_BOUNDS:
                continue

            if status in (ENCODE_STATUS_SUCCEEDED, ENCODE_STATUS_SKIPPED_EXISTS):
                if out_path.is_file() and out_path.stat().st_size > 0:
                    clip_index.at[idx, "clip_exists"] = True
                    continue
                # Successful ledger row but missing file → re-encode on retry/pass.
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_PENDING
                status = ENCODE_STATUS_PENDING

            if status not in (ENCODE_STATUS_PENDING, ENCODE_STATUS_FAILED):
                continue

            if out_path.is_file() and out_path.stat().st_size > 0:
                # Never overwrite an existing successful file.
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_SKIPPED_EXISTS
                clip_index.at[idx, "clip_exists"] = True
                clip_index.at[idx, "error"] = None
                continue

            if pd.isna(row.get("clip_frame_idx_start")) or pd.isna(
                row.get("clip_frame_count")
            ):
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_OMITTED_INVALID_BOUNDS
                clip_index.at[idx, "error"] = "Missing clip bounds."
                _append_failure(paths.failures, bout_id, "Missing clip bounds.")
                continue

            try:
                bounds = clip_bounds_from_index_row(row)
                encode_mp4_clip(
                    source_video=source.source_video_path,
                    output_path=out_path,
                    bounds=bounds,
                    crf=settings.crf,
                    codec=settings.codec,
                    pixel_format=settings.pixel_format,
                )
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_SUCCEEDED
                clip_index.at[idx, "clip_exists"] = True
                clip_index.at[idx, "error"] = None
            except ExportVideoError as exc:
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_FAILED
                clip_index.at[idx, "clip_exists"] = False
                clip_index.at[idx, "error"] = str(exc)
                _append_failure(paths.failures, bout_id, str(exc))
                # Clean partial tmp if any.
                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                if tmp.exists():
                    with suppress(OSError):
                        tmp.unlink()
                if settings.fail_fast:
                    hard_error = str(exc)
                    break

            # Persist ledger after each clip for crash resilience.
            atomic_parquet_write(paths.clip_index, clip_index)

        assert_source_unchanged(source, require_video=True)
    except Exception as exc:
        hard_error = str(exc)
        atomic_parquet_write(paths.clip_index, clip_index)

    counts = _count_encode_statuses(clip_index)
    error_message: str | None = None
    if hard_error and counts["n_succeeded"] == 0 and counts["n_skipped"] == 0:
        final_status = STATUS_FAILED
        error_message = hard_error
    elif counts["n_failed"] > 0 or counts["n_pending"] > 0:
        if settings.accept_partial and counts["n_pending"] == 0:
            final_status = STATUS_COMPLETED_WITH_WARNINGS
            warnings.append(
                f"Accepted partial media export: {counts['n_failed']} failed clips."
            )
            error_message = hard_error
        else:
            final_status = STATUS_INCOMPLETE
            error_message = hard_error
            if counts["n_failed"] > 0:
                warnings.append(
                    f"Media export incomplete: {counts['n_failed']} failed clips; "
                    "retry with retry_media_export_job()."
                )
    else:
        final_status = STATUS_COMPLETED
        error_message = None

    completed_at = _iso_now()
    qc = _media_qc(
        status=final_status,
        counts=counts,
        warnings=warnings,
        error_message=error_message,
        source=source,
        settings=settings,
        clip_index=clip_index,
    )
    manifest = _media_manifest(
        job_id=job_id,
        created_at=created_at,
        completed_at=completed_at,
        status=final_status,
        source=source,
        settings_snapshot=settings_snapshot,
        paths=paths,
        counts=counts,
        warnings=warnings,
        error_message=error_message,
        hash_clips=settings.hash_clips,
    )
    write_json(paths.qc, qc)
    write_json(paths.manifest, manifest)
    write_json(
        paths.run_meta,
        _media_run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=completed_at,
            status=final_status,
            session_root=session_root,
            source=source,
            settings_snapshot=settings_snapshot,
            paths=paths,
            counts=counts,
            error_message=error_message,
            warnings=warnings,
        ),
    )
    write_yaml(paths.settings_used, {**dict(settings_snapshot), "status": final_status})
    atomic_parquet_write(paths.clip_index, clip_index)
    paths.processing_log.write_text(
        "\n".join(
            [
                f"export_job_id: {job_id}",
                f"job_kind: {JOB_KIND_MEDIA}",
                f"status: {final_status}",
                f"n_selected: {counts['n_selected']}",
                f"n_succeeded: {counts['n_succeeded']}",
                f"n_failed: {counts['n_failed']}",
                f"n_skipped: {counts['n_skipped']}",
                f"n_omitted: {counts['n_omitted']}",
                f"source_package_run_id: {source.package_run_id}",
                f"error: {error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _report(
        progress_callback,
        phase="complete" if final_status == STATUS_COMPLETED else final_status,
        message=f"Media export {final_status}.",
        completed_units=total,
        total_units=max(total, 1),
        indeterminate=False,
    )

    return MediaExportJobResult(
        success=final_status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS),
        status=final_status,
        export_job_id=job_id,
        job_dir=paths.job_dir,
        session_root=session_root,
        source_package_run_id=source.package_run_id,
        source_package_dir=source.package_dir,
        n_selected=counts["n_selected"],
        n_succeeded=counts["n_succeeded"],
        n_failed=counts["n_failed"],
        n_skipped=counts["n_skipped"],
        n_omitted=counts["n_omitted"],
        clip_index_path=paths.clip_index,
        export_job_manifest_path=paths.manifest,
        qc_path=paths.qc,
        warnings=tuple(warnings),
        error_message=error_message,
        diagnostics={"retry": retry, "counts": counts},
    )


def _count_encode_statuses(clip_index: pd.DataFrame) -> dict[str, int]:
    statuses = (
        clip_index["encode_status"].astype(str)
        if not clip_index.empty
        else pd.Series(dtype=str)
    )
    return {
        "n_selected": int(len(clip_index)),
        "n_succeeded": int((statuses == ENCODE_STATUS_SUCCEEDED).sum()),
        "n_skipped": int((statuses == ENCODE_STATUS_SKIPPED_EXISTS).sum()),
        "n_failed": int((statuses == ENCODE_STATUS_FAILED).sum()),
        "n_omitted": int((statuses == ENCODE_STATUS_OMITTED_INVALID_BOUNDS).sum()),
        "n_pending": int((statuses == ENCODE_STATUS_PENDING).sum()),
    }


def _media_paths(
    session_root: Path, job_id: str, *, output_root: Path | None
) -> _MediaPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (session_root / RESULTS_EXPORTS_DIRNAME)
    )
    return _media_paths_from_job_dir(root / job_id)


def _media_paths_from_job_dir(job_dir: Path) -> _MediaPaths:
    return _MediaPaths(
        job_dir=job_dir,
        run_meta=job_dir / RUN_META_FILENAME,
        settings_used=job_dir / SETTINGS_USED_FILENAME,
        manifest=job_dir / EXPORT_JOB_MANIFEST_FILENAME,
        qc=job_dir / QC_FILENAME,
        processing_log=job_dir / PROCESSING_LOG_FILENAME,
        clip_index=job_dir / CLIP_INDEX_FILENAME,
        failures=job_dir / FAILURES_JSONL_FILENAME,
        clips_root=job_dir / CLIPS_DIRNAME,
    )


def _csv_paths(
    session_root: Path, job_id: str, *, output_root: Path | None
) -> _CsvPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (session_root / RESULTS_EXPORTS_DIRNAME)
    )
    job_dir = root / job_id
    return _CsvPaths(
        job_dir=job_dir,
        run_meta=job_dir / RUN_META_FILENAME,
        settings_used=job_dir / SETTINGS_USED_FILENAME,
        manifest=job_dir / EXPORT_JOB_MANIFEST_FILENAME,
        qc=job_dir / QC_FILENAME,
        processing_log=job_dir / PROCESSING_LOG_FILENAME,
        csv_path=job_dir / CSV_EXPORT_FILENAME,
    )


def _reject_output_overlap(job_dir: Path, session_root: Path, package_dir: Path) -> None:
    resolved = job_dir.resolve()
    protected = [package_dir.resolve(), *(session_root / name for name in _PROTECTED_SESSION_SUBDIRS)]
    for item in protected:
        try:
            resolved_item = item.resolve()
        except OSError:
            continue
        if resolved == resolved_item or _is_relative_to(resolved, resolved_item) or _is_relative_to(
            resolved_item, resolved
        ):
            raise ExportPublicationError(
                "Export job directory must not overlap S1–S9 package/source trees: "
                f"{resolved} vs {resolved_item}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _append_failure(path: Path, bout_id: str, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"bout_id": bout_id, "error": error, "at": _iso_now()},
                sort_keys=True,
            )
            + "\n"
        )


def _media_run_meta(**kwargs: Any) -> dict[str, Any]:
    source: ExportSourcePackage = kwargs["source"]
    paths: _MediaPaths = kwargs["paths"]
    counts = kwargs["counts"]
    return {
        "schema_version": EXPORT_RUN_META_SCHEMA_VERSION,
        "export_job_id": kwargs["job_id"],
        "job_kind": JOB_KIND_MEDIA,
        "created_at": kwargs["created_at"],
        "completed_at": kwargs["completed_at"],
        "status": kwargs["status"],
        "session_root": str(kwargs["session_root"]),
        "source_package": {
            "run_id": source.package_run_id,
            "path": str(source.package_dir),
            "status": source.package_status,
            "behavior_bouts_sha256": source.behavior_bouts_sha256,
            "sync_index_sha256": source.sync_index_sha256,
            "package_manifest_sha256": source.package_manifest_sha256,
            "currently_eligible": source.currently_eligible,
        },
        "source_s8": {"run_id": source.source_s8_run_id, "run_dir": str(source.source_s8_run_dir) if source.source_s8_run_dir else None},
        "source_s7": {"run_id": source.source_s7_run_id},
        "source_s6": {"run_id": source.source_s6_run_id},
        "source_video": {
            "kind": source.source_video_kind,
            "path": str(source.source_video_path) if source.source_video_path else None,
            "fps": source.video_probe.fps if source.video_probe else None,
            "n_frames": source.video_probe.n_frames if source.video_probe else None,
            "fingerprint": source.source_video_fingerprint,
            "sha256": source.source_video_sha256,
            "verification_strength": source.source_video_verification_strength,
        },
        "settings": dict(kwargs["settings_snapshot"]),
        "counts": counts,
        "outputs": {
            "clip_index": str(paths.clip_index),
            "clips_root": str(paths.clips_root),
            "export_job_manifest": str(paths.manifest),
            "qc": str(paths.qc),
            "failures": str(paths.failures),
        },
        "warnings": list(kwargs["warnings"]),
        "error": kwargs["error_message"],
        "timing_conventions": {
            "behavioral_fps": BEHAVIORAL_FPS,
            "clip_playback_fps": "source_video_fps",
            "context_padding_fps": "source_video_fps",
        },
    }


def _csv_run_meta(**kwargs: Any) -> dict[str, Any]:
    source: ExportSourcePackage = kwargs["source"]
    paths: _CsvPaths = kwargs["paths"]
    return {
        "schema_version": EXPORT_RUN_META_SCHEMA_VERSION,
        "export_job_id": kwargs["job_id"],
        "job_kind": JOB_KIND_CSV,
        "created_at": kwargs["created_at"],
        "completed_at": kwargs["completed_at"],
        "status": kwargs["status"],
        "session_root": str(kwargs["session_root"]),
        "source_package": {
            "run_id": source.package_run_id,
            "path": str(source.package_dir),
            "status": source.package_status,
            "behavior_bouts_sha256": source.behavior_bouts_sha256,
            "sync_index_sha256": source.sync_index_sha256,
            "package_manifest_sha256": source.package_manifest_sha256,
            "currently_eligible": source.currently_eligible,
        },
        "source_s8": {"run_id": source.source_s8_run_id},
        "source_s7": {"run_id": source.source_s7_run_id},
        "source_s6": {"run_id": source.source_s6_run_id},
        "settings": dict(kwargs["settings_snapshot"]),
        "n_rows": kwargs["n_rows"],
        "outputs": {
            "csv": str(paths.csv_path),
            "export_job_manifest": str(paths.manifest),
            "qc": str(paths.qc),
        },
        "warnings": list(kwargs["warnings"]),
        "error": kwargs["error_message"],
    }


def _media_qc(**kwargs: Any) -> dict[str, Any]:
    counts = kwargs["counts"]
    clip_index: pd.DataFrame = kwargs["clip_index"]
    settings: MediaExportSettings = kwargs["settings"]
    source: ExportSourcePackage = kwargs["source"]
    return {
        "schema_version": EXPORT_QC_SCHEMA_VERSION,
        "status": kwargs["status"],
        "structural_ok": kwargs["status"] != STATUS_FAILED,
        "counts": counts,
        "warnings": list(kwargs["warnings"]),
        "failures": [kwargs["error_message"]] if kwargs["error_message"] else [],
        "selection_mode": settings.selection.mode,
        "source_package_run_id": source.package_run_id,
        "timing_conventions": {
            "behavioral_fps": BEHAVIORAL_FPS,
            "clip_playback_fps": source.video_probe.fps if source.video_probe else None,
        },
        "clip_duration_sec_sum": (
            float(clip_index["clip_duration_sec"].fillna(0).sum())
            if not clip_index.empty
            else 0.0
        ),
    }


def _csv_qc(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_QC_SCHEMA_VERSION,
        "status": kwargs["status"],
        "structural_ok": kwargs["status"] == STATUS_COMPLETED,
        "n_rows": kwargs["n_rows"],
        "warnings": list(kwargs["warnings"]),
        "failures": [kwargs["error_message"]] if kwargs["error_message"] else [],
        "selection_mode": kwargs["selection_mode"],
        "source_package_run_id": kwargs["source"].package_run_id,
        "export_provenance": "automated",
    }


def _media_manifest(**kwargs: Any) -> dict[str, Any]:
    paths: _MediaPaths = kwargs["paths"]
    source: ExportSourcePackage = kwargs["source"]
    artifacts = [
        {
            "role": "clip_index",
            "relative_path": CLIP_INDEX_FILENAME,
            "sha256": sha256_file(paths.clip_index) if paths.clip_index.is_file() else None,
        },
        {
            "role": "settings_used",
            "relative_path": SETTINGS_USED_FILENAME,
            "sha256": sha256_file(paths.settings_used) if paths.settings_used.is_file() else None,
        },
    ]
    clip_listing: list[dict[str, Any]] = []
    if paths.clips_root.is_dir():
        for mp4 in sorted(paths.clips_root.rglob("*.mp4")):
            stat = mp4.stat()
            entry: dict[str, Any] = {
                "relative_path": str(mp4.relative_to(paths.job_dir)).replace("\\", "/"),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            if kwargs["hash_clips"]:
                entry["sha256"] = sha256_file(mp4)
            clip_listing.append(entry)
    return {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "export_job_id": kwargs["job_id"],
        "job_kind": JOB_KIND_MEDIA,
        "created_at": kwargs["created_at"],
        "completed_at": kwargs["completed_at"],
        "status": kwargs["status"],
        "source_package": {
            "run_id": source.package_run_id,
            "path": str(source.package_dir),
            "bouts_sha256": source.behavior_bouts_sha256,
            "sync_sha256": source.sync_index_sha256,
            "manifest_sha256": source.package_manifest_sha256,
        },
        "source_s8_run_id": source.source_s8_run_id,
        "source_s7_run_id": source.source_s7_run_id,
        "source_s6_run_id": source.source_s6_run_id,
        "source_video": {
            "path": str(source.source_video_path) if source.source_video_path else None,
            "fps": source.video_probe.fps if source.video_probe else None,
            "n_frames": source.video_probe.n_frames if source.video_probe else None,
            "fingerprint": source.source_video_fingerprint,
            "sha256": source.source_video_sha256,
        },
        "selection": dict(kwargs["settings_snapshot"]).get("selection"),
        "counts": kwargs["counts"],
        "artifacts": artifacts,
        "clips": clip_listing,
        "warnings": list(kwargs["warnings"]),
        "error": kwargs["error_message"],
    }


def _csv_manifest(**kwargs: Any) -> dict[str, Any]:
    paths: _CsvPaths = kwargs["paths"]
    source: ExportSourcePackage = kwargs["source"]
    return {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "export_job_id": kwargs["job_id"],
        "job_kind": JOB_KIND_CSV,
        "created_at": kwargs["created_at"],
        "completed_at": kwargs["completed_at"],
        "status": kwargs["status"],
        "source_package": {
            "run_id": source.package_run_id,
            "path": str(source.package_dir),
            "bouts_sha256": source.behavior_bouts_sha256,
            "sync_sha256": source.sync_index_sha256,
            "manifest_sha256": source.package_manifest_sha256,
        },
        "source_s8_run_id": source.source_s8_run_id,
        "source_s7_run_id": source.source_s7_run_id,
        "source_s6_run_id": source.source_s6_run_id,
        "selection": dict(kwargs["settings_snapshot"]).get("selection"),
        "n_rows": kwargs["n_rows"],
        "artifacts": [
            {
                "role": "behavior_bouts_automated_csv",
                "relative_path": CSV_EXPORT_FILENAME,
                "sha256": (
                    sha256_file(paths.csv_path)
                    if kwargs["csv_written"] and paths.csv_path.is_file()
                    else None
                ),
            }
        ],
        "warnings": list(kwargs["warnings"]),
        "error": kwargs["error_message"],
        "export_provenance": "automated",
    }


def _report(
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed_units: int | None,
    total_units: int | None,
    indeterminate: bool,
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
            is_indeterminate=indeterminate,
            processed_video_time_sec=None,
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExportPublicationError(f"Missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExportPublicationError(f"JSON root must be object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise ExportPublicationError(f"Missing YAML: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExportPublicationError(f"YAML root must be mapping: {path}")
    return payload


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^0-9T]", "", str(value).strip())
    if re.fullmatch(r"\d{8}T\d{6}", cleaned) is None:
        raise ExportPublicationError(f"Invalid export job timestamp: {value!r}")
    return cleaned


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")
