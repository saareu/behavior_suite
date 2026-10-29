"""Discover and reopen media/CSV export jobs under results_exports/."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from results_packaging.contracts import (
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from results_packaging.media_csv_export.contracts import (
    CLIP_INDEX_FILENAME,
    CSV_EXPORT_FILENAME,
    CSV_EXPORT_RUN_PREFIX,
    EXPORT_COMPLETED_WITH_WARNINGS,
    EXPORT_ELIGIBLE,
    EXPORT_FAILED_OR_INCOMPLETE,
    EXPORT_INCOMPLETE,
    EXPORT_JOB_MANIFEST_FILENAME,
    EXPORT_MISSING_REQUIRED,
    EXPORT_STALE_SOURCE,
    JOB_KIND_CSV,
    JOB_KIND_MEDIA,
    MEDIA_EXPORT_RUN_PREFIX,
    RESULTS_EXPORTS_DIRNAME,
    STATUS_INCOMPLETE,
    ExportPublicationError,
)
from results_packaging.media_csv_export.source import validate_package_for_export


@dataclass(frozen=True, slots=True)
class ExportJobSummary:
    job_dir: Path
    export_job_id: str
    job_kind: str | None
    status: str | None
    classification: str
    source_package_run_id: str | None
    source_package_dir: Path | None
    created_at: str | None
    completed_at: str | None
    n_selected: int | None
    n_succeeded: int | None
    n_failed: int | None
    n_rows: int | None
    error_message: str | None
    staleness_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExportProjectSummary:
    session_root: Path
    exports_root: Path
    exports_present: bool
    jobs: tuple[ExportJobSummary, ...]

    @property
    def media_jobs(self) -> tuple[ExportJobSummary, ...]:
        return tuple(j for j in self.jobs if j.job_kind == JOB_KIND_MEDIA)

    @property
    def csv_jobs(self) -> tuple[ExportJobSummary, ...]:
        return tuple(j for j in self.jobs if j.job_kind == JOB_KIND_CSV)

    @property
    def completed_jobs(self) -> tuple[ExportJobSummary, ...]:
        return tuple(
            j
            for j in self.jobs
            if j.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        )


def summarize_export_project(session_root: Path | str) -> ExportProjectSummary:
    root = Path(session_root).expanduser().resolve()
    exports_root = root / RESULTS_EXPORTS_DIRNAME
    jobs = tuple(
        sorted(_discover_jobs(exports_root), key=_job_sort_key, reverse=True)
    )
    return ExportProjectSummary(
        session_root=root,
        exports_root=exports_root,
        exports_present=exports_root.is_dir(),
        jobs=jobs,
    )


def list_completed_export_jobs(
    session_root: Path | str,
    *,
    job_kind: str | None = None,
) -> tuple[ExportJobSummary, ...]:
    summary = summarize_export_project(session_root)
    jobs = summary.completed_jobs
    if job_kind is not None:
        jobs = tuple(j for j in jobs if j.job_kind == job_kind)
    return jobs


def reopen_export_job(job_dir: Path | str) -> dict[str, Any]:
    """Load an export job and re-check source package digests (no mutation)."""

    path = Path(job_dir).expanduser().resolve()
    if not path.is_dir():
        raise ExportPublicationError(f"Export job directory missing: {path}")
    meta_path = path / RUN_META_FILENAME
    if not meta_path.is_file():
        raise ExportPublicationError(f"Export run_meta.json missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ExportPublicationError("Export run_meta.json must be an object.")

    source_package = meta.get("source_package")
    package_dir_text = None
    if isinstance(source_package, Mapping):
        package_dir_text = source_package.get("path")
    if not package_dir_text:
        raise ExportPublicationError("Export job missing source_package.path.")

    require_video = meta.get("job_kind") == JOB_KIND_MEDIA
    source = validate_package_for_export(
        Path(str(package_dir_text)),
        require_video=require_video,
        allow_stale_source=True,
    )
    recorded = (
        source_package.get("behavior_bouts_sha256")
        if isinstance(source_package, Mapping)
        else None
    )
    stale: list[str] = []
    if recorded and str(recorded).lower() != source.behavior_bouts_sha256.lower():
        stale.append("behavior_bouts digest drift vs export job record")
    if source.stale_warning:
        stale.append(source.stale_warning)

    payload: dict[str, Any] = {
        "job_dir": path,
        "export_job_id": meta.get("export_job_id") or path.name,
        "job_kind": meta.get("job_kind"),
        "status": meta.get("status"),
        "run_meta": meta,
        "source": source,
        "stale_reasons": tuple(stale),
        "currently_source_ok": not stale,
    }
    manifest_path = path / EXPORT_JOB_MANIFEST_FILENAME
    if manifest_path.is_file():
        payload["export_job_manifest"] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    qc_path = path / QC_FILENAME
    if qc_path.is_file():
        payload["qc"] = json.loads(qc_path.read_text(encoding="utf-8"))
    if meta.get("job_kind") == JOB_KIND_MEDIA and (path / CLIP_INDEX_FILENAME).is_file():
        import pandas as pd  # type: ignore[import-untyped]

        payload["clip_index"] = pd.read_parquet(path / CLIP_INDEX_FILENAME)
    if meta.get("job_kind") == JOB_KIND_CSV and (path / CSV_EXPORT_FILENAME).is_file():
        payload["csv_path"] = path / CSV_EXPORT_FILENAME
    return payload


def _discover_jobs(exports_root: Path) -> list[ExportJobSummary]:
    if not exports_root.is_dir():
        return []
    jobs: list[ExportJobSummary] = []
    for child in exports_root.iterdir():
        if not child.is_dir():
            continue
        if not (
            child.name.startswith(f"{MEDIA_EXPORT_RUN_PREFIX}__")
            or child.name.startswith(f"{CSV_EXPORT_RUN_PREFIX}__")
        ):
            continue
        jobs.append(_summarize_job(child))
    return jobs


def _summarize_job(job_dir: Path) -> ExportJobSummary:
    meta_path = job_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        return ExportJobSummary(
            job_dir=job_dir,
            export_job_id=job_dir.name,
            job_kind=_infer_kind(job_dir.name),
            status=None,
            classification=EXPORT_MISSING_REQUIRED,
            source_package_run_id=None,
            source_package_dir=None,
            created_at=None,
            completed_at=None,
            n_selected=None,
            n_succeeded=None,
            n_failed=None,
            n_rows=None,
            error_message="missing run_meta.json",
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ExportJobSummary(
            job_dir=job_dir,
            export_job_id=job_dir.name,
            job_kind=_infer_kind(job_dir.name),
            status=None,
            classification=EXPORT_MISSING_REQUIRED,
            source_package_run_id=None,
            source_package_dir=None,
            created_at=None,
            completed_at=None,
            n_selected=None,
            n_succeeded=None,
            n_failed=None,
            n_rows=None,
            error_message=str(exc),
        )

    status = meta.get("status") if isinstance(meta, dict) else None
    job_kind = meta.get("job_kind") if isinstance(meta, dict) else _infer_kind(job_dir.name)
    source_package = meta.get("source_package") if isinstance(meta, dict) else None
    package_run_id = None
    package_dir = None
    if isinstance(source_package, Mapping):
        package_run_id = source_package.get("run_id")
        path_text = source_package.get("path")
        if path_text:
            package_dir = Path(str(path_text))

    counts = meta.get("counts") if isinstance(meta, dict) else None
    n_selected = n_succeeded = n_failed = None
    if isinstance(counts, Mapping):
        n_selected = _as_int(counts.get("n_selected"))
        n_succeeded = _as_int(counts.get("n_succeeded"))
        n_failed = _as_int(counts.get("n_failed"))
    n_rows = _as_int(meta.get("n_rows")) if isinstance(meta, dict) else None

    classification, stale = _classify(job_dir, status=status, job_kind=job_kind, meta=meta)
    return ExportJobSummary(
        job_dir=job_dir,
        export_job_id=str(meta.get("export_job_id") or job_dir.name),
        job_kind=str(job_kind) if job_kind else None,
        status=str(status) if status else None,
        classification=classification,
        source_package_run_id=str(package_run_id) if package_run_id else None,
        source_package_dir=package_dir,
        created_at=_optional_str(meta.get("created_at") if isinstance(meta, dict) else None),
        completed_at=_optional_str(
            meta.get("completed_at") if isinstance(meta, dict) else None
        ),
        n_selected=n_selected,
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        n_rows=n_rows,
        error_message=_optional_str(meta.get("error") if isinstance(meta, dict) else None),
        staleness_reasons=stale,
    )


def _classify(
    job_dir: Path,
    *,
    status: Any,
    job_kind: Any,
    meta: Any,
) -> tuple[str, tuple[str, ...]]:
    if status == STATUS_INCOMPLETE:
        base = EXPORT_INCOMPLETE
    elif status == STATUS_COMPLETED:
        base = EXPORT_ELIGIBLE
    elif status == STATUS_COMPLETED_WITH_WARNINGS:
        base = EXPORT_COMPLETED_WITH_WARNINGS
    elif status in (STATUS_FAILED, STATUS_RUNNING) or status is None:
        base = EXPORT_FAILED_OR_INCOMPLETE
    else:
        base = EXPORT_FAILED_OR_INCOMPLETE

    required = [RUN_META_FILENAME, EXPORT_JOB_MANIFEST_FILENAME, QC_FILENAME]
    if job_kind == JOB_KIND_MEDIA:
        required.append(CLIP_INDEX_FILENAME)
    if job_kind == JOB_KIND_CSV and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        required.append(CSV_EXPORT_FILENAME)
    for name in required:
        if status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS) and not (
            job_dir / name
        ).is_file():
            return EXPORT_MISSING_REQUIRED, ()

    stale: list[str] = []
    if isinstance(meta, dict):
        source_package = meta.get("source_package")
        if isinstance(source_package, Mapping) and source_package.get("path"):
            try:
                require_video = job_kind == JOB_KIND_MEDIA
                source = validate_package_for_export(
                    Path(str(source_package["path"])),
                    require_video=require_video,
                    allow_stale_source=True,
                )
                recorded = source_package.get("behavior_bouts_sha256")
                if (
                    recorded
                    and str(recorded).lower() != source.behavior_bouts_sha256.lower()
                ):
                    stale.append("package bouts digest drift")
                if source.stale_warning:
                    stale.append("upstream stale")
            except Exception as exc:  # noqa: BLE001 - discovery must not raise
                stale.append(str(exc))
    if stale and base in (EXPORT_ELIGIBLE, EXPORT_COMPLETED_WITH_WARNINGS):
        return EXPORT_STALE_SOURCE, tuple(stale)
    return base, tuple(stale)


def _infer_kind(name: str) -> str | None:
    if name.startswith(f"{MEDIA_EXPORT_RUN_PREFIX}__"):
        return JOB_KIND_MEDIA
    if name.startswith(f"{CSV_EXPORT_RUN_PREFIX}__"):
        return JOB_KIND_CSV
    return None


def _job_sort_key(job: ExportJobSummary) -> tuple[Any, ...]:
    created = job.created_at or ""
    try:
        ts = datetime.fromisoformat(created)
    except ValueError:
        ts = datetime.min
    return (ts, job.export_job_id)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
