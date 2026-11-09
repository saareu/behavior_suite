"""Headless entry points for media and CSV export jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from preprocess.models import OperationProgress
from results_packaging.media_csv_export.config import (
    CsvExportSettings,
    MediaExportSettings,
    SelectionSpec,
)
from results_packaging.media_csv_export.publication import (
    publish_csv_export,
    publish_media_export,
    retry_media_export_job,
)
from results_packaging.media_csv_export.results import (
    CsvExportJobResult,
    MediaExportJobResult,
)
from results_packaging.media_csv_export.source import validate_package_for_export

ProgressCallback = Callable[[OperationProgress], None]


@dataclass(frozen=True, slots=True)
class MediaExportRequest:
    session_root: Path
    package_dir: Path
    settings: MediaExportSettings | None = None
    selection: SelectionSpec | None = None
    pre_context_sec: float | None = None
    post_context_sec: float | None = None
    timestamp: str | None = None
    export_job_id: str | None = None
    output_root: Path | None = None
    allow_stale_source: bool = False


@dataclass(frozen=True, slots=True)
class CsvExportRequest:
    session_root: Path
    package_dir: Path
    settings: CsvExportSettings | None = None
    selection: SelectionSpec | None = None
    timestamp: str | None = None
    export_job_id: str | None = None
    output_root: Path | None = None
    allow_stale_source: bool = False
    pose_run_dir: Path | str | None = None


def run_media_export(
    request: MediaExportRequest,
    progress_callback: ProgressCallback | None = None,
) -> MediaExportJobResult:
    """Validate package and publish a new media export job."""

    settings = _resolve_media_settings(request)
    source = validate_package_for_export(
        request.package_dir,
        require_video=True,
        allow_stale_source=request.allow_stale_source,
    )
    session_root = Path(request.session_root).expanduser().resolve()
    return publish_media_export(
        session_root=session_root,
        source=source,
        settings=settings,
        export_job_id=request.export_job_id,
        timestamp=request.timestamp,
        output_root=request.output_root,
        progress_callback=progress_callback,
    )


def retry_media_export(
    job_dir: Path | str,
    progress_callback: ProgressCallback | None = None,
) -> MediaExportJobResult:
    """Resume an incomplete/failed media export job (same-job retry model)."""

    return retry_media_export_job(job_dir, progress_callback=progress_callback)


def run_csv_export(
    request: CsvExportRequest,
    progress_callback: ProgressCallback | None = None,
) -> CsvExportJobResult:
    """Validate package and publish a new automated CSV export job.

    Default format profile is ``legacy_framewise_v2``. Pass
    ``format_profile=legacy_framewise_v1`` for the dense-row laboratory CSV, or
    ``format_profile=package_bouts_automated`` for the historical bout table.
    """

    settings = _resolve_csv_settings(request)
    from results_packaging.legacy_framewise_csv.contracts import (
        is_legacy_framewise_profile,
    )
    from results_packaging.legacy_framewise_csv.runner import (
        LegacyFramewiseExportRequest,
        run_legacy_framewise_csv_export,
    )

    if is_legacy_framewise_profile(settings.format_profile):
        # Pose is resolved from S6 lineage on the package; explicit pose_run_dir
        # is required when lineage is absent (never latest-S4 fallback).
        pose_dir = request.pose_run_dir
        if pose_dir is None and settings.pose_run_dir:
            pose_dir = settings.pose_run_dir
        return run_legacy_framewise_csv_export(
            LegacyFramewiseExportRequest(
                session_root=Path(request.session_root).expanduser().resolve(),
                package_dir=Path(request.package_dir).expanduser().resolve(),
                behavior_source_kind="automated_s8",
                format_profile=settings.format_profile,
                timestamp=request.timestamp,
                export_job_id=request.export_job_id,
                output_root=request.output_root,
                allow_stale_source=request.allow_stale_source,
                pose_run_dir=(
                    Path(pose_dir).expanduser().resolve() if pose_dir is not None else None
                ),
            ),
            progress_callback=progress_callback,
        )

    source = validate_package_for_export(
        request.package_dir,
        require_video=False,
        allow_stale_source=request.allow_stale_source,
    )
    session_root = Path(request.session_root).expanduser().resolve()
    return publish_csv_export(
        session_root=session_root,
        source=source,
        settings=settings,
        export_job_id=request.export_job_id,
        timestamp=request.timestamp,
        output_root=request.output_root,
        progress_callback=progress_callback,
    )


def _resolve_media_settings(request: MediaExportRequest) -> MediaExportSettings:
    if request.settings is not None:
        cfg = request.settings.validated()
    else:
        cfg = MediaExportSettings().validated()
    updates: dict[str, object] = {}
    if request.selection is not None:
        updates["selection"] = request.selection.validated()
    if request.pre_context_sec is not None:
        updates["pre_context_sec"] = float(request.pre_context_sec)
    if request.post_context_sec is not None:
        updates["post_context_sec"] = float(request.post_context_sec)
    if updates:
        from dataclasses import replace

        cfg = replace(cfg, **updates).validated()  # type: ignore[arg-type]
    return cfg


def _resolve_csv_settings(request: CsvExportRequest) -> CsvExportSettings:
    if request.settings is not None:
        cfg = request.settings.validated()
    else:
        cfg = CsvExportSettings().validated()
    if request.selection is not None:
        from dataclasses import replace

        cfg = replace(cfg, selection=request.selection.validated()).validated()
    return cfg
