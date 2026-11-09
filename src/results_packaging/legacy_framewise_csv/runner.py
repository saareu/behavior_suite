"""Headless entry points for legacy framewise CSV export (v1 / v2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from preprocess.models import OperationProgress
from results_packaging.legacy_framewise_csv.config import LegacyFramewiseExportSettings
from results_packaging.legacy_framewise_csv.contracts import (
    BEHAVIOR_SOURCE_AUTOMATED,
    BEHAVIOR_SOURCE_REVIEWED,
    DEFAULT_CSV_FORMAT_PROFILE,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LegacyFramewiseError,
    is_legacy_framewise_profile,
)
from results_packaging.legacy_framewise_csv.publication import publish_legacy_framewise_csv
from results_packaging.legacy_framewise_csv.source import (
    resolve_automated_legacy_framewise_source,
    resolve_reviewed_legacy_framewise_source,
)
from results_packaging.media_csv_export.results import CsvExportJobResult

ProgressCallback = Callable[[OperationProgress], None]


@dataclass(frozen=True, slots=True)
class LegacyFramewiseExportRequest:
    session_root: Path
    package_dir: Path | None = None
    review_run_dir: Path | None = None
    behavior_source_kind: str = BEHAVIOR_SOURCE_AUTOMATED
    settings: LegacyFramewiseExportSettings | None = None
    format_profile: str | None = None
    timestamp: str | None = None
    export_job_id: str | None = None
    output_root: Path | None = None
    allow_stale_source: bool = False
    # Required only when S6→S4 lineage is unavailable; ignored when it matches lineage.
    pose_run_dir: Path | None = None


def run_legacy_framewise_csv_export(
    request: LegacyFramewiseExportRequest,
    progress_callback: ProgressCallback | None = None,
) -> CsvExportJobResult:
    """Publish a legacy framewise CSV from automated S8 or published S10."""

    profile = _resolve_format_profile(request)
    settings = (
        request.settings.validated()
        if request.settings is not None
        else LegacyFramewiseExportSettings(
            behavior_source_kind=request.behavior_source_kind,
            format_profile=profile,
        ).validated()
    )
    if settings.format_profile != profile:
        raise LegacyFramewiseError(
            "Request format_profile disagrees with settings: "
            f"{profile!r} vs {settings.format_profile!r}."
        )
    kind = str(request.behavior_source_kind or settings.behavior_source_kind)
    session_root = Path(request.session_root).expanduser().resolve()

    if kind == BEHAVIOR_SOURCE_AUTOMATED:
        if request.package_dir is None:
            raise LegacyFramewiseError(
                f"Automated {profile} export requires package_dir."
            )
        if request.review_run_dir is not None:
            raise LegacyFramewiseError(
                "Automated export must not receive review_run_dir."
            )
        source = resolve_automated_legacy_framewise_source(
            session_root=session_root,
            package_dir=request.package_dir,
            allow_stale_source=request.allow_stale_source,
            pose_run_dir=request.pose_run_dir,
            format_profile=profile,
        )
    elif kind == BEHAVIOR_SOURCE_REVIEWED:
        if request.review_run_dir is None:
            raise LegacyFramewiseError(
                f"Reviewed {profile} export requires review_run_dir."
            )
        source = resolve_reviewed_legacy_framewise_source(
            session_root=session_root,
            review_run_dir=request.review_run_dir,
            package_dir=request.package_dir,
            allow_stale_source=request.allow_stale_source,
            pose_run_dir=request.pose_run_dir,
            format_profile=profile,
        )
    else:
        raise LegacyFramewiseError(f"Unsupported behavior_source_kind {kind!r}.")

    return publish_legacy_framewise_csv(
        source=source,
        settings_snapshot=settings.to_settings_dict(),
        format_profile=profile,
        export_job_id=request.export_job_id,
        timestamp=request.timestamp,
        output_root=request.output_root,
        progress_callback=progress_callback,
    )


def _resolve_format_profile(request: LegacyFramewiseExportRequest) -> str:
    if request.settings is not None:
        profile = str(request.settings.validated().format_profile).strip()
    elif request.format_profile is not None:
        profile = str(request.format_profile).strip()
    else:
        profile = str(DEFAULT_CSV_FORMAT_PROFILE).strip()
    if not is_legacy_framewise_profile(profile):
        raise LegacyFramewiseError(
            "format_profile must be "
            f"{LEGACY_FRAMEWISE_PROFILE_ID!r} or {LEGACY_FRAMEWISE_V2_PROFILE_ID!r}; "
            f"got {profile!r}."
        )
    return profile
