"""Export-specific UI settings and preview helpers for Subsystem 09."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BOUT_INDEX_FILENAME,
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    EXPORT_PROFILES_DIRNAME,
    STALE_SOURCE,
)
from results_packaging.media_csv_export.config import (
    CsvExportSettings,
    MediaExportSettings,
    SelectionSpec,
)
from results_packaging.media_csv_export.contracts import (
    SELECTION_ALL_PACKAGE_BOUTS,
    SELECTION_CHANNELS_BEHAVIORS,
    SELECTION_EXPORT_PROFILE,
    ExportConfigError,
    ExportSelectionError,
)
from results_packaging.media_csv_export.selection import select_bouts_for_export

# Packages currently acceptable as export sources (MVP; archival deferred).
EXPORT_ELIGIBLE_CLASSIFICATIONS = frozenset(
    {ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS}
)


@dataclass(frozen=True, slots=True)
class ExportUiSettings:
    """GUI working settings snapshotted into each CSV/media export request."""

    selection_mode: str = SELECTION_ALL_PACKAGE_BOUTS
    channels_text: str = ""
    behaviors_text: str = ""
    profile_id: str | None = None
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0

    def validated(self) -> ExportUiSettings:
        mode = str(self.selection_mode).strip() or SELECTION_ALL_PACKAGE_BOUTS
        channels = _split_csv_tokens(self.channels_text)
        behaviors = _split_csv_tokens(self.behaviors_text)
        profile = (
            str(self.profile_id).strip() if self.profile_id is not None else None
        ) or None
        if float(self.pre_context_sec) < 0 or float(self.post_context_sec) < 0:
            raise ExportConfigError("Context padding seconds must be >= 0.")
        # Validate via SelectionSpec construction.
        self.to_selection_spec()
        return replace(
            self,
            selection_mode=mode,
            channels_text=",".join(channels),
            behaviors_text=",".join(behaviors),
            profile_id=profile,
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
        )

    def to_selection_spec(self) -> SelectionSpec:
        mode = str(self.selection_mode).strip() or SELECTION_ALL_PACKAGE_BOUTS
        channels = _split_csv_tokens(self.channels_text)
        behaviors = _split_csv_tokens(self.behaviors_text)
        profile = (
            str(self.profile_id).strip() if self.profile_id is not None else None
        ) or None
        if mode == SELECTION_ALL_PACKAGE_BOUTS:
            return SelectionSpec(mode=SELECTION_ALL_PACKAGE_BOUTS).validated()
        if mode == SELECTION_CHANNELS_BEHAVIORS:
            return SelectionSpec(
                mode=SELECTION_CHANNELS_BEHAVIORS,
                channels=channels,
                behavior_names=behaviors,
            ).validated()
        if mode == SELECTION_EXPORT_PROFILE:
            return SelectionSpec(
                mode=SELECTION_EXPORT_PROFILE,
                profile_id=profile,
            ).validated()
        raise ExportConfigError(f"Unsupported export selection_mode {mode!r}.")

    def to_csv_settings(self) -> CsvExportSettings:
        return CsvExportSettings(selection=self.to_selection_spec()).validated()

    def to_media_settings(self) -> MediaExportSettings:
        return MediaExportSettings(
            selection=self.to_selection_spec(),
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
        ).validated()


@dataclass(frozen=True, slots=True)
class ExportRequestIdentity:
    """Identity used to reject stale asynchronous export results."""

    request_id: str
    session_root: Path
    package_run_id: str
    package_dir: Path
    job_kind: str  # csv_export | media_export | media_retry
    settings: ExportUiSettings
    retry_job_dir: Path | None = None

    @staticmethod
    def new(
        *,
        session_root: Path,
        package_run_id: str,
        package_dir: Path,
        job_kind: str,
        settings: ExportUiSettings,
        retry_job_dir: Path | None = None,
    ) -> ExportRequestIdentity:
        return ExportRequestIdentity(
            request_id=str(uuid4()),
            session_root=session_root.resolve(strict=False),
            package_run_id=package_run_id,
            package_dir=package_dir.resolve(strict=False),
            job_kind=job_kind,
            settings=settings.validated(),
            retry_job_dir=(
                retry_job_dir.resolve(strict=False) if retry_job_dir is not None else None
            ),
        )


def package_is_export_eligible(classification: str | None) -> bool:
    return classification in EXPORT_ELIGIBLE_CLASSIFICATIONS


def stale_export_block_reason(classification: str | None) -> str:
    label = classification or "unknown"
    if classification == STALE_SOURCE:
        return (
            f"Selected S9 package is currently stale ({label}). "
            "Normal exports require a currently eligible package; "
            "archival export from stale packages is deferred. "
            "Historical export jobs remain inspectable."
        )
    return (
        f"Selected S9 package is not currently eligible for export ({label})."
    )


def list_package_export_profiles(package_dir: Path | str) -> tuple[str, ...]:
    root = Path(package_dir) / EXPORT_PROFILES_DIRNAME
    if not root.is_dir():
        return ()
    profiles: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / BOUT_INDEX_FILENAME).is_file():
            profiles.append(child.name)
    return tuple(profiles)


def preview_selection_count(
    package_dir: Path | str,
    settings: ExportUiSettings,
) -> int:
    """Count selected bout rows for GUI preview (read-only)."""

    path = Path(package_dir) / BEHAVIOR_BOUTS_FILENAME
    if not path.is_file():
        raise ExportSelectionError(f"Missing behavior_bouts.parquet: {path}")
    bouts = pd.read_parquet(path)
    selected = select_bouts_for_export(
        bouts,
        settings.to_selection_spec(),
        package_dir=Path(package_dir),
    )
    return int(len(selected))


def export_provenance_summary(
    *,
    package_run_id: str,
    settings: ExportUiSettings,
    n_selected: int | None,
) -> str:
    parts = [
        "export_provenance=automated",
        f"source_package={package_run_id}",
        f"selection_mode={settings.selection_mode}",
    ]
    if settings.selection_mode == SELECTION_EXPORT_PROFILE and settings.profile_id:
        parts.append(f"profile_id={settings.profile_id}")
    if settings.selection_mode == SELECTION_CHANNELS_BEHAVIORS:
        if settings.channels_text:
            parts.append(f"channels={settings.channels_text}")
        if settings.behaviors_text:
            parts.append(f"behaviors={settings.behaviors_text}")
    if n_selected is not None:
        parts.append(f"n_selected={n_selected}")
    return "; ".join(parts)


def _split_csv_tokens(text: str) -> tuple[str, ...]:
    if not text or not str(text).strip():
        return ()
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def export_job_table_rows(jobs: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for job in jobs:
        counts = "—"
        if getattr(job, "job_kind", None) == "csv_export":
            counts = f"rows={job.n_rows}" if job.n_rows is not None else "—"
        else:
            parts = []
            if job.n_selected is not None:
                parts.append(f"sel={job.n_selected}")
            if job.n_succeeded is not None:
                parts.append(f"ok={job.n_succeeded}")
            if job.n_failed is not None:
                parts.append(f"fail={job.n_failed}")
            counts = ", ".join(parts) if parts else "—"
        rows.append(
            (
                str(job.export_job_id),
                str(job.source_package_run_id or "—"),
                str(job.job_kind or "—"),
                str(job.status or "—"),
                str(job.classification),
                counts,
                str(job.error_message or "—"),
                str(job.job_dir),
            )
        )
    return tuple(rows)
