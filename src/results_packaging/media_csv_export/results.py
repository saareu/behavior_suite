"""Result dataclasses for media/CSV export jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MediaExportJobResult:
    success: bool
    status: str
    export_job_id: str
    job_dir: Path
    session_root: Path
    source_package_run_id: str
    source_package_dir: Path
    n_selected: int
    n_succeeded: int
    n_failed: int
    n_skipped: int
    n_omitted: int
    clip_index_path: Path | None
    export_job_manifest_path: Path | None
    qc_path: Path | None
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CsvExportJobResult:
    success: bool
    status: str
    export_job_id: str
    job_dir: Path
    session_root: Path
    source_package_run_id: str
    source_package_dir: Path
    n_rows: int
    csv_path: Path | None
    export_job_manifest_path: Path | None
    qc_path: Path | None
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
