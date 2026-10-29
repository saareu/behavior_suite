"""In-memory and published results-packaging result contracts (Sprint 1–3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import (
    SPRINT1_RESULT_SCHEMA_VERSION,
    SPRINT2_RESULT_SCHEMA_VERSION,
    SPRINT3_RESULT_SCHEMA_VERSION,
)
from results_packaging.dependencies import DependencyManifest
from results_packaging.eligibility import S9SourceValidationResult
from results_packaging.export_profiles import ExportProfileResult
from results_packaging.sync_index import SyncIndexBuildResult


@dataclass(frozen=True, slots=True)
class Sprint1PackagingResult:
    """Headless Sprint 1 output: validated source, sync index, bout mappings.

    Does **not** publish an immutable S9 run directory.
    """

    source: S9SourceValidationResult
    sync: SyncIndexBuildResult
    bout_boundary_map: pd.DataFrame
    dependencies: DependencyManifest
    settings_used: dict[str, Any]
    session_root: Path
    warnings: tuple[str, ...] = ()
    schema_version: str = SPRINT1_RESULT_SCHEMA_VERSION
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_root": str(self.session_root),
            "s8_run_id": self.source.s8_run_id,
            "s8_run_dir": str(self.source.s8_run_dir),
            "s8_status": self.source.s8_status,
            "s8_classification": self.source.s8_classification,
            "source_s7_run_id": self.source.source_s7_run_id,
            "source_s6_run_id": self.source.source_s6_run_id,
            "n_dense_samples": self.source.n_dense_samples,
            "n_directional_rows": self.source.n_directional_rows,
            "n_bouts": self.source.n_bouts,
            "sync_audit": self.sync.audit_dict(),
            "dependencies": self.dependencies.to_dict(),
            "warnings": list(self.warnings),
            "settings_used": dict(self.settings_used),
            "diagnostics": dict(self.diagnostics),
            "published": False,
        }


@dataclass(frozen=True, slots=True)
class Sprint2PackagingResult:
    """Headless Sprint 2 in-memory package (no immutable publication).

    Contains the full validated sync index, lossless annotated bout table,
    ethogram summary, optional export-profile view, dependency identities,
    processing settings, and structural diagnostics.
    """

    source: S9SourceValidationResult
    sync: SyncIndexBuildResult
    bout_boundary_map: pd.DataFrame
    behavior_bouts: pd.DataFrame
    ethogram_summary: pd.DataFrame
    dependencies: DependencyManifest
    settings_used: dict[str, Any]
    session_root: Path
    export_profiles: dict[str, ExportProfileResult] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = SPRINT2_RESULT_SCHEMA_VERSION
    diagnostics: dict[str, Any] = field(default_factory=dict)
    package_run_id: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        profile_meta = {
            profile_id: result.to_manifest_dict()
            for profile_id, result in self.export_profiles.items()
        }
        return {
            "schema_version": self.schema_version,
            "session_root": str(self.session_root),
            "package_run_id": self.package_run_id,
            "s8_run_id": self.source.s8_run_id,
            "s8_run_dir": str(self.source.s8_run_dir),
            "s8_status": self.source.s8_status,
            "s8_classification": self.source.s8_classification,
            "source_s7_run_id": self.source.source_s7_run_id,
            "source_s6_run_id": self.source.source_s6_run_id,
            "n_dense_samples": self.source.n_dense_samples,
            "n_directional_rows": self.source.n_directional_rows,
            "n_bouts_source": self.source.n_bouts,
            "n_bouts_packaged": int(len(self.behavior_bouts)),
            "n_ethogram_summary_rows": int(len(self.ethogram_summary)),
            "sync_audit": self.sync.audit_dict(),
            "dependencies": self.dependencies.to_dict(),
            "export_profiles": profile_meta,
            "warnings": list(self.warnings),
            "settings_used": dict(self.settings_used),
            "diagnostics": dict(self.diagnostics),
            "published": False,
            "default_profile_id": "analysis_full",
        }


@dataclass(frozen=True, slots=True)
class ResultsPackageRunResult:
    """Immutable S9 publication result (Sprint 3)."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    session_root: Path
    source_s8_run_id: str
    source_s8_run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    qc_path: Path | None
    package_manifest_path: Path | None
    sync_index_path: Path | None
    behavior_bouts_path: Path | None
    ethogram_summary_path: Path | None
    export_profile_dirs: dict[str, Path] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    qc: dict[str, Any] | None = None
    currently_eligible: bool = False
    classification: str | None = None
    schema_version: str = SPRINT3_RESULT_SCHEMA_VERSION
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "success": self.success,
            "status": self.status,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "session_root": str(self.session_root),
            "source_s8_run_id": self.source_s8_run_id,
            "source_s8_run_dir": str(self.source_s8_run_dir),
            "published": self.success,
            "currently_eligible": self.currently_eligible,
            "classification": self.classification,
            "warnings": list(self.warnings),
            "error_message": self.error_message,
            "qc": dict(self.qc) if self.qc is not None else None,
            "outputs": {
                "run_meta": str(self.run_meta_path),
                "settings_used": str(self.settings_used_path),
                "processing_log": str(self.processing_log_path),
                "qc": str(self.qc_path) if self.qc_path is not None else None,
                "package_manifest": (
                    str(self.package_manifest_path)
                    if self.package_manifest_path is not None
                    else None
                ),
                "sync_index": (
                    str(self.sync_index_path) if self.sync_index_path is not None else None
                ),
                "behavior_bouts": (
                    str(self.behavior_bouts_path)
                    if self.behavior_bouts_path is not None
                    else None
                ),
                "ethogram_summary": (
                    str(self.ethogram_summary_path)
                    if self.ethogram_summary_path is not None
                    else None
                ),
                "export_profiles": {
                    k: str(v) for k, v in self.export_profile_dirs.items()
                },
            },
            "diagnostics": dict(self.diagnostics),
        }
