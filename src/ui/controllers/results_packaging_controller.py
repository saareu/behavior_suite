"""Headless coordination for Subsystem 09 Results Packaging GUI workflow.

Owns session discovery, S9-ready S8 source selection, packaging settings,
completed-run inspection, and dispatch of ``run_results_packaging_publication``.
Packaging science stays in ``results_packaging`` — never duplicated here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from behavioral_arbitration.run_discovery import summarize_behavioral_arbitration_project
from preprocess.models import OperationProgress
from results_packaging import (
    ELIGIBLE_DOWNSTREAM,
    REVIEW_EMIT_PROFILE_ID,
    ConfigError,
    ExecutionSettings,
    IntakeError,
    ProvenanceError,
    PublicationError,
    ResultsPackageProjectSummary,
    ResultsPackageRunResult,
    ResultsPackageRunSummary,
    ResultsPackagingConfig,
    ResultsPackagingError,
    ResultsPackagingRequest,
    ReviewEmitV1Defaults,
    load_results_packaging_config,
    run_results_packaging_publication,
    summarize_results_package_project,
    validate_s8_source_for_packaging,
)
from results_packaging.contracts import (
    COMPLETED_WITH_WARNINGS,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_FAILED,
)
from results_packaging.media_csv_export import (
    STATUS_INCOMPLETE,
    CsvExportJobResult,
    CsvExportRequest,
    ExportJobSummary,
    ExportProjectSummary,
    MediaExportJobResult,
    MediaExportRequest,
    retry_media_export,
    run_csv_export,
    run_media_export,
    summarize_export_project,
)
from results_packaging.media_csv_export.contracts import (
    JOB_KIND_CSV,
    JOB_KIND_MEDIA,
    ExportConfigError,
    ExportSelectionError,
)
from ui.controllers.results_export_support import (
    ExportRequestIdentity,
    ExportUiSettings,
    export_job_table_rows,
    export_provenance_summary,
    list_package_export_profiles,
    package_is_export_eligible,
    preview_selection_count,
    stale_export_block_reason,
)
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

ProgressCallback = Callable[[OperationProgress], None]
FolderOpener = Callable[[Path], bool]
ResultsPackagingRunner = Callable[..., ResultsPackageRunResult]
CsvExportRunner = Callable[..., CsvExportJobResult]
MediaExportRunner = Callable[..., MediaExportJobResult]
MediaRetryRunner = Callable[..., MediaExportJobResult]

READINESS_NO_SESSION = "no_session"
READINESS_DISCOVERING = "discovering"
READINESS_NO_S8 = "no_s8"
READINESS_S8_NOT_S9_READY = "s8_not_s9_ready"
READINESS_READY = "ready"
READINESS_OTHER = "other"

_ELIGIBLE_S8 = frozenset({ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS})


class ResultsPackagingUiError(ValueError):
    """Expected, actionable Subsystem 09 UI validation error."""


@dataclass(frozen=True, slots=True)
class ResultsPackagingUiSettings:
    """GUI working settings snapshotted into each packaging request."""

    fps: float = 120.0
    default_profile_id: str = "analysis_full"
    enable_review_emit_v1: bool = False
    review_emit_v1: ReviewEmitV1Defaults = field(default_factory=ReviewEmitV1Defaults)
    require_prepared_video: bool = True
    hash_prepared_video: bool = False

    @classmethod
    def from_config(cls, config: ResultsPackagingConfig) -> ResultsPackagingUiSettings:
        validated = config.validated()
        profiles = validated.execution.selected_export_profiles
        return cls(
            fps=validated.fps,
            default_profile_id=validated.default_profile_id,
            enable_review_emit_v1=REVIEW_EMIT_PROFILE_ID in profiles,
            review_emit_v1=validated.review_emit_v1_defaults,
            require_prepared_video=validated.execution.require_prepared_video,
            hash_prepared_video=validated.execution.hash_prepared_video,
        )

    def to_config(self) -> ResultsPackagingConfig:
        profiles = (REVIEW_EMIT_PROFILE_ID,) if self.enable_review_emit_v1 else ()
        return ResultsPackagingConfig(
            fps=float(self.fps),
            default_profile_id=str(self.default_profile_id),
            execution=ExecutionSettings(
                require_prepared_video=bool(self.require_prepared_video),
                hash_prepared_video=bool(self.hash_prepared_video),
                selected_export_profiles=profiles,
                publish_run=True,
            ),
            review_emit_v1_defaults=self.review_emit_v1,
        ).validated()

    def validated(self) -> ResultsPackagingUiSettings:
        return self.from_config(self.to_config())


@dataclass(frozen=True, slots=True)
class PackagingRequestIdentity:
    """Identity used to reject stale asynchronous packaging results."""

    request_id: str
    session_root: Path
    s8_run_id: str
    s8_run_dir: Path
    settings: ResultsPackagingUiSettings


@dataclass(frozen=True, slots=True)
class DiscoveryRequestIdentity:
    """Identity used to reject stale asynchronous discovery results."""

    request_id: str
    generation: int
    session_root: Path


@dataclass(frozen=True, slots=True)
class S8SourceOption:
    """One assessed S8 run offered as an S9 packaging input candidate."""

    run_id: str
    run_dir: Path
    historical_status: str | None
    s8_classification: str
    s9_ready: bool
    label: str
    linked_s7_run_id: str | None = None
    linked_s7_run_dir: Path | None = None
    linked_s6_run_id: str | None = None
    linked_s6_run_dir: Path | None = None
    n_bouts: int | None = None
    n_dense_samples: int | None = None
    prepared_sync_path: str | None = None
    prepared_video_path: str | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Payload collected off the GUI thread for one discovery generation."""

    identity: DiscoveryRequestIdentity
    s8_options: tuple[S8SourceOption, ...]
    s9_summary: ResultsPackageProjectSummary
    export_summary: ExportProjectSummary | None = None


@dataclass(slots=True)
class ResultsPackagingUiState:
    """Mutable S9 workspace state (working settings vs run snapshots)."""

    session_root: Path | None = None
    s8_options: tuple[S8SourceOption, ...] = ()
    selected_s8_run_id: str | None = None
    s8_selection_manual: bool = False
    selected_s9_run_id: str | None = None
    selected_export_job_id: str | None = None
    summary: ResultsPackageProjectSummary | None = None
    export_summary: ExportProjectSummary | None = None
    readiness_kind: str = READINESS_NO_SESSION
    readiness_error: str | None = None
    readiness_warnings: tuple[str, ...] = ()
    working_settings: ResultsPackagingUiSettings = field(
        default_factory=ResultsPackagingUiSettings
    )
    export_settings: ExportUiSettings = field(default_factory=ExportUiSettings)
    last_packaging_settings: ResultsPackagingUiSettings | None = None
    task_running: bool = False
    discovery_running: bool = False
    discovery_generation: int = 0
    active_discovery_identity: DiscoveryRequestIdentity | None = None
    active_packaging_identity: PackagingRequestIdentity | None = None
    active_export_identity: ExportRequestIdentity | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: ResultsPackageRunResult | None = None
    last_csv_export_result: CsvExportJobResult | None = None
    last_media_export_result: MediaExportJobResult | None = None
    unexpected_error_detail: str | None = None
    export_preview_count: int | None = None
    export_preview_text: str = ""


class ResultsPackagingController:
    """Session discovery, S8 source selection, settings, packaging/export dispatch."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], ResultsPackagingConfig] | None = None,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        runner: ResultsPackagingRunner = run_results_packaging_publication,
        csv_export_runner: CsvExportRunner = run_csv_export,
        media_export_runner: MediaExportRunner = run_media_export,
        media_retry_runner: MediaRetryRunner = retry_media_export,
    ) -> None:
        self.state = ResultsPackagingUiState()
        self._config_loader = config_loader or (
            lambda: load_results_packaging_config(None)
        )
        self._folder_opener = folder_opener
        self._runner = runner
        self._csv_export_runner = csv_export_runner
        self._media_export_runner = media_export_runner
        self._media_retry_runner = media_retry_runner
        self._pending_launch_timestamp: str | None = None
        self.reset_working_settings_to_defaults()

    def reset_working_settings_to_defaults(self) -> ResultsPackagingUiSettings:
        try:
            settings = ResultsPackagingUiSettings.from_config(self._config_loader())
        except (ResultsPackagingError, ConfigError) as exc:
            raise ResultsPackagingUiError(str(exc)) from exc
        self.state.working_settings = settings
        self._recompute_readiness()
        return settings

    def set_session(
        self, session_root: Path | str
    ) -> ResultsPackageProjectSummary:
        self.prepare_session(session_root)
        return self.refresh_discovery()

    def prepare_session(self, session_root: Path | str) -> DiscoveryRequestIdentity:
        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise ResultsPackagingUiError(f"Session directory does not exist: {root}")
        previous = self.state.session_root
        if (
            self.state.task_running
            and previous is not None
            and previous.resolve(strict=False) != root
        ):
            raise ResultsPackagingUiError(
                "Cannot change S9 session while results packaging is active."
            )
        self.state.session_root = root
        self.state.readiness_error = None
        self.state.readiness_kind = READINESS_DISCOVERING
        self.state.run_error = None
        self.state.run_stage = "Not started"
        if previous is None or previous.resolve(strict=False) != root:
            self.state.last_result = None
            self.state.last_packaging_settings = None
            self.state.last_csv_export_result = None
            self.state.last_media_export_result = None
            self.state.selected_s9_run_id = None
            self.state.selected_export_job_id = None
            self.state.selected_s8_run_id = None
            self.state.s8_selection_manual = False
            self.state.s8_options = ()
            self.state.summary = None
            self.state.export_summary = None
            self.state.active_export_identity = None
            self.state.export_preview_count = None
            self.state.export_preview_text = ""
        return self.begin_discovery()

    def clear_session(self) -> None:
        if self.state.task_running or self.state.discovery_running:
            raise ResultsPackagingUiError(
                "Cannot clear the S9 session while a task is active."
            )
        working = self.state.working_settings
        self.state = ResultsPackagingUiState(working_settings=working)
        self._recompute_readiness()

    def begin_discovery(self) -> DiscoveryRequestIdentity:
        root = self.state.session_root
        if root is None:
            raise ResultsPackagingUiError("Select a project/session first.")
        self.state.discovery_generation += 1
        identity = DiscoveryRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.discovery_generation,
            session_root=root.resolve(strict=False),
        )
        self.state.active_discovery_identity = identity
        self.state.discovery_running = True
        self.state.readiness_kind = READINESS_DISCOVERING
        self.state.readiness_error = (
            "Discovering S9-ready S8 runs and completed S9 packages…"
        )
        return identity

    def collect_discovery_snapshot(
        self, identity: DiscoveryRequestIdentity
    ) -> DiscoverySnapshot:
        root = identity.session_root
        require_video = self.state.working_settings.require_prepared_video
        try:
            s8_project = summarize_behavioral_arbitration_project(root)
            options = tuple(
                _assess_s8_option(root, run, require_prepared_video=require_video)
                for run in s8_project.runs
            )
            eligible_s8 = any(opt.s9_ready for opt in options)
            s9_summary = summarize_results_package_project(
                root,
                eligible_s8_available=eligible_s8,
                no_s8_reason=None if eligible_s8 else "no S9-ready S8 run",
            )
            export_summary = summarize_export_project(root)
        except (ResultsPackagingError, PublicationError, OSError) as exc:
            raise ResultsPackagingUiError(str(exc)) from exc
        return DiscoverySnapshot(
            identity=identity,
            s8_options=options,
            s9_summary=s9_summary,
            export_summary=export_summary,
        )

    def apply_discovery_snapshot(self, snapshot: DiscoverySnapshot) -> bool:
        if not self._is_current_discovery_result(snapshot.identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.s8_options = snapshot.s8_options
        self.state.summary = snapshot.s9_summary
        self.state.export_summary = snapshot.export_summary
        self._reconcile_s8_selection()
        self._reconcile_s9_selection()
        self._reconcile_export_selection()
        self._recompute_readiness()
        self.refresh_export_preview()
        return True

    def record_discovery_failure(
        self, message: str, *, expected_identity: DiscoveryRequestIdentity | None
    ) -> bool:
        if not self._is_current_discovery_result(expected_identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.readiness_kind = READINESS_OTHER
        self.state.readiness_error = message
        return True

    def refresh_discovery(self) -> ResultsPackageProjectSummary:
        identity = self.begin_discovery()
        snapshot = self.collect_discovery_snapshot(identity)
        applied = self.apply_discovery_snapshot(snapshot)
        if not applied or self.state.summary is None:
            raise ResultsPackagingUiError(
                "Discovery result was superseded by a newer session selection."
            )
        return self.state.summary

    def _is_current_discovery_result(
        self, expected: DiscoveryRequestIdentity | None
    ) -> bool:
        if expected is None:
            return False
        active = self.state.active_discovery_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        if expected.generation != self.state.discovery_generation:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return (
            current_root.resolve(strict=False)
            == expected.session_root.resolve(strict=False)
        )

    def select_s8_source(
        self, run_id: str | None, *, manual: bool = True
    ) -> S8SourceOption | None:
        if self.state.task_running or self.state.discovery_running:
            raise ResultsPackagingUiError(
                "Cannot change S8 source while a Subsystem 9 task is active."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s8_run_id = None
            if manual:
                self.state.s8_selection_manual = True
            self._recompute_readiness()
            return None
        match = next((opt for opt in self.state.s8_options if opt.run_id == run_id), None)
        if match is None:
            raise ResultsPackagingUiError(
                f"S8 run is not available in the current session: {run_id}"
            )
        self.state.selected_s8_run_id = match.run_id
        if manual:
            self.state.s8_selection_manual = True
        self._recompute_readiness()
        return match

    def select_s9_run(
        self, run_id: str | None
    ) -> ResultsPackageRunSummary | None:
        if self.state.task_running:
            raise ResultsPackagingUiError(
                "Cannot change S9 package selection while a Subsystem 9 task is active."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s9_run_id = None
            self.refresh_export_preview()
            return None
        summary = self.state.summary
        if summary is None:
            raise ResultsPackagingUiError(
                "Refresh discovery before selecting an S9 run."
            )
        match = next((run for run in summary.runs if run.run_id == run_id), None)
        if match is None:
            raise ResultsPackagingUiError(f"S9 run was not found: {run_id}")
        self.state.selected_s9_run_id = match.run_id
        self.refresh_export_preview()
        return match

    def selected_s8_source(self) -> S8SourceOption | None:
        selected = self.state.selected_s8_run_id
        if selected is None:
            return None
        return next((opt for opt in self.state.s8_options if opt.run_id == selected), None)

    def selected_s9_run(self) -> ResultsPackageRunSummary | None:
        summary = self.state.summary
        selected = self.state.selected_s9_run_id
        if summary is None or selected is None:
            return None
        return next((run for run in summary.runs if run.run_id == selected), None)

    def update_settings(
        self, settings: ResultsPackagingUiSettings
    ) -> ResultsPackagingUiSettings:
        if self.state.task_running or self.state.discovery_running:
            raise ResultsPackagingUiError(
                "Cannot change S9 settings while a Subsystem 9 task is active."
            )
        validated = self.validate_settings(settings)
        self.state.working_settings = validated
        self._recompute_readiness()
        return validated

    def validate_settings(
        self, settings: ResultsPackagingUiSettings | None = None
    ) -> ResultsPackagingUiSettings:
        candidate = settings if settings is not None else self.state.working_settings
        try:
            return candidate.validated()
        except (ResultsPackagingError, ConfigError) as exc:
            raise ResultsPackagingUiError(str(exc)) from exc

    def can_run_packaging(self) -> bool:
        return self.cannot_run_reason() is None

    def cannot_run_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 9 packaging/export task is already running."
        if self.state.discovery_running:
            return "S9 discovery is still running; wait for readiness to finish."
        if self.state.session_root is None:
            return "Select a project/session first."
        if self.state.readiness_kind != READINESS_READY:
            return self.state.readiness_error or "Subsystem 9 is not ready for packaging."
        source = self.selected_s8_source()
        if source is None:
            return "Select an S9-ready S8 run."
        if not source.s9_ready:
            return "Selected S8 run is not S9-ready: " + (
                "; ".join(source.reasons) if source.reasons else "see readiness panel"
            )
        return None

    def can_run_export(self) -> bool:
        return self.cannot_export_reason() is None

    def cannot_export_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 9 packaging/export task is already running."
        if self.state.discovery_running:
            return "S9 discovery is still running; wait for readiness to finish."
        if self.state.session_root is None:
            return "Select a project/session first."
        run = self.selected_s9_run()
        if run is None:
            return "Select a completed S9 package to export from."
        if not package_is_export_eligible(run.classification):
            return stale_export_block_reason(run.classification)
        try:
            self.state.export_settings.validated()
        except (ExportConfigError, ExportSelectionError) as exc:
            return str(exc)
        return None

    def update_export_settings(self, settings: ExportUiSettings) -> ExportUiSettings:
        if self.state.task_running or self.state.discovery_running:
            raise ResultsPackagingUiError(
                "Cannot change export settings while a Subsystem 9 task is active."
            )
        try:
            validated = settings.validated()
        except (ExportConfigError, ExportSelectionError) as exc:
            raise ResultsPackagingUiError(str(exc)) from exc
        self.state.export_settings = validated
        self.refresh_export_preview()
        return validated

    def refresh_export_preview(self) -> str:
        run = self.selected_s9_run()
        if run is None:
            self.state.export_preview_count = None
            self.state.export_preview_text = "Select an eligible S9 package to preview export."
            return self.state.export_preview_text
        if not package_is_export_eligible(run.classification):
            self.state.export_preview_count = None
            self.state.export_preview_text = stale_export_block_reason(run.classification)
            return self.state.export_preview_text
        try:
            settings = self.state.export_settings.validated()
            count = preview_selection_count(run.run_dir, settings)
        except (ExportConfigError, ExportSelectionError, OSError, ValueError) as exc:
            self.state.export_preview_count = None
            self.state.export_preview_text = f"Export preview unavailable: {exc}"
            return self.state.export_preview_text
        self.state.export_settings = settings
        self.state.export_preview_count = count
        self.state.export_preview_text = export_provenance_summary(
            package_run_id=run.run_id,
            settings=settings,
            n_selected=count,
        )
        return self.state.export_preview_text

    def available_export_profiles(self) -> tuple[str, ...]:
        run = self.selected_s9_run()
        if run is None:
            return ()
        return list_package_export_profiles(run.run_dir)

    def select_export_job(self, job_id: str | None) -> ExportJobSummary | None:
        if job_id is None or not str(job_id).strip():
            self.state.selected_export_job_id = None
            return None
        summary = self.state.export_summary
        if summary is None:
            raise ResultsPackagingUiError("Refresh discovery before selecting an export job.")
        match = next((job for job in summary.jobs if job.export_job_id == job_id), None)
        if match is None:
            raise ResultsPackagingUiError(f"Export job was not found: {job_id}")
        self.state.selected_export_job_id = match.export_job_id
        return match

    def selected_export_job(self) -> ExportJobSummary | None:
        summary = self.state.export_summary
        selected = self.state.selected_export_job_id
        if summary is None or selected is None:
            return None
        return next((job for job in summary.jobs if job.export_job_id == selected), None)

    def export_jobs_table_rows(self) -> tuple[tuple[str, ...], ...]:
        summary = self.state.export_summary
        if summary is None:
            return ()
        return export_job_table_rows(summary.jobs)

    def begin_csv_export_launch(
        self, settings: ExportUiSettings | None = None
    ) -> ExportRequestIdentity:
        reason = self.cannot_export_reason()
        if reason is not None:
            raise ResultsPackagingUiError(reason)
        run = self.selected_s9_run()
        root = self.state.session_root
        if run is None or root is None:
            raise ResultsPackagingUiError("Select an eligible S9 package first.")
        validated = (settings or self.state.export_settings).validated()
        identity = ExportRequestIdentity.new(
            session_root=root,
            package_run_id=run.run_id,
            package_dir=run.run_dir,
            job_kind=JOB_KIND_CSV,
            settings=validated,
        )
        self.state.export_settings = validated
        self.state.active_export_identity = identity
        self.state.active_packaging_identity = None
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Launching CSV export…"
        return identity

    def execute_csv_export_launch(
        self,
        identity: ExportRequestIdentity,
        progress_callback: ProgressCallback | None = None,
    ) -> CsvExportJobResult:
        if not self._is_current_export_result(identity):
            raise ResultsPackagingUiError(
                "CSV export launch was superseded by a newer session/package selection."
            )
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="validate",
                    message="Validating eligible S9 package for CSV export…",
                    completed_units=0,
                    total_units=None,
                    total_is_estimate=True,
                    is_indeterminate=True,
                    processed_video_time_sec=None,
                )
            )
        self.state.run_stage = "Running CSV export…"
        return self._csv_export_runner(
            CsvExportRequest(
                session_root=identity.session_root,
                package_dir=identity.package_dir,
                settings=identity.settings.to_csv_settings(),
                allow_stale_source=False,
            ),
            progress_callback=progress_callback,
        )

    def begin_media_export_launch(
        self, settings: ExportUiSettings | None = None
    ) -> ExportRequestIdentity:
        reason = self.cannot_export_reason()
        if reason is not None:
            raise ResultsPackagingUiError(reason)
        run = self.selected_s9_run()
        root = self.state.session_root
        if run is None or root is None:
            raise ResultsPackagingUiError("Select an eligible S9 package first.")
        validated = (settings or self.state.export_settings).validated()
        identity = ExportRequestIdentity.new(
            session_root=root,
            package_run_id=run.run_id,
            package_dir=run.run_dir,
            job_kind=JOB_KIND_MEDIA,
            settings=validated,
        )
        self.state.export_settings = validated
        self.state.active_export_identity = identity
        self.state.active_packaging_identity = None
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Launching media export…"
        return identity

    def execute_media_export_launch(
        self,
        identity: ExportRequestIdentity,
        progress_callback: ProgressCallback | None = None,
    ) -> MediaExportJobResult:
        if not self._is_current_export_result(identity):
            raise ResultsPackagingUiError(
                "Media export launch was superseded by a newer session/package selection."
            )
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="validate",
                    message="Validating eligible S9 package and prepared video…",
                    completed_units=0,
                    total_units=None,
                    total_is_estimate=True,
                    is_indeterminate=True,
                    processed_video_time_sec=None,
                )
            )
        self.state.run_stage = "Running media export…"
        return self._media_export_runner(
            MediaExportRequest(
                session_root=identity.session_root,
                package_dir=identity.package_dir,
                settings=identity.settings.to_media_settings(),
                allow_stale_source=False,
            ),
            progress_callback=progress_callback,
        )

    def can_retry_selected_media_job(self) -> bool:
        return self.cannot_retry_media_reason() is None

    def cannot_retry_media_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 9 packaging/export task is already running."
        if self.state.discovery_running:
            return "S9 discovery is still running; wait for readiness to finish."
        job = self.selected_export_job()
        if job is None:
            return "Select an incomplete media export job to retry."
        if job.job_kind != JOB_KIND_MEDIA:
            return "Only media export jobs support same-job retry."
        if job.status not in (STATUS_INCOMPLETE, STATUS_FAILED):
            return (
                f"Media job status {job.status!r} is immutable; "
                "completed exports cannot be overwritten."
            )
        if job.source_package_dir is None:
            return "Selected media job is missing source package path."
        # Retry still requires current package eligibility (MVP).
        run = next(
            (
                r
                for r in (self.state.summary.runs if self.state.summary is not None else ())
                if r.run_id == job.source_package_run_id
            ),
            None,
        )
        if run is not None and not package_is_export_eligible(run.classification):
            return stale_export_block_reason(run.classification)
        return None

    def begin_media_retry_launch(self) -> ExportRequestIdentity:
        reason = self.cannot_retry_media_reason()
        if reason is not None:
            raise ResultsPackagingUiError(reason)
        job = self.selected_export_job()
        root = self.state.session_root
        if job is None or root is None or job.source_package_dir is None:
            raise ResultsPackagingUiError("Select an incomplete media export job first.")
        settings = self.state.export_settings.validated()
        identity = ExportRequestIdentity.new(
            session_root=root,
            package_run_id=str(job.source_package_run_id or "unknown"),
            package_dir=job.source_package_dir,
            job_kind="media_retry",
            settings=settings,
            retry_job_dir=job.job_dir,
        )
        self.state.active_export_identity = identity
        self.state.active_packaging_identity = None
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Launching media export retry…"
        return identity

    def execute_media_retry_launch(
        self,
        identity: ExportRequestIdentity,
        progress_callback: ProgressCallback | None = None,
    ) -> MediaExportJobResult:
        if not self._is_current_export_result(identity):
            raise ResultsPackagingUiError(
                "Media retry launch was superseded by a newer session selection."
            )
        if identity.retry_job_dir is None:
            raise ResultsPackagingUiError("Retry identity missing job directory.")
        self.state.run_stage = "Retrying media export…"
        return self._media_retry_runner(
            identity.retry_job_dir, progress_callback=progress_callback
        )

    def record_csv_export_success(
        self,
        result: CsvExportJobResult,
        *,
        expected_identity: ExportRequestIdentity | None = None,
    ) -> bool:
        self.state.task_running = False
        if not self._is_current_export_result(expected_identity):
            return False
        self.state.last_csv_export_result = result
        self.state.run_stage = f"CSV export {result.status}"
        self.state.run_error = None
        self.state.active_export_identity = None
        self.state.selected_export_job_id = result.export_job_id
        return True

    def record_media_export_success(
        self,
        result: MediaExportJobResult,
        *,
        expected_identity: ExportRequestIdentity | None = None,
    ) -> bool:
        self.state.task_running = False
        if not self._is_current_export_result(expected_identity):
            return False
        self.state.last_media_export_result = result
        self.state.run_stage = f"Media export {result.status}"
        self.state.run_error = None
        self.state.active_export_identity = None
        self.state.selected_export_job_id = result.export_job_id
        return True

    def record_export_failure(
        self,
        message: str,
        *,
        expected_identity: ExportRequestIdentity | None = None,
    ) -> bool:
        if expected_identity is not None:
            active = self.state.active_export_identity
            if active is not None and active.request_id != expected_identity.request_id:
                return False
        self.state.task_running = False
        self.state.last_csv_export_result = None
        self.state.last_media_export_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message
        self.state.active_export_identity = None
        return True

    def open_selected_export_folder(self) -> bool:
        job = self.selected_export_job()
        if job is None and self.state.last_csv_export_result is not None:
            return bool(self._folder_opener(self.state.last_csv_export_result.job_dir))
        if job is None and self.state.last_media_export_result is not None:
            return bool(self._folder_opener(self.state.last_media_export_result.job_dir))
        if job is None:
            raise ResultsPackagingUiError("Select an export job first.")
        return bool(self._folder_opener(job.job_dir))

    def _is_current_export_result(
        self, expected: ExportRequestIdentity | None
    ) -> bool:
        if expected is None:
            return False
        active = self.state.active_export_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return (
            current_root.resolve(strict=False)
            == expected.session_root.resolve(strict=False)
        )

    def _reconcile_export_selection(self) -> None:
        summary = self.state.export_summary
        if summary is None or not summary.jobs:
            self.state.selected_export_job_id = None
            return
        selected = self.state.selected_export_job_id
        if selected is not None and any(
            job.export_job_id == selected for job in summary.jobs
        ):
            return
        self.state.selected_export_job_id = summary.jobs[0].export_job_id

    def source_status_text(self) -> str:
        kind = self.state.readiness_kind
        if kind == READINESS_NO_SESSION or self.state.session_root is None:
            return "Select a project/session with an S9-ready S8 run."
        if kind == READINESS_DISCOVERING or self.state.discovery_running:
            return self.state.readiness_error or "Discovering S8/S9 artifacts…"
        if not self.state.s8_options:
            return (
                "No S8 runs were found for this session. "
                "Complete Subsystem 8 arbitration, then Refresh."
            )
        source = self.selected_s8_source()
        if source is None:
            return self.state.readiness_error or "Select an S8 run."
        manual = " (manual)" if self.state.s8_selection_manual else " (default S9-ready)"
        ready = (
            "Ready for packaging."
            if kind == READINESS_READY and source.s9_ready
            else f"Blocked [{kind}]: {self.state.readiness_error}"
        )
        bouts = (
            f"{source.n_bouts} bouts"
            if source.n_bouts is not None
            else "bout count unknown"
        )
        dense = (
            f"{source.n_dense_samples} dense samples"
            if source.n_dense_samples is not None
            else "dense count unknown"
        )
        return (
            f"S8={source.run_id}: historical={source.historical_status or '—'}, "
            f"S8 eligibility={source.s8_classification}, "
            f"S9-ready={source.s9_ready}{manual}; linked S7={source.linked_s7_run_id or '—'}; "
            f"linked S6={source.linked_s6_run_id or '—'}; {bouts}; {dense}. {ready}"
        )

    def begin_packaging_launch(
        self,
        settings: ResultsPackagingUiSettings | None = None,
        *,
        timestamp: str | None = None,
    ) -> PackagingRequestIdentity:
        """Accept a run on the GUI thread without full S8 revalidation.

        Marks the task active immediately so duplicate clicks and source/settings
        mutations are blocked. Heavy ``validate_s8_source_for_packaging`` runs
        later in :meth:`execute_packaging_launch` on the worker thread.
        """

        reason = self.cannot_run_reason()
        if reason is not None:
            raise ResultsPackagingUiError(reason)
        source = self.selected_s8_source()
        root = self.state.session_root
        if source is None or root is None:
            raise ResultsPackagingUiError("Select an S9-ready S8 run first.")
        if not source.s9_ready:
            raise ResultsPackagingUiError(
                "Selected S8 run is not S9-ready: "
                + ("; ".join(source.reasons) if source.reasons else "see readiness panel")
            )
        validated = self.validate_settings(settings)
        identity = PackagingRequestIdentity(
            request_id=str(uuid4()),
            session_root=root.resolve(strict=False),
            s8_run_id=source.run_id,
            s8_run_dir=source.run_dir.resolve(strict=False),
            settings=validated,
        )
        self.state.active_packaging_identity = identity
        self.state.active_export_identity = None
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Launching packaging…"
        self.state.readiness_error = None
        self._pending_launch_timestamp = timestamp
        return identity

    def execute_packaging_launch(
        self,
        identity: PackagingRequestIdentity,
        progress_callback: ProgressCallback | None = None,
    ) -> ResultsPackageRunResult:
        """Worker-thread entry: revalidate S8 source, then publish package."""

        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="validate",
                    message="Revalidating S8 source for packaging…",
                    completed_units=0,
                    total_units=None,
                    total_is_estimate=True,
                    is_indeterminate=True,
                    processed_video_time_sec=None,
                )
            )
        request = self._build_request_after_revalidate(identity)
        self.state.run_stage = "Running results packaging…"
        return self._runner(request, progress_callback=progress_callback)

    def _build_request_after_revalidate(
        self, identity: PackagingRequestIdentity
    ) -> ResultsPackagingRequest:
        if not self._is_current_packaging_result(identity):
            raise ResultsPackagingUiError(
                "Packaging launch was superseded by a newer session/source selection."
            )
        root = identity.session_root
        settings = identity.settings
        try:
            validation = validate_s8_source_for_packaging(
                root,
                s8_run_dir=identity.s8_run_dir,
                require_prepared_video=settings.require_prepared_video,
            )
        except (IntakeError, ProvenanceError) as exc:
            raise ResultsPackagingUiError(str(exc)) from exc
        if not validation.s8_eligible:
            detail = "; ".join(validation.warnings) or "S8 source is not currently eligible"
            self.state.readiness_kind = READINESS_S8_NOT_S9_READY
            self.state.readiness_error = detail
            raise ResultsPackagingUiError(
                f"Selected S8 run is no longer S9-ready: {detail}"
            )
        if validation.s8_run_id != identity.s8_run_id:
            raise ResultsPackagingUiError(
                "Selected S8 run identity changed during launch revalidation."
            )
        timestamp = self._pending_launch_timestamp
        self._pending_launch_timestamp = None
        return ResultsPackagingRequest(
            session_root=root,
            s8_run_dir=identity.s8_run_dir,
            config=settings.to_config(),
            run_purpose="gui",
            timestamp=timestamp,
        )

    def record_success(
        self,
        result: ResultsPackageRunResult,
        *,
        expected_identity: PackagingRequestIdentity | None = None,
        refresh_discovery: bool = False,
    ) -> bool:
        self.state.task_running = False
        if not self._is_current_packaging_result(expected_identity):
            return False
        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.last_packaging_settings = (
            expected_identity.settings
            if expected_identity is not None
            else self.state.working_settings
        )
        self.state.selected_s9_run_id = result.run_id
        self.state.active_packaging_identity = None
        if refresh_discovery:
            try:
                self.refresh_discovery()
                self.state.selected_s9_run_id = result.run_id
            except ResultsPackagingUiError:
                pass
        return True

    def record_failure(
        self,
        message: str,
        *,
        expected_identity: PackagingRequestIdentity | None = None,
    ) -> bool:
        self.state.task_running = False
        if (
            expected_identity is not None
            and self.state.active_packaging_identity is not None
            and expected_identity.request_id
            != self.state.active_packaging_identity.request_id
        ):
            return False
        self.state.last_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message
        self.state.active_packaging_identity = None
        return True

    def open_selected_s9_folder(self) -> bool:
        run = self.selected_s9_run()
        if run is None and self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.run_dir))
        if run is None:
            raise ResultsPackagingUiError("Select a completed S9 package first.")
        return bool(self._folder_opener(run.run_dir))

    def settings_display_rows(self) -> tuple[tuple[str, str], ...]:
        settings = self.state.working_settings
        review = settings.review_emit_v1
        floor = max(
            int(review.min_emit_bout_frames),
            int(math.ceil(review.min_emit_bout_seconds * review.target_fps)),
        )
        rows: list[tuple[str, str]] = [
            ("default_profile_id", settings.default_profile_id),
            (
                "Full package",
                "Always includes every authoritative S8 bout (analysis_full)",
            ),
            (
                "Optional review_emit_v1",
                "enabled" if settings.enable_review_emit_v1 else "disabled",
            ),
            ("review include_contact", str(review.include_contact)),
            ("review include_engaged", str(review.include_engaged)),
            ("review include_side_by_side", str(review.include_side_by_side)),
            (
                "review emission floor",
                (
                    f"max({review.min_emit_bout_frames} frames, "
                    f"ceil({review.min_emit_bout_seconds:g}s × {review.target_fps:g} Hz))"
                    f" = {floor} dense frames"
                ),
            ),
            ("require_prepared_video", str(settings.require_prepared_video)),
            (
                "hash_prepared_video",
                (
                    f"{settings.hash_prepared_video} "
                    "(optional; expensive; worker-only)"
                ),
            ),
            (
                "package_mode",
                "reference_lightweight (not a self-contained video archive)",
            ),
        ]
        return tuple(rows)

    def package_summary_rows(self) -> tuple[tuple[str, str], ...]:
        result = self.state.last_result
        run = self.selected_s9_run()
        use_last = result is not None and (run is None or run.run_id == result.run_id)
        meta = self._selected_package_meta()
        qc = self._selected_qc()
        manifest = self._selected_manifest()

        run_id = None
        status = None
        classification = None
        source_s8 = None
        run_dir = None
        if use_last and result is not None:
            run_id = result.run_id
            status = result.status
            classification = result.classification
            source_s8 = result.source_s8_run_id
            run_dir = str(result.run_dir)
            if run is not None:
                classification = run.classification or classification
        elif run is not None:
            run_id = run.run_id
            status = run.status
            classification = run.classification
            source_s8 = run.source_s8_run_id
            run_dir = str(run.run_dir)

        if run_id is None:
            return ()

        metrics = qc.get("metrics") if isinstance(qc, Mapping) else None
        if not isinstance(metrics, Mapping):
            metrics = {}
        channel_counts = metrics.get("channel_counts")
        if not isinstance(channel_counts, Mapping):
            channel_counts = {}
        sync_audit = metrics.get("sync_audit")
        if not isinstance(sync_audit, Mapping):
            sync_audit = {}
        if not sync_audit and isinstance(manifest, Mapping):
            timeline = manifest.get("timeline")
            if isinstance(timeline, Mapping):
                sync_audit = {"n_dense": timeline.get("n_sync_rows")}

        sync_rows = sync_audit.get("n_dense") or metrics.get("n_dense_samples")
        if sync_rows is None and isinstance(manifest, Mapping):
            row_counts = manifest.get("row_counts")
            if isinstance(row_counts, Mapping):
                sync_rows = row_counts.get("sync_index")

        profiles = metrics.get("export_profiles")
        if not isinstance(profiles, Mapping):
            profiles = {}
        review = profiles.get(REVIEW_EMIT_PROFILE_ID)
        if not isinstance(review, Mapping):
            review = {}

        rows: list[tuple[str, str]] = [
            ("S9 run ID", run_id),
            ("Historical completion status", status or "—"),
            (
                "Current downstream eligibility",
                _eligibility_label(classification),
            ),
            ("Exact S8 source run", source_s8 or "—"),
            ("Source sync-row count (dense)", _format_metric(sync_rows)),
            (
                "Total authoritative bouts (full package)",
                _format_metric(metrics.get("n_bouts_packaged")),
            ),
            ("Exclusive bouts", _format_metric(channel_counts.get("exclusive"))),
            ("Contact bouts", _format_metric(channel_counts.get("contact"))),
            ("Side-by-side bouts", _format_metric(channel_counts.get("side_by_side"))),
        ]
        if review:
            rows.append(
                (
                    "Optional profile",
                    (
                        f"{REVIEW_EMIT_PROFILE_ID} "
                        "(additional filtered view; full package unchanged)"
                    ),
                )
            )
            rows.append(
                (
                    "Profile included / excluded",
                    (
                        f"included={_format_metric(review.get('n_included'))}, "
                        f"excluded={_format_metric(review.get('n_excluded'))}"
                    ),
                )
            )
        else:
            rows.append(("Optional profile", "none (full package only)"))

        if isinstance(manifest, Mapping):
            refs = manifest.get("source_references")
            if isinstance(refs, Mapping):
                video = refs.get("prepared_video") or refs.get("source_video")
                sync = refs.get("prepared_sync") or refs.get("source_sync")
                if isinstance(video, Mapping):
                    rows.append(
                        (
                            "Prepared-video reference",
                            (
                                f"path={video.get('path') or '—'}; "
                                f"present={video.get('exists', video.get('path') is not None)}; "
                                f"fingerprint={video.get('fingerprint_type') or '—'}"
                            ),
                        )
                    )
                if isinstance(sync, Mapping):
                    rows.append(
                        (
                            "Synchronization reference",
                            (
                                f"path={sync.get('path') or '—'}; "
                                f"fingerprint={sync.get('fingerprint_type') or '—'}"
                            ),
                        )
                    )
            rows.append(
                (
                    "Package mode",
                    str(manifest.get("package_mode") or "reference_lightweight"),
                )
            )
        rows.append(("Package directory", run_dir or "—"))
        if isinstance(meta, Mapping):
            outputs = meta.get("outputs")
            if isinstance(outputs, Mapping):
                for key in (
                    "behavior_bouts",
                    "sync_index",
                    "ethogram_summary",
                    "package_manifest",
                    "qc",
                ):
                    if outputs.get(key):
                        rows.append((f"Artifact: {key}", str(outputs[key])))
        return tuple(rows)

    def qc_display_rows(self) -> tuple[tuple[str, str], ...]:
        result = self.state.last_result
        run = self.selected_s9_run()
        use_last = result is not None and (run is None or run.run_id == result.run_id)
        qc = self._selected_qc()
        classification = None
        status = None
        source_s8 = None
        if use_last and result is not None:
            status = result.status
            classification = result.classification
            source_s8 = result.source_s8_run_id
            if run is not None:
                classification = run.classification or classification
        elif run is not None:
            status = run.status
            classification = run.classification
            source_s8 = run.source_s8_run_id
        else:
            return ()

        rows: list[tuple[str, str]] = [
            ("Historical status", status or "—"),
            ("Current eligibility", _eligibility_label(classification)),
            ("Source S8", source_s8 or "—"),
        ]
        if use_last and result is not None and result.warnings:
            rows.append(("Run warnings", "; ".join(result.warnings)))
        if not isinstance(qc, Mapping):
            return tuple(rows)

        structural = qc.get("structural_ok")
        if structural is not None:
            rows.append(("Structural QC", "OK" if structural else "FAILED"))

        failures = qc.get("failures")
        if isinstance(failures, list) and failures:
            rows.append(("— Structural errors —", ""))
            for item in failures:
                rows.append(("Structural error", str(item)))

        warnings = qc.get("warnings")
        if isinstance(warnings, list) and warnings:
            rows.append(("— Warnings (scientifically meaningful) —", ""))
            for item in warnings:
                rows.append(("Warning", str(item)))

        informational = qc.get("informational")
        if isinstance(informational, list) and informational:
            rows.append(("— Information (expected configuration / residuals) —", ""))
            for item in informational:
                rows.append(("Information", str(item)))

        if run is not None and run.staleness_reasons:
            rows.append(("— Dependency / staleness —", ""))
            for reason in run.staleness_reasons:
                rows.append(("Dependency note", reason))
        if run is not None and run.metadata_errors:
            rows.append(("— Metadata issues —", ""))
            for err in run.metadata_errors:
                rows.append(("Metadata", err))
        return tuple(rows)

    def _selected_qc(self) -> Mapping[str, Any] | None:
        result = self.state.last_result
        run = self.selected_s9_run()
        use_last = result is not None and (run is None or run.run_id == result.run_id)
        if use_last and result is not None and isinstance(result.qc, Mapping):
            return result.qc
        if run is not None:
            return _load_json_mapping(run.run_dir / QC_FILENAME)
        return None

    def _selected_manifest(self) -> Mapping[str, Any] | None:
        result = self.state.last_result
        run = self.selected_s9_run()
        use_last = result is not None and (run is None or run.run_id == result.run_id)
        path: Path | None = None
        if use_last and result is not None and result.package_manifest_path is not None:
            path = result.package_manifest_path
        elif run is not None and run.package_manifest_path is not None:
            path = run.package_manifest_path
        elif run is not None:
            path = run.run_dir / PACKAGE_MANIFEST_FILENAME
        return _load_json_mapping(path) if path is not None else None

    def _selected_package_meta(self) -> Mapping[str, Any] | None:
        result = self.state.last_result
        run = self.selected_s9_run()
        use_last = result is not None and (run is None or run.run_id == result.run_id)
        if use_last and result is not None:
            return result.to_metadata()
        if run is not None:
            return _load_json_mapping(run.run_dir / RUN_META_FILENAME)
        return None

    def _reconcile_s8_selection(self) -> None:
        options = self.state.s8_options
        selected = self.state.selected_s8_run_id
        ids = {opt.run_id for opt in options}
        if self.state.s8_selection_manual:
            if selected is not None and selected in ids:
                return
            self.state.selected_s8_run_id = None
            return
        ready = next((opt for opt in options if opt.s9_ready), None)
        self.state.selected_s8_run_id = ready.run_id if ready is not None else None

    def _reconcile_s9_selection(self) -> None:
        summary = self.state.summary
        selected = self.state.selected_s9_run_id
        if summary is None:
            return
        ids = {run.run_id for run in summary.runs}
        if selected is not None and selected in ids:
            return
        if self.state.last_result is not None and self.state.last_result.run_id in ids:
            self.state.selected_s9_run_id = self.state.last_result.run_id
            return
        eligible = summary.eligible_runs
        if eligible:
            self.state.selected_s9_run_id = eligible[0].run_id

    def _recompute_readiness(self) -> None:
        if self.state.session_root is None:
            self.state.readiness_kind = READINESS_NO_SESSION
            self.state.readiness_error = "Select a project/session first."
            self.state.readiness_warnings = ()
            return
        if self.state.discovery_running:
            self.state.readiness_kind = READINESS_DISCOVERING
            return
        if not self.state.s8_options:
            self.state.readiness_kind = READINESS_NO_S8
            self.state.readiness_error = (
                "No S8 runs discovered. Complete Subsystem 8 arbitration first."
            )
            self.state.readiness_warnings = ()
            return
        source = self.selected_s8_source()
        if source is None:
            ready_any = any(opt.s9_ready for opt in self.state.s8_options)
            self.state.readiness_kind = (
                READINESS_S8_NOT_S9_READY if not ready_any else READINESS_OTHER
            )
            self.state.readiness_error = (
                "No S9-ready S8 run is available."
                if not ready_any
                else "Select an S8 run (manual selection cleared or not S9-ready)."
            )
            self.state.readiness_warnings = ()
            return
        warnings = list(source.warnings)
        if not source.s9_ready:
            self.state.readiness_kind = READINESS_S8_NOT_S9_READY
            self.state.readiness_error = "; ".join(source.reasons) or (
                "Selected S8 run fails S9 packaging readiness."
            )
            self.state.readiness_warnings = tuple(warnings)
            return
        self.state.readiness_kind = READINESS_READY
        self.state.readiness_error = None
        self.state.readiness_warnings = tuple(warnings)

    def _is_current_packaging_result(
        self, expected: PackagingRequestIdentity | None
    ) -> bool:
        if expected is None:
            return self.state.active_packaging_identity is None
        active = self.state.active_packaging_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return (
            current_root.resolve(strict=False)
            == expected.session_root.resolve(strict=False)
        )


def _assess_s8_option(
    session_root: Path,
    run: Any,
    *,
    require_prepared_video: bool,
) -> S8SourceOption:
    """Build one S8 source option; full validation for currently eligible runs."""

    prelim_ready = run.classification in _ELIGIBLE_S8
    tag = "candidate" if prelim_ready else "not eligible"
    base = S8SourceOption(
        run_id=run.run_id,
        run_dir=run.run_dir,
        historical_status=run.status,
        s8_classification=run.classification,
        s9_ready=False,
        label=f"{run.run_id} [{tag}]",
        linked_s7_run_id=run.source_s7_run_id,
        linked_s7_run_dir=run.source_s7_run_dir,
        linked_s6_run_id=run.source_s6_run_id,
        linked_s6_run_dir=run.source_s6_run_dir,
        reasons=tuple(run.staleness_reasons) if run.staleness_reasons else (),
    )
    if not prelim_ready:
        reasons = list(base.reasons) or [f"S8 classification={run.classification}"]
        return replace(
            base,
            reasons=tuple(reasons),
            label=f"{run.run_id} [not S9-ready]",
        )
    try:
        validation = validate_s8_source_for_packaging(
            session_root,
            s8_run_dir=run.run_dir,
            require_prepared_video=require_prepared_video,
        )
    except (IntakeError, ProvenanceError, OSError) as exc:
        return replace(
            base,
            s9_ready=False,
            reasons=(str(exc),),
            label=f"{run.run_id} [not S9-ready]",
        )
    sync = validation.sync_video
    return S8SourceOption(
        run_id=validation.s8_run_id,
        run_dir=validation.s8_run_dir,
        historical_status=validation.s8_status,
        s8_classification=validation.s8_classification,
        s9_ready=bool(validation.s8_eligible),
        label=(
            f"{validation.s8_run_id} "
            f"[{'S9-ready' if validation.s8_eligible else 'not S9-ready'}]"
        ),
        linked_s7_run_id=validation.source_s7_run_id,
        linked_s7_run_dir=validation.source_s7_run_dir,
        linked_s6_run_id=validation.source_s6_run_id,
        linked_s6_run_dir=validation.source_s6_run_dir,
        n_bouts=validation.n_bouts,
        n_dense_samples=validation.n_dense_samples,
        prepared_sync_path=str(sync.prepared_sync_path),
        prepared_video_path=(
            str(sync.prepared_video_path) if sync.prepared_video_path is not None else None
        ),
        reasons=(),
        warnings=tuple(validation.warnings),
    )


def _eligibility_label(classification: str | None) -> str:
    mapping = {
        "eligible": "eligible (current)",
        "completed_with_warnings": "completed with warnings (current)",
        "stale_source": "stale upstream source",
        "failed_or_incomplete": "failed / incomplete",
        "missing_required_artifacts": "missing / invalid artifacts",
    }
    if classification is None:
        return "—"
    return mapping.get(classification, classification)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        parts = [f"{k}={_format_metric(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    return str(value)
