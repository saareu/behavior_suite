"""Headless coordination for Subsystem 08 Behavioral Arbitration GUI workflow.

Owns session discovery, S8-ready S7 source selection, scientific settings,
completed-run inspection, and dispatch of ``run_behavioral_arbitration_publication``.
Arbitration science stays in ``behavioral_arbitration`` — never duplicated here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from behavioral_arbitration import (
    LABEL_ENGAGED,
    LABEL_UNDEFINED,
    BehavioralArbitrationConfig,
    BehavioralArbitrationError,
    BehavioralArbitrationProjectSummary,
    BehavioralArbitrationRequest,
    BehavioralArbitrationRunResult,
    BehavioralArbitrationRunSummary,
    ConfigError,
    ExecutionSettings,
    IntakeError,
    PublicationError,
    S7S8ReadinessAssessment,
    ScientificSettings,
    assess_s7_for_s8,
    list_s7_options_for_s8,
    load_behavioral_arbitration_config,
    run_behavioral_arbitration_publication,
    summarize_behavioral_arbitration_project,
)
from behavioral_arbitration.config import PosturalSettings
from behavioral_arbitration.contracts import QC_FILENAME
from preprocess.models import OperationProgress
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

ProgressCallback = Callable[[OperationProgress], None]
FolderOpener = Callable[[Path], bool]
BehavioralArbitrationRunner = Callable[..., BehavioralArbitrationRunResult]

READINESS_NO_SESSION = "no_session"
READINESS_DISCOVERING = "discovering"
READINESS_NO_S7 = "no_s7"
READINESS_S7_NOT_S8_READY = "s7_not_s8_ready"
READINESS_READY = "ready"
READINESS_OTHER = "other"


class BehavioralArbitrationUiError(ValueError):
    """Expected, actionable Subsystem 08 UI validation error."""


@dataclass(frozen=True, slots=True)
class BehavioralArbitrationUiSettings:
    """GUI working settings snapshotted into each arbitration request."""

    fps: float = 120.0
    require_exactly_two_mice: bool = True
    detector_coverage_profile: str = "full_ethogram"
    working_thresholds: Mapping[str, Any] = field(default_factory=dict)
    scientific: ScientificSettings = field(default_factory=ScientificSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    @classmethod
    def from_config(
        cls, config: BehavioralArbitrationConfig
    ) -> BehavioralArbitrationUiSettings:
        validated = config.validated()
        return cls(
            fps=validated.fps,
            require_exactly_two_mice=validated.require_exactly_two_mice,
            detector_coverage_profile=validated.detector_coverage_profile,
            working_thresholds=dict(validated.working_thresholds),
            scientific=validated.scientific,
            execution=validated.execution,
        )

    def to_config(self) -> BehavioralArbitrationConfig:
        return BehavioralArbitrationConfig(
            fps=float(self.fps),
            require_exactly_two_mice=bool(self.require_exactly_two_mice),
            detector_coverage_profile=str(self.detector_coverage_profile),
            working_thresholds=dict(self.working_thresholds),
            scientific=self.scientific,
            execution=self.execution,
        ).validated()

    def validated(self) -> BehavioralArbitrationUiSettings:
        return self.from_config(self.to_config())


@dataclass(frozen=True, slots=True)
class ArbitrationRequestIdentity:
    """Identity used to reject stale asynchronous arbitration results."""

    request_id: str
    session_root: Path
    s7_run_id: str
    s7_run_dir: Path
    settings: BehavioralArbitrationUiSettings


@dataclass(frozen=True, slots=True)
class DiscoveryRequestIdentity:
    """Identity used to reject stale asynchronous discovery results."""

    request_id: str
    generation: int
    session_root: Path


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Payload collected off the GUI thread for one discovery generation."""

    identity: DiscoveryRequestIdentity
    s7_options: tuple[S7SourceOption, ...]
    s8_summary: BehavioralArbitrationProjectSummary


@dataclass(frozen=True, slots=True)
class S7SourceOption:
    """One assessed S7 run offered as an S8 input candidate."""

    run_id: str
    run_dir: Path
    historical_status: str | None
    s7_classification: str
    s8_ready: bool
    label: str
    linked_s6_run_id: str | None = None
    linked_s6_run_dir: Path | None = None
    mouse_ids: tuple[int, ...] = ()
    n_frames: int | None = None
    detectors_dyadic: tuple[str, ...] = ()
    detectors_focal: tuple[str, ...] = ()
    missing_detectors: tuple[str, ...] = ()
    missing_s6_columns: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    lineage_aligned: bool | None = None


@dataclass(slots=True)
class BehavioralArbitrationUiState:
    """Mutable S8 workspace state (working settings vs run snapshots)."""

    session_root: Path | None = None
    s7_options: tuple[S7SourceOption, ...] = ()
    selected_s7_run_id: str | None = None
    s7_selection_manual: bool = False
    selected_s8_run_id: str | None = None
    summary: BehavioralArbitrationProjectSummary | None = None
    readiness_kind: str = READINESS_NO_SESSION
    readiness_error: str | None = None
    readiness_warnings: tuple[str, ...] = ()
    working_settings: BehavioralArbitrationUiSettings = field(
        default_factory=BehavioralArbitrationUiSettings
    )
    last_arbitration_settings: BehavioralArbitrationUiSettings | None = None
    task_running: bool = False
    discovery_running: bool = False
    discovery_generation: int = 0
    active_discovery_identity: DiscoveryRequestIdentity | None = None
    active_arbitration_identity: ArbitrationRequestIdentity | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: BehavioralArbitrationRunResult | None = None
    unexpected_error_detail: str | None = None


class BehavioralArbitrationController:
    """Session discovery, S7 source selection, settings, arbitration dispatch."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], BehavioralArbitrationConfig] | None = None,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        runner: BehavioralArbitrationRunner = run_behavioral_arbitration_publication,
    ) -> None:
        self.state = BehavioralArbitrationUiState()
        self._config_loader = config_loader or (
            lambda: load_behavioral_arbitration_config(None)
        )
        self._folder_opener = folder_opener
        self._runner = runner
        self._pending_launch_timestamp: str | None = None
        self.reset_working_settings_to_defaults()

    def reset_working_settings_to_defaults(self) -> BehavioralArbitrationUiSettings:
        try:
            settings = BehavioralArbitrationUiSettings.from_config(self._config_loader())
        except (BehavioralArbitrationError, ConfigError) as exc:
            raise BehavioralArbitrationUiError(str(exc)) from exc
        self.state.working_settings = settings
        self._recompute_readiness()
        return settings

    def set_session(
        self, session_root: Path | str
    ) -> BehavioralArbitrationProjectSummary:
        self.prepare_session(session_root)
        return self.refresh_discovery()

    def prepare_session(self, session_root: Path | str) -> DiscoveryRequestIdentity:
        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise BehavioralArbitrationUiError(f"Session directory does not exist: {root}")
        previous = self.state.session_root
        if (
            self.state.task_running
            and previous is not None
            and previous.resolve(strict=False) != root
        ):
            raise BehavioralArbitrationUiError(
                "Cannot change S8 session while behavioral arbitration is active."
            )
        self.state.session_root = root
        self.state.readiness_error = None
        self.state.readiness_kind = READINESS_DISCOVERING
        self.state.run_error = None
        self.state.run_stage = "Not started"
        if previous is None or previous.resolve(strict=False) != root:
            self.state.last_result = None
            self.state.last_arbitration_settings = None
            self.state.selected_s8_run_id = None
            self.state.selected_s7_run_id = None
            self.state.s7_selection_manual = False
            self.state.s7_options = ()
            self.state.summary = None
        return self.begin_discovery()

    def clear_session(self) -> None:
        if self.state.task_running or self.state.discovery_running:
            raise BehavioralArbitrationUiError(
                "Cannot clear the S8 session while a task is active."
            )
        working = self.state.working_settings
        self.state = BehavioralArbitrationUiState(working_settings=working)
        self._recompute_readiness()

    def begin_discovery(self) -> DiscoveryRequestIdentity:
        root = self.state.session_root
        if root is None:
            raise BehavioralArbitrationUiError("Select a project/session first.")
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
            "Discovering S8-ready S7 runs and completed S8 artifacts…"
        )
        return identity

    def collect_discovery_snapshot(
        self, identity: DiscoveryRequestIdentity
    ) -> DiscoverySnapshot:
        root = identity.session_root
        try:
            assessments = list_s7_options_for_s8(root)
            eligible_s7 = any(
                item.s7_classification
                in ("eligible", "completed_with_warnings")
                for item in assessments
            )
            s8_summary = summarize_behavioral_arbitration_project(
                root,
                eligible_s7_available=eligible_s7,
                no_s7_reason=None if eligible_s7 else "no eligible S7 run",
            )
        except (BehavioralArbitrationError, PublicationError, OSError) as exc:
            raise BehavioralArbitrationUiError(str(exc)) from exc
        options = tuple(_option_from_assessment(item) for item in assessments)
        return DiscoverySnapshot(
            identity=identity,
            s7_options=options,
            s8_summary=s8_summary,
        )

    def apply_discovery_snapshot(self, snapshot: DiscoverySnapshot) -> bool:
        if not self._is_current_discovery_result(snapshot.identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.s7_options = snapshot.s7_options
        self.state.summary = snapshot.s8_summary
        self._reconcile_s7_selection()
        self._reconcile_s8_selection()
        self._recompute_readiness()
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

    def refresh_discovery(self) -> BehavioralArbitrationProjectSummary:
        identity = self.begin_discovery()
        snapshot = self.collect_discovery_snapshot(identity)
        applied = self.apply_discovery_snapshot(snapshot)
        if not applied or self.state.summary is None:
            raise BehavioralArbitrationUiError(
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

    def select_s7_source(
        self, run_id: str | None, *, manual: bool = True
    ) -> S7SourceOption | None:
        if self.state.task_running or self.state.discovery_running:
            raise BehavioralArbitrationUiError(
                "Cannot change S7 source while a Subsystem 8 task is active."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s7_run_id = None
            if manual:
                self.state.s7_selection_manual = True
            self._recompute_readiness()
            return None
        match = next((opt for opt in self.state.s7_options if opt.run_id == run_id), None)
        if match is None:
            raise BehavioralArbitrationUiError(
                f"S7 run is not available in the current session: {run_id}"
            )
        self.state.selected_s7_run_id = match.run_id
        if manual:
            self.state.s7_selection_manual = True
        self._recompute_readiness()
        return match

    def select_s8_run(
        self, run_id: str | None
    ) -> BehavioralArbitrationRunSummary | None:
        if run_id is None or not str(run_id).strip():
            self.state.selected_s8_run_id = None
            return None
        summary = self.state.summary
        if summary is None:
            raise BehavioralArbitrationUiError(
                "Refresh discovery before selecting an S8 run."
            )
        match = next((run for run in summary.runs if run.run_id == run_id), None)
        if match is None:
            raise BehavioralArbitrationUiError(f"S8 run was not found: {run_id}")
        self.state.selected_s8_run_id = match.run_id
        return match

    def selected_s7_source(self) -> S7SourceOption | None:
        selected = self.state.selected_s7_run_id
        if selected is None:
            return None
        return next((opt for opt in self.state.s7_options if opt.run_id == selected), None)

    def selected_s8_run(self) -> BehavioralArbitrationRunSummary | None:
        summary = self.state.summary
        selected = self.state.selected_s8_run_id
        if summary is None or selected is None:
            return None
        return next((run for run in summary.runs if run.run_id == selected), None)

    def update_settings(
        self, settings: BehavioralArbitrationUiSettings
    ) -> BehavioralArbitrationUiSettings:
        if self.state.task_running or self.state.discovery_running:
            raise BehavioralArbitrationUiError(
                "Cannot change S8 settings while a Subsystem 8 task is active."
            )
        validated = self.validate_settings(settings)
        self.state.working_settings = validated
        self._recompute_readiness()
        return validated

    def validate_settings(
        self, settings: BehavioralArbitrationUiSettings | None = None
    ) -> BehavioralArbitrationUiSettings:
        candidate = settings if settings is not None else self.state.working_settings
        try:
            return candidate.validated()
        except (BehavioralArbitrationError, ConfigError) as exc:
            raise BehavioralArbitrationUiError(str(exc)) from exc

    def can_run_arbitration(self) -> bool:
        return self.cannot_run_reason() is None

    def cannot_run_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 8 arbitration task is already running."
        if self.state.discovery_running:
            return "S8 discovery is still running; wait for readiness to finish."
        if self.state.session_root is None:
            return "Select a project/session first."
        if self.state.readiness_kind != READINESS_READY:
            return self.state.readiness_error or "Subsystem 8 is not ready for arbitration."
        source = self.selected_s7_source()
        if source is None:
            return "Select an S8-ready S7 run."
        if not source.s8_ready:
            return "Selected S7 run is not S8-ready: " + (
                "; ".join(source.reasons) if source.reasons else "see readiness panel"
            )
        return None

    def source_status_text(self) -> str:
        kind = self.state.readiness_kind
        if kind == READINESS_NO_SESSION or self.state.session_root is None:
            return "Select a project/session with an S8-ready S7 run."
        if kind == READINESS_DISCOVERING or self.state.discovery_running:
            return self.state.readiness_error or "Discovering S7/S8 artifacts…"
        if not self.state.s7_options:
            return (
                "No S7 runs were found for this session. "
                "Complete Subsystem 7 with full ethogram coverage, then Refresh."
            )
        source = self.selected_s7_source()
        if source is None:
            return self.state.readiness_error or "Select an S7 run."
        manual = " (manual)" if self.state.s7_selection_manual else " (default S8-ready)"
        ready = (
            "Ready for arbitration."
            if kind == READINESS_READY and source.s8_ready
            else f"Blocked [{kind}]: {self.state.readiness_error}"
        )
        s6 = source.linked_s6_run_id or "—"
        mice = (
            f"mice={list(source.mouse_ids)}"
            if source.mouse_ids
            else "mice unknown"
        )
        frames = (
            f"{source.n_frames} samples"
            if source.n_frames is not None
            else "sample count unknown"
        )
        return (
            f"S7={source.run_id}: historical={source.historical_status or '—'}, "
            f"S7 eligibility={source.s7_classification}, "
            f"S8-ready={source.s8_ready}{manual}; linked S6={s6}; "
            f"{mice}; {frames}. {ready}"
        )

    def begin_arbitration_launch(
        self,
        settings: BehavioralArbitrationUiSettings | None = None,
        *,
        timestamp: str | None = None,
    ) -> ArbitrationRequestIdentity:
        """Accept a run on the GUI thread without joint S7/S6 revalidation.

        Marks the task active immediately so duplicate clicks and source/settings
        mutations are blocked. Heavy ``assess_s7_for_s8`` runs later in
        :meth:`execute_arbitration_launch` on the worker thread.
        """

        reason = self.cannot_run_reason()
        if reason is not None:
            raise BehavioralArbitrationUiError(reason)
        source = self.selected_s7_source()
        root = self.state.session_root
        if source is None or root is None:
            raise BehavioralArbitrationUiError("Select an S8-ready S7 run first.")
        if not source.s8_ready:
            raise BehavioralArbitrationUiError(
                "Selected S7 run is not S8-ready: "
                + ("; ".join(source.reasons) if source.reasons else "see readiness panel")
            )
        validated = self.validate_settings(settings)
        identity = ArbitrationRequestIdentity(
            request_id=str(uuid4()),
            session_root=root.resolve(strict=False),
            s7_run_id=source.run_id,
            s7_run_dir=source.run_dir.resolve(strict=False),
            settings=validated,
        )
        # Stash optional publish timestamp on the identity via settings object
        # attribute is not available; store on state for the worker.
        self.state.active_arbitration_identity = identity
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Launching arbitration…"
        self.state.readiness_error = None
        self._pending_launch_timestamp = timestamp
        return identity

    def build_request(
        self, *, timestamp: str | None = None
    ) -> tuple[BehavioralArbitrationRequest, ArbitrationRequestIdentity]:
        """Validate readiness (including joint assess) and build a request.

        Prefer :meth:`begin_arbitration_launch` + :meth:`execute_arbitration_launch`
        on the GUI path so joint validation stays off the Qt main thread.
        """

        identity = self.begin_arbitration_launch(timestamp=timestamp)
        request = self._build_request_after_revalidate(identity)
        self.state.run_stage = "Running behavioral arbitration…"
        return request, identity

    def _build_request_after_revalidate(
        self, identity: ArbitrationRequestIdentity
    ) -> BehavioralArbitrationRequest:
        if not self._is_current_arbitration_result(identity):
            raise BehavioralArbitrationUiError(
                "Arbitration launch was superseded by a newer session/source selection."
            )
        root = identity.session_root
        settings = identity.settings
        try:
            assessment = assess_s7_for_s8(
                root,
                identity.s7_run_dir,
                require_exactly_two_mice=settings.require_exactly_two_mice,
            )
        except IntakeError as exc:
            raise BehavioralArbitrationUiError(str(exc)) from exc
        if not assessment.s8_ready:
            detail = "; ".join(assessment.reasons) or "joint S7/S6 validation failed"
            self.state.readiness_kind = READINESS_S7_NOT_S8_READY
            self.state.readiness_error = detail
            raise BehavioralArbitrationUiError(
                f"Selected S7 run is no longer S8-ready: {detail}"
            )
        if assessment.run_id != identity.s7_run_id:
            raise BehavioralArbitrationUiError(
                "Selected S7 run identity changed during launch revalidation."
            )
        timestamp = getattr(self, "_pending_launch_timestamp", None)
        self._pending_launch_timestamp = None
        return BehavioralArbitrationRequest(
            session_root=root,
            s7_run_dir=identity.s7_run_dir,
            config=settings.to_config(),
            run_purpose="gui",
            timestamp=timestamp,
        )

    def execute_request(
        self,
        request: BehavioralArbitrationRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> BehavioralArbitrationRunResult:
        return self._runner(request, progress_callback=progress_callback)

    def execute_arbitration_launch(
        self,
        identity: ArbitrationRequestIdentity,
        progress_callback: ProgressCallback | None = None,
    ) -> BehavioralArbitrationRunResult:
        """Worker-thread entry: revalidate sources, then publish arbitration."""

        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="intake",
                    message="Revalidating S7/S6 sources for S8…",
                    completed_units=0,
                    total_units=None,
                    total_is_estimate=True,
                    is_indeterminate=True,
                    processed_video_time_sec=None,
                )
            )
        request = self._build_request_after_revalidate(identity)
        self.state.run_stage = "Running behavioral arbitration…"
        return self._runner(request, progress_callback=progress_callback)

    def record_success(
        self,
        result: BehavioralArbitrationRunResult,
        *,
        expected_identity: ArbitrationRequestIdentity | None = None,
        refresh_discovery: bool = False,
    ) -> bool:
        self.state.task_running = False
        if not self._is_current_arbitration_result(expected_identity):
            return False
        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.last_arbitration_settings = (
            expected_identity.settings
            if expected_identity is not None
            else self.state.working_settings
        )
        self.state.selected_s8_run_id = result.run_id
        self.state.active_arbitration_identity = None
        if refresh_discovery:
            # Synchronous path for headless/controller tests only.
            try:
                self.refresh_discovery()
                self.state.selected_s8_run_id = result.run_id
            except BehavioralArbitrationUiError:
                pass
        return True

    def record_failure(
        self,
        message: str,
        *,
        expected_identity: ArbitrationRequestIdentity | None = None,
    ) -> bool:
        self.state.task_running = False
        if (
            expected_identity is not None
            and self.state.active_arbitration_identity is not None
            and expected_identity.request_id
            != self.state.active_arbitration_identity.request_id
        ):
            return False
        self.state.last_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message
        self.state.active_arbitration_identity = None
        return True

    def open_selected_s8_folder(self) -> bool:
        run = self.selected_s8_run()
        if run is None and self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.run_dir))
        if run is None:
            raise BehavioralArbitrationUiError("Select a completed S8 run first.")
        return bool(self._folder_opener(run.run_dir))

    def settings_display_rows(self) -> tuple[tuple[str, str], ...]:
        settings = self.state.working_settings
        sci = settings.scientific
        contact = sci.contact_post_processing
        rows: list[tuple[str, str]] = [
            ("fps", f"{settings.fps:g} Hz"),
            ("detector_coverage_profile", settings.detector_coverage_profile),
            ("require_exactly_two_mice", str(settings.require_exactly_two_mice)),
            ("decay_rate", f"{sci.decay_rate:g}"),
            ("conf_frames", f"{sci.conf_frames} frames"),
            ("engaged_min_threshold", f"{sci.engaged_min_threshold:g}"),
            (
                "Stage 8 contact morphological max_gap_to_fill",
                f"{contact.max_gap_to_fill} frames (0=disabled)",
            ),
            (
                "Stage 8 contact min_bout_duration",
                f"{contact.min_bout_duration} frames",
            ),
            (
                "Stage 10 gap_fill_seconds",
                f"{sci.temporal.gap_fill_seconds:g} s "
                f"({sci.temporal.gap_fill_seconds * settings.fps:g} frames @ fps)",
            ),
            (
                "Stage 10 min_bout_seconds",
                f"{sci.temporal.min_bout_seconds:g} s "
                f"({sci.temporal.min_bout_seconds * settings.fps:g} frames @ fps)",
            ),
            (
                "final_engaged_hysteresis_frames",
                f"{sci.final_engaged_hysteresis_frames} frames",
            ),
            (
                "min_mutually_disengaged_frames",
                f"{sci.min_mutually_disengaged_frames} frames",
            ),
            (
                "emit_arbitration_diagnostics",
                str(settings.execution.emit_arbitration_diagnostics),
            ),
        ]
        return tuple(rows)

    def ethogram_summary_rows(self) -> tuple[tuple[str, str], ...]:
        metrics = self._qc_metrics()
        if metrics is None:
            return ()
        rows: list[tuple[str, str]] = []
        directional = metrics.get("directional")
        if isinstance(directional, Mapping):
            rows.append(
                (
                    "Synchronized samples (unique frames)",
                    _format_metric(directional.get("n_unique_frames")),
                )
            )
            rows.append(
                (
                    "Directional state rows",
                    _format_metric(directional.get("n_directional_rows")),
                )
            )
        label_counts = metrics.get("label_counts")
        if isinstance(label_counts, Mapping):
            total = sum(int(v) for v in label_counts.values() if _is_intish(v))
            rows.append(("— Exclusive ethogram labels —", ""))
            for label, count in sorted(label_counts.items(), key=lambda kv: str(kv[0])):
                frac = (float(count) / total) if total else 0.0
                note = ""
                if str(label) == LABEL_ENGAGED:
                    note = " (valid residual; not parallel)"
                elif str(label) == LABEL_UNDEFINED:
                    note = " (unresolved invalid evidence)"
                rows.append(
                    (
                        f"Exclusive label: {label}{note}",
                        f"n={count}, fraction={frac:.4f}",
                    )
                )
        if metrics.get("n_undefined") is not None:
            rows.append(("Undefined count", _format_metric(metrics.get("n_undefined"))))
        if metrics.get("n_unresolved") is not None:
            rows.append(
                ("Unresolved (provisional) count", _format_metric(metrics.get("n_unresolved")))
            )
        parallel = metrics.get("parallel_coverage")
        if isinstance(parallel, Mapping):
            rows.append(("— Parallel channels (not exclusive) —", ""))
            for channel, info in parallel.items():
                if not isinstance(info, Mapping):
                    continue
                rows.append(
                    (
                        f"Parallel {channel}",
                        (
                            f"n_positive={info.get('n_positive')}, "
                            f"fraction={_format_metric(info.get('positive_fraction'))}"
                        ),
                    )
                )
        return tuple(rows)

    def bout_summary_rows(self) -> tuple[tuple[str, str], ...]:
        metrics = self._qc_metrics()
        if metrics is None:
            return ()
        summary = metrics.get("bout_summary")
        if not isinstance(summary, Mapping):
            return ()
        rows: list[tuple[str, str]] = [
            ("Total bouts", _format_metric(summary.get("n_bouts"))),
        ]
        by_channel = summary.get("by_channel")
        if isinstance(by_channel, Mapping):
            for channel, count in sorted(by_channel.items(), key=lambda kv: str(kv[0])):
                rows.append((f"Bouts channel={channel}", _format_metric(count)))
        duration = summary.get("duration_frames")
        if isinstance(duration, Mapping):
            rows.append(
                (
                    "Bout duration (frames)",
                    (
                        f"min={_format_metric(duration.get('min'))}, "
                        f"median={_format_metric(duration.get('median'))}, "
                        f"max={_format_metric(duration.get('max'))}"
                    ),
                )
            )
        return tuple(rows)

    def qc_display_rows(self) -> tuple[tuple[str, str], ...]:
        result = self.state.last_result
        run = self.selected_s8_run()
        meta_qc: Mapping[str, Any] | None = None
        status = None
        output = None
        classification = None
        source_s7 = None
        source_s6 = None

        use_last = result is not None and (run is None or run.run_id == result.run_id)
        if use_last and result is not None:
            status = result.status
            output = (
                str(result.behavior_states_path)
                if result.behavior_states_path is not None
                else str(result.run_dir)
            )
            meta_qc = result.qc
            source_s7 = result.source_s7_run_id
            source_s6 = result.source_s6_run_id
            if run is not None:
                classification = run.classification
                source_s7 = run.source_s7_run_id or source_s7
                source_s6 = run.source_s6_run_id or source_s6
        elif run is not None:
            status = run.status
            output = (
                str(run.behavior_states_path)
                if run.behavior_states_path
                else str(run.run_dir)
            )
            classification = run.classification
            source_s7 = run.source_s7_run_id
            source_s6 = run.source_s6_run_id
            meta_qc = _load_json_mapping(run.run_dir / QC_FILENAME)
        else:
            return ()

        rows: list[tuple[str, str]] = [
            ("Historical status", status or "—"),
            (
                "Current eligibility",
                _eligibility_label(classification) if classification else "—",
            ),
            ("Source S7", source_s7 or "—"),
            ("Source S6 (linked)", source_s6 or "—"),
            ("Output", output or "—"),
        ]
        if use_last and result is not None and result.warnings:
            rows.append(("Run warnings", "; ".join(result.warnings)))

        if isinstance(meta_qc, Mapping):
            structural = meta_qc.get("structural_ok")
            if structural is not None:
                rows.append(("Structural QC", "OK" if structural else "FAILED"))
            warnings = meta_qc.get("warnings")
            if isinstance(warnings, list) and warnings:
                rows.append(("QC warnings", "; ".join(str(w) for w in warnings)))
            failures = meta_qc.get("failures")
            if isinstance(failures, list) and failures:
                rows.append(("QC failures", "; ".join(str(f) for f in failures)))
            metrics = meta_qc.get("metrics")
            if isinstance(metrics, Mapping):
                reciprocity = metrics.get("reciprocity_repair_counts")
                if isinstance(reciprocity, Mapping):
                    rows.append(("Reciprocity repairs", _format_metric(reciprocity)))
                rule_changes = metrics.get("rule_change_counts")
                if isinstance(rule_changes, Mapping) and rule_changes:
                    rows.append(
                        (
                            "Rule-stage change counts",
                            _format_metric(dict(list(rule_changes.items())[:12])),
                        )
                    )
                if metrics.get("n_routing_sentinel_cells") is not None:
                    rows.append(
                        (
                            "Routing-sentinel cells",
                            _format_metric(metrics.get("n_routing_sentinel_cells")),
                        )
                    )
                provenance = metrics.get("source_provenance")
                if isinstance(provenance, Mapping):
                    src7 = provenance.get("source_s7")
                    src6 = provenance.get("source_s6")
                    if isinstance(src7, Mapping):
                        rows.append(
                            (
                                "S7 digests",
                                _format_metric(
                                    {
                                        k: src7.get(k)
                                        for k in (
                                            "predictions_sha256",
                                            "dyadic_wide_sha256",
                                            "focal_wide_sha256",
                                        )
                                        if src7.get(k)
                                    }
                                ),
                            )
                        )
                    if isinstance(src6, Mapping):
                        rows.append(
                            (
                                "S6 digest",
                                _format_metric(
                                    {
                                        "dyadic_features_sha256": src6.get(
                                            "dyadic_features_sha256"
                                        )
                                    }
                                ),
                            )
                        )
        return tuple(rows)

    def _qc_metrics(self) -> Mapping[str, Any] | None:
        result = self.state.last_result
        run = self.selected_s8_run()
        meta_qc: Mapping[str, Any] | None = None
        use_last = result is not None and (run is None or run.run_id == result.run_id)
        if use_last and result is not None and isinstance(result.qc, Mapping):
            meta_qc = result.qc
        elif run is not None:
            meta_qc = _load_json_mapping(run.run_dir / QC_FILENAME)
        if not isinstance(meta_qc, Mapping):
            return None
        metrics = meta_qc.get("metrics")
        return metrics if isinstance(metrics, Mapping) else None

    def _reconcile_s7_selection(self) -> None:
        options = self.state.s7_options
        selected = self.state.selected_s7_run_id
        ids = {opt.run_id for opt in options}
        if self.state.s7_selection_manual:
            if selected is not None and selected in ids:
                return
            self.state.selected_s7_run_id = None
            return
        ready = next((opt for opt in options if opt.s8_ready), None)
        self.state.selected_s7_run_id = ready.run_id if ready is not None else None

    def _reconcile_s8_selection(self) -> None:
        summary = self.state.summary
        selected = self.state.selected_s8_run_id
        if summary is None:
            return
        ids = {run.run_id for run in summary.runs}
        if selected is not None and selected in ids:
            return
        if self.state.last_result is not None and self.state.last_result.run_id in ids:
            self.state.selected_s8_run_id = self.state.last_result.run_id

    def _recompute_readiness(self) -> None:
        if self.state.session_root is None:
            self.state.readiness_kind = READINESS_NO_SESSION
            self.state.readiness_error = "Select a project/session first."
            self.state.readiness_warnings = ()
            return
        if self.state.discovery_running:
            self.state.readiness_kind = READINESS_DISCOVERING
            return
        if not self.state.s7_options:
            self.state.readiness_kind = READINESS_NO_S7
            self.state.readiness_error = (
                "No S7 runs discovered. Complete full-ethogram S7 first."
            )
            self.state.readiness_warnings = ()
            return
        source = self.selected_s7_source()
        if source is None:
            ready_any = any(opt.s8_ready for opt in self.state.s7_options)
            self.state.readiness_kind = (
                READINESS_S7_NOT_S8_READY if not ready_any else READINESS_OTHER
            )
            self.state.readiness_error = (
                "No S8-ready S7 run is available."
                if not ready_any
                else "Select an S7 run (manual selection cleared or not S8-ready)."
            )
            self.state.readiness_warnings = ()
            return
        warnings = list(source.warnings)
        if not source.s8_ready:
            self.state.readiness_kind = READINESS_S7_NOT_S8_READY
            self.state.readiness_error = "; ".join(source.reasons) or (
                "Selected S7 run fails joint S7/S6 S8 readiness."
            )
            self.state.readiness_warnings = tuple(warnings)
            return
        self.state.readiness_kind = READINESS_READY
        self.state.readiness_error = None
        self.state.readiness_warnings = tuple(warnings)

    def _is_current_arbitration_result(
        self, expected: ArbitrationRequestIdentity | None
    ) -> bool:
        if expected is None:
            return self.state.active_arbitration_identity is None
        active = self.state.active_arbitration_identity
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


def _option_from_assessment(item: S7S8ReadinessAssessment) -> S7SourceOption:
    tag = "S8-ready" if item.s8_ready else "not S8-ready"
    return S7SourceOption(
        run_id=item.run_id,
        run_dir=item.run_dir,
        historical_status=item.historical_status,
        s7_classification=item.s7_classification,
        s8_ready=item.s8_ready,
        label=f"{item.run_id} [{tag}]",
        linked_s6_run_id=item.linked_s6_run_id,
        linked_s6_run_dir=item.linked_s6_run_dir,
        mouse_ids=item.mouse_ids,
        n_frames=item.n_frames,
        detectors_dyadic=item.detectors_dyadic,
        detectors_focal=item.detectors_focal,
        missing_detectors=item.missing_detectors,
        missing_s6_columns=item.missing_s6_columns,
        reasons=item.reasons,
        warnings=item.warnings,
        lineage_aligned=item.lineage_aligned,
    )


def replace_scientific(
    settings: BehavioralArbitrationUiSettings, **kwargs: Any
) -> BehavioralArbitrationUiSettings:
    """Helper for page widgets to patch nested scientific settings."""

    sci = settings.scientific
    contact_kwargs = kwargs.pop("contact", None)
    postural_approach = kwargs.pop("postural_approach", None)
    postural_retract = kwargs.pop("postural_retract", None)
    hierarchy_kwargs = kwargs.pop("hierarchy", None)
    temporal_kwargs = kwargs.pop("temporal", None)
    adaptive_kwargs = kwargs.pop("adaptive", None)
    rescue_kwargs = kwargs.pop("rescue", None)
    stitch_kwargs = kwargs.pop("stitch_dynamic", None)

    contact = sci.contact_post_processing
    if isinstance(contact_kwargs, Mapping):
        contact = replace(contact, **dict(contact_kwargs)).validated()

    approach = sci.postural.approach
    if isinstance(postural_approach, Mapping):
        approach = replace(approach, **dict(postural_approach)).validated()
    retract = sci.postural.retract
    if isinstance(postural_retract, Mapping):
        retract = replace(retract, **dict(postural_retract)).validated()
    postural = PosturalSettings(approach=approach, retract=retract).validated()

    hierarchy = sci.hierarchy
    if isinstance(hierarchy_kwargs, Mapping):
        hierarchy = replace(hierarchy, **dict(hierarchy_kwargs))

    temporal = sci.temporal
    if isinstance(temporal_kwargs, Mapping):
        temporal = replace(temporal, **dict(temporal_kwargs))

    adaptive = sci.adaptive
    if isinstance(adaptive_kwargs, Mapping):
        adaptive = replace(adaptive, **dict(adaptive_kwargs))

    rescue = sci.rescue
    if isinstance(rescue_kwargs, Mapping):
        rescue = replace(rescue, **dict(rescue_kwargs))

    stitch = sci.stitch_dynamic
    if isinstance(stitch_kwargs, Mapping):
        stitch = replace(stitch, **dict(stitch_kwargs))

    new_sci = replace(
        sci,
        contact_post_processing=contact,
        postural=postural,
        hierarchy=hierarchy,
        temporal=temporal,
        adaptive=adaptive,
        rescue=rescue,
        stitch_dynamic=stitch,
        **kwargs,
    )
    return replace(settings, scientific=new_sci)


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


def _is_intish(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
