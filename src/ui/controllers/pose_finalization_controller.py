"""Headless coordination for Subsystem 04 pose-finalization GUI workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from pose_finalization import (
    GAP_KALMAN_MAX_FRAMES_DEFAULT,
    GAP_LINEAR_MAX_FRAMES_DEFAULT,
    SAVGOL_ENABLED_DEFAULT,
    SAVGOL_POLYORDER_DEFAULT,
    SAVGOL_WINDOW_LENGTH_DEFAULT,
    STATUS_CANCELED,
    TARGET_FPS_DEFAULT,
    AcceptedS3Run,
    GapFillConfig,
    PoseFinalizationError,
    PoseFinalizationRequest,
    PoseFinalizationResult,
    SmoothConfig,
    list_accepted_s3_runs,
    run_pose_finalization,
    validate_accepted_s3_run,
)
from pose_finalization.run_discovery import (
    PoseFinalizationProjectSummary,
    PoseFinalizationRunSummary,
    summarize_pose_finalization_project,
)
from preprocess.models import OperationProgress
from runtime.cancellation import CancellationCallback
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

FolderOpener = Callable[[Path], bool]
ProgressCallback = Callable[[OperationProgress], None]
PublicationStartedCallback = Callable[[], None]
PoseFinalizationRunner = Callable[..., PoseFinalizationResult]


class PoseFinalizationUiError(ValueError):
    """Expected, actionable Subsystem 04 UI validation error."""


@dataclass(frozen=True, slots=True)
class PoseFinalizationUiSettings:
    """GUI-exposed Advanced Settings for one S4 run."""

    linear_max_frames: int = GAP_LINEAR_MAX_FRAMES_DEFAULT
    kalman_max_frames: int = GAP_KALMAN_MAX_FRAMES_DEFAULT
    process_var: float = 10.0
    measurement_var: float = 1.0
    smooth_enabled: bool = SAVGOL_ENABLED_DEFAULT
    savgol_window_length: int = SAVGOL_WINDOW_LENGTH_DEFAULT
    savgol_polyorder: int = SAVGOL_POLYORDER_DEFAULT
    target_fps: float = TARGET_FPS_DEFAULT


@dataclass(frozen=True, slots=True)
class DiscoveryRequestIdentity:
    """Identity used to reject stale asynchronous discovery results."""

    request_id: str
    generation: int
    session_root: Path


@dataclass(frozen=True, slots=True)
class FinalizationRequestIdentity:
    """Identity used to reject stale asynchronous finalization results."""

    request_id: str
    session_root: Path
    s3_run_id: str
    s3_run_dir: Path


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Payload collected off the GUI thread for one discovery generation."""

    identity: DiscoveryRequestIdentity
    summary: PoseFinalizationProjectSummary
    accepted_s3_runs: tuple[AcceptedS3Run, ...]


@dataclass(slots=True)
class PoseFinalizationUiState:
    """Compact mutable state for the S4 page (no pose arrays)."""

    session_root: Path | None = None
    summary: PoseFinalizationProjectSummary | None = None
    accepted_s3_runs: tuple[AcceptedS3Run, ...] = ()
    selected_s3_run_id: str | None = None
    selected_s4_run_id: str | None = None
    settings: PoseFinalizationUiSettings = field(default_factory=PoseFinalizationUiSettings)
    task_running: bool = False
    discovery_running: bool = False
    discovery_generation: int = 0
    active_discovery_identity: DiscoveryRequestIdentity | None = None
    active_finalization_identity: FinalizationRequestIdentity | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: PoseFinalizationResult | None = None
    unexpected_error_detail: str | None = None


class PoseFinalizationController:
    """Session discovery, settings validation, and S4 runner dispatch."""

    def __init__(
        self,
        *,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        runner: PoseFinalizationRunner = run_pose_finalization,
    ) -> None:
        self.state = PoseFinalizationUiState()
        self._folder_opener = folder_opener
        self._runner = runner

    def set_session(self, session_root: Path | str) -> PoseFinalizationProjectSummary:
        """Open a project/session and refresh accepted-S3 / completed-S4 discovery (sync)."""

        self.prepare_session(session_root)
        return self.refresh_discovery()

    def prepare_session(self, session_root: Path | str) -> DiscoveryRequestIdentity:
        """Bind a session and bump discovery generation without scanning yet."""

        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise PoseFinalizationUiError(f"Session directory does not exist: {root}")
        previous = self.state.session_root
        if (
            self.state.task_running
            and previous is not None
            and previous.resolve(strict=False) != root
        ):
            raise PoseFinalizationUiError(
                "Cannot change S4 session while pose finalization is active."
            )
        self.state.session_root = root
        self.state.last_result = None
        self.state.run_error = None
        self.state.run_stage = "Not started"
        if previous is None or previous.resolve(strict=False) != root:
            self.state.summary = None
            self.state.accepted_s3_runs = ()
            self.state.selected_s3_run_id = None
            self.state.selected_s4_run_id = None
        return self.begin_discovery()

    def begin_discovery(self) -> DiscoveryRequestIdentity:
        """Mark discovery in progress and return an identity for async apply."""

        root = self.state.session_root
        if root is None:
            raise PoseFinalizationUiError("Select a project/session first.")
        self.state.discovery_generation += 1
        identity = DiscoveryRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.discovery_generation,
            session_root=root.resolve(strict=False),
        )
        self.state.active_discovery_identity = identity
        self.state.discovery_running = True
        return identity

    def collect_discovery_snapshot(
        self, identity: DiscoveryRequestIdentity
    ) -> DiscoverySnapshot:
        """Heavy filesystem / hashing work safe to run off the GUI thread."""

        root = identity.session_root
        try:
            summary = summarize_pose_finalization_project(root)
            accepted = list_accepted_s3_runs(root)
        except (PoseFinalizationError, OSError) as exc:
            raise PoseFinalizationUiError(str(exc)) from exc
        return DiscoverySnapshot(
            identity=identity,
            summary=summary,
            accepted_s3_runs=tuple(accepted),
        )

    def apply_discovery_snapshot(self, snapshot: DiscoverySnapshot) -> bool:
        """Apply a discovery result only when its generation is still current."""

        if not self._is_current_discovery_result(snapshot.identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.summary = snapshot.summary
        self.state.accepted_s3_runs = snapshot.accepted_s3_runs
        self._reconcile_s3_selection()
        self._reconcile_s4_selection()
        return True

    def record_discovery_failure(
        self, message: str, *, expected_identity: DiscoveryRequestIdentity | None
    ) -> bool:
        """Record a discovery failure when the request is still current."""

        if not self._is_current_discovery_result(expected_identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.run_error = message
        return True

    def refresh_discovery(self) -> PoseFinalizationProjectSummary:
        """Re-scan accepted S3 inputs and completed S4 outputs (synchronous)."""

        identity = self.begin_discovery()
        try:
            snapshot = self.collect_discovery_snapshot(identity)
        except PoseFinalizationUiError:
            # Synchronous callers see the error directly; do not leave discovery "running".
            if self._is_current_discovery_result(identity):
                self.state.discovery_running = False
                self.state.active_discovery_identity = None
            raise
        applied = self.apply_discovery_snapshot(snapshot)
        if not applied or self.state.summary is None:
            raise PoseFinalizationUiError(
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
        return current_root.resolve(strict=False) == expected.session_root.resolve(
            strict=False
        )

    def select_accepted_s3_run(self, run_id: str | None) -> AcceptedS3Run | None:
        """Select an accepted S3 run by id, or clear selection."""

        if self.state.task_running or self.state.discovery_running:
            raise PoseFinalizationUiError(
                "Cannot change the accepted S3 source while an S4 task is active."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s3_run_id = None
            return None
        match = next((run for run in self.state.accepted_s3_runs if run.run_id == run_id), None)
        if match is None:
            raise PoseFinalizationUiError(
                f"Accepted S3 run is not available in the current session: {run_id}"
            )
        # Re-validate immediately so superseded/SHA-mismatched runs cannot be selected.
        try:
            validated = validate_accepted_s3_run(match.run_dir)
        except PoseFinalizationError as exc:
            self.refresh_discovery()
            raise PoseFinalizationUiError(str(exc)) from exc
        self.state.selected_s3_run_id = validated.run_id
        return validated

    def select_s4_run(self, run_id: str | None) -> PoseFinalizationRunSummary | None:
        """Select a discovered S4 run for QC/output inspection."""

        if run_id is None or not str(run_id).strip():
            self.state.selected_s4_run_id = None
            return None
        summary = self.state.summary
        if summary is None:
            raise PoseFinalizationUiError("Refresh discovery before selecting an S4 run.")
        match = next((run for run in summary.runs if run.run_id == run_id), None)
        if match is None:
            raise PoseFinalizationUiError(f"S4 run was not found: {run_id}")
        self.state.selected_s4_run_id = match.run_id
        return match

    def update_settings(self, settings: PoseFinalizationUiSettings) -> PoseFinalizationUiSettings:
        """Validate and store Advanced Settings (backend defaults when untouched)."""

        validated = self.validate_settings(settings)
        self.state.settings = validated
        return validated

    def validate_settings(
        self, settings: PoseFinalizationUiSettings | None = None
    ) -> PoseFinalizationUiSettings:
        """Reject incompatible Advanced Settings before execution."""

        candidate = settings if settings is not None else self.state.settings
        if abs(float(candidate.target_fps) - TARGET_FPS_DEFAULT) > 1e-9:
            raise PoseFinalizationUiError(
                f"Target FPS must remain {TARGET_FPS_DEFAULT:g} Hz for the MVP timing contract."
            )
        try:
            GapFillConfig(
                linear_max_frames=int(candidate.linear_max_frames),
                kalman_max_frames=int(candidate.kalman_max_frames),
                process_var=float(candidate.process_var),
                measurement_var=float(candidate.measurement_var),
            )
            SmoothConfig(
                enabled=bool(candidate.smooth_enabled),
                window_length=int(candidate.savgol_window_length),
                polyorder=int(candidate.savgol_polyorder),
            )
        except PoseFinalizationError as exc:
            raise PoseFinalizationUiError(str(exc)) from exc
        return PoseFinalizationUiSettings(
            linear_max_frames=int(candidate.linear_max_frames),
            kalman_max_frames=int(candidate.kalman_max_frames),
            process_var=float(candidate.process_var),
            measurement_var=float(candidate.measurement_var),
            smooth_enabled=bool(candidate.smooth_enabled),
            savgol_window_length=int(candidate.savgol_window_length),
            savgol_polyorder=int(candidate.savgol_polyorder),
            target_fps=TARGET_FPS_DEFAULT,
        )

    def selected_accepted_s3(self) -> AcceptedS3Run | None:
        """Return the currently selected accepted S3 run, if any."""

        selected_id = self.state.selected_s3_run_id
        if selected_id is None:
            return None
        return next(
            (run for run in self.state.accepted_s3_runs if run.run_id == selected_id),
            None,
        )

    def selected_s4_run(self) -> PoseFinalizationRunSummary | None:
        """Return the currently selected completed/discovered S4 run, if any."""

        summary = self.state.summary
        selected_id = self.state.selected_s4_run_id
        if summary is None or selected_id is None:
            return None
        return next((run for run in summary.runs if run.run_id == selected_id), None)

    def can_run_finalization(self) -> bool:
        """Return whether Run Finalization is currently allowed."""

        return (
            not self.state.task_running
            and not self.state.discovery_running
            and self.state.session_root is not None
            and self.selected_accepted_s3() is not None
        )

    def cannot_run_reason(self) -> str | None:
        """Explain why finalization cannot start, when applicable."""

        if self.state.task_running:
            return "Pose finalization is already running."
        if self.state.discovery_running:
            return "S4 discovery is still running."
        if self.state.session_root is None:
            return "Select a project/session first."
        if self.selected_accepted_s3() is None:
            summary = self.state.summary
            if summary is not None and summary.no_accepted_s3_reason:
                return summary.no_accepted_s3_reason
            return (
                "No eligible accepted S3 result is selected. Accept a Subsystem 3 "
                "run before running pose finalization."
            )
        return None

    def build_request(
        self, *, timestamp: str | None = None
    ) -> tuple[PoseFinalizationRequest, FinalizationRequestIdentity]:
        """Build a validated runner request and identity from the current GUI state."""

        if self.state.discovery_running:
            raise PoseFinalizationUiError("S4 discovery is still running.")
        if self.state.task_running:
            raise PoseFinalizationUiError("Pose finalization is already running.")
        accepted = self.selected_accepted_s3()
        if accepted is None:
            reason = self.cannot_run_reason() or "No accepted S3 run selected."
            raise PoseFinalizationUiError(reason)
        settings = self.validate_settings()
        try:
            # Fresh validation immediately before dispatch.
            validated = validate_accepted_s3_run(accepted.run_dir)
        except PoseFinalizationError as exc:
            self.refresh_discovery()
            raise PoseFinalizationUiError(str(exc)) from exc
        request = PoseFinalizationRequest(
            s3_run_dir=validated.run_dir,
            session_root=validated.session_root,
            run_purpose="gui",
            timestamp=timestamp,
            gap_fill_config=GapFillConfig(
                linear_max_frames=settings.linear_max_frames,
                kalman_max_frames=settings.kalman_max_frames,
                process_var=settings.process_var,
                measurement_var=settings.measurement_var,
            ),
            smooth_config=SmoothConfig(
                enabled=settings.smooth_enabled,
                window_length=settings.savgol_window_length,
                polyorder=settings.savgol_polyorder,
            ),
        )
        identity = FinalizationRequestIdentity(
            request_id=str(uuid4()),
            session_root=validated.session_root.resolve(strict=False),
            s3_run_id=validated.run_id,
            s3_run_dir=validated.run_dir.resolve(strict=False),
        )
        self.state.active_finalization_identity = identity
        self.state.task_running = True
        self.state.run_stage = "Running pose finalization…"
        self.state.run_error = None
        return request, identity

    def execute_request(
        self,
        request: PoseFinalizationRequest,
        progress_callback: ProgressCallback | None = None,
        *,
        cancellation_requested: CancellationCallback | None = None,
        publication_started: PublicationStartedCallback | None = None,
    ) -> PoseFinalizationResult:
        """Run one validated request (worker-thread safe; no Qt updates)."""

        return self._runner(
            request,
            progress_callback=progress_callback,
            cancellation_requested=cancellation_requested,
            publication_started=publication_started,
        )

    def record_success(
        self,
        result: PoseFinalizationResult,
        *,
        expected_identity: FinalizationRequestIdentity | None = None,
    ) -> bool:
        """Apply a successful result only when the finalization identity is current."""

        self.state.task_running = False
        if not self._is_current_finalization_result(expected_identity):
            return False
        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.selected_s4_run_id = result.run_id
        self.state.active_finalization_identity = None
        try:
            self.refresh_discovery()
            self.state.selected_s4_run_id = result.run_id
        except PoseFinalizationUiError:
            pass
        return True

    def record_failure(
        self,
        message: str,
        *,
        expected_identity: FinalizationRequestIdentity | None = None,
        canceled: bool = False,
    ) -> bool:
        """Record a failed or canceled run without marking it completed."""

        self.state.task_running = False
        if (
            expected_identity is not None
            and self.state.active_finalization_identity is not None
            and expected_identity.request_id
            != self.state.active_finalization_identity.request_id
        ):
            return False
        self.state.last_result = None
        self.state.run_stage = STATUS_CANCELED if canceled else "Failed"
        self.state.run_error = message
        self.state.active_finalization_identity = None
        return True

    def _is_current_finalization_result(
        self, expected: FinalizationRequestIdentity | None
    ) -> bool:
        if expected is None:
            return self.state.active_finalization_identity is None
        active = self.state.active_finalization_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return current_root.resolve(strict=False) == expected.session_root.resolve(
            strict=False
        )

    def open_selected_s4_folder(self) -> bool:
        """Open the selected S4 run directory in the platform file browser."""

        run = self.selected_s4_run()
        if run is None and self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.run_dir))
        if run is None:
            raise PoseFinalizationUiError("Select a completed S4 run first.")
        return bool(self._folder_opener(run.run_dir))

    def qc_display_rows(self) -> tuple[tuple[str, str], ...]:
        """Compact QC rows for the results panel."""

        result = self.state.last_result
        run = self.selected_s4_run()
        meta_qc: Mapping[str, Any] | None = None
        status = None
        output = None
        smooth_dir: Path | None = None

        use_last_result = result is not None and (
            run is None or run.run_id == result.run_id
        )
        if use_last_result and result is not None:
            status = result.status
            output = (
                str(result.final_pose_path)
                if result.final_pose_path is not None
                else None
            )
            meta_qc = result.qc
            smooth_dir = result.run_dir
        elif run is not None:
            status = run.status
            output = str(run.final_pose_path) if run.final_pose_path else None
            meta_qc = _load_run_qc(run.run_dir)
            smooth_dir = run.run_dir
        else:
            return ()

        rows: list[tuple[str, str]] = [
            ("Status", status or "—"),
            ("Output", output or "—"),
        ]
        if not isinstance(meta_qc, Mapping):
            return tuple(rows)
        metrics = meta_qc.get("metrics")
        if isinstance(metrics, Mapping):
            missing = metrics.get("missing_fraction_overall", metrics.get("missing_fraction"))
            imputed = metrics.get("imputed_fraction")
            if missing is not None:
                rows.append(("Missingness", f"{float(missing) * 100:.2f}%"))
            if imputed is not None:
                rows.append(
                    ("Reconstructed coordinates", f"{float(imputed) * 100:.2f}%")
                )
        smooth = _load_smoothing_summary(smooth_dir) if smooth_dir is not None else None
        if isinstance(smooth, Mapping):
            enabled = smooth.get("enabled")
            count = smooth.get("smoothed_sample_count")
            rows.append(
                (
                    "Smoothing",
                    (
                        f"enabled={enabled}, smoothed_samples={count}"
                        if count is not None
                        else f"enabled={enabled}"
                    ),
                )
            )
        warnings = meta_qc.get("warnings")
        if isinstance(warnings, list) and warnings:
            rows.append(("Warnings", "; ".join(str(w) for w in warnings)))
        elif isinstance(meta_qc.get("warning_count"), int):
            rows.append(("Warnings", str(meta_qc["warning_count"])))
        return tuple(rows)

    def _reconcile_s3_selection(self) -> None:
        accepted = self.state.accepted_s3_runs
        if not accepted:
            self.state.selected_s3_run_id = None
            return
        current = self.state.selected_s3_run_id
        if current is not None and any(run.run_id == current for run in accepted):
            return
        # ``accepted`` is already the validated, newest-first list produced by
        # ``list_accepted_s3_runs``; its head is exactly what
        # ``find_latest_accepted_s3_run`` would return, without a second full scan
        # (and SHA validation) on the GUI thread.
        self.state.selected_s3_run_id = accepted[0].run_id

    def _reconcile_s4_selection(self) -> None:
        summary = self.state.summary
        if summary is None or not summary.runs:
            if self.state.last_result is None:
                self.state.selected_s4_run_id = None
            return
        current = self.state.selected_s4_run_id
        if current is not None and any(run.run_id == current for run in summary.runs):
            return
        # Prefer newest completed run for inspection.
        completed = summary.completed_runs
        self.state.selected_s4_run_id = completed[0].run_id if completed else summary.runs[0].run_id


def _load_run_qc(run_dir: Path) -> Mapping[str, Any] | None:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    qc = payload.get("qc")
    return qc if isinstance(qc, Mapping) else None


def _load_smoothing_summary(run_dir: Path) -> Mapping[str, Any] | None:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("smoothing_summary")
    return summary if isinstance(summary, Mapping) else None
