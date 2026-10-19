"""Headless coordination for Subsystem 04 pose-finalization GUI workflow."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pose_finalization import (
    AcceptedS3Run,
    GAP_KALMAN_MAX_FRAMES_DEFAULT,
    GAP_LINEAR_MAX_FRAMES_DEFAULT,
    GapFillConfig,
    PoseFinalizationError,
    PoseFinalizationRequest,
    PoseFinalizationResult,
    SAVGOL_ENABLED_DEFAULT,
    SAVGOL_POLYORDER_DEFAULT,
    SAVGOL_WINDOW_LENGTH_DEFAULT,
    SmoothConfig,
    TARGET_FPS_DEFAULT,
    find_latest_accepted_s3_run,
    list_accepted_s3_runs,
    run_pose_finalization,
    validate_accepted_s3_run,
)
from pose_finalization.run_discovery import (
    PoseFinalizationProjectSummary,
    PoseFinalizationRunSummary,
    summarize_pose_finalization_project,
)
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

FolderOpener = Callable[[Path], bool]


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
        runner: Callable[[PoseFinalizationRequest], PoseFinalizationResult] = run_pose_finalization,
    ) -> None:
        self.state = PoseFinalizationUiState()
        self._folder_opener = folder_opener
        self._runner = runner

    def set_session(self, session_root: Path | str) -> PoseFinalizationProjectSummary:
        """Open a project/session and refresh accepted-S3 / completed-S4 discovery."""

        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise PoseFinalizationUiError(f"Session directory does not exist: {root}")
        self.state.session_root = root
        self.state.last_result = None
        self.state.run_error = None
        self.state.run_stage = "Not started"
        return self.refresh_discovery()

    def refresh_discovery(self) -> PoseFinalizationProjectSummary:
        """Re-scan accepted S3 inputs and completed S4 outputs for the session."""

        root = self.state.session_root
        if root is None:
            raise PoseFinalizationUiError("Select a project/session first.")
        try:
            summary = summarize_pose_finalization_project(root)
            accepted = list_accepted_s3_runs(root)
        except PoseFinalizationError as exc:
            raise PoseFinalizationUiError(str(exc)) from exc
        self.state.summary = summary
        self.state.accepted_s3_runs = accepted
        self._reconcile_s3_selection()
        self._reconcile_s4_selection()
        return summary

    def select_accepted_s3_run(self, run_id: str | None) -> AcceptedS3Run | None:
        """Select an accepted S3 run by id, or clear selection."""

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
            and self.state.session_root is not None
            and self.selected_accepted_s3() is not None
        )

    def cannot_run_reason(self) -> str | None:
        """Explain why finalization cannot start, when applicable."""

        if self.state.task_running:
            return "Pose finalization is already running."
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

    def build_request(self, *, timestamp: str | None = None) -> PoseFinalizationRequest:
        """Build a validated runner request from the current GUI state."""

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
        return PoseFinalizationRequest(
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

    def execute_request(
        self, request: PoseFinalizationRequest
    ) -> PoseFinalizationResult:
        """Run one validated request (worker-thread safe; no Qt updates)."""

        return self._runner(request)

    def record_success(self, result: PoseFinalizationResult) -> None:
        """Apply a successful result on the GUI thread after the worker finishes."""

        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.selected_s4_run_id = result.run_id
        try:
            self.refresh_discovery()
            self.state.selected_s4_run_id = result.run_id
        except PoseFinalizationUiError:
            pass

    def record_failure(self, message: str) -> None:
        """Record a failed/interrupted run without marking it completed."""

        self.state.last_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message

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
        latest = find_latest_accepted_s3_run(self.state.session_root) if self.state.session_root else None
        self.state.selected_s3_run_id = (
            latest.run_id if latest is not None else accepted[0].run_id
        )

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
