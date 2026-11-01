"""Subsystem 10 — PI report export controller (Increment B2).

Owns export settings, request identity, discovery refresh, and worker execution
for immutable PI HTML reports. Does not mutate S8/S9/S10 science trees or gallery
edit state. Encoding/hashing run only through GuiTaskRunner progressive ops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from behavioral_review.pi_report import (
    PI_REPORT_RUN_PREFIX,
    PI_SOURCE_AUTOMATED,
    PI_SOURCE_REVIEWED,
    RESULTS_EXPORTS_DIRNAME,
    STATUS_CANCELED,
    PiReportProjectSummary,
    PiReportResult,
    publish_pi_report,
    summarize_pi_report_project,
)
from preprocess.models import OperationProgress
from ui.controllers.pi_report_support import (
    PiReportRequestIdentity,
    PiReportUiSettings,
    pi_report_table_rows,
    source_identity_summary,
)

ProgressCallback = Callable[[OperationProgress], None]
FolderOpener = Callable[[Path], bool]
HtmlOpener = Callable[[Path], bool]


class PiReportUiError(RuntimeError):
    """Operator-facing PI report GUI error."""


@dataclass(slots=True)
class PiReportUiState:
    """Mutable PI report export pane state."""

    settings: PiReportUiSettings = field(default_factory=PiReportUiSettings)
    project_summary: PiReportProjectSummary | None = None
    selected_report_id: str | None = None
    task_running: bool = False
    active_identity: PiReportRequestIdentity | None = None
    last_result: PiReportResult | None = None
    last_outcome: str | None = None  # completed | failed | canceled
    last_error: str | None = None
    run_stage: str = "Idle"
    source_summary: str = "No PI report source selected."
    output_destination: str = ""
    reserved_output_dirs: set[str] = field(default_factory=set)


class PiReportController:
    """Launch, track, and discover PI report exports for one session."""

    def __init__(
        self,
        *,
        publish_runner: Callable[..., PiReportResult] | None = None,
        folder_opener: FolderOpener | None = None,
        html_opener: HtmlOpener | None = None,
    ) -> None:
        self.state = PiReportUiState()
        self._publish_runner = publish_runner or publish_pi_report
        self._folder_opener = folder_opener or _default_folder_opener
        self._html_opener = html_opener or _default_html_opener

    def update_settings(self, settings: PiReportUiSettings) -> PiReportUiSettings:
        if self.state.task_running:
            raise PiReportUiError("Cannot change PI report settings while export runs.")
        validated = settings.validated()
        self.state.settings = validated
        return validated

    def refresh_discovery(self, session_root: Path | str | None) -> PiReportProjectSummary | None:
        if session_root is None:
            self.state.project_summary = None
            self.state.selected_report_id = None
            self.state.output_destination = ""
            return None
        root = Path(session_root).expanduser().resolve(strict=False)
        summary = summarize_pi_report_project(root)
        self.state.project_summary = summary
        self.state.output_destination = str(summary.exports_root)
        self._reconcile_selection()
        return summary

    def report_table_rows(self) -> tuple[tuple[str, ...], ...]:
        summary = self.state.project_summary
        rows = list(pi_report_table_rows(summary.reports if summary is not None else ()))
        # Surface the latest failed/canceled attempt that left no durable folder.
        if (
            self.state.last_outcome in {STATUS_CANCELED, "failed"}
            and self.state.active_identity is None
            and self.state.last_result is None
            and self.state.last_error
        ):
            rows.insert(
                0,
                (
                    "(last attempt)",
                    str(self.state.last_outcome),
                    "—",
                    self.state.settings.report_scope_label(),
                    "—",
                    "no",
                    "—",
                    "—",
                    self.state.last_error,
                ),
            )
        return tuple(rows)

    def select_report(self, report_id: str | None) -> None:
        self.state.selected_report_id = report_id

    def selected_report(self) -> Any | None:
        summary = self.state.project_summary
        selected = self.state.selected_report_id
        if summary is None or selected is None:
            return None
        return next((r for r in summary.reports if r.report_id == selected), None)

    def describe_source(
        self,
        *,
        source_kind: str,
        package_run_id: str | None,
        review_run_id: str | None,
        settings: PiReportUiSettings | None = None,
    ) -> str:
        cfg = (settings or self.state.settings).validated()
        if not package_run_id:
            return "Select an eligible automated S9 package or published S10 review."
        if source_kind == PI_SOURCE_REVIEWED and not review_run_id:
            return "Select a published S10 review (working drafts are not eligible)."
        return source_identity_summary(
            source_kind=source_kind,
            package_run_id=package_run_id,
            review_run_id=review_run_id,
            settings=cfg,
        )

    def cannot_export_reason(
        self,
        *,
        session_root: Path | None,
        source_kind: str,
        package_dir: Path | None,
        package_run_id: str | None,
        review_run_dir: Path | None,
        review_run_id: str | None,
        review_is_draft: bool | None = None,
        settings: PiReportUiSettings | None = None,
    ) -> str | None:
        if self.state.task_running:
            return "A PI report export is already running."
        if session_root is None:
            return "Open a session before exporting a PI report."
        try:
            cfg = (settings or self.state.settings).validated()
        except ValueError as exc:
            return str(exc)
        if source_kind not in {PI_SOURCE_AUTOMATED, PI_SOURCE_REVIEWED}:
            return "Choose Automated S9 or Published S10 as the PI report source."
        if package_dir is None or package_run_id is None:
            return "Select an eligible S9 package for the PI report source."
        if source_kind == PI_SOURCE_REVIEWED:
            if review_is_draft:
                return (
                    "Working drafts cannot be used as PI report sources. "
                    "Publish the review first, or export from an automated S9 package."
                )
            if review_run_dir is None or review_run_id is None:
                return "Select a published S10 review for the reviewed PI report source."
        elif review_run_dir is not None:
            return (
                "Automated PI report must not include a review run "
                "(refusing silent merge)."
            )
        if cfg.bout_scope == "selected_bouts" and not cfg.selected_bout_ids:
            return "Select at least one eligible exclusive bout for a subset report."
        return None

    def begin_export_launch(
        self,
        *,
        session_root: Path,
        source_kind: str,
        package_dir: Path,
        package_run_id: str,
        review_run_dir: Path | None,
        review_run_id: str | None,
        review_is_draft: bool | None = None,
        settings: PiReportUiSettings | None = None,
        matched_selected_bout_count: int | None = None,
    ) -> PiReportRequestIdentity:
        """Snapshot settings + source paths only (no encode/hash on the GUI thread)."""

        reason = self.cannot_export_reason(
            session_root=session_root,
            source_kind=source_kind,
            package_dir=package_dir,
            package_run_id=package_run_id,
            review_run_dir=review_run_dir,
            review_run_id=review_run_id,
            review_is_draft=review_is_draft,
            settings=settings,
        )
        if reason is not None:
            raise PiReportUiError(reason)

        validated = (settings or self.state.settings).validated()
        if (
            validated.bout_scope == "selected_bouts"
            and matched_selected_bout_count is not None
            and matched_selected_bout_count != len(validated.selected_bout_ids)
        ):
            raise PiReportUiError(
                "One or more selected bout ids are not eligible exclusive bouts "
                f"in the chosen source ({matched_selected_bout_count} matched of "
                f"{len(validated.selected_bout_ids)} selected)."
            )

        report_id = f"{PI_REPORT_RUN_PREFIX}__{_now_timestamp()}_{uuid4().hex[:8]}"
        output_dir = Path(session_root) / RESULTS_EXPORTS_DIRNAME / report_id
        output_key = str(output_dir.resolve(strict=False))
        if output_key in self.state.reserved_output_dirs or output_dir.exists():
            raise PiReportUiError(
                f"Refusing overlapping PI report output path: {output_dir}"
            )

        identity = PiReportRequestIdentity.new(
            session_root=Path(session_root),
            source_kind=source_kind,
            package_run_id=package_run_id,
            package_dir=package_dir,
            review_run_id=review_run_id if source_kind == PI_SOURCE_REVIEWED else None,
            review_run_dir=(
                review_run_dir if source_kind == PI_SOURCE_REVIEWED else None
            ),
            report_id=report_id,
            output_dir=output_dir,
            settings=validated,
            # Digests are pinned inside publish_pi_report on the worker thread.
            pinned_source_digests={},
        )
        self.state.settings = validated
        self.state.active_identity = identity
        self.state.task_running = True
        self.state.last_error = None
        self.state.last_outcome = None
        self.state.last_result = None
        self.state.run_stage = "Launching PI report export…"
        self.state.source_summary = source_identity_summary(
            source_kind=source_kind,
            package_run_id=package_run_id,
            review_run_id=review_run_id,
            settings=validated,
        )
        self.state.output_destination = str(
            Path(session_root) / RESULTS_EXPORTS_DIRNAME
        )
        self.state.reserved_output_dirs.add(output_key)
        return identity

    def execute_export_launch(
        self,
        identity: PiReportRequestIdentity,
        progress_callback: ProgressCallback | None = None,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> PiReportResult:
        if not self._is_current_result(identity):
            raise PiReportUiError(
                "PI report launch was superseded by a newer session/source selection."
            )
        if identity.output_dir.exists():
            raise PiReportUiError(
                f"PI report output already exists (immutability): {identity.output_dir}"
            )

        def _bridge(phase: str, done: int, total: int) -> None:
            if progress_callback is None:
                return
            if phase == "encode_media":
                message = f"Encoding bout previews {done}/{total}…"
                progress_callback(
                    OperationProgress(
                        phase=phase,
                        message=message,
                        completed_units=done,
                        total_units=max(total, 1),
                        total_is_estimate=False,
                        is_indeterminate=False,
                        processed_video_time_sec=None,
                    )
                )
            else:
                progress_callback(
                    OperationProgress(
                        phase=phase,
                        message=f"PI report phase: {phase}",
                        completed_units=done,
                        total_units=max(total, 1),
                        total_is_estimate=True,
                        is_indeterminate=phase in {"resolve_source", "finalize"},
                        processed_video_time_sec=None,
                    )
                )

        self.state.run_stage = "Running PI report export…"
        return self._publish_runner(
            session_root=identity.session_root,
            source_type=identity.source_kind,
            package_dir=identity.package_dir,
            review_run_dir=identity.review_run_dir,
            settings=identity.settings.to_pi_settings(),
            query=identity.settings.to_gallery_query(),
            report_id=identity.report_id,
            progress_callback=_bridge,
            cancellation_requested=cancellation_requested,
        )

    def record_export_success(
        self,
        result: PiReportResult,
        *,
        expected_identity: PiReportRequestIdentity | None = None,
    ) -> bool:
        self.state.task_running = False
        if not self._is_current_result(expected_identity):
            self._release_output(expected_identity)
            return False
        self.state.last_result = result
        self.state.last_outcome = "completed"
        self.state.last_error = None
        self.state.run_stage = (
            f"PI report completed: {result.report_id} ({result.n_bouts} bouts)"
        )
        self.state.selected_report_id = result.report_id
        self.state.active_identity = None
        self._release_output(expected_identity)
        return True

    def record_export_failure(
        self,
        message: str,
        *,
        expected_identity: PiReportRequestIdentity | None = None,
        canceled: bool = False,
    ) -> bool:
        if expected_identity is not None:
            active = self.state.active_identity
            if active is not None and active.request_id != expected_identity.request_id:
                return False
        self.state.task_running = False
        self.state.last_result = None
        self.state.last_outcome = STATUS_CANCELED if canceled else "failed"
        self.state.last_error = message
        self.state.run_stage = "Canceled" if canceled else "Failed"
        self.state.active_identity = None
        self._release_output(expected_identity)
        return True

    def open_selected_report_folder(self) -> bool:
        report = self.selected_report()
        if report is not None:
            return bool(self._folder_opener(Path(report.report_dir)))
        if self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.report_dir))
        raise PiReportUiError("Select a completed PI report first.")

    def open_selected_report_html(self) -> bool:
        report = self.selected_report()
        if report is not None and report.has_index_html:
            return bool(self._html_opener(Path(report.report_dir) / "index.html"))
        if self.state.last_result is not None:
            return bool(self._html_opener(self.state.last_result.index_html_path))
        raise PiReportUiError("Select a completed PI report with index.html first.")

    def open_selected_report_zip(self) -> bool:
        report = self.selected_report()
        if report is not None and report.has_zip and report.zip_path is not None:
            return bool(self._folder_opener(Path(report.zip_path).parent))
        if self.state.last_result is not None and self.state.last_result.zip_path is not None:
            return bool(self._folder_opener(self.state.last_result.zip_path.parent))
        raise PiReportUiError("Selected PI report has no ZIP archive.")

    def _is_current_result(
        self, expected: PiReportRequestIdentity | None
    ) -> bool:
        if expected is None:
            return False
        active = self.state.active_identity
        if active is None:
            return False
        return expected.request_id == active.request_id

    def _release_output(self, identity: PiReportRequestIdentity | None) -> None:
        if identity is None:
            return
        self.state.reserved_output_dirs.discard(
            str(identity.output_dir.resolve(strict=False))
        )

    def _reconcile_selection(self) -> None:
        summary = self.state.project_summary
        if summary is None or not summary.reports:
            self.state.selected_report_id = None
            return
        selected = self.state.selected_report_id
        if selected and any(r.report_id == selected for r in summary.reports):
            return
        completed = summary.completed_reports
        self.state.selected_report_id = (
            completed[0].report_id if completed else summary.reports[0].report_id
        )


def _now_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _default_folder_opener(path: Path) -> bool:
    import os
    import subprocess
    import sys

    target = Path(path)
    if not target.exists():
        return False
    platform = sys.platform
    if platform.startswith("win"):
        os.startfile(str(target))
    elif platform == "darwin":
        subprocess.Popen(["open", str(target)])  # noqa: S603
    else:
        subprocess.Popen(["xdg-open", str(target)])  # noqa: S603
    return True


def _default_html_opener(path: Path) -> bool:
    import webbrowser

    target = Path(path)
    if not target.is_file():
        return False
    return bool(webbrowser.open(target.resolve().as_uri()))


__all__ = [
    "PiReportController",
    "PiReportUiError",
    "PiReportUiState",
]
