"""Shared QC overlay video export controller (Increment C2-C).

Owns settings, request identity, discovery, and worker execution for offline
prepared-rate QC overlays. Does not mutate scientific trees. Encoding/hashing
run only through GuiTaskRunner progressive ops via the dialog/page host.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from overlay.contracts import (
    OVERLAY_QC_EXPORT_DIRNAME,
    BehaviorSourceKind,
    OverlayExportCancelled,
)
from overlay.discovery import (
    STATUS_CANCELED,
    OverlayQcJobSummary,
    OverlayQcProjectSummary,
    summarize_overlay_qc_project,
)
from overlay.export import (
    OverlayQcExportResult,
    OverlayS4PoseQcExportResult,
    export_behavioral_qc_clip,
    export_s4_pose_qc_clip,
)
from overlay.export_s2 import OverlayS2PoseQcExportResult, export_s2_pose_qc_clip
from overlay.export_s3 import OverlayS3PoseQcExportResult, export_s3_pose_qc_clip
from overlay.export_s5 import (
    OverlayS5SegmentationQcExportResult,
    export_s5_segmentation_qc_clip,
)
from overlay.plan import (
    plan_bout_span,
    plan_prepared_interval,
    plan_s2_prepared_interval,
    plan_s3_prepared_interval,
    plan_s4_prepared_interval,
    plan_s5_prepared_interval,
)
from overlay.source import resolve_overlay_qc_source
from overlay.source_s2 import resolve_overlay_s2_qc_source
from overlay.source_s3 import resolve_overlay_s3_qc_source
from overlay.source_s4 import resolve_overlay_s4_qc_source
from overlay.source_s5 import resolve_overlay_s5_qc_source
from preprocess.models import OperationProgress
from ui.controllers.overlay_qc_export_support import (
    OverlayQcBoutOption,
    OverlayQcRequestIdentity,
    OverlayQcUiSettings,
    overlay_qc_table_rows,
    source_identity_summary,
)
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

ProgressCallback = Callable[[OperationProgress], None]
FolderOpener = Callable[[Path], bool]
FileOpener = Callable[[Path], bool]
OverlayExportResult = (
    OverlayQcExportResult
    | OverlayS4PoseQcExportResult
    | OverlayS2PoseQcExportResult
    | OverlayS3PoseQcExportResult
    | OverlayS5SegmentationQcExportResult
)


class OverlayQcExportUiError(RuntimeError):
    """Operator-facing QC overlay export GUI error."""


BOUT_COMBO_LIMIT = 250


@dataclass(slots=True)
class OverlayQcExportUiState:
    """Mutable shared QC overlay export pane/dialog state."""

    settings: OverlayQcUiSettings = field(default_factory=OverlayQcUiSettings)
    project_summary: OverlayQcProjectSummary | None = None
    selected_job_id: str | None = None
    bout_options: tuple[OverlayQcBoutOption, ...] = ()
    bout_options_truncated: bool = False
    bout_options_total_exclusive: int | None = None
    task_running: bool = False
    active_identity: OverlayQcRequestIdentity | None = None
    last_result: OverlayExportResult | None = None
    last_outcome: str | None = None  # completed | failed | canceled
    last_error: str | None = None
    run_stage: str = "Idle"
    source_summary: str = "No QC overlay source selected."
    output_destination: str = ""
    bound_session_root: Path | None = None


class OverlayQcExportController:
    """Launch, track, and discover overlay QC exports for one session."""

    def __init__(
        self,
        *,
        behavioral_exporter: Callable[..., OverlayQcExportResult] | None = None,
        s4_exporter: Callable[..., OverlayS4PoseQcExportResult] | None = None,
        s2_exporter: Callable[..., OverlayS2PoseQcExportResult] | None = None,
        s3_exporter: Callable[..., OverlayS3PoseQcExportResult] | None = None,
        s5_exporter: Callable[..., OverlayS5SegmentationQcExportResult] | None = None,
        folder_opener: FolderOpener | None = None,
        file_opener: FileOpener | None = None,
        bout_loader: Callable[[Path], tuple[OverlayQcBoutOption, ...]] | None = None,
    ) -> None:
        self.state = OverlayQcExportUiState()
        self._behavioral_exporter = behavioral_exporter or export_behavioral_qc_clip
        self._s4_exporter = s4_exporter or export_s4_pose_qc_clip
        self._s2_exporter = s2_exporter or export_s2_pose_qc_clip
        self._s3_exporter = s3_exporter or export_s3_pose_qc_clip
        self._s5_exporter = s5_exporter or export_s5_segmentation_qc_clip
        self._folder_opener = folder_opener or open_folder_in_platform_explorer
        self._file_opener = file_opener or _default_file_opener
        self._bout_loader = bout_loader or load_bout_options_from_package

    def set_session(self, session_root: Path | str | None) -> OverlayQcProjectSummary | None:
        if self.state.task_running:
            raise OverlayQcExportUiError(
                "Cannot retarget the session while a QC overlay export is running."
            )
        if session_root is None:
            self.state.bound_session_root = None
            self.state.project_summary = None
            self.state.selected_job_id = None
            self.state.bout_options = ()
            self.state.output_destination = ""
            self.state.source_summary = "No QC overlay source selected."
            return None
        root = Path(session_root).expanduser().resolve(strict=False)
        self.state.bound_session_root = root
        return self.refresh_discovery(root)

    def update_settings(self, settings: OverlayQcUiSettings) -> OverlayQcUiSettings:
        if self.state.task_running:
            raise OverlayQcExportUiError(
                "Cannot change QC overlay settings while export runs."
            )
        validated = settings.validated()
        self.state.settings = validated
        return validated

    def refresh_discovery(
        self, session_root: Path | str | None = None
    ) -> OverlayQcProjectSummary | None:
        root = session_root
        if root is None:
            root = self.state.bound_session_root
        if root is None:
            self.state.project_summary = None
            self.state.selected_job_id = None
            self.state.output_destination = ""
            return None
        resolved = Path(root).expanduser().resolve(strict=False)
        summary = summarize_overlay_qc_project(resolved)
        self.state.project_summary = summary
        self.state.output_destination = str(summary.exports_root)
        self._reconcile_selection()
        return summary

    def refresh_bout_options(self, package_dir: Path | str | None) -> tuple[OverlayQcBoutOption, ...]:
        if package_dir is None:
            self.state.bout_options = ()
            self.state.bout_options_truncated = False
            self.state.bout_options_total_exclusive = None
            return ()
        root = Path(package_dir)
        if self._bout_loader is load_bout_options_from_package:
            options, total = load_bout_options_from_package_with_total(
                root, limit=BOUT_COMBO_LIMIT
            )
        else:
            options = self._bout_loader(root)
            total = len(options)
        self.state.bout_options = options
        self.state.bout_options_total_exclusive = int(total)
        self.state.bout_options_truncated = int(total) > len(options)
        return options

    def ensure_preferred_bout_option(
        self,
        package_dir: Path | str | None,
        *,
        preferred_bout_id: str | None = None,
        preferred_start_sample_idx: int | None = None,
        preferred_end_sample_idx: int | None = None,
    ) -> OverlayQcBoutOption | None:
        """Ensure a preferred bout is selectable even when outside the first-250 list."""

        if package_dir is None:
            return None
        root = Path(package_dir)
        existing = {
            opt.bout_id: opt for opt in self.state.bout_options if opt.bout_id
        }
        if preferred_bout_id and preferred_bout_id in existing:
            return existing[preferred_bout_id]

        match = lookup_bout_option_from_package(
            root,
            bout_id=preferred_bout_id,
            start_sample_idx=preferred_start_sample_idx,
            end_sample_idx=preferred_end_sample_idx,
        )
        if match is None:
            return None
        if match.bout_id not in existing:
            self.state.bout_options = (match, *self.state.bout_options)
        return match

    def job_table_rows(self) -> tuple[tuple[str, ...], ...]:
        summary = self.state.project_summary
        rows = list(
            overlay_qc_table_rows(summary.jobs if summary is not None else ())
        )
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
                    self.state.settings.export_family,
                    str(self.state.last_outcome),
                    "—",
                    "—",
                    "—",
                    "—",
                    "no",
                    self.state.last_error,
                ),
            )
        return tuple(rows)

    def select_job(self, job_id: str | None) -> None:
        self.state.selected_job_id = job_id

    def selected_job(self) -> OverlayQcJobSummary | None:
        summary = self.state.project_summary
        selected = self.state.selected_job_id
        if summary is None or selected is None:
            return None
        return next((job for job in summary.jobs if job.job_id == selected), None)

    def describe_source(
        self,
        *,
        settings: OverlayQcUiSettings | None = None,
        package_run_id: str | None,
        review_run_id: str | None,
        s4_run_id: str | None,
        s2_run_id: str | None = None,
        s3_run_id: str | None = None,
        s5_run_id: str | None = None,
    ) -> str:
        cfg = (settings or self.state.settings).validated()
        return source_identity_summary(
            settings=cfg,
            package_run_id=package_run_id,
            review_run_id=review_run_id,
            s4_run_id=s4_run_id,
            s2_run_id=s2_run_id,
            s3_run_id=s3_run_id,
            s5_run_id=s5_run_id,
        )

    def cannot_export_reason(
        self,
        *,
        session_root: Path | None,
        settings: OverlayQcUiSettings | None = None,
        package_dir: Path | None = None,
        package_run_id: str | None = None,
        review_run_dir: Path | None = None,
        review_run_id: str | None = None,
        review_is_draft: bool | None = None,
        s4_run_dir: Path | None = None,
        s4_run_id: str | None = None,
        s2_run_dir: Path | None = None,
        s2_run_id: str | None = None,
        s3_run_dir: Path | None = None,
        s3_run_id: str | None = None,
        s5_run_dir: Path | None = None,
        s5_run_id: str | None = None,
    ) -> str | None:
        if self.state.task_running:
            return "A QC overlay export is already running."
        if session_root is None:
            return "Open a session before exporting a QC overlay video."
        bound = self.state.bound_session_root
        if bound is not None and Path(session_root).resolve(strict=False) != bound:
            return (
                "Session root changed while the QC overlay dialog was open. "
                "Close and reopen the exporter for the active session."
            )
        try:
            cfg = (settings or self.state.settings).validated()
        except ValueError as exc:
            return str(exc)

        if cfg.export_family == "s4_pose":
            if s4_run_dir is None or s4_run_id is None:
                return "Select an eligible completed S4 run (never auto-picks newest)."
            return None
        if cfg.export_family == "s2_pose":
            if s2_run_dir is None or s2_run_id is None:
                return "Select an eligible completed S2 run (never auto-picks newest)."
            return None
        if cfg.export_family == "s3_pose":
            if s3_run_dir is None or s3_run_id is None:
                return (
                    "Select an accepted S3 run (acceptance_state == accepted; "
                    "never auto-picks newest)."
                )
            return None
        if cfg.export_family == "s5_segmentation":
            if s5_run_dir is None or s5_run_id is None:
                return (
                    "Select an eligible completed S5 run "
                    "(never auto-picks newest)."
                )
            return None

        if cfg.source_kind not in {
            BehaviorSourceKind.S8_AUTOMATED.value,
            BehaviorSourceKind.S10_REVIEWED.value,
        }:
            return "Choose Automated S8 via S9 package or Published S10 as the source."
        if package_dir is None or package_run_id is None:
            return "Select an eligible S9 package for the behavioral QC source."
        if cfg.source_kind == BehaviorSourceKind.S10_REVIEWED.value:
            if review_is_draft:
                return (
                    "Working drafts cannot be used as QC overlay sources. "
                    "Publish the review first, or export from an automated S9 package."
                )
            if review_run_dir is None or review_run_id is None:
                return "Select a published S10 review for the reviewed QC source."
        elif review_run_dir is not None:
            return (
                "Automated behavioral QC must not include a review run "
                "(refusing silent merge)."
            )
        return None

    def begin_export_launch(
        self,
        *,
        session_root: Path,
        settings: OverlayQcUiSettings | None = None,
        package_dir: Path | None = None,
        package_run_id: str | None = None,
        review_run_dir: Path | None = None,
        review_run_id: str | None = None,
        review_is_draft: bool | None = None,
        s4_run_dir: Path | None = None,
        s4_run_id: str | None = None,
        s2_run_dir: Path | None = None,
        s2_run_id: str | None = None,
        s3_run_dir: Path | None = None,
        s3_run_id: str | None = None,
        s5_run_dir: Path | None = None,
        s5_run_id: str | None = None,
    ) -> OverlayQcRequestIdentity:
        """Snapshot settings + source paths only (no encode/hash on the GUI thread)."""

        reason = self.cannot_export_reason(
            session_root=session_root,
            settings=settings,
            package_dir=package_dir,
            package_run_id=package_run_id,
            review_run_dir=review_run_dir,
            review_run_id=review_run_id,
            review_is_draft=review_is_draft,
            s4_run_dir=s4_run_dir,
            s4_run_id=s4_run_id,
            s2_run_dir=s2_run_dir,
            s2_run_id=s2_run_id,
            s3_run_dir=s3_run_dir,
            s3_run_id=s3_run_id,
            s5_run_dir=s5_run_dir,
            s5_run_id=s5_run_id,
        )
        if reason is not None:
            raise OverlayQcExportUiError(reason)

        validated = (settings or self.state.settings).validated()
        identity = OverlayQcRequestIdentity.new(
            session_root=Path(session_root),
            export_family=validated.export_family,
            source_kind=(
                None
                if validated.export_family
                in {"s4_pose", "s2_pose", "s3_pose", "s5_segmentation"}
                else validated.source_kind
            ),
            package_run_id=package_run_id,
            package_dir=package_dir,
            review_run_id=(
                review_run_id
                if validated.export_family == "behavioral"
                and validated.source_kind == BehaviorSourceKind.S10_REVIEWED.value
                else None
            ),
            review_run_dir=(
                review_run_dir
                if validated.export_family == "behavioral"
                and validated.source_kind == BehaviorSourceKind.S10_REVIEWED.value
                else None
            ),
            s4_run_id=s4_run_id if validated.export_family == "s4_pose" else None,
            s4_run_dir=s4_run_dir if validated.export_family == "s4_pose" else None,
            s2_run_id=s2_run_id if validated.export_family == "s2_pose" else None,
            s2_run_dir=s2_run_dir if validated.export_family == "s2_pose" else None,
            s3_run_id=s3_run_id if validated.export_family == "s3_pose" else None,
            s3_run_dir=s3_run_dir if validated.export_family == "s3_pose" else None,
            s5_run_id=(
                s5_run_id if validated.export_family == "s5_segmentation" else None
            ),
            s5_run_dir=(
                s5_run_dir if validated.export_family == "s5_segmentation" else None
            ),
            settings=validated,
        )
        self.state.settings = validated
        self.state.active_identity = identity
        self.state.task_running = True
        self.state.last_error = None
        self.state.last_outcome = None
        self.state.last_result = None
        self.state.run_stage = "Launching QC overlay export…"
        self.state.source_summary = source_identity_summary(
            settings=validated,
            package_run_id=package_run_id,
            review_run_id=review_run_id,
            s4_run_id=s4_run_id,
            s2_run_id=s2_run_id,
            s3_run_id=s3_run_id,
            s5_run_id=s5_run_id,
        )
        self.state.output_destination = str(
            Path(session_root) / OVERLAY_QC_EXPORT_DIRNAME
        )
        return identity

    def execute_export_launch(
        self,
        identity: OverlayQcRequestIdentity,
        progress_callback: ProgressCallback | None = None,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> OverlayExportResult:
        if not self._is_current_result(identity):
            raise OverlayQcExportUiError(
                "QC overlay launch was superseded by a newer session/source selection."
            )

        def _bridge(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            stage = str(event.get("stage") or "export")
            frames_written = event.get("frames_written")
            expected = event.get("expected_frames")
            done = int(frames_written) if isinstance(frames_written, (int, float)) else 0
            total = int(expected) if isinstance(expected, (int, float)) else max(done, 1)
            if stage in {"encode_frame", "encode_start", "encode"}:
                message = f"Encoding QC frames {done}/{total}…"
                indeterminate = False
            elif stage in {"resolve_source", "open_source", "plan"}:
                message = f"QC overlay phase: {stage}"
                indeterminate = True
            else:
                message = f"QC overlay phase: {stage}"
                indeterminate = stage not in {"finalize", "publish"}
            progress_callback(
                OperationProgress(
                    phase=stage,
                    message=message,
                    completed_units=done,
                    total_units=max(total, 1),
                    total_is_estimate=indeterminate,
                    is_indeterminate=indeterminate,
                    processed_video_time_sec=None,
                )
            )

        self.state.run_stage = "Running QC overlay export…"
        settings = identity.settings
        if settings.export_family == "s4_pose":
            if identity.s4_run_dir is None:
                raise OverlayQcExportUiError("S4 pose QC launch is missing s4_run_dir.")
            s4_source = resolve_overlay_s4_qc_source(s4_run_dir=identity.s4_run_dir)
            s4_plan = plan_s4_prepared_interval(
                prepared_frame_idx_start=settings.prepared_frame_idx_start,
                prepared_frame_idx_end=settings.prepared_frame_idx_end,
                source_fps=float(s4_source.video_probe.fps),
                source_n_frames=int(s4_source.video_probe.n_frames),
                include_frame_counter=settings.include_frame_counter,
                draw_skeleton=settings.draw_skeleton,
                pre_context_sec=settings.pre_context_sec,
                post_context_sec=settings.post_context_sec,
            )
            # Always publish under the GUI-bound session, never the package
            # manifest's embedded session_root (D0 containment for isolated copies).
            output_root = Path(identity.session_root) / OVERLAY_QC_EXPORT_DIRNAME
            return self._s4_exporter(
                source=s4_source,
                plan=s4_plan,
                output_root=output_root,
                progress_callback=_bridge,
                cancellation_requested=cancellation_requested,
            )
        if settings.export_family == "s2_pose":
            if identity.s2_run_dir is None:
                raise OverlayQcExportUiError("S2 pose QC launch is missing s2_run_dir.")
            s2_source = resolve_overlay_s2_qc_source(s2_run_dir=identity.s2_run_dir)
            s2_plan = plan_s2_prepared_interval(
                prepared_frame_idx_start=settings.prepared_frame_idx_start,
                prepared_frame_idx_end=settings.prepared_frame_idx_end,
                source_fps=float(s2_source.video_probe.fps),
                source_n_frames=int(s2_source.video_probe.n_frames),
                include_frame_counter=settings.include_frame_counter,
                draw_skeleton=settings.draw_skeleton,
                show_confidence=settings.show_confidence,
                pre_context_sec=settings.pre_context_sec,
                post_context_sec=settings.post_context_sec,
            )
            output_root = Path(identity.session_root) / OVERLAY_QC_EXPORT_DIRNAME
            return self._s2_exporter(
                source=s2_source,
                plan=s2_plan,
                output_root=output_root,
                progress_callback=_bridge,
                cancellation_requested=cancellation_requested,
            )
        if settings.export_family == "s3_pose":
            if identity.s3_run_dir is None:
                raise OverlayQcExportUiError("S3 pose QC launch is missing s3_run_dir.")
            s3_source = resolve_overlay_s3_qc_source(s3_run_dir=identity.s3_run_dir)
            s3_plan = plan_s3_prepared_interval(
                prepared_frame_idx_start=settings.prepared_frame_idx_start,
                prepared_frame_idx_end=settings.prepared_frame_idx_end,
                source_fps=float(s3_source.video_probe.fps),
                source_n_frames=int(s3_source.video_probe.n_frames),
                include_frame_counter=settings.include_frame_counter,
                draw_skeleton=settings.draw_skeleton,
                show_machine_corrections=settings.show_machine_corrections,
                show_manual_corrections=settings.show_manual_corrections,
                pre_context_sec=settings.pre_context_sec,
                post_context_sec=settings.post_context_sec,
            )
            output_root = Path(identity.session_root) / OVERLAY_QC_EXPORT_DIRNAME
            return self._s3_exporter(
                source=s3_source,
                plan=s3_plan,
                output_root=output_root,
                progress_callback=_bridge,
                cancellation_requested=cancellation_requested,
            )
        if settings.export_family == "s5_segmentation":
            if identity.s5_run_dir is None:
                raise OverlayQcExportUiError(
                    "S5 segmentation QC launch is missing s5_run_dir."
                )
            s5_source = resolve_overlay_s5_qc_source(s5_run_dir=identity.s5_run_dir)
            s5_plan = plan_s5_prepared_interval(
                prepared_frame_idx_start=settings.prepared_frame_idx_start,
                prepared_frame_idx_end=settings.prepared_frame_idx_end,
                source_fps=float(s5_source.video_probe.fps),
                source_n_frames=int(s5_source.video_probe.n_frames),
                include_frame_counter=settings.include_frame_counter,
                show_contours=settings.show_contours,
                show_masks=settings.show_masks,
                show_scalar_text=settings.show_scalar_text,
                mask_opacity=settings.mask_opacity,
                pre_context_sec=settings.pre_context_sec,
                post_context_sec=settings.post_context_sec,
            )
            output_root = Path(identity.session_root) / OVERLAY_QC_EXPORT_DIRNAME
            return self._s5_exporter(
                source=s5_source,
                plan=s5_plan,
                output_root=output_root,
                progress_callback=_bridge,
                cancellation_requested=cancellation_requested,
            )

        if identity.package_dir is None or identity.source_kind is None:
            raise OverlayQcExportUiError(
                "Behavioral QC launch is missing package_dir/source_kind."
            )
        behavioral_source = resolve_overlay_qc_source(
            source_kind=identity.source_kind,
            package_dir=identity.package_dir,
            review_run_dir=identity.review_run_dir,
        )
        if settings.scope_kind == "bout_span":
            behavioral_plan = plan_bout_span(
                sync=behavioral_source.sync,
                start_sample_idx=int(settings.start_sample_idx or 0),
                end_sample_idx=int(settings.end_sample_idx or 0),
                source_fps=float(behavioral_source.video_probe.fps),
                source_n_frames=int(behavioral_source.video_probe.n_frames),
                bout_id=settings.bout_id,
                packaged_prepared_start=settings.packaged_prepared_start,
                packaged_prepared_end=settings.packaged_prepared_end,
                hud_mode=settings.hud_mode,
                include_contact=settings.include_contact,
                include_sbs=settings.include_sbs,
                include_frame_counter=settings.include_frame_counter,
                pre_context_sec=settings.pre_context_sec,
                post_context_sec=settings.post_context_sec,
            )
        else:
            behavioral_plan = plan_prepared_interval(
                prepared_frame_idx_start=settings.prepared_frame_idx_start,
                prepared_frame_idx_end=settings.prepared_frame_idx_end,
                source_fps=float(behavioral_source.video_probe.fps),
                source_n_frames=int(behavioral_source.video_probe.n_frames),
                hud_mode=settings.hud_mode,
                include_contact=settings.include_contact,
                include_sbs=settings.include_sbs,
                include_frame_counter=settings.include_frame_counter,
                pre_context_sec=settings.pre_context_sec,
                post_context_sec=settings.post_context_sec,
            )
        output_root = Path(identity.session_root) / OVERLAY_QC_EXPORT_DIRNAME
        return self._behavioral_exporter(
            source=behavioral_source,
            plan=behavioral_plan,
            output_root=output_root,
            progress_callback=_bridge,
            cancellation_requested=cancellation_requested,
        )

    def record_export_success(
        self,
        result: OverlayExportResult,
        *,
        expected_identity: OverlayQcRequestIdentity | None = None,
    ) -> bool:
        self.state.task_running = False
        if not self._is_current_result(expected_identity):
            return False
        if expected_identity is not None:
            bound = self.state.bound_session_root
            if bound is not None and expected_identity.session_root != bound:
                self.state.active_identity = None
                return False
        self.state.last_result = result
        self.state.last_outcome = "completed"
        self.state.last_error = None
        job_name = Path(result.job_dir).name
        self.state.run_stage = (
            f"QC overlay completed: {job_name} ({result.frames_written} frames)"
        )
        self.state.selected_job_id = job_name
        self.state.active_identity = None
        return True

    def record_export_failure(
        self,
        message: str,
        *,
        expected_identity: OverlayQcRequestIdentity | None = None,
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
        return True

    def open_selected_job_folder(self) -> bool:
        job = self.selected_job()
        if job is not None:
            return bool(self._folder_opener(Path(job.job_dir)))
        if self.state.last_result is not None:
            return bool(self._folder_opener(Path(self.state.last_result.job_dir)))
        raise OverlayQcExportUiError("Select a completed QC overlay export first.")

    def open_selected_video(self) -> bool:
        job = self.selected_job()
        if job is not None and job.is_completed and job.video_path is not None:
            return bool(self._file_opener(Path(job.video_path)))
        if self.state.last_result is not None:
            return bool(self._file_opener(Path(self.state.last_result.video_path)))
        raise OverlayQcExportUiError(
            "Select a completed QC overlay export with an MP4 first."
        )

    def _is_current_result(
        self, expected: OverlayQcRequestIdentity | None
    ) -> bool:
        if expected is None:
            return False
        active = self.state.active_identity
        if active is None:
            return False
        return expected.request_id == active.request_id

    def _reconcile_selection(self) -> None:
        summary = self.state.project_summary
        if summary is None or not summary.jobs:
            self.state.selected_job_id = None
            return
        selected = self.state.selected_job_id
        if selected and any(job.job_id == selected for job in summary.jobs):
            return
        completed = summary.completed_jobs
        self.state.selected_job_id = (
            completed[0].job_id if completed else summary.jobs[0].job_id
        )


def _bout_option_from_record(rec: dict[str, Any]) -> OverlayQcBoutOption | None:
    try:
        return OverlayQcBoutOption(
            bout_id=str(rec["bout_id"]),
            start_sample_idx=int(rec["start_sample_idx"]),
            end_sample_idx=int(rec["end_sample_idx"]),
            prepared_frame_idx_start=_optional_int(rec.get("prepared_frame_idx_start")),
            prepared_frame_idx_end=_optional_int(rec.get("prepared_frame_idx_end")),
            channel=None if rec.get("channel") is None else str(rec.get("channel")),
            behavior_name=(
                None
                if rec.get("behavior_name") is None
                else str(rec.get("behavior_name"))
            ),
        )
    except Exception:  # noqa: BLE001
        return None


def _load_packaged_bout_frame(
    package_dir: Path | str,
    *,
    exclusive_only: bool = True,
) -> pd.DataFrame | None:
    root = Path(package_dir)
    path = root / "behavior_bouts.parquet"
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if frame is None or frame.empty or "bout_id" not in frame.columns:
        return None
    if exclusive_only and "channel" in frame.columns:
        exclusive = frame.loc[frame["channel"].astype(str) == "exclusive"]
        if not exclusive.empty:
            frame = exclusive
    if "start_sample_idx" in frame.columns:
        frame = frame.sort_values("start_sample_idx", kind="mergesort")
    return frame


def load_bout_options_from_package_with_total(
    package_dir: Path | str,
    *,
    exclusive_only: bool = True,
    limit: int = BOUT_COMBO_LIMIT,
) -> tuple[tuple[OverlayQcBoutOption, ...], int]:
    """Load combo options plus total exclusive count (for truncation messaging)."""

    frame = _load_packaged_bout_frame(package_dir, exclusive_only=exclusive_only)
    if frame is None:
        return (), 0
    total = int(len(frame))
    options: list[OverlayQcBoutOption] = []
    for rec in frame.head(int(limit)).to_dict(orient="records"):
        opt = _bout_option_from_record(rec)
        if opt is not None:
            options.append(opt)
    return tuple(options), total


def load_bout_options_from_package(
    package_dir: Path | str,
    *,
    exclusive_only: bool = True,
    limit: int = BOUT_COMBO_LIMIT,
) -> tuple[OverlayQcBoutOption, ...]:
    """Load a short list of packaged bouts for GUI selection (read-only)."""

    options, _total = load_bout_options_from_package_with_total(
        package_dir, exclusive_only=exclusive_only, limit=limit
    )
    return options


def lookup_bout_option_from_package(
    package_dir: Path | str,
    *,
    bout_id: str | None = None,
    start_sample_idx: int | None = None,
    end_sample_idx: int | None = None,
    exclusive_only: bool = True,
) -> OverlayQcBoutOption | None:
    """Resolve one packaged bout by id or dense interval (read-only; no 250 cap)."""

    frame = _load_packaged_bout_frame(package_dir, exclusive_only=exclusive_only)
    if frame is None:
        return None
    match = None
    if bout_id:
        hits = frame.loc[frame["bout_id"].astype(str) == str(bout_id)]
        if not hits.empty:
            match = hits.iloc[0]
    if match is None and start_sample_idx is not None and end_sample_idx is not None:
        hits = frame.loc[
            (frame["start_sample_idx"].astype(int) == int(start_sample_idx))
            & (frame["end_sample_idx"].astype(int) == int(end_sample_idx))
        ]
        if not hits.empty:
            match = hits.iloc[0]
    if match is None:
        return None
    return _bout_option_from_record(match.to_dict())


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _default_file_opener(path: Path) -> bool:
    import os
    import subprocess
    import sys

    target = Path(path)
    if not target.is_file():
        return False
    platform = sys.platform
    if platform.startswith("win"):
        os.startfile(str(target))  # noqa: S606
    elif platform == "darwin":
        subprocess.Popen(["open", str(target)])  # noqa: S603
    else:
        subprocess.Popen(["xdg-open", str(target)])  # noqa: S603
    return True


# Re-export cancel type for dialogs/tests.
__all__ = [
    "BOUT_COMBO_LIMIT",
    "OverlayExportCancelled",
    "OverlayQcExportController",
    "OverlayQcExportUiError",
    "OverlayQcExportUiState",
    "load_bout_options_from_package",
    "load_bout_options_from_package_with_total",
    "lookup_bout_option_from_package",
]
