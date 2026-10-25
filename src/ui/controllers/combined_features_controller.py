"""Headless coordination for Subsystem 06 Combined Features GUI workflow.

Owns session discovery, S4/S5 source selection and joint alignment, working
settings (including optional model registry), completed-run inspection, and
dispatch of ``run_combined_feature_extraction``. Scientific work stays in
``combined_features`` — never duplicated in the GUI.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from blob_features.contracts import BlobFeaturesError
from blob_features.run_discovery import (
    BlobFeaturesRunSummary,
    summarize_blob_features_project,
)
from combined_features import (
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    MODEL_COMPAT_FILENAME,
    QC_FILENAME,
    STALE_SOURCE,
    CombinedFeaturesConfig,
    CombinedFeaturesError,
    CombinedFeaturesRequest,
    CombinedFeaturesResult,
    ModelRegistrySettings,
    build_model_compat_report,
    discover_model_registry,
    load_combined_features_config,
    load_dyadic_features_table,
    run_combined_feature_extraction,
    select_default_joint_pair,
    summarize_combined_features_project,
    validate_joint_s4_s5_pair,
)
from combined_features.run_discovery import (
    CombinedFeaturesProjectSummary,
    CombinedFeaturesRunSummary,
)
from combined_features.upstream import (
    JointAlignmentReport,
    list_eligible_joint_pairs,
)
from pose_finalization.contracts import PoseFinalizationError
from pose_finalization.run_discovery import (
    PoseFinalizationRunSummary,
    summarize_pose_finalization_project,
)
from preprocess.models import OperationProgress
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

ProgressCallback = Callable[[OperationProgress], None]
FolderOpener = Callable[[Path], bool]
CombinedFeaturesRunner = Callable[..., CombinedFeaturesResult]


class CombinedFeaturesUiError(ValueError):
    """Expected, actionable Subsystem 06 UI validation error."""


@dataclass(frozen=True, slots=True)
class CombinedFeaturesUiSettings:
    """GUI-exposed working settings snapshotted into each extraction request."""

    smooth_win: int = 9
    diff_dt: int = 1
    iteration_name: str = "suite"
    exp_name: str | None = None
    model_registry_root: Path | None = None
    include_focal: bool = True
    include_obsolete: bool = False

    @classmethod
    def from_config(cls, config: CombinedFeaturesConfig) -> CombinedFeaturesUiSettings:
        validated = config.validated()
        return cls(
            smooth_win=validated.smooth_win,
            diff_dt=validated.diff_dt,
            iteration_name=validated.iteration_name,
            exp_name=validated.exp_name,
            model_registry_root=validated.model_registry.root,
            include_focal=validated.model_registry.include_focal,
            include_obsolete=validated.model_registry.include_obsolete,
        )

    def to_config(self) -> CombinedFeaturesConfig:
        return CombinedFeaturesConfig(
            smooth_win=int(self.smooth_win),
            diff_dt=int(self.diff_dt),
            iteration_name=str(self.iteration_name),
            exp_name=self.exp_name,
            model_registry=ModelRegistrySettings(
                root=self.model_registry_root,
                include_focal=bool(self.include_focal),
                include_obsolete=bool(self.include_obsolete),
            ),
        ).validated()

    def validated(self) -> CombinedFeaturesUiSettings:
        return self.from_config(self.to_config())


@dataclass(frozen=True, slots=True)
class ExtractionRequestIdentity:
    """Identity used to reject stale asynchronous extraction results."""

    request_id: str
    session_root: Path
    s4_run_id: str
    s5_run_id: str
    s4_run_dir: Path
    s5_run_dir: Path
    settings: CombinedFeaturesUiSettings


@dataclass(frozen=True, slots=True)
class SourceRunOption:
    """One selectable upstream run for the S4 or S5 combo."""

    run_id: str
    run_dir: Path
    status: str | None
    classification: str
    label: str


@dataclass(slots=True)
class CombinedFeaturesUiState:
    """Mutable S6 workspace state (working settings vs extraction snapshots)."""

    session_root: Path | None = None
    s4_options: tuple[SourceRunOption, ...] = ()
    s5_options: tuple[SourceRunOption, ...] = ()
    joint_pair_reports: tuple[
        tuple[PoseFinalizationRunSummary, BlobFeaturesRunSummary, JointAlignmentReport],
        ...,
    ] = ()
    selected_s4_run_id: str | None = None
    selected_s5_run_id: str | None = None
    s4_selection_manual: bool = False
    s5_selection_manual: bool = False
    alignment: JointAlignmentReport | None = None
    readiness_error: str | None = None
    summary: CombinedFeaturesProjectSummary | None = None
    selected_s6_run_id: str | None = None
    working_settings: CombinedFeaturesUiSettings = field(
        default_factory=CombinedFeaturesUiSettings
    )
    last_extraction_settings: CombinedFeaturesUiSettings | None = None
    task_running: bool = False
    active_extraction_identity: ExtractionRequestIdentity | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: CombinedFeaturesResult | None = None
    model_compat_preview: Mapping[str, Any] | None = None
    unexpected_error_detail: str | None = None


class CombinedFeaturesController:
    """Session discovery, S4/S5 pair selection, settings, extraction dispatch."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], CombinedFeaturesConfig] | None = None,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        runner: CombinedFeaturesRunner = run_combined_feature_extraction,
    ) -> None:
        self.state = CombinedFeaturesUiState()
        self._config_loader = config_loader or (lambda: load_combined_features_config(None))
        self._folder_opener = folder_opener
        self._runner = runner
        self.reset_working_settings_to_defaults()

    def reset_working_settings_to_defaults(self) -> CombinedFeaturesUiSettings:
        try:
            settings = CombinedFeaturesUiSettings.from_config(self._config_loader())
        except CombinedFeaturesError as exc:
            raise CombinedFeaturesUiError(str(exc)) from exc
        self.state.working_settings = settings
        self._refresh_model_compat_preview()
        return settings

    def set_session(self, session_root: Path | str) -> CombinedFeaturesProjectSummary:
        """Open a project/session and refresh upstream + S6 discovery."""

        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise CombinedFeaturesUiError(f"Session directory does not exist: {root}")
        previous = self.state.session_root
        if (
            self.state.task_running
            and previous is not None
            and previous.resolve(strict=False) != root
        ):
            raise CombinedFeaturesUiError(
                "Cannot change S6 session while a combined-feature extraction is active."
            )
        self.state.session_root = root
        self.state.readiness_error = None
        self.state.run_error = None
        self.state.run_stage = "Not started"
        if previous is None or previous.resolve(strict=False) != root:
            self.state.last_result = None
            self.state.last_extraction_settings = None
            self.state.selected_s6_run_id = None
            self.state.selected_s4_run_id = None
            self.state.selected_s5_run_id = None
            self.state.s4_selection_manual = False
            self.state.s5_selection_manual = False
            self.state.alignment = None
        return self.refresh_discovery()

    def clear_session(self) -> None:
        if self.state.task_running:
            raise CombinedFeaturesUiError(
                "Cannot clear the S6 session while a task is active."
            )
        self.state = CombinedFeaturesUiState(working_settings=self.state.working_settings)

    def refresh_discovery(self) -> CombinedFeaturesProjectSummary:
        """Re-scan eligible S4/S5 inputs and completed S6 outputs."""

        root = self.state.session_root
        if root is None:
            raise CombinedFeaturesUiError("Select a project/session first.")
        try:
            s4_summary = summarize_pose_finalization_project(root)
            s5_summary = summarize_blob_features_project(root)
            s6_summary = summarize_combined_features_project(root)
            pairs = list_eligible_joint_pairs(root)
        except (
            CombinedFeaturesError,
            BlobFeaturesError,
            PoseFinalizationError,
            OSError,
        ) as exc:
            raise CombinedFeaturesUiError(str(exc)) from exc

        self.state.s4_options = tuple(
            _source_option_from_s4(run) for run in s4_summary.eligible_runs
        )
        self.state.s5_options = tuple(
            _source_option_from_s5(run) for run in s5_summary.eligible_runs
        )
        self.state.joint_pair_reports = pairs
        self.state.summary = s6_summary
        self._reconcile_source_selection()
        self._revalidate_alignment()
        self._reconcile_s6_selection()
        self._refresh_model_compat_preview()
        return s6_summary

    def select_s4_source(self, run_id: str | None, *, manual: bool = True) -> SourceRunOption | None:
        """Select an eligible S4 input run (manual selections are preserved)."""

        if self.state.task_running:
            raise CombinedFeaturesUiError(
                "Cannot change S4 source while extraction is running."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s4_run_id = None
            if manual:
                self.state.s4_selection_manual = True
            self._revalidate_alignment()
            return None
        match = next((opt for opt in self.state.s4_options if opt.run_id == run_id), None)
        if match is None:
            raise CombinedFeaturesUiError(
                f"Eligible S4 run is not available in the current session: {run_id}"
            )
        self.state.selected_s4_run_id = match.run_id
        if manual:
            self.state.s4_selection_manual = True
        self._revalidate_alignment()
        return match

    def select_s5_source(self, run_id: str | None, *, manual: bool = True) -> SourceRunOption | None:
        """Select an eligible S5 input run (manual selections are preserved)."""

        if self.state.task_running:
            raise CombinedFeaturesUiError(
                "Cannot change S5 source while extraction is running."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s5_run_id = None
            if manual:
                self.state.s5_selection_manual = True
            self._revalidate_alignment()
            return None
        match = next((opt for opt in self.state.s5_options if opt.run_id == run_id), None)
        if match is None:
            raise CombinedFeaturesUiError(
                f"Eligible S5 run is not available in the current session: {run_id}"
            )
        self.state.selected_s5_run_id = match.run_id
        if manual:
            self.state.s5_selection_manual = True
        self._revalidate_alignment()
        return match

    def select_s6_run(self, run_id: str | None) -> CombinedFeaturesRunSummary | None:
        """Select a discovered S6 run for QC / model-compat inspection."""

        if run_id is None or not str(run_id).strip():
            self.state.selected_s6_run_id = None
            self._refresh_model_compat_preview()
            return None
        summary = self.state.summary
        if summary is None:
            raise CombinedFeaturesUiError("Refresh discovery before selecting an S6 run.")
        match = next((run for run in summary.runs if run.run_id == run_id), None)
        if match is None:
            raise CombinedFeaturesUiError(f"S6 run was not found: {run_id}")
        self.state.selected_s6_run_id = match.run_id
        self._refresh_model_compat_preview()
        return match

    def selected_s4_source(self) -> SourceRunOption | None:
        selected = self.state.selected_s4_run_id
        if selected is None:
            return None
        return next((opt for opt in self.state.s4_options if opt.run_id == selected), None)

    def selected_s5_source(self) -> SourceRunOption | None:
        selected = self.state.selected_s5_run_id
        if selected is None:
            return None
        return next((opt for opt in self.state.s5_options if opt.run_id == selected), None)

    def selected_s6_run(self) -> CombinedFeaturesRunSummary | None:
        summary = self.state.summary
        selected = self.state.selected_s6_run_id
        if summary is None or selected is None:
            return None
        return next((run for run in summary.runs if run.run_id == selected), None)

    def update_settings(
        self, settings: CombinedFeaturesUiSettings
    ) -> CombinedFeaturesUiSettings:
        """Validate and store working settings (registry changes do not alter science)."""

        if self.state.task_running:
            raise CombinedFeaturesUiError(
                "Cannot change S6 settings while extraction is running."
            )
        validated = self.validate_settings(settings)
        self.state.working_settings = validated
        self._refresh_model_compat_preview()
        return validated

    def validate_settings(
        self, settings: CombinedFeaturesUiSettings | None = None
    ) -> CombinedFeaturesUiSettings:
        candidate = settings if settings is not None else self.state.working_settings
        try:
            return candidate.validated()
        except CombinedFeaturesError as exc:
            raise CombinedFeaturesUiError(str(exc)) from exc

    def set_model_registry_root(self, root: Path | str | None) -> CombinedFeaturesUiSettings:
        """Update only the optional model-registry path."""

        path: Path | None
        if root is None or str(root).strip() == "":
            path = None
        else:
            path = Path(root).expanduser().resolve(strict=False)
        return self.update_settings(
            replace(self.state.working_settings, model_registry_root=path)
        )

    def can_run_extraction(self) -> bool:
        return (
            not self.state.task_running
            and self.state.session_root is not None
            and self.selected_s4_source() is not None
            and self.selected_s5_source() is not None
            and self.state.alignment is not None
            and self.state.alignment.eligible
        )

    def cannot_run_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 6 extraction is already running."
        if self.state.session_root is None:
            return "Select a project/session first."
        if not self.state.s4_options:
            return (
                "No currently eligible S4 pose-finalization run is available. "
                "Complete Subsystem 4 before Combined Features."
            )
        if not self.state.s5_options:
            return (
                "No currently eligible S5 blob-features run is available. "
                "Complete Subsystem 5 before Combined Features."
            )
        if self.selected_s4_source() is None or self.selected_s5_source() is None:
            return (
                self.state.readiness_error
                or "Select eligible S4 and S5 runs that pass joint alignment."
            )
        alignment = self.state.alignment
        if alignment is None:
            return self.state.readiness_error or "Joint S4/S5 alignment has not been validated."
        if not alignment.eligible:
            code = alignment.reason_code or "alignment_failed"
            return f"Selected S4/S5 pair is incompatible ({code}): {alignment.message}"
        return None

    def alignment_status_text(self) -> str:
        """Human-readable readiness / alignment summary for the page."""

        if self.state.session_root is None:
            return "Select a project/session with eligible S4 and S5 runs."
        if not self.state.s4_options and not self.state.s5_options:
            return (
                "Missing eligible upstream results: no eligible S4 and no eligible S5 "
                "runs were found for this session."
            )
        if not self.state.s4_options:
            return (
                "Eligible S5 is present, but no currently eligible S4 pose-finalization "
                "run was found. Joint alignment cannot proceed."
            )
        if not self.state.s5_options:
            return (
                "Eligible S4 is present, but no currently eligible S5 blob-features "
                "run was found. Joint alignment cannot proceed."
            )
        s4 = self.selected_s4_source()
        s5 = self.selected_s5_source()
        if s4 is None or s5 is None:
            return (
                self.state.readiness_error
                or "Select an eligible S4 run and an eligible S5 run."
            )
        alignment = self.state.alignment
        if alignment is None:
            return self.state.readiness_error or "Alignment not yet evaluated."
        sample = (
            f"{alignment.sample_count} samples @ {alignment.target_fps:g} Hz"
            if alignment.sample_count is not None
            else "sample count unknown"
        )
        if alignment.eligible:
            return (
                f"Aligned: S4={s4.run_id} ({s4.classification}), "
                f"S5={s5.run_id} ({s5.classification}); {sample}. "
                "Ready for extraction."
            )
        code = alignment.reason_code or "alignment_failed"
        return (
            f"Incompatible: S4={s4.run_id}, S5={s5.run_id} — {code}: "
            f"{alignment.message}"
        )

    def compatible_pair_summaries(self) -> tuple[str, ...]:
        """Short labels for eligible×eligible pairs that pass joint alignment."""

        rows: list[str] = []
        for s4, s5, report in self.state.joint_pair_reports:
            if report.eligible:
                rows.append(
                    f"{s4.run_id} × {s5.run_id} "
                    f"({report.sample_count or '?'} samples)"
                )
        return tuple(rows)

    def build_request(
        self, *, timestamp: str | None = None
    ) -> tuple[CombinedFeaturesRequest, ExtractionRequestIdentity]:
        """Validate the selected pair, snapshot settings, and build a runner request."""

        reason = self.cannot_run_reason()
        if reason is not None:
            raise CombinedFeaturesUiError(reason)
        s4 = self.selected_s4_source()
        s5 = self.selected_s5_source()
        root = self.state.session_root
        if s4 is None or s5 is None or root is None:
            raise CombinedFeaturesUiError("Select compatible S4 and S5 runs first.")
        settings = self.validate_settings()
        try:
            pair = validate_joint_s4_s5_pair(
                session_root=root,
                s4_run_dir=s4.run_dir,
                s5_run_dir=s5.run_dir,
            )
        except CombinedFeaturesError as exc:
            self.state.readiness_error = str(exc)
            self.state.alignment = None
            raise CombinedFeaturesUiError(str(exc)) from exc

        config_snapshot = settings.to_config()
        request = CombinedFeaturesRequest(
            session_root=pair.session_root,
            s4_run_dir=pair.s4_run_dir,
            s5_run_dir=pair.s5_run_dir,
            config=config_snapshot,
            run_purpose="gui",
            timestamp=timestamp,
        )
        identity = ExtractionRequestIdentity(
            request_id=str(uuid4()),
            session_root=pair.session_root,
            s4_run_id=pair.s4_run_id,
            s5_run_id=pair.s5_run_id,
            s4_run_dir=pair.s4_run_dir,
            s5_run_dir=pair.s5_run_dir,
            settings=settings,
        )
        self.state.active_extraction_identity = identity
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Running combined feature extraction…"
        self.state.alignment = pair.alignment
        self.state.readiness_error = None
        return request, identity

    def execute_request(
        self,
        request: CombinedFeaturesRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> CombinedFeaturesResult:
        """Run one validated request (worker-thread safe; no Qt updates)."""

        return self._runner(request, progress_callback=progress_callback)

    def record_success(
        self,
        result: CombinedFeaturesResult,
        *,
        expected_identity: ExtractionRequestIdentity | None = None,
    ) -> bool:
        """Apply a successful result only when the request identity is still current."""

        self.state.task_running = False
        if not self._is_current_extraction_result(expected_identity):
            return False
        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.last_extraction_settings = (
            expected_identity.settings
            if expected_identity is not None
            else self.state.working_settings
        )
        self.state.selected_s6_run_id = result.run_id
        self.state.active_extraction_identity = None
        try:
            self.refresh_discovery()
            self.state.selected_s6_run_id = result.run_id
            self._refresh_model_compat_preview()
        except CombinedFeaturesUiError:
            pass
        return True

    def record_failure(
        self,
        message: str,
        *,
        expected_identity: ExtractionRequestIdentity | None = None,
    ) -> bool:
        """Record a failed run without marking it completed."""

        self.state.task_running = False
        if (
            expected_identity is not None
            and self.state.active_extraction_identity is not None
            and expected_identity.request_id
            != self.state.active_extraction_identity.request_id
        ):
            return False
        self.state.last_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message
        self.state.active_extraction_identity = None
        return True

    def open_selected_s6_folder(self) -> bool:
        run = self.selected_s6_run()
        if run is None and self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.run_dir))
        if run is None:
            raise CombinedFeaturesUiError("Select a completed S6 run first.")
        return bool(self._folder_opener(run.run_dir))

    def qc_display_rows(self) -> tuple[tuple[str, str], ...]:
        """Compact QC rows for the results panel."""

        result = self.state.last_result
        run = self.selected_s6_run()
        meta_qc: Mapping[str, Any] | None = None
        model_compat: Mapping[str, Any] | None = None
        status = None
        output = None
        classification = None
        source_s4 = None
        source_s5 = None

        use_last = result is not None and (run is None or run.run_id == result.run_id)
        if use_last and result is not None:
            status = result.status
            output = (
                str(result.dyadic_features_path)
                if result.dyadic_features_path is not None
                else None
            )
            meta_qc = result.qc
            model_compat = result.model_compat
            source_s4, source_s5 = _source_ids_from_result(result)
            if run is not None:
                classification = run.classification
                source_s4 = run.source_s4_run_id or source_s4
                source_s5 = run.source_s5_run_id or source_s5
        elif run is not None:
            status = run.status
            output = str(run.dyadic_features_path) if run.dyadic_features_path else None
            classification = run.classification
            source_s4 = run.source_s4_run_id
            source_s5 = run.source_s5_run_id
            meta_qc = _load_json_mapping(run.run_dir / QC_FILENAME)
            model_compat = _load_json_mapping(run.run_dir / MODEL_COMPAT_FILENAME)
        else:
            return ()

        rows: list[tuple[str, str]] = [
            ("Historical status", status or "—"),
            (
                "Current eligibility",
                _eligibility_label(classification) if classification else "—",
            ),
            ("Source S4", source_s4 or "—"),
            ("Source S5", source_s5 or "—"),
            ("Output", output or "—"),
        ]
        if isinstance(meta_qc, Mapping):
            metrics = meta_qc.get("metrics")
            if isinstance(metrics, Mapping):
                for key, label in (
                    ("n_samples", "120 Hz samples"),
                    ("sample_count", "120 Hz samples"),
                    ("n_rows", "Directional rows"),
                    ("row_count", "Directional rows"),
                    ("n_tracks", "Track count"),
                    ("tracks", "Track identities"),
                    ("track_ids", "Track identities"),
                ):
                    value = metrics.get(key)
                    if value is not None and not any(row[0] == label for row in rows):
                        rows.append((label, _format_metric(value)))
                implant_status = metrics.get("implantation_status")
                if implant_status is not None:
                    reason = metrics.get("implantation_reason")
                    text = str(implant_status)
                    if reason:
                        text = f"{text} ({reason})"
                    rows.append(("Implantation", text))
            implantation = meta_qc.get("implantation")
            if isinstance(implantation, Mapping) and not any(
                name == "Implantation" for name, _ in rows
            ):
                rows.append(
                    (
                        "Implantation",
                        str(
                            implantation.get("status")
                            or implantation.get("resolution")
                            or implantation
                        ),
                    )
                )
            elif meta_qc.get("implantation_status") is not None and not any(
                name == "Implantation" for name, _ in rows
            ):
                rows.append(("Implantation", str(meta_qc["implantation_status"])))
            structural = meta_qc.get("structural_ok")
            if structural is not None:
                rows.append(("Structural QC", "OK" if structural else "FAILED"))
            warnings = meta_qc.get("warnings")
            if isinstance(warnings, list) and warnings:
                rows.append(("Warnings", "; ".join(str(w) for w in warnings)))
            elif isinstance(meta_qc.get("warning_count"), int):
                rows.append(("Warnings", str(meta_qc["warning_count"])))
            failures = meta_qc.get("failures")
            if isinstance(failures, list) and failures:
                rows.append(("Failures", "; ".join(str(f) for f in failures)))
            missingness = meta_qc.get("missingness")
            if isinstance(missingness, Mapping):
                rows.append(("Missingness", _format_metric(missingness)))
            nonfinite = meta_qc.get("nonfinite_warnings") or meta_qc.get(
                "non_finite_warnings"
            )
            if isinstance(nonfinite, list) and nonfinite:
                rows.append(("Non-finite warnings", "; ".join(str(x) for x in nonfinite)))

        compat = model_compat if isinstance(model_compat, Mapping) else None
        if compat is None:
            compat = self.state.model_compat_preview
        if isinstance(compat, Mapping):
            rows.append(
                (
                    "Model registry",
                    str(compat.get("registry_status") or "—"),
                )
            )
            detectors = compat.get("detectors")
            if isinstance(detectors, list) and detectors:
                summary_bits = [
                    f"{d.get('detector_name')}={d.get('status')}"
                    for d in detectors
                    if isinstance(d, Mapping)
                ]
                if summary_bits:
                    rows.append(("Model compatibility", "; ".join(summary_bits)))
            notes = compat.get("notes")
            if isinstance(notes, list) and notes:
                rows.append(("Compat notes", "; ".join(str(n) for n in notes)))
            rows.append(
                (
                    "Inference readiness",
                    "Not established by S6 (schema/availability only; no predictions).",
                )
            )
        return tuple(rows)

    def model_compat_display_rows(self) -> tuple[tuple[str, str, str, str], ...]:
        """Rows for the model-compat table: name, family, status, detail."""

        payload = self.state.model_compat_preview
        if not isinstance(payload, Mapping):
            return ()
        detectors = payload.get("detectors")
        if not isinstance(detectors, list):
            return ()
        rows: list[tuple[str, str, str, str]] = []
        for item in detectors:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("detector_name") or "—")
            family = str(item.get("family") or "—")
            status = str(item.get("status") or "—")
            details: list[str] = []
            missing = item.get("missing_columns")
            if isinstance(missing, list) and missing:
                details.append(f"missing: {', '.join(str(x) for x in missing)}")
            dtype_warn = item.get("dtype_warnings")
            if isinstance(dtype_warn, list) and dtype_warn:
                details.append(f"dtype: {', '.join(str(x) for x in dtype_warn)}")
            present = item.get("model_file_present")
            if present is False:
                details.append("model artifact missing")
            elif present is True:
                details.append("model artifact present")
            notes = item.get("notes")
            if isinstance(notes, list) and notes:
                details.extend(str(n) for n in notes)
            details.append("inference not run")
            rows.append((name, family, status, "; ".join(details) if details else "—"))
        return tuple(rows)

    def _reconcile_source_selection(self) -> None:
        """Default to newest aligned pair unless the user chose manually."""

        s4_ids = {opt.run_id for opt in self.state.s4_options}
        s5_ids = {opt.run_id for opt in self.state.s5_options}
        if self.state.selected_s4_run_id not in s4_ids:
            self.state.selected_s4_run_id = None
            self.state.s4_selection_manual = False
        if self.state.selected_s5_run_id not in s5_ids:
            self.state.selected_s5_run_id = None
            self.state.s5_selection_manual = False

        if self.state.s4_selection_manual or self.state.s5_selection_manual:
            return
        if self.state.session_root is None:
            return
        try:
            pair = select_default_joint_pair(self.state.session_root)
        except CombinedFeaturesError as exc:
            self.state.readiness_error = str(exc)
            self.state.selected_s4_run_id = None
            self.state.selected_s5_run_id = None
            self.state.alignment = None
            return
        self.state.selected_s4_run_id = pair.s4_run_id
        self.state.selected_s5_run_id = pair.s5_run_id
        self.state.alignment = pair.alignment
        self.state.readiness_error = None

    def _revalidate_alignment(self) -> None:
        s4 = self.selected_s4_source()
        s5 = self.selected_s5_source()
        root = self.state.session_root
        if s4 is None or s5 is None or root is None:
            self.state.alignment = None
            if (
                self.state.readiness_error is None
                and root is not None
                and (not self.state.s4_options or not self.state.s5_options)
            ):
                self.state.readiness_error = self.cannot_run_reason()
            return
        # Prefer cached report from the pair matrix when available.
        for left, right, report in self.state.joint_pair_reports:
            if left.run_id == s4.run_id and right.run_id == s5.run_id:
                self.state.alignment = report
                self.state.readiness_error = None if report.eligible else report.message
                return
        try:
            pair = validate_joint_s4_s5_pair(
                session_root=root,
                s4_run_dir=s4.run_dir,
                s5_run_dir=s5.run_dir,
            )
        except CombinedFeaturesError as exc:
            self.state.alignment = None
            self.state.readiness_error = str(exc)
            return
        self.state.alignment = pair.alignment
        self.state.readiness_error = None

    def _reconcile_s6_selection(self) -> None:
        summary = self.state.summary
        if summary is None or not summary.runs:
            if self.state.last_result is None:
                self.state.selected_s6_run_id = None
            return
        current = self.state.selected_s6_run_id
        if current is not None and any(run.run_id == current for run in summary.runs):
            return
        # Prefer newest currently eligible for inspection default; never prefer stale.
        eligible = summary.eligible_runs
        if eligible:
            self.state.selected_s6_run_id = eligible[0].run_id
            return
        completed = summary.completed_runs
        self.state.selected_s6_run_id = (
            completed[0].run_id if completed else summary.runs[0].run_id
        )

    def _refresh_model_compat_preview(self) -> None:
        settings = self.state.working_settings
        try:
            registry = settings.to_config().model_registry
        except CombinedFeaturesError:
            self.state.model_compat_preview = {
                "registry_status": "unavailable",
                "notes": ["Invalid model registry settings."],
                "detectors": [],
            }
            return

        features = self._features_for_compat_preview()
        if features is not None:
            try:
                self.state.model_compat_preview = build_model_compat_report(
                    features=features,
                    settings=registry,
                )
                return
            except CombinedFeaturesError as exc:
                self.state.model_compat_preview = {
                    "registry_status": "unavailable",
                    "notes": [str(exc)],
                    "detectors": [],
                }
                return

        status, detectors, notes = discover_model_registry(registry)
        preview_detectors: list[dict[str, Any]] = []
        for detector in detectors:
            if detector.obsolete:
                preview_detectors.append(
                    {
                        "detector_name": detector.detector_name,
                        "family": detector.family,
                        "status": "unsupported",
                        "missing_columns": [],
                        "dtype_warnings": [],
                        "model_file_present": None,
                        "notes": [
                            "Obsolete detector; schema check deferred until a "
                            "feature artifact is available.",
                            "S6 does not run inference.",
                        ],
                    }
                )
            else:
                preview_detectors.append(
                    {
                        "detector_name": detector.detector_name,
                        "family": detector.family,
                        "status": "pending_features",
                        "missing_columns": [],
                        "dtype_warnings": [],
                        "model_file_present": None,
                        "notes": [
                            "Registry discovered; feature-schema compatibility "
                            "requires a completed S6 parquet.",
                            "S6 does not run inference.",
                        ],
                    }
                )
        self.state.model_compat_preview = {
            "registry_status": status,
            "notes": list(notes)
            + (
                ["No feature artifact loaded — schema compatibility pending."]
                if status != "unavailable"
                else []
            ),
            "detectors": preview_detectors,
            "inference_readiness": "not_established",
        }

    def _features_for_compat_preview(self) -> Any | None:
        run = self.selected_s6_run()
        if run is not None and run.dyadic_features_path is not None:
            path = run.dyadic_features_path
            if path.is_file():
                try:
                    return load_dyadic_features_table(path)
                except CombinedFeaturesError:
                    return None
        result = self.state.last_result
        if (
            result is not None
            and result.dyadic_features_path is not None
            and result.dyadic_features_path.is_file()
        ):
            try:
                return load_dyadic_features_table(result.dyadic_features_path)
            except CombinedFeaturesError:
                return None
        return None

    def _is_current_extraction_result(
        self, expected: ExtractionRequestIdentity | None
    ) -> bool:
        if expected is None:
            return self.state.active_extraction_identity is None
        active = self.state.active_extraction_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return (
            current_root.resolve(strict=False) == expected.session_root.resolve(strict=False)
            and self.state.selected_s4_run_id == expected.s4_run_id
            and self.state.selected_s5_run_id == expected.s5_run_id
        )


def _source_option_from_s4(run: PoseFinalizationRunSummary) -> SourceRunOption:
    return SourceRunOption(
        run_id=run.run_id,
        run_dir=run.run_dir,
        status=run.status,
        classification=run.classification,
        label=f"{run.run_id}  [{_eligibility_label(run.classification)}]",
    )


def _source_option_from_s5(run: BlobFeaturesRunSummary) -> SourceRunOption:
    return SourceRunOption(
        run_id=run.run_id,
        run_dir=run.run_dir,
        status=run.status,
        classification=run.classification,
        label=f"{run.run_id}  [{_eligibility_label(run.classification)}]",
    )


def _eligibility_label(classification: str | None) -> str:
    if classification is None:
        return "—"
    mapping = {
        ELIGIBLE_DOWNSTREAM: "eligible",
        COMPLETED_WITH_WARNINGS: "eligible (warnings)",
        STALE_SOURCE: "stale source",
        FAILED_OR_INCOMPLETE: "failed/incomplete",
        MISSING_REQUIRED_ARTIFACTS: "missing artifacts",
    }
    return mapping.get(classification, classification)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _source_ids_from_result(
    result: CombinedFeaturesResult,
) -> tuple[str | None, str | None]:
    qc = result.qc if isinstance(result.qc, Mapping) else {}
    s4 = None
    s5 = None
    if isinstance(qc, Mapping):
        alignment = qc.get("alignment")
        if isinstance(alignment, Mapping):
            s4 = alignment.get("s4_run_id") or s4
            s5 = alignment.get("s5_run_id") or s5
    meta = _load_json_mapping(result.run_meta_path)
    if isinstance(meta, Mapping):
        source_s4 = meta.get("source_s4")
        source_s5 = meta.get("source_s5")
        if isinstance(source_s4, Mapping):
            s4 = str(source_s4.get("run_id") or s4) if source_s4.get("run_id") else s4
        if isinstance(source_s5, Mapping):
            s5 = str(source_s5.get("run_id") or s5) if source_s5.get("run_id") else s5
    return s4, s5
