"""Subsystem 09 — Results Packaging & Export GUI page (Group 4)."""

from __future__ import annotations

import contextlib
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from results_packaging import ResultsPackageRunResult, ResultsPackagingError
from results_packaging.config import ReviewEmitV1Defaults
from results_packaging.media_csv_export import (
    CsvExportJobResult,
    MediaExportJobResult,
)
from results_packaging.media_csv_export.contracts import (
    SELECTION_ALL_PACKAGE_BOUTS,
    SELECTION_CHANNELS_BEHAVIORS,
    SELECTION_EXPORT_PROFILE,
    ExportError,
)
from ui.controllers.results_export_support import (
    ExportRequestIdentity,
    ExportUiSettings,
)
from ui.controllers.results_packaging_controller import (
    DiscoveryRequestIdentity,
    DiscoverySnapshot,
    PackagingRequestIdentity,
    ResultsPackagingController,
    ResultsPackagingUiError,
    ResultsPackagingUiSettings,
)
from ui.tasks import (
    GuiTaskContext,
    GuiTaskKind,
    GuiTaskProgressEvent,
    GuiTaskRunner,
    TaskAlreadyRunningError,
    format_pipeline_progress,
    format_task_duration,
)


class ResultsPackagingPage(QWidget):
    """Group 4 page for S9 source selection, settings, run, QC, and discovery."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: ResultsPackagingController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or ResultsPackagingController()
        self.task_runner = GuiTaskRunner(self)
        self.discovery_runner = GuiTaskRunner(self)
        self._syncing_widgets = False
        self._active_discovery_identity: DiscoveryRequestIdentity | None = None
        self._active_packaging_identity: PackagingRequestIdentity | None = None
        self._active_export_identity: ExportRequestIdentity | None = None
        self._started_at: float | None = None
        self._last_running_state = False
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    @property
    def has_running_task(self) -> bool:
        return (
            self.task_runner.is_running
            or self.discovery_runner.is_running
            or self.controller.state.task_running
            or self.controller.state.discovery_running
        )

    def set_session(self, session_root: Path | str) -> None:
        try:
            identity = self.controller.prepare_session(session_root)
        except ResultsPackagingUiError as exc:
            QMessageBox.warning(self, "Cannot open session", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S9 session opened: {Path(session_root).name}; discovering…"
        )
        self._start_discovery_task(identity)

    def refresh_from_state(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            self.session_path.setText(
                str(state.session_root) if state.session_root is not None else ""
            )
            self._populate_s8_combo()
            self._sync_settings_widgets(state.working_settings)
            self._sync_export_widgets(state.export_settings)
            self._refresh_source_panel()
            self._populate_runs_table()
            self._populate_exports_table()
            self._refresh_summary_panels()
            self.export_preview_label.setText(state.export_preview_text or "")
            if state.task_running and self._started_at is None:
                self._started_at = time.perf_counter()
                self.elapsed_timer.start()
                self.progress.setVisible(True)
                self.progress.setRange(0, 0)
                self.stage_label.setText(state.run_stage)
            elif not state.task_running and not self.task_runner.is_running:
                if state.run_error:
                    self.stage_label.setText(f"Failed: {state.run_error}")
                else:
                    self.stage_label.setText(state.run_stage)
        finally:
            self._syncing_widgets = False
        self._update_enabled()
        self._emit_running_changed()

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 9 — Results Packaging & Export")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Assemble an immutable analysis package from one currently eligible "
            "Subsystem 8 run (full bout table, sync index, ethogram summary, QC, "
            "and optional review_emit_v1 export view). Optional Export CSV / "
            "Export Bout Videos create independent jobs under results_exports/ "
            "from a currently eligible S9 package — they never mutate the package. "
            "No video playback or bout editing (S10). Cancellation is not "
            "supported; partial packaging artifacts are never published as "
            "completed. Historical export jobs remain inspectable if sources "
            "later go stale."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session and eligible S8 source")
        session_form = QFormLayout(session_group)
        self.session_path = QLineEdit()
        self.session_browse_button = QPushButton("Browse…")
        self.session_browse_button.clicked.connect(self._browse_session)
        self.open_session_button = QPushButton("Open session")
        self.open_session_button.clicked.connect(self._open_typed_session)
        session_row = QHBoxLayout()
        session_row.addWidget(self.session_path, 1)
        session_row.addWidget(self.session_browse_button)
        session_row.addWidget(self.open_session_button)
        self.s8_combo = QComboBox()
        self.s8_combo.currentIndexChanged.connect(self._s8_selection_changed)
        self.s8_status_label = QLabel("No S8 selected.")
        self.s8_status_label.setWordWrap(True)
        self.s8_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.source_status_label = QLabel(
            "Select a project/session with an S9-ready S8 run."
        )
        self.source_status_label.setWordWrap(True)
        self.refresh_button = QPushButton("Refresh discovery")
        self.refresh_button.clicked.connect(self._refresh_discovery)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("S8 run (S9 readiness assessed)", self.s8_combo)
        session_form.addRow("S8 details / provenance", self.s8_status_label)
        session_form.addRow("Readiness", self.source_status_label)
        session_form.addRow(self.refresh_button)

        settings_group = QGroupBox("Packaging configuration (frozen at run)")
        settings_layout = QVBoxLayout(settings_group)
        settings_hint = QLabel(
            "Scientific defaults: the default full package always retains every "
            "authoritative S8 bout. Optional review_emit_v1 creates an additional "
            "filtered view only — selecting it never modifies the full package. "
            "Seconds-based emission floors use the explicit 120 Hz dense timeline. "
            "Execution options below are distinguishable from scientific profile "
            "parameters."
        )
        settings_hint.setWordWrap(True)
        settings_layout.addWidget(settings_hint)

        scientific_box = QGroupBox("Scientific / profile settings")
        scientific_form = QFormLayout(scientific_box)
        self.enable_review_check = QCheckBox(
            "Generate optional review_emit_v1 export profile"
        )
        self.include_contact_check = QCheckBox("Include contact in review profile")
        self.include_engaged_check = QCheckBox("Include engaged in review profile")
        self.include_sbs_check = QCheckBox("Include side-by-side in review profile")
        self.min_frames_spin = QSpinBox()
        self.min_frames_spin.setRange(1, 10_000)
        self.min_seconds_spin = QDoubleSpinBox()
        self.min_seconds_spin.setDecimals(3)
        self.min_seconds_spin.setRange(0.0, 60.0)
        self.min_seconds_spin.setSingleStep(0.01)
        self.target_fps_label = QLabel("120 Hz (fixed; dense timeline)")
        scientific_form.addRow(self.enable_review_check)
        scientific_form.addRow(self.include_contact_check)
        scientific_form.addRow(self.include_engaged_check)
        scientific_form.addRow(self.include_sbs_check)
        scientific_form.addRow("min_emit_bout_frames", self.min_frames_spin)
        scientific_form.addRow("min_emit_bout_seconds", self.min_seconds_spin)
        scientific_form.addRow("target_fps for seconds floor", self.target_fps_label)
        settings_layout.addWidget(scientific_box)

        execution_box = QGroupBox("Execution / provenance options")
        execution_form = QFormLayout(execution_box)
        self.require_video_check = QCheckBox("Require prepared video reference")
        self.hash_video_check = QCheckBox(
            "Full prepared-video SHA256 verification (optional; expensive)"
        )
        execution_form.addRow(self.require_video_check)
        execution_form.addRow(self.hash_video_check)
        package_mode_note = QLabel(
            "Package mode is fixed to reference_lightweight for MVP."
        )
        package_mode_note.setWordWrap(True)
        execution_form.addRow(package_mode_note)
        settings_layout.addWidget(execution_box)

        self.effective_settings_label = QLabel("")
        self.effective_settings_label.setWordWrap(True)
        self.effective_settings_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        settings_layout.addWidget(self.effective_settings_label)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Packaging")
        self.run_button.clicked.connect(self._run_packaging)
        self.open_output_button = QPushButton("Open S9 package folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.apply_settings_button = QPushButton("Apply settings")
        self.apply_settings_button.clicked.connect(self._apply_settings)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.open_output_button)
        run_row.addWidget(self.apply_settings_button)
        run_row.addStretch(1)

        status_group = QGroupBox("Processing status")
        status_layout = QVBoxLayout(status_group)
        self.stage_label = QLabel("Not started")
        self.stage_label.setWordWrap(True)
        self.elapsed_label = QLabel("Elapsed: —")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.elapsed_label)
        status_layout.addWidget(self.progress)

        summary_group = QGroupBox("Package summary (full package vs optional export)")
        summary_layout = QVBoxLayout(summary_group)
        summary_note = QLabel(
            "Authoritative bout counts describe the full package. Optional "
            "profile omissions do not delete data from the package."
        )
        summary_note.setWordWrap(True)
        summary_layout.addWidget(summary_note)
        self.summary_table = self._metric_table(max_height=320)
        summary_layout.addWidget(self.summary_table)

        qc_group = QGroupBox("QC, warnings, information, and dependencies")
        qc_layout = QVBoxLayout(qc_group)
        qc_note = QLabel(
            "Structural errors block usability. Warnings need attention. "
            "Information covers expected configuration choices, optional profile "
            "omissions, normal sync residuals, and reference-based packaging."
        )
        qc_note.setWordWrap(True)
        qc_layout.addWidget(qc_note)
        self.qc_table = self._metric_table(max_height=320)
        qc_layout.addWidget(self.qc_table)

        runs_group = QGroupBox("Completed S9 packages (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(
            (
                "Run ID",
                "Status",
                "Current eligibility",
                "Source S8",
                "Warnings",
                "Completed",
                "Output",
            )
        )
        self.runs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.verticalHeader().setVisible(False)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.runs_table.itemSelectionChanged.connect(self._s9_selection_changed)
        runs_layout.addWidget(self.runs_table)

        export_group = QGroupBox(
            "Optional exports (from selected eligible S9 package)"
        )
        export_layout = QVBoxLayout(export_group)
        export_hint = QLabel(
            "Export CSV and Export Bout Videos require a currently eligible "
            "completed S9 package. Stale packages stay listed for inspection; "
            "archival export from stale packages is deferred. Exports write "
            "only under results_exports/."
        )
        export_hint.setWordWrap(True)
        export_layout.addWidget(export_hint)
        export_form = QFormLayout()
        self.export_selection_combo = QComboBox()
        self.export_selection_combo.addItem("All package bouts", SELECTION_ALL_PACKAGE_BOUTS)
        self.export_selection_combo.addItem(
            "Channels / behaviors", SELECTION_CHANNELS_BEHAVIORS
        )
        self.export_selection_combo.addItem(
            "Existing export profile", SELECTION_EXPORT_PROFILE
        )
        self.export_selection_combo.currentIndexChanged.connect(
            self._export_settings_changed
        )
        self.export_channels_edit = QLineEdit()
        self.export_channels_edit.setPlaceholderText("e.g. contact,exclusive")
        self.export_channels_edit.editingFinished.connect(self._export_settings_changed)
        self.export_behaviors_edit = QLineEdit()
        self.export_behaviors_edit.setPlaceholderText("e.g. approach,contact")
        self.export_behaviors_edit.editingFinished.connect(self._export_settings_changed)
        self.export_profile_combo = QComboBox()
        self.export_profile_combo.currentIndexChanged.connect(
            self._export_settings_changed
        )
        self.export_pre_context_spin = QDoubleSpinBox()
        self.export_pre_context_spin.setDecimals(3)
        self.export_pre_context_spin.setRange(0.0, 60.0)
        self.export_pre_context_spin.setSingleStep(0.05)
        self.export_pre_context_spin.valueChanged.connect(self._export_settings_changed)
        self.export_post_context_spin = QDoubleSpinBox()
        self.export_post_context_spin.setDecimals(3)
        self.export_post_context_spin.setRange(0.0, 60.0)
        self.export_post_context_spin.setSingleStep(0.05)
        self.export_post_context_spin.valueChanged.connect(self._export_settings_changed)
        export_form.addRow("Selection mode", self.export_selection_combo)
        export_form.addRow("Channels (comma-separated)", self.export_channels_edit)
        export_form.addRow("Behaviors (comma-separated)", self.export_behaviors_edit)
        export_form.addRow("Packaged profile", self.export_profile_combo)
        export_form.addRow("pre_context_sec", self.export_pre_context_spin)
        export_form.addRow("post_context_sec", self.export_post_context_spin)
        export_layout.addLayout(export_form)
        self.export_preview_label = QLabel("")
        self.export_preview_label.setWordWrap(True)
        self.export_preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        export_layout.addWidget(self.export_preview_label)
        export_buttons = QHBoxLayout()
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self._run_csv_export)
        self.export_media_button = QPushButton("Export Bout Videos")
        self.export_media_button.clicked.connect(self._run_media_export)
        self.retry_media_button = QPushButton("Retry incomplete media job")
        self.retry_media_button.clicked.connect(self._retry_media_export)
        self.open_export_button = QPushButton("Open export folder")
        self.open_export_button.clicked.connect(self._open_export_folder)
        export_buttons.addWidget(self.export_csv_button)
        export_buttons.addWidget(self.export_media_button)
        export_buttons.addWidget(self.retry_media_button)
        export_buttons.addWidget(self.open_export_button)
        export_buttons.addStretch(1)
        export_layout.addLayout(export_buttons)

        exports_group = QGroupBox("Export jobs (newest first)")
        exports_layout = QVBoxLayout(exports_group)
        self.exports_table = QTableWidget(0, 8)
        self.exports_table.setHorizontalHeaderLabels(
            (
                "Job ID",
                "Source S9 package",
                "Type",
                "Status",
                "Current eligibility",
                "Counts",
                "Failure",
                "Output",
            )
        )
        self.exports_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.exports_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.exports_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.exports_table.verticalHeader().setVisible(False)
        exports_header = self.exports_table.horizontalHeader()
        exports_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        exports_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.exports_table.itemSelectionChanged.connect(self._export_job_selection_changed)
        exports_layout.addWidget(self.exports_table)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(session_group)
        layout.addWidget(settings_group)
        layout.addLayout(run_row)
        layout.addWidget(status_group)
        layout.addWidget(summary_group)
        layout.addWidget(qc_group)
        layout.addWidget(runs_group)
        layout.addWidget(export_group)
        layout.addWidget(exports_group, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _metric_table(self, *, max_height: int = 220) -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(("Metric", "Value"))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setMaximumHeight(max_height)
        return table

    def _browse_session(self) -> None:
        if self.controller.state.task_running or self.task_runner.is_running:
            self.status_message.emit(
                "Cannot change session while results packaging is active."
            )
            return
        start = self.session_path.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if not chosen:
            return
        self.set_session(chosen)

    def _open_typed_session(self) -> None:
        if self.controller.state.task_running or self.task_runner.is_running:
            self.status_message.emit(
                "Cannot change session while results packaging is active."
            )
            return
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _refresh_discovery(self) -> None:
        if not self._guard_idle("refresh discovery"):
            return
        if self.controller.state.session_root is None:
            QMessageBox.information(
                self, "Session required", "Open a project/session first."
            )
            return
        try:
            identity = self.controller.begin_discovery()
        except ResultsPackagingUiError as exc:
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("S9 discovery started in background…")
        self._start_discovery_task(identity)

    def _start_discovery_task(self, identity: DiscoveryRequestIdentity) -> None:
        self._active_discovery_identity = identity
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.stage_label.setText("Discovering S8/S9 artifacts…")
        self._update_enabled()
        self._emit_running_changed()
        if self.discovery_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.RESULTS_PACKAGING_DISCOVERY,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> DiscoverySnapshot:
            return self.controller.collect_discovery_snapshot(identity)

        try:
            self.discovery_runner.start(
                _operation,
                task_name="S9 discovery",
                context=context,
                on_success=lambda result: self._on_discovery_success(result, identity),
                on_error=lambda exc: self._on_discovery_error(exc, identity),
                on_finished=lambda: self._on_discovery_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_discovery_success(
        self, result: object, identity: DiscoveryRequestIdentity
    ) -> None:
        if not isinstance(result, DiscoverySnapshot):
            self.unexpected_error.emit(
                f"Unexpected S9 discovery result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_discovery_snapshot(result)
        if not accepted:
            self.status_message.emit(
                "Ignored stale S9 discovery result after session change."
            )
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S9 discovery complete for {identity.session_root.name}."
        )

    def _on_discovery_error(
        self, exc: BaseException, identity: DiscoveryRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.record_discovery_failure(
            message, expected_identity=identity
        )
        if not accepted:
            return
        self.refresh_from_state()
        if isinstance(exc, ResultsPackagingUiError):
            QMessageBox.warning(self, "Discovery failed", message)
            self.status_message.emit(f"S9 discovery failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_discovery_finished(self, identity: DiscoveryRequestIdentity) -> None:
        latest = self.controller.state.active_discovery_identity
        if (
            self.controller.state.discovery_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            self._start_discovery_task(latest)
            return
        if (
            self._active_discovery_identity is not None
            and identity.request_id == self._active_discovery_identity.request_id
        ):
            self._active_discovery_identity = None
        if not self.controller.state.task_running and not self.discovery_runner.is_running:
            self.progress.setVisible(False)
            if self.controller.state.run_stage == "Not started":
                self.stage_label.setText(self.controller.state.run_stage)
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _apply_settings(self) -> None:
        if not self._guard_idle("apply settings"):
            return
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
        except ResultsPackagingUiError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            "S9 settings applied (immutable snapshot taken at run time)."
        )

    def _populate_s8_combo(self) -> None:
        self.s8_combo.blockSignals(True)
        self.s8_combo.clear()
        self.s8_combo.addItem("(none)", None)
        selected = self.controller.state.selected_s8_run_id
        index = 0
        for i, opt in enumerate(self.controller.state.s8_options, start=1):
            self.s8_combo.addItem(opt.label, opt.run_id)
            if selected is not None and opt.run_id == selected:
                index = i
        self.s8_combo.setCurrentIndex(index)
        self.s8_combo.blockSignals(False)

    def _refresh_source_panel(self) -> None:
        self.source_status_label.setText(self.controller.source_status_text())
        source = self.controller.selected_s8_source()
        if source is None:
            self.s8_status_label.setText("No S8 selected.")
        else:
            reasons = "; ".join(source.reasons) if source.reasons else "—"
            warnings = "; ".join(source.warnings) if source.warnings else "—"
            self.s8_status_label.setText(
                f"Historical status={source.historical_status or '—'}; "
                f"S8 classification={source.s8_classification}; "
                f"S9-ready={source.s9_ready}; "
                f"bouts={source.n_bouts if source.n_bouts is not None else '—'}; "
                f"dense={source.n_dense_samples if source.n_dense_samples is not None else '—'}; "
                f"S7={source.linked_s7_run_id or '—'}; "
                f"S6={source.linked_s6_run_id or '—'}; "
                f"prepared_sync={source.prepared_sync_path or '—'}; "
                f"prepared_video={source.prepared_video_path or '—'}; "
                f"reasons={reasons}; warnings={warnings}"
            )
        effective = self.controller.settings_display_rows()
        self.effective_settings_label.setText(
            "Effective highlights: "
            + "; ".join(f"{k}={v}" for k, v in effective)
        )

    def _populate_runs_table(self) -> None:
        summary = self.controller.state.summary
        runs = list(summary.runs) if summary is not None else []
        self.runs_table.blockSignals(True)
        self.runs_table.setRowCount(len(runs))
        selected = self.controller.state.selected_s9_run_id
        select_row = -1
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.status or "—",
                run.classification,
                run.source_s8_run_id or "—",
                str(run.warning_count),
                run.completed_at or "—",
                str(run.run_dir),
            )
            for col, value in enumerate(values):
                self.runs_table.setItem(row, col, QTableWidgetItem(value))
            if selected is not None and run.run_id == selected:
                select_row = row
        if select_row >= 0:
            self.runs_table.selectRow(select_row)
        self.runs_table.blockSignals(False)

    def _refresh_summary_panels(self) -> None:
        self._fill_metric_table(
            self.summary_table, self.controller.package_summary_rows()
        )
        self._fill_metric_table(self.qc_table, self.controller.qc_display_rows())

    def _fill_metric_table(
        self, table: QTableWidget, rows: tuple[tuple[str, str], ...]
    ) -> None:
        table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            table.setItem(index, 0, QTableWidgetItem(name))
            table.setItem(index, 1, QTableWidgetItem(value))

    def _sync_settings_widgets(self, settings: ResultsPackagingUiSettings) -> None:
        review = settings.review_emit_v1
        for widget, checked in (
            (self.enable_review_check, settings.enable_review_emit_v1),
            (self.include_contact_check, review.include_contact),
            (self.include_engaged_check, review.include_engaged),
            (self.include_sbs_check, review.include_side_by_side),
            (self.require_video_check, settings.require_prepared_video),
            (self.hash_video_check, settings.hash_prepared_video),
        ):
            widget.blockSignals(True)
            widget.setChecked(checked)
            widget.blockSignals(False)
        self.min_frames_spin.blockSignals(True)
        self.min_frames_spin.setValue(int(review.min_emit_bout_frames))
        self.min_frames_spin.blockSignals(False)
        self.min_seconds_spin.blockSignals(True)
        self.min_seconds_spin.setValue(float(review.min_emit_bout_seconds))
        self.min_seconds_spin.blockSignals(False)

    def _read_settings_from_widgets(self) -> ResultsPackagingUiSettings:
        base = self.controller.state.working_settings
        review = ReviewEmitV1Defaults(
            include_contact=self.include_contact_check.isChecked(),
            include_engaged=self.include_engaged_check.isChecked(),
            include_side_by_side=self.include_sbs_check.isChecked(),
            min_emit_bout_frames=int(self.min_frames_spin.value()),
            min_emit_bout_seconds=float(self.min_seconds_spin.value()),
            target_fps=120.0,
        )
        return replace(
            base,
            enable_review_emit_v1=self.enable_review_check.isChecked(),
            review_emit_v1=review,
            require_prepared_video=self.require_video_check.isChecked(),
            hash_prepared_video=self.hash_video_check.isChecked(),
        ).validated()

    def _s8_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        if not self._guard_idle("change S8 source"):
            self.refresh_from_state()
            return
        run_id = self.s8_combo.currentData()
        try:
            self.controller.select_s8_source(
                str(run_id) if run_id is not None else None, manual=True
            )
        except ResultsPackagingUiError as exc:
            QMessageBox.warning(self, "Cannot select S8", str(exc))
            self.status_message.emit(str(exc))
        self.refresh_from_state()

    def _s9_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        row = self.runs_table.currentRow()
        if row < 0:
            return
        item = self.runs_table.item(row, 0)
        if item is None:
            return
        try:
            self.controller.select_s9_run(item.text())
        except ResultsPackagingUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_summary_panels()
        self._update_enabled()

    def _run_packaging(self) -> None:
        if not self._guard_idle("start results packaging"):
            return
        self.run_button.setEnabled(False)
        self.stage_label.setText("Launching packaging…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_message.emit("S9 packaging request accepted; launching…")
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
            identity = self.controller.begin_packaging_launch(settings)
        except ResultsPackagingUiError as exc:
            self.progress.setVisible(False)
            QMessageBox.warning(self, "Cannot run S9 packaging", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return

        self._active_packaging_identity = identity
        self._started_at = time.perf_counter()
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        self.status_message.emit(
            "S9 packaging started (settings snapshotted; cancellation unsupported)."
        )

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.RESULTS_PACKAGING,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )

        def _operation(report: object) -> ResultsPackageRunResult:
            return self.controller.execute_packaging_launch(
                identity, progress_callback=report  # type: ignore[arg-type]
            )

        try:
            self.task_runner.start_progressive(
                _operation,
                task_name="S9 results packaging",
                context=context,
                on_progress=lambda event: self._on_packaging_progress(event, identity),
                on_success=lambda result: self._on_packaging_success(result, identity),
                on_error=lambda exc: self._on_packaging_error(exc, identity),
                on_finished=lambda: self._on_packaging_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.controller.record_failure(
                "Packaging already running.", expected_identity=identity
            )
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A Subsystem 9 packaging task is already active.",
            )

    def _on_packaging_progress(
        self,
        event: object,
        identity: PackagingRequestIdentity,
    ) -> None:
        if self._active_packaging_identity is None:
            return
        if identity.request_id != self._active_packaging_identity.request_id:
            return
        if not isinstance(event, GuiTaskProgressEvent):
            return
        self.stage_label.setText(format_pipeline_progress(event))
        if (
            not event.is_indeterminate
            and event.total_units is not None
            and event.completed_units is not None
        ):
            self.progress.setRange(0, int(event.total_units))
            self.progress.setValue(int(event.completed_units))
        else:
            self.progress.setRange(0, 0)

    def _on_packaging_success(
        self, result: object, identity: PackagingRequestIdentity
    ) -> None:
        if not isinstance(result, ResultsPackageRunResult):
            self.unexpected_error.emit(f"Unexpected S9 result type: {type(result)!r}")
            return
        accepted = self.controller.record_success(
            result, expected_identity=identity, refresh_discovery=False
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale S9 packaging result after session/source change."
            )
            return
        self.stage_label.setText(f"Finished: {result.status}")
        self.refresh_from_state()
        warnings = f" ({len(result.warnings)} warnings)" if result.warnings else ""
        self.status_message.emit(
            f"S9 {result.status}{warnings}. Output: {result.run_dir}"
        )
        if self.controller.state.session_root is not None:
            with contextlib.suppress(ResultsPackagingUiError):
                discovery_identity = self.controller.begin_discovery()
                self.controller.state.selected_s9_run_id = result.run_id
                self._start_discovery_task(discovery_identity)

    def _on_packaging_error(
        self, exc: BaseException, identity: PackagingRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.record_failure(
            message, expected_identity=identity
        )
        if not accepted:
            return
        self.stage_label.setText(f"Failed: {message}")
        self.refresh_from_state()
        if self.controller.state.session_root is not None:
            with contextlib.suppress(ResultsPackagingUiError):
                discovery_identity = self.controller.begin_discovery()
                self._start_discovery_task(discovery_identity)
        if isinstance(exc, (ResultsPackagingUiError, ResultsPackagingError)):
            QMessageBox.warning(self, "Results packaging failed", message)
            self.status_message.emit(f"S9 failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_packaging_finished(self, identity: PackagingRequestIdentity) -> None:
        if (
            self._active_packaging_identity is not None
            and identity.request_id == self._active_packaging_identity.request_id
        ):
            self._active_packaging_identity = None
        self.controller.state.task_running = False
        self.elapsed_timer.stop()
        self._started_at = None
        if not (
            self.controller.state.discovery_running or self.discovery_runner.is_running
        ):
            self.progress.setVisible(False)
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_output_folder(self) -> None:
        try:
            opened = self.controller.open_selected_s9_folder()
        except ResultsPackagingUiError as exc:
            QMessageBox.information(self, "No S9 output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the S9 package folder.")
        else:
            self.status_message.emit("Opened S9 package folder.")

    def _sync_export_widgets(self, settings: ExportUiSettings) -> None:
        mode_index = self.export_selection_combo.findData(settings.selection_mode)
        self.export_selection_combo.blockSignals(True)
        self.export_selection_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self.export_selection_combo.blockSignals(False)
        for widget, text in (
            (self.export_channels_edit, settings.channels_text),
            (self.export_behaviors_edit, settings.behaviors_text),
        ):
            widget.blockSignals(True)
            widget.setText(text)
            widget.blockSignals(False)
        profiles = self.controller.available_export_profiles()
        self.export_profile_combo.blockSignals(True)
        self.export_profile_combo.clear()
        self.export_profile_combo.addItem("(none)", None)
        selected_profile = 0
        for i, profile_id in enumerate(profiles, start=1):
            self.export_profile_combo.addItem(profile_id, profile_id)
            if settings.profile_id == profile_id:
                selected_profile = i
        self.export_profile_combo.setCurrentIndex(selected_profile)
        self.export_profile_combo.blockSignals(False)
        self.export_pre_context_spin.blockSignals(True)
        self.export_pre_context_spin.setValue(float(settings.pre_context_sec))
        self.export_pre_context_spin.blockSignals(False)
        self.export_post_context_spin.blockSignals(True)
        self.export_post_context_spin.setValue(float(settings.post_context_sec))
        self.export_post_context_spin.blockSignals(False)

    def _read_export_settings_from_widgets(self) -> ExportUiSettings:
        return ExportUiSettings(
            selection_mode=str(
                self.export_selection_combo.currentData() or SELECTION_ALL_PACKAGE_BOUTS
            ),
            channels_text=self.export_channels_edit.text(),
            behaviors_text=self.export_behaviors_edit.text(),
            profile_id=(
                str(self.export_profile_combo.currentData())
                if self.export_profile_combo.currentData() is not None
                else None
            ),
            pre_context_sec=float(self.export_pre_context_spin.value()),
            post_context_sec=float(self.export_post_context_spin.value()),
        )

    def _export_settings_changed(self) -> None:
        if self._syncing_widgets:
            return
        if not self._guard_idle("change export settings"):
            self.refresh_from_state()
            return
        try:
            settings = self._read_export_settings_from_widgets()
            self.controller.update_export_settings(settings)
        except ResultsPackagingUiError as exc:
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.export_preview_label.setText(self.controller.state.export_preview_text)
        self._update_enabled()

    def _populate_exports_table(self) -> None:
        rows = self.controller.export_jobs_table_rows()
        self.exports_table.blockSignals(True)
        self.exports_table.setRowCount(len(rows))
        selected = self.controller.state.selected_export_job_id
        select_row = -1
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                self.exports_table.setItem(row, col, QTableWidgetItem(value))
            if selected is not None and values[0] == selected:
                select_row = row
        if select_row >= 0:
            self.exports_table.selectRow(select_row)
        self.exports_table.blockSignals(False)

    def _export_job_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        row = self.exports_table.currentRow()
        if row < 0:
            return
        item = self.exports_table.item(row, 0)
        if item is None:
            return
        try:
            self.controller.select_export_job(item.text())
        except ResultsPackagingUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._update_enabled()

    def _run_csv_export(self) -> None:
        if not self._guard_idle("start CSV export"):
            return
        self.export_csv_button.setEnabled(False)
        self.stage_label.setText("Launching CSV export…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_message.emit("CSV export request accepted; launching…")
        try:
            settings = self._read_export_settings_from_widgets()
            self.controller.update_export_settings(settings)
            identity = self.controller.begin_csv_export_launch(settings)
        except ResultsPackagingUiError as exc:
            self.progress.setVisible(False)
            QMessageBox.warning(self, "Cannot export CSV", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._start_export_task(
            identity,
            task_kind=GuiTaskKind.RESULTS_CSV_EXPORT,
            task_name="S9 CSV export",
            operation=lambda report: self.controller.execute_csv_export_launch(
                identity, progress_callback=report
            ),
            on_success=self._on_csv_export_success,
        )

    def _run_media_export(self) -> None:
        if not self._guard_idle("start media export"):
            return
        self.export_media_button.setEnabled(False)
        self.stage_label.setText("Launching media export…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_message.emit("Media export request accepted; launching…")
        try:
            settings = self._read_export_settings_from_widgets()
            self.controller.update_export_settings(settings)
            identity = self.controller.begin_media_export_launch(settings)
        except ResultsPackagingUiError as exc:
            self.progress.setVisible(False)
            QMessageBox.warning(self, "Cannot export bout videos", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._start_export_task(
            identity,
            task_kind=GuiTaskKind.RESULTS_MEDIA_EXPORT,
            task_name="S9 media export",
            operation=lambda report: self.controller.execute_media_export_launch(
                identity, progress_callback=report
            ),
            on_success=self._on_media_export_success,
        )

    def _retry_media_export(self) -> None:
        if not self._guard_idle("retry media export"):
            return
        self.retry_media_button.setEnabled(False)
        self.stage_label.setText("Launching media export retry…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_message.emit("Media export retry accepted; launching…")
        try:
            identity = self.controller.begin_media_retry_launch()
        except ResultsPackagingUiError as exc:
            self.progress.setVisible(False)
            QMessageBox.warning(self, "Cannot retry media export", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._start_export_task(
            identity,
            task_kind=GuiTaskKind.RESULTS_MEDIA_EXPORT,
            task_name="S9 media export retry",
            operation=lambda report: self.controller.execute_media_retry_launch(
                identity, progress_callback=report
            ),
            on_success=self._on_media_export_success,
        )

    def _start_export_task(
        self,
        identity: ExportRequestIdentity,
        *,
        task_kind: GuiTaskKind,
        task_name: str,
        operation: object,
        on_success: object,
    ) -> None:
        self._active_export_identity = identity
        self._active_packaging_identity = None
        self._started_at = time.perf_counter()
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=task_kind,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )
        try:
            self.task_runner.start_progressive(
                operation,  # type: ignore[arg-type]
                task_name=task_name,
                context=context,
                on_progress=lambda event: self._on_export_progress(event, identity),
                on_success=lambda result: on_success(result, identity),  # type: ignore[operator]
                on_error=lambda exc: self._on_export_error(exc, identity),
                on_finished=lambda: self._on_export_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.controller.record_export_failure(
                "Export already running.", expected_identity=identity
            )
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A Subsystem 9 packaging/export task is already active.",
            )

    def _on_export_progress(
        self, event: object, identity: ExportRequestIdentity
    ) -> None:
        if self._active_export_identity is None:
            return
        if identity.request_id != self._active_export_identity.request_id:
            return
        if not isinstance(event, GuiTaskProgressEvent):
            return
        self.stage_label.setText(format_pipeline_progress(event))
        if (
            not event.is_indeterminate
            and event.total_units is not None
            and event.completed_units is not None
        ):
            self.progress.setRange(0, int(event.total_units))
            self.progress.setValue(int(event.completed_units))
        else:
            self.progress.setRange(0, 0)

    def _on_csv_export_success(
        self, result: object, identity: ExportRequestIdentity
    ) -> None:
        if not isinstance(result, CsvExportJobResult):
            self.unexpected_error.emit(f"Unexpected CSV export result: {type(result)!r}")
            return
        accepted = self.controller.record_csv_export_success(
            result, expected_identity=identity
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale CSV export result after session/package change."
            )
            return
        self.stage_label.setText(f"CSV export finished: {result.status}")
        self.refresh_from_state()
        self.status_message.emit(
            f"CSV export {result.status}; rows={result.n_rows}. Output: {result.job_dir}"
        )
        if self.controller.state.session_root is not None:
            with contextlib.suppress(ResultsPackagingUiError):
                discovery_identity = self.controller.begin_discovery()
                self._start_discovery_task(discovery_identity)

    def _on_media_export_success(
        self, result: object, identity: ExportRequestIdentity
    ) -> None:
        if not isinstance(result, MediaExportJobResult):
            self.unexpected_error.emit(
                f"Unexpected media export result: {type(result)!r}"
            )
            return
        accepted = self.controller.record_media_export_success(
            result, expected_identity=identity
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale media export result after session/package change."
            )
            return
        self.stage_label.setText(f"Media export finished: {result.status}")
        self.refresh_from_state()
        self.status_message.emit(
            f"Media export {result.status}; "
            f"ok={result.n_succeeded} fail={result.n_failed}. "
            f"Output: {result.job_dir}"
        )
        if self.controller.state.session_root is not None:
            with contextlib.suppress(ResultsPackagingUiError):
                discovery_identity = self.controller.begin_discovery()
                self._start_discovery_task(discovery_identity)

    def _on_export_error(
        self, exc: BaseException, identity: ExportRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.record_export_failure(
            message, expected_identity=identity
        )
        if not accepted:
            return
        self.stage_label.setText(f"Failed: {message}")
        self.refresh_from_state()
        if self.controller.state.session_root is not None:
            with contextlib.suppress(ResultsPackagingUiError):
                discovery_identity = self.controller.begin_discovery()
                self._start_discovery_task(discovery_identity)
        if isinstance(
            exc, (ResultsPackagingUiError, ResultsPackagingError, ExportError)
        ):
            QMessageBox.warning(self, "Export failed", message)
            self.status_message.emit(f"S9 export failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_export_finished(self, identity: ExportRequestIdentity) -> None:
        if (
            self._active_export_identity is not None
            and identity.request_id == self._active_export_identity.request_id
        ):
            self._active_export_identity = None
        self.controller.state.task_running = False
        self.elapsed_timer.stop()
        self._started_at = None
        if not (
            self.controller.state.discovery_running or self.discovery_runner.is_running
        ):
            self.progress.setVisible(False)
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_export_folder(self) -> None:
        try:
            opened = self.controller.open_selected_export_folder()
        except ResultsPackagingUiError as exc:
            QMessageBox.information(self, "No export output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the export folder.")
        else:
            self.status_message.emit("Opened export folder.")

    def _update_elapsed(self) -> None:
        if self._started_at is None:
            self.elapsed_label.setText("Elapsed: —")
            return
        elapsed = time.perf_counter() - self._started_at
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(elapsed)}")

    def _update_enabled(self) -> None:
        busy = self.has_running_task
        packaging_busy = (
            self.controller.state.task_running or self.task_runner.is_running
        )
        self.session_browse_button.setEnabled(not packaging_busy)
        self.open_session_button.setEnabled(not packaging_busy)
        self.refresh_button.setEnabled(not busy)
        self.s8_combo.setEnabled(not busy)
        self.apply_settings_button.setEnabled(not busy)
        for widget in (
            self.enable_review_check,
            self.include_contact_check,
            self.include_engaged_check,
            self.include_sbs_check,
            self.require_video_check,
            self.hash_video_check,
            self.min_frames_spin,
            self.min_seconds_spin,
            self.export_selection_combo,
            self.export_channels_edit,
            self.export_behaviors_edit,
            self.export_profile_combo,
            self.export_pre_context_spin,
            self.export_post_context_spin,
            self.runs_table,
            self.exports_table,
        ):
            widget.setEnabled(not busy)
        can_run = self.controller.can_run_packaging() and not busy
        self.run_button.setEnabled(can_run)
        can_export = self.controller.can_run_export() and not busy
        self.export_csv_button.setEnabled(can_export)
        self.export_media_button.setEnabled(can_export)
        self.retry_media_button.setEnabled(
            self.controller.can_retry_selected_media_job() and not busy
        )
        has_output = (
            self.controller.selected_s9_run() is not None
            or self.controller.state.last_result is not None
        )
        self.open_output_button.setEnabled(has_output and not busy)
        has_export = (
            self.controller.selected_export_job() is not None
            or self.controller.state.last_csv_export_result is not None
            or self.controller.state.last_media_export_result is not None
        )
        self.open_export_button.setEnabled(has_export and not busy)

    def _guard_idle(self, action: str) -> bool:
        if self.has_running_task:
            self.status_message.emit(f"Cannot {action} while an S9 task is active.")
            return False
        return True

    def _emit_running_changed(self) -> None:
        running = self.has_running_task
        if running != self._last_running_state:
            self._last_running_state = running
            self.task_running_changed.emit(running)
