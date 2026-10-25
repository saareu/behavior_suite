"""Subsystem 06 Combined Features GUI: eligible S4+S5 → dyadic feature artifact."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from combined_features import CombinedFeaturesError, CombinedFeaturesResult
from combined_features.contracts import (
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    STALE_SOURCE,
)
from ui.controllers.combined_features_controller import (
    CombinedFeaturesController,
    CombinedFeaturesUiError,
    CombinedFeaturesUiSettings,
    ExtractionRequestIdentity,
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


class CombinedFeaturesPage(QWidget):
    """S6 workspace: joint S4/S5 selection, extraction, QC, and model-compat."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: CombinedFeaturesController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or CombinedFeaturesController()
        self.task_runner = GuiTaskRunner(self)
        self._syncing_widgets = False
        self._started_at: float | None = None
        self._last_running_state: bool | None = None
        self._active_extraction_identity: ExtractionRequestIdentity | None = None
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    @property
    def has_running_task(self) -> bool:
        return self.task_runner.is_running or self.controller.state.task_running

    def set_session(self, session_root: Path | str) -> None:
        """Open a project/session and refresh S4/S5/S6 discovery."""

        try:
            self.controller.set_session(session_root)
        except CombinedFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot open S6 session", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(f"Opened session for S6: {session_root}")

    def refresh_from_state(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            self.session_path.setText(
                str(state.session_root) if state.session_root is not None else ""
            )
            self._sync_settings_widgets(state.working_settings)
            self._populate_source_combos()
            self._refresh_alignment_panel()
            self._populate_runs_table()
            self._refresh_qc_panel()
            self._refresh_compat_panel()
            self._update_enabled()
            self._emit_running_changed()
        finally:
            self._syncing_widgets = False

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 6 — Combined Pose & Video Features")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Combine currently eligible Subsystem 4 final pose with Subsystem 5 "
            "blob features into a directional dyadic feature artifact and a separate "
            "model-compatibility report. S6 does not train models or run inference. "
            "Cancellation is not supported; partial parquet is never published as completed."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session and upstream sources")
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

        self.s4_combo = QComboBox()
        self.s4_combo.currentIndexChanged.connect(self._s4_selection_changed)
        self.s5_combo = QComboBox()
        self.s5_combo.currentIndexChanged.connect(self._s5_selection_changed)
        self.s4_status_label = QLabel("No S4 selected.")
        self.s4_status_label.setWordWrap(True)
        self.s5_status_label = QLabel("No S5 selected.")
        self.s5_status_label.setWordWrap(True)
        self.alignment_label = QLabel(
            "Select a project/session with eligible S4 and S5 runs."
        )
        self.alignment_label.setWordWrap(True)
        self.alignment_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.compatible_pairs_label = QLabel("Compatible pairs: —")
        self.compatible_pairs_label.setWordWrap(True)
        self.refresh_button = QPushButton("Refresh discovery")
        self.refresh_button.clicked.connect(self._refresh_discovery)

        session_form.addRow("Session/project", session_row)
        session_form.addRow("Eligible S4 run", self.s4_combo)
        session_form.addRow("S4 status", self.s4_status_label)
        session_form.addRow("Eligible S5 run", self.s5_combo)
        session_form.addRow("S5 status", self.s5_status_label)
        session_form.addRow("Alignment", self.alignment_label)
        session_form.addRow("Passing pairs", self.compatible_pairs_label)
        session_form.addRow(self.refresh_button)

        settings_group = QGroupBox("S6 settings")
        settings_form = QFormLayout(settings_group)
        self.smooth_win_spin = QSpinBox()
        self.smooth_win_spin.setRange(1, 101)
        self.smooth_win_spin.setSingleStep(2)
        self.diff_dt_spin = QSpinBox()
        self.diff_dt_spin.setRange(1, 100)
        self.iteration_edit = QLineEdit()
        self.exp_name_edit = QLineEdit()
        self.exp_name_edit.setPlaceholderText("(default: session folder name)")
        settings_form.addRow("smooth_win (odd)", self.smooth_win_spin)
        settings_form.addRow("diff_dt", self.diff_dt_spin)
        settings_form.addRow("iteration_name", self.iteration_edit)
        settings_form.addRow("exp_name", self.exp_name_edit)

        registry_group = QGroupBox("Model registry (optional — no inference)")
        registry_form = QFormLayout(registry_group)
        self.registry_path = QLineEdit()
        self.registry_browse_button = QPushButton("Browse…")
        self.registry_browse_button.clicked.connect(self._browse_registry)
        self.registry_clear_button = QPushButton("Clear")
        self.registry_clear_button.clicked.connect(self._clear_registry)
        registry_row = QHBoxLayout()
        registry_row.addWidget(self.registry_path, 1)
        registry_row.addWidget(self.registry_browse_button)
        registry_row.addWidget(self.registry_clear_button)
        self.include_focal_check = QCheckBox("Include focal detectors")
        self.include_focal_check.setChecked(True)
        self.registry_status_label = QLabel(
            "Registry absent is allowed; feature extraction still proceeds. "
            "Compatibility reports schema/availability only — not inference readiness."
        )
        self.registry_status_label.setWordWrap(True)
        registry_form.addRow("Registry root", registry_row)
        registry_form.addRow(self.include_focal_check)
        registry_form.addRow(self.registry_status_label)

        self.compat_table = QTableWidget(0, 4)
        self.compat_table.setHorizontalHeaderLabels(
            ("Detector", "Family", "Schema status", "Details")
        )
        self.compat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.compat_table.verticalHeader().setVisible(False)
        compat_header = self.compat_table.horizontalHeader()
        compat_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        compat_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.compat_table.setMaximumHeight(160)
        registry_form.addRow(self.compat_table)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Combined Feature Extraction")
        self.run_button.clicked.connect(self._run_extraction)
        self.open_output_button = QPushButton("Open S6 run folder")
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

        qc_group = QGroupBox("QC and results")
        qc_layout = QVBoxLayout(qc_group)
        self.qc_table = QTableWidget(0, 2)
        self.qc_table.setHorizontalHeaderLabels(("Metric", "Value"))
        self.qc_table.verticalHeader().setVisible(False)
        self.qc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.qc_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.qc_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.qc_table.setMaximumHeight(220)
        qc_layout.addWidget(self.qc_table)

        runs_group = QGroupBox("Completed S6 runs (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(
            (
                "Run ID",
                "Status",
                "Current eligibility",
                "Source S4",
                "Source S5",
                "Warnings",
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
        self.runs_table.itemSelectionChanged.connect(self._s6_selection_changed)
        runs_layout.addWidget(self.runs_table)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(session_group)
        layout.addWidget(settings_group)
        layout.addWidget(registry_group)
        layout.addLayout(run_row)
        layout.addWidget(status_group)
        layout.addWidget(qc_group)
        layout.addWidget(runs_group, 1)

    def _browse_session(self) -> None:
        if not self._guard_idle("change session"):
            return
        start = self.session_path.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if not chosen:
            return
        self.set_session(chosen)

    def _open_typed_session(self) -> None:
        if not self._guard_idle("change session"):
            return
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _browse_registry(self) -> None:
        if not self._guard_idle("change model registry"):
            return
        start = self.registry_path.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, "Select model registry root", start
        )
        if not chosen:
            return
        self.registry_path.setText(chosen)
        self._apply_settings()

    def _clear_registry(self) -> None:
        if not self._guard_idle("clear model registry"):
            return
        self.registry_path.clear()
        self._apply_settings()

    def _refresh_discovery(self) -> None:
        if not self._guard_idle("refresh discovery"):
            return
        try:
            self.controller.refresh_discovery()
        except CombinedFeaturesUiError as exc:
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("S6 discovery refreshed.")

    def _apply_settings(self) -> None:
        if not self._guard_idle("apply settings"):
            return
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
        except CombinedFeaturesUiError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            "S6 settings applied (registry changes do not alter feature formulas)."
        )

    def _populate_source_combos(self) -> None:
        state = self.controller.state
        self.s4_combo.blockSignals(True)
        self.s5_combo.blockSignals(True)
        self.s4_combo.clear()
        self.s5_combo.clear()

        if not state.s4_options:
            self.s4_combo.addItem("(none eligible)", None)
            self.s4_status_label.setText("No currently eligible S4 run.")
        else:
            for opt in state.s4_options:
                self.s4_combo.addItem(opt.label, opt.run_id)
            selected = state.selected_s4_run_id
            if selected is not None:
                index = self.s4_combo.findData(selected)
                if index >= 0:
                    self.s4_combo.setCurrentIndex(index)
            current = self.controller.selected_s4_source()
            if current is not None:
                manual = " (manual)" if state.s4_selection_manual else " (default)"
                self.s4_status_label.setText(
                    f"{current.run_id}: status={current.status or '—'}, "
                    f"eligibility={current.classification}{manual}"
                )

        if not state.s5_options:
            self.s5_combo.addItem("(none eligible)", None)
            self.s5_status_label.setText("No currently eligible S5 run.")
        else:
            for opt in state.s5_options:
                self.s5_combo.addItem(opt.label, opt.run_id)
            selected = state.selected_s5_run_id
            if selected is not None:
                index = self.s5_combo.findData(selected)
                if index >= 0:
                    self.s5_combo.setCurrentIndex(index)
            current = self.controller.selected_s5_source()
            if current is not None:
                manual = " (manual)" if state.s5_selection_manual else " (default)"
                self.s5_status_label.setText(
                    f"{current.run_id}: status={current.status or '—'}, "
                    f"eligibility={current.classification}{manual}"
                )

        self.s4_combo.blockSignals(False)
        self.s5_combo.blockSignals(False)

    def _refresh_alignment_panel(self) -> None:
        self.alignment_label.setText(self.controller.alignment_status_text())
        pairs = self.controller.compatible_pair_summaries()
        if pairs:
            self.compatible_pairs_label.setText("Compatible pairs: " + "; ".join(pairs))
        else:
            self.compatible_pairs_label.setText(
                "Compatible pairs: none (independent eligibility does not prove joint alignment)."
            )
        alignment = self.controller.state.alignment
        if alignment is not None and alignment.sample_count is not None:
            self.alignment_label.setToolTip(
                f"session={alignment.session_root}\n"
                f"sync={alignment.prepared_sync}\n"
                f"sha256={alignment.prepared_sync_sha256}\n"
                f"checks={dict(alignment.checks)}"
            )

    def _populate_runs_table(self) -> None:
        summary = self.controller.state.summary
        runs = summary.runs if summary is not None else ()
        self.runs_table.setRowCount(len(runs))
        selected_id = self.controller.state.selected_s6_run_id
        selected_row = -1
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.status or "—",
                _eligibility_label(run.classification),
                run.source_s4_run_id or "—",
                run.source_s5_run_id or "—",
                str(run.warning_count),
                str(run.dyadic_features_path) if run.dyadic_features_path else "—",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                self.runs_table.setItem(row, col, item)
            if run.run_id == selected_id:
                selected_row = row
        if selected_row >= 0:
            self.runs_table.selectRow(selected_row)

    def _sync_settings_widgets(self, settings: CombinedFeaturesUiSettings) -> None:
        self.smooth_win_spin.setValue(int(settings.smooth_win))
        self.diff_dt_spin.setValue(int(settings.diff_dt))
        self.iteration_edit.setText(settings.iteration_name)
        self.exp_name_edit.setText(settings.exp_name or "")
        self.registry_path.setText(
            str(settings.model_registry_root)
            if settings.model_registry_root is not None
            else ""
        )
        self.include_focal_check.setChecked(bool(settings.include_focal))

    def _read_settings_from_widgets(self) -> CombinedFeaturesUiSettings:
        registry_text = self.registry_path.text().strip()
        exp_text = self.exp_name_edit.text().strip()
        return CombinedFeaturesUiSettings(
            smooth_win=self.smooth_win_spin.value(),
            diff_dt=self.diff_dt_spin.value(),
            iteration_name=self.iteration_edit.text().strip() or "suite",
            exp_name=exp_text or None,
            model_registry_root=Path(registry_text) if registry_text else None,
            include_focal=self.include_focal_check.isChecked(),
            include_obsolete=False,
        )

    def _s4_selection_changed(self) -> None:
        if self._syncing_widgets or self.has_running_task:
            return
        run_id = self.s4_combo.currentData()
        try:
            self.controller.select_s4_source(run_id, manual=True)
        except CombinedFeaturesUiError as exc:
            self.alignment_label.setText(str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._refresh_alignment_panel()
        self._update_enabled()

    def _s5_selection_changed(self) -> None:
        if self._syncing_widgets or self.has_running_task:
            return
        run_id = self.s5_combo.currentData()
        try:
            self.controller.select_s5_source(run_id, manual=True)
        except CombinedFeaturesUiError as exc:
            self.alignment_label.setText(str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._refresh_alignment_panel()
        self._update_enabled()

    def _s6_selection_changed(self) -> None:
        items = self.runs_table.selectedItems()
        if not items:
            return
        run_id = items[0].data(Qt.ItemDataRole.UserRole)
        try:
            self.controller.select_s6_run(str(run_id) if run_id else None)
        except CombinedFeaturesUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_qc_panel()
        self._refresh_compat_panel()
        self._update_enabled()

    def _run_extraction(self) -> None:
        if not self._guard_idle("start combined feature extraction"):
            return
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
            request, identity = self.controller.build_request()
        except CombinedFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot run S6 extraction", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return

        self._active_extraction_identity = identity
        self._started_at = time.perf_counter()
        self.stage_label.setText("Running combined feature extraction…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        self.status_message.emit(
            "S6 extraction started (settings snapshotted; cancellation unsupported)."
        )

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.COMBINED_FEATURES_EXTRACTION,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )

        def _operation(report: object) -> CombinedFeaturesResult:
            return self.controller.execute_request(
                request, progress_callback=report  # type: ignore[arg-type]
            )

        try:
            self.task_runner.start_progressive(
                _operation,
                task_name="S6 combined feature extraction",
                context=context,
                on_progress=lambda event: self._on_extraction_progress(event, identity),
                on_success=lambda result: self._on_extraction_success(result, identity),
                on_error=lambda exc: self._on_extraction_error(exc, identity),
                on_finished=lambda: self._on_extraction_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.controller.record_failure(
                "Extraction already running.", expected_identity=identity
            )
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A Subsystem 6 extraction is already active.",
            )

    def _on_extraction_progress(
        self,
        event: object,
        identity: ExtractionRequestIdentity,
    ) -> None:
        if self._active_extraction_identity is None:
            return
        if identity.request_id != self._active_extraction_identity.request_id:
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

    def _on_extraction_success(
        self, result: object, identity: ExtractionRequestIdentity
    ) -> None:
        if not isinstance(result, CombinedFeaturesResult):
            self.unexpected_error.emit(f"Unexpected S6 result type: {type(result)!r}")
            return
        accepted = self.controller.record_success(
            result, expected_identity=identity
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale S6 extraction result after session/source change."
            )
            return
        self.stage_label.setText(f"Finished: {result.status}")
        self.refresh_from_state()
        warnings = f" ({len(result.warnings)} warnings)" if result.warnings else ""
        self.status_message.emit(
            f"S6 {result.status}{warnings}. Output: {result.dyadic_features_path}"
        )

    def _on_extraction_error(
        self, exc: BaseException, identity: ExtractionRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.record_failure(
            message, expected_identity=identity
        )
        if not accepted:
            return
        self.stage_label.setText(f"Failed: {message}")
        with contextlib.suppress(CombinedFeaturesUiError):
            self.controller.refresh_discovery()
        self.refresh_from_state()
        if isinstance(exc, (CombinedFeaturesUiError, CombinedFeaturesError)):
            QMessageBox.warning(self, "Combined feature extraction failed", message)
            self.status_message.emit(f"S6 failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_extraction_finished(self, identity: ExtractionRequestIdentity) -> None:
        if (
            self._active_extraction_identity is not None
            and identity.request_id == self._active_extraction_identity.request_id
        ):
            self._active_extraction_identity = None
        self.controller.state.task_running = False
        self.progress.setVisible(False)
        self.elapsed_timer.stop()
        self._started_at = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_output_folder(self) -> None:
        try:
            opened = self.controller.open_selected_s6_folder()
        except CombinedFeaturesUiError as exc:
            QMessageBox.information(self, "No S6 output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the S6 run folder.")
        else:
            self.status_message.emit("Opened S6 run folder.")

    def _refresh_qc_panel(self) -> None:
        rows = self.controller.qc_display_rows()
        self.qc_table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.qc_table.setItem(index, 0, QTableWidgetItem(name))
            self.qc_table.setItem(index, 1, QTableWidgetItem(value))
        state = self.controller.state
        if state.run_error:
            self.stage_label.setText(f"Failed: {state.run_error}")
        elif not self.has_running_task:
            self.stage_label.setText(state.run_stage)

    def _refresh_compat_panel(self) -> None:
        preview = self.controller.state.model_compat_preview
        if isinstance(preview, dict):
            status = preview.get("registry_status", "—")
            notes = preview.get("notes") or []
            note_text = "; ".join(str(n) for n in notes) if notes else ""
            self.registry_status_label.setText(
                f"Registry status: {status}. {note_text} "
                "Feature-schema compatibility ≠ model-artifact presence ≠ inference readiness."
            )
        rows = self.controller.model_compat_display_rows()
        self.compat_table.setRowCount(len(rows))
        for index, values in enumerate(rows):
            for col, value in enumerate(values):
                self.compat_table.setItem(index, col, QTableWidgetItem(value))

    def _update_elapsed(self) -> None:
        if self._started_at is None:
            self.elapsed_label.setText("Elapsed: —")
            return
        elapsed = time.perf_counter() - self._started_at
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(elapsed)}")

    def _update_enabled(self) -> None:
        busy = self.has_running_task
        self.session_browse_button.setEnabled(not busy)
        self.open_session_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.s4_combo.setEnabled(not busy)
        self.s5_combo.setEnabled(not busy)
        self.smooth_win_spin.setEnabled(not busy)
        self.diff_dt_spin.setEnabled(not busy)
        self.iteration_edit.setEnabled(not busy)
        self.exp_name_edit.setEnabled(not busy)
        self.registry_path.setEnabled(not busy)
        self.registry_browse_button.setEnabled(not busy)
        self.registry_clear_button.setEnabled(not busy)
        self.include_focal_check.setEnabled(not busy)
        self.apply_settings_button.setEnabled(not busy)
        can_run = self.controller.can_run_extraction() and not busy
        self.run_button.setEnabled(can_run)
        has_output = (
            self.controller.selected_s6_run() is not None
            or self.controller.state.last_result is not None
        )
        self.open_output_button.setEnabled(has_output and not busy)

    def _guard_idle(self, action: str) -> bool:
        if self.has_running_task:
            self.status_message.emit(f"Cannot {action} while an S6 task is active.")
            return False
        return True

    def _emit_running_changed(self) -> None:
        running = self.has_running_task
        if running != self._last_running_state:
            self._last_running_state = running
            self.task_running_changed.emit(running)


def _eligibility_label(classification: str) -> str:
    mapping = {
        ELIGIBLE_DOWNSTREAM: "eligible",
        COMPLETED_WITH_WARNINGS: "eligible (warnings)",
        STALE_SOURCE: "stale source",
        FAILED_OR_INCOMPLETE: "failed/incomplete",
        MISSING_REQUIRED_ARTIFACTS: "missing artifacts",
    }
    return mapping.get(classification, classification)
