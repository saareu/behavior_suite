"""Subsystem 04 pose-finalization GUI: accepted S3 → final_pose.parquet."""

from __future__ import annotations

import time
from pathlib import Path

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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pose_finalization import TARGET_FPS_DEFAULT
from pose_finalization.run_discovery import (
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    STALE_SOURCE,
)
from ui.controllers.pose_finalization_controller import (
    PoseFinalizationController,
    PoseFinalizationUiError,
    PoseFinalizationUiSettings,
)
from ui.tasks import GuiTaskRunner, TaskAlreadyRunningError, format_task_duration


class PoseFinalizationPage(QWidget):
    """Compact S4 screen: session, accepted S3, Advanced Settings, QC summary."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: PoseFinalizationController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or PoseFinalizationController()
        self.task_runner = GuiTaskRunner(self)
        self._started_at: float | None = None
        self._last_running_state: bool | None = None
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    @property
    def has_running_task(self) -> bool:
        return self.task_runner.is_running or self.controller.state.task_running

    def set_session(self, session_root: Path | str) -> None:
        """Open a project/session directory and refresh discovery."""

        try:
            self.controller.set_session(session_root)
        except PoseFinalizationUiError as exc:
            QMessageBox.warning(self, "Cannot open session", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit(f"Opened session for S4: {session_root}")

    def refresh_from_state(self) -> None:
        """Push controller state into widgets without discarding defaults."""

        state = self.controller.state
        self.session_path.setText(str(state.session_root) if state.session_root else "")
        self._populate_s3_combo()
        self._populate_s4_table()
        self._sync_settings_widgets(state.settings)
        self._refresh_qc_panel()
        self._update_enabled()
        self._emit_running_changed()

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 4 — Pose Finalization")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Finalize an explicitly accepted Subsystem 3 result into dense 120 Hz "
            "final_pose.parquet. No visual review or S4 acceptance step."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session and accepted S3 source")
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
        self.s3_combo = QComboBox()
        self.s3_combo.currentIndexChanged.connect(self._s3_selection_changed)
        self.s3_status_label = QLabel("Select a project/session.")
        self.s3_status_label.setWordWrap(True)
        self.s3_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.refresh_button = QPushButton("Refresh discovery")
        self.refresh_button.clicked.connect(self._refresh_discovery)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("Accepted S3 run", self.s3_combo)
        session_form.addRow("Eligibility", self.s3_status_label)
        session_form.addRow(self.refresh_button)

        advanced = QGroupBox("Advanced Settings")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_form = QFormLayout(advanced)
        self.linear_max_spin = QSpinBox()
        self.linear_max_spin.setRange(0, 10_000)
        self.kalman_max_spin = QSpinBox()
        self.kalman_max_spin.setRange(0, 10_000)
        self.process_var_spin = QDoubleSpinBox()
        self.process_var_spin.setDecimals(4)
        self.process_var_spin.setRange(1e-6, 1e6)
        self.process_var_spin.setSingleStep(0.1)
        self.measurement_var_spin = QDoubleSpinBox()
        self.measurement_var_spin.setDecimals(4)
        self.measurement_var_spin.setRange(1e-6, 1e6)
        self.measurement_var_spin.setSingleStep(0.1)
        self.smooth_enabled = QCheckBox("Enable Savitzky–Golay smoothing")
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 501)
        self.window_spin.setSingleStep(2)
        self.polyorder_spin = QSpinBox()
        self.polyorder_spin.setRange(0, 20)
        self.target_fps_label = QLabel(f"{TARGET_FPS_DEFAULT:g} Hz (fixed for MVP)")
        advanced_form.addRow("Linear maximum gap (frames)", self.linear_max_spin)
        advanced_form.addRow("Kalman maximum gap (frames)", self.kalman_max_spin)
        advanced_form.addRow("Kalman process variance", self.process_var_spin)
        advanced_form.addRow("Kalman measurement variance", self.measurement_var_spin)
        advanced_form.addRow(self.smooth_enabled)
        advanced_form.addRow("SavGol window (frames)", self.window_spin)
        advanced_form.addRow("SavGol polynomial order", self.polyorder_spin)
        advanced_form.addRow("Output sampling", self.target_fps_label)
        self._sync_settings_widgets(self.controller.state.settings)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Finalization")
        self.run_button.clicked.connect(self._run_finalization)
        self.open_output_button = QPushButton("Open S4 run folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.open_output_button)
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

        qc_group = QGroupBox("QC summary")
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
        self.qc_table.setMaximumHeight(180)
        qc_layout.addWidget(self.qc_table)

        runs_group = QGroupBox("Completed S4 runs (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 6)
        self.runs_table.setHorizontalHeaderLabels(
            ("Run ID", "Status", "Eligibility", "Source S3", "Warnings", "Output")
        )
        self.runs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.verticalHeader().setVisible(False)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.runs_table.itemSelectionChanged.connect(self._s4_selection_changed)
        runs_layout.addWidget(self.runs_table)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(session_group)
        layout.addWidget(advanced)
        layout.addLayout(run_row)
        layout.addWidget(status_group)
        layout.addWidget(qc_group)
        layout.addWidget(runs_group, 1)

    def _browse_session(self) -> None:
        start = self.session_path.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if not chosen:
            return
        self.set_session(chosen)

    def _open_typed_session(self) -> None:
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _refresh_discovery(self) -> None:
        if self.has_running_task:
            self.status_message.emit("Cannot refresh while finalization is running.")
            return
        try:
            self.controller.refresh_discovery()
        except PoseFinalizationUiError as exc:
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("S4 discovery refreshed.")

    def _populate_s3_combo(self) -> None:
        self.s3_combo.blockSignals(True)
        self.s3_combo.clear()
        accepted = self.controller.state.accepted_s3_runs
        if not accepted:
            self.s3_combo.addItem("(none eligible)", None)
            reason = self.controller.cannot_run_reason()
            self.s3_status_label.setText(reason or "No accepted S3 run available.")
        else:
            for run in accepted:
                label = f"{run.run_id}  (accepted {run.accepted_at or '—'})"
                self.s3_combo.addItem(label, run.run_id)
            selected = self.controller.state.selected_s3_run_id
            if selected is not None:
                index = self.s3_combo.findData(selected)
                if index >= 0:
                    self.s3_combo.setCurrentIndex(index)
            current = self.controller.selected_accepted_s3()
            if current is not None:
                self.s3_status_label.setText(
                    f"Using accepted S3 run {current.run_id}. "
                    f"tracked_pose SHA256={current.tracked_pose_sha256[:12]}…"
                )
        self.s3_combo.blockSignals(False)

    def _populate_s4_table(self) -> None:
        summary = self.controller.state.summary
        runs = summary.runs if summary is not None else ()
        self.runs_table.setRowCount(len(runs))
        selected_id = self.controller.state.selected_s4_run_id
        selected_row = -1
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.status or "—",
                _eligibility_label(run.classification),
                run.source_s3_run_id or "—",
                str(run.warning_count),
                str(run.final_pose_path) if run.final_pose_path else "—",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                self.runs_table.setItem(row, col, item)
            if run.run_id == selected_id:
                selected_row = row
        if selected_row >= 0:
            self.runs_table.selectRow(selected_row)

    def _sync_settings_widgets(self, settings: PoseFinalizationUiSettings) -> None:
        self.linear_max_spin.setValue(settings.linear_max_frames)
        self.kalman_max_spin.setValue(settings.kalman_max_frames)
        self.process_var_spin.setValue(settings.process_var)
        self.measurement_var_spin.setValue(settings.measurement_var)
        self.smooth_enabled.setChecked(settings.smooth_enabled)
        self.window_spin.setValue(settings.savgol_window_length)
        self.polyorder_spin.setValue(settings.savgol_polyorder)

    def _read_settings_from_widgets(self) -> PoseFinalizationUiSettings:
        return PoseFinalizationUiSettings(
            linear_max_frames=self.linear_max_spin.value(),
            kalman_max_frames=self.kalman_max_spin.value(),
            process_var=self.process_var_spin.value(),
            measurement_var=self.measurement_var_spin.value(),
            smooth_enabled=self.smooth_enabled.isChecked(),
            savgol_window_length=self.window_spin.value(),
            savgol_polyorder=self.polyorder_spin.value(),
            target_fps=TARGET_FPS_DEFAULT,
        )

    def _s3_selection_changed(self) -> None:
        if self.has_running_task:
            return
        run_id = self.s3_combo.currentData()
        try:
            self.controller.select_accepted_s3_run(run_id)
        except PoseFinalizationUiError as exc:
            self.s3_status_label.setText(str(exc))
            self.status_message.emit(str(exc))
            self._populate_s3_combo()
            self._update_enabled()
            return
        self.refresh_from_state()

    def _s4_selection_changed(self) -> None:
        items = self.runs_table.selectedItems()
        if not items:
            return
        run_id = items[0].data(Qt.ItemDataRole.UserRole)
        try:
            self.controller.select_s4_run(str(run_id) if run_id else None)
        except PoseFinalizationUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_qc_panel()
        self._update_enabled()

    def _run_finalization(self) -> None:
        if not self._guard_idle("start pose finalization"):
            return
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
            request = self.controller.build_request()
        except PoseFinalizationUiError as exc:
            QMessageBox.warning(self, "Cannot run finalization", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return

        self._started_at = time.perf_counter()
        self.controller.state.task_running = True
        self.controller.state.run_stage = "Running pose finalization…"
        self.controller.state.run_error = None
        self.stage_label.setText("Running pose finalization…")
        self.progress.setVisible(True)
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        self.status_message.emit("Pose finalization started.")

        try:
            self.task_runner.start(
                lambda: self.controller.execute_request(request),
                task_name="pose finalization",
                on_success=self._on_run_success,
                on_error=self._on_run_error,
                on_finished=self._on_run_finished,
            )
        except TaskAlreadyRunningError:
            self.controller.state.task_running = False
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A pose finalization task is already active.",
            )

    def _on_run_success(self, result: object) -> None:
        from pose_finalization import PoseFinalizationResult

        if not isinstance(result, PoseFinalizationResult):
            self.unexpected_error.emit(f"Unexpected S4 result type: {type(result)!r}")
            return
        self.controller.record_success(result)
        self.stage_label.setText(f"Finished: {result.status}")
        self.refresh_from_state()
        warnings = f" ({len(result.warnings)} warnings)" if result.warnings else ""
        self.status_message.emit(
            f"S4 {result.status}{warnings}. Output: {result.final_pose_path}"
        )

    def _on_run_error(self, exc: BaseException) -> None:
        message = str(exc)
        self.controller.record_failure(message)
        self.stage_label.setText(f"Failed: {message}")
        try:
            self.controller.refresh_discovery()
        except PoseFinalizationUiError:
            pass
        self.refresh_from_state()
        from pose_finalization import PoseFinalizationError

        if isinstance(exc, (PoseFinalizationUiError, PoseFinalizationError)):
            QMessageBox.warning(self, "Pose finalization failed", message)
        else:
            self.unexpected_error.emit(message)
        self.status_message.emit(f"S4 failed: {message}")

    def _on_run_finished(self) -> None:
        self.controller.state.task_running = False
        self.progress.setVisible(False)
        self.elapsed_timer.stop()
        self._started_at = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_output_folder(self) -> None:
        try:
            opened = self.controller.open_selected_s4_folder()
        except PoseFinalizationUiError as exc:
            QMessageBox.information(self, "No S4 output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the S4 run folder.")
        else:
            self.status_message.emit("Opened S4 run folder.")

    def _refresh_qc_panel(self) -> None:
        rows = self.controller.qc_display_rows()
        self.qc_table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.qc_table.setItem(index, 0, QTableWidgetItem(name))
            self.qc_table.setItem(index, 1, QTableWidgetItem(value))
        if not rows:
            state = self.controller.state
            if state.run_error:
                self.stage_label.setText(f"Failed: {state.run_error}")
            else:
                self.stage_label.setText(state.run_stage)

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
        self.s3_combo.setEnabled(not busy)
        self.linear_max_spin.setEnabled(not busy)
        self.kalman_max_spin.setEnabled(not busy)
        self.process_var_spin.setEnabled(not busy)
        self.measurement_var_spin.setEnabled(not busy)
        self.smooth_enabled.setEnabled(not busy)
        self.window_spin.setEnabled(not busy)
        self.polyorder_spin.setEnabled(not busy)
        can_run = self.controller.can_run_finalization() and not busy
        self.run_button.setEnabled(can_run)
        has_output = (
            self.controller.selected_s4_run() is not None
            or self.controller.state.last_result is not None
        )
        self.open_output_button.setEnabled(has_output and not busy)

    def _guard_idle(self, action: str) -> bool:
        if self.has_running_task:
            self.status_message.emit(f"Cannot {action} while a task is active.")
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
        STALE_SOURCE: "stale (S3 superseded/invalid)",
        FAILED_OR_INCOMPLETE: "failed/incomplete",
        MISSING_REQUIRED_ARTIFACTS: "missing artifacts",
    }
    return mapping.get(classification, classification)
