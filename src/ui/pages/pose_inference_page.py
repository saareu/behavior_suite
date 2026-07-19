"""Subsystem 02 run discovery, configuration, and asynchronous execution page."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pose_inference.run_discovery import (
    COMPLETE_REVIEWABLE,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    PoseInferenceRunSummary,
)
from pose_inference.runner import PoseInferencePreflightResult, PoseInferenceResult
from ui.controllers.pose_inference_controller import (
    InferenceMode,
    PoseInferenceController,
    PoseInferenceUiError,
    S1UiStatus,
    S3PoseInput,
)
from ui.tasks import GuiTaskRunner, TaskAlreadyRunningError, format_task_duration


class PoseInferencePage(QWidget):
    """Practical metadata-first MVP screen for Subsystem 02."""

    back_to_s1_requested = Signal()
    s3_handoff_requested = Signal(object)
    status_message = Signal(str)
    unexpected_error = Signal(str)
    backend_progress = Signal(int, str, object)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(self, controller: PoseInferenceController) -> None:
        super().__init__()
        self.controller = controller
        self.task_runner = GuiTaskRunner(self)
        self._started_at: float | None = None
        self._active_token: int | None = None
        self._last_running_state: bool | None = None
        self.backend_progress.connect(self._backend_progress_changed)
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 2 — Pose Inference")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Run or review SLEAP-NN pose inference. Technical completeness here is "
            "not a final identity or scientific-usability decision."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session and S1 handoff")
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
        self.prepared_video_label = self._selectable_label()
        self.s1_status_label = QLabel()
        self.s1_status_label.setWordWrap(True)
        self.frame_count_label = QLabel("—")
        self.timing_label = self._selectable_label()
        self.back_to_s1_button = QPushButton("Back to Subsystem 1")
        self.back_to_s1_button.clicked.connect(self.back_to_s1_requested.emit)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("Prepared video", self.prepared_video_label)
        session_form.addRow("S1 state", self.s1_status_label)
        session_form.addRow("Prepared frames", self.frame_count_label)
        session_form.addRow("Timing", self.timing_label)
        session_form.addRow(self.back_to_s1_button)

        runs_group = QGroupBox("Existing S2 runs (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 9)
        self.runs_table.setHorizontalHeaderLabels(
            (
                "Run ID",
                "Mode",
                "Model",
                "Timestamp",
                "Status",
                "Classification",
                "QC",
                "Review",
                "Artifacts",
            )
        )
        self.runs_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.runs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.verticalHeader().setVisible(False)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.runs_table.itemSelectionChanged.connect(self._run_selection_changed)
        self.runs_table.itemActivated.connect(self._activate_selected_run)
        self.refresh_runs_button = QPushButton("Refresh runs")
        self.refresh_runs_button.clicked.connect(self._refresh_discovery)
        runs_layout.addWidget(self.runs_table)
        self.empty_runs_label = QLabel(
            "No Subsystem 2 runs were found for this session. Configure a new "
            "inference run below."
        )
        self.empty_runs_label.setWordWrap(True)
        self.empty_runs_label.setStyleSheet("color: #555;")
        runs_layout.addWidget(self.empty_runs_label)
        runs_layout.addWidget(
            self.refresh_runs_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        self.selected_group = QGroupBox("Selected run")
        selected_layout = QVBoxLayout(self.selected_group)
        self.selected_summary = QLabel("Select a run.")
        self.selected_summary.setWordWrap(True)
        action_row = QHBoxLayout()
        self.open_overlay_button = QPushButton("Open overlay")
        self.open_overlay_button.clicked.connect(self._open_overlay)
        self.open_run_folder_button = QPushButton("Open run folder")
        self.open_run_folder_button.clicked.connect(self._open_run_folder)
        self.copy_settings_button = QPushButton("Rerun with copied settings")
        self.copy_settings_button.clicked.connect(self._copy_selected_settings)
        self.continue_s3_button = QPushButton("Select for Subsystem 3")
        self.continue_s3_button.clicked.connect(self._continue_to_s3)
        for button in (
            self.open_overlay_button,
            self.open_run_folder_button,
            self.copy_settings_button,
            self.continue_s3_button,
        ):
            action_row.addWidget(button)
        action_row.addStretch(1)
        self.technical_details = QPlainTextEdit()
        self.technical_details.setReadOnly(True)
        self.technical_details.setPlaceholderText(
            "Discovery metadata, command, QC findings, intervals, and artifact paths."
        )
        self.technical_details.setMaximumBlockCount(4000)
        self.technical_details.setMinimumHeight(80)
        self.technical_details.setMaximumHeight(220)
        selected_layout.addWidget(self.selected_summary)
        selected_layout.addLayout(action_row)
        selected_layout.addWidget(self.technical_details)

        config_group = QGroupBox("New inference configuration")
        config_layout = QVBoxLayout(config_group)
        config_form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Bottom-up", InferenceMode.BOTTOMUP.value)
        self.mode_combo.addItem("Top-down", InferenceMode.TOPDOWN.value)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        config_form.addRow("Inference mode", self.mode_combo)

        self.bottomup_row = QWidget()
        bottomup_layout = QHBoxLayout(self.bottomup_row)
        bottomup_layout.setContentsMargins(0, 0, 0, 0)
        self.bottomup_model = QLineEdit()
        self.bottomup_browse_button = QPushButton("Browse…")
        self.bottomup_browse_button.clicked.connect(
            lambda: self._browse_model(self.bottomup_model, "Choose bottom-up model")
        )
        bottomup_layout.addWidget(self.bottomup_model, 1)
        bottomup_layout.addWidget(self.bottomup_browse_button)
        config_form.addRow("Bottom-up model", self.bottomup_row)

        self.centroid_row = QWidget()
        centroid_layout = QHBoxLayout(self.centroid_row)
        centroid_layout.setContentsMargins(0, 0, 0, 0)
        self.centroid_model = QLineEdit()
        self.centroid_browse_button = QPushButton("Browse…")
        self.centroid_browse_button.clicked.connect(
            lambda: self._browse_model(self.centroid_model, "Choose centroid model")
        )
        centroid_layout.addWidget(self.centroid_model, 1)
        centroid_layout.addWidget(self.centroid_browse_button)
        config_form.addRow("Centroid model", self.centroid_row)

        self.centered_row = QWidget()
        centered_layout = QHBoxLayout(self.centered_row)
        centered_layout.setContentsMargins(0, 0, 0, 0)
        self.centered_model = QLineEdit()
        self.centered_browse_button = QPushButton("Browse…")
        self.centered_browse_button.clicked.connect(
            lambda: self._browse_model(
                self.centered_model, "Choose centered-instance model"
            )
        )
        centered_layout.addWidget(self.centered_model, 1)
        centered_layout.addWidget(self.centered_browse_button)
        config_form.addRow("Centered-instance model", self.centered_row)

        profile_row = QHBoxLayout()
        self.profile_path = QLineEdit()
        self.profile_browse_button = QPushButton("Browse…")
        self.profile_browse_button.clicked.connect(self._browse_profile)
        self.use_default_profile_button = QPushButton("Use mode default")
        self.use_default_profile_button.clicked.connect(self._use_default_profile)
        profile_row.addWidget(self.profile_path, 1)
        profile_row.addWidget(self.profile_browse_button)
        profile_row.addWidget(self.use_default_profile_button)
        config_form.addRow("Inference profile", profile_row)

        executable_row = QHBoxLayout()
        self.sleap_executable = QLineEdit()
        self.sleap_executable.setPlaceholderText(
            "Optional — normally resolved beside the current Python executable"
        )
        self.executable_browse_button = QPushButton("Browse…")
        self.executable_browse_button.clicked.connect(self._browse_executable)
        executable_row.addWidget(self.sleap_executable, 1)
        executable_row.addWidget(self.executable_browse_button)
        config_form.addRow("SLEAP executable", executable_row)
        self.run_purpose = QLineEdit()
        config_form.addRow("Run purpose / label", self.run_purpose)
        self.profile_settings = QLabel()
        self.profile_settings.setWordWrap(True)
        config_form.addRow("Profile-driven settings", self.profile_settings)
        self.active_run_dir_label = self._selectable_label()
        config_form.addRow("Active/result run directory", self.active_run_dir_label)
        config_layout.addLayout(config_form)
        self.configuration_summary = QLabel()
        self.configuration_summary.setWordWrap(True)
        self.configuration_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        config_layout.addWidget(self.configuration_summary)

        run_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate configuration")
        self.validate_button.clicked.connect(self._validate_configuration)
        self.run_button = QPushButton("Run Pose Inference")
        self.run_button.clicked.connect(self._run_inference)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.stage_label = QLabel("Not started")
        self.elapsed_label = QLabel("Elapsed: 00:00")
        run_row.addWidget(self.validate_button)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.progress)
        run_row.addWidget(self.stage_label, 1)
        run_row.addWidget(self.elapsed_label)
        config_layout.addLayout(run_row)
        progress_note = QLabel(
            "The backend reports coarse stage transitions and the run directory. "
            "The UI remains responsive and uses indeterminate activity because no "
            "reliable percentage exists; the processing log becomes authoritative "
            "when the backend returns."
        )
        progress_note.setWordWrap(True)
        config_layout.addWidget(progress_note)
        self.log_excerpt = QPlainTextEdit()
        self.log_excerpt.setReadOnly(True)
        self.log_excerpt.setPlaceholderText("Processing-log excerpt appears after a run returns.")
        self.log_excerpt.setMaximumHeight(150)
        config_layout.addWidget(self.log_excerpt)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(session_group)
        content_layout.addWidget(config_group)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(runs_group)
        splitter.addWidget(self.selected_group)
        splitter.setSizes([260, 180])
        content_layout.addWidget(splitter, 1)
        content_layout.addStretch(1)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.error_label)

        for edit in (
            self.bottomup_model,
            self.centroid_model,
            self.centered_model,
            self.profile_path,
            self.sleap_executable,
            self.run_purpose,
        ):
            edit.textChanged.connect(self._configuration_changed)

    @staticmethod
    def _selectable_label() -> QLabel:
        label = QLabel("—")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def set_session(self, session_root: Path | str) -> None:
        """Open S2 for a known S1 or main-application session context."""

        if not self._guard_idle("change the selected session"):
            return
        self.controller.set_session(session_root)
        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        """Render state without loading pose.slp, parquet, or video frames."""

        state = self.controller.state
        self.session_path.setText(str(state.session_root) if state.session_root else "")
        self.prepared_video_label.setText(
            str(state.prepared_video_path) if state.prepared_video_path else "—"
        )
        self.s1_status_label.setText(
            f"{state.s1_status.value.replace('_', ' ').title()}: {state.s1_message}"
        )
        self.s1_status_label.setStyleSheet(
            "color: #176b2c;"
            if state.s1_status is S1UiStatus.READY
            else "color: #b00020;"
        )
        self.frame_count_label.setText(
            f"{state.prepared_frame_count:,}"
            if state.prepared_frame_count is not None
            else "—"
        )
        self.timing_label.setText(state.timing_status or "—")
        mode_index = self.mode_combo.findData(state.inference_mode.value)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(mode_index)
        self.mode_combo.blockSignals(False)
        configured_values = (
            (self.bottomup_model, state.bottomup_model_path),
            (self.centroid_model, state.centroid_model_path),
            (self.centered_model, state.centered_instance_model_path),
            (self.profile_path, self.controller.active_profile_path()),
            (self.sleap_executable, state.sleap_executable_path),
            (self.run_purpose, state.run_purpose),
        )
        for edit, value in configured_values:
            edit.blockSignals(True)
            edit.setText(value)
            edit.blockSignals(False)
        self._render_mode_fields()
        self._render_profile_summary()
        self._populate_runs()
        self._render_selected_run()
        running = state.task_running
        self.progress.setVisible(running)
        self.run_button.setEnabled(self.controller.can_submit)
        self.validate_button.setEnabled(self.controller.can_submit)
        self.back_to_s1_button.setEnabled(not running)
        self.runs_table.setEnabled(not running)
        self.refresh_runs_button.setEnabled(not running)
        self.session_path.setEnabled(not running)
        self.session_browse_button.setEnabled(not running)
        self.open_session_button.setEnabled(not running)
        self.stage_label.setText(state.run_stage)
        self.active_run_dir_label.setText(
            str(state.active_run_dir) if state.active_run_dir is not None else "—"
        )
        self.error_label.setText(state.run_error or "")
        self.configuration_summary.setText(self.controller.configuration_summary())
        self._set_configuration_enabled(not running)
        if self._last_running_state is not running:
            self._last_running_state = running
            self.task_running_changed.emit(running)

    def _populate_runs(self) -> None:
        selected_id = self.controller.state.selected_run_id
        if (
            selected_id is None
            and self.controller.runs
            and not self.controller.state.task_running
        ):
            selected_id = self.controller.runs[0].run_id
            self.controller.select_run(selected_id)
        self.runs_table.blockSignals(True)
        self.runs_table.setRowCount(len(self.controller.runs))
        selected_row: int | None = None
        for row, run in enumerate(self.controller.runs):
            model = run.model_id or run.centroid_model_id or "—"
            timestamp = run.completed_at or run.created_at or "—"
            artifacts = " ".join(
                f"{label}:{'yes' if run.artifact_presence.get(key, False) else 'no'}"
                for label, key in (
                    ("SLP", "pose_slp"),
                    ("PQ", "pose_parquet"),
                    ("OV", "overlay_mp4"),
                    ("META", "pose_meta"),
                )
            )
            review = (
                "optional review"
                if run.pose_qc_outcome == "review_recommended"
                else "—"
            )
            values = (
                run.run_id,
                run.inference_mode or "—",
                model,
                timestamp,
                run.status or "—",
                run.classification,
                run.pose_qc_outcome or "—",
                review,
                artifacts,
            )
            color = self._run_color(run)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                if color is not None:
                    item.setBackground(color)
                self.runs_table.setItem(row, column, item)
            if run.run_id == selected_id:
                selected_row = row
        if selected_row is not None:
            self.runs_table.selectRow(selected_row)
        else:
            self.runs_table.clearSelection()
        self.runs_table.blockSignals(False)
        has_runs = bool(self.controller.runs)
        self.empty_runs_label.setVisible(not has_runs)
        self.runs_table.setVisible(has_runs)

    @staticmethod
    def _run_color(run: PoseInferenceRunSummary) -> QColor | None:
        if run.dry_run:
            return QColor("#eee7f5")
        if run.classification == FAILED_OR_INCOMPLETE:
            return QColor("#f8dddd")
        if run.classification == MISSING_REQUIRED_ARTIFACTS:
            return QColor("#f9e6cf")
        if run.classification == COMPLETE_REVIEWABLE:
            return QColor("#fff4c7" if run.pose_qc_outcome == "review_recommended" else "#dff1e3")
        return None

    def _render_selected_run(self) -> None:
        run = self.controller.selected_run
        if run is None:
            self.selected_summary.setText("Select a run.")
            self.technical_details.setPlainText(
                self.controller.technical_details()
            )
            for button in (
                self.open_overlay_button,
                self.open_run_folder_button,
                self.copy_settings_button,
                self.continue_s3_button,
            ):
                button.setEnabled(False)
            self.technical_details.setMaximumHeight(100)
            return
        self.technical_details.setMaximumHeight(220)
        reasons = ", ".join(run.review_recommendation_reasons) or "none"
        intervals, interval_diagnostics = self.controller.normalized_flagged_intervals(run)
        interval_count = sum(len(items) for items in intervals.values())
        interval_lines = self.controller.flagged_interval_display_lines(run)
        review_note = (
            "Review is optional; this run remains eligible for S3."
            if run.pose_qc_outcome == "review_recommended"
            else ""
        )
        self.selected_summary.setText(
            f"{run.run_id} — {run.classification}; QC: {run.pose_qc_outcome or 'unknown'}. "
            f"Review reasons: {reasons}. Flagged intervals: {interval_count}. {review_note}"
            + (f"\n{interval_lines[0]}" if interval_lines else "")
            + (
                f"\nMetadata diagnostic: {interval_diagnostics[0]}"
                if interval_diagnostics
                else ""
            )
        )
        self.technical_details.setPlainText(self.controller.technical_details(run))
        self.open_overlay_button.setEnabled(
            bool(run.artifact_presence.get("overlay_mp4"))
            and not self.controller.state.task_running
        )
        self.open_run_folder_button.setEnabled(
            run.run_dir.is_dir() and not self.controller.state.task_running
        )
        rerun_reason = self.controller.rerun_unavailable_reason(run)
        self.copy_settings_button.setEnabled(rerun_reason is None)
        self.copy_settings_button.setToolTip(rerun_reason or "")
        self.continue_s3_button.setEnabled(
            self.controller.selected_run_can_feed_s3()
            and not self.controller.state.task_running
        )

    def _render_mode_fields(self) -> None:
        topdown = self.controller.state.inference_mode is InferenceMode.TOPDOWN
        self.bottomup_row.setVisible(not topdown)
        self.centroid_row.setVisible(topdown)
        self.centered_row.setVisible(topdown)

    def _render_profile_summary(self) -> None:
        summary = self.controller.state.profile_summary
        if summary is None:
            self.profile_settings.setText(
                "Profile is missing or unreadable; backend preflight will report details."
            )
            self.profile_settings.setStyleSheet("color: #b00020;")
            return
        self.profile_settings.setText(
            f"profile={summary.profile_id or 'unnamed'}; mode={summary.inference_mode}; "
            f"device={summary.device or 'default'}; batch={summary.batch_size or 'default'}; "
            f"expected animals={summary.expected_animals or 'default'}; "
            f"tracking={'enabled' if summary.tracking_enabled else 'disabled'}"
        )
        matches = summary.inference_mode == self.controller.state.inference_mode.value
        self.profile_settings.setStyleSheet(
            "color: #176b2c;" if matches else "color: #b00020;"
        )

    def _set_configuration_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mode_combo,
            self.bottomup_model,
            self.centroid_model,
            self.centered_model,
            self.profile_path,
            self.sleap_executable,
            self.run_purpose,
            self.bottomup_browse_button,
            self.centroid_browse_button,
            self.centered_browse_button,
            self.profile_browse_button,
            self.use_default_profile_button,
            self.executable_browse_button,
        ):
            widget.setEnabled(enabled)

    def _sync_configuration_to_state(self) -> None:
        state = self.controller.state
        state.bottomup_model_path = self.bottomup_model.text().strip()
        state.centroid_model_path = self.centroid_model.text().strip()
        state.centered_instance_model_path = self.centered_model.text().strip()
        state.sleap_executable_path = self.sleap_executable.text().strip()
        state.run_purpose = self.run_purpose.text().strip()
        self.controller.set_active_profile_path(self.profile_path.text())

    def _configuration_changed(self) -> None:
        self._sync_configuration_to_state()
        self._render_profile_summary()
        self.configuration_summary.setText(self.controller.configuration_summary())
        self.run_button.setEnabled(self.controller.can_submit)
        self.validate_button.setEnabled(self.controller.can_submit)

    def _mode_changed(self) -> None:
        if not self._guard_idle("change inference mode"):
            return
        mode = InferenceMode(self.mode_combo.currentData())
        self._sync_configuration_to_state()
        self.controller.set_mode(mode)
        self.profile_path.blockSignals(True)
        self.profile_path.setText(self.controller.active_profile_path())
        self.profile_path.blockSignals(False)
        self._render_mode_fields()
        self._configuration_changed()

    def _browse_session(self) -> None:
        if not self._guard_idle("browse for a session"):
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose project/session")
        if directory:
            self.session_path.setText(directory)

    def _open_typed_session(self) -> None:
        if not self._guard_idle("open a session"):
            return
        self.set_session(self.session_path.text())

    def _browse_model(self, target: QLineEdit, title: str) -> None:
        if not self._guard_idle("browse for a model"):
            return
        directory = QFileDialog.getExistingDirectory(None, title)
        if directory:
            target.setText(directory)

    def _browse_profile(self) -> None:
        if not self._guard_idle("browse for a profile"):
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose inference profile", str(Path.cwd()), "YAML files (*.yaml *.yml)"
        )
        if filename:
            self.profile_path.setText(filename)

    def _use_default_profile(self) -> None:
        if not self._guard_idle("change the inference profile"):
            return
        path = str(
            self.controller.default_profile_path(self.controller.state.inference_mode)
        )
        self.controller.set_active_profile_path(
            path,
            explicitly_selected_default=True,
        )
        self.profile_path.blockSignals(True)
        self.profile_path.setText(path)
        self.profile_path.blockSignals(False)
        self._configuration_changed()

    def _browse_executable(self) -> None:
        if not self._guard_idle("browse for a SLEAP executable"):
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose SLEAP-NN executable", str(Path.cwd()), "Executables (*)"
        )
        if filename:
            self.sleap_executable.setText(filename)

    def _refresh_discovery(self) -> None:
        if not self._guard_idle("refresh run discovery"):
            return
        try:
            self.controller.refresh_discovery()
        except PoseInferenceUiError as exc:
            self.error_label.setText(str(exc))
        self.refresh_from_state()

    def _run_selection_changed(self) -> None:
        if not self._guard_idle("change the selected run"):
            return
        selected = self.runs_table.selectedItems()
        run_id = selected[0].data(Qt.ItemDataRole.UserRole) if selected else None
        try:
            self.controller.select_run(str(run_id) if run_id else None)
        except PoseInferenceUiError as exc:
            self.error_label.setText(str(exc))
        self._render_selected_run()

    def _activate_selected_run(self, _item: QTableWidgetItem) -> None:
        """Reveal details only for explicit activation, never automatic selection."""

        self.scroll_area.ensureWidgetVisible(self.selected_group)
        self.technical_details.setFocus(Qt.FocusReason.OtherFocusReason)

    def _run_inference(self) -> None:
        if not self._guard_idle("start another pose inference run"):
            return
        self._sync_configuration_to_state()
        token: int | None = None
        try:
            request = self.controller.begin_run()
            token = self.controller.active_task_token
            self._active_token = token
            self._started_at = time.perf_counter()
            self.error_label.clear()
            self.log_excerpt.clear()
            self.elapsed_timer.start()
            self.refresh_from_state()
            self.task_runner.start(
                lambda: self.controller.execute_request(
                    request,
                    progress_callback=lambda stage, run_dir: self.backend_progress.emit(
                        token, stage, run_dir
                    ),
                ),
                task_name="pose inference pipeline",
                on_success=lambda value: self._run_succeeded(value, token),
                on_error=lambda exc: self._run_failed(exc, token),
                on_finished=lambda: self._run_finished(token),
            )
        except PoseInferenceUiError as exc:
            self.error_label.setText(str(exc))
            self.refresh_from_state()
        except TaskAlreadyRunningError as exc:
            if token is not None:
                self.controller.abort_task_submission(token, str(exc))
                self._active_token = None
            self.error_label.setText(str(exc))
            self.refresh_from_state()

    def _run_succeeded(self, value: object, token: int) -> None:
        try:
            if not isinstance(value, PoseInferenceResult):
                raise TypeError("Pose inference worker returned an unexpected result.")
            if self.controller.apply_result(value, token=token):
                self._load_result_log(value)
        except BaseException as exc:
            self.controller.apply_error(exc, token=token)
            if not isinstance(exc, (PoseInferenceUiError, ValueError, OSError)):
                self.unexpected_error.emit(str(exc))
        self.refresh_from_state()

    def _run_failed(self, exc: BaseException, token: int) -> None:
        self.controller.apply_error(exc, token=token)
        if self.controller.state.preflight_report is not None:
            self.log_excerpt.setPlainText(self.controller.state.preflight_report)
        if self.controller.state.unexpected_error_detail is not None:
            self.unexpected_error.emit(str(exc))
        self.refresh_from_state()

    def _run_finished(self, token: int) -> None:
        if token != self._active_token:
            return
        self.controller.finish_run(token=token)
        self._active_token = None
        self.elapsed_timer.stop()
        self.progress.setVisible(False)
        self.refresh_from_state()
        self.status_message.emit(self.controller.state.run_stage)
        self.task_finished.emit()

    def _validate_configuration(self) -> None:
        if not self._guard_idle("validate another configuration"):
            return
        self._sync_configuration_to_state()
        token: int | None = None
        try:
            request = self.controller.begin_preflight()
            token = self.controller.active_task_token
            self._active_token = token
            self._started_at = time.perf_counter()
            self.error_label.clear()
            self.log_excerpt.clear()
            self.elapsed_timer.start()
            self.refresh_from_state()
            self.task_runner.start(
                lambda: self.controller.execute_preflight(request),
                task_name="pose inference preflight",
                on_success=lambda value: self._preflight_succeeded(value, token),
                on_error=lambda exc: self._run_failed(exc, token),
                on_finished=lambda: self._run_finished(token),
            )
        except PoseInferenceUiError as exc:
            self.error_label.setText(str(exc))
            self.refresh_from_state()
        except TaskAlreadyRunningError as exc:
            if token is not None:
                self.controller.abort_task_submission(token, str(exc))
                self._active_token = None
            self.error_label.setText(str(exc))
            self.refresh_from_state()

    def _preflight_succeeded(self, value: object, token: int) -> None:
        try:
            if not isinstance(value, PoseInferencePreflightResult):
                raise TypeError("Pose preflight worker returned an unexpected result.")
            if self.controller.apply_preflight_result(value, token=token):
                self.log_excerpt.setPlainText(
                    self.controller.state.preflight_report or "Configuration valid."
                )
        except BaseException as exc:
            self.controller.apply_error(exc, token=token)
            if not isinstance(exc, (PoseInferenceUiError, ValueError, OSError)):
                self.unexpected_error.emit(str(exc))
        self.refresh_from_state()

    def _load_result_log(self, result: PoseInferenceResult) -> None:
        path = result.processing_log_path
        try:
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError as exc:
            text = f"Could not read processing log: {exc}"
        self.log_excerpt.setPlainText(text[-12000:])

    def _update_elapsed(self) -> None:
        elapsed = 0.0 if self._started_at is None else time.perf_counter() - self._started_at
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(elapsed)}")

    def _backend_progress_changed(
        self,
        token: int,
        stage: str,
        run_dir: object,
    ) -> None:
        path = run_dir if isinstance(run_dir, Path) else None
        if self.controller.apply_progress(stage, path, token=token):
            self.stage_label.setText(stage)
            active_path = self.controller.state.active_run_dir
            self.active_run_dir_label.setText(
                str(active_path) if active_path is not None else "—"
            )
            self._refresh_active_log(path)

    def _refresh_active_log(self, run_dir: Path | None) -> None:
        if run_dir is None:
            return
        log_path = run_dir / "processing_log.txt"
        if not log_path.is_file():
            return
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return
        self.log_excerpt.setPlainText(text[-12000:])

    def _open_overlay(self) -> None:
        self._perform_action(self.controller.open_selected_overlay, "Opened overlay")

    def _open_run_folder(self) -> None:
        self._perform_action(self.controller.open_selected_run_folder, "Opened run folder")

    def _perform_action(self, action, success_message: str) -> None:
        if not self._guard_idle(success_message.lower()):
            return
        try:
            path = action()
        except (PoseInferenceUiError, OSError, subprocess.SubprocessError) as exc:
            self.error_label.setText(str(exc))
            return
        self.error_label.clear()
        self.status_message.emit(f"{success_message}: {path}")

    def _copy_selected_settings(self) -> None:
        if not self._guard_idle("copy settings for a rerun"):
            return
        try:
            self.controller.copy_selected_run_settings()
        except PoseInferenceUiError as exc:
            self.error_label.setText(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("Copied discoverable settings; review them before rerunning.")

    def _continue_to_s3(self) -> None:
        if not self._guard_idle("continue to Subsystem 3"):
            return
        try:
            handoff: S3PoseInput = self.controller.build_s3_handoff()
        except PoseInferenceUiError as exc:
            self.error_label.setText(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.s3_handoff_requested.emit(handoff)

    def _guard_idle(self, action: str) -> bool:
        if not self.has_running_task:
            return True
        self.error_label.setText(
            f"Cannot {action} while a Subsystem 2 task is active."
        )
        return False

    @property
    def has_running_task(self) -> bool:
        """Return whether navigation/close must wait for the backend call."""

        return self.task_runner.is_running or self.controller.state.task_running
