"""Optional MATLAB external-timing selection page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from preprocess.exceptions import PreprocessError, VideoProbeCancelledError
from preprocess.models import TimingUnit
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.timing_controller import (
    TimingController,
    TimingUnexpectedError,
    TimingValidationError,
)
from ui.state import RawReadableCountStatus, TimingMode
from ui.tasks import GuiTaskContext, GuiTaskRunner, TaskAlreadyRunningError

_TABLE_HEADERS = (
    "Variable",
    "Shape",
    "Type",
    "Length",
    "First",
    "Last",
    "Median dt",
    "Estimated FPS",
    "Length match",
)


class TimingPage(QWidget):
    """Collect one explicit timing choice through the headless controller."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)
    raw_probe_changed = Signal()

    def __init__(self, setup_controller: PreprocessSetupController) -> None:
        super().__init__()
        self.setup_controller = setup_controller
        self.controller = TimingController(
            setup_controller.state,
            task_coordinator=setup_controller.task_coordinator,
        )
        self.task_runner = GuiTaskRunner(self)
        self._refreshing = False

        title = QLabel("External Timing")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        description = QLabel(
            "Choose no external timing, or explicitly select and validate one "
            "numeric vector from a MATLAB workspace."
        )
        description.setWordWrap(True)

        self.no_timing = QRadioButton("No external timing")
        self.matlab_timing = QRadioButton("Use MATLAB .mat timing file")
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.no_timing)
        mode_row.addWidget(self.matlab_timing)
        mode_row.addStretch(1)

        self.mat_group = QGroupBox("MATLAB timing selection")
        self.mat_path = QLineEdit()
        self.mat_path.setReadOnly(True)
        browse_button = QPushButton("Browse .mat File")
        browse_button.clicked.connect(self._browse_mat_file)
        self.load_button = QPushButton("Load Workspace Variables")
        self.load_button.clicked.connect(self._load_variables)
        file_row = QHBoxLayout()
        file_row.addWidget(self.mat_path, 1)
        file_row.addWidget(browse_button)
        file_row.addWidget(self.load_button)

        self.candidate_table = QTableWidget(0, len(_TABLE_HEADERS))
        self.candidate_table.setHorizontalHeaderLabels(_TABLE_HEADERS)
        self.candidate_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.candidate_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.candidate_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)

        self.units = QComboBox()
        self.units.addItem("Select units…", None)
        for unit in TimingUnit:
            self.units.addItem(unit.value, unit.value)
        self.units.currentIndexChanged.connect(self._units_selected)

        self.raw_count_status = QLabel()
        self.raw_count_explanation = QLabel(
            "Exact TTL timing validation requires a full sequential count of "
            "readable raw frames."
        )
        self.raw_count_explanation.setWordWrap(True)
        self.raw_count_explanation.setStyleSheet("color: #8a5a00;")
        self.count_button = QPushButton("Count All Readable Raw Frames Now")
        self.count_button.clicked.connect(self._count_raw_frames)

        self.validate_button = QPushButton("Validate Selected Timing")
        self.validate_button.clicked.connect(self._validate_selection)
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setVisible(False)

        mat_layout = QVBoxLayout(self.mat_group)
        mat_layout.addLayout(file_row)
        mat_layout.addWidget(QLabel("Candidate numeric vectors"))
        mat_layout.addWidget(self.candidate_table, 1)
        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Selected vector units"))
        units_row.addWidget(self.units)
        units_row.addStretch(1)
        mat_layout.addLayout(units_row)
        mat_layout.addWidget(self.raw_count_status)
        mat_layout.addWidget(self.raw_count_explanation)
        mat_layout.addWidget(self.count_button)
        mat_layout.addWidget(self.busy_indicator)
        mat_layout.addWidget(self.validate_button)

        self.success_label = QLabel()
        self.success_label.setWordWrap(True)
        self.success_label.setStyleSheet("color: #176b2c;")
        self.warning_panel = QFrame()
        self.warning_panel.setObjectName("timingWarningPanel")
        self.warning_panel.setStyleSheet(
            "QFrame#timingWarningPanel {"
            "background-color: #fff3cd; border: 2px solid #a86100; "
            "border-radius: 4px; padding: 8px;"
            "}"
        )
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        warning_layout = QVBoxLayout(self.warning_panel)
        warning_layout.addWidget(self.warning_label)
        self.warning_panel.setVisible(False)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(mode_row)
        layout.addWidget(self.mat_group, 1)
        layout.addWidget(self.success_label)
        layout.addWidget(self.warning_panel)
        layout.addWidget(self.error_label)

        self.no_timing.toggled.connect(self._mode_changed)
        self.matlab_timing.toggled.connect(self._mode_changed)
        self.refresh_from_state()

    def is_valid(self) -> bool:
        """Return whether the current timing choice permits navigation."""

        return self.controller.can_advance()

    def commit(self) -> bool:
        """Validate only timing state already accepted by the controller."""

        try:
            self.controller.validate_navigation()
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))
            return False
        return True

    def on_activated(self) -> None:
        """Refresh raw-count and current timing state when the page opens."""

        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        """Render compact typed state without exposing full timing arrays."""

        self._refreshing = True
        state = self.controller.state
        self.no_timing.setChecked(state.timing_mode is TimingMode.NO_EXTERNAL)
        self.matlab_timing.setChecked(state.timing_mode is TimingMode.MATLAB)
        self.mat_group.setEnabled(state.timing_mode is TimingMode.MATLAB)
        self.mat_path.setText(
            "" if state.external_time_mat_path is None else str(state.external_time_mat_path)
        )
        units_index = (
            0
            if state.selected_timing_units is None
            else self.units.findData(state.selected_timing_units.value)
        )
        self.units.setCurrentIndex(max(units_index, 0))
        self._refreshing = False
        self._refresh_candidate_table()
        self._refresh_raw_count()
        self._refresh_result()

    def _mode_changed(self) -> None:
        if self._refreshing:
            return
        if self.no_timing.isChecked():
            self.controller.choose_no_external_timing()
            self.status_message.emit("External timing is not provided.")
        elif self.matlab_timing.isChecked():
            self.controller.choose_matlab_timing()
            self.status_message.emit("Select and validate a MATLAB timing vector.")
        self.error_label.clear()
        self.refresh_from_state()
        self.validity_changed.emit()

    def _browse_mat_file(self) -> None:
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose MATLAB timing file",
            str(Path.cwd()),
            "MATLAB files (*.mat);;All files (*)",
        )
        if filename:
            self.controller.set_mat_file_path(Path(filename))
            self.refresh_from_state()
            self.validity_changed.emit()

    def _load_variables(self) -> None:
        try:
            self.controller.load_mat_file()
            self.error_label.clear()
            self.success_label.setText("MATLAB candidate variables loaded.")
            self.refresh_from_state()
            self.status_message.emit("MATLAB timing candidates loaded.")
            self.validity_changed.emit()
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))
        except TimingUnexpectedError as exc:
            self._show_unexpected(exc)

    def _candidate_selected(self) -> None:
        if self._refreshing:
            return
        selected_rows = self.candidate_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        variable_name = self.candidate_table.item(selected_rows[0].row(), 0).text()
        try:
            self.controller.select_candidate(variable_name)
            self._refresh_result()
            self.validity_changed.emit()
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))

    def _units_selected(self) -> None:
        if self._refreshing:
            return
        units = self.units.currentData()
        if units is None:
            self.controller.clear_units_selection()
            self._refresh_result()
            self.validity_changed.emit()
            return
        try:
            self.controller.select_units(units)
            self._refresh_result()
            self.validity_changed.emit()
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))

    def _count_raw_frames(self) -> None:
        context: GuiTaskContext | None = None
        try:
            force_recount = self.controller.raw_frame_count_action() == "recount"
            path, context = self.controller.begin_full_raw_frame_count(
                force_recount=force_recount
            )
            self._set_busy(True)
            self.task_runner.start(
                lambda: self.controller.compute_full_raw_frame_count(path, context),
                task_name=(
                    "sequential raw-frame recount"
                    if force_recount
                    else "sequential raw-frame count"
                ),
                context=context,
                on_success=lambda result: self._raw_count_succeeded(result, context),
                on_error=lambda exc: self._raw_count_failed(exc, context),
                on_finished=lambda: self._raw_count_finished(context),
            )
        except (TimingValidationError, TaskAlreadyRunningError) as exc:
            if context is not None and not self.task_runner.is_running:
                self.controller.finish_task(context)
            self._set_busy(False)
            self._show_expected_error(str(exc))
        except Exception as exc:
            if context is not None and not self.task_runner.is_running:
                self.controller.finish_task(context)
            self._set_busy(False)
            self._show_unexpected(exc)

    def _raw_count_succeeded(
        self,
        result: object,
        context: GuiTaskContext,
    ) -> None:
        try:
            count = self.controller.apply_full_raw_frame_count(
                result,
                context=context,
            )
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))
            return
        except Exception as exc:
            self._show_unexpected(exc)
            return
        if count is None:
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit(f"Sequential readable-frame count completed: {count}.")
        self.validity_changed.emit()

    def _raw_count_failed(
        self,
        exc: BaseException,
        context: GuiTaskContext,
    ) -> None:
        applied = self.controller.record_full_raw_frame_count_failure(
            exc,
            context=context,
        )
        if not applied:
            return
        if isinstance(exc, VideoProbeCancelledError):
            self.refresh_from_state()
            self.status_message.emit("Raw readable-frame count cancelled.")
            self.validity_changed.emit()
            return
        if isinstance(exc, (PreprocessError, OSError, ValueError)):
            self._show_expected_error(
                self.controller.state.last_validation_error
                or "Could not count readable raw frames."
            )
        else:
            self._show_unexpected(
                TimingUnexpectedError(
                    "An unexpected error occurred while counting readable raw frames."
                )
            )

    def _raw_count_finished(self, context: GuiTaskContext) -> None:
        self.controller.finish_task(context)
        self._set_busy(False)
        self.refresh_from_state()
        self.raw_probe_changed.emit()

    def _set_busy(self, busy: bool) -> None:
        self.busy_indicator.setVisible(busy)
        self.no_timing.setEnabled(not busy)
        self.matlab_timing.setEnabled(not busy)
        self.load_button.setEnabled(not busy)
        self.candidate_table.setEnabled(not busy)
        self.units.setEnabled(not busy)
        self._refresh_raw_count(busy=busy)

    def _validate_selection(self) -> None:
        try:
            selection = self.controller.validate_selected_timing()
            self.error_label.clear()
            self.refresh_from_state()
            self.status_message.emit(
                f"Timing variable '{selection.selected_variable}' is valid."
            )
            self.validity_changed.emit()
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))
        except TimingUnexpectedError as exc:
            self._show_unexpected(exc)

    def _refresh_candidate_table(self) -> None:
        summaries = self.controller.candidate_summaries()
        selected_name = self.controller.state.selected_timing_variable
        self._refreshing = True
        self.candidate_table.setRowCount(len(summaries))
        for row_index, summary in enumerate(summaries):
            values = (
                summary.variable_name,
                str(summary.shape),
                summary.dtype,
                str(summary.length_after_squeeze),
                self._format_number(summary.first_value),
                self._format_number(summary.last_value),
                self._format_number(summary.median_difference),
                self._format_number(summary.estimated_fps),
                self._format_length_match(summary.length_matches_raw),
            )
            for column_index, value in enumerate(values):
                self.candidate_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )
            if summary.variable_name == selected_name:
                self.candidate_table.selectRow(row_index)
        self._refreshing = False

    def _refresh_raw_count(self, *, busy: bool = False) -> None:
        state = self.controller.state
        readable_count = self.controller.reusable_raw_frame_count()
        status = state.raw_readable_count_status
        is_matlab = state.timing_mode is TimingMode.MATLAB
        count_active = status in {
            RawReadableCountStatus.COUNTING,
            RawReadableCountStatus.RECOUNTING,
        }
        controls_blocked = busy or count_active
        if readable_count is not None:
            if status is RawReadableCountStatus.RECOUNTING:
                count_status = "Raw sequential readable-frame count: Recounting…"
            elif status is RawReadableCountStatus.RECOUNT_FAILED:
                count_status = (
                    "Raw sequential readable-frame count:\n"
                    "Recount failed; existing verified count retained"
                )
            elif status is RawReadableCountStatus.COUNT_CANCELLED:
                count_status = "Raw sequential readable-frame count: Count cancelled"
            elif status is RawReadableCountStatus.RECORDED_PRIOR_RUN:
                count_status = (
                    "Raw sequential readable-frame count:\n"
                    "Recorded by prior completed preprocessing run: "
                    f"{readable_count:,}"
                )
            else:
                count_status = (
                    "Raw sequential readable-frame count:\n"
                    f"Verified during this GUI session: {readable_count:,}"
                )
            self.raw_count_status.setText(count_status)
            self.raw_count_explanation.setText(
                "This source-matched sequential count is reused automatically for "
                "exact TTL validation. A diagnostic recount decodes the entire video "
                "again and may take substantial time."
            )
            self.count_button.setText("Recount Readable Raw Frames")
            self.validate_button.setEnabled(not controls_blocked and is_matlab)
        else:
            if status is RawReadableCountStatus.COUNTING:
                count_status = "Full sequential readable-frame count in progress…"
            elif status is RawReadableCountStatus.COUNT_FAILED:
                count_status = "Readable raw-frame count failed."
            elif status is RawReadableCountStatus.COUNT_CANCELLED:
                count_status = "Raw sequential readable-frame count: Count cancelled"
            else:
                count_status = "Raw sequential readable-frame count: Not measured"
            self.raw_count_status.setText(count_status)
            self.raw_count_explanation.setText(
                "Exact TTL timing validation requires a full sequential readable-frame "
                "count. Reported ffprobe or OpenCV metadata counts are not substitutes."
            )
            self.count_button.setText("Count All Readable Raw Frames Now")
            self.validate_button.setEnabled(False)
        self.count_button.setEnabled(not controls_blocked and is_matlab)

    def _refresh_result(self) -> None:
        state = self.controller.state
        if state.timing_mode is TimingMode.NO_EXTERNAL:
            self.success_label.setText("No external timing will be used.")
        elif state.timing_valid and state.external_time_selection is not None:
            suffix = (
                "converted to seconds"
                if state.external_time_vector_seconds is not None
                else "retained without a seconds conversion"
            )
            self.success_label.setText(
                f"Validated {state.selected_timing_variable} ({suffix})."
            )
        else:
            self.success_label.setText("MATLAB timing selection is not yet validated.")
        self._refresh_warning()

    def _refresh_warning(self) -> None:
        assessment = self.controller.state.timing_plausibility_assessment
        if assessment is None or not assessment.warning_triggered:
            self.warning_label.clear()
            self.warning_panel.setVisible(False)
            return
        self.warning_label.setText(
            "Selected timing units: "
            f"{assessment.selected_timing_units.value}\n"
            "External timing estimated FPS: "
            f"{self._format_number(assessment.external_timing_estimated_fps)}\n"
            "Raw video nominal FPS: "
            f"{self._format_number(assessment.raw_video_nominal_fps)}\n"
            "Mismatch factor: "
            f"{self._format_number(assessment.symmetric_fps_mismatch_factor)}x\n\n"
            "Warning:\n"
            "The selected timing units produce an estimated frame rate that "
            "differs substantially from the raw video frame rate.\n"
            "Check the timing units before preprocessing. Incorrect units may "
            "prevent video encoding."
        )
        self.warning_panel.setVisible(True)

    @staticmethod
    def _format_number(value: float | None) -> str:
        return "—" if value is None else f"{value:.8g}"

    @staticmethod
    def _format_length_match(value: bool | None) -> str:
        if value is None:
            return "Raw count unavailable"
        return "Yes" if value else "No"

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.success_label.clear()
        self._refresh_warning()
        self.error_message.emit(message)
        self.validity_changed.emit()

    def _show_unexpected(self, exc: BaseException) -> None:
        self.error_label.setText(str(exc))
        self.success_label.clear()
        self.unexpected_error.emit(str(exc))
        self.validity_changed.emit()
