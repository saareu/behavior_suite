"""Optional MATLAB external-timing selection page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
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

from preprocess.exceptions import PreprocessError
from preprocess.models import TimingUnit
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.timing_controller import (
    TimingController,
    TimingUnexpectedError,
    TimingValidationError,
)
from ui.state import TimingMode
from ui.tasks import GuiTaskRunner, TaskAlreadyRunningError

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

    def __init__(self, setup_controller: PreprocessSetupController) -> None:
        super().__init__()
        self.setup_controller = setup_controller
        self.controller = TimingController(setup_controller.state)
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
        raw_count_explanation = QLabel(
            "External timing requires an exact full sequential count of readable "
            "raw frames. Counting may take time for a long video."
        )
        raw_count_explanation.setWordWrap(True)
        raw_count_explanation.setStyleSheet("color: #8a5a00;")
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
        mat_layout.addWidget(raw_count_explanation)
        mat_layout.addWidget(self.count_button)
        mat_layout.addWidget(self.busy_indicator)
        mat_layout.addWidget(self.validate_button)

        self.success_label = QLabel()
        self.success_label.setWordWrap(True)
        self.success_label.setStyleSheet("color: #176b2c;")
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(mode_row)
        layout.addWidget(self.mat_group, 1)
        layout.addWidget(self.success_label)
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
        try:
            path = self.controller.prepare_full_raw_frame_count()
            self._set_busy(True)
            self.task_runner.start(
                lambda: self.controller.compute_full_raw_frame_count(path),
                task_name="sequential raw-frame count",
                on_success=self._raw_count_succeeded,
                on_error=self._raw_count_failed,
                on_finished=lambda: self._set_busy(False),
            )
        except (TimingValidationError, TaskAlreadyRunningError) as exc:
            self._set_busy(False)
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._set_busy(False)
            self._show_unexpected(exc)

    def _raw_count_succeeded(self, result: object) -> None:
        try:
            count = self.controller.apply_full_raw_frame_count(result)
        except TimingValidationError as exc:
            self._show_expected_error(str(exc))
            return
        except Exception as exc:
            self._show_unexpected(exc)
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit(f"Sequential readable-frame count completed: {count}.")
        self.validity_changed.emit()

    def _raw_count_failed(self, exc: BaseException) -> None:
        self.controller.record_full_raw_frame_count_failure(exc)
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

    def _set_busy(self, busy: bool) -> None:
        self.busy_indicator.setVisible(busy)
        self.no_timing.setEnabled(not busy)
        self.matlab_timing.setEnabled(not busy)
        self.load_button.setEnabled(not busy)
        self.count_button.setEnabled(not busy)
        self.validate_button.setEnabled(not busy)
        self.candidate_table.setEnabled(not busy)
        self.units.setEnabled(not busy)

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

    def _refresh_raw_count(self) -> None:
        probe = self.controller.state.raw_probe
        readable_count = (
            probe.frame_count_opencv_readable if probe is not None else None
        )
        self.raw_count_status.setText(
            "Raw OpenCV-readable frame count: "
            + (str(readable_count) if readable_count is not None else "Not counted")
        )

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
        self.error_message.emit(message)
        self.validity_changed.emit()

    def _show_unexpected(self, exc: BaseException) -> None:
        self.error_label.setText(str(exc))
        self.success_label.clear()
        self.unexpected_error.emit(str(exc))
        self.validity_changed.emit()
