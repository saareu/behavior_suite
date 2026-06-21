"""Stacked preprocessing setup workflow with guarded navigation."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)
from ui.pages.project_page import ProjectPage
from ui.pages.raw_video_page import RawVideoPage
from ui.pages.timing_page import TimingPage
from ui.pages.trim_precrop_page import TrimPreCropPage
from ui.state import WORKFLOW_STEP_LABELS, WorkflowStep


class PreprocessWizard(QWidget):
    """Render workflow steps while the controller owns validation state."""

    unexpected_error = Signal(str)

    def __init__(self, controller: PreprocessSetupController) -> None:
        super().__init__()
        self.controller = controller
        self.navigation = QListWidget()
        self.navigation.setFixedWidth(190)
        for index, label in enumerate(WORKFLOW_STEP_LABELS):
            item = QListWidgetItem(f"{index + 1}. {label}")
            if index > WorkflowStep.TIMING:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip("Not implemented in this GUI milestone.")
            self.navigation.addItem(item)
        self.navigation.setCurrentRow(0)

        self.pages = QStackedWidget()
        self.project_page = ProjectPage(controller)
        self.raw_video_page = RawVideoPage(controller)
        self.trim_pre_crop_page = TrimPreCropPage(controller)
        self.timing_page = TimingPage(controller)
        interactive_pages = (
            self.project_page,
            self.raw_video_page,
            self.trim_pre_crop_page,
            self.timing_page,
        )
        for page in interactive_pages:
            self.pages.addWidget(page)
            page.validity_changed.connect(self._update_buttons)
            page.status_message.connect(self._set_status)
            page.error_message.connect(self._set_error)
            page.unexpected_error.connect(self.unexpected_error.emit)
        for label in WORKFLOW_STEP_LABELS[4:]:
            self.pages.addWidget(self._placeholder_page(label))

        self.status_label = QLabel("Select or create a project.")
        self.status_label.setWordWrap(True)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._go_next)
        button_row = QHBoxLayout()
        button_row.addWidget(self.back_button)
        button_row.addStretch(1)
        button_row.addWidget(self.next_button)

        central_layout = QVBoxLayout()
        central_layout.addWidget(self.pages, 1)
        central_layout.addWidget(self.status_label)
        central_layout.addWidget(self.error_label)
        central_layout.addLayout(button_row)
        layout = QHBoxLayout(self)
        layout.addWidget(self.navigation)
        layout.addLayout(central_layout, 1)
        self._update_buttons()

    @staticmethod
    def _placeholder_page(label: str) -> QWidget:
        page = QWidget()
        message = QLabel(f"{label}\n\nNot implemented in this GUI milestone.")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setStyleSheet("font-size: 16px; color: #666;")
        layout = QVBoxLayout(page)
        layout.addWidget(message, 1)
        return page

    def show_initial_error(self, message: str) -> None:
        """Display a startup configuration error without crashing the GUI."""

        self._set_error(message)

    def _current_interactive_page(self):
        index = self.pages.currentIndex()
        if index == WorkflowStep.PROJECT:
            return self.project_page
        if index == WorkflowStep.RAW_VIDEO:
            return self.raw_video_page
        if index == WorkflowStep.TRIM_PRE_CROP:
            return self.trim_pre_crop_page
        if index == WorkflowStep.TIMING:
            return self.timing_page
        return None

    def _go_next(self) -> None:
        index = self.pages.currentIndex()
        page = self._current_interactive_page()
        if page is None or not page.commit():
            self._set_error(
                self.controller.state.last_validation_error
                or "Complete the current page before continuing."
            )
            return
        try:
            self.controller.validate_step(index)
        except SetupValidationError as exc:
            self._set_error(str(exc))
            return
        if index >= WorkflowStep.TIMING:
            self._set_page(WorkflowStep.CROP_REVIEW)
            self._set_status("Not implemented in this GUI milestone.")
            return
        self._set_page(index + 1)

    def _go_back(self) -> None:
        index = self.pages.currentIndex()
        if index > 0:
            self._set_page(index - 1)

    def _set_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.navigation.setCurrentRow(index)
        self.controller.state.current_step = WorkflowStep(index)
        self.error_label.clear()
        page = self._current_interactive_page()
        if page is not None and hasattr(page, "on_activated"):
            page.on_activated()
        self._update_buttons()

    def _update_buttons(self) -> None:
        index = self.pages.currentIndex()
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(
            index <= WorkflowStep.TIMING and self.controller.can_advance(index)
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_error(self, message: str) -> None:
        self.error_label.setText(message)
