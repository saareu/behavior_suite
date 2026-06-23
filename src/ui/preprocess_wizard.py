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
from ui.pages.crop_review_page import CropReviewPage
from ui.pages.encode_settings_page import EncodeSettingsPage
from ui.pages.project_page import ProjectPage
from ui.pages.raw_video_page import RawVideoPage
from ui.pages.run_validate_page import RunValidatePage
from ui.pages.timing_page import TimingPage
from ui.pages.trim_precrop_page import TrimPreCropPage
from ui.state import WORKFLOW_STEP_LABELS, WorkflowStep
from ui.tasks import GuiTaskRunner


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
            if index > WorkflowStep.CROP_REVIEW:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip("Complete the preceding workflow steps first.")
            self.navigation.addItem(item)
        self.navigation.setCurrentRow(0)

        self.pages = QStackedWidget()
        self.project_page = ProjectPage(controller)
        self.raw_video_page = RawVideoPage(controller)
        self.trim_pre_crop_page = TrimPreCropPage(controller)
        self.timing_page = TimingPage(controller)
        self.crop_review_page = CropReviewPage(controller)
        self.encode_settings_page = EncodeSettingsPage(controller)
        self.run_validate_page = RunValidatePage(controller)
        interactive_pages = (
            self.project_page,
            self.raw_video_page,
            self.trim_pre_crop_page,
            self.timing_page,
            self.crop_review_page,
            self.encode_settings_page,
            self.run_validate_page,
        )
        for page in interactive_pages:
            self.pages.addWidget(page)
            page.validity_changed.connect(self._update_buttons)
            page.status_message.connect(self._set_status)
            page.error_message.connect(self._set_error)
            page.unexpected_error.connect(self.unexpected_error.emit)
        self.project_page.project_changed.connect(self._refresh_probe_pages)
        self.raw_video_page.raw_probe_changed.connect(self._refresh_probe_pages)
        self.timing_page.raw_probe_changed.connect(self._refresh_probe_pages)

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

    def show_initial_error(self, message: str) -> None:
        """Display a startup configuration error without crashing the GUI."""

        self._set_error(message)

    def request_task_cancellation(self) -> None:
        """Request cooperative cancellation for tasks owned by wizard pages."""

        self.controller.cancel_all_tasks()
        for runner in self._task_runners():
            runner.request_cancellation()

    def has_running_tasks(self) -> bool:
        """Return whether any wizard page still owns a worker thread."""

        return any(runner.is_running for runner in self._task_runners())

    def _task_runners(self) -> tuple[GuiTaskRunner, ...]:
        return (
            self.raw_video_page.task_runner,
            self.timing_page.task_runner,
            self.crop_review_page.task_runner,
            self.run_validate_page.task_runner,
        )

    def _refresh_probe_pages(self) -> None:
        self.raw_video_page.on_activated()
        self.timing_page.refresh_from_state()
        self._update_buttons()

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
        if index == WorkflowStep.CROP_REVIEW:
            return self.crop_review_page
        if index == WorkflowStep.ENCODE_SETTINGS:
            return self.encode_settings_page
        if index == WorkflowStep.RUN_VALIDATE:
            return self.run_validate_page
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
        if index == WorkflowStep.TIMING:
            self._set_page(WorkflowStep.CROP_REVIEW)
            return
        if index == WorkflowStep.CROP_REVIEW:
            self._set_page(WorkflowStep.ENCODE_SETTINGS)
            self._set_status("Review the active encoding settings before running.")
            return
        if index == WorkflowStep.ENCODE_SETTINGS:
            self._set_page(WorkflowStep.RUN_VALIDATE)
            self._set_status("Review the final inputs and run preprocessing.")
            return
        self._set_page(index + 1)

    def _go_back(self) -> None:
        if self.controller.state.preprocess_task_running:
            self._set_error("Back navigation is disabled while preprocessing is active.")
            return
        index = self.pages.currentIndex()
        if index > 0:
            self.controller.request_current_input_task_cancellation()
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
        running = self.controller.state.preprocess_task_running
        self.navigation.setEnabled(not running)
        self.back_button.setEnabled(index > 0 and not running)
        self.next_button.setVisible(index != WorkflowStep.RUN_VALIDATE)
        self.next_button.setEnabled(
            not running
            and index <= WorkflowStep.ENCODE_SETTINGS
            and self.controller.can_advance(index)
        )
        encode_item = self.navigation.item(WorkflowStep.ENCODE_SETTINGS)
        run_item = self.navigation.item(WorkflowStep.RUN_VALIDATE)
        accepted = self.controller.state.accepted_crop_plan
        if accepted is not None and accepted.accepted_by_user:
            encode_item.setFlags(encode_item.flags() | Qt.ItemFlag.ItemIsEnabled)
            encode_item.setToolTip("Review the active validated encoding settings.")
        else:
            encode_item.setFlags(encode_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            encode_item.setToolTip("Accept a Crop Review candidate first.")
        if self.controller.can_advance(WorkflowStep.ENCODE_SETTINGS):
            run_item.setFlags(run_item.flags() | Qt.ItemFlag.ItemIsEnabled)
            run_item.setToolTip("Review inputs and run preprocessing.")
        else:
            run_item.setFlags(run_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            run_item.setToolTip("Complete Crop Review and Encode Settings first.")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_error(self, message: str) -> None:
        self.error_label.setText(message)
