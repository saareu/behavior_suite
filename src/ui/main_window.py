"""Main Behavior Suite desktop window with subsystem-level navigation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.pose_inference_controller import (
    PoseInferenceController,
    S3PoseInput,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.pages.pose_inference_page import PoseInferencePage
from ui.preprocess_wizard import PreprocessWizard


class MainWindow(QMainWindow):
    """Application shell preserving one session across subsystem screens."""

    def __init__(
        self,
        controller: PreprocessSetupController,
        pose_controller: PoseInferenceController | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.pose_controller = pose_controller or PoseInferenceController(
            settings=QSettings("BehaviorSuite", "BehaviorSuite")
        )
        self.setWindowTitle("behavior_suite")
        self.resize(1260, 900)
        self._session_root: Path | None = controller.state.project_dir
        self._shutdown_requested = False
        self._close_after_pose_task = False

        self.wizard = PreprocessWizard(controller)
        self.pose_page = PoseInferencePage(self.pose_controller)
        self.s3_placeholder = self._build_s3_placeholder()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.wizard)
        self.pages.addWidget(self.pose_page)
        self.pages.addWidget(self.s3_placeholder)

        self.s1_button = QPushButton("Subsystem 1 — Preprocess")
        self.s2_button = QPushButton("Subsystem 2 — Pose Inference")
        self.s1_button.clicked.connect(self.show_subsystem_1)
        self.s2_button.clicked.connect(self.open_pose_inference)
        navigation = QHBoxLayout()
        navigation.addWidget(self.s1_button)
        navigation.addWidget(self.s2_button)
        navigation.addStretch(1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(navigation)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(central)

        self.wizard.unexpected_error.connect(self._show_unexpected_error)
        self.wizard.session_changed.connect(self._remember_session)
        self.wizard.preprocessing_completed.connect(self._s1_completed)
        self.pose_page.back_to_s1_requested.connect(self.show_subsystem_1)
        self.pose_page.s3_handoff_requested.connect(self._show_s3_placeholder)
        self.pose_page.unexpected_error.connect(self._show_unexpected_error)
        self.pose_page.status_message.connect(self.statusBar().showMessage)
        self.pose_page.task_running_changed.connect(self._pose_task_running_changed)
        self.pose_page.task_finished.connect(self._pose_task_finished)
        self.statusBar().showMessage(
            "Select a project in S1 or open an existing session in S2."
        )

    def _build_s3_placeholder(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("Subsystem 3")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.s3_message = QLabel(
            "Subsystem 3 is not implemented yet. No identity or final-usability "
            "decision has been made."
        )
        self.s3_message.setWordWrap(True)
        self.s3_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.s3_back_button = QPushButton("Back to Pose Inference")
        self.s3_back_button.clicked.connect(self.open_pose_inference)
        layout.addWidget(title)
        layout.addWidget(self.s3_message)
        layout.addWidget(self.s3_back_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return page

    def show_startup_error(self, message: str) -> None:
        """Show a non-fatal startup configuration error in the S1 wizard."""

        self.wizard.show_initial_error(message)
        self.statusBar().showMessage("Startup configuration requires attention.")

    def show_subsystem_1(self) -> None:
        """Return to S1 without discarding session or S2 selection state."""

        if self.pose_page.has_running_task:
            self.statusBar().showMessage(
                "Navigation is disabled while pose inference is running."
            )
            return
        self.pages.setCurrentWidget(self.wizard)
        self._update_navigation()

    def open_pose_inference(self, session_root: object = None) -> None:
        """Open S2 from normal navigation or an explicit same-session handoff."""

        if self.pose_page.has_running_task:
            self.statusBar().showMessage(
                "Navigation is disabled while a Subsystem 2 task is active."
            )
            return
        if isinstance(session_root, Path | str) and str(session_root).strip():
            self._remember_session(session_root)
        if self._session_root is not None:
            self.pose_page.set_session(self._session_root)
        else:
            self.pose_page.refresh_from_state()
        self.pages.setCurrentWidget(self.pose_page)
        self._update_navigation()
        self.statusBar().showMessage("Subsystem 2 pose inference")

    def _remember_session(self, session_root: object) -> None:
        if isinstance(session_root, Path | str) and str(session_root).strip():
            self._session_root = Path(session_root).expanduser().resolve(strict=False)

    def _s1_completed(self, session_root: object) -> None:
        """Continue directly to S2 after success without starting inference."""

        self.open_pose_inference(session_root)

    def _show_s3_placeholder(self, handoff: object) -> None:
        if self.pose_page.has_running_task:
            self.statusBar().showMessage(
                "Cannot continue to Subsystem 3 while a Subsystem 2 task is active."
            )
            return
        if not isinstance(handoff, S3PoseInput):
            self._show_unexpected_error("Unexpected Subsystem 3 handoff type.")
            return
        self._session_root = handoff.session_root
        self.s3_message.setText(
            "Subsystem 3 is not implemented yet. The technically complete S2 run "
            "below is selected only as intended S3 input; identity correctness and "
            "final usability remain undecided.\n\n"
            f"Run: {handoff.selected_run_dir}\n"
            f"Mode: {handoff.inference_mode or 'unknown'}\n"
            f"QC: {handoff.qc_outcome or 'unknown'}"
        )
        self.pages.setCurrentWidget(self.s3_placeholder)
        self._update_navigation()

    def _update_navigation(self) -> None:
        running = self.pose_page.has_running_task
        self.s1_button.setEnabled(not running)
        self.s2_button.setEnabled(not running)
        self.s3_back_button.setEnabled(not running)

    def _pose_task_running_changed(self, _running: bool) -> None:
        self._update_navigation()

    def _pose_task_finished(self) -> None:
        self._update_navigation()
        if self._close_after_pose_task:
            self._close_after_pose_task = False
            self.close()

    def _show_unexpected_error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Unexpected Error",
            "An unexpected error occurred. Detailed information was preserved "
            f"for future logging.\n\n{message}",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Wait for non-cancellable S2 work or cooperatively cancel S1 work."""

        if self.pose_page.has_running_task:
            event.ignore()
            self._close_after_pose_task = True
            self.statusBar().showMessage(
                "A Subsystem 2 task is active and has no safe cancellation hook. "
                "The window remains responsive and will close when the task finishes."
            )
            return
        if self.wizard.has_running_tasks():
            if not self._shutdown_requested:
                self._shutdown_requested = True
                self.wizard.request_task_cancellation()
                self.statusBar().showMessage(
                    "Cancelling background preprocessing work before closing…"
                )
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        super().closeEvent(event)
