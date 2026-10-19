"""Main Behavior Suite desktop window with subsystem-level navigation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.pose_finalization_controller import PoseFinalizationController
from ui.controllers.pose_inference_controller import (
    PoseInferenceController,
    S3PoseInput,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.tracking_review_controller import TrackingReviewController
from ui.pages.pose_finalization_page import PoseFinalizationPage
from ui.pages.pose_inference_page import PoseInferencePage
from ui.pages.tracking_review_page import TrackingReviewPage
from ui.preprocess_wizard import PreprocessWizard


class MainWindow(QMainWindow):
    """Application shell preserving one session across subsystem screens."""

    def __init__(
        self,
        controller: PreprocessSetupController,
        pose_controller: PoseInferenceController | None = None,
        review_controller: TrackingReviewController | None = None,
        finalization_controller: PoseFinalizationController | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.pose_controller = pose_controller or PoseInferenceController(
            settings=QSettings("BehaviorSuite", "BehaviorSuite")
        )
        self.review_controller = review_controller or TrackingReviewController()
        self.finalization_controller = (
            finalization_controller or PoseFinalizationController()
        )
        self.setWindowTitle("behavior_suite")
        self.resize(1260, 900)
        self._session_root: Path | None = controller.state.project_dir
        self._shutdown_requested = False
        self._close_after_pose_task = False
        self._close_after_s4_task = False

        self.wizard = PreprocessWizard(controller)
        self.pose_page = PoseInferencePage(self.pose_controller)
        self.review_page = TrackingReviewPage(self.review_controller)
        self.finalization_page = PoseFinalizationPage(self.finalization_controller)
        self.s3_shell = self._build_s3_shell()
        self.s4_shell = self._build_s4_shell()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.wizard)
        self.pages.addWidget(self.pose_page)
        self.pages.addWidget(self.s3_shell)
        self.pages.addWidget(self.s4_shell)

        self.s1_button = QPushButton("Subsystem 1 — Preprocess")
        self.s2_button = QPushButton("Subsystem 2 — Pose Inference")
        self.s3_button = QPushButton("Subsystem 3 — Tracking Review")
        self.s4_button = QPushButton("Subsystem 4 — Pose Finalization")
        self.s1_button.clicked.connect(self.show_subsystem_1)
        self.s2_button.clicked.connect(self.open_pose_inference)
        self.s3_button.clicked.connect(self.open_tracking_review)
        self.s4_button.clicked.connect(self.open_pose_finalization)
        navigation = QHBoxLayout()
        navigation.addWidget(self.s1_button)
        navigation.addWidget(self.s2_button)
        navigation.addWidget(self.s3_button)
        navigation.addWidget(self.s4_button)
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
        self.pose_page.s3_handoff_requested.connect(self._show_s3_from_s2)
        self.pose_page.unexpected_error.connect(self._show_unexpected_error)
        self.pose_page.status_message.connect(self.statusBar().showMessage)
        self.pose_page.task_running_changed.connect(self._pose_task_running_changed)
        self.pose_page.task_finished.connect(self._pose_task_finished)
        self.review_page.unexpected_error.connect(self._show_unexpected_error)
        self.review_page.status_message.connect(self.statusBar().showMessage)
        self.review_page.accepted.connect(self._s3_accepted)
        self.finalization_page.unexpected_error.connect(self._show_unexpected_error)
        self.finalization_page.status_message.connect(self.statusBar().showMessage)
        self.finalization_page.task_running_changed.connect(
            self._s4_task_running_changed
        )
        self.finalization_page.task_finished.connect(self._s4_task_finished)
        self.statusBar().showMessage(
            "Select a project in S1 or open an existing session in S2."
        )

    def _build_s3_shell(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        title = QLabel("Subsystem 3 — Tracking Review")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.open_s3_button = QPushButton("Open project/session…")
        self.open_s3_button.clicked.connect(self._browse_s3_project)
        self.s3_back_button = QPushButton("Back to Pose Inference")
        self.s3_back_button.clicked.connect(self.open_pose_inference)
        self.s3_continue_s4_button = QPushButton("Continue to Pose Finalization")
        self.s3_continue_s4_button.clicked.connect(self.open_pose_finalization)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.open_s3_button)
        header.addWidget(self.s3_back_button)
        header.addWidget(self.s3_continue_s4_button)
        self.s3_message = QLabel(
            "Open a project/session directory (containing preprocess/, "
            "pose_inference/, and tracking_correction/). Completed S3 runs are "
            "discovered automatically; when several exist, choose one in the run "
            "selector. The resolved S3 output path is shown for provenance. "
            "After explicit acceptance, continue to Subsystem 4 for dense "
            "finalization. Advanced CLI: python -m tracking_correction review "
            "--s3-run <run_dir>"
        )
        self.s3_message.setWordWrap(True)
        self.s3_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addLayout(header)
        layout.addWidget(self.s3_message)
        layout.addWidget(self.review_page, 1)
        return page

    def _build_s4_shell(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        title = QLabel("Subsystem 4 — Pose Finalization")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.open_s4_button = QPushButton("Open project/session…")
        self.open_s4_button.clicked.connect(self._browse_s4_project)
        self.s4_back_button = QPushButton("Back to Tracking Review")
        self.s4_back_button.clicked.connect(self.open_tracking_review)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.open_s4_button)
        header.addWidget(self.s4_back_button)
        self.s4_message = QLabel(
            "Open a project/session with an accepted Subsystem 3 result. "
            "S4 automatically selects the latest eligible accepted S3 run, "
            "builds dense 120 Hz pose from authoritative S1 timing, and writes "
            "final_pose.parquet. There is no S4 visual review or acceptance step."
        )
        self.s4_message.setWordWrap(True)
        self.s4_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addLayout(header)
        layout.addWidget(self.s4_message)
        layout.addWidget(self.finalization_page, 1)
        return page

    def show_startup_error(self, message: str) -> None:
        """Show a non-fatal startup configuration error in the S1 wizard."""

        self.wizard.show_initial_error(message)
        self.statusBar().showMessage("Startup configuration requires attention.")

    def show_subsystem_1(self) -> None:
        """Return to S1 without discarding session or S2 selection state."""

        if self._navigation_blocked():
            return
        self.pages.setCurrentWidget(self.wizard)
        self._update_navigation()

    def open_pose_inference(self, session_root: object = None) -> None:
        """Open S2 from normal navigation or an explicit same-session handoff."""

        if self._navigation_blocked():
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

    def open_tracking_review(self) -> None:
        """Show the S3 review workspace."""

        if self._navigation_blocked():
            return
        self.pages.setCurrentWidget(self.s3_shell)
        self._update_navigation()
        self.statusBar().showMessage("Subsystem 3 tracking review")

    def open_pose_finalization(self, session_root: object = None) -> None:
        """Show the S4 finalization workspace for the current or provided session."""

        if self._navigation_blocked():
            return
        if isinstance(session_root, Path | str) and str(session_root).strip():
            self._remember_session(session_root)
        if self._session_root is not None:
            self.finalization_page.set_session(self._session_root)
        else:
            self.finalization_page.refresh_from_state()
        self.pages.setCurrentWidget(self.s4_shell)
        self._update_navigation()
        self.statusBar().showMessage("Subsystem 4 pose finalization")

    def _remember_session(self, session_root: object) -> None:
        if isinstance(session_root, Path | str) and str(session_root).strip():
            self._session_root = Path(session_root).expanduser().resolve(strict=False)

    def _s1_completed(self, session_root: object) -> None:
        """Continue directly to S2 after success without starting inference."""

        self.open_pose_inference(session_root)

    def _show_s3_from_s2(self, handoff: object) -> None:
        if self._navigation_blocked():
            return
        if not isinstance(handoff, S3PoseInput):
            self._show_unexpected_error("Unexpected Subsystem 3 handoff type.")
            return
        self._session_root = handoff.session_root
        s3_run = handoff.existing_s3_run_dir
        if s3_run is None:
            self.s3_message.setText(
                "S2 handoff received, but no completed Subsystem 3 run is available yet.\n\n"
                f"Project: {handoff.session_root}\n"
                f"S2 run: {handoff.selected_run_dir}"
            )
            self.open_tracking_review()
            return
        self.s3_message.setText(
            "Opened Subsystem 3 from the selected S2 run.\n\n"
            f"Project: {handoff.session_root}\n"
            f"S2 run: {handoff.selected_run_dir}\n"
            f"S3 run: {s3_run}\n"
            f"Mode: {handoff.inference_mode or 'unknown'}\n"
            f"QC: {handoff.qc_outcome or 'unknown'}"
        )
        self.open_tracking_review()
        self.review_page.open_run(s3_run)

    def _s3_accepted(self, result: object) -> None:
        """Keep session context after S3 acceptance for an optional S4 handoff."""

        session = getattr(result, "session_root", None)
        if isinstance(session, Path | str) and str(session).strip():
            self._remember_session(session)
        elif self.review_controller.session is not None:
            self._remember_session(self.review_controller.session.session_root)
        tracked = getattr(result, "tracked_pose_path", None)
        self.s3_message.setText(
            "Tracking accepted. Continue to Subsystem 4 to finalize dense pose, "
            "or keep reviewing.\n\n"
            f"Project: {self._session_root}\n"
            f"Accepted pose: {tracked or 'tracked_pose.parquet'}"
        )
        self.statusBar().showMessage(
            "S3 accepted. Subsystem 4 can now discover this accepted run."
        )

    def _browse_s3_project(self) -> None:
        if self._navigation_blocked():
            return
        start = str(self._session_root) if self._session_root is not None else ""
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if not chosen:
            return
        root = Path(chosen)
        self._remember_session(root)
        self.review_page.open_path(root)

    def _browse_s4_project(self) -> None:
        if self._navigation_blocked():
            return
        start = str(self._session_root) if self._session_root is not None else ""
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if not chosen:
            return
        root = Path(chosen)
        self._remember_session(root)
        self.finalization_page.set_session(root)

    def _navigation_blocked(self) -> bool:
        if self.pose_page.has_running_task:
            self.statusBar().showMessage(
                "Navigation is disabled while a Subsystem 2 task is active."
            )
            return True
        if self.finalization_page.has_running_task:
            self.statusBar().showMessage(
                "Navigation is disabled while pose finalization is running."
            )
            return True
        return False

    def _update_navigation(self) -> None:
        running = self.pose_page.has_running_task or self.finalization_page.has_running_task
        self.s1_button.setEnabled(not running)
        self.s2_button.setEnabled(not running)
        self.s3_button.setEnabled(not running)
        self.s4_button.setEnabled(not running)
        self.s3_back_button.setEnabled(not running)
        self.open_s3_button.setEnabled(not running)
        self.s3_continue_s4_button.setEnabled(not running)
        self.s4_back_button.setEnabled(not running)
        self.open_s4_button.setEnabled(not running)

    def _pose_task_running_changed(self, _running: bool) -> None:
        self._update_navigation()

    def _pose_task_finished(self) -> None:
        self._update_navigation()
        if self._close_after_pose_task:
            self._close_after_pose_task = False
            self.close()

    def _s4_task_running_changed(self, _running: bool) -> None:
        self._update_navigation()

    def _s4_task_finished(self) -> None:
        self._update_navigation()
        if self._close_after_s4_task:
            self._close_after_s4_task = False
            self.close()

    def _show_unexpected_error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Unexpected Error",
            "An unexpected error occurred. Detailed information was preserved "
            f"for future logging.\n\n{message}",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Wait for non-cancellable S2/S4 work or cooperatively cancel S1 work."""

        if self.pose_page.has_running_task:
            event.ignore()
            self._close_after_pose_task = True
            self.statusBar().showMessage(
                "A Subsystem 2 task is active and has no safe cancellation hook. "
                "The window remains responsive and will close when the task finishes."
            )
            return
        if self.finalization_page.has_running_task:
            event.ignore()
            self._close_after_s4_task = True
            self.statusBar().showMessage(
                "Pose finalization is active and has no safe cancellation hook. "
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
