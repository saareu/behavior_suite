"""Launch helper for the Subsystem 03 tracking review workspace."""

from __future__ import annotations

import sys
from pathlib import Path

from tracking_correction.contracts import TrackingCorrectionError


class GuiDependencyError(RuntimeError):
    """Raised when optional desktop GUI dependencies are unavailable."""


GUI_INSTALL_GUIDANCE = 'GUI support is not installed. Install it with:\npip install -e ".[dev,gui]"'


def launch_tracking_review(s3_run_dir: Path) -> int:
    """Open the PySide6 tracking review workspace for one S3 run."""

    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

        from ui.pages.tracking_review_page import TrackingReviewPage
    except (ImportError, ModuleNotFoundError) as exc:
        raise GuiDependencyError(GUI_INSTALL_GUIDANCE) from exc

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv[:1])

    window = QMainWindow()
    window.setWindowTitle("behavior_suite — tracking review")
    window.resize(1100, 820)
    page = TrackingReviewPage()
    window.setCentralWidget(page)
    page.status_message.connect(window.statusBar().showMessage)
    page.unexpected_error.connect(
        lambda message: QMessageBox.critical(window, "Unexpected Error", message)
    )
    window.show()
    try:
        page.open_run(Path(s3_run_dir))
    except TrackingCorrectionError as exc:
        QMessageBox.critical(window, "Cannot open S3 run", str(exc))
        if owns_application:
            return 1
        raise
    if owns_application:
        return application.exec()
    return 0
