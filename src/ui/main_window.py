"""Main desktop window for preprocessing setup."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QMessageBox

from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.preprocess_wizard import PreprocessWizard


class MainWindow(QMainWindow):
    """Application shell containing the guarded preprocessing wizard."""

    def __init__(self, controller: PreprocessSetupController) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("behavior_suite — Video Preprocessing")
        self.resize(1180, 820)
        self.wizard = PreprocessWizard(controller)
        self.wizard.unexpected_error.connect(self._show_unexpected_error)
        self.setCentralWidget(self.wizard)
        self.statusBar().showMessage(
            "Full preprocessing setup, explicit crop acceptance, and validated Run are available."
        )

    def show_startup_error(self, message: str) -> None:
        """Show a non-fatal startup configuration error in the window."""

        self.wizard.show_initial_error(message)
        self.statusBar().showMessage("Startup configuration requires attention.")

    def _show_unexpected_error(self, message: str) -> None:
        QMessageBox.critical(
            self,
            "Unexpected Error",
            "An unexpected error occurred. Detailed information was preserved "
            f"for future logging.\n\n{message}",
        )
