"""Lazy PySide6 application launcher."""

from __future__ import annotations

import sys
from pathlib import Path


class GuiDependencyError(RuntimeError):
    """Raised when optional desktop GUI dependencies are unavailable."""


GUI_INSTALL_GUIDANCE = 'GUI support is not installed. Install it with:\npip install -e ".[dev,gui]"'


def launch_gui(config_path: Path | None = None) -> int:
    """Create and run the PySide6 desktop application.

    PySide6 and all Qt widget modules are imported lazily so the base CLI and
    headless controller tests do not require GUI dependencies.

    Raises:
        GuiDependencyError: If PySide6 cannot be imported.
    """

    try:
        from PySide6.QtWidgets import QApplication

        from ui.main_window import MainWindow
    except (ImportError, ModuleNotFoundError) as exc:
        raise GuiDependencyError(GUI_INSTALL_GUIDANCE) from exc

    from ui.controllers.preprocess_setup_controller import (
        PreprocessSetupController,
        SetupValidationError,
    )

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication(sys.argv[:1])
    controller = PreprocessSetupController()
    startup_error: str | None = None
    try:
        controller.load_startup_config(config_path)
    except SetupValidationError as exc:
        startup_error = str(exc)
    except Exception as exc:
        controller.record_unexpected_error(exc)
        startup_error = "An unexpected startup error occurred."
    window = MainWindow(controller)
    if startup_error is not None:
        window.show_startup_error(startup_error)
    window.show()
    if owns_application:
        return application.exec()
    return 0
