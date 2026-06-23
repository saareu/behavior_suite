"""Project creation/opening and configuration selection page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)


class ProjectPage(QWidget):
    """Collect project and configuration inputs through the setup controller."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)
    project_changed = Signal()

    def __init__(self, controller: PreprocessSetupController) -> None:
        super().__init__()
        self.controller = controller

        title = QLabel("Project Setup")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.config_path = QLineEdit()
        self.config_path.setReadOnly(True)
        config_browse = QPushButton("Choose Configuration YAML")
        config_browse.clicked.connect(self._choose_config)
        config_row = QHBoxLayout()
        config_row.addWidget(self.config_path, 1)
        config_row.addWidget(config_browse)

        create_group = QGroupBox("Create new project")
        self.parent_directory = QLineEdit()
        parent_browse = QPushButton("Browse Parent Directory")
        parent_browse.clicked.connect(self._browse_parent)
        parent_row = QHBoxLayout()
        parent_row.addWidget(self.parent_directory, 1)
        parent_row.addWidget(parent_browse)
        self.project_name = QLineEdit()
        create_button = QPushButton("Create Project")
        create_button.clicked.connect(self._create_project)
        create_form = QFormLayout(create_group)
        create_form.addRow("Parent directory", parent_row)
        create_form.addRow("Project name", self.project_name)
        create_form.addRow(create_button)

        open_group = QGroupBox("Open existing project")
        self.existing_directory = QLineEdit()
        existing_browse = QPushButton("Browse Project Directory")
        existing_browse.clicked.connect(self._browse_existing)
        existing_row = QHBoxLayout()
        existing_row.addWidget(self.existing_directory, 1)
        existing_row.addWidget(existing_browse)
        open_button = QPushButton("Open Project")
        open_button.clicked.connect(self._open_project)
        open_form = QFormLayout(open_group)
        open_form.addRow("Project directory", existing_row)
        open_form.addRow(open_button)

        summary_group = QGroupBox("Selected project")
        self.summary_name = QLabel("—")
        self.summary_directory = QLabel("—")
        self.summary_directory.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.summary_preprocess = QLabel("—")
        self.summary_preprocess.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.summary_prior_run = QLabel("—")
        self.summary_prior_run.setWordWrap(True)
        summary_form = QFormLayout(summary_group)
        summary_form.addRow("Project name", self.summary_name)
        summary_form.addRow("Project directory", self.summary_directory)
        summary_form.addRow("Preprocess directory", self.summary_preprocess)
        summary_form.addRow("Prior preprocessing context", self.summary_prior_run)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #b00020;")
        self.error_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(QLabel("Validated configuration"))
        layout.addLayout(config_row)
        layout.addWidget(create_group)
        layout.addWidget(open_group)
        layout.addWidget(summary_group)
        layout.addWidget(self.error_label)
        layout.addStretch(1)
        self.refresh_from_state()

    def refresh_from_state(self) -> None:
        """Refresh read-only labels from shared setup state."""

        state = self.controller.state
        self.config_path.setText(str(state.config_path) if state.config_path else "Typed defaults")
        if state.project is None:
            self.summary_name.setText("—")
            self.summary_directory.setText("—")
            self.summary_preprocess.setText("—")
            self.summary_prior_run.setText("—")
        else:
            self.summary_name.setText(state.project.name)
            self.summary_directory.setText(str(state.project_dir))
            self.summary_preprocess.setText(str(state.preprocess_dir))
            self.summary_prior_run.setText(
                state.project_hydration_message or "No prior context loaded."
            )
        self.validity_changed.emit()

    def is_valid(self) -> bool:
        """Return whether a validated project is selected."""

        return self.controller.state.project is not None

    def commit(self) -> bool:
        """Validate that project selection permits forward navigation."""

        try:
            self.controller.validate_step(0)
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
            return False
        return True

    def _choose_config(self) -> None:
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose preprocess configuration",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml)",
        )
        if not filename:
            return
        try:
            self.controller.load_config(Path(filename))
            self.error_label.clear()
            self.refresh_from_state()
            self.status_message.emit("Configuration loaded and validated.")
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._show_unexpected(exc)

    def _browse_parent(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose parent directory")
        if directory:
            self.parent_directory.setText(directory)

    def _browse_existing(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose project directory")
        if directory:
            self.existing_directory.setText(directory)

    def _create_project(self) -> None:
        try:
            self.controller.create_project(
                Path(self.parent_directory.text()),
                self.project_name.text(),
            )
            self.error_label.clear()
            self.refresh_from_state()
            self.project_changed.emit()
            self.status_message.emit(
                self.controller.state.project_hydration_message
                or "Project created successfully."
            )
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._show_unexpected(exc)

    def _open_project(self) -> None:
        try:
            self.controller.open_project(Path(self.existing_directory.text()))
            self.error_label.clear()
            self.refresh_from_state()
            self.project_changed.emit()
            self.status_message.emit(
                self.controller.state.project_hydration_message
                or "Project opened successfully."
            )
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._show_unexpected(exc)

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_message.emit(message)
        self.validity_changed.emit()

    def _show_unexpected(self, exc: BaseException) -> None:
        self.controller.record_unexpected_error(exc)
        self.error_label.setText("An unexpected error occurred.")
        self.unexpected_error.emit(str(exc))
