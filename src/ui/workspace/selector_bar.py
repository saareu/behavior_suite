"""Compact workspace selector bar for the application shell (D0-D1)."""

from __future__ import annotations

import contextlib
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ui.workspace.rebind_dialog import RebindPreviewDialog
from ui.workspace.registration_dialog import RegistrationPreviewDialog
from ui.workspace.shell_controller import WorkspaceShellController
from workspace.errors import (
    WorkspaceError,
    WorkspaceNavigationBlocked,
    WorkspaceValidationError,
)
from workspace.migrate import RebindPreview, RegistrationPreview
from workspace.models import RecentTargetEntry


class WorkspaceSelectorBar(QWidget):
    """Browse / open / register / recent / rebind controls for shared workspace."""

    session_activated = Signal(object)  # Path | None discovery root

    def __init__(
        self,
        controller: WorkspaceShellController,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller

        self._status = QLabel("Workspace: none")
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._busy = QLabel("")
        self._recent = QComboBox()
        self._recent.setMinimumWidth(280)
        self._recent.setToolTip("Remember-only recent sessions (not auto-opened at launch)")

        self._browse_button = QPushButton("Browse…")
        self._browse_button.setToolTip("Open a session directory without registration")
        self._register_button = QPushButton("Register…")
        self._register_button.setToolTip(
            "Background discovery + registration preview (catalog write only on confirm)"
        )
        self._rescan_button = QPushButton("Rescan")
        self._rescan_button.setToolTip("Rescan the bound catalog (preview then confirm)")
        self._rebind_button = QPushButton("Rebind…")
        self._rebind_button.setToolTip("Relocate a missing registered session path")
        self._open_recent_button = QPushButton("Open recent")
        self._clear_button = QPushButton("Clear")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Workspace"))
        row.addWidget(self._recent, 1)
        row.addWidget(self._open_recent_button)
        row.addWidget(self._browse_button)
        row.addWidget(self._register_button)
        row.addWidget(self._rescan_button)
        row.addWidget(self._rebind_button)
        row.addWidget(self._clear_button)
        row.addWidget(self._busy)
        row.addWidget(self._status, 2)

        self._browse_button.clicked.connect(self._on_browse)
        self._register_button.clicked.connect(self._on_register)
        self._rescan_button.clicked.connect(self._on_rescan)
        self._rebind_button.clicked.connect(self._on_rebind)
        self._open_recent_button.clicked.connect(self._on_open_recent)
        self._clear_button.clicked.connect(self._on_clear)

        self._controller.workspace_changed.connect(self._on_workspace_changed)
        self._controller.background_busy_changed.connect(self._on_busy)
        self._controller.registration_preview_ready.connect(self._on_registration_preview)
        self._controller.rebind_preview_ready.connect(self._on_rebind_preview)
        self._controller.error_message.connect(self._on_error)
        self._controller.status_message.connect(self._on_status)

        self.refresh_recent()
        self._refresh_status()

    def refresh_recent(self) -> None:
        self._recent.blockSignals(True)
        self._recent.clear()
        self._recent.addItem("(recent sessions — not auto-opened)", None)
        for entry in self._controller.recent_targets:
            label = self._format_recent(entry)
            self._recent.addItem(label, entry)
        self._recent.blockSignals(False)

    def _format_recent(self, entry: RecentTargetEntry) -> str:
        missing = not entry.session_root.is_dir()
        incomplete = not (entry.session_root / "preprocess").is_dir()
        flags: list[str] = []
        if missing:
            flags.append("missing")
        elif incomplete:
            flags.append("incomplete")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"{entry.display_name} — {entry.session_root}{suffix}"

    def _refresh_status(self) -> None:
        self._status.setText(self._controller.status_line())
        registered = self._controller.registration_status().registered
        self._rescan_button.setEnabled(registered or self._controller.catalog is not None)
        self._rebind_button.setEnabled(self._controller.catalog is not None)
        busy = self._controller.is_background_busy
        self._browse_button.setEnabled(not busy)
        self._register_button.setEnabled(not busy)
        self._open_recent_button.setEnabled(not busy)
        self._clear_button.setEnabled(not busy)

    def _on_workspace_changed(self, _snap: object) -> None:
        self.refresh_recent()
        self._refresh_status()
        self.session_activated.emit(self._controller.discovery_root)

    def _on_busy(self, busy: bool) -> None:
        self._busy.setText("Scanning…" if busy else "")
        self._refresh_status()

    def _on_status(self, message: str) -> None:
        if message:
            self._status.setToolTip(message)

    def _on_error(self, title: str, detail: str) -> None:
        QMessageBox.warning(self, title, detail)

    def _on_browse(self) -> None:
        start = ""
        if self._controller.discovery_root is not None:
            start = str(self._controller.discovery_root)
        chosen = QFileDialog.getExistingDirectory(
            self, "Open session directory (without registration)", start
        )
        if not chosen:
            return
        try:
            self._controller.open_session_path(chosen, require_preprocess=False)
        except WorkspaceNavigationBlocked as exc:
            QMessageBox.information(self, "Workspace switch blocked", str(exc))
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Open failed", str(exc))

    def _on_register(self) -> None:
        start = ""
        if self._controller.discovery_root is not None:
            start = str(self._controller.discovery_root)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select folder to register", start
        )
        if not chosen:
            return
        try:
            self._controller.begin_registration_preview(chosen)
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Registration failed", str(exc))

    def _on_rescan(self) -> None:
        try:
            self._controller.begin_rescan()
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Rescan failed", str(exc))

    def _on_rebind(self) -> None:
        catalog = self._controller.catalog
        if catalog is None:
            QMessageBox.information(
                self,
                "No catalog",
                "Bind or register a catalog before rebinding a session path.",
            )
            return
        missing = [
            session
            for session in catalog.sessions
            if session.path_status.value == "missing" or not session.session_root.is_dir()
        ]
        sessions = missing or list(catalog.sessions)
        # Prefer missing; otherwise allow explicit relocate of current.
        session = sessions[0]
        if len(sessions) > 1:
            names = [f"{s.display_name} ({s.session_root})" for s in sessions]
            name, ok = QInputDialog.getItem(
                self,
                "Select session to rebind",
                "Session:",
                names,
                0,
                False,
            )
            if not ok:
                return
            session = sessions[names.index(name)]

        chosen = QFileDialog.getExistingDirectory(
            self,
            f"Browse replacement directory for {session.display_name}",
            str(session.session_root.parent if session.session_root.parent.exists() else ""),
        )
        if not chosen:
            return
        try:
            self._controller.begin_rebind_preview(
                session_id=session.session_id,
                new_session_root=chosen,
            )
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Rebind preview failed", str(exc))

    def _on_open_recent(self) -> None:
        entry = self._recent.currentData()
        if not isinstance(entry, RecentTargetEntry):
            QMessageBox.information(
                self,
                "No recent selection",
                "Choose a recent session from the list. Launch never auto-opens.",
            )
            return
        if not entry.session_root.is_dir():
            answer = QMessageBox.question(
                self,
                "Missing path",
                f"Path is missing:\n{entry.session_root}\n\n"
                "Use Rebind… to point the registration at a relocated directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                # Load catalog if available then open rebind flow.
                if entry.catalog_path and Path(entry.catalog_path).is_file():
                    with contextlib.suppress(WorkspaceError):
                        self._controller.activate_registered_session(
                            entry.catalog_path,
                            session_id=entry.session_id,
                            require_preprocess=False,
                        )
                self._on_rebind()
            return
        try:
            self._controller.activate_recent(entry)
        except WorkspaceNavigationBlocked as exc:
            QMessageBox.information(self, "Workspace switch blocked", str(exc))
        except WorkspaceValidationError as exc:
            QMessageBox.warning(self, "Cannot open recent session", str(exc))
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Open failed", str(exc))

    def _on_clear(self) -> None:
        try:
            self._controller.clear_workspace()
        except WorkspaceNavigationBlocked as exc:
            QMessageBox.information(self, "Workspace switch blocked", str(exc))

    def _on_registration_preview(self, preview: object) -> None:
        if not isinstance(preview, RegistrationPreview):
            return
        dialog = RegistrationPreviewDialog(preview, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._controller.discard_pending_registration()
            return
        try:
            self._controller.commit_pending_registration(
                approved_proposal_ids=dialog.selected_proposal_ids(),
                operator_approved=True,
                activate=True,
            )
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Registration commit failed", str(exc))

    def _on_rebind_preview(self, preview: object) -> None:
        if not isinstance(preview, RebindPreview):
            return
        dialog = RebindPreviewDialog(preview, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._controller.discard_pending_rebind()
            return
        try:
            self._controller.commit_pending_rebind(
                operator_approved=True,
                allow_without_fingerprint_match=dialog.allow_without_fingerprint_match,
            )
        except WorkspaceError as exc:
            QMessageBox.warning(self, "Rebind commit failed", str(exc))
