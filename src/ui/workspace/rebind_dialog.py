"""Operator confirmation dialog for catalog-only path rebind."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from workspace.migrate import RebindPreview


class RebindPreviewDialog(QDialog):
    """Preview affected associations and require explicit rebind confirmation."""

    def __init__(
        self,
        preview: RebindPreview,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm workspace path rebind")
        self._preview = preview
        self.resize(640, 420)

        layout = QVBoxLayout(self)
        summary = QLabel(
            f"Session id: {preview.session_id}\n"
            f"Old path: {preview.old_session_root}\n"
            f"New path: {preview.new_session_root}\n"
            f"Fingerprint match: {'yes' if preview.fingerprint_match else 'NO'}\n"
            f"Rebind project root: {'yes' if preview.rebind_project_root else 'no'}\n"
            f"New project root: {preview.new_project_root or '(unchanged)'}"
        )
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        affected = QLabel(
            "Affected paths:\n"
            + "\n".join(f"  • {path}" for path in preview.affected_session_paths)
        )
        affected.setWordWrap(True)
        layout.addWidget(affected)

        if preview.warnings:
            warn = QLabel("Warnings:\n" + "\n".join(f"  • {w}" for w in preview.warnings))
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #8a5a00;")
            layout.addWidget(warn)

        if preview.conflicts:
            conflict = QLabel(
                "Conflicts:\n" + "\n".join(f"  • {c}" for c in preview.conflicts)
            )
            conflict.setWordWrap(True)
            conflict.setStyleSheet("color: #a40000;")
            layout.addWidget(conflict)

        self._allow_mismatch = QCheckBox(
            "Allow commit without fingerprint match (explicit override)"
        )
        # Digest/path conflicts cannot be overridden; only warning-only mismatch can.
        self._allow_mismatch.setEnabled(
            (not preview.fingerprint_match)
            and not preview.conflicts
            and bool(preview.warnings)
        )
        layout.addWidget(self._allow_mismatch)

        note = QLabel(
            "Rebind updates catalog paths only and retains UUID identities. "
            "Historical science manifests are not rewritten."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Confirm rebind")
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def allow_without_fingerprint_match(self) -> bool:
        return self._allow_mismatch.isChecked()

    def _accept_checked(self) -> None:
        if self._preview.conflicts:
            QMessageBox.warning(
                self,
                "Conflicts present",
                "Cannot commit while conflicts remain (digest mismatch or path "
                "collision). Choose a different replacement directory.",
            )
            return
        if (
            not self._preview.fingerprint_match
            and not self.allow_without_fingerprint_match
            and self._preview.warnings
        ):
            answer = QMessageBox.question(
                self,
                "Fingerprint not matched",
                "Fingerprint did not match. Commit anyway with explicit override?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._allow_mismatch.setChecked(True)
        self.accept()
