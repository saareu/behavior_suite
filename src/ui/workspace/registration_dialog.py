"""Operator confirmation dialog for registration / rescan previews."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from workspace.migrate import RegistrationPreview


class RegistrationPreviewDialog(QDialog):
    """Show a non-writing registration preview and collect explicit confirmation."""

    def __init__(
        self,
        preview: RegistrationPreview,
        *,
        parent: QWidget | None = None,
        title: str = "Confirm workspace registration",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._preview = preview
        self.resize(720, 480)

        layout = QVBoxLayout(self)
        summary = QLabel(
            f"Project: {preview.project_display_name}\n"
            f"Root: {preview.project_root}\n"
            f"Catalog: {preview.catalog_path}\n"
            f"Discovery: {preview.discovery.kind}\n"
            f"Proposals: {len(preview.proposals)}"
        )
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        self._list = QListWidget()
        for proposal in preview.proposals:
            runs = ", ".join(
                f"{name}:{count}" for name, count in sorted(proposal.run_counts.items())
            ) or "(no runs)"
            text = (
                f"[{proposal.action}] {proposal.session_display_name}\n"
                f"  path: {proposal.session_root}\n"
                f"  preprocess: {'yes' if proposal.has_preprocess_dir else 'NO'}\n"
                f"  status: {proposal.path_status.value}\n"
                f"  runs: {runs}"
            )
            if proposal.issues:
                text += "\n  issues: " + "; ".join(proposal.issues)
            if proposal.duplicates:
                text += "\n  duplicates: " + "; ".join(
                    f"{d.kind.value}:{d.message}" for d in proposal.duplicates
                )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, proposal.proposal_id)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if proposal.selected_by_default
                else Qt.CheckState.Unchecked
            )
            if proposal.action == "skip":
                item.setCheckState(Qt.CheckState.Unchecked)
            self._list.addItem(item)
        layout.addWidget(self._list, 1)

        warnings: list[str] = []
        if preview.ambiguous_associations:
            warnings.append(
                "Ambiguous folder layout detected — new creates are blocked until "
                "associations are resolved."
            )
            for assoc in preview.ambiguous_associations:
                warnings.append(f"  ambiguous[{assoc.code}]: {assoc.message}")
        if preview.duplicate_candidates:
            warnings.append(
                f"{len(preview.duplicate_candidates)} duplicate candidate(s) flagged."
            )
        if preview.missing_or_relocated:
            warnings.append("Missing/relocated registered paths:")
            warnings.extend(f"  {note}" for note in preview.missing_or_relocated)
        if warnings:
            warn = QLabel("\n".join(warnings))
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #8a5a00;")
            layout.addWidget(warn)

        note = QLabel(
            "Preview does not modify the catalog or scientific outputs. "
            "Commit writes only ``.behavior_suite/workspace_catalog_v1.json``."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Confirm registration")
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_proposal_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                proposal_id = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(proposal_id, str):
                    ids.append(proposal_id)
        return tuple(ids)

    def _accept_checked(self) -> None:
        if self._preview.ambiguous_associations:
            creates = [
                p
                for p in self._preview.proposals
                if p.action == "create" and p.proposal_id in self.selected_proposal_ids()
            ]
            if creates:
                QMessageBox.warning(
                    self,
                    "Ambiguous associations",
                    "Cannot create new registrations while the folder layout is "
                    "ambiguous. Resolve associations or select only reuse/update rows.",
                )
                return
        if not self.selected_proposal_ids():
            QMessageBox.information(
                self,
                "Nothing selected",
                "Select at least one proposal to commit, or cancel.",
            )
            return
        self.accept()
