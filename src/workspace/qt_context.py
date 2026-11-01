"""Optional Qt signal adapter for WorkspaceContext (D0-A; no MainWindow wiring)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from workspace.context import WorkspaceContext, WorkspaceContextSnapshot
from workspace.errors import WorkspaceValidationError

if TYPE_CHECKING:
    from PySide6.QtCore import QObject


def create_qt_workspace_bridge(context: WorkspaceContext) -> QObject:
    """Return a QObject that re-emits context snapshots as Qt signals.

    Import is deferred so headless installs without PySide6 can still use
    ``workspace.context``. Full GUI integration is D0-D.
    """

    from PySide6.QtCore import QObject, Signal

    class WorkspaceContextBridge(QObject):
        projectChanged = Signal(object, object)  # project_id, root_path
        sessionChanged = Signal(object, object)  # session_id, session_root
        subsessionChanged = Signal(object)  # subsession_id | None
        analysisTargetChanged = Signal(object)  # AnalysisTarget | None
        contextCleared = Signal()
        contextChanged = Signal(object)  # WorkspaceContextSnapshot

        def __init__(self, owner: WorkspaceContext) -> None:
            super().__init__()
            self._owner = owner
            self._owner.add_listener(self._on_snapshot)

        def _on_snapshot(self, snap: WorkspaceContextSnapshot) -> None:
            self.contextChanged.emit(snap)
            if snap.project_id is None and snap.session_id is None:
                self.contextCleared.emit()
                self.analysisTargetChanged.emit(None)
                return
            root = None
            session_root = None
            catalog = self._owner.catalog
            if catalog is not None:
                root = str(catalog.project.root_path)
                if snap.session_id is not None:
                    try:
                        session_root = str(catalog.get_session(snap.session_id).session_root)
                    except WorkspaceValidationError:
                        session_root = None
            self.projectChanged.emit(snap.project_id, root)
            self.sessionChanged.emit(snap.session_id, session_root)
            self.subsessionChanged.emit(snap.subsession_id)
            self.analysisTargetChanged.emit(snap.analysis_target)

    return WorkspaceContextBridge(context)
