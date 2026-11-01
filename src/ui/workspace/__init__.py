"""D0 workspace GUI shell: shared context, selector, register/rebind, page bind."""

from ui.workspace.page_binding import (
    bind_workspace_session_opener,
    request_shared_session,
)
from ui.workspace.recent_store import WorkspaceRecentStore
from ui.workspace.shell_controller import WorkspaceShellController

__all__ = [
    "WorkspaceRecentStore",
    "WorkspaceShellController",
    "bind_workspace_session_opener",
    "request_shared_session",
]
