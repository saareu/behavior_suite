"""Helpers for D0-D2 shared workspace ↔ subsystem page binding.

Pages must not own a separate :class:`WorkspaceContext`. Browse/Open write through
``MainWindow`` / :class:`WorkspaceShellController`; pages then consume the shared
``discovery_root`` via ``set_session`` with same-root short-circuits so run
selection and open S10 drafts are not silently destroyed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class SupportsWorkspaceSessionBind(Protocol):
    """Minimal page surface used by shared workspace binding."""

    def set_session(self, session_root: Path | str) -> None: ...

    def refresh_from_state(self) -> None: ...


WorkspaceSessionOpener = Callable[[Path | str], None]


def normalize_session_root(path: Path | str | None) -> Path | None:
    """Return a resolved session root, or None when empty/invalid input."""

    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False)


def bind_workspace_session_opener(
    page: Any,
    opener: WorkspaceSessionOpener | None,
) -> None:
    """Attach a shared Browse/Open write-back callback to a subsystem page."""

    page._workspace_session_opener = opener


def request_shared_session(page: Any, path: Path | str) -> bool:
    """Route a page Browse/Open through the shared workspace opener when bound.

    Returns True when the shared opener handled the request (caller must not call
    ``set_session`` itself — MainWindow propagation applies the discovery root).
    Returns False when unbound (unit tests / standalone pages keep local open).
    """

    opener = getattr(page, "_workspace_session_opener", None)
    if opener is None:
        return False
    opener(path)
    return True


def controller_session_root(controller: Any) -> Path | None:
    """Best-effort read of a page controller's bound session root."""

    state = getattr(controller, "state", None)
    if state is not None:
        root = getattr(state, "session_root", None)
        if isinstance(root, Path | str) and str(root).strip():
            return normalize_session_root(root)
    project = getattr(controller, "project", None)
    if project is not None:
        root = getattr(project, "session_root", None)
        if isinstance(root, Path | str) and str(root).strip():
            return normalize_session_root(root)
    session = getattr(controller, "session", None)
    if session is not None:
        root = getattr(session, "session_root", None)
        if isinstance(root, Path | str) and str(root).strip():
            return normalize_session_root(root)
    return None


def same_session_root(
    left: Path | str | None,
    right: Path | str | None,
) -> bool:
    a = normalize_session_root(left)
    b = normalize_session_root(right)
    if a is None or b is None:
        return False
    return a == b
