"""Exceptions for workspace catalog and context operations."""


class WorkspaceError(Exception):
    """Base exception for workspace infrastructure failures."""


class WorkspaceValidationError(WorkspaceError, ValueError):
    """Raised when workspace hierarchy, identity, or selection is invalid."""


class WorkspaceDuplicateError(WorkspaceValidationError):
    """Raised when catalog identities or normalized paths collide."""


class WorkspaceCatalogError(WorkspaceError):
    """Raised when a catalog cannot be loaded or persisted."""


class WorkspaceCatalogVersionError(WorkspaceCatalogError):
    """Raised when a catalog schema version is unsupported."""


class WorkspaceContextError(WorkspaceError, ValueError):
    """Raised when WorkspaceContext selection rules are violated."""


class WorkspaceNavigationBlocked(WorkspaceContextError):
    """Raised when a context change would retarget or interrupt an active job unsafely."""
