"""Validation helpers for shared project lifecycle operations."""

from pathlib import Path


class ProjectError(Exception):
    """Base exception for project lifecycle failures."""


class ProjectValidationError(ProjectError, ValueError):
    """Raised when a project name or directory is invalid."""


class ProjectCreationError(ProjectError):
    """Raised when project directories cannot be created."""


_INVALID_PROJECT_NAME_CHARACTERS = frozenset(r'<>:"/\|?*')


def validate_project_name(project_name: str) -> str:
    """Validate and return a project directory name.

    Raises:
        ProjectValidationError: If the name is empty, unsafe, or not a single
            portable path component.
    """

    if not isinstance(project_name, str):
        raise ProjectValidationError("Project name must be a string.")
    if not project_name or project_name != project_name.strip():
        raise ProjectValidationError("Project name must be non-empty without outer whitespace.")
    if project_name in {".", ".."}:
        raise ProjectValidationError("Project name cannot be '.' or '..'.")
    if project_name.endswith((" ", ".")):
        raise ProjectValidationError("Project name cannot end with a space or period.")
    if any(character in _INVALID_PROJECT_NAME_CHARACTERS for character in project_name):
        raise ProjectValidationError("Project name contains a reserved path character.")
    if any(ord(character) < 32 for character in project_name):
        raise ProjectValidationError("Project name contains a control character.")
    return project_name


def validate_project_root(project_dir: Path, *, require_preprocess_dir: bool = True) -> Path:
    """Validate and return an absolute project root path.

    Raises:
        ProjectValidationError: If the path is not a directory or, when
            requested, does not contain a preprocess directory.
    """

    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise ProjectValidationError(f"Project directory does not exist: {project_dir}")
    if not project_dir.is_dir():
        raise ProjectValidationError(f"Project path is not a directory: {project_dir}")

    preprocess_dir = project_dir / "preprocess"
    if require_preprocess_dir and not preprocess_dir.is_dir():
        raise ProjectValidationError(
            f"Project directory is missing its preprocess directory: {preprocess_dir}"
        )
    return project_dir
