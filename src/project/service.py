"""Shared project creation and opening service."""

from pathlib import Path

from project.models import Project
from project.paths import PREPROCESS_DIRNAME
from project.validation import (
    ProjectCreationError,
    ProjectValidationError,
    validate_project_name,
    validate_project_root,
)


class ProjectService:
    """Create and open behavior-suite projects."""

    def create_project(self, parent_dir: Path, project_name: str) -> Project:
        """Create a project root and its initial preprocess directory.

        Raises:
            ProjectValidationError: If the parent or project name is invalid,
                or the target already exists.
            ProjectCreationError: If directory creation fails.
        """

        name = validate_project_name(project_name)
        parent = Path(parent_dir).expanduser().resolve()
        if not parent.exists():
            raise ProjectValidationError(f"Parent directory does not exist: {parent}")
        if not parent.is_dir():
            raise ProjectValidationError(f"Parent path is not a directory: {parent}")

        project_dir = parent / name
        if project_dir.exists():
            raise ProjectValidationError(f"Project path already exists: {project_dir}")

        try:
            project_dir.mkdir()
            (project_dir / PREPROCESS_DIRNAME).mkdir()
        except OSError as exc:
            raise ProjectCreationError(f"Could not create project at {project_dir}: {exc}") from exc

        return Project(root_dir=project_dir, name=name)

    def open_project(self, project_dir: Path) -> Project:
        """Open an existing project with a valid preprocess directory.

        Raises:
            ProjectValidationError: If the path is not a valid project root.
        """

        root = validate_project_root(project_dir, require_preprocess_dir=True)
        name = validate_project_name(root.name)
        return Project(root_dir=root, name=name)
