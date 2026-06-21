"""Path resolution for shared project subsystem directories."""

from pathlib import Path

from project.models import Project

PREPROCESS_DIRNAME = "preprocess"


def get_preprocess_dir(project: Project) -> Path:
    """Return the project's official preprocessing artifact directory path.

    This resolver has no filesystem side effects; project creation and opening
    are handled by ProjectService.
    """

    return project.root_dir / PREPROCESS_DIRNAME
