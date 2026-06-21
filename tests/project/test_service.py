from pathlib import Path

import pytest

from project.service import ProjectService
from project.validation import ProjectValidationError


def test_create_project_creates_only_root_and_preprocess(tmp_path: Path) -> None:
    project = ProjectService().create_project(tmp_path, "MouseStudy")

    assert project.name == "MouseStudy"
    assert project.root_dir == (tmp_path / "MouseStudy").resolve()
    assert project.root_dir.is_dir()
    assert sorted(path.name for path in project.root_dir.iterdir()) == ["preprocess"]
    assert (project.root_dir / "preprocess").is_dir()


def test_create_project_rejects_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(ProjectValidationError, match="reserved path character"):
        ProjectService().create_project(tmp_path, "mouse/study")


def test_create_project_rejects_existing_target(tmp_path: Path) -> None:
    (tmp_path / "Existing").mkdir()

    with pytest.raises(ProjectValidationError, match="already exists"):
        ProjectService().create_project(tmp_path, "Existing")


def test_open_created_project(tmp_path: Path) -> None:
    service = ProjectService()
    created = service.create_project(tmp_path, "MouseStudy")

    opened = service.open_project(created.root_dir)

    assert opened == created


@pytest.mark.parametrize("invalid_kind", ["missing", "file", "no_preprocess"])
def test_open_project_rejects_invalid_project_path(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    if invalid_kind == "missing":
        project_path = tmp_path / "missing"
    elif invalid_kind == "file":
        project_path = tmp_path / "not-a-directory"
        project_path.write_text("not a project", encoding="utf-8")
    else:
        project_path = tmp_path / "not-a-project"
        project_path.mkdir()

    with pytest.raises(ProjectValidationError):
        ProjectService().open_project(project_path)
