from pathlib import Path

from project.models import Project
from project.paths import get_preprocess_dir


def test_get_preprocess_dir_resolves_official_artifact_directory(tmp_path: Path) -> None:
    project = Project(root_dir=tmp_path / "Study", name="Study")

    preprocess_dir = get_preprocess_dir(project)

    assert preprocess_dir == tmp_path / "Study" / "preprocess"
    assert not preprocess_dir.exists()
