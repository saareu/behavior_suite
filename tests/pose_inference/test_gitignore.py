from pathlib import Path


def test_generated_pose_inference_artifacts_are_ignored() -> None:
    ignored_patterns = Path(".gitignore").read_text(encoding="utf-8")

    assert "test_cases/**/preprocess/" in ignored_patterns
    assert "test_runs/" in ignored_patterns
    assert "acceptance_tests/" in ignored_patterns
    assert "bsuite_runs/" in ignored_patterns
    assert "*.slp" in ignored_patterns
    assert "*.mp4" in ignored_patterns
    assert "*.avi" in ignored_patterns
    assert "*.npz" in ignored_patterns
    assert "Thumbs.db" in ignored_patterns
    assert "docs/subsystem_02/evidence/" not in ignored_patterns
