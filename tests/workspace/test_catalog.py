import json
from pathlib import Path

import pytest

from workspace.catalog import (
    detect_duplicate_session_root,
    load_catalog,
    load_catalog_safe,
    serialize_catalog,
    write_catalog,
)
from workspace.errors import WorkspaceCatalogVersionError
from workspace.models import (
    SCHEMA_VERSION,
    build_legacy_self_project_catalog,
    catalog_path_for_root,
    with_updated_path_status,
)


def _session_root(tmp_path: Path, name: str = "MouseStudy") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "preprocess").mkdir()
    (root / "results_package").mkdir()
    (root / "results_package" / "results_package__keep.json").write_text(
        '{"keep": true}\n',
        encoding="utf-8",
    )
    return root


def test_catalog_ownership_location(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    assert catalog_path_for_root(root) == root / ".behavior_suite" / "workspace_catalog_v1.json"


def test_atomic_persistence_and_reopen(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    path = write_catalog(catalog)
    assert path == catalog_path_for_root(root)
    assert path.is_file()
    loaded = load_catalog(path)
    assert loaded.catalog_id == catalog.catalog_id
    assert loaded.sessions[0].session_id == catalog.sessions[0].session_id
    assert loaded.schema_version == SCHEMA_VERSION
    # Scientific sibling untouched.
    science = root / "results_package" / "results_package__keep.json"
    assert json.loads(science.read_text(encoding="utf-8")) == {"keep": True}


def test_idempotent_serialization(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    text_a = serialize_catalog(catalog)
    text_b = serialize_catalog(catalog)
    assert text_a == text_b
    assert '"schema_version": "workspace_catalog_v1"' in text_a


def test_duplicate_detection(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    assert detect_duplicate_session_root(catalog, root) == catalog.sessions[0].session_id
    assert detect_duplicate_session_root(
        catalog,
        root,
        ignore_session_id=catalog.sessions[0].session_id,
    ) is None


def test_missing_and_relocated_paths(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    write_catalog(catalog)

    # Relocate the session directory. Legacy self-project catalogs live inside the
    # session tree, so probe refresh uses the in-memory catalog still pointing at
    # the pre-move path (D0-B will handle rebind proposals).
    moved = tmp_path / "MovedStudy"
    root.rename(moved)
    refreshed = with_updated_path_status(catalog)
    assert refreshed.sessions[0].path_status.value == "missing"
    assert refreshed.sessions[0].has_preprocess_dir is False
    assert moved.is_dir()


def test_safe_recovery_from_malformed_catalog(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    path = catalog_path_for_root(root)
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    result = load_catalog_safe(path)
    assert result.catalog is None
    assert result.recovered is True
    assert result.quarantine_path is not None
    assert result.quarantine_path.is_file()
    assert not path.exists()


def test_unsupported_catalog_version(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    path = write_catalog(catalog)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "workspace_catalog_v99"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WorkspaceCatalogVersionError, match="Unsupported"):
        load_catalog(path)

    safe = load_catalog_safe(path)
    assert safe.catalog is None
    assert safe.recovered is False
    assert path.is_file()  # future versions are not quarantined


def test_read_only_loading_does_not_rewrite(tmp_path: Path) -> None:
    root = _session_root(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    path = write_catalog(catalog)
    before = path.read_bytes()
    loaded = load_catalog(path)
    assert loaded.project.display_name == catalog.project.display_name
    assert path.read_bytes() == before
