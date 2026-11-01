from pathlib import Path

import pytest
from pydantic import ValidationError

from workspace.catalog import replace_sessions
from workspace.errors import (
    WorkspaceDuplicateError,
    WorkspaceValidationError,
)
from workspace.models import (
    PathStatus,
    ProjectRecord,
    SessionFingerprint,
    SessionRecord,
    SubsessionRecord,
    TemporalBounds,
    WorkspaceCatalog,
    build_legacy_self_project_catalog,
    default_subsession_analysis_root,
    new_workspace_id,
    parse_analysis_target_id,
    session_analysis_target_id,
    subsession_analysis_target_id,
    validate_subsession_slug,
)


def _catalog_with_session(tmp_path: Path, *, with_preprocess: bool = True) -> WorkspaceCatalog:
    root = tmp_path / "MouseStudy"
    root.mkdir()
    if with_preprocess:
        (root / "preprocess").mkdir()
    return build_legacy_self_project_catalog(root)


def test_uuid_identity_separate_from_slug(tmp_path: Path) -> None:
    catalog = _catalog_with_session(tmp_path)
    session = catalog.sessions[0]
    assert session.session_id != session.display_name
    assert len(session.session_id) == 36
    slug = validate_subsession_slug("day1_am")
    assert slug == "day1_am"
    assert default_subsession_analysis_root(session.session_root, slug).name == slug


def test_analysis_target_id_helpers() -> None:
    sid = new_workspace_id()
    ssid = new_workspace_id()
    assert parse_analysis_target_id(session_analysis_target_id(sid)) == ("session", sid)
    assert parse_analysis_target_id(subsession_analysis_target_id(ssid)) == (
        "subsession",
        ssid,
    )
    with pytest.raises(WorkspaceValidationError):
        parse_analysis_target_id("run:abc")


def test_session_only_backward_compatible(tmp_path: Path) -> None:
    catalog = _catalog_with_session(tmp_path)
    session = catalog.sessions[0]
    assert session.subsessions == ()
    assert session.legacy_self_project is True
    assert session.has_preprocess_dir is True
    assert catalog.project.root_path == session.session_root


def test_incomplete_session_may_be_registered(tmp_path: Path) -> None:
    root = tmp_path / "Incomplete"
    root.mkdir()
    catalog = build_legacy_self_project_catalog(root, allow_incomplete=True)
    assert catalog.sessions[0].has_preprocess_dir is False
    assert catalog.sessions[0].path_status is PathStatus.OK


def test_valid_and_invalid_hierarchy(tmp_path: Path) -> None:
    catalog = _catalog_with_session(tmp_path)
    session = catalog.sessions[0]
    sub_a = SubsessionRecord(
        subsession_id=new_workspace_id(),
        display_name="Segment A",
        kind="temporal_segment",
        preprocess_binding="shared_parent",
        slug="seg_a",
        temporal_bounds=TemporalBounds(domain="prepared", start_idx=0, end_idx_exclusive=100),
    )
    sub_b = SubsessionRecord(
        subsession_id=new_workspace_id(),
        display_name="Rec B",
        kind="separate_recording",
        preprocess_binding="independent",
        slug="rec_b",
        analysis_root=session.session_root / "subsessions" / "rec_b",
    )
    updated_session = SessionRecord.model_validate(
        {
            **session.model_dump(mode="json"),
            "subsessions": [sub_a.model_dump(mode="json"), sub_b.model_dump(mode="json")],
        }
    )
    ok = replace_sessions(catalog, (updated_session,))
    assert len(ok.sessions[0].subsessions) == 2

    with pytest.raises((WorkspaceDuplicateError, ValidationError)):
        SessionRecord.model_validate(
            {
                **session.model_dump(mode="json"),
                "subsessions": [sub_a.model_dump(mode="json"), sub_a.model_dump(mode="json")],
            }
        )


def test_duplicate_session_roots_rejected(tmp_path: Path) -> None:
    catalog = _catalog_with_session(tmp_path)
    session = catalog.sessions[0]
    twin = session.model_copy(update={"session_id": new_workspace_id()})
    with pytest.raises((WorkspaceDuplicateError, ValidationError, WorkspaceValidationError)):
        replace_sessions(catalog, (session, twin))


def test_independent_subsession_requires_analysis_root() -> None:
    with pytest.raises((WorkspaceValidationError, ValidationError)):
        SubsessionRecord(
            subsession_id=new_workspace_id(),
            display_name="Bad",
            kind="separate_recording",
            preprocess_binding="independent",
        )


def test_temporal_bounds_invalid() -> None:
    with pytest.raises((WorkspaceValidationError, ValidationError)):
        TemporalBounds(domain="prepared", start_idx=10, end_idx_exclusive=10)


def test_project_record_paths_normalized(tmp_path: Path) -> None:
    root = tmp_path / "P"
    root.mkdir()
    project = ProjectRecord(
        project_id=new_workspace_id(),
        display_name="P",
        root_path=root,
    )
    assert project.root_path.is_absolute()
    fingerprint = SessionFingerprint(preprocess_present=False)
    assert fingerprint.prepared_video_sha256 is None
