from pathlib import Path

from workspace.models import (
    SubsessionRecord,
    TemporalBounds,
    build_legacy_self_project_catalog,
    new_workspace_id,
)
from workspace.resolve import (
    analysis_open_blocked_reason,
    resolve_analysis_target,
    science_writes_allowed,
)


def test_session_default_resolve(tmp_path: Path) -> None:
    root = tmp_path / "S"
    root.mkdir()
    (root / "preprocess").mkdir()
    catalog = build_legacy_self_project_catalog(root)
    target = resolve_analysis_target(catalog, session_id=catalog.sessions[0].session_id)
    assert target.kind == "session"
    assert target.discovery_root == root.resolve()
    assert target.preprocess_root == (root / "preprocess").resolve()
    assert target.science_writes_allowed is True
    assert analysis_open_blocked_reason(target) is None


def test_optional_shared_parent_subsession_read_only(tmp_path: Path) -> None:
    root = tmp_path / "S"
    root.mkdir()
    (root / "preprocess").mkdir()
    catalog = build_legacy_self_project_catalog(root)
    session = catalog.sessions[0]
    sub = SubsessionRecord(
        subsession_id=new_workspace_id(),
        display_name="Window",
        kind="temporal_segment",
        preprocess_binding="shared_parent",
        temporal_bounds=TemporalBounds(domain="prepared", start_idx=10, end_idx_exclusive=50),
    )
    catalog = catalog.model_copy(
        update={"sessions": (session.model_copy(update={"subsessions": (sub,)}),)}
    )
    target = resolve_analysis_target(
        catalog,
        session_id=session.session_id,
        subsession_id=sub.subsession_id,
    )
    assert target.kind == "subsession"
    assert target.discovery_root == root.resolve()
    assert target.preprocess_binding == "shared_parent"
    assert target.science_writes_allowed is False
    assert science_writes_allowed("shared_parent") is False
    assert target.temporal_filter is not None


def test_independent_subsession_discovery_root(tmp_path: Path) -> None:
    root = tmp_path / "S"
    root.mkdir()
    (root / "preprocess").mkdir()
    analysis = root / "subsessions" / "cam2"
    analysis.mkdir(parents=True)
    (analysis / "preprocess").mkdir()
    catalog = build_legacy_self_project_catalog(root)
    session = catalog.sessions[0]
    sub = SubsessionRecord(
        subsession_id=new_workspace_id(),
        display_name="Cam2",
        kind="separate_recording",
        preprocess_binding="independent",
        slug="cam2",
        analysis_root=analysis,
    )
    catalog = catalog.model_copy(
        update={"sessions": (session.model_copy(update={"subsessions": (sub,)}),)}
    )
    target = resolve_analysis_target(
        catalog,
        session_id=session.session_id,
        subsession_id=sub.subsession_id,
    )
    assert target.discovery_root == analysis.resolve()
    assert target.science_writes_allowed is True


def test_incomplete_session_blocks_analysis_open(tmp_path: Path) -> None:
    root = tmp_path / "Incomplete"
    root.mkdir()
    catalog = build_legacy_self_project_catalog(root, allow_incomplete=True)
    target = resolve_analysis_target(catalog, session_id=catalog.sessions[0].session_id)
    reason = analysis_open_blocked_reason(target)
    assert reason is not None
    assert "preprocess" in reason
