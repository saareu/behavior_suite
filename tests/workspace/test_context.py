from pathlib import Path

import pytest

from workspace.catalog import write_catalog
from workspace.context import WorkspaceContext
from workspace.errors import WorkspaceNavigationBlocked, WorkspaceValidationError
from workspace.models import (
    RecentTargetEntry,
    SubsessionRecord,
    build_legacy_self_project_catalog,
    catalog_path_for_root,
    new_workspace_id,
)
from workspace.policy import ContextChangeRequest


def _prep(tmp_path: Path, name: str = "MouseStudy") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "preprocess").mkdir()
    return root


def test_context_switching_and_notifications(tmp_path: Path) -> None:
    root_a = _prep(tmp_path, "A")
    root_b = _prep(tmp_path, "B")
    ctx = WorkspaceContext()
    events: list[str | None] = []
    ctx.add_listener(lambda snap: events.append(snap.analysis_target_id))

    target_a = ctx.open_session_path(root_a)
    assert ctx.discovery_root == root_a.resolve()
    assert target_a.kind == "session"
    assert events[-1] == target_a.analysis_target_id

    target_b = ctx.open_session_path(root_b)
    assert ctx.discovery_root == root_b.resolve()
    assert target_b.analysis_target_id != target_a.analysis_target_id
    assert ctx.generation >= 2


def test_inconsistent_project_session_rejected(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    ctx = WorkspaceContext()
    ctx.bind_catalog(catalog, select_session_id=catalog.sessions[0].session_id)
    with pytest.raises(WorkspaceValidationError, match="different project"):
        ctx.set_session(catalog.sessions[0].session_id, project_id=new_workspace_id())


def test_optional_subsession_switch(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    session = catalog.sessions[0]
    sub = SubsessionRecord(
        subsession_id=new_workspace_id(),
        display_name="Seg",
        kind="temporal_segment",
        preprocess_binding="shared_parent",
    )
    catalog = catalog.model_copy(
        update={"sessions": (session.model_copy(update={"subsessions": (sub,)}),)}
    )
    ctx = WorkspaceContext()
    ctx.bind_catalog(catalog, select_session_id=session.session_id)
    assert ctx.subsession_id is None
    ctx.set_subsession(sub.subsession_id)
    assert ctx.subsession_id == sub.subsession_id
    assert ctx.analysis_target is not None
    assert ctx.analysis_target.science_writes_allowed is False
    ctx.clear_subsession()
    assert ctx.subsession_id is None
    assert ctx.analysis_target.kind == "session"


def test_safe_selection_restoration(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    catalog = build_legacy_self_project_catalog(root)
    path = write_catalog(catalog)
    ctx = WorkspaceContext()
    ctx.load_from_path(path, select_first_session=True, require_analysis_open=True)
    entry = ctx.remember_current()
    assert entry is not None

    ctx.clear()
    assert ctx.analysis_target is None

    # Validation-gated restore succeeds when paths remain valid.
    restored = ctx.restore_selection(entry)
    assert restored.session_id == catalog.sessions[0].session_id
    assert ctx.discovery_root == root.resolve()

    # Missing path fails the gate.
    gone = RecentTargetEntry(
        project_id=entry.project_id,
        session_id=entry.session_id,
        subsession_id=None,
        analysis_target_id=entry.analysis_target_id,
        project_root=entry.project_root,
        session_root=tmp_path / "missing_session",
        discovery_root=tmp_path / "missing_session",
        display_name="gone",
        catalog_path=entry.catalog_path,
    )
    assert ctx.propose_restore(gone) is None
    with pytest.raises(WorkspaceValidationError, match="rejected"):
        ctx.restore_selection(gone)


def test_hard_lock_blocks_context_change(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    ctx = WorkspaceContext()
    ctx.open_session_path(root)
    with pytest.raises(WorkspaceNavigationBlocked, match="hard-locked"):
        ctx.open_session_path(
            _prep(tmp_path, "Other"),
            request=ContextChangeRequest(kind="session", hard_locked=True),
        )


def test_job_binding_not_silently_retargeted(tmp_path: Path) -> None:
    root_a = _prep(tmp_path, "A")
    root_b = _prep(tmp_path, "B")
    ctx = WorkspaceContext()
    ctx.open_session_path(root_a)
    binding = ctx.bind_job(job_kind="discovery")
    assert ctx.is_stale_job_result(binding) is False
    ctx.open_session_path(root_b)
    assert ctx.is_stale_job_result(binding) is True
    assert binding.discovery_root == root_a.resolve()


def test_incomplete_open_blocked_but_listable(tmp_path: Path) -> None:
    root = tmp_path / "Incomplete"
    root.mkdir()
    ctx = WorkspaceContext()
    with pytest.raises(WorkspaceValidationError, match="preprocess"):
        ctx.open_session_path(root, require_preprocess=True)
    target = ctx.open_session_path(root, require_preprocess=False)
    assert target.has_preprocess_dir is False


def test_persist_ephemeral_writes_only_catalog(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    science = root / "behavioral_review" / "keep.txt"
    science.parent.mkdir()
    science.write_text("immutable\n", encoding="utf-8")
    ctx = WorkspaceContext()
    ctx.open_session_path(root, persist=True)
    assert catalog_path_for_root(root).is_file()
    assert science.read_text(encoding="utf-8") == "immutable\n"
    assert ctx.ephemeral is False
