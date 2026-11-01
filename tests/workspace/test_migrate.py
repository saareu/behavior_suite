"""D0-B discovery, registration, rebind, and safe-reopen tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from workspace.catalog import write_catalog
from workspace.errors import WorkspaceDuplicateError, WorkspaceValidationError
from workspace.migrate import (
    commit_rebind,
    commit_registration,
    preview_rebind,
    preview_registration,
    reopen_registered_session,
    rescan_catalog,
    validate_catalog,
)
from workspace.models import (
    PathStatus,
    build_legacy_self_project_catalog,
    catalog_path_for_root,
    new_workspace_id,
)
from workspace.scan import scan_candidate

E_SESSION = Path(__file__).resolve().parents[2] / "test_cases" / "E_full_session_integration"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _make_session(tmp_path: Path, name: str, *, with_preprocess: bool = True) -> Path:
    root = tmp_path / name
    root.mkdir()
    if with_preprocess:
        prep = root / "preprocess"
        prep.mkdir()
        (prep / "prepare_meta.json").write_text('{"schema_version": "test"}\n', encoding="utf-8")
        (prep / "prepared_video.mp4").write_bytes(b"fake")
        (prep / "prepared_sync.npz").write_bytes(b"fake")
    return root


def test_legacy_self_project_registration(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "LegacyMouse")
    (root / "pose_inference").mkdir()
    preview = preview_registration(root, discover_runs=True)
    assert preview.legacy_self_project is True
    assert len(preview.proposals) == 1
    assert preview.proposals[0].action == "create"
    assert preview.proposals[0].has_preprocess_dir is True

    with pytest.raises(WorkspaceValidationError, match="operator approval"):
        commit_registration(preview, operator_approved=False)

    result = commit_registration(preview, operator_approved=True)
    assert result.catalog_path == catalog_path_for_root(root)
    assert result.catalog_path.is_file()
    assert len(result.catalog.sessions) == 1
    assert result.catalog.sessions[0].legacy_self_project is True
    assert result.catalog.project.root_path == root.resolve()


def test_multiple_sessions_under_one_logical_project(tmp_path: Path) -> None:
    project = tmp_path / "ColonyA"
    project.mkdir()
    s1 = _make_session(project, "Day1")
    s2 = _make_session(project, "Day2")
    preview = preview_registration(project)
    assert preview.discovery.kind == "parent_project"
    assert len(preview.proposals) == 2
    roots = {p.session_root for p in preview.proposals}
    assert s1.resolve() in roots
    assert s2.resolve() in roots
    assert all(p.project_id == preview.project_id for p in preview.proposals)

    result = commit_registration(preview, operator_approved=True)
    assert len(result.catalog.sessions) == 2
    assert result.catalog.project.root_path == project.resolve()
    assert not any(s.legacy_self_project for s in result.catalog.sessions)


def test_incomplete_session_registration(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "Partial", with_preprocess=False)
    (root / "results_package").mkdir()
    (root / "results_package" / "results_package__demo").mkdir()
    preview = preview_registration(root, allow_incomplete=True)
    assert len(preview.proposals) == 1
    assert preview.proposals[0].has_preprocess_dir is False
    result = commit_registration(preview, operator_approved=True)
    session = result.catalog.sessions[0]
    assert session.has_preprocess_dir is False

    reopened = reopen_registered_session(result.catalog, session.session_id)
    assert reopened.analysis_open_blocked is not None
    assert "preprocess" in reopened.analysis_open_blocked.lower()


def test_idempotent_rescan_preserves_uuids(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "StableIds")
    first = commit_registration(preview_registration(root), operator_approved=True)
    session_id = first.catalog.sessions[0].session_id
    project_id = first.catalog.project.project_id
    catalog_id = first.catalog.catalog_id

    again = rescan_catalog(first.catalog)
    assert len(again.proposals) == 1
    assert again.proposals[0].action == "reuse"
    assert again.proposals[0].session_id == session_id
    second = commit_registration(again, operator_approved=True)
    assert second.catalog.sessions[0].session_id == session_id
    assert second.catalog.project.project_id == project_id
    assert second.catalog.catalog_id == catalog_id


def test_duplicate_paths_rejected_on_create(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "DupPath")
    catalog = build_legacy_self_project_catalog(root)
    write_catalog(catalog)
    preview = preview_registration(root, existing_catalog=catalog)
    assert preview.proposals[0].action == "reuse"

    # Force a conflicting create proposal against the same path.
    rogue = preview.proposals[0]
    from workspace.migrate import RegistrationProposal

    conflict = RegistrationProposal(
        proposal_id=new_workspace_id(),
        action="create",
        project_display_name=rogue.project_display_name,
        project_root=rogue.project_root,
        project_id=rogue.project_id,
        session_display_name="Other",
        session_root=rogue.session_root,
        session_id=new_workspace_id(),
        legacy_self_project=True,
        has_preprocess_dir=True,
        path_status=PathStatus.OK,
        fingerprint=rogue.fingerprint,
        run_counts={},
        present_subsystems=(),
        missing_or_incomplete_stages=(),
        issues=(),
        duplicates=(),
    )
    bad_preview = preview.__class__(
        catalog_path=preview.catalog_path,
        project_root=preview.project_root,
        project_id=preview.project_id,
        project_display_name=preview.project_display_name,
        legacy_self_project=True,
        discovery=preview.discovery,
        existing_catalog=catalog,
        proposals=(conflict,),
        ambiguous_associations=(),
        duplicate_candidates=(),
        missing_or_relocated=(),
        proposed_catalog=preview.proposed_catalog,
    )
    with pytest.raises(WorkspaceDuplicateError):
        commit_registration(
            bad_preview,
            approved_proposal_ids=[conflict.proposal_id],
            operator_approved=True,
        )


def test_ambiguous_hierarchy_blocks_new_creates(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "ParentAndSelf")
    child = _make_session(root, "ChildDay")
    discovery = scan_candidate(root)
    assert discovery.kind == "ambiguous"
    preview = preview_registration(root)
    assert preview.ambiguous_associations
    assert any(p.session_root == root.resolve() for p in preview.proposals)
    assert any(p.session_root == child.resolve() for p in preview.proposals)
    with pytest.raises(WorkspaceValidationError, match="Ambiguous hierarchy"):
        commit_registration(preview, operator_approved=True)


def test_catalog_relocation_and_explicit_rebind(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "BeforeMove")
    # Plant a digest so fingerprint match can succeed after copy.
    meta = json.loads((root / "preprocess" / "prepare_meta.json").read_text(encoding="utf-8"))
    meta["prepared_video_sha256"] = "abc123fingerprint"
    (root / "preprocess" / "prepare_meta.json").write_text(
        json.dumps(meta) + "\n", encoding="utf-8"
    )
    result = commit_registration(preview_registration(root), operator_approved=True)
    session_id = result.catalog.sessions[0].session_id

    moved = tmp_path / "AfterMove"
    shutil.copytree(root, moved)
    # Simulate missing old path while catalog still points at it.
    shutil.rmtree(root)

    refreshed = validate_catalog(result.catalog)
    from workspace.models import with_updated_path_status

    missing = with_updated_path_status(refreshed)
    assert missing.sessions[0].path_status is PathStatus.MISSING

    rebind = preview_rebind(missing, session_id=session_id, new_session_root=moved)
    assert rebind.fingerprint_match is True
    assert not rebind.conflicts
    assert rebind.proposed_catalog.sessions[0].session_id == session_id

    with pytest.raises(WorkspaceValidationError, match="operator approval"):
        commit_rebind(missing, rebind, operator_approved=False)

    rebound, path = commit_rebind(
        missing,
        rebind,
        operator_approved=True,
        catalog_path=tmp_path / "rebind_catalog.json",
    )
    assert path.is_file()
    assert rebound.sessions[0].session_root == moved.resolve()
    assert rebound.sessions[0].session_id == session_id
    assert rebound.sessions[0].path_status is PathStatus.OK


def test_rebind_rejects_unrelated_folder_without_override(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "Original")
    meta = json.loads((root / "preprocess" / "prepare_meta.json").read_text(encoding="utf-8"))
    meta["prepared_video_sha256"] = "digest-original"
    (root / "preprocess" / "prepare_meta.json").write_text(
        json.dumps(meta) + "\n", encoding="utf-8"
    )
    other = _make_session(tmp_path, "Unrelated")
    other_meta = json.loads(
        (other / "preprocess" / "prepare_meta.json").read_text(encoding="utf-8")
    )
    other_meta["prepared_video_sha256"] = "digest-other"
    (other / "preprocess" / "prepare_meta.json").write_text(
        json.dumps(other_meta) + "\n", encoding="utf-8"
    )
    catalog = commit_registration(
        preview_registration(root), operator_approved=True
    ).catalog
    session_id = catalog.sessions[0].session_id
    preview = preview_rebind(catalog, session_id=session_id, new_session_root=other)
    assert preview.conflicts
    with pytest.raises(WorkspaceValidationError, match="conflicts"):
        commit_rebind(catalog, preview, operator_approved=True)


def test_missing_paths_remain_visible(tmp_path: Path) -> None:
    root = _make_session(tmp_path, "GoneSoon")
    result = commit_registration(preview_registration(root), operator_approved=True)
    session_id = result.catalog.sessions[0].session_id
    shutil.rmtree(root)
    reopened = reopen_registered_session(result.catalog, session_id)
    assert reopened.session.path_status is PathStatus.MISSING
    assert reopened.analysis_open_blocked is not None
    # Still listed in catalog identities.
    assert session_id in {s.session_id for s in reopened.catalog.sessions}


def test_catalog_validation_after_mutations(tmp_path: Path) -> None:
    a = _make_session(tmp_path, "A")
    b = _make_session(tmp_path, "B")
    project = tmp_path / "Proj"
    project.mkdir()
    # Move sessions under project for a multi-session catalog.
    a2 = project / "A"
    b2 = project / "B"
    shutil.copytree(a, a2)
    shutil.copytree(b, b2)
    result = commit_registration(preview_registration(project), operator_approved=True)
    catalog = result.catalog
    validate_catalog(catalog)

    # Mutate to introduce a duplicate path and ensure validation catches it.
    s0 = catalog.sessions[0]
    s1 = catalog.sessions[1]
    broken = catalog.model_copy(
        update={
            "sessions": (
                s0,
                s1.model_copy(update={"session_root": s0.session_root}),
            )
        }
    )
    with pytest.raises(WorkspaceValidationError, match="Duplicate session_root"):
        validate_catalog(broken)


def test_reopening_does_not_require_catalog_for_ephemeral_path(tmp_path: Path) -> None:
    from workspace.migrate import ephemeral_open_without_catalog

    root = _make_session(tmp_path, "Ephemeral")
    target = ephemeral_open_without_catalog(root)
    assert target.discovery_root == root.resolve()


@pytest.mark.integration
def test_real_e_session_smoke_read_only() -> None:
    """Read-only smoke against the lab E-session when present."""

    if not E_SESSION.is_dir() or not (E_SESSION / "preprocess").is_dir():
        pytest.skip("E_full_session_integration session not available")

    science_files = [
        E_SESSION / "preprocess" / "prepare_meta.json",
        E_SESSION / "preprocess" / "prepared_sync.npz",
    ]
    # Prefer a small S9 manifest if present for stronger integrity evidence.
    manifests = sorted((E_SESSION / "results_package").glob("*/package_manifest.json"))
    if manifests:
        science_files.append(manifests[0])
    before = {str(path): _sha256(path) for path in science_files if path.is_file()}
    assert before

    discovery = scan_candidate(E_SESSION, discover_runs=True)
    assert discovery.kind == "legacy_self_project"
    assert len(discovery.sessions) == 1
    session = discovery.sessions[0]
    assert session.has_preprocess_dir is True
    assert session.s1_handoff_complete is True
    # Previously analyzed session should expose multiple subsystem runs.
    assert session.run_counts.get("s2", 0) >= 1
    assert session.run_counts.get("s9", 0) >= 1
    assert session.run_counts.get("s10", 0) >= 1

    preview = preview_registration(E_SESSION, discover_runs=True)
    assert preview.proposals[0].action in {"create", "reuse"}
    # Commit catalog sidecar outside the science tree (temp file only).
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "workspace_catalog_v1.json"
        result = commit_registration(
            preview,
            operator_approved=True,
            catalog_path=out,
            write_backup=False,
        )
        reopened = reopen_registered_session(
            result.catalog_path,
            result.catalog.sessions[0].session_id,
            discover_runs=True,
        )
        assert reopened.analysis_target.discovery_root == E_SESSION.resolve()
        assert reopened.analysis_open_blocked is None
        assert reopened.discovery is not None
        assert reopened.discovery.run_counts.get("s9", 0) >= 1

        # Existing discovery interfaces still work through the resolved root.
        from behavioral_review.discovery import summarize_behavioral_review_project
        from results_packaging.run_discovery import summarize_results_package_project

        s9 = summarize_results_package_project(reopened.analysis_target.discovery_root)
        s10 = summarize_behavioral_review_project(
            reopened.analysis_target.discovery_root
        )
        assert len(s9.runs) >= 1
        assert len(s10.runs) >= 1

    after = {str(path): _sha256(path) for path in science_files if path.is_file()}
    assert after == before
