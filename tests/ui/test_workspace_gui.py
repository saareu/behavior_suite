"""D0-D1 GUI/controller tests for shared workspace shell integration."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import GROUP_DATA_PREPARATION, MainWindow
from ui.workspace.recent_store import WorkspaceRecentStore
from ui.workspace.shell_controller import WorkspaceShellController
from workspace.catalog import write_catalog
from workspace.context import WorkspaceContext
from workspace.errors import WorkspaceNavigationBlocked, WorkspaceValidationError
from workspace.migrate import preview_rebind, preview_registration
from workspace.models import (
    RecentTargetEntry,
    build_legacy_self_project_catalog,
    catalog_path_for_root,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APPLICATION = QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout_sec: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        _APPLICATION.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def _prep(tmp_path: Path, name: str = "MouseStudy") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "preprocess").mkdir()
    return root.resolve()


def _settings(tmp_path: Path) -> QSettings:
    path = tmp_path / "workspace_settings.ini"
    return QSettings(str(path), QSettings.Format.IniFormat)


def test_mainwindow_workspace_initialization() -> None:
    window = MainWindow(PreprocessSetupController())
    assert isinstance(window.workspace, WorkspaceContext)
    assert isinstance(window.workspace_controller, WorkspaceShellController)
    assert window.workspace_selector is not None
    # Shared controller — not per-page copies.
    assert window.workspace_controller.context is window.workspace
    # Launch does not auto-activate a session (W4(c)).
    assert window.workspace.session_id is None
    assert window.workspace.discovery_root is None


def test_remember_only_launch_loads_recent_without_activation(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    settings = _settings(tmp_path)
    store = WorkspaceRecentStore(settings)
    catalog = build_legacy_self_project_catalog(session)
    catalog_path = write_catalog(catalog)
    entry = RecentTargetEntry(
        project_id=catalog.project.project_id,
        session_id=catalog.sessions[0].session_id,
        subsession_id=None,
        analysis_target_id=f"session:{catalog.sessions[0].session_id}",
        project_root=session,
        session_root=session,
        discovery_root=session,
        display_name=catalog.sessions[0].display_name,
        catalog_path=catalog_path,
    )
    store.remember(entry)

    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=settings,
        nav_probe=lambda: (False, False, False),
    )
    assert len(controller.recent_targets) == 1
    assert controller.recent_targets[0].session_root == session
    assert controller.discovery_root is None
    assert controller.session_id is None
    before = catalog_path.read_bytes()
    _APPLICATION.processEvents()
    assert catalog_path.read_bytes() == before


def test_browse_and_ephemeral_open(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    settings = _settings(tmp_path)
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=settings,
        nav_probe=lambda: (False, False, False),
    )
    target = controller.open_session_path(session, require_preprocess=True)
    assert target.discovery_root == session
    assert controller.ephemeral is True
    assert controller.registration_status().registered is False
    assert controller.recent_targets
    assert controller.recent_targets[0].session_root == session
    # Ephemeral open must not create a durable catalog by default.
    assert not catalog_path_for_root(session).is_file()


def test_registration_preview_does_not_write_and_confirm_commits(
    tmp_path: Path,
) -> None:
    session = _prep(tmp_path)
    settings = _settings(tmp_path)
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=settings,
        nav_probe=lambda: (False, False, False),
    )
    catalog_file = catalog_path_for_root(session)
    assert not catalog_file.is_file()

    request = controller.begin_registration_preview(session)
    _wait_until(lambda: controller.pending_registration_preview is not None)
    preview = controller.pending_registration_preview
    assert preview is not None
    assert not catalog_file.is_file()
    assert not controller.is_stale_scan(request)

    result = controller.commit_pending_registration(
        operator_approved=True,
        activate=True,
    )
    assert catalog_file.is_file()
    assert result.catalog_path == catalog_file
    assert controller.ephemeral is False
    assert controller.discovery_root == session
    assert controller.session_id in result.committed_session_ids


def test_background_scan_stale_rejection_and_navigate_away(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=_settings(tmp_path),
        nav_probe=lambda: (False, False, False),
    )
    request = controller.begin_registration_preview(session)
    controller.cancel_background_scan(reason="navigated away")
    assert controller.is_stale_scan(request)
    assert controller.pending_registration_preview is None
    # Wait for worker to finish without accepting a catalog write.
    _wait_until(lambda: not controller.is_background_busy, timeout_sec=5.0)
    assert not catalog_path_for_root(session).is_file()


def test_rescan_idempotency(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    catalog = build_legacy_self_project_catalog(session)
    path = write_catalog(catalog)
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=_settings(tmp_path),
        nav_probe=lambda: (False, False, False),
    )
    controller.activate_registered_session(
        path,
        session_id=catalog.sessions[0].session_id,
        require_preprocess=False,
    )
    original_ids = {
        "catalog": controller.catalog.catalog_id if controller.catalog else None,
        "project": controller.project_id,
        "session": controller.session_id,
    }
    controller.begin_rescan()
    _wait_until(lambda: controller.pending_registration_preview is not None)
    preview = controller.pending_registration_preview
    assert preview is not None
    assert all(p.action in {"reuse", "update", "skip"} for p in preview.proposals)
    result = controller.commit_pending_registration(operator_approved=True)
    assert controller.catalog is not None
    assert controller.catalog.catalog_id == original_ids["catalog"]
    assert controller.project_id == original_ids["project"]
    assert controller.session_id == original_ids["session"]
    assert result.committed_session_ids


def test_rebind_approval_and_uuid_retention(tmp_path: Path) -> None:
    old_root = _prep(tmp_path, "OldSession")
    new_root = _prep(tmp_path, "NewSession")
    catalog = build_legacy_self_project_catalog(old_root)
    # Keep catalog outside the session tree so relocating the session leaves it readable.
    catalog_path = tmp_path / "workspace_catalog_v1.json"
    path = write_catalog(catalog, catalog_path)
    session_id = catalog.sessions[0].session_id
    # Simulate missing old path by renaming after registration.
    old_root.rename(tmp_path / "missing_old")
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=_settings(tmp_path),
        nav_probe=lambda: (False, False, False),
    )
    controller.activate_registered_session(
        path,
        session_id=session_id,
        require_preprocess=False,
    )
    assert controller.catalog is not None
    preview = preview_rebind(
        controller.catalog,
        session_id=session_id,
        new_session_root=new_root,
    )
    controller._pending_rebind = preview  # noqa: SLF001 — test injects preview
    updated = controller.commit_pending_rebind(
        operator_approved=True,
        allow_without_fingerprint_match=True,
        catalog_path=path,
    )
    assert updated.get_session(session_id).session_root == new_root
    assert updated.get_session(session_id).session_id == session_id
    assert updated.project.project_id == catalog.project.project_id
    assert controller.discovery_root == new_root


def test_missing_path_recent_blocked(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    catalog = build_legacy_self_project_catalog(session)
    catalog_path = write_catalog(catalog)
    entry = RecentTargetEntry(
        project_id=catalog.project.project_id,
        session_id=catalog.sessions[0].session_id,
        subsession_id=None,
        analysis_target_id=f"session:{catalog.sessions[0].session_id}",
        project_root=session,
        session_root=session,
        discovery_root=session,
        display_name="gone",
        catalog_path=catalog_path,
    )
    session.rename(tmp_path / "relocated_away")
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=_settings(tmp_path),
        nav_probe=lambda: (False, False, False),
    )
    controller.context.set_recent_targets([entry])
    with pytest.raises(WorkspaceValidationError, match="missing|incomplete"):
        controller.activate_recent(entry)


def test_unsafe_switching_during_hard_and_soft_jobs(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    other = _prep(tmp_path, "Other")
    hard = {"locked": False}
    soft = {"running": False}
    controller = WorkspaceShellController(
        WorkspaceContext(),
        settings=_settings(tmp_path),
        nav_probe=lambda: (hard["locked"], soft["running"], False),
    )
    controller.open_session_path(session, require_preprocess=False)

    hard["locked"] = True
    with pytest.raises(WorkspaceNavigationBlocked, match="hard-locked"):
        controller.open_session_path(other, require_preprocess=False)
    assert controller.discovery_root == session

    hard["locked"] = False
    soft["running"] = True
    with pytest.raises(WorkspaceNavigationBlocked, match="soft-nav"):
        controller.open_session_path(other, require_preprocess=False)
    assert controller.discovery_root == session


def test_mainwindow_preserves_navigation_and_legacy_session_root(
    tmp_path: Path,
) -> None:
    session = _prep(tmp_path)
    window = MainWindow(PreprocessSetupController())
    assert window.workspace is not None
    assert window.workspace_controller is not None
    assert window.workspace_selector is not None

    window._remember_session(session)
    assert window._session_root == session
    assert window.workspace.discovery_root == session

    # Avoid page discovery on a synthetic tree; preserve soft-nav + run selection.
    window._session_root = None
    controller = window.blob_features_controller
    prior_run = controller.state.selected_s5_run_id
    window.open_blob_features()
    window.set_application_group(GROUP_DATA_PREPARATION)
    window.show_subsystem_1()
    assert controller.state.selected_s5_run_id == prior_run
    # Shared workspace context remains the source of truth for the last open.
    assert window.workspace.discovery_root == session
    assert window.blob_features_page is window.pages.widget(4)


def test_no_unwanted_scientific_writes_on_preview(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    science = session / "pose_inference" / "fake_run"
    science.mkdir(parents=True)
    marker = science / "run_meta.json"
    marker.write_text("{}", encoding="utf-8")
    before = marker.read_bytes()
    preview = preview_registration(session)
    assert preview.proposals
    assert marker.read_bytes() == before
    assert not catalog_path_for_root(session).is_file()


def test_mainwindow_blocks_workspace_switch_while_s2_running(tmp_path: Path) -> None:
    session = _prep(tmp_path)
    other = _prep(tmp_path, "Other")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(session, require_preprocess=False)

    class _Busy:
        has_running_task = True

    window.pose_page = _Busy()  # type: ignore[assignment]
    with pytest.raises(WorkspaceNavigationBlocked):
        window.workspace_controller.open_session_path(other, require_preprocess=False)
    assert window.workspace.discovery_root == session
