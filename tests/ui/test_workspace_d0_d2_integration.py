"""D0-D2 integration: shared workspace context across S1–S10 pages."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from ui.controllers.behavioral_review_controller import (
    BehavioralReviewController,
    BehavioralReviewUiError,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.workspace.page_binding import request_shared_session
from workspace.errors import WorkspaceNavigationBlocked
from workspace.migrate import commit_registration, preview_registration
from workspace.models import catalog_path_for_root

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APPLICATION = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_blocking_qt_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offscreen CI cannot dismiss modal QMessageBox dialogs."""

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: 0))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: 0))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: 0))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: 0))

E_SESSION = (
    Path(__file__).resolve().parents[2] / "test_cases" / "E_full_session_integration"
)
F_SESSION = (
    Path(__file__).resolve().parents[2] / "test_cases" / "F_v011_fresh_short_smoke"
)


def _prep(tmp_path: Path, name: str = "SessionA") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "preprocess").mkdir()
    return root.resolve()


def _soft_pages(window: MainWindow) -> tuple:
    return (
        window.blob_features_page,
        window.combined_features_page,
        window.behavioral_inference_page,
        window.behavioral_arbitration_page,
        window.results_packaging_page,
        window.behavioral_review_page,
    )


def _wait_soft_idle(window: MainWindow, *, timeout_sec: float = 8.0) -> None:
    """Let S5–S10 discovery workers finish so workspace switches are allowed."""

    deadline = time.perf_counter() + timeout_sec
    while time.perf_counter() < deadline:
        _APPLICATION.processEvents()
        soft_busy = any(page.has_running_task for page in _soft_pages(window))
        hard_busy = (
            window.pose_page.has_running_task
            or window.finalization_page.has_running_task
        )
        if not soft_busy and not hard_busy:
            return
        time.sleep(0.01)
    raise AssertionError("soft/hard page tasks still running")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _science_fingerprint(session: Path) -> dict[str, str]:
    """Collect SHA256 of a few immutable science artifacts when present."""

    out: dict[str, str] = {}
    candidates = [
        session / "preprocess" / "prepare_meta.json",
        session / "preprocess" / "prepared_sync.json",
    ]
    packages = session / "results_package"
    if packages.is_dir():
        for manifest in sorted(packages.glob("*/package_manifest.json"))[:2]:
            candidates.append(manifest)
    for path in candidates:
        if path.is_file():
            out[str(path.relative_to(session))] = _sha256(path)
    return out


def test_browse_s1_then_open_s8_s10_shares_session(tmp_path: Path) -> None:
    session = _prep(tmp_path, "FromS1")
    window = MainWindow(PreprocessSetupController())
    window._remember_session(session)
    assert window.workspace.discovery_root == session
    assert window._session_root == session

    window.open_behavioral_arbitration()
    assert window.behavioral_arbitration_controller.state.session_root == session
    window.open_behavioral_review()
    assert window.behavioral_review_controller.state.session_root == session
    assert window.behavioral_review_controller.workspace_discovery_root == session
    _wait_soft_idle(window)


def test_browse_s9_then_open_s2_s5_shares_session(tmp_path: Path) -> None:
    session = _prep(tmp_path, "FromS9")
    window = MainWindow(PreprocessSetupController())
    # Simulate page Browse write-back while S9 is the active page.
    window.pages.setCurrentWidget(window.results_packaging_page)
    assert request_shared_session(window.results_packaging_page, session) is True
    assert window.workspace.discovery_root == session
    assert window.results_packaging_controller.state.session_root == session
    _wait_soft_idle(window)

    window.open_pose_inference()
    assert window.pose_controller.state.session_root == session
    # S2 discovery runs asynchronously and hard-locks navigation until it finishes.
    _wait_soft_idle(window)
    window.open_blob_features()
    assert window.blob_features_controller.state.session_root == session
    _wait_soft_idle(window)


@pytest.mark.parametrize(
    "page_attr",
    [
        "pose_page",
        "finalization_page",
        "blob_features_page",
        "combined_features_page",
        "behavioral_inference_page",
        "behavioral_arbitration_page",
        "results_packaging_page",
        "behavioral_review_page",
    ],
)
def test_page_browse_updates_shared_context(tmp_path: Path, page_attr: str) -> None:
    session = _prep(tmp_path, f"Browse_{page_attr}")
    window = MainWindow(PreprocessSetupController())
    page = getattr(window, page_attr)
    # S4 lives inside s4_shell on the stack (not as a direct stack page).
    if page_attr == "finalization_page":
        window.pages.setCurrentWidget(window.s4_shell)
    else:
        window.pages.setCurrentWidget(page)
    assert request_shared_session(page, session) is True
    assert window.workspace.discovery_root == session
    assert window._session_root == session
    assert getattr(page.controller.state, "session_root", None) == session
    _wait_soft_idle(window)


def test_ephemeral_and_registered_sessions(tmp_path: Path) -> None:
    ephemeral = _prep(tmp_path, "Ephemeral")
    registered = _prep(tmp_path, "Registered")
    window = MainWindow(PreprocessSetupController())

    window.workspace_controller.open_session_path(ephemeral, require_preprocess=True)
    assert window.workspace.ephemeral is True
    assert not catalog_path_for_root(ephemeral).is_file()

    preview = preview_registration(registered)
    result = commit_registration(preview, operator_approved=True)
    window.workspace_controller.activate_registered_session(
        result.catalog_path,
        session_id=result.committed_session_ids[0],
        require_preprocess=False,
    )
    assert window.workspace.ephemeral is False
    assert window.workspace.discovery_root == registered
    assert window._session_root == registered


def test_incomplete_session_listable_and_openable_without_catalog(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "Incomplete"
    incomplete.mkdir()
    # No preprocess/ — still browseable with require_preprocess=False.
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(
        incomplete,
        require_preprocess=False,
    )
    assert window.workspace.discovery_root == incomplete.resolve()
    assert window.workspace_controller.registration_status().incomplete is True


def test_switching_among_sessions_invalidates_run_selection(tmp_path: Path) -> None:
    first = _prep(tmp_path, "First")
    second = _prep(tmp_path, "Second")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(first, require_preprocess=False)
    window.open_combined_features()
    _wait_soft_idle(window)
    controller = window.combined_features_controller
    controller.state.selected_s4_run_id = "s4_from_first"
    controller.state.selected_s5_run_id = "s5_from_first"
    controller.state.selected_s6_run_id = "s6_from_first"

    window.workspace_controller.open_session_path(second, require_preprocess=False)
    assert controller.state.session_root == second
    assert controller.state.selected_s4_run_id is None
    assert controller.state.selected_s5_run_id is None
    assert controller.state.selected_s6_run_id is None


def test_same_root_workspace_signal_preserves_run_selection(tmp_path: Path) -> None:
    session = _prep(tmp_path, "SameRoot")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(session, require_preprocess=False)
    window.open_pose_inference()
    _wait_soft_idle(window)
    window.pose_controller.state.selected_run_id = "keep_me"
    # Re-open same path (generation bumps) must not clear S2 selection.
    window.workspace_controller.open_session_path(session, require_preprocess=False)
    assert window.pose_controller.state.selected_run_id == "keep_me"
    assert window.pose_controller.state.session_root == session


def test_stale_worker_callback_rejected_after_target_change(tmp_path: Path) -> None:
    first = _prep(tmp_path, "WorkerA")
    second = _prep(tmp_path, "WorkerB")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(first, require_preprocess=False)
    window.open_results_packaging()
    _wait_soft_idle(window)
    controller = window.results_packaging_controller
    identity = controller.prepare_session(first)
    snapshot = controller.collect_discovery_snapshot(identity)
    # Finish the synchronous rediscovery bookkeeping without leaving workers busy.
    controller.state.discovery_running = False
    controller.state.task_running = False

    window.workspace_controller.open_session_path(second, require_preprocess=False)
    assert controller.state.session_root == second
    # Stale discovery for the previous target must be ignored.
    assert controller.apply_discovery_snapshot(snapshot) is False


def test_s10_context_switch_closes_draft_without_auto_reopen(tmp_path: Path) -> None:
    first = _prep(tmp_path, "DraftA")
    second = _prep(tmp_path, "DraftB")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(first, require_preprocess=False)
    window.open_behavioral_review()
    _wait_soft_idle(window)
    review = window.behavioral_review_controller
    review.state.draft_open = True
    review.state.editing_session = MagicMock()
    review.state.editing_session.draft = MagicMock()
    review.state.session_root = first

    closed: list[bool] = []

    def _close(*, release_lock: bool = True) -> None:
        closed.append(release_lock)
        review.state.draft_open = False
        review.state.editing_session = None
        review.state.session_root = None

    review.close_session = _close  # type: ignore[method-assign]
    window.workspace_controller.open_session_path(second, require_preprocess=False)
    # Workspace change releases the draft; page rebind may close again via prepare_session.
    assert closed and all(closed)
    assert review.state.draft_open is False
    # Returning to S10 binds the new target; does not invent a draft reopen.
    window.open_behavioral_review()
    _wait_soft_idle(window)
    assert review.state.session_root == second
    assert review.state.draft_open is False


def test_context_switch_blocked_during_s10_mutation(tmp_path: Path) -> None:
    first = _prep(tmp_path, "MutA")
    second = _prep(tmp_path, "MutB")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(first, require_preprocess=False)
    window.behavioral_review_controller.state.edit_running = True
    window.behavioral_review_controller.state.active_edit_identity = MagicMock()
    with pytest.raises(WorkspaceNavigationBlocked, match="soft-nav|task"):
        window.workspace_controller.open_session_path(second, require_preprocess=False)
    assert window.workspace.discovery_root == first


def test_context_switch_blocked_during_s10_publication(tmp_path: Path) -> None:
    first = _prep(tmp_path, "PubA")
    second = _prep(tmp_path, "PubB")
    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(first, require_preprocess=False)
    window.behavioral_review_controller.state.publish_running = True
    window.behavioral_review_controller.state.active_publish_identity = MagicMock()
    with pytest.raises(WorkspaceNavigationBlocked):
        window.workspace_controller.open_session_path(second, require_preprocess=False)
    assert window.workspace.discovery_root == first


def test_s10_publish_blocked_on_workspace_target_mismatch(tmp_path: Path) -> None:
    session = _prep(tmp_path, "PublishMatch")
    other = _prep(tmp_path, "OtherTarget")
    controller = BehavioralReviewController()
    controller.state.session_root = session
    controller.set_workspace_discovery_root(other)
    draft = MagicMock()
    draft.review_run_id = "review_x"
    source = MagicMock()
    source.package_dir = session / "results_package" / "pkg"
    source.package_dir.mkdir(parents=True)
    inspection = MagicMock()
    inspection.source = source
    editing = MagicMock()
    editing.draft = draft
    editing.inspection = inspection
    editing.package_run_id = "pkg"
    controller.state.editing_session = editing
    controller.state.editing_active = True
    controller.state.draft_open = True
    controller.state.review_run_dir = session / "behavioral_review" / "review_x"
    controller.state.review_run_dir.mkdir(parents=True)

    with pytest.raises(BehavioralReviewUiError, match="workspace analysis target"):
        controller.build_publish_preview()


def test_s10_publish_blocked_when_package_outside_session(tmp_path: Path) -> None:
    session = _prep(tmp_path, "PubSession")
    foreign = _prep(tmp_path, "ForeignPkgRoot")
    controller = BehavioralReviewController()
    controller.state.session_root = session
    controller.set_workspace_discovery_root(session)
    draft = MagicMock()
    draft.review_run_id = "review_y"
    source = MagicMock()
    source.package_dir = foreign / "results_package" / "pkg"
    source.package_dir.mkdir(parents=True)
    inspection = MagicMock()
    inspection.source = source
    editing = MagicMock()
    editing.draft = draft
    editing.inspection = inspection
    editing.package_run_id = "pkg"
    controller.state.editing_session = editing
    controller.state.editing_active = True
    controller.state.draft_open = True

    with pytest.raises(BehavioralReviewUiError, match="not under the active"):
        controller._assert_publish_matches_active_target()


def test_s10_publish_blocked_for_similar_prefix_sibling_session(
    tmp_path: Path,
) -> None:
    """Sibling dirs sharing a path prefix must not pass package containment."""

    session_a = _prep(tmp_path, "session_A")
    session_a_copy = _prep(tmp_path, "session_A_copy")
    controller = BehavioralReviewController()
    controller.state.session_root = session_a
    controller.set_workspace_discovery_root(session_a)
    draft = MagicMock()
    draft.review_run_id = "review_sibling"
    source = MagicMock()
    source.package_dir = session_a_copy / "results_package" / "pkg"
    source.package_dir.mkdir(parents=True)
    inspection = MagicMock()
    inspection.source = source
    editing = MagicMock()
    editing.draft = draft
    editing.inspection = inspection
    editing.package_run_id = "pkg"
    controller.state.editing_session = editing
    controller.state.editing_active = True
    controller.state.draft_open = True
    controller.state.review_run_dir = session_a_copy / "behavioral_review" / "review_sibling"
    controller.state.review_run_dir.mkdir(parents=True)

    with pytest.raises(BehavioralReviewUiError, match="not under the active"):
        controller._assert_publish_matches_active_target()


def test_workspace_navigation_does_not_modify_science_artifacts(tmp_path: Path) -> None:
    session = _prep(tmp_path, "NoScienceWrite")
    science = session / "pose_inference" / "fake_run"
    science.mkdir(parents=True)
    marker = science / "run_meta.json"
    marker.write_text('{"status":"completed"}', encoding="utf-8")
    before = marker.read_bytes()
    window = MainWindow(PreprocessSetupController())
    window.pages.setCurrentWidget(window.pose_page)
    window._open_session_from_page_browse(session)
    window.open_pose_inference()
    window.open_behavioral_review()
    _wait_soft_idle(window)
    other = _prep(tmp_path, "OtherNoWrite")
    window.workspace_controller.open_session_path(other, require_preprocess=False)
    assert marker.read_bytes() == before
    assert not catalog_path_for_root(session).is_file()


def test_no_duplicate_page_workspace_contexts() -> None:
    window = MainWindow(PreprocessSetupController())
    assert window.workspace_controller.context is window.workspace
    for page in (
        window.pose_page,
        window.blob_features_page,
        window.behavioral_review_page,
    ):
        assert getattr(page, "workspace", None) is None
        assert getattr(page, "_workspace_session_opener", None) is not None


@pytest.mark.skipif(
    not E_SESSION.is_dir() or not F_SESSION.is_dir(),
    reason="E/F laboratory sessions not available",
)
def test_laboratory_e_and_f_shared_context_smoke() -> None:
    """Read-only lab smoke: select E then F; navigate; confirm science unchanged."""

    e_before = _science_fingerprint(E_SESSION)
    f_before = _science_fingerprint(F_SESSION)
    assert e_before, "E session missing expected science artifacts"
    assert f_before, "F session missing expected science artifacts"

    window = MainWindow(PreprocessSetupController())
    window.workspace_controller.open_session_path(E_SESSION, require_preprocess=False)
    assert window.workspace.discovery_root == E_SESSION.resolve()
    window.open_pose_inference()
    window.open_behavioral_arbitration()
    _wait_soft_idle(window, timeout_sec=60.0)
    window.open_results_packaging()
    window.open_behavioral_review()
    _wait_soft_idle(window, timeout_sec=60.0)
    assert window.behavioral_review_controller.state.session_root == E_SESSION.resolve()

    window.workspace_controller.open_session_path(F_SESSION, require_preprocess=False)
    assert window.workspace.discovery_root == F_SESSION.resolve()
    window.open_blob_features()
    window.open_combined_features()
    _wait_soft_idle(window, timeout_sec=60.0)
    window.open_behavioral_inference()
    window.open_behavioral_review()
    _wait_soft_idle(window, timeout_sec=60.0)
    assert window.behavioral_review_controller.state.session_root == F_SESSION.resolve()
    assert window.behavioral_review_controller.state.draft_open is False

    # Switch back to previously analyzed E-session.
    window.workspace_controller.open_session_path(E_SESSION, require_preprocess=False)
    window.open_results_packaging()
    _wait_soft_idle(window, timeout_sec=60.0)
    assert window.results_packaging_controller.state.session_root == E_SESSION.resolve()

    assert _science_fingerprint(E_SESSION) == e_before
    assert _science_fingerprint(F_SESSION) == f_before
