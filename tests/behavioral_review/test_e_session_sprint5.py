"""S10 Sprint 5 E-session laboratory validation (inspection + controlled GUI path).

Uses eligible package ``results_package__20261028T181714`` when present.
Controlled synthetic edits are **not** genuine biological annotations.
Does not modify S8/S9 automated artifacts.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QApplication

from behavioral_review.discovery import (
    is_s10_run_eligible_for_downstream,
    reopen_reviewed_run,
    summarize_behavioral_review_project,
)
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from results_packaging.publication import is_s9_run_eligible_for_downstream
from ui.controllers.behavioral_review_controller import BehavioralReviewController

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE = SESSION / "results_package" / "results_package__20261028T181714"
ISOLATED_REVIEW_ROOT = (
    Path(r"C:\behavior_suite\tmp_s10_sprint5_review_gui")
    / "isolated_e_session_controlled_gui"
)

EXPECTED_N_DENSE = 36528
EXPECTED_N_BOUTS = 609

os_environ_qt = __import__("os").environ
os_environ_qt.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _e_session_package_currently_eligible() -> bool:
    if not PACKAGE.is_dir():
        return False
    try:
        return bool(is_s9_run_eligible_for_downstream(PACKAGE))
    except Exception:  # noqa: BLE001
        return False


def _skip_reason() -> str:
    if not PACKAGE.is_dir():
        return (
            "E-session package results_package__20261028T181714 is missing; "
            "skip Sprint 5 E-session GUI validation."
        )
    try:
        if is_s9_run_eligible_for_downstream(PACKAGE):
            return "unexpected: package became eligible during skip-reason build"
    except Exception as exc:  # noqa: BLE001
        return (
            "E-session package results_package__20261028T181714 is not currently "
            f"eligible for S10 open ({exc}); report missing dependencies rather "
            "than weakening gates."
        )
    return "E-session package not currently eligible."


pytestmark = pytest.mark.skipif(
    not _e_session_package_currently_eligible(),
    reason=_skip_reason(),
)


def _wait_until(predicate, timeout_sec: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for E-session GUI state")
        time.sleep(0.02)
    _APPLICATION.processEvents()


def test_e_session_inspection_workspace_counts_and_sync() -> None:
    controller = BehavioralReviewController()
    identity = controller.prepare_session(SESSION)
    snap = controller.collect_discovery_snapshot(identity)
    assert controller.apply_discovery_snapshot(snap)
    controller.select_s9_package(PACKAGE.name)
    open_id = controller.begin_open_automated()
    open_snap = controller.open_source_worker(open_id)
    assert controller.apply_open_snapshot(open_snap)

    assert controller.state.n_dense == EXPECTED_N_DENSE
    assert controller.state.n_bouts == EXPECTED_N_BOUTS
    assert len(controller.state.directions) >= 2

    session = controller.state.editing_session
    assert session is not None
    snaps = snapshot_upstream_fingerprints(session.inspection.source)

    # Dual clocks + plateau-safe seek across a window.
    t0 = time.perf_counter()
    for sample in (0, 1, 2, 120, 240, EXPECTED_N_DENSE // 2, EXPECTED_N_DENSE - 1):
        decode_id = controller.seek_sample(sample)
        if decode_id is not None:
            controller.apply_decode_snapshot(controller.decode_worker(decode_id))
        win_id = controller.begin_window_request(half_window=120, include_s7=False)
        controller.apply_window_snapshot(controller.window_worker(win_id))
        assert controller.state.sample_idx == sample
        assert controller.state.timeline is not None
    elapsed = time.perf_counter() - t0

    assert controller.state.evidence_text
    # Bout table is windowed/capped for GUI responsiveness; use full packaged
    # channel set for scientific channel coverage.
    packaged = session.inspection.source.behavior_bouts
    channels = set(str(x) for x in packaged["channel"].unique().tolist())
    assert "exclusive" in channels
    assert "contact" in channels
    assert "side_by_side" in channels

    src = session.inspection.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path if src.source_video else None,
        snapshots=snaps,
    )
    # Soft responsiveness budget for seven seeks + windows on E-session scale.
    assert elapsed < 90.0, f"Inspection scrub too slow: {elapsed:.1f}s"
    print(f"E-session inspection scrub elapsed_sec={elapsed:.2f}")


def test_e_session_controlled_edit_publish_reopen(tmp_path: Path) -> None:
    token = uuid4().hex[:12]
    review_root = ISOLATED_REVIEW_ROOT / f"{tmp_path.name}_{token}"
    review_root.mkdir(parents=True, exist_ok=True)

    controller = BehavioralReviewController()
    identity = controller.prepare_session(SESSION)
    snap = controller.collect_discovery_snapshot(identity)
    assert controller.apply_discovery_snapshot(snap)
    controller.select_s9_package(PACKAGE.name)
    open_id = controller.begin_open_automated()
    assert controller.apply_open_snapshot(controller.open_source_worker(open_id))

    session = controller.state.editing_session
    assert session is not None
    snaps = snapshot_upstream_fingerprints(session.inspection.source)

    controller.set_reviewer_id(f"s10s5_controlled_{token}")
    controller.enter_editing_mode()
    controller.create_draft(review_root=review_root)

    win_id = controller.begin_window_request(half_window=60)
    controller.apply_window_snapshot(controller.window_worker(win_id))
    exclusive = [r for r in controller.state.bout_rows if r["channel"] == "exclusive"]
    assert exclusive, "Expected exclusive bouts on E-session"
    first = exclusive[0]
    controller.select_bout(str(first["bout_id"]))

    # Controlled (non-biological) edits.
    start = int(first["start_sample_idx"])
    end = int(first["end_sample_idx"])
    mid = start + max(1, (end - start) // 2)
    controller.set_exclusive_label(
        start_sample_idx=start,
        end_sample_idx=min(start + 5, end),
        label="nose_to_body",
        focal_id=int(first["focal_id"]) if first["focal_id"] is not None else 0,
        partner_id=int(first["partner_id"]) if first["partner_id"] is not None else 1,
    )
    controller.trim_selected_bout(
        new_start_sample_idx=start,
        new_end_sample_idx=max(start, end - 1),
    )
    controller.set_parallel_pred(
        channel="contact",
        start_sample_idx=start,
        end_sample_idx=min(start + 3, EXPECTED_N_DENSE - 1),
        pred_value=0,
    )
    controller.set_bout_status("accepted")
    controller.add_note(
        "CONTROLLED S10-5 GUI VALIDATION — not a genuine biological annotation."
    )
    assert controller.state.can_undo
    controller.undo()
    assert controller.state.can_redo
    controller.redo()
    controller.set_bout_status("rejected")

    # Restart recovery: reopen draft from isolated review root (not session tree).
    run_id = controller.state.review_run_id
    run_dir = controller.state.review_run_dir
    assert run_id is not None and run_dir is not None
    controller.close_session(release_lock=True)

    from ui.controllers.behavioral_review_controller import ReviewRunOption

    controller2 = BehavioralReviewController()
    identity2 = controller2.prepare_session(SESSION)
    snap2 = controller2.collect_discovery_snapshot(identity2)
    assert controller2.apply_discovery_snapshot(snap2)
    # Isolated controlled drafts are under review_root, not session/behavioral_review.
    summary_drafts = summarize_behavioral_review_project(
        SESSION, review_root=review_root
    )
    assert any(r.review_run_id == run_id for r in summary_drafts.runs)
    controller2.state.review_options = (
        ReviewRunOption(
            run_id=run_id,
            run_dir=Path(run_dir),
            classification="draft",
            currently_eligible=False,
            parent_package_run_id=PACKAGE.name,
            publication_state="draft",
            historical_status=None,
            superseded_by=None,
            label=f"{run_id} · draft",
            is_draft=True,
        ),
    )
    # Ensure parent package remains selectable/openable.
    from ui.controllers.behavioral_review_controller import S9PackageOption

    controller2.state.s9_options = tuple(
        S9PackageOption(
            run_id=o.run_id,
            run_dir=o.run_dir,
            historical_status=o.historical_status,
            classification="eligible",
            currently_eligible=True,
            n_bouts=o.n_bouts,
            n_dense_samples=o.n_dense_samples,
            label=o.label,
            reasons=(),
        )
        for o in controller2.state.s9_options
    )
    controller2.select_s9_package(PACKAGE.name)
    controller2.select_review_run(run_id)
    controller2.set_reviewer_id(f"s10s5_controlled_{token}")
    open_draft = controller2.begin_open_draft()
    open_snap = controller2.open_source_worker(
        open_draft,
        reviewer_id=f"s10s5_controlled_{token}",
        force_stale_lock=True,
    )
    assert open_snap.draft_resumed
    assert controller2.apply_open_snapshot(
        open_snap,
        reviewer_id=f"s10s5_controlled_{token}",
        force_stale_lock=True,
    )
    assert controller2.state.draft_open

    preview = controller2.build_publish_preview()
    if not preview.eligible:
        pytest.fail("Publication unexpectedly blocked: " + "; ".join(preview.blockers))

    pub_id = controller2.begin_publish()
    result = controller2.publish_worker(pub_id)
    assert controller2.apply_publish_result(pub_id, result)
    assert result.success, result.error_message
    assert is_s10_run_eligible_for_downstream(result.review_run_dir)
    reopened = reopen_reviewed_run(result.review_run_dir)
    assert reopened

    # Parent S8/S9 untouched.
    assert_no_upstream_writes(
        package_dir=PACKAGE,
        s8_run_dir=session.inspection.source.source_s8.run_dir,
        video_path=session.inspection.source.source_video.path,
        snapshots=snaps,
    )

    summary = summarize_behavioral_review_project(SESSION, review_root=review_root)
    assert any(r.review_run_id == result.review_run_id for r in summary.runs)
    del mid
