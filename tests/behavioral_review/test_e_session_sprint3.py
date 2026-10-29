"""S10 Sprint 3 E-session controlled-edit validation.

Uses eligible package ``results_package__20261028T181714`` when present.
Creates an isolated review draft under a designated test location — never in
the production session ``behavioral_review/`` tree unless explicitly labeled.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from behavioral_review.contracts import STATUS_ACCEPTED, STATUS_REJECTED
from behavioral_review.editing import open_editing_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from results_packaging.publication import is_s9_run_eligible_for_downstream

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE = SESSION / "results_package" / "results_package__20261028T181714"
# Isolated draft root — explicitly labeled, outside session behavioral_review/.
ISOLATED_REVIEW_ROOT = (
    Path(r"C:\behavior_suite\tmp_s10_sprint3_review_drafts")
    / "isolated_e_session_controlled_edits"
)


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
            "skip Sprint 3 E-session controlled edits."
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


def test_e_session_controlled_edits_recoverable_draft(tmp_path: Path) -> None:
    token = uuid4().hex[:12]
    review_root = ISOLATED_REVIEW_ROOT / f"{tmp_path.name}_{token}"
    review_root.mkdir(parents=True, exist_ok=True)
    review_run_id = f"behavioral_review__s10s3_e_{token}"

    session = open_editing_session(
        PACKAGE,
        require_video=False,
        require_s8_states=True,
    )
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    auto_bouts_path = PACKAGE / "behavior_bouts.parquet"
    auto_before = pd.read_parquet(auto_bouts_path)
    assert len(auto_before) == 609

    # Inspection-only seeking must not create a draft.
    session.seek_sample(1000)
    assert session.draft is None

    session.enter_editing_mode("e_session_sprint3_tester")
    draft = session.create_draft(
        review_root=review_root,
        review_run_id=review_run_id,
    )

    # Pick a representative exclusive bout from the package.
    exclusive = auto_before[auto_before["channel"] == "exclusive"].iloc[0]
    source_id = str(exclusive["source_s8_bout_id"])
    focal = int(exclusive["focal_id"])
    partner = int(exclusive["partner_id"])
    start = int(exclusive["start_sample_idx"])
    end = int(exclusive["end_sample_idx"])
    mid = start + max(0, (end - start) // 2)

    # 1) Change a directional label (single sample, inclusive).
    session.set_exclusive_label(
        focal_id=focal,
        partner_id=partner,
        start_sample_idx=mid,
        end_sample_idx=mid,
        label="nose_to_body",
    )

    # 2) Adjust inclusive bout boundary (shrink by one sample at end if possible).
    if end > start:
        session.trim_bout_boundary(
            target_bout_id=source_id,
            new_start_sample_idx=start,
            new_end_sample_idx=end - 1,
            fill_label="mutually_disengaged",
        )

    # 3) Correct contact or SBS independently when a parallel bout exists.
    contact = auto_before[auto_before["channel"] == "contact"]
    if len(contact):
        crow = contact.iloc[0]
        session.set_parallel_pred(
            channel="contact",
            mouse_a_id=int(crow["mouse_a_id"]),
            mouse_b_id=int(crow["mouse_b_id"]),
            start_sample_idx=int(crow["start_sample_idx"]),
            end_sample_idx=int(crow["start_sample_idx"]),
            pred_value=0,
        )
    else:
        sbs = auto_before[auto_before["channel"] == "side_by_side"].iloc[0]
        session.set_parallel_pred(
            channel="side_by_side",
            mouse_a_id=int(sbs["mouse_a_id"]),
            mouse_b_id=int(sbs["mouse_b_id"]),
            start_sample_idx=int(sbs["start_sample_idx"]),
            end_sample_idx=int(sbs["start_sample_idx"]),
            pred_value=0,
        )

    # 4) Accept / reject a bout (status-only).
    other = auto_before[auto_before["channel"] == "exclusive"].iloc[1]
    other_id = str(other["source_s8_bout_id"])
    session.set_bout_review_status(target_bout_id=source_id, status=STATUS_REJECTED)
    session.set_bout_review_status(target_bout_id=other_id, status=STATUS_ACCEPTED)

    labels_before_undo = session.current_reviewed_states()[
        (session.current_reviewed_states()["focal_id"] == focal)
        & (session.current_reviewed_states()["partner_id"] == partner)
        & (session.current_reviewed_states()["dense_frame_idx"] == mid)
    ].iloc[0]["behavior_label"]
    assert str(labels_before_undo) == "nose_to_body"

    # 5) Undo / redo.
    session.undo()  # undo accept other
    session.undo()  # undo reject source
    assert session.overlay.status_for_source_bout(source_id) == "unreviewed"
    session.redo()
    assert session.overlay.status_for_source_bout(source_id) == STATUS_REJECTED

    snap = session.snapshot()
    cursor_seq = snap["journal_cursor_event_seq"]
    n_active = snap["n_active_events"]
    run_dir = draft.review_run_dir

    # 6) Reopen draft and verify exact recovery.
    session.close_draft()
    session2 = open_editing_session(
        PACKAGE, require_video=False, require_s8_states=True
    )
    session2.enter_editing_mode("e_session_sprint3_tester")
    session2.open_draft(run_dir)
    snap2 = session2.snapshot()
    assert snap2["journal_cursor_event_seq"] == cursor_seq
    assert snap2["n_active_events"] == n_active
    lab = session2.current_reviewed_states()[
        (session2.current_reviewed_states()["focal_id"] == focal)
        & (session2.current_reviewed_states()["partner_id"] == partner)
        & (session2.current_reviewed_states()["dense_frame_idx"] == mid)
    ].iloc[0]["behavior_label"]
    assert str(lab) == "nose_to_body"
    assert session2.overlay.status_for_source_bout(source_id) == STATUS_REJECTED

    # All 609 automated bouts unchanged in the source package.
    auto_after = pd.read_parquet(auto_bouts_path)
    assert len(auto_after) == 609
    assert auto_after.equals(auto_before)
    src = session2.inspection.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path,
        snapshots=snaps,
    )

    session2.close_draft()
