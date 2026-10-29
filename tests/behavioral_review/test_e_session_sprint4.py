"""S10 Sprint 4 E-session controlled publication validation.

Uses eligible package ``results_package__20261028T181714`` when present.
Creates an isolated controlled draft with synthetic review decisions, publishes
a unique S10 reviewed run, and verifies immutability / eligibility / reopen.

Controlled synthetic decisions are **not** genuine biological annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from behavioral_review.contracts import (
    SOURCE_TYPE_REVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
)
from behavioral_review.discovery import (
    is_s10_run_eligible_for_downstream,
    reopen_reviewed_run,
    validate_s10_run_for_downstream,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from project.file_digest import sha256_file
from results_packaging.publication import is_s9_run_eligible_for_downstream

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE = SESSION / "results_package" / "results_package__20261028T181714"
ISOLATED_REVIEW_ROOT = (
    Path(r"C:\behavior_suite\tmp_s10_sprint4_review_publications")
    / "isolated_e_session_controlled_publish"
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
            "skip Sprint 4 E-session publication."
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


def test_e_session_controlled_publication(tmp_path: Path) -> None:
    token = uuid4().hex[:12]
    review_root = ISOLATED_REVIEW_ROOT / f"{tmp_path.name}_{token}"
    review_root.mkdir(parents=True, exist_ok=True)
    review_run_id = f"behavioral_review__s10s4_e_{token}"

    session = open_editing_session(
        PACKAGE,
        require_video=False,
        require_s8_states=True,
    )
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    auto_bouts_path = PACKAGE / "behavior_bouts.parquet"
    auto_before = pd.read_parquet(auto_bouts_path)
    auto_sha_before = sha256_file(auto_bouts_path)
    s8_states = session.inspection.source.source_s8.states_path
    assert s8_states is not None
    s8_states_sha_before = sha256_file(s8_states)
    s8_bouts = session.inspection.source.source_s8.bouts_path
    assert s8_bouts is not None
    s8_bouts_sha_before = sha256_file(s8_bouts)

    assert session.n_dense_samples == 36528
    assert len(auto_before) == 609

    session.enter_editing_mode("e_session_sprint4_controlled_tester")
    draft = session.create_draft(
        review_root=review_root,
        review_run_id=review_run_id,
    )

    exclusive = auto_before[auto_before["channel"] == "exclusive"].iloc[0]
    source_id = str(exclusive["source_s8_bout_id"])
    focal = int(exclusive["focal_id"])
    partner = int(exclusive["partner_id"])
    start = int(exclusive["start_sample_idx"])
    end = int(exclusive["end_sample_idx"])
    mid = start + max(0, (end - start) // 2)

    # Controlled synthetic edits (not biological truth).
    session.set_exclusive_label(
        focal_id=focal,
        partner_id=partner,
        start_sample_idx=mid,
        end_sample_idx=mid,
        label="nose_to_body",
    )
    if end > start:
        session.trim_bout_boundary(
            target_bout_id=source_id,
            new_start_sample_idx=start,
            new_end_sample_idx=end - 1,
            fill_label="mutually_disengaged",
        )
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
    other = auto_before[auto_before["channel"] == "exclusive"].iloc[1]
    other_id = str(other["source_s8_bout_id"])
    session.set_bout_review_status(target_bout_id=source_id, status=STATUS_REJECTED)
    session.set_bout_review_status(target_bout_id=other_id, status=STATUS_ACCEPTED)

    # Undo then redo to exercise cursor; then leave an extra undone branch
    # superseded by a new edit so history retains superseded events.
    session.undo()
    session.redo()
    session.undo()
    session.set_bout_review_status(target_bout_id=other_id, status=STATUS_ACCEPTED)

    result = session.publish_reviewed_run(
        settings={
            "note": (
                "Controlled synthetic Sprint 4 publication — not genuine "
                "biological annotations."
            )
        }
    )
    assert result.success, result.error_message
    assert result.source_type == SOURCE_TYPE_REVIEWED
    assert result.currently_eligible
    run_dir = result.review_run_dir

    # Dense sample + directional cardinality.
    states = pd.read_parquet(run_dir / "reviewed_behavior_states.parquet")
    assert states["dense_frame_idx"].nunique() == 36528
    assert int(len(states)) == 36528 * 2
    counts = states.groupby("dense_frame_idx").size()
    assert set(counts.unique().tolist()) == {2}

    bouts = pd.read_parquet(run_dir / "reviewed_behavior_bouts.parquet")
    assert len(bouts) > 0
    assert all(str(b).startswith("rb__") for b in bouts["bout_id"].astype(str))
    # Inclusive durations.
    for _, row in bouts.head(50).iterrows():
        expected = int(row["end_sample_idx"]) - int(row["start_sample_idx"]) + 1
        assert int(row["bout_duration_frames"]) == expected
        assert abs(float(row["bout_duration_sec"]) - expected / 120.0) < 1e-9

    applied = pd.read_parquet(run_dir / "correction_events.parquet")
    history = pd.read_parquet(run_dir / "correction_events_history.parquet")
    assert len(history) >= len(applied)
    assert history["applied"].astype(bool).sum() == len(applied)

    statuses = pd.read_parquet(run_dir / "review_bout_status.parquet")
    assert STATUS_REJECTED in set(statuses["status"].astype(str))
    assert STATUS_ACCEPTED in set(statuses["status"].astype(str))

    # Upstream immutability.
    assert sha256_file(auto_bouts_path) == auto_sha_before
    assert sha256_file(s8_states) == s8_states_sha_before
    assert sha256_file(s8_bouts) == s8_bouts_sha_before
    assert_no_upstream_writes(
        package_dir=PACKAGE,
        s8_run_dir=session.inspection.source.source_s8.run_dir,
        video_path=session.inspection.source.source_video.path,
        snapshots=snaps,
    )

    # Working draft intact after publish.
    assert (draft.working_dir / "correction_events.jsonl").is_file()
    assert draft.publication_state == "published"

    # Discovery / eligibility / reopen.
    validated = validate_s10_run_for_downstream(run_dir)
    assert validated["currently_eligible"] is True
    assert is_s10_run_eligible_for_downstream(run_dir)
    reopened = reopen_reviewed_run(run_dir)
    assert reopened["review_run_id"] == review_run_id
    assert int(len(reopened["reviewed_behavior_states"])) == 36528 * 2

    # Evidence sidecar for architectural review (not a commit).
    evidence = {
        "note": "Controlled synthetic Sprint 4 E-session publication evidence.",
        "biological_annotation": False,
        "review_run_id": review_run_id,
        "review_run_dir": str(run_dir),
        "parent_package": str(PACKAGE),
        "n_dense_samples": 36528,
        "n_reviewed_state_rows": int(len(states)),
        "n_reviewed_bouts": int(len(bouts)),
        "n_events_applied": int(len(applied)),
        "n_events_history": int(len(history)),
        "status": result.status,
        "currently_eligible": result.currently_eligible,
        "upstream_auto_bouts_unchanged": True,
        "upstream_s8_states_unchanged": True,
        "upstream_s8_bouts_unchanged": True,
    }
    evidence_path = review_root / "sprint4_e_session_publication_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
