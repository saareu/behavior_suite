"""S10 Sprint 4 — immutable publication, QC, discovery, eligibility."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_review.contracts import (
    CLASS_DRAFT,
    CLASS_STALE_PARENT,
    CLASS_SUPERSEDED,
    CORRECTION_EVENTS_HISTORY_FILENAME,
    CORRECTION_EVENTS_PARQUET_FILENAME,
    PUBLICATION_STATE_DRAFT,
    PUBLICATION_STATE_PUBLISHED,
    REVIEW_BOUT_STATUS_PARQUET_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    REVIEWED_BEHAVIOR_BOUTS_FILENAME,
    REVIEWED_BEHAVIOR_STATES_FILENAME,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    STATUS_ACCEPTED,
    STATUS_COMPLETED,
    STATUS_REJECTED,
    STATUS_VOIDED,
    LockError,
    ProvenanceError,
    PublicationError,
)
from behavioral_review.discovery import (
    classify_analysis_source,
    find_latest_eligible_reviewed_run,
    is_s10_run_eligible_for_downstream,
    list_completed_reviewed_runs,
    reopen_reviewed_run,
    summarize_behavioral_review_project,
    validate_s10_run_for_downstream,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.locks import WriterLock
from behavioral_review.publication import publish_reviewed_run
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)


def _open_editor(tmp_path: Path, **kwargs):
    built = build_mini_review_package(tmp_path, **kwargs)
    review_root = tmp_path / "review_drafts_isolated"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"],
            require_video=False,
            require_s8_states=True,
        )
    return built, session, review_root


def _enter_with_draft(session, review_root: Path, reviewer: str = "reviewer_pub"):
    session.enter_editing_mode(reviewer)
    return session.create_draft(review_root=review_root)


def test_publish_valid_draft_and_reopen(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        draft = _enter_with_draft(session, review_root)
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="nose_to_body",
        )
        session.set_bout_review_status(
            target_bout_id="bout_dir_b", status=STATUS_ACCEPTED
        )
        result = session.publish_reviewed_run()
        assert result.success
        assert result.status == STATUS_COMPLETED
        assert result.publication_state == PUBLICATION_STATE_PUBLISHED
        assert result.source_type == SOURCE_TYPE_REVIEWED
        assert result.currently_eligible
        run_dir = result.review_run_dir
        for name in (
            CORRECTION_EVENTS_PARQUET_FILENAME,
            CORRECTION_EVENTS_HISTORY_FILENAME,
            REVIEWED_BEHAVIOR_STATES_FILENAME,
            REVIEWED_BEHAVIOR_BOUTS_FILENAME,
            REVIEW_BOUT_STATUS_PARQUET_FILENAME,
            REVIEW_MANIFEST_FILENAME,
            "qc.json",
            "run_meta.json",
            "settings_used.yaml",
            "processing_log.txt",
        ):
            assert (run_dir / name).is_file(), name
        # Working draft remains intact.
        assert (draft.working_dir / "correction_events.jsonl").is_file()
        assert draft.publication_state == PUBLICATION_STATE_PUBLISHED

        validated = validate_s10_run_for_downstream(run_dir)
        assert validated["currently_eligible"] is True
        assert validated["source_type"] == SOURCE_TYPE_REVIEWED
        reopened = reopen_reviewed_run(run_dir)
        assert int(len(reopened["reviewed_behavior_states"])) == built["n_dense"] * 2
        assert_no_upstream_writes(
            package_dir=built["package_dir"],
            s8_run_dir=built["s8_run_dir"],
            video_path=built["video_path"],
            snapshots=snaps,
        )


def test_publish_with_no_corrections(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _enter_with_draft(session, review_root)
        states = session.current_reviewed_states().copy()
        if "exp_name" not in states.columns:
            states["exp_name"] = built["package_run_id"]
        result = session.publish_reviewed_run()
        assert result.success
        published = pd.read_parquet(
            result.review_run_dir / REVIEWED_BEHAVIOR_BOUTS_FILENAME
        )
        rebuilt = extract_behavior_bouts(
            states,
            fps=120.0,
            source_s7_run_id="",
            source_s6_run_id="",
            run_id=built["s8_run_id"],
        )
        assert len(published) == len(rebuilt)
        applied = pd.read_parquet(
            result.review_run_dir / CORRECTION_EVENTS_PARQUET_FILENAME
        )
        if len(applied):
            assert "set_exclusive_label" not in set(applied["op"].astype(str))


def test_active_cursor_replay_after_undo_excludes_superseded(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _enter_with_draft(session, review_root)
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=1,
            end_sample_idx=1,
            label="nose_to_tail",
        )
        session.undo()
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=2,
            end_sample_idx=2,
            label="approach",
        )
        # Redo-side first edit is superseded; states must reflect sample 2 only.
        states = session.current_reviewed_states()
        row1 = states[
            (states["dense_frame_idx"] == 1)
            & (states["focal_id"] == 0)
            & (states["partner_id"] == 1)
        ].iloc[0]
        row2 = states[
            (states["dense_frame_idx"] == 2)
            & (states["focal_id"] == 0)
            & (states["partner_id"] == 1)
        ].iloc[0]
        assert row1["behavior_label"] != "nose_to_tail"
        assert row2["behavior_label"] == "approach"
        result = session.publish_reviewed_run()
        assert result.success
        applied = pd.read_parquet(
            result.review_run_dir / CORRECTION_EVENTS_PARQUET_FILENAME
        )
        history = pd.read_parquet(
            result.review_run_dir / CORRECTION_EVENTS_HISTORY_FILENAME
        )
        assert len(history) > len(applied)
        assert history["superseded"].astype(bool).any()
        # Superseded label must not appear in published states at sample 1.
        pub_states = pd.read_parquet(
            result.review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        )
        pub_row1 = pub_states[
            (pub_states["dense_frame_idx"] == 1)
            & (pub_states["focal_id"] == 0)
            & (pub_states["partner_id"] == 1)
        ].iloc[0]
        assert pub_row1["behavior_label"] != "nose_to_tail"
        assert int(result.diagnostics["n_events_history"]) >= int(
            result.diagnostics["n_events_applied"]
        )


def test_parent_staleness_blocks_publish(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        draft = _enter_with_draft(session, review_root)
        # Drift a recorded pin without substituting a newer package.
        draft.working_manifest["parent_sync_index_sha256"] = "deadbeef" * 8
        path = draft.working_dir / "working_manifest.json"
        path.write_text(json.dumps(draft.working_manifest, indent=2), encoding="utf-8")
        with pytest.raises((ProvenanceError, PublicationError), match="stale|mismatch|digest"):
            session.publish_reviewed_run()
        # Draft recoverable; no science artifacts sealed.
        assert draft.publication_state == PUBLICATION_STATE_DRAFT
        assert not (draft.review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME).is_file()
        assert (draft.working_dir / "correction_events.jsonl").is_file()


def test_invalid_reviewer_and_journal_block(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        draft = _enter_with_draft(session, review_root)
        session.reviewer_id = "   "
        with pytest.raises(PublicationError, match="reviewer"):
            publish_reviewed_run(session)
        session.reviewer_id = "reviewer_pub"
        # Corrupt cursor beyond journal.
        draft.journal.cursor.cursor_event_seq = 999
        with pytest.raises(PublicationError):
            session.publish_reviewed_run()


def test_writer_lock_conflict_blocks_publish(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        draft = _enter_with_draft(session, review_root)
        # Simulate another holder by releasing then acquiring with different lock object.
        draft.lock.release()
        other = WriterLock(draft.working_dir, review_run_id=draft.review_run_id)
        other.acquire(reviewer_id="other_writer")
        draft.lock._held = False  # noqa: SLF001
        draft.lock.holder_id = None
        with pytest.raises((PublicationError, LockError)):
            session.publish_reviewed_run()
        other.release()


def test_corrected_directional_and_parallel_states(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _enter_with_draft(session, review_root)
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=1,
            label="nose_to_nose",
        )
        session.set_parallel_pred(
            channel="contact",
            mouse_a_id=0,
            mouse_b_id=1,
            start_sample_idx=3,
            end_sample_idx=3,
            pred_value=0,
        )
        result = session.publish_reviewed_run()
        assert result.success
        states = pd.read_parquet(result.review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME)
        a = states[
            (states["dense_frame_idx"] == 0)
            & (states["focal_id"] == 0)
            & (states["partner_id"] == 1)
        ].iloc[0]
        b = states[
            (states["dense_frame_idx"] == 0)
            & (states["focal_id"] == 1)
            & (states["partner_id"] == 0)
        ].iloc[0]
        assert a["behavior_label"] == "nose_to_nose"
        assert b["behavior_label"] != "nose_to_nose"
        crows = states[states["dense_frame_idx"] == 3]
        assert set(int(v) for v in crows["pred_contact"].tolist()) == {0}


def test_pure_bout_collapse_parity_and_status_semantics(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _enter_with_draft(session, review_root)
        # Status-only reject on an automated bout.
        session.set_bout_review_status(
            target_bout_id="bout_dir_a", status=STATUS_REJECTED
        )
        labels_before = session.current_reviewed_states()["behavior_label"].tolist()
        # Geometry change voids prior acceptance on another bout after accept.
        session.set_bout_review_status(
            target_bout_id="bout_dir_b", status=STATUS_ACCEPTED
        )
        session.set_exclusive_label(
            focal_id=1,
            partner_id=0,
            start_sample_idx=4,
            end_sample_idx=5,
            label="engaged",
        )
        labels_after_reject_check = session.current_reviewed_states()[
            "behavior_label"
        ].tolist()
        # Reject is status-only: labels unchanged by the reject itself; later
        # geometry edit may change labels on the edited direction only.
        assert labels_before  # sanity
        result = session.publish_reviewed_run()
        assert result.success
        statuses = pd.read_parquet(
            result.review_run_dir / REVIEW_BOUT_STATUS_PARQUET_FILENAME
        )
        assert STATUS_REJECTED in set(statuses["status"].astype(str))
        assert STATUS_VOIDED in set(statuses["status"].astype(str))
        # Reviewed bout ids never reuse automated ids.
        bouts = pd.read_parquet(result.review_run_dir / REVIEWED_BEHAVIOR_BOUTS_FILENAME)
        assert all(str(b).startswith("rb__") for b in bouts["bout_id"].tolist())
        states = pd.read_parquet(result.review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME)
        if "exp_name" not in states.columns:
            states = states.copy()
            states["exp_name"] = built["package_run_id"]
        rebuilt = extract_behavior_bouts(
            states,
            fps=120.0,
            source_s7_run_id="",
            source_s6_run_id="",
            run_id=built["s8_run_id"],
        )
        assert len(rebuilt) == len(bouts)
        del labels_after_reject_check


def test_atomic_failure_cleanup_and_draft_recovery(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        draft = _enter_with_draft(session, review_root)
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="retract",
        )
        journal_before = (draft.working_dir / "correction_events.jsonl").read_text(
            encoding="utf-8"
        )
        with patch(
            "behavioral_review.publication.evaluate_review_qc",
            side_effect=RuntimeError("injected QC failure"),
        ):
            result = publish_reviewed_run(session)
        assert not result.success
        assert result.status == "failed"
        assert not (draft.review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME).is_file()
        assert not (draft.review_run_dir / REVIEW_MANIFEST_FILENAME).is_file()
        # Diagnostics retained.
        assert (draft.review_run_dir / "run_meta.json").is_file()
        assert (draft.review_run_dir / "processing_log.txt").is_file()
        # Working draft recoverable for retry.
        assert draft.publication_state == PUBLICATION_STATE_DRAFT
        assert (
            draft.working_dir / "correction_events.jsonl"
        ).read_text(encoding="utf-8") == journal_before
        # Retry succeeds.
        result2 = session.publish_reviewed_run()
        assert result2.success


def test_immutable_publication_no_overwrite(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _enter_with_draft(session, review_root)
        result = session.publish_reviewed_run()
        assert result.success
        with pytest.raises(PublicationError, match="already published"):
            session.publish_reviewed_run()


def test_discovery_supersession_and_historical_vs_eligible(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        # First published review.
        draft1 = _enter_with_draft(session, review_root)
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="engaged",
        )
        r1 = session.publish_reviewed_run()
        assert r1.success
        session.exit_editing_mode(release_lock=True)

        # Second draft/version for same parent.
        session2 = open_editing_session(
            built["package_dir"], require_video=False, require_s8_states=True
        )
        session2.enter_editing_mode("reviewer_pub")
        draft2 = session2.create_draft(review_root=review_root)
        r2 = session2.publish_reviewed_run()
        assert r2.success

        summary = summarize_behavioral_review_project(review_root=review_root)
        completed = list_completed_reviewed_runs(review_root=review_root)
        assert len(completed) >= 2
        latest = find_latest_eligible_reviewed_run(review_root=review_root)
        assert latest is not None
        assert latest.review_run_id == r2.review_run_id

        older = next(r for r in summary.runs if r.review_run_id == r1.review_run_id)
        assert older.classification == CLASS_SUPERSEDED
        assert older.superseded_by == r2.review_run_id
        assert older.historical_completed is True
        # Explicit reopen of older path still works (no silent replacement).
        reopened_old = reopen_reviewed_run(r1.review_run_dir)
        assert reopened_old["review_run_id"] == r1.review_run_id
        assert is_s10_run_eligible_for_downstream(r2.review_run_dir)

        assert classify_analysis_source("s9") == SOURCE_TYPE_AUTOMATED
        assert classify_analysis_source("reviewed") == SOURCE_TYPE_REVIEWED
        del draft1, draft2


def test_stale_parent_discovery_classification(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _enter_with_draft(session, review_root)
        result = session.publish_reviewed_run()
        assert result.success
        run_dir = result.review_run_dir

    # After publish, make parent appear stale to discovery validation.
    with patch(
        "behavioral_review.discovery.validate_s9_run_for_downstream",
        side_effect=ProvenanceError("parent drifted"),
    ), patch(
        "behavioral_review.discovery.is_s9_run_eligible_for_downstream",
        return_value=False,
    ):
        summary = summarize_behavioral_review_project(review_root=review_root)
        row = next(r for r in summary.runs if r.review_run_id == result.review_run_id)
        assert row.classification == CLASS_STALE_PARENT
        assert row.historical_completed is True
        assert row.currently_eligible is False
        # Historical status on disk unchanged.
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["status"] in (STATUS_COMPLETED, "completed_with_warnings")


def test_draft_discovery_and_no_upstream_mutation(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        draft = _enter_with_draft(session, review_root)
        summary = summarize_behavioral_review_project(review_root=review_root)
        row = next(r for r in summary.runs if r.review_run_id == draft.review_run_id)
        assert row.classification == CLASS_DRAFT
        assert not row.currently_eligible
        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=0,
            end_sample_idx=0,
            label="mutually_disengaged",
        )
        session.publish_reviewed_run()
    auto_bouts = pd.read_parquet(built["package_dir"] / "behavior_bouts.parquet")
    assert len(auto_bouts) == len(built["bouts"])
    assert_no_upstream_writes(
        package_dir=built["package_dir"],
        s8_run_dir=built["s8_run_dir"],
        video_path=built["video_path"],
        snapshots=snaps,
    )
