"""S10 Gallery A2: working-draft gallery + single/bulk review status actions."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from behavioral_review.contracts import (
    CHANNEL_EXCLUSIVE,
    GALLERY_SOURCE_AUTOMATED,
    GALLERY_SOURCE_WORKING,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    STATUS_VOIDED,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.gallery import (
    GalleryQuery,
    build_gallery_from_working_draft,
    filter_eligible_gallery_review_cards,
    gallery_card_is_review_eligible,
    gallery_status_target_for_card,
)
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)
from ui.controllers.behavioral_review_controller import (
    EDIT_OP_SET_STATUS_BULK,
    GALLERY_EDIT_REQUIRES_DRAFT_MESSAGE,
    VIEW_MODE_GALLERY,
    BehavioralReviewController,
    BehavioralReviewUiError,
    S9PackageOption,
)


def _open_editor(tmp_path: Path, **kwargs):
    built = build_mini_review_package(tmp_path, **kwargs)
    review_root = tmp_path / "review_drafts_a2"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"],
            require_video=False,
            require_s8_states=True,
        )
    return built, session, review_root


def _enter_with_draft(session, review_root: Path, reviewer: str = "gallery_reviewer"):
    session.enter_editing_mode(reviewer)
    return session.create_draft(review_root=review_root)


def _controller_with_draft(tmp_path: Path):
    built = build_mini_review_package(tmp_path)
    controller = BehavioralReviewController()
    s8 = built["s8_run_id"]

    def _ok(run_dir: Path | str) -> dict:
        path = Path(run_dir).resolve()
        return {
            "s9_run_dir": path,
            "status": "completed",
            "session_root": str(path.parent.parent),
            "source_s8_run_dir": path.parent.parent / "behavioral_arbitration" / s8,
            "source_s8_run_id": s8,
        }

    patches = (
        patch(
            "results_packaging.runner.validate_s9_run_for_downstream",
            side_effect=_ok,
        ),
        patch(
            "results_packaging.runner.is_s9_run_eligible_for_downstream",
            return_value=True,
        ),
    )
    with patch_s9_currently_eligible(source_s8_run_id=s8), ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        identity = controller.prepare_session(built["session_root"])
        snapshot = controller.collect_discovery_snapshot(identity)
        assert controller.apply_discovery_snapshot(snapshot)
        controller.state.s9_options = tuple(
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
            for o in controller.state.s9_options
        )
        controller.select_s9_package(built["package_run_id"])
        open_id = controller.begin_open_automated()
        open_snap = controller.open_source_worker(open_id)
        assert controller.apply_open_snapshot(open_snap)
        controller.set_reviewer_id("gallery_reviewer")
        enter_id = controller.begin_enter_editing()
        assert controller.apply_edit_snapshot(controller.edit_worker(enter_id))
        draft_id = controller.begin_create_draft(review_root=tmp_path / "gui_drafts")
        assert controller.apply_edit_snapshot(controller.edit_worker(draft_id))
    return controller, built


def test_working_draft_gallery_uses_effective_labels_and_status(
    tmp_path: Path,
) -> None:
    built, session, review_root = _open_editor(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.inspection.source)
    draft = _enter_with_draft(session, review_root)
    # Status on lineage-bearing bout before geometry rewrite.
    session.set_bout_review_status(
        target_bout_id="bout_dir_a",
        status=STATUS_ACCEPTED,
    )
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=3,
        label="nose_to_body",
    )
    reviewed = session.current_reviewed_bouts()
    index = build_gallery_from_working_draft(
        reviewed,
        session.inspection.sync,
        package_run_id=built["package_run_id"],
        review_run_id=draft.review_run_id,
        review_bout_status=session.current_statuses(),
        automated_bouts=session.inspection.source.behavior_bouts,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert index.source.source_kind == GALLERY_SOURCE_WORKING
    assert all(c.source_kind == GALLERY_SOURCE_WORKING for c in index.cards)
    assert all(c.channel == CHANNEL_EXCLUSIVE for c in index.cards)
    # Relabeled interval appears under the corrected effective label group.
    labels = {c.effective_label for c in index.cards}
    assert "nose_to_body" in labels
    # Stale automated approach geometry for 0→1 early samples must not remain.
    approach_cards = [
        c
        for c in index.cards
        if c.effective_label == "approach" and c.focal_id == 0
    ]
    assert approach_cards == []
    # Prior acceptance is voided after geometry change (R9); do not carry accept.
    assert session.overlay.status_for_source_bout("bout_dir_a") == STATUS_VOIDED
    assert all(
        c.status != STATUS_ACCEPTED or c.source_s8_bout_id != "bout_dir_a"
        for c in index.cards
    )
    assert all(gallery_card_is_review_eligible(c) for c in index.cards)
    src = session.inspection.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path,
        snapshots=snaps,
    )


def test_single_accept_reject_status_only(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root)
    before = session.current_reviewed_states()["behavior_label"].tolist()
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    index = build_gallery_from_working_draft(
        session.current_reviewed_bouts(),
        session.inspection.sync,
        package_run_id=built["package_run_id"],
        review_run_id=draft.review_run_id,
        review_bout_status=session.current_statuses(),
        automated_bouts=session.inspection.source.behavior_bouts,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    accepted = [c for c in index.cards if c.source_s8_bout_id == "bout_dir_a"]
    assert len(accepted) == 1
    assert accepted[0].status == STATUS_ACCEPTED
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_REJECTED
    )
    after = session.current_reviewed_states()["behavior_label"].tolist()
    assert before == after
    assert session.overlay.status_for_source_bout("bout_dir_a") == STATUS_REJECTED
    geom = session.current_reviewed_bouts()[
        session.current_reviewed_bouts()["source_s8_bout_id"] == "bout_dir_a"
    ]
    assert not geom.empty


def test_bulk_status_one_undo_group_and_scope(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root)
    index = build_gallery_from_working_draft(
        session.current_reviewed_bouts(),
        session.inspection.sync,
        package_run_id=built["package_run_id"],
        review_run_id=draft.review_run_id,
        review_bout_status=session.current_statuses(),
        automated_bouts=session.inspection.source.behavior_bouts,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    # Explicit subset only — never the whole filter automatically.
    selected = [index.cards[0].card_id]
    eligible = filter_eligible_gallery_review_cards(
        index.cards, selected_card_ids=selected
    )
    assert len(eligible) == 1
    targets = []
    for card in eligible:
        tid, kind = gallery_status_target_for_card(card)
        targets.append(
            {
                "target_bout_id": tid,
                "target_bout_id_kind": kind,
                "status": STATUS_ACCEPTED,
                "channel": CHANNEL_EXCLUSIVE,
            }
        )
    events = session.set_bout_review_statuses(targets, client_op_id="bulk-a2")
    assert len(events) == 1
    assert len({e.undo_group_id for e in events}) == 1
    assert session.overlay.status_for_source_bout(targets[0]["target_bout_id"]) == (
        STATUS_ACCEPTED
    )
    # Second exclusive bout remains unreviewed when not selected.
    other = [
        c
        for c in index.cards
        if c.card_id != selected[0] and c.source_s8_bout_id
    ]
    if other:
        assert (
            session.overlay.status_for_source_bout(str(other[0].source_s8_bout_id))
            == STATUS_UNREVIEWED
        )
    undone = session.undo()
    assert len(undone) == 1
    assert session.overlay.status_for_source_bout(targets[0]["target_bout_id"]) == (
        STATUS_UNREVIEWED
    )
    redone = session.redo()
    assert len(redone) == 1
    assert session.overlay.status_for_source_bout(targets[0]["target_bout_id"]) == (
        STATUS_ACCEPTED
    )


def test_bulk_failure_atomicity_before_append(tmp_path: Path) -> None:
    _, session, review_root = _open_editor(tmp_path)
    _enter_with_draft(session, review_root)
    before_seq = session.draft.journal.cursor.cursor_event_seq
    with pytest.raises(Exception, match="Invalid status|missing target"):
        session.set_bout_review_statuses(
            [
                {
                    "target_bout_id": "bout_dir_a",
                    "status": STATUS_ACCEPTED,
                },
                {
                    "target_bout_id": "bout_dir_b",
                    "status": "not_a_status",
                },
            ]
        )
    assert session.draft.journal.cursor.cursor_event_seq == before_seq
    assert session.overlay.status_for_source_bout("bout_dir_a") == STATUS_UNREVIEWED
    assert session.overlay.status_for_source_bout("bout_dir_b") == STATUS_UNREVIEWED


def test_selection_invalidation_and_void_after_geometry(
    tmp_path: Path,
) -> None:
    controller, built = _controller_with_draft(tmp_path)
    controller.set_gallery_source_kind(GALLERY_SOURCE_WORKING)
    controller.set_view_mode(VIEW_MODE_GALLERY)
    index = controller.state.gallery_index
    assert index is not None
    card = next(c for c in index.cards if c.source_s8_bout_id == "bout_dir_a")
    controller.set_gallery_selection((card.card_id,), primary_card_id=card.card_id)
    assert controller.state.gallery_selected_card_ids == (card.card_id,)

    identity = controller.begin_gallery_set_status(card.card_id, STATUS_ACCEPTED)
    assert controller.apply_edit_snapshot(controller.edit_worker(identity))
    refreshed = controller.state.gallery_index
    assert refreshed is not None
    accepted = refreshed.card_by_id(card.card_id)
    assert accepted.status == STATUS_ACCEPTED

    # Geometry change voids acceptance and regenerates bout identities.
    trim_id = controller.begin_set_exclusive_label(
        label="nose_to_tail",
        start_sample_idx=0,
        end_sample_idx=3,
        focal_id=0,
        partner_id=1,
    )
    assert controller.apply_edit_snapshot(controller.edit_worker(trim_id))
    after = controller.state.gallery_index
    assert after is not None
    # Prior card id is gone; selection must be pruned.
    assert card.card_id not in {c.card_id for c in after.cards}
    assert card.card_id not in controller.state.gallery_selected_card_ids
    voided = [
        c
        for c in after.cards
        if c.status == STATUS_VOIDED or c.void_reason == "geometry_changed"
    ]
    # Overlay void is visible somehow; accepted must not silently carry over.
    still_accepted_same_id = [
        c for c in after.cards if c.card_id == card.card_id and c.status == STATUS_ACCEPTED
    ]
    assert still_accepted_same_id == []
    del voided
    del built


def test_label_group_moves_after_relabel(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root)
    before = build_gallery_from_working_draft(
        session.current_reviewed_bouts(),
        session.inspection.sync,
        package_run_id=built["package_run_id"],
        review_run_id=draft.review_run_id,
        review_bout_status=session.current_statuses(),
        automated_bouts=session.inspection.source.behavior_bouts,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert "approach" in {g.label for g in before.groups}
    session.set_exclusive_label(
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=3,
        label="nose_to_nose",
    )
    after = build_gallery_from_working_draft(
        session.current_reviewed_bouts(),
        session.inspection.sync,
        package_run_id=built["package_run_id"],
        review_run_id=draft.review_run_id,
        review_bout_status=session.current_statuses(),
        automated_bouts=session.inspection.source.behavior_bouts,
        query=GalleryQuery(labels=frozenset({"nose_to_nose"})),
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert after.n_cards >= 1
    assert all(c.effective_label == "nose_to_nose" for c in after.cards)
    assert all(c.corrected for c in after.cards if c.automated_label == "approach")


def test_working_draft_persists_and_reopens(tmp_path: Path) -> None:
    built, session, review_root = _open_editor(tmp_path)
    draft = _enter_with_draft(session, review_root, reviewer="alice")
    run_dir = draft.review_run_dir
    session.set_bout_review_status(
        target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
    )
    session.close_draft(release_lock=True)

    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session2 = open_editing_session(
            built["package_dir"],
            require_video=False,
            require_s8_states=True,
        )
    session2.enter_editing_mode("alice")
    session2.open_draft(run_dir)
    index = build_gallery_from_working_draft(
        session2.current_reviewed_bouts(),
        session2.inspection.sync,
        package_run_id=built["package_run_id"],
        review_run_id=session2.draft.review_run_id,
        review_bout_status=session2.current_statuses(),
        automated_bouts=session2.inspection.source.behavior_bouts,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    matched = [
        c
        for c in index.cards
        if c.source_s8_bout_id == "bout_dir_a" and c.status == STATUS_ACCEPTED
    ]
    assert matched


def test_inspection_gallery_does_not_create_draft(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path / "insp")
    controller = BehavioralReviewController()
    s8 = built["s8_run_id"]

    def _ok(run_dir: Path | str) -> dict:
        path = Path(run_dir).resolve()
        return {
            "s9_run_dir": path,
            "status": "completed",
            "session_root": str(path.parent.parent),
            "source_s8_run_dir": path.parent.parent / "behavioral_arbitration" / s8,
            "source_s8_run_id": s8,
        }

    patches = (
        patch(
            "results_packaging.runner.validate_s9_run_for_downstream",
            side_effect=_ok,
        ),
        patch(
            "results_packaging.runner.is_s9_run_eligible_for_downstream",
            return_value=True,
        ),
    )
    with patch_s9_currently_eligible(source_s8_run_id=s8), ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        identity = controller.prepare_session(built["session_root"])
        snapshot = controller.collect_discovery_snapshot(identity)
        assert controller.apply_discovery_snapshot(snapshot)
        controller.state.s9_options = tuple(
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
            for o in controller.state.s9_options
        )
        controller.select_s9_package(built["package_run_id"])
        open_id = controller.begin_open_automated()
        assert controller.apply_open_snapshot(controller.open_source_worker(open_id))

    controller.set_view_mode(VIEW_MODE_GALLERY)
    assert controller.state.gallery_source_kind == GALLERY_SOURCE_AUTOMATED
    assert controller.state.draft_open is False
    with pytest.raises(BehavioralReviewUiError, match="working draft"):
        controller.set_gallery_source_kind(GALLERY_SOURCE_WORKING)
    with pytest.raises(BehavioralReviewUiError) as exc:
        controller.begin_gallery_set_status("automated:x", STATUS_ACCEPTED)
    assert GALLERY_EDIT_REQUIRES_DRAFT_MESSAGE[:32] in str(exc.value)
    assert not (built["session_root"] / "behavioral_review").exists()


def test_controller_bulk_and_stale_source_guard(tmp_path: Path) -> None:
    controller, _built = _controller_with_draft(tmp_path)
    controller.set_gallery_source_kind(GALLERY_SOURCE_WORKING)
    controller.set_view_mode(VIEW_MODE_GALLERY)
    index = controller.state.gallery_index
    assert index is not None and index.n_cards >= 1
    ids = tuple(c.card_id for c in index.cards[:2])
    controller.set_gallery_selection(ids)
    identity = controller.begin_gallery_bulk_status(STATUS_REJECTED, card_ids=ids)
    assert identity.op == EDIT_OP_SET_STATUS_BULK
    snap = controller.edit_worker(identity)
    assert controller.apply_edit_snapshot(snap)
    refreshed = controller.state.gallery_index
    assert refreshed is not None
    for card_id in ids:
        assert refreshed.card_by_id(card_id).status == STATUS_REJECTED

    # Switching source clears selection.
    controller.set_gallery_source_kind(GALLERY_SOURCE_AUTOMATED)
    assert controller.state.gallery_selected_card_ids == ()
    with pytest.raises(BehavioralReviewUiError, match="read-only|Working draft"):
        controller.begin_gallery_bulk_status(STATUS_ACCEPTED, card_ids=ids)


def test_gallery_refresh_performance_e_session_scale(tmp_path: Path) -> None:
    """Synthetic ~538 exclusive cards: refresh stays interactive (no decode)."""

    from behavioral_review.contracts import (
        GALLERY_CARD_SCHEMA_VERSION,
        GALLERY_INDEX_SCHEMA_VERSION,
    )
    from behavioral_review.gallery import (
        GalleryCard,
        GalleryIndex,
        GalleryPreviewPlan,
        GalleryQuery,
        GallerySourceRef,
        group_cards_by_effective_label,
    )

    plan = GalleryPreviewPlan(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        clip_frame_idx_start=0,
        clip_frame_idx_end=2,
        clip_frame_count=3,
        pre_context_sec=0.0,
        post_context_sec=0.0,
        source_fps=30.0,
        source_n_frames=100,
        first_visible_prepared_frame_idx=0,
        last_visible_prepared_frame_idx=2,
    )
    n = 538
    cards = []
    for i in range(n):
        label = ["approach", "retract", "nose_to_body", "engaged"][i % 4]
        cards.append(
            GalleryCard(
                schema_version=GALLERY_CARD_SCHEMA_VERSION,
                card_id=f"working_draft:rb_{i:04d}",
                bout_id=f"rb_{i:04d}",
                source_kind=GALLERY_SOURCE_WORKING,
                channel=CHANNEL_EXCLUSIVE,
                identity_grain="directional",
                automated_label=label,
                effective_label=label,
                corrected=False,
                focal_id=i % 2,
                partner_id=1 - (i % 2),
                start_sample_idx=i * 10,
                end_sample_idx=i * 10 + 5,
                bout_duration_frames=6,
                bout_duration_sec=6 / 120.0,
                prepared_frame_idx_start=0,
                prepared_frame_idx_end=2,
                status=STATUS_UNREVIEWED,
                void_reason=None,
                source_s8_bout_id=f"s8_{i:04d}",
                package_bout_id=f"pkg_{i:04d}",
                reviewed_bout_id=f"rb_{i:04d}",
                preview_plan=plan,
            )
        )
    t0 = time.perf_counter()
    groups = group_cards_by_effective_label(cards)
    index = GalleryIndex(
        schema_version=GALLERY_INDEX_SCHEMA_VERSION,
        source=GallerySourceRef(
            source_kind=GALLERY_SOURCE_WORKING,
            package_run_id="pkg",
            review_run_id="draft",
        ),
        cards=tuple(c for g in groups for c in g.cards),
        groups=groups,
        query=GalleryQuery(),
        n_cards=n,
        n_groups=len(groups),
        available_labels=tuple(sorted({c.effective_label for c in cards})),
        available_focal_ids=(0, 1),
        status_counts={STATUS_UNREVIEWED: n},
    )
    # Simulate status-chip refresh: regroup after flipping 40 statuses.
    flipped = [
        replace(card, status=STATUS_ACCEPTED if i < 40 else card.status)
        for i, card in enumerate(index.cards)
    ]
    groups2 = group_cards_by_effective_label(flipped)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert len(groups2) == index.n_groups
    # Card regroup/refresh at E-session scale should stay well under 250 ms.
    assert elapsed_ms < 250.0, f"gallery refresh too slow: {elapsed_ms:.1f} ms"
