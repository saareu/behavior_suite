"""S10 Gallery A1: read-only behavior-grouped bout browser tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from behavioral_review.contracts import (
    GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS,
    GALLERY_SOURCE_AUTOMATED,
    GALLERY_SOURCE_PUBLISHED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
)
from behavioral_review.gallery import (
    GalleryQuery,
    apply_gallery_query,
    build_gallery_from_automated_bouts,
    build_gallery_from_published_review,
    build_preview_plan,
    deep_link_target_for_card,
    group_cards_by_effective_label,
)
from behavioral_review.gallery_preview import (
    GalleryPreviewScheduler,
    measure_preview_concurrency_capacity,
)
from behavioral_review.session import open_inspection_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
    write_tiny_avi,
)


def _open_package(tmp_path: Path, **kwargs):
    built = build_mini_review_package(tmp_path, **kwargs)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(
            built["package_dir"],
            require_video=True,
            require_s8_states=True,
            decode_initial_frame=False,
        )
    return built, session


def test_label_grouping_and_deterministic_order(tmp_path: Path) -> None:
    built, session = _open_package(tmp_path)
    index = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert index.source.source_kind == GALLERY_SOURCE_AUTOMATED
    assert index.n_cards == 2  # exclusive only
    labels = [g.label for g in index.groups]
    assert labels == sorted(labels)
    assert set(labels) == {"approach", "retract"}
    # Within a group, time then bout_id.
    for group in index.groups:
        starts = [c.start_sample_idx for c in group.cards]
        assert starts == sorted(starts)
    # Stable card identities.
    ids = [c.card_id for c in index.cards]
    assert ids == sorted(ids, key=lambda x: (x.split(":")[0], x))
    assert all(c.card_id.startswith("automated:") for c in index.cards)
    assert all(c.package_bout_id == c.bout_id for c in index.cards)


def test_mouse_and_status_filters(tmp_path: Path) -> None:
    built, session = _open_package(tmp_path)
    base = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        pre_context_sec=0.0,
        post_context_sec=0.0,
        status_by_bout_id={
            f"{built['package_run_id']}__bout_dir_a": STATUS_ACCEPTED,
        },
    )
    focal0 = apply_gallery_query(
        base.cards, GalleryQuery(focal_ids=frozenset({0}))
    )
    assert len(focal0) == 1
    assert focal0[0].focal_id == 0
    accepted = apply_gallery_query(
        base.cards, GalleryQuery(statuses=frozenset({STATUS_ACCEPTED}))
    )
    assert len(accepted) == 1
    assert accepted[0].status == STATUS_ACCEPTED
    approach = apply_gallery_query(
        base.cards, GalleryQuery(labels=frozenset({"approach"}))
    )
    assert {c.effective_label for c in approach} == {"approach"}
    grouped = group_cards_by_effective_label(approach)
    assert len(grouped) == 1


def test_automated_versus_published_sources_do_not_merge(tmp_path: Path) -> None:
    built, session = _open_package(tmp_path)
    auto = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    reviewed = session.source.behavior_bouts.copy()
    exclusive = reviewed[reviewed["channel"] == "exclusive"].copy()
    exclusive["bout_id"] = [
        f"rb__review__{i}" for i in range(len(exclusive))
    ]
    exclusive["source_package_bout_id"] = (
        session.source.behavior_bouts[session.source.behavior_bouts["channel"] == "exclusive"][
            "bout_id"
        ]
        .astype(str)
        .tolist()
    )
    exclusive.loc[exclusive.index[0], "behavior_name"] = "nose_to_body"
    status = pd.DataFrame(
        [
            {
                "reviewed_bout_id": exclusive.iloc[0]["bout_id"],
                "status": STATUS_REJECTED,
                "active": True,
                "void_reason": None,
            }
        ]
    )
    pub = build_gallery_from_published_review(
        exclusive,
        session.sync,
        package_run_id=built["package_run_id"],
        review_run_id="behavioral_review__synth",
        review_bout_status=status,
        automated_bouts=session.source.behavior_bouts,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert pub.source.source_kind == GALLERY_SOURCE_PUBLISHED
    assert all(c.source_kind == GALLERY_SOURCE_PUBLISHED for c in pub.cards)
    assert all(c.reviewed_bout_id is not None for c in pub.cards)
    # Identities never silently reused from automated bout_id.
    assert {c.bout_id for c in pub.cards}.isdisjoint({c.bout_id for c in auto.cards})
    corrected = [c for c in pub.cards if c.corrected]
    assert len(corrected) == 1
    assert corrected[0].effective_label == "nose_to_body"
    assert corrected[0].automated_label == "approach"
    rejected = [c for c in pub.cards if c.status == STATUS_REJECTED]
    assert len(rejected) == 1
    # Automated gallery remains unreviewed / uncorrected (no silent transfer).
    assert all(c.status == STATUS_UNREVIEWED for c in auto.cards)
    assert all(not c.corrected for c in auto.cards)


def test_preview_plan_first_last_and_dense_prepared_mapping(tmp_path: Path) -> None:
    built, session = _open_package(tmp_path, plateau_repeats=2, n_prepared=5)
    index = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    card = next(c for c in index.cards if c.effective_label == "approach")
    # Dense 0..3 → prepared 0,0,1,1 with plateau_repeats=2.
    assert card.prepared_frame_idx_start == int(
        built["sync"].loc[card.start_sample_idx, "prepared_frame_idx"]
    )
    assert card.prepared_frame_idx_end == int(
        built["sync"].loc[card.end_sample_idx, "prepared_frame_idx"]
    )
    plan = card.preview_plan
    assert plan.first_visible_prepared_frame_idx == plan.clip_frame_idx_start
    assert plan.last_visible_prepared_frame_idx == plan.clip_frame_idx_end
    assert plan.first_visible_prepared_frame_idx == card.prepared_frame_idx_start
    assert plan.last_visible_prepared_frame_idx == card.prepared_frame_idx_end
    # Pad clamp at edges.
    padded = build_preview_plan(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=1,
        source_fps=10.0,
        source_n_frames=5,
        pre_context_sec=1.0,
        post_context_sec=1.0,
    )
    assert padded.first_visible_prepared_frame_idx == 0
    assert padded.last_visible_prepared_frame_idx <= 4


def test_preview_queue_bounds_and_visibility(tmp_path: Path) -> None:
    video = write_tiny_avi(tmp_path / "preview.avi", frame_count=8, fps=10.0)
    frames_read: list[int] = []

    def reader(path: Path, idx: int) -> np.ndarray:
        del path
        frames_read.append(int(idx))
        time.sleep(0.01)
        return np.full((24, 32, 3), int(idx) % 256, dtype=np.uint8)

    built, session = _open_package(tmp_path / "pkg")
    index = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        source_fps=10.0,
        source_n_frames=8,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    # Inflate card list artificially for queue pressure.
    cards = list(index.cards) * 10
    scheduler = GalleryPreviewScheduler(
        video,
        max_concurrent=2,
        max_queued=3,
        cache_size=4,
        frame_reader=reader,
        source_token="tok-a",
    )
    try:
        scheduler.register_cards(cards)
        # Only two visible.
        visible = [c.card_id for c in cards[:2]]
        scheduler.set_visible_card_ids(visible)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            scheduler.drain_completed()
            stats = scheduler.stats()
            assert stats["n_in_flight"] <= 2
            assert stats["n_pending"] <= 3
            if stats["completed_count"] >= 2:
                break
            time.sleep(0.02)
        stats = scheduler.stats()
        assert stats["n_in_flight"] <= 2
        assert stats["max_concurrent"] == 2
        assert stats["max_queued"] == 3
        # Hide all → release playback.
        scheduler.set_visible_card_ids(())
        assert scheduler.stats()["n_visible"] == 0
        assert scheduler.stats()["n_pending"] == 0
    finally:
        scheduler.close()


def test_source_switch_rejects_stale_callbacks(tmp_path: Path) -> None:
    video = write_tiny_avi(tmp_path / "stale.avi", frame_count=6, fps=10.0)
    gate = {"hold": True}

    def reader(path: Path, idx: int) -> np.ndarray:
        del path
        while gate["hold"]:
            time.sleep(0.01)
        return np.zeros((24, 32, 3), dtype=np.uint8)

    built, session = _open_package(tmp_path / "pkg2")
    index = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        source_fps=10.0,
        source_n_frames=6,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    delivered: list[str] = []
    scheduler = GalleryPreviewScheduler(
        video,
        max_concurrent=2,
        max_queued=8,
        frame_reader=reader,
        source_token="gen1",
    )
    try:
        scheduler.set_callback(lambda frame: delivered.append(frame.request.source_token))
        scheduler.register_cards(index.cards)
        scheduler.set_visible_card_ids([index.cards[0].card_id])
        time.sleep(0.05)
        new_gen = scheduler.invalidate(new_source_token="gen2")
        assert new_gen == 2
        gate["hold"] = False
        time.sleep(0.1)
        frames = scheduler.drain_completed()
        # Stale generation/source must not be accepted.
        assert all(f.request.source_token == "gen2" for f in frames)
        assert all(tok == "gen2" for tok in delivered)
        assert scheduler.stats()["rejected_stale_count"] >= 0
    finally:
        gate["hold"] = False
        scheduler.close()


def test_plateau_skips_redecode(tmp_path: Path) -> None:
    video = write_tiny_avi(tmp_path / "plateau.avi", frame_count=4, fps=10.0)
    reads: list[int] = []

    def reader(path: Path, idx: int) -> np.ndarray:
        del path
        reads.append(int(idx))
        return np.full((24, 32, 3), 7, dtype=np.uint8)

    plan = build_preview_plan(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=1,
        source_fps=10.0,
        source_n_frames=4,
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    # Single-frame preview plan → looping same prepared index.
    from behavioral_review.contracts import GALLERY_CARD_SCHEMA_VERSION
    from behavioral_review.gallery import GalleryCard

    card = GalleryCard(
        schema_version=GALLERY_CARD_SCHEMA_VERSION,
        card_id="automated:plateau",
        bout_id="plateau",
        source_kind=GALLERY_SOURCE_AUTOMATED,
        channel="exclusive",
        identity_grain="directional",
        automated_label="approach",
        effective_label="approach",
        corrected=False,
        focal_id=0,
        partner_id=1,
        start_sample_idx=2,
        end_sample_idx=3,
        bout_duration_frames=2,
        bout_duration_sec=2 / 120.0,
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=1,
        status=STATUS_UNREVIEWED,
        void_reason=None,
        source_s8_bout_id="x",
        package_bout_id="plateau",
        reviewed_bout_id=None,
        preview_plan=plan,
    )
    scheduler = GalleryPreviewScheduler(
        video, max_concurrent=1, frame_reader=reader, source_token="p"
    )
    try:
        scheduler.register_cards([card])
        scheduler.set_visible_card_ids([card.card_id])
        for _ in range(5):
            scheduler.tick()
            time.sleep(0.03)
            scheduler.drain_completed()
        # First decode only; subsequent same prepared index are plateau skips.
        assert reads.count(1) == 1
    finally:
        scheduler.close()


def _assert_no_writes(session, snaps) -> None:
    src = session.source
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path if src.source_video else None,
        snapshots=snaps,
    )


def test_timeline_deep_link_target_preserves_direction(tmp_path: Path) -> None:
    built, session = _open_package(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.source)
    index = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    card = next(c for c in index.cards if c.focal_id == 1)
    target = deep_link_target_for_card(card)
    assert target["focal_id"] == 1
    assert target["partner_id"] == 0
    assert target["seek_bout_id"] == card.package_bout_id
    assert target["channel"] == "exclusive"
    mapped = session.seek_bout_start(str(target["seek_bout_id"]), decode=False)
    assert mapped.start.sample_idx == card.start_sample_idx
    assert session.selected_direction.focal_id == 1
    assert session.selected_direction.partner_id == 0
    _assert_no_writes(session, snaps)


def test_gallery_inspection_never_writes_journal_or_science(tmp_path: Path) -> None:
    built, session = _open_package(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.source)
    index = build_gallery_from_automated_bouts(
        session.source.behavior_bouts,
        session.sync,
        package_run_id=built["package_run_id"],
        pre_context_sec=0.0,
        post_context_sec=0.0,
    )
    assert index.n_cards >= 1
    review_root = built["session_root"] / "behavioral_review"
    assert not review_root.exists()
    _assert_no_writes(session, snaps)


def test_practical_concurrency_limit_from_measurement(tmp_path: Path) -> None:
    video = write_tiny_avi(tmp_path / "bench.avi", frame_count=12, fps=10.0)

    def reader(path: Path, idx: int) -> np.ndarray:
        del path
        time.sleep(0.002)
        return np.full((24, 32, 3), int(idx) % 256, dtype=np.uint8)

    result = measure_preview_concurrency_capacity(
        video_path=video,
        prepared_indices=list(range(12)),
        candidates=(1, 2, 4, 8),
        repeats=2,
        frame_reader=reader,
    )
    assert result["recommended_workers"] in {1, 2, 4, 8}
    assert result["recommended_workers"] <= 8
    # Default A1 constant stays in the measured practical band.
    assert GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS in {2, 4, 6, 8}
    assert result["recommended_workers"] <= GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS or (
        result["best_workers"] >= GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS
    )


def test_controller_gallery_deep_link_and_no_draft(tmp_path: Path) -> None:
    from contextlib import ExitStack
    from unittest.mock import patch

    from ui.controllers.behavioral_review_controller import (
        VIEW_MODE_GALLERY,
        VIEW_MODE_TIMELINE,
        BehavioralReviewController,
        S9PackageOption,
    )

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

    src = controller.state.editing_session.inspection.source  # type: ignore[union-attr]
    snaps = snapshot_upstream_fingerprints(src)
    controller.set_view_mode(VIEW_MODE_GALLERY)
    index = controller.state.gallery_index
    assert index is not None and index.n_cards >= 1
    card = index.cards[0]
    decode_id = controller.select_gallery_card(card.card_id, switch_to_timeline=True)
    assert controller.state.view_mode == VIEW_MODE_TIMELINE
    assert controller.state.sample_idx == card.start_sample_idx
    assert controller.state.selected_focal_id == card.focal_id
    assert controller.state.selected_partner_id == card.partner_id
    assert controller.state.gallery_highlight_start == card.start_sample_idx
    assert decode_id is not None
    assert controller.state.draft_open is False
    assert controller.state.editing_active is False
    assert_no_upstream_writes(
        package_dir=src.package_dir,
        s8_run_dir=src.source_s8.run_dir,
        video_path=src.source_video.path if src.source_video else None,
        snapshots=snaps,
    )
    review_root = built["session_root"] / "behavioral_review"
    assert not review_root.exists()
