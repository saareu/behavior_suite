"""S10 responsiveness regressions: projection regen, draft resume, first decode."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from behavioral_review.apply import replay_active_correction_chain
from behavioral_review.contracts import OP_SET_EXCLUSIVE_LABEL
from behavioral_review.editing import open_editing_session
from behavioral_review.events import build_event
from behavioral_review.regenerate import regenerate_reviewed_bouts
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)
from ui.controllers.behavioral_review_controller import BehavioralReviewController


def _open_editor(tmp_path: Path):
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"], require_video=False, require_s8_states=True
        )
    return built, session


def test_inspection_open_defers_baseline_states_load(tmp_path: Path) -> None:
    _, session = _open_editor(tmp_path)
    assert session.baseline_states is None
    session.enter_editing_mode("resp_auditor")
    assert session.baseline_states is not None


def test_replay_skips_redundant_regenerations(tmp_path: Path) -> None:
    built, session = _open_editor(tmp_path)
    session.enter_editing_mode("resp_auditor")
    session.create_draft(review_root=tmp_path / "resp_drafts")
    assert session.baseline_states is not None
    source = session.inspection.source
    event = build_event(
        op=OP_SET_EXCLUSIVE_LABEL,
        reviewer_id="resp_auditor",
        event_seq=1,
        channel="exclusive",
        identity_grain="directional",
        focal_id=0,
        partner_id=1,
        start_sample_idx=0,
        end_sample_idx=1,
        label="nose_to_body",
    )

    calls = {"n": 0}
    real = regenerate_reviewed_bouts

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    with patch(
        "behavioral_review.apply.regenerate_reviewed_bouts", side_effect=_counting
    ):
        working, reviewed, changed, n = replay_active_correction_chain(
            session.baseline_states,
            [event],
            review_run_id="review_x",
            parent_package_run_id=source.package_run_id,
            source_s8_run_id=source.source_s8.run_id,
            source_s7_run_id=source.source_s7_run_id,
            source_s6_run_id=source.source_s6_run_id,
            automated_bouts=source.behavior_bouts,
            sync_index=source.sync_index,
        )
    assert n == 1
    assert not working.empty
    assert not reviewed.empty
    assert "exclusive" in changed
    # One exclusive label must not trigger the old initial+post+final triple regen.
    assert calls["n"] == 1

    calls["n"] = 0
    with patch(
        "behavioral_review.apply.regenerate_reviewed_bouts", side_effect=_counting
    ):
        replay_active_correction_chain(
            session.baseline_states,
            [],
            review_run_id="review_x",
            parent_package_run_id=source.package_run_id,
            source_s8_run_id=source.source_s8.run_id,
            source_s7_run_id=source.source_s7_run_id,
            source_s6_run_id=source.source_s6_run_id,
            automated_bouts=source.behavior_bouts,
            sync_index=source.sync_index,
        )
    assert calls["n"] == 1
    del built


def test_decode_worker_never_plateaus_without_pixmap(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path, plateau_repeats=2)
    controller = BehavioralReviewController()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        identity = controller.prepare_session(built["session_root"])
        assert controller.apply_discovery_snapshot(
            controller.collect_discovery_snapshot(identity)
        )
        from ui.controllers.behavioral_review_controller import S9PackageOption

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

    # Seek to the already-current sample (open lands on 0). Must still produce a frame.
    decode_id = controller.seek_sample(0)
    assert decode_id is not None
    snap = controller.decode_worker(decode_id)
    assert controller.apply_decode_snapshot(snap)
    assert snap.frame_bgr is not None
    assert controller.state.frame_bgr is not None


def test_draft_resume_completes_in_open_worker(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    controller = BehavioralReviewController()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        identity = controller.prepare_session(built["session_root"])
        assert controller.apply_discovery_snapshot(
            controller.collect_discovery_snapshot(identity)
        )
        from ui.controllers.behavioral_review_controller import S9PackageOption

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
        controller.set_reviewer_id("resp_operator")
        controller.enter_editing_mode()
        controller.create_draft(review_root=tmp_path / "resp_resume")
        controller.set_exclusive_label(
            start_sample_idx=0, end_sample_idx=0, label="retract"
        )
        draft_dir = controller.state.review_run_dir
        run_id = controller.state.review_run_id
        assert draft_dir is not None and run_id is not None
        controller.close_session(release_lock=True)

        controller2 = BehavioralReviewController()
        identity2 = controller2.prepare_session(built["session_root"])
        assert controller2.apply_discovery_snapshot(
            controller2.collect_discovery_snapshot(identity2)
        )
        from ui.controllers.behavioral_review_controller import (
            ReviewRunOption,
            S9PackageOption,
        )

        controller2.state.s9_options = (
            S9PackageOption(
                run_id=built["package_run_id"],
                run_dir=Path(built["package_dir"]),
                historical_status="completed",
                classification="eligible",
                currently_eligible=True,
                n_bouts=None,
                n_dense_samples=None,
                label=built["package_run_id"],
                reasons=(),
            ),
        )
        controller2.state.review_options = (
            ReviewRunOption(
                run_id=str(run_id),
                run_dir=Path(draft_dir),
                classification="draft",
                currently_eligible=False,
                parent_package_run_id=built["package_run_id"],
                publication_state="draft",
                historical_status=None,
                superseded_by=None,
                label=f"{run_id} · draft",
                is_draft=True,
            ),
        )
        controller2.select_s9_package(built["package_run_id"])
        controller2.select_review_run(str(run_id))
        open_draft = controller2.begin_open_draft()
        snap = controller2.open_source_worker(
            open_draft, reviewer_id="resp_operator", force_stale_lock=True
        )
        assert snap.draft_resumed
        assert snap.session.draft is not None
        assert controller2.apply_open_snapshot(snap, reviewer_id="resp_operator")
        assert controller2.state.draft_open
        assert controller2.state.can_undo
