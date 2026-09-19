"""Sprint 4 stabilization: output root, legend, edit latency, initial run load."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication
from tests.tracking_correction.test_manual_edits import (
    TIMESTAMP,
    _completed_s2,
    _completed_s3,
    _flush_working,
)
from tests.tracking_correction.test_review import _completed_s3 as _review_completed_s3

from tracking_correction.contracts import TrackingCorrectionRequest
from tracking_correction.review import (
    TRACK_COLORS_BGR,
    CorrectionEpisode,
    load_review_session,
    track_legend_entries,
)
from tracking_correction.runner import project_root_from_s2_run_dir, run_tracking_correction
from ui.controllers.pose_inference_controller import PoseInferenceController
from ui.pages.tracking_review_page import TrackingReviewPage

_APPLICATION = QApplication.instance() or QApplication([])


def test_s3_output_follows_s2_layout_not_stale_provenance(tmp_path: Path) -> None:
    """S2 metadata may reference another drive; S3 output stays under active project."""

    active_host = tmp_path / "active_host"
    foreign = tmp_path / "foreign_nas_Z"
    s2_run, s1 = _completed_s2(active_host)
    active = s1["session_root"]
    # Rewrite provenance to a different "drive"/root while keeping readable S1 copies.
    foreign.mkdir()
    for key in ("prepared_video", "prepare_meta", "prepared_sync"):
        target = foreign / Path(s1[key]).name
        target.write_bytes(s1[key].read_bytes())
        s1[key] = target
    foreign_preprocess = foreign / "preprocess"
    foreign_preprocess.mkdir(exist_ok=True)
    pose_meta = json.loads((s2_run / "pose_meta.json").read_text(encoding="utf-8"))
    pose_meta["input"] = {
        "session_root": str(foreign.resolve()),
        "preprocess_dir": str(foreign_preprocess.resolve()),
        "prepared_video": str(s1["prepared_video"].resolve()),
        "prepare_meta": str(s1["prepare_meta"].resolve()),
        "prepared_sync": str(s1["prepared_sync"].resolve()),
    }
    (s2_run / "pose_meta.json").write_text(json.dumps(pose_meta, indent=2) + "\n", encoding="utf-8")

    assert project_root_from_s2_run_dir(s2_run) == active.resolve()
    result = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=s2_run,
            dry_run=False,
            timestamp=TIMESTAMP,
        )
    )
    assert result.success is True
    assert result.run_dir.parent == (active / "tracking_correction").resolve()
    assert active.resolve() in result.run_dir.resolve().parents
    assert foreign.resolve() not in result.run_dir.resolve().parents

    meta = json.loads(result.run_meta_path.read_text(encoding="utf-8"))
    # Historical provenance remains the foreign paths; only placement changed.
    assert Path(meta["input"]["session_root"]).resolve() == foreign.resolve()
    assert Path(meta["input"]["prepared_video"]).resolve() == s1["prepared_video"].resolve()


def test_gui_s3_request_binds_output_root_to_opened_project(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    controller = PoseInferenceController(
        discovery=lambda _root: (_ for _ in ()).throw(RuntimeError("unused"))
    )
    controller.state.session_root = s1["session_root"].resolve()
    controller.state.task_running = False
    request = controller.begin_s3_correction(s2_run)
    assert request.output_root is not None
    assert Path(request.output_root).resolve() == (
        s1["session_root"] / "tracking_correction"
    ).resolve()
    assert Path(request.s2_run_dir).resolve() == s2_run.resolve()


def test_track_legend_matches_overlay_colors() -> None:
    entries = track_legend_entries()
    assert {entry.track for entry in entries} == set(TRACK_COLORS_BGR)
    for entry in entries:
        assert entry.color_bgr == TRACK_COLORS_BGR[entry.track]
        # Overlay lookup uses the same mapping.
        assert TRACK_COLORS_BGR.get(entry.track) == entry.color_bgr
    # Authoritative profile role for track 0.
    by_track = {entry.track: entry for entry in entries}
    assert by_track[0].role_label == "implanted/headstage"
    assert by_track[1].role_label is None


def test_manual_edit_updates_overlay_immediately_without_waiting_for_disk(
    tmp_path: Path,
) -> None:
    s3_dir, _s2, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    session.set_frame(3)
    before = session.pose_points_for_frame(3)
    nose_before = next(p for p in before if p.track == 0 and p.node == "nose")

    t0 = time.perf_counter()
    session.blank_node("nose", track=0, frame_idx=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    after = session.pose_points_for_frame(3)
    assert not any(p.track == 0 and p.node == "nose" for p in after)
    assert any(p.node == "nose" for p in before)
    assert nose_before.x == pytest.approx(nose_before.x)
    # In-memory path should return well under a second even on modest fixtures.
    assert elapsed_ms < 500.0

    # Disk catch-up remains durable after flush.
    _flush_working(session)
    disk = pd.read_parquet(session.working_tracked_pose_path)
    row = disk[(disk["frame_idx"] == 3) & (disk["track"].astype(int) == 0) & (disk["node"] == "nose")]
    assert len(row) == 1
    assert pd.isna(row.iloc[0]["x"])


def test_acceptance_flushes_latest_edit(tmp_path: Path) -> None:
    s3_dir, _s2, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    session.blank_node("nose", track=0, frame_idx=1)
    # Do not flush explicitly; accept must wait for the latest generation.
    result = session.accept_tracking()
    assert result.already_accepted is False
    tracked = pd.read_parquet(result.tracked_pose_path)
    row = tracked[
        (tracked["frame_idx"] == 1)
        & (tracked["track"].astype(int) == 0)
        & (tracked["node"] == "nose")
    ]
    assert len(row) == 1
    assert pd.isna(row.iloc[0]["x"])
    assert result.working_sha256 == hashlib.sha256(
        session.working_tracked_pose_path.read_bytes()
    ).hexdigest()


def test_first_discovered_s3_run_is_loaded(tmp_path: Path) -> None:
    s2_run, s1 = _completed_s2(tmp_path)
    first = run_tracking_correction(
        TrackingCorrectionRequest(s2_run_dir=s2_run, dry_run=False, timestamp="20260301T100000")
    )
    second = run_tracking_correction(
        TrackingCorrectionRequest(s2_run_dir=s2_run, dry_run=False, timestamp="20260301T110000")
    )
    assert first.success and second.success

    page = TrackingReviewPage()
    page.open_project(s1["session_root"])
    assert page.controller.session is not None
    assert page.run_combo.count() == 2
    # Discovery sorts newest first; the visually selected item must match the loaded run.
    selected_data = Path(str(page.run_combo.currentData()))
    assert page.controller.session.run_dir.resolve() == selected_data.resolve()
    assert page.run_combo.currentIndex() == 0


def test_episode_list_selection_navigates(tmp_path: Path) -> None:
    s3_dir, _s2, _s1 = _review_completed_s3(tmp_path)
    corrections = s3_dir / "machine_corrections.json"
    corrections.write_text(
        json.dumps(
            [
                {"frame": 1, "type": "node_overlap"},
                {"frame": 2, "type": "node_overlap"},
                {"frame": 5, "type": "jump"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    page = TrackingReviewPage()
    page.open_run(s3_dir)
    # Visible list is manual edits only; machine records stay on Auto ◀/▶.
    assert page.episode_list.count() == 0
    assert page.controller.session is not None
    assert len(page.controller.session.correction_episodes) >= 2
    page.controller.next_correction_episode()
    assert page.controller.session.current_frame == 1

    session = page.controller.session
    session.swap_identities(1, 3)
    session.blank_node("nose", track=0, frame_idx=5)
    page._populate_episode_list(session.navigable_episodes)
    assert page.episode_list.count() == 2
    texts = [
        page.episode_list.item(i).text() for i in range(page.episode_list.count())
    ]
    assert texts[0].startswith("1 | swap_identities | 1–3 |")
    assert "tracks" in texts[0]
    assert texts[1].startswith("2 | blank_node | f5 |")
    assert "nose" in texts[1]
    assert "track 0" in texts[1]

    page.episode_list.setCurrentRow(1)
    assert page.controller.session.current_frame == 5
    page.episode_list.setCurrentRow(0)
    assert page.controller.session.current_frame == 1


def test_jump_to_episode_api(tmp_path: Path) -> None:
    s3_dir, _s2, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    episode = CorrectionEpisode(
        start_frame=3,
        end_frame=3,
        correction_types=("blank_node",),
        source="manual",
        node="nose",
        track=0,
    )
    assert session.jump_to_episode(episode) == 3
    assert session.current_frame == 3
