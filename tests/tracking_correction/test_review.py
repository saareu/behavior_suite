"""Sprint 3: S3 review session loading, navigation, and acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction import (
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    load_review_session,
    run_tracking_correction,
)
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    FINAL_TRACKED_POSE_FILENAME,
    WORKING_TRACKED_POSE_FILENAME,
)
from tracking_correction.review import PoseDisplaySource
from ui.controllers.tracking_review_controller import TrackingReviewController

REQUIRED_NODES = ("nose", "neck", "spine_base", "tail_base", "headstage")
_NODE_XY = {
    "nose": (0.0, 0.0),
    "neck": (30.0, -40.0),
    "spine_base": (55.0, -20.0),
    "tail_base": (90.0, -10.0),
    "headstage": (10.0, -55.0),
}
TIMESTAMP = "20260301T120000"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {path: (_sha256(path), path.stat().st_size) for path in paths if path.exists()}


def _write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (64, 64),
    )
    assert writer.isOpened()
    for frame_idx in range(frame_count):
        writer.write(np.full((64, 64, 3), frame_idx % 256, dtype=np.uint8))
    writer.release()


def _write_s1(session_root: Path, *, frame_count: int = 8) -> dict[str, Path]:
    preprocess = session_root / "preprocess"
    preprocess.mkdir(parents=True)
    video = preprocess / "prepared_video.mp4"
    meta = preprocess / "prepare_meta.json"
    sync_path = preprocess / "prepared_sync.npz"
    _write_video(video, frame_count)
    meta.write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "opencv_reported_frame_count": frame_count,
                    "opencv_readable_frame_count": frame_count,
                    "frame_count_used_for_sleap": frame_count,
                    "fps_header": 20.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=20.0,
        raw_fps_effective=20.0,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / 20.0,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(sync_path, sync)
    return {
        "session_root": session_root,
        "preprocess_dir": preprocess,
        "prepared_video": video,
        "prepare_meta": meta,
        "prepared_sync": sync_path,
    }


def _pose_rows(
    *,
    frame_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_idx in frame_indices:
        for track_index, track in ((0, "track_0"), (1, "track_1")):
            origin_x = 140.0 + track_index * 180.0 + frame_idx
            origin_y = 220.0
            for node in REQUIRED_NODES:
                if node == "headstage" and track_index == 1:
                    continue
                dx, dy = _NODE_XY[node]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "video_index": 0,
                        "track": track,
                        "node": node,
                        "x": float(origin_x + dx),
                        "y": float(origin_y + dy),
                        "score": 0.9,
                    }
                )
    return rows


def _write_pose(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _completed_s2(tmp_path: Path, *, frame_count: int = 8) -> tuple[Path, dict[str, Path]]:
    session = tmp_path / "session"
    s1 = _write_s1(session, frame_count=frame_count)
    run_dir = session / "pose_inference" / "run_a"
    run_dir.mkdir(parents=True)
    _write_pose(run_dir / "pose.parquet", _pose_rows(frame_indices=tuple(range(frame_count))))
    (run_dir / "pose_meta.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dry_run": False,
                "run_id": "run_a",
                "input": {
                    "session_root": str(s1["session_root"]),
                    "preprocess_dir": str(s1["preprocess_dir"]),
                    "prepared_video": str(s1["prepared_video"]),
                    "prepare_meta": str(s1["prepare_meta"]),
                    "prepared_sync": str(s1["prepared_sync"]),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, s1


def _protected_inputs(s2_run: Path, s1: dict[str, Path]) -> list[Path]:
    return [
        s2_run / "pose.parquet",
        s2_run / "pose_meta.json",
        s1["prepared_video"],
        s1["prepare_meta"],
        s1["prepared_sync"],
    ]


def _completed_s3(tmp_path: Path, *, frame_count: int = 8) -> tuple[Path, Path, dict[str, Path]]:
    s2_run, s1 = _completed_s2(tmp_path, frame_count=frame_count)
    result = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=s2_run,
            dry_run=False,
            timestamp=TIMESTAMP,
        )
    )
    assert result.success is True
    assert result.run_dir.is_dir()
    return result.run_dir, s2_run, s1


def test_load_completed_s3_run_resolves_video_and_s2_pose(tmp_path: Path) -> None:
    s3_dir, s2_run, s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)

    assert session.run_dir == s3_dir.resolve()
    assert session.prepared_video_path == s1["prepared_video"].resolve()
    assert session.s2_pose_path == (s2_run / "pose.parquet").resolve()
    assert session.working_tracked_pose_path.name == WORKING_TRACKED_POSE_FILENAME
    assert session.frame_count == 8
    assert session.fps_header == 20.0
    assert session.fps_source == "S1 timing"
    assert session.fps_source_detail == "prepared_sync.npz"
    assert session.acceptance_state == "not_accepted"
    assert session.current_frame == 0


def test_review_playback_fps_prefers_s1_timing_over_video_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracking_correction.review import (
        FPS_SOURCE_S1_TIMING,
        FPS_SOURCE_VIDEO_HEADER_FALLBACK,
    )

    s3_dir, _s2_run, s1 = _completed_s3(tmp_path)
    s1_fps = 119.048
    video_header_fps = 119.0

    # Authoritative S1 sync/meta use the precise prepared time base.
    sync = build_prepared_sync(
        prepared_frame_count=8,
        start_frame=0,
        end_frame_exclusive=8,
        fps_header=s1_fps,
        raw_fps_effective=s1_fps,
        raw_pts_time_sec=np.arange(8, dtype=float) / s1_fps,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=8,
        prepared_frame_count_opencv_reported=8,
        prepared_frame_count_opencv_readable=8,
    )
    write_prepared_sync_npz(s1["prepared_sync"], sync)
    meta_payload = json.loads(s1["prepare_meta"].read_text(encoding="utf-8"))
    meta_payload["prepared_video"]["fps_header"] = s1_fps
    s1["prepare_meta"].write_text(json.dumps(meta_payload) + "\n", encoding="utf-8")

    # Record a rounded video-header rate in S3 run_meta.
    run_meta_path = s3_dir / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_meta["timing"]["fps_header"] = video_header_fps
    run_meta_path.write_text(json.dumps(run_meta) + "\n", encoding="utf-8")

    session = load_review_session(s3_dir)
    assert session.fps_header == pytest.approx(s1_fps)
    assert session.fps_source == FPS_SOURCE_S1_TIMING
    assert session.fps_source_detail == "prepared_sync.npz"
    assert session.fps_header != video_header_fps

    # Without S1 sync/meta, fall back to video-header FPS.
    s1["prepared_sync"].unlink()
    s1["prepare_meta"].unlink()
    monkeypatch.setattr(
        "tracking_correction.review._fps_from_video_header",
        lambda _path: video_header_fps,
    )
    fallback = load_review_session(s3_dir)
    assert fallback.fps_header == pytest.approx(video_header_fps)
    assert fallback.fps_source == FPS_SOURCE_VIDEO_HEADER_FALLBACK


def test_working_pose_overlay_lookup_by_frame(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)

    points = session.pose_points_for_frame(3)
    assert points
    assert all(point.track in {0, 1} for point in points)
    assert {point.node for point in points}

    session.set_frame(3)
    assert session.current_frame == 3
    assert session.pose_points_for_frame() == points


def test_source_corrected_toggle_selects_pose_tables(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    session.set_frame(2)

    corrected = session.pose_points_for_frame(source=PoseDisplaySource.CORRECTED)
    provisional = session.pose_points_for_frame(source=PoseDisplaySource.PROVISIONAL)
    assert corrected
    assert provisional

    session.set_pose_source(PoseDisplaySource.PROVISIONAL)
    assert session.pose_source is PoseDisplaySource.PROVISIONAL
    assert session.pose_points_for_frame() == provisional
    session.set_pose_source(PoseDisplaySource.CORRECTED)
    assert session.pose_points_for_frame() == corrected


def test_correction_episode_navigation(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    corrections_path = s3_dir / "machine_corrections.json"
    corrections_path.write_text(
        json.dumps(
            [
                {"frame": 1, "type": "node_overlap", "track": 0},
                {"frame": 1, "type": "node_overlap", "track": 1},
                {"frame": 1, "type": "skeleton_stretch", "track": 0},
                {"frame": 2, "type": "node_overlap", "track": 0},
                {"frame": 3, "type": "skeleton_stretch", "track": 1},
                {"frame": 5, "type": "jump_overlap_blank", "track": 1},
                {"frame": 7, "type": "track_reassignment"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    session = load_review_session(s3_dir)
    assert session.correction_frames == (1, 2, 3, 5, 7)
    assert len(session.correction_episodes) == 3
    assert session.correction_episodes[0].start_frame == 1
    assert session.correction_episodes[0].end_frame == 3
    assert session.correction_episodes[0].correction_types == (
        "node_overlap",
        "skeleton_stretch",
    )
    assert session.correction_episodes[1].start_frame == 5
    assert session.correction_episodes[1].end_frame == 5
    assert session.correction_episodes[2].start_frame == 7

    first = session.next_correction_episode()
    assert first is not None
    assert first.start_frame == 1
    assert session.current_frame == 1
    session.set_frame(2)
    assert session.current_correction_episode() is first
    assert session.next_correction_episode() is not None
    assert session.current_frame == 5
    assert session.next_correction_episode() is not None
    assert session.current_frame == 7
    assert session.next_correction_episode() is None
    assert session.previous_correction_episode() is not None
    assert session.current_frame == 5
    session.set_frame(6)
    assert session.previous_correction_episode() is not None
    assert session.current_frame == 5
    assert session.previous_correction_episode() is not None
    assert session.current_frame == 1
    assert session.previous_correction_episode() is None


def test_frame_domain_preserved_in_navigation_and_controller(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path, frame_count=5)
    session = load_review_session(s3_dir)
    assert session.frame_count == 5
    assert session.max_frame_idx == 4

    with pytest.raises(TrackingCorrectionError, match="outside prepared-frame domain"):
        session.set_frame(5)
    with pytest.raises(TrackingCorrectionError, match="outside prepared-frame domain"):
        session.set_frame(-1)

    controller = TrackingReviewController()
    controller.open_run(s3_dir)
    view = controller.current_view()
    assert view.frame_idx == 0
    assert view.frame_count == 5
    assert view.image_bgr.shape[0] == 64
    controller.set_frame(4)
    assert controller.require_session().current_frame == 4
    assert controller.current_view().frame_idx == 4


def test_acceptance_writes_tracked_pose_equal_to_working(tmp_path: Path) -> None:
    s3_dir, s2_run, s1 = _completed_s3(tmp_path)
    protected = _snapshot(_protected_inputs(s2_run, s1))
    working = s3_dir / WORKING_TRACKED_POSE_FILENAME
    working_before = _sha256(working)
    session = load_review_session(s3_dir)

    result = session.accept_tracking()

    tracked = s3_dir / FINAL_TRACKED_POSE_FILENAME
    assert result.already_accepted is False
    assert tracked.is_file()
    assert _sha256(tracked) == working_before
    assert _sha256(working) == working_before
    assert pd.read_parquet(tracked).equals(pd.read_parquet(working))

    meta = json.loads((s3_dir / "run_meta.json").read_text(encoding="utf-8"))
    settings = yaml.safe_load((s3_dir / "settings_used.yaml").read_text(encoding="utf-8"))
    log = (s3_dir / "processing_log.txt").read_text(encoding="utf-8")
    assert meta["acceptance_state"] == ACCEPTANCE_ACCEPTED
    assert meta["acceptance"]["working_tracked_pose_sha256"] == working_before
    assert meta["acceptance"]["accepted_at"]
    assert meta["outputs"]["tracked_pose_parquet"] == str(tracked.resolve())
    assert settings["acceptance_state"] == ACCEPTANCE_ACCEPTED
    assert "acceptance_state: accepted" in log
    assert _snapshot(_protected_inputs(s2_run, s1)) == protected


def test_acceptance_is_idempotent(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    first = session.accept_tracking()
    tracked_sha = _sha256(first.tracked_pose_path)

    second = load_review_session(s3_dir).accept_tracking()
    assert second.already_accepted is True
    assert _sha256(second.tracked_pose_path) == tracked_sha


def test_refuse_incomplete_failed_and_missing_working_pose(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)

    dry = json.loads((s3_dir / "run_meta.json").read_text(encoding="utf-8"))
    dry["dry_run"] = True
    (s3_dir / "run_meta.json").write_text(json.dumps(dry) + "\n", encoding="utf-8")
    with pytest.raises(TrackingCorrectionError, match="dry-run"):
        load_review_session(s3_dir)

    failed = dict(dry)
    failed["dry_run"] = False
    failed["status"] = "correction_failed"
    failed["backend_status"] = "failed"
    (s3_dir / "run_meta.json").write_text(json.dumps(failed) + "\n", encoding="utf-8")
    with pytest.raises(TrackingCorrectionError, match="not ready"):
        load_review_session(s3_dir)

    ok = dict(failed)
    ok["status"] = "correction_complete"
    ok["backend_status"] = "completed"
    (s3_dir / "run_meta.json").write_text(json.dumps(ok) + "\n", encoding="utf-8")
    (s3_dir / WORKING_TRACKED_POSE_FILENAME).unlink()
    with pytest.raises(TrackingCorrectionError, match="Working tracked pose is missing"):
        load_review_session(s3_dir)


def test_review_controller_toggle_and_accept(tmp_path: Path) -> None:
    s3_dir, s2_run, s1 = _completed_s3(tmp_path)
    before = _snapshot(_protected_inputs(s2_run, s1))
    controller = TrackingReviewController()
    controller.open_run(s3_dir)
    controller.set_pose_source(PoseDisplaySource.PROVISIONAL)
    provisional_view = controller.current_view()
    controller.set_pose_source(PoseDisplaySource.CORRECTED)
    corrected_view = controller.current_view()
    assert provisional_view.pose_source == PoseDisplaySource.PROVISIONAL.value
    assert corrected_view.pose_source == PoseDisplaySource.CORRECTED.value

    result = controller.accept_tracking()
    assert result.tracked_pose_path.is_file()
    assert _snapshot(_protected_inputs(s2_run, s1)) == before


def test_playback_follows_wall_clock_and_skips_frames(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path, frame_count=40)
    controller = TrackingReviewController(
        frame_reader=lambda _path, frame_idx: np.full(
            (8, 8, 3), frame_idx % 256, dtype=np.uint8
        )
    )
    session = controller.open_run(s3_dir)
    assert session.fps_header == 20.0
    controller.set_frame(0)
    controller.begin_playback(now_monotonic=100.0)
    assert controller.sync_playback_frame(now_monotonic=100.0) == 0
    # 0.25s at 20 FPS * 1.0× => target frame 5
    assert controller.sync_playback_frame(now_monotonic=100.25) == 5
    controller.set_playback_speed(2.0)
    controller.begin_playback(now_monotonic=200.0)
    # Speed change re-anchors at current frame 5.
    assert controller.sync_playback_frame(now_monotonic=200.0) == 5
    # 0.25s at 20 FPS * 2.0× => +10 frames => 15
    assert controller.sync_playback_frame(now_monotonic=200.25) == 15
    # Past the end stops playback.
    assert controller.sync_playback_frame(now_monotonic=250.0) is None
    assert controller.playback_active is False
    assert controller.require_session().current_frame == 39


def test_playback_target_frame_formula() -> None:
    from ui.controllers.tracking_review_controller import PlaybackTimingState

    clock = PlaybackTimingState(
        active=True,
        anchor_frame=10,
        anchor_monotonic=0.0,
        source_fps=119.0,
        playback_speed=1.0,
    )
    assert clock.target_frame(0.0) == 10
    assert clock.target_frame(1.0) == 129
    clock_half = PlaybackTimingState(
        active=True,
        anchor_frame=0,
        anchor_monotonic=0.0,
        source_fps=119.0,
        playback_speed=0.5,
    )
    assert clock_half.target_frame(2.0) == 119


def test_paused_frame_stepping_remains_deterministic(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path, frame_count=5)
    controller = TrackingReviewController(
        frame_reader=lambda _path, frame_idx: np.full(
            (4, 4, 3), frame_idx % 256, dtype=np.uint8
        )
    )
    controller.open_run(s3_dir)
    controller.begin_playback(now_monotonic=1.0)
    controller.step_forward()
    assert controller.playback_active is False
    assert controller.require_session().current_frame == 1
    controller.step_forward()
    controller.step_backward()
    assert controller.require_session().current_frame == 1
