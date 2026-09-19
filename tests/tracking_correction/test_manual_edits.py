"""Sprint 4: minimal manual tracking correction on the S3 working pose."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tracking_correction import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    is_currently_accepted,
    load_review_session,
    run_tracking_correction,
)
from tracking_correction.contracts import (
    AUTOMATIC_TRACKED_POSE_FILENAME,
    FINAL_TRACKED_POSE_FILENAME,
    MACHINE_CORRECTIONS_FILENAME,
    MANUAL_CORRECTIONS_FILENAME,
)
from tracking_correction.manual_edits import (
    apply_swap_identities,
    apply_swap_node,
    frame_domain_signature,
    read_working_pose,
)
from ui.controllers.tracking_review_controller import TrackingReviewController

REQUIRED_NODES = ("nose", "neck", "spine_base", "tail_base", "headstage")
_NODE_XY = {
    "nose": (0.0, 0.0),
    "neck": (30.0, -40.0),
    "spine_base": (55.0, -20.0),
    "tail_base": (90.0, -10.0),
    "headstage": (10.0, -55.0),
}
TIMESTAMP = "20260301T130000"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(paths: list[Path]) -> dict[Path, tuple[str, int]]:
    return {path: (_sha256(path), path.stat().st_size) for path in paths if path.exists()}


def _write_video(path: Path, frame_count: int) -> None:
    import cv2

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
    from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz

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
    return result.run_dir, s2_run, s1


def _flush_working(session) -> None:
    session.flush_working_pose_persistence()


def _disk_working(session) -> pd.DataFrame:
    _flush_working(session)
    return read_working_pose(session.working_tracked_pose_path)


def _coords(frame: pd.DataFrame, *, frame_idx: int, track: int, node: str) -> tuple[float, float]:
    frame_col = "frame_idx" if "frame_idx" in frame.columns else "frame"
    match = frame[
        (frame[frame_col] == frame_idx)
        & (frame["track"].astype(int) == track)
        & (frame["node"].astype(str) == node)
    ]
    assert len(match) == 1
    return float(match.iloc[0]["x"]), float(match.iloc[0]["y"])


def test_interval_identity_swap_and_one_frame_swap(tmp_path: Path) -> None:
    s3_dir, s2_run, s1 = _completed_s3(tmp_path)
    protected = _snapshot(_protected_inputs(s2_run, s1))
    session = load_review_session(s3_dir)
    before = _disk_working(session)
    domain_before = frame_domain_signature(before)
    nose_0 = _coords(before, frame_idx=2, track=0, node="nose")
    nose_1 = _coords(before, frame_idx=2, track=1, node="nose")
    machine_before = _sha256(s3_dir / MACHINE_CORRECTIONS_FILENAME)

    result = session.swap_identities(2, 4)
    assert result.action == "swap_identities"
    assert result.start_frame == 2
    assert result.end_frame == 4
    assert session.manual_edit_count == 1

    after = _disk_working(session)
    assert frame_domain_signature(after) == domain_before
    assert _coords(after, frame_idx=2, track=0, node="nose") == pytest.approx(nose_1)
    assert _coords(after, frame_idx=2, track=1, node="nose") == pytest.approx(nose_0)
    assert _coords(after, frame_idx=1, track=0, node="nose") == pytest.approx(
        _coords(before, frame_idx=1, track=0, node="nose")
    )
    assert (s3_dir / AUTOMATIC_TRACKED_POSE_FILENAME).is_file()
    baseline = read_working_pose(s3_dir / AUTOMATIC_TRACKED_POSE_FILENAME)
    assert _coords(baseline, frame_idx=2, track=0, node="nose") == pytest.approx(nose_0)
    _flush_working(session)
    assert not pd.read_parquet(session.working_tracked_pose_path).equals(baseline)

    one = session.swap_identities(5, 5)
    assert one.start_frame == 5 and one.end_frame == 5
    after_one = _disk_working(session)
    assert _coords(after_one, frame_idx=5, track=0, node="nose") == pytest.approx(
        _coords(before, frame_idx=5, track=1, node="nose")
    )
    assert _sha256(s3_dir / MACHINE_CORRECTIONS_FILENAME) == machine_before
    assert _snapshot(_protected_inputs(s2_run, s1)) == protected


def test_node_swap_and_blank(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    session.set_frame(3)
    before = _disk_working(session)
    domain_before = frame_domain_signature(before)
    nose_0 = _coords(before, frame_idx=3, track=0, node="nose")
    nose_1 = _coords(before, frame_idx=3, track=1, node="nose")

    session.swap_node("nose")
    mid = _disk_working(session)
    assert frame_domain_signature(mid) == domain_before
    assert _coords(mid, frame_idx=3, track=0, node="nose") == pytest.approx(nose_1)
    assert _coords(mid, frame_idx=3, track=1, node="nose") == pytest.approx(nose_0)
    # Other nodes unchanged on this frame.
    assert _coords(mid, frame_idx=3, track=0, node="neck") == pytest.approx(
        _coords(before, frame_idx=3, track=0, node="neck")
    )

    session.blank_node("neck", track=1)
    blanked = _disk_working(session)
    assert frame_domain_signature(blanked) == domain_before
    row = blanked[
        (blanked["frame_idx"] == 3)
        & (blanked["track"].astype(int) == 1)
        & (blanked["node"].astype(str) == "neck")
    ].iloc[0]
    assert pd.isna(row["x"]) and pd.isna(row["y"])
    points = session.pose_points_for_frame(3)
    assert not any(p.track == 1 and p.node == "neck" for p in points)


def test_automatic_baseline_preserved_and_reset(tmp_path: Path) -> None:
    s3_dir, s2_run, s1 = _completed_s3(tmp_path)
    protected = _snapshot(_protected_inputs(s2_run, s1))
    session = load_review_session(s3_dir)
    original_sha = _sha256(session.working_tracked_pose_path)
    original = _disk_working(session)

    session.swap_identities(0, 1)
    session.blank_node("nose", track=0, frame_idx=2)
    assert session.manual_edit_count == 2
    baseline_sha = _sha256(s3_dir / AUTOMATIC_TRACKED_POSE_FILENAME)
    assert baseline_sha == original_sha
    _flush_working(session)
    assert _sha256(session.working_tracked_pose_path) != original_sha

    session.reset_manual_edits()
    assert session.manual_edit_count == 0
    assert _sha256(session.working_tracked_pose_path) == original_sha
    assert pd.read_parquet(session.working_tracked_pose_path).equals(original)
    records = json.loads((s3_dir / MANUAL_CORRECTIONS_FILENAME).read_text(encoding="utf-8"))
    assert records == []
    assert _snapshot(_protected_inputs(s2_run, s1)) == protected


def test_undo_last_manual_edit(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    before = _disk_working(session)
    nose_before = _coords(before, frame_idx=1, track=0, node="nose")

    session.swap_identities(1, 1)
    session.blank_node("nose", track=0, frame_idx=1)
    assert session.manual_edit_count == 2

    session.undo_last_manual_edit()
    assert session.manual_edit_count == 1
    mid = _disk_working(session)
    # Blank undone; identity swap still active.
    assert _coords(mid, frame_idx=1, track=0, node="nose") == pytest.approx(
        _coords(before, frame_idx=1, track=1, node="nose")
    )

    session.undo_last_manual_edit()
    assert session.manual_edit_count == 0
    restored = _disk_working(session)
    assert _coords(restored, frame_idx=1, track=0, node="nose") == pytest.approx(nose_before)
    with pytest.raises(TrackingCorrectionError, match="No manual edits"):
        session.undo_last_manual_edit()


def test_manual_corrections_json_provenance(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    session.set_frame(4)
    session.swap_identities(3, 4)
    session.swap_node("nose")
    session.blank_node("tail_base", track=1)

    path = s3_dir / MANUAL_CORRECTIONS_FILENAME
    assert path.is_file()
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == 3
    assert records[0]["action"] == "swap_identities"
    assert records[0]["start_frame"] == 3
    assert records[0]["end_frame"] == 4
    assert records[0]["tracks"] == [0, 1]
    assert records[0]["source"] == "user"
    assert records[0]["timestamp"]
    assert records[1]["action"] == "swap_node"
    assert records[1]["node"] == "nose"
    assert records[2]["action"] == "blank_node"
    assert records[2]["node"] == "tail_base"
    assert records[2]["track"] == 1
    assert "before" in records[2]
    assert "x" in records[2]["before"]
    # Machine corrections remain a separate artifact.
    machine = json.loads((s3_dir / MACHINE_CORRECTIONS_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(machine, list)


def test_acceptance_after_manual_edits_and_invalidation(tmp_path: Path) -> None:
    s3_dir, s2_run, s1 = _completed_s3(tmp_path)
    protected = _snapshot(_protected_inputs(s2_run, s1))
    session = load_review_session(s3_dir)

    first = session.accept_tracking()
    assert first.already_accepted is False
    assert first.manual_corrections_present is False
    tracked_sha = _sha256(s3_dir / FINAL_TRACKED_POSE_FILENAME)

    edit = session.swap_identities(0, 0)
    assert edit.acceptance_invalidated is True
    assert session.acceptance_state == ACCEPTANCE_SUPERSEDED
    meta = json.loads((s3_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["acceptance_state"] == ACCEPTANCE_SUPERSEDED
    settings = yaml.safe_load((s3_dir / "settings_used.yaml").read_text(encoding="utf-8"))
    assert settings["acceptance_state"] == ACCEPTANCE_SUPERSEDED

    _flush_working(session)
    working_sha = _sha256(session.working_tracked_pose_path)
    assert working_sha != tracked_sha

    reaccepted = session.accept_tracking()
    assert reaccepted.already_accepted is False
    assert reaccepted.manual_corrections_present is True
    assert reaccepted.manual_correction_count == 1
    assert _sha256(s3_dir / FINAL_TRACKED_POSE_FILENAME) == working_sha
    assert pd.read_parquet(s3_dir / FINAL_TRACKED_POSE_FILENAME).equals(
        pd.read_parquet(session.working_tracked_pose_path)
    )
    meta_after = json.loads((s3_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta_after["acceptance_state"] == ACCEPTANCE_ACCEPTED
    assert meta_after["acceptance"]["manual_corrections_present"] is True
    assert meta_after["acceptance"]["working_tracked_pose_sha256"] == working_sha
    assert _snapshot(_protected_inputs(s2_run, s1)) == protected


def test_controller_manual_edit_keeps_frame_and_updates_overlay(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    controller = TrackingReviewController(
        frame_reader=lambda _path, frame_idx: np.full(
            (8, 8, 3), frame_idx % 256, dtype=np.uint8
        )
    )
    controller.open_run(s3_dir)
    controller.set_frame(2)
    before_points = {
        (p.track, p.node): (p.x, p.y) for p in controller.current_view().pose_points
    }
    controller.swap_identities(2, 2)
    assert controller.require_session().current_frame == 2
    view = controller.current_view()
    assert view.frame_idx == 2
    assert view.manual_edit_count == 1
    after_points = {(p.track, p.node): (p.x, p.y) for p in view.pose_points}
    assert after_points[(0, "nose")] == pytest.approx(before_points[(1, "nose")])
    assert after_points[(1, "nose")] == pytest.approx(before_points[(0, "nose")])


def _row(
    frame: pd.DataFrame, *, frame_idx: int, track: int, node: str
) -> pd.Series:
    frame_col = "frame_idx" if "frame_idx" in frame.columns else "frame"
    match = frame[
        (frame[frame_col] == frame_idx)
        & (frame["track"].astype(int) == track)
        & (frame["node"].astype(str) == node)
    ]
    assert len(match) == 1
    return match.iloc[0]


def test_sparse_identity_swap_moves_complete_detection_rows() -> None:
    """Identity swap must move sparse nodes with all pose/provenance fields."""

    df = pd.DataFrame(
        [
            {
                "frame_idx": 1,
                "frame": 1,
                "track": 0,
                "node": "nose",
                "x": np.float32(1.0),
                "y": np.float32(2.0),
                "point_score": np.float32(0.9),
                "instance_score": np.float32(0.95),
                "x_raw": np.float32(1.5),
                "y_raw": np.float32(2.5),
                "point_score_raw": np.float32(0.91),
                "instance_score_raw": np.float32(0.96),
                "was_track_swapped": np.uint8(0),
                "was_node_blanked": np.uint8(0),
            },
            {
                "frame_idx": 1,
                "frame": 1,
                "track": 0,
                "node": "headstage",
                "x": np.float32(3.0),
                "y": np.float32(4.0),
                "point_score": np.float32(0.8),
                "instance_score": np.float32(0.85),
                "x_raw": np.float32(3.5),
                "y_raw": np.float32(4.5),
                "point_score_raw": np.float32(0.81),
                "instance_score_raw": np.float32(0.86),
                "was_track_swapped": np.uint8(0),
                "was_node_blanked": np.uint8(0),
            },
            {
                "frame_idx": 1,
                "frame": 1,
                "track": 1,
                "node": "nose",
                "x": np.float32(10.0),
                "y": np.float32(20.0),
                "point_score": np.float32(0.7),
                "instance_score": np.float32(0.75),
                "x_raw": np.float32(10.5),
                "y_raw": np.float32(20.5),
                "point_score_raw": np.float32(0.71),
                "instance_score_raw": np.float32(0.76),
                "was_track_swapped": np.uint8(0),
                "was_node_blanked": np.uint8(0),
            },
        ]
    )
    domain = frame_domain_signature(df)
    out = apply_swap_identities(df, start_frame=1, end_frame=1)
    assert frame_domain_signature(out) == domain
    # headstage existed only on track 0; after swap it belongs to track 1 with
    # every associated field intact (no invented track-0 headstage).
    assert len(out[(out["track"] == 0) & (out["node"] == "headstage")]) == 0
    hs = _row(out, frame_idx=1, track=1, node="headstage")
    assert float(hs["x"]) == pytest.approx(3.0)
    assert float(hs["x_raw"]) == pytest.approx(3.5)
    assert float(hs["point_score"]) == pytest.approx(0.8)
    assert float(hs["point_score_raw"]) == pytest.approx(0.81)
    assert float(hs["instance_score_raw"]) == pytest.approx(0.86)
    assert int(hs["was_track_swapped"]) == 0
    assert int(hs["was_node_blanked"]) == 0
    assert _coords(out, frame_idx=1, track=0, node="nose") == pytest.approx((10.0, 20.0))
    # Undo via second swap restores ownership without corrupting provenance.
    restored = apply_swap_identities(out, start_frame=1, end_frame=1)
    assert restored.sort_values(["frame_idx", "track", "node"]).reset_index(
        drop=True
    ).equals(df.sort_values(["frame_idx", "track", "node"]).reset_index(drop=True))


def test_sparse_node_swap_moves_single_sided_detection() -> None:
    df = pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "frame": 0,
                "track": 0,
                "node": "nose",
                "x": 1.0,
                "y": 2.0,
                "point_score": 0.9,
                "x_raw": 1.1,
                "y_raw": 2.1,
                "point_score_raw": 0.91,
                "was_track_swapped": np.uint8(0),
                "was_node_blanked": np.uint8(0),
            },
            {
                "frame_idx": 0,
                "frame": 0,
                "track": 0,
                "node": "headstage",
                "x": 3.0,
                "y": 4.0,
                "point_score": 0.8,
                "x_raw": 3.1,
                "y_raw": 4.1,
                "point_score_raw": 0.81,
                "was_track_swapped": np.uint8(0),
                "was_node_blanked": np.uint8(0),
            },
            {
                "frame_idx": 0,
                "frame": 0,
                "track": 1,
                "node": "nose",
                "x": 10.0,
                "y": 20.0,
                "point_score": 0.7,
                "x_raw": 10.1,
                "y_raw": 20.1,
                "point_score_raw": 0.71,
                "was_track_swapped": np.uint8(0),
                "was_node_blanked": np.uint8(0),
            },
        ]
    )
    domain = frame_domain_signature(df)
    moved = apply_swap_node(df, frame_idx=0, node="headstage")
    assert frame_domain_signature(moved) == domain
    assert len(moved[(moved["track"] == 0) & (moved["node"] == "headstage")]) == 0
    hs = _row(moved, frame_idx=0, track=1, node="headstage")
    assert float(hs["x"]) == pytest.approx(3.0)
    assert float(hs["x_raw"]) == pytest.approx(3.1)
    assert float(hs["point_score_raw"]) == pytest.approx(0.81)
    # Re-applying moves it back (undo path).
    restored = apply_swap_node(moved, frame_idx=0, node="headstage")
    assert len(restored[(restored["track"] == 1) & (restored["node"] == "headstage")]) == 0
    hs0 = _row(restored, frame_idx=0, track=0, node="headstage")
    assert float(hs0["x_raw"]) == pytest.approx(3.1)


def test_blank_undo_restores_complete_row_including_provenance(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    before = _disk_working(session)
    target = _row(before, frame_idx=2, track=0, node="nose")
    provenance_cols = [
        c
        for c in (
            "x",
            "y",
            "point_score",
            "instance_score",
            "x_raw",
            "y_raw",
            "point_score_raw",
            "instance_score_raw",
            "was_track_swapped",
            "was_node_blanked",
        )
        if c in before.columns
    ]
    expected = {col: target[col] for col in provenance_cols}

    session.blank_node("nose", track=0, frame_idx=2)
    blanked = _row(
        _disk_working(session),
        frame_idx=2,
        track=0,
        node="nose",
    )
    assert pd.isna(blanked["x"]) and pd.isna(blanked["y"])
    if "point_score" in blanked.index:
        assert float(blanked["point_score"]) == pytest.approx(0.0)
    if "was_node_blanked" in blanked.index:
        assert int(blanked["was_node_blanked"]) == 1
    # Raw provenance must remain untouched by blank itself.
    if "x_raw" in blanked.index:
        assert float(blanked["x_raw"]) == pytest.approx(float(expected["x_raw"]))

    session.undo_last_manual_edit()
    restored = _row(
        _disk_working(session),
        frame_idx=2,
        track=0,
        node="nose",
    )
    for col, value in expected.items():
        if pd.isna(value):
            assert pd.isna(restored[col])
        else:
            assert restored[col] == pytest.approx(value)


def test_manual_edits_persist_across_session_reload(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    original = _disk_working(session)
    session.swap_identities(1, 2)
    session.blank_node("nose", track=0, frame_idx=3)
    _flush_working(session)
    working_sha = _sha256(session.working_tracked_pose_path)
    records_on_disk = json.loads(
        (s3_dir / MANUAL_CORRECTIONS_FILENAME).read_text(encoding="utf-8")
    )
    assert len(records_on_disk) == 2

    reloaded = load_review_session(s3_dir)
    assert reloaded.manual_edit_count == 2
    assert _sha256(reloaded.working_tracked_pose_path) == working_sha
    assert reloaded.has_automatic_baseline is True

    reloaded.undo_last_manual_edit()
    assert reloaded.manual_edit_count == 1
    assert json.loads(
        (s3_dir / MANUAL_CORRECTIONS_FILENAME).read_text(encoding="utf-8")
    ) == reloaded.manual_edit_stack

    reloaded.reset_manual_edits()
    assert reloaded.manual_edit_count == 0
    assert pd.read_parquet(reloaded.working_tracked_pose_path).equals(original)
    assert (
        json.loads((s3_dir / MANUAL_CORRECTIONS_FILENAME).read_text(encoding="utf-8"))
        == []
    )


def test_superseded_acceptance_ignores_stale_tracked_pose_file(tmp_path: Path) -> None:
    s3_dir, _s2_run, _s1 = _completed_s3(tmp_path)
    session = load_review_session(s3_dir)
    session.accept_tracking()
    assert session.is_currently_accepted is True
    assert (s3_dir / FINAL_TRACKED_POSE_FILENAME).is_file()
    stale_tracked_sha = _sha256(s3_dir / FINAL_TRACKED_POSE_FILENAME)

    session.swap_identities(0, 0)
    assert session.acceptance_state == ACCEPTANCE_SUPERSEDED
    assert session.is_currently_accepted is False
    assert is_currently_accepted(session.acceptance_state) is False
    # Stale accepted artifact may still exist on disk.
    assert (s3_dir / FINAL_TRACKED_POSE_FILENAME).is_file()
    assert _sha256(s3_dir / FINAL_TRACKED_POSE_FILENAME) == stale_tracked_sha

    reopened = load_review_session(s3_dir)
    assert reopened.acceptance_state == ACCEPTANCE_SUPERSEDED
    assert reopened.is_currently_accepted is False
    assert (s3_dir / FINAL_TRACKED_POSE_FILENAME).is_file()
    # Downstream accept path must not treat this as currently accepted.
    result = reopened.accept_tracking()
    assert result.already_accepted is False
    assert reopened.is_currently_accepted is True
    assert _sha256(s3_dir / FINAL_TRACKED_POSE_FILENAME) == _sha256(
        reopened.working_tracked_pose_path
    )
