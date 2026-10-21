"""S5 completed-run discovery and stale-source classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from blob_features import (
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    MISSING_REQUIRED_ARTIFACTS,
    STALE_SOURCE,
    BlobExecutionSettings,
    BlobFeaturesConfig,
    BlobFeaturesRequest,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    find_latest_eligible_s5_run,
    is_s5_run_eligible_for_downstream,
    run_blob_feature_extraction,
    summarize_blob_features_project,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


def _soft_config() -> BlobFeaturesConfig:
    return BlobFeaturesConfig(
        segmentation=BlobSegmentationSettings(
            blob_threshold=-1,
            blob_blur_kernel_size=5,
            blob_kernel_size=5,
            blob_tail_open_kernel_size=3,
            blob_min_area=20,
            blob_max_area=0,
            blob_prune_min_radius=1,
            blob_prune_regrow_iters=1,
            blob_prune_max_regrow_dist=2,
            blob_post_prune_min_area=20,
            blob_post_prune_max_area=0,
            blob_final_smooth_kernel_size=3,
            abs_dark_threshold=250,
        ),
        temporal=BlobTemporalSettings(cc_count_gap_fill_max_gap=2),
        execution=BlobExecutionSettings(n_workers=1, max_in_flight=4),
    ).validated()


def _write_eligible_s1(
    root: Path,
    *,
    frame_count: int = 4,
    fps: float = 20.0,
    size: tuple[int, int] = (64, 48),
) -> Path:
    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    width, height = size
    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    background = np.full((height, width, 3), 220, dtype=np.uint8)
    for _ in range(frame_count):
        frame = background.copy()
        frame[10:30, 8:28] = 5
        frame[10:30, 36:56] = 5
        writer.write(frame)
    writer.release()
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "width": width,
                    "height": height,
                    "opencv_reported_frame_count": frame_count,
                    "opencv_readable_frame_count": frame_count,
                    "frame_count_used_for_sleap": frame_count,
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
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
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
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    return session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_discovery_classifies_completed_and_stale(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T000001",
        )
    )
    assert result.success is True
    assert is_s5_run_eligible_for_downstream(result.run_dir) is True

    summary = summarize_blob_features_project(session)
    assert len(summary.completed_runs) == 1
    assert summary.runs[0].classification in (
        ELIGIBLE_DOWNSTREAM,
        COMPLETED_WITH_WARNINGS,
    )
    assert summary.eligible_runs
    assert find_latest_eligible_s5_run(session) is not None

    # Re-prepare S1 video content → digests change; completed run becomes stale.
    video_path = session / "preprocess" / "prepared_video.mp4"
    original_sha = _sha256(video_path)
    width, height = 64, 48
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height),
    )
    assert writer.isOpened()
    for _ in range(4):
        frame = np.full((height, width, 3), 180, dtype=np.uint8)
        frame[12:28, 10:26] = 8
        writer.write(frame)
    writer.release()
    assert _sha256(video_path) != original_sha

    stale = summarize_blob_features_project(session)
    assert stale.runs[0].classification == STALE_SOURCE
    assert is_s5_run_eligible_for_downstream(result.run_dir) is False
    assert find_latest_eligible_s5_run(session) is None
    # Historical completion status is preserved.
    meta = json.loads((result.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] in {"completed", "completed_with_warnings"}


def test_failed_and_structurally_invalid_runs(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    run_root = session / "blob_features"
    failed = run_root / "blob_features__failed"
    failed.mkdir(parents=True)
    (failed / "run_meta.json").write_text(
        json.dumps({"run_id": "blob_features__failed", "status": "failed"}) + "\n",
        encoding="utf-8",
    )
    incomplete = run_root / "blob_features__incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "blob_features__incomplete",
                "status": "completed",
                "source_s1": {"session_root": str(session)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    broken = run_root / "blob_features__broken"
    broken.mkdir(parents=True)

    summary = summarize_blob_features_project(session)
    by_id = {run.run_id: run for run in summary.runs}
    assert by_id["blob_features__failed"].classification == FAILED_OR_INCOMPLETE
    assert by_id["blob_features__incomplete"].classification == MISSING_REQUIRED_ARTIFACTS
    assert by_id["blob_features__broken"].classification == MISSING_REQUIRED_ARTIFACTS


def test_progress_callback_reports_prepared_frames(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path, frame_count=3)
    events: list[str] = []

    def _progress(event: object) -> None:
        phase = getattr(event, "phase", "")
        events.append(str(phase))

    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T000002",
        ),
        progress_callback=_progress,
    )
    assert result.success is True
    assert "extracting_frames" in events
    assert "completed" in events


def test_s5_available_without_s2_s4(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    assert not (session / "pose_inference").exists()
    assert not (session / "pose_finalization").exists()
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T000003",
        )
    )
    assert result.success is True
    summary = summarize_blob_features_project(session)
    assert summary.s1_eligible is True
    assert summary.eligible_runs


def test_completed_with_warnings_remains_eligible(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T000004",
        )
    )
    assert result.success is True
    meta_path = result.run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "completed_with_warnings"
    meta.setdefault("qc", {})["warning_count"] = 2
    meta["qc"]["warnings"] = ["synthetic warning"]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    summary = summarize_blob_features_project(session)
    assert summary.runs[0].classification == COMPLETED_WITH_WARNINGS
    assert summary.eligible_runs
    latest = find_latest_eligible_s5_run(session)
    assert latest is not None
    assert latest.run_id == result.run_id
    assert is_s5_run_eligible_for_downstream(result.run_dir) is True


def test_background_digest_change_marks_run_stale(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T000005",
        )
    )
    assert result.success is True
    assert is_s5_run_eligible_for_downstream(result.run_dir) is True

    background = session / "preprocess" / "cropped_background.png"
    original_sha = _sha256(background)
    altered = np.full((48, 64, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(background), altered)
    assert _sha256(background) != original_sha

    stale = summarize_blob_features_project(session)
    assert stale.runs[0].classification == STALE_SOURCE
    assert is_s5_run_eligible_for_downstream(result.run_dir) is False
    assert find_latest_eligible_s5_run(session) is None
    # Historical completion status remains unchanged.
    meta = json.loads((result.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] in {"completed", "completed_with_warnings"}
