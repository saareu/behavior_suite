"""S5 cooperative cancellation, publication gate, and eligibility tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from blob_features import (
    FAILED_OR_INCOMPLETE,
    STATUS_CANCELED,
    BlobExecutionSettings,
    BlobFeaturesCancelledError,
    BlobFeaturesConfig,
    BlobFeaturesError,
    BlobFeaturesRequest,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    is_s5_run_eligible_for_downstream,
    run_blob_feature_extraction,
    summarize_blob_features_project,
    validate_s5_run_for_downstream,
)
from blob_features.contracts import (
    BLOB_FEATURES_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


def _soft_config(**overrides: object) -> BlobFeaturesConfig:
    seg_values: dict[str, Any] = {
        "blob_threshold": -1,
        "blob_blur_kernel_size": 5,
        "blob_kernel_size": 5,
        "blob_tail_open_kernel_size": 3,
        "blob_min_area": 40,
        "blob_max_area": 0,
        "blob_prune_min_radius": 1,
        "blob_prune_regrow_iters": 1,
        "blob_prune_max_regrow_dist": 2,
        "blob_post_prune_min_area": 40,
        "blob_post_prune_max_area": 0,
        "blob_final_smooth_kernel_size": 3,
        "abs_dark_threshold": 250,
    }
    temporal_values: dict[str, Any] = {"cc_count_gap_fill_max_gap": 2}
    for key, value in overrides.items():
        if key in seg_values:
            seg_values[key] = value
        elif key in temporal_values:
            temporal_values[key] = value
    return BlobFeaturesConfig(
        segmentation=BlobSegmentationSettings(**seg_values).validated(),
        temporal=BlobTemporalSettings(**temporal_values).validated(),
        execution=BlobExecutionSettings(n_workers=1, max_in_flight=4).validated(),
    ).validated()


def _write_synthetic_s1(
    root: Path,
    *,
    frame_count: int = 8,
    fps: float = 20.0,
    size: tuple[int, int] = (120, 80),
) -> Path:
    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    width, height = size

    background = np.full((height, width, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)

    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame_idx in range(frame_count):
        frame = background.copy()
        if frame_idx < frame_count // 2:
            frame[20:45, 15:40] = 5
            frame[20:45, 75:100] = 5
        else:
            cv2.ellipse(frame, (60, 40), (28, 16), 0, 0, 360, (5, 5, 5), -1)
        writer.write(frame)
    writer.release()

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


class _CancelOnNthCheck:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def _run_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))


def test_no_callback_preserves_successful_output(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T100000",
        )
    )
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert result.blob_features_path is not None
    assert result.blob_features_path.is_file()


def test_false_callback_preserves_successful_output(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T100001",
        ),
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert result.blob_features_path is not None


@pytest.mark.parametrize(
    ("nth", "label"),
    [
        (1, "before_source_open"),
        (2, "before_video_capture"),
        (3, "after_source_prep"),
        (4, "during_first_frame"),
        (5, "during_second_frame"),
    ],
)
def test_cancel_at_safe_checkpoints(
    tmp_path: Path, nth: int, label: str
) -> None:
    del label
    session = _write_synthetic_s1(tmp_path, frame_count=6)
    cancel = _CancelOnNthCheck(nth)
    with pytest.raises(BlobFeaturesCancelledError):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_soft_config(),
                run_purpose="test",
                timestamp=f"20260101T11{nth:04d}",
            ),
            cancellation_requested=cancel,
        )
    runs = list((session / "blob_features").iterdir())
    assert len(runs) == 1
    meta = _run_meta(runs[0])
    assert meta["status"] == STATUS_CANCELED
    assert not (runs[0] / BLOB_FEATURES_FILENAME).exists()
    assert not (runs[0] / QC_FILENAME).exists()
    assert (runs[0] / RUN_META_FILENAME).is_file()
    assert "cancelled" in (meta.get("error") or "").lower() or "cancel" in (
        meta.get("error") or ""
    ).lower()


def test_cancel_after_extraction_before_postprocess(tmp_path: Path) -> None:
    """Checkpoint 4: after the frame loop, before temporal/QC."""

    session = _write_synthetic_s1(tmp_path, frame_count=3)
    # checks: 1 start, 2 before open, 3 after open, 4..6 per frame, 7 after extract
    cancel = _CancelOnNthCheck(7)
    with pytest.raises(BlobFeaturesCancelledError):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_soft_config(),
                run_purpose="test",
                timestamp="20260101T120000",
            ),
            cancellation_requested=cancel,
        )
    run_dir = next((session / "blob_features").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_CANCELED


def test_cancel_at_final_publication_gate(tmp_path: Path) -> None:
    """Checkpoint 5: after revalidation, immediately before publication."""

    session = _write_synthetic_s1(tmp_path, frame_count=3)
    # 1 start + 2 open + 3 prep + 3 frames + 7 after extract + 8 publish gate
    cancel = _CancelOnNthCheck(8)
    with pytest.raises(BlobFeaturesCancelledError):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_soft_config(),
                run_purpose="test",
                timestamp="20260101T120100",
            ),
            cancellation_requested=cancel,
        )
    run_dir = next((session / "blob_features").iterdir())
    meta = _run_meta(run_dir)
    assert meta["status"] == STATUS_CANCELED
    assert not (run_dir / BLOB_FEATURES_FILENAME).exists()


def test_cancel_after_publication_gate_completes(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=3)
    state = {"in_publication": False}

    def cancel() -> bool:
        return state["in_publication"]

    def publication_started() -> None:
        state["in_publication"] = True

    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T120200",
        ),
        cancellation_requested=cancel,
        publication_started=publication_started,
    )
    assert state["in_publication"] is True
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert result.blob_features_path is not None
    assert result.blob_features_path.is_file()


def test_canceled_run_rejected_downstream(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4)
    with pytest.raises(BlobFeaturesCancelledError):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_soft_config(),
                run_purpose="test",
                timestamp="20260101T130000",
            ),
            cancellation_requested=lambda: True,
        )
    run_dir = next((session / "blob_features").iterdir())
    assert is_s5_run_eligible_for_downstream(run_dir) is False
    with pytest.raises(BlobFeaturesError, match="not completed"):
        validate_s5_run_for_downstream(run_dir)
    summary = summarize_blob_features_project(session)
    assert summary.runs
    assert summary.runs[0].classification == FAILED_OR_INCOMPLETE
    assert summary.eligible_runs == ()


def test_canceled_run_does_not_affect_prior_completed(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4)
    completed = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T140000",
        )
    )
    with pytest.raises(BlobFeaturesCancelledError):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_soft_config(),
                run_purpose="test",
                timestamp="20260101T140100",
            ),
            cancellation_requested=lambda: True,
        )
    assert completed.blob_features_path is not None
    assert completed.blob_features_path.is_file()
    assert is_s5_run_eligible_for_downstream(completed.run_dir) is True
    summary = summarize_blob_features_project(session)
    assert len(summary.completed_runs) == 1
    assert summary.eligible_runs
    assert summary.eligible_runs[0].run_id == completed.run_id


def test_ordinary_failure_remains_failed_even_if_cancel_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=3)
    state = {"cancel": False}

    def boom(*_args: object, **_kwargs: object) -> None:
        state["cancel"] = True
        raise BlobFeaturesError("injected structural failure")

    monkeypatch.setattr(
        "blob_features.runner.assert_structural_qc",
        boom,
    )
    with pytest.raises(BlobFeaturesError, match="injected structural failure"):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_soft_config(),
                run_purpose="test",
                timestamp="20260101T150000",
            ),
            cancellation_requested=lambda: state["cancel"],
        )
    assert state["cancel"] is True
    run_dir = next((session / "blob_features").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_FAILED


def test_progress_throttling_independent_of_cancel_polling(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=5)
    cancel_calls = {"n": 0}
    progress_events: list[str] = []

    def cancel() -> bool:
        cancel_calls["n"] += 1
        return False

    def progress(event: object) -> None:
        progress_events.append(getattr(event, "phase", ""))

    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_config(),
            run_purpose="test",
            timestamp="20260101T160000",
        ),
        progress_callback=progress,
        cancellation_requested=cancel,
    )
    assert result.success is True
    # One cancel poll before open + after open + per frame + after extract + gate.
    assert cancel_calls["n"] >= 5 + 2
    assert progress_events.count("extracting_frames") <= 5
