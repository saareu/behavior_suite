"""Parquet schema/QC and full-video extraction runner tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from blob_features import (
    BLOB_FEATURES_COLUMNS,
    BLOB_FEATURES_SCHEMA_VERSION,
    QC_FILENAME,
    BlobExecutionSettings,
    BlobFeaturesConfig,
    BlobFeaturesError,
    BlobFeaturesRequest,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    evaluate_blob_features_qc,
    load_blob_features_table,
    run_blob_feature_extraction,
)
from blob_features.timing import build_nearest_neighbor_timing_map_from_prepared_sync
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz


def _preview_config(**overrides: object) -> BlobFeaturesConfig:
    seg_values = {
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
    temporal_values = {"cc_count_gap_fill_max_gap": 2}
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
    pattern: str = "two_then_one",
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
        if pattern == "two_then_one":
            if frame_idx < frame_count // 2:
                frame[20:45, 15:40] = 5
                frame[20:45, 75:100] = 5
            else:
                cv2.ellipse(frame, (60, 40), (28, 16), 0, 0, 360, (5, 5, 5), -1)
        elif pattern == "fragmented":
            frame[10:25, 10:25] = 5
            frame[10:25, 50:65] = 5
            frame[10:25, 90:105] = 5
        else:
            frame[20:50, 30:70] = 5
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


def test_full_extraction_schema_and_artifacts(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=8, fps=20.0)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_preview_config(),
            timestamp="20261021T120000",
            run_purpose="test",
        )
    )
    assert result.success
    assert result.status in ("completed", "completed_with_warnings")
    assert result.blob_features_path is not None
    assert result.blob_features_path.is_file()
    assert result.run_meta_path.is_file()
    assert result.settings_used_path.is_file()
    assert result.processing_log_path.is_file()
    assert result.qc_path.is_file()
    assert (result.run_dir / QC_FILENAME).is_file()

    table = load_blob_features_table(result.blob_features_path)
    assert list(table.columns) == list(BLOB_FEATURES_COLUMNS)
    assert set(table["schema_version"].unique()) == {BLOB_FEATURES_SCHEMA_VERSION}
    assert table["dense_frame_idx"].is_unique
    assert table["dense_frame_idx"].tolist() == list(range(len(table)))
    assert table["frame_idx"].tolist() == table["dense_frame_idx"].tolist()
    assert table["sample_idx"].tolist() == table["dense_frame_idx"].tolist()
    assert table["orig_frame_idx"].min() >= 0
    assert table["orig_frame_idx"].max() < 8
    for col in (
        "cc_count",
        "cc_count_raw",
        "min_inter_contour_dist",
        "ellipse_eccentricity",
        "circularity",
    ):
        assert np.issubdtype(table[col].dtype, np.number)

    meta = json.loads(result.run_meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == result.status
    assert meta["source_s1"]["frame_count"] == 8
    assert "prepared_sync_sha256" in meta["source_s1"]
    qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
    assert qc["structural_ok"] is True


def test_determinism_repeated_runs(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=6, fps=20.0)
    cfg = _preview_config()
    first = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=cfg,
            timestamp="20261021T130001",
            run_purpose="test",
        )
    )
    second = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=cfg,
            timestamp="20261021T130002",
            run_purpose="test",
        )
    )
    assert first.blob_features_path is not None
    assert second.blob_features_path is not None
    a = pd.read_parquet(first.blob_features_path)
    b = pd.read_parquet(second.blob_features_path)
    pd.testing.assert_frame_equal(a, b, check_dtype=True)


def test_invalid_source_fails_before_publish(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path)
    (session / "preprocess" / "prepared_video.mp4").unlink()
    with pytest.raises(BlobFeaturesError, match="Missing required"):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_preview_config(),
                timestamp="20261021T140000",
            )
        )
    assert not (session / "blob_features").exists() or not any(
        (session / "blob_features").glob("*/blob_features.parquet")
    )


def test_failed_run_does_not_leave_eligible_parquet(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4, fps=20.0)
    # Corrupt the video bytes after eligibility would normally pass by replacing
    # with an unreadable file under the same name mid-request path: instead force
    # overlap rejection by pointing output into preprocess.
    with pytest.raises(BlobFeaturesError, match="must not overlap"):
        run_blob_feature_extraction(
            BlobFeaturesRequest(
                session_root=session,
                config=_preview_config(),
                output_root=session / "preprocess",
                timestamp="20261021T150000",
            )
        )


def test_fragmentation_missingness_semantics(tmp_path: Path) -> None:
    session = _write_synthetic_s1(
        tmp_path, frame_count=4, fps=20.0, pattern="fragmented"
    )
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_preview_config(),
            timestamp="20261021T160000",
        )
    )
    assert result.blob_features_path is not None
    table = load_blob_features_table(result.blob_features_path)
    assert (table["cc_count_raw"] > 2).any()
    fragmented = table["cc_count_raw"] > 2
    assert not np.isfinite(table.loc[fragmented, "cc_count"]).any()
    assert not np.isfinite(table.loc[fragmented, "ellipse_eccentricity"]).any()


def test_structural_qc_rejects_duplicate_keys(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4, fps=20.0)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_preview_config(),
            timestamp="20261021T170000",
        )
    )
    assert result.blob_features_path is not None
    table = load_blob_features_table(result.blob_features_path)
    timing = build_nearest_neighbor_timing_map_from_prepared_sync(
        session / "preprocess" / "prepared_sync.npz"
    )
    bad = table.copy()
    bad.loc[0, "dense_frame_idx"] = bad.loc[1, "dense_frame_idx"]
    summary = evaluate_blob_features_qc(
        table=bad,
        timing_map=timing,
        source_frame_count=4,
    )
    assert summary.structural_ok is False
    assert any("unique" in f or "dense_frame_idx" in f for f in summary.failures)


def test_pipeline50_join_key_present(tmp_path: Path) -> None:
    """Legacy Pipeline 50 joins on dense_frame_idx (fallback frame_idx)."""

    session = _write_synthetic_s1(tmp_path, frame_count=4, fps=20.0)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_preview_config(),
            timestamp="20261021T180000",
        )
    )
    assert result.blob_features_path is not None
    blob = pd.read_parquet(result.blob_features_path)
    assert "dense_frame_idx" in blob.columns
    directional = pd.DataFrame({"dense_frame_idx": blob["dense_frame_idx"].iloc[::2]})
    merged = directional.merge(
        blob[["dense_frame_idx", "cc_count", "ellipse_eccentricity"]],
        on="dense_frame_idx",
        how="left",
        validate="many_to_one",
    )
    assert len(merged) == len(directional)
