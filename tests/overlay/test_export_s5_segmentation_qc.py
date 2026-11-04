"""C3-3 S5 segmentation QC export: eligibility, timing, cancel, digests, replay."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from tests.results_packaging._export_fixtures import count_video_frames

from blob_features import (
    BlobFeaturesConfig,
    BlobFeaturesRequest,
    BlobSegmentationSettings,
    run_blob_feature_extraction,
    segment_frame,
)
from blob_features.config import BlobExecutionSettings, BlobTemporalSettings
from overlay import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_S5_SEGMENTATION_RENDERER_VERSION,
    LayerId,
    OverlayExportCancelled,
    OverlaySourceError,
    export_s5_segmentation_qc_clip,
    parse_overlay_artifact_manifest,
    plan_s5_prepared_interval,
    resolve_overlay_s5_qc_source,
)
from overlay.adapters.segmentation import (
    PreparedFrameReplayCache,
    make_s5_segmentation_adapter,
)
from overlay.contracts import (
    OVERLAY_QC_S5_SEGMENTATION_JOB_PREFIX,
    OVERLAY_QC_S5_SEGMENTATION_VIDEO_FILENAME,
    PlateauPolicy,
)
from overlay.timing import PreparedFrameTiming
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from project.file_digest import sha256_file


def _preview_settings(**overrides: int) -> BlobSegmentationSettings:
    values = {
        "blob_threshold": -1,
        "blob_blur_kernel_size": 5,
        "blob_kernel_size": 5,
        "blob_tail_open_kernel_size": 3,
        "blob_min_area": 20,
        "blob_max_area": 0,
        "blob_prune_min_radius": 1,
        "blob_prune_regrow_iters": 1,
        "blob_prune_max_regrow_dist": 2,
        "blob_post_prune_min_area": 20,
        "blob_post_prune_max_area": 0,
        "blob_final_smooth_kernel_size": 3,
        "abs_dark_threshold": 250,
    }
    values.update(overrides)
    return BlobSegmentationSettings(**values).validated()


def _preview_config() -> BlobFeaturesConfig:
    return BlobFeaturesConfig(
        segmentation=_preview_settings(),
        temporal=BlobTemporalSettings(cc_count_gap_fill_max_gap=0),
        execution=BlobExecutionSettings(n_workers=1, max_in_flight=1),
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


def _completed_s5_run(tmp_path: Path, *, frame_count: int = 8):
    session = _write_synthetic_s1(tmp_path, frame_count=frame_count, fps=20.0)
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_preview_config(),
            timestamp="20261104T180000",
            run_purpose="overlay_qc_test",
        )
    )
    assert result.success
    return session, result.run_dir


def test_s5_resolve_eligible_and_pins_replay_dependencies(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    assert source.s5_run_id == run_dir.name
    assert source.settings_fingerprint
    assert source.cropped_background_path.is_file()
    assert source.prepared_video_path.is_file()
    assert source.sync_build_audit["dense_120hz"] is False
    disclosure = source.replay_disclosure()
    assert disclosure["no_contours_from_parquet_scalars"] is True
    assert "segment_frame" in source.provenance["segmentation_implementation_id"]
    runtime = disclosure["runtime"]
    assert runtime["historical_environment_equivalence_claimed"] is False
    assert runtime["historical_environment_recorded"] is False
    assert runtime["multi_otsu_required"] is False  # fixture uses Otsu (-1)
    assert runtime["opencv_version"]
    assert runtime["numpy_version"]


def test_s5_rejects_incomplete_and_missing_settings(tmp_path: Path) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4)
    incomplete = session / "blob_features" / "blob_features__incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "run_meta.json").write_text(
        json.dumps({"status": "running", "source_s1": {"session_root": str(session)}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OverlaySourceError, match="not eligible|not completed"):
        resolve_overlay_s5_qc_source(s5_run_dir=incomplete)

    _session, run_dir = _completed_s5_run(tmp_path / "ok", frame_count=4)
    (run_dir / "settings_used.yaml").unlink()
    with pytest.raises(OverlaySourceError, match="settings_used|not eligible"):
        resolve_overlay_s5_qc_source(s5_run_dir=run_dir)


def test_s5_rejects_settings_fingerprint_mismatch(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=4)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    adapter = make_s5_segmentation_adapter(
        prepared_video_path=source.prepared_video_path,
        cropped_background_path=source.cropped_background_path,
        segmentation_settings=source.segmentation_settings,
        settings_fingerprint="deadbeef",
        expected_settings_fingerprint=source.settings_fingerprint,
    )
    assert not adapter.available
    timing = PreparedFrameTiming(
        prepared_frame_idx=0,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=0,
        plateau_sample_indices=(0,),
        mapping=None,
    )
    obs = adapter.observe(timing, prepared_frame_idx=0)
    assert not obs.available
    assert "fingerprint" in str(obs.reason).lower()


def test_s5_rejects_missing_background(tmp_path: Path) -> None:
    session, run_dir = _completed_s5_run(tmp_path, frame_count=4)
    bg = session / "preprocess" / "cropped_background.png"
    bg.unlink()
    with pytest.raises(OverlaySourceError, match="not eligible|background|Missing"):
        resolve_overlay_s5_qc_source(s5_run_dir=run_dir)


def test_s5_rejects_missing_skimage_for_multi_otsu(tmp_path: Path, monkeypatch) -> None:
    session = _write_synthetic_s1(tmp_path, frame_count=4)
    config = BlobFeaturesConfig(
        segmentation=_preview_settings(blob_threshold=-2),
        temporal=BlobTemporalSettings(cc_count_gap_fill_max_gap=0),
        execution=BlobExecutionSettings(n_workers=1, max_in_flight=1),
    ).validated()
    # Build a completed Multi-Otsu run only if skimage is present; then simulate
    # a missing dependency for the resolve-time hard fail.
    result = run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=config,
            timestamp="20261104T180001",
            run_purpose="overlay_qc_multiotsu",
        )
    )
    assert result.success
    monkeypatch.setattr(
        "overlay.source_s5._probe_skimage",
        lambda: (None, False, "ImportError: simulated"),
    )
    with pytest.raises(OverlaySourceError, match="Multi-Otsu|scikit-image"):
        resolve_overlay_s5_qc_source(s5_run_dir=result.run_dir)


def test_s5_export_frame_count_manifest_and_immutability(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=8)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    plan = plan_s5_prepared_interval(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=4,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        show_contours=True,
        show_masks=True,
        show_scalar_text=True,
        mask_opacity=0.4,
    )
    assert plan.prepared_interval.frame_count == 4
    assert LayerId.S5_SEGMENTATION in plan.selected_layers

    before_parquet = sha256_file(source.blob_features_path)
    before_settings = sha256_file(source.settings_used_path)
    before_bg = sha256_file(source.cropped_background_path)
    result = export_s5_segmentation_qc_clip(
        source=source,
        plan=plan,
        output_root=tmp_path / "results_exports",
    )
    assert sha256_file(source.blob_features_path) == before_parquet
    assert sha256_file(source.settings_used_path) == before_settings
    assert sha256_file(source.cropped_background_path) == before_bg
    assert result.frames_written == 4
    assert result.qc_status == "completed"
    assert result.video_path.name == OVERLAY_QC_S5_SEGMENTATION_VIDEO_FILENAME
    assert result.job_dir.name.startswith(OVERLAY_QC_S5_SEGMENTATION_JOB_PREFIX)
    assert (result.job_dir / "run_meta.json").is_file()
    assert (result.job_dir / "settings_used.yaml").is_file()
    assert (result.job_dir / "processing_log.txt").is_file()
    fps, n = count_video_frames(result.video_path)
    assert n == 4
    assert abs(fps - source.video_probe.fps) / source.video_probe.fps <= 0.02

    manifest = parse_overlay_artifact_manifest(
        json.loads(result.manifest_path.read_text(encoding="utf-8"))
    )
    assert manifest.schema_version == OVERLAY_ARTIFACT_MANIFEST_SCHEMA
    assert manifest.renderer_version == OVERLAY_S5_SEGMENTATION_RENDERER_VERSION
    assert manifest.extra["export_kind"] == "s5_segmentation_qc_clip"
    assert manifest.extra["replay"]["no_invented_blob_identities"] is True
    assert manifest.extra["source"]["settings_fingerprint"] == source.settings_fingerprint
    assert "s5.segmentation" in manifest.layer_selections
    assert manifest.extra["plan"]["show_masks"] is True
    assert manifest.extra["plan"]["mask_opacity"] == pytest.approx(0.4)


def test_s5_cancel_cleans_unpublished_job(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=8)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    plan = plan_s5_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=5,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(OverlayExportCancelled):
        export_s5_segmentation_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "out",
            cancellation_requested=cancel,
        )
    exports = list((tmp_path / "out").glob("overlay_qc_s5_segmentation__*"))
    assert exports == []


def test_s5_contour_equivalence_against_segment_frame(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=6)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    cache = PreparedFrameReplayCache()
    adapter = make_s5_segmentation_adapter(
        prepared_video_path=source.prepared_video_path,
        cropped_background_path=source.cropped_background_path,
        segmentation_settings=source.segmentation_settings,
        settings_fingerprint=source.settings_fingerprint,
        expected_settings_fingerprint=source.settings_fingerprint,
        frame_cache=cache,
    )
    from blob_features.preview import load_cropped_background, read_prepared_video_frame

    prep = 1
    frame_bgr = read_prepared_video_frame(source.prepared_video_path, prep)
    background = load_cropped_background(source.cropped_background_path)
    expected = segment_frame(frame_bgr, background, source.segmentation_settings)
    cache.set(prep, frame_bgr)
    timing = PreparedFrameTiming(
        prepared_frame_idx=prep,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=prep,
        plateau_sample_indices=(prep,),
        mapping=None,
    )
    obs = adapter.observe(timing, prepared_frame_idx=prep)
    assert obs.available
    assert "track" not in obs.payload
    assert obs.payload.get("no_invented_blob_identities") is True
    contours = obs.payload["contours"]
    assert len(contours) == len(expected.contours)
    for left, right in zip(contours, expected.contours, strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert np.array_equal(np.asarray(obs.payload["mask"]), expected.mask)


def test_s5_prepared_identity_timing_not_dense(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=5)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    assert source.sync_build_audit["kind"] == "s1_prepared_frame_identity"
    assert source.sync_build_audit["dense_120hz"] is False
    # Identity map: sample_idx == prepared_frame_idx for every prepared frame.
    table = source.sync.sync_index
    assert np.array_equal(
        table["sample_idx"].to_numpy(),
        table["prepared_frame_idx"].to_numpy(),
    )


def test_s5_source_digest_mismatch_on_revalidate(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=4)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    # Corrupt parquet after pin.
    pd.DataFrame({"x": [1]}).to_parquet(source.blob_features_path, index=False)
    from overlay.source_s5 import verify_overlay_s5_qc_source_integrity

    with pytest.raises(OverlaySourceError, match="digest mismatch|eligibility"):
        verify_overlay_s5_qc_source_integrity(source)


def test_s5_refuse_overwrite_existing_job(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=4)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    plan = plan_s5_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=1,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    job = tmp_path / "results_exports" / "overlay_qc_s5_segmentation__fixed"
    job.mkdir(parents=True)
    (job / OVERLAY_QC_S5_SEGMENTATION_VIDEO_FILENAME).write_bytes(b"existing")
    from overlay.contracts import OverlayExportError

    with pytest.raises(OverlayExportError, match="Refusing to overwrite"):
        export_s5_segmentation_qc_clip(
            source=source,
            plan=plan,
            job_dir=job,
        )


def test_s5_settings_sidecar_records_fingerprint(tmp_path: Path) -> None:
    _session, run_dir = _completed_s5_run(tmp_path, frame_count=4)
    source = resolve_overlay_s5_qc_source(s5_run_dir=run_dir)
    plan = plan_s5_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=1,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        show_contours=True,
        show_masks=False,
    )
    result = export_s5_segmentation_qc_clip(
        source=source,
        plan=plan,
        output_root=tmp_path / "out_settings",
    )
    payload = yaml.safe_load(
        (result.job_dir / "settings_used.yaml").read_text(encoding="utf-8")
    )
    assert payload["settings_fingerprint"] == source.settings_fingerprint
    assert payload["plan"]["show_contours"] is True
