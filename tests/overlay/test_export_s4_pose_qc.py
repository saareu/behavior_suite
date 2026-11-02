"""C2-B S4 pose QC export: timing, cancel/cleanup, digest, immutability."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.results_packaging._export_fixtures import count_video_frames, write_synthetic_mp4

from behavioral_review.sync import SyncMap
from overlay import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_S4_POSE_RENDERER_VERSION,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    OverlaySourceError,
    export_s4_pose_qc_clip,
    parse_overlay_artifact_manifest,
    plan_s4_prepared_interval,
)
from overlay.contracts import (
    OVERLAY_QC_S4_POSE_VIDEO_FILENAME,
    S4_PROVENANCE_IDENTITY_CORRECTED,
    S4_PROVENANCE_KALMAN_GAP_FILL,
    S4_PROVENANCE_ORIGINAL,
    S4_PROVENANCE_TEMPORAL_RESAMPLE,
)
from overlay.source_s4 import OverlayS4QcSource
from project.file_digest import sha256_file
from results_packaging.media_csv_export.video import probe_video


def _write_s4_pose_for_sync(path: Path, sync: SyncMap) -> Path:
    rows: list[dict[str, object]] = []
    for sample in range(int(sync.n_dense)):
        for track in (0, 1):
            for node_i, node in enumerate(("nose", "neck", "spine_base")):
                codes = (
                    S4_PROVENANCE_ORIGINAL,
                    S4_PROVENANCE_TEMPORAL_RESAMPLE,
                    S4_PROVENANCE_KALMAN_GAP_FILL,
                    S4_PROVENANCE_IDENTITY_CORRECTED,
                )
                prov = codes[(sample + track + node_i) % len(codes)]
                names = {
                    S4_PROVENANCE_ORIGINAL: "original",
                    S4_PROVENANCE_TEMPORAL_RESAMPLE: "temporal_resample",
                    S4_PROVENANCE_KALMAN_GAP_FILL: "kalman_gap_fill",
                    S4_PROVENANCE_IDENTITY_CORRECTED: "identity_corrected",
                }
                rows.append(
                    {
                        "sample_idx": sample,
                        "time_sec": float(sync.time_sec[sample]),
                        "track": track,
                        "node": node,
                        "x": 8.0 + sample + track * 20 + node_i * 3,
                        "y": 10.0 + track * 8 + node_i,
                        "provenance": prov,
                        "provenance_name": names[prov],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _synthetic_s4_source(tmp_path: Path, *, n_prepared: int = 12, fps: float = 10.0) -> OverlayS4QcSource:
    sync_df = build_hand_computed_sync(n_prepared=n_prepared, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    s4_dir = tmp_path / "pose_finalization" / "pose_finalization__synth"
    s4_dir.mkdir(parents=True)
    pose = _write_s4_pose_for_sync(s4_dir / "final_pose.parquet", sync)
    video = write_synthetic_mp4(
        tmp_path / "preprocess" / "prepared_video.mp4",
        n_frames=n_prepared,
        fps=fps,
        width=96,
        height=64,
    )
    probe = probe_video(video)
    pose_sha = sha256_file(pose)
    video_sha = sha256_file(video)
    # prepared_sync placeholder digest (file may be absent in synthetic unit tests).
    sync_path = tmp_path / "preprocess" / "prepared_sync.npz"
    sync_path.write_bytes(b"synthetic-sync-placeholder")
    sync_sha = sha256_file(sync_path)
    return OverlayS4QcSource(
        s4_run_id="pose_finalization__synth",
        s4_run_dir=s4_dir,
        s4_status="completed",
        final_pose_path=pose,
        final_pose_sha256=pose_sha,
        session_root=tmp_path,
        prepared_sync_path=sync_path,
        prepared_sync_sha256=sync_sha,
        prepared_video_path=video,
        prepared_video_sha256=video_sha,
        video_probe=probe,
        sync=sync,
        sync_build_audit={"n_dense": int(sync.n_dense), "synthetic": True},
        source_s3_run_id="tracking_correction__synth",
        source_s3_run_dir=tmp_path / "tracking_correction" / "tracking_correction__synth",
        source_s3_tracked_pose_sha256="s3sha",
        skeleton_edges=(("nose", "neck"), ("neck", "spine_base")),
        skeleton_source="synthetic",
        settings_used_path=None,
        run_meta={"run_id": "pose_finalization__synth", "status": "completed"},
        source_digests={
            "s4_final_pose_sha256": pose_sha,
            "s1_prepared_sync_sha256": sync_sha,
            "s1_prepared_video_sha256": video_sha,
            "s3_tracked_pose_sha256": "s3sha",
        },
        provenance={"source_kind": "s4_finalized_pose", "synthetic": True},
    )


def test_s4_export_frame_count_clip_boundaries_and_manifest(tmp_path: Path) -> None:
    source = _synthetic_s4_source(tmp_path)
    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=4,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    assert plan.prepared_interval.frame_count == 4
    assert LayerId.S4_POSE_FINAL in plan.selected_layers

    with patch("overlay.export.verify_overlay_s4_qc_source_integrity"):
        result = export_s4_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )

    assert result.frames_written == 4
    assert result.qc_status == "completed"
    assert result.video_path.name == OVERLAY_QC_S4_POSE_VIDEO_FILENAME
    assert result.video_path.is_file()
    assert (result.job_dir / "run_meta.json").is_file()
    assert (result.job_dir / "processing_log.txt").is_file()
    assert (result.job_dir / "settings_used.yaml").is_file()
    fps, n = count_video_frames(result.video_path)
    assert n == 4
    assert abs(fps - source.video_probe.fps) / source.video_probe.fps <= 0.02

    manifest = parse_overlay_artifact_manifest(
        json.loads(result.manifest_path.read_text(encoding="utf-8"))
    )
    assert manifest.schema_version == OVERLAY_ARTIFACT_MANIFEST_SCHEMA
    assert manifest.renderer_version == OVERLAY_S4_POSE_RENDERER_VERSION
    assert manifest.plateau_policy == "last_in_plateau"
    assert manifest.output_frame_domain == "prepared"
    assert manifest.extra["export_kind"] == "s4_pose_qc_clip"
    assert manifest.extra["qc_status"] == "completed"
    assert manifest.extra["output"]["frame_count"] == 4
    assert "s4.pose_final" in manifest.layer_selections
    assert "1" in manifest.extra["s4_provenance_visual_map"]
    assert manifest.extra["s4_provenance_visual_map"]["3"]["visual_class"] == (
        "temporal_resample"
    )
    assert manifest.extra["source"]["final_pose_sha256"] == source.final_pose_sha256


def test_s4_export_plateau_last_in_policy_notes(tmp_path: Path) -> None:
    source = _synthetic_s4_source(tmp_path, n_prepared=6)
    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with patch("overlay.export.verify_overlay_s4_qc_source_integrity"):
        result = export_s4_pose_qc_clip(
            source=source, plan=plan, output_root=tmp_path / "out"
        )
    # plateau_repeats=2 → prepared 0 maps to dense {0,1} → last = 1
    assert any("dense_sample_idx=1" in n for n in result.notes)
    first = result.manifest.extra["first_frame_timing"]
    assert first["plateau_policy"] == "last_in_plateau"
    assert first["dense_sample_idx"] == 1


def test_s4_cancellation_cleans_staging(tmp_path: Path) -> None:
    source = _synthetic_s4_source(tmp_path, n_prepared=20)
    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=15,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    job = tmp_path / "results_exports" / "overlay_qc_s4_pose__cancel"
    with (
        patch("overlay.export.verify_overlay_s4_qc_source_integrity"),
        pytest.raises(OverlayExportCancelled),
    ):
        export_s4_pose_qc_clip(
            source=source,
            plan=plan,
            job_dir=job,
            cancellation_requested=cancel,
        )
    staging = job / "_overlay_qc_staging"
    assert not staging.exists()
    assert not (job / OVERLAY_QC_S4_POSE_VIDEO_FILENAME).exists()
    assert not (job / "overlay_artifact_manifest.json").exists()


def test_s4_source_digest_mismatch(tmp_path: Path) -> None:
    source = _synthetic_s4_source(tmp_path)
    bad = OverlayS4QcSource(
        s4_run_id=source.s4_run_id,
        s4_run_dir=source.s4_run_dir,
        s4_status=source.s4_status,
        final_pose_path=source.final_pose_path,
        final_pose_sha256="deadbeef",
        session_root=source.session_root,
        prepared_sync_path=source.prepared_sync_path,
        prepared_sync_sha256=source.prepared_sync_sha256,
        prepared_video_path=source.prepared_video_path,
        prepared_video_sha256=source.prepared_video_sha256,
        video_probe=source.video_probe,
        sync=source.sync,
        sync_build_audit=source.sync_build_audit,
        source_s3_run_id=source.source_s3_run_id,
        source_s3_run_dir=source.source_s3_run_dir,
        source_s3_tracked_pose_sha256=source.source_s3_tracked_pose_sha256,
        skeleton_edges=source.skeleton_edges,
        skeleton_source=source.skeleton_source,
        settings_used_path=source.settings_used_path,
        run_meta=source.run_meta,
        source_digests={
            **source.source_digests,
            "s4_final_pose_sha256": "deadbeef",
        },
        provenance=source.provenance,
    )
    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=1,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with (
        patch(
            "overlay.source_s4.validate_s4_run_for_downstream",
            return_value={
                "s4_run_dir": source.s4_run_dir,
                "status": "completed",
                "source_s3_run_dir": source.source_s3_run_dir,
                "source_s3_run_id": source.source_s3_run_id,
                "tracked_pose_sha256": "s3sha",
            },
        ),
        pytest.raises(OverlaySourceError, match="digest mismatch"),
    ):
        from overlay.source_s4 import verify_overlay_s4_qc_source_integrity

        verify_overlay_s4_qc_source_integrity(bad)

    with (
        patch("overlay.export.verify_overlay_s4_qc_source_integrity") as verify,
        pytest.raises(OverlaySourceError, match="digest mismatch"),
    ):
        # First verify ok, second (pre-publish) fails.
        verify.side_effect = [None, OverlaySourceError("digest mismatch for s4")]
        export_s4_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "exports",
            verify_encoded_boundaries=False,
        )


def test_s4_source_immutability_during_export(tmp_path: Path) -> None:
    source = _synthetic_s4_source(tmp_path)
    before_pose = sha256_file(source.final_pose_path)
    before_video = sha256_file(source.prepared_video_path)
    before_df = pd.read_parquet(source.final_pose_path).copy()
    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=3,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with patch("overlay.export.verify_overlay_s4_qc_source_integrity"):
        export_s4_pose_qc_clip(
            source=source, plan=plan, output_root=tmp_path / "exports"
        )
    assert sha256_file(source.final_pose_path) == before_pose
    assert sha256_file(source.prepared_video_path) == before_video
    after_df = pd.read_parquet(source.final_pose_path)
    pd.testing.assert_frame_equal(before_df, after_df)


def test_s4_no_overwrite_previous(tmp_path: Path) -> None:
    source = _synthetic_s4_source(tmp_path)
    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    job = tmp_path / "results_exports" / "overlay_qc_s4_pose__fixed"
    job.mkdir(parents=True)
    with patch("overlay.export.verify_overlay_s4_qc_source_integrity"):
        first = export_s4_pose_qc_clip(source=source, plan=plan, job_dir=job)
    assert first.video_path.is_file()
    with (
        patch("overlay.export.verify_overlay_s4_qc_source_integrity"),
        pytest.raises(OverlayExportError, match="overwrite"),
    ):
        export_s4_pose_qc_clip(source=source, plan=plan, job_dir=job)


def test_resolve_rejects_missing_s4_run(tmp_path: Path) -> None:
    from overlay import resolve_overlay_s4_qc_source

    with pytest.raises(OverlaySourceError, match="does not exist"):
        resolve_overlay_s4_qc_source(s4_run_dir=tmp_path / "missing_s4")
