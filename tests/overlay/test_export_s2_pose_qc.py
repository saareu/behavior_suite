"""C3-1 S2 pose QC export: eligibility, timing, cancel, digests, immutability."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from tests.results_packaging._export_fixtures import count_video_frames, write_synthetic_mp4

from behavioral_review.sync import SyncMap
from overlay import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_S2_POSE_RENDERER_VERSION,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    OverlaySourceError,
    export_s2_pose_qc_clip,
    parse_overlay_artifact_manifest,
    plan_s2_prepared_interval,
)
from overlay.contracts import OVERLAY_QC_S2_POSE_VIDEO_FILENAME, S2_SCORE_VISUAL_MAPPING_ID
from overlay.source_s2 import OverlayS2QcSource, resolve_overlay_s2_qc_source
from project.file_digest import sha256_file
from results_packaging.contracts import SYNC_SCHEMA_VERSION
from results_packaging.media_csv_export.video import probe_video


def _prepared_identity_sync(*, n_prepared: int = 12, fps: float = 10.0) -> SyncMap:
    idx = np.arange(n_prepared, dtype=np.int64)
    times = idx.astype(np.float64) / float(fps)
    table = pd.DataFrame(
        {
            "schema_version": SYNC_SCHEMA_VERSION,
            "sample_idx": idx,
            "dense_frame_idx": idx,
            "time_sec": times,
            "prepared_frame_idx": idx,
            "prepared_time_sec": times,
            "is_exact_source_sample": np.ones(n_prepared, dtype=bool),
            "timing_residual_sec": np.zeros(n_prepared, dtype=np.float64),
        }
    )
    return SyncMap.from_sync_index(table, fps=fps)


def _write_s2_pose(
    path: Path,
    *,
    n_prepared: int = 12,
    include_missing_score: bool = True,
) -> Path:
    rows: list[dict[str, object]] = []
    for prep in range(n_prepared):
        for track_label, track_i in (("track_0", 0), ("track_1", 1)):
            for node_i, node in enumerate(("nose", "neck", "spine_base")):
                score: float | None
                if include_missing_score and prep == 0 and track_i == 0 and node == "nose":
                    score = None
                elif prep % 5 == 0 and node == "neck":
                    score = 0.0
                else:
                    score = 0.2 + 0.15 * node_i + 0.1 * track_i
                rows.append(
                    {
                        "frame_idx": prep,
                        "prepared_frame_idx": prep,
                        "instance_index": track_i,
                        "track": track_label,
                        "node": node,
                        "x": 8.0 + prep + track_i * 20 + node_i * 3,
                        "y": 10.0 + track_i * 8 + node_i,
                        "score": score,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _synthetic_s2_source(
    tmp_path: Path, *, n_prepared: int = 12, fps: float = 10.0
) -> OverlayS2QcSource:
    sync = _prepared_identity_sync(n_prepared=n_prepared, fps=fps)
    s2_dir = tmp_path / "pose_inference" / "bottomup__synth"
    s2_dir.mkdir(parents=True)
    pose = _write_s2_pose(s2_dir / "pose.parquet", n_prepared=n_prepared)
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
    sync_path = tmp_path / "preprocess" / "prepared_sync.npz"
    sync_path.write_bytes(b"synthetic-sync-placeholder")
    sync_sha = sha256_file(sync_path)
    prepare_meta = tmp_path / "preprocess" / "prepare_meta.json"
    prepare_meta.write_text("{}", encoding="utf-8")
    return OverlayS2QcSource(
        s2_run_id="bottomup__synth",
        s2_run_dir=s2_dir,
        s2_status="completed",
        pose_parquet_path=pose,
        pose_parquet_sha256=pose_sha,
        session_root=tmp_path,
        prepared_sync_path=sync_path,
        prepared_sync_sha256=sync_sha,
        prepared_video_path=video,
        prepared_video_sha256=video_sha,
        prepare_meta_path=prepare_meta,
        video_probe=probe,
        sync=sync,
        sync_build_audit={
            "kind": "s1_prepared_frame_identity",
            "n_prepared": n_prepared,
            "fps_header": fps,
            "dense_120hz": False,
        },
        prepared_frame_count=n_prepared,
        prepared_fps=fps,
        skeleton_edges=(("nose", "neck"), ("neck", "spine_base")),
        skeleton_source="synthetic",
        pose_slp_path=None,
        settings_used_path=None,
        pose_meta={"run_id": "bottomup__synth", "status": "completed"},
        source_digests={
            "s2_pose_parquet_sha256": pose_sha,
            "s1_prepared_sync_sha256": sync_sha,
            "s1_prepared_video_sha256": video_sha,
        },
        provenance={"source_kind": "s2_inferred_pose", "synthetic": True},
    )


def test_s2_export_frame_count_clip_boundaries_and_manifest(tmp_path: Path) -> None:
    source = _synthetic_s2_source(tmp_path)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=4,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        show_confidence=True,
    )
    assert plan.prepared_interval.frame_count == 4
    assert LayerId.S2_POSE_INFERRED in plan.selected_layers

    with patch("overlay.export_s2.verify_overlay_s2_qc_source_integrity"):
        result = export_s2_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )

    assert result.frames_written == 4
    assert result.qc_status == "completed"
    assert result.video_path.name == OVERLAY_QC_S2_POSE_VIDEO_FILENAME
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
    assert manifest.renderer_version == OVERLAY_S2_POSE_RENDERER_VERSION
    assert manifest.output_frame_domain == "prepared"
    assert manifest.extra["export_kind"] == "s2_pose_qc_clip"
    assert manifest.extra["confidence"]["mapping_id"] == S2_SCORE_VISUAL_MAPPING_ID
    assert manifest.extra["confidence"]["no_confidence_filtering"] is True
    assert manifest.extra["identity_semantics"]["label_prefix"] == "P"
    assert "s2.pose_inferred" in manifest.layer_selections
    assert manifest.extra["source"]["pose_parquet_sha256"] == source.pose_parquet_sha256
    assert "dense 120" in " ".join(result.notes).lower() or any(
        "prepared-identity" in n for n in result.notes
    )


def test_s2_prepared_identity_timing_notes(tmp_path: Path) -> None:
    source = _synthetic_s2_source(tmp_path, n_prepared=6)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with patch("overlay.export_s2.verify_overlay_s2_qc_source_integrity"):
        result = export_s2_pose_qc_clip(
            source=source, plan=plan, output_root=tmp_path / "out"
        )
    first = result.manifest.extra["first_frame_timing"]
    assert first["prepared_frame_idx"] == 0
    assert first["dense_sample_idx"] == 0  # identity map


def test_s2_cancellation_cleans_staging(tmp_path: Path) -> None:
    source = _synthetic_s2_source(tmp_path, n_prepared=20)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=15,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with (
        patch("overlay.export_s2.verify_overlay_s2_qc_source_integrity"),
        pytest.raises(OverlayExportCancelled),
    ):
        export_s2_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "out",
            cancellation_requested=cancel,
        )
    exports = list((tmp_path / "out").glob("overlay_qc_s2_pose__*"))
    for job in exports:
        assert not (job / OVERLAY_QC_S2_POSE_VIDEO_FILENAME).exists()
        assert not (job / "_overlay_qc_staging").exists()


def test_s2_digest_mismatch_before_publish(tmp_path: Path) -> None:
    source = _synthetic_s2_source(tmp_path)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    live_calls = {"n": 0}

    def _integrity(src: OverlayS2QcSource) -> None:
        live_calls["n"] += 1
        if live_calls["n"] >= 2:
            raise OverlaySourceError(
                "Overlay S2 QC source digest mismatch for s2_pose_parquet_sha256: "
                "pinned='aaa' live='bbb'."
            )

    with (
        patch(
            "overlay.export_s2.verify_overlay_s2_qc_source_integrity",
            side_effect=_integrity,
        ),
        pytest.raises(OverlaySourceError, match="digest mismatch"),
    ):
        export_s2_pose_qc_clip(
            source=source, plan=plan, output_root=tmp_path / "out"
        )


def test_s2_refuses_overwrite_existing_artifacts(tmp_path: Path) -> None:
    source = _synthetic_s2_source(tmp_path)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=1,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    job = tmp_path / "results_exports" / "overlay_qc_s2_pose__fixed"
    job.mkdir(parents=True)
    (job / OVERLAY_QC_S2_POSE_VIDEO_FILENAME).write_bytes(b"x")
    with (
        patch("overlay.export_s2.verify_overlay_s2_qc_source_integrity"),
        pytest.raises(OverlayExportError, match="Refusing to overwrite"),
    ):
        export_s2_pose_qc_clip(source=source, plan=plan, job_dir=job)


def test_s2_source_resolver_rejects_incomplete(tmp_path: Path) -> None:
    run = tmp_path / "pose_inference" / "incomplete"
    run.mkdir(parents=True)
    with pytest.raises(OverlaySourceError, match="not eligible|does not exist|pose.parquet"):
        resolve_overlay_s2_qc_source(s2_run_dir=run)


def test_s2_source_never_auto_selects(tmp_path: Path) -> None:
    # Explicit path required — no discovery helper picks newest.
    with pytest.raises(TypeError):
        resolve_overlay_s2_qc_source()  # type: ignore[call-arg]


def test_s2_upstream_pose_digest_immutable_across_export(tmp_path: Path) -> None:
    source = _synthetic_s2_source(tmp_path)
    before = sha256_file(source.pose_parquet_path)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with patch("overlay.export_s2.verify_overlay_s2_qc_source_integrity"):
        export_s2_pose_qc_clip(
            source=source, plan=plan, output_root=tmp_path / "out"
        )
    assert sha256_file(source.pose_parquet_path) == before
