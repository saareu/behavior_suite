"""C3-2 S3 pose QC export: eligibility, timing, cancel, digests, immutability."""

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
    OVERLAY_S3_POSE_RENDERER_VERSION,
    LayerId,
    OverlayExportCancelled,
    OverlaySourceError,
    export_s3_pose_qc_clip,
    parse_overlay_artifact_manifest,
    plan_s3_prepared_interval,
)
from overlay.contracts import OVERLAY_QC_S3_POSE_VIDEO_FILENAME
from overlay.corrections_s3 import (
    load_machine_correction_index,
    load_manual_correction_index,
)
from overlay.source_s3 import OverlayS3QcSource, resolve_overlay_s3_qc_source
from project.file_digest import sha256_file
from results_packaging.contracts import SYNC_SCHEMA_VERSION
from results_packaging.media_csv_export.video import probe_video
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_NOT_ACCEPTED,
    FINAL_TRACKED_POSE_FILENAME,
    MACHINE_CORRECTIONS_FILENAME,
    MANUAL_CORRECTIONS_FILENAME,
    RUN_META_FILENAME,
)


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


def _write_tracked_pose(path: Path, *, n_prepared: int = 12) -> Path:
    rows: list[dict[str, object]] = []
    for prep in range(n_prepared):
        for track in (0, 1):
            for node_i, node in enumerate(("nose", "neck", "spine_base")):
                rows.append(
                    {
                        "frame_idx": prep,
                        "frame": prep,
                        "track": track,
                        "node": node,
                        "x": 8.0 + prep + track * 20 + node_i * 3,
                        "y": 10.0 + track * 8 + node_i,
                        "score": 0.8,
                        "was_track_swapped": np.uint8(1 if prep == 3 else 0),
                        "was_node_blanked": np.uint8(0),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _synthetic_s3_source(
    tmp_path: Path,
    *,
    n_prepared: int = 12,
    fps: float = 10.0,
    with_machine: bool = True,
    with_manual: bool = True,
) -> OverlayS3QcSource:
    sync = _prepared_identity_sync(n_prepared=n_prepared, fps=fps)
    s3_dir = tmp_path / "tracking_correction" / "s3__synth"
    s3_dir.mkdir(parents=True)
    tracked = _write_tracked_pose(s3_dir / FINAL_TRACKED_POSE_FILENAME, n_prepared=n_prepared)
    video = write_synthetic_mp4(
        tmp_path / "preprocess" / "prepared_video.mp4",
        n_frames=n_prepared,
        fps=fps,
        width=96,
        height=64,
    )
    probe = probe_video(video)
    sync_path = tmp_path / "preprocess" / "prepared_sync.npz"
    sync_path.write_bytes(b"synthetic-sync-placeholder")
    prepare_meta = tmp_path / "preprocess" / "prepare_meta.json"
    prepare_meta.write_text("{}", encoding="utf-8")

    machine_path = s3_dir / MACHINE_CORRECTIONS_FILENAME
    manual_path = s3_dir / MANUAL_CORRECTIONS_FILENAME
    if with_machine:
        machine_path.write_text(
            json.dumps(
                [
                    {"frame": 1, "type": "node_overlap", "track": 0},
                    {"frame": 2, "type": "identity_swap"},
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    if with_manual:
        manual_path.write_text(
            json.dumps(
                [
                    {
                        "action": "blank_node",
                        "start_frame": 1,
                        "end_frame": 2,
                        "track": 1,
                        "node": "nose",
                        "tracks": [1],
                        "source": "user",
                    }
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    tracked_sha = sha256_file(tracked)
    video_sha = sha256_file(video)
    sync_sha = sha256_file(sync_path)
    machine_sha = sha256_file(machine_path) if machine_path.is_file() else None
    manual_sha = sha256_file(manual_path) if manual_path.is_file() else None
    machine_events = load_machine_correction_index(
        machine_path if machine_path.is_file() else None,
        digest_sha256=machine_sha,
    )
    manual_events = load_manual_correction_index(
        manual_path if manual_path.is_file() else None,
        digest_sha256=manual_sha,
    )
    return OverlayS3QcSource(
        s3_run_id="s3__synth",
        s3_run_dir=s3_dir,
        acceptance_state=ACCEPTANCE_ACCEPTED,
        tracked_pose_path=tracked,
        tracked_pose_sha256=tracked_sha,
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
        source_s2_run_id="bottomup__synth",
        source_s2_run_dir=tmp_path / "pose_inference" / "bottomup__synth",
        source_s2_pose_parquet_path=None,
        machine_corrections_path=machine_path if machine_path.is_file() else None,
        manual_corrections_path=manual_path if manual_path.is_file() else None,
        machine_events=machine_events,
        manual_events=manual_events,
        run_meta={"run_id": "s3__synth", "acceptance_state": ACCEPTANCE_ACCEPTED},
        source_digests={
            "s3_tracked_pose_sha256": tracked_sha,
            "s1_prepared_sync_sha256": sync_sha,
            "s1_prepared_video_sha256": video_sha,
            "s3_machine_corrections_sha256": machine_sha,
            "s3_manual_corrections_sha256": manual_sha,
        },
        provenance={"source_kind": "s3_accepted_tracked_pose", "synthetic": True},
    )


def test_s3_export_frame_count_manifest_and_settings(tmp_path: Path) -> None:
    source = _synthetic_s3_source(tmp_path)
    plan = plan_s3_prepared_interval(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=4,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        show_machine_corrections=True,
        show_manual_corrections=True,
    )
    assert plan.prepared_interval.frame_count == 4
    assert LayerId.S3_POSE_TRACKED in plan.selected_layers

    before = sha256_file(source.tracked_pose_path)
    with patch("overlay.export_s3.verify_overlay_s3_qc_source_integrity"):
        result = export_s3_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )
    after = sha256_file(source.tracked_pose_path)
    assert before == after
    assert result.frames_written == 4
    assert result.qc_status == "completed"
    assert result.video_path.name == OVERLAY_QC_S3_POSE_VIDEO_FILENAME
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
    assert manifest.renderer_version == OVERLAY_S3_POSE_RENDERER_VERSION
    assert manifest.extra["export_kind"] == "s3_pose_qc_clip"
    assert manifest.extra["identity_semantics"]["label_prefix"] == "M"
    assert manifest.extra["corrections"]["no_suspicious_frame_fabrication"] is True
    assert manifest.extra["corrections"]["no_was_track_swapped_per_node_events"] is True
    assert manifest.extra["source"]["tracked_pose_sha256"] == source.tracked_pose_sha256
    assert "s3.pose_tracked" in manifest.layer_selections


def test_s3_cancel_cleans_unpublished_job(tmp_path: Path) -> None:
    source = _synthetic_s3_source(tmp_path)
    plan = plan_s3_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=5,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with (
        patch("overlay.export_s3.verify_overlay_s3_qc_source_integrity"),
        pytest.raises(OverlayExportCancelled),
    ):
        export_s3_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "out",
            cancellation_requested=cancel,
        )
    exports = list((tmp_path / "out").glob("overlay_qc_s3_pose__*"))
    assert exports == []


def test_s3_rejects_unaccepted_and_working_only(tmp_path: Path) -> None:
    session = tmp_path / "session"
    run_dir = session / "tracking_correction" / "s3_unaccepted"
    run_dir.mkdir(parents=True)
    (session / "preprocess").mkdir(parents=True)
    (run_dir / RUN_META_FILENAME).write_text(
        json.dumps(
            {
                "run_id": "s3_unaccepted",
                "acceptance_state": ACCEPTANCE_NOT_ACCEPTED,
                "input": {
                    "session_root": str(session),
                    "prepared_sync": str(session / "preprocess" / "prepared_sync.npz"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "working_tracked_pose.parquet").write_bytes(b"working-only")
    with pytest.raises(OverlaySourceError, match="not eligible|not currently accepted"):
        resolve_overlay_s3_qc_source(s3_run_dir=run_dir)


def test_s3_settings_propagation_and_nodes_only_disclosure(tmp_path: Path) -> None:
    from dataclasses import replace

    source = replace(
        _synthetic_s3_source(tmp_path),
        skeleton_edges=(),
        skeleton_source=None,
    )
    plan = plan_s3_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        draw_skeleton=True,
        show_machine_corrections=False,
        show_manual_corrections=True,
    )
    with patch("overlay.export_s3.verify_overlay_s3_qc_source_integrity"):
        result = export_s3_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "out2",
        )
    assert result.manifest.extra["plan"]["show_machine_corrections"] is False
    assert result.manifest.extra["plan"]["show_manual_corrections"] is True
    assert result.manifest.extra["skeleton"]["skeleton_nodes_only"] is True


def test_s3_missing_journals_disclosed_not_fabricated(tmp_path: Path) -> None:
    source = _synthetic_s3_source(tmp_path, with_machine=False, with_manual=False)
    assert source.machine_events.status == "absent"
    assert source.manual_events.status == "absent"
    plan = plan_s3_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=1,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with patch("overlay.export_s3.verify_overlay_s3_qc_source_integrity"):
        result = export_s3_pose_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "out3",
        )
    corrections = result.manifest.extra["corrections"]
    assert corrections["machine"]["status"] == "absent"
    assert corrections["manual"]["status"] == "absent"
    assert corrections["machine_event_draws_in_clip"] == 0
    assert corrections["manual_event_draws_in_clip"] == 0


def test_s3_track_identity_mapping_in_adapter(tmp_path: Path) -> None:
    from overlay.adapters.pose import make_s3_tracked_pose_adapter
    from overlay.contracts import PlateauPolicy
    from overlay.timing import PreparedFrameTiming

    source = _synthetic_s3_source(tmp_path, n_prepared=4)
    adapter = make_s3_tracked_pose_adapter(
        source.tracked_pose_path,
        run_id=source.s3_run_id,
        run_dir=source.s3_run_dir,
        prepared_frame_range=(0, 3),
    )
    timing = PreparedFrameTiming(
        prepared_frame_idx=1,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=1,
        plateau_sample_indices=(1,),
        mapping=None,
    )
    obs = adapter.observe(timing, prepared_frame_idx=1)
    assert obs.available
    tracks = {int(p["track"]) for p in obs.payload["points"]}
    assert tracks == {0, 1}
    assert "S3 accepted tracked pose" in str(obs.payload["legend"])
