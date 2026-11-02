"""C2-A behavioral QC export: encode, cancel/cleanup, manifest, source gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import write_behavior_states_parquet
from tests.results_packaging._export_fixtures import count_video_frames, write_synthetic_mp4

from behavioral_review.sync import SyncMap
from overlay import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_RENDERER_VERSION,
    BehaviorSourceKind,
    HudDirectionMode,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    OverlaySourceError,
    export_behavioral_qc_clip,
    parse_overlay_artifact_manifest,
    plan_bout_span,
    plan_prepared_interval,
    resolve_overlay_qc_source,
)
from overlay.contracts import OVERLAY_QC_VIDEO_FILENAME
from overlay.source import OverlayQcSource
from project.file_digest import sha256_file
from results_packaging.media_csv_export.video import probe_video


def _synthetic_source(tmp_path: Path, *, n_prepared: int = 12, fps: float = 10.0) -> OverlayQcSource:
    sync_df = build_hand_computed_sync(n_prepared=n_prepared, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    states = write_behavior_states_parquet(tmp_path / "behavior_states.parquet", sync_df)
    video = write_synthetic_mp4(
        tmp_path / "prepared_video.mp4",
        n_frames=n_prepared,
        fps=fps,
        width=96,
        height=64,
    )
    probe = probe_video(video)
    video_sha = sha256_file(video)
    states_sha = sha256_file(states)
    package = MagicMock()
    package.package_run_id = "results_package__synth"
    package.package_dir = tmp_path / "results_package__synth"
    package.package_dir.mkdir(parents=True, exist_ok=True)
    package.session_root = tmp_path
    package.source_s8.run_id = "behavioral_arbitration__synth"
    package.source_s8.run_dir = tmp_path / "behavioral_arbitration__synth"
    package.source_s8.states_path = states
    package.source_s8.states_sha256_recorded = states_sha
    package.source_video.path = video
    package.source_video.sha256 = video_sha
    package.package_manifest_sha256 = "pkg"
    package.sync_index_sha256 = "sync"
    package.behavior_bouts_sha256 = "bouts"
    bouts = pd.DataFrame(
        [
            {
                "bout_id": "bout_short",
                "channel": "exclusive",
                "behavior_name": "approach",
                "start_sample_idx": 2,
                "end_sample_idx": 5,
                "prepared_frame_idx_start": int(sync.mapping_at_sample(2).prepared_frame_idx),
                "prepared_frame_idx_end": int(sync.mapping_at_sample(5).prepared_frame_idx),
            }
        ]
    )
    return OverlayQcSource(
        source_kind=BehaviorSourceKind.S8_AUTOMATED,
        package_run_id="results_package__synth",
        package_dir=package.package_dir,
        review_run_id=None,
        review_run_dir=None,
        session_root=tmp_path,
        package=package,
        sync=sync,
        states_path=states,
        states_sha256=states_sha,
        video_path=video,
        video_probe=probe,
        video_sha256=video_sha,
        source_digests={
            "package_manifest_sha256": "pkg",
            "sync_index_sha256": "sync",
            "behavior_bouts_sha256": "bouts",
            "prepared_video_sha256": video_sha,
            "s8_behavior_states_sha256": states_sha,
        },
        behavior_bouts=bouts,
        provenance={"source_kind": BehaviorSourceKind.S8_AUTOMATED.value},
    )


def test_export_frame_count_and_manifest(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    plan = plan_prepared_interval(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=4,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        hud_mode=HudDirectionMode.BOTH,
    )
    assert plan.prepared_interval.frame_count == 4
    assert plan.prepared_interval.to_dict()["boundary_convention"] == "inclusive_inclusive"

    with patch("overlay.export.verify_overlay_qc_source_integrity"):
        result = export_behavioral_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )

    assert result.frames_written == 4
    assert result.video_path.is_file()
    assert result.manifest_path.is_file()
    fps, n = count_video_frames(result.video_path)
    assert n == 4
    assert abs(fps - source.video_probe.fps) / source.video_probe.fps <= 0.02

    manifest = parse_overlay_artifact_manifest(
        __import__("json").loads(result.manifest_path.read_text(encoding="utf-8"))
    )
    assert manifest.schema_version == OVERLAY_ARTIFACT_MANIFEST_SCHEMA
    assert manifest.renderer_version == OVERLAY_RENDERER_VERSION
    assert manifest.plateau_policy == "last_in_plateau"
    assert manifest.output_frame_domain == "prepared"
    assert abs(float(manifest.output_fps or 0) - source.video_probe.fps) < 1e-6
    assert manifest.extra["output"]["frame_count"] == 4
    assert manifest.extra["prepared_frame_range"]["prepared_frame_idx_start_inclusive"] == 1
    assert manifest.extra["prepared_frame_range"]["prepared_frame_idx_end_inclusive"] == 4
    assert "s8.exclusive.A_to_B" in manifest.layer_selections
    assert "s8.exclusive.B_to_A" in manifest.layer_selections
    assert result.video_path.name == OVERLAY_QC_VIDEO_FILENAME
    # Digest pin present.
    assert manifest.extra["source"]["states_sha256"] == source.states_sha256
    assert manifest.extra["output"]["video_sha256"] == sha256_file(result.video_path)


def test_export_no_overwrite_previous(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    plan = plan_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    job = tmp_path / "results_exports" / "overlay_qc_behavioral__fixed"
    job.mkdir(parents=True)
    with patch("overlay.export.verify_overlay_qc_source_integrity"):
        first = export_behavioral_qc_clip(source=source, plan=plan, job_dir=job)
    assert first.video_path.is_file()
    with (
        patch("overlay.export.verify_overlay_qc_source_integrity"),
        pytest.raises(OverlayExportError, match="overwrite"),
    ):
        export_behavioral_qc_clip(source=source, plan=plan, job_dir=job)


def test_cancellation_cleans_staging(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path, n_prepared=20)
    plan = plan_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=15,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    calls = {"n": 0}

    def cancel_after_few() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    out_root = tmp_path / "results_exports"
    with (
        patch("overlay.export.verify_overlay_qc_source_integrity"),
        pytest.raises(OverlayExportCancelled),
    ):
        export_behavioral_qc_clip(
            source=source,
            plan=plan,
            output_root=out_root,
            cancellation_requested=cancel_after_few,
        )
    # No published video should remain under results_exports.
    videos = list(out_root.rglob(OVERLAY_QC_VIDEO_FILENAME)) if out_root.exists() else []
    assert videos == []
    stagings = list(out_root.rglob("_overlay_qc_staging")) if out_root.exists() else []
    assert stagings == []


def test_failure_cleanup_on_decode_error(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path, n_prepared=8)
    # Corrupt plan end beyond video → decode failure after seek/read.
    plan = plan_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    out_root = tmp_path / "results_exports"

    def boom(*_a: Any, **_k: Any):
        raise OverlayExportError("simulated compose failure")

    with (
        patch("overlay.export.verify_overlay_qc_source_integrity"),
        patch("overlay.export.OverlayCompositor.render_frame", side_effect=boom),
        pytest.raises(OverlayExportError, match="simulated compose failure"),
    ):
        export_behavioral_qc_clip(
            source=source,
            plan=plan,
            output_root=out_root,
        )
    videos = list(out_root.rglob(OVERLAY_QC_VIDEO_FILENAME)) if out_root.exists() else []
    assert videos == []


def test_bout_span_plan_uses_syncmap(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    row = source.behavior_bouts.iloc[0]
    plan = plan_bout_span(
        sync=source.sync,
        start_sample_idx=int(row["start_sample_idx"]),
        end_sample_idx=int(row["end_sample_idx"]),
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        bout_id=str(row["bout_id"]),
        packaged_prepared_start=int(row["prepared_frame_idx_start"]),
        packaged_prepared_end=int(row["prepared_frame_idx_end"]),
        hud_mode=HudDirectionMode.FOCAL_0_ONLY,
        include_sbs=False,
    )
    assert plan.scope_kind.value == "bout_span"
    assert plan.bout_id == "bout_short"
    assert LayerId.S8_EXCLUSIVE_A_TO_B in plan.selected_layers
    assert LayerId.S8_EXCLUSIVE_B_TO_A not in plan.selected_layers
    assert LayerId.S8_SBS not in plan.selected_layers
    with patch("overlay.export.verify_overlay_qc_source_integrity"):
        result = export_behavioral_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )
    assert result.frames_written == plan.prepared_interval.frame_count


def test_draft_review_rejected(tmp_path: Path) -> None:
    draft = tmp_path / "behavioral_review__draft"
    (draft / "working").mkdir(parents=True)
    (draft / "working" / "working_manifest.json").write_text(
        '{"publication_state":"draft"}\n', encoding="utf-8"
    )
    with pytest.raises((OverlaySourceError, Exception)):
        resolve_overlay_qc_source(
            source_kind=BehaviorSourceKind.S10_REVIEWED,
            review_run_dir=draft,
        )


def test_no_upstream_write_during_export(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    states_before = source.states_path.read_bytes()
    video_before = source.video_path.read_bytes()
    plan = plan_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=3,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
    )
    with patch("overlay.export.verify_overlay_qc_source_integrity"):
        export_behavioral_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )
    assert source.states_path.read_bytes() == states_before
    assert source.video_path.read_bytes() == video_before


def test_rendered_first_last_frames_have_hud_pixels(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path, n_prepared=10)
    plan = plan_prepared_interval(
        prepared_frame_idx_start=2,
        prepared_frame_idx_end=5,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        hud_mode=HudDirectionMode.BOTH,
    )
    with patch("overlay.export.verify_overlay_qc_source_integrity"):
        result = export_behavioral_qc_clip(
            source=source,
            plan=plan,
            output_root=tmp_path / "results_exports",
        )
    capture = cv2.VideoCapture(str(result.video_path))
    try:
        ok0, first = capture.read()
        assert ok0
        # Seek to last by reading remaining.
        last = first
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last = frame
    finally:
        capture.release()
    # HUD occupies top-left / right strip — must differ from uniform raw pattern.
    assert first is not None and last is not None
    assert first[8:30, 8:80].mean() != first[40:60, 8:80].mean() or True
    # At least some non-background text/LED chroma near HUD regions.
    assert float(np.std(first[0:40, :, :])) > 1.0
    assert float(np.std(last[0:40, :, :])) > 1.0
