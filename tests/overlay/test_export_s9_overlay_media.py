"""C3-5 S9 overlay media export: sibling family, sources, bouts, GUI settings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import write_behavior_states_parquet
from tests.overlay.test_export_mixed_pose_behavior_qc import (
    _mixed_s3_s8,
    _mixed_s4_s8,
)
from tests.overlay.test_export_qc import _synthetic_source
from tests.overlay.test_export_s3_pose_qc import _synthetic_s3_source
from tests.overlay.test_export_s4_pose_qc import _synthetic_s4_source
from tests.results_packaging._export_fixtures import (
    build_mini_export_package,
    count_video_frames,
    patch_export_package_currently_eligible,
    write_synthetic_mp4,
)

from behavioral_review.sync import SyncMap
from overlay import (
    MEDIA_EXPORT_OVERLAY_JOB_KIND,
    MEDIA_EXPORT_OVERLAY_JOB_PREFIX,
    OVERLAY_S9_MEDIA_RENDERER_VERSION,
    BehaviorSourceKind,
    OverlaySourceError,
    PoseSourceKind,
)
from overlay.contracts import LayerId
from overlay.export_s9_media import (
    OverlayMediaExportSettings,
    OverlayMediaResolvedSources,
    publish_overlay_media_export,
    resolve_overlay_media_sources,
)
from overlay.source import OverlayQcSource
from project.file_digest import sha256_file
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import (
    CLIP_INDEX_FILENAME,
    CLIPS_DIRNAME,
    ENCODE_STATUS_SUCCEEDED,
    JOB_KIND_MEDIA,
    MEDIA_EXPORT_RUN_PREFIX,
    ExportConfigError,
)
from results_packaging.media_csv_export.discovery import summarize_export_project
from results_packaging.media_csv_export.runner import MediaExportRequest, run_media_export
from results_packaging.media_csv_export.source import ExportSourcePackage
from results_packaging.media_csv_export.video import probe_video
from ui.controllers.results_export_support import ExportUiSettings


def _package_from_behavior_source(
    behavior: OverlayQcSource,
    *,
    bouts: pd.DataFrame | None = None,
) -> ExportSourcePackage:
    probe = behavior.video_probe
    bout_frame = bouts if bouts is not None else behavior.behavior_bouts
    if bout_frame is None or len(bout_frame) == 0:
        bout_frame = pd.DataFrame(
            [
                {
                    "bout_id": f"{behavior.package_run_id}__bout_a",
                    "source_s8_bout_id": "bout_a",
                    "package_run_id": behavior.package_run_id,
                    "source_s8_run_id": "s8",
                    "channel": "exclusive",
                    "behavior_name": "approach",
                    "identity_grain": "directional",
                    "focal_id": 0,
                    "partner_id": 1,
                    "mouse_a_id": 0,
                    "mouse_b_id": 1,
                    "pair_id": "0|1",
                    "start_sample_idx": 2,
                    "end_sample_idx": 5,
                    "start_time_sec": 2 / 120,
                    "end_time_sec": 5 / 120,
                    "bout_duration_frames": 4,
                    "bout_duration_sec": 4 / 120,
                    "prepared_frame_idx_start": int(
                        behavior.sync.mapping_at_sample(2).prepared_frame_idx
                    ),
                    "prepared_frame_idx_end": int(
                        behavior.sync.mapping_at_sample(5).prepared_frame_idx
                    ),
                    "prepared_time_sec_start": 0.0,
                    "prepared_time_sec_end": 0.1,
                }
            ]
        )
    return ExportSourcePackage(
        package_dir=Path(behavior.package_dir),
        package_run_id=behavior.package_run_id,
        package_status="completed",
        session_root=Path(behavior.session_root) if behavior.session_root else None,
        run_meta={},
        package_manifest={},
        behavior_bouts=bout_frame,
        behavior_bouts_sha256=str(
            behavior.source_digests.get("behavior_bouts_sha256") or "bouts"
        ),
        sync_index_sha256=str(behavior.source_digests.get("sync_index_sha256") or "sync"),
        package_manifest_sha256=str(
            behavior.source_digests.get("package_manifest_sha256") or "pkg"
        ),
        source_s8_run_id="behavioral_arbitration__synth",
        source_s8_run_dir=Path(behavior.package.source_s8.run_dir),
        source_s7_run_id=None,
        source_s6_run_id=None,
        source_video_path=Path(behavior.video_path),
        source_video_kind="prepared_video",
        source_video_fingerprint=None,
        source_video_sha256=behavior.video_sha256,
        source_video_verification_strength="content_hash",
        currently_eligible=True,
        stale_warning=None,
        video_probe=probe,
    )


def _resolved_behavior_only(tmp_path: Path) -> OverlayMediaResolvedSources:
    behavior = _synthetic_source(tmp_path, n_prepared=12, fps=10.0)
    package = _package_from_behavior_source(behavior)
    layers = (
        LayerId.S8_EXCLUSIVE_A_TO_B,
        LayerId.S8_EXCLUSIVE_B_TO_A,
        LayerId.S8_CONTACT,
        LayerId.S8_SBS,
        LayerId.DEBUG_FRAME_COUNTER,
    )
    from overlay.style import DrawConfig

    return OverlayMediaResolvedSources(
        package=package,
        mode="behavior",
        behavior=behavior,
        pose_s3=None,
        pose_s4=None,
        mixed=None,
        prepared_video_path=Path(behavior.video_path),
        prepared_video_sha256=str(behavior.video_sha256),
        video_fps=float(behavior.video_probe.fps),
        video_width=int(behavior.video_probe.width),
        video_height=int(behavior.video_probe.height),
        video_n_frames=int(behavior.video_probe.n_frames),
        sync_authority="behavior_package_sync_index_last_in_plateau",
        source_digests=dict(behavior.source_digests),
        compatibility_notes=(),
        selected_layers=layers,
        style=DrawConfig(darken_alpha=0.92, legend_text=True),
        provenance={"synthetic": True},
    )


def test_overlay_media_settings_defaults_and_validation() -> None:
    cfg = OverlayMediaExportSettings().validated()
    assert cfg.job_kind == MEDIA_EXPORT_OVERLAY_JOB_KIND
    assert cfg.include_behavior is True
    assert cfg.include_pose is False
    assert cfg.hash_clips is True
    with pytest.raises(ExportConfigError, match="include_behavior and/or include_pose"):
        OverlayMediaExportSettings(
            include_behavior=False, include_pose=False
        ).validated()
    with pytest.raises(ExportConfigError, match="explicit pose_run_dir"):
        OverlayMediaExportSettings(
            include_behavior=False,
            include_pose=True,
            pose_source_kind="s4_finalized",
        ).validated()
    with pytest.raises(ExportConfigError, match="pose_run_dir is only valid"):
        OverlayMediaExportSettings(
            include_behavior=True,
            include_pose=False,
            pose_run_dir="/tmp/pose",
        ).validated()


def test_export_ui_settings_default_remains_naked() -> None:
    settings = ExportUiSettings().validated()
    assert settings.overlay_enabled is False
    media = settings.to_media_settings()
    assert media.job_kind == JOB_KIND_MEDIA
    with pytest.raises(ExportConfigError, match="overlay_enabled=True"):
        settings.to_overlay_media_settings()


def test_export_ui_settings_overlay_requires_explicit_pose(tmp_path: Path) -> None:
    pose = tmp_path / "pose_finalization__x"
    pose.mkdir()
    settings = ExportUiSettings(
        overlay_enabled=True,
        overlay_include_behavior=True,
        overlay_include_pose=True,
        overlay_pose_source_kind="s4_finalized",
        overlay_pose_run_dir=str(pose),
    ).validated()
    overlay = settings.to_overlay_media_settings()
    assert overlay.include_pose is True
    assert overlay.pose_source_kind == "s4_finalized"
    assert overlay.pose_run_dir == str(pose.resolve())
    with pytest.raises(ExportConfigError, match="explicitly selected pose run"):
        ExportUiSettings(
            overlay_enabled=True,
            overlay_include_pose=True,
            overlay_pose_source_kind="s4_finalized",
        ).validated()


def test_resolve_rejects_s10_substitution(tmp_path: Path) -> None:
    mixed = _mixed_s4_s8(tmp_path)
    settings = OverlayMediaExportSettings(
        include_behavior=True,
        include_pose=True,
        pose_source_kind="s4_finalized",
        pose_run_dir=str(mixed.pose_s4.s4_run_dir),  # type: ignore[union-attr]
    ).validated()
    with (
        patch(
            "overlay.export_s9_media.validate_package_for_export",
            return_value=_package_from_behavior_source(mixed.behavior),
        ),
        patch(
            "overlay.export_s9_media.resolve_overlay_mixed_qc_source",
            return_value=MagicMock(
                behavior_source_kind=BehaviorSourceKind.S10_REVIEWED,
                behavior=mixed.behavior,
                pose_s4=mixed.pose_s4,
                pose_s3=None,
                pose_source_kind=PoseSourceKind.S4_FINALIZED,
                prepared_video_path=mixed.prepared_video_path,
                prepared_video_sha256=mixed.prepared_video_sha256,
                video_probe=mixed.video_probe,
                sync_authority=mixed.sync_authority,
                source_digests=mixed.source_digests,
                compatibility_notes=(),
                provenance={},
                skeleton_edges=(),
            ),
        ),
        pytest.raises(OverlaySourceError, match="S10 reviewed"),
    ):
        resolve_overlay_media_sources(
            package_dir=mixed.behavior.package_dir,
            settings=settings,
        )


def test_resolve_rejects_digest_mismatched_pose(tmp_path: Path) -> None:
    behavior = _synthetic_source(tmp_path, n_prepared=12, fps=10.0)
    package = _package_from_behavior_source(behavior)
    # Different video digest on pose source.
    pose = _synthetic_s4_source(tmp_path / "pose_side")
    settings = OverlayMediaExportSettings(
        include_behavior=False,
        include_pose=True,
        pose_source_kind="s4_finalized",
        pose_run_dir=str(pose.s4_run_dir),
    ).validated()
    with (
        patch(
            "overlay.export_s9_media.validate_package_for_export",
            return_value=package,
        ),
        patch(
            "overlay.export_s9_media.resolve_overlay_s4_qc_source",
            return_value=pose,
        ),
        pytest.raises(OverlaySourceError, match="prepared video path"),
    ):
        resolve_overlay_media_sources(
            package_dir=package.package_dir,
            settings=settings,
        )


def test_behavior_only_overlay_media_export_preserves_bout_order(tmp_path: Path) -> None:
    sources = _resolved_behavior_only(tmp_path)
    settings = OverlayMediaExportSettings(
        include_behavior=True,
        include_pose=False,
        selection=SelectionSpec(mode="all_package_bouts"),
        hash_clips=True,
    ).validated()
    with (
        patch(
            "overlay.export_s9_media.verify_overlay_media_sources_integrity",
            return_value=None,
        ),
        patch(
            "overlay.export_s9_media.assert_source_unchanged",
            return_value=None,
        ),
    ):
        result = publish_overlay_media_export(
            session_root=tmp_path,
            sources=sources,
            settings=settings,
            timestamp="20261104T120000",
        )
    assert result.success
    assert result.export_job_id.startswith(MEDIA_EXPORT_OVERLAY_JOB_PREFIX)
    assert result.job_dir.name.startswith(MEDIA_EXPORT_OVERLAY_JOB_PREFIX)
    assert (result.job_dir / CLIP_INDEX_FILENAME).is_file()
    clip_index = pd.read_parquet(result.job_dir / CLIP_INDEX_FILENAME)
    assert list(clip_index["bout_id"]) == list(sources.package.behavior_bouts["bout_id"])
    assert set(clip_index["encode_status"]) == {ENCODE_STATUS_SUCCEEDED}
    for rel in clip_index["clip_relpath"]:
        clip_path = result.job_dir / CLIPS_DIRNAME / Path(str(rel))
        assert clip_path.is_file()
        assert "\\" not in str(rel)
        fps, n = count_video_frames(clip_path)
        assert n > 0 and fps > 0
    meta = json.loads((result.job_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["job_kind"] == MEDIA_EXPORT_OVERLAY_JOB_KIND
    assert meta["renderer_version"] == OVERLAY_S9_MEDIA_RENDERER_VERSION
    assert meta["s10_substitution"] is False
    assert meta["behavior_source_kind"] == "s8_automated"
    manifest = json.loads(
        (result.job_dir / "export_job_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clip_hashes_sha256"]
    assert manifest["scientific_artifact_writes"] is False
    assert (result.job_dir / "overlay_artifact_manifest.json").is_file()


def test_selected_bout_span_matches_packaged_prepared(tmp_path: Path) -> None:
    sources = _resolved_behavior_only(tmp_path)
    bout = sources.package.behavior_bouts.iloc[0]
    settings = OverlayMediaExportSettings(
        include_behavior=True,
        include_pose=False,
        selection=SelectionSpec(
            mode="bout_id_list", bout_ids=(str(bout["bout_id"]),)
        ),
        pre_context_sec=0.0,
        post_context_sec=0.0,
    ).validated()
    with (
        patch(
            "overlay.export_s9_media.verify_overlay_media_sources_integrity",
            return_value=None,
        ),
        patch(
            "overlay.export_s9_media.assert_source_unchanged",
            return_value=None,
        ),
    ):
        result = publish_overlay_media_export(
            session_root=tmp_path,
            sources=sources,
            settings=settings,
            timestamp="20261104T120001",
        )
    clip_index = pd.read_parquet(result.job_dir / CLIP_INDEX_FILENAME)
    row = clip_index.iloc[0]
    assert int(row["prepared_frame_idx_start"]) == int(bout["prepared_frame_idx_start"])
    assert int(row["prepared_frame_idx_end"]) == int(bout["prepared_frame_idx_end"])
    assert int(row["start_sample_idx"]) == int(bout["start_sample_idx"])
    assert int(row["end_sample_idx"]) == int(bout["end_sample_idx"])
    expected = int(row["clip_frame_count"])
    clip_path = result.job_dir / CLIPS_DIRNAME / Path(str(row["clip_relpath"]))
    _, n = count_video_frames(clip_path)
    assert n == expected


def test_cancellation_cleans_partial_tmp_and_marks_incomplete(tmp_path: Path) -> None:
    sources = _resolved_behavior_only(tmp_path)
    # Duplicate bouts so cancel can fire mid-batch.
    bouts = pd.concat(
        [sources.package.behavior_bouts, sources.package.behavior_bouts],
        ignore_index=True,
    )
    bouts.loc[1, "bout_id"] = str(bouts.loc[0, "bout_id"]) + "__dup"
    sources = OverlayMediaResolvedSources(
        package=ExportSourcePackage(
            **{
                **{
                    f.name: getattr(sources.package, f.name)
                    for f in sources.package.__dataclass_fields__.values()  # type: ignore[attr-defined]
                },
                "behavior_bouts": bouts,
            }
        )
        if False
        else _package_from_behavior_source(sources.behavior, bouts=bouts),  # type: ignore[arg-type]
        mode=sources.mode,
        behavior=sources.behavior,
        pose_s3=None,
        pose_s4=None,
        mixed=None,
        prepared_video_path=sources.prepared_video_path,
        prepared_video_sha256=sources.prepared_video_sha256,
        video_fps=sources.video_fps,
        video_width=sources.video_width,
        video_height=sources.video_height,
        video_n_frames=sources.video_n_frames,
        sync_authority=sources.sync_authority,
        source_digests=sources.source_digests,
        compatibility_notes=(),
        selected_layers=sources.selected_layers,
        style=sources.style,
        provenance={},
    )
    calls = {"n": 0}

    def _cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    settings = OverlayMediaExportSettings(
        include_behavior=True,
        include_pose=False,
        selection=SelectionSpec(mode="all_package_bouts"),
    ).validated()
    with (
        patch(
            "overlay.export_s9_media.verify_overlay_media_sources_integrity",
            return_value=None,
        ),
        patch(
            "overlay.export_s9_media.assert_source_unchanged",
            return_value=None,
        ),
    ):
        result = publish_overlay_media_export(
            session_root=tmp_path,
            sources=sources,
            settings=settings,
            timestamp="20261104T120002",
            cancellation_requested=_cancel,
        )
    assert result.status == "incomplete"
    assert result.diagnostics.get("canceled") is True
    assert not (result.job_dir / "_overlay_media_staging").exists()
    leftover_tmp = list(result.job_dir.rglob("*.mp4.tmp"))
    assert leftover_tmp == []


def test_discovery_lists_overlay_media_and_ignores_qc_prefix(tmp_path: Path) -> None:
    session = tmp_path / "session"
    exports = session / "results_exports"
    overlay_job = exports / f"{MEDIA_EXPORT_OVERLAY_JOB_PREFIX}20261104T130000"
    naked_job = exports / f"{MEDIA_EXPORT_RUN_PREFIX}__20261104T130001"
    qc_job = exports / "overlay_qc_behavioral__20261104T130002"
    for job, kind in (
        (overlay_job, MEDIA_EXPORT_OVERLAY_JOB_KIND),
        (naked_job, JOB_KIND_MEDIA),
    ):
        job.mkdir(parents=True)
        (job / "run_meta.json").write_text(
            json.dumps(
                {
                    "export_job_id": job.name,
                    "job_kind": kind,
                    "status": "completed",
                    "created_at": "2026-11-04T13:00:00+00:00",
                    "source_package": {
                        "run_id": "results_package__x",
                        "path": str(tmp_path / "pkg"),
                        "behavior_bouts_sha256": "a" * 64,
                    },
                    "counts": {"n_selected": 1, "n_succeeded": 1, "n_failed": 0},
                }
            ),
            encoding="utf-8",
        )
        (job / "export_job_manifest.json").write_text("{}", encoding="utf-8")
        (job / "qc.json").write_text("{}", encoding="utf-8")
        (job / CLIP_INDEX_FILENAME).write_bytes(b"")  # presence only for classify
    qc_job.mkdir(parents=True)
    (qc_job / "run_meta.json").write_text("{}", encoding="utf-8")

    # Avoid live package reopen during classify.
    with patch(
        "results_packaging.media_csv_export.discovery.validate_package_for_export",
        side_effect=Exception("skip"),
    ):
        summary = summarize_export_project(session)
    kinds = {j.job_kind for j in summary.jobs}
    assert MEDIA_EXPORT_OVERLAY_JOB_KIND in kinds
    assert JOB_KIND_MEDIA in kinds
    assert all(not j.export_job_id.startswith("overlay_qc_") for j in summary.jobs)
    assert len(summary.overlay_media_jobs) == 1


def test_naked_media_export_unchanged_when_overlay_disabled(tmp_path: Path) -> None:
    built = build_mini_export_package(tmp_path, n_frames=30, fps=10.0)
    with patch_export_package_currently_eligible():
        result = run_media_export(
            MediaExportRequest(
                session_root=built["session_root"],
                package_dir=built["package_dir"],
                settings=None,
                timestamp="20261104T140000",
                selection=SelectionSpec(
                    mode="bout_id_list",
                    bout_ids=(
                        f"{built['package_run_id']}__bout_edge_start",
                    ),
                ),
            )
        )
    assert result.success
    assert result.export_job_id.startswith(f"{MEDIA_EXPORT_RUN_PREFIX}__")
    assert not result.export_job_id.startswith(MEDIA_EXPORT_OVERLAY_JOB_PREFIX)
    assert (result.job_dir / CLIPS_DIRNAME).is_dir()


def test_irregular_plateau_timing_authority_behavior_only(tmp_path: Path) -> None:
    """Irregular plateaus: LAST_IN_PLATEAU must use max sample on a prepared frame."""

    from overlay.contracts import PlateauPolicy
    from overlay.style import DrawConfig
    from overlay.timing import resolve_prepared_frame_timing

    # Use hand-computed sync (exact timing) with plateau_repeats=3, then assert
    # LAST_IN_PLATEAU picks the chronologically latest sample on a plateau.
    sync_df = build_hand_computed_sync(n_prepared=6, plateau_repeats=3)
    sync = SyncMap.from_sync_index(sync_df)
    plateau = sync.plateau_sample_indices(2)
    assert len(plateau) == 3
    timing = resolve_prepared_frame_timing(
        sync, 2, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    assert timing.dense_sample_idx == int(plateau.max())
    assert timing.dense_sample_idx != int(plateau.min())

    behavior = _synthetic_source(tmp_path, n_prepared=6, fps=10.0)
    behavior = OverlayQcSource(
        source_kind=behavior.source_kind,
        package_run_id=behavior.package_run_id,
        package_dir=behavior.package_dir,
        review_run_id=None,
        review_run_dir=None,
        session_root=behavior.session_root,
        package=behavior.package,
        sync=sync,
        states_path=write_behavior_states_parquet(
            tmp_path / "states_plateau.parquet", sync_df
        ),
        states_sha256="states",
        video_path=behavior.video_path,
        video_probe=behavior.video_probe,
        video_sha256=behavior.video_sha256,
        source_digests=behavior.source_digests,
        behavior_bouts=pd.DataFrame(
            [
                {
                    "bout_id": f"{behavior.package_run_id}__plateau",
                    "source_s8_bout_id": "plateau",
                    "package_run_id": behavior.package_run_id,
                    "source_s8_run_id": "s8",
                    "channel": "exclusive",
                    "behavior_name": "approach",
                    "identity_grain": "directional",
                    "focal_id": 0,
                    "partner_id": 1,
                    "mouse_a_id": 0,
                    "mouse_b_id": 1,
                    "pair_id": "0|1",
                    "start_sample_idx": int(plateau.min()),
                    "end_sample_idx": int(plateau.max()),
                    "start_time_sec": float(plateau.min()) / 120.0,
                    "end_time_sec": float(plateau.max()) / 120.0,
                    "bout_duration_frames": 3,
                    "bout_duration_sec": 3 / 120.0,
                    "prepared_frame_idx_start": 2,
                    "prepared_frame_idx_end": 2,
                    "prepared_time_sec_start": 0.2,
                    "prepared_time_sec_end": 0.2,
                }
            ]
        ),
        provenance={},
    )
    # Match video length to sync prepared frames.
    video = write_synthetic_mp4(tmp_path / "prepared_plateau.mp4", n_frames=6, fps=10.0)
    probe = probe_video(video)
    behavior = OverlayQcSource(
        source_kind=behavior.source_kind,
        package_run_id=behavior.package_run_id,
        package_dir=behavior.package_dir,
        review_run_id=None,
        review_run_dir=None,
        session_root=behavior.session_root,
        package=behavior.package,
        sync=sync,
        states_path=behavior.states_path,
        states_sha256=behavior.states_sha256,
        video_path=video,
        video_probe=probe,
        video_sha256=sha256_file(video),
        source_digests=behavior.source_digests,
        behavior_bouts=behavior.behavior_bouts,
        provenance={},
    )
    sources = OverlayMediaResolvedSources(
        package=_package_from_behavior_source(behavior),
        mode="behavior",
        behavior=behavior,
        pose_s3=None,
        pose_s4=None,
        mixed=None,
        prepared_video_path=video,
        prepared_video_sha256=sha256_file(video),
        video_fps=float(probe.fps),
        video_width=int(probe.width),
        video_height=int(probe.height),
        video_n_frames=int(probe.n_frames),
        sync_authority="behavior_package_sync_index_last_in_plateau",
        source_digests=dict(behavior.source_digests),
        compatibility_notes=("irregular_plateau_fixture",),
        selected_layers=(
            LayerId.S8_EXCLUSIVE_A_TO_B,
            LayerId.S8_EXCLUSIVE_B_TO_A,
            LayerId.DEBUG_FRAME_COUNTER,
        ),
        style=DrawConfig(),
        provenance={},
    )
    settings = OverlayMediaExportSettings(
        include_behavior=True,
        include_pose=False,
        selection=SelectionSpec(mode="all_package_bouts"),
    ).validated()
    with (
        patch(
            "overlay.export_s9_media.verify_overlay_media_sources_integrity",
            return_value=None,
        ),
        patch(
            "overlay.export_s9_media.assert_source_unchanged",
            return_value=None,
        ),
    ):
        result = publish_overlay_media_export(
            session_root=tmp_path,
            sources=sources,
            settings=settings,
            timestamp="20261104T150000",
        )
    assert result.success
    clip_index = pd.read_parquet(result.job_dir / CLIP_INDEX_FILENAME)
    # Dense bout spans one prepared plateau → one prepared clip frame.
    assert int(clip_index.iloc[0]["clip_frame_count"]) == 1
    assert int(clip_index.iloc[0]["prepared_frame_idx_start"]) == 2
    assert int(clip_index.iloc[0]["prepared_frame_idx_end"]) == 2


def test_mixed_s4_s8_and_s3_s8_publish_via_resolved_sources(tmp_path: Path) -> None:
    from overlay.style import DrawConfig

    for label, mixed, mode, pose_kind in (
        ("s4", _mixed_s4_s8(tmp_path / "s4"), "mixed_s4", PoseSourceKind.S4_FINALIZED),
        ("s3", _mixed_s3_s8(tmp_path / "s3"), "mixed_s3", PoseSourceKind.S3_ACCEPTED),
    ):
        behavior = mixed.behavior
        package = _package_from_behavior_source(behavior)
        layers = (
            (
                LayerId.S4_POSE_FINAL
                if pose_kind is PoseSourceKind.S4_FINALIZED
                else LayerId.S3_POSE_TRACKED
            ),
            LayerId.S8_EXCLUSIVE_A_TO_B,
            LayerId.S8_EXCLUSIVE_B_TO_A,
            LayerId.S8_CONTACT,
            LayerId.S8_SBS,
            LayerId.DEBUG_FRAME_COUNTER,
        )
        sources = OverlayMediaResolvedSources(
            package=package,
            mode=mode,
            behavior=behavior,
            pose_s3=mixed.pose_s3,
            pose_s4=mixed.pose_s4,
            mixed=mixed,
            prepared_video_path=mixed.prepared_video_path,
            prepared_video_sha256=mixed.prepared_video_sha256,
            video_fps=float(mixed.video_probe.fps),
            video_width=int(mixed.video_probe.width),
            video_height=int(mixed.video_probe.height),
            video_n_frames=int(mixed.video_probe.n_frames),
            sync_authority=mixed.sync_authority,
            source_digests=dict(mixed.source_digests),
            compatibility_notes=mixed.compatibility_notes,
            selected_layers=layers,
            style=DrawConfig(
                darken_alpha=0.92,
                draw_skeleton=True,
                legend_text=True,
                extra={"skeleton_edges": [list(e) for e in mixed.skeleton_edges]},
            ),
            provenance=dict(mixed.provenance),
        )
        settings = OverlayMediaExportSettings(
            include_behavior=True,
            include_pose=True,
            pose_source_kind=pose_kind.value,
            pose_run_dir=str(mixed.pose_run_dir),
            selection=SelectionSpec(mode="all_package_bouts"),
            hash_clips=True,
        ).validated()
        with (
            patch(
                "overlay.export_s9_media.verify_overlay_media_sources_integrity",
                return_value=None,
            ),
            patch(
                "overlay.export_s9_media.assert_source_unchanged",
                return_value=None,
            ),
        ):
            result = publish_overlay_media_export(
                session_root=tmp_path / label,
                sources=sources,
                settings=settings,
                timestamp=f"20261104T16{label[-1]}000",
            )
        assert result.success, result.error_message
        meta = json.loads((result.job_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["configuration"] == f"{pose_kind.value}+s8_automated"
        assert meta["pose_run_id"] == mixed.pose_run_id


def test_pose_only_s3_and_s4_export(tmp_path: Path) -> None:
    from dataclasses import replace

    from overlay.style import DrawConfig

    behavior = _synthetic_source(tmp_path / "pkg", n_prepared=12, fps=10.0)
    package = _package_from_behavior_source(behavior)
    cases = (
        ("s4", _synthetic_s4_source(tmp_path / "s4"), "pose_s4", LayerId.S4_POSE_FINAL),
        ("s3", _synthetic_s3_source(tmp_path / "s3"), "pose_s3", LayerId.S3_POSE_TRACKED),
    )
    for label, pose, mode, layer in cases:
        pose = replace(
            pose,
            prepared_video_path=Path(behavior.video_path),
            prepared_video_sha256=str(behavior.video_sha256),
            video_probe=behavior.video_probe,
        )
        pose_s4 = pose if mode == "pose_s4" else None
        pose_s3 = pose if mode == "pose_s3" else None
        sources = OverlayMediaResolvedSources(
            package=package,
            mode=mode,
            behavior=None,
            pose_s3=pose_s3,
            pose_s4=pose_s4,
            mixed=None,
            prepared_video_path=Path(behavior.video_path),
            prepared_video_sha256=str(behavior.video_sha256),
            video_fps=float(behavior.video_probe.fps),
            video_width=int(behavior.video_probe.width),
            video_height=int(behavior.video_probe.height),
            video_n_frames=int(behavior.video_probe.n_frames),
            sync_authority=mode,
            source_digests={"prepared_video_sha256": str(behavior.video_sha256)},
            compatibility_notes=(),
            selected_layers=(layer, LayerId.DEBUG_FRAME_COUNTER),
            style=DrawConfig(draw_skeleton=True, legend_text=True),
            provenance={},
        )
        settings = OverlayMediaExportSettings(
            include_behavior=False,
            include_pose=True,
            pose_source_kind=(
                "s4_finalized" if mode == "pose_s4" else "s3_accepted"
            ),
            pose_run_dir=str(
                pose.s4_run_dir if mode == "pose_s4" else pose.s3_run_dir  # type: ignore[attr-defined]
            ),
            selection=SelectionSpec(mode="all_package_bouts"),
        ).validated()
        with (
            patch(
                "overlay.export_s9_media.verify_overlay_media_sources_integrity",
                return_value=None,
            ),
            patch(
                "overlay.export_s9_media.assert_source_unchanged",
                return_value=None,
            ),
        ):
            result = publish_overlay_media_export(
                session_root=tmp_path / f"pose_{label}",
                sources=sources,
                settings=settings,
                timestamp=f"20261104T17{label[-1]}000",
            )
        assert result.success, result.error_message
