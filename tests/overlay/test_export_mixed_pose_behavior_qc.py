"""C3-4 mixed pose + behavioral QC: compatibility, timing, export, discovery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import write_behavior_states_parquet
from tests.overlay.test_export_s3_pose_qc import _synthetic_s3_source
from tests.overlay.test_export_s4_pose_qc import _synthetic_s4_source
from tests.results_packaging._export_fixtures import count_video_frames, write_synthetic_mp4

from behavioral_review.sync import SyncMap
from overlay import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_MIXED_RENDERER_VERSION,
    BehaviorSourceKind,
    LayerId,
    OverlayExportCancelled,
    OverlaySourceError,
    PoseSourceKind,
    export_mixed_pose_behavior_qc_clip,
    parse_overlay_artifact_manifest,
    plan_mixed_prepared_interval,
    resolve_overlay_mixed_qc_source,
    summarize_overlay_qc_project,
)
from overlay.contracts import (
    OVERLAY_QC_MIXED_JOB_PREFIX,
    OVERLAY_QC_MIXED_VIDEO_FILENAME,
    S4_PROVENANCE_ORIGINAL,
)
from overlay.source import OverlayQcSource
from overlay.source_mixed import OverlayMixedQcSource
from project.file_digest import sha256_file
from results_packaging.media_csv_export.video import probe_video


def _synthetic_behavior(
    tmp_path: Path,
    *,
    video: Path,
    sync: SyncMap,
    sync_df: pd.DataFrame,
    source_kind: BehaviorSourceKind = BehaviorSourceKind.S8_AUTOMATED,
    review_run_dir: Path | None = None,
) -> OverlayQcSource:
    states = write_behavior_states_parquet(
        tmp_path / "behavior_states.parquet", sync_df
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
    digests = {
        "package_manifest_sha256": "pkg",
        "sync_index_sha256": "sync",
        "behavior_bouts_sha256": "bouts",
        "prepared_video_sha256": video_sha,
        "s8_behavior_states_sha256": states_sha,
    }
    if source_kind is BehaviorSourceKind.S10_REVIEWED:
        assert review_run_dir is not None
        digests["reviewed_behavior_states_sha256"] = states_sha
    return OverlayQcSource(
        source_kind=source_kind,
        package_run_id="results_package__synth",
        package_dir=package.package_dir,
        review_run_id=(
            "behavioral_review__synth"
            if source_kind is BehaviorSourceKind.S10_REVIEWED
            else None
        ),
        review_run_dir=review_run_dir,
        session_root=tmp_path,
        package=package,
        sync=sync,
        states_path=states,
        states_sha256=states_sha,
        video_path=video,
        video_probe=probe,
        video_sha256=video_sha,
        source_digests=digests,
        behavior_bouts=pd.DataFrame(),
        provenance={"source_kind": source_kind.value, "synthetic": True},
    )


def _mixed_s4_s8(tmp_path: Path) -> OverlayMixedQcSource:
    pose = _synthetic_s4_source(tmp_path)
    sync_df = build_hand_computed_sync(n_prepared=12, plateau_repeats=2)
    # Reuse the same SyncMap identity as pose for agreement checks.
    behavior = _synthetic_behavior(
        tmp_path,
        video=pose.prepared_video_path,
        sync=pose.sync,
        sync_df=sync_df,
    )
    return OverlayMixedQcSource(
        pose_source_kind=PoseSourceKind.S4_FINALIZED,
        behavior_source_kind=BehaviorSourceKind.S8_AUTOMATED,
        pose_s3=None,
        pose_s4=pose,
        behavior=behavior,
        session_root=tmp_path,
        prepared_video_path=pose.prepared_video_path,
        prepared_video_sha256=pose.prepared_video_sha256,
        video_probe=pose.video_probe,
        sync=behavior.sync,
        sync_authority=(
            "dual_independent:"
            "pose_own_map_or_prepared_direct+"
            "behavior_package_sync_index_last_in_plateau"
        ),
        compatibility_notes=("synthetic_compatible",),
        source_digests={
            "prepared_video_sha256": pose.prepared_video_sha256,
            **{f"pose_{k}": v for k, v in pose.source_digests.items()},
            **{f"behavior_{k}": v for k, v in behavior.source_digests.items()},
        },
        provenance={
            "pose_timing": "s4_dense_via_pose_syncmap_last_in_plateau",
            "behavior_timing": "dense_via_behavior_syncmap_last_in_plateau",
        },
    )


def _mixed_s4_s10(tmp_path: Path) -> OverlayMixedQcSource:
    base = _mixed_s4_s8(tmp_path)
    review_dir = tmp_path / "behavioral_review" / "behavioral_review__synth"
    review_dir.mkdir(parents=True, exist_ok=True)
    reviewed = _synthetic_behavior(
        tmp_path,
        video=base.prepared_video_path,
        sync=base.sync,
        sync_df=build_hand_computed_sync(n_prepared=12, plateau_repeats=2),
        source_kind=BehaviorSourceKind.S10_REVIEWED,
        review_run_dir=review_dir,
    )
    return OverlayMixedQcSource(
        pose_source_kind=PoseSourceKind.S4_FINALIZED,
        behavior_source_kind=BehaviorSourceKind.S10_REVIEWED,
        pose_s3=None,
        pose_s4=base.pose_s4,
        behavior=reviewed,
        session_root=tmp_path,
        prepared_video_path=base.prepared_video_path,
        prepared_video_sha256=base.prepared_video_sha256,
        video_probe=base.video_probe,
        sync=reviewed.sync,
        sync_authority=base.sync_authority,
        compatibility_notes=("synthetic_compatible_s10",),
        source_digests={
            "prepared_video_sha256": base.prepared_video_sha256,
            **{f"pose_{k}": v for k, v in base.pose_s4.source_digests.items()},  # type: ignore[union-attr]
            **{f"behavior_{k}": v for k, v in reviewed.source_digests.items()},
        },
        provenance={
            "pose_timing": "s4_dense_via_pose_syncmap_last_in_plateau",
            "behavior_timing": "dense_via_behavior_syncmap_last_in_plateau",
        },
    )


def _mixed_s3_s8(tmp_path: Path) -> OverlayMixedQcSource:
    pose = _synthetic_s3_source(tmp_path)
    # Behavioral SyncMap can be dense-plateau while S3 pose is prepared-domain.
    sync_df = build_hand_computed_sync(n_prepared=12, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    behavior = _synthetic_behavior(
        tmp_path,
        video=pose.prepared_video_path,
        sync=sync,
        sync_df=sync_df,
    )
    return OverlayMixedQcSource(
        pose_source_kind=PoseSourceKind.S3_ACCEPTED,
        behavior_source_kind=BehaviorSourceKind.S8_AUTOMATED,
        pose_s3=pose,
        pose_s4=None,
        behavior=behavior,
        session_root=tmp_path,
        prepared_video_path=pose.prepared_video_path,
        prepared_video_sha256=pose.prepared_video_sha256,
        video_probe=pose.video_probe,
        sync=behavior.sync,
        sync_authority=(
            "dual_independent:"
            "pose_own_map_or_prepared_direct+"
            "behavior_package_sync_index_last_in_plateau"
        ),
        compatibility_notes=("synthetic_compatible_s3",),
        source_digests={
            "prepared_video_sha256": pose.prepared_video_sha256,
            **{f"pose_{k}": v for k, v in pose.source_digests.items()},
            **{f"behavior_{k}": v for k, v in behavior.source_digests.items()},
        },
        provenance={
            "pose_timing": "s3_prepared_frame_direct_mapping",
            "behavior_timing": "dense_via_behavior_syncmap_last_in_plateau",
        },
    )


def _plan_for(source: OverlayMixedQcSource) -> object:
    return plan_mixed_prepared_interval(
        prepared_frame_idx_start=1,
        prepared_frame_idx_end=4,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        pose_source_kind=source.pose_source_kind,
        hud_mode="both",
        include_contact=True,
        include_sbs=True,
        draw_skeleton=True,
    )


@pytest.mark.parametrize(
    "factory",
    [_mixed_s4_s8, _mixed_s4_s10, _mixed_s3_s8],
    ids=["s4_s8", "s4_s10", "s3_s8"],
)
def test_valid_mixed_combinations_export(tmp_path: Path, factory) -> None:
    source = factory(tmp_path)
    plan = _plan_for(source)
    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            return_value=None,
        ),
        patch("overlay.export.probe_encoded_clip") as probe,
    ):
        probe.return_value = MagicMock(n_frames=4, fps=source.video_probe.fps)
        result = export_mixed_pose_behavior_qc_clip(
            source=source,
            plan=plan,  # type: ignore[arg-type]
            job_dir=tmp_path / "results_exports" / f"{OVERLAY_QC_MIXED_JOB_PREFIX}unit",
            verify_encoded_boundaries=True,
        )
    assert result.frames_written == 4
    assert result.video_path.name == OVERLAY_QC_MIXED_VIDEO_FILENAME
    _fps, n_frames = count_video_frames(result.video_path)
    assert n_frames == 4
    assert _fps > 0
    manifest = parse_overlay_artifact_manifest(
        json.loads(result.manifest_path.read_text(encoding="utf-8"))
    )
    assert manifest.schema_version == OVERLAY_ARTIFACT_MANIFEST_SCHEMA
    assert manifest.renderer_version == OVERLAY_MIXED_RENDERER_VERSION
    assert LayerId.S8_EXCLUSIVE_A_TO_B.value in manifest.layer_selections
    assert LayerId.S8_EXCLUSIVE_B_TO_A.value in manifest.layer_selections
    extra = manifest.extra
    assert extra["pose_source_kind"] == source.pose_source_kind.value
    assert extra["behavior_source_kind"] == source.behavior_source_kind.value
    assert extra["source"]["behavior_source_digests"]
    assert extra["source"]["pose_source_digests"]


def test_cross_session_rejection(tmp_path: Path) -> None:
    pose = _synthetic_s4_source(tmp_path / "session_a")
    other = tmp_path / "session_b"
    other.mkdir()
    write_synthetic_mp4(
        other / "preprocess" / "prepared_video.mp4",
        n_frames=12,
        fps=10.0,
        width=96,
        height=64,
    )
    sync_df = build_hand_computed_sync(n_prepared=12, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    # Same video bytes path intentionally different session root.
    behavior = _synthetic_behavior(
        other,
        video=pose.prepared_video_path,
        sync=sync,
        sync_df=sync_df,
    )
    with (
        patch(
            "overlay.source_mixed.resolve_overlay_s4_qc_source",
            return_value=pose,
        ),
        patch(
            "overlay.source_mixed.resolve_overlay_qc_source",
            return_value=behavior,
        ),
        pytest.raises(OverlaySourceError, match="cross-session"),
    ):
        resolve_overlay_mixed_qc_source(
            pose_source_kind=PoseSourceKind.S4_FINALIZED,
            behavior_source_kind=BehaviorSourceKind.S8_AUTOMATED,
            s4_run_dir=pose.s4_run_dir,
            package_dir=behavior.package_dir,
            prepared_interval=(1, 4),
        )


def test_video_digest_mismatch_rejection(tmp_path: Path) -> None:
    pose = _synthetic_s4_source(tmp_path)
    sync_df = build_hand_computed_sync(n_prepared=12, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    other_video = write_synthetic_mp4(
        tmp_path / "other_prepared.mp4",
        n_frames=12,
        fps=10.0,
        width=96,
        height=64,
    )
    behavior = _synthetic_behavior(
        tmp_path,
        video=other_video,
        sync=sync,
        sync_df=sync_df,
    )
    with (
        patch(
            "overlay.source_mixed.resolve_overlay_s4_qc_source",
            return_value=pose,
        ),
        patch(
            "overlay.source_mixed.resolve_overlay_qc_source",
            return_value=behavior,
        ),
        pytest.raises(OverlaySourceError, match="prepared video"),
    ):
        resolve_overlay_mixed_qc_source(
            pose_source_kind="s4_finalized",
            behavior_source_kind="s8_automated",
            s4_run_dir=pose.s4_run_dir,
            package_dir=behavior.package_dir,
            prepared_interval=(1, 4),
        )


def test_s3_plus_s10_rejected_no_silent_substitution(tmp_path: Path) -> None:
    with pytest.raises(OverlaySourceError, match="Unsupported mixed QC combination"):
        resolve_overlay_mixed_qc_source(
            pose_source_kind=PoseSourceKind.S3_ACCEPTED,
            behavior_source_kind=BehaviorSourceKind.S10_REVIEWED,
            s3_run_dir=tmp_path / "s3",
            review_run_dir=tmp_path / "review",
        )


def test_no_silent_s8_s10_or_s3_s4_substitution(tmp_path: Path) -> None:
    with pytest.raises(OverlaySourceError, match="must not receive review_run_dir"):
        resolve_overlay_mixed_qc_source(
            pose_source_kind="s4_finalized",
            behavior_source_kind="s8_automated",
            s4_run_dir=tmp_path / "s4",
            package_dir=tmp_path / "pkg",
            review_run_dir=tmp_path / "review",
        )
    with pytest.raises(OverlaySourceError, match="must not receive s3_run_dir"):
        resolve_overlay_mixed_qc_source(
            pose_source_kind="s4_finalized",
            behavior_source_kind="s8_automated",
            s4_run_dir=tmp_path / "s4",
            s3_run_dir=tmp_path / "s3",
            package_dir=tmp_path / "pkg",
        )
    with pytest.raises(OverlaySourceError, match="must not receive s4_run_dir"):
        resolve_overlay_mixed_qc_source(
            pose_source_kind="s3_accepted",
            behavior_source_kind="s8_automated",
            s3_run_dir=tmp_path / "s3",
            s4_run_dir=tmp_path / "s4",
            package_dir=tmp_path / "pkg",
        )


def test_last_in_plateau_and_s4_provenance(tmp_path: Path) -> None:
    source = _mixed_s4_s8(tmp_path)
    plan = _plan_for(source)
    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            return_value=None,
        ),
        patch("overlay.export.probe_encoded_clip") as probe,
    ):
        probe.return_value = MagicMock(n_frames=4, fps=source.video_probe.fps)
        result = export_mixed_pose_behavior_qc_clip(
            source=source,
            plan=plan,  # type: ignore[arg-type]
            job_dir=tmp_path / "job_prov",
            verify_encoded_boundaries=False,
        )
    notes = " ".join(result.notes)
    assert "policy=last_in_plateau" in notes
    extra = result.manifest.extra
    assert extra["s4_provenance_visual_map"] is not None
    assert str(S4_PROVENANCE_ORIGINAL) in extra["s4_provenance_visual_map"]
    assert LayerId.S8_EXCLUSIVE_A_TO_B in plan.selected_layers  # type: ignore[attr-defined]
    assert LayerId.S8_EXCLUSIVE_B_TO_A in plan.selected_layers  # type: ignore[attr-defined]
    assert LayerId.S8_CONTACT in plan.selected_layers  # type: ignore[attr-defined]
    assert LayerId.S8_SBS in plan.selected_layers  # type: ignore[attr-defined]


def test_contact_sbs_independence_in_plan() -> None:
    plan = plan_mixed_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=10.0,
        source_n_frames=12,
        pose_source_kind="s4_finalized",
        include_contact=False,
        include_sbs=True,
    )
    assert LayerId.S8_CONTACT not in plan.selected_layers
    assert LayerId.S8_SBS in plan.selected_layers
    plan2 = plan_mixed_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=2,
        source_fps=10.0,
        source_n_frames=12,
        pose_source_kind="s3_accepted",
        include_contact=True,
        include_sbs=False,
    )
    assert LayerId.S8_CONTACT in plan2.selected_layers
    assert LayerId.S8_SBS not in plan2.selected_layers
    assert LayerId.S3_POSE_TRACKED in plan2.selected_layers
    assert LayerId.S4_POSE_FINAL not in plan2.selected_layers


def test_cancellation_cleans_staging(tmp_path: Path) -> None:
    source = _mixed_s4_s8(tmp_path)
    plan = _plan_for(source)
    job = tmp_path / "job_cancel"
    calls = {"n": 0}

    def _cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            return_value=None,
        ),
        pytest.raises(OverlayExportCancelled),
    ):
        export_mixed_pose_behavior_qc_clip(
            source=source,
            plan=plan,  # type: ignore[arg-type]
            job_dir=job,
            cancellation_requested=_cancel,
            verify_encoded_boundaries=False,
        )
    staging = job / "_overlay_qc_staging"
    assert not staging.exists()
    assert not (job / OVERLAY_QC_MIXED_VIDEO_FILENAME).exists()


def test_source_digest_revalidation_failure(tmp_path: Path) -> None:
    source = _mixed_s4_s8(tmp_path)
    plan = _plan_for(source)

    def _fail_verify(_source: OverlayMixedQcSource) -> None:
        raise OverlaySourceError("Overlay QC source digest mismatch for prepared_video_sha256")

    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            side_effect=_fail_verify,
        ),
        pytest.raises(OverlaySourceError, match="digest mismatch"),
    ):
        export_mixed_pose_behavior_qc_clip(
            source=source,
            plan=plan,  # type: ignore[arg-type]
            job_dir=tmp_path / "job_digest",
        )


def test_output_discovery(tmp_path: Path) -> None:
    source = _mixed_s4_s8(tmp_path)
    plan = _plan_for(source)
    job = tmp_path / "results_exports" / f"{OVERLAY_QC_MIXED_JOB_PREFIX}disc"
    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            return_value=None,
        ),
        patch("overlay.export.probe_encoded_clip") as probe,
    ):
        probe.return_value = MagicMock(n_frames=4, fps=source.video_probe.fps)
        export_mixed_pose_behavior_qc_clip(
            source=source,
            plan=plan,  # type: ignore[arg-type]
            job_dir=job,
            verify_encoded_boundaries=False,
        )
    summary = summarize_overlay_qc_project(tmp_path)
    mixed = [j for j in summary.jobs if j.export_family == "mixed_pose_behavior"]
    assert len(mixed) == 1
    assert mixed[0].is_completed
    assert mixed[0].has_video
    assert mixed[0].video_filename == OVERLAY_QC_MIXED_VIDEO_FILENAME


def test_upstream_scientific_immutability(tmp_path: Path) -> None:
    source = _mixed_s4_s8(tmp_path)
    plan = _plan_for(source)
    assert source.pose_s4 is not None
    pose_sha_before = sha256_file(source.pose_s4.final_pose_path)
    states_sha_before = sha256_file(source.behavior.states_path)
    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            return_value=None,
        ),
        patch("overlay.export.probe_encoded_clip") as probe,
    ):
        probe.return_value = MagicMock(n_frames=4, fps=source.video_probe.fps)
        export_mixed_pose_behavior_qc_clip(
            source=source,
            plan=plan,  # type: ignore[arg-type]
            job_dir=tmp_path / "job_immut",
            verify_encoded_boundaries=False,
        )
    assert sha256_file(source.pose_s4.final_pose_path) == pose_sha_before
    assert sha256_file(source.behavior.states_path) == states_sha_before


def _irregular_independent_maps(
    tmp_path: Path,
) -> tuple[OverlayMixedQcSource, dict[str, object]]:
    """S4+S8 pair with independent plateau maps and a pose coverage gap.

    Pose SyncMap uses plateau_repeats=2 with prepared 2 removed. Behavior
    SyncMap uses plateau_repeats=3 over the full prepared domain. LAST_IN_PLATEAU
    therefore yields different dense sample indices where both cover a frame.
    """

    from dataclasses import replace

    from tests.overlay._fixtures import build_gap_sync_with_missing_prepared
    from tests.overlay.test_export_s4_pose_qc import _write_s4_pose_for_sync

    from overlay.contracts import PlateauPolicy
    from overlay.evidence import collect_evidence
    from overlay.export_mixed import build_mixed_adapters
    from overlay.timing import resolve_prepared_frame_timing

    n_prepared = 5
    pose_sync_df = build_gap_sync_with_missing_prepared()
    pose_sync = SyncMap.from_sync_index(pose_sync_df)
    behavior_sync_df = build_hand_computed_sync(
        n_prepared=n_prepared, plateau_repeats=3
    )
    behavior_sync = SyncMap.from_sync_index(behavior_sync_df)

    # Distinct LAST_IN_PLATEAU dense choices on overlapping coverage.
    pose_at_0 = resolve_prepared_frame_timing(pose_sync, 0).dense_sample_idx
    behavior_at_0 = resolve_prepared_frame_timing(behavior_sync, 0).dense_sample_idx
    assert pose_at_0 is not None and behavior_at_0 is not None
    assert pose_at_0 != behavior_at_0

    base_pose = _synthetic_s4_source(tmp_path, n_prepared=n_prepared)
    pose_path = _write_s4_pose_for_sync(
        base_pose.s4_run_dir / "final_pose.parquet", pose_sync
    )
    # Encode dense sample into nose x so rendered/observed pose reveals authority.
    pose_df = pd.read_parquet(pose_path)
    pose_df.loc[pose_df["node"] == "nose", "x"] = (
        pose_df.loc[pose_df["node"] == "nose", "sample_idx"].astype(float) + 0.25
    )
    pose_df.to_parquet(pose_path, index=False)
    pose = replace(
        base_pose,
        final_pose_path=pose_path,
        final_pose_sha256=sha256_file(pose_path),
        sync=pose_sync,
        sync_build_audit={"n_dense": int(pose_sync.n_dense), "irregular": True},
        source_digests={
            **base_pose.source_digests,
            "s4_final_pose_sha256": sha256_file(pose_path),
        },
    )

    # Behavior labels encode the dense sample so HUD authority is observable.
    states = tmp_path / "behavior_states_irregular.parquet"
    rows: list[dict[str, object]] = []
    for i in range(int(len(behavior_sync_df))):
        sample = int(behavior_sync_df.loc[i, "sample_idx"])
        t = float(behavior_sync_df.loc[i, "time_sec"])
        for focal, partner in ((0, 1), (1, 0)):
            rows.append(
                {
                    "schema_version": "behavioral_arbitration_states_v1",
                    "dense_frame_idx": sample,
                    "sample_idx": sample,
                    "time_sec": t,
                    "focal_id": focal,
                    "partner_id": partner,
                    "behavior_label": f"lab_{sample}_f{focal}",
                    "label_reconstruction": "selected",
                    "undefined_reason": "",
                    "hierarchy_residual_kind": "",
                    "pred_contact": 0,
                    "pred_side_by_side": 0,
                }
            )
    states.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(states, index=False)
    behavior = _synthetic_behavior(
        tmp_path,
        video=pose.prepared_video_path,
        sync=behavior_sync,
        sync_df=behavior_sync_df,
    )
    # Replace states path/digest to the encoded-label table.
    behavior = OverlayQcSource(
        source_kind=behavior.source_kind,
        package_run_id=behavior.package_run_id,
        package_dir=behavior.package_dir,
        review_run_id=behavior.review_run_id,
        review_run_dir=behavior.review_run_dir,
        session_root=behavior.session_root,
        package=behavior.package,
        sync=behavior_sync,
        states_path=states,
        states_sha256=sha256_file(states),
        video_path=behavior.video_path,
        video_probe=behavior.video_probe,
        video_sha256=behavior.video_sha256,
        source_digests={
            **behavior.source_digests,
            "s8_behavior_states_sha256": sha256_file(states),
        },
        behavior_bouts=behavior.behavior_bouts,
        provenance=dict(behavior.provenance),
    )

    with (
        patch(
            "overlay.source_mixed.resolve_overlay_s4_qc_source",
            return_value=pose,
        ),
        patch(
            "overlay.source_mixed.resolve_overlay_qc_source",
            return_value=behavior,
        ),
    ):
        mixed = resolve_overlay_mixed_qc_source(
            pose_source_kind=PoseSourceKind.S4_FINALIZED,
            behavior_source_kind=BehaviorSourceKind.S8_AUTOMATED,
            s4_run_dir=pose.s4_run_dir,
            package_dir=behavior.package_dir,
            prepared_interval=(0, n_prepared - 1),
        )

    meta = {
        "pose_sync": pose_sync,
        "behavior_sync": behavior_sync,
        "pose_at_0": pose_at_0,
        "behavior_at_0": behavior_at_0,
        "n_prepared": n_prepared,
        "PlateauPolicy": PlateauPolicy,
        "collect_evidence": collect_evidence,
        "build_mixed_adapters": build_mixed_adapters,
        "resolve_prepared_frame_timing": resolve_prepared_frame_timing,
    }
    return mixed, meta


def test_independent_timing_authorities_irregular_plateau(tmp_path: Path) -> None:
    mixed, meta = _irregular_independent_maps(tmp_path)
    assert any("independent_dense_at_prep=" in n for n in mixed.compatibility_notes)
    assert any(
        "dense_sample_indices_not_forced_equal" in n for n in mixed.compatibility_notes
    )
    assert any(
        "pose_timing_unavailable_at_prepared_frame=2" in n
        for n in mixed.compatibility_notes
    )

    plan = plan_mixed_prepared_interval(
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=int(meta["n_prepared"]) - 1,
        source_fps=mixed.video_probe.fps,
        source_n_frames=mixed.video_probe.n_frames,
        pose_source_kind=PoseSourceKind.S4_FINALIZED,
    )
    adapters = meta["build_mixed_adapters"](mixed, plan=plan)

    # First prepared frame: pose and behavior dense indices differ and are applied.
    pose_timing = meta["resolve_prepared_frame_timing"](
        meta["pose_sync"], 0, plateau_policy=meta["PlateauPolicy"].LAST_IN_PLATEAU
    )
    evidence0 = meta["collect_evidence"](
        mixed.sync,
        0,
        adapters,
        selected_layers=plan.selected_layers,
        plateau_policy=meta["PlateauPolicy"].LAST_IN_PLATEAU,
        layer_timing={LayerId.S4_POSE_FINAL: pose_timing},
    )
    pose_obs = evidence0.get(LayerId.S4_POSE_FINAL)
    beh_obs = evidence0.get(LayerId.S8_EXCLUSIVE_A_TO_B)
    assert pose_obs is not None and pose_obs.available
    assert beh_obs is not None and beh_obs.available
    assert pose_obs.payload["dense_sample_idx"] == meta["pose_at_0"]
    assert beh_obs.payload["dense_sample_idx"] == meta["behavior_at_0"]
    assert meta["pose_at_0"] != meta["behavior_at_0"]
    assert beh_obs.payload["behavior_label"] == f"lab_{meta['behavior_at_0']}_f0"
    nose = next(p for p in pose_obs.payload["points"] if p.get("node") == "nose")
    assert abs(float(nose["x"]) - (float(meta["pose_at_0"]) + 0.25)) < 1e-6

    # Missing pose coverage at prepared 2; behavior still available independently.
    pose_gap = meta["resolve_prepared_frame_timing"](
        meta["pose_sync"], 2, plateau_policy=meta["PlateauPolicy"].LAST_IN_PLATEAU
    )
    assert pose_gap.available is False
    evidence2 = meta["collect_evidence"](
        mixed.sync,
        2,
        adapters,
        selected_layers=plan.selected_layers,
        plateau_policy=meta["PlateauPolicy"].LAST_IN_PLATEAU,
        layer_timing={LayerId.S4_POSE_FINAL: pose_gap},
    )
    assert evidence2.get(LayerId.S4_POSE_FINAL) is not None
    assert evidence2.get(LayerId.S4_POSE_FINAL).available is False  # type: ignore[union-attr]
    assert evidence2.get(LayerId.S8_EXCLUSIVE_A_TO_B).available is True  # type: ignore[union-attr]
    assert evidence2.timing.available is True

    # Last prepared frame still resolves both authorities.
    last = int(meta["n_prepared"]) - 1
    pose_last = meta["resolve_prepared_frame_timing"](
        meta["pose_sync"], last, plateau_policy=meta["PlateauPolicy"].LAST_IN_PLATEAU
    )
    evidence_last = meta["collect_evidence"](
        mixed.sync,
        last,
        adapters,
        selected_layers=plan.selected_layers,
        plateau_policy=meta["PlateauPolicy"].LAST_IN_PLATEAU,
        layer_timing={LayerId.S4_POSE_FINAL: pose_last},
    )
    assert evidence_last.get(LayerId.S4_POSE_FINAL).available is True  # type: ignore[union-attr]
    assert evidence_last.get(LayerId.S8_EXCLUSIVE_A_TO_B).available is True  # type: ignore[union-attr]
    assert (
        evidence_last.get(LayerId.S4_POSE_FINAL).payload["dense_sample_idx"]  # type: ignore[union-attr]
        != evidence_last.get(LayerId.S8_EXCLUSIVE_A_TO_B).payload[  # type: ignore[union-attr]
            "dense_sample_idx"
        ]
    )

    # Full export renders with independent authorities disclosed in notes.
    with (
        patch(
            "overlay.export_mixed.verify_overlay_mixed_qc_source_integrity",
            return_value=None,
        ),
        patch("overlay.export.probe_encoded_clip") as probe,
    ):
        probe.return_value = MagicMock(
            n_frames=plan.prepared_interval.frame_count,
            fps=mixed.video_probe.fps,
        )
        result = export_mixed_pose_behavior_qc_clip(
            source=mixed,
            plan=plan,
            job_dir=tmp_path / "job_irregular",
            verify_encoded_boundaries=False,
        )
    notes = " ".join(result.notes)
    assert "behavior_dense_sample_idx=" in notes
    assert "pose_dense_sample_idx=" in notes
    assert result.manifest.extra["source"]["pose_timing"] == (
        "s4_dense_via_pose_syncmap_last_in_plateau"
    )
    assert result.manifest.extra["source"]["behavior_timing"] == (
        "dense_via_behavior_syncmap_last_in_plateau"
    )
    assert result.frames_written == plan.prepared_interval.frame_count


def test_published_s10_mixed_export_via_controller(tmp_path: Path) -> None:
    """Isolated published S10 + synthetic S4 through the mixed controller path."""

    from tests.behavioral_review._fixtures import (
        build_mini_review_package,
        patch_s9_currently_eligible,
    )
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4
    from tests.overlay.test_export_s4_pose_qc import _write_s4_pose_for_sync

    from behavioral_review.contracts import (
        REVIEWED_BEHAVIOR_STATES_FILENAME,
        STATUS_ACCEPTED,
    )
    from behavioral_review.editing import open_editing_session
    from behavioral_review.pi_report.source import assert_not_draft_source
    from overlay.export_mixed import OverlayMixedQcExportResult
    from overlay.source_s4 import OverlayS4QcSource
    from ui.controllers.overlay_qc_export_controller import OverlayQcExportController
    from ui.controllers.overlay_qc_export_support import OverlayQcUiSettings

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    session_root = Path(built["session_root"])
    package_dir = Path(built["package_dir"])
    video = Path(built["video_path"])
    sync_df = built["sync"]
    sync = SyncMap.from_sync_index(sync_df)
    review_root = tmp_path / "review_drafts_isolated"
    changed_sample = 2
    reviewed_label = "nose_to_body"

    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            package_dir, require_video=True, require_s8_states=True
        )
        session.enter_editing_mode("c34_s10_fixture")
        draft = session.create_draft(review_root=review_root)
        draft_dir = draft.review_run_dir
        auto_states = pd.read_parquet(session.inspection.source.source_s8.states_path)
        auto_label = str(
            auto_states[
                (auto_states["sample_idx"] == changed_sample)
                & (auto_states["focal_id"] == 0)
            ].iloc[0]["behavior_label"]
        )
        assert auto_label != reviewed_label
        # Draft must be rejected before publication (behavior eligibility gate).
        with pytest.raises(Exception, match="[Dd]raft|working|publish"):
            assert_not_draft_source(draft_dir)
        with (
            patch(
                "overlay.source_mixed.resolve_overlay_s4_qc_source",
                return_value=MagicMock(
                    session_root=session_root,
                    prepared_video_path=video,
                    prepared_video_sha256="pending",
                    prepared_sync_sha256="pending",
                    video_probe=probe_video(video),
                    sync=sync,
                    source_digests={},
                    provenance={},
                    s4_run_id="pending",
                    s4_run_dir=session_root / "pose_finalization" / "pending",
                ),
            ),
            pytest.raises(Exception, match="[Dd]raft|working|publish|Refusing draft"),
        ):
            resolve_overlay_mixed_qc_source(
                pose_source_kind=PoseSourceKind.S4_FINALIZED,
                behavior_source_kind=BehaviorSourceKind.S10_REVIEWED,
                s4_run_dir=session_root / "pose_finalization" / "pending",
                review_run_dir=draft_dir,
                package_dir=package_dir,
            )

        session.set_exclusive_label(
            focal_id=0,
            partner_id=1,
            start_sample_idx=changed_sample,
            end_sample_idx=changed_sample,
            label=reviewed_label,
        )
        session.set_bout_review_status(
            target_bout_id="bout_dir_a", status=STATUS_ACCEPTED
        )
        published = session.publish_reviewed_run()
        assert published.success
        review_run_dir = Path(published.review_run_dir)
        review_run_id = str(published.review_run_id)
        reviewed_states = pd.read_parquet(
            review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        )
        reviewed_at = str(
            reviewed_states[
                (reviewed_states["sample_idx"] == changed_sample)
                & (reviewed_states["focal_id"] == 0)
            ].iloc[0]["behavior_label"]
        )
        assert reviewed_at == reviewed_label
        assert reviewed_at != auto_label

        # Synthetic S4 in the same D0 session; own SyncMap object (authoritative).
        s4_dir = session_root / "pose_finalization" / "pose_finalization__c34_s10"
        s4_dir.mkdir(parents=True, exist_ok=True)
        pose_path = _write_s4_pose_for_sync(s4_dir / "final_pose.parquet", sync)
        sync_path = session_root / "preprocess" / "prepared_sync.npz"
        if not sync_path.is_file():
            sync_path.write_bytes(b"synthetic-prepared-sync")
        probe = probe_video(video)
        pose_sha = sha256_file(pose_path)
        video_sha = sha256_file(video)
        sync_sha = sha256_file(sync_path)
        pose_s4 = OverlayS4QcSource(
            s4_run_id="pose_finalization__c34_s10",
            s4_run_dir=s4_dir,
            s4_status="completed",
            final_pose_path=pose_path,
            final_pose_sha256=pose_sha,
            session_root=session_root,
            prepared_sync_path=sync_path,
            prepared_sync_sha256=sync_sha,
            prepared_video_path=video,
            prepared_video_sha256=video_sha,
            video_probe=probe,
            sync=sync,
            sync_build_audit={"n_dense": int(sync.n_dense), "c34_s10_fixture": True},
            source_s3_run_id="tracking_correction__synth",
            source_s3_run_dir=session_root
            / "tracking_correction"
            / "tracking_correction__synth",
            source_s3_tracked_pose_sha256="s3sha",
            skeleton_edges=(("nose", "neck"), ("neck", "spine_base")),
            skeleton_source="synthetic",
            settings_used_path=None,
            run_meta={"run_id": "pose_finalization__c34_s10", "status": "completed"},
            source_digests={
                "s4_final_pose_sha256": pose_sha,
                "s1_prepared_sync_sha256": sync_sha,
                "s1_prepared_video_sha256": video_sha,
                "s3_tracked_pose_sha256": "s3sha",
            },
            provenance={"source_kind": "s4_finalized_pose", "synthetic": True},
        )

        upstream_pose = pose_path.read_bytes()
        upstream_reviewed = (
            review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        ).read_bytes()
        upstream_s8 = Path(built["s8_run_dir"]) / "behavior_states.parquet"
        upstream_s8_bytes = upstream_s8.read_bytes() if upstream_s8.is_file() else b""

        ctrl = OverlayQcExportController()
        ctrl.set_session(session_root)
        identity = ctrl.begin_export_launch(
            session_root=session_root,
            settings=OverlayQcUiSettings(
                export_family="mixed_pose_behavior",
                pose_source_kind="s4_finalized",
                source_kind="s10_reviewed",
                prepared_frame_idx_start=0,
                prepared_frame_idx_end=3,
                hud_mode="both",
                include_contact=True,
                include_sbs=True,
                draw_skeleton=True,
            ),
            package_dir=package_dir,
            package_run_id=str(built["package_run_id"]),
            review_run_dir=review_run_dir,
            review_run_id=review_run_id,
            review_is_draft=False,
            s4_run_dir=s4_dir,
            s4_run_id=pose_s4.s4_run_id,
        )
        assert identity.review_run_id == review_run_id
        assert identity.s4_run_id == pose_s4.s4_run_id

        with (
            patch(
                "overlay.source_mixed.resolve_overlay_s4_qc_source",
                return_value=pose_s4,
            ),
            patch(
                "overlay.source_mixed.verify_overlay_s4_qc_source_integrity",
                return_value=None,
            ),
            patch("overlay.export.probe_encoded_clip") as probe_enc,
        ):
            probe_enc.return_value = MagicMock(n_frames=4, fps=float(probe.fps))
            result = ctrl.execute_export_launch(identity)

        assert isinstance(result, OverlayMixedQcExportResult)
        assert result.frames_written == 4
        assert result.video_path.is_file()
        assert result.video_path.name == OVERLAY_QC_MIXED_VIDEO_FILENAME
        manifest = parse_overlay_artifact_manifest(
            json.loads(result.manifest_path.read_text(encoding="utf-8"))
        )
        extra = manifest.extra
        assert extra["pose_source_kind"] == PoseSourceKind.S4_FINALIZED.value
        assert extra["behavior_source_kind"] == BehaviorSourceKind.S10_REVIEWED.value
        assert extra["source"]["review_run_id"] == review_run_id
        assert extra["source"]["pose_run_id"] == pose_s4.s4_run_id
        assert extra["source"]["pose_timing"] == (
            "s4_dense_via_pose_syncmap_last_in_plateau"
        )
        assert reviewed_label in extra["observed_behavior_labels_in_clip"]
        # Upstream scientific trees must remain untouched.
        assert pose_path.read_bytes() == upstream_pose
        assert (
            review_run_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        ).read_bytes() == upstream_reviewed
        if upstream_s8_bytes:
            assert upstream_s8.read_bytes() == upstream_s8_bytes
        ctrl.record_export_success(result, expected_identity=identity)
