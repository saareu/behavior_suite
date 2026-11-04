"""C3-6 S10 reviewed overlay media: sibling family, eligibility, PI defaults."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)

from behavioral_review.contracts import (
    REVIEWED_BEHAVIOR_STATES_FILENAME,
    STATUS_ACCEPTED,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.sync import SyncMap
from overlay.contracts import (
    OVERLAY_S10_MEDIA_RENDERER_VERSION,
    REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND,
    REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_PREFIX,
    BehaviorSourceKind,
    PoseSourceKind,
)
from overlay.export_s10_media import (
    ReviewedOverlayMediaExportRequest,
    ReviewedOverlayMediaExportSettings,
    resolve_reviewed_overlay_media_sources,
    run_reviewed_overlay_media_export,
)
from overlay.source_s4 import OverlayS4QcSource
from project.file_digest import sha256_file
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import (
    CLIP_INDEX_FILENAME,
    ENCODE_STATUS_SUCCEEDED,
    SELECTION_BOUT_ID_LIST,
    ExportConfigError,
)
from results_packaging.media_csv_export.discovery import summarize_export_project
from results_packaging.media_csv_export.video import probe_video
from ui.controllers.pi_report_support import PiReportUiSettings


def _publish_reviewed_with_label_change(tmp_path: Path):
    """Build package + publish. Returns with eligibility still needing to be patched."""

    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    return _publish_reviewed_with_label_change_from_built(built, tmp_path)


def _publish_reviewed_with_label_change_from_built(built: dict, tmp_path: Path):
    """Publish under an already-active ``patch_s9_currently_eligible`` context."""

    package_dir = Path(built["package_dir"])
    review_root = tmp_path / "review_drafts_isolated"
    changed_sample = 2
    reviewed_label = "nose_to_body"
    session = open_editing_session(
        package_dir, require_video=True, require_s8_states=True
    )
    session.enter_editing_mode("c36_s10_fixture")
    session.create_draft(review_root=review_root)
    auto_states = pd.read_parquet(session.inspection.source.source_s8.states_path)
    auto_label = str(
        auto_states[
            (auto_states["sample_idx"] == changed_sample)
            & (auto_states["focal_id"] == 0)
        ].iloc[0]["behavior_label"]
    )
    assert auto_label != reviewed_label
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
    return built, Path(published.review_run_dir), auto_label, reviewed_label, changed_sample


def test_reviewed_overlay_settings_defaults_and_validation(tmp_path: Path) -> None:
    cfg = ReviewedOverlayMediaExportSettings().validated()
    assert cfg.job_kind == REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND
    assert cfg.include_behavior is True
    assert cfg.include_pose is False
    assert cfg.to_settings_dict()["behavior_source_kind"] == (
        BehaviorSourceKind.S10_REVIEWED.value
    )
    assert cfg.to_settings_dict()["s8_label_fallback"] is False
    with pytest.raises(ExportConfigError, match="include_behavior and/or include_pose"):
        ReviewedOverlayMediaExportSettings(
            include_behavior=False, include_pose=False
        ).validated()
    with pytest.raises(ExportConfigError, match="S3\\+S10 is unsupported"):
        ReviewedOverlayMediaExportSettings(
            include_behavior=False,
            include_pose=True,
            pose_source_kind="s3_accepted",
            pose_run_dir=str(tmp_path / "s3"),
        ).validated()


def test_pi_defaults_remain_naked_without_overlays() -> None:
    from behavioral_review.pi_report.publication import PiReportSettings

    settings = PiReportSettings()
    assert settings.overlays_enabled is False
    payload = settings.to_dict()
    assert payload["overlays_requested"] is False
    assert payload["overlays_enabled"] is False
    ui = PiReportUiSettings().validated()
    assert ui.overlays_enabled is False
    assert ui.to_pi_settings().overlays_enabled is False


def test_pi_ui_overlays_require_published_review(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Published S10"):
        PiReportUiSettings(
            source_kind="automated",
            overlays_enabled=True,
            overlay_include_behavior=True,
        ).validated()
    pose = tmp_path / "pose_finalization__x"
    pose.mkdir()
    settings = PiReportUiSettings(
        source_kind="reviewed",
        overlays_enabled=True,
        overlay_include_behavior=True,
        overlay_include_pose=True,
        overlay_pose_run_dir=str(pose),
    ).validated()
    media = settings.to_reviewed_overlay_media_settings()
    assert media.job_kind == REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND
    assert media.include_pose is True
    assert media.pose_source_kind == PoseSourceKind.S4_FINALIZED.value


def test_draft_rejected_and_no_s8_fallback(tmp_path: Path) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    review_root = tmp_path / "drafts"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"], require_video=True, require_s8_states=True
        )
        session.enter_editing_mode("draft_only")
        draft = session.create_draft(review_root=review_root)
        with pytest.raises(Exception, match="[Dd]raft|working|publish"):
            resolve_reviewed_overlay_media_sources(
                review_run_dir=draft.review_run_dir,
                settings=ReviewedOverlayMediaExportSettings().validated(),
            )


def test_behavior_only_reviewed_export_preserves_bout_and_label(
    tmp_path: Path,
) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        built2, review_dir, auto_label, reviewed_label, sample = (
            _publish_reviewed_with_label_change_from_built(built, tmp_path)
        )
        reviewed_states = pd.read_parquet(
            review_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        )
        reviewed_at = str(
            reviewed_states[
                (reviewed_states["sample_idx"] == sample)
                & (reviewed_states["focal_id"] == 0)
            ].iloc[0]["behavior_label"]
        )
        assert reviewed_at == reviewed_label
        assert reviewed_at != auto_label

        sources = resolve_reviewed_overlay_media_sources(
            review_run_dir=review_dir,
            settings=ReviewedOverlayMediaExportSettings(
                include_behavior=True, include_pose=False
            ).validated(),
        )
        assert sources.mode == "behavior"
        assert sources.behavior is not None
        assert sources.behavior.source_kind is BehaviorSourceKind.S10_REVIEWED
        bout_id = str(sources.reviewed_bouts.iloc[0]["bout_id"])
        request = ReviewedOverlayMediaExportRequest(
            session_root=Path(built2["session_root"]),
            review_run_dir=review_dir,
            settings=ReviewedOverlayMediaExportSettings(
                selection=SelectionSpec(
                    mode=SELECTION_BOUT_ID_LIST, bout_ids=(bout_id,)
                ),
                include_behavior=True,
                include_pose=False,
                pre_context_sec=0.0,
                post_context_sec=0.0,
            ),
            timestamp="20261104T220001",
        )
        before_states = sha256_file(review_dir / REVIEWED_BEHAVIOR_STATES_FILENAME)
        before_video = sha256_file(Path(built2["video_path"]))
        result = run_reviewed_overlay_media_export(request)
        assert result.success
        assert result.export_job_id.startswith(REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_PREFIX)
        assert result.n_succeeded >= 1
        clip_index = pd.read_parquet(result.job_dir / CLIP_INDEX_FILENAME)
        assert str(clip_index.iloc[0]["bout_id"]) == bout_id
        assert str(clip_index.iloc[0]["encode_status"]) == ENCODE_STATUS_SUCCEEDED
        clips = list((result.job_dir / "clips").rglob("*.mp4"))
        assert clips
        rel = str(clip_index.iloc[0]["clip_relpath"]).replace("\\", "/")
        assert not Path(rel).is_absolute()
        manifest = (result.job_dir / "export_job_manifest.json").read_text(
            encoding="utf-8"
        )
        assert BehaviorSourceKind.S10_REVIEWED.value in manifest
        assert "s8_label_fallback" in manifest
        assert OVERLAY_S10_MEDIA_RENDERER_VERSION in manifest
        assert (
            sha256_file(review_dir / REVIEWED_BEHAVIOR_STATES_FILENAME) == before_states
        )
        assert sha256_file(Path(built2["video_path"])) == before_video

        summary = summarize_export_project(built2["session_root"])
        assert any(
            j.job_kind == REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND for j in summary.jobs
        )


def test_s4_plus_reviewed_compatibility_and_independent_timing(tmp_path: Path) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4
    from tests.overlay.test_export_s4_pose_qc import _write_s4_pose_for_sync

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        built, review_dir, _auto, _rev, _sample = (
            _publish_reviewed_with_label_change_from_built(built, tmp_path)
        )
        session_root = Path(built["session_root"])
        video = Path(built["video_path"])
        sync = SyncMap.from_sync_index(built["sync"])
        s4_dir = session_root / "pose_finalization" / "pose_finalization__c36"
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
            s4_run_id="pose_finalization__c36",
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
            sync_build_audit={"n_dense": int(sync.n_dense), "c36_fixture": True},
            source_s3_run_id="tracking_correction__synth",
            source_s3_run_dir=session_root
            / "tracking_correction"
            / "tracking_correction__synth",
            source_s3_tracked_pose_sha256="s3sha",
            skeleton_edges=(("nose", "neck"),),
            skeleton_source="synthetic",
            settings_used_path=None,
            run_meta={"run_id": "pose_finalization__c36", "status": "completed"},
            source_digests={
                "s4_final_pose_sha256": pose_sha,
                "s1_prepared_sync_sha256": sync_sha,
                "s1_prepared_video_sha256": video_sha,
                "s3_tracked_pose_sha256": "s3sha",
            },
            provenance={"source_kind": "s4_finalized_pose", "synthetic": True},
        )
        with patch(
            "overlay.export_s10_media.resolve_overlay_s4_qc_source",
            return_value=pose_s4,
        ), patch(
            "overlay.source_mixed.resolve_overlay_s4_qc_source",
            return_value=pose_s4,
        ), patch(
            "overlay.export_s10_media.verify_overlay_s4_qc_source_integrity",
        ), patch(
            "overlay.source_mixed.verify_overlay_s4_qc_source_integrity",
        ), patch(
            "overlay.export_s10_media.verify_overlay_mixed_qc_source_integrity",
        ):
            sources = resolve_reviewed_overlay_media_sources(
                review_run_dir=review_dir,
                settings=ReviewedOverlayMediaExportSettings(
                    include_behavior=True,
                    include_pose=True,
                    pose_source_kind=PoseSourceKind.S4_FINALIZED.value,
                    pose_run_dir=str(s4_dir),
                ).validated(),
            )
            assert sources.mode == "mixed_s4"
            assert sources.mixed is not None
            assert sources.mixed.behavior_source_kind is BehaviorSourceKind.S10_REVIEWED
            assert "dual_independent" in sources.sync_authority
            bout_id = str(sources.reviewed_bouts.iloc[0]["bout_id"])
            result = run_reviewed_overlay_media_export(
                ReviewedOverlayMediaExportRequest(
                    session_root=session_root,
                    review_run_dir=review_dir,
                    settings=ReviewedOverlayMediaExportSettings(
                        selection=SelectionSpec(
                            mode=SELECTION_BOUT_ID_LIST, bout_ids=(bout_id,)
                        ),
                        include_behavior=True,
                        include_pose=True,
                        pose_source_kind=PoseSourceKind.S4_FINALIZED.value,
                        pose_run_dir=str(s4_dir),
                    ),
                    timestamp="20261104T220002",
                )
            )
        assert result.success, result.error_message
        assert result.diagnostics["behavior_source_kind"] == (
            BehaviorSourceKind.S10_REVIEWED.value
        )


def test_digest_mismatch_rejected(tmp_path: Path) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    from overlay.source import verify_overlay_qc_source_integrity

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        _built, review_dir, *_ = _publish_reviewed_with_label_change_from_built(
            built, tmp_path
        )
        sources = resolve_reviewed_overlay_media_sources(
            review_run_dir=review_dir,
            settings=ReviewedOverlayMediaExportSettings().validated(),
        )
        assert sources.behavior is not None
        states = review_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
        states.write_bytes(states.read_bytes() + b"\x00tamper")
        with pytest.raises(Exception, match="digest"):
            verify_overlay_qc_source_integrity(sources.behavior)


def test_pi_report_naked_default_unchanged_for_reviewed(tmp_path: Path) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    from behavioral_review.pi_report.publication import PiReportSettings, publish_pi_report

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        built, review_dir, *_ = _publish_reviewed_with_label_change_from_built(
            built, tmp_path
        )
        result = publish_pi_report(
            session_root=built["session_root"],
            source_type="reviewed",
            review_run_dir=review_dir,
            settings=PiReportSettings(write_zip=False, overlays_enabled=False),
            timestamp="20261104T220010",
        )
    assert result.n_media_succeeded >= 1
    manifest = (result.report_dir / "report_manifest.json").read_text(encoding="utf-8")
    assert "overlays_requested" in manifest.lower()
    assert "true" not in manifest.lower().split("overlays_requested")[1][:40]
    assert "annotated_video_previews" not in manifest


def test_pi_report_overlay_requires_reviewed_source(tmp_path: Path) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    from behavioral_review.pi_report.publication import PiReportSettings, publish_pi_report

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    with (
        patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]),
        pytest.raises(Exception, match="published S10|reviewed"),
    ):
        publish_pi_report(
            session_root=built["session_root"],
            source_type="automated",
            package_dir=built["package_dir"],
            settings=PiReportSettings(
                write_zip=False,
                overlays_enabled=True,
                overlay_include_behavior=True,
            ),
            timestamp="20261104T220011",
        )


def test_cancellation_cleans_staging(tmp_path: Path) -> None:
    from tests.behavioral_review.test_pi_report_b1 import _replace_prepared_with_mp4

    built = build_mini_review_package(tmp_path, n_prepared=8, plateau_repeats=2)
    _replace_prepared_with_mp4(built)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        built, review_dir, *_ = _publish_reviewed_with_label_change_from_built(
            built, tmp_path
        )
        sources = resolve_reviewed_overlay_media_sources(
            review_run_dir=review_dir,
            settings=ReviewedOverlayMediaExportSettings().validated(),
        )
        bout_ids = tuple(str(b) for b in sources.reviewed_bouts["bout_id"].tolist()[:2])
        assert bout_ids
        canceled = {"n": 0}

        def _cancel() -> bool:
            canceled["n"] += 1
            return canceled["n"] > 1

        result = run_reviewed_overlay_media_export(
            ReviewedOverlayMediaExportRequest(
                session_root=Path(built["session_root"]),
                review_run_dir=review_dir,
                settings=ReviewedOverlayMediaExportSettings(
                    selection=SelectionSpec(
                        mode=SELECTION_BOUT_ID_LIST, bout_ids=bout_ids
                    ),
                    include_behavior=True,
                    include_pose=False,
                ),
                timestamp="20261104T220012",
                cancellation_requested=_cancel,
            )
        )
    assert result.status in {"incomplete", "failed", "completed_with_warnings"}
    assert not (result.job_dir / "_reviewed_overlay_media_staging").exists()
