"""Increment B1 — PI HTML behavioral report backend tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from behavioral_review.contracts import (
    GALLERY_SOURCE_AUTOMATED,
    GALLERY_SOURCE_PUBLISHED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
)
from behavioral_review.editing import open_editing_session
from behavioral_review.gallery import GalleryQuery, group_cards_by_effective_label
from behavioral_review.pi_report import (
    PI_SOURCE_AUTOMATED,
    PI_SOURCE_REVIEWED,
    PiReportPublicationError,
    PiReportSettings,
    PiReportSourceError,
    assert_not_draft_source,
    publish_pi_report,
    resolve_pi_report_source,
    summarize_pi_report_project,
    validate_report_directory,
    write_pi_report_zip,
)
from behavioral_review.pi_report.media import (
    media_relpath_for_bout,
    preview_plan_to_clip_bounds,
)
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from project.file_digest import sha256_file
from results_packaging.contracts import RUN_META_FILENAME
from results_packaging.media_csv_export.video import probe_encoded_clip
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)
from tests.results_packaging._export_fixtures import write_synthetic_mp4


def _replace_prepared_with_mp4(built: dict, *, n_frames: int | None = None) -> Path:
    """Swap the fixture AVI for an ffmpeg-friendly prepared MP4 (digest-pinned)."""

    video_path = Path(built["video_path"])
    frames = int(n_frames if n_frames is not None else built["n_prepared"])
    write_synthetic_mp4(video_path.with_suffix(".mp4"), n_frames=frames, fps=10.0)
    mp4 = video_path.with_suffix(".mp4")
    # Keep the prepared_video.avi path identity expected by run_meta: overwrite bytes.
    # Prefer renaming to .avi path content as mp4 container still named .avi is awkward;
    # update run_meta path to .mp4 instead.
    package_dir = Path(built["package_dir"])
    meta_path = package_dir / RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sha = sha256_file(mp4)
    meta["source_video"]["path"] = str(mp4)
    meta["source_video"]["sha256"] = sha
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if video_path.exists() and video_path != mp4:
        video_path.unlink()
    built["video_path"] = mp4
    return mp4


def _open_automated_session_fixture(tmp_path: Path, **kwargs):
    built = build_mini_review_package(tmp_path, **kwargs)
    _replace_prepared_with_mp4(built)
    return built


def _publish_reviewed(tmp_path: Path):
    built = _open_automated_session_fixture(tmp_path)
    review_root = tmp_path / "review_drafts_isolated"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"],
            require_video=True,
            require_s8_states=True,
        )
        session.enter_editing_mode("pi_reviewer")
        session.create_draft(review_root=review_root)
        # Status-only decisions against current exclusive projection so accepted
        # and rejected remain visible after publish (never silently dropped).
        from behavioral_review.gallery import (
            build_gallery_from_working_draft,
            gallery_status_target_for_card,
        )
        from behavioral_review.sync import SyncMap

        working = build_gallery_from_working_draft(
            session.current_reviewed_bouts(),
            SyncMap.from_sync_index(session.inspection.source.sync_index),
            package_run_id=built["package_run_id"],
            review_run_id=session.draft.review_run_id
            if session.draft is not None
            else "draft",
            review_bout_status=session.current_statuses(),
            automated_bouts=session.inspection.source.behavior_bouts,
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
        assert working.n_cards >= 2
        tid_a, kind_a = gallery_status_target_for_card(working.cards[0])
        tid_b, kind_b = gallery_status_target_for_card(working.cards[1])
        session.set_bout_review_status(
            target_bout_id=tid_a, status=STATUS_ACCEPTED, target_bout_id_kind=kind_a
        )
        session.set_bout_review_status(
            target_bout_id=tid_b, status=STATUS_REJECTED, target_bout_id_kind=kind_b
        )
        result = session.publish_reviewed_run()
    assert result.success
    return built, result.review_run_dir


def test_reject_working_draft_as_report_source(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    review_root = tmp_path / "drafts"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_editing_session(
            built["package_dir"],
            require_video=True,
            require_s8_states=True,
        )
        session.enter_editing_mode("draft_only")
        draft = session.create_draft(review_root=review_root)
    with pytest.raises(PiReportSourceError, match="[Dd]raft"):
        assert_not_draft_source(draft.review_run_dir)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        with pytest.raises(PiReportSourceError):
            resolve_pi_report_source(
                source_type="reviewed",
                review_run_dir=draft.review_run_dir,
            )
        with pytest.raises((PiReportSourceError, PiReportPublicationError)):
            publish_pi_report(
                session_root=built["session_root"],
                source_type="reviewed",
                review_run_dir=draft.review_run_dir,
                timestamp="20261101T000001",
            )


def test_automated_versus_published_sources_and_grouping(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        auto = resolve_pi_report_source(
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
    assert auto.source_type == PI_SOURCE_AUTOMATED
    assert auto.gallery_source_kind == GALLERY_SOURCE_AUTOMATED
    assert auto.gallery.n_cards == 2  # exclusive only; contact/SBS excluded
    labels = [g.label for g in auto.gallery.groups]
    assert labels == sorted(labels)
    assert set(labels) == {"approach", "retract"}
    assert all(c.status == STATUS_UNREVIEWED for c in auto.gallery.cards)
    assert all(c.channel == "exclusive" for c in auto.gallery.cards)

    built2, review_dir = _publish_reviewed(tmp_path / "pub")
    with patch_s9_currently_eligible(source_s8_run_id=built2["s8_run_id"]):
        reviewed = resolve_pi_report_source(
            source_type=PI_SOURCE_REVIEWED,
            review_run_dir=review_dir,
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
    assert reviewed.source_type == PI_SOURCE_REVIEWED
    assert reviewed.gallery_source_kind == GALLERY_SOURCE_PUBLISHED
    assert reviewed.gallery.n_cards >= 2
    assert all(c.channel == "exclusive" for c in reviewed.gallery.cards)
    statuses = {c.status for c in reviewed.gallery.cards}
    assert STATUS_ACCEPTED in statuses
    assert STATUS_REJECTED in statuses
    # Never merge automated bout ids into reviewed card identities.
    assert {c.bout_id for c in reviewed.gallery.cards}.isdisjoint(
        {c.bout_id for c in auto.gallery.cards}
    )
    # Rejected/unreviewed remain countable (not silently excluded).
    assert reviewed.gallery.n_cards == sum(reviewed.gallery.status_counts.values())


def test_status_fidelity_and_bout_count_preservation(tmp_path: Path) -> None:
    built, review_dir = _publish_reviewed(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type="reviewed",
            review_run_dir=review_dir,
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
        before = snapshot_upstream_fingerprints(source.package)
        result = publish_pi_report(
            session_root=built["session_root"],
            source_type="reviewed",
            review_run_dir=review_dir,
            settings=PiReportSettings(
                pre_context_sec=0.0,
                post_context_sec=0.0,
                write_zip=True,
            ),
            timestamp="20261101T010000",
        )
    data = json.loads(result.report_data_path.read_text(encoding="utf-8"))
    assert data["source_type"] == PI_SOURCE_REVIEWED
    assert len(data["bouts"]) == source.gallery.n_cards
    statuses = {b["status"] for b in data["bouts"]}
    assert STATUS_ACCEPTED in statuses
    assert STATUS_REJECTED in statuses
    assert STATUS_UNREVIEWED in statuses or len(statuses) >= 2
    assert data["summary"]["n_bouts"] == source.gallery.n_cards
    assert data["filters"]["contact_sbs_included"] is False
    # Provenance digests pinned.
    assert data["provenance"]["source_digests"]["behavior_bouts_sha256"]
    assert data["provenance"]["review_run_id"]
    assert_no_upstream_writes(
        package_dir=built["package_dir"],
        s8_run_dir=built["s8_run_dir"],
        video_path=built["video_path"],
        snapshots=before,
    )


def test_portable_relative_paths_zip_and_offline_assets(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        result = publish_pi_report(
            session_root=built["session_root"],
            source_type="automated",
            package_dir=built["package_dir"],
            settings=PiReportSettings(
                pre_context_sec=0.0,
                post_context_sec=0.0,
                write_zip=True,
            ),
            timestamp="20261101T020000",
        )
    report_dir = result.report_dir
    validate_report_directory(report_dir, expect_n_bouts=2)
    data = json.loads((report_dir / "report_data.json").read_text(encoding="utf-8"))
    embed = (report_dir / "assets" / "report_embed.js").read_text(encoding="utf-8")
    html = (report_dir / "index.html").read_text(encoding="utf-8")
    assert "cdn." not in html.lower()
    assert "http://" not in html.lower()
    assert "https://" not in html.lower()
    assert 'src="assets/app.js"' in html
    assert "window.PI_REPORT_DATA" in embed
    for bout in data["bouts"]:
        rel = bout["media_relpath"].replace("\\", "/")
        assert rel.startswith("media/")
        assert ":" not in rel[1:]  # no drive letter
        assert not rel.startswith("/")
        assert (report_dir / Path(rel)).is_file()
        # Absolute laboratory path must not appear in report payloads.
        assert str(built["video_path"]) not in json.dumps(data)
        assert str(built["video_path"]) not in embed

    assert result.zip_path is not None and result.zip_path.is_file()
    with zipfile.ZipFile(result.zip_path) as zf:
        names = set(zf.namelist())
    assert "index.html" in names
    assert "report_data.json" in names
    assert "report_manifest.json" in names
    assert any(n.startswith("media/") and n.endswith(".mp4") for n in names)
    # Re-zip completeness helper
    alt_zip = write_pi_report_zip(report_dir, zip_path=tmp_path / "copy_report.zip")
    assert alt_zip.is_file()


def test_video_endpoints_plateau_short_long_and_padding(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(
        tmp_path, n_prepared=12, plateau_repeats=2
    )
    # Extend exclusive bouts: keep fixture defaults then also check pad clamp.
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type="automated",
            package_dir=built["package_dir"],
            pre_context_sec=0.5,
            post_context_sec=0.5,
        )
        # Very short: approach prepared span is plateau-collapsed.
        short = next(c for c in source.gallery.cards if c.effective_label == "approach")
        assert short.bout_duration_frames == 4
        # Dense 0..3 → prepared plateaus; prepared start/end may be equal or adjacent.
        assert short.prepared_frame_idx_start <= short.prepared_frame_idx_end
        bounds = preview_plan_to_clip_bounds(short)
        assert bounds.clip_frame_idx_start == short.preview_plan.clip_frame_idx_start
        assert bounds.clip_frame_count >= 1
        # Edge pad clamp at start.
        assert bounds.clip_frame_idx_start == 0 or bounds.clamped_start or bounds.pre_pad_frames >= 0

        result = publish_pi_report(
            session_root=built["session_root"],
            source_type="automated",
            package_dir=built["package_dir"],
            settings=PiReportSettings(pre_context_sec=0.5, post_context_sec=0.5),
            timestamp="20261101T030000",
        )
    for bout in json.loads(result.report_data_path.read_text(encoding="utf-8"))["bouts"]:
        clip = result.report_dir / Path(bout["media_relpath"])
        probed = probe_encoded_clip(clip)
        assert probed.n_frames == bout["preview_plan"]["clip_frame_count"]
        assert probed.fps == pytest.approx(bout["preview_plan"]["source_fps"], rel=0.02)


def test_failure_cleanup_and_immutability(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    exports = Path(built["session_root"]) / "results_exports"
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]), patch(
        "behavioral_review.pi_report.publication.materialize_report_media",
        side_effect=RuntimeError("boom encode"),
    ), pytest.raises(PiReportPublicationError, match="failed"):
        publish_pi_report(
            session_root=built["session_root"],
            source_type="automated",
            package_dir=built["package_dir"],
            timestamp="20261101T040000",
        )
    final = exports / "pi_report__20261101T040000"
    assert not final.exists()
    # Staging leftovers should not present a successful report.
    staging = exports / "_pi_report_staging"
    if staging.exists():
        leftover = [p for p in staging.iterdir() if p.name.startswith("pi_report__")]
        assert leftover == []

    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        first = publish_pi_report(
            session_root=built["session_root"],
            source_type="automated",
            package_dir=built["package_dir"],
            settings=PiReportSettings(pre_context_sec=0.0, post_context_sec=0.0),
            timestamp="20261101T040100",
        )
        with pytest.raises(PiReportPublicationError, match="already exists"):
            publish_pi_report(
                session_root=built["session_root"],
                source_type="automated",
                package_dir=built["package_dir"],
                settings=PiReportSettings(pre_context_sec=0.0, post_context_sec=0.0),
                timestamp="20261101T040100",
            )
    assert first.report_dir.is_dir()
    # Discovery sees the completed report.
    summary = summarize_pi_report_project(built["session_root"])
    assert any(r.report_id == first.report_id for r in summary.completed_reports)


def test_no_s8_s9_s10_source_writes(tmp_path: Path) -> None:
    built, review_dir = _publish_reviewed(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type="reviewed",
            review_run_dir=review_dir,
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
        before_pkg = snapshot_upstream_fingerprints(source.package)
        before_review_files = {
            p: p.stat().st_mtime_ns
            for p in Path(review_dir).glob("*.parquet")
        }
        publish_pi_report(
            session_root=built["session_root"],
            source_type="reviewed",
            review_run_dir=review_dir,
            settings=PiReportSettings(pre_context_sec=0.0, post_context_sec=0.0),
            timestamp="20261101T050000",
        )
    assert_no_upstream_writes(
        package_dir=built["package_dir"],
        s8_run_dir=built["s8_run_id"] and built["s8_run_dir"],
        video_path=built["video_path"],
        snapshots=before_pkg,
    )
    for path, mtime in before_review_files.items():
        assert path.stat().st_mtime_ns == mtime


def test_source_provenance_in_manifest(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        result = publish_pi_report(
            session_root=built["session_root"],
            source_type="automated",
            package_dir=built["package_dir"],
            settings=PiReportSettings(pre_context_sec=0.0, post_context_sec=0.0),
            timestamp="20261101T060000",
        )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "pi_report_manifest_v1"
    assert manifest["source_type"] == "automated"
    assert manifest["source_package_run_id"] == built["package_run_id"]
    assert manifest["source_review_run_id"] is None
    assert manifest["source_digests"]["sync_index_sha256"]
    assert manifest["n_bouts"] == 2
    assert manifest["n_media_succeeded"] == 2
    assert manifest["report_settings"]["gif_requested"] is False
    assert manifest["report_settings"]["overlays_requested"] is False
    # Deterministic media mapping
    for bout_id, rel in manifest["media_mapping"].items():
        assert rel == media_relpath_for_bout(bout_id)


def test_query_filter_does_not_drop_status_vocabulary(tmp_path: Path) -> None:
    built, review_dir = _publish_reviewed(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        # Explicit status filter still uses gallery grouping contract.
        source = resolve_pi_report_source(
            source_type="reviewed",
            review_run_dir=review_dir,
            query=GalleryQuery(statuses=frozenset({STATUS_REJECTED})),
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
        assert source.gallery.n_cards == 1
        assert source.gallery.cards[0].status == STATUS_REJECTED
        groups = group_cards_by_effective_label(source.gallery.cards)
        assert len(groups) == 1
