"""Increment B2 — PI report selected-bout scope, cancel, integrity, cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from behavioral_review.gallery import GalleryQuery
from behavioral_review.pi_report import (
    PI_SOURCE_AUTOMATED,
    REPORT_SCOPE_SELECTED,
    STATUS_CANCELED,
    PiReportCancelledError,
    PiReportPublicationError,
    PiReportSettings,
    PiReportSourceError,
    publish_pi_report,
    summarize_pi_report_project,
    verify_pinned_source_integrity,
)
from behavioral_review.pi_report.media import materialize_report_media
from behavioral_review.pi_report.source import resolve_pi_report_source
from results_packaging.contracts import RUN_META_FILENAME
from results_packaging.media_csv_export.video import encode_mp4_clip
from tests.behavioral_review._fixtures import patch_s9_currently_eligible
from tests.behavioral_review.test_pi_report_b1 import (
    _open_automated_session_fixture,
    _publish_reviewed,
)


def test_selected_bout_scope_recorded_and_labeled(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            require_video=True,
        )
        assert source.gallery.n_cards >= 1
        chosen = source.gallery.cards[0].bout_id
        result = publish_pi_report(
            session_root=built["session_root"],
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            settings=PiReportSettings(
                write_zip=True, pre_context_sec=0.0, post_context_sec=0.0
            ),
            query=GalleryQuery(bout_ids=frozenset({chosen})),
            report_id="pi_report__selected_scope",
        )
    assert result.n_bouts == 1
    data = json.loads(result.report_data_path.read_text(encoding="utf-8"))
    assert data["report_scope"] == REPORT_SCOPE_SELECTED
    assert data["subset_of_source"] is True
    assert data["selected_bout_ids"] == [chosen]
    assert "SUBSET REPORT" in (
        result.index_html_path.parent / "assets" / "app.js"
    ).read_text(encoding="utf-8")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["report_scope"] == REPORT_SCOPE_SELECTED
    assert manifest["selected_bout_ids"] == [chosen]
    assert manifest["selection"]["bout_ids"] == [chosen]
    html = result.index_html_path.read_text(encoding="utf-8")
    assert 'id="scope-banner"' in html


def test_empty_selected_bout_scope_rejected(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with (
        patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]),
        pytest.raises(PiReportPublicationError, match="no eligible exclusive"),
    ):
        publish_pi_report(
            session_root=built["session_root"],
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            settings=PiReportSettings(
                write_zip=False, pre_context_sec=0.0, post_context_sec=0.0
            ),
            query=GalleryQuery(bout_ids=frozenset({"missing_bout"})),
            report_id="pi_report__empty_selection",
        )


def test_cooperative_cancel_cleans_incomplete_export(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    state = {"cancel": False}

    def cancel_requested() -> bool:
        return bool(state["cancel"])

    def progress(phase: str, done: int, total: int) -> None:
        if phase == "encode_media" and done >= 1:
            state["cancel"] = True

    with (
        patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]),
        pytest.raises(PiReportCancelledError, match="cancel"),
    ):
        publish_pi_report(
            session_root=built["session_root"],
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            settings=PiReportSettings(
                write_zip=False, pre_context_sec=0.0, post_context_sec=0.0
            ),
            report_id="pi_report__cancel_mid",
            progress_callback=progress,
            cancellation_requested=cancel_requested,
        )
    exports = Path(built["session_root"]) / "results_exports"
    assert not (exports / "pi_report__cancel_mid").exists()
    staging = exports / "_pi_report_staging"
    if staging.exists():
        leftovers = [
            p for p in staging.iterdir() if p.name.startswith("pi_report__cancel_mid")
        ]
        assert leftovers == []
    summary = summarize_pi_report_project(built["session_root"])
    assert all(r.report_id != "pi_report__cancel_mid" for r in summary.reports)


def test_source_digest_mismatch_rejects_before_publish(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            require_video=True,
        )
        broken = source
        object.__setattr__(
            broken,
            "source_digests",
            {**source.source_digests, "behavior_bouts_sha256": "0" * 64},
        )
        with pytest.raises(PiReportSourceError, match="digest mismatch"):
            verify_pinned_source_integrity(broken)

        with patch(
            "behavioral_review.pi_report.publication.verify_pinned_source_integrity",
            side_effect=PiReportSourceError("digest mismatch for test"),
        ), pytest.raises(PiReportPublicationError, match="digest mismatch"):
            publish_pi_report(
                session_root=built["session_root"],
                source_type=PI_SOURCE_AUTOMATED,
                package_dir=built["package_dir"],
                settings=PiReportSettings(
                    write_zip=False, pre_context_sec=0.0, post_context_sec=0.0
                ),
                report_id="pi_report__digest_reject",
            )
    assert not (
        Path(built["session_root"]) / "results_exports" / "pi_report__digest_reject"
    ).exists()


def test_no_upstream_writes_on_selected_and_cancel_paths(tmp_path: Path) -> None:
    built, review_dir = _publish_reviewed(tmp_path)
    package_dir = Path(built["package_dir"])
    before = {
        p.name: p.stat().st_mtime_ns for p in package_dir.iterdir() if p.is_file()
    }
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type="reviewed",
            review_run_dir=review_dir,
            require_video=True,
        )
        chosen = source.gallery.cards[0].bout_id
        publish_pi_report(
            session_root=built["session_root"],
            source_type="reviewed",
            review_run_dir=review_dir,
            settings=PiReportSettings(
                write_zip=False, pre_context_sec=0.0, post_context_sec=0.0
            ),
            query=GalleryQuery(bout_ids=frozenset({chosen})),
            report_id="pi_report__reviewed_subset",
        )
    after = {
        p.name: p.stat().st_mtime_ns for p in package_dir.iterdir() if p.is_file()
    }
    assert before == after
    assert (Path(review_dir) / RUN_META_FILENAME).is_file()


def test_s9_encoder_path_still_used_for_previews(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        full = resolve_pi_report_source(
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            require_video=True,
        )
        source = resolve_pi_report_source(
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            require_video=True,
            query=GalleryQuery(bout_ids=frozenset({full.gallery.cards[0].bout_id})),
        )
        out = tmp_path / "media_root"
        out.mkdir()
        with patch(
            "behavioral_review.pi_report.media.encode_mp4_clip",
            wraps=encode_mp4_clip,
        ) as mocked:
            materialize_report_media(
                cards=source.gallery.cards[:1],
                source_video=source.video_path,
                report_root=out,
            )
            assert mocked.called


def test_discovery_lists_completed_reports_with_scope(tmp_path: Path) -> None:
    built = _open_automated_session_fixture(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = resolve_pi_report_source(
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            require_video=True,
        )
        chosen = source.gallery.cards[0].bout_id
        publish_pi_report(
            session_root=built["session_root"],
            source_type=PI_SOURCE_AUTOMATED,
            package_dir=built["package_dir"],
            settings=PiReportSettings(
                write_zip=True, pre_context_sec=0.0, post_context_sec=0.0
            ),
            query=GalleryQuery(bout_ids=frozenset({chosen})),
            report_id="pi_report__discover_me",
        )
    summary = summarize_pi_report_project(built["session_root"])
    assert any(r.report_id == "pi_report__discover_me" for r in summary.completed_reports)
    hit = next(r for r in summary.reports if r.report_id == "pi_report__discover_me")
    assert hit.report_scope == REPORT_SCOPE_SELECTED
    assert hit.has_zip is True
    assert hit.status in {"completed", "completed_with_warnings"}
    assert hit.status != STATUS_CANCELED
