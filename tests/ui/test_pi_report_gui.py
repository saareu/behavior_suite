"""B2 GUI: PI report source selection, draft rejection, stale callbacks, discovery."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from behavioral_review.pi_report import (
    PI_SOURCE_AUTOMATED,
    PI_SOURCE_REVIEWED,
    REPORT_SCOPE_SELECTED,
    PiReportResult,
)
from ui.controllers.behavioral_review_controller import (
    BehavioralReviewController,
    ReviewRunOption,
    S9PackageOption,
)
from ui.controllers.pi_report_controller import PiReportController, PiReportUiError
from ui.controllers.pi_report_support import PiReportUiSettings
from ui.pages.behavioral_review_page import BehavioralReviewPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _s9_option(tmp_path: Path) -> S9PackageOption:
    package_dir = tmp_path / "results_package" / "results_package__demo"
    package_dir.mkdir(parents=True)
    return S9PackageOption(
        run_id="results_package__demo",
        run_dir=package_dir,
        historical_status="completed",
        classification="eligible",
        currently_eligible=True,
        n_bouts=3,
        n_dense_samples=100,
        label="results_package__demo",
        reasons=(),
    )


def _draft_option(tmp_path: Path) -> ReviewRunOption:
    run_dir = tmp_path / "behavioral_review" / "behavioral_review__draft"
    run_dir.mkdir(parents=True)
    return ReviewRunOption(
        run_id="behavioral_review__draft",
        run_dir=run_dir,
        classification="draft",
        currently_eligible=True,
        parent_package_run_id="results_package__demo",
        publication_state="draft",
        historical_status="running",
        superseded_by=None,
        label="draft",
        is_draft=True,
    )


def _published_option(tmp_path: Path) -> ReviewRunOption:
    run_dir = tmp_path / "behavioral_review" / "behavioral_review__pub"
    run_dir.mkdir(parents=True)
    return ReviewRunOption(
        run_id="behavioral_review__pub",
        run_dir=run_dir,
        classification="published",
        currently_eligible=True,
        parent_package_run_id="results_package__demo",
        publication_state="published",
        historical_status="completed",
        superseded_by=None,
        label="published",
        is_draft=False,
    )


def test_gui_source_selection_and_draft_rejection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    review = BehavioralReviewController()
    review.state.session_root = session
    review.state.s9_options = (_s9_option(session),)
    review.state.selected_s9_run_id = "results_package__demo"
    review.state.review_options = (_draft_option(session), _published_option(session))
    review.state.selected_review_run_id = "behavioral_review__draft"

    pi = PiReportController()
    page = BehavioralReviewPage(review, pi_report_controller=pi)
    page.show()
    page.refresh_from_state()

    # Automated source is available.
    idx = page.pi_source_combo.findData(PI_SOURCE_AUTOMATED)
    page.pi_source_combo.setCurrentIndex(idx)
    page._sync_pi_source_identity()
    assert "package=results_package__demo" in page.pi_source_identity_label.text()

    # Draft is rejected when reviewed source is chosen.
    idx = page.pi_source_combo.findData(PI_SOURCE_REVIEWED)
    page.pi_source_combo.setCurrentIndex(idx)
    page._sync_pi_source_identity()
    assert "working draft" in page.pi_source_identity_label.text().lower()

    reason = pi.cannot_export_reason(
        session_root=session,
        source_kind=PI_SOURCE_REVIEWED,
        package_dir=review.selected_s9_option().run_dir if review.selected_s9_option() else None,
        package_run_id="results_package__demo",
        review_run_dir=review.selected_review_option().run_dir
        if review.selected_review_option()
        else None,
        review_run_id="behavioral_review__draft",
        review_is_draft=True,
        settings=PiReportUiSettings(source_kind="reviewed"),
    )
    assert reason is not None
    assert "draft" in reason.lower()


def test_selected_bout_scope_requires_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        PiReportUiSettings(
            bout_scope="selected_bouts", selected_bout_ids=()
        ).validated()
    settings = PiReportUiSettings(
        bout_scope="selected_bouts",
        selected_bout_ids=("bout_a", "bout_b"),
    ).validated()
    assert settings.report_scope_label() == REPORT_SCOPE_SELECTED
    assert settings.to_gallery_query() is not None
    assert settings.to_gallery_query().bout_ids == frozenset({"bout_a", "bout_b"})


def test_stale_callback_rejection_and_launch_conflict(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package_dir = session / "pkg"
    package_dir.mkdir()
    pi = PiReportController(
        publish_runner=MagicMock(
            return_value=PiReportResult(
                report_id="pi_report__x",
                report_dir=session / "results_exports" / "pi_report__x",
                zip_path=None,
                source_type=PI_SOURCE_AUTOMATED,
                package_run_id="pkg",
                review_run_id=None,
                n_bouts=1,
                n_media_succeeded=1,
                n_media_failed=0,
                manifest_path=session / "m.json",
                report_data_path=session / "d.json",
                qc_path=session / "q.json",
                index_html_path=session / "index.html",
            )
        )
    )
    identity = pi.begin_export_launch(
        session_root=session,
        source_kind=PI_SOURCE_AUTOMATED,
        package_dir=package_dir,
        package_run_id="pkg",
        review_run_dir=None,
        review_run_id=None,
        settings=PiReportUiSettings(write_zip=False),
    )
    assert pi.state.task_running is True
    with pytest.raises(PiReportUiError, match="already running"):
        pi.begin_export_launch(
            session_root=session,
            source_kind=PI_SOURCE_AUTOMATED,
            package_dir=package_dir,
            package_run_id="pkg",
            review_run_dir=None,
            review_run_id=None,
            settings=PiReportUiSettings(write_zip=False),
        )
    # Stale success after identity cleared / superseded.
    pi.state.active_identity = None
    assert (
        pi.record_export_success(
            PiReportResult(
                report_id="pi_report__stale",
                report_dir=session / "results_exports" / "pi_report__stale",
                zip_path=None,
                source_type=PI_SOURCE_AUTOMATED,
                package_run_id="pkg",
                review_run_id=None,
                n_bouts=1,
                n_media_succeeded=1,
                n_media_failed=0,
                manifest_path=session / "m.json",
                report_data_path=session / "d.json",
                qc_path=session / "q.json",
                index_html_path=session / "index.html",
            ),
            expected_identity=identity,
        )
        is False
    )


def test_progress_cancel_flag_and_discovery_rows(tmp_path: Path) -> None:
    session = tmp_path / "session"
    exports = session / "results_exports"
    report_dir = exports / "pi_report__done"
    report_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (report_dir / "report_manifest.json").write_text(
        '{"schema_version":"pi_report_manifest_v1","n_bouts":2,'
        '"source_type":"automated","report_scope":"selected_bouts",'
        '"status":"completed"}',
        encoding="utf-8",
    )
    (report_dir / "run_meta.json").write_text(
        '{"status":"completed","source_type":"automated",'
        '"counts":{"n_bouts":2,"report_scope":"selected_bouts"}}',
        encoding="utf-8",
    )
    (exports / "pi_report__done.zip").write_bytes(b"PK")

    pi = PiReportController()
    summary = pi.refresh_discovery(session)
    assert summary is not None
    assert any(r.report_id == "pi_report__done" for r in summary.reports)
    rows = pi.report_table_rows()
    assert any(row[0] == "pi_report__done" for row in rows)
    hit = next(r for r in summary.reports if r.report_id == "pi_report__done")
    assert hit.has_zip is True
    assert hit.report_scope == REPORT_SCOPE_SELECTED

    # Cancellation path records canceled outcome without publishing a folder.
    package_dir = session / "pkg"
    package_dir.mkdir()
    identity = pi.begin_export_launch(
        session_root=session,
        source_kind=PI_SOURCE_AUTOMATED,
        package_dir=package_dir,
        package_run_id="pkg",
        review_run_dir=None,
        review_run_id=None,
        settings=PiReportUiSettings(write_zip=False),
    )
    assert pi.record_export_failure(
        "cancelled by operator", expected_identity=identity, canceled=True
    )
    assert pi.state.last_outcome == "canceled"
    assert pi.state.task_running is False
