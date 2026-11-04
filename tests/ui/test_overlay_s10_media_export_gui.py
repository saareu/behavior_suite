"""C3-6 GUI settings for opt-in S10 reviewed overlay media / PI overlays."""

from __future__ import annotations

from pathlib import Path

import pytest

from overlay.contracts import REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND
from ui.controllers.pi_report_support import PiReportUiSettings, source_identity_summary


def test_pi_ui_default_overlays_off() -> None:
    settings = PiReportUiSettings().validated()
    assert settings.overlays_enabled is False
    pi = settings.to_pi_settings()
    assert pi.overlays_enabled is False
    summary = source_identity_summary(
        source_kind="reviewed",
        package_run_id="results_package__x",
        review_run_id="behavioral_review__x",
        settings=settings,
    )
    assert "overlays=off" in summary


def test_pi_ui_overlay_builds_reviewed_media_settings(tmp_path: Path) -> None:
    pose = tmp_path / "pose_finalization__explicit"
    pose.mkdir()
    settings = PiReportUiSettings(
        source_kind="reviewed",
        bout_scope="selected_bouts",
        selected_bout_ids=("bout_a",),
        overlays_enabled=True,
        overlay_include_behavior=True,
        overlay_include_pose=True,
        overlay_pose_run_dir=str(pose),
        pre_context_sec=0.25,
        post_context_sec=0.5,
    ).validated()
    media = settings.to_reviewed_overlay_media_settings()
    assert media.job_kind == REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND
    assert media.include_behavior is True
    assert media.include_pose is True
    assert media.pose_run_dir == str(pose.resolve())
    assert media.pre_context_sec == pytest.approx(0.25)
    summary = source_identity_summary(
        source_kind="reviewed",
        package_run_id="results_package__x",
        review_run_id="behavioral_review__x",
        settings=settings,
    )
    assert "overlays=" in summary
    assert "s10_reviewed" in summary
    assert "s4_finalized" in summary


def test_pi_ui_overlay_rejects_automated_source() -> None:
    with pytest.raises(ValueError, match="Published S10"):
        PiReportUiSettings(
            source_kind="automated",
            overlays_enabled=True,
            overlay_include_behavior=True,
        ).validated()


def test_pi_ui_overlay_requires_layer() -> None:
    with pytest.raises(ValueError, match="behavioral labels and/or S4"):
        PiReportUiSettings(
            source_kind="reviewed",
            overlays_enabled=True,
            overlay_include_behavior=False,
            overlay_include_pose=False,
        ).validated()


def test_page_overlay_widgets_default_off() -> None:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ui.pages.behavioral_review_page import BehavioralReviewPage

    _ = QApplication.instance() or QApplication([])
    page = BehavioralReviewPage()
    assert page.pi_overlay_check.isChecked() is False
    assert page.pi_overlay_behavior_check.isChecked() is True
    assert page.pi_overlay_pose_check.isChecked() is False
    assert page.pi_overlay_media_button is not None
