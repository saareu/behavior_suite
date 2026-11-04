"""C3-5 GUI settings for opt-in S9 overlay media export."""

from __future__ import annotations

from pathlib import Path

import pytest

from overlay.contracts import MEDIA_EXPORT_OVERLAY_JOB_KIND
from results_packaging.media_csv_export.contracts import (
    JOB_KIND_MEDIA,
    ExportConfigError,
)
from ui.controllers.results_export_support import (
    ExportUiSettings,
    export_provenance_summary,
)


def test_default_export_ui_is_naked_media() -> None:
    settings = ExportUiSettings().validated()
    assert settings.overlay_enabled is False
    assert settings.to_media_settings().job_kind == JOB_KIND_MEDIA
    summary = export_provenance_summary(
        package_run_id="results_package__x",
        settings=settings,
        n_selected=3,
    )
    assert "media_family=media_export" in summary
    assert "media_export_overlay" not in summary


def test_overlay_ui_settings_build_sibling_job_settings(tmp_path: Path) -> None:
    pose = tmp_path / "pose_finalization__explicit"
    pose.mkdir()
    settings = ExportUiSettings(
        overlay_enabled=True,
        overlay_include_behavior=True,
        overlay_include_pose=True,
        overlay_pose_source_kind="s4_finalized",
        overlay_pose_run_dir=str(pose),
        overlay_hud_mode="both",
        overlay_include_contact=True,
        overlay_include_sbs=True,
        overlay_draw_skeleton=True,
        pre_context_sec=0.1,
        post_context_sec=0.2,
    ).validated()
    overlay = settings.to_overlay_media_settings()
    assert overlay.job_kind == MEDIA_EXPORT_OVERLAY_JOB_KIND
    assert overlay.include_behavior is True
    assert overlay.include_pose is True
    assert overlay.pose_source_kind == "s4_finalized"
    assert overlay.pose_run_dir == str(pose.resolve())
    assert overlay.pre_context_sec == pytest.approx(0.1)
    assert overlay.post_context_sec == pytest.approx(0.2)
    summary = export_provenance_summary(
        package_run_id="results_package__x",
        settings=settings,
        n_selected=2,
    )
    assert "media_family=media_export_overlay" in summary
    assert "s8_automated" in summary
    assert "s4_finalized" in summary


def test_overlay_ui_rejects_pose_without_explicit_run() -> None:
    with pytest.raises(ExportConfigError, match="explicitly selected pose run"):
        ExportUiSettings(
            overlay_enabled=True,
            overlay_include_behavior=False,
            overlay_include_pose=True,
            overlay_pose_source_kind="s3_accepted",
        ).validated()


def test_overlay_ui_rejects_empty_layer_selection() -> None:
    with pytest.raises(ExportConfigError, match="behavioral labels and/or pose"):
        ExportUiSettings(
            overlay_enabled=True,
            overlay_include_behavior=False,
            overlay_include_pose=False,
        ).validated()


def test_page_behavior_only_overlay_ignores_stale_pose_combo() -> None:
    """Behavior-only overlay must not fail validation on retained pose-kind combo."""

    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ui.pages.results_packaging_page import ResultsPackagingPage

    _ = QApplication.instance() or QApplication([])
    page = ResultsPackagingPage()
    page.export_overlay_pose_checkbox.setChecked(False)
    page.export_overlay_behavior_checkbox.setChecked(True)
    # Combo still holds a kind while pose is off — must be ignored on read.
    assert page.export_overlay_pose_kind_combo.currentData() is not None
    page.export_overlay_checkbox.setChecked(True)
    page._export_settings_changed()
    settings = page.controller.state.export_settings
    assert settings.overlay_enabled is True
    assert settings.overlay_include_behavior is True
    assert settings.overlay_include_pose is False
    assert settings.overlay_pose_source_kind is None
    assert settings.overlay_pose_run_dir is None
    assert "media_export_overlay" in page.export_overlay_dest_label.text()
    assert page.export_overlay_hud_combo.isEnabled()
    assert page.export_overlay_contact_checkbox.isEnabled()
    assert page.export_overlay_pose_kind_combo.isEnabled() is False


def test_incomplete_overlay_media_requires_new_job_not_retry(
    tmp_path: Path,
) -> None:
    """C3-5: failed/canceled overlay jobs are not same-job-retryable."""

    from results_packaging.media_csv_export.contracts import STATUS_INCOMPLETE
    from results_packaging.media_csv_export.discovery import ExportJobSummary
    from ui.controllers.results_packaging_controller import ResultsPackagingController

    controller = ResultsPackagingController()
    job_dir = tmp_path / "media_export_overlay__partial"
    job_dir.mkdir()
    job = ExportJobSummary(
        job_dir=job_dir,
        export_job_id="media_export_overlay__partial",
        job_kind=MEDIA_EXPORT_OVERLAY_JOB_KIND,
        status=STATUS_INCOMPLETE,
        classification="incomplete",
        source_package_run_id="results_package__x",
        source_package_dir=tmp_path / "pkg",
        created_at=None,
        completed_at=None,
        n_selected=1,
        n_succeeded=0,
        n_failed=1,
        n_rows=None,
        error_message="cancelled",
    )
    controller.state.export_summary = type("Exp", (), {"jobs": (job,)})()
    controller.state.selected_export_job_id = job.export_job_id
    reason = controller.cannot_retry_media_reason()
    assert reason is not None
    assert "Only media export jobs support same-job retry" in reason
    assert controller.can_retry_selected_media_job() is False
