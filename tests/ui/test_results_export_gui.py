"""S9 export GUI: eligibility gates, launch safety, job history, stale callbacks."""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication
from tests.results_packaging._export_fixtures import (
    build_mini_export_package,
    patch_export_package_currently_eligible,
)

from results_packaging.contracts import STALE_SOURCE, STATUS_COMPLETED
from results_packaging.media_csv_export import (
    CsvExportJobResult,
    MediaExportJobResult,
)
from results_packaging.media_csv_export.contracts import (
    JOB_KIND_MEDIA,
    STATUS_INCOMPLETE,
)
from results_packaging.media_csv_export.discovery import ExportJobSummary
from results_packaging.run_discovery import ResultsPackageRunSummary
from ui.controllers.results_export_support import ExportUiSettings
from ui.controllers.results_packaging_controller import (
    ResultsPackagingController,
    ResultsPackagingUiError,
)
from ui.pages.results_packaging_page import ResultsPackagingPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _package_summary(export_info: dict[str, Any], *, classification: str) -> Any:
    run = ResultsPackageRunSummary(
        run_dir=export_info["package_dir"],
        run_id=export_info["package_run_id"],
        status=STATUS_COMPLETED,
        classification=classification,
        source_s8_run_id="s8",
        source_s8_run_dir=None,
        package_manifest_path=None,
        sync_index_path=None,
        behavior_bouts_path=None,
        ethogram_summary_path=None,
        created_at=None,
        completed_at=None,
        warning_count=0,
        error_message=None,
        staleness_reasons=("upstream",) if classification == STALE_SOURCE else (),
    )
    return type("Summary", (), {"runs": (run,)})()


def _controller_with_package(
    tmp_path: Path, *, classification: str = "eligible"
) -> tuple[ResultsPackagingController, dict[str, Any], ExitStack]:
    export_info = build_mini_export_package(tmp_path / "export")
    session = export_info["session_root"]
    stack = ExitStack()
    stack.enter_context(patch_export_package_currently_eligible())
    controller = ResultsPackagingController()
    controller.state.session_root = session
    controller.state.summary = _package_summary(export_info, classification=classification)
    controller.state.selected_s9_run_id = export_info["package_run_id"]
    controller.refresh_export_preview()
    return controller, export_info, stack


def test_stale_package_blocks_export_controls(tmp_path: Path) -> None:
    controller, _export_info, stack = _controller_with_package(
        tmp_path, classification=STALE_SOURCE
    )
    with stack:
        reason = controller.cannot_export_reason()
        assert reason is not None
        assert "stale" in reason.lower()
        assert controller.can_run_export() is False


def test_csv_export_immediate_launch_and_conflict(tmp_path: Path) -> None:
    controller, export_info, stack = _controller_with_package(tmp_path)
    with stack:
        page = ResultsPackagingPage(controller)
        page.show()
        page.refresh_from_state()
        assert page.export_csv_button.isEnabled()

        t0 = time.perf_counter()
        identity = controller.begin_csv_export_launch(ExportUiSettings())
        assert time.perf_counter() - t0 < 0.25
        assert controller.state.task_running is True
        assert identity.job_kind == "csv_export"
        page.refresh_from_state()
        assert page.export_csv_button.isEnabled() is False
        assert page.export_media_button.isEnabled() is False
        with pytest.raises(ResultsPackagingUiError, match="already running"):
            controller.begin_media_export_launch(ExportUiSettings())
        # Finish without starting a QThread (avoids offscreen Qt runner crashes).
        result = CsvExportJobResult(
            success=True,
            status="completed",
            export_job_id="csv_export__test",
            job_dir=export_info["session_root"]
            / "results_exports"
            / "csv_export__test",
            session_root=export_info["session_root"],
            source_package_run_id=export_info["package_run_id"],
            source_package_dir=export_info["package_dir"],
            n_rows=3,
            csv_path=None,
            export_job_manifest_path=None,
            qc_path=None,
        )
        assert controller.record_csv_export_success(
            result, expected_identity=identity
        )
        page.refresh_from_state()
        assert page.export_csv_button.isEnabled()


def test_export_settings_immutable_and_stale_callback_rejection(tmp_path: Path) -> None:
    controller, export_info, stack = _controller_with_package(tmp_path)
    with stack:
        identity = controller.begin_csv_export_launch(ExportUiSettings())
        assert controller.state.task_running
        with pytest.raises(ResultsPackagingUiError, match="Cannot change export"):
            controller.update_export_settings(
                ExportUiSettings(
                    selection_mode="channels_behaviors", channels_text="contact"
                )
            )
        controller.state.active_export_identity = None
        fake = CsvExportJobResult(
            success=True,
            status="completed",
            export_job_id="csv_export__stale",
            job_dir=export_info["session_root"]
            / "results_exports"
            / "csv_export__stale",
            session_root=export_info["session_root"],
            source_package_run_id=export_info["package_run_id"],
            source_package_dir=export_info["package_dir"],
            n_rows=1,
            csv_path=None,
            export_job_manifest_path=None,
            qc_path=None,
        )
        assert (
            controller.record_csv_export_success(fake, expected_identity=identity)
            is False
        )


def test_incomplete_media_retry_gate(tmp_path: Path) -> None:
    controller, export_info, stack = _controller_with_package(tmp_path)
    with stack:
        job_dir = (
            export_info["session_root"] / "results_exports" / "media_export__retry"
        )
        job_dir.mkdir(parents=True)
        controller.state.export_summary = type(
            "Exp",
            (),
            {
                "jobs": (
                    ExportJobSummary(
                        job_dir=job_dir,
                        export_job_id="media_export__retry",
                        job_kind=JOB_KIND_MEDIA,
                        status=STATUS_INCOMPLETE,
                        classification="incomplete",
                        source_package_run_id=export_info["package_run_id"],
                        source_package_dir=export_info["package_dir"],
                        created_at=None,
                        completed_at=None,
                        n_selected=3,
                        n_succeeded=1,
                        n_failed=2,
                        n_rows=None,
                        error_message="partial",
                    ),
                )
            },
        )()
        controller.state.selected_export_job_id = "media_export__retry"
        assert controller.can_retry_selected_media_job()
        retried: list[Path] = []

        def fake_retry(path, progress_callback=None):
            retried.append(Path(path))
            return MediaExportJobResult(
                success=True,
                status="completed",
                export_job_id="media_export__retry",
                job_dir=job_dir,
                session_root=export_info["session_root"],
                source_package_run_id=export_info["package_run_id"],
                source_package_dir=export_info["package_dir"],
                n_selected=3,
                n_succeeded=3,
                n_failed=0,
                n_skipped=0,
                n_omitted=0,
                clip_index_path=None,
                export_job_manifest_path=None,
                qc_path=None,
            )

        controller._media_retry_runner = fake_retry  # type: ignore[method-assign]
        identity = controller.begin_media_retry_launch()
        result = controller.execute_media_retry_launch(identity)
        assert result.status == "completed"
        assert retried == [job_dir]
