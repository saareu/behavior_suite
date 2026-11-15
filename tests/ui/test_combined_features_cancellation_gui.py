"""S6 GUI Cancel enablement, feedback, and MainWindow deferred-close tests."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from tests.combined_features.helpers import write_joint_session

from combined_features import (
    STATUS_CANCELED,
    CombinedFeaturesRequest,
    CombinedFeaturesResult,
    run_combined_feature_extraction,
)
from preprocess.models import OperationProgress
from ui.controllers.combined_features_controller import CombinedFeaturesController
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.combined_features_page import CombinedFeaturesPage
from ui.tasks import GuiTaskContext, GuiTaskKind

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _upstream_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic joint sessions lack full S3/S1 trees; mark upstream eligible."""

    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "pose_finalization.run_discovery.is_s4_run_eligible_for_downstream",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "blob_features.run_discovery.is_s5_run_eligible_for_downstream",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "combined_features.runner.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
        raising=False,
    )
    monkeypatch.setattr(
        "combined_features.runner.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
        raising=False,
    )


def _wait_until(predicate, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


@pytest.fixture
def silence_modals(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("warning", "information", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))


class _CancelAfterNFalsePolls:
    """Return False for the first ``n`` polls, then True (post-bootstrap cancel)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


def _gated_cancelable_runner(
    *,
    worker_started: threading.Event,
    hold_until_cancel: bool = True,
):
    def runner(
        request: CombinedFeaturesRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> CombinedFeaturesResult:
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="computing_features",
                    message="Computing combined features",
                    completed_units=0,
                    total_units=None,
                    is_indeterminate=True,
                )
            )
        if hold_until_cancel:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if cancellation_requested is not None and cancellation_requested():
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("cancel was never requested")
        # Skip the pre-bootstrap checkpoint so a canceled run directory is persisted.
        cancel_after_bootstrap = _CancelAfterNFalsePolls(1)
        return run_combined_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=cancel_after_bootstrap,
            publication_started=publication_started,
        )

    return runner


def test_cancel_button_enablement_and_feedback(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    paths = write_joint_session(tmp_path, n_samples=4)
    worker_started = threading.Event()
    controller = CombinedFeaturesController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    page = CombinedFeaturesPage(controller)
    page.set_session(paths["session_root"])
    _wait_until(lambda: not page.has_running_task)
    assert page.cancel_button.isEnabled() is False

    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)
    assert page.run_button.isEnabled() is False

    page.cancel_button.click()
    assert page.cancel_button.isEnabled() is False
    assert "Cancel requested" in page.stage_label.text()
    assert "safe processing checkpoint" in page.stage_label.text()
    # Repeated Cancel clicks are harmless.
    page.cancel_button.click()
    page.cancel_button.click()

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    assert page.controller.state.run_stage == STATUS_CANCELED
    assert page.controller.state.summary is not None
    assert page.controller.state.summary.eligible_runs == ()
    assert page.run_button.isEnabled() is True
    assert page.cancel_button.isEnabled() is False
    page.close()


def test_cancel_unavailable_in_publication_critical_section(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    paths = write_joint_session(tmp_path, n_samples=4)
    entered = threading.Event()
    release = threading.Event()

    def runner(
        request: CombinedFeaturesRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> CombinedFeaturesResult:
        del cancellation_requested
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="writing_artifacts",
                    message="Writing artifacts",
                    completed_units=1,
                    total_units=1,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        return run_combined_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page = CombinedFeaturesPage(CombinedFeaturesController(runner=runner))
    page.set_session(paths["session_root"])
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )
    assert page.cancel_button.isEnabled() is False
    page._cancel_extraction()
    assert "Publication in progress" in page.stage_label.text()
    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    page.close()


def test_stale_progress_ignored_after_session_switch(
    tmp_path: Path, silence_modals: None
) -> None:
    del silence_modals
    paths_a = write_joint_session(tmp_path / "a", n_samples=3)
    paths_b = write_joint_session(tmp_path / "b", n_samples=3)
    hold = threading.Event()
    started = threading.Event()

    def runner(
        request: CombinedFeaturesRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> CombinedFeaturesResult:
        del cancellation_requested, publication_started
        started.set()
        assert hold.wait(timeout=10.0)
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="computing_features",
                    message="stale progress",
                    completed_units=0,
                    total_units=None,
                    is_indeterminate=True,
                )
            )
        return run_combined_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
        )

    page = CombinedFeaturesPage(CombinedFeaturesController(runner=runner))
    page.set_session(paths_a["session_root"])
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: started.is_set())
    # Attempting to switch session while extraction is active must be blocked.
    page.set_session(paths_b["session_root"])
    assert page.controller.state.session_root == paths_a["session_root"]
    hold.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    page.close()


def test_mainwindow_close_during_publication_waits_without_cancel(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close during publication must wait; never force-cancel or kill the QThread."""

    del silence_modals
    paths = write_joint_session(tmp_path, n_samples=4)
    entered = threading.Event()
    release = threading.Event()
    cancel_polled = {"requested": False}

    def runner(
        request: CombinedFeaturesRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> CombinedFeaturesResult:
        if publication_started is not None:
            publication_started()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="writing_artifacts",
                    message="Writing artifacts",
                    completed_units=1,
                    total_units=1,
                    is_indeterminate=False,
                )
            )
        entered.set()
        assert release.wait(timeout=10.0)
        if cancellation_requested is not None:
            cancel_polled["requested"] = bool(cancellation_requested())
        return run_combined_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=lambda: False,
            publication_started=None,
        )

    page = CombinedFeaturesPage(CombinedFeaturesController(runner=runner))
    window = MainWindow(PreprocessSetupController())
    monkeypatch.setattr(window, "combined_features_page", page)
    page.set_session(paths["session_root"])
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: entered.is_set())
    _wait_until(
        lambda: page.task_runner.active_context is not None
        and page.task_runner.active_context.is_publication_critical_section()
    )

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert window._close_after_s6_task is True
    assert page._extraction_cancel_requested is False
    assert page.task_runner.is_running is True
    assert page.cancel_button.isEnabled() is False
    assert "publishing" in page.request_cooperative_shutdown().lower()

    # Repeated close remains deferred and does not force-stop the worker.
    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted() is False
    assert page.task_runner.is_running is True

    release.set()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert cancel_polled["requested"] is False
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.success is True
    assert page.controller.state.run_stage != STATUS_CANCELED
    assert page.stage_label.text() != "Canceled"
    assert page.controller.state.last_result.status == "completed"
    page.close()
    window.close()


def test_mainwindow_defers_close_and_requests_s6_cancel(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    paths = write_joint_session(tmp_path, n_samples=4)
    worker_started = threading.Event()
    page = CombinedFeaturesPage(
        CombinedFeaturesController(
            runner=_gated_cancelable_runner(worker_started=worker_started)
        )
    )
    window = MainWindow(PreprocessSetupController())
    monkeypatch.setattr(window, "combined_features_page", page)
    page.set_session(paths["session_root"])
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    _wait_until(lambda: page.cancel_button.isEnabled() is True)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert page._extraction_cancel_requested is True
    assert "Cancel requested" in page.stage_label.text()

    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"
    page.close()
    window.close()


def test_rerun_after_gui_cancel(
    tmp_path: Path, silence_modals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del silence_modals
    paths = write_joint_session(tmp_path, n_samples=4)
    clock = {"n": 0}

    def _fake_now() -> str:
        clock["n"] += 1
        return f"20260101T2{clock['n']:05d}"

    monkeypatch.setattr("combined_features.runner._now_timestamp", _fake_now)
    worker_started = threading.Event()
    controller = CombinedFeaturesController(
        runner=_gated_cancelable_runner(worker_started=worker_started)
    )
    page = CombinedFeaturesPage(controller)
    page.set_session(paths["session_root"])
    _wait_until(lambda: not page.has_running_task)
    page.run_button.click()
    _wait_until(lambda: worker_started.is_set())
    page.cancel_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text() == "Canceled"

    # Real runner for the successful rerun (distinct timestamp via fake clock).
    page.controller._runner = run_combined_feature_extraction
    page.run_button.click()
    _wait_until(lambda: not page.has_running_task, timeout_sec=60.0)
    assert page.stage_label.text().startswith("Finished:")
    assert page.controller.state.last_result is not None
    assert page.controller.state.last_result.status == "completed"
    page.close()


def test_gui_task_context_publication_flag() -> None:
    context = GuiTaskContext(
        task_id="s6",
        task_kind=GuiTaskKind.COMBINED_FEATURES_EXTRACTION,
        project_dir=None,
        raw_video_path=None,
        generation=0,
    )
    assert context.is_publication_critical_section() is False
    context.mark_publication_critical_section()
    assert context.is_publication_critical_section() is True
    context.request_cancellation()
    assert context.is_cancellation_requested() is True
