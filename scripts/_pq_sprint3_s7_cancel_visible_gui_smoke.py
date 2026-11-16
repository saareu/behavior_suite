"""Visible Windows GUI smoke for PQ Sprint 3 Increment 5 (S7 cancellation).

Uses a disposable synthetic S6 session plus an instrumented slow worker so Cancel
can be exercised reliably. Does not modify scientific prediction formulas and
does not touch lab/test_cases scientific artifacts.

Method: instrumented slow worker + real cancel-persist path
(not live lab predict_proba).
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from _pytest.monkeypatch import MonkeyPatch  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from behavioral_inference import (  # noqa: E402
    PREDICTIONS_FILENAME,
    RUN_META_FILENAME,
    STATUS_CANCELED,
    BehavioralInferenceRequest,
    BehavioralInferenceRunResult,
    find_latest_eligible_s7_run,
    is_s7_run_eligible_for_downstream,
    run_behavioral_inference_publication,
    summarize_behavioral_inference_project,
)
from behavioral_inference.config import DEFAULT_WORKING_THRESHOLDS  # noqa: E402
from preprocess.models import OperationProgress  # noqa: E402
from tests.ui.test_behavioral_inference_gui import _ready_controller  # noqa: E402
from ui.controllers.behavioral_inference_controller import (  # noqa: E402
    BehavioralInferenceUiSettings,
)
from ui.controllers.preprocess_setup_controller import (  # noqa: E402
    PreprocessSetupController,
)
from ui.main_window import MainWindow  # noqa: E402
from ui.pages.behavioral_inference_page import BehavioralInferencePage  # noqa: E402

WORK = ROOT / "tmp_pq_sprint3_s7_cancel_gui"
EVIDENCE = ROOT / "docs" / "roadmap" / "_pq_sprint3_s7_cancel_visible_gui_evidence.txt"


def _wait_until(app: QApplication, predicate, timeout_sec: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for GUI state")
        time.sleep(0.02)
    app.processEvents()


class _CancelAfterNFalsePolls:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


def _gated_runner(worker_started: threading.Event):
    def runner(
        request: BehavioralInferenceRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralInferenceRunResult:
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="running_detector_inference",
                    message=(
                        "Running detector inference: approach "
                        "(0/2 detectors completed)"
                    ),
                    completed_units=0,
                    total_units=2,
                    is_indeterminate=False,
                )
            )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if cancellation_requested is not None and cancellation_requested():
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("cancel was never requested")
        return run_behavioral_inference_publication(
            request,
            progress_callback=progress_callback,
            cancellation_requested=_CancelAfterNFalsePolls(2),
            publication_started=publication_started,
        )

    return runner


def main() -> int:
    lines: list[str] = [
        "PQ Sprint 3 Increment 5 — S7 visible GUI cancellation smoke",
        f"work_root: {WORK}",
        "method: instrumented slow worker + real cancel-persist path",
        "inference_mode: instrumented worker (not live lab predict_proba)",
    ]
    if WORK.exists():
        for path in sorted(WORK.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    WORK.mkdir(parents=True, exist_ok=True)

    try:
        mp = MonkeyPatch()
        controller, session, registry = _ready_controller(mp, WORK)
        completed = run_behavioral_inference_publication(
            BehavioralInferenceRequest(
                session_root=session,
                s6_run_dir=next((session / "combined_features").iterdir()),
                config=controller.state.working_settings.to_config(),
                selected_detectors_dyadic=["nose_to_nose"],
                selected_detectors_focal=["approach"],
                run_purpose="smoke_prior_completed",
                timestamp="20260101T090000",
            )
        )
        assert completed.predictions_path is not None
        assert is_s7_run_eligible_for_downstream(completed.run_dir)

        app = QApplication.instance() or QApplication([])
        for name in ("warning", "information", "critical", "question"):
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))

        worker_started = threading.Event()
        controller._runner = _gated_runner(worker_started)
        controller.update_settings(
            BehavioralInferenceUiSettings(
                parity_mode="legacy_parity",
                model_registry_root=registry,
                selected_detectors_dyadic=("nose_to_nose",),
                selected_detectors_focal=("approach",),
                working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
            )
        )
        page = BehavioralInferencePage(controller)
        window = MainWindow(PreprocessSetupController())
        window.behavioral_inference_page = page
        window.show()
        page.show()
        page.set_session(session)
        _wait_until(app, lambda: not page.has_running_task)
        page.refresh_from_state()
        _wait_until(app, lambda: page.controller.can_run_inference())

        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        lines.append("PASS: Cancel enabled during cancellable S7 detector inference")

        page.cancel_button.click()
        assert page.cancel_button.isEnabled() is False
        assert "Cancel requested" in page.stage_label.text()
        assert "current detector" in page.stage_label.text()
        lines.append("PASS: Cancel requested feedback visible")

        _wait_until(app, lambda: not page.has_running_task, timeout_sec=60.0)
        assert page.stage_label.text() == "Canceled"
        assert page.controller.state.run_stage == STATUS_CANCELED
        summary = summarize_behavioral_inference_project(session)
        canceled_runs = [r for r in summary.runs if r.status == STATUS_CANCELED]
        assert canceled_runs
        assert not (canceled_runs[0].run_dir / PREDICTIONS_FILENAME).exists()
        assert (canceled_runs[0].run_dir / RUN_META_FILENAME).is_file()
        assert is_s7_run_eligible_for_downstream(canceled_runs[0].run_dir) is False
        assert is_s7_run_eligible_for_downstream(completed.run_dir) is True
        latest = find_latest_eligible_s7_run(session)
        assert latest is not None
        assert latest.run_id == completed.run_id
        lines.append("PASS: worker-confirmed Canceled; unpublished artifacts removed")
        lines.append("PASS: canceled excluded from S8; prior completed remains eligible")

        # Confirm rerun works after cancel (before deferred-close exercise).
        page.controller._runner = run_behavioral_inference_publication
        _wait_until(
            app,
            lambda: (
                not page.has_running_task
                and page.controller.can_run_inference()
            ),
            timeout_sec=60.0,
        )
        page.run_button.click()
        _wait_until(app, lambda: page.has_running_task or page.controller.state.last_result is not None or page.controller.state.run_error, timeout_sec=10.0)
        _wait_until(app, lambda: not page.has_running_task, timeout_sec=60.0)
        if page.controller.state.last_result is None:
            raise AssertionError(
                "rerun produced no result; "
                f"stage={page.stage_label.text()!r} "
                f"run_stage={page.controller.state.run_stage!r} "
                f"run_error={page.controller.state.run_error!r}"
            )
        assert page.controller.state.last_result.status in {
            "completed",
            "completed_with_warnings",
        }
        lines.append("PASS: rerun after cancellation succeeds")

        worker_started.clear()
        page.controller._runner = _gated_runner(worker_started)
        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        page.cancel_button.click()
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted() is False
        assert window._close_after_s7_task is True
        assert page.task_runner.is_running is True
        lines.append("PASS: close during active cancel defers shutdown; worker retained")
        _wait_until(app, lambda: not page.has_running_task, timeout_sec=60.0)
        lines.append("PASS: deferred close waited for worker completion")
        lines.append("RESULT: PASS")
        EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        window.close()
        print("\n".join(lines))
        return 0
    except Exception:
        lines.append("RESULT: FAIL")
        lines.append(traceback.format_exc())
        EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
