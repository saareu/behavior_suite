"""Visible Windows GUI smoke for PQ Sprint 3 Increment 4 (S6 cancellation).

Uses a disposable synthetic joint S4/S5 session plus an instrumented slow worker
so Cancel can be exercised reliably (real synthetic S6 extraction can finish
before a human click). Does not modify scientific feature formulas and does not
touch lab/test_cases scientific artifacts.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Force the native Windows Qt platform for visible validation.
os.environ.pop("QT_QPA_PLATFORM", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from behavioral_inference.run_discovery import find_latest_eligible_s6_for_s7  # noqa: E402
from combined_features import (  # noqa: E402
    DYADIC_FEATURES_FILENAME,
    RUN_META_FILENAME,
    STATUS_CANCELED,
    CombinedFeaturesRequest,
    CombinedFeaturesResult,
    is_s6_run_eligible_for_downstream,
    run_combined_feature_extraction,
    summarize_combined_features_project,
)
from preprocess.models import OperationProgress  # noqa: E402
from tests.combined_features.helpers import write_joint_session  # noqa: E402
from ui.controllers.combined_features_controller import (  # noqa: E402
    CombinedFeaturesController,
)
from ui.controllers.preprocess_setup_controller import (  # noqa: E402
    PreprocessSetupController,
)
from ui.main_window import MainWindow  # noqa: E402

WORK = ROOT / "tmp_pq_sprint3_s6_cancel_gui"
EVIDENCE = ROOT / "docs" / "roadmap" / "_pq_sprint3_s6_cancel_visible_gui_evidence.txt"


def _wait_until(app: QApplication, predicate, timeout_sec: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for GUI state")
        time.sleep(0.02)
    app.processEvents()


def _patch_upstream_eligible() -> None:
    import blob_features.run_discovery as s5_discovery
    import combined_features.upstream as upstream
    import pose_finalization.run_discovery as s4_discovery

    upstream.validate_s4_run_for_downstream = lambda _path: {"ok": True}  # type: ignore[assignment]
    upstream.validate_s5_run_for_downstream = lambda _path: {"ok": True}  # type: ignore[assignment]
    s4_discovery.is_s4_run_eligible_for_downstream = lambda _path: True  # type: ignore[assignment]
    s5_discovery.is_s5_run_eligible_for_downstream = lambda _path: True  # type: ignore[assignment]


class _CancelAfterNFalsePolls:
    """Return False for the first ``n`` polls, then True (post-bootstrap cancel)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


def _gated_runner(worker_started: threading.Event):
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
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if cancellation_requested is not None and cancellation_requested():
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("cancel was never requested")
        # Skip the pre-bootstrap checkpoint so a canceled run directory is persisted.
        return run_combined_feature_extraction(
            request,
            progress_callback=progress_callback,
            cancellation_requested=_CancelAfterNFalsePolls(1),
            publication_started=publication_started,
        )

    return runner


def main() -> int:
    lines: list[str] = [
        "PQ Sprint 3 Increment 4 — S6 visible GUI cancellation smoke",
        f"work_root: {WORK}",
        "method: instrumented slow worker + real cancel-persist path",
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        for path in sorted(WORK.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    try:
        _patch_upstream_eligible()
        paths = write_joint_session(WORK, n_samples=6)
        session = paths["session_root"]
        completed = run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=session,
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="smoke_prior_completed",
                timestamp="20260101T090000",
            )
        )
        assert completed.dyadic_features_path is not None
        assert is_s6_run_eligible_for_downstream(completed.run_dir)

        app = QApplication.instance() or QApplication([])
        for name in ("warning", "information", "critical", "question"):
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))

        worker_started = threading.Event()
        controller = CombinedFeaturesController(runner=_gated_runner(worker_started))
        window = MainWindow(
            PreprocessSetupController(),
            combined_features_controller=controller,
        )
        window.show()
        window.open_combined_features(session)
        page = window.combined_features_page
        _wait_until(app, lambda: not page.has_running_task)

        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        lines.append("PASS: Cancel enabled during cancellable S6 processing")

        page.cancel_button.click()
        assert page.cancel_button.isEnabled() is False
        assert "Cancel requested" in page.stage_label.text()
        assert "safe processing checkpoint" in page.stage_label.text()
        lines.append("PASS: Cancel requested feedback visible")

        _wait_until(app, lambda: not page.has_running_task, timeout_sec=60.0)
        assert page.stage_label.text() == "Canceled"
        assert page.controller.state.run_stage == STATUS_CANCELED
        summary = summarize_combined_features_project(session)
        canceled_runs = [r for r in summary.runs if r.status == STATUS_CANCELED]
        assert canceled_runs
        assert not (canceled_runs[0].run_dir / DYADIC_FEATURES_FILENAME).exists()
        assert (canceled_runs[0].run_dir / RUN_META_FILENAME).is_file()
        assert is_s6_run_eligible_for_downstream(canceled_runs[0].run_dir) is False
        assert is_s6_run_eligible_for_downstream(completed.run_dir) is True
        latest_for_s7 = find_latest_eligible_s6_for_s7(session)
        assert latest_for_s7 is not None
        assert latest_for_s7.run_id == completed.run_id
        lines.append("PASS: worker-confirmed Canceled; unpublished artifacts removed")
        lines.append("PASS: canceled excluded from S7; prior completed remains eligible")

        # Second run after cancel, then deferred close during cancel.
        worker_started.clear()
        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        page.cancel_button.click()
        event = QCloseEvent()
        window.closeEvent(event)
        assert event.isAccepted() is False
        assert window._close_after_s6_task is True
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
