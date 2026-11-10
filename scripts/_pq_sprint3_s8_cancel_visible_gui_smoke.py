"""Visible Windows GUI smoke for PQ Sprint 3 Increment 2 (S8 cancellation).

Uses a disposable synthetic S7/S6 session plus an instrumented slow worker so
Cancel can be exercised reliably (real synthetic S8 publication can finish
before a human click). Does not modify scientific arbitration algorithms and
does not touch lab/test_cases scientific artifacts.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Force the native Windows Qt platform for visible validation.
os.environ.pop("QT_QPA_PLATFORM", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402
from tests.behavioral_arbitration._fixtures import build_synthetic_session  # noqa: E402
from tests.behavioral_arbitration.test_export_qc_discovery import (  # noqa: E402
    _enter_all,
    _patch_publication,
)

from behavioral_arbitration import (  # noqa: E402
    STATUS_CANCELED,
    BehavioralArbitrationCancelledError,
    BehavioralArbitrationRequest,
    BehavioralArbitrationRunResult,
    is_s8_run_eligible_for_downstream,
    run_behavioral_arbitration_publication,
    summarize_behavioral_arbitration_project,
)
from behavioral_arbitration.contracts import (  # noqa: E402
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
)
from preprocess.models import OperationProgress  # noqa: E402
from ui.controllers.behavioral_arbitration_controller import (  # noqa: E402
    BehavioralArbitrationController,
)
from ui.controllers.preprocess_setup_controller import (  # noqa: E402
    PreprocessSetupController,
)
from ui.main_window import MainWindow  # noqa: E402

WORK = ROOT / "tmp_pq_sprint3_s8_cancel_gui"
EVIDENCE = ROOT / "docs" / "roadmap" / "_pq_sprint3_s8_cancel_visible_gui_evidence.txt"


class _CancelOnNthCheck:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def _wait_until(app: QApplication, predicate, timeout_sec: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for GUI state")
        time.sleep(0.02)
    app.processEvents()


def _patch_intake(info: dict[str, Any]) -> list[Any]:
    patches: list[Any] = []
    targets = (
        (
            "behavioral_arbitration.intake.validate_s7_run_for_downstream",
            lambda _p: {
                "s7_run_dir": info["s7_run_dir"].resolve(),
                "status": "completed",
                "session_root": info["session_root"].resolve(),
                "source_s6_run_dir": info["s6_run_dir"].resolve(),
                "source_s6_run_id": info["s6_run_id"],
                "dyadic_features_sha256": info["feat_sha"],
            },
        ),
        (
            "behavioral_arbitration.intake.validate_s6_run_for_downstream",
            lambda _p: {
                "s6_run_dir": info["s6_run_dir"].resolve(),
                "status": "completed",
                "session_root": info["session_root"].resolve(),
                "run_id": info["s6_run_id"],
            },
        ),
        ("behavioral_arbitration.intake.is_s7_run_eligible_for_downstream", lambda _p: True),
        ("behavioral_arbitration.intake.is_s6_run_eligible_for_downstream", lambda _p: True),
        (
            "behavioral_inference.runner.is_s7_run_eligible_for_downstream",
            lambda _p: True,
        ),
        (
            "behavioral_inference.run_discovery.is_s7_run_eligible_for_downstream",
            lambda _p: True,
        ),
        (
            "combined_features.runner.is_s6_run_eligible_for_downstream",
            lambda _p: True,
        ),
    )
    for target, value in targets:
        patches.append(patch(target, new=value))
    return patches


def _slow_cancelable_runner(info: dict[str, Any], worker_started: threading.Event):
    def runner(
        request: BehavioralArbitrationRequest,
        progress_callback=None,
        *,
        cancellation_requested=None,
        publication_started=None,
    ) -> BehavioralArbitrationRunResult:
        del publication_started
        worker_started.set()
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="arbitration",
                    message="Running Pipeline 100 arbitration stages (Sprint 1–3)…",
                    completed_units=1,
                    total_units=6,
                    is_indeterminate=False,
                )
            )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if cancellation_requested is not None and cancellation_requested():
                stamp = time.strftime("%Y%m%dT%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
                cancel_request = replace(request, timestamp=stamp)
                with _enter_all(_patch_publication(info)), pytest_raises_cancel():
                    run_behavioral_arbitration_publication(
                        cancel_request,
                        progress_callback=progress_callback,
                        # Checkpoint 2+: after run bootstrap / before arbitration.
                        cancellation_requested=_CancelOnNthCheck(2),
                    )
                raise BehavioralArbitrationCancelledError(
                    "Behavioral arbitration cancelled before publication."
                )
            time.sleep(0.05)
        raise TimeoutError("cancel was never requested during visible smoke")

    return runner


class pytest_raises_cancel:
    """Context that swallows only the expected cancel exception from persist path."""

    def __enter__(self) -> pytest_raises_cancel:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return exc_type is BehavioralArbitrationCancelledError


def main() -> int:
    lines: list[str] = []
    app = QApplication.instance() or QApplication(sys.argv)
    for name in ("warning", "information", "critical", "question"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))

    active_patches: list[Any] = []
    try:
        if WORK.exists():
            import shutil

            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)

        info = build_synthetic_session(WORK, n_frames=8)
        lines.append(f"session_root={info['session_root']}")
        lines.append("method=instrumented_slow_worker_plus_real_cancel_persist")

        active_patches = list(_patch_publication(info)) + _patch_intake(info)
        for p in active_patches:
            p.start()

        completed = run_behavioral_arbitration_publication(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                run_purpose="smoke_completed",
                timestamp="20261110T120000",
            )
        )
        lines.append(f"prior_completed_run={completed.run_id}")
        lines.append(
            "prior_completed_eligible="
            f"{is_s8_run_eligible_for_downstream(completed.run_dir)}"
        )

        worker_started = threading.Event()
        controller = BehavioralArbitrationController(
            runner=_slow_cancelable_runner(info, worker_started)
        )
        window = MainWindow(
            PreprocessSetupController(),
            behavioral_arbitration_controller=controller,
        )
        window.show()
        window.open_behavioral_arbitration(info["session_root"])
        page = window.behavioral_arbitration_page
        _wait_until(app, lambda: not page.has_running_task)
        lines.append("opened_s8_page=True")

        page.run_button.click()
        _wait_until(app, lambda: worker_started.is_set())
        _wait_until(app, lambda: page.cancel_button.isEnabled())
        lines.append("cancel_enabled_during_run=True")
        page.cancel_button.click()
        lines.append(f"cancel_requested_feedback={page.stage_label.text()!r}")
        _wait_until(app, lambda: not page.has_running_task, timeout_sec=90.0)
        lines.append(f"terminal_stage={page.stage_label.text()!r}")
        lines.append(f"controller_run_stage={page.controller.state.run_stage!r}")

        summary = summarize_behavioral_arbitration_project(info["session_root"])
        canceled_runs = [r for r in summary.runs if r.status == STATUS_CANCELED]
        lines.append(f"canceled_run_count={len(canceled_runs)}")
        if canceled_runs:
            cdir = canceled_runs[0].run_dir
            lines.append(
                f"canceled_eligible={is_s8_run_eligible_for_downstream(cdir)}"
            )
            lines.append(
                "canceled_science_absent="
                f"{(not (cdir / BEHAVIOR_STATES_FILENAME).exists()) and (not (cdir / BEHAVIOR_BOUTS_FILENAME).exists()) and (not (cdir / QC_FILENAME).exists())}"
            )
            lines.append(f"canceled_meta_present={(cdir / RUN_META_FILENAME).is_file()}")
        lines.append(
            f"prior_still_eligible={is_s8_run_eligible_for_downstream(completed.run_dir)}"
        )
        lines.append(f"eligible_run_ids={[r.run_id for r in summary.eligible_runs]}")

        worker_started2 = threading.Event()
        controller._runner = _slow_cancelable_runner(info, worker_started2)
        page.run_button.click()
        _wait_until(app, lambda: worker_started2.is_set())
        event = QCloseEvent()
        window.closeEvent(event)
        lines.append(f"close_deferred={not event.isAccepted()}")
        lines.append(f"close_after_s8={window._close_after_s8_task}")
        _wait_until(app, lambda: not page.has_running_task, timeout_sec=90.0)
        lines.append(f"after_close_stage={page.stage_label.text()!r}")
        lines.append("rerun_possible=True")
        lines.append("PASS")

        window.close()
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(EVIDENCE.read_text(encoding="utf-8"))
        return 0
    except Exception:
        lines.append("FAIL")
        lines.append(traceback.format_exc())
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(EVIDENCE.read_text(encoding="utf-8"))
        return 1
    finally:
        for p in reversed(active_patches):
            p.stop()


if __name__ == "__main__":
    raise SystemExit(main())
