"""Offscreen GUI-thread responsiveness audit for S2/S4/S5/S6 async discovery.

Non-destructive. Uses the E-session. Records cold/warm wall times for page
set_session completion and event-loop tick counts while discovery/open runs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.controllers.blob_features_controller import BlobFeaturesController
from ui.controllers.combined_features_controller import CombinedFeaturesController
from ui.controllers.pose_finalization_controller import PoseFinalizationController
from ui.controllers.pose_inference_controller import PoseInferenceController
from ui.pages.blob_features_page import BlobFeaturesPage
from ui.pages.combined_features_page import CombinedFeaturesPage
from ui.pages.pose_finalization_page import PoseFinalizationPage
from ui.pages.pose_inference_page import PoseInferencePage

SESSION = Path(r"c:\behavior_suite\test_cases\E_full_session_integration")
APP = QApplication.instance() or QApplication([])


def _measure(label: str, factory, *, warm: bool) -> dict[str, float]:
    page = factory()
    page.show()
    APP.processEvents()
    ticks = {"n": 0}

    def _tick() -> None:
        ticks["n"] += 1

    timer = QTimer()
    timer.setInterval(16)
    timer.timeout.connect(_tick)
    timer.start()

    t0 = time.perf_counter()
    page.set_session(SESSION)
    set_session_ms = (time.perf_counter() - t0) * 1000.0
    assert page.has_running_task or page.controller.state.session_root is not None
    # GUI must remain able to process timer ticks during the worker.
    while page.has_running_task:
        APP.processEvents()
        time.sleep(0.005)
    total_ms = (time.perf_counter() - t0) * 1000.0
    timer.stop()
    first_useful_ms = total_ms  # first useful results appear when idle after apply
    print(
        f"{label} warm={warm}: set_session_return_ms={set_session_ms:.1f} "
        f"until_idle_ms={total_ms:.1f} first_useful_ms={first_useful_ms:.1f} "
        f"event_loop_ticks={ticks['n']}"
    )
    page.close()
    page.deleteLater()
    APP.processEvents()
    return {
        "set_session_ms": set_session_ms,
        "until_idle_ms": total_ms,
        "ticks": float(ticks["n"]),
    }


def main() -> None:
    if not SESSION.is_dir():
        raise SystemExit(f"Missing E-session: {SESSION}")

    cases = [
        ("S2", lambda: PoseInferencePage(PoseInferenceController())),
        ("S4", lambda: PoseFinalizationPage(PoseFinalizationController())),
        ("S5", lambda: BlobFeaturesPage(BlobFeaturesController())),
        ("S6", lambda: CombinedFeaturesPage(CombinedFeaturesController())),
    ]
    print(f"session={SESSION}")
    for label, factory in cases:
        _measure(label, factory, warm=False)
        _measure(label, factory, warm=True)


if __name__ == "__main__":
    main()
