"""S8 GUI responsiveness: async discovery, stale rejection, nonblocking launch."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication
from tests.behavioral_arbitration._fixtures import build_synthetic_session

from behavioral_arbitration import (
    BehavioralArbitrationRequest,
    BehavioralArbitrationRunResult,
)
from behavioral_arbitration.contracts import STATUS_COMPLETED
from ui.controllers.behavioral_arbitration_controller import (
    BehavioralArbitrationController,
    BehavioralArbitrationUiError,
    DiscoverySnapshot,
)
from ui.pages.behavioral_arbitration_page import BehavioralArbitrationPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


def _patch_s7_s6_ok(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, Any]) -> None:
    s7 = fixture["s7_run_dir"]
    s6 = fixture["s6_run_dir"]
    session = fixture["session_root"]

    def _validate_s7(_path: Any) -> dict[str, Any]:
        return {
            "s7_run_dir": Path(s7).resolve(),
            "status": "completed",
            "session_root": session.resolve(),
            "source_s6_run_dir": Path(s6).resolve(),
            "source_s6_run_id": fixture["s6_run_id"],
            "dyadic_features_sha256": fixture["feat_sha"],
        }

    def _validate_s6(_path: Any) -> dict[str, Any]:
        return {
            "s6_run_dir": Path(s6).resolve(),
            "status": "completed",
            "session_root": session.resolve(),
            "run_id": fixture["s6_run_id"],
        }

    for target in (
        "behavioral_arbitration.intake.validate_s7_run_for_downstream",
        "behavioral_arbitration.runner.validate_s7_run_for_downstream",
        "behavioral_inference.runner.validate_s7_run_for_downstream",
        "behavioral_inference.run_discovery.validate_s7_run_for_downstream",
    ):
        monkeypatch.setattr(target, _validate_s7, raising=False)
    for target in (
        "behavioral_arbitration.intake.validate_s6_run_for_downstream",
        "behavioral_arbitration.runner.validate_s6_run_for_downstream",
        "combined_features.runner.validate_s6_run_for_downstream",
    ):
        monkeypatch.setattr(target, _validate_s6, raising=False)
    for target in (
        "behavioral_arbitration.intake.is_s7_run_eligible_for_downstream",
        "behavioral_arbitration.runner.is_s7_run_eligible_for_downstream",
        "behavioral_inference.runner.is_s7_run_eligible_for_downstream",
    ):
        monkeypatch.setattr(target, lambda _p: True, raising=False)
    for target in (
        "behavioral_arbitration.intake.is_s6_run_eligible_for_downstream",
        "behavioral_arbitration.runner.is_s6_run_eligible_for_downstream",
        "combined_features.runner.is_s6_run_eligible_for_downstream",
        "behavioral_inference.runner.is_s6_run_eligible_for_downstream",
    ):
        monkeypatch.setattr(target, lambda _p: True, raising=False)


def test_async_discovery_and_stale_result_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_synthetic_session(tmp_path)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller = BehavioralArbitrationController()
    page = BehavioralArbitrationPage(controller)
    page.show()

    page.set_session(fixture["session_root"])
    assert controller.state.discovery_running
    first_identity = controller.state.active_discovery_identity
    assert first_identity is not None

    # Switch session while discovery is pending; newer generation must win.
    other = tmp_path / "other_session"
    other.mkdir()
    page.set_session(other)
    second_identity = controller.state.active_discovery_identity
    assert second_identity is not None
    assert second_identity.request_id != first_identity.request_id
    assert second_identity.generation > first_identity.generation

    from behavioral_arbitration import summarize_behavioral_arbitration_project

    stale = DiscoverySnapshot(
        identity=first_identity,
        s7_options=(),
        s8_summary=summarize_behavioral_arbitration_project(
            other, eligible_s7_available=False
        ),
    )
    assert controller.apply_discovery_snapshot(stale) is False
    _wait_until(lambda: not page.has_running_task, timeout_sec=15.0)
    assert controller.state.session_root == other.resolve()


def test_begin_arbitration_launch_is_nonblocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_synthetic_session(tmp_path)
    _patch_s7_s6_ok(monkeypatch, fixture)
    assess_calls = {"n": 0}

    real_assess = __import__(
        "behavioral_arbitration.intake", fromlist=["assess_s7_for_s8"]
    ).assess_s7_for_s8

    def _counting_assess(*args: Any, **kwargs: Any):
        assess_calls["n"] += 1
        time.sleep(0.05)
        return real_assess(*args, **kwargs)

    monkeypatch.setattr(
        "ui.controllers.behavioral_arbitration_controller.assess_s7_for_s8",
        _counting_assess,
    )
    controller = BehavioralArbitrationController()
    controller.set_session(fixture["session_root"])
    before = assess_calls["n"]
    t0 = time.perf_counter()
    identity = controller.begin_arbitration_launch()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0
    assert assess_calls["n"] == before
    assert controller.state.task_running
    assert controller.state.run_stage.startswith("Launching")
    with pytest.raises(BehavioralArbitrationUiError):
        controller.begin_arbitration_launch()
    # Worker-side revalidation still performs assess.
    controller.execute_arbitration_launch(identity)
    assert assess_calls["n"] == before + 1


def test_page_run_button_accepts_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_synthetic_session(tmp_path)
    _patch_s7_s6_ok(monkeypatch, fixture)
    started = {"at": None}

    def _slow_runner(request: BehavioralArbitrationRequest, progress_callback=None):
        started["at"] = time.perf_counter()
        time.sleep(0.25)
        return BehavioralArbitrationRunResult(
            success=True,
            status=STATUS_COMPLETED,
            run_id="behavioral_arbitration__resp",
            run_dir=fixture["session_root"]
            / "behavioral_arbitration"
            / "behavioral_arbitration__resp",
            run_meta_path=fixture["session_root"] / "x.json",
            settings_used_path=fixture["session_root"] / "x.yaml",
            processing_log_path=fixture["session_root"] / "x.txt",
            qc_path=fixture["session_root"] / "qc.json",
            behavior_states_path=None,
            behavior_bouts_path=None,
            diagnostics_path=None,
            session_root=fixture["session_root"],
            source_s7_run_id=fixture["s7_run_id"],
            source_s6_run_id=fixture["s6_run_id"],
            warnings=(),
            error_message=None,
            qc={"structural_ok": True, "metrics": {}, "warnings": [], "failures": []},
        )

    controller = BehavioralArbitrationController(runner=_slow_runner)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller.set_session(fixture["session_root"])
    page = BehavioralArbitrationPage(controller)
    page.show()
    _wait_until(lambda: not page.has_running_task, timeout_sec=15.0)
    t0 = time.perf_counter()
    page._run_arbitration()
    gui_return_ms = (time.perf_counter() - t0) * 1000.0
    assert gui_return_ms < 200.0
    assert page.has_running_task
    assert not page.run_button.isEnabled()
    _wait_until(lambda: not page.has_running_task, timeout_sec=15.0)
    assert started["at"] is not None


def test_duplicate_launch_blocked_after_begin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_synthetic_session(tmp_path)
    _patch_s7_s6_ok(monkeypatch, fixture)
    controller = BehavioralArbitrationController()
    controller.set_session(fixture["session_root"])
    controller.begin_arbitration_launch()
    with pytest.raises(BehavioralArbitrationUiError):
        controller.begin_arbitration_launch()
    with pytest.raises(BehavioralArbitrationUiError):
        controller.select_s7_source(fixture["s7_run_id"], manual=True)
