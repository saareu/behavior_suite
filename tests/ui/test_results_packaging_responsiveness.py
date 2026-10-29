"""S9 GUI responsiveness: async discovery, stale rejection, nonblocking launch."""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication
from tests.results_packaging._fixtures import build_packaging_session

from behavioral_arbitration.contracts import STATUS_COMPLETED
from results_packaging import (
    ResultsPackageRunResult,
    ResultsPackagingRequest,
    summarize_results_package_project,
)
from results_packaging.contracts import STATUS_COMPLETED as S9_COMPLETED
from ui.controllers.results_packaging_controller import (
    DiscoverySnapshot,
    ResultsPackagingController,
    ResultsPackagingUiError,
)
from ui.pages.results_packaging_page import ResultsPackagingPage

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


def _eligible_patches(info: dict[str, Any]):
    def fake_validate(path: Path | str) -> dict[str, Any]:
        return {
            "s8_run_dir": Path(path).resolve(),
            "status": STATUS_COMPLETED,
            "session_root": info["session_root"].resolve(),
            "source_s7_run_dir": info["s7_run_dir"].resolve(),
            "source_s7_run_id": info["s7_run_id"],
            "source_s6_run_dir": info["s6_run_dir"].resolve(),
            "source_s6_run_id": info["s6_run_id"],
        }

    return (
        patch(
            "results_packaging.eligibility.validate_s8_run_for_downstream",
            side_effect=fake_validate,
        ),
        patch(
            "results_packaging.eligibility.is_s8_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.runner.validate_s8_run_for_downstream",
            side_effect=fake_validate,
            raising=False,
        ),
        patch(
            "behavioral_arbitration.runner.is_s8_run_eligible_for_downstream",
            return_value=True,
            raising=False,
        ),
        patch(
            "behavioral_arbitration.run_discovery.validate_s8_run_for_downstream",
            side_effect=fake_validate,
            raising=False,
        ),
        patch(
            "behavioral_arbitration.run_discovery.is_s8_run_eligible_for_downstream",
            return_value=True,
            raising=False,
        ),
    )


def test_async_discovery_and_stale_result_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_packaging_session(tmp_path)
    stack = ExitStack()
    for p in _eligible_patches(fixture):
        stack.enter_context(p)
    with stack:
        controller = ResultsPackagingController()
        page = ResultsPackagingPage(controller)
        page.show()

        page.set_session(fixture["session_root"])
        assert controller.state.discovery_running
        first_identity = controller.state.active_discovery_identity
        assert first_identity is not None

        other = tmp_path / "other_session"
        other.mkdir()
        page.set_session(other)
        second_identity = controller.state.active_discovery_identity
        assert second_identity is not None
        assert second_identity.request_id != first_identity.request_id
        assert second_identity.generation > first_identity.generation

        stale = DiscoverySnapshot(
            identity=first_identity,
            s8_options=(),
            s9_summary=summarize_results_package_project(
                other, eligible_s8_available=False
            ),
        )
        assert controller.apply_discovery_snapshot(stale) is False
        _wait_until(lambda: not page.has_running_task, timeout_sec=15.0)
        assert controller.state.session_root == other.resolve()


def test_begin_packaging_launch_is_nonblocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = build_packaging_session(tmp_path)
    stack = ExitStack()
    for p in _eligible_patches(fixture):
        stack.enter_context(p)
    validate_calls = {"n": 0}

    real_validate = __import__(
        "results_packaging.eligibility", fromlist=["validate_s8_source_for_packaging"]
    ).validate_s8_source_for_packaging

    def _counting_validate(*args: Any, **kwargs: Any):
        validate_calls["n"] += 1
        time.sleep(0.05)
        return real_validate(*args, **kwargs)

    def _fake_runner(request: ResultsPackagingRequest, progress_callback=None):
        run_dir = fixture["session_root"] / "results_package" / "results_package__nb"
        run_dir.mkdir(parents=True, exist_ok=True)
        return ResultsPackageRunResult(
            success=True,
            status=S9_COMPLETED,
            run_id="results_package__nb",
            run_dir=run_dir,
            session_root=fixture["session_root"],
            source_s8_run_id=fixture["s8_run_id"],
            source_s8_run_dir=fixture["s8_run_dir"],
            run_meta_path=run_dir / "run_meta.json",
            settings_used_path=run_dir / "settings_used.yaml",
            processing_log_path=run_dir / "processing_log.txt",
            qc_path=None,
            package_manifest_path=None,
            sync_index_path=None,
            behavior_bouts_path=None,
            ethogram_summary_path=None,
            warnings=(),
            error_message=None,
            qc={"structural_ok": True, "metrics": {}, "warnings": [], "failures": []},
            currently_eligible=True,
            classification="eligible",
        )

    with stack:
        controller = ResultsPackagingController(runner=_fake_runner)
        controller.set_session(fixture["session_root"])
        monkeypatch.setattr(
            "ui.controllers.results_packaging_controller.validate_s8_source_for_packaging",
            _counting_validate,
        )
        before = validate_calls["n"]
        t0 = time.perf_counter()
        identity = controller.begin_packaging_launch()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0
        assert validate_calls["n"] == before
        assert controller.state.task_running
        assert controller.state.run_stage.startswith("Launching")
        with pytest.raises(ResultsPackagingUiError):
            controller.begin_packaging_launch()
        controller.execute_packaging_launch(identity)
        assert validate_calls["n"] == before + 1
        controller.record_failure("done", expected_identity=identity)


def test_page_run_button_accepts_immediately(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path)
    stack = ExitStack()
    for p in _eligible_patches(fixture):
        stack.enter_context(p)
    started = {"at": None}

    def _slow_runner(request: ResultsPackagingRequest, progress_callback=None):
        started["at"] = time.perf_counter()
        time.sleep(0.25)
        run_dir = fixture["session_root"] / "results_package" / "results_package__resp"
        run_dir.mkdir(parents=True, exist_ok=True)
        return ResultsPackageRunResult(
            success=True,
            status=S9_COMPLETED,
            run_id="results_package__resp",
            run_dir=run_dir,
            session_root=fixture["session_root"],
            source_s8_run_id=fixture["s8_run_id"],
            source_s8_run_dir=fixture["s8_run_dir"],
            run_meta_path=run_dir / "run_meta.json",
            settings_used_path=run_dir / "settings_used.yaml",
            processing_log_path=run_dir / "processing_log.txt",
            qc_path=None,
            package_manifest_path=None,
            sync_index_path=None,
            behavior_bouts_path=None,
            ethogram_summary_path=None,
            warnings=(),
            error_message=None,
            qc={"structural_ok": True, "metrics": {}, "warnings": [], "failures": []},
            currently_eligible=True,
            classification="eligible",
        )

    with stack:
        controller = ResultsPackagingController(runner=_slow_runner)
        controller.set_session(fixture["session_root"])
        page = ResultsPackagingPage(controller)
        page.show()
        page.refresh_from_state()
        t0 = time.perf_counter()
        page._run_packaging()
        accept_ms = (time.perf_counter() - t0) * 1000.0
        assert accept_ms < 100.0
        assert controller.state.task_running
        assert started["at"] is None or started["at"] >= t0
        with pytest.raises(ResultsPackagingUiError):
            controller.begin_packaging_launch()
        _wait_until(lambda: not page.has_running_task, timeout_sec=30.0)
        assert controller.state.last_result is not None
