"""S7 installation help, registry persistence, and readiness remediation tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from behavioral_inference import (
    READINESS_MISSING_RUNTIME_DEPS,
    READINESS_READY,
    READINESS_REGISTRY_NOT_CONFIGURED,
    S7_RUNTIME_INSTALL_HINT,
)
from behavioral_inference.contracts import OptionalDependencyError
from behavioral_inference.gui_persistence import (
    REGISTRY_ROOT_KEY,
    load_persisted_registry_root,
    persist_registry_root,
)
from behavioral_inference.loader import require_runtime_dependencies
from project.file_digest import clear_digest_cache, sha256_file
from tests.ui.test_behavioral_inference_gui import (
    _patch_runtime_deps,
    _patch_s6_ok,
    _ready_controller,
    _wait_until,
    _write_fake_registry,
    _write_s6_run,
)
from ui.controllers.behavioral_inference_controller import (
    BehavioralInferenceController,
    BehavioralInferenceUiSettings,
    DiscoveryRequestIdentity,
    DiscoverySnapshot,
)
from ui.pages.behavioral_inference_page import BehavioralInferencePage

os_environ_qt = __import__("os").environ
os_environ_qt.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


class _FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, defaultValue: object = None) -> object:
        return self.values.get(key, defaultValue)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value


def test_install_help_is_powershell_safe() -> None:
    assert '".[s7]"' not in S7_RUNTIME_INSTALL_HINT
    assert ".$7" not in S7_RUNTIME_INSTALL_HINT
    assert "'.[s2,s7]'" in S7_RUNTIME_INSTALL_HINT
    assert "install_windows_gui.bat" in S7_RUNTIME_INSTALL_HINT


def test_optional_dependency_error_uses_safe_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from behavioral_inference.loader import RuntimeDependencyStatus

    monkeypatch.setattr(
        "behavioral_inference.loader.probe_runtime_dependencies",
        lambda: RuntimeDependencyStatus(
            available=False,
            versions={"xgboost": None, "sklearn": None, "joblib": None},
            missing=("xgboost", "sklearn", "joblib"),
            import_errors={},
        ),
    )
    with pytest.raises(OptionalDependencyError) as excinfo:
        require_runtime_dependencies()
    message = str(excinfo.value)
    assert "'.[s2,s7]'" in message
    assert '".[s7]"' not in message


def test_registry_configuration_persists_across_controller_restarts(
    tmp_path: Path,
) -> None:
    settings = _FakeSettings()
    registry = tmp_path / "models"
    registry.mkdir()
    persist_registry_root(settings, registry)
    assert settings.values[REGISTRY_ROOT_KEY] == str(registry.resolve())

    first = BehavioralInferenceController(settings=settings)
    assert first.state.working_settings.model_registry_root == registry.resolve()

    second = BehavioralInferenceController(settings=settings)
    assert second.state.working_settings.model_registry_root == registry.resolve()
    assert load_persisted_registry_root(settings) == registry.resolve()


def test_missing_registry_does_not_crash_gui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    controller = BehavioralInferenceController(settings=_FakeSettings())
    controller.set_session(session)
    page = BehavioralInferencePage(controller)
    page.refresh_from_state()
    assert controller.state.readiness_kind == READINESS_REGISTRY_NOT_CONFIGURED
    assert controller.can_run_inference() is False
    assert "Registry not configured" in (controller.state.readiness_error or "")


def test_valid_registry_enables_detector_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _session, registry = _ready_controller(monkeypatch, tmp_path)
    assert controller.state.readiness_kind == READINESS_READY
    assert controller.can_run_inference() is True
    assert any(d.ready for d in controller.state.detector_availability)
    assert registry.is_dir()


def test_missing_runtime_deps_distinct_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from behavioral_inference.loader import RuntimeDependencyStatus

    status = RuntimeDependencyStatus(
        available=False,
        versions={"xgboost": None, "sklearn": None, "joblib": None},
        missing=("xgboost", "sklearn", "joblib"),
        import_errors={},
    )
    monkeypatch.setattr(
        "ui.controllers.behavioral_inference_controller.probe_runtime_dependencies",
        lambda: status,
    )
    session = tmp_path / "session"
    session.mkdir()
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    registry = _write_fake_registry(
        tmp_path / "models",
        names={"nose_to_nose": "dyadic", "approach": "focal"},
    )
    controller = BehavioralInferenceController()
    controller.set_session(session)
    controller.update_settings(
        BehavioralInferenceUiSettings(
            model_registry_root=registry,
            selected_detectors_dyadic=("nose_to_nose",),
            selected_detectors_focal=("approach",),
        )
    )
    assert controller.state.readiness_kind == READINESS_MISSING_RUNTIME_DEPS
    assert "'.[s2,s7]'" in controller.state.runtime_deps_detail
    assert controller.state.python_executable


def test_digest_cache_avoids_rehash(tmp_path: Path) -> None:
    clear_digest_cache()
    path = tmp_path / "blob.bin"
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    t0 = time.perf_counter()
    first = sha256_file(path)
    t1 = time.perf_counter()
    second = sha256_file(path)
    t2 = time.perf_counter()
    assert first == second
    assert (t2 - t1) < (t1 - t0)


def test_stale_discovery_results_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    session_a.mkdir()
    session_b.mkdir()
    s6_a = _write_s6_run(session_a, n_frames=2)
    s6_b = _write_s6_run(session_b, n_frames=2)
    _patch_s6_ok(monkeypatch, s6_a, session_a)
    controller = BehavioralInferenceController()
    first = controller.prepare_session(session_a)
    snapshot = controller.collect_discovery_snapshot(first)
    second = controller.prepare_session(session_b)
    assert second.generation > first.generation
    assert controller.apply_discovery_snapshot(snapshot) is False
    assert controller.state.discovery_running is True
    _patch_s6_ok(monkeypatch, s6_b, session_b)
    fresh = controller.collect_discovery_snapshot(second)
    assert controller.apply_discovery_snapshot(fresh) is True
    assert controller.state.session_root == session_b.resolve()
    assert controller.state.discovery_running is False


def test_gui_stays_responsive_during_background_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_runtime_deps(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)

    original = BehavioralInferenceController.collect_discovery_snapshot

    def _slow(
        self: BehavioralInferenceController, identity: DiscoveryRequestIdentity
    ) -> DiscoverySnapshot:
        time.sleep(0.25)
        return original(self, identity)

    monkeypatch.setattr(
        BehavioralInferenceController,
        "collect_discovery_snapshot",
        _slow,
    )
    controller = BehavioralInferenceController(settings=_FakeSettings())
    page = BehavioralInferencePage(controller)
    started = time.perf_counter()
    page.set_session(session)
    # set_session must return before discovery finishes.
    assert time.perf_counter() - started < 0.2
    assert page.has_running_task is True
    events = 0

    def _pump() -> bool:
        nonlocal events
        events += 1
        _APPLICATION.processEvents()
        return not page.discovery_runner.is_running

    _wait_until(_pump, timeout_sec=5.0)
    assert events > 1
    assert controller.state.discovery_running is False
    assert controller.state.session_root == session.resolve()


def test_pyproject_and_installer_agree_on_s7_extra() -> None:
    text = Path("scripts/install_windows_gui.bat").read_text(encoding="utf-8")
    assert 'pip install -e ".[s2,s7]"' in text
    assert "Model binaries are not installed" in text
