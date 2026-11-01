"""Unit tests for nonblocking doctor inference advisories."""

from __future__ import annotations

from pathlib import Path

from cli.doctor_advisories import (
    DoctorAdvisory,
    collect_inference_advisories,
    collect_s7_dependency_advisories,
    collect_s7_registry_advisories,
    collect_sleap_nn_advisories,
    format_inference_advisories,
)
from pose_inference.runner import SleapNnProbeResult


def test_format_inference_advisories_prefixes_levels() -> None:
    text = format_inference_advisories(
        [
            DoctorAdvisory(level="ok", message="ready"),
            DoctorAdvisory(level="warn", message="missing"),
            DoctorAdvisory(level="info", message="note"),
        ]
    )
    assert "[OK] ready" in text
    assert "[WARN] missing" in text
    assert "[INFO] note" in text


def test_sleap_nn_advisory_warns_when_missing() -> None:
    probe = SleapNnProbeResult(
        available=False,
        compatible=False,
        executable_path=None,
        version=None,
        message="S2 not ready: sleap-nn missing or unusable (not found)",
    )
    advisories = collect_sleap_nn_advisories(sleap_probe=probe)
    assert advisories[0].level == "warn"
    assert "sleap-nn" in advisories[0].message


def test_sleap_nn_advisory_ok_when_compatible() -> None:
    probe = SleapNnProbeResult(
        available=True,
        compatible=True,
        executable_path=Path("C:/tools/sleap-nn.exe"),
        version="0.3.1",
        message="S2 sleap-nn: ready (0.3.1 at C:/tools/sleap-nn.exe)",
    )
    advisories = collect_sleap_nn_advisories(sleap_probe=probe)
    assert len(advisories) == 1
    assert advisories[0].level == "ok"


def test_s7_dependency_advisory_reports_status() -> None:
    advisories = collect_s7_dependency_advisories()
    assert len(advisories) == 1
    assert advisories[0].level in {"ok", "warn"}
    assert "S7" in advisories[0].message


def test_s7_registry_advisory_uses_suggested_models(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "dyadic" / "v1" / "approach").mkdir(parents=True)
    advisories = collect_s7_registry_advisories(settings=None, repo_root=tmp_path)
    assert advisories
    assert any("S7 registry" in item.message for item in advisories)


def test_collect_inference_advisories_never_raises_without_models(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "cli.doctor_advisories.probe_sleap_nn_runtime",
        lambda: SleapNnProbeResult(
            available=False,
            compatible=False,
            executable_path=None,
            version=None,
            message="S2 not ready: sleap-nn missing or unusable (not found)",
        ),
    )
    monkeypatch.setattr(
        "cli.doctor_advisories.suggest_lab_model_registry",
        lambda _repo_root=None: None,
    )
    monkeypatch.setattr(
        "cli.doctor_advisories.load_persisted_registry_root",
        lambda _settings: None,
    )
    advisories = collect_inference_advisories(repo_root=tmp_path)
    assert any(item.level == "warn" for item in advisories)
    text = format_inference_advisories(advisories)
    assert "nonblocking" in text
