import importlib.util
import tomllib
from pathlib import Path
from types import SimpleNamespace

import yaml


def _script_lines(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def _active_conda_lines(path: str) -> list[str]:
    lines = []
    for raw_line in _script_lines(path):
        line = raw_line.strip()
        if not line or line.lower().startswith("rem "):
            continue
        lowered = line.lower()
        if lowered.startswith("conda ") or lowered.startswith("call conda "):
            lines.append(line)
    return lines


def test_environment_gui_is_minimal_pinned_conda_runtime() -> None:
    environment = yaml.safe_load(Path("environment-gui.yml").read_text(encoding="utf-8"))

    assert environment["name"] == "behavior_suite_gui"
    assert environment["channels"] == ["conda-forge"]
    dependencies = environment["dependencies"]
    assert "python=3.12" in dependencies
    assert "ffmpeg=7.1.1" in dependencies
    assert "pyside6=6.11.1" in dependencies
    assert "pip" in dependencies
    assert not any(
        isinstance(item, dict)
        and "pip" in item
        and any("-e" in str(value) for value in item["pip"])
        for item in dependencies
    )


def test_s2_optional_dependency_pins_sleap_io() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["optional-dependencies"]["s2"] == ["sleap-io==0.8.0"]


def test_windows_installer_recreates_dedicated_environment_without_pip_uninstall() -> None:
    text = Path("scripts/install_windows_gui.bat").read_text(encoding="utf-8")
    conda_lines = _active_conda_lines("scripts/install_windows_gui.bat")

    assert "powershell" not in text.lower()
    assert "conda activate" not in text.lower()
    assert all(line.lower().startswith("call conda ") for line in conda_lines)
    assert "call conda env list" in text
    assert "call conda env remove -n %ENV_NAME% -y" in text
    assert 'call conda env create -f "%ENV_FILE%" -y' in text
    assert "conda env update" not in text
    assert "pip uninstall" not in text
    assert 'call conda run -n %ENV_NAME% python -m pip install -e ".[s2]"' in text
    assert '".[gui]"' not in text
    assert "python scripts\\validate_windows_gui_runtime.py" in text
    assert "python -m pip check" in text
    assert "call conda run -n %ENV_NAME% behavior-suite doctor" in text
    assert "import cli.preprocess as p" in text
    assert "repo=Path.cwd().resolve()" in text
    assert "where conda" in text
    assert "SCRIPT_DIR=%~dp0" in text
    assert "REPO_ROOT=%SCRIPT_DIR%.." in text
    assert 'cd /d "%REPO_ROOT%"' in text
    assert "if errorlevel 1" in text.lower()
    assert "broken pip metadata" in text
    assert "Conda-forge PySide6 6.11.1 and sleap-io 0.8.0" in text
    assert "rmdir /s" not in text.lower()
    assert "del /s" not in text.lower()
    pip_install_lines = [
        line for line in text.splitlines() if "python -m pip install" in line
    ]
    assert pip_install_lines == [
        'call conda run -n %ENV_NAME% python -m pip install -e ".[s2]"'
    ]
    assert all("pyside" not in line.lower() for line in pip_install_lines)
    assert "python -m pip install -U PySide6" not in text
    assert "python -m pip install PySide6" not in text
    assert text.index("conda env remove -n %ENV_NAME%") < text.index(
        'conda env create -f "%ENV_FILE%"'
    )
    assert text.index('conda env create -f "%ENV_FILE%"') < text.index(
        'python -m pip install -e ".[s2]"'
    )
    assert text.index('python -m pip install -e ".[s2]"') < text.index(
        "Python startup: ok"
    )
    assert text.index("Python startup: ok") < text.index(
        "python scripts\\validate_windows_gui_runtime.py"
    )
    assert text.index("python scripts\\validate_windows_gui_runtime.py") < text.index(
        "python -m pip check"
    )
    assert text.index("python -m pip check") < text.index("behavior-suite doctor")


def test_windows_runtime_validator_imports_full_application_surface(
    monkeypatch,
    capsys,
) -> None:
    path = Path("scripts/validate_windows_gui_runtime.py")
    spec = importlib.util.spec_from_file_location("validate_windows_gui_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    imported: list[str] = []

    def fake_import(name: str):
        imported.append(name)
        versions = {"PySide6": "6.11.1", "sleap_io": "0.8.0"}
        return SimpleNamespace(__version__=versions.get(name))

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    assert module.main() == 0
    assert imported == [
        *module.DEPENDENCY_MODULES,
        *module.QT_MODULES,
        *module.S2_DEPENDENCY_MODULES,
        *module.S2_ARTIFACT_MODULES,
        *module.APPLICATION_MODULES,
    ]
    output = capsys.readouterr().out
    assert "PySide6 import/version: ok (6.11.1)" in output
    assert "sleap-io import/version: ok (0.8.0)" in output
    assert "Application dependency imports: ok" in output
    assert "S2 artifact module imports: ok" in output


def test_windows_runtime_validator_fails_when_sleap_io_is_missing(
    monkeypatch,
    capsys,
) -> None:
    path = Path("scripts/validate_windows_gui_runtime.py")
    spec = importlib.util.spec_from_file_location("validate_windows_gui_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_import(name: str):
        if name == "sleap_io":
            raise ModuleNotFoundError("No module named 'sleap_io'")
        return SimpleNamespace(__version__="6.11.1" if name == "PySide6" else None)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    assert module.main() == 1
    output = capsys.readouterr().out
    assert "Windows GUI runtime validation failed:" in output
    assert "sleap_io: ModuleNotFoundError: No module named 'sleap_io'" in output


def test_windows_runtime_validator_rejects_incompatible_sleap_io_version(
    monkeypatch,
    capsys,
) -> None:
    path = Path("scripts/validate_windows_gui_runtime.py")
    spec = importlib.util.spec_from_file_location("validate_windows_gui_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_import(name: str):
        versions = {"PySide6": "6.11.1", "sleap_io": "0.9.0"}
        return SimpleNamespace(__version__=versions.get(name))

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    assert module.main() == 1
    output = capsys.readouterr().out
    assert "sleap-io version mismatch: expected 0.8.0, found 0.9.0" in output


def test_windows_launcher_checks_environment_doctor_then_gui() -> None:
    text = Path("scripts/launch_windows_gui.bat").read_text(encoding="utf-8")
    conda_lines = _active_conda_lines("scripts/launch_windows_gui.bat")

    assert "powershell" not in text.lower()
    assert "conda activate" not in text.lower()
    assert all(line.lower().startswith("call conda ") for line in conda_lines)
    assert "call conda run -n %ENV_NAME% python -c" in text
    assert "call conda run -n %ENV_NAME% behavior-suite doctor" in text
    assert "call conda run --live-stream -n %ENV_NAME% behavior-suite gui" in text
    assert "install_windows_gui.bat" in text
    assert "where conda" in text
    assert "SCRIPT_DIR=%~dp0" in text
    assert "REPO_ROOT=%SCRIPT_DIR%.." in text
    assert 'cd /d "%REPO_ROOT%"' in text
    assert "if errorlevel 1" in text.lower()
    assert text.index("behavior-suite doctor") < text.index("behavior-suite gui")
