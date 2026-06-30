from pathlib import Path

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


def test_windows_installer_uses_conda_run_and_doctor_without_powershell() -> None:
    text = Path("scripts/install_windows_gui.bat").read_text(encoding="utf-8")
    conda_lines = _active_conda_lines("scripts/install_windows_gui.bat")

    assert "powershell" not in text.lower()
    assert "conda activate" not in text.lower()
    assert all(line.lower().startswith("call conda ") for line in conda_lines)
    assert "call conda env update -f" in text
    assert "call conda run -n %ENV_NAME% python -m pip install -e ." in text
    assert '".[gui]"' not in text
    assert "from PySide6.QtWidgets import QApplication" in text
    assert "call conda run -n %ENV_NAME% behavior-suite doctor" in text
    assert "import cli.preprocess as p" in text
    assert "repo=Path.cwd().resolve()" in text
    assert "where conda" in text
    assert "SCRIPT_DIR=%~dp0" in text
    assert "REPO_ROOT=%SCRIPT_DIR%.." in text
    assert 'cd /d "%REPO_ROOT%"' in text
    assert "if errorlevel 1" in text.lower()
    assert "rmdir /s" not in text.lower()
    assert "del /s" not in text.lower()
    pip_install_lines = [
        line for line in text.splitlines() if "python -m pip install" in line
    ]
    assert pip_install_lines == [
        "call conda run -n %ENV_NAME% python -m pip install -e ."
    ]
    assert all("pyside" not in line.lower() for line in pip_install_lines)
    assert text.index("python -m pip install -e .") < text.index(
        "from PySide6.QtWidgets import QApplication"
    )
    assert text.index("from PySide6.QtWidgets import QApplication") < text.index(
        "behavior-suite doctor"
    )


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
