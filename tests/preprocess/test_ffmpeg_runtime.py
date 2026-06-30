from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from preprocess import ffmpeg_runtime

MODERN_HELP = """
Global options:
  -fps_mode mode      set framerate mode for matching video streams
  -enc_time_base ratio  set the desired time base for the encoder
"""

LEGACY_HELP = """
Global options:
  -vsync parameter    video sync method
  -enc_time_base ratio  set the desired time base for the encoder
"""


def _touch(path: Path) -> Path:
    path.write_text("fake executable", encoding="utf-8")
    return path


class _Completed:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    help_text: str = MODERN_HELP,
    probe_returncode: int = 0,
    probe_stderr: str = "",
    calls: list[list[str]] | None = None,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> _Completed:
        assert kwargs["shell"] is not True if "shell" in kwargs else True
        if calls is not None:
            calls.append(list(command))
        executable_name = Path(command[0]).stem
        if command[1:] == ["-version"]:
            return _Completed(stdout=f"{executable_name} version 7.1.1\n")
        if command[1:] == ["-hide_banner", "-h", "full"]:
            return _Completed(stdout=help_text)
        if "-enc_time_base" in command and "-fps_mode" in command:
            assert command[command.index("-enc_time_base") + 1] == "demux"
            assert command[command.index("-fps_mode") + 1] == "passthrough"
            assert command[-2:] == ["null", "-"]
            return _Completed(stderr=probe_stderr, returncode=probe_returncode)
        raise AssertionError(f"Unexpected command: {command!r}")

    monkeypatch.setattr(ffmpeg_runtime.subprocess, "run", fake_run)


@pytest.fixture(autouse=True)
def _clear_runtime_cache() -> None:
    ffmpeg_runtime.clear_ffmpeg_runtime_cache()


def test_help_without_demux_but_successful_behavioral_probe_passes_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = _touch(tmp_path / "ffmpeg.exe")
    ffprobe = _touch(tmp_path / "ffprobe.exe")
    _install_fake_run(monkeypatch)

    result = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )

    assert result.supported is True
    assert result.fps_mode_supported is True
    assert result.enc_time_base_demux_supported is True
    assert result.modern_stage_timestamp_contract_supported is True
    assert result.ffmpeg.path == ffmpeg.resolve()
    assert result.ffprobe.path == ffprobe.resolve()


def test_behavioral_probe_fps_mode_failure_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = _touch(tmp_path / "ffmpeg.exe")
    ffprobe = _touch(tmp_path / "ffprobe.exe")
    _install_fake_run(
        monkeypatch,
        help_text=LEGACY_HELP,
        probe_returncode=1,
        probe_stderr="Unrecognized option 'fps_mode'.",
    )

    result = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )

    assert result.supported is False
    assert result.fps_mode_supported is False
    assert result.enc_time_base_demux_supported is False
    assert result.modern_stage_timestamp_contract_supported is False
    assert any("Unrecognized option 'fps_mode'" in error for error in result.errors)
    assert any("modern Stage A timestamp-contract probe failed" in error for error in result.errors)
    assert any("Command:" in error for error in result.errors)
    assert "install_windows_gui.bat" in result.remediation


def test_behavioral_probe_demux_failure_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = _touch(tmp_path / "ffmpeg.exe")
    ffprobe = _touch(tmp_path / "ffprobe.exe")
    _install_fake_run(
        monkeypatch,
        help_text=MODERN_HELP,
        probe_returncode=1,
        probe_stderr="Unable to parse option value \"demux\"",
    )

    result = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )

    assert result.supported is False
    assert result.fps_mode_supported is True
    assert result.enc_time_base_demux_supported is False
    assert result.modern_stage_timestamp_contract_supported is False
    assert any("Unable to parse option value" in error for error in result.errors)
    assert any("-enc_time_base demux" in error for error in result.errors)


def test_missing_ffmpeg_fails_with_discovery_diagnostics(tmp_path: Path) -> None:
    ffprobe = _touch(tmp_path / "ffprobe.exe")

    result = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=tmp_path / "missing-ffmpeg.exe",
        ffprobe_path=ffprobe,
    )

    assert result.supported is False
    assert result.ffmpeg.callable is False
    assert any("Configured ffmpeg executable does not exist" in error for error in result.errors)


def test_missing_ffprobe_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = _touch(tmp_path / "ffmpeg.exe")
    _install_fake_run(monkeypatch)

    result = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg,
        ffprobe_path=tmp_path / "missing-ffprobe.exe",
    )

    assert result.supported is False
    assert result.ffprobe.callable is False
    assert any("Configured ffprobe executable does not exist" in error for error in result.errors)


def test_preflight_results_are_cached_per_executable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = _touch(tmp_path / "ffmpeg.exe")
    ffprobe = _touch(tmp_path / "ffprobe.exe")
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, calls=calls)

    first = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )
    second = ffmpeg_runtime.preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )

    assert first is second
    assert len(calls) == 4


def test_runtime_discovery_and_preflight_do_not_use_shell_true() -> None:
    tree = ast.parse(Path(ffmpeg_runtime.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in {"run", "Popen"}:
            assert not any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
