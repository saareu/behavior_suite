"""Shared FFmpeg/ffprobe executable discovery and runtime preflight."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from preprocess.exceptions import FFmpegRuntimeError

SUPPORTED_CONDA_ENV_NAME = "behavior_suite_gui"
SUPPORTED_WINDOWS_PYTHON = "3.12"
SUPPORTED_FFMPEG_MINOR = "7.1.x"
FFMPEG_VERSION_TIMEOUT_SEC = 10
FFMPEG_HELP_TIMEOUT_SEC = 20
FFMPEG_TIMESTAMP_CONTRACT_TIMEOUT_SEC = 10
INSTALLER_REMEDIATION = (
    r"Install or repair the supported Windows GUI runtime by running "
    r"scripts\install_windows_gui.bat from the repository root."
)

_BINARY_DIRECTORY_NAME = "bin"
_VERSION_CACHE: dict[Path, ExecutableStatus] = {}
_FFMPEG_CAPABILITY_CACHE: dict[Path, FFmpegCapabilityStatus] = {}
_PREFLIGHT_CACHE: dict[tuple[Path, Path], FFmpegRuntimePreflight] = {}


@dataclass(frozen=True, slots=True)
class ExecutableStatus:
    """Callable/version status for one resolved executable."""

    name: str
    path: Path | None
    callable: bool
    banner: str | None
    returncode: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FFmpegCapabilityStatus:
    """Required FFmpeg option and timestamp-contract support."""

    path: Path
    callable: bool
    fps_mode_supported: bool
    enc_time_base_demux_supported: bool
    returncode: int | None
    error: str | None = None
    modern_stage_timestamp_contract_supported: bool = False
    probe_command: tuple[str, ...] = ()
    probe_returncode: int | None = None
    probe_stderr: str | None = None


@dataclass(frozen=True, slots=True)
class FFmpegRuntimePreflight:
    """Complete runtime decision for the FFmpeg tool pair."""

    ffmpeg: ExecutableStatus
    ffprobe: ExecutableStatus
    capabilities: FFmpegCapabilityStatus | None
    supported: bool
    errors: tuple[str, ...]
    remediation: str = INSTALLER_REMEDIATION

    @property
    def fps_mode_supported(self) -> bool:
        return bool(self.capabilities and self.capabilities.fps_mode_supported)

    @property
    def enc_time_base_demux_supported(self) -> bool:
        return bool(self.capabilities and self.capabilities.enc_time_base_demux_supported)

    @property
    def modern_stage_timestamp_contract_supported(self) -> bool:
        return bool(
            self.capabilities
            and self.capabilities.modern_stage_timestamp_contract_supported
        )

    def raise_if_unsupported(self) -> None:
        """Raise a user-facing runtime error when any required check failed."""

        if self.supported:
            return
        detail = "\n".join(f"- {error}" for error in self.errors)
        raise FFmpegRuntimeError(
            "Unsupported FFmpeg runtime.\n"
            f"{detail}\n"
            f"Remedy: {self.remediation}"
        )


def clear_ffmpeg_runtime_cache() -> None:
    """Clear per-process executable/capability preflight caches."""

    _VERSION_CACHE.clear()
    _FFMPEG_CAPABILITY_CACHE.clear()
    _PREFLIGHT_CACHE.clear()


def _executable_name(binary_name: str) -> str:
    return f"{binary_name}.exe" if os.name == "nt" else binary_name


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundled_binary_candidates(binary_name: str) -> tuple[Path, ...]:
    executable_name = _executable_name(binary_name)
    candidates = [_repository_root() / _BINARY_DIRECTORY_NAME / executable_name]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        candidates.insert(0, Path(str(frozen_root)) / _BINARY_DIRECTORY_NAME / executable_name)
    return tuple(candidates)


def _active_conda_candidates(binary_name: str) -> tuple[Path, ...]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return ()
    prefix = Path(conda_prefix).expanduser()
    executable_name = _executable_name(binary_name)
    return (
        prefix / "Library" / "bin" / executable_name,
        prefix / "bin" / executable_name,
        prefix / "Scripts" / executable_name,
        prefix / executable_name,
    )


def _first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.expanduser().resolve()
    return None


def resolve_runtime_executable(binary_name: str, configured_path: Path | None = None) -> Path:
    """Resolve a runtime executable with Conda/bundled precedence over PATH.

    Explicit config paths are honored first and still pass through preflight.
    Without an explicit path, the active Conda environment is preferred when it
    contains the requested tool, then repository/frozen bundled binaries, and
    only then normal PATH discovery. This prevents Windows GUI launches through
    the supported Conda environment from accidentally selecting an older system
    ``ffmpeg.exe`` that appears earlier on global PATH.
    """

    if configured_path is not None:
        explicit_path = Path(configured_path).expanduser().resolve()
        if not explicit_path.is_file():
            raise FFmpegRuntimeError(
                f"Configured {binary_name} executable does not exist: {explicit_path}"
            )
        return explicit_path

    conda_candidate = _first_existing(_active_conda_candidates(binary_name))
    if conda_candidate is not None:
        return conda_candidate

    bundled_candidate = _first_existing(_bundled_binary_candidates(binary_name))
    if bundled_candidate is not None:
        return bundled_candidate

    discovered = shutil.which(binary_name)
    if discovered is None:
        raise FFmpegRuntimeError(
            f"{binary_name} executable was not found in the active Conda environment, "
            "bundled repository binaries, or PATH."
        )
    return Path(discovered).expanduser().resolve()


def resolve_ffmpeg_binary(configured_path: Path | None = None) -> Path:
    """Resolve the FFmpeg executable that the application will use."""

    return resolve_runtime_executable("ffmpeg", configured_path)


def resolve_ffprobe_binary(configured_path: Path | None = None) -> Path:
    """Resolve the ffprobe executable that the application will use."""

    return resolve_runtime_executable("ffprobe", configured_path)


def check_executable_version(name: str, path: Path) -> ExecutableStatus:
    """Run ``<tool> -version`` with a bounded subprocess and cache by path."""

    executable_path = Path(path).expanduser().resolve()
    cached = _VERSION_CACHE.get(executable_path)
    if cached is not None and cached.name == name:
        return cached
    try:
        completed = subprocess.run(
            [str(executable_path), "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=FFMPEG_VERSION_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status = ExecutableStatus(
            name=name,
            path=executable_path,
            callable=False,
            banner=None,
            returncode=None,
            error=str(exc),
        )
    else:
        output = (completed.stdout or completed.stderr or "").splitlines()
        banner = output[0].strip() if output else None
        status = ExecutableStatus(
            name=name,
            path=executable_path,
            callable=completed.returncode == 0,
            banner=banner,
            returncode=completed.returncode,
            error=None if completed.returncode == 0 else (completed.stderr.strip() or None),
        )
    _VERSION_CACHE[executable_path] = status
    return status


def build_timestamp_contract_probe_command(path: Path) -> tuple[str, ...]:
    """Return the synthetic FFmpeg command used to verify timestamp options."""

    return (
        str(Path(path).expanduser().resolve()),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=16x16:r=25:d=0.04",
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-enc_time_base",
        "demux",
        "-fps_mode",
        "passthrough",
        "-f",
        "null",
        "-",
    )


def check_ffmpeg_capabilities(path: Path) -> FFmpegCapabilityStatus:
    """Detect required modern FFmpeg timestamp option support."""

    executable_path = Path(path).expanduser().resolve()
    cached = _FFMPEG_CAPABILITY_CACHE.get(executable_path)
    if cached is not None:
        return cached
    help_returncode: int | None = None
    help_error: str | None = None
    help_callable = False
    fps_mode_supported_from_help = False
    try:
        help_completed = subprocess.run(
            [str(executable_path), "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
            check=False,
            timeout=FFMPEG_HELP_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        help_error = str(exc)
    else:
        help_returncode = help_completed.returncode
        help_callable = help_completed.returncode == 0
        help_text = f"{help_completed.stdout or ''}\n{help_completed.stderr or ''}"
        fps_mode_supported_from_help = "-fps_mode" in help_text
        if help_completed.returncode != 0:
            help_error = help_completed.stderr.strip() or None

    probe_command = build_timestamp_contract_probe_command(executable_path)
    probe_returncode: int | None
    probe_stderr: str | None
    try:
        probe_completed = subprocess.run(
            list(probe_command),
            capture_output=True,
            text=True,
            check=False,
            timeout=FFMPEG_TIMESTAMP_CONTRACT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        probe_returncode = None
        probe_stderr = str(exc)
    else:
        probe_returncode = probe_completed.returncode
        probe_stderr = probe_completed.stderr.strip() or None

    timestamp_contract_supported = probe_returncode == 0
    status = FFmpegCapabilityStatus(
        path=executable_path,
        callable=help_callable,
        fps_mode_supported=fps_mode_supported_from_help or timestamp_contract_supported,
        enc_time_base_demux_supported=timestamp_contract_supported,
        modern_stage_timestamp_contract_supported=timestamp_contract_supported,
        returncode=help_returncode,
        error=help_error,
        probe_command=probe_command,
        probe_returncode=probe_returncode,
        probe_stderr=probe_stderr,
    )
    _FFMPEG_CAPABILITY_CACHE[executable_path] = status
    return status


def preflight_ffmpeg_runtime(
    *,
    ffmpeg_path: Path | None = None,
    ffprobe_path: Path | None = None,
) -> FFmpegRuntimePreflight:
    """Resolve and validate the FFmpeg runtime before preprocessing starts."""

    errors: list[str] = []
    try:
        resolved_ffmpeg = resolve_ffmpeg_binary(ffmpeg_path)
    except FFmpegRuntimeError as exc:
        resolved_ffmpeg = None
        errors.append(str(exc))
    try:
        resolved_ffprobe = resolve_ffprobe_binary(ffprobe_path)
    except FFmpegRuntimeError as exc:
        resolved_ffprobe = None
        errors.append(str(exc))

    if resolved_ffmpeg is None:
        ffmpeg_status = ExecutableStatus(
            name="ffmpeg",
            path=None,
            callable=False,
            banner=None,
            returncode=None,
            error="ffmpeg executable could not be resolved.",
        )
    else:
        ffmpeg_status = check_executable_version("ffmpeg", resolved_ffmpeg)
    if resolved_ffprobe is None:
        ffprobe_status = ExecutableStatus(
            name="ffprobe",
            path=None,
            callable=False,
            banner=None,
            returncode=None,
            error="ffprobe executable could not be resolved.",
        )
    else:
        ffprobe_status = check_executable_version("ffprobe", resolved_ffprobe)

    cache_key = (
        ffmpeg_status.path or Path("<missing-ffmpeg>"),
        ffprobe_status.path or Path("<missing-ffprobe>"),
    )
    cached = _PREFLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    capabilities: FFmpegCapabilityStatus | None = None
    if not ffmpeg_status.callable:
        errors.append(
            f"ffmpeg is not callable: {ffmpeg_status.path or '<unresolved>'}"
            + (f" ({ffmpeg_status.error})" if ffmpeg_status.error else "")
        )
    else:
        assert ffmpeg_status.path is not None
        capabilities = check_ffmpeg_capabilities(ffmpeg_status.path)
        if not capabilities.modern_stage_timestamp_contract_supported:
            command = subprocess.list2cmdline(list(capabilities.probe_command))
            stderr = capabilities.probe_stderr or "no diagnostic output"
            errors.append(
                "ffmpeg modern Stage A timestamp-contract probe failed for "
                f"{capabilities.path}. "
                f"Banner: {ffmpeg_status.banner or 'unavailable'}. "
                "Required contract: -enc_time_base demux with "
                "-fps_mode passthrough. "
                f"Command: {command}. "
                f"Exit code: {capabilities.probe_returncode!r}. "
                f"Error: {stderr}"
            )
    if not ffprobe_status.callable:
        errors.append(
            f"ffprobe is not callable: {ffprobe_status.path or '<unresolved>'}"
            + (f" ({ffprobe_status.error})" if ffprobe_status.error else "")
        )

    preflight = FFmpegRuntimePreflight(
        ffmpeg=ffmpeg_status,
        ffprobe=ffprobe_status,
        capabilities=capabilities,
        supported=not errors,
        errors=tuple(errors),
    )
    _PREFLIGHT_CACHE[cache_key] = preflight
    return preflight


def ensure_supported_ffmpeg_runtime(
    *,
    ffmpeg_path: Path | None = None,
    ffprobe_path: Path | None = None,
) -> FFmpegRuntimePreflight:
    """Return supported preflight details or raise ``FFmpegRuntimeError``."""

    preflight = preflight_ffmpeg_runtime(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    )
    preflight.raise_if_unsupported()
    return preflight


def format_preflight_summary(preflight: FFmpegRuntimePreflight) -> str:
    """Return a concise human-readable supported/unsupported runtime summary."""

    ffmpeg_path = str(preflight.ffmpeg.path) if preflight.ffmpeg.path else "<not resolved>"
    ffprobe_path = str(preflight.ffprobe.path) if preflight.ffprobe.path else "<not resolved>"
    status = "supported" if preflight.supported else "unsupported"
    lines = [
        f"FFmpeg runtime: {status}",
        f"ffmpeg: {ffmpeg_path}",
        f"ffmpeg banner: {preflight.ffmpeg.banner or 'unavailable'}",
        f"ffmpeg callable: {'yes' if preflight.ffmpeg.callable else 'no'}",
        f"ffprobe: {ffprobe_path}",
        f"ffprobe banner: {preflight.ffprobe.banner or 'unavailable'}",
        f"ffprobe callable: {'yes' if preflight.ffprobe.callable else 'no'}",
        f"supports_fps_mode: {'yes' if preflight.fps_mode_supported else 'no'}",
        (
            "supports_named_enc_time_base_demux: yes"
            if preflight.enc_time_base_demux_supported
            else "supports_named_enc_time_base_demux: no"
        ),
        (
            "supports_modern_stage_timestamp_contract: yes"
            if preflight.modern_stage_timestamp_contract_supported
            else "supports_modern_stage_timestamp_contract: no"
        ),
    ]
    if preflight.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in preflight.errors)
        lines.append(f"Remedy: {preflight.remediation}")
    return "\n".join(lines)
