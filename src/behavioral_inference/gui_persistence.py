"""Persist Subsystem 07 GUI convenience state (model-registry root).

Uses the same QSettings organization as Subsystem 02 path persistence.
Tracked suite YAML keeps ``model_registry.root: null``; operators configure a
trusted local models directory once and subsequent launches reuse it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SettingsStore(Protocol):
    """Minimal QSettings-compatible store."""

    def value(self, key: str, defaultValue: object = None) -> object: ...

    def setValue(self, key: str, value: object) -> None: ...


_SETTINGS_PREFIX = "s7/"
REGISTRY_ROOT_KEY = _SETTINGS_PREFIX + "last_model_registry_root"


def load_persisted_registry_root(settings: SettingsStore | None) -> Path | None:
    """Return a previously saved registry root when the path still exists."""

    if settings is None:
        return None
    value = settings.value(REGISTRY_ROOT_KEY, "")
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None
    path = Path(raw).expanduser().resolve(strict=False)
    if path.is_dir():
        return path
    return path  # Keep text so the GUI can show "unavailable" + recovery.


def persist_registry_root(
    settings: SettingsStore | None, root: Path | str | None
) -> None:
    """Save or clear the last trusted model-registry root."""

    if settings is None:
        return
    if root is None or str(root).strip() == "":
        settings.setValue(REGISTRY_ROOT_KEY, "")
        return
    path = Path(root).expanduser().resolve(strict=False)
    settings.setValue(REGISTRY_ROOT_KEY, str(path))


def suggest_lab_model_registry(repo_root: Path | str | None = None) -> Path | None:
    """Return a local laboratory models directory when present (never implied as default YAML).

    Used only as a browse-dialog starting point / first-run suggestion when the
    operator has not yet persisted a registry root.
    """

    if repo_root is None:
        # src/behavioral_inference/gui_persistence.py -> repo root
        base = Path(__file__).resolve().parents[2]
    else:
        base = Path(repo_root).expanduser().resolve(strict=False)
    candidate = base / "legacy" / "models"
    return candidate if candidate.is_dir() else None
