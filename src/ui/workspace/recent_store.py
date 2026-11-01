"""Remember-only recent workspace targets via QSettings (W4(c))."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings

from workspace.models import RecentTargetEntry

_SETTINGS_ORG = "BehaviorSuite"
_SETTINGS_APP = "BehaviorSuite"
_RECENT_KEY = "workspace/recent_targets_v1"
_MAX_RECENT = 12


class WorkspaceRecentStore:
    """Persist recent analysis targets without science run ids or auto-restore."""

    def __init__(
        self,
        settings: QSettings | None = None,
        *,
        max_entries: int = _MAX_RECENT,
    ) -> None:
        self._settings = settings or QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._max_entries = max(1, max_entries)

    def load(self) -> list[RecentTargetEntry]:
        raw = self._settings.value(_RECENT_KEY, "")
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        entries: list[RecentTargetEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(RecentTargetEntry.model_validate(item))
            except Exception:
                continue
        return entries[: self._max_entries]

    def save(self, entries: list[RecentTargetEntry] | tuple[RecentTargetEntry, ...]) -> None:
        trimmed = list(entries)[: self._max_entries]
        payload = [entry.model_dump(mode="json") for entry in trimmed]
        self._settings.setValue(_RECENT_KEY, json.dumps(payload))
        self._settings.sync()

    def remember(self, entry: RecentTargetEntry) -> list[RecentTargetEntry]:
        current = [item for item in self.load() if item != entry]
        updated = [entry, *current][: self._max_entries]
        self.save(updated)
        return updated

    def clear(self) -> None:
        self._settings.remove(_RECENT_KEY)
        self._settings.sync()

    @staticmethod
    def path_label(path: Path | None) -> str:
        if path is None:
            return "(none)"
        return str(path)
