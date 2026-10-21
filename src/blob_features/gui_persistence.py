"""Persist S5 working settings keyed by eligible S1 source identity.

Working settings are diagnostic GUI state associated with the current
``prepared_sync`` / ``prepared_video`` digests. They never rewrite
``settings_used.yaml`` of a completed extraction run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blob_features.config import (
    BlobExecutionSettings,
    BlobFeaturesConfig,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    load_blob_features_config,
)
from blob_features.contracts import BlobFeaturesError
from blob_features.s1_eligibility import EligibleS1Input

WORKING_SETTINGS_SCHEMA_VERSION = "s5_working_settings_v1"
WORKING_SETTINGS_FILENAME = "working_settings.json"
_GUI_SUBDIR = "_gui"


@dataclass(frozen=True, slots=True)
class S1SourceIdentity:
    """Stable identity for one eligible S1 preprocess source.

    Includes the cropped background digest so background-only changes invalidate
    preview caches and persisted working-settings matches.
    """

    session_root: Path
    preprocess_dir: Path
    prepared_video: Path
    prepared_sync: Path
    cropped_background: Path
    prepared_video_sha256: str
    prepared_sync_sha256: str
    cropped_background_sha256: str

    @classmethod
    def from_eligible(cls, eligible: EligibleS1Input) -> S1SourceIdentity:
        return cls(
            session_root=eligible.session_root,
            preprocess_dir=eligible.preprocess_dir,
            prepared_video=eligible.prepared_video,
            prepared_sync=eligible.prepared_sync,
            cropped_background=eligible.cropped_background,
            prepared_video_sha256=_sha256(eligible.prepared_video),
            prepared_sync_sha256=_sha256(eligible.prepared_sync),
            cropped_background_sha256=_sha256(eligible.cropped_background),
        )

    def cache_key(self) -> str:
        """Compact identity string for in-memory preview cache keys."""

        return (
            f"{self.prepared_video_sha256.lower()}:"
            f"{self.cropped_background_sha256.lower()}:"
            f"{self.prepared_sync_sha256.lower()}"
        )


def working_settings_path_for_session(session_root: Path | str) -> Path:
    root = Path(session_root).expanduser().resolve(strict=False)
    return root / "blob_features" / _GUI_SUBDIR / WORKING_SETTINGS_FILENAME


def load_working_settings_for_source(
    identity: S1SourceIdentity,
) -> BlobFeaturesConfig | None:
    """Load persisted working settings when they match the current S1 identity."""

    path = working_settings_path_for_session(identity.session_root)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    if str(raw.get("schema_version", "")) != WORKING_SETTINGS_SCHEMA_VERSION:
        return None
    if str(raw.get("prepared_sync_sha256", "")).lower() != identity.prepared_sync_sha256.lower():
        return None
    if str(raw.get("prepared_video_sha256", "")).lower() != identity.prepared_video_sha256.lower():
        return None
    # Background digest is required when present in newer saves; older saves that
    # omit it remain valid so explicit user settings are not silently discarded.
    recorded_bg = raw.get("cropped_background_sha256")
    if (
        isinstance(recorded_bg, str)
        and recorded_bg.strip()
        and recorded_bg.lower() != identity.cropped_background_sha256.lower()
    ):
        return None
    settings_raw = raw.get("settings")
    if not isinstance(settings_raw, Mapping):
        return None
    try:
        seg = _mapping_to_kwargs(settings_raw.get("segmentation"))
        temporal = _mapping_to_kwargs(settings_raw.get("temporal"))
        execution = _mapping_to_kwargs(settings_raw.get("execution"))
        return BlobFeaturesConfig(
            segmentation=BlobSegmentationSettings(**seg),
            temporal=BlobTemporalSettings(**temporal),
            execution=BlobExecutionSettings(**execution),
            target_fps=float(settings_raw.get("target_fps", 120.0)),
            schema_version=str(
                settings_raw.get("schema_version", BlobFeaturesConfig().schema_version)
            ),
        ).validated()
    except (BlobFeaturesError, TypeError, ValueError, KeyError):
        return None


def save_working_settings_for_source(
    identity: S1SourceIdentity,
    config: BlobFeaturesConfig,
) -> Path:
    """Atomically persist working settings for the given S1 identity."""

    validated = config.validated()
    target = working_settings_path_for_session(identity.session_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": WORKING_SETTINGS_SCHEMA_VERSION,
        "session_root": str(identity.session_root),
        "preprocess_dir": str(identity.preprocess_dir),
        "prepared_video": str(identity.prepared_video),
        "prepared_sync": str(identity.prepared_sync),
        "cropped_background": str(identity.cropped_background),
        "prepared_video_sha256": identity.prepared_video_sha256,
        "prepared_sync_sha256": identity.prepared_sync_sha256,
        "cropped_background_sha256": identity.cropped_background_sha256,
        "settings": validated.to_settings_dict(),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise BlobFeaturesError(f"Could not save working settings: {exc}") from exc
    return target


def default_working_config() -> BlobFeaturesConfig:
    return load_blob_features_config()


def _mapping_to_kwargs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
