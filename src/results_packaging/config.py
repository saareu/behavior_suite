"""Versioned Subsystem 09 configuration (Sprint 1–3 packaging settings)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from results_packaging.contracts import (
    DEFAULT_PROFILE_ID,
    PACKAGE_MODE_REFERENCE_LIGHTWEIGHT,
    REVIEW_EMIT_PROFILE_ID,
    REVIEW_EMIT_V1_INCLUDE_CONTACT,
    REVIEW_EMIT_V1_INCLUDE_ENGAGED,
    REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE,
    REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES,
    REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS,
    REVIEW_EMIT_V1_TARGET_FPS,
    SETTINGS_SCHEMA_VERSION,
    SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC,
    TARGET_FPS,
    ConfigError,
)

_SUPPORTED_EXPORT_PROFILES = frozenset({REVIEW_EMIT_PROFILE_ID})


@dataclass(frozen=True, slots=True)
class ReviewEmitV1Defaults:
    """Approved ``review_emit_v1`` defaults (U-R1/U-R2)."""

    include_contact: bool = REVIEW_EMIT_V1_INCLUDE_CONTACT
    include_engaged: bool = REVIEW_EMIT_V1_INCLUDE_ENGAGED
    include_side_by_side: bool = REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE
    min_emit_bout_frames: int = REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES
    min_emit_bout_seconds: float = REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS
    target_fps: float = REVIEW_EMIT_V1_TARGET_FPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_contact": self.include_contact,
            "include_engaged": self.include_engaged,
            "include_side_by_side": self.include_side_by_side,
            "min_emit_bout_frames": int(self.min_emit_bout_frames),
            "min_emit_bout_seconds": float(self.min_emit_bout_seconds),
            "target_fps": float(self.target_fps),
            "note": (
                "Seconds→frames uses explicit 120 Hz (deliberate difference from "
                "legacy Pipeline 110 source_video.fps). Applied only when "
                "review_emit_v1 is explicitly selected."
            ),
        }


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Non-scientific packaging execution flags for Sprint 1–3."""

    sync_construction_method: str = SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC
    package_mode: str = PACKAGE_MODE_REFERENCE_LIGHTWEIGHT
    require_prepared_video: bool = True
    # Informational for settings snapshots; publication entry points always publish.
    publish_run: bool = False
    # Optional full prepared-video SHA256 (U-R6). Never implicit on GUI polls.
    hash_prepared_video: bool = False
    selected_export_profiles: tuple[str, ...] = ()

    def validated(self) -> ExecutionSettings:
        method = str(self.sync_construction_method).strip()
        if method != SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC:
            raise ConfigError(
                "S9 MVP supports only sync_construction_method="
                f"{SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC!r}; got {method!r}."
            )
        mode = str(self.package_mode).strip()
        if mode != PACKAGE_MODE_REFERENCE_LIGHTWEIGHT:
            raise ConfigError(
                "S9 MVP supports only package_mode="
                f"{PACKAGE_MODE_REFERENCE_LIGHTWEIGHT!r} (U-R3: portable "
                f"embed deferred); got {mode!r}."
            )
        profiles = tuple(
            str(p).strip() for p in self.selected_export_profiles if str(p).strip()
        )
        unknown = [p for p in profiles if p not in _SUPPORTED_EXPORT_PROFILES]
        if unknown:
            raise ConfigError(
                "Unsupported export profile(s) for S9 MVP: "
                f"{unknown!r}. Supported: {sorted(_SUPPORTED_EXPORT_PROFILES)}. "
                "Symmetric collapse is deferred (U-R7)."
            )
        return replace(
            self,
            sync_construction_method=method,
            package_mode=mode,
            selected_export_profiles=profiles,
        )


@dataclass(frozen=True, slots=True)
class ResultsPackagingConfig:
    """Headless packaging configuration (Sprint 1–3)."""

    schema_version: str = SETTINGS_SCHEMA_VERSION
    fps: float = TARGET_FPS
    default_profile_id: str = DEFAULT_PROFILE_ID
    execution: ExecutionSettings = ExecutionSettings()
    review_emit_v1_defaults: ReviewEmitV1Defaults = ReviewEmitV1Defaults()

    def validated(self) -> ResultsPackagingConfig:
        if self.schema_version != SETTINGS_SCHEMA_VERSION:
            raise ConfigError(
                f"Unsupported settings schema_version {self.schema_version!r}; "
                f"expected {SETTINGS_SCHEMA_VERSION!r}."
            )
        if abs(float(self.fps) - TARGET_FPS) > 1e-9:
            raise ConfigError(
                f"S9 target fps must be {TARGET_FPS}; got {self.fps!r}."
            )
        if self.default_profile_id != DEFAULT_PROFILE_ID:
            raise ConfigError(
                f"default_profile_id must be {DEFAULT_PROFILE_ID!r}."
            )
        review = self.review_emit_v1_defaults
        if abs(float(review.target_fps) - TARGET_FPS) > 1e-9:
            raise ConfigError(
                "review_emit_v1_defaults.target_fps must be 120.0 "
                f"(got {review.target_fps!r})."
            )
        return replace(
            self,
            fps=float(self.fps),
            execution=self.execution.validated(),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        cfg = self.validated()
        return {
            "schema_version": cfg.schema_version,
            "fps": cfg.fps,
            "default_profile_id": cfg.default_profile_id,
            "execution": {
                "sync_construction_method": cfg.execution.sync_construction_method,
                "package_mode": cfg.execution.package_mode,
                "require_prepared_video": cfg.execution.require_prepared_video,
                "publish_run": cfg.execution.publish_run,
                "hash_prepared_video": cfg.execution.hash_prepared_video,
                "selected_export_profiles": list(cfg.execution.selected_export_profiles),
            },
            "approved_profile_defaults": {
                REVIEW_EMIT_PROFILE_ID: cfg.review_emit_v1_defaults.to_dict(),
            },
            "notes": {
                "default_package_retains_all_s8_bouts": True,
                "profiles_do_not_change_s8_bout_boundaries": True,
                "review_emit_seconds_use_120hz": True,
                "deliberate_difference_from_pipeline_110_source_fps": True,
                "review_emit_not_applied_unless_explicitly_selected": True,
                "portable_embed_deferred_u_r3": True,
                "clips_gifs_deferred_u_r4": True,
                "symmetric_collapse_deferred_u_r7": True,
                "size_mtime_is_not_sha256_u_r6": True,
            },
        }


def default_suite_config_path() -> Path:
    """Return the tracked default YAML path when present."""

    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "subsystem_09"
        / "results_packaging_default.yaml"
    )


def load_results_packaging_config(path: Path | str | None = None) -> ResultsPackagingConfig:
    """Load packaging config from YAML or return validated defaults."""

    if path is None:
        candidate = default_suite_config_path()
        if not candidate.is_file():
            return ResultsPackagingConfig().validated()
        path = candidate
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"Results packaging config not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load results packaging config: {exc}") from exc
    if raw is None:
        return ResultsPackagingConfig().validated()
    if not isinstance(raw, Mapping):
        raise ConfigError("Results packaging config root must be a mapping.")
    return config_from_mapping(raw)


def config_from_mapping(data: Mapping[str, Any]) -> ResultsPackagingConfig:
    """Build a validated config from a mapping (YAML / dict)."""

    schema = str(data.get("schema_version", SETTINGS_SCHEMA_VERSION))
    fps = float(data.get("fps", TARGET_FPS))
    default_profile = str(data.get("default_profile_id", DEFAULT_PROFILE_ID))

    exec_raw = data.get("execution", {})
    if exec_raw is None:
        exec_raw = {}
    if not isinstance(exec_raw, Mapping):
        raise ConfigError("execution must be a mapping.")
    execution = ExecutionSettings(
        sync_construction_method=str(
            exec_raw.get(
                "sync_construction_method",
                SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC,
            )
        ),
        package_mode=str(
            exec_raw.get("package_mode", PACKAGE_MODE_REFERENCE_LIGHTWEIGHT)
        ),
        require_prepared_video=bool(exec_raw.get("require_prepared_video", True)),
        publish_run=bool(exec_raw.get("publish_run", False)),
        hash_prepared_video=bool(exec_raw.get("hash_prepared_video", False)),
        selected_export_profiles=tuple(
            exec_raw.get("selected_export_profiles", ()) or ()
        ),
    )

    review_raw = data.get("review_emit_v1_defaults", {})
    if review_raw is None:
        review_raw = {}
    if not isinstance(review_raw, Mapping):
        raise ConfigError("review_emit_v1_defaults must be a mapping.")
    review = ReviewEmitV1Defaults(
        include_contact=bool(
            review_raw.get("include_contact", REVIEW_EMIT_V1_INCLUDE_CONTACT)
        ),
        include_engaged=bool(
            review_raw.get("include_engaged", REVIEW_EMIT_V1_INCLUDE_ENGAGED)
        ),
        include_side_by_side=bool(
            review_raw.get(
                "include_side_by_side", REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE
            )
        ),
        min_emit_bout_frames=int(
            review_raw.get("min_emit_bout_frames", REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES)
        ),
        min_emit_bout_seconds=float(
            review_raw.get(
                "min_emit_bout_seconds", REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS
            )
        ),
        target_fps=float(review_raw.get("target_fps", REVIEW_EMIT_V1_TARGET_FPS)),
    )

    return ResultsPackagingConfig(
        schema_version=schema,
        fps=fps,
        default_profile_id=default_profile,
        execution=execution,
        review_emit_v1_defaults=review,
    ).validated()
