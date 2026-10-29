"""Settings snapshots for media and CSV export jobs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
    EXPORT_SETTINGS_SCHEMA_VERSION,
    JOB_KIND_CSV,
    JOB_KIND_MEDIA,
    SELECTION_ALL_PACKAGE_BOUTS,
    SELECTION_BOUT_ID_LIST,
    SELECTION_CHANNELS_BEHAVIORS,
    SELECTION_EXPORT_PROFILE,
    SELECTION_MODES,
    SOURCE_VIDEO_KIND_PREPARED,
    ExportConfigError,
)


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    """Exactly one bout-selection mode with parameters."""

    mode: str = SELECTION_ALL_PACKAGE_BOUTS
    channels: tuple[str, ...] = ()
    behavior_names: tuple[str, ...] = ()
    profile_id: str | None = None
    bout_ids: tuple[str, ...] = ()

    def validated(self) -> SelectionSpec:
        mode = str(self.mode).strip()
        if mode not in SELECTION_MODES:
            raise ExportConfigError(
                f"Unsupported selection_mode {mode!r}; expected one of "
                f"{sorted(SELECTION_MODES)}."
            )
        channels = tuple(str(c).strip() for c in self.channels if str(c).strip())
        behaviors = tuple(
            str(b).strip() for b in self.behavior_names if str(b).strip()
        )
        profile_id = (
            str(self.profile_id).strip() if self.profile_id is not None else None
        ) or None
        bout_ids = tuple(str(b).strip() for b in self.bout_ids if str(b).strip())

        if mode == SELECTION_CHANNELS_BEHAVIORS and not channels and not behaviors:
            raise ExportConfigError(
                "channels_behaviors selection requires channels and/or behavior_names."
            )
        if mode == SELECTION_EXPORT_PROFILE and not profile_id:
            raise ExportConfigError("export_profile selection requires profile_id.")
        if mode == SELECTION_BOUT_ID_LIST and not bout_ids:
            raise ExportConfigError("bout_id_list selection requires bout_ids.")
        if mode != SELECTION_EXPORT_PROFILE and profile_id is not None:
            raise ExportConfigError(
                "profile_id is only valid with selection_mode=export_profile."
            )
        if mode != SELECTION_BOUT_ID_LIST and bout_ids:
            raise ExportConfigError(
                "bout_ids are only valid with selection_mode=bout_id_list."
            )
        if mode != SELECTION_CHANNELS_BEHAVIORS and (channels or behaviors):
            raise ExportConfigError(
                "channels/behavior_names require selection_mode=channels_behaviors."
            )
        return replace(
            self,
            mode=mode,
            channels=channels,
            behavior_names=behaviors,
            profile_id=profile_id,
            bout_ids=bout_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        spec = self.validated()
        return {
            "mode": spec.mode,
            "channels": list(spec.channels),
            "behavior_names": list(spec.behavior_names),
            "profile_id": spec.profile_id,
            "bout_ids": list(spec.bout_ids),
        }


@dataclass(frozen=True, slots=True)
class MediaExportSettings:
    """Immutable media-export settings snapshot."""

    schema_version: str = EXPORT_SETTINGS_SCHEMA_VERSION
    job_kind: str = JOB_KIND_MEDIA
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    crf: int = DEFAULT_CRF
    codec: str = DEFAULT_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    source_video_kind: str = SOURCE_VIDEO_KIND_PREPARED
    fail_fast: bool = False
    accept_partial: bool = False
    hash_clips: bool = False

    def validated(self) -> MediaExportSettings:
        if self.schema_version != EXPORT_SETTINGS_SCHEMA_VERSION:
            raise ExportConfigError(
                f"Unsupported export settings schema_version {self.schema_version!r}."
            )
        if self.job_kind != JOB_KIND_MEDIA:
            raise ExportConfigError(f"Media settings job_kind must be {JOB_KIND_MEDIA!r}.")
        if float(self.pre_context_sec) < 0 or float(self.post_context_sec) < 0:
            raise ExportConfigError("Context padding seconds must be >= 0.")
        if int(self.crf) < 0 or int(self.crf) > 51:
            raise ExportConfigError(f"crf out of range: {self.crf}")
        if self.source_video_kind != SOURCE_VIDEO_KIND_PREPARED:
            raise ExportConfigError(
                "MVP media export supports only source_video_kind="
                f"{SOURCE_VIDEO_KIND_PREPARED!r}."
            )
        if str(self.codec).strip() != DEFAULT_CODEC:
            raise ExportConfigError(f"MVP codec must be {DEFAULT_CODEC!r}.")
        if str(self.pixel_format).strip() != DEFAULT_PIXEL_FORMAT:
            raise ExportConfigError(
                f"MVP pixel_format must be {DEFAULT_PIXEL_FORMAT!r}."
            )
        return replace(
            self,
            selection=self.selection.validated(),
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
            crf=int(self.crf),
            codec=str(self.codec).strip(),
            pixel_format=str(self.pixel_format).strip(),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        cfg = self.validated()
        return {
            "schema_version": cfg.schema_version,
            "job_kind": cfg.job_kind,
            "selection": cfg.selection.to_dict(),
            "pre_context_sec": cfg.pre_context_sec,
            "post_context_sec": cfg.post_context_sec,
            "crf": cfg.crf,
            "codec": cfg.codec,
            "pixel_format": cfg.pixel_format,
            "source_video_kind": cfg.source_video_kind,
            "fail_fast": cfg.fail_fast,
            "accept_partial": cfg.accept_partial,
            "hash_clips": cfg.hash_clips,
            "notes": {
                "behavioral_fps": 120.0,
                "clip_padding_uses_source_video_fps": True,
                "playback_uses_source_video_fps": True,
                "never_mutates_s9_package": True,
                "retry_model": "same_job_resume_for_incomplete_or_failed",
            },
        }


@dataclass(frozen=True, slots=True)
class CsvExportSettings:
    """Immutable CSV-export settings snapshot."""

    schema_version: str = EXPORT_SETTINGS_SCHEMA_VERSION
    job_kind: str = JOB_KIND_CSV
    selection: SelectionSpec = field(default_factory=SelectionSpec)

    def validated(self) -> CsvExportSettings:
        if self.schema_version != EXPORT_SETTINGS_SCHEMA_VERSION:
            raise ExportConfigError(
                f"Unsupported export settings schema_version {self.schema_version!r}."
            )
        if self.job_kind != JOB_KIND_CSV:
            raise ExportConfigError(f"CSV settings job_kind must be {JOB_KIND_CSV!r}.")
        return replace(self, selection=self.selection.validated())

    def to_settings_dict(self) -> dict[str, Any]:
        cfg = self.validated()
        return {
            "schema_version": cfg.schema_version,
            "job_kind": cfg.job_kind,
            "selection": cfg.selection.to_dict(),
            "notes": {
                "export_provenance": "automated",
                "no_bout_transformations": True,
                "framewise_csv_deferred": True,
                "reviewed_csv_deferred_to_s10": True,
                "never_mutates_s9_package": True,
            },
        }


def selection_from_mapping(data: Mapping[str, Any] | None) -> SelectionSpec:
    if data is None:
        return SelectionSpec().validated()
    if not isinstance(data, Mapping):
        raise ExportConfigError("selection must be a mapping.")
    return SelectionSpec(
        mode=str(data.get("mode", SELECTION_ALL_PACKAGE_BOUTS)),
        channels=tuple(data.get("channels", ()) or ()),
        behavior_names=tuple(data.get("behavior_names", ()) or ()),
        profile_id=data.get("profile_id"),
        bout_ids=tuple(data.get("bout_ids", ()) or ()),
    ).validated()


def media_settings_from_mapping(data: Mapping[str, Any]) -> MediaExportSettings:
    if not isinstance(data, Mapping):
        raise ExportConfigError("Media export settings root must be a mapping.")
    return MediaExportSettings(
        schema_version=str(
            data.get("schema_version", EXPORT_SETTINGS_SCHEMA_VERSION)
        ),
        job_kind=str(data.get("job_kind", JOB_KIND_MEDIA)),
        selection=selection_from_mapping(data.get("selection")),
        pre_context_sec=float(data.get("pre_context_sec", 0.0)),
        post_context_sec=float(data.get("post_context_sec", 0.0)),
        crf=int(data.get("crf", DEFAULT_CRF)),
        codec=str(data.get("codec", DEFAULT_CODEC)),
        pixel_format=str(data.get("pixel_format", DEFAULT_PIXEL_FORMAT)),
        source_video_kind=str(
            data.get("source_video_kind", SOURCE_VIDEO_KIND_PREPARED)
        ),
        fail_fast=bool(data.get("fail_fast", False)),
        accept_partial=bool(data.get("accept_partial", False)),
        hash_clips=bool(data.get("hash_clips", False)),
    ).validated()


def csv_settings_from_mapping(data: Mapping[str, Any]) -> CsvExportSettings:
    if not isinstance(data, Mapping):
        raise ExportConfigError("CSV export settings root must be a mapping.")
    return CsvExportSettings(
        schema_version=str(
            data.get("schema_version", EXPORT_SETTINGS_SCHEMA_VERSION)
        ),
        job_kind=str(data.get("job_kind", JOB_KIND_CSV)),
        selection=selection_from_mapping(data.get("selection")),
    ).validated()


def coerce_str_sequence(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(v).strip() for v in value if str(v).strip())
