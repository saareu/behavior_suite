"""Settings for legacy framewise / format-profile CSV exports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from results_packaging.legacy_framewise_csv.contracts import (
    CSV_FORMAT_PROFILES,
    DEFAULT_CSV_FORMAT_PROFILE,
    FRAMEWISE_JOB_KIND,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LegacyFramewiseError,
)
from results_packaging.media_csv_export.config import (
    CsvExportSettings,
    selection_from_mapping,
)
from results_packaging.media_csv_export.contracts import (
    EXPORT_SETTINGS_SCHEMA_VERSION,
    JOB_KIND_CSV,
    ExportConfigError,
)


@dataclass(frozen=True, slots=True)
class LegacyFramewiseExportSettings:
    """Immutable settings for a legacy_framewise_v1 export job."""

    schema_version: str = EXPORT_SETTINGS_SCHEMA_VERSION
    job_kind: str = FRAMEWISE_JOB_KIND
    format_profile: str = LEGACY_FRAMEWISE_PROFILE_ID
    behavior_source_kind: str = "automated_s8"

    def validated(self) -> LegacyFramewiseExportSettings:
        if self.schema_version != EXPORT_SETTINGS_SCHEMA_VERSION:
            raise LegacyFramewiseError(
                f"Unsupported export settings schema_version {self.schema_version!r}."
            )
        if self.job_kind != JOB_KIND_CSV:
            raise LegacyFramewiseError(f"job_kind must be {JOB_KIND_CSV!r}.")
        profile = str(self.format_profile).strip()
        if profile != LEGACY_FRAMEWISE_PROFILE_ID:
            raise LegacyFramewiseError(
                f"LegacyFramewiseExportSettings only accepts "
                f"{LEGACY_FRAMEWISE_PROFILE_ID!r}; got {profile!r}."
            )
        kind = str(self.behavior_source_kind).strip()
        if kind not in {"automated_s8", "published_s10"}:
            raise LegacyFramewiseError(
                f"Unsupported behavior_source_kind {kind!r}."
            )
        return replace(self, format_profile=profile, behavior_source_kind=kind)

    def to_settings_dict(self) -> dict[str, Any]:
        cfg = self.validated()
        return {
            "schema_version": cfg.schema_version,
            "job_kind": cfg.job_kind,
            "format_profile": cfg.format_profile,
            "behavior_source_kind": cfg.behavior_source_kind,
            "notes": {
                "never_mutates_upstream_science": True,
                "original_frame_authority": "sync_index.raw_decode_frame_idx",
                "pose_authority": "s4_final_pose",
                "coordinate_space": "prepared_video_pixels",
                "coordinate_format": "round_to_integer_one_decimal",
                "crlf": True,
            },
        }


def normalize_csv_format_profile(value: str | None) -> str:
    profile = str(value or DEFAULT_CSV_FORMAT_PROFILE).strip()
    if profile not in CSV_FORMAT_PROFILES:
        raise ExportConfigError(
            f"Unsupported CSV format_profile {profile!r}; expected one of "
            f"{sorted(CSV_FORMAT_PROFILES)}."
        )
    return profile


def csv_settings_with_format_profile(
    data: Mapping[str, Any] | None = None,
    *,
    default_format_profile: str = DEFAULT_CSV_FORMAT_PROFILE,
) -> CsvExportSettings:
    """Build CsvExportSettings including format_profile (default legacy)."""

    raw = dict(data or {})
    format_profile = normalize_csv_format_profile(
        raw.get("format_profile", default_format_profile)
    )
    settings = CsvExportSettings(
        schema_version=str(
            raw.get("schema_version", EXPORT_SETTINGS_SCHEMA_VERSION)
        ),
        job_kind=str(raw.get("job_kind", JOB_KIND_CSV)),
        selection=selection_from_mapping(raw.get("selection")),
        format_profile=format_profile,
    )
    return settings.validated()
