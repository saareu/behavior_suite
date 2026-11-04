"""Export-specific UI settings and preview helpers for Subsystem 09."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BOUT_INDEX_FILENAME,
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    EXPORT_PROFILES_DIRNAME,
    STALE_SOURCE,
)
from results_packaging.media_csv_export.config import (
    CsvExportSettings,
    MediaExportSettings,
    SelectionSpec,
)
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CSV_FORMAT_PROFILE,
    LEGACY_FRAMEWISE_PROFILE_ID,
    PACKAGE_BOUTS_AUTOMATED_PROFILE_ID,
    SELECTION_ALL_PACKAGE_BOUTS,
    SELECTION_CHANNELS_BEHAVIORS,
    SELECTION_EXPORT_PROFILE,
    ExportConfigError,
    ExportSelectionError,
)
from results_packaging.media_csv_export.selection import select_bouts_for_export

# Packages currently acceptable as export sources (MVP; archival deferred).
EXPORT_ELIGIBLE_CLASSIFICATIONS = frozenset(
    {ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS}
)


@dataclass(frozen=True, slots=True)
class ExportUiSettings:
    """GUI working settings snapshotted into each CSV/media export request."""

    selection_mode: str = SELECTION_ALL_PACKAGE_BOUTS
    channels_text: str = ""
    behaviors_text: str = ""
    profile_id: str | None = None
    format_profile: str = DEFAULT_CSV_FORMAT_PROFILE
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    # Explicit S4 when lineage missing (legacy_framewise_v1 only; never auto-newest).
    pose_run_dir: str | None = None
    # C3-5 opt-in overlay media (default remains naked / unannotated).
    overlay_enabled: bool = False
    overlay_include_behavior: bool = True
    overlay_include_pose: bool = False
    overlay_pose_source_kind: str | None = None  # s3_accepted | s4_finalized
    overlay_pose_run_dir: str | None = None
    overlay_hud_mode: str = "both"
    overlay_include_contact: bool = True
    overlay_include_sbs: bool = True
    overlay_draw_skeleton: bool = True

    def validated(self) -> ExportUiSettings:
        mode = str(self.selection_mode).strip() or SELECTION_ALL_PACKAGE_BOUTS
        channels = _split_csv_tokens(self.channels_text)
        behaviors = _split_csv_tokens(self.behaviors_text)
        profile = (
            str(self.profile_id).strip() if self.profile_id is not None else None
        ) or None
        format_profile = (
            str(self.format_profile or DEFAULT_CSV_FORMAT_PROFILE).strip()
            or DEFAULT_CSV_FORMAT_PROFILE
        )
        if format_profile not in {
            LEGACY_FRAMEWISE_PROFILE_ID,
            PACKAGE_BOUTS_AUTOMATED_PROFILE_ID,
        }:
            raise ExportConfigError(
                f"Unsupported CSV format_profile {format_profile!r}."
            )
        if float(self.pre_context_sec) < 0 or float(self.post_context_sec) < 0:
            raise ExportConfigError("Context padding seconds must be >= 0.")
        pose_dir = (
            str(Path(str(self.pose_run_dir).strip()).expanduser().resolve())
            if self.pose_run_dir is not None and str(self.pose_run_dir).strip()
            else None
        )
        if pose_dir is not None and format_profile != LEGACY_FRAMEWISE_PROFILE_ID:
            raise ExportConfigError(
                "pose_run_dir is only valid with format_profile="
                f"{LEGACY_FRAMEWISE_PROFILE_ID!r}."
            )
        overlay_enabled = bool(self.overlay_enabled)
        overlay_include_behavior = bool(self.overlay_include_behavior)
        overlay_include_pose = bool(self.overlay_include_pose)
        overlay_pose_kind = (
            None
            if self.overlay_pose_source_kind is None
            else str(self.overlay_pose_source_kind).strip().lower() or None
        )
        overlay_pose_dir = (
            str(Path(str(self.overlay_pose_run_dir).strip()).expanduser().resolve())
            if self.overlay_pose_run_dir is not None
            and str(self.overlay_pose_run_dir).strip()
            else None
        )
        overlay_hud = str(self.overlay_hud_mode or "both").strip().lower() or "both"
        if overlay_enabled:
            if not overlay_include_behavior and not overlay_include_pose:
                raise ExportConfigError(
                    "Overlay media requires behavioral labels and/or pose enabled."
                )
            if overlay_include_pose:
                if overlay_pose_kind not in {"s3_accepted", "s4_finalized"}:
                    raise ExportConfigError(
                        "Overlay pose requires explicit s3_accepted or s4_finalized."
                    )
                if overlay_pose_dir is None:
                    raise ExportConfigError(
                        "Overlay pose requires an explicitly selected pose run "
                        "(never auto-selects newest)."
                    )
            elif overlay_pose_dir is not None or overlay_pose_kind is not None:
                raise ExportConfigError(
                    "Overlay pose source fields require overlay_include_pose=True."
                )
            if overlay_hud not in {"both", "focal_0_only", "focal_1_only"}:
                raise ExportConfigError(
                    "overlay_hud_mode must be both, focal_0_only, or focal_1_only."
                )
        # Validate via SelectionSpec construction (bout selection unused for
        # legacy_framewise_v1 but still kept valid for media export).
        self.to_selection_spec()
        return replace(
            self,
            selection_mode=mode,
            channels_text=",".join(channels),
            behaviors_text=",".join(behaviors),
            profile_id=profile,
            format_profile=format_profile,
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
            pose_run_dir=pose_dir,
            overlay_enabled=overlay_enabled,
            overlay_include_behavior=overlay_include_behavior,
            overlay_include_pose=overlay_include_pose,
            overlay_pose_source_kind=overlay_pose_kind,
            overlay_pose_run_dir=overlay_pose_dir,
            overlay_hud_mode=overlay_hud,
            overlay_include_contact=bool(self.overlay_include_contact),
            overlay_include_sbs=bool(self.overlay_include_sbs),
            overlay_draw_skeleton=bool(self.overlay_draw_skeleton),
        )

    def to_selection_spec(self) -> SelectionSpec:
        mode = str(self.selection_mode).strip() or SELECTION_ALL_PACKAGE_BOUTS
        channels = _split_csv_tokens(self.channels_text)
        behaviors = _split_csv_tokens(self.behaviors_text)
        profile = (
            str(self.profile_id).strip() if self.profile_id is not None else None
        ) or None
        if mode == SELECTION_ALL_PACKAGE_BOUTS:
            return SelectionSpec(mode=SELECTION_ALL_PACKAGE_BOUTS).validated()
        if mode == SELECTION_CHANNELS_BEHAVIORS:
            return SelectionSpec(
                mode=SELECTION_CHANNELS_BEHAVIORS,
                channels=channels,
                behavior_names=behaviors,
            ).validated()
        if mode == SELECTION_EXPORT_PROFILE:
            return SelectionSpec(
                mode=SELECTION_EXPORT_PROFILE,
                profile_id=profile,
            ).validated()
        raise ExportConfigError(f"Unsupported export selection_mode {mode!r}.")

    def to_csv_settings(self) -> CsvExportSettings:
        return CsvExportSettings(
            selection=self.to_selection_spec(),
            format_profile=str(self.format_profile or DEFAULT_CSV_FORMAT_PROFILE),
            pose_run_dir=self.pose_run_dir,
        ).validated()

    def to_media_settings(self) -> MediaExportSettings:
        return MediaExportSettings(
            selection=self.to_selection_spec(),
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
        ).validated()

    def to_overlay_media_settings(self) -> Any:
        """Build C3-5 overlay media settings (opt-in sibling of naked media)."""

        from overlay.export_s9_media import OverlayMediaExportSettings

        cfg = self.validated()
        if not cfg.overlay_enabled:
            raise ExportConfigError(
                "to_overlay_media_settings requires overlay_enabled=True."
            )
        return OverlayMediaExportSettings(
            selection=cfg.to_selection_spec(),
            pre_context_sec=float(cfg.pre_context_sec),
            post_context_sec=float(cfg.post_context_sec),
            include_behavior=bool(cfg.overlay_include_behavior),
            include_pose=bool(cfg.overlay_include_pose),
            pose_source_kind=cfg.overlay_pose_source_kind,
            pose_run_dir=cfg.overlay_pose_run_dir,
            hud_mode=str(cfg.overlay_hud_mode),
            include_contact=bool(cfg.overlay_include_contact),
            include_sbs=bool(cfg.overlay_include_sbs),
            draw_skeleton=bool(cfg.overlay_draw_skeleton),
            include_frame_counter=True,
            hash_clips=True,
        ).validated()


@dataclass(frozen=True, slots=True)
class ExportRequestIdentity:
    """Identity used to reject stale asynchronous export results."""

    request_id: str
    session_root: Path
    package_run_id: str
    package_dir: Path
    job_kind: str  # csv_export | media_export | media_retry
    settings: ExportUiSettings
    retry_job_dir: Path | None = None

    @staticmethod
    def new(
        *,
        session_root: Path,
        package_run_id: str,
        package_dir: Path,
        job_kind: str,
        settings: ExportUiSettings,
        retry_job_dir: Path | None = None,
    ) -> ExportRequestIdentity:
        return ExportRequestIdentity(
            request_id=str(uuid4()),
            session_root=session_root.resolve(strict=False),
            package_run_id=package_run_id,
            package_dir=package_dir.resolve(strict=False),
            job_kind=job_kind,
            settings=settings.validated(),
            retry_job_dir=(
                retry_job_dir.resolve(strict=False) if retry_job_dir is not None else None
            ),
        )


def package_is_export_eligible(classification: str | None) -> bool:
    return classification in EXPORT_ELIGIBLE_CLASSIFICATIONS


def stale_export_block_reason(classification: str | None) -> str:
    label = classification or "unknown"
    if classification == STALE_SOURCE:
        return (
            f"Selected S9 package is currently stale ({label}). "
            "Normal exports require a currently eligible package; "
            "archival export from stale packages is deferred. "
            "Historical export jobs remain inspectable."
        )
    return (
        f"Selected S9 package is not currently eligible for export ({label})."
    )


def list_package_export_profiles(package_dir: Path | str) -> tuple[str, ...]:
    root = Path(package_dir) / EXPORT_PROFILES_DIRNAME
    if not root.is_dir():
        return ()
    profiles: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / BOUT_INDEX_FILENAME).is_file():
            profiles.append(child.name)
    return tuple(profiles)


def preview_selection_count(
    package_dir: Path | str,
    settings: ExportUiSettings,
) -> int:
    """Count selected bout rows for GUI preview (read-only)."""

    path = Path(package_dir) / BEHAVIOR_BOUTS_FILENAME
    if not path.is_file():
        raise ExportSelectionError(f"Missing behavior_bouts.parquet: {path}")
    bouts = pd.read_parquet(path)
    selected = select_bouts_for_export(
        bouts,
        settings.to_selection_spec(),
        package_dir=Path(package_dir),
    )
    return int(len(selected))


def export_provenance_summary(
    *,
    package_run_id: str,
    settings: ExportUiSettings,
    n_selected: int | None,
) -> str:
    parts = [
        "export_provenance=automated",
        f"source_package={package_run_id}",
        f"format_profile={settings.format_profile}",
        f"selection_mode={settings.selection_mode}",
    ]
    if settings.selection_mode == SELECTION_EXPORT_PROFILE and settings.profile_id:
        parts.append(f"profile_id={settings.profile_id}")
    if settings.selection_mode == SELECTION_CHANNELS_BEHAVIORS:
        if settings.channels_text:
            parts.append(f"channels={settings.channels_text}")
        if settings.behaviors_text:
            parts.append(f"behaviors={settings.behaviors_text}")
    if settings.pose_run_dir:
        parts.append(f"pose_run_dir={Path(settings.pose_run_dir).name}")
    if settings.overlay_enabled:
        parts.append("media_family=media_export_overlay")
        layers = []
        if settings.overlay_include_behavior:
            layers.append("s8_automated")
        if settings.overlay_include_pose:
            layers.append(str(settings.overlay_pose_source_kind or "pose"))
        parts.append(f"overlay_layers={'+'.join(layers) if layers else 'none'}")
        if settings.overlay_pose_run_dir:
            parts.append(
                f"overlay_pose_run={Path(settings.overlay_pose_run_dir).name}"
            )
    else:
        parts.append("media_family=media_export")
    if n_selected is not None:
        parts.append(f"n_selected={n_selected}")
    return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class LegacyPoseUiState:
    """GUI pose-source state for legacy_framewise_v1 (lineage default vs picker)."""

    format_is_legacy: bool
    lineage_available: bool
    lineage_run_id: str | None
    lineage_run_dir: str | None
    requires_explicit_pose: bool
    reason: str
    candidates: tuple[Any, ...]
    selected_pose_run_dir: str | None


def build_legacy_pose_ui_state(
    *,
    session_root: Path | str | None,
    package_dir: Path | str | None,
    format_profile: str,
    selected_pose_run_dir: str | None,
) -> LegacyPoseUiState:
    """Compute whether the S4 picker must be shown (never auto-select newest)."""

    from results_packaging.legacy_framewise_csv import (
        LEGACY_FRAMEWISE_PROFILE_ID,
        inspect_package_pose_lineage,
        list_pose_run_candidates,
    )

    profile = str(format_profile or DEFAULT_CSV_FORMAT_PROFILE).strip()
    is_legacy = profile == LEGACY_FRAMEWISE_PROFILE_ID
    if not is_legacy:
        return LegacyPoseUiState(
            format_is_legacy=False,
            lineage_available=False,
            lineage_run_id=None,
            lineage_run_dir=None,
            requires_explicit_pose=False,
            reason="Historical bout CSV does not use S4 pose selection.",
            candidates=(),
            selected_pose_run_dir=None,
        )
    if package_dir is None:
        return LegacyPoseUiState(
            format_is_legacy=True,
            lineage_available=False,
            lineage_run_id=None,
            lineage_run_dir=None,
            requires_explicit_pose=False,
            reason="Select an eligible S9 package first.",
            candidates=(),
            selected_pose_run_dir=None,
        )
    lineage = inspect_package_pose_lineage(package_dir)
    candidates = (
        list_pose_run_candidates(session_root) if session_root is not None else ()
    )
    selected = (
        str(Path(str(selected_pose_run_dir).strip()).expanduser().resolve())
        if selected_pose_run_dir and str(selected_pose_run_dir).strip()
        else None
    )
    if lineage.available:
        return LegacyPoseUiState(
            format_is_legacy=True,
            lineage_available=True,
            lineage_run_id=lineage.run_id,
            lineage_run_dir=str(lineage.run_dir) if lineage.run_dir else None,
            requires_explicit_pose=False,
            reason=lineage.reason,
            candidates=candidates,
            selected_pose_run_dir=None,
        )
    return LegacyPoseUiState(
        format_is_legacy=True,
        lineage_available=False,
        lineage_run_id=None,
        lineage_run_dir=None,
        requires_explicit_pose=True,
        reason=lineage.reason,
        candidates=candidates,
        selected_pose_run_dir=selected,
    )


def _split_csv_tokens(text: str) -> tuple[str, ...]:
    if not text or not str(text).strip():
        return ()
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def export_job_table_rows(jobs: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for job in jobs:
        counts = "—"
        if getattr(job, "job_kind", None) == "csv_export":
            counts = f"rows={job.n_rows}" if job.n_rows is not None else "—"
        else:
            parts = []
            if job.n_selected is not None:
                parts.append(f"sel={job.n_selected}")
            if job.n_succeeded is not None:
                parts.append(f"ok={job.n_succeeded}")
            if job.n_failed is not None:
                parts.append(f"fail={job.n_failed}")
            counts = ", ".join(parts) if parts else "—"
        rows.append(
            (
                str(job.export_job_id),
                str(job.source_package_run_id or "—"),
                str(job.job_kind or "—"),
                str(job.status or "—"),
                str(job.classification),
                counts,
                str(job.error_message or "—"),
                str(job.job_dir),
            )
        )
    return tuple(rows)
