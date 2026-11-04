"""C3-6 opt-in S10 reviewed overlay media export (sibling of naked/PI media).

Publishes annotated clips under ``reviewed_media_export_overlay__{timestamp}/``.
Requires an explicit published S10 review. Never falls back to S8 automated
labels. Pose overlays accept compatible S4 finalized only (S3+S10 unsupported).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from overlay.adapters.base import OverlayStageAdapter, SourceIdentity
from overlay.contracts import (
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_S10_MEDIA_RENDERER_VERSION,
    REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND,
    REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_PREFIX,
    REVIEWED_MEDIA_EXPORT_OVERLAY_MANIFEST_FILENAME,
    REVIEWED_MEDIA_EXPORT_OVERLAY_SETTINGS_SCHEMA,
    REVIEWED_MEDIA_EXPORT_OVERLAY_STAGING_DIRNAME,
    BehaviorSourceKind,
    FrameDomain,
    HudDirectionMode,
    LayerId,
    OverlayContractError,
    OverlayExportCancelled,
    OverlayExportError,
    OverlaySourceError,
    PlateauPolicy,
    PoseSourceKind,
    default_behavioral_qc_layers,
    default_mixed_pose_behavior_qc_layers,
    default_s4_pose_qc_layers,
)
from overlay.export import build_behavioral_adapters, build_s4_pose_adapters
from overlay.export_mixed import build_mixed_adapters
from overlay.plan import (
    MixedPoseBehaviorQcExportPlan,
    PreparedFrameInterval,
    dense_sample_window_for_prepared_interval,
    plan_bout_span,
    plan_s4_prepared_interval,
)
from overlay.provenance import (
    OverlayArtifactManifest,
    PreparedVideoIdentity,
    build_overlay_artifact_manifest,
)
from overlay.source import (
    OverlayQcSource,
    resolve_overlay_qc_source,
    verify_overlay_qc_source_integrity,
)
from overlay.source_mixed import (
    OverlayMixedQcSource,
    resolve_overlay_mixed_qc_source,
    verify_overlay_mixed_qc_source_integrity,
)
from overlay.source_s4 import (
    OverlayS4QcSource,
    resolve_overlay_s4_qc_source,
    verify_overlay_s4_qc_source_integrity,
)
from overlay.stream_clip import (
    CancellationCallback,
    ProgressCallback,
    StreamComposedClipRequest,
    stream_composed_prepared_clip,
)
from overlay.style import DrawConfig
from overlay.timing import PreparedFrameTiming, resolve_prepared_frame_timing
from preprocess.models import OperationProgress
from project.file_digest import sha256_file
from results_packaging.contracts import (
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from results_packaging.export import atomic_parquet_write, write_json, write_yaml
from results_packaging.media_csv_export.clip_index import (
    build_pending_clip_index,
    clip_bounds_from_index_row,
)
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import (
    CLIP_INDEX_FILENAME,
    CLIPS_DIRNAME,
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
    ENCODE_STATUS_FAILED,
    ENCODE_STATUS_OMITTED_INVALID_BOUNDS,
    ENCODE_STATUS_PENDING,
    ENCODE_STATUS_SKIPPED_EXISTS,
    ENCODE_STATUS_SUCCEEDED,
    FAILURES_JSONL_FILENAME,
    RESULTS_EXPORTS_DIRNAME,
    STATUS_INCOMPLETE,
    ExportConfigError,
    ExportPublicationError,
    ExportSourceError,
)
from results_packaging.media_csv_export.results import MediaExportJobResult
from results_packaging.media_csv_export.selection import select_bouts_for_export

OpProgressCallback = Callable[[OperationProgress], None]

_PROTECTED_SESSION_SUBDIRS = (
    "preprocess",
    "pose_inference",
    "tracking_correction",
    "pose_finalization",
    "blob_features",
    "combined_features",
    "behavioral_inference",
    "behavioral_arbitration",
    "results_package",
    "behavioral_review",
)


@dataclass(frozen=True, slots=True)
class ReviewedOverlayMediaExportSettings:
    """Opt-in reviewed overlay media settings (S10 sibling export family)."""

    schema_version: str = REVIEWED_MEDIA_EXPORT_OVERLAY_SETTINGS_SCHEMA
    job_kind: str = REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    include_behavior: bool = True
    include_pose: bool = False
    pose_source_kind: str | None = None
    pose_run_dir: str | None = None
    hud_mode: str = HudDirectionMode.BOTH.value
    include_contact: bool = True
    include_sbs: bool = True
    draw_skeleton: bool = True
    include_frame_counter: bool = True
    crf: int = DEFAULT_CRF
    codec: str = DEFAULT_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    fail_fast: bool = False
    accept_partial: bool = False
    hash_clips: bool = True

    def validated(self) -> ReviewedOverlayMediaExportSettings:
        if self.schema_version != REVIEWED_MEDIA_EXPORT_OVERLAY_SETTINGS_SCHEMA:
            raise ExportConfigError(
                f"Unsupported reviewed overlay media settings schema "
                f"{self.schema_version!r}."
            )
        if self.job_kind != REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND:
            raise ExportConfigError(
                "Reviewed overlay media job_kind must be "
                f"{REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND!r}."
            )
        if float(self.pre_context_sec) < 0 or float(self.post_context_sec) < 0:
            raise ExportConfigError("Context padding seconds must be >= 0.")
        include_behavior = bool(self.include_behavior)
        include_pose = bool(self.include_pose)
        if not include_behavior and not include_pose:
            raise ExportConfigError(
                "Reviewed overlay media requires include_behavior and/or include_pose."
            )
        pose_kind: str | None = None
        pose_dir: str | None = None
        if include_pose:
            raw_kind = (
                None
                if self.pose_source_kind is None
                else str(self.pose_source_kind).strip().lower()
            )
            if raw_kind != PoseSourceKind.S4_FINALIZED.value:
                raise ExportConfigError(
                    "Reviewed overlay media pose overlays require "
                    f"pose_source_kind={PoseSourceKind.S4_FINALIZED.value!r} "
                    "(S3+S10 is unsupported)."
                )
            pose_kind = raw_kind
            if self.pose_run_dir is None or not str(self.pose_run_dir).strip():
                raise ExportConfigError(
                    "include_pose requires an explicit pose_run_dir "
                    "(never auto-selects newest)."
                )
            pose_dir = str(Path(str(self.pose_run_dir).strip()).expanduser().resolve())
        elif self.pose_run_dir is not None and str(self.pose_run_dir).strip():
            raise ExportConfigError(
                "pose_run_dir is only valid when include_pose=True."
            )
        elif self.pose_source_kind is not None and str(self.pose_source_kind).strip():
            raise ExportConfigError(
                "pose_source_kind is only valid when include_pose=True."
            )
        try:
            HudDirectionMode(str(self.hud_mode).strip().lower())
        except ValueError as exc:
            raise ExportConfigError(
                "hud_mode must be focal_0_only, focal_1_only, or both."
            ) from exc
        if self.codec != DEFAULT_CODEC:
            raise ExportConfigError(f"Only codec={DEFAULT_CODEC!r} is supported.")
        if self.pixel_format != DEFAULT_PIXEL_FORMAT:
            raise ExportConfigError(
                f"Only pixel_format={DEFAULT_PIXEL_FORMAT!r} is supported."
            )
        if int(self.crf) < 0 or int(self.crf) > 51:
            raise ExportConfigError("crf must be in [0, 51].")
        return replace(
            self,
            selection=self.selection.validated(),
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
            include_behavior=include_behavior,
            include_pose=include_pose,
            pose_source_kind=pose_kind,
            pose_run_dir=pose_dir,
            hud_mode=str(self.hud_mode).strip().lower(),
            include_contact=bool(self.include_contact),
            include_sbs=bool(self.include_sbs),
            draw_skeleton=bool(self.draw_skeleton),
            include_frame_counter=bool(self.include_frame_counter),
            crf=int(self.crf),
            codec=str(self.codec),
            pixel_format=str(self.pixel_format),
            fail_fast=bool(self.fail_fast),
            accept_partial=bool(self.accept_partial),
            hash_clips=bool(self.hash_clips),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        cfg = self.validated()
        return {
            "schema_version": cfg.schema_version,
            "job_kind": cfg.job_kind,
            "selection": cfg.selection.to_dict(),
            "pre_context_sec": cfg.pre_context_sec,
            "post_context_sec": cfg.post_context_sec,
            "include_behavior": cfg.include_behavior,
            "include_pose": cfg.include_pose,
            "pose_source_kind": cfg.pose_source_kind,
            "pose_run_dir": cfg.pose_run_dir,
            "hud_mode": cfg.hud_mode,
            "include_contact": cfg.include_contact,
            "include_sbs": cfg.include_sbs,
            "draw_skeleton": cfg.draw_skeleton,
            "include_frame_counter": cfg.include_frame_counter,
            "crf": cfg.crf,
            "codec": cfg.codec,
            "pixel_format": cfg.pixel_format,
            "fail_fast": cfg.fail_fast,
            "accept_partial": cfg.accept_partial,
            "hash_clips": cfg.hash_clips,
            "renderer_version": OVERLAY_S10_MEDIA_RENDERER_VERSION,
            "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
            "s8_label_fallback": False,
        }

    def configuration_label(self) -> str:
        cfg = self.validated()
        if cfg.include_behavior and cfg.include_pose:
            return "s4_finalized+s10_reviewed"
        if cfg.include_behavior:
            return "s10_reviewed"
        return PoseSourceKind.S4_FINALIZED.value


@dataclass(frozen=True, slots=True)
class ReviewedOverlayMediaExportRequest:
    session_root: Path
    review_run_dir: Path
    settings: ReviewedOverlayMediaExportSettings | None = None
    selection: SelectionSpec | None = None
    timestamp: str | None = None
    export_job_id: str | None = None
    output_root: Path | None = None
    cancellation_requested: CancellationCallback | None = None


@dataclass(frozen=True, slots=True)
class ReviewedOverlayMediaResolvedSources:
    """Pinned published-S10 sources for one reviewed overlay media job."""

    review_run_id: str
    review_run_dir: Path
    package_run_id: str
    package_dir: Path
    mode: str  # behavior | pose_s4 | mixed_s4
    behavior: OverlayQcSource | None
    pose_s4: OverlayS4QcSource | None
    mixed: OverlayMixedQcSource | None
    reviewed_bouts: pd.DataFrame
    review_bout_status: pd.DataFrame | None
    prepared_video_path: Path
    prepared_video_sha256: str
    video_fps: float
    video_width: int
    video_height: int
    video_n_frames: int
    sync_authority: str
    source_digests: dict[str, str | None]
    compatibility_notes: tuple[str, ...]
    selected_layers: tuple[LayerId, ...]
    style: DrawConfig
    provenance: dict[str, Any]


def resolve_reviewed_overlay_media_sources(
    *,
    review_run_dir: Path | str,
    settings: ReviewedOverlayMediaExportSettings,
) -> ReviewedOverlayMediaResolvedSources:
    """Resolve explicit published S10 (+ optional S4) sources for overlay media."""

    cfg = settings.validated()
    hud = HudDirectionMode(cfg.hud_mode)
    behavior: OverlayQcSource | None = None
    pose_s4: OverlayS4QcSource | None = None
    mixed: OverlayMixedQcSource | None = None
    notes: list[str] = []
    provenance: dict[str, Any] = {
        "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
        "s8_label_fallback": False,
        "include_behavior": cfg.include_behavior,
        "include_pose": cfg.include_pose,
    }

    if cfg.include_behavior and cfg.include_pose:
        assert cfg.pose_run_dir is not None
        mixed = resolve_overlay_mixed_qc_source(
            pose_source_kind=PoseSourceKind.S4_FINALIZED,
            behavior_source_kind=BehaviorSourceKind.S10_REVIEWED,
            s4_run_dir=cfg.pose_run_dir,
            review_run_dir=review_run_dir,
        )
        if mixed.behavior_source_kind is not BehaviorSourceKind.S10_REVIEWED:
            raise OverlaySourceError(
                "C3-6 reviewed overlay media refuses S8 automated label fallback."
            )
        if mixed.pose_source_kind is not PoseSourceKind.S4_FINALIZED:
            raise OverlaySourceError(
                "C3-6 reviewed overlay media refuses non-S4 pose sources."
            )
        behavior = mixed.behavior
        pose_s4 = mixed.pose_s4
        mode = "mixed_s4"
        layers = default_mixed_pose_behavior_qc_layers(
            pose_source_kind=PoseSourceKind.S4_FINALIZED,
            hud_mode=hud,
            include_contact=cfg.include_contact,
            include_sbs=cfg.include_sbs,
            include_frame_counter=cfg.include_frame_counter,
        )
        style = DrawConfig(
            darken_alpha=0.92,
            draw_skeleton=bool(cfg.draw_skeleton),
            legend_text=True,
            prepared_video_fps=float(mixed.video_probe.fps),
            hud_direction_mode=hud.value,
            contact_led_enabled=bool(cfg.include_contact),
            sbs_led_enabled=bool(cfg.include_sbs),
            extra={
                "skeleton_edges": [list(e) for e in mixed.skeleton_edges],
                "show_machine_corrections": False,
                "show_manual_corrections": False,
            },
        )
        video_path = mixed.prepared_video_path
        video_sha = mixed.prepared_video_sha256
        probe = mixed.video_probe
        sync_authority = mixed.sync_authority
        digests = dict(mixed.source_digests)
        notes.extend(mixed.compatibility_notes)
        provenance.update(dict(mixed.provenance))
    elif cfg.include_behavior:
        behavior = resolve_overlay_qc_source(
            source_kind=BehaviorSourceKind.S10_REVIEWED,
            review_run_dir=review_run_dir,
        )
        if behavior.source_kind is not BehaviorSourceKind.S10_REVIEWED:
            raise OverlaySourceError(
                "C3-6 reviewed overlay media refuses S8 automated label fallback."
            )
        mode = "behavior"
        layers = default_behavioral_qc_layers(
            hud_mode=hud,
            include_contact=cfg.include_contact,
            include_sbs=cfg.include_sbs,
            include_frame_counter=cfg.include_frame_counter,
        )
        style = DrawConfig(
            darken_alpha=0.92,
            legend_text=True,
            prepared_video_fps=float(behavior.video_probe.fps),
            hud_direction_mode=hud.value,
            contact_led_enabled=bool(cfg.include_contact),
            sbs_led_enabled=bool(cfg.include_sbs),
        )
        video_path = Path(behavior.video_path)
        video_sha = str(behavior.video_sha256 or "")
        probe = behavior.video_probe
        sync_authority = "behavior_package_sync_index_last_in_plateau"
        digests = dict(behavior.source_digests)
        provenance.update(dict(behavior.provenance))
    else:
        assert cfg.pose_run_dir is not None
        # Pose-only still requires published review for authoritative bout spans.
        behavior = resolve_overlay_qc_source(
            source_kind=BehaviorSourceKind.S10_REVIEWED,
            review_run_dir=review_run_dir,
        )
        pose_s4 = resolve_overlay_s4_qc_source(s4_run_dir=cfg.pose_run_dir)
        _assert_video_matches(
            review_video=Path(behavior.video_path),
            review_sha=str(behavior.video_sha256 or ""),
            other_video=pose_s4.prepared_video_path,
            other_sha=pose_s4.prepared_video_sha256,
        )
        if behavior.session_root is not None and pose_s4.session_root.resolve() != (
            Path(behavior.session_root).resolve()
        ):
            raise OverlaySourceError(
                "S4 pose session_root does not match published review session "
                "(refusing cross-session source mixing)."
            )
        mode = "pose_s4"
        layers = default_s4_pose_qc_layers(
            include_frame_counter=cfg.include_frame_counter
        )
        style = DrawConfig(
            darken_alpha=0.90,
            draw_skeleton=bool(cfg.draw_skeleton),
            node_radius=5,
            edge_thickness=2,
            legend_text=True,
            prepared_video_fps=float(pose_s4.video_probe.fps),
            extra={"skeleton_edges": [list(e) for e in pose_s4.skeleton_edges]},
        )
        video_path = pose_s4.prepared_video_path
        video_sha = pose_s4.prepared_video_sha256
        probe = pose_s4.video_probe
        sync_authority = "pose_syncmap_last_in_plateau"
        digests = {
            **dict(behavior.source_digests),
            **{f"pose_{k}": v for k, v in pose_s4.source_digests.items()},
        }
        notes.append("pose_only_uses_published_reviewed_bout_spans")
        provenance.update(
            {
                **dict(behavior.provenance),
                "pose_source_kind": PoseSourceKind.S4_FINALIZED.value,
                "pose_run_id": pose_s4.s4_run_id,
                "pose_run_dir": str(pose_s4.s4_run_dir),
            }
        )

    assert behavior is not None
    if behavior.behavior_bouts is None or len(behavior.behavior_bouts) == 0:
        raise ExportSourceError(
            "Published review has no reviewed_behavior_bouts for overlay media."
        )
    if behavior.review_run_id is None or behavior.review_run_dir is None:
        raise OverlaySourceError(
            "Reviewed overlay media requires pinned review_run_id/dir."
        )
    if not video_sha:
        raise OverlaySourceError("Prepared video digest missing for reviewed overlay.")

    status_table = _load_review_bout_status(behavior.review_run_dir)
    return ReviewedOverlayMediaResolvedSources(
        review_run_id=str(behavior.review_run_id),
        review_run_dir=Path(behavior.review_run_dir).resolve(),
        package_run_id=behavior.package_run_id,
        package_dir=Path(behavior.package_dir).resolve(),
        mode=mode,
        behavior=behavior if cfg.include_behavior or mode == "pose_s4" else None,
        pose_s4=pose_s4,
        mixed=mixed,
        reviewed_bouts=behavior.behavior_bouts.copy(),
        review_bout_status=status_table,
        prepared_video_path=Path(video_path).resolve(),
        prepared_video_sha256=str(video_sha),
        video_fps=float(probe.fps),
        video_width=int(probe.width),
        video_height=int(probe.height),
        video_n_frames=int(probe.n_frames),
        sync_authority=sync_authority,
        source_digests=digests,
        compatibility_notes=tuple(notes),
        selected_layers=layers,
        style=style,
        provenance=provenance,
    )


def verify_reviewed_overlay_media_sources_integrity(
    sources: ReviewedOverlayMediaResolvedSources,
) -> None:
    """Revalidate pinned published review / pose integrity before publication."""

    if sources.mixed is not None:
        verify_overlay_mixed_qc_source_integrity(sources.mixed)
        return
    if sources.behavior is not None:
        verify_overlay_qc_source_integrity(sources.behavior)
    if sources.pose_s4 is not None:
        verify_overlay_s4_qc_source_integrity(sources.pose_s4)


def run_reviewed_overlay_media_export(
    request: ReviewedOverlayMediaExportRequest,
    progress_callback: OpProgressCallback | None = None,
    *,
    overlay_progress_callback: ProgressCallback | None = None,
) -> MediaExportJobResult:
    """Validate published review/sources and publish reviewed overlay media."""

    settings = _resolve_settings(request)
    sources = resolve_reviewed_overlay_media_sources(
        review_run_dir=request.review_run_dir,
        settings=settings,
    )
    session_root = Path(request.session_root).expanduser().resolve()
    return publish_reviewed_overlay_media_export(
        session_root=session_root,
        sources=sources,
        settings=settings,
        export_job_id=request.export_job_id,
        timestamp=request.timestamp,
        output_root=request.output_root,
        progress_callback=progress_callback,
        overlay_progress_callback=overlay_progress_callback,
        cancellation_requested=request.cancellation_requested,
    )


def publish_reviewed_overlay_media_export(
    *,
    session_root: Path,
    sources: ReviewedOverlayMediaResolvedSources,
    settings: ReviewedOverlayMediaExportSettings,
    export_job_id: str | None = None,
    timestamp: str | None = None,
    output_root: Path | None = None,
    progress_callback: OpProgressCallback | None = None,
    overlay_progress_callback: ProgressCallback | None = None,
    cancellation_requested: CancellationCallback | None = None,
) -> MediaExportJobResult:
    """Create ``reviewed_media_export_overlay__*`` and encode annotated clips."""

    cfg = settings.validated()
    root = Path(session_root).expanduser().resolve()
    ts = _sanitize_timestamp(timestamp or _now_timestamp())
    job_id = export_job_id or f"{REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_PREFIX}{ts}"
    if not str(job_id).startswith(REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_PREFIX):
        job_id = f"{REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_PREFIX}{ts}"
    paths = _paths(root, job_id, output_root=output_root)
    _reject_output_overlap(paths.job_dir, root, sources.package_dir, sources.review_run_dir)
    if paths.job_dir.exists():
        raise ExportPublicationError(
            f"Reviewed overlay media export job already exists: {paths.job_dir}"
        )

    selected = select_bouts_for_export(
        sources.reviewed_bouts,
        cfg.selection,
        package_dir=sources.package_dir,
    )
    selected = _attach_review_status(selected, sources.review_bout_status)
    created_at = _iso_now()
    paths.job_dir.mkdir(parents=True, exist_ok=False)
    settings_snapshot = {
        **cfg.to_settings_dict(),
        "source_package_run_id": sources.package_run_id,
        "source_package_dir": str(sources.package_dir),
        "source_review_run_id": sources.review_run_id,
        "source_review_run_dir": str(sources.review_run_dir),
        "pose_run_id": _pose_run_id(sources),
        "configuration": cfg.configuration_label(),
        "sync_authority": sources.sync_authority,
        "selected_layers": [x.value for x in sources.selected_layers],
        "export_job_id": job_id,
        "source_digests": dict(sources.source_digests),
        "compatibility_notes": list(sources.compatibility_notes),
    }
    warnings: list[str] = []
    write_yaml(paths.settings_used, {**settings_snapshot, "status": STATUS_RUNNING})
    write_json(
        paths.run_meta,
        _run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=root,
            sources=sources,
            settings_snapshot=settings_snapshot,
            paths=paths,
            counts=None,
            error_message=None,
            warnings=warnings,
        ),
    )

    clip_index = build_pending_clip_index(
        selected,
        export_job_id=job_id,
        selection_mode=cfg.selection.mode,
        source_video_kind="prepared_video",
        source_video_path=str(sources.prepared_video_path),
        source_fps=sources.video_fps,
        source_n_frames=sources.video_n_frames,
        pre_context_sec=cfg.pre_context_sec,
        post_context_sec=cfg.post_context_sec,
        crf=cfg.crf,
        codec=cfg.codec,
    )
    if "review_status" in selected.columns:
        status_by_id = {
            str(r["bout_id"]): r.get("review_status")
            for r in selected.to_dict(orient="records")
        }
        clip_index["review_status"] = [
            status_by_id.get(str(b)) for b in clip_index["bout_id"].astype(str)
        ]
    atomic_parquet_write(paths.clip_index, clip_index)
    paths.clips_root.mkdir(parents=True, exist_ok=True)
    paths.staging.mkdir(parents=True, exist_ok=True)

    return _encode_batch(
        session_root=root,
        sources=sources,
        settings=cfg,
        settings_snapshot=settings_snapshot,
        paths=paths,
        job_id=job_id,
        created_at=created_at,
        clip_index=clip_index,
        warnings=warnings,
        progress_callback=progress_callback,
        overlay_progress_callback=overlay_progress_callback,
        cancellation_requested=cancellation_requested,
    )


def _encode_batch(
    *,
    session_root: Path,
    sources: ReviewedOverlayMediaResolvedSources,
    settings: ReviewedOverlayMediaExportSettings,
    settings_snapshot: Mapping[str, Any],
    paths: _Paths,
    job_id: str,
    created_at: str,
    clip_index: pd.DataFrame,
    warnings: list[str],
    progress_callback: OpProgressCallback | None,
    overlay_progress_callback: ProgressCallback | None,
    cancellation_requested: CancellationCallback | None,
) -> MediaExportJobResult:
    total = int(len(clip_index))
    hard_error: str | None = None
    canceled = False
    clip_hashes: dict[str, str] = {}
    clip_manifests: list[dict[str, Any]] = []
    try:
        for i, idx in enumerate(list(clip_index.index)):
            if cancellation_requested is not None and cancellation_requested():
                canceled = True
                hard_error = "Reviewed overlay media export cancelled by caller."
                break
            row = clip_index.loc[idx]
            bout_id = str(row["bout_id"])
            status = str(row["encode_status"])
            rel = str(row["clip_relpath"]).replace("\\", "/")
            out_path = paths.clips_root / Path(rel)
            _report(
                progress_callback,
                phase="encode",
                message=f"Encoding reviewed overlay clip {i + 1}/{total}: {bout_id}",
                completed_units=i,
                total_units=max(total, 1),
                indeterminate=False,
            )
            if status == ENCODE_STATUS_OMITTED_INVALID_BOUNDS:
                continue
            if status in (ENCODE_STATUS_SUCCEEDED, ENCODE_STATUS_SKIPPED_EXISTS):
                if out_path.is_file() and out_path.stat().st_size > 0:
                    clip_index.at[idx, "clip_exists"] = True
                    continue
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_PENDING
                status = ENCODE_STATUS_PENDING
            if status not in (ENCODE_STATUS_PENDING, ENCODE_STATUS_FAILED):
                continue
            if out_path.is_file() and out_path.stat().st_size > 0:
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_SKIPPED_EXISTS
                clip_index.at[idx, "clip_exists"] = True
                clip_index.at[idx, "error"] = None
                continue
            if pd.isna(row.get("clip_frame_idx_start")) or pd.isna(
                row.get("clip_frame_count")
            ):
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_OMITTED_INVALID_BOUNDS
                clip_index.at[idx, "error"] = "Missing clip bounds."
                _append_failure(paths.failures, bout_id, "Missing clip bounds.")
                continue
            try:
                encoded = _encode_one_bout(
                    sources=sources,
                    settings=settings,
                    row=row,
                    output_path=out_path,
                    staging_dir=paths.staging,
                    overlay_progress_callback=overlay_progress_callback,
                    cancellation_requested=cancellation_requested,
                )
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_SUCCEEDED
                clip_index.at[idx, "clip_exists"] = True
                clip_index.at[idx, "error"] = None
                if settings.hash_clips:
                    clip_hashes[rel] = encoded["video_sha256"]
                clip_manifests.append(encoded)
            except OverlayExportCancelled as exc:
                canceled = True
                hard_error = str(exc)
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_FAILED
                clip_index.at[idx, "clip_exists"] = False
                clip_index.at[idx, "error"] = hard_error
                _append_failure(paths.failures, bout_id, hard_error)
                break
            except (OverlayExportError, OverlaySourceError, OverlayContractError) as exc:
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_FAILED
                clip_index.at[idx, "clip_exists"] = False
                clip_index.at[idx, "error"] = str(exc)
                _append_failure(paths.failures, bout_id, str(exc))
                if settings.fail_fast:
                    hard_error = str(exc)
                    break
            except Exception as exc:  # noqa: BLE001 - ledger partial failure
                clip_index.at[idx, "encode_status"] = ENCODE_STATUS_FAILED
                clip_index.at[idx, "clip_exists"] = False
                clip_index.at[idx, "error"] = str(exc)
                _append_failure(paths.failures, bout_id, str(exc))
                if settings.fail_fast:
                    hard_error = str(exc)
                    break
            atomic_parquet_write(paths.clip_index, clip_index)

        if not canceled:
            verify_reviewed_overlay_media_sources_integrity(sources)
    except Exception as exc:
        hard_error = str(exc)
        atomic_parquet_write(paths.clip_index, clip_index)

    counts = _count_encode_statuses(clip_index)
    error_message: str | None = None
    if canceled:
        final_status = STATUS_INCOMPLETE
        error_message = hard_error or "Reviewed overlay media export cancelled."
        warnings.append("Reviewed overlay media export cancelled; partial clips retained.")
    elif hard_error and counts["n_succeeded"] == 0 and counts["n_skipped"] == 0:
        final_status = STATUS_FAILED
        error_message = hard_error
    elif counts["n_failed"] > 0 or counts["n_pending"] > 0:
        if settings.accept_partial and counts["n_pending"] == 0:
            final_status = STATUS_COMPLETED_WITH_WARNINGS
            warnings.append(
                "Accepted partial reviewed overlay media export: "
                f"{counts['n_failed']} failed clips."
            )
            error_message = hard_error
        else:
            final_status = STATUS_INCOMPLETE
            error_message = hard_error
            if counts["n_failed"] > 0:
                warnings.append(
                    "Reviewed overlay media export incomplete: "
                    f"{counts['n_failed']} failed clips."
                )
    else:
        final_status = STATUS_COMPLETED
        error_message = None

    completed_at = _iso_now()
    if paths.staging.exists():
        shutil.rmtree(paths.staging, ignore_errors=True)

    job_manifest = _job_manifest(
        job_id=job_id,
        created_at=created_at,
        completed_at=completed_at,
        status=final_status,
        sources=sources,
        settings_snapshot=settings_snapshot,
        paths=paths,
        counts=counts,
        warnings=warnings,
        error_message=error_message,
        clip_hashes=clip_hashes,
        clip_manifests=clip_manifests,
    )
    qc = {
        "schema_version": "reviewed_overlay_media_export_qc_v1",
        "status": final_status,
        "configuration": settings.configuration_label(),
        "counts": counts,
        "warnings": list(warnings),
        "error_message": error_message,
        "sync_authority": sources.sync_authority,
        "selected_layers": [x.value for x in sources.selected_layers],
        "hash_clips": bool(settings.hash_clips),
        "n_clip_hashes": len(clip_hashes),
        "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
        "s8_label_fallback": False,
    }
    write_json(paths.qc, qc)
    write_json(paths.manifest, job_manifest)
    write_json(
        paths.run_meta,
        _run_meta(
            job_id=job_id,
            created_at=created_at,
            completed_at=completed_at,
            status=final_status,
            session_root=session_root,
            sources=sources,
            settings_snapshot=settings_snapshot,
            paths=paths,
            counts=counts,
            error_message=error_message,
            warnings=warnings,
        ),
    )
    write_yaml(paths.settings_used, {**dict(settings_snapshot), "status": final_status})
    _write_overlay_artifact_manifest(
        paths=paths,
        sources=sources,
        settings=settings,
        clip_hashes=clip_hashes,
        status=final_status,
    )
    atomic_parquet_write(paths.clip_index, clip_index)
    paths.processing_log.write_text(
        "\n".join(
            [
                f"export_job_id: {job_id}",
                f"job_kind: {REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND}",
                f"renderer_version: {OVERLAY_S10_MEDIA_RENDERER_VERSION}",
                f"status: {final_status}",
                f"configuration: {settings.configuration_label()}",
                f"n_selected: {counts['n_selected']}",
                f"n_succeeded: {counts['n_succeeded']}",
                f"n_failed: {counts['n_failed']}",
                f"n_skipped: {counts['n_skipped']}",
                f"n_omitted: {counts['n_omitted']}",
                f"source_review_run_id: {sources.review_run_id}",
                f"source_package_run_id: {sources.package_run_id}",
                f"pose_run_id: {_pose_run_id(sources)}",
                f"sync_authority: {sources.sync_authority}",
                f"error: {error_message}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _report(
        progress_callback,
        phase="complete" if final_status == STATUS_COMPLETED else final_status,
        message=f"Reviewed overlay media export {final_status}.",
        completed_units=total,
        total_units=max(total, 1),
        indeterminate=False,
    )
    return MediaExportJobResult(
        success=final_status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS),
        status=final_status,
        export_job_id=job_id,
        job_dir=paths.job_dir,
        session_root=session_root,
        source_package_run_id=sources.package_run_id,
        source_package_dir=sources.package_dir,
        n_selected=counts["n_selected"],
        n_succeeded=counts["n_succeeded"],
        n_failed=counts["n_failed"],
        n_skipped=counts["n_skipped"],
        n_omitted=counts["n_omitted"],
        clip_index_path=paths.clip_index,
        export_job_manifest_path=paths.manifest,
        qc_path=paths.qc,
        warnings=tuple(warnings),
        error_message=error_message,
        diagnostics={
            "job_kind": REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND,
            "configuration": settings.configuration_label(),
            "clip_hashes": clip_hashes,
            "canceled": canceled,
            "renderer_version": OVERLAY_S10_MEDIA_RENDERER_VERSION,
            "review_run_id": sources.review_run_id,
            "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
            "s8_label_fallback": False,
        },
    )


def _encode_one_bout(
    *,
    sources: ReviewedOverlayMediaResolvedSources,
    settings: ReviewedOverlayMediaExportSettings,
    row: pd.Series,
    output_path: Path,
    staging_dir: Path,
    overlay_progress_callback: ProgressCallback | None,
    cancellation_requested: CancellationCallback | None,
) -> dict[str, Any]:
    bounds = clip_bounds_from_index_row(row)
    interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    bout_id = str(row["bout_id"])
    start_sample = int(row["start_sample_idx"])
    end_sample = int(row["end_sample_idx"])
    packaged_prep_start = int(row["prepared_frame_idx_start"])
    packaged_prep_end = int(row["prepared_frame_idx_end"])
    adapters, sync, layer_timing_resolver, plan_notes = build_reviewed_bout_adapters(
        sources=sources,
        settings=settings,
        interval=interval,
        bout_id=bout_id,
        start_sample_idx=start_sample,
        end_sample_idx=end_sample,
        packaged_prepared_start=packaged_prep_start,
        packaged_prepared_end=packaged_prep_end,
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_out = staging_dir / f"{bout_id}.mp4"
    if staging_out.exists():
        staging_out.unlink()
    result = stream_composed_prepared_clip(
        StreamComposedClipRequest(
            prepared_video_path=sources.prepared_video_path,
            sync=sync,
            adapters=adapters,
            prepared_interval=interval,
            selected_layers=sources.selected_layers,
            style=sources.style,
            output_path=staging_out,
            fps=sources.video_fps,
            width=sources.video_width,
            height=sources.video_height,
            layer_timing_resolver=layer_timing_resolver,
            crf=settings.crf,
            codec=settings.codec,
            pixel_format=settings.pixel_format,
            progress_callback=overlay_progress_callback,
            cancellation_requested=cancellation_requested,
            progress_extra={"bout_id": bout_id},
        )
    )
    verify_reviewed_overlay_media_sources_integrity(sources)
    video_sha = sha256_file(result.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise OverlayExportError(
            f"Refusing to overwrite existing reviewed overlay clip: {output_path}"
        )
    result.output_path.replace(output_path)
    return {
        "bout_id": bout_id,
        "clip_relpath": str(row["clip_relpath"]).replace("\\", "/"),
        "video_sha256": video_sha,
        "frames_written": result.frames_written,
        "prepared_interval": interval.to_dict(),
        "start_sample_idx": start_sample,
        "end_sample_idx": end_sample,
        "packaged_prepared_frame_idx_start": packaged_prep_start,
        "packaged_prepared_frame_idx_end": packaged_prep_end,
        "review_status": (
            None if "review_status" not in row.index else row.get("review_status")
        ),
        "unavailable_layers": list(result.unavailable_layers),
        "notes": list(plan_notes) + list(result.notes),
        "first_frame_timing": (
            None if result.first_timing is None else result.first_timing.to_dict()
        ),
        "last_frame_timing": (
            None if result.last_timing is None else result.last_timing.to_dict()
        ),
        "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
    }


def build_reviewed_bout_adapters(
    *,
    sources: ReviewedOverlayMediaResolvedSources,
    settings: ReviewedOverlayMediaExportSettings,
    interval: PreparedFrameInterval,
    bout_id: str,
    start_sample_idx: int,
    end_sample_idx: int,
    packaged_prepared_start: int,
    packaged_prepared_end: int,
) -> tuple[
    dict[LayerId, OverlayStageAdapter],
    Any,
    Callable[[int], dict[LayerId, PreparedFrameTiming] | None] | None,
    tuple[str, ...],
]:
    if sources.mode == "mixed_s4":
        assert sources.mixed is not None
        seek_plan = plan_bout_span(
            sync=sources.mixed.sync,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
            source_fps=sources.video_fps,
            source_n_frames=sources.video_n_frames,
            bout_id=bout_id,
            packaged_prepared_start=packaged_prepared_start,
            packaged_prepared_end=packaged_prepared_end,
            hud_mode=settings.hud_mode,
            include_contact=settings.include_contact,
            include_sbs=settings.include_sbs,
            include_frame_counter=settings.include_frame_counter,
            pre_context_sec=settings.pre_context_sec,
            post_context_sec=settings.post_context_sec,
        )
        if (
            seek_plan.prepared_interval.start != interval.start
            or seek_plan.prepared_interval.end != interval.end
        ):
            raise OverlayContractError(
                "Mixed reviewed overlay bout padded interval mismatch "
                f"(plan={seek_plan.prepared_interval.to_dict()} "
                f"clip_index={interval.to_dict()})."
            )
        plan = MixedPoseBehaviorQcExportPlan(
            scope_kind=seek_plan.scope_kind,
            prepared_interval=interval,
            pose_source_kind=sources.mixed.pose_source_kind,
            hud_mode=HudDirectionMode(settings.hud_mode),
            selected_layers=sources.selected_layers,
            plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            include_contact=settings.include_contact,
            include_sbs=settings.include_sbs,
            draw_skeleton=settings.draw_skeleton,
            style=sources.style,
            pre_context_sec=settings.pre_context_sec,
            post_context_sec=settings.post_context_sec,
            clip_bounds=seek_plan.clip_bounds,
            notes=seek_plan.notes,
        )
        adapters = build_mixed_adapters(sources.mixed, plan=plan)
        sync = sources.mixed.sync

        def _timing(prep: int) -> dict[LayerId, PreparedFrameTiming] | None:
            if sources.mixed is None or sources.mixed.pose_s4 is None:
                return None
            pose_timing = resolve_prepared_frame_timing(
                sources.mixed.pose_s4.sync,
                prep,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            )
            return {LayerId.S4_POSE_FINAL: pose_timing}

        return adapters, sync, _timing, tuple(seek_plan.notes)

    if sources.mode == "behavior":
        assert sources.behavior is not None
        behavior_plan = plan_bout_span(
            sync=sources.behavior.sync,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
            source_fps=sources.video_fps,
            source_n_frames=sources.video_n_frames,
            bout_id=bout_id,
            packaged_prepared_start=packaged_prepared_start,
            packaged_prepared_end=packaged_prepared_end,
            hud_mode=settings.hud_mode,
            include_contact=settings.include_contact,
            include_sbs=settings.include_sbs,
            include_frame_counter=settings.include_frame_counter,
            selected_layers=sources.selected_layers,
            pre_context_sec=settings.pre_context_sec,
            post_context_sec=settings.post_context_sec,
            style=sources.style,
        )
        if (
            behavior_plan.prepared_interval.start != interval.start
            or behavior_plan.prepared_interval.end != interval.end
        ):
            raise OverlayContractError(
                "Reviewed behavioral overlay bout padded interval mismatch "
                f"(plan={behavior_plan.prepared_interval.to_dict()} "
                f"clip_index={interval.to_dict()})."
            )
        dense_range = dense_sample_window_for_prepared_interval(
            sources.behavior.sync, interval
        )
        adapters = build_behavioral_adapters(
            sources.behavior,
            dense_sample_range=dense_range,
            selected_layers=sources.selected_layers,
        )
        return adapters, sources.behavior.sync, None, tuple(behavior_plan.notes)

    if sources.mode == "pose_s4":
        assert sources.pose_s4 is not None
        s4_plan = plan_s4_prepared_interval(
            prepared_frame_idx_start=packaged_prepared_start,
            prepared_frame_idx_end=packaged_prepared_end,
            source_fps=sources.video_fps,
            source_n_frames=sources.video_n_frames,
            include_frame_counter=settings.include_frame_counter,
            draw_skeleton=settings.draw_skeleton,
            selected_layers=sources.selected_layers,
            pre_context_sec=settings.pre_context_sec,
            post_context_sec=settings.post_context_sec,
            style=sources.style,
        )
        if (
            s4_plan.prepared_interval.start != interval.start
            or s4_plan.prepared_interval.end != interval.end
        ):
            raise OverlayContractError(
                "S4 reviewed overlay bout padded interval mismatch "
                f"(plan={s4_plan.prepared_interval.to_dict()} "
                f"clip_index={interval.to_dict()})."
            )
        dense_range = dense_sample_window_for_prepared_interval(
            sources.pose_s4.sync, interval
        )
        adapters = build_s4_pose_adapters(
            sources.pose_s4,
            dense_sample_range=dense_range,
            selected_layers=sources.selected_layers,
        )
        return adapters, sources.pose_s4.sync, None, tuple(s4_plan.notes)

    raise OverlayExportError(f"Unsupported reviewed overlay media mode: {sources.mode!r}.")


@dataclass(frozen=True, slots=True)
class _Paths:
    job_dir: Path
    run_meta: Path
    settings_used: Path
    manifest: Path
    qc: Path
    processing_log: Path
    clip_index: Path
    failures: Path
    clips_root: Path
    staging: Path
    overlay_manifest: Path


def _paths(
    session_root: Path, job_id: str, *, output_root: Path | None
) -> _Paths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else Path(session_root) / RESULTS_EXPORTS_DIRNAME
    )
    job_dir = root / job_id
    return _Paths(
        job_dir=job_dir,
        run_meta=job_dir / RUN_META_FILENAME,
        settings_used=job_dir / SETTINGS_USED_FILENAME,
        manifest=job_dir / REVIEWED_MEDIA_EXPORT_OVERLAY_MANIFEST_FILENAME,
        qc=job_dir / QC_FILENAME,
        processing_log=job_dir / PROCESSING_LOG_FILENAME,
        clip_index=job_dir / CLIP_INDEX_FILENAME,
        failures=job_dir / FAILURES_JSONL_FILENAME,
        clips_root=job_dir / CLIPS_DIRNAME,
        staging=job_dir / REVIEWED_MEDIA_EXPORT_OVERLAY_STAGING_DIRNAME,
        overlay_manifest=job_dir / OVERLAY_QC_MANIFEST_FILENAME,
    )


def _resolve_settings(
    request: ReviewedOverlayMediaExportRequest,
) -> ReviewedOverlayMediaExportSettings:
    cfg = (
        request.settings.validated()
        if request.settings is not None
        else ReviewedOverlayMediaExportSettings().validated()
    )
    if request.selection is not None:
        cfg = replace(cfg, selection=request.selection.validated()).validated()
    return cfg


def _assert_video_matches(
    *,
    review_video: Path,
    review_sha: str,
    other_video: Path,
    other_sha: str,
) -> None:
    if Path(review_video).resolve() != Path(other_video).resolve():
        raise OverlaySourceError(
            "Pose prepared video path does not match published review video "
            f"({Path(other_video).resolve()} vs {Path(review_video).resolve()})."
        )
    if review_sha and other_sha and str(review_sha).lower() != str(other_sha).lower():
        raise OverlaySourceError(
            "Pose prepared-video digest does not match published review "
            "(refusing cross-session / digest-mismatched sources)."
        )


def _load_review_bout_status(review_run_dir: Path) -> pd.DataFrame | None:
    from behavioral_review.contracts import REVIEW_BOUT_STATUS_PARQUET_FILENAME

    path = Path(review_run_dir) / REVIEW_BOUT_STATUS_PARQUET_FILENAME
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _attach_review_status(
    bouts: pd.DataFrame, status: pd.DataFrame | None
) -> pd.DataFrame:
    frame = bouts.copy()
    if status is None or status.empty or "bout_id" not in frame.columns:
        if "review_status" not in frame.columns:
            frame["review_status"] = None
        return frame
    status_col = None
    for candidate in ("status", "review_status", "bout_status"):
        if candidate in status.columns:
            status_col = candidate
            break
    id_col = None
    for candidate in ("bout_id", "reviewed_bout_id", "target_bout_id"):
        if candidate in status.columns:
            id_col = candidate
            break
    if status_col is None or id_col is None:
        frame["review_status"] = None
        return frame
    mapping = {
        str(r[id_col]): r[status_col] for r in status.to_dict(orient="records")
    }
    frame["review_status"] = [mapping.get(str(b)) for b in frame["bout_id"].astype(str)]
    return frame


def _pose_run_id(sources: ReviewedOverlayMediaResolvedSources) -> str | None:
    if sources.pose_s4 is not None:
        return sources.pose_s4.s4_run_id
    if sources.mixed is not None:
        return sources.mixed.pose_run_id
    return None


def _count_encode_statuses(clip_index: pd.DataFrame) -> dict[str, int]:
    statuses = (
        clip_index["encode_status"].astype(str).tolist() if len(clip_index) else []
    )
    return {
        "n_selected": int(len(clip_index)),
        "n_succeeded": sum(1 for s in statuses if s == ENCODE_STATUS_SUCCEEDED),
        "n_failed": sum(1 for s in statuses if s == ENCODE_STATUS_FAILED),
        "n_skipped": sum(1 for s in statuses if s == ENCODE_STATUS_SKIPPED_EXISTS),
        "n_omitted": sum(
            1 for s in statuses if s == ENCODE_STATUS_OMITTED_INVALID_BOUNDS
        ),
        "n_pending": sum(1 for s in statuses if s == ENCODE_STATUS_PENDING),
    }


def _append_failure(path: Path, bout_id: str, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "bout_id": bout_id,
                    "error": error,
                    "at_utc": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _reject_output_overlap(
    job_dir: Path,
    session_root: Path,
    package_dir: Path,
    review_run_dir: Path,
) -> None:
    resolved = job_dir.resolve()
    for protected in _PROTECTED_SESSION_SUBDIRS:
        protected_root = (session_root / protected).resolve()
        if resolved == protected_root or protected_root in resolved.parents:
            raise ExportPublicationError(
                f"Reviewed overlay media export must not write under {protected}/."
            )
    for science in (package_dir, review_run_dir):
        path = Path(science).resolve()
        if resolved == path or path in resolved.parents:
            raise ExportPublicationError(
                "Reviewed overlay media export must not write into science trees."
            )


def _run_meta(
    *,
    job_id: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    session_root: Path,
    sources: ReviewedOverlayMediaResolvedSources,
    settings_snapshot: Mapping[str, Any],
    paths: _Paths,
    counts: dict[str, int] | None,
    error_message: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "reviewed_overlay_media_export_run_meta_v1",
        "export_job_id": job_id,
        "job_kind": REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND,
        "renderer_version": OVERLAY_S10_MEDIA_RENDERER_VERSION,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "session_root": str(session_root),
        "job_dir": str(paths.job_dir),
        "configuration": settings_snapshot.get("configuration"),
        "source_package": {
            "run_id": sources.package_run_id,
            "path": str(sources.package_dir),
            "prepared_video_path": str(sources.prepared_video_path),
            "prepared_video_sha256": sources.prepared_video_sha256,
        },
        "source_review": {
            "run_id": sources.review_run_id,
            "path": str(sources.review_run_dir),
        },
        "pose_run_id": _pose_run_id(sources),
        "pose_source_kind": settings_snapshot.get("pose_source_kind"),
        "include_behavior": settings_snapshot.get("include_behavior"),
        "include_pose": settings_snapshot.get("include_pose"),
        "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
        "s8_label_fallback": False,
        "sync_authority": sources.sync_authority,
        "selected_layers": list(settings_snapshot.get("selected_layers") or []),
        "source_digests": dict(sources.source_digests),
        "counts": counts,
        "error_message": error_message,
        "warnings": list(warnings),
        "clip_index_path": str(paths.clip_index),
        "clips_root": str(paths.clips_root),
    }


def _job_manifest(
    *,
    job_id: str,
    created_at: str,
    completed_at: str,
    status: str,
    sources: ReviewedOverlayMediaResolvedSources,
    settings_snapshot: Mapping[str, Any],
    paths: _Paths,
    counts: dict[str, int],
    warnings: list[str],
    error_message: str | None,
    clip_hashes: Mapping[str, str],
    clip_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "reviewed_overlay_media_export_manifest_v1",
        "export_job_id": job_id,
        "job_kind": REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND,
        "renderer_version": OVERLAY_S10_MEDIA_RENDERER_VERSION,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "configuration": settings_snapshot.get("configuration"),
        "settings": dict(settings_snapshot),
        "source_package_run_id": sources.package_run_id,
        "source_package_dir": str(sources.package_dir),
        "source_review_run_id": sources.review_run_id,
        "source_review_run_dir": str(sources.review_run_dir),
        "pose_run_id": _pose_run_id(sources),
        "pose_source_kind": settings_snapshot.get("pose_source_kind"),
        "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
        "s8_label_fallback": False,
        "prepared_video_path": str(sources.prepared_video_path),
        "prepared_video_sha256": sources.prepared_video_sha256,
        "sync_authority": sources.sync_authority,
        "per_source_timing_authority": {
            "behavior": (
                "package_sync_index_last_in_plateau"
                if sources.behavior is not None or sources.mixed is not None
                else None
            ),
            "pose_s4": (
                "pose_syncmap_last_in_plateau"
                if sources.pose_s4 is not None
                or (sources.mixed is not None and sources.mixed.pose_s4 is not None)
                else None
            ),
        },
        "selected_layers": [x.value for x in sources.selected_layers],
        "source_digests": dict(sources.source_digests),
        "compatibility_notes": list(sources.compatibility_notes),
        "counts": counts,
        "warnings": list(warnings),
        "error_message": error_message,
        "clips_root": str(paths.clips_root),
        "clip_index_path": str(paths.clip_index),
        "clip_hashes_sha256": dict(clip_hashes),
        "clips": clip_manifests,
        "portable_clip_paths": True,
        "scientific_artifact_writes": False,
    }


def _write_overlay_artifact_manifest(
    *,
    paths: _Paths,
    sources: ReviewedOverlayMediaResolvedSources,
    settings: ReviewedOverlayMediaExportSettings,
    clip_hashes: Mapping[str, str],
    status: str,
) -> None:
    sync = (
        sources.mixed.sync
        if sources.mixed is not None
        else sources.behavior.sync
        if sources.behavior is not None
        else sources.pose_s4.sync  # type: ignore[union-attr]
    )
    example_timing = resolve_prepared_frame_timing(
        sync,
        0,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
    )
    prepared_video = PreparedVideoIdentity(
        path=sources.prepared_video_path,
        digest_sha256=sources.prepared_video_sha256,
        fps=sources.video_fps,
        frame_count=sources.video_n_frames,
        extra={
            "width": sources.video_width,
            "height": sources.video_height,
            "sync_authority": sources.sync_authority,
        },
    )
    identities: tuple[SourceIdentity, ...] = ()
    if sources.mixed is not None:
        from overlay.export_mixed import _build_source_identities

        identities = _build_source_identities(sources.mixed)
    manifest = build_overlay_artifact_manifest(
        sync=sync,
        prepared_video=prepared_video,
        source_identities=identities,
        selected_layers=sources.selected_layers,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        style=sources.style,
        timing=example_timing,
        output_frame_domain=FrameDomain.PREPARED,
        output_fps=sources.video_fps,
        notes=sources.compatibility_notes,
        renderer_version=OVERLAY_S10_MEDIA_RENDERER_VERSION,
        extra={
            "export_kind": "s10_reviewed_overlay_media_batch",
            "job_kind": REVIEWED_MEDIA_EXPORT_OVERLAY_JOB_KIND,
            "qc_status": status,
            "configuration": settings.configuration_label(),
            "clip_hashes_sha256": dict(clip_hashes),
            "source_digests": dict(sources.source_digests),
            "provenance": dict(sources.provenance),
            "behavior_source_kind": BehaviorSourceKind.S10_REVIEWED.value,
            "s8_label_fallback": False,
            "review_run_id": sources.review_run_id,
        },
    )
    OverlayArtifactManifest(
        schema_version=manifest.schema_version,
        renderer_version=manifest.renderer_version,
        source_runs=manifest.source_runs,
        source_digests=manifest.source_digests,
        prepared_video=manifest.prepared_video,
        timing_mapping=manifest.timing_mapping,
        plateau_policy=manifest.plateau_policy,
        layer_selections=manifest.layer_selections,
        unavailable_layers=manifest.unavailable_layers,
        output_frame_domain=manifest.output_frame_domain,
        output_fps=manifest.output_fps,
        render_settings=manifest.render_settings,
        pixel_format=manifest.pixel_format,
        notes=manifest.notes,
        extra=manifest.extra,
    ).write_json(paths.overlay_manifest)


def _report(
    callback: OpProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed_units: int | None,
    total_units: int | None,
    indeterminate: bool,
) -> None:
    if callback is None:
        return
    callback(
        OperationProgress(
            phase=phase,
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            total_is_estimate=False,
            is_indeterminate=indeterminate,
            processed_video_time_sec=None,
        )
    )


def _now_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _sanitize_timestamp(value: str) -> str:
    text = "".join(ch for ch in str(value) if ch.isalnum() or ch in {"T", "_", "-"})
    return text or _now_timestamp()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "ReviewedOverlayMediaExportRequest",
    "ReviewedOverlayMediaExportSettings",
    "ReviewedOverlayMediaResolvedSources",
    "build_reviewed_bout_adapters",
    "publish_reviewed_overlay_media_export",
    "resolve_reviewed_overlay_media_sources",
    "run_reviewed_overlay_media_export",
    "verify_reviewed_overlay_media_sources_integrity",
]
