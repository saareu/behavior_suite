"""Materialize browser-compatible PI report preview clips."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from behavioral_review.gallery import GalleryCard
from behavioral_review.pi_report.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_MOVFLAGS,
    DEFAULT_PIXEL_FORMAT,
    MEDIA_DIRNAME,
    MEDIA_ENCODE_FAILED,
    MEDIA_ENCODE_SUCCEEDED,
    PiReportCancelledError,
    PiReportMediaError,
)
from results_packaging.media_csv_export.clip_bounds import (
    ClipBounds,
    compute_clip_bounds,
)
from results_packaging.media_csv_export.contracts import ExportVideoError
from results_packaging.media_csv_export.video import (
    encode_mp4_clip,
    probe_encoded_clip,
    sanitize_path_token,
    validate_bout_id_for_filename,
)

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class MediaClipResult:
    """One bout → local relative MP4 mapping."""

    bout_id: str
    card_id: str
    media_relpath: str
    encode_status: str
    clip_frame_idx_start: int
    clip_frame_idx_end: int
    clip_frame_count: int
    prepared_frame_idx_start: int
    prepared_frame_idx_end: int
    source_fps: float
    sha256: str | None
    error_message: str | None = None
    annotated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "bout_id": self.bout_id,
            "card_id": self.card_id,
            "media_relpath": self.media_relpath,
            "encode_status": self.encode_status,
            "clip_frame_idx_start": self.clip_frame_idx_start,
            "clip_frame_idx_end": self.clip_frame_idx_end,
            "clip_frame_count": self.clip_frame_count,
            "prepared_frame_idx_start": self.prepared_frame_idx_start,
            "prepared_frame_idx_end": self.prepared_frame_idx_end,
            "source_fps": self.source_fps,
            "sha256": self.sha256,
            "error_message": self.error_message,
            "annotated": bool(self.annotated),
        }


@dataclass(frozen=True, slots=True)
class MediaMaterializationResult:
    """Aggregate media encode outcome for one report."""

    clips: tuple[MediaClipResult, ...]
    n_succeeded: int
    n_failed: int
    media_mapping: dict[str, str]
    overlay_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "n_clips": len(self.clips),
            "media_mapping": dict(self.media_mapping),
            "clips": [c.to_dict() for c in self.clips],
            "overlay_provenance": dict(self.overlay_provenance),
        }


def media_relpath_for_bout(bout_id: str) -> str:
    """Deterministic portable relative path under ``media/`` (no absolute paths)."""

    safe = sanitize_path_token(validate_bout_id_for_filename(bout_id), fallback="bout")
    return f"{MEDIA_DIRNAME}/{safe}.mp4"


def preview_plan_to_clip_bounds(card: GalleryCard) -> ClipBounds:
    """Recompute S9 ClipBounds from gallery prepared endpoints + pad.

    Reuses ``compute_clip_bounds`` so pad / clamp semantics match gallery and
    media export exactly (including plateau-collapsed prepared spans).
    """

    plan = card.preview_plan
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=int(plan.prepared_frame_idx_start),
        prepared_frame_idx_end=int(plan.prepared_frame_idx_end),
        source_fps=float(plan.source_fps),
        source_n_frames=int(plan.source_n_frames),
        pre_context_sec=float(plan.pre_context_sec),
        post_context_sec=float(plan.post_context_sec),
    )
    if (
        bounds.clip_frame_idx_start != int(plan.clip_frame_idx_start)
        or bounds.clip_frame_idx_end != int(plan.clip_frame_idx_end)
        or bounds.clip_frame_count != int(plan.clip_frame_count)
    ):
        raise PiReportMediaError(
            f"Gallery preview_plan diverges from recomputed clip bounds for "
            f"bout {card.bout_id}."
        )
    return bounds


def materialize_report_media(
    *,
    cards: Sequence[GalleryCard],
    source_video: Path,
    report_root: Path,
    ffmpeg_executable: Path | None = None,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    movflags: str | None = DEFAULT_MOVFLAGS,
    fail_fast: bool = True,
    progress_callback: ProgressCallback | None = None,
    digest_fn: Callable[[Path], str] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> MediaMaterializationResult:
    """Encode one H.264 MP4 preview per exclusive card into ``report_root/media``.

    Uses S9 frame-accurate ``encode_mp4_clip`` (ffmpeg stream; no full-video load).
    Relative media paths only; never records laboratory absolute video paths.
    Cooperative cancel is checked between bout encodes (not mid-ffmpeg).
    """

    from project.file_digest import sha256_file

    digest = digest_fn or sha256_file
    root = Path(report_root).expanduser().resolve()
    media_dir = root / MEDIA_DIRNAME
    media_dir.mkdir(parents=True, exist_ok=True)

    results: list[MediaClipResult] = []
    mapping: dict[str, str] = {}
    total = len(cards)
    for index, card in enumerate(cards, start=1):
        if cancellation_requested is not None and cancellation_requested():
            raise PiReportCancelledError(
                f"PI report media encode cancelled before bout {card.bout_id} "
                f"({index - 1}/{total} completed)."
            )
        if progress_callback is not None:
            progress_callback(index, total, card.bout_id)
        relpath = media_relpath_for_bout(card.bout_id)
        out_path = root / Path(relpath)
        bounds = preview_plan_to_clip_bounds(card)
        try:
            encode_mp4_clip(
                source_video=Path(source_video),
                output_path=out_path,
                bounds=bounds,
                ffmpeg_executable=ffmpeg_executable,
                crf=crf,
                codec=codec,
                pixel_format=pixel_format,
                verify_encoded_boundaries=True,
                movflags=movflags,
            )
            probed = probe_encoded_clip(out_path)
            if int(probed.n_frames) != int(bounds.clip_frame_count):
                raise PiReportMediaError(
                    f"Encoded preview frame count mismatch for bout "
                    f"{card.bout_id}: {probed.n_frames} vs {bounds.clip_frame_count}."
                )
            clip = MediaClipResult(
                bout_id=card.bout_id,
                card_id=card.card_id,
                media_relpath=relpath.replace("\\", "/"),
                encode_status=MEDIA_ENCODE_SUCCEEDED,
                clip_frame_idx_start=bounds.clip_frame_idx_start,
                clip_frame_idx_end=bounds.clip_frame_idx_end,
                clip_frame_count=bounds.clip_frame_count,
                prepared_frame_idx_start=bounds.prepared_frame_idx_start,
                prepared_frame_idx_end=bounds.prepared_frame_idx_end,
                source_fps=bounds.source_fps,
                sha256=digest(out_path),
            )
            mapping[card.bout_id] = clip.media_relpath
            results.append(clip)
        except (ExportVideoError, PiReportMediaError, OSError) as exc:
            if fail_fast:
                raise PiReportMediaError(
                    f"Failed encoding PI report preview for bout {card.bout_id}: {exc}"
                ) from exc
            results.append(
                MediaClipResult(
                    bout_id=card.bout_id,
                    card_id=card.card_id,
                    media_relpath=relpath.replace("\\", "/"),
                    encode_status=MEDIA_ENCODE_FAILED,
                    clip_frame_idx_start=bounds.clip_frame_idx_start,
                    clip_frame_idx_end=bounds.clip_frame_idx_end,
                    clip_frame_count=bounds.clip_frame_count,
                    prepared_frame_idx_start=bounds.prepared_frame_idx_start,
                    prepared_frame_idx_end=bounds.prepared_frame_idx_end,
                    source_fps=bounds.source_fps,
                    sha256=None,
                    error_message=str(exc),
                )
            )

    n_succeeded = sum(1 for c in results if c.encode_status == MEDIA_ENCODE_SUCCEEDED)
    n_failed = sum(1 for c in results if c.encode_status == MEDIA_ENCODE_FAILED)
    return MediaMaterializationResult(
        clips=tuple(results),
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        media_mapping=mapping,
    )


def materialize_report_overlay_media(
    *,
    cards: Sequence[GalleryCard],
    review_run_dir: Path,
    report_root: Path,
    include_behavior: bool = True,
    include_pose: bool = False,
    pose_run_dir: Path | str | None = None,
    hud_mode: str = "both",
    include_contact: bool = True,
    include_sbs: bool = True,
    draw_skeleton: bool = True,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    fail_fast: bool = True,
    progress_callback: ProgressCallback | None = None,
    digest_fn: Callable[[Path], str] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> MediaMaterializationResult:
    """Encode one annotated H.264 MP4 per exclusive card (replaces naked media).

    Uses published S10 reviewed states via the shared compositor. Never falls
    back to S8 automated labels. Optional compatible S4 pose only.
    Preserves gallery preview pad / prepared spans (one representation per bout).
    """

    import shutil

    from overlay.contracts import (
        OVERLAY_S10_MEDIA_RENDERER_VERSION,
        OverlayExportCancelled,
        OverlayExportError,
        OverlaySourceError,
        PoseSourceKind,
    )
    from overlay.export_s10_media import (
        ReviewedOverlayMediaExportSettings,
        build_reviewed_bout_adapters,
        resolve_reviewed_overlay_media_sources,
        verify_reviewed_overlay_media_sources_integrity,
    )
    from overlay.plan import PreparedFrameInterval
    from overlay.stream_clip import StreamComposedClipRequest, stream_composed_prepared_clip
    from project.file_digest import sha256_file

    digest = digest_fn or sha256_file
    root = Path(report_root).expanduser().resolve()
    media_dir = root / MEDIA_DIRNAME
    media_dir.mkdir(parents=True, exist_ok=True)

    if not include_behavior and not include_pose:
        raise PiReportMediaError(
            "PI report overlays require reviewed behavioral labels and/or S4 pose."
        )

    settings = ReviewedOverlayMediaExportSettings(
        include_behavior=bool(include_behavior),
        include_pose=bool(include_pose),
        pose_source_kind=(
            PoseSourceKind.S4_FINALIZED.value if include_pose else None
        ),
        pose_run_dir=str(pose_run_dir) if include_pose and pose_run_dir else None,
        hud_mode=str(hud_mode),
        include_contact=bool(include_contact),
        include_sbs=bool(include_sbs),
        draw_skeleton=bool(draw_skeleton),
        crf=int(crf),
        codec=str(codec),
        pixel_format=str(pixel_format),
        fail_fast=bool(fail_fast),
        hash_clips=True,
    ).validated()
    sources = resolve_reviewed_overlay_media_sources(
        review_run_dir=review_run_dir,
        settings=settings,
    )

    results: list[MediaClipResult] = []
    mapping: dict[str, str] = {}
    total = len(cards)
    staging = root / "_pi_overlay_staging"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for index, card in enumerate(cards, start=1):
            if cancellation_requested is not None and cancellation_requested():
                raise PiReportCancelledError(
                    f"PI report overlay media encode cancelled before bout "
                    f"{card.bout_id} ({index - 1}/{total} completed)."
                )
            if progress_callback is not None:
                progress_callback(index, total, card.bout_id)
            relpath = media_relpath_for_bout(card.bout_id)
            out_path = root / Path(relpath)
            bounds = preview_plan_to_clip_bounds(card)
            try:
                interval = PreparedFrameInterval(
                    start=int(bounds.clip_frame_idx_start),
                    end=int(bounds.clip_frame_idx_end),
                )
                # plan_bout_span validates science endpoints + pad against the
                # gallery clip window; feed the card pad through settings.
                pad_settings = ReviewedOverlayMediaExportSettings(
                    include_behavior=settings.include_behavior,
                    include_pose=settings.include_pose,
                    pose_source_kind=settings.pose_source_kind,
                    pose_run_dir=settings.pose_run_dir,
                    hud_mode=settings.hud_mode,
                    include_contact=settings.include_contact,
                    include_sbs=settings.include_sbs,
                    draw_skeleton=settings.draw_skeleton,
                    pre_context_sec=float(card.preview_plan.pre_context_sec),
                    post_context_sec=float(card.preview_plan.post_context_sec),
                    crf=settings.crf,
                    codec=settings.codec,
                    pixel_format=settings.pixel_format,
                ).validated()
                adapters, sync, layer_timing_resolver, _notes = build_reviewed_bout_adapters(
                    sources=sources,
                    settings=pad_settings,
                    interval=interval,
                    bout_id=str(card.bout_id),
                    start_sample_idx=int(card.start_sample_idx),
                    end_sample_idx=int(card.end_sample_idx),
                    packaged_prepared_start=int(card.prepared_frame_idx_start),
                    packaged_prepared_end=int(card.prepared_frame_idx_end),
                )
                staging_out = staging / f"{sanitize_path_token(card.bout_id)}.mp4"
                if staging_out.exists():
                    staging_out.unlink()
                composed = stream_composed_prepared_clip(
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
                        cancellation_requested=cancellation_requested,
                        progress_extra={"bout_id": card.bout_id},
                    )
                )
                verify_reviewed_overlay_media_sources_integrity(sources)
                probed = probe_encoded_clip(composed.output_path)
                if int(probed.n_frames) != int(bounds.clip_frame_count):
                    raise PiReportMediaError(
                        f"Annotated preview frame count mismatch for bout "
                        f"{card.bout_id}: {probed.n_frames} vs {bounds.clip_frame_count}."
                    )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    out_path.unlink()
                composed.output_path.replace(out_path)
                clip = MediaClipResult(
                    bout_id=card.bout_id,
                    card_id=card.card_id,
                    media_relpath=relpath.replace("\\", "/"),
                    encode_status=MEDIA_ENCODE_SUCCEEDED,
                    clip_frame_idx_start=bounds.clip_frame_idx_start,
                    clip_frame_idx_end=bounds.clip_frame_idx_end,
                    clip_frame_count=bounds.clip_frame_count,
                    prepared_frame_idx_start=bounds.prepared_frame_idx_start,
                    prepared_frame_idx_end=bounds.prepared_frame_idx_end,
                    source_fps=bounds.source_fps,
                    sha256=digest(out_path),
                    annotated=True,
                )
                mapping[card.bout_id] = clip.media_relpath
                results.append(clip)
            except OverlayExportCancelled as exc:
                raise PiReportCancelledError(str(exc)) from exc
            except (
                PiReportMediaError,
                OverlayExportError,
                OverlaySourceError,
                ExportVideoError,
                OSError,
            ) as exc:
                if fail_fast:
                    raise PiReportMediaError(
                        f"Failed encoding annotated PI preview for bout "
                        f"{card.bout_id}: {exc}"
                    ) from exc
                results.append(
                    MediaClipResult(
                        bout_id=card.bout_id,
                        card_id=card.card_id,
                        media_relpath=relpath.replace("\\", "/"),
                        encode_status=MEDIA_ENCODE_FAILED,
                        clip_frame_idx_start=bounds.clip_frame_idx_start,
                        clip_frame_idx_end=bounds.clip_frame_idx_end,
                        clip_frame_count=bounds.clip_frame_count,
                        prepared_frame_idx_start=bounds.prepared_frame_idx_start,
                        prepared_frame_idx_end=bounds.prepared_frame_idx_end,
                        source_fps=bounds.source_fps,
                        sha256=None,
                        error_message=str(exc),
                        annotated=True,
                    )
                )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    n_succeeded = sum(1 for c in results if c.encode_status == MEDIA_ENCODE_SUCCEEDED)
    n_failed = sum(1 for c in results if c.encode_status == MEDIA_ENCODE_FAILED)
    provenance = {
        "overlays_enabled": True,
        "representation": "annotated_only",
        "renderer_version": OVERLAY_S10_MEDIA_RENDERER_VERSION,
        "behavior_source_kind": "s10_reviewed",
        "s8_label_fallback": False,
        "include_behavior": bool(settings.include_behavior),
        "include_pose": bool(settings.include_pose),
        "pose_source_kind": settings.pose_source_kind,
        "pose_run_dir": settings.pose_run_dir,
        "review_run_id": sources.review_run_id,
        "package_run_id": sources.package_run_id,
        "source_digests": dict(sources.source_digests),
        "sync_authority": sources.sync_authority,
        "selected_layers": [x.value for x in sources.selected_layers],
        "configuration": settings.configuration_label(),
        "n_cards": int(total),
        "reviewed_bout_ids": [c.bout_id for c in cards],
        "reviewed_bout_table_rows": int(len(sources.reviewed_bouts)),
    }
    return MediaMaterializationResult(
        clips=tuple(results),
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        media_mapping=mapping,
        overlay_provenance=provenance,
    )


__all__ = [
    "MediaClipResult",
    "MediaMaterializationResult",
    "materialize_report_media",
    "materialize_report_overlay_media",
    "media_relpath_for_bout",
    "preview_plan_to_clip_bounds",
]
