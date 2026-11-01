"""Materialize browser-compatible PI report preview clips."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
        }


@dataclass(frozen=True, slots=True)
class MediaMaterializationResult:
    """Aggregate media encode outcome for one report."""

    clips: tuple[MediaClipResult, ...]
    n_succeeded: int
    n_failed: int
    media_mapping: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "n_clips": len(self.clips),
            "media_mapping": dict(self.media_mapping),
            "clips": [c.to_dict() for c in self.clips],
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


__all__ = [
    "MediaClipResult",
    "MediaMaterializationResult",
    "materialize_report_media",
    "media_relpath_for_bout",
    "preview_plan_to_clip_bounds",
]
