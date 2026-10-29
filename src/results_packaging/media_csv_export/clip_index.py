"""Build and update media-export clip_index rows."""

from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.media_csv_export.clip_bounds import (
    ClipBounds,
    compute_clip_bounds_for_bout_row,
)
from results_packaging.media_csv_export.contracts import (
    CLIP_INDEX_COLUMNS,
    CLIP_INDEX_SCHEMA_VERSION,
    DEFAULT_CODEC,
    ENCODE_STATUS_PENDING,
    ClipBoundsError,
)
from results_packaging.media_csv_export.video import (
    clip_relpath_for_bout,
    validate_bout_id_for_filename,
)


def _optional(row: dict[str, Any], key: str) -> Any:
    return row.get(key)


def build_pending_clip_index(
    selected_bouts: pd.DataFrame,
    *,
    export_job_id: str,
    selection_mode: str,
    source_video_kind: str,
    source_video_path: str,
    source_fps: float,
    source_n_frames: int,
    pre_context_sec: float,
    post_context_sec: float,
    crf: int,
    codec: str = DEFAULT_CODEC,
) -> pd.DataFrame:
    """Create one pending clip-index row per selected bout."""

    rows: list[dict[str, Any]] = []
    for rec in selected_bouts.to_dict(orient="records"):
        bout_id = validate_bout_id_for_filename(str(rec["bout_id"]))
        channel = str(rec.get("channel", "unknown"))
        behavior = str(rec.get("behavior_name", "unknown"))
        rel = clip_relpath_for_bout(
            channel=channel, behavior_name=behavior, bout_id=bout_id
        )
        error: str | None = None
        encode_status = ENCODE_STATUS_PENDING
        bounds: ClipBounds | None
        try:
            bounds = compute_clip_bounds_for_bout_row(
                rec,
                source_fps=source_fps,
                source_n_frames=source_n_frames,
                pre_context_sec=pre_context_sec,
                post_context_sec=post_context_sec,
            )
        except ClipBoundsError as exc:
            bounds = None
            encode_status = "omitted_invalid_bounds"
            error = str(exc)

        rows.append(
            {
                "bout_id": bout_id,
                "source_s8_bout_id": _optional(rec, "source_s8_bout_id"),
                "package_run_id": _optional(rec, "package_run_id"),
                "source_s8_run_id": _optional(rec, "source_s8_run_id"),
                "channel": channel,
                "behavior_name": behavior,
                "identity_grain": _optional(rec, "identity_grain"),
                "focal_id": _optional(rec, "focal_id"),
                "partner_id": _optional(rec, "partner_id"),
                "mouse_a_id": _optional(rec, "mouse_a_id"),
                "mouse_b_id": _optional(rec, "mouse_b_id"),
                "pair_id": _optional(rec, "pair_id"),
                "start_sample_idx": _optional(rec, "start_sample_idx"),
                "end_sample_idx": _optional(rec, "end_sample_idx"),
                "start_time_sec": _optional(rec, "start_time_sec"),
                "end_time_sec": _optional(rec, "end_time_sec"),
                "bout_duration_frames": _optional(rec, "bout_duration_frames"),
                "bout_duration_sec": _optional(rec, "bout_duration_sec"),
                "prepared_frame_idx_start": _optional(rec, "prepared_frame_idx_start"),
                "prepared_frame_idx_end": _optional(rec, "prepared_frame_idx_end"),
                "prepared_time_sec_start": _optional(rec, "prepared_time_sec_start"),
                "prepared_time_sec_end": _optional(rec, "prepared_time_sec_end"),
                "clip_frame_idx_start": (
                    bounds.clip_frame_idx_start if bounds is not None else None
                ),
                "clip_frame_idx_end": (
                    bounds.clip_frame_idx_end if bounds is not None else None
                ),
                "clip_frame_count": (
                    bounds.clip_frame_count if bounds is not None else None
                ),
                "clip_time_sec_start": (
                    bounds.clip_time_sec_start if bounds is not None else None
                ),
                "clip_time_sec_end": (
                    bounds.clip_time_sec_end if bounds is not None else None
                ),
                "clip_duration_sec": (
                    bounds.clip_duration_sec if bounds is not None else None
                ),
                "pre_context_sec": float(pre_context_sec),
                "post_context_sec": float(post_context_sec),
                "source_video_kind": source_video_kind,
                "source_video_path": source_video_path,
                "source_fps": float(source_fps),
                "source_n_frames": int(source_n_frames),
                "clip_relpath": str(rel).replace("\\", "/"),
                "clip_exists": False,
                "encode_status": encode_status,
                "error": error,
                "export_job_id": export_job_id,
                "selection_mode": selection_mode,
                "crf": int(crf),
                "codec": str(codec),
                "schema_version": CLIP_INDEX_SCHEMA_VERSION,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(CLIP_INDEX_COLUMNS))
    return pd.DataFrame(rows)[list(CLIP_INDEX_COLUMNS)]


def clip_bounds_from_index_row(row: pd.Series | dict[str, Any]) -> ClipBounds:
    data = row if isinstance(row, dict) else row.to_dict()
    return ClipBounds(
        prepared_frame_idx_start=int(data["prepared_frame_idx_start"]),
        prepared_frame_idx_end=int(data["prepared_frame_idx_end"]),
        clip_frame_idx_start=int(data["clip_frame_idx_start"]),
        clip_frame_idx_end=int(data["clip_frame_idx_end"]),
        clip_frame_count=int(data["clip_frame_count"]),
        clip_time_sec_start=float(data["clip_time_sec_start"]),
        clip_time_sec_end=float(data["clip_time_sec_end"]),
        clip_duration_sec=float(data["clip_duration_sec"]),
        pre_context_sec=float(data["pre_context_sec"]),
        post_context_sec=float(data["post_context_sec"]),
        source_fps=float(data["source_fps"]),
        source_n_frames=int(data["source_n_frames"]),
        pre_pad_frames=0,
        post_pad_frames=0,
        clamped_start=False,
        clamped_end=False,
    )
