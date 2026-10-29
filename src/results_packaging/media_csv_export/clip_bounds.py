"""Prepared-frame clip interval calculation (context pad + video clamp)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.media_csv_export.contracts import ClipBoundsError


@dataclass(frozen=True, slots=True)
class ClipBounds:
    """Inclusive prepared-frame clip window with timing derived from source fps."""

    prepared_frame_idx_start: int
    prepared_frame_idx_end: int
    clip_frame_idx_start: int
    clip_frame_idx_end: int
    clip_frame_count: int
    clip_time_sec_start: float
    clip_time_sec_end: float
    clip_duration_sec: float
    pre_context_sec: float
    post_context_sec: float
    source_fps: float
    source_n_frames: int
    pre_pad_frames: int
    post_pad_frames: int
    clamped_start: bool
    clamped_end: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared_frame_idx_start": self.prepared_frame_idx_start,
            "prepared_frame_idx_end": self.prepared_frame_idx_end,
            "clip_frame_idx_start": self.clip_frame_idx_start,
            "clip_frame_idx_end": self.clip_frame_idx_end,
            "clip_frame_count": self.clip_frame_count,
            "clip_time_sec_start": self.clip_time_sec_start,
            "clip_time_sec_end": self.clip_time_sec_end,
            "clip_duration_sec": self.clip_duration_sec,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "source_fps": self.source_fps,
            "source_n_frames": self.source_n_frames,
            "pre_pad_frames": self.pre_pad_frames,
            "post_pad_frames": self.post_pad_frames,
            "clamped_start": self.clamped_start,
            "clamped_end": self.clamped_end,
        }


def compute_clip_bounds(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
) -> ClipBounds:
    """Map prepared bout span + context onto inclusive clip frames.

    Padding uses ``source_fps`` (prepared video), never the 120 Hz behavioral
    timebase. Behavioral endpoints are not modified by this function.
    """

    if source_fps is None or float(source_fps) <= 0:
        raise ClipBoundsError(
            f"source_fps must be positive for clip timing; got {source_fps!r}."
        )
    if source_n_frames is None or int(source_n_frames) <= 0:
        raise ClipBoundsError(
            f"source_n_frames must be positive; got {source_n_frames!r}."
        )
    if float(pre_context_sec) < 0 or float(post_context_sec) < 0:
        raise ClipBoundsError("Context padding seconds must be >= 0.")

    prep_start = int(prepared_frame_idx_start)
    prep_end = int(prepared_frame_idx_end)
    n_frames = int(source_n_frames)
    fps = float(source_fps)

    if prep_start > prep_end:
        raise ClipBoundsError(
            f"Invalid prepared bout span: start ({prep_start}) > end ({prep_end})."
        )
    if prep_start < 0 or prep_end >= n_frames:
        raise ClipBoundsError(
            f"Prepared bout span [{prep_start}, {prep_end}] outside video "
            f"[0, {n_frames - 1}]."
        )

    pre_pad = int(round(float(pre_context_sec) * fps))
    post_pad = int(round(float(post_context_sec) * fps))
    raw_start = prep_start - pre_pad
    raw_end = prep_end + post_pad
    clip_start = max(0, raw_start)
    clip_end = min(n_frames - 1, raw_end)
    if clip_start > clip_end:
        raise ClipBoundsError(
            f"Clamped clip window empty: start={clip_start}, end={clip_end}."
        )

    frame_count = clip_end - clip_start + 1
    t0 = clip_start / fps
    duration = frame_count / fps
    t1 = t0 + duration  # exclusive-end media time

    return ClipBounds(
        prepared_frame_idx_start=prep_start,
        prepared_frame_idx_end=prep_end,
        clip_frame_idx_start=clip_start,
        clip_frame_idx_end=clip_end,
        clip_frame_count=frame_count,
        clip_time_sec_start=t0,
        clip_time_sec_end=t1,
        clip_duration_sec=duration,
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        source_fps=fps,
        source_n_frames=n_frames,
        pre_pad_frames=pre_pad,
        post_pad_frames=post_pad,
        clamped_start=raw_start < 0,
        clamped_end=raw_end > n_frames - 1,
    )


def compute_clip_bounds_for_bout_row(
    row: pd.Series | dict[str, Any],
    *,
    source_fps: float,
    source_n_frames: int,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
) -> ClipBounds:
    """Convenience wrapper reading prepared endpoints from a bout row."""

    data = row if isinstance(row, dict) else row.to_dict()
    try:
        prep_start = int(data["prepared_frame_idx_start"])
        prep_end = int(data["prepared_frame_idx_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClipBoundsError(
            "Bout row missing valid prepared_frame_idx_start/end."
        ) from exc
    return compute_clip_bounds(
        prepared_frame_idx_start=prep_start,
        prepared_frame_idx_end=prep_end,
        source_fps=source_fps,
        source_n_frames=source_n_frames,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
    )
