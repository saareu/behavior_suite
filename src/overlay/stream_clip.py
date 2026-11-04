"""Shared prepared-rate compose→ffmpeg streaming helper for overlay clips.

Reuses the established compositor and ``overlay.export`` ffmpeg pipe. Does not
implement a second compositor or encoder. Used by C3-5 S9 overlay media export
and available to other bout/interval exporters.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from behavioral_review.sync import SyncMap
from overlay import export as _shared
from overlay.adapters.base import OverlayStageAdapter
from overlay.compositor import OverlayCompositor
from overlay.contracts import (
    FrameDomain,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    PlateauPolicy,
)
from overlay.evidence import collect_evidence
from overlay.plan import PreparedFrameInterval
from overlay.style import DrawConfig
from overlay.timing import PreparedFrameTiming
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
)
from results_packaging.media_csv_export.video import probe_encoded_clip

ProgressCallback = Callable[[dict[str, Any]], None]
CancellationCallback = Callable[[], bool]
LayerTimingResolver = Callable[[int], Mapping[LayerId, PreparedFrameTiming] | None]


@dataclass(frozen=True, slots=True)
class StreamComposedClipResult:
    """Result of streaming one composed prepared-rate clip to disk."""

    output_path: Path
    frames_written: int
    output_fps: float
    prepared_interval: PreparedFrameInterval
    unavailable_layers: tuple[str, ...]
    first_timing: PreparedFrameTiming | None = None
    last_timing: PreparedFrameTiming | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StreamComposedClipRequest:
    """Inputs for one prepared-interval composed encode."""

    prepared_video_path: Path
    sync: SyncMap
    adapters: Mapping[LayerId, OverlayStageAdapter]
    prepared_interval: PreparedFrameInterval
    selected_layers: tuple[LayerId, ...]
    style: DrawConfig
    output_path: Path
    fps: float
    width: int
    height: int
    layer_timing_resolver: LayerTimingResolver | None = None
    ffmpeg_executable: Path | str | None = None
    crf: int = DEFAULT_CRF
    codec: str = DEFAULT_CODEC
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    verify_encoded_boundaries: bool = True
    progress_callback: ProgressCallback | None = None
    cancellation_requested: CancellationCallback | None = None
    progress_extra: Mapping[str, Any] = field(default_factory=dict)


def stream_composed_prepared_clip(
    request: StreamComposedClipRequest,
) -> StreamComposedClipResult:
    """Decode prepared frames, compose overlays, and pipe to libx264.

    Output path is written atomically via ``*.tmp`` then ``replace``.
    """

    interval = request.prepared_interval
    expected = int(interval.frame_count)
    if expected <= 0:
        raise OverlayExportError(
            f"Prepared interval has non-positive frame count: {interval.to_dict()}."
        )
    out_final = Path(request.output_path).expanduser().resolve()
    out_tmp = out_final.with_suffix(out_final.suffix + ".tmp")
    if out_final.exists():
        raise OverlayExportError(
            f"Refusing to overwrite existing overlay clip: {out_final}"
        )
    if out_tmp.exists():
        out_tmp.unlink()
    out_final.parent.mkdir(parents=True, exist_ok=True)

    compositor = OverlayCompositor(
        frame_domain=FrameDomain.PREPARED,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
    )
    unavailable: set[str] = set()
    notes: list[str] = []
    frames_written = 0
    first_timing: PreparedFrameTiming | None = None
    last_timing: PreparedFrameTiming | None = None

    _shared._emit(
        request.progress_callback,
        {
            "stage": "encode_start",
            "export_kind": "stream_composed_prepared_clip",
            "output_path": str(out_final),
            "prepared_frame_idx_start": interval.start,
            "prepared_frame_idx_end": interval.end,
            "expected_frames": expected,
            "fps": float(request.fps),
            **dict(request.progress_extra),
        },
    )

    pipe: _shared._FfmpegPipeSession | None = None
    capture: cv2.VideoCapture | None = None
    try:
        _shared._check_cancel(request.cancellation_requested)
        capture = _shared._open_capture_at(
            Path(request.prepared_video_path),
            prepared_frame_idx=interval.start,
        )
        pipe = _shared._start_ffmpeg_pipe(
            output_path=out_tmp,
            width=int(request.width),
            height=int(request.height),
            fps=float(request.fps),
            ffmpeg_executable=request.ffmpeg_executable,
            crf=int(request.crf),
            codec=str(request.codec),
            pixel_format=str(request.pixel_format),
        )
        assert pipe.proc.stdin is not None

        for prep in range(interval.start, interval.end + 1):
            _shared._check_cancel(request.cancellation_requested)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OverlayExportError(
                    f"Failed to decode prepared_frame_idx={prep} from "
                    f"{request.prepared_video_path}."
                )
            if frame.shape[1] != int(request.width) or frame.shape[0] != int(
                request.height
            ):
                raise OverlayExportError(
                    f"Decoded frame size mismatch at prepared_frame_idx={prep}: "
                    f"got {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {request.width}x{request.height}."
                )

            layer_timing = None
            if request.layer_timing_resolver is not None:
                layer_timing = request.layer_timing_resolver(int(prep))
            evidence = collect_evidence(
                request.sync,
                int(prep),
                dict(request.adapters),
                selected_layers=request.selected_layers,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
                layer_timing=layer_timing,
            )
            composed = compositor.render_frame(
                frame,
                prepared_frame_idx=int(prep),
                evidence=evidence,
                selected_layers=request.selected_layers,
                style=request.style,
            )
            unavailable.update(x.value for x in composed.unavailable_layers)
            if first_timing is None:
                first_timing = evidence.timing
            last_timing = evidence.timing

            raw = np.ascontiguousarray(composed.frame_bgr)
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8, copy=False)
            try:
                pipe.proc.stdin.write(raw.tobytes())
            except BrokenPipeError as exc:
                detail = _shared._ffmpeg_stderr(pipe)
                raise OverlayExportError(
                    f"ffmpeg pipe broke while writing prepared_frame_idx={prep}: "
                    f"{detail}"
                ) from exc
            frames_written += 1
            if (
                frames_written == 1
                or frames_written == expected
                or frames_written % 30 == 0
            ):
                _shared._emit(
                    request.progress_callback,
                    {
                        "stage": "encoding",
                        "frames_written": frames_written,
                        "expected_frames": expected,
                        "prepared_frame_idx": prep,
                        "dense_sample_idx": evidence.timing.dense_sample_idx,
                        **dict(request.progress_extra),
                    },
                )

        pipe.proc.stdin.close()
        rc = pipe.proc.wait(timeout=120)
        if rc != 0 or not out_tmp.is_file() or out_tmp.stat().st_size <= 0:
            detail = _shared._ffmpeg_stderr(pipe)
            raise OverlayExportError(
                f"ffmpeg encode failed (code={rc}): {detail[-1200:]}"
            )
        if frames_written != expected:
            raise OverlayExportError(
                f"Wrote {frames_written} frames but planned inclusive count is {expected}."
            )
        if request.verify_encoded_boundaries:
            probed = probe_encoded_clip(out_tmp)
            if int(probed.n_frames) != expected:
                raise OverlayExportError(
                    f"Encoded clip frame count {probed.n_frames} does not match "
                    f"planned inclusive count {expected}."
                )
            fps = float(request.fps)
            if probed.fps <= 0 or abs(probed.fps - fps) / fps > 0.02:
                raise OverlayExportError(
                    f"Encoded clip fps {probed.fps} does not match source fps {fps}."
                )

        out_tmp.replace(out_final)
        _shared._emit(
            request.progress_callback,
            {
                "stage": "clip_published",
                "export_kind": "stream_composed_prepared_clip",
                "output_path": str(out_final),
                "frames_written": frames_written,
                **dict(request.progress_extra),
            },
        )
        return StreamComposedClipResult(
            output_path=out_final,
            frames_written=frames_written,
            output_fps=float(request.fps),
            prepared_interval=interval,
            unavailable_layers=tuple(sorted(unavailable)),
            first_timing=first_timing,
            last_timing=last_timing,
            notes=tuple(notes),
        )
    except OverlayExportCancelled:
        _cleanup_failed_stream(pipe=pipe, tmp_path=out_tmp)
        raise
    except Exception:
        _cleanup_failed_stream(pipe=pipe, tmp_path=out_tmp)
        raise
    finally:
        if capture is not None:
            capture.release()


def _cleanup_failed_stream(
    *,
    pipe: _shared._FfmpegPipeSession | None,
    tmp_path: Path,
) -> None:
    if pipe is not None:
        with contextlib.suppress(Exception):
            if pipe.proc.stdin is not None and not pipe.proc.stdin.closed:
                pipe.proc.stdin.close()
        with contextlib.suppress(Exception):
            pipe.proc.kill()
        with contextlib.suppress(Exception):
            pipe.proc.wait(timeout=5)
    with contextlib.suppress(OSError):
        if tmp_path.exists():
            tmp_path.unlink()


__all__ = [
    "CancellationCallback",
    "LayerTimingResolver",
    "ProgressCallback",
    "StreamComposedClipRequest",
    "StreamComposedClipResult",
    "stream_composed_prepared_clip",
]
