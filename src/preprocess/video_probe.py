"""Independent ffprobe and OpenCV metadata collection for raw videos."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from preprocess.exceptions import FFmpegRuntimeError, VideoProbeCancelledError, VideoProbeError
from preprocess.ffmpeg_runtime import resolve_ffprobe_binary
from preprocess.models import (
    OperationProgress,
    RawReadableCountProvenance,
    RawVideoFingerprint,
    VideoProbeResult,
)
from preprocess.raw_probe_cache import (
    build_raw_probe_cache_record,
    fingerprint_raw_video,
    load_matching_raw_probe_cache,
    write_raw_probe_cache_atomic,
)

_FFPROBE_TIMEOUT_SECONDS = 30
_COUNT_PROGRESS_FRAME_INTERVAL = 250
_COUNT_PROGRESS_TIME_INTERVAL_SECONDS = 0.25
_RAW_FRAME_FALLBACK_MAX_DECODE_FRAMES = 100_000
_LOGGER = logging.getLogger(__name__)


def _validate_video_path(path: Path) -> Path:
    video_path = Path(path).expanduser()
    if not video_path.is_file():
        raise VideoProbeError(f"Video file does not exist: {video_path}")
    return video_path.resolve()


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _nonnegative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _frame_count(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _rational_to_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" not in text:
        return _positive_float(text)
    numerator_text, denominator_text = text.split("/", maxsplit=1)
    numerator = _positive_float(numerator_text)
    denominator = _positive_float(denominator_text)
    if numerator is None or denominator is None:
        return None
    return numerator / denominator


def probe_video_ffprobe(path: Path, *, ffprobe_path: Path | None = None) -> dict[str, Any]:
    """Return ffprobe JSON metadata for the first video stream.

    Raises:
        VideoProbeError: If ffprobe is unavailable, fails, times out, or emits
            invalid JSON. Callers may treat this as non-fatal when OpenCV can
            read the video.
    """

    video_path = _validate_video_path(path)
    try:
        ffprobe_binary = resolve_ffprobe_binary(ffprobe_path)
    except FFmpegRuntimeError as exc:
        raise VideoProbeError(str(exc)) from exc

    command = [
        str(ffprobe_binary),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=width,height,codec_name,codec_tag_string,pix_fmt,"
            "avg_frame_rate,r_frame_rate,time_base,nb_frames"
        ),
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoProbeError(f"ffprobe could not probe {video_path}: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no diagnostic output"
        raise VideoProbeError(
            f"ffprobe failed for {video_path} with exit code "
            f"{completed.returncode}: {detail}"
        )
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"ffprobe emitted invalid JSON for {video_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise VideoProbeError(f"ffprobe emitted a non-object JSON result for {video_path}.")
    return metadata


def _open_video(path: Path) -> cv2.VideoCapture:
    video_path = _validate_video_path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProbeError(f"OpenCV could not open video: {video_path}")
    return capture


def get_opencv_reported_frame_count(path: Path) -> int:
    """Return OpenCV's container-reported frame count.

    A zero result means OpenCV did not report a usable count; it is kept
    distinct from a sequential readable-frame count.
    """

    capture = _open_video(path)
    try:
        value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(value) or value <= 0:
        return 0
    return int(value)


def count_opencv_readable_frames(
    path: Path,
    *,
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[OperationProgress], None] | None = None,
) -> int:
    """Sequentially decode a video from the first frame and return the count.

    Raises:
        VideoProbeError: If OpenCV cannot open the video or decode any frame.
    """

    capture = _open_video(path)
    readable_count = 0
    reported_total: int | None = None
    get_property = getattr(capture, "get", None)
    if callable(get_property):
        try:
            reported_value = float(get_property(cv2.CAP_PROP_FRAME_COUNT))
        except (TypeError, ValueError, OverflowError, cv2.error):
            reported_value = 0.0
        if math.isfinite(reported_value) and reported_value > 0:
            parsed_total = int(reported_value)
            reported_total = parsed_total if parsed_total > 0 else None
    last_emitted_count = 0
    last_emitted_at = time.perf_counter()

    def emit(phase: str, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(
            OperationProgress(
                phase=phase,
                message=message,
                completed_units=readable_count,
                total_units=reported_total,
                total_is_estimate=reported_total is not None,
                is_indeterminate=reported_total is None,
            )
        )

    emit("counting", "Counting sequentially readable frames")
    try:
        while True:
            if cancellation_requested is not None and cancellation_requested():
                emit("cancelling", "Cancellation requested; stopping frame count")
                raise VideoProbeCancelledError(
                    f"Raw readable-frame count was cancelled for video: {path}"
                )
            success, _frame = capture.read()
            if not success:
                break
            readable_count += 1
            now = time.perf_counter()
            if (
                readable_count - last_emitted_count
                >= _COUNT_PROGRESS_FRAME_INTERVAL
                or now - last_emitted_at
                >= _COUNT_PROGRESS_TIME_INTERVAL_SECONDS
            ):
                emit("counting", "Counting sequentially readable frames")
                last_emitted_count = readable_count
                last_emitted_at = now
    finally:
        capture.release()
    if readable_count == 0:
        raise VideoProbeError(f"OpenCV could not decode any frames from video: {path}")
    emit("complete", "Sequential readable-frame count complete")
    return readable_count


def _opencv_frame_position(capture: cv2.VideoCapture) -> float | None:
    try:
        value = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    except (TypeError, ValueError, OverflowError, cv2.error):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _position_matches(actual: float | None, expected: int) -> bool:
    return actual is not None and abs(actual - expected) <= 0.5


def _read_raw_frame_by_bounded_sequential_decode(
    path: Path,
    raw_decode_frame_idx: int,
    *,
    max_fallback_decode_frames: int,
    direct_seek_diagnostic: str,
) -> np.ndarray:
    if raw_decode_frame_idx > max_fallback_decode_frames:
        raise VideoProbeError(
            "OpenCV could not verify exact direct seek to raw decode frame "
            f"{raw_decode_frame_idx:,} ({direct_seek_diagnostic}); bounded "
            "fallback decoding is limited to "
            f"{max_fallback_decode_frames:,} frames."
        )
    capture = _open_video(path)
    try:
        for decoded_index in range(raw_decode_frame_idx + 1):
            success, frame = capture.read()
            if not success or frame is None:
                raise VideoProbeError(
                    "Raw decode frame "
                    f"{raw_decode_frame_idx:,} is unreadable; reached the end "
                    f"before frame {decoded_index:,}."
                )
    finally:
        capture.release()
    return frame


def read_raw_frame_at_index(
    video_path: Path,
    raw_decode_frame_idx: int,
    *,
    max_fallback_decode_frames: int = _RAW_FRAME_FALLBACK_MAX_DECODE_FRAMES,
) -> np.ndarray:
    """Return the exact requested raw decode-order frame as a copied array.

    The requested index is zero-based raw sequential decode order. Raw PTS and
    display timestamps are never used as identity.

    OpenCV random seeking is backend/container dependent: ``set(POS_FRAMES, n)``
    may seek to a keyframe or report imprecise position information. This helper
    first attempts direct seeking for responsiveness and returns that frame only
    when OpenCV's frame-position reporting confirms the requested index as far
    as the backend permits. If direct seeking cannot be verified, it falls back
    to bounded sequential decoding from frame zero, which defines raw decode
    identity. If neither path can prove the requested frame, a ``VideoProbeError``
    is raised; a neighboring frame is never returned silently.

    Args:
        video_path: Raw video path.
        raw_decode_frame_idx: Zero-based raw decode-frame index to read.
        max_fallback_decode_frames: Maximum target index allowed for the exact
            sequential fallback when direct seeking is unavailable or
            unverifiable.

    Raises:
        VideoProbeError: If the source cannot be opened, the index is invalid,
            direct seeking is unverifiable beyond the fallback bound, or the
            requested frame cannot be decoded.
    """

    if (
        isinstance(raw_decode_frame_idx, bool)
        or not isinstance(raw_decode_frame_idx, int)
        or raw_decode_frame_idx < 0
    ):
        raise VideoProbeError("Raw decode frame index must be a non-negative integer.")
    if (
        isinstance(max_fallback_decode_frames, bool)
        or not isinstance(max_fallback_decode_frames, int)
        or max_fallback_decode_frames < 0
    ):
        raise VideoProbeError("Fallback decode bound must be a non-negative integer.")

    path = _validate_video_path(video_path)
    if raw_decode_frame_idx == 0:
        capture = _open_video(path)
        try:
            success, frame = capture.read()
        finally:
            capture.release()
        if not success or frame is None:
            raise VideoProbeError("Raw decode frame 0 is unreadable.")
        return frame.copy()

    direct_seek_diagnostic = "direct seek was not attempted"
    capture = _open_video(path)
    try:
        try:
            seek_succeeded = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, raw_decode_frame_idx))
        except cv2.error as exc:
            seek_succeeded = False
            direct_seek_diagnostic = f"OpenCV seek failed: {exc}"
        if seek_succeeded:
            position_before = _opencv_frame_position(capture)
            success, frame = capture.read()
            position_after = _opencv_frame_position(capture)
            if (
                success
                and frame is not None
                and _position_matches(position_before, raw_decode_frame_idx)
                and _position_matches(position_after, raw_decode_frame_idx + 1)
            ):
                return frame.copy()
            if not success or frame is None:
                direct_seek_diagnostic = "direct seek returned no readable frame"
            elif not _position_matches(position_before, raw_decode_frame_idx):
                direct_seek_diagnostic = (
                    "direct seek reported frame position "
                    f"{position_before!r} before read"
                )
            else:
                direct_seek_diagnostic = (
                    "direct seek reported frame position "
                    f"{position_after!r} after read"
                )
        else:
            direct_seek_diagnostic = "OpenCV did not accept CAP_PROP_POS_FRAMES seek"
    finally:
        capture.release()

    frame = _read_raw_frame_by_bounded_sequential_decode(
        path,
        raw_decode_frame_idx,
        max_fallback_decode_frames=max_fallback_decode_frames,
        direct_seek_diagnostic=direct_seek_diagnostic,
    )
    return frame.copy()


def probe_video(
    path: Path,
    require_sequential_count: bool,
    *,
    cancellation_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[OperationProgress], None] | None = None,
    cache_path: Path | None = None,
    force_recount: bool = False,
    ffprobe_path: Path | None = None,
) -> VideoProbeResult:
    """Probe a readable video with OpenCV and ffprobe when available.

    OpenCV readability is mandatory. ffprobe failure is captured in the result
    so a readable video remains probeable. A full sequential decode occurs only
    when require_sequential_count is true. A project-local cache is consulted only
    when its exact fingerprint matches; force_recount bypasses that cache.

    Raises:
        VideoProbeError: If the file is absent, OpenCV cannot open it, or no
            frame can be decoded.
    """

    if cancellation_requested is not None and cancellation_requested():
        raise VideoProbeCancelledError(f"Raw-video probe was cancelled for: {path}")
    if force_recount and not require_sequential_count:
        raise ValueError("force_recount requires a sequential readable-frame count.")
    video_path = _validate_video_path(path)
    capture = _open_video(video_path)
    try:
        reported_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        opencv_fps = _positive_float(capture.get(cv2.CAP_PROP_FPS))
        success, first_frame = capture.read()
    finally:
        capture.release()
    if not success or first_frame is None:
        raise VideoProbeError(f"OpenCV could not decode the first frame of video: {video_path}")

    height, width = first_frame.shape[:2]
    reported_count = (
        int(reported_count_value)
        if math.isfinite(reported_count_value) and reported_count_value > 0
        else 0
    )
    fingerprint_before_count: RawVideoFingerprint | None = None
    cached_count: int | None = None
    if cache_path is not None:
        try:
            fingerprint_before_count = fingerprint_raw_video(video_path)
            if not force_recount:
                cached = load_matching_raw_probe_cache(
                    cache_path,
                    fingerprint_before_count,
                )
                if cached is not None:
                    cached_count = cached.frame_count_opencv_readable
        except (OSError, ValueError):
            _LOGGER.debug(
                "raw probe cache fingerprint/read failed for %s",
                video_path,
                exc_info=True,
            )

    performed_sequential_count = require_sequential_count and cached_count is None
    readable_count = cached_count
    count_provenance = (
        RawReadableCountProvenance.VALIDATED_REUSABLE_CACHE
        if cached_count is not None
        else None
    )
    if performed_sequential_count:
        readable_count = count_opencv_readable_frames(
            video_path,
            cancellation_requested=cancellation_requested,
            progress_callback=progress_callback,
        )
        count_provenance = RawReadableCountProvenance.VERIFIED_CURRENT_SESSION

    ffprobe_metadata: dict[str, Any] | None = None
    ffprobe_error: str | None = None
    try:
        ffprobe_metadata = probe_video_ffprobe(video_path, ffprobe_path=ffprobe_path)
    except VideoProbeError as exc:
        ffprobe_error = str(exc)

    streams: list[Any] = []
    format_metadata: dict[str, Any] = {}
    if ffprobe_metadata is not None:
        raw_streams = ffprobe_metadata.get("streams", [])
        if isinstance(raw_streams, list):
            streams = raw_streams
        raw_format = ffprobe_metadata.get("format", {})
        if isinstance(raw_format, dict):
            format_metadata = raw_format
    stream = streams[0] if streams and isinstance(streams[0], dict) else {}

    avg_frame_rate = (
        str(stream["avg_frame_rate"]) if stream.get("avg_frame_rate") is not None else None
    )
    r_frame_rate = (
        str(stream["r_frame_rate"]) if stream.get("r_frame_rate") is not None else None
    )
    avg_fps = _rational_to_positive_float(avg_frame_rate)
    r_fps = _rational_to_positive_float(r_frame_rate)
    if avg_fps is not None:
        raw_fps_effective = avg_fps
        raw_fps_method = "ffprobe_avg_frame_rate"
    elif r_fps is not None:
        raw_fps_effective = r_fps
        raw_fps_method = "ffprobe_r_frame_rate"
    elif opencv_fps is not None:
        raw_fps_effective = opencv_fps
        raw_fps_method = "opencv_fps"
    else:
        raw_fps_effective = None
        raw_fps_method = None

    result = VideoProbeResult(
        source_path=video_path,
        width=int(width),
        height=int(height),
        codec=stream.get("codec_name") or stream.get("codec_tag_string"),
        pixel_format=stream.get("pix_fmt"),
        duration_sec=_nonnegative_float(format_metadata.get("duration")),
        avg_frame_rate=avg_frame_rate,
        r_frame_rate=r_frame_rate,
        time_base=str(stream["time_base"]) if stream.get("time_base") is not None else None,
        frame_count_ffprobe=_frame_count(stream.get("nb_frames")),
        frame_count_opencv_reported=reported_count,
        frame_count_opencv_readable=readable_count,
        raw_readable_count_provenance=count_provenance,
        opencv_fps=opencv_fps,
        raw_fps_effective=raw_fps_effective,
        raw_fps_effective_method=raw_fps_method,
        pts_status="not_extracted",
        ffprobe_succeeded=ffprobe_metadata is not None,
        ffprobe_error=ffprobe_error,
        ffprobe_metadata=ffprobe_metadata,
    )
    if (
        cache_path is not None
        and performed_sequential_count
        and fingerprint_before_count is not None
    ):
        try:
            fingerprint_after_count = fingerprint_raw_video(video_path)
            if fingerprint_after_count == fingerprint_before_count:
                record = build_raw_probe_cache_record(fingerprint_after_count, result)
                write_raw_probe_cache_atomic(cache_path, record)
            else:
                _LOGGER.debug(
                    "raw video changed during sequential count; cache was not written: %s",
                    video_path,
                )
        except (OSError, ValueError):
            _LOGGER.debug(
                "raw probe cache write failed for %s",
                video_path,
                exc_info=True,
            )
    return result
