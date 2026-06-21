"""Independent ffprobe and OpenCV metadata collection for raw videos."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

from preprocess.exceptions import VideoProbeError
from preprocess.models import VideoProbeResult

_FFPROBE_TIMEOUT_SECONDS = 30


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


def probe_video_ffprobe(path: Path) -> dict[str, Any]:
    """Return ffprobe JSON metadata for the first video stream.

    Raises:
        VideoProbeError: If ffprobe is unavailable, fails, times out, or emits
            invalid JSON. Callers may treat this as non-fatal when OpenCV can
            read the video.
    """

    video_path = _validate_video_path(path)
    ffprobe_binary = shutil.which("ffprobe")
    if ffprobe_binary is None:
        raise VideoProbeError("ffprobe executable was not found.")

    command = [
        ffprobe_binary,
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


def count_opencv_readable_frames(path: Path) -> int:
    """Sequentially decode a video from the first frame and return the count.

    Raises:
        VideoProbeError: If OpenCV cannot open the video or decode any frame.
    """

    capture = _open_video(path)
    readable_count = 0
    try:
        while True:
            success, _frame = capture.read()
            if not success:
                break
            readable_count += 1
    finally:
        capture.release()
    if readable_count == 0:
        raise VideoProbeError(f"OpenCV could not decode any frames from video: {path}")
    return readable_count


def probe_video(path: Path, require_sequential_count: bool) -> VideoProbeResult:
    """Probe a readable video with OpenCV and ffprobe when available.

    OpenCV readability is mandatory. ffprobe failure is captured in the result
    so a readable video remains probeable. A full sequential decode occurs only
    when require_sequential_count is true.

    Raises:
        VideoProbeError: If the file is absent, OpenCV cannot open it, or no
            frame can be decoded.
    """

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
    readable_count = (
        count_opencv_readable_frames(video_path) if require_sequential_count else None
    )

    ffprobe_metadata: dict[str, Any] | None = None
    ffprobe_error: str | None = None
    try:
        ffprobe_metadata = probe_video_ffprobe(video_path)
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

    return VideoProbeResult(
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
        opencv_fps=opencv_fps,
        raw_fps_effective=raw_fps_effective,
        raw_fps_effective_method=raw_fps_method,
        pts_status="not_extracted",
        ffprobe_succeeded=ffprobe_metadata is not None,
        ffprobe_error=ffprobe_error,
        ffprobe_metadata=ffprobe_metadata,
    )
