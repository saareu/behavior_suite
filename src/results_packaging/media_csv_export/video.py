"""Prepared-video probe and ffmpeg MP4 clip encoding."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from results_packaging.media_csv_export.clip_bounds import ClipBounds
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
    ExportVideoError,
)


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """OpenCV probe of an encode-source video."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    def validated(self) -> VideoProbe:
        if self.fps <= 0:
            raise ExportVideoError(
                f"Video fps is not positive ({self.fps}); cannot establish clip timebase: "
                f"{self.path}"
            )
        if self.n_frames <= 0:
            raise ExportVideoError(
                f"Video frame count is not positive ({self.n_frames}): {self.path}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ExportVideoError(
                f"Video dimensions invalid ({self.width}x{self.height}): {self.path}"
            )
        return self


def resolve_ffmpeg_executable(configured: Path | str | None = None) -> Path:
    """Resolve ffmpeg: configured path → PATH → suite preprocess resolver."""

    if configured is not None:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
        raise ExportVideoError(f"Configured ffmpeg not found: {path}")

    which = shutil.which("ffmpeg")
    if which:
        return Path(which).resolve()

    try:
        from preprocess.ffmpeg_runtime import resolve_ffmpeg_binary

        return Path(resolve_ffmpeg_binary()).resolve()
    except Exception as exc:  # noqa: BLE001 - surface as export dependency error
        raise ExportVideoError(
            "ffmpeg executable not found on PATH and suite resolver failed. "
            f"Detail: {exc}"
        ) from exc


def probe_video(path: Path | str) -> VideoProbe:
    """Probe fps / frame count with OpenCV; fail hard on unreliable metadata."""

    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise ExportVideoError(f"Source video missing: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ExportVideoError(f"Could not open source video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    return VideoProbe(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        n_frames=n_frames,
    ).validated()


_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_path_token(token: str, *, fallback: str = "unknown") -> str:
    """Filesystem-safe single path component (no separators / traversal)."""

    text = str(token).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        text = fallback
    cleaned = _SAFE_TOKEN_RE.sub("_", text).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        cleaned = fallback
    if ".." in cleaned:
        cleaned = fallback
    return cleaned


def clip_relpath_for_bout(
    *,
    channel: str,
    behavior_name: str,
    bout_id: str,
) -> Path:
    """Deterministic relative clip path under ``clips/``."""

    safe_bout = sanitize_path_token(bout_id, fallback="bout")
    # Reject if sanitization changed identity-critical separators into ambiguity
    # by requiring bout_id itself has no path separators (validated upstream).
    return Path(
        sanitize_path_token(channel, fallback="channel"),
        sanitize_path_token(behavior_name, fallback="behavior"),
        f"{safe_bout}.mp4",
    )


def validate_bout_id_for_filename(bout_id: str) -> str:
    text = str(bout_id).strip()
    if not text:
        raise ExportVideoError("Empty bout_id cannot be used as clip filename.")
    if text in {".", ".."} or "/" in text or "\\" in text or ":" in text:
        raise ExportVideoError(
            f"bout_id contains illegal path characters: {bout_id!r}"
        )
    return text


def encode_mp4_clip(
    *,
    source_video: Path,
    output_path: Path,
    bounds: ClipBounds,
    ffmpeg_executable: Path | None = None,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    verify_encoded_boundaries: bool = True,
) -> Path:
    """Encode one clip with ffmpeg; atomic temp → final replace.

    Uses frame-addressed ``trim=start_frame:end_frame`` (inclusive prepared
    indices) rather than time ``-ss`` seeking, which can miss source frames on
    long-GOP inputs. Streams via ffmpeg subprocess (does not load the full
    video into memory). When ``verify_encoded_boundaries`` is True, probes the
    encoded clip and fails if frame count or playback fps diverge from the
    planned bounds — never silently approximates boundaries.
    """

    if bounds.clip_frame_count <= 0:
        raise ExportVideoError("Refusing to encode empty clip.")
    if abs(bounds.source_fps - 0.0) < 1e-12 or bounds.source_fps <= 0:
        raise ExportVideoError("Refusing to encode without positive source_fps.")

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    ffmpeg = resolve_ffmpeg_executable(ffmpeg_executable)
    start = int(bounds.clip_frame_idx_start)
    end = int(bounds.clip_frame_idx_end)
    count = int(bounds.clip_frame_count)
    fps = float(bounds.source_fps)
    # ffmpeg trim end_frame is exclusive; prepared/clip indices are inclusive.
    vf = f"trim=start_frame={start}:end_frame={end + 1},setpts=PTS-STARTPTS"
    # Force mp4 muxer: temp suffix ``.mp4.tmp`` is not a recognized extension.
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(Path(source_video).expanduser().resolve()),
        "-vf",
        vf,
        "-frames:v",
        str(count),
        "-c:v",
        str(codec),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        str(pixel_format),
        "-fps_mode",
        "cfr",
        "-r",
        f"{fps:.9f}",
        "-an",
        "-f",
        "mp4",
        str(tmp),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise ExportVideoError(f"Failed to invoke ffmpeg: {exc}") from exc

    if completed.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        # Prefer the trailing diagnostics (ffmpeg prints banners first).
        snippet = detail[-1200:] if len(detail) > 1200 else detail
        raise ExportVideoError(
            f"ffmpeg encode failed (code={completed.returncode}): {snippet}"
        )

    if verify_encoded_boundaries:
        try:
            probed = probe_encoded_clip(tmp)
        except ExportVideoError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        if int(probed.n_frames) != count:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise ExportVideoError(
                f"Encoded clip frame count {probed.n_frames} does not match "
                f"planned inclusive count {count} "
                f"(frames [{start}, {end}]); refusing silent boundary "
                "approximation."
            )
        if probed.fps <= 0 or abs(probed.fps - fps) / fps > 0.02:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise ExportVideoError(
                f"Encoded clip fps {probed.fps} does not match source fps "
                f"{fps}; refusing silent playback-rate approximation."
            )

    tmp.replace(out)
    return out


def probe_encoded_clip(path: Path | str) -> VideoProbe:
    """Probe an encoded clip for frame-count / fps QC."""

    return probe_video(path)
