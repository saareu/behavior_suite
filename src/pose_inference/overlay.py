"""overlay.mp4 generation for Subsystem 02 pose inference outputs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

DEFAULT_CODEC = "mp4v"
DEFAULT_FPS = 30.0
DEFAULT_NODE_RADIUS = 3
DEFAULT_NODE_THICKNESS = -1


class OverlayGenerationError(RuntimeError):
    """Expected failure while generating overlay.mp4."""


@dataclass(frozen=True)
class OverlaySummary:
    """Machine-readable summary of an overlay generation attempt."""

    status: str
    path: Path
    source: str = "pose.parquet"
    frames_written: int = 0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str = DEFAULT_CODEC
    error: str | None = None
    reason: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": self.status,
            "path": str(self.path.resolve()),
            "source": self.source,
            "frames_written": self.frames_written,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "error": self.error,
            "reason": self.reason,
        }


def _require_columns(columns: object, required: set[str]) -> None:
    present = {str(column) for column in columns}
    missing = sorted(required - present)
    if missing:
        raise OverlayGenerationError(
            "pose.parquet is missing required overlay columns: " + ", ".join(missing)
        )


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _identity_value(row: Mapping[str, Any]) -> str:
    track = row.get("track")
    if track is not None and not isinstance(track, float):
        text = str(track)
        if text and text != "<NA>":
            return f"track:{text}"
    instance_index = row.get("instance_index")
    return f"instance:{instance_index}"


def _identity_color(identity: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return (
        64 + digest[0] % 192,
        64 + digest[1] % 192,
        64 + digest[2] % 192,
    )


def _load_pose_rows_by_frame(pose_parquet_path: Path) -> dict[int, list[dict[str, object]]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise OverlayGenerationError("pandas is required to read pose.parquet.") from exc

    path = Path(pose_parquet_path)
    if not path.is_file():
        raise OverlayGenerationError(f"pose.parquet does not exist: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise OverlayGenerationError(f"Could not read pose.parquet for overlay generation: {exc}") from exc

    _require_columns(frame.columns, {"frame_idx", "instance_index", "x", "y"})
    work = frame.copy()
    work["frame_idx"] = pd.to_numeric(work["frame_idx"], errors="coerce")
    work["instance_index"] = pd.to_numeric(work["instance_index"], errors="coerce")
    work["x"] = pd.to_numeric(work["x"], errors="coerce")
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["frame_idx", "instance_index"])

    rows_by_frame: dict[int, list[dict[str, object]]] = {}
    if work.empty:
        return rows_by_frame
    for frame_idx, group in work.groupby("frame_idx", sort=False):
        numeric_frame_idx = _finite_float(frame_idx)
        if numeric_frame_idx is None:
            continue
        rows_by_frame[int(numeric_frame_idx)] = group.to_dict(orient="records")
    return rows_by_frame


def _draw_pose_rows(
    frame: Any,
    rows: list[dict[str, object]],
    *,
    node_radius: int,
    node_thickness: int,
) -> None:
    for row in rows:
        x = _finite_float(row.get("x"))
        y = _finite_float(row.get("y"))
        if x is None or y is None:
            continue
        identity = _identity_value(row)
        cv2.circle(
            frame,
            (int(round(x)), int(round(y))),
            node_radius,
            _identity_color(identity),
            node_thickness,
            lineType=cv2.LINE_AA,
        )


def _validated_fps(capture: Any, fps_override: float | None) -> float:
    if fps_override is not None and fps_override > 0:
        return float(fps_override)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if math.isfinite(fps) and fps > 0 else DEFAULT_FPS


def generate_overlay_video(
    *,
    prepared_video_path: Path,
    pose_parquet_path: Path,
    output_path: Path,
    fps_override: float | None = None,
    codec: str = DEFAULT_CODEC,
    node_radius: int = DEFAULT_NODE_RADIUS,
    node_thickness: int = DEFAULT_NODE_THICKNESS,
) -> OverlaySummary:
    """Generate one visual-review ``overlay.mp4`` from prepared video and pose.parquet."""

    video_path = Path(prepared_video_path)
    if not video_path.is_file():
        raise OverlayGenerationError(f"prepared_video.mp4 does not exist: {video_path}")
    rows_by_frame = _load_pose_rows_by_frame(Path(pose_parquet_path))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OverlayGenerationError(f"Could not open prepared video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise OverlayGenerationError(f"Prepared video has invalid dimensions: {width}x{height}")
    fps = _validated_fps(capture, fps_override)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OverlayGenerationError(f"Could not open overlay writer: {output}")

    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _draw_pose_rows(
                frame,
                rows_by_frame.get(frames_written, []),
                node_radius=node_radius,
                node_thickness=node_thickness,
            )
            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        writer.release()

    if frames_written == 0:
        raise OverlayGenerationError(f"Prepared video contains no readable frames: {video_path}")
    if not output.is_file() or output.stat().st_size == 0:
        raise OverlayGenerationError(f"Overlay writer did not create a readable output file: {output}")

    return OverlaySummary(
        status="generated",
        path=output,
        source="pose.parquet",
        frames_written=frames_written,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
    )
