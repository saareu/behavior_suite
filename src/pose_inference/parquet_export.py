"""pose.parquet v1 export for Subsystem 02.

The public export function lazily imports SLEAP and pandas dependencies so the
dry-run runner and most tests do not require SLEAP-NN, a GPU, or a Parquet
engine.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSE_PARQUET_SCHEMA_VERSION = "pose_parquet_v1"
TIMING_VECTOR_KEYS = (
    "time_sec",
    "times_sec",
    "timestamps_sec",
    "prepared_time_sec",
    "frame_time_sec",
    "frame_times_sec",
    "pts_sec",
    "pts",
)
FPS_KEYS = (
    "prepared_video_fps",
    "prepared_fps",
    "fps",
    "fps_header",
    "output_fps",
    "opencv_reported_fps",
)
PARQUET_COLUMNS = (
    "schema_version",
    "video_index",
    "video_path",
    "frame_idx",
    "time_sec",
    "timestamp_source",
    "prepared_frame_idx",
    "raw_decode_frame_idx",
    "prepared_time_sec",
    "raw_pts_time_sec",
    "external_time_sec",
    "raw_pts_status",
    "external_time_status",
    "external_time_source",
    "external_time_variable_name",
    "external_time_units",
    "instance_index",
    "track",
    "node",
    "x",
    "y",
    "score",
    "instance_score",
    "frame_valid",
    "original_frame_idx",
    "source_video_frame_idx",
    "video_fps",
    "fps_header",
    "raw_fps_effective",
    "sync_frame_count_used_for_sleap",
    "n_instances_in_frame",
    "n_valid_keypoints_in_instance",
)


class ParquetExportError(RuntimeError):
    """Expected failure while exporting pose.parquet."""


@dataclass(frozen=True)
class TimingLookup:
    """Resolved Subsystem 01 timing and frame mapping for pose.parquet export."""

    source: str
    status: str = "unavailable"
    times_sec: tuple[float, ...] | None = None
    fps: float | None = None
    sync_schema_version: str | None = None
    prepared_frame_idx: tuple[int, ...] | None = None
    raw_decode_frame_idx: tuple[int, ...] | None = None
    prepared_time_sec: tuple[float, ...] | None = None
    raw_pts_time_sec: tuple[float, ...] | None = None
    external_time_sec: tuple[float, ...] | None = None
    raw_pts_status: str | None = None
    external_time_status: str | None = None
    external_time_source: str | None = None
    external_time_variable_name: str | None = None
    external_time_units: str | None = None
    fps_header: float | None = None
    raw_fps_effective: float | None = None
    start_frame: int | None = None
    end_frame_exclusive: int | None = None
    raw_frame_count_opencv_readable: int | None = None
    prepared_frame_count_opencv_reported: int | None = None
    prepared_frame_count_opencv_readable: int | None = None
    frame_count_used_for_sleap: int | None = None

    def _float_at(self, values: tuple[float, ...] | None, frame_idx: int) -> float | None:
        if values is None or frame_idx < 0 or frame_idx >= len(values):
            return None
        value = values[frame_idx]
        return value if math.isfinite(value) else None

    def _int_at(self, values: tuple[int, ...] | None, frame_idx: int) -> int | None:
        if values is None or frame_idx < 0 or frame_idx >= len(values):
            return None
        return int(values[frame_idx])

    def timestamp_for_frame(self, frame_idx: int) -> tuple[float | None, str]:
        external_time = self._float_at(self.external_time_sec, frame_idx)
        if self.external_time_status == "valid" and external_time is not None:
            return external_time, "prepared_sync:external_time_sec"

        raw_pts_time = self._float_at(self.raw_pts_time_sec, frame_idx)
        if raw_pts_time is not None:
            return raw_pts_time, "prepared_sync:raw_pts_time_sec"

        prepared_time = self._float_at(self.prepared_time_sec, frame_idx)
        if prepared_time is not None:
            return prepared_time, "prepared_sync:prepared_time_sec"

        generic_time = self._float_at(self.times_sec, frame_idx)
        if generic_time is not None:
            return generic_time, self.source

        if self.fps is not None and self.fps > 0:
            return frame_idx / self.fps, "prepare_meta_fps_fallback"
        return None, "unavailable"

    def time_for_frame(self, frame_idx: int) -> float | None:
        return self.timestamp_for_frame(frame_idx)[0]

    def fields_for_frame(self, frame_idx: int) -> dict[str, object]:
        time_sec, timestamp_source = self.timestamp_for_frame(frame_idx)
        return {
            "time_sec": time_sec,
            "timestamp_source": timestamp_source,
            "prepared_frame_idx": self._int_at(self.prepared_frame_idx, frame_idx),
            "raw_decode_frame_idx": self._int_at(self.raw_decode_frame_idx, frame_idx),
            "prepared_time_sec": self._float_at(self.prepared_time_sec, frame_idx),
            "raw_pts_time_sec": self._float_at(self.raw_pts_time_sec, frame_idx),
            "external_time_sec": self._float_at(self.external_time_sec, frame_idx),
            "raw_pts_status": self.raw_pts_status,
            "external_time_status": self.external_time_status,
            "external_time_source": self.external_time_source,
            "external_time_variable_name": self.external_time_variable_name,
            "external_time_units": self.external_time_units,
            "fps_header": self.fps_header,
            "raw_fps_effective": self.raw_fps_effective,
            "sync_frame_count_used_for_sleap": self.frame_count_used_for_sleap,
        }

    def to_metadata(self, frame_indices: Sequence[int] = ()) -> dict[str, object]:
        if frame_indices:
            timestamp_sources = sorted(
                {self.timestamp_for_frame(frame_idx)[1] for frame_idx in frame_indices}
            )
            frames_with_time = sum(
                1 for frame_idx in frame_indices if self.timestamp_for_frame(frame_idx)[0] is not None
            )
            represented_frames = len(set(frame_indices))
        else:
            timestamp_sources = [self.source]
            frames_with_time = 0
            represented_frames = 0
        return {
            "status": self.status,
            "source": self.source,
            "timestamp_sources": timestamp_sources,
            "represented_frames": represented_frames,
            "frames_with_time": frames_with_time,
            "sync_schema_version": self.sync_schema_version,
            "raw_pts_status": self.raw_pts_status,
            "external_time_status": self.external_time_status,
            "external_time_source": self.external_time_source,
            "external_time_variable_name": self.external_time_variable_name,
            "external_time_units": self.external_time_units,
            "fps_header": self.fps_header,
            "raw_fps_effective": self.raw_fps_effective,
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
            "raw_frame_count_opencv_readable": self.raw_frame_count_opencv_readable,
            "prepared_frame_count_opencv_reported": self.prepared_frame_count_opencv_reported,
            "prepared_frame_count_opencv_readable": self.prepared_frame_count_opencv_readable,
            "frame_count_used_for_sleap": self.frame_count_used_for_sleap,
            "fallback_is_authoritative": self.status == "prepared_sync",
        }


@dataclass(frozen=True)
class PoseNodeRecord:
    """One node prediction in prepared-video coordinates."""

    node: str
    x: float | None
    y: float | None
    score: float | None = None


@dataclass(frozen=True)
class PoseInstanceRecord:
    """One retained predicted instance in a frame."""

    instance_index: int
    nodes: tuple[PoseNodeRecord, ...]
    track: str | None = None
    instance_score: float | None = None


@dataclass(frozen=True)
class PoseFrameRecord:
    """One prepared-video frame with retained predicted instances."""

    frame_idx: int
    instances: tuple[PoseInstanceRecord, ...]
    video_index: int = 0
    video_path: str | None = None
    frame_valid: bool = True
    original_frame_idx: int | None = None
    prepared_frame_idx: int | None = None
    source_video_frame_idx: int | None = None
    video_fps: float | None = None


@dataclass(frozen=True)
class ParquetExportSummary:
    """Machine-readable summary of a pose.parquet export attempt."""

    status: str
    path: Path
    schema_version: str = POSE_PARQUET_SCHEMA_VERSION
    rows: int = 0
    columns: tuple[str, ...] = PARQUET_COLUMNS
    timestamp_sources: tuple[str, ...] = ("unavailable",)
    timing: dict[str, object] | None = None
    error: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": self.status,
            "path": str(self.path.resolve()),
            "schema_version": self.schema_version,
            "rows": self.rows,
            "columns": list(self.columns),
            "timestamp_sources": list(self.timestamp_sources),
            "timing": self.timing or {"status": "unavailable"},
            "error": self.error,
        }


ParquetWriter = Callable[[Sequence[Mapping[str, object]], Path, Sequence[str]], None]
LabelsLoader = Callable[[Path], object]


def _safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _walk_values(payload: object, key_candidates: set[str]) -> object | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in key_candidates:
                return value
        for value in payload.values():
            found = _walk_values(value, key_candidates)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        for value in payload:
            found = _walk_values(value, key_candidates)
            if found is not None:
                return found
    return None


def _numeric_vector(value: object, *, frame_count: int | None) -> tuple[float, ...] | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0:
        return None
    if frame_count is not None and array.size != frame_count:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return tuple(float(item) for item in array)


def _int_vector(value: object, *, frame_count: int | None) -> tuple[int, ...] | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0:
        return None
    if frame_count is not None and array.size != frame_count:
        return None
    if not np.issubdtype(array.dtype, np.integer):
        return None
    return tuple(int(item) for item in array)


def _sync_vector(value: object) -> tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64))


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _load_s1_sync_timing(prepared_sync_path: Path) -> TimingLookup | None:
    if not prepared_sync_path.is_file():
        return None
    try:
        from preprocess.sync_writer import load_prepared_sync_npz

        sync = load_prepared_sync_npz(prepared_sync_path)
    except Exception:
        return None
    return TimingLookup(
        source="prepared_sync",
        status="prepared_sync",
        sync_schema_version=sync.sync_schema_version,
        prepared_frame_idx=tuple(int(item) for item in sync.prepared_frame_idx),
        raw_decode_frame_idx=tuple(int(item) for item in sync.raw_decode_frame_idx),
        prepared_time_sec=_sync_vector(sync.prepared_time_sec),
        raw_pts_time_sec=_sync_vector(sync.raw_pts_time_sec),
        external_time_sec=_sync_vector(sync.external_time_sec),
        raw_pts_status=sync.raw_pts_status,
        external_time_status=sync.external_time_status,
        external_time_source=_optional_text(sync.external_time_source),
        external_time_variable_name=_optional_text(sync.external_time_variable_name),
        external_time_units=sync.external_time_units,
        fps_header=float(sync.fps_header),
        raw_fps_effective=(
            float(sync.raw_fps_effective) if sync.raw_fps_effective is not None else None
        ),
        start_frame=sync.start_frame,
        end_frame_exclusive=sync.end_frame_exclusive,
        raw_frame_count_opencv_readable=sync.raw_frame_count_opencv_readable,
        prepared_frame_count_opencv_reported=sync.prepared_frame_count_opencv_reported,
        prepared_frame_count_opencv_readable=sync.prepared_frame_count_opencv_readable,
        frame_count_used_for_sleap=sync.frame_count_used_for_sleap,
    )


def _load_sync_timing(prepared_sync_path: Path, *, frame_count: int | None) -> TimingLookup | None:
    if not prepared_sync_path.is_file():
        return None
    s1_timing = _load_s1_sync_timing(prepared_sync_path)
    if s1_timing is not None:
        return s1_timing
    try:
        with np.load(prepared_sync_path, allow_pickle=False) as sync:
            for key in TIMING_VECTOR_KEYS:
                if key not in sync:
                    continue
                vector = _numeric_vector(sync[key], frame_count=frame_count)
                if vector is not None:
                    prepared_idx = (
                        _int_vector(sync["prepared_frame_idx"], frame_count=frame_count)
                        if "prepared_frame_idx" in sync
                        else None
                    )
                    raw_idx = (
                        _int_vector(sync["raw_decode_frame_idx"], frame_count=frame_count)
                        if "raw_decode_frame_idx" in sync
                        else None
                    )
                    return TimingLookup(
                        source=f"prepared_sync:{key}",
                        status="prepared_sync_generic",
                        times_sec=vector,
                        prepared_frame_idx=prepared_idx,
                        raw_decode_frame_idx=raw_idx,
                    )
            for key in sync.files:
                vector = _numeric_vector(sync[key], frame_count=frame_count)
                if vector is not None:
                    return TimingLookup(
                        source=f"prepared_sync:{key}",
                        status="prepared_sync_generic",
                        times_sec=vector,
                    )
    except (OSError, ValueError, EOFError):
        return None
    return None


def _load_meta_fps(prepare_meta_path: Path) -> float | None:
    payload = _read_json_mapping(prepare_meta_path)
    value = _walk_values(payload, set(FPS_KEYS))
    fps = _safe_float(value)
    return fps if fps is not None and fps > 0 else None


def load_timing_lookup(preprocess_dir: Path, *, frame_count: int | None) -> TimingLookup:
    """Resolve S1 timing for pose.parquet rows without making it mandatory."""

    preprocess = Path(preprocess_dir)
    sync_timing = _load_sync_timing(
        preprocess / "prepared_sync.npz",
        frame_count=frame_count,
    )
    if sync_timing is not None:
        return sync_timing
    fps = _load_meta_fps(preprocess / "prepare_meta.json")
    if fps is not None:
        return TimingLookup(source="prepare_meta_fps_fallback", status="fallback_fps", fps=fps)
    return TimingLookup(source="unavailable", status="unavailable")


def _is_valid_coordinate(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def pose_frames_to_rows(
    frames: Sequence[PoseFrameRecord],
    timing: TimingLookup,
) -> list[dict[str, object]]:
    """Convert normalized pose frame records to long/tidy pose_parquet_v1 rows."""

    rows: list[dict[str, object]] = []
    for frame in frames:
        n_instances = len(frame.instances)
        timing_fields = timing.fields_for_frame(frame.frame_idx)
        prepared_frame_idx = timing_fields["prepared_frame_idx"]
        if prepared_frame_idx is None:
            prepared_frame_idx = (
                frame.prepared_frame_idx if frame.prepared_frame_idx is not None else frame.frame_idx
            )
        raw_decode_frame_idx = timing_fields["raw_decode_frame_idx"]
        original_frame_idx = (
            raw_decode_frame_idx
            if raw_decode_frame_idx is not None
            else frame.original_frame_idx
        )
        source_video_frame_idx = (
            raw_decode_frame_idx
            if raw_decode_frame_idx is not None
            else frame.source_video_frame_idx
        )
        video_fps = frame.video_fps if frame.video_fps is not None else timing.fps_header
        for instance in frame.instances:
            valid_keypoints = sum(
                1
                for node in instance.nodes
                if _is_valid_coordinate(node.x) and _is_valid_coordinate(node.y)
            )
            for node in instance.nodes:
                rows.append(
                    {
                        "schema_version": POSE_PARQUET_SCHEMA_VERSION,
                        "video_index": frame.video_index,
                        "video_path": frame.video_path,
                        "frame_idx": frame.frame_idx,
                        **timing_fields,
                        "prepared_frame_idx": prepared_frame_idx,
                        "raw_decode_frame_idx": raw_decode_frame_idx,
                        "instance_index": instance.instance_index,
                        "track": instance.track,
                        "node": node.node,
                        "x": node.x,
                        "y": node.y,
                        "score": node.score,
                        "instance_score": instance.instance_score,
                        "frame_valid": frame.frame_valid,
                        "original_frame_idx": original_frame_idx,
                        "source_video_frame_idx": source_video_frame_idx,
                        "video_fps": video_fps,
                        "n_instances_in_frame": n_instances,
                        "n_valid_keypoints_in_instance": valid_keypoints,
                    }
                )
    return rows


def _track_name(track: object) -> str | None:
    if track is None:
        return None
    for attr in ("name", "id", "spawned_on"):
        value = getattr(track, attr, None)
        if value is not None:
            return str(value)
    text = str(track)
    return text if text else None


def _video_path(video: object) -> str | None:
    if video is None:
        return None
    for attr in ("filename", "filepath", "path"):
        value = getattr(video, attr, None)
        if value is not None:
            return str(value)
    backend = getattr(video, "backend", None)
    for attr in ("filename", "filepath", "path"):
        value = getattr(backend, attr, None)
        if value is not None:
            return str(value)
    return None


def _video_fps(video: object) -> float | None:
    if video is None:
        return None
    return _safe_float(getattr(video, "fps", None))


def _frame_video_index(video: object, videos: Sequence[object]) -> int:
    for index, candidate in enumerate(videos):
        if candidate is video:
            return index
    return 0


def _node_names(instance: object, point_count: int) -> list[str]:
    skeleton = getattr(instance, "skeleton", None)
    nodes = getattr(skeleton, "nodes", None)
    if nodes:
        names = [str(getattr(node, "name", node)) for node in nodes]
        if len(names) >= point_count:
            return names[:point_count]
    return [f"node_{index}" for index in range(point_count)]


def _numeric_or_nan(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(numeric):
        return math.nan
    return numeric if math.isfinite(numeric) else None


def _node_name(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return fallback
    else:
        text = str(value)
    return text if text else fallback


def _structured_instance_nodes(instance: object, points: object) -> tuple[PoseNodeRecord, ...] | None:
    try:
        point_array = np.asarray(points)
    except (TypeError, ValueError):
        return None
    field_names = point_array.dtype.names
    if not field_names or "xy" not in field_names:
        return None

    flat_points = point_array.reshape(-1)
    fallback_names = _node_names(instance, len(flat_points))
    records = []
    for index, point in enumerate(flat_points):
        xy = point["xy"]
        x = _numeric_or_nan(xy[0]) if len(xy) > 0 else None
        y = _numeric_or_nan(xy[1]) if len(xy) > 1 else None
        score = _numeric_or_nan(point["score"]) if "score" in field_names else None
        node = (
            _node_name(point["name"], fallback_names[index])
            if "name" in field_names
            else fallback_names[index]
        )
        records.append(PoseNodeRecord(node=node, x=x, y=y, score=score))
    return tuple(records)


def _point_xy_score(point: object, score: object = None) -> tuple[float | None, float | None, float | None]:
    if isinstance(point, Mapping):
        return (
            _safe_float(point.get("x")),
            _safe_float(point.get("y")),
            _safe_float(point.get("score", score)),
        )
    if isinstance(point, np.ndarray):
        point = point.tolist()
    if isinstance(point, Sequence) and not isinstance(point, str | bytes | bytearray):
        x = _safe_float(point[0]) if len(point) > 0 else None
        y = _safe_float(point[1]) if len(point) > 1 else None
        inferred_score = _safe_float(point[2]) if len(point) > 2 else _safe_float(score)
        return x, y, inferred_score
    return (
        _safe_float(getattr(point, "x", None)),
        _safe_float(getattr(point, "y", None)),
        _safe_float(getattr(point, "score", score)),
    )


def _instance_nodes(instance: object) -> tuple[PoseNodeRecord, ...]:
    points = getattr(instance, "points", None)
    scores = getattr(instance, "scores", None)
    if scores is None:
        scores = getattr(instance, "point_scores", None)
    if points is None:
        points = getattr(instance, "points_array", None)
    if points is None:
        return ()
    if isinstance(points, Mapping):
        records = []
        for name, point in points.items():
            x, y, score = _point_xy_score(point)
            records.append(PoseNodeRecord(node=str(name), x=x, y=y, score=score))
        return tuple(records)
    structured_nodes = _structured_instance_nodes(instance, points)
    if structured_nodes is not None:
        return structured_nodes

    point_sequence = list(np.asarray(points, dtype=object))
    score_sequence = [] if scores is None else list(np.asarray(scores, dtype=object))
    names = _node_names(instance, len(point_sequence))
    records = []
    for index, point in enumerate(point_sequence):
        score = score_sequence[index] if index < len(score_sequence) else None
        x, y, node_score = _point_xy_score(point, score)
        records.append(PoseNodeRecord(node=names[index], x=x, y=y, score=node_score))
    return tuple(records)


def _instance_score(instance: object) -> float | None:
    for attr in ("score", "instance_score"):
        value = _safe_float(getattr(instance, attr, None))
        if value is not None:
            return value
    return None


def sleap_labels_to_frame_records(labels: object) -> list[PoseFrameRecord]:
    """Best-effort conversion from a sleap_io labels object to frame records."""

    videos = list(getattr(labels, "videos", []) or [])
    labeled_frames = getattr(labels, "labeled_frames", None)
    if labeled_frames is None:
        labeled_frames = getattr(labels, "frames", None)
    if labeled_frames is None:
        raise ParquetExportError("Loaded SLEAP labels object does not expose labeled frames.")

    frames: list[PoseFrameRecord] = []
    for labeled_frame in labeled_frames:
        video = getattr(labeled_frame, "video", None)
        frame_idx = getattr(labeled_frame, "frame_idx", None)
        if frame_idx is None:
            frame_idx = getattr(labeled_frame, "frame_index", None)
        if frame_idx is None:
            raise ParquetExportError("A SLEAP labeled frame is missing frame_idx.")
        instances = getattr(labeled_frame, "instances", None)
        if instances is None:
            instances = getattr(labeled_frame, "predicted_instances", None)
        instances = list(instances or [])
        instance_records = []
        for index, instance in enumerate(instances):
            nodes = _instance_nodes(instance)
            instance_records.append(
                PoseInstanceRecord(
                    instance_index=index,
                    nodes=nodes,
                    track=_track_name(getattr(instance, "track", None)),
                    instance_score=_instance_score(instance),
                )
            )
        frames.append(
            PoseFrameRecord(
                frame_idx=int(frame_idx),
                instances=tuple(instance_records),
                video_index=_frame_video_index(video, videos),
                video_path=_video_path(video),
                original_frame_idx=int(frame_idx),
                prepared_frame_idx=int(frame_idx),
                source_video_frame_idx=int(frame_idx),
                video_fps=_video_fps(video),
            )
        )
    return frames


def load_sleap_frame_records(pose_slp_path: Path) -> list[PoseFrameRecord]:
    """Load a native SLEAP file through sleap_io and normalize frame records."""

    try:
        import sleap_io  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ParquetExportError(
            "sleap_io is required to export pose.parquet from pose.slp. "
            "Install the Subsystem 02 SLEAP-NN environment and retry."
        ) from exc

    if hasattr(sleap_io, "load_slp"):
        labels = sleap_io.load_slp(pose_slp_path)
    elif hasattr(sleap_io, "load_file"):
        labels = sleap_io.load_file(pose_slp_path)
    else:
        raise ParquetExportError("Installed sleap_io does not expose load_slp/load_file.")
    return sleap_labels_to_frame_records(labels)


def _write_rows_with_pandas(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    columns: Sequence[str],
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ParquetExportError(
            "pandas is required to write pose.parquet. Install pandas and a Parquet engine."
        ) from exc
    frame = pd.DataFrame(rows, columns=list(columns))
    try:
        frame.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
    except ImportError as exc:
        raise ParquetExportError(
            "A Parquet engine is required to write pose.parquet. Install pyarrow or fastparquet."
        ) from exc
    except Exception as exc:
        raise ParquetExportError(f"Could not write pose.parquet: {exc}") from exc


def _validate_prediction_frame_contract(
    frames: Sequence[PoseFrameRecord],
    timing: TimingLookup,
) -> tuple[tuple[int, int], ...]:
    authoritative = (
        timing.status == "prepared_sync"
        and timing.frame_count_used_for_sleap is not None
    )
    expected_count = int(timing.frame_count_used_for_sleap) if authoritative else None
    canonical_keys: list[tuple[int, int]] = []
    invalid_indices: list[int] = []
    for frame in frames:
        frame_idx = int(frame.frame_idx)
        if expected_count is not None and not 0 <= frame_idx < expected_count:
            invalid_indices.append(frame_idx)
            continue
        expected_prepared_idx = (
            int(timing.prepared_frame_idx[frame_idx])
            if authoritative and timing.prepared_frame_idx is not None
            else (
                int(frame.prepared_frame_idx)
                if frame.prepared_frame_idx is not None
                else frame_idx
            )
        )
        if (
            authoritative
            and frame.prepared_frame_idx is not None
            and int(frame.prepared_frame_idx) != expected_prepared_idx
        ):
            raise ParquetExportError(
                "Prediction prepared_frame_idx does not match the authoritative S1 "
                f"mapping for video_index={frame.video_index}, frame_idx={frame_idx}: "
                f"received {frame.prepared_frame_idx}, expected {expected_prepared_idx}."
            )
        canonical_keys.append((int(frame.video_index), expected_prepared_idx))

    if len(canonical_keys) != len(set(canonical_keys)):
        raise ParquetExportError(
            "Prediction frame keys cannot be reconciled with the authoritative S1 "
            "prepared-frame contract because duplicate "
            "(video_index, prepared_frame_idx) records are present."
        )
    if invalid_indices:
        assert expected_count is not None
        raise ParquetExportError(
            "Prediction frame indices cannot be reconciled with the authoritative "
            f"S1 prepared-frame range 0..{expected_count - 1}: "
            f"{sorted(invalid_indices)}."
        )
    return tuple(canonical_keys)


def _validate_written_parquet(path: Path) -> None:
    if not path.is_file():
        raise ParquetExportError(f"Required pose.parquet was not generated: {path}")
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise ParquetExportError(f"Required pose.parquet is unreadable: {path}: {exc}") from exc


def export_pose_parquet(
    *,
    pose_slp_path: Path,
    preprocess_dir: Path,
    output_path: Path,
    labels_loader: Callable[[Path], Sequence[PoseFrameRecord]] | None = None,
    parquet_writer: ParquetWriter | None = None,
) -> ParquetExportSummary:
    """Export long/tidy ``pose_parquet_v1`` rows from one native ``pose.slp``."""

    pose_slp = Path(pose_slp_path)
    if not pose_slp.is_file():
        raise ParquetExportError(f"pose.slp does not exist: {pose_slp}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    loader = labels_loader or load_sleap_frame_records
    frames = list(loader(pose_slp))
    frame_count = max((frame.frame_idx for frame in frames), default=-1) + 1
    timing = load_timing_lookup(preprocess_dir, frame_count=frame_count or None)
    canonical_frame_keys = _validate_prediction_frame_contract(frames, timing)
    rows = pose_frames_to_rows(frames, timing)

    writer = parquet_writer or _write_rows_with_pandas
    writer(rows, output, PARQUET_COLUMNS)
    _validate_written_parquet(output)
    frame_indices = [frame.frame_idx for frame in frames]
    timing_metadata = timing.to_metadata(frame_indices)
    timing_metadata["represented_frames"] = len(set(canonical_frame_keys))
    return ParquetExportSummary(
        status="generated",
        path=output,
        rows=len(rows),
        columns=PARQUET_COLUMNS,
        timestamp_sources=tuple(sorted({str(row["timestamp_source"]) for row in rows}))
        if rows
        else (timing.source,),
        timing=timing_metadata,
    )
