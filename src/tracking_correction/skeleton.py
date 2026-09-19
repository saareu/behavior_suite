"""Resolve S2 skeleton topology for S3 review overlays.

S3 correction changes identity/coordinates, not biological skeleton edges.
Review overlays therefore reuse the exact edges recorded in the S2 pose.slp
artifact when sleap_io can read them. If topology cannot be resolved, overlays
draw nodes only — never a guessed lab-specific edge list.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tracking_correction.contracts import TrackingCorrectionError

POSE_SLP_FILENAME = "pose.slp"


def resolve_s2_pose_slp_path(
    *,
    s2_run_dir: Path | None,
    pose_parquet: Path | None = None,
) -> Path | None:
    """Locate ``pose.slp`` beside the S2 run / pose.parquet when present."""

    candidates: list[Path] = []
    if s2_run_dir is not None:
        candidates.append(Path(s2_run_dir) / POSE_SLP_FILENAME)
    if pose_parquet is not None:
        candidates.append(Path(pose_parquet).with_name(POSE_SLP_FILENAME))
    for path in candidates:
        resolved = path.expanduser()
        if resolved.is_file():
            return resolved.resolve()
    return None


def resolve_review_skeleton_edges(
    pose_slp_path: Path | None,
    *,
    labels: object | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return undirected-ready directed edges from an S2 pose.slp skeleton.

    ``labels`` may be injected for tests. When neither labels nor a readable
    pose.slp are available, returns an empty edge tuple (nodes-only overlay).
    """

    if labels is None:
        if pose_slp_path is None:
            return ()
        try:
            labels = _load_sleap_labels(pose_slp_path)
        except TrackingCorrectionError:
            return ()
    try:
        return skeleton_edges_from_labels(labels)
    except TrackingCorrectionError:
        return ()


def skeleton_edges_from_labels(labels: object) -> tuple[tuple[str, str], ...]:
    """Extract ``(source, destination)`` node-name edges from a sleap labels object."""

    skeleton = _first_skeleton(labels)
    if skeleton is None:
        raise TrackingCorrectionError("SLEAP labels do not expose a skeleton.")
    return edges_from_skeleton(skeleton)


def edges_from_skeleton(skeleton: object) -> tuple[tuple[str, str], ...]:
    """Normalize sleap_io skeleton edges to ``(source_name, dest_name)`` pairs."""

    edge_names = getattr(skeleton, "edge_names", None)
    if edge_names is not None:
        named_pairs = _pairs_from_edge_names(edge_names)
        if named_pairs:
            return named_pairs

    edges = getattr(skeleton, "edges", None)
    if edges is None:
        raise TrackingCorrectionError("Skeleton does not expose edges.")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        source_name, dest_name = _edge_endpoint_names(edge)
        if not source_name or not dest_name or source_name == dest_name:
            continue
        key = (source_name, dest_name)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return tuple(pairs)


def _pairs_from_edge_names(edge_names: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(edge_names, Sequence) or isinstance(edge_names, (str, bytes)):
        return ()
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in edge_names:
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) >= 2
        ):
            source_name = str(item[0]).strip()
            dest_name = str(item[1]).strip()
        else:
            continue
        if not source_name or not dest_name or source_name == dest_name:
            continue
        key = (source_name, dest_name)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return tuple(pairs)


def _edge_endpoint_names(edge: object) -> tuple[str, str]:
    if isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) >= 2:
        return str(edge[0]).strip(), str(edge[1]).strip()
    source = getattr(edge, "source", None)
    destination = getattr(edge, "destination", None)
    return _node_name(source), _node_name(destination)


def _node_name(node: object) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    name = getattr(node, "name", None)
    if isinstance(name, str):
        return name.strip()
    return str(node).strip()


def _first_skeleton(labels: object) -> object | None:
    skeletons = getattr(labels, "skeletons", None)
    if skeletons:
        try:
            return cast(object, skeletons[0])
        except (TypeError, IndexError, KeyError):
            pass
    skeleton = getattr(labels, "skeleton", None)
    if skeleton is not None:
        return cast(object, skeleton)
    labeled_frames = getattr(labels, "labeled_frames", None)
    if labeled_frames is None:
        labeled_frames = getattr(labels, "frames", None)
    if not labeled_frames:
        return None
    for labeled_frame in labeled_frames:
        instances = getattr(labeled_frame, "instances", None)
        if instances is None:
            instances = getattr(labeled_frame, "predicted_instances", None)
        for instance in list(instances or []):
            instance_skeleton = getattr(instance, "skeleton", None)
            if instance_skeleton is not None:
                return cast(object, instance_skeleton)
    return None


def _load_sleap_labels(pose_slp_path: Path) -> Any:
    path = Path(pose_slp_path).expanduser()
    if not path.is_file():
        raise TrackingCorrectionError(f"S2 pose.slp does not exist: {path}")
    try:
        import sleap_io  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TrackingCorrectionError(
            "sleap_io is required to read S2 skeleton topology from pose.slp."
        ) from exc
    try:
        if hasattr(sleap_io, "load_slp"):
            return sleap_io.load_slp(path)
        if hasattr(sleap_io, "load_file"):
            return sleap_io.load_file(path)
    except Exception as exc:
        raise TrackingCorrectionError(f"Could not load S2 pose.slp: {exc}") from exc
    raise TrackingCorrectionError("Installed sleap_io does not expose load_slp/load_file.")
