"""Resolve and pin an explicitly selected eligible S4 finalized-pose QC source.

Never silently selects the newest S4 run. Scientific trees remain read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.sync import SyncMap
from overlay.contracts import OverlaySourceError
from pose_finalization.contracts import (
    FINAL_POSE_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
)
from pose_finalization.runner import validate_s4_run_for_downstream
from project.file_digest import sha256_file
from results_packaging.media_csv_export.video import VideoProbe, probe_video
from results_packaging.sync_index import build_sync_index_from_prepared_sync
from tracking_correction.skeleton import (
    resolve_review_skeleton_edges,
    resolve_s2_pose_slp_path,
)


@dataclass(frozen=True, slots=True)
class OverlayS4QcSource:
    """Pinned S4 finalized-pose overlay source (explicit run only)."""

    s4_run_id: str
    s4_run_dir: Path
    s4_status: str
    final_pose_path: Path
    final_pose_sha256: str
    session_root: Path
    prepared_sync_path: Path
    prepared_sync_sha256: str
    prepared_video_path: Path
    prepared_video_sha256: str
    video_probe: VideoProbe
    sync: SyncMap
    sync_build_audit: dict[str, Any]
    source_s3_run_id: str
    source_s3_run_dir: Path
    source_s3_tracked_pose_sha256: str
    skeleton_edges: tuple[tuple[str, str], ...]
    skeleton_source: str | None
    settings_used_path: Path | None
    run_meta: dict[str, Any]
    source_digests: dict[str, str | None]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def skeleton_mode(self) -> str:
        """``topology_edges`` when real S2 edges resolved; else ``nodes_only``."""

        return "topology_edges" if self.skeleton_edges else "nodes_only"

    def skeleton_disclosure(self) -> dict[str, Any]:
        """Honest topology disclosure for QC log / manifest (never invents edges)."""

        mode = self.skeleton_mode
        if mode == "topology_edges":
            note = (
                "Skeleton edges drawn from resolved S2 pose.slp topology; "
                "edges only between drawable endpoints."
            )
        elif self.skeleton_source and str(self.skeleton_source).startswith("unresolved:"):
            note = (
                "Nodes-only overlay: pose.slp was located but topology could not "
                "be read (optional sleap_io missing/unreadable). No edges invented."
            )
        else:
            note = (
                "Nodes-only overlay: S2 pose.slp topology unavailable. "
                "No edges invented."
            )
        return {
            "skeleton_mode": mode,
            "skeleton_nodes_only": mode == "nodes_only",
            "skeleton_source": self.skeleton_source,
            "skeleton_edge_count": len(self.skeleton_edges),
            "skeleton_disclosure": note,
        }


def resolve_overlay_s4_qc_source(
    *,
    s4_run_dir: Path | str,
) -> OverlayS4QcSource:
    """Open one explicitly selected eligible S4 run for pose QC export.

    Rejects missing / incomplete / incompatible artifacts. Does not discover or
    fall back to another S4 run.
    """

    run_dir = Path(s4_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise OverlaySourceError(f"S4 run directory does not exist: {run_dir}")

    try:
        validated = validate_s4_run_for_downstream(run_dir)
    except Exception as exc:  # noqa: BLE001 - surface as source eligibility error
        raise OverlaySourceError(
            f"S4 run is not eligible for downstream overlay QC: {exc}"
        ) from exc

    meta_path = run_dir / RUN_META_FILENAME
    meta = _load_json_object(meta_path)
    source_s3 = meta.get("source_s3")
    if not isinstance(source_s3, dict):
        raise OverlaySourceError("S4 run_meta.json is missing source_s3 provenance.")

    prepared_sync_path = _resolve_prepared_sync(run_dir, source_s3)
    session_root = _resolve_session_root(run_dir, source_s3)
    prepared_video_path = session_root / "preprocess" / "prepared_video.mp4"
    if not prepared_video_path.is_file():
        raise OverlaySourceError(
            f"S1 prepared_video.mp4 missing for S4 QC: {prepared_video_path}"
        )

    final_pose_path = run_dir / FINAL_POSE_FILENAME
    if not final_pose_path.is_file():
        raise OverlaySourceError(f"S4 final_pose.parquet missing: {final_pose_path}")

    try:
        unique_dense = _unique_dense_timeline_from_final_pose(final_pose_path)
        sync_build = build_sync_index_from_prepared_sync(
            prepared_sync_path,
            s8_unique_dense_times=unique_dense,
            package_run_id=None,
        )
        sync = SyncMap.from_sync_index(sync_build.sync_index)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            "Authoritative dense↔prepared timing could not be built from S1 "
            f"prepared_sync + S4 dense timeline: {exc}"
        ) from exc

    skeleton_edges, skeleton_source = _resolve_skeleton_edges(
        s3_run_dir=Path(validated["source_s3_run_dir"])
    )

    settings_used = run_dir / SETTINGS_USED_FILENAME
    settings_path = settings_used if settings_used.is_file() else None

    final_pose_sha = sha256_file(final_pose_path)
    prepared_sync_sha = sha256_file(prepared_sync_path)
    video_sha = sha256_file(prepared_video_path)
    try:
        video_probe = probe_video(prepared_video_path)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"Failed to probe S1 prepared video for S4 QC: {exc}"
        ) from exc

    run_id = str(meta.get("run_id") or run_dir.name)
    source_digests: dict[str, str | None] = {
        "s4_final_pose_sha256": final_pose_sha,
        "s1_prepared_sync_sha256": prepared_sync_sha,
        "s1_prepared_video_sha256": video_sha,
        "s3_tracked_pose_sha256": str(validated["tracked_pose_sha256"]),
    }

    return OverlayS4QcSource(
        s4_run_id=run_id,
        s4_run_dir=run_dir,
        s4_status=str(validated["status"]),
        final_pose_path=final_pose_path,
        final_pose_sha256=final_pose_sha,
        session_root=session_root,
        prepared_sync_path=prepared_sync_path,
        prepared_sync_sha256=prepared_sync_sha,
        prepared_video_path=prepared_video_path,
        prepared_video_sha256=video_sha,
        video_probe=video_probe,
        sync=sync,
        sync_build_audit=sync_build.audit_dict(),
        source_s3_run_id=str(validated["source_s3_run_id"]),
        source_s3_run_dir=Path(validated["source_s3_run_dir"]),
        source_s3_tracked_pose_sha256=str(validated["tracked_pose_sha256"]),
        skeleton_edges=skeleton_edges,
        skeleton_source=skeleton_source,
        settings_used_path=settings_path,
        run_meta=meta,
        source_digests=source_digests,
        provenance={
            "source_kind": "s4_finalized_pose",
            "s4_run_id": run_id,
            "s4_run_dir": str(run_dir),
            "s4_status": str(validated["status"]),
            "source_s3": dict(source_s3),
            "prepared_sync": str(prepared_sync_path),
            "prepared_video": str(prepared_video_path),
            "timing_authority": "s1_prepared_sync_nn_map_validated_against_s4_dense",
            "skeleton_source": skeleton_source,
            "skeleton_mode": (
                "topology_edges" if skeleton_edges else "nodes_only"
            ),
            "skeleton_nodes_only": not bool(skeleton_edges),
            "skeleton_edge_count": len(skeleton_edges),
        },
    )


def verify_overlay_s4_qc_source_integrity(source: OverlayS4QcSource) -> None:
    """Recheck digests and S4/S3 eligibility before publication."""

    try:
        validate_s4_run_for_downstream(source.s4_run_dir)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S4 eligibility failed during integrity recheck: {exc}"
        ) from exc

    live: dict[str, str | None] = {
        "s4_final_pose_sha256": (
            sha256_file(source.final_pose_path)
            if source.final_pose_path.is_file()
            else None
        ),
        "s1_prepared_sync_sha256": (
            sha256_file(source.prepared_sync_path)
            if source.prepared_sync_path.is_file()
            else None
        ),
        "s1_prepared_video_sha256": (
            sha256_file(source.prepared_video_path)
            if source.prepared_video_path.is_file()
            else None
        ),
    }
    for key, expected in source.source_digests.items():
        if expected is None or key not in live:
            continue
        actual = live.get(key)
        if actual is None or str(actual) != str(expected):
            raise OverlaySourceError(
                f"Overlay S4 QC source digest mismatch for {key}: "
                f"pinned={expected!r} live={actual!r}."
            )


def _unique_dense_timeline_from_final_pose(final_pose_path: Path) -> pd.DataFrame:
    cols = ["sample_idx", "time_sec"]
    frame = pd.read_parquet(final_pose_path, columns=cols)
    if frame.empty:
        raise OverlaySourceError("S4 final_pose.parquet has no rows.")
    grouped = (
        frame.groupby("sample_idx", sort=True, as_index=False)["time_sec"]
        .first()
        .sort_values("sample_idx", kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["dense_frame_idx"] = grouped["sample_idx"].astype("int64")
    grouped["sample_idx"] = grouped["sample_idx"].astype("int64")
    return grouped[["dense_frame_idx", "sample_idx", "time_sec"]]


def _resolve_prepared_sync(run_dir: Path, source_s3: dict[str, Any]) -> Path:
    recorded = source_s3.get("prepared_sync")
    if isinstance(recorded, str) and recorded.strip():
        path = Path(recorded).expanduser()
        if path.is_file():
            return path.resolve()
    session = source_s3.get("session_root")
    if isinstance(session, str) and session.strip():
        candidate = Path(session).expanduser() / "preprocess" / "prepared_sync.npz"
        if candidate.is_file():
            return candidate.resolve()
    # Fall back relative to run_dir → session/preprocess.
    candidate = run_dir.parent.parent / "preprocess" / "prepared_sync.npz"
    if candidate.is_file():
        return candidate.resolve()
    raise OverlaySourceError(
        "Authoritative S1 prepared_sync.npz could not be resolved from S4 provenance."
    )


def _resolve_session_root(run_dir: Path, source_s3: dict[str, Any]) -> Path:
    session = source_s3.get("session_root")
    if isinstance(session, str) and session.strip():
        path = Path(session).expanduser()
        if path.is_dir():
            return path.resolve()
    return run_dir.parent.parent.resolve()


def _resolve_skeleton_edges(
    *,
    s3_run_dir: Path,
) -> tuple[tuple[tuple[str, str], ...], str | None]:
    """Resolve S2 skeleton edges via S3 provenance; never invent topology."""

    meta_path = s3_run_dir / "run_meta.json"
    s2_run_dir: Path | None = None
    if meta_path.is_file():
        try:
            meta = _load_json_object(meta_path)
        except OverlaySourceError:
            meta = {}
        input_block = meta.get("input") if isinstance(meta.get("input"), dict) else {}
        raw = input_block.get("s2_run_dir") if isinstance(input_block, dict) else None
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw).expanduser()
            if candidate.is_dir():
                s2_run_dir = candidate.resolve()
    pose_slp = resolve_s2_pose_slp_path(s2_run_dir=s2_run_dir)
    edges = resolve_review_skeleton_edges(pose_slp)
    if pose_slp is not None and edges:
        return edges, f"s2_pose_slp:{pose_slp}"
    if pose_slp is not None:
        return (), f"unresolved:{pose_slp}"
    return (), None


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(f"Failed to read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OverlaySourceError(f"Expected JSON object at {path}.")
    return payload
