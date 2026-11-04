"""Resolve and pin an explicitly selected eligible S2 inferred-pose QC source.

Uses Subsystem 3 handoff eligibility (``validate_s2_handoff``) — completed S2
with authoritative ``pose.parquet`` and resolvable S1 prepared video/timing.
Never silently selects the newest S2 run. Never applies S3 acceptance gates.
Scientific trees remain read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.sync import SyncMap
from overlay.contracts import OverlaySourceError
from preprocess.sync_writer import load_prepared_sync_npz
from project.file_digest import sha256_file
from results_packaging.contracts import SYNC_SCHEMA_VERSION
from results_packaging.media_csv_export.video import VideoProbe, probe_video
from tracking_correction.runner import validate_s2_handoff
from tracking_correction.skeleton import (
    resolve_review_skeleton_edges,
    resolve_s2_pose_slp_path,
)

_POSE_META_INPUT_KEYS = (
    "session_root",
    "preprocess_dir",
    "prepared_video",
    "prepare_meta",
    "prepared_sync",
)


@dataclass(frozen=True, slots=True)
class OverlayS2QcSource:
    """Pinned S2 inferred-pose overlay source (explicit run only)."""

    s2_run_id: str
    s2_run_dir: Path
    s2_status: str | None
    pose_parquet_path: Path
    pose_parquet_sha256: str
    session_root: Path
    prepared_sync_path: Path
    prepared_sync_sha256: str
    prepared_video_path: Path
    prepared_video_sha256: str
    prepare_meta_path: Path
    video_probe: VideoProbe
    sync: SyncMap
    sync_build_audit: dict[str, Any]
    prepared_frame_count: int
    prepared_fps: float
    skeleton_edges: tuple[tuple[str, str], ...]
    skeleton_source: str | None
    pose_slp_path: Path | None
    settings_used_path: Path | None
    pose_meta: dict[str, Any]
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


def resolve_overlay_s2_qc_source(
    *,
    s2_run_dir: Path | str,
) -> OverlayS2QcSource:
    """Open one explicitly selected eligible S2 run for pose QC export.

    Rejects missing / incomplete / incompatible artifacts. Does not discover or
    fall back to another S2 run. Eligibility matches S3 handoff validation
    (completed pose inference + authoritative pose.parquet + S1 timing), not
    S3 acceptance.
    """

    run_dir = Path(s2_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise OverlaySourceError(f"S2 run directory does not exist: {run_dir}")

    try:
        validated_dir = validate_s2_handoff(run_dir)
    except Exception as exc:  # noqa: BLE001 - surface as source eligibility error
        raise OverlaySourceError(
            f"S2 run is not eligible for overlay QC (S2 handoff validation): {exc}"
        ) from exc
    run_dir = Path(validated_dir).resolve()

    pose_parquet = run_dir / "pose.parquet"
    if not pose_parquet.is_file():
        raise OverlaySourceError(f"Authoritative S2 pose.parquet missing: {pose_parquet}")

    pose_meta = _load_optional_json(run_dir / "pose_meta.json")
    provenance = _resolve_s1_paths(run_dir, pose_meta)
    prepared_video = provenance["prepared_video"]
    prepared_sync = provenance["prepared_sync"]
    prepare_meta = provenance["prepare_meta"]
    session_root = provenance["session_root"]

    if not prepared_video.is_file():
        raise OverlaySourceError(f"S1 prepared_video.mp4 missing for S2 QC: {prepared_video}")
    if not prepared_sync.is_file():
        raise OverlaySourceError(f"S1 prepared_sync.npz missing for S2 QC: {prepared_sync}")
    if not prepare_meta.is_file():
        raise OverlaySourceError(f"S1 prepare_meta.json missing for S2 QC: {prepare_meta}")

    try:
        sync, sync_audit = _build_prepared_identity_sync(prepared_sync)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            "Authoritative prepared-frame timing could not be built from S1 "
            f"prepared_sync.npz (no dense 120 Hz mapping used): {exc}"
        ) from exc

    pose_slp = resolve_s2_pose_slp_path(s2_run_dir=run_dir, pose_parquet=pose_parquet)
    skeleton_edges, skeleton_source = _resolve_skeleton_edges(pose_slp)

    settings_used = run_dir / "settings_used.yaml"
    settings_path = settings_used if settings_used.is_file() else None

    pose_sha = sha256_file(pose_parquet)
    prepared_sync_sha = sha256_file(prepared_sync)
    video_sha = sha256_file(prepared_video)
    try:
        video_probe = probe_video(prepared_video)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"Failed to probe S1 prepared video for S2 QC: {exc}"
        ) from exc

    run_id = str(
        (pose_meta or {}).get("run_id")
        or run_dir.name
    )
    status = None
    if pose_meta is not None:
        raw_status = pose_meta.get("status")
        if isinstance(raw_status, str) and raw_status.strip():
            status = raw_status.strip()

    prepared_frame_count = int(sync_audit["n_prepared"])
    prepared_fps = float(sync_audit["fps_header"])
    if int(video_probe.n_frames) > 0 and int(video_probe.n_frames) != prepared_frame_count:
        # Soft consistency check: probe vs sync must agree when both authoritative.
        raise OverlaySourceError(
            "Prepared video frame count does not match prepared_sync frame domain: "
            f"video={video_probe.n_frames} sync={prepared_frame_count}."
        )

    source_digests: dict[str, str | None] = {
        "s2_pose_parquet_sha256": pose_sha,
        "s1_prepared_sync_sha256": prepared_sync_sha,
        "s1_prepared_video_sha256": video_sha,
    }

    return OverlayS2QcSource(
        s2_run_id=run_id,
        s2_run_dir=run_dir,
        s2_status=status,
        pose_parquet_path=pose_parquet,
        pose_parquet_sha256=pose_sha,
        session_root=session_root,
        prepared_sync_path=prepared_sync,
        prepared_sync_sha256=prepared_sync_sha,
        prepared_video_path=prepared_video,
        prepared_video_sha256=video_sha,
        prepare_meta_path=prepare_meta,
        video_probe=video_probe,
        sync=sync,
        sync_build_audit=sync_audit,
        prepared_frame_count=prepared_frame_count,
        prepared_fps=prepared_fps,
        skeleton_edges=skeleton_edges,
        skeleton_source=skeleton_source,
        pose_slp_path=pose_slp,
        settings_used_path=settings_path,
        pose_meta=dict(pose_meta or {}),
        source_digests=source_digests,
        provenance={
            "source_kind": "s2_inferred_pose",
            "s2_run_id": run_id,
            "s2_run_dir": str(run_dir),
            "s2_status": status,
            "identity_semantics": "provisional_s2_predictions",
            "identity_note": (
                "Track/instance colors and P{{id}} labels are provisional S2 "
                "predictions — not S3-accepted identities."
            ),
            "prepared_sync": str(prepared_sync),
            "prepared_video": str(prepared_video),
            "timing_authority": "s1_prepared_frame_identity",
            "timing_note": (
                "S2 pose is already in the prepared-frame domain. QC uses S1 "
                "prepared-frame indices and prepared-video FPS. SyncMap is a "
                "1:1 prepared-identity map for compositor reuse — not S4 dense "
                "120 Hz duplication."
            ),
            "skeleton_source": skeleton_source,
            "skeleton_mode": (
                "topology_edges" if skeleton_edges else "nodes_only"
            ),
            "skeleton_nodes_only": not bool(skeleton_edges),
            "skeleton_edge_count": len(skeleton_edges),
            "eligibility": "validate_s2_handoff",
        },
    )


def verify_overlay_s2_qc_source_integrity(source: OverlayS2QcSource) -> None:
    """Recheck digests and S2 handoff eligibility before publication."""

    try:
        validate_s2_handoff(source.s2_run_dir)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S2 eligibility failed during integrity recheck: {exc}"
        ) from exc

    live: dict[str, str | None] = {
        "s2_pose_parquet_sha256": (
            sha256_file(source.pose_parquet_path)
            if source.pose_parquet_path.is_file()
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
                f"Overlay S2 QC source digest mismatch for {key}: "
                f"pinned={expected!r} live={actual!r}."
            )


def _build_prepared_identity_sync(
    prepared_sync_path: Path,
) -> tuple[SyncMap, dict[str, Any]]:
    """Build a 1:1 prepared-frame SyncMap (prepared rate — not dense 120 Hz)."""

    sync_data = load_prepared_sync_npz(prepared_sync_path)
    n = int(sync_data.frame_count_used_for_sleap)
    if n <= 0:
        raise OverlaySourceError("prepared_sync.npz reports non-positive frame count.")
    prepared_idx = np.asarray(sync_data.prepared_frame_idx, dtype=np.int64)[:n]
    prepared_time = np.asarray(sync_data.prepared_time_sec, dtype=np.float64)[:n]
    if prepared_idx.shape[0] != n or prepared_time.shape[0] != n:
        raise OverlaySourceError(
            "prepared_sync.npz prepared arrays shorter than frame_count_used_for_sleap."
        )
    expected = np.arange(n, dtype=np.int64)
    if not np.array_equal(prepared_idx, expected):
        raise OverlaySourceError(
            "prepared_sync.npz prepared_frame_idx must equal contiguous 0..N-1."
        )
    fps = float(sync_data.fps_header)
    if not np.isfinite(fps) or fps <= 0:
        raise OverlaySourceError(
            f"prepared_sync.npz fps_header must be positive; got {fps}."
        )

    table = pd.DataFrame(
        {
            "schema_version": SYNC_SCHEMA_VERSION,
            "sample_idx": prepared_idx,
            "dense_frame_idx": prepared_idx,
            "time_sec": prepared_time,
            "prepared_frame_idx": prepared_idx,
            "prepared_time_sec": prepared_time,
            "is_exact_source_sample": np.ones(n, dtype=bool),
            "timing_residual_sec": np.zeros(n, dtype=np.float64),
            "raw_decode_frame_idx": np.asarray(
                sync_data.raw_decode_frame_idx, dtype=np.int64
            )[:n],
        }
    )
    sync = SyncMap.from_sync_index(table, fps=fps)
    audit = {
        "kind": "s1_prepared_frame_identity",
        "n_prepared": n,
        "fps_header": fps,
        "max_many_to_one": 1,
        "dense_120hz": False,
        "note": (
            "Identity prepared↔sample map for compositor timing reuse; "
            "not an S4 dense 120 Hz timeline."
        ),
    }
    return sync, audit


def _resolve_skeleton_edges(
    pose_slp: Path | None,
) -> tuple[tuple[tuple[str, str], ...], str | None]:
    """Resolve S2 skeleton edges from pose.slp only; never invent topology."""

    edges = resolve_review_skeleton_edges(pose_slp)
    if pose_slp is not None and edges:
        return edges, f"s2_pose_slp:{pose_slp}"
    if pose_slp is not None:
        return (), f"unresolved:{pose_slp}"
    return (), None


def _resolve_s1_paths(
    run_dir: Path,
    pose_meta: dict[str, Any] | None,
) -> dict[str, Path]:
    recorded = pose_meta.get("input") if isinstance(pose_meta, dict) else None
    if isinstance(recorded, dict) and any(key in recorded for key in _POSE_META_INPUT_KEYS):
        missing = [
            key
            for key in _POSE_META_INPUT_KEYS
            if not (isinstance(recorded.get(key), str) and str(recorded.get(key)).strip())
        ]
        if missing:
            raise OverlaySourceError(
                "S2 pose_meta.json does not record the S1 handoff paths: "
                + ", ".join(missing)
            )
        return {
            "session_root": _resolve_recorded_path(str(recorded["session_root"]), run_dir),
            "preprocess_dir": _resolve_recorded_path(
                str(recorded["preprocess_dir"]), run_dir
            ),
            "prepared_video": _resolve_recorded_path(
                str(recorded["prepared_video"]), run_dir
            ),
            "prepare_meta": _resolve_recorded_path(str(recorded["prepare_meta"]), run_dir),
            "prepared_sync": _resolve_recorded_path(
                str(recorded["prepared_sync"]), run_dir
            ),
        }

    if run_dir.parent.name != "pose_inference":
        raise OverlaySourceError(
            "S1 prepared video and timing cannot be resolved from S2 provenance. "
            f"pose_meta.json input paths are missing and {run_dir} is not under "
            "a pose_inference directory."
        )
    session_root = run_dir.parent.parent.resolve()
    preprocess_dir = session_root / "preprocess"
    return {
        "session_root": session_root,
        "preprocess_dir": preprocess_dir,
        "prepared_video": preprocess_dir / "prepared_video.mp4",
        "prepare_meta": preprocess_dir / "prepare_meta.json",
        "prepared_sync": preprocess_dir / "prepared_sync.npz",
    }


def _resolve_recorded_path(raw: str, run_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return (
        (run_dir / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(f"Failed to read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OverlaySourceError(f"Expected JSON object at {path}.")
    return payload
