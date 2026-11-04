"""Resolve and pin an explicitly selected eligible accepted S3 QC source.

Uses ``validate_accepted_s3_run`` — ``acceptance_state == accepted`` with
authoritative ``tracked_pose.parquet`` digest match. Never silently selects the
newest S3 run. Never uses working-only pose. Scientific trees remain read-only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.sync import SyncMap
from overlay.contracts import OverlaySourceError
from overlay.corrections_s3 import (
    CorrectionJournalIndex,
    load_machine_correction_index,
    load_manual_correction_index,
)
from pose_finalization.accepted_s3 import AcceptedS3Run, validate_accepted_s3_run
from preprocess.sync_writer import load_prepared_sync_npz
from project.file_digest import sha256_file
from results_packaging.contracts import SYNC_SCHEMA_VERSION
from results_packaging.media_csv_export.video import VideoProbe, probe_video
from tracking_correction.contracts import (
    MACHINE_CORRECTIONS_FILENAME,
    MANUAL_CORRECTIONS_FILENAME,
    RUN_META_FILENAME,
)
from tracking_correction.skeleton import (
    resolve_review_skeleton_edges,
    resolve_s2_pose_slp_path,
)

_INPUT_KEYS = (
    "session_root",
    "preprocess_dir",
    "prepared_video",
    "prepare_meta",
    "prepared_sync",
)


@dataclass(frozen=True, slots=True)
class OverlayS3QcSource:
    """Pinned accepted S3 tracked-pose overlay source (explicit run only)."""

    s3_run_id: str
    s3_run_dir: Path
    acceptance_state: str
    tracked_pose_path: Path
    tracked_pose_sha256: str
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
    source_s2_run_id: str | None
    source_s2_run_dir: Path | None
    source_s2_pose_parquet_path: Path | None
    machine_corrections_path: Path | None
    manual_corrections_path: Path | None
    machine_events: CorrectionJournalIndex
    manual_events: CorrectionJournalIndex
    run_meta: dict[str, Any]
    source_digests: dict[str, str | None]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def skeleton_mode(self) -> str:
        return "topology_edges" if self.skeleton_edges else "nodes_only"

    def skeleton_disclosure(self) -> dict[str, Any]:
        mode = self.skeleton_mode
        if mode == "topology_edges":
            note = (
                "Skeleton edges drawn from resolved S2 pose.slp topology via S3 "
                "lineage; edges only between drawable endpoints."
            )
        elif self.skeleton_source and str(self.skeleton_source).startswith("unresolved:"):
            note = (
                "Nodes-only overlay: pose.slp was located but topology could not "
                "be read (optional sleap_io missing/unreadable). No edges invented."
            )
        else:
            note = (
                "Nodes-only overlay: S2 pose.slp topology unavailable from S3 "
                "lineage. No edges invented."
            )
        return {
            "skeleton_mode": mode,
            "skeleton_nodes_only": mode == "nodes_only",
            "skeleton_source": self.skeleton_source,
            "skeleton_edge_count": len(self.skeleton_edges),
            "skeleton_disclosure": note,
        }

    def correction_disclosure(self) -> dict[str, Any]:
        return {
            "machine": self.machine_events.to_disclosure(),
            "manual": self.manual_events.to_disclosure(),
            "was_track_swapped_policy": (
                "Frame-wide was_track_swapped parquet flags are never treated as "
                "per-node correction events and never invent suspicious-frame markers."
            ),
        }


def resolve_overlay_s3_qc_source(
    *,
    s3_run_dir: Path | str,
) -> OverlayS3QcSource:
    """Open one explicitly selected accepted S3 run for pose QC export.

    Rejects working-only / unaccepted / reviewable-only / mismatched lineage.
    Does not discover or fall back to another S3 run.
    """

    run_dir = Path(s3_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise OverlaySourceError(f"S3 run directory does not exist: {run_dir}")

    try:
        accepted: AcceptedS3Run = validate_accepted_s3_run(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S3 run is not eligible for overlay QC (accepted-run validation): {exc}"
        ) from exc
    run_dir = Path(accepted.run_dir).resolve()

    meta_path = run_dir / RUN_META_FILENAME
    run_meta = _load_json_object(meta_path)
    input_block = run_meta.get("input")
    if not isinstance(input_block, dict):
        input_block = {}

    session_root = Path(accepted.session_root).resolve()
    provenance_paths = _resolve_s1_paths(
        session_root=session_root,
        run_dir=run_dir,
        input_block=input_block,
    )
    prepared_video = provenance_paths["prepared_video"]
    prepared_sync = provenance_paths["prepared_sync"]
    prepare_meta = provenance_paths["prepare_meta"]

    if not prepared_video.is_file():
        raise OverlaySourceError(f"S1 prepared_video.mp4 missing for S3 QC: {prepared_video}")
    if not prepared_sync.is_file():
        raise OverlaySourceError(f"S1 prepared_sync.npz missing for S3 QC: {prepared_sync}")
    if not prepare_meta.is_file():
        raise OverlaySourceError(f"S1 prepare_meta.json missing for S3 QC: {prepare_meta}")

    # Prefer the sync path already validated by AcceptedS3Run when present.
    if accepted.prepared_sync_path.is_file():
        prepared_sync = Path(accepted.prepared_sync_path).resolve()

    try:
        sync, sync_audit = _build_prepared_identity_sync(prepared_sync)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            "Authoritative prepared-frame timing could not be built from S1 "
            f"prepared_sync.npz (no dense 120 Hz mapping used): {exc}"
        ) from exc

    s2_lineage = _resolve_s2_lineage(run_dir=run_dir, input_block=input_block)
    pose_slp = resolve_s2_pose_slp_path(
        s2_run_dir=s2_lineage["s2_run_dir"],
        pose_parquet=s2_lineage["pose_parquet"],
    )
    skeleton_edges, skeleton_source = _resolve_skeleton_edges(pose_slp)

    machine_path = run_dir / MACHINE_CORRECTIONS_FILENAME
    manual_path = run_dir / MANUAL_CORRECTIONS_FILENAME
    machine_sha = sha256_file(machine_path) if machine_path.is_file() else None
    manual_sha = sha256_file(manual_path) if manual_path.is_file() else None
    machine_events = load_machine_correction_index(
        machine_path if machine_path.is_file() else machine_path,
        digest_sha256=machine_sha,
    )
    manual_events = load_manual_correction_index(
        manual_path if manual_path.is_file() else manual_path,
        digest_sha256=manual_sha,
    )

    tracked_sha = str(accepted.tracked_pose_sha256)
    prepared_sync_sha = sha256_file(prepared_sync)
    video_sha = sha256_file(prepared_video)
    try:
        video_probe = probe_video(prepared_video)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"Failed to probe S1 prepared video for S3 QC: {exc}"
        ) from exc

    prepared_frame_count = int(sync_audit["n_prepared"])
    prepared_fps = float(sync_audit["fps_header"])
    if int(video_probe.n_frames) > 0 and int(video_probe.n_frames) != prepared_frame_count:
        raise OverlaySourceError(
            "Prepared video frame count does not match prepared_sync frame domain: "
            f"video={video_probe.n_frames} sync={prepared_frame_count}."
        )

    source_digests: dict[str, str | None] = {
        "s3_tracked_pose_sha256": tracked_sha,
        "s1_prepared_sync_sha256": prepared_sync_sha,
        "s1_prepared_video_sha256": video_sha,
        "s3_machine_corrections_sha256": machine_sha,
        "s3_manual_corrections_sha256": manual_sha,
    }

    return OverlayS3QcSource(
        s3_run_id=str(accepted.run_id),
        s3_run_dir=run_dir,
        acceptance_state=str(accepted.acceptance_state),
        tracked_pose_path=Path(accepted.tracked_pose_path).resolve(),
        tracked_pose_sha256=tracked_sha,
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
        source_s2_run_id=s2_lineage["s2_run_id"],
        source_s2_run_dir=s2_lineage["s2_run_dir"],
        source_s2_pose_parquet_path=s2_lineage["pose_parquet"],
        machine_corrections_path=machine_path if machine_path.is_file() else None,
        manual_corrections_path=manual_path if manual_path.is_file() else None,
        machine_events=machine_events,
        manual_events=manual_events,
        run_meta=dict(run_meta),
        source_digests=source_digests,
        provenance={
            "source_kind": "s3_accepted_tracked_pose",
            "s3_run_id": str(accepted.run_id),
            "s3_run_dir": str(run_dir),
            "acceptance_state": str(accepted.acceptance_state),
            "identity_semantics": "s3_accepted_tracked_identities",
            "identity_note": (
                "M{id} colors/labels are S3-accepted tracked identities from "
                "tracked_pose.parquet — not provisional S2 predictions and not "
                "working-pose edits."
            ),
            "source_s2_run_id": s2_lineage["s2_run_id"],
            "source_s2_run_dir": (
                None
                if s2_lineage["s2_run_dir"] is None
                else str(s2_lineage["s2_run_dir"])
            ),
            "prepared_sync": str(prepared_sync),
            "prepared_video": str(prepared_video),
            "timing_authority": "s1_prepared_frame_identity",
            "timing_note": (
                "S3 tracked pose is in the prepared-frame domain. QC uses S1 "
                "prepared-frame indices and prepared-video FPS. SyncMap is a "
                "1:1 prepared-identity map for compositor reuse — not S4 dense "
                "120 Hz duplication."
            ),
            "skeleton_source": skeleton_source,
            "skeleton_mode": "topology_edges" if skeleton_edges else "nodes_only",
            "skeleton_nodes_only": not bool(skeleton_edges),
            "skeleton_edge_count": len(skeleton_edges),
            "eligibility": "validate_accepted_s3_run",
            "corrections": {
                "machine_status": machine_events.status,
                "manual_status": manual_events.status,
            },
        },
    )


def verify_overlay_s3_qc_source_integrity(source: OverlayS3QcSource) -> None:
    """Recheck accepted-run eligibility and digests before publication."""

    try:
        validate_accepted_s3_run(source.s3_run_dir)
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(
            f"S3 accepted-run eligibility failed during integrity recheck: {exc}"
        ) from exc

    live: dict[str, str | None] = {
        "s3_tracked_pose_sha256": (
            sha256_file(source.tracked_pose_path)
            if source.tracked_pose_path.is_file()
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
        "s3_machine_corrections_sha256": (
            sha256_file(source.machine_corrections_path)
            if source.machine_corrections_path is not None
            and source.machine_corrections_path.is_file()
            else None
        ),
        "s3_manual_corrections_sha256": (
            sha256_file(source.manual_corrections_path)
            if source.manual_corrections_path is not None
            and source.manual_corrections_path.is_file()
            else None
        ),
    }
    for key, expected in source.source_digests.items():
        if expected is None or key not in live:
            continue
        actual = live.get(key)
        if actual is None or str(actual) != str(expected):
            raise OverlaySourceError(
                f"Overlay S3 QC source digest mismatch for {key}: "
                f"pinned={expected!r} live={actual!r}."
            )


def _build_prepared_identity_sync(
    prepared_sync_path: Path,
) -> tuple[SyncMap, dict[str, Any]]:
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
    edges = resolve_review_skeleton_edges(pose_slp)
    if pose_slp is not None and edges:
        return edges, f"s2_pose_slp:{pose_slp}"
    if pose_slp is not None:
        return (), f"unresolved:{pose_slp}"
    return (), None


def _resolve_s1_paths(
    *,
    session_root: Path,
    run_dir: Path,
    input_block: Mapping[str, Any],
) -> dict[str, Path]:
    layout_preprocess = session_root / "preprocess"
    layout = {
        "session_root": session_root,
        "preprocess_dir": layout_preprocess,
        "prepared_video": layout_preprocess / "prepared_video.mp4",
        "prepare_meta": layout_preprocess / "prepare_meta.json",
        "prepared_sync": layout_preprocess / "prepared_sync.npz",
    }
    if layout["prepared_video"].is_file() and layout["prepared_sync"].is_file():
        return layout

    if any(_optional_text(input_block.get(key)) for key in _INPUT_KEYS):
        missing = [
            key
            for key in _INPUT_KEYS
            if _optional_text(input_block.get(key)) is None
        ]
        if missing:
            raise OverlaySourceError(
                "S3 run_meta.json does not record the S1 handoff paths: "
                + ", ".join(missing)
            )
        return {
            "session_root": _resolve_recorded_path(
                str(input_block["session_root"]), run_dir
            ),
            "preprocess_dir": _resolve_recorded_path(
                str(input_block["preprocess_dir"]), run_dir
            ),
            "prepared_video": _resolve_recorded_path(
                str(input_block["prepared_video"]), run_dir
            ),
            "prepare_meta": _resolve_recorded_path(
                str(input_block["prepare_meta"]), run_dir
            ),
            "prepared_sync": _resolve_recorded_path(
                str(input_block["prepared_sync"]), run_dir
            ),
        }

    raise OverlaySourceError(
        "S1 prepared video and timing cannot be resolved for S3 QC "
        f"(layout missing under {layout_preprocess} and run_meta input paths absent)."
    )


def _resolve_s2_lineage(
    *,
    run_dir: Path,
    input_block: Mapping[str, Any],
) -> dict[str, Any]:
    s2_run_id = _optional_text(input_block.get("s2_run_id"))
    s2_run_dir_raw = _optional_text(input_block.get("s2_run_dir"))
    pose_raw = _optional_text(input_block.get("pose_parquet"))
    s2_run_dir: Path | None = None
    pose_parquet: Path | None = None
    if s2_run_dir_raw is not None:
        candidate = _resolve_recorded_path(s2_run_dir_raw, run_dir)
        if candidate.is_dir():
            s2_run_dir = candidate
        else:
            raise OverlaySourceError(
                f"Recorded S2 run directory from S3 lineage is missing: {candidate}"
            )
    if pose_raw is not None:
        candidate = _resolve_recorded_path(pose_raw, run_dir)
        if candidate.is_file():
            pose_parquet = candidate
        else:
            raise OverlaySourceError(
                f"Recorded S2 pose.parquet from S3 lineage is missing: {candidate}"
            )
    if s2_run_id is None and s2_run_dir is not None:
        s2_run_id = s2_run_dir.name
    return {
        "s2_run_id": s2_run_id,
        "s2_run_dir": s2_run_dir,
        "pose_parquet": pose_parquet,
    }


def _resolve_recorded_path(raw: str, run_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return (run_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise OverlaySourceError(f"Failed to read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OverlaySourceError(f"Expected JSON object at {path}.")
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
