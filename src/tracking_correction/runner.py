"""Subsystem 03 orchestration: validate S2 handoff and run the current-lab backend.

This module does not contain tracking heuristics. Automatic correction is
delegated to ``legacy_backend`` after handoff validation succeeds.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml

from preprocess.exceptions import SyncValidationError
from preprocess.sync_writer import load_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_NOT_ACCEPTED,
    BACKEND_STATUS_COMPLETED,
    BACKEND_STATUS_FAILED,
    BACKEND_STATUS_NOT_RUN,
    BACKEND_STATUS_RUNNING,
    CURRENT_BACKEND_ID,
    CURRENT_PROFILE_ID,
    EXPECTED_TRACK_INDICES,
    FINAL_TRACKED_POSE_FILENAME,
    MACHINE_CORRECTIONS_FILENAME,
    REQUIRED_POSE_COLUMNS,
    REQUIRED_PROFILE_NODES,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION,
    STATUS_CORRECTION_COMPLETE,
    STATUS_CORRECTION_FAILED,
    STATUS_DRY_RUN_COMPLETE,
    WORKING_TRACKED_POSE_FILENAME,
    BackendResult,
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    TrackingCorrectionResult,
)
from tracking_correction.legacy_backend import load_current_profile_config, run_current_lab_backend

_TRACK_LABEL = re.compile(r"^(?:track_)?(\d+)$", re.IGNORECASE)
_INCOMPLETE_S2_STATUSES = {
    "canceled",
    "cancelled",
    "dry_run",
    "dry_run_complete",
    "export_failed",
    "incomplete",
    "inference_failed",
    "not_generated",
    "pending",
    "running",
    "validation_failed",
}
_POSE_META_INPUT_KEYS = (
    "session_root",
    "preprocess_dir",
    "prepared_video",
    "prepare_meta",
    "prepared_sync",
)


@dataclass(frozen=True)
class _PoseInspection:
    """Concrete pose-table facts needed by the shell. No dataframe is retained."""

    rows: int
    frame_min: int
    frame_max: int
    track_labels: tuple[str, ...]
    track_indices: tuple[int, ...]
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedHandoff:
    """Read-only S2/S1 inputs resolved for one S3 workspace."""

    s2_run_dir: Path
    s2_run_id: str | None
    pose_parquet: Path
    session_root: Path
    preprocess_dir: Path
    prepared_video: Path
    prepare_meta: Path
    prepared_sync: Path
    provenance_source: str
    pose: _PoseInspection
    frame_count: int
    fps_header: float


@dataclass(frozen=True)
class _RunPaths:
    run_dir: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path
    working_tracked_pose: Path
    machine_corrections: Path
    final_tracked_pose: Path


def selected_profile_id() -> str:
    """Return the only MVP tracking profile.

    Profile choice is an internal boundary. The MVP does not expose a
    profile-selection UI.
    """

    return CURRENT_PROFILE_ID


def validate_s2_handoff(s2_run_dir: Path | str) -> Path:
    """Validate an S2 run for Subsystem 3 using the same checks as the CLI runner.

    Returns the resolved S2 run directory. Raises ``TrackingCorrectionError`` when
    the run is incomplete, missing pose data, or cannot resolve S1 inputs.
    """

    return _validate_handoff(Path(s2_run_dir)).s2_run_dir


def run_tracking_correction(request: TrackingCorrectionRequest) -> TrackingCorrectionResult:
    """Validate a completed S2 run, then run the current-lab corrector unless dry-run.

    Dry-run validates the handoff and writes S3 records with ``backend_status=not_run``.
    Live runs invoke the corrector only after handoff validation succeeds.
    """

    handoff = _validate_handoff(request.s2_run_dir)
    purpose = request.run_purpose.strip()
    if not purpose:
        raise TrackingCorrectionError("run_purpose must not be empty.")

    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    profile_id = selected_profile_id()
    run_id = f"{profile_id}__{timestamp}"
    paths = _resolve_run_paths(
        session_root=handoff.session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            handoff.s2_run_dir,
            handoff.preprocess_dir,
            handoff.prepared_video,
            handoff.prepare_meta,
            handoff.prepared_sync,
            handoff.pose_parquet,
        ),
    )
    if paths.run_dir.exists():
        raise TrackingCorrectionError(f"S3 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrackingCorrectionError(
            f"S3 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    if request.dry_run:
        completed_at = _iso_now()
        status = STATUS_DRY_RUN_COMPLETE
        backend_status = BACKEND_STATUS_NOT_RUN
        _persist_records(
            paths=paths,
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            dry_run=True,
            handoff=handoff,
            backend_status=backend_status,
            backend_result=None,
            error_message=None,
            corrector_invoked=False,
            profile_config=None,
        )
        return TrackingCorrectionResult(
            success=True,
            status=status,
            run_id=run_id,
            run_dir=paths.run_dir,
            run_meta_path=paths.run_meta,
            settings_used_path=paths.settings_used,
            processing_log_path=paths.processing_log,
            profile_id=profile_id,
            backend_id=CURRENT_BACKEND_ID,
            backend_status=backend_status,
            pose_parquet_path=handoff.pose_parquet,
            prepared_video_path=handoff.prepared_video,
        )

    profile_config = load_current_profile_config()
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=None,
            status=BACKEND_STATUS_RUNNING,
            dry_run=False,
            handoff=handoff,
            paths=paths,
            backend_status=BACKEND_STATUS_RUNNING,
            backend_result=None,
            error_message=None,
        ),
    )
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            handoff=handoff,
            dry_run=False,
            corrector_invoked=True,
            backend_status=BACKEND_STATUS_RUNNING,
            profile_config=profile_config,
        ),
    )

    backend_result: BackendResult | None = None
    error_message: str | None = None
    try:
        backend_result = run_current_lab_backend(
            pose_parquet=handoff.pose_parquet,
            output_dir=paths.run_dir,
            fps=float(profile_config.get("fps", handoff.fps_header)),
            profile_config=profile_config,
            show_progress=False,
        )
        if paths.final_tracked_pose.exists():
            raise TrackingCorrectionError(
                "Backend must not write final tracked_pose.parquet before acceptance."
            )
        status = STATUS_CORRECTION_COMPLETE
        backend_status = BACKEND_STATUS_COMPLETED
        success = True
    except Exception as exc:
        status = STATUS_CORRECTION_FAILED
        backend_status = BACKEND_STATUS_FAILED
        success = False
        error_message = str(exc)
        if paths.final_tracked_pose.exists():
            with suppress(OSError):
                paths.final_tracked_pose.unlink()

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        dry_run=False,
        handoff=handoff,
        backend_status=backend_status,
        backend_result=backend_result,
        error_message=error_message,
        corrector_invoked=True,
        profile_config=profile_config,
    )
    if not success:
        raise TrackingCorrectionError(error_message or "Current-lab backend failed.")

    assert backend_result is not None
    return TrackingCorrectionResult(
        success=True,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        profile_id=profile_id,
        backend_id=CURRENT_BACKEND_ID,
        backend_status=backend_status,
        pose_parquet_path=handoff.pose_parquet,
        prepared_video_path=handoff.prepared_video,
        working_tracked_pose_path=backend_result.working_tracked_pose_path,
        machine_corrections_path=backend_result.machine_corrections_path,
        correction_summary=dict(backend_result.correction_summary),
    )


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    dry_run: bool,
    handoff: _ResolvedHandoff,
    backend_status: str,
    backend_result: BackendResult | None,
    error_message: str | None,
    corrector_invoked: bool,
    profile_config: Mapping[str, Any] | None,
) -> None:
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            dry_run=dry_run,
            handoff=handoff,
            paths=paths,
            backend_status=backend_status,
            backend_result=backend_result,
            error_message=error_message,
        ),
    )
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            handoff=handoff,
            dry_run=dry_run,
            corrector_invoked=corrector_invoked,
            backend_status=backend_status,
            profile_config=profile_config,
        ),
    )
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        dry_run=dry_run,
        handoff=handoff,
        paths=paths,
        backend_status=backend_status,
        backend_result=backend_result,
        error_message=error_message,
        corrector_invoked=corrector_invoked,
    )


def _validate_handoff(s2_run_dir: Path) -> _ResolvedHandoff:
    run_dir = Path(s2_run_dir).expanduser()
    if not run_dir.exists():
        raise TrackingCorrectionError(f"S2 run does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise TrackingCorrectionError(f"S2 run must be a directory: {run_dir}")
    run_dir = run_dir.resolve()

    pose_parquet = run_dir / "pose.parquet"
    if not pose_parquet.is_file():
        raise TrackingCorrectionError(f"pose.parquet does not exist: {pose_parquet}")

    pose_meta = _load_optional_object(run_dir / "pose_meta.json", "S2 pose_meta.json")
    _reject_incomplete_s2_run(run_dir, pose_meta)
    provenance = _resolve_s1_provenance(run_dir, pose_meta)
    _require_prepared_video(provenance.prepared_video)
    frame_count, fps_header = _resolve_timing(provenance.prepare_meta, provenance.prepared_sync)
    pose = _inspect_pose_parquet(pose_parquet, frame_count)
    return _ResolvedHandoff(
        s2_run_dir=run_dir,
        s2_run_id=_optional_text(pose_meta.get("run_id")) if pose_meta else None,
        pose_parquet=pose_parquet,
        session_root=provenance.session_root,
        preprocess_dir=provenance.preprocess_dir,
        prepared_video=provenance.prepared_video,
        prepare_meta=provenance.prepare_meta,
        prepared_sync=provenance.prepared_sync,
        provenance_source=provenance.source,
        pose=pose,
        frame_count=frame_count,
        fps_header=fps_header,
    )


@dataclass(frozen=True)
class _S1Provenance:
    source: str
    session_root: Path
    preprocess_dir: Path
    prepared_video: Path
    prepare_meta: Path
    prepared_sync: Path


def _reject_incomplete_s2_run(
    run_dir: Path,
    pose_meta: Mapping[str, Any] | None,
) -> None:
    status = _optional_text(pose_meta.get("status")) if pose_meta else None
    dry_run = _optional_bool(pose_meta.get("dry_run")) if pose_meta else None
    if status is None or dry_run is None:
        manifest = _load_optional_object(run_dir / "job_manifest.yaml", "S2 job_manifest.yaml")
        if manifest is not None:
            if status is None:
                status = _optional_text(manifest.get("status"))
            if dry_run is None:
                dry_run = _optional_bool(manifest.get("dry_run"))
    if dry_run is True or _status_is_incomplete(status):
        recorded = status if status is not None else "unknown"
        raise TrackingCorrectionError(
            "S2 run is not a completed pose-inference run: "
            f"status={recorded}, dry_run={dry_run is True}."
        )


def _status_is_incomplete(status: str | None) -> bool:
    if status is None:
        return False
    return status.strip().lower() in _INCOMPLETE_S2_STATUSES


def _resolve_s1_provenance(
    run_dir: Path,
    pose_meta: Mapping[str, Any] | None,
) -> _S1Provenance:
    recorded = pose_meta.get("input") if pose_meta is not None else None
    if isinstance(recorded, Mapping) and any(key in recorded for key in _POSE_META_INPUT_KEYS):
        return _provenance_from_pose_meta(run_dir, recorded)
    return _provenance_from_layout(run_dir)


def _provenance_from_pose_meta(run_dir: Path, recorded: Mapping[str, Any]) -> _S1Provenance:
    missing = [key for key in _POSE_META_INPUT_KEYS if not _optional_text(recorded.get(key))]
    if missing:
        raise TrackingCorrectionError(
            "S2 pose_meta.json does not record the S1 handoff paths: " + ", ".join(missing)
        )
    prepared_video = _resolve_recorded_path(recorded["prepared_video"], run_dir)
    prepare_meta = _resolve_recorded_path(recorded["prepare_meta"], run_dir)
    prepared_sync = _resolve_recorded_path(recorded["prepared_sync"], run_dir)
    session_root = _resolve_recorded_path(recorded["session_root"], run_dir)
    preprocess_dir = _resolve_recorded_path(recorded["preprocess_dir"], run_dir)
    if not session_root.is_dir():
        raise TrackingCorrectionError(
            f"S2 pose_meta.json session_root does not exist: {session_root}"
        )
    return _S1Provenance(
        source="pose_meta",
        session_root=session_root,
        preprocess_dir=preprocess_dir,
        prepared_video=prepared_video,
        prepare_meta=prepare_meta,
        prepared_sync=prepared_sync,
    )


def _provenance_from_layout(run_dir: Path) -> _S1Provenance:
    if run_dir.parent.name != "pose_inference":
        raise TrackingCorrectionError(
            "S1 prepared video and timing cannot be resolved from S2 provenance. "
            f"pose_meta.json input paths are missing and {run_dir} is not under "
            "a pose_inference directory."
        )
    session_root = run_dir.parent.parent
    preprocess_dir = session_root / "preprocess"
    if not session_root.is_dir():
        raise TrackingCorrectionError(f"S2 session root does not exist: {session_root}")
    return _S1Provenance(
        source="s2_run_layout",
        session_root=session_root,
        preprocess_dir=preprocess_dir,
        prepared_video=preprocess_dir / "prepared_video.mp4",
        prepare_meta=preprocess_dir / "prepare_meta.json",
        prepared_sync=preprocess_dir / "prepared_sync.npz",
    )


def _require_prepared_video(path: Path) -> None:
    if not path.is_file():
        raise TrackingCorrectionError(f"Prepared video does not exist: {path}")
    try:
        with path.open("rb") as stream:
            if not stream.read(1):
                raise TrackingCorrectionError(f"Prepared video is unreadable: {path}")
    except TrackingCorrectionError:
        raise
    except OSError as exc:
        raise TrackingCorrectionError(f"Prepared video is unreadable: {path}: {exc}") from exc


def _resolve_timing(prepare_meta_path: Path, prepared_sync_path: Path) -> tuple[int, float]:
    prepare_meta = _load_timing_meta(prepare_meta_path)
    if not prepared_sync_path.is_file():
        raise TrackingCorrectionError(
            "Timing metadata cannot be resolved: "
            f"prepared_sync.npz does not exist: {prepared_sync_path}"
        )
    try:
        sync = load_prepared_sync_npz(prepared_sync_path)
    except (OSError, SyncValidationError) as exc:
        raise TrackingCorrectionError(f"Timing metadata cannot be resolved: {exc}") from exc

    frame_count = int(sync.frame_count_used_for_sleap)
    fps_header = float(sync.fps_header)
    if frame_count <= 0 or not math.isfinite(fps_header) or fps_header <= 0:
        raise TrackingCorrectionError(
            "Timing metadata cannot be resolved: prepared_sync.npz does not contain "
            "a positive frame count and fps_header."
        )
    _reject_meta_frame_count_mismatch(prepare_meta, frame_count)
    return frame_count, fps_header


def _load_timing_meta(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise TrackingCorrectionError(
            "Timing metadata cannot be resolved: "
            f"prepare_meta.json does not exist: {path}"
        )
    payload = _load_optional_object(path, "S1 prepare_meta.json")
    if payload is None:
        raise TrackingCorrectionError(
            f"Timing metadata cannot be resolved: prepare_meta.json is unreadable: {path}"
        )
    return payload


def _reject_meta_frame_count_mismatch(prepare_meta: Mapping[str, Any], frame_count: int) -> None:
    prepared = prepare_meta.get("prepared_video")
    if not isinstance(prepared, Mapping):
        return
    recorded = prepared.get("frame_count_used_for_sleap")
    if isinstance(recorded, bool) or not isinstance(recorded, int):
        return
    if recorded != frame_count:
        raise TrackingCorrectionError(
            "Pose frame domain cannot be reconciled: prepare_meta.json "
            f"frame_count_used_for_sleap={recorded} does not match prepared_sync.npz "
            f"frame_count_used_for_sleap={frame_count}."
        )


def _inspect_pose_parquet(path: Path, frame_count: int) -> _PoseInspection:
    frame = _read_pose_frame(path)
    missing = [column for column in REQUIRED_POSE_COLUMNS if column not in frame.columns]
    if missing:
        raise TrackingCorrectionError(
            "pose.parquet is missing required columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise TrackingCorrectionError("pose.parquet contains no pose rows.")

    frame_idx = _integer_series(frame["frame_idx"], "frame_idx")
    _reject_frame_domain_mismatch(frame, frame_idx, frame_count)
    _reject_extra_video_domain(frame)
    track_indices, track_labels = _validated_track_indices(frame["track"])
    nodes = _validated_nodes(frame["node"])
    _require_numeric_column(frame["x"], "x")
    _require_numeric_column(frame["y"], "y")
    _require_numeric_column(frame["score"], "score")
    _reject_duplicate_pose_keys(frame_idx, track_indices, frame["node"])

    unique_indices = tuple(sorted(set(track_indices)))
    unique_nodes = tuple(sorted(set(nodes)))
    return _PoseInspection(
        rows=int(len(frame_idx)),
        frame_min=int(min(frame_idx)),
        frame_max=int(max(frame_idx)),
        track_labels=tuple(sorted(set(track_labels))),
        track_indices=unique_indices,
        nodes=unique_nodes,
    )


def _read_pose_frame(path: Path) -> Any:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise TrackingCorrectionError(f"Could not read pose.parquet: {exc}") from exc


def _integer_series(values: Any, column: str) -> list[int]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise TrackingCorrectionError(
            f"pose.parquet column {column} has missing or non-numeric values."
        )
    rounded = numeric.round()
    if not numeric.eq(rounded).all():
        raise TrackingCorrectionError(
            f"pose.parquet column {column} has non-integer values."
        )
    return [int(value) for value in rounded.tolist()]


def _reject_frame_domain_mismatch(
    frame: Any,
    frame_idx: Sequence[int],
    frame_count: int,
) -> None:
    invalid = sorted({index for index in frame_idx if index < 0 or index >= frame_count})
    if invalid:
        raise TrackingCorrectionError(
            "Pose frame indices are inconsistent with the inherited S1/S2 "
            f"prepared-frame domain 0..{frame_count - 1}: {_preview(invalid)}."
        )
    if "prepared_frame_idx" not in frame.columns:
        return
    recorded = _integer_series(frame["prepared_frame_idx"], "prepared_frame_idx")
    mismatched = sorted(
        {
            frame_idx[position]
            for position, prepared_idx in enumerate(recorded)
            if prepared_idx != frame_idx[position]
        }
    )
    if mismatched:
        raise TrackingCorrectionError(
            "Pose frame indices are inconsistent with the inherited S1/S2 "
            f"prepared-frame mapping: {_preview(mismatched)}."
        )


def _reject_extra_video_domain(frame: Any) -> None:
    if "video_index" not in frame.columns:
        return
    video_indices = _integer_series(frame["video_index"], "video_index")
    unexpected = sorted({index for index in video_indices if index != 0})
    if unexpected:
        raise TrackingCorrectionError(
            "The current MVP profile assumes one prepared-video frame domain, "
            f"but pose.parquet video_index includes {_preview(unexpected)}."
        )


def _validated_track_indices(values: Any) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    labels: list[str] = []
    unparsed: list[str] = []
    for value in values.tolist():
        parsed = _parse_track_index(value)
        label = str(value).strip()
        if parsed is None:
            unparsed.append(label)
            continue
        indices.append(parsed)
        labels.append(label)
    if unparsed:
        raise TrackingCorrectionError(
            "The current MVP profile cannot verify two-animal tracks because "
            "pose.parquet contains non-index track labels: "
            + ", ".join(sorted(set(unparsed)))
            + "."
        )
    observed = set(indices)
    expected = set(EXPECTED_TRACK_INDICES)
    if observed != expected:
        raise TrackingCorrectionError(
            "The current MVP profile assumes exactly two tracks with indices "
            f"{list(EXPECTED_TRACK_INDICES)}, but pose.parquet has "
            f"{sorted(observed)}."
        )
    return indices, labels


def _parse_track_index(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, str | bytes):
        try:
            unpacked = item()
        except (TypeError, ValueError):
            unpacked = value
        if unpacked is not value:
            return _parse_track_index(unpacked)
    match = _TRACK_LABEL.fullmatch(str(value).strip())
    if match is None:
        return None
    return int(match.group(1))


def _validated_nodes(values: Any) -> list[str]:
    nodes: list[str] = []
    for value in values.tolist():
        if value is None:
            raise TrackingCorrectionError("pose.parquet contains an empty node name.")
        node = str(value).strip()
        if not node or node.lower() == "nan":
            raise TrackingCorrectionError("pose.parquet contains an empty node name.")
        nodes.append(node)
    missing = [node for node in REQUIRED_PROFILE_NODES if node not in set(nodes)]
    if missing:
        raise TrackingCorrectionError(
            "The current MVP profile assumes skeleton nodes that are absent from "
            "pose.parquet: " + ", ".join(missing) + "."
        )
    return nodes


def _require_numeric_column(values: Any, column: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    original_missing = values.isna()
    if (~original_missing & numeric.isna()).any():
        raise TrackingCorrectionError(
            f"pose.parquet column {column} has non-numeric values."
        )


def _reject_duplicate_pose_keys(
    frame_idx: Sequence[int],
    track_indices: Sequence[int],
    nodes: Any,
) -> None:
    node_labels = [str(node).strip() for node in nodes.tolist()]
    keys = list(zip(frame_idx, track_indices, node_labels, strict=True))
    if len(keys) != len(set(keys)):
        raise TrackingCorrectionError(
            "pose.parquet contains duplicate (frame_idx, track, node) rows."
        )


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / "tracking_correction"
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        run_meta=run_dir / "run_meta.json",
        settings_used=run_dir / "settings_used.yaml",
        processing_log=run_dir / "processing_log.txt",
        working_tracked_pose=run_dir / WORKING_TRACKED_POSE_FILENAME,
        machine_corrections=run_dir / MACHINE_CORRECTIONS_FILENAME,
        final_tracked_pose=run_dir / FINAL_TRACKED_POSE_FILENAME,
    )


def _reject_output_overlap(run_dir: Path, protected: Sequence[Path]) -> None:
    resolved_run = run_dir.resolve()
    for protected_path in protected:
        resolved_protected = protected_path.resolve()
        if (
            resolved_run == resolved_protected
            or _is_relative_to(resolved_run, resolved_protected)
            or _is_relative_to(resolved_protected, resolved_run)
        ):
            raise TrackingCorrectionError(
                "S3 run directory must not overlap S1 or S2 artifacts: "
                f"{resolved_run}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _run_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    dry_run: bool,
    handoff: _ResolvedHandoff,
    paths: _RunPaths,
    backend_status: str,
    backend_result: BackendResult | None,
    error_message: str | None,
) -> dict[str, Any]:
    working_path = (
        _path_text(backend_result.working_tracked_pose_path)
        if backend_result is not None
        else None
    )
    corrections_path = (
        _path_text(backend_result.machine_corrections_path)
        if backend_result is not None
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "dry_run": dry_run,
        "acceptance_state": ACCEPTANCE_NOT_ACCEPTED,
        "profile_id": selected_profile_id(),
        "backend_id": CURRENT_BACKEND_ID,
        "backend_status": backend_status,
        "provenance_source": handoff.provenance_source,
        "input": {
            "s2_run_dir": _path_text(handoff.s2_run_dir),
            "s2_run_id": handoff.s2_run_id,
            "pose_parquet": _path_text(handoff.pose_parquet),
            "session_root": _path_text(handoff.session_root),
            "preprocess_dir": _path_text(handoff.preprocess_dir),
            "prepared_video": _path_text(handoff.prepared_video),
            "prepare_meta": _path_text(handoff.prepare_meta),
            "prepared_sync": _path_text(handoff.prepared_sync),
        },
        "timing": {
            "source": "prepared_sync.npz",
            "frame_count_used_for_sleap": handoff.frame_count,
            "fps_header": handoff.fps_header,
        },
        "pose_summary": {
            "rows": handoff.pose.rows,
            "frame_min": handoff.pose.frame_min,
            "frame_max": handoff.pose.frame_max,
            "track_labels": list(handoff.pose.track_labels),
            "track_indices": list(handoff.pose.track_indices),
            "nodes": list(handoff.pose.nodes),
        },
        "outputs": {
            "working_tracked_pose_parquet": working_path,
            "machine_corrections_json": corrections_path,
            "tracked_pose_parquet": None,
            "settings_used": _path_text(paths.settings_used),
            "processing_log": _path_text(paths.processing_log),
        },
        "correction_summary": (
            dict(backend_result.correction_summary) if backend_result is not None else None
        ),
    }
    if error_message is not None:
        payload["error"] = error_message
    return payload


def _settings_payload(
    *,
    handoff: _ResolvedHandoff,
    dry_run: bool,
    corrector_invoked: bool,
    backend_status: str,
    profile_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "profile_id": selected_profile_id(),
        "backend_id": CURRENT_BACKEND_ID,
        "backend_status": backend_status,
        "dry_run": dry_run,
        "corrector_invoked": corrector_invoked,
        "acceptance_state": ACCEPTANCE_NOT_ACCEPTED,
        "pose_parquet": _path_text(handoff.pose_parquet),
        "prepared_video": _path_text(handoff.prepared_video),
        "prepare_meta": _path_text(handoff.prepare_meta),
        "prepared_sync": _path_text(handoff.prepared_sync),
        "fps_header": handoff.fps_header,
        "frame_count_used_for_sleap": handoff.frame_count,
        "required_pose_columns": list(REQUIRED_POSE_COLUMNS),
        "required_nodes": list(REQUIRED_PROFILE_NODES),
        "expected_track_indices": list(EXPECTED_TRACK_INDICES),
        "working_tracked_pose_filename": WORKING_TRACKED_POSE_FILENAME,
        "machine_corrections_filename": MACHINE_CORRECTIONS_FILENAME,
        "final_tracked_pose_filename": FINAL_TRACKED_POSE_FILENAME,
    }
    if profile_config is not None:
        payload["profile_config"] = dict(profile_config)
    return payload


def _write_processing_log(
    *,
    path: Path,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    dry_run: bool,
    handoff: _ResolvedHandoff,
    paths: _RunPaths,
    backend_status: str,
    backend_result: BackendResult | None,
    error_message: str | None,
    corrector_invoked: bool,
) -> None:
    lines = [
        f"start_time: {created_at}",
        f"completed_time: {completed_at}",
        f"status: {status}",
        f"dry_run: {dry_run}",
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"profile_id: {selected_profile_id()}",
        f"backend_id: {CURRENT_BACKEND_ID}",
        f"backend_status: {backend_status}",
        f"corrector_invoked: {corrector_invoked}",
        f"acceptance_state: {ACCEPTANCE_NOT_ACCEPTED}",
        f"provenance_source: {handoff.provenance_source}",
        f"s2_run_dir: {_path_text(handoff.s2_run_dir)}",
        f"pose_parquet: {_path_text(handoff.pose_parquet)}",
        f"prepared_video: {_path_text(handoff.prepared_video)}",
        f"prepare_meta: {_path_text(handoff.prepare_meta)}",
        f"prepared_sync: {_path_text(handoff.prepared_sync)}",
        "timing_source: prepared_sync.npz",
        f"frame_count_used_for_sleap: {handoff.frame_count}",
        f"fps_header: {handoff.fps_header}",
        f"pose_rows: {handoff.pose.rows}",
        f"frame_min: {handoff.pose.frame_min}",
        f"frame_max: {handoff.pose.frame_max}",
        f"track_indices: {list(handoff.pose.track_indices)}",
        f"run_dir: {_path_text(paths.run_dir)}",
        (
            "working_tracked_pose_parquet: "
            + (
                _path_text(backend_result.working_tracked_pose_path)
                if backend_result
                else "not_generated"
            )
        ),
        (
            "machine_corrections_json: "
            + (
                _path_text(backend_result.machine_corrections_path)
                if backend_result
                else "not_generated"
            )
        ),
        "tracked_pose_parquet: not_generated",
    ]
    if backend_result is not None:
        summary = backend_result.correction_summary
        lines.append(f"total_corrections: {summary.get('total_corrections', 0)}")
        lines.append(f"correction_counts_by_type: {summary.get('by_type', {})}")
        lines.append(f"blanked_node_count: {backend_result.blanked_node_count}")
        lines.append(f"swapped_frame_count: {backend_result.swapped_frame_count}")
        lines.append(f"input_row_count: {backend_result.input_row_count}")
        lines.append(f"output_row_count: {backend_result.output_row_count}")
    if error_message is not None:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _load_optional_object(path: Path, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TrackingCorrectionError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingCorrectionError(f"{label} must contain an object.")
    return payload


def _resolve_recorded_path(value: object, base: Path) -> Path:
    text = _optional_text(value)
    if text is None:
        raise TrackingCorrectionError("S2 pose_meta.json contains an empty S1 path.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _preview(values: Sequence[int], limit: int = 5) -> str:
    shown = list(values[:limit])
    suffix = ", ..." if len(values) > limit else ""
    return ", ".join(str(value) for value in shown) + suffix


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value.strip())
    if not cleaned:
        raise TrackingCorrectionError("timestamp is empty after sanitization.")
    return cleaned


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
