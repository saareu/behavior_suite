"""Subsystem 04 end-to-end runner: accepted S3 → final_pose.parquet.

Pipeline:
  validate accepted S3
  → authoritative S1 timing
  → dense 120 Hz construction + gap fill
  → Savitzky–Golay smoothing
  → QC
  → revalidate S3 acceptance
  → write final artifacts

Upstream S1/S2/S3 artifacts remain read-only. Partial/invalid outputs are never
marked completed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pose_finalization.accepted_s3 import AcceptedS3Run, validate_accepted_s3_run
from pose_finalization.contracts import (
    FINAL_POSE_FILENAME,
    POSE_FINALIZATION_DIRNAME,
    PROCESSING_LOG_FILENAME,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    PoseFinalizationError,
)
from pose_finalization.dense_pose import DensePoseResult, construct_dense_pose
from pose_finalization.export import build_final_pose_table, write_final_pose_parquet
from pose_finalization.gap_fill import GapFillConfig
from pose_finalization.qc import QcConfig, QcSummary, evaluate_finalization_qc
from pose_finalization.smoothing import SmoothConfig, SmoothResult, apply_savgol_smoothing
from pose_finalization.timing import build_timing_map_from_prepared_sync


@dataclass(frozen=True, slots=True)
class PoseFinalizationRequest:
    """Inputs for one Subsystem 04 finalization run."""

    session_root: Path | None = None
    s3_run_dir: Path | None = None
    output_root: Path | None = None
    run_purpose: str = "development"
    timestamp: str | None = None
    gap_fill_config: GapFillConfig | None = None
    smooth_config: SmoothConfig | None = None
    qc_config: QcConfig | None = None


@dataclass(frozen=True, slots=True)
class PoseFinalizationResult:
    """Summary returned after an S4 run directory is recorded."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    final_pose_path: Path | None
    source_s3_run_id: str
    source_s3_run_dir: Path
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    qc: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _RunPaths:
    run_dir: Path
    final_pose: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path


def run_pose_finalization(request: PoseFinalizationRequest) -> PoseFinalizationResult:
    """Run the Sprint 3 pose-finalization pipeline and persist S4 artifacts."""

    purpose = request.run_purpose.strip()
    if not purpose:
        raise PoseFinalizationError("run_purpose must not be empty.")

    accepted = _resolve_input(request)
    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    run_id = f"pose_finalization__{timestamp}"
    session_root = accepted.session_root
    paths = _resolve_run_paths(
        session_root=session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            accepted.run_dir,
            accepted.tracked_pose_path,
            accepted.prepared_sync_path,
            accepted.session_root / "preprocess",
            accepted.session_root / "pose_inference",
            accepted.session_root / "tracking_correction",
        ),
    )
    if paths.run_dir.exists():
        raise PoseFinalizationError(f"S4 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PoseFinalizationError(
            f"S4 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    gap_cfg = request.gap_fill_config if request.gap_fill_config is not None else GapFillConfig()
    smooth_cfg = (
        request.smooth_config if request.smooth_config is not None else SmoothConfig()
    )
    qc_cfg = request.qc_config if request.qc_config is not None else QcConfig()

    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            accepted=accepted,
            paths=paths,
            gap_cfg=gap_cfg,
            smooth_cfg=smooth_cfg,
            qc_cfg=qc_cfg,
            qc_summary=None,
            error_message=None,
            final_pose_written=False,
        ),
    )
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            accepted=accepted,
            gap_cfg=gap_cfg,
            smooth_cfg=smooth_cfg,
            qc_cfg=qc_cfg,
            status=STATUS_RUNNING,
        ),
    )

    dense: DensePoseResult | None = None
    smooth: SmoothResult | None = None
    qc_summary: QcSummary | None = None
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    final_pose_written = False

    try:
        timing = build_timing_map_from_prepared_sync(accepted.prepared_sync_path)
        dense = construct_dense_pose(
            accepted.tracked_pose_path,
            timing,
            gap_fill_config=gap_cfg,
            apply_gap_fill=True,
        )
        smooth = apply_savgol_smoothing(
            x=dense.x,
            y=dense.y,
            protected=dense.protected,
            config=smooth_cfg,
        )
        table = build_final_pose_table(dense, smooth)
        qc_summary = evaluate_finalization_qc(
            dense=dense,
            smooth=smooth,
            final_table=table,
            config=qc_cfg,
        )
        if not qc_summary.structural_ok:
            raise PoseFinalizationError(
                "S4 structural QC failed: " + "; ".join(qc_summary.failures)
            )

        # Revalidate immediately before completion so a superseded/mutated S3
        # acceptance cannot produce a completed S4 artifact.
        revalidated = validate_accepted_s3_run(accepted.run_dir)
        if revalidated.tracked_pose_sha256 != accepted.tracked_pose_sha256:
            raise PoseFinalizationError(
                "Accepted S3 tracked_pose.parquet changed during S4 processing."
            )

        write_final_pose_parquet(paths.final_pose, table)
        final_pose_written = True

        if qc_summary.has_warnings:
            status = STATUS_COMPLETED_WITH_WARNINGS
            warnings = qc_summary.warnings
        else:
            status = STATUS_COMPLETED
            warnings = ()
        success = True
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        if paths.final_pose.exists():
            with suppress(OSError):
                paths.final_pose.unlink()
        final_pose_written = False

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        accepted=accepted,
        gap_cfg=gap_cfg,
        smooth_cfg=smooth_cfg,
        qc_cfg=qc_cfg,
        qc_summary=qc_summary,
        dense=dense,
        smooth=smooth,
        error_message=error_message,
        final_pose_written=final_pose_written,
    )

    if not success:
        raise PoseFinalizationError(error_message or "Pose finalization failed.")

    return PoseFinalizationResult(
        success=True,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        final_pose_path=paths.final_pose if final_pose_written else None,
        source_s3_run_id=accepted.run_id,
        source_s3_run_dir=accepted.run_dir,
        warnings=warnings,
        error_message=None,
        qc=qc_summary.to_metadata() if qc_summary is not None else None,
    )


def is_s4_run_eligible_for_downstream(s4_run_dir: Path | str) -> bool:
    """Return True when a completed S4 run still has an accepted source S3.

    A previously completed S4 run becomes ineligible when its recorded source
    S3 acceptance is superseded, SHA-mismatched, or otherwise invalid.
    """

    try:
        validate_s4_run_for_downstream(s4_run_dir)
    except PoseFinalizationError:
        return False
    return True


def validate_s4_run_for_downstream(s4_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed S4 run for downstream use (staleness-aware)."""

    run_dir = Path(s4_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise PoseFinalizationError(f"S4 run directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise PoseFinalizationError(f"S4 run_meta.json is missing: {meta_path}")
    meta = _load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise PoseFinalizationError(
            f"S4 run is not completed (status={status or 'missing'!r})."
        )

    final_pose = run_dir / FINAL_POSE_FILENAME
    if not final_pose.is_file():
        raise PoseFinalizationError(
            f"Completed S4 run is missing {FINAL_POSE_FILENAME}: {final_pose}"
        )

    source = meta.get("source_s3")
    if not isinstance(source, Mapping):
        raise PoseFinalizationError("S4 run_meta.json is missing source_s3 provenance.")
    s3_run_dir = _optional_text(source.get("run_dir"))
    if s3_run_dir is None:
        raise PoseFinalizationError("S4 run_meta.json source_s3.run_dir is missing.")

    accepted = validate_accepted_s3_run(s3_run_dir)
    recorded_sha = _optional_text(source.get("tracked_pose_sha256"))
    if recorded_sha is not None and recorded_sha.lower() != accepted.tracked_pose_sha256:
        raise PoseFinalizationError(
            "S4 source S3 tracked_pose SHA256 no longer matches the recorded digest."
        )
    return {
        "s4_run_dir": run_dir,
        "status": status,
        "source_s3_run_dir": accepted.run_dir,
        "source_s3_run_id": accepted.run_id,
        "tracked_pose_sha256": accepted.tracked_pose_sha256,
    }


def _resolve_input(request: PoseFinalizationRequest) -> AcceptedS3Run:
    if request.s3_run_dir is not None:
        return validate_accepted_s3_run(request.s3_run_dir)
    if request.session_root is None:
        raise PoseFinalizationError(
            "Either s3_run_dir or session_root is required for pose finalization."
        )
    from pose_finalization.accepted_s3 import find_latest_accepted_s3_run

    latest = find_latest_accepted_s3_run(request.session_root)
    if latest is None:
        raise PoseFinalizationError(
            "No eligible accepted S3 run was found for pose finalization."
        )
    return latest


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / POSE_FINALIZATION_DIRNAME
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        final_pose=run_dir / FINAL_POSE_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
    )


def _reject_output_overlap(run_dir: Path, protected: Sequence[Path]) -> None:
    resolved_run = run_dir.resolve()
    for protected_path in protected:
        resolved_protected = Path(protected_path).resolve()
        if (
            resolved_run == resolved_protected
            or _is_relative_to(resolved_run, resolved_protected)
            or _is_relative_to(resolved_protected, resolved_run)
        ):
            raise PoseFinalizationError(
                "S4 run directory must not overlap S1/S2/S3 artifacts: "
                f"{resolved_run}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    accepted: AcceptedS3Run,
    gap_cfg: GapFillConfig,
    smooth_cfg: SmoothConfig,
    qc_cfg: QcConfig,
    qc_summary: QcSummary | None,
    dense: DensePoseResult | None,
    smooth: SmoothResult | None,
    error_message: str | None,
    final_pose_written: bool,
) -> None:
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            accepted=accepted,
            paths=paths,
            gap_cfg=gap_cfg,
            smooth_cfg=smooth_cfg,
            qc_cfg=qc_cfg,
            qc_summary=qc_summary,
            error_message=error_message,
            final_pose_written=final_pose_written,
            dense=dense,
            smooth=smooth,
        ),
    )
    _write_yaml(
        paths.settings_used,
        _settings_payload(
            accepted=accepted,
            gap_cfg=gap_cfg,
            smooth_cfg=smooth_cfg,
            qc_cfg=qc_cfg,
            status=status,
        ),
    )
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        accepted=accepted,
        paths=paths,
        gap_cfg=gap_cfg,
        smooth_cfg=smooth_cfg,
        qc_summary=qc_summary,
        error_message=error_message,
        final_pose_written=final_pose_written,
        dense=dense,
        smooth=smooth,
    )


def _run_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    accepted: AcceptedS3Run,
    paths: _RunPaths,
    gap_cfg: GapFillConfig,
    smooth_cfg: SmoothConfig,
    qc_cfg: QcConfig,
    qc_summary: QcSummary | None,
    error_message: str | None,
    final_pose_written: bool,
    dense: DensePoseResult | None = None,
    smooth: SmoothResult | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "source_s3": {
            "run_dir": _path_text(accepted.run_dir),
            "run_id": accepted.run_id,
            "tracked_pose_parquet": _path_text(accepted.tracked_pose_path),
            "tracked_pose_sha256": accepted.tracked_pose_sha256,
            "accepted_at": accepted.accepted_at,
            "profile_id": accepted.profile_id,
            "prepared_sync": _path_text(accepted.prepared_sync_path),
            "session_root": _path_text(accepted.session_root),
        },
        "settings": {
            "gap_fill": _gap_settings(gap_cfg),
            "smoothing": _smooth_settings(smooth_cfg),
            "qc": _qc_settings(qc_cfg),
            "geometric_refinement": {
                "enabled": False,
                "status": "deferred",
                "reason": (
                    "No sufficiently justified general-purpose geometric refinement "
                    "method is available for the current-lab MVP; implanted-nose "
                    "legacy models are intentionally out of scope."
                ),
            },
        },
        "outputs": {
            "final_pose_parquet": (
                _path_text(paths.final_pose) if final_pose_written else None
            ),
            "settings_used": _path_text(paths.settings_used),
            "processing_log": _path_text(paths.processing_log),
        },
        "qc": qc_summary.to_metadata() if qc_summary is not None else None,
    }
    if dense is not None:
        payload["dense_summary"] = {
            "n_samples": dense.n_samples,
            "n_tracks": dense.n_tracks,
            "n_nodes": dense.n_nodes,
            "track_ids": [int(v) for v in dense.track_ids.tolist()],
            "node_names": [str(v) for v in dense.node_names.tolist()],
            "target_fps": float(dense.timing_map.target_fps),
            "source_fps_header": float(dense.timing_map.source_fps_header),
            "source_frame_count": int(dense.timing_map.source_frame_count),
        }
    if smooth is not None:
        payload["smoothing_summary"] = {
            "enabled": bool(smooth.config.enabled),
            "smoothed_sample_count": int(smooth.was_smoothed.sum()),
            "window_length": int(smooth.config.window_length),
            "polyorder": int(smooth.config.polyorder),
        }
    if error_message is not None:
        payload["error"] = error_message
    return payload


def _settings_payload(
    *,
    accepted: AcceptedS3Run,
    gap_cfg: GapFillConfig,
    smooth_cfg: SmoothConfig,
    qc_cfg: QcConfig,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "status": status,
        "source_s3_run_id": accepted.run_id,
        "source_s3_run_dir": _path_text(accepted.run_dir),
        "tracked_pose_sha256": accepted.tracked_pose_sha256,
        "gap_fill": _gap_settings(gap_cfg),
        "smoothing": _smooth_settings(smooth_cfg),
        "qc": _qc_settings(qc_cfg),
        "geometric_refinement": {
            "enabled": False,
            "status": "deferred",
        },
    }


def _gap_settings(cfg: GapFillConfig) -> dict[str, Any]:
    return {
        "linear_max_frames": int(cfg.linear_max_frames),
        "kalman_max_frames": int(cfg.kalman_max_frames),
        "process_var": float(cfg.process_var),
        "measurement_var": float(cfg.measurement_var),
        "excluded_nodes": list(cfg.excluded_nodes),
    }


def _smooth_settings(cfg: SmoothConfig) -> dict[str, Any]:
    return {
        "enabled": bool(cfg.enabled),
        "window_length": int(cfg.window_length),
        "polyorder": int(cfg.polyorder),
    }


def _qc_settings(cfg: QcConfig) -> dict[str, Any]:
    return {
        "missing_fraction_warn": float(cfg.missing_fraction_warn),
        "imputed_fraction_warn": float(cfg.imputed_fraction_warn),
        "jump_px_warn": float(cfg.jump_px_warn),
        "smooth_displacement_px_warn": float(cfg.smooth_displacement_px_warn),
    }


def _write_processing_log(
    *,
    path: Path,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    accepted: AcceptedS3Run,
    paths: _RunPaths,
    gap_cfg: GapFillConfig,
    smooth_cfg: SmoothConfig,
    qc_summary: QcSummary | None,
    error_message: str | None,
    final_pose_written: bool,
    dense: DensePoseResult | None,
    smooth: SmoothResult | None,
) -> None:
    lines = [
        "Subsystem 04 pose finalization",
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"created_at: {created_at}",
        f"completed_at: {completed_at}",
        f"status: {status}",
        f"source_s3_run_id: {accepted.run_id}",
        f"source_s3_run_dir: {_path_text(accepted.run_dir)}",
        f"tracked_pose_sha256: {accepted.tracked_pose_sha256}",
        f"prepared_sync: {_path_text(accepted.prepared_sync_path)}",
        f"gap_fill.linear_max_frames: {gap_cfg.linear_max_frames}",
        f"gap_fill.kalman_max_frames: {gap_cfg.kalman_max_frames}",
        f"smoothing.enabled: {smooth_cfg.enabled}",
        f"smoothing.window_length: {smooth_cfg.window_length}",
        f"smoothing.polyorder: {smooth_cfg.polyorder}",
        "geometric_refinement: deferred",
    ]
    if dense is not None:
        lines.extend(
            [
                f"dense.n_samples: {dense.n_samples}",
                f"dense.n_tracks: {dense.n_tracks}",
                f"dense.n_nodes: {dense.n_nodes}",
            ]
        )
    if smooth is not None:
        lines.append(f"smoothed_sample_count: {int(smooth.was_smoothed.sum())}")
    if qc_summary is not None:
        lines.append(f"qc.structural_ok: {qc_summary.structural_ok}")
        lines.append(f"qc.warning_count: {len(qc_summary.warnings)}")
        for warning in qc_summary.warnings:
            lines.append(f"qc.warning: {warning}")
        for failure in qc_summary.failures:
            lines.append(f"qc.failure: {failure}")
    lines.append(
        "final_pose_parquet: "
        + (_path_text(paths.final_pose) if final_pose_written else "not_written")
    )
    if error_message:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value.strip())
    if not cleaned:
        raise PoseFinalizationError("timestamp is empty after sanitization.")
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


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseFinalizationError(
            f"S4 run_meta.json is missing or malformed: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PoseFinalizationError(f"S4 run_meta.json must be a JSON object: {path}")
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
