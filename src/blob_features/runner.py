"""Subsystem 05 end-to-end runner: eligible S1 → blob_features.parquet.

Pipeline (legacy Pipeline 40 order, S1/S4 timing authority):
  validate S1 eligibility
  → decode prepared video in deterministic frame order
  → per-frame segmentation + occlusion static features (source rate)
  → short-gap cc_count imputation
  → bout-scoped temporal occlusion features (source rate)
  → nearest-neighbor map onto authoritative 120 Hz timeline
  → structural QC
  → write parquet / meta / settings / log / qc.json

Upstream S1 artifacts remain read-only. Partial or structurally invalid
parquet outputs are never marked completed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from blob_features.config import BlobFeaturesConfig, load_blob_features_config
from blob_features.contracts import (
    BLOB_FEATURES_DIRNAME,
    BLOB_FEATURES_FILENAME,
    BLOB_FEATURES_SCHEMA_VERSION,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    TARGET_FPS_DEFAULT,
    BlobFeaturesError,
)
from blob_features.export import build_blob_features_table, write_blob_features_parquet
from blob_features.features import (
    apply_source_temporal_pipeline,
    build_source_feature_table,
    compute_frame_features,
    map_source_features_to_target,
)
from blob_features.qc import QcSummary, assert_structural_qc, evaluate_blob_features_qc
from blob_features.s1_eligibility import EligibleS1Input, validate_s1_for_blob_features
from blob_features.timing import (
    NearestNeighborTimingMap,
    build_nearest_neighbor_timing_map_from_prepared_sync,
)
from preprocess.models import OperationProgress
from project.file_digest import sha256_file

ProgressCallback = Callable[[OperationProgress], None]


@dataclass(frozen=True, slots=True)
class BlobFeaturesRequest:
    """Inputs for one Subsystem 05 blob-feature extraction run."""

    session_root: Path | None = None
    preprocess_dir: Path | None = None
    output_root: Path | None = None
    config: BlobFeaturesConfig | None = None
    config_path: Path | None = None
    run_purpose: str = "development"
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class BlobFeaturesResult:
    """Summary returned after an S5 run directory is recorded."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    qc_path: Path
    blob_features_path: Path | None
    session_root: Path
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    qc: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _RunPaths:
    run_dir: Path
    blob_features: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path
    qc: Path


def run_blob_feature_extraction(
    request: BlobFeaturesRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> BlobFeaturesResult:
    """Run the Sprint 2 blob-feature pipeline and persist S5 artifacts.

    Cancellation is not supported: interrupting mid-run leaves ``status=failed``
    (or ``running`` if the process is killed) and never publishes a partial
    parquet as a completed artifact. GUI Cancel is intentionally omitted until
    a cooperative cancellation hook exists.
    """

    purpose = request.run_purpose.strip()
    if not purpose:
        raise BlobFeaturesError("run_purpose must not be empty.")

    eligible = _resolve_input(request)
    config = _resolve_config(request)
    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    run_id = f"blob_features__{timestamp}"
    paths = _resolve_run_paths(
        session_root=eligible.session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _report_progress(
        progress_callback,
        phase="starting",
        message="Validating S1 source and creating run directory",
        completed_units=0,
        total_units=None,
        is_indeterminate=True,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            eligible.preprocess_dir,
            eligible.prepared_video,
            eligible.cropped_background,
            eligible.prepared_sync,
            eligible.prepare_meta,
            eligible.session_root / "preprocess",
            eligible.session_root / "pose_inference",
            eligible.session_root / "tracking_correction",
            eligible.session_root / "pose_finalization",
        ),
    )
    if paths.run_dir.exists():
        raise BlobFeaturesError(f"S5 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise BlobFeaturesError(
            f"S5 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    provenance = _s1_provenance(eligible)
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            eligible=eligible,
            paths=paths,
            config=config,
            provenance=provenance,
            timing_map=None,
            qc_summary=None,
            frame_errors=None,
            error_message=None,
            blob_features_written=False,
        ),
    )
    _write_yaml(paths.settings_used, _settings_payload(config=config, status=STATUS_RUNNING))

    timing_map: NearestNeighborTimingMap | None = None
    qc_summary: QcSummary | None = None
    frame_errors: dict[str, Any] | None = None
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    blob_features_written = False

    try:
        _report_progress(
            progress_callback,
            phase="timing",
            message="Building 120 Hz nearest-neighbor timing map",
            completed_units=0,
            total_units=None,
            is_indeterminate=True,
        )
        timing_map = build_nearest_neighbor_timing_map_from_prepared_sync(
            eligible.prepared_sync,
            target_fps=float(config.target_fps),
        )
        background = _load_background(eligible.cropped_background)
        source_records, frame_errors = _extract_source_features(
            video_path=eligible.prepared_video,
            background=background,
            frame_count=eligible.frame_count,
            config=config,
            progress_callback=progress_callback,
        )
        _report_progress(
            progress_callback,
            phase="temporal",
            message="Applying temporal features and 120 Hz mapping",
            completed_units=eligible.frame_count,
            total_units=eligible.frame_count,
            is_indeterminate=False,
        )
        df_source = build_source_feature_table(source_records)
        df_temporal = apply_source_temporal_pipeline(
            df_source,
            max_gap_frames=config.temporal.cc_count_gap_fill_max_gap,
        )
        df_mapped = map_source_features_to_target(df_temporal, timing_map)
        table = build_blob_features_table(df_mapped, timing_map)
        qc_summary = evaluate_blob_features_qc(
            table=table,
            timing_map=timing_map,
            source_frame_count=eligible.frame_count,
            frame_errors=frame_errors,
        )
        assert_structural_qc(qc_summary)

        # Revalidate S1 eligibility immediately before publishing.
        revalidated = validate_s1_for_blob_features(eligible.session_root)
        if revalidated.prepared_sync.resolve() != eligible.prepared_sync.resolve():
            raise BlobFeaturesError("S1 prepared_sync path changed during S5 processing.")
        if _sha256(revalidated.prepared_sync) != provenance["prepared_sync_sha256"]:
            raise BlobFeaturesError(
                "S1 prepared_sync.npz changed during S5 processing (SHA256 mismatch)."
            )

        _report_progress(
            progress_callback,
            phase="writing",
            message="Writing blob_features.parquet and QC artifacts",
            completed_units=eligible.frame_count,
            total_units=eligible.frame_count,
            is_indeterminate=False,
        )
        write_blob_features_parquet(paths.blob_features, table)
        blob_features_written = True
        _write_json(paths.qc, qc_summary.to_metadata())

        if qc_summary.has_warnings:
            status = STATUS_COMPLETED_WITH_WARNINGS
            warnings = qc_summary.warnings
        else:
            status = STATUS_COMPLETED
            warnings = ()
        success = True
        _report_progress(
            progress_callback,
            phase="completed",
            message=f"Extraction finished ({status})",
            completed_units=eligible.frame_count,
            total_units=eligible.frame_count,
            is_indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        if paths.blob_features.exists():
            with suppress(OSError):
                paths.blob_features.unlink()
        if paths.qc.exists():
            with suppress(OSError):
                paths.qc.unlink()
        blob_features_written = False

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        eligible=eligible,
        config=config,
        provenance=provenance,
        timing_map=timing_map,
        qc_summary=qc_summary,
        frame_errors=frame_errors,
        error_message=error_message,
        blob_features_written=blob_features_written,
    )

    if not success:
        raise BlobFeaturesError(error_message or "Blob feature extraction failed.")

    return BlobFeaturesResult(
        success=True,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        qc_path=paths.qc,
        blob_features_path=paths.blob_features if blob_features_written else None,
        session_root=eligible.session_root,
        warnings=warnings,
        error_message=None,
        qc=qc_summary.to_metadata() if qc_summary is not None else None,
    )


def _resolve_input(request: BlobFeaturesRequest) -> EligibleS1Input:
    if request.preprocess_dir is not None:
        return validate_s1_for_blob_features(request.preprocess_dir)
    if request.session_root is None:
        raise BlobFeaturesError(
            "Either session_root or preprocess_dir is required for blob feature extraction."
        )
    return validate_s1_for_blob_features(request.session_root)


def _resolve_config(request: BlobFeaturesRequest) -> BlobFeaturesConfig:
    if request.config is not None:
        return request.config.validated()
    if request.config_path is not None:
        return load_blob_features_config(request.config_path)
    return load_blob_features_config(None)


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else session_root / BLOB_FEATURES_DIRNAME
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        blob_features=run_dir / BLOB_FEATURES_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
        qc=run_dir / QC_FILENAME,
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
            raise BlobFeaturesError(
                "S5 run directory must not overlap S1–S4 artifacts: "
                f"{resolved_run}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _load_background(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise BlobFeaturesError(f"cropped_background.png is unreadable: {path}")
    return image


def _extract_source_features(
    *,
    video_path: Path,
    background: np.ndarray,
    frame_count: int,
    config: BlobFeaturesConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode prepared video sequentially and compute per-frame features.

    Processing is intentionally sequential for deterministic ordering. Execution
    knobs (``n_workers`` / ``max_in_flight``) are retained in settings for
    provenance and future optional parallelism.
    """

    _ = config.execution  # recorded in settings; sequential path for Sprint 2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise BlobFeaturesError(f"Cannot open prepared video: {video_path}")

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    # Report every frame for short videos; throttle for long ones.
    report_every = 1 if frame_count <= 120 else max(1, frame_count // 100)
    try:
        for frame_idx in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise BlobFeaturesError(
                    f"Failed to decode prepared-video frame {frame_idx} "
                    f"(expected {frame_count} frames)."
                )
            try:
                features = compute_frame_features(
                    frame, background, config.segmentation
                )
                record = features.as_record(frame_idx=frame_idx)
                if features.occlusion_static_error:
                    errors.append(
                        {
                            "frame_idx": frame_idx,
                            "error": "occlusion_static_feature_error",
                        }
                    )
                records.append(record)
            except Exception as exc:  # noqa: BLE001 — per-frame isolation
                errors.append({"frame_idx": frame_idx, "error": str(exc)})
                nan_record: dict[str, Any] = {
                    "frame_idx": frame_idx,
                    "cc_count": float("nan"),
                    "cc_count_raw": -1,
                    "min_inter_contour_dist": float("nan"),
                }
                for name in (
                    "ellipse_eccentricity",
                    "circularity",
                    "solidity",
                    "num_dt_peaks",
                    "furthest_endpoint_distance_px",
                ):
                    nan_record[name] = float("nan")
                records.append(nan_record)
            completed = frame_idx + 1
            if (
                completed == 1
                or completed == frame_count
                or completed % report_every == 0
            ):
                _report_progress(
                    progress_callback,
                    phase="extracting_frames",
                    message=(
                        f"Processed prepared-video frame {completed}/{frame_count}"
                    ),
                    completed_units=completed,
                    total_units=frame_count,
                    is_indeterminate=False,
                )
    finally:
        cap.release()

    if len(records) != frame_count:
        raise BlobFeaturesError(
            f"Processed {len(records)} frames but expected {frame_count}."
        )
    return records, {"count": len(errors), "items": errors[:50]}


def is_s5_run_eligible_for_downstream(s5_run_dir: Path | str) -> bool:
    """Return True when a completed S5 run still matches its recorded S1 source.

    A previously completed S5 run becomes ineligible when its referenced S1
    artifacts are missing, re-prepared, or digest-mismatched. Historical
    ``status=completed`` is left unchanged.
    """

    try:
        validate_s5_run_for_downstream(s5_run_dir)
    except BlobFeaturesError:
        return False
    return True


def validate_s5_run_for_downstream(s5_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed S5 run for downstream use (staleness-aware)."""

    run_dir = Path(s5_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise BlobFeaturesError(f"S5 run directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise BlobFeaturesError(f"S5 run_meta.json is missing: {meta_path}")
    meta = _load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise BlobFeaturesError(
            f"S5 run is not completed (status={status or 'missing'!r})."
        )

    parquet = run_dir / BLOB_FEATURES_FILENAME
    if not parquet.is_file():
        raise BlobFeaturesError(
            f"Completed S5 run is missing {BLOB_FEATURES_FILENAME}: {parquet}"
        )

    source = meta.get("source_s1")
    if not isinstance(source, Mapping):
        raise BlobFeaturesError("S5 run_meta.json is missing source_s1 provenance.")

    session_root_text = _optional_text(source.get("session_root"))
    if session_root_text is None:
        raise BlobFeaturesError("S5 run_meta.json source_s1.session_root is missing.")

    eligible = validate_s1_for_blob_features(session_root_text)
    recorded_sync = _optional_text(source.get("prepared_sync"))
    if (
        recorded_sync is not None
        and eligible.prepared_sync.resolve() != Path(recorded_sync).resolve()
    ):
        raise BlobFeaturesError(
            "S5 source S1 prepared_sync path no longer matches the recorded path."
        )
    recorded_sync_sha = _optional_text(source.get("prepared_sync_sha256"))
    if recorded_sync_sha is not None:
        current_sha = _sha256(eligible.prepared_sync)
        if current_sha.lower() != recorded_sync_sha.lower():
            raise BlobFeaturesError(
                "S5 source S1 prepared_sync SHA256 no longer matches the recorded digest."
            )
    recorded_video_sha = _optional_text(source.get("prepared_video_sha256"))
    if recorded_video_sha is not None:
        current_video_sha = _sha256(eligible.prepared_video)
        if current_video_sha.lower() != recorded_video_sha.lower():
            raise BlobFeaturesError(
                "S5 source S1 prepared_video SHA256 no longer matches the recorded digest."
            )
    recorded_bg_sha = _optional_text(source.get("cropped_background_sha256"))
    if recorded_bg_sha is not None:
        current_bg_sha = _sha256(eligible.cropped_background)
        if current_bg_sha.lower() != recorded_bg_sha.lower():
            raise BlobFeaturesError(
                "S5 source S1 cropped_background SHA256 no longer matches the recorded digest."
            )
    return {
        "s5_run_dir": run_dir,
        "status": status,
        "session_root": eligible.session_root,
        "prepared_sync_sha256": _sha256(eligible.prepared_sync),
        "prepared_video_sha256": _sha256(eligible.prepared_video),
        "cropped_background_sha256": _sha256(eligible.cropped_background),
    }


def _report_progress(
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed_units: int | None,
    total_units: int | None,
    is_indeterminate: bool,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        OperationProgress(
            phase=phase,
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            total_is_estimate=False,
            is_indeterminate=is_indeterminate,
            processed_video_time_sec=None,
        )
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise BlobFeaturesError(f"Could not load JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BlobFeaturesError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _s1_provenance(eligible: EligibleS1Input) -> dict[str, Any]:
    return {
        "session_root": _path_text(eligible.session_root),
        "preprocess_dir": _path_text(eligible.preprocess_dir),
        "prepared_video": _path_text(eligible.prepared_video),
        "cropped_background": _path_text(eligible.cropped_background),
        "prepared_sync": _path_text(eligible.prepared_sync),
        "prepare_meta": _path_text(eligible.prepare_meta),
        "prepared_video_sha256": _sha256(eligible.prepared_video),
        "cropped_background_sha256": _sha256(eligible.cropped_background),
        "prepared_sync_sha256": _sha256(eligible.prepared_sync),
        "prepare_meta_sha256": _sha256(eligible.prepare_meta),
        "frame_count": int(eligible.frame_count),
        "fps_header": float(eligible.fps_header),
        "frame_width": int(eligible.frame_width),
        "frame_height": int(eligible.frame_height),
        "target_fps": float(TARGET_FPS_DEFAULT),
        "blob_features_schema_version": BLOB_FEATURES_SCHEMA_VERSION,
    }


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    eligible: EligibleS1Input,
    config: BlobFeaturesConfig,
    provenance: Mapping[str, Any],
    timing_map: NearestNeighborTimingMap | None,
    qc_summary: QcSummary | None,
    frame_errors: Mapping[str, Any] | None,
    error_message: str | None,
    blob_features_written: bool,
) -> None:
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            eligible=eligible,
            paths=paths,
            config=config,
            provenance=provenance,
            timing_map=timing_map,
            qc_summary=qc_summary,
            frame_errors=frame_errors,
            error_message=error_message,
            blob_features_written=blob_features_written,
        ),
    )
    _write_yaml(paths.settings_used, _settings_payload(config=config, status=status))
    if qc_summary is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        _write_json(paths.qc, qc_summary.to_metadata())
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        eligible=eligible,
        paths=paths,
        config=config,
        timing_map=timing_map,
        qc_summary=qc_summary,
        frame_errors=frame_errors,
        error_message=error_message,
        blob_features_written=blob_features_written,
    )


def _run_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    eligible: EligibleS1Input,
    paths: _RunPaths,
    config: BlobFeaturesConfig,
    provenance: Mapping[str, Any],
    timing_map: NearestNeighborTimingMap | None,
    qc_summary: QcSummary | None,
    frame_errors: Mapping[str, Any] | None,
    error_message: str | None,
    blob_features_written: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "source_s1": dict(provenance),
        "settings": config.to_settings_dict(),
        "outputs": {
            "blob_features_parquet": (
                _path_text(paths.blob_features) if blob_features_written else None
            ),
            "settings_used": _path_text(paths.settings_used),
            "processing_log": _path_text(paths.processing_log),
            "qc": _path_text(paths.qc),
        },
        "qc": qc_summary.to_metadata() if qc_summary is not None else None,
        "frame_processing_errors": dict(frame_errors) if frame_errors else None,
        "session_root": _path_text(eligible.session_root),
    }
    if timing_map is not None:
        payload["timing_summary"] = {
            "target_fps": float(timing_map.target_fps),
            "target_sample_count": int(timing_map.target_sample_count),
            "source_frame_count": int(timing_map.source_frame_count),
            "source_fps_header": float(timing_map.source_fps_header),
        }
    if error_message is not None:
        payload["error"] = error_message
    return payload


def _settings_payload(*, config: BlobFeaturesConfig, status: str) -> dict[str, Any]:
    payload = config.to_settings_dict()
    payload["status"] = status
    return payload


def _write_processing_log(
    *,
    path: Path,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    eligible: EligibleS1Input,
    paths: _RunPaths,
    config: BlobFeaturesConfig,
    timing_map: NearestNeighborTimingMap | None,
    qc_summary: QcSummary | None,
    frame_errors: Mapping[str, Any] | None,
    error_message: str | None,
    blob_features_written: bool,
) -> None:
    lines = [
        "Subsystem 05 blob feature extraction",
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"created_at: {created_at}",
        f"completed_at: {completed_at}",
        f"status: {status}",
        f"session_root: {_path_text(eligible.session_root)}",
        f"prepared_video: {_path_text(eligible.prepared_video)}",
        f"prepared_sync: {_path_text(eligible.prepared_sync)}",
        f"source_frame_count: {eligible.frame_count}",
        f"fps_header: {eligible.fps_header}",
        f"target_fps: {config.target_fps}",
        f"cc_count_gap_fill_max_gap: {config.temporal.cc_count_gap_fill_max_gap}",
        f"blob_threshold: {config.segmentation.blob_threshold}",
        (
            "execution.n_workers: "
            f"{config.execution.n_workers} (sequential decode in Sprint 2)"
        ),
        f"execution.max_in_flight: {config.execution.max_in_flight}",
    ]
    if timing_map is not None:
        lines.extend(
            [
                f"target_sample_count: {timing_map.target_sample_count}",
                f"source_fps_header: {timing_map.source_fps_header}",
            ]
        )
    if frame_errors is not None:
        lines.append(f"frame_processing_error_count: {frame_errors.get('count', 0)}")
    if qc_summary is not None:
        lines.append(f"qc.structural_ok: {qc_summary.structural_ok}")
        lines.append(f"qc.warning_count: {len(qc_summary.warnings)}")
        for warning in qc_summary.warnings:
            lines.append(f"qc.warning: {warning}")
        for failure in qc_summary.failures:
            lines.append(f"qc.failure: {failure}")
    lines.append(
        "blob_features_parquet: "
        + (_path_text(paths.blob_features) if blob_features_written else "not_written")
    )
    if error_message:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value.strip())
    if not cleaned:
        raise BlobFeaturesError("timestamp is empty after sanitization.")
    return cleaned


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _sha256(path: Path) -> str:
    """Content digest with process-local (path, size, mtime) caching."""

    return sha256_file(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
