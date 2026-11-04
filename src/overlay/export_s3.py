"""C3-2 S3 accepted tracked-pose QC export (reuses C2-A/C2-B streaming ffmpeg pipe).

Does not modify S3 working-pose visualization or the acceptance workflow.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from overlay import export as _shared
from overlay.adapters.base import OverlayStageAdapter, SourceIdentity
from overlay.adapters.pose import make_s3_tracked_pose_adapter
from overlay.compositor import OverlayCompositor
from overlay.contracts import (
    OVERLAY_QC_EXPORT_DIRNAME,
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_QC_PROCESSING_LOG_FILENAME,
    OVERLAY_QC_RUN_META_FILENAME,
    OVERLAY_QC_S3_POSE_JOB_PREFIX,
    OVERLAY_QC_S3_POSE_VIDEO_FILENAME,
    OVERLAY_QC_SETTINGS_FILENAME,
    OVERLAY_QC_STAGING_DIRNAME,
    OVERLAY_S3_POSE_RENDERER_VERSION,
    FrameDomain,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    PlateauPolicy,
)
from overlay.evidence import collect_evidence
from overlay.plan import S3PoseQcExportPlan
from overlay.provenance import (
    OverlayArtifactManifest,
    PreparedVideoIdentity,
    build_overlay_artifact_manifest,
)
from overlay.source_s3 import OverlayS3QcSource, verify_overlay_s3_qc_source_integrity
from overlay.style import DrawConfig
from overlay.timing import resolve_prepared_frame_timing
from project.file_digest import sha256_file
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
)
from results_packaging.media_csv_export.video import probe_encoded_clip


@dataclass(frozen=True, slots=True)
class OverlayS3PoseQcExportResult:
    """Published S3 pose QC artifact references."""

    job_dir: Path
    video_path: Path
    manifest_path: Path
    manifest: OverlayArtifactManifest
    frames_written: int
    output_fps: float
    unavailable_layers: tuple[str, ...]
    qc_status: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "video_path": str(self.video_path),
            "manifest_path": str(self.manifest_path),
            "frames_written": self.frames_written,
            "output_fps": self.output_fps,
            "unavailable_layers": list(self.unavailable_layers),
            "qc_status": self.qc_status,
            "notes": list(self.notes),
            "manifest": self.manifest.to_dict(),
        }


def build_s3_pose_adapters(
    source: OverlayS3QcSource,
    *,
    prepared_frame_range: tuple[int, int] | None,
    selected_layers: tuple[LayerId, ...],
) -> dict[LayerId, OverlayStageAdapter]:
    """Build S3 pose (+ optional debug) adapters for the selected layers."""

    adapters: dict[LayerId, OverlayStageAdapter] = {}
    if LayerId.S3_POSE_TRACKED in selected_layers:
        adapters[LayerId.S3_POSE_TRACKED] = make_s3_tracked_pose_adapter(
            source.tracked_pose_path,
            run_id=source.s3_run_id,
            run_dir=source.s3_run_dir,
            prepared_frame_range=prepared_frame_range,
        )
    if LayerId.DEBUG_FRAME_COUNTER in selected_layers:
        adapters[LayerId.DEBUG_FRAME_COUNTER] = _shared._DebugAdapter()
    return adapters


def export_s3_pose_qc_clip(
    *,
    source: OverlayS3QcSource,
    plan: S3PoseQcExportPlan,
    output_root: Path | str | None = None,
    job_dir: Path | str | None = None,
    ffmpeg_executable: Path | str | None = None,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    progress_callback: _shared.ProgressCallback | None = None,
    cancellation_requested: _shared.CancellationCallback | None = None,
    verify_encoded_boundaries: bool = True,
) -> OverlayS3PoseQcExportResult:
    """Export one short prepared-rate S3 accepted tracked-pose QC MP4.

    Opt-in only. Pins S3 / S1 digests and revalidates before publication. Does
    not write scientific artifacts and does not change S3 acceptance workflow.
    """

    if plan.frame_domain is not FrameDomain.PREPARED:
        raise OverlayExportError(
            "C3-2 S3 pose QC export requires prepared frame domain "
            f"(got {plan.frame_domain.value})."
        )
    if plan.plateau_policy is not PlateauPolicy.LAST_IN_PLATEAU:
        raise OverlayExportError(
            "C3-2 S3 pose QC export requires LAST_IN_PLATEAU policy "
            f"(got {plan.plateau_policy.value})."
        )
    if LayerId.S3_POSE_TRACKED not in plan.selected_layers:
        raise OverlayExportError("C3-2 S3 pose QC requires s3.pose_tracked layer.")

    performance: dict[str, float] = {}
    t_phase = time.perf_counter()
    verify_overlay_s3_qc_source_integrity(source)
    performance["source_validation_and_hashing_sec"] = time.perf_counter() - t_phase

    out_job = _resolve_s3_pose_job_dir(
        source, output_root=output_root, job_dir=job_dir
    )
    staging = out_job / OVERLAY_QC_STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    video_final = out_job / OVERLAY_QC_S3_POSE_VIDEO_FILENAME
    manifest_final = out_job / OVERLAY_QC_MANIFEST_FILENAME
    if video_final.exists() or manifest_final.exists():
        raise OverlayExportError(
            f"Refusing to overwrite existing S3 pose QC artifacts in {out_job}."
        )

    video_tmp = staging / OVERLAY_QC_S3_POSE_VIDEO_FILENAME
    prepared_range = (
        int(plan.prepared_interval.start),
        int(plan.prepared_interval.end),
    )
    t_phase = time.perf_counter()
    adapters = build_s3_pose_adapters(
        source,
        prepared_frame_range=prepared_range,
        selected_layers=plan.selected_layers,
    )
    performance["s3_pose_loading_sec"] = time.perf_counter() - t_phase

    machine_by_frame = (
        source.machine_events.by_frame_dict(prepared_range)
        if plan.show_machine_corrections
        else {}
    )
    manual_by_frame = (
        source.manual_events.by_frame_dict(prepared_range)
        if plan.show_manual_corrections
        else {}
    )
    style = DrawConfig(
        **{
            **plan.style.to_dict(),
            "draw_skeleton": bool(plan.draw_skeleton),
            "prepared_video_fps": float(source.video_probe.fps),
            "extra": {
                **dict(plan.style.extra),
                "skeleton_edges": [list(e) for e in source.skeleton_edges],
                "show_machine_corrections": bool(plan.show_machine_corrections),
                "show_manual_corrections": bool(plan.show_manual_corrections),
                "s3_machine_events_by_frame": machine_by_frame,
                "s3_manual_events_by_frame": manual_by_frame,
                "s3_machine_events_status": source.machine_events.status,
                "s3_manual_events_status": source.manual_events.status,
            },
        }
    )
    compositor = OverlayCompositor(
        frame_domain=FrameDomain.PREPARED,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
    )

    fps = float(source.video_probe.fps)
    width = int(source.video_probe.width)
    height = int(source.video_probe.height)
    expected = int(plan.prepared_interval.frame_count)
    unavailable_union: set[str] = set()
    notes: list[str] = []
    frames_written = 0
    first_timing = None
    last_timing = None
    seen_tracks: set[int] = set()
    machine_events_drawn = 0
    manual_events_drawn = 0
    timing_resolution_sec = 0.0
    frame_composition_sec = 0.0
    ffmpeg_write_sec = 0.0
    decode_sec = 0.0

    _shared._emit(
        progress_callback,
        {
            "stage": "encode_start",
            "export_kind": "s3_pose_qc_clip",
            "job_dir": str(out_job),
            "prepared_frame_idx_start": plan.prepared_interval.start,
            "prepared_frame_idx_end": plan.prepared_interval.end,
            "expected_frames": expected,
            "fps": fps,
        },
    )

    pipe: _shared._FfmpegPipeSession | None = None
    capture: cv2.VideoCapture | None = None
    try:
        _shared._check_cancel(cancellation_requested)
        capture = _shared._open_capture_at(
            source.prepared_video_path,
            prepared_frame_idx=plan.prepared_interval.start,
        )
        pipe = _shared._start_ffmpeg_pipe(
            output_path=video_tmp,
            width=width,
            height=height,
            fps=fps,
            ffmpeg_executable=ffmpeg_executable,
            crf=crf,
            codec=codec,
            pixel_format=pixel_format,
        )
        assert pipe.proc.stdin is not None

        for prep in range(plan.prepared_interval.start, plan.prepared_interval.end + 1):
            _shared._check_cancel(cancellation_requested)
            t_decode = time.perf_counter()
            ok, frame = capture.read()
            decode_sec += time.perf_counter() - t_decode
            if not ok or frame is None:
                raise OverlayExportError(
                    f"Failed to decode prepared_frame_idx={prep} from "
                    f"{source.prepared_video_path}."
                )
            if frame.shape[1] != width or frame.shape[0] != height:
                raise OverlayExportError(
                    f"Decoded frame size mismatch at prepared_frame_idx={prep}: "
                    f"got {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}."
                )

            t_timing = time.perf_counter()
            evidence = collect_evidence(
                source.sync,
                prep,
                adapters,
                selected_layers=plan.selected_layers,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            )
            timing_resolution_sec += time.perf_counter() - t_timing
            obs = evidence.get(LayerId.S3_POSE_TRACKED)
            if obs is not None and obs.available:
                for pt in obs.payload.get("points") or []:
                    seen_tracks.add(int(pt["track"]))
            machine_events_drawn += len(machine_by_frame.get(prep, ()))
            manual_events_drawn += len(manual_by_frame.get(prep, ()))

            t_compose = time.perf_counter()
            composed = compositor.render_frame(
                frame,
                prepared_frame_idx=prep,
                evidence=evidence,
                selected_layers=plan.selected_layers,
                style=style,
            )
            frame_composition_sec += time.perf_counter() - t_compose
            unavailable_union.update(x.value for x in composed.unavailable_layers)
            if first_timing is None:
                first_timing = evidence.timing
            last_timing = evidence.timing

            if prep in {
                plan.prepared_interval.start,
                plan.prepared_interval.end,
            }:
                notes.append(
                    "prepared_frame_idx="
                    f"{prep}->sample_idx={evidence.timing.dense_sample_idx}"
                    f" policy={evidence.timing.plateau_policy.value}"
                    f" available={evidence.timing.available}"
                    " (prepared-identity map; not dense 120 Hz)"
                )

            raw = np.ascontiguousarray(composed.frame_bgr)
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8, copy=False)
            try:
                t_ff = time.perf_counter()
                pipe.proc.stdin.write(raw.tobytes())
                ffmpeg_write_sec += time.perf_counter() - t_ff
            except BrokenPipeError as exc:
                detail = _shared._ffmpeg_stderr(pipe)
                raise OverlayExportError(
                    f"ffmpeg pipe broke while writing prepared_frame_idx={prep}: "
                    f"{detail}"
                ) from exc
            frames_written += 1
            if frames_written == 1 or frames_written == expected or frames_written % 30 == 0:
                _shared._emit(
                    progress_callback,
                    {
                        "stage": "encoding",
                        "frames_written": frames_written,
                        "expected_frames": expected,
                        "prepared_frame_idx": prep,
                        "dense_sample_idx": evidence.timing.dense_sample_idx,
                    },
                )

        t_ff_close = time.perf_counter()
        pipe.proc.stdin.close()
        rc = pipe.proc.wait(timeout=120)
        ffmpeg_close_sec = time.perf_counter() - t_ff_close
        if rc != 0 or not video_tmp.is_file() or video_tmp.stat().st_size <= 0:
            detail = _shared._ffmpeg_stderr(pipe)
            raise OverlayExportError(
                f"ffmpeg encode failed (code={rc}): {detail[-1200:]}"
            )
        if frames_written != expected:
            raise OverlayExportError(
                f"Wrote {frames_written} frames but planned inclusive count is {expected}."
            )
        if verify_encoded_boundaries:
            probed = probe_encoded_clip(video_tmp)
            if int(probed.n_frames) != expected:
                raise OverlayExportError(
                    f"Encoded QC clip frame count {probed.n_frames} does not match "
                    f"planned inclusive count {expected}."
                )
            if probed.fps <= 0 or abs(probed.fps - fps) / fps > 0.02:
                raise OverlayExportError(
                    f"Encoded QC clip fps {probed.fps} does not match source fps {fps}."
                )

        verify_overlay_s3_qc_source_integrity(source)

        source_identities = (
            SourceIdentity(
                subsystem="tracking_correction",
                run_id=source.s3_run_id,
                run_dir=source.s3_run_dir,
                artifact_path=source.tracked_pose_path,
                digest_sha256=source.tracked_pose_sha256,
                extra={
                    "legend": "S3 accepted tracked pose",
                    "domain": "prepared",
                    "identity_kind": "accepted_or_tracked",
                    "acceptance_state": source.acceptance_state,
                },
            ),
        )
        prepared_video = PreparedVideoIdentity(
            path=source.prepared_video_path,
            digest_sha256=source.prepared_video_sha256,
            fps=fps,
            frame_count=int(source.video_probe.n_frames),
            extra={
                "width": width,
                "height": height,
                "prepared_sync_path": str(source.prepared_sync_path),
                "prepared_sync_sha256": source.prepared_sync_sha256,
            },
        )
        example_timing = last_timing or first_timing
        if example_timing is None:
            example_timing = resolve_prepared_frame_timing(
                source.sync,
                plan.prepared_interval.start,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            )

        skeleton_disclosure = source.skeleton_disclosure()
        correction_disclosure = source.correction_disclosure()
        correction_disclosure = {
            **correction_disclosure,
            "show_machine_corrections": bool(plan.show_machine_corrections),
            "show_manual_corrections": bool(plan.show_manual_corrections),
            "machine_event_draws_in_clip": int(machine_events_drawn),
            "manual_event_draws_in_clip": int(manual_events_drawn),
            "no_suspicious_frame_fabrication": True,
            "no_was_track_swapped_per_node_events": True,
        }
        for note in (
            source.machine_events.disclosure,
            source.manual_events.disclosure,
            skeleton_disclosure["skeleton_disclosure"],
        ):
            if note and note not in notes:
                notes.append(str(note))

        performance["prepared_video_decode_sec"] = decode_sec
        performance["timing_resolution_sec"] = timing_resolution_sec
        performance["frame_composition_sec"] = frame_composition_sec
        performance["ffmpeg_encoding_sec"] = ffmpeg_write_sec + ffmpeg_close_sec
        performance["ffmpeg_stdin_write_sec"] = ffmpeg_write_sec
        performance["ffmpeg_close_sec"] = ffmpeg_close_sec
        qc_status = "completed"
        t_publish = time.perf_counter()
        video_sha = sha256_file(video_tmp)
        _write_s3_pose_run_meta(
            out_job,
            source=source,
            plan=plan,
            frames_written=frames_written,
            qc_status=qc_status,
        )
        _write_s3_pose_settings(out_job, source=source, plan=plan, style=style)
        performance["sidecar_publication_sec"] = time.perf_counter() - t_publish
        perf_rounded = {k: round(float(v), 6) for k, v in performance.items()}
        manifest = build_overlay_artifact_manifest(
            sync=source.sync,
            prepared_video=prepared_video,
            source_identities=source_identities,
            selected_layers=plan.selected_layers,
            plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            style=style,
            timing=example_timing,
            output_frame_domain=FrameDomain.PREPARED,
            output_fps=fps,
            notes=tuple(notes),
            renderer_version=OVERLAY_S3_POSE_RENDERER_VERSION,
            extra={
                "export_kind": "s3_pose_qc_clip",
                "qc_status": qc_status,
                "renderer_version": OVERLAY_S3_POSE_RENDERER_VERSION,
                "identity_semantics": {
                    "kind": "s3_accepted_tracked_identities",
                    "label_prefix": "M",
                    "note": (
                        "M{id} colors/labels are S3-accepted tracked identities "
                        "from tracked_pose.parquet."
                    ),
                    "observed_accepted_tracks_in_clip": sorted(seen_tracks),
                },
                "corrections": correction_disclosure,
                "skeleton": skeleton_disclosure,
                "performance_breakdown_sec": perf_rounded,
                "source": {
                    "source_kind": "s3_accepted_tracked_pose",
                    "s3_run_id": source.s3_run_id,
                    "s3_run_dir": str(source.s3_run_dir),
                    "acceptance_state": source.acceptance_state,
                    "tracked_pose_path": str(source.tracked_pose_path),
                    "tracked_pose_sha256": source.tracked_pose_sha256,
                    "source_s2_run_id": source.source_s2_run_id,
                    "source_s2_run_dir": (
                        None
                        if source.source_s2_run_dir is None
                        else str(source.source_s2_run_dir)
                    ),
                    "prepared_sync_path": str(source.prepared_sync_path),
                    "prepared_sync_sha256": source.prepared_sync_sha256,
                    "prepared_video_path": str(source.prepared_video_path),
                    "prepared_video_sha256": source.prepared_video_sha256,
                    "source_digests": dict(source.source_digests),
                    "eligibility": (
                        "validate_accepted_s3_run_at_open_and_before_publish"
                    ),
                    "timing_authority": source.provenance.get("timing_authority"),
                    "sync_build_audit": dict(source.sync_build_audit),
                    "skeleton_source": source.skeleton_source,
                    "skeleton_edge_count": len(source.skeleton_edges),
                    "skeleton_mode": skeleton_disclosure["skeleton_mode"],
                    "skeleton_nodes_only": skeleton_disclosure["skeleton_nodes_only"],
                    "skeleton_disclosure": skeleton_disclosure["skeleton_disclosure"],
                    "provenance": dict(source.provenance),
                },
                "plan": plan.to_dict(),
                "prepared_frame_range": plan.prepared_interval.to_dict(),
                "output": {
                    "video_filename": OVERLAY_QC_S3_POSE_VIDEO_FILENAME,
                    "codec": codec,
                    "pixel_format": pixel_format,
                    "crf": int(crf),
                    "output_fps": fps,
                    "frame_count": frames_written,
                    "video_sha256": video_sha,
                },
                "unavailable_layers": sorted(unavailable_union),
                "first_frame_timing": (
                    None if first_timing is None else first_timing.to_dict()
                ),
                "last_frame_timing": (
                    None if last_timing is None else last_timing.to_dict()
                ),
            },
        )
        manifest = OverlayArtifactManifest(
            schema_version=manifest.schema_version,
            renderer_version=manifest.renderer_version,
            source_runs=manifest.source_runs,
            source_digests=manifest.source_digests,
            prepared_video=manifest.prepared_video,
            timing_mapping=manifest.timing_mapping,
            plateau_policy=manifest.plateau_policy,
            layer_selections=manifest.layer_selections,
            unavailable_layers=tuple(sorted(unavailable_union)),
            output_frame_domain=manifest.output_frame_domain,
            output_fps=manifest.output_fps,
            render_settings=manifest.render_settings,
            pixel_format=manifest.pixel_format,
            notes=manifest.notes,
            extra=manifest.extra,
        )

        manifest_tmp = staging / OVERLAY_QC_MANIFEST_FILENAME
        manifest.write_json(manifest_tmp)
        video_tmp.replace(video_final)
        manifest_tmp.replace(manifest_final)
        _write_s3_pose_processing_log(
            out_job,
            source=source,
            plan=plan,
            frames_written=frames_written,
            qc_status=qc_status,
            corrections=correction_disclosure,
            performance=perf_rounded,
        )
        shutil.rmtree(staging, ignore_errors=True)

        _shared._emit(
            progress_callback,
            {
                "stage": "published",
                "export_kind": "s3_pose_qc_clip",
                "job_dir": str(out_job),
                "video_path": str(video_final),
                "manifest_path": str(manifest_final),
                "frames_written": frames_written,
                "qc_status": qc_status,
                "performance_breakdown_sec": perf_rounded,
                "skeleton": skeleton_disclosure,
                "corrections": correction_disclosure,
            },
        )
        return OverlayS3PoseQcExportResult(
            job_dir=out_job,
            video_path=video_final,
            manifest_path=manifest_final,
            manifest=manifest,
            frames_written=frames_written,
            output_fps=fps,
            unavailable_layers=tuple(sorted(unavailable_union)),
            qc_status=qc_status,
            notes=tuple(notes),
        )
    except OverlayExportCancelled:
        _cleanup_failed_s3_pose_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    except Exception:
        _cleanup_failed_s3_pose_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    finally:
        if capture is not None:
            capture.release()


def _resolve_s3_pose_job_dir(
    source: OverlayS3QcSource,
    *,
    output_root: Path | str | None,
    job_dir: Path | str | None,
) -> Path:
    if job_dir is not None:
        out = Path(job_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    if output_root is not None:
        root = Path(output_root).expanduser().resolve()
    else:
        root = Path(source.session_root) / OVERLAY_QC_EXPORT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = root / f"{OVERLAY_QC_S3_POSE_JOB_PREFIX}{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _cleanup_failed_s3_pose_export(
    *,
    pipe: _shared._FfmpegPipeSession | None,
    staging: Path,
    job_dir: Path,
) -> None:
    if pipe is not None:
        try:
            if pipe.proc.stdin is not None and not pipe.proc.stdin.closed:
                pipe.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if pipe.proc.poll() is None:
                pipe.proc.kill()
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(staging, ignore_errors=True)
    if job_dir.is_dir():
        remaining = [p for p in job_dir.iterdir() if p.name != OVERLAY_QC_STAGING_DIRNAME]
        if not remaining:
            shutil.rmtree(job_dir, ignore_errors=True)


def _write_s3_pose_run_meta(
    job_dir: Path,
    *,
    source: OverlayS3QcSource,
    plan: S3PoseQcExportPlan,
    frames_written: int,
    qc_status: str,
) -> None:
    payload = {
        "schema_version": "overlay_s3_pose_qc_run_meta_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "export_kind": "s3_pose_qc_clip",
        "qc_status": qc_status,
        "s3_run_id": source.s3_run_id,
        "s3_run_dir": str(source.s3_run_dir),
        "acceptance_state": source.acceptance_state,
        "source_s2_run_id": source.source_s2_run_id,
        "prepared_interval": plan.prepared_interval.to_dict(),
        "frames_written": int(frames_written),
        "renderer_version": OVERLAY_S3_POSE_RENDERER_VERSION,
    }
    (job_dir / OVERLAY_QC_RUN_META_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_s3_pose_settings(
    job_dir: Path,
    *,
    source: OverlayS3QcSource,
    plan: S3PoseQcExportPlan,
    style: DrawConfig,
) -> None:
    payload = {
        "schema_version": "overlay_s3_pose_qc_settings_v1",
        "export_kind": "s3_pose_qc_clip",
        "s3_run_id": source.s3_run_id,
        "plan": plan.to_dict(),
        "style": style.to_dict(),
        "source_digests": dict(source.source_digests),
        "corrections": source.correction_disclosure(),
        "skeleton": source.skeleton_disclosure(),
    }
    (job_dir / OVERLAY_QC_SETTINGS_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_s3_pose_processing_log(
    job_dir: Path,
    *,
    source: OverlayS3QcSource,
    plan: S3PoseQcExportPlan,
    frames_written: int,
    qc_status: str,
    corrections: dict[str, Any],
    performance: dict[str, float],
) -> None:
    lines = [
        "overlay_s3_pose_qc processing log",
        f"qc_status: {qc_status}",
        f"s3_run_id: {source.s3_run_id}",
        f"acceptance_state: {source.acceptance_state}",
        f"tracked_pose_sha256: {source.tracked_pose_sha256}",
        f"prepared_interval: [{plan.prepared_interval.start}, {plan.prepared_interval.end}]",
        f"frames_written: {frames_written}",
        f"draw_skeleton: {plan.draw_skeleton}",
        f"show_machine_corrections: {plan.show_machine_corrections}",
        f"show_manual_corrections: {plan.show_manual_corrections}",
        f"skeleton_mode: {source.skeleton_mode}",
        f"machine_events_status: {source.machine_events.status}",
        f"manual_events_status: {source.manual_events.status}",
        f"machine_disclosure: {source.machine_events.disclosure}",
        f"manual_disclosure: {source.manual_events.disclosure}",
        f"corrections_json: {json.dumps(corrections, sort_keys=True)}",
        f"performance_sec: {json.dumps(performance, sort_keys=True)}",
        "note: was_track_swapped never invents per-node or suspicious-frame markers",
    ]
    (job_dir / OVERLAY_QC_PROCESSING_LOG_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


__all__ = [
    "OverlayS3PoseQcExportResult",
    "build_s3_pose_adapters",
    "export_s3_pose_qc_clip",
]
