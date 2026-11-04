"""C3-1 S2 inferred-pose QC export (reuses C2-A/C2-B streaming ffmpeg pipe).

Does not modify the existing S2 ``overlay.mp4`` generation path in
``pose_inference.overlay``.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from overlay import export as _shared
from overlay.adapters.base import OverlayStageAdapter, SourceIdentity
from overlay.adapters.pose import make_s2_inferred_pose_adapter
from overlay.compositor import OverlayCompositor
from overlay.contracts import (
    OVERLAY_QC_EXPORT_DIRNAME,
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_QC_PROCESSING_LOG_FILENAME,
    OVERLAY_QC_RUN_META_FILENAME,
    OVERLAY_QC_S2_POSE_JOB_PREFIX,
    OVERLAY_QC_S2_POSE_VIDEO_FILENAME,
    OVERLAY_QC_SETTINGS_FILENAME,
    OVERLAY_QC_STAGING_DIRNAME,
    OVERLAY_S2_POSE_RENDERER_VERSION,
    S2_SCORE_VISUAL_MAPPING_ID,
    S2_SCORE_VISUAL_MAPPING_SUMMARY,
    S2_SCORE_VISUAL_RANGE_MAX,
    S2_SCORE_VISUAL_RANGE_MIN,
    FrameDomain,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    PlateauPolicy,
)
from overlay.evidence import collect_evidence
from overlay.plan import S2PoseQcExportPlan
from overlay.provenance import (
    OverlayArtifactManifest,
    PreparedVideoIdentity,
    build_overlay_artifact_manifest,
)
from overlay.source_s2 import OverlayS2QcSource, verify_overlay_s2_qc_source_integrity
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
class OverlayS2PoseQcExportResult:
    """Published S2 pose QC artifact references."""

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


def build_s2_pose_adapters(
    source: OverlayS2QcSource,
    *,
    prepared_frame_range: tuple[int, int] | None,
    selected_layers: tuple[LayerId, ...],
) -> dict[LayerId, OverlayStageAdapter]:
    """Build S2 pose (+ optional debug) adapters for the selected layers."""

    adapters: dict[LayerId, OverlayStageAdapter] = {}
    if LayerId.S2_POSE_INFERRED in selected_layers:
        adapters[LayerId.S2_POSE_INFERRED] = make_s2_inferred_pose_adapter(
            source.pose_parquet_path,
            run_id=source.s2_run_id,
            run_dir=source.s2_run_dir,
            prepared_frame_range=prepared_frame_range,
        )
    if LayerId.DEBUG_FRAME_COUNTER in selected_layers:
        adapters[LayerId.DEBUG_FRAME_COUNTER] = _shared._DebugAdapter()
    return adapters


def export_s2_pose_qc_clip(
    *,
    source: OverlayS2QcSource,
    plan: S2PoseQcExportPlan,
    output_root: Path | str | None = None,
    job_dir: Path | str | None = None,
    ffmpeg_executable: Path | str | None = None,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    progress_callback: _shared.ProgressCallback | None = None,
    cancellation_requested: _shared.CancellationCallback | None = None,
    verify_encoded_boundaries: bool = True,
) -> OverlayS2PoseQcExportResult:
    """Export one short prepared-rate S2 inferred-pose QC MP4.

    Opt-in only. Reuses the shared streaming ffmpeg architecture. Pins S2 / S1
    digests and revalidates before publication. Does not write scientific
    artifacts and does not touch existing S2 ``overlay.mp4``.
    """

    if plan.frame_domain is not FrameDomain.PREPARED:
        raise OverlayExportError(
            "C3-1 S2 pose QC export requires prepared frame domain "
            f"(got {plan.frame_domain.value})."
        )
    if plan.plateau_policy is not PlateauPolicy.LAST_IN_PLATEAU:
        raise OverlayExportError(
            "C3-1 S2 pose QC export requires LAST_IN_PLATEAU policy "
            f"(got {plan.plateau_policy.value})."
        )
    if LayerId.S2_POSE_INFERRED not in plan.selected_layers:
        raise OverlayExportError("C3-1 S2 pose QC requires s2.pose_inferred layer.")

    performance: dict[str, float] = {}
    t_phase = time.perf_counter()
    verify_overlay_s2_qc_source_integrity(source)
    performance["source_validation_and_hashing_sec"] = time.perf_counter() - t_phase

    out_job = _resolve_s2_pose_job_dir(
        source, output_root=output_root, job_dir=job_dir
    )
    staging = out_job / OVERLAY_QC_STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    video_final = out_job / OVERLAY_QC_S2_POSE_VIDEO_FILENAME
    manifest_final = out_job / OVERLAY_QC_MANIFEST_FILENAME
    if video_final.exists() or manifest_final.exists():
        raise OverlayExportError(
            f"Refusing to overwrite existing S2 pose QC artifacts in {out_job}."
        )

    video_tmp = staging / OVERLAY_QC_S2_POSE_VIDEO_FILENAME
    prepared_range = (
        int(plan.prepared_interval.start),
        int(plan.prepared_interval.end),
    )
    t_phase = time.perf_counter()
    adapters = build_s2_pose_adapters(
        source,
        prepared_frame_range=prepared_range,
        selected_layers=plan.selected_layers,
    )
    performance["s2_pose_loading_sec"] = time.perf_counter() - t_phase
    style = DrawConfig(
        **{
            **plan.style.to_dict(),
            "draw_skeleton": bool(plan.draw_skeleton),
            "prepared_video_fps": float(source.video_probe.fps),
            "extra": {
                **dict(plan.style.extra),
                "skeleton_edges": [list(e) for e in source.skeleton_edges],
                "show_confidence": bool(plan.show_confidence),
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
    seen_scores: list[float] = []
    seen_missing_scores = 0
    seen_tracks: set[int] = set()
    timing_resolution_sec = 0.0
    frame_composition_sec = 0.0
    ffmpeg_write_sec = 0.0
    decode_sec = 0.0

    _shared._emit(
        progress_callback,
        {
            "stage": "encode_start",
            "export_kind": "s2_pose_qc_clip",
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
            obs = evidence.get(LayerId.S2_POSE_INFERRED)
            if obs is not None and obs.available:
                for pt in obs.payload.get("points") or []:
                    seen_tracks.add(int(pt["track"]))
                    score = pt.get("score")
                    if score is None or bool(pt.get("score_missing")):
                        seen_missing_scores += 1
                    else:
                        seen_scores.append(float(score))

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

        verify_overlay_s2_qc_source_integrity(source)

        source_identities = (
            SourceIdentity(
                subsystem="pose_inference",
                run_id=source.s2_run_id,
                run_dir=source.s2_run_dir,
                artifact_path=source.pose_parquet_path,
                digest_sha256=source.pose_parquet_sha256,
                extra={
                    "legend": "inferred pose (S2 predictions; provisional IDs)",
                    "domain": "prepared",
                    "identity_kind": "provisional",
                    "s2_status": source.s2_status,
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
        confidence_disclosure = {
            "enabled": bool(plan.show_confidence),
            "mapping_id": S2_SCORE_VISUAL_MAPPING_ID,
            "mapping_summary": S2_SCORE_VISUAL_MAPPING_SUMMARY,
            "assumed_score_range": [
                float(S2_SCORE_VISUAL_RANGE_MIN),
                float(S2_SCORE_VISUAL_RANGE_MAX),
            ],
            "clamp_policy": (
                "Values outside assumed range are clamped for drawing intensity "
                "only; original score values are preserved in observations."
            ),
            "missing_score_visual": "hollow_ring",
            "zero_score_visual": "filled_disc_minimum_intensity",
            "no_confidence_filtering": True,
            "observed_score_count_in_clip": len(seen_scores),
            "observed_missing_score_count_in_clip": int(seen_missing_scores),
            "observed_score_min_in_clip": (
                None if not seen_scores else float(min(seen_scores))
            ),
            "observed_score_max_in_clip": (
                None if not seen_scores else float(max(seen_scores))
            ),
            "scores_outside_assumed_range_in_clip": int(
                sum(
                    1
                    for s in seen_scores
                    if s < float(S2_SCORE_VISUAL_RANGE_MIN)
                    or s > float(S2_SCORE_VISUAL_RANGE_MAX)
                )
            ),
        }
        performance["prepared_video_decode_sec"] = decode_sec
        performance["timing_resolution_sec"] = timing_resolution_sec
        performance["frame_composition_sec"] = frame_composition_sec
        performance["ffmpeg_encoding_sec"] = ffmpeg_write_sec + ffmpeg_close_sec
        performance["ffmpeg_stdin_write_sec"] = ffmpeg_write_sec
        performance["ffmpeg_close_sec"] = ffmpeg_close_sec
        qc_status = "completed"
        t_publish = time.perf_counter()
        video_sha = sha256_file(video_tmp)
        _write_s2_pose_run_meta(
            out_job,
            source=source,
            plan=plan,
            frames_written=frames_written,
            qc_status=qc_status,
        )
        _write_s2_pose_settings(out_job, source=source, plan=plan, style=style)
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
            renderer_version=OVERLAY_S2_POSE_RENDERER_VERSION,
            extra={
                "export_kind": "s2_pose_qc_clip",
                "qc_status": qc_status,
                "renderer_version": OVERLAY_S2_POSE_RENDERER_VERSION,
                "identity_semantics": {
                    "kind": "provisional_s2_predictions",
                    "label_prefix": "P",
                    "note": (
                        "P{id} colors/labels are provisional S2 predictions, "
                        "not S3-accepted identities."
                    ),
                    "observed_provisional_tracks_in_clip": sorted(seen_tracks),
                },
                "confidence": confidence_disclosure,
                "skeleton": skeleton_disclosure,
                "performance_breakdown_sec": perf_rounded,
                "source": {
                    "source_kind": "s2_inferred_pose",
                    "s2_run_id": source.s2_run_id,
                    "s2_run_dir": str(source.s2_run_dir),
                    "s2_status": source.s2_status,
                    "pose_parquet_path": str(source.pose_parquet_path),
                    "pose_parquet_sha256": source.pose_parquet_sha256,
                    "prepared_sync_path": str(source.prepared_sync_path),
                    "prepared_sync_sha256": source.prepared_sync_sha256,
                    "prepared_video_path": str(source.prepared_video_path),
                    "prepared_video_sha256": source.prepared_video_sha256,
                    "source_digests": dict(source.source_digests),
                    "eligibility": "validate_s2_handoff_at_open_and_before_publish",
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
                    "video_filename": OVERLAY_QC_S2_POSE_VIDEO_FILENAME,
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
        _write_s2_pose_processing_log(
            out_job,
            source=source,
            plan=plan,
            frames_written=frames_written,
            qc_status=qc_status,
            confidence=confidence_disclosure,
            performance=perf_rounded,
        )
        shutil.rmtree(staging, ignore_errors=True)

        _shared._emit(
            progress_callback,
            {
                "stage": "published",
                "export_kind": "s2_pose_qc_clip",
                "job_dir": str(out_job),
                "video_path": str(video_final),
                "manifest_path": str(manifest_final),
                "frames_written": frames_written,
                "qc_status": qc_status,
                "performance_breakdown_sec": perf_rounded,
                "skeleton": skeleton_disclosure,
                "confidence": confidence_disclosure,
            },
        )
        return OverlayS2PoseQcExportResult(
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
        _cleanup_failed_s2_pose_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    except Exception:
        _cleanup_failed_s2_pose_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    finally:
        if capture is not None:
            capture.release()


def _resolve_s2_pose_job_dir(
    source: OverlayS2QcSource,
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
    candidate = root / f"{OVERLAY_QC_S2_POSE_JOB_PREFIX}{stamp}"
    if candidate.exists():
        for i in range(1, 1000):
            alt = root / f"{OVERLAY_QC_S2_POSE_JOB_PREFIX}{stamp}_{i:03d}"
            if not alt.exists():
                candidate = alt
                break
        else:
            raise OverlayExportError(
                "Could not allocate a unique S2 pose QC job directory."
            )
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _cleanup_failed_s2_pose_export(
    *,
    pipe: _shared._FfmpegPipeSession | None,
    staging: Path,
    job_dir: Path,
) -> None:
    _shared._cleanup_failed_export(pipe=pipe, staging=staging, job_dir=job_dir)
    if not job_dir.is_dir():
        return
    video = job_dir / OVERLAY_QC_S2_POSE_VIDEO_FILENAME
    manifest = job_dir / OVERLAY_QC_MANIFEST_FILENAME
    if video.exists() or manifest.exists():
        return
    remaining = [p for p in job_dir.iterdir()]
    if not remaining:
        shutil.rmtree(job_dir, ignore_errors=True)


def _write_s2_pose_run_meta(
    job_dir: Path,
    *,
    source: OverlayS2QcSource,
    plan: S2PoseQcExportPlan,
    frames_written: int,
    qc_status: str,
) -> None:
    payload = {
        "schema_version": "overlay_s2_pose_qc_run_meta_v1",
        "renderer_version": OVERLAY_S2_POSE_RENDERER_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "qc_status": qc_status,
        "s2_run_id": source.s2_run_id,
        "s2_run_dir": str(source.s2_run_dir),
        "s2_status": source.s2_status,
        "source_kind": "s2_inferred_pose",
        "frames_written": frames_written,
        "plan": plan.to_dict(),
        "source_digests": dict(source.source_digests),
    }
    (job_dir / OVERLAY_QC_RUN_META_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_s2_pose_settings(
    job_dir: Path,
    *,
    source: OverlayS2QcSource,
    plan: S2PoseQcExportPlan,
    style: DrawConfig,
) -> None:
    payload = {
        "schema_version": "overlay_s2_pose_qc_settings_v1",
        "renderer_version": OVERLAY_S2_POSE_RENDERER_VERSION,
        "export_kind": "s2_pose_qc_clip",
        "s2_run_id": source.s2_run_id,
        "draw_skeleton": plan.draw_skeleton,
        "show_confidence": plan.show_confidence,
        "include_frame_counter": plan.include_frame_counter,
        "prepared_interval": plan.prepared_interval.to_dict(),
        "selected_layers": [x.value for x in plan.selected_layers],
        "style": style.to_dict(),
        "skeleton": source.skeleton_disclosure(),
        "confidence_mapping_id": S2_SCORE_VISUAL_MAPPING_ID,
    }
    (job_dir / OVERLAY_QC_SETTINGS_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )


def _write_s2_pose_processing_log(
    job_dir: Path,
    *,
    source: OverlayS2QcSource,
    plan: S2PoseQcExportPlan,
    frames_written: int,
    qc_status: str,
    confidence: Mapping[str, Any],
    performance: Mapping[str, float] | None = None,
) -> None:
    disclosure = source.skeleton_disclosure()
    lines = [
        "overlay_s2_pose_qc processing log",
        f"created_at_utc: {datetime.now(UTC).isoformat()}",
        f"qc_status: {qc_status}",
        f"renderer_version: {OVERLAY_S2_POSE_RENDERER_VERSION}",
        f"s2_run_id: {source.s2_run_id}",
        f"s2_run_dir: {source.s2_run_dir}",
        f"pose_parquet_sha256: {source.pose_parquet_sha256}",
        f"prepared_video: {source.prepared_video_path}",
        f"prepared_video_sha256: {source.prepared_video_sha256}",
        f"prepared_sync_sha256: {source.prepared_sync_sha256}",
        "timing_authority: s1_prepared_frame_identity (not dense 120 Hz)",
        "identity_semantics: provisional S2 predictions (P{{id}}; not S3-accepted)",
        f"plateau_policy: {plan.plateau_policy.value}",
        (
            "prepared_interval: "
            f"[{plan.prepared_interval.start}, {plan.prepared_interval.end}] "
            f"({plan.prepared_interval.frame_count} frames)"
        ),
        f"frames_written: {frames_written}",
        f"show_confidence: {plan.show_confidence}",
        f"confidence_mapping_id: {confidence.get('mapping_id')}",
        f"confidence_summary: {confidence.get('mapping_summary')}",
        f"skeleton_mode: {disclosure['skeleton_mode']}",
        f"skeleton_nodes_only: {disclosure['skeleton_nodes_only']}",
        f"skeleton_source: {source.skeleton_source}",
        f"skeleton_edge_count: {len(source.skeleton_edges)}",
        f"skeleton_disclosure: {disclosure['skeleton_disclosure']}",
        "existing_s2_overlay_mp4_path_unchanged: true",
        "scientific_artifacts_mutated: false",
    ]
    if performance:
        lines.append("performance_breakdown_sec:")
        for key in sorted(performance):
            lines.append(f"  {key}: {float(performance[key]):.6f}")
    (job_dir / OVERLAY_QC_PROCESSING_LOG_FILENAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "OverlayS2PoseQcExportResult",
    "build_s2_pose_adapters",
    "export_s2_pose_qc_clip",
]
