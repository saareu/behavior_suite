"""C3-4 mixed pose + behavioral QC export (shared compositor + streaming ffmpeg).

Does not invent scientific pose or behavioral states. Does not implement a new
rendering engine. Supports only explicit compatible S4+S8, S4+S10, S3+S8 pairs.
"""

from __future__ import annotations

import contextlib
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
from overlay.compositor import OverlayCompositor
from overlay.contracts import (
    OVERLAY_MIXED_RENDERER_VERSION,
    OVERLAY_QC_EXPORT_DIRNAME,
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_QC_MIXED_JOB_PREFIX,
    OVERLAY_QC_MIXED_VIDEO_FILENAME,
    OVERLAY_QC_PROCESSING_LOG_FILENAME,
    OVERLAY_QC_RUN_META_FILENAME,
    OVERLAY_QC_SETTINGS_FILENAME,
    OVERLAY_QC_STAGING_DIRNAME,
    S4_PROVENANCE_NAMES,
    S4_PROVENANCE_VISUAL_CLASS_BY_CODE,
    BehaviorSourceKind,
    FrameDomain,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    PlateauPolicy,
    PoseSourceKind,
)
from overlay.evidence import collect_evidence
from overlay.export import build_behavioral_adapters, build_s4_pose_adapters
from overlay.export_s3 import build_s3_pose_adapters
from overlay.plan import (
    MixedPoseBehaviorQcExportPlan,
    dense_sample_window_for_prepared_interval,
)
from overlay.provenance import (
    OverlayArtifactManifest,
    PreparedVideoIdentity,
    build_overlay_artifact_manifest,
)
from overlay.source_mixed import (
    OverlayMixedQcSource,
    verify_overlay_mixed_qc_source_integrity,
)
from overlay.style import DrawConfig
from overlay.timing import PreparedFrameTiming, resolve_prepared_frame_timing
from project.file_digest import sha256_file
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
)
from results_packaging.media_csv_export.video import probe_encoded_clip


@dataclass(frozen=True, slots=True)
class OverlayMixedQcExportResult:
    """Published mixed pose + behavioral QC artifact references."""

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


def build_mixed_adapters(
    source: OverlayMixedQcSource,
    *,
    plan: MixedPoseBehaviorQcExportPlan,
) -> dict[LayerId, OverlayStageAdapter]:
    """Compose pose + behavioral adapters for the selected mixed layers.

    Dense windows are resolved from each source's own SyncMap. S4 pose never
    inherits the behavioral dense range; S3 pose uses prepared-frame indexing.
    """

    behavior_dense_range = dense_sample_window_for_prepared_interval(
        source.sync, plan.prepared_interval
    )
    adapters: dict[LayerId, OverlayStageAdapter] = {}

    pose_layers = tuple(
        layer
        for layer in plan.selected_layers
        if layer in {LayerId.S3_POSE_TRACKED, LayerId.S4_POSE_FINAL}
    )
    behavior_layers = tuple(
        layer
        for layer in plan.selected_layers
        if layer
        in {
            LayerId.S8_EXCLUSIVE_A_TO_B,
            LayerId.S8_EXCLUSIVE_B_TO_A,
            LayerId.S8_CONTACT,
            LayerId.S8_SBS,
            LayerId.DEBUG_FRAME_COUNTER,
        }
    )

    if source.pose_source_kind is PoseSourceKind.S4_FINALIZED:
        if source.pose_s4 is None:
            raise OverlayExportError("Mixed S4 export missing pose_s4 source.")
        pose_dense_range = dense_sample_window_for_prepared_interval(
            source.pose_s4.sync, plan.prepared_interval
        )
        adapters.update(
            build_s4_pose_adapters(
                source.pose_s4,
                dense_sample_range=pose_dense_range,
                selected_layers=pose_layers,
            )
        )
    else:
        if source.pose_s3 is None:
            raise OverlayExportError("Mixed S3 export missing pose_s3 source.")
        prepared_range = (
            int(plan.prepared_interval.start),
            int(plan.prepared_interval.end),
        )
        adapters.update(
            build_s3_pose_adapters(
                source.pose_s3,
                prepared_frame_range=prepared_range,
                selected_layers=pose_layers,
            )
        )

    # Behavioral adapters own the debug counter when selected; avoid double-register.
    behavior_only = tuple(
        layer for layer in behavior_layers if layer != LayerId.DEBUG_FRAME_COUNTER
    )
    adapters.update(
        build_behavioral_adapters(
            source.behavior,
            dense_sample_range=behavior_dense_range,
            selected_layers=behavior_only,
        )
    )
    if LayerId.DEBUG_FRAME_COUNTER in plan.selected_layers:
        adapters[LayerId.DEBUG_FRAME_COUNTER] = _shared._DebugAdapter()
    return adapters


def export_mixed_pose_behavior_qc_clip(
    *,
    source: OverlayMixedQcSource,
    plan: MixedPoseBehaviorQcExportPlan,
    output_root: Path | str | None = None,
    job_dir: Path | str | None = None,
    ffmpeg_executable: Path | str | None = None,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    progress_callback: _shared.ProgressCallback | None = None,
    cancellation_requested: _shared.CancellationCallback | None = None,
    verify_encoded_boundaries: bool = True,
) -> OverlayMixedQcExportResult:
    """Export one short prepared-rate mixed pose + behavioral QC MP4.

    Opt-in only. Pins pose and behavioral digests independently and revalidates
    before publication. Encodes at prepared-video FPS via the shared compositor.
    """

    if plan.frame_domain is not FrameDomain.PREPARED:
        raise OverlayExportError(
            "C3-4 mixed QC export requires prepared frame domain "
            f"(got {plan.frame_domain.value})."
        )
    if plan.plateau_policy is not PlateauPolicy.LAST_IN_PLATEAU:
        raise OverlayExportError(
            "C3-4 mixed QC export requires LAST_IN_PLATEAU policy "
            f"(got {plan.plateau_policy.value})."
        )
    if plan.pose_source_kind is not source.pose_source_kind:
        raise OverlayExportError(
            "Mixed plan pose_source_kind does not match resolved source "
            f"({plan.pose_source_kind.value} vs {source.pose_source_kind.value})."
        )
    if source.behavior_source_kind is BehaviorSourceKind.S10_REVIEWED and (
        plan.pose_source_kind is PoseSourceKind.S3_ACCEPTED
    ):
        raise OverlayExportError(
            "C3-4 does not support S3+S10 mixed export "
            "(refusing silent combination expansion)."
        )

    performance: dict[str, float] = {}
    t_phase = time.perf_counter()
    verify_overlay_mixed_qc_source_integrity(source)
    performance["source_validation_and_hashing_sec"] = time.perf_counter() - t_phase

    out_job = _resolve_mixed_job_dir(source, output_root=output_root, job_dir=job_dir)
    staging = out_job / OVERLAY_QC_STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    video_final = out_job / OVERLAY_QC_MIXED_VIDEO_FILENAME
    manifest_final = out_job / OVERLAY_QC_MANIFEST_FILENAME
    if video_final.exists() or manifest_final.exists():
        raise OverlayExportError(
            f"Refusing to overwrite existing mixed QC artifacts in {out_job}."
        )

    video_tmp = staging / OVERLAY_QC_MIXED_VIDEO_FILENAME
    t_phase = time.perf_counter()
    adapters = build_mixed_adapters(source, plan=plan)
    performance["adapter_setup_sec"] = time.perf_counter() - t_phase

    style_extra: dict[str, Any] = {
        **dict(plan.style.extra),
        "skeleton_edges": [list(e) for e in source.skeleton_edges],
    }
    if source.pose_source_kind is PoseSourceKind.S3_ACCEPTED and source.pose_s3 is not None:
        # Mixed S3 path does not burn correction-event markers by default
        # (behavioral HUD is the primary annotation); keep journals available
        # but disabled unless plan style already requested them.
        style_extra.setdefault("show_machine_corrections", False)
        style_extra.setdefault("show_manual_corrections", False)
        if style_extra["show_machine_corrections"]:
            style_extra["s3_machine_events_by_frame"] = (
                source.pose_s3.machine_events.by_frame_dict(
                    (
                        int(plan.prepared_interval.start),
                        int(plan.prepared_interval.end),
                    )
                )
            )
        if style_extra["show_manual_corrections"]:
            style_extra["s3_manual_events_by_frame"] = (
                source.pose_s3.manual_events.by_frame_dict(
                    (
                        int(plan.prepared_interval.start),
                        int(plan.prepared_interval.end),
                    )
                )
            )

    style = DrawConfig(
        **{
            **plan.style.to_dict(),
            "draw_skeleton": bool(plan.draw_skeleton),
            "prepared_video_fps": float(source.video_probe.fps),
            "hud_direction_mode": plan.hud_mode.value,
            "contact_led_enabled": bool(plan.include_contact),
            "sbs_led_enabled": bool(plan.include_sbs),
            "legend_text": True,
            "extra": style_extra,
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
    notes: list[str] = list(source.compatibility_notes)
    notes.extend(plan.notes)
    skeleton = source.skeleton_disclosure()
    notes.append(str(skeleton["skeleton_disclosure"]))
    frames_written = 0
    first_timing = None
    last_timing = None
    seen_visual_classes: set[str] = set()
    seen_provenance_codes: set[int] = set()
    seen_behavior_labels: set[str] = set()
    timing_resolution_sec = 0.0
    frame_composition_sec = 0.0
    ffmpeg_write_sec = 0.0
    decode_sec = 0.0

    _shared._emit(
        progress_callback,
        {
            "stage": "encode_start",
            "export_kind": "mixed_pose_behavior_qc_clip",
            "job_dir": str(out_job),
            "prepared_frame_idx_start": plan.prepared_interval.start,
            "prepared_frame_idx_end": plan.prepared_interval.end,
            "expected_frames": expected,
            "fps": fps,
            "pose_source_kind": source.pose_source_kind.value,
            "behavior_source_kind": source.behavior_source_kind.value,
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
            # Independent timing authorities per prepared frame:
            # - behavior: package SyncMap LAST_IN_PLATEAU (bundle primary timing)
            # - S4 pose: pose SyncMap LAST_IN_PLATEAU (layer override)
            # - S3 pose: prepared_frame_idx direct (adapter ignores dense timing)
            layer_timing: dict[LayerId, PreparedFrameTiming] = {}
            pose_timing_note: PreparedFrameTiming | None = None
            if (
                source.pose_source_kind is PoseSourceKind.S4_FINALIZED
                and source.pose_s4 is not None
            ):
                pose_timing_note = resolve_prepared_frame_timing(
                    source.pose_s4.sync,
                    prep,
                    plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
                )
                layer_timing[LayerId.S4_POSE_FINAL] = pose_timing_note
            evidence = collect_evidence(
                source.sync,
                prep,
                adapters,
                selected_layers=plan.selected_layers,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
                layer_timing=layer_timing or None,
            )
            timing_resolution_sec += time.perf_counter() - t_timing

            pose_layer = (
                LayerId.S4_POSE_FINAL
                if source.pose_source_kind is PoseSourceKind.S4_FINALIZED
                else LayerId.S3_POSE_TRACKED
            )
            pose_obs = evidence.get(pose_layer)
            if pose_obs is not None and pose_obs.available:
                for pt in pose_obs.payload.get("points") or []:
                    if "provenance" in pt:
                        seen_provenance_codes.add(int(pt["provenance"]))
                    if pt.get("visual_class"):
                        seen_visual_classes.add(str(pt["visual_class"]))
            for layer_id in (
                LayerId.S8_EXCLUSIVE_A_TO_B,
                LayerId.S8_EXCLUSIVE_B_TO_A,
            ):
                beh = evidence.get(layer_id)
                if beh is not None and beh.available:
                    label = beh.payload.get("behavior_label") or beh.payload.get("label")
                    if label is not None:
                        seen_behavior_labels.add(str(label))

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
                pose_dense = (
                    None
                    if pose_timing_note is None
                    else pose_timing_note.dense_sample_idx
                )
                notes.append(
                    "prepared_frame_idx="
                    f"{prep}->behavior_dense_sample_idx="
                    f"{evidence.timing.dense_sample_idx}"
                    f" pose_dense_sample_idx={pose_dense}"
                    f" policy={evidence.timing.plateau_policy.value}"
                    f" behavior_available={evidence.timing.available}"
                    f" pose={source.pose_source_kind.value}"
                    f" behavior={source.behavior_source_kind.value}"
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
            if (
                frames_written == 1
                or frames_written == expected
                or frames_written % 30 == 0
            ):
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

        verify_overlay_mixed_qc_source_integrity(source)

        source_identities = _build_source_identities(source)
        prepared_video = PreparedVideoIdentity(
            path=source.prepared_video_path,
            digest_sha256=source.prepared_video_sha256,
            fps=fps,
            frame_count=int(source.video_probe.n_frames),
            extra={
                "width": width,
                "height": height,
                "sync_authority": source.sync_authority,
            },
        )
        example_timing = last_timing or first_timing
        if example_timing is None:
            example_timing = resolve_prepared_frame_timing(
                source.sync,
                plan.prepared_interval.start,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            )

        performance["prepared_video_decode_sec"] = decode_sec
        performance["timing_resolution_sec"] = timing_resolution_sec
        performance["frame_composition_sec"] = frame_composition_sec
        performance["ffmpeg_encoding_sec"] = ffmpeg_write_sec + ffmpeg_close_sec
        performance["ffmpeg_stdin_write_sec"] = ffmpeg_write_sec
        performance["ffmpeg_close_sec"] = ffmpeg_close_sec
        qc_status = "completed"
        t_publish = time.perf_counter()
        video_sha = sha256_file(video_tmp)
        _write_mixed_run_meta(
            out_job,
            source=source,
            plan=plan,
            frames_written=frames_written,
            qc_status=qc_status,
        )
        _write_mixed_settings(out_job, source=source, plan=plan, style=style)
        performance["sidecar_publication_sec"] = time.perf_counter() - t_publish
        perf_rounded = {k: round(float(v), 6) for k, v in performance.items()}

        provenance_map = {
            str(code): {
                "provenance_name": S4_PROVENANCE_NAMES[code],
                "visual_class": S4_PROVENANCE_VISUAL_CLASS_BY_CODE[code].value,
            }
            for code in sorted(S4_PROVENANCE_NAMES)
        }
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
            renderer_version=OVERLAY_MIXED_RENDERER_VERSION,
            extra={
                "export_kind": "mixed_pose_behavior_qc_clip",
                "qc_status": qc_status,
                "renderer_version": OVERLAY_MIXED_RENDERER_VERSION,
                "pose_source_kind": source.pose_source_kind.value,
                "behavior_source_kind": source.behavior_source_kind.value,
                "s4_provenance_visual_map": (
                    provenance_map
                    if source.pose_source_kind is PoseSourceKind.S4_FINALIZED
                    else None
                ),
                "observed_provenance_codes_in_clip": sorted(seen_provenance_codes),
                "observed_visual_classes_in_clip": sorted(seen_visual_classes),
                "observed_behavior_labels_in_clip": sorted(seen_behavior_labels),
                "skeleton": skeleton,
                "performance_breakdown_sec": perf_rounded,
                "compatibility_notes": list(source.compatibility_notes),
                "source": {
                    "export_family": "mixed_pose_behavior",
                    "pose_source_kind": source.pose_source_kind.value,
                    "behavior_source_kind": source.behavior_source_kind.value,
                    "pose_run_id": source.pose_run_id,
                    "pose_run_dir": str(source.pose_run_dir),
                    "package_run_id": source.behavior.package_run_id,
                    "package_dir": str(source.behavior.package_dir),
                    "review_run_id": source.behavior.review_run_id,
                    "review_run_dir": (
                        None
                        if source.behavior.review_run_dir is None
                        else str(source.behavior.review_run_dir)
                    ),
                    "prepared_video_path": str(source.prepared_video_path),
                    "prepared_video_sha256": source.prepared_video_sha256,
                    "source_digests": dict(source.source_digests),
                    "pose_source_digests": source.pose_source_digests(),
                    "behavior_source_digests": dict(source.behavior.source_digests),
                    "eligibility": "validated_at_open_and_before_publish",
                    "timing_authority": source.sync_authority,
                    "pose_timing": source.provenance.get("pose_timing"),
                    "behavior_timing": source.provenance.get("behavior_timing"),
                    "skeleton_source": source.skeleton_source,
                    "skeleton_edge_count": len(source.skeleton_edges),
                    "provenance": dict(source.provenance),
                },
                "plan": plan.to_dict(),
                "prepared_frame_range": plan.prepared_interval.to_dict(),
                "output": {
                    "video_filename": OVERLAY_QC_MIXED_VIDEO_FILENAME,
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
        _write_mixed_processing_log(
            out_job,
            source=source,
            plan=plan,
            frames_written=frames_written,
            qc_status=qc_status,
            skeleton=skeleton,
            performance=perf_rounded,
        )
        shutil.rmtree(staging, ignore_errors=True)

        _shared._emit(
            progress_callback,
            {
                "stage": "published",
                "export_kind": "mixed_pose_behavior_qc_clip",
                "job_dir": str(out_job),
                "video_path": str(video_final),
                "manifest_path": str(manifest_final),
                "frames_written": frames_written,
                "qc_status": qc_status,
                "performance_breakdown_sec": perf_rounded,
            },
        )
        return OverlayMixedQcExportResult(
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
        _cleanup_failed_mixed_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    except Exception:
        _cleanup_failed_mixed_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    finally:
        if capture is not None:
            capture.release()


def _build_source_identities(
    source: OverlayMixedQcSource,
) -> tuple[SourceIdentity, ...]:
    identities: list[SourceIdentity] = []
    if source.pose_source_kind is PoseSourceKind.S4_FINALIZED:
        assert source.pose_s4 is not None
        identities.append(
            SourceIdentity(
                subsystem="pose_finalization",
                run_id=source.pose_s4.s4_run_id,
                run_dir=source.pose_s4.s4_run_dir,
                artifact_path=source.pose_s4.final_pose_path,
                digest_sha256=source.pose_s4.final_pose_sha256,
                extra={
                    "legend": "final pose",
                    "domain": "dense",
                    "pose_source_kind": PoseSourceKind.S4_FINALIZED.value,
                    "s4_status": source.pose_s4.s4_status,
                },
            )
        )
    else:
        assert source.pose_s3 is not None
        identities.append(
            SourceIdentity(
                subsystem="tracking_correction",
                run_id=source.pose_s3.s3_run_id,
                run_dir=source.pose_s3.s3_run_dir,
                artifact_path=source.pose_s3.tracked_pose_path,
                digest_sha256=source.pose_s3.tracked_pose_sha256,
                extra={
                    "legend": "accepted tracked pose",
                    "domain": "prepared",
                    "pose_source_kind": PoseSourceKind.S3_ACCEPTED.value,
                    "acceptance_state": source.pose_s3.acceptance_state,
                    "not_s4_finalized": True,
                },
            )
        )

    reviewed = source.behavior_source_kind is BehaviorSourceKind.S10_REVIEWED
    identities.append(
        SourceIdentity(
            subsystem=(
                "behavioral_review"
                if reviewed
                else "behavioral_arbitration"
            ),
            run_id=(
                source.behavior.review_run_id
                if reviewed
                else source.behavior.package.source_s8.run_id
            ),
            run_dir=(
                source.behavior.review_run_dir
                if reviewed
                else source.behavior.package.source_s8.run_dir
            ),
            artifact_path=source.behavior.states_path,
            digest_sha256=source.behavior.states_sha256,
            extra={
                "legend": "reviewed behavior" if reviewed else "automated behavior",
                "domain": "dense",
                "behavior_source_kind": source.behavior_source_kind.value,
                "package_run_id": source.behavior.package_run_id,
                "review_run_id": source.behavior.review_run_id,
            },
        )
    )
    return tuple(identities)


def _resolve_mixed_job_dir(
    source: OverlayMixedQcSource,
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
    out = root / f"{OVERLAY_QC_MIXED_JOB_PREFIX}{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _cleanup_failed_mixed_export(
    *,
    pipe: _shared._FfmpegPipeSession | None,
    staging: Path,
    job_dir: Path,
) -> None:
    if pipe is not None:
        with contextlib.suppress(Exception):
            if pipe.proc.stdin is not None and not pipe.proc.stdin.closed:
                pipe.proc.stdin.close()
        with contextlib.suppress(Exception):
            pipe.proc.kill()
        with contextlib.suppress(Exception):
            pipe.proc.wait(timeout=5)
    shutil.rmtree(staging, ignore_errors=True)
    # Remove empty job dir created for this failed attempt.
    try:
        if job_dir.is_dir() and not any(job_dir.iterdir()):
            job_dir.rmdir()
    except OSError:
        pass


def _write_mixed_run_meta(
    job_dir: Path,
    *,
    source: OverlayMixedQcSource,
    plan: MixedPoseBehaviorQcExportPlan,
    frames_written: int,
    qc_status: str,
) -> None:
    payload = {
        "schema": "overlay_qc_run_meta_v1",
        "export_family": "mixed_pose_behavior",
        "export_kind": "mixed_pose_behavior_qc_clip",
        "qc_status": qc_status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "renderer_version": OVERLAY_MIXED_RENDERER_VERSION,
        "pose_source_kind": source.pose_source_kind.value,
        "behavior_source_kind": source.behavior_source_kind.value,
        "source_kind": source.behavior_source_kind.value,
        "s4_run_id": (
            source.pose_s4.s4_run_id if source.pose_s4 is not None else None
        ),
        "s3_run_id": (
            source.pose_s3.s3_run_id if source.pose_s3 is not None else None
        ),
        "package_run_id": source.behavior.package_run_id,
        "review_run_id": source.behavior.review_run_id,
        "frames_written": int(frames_written),
        "prepared_interval": plan.prepared_interval.to_dict(),
        "plan": plan.to_dict(),
        "source_digests": dict(source.source_digests),
    }
    (job_dir / OVERLAY_QC_RUN_META_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_mixed_settings(
    job_dir: Path,
    *,
    source: OverlayMixedQcSource,
    plan: MixedPoseBehaviorQcExportPlan,
    style: DrawConfig,
) -> None:
    payload = {
        "export_family": "mixed_pose_behavior",
        "pose_source_kind": source.pose_source_kind.value,
        "behavior_source_kind": source.behavior_source_kind.value,
        "plan": plan.to_dict(),
        "style": style.to_dict(),
        "plateau_policy": PlateauPolicy.LAST_IN_PLATEAU.value,
        "frame_domain": FrameDomain.PREPARED.value,
        "renderer_version": OVERLAY_MIXED_RENDERER_VERSION,
    }
    (job_dir / OVERLAY_QC_SETTINGS_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_mixed_processing_log(
    job_dir: Path,
    *,
    source: OverlayMixedQcSource,
    plan: MixedPoseBehaviorQcExportPlan,
    frames_written: int,
    qc_status: str,
    skeleton: dict[str, Any],
    performance: dict[str, float],
) -> None:
    lines = [
        "C3-4 mixed pose + behavioral QC export",
        f"status={qc_status}",
        f"pose={source.pose_source_kind.value} run={source.pose_run_id}",
        f"behavior={source.behavior_source_kind.value} "
        f"package={source.behavior.package_run_id} "
        f"review={source.behavior.review_run_id}",
        f"prepared_interval={plan.prepared_interval.start}..{plan.prepared_interval.end}",
        f"frames_written={frames_written}",
        f"output_fps={source.video_probe.fps}",
        f"layers={[x.value for x in plan.selected_layers]}",
        f"skeleton_mode={skeleton.get('skeleton_mode')}",
        f"timing_authority={source.sync_authority}",
        "compatibility:",
        *[f"  - {n}" for n in source.compatibility_notes],
        "performance_sec:",
        *[f"  - {k}: {v}" for k, v in performance.items()],
    ]
    (job_dir / OVERLAY_QC_PROCESSING_LOG_FILENAME).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "OverlayMixedQcExportResult",
    "build_mixed_adapters",
    "export_mixed_pose_behavior_qc_clip",
]
