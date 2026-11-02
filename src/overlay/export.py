"""Streaming prepared-rate behavioral QC MP4 export (Increment C2-A).

Uses the shared compositor and ffmpeg/libx264. Does not mutate S8/S9/S10 science
and does not alter naked S9 or PI-report exports.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from overlay.adapters.base import OverlayStageAdapter, SourceIdentity
from overlay.adapters.behavior import (
    make_s8_contact,
    make_s8_exclusive_a_to_b,
    make_s8_exclusive_b_to_a,
    make_s8_sbs,
    make_s10_reviewed_contact,
    make_s10_reviewed_exclusive_a_to_b,
    make_s10_reviewed_exclusive_b_to_a,
    make_s10_reviewed_sbs,
)
from overlay.compositor import OverlayCompositor
from overlay.contracts import (
    OVERLAY_QC_EXPORT_DIRNAME,
    OVERLAY_QC_JOB_PREFIX,
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_QC_STAGING_DIRNAME,
    OVERLAY_QC_VIDEO_FILENAME,
    OVERLAY_RENDERER_VERSION,
    BehaviorSourceKind,
    FrameDomain,
    LayerId,
    OverlayExportCancelled,
    OverlayExportError,
    PlateauPolicy,
)
from overlay.evidence import collect_evidence, debug_counter_observation
from overlay.plan import (
    BehavioralQcExportPlan,
    dense_sample_window_for_prepared_interval,
)
from overlay.provenance import (
    OverlayArtifactManifest,
    PreparedVideoIdentity,
    build_overlay_artifact_manifest,
)
from overlay.source import (
    OverlayQcSource,
    verify_overlay_qc_source_integrity,
)
from overlay.style import DrawConfig
from overlay.timing import resolve_prepared_frame_timing
from project.file_digest import sha256_file
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
)
from results_packaging.media_csv_export.video import (
    probe_encoded_clip,
    resolve_ffmpeg_executable,
)

ProgressCallback = Callable[[dict[str, Any]], None]
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class OverlayQcExportResult:
    """Published behavioral QC artifact references."""

    job_dir: Path
    video_path: Path
    manifest_path: Path
    manifest: OverlayArtifactManifest
    frames_written: int
    output_fps: float
    unavailable_layers: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "video_path": str(self.video_path),
            "manifest_path": str(self.manifest_path),
            "frames_written": self.frames_written,
            "output_fps": self.output_fps,
            "unavailable_layers": list(self.unavailable_layers),
            "notes": list(self.notes),
            "manifest": self.manifest.to_dict(),
        }


@dataclass
class _DebugAdapter:
    layer_id: LayerId = LayerId.DEBUG_FRAME_COUNTER

    @property
    def source_identity(self) -> SourceIdentity:
        return SourceIdentity(subsystem="overlay_debug", run_id=None)

    @property
    def available(self) -> bool:
        return True

    def observe(self, timing, *, prepared_frame_idx: int):  # type: ignore[no-untyped-def]
        _ = prepared_frame_idx
        return debug_counter_observation(timing)


def build_behavioral_adapters(
    source: OverlayQcSource,
    *,
    dense_sample_range: tuple[int, int] | None,
    selected_layers: tuple[LayerId, ...],
) -> dict[LayerId, OverlayStageAdapter]:
    """Build behavior-only adapters (no pose tables) for the selected layers."""

    adapters: dict[LayerId, OverlayStageAdapter] = {}
    path = source.states_path
    run_id = (
        source.review_run_id
        if source.source_kind is BehaviorSourceKind.S10_REVIEWED
        else source.package.source_s8.run_id
    )
    run_dir = (
        source.review_run_dir
        if source.source_kind is BehaviorSourceKind.S10_REVIEWED
        else source.package.source_s8.run_dir
    )
    reviewed = source.source_kind is BehaviorSourceKind.S10_REVIEWED

    if LayerId.S8_EXCLUSIVE_A_TO_B in selected_layers:
        adapters[LayerId.S8_EXCLUSIVE_A_TO_B] = (
            make_s10_reviewed_exclusive_a_to_b(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
            if reviewed
            else make_s8_exclusive_a_to_b(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
        )
    if LayerId.S8_EXCLUSIVE_B_TO_A in selected_layers:
        adapters[LayerId.S8_EXCLUSIVE_B_TO_A] = (
            make_s10_reviewed_exclusive_b_to_a(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
            if reviewed
            else make_s8_exclusive_b_to_a(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
        )
    if LayerId.S8_CONTACT in selected_layers:
        adapters[LayerId.S8_CONTACT] = (
            make_s10_reviewed_contact(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
            if reviewed
            else make_s8_contact(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
        )
    if LayerId.S8_SBS in selected_layers:
        adapters[LayerId.S8_SBS] = (
            make_s10_reviewed_sbs(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
            if reviewed
            else make_s8_sbs(
                path,
                run_id=run_id,
                run_dir=run_dir,
                dense_sample_range=dense_sample_range,
            )
        )
    if LayerId.DEBUG_FRAME_COUNTER in selected_layers:
        adapters[LayerId.DEBUG_FRAME_COUNTER] = _DebugAdapter()
    return adapters


def export_behavioral_qc_clip(
    *,
    source: OverlayQcSource,
    plan: BehavioralQcExportPlan,
    output_root: Path | str | None = None,
    job_dir: Path | str | None = None,
    ffmpeg_executable: Path | str | None = None,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    progress_callback: ProgressCallback | None = None,
    cancellation_requested: CancellationCallback | None = None,
    verify_encoded_boundaries: bool = True,
) -> OverlayQcExportResult:
    """Export one short prepared-rate behavioral QC MP4 with provenance manifest.

    Writes under a new timestamped job directory (never overwrites a prior QC
    export). Uses a staging dir + atomic replace. Revalidates pinned source
    integrity before final publication.
    """

    if plan.frame_domain is not FrameDomain.PREPARED:
        raise OverlayExportError(
            "C2-A behavioral QC export requires prepared frame domain "
            f"(got {plan.frame_domain.value})."
        )
    if plan.plateau_policy is not PlateauPolicy.LAST_IN_PLATEAU:
        raise OverlayExportError(
            "C2-A behavioral QC export requires LAST_IN_PLATEAU policy "
            f"(got {plan.plateau_policy.value})."
        )

    verify_overlay_qc_source_integrity(source)

    out_job = _resolve_job_dir(source, output_root=output_root, job_dir=job_dir)
    staging = out_job / OVERLAY_QC_STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    video_final = out_job / OVERLAY_QC_VIDEO_FILENAME
    manifest_final = out_job / OVERLAY_QC_MANIFEST_FILENAME
    if video_final.exists() or manifest_final.exists():
        raise OverlayExportError(
            f"Refusing to overwrite existing overlay QC artifacts in {out_job}."
        )

    video_tmp = staging / OVERLAY_QC_VIDEO_FILENAME
    dense_range = dense_sample_window_for_prepared_interval(
        source.sync, plan.prepared_interval
    )
    adapters = build_behavioral_adapters(
        source,
        dense_sample_range=dense_range,
        selected_layers=plan.selected_layers,
    )
    style = DrawConfig(
        **{
            **plan.style.to_dict(),
            "hud_direction_mode": plan.hud_mode.value,
            "prepared_video_fps": float(source.video_probe.fps),
            "contact_led_enabled": plan.include_contact,
            "sbs_led_enabled": plan.include_sbs,
            "extra": dict(plan.style.extra),
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

    _emit(
        progress_callback,
        {
            "stage": "encode_start",
            "job_dir": str(out_job),
            "prepared_frame_idx_start": plan.prepared_interval.start,
            "prepared_frame_idx_end": plan.prepared_interval.end,
            "expected_frames": expected,
            "fps": fps,
        },
    )

    pipe: _FfmpegPipeSession | None = None
    capture: cv2.VideoCapture | None = None
    try:
        _check_cancel(cancellation_requested)
        capture = _open_capture_at(
            source.video_path, prepared_frame_idx=plan.prepared_interval.start
        )
        pipe = _start_ffmpeg_pipe(
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
            _check_cancel(cancellation_requested)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OverlayExportError(
                    f"Failed to decode prepared_frame_idx={prep} from "
                    f"{source.video_path}."
                )
            if frame.shape[1] != width or frame.shape[0] != height:
                raise OverlayExportError(
                    f"Decoded frame size mismatch at prepared_frame_idx={prep}: "
                    f"got {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}."
                )

            evidence = collect_evidence(
                source.sync,
                prep,
                adapters,
                selected_layers=plan.selected_layers,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            )
            composed = compositor.render_frame(
                frame,
                prepared_frame_idx=prep,
                evidence=evidence,
                selected_layers=plan.selected_layers,
                style=style,
            )
            unavailable_union.update(x.value for x in composed.unavailable_layers)
            if first_timing is None:
                first_timing = evidence.timing
            last_timing = evidence.timing

            # Traceability note for first/last frames only (avoid huge manifests).
            if prep in {
                plan.prepared_interval.start,
                plan.prepared_interval.end,
            }:
                notes.append(
                    "prepared_frame_idx="
                    f"{prep}->dense_sample_idx={evidence.timing.dense_sample_idx}"
                    f" policy={evidence.timing.plateau_policy.value}"
                    f" available={evidence.timing.available}"
                )

            raw = np.ascontiguousarray(composed.frame_bgr)
            if raw.dtype != np.uint8:
                raw = raw.astype(np.uint8, copy=False)
            try:
                pipe.proc.stdin.write(raw.tobytes())
            except BrokenPipeError as exc:
                detail = _ffmpeg_stderr(pipe)
                raise OverlayExportError(
                    f"ffmpeg pipe broke while writing prepared_frame_idx={prep}: "
                    f"{detail}"
                ) from exc
            frames_written += 1
            if frames_written == 1 or frames_written == expected or frames_written % 30 == 0:
                _emit(
                    progress_callback,
                    {
                        "stage": "encoding",
                        "frames_written": frames_written,
                        "expected_frames": expected,
                        "prepared_frame_idx": prep,
                        "dense_sample_idx": evidence.timing.dense_sample_idx,
                    },
                )

        pipe.proc.stdin.close()
        rc = pipe.proc.wait(timeout=120)
        if rc != 0 or not video_tmp.is_file() or video_tmp.stat().st_size <= 0:
            detail = _ffmpeg_stderr(pipe)
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

        # Integrity recheck immediately before publication.
        verify_overlay_qc_source_integrity(source)

        source_identities = tuple(
            adapters[layer].source_identity
            for layer in plan.selected_layers
            if layer in adapters and layer is not LayerId.DEBUG_FRAME_COUNTER
        )
        # Always pin source digests even for debug-only selections.
        if not source_identities:
            source_identities = (
                SourceIdentity(
                    subsystem=(
                        "behavioral_review"
                        if source.source_kind is BehaviorSourceKind.S10_REVIEWED
                        else "behavioral_arbitration"
                    ),
                    run_id=(
                        source.review_run_id
                        if source.source_kind is BehaviorSourceKind.S10_REVIEWED
                        else source.package.source_s8.run_id
                    ),
                    run_dir=(
                        source.review_run_dir
                        if source.source_kind is BehaviorSourceKind.S10_REVIEWED
                        else source.package.source_s8.run_dir
                    ),
                    artifact_path=source.states_path,
                    digest_sha256=source.states_sha256,
                    extra={"behavior_source_kind": source.source_kind.value},
                ),
            )

        prepared_video = PreparedVideoIdentity(
            path=source.video_path,
            digest_sha256=source.video_sha256,
            fps=fps,
            frame_count=int(source.video_probe.n_frames),
            extra={
                "width": width,
                "height": height,
                "probe_path": str(source.video_path),
            },
        )
        example_timing = last_timing or first_timing
        if example_timing is None:
            example_timing = resolve_prepared_frame_timing(
                source.sync,
                plan.prepared_interval.start,
                plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
            )
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
            extra={
                "export_kind": "behavioral_qc_clip",
                "renderer_version": OVERLAY_RENDERER_VERSION,
                "source": {
                    "source_kind": source.source_kind.value,
                    "package_run_id": source.package_run_id,
                    "package_dir": str(source.package_dir),
                    "review_run_id": source.review_run_id,
                    "review_run_dir": (
                        None
                        if source.review_run_dir is None
                        else str(source.review_run_dir)
                    ),
                    "states_path": str(source.states_path),
                    "states_sha256": source.states_sha256,
                    "source_digests": dict(source.source_digests),
                    "eligibility": "validated_at_open_and_before_publish",
                    "provenance": dict(source.provenance),
                },
                "plan": plan.to_dict(),
                "prepared_frame_range": plan.prepared_interval.to_dict(),
                "dense_sample_range": (
                    None if dense_range is None else [dense_range[0], dense_range[1]]
                ),
                "output": {
                    "video_filename": OVERLAY_QC_VIDEO_FILENAME,
                    "codec": codec,
                    "pixel_format": pixel_format,
                    "crf": int(crf),
                    "output_fps": fps,
                    "frame_count": frames_written,
                    "video_sha256": sha256_file(video_tmp),
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
        # Stamp unavailable layers from the union observed across frames.
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
        # Atomic publication: video then manifest.
        video_tmp.replace(video_final)
        manifest_tmp.replace(manifest_final)
        _write_run_meta(out_job, source=source, plan=plan, frames_written=frames_written)
        shutil.rmtree(staging, ignore_errors=True)

        _emit(
            progress_callback,
            {
                "stage": "published",
                "job_dir": str(out_job),
                "video_path": str(video_final),
                "manifest_path": str(manifest_final),
                "frames_written": frames_written,
            },
        )
        return OverlayQcExportResult(
            job_dir=out_job,
            video_path=video_final,
            manifest_path=manifest_final,
            manifest=manifest,
            frames_written=frames_written,
            output_fps=fps,
            unavailable_layers=tuple(sorted(unavailable_union)),
            notes=tuple(notes),
        )
    except OverlayExportCancelled:
        _cleanup_failed_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    except Exception:
        _cleanup_failed_export(pipe=pipe, staging=staging, job_dir=out_job)
        raise
    finally:
        if capture is not None:
            capture.release()


def _resolve_job_dir(
    source: OverlayQcSource,
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
    elif source.session_root is not None:
        root = Path(source.session_root) / OVERLAY_QC_EXPORT_DIRNAME
    else:
        root = Path(source.package_dir).parent.parent / OVERLAY_QC_EXPORT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    candidate = root / f"{OVERLAY_QC_JOB_PREFIX}{stamp}"
    if candidate.exists():
        # Collision-safe suffix; still never overwrite.
        for i in range(1, 1000):
            alt = root / f"{OVERLAY_QC_JOB_PREFIX}{stamp}_{i:03d}"
            if not alt.exists():
                candidate = alt
                break
        else:
            raise OverlayExportError("Could not allocate a unique overlay QC job directory.")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _open_capture_at(video_path: Path, *, prepared_frame_idx: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OverlayExportError(f"Could not open prepared video: {video_path}")
    if prepared_frame_idx > 0:
        # Prefer random-access seek; fall back to sequential skip when the
        # backend reports a large position mismatch after set.
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(prepared_frame_idx))
        pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(pos - int(prepared_frame_idx)) > 2:
            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise OverlayExportError(
                    f"Could not reopen prepared video for sequential seek: {video_path}"
                )
            for i in range(int(prepared_frame_idx)):
                ok, _ = capture.read()
                if not ok:
                    capture.release()
                    raise OverlayExportError(
                        f"Sequential seek failed before prepared_frame_idx="
                        f"{prepared_frame_idx} (stopped at {i})."
                    )
    return capture


@dataclass
class _FfmpegPipeSession:
    proc: subprocess.Popen[bytes]
    stderr_chunks: list[bytes]
    stderr_thread: threading.Thread


def _start_ffmpeg_pipe(
    *,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    ffmpeg_executable: Path | str | None,
    crf: int,
    codec: str,
    pixel_format: str,
) -> _FfmpegPipeSession:
    ffmpeg = resolve_ffmpeg_executable(ffmpeg_executable)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    cmd = [
        str(ffmpeg),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        f"{float(fps):.9f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        str(codec),
        "-pix_fmt",
        str(pixel_format),
        "-crf",
        str(int(crf)),
        "-fps_mode",
        "cfr",
        "-r",
        f"{float(fps):.9f}",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise OverlayExportError(f"Failed to invoke ffmpeg: {exc}") from exc

    chunks: list[bytes] = []

    def _drain() -> None:
        assert proc.stderr is not None
        while True:
            block = proc.stderr.read(4096)
            if not block:
                break
            chunks.append(block)

    thread = threading.Thread(target=_drain, name="overlay-ffmpeg-stderr", daemon=True)
    thread.start()
    return _FfmpegPipeSession(proc=proc, stderr_chunks=chunks, stderr_thread=thread)


def _ffmpeg_stderr(session: _FfmpegPipeSession | subprocess.Popen[bytes] | None) -> str:
    if session is None:
        return ""
    if isinstance(session, _FfmpegPipeSession):
        session.stderr_thread.join(timeout=2.0)
        blob = b"".join(session.stderr_chunks)
    else:
        try:
            blob = session.stderr.read() if session.stderr is not None else b""
        except Exception:  # noqa: BLE001
            return ""
    try:
        return blob.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return repr(blob)


def _cleanup_failed_export(
    *,
    pipe: _FfmpegPipeSession | None,
    staging: Path,
    job_dir: Path,
) -> None:
    if pipe is not None:
        proc = pipe.proc
        with contextlib.suppress(Exception):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            pipe.stderr_thread.join(timeout=1.0)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    # Remove empty newly created job dir if publication never completed.
    video = job_dir / OVERLAY_QC_VIDEO_FILENAME
    manifest = job_dir / OVERLAY_QC_MANIFEST_FILENAME
    if job_dir.is_dir() and not video.exists() and not manifest.exists():
        remaining = [p for p in job_dir.iterdir()]
        if not remaining:
            shutil.rmtree(job_dir, ignore_errors=True)


def _write_run_meta(
    job_dir: Path,
    *,
    source: OverlayQcSource,
    plan: BehavioralQcExportPlan,
    frames_written: int,
) -> None:
    payload = {
        "schema_version": "overlay_qc_run_meta_v1",
        "renderer_version": OVERLAY_RENDERER_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_kind": source.source_kind.value,
        "package_run_id": source.package_run_id,
        "review_run_id": source.review_run_id,
        "frames_written": frames_written,
        "plan": plan.to_dict(),
    }
    (job_dir / "run_meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _check_cancel(callback: CancellationCallback | None) -> None:
    if callback is not None and bool(callback()):
        raise OverlayExportCancelled("Overlay QC export cancelled by caller.")


def _emit(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    if callback is not None:
        callback(dict(event))
