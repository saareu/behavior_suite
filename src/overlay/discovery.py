"""Discover finalized overlay QC export jobs under results_exports/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from overlay.contracts import (
    OVERLAY_QC_EXPORT_DIRNAME,
    OVERLAY_QC_JOB_PREFIX,
    OVERLAY_QC_MANIFEST_FILENAME,
    OVERLAY_QC_RUN_META_FILENAME,
    OVERLAY_QC_S4_POSE_JOB_PREFIX,
    OVERLAY_QC_S4_POSE_VIDEO_FILENAME,
    OVERLAY_QC_VIDEO_FILENAME,
)
from results_packaging.export import load_json_object

EXPORT_FAMILY_BEHAVIORAL = "behavioral"
EXPORT_FAMILY_S4_POSE = "s4_pose"

STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"


@dataclass(frozen=True, slots=True)
class OverlayQcJobSummary:
    """Discovery summary for one overlay QC export job directory."""

    job_id: str
    job_dir: Path
    export_family: str
    status: str
    source_kind: str | None
    package_run_id: str | None
    review_run_id: str | None
    s4_run_id: str | None
    layers: tuple[str, ...]
    prepared_frame_idx_start: int | None
    prepared_frame_idx_end: int | None
    frames_written: int | None
    created_at_utc: str | None
    has_manifest: bool
    has_video: bool
    video_path: Path | None
    video_filename: str | None
    notes: tuple[str, ...] = ()

    @property
    def is_completed(self) -> bool:
        return (
            self.status == STATUS_COMPLETED
            and self.has_manifest
            and self.has_video
            and self.video_path is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "export_family": self.export_family,
            "status": self.status,
            "source_kind": self.source_kind,
            "package_run_id": self.package_run_id,
            "review_run_id": self.review_run_id,
            "s4_run_id": self.s4_run_id,
            "layers": list(self.layers),
            "prepared_frame_idx_start": self.prepared_frame_idx_start,
            "prepared_frame_idx_end": self.prepared_frame_idx_end,
            "frames_written": self.frames_written,
            "created_at_utc": self.created_at_utc,
            "has_manifest": self.has_manifest,
            "has_video": self.has_video,
            "video_path": None if self.video_path is None else str(self.video_path),
            "video_filename": self.video_filename,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class OverlayQcProjectSummary:
    session_root: Path
    exports_root: Path
    jobs: tuple[OverlayQcJobSummary, ...]

    @property
    def completed_jobs(self) -> tuple[OverlayQcJobSummary, ...]:
        return tuple(job for job in self.jobs if job.is_completed)

    @property
    def behavioral_jobs(self) -> tuple[OverlayQcJobSummary, ...]:
        return tuple(
            job for job in self.jobs if job.export_family == EXPORT_FAMILY_BEHAVIORAL
        )

    @property
    def s4_pose_jobs(self) -> tuple[OverlayQcJobSummary, ...]:
        return tuple(
            job for job in self.jobs if job.export_family == EXPORT_FAMILY_S4_POSE
        )


def summarize_overlay_qc_project(
    session_root: Path | str,
    *,
    exports_root: Path | str | None = None,
) -> OverlayQcProjectSummary:
    """Scan ``results_exports/`` for ``overlay_qc_behavioral__*`` and ``overlay_qc_s4_pose__*``.

    Other export families (media, CSV, PI report) are ignored.
    """

    root = Path(session_root).expanduser().resolve()
    exports = (
        Path(exports_root).expanduser().resolve()
        if exports_root is not None
        else (root / OVERLAY_QC_EXPORT_DIRNAME)
    )
    jobs: list[OverlayQcJobSummary] = []
    if exports.is_dir():
        for child in sorted(exports.iterdir(), key=lambda p: p.name, reverse=True):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            family = _family_for_dirname(child.name)
            if family is None:
                continue
            jobs.append(_summarize_job(child, export_family=family))
    return OverlayQcProjectSummary(
        session_root=root,
        exports_root=exports,
        jobs=tuple(jobs),
    )


def _family_for_dirname(name: str) -> str | None:
    if name.startswith(OVERLAY_QC_JOB_PREFIX):
        return EXPORT_FAMILY_BEHAVIORAL
    if name.startswith(OVERLAY_QC_S4_POSE_JOB_PREFIX):
        return EXPORT_FAMILY_S4_POSE
    return None


def _summarize_job(job_dir: Path, *, export_family: str) -> OverlayQcJobSummary:
    meta_path = job_dir / OVERLAY_QC_RUN_META_FILENAME
    manifest_path = job_dir / OVERLAY_QC_MANIFEST_FILENAME
    video_name = (
        OVERLAY_QC_VIDEO_FILENAME
        if export_family == EXPORT_FAMILY_BEHAVIORAL
        else OVERLAY_QC_S4_POSE_VIDEO_FILENAME
    )
    video_path = job_dir / video_name
    has_video = video_path.is_file()
    has_manifest = manifest_path.is_file()

    source_kind: str | None = None
    package_run_id: str | None = None
    review_run_id: str | None = None
    s4_run_id: str | None = None
    layers: tuple[str, ...] = ()
    prep_start: int | None = None
    prep_end: int | None = None
    frames_written: int | None = None
    created_at: str | None = None
    notes: list[str] = []
    meta_status: str | None = None

    if meta_path.is_file():
        try:
            meta = load_json_object(meta_path)
            meta_status = _optional_str(meta.get("qc_status") or meta.get("status"))
            source_kind = _optional_str(meta.get("source_kind"))
            package_run_id = _optional_str(meta.get("package_run_id"))
            review_run_id = _optional_str(meta.get("review_run_id"))
            s4_run_id = _optional_str(meta.get("s4_run_id"))
            created_at = _optional_str(meta.get("created_at_utc"))
            frames_written = _optional_int(meta.get("frames_written"))
            plan = meta.get("plan")
            if isinstance(plan, dict):
                layers = _layers_from_plan(plan)
                prep_start, prep_end = _interval_from_plan(plan)
                note_raw = plan.get("notes")
                if isinstance(note_raw, list):
                    notes.extend(str(n) for n in note_raw)
        except Exception:  # noqa: BLE001 - discovery must stay best-effort
            notes.append("run_meta unreadable")

    if has_manifest:
        try:
            manifest = load_json_object(manifest_path)
            if not layers:
                layer_raw = manifest.get("layers") or manifest.get("selected_layers")
                if isinstance(layer_raw, list):
                    layers = tuple(str(x) for x in layer_raw)
            extra = manifest.get("extra")
            if isinstance(extra, dict):
                source_kind = source_kind or _optional_str(extra.get("source_kind"))
                package_run_id = package_run_id or _optional_str(
                    extra.get("package_run_id")
                )
                review_run_id = review_run_id or _optional_str(extra.get("review_run_id"))
                s4_run_id = s4_run_id or _optional_str(extra.get("s4_run_id"))
            source_runs = manifest.get("source_runs")
            if isinstance(source_runs, list):
                for run in source_runs:
                    if not isinstance(run, dict):
                        continue
                    rid = _optional_str(run.get("run_id"))
                    kind = _optional_str(run.get("kind") or run.get("stage"))
                    if rid and kind and "s4" in kind.lower() and s4_run_id is None:
                        s4_run_id = rid
            frames_written = frames_written or _optional_int(
                manifest.get("output_frame_count") or manifest.get("frame_count")
            )
        except Exception:  # noqa: BLE001
            notes.append("manifest unreadable")

    status = _classify_status(
        meta_status=meta_status,
        has_manifest=has_manifest,
        has_video=has_video,
    )
    return OverlayQcJobSummary(
        job_id=job_dir.name,
        job_dir=job_dir,
        export_family=export_family,
        status=status,
        source_kind=source_kind,
        package_run_id=package_run_id,
        review_run_id=review_run_id,
        s4_run_id=s4_run_id,
        layers=layers,
        prepared_frame_idx_start=prep_start,
        prepared_frame_idx_end=prep_end,
        frames_written=frames_written,
        created_at_utc=created_at,
        has_manifest=has_manifest,
        has_video=has_video,
        video_path=video_path if has_video else None,
        video_filename=video_name if has_video else None,
        notes=tuple(notes),
    )


def _classify_status(
    *,
    meta_status: str | None,
    has_manifest: bool,
    has_video: bool,
) -> str:
    if meta_status in {STATUS_CANCELED, STATUS_FAILED}:
        return meta_status
    if has_manifest and has_video:
        return STATUS_COMPLETED
    if meta_status == STATUS_COMPLETED and not (has_manifest and has_video):
        return STATUS_INCOMPLETE
    return STATUS_INCOMPLETE


def _layers_from_plan(plan: dict[str, Any]) -> tuple[str, ...]:
    raw = plan.get("selected_layers")
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    return ()


def _interval_from_plan(plan: dict[str, Any]) -> tuple[int | None, int | None]:
    interval = plan.get("prepared_interval")
    if isinstance(interval, dict):
        return (
            _optional_int(interval.get("prepared_frame_idx_start_inclusive")),
            _optional_int(interval.get("prepared_frame_idx_end_inclusive")),
        )
    return None, None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


__all__ = [
    "EXPORT_FAMILY_BEHAVIORAL",
    "EXPORT_FAMILY_S4_POSE",
    "STATUS_CANCELED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_INCOMPLETE",
    "OverlayQcJobSummary",
    "OverlayQcProjectSummary",
    "summarize_overlay_qc_project",
]
