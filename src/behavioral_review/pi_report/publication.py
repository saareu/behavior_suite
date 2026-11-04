"""Immutable PI report publication: stage → validate → finalize (never overwrite)."""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from behavioral_review.gallery import GalleryQuery
from behavioral_review.pi_report.bundle import (
    build_report_data_payload,
    write_report_bundle_files,
)
from behavioral_review.pi_report.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_MOVFLAGS,
    DEFAULT_PIXEL_FORMAT,
    DEFAULT_POST_CONTEXT_SEC,
    DEFAULT_PRE_CONTEXT_SEC,
    INDEX_HTML_FILENAME,
    MEDIA_DIRNAME,
    PI_REPORT_JOB_KIND,
    PI_REPORT_MANIFEST_SCHEMA_VERSION,
    PI_REPORT_QC_SCHEMA_VERSION,
    PI_REPORT_RUN_META_SCHEMA_VERSION,
    PI_REPORT_RUN_PREFIX,
    PI_REPORT_SETTINGS_SCHEMA_VERSION,
    PI_SOURCE_REVIEWED,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    REPORT_DATA_FILENAME,
    REPORT_MANIFEST_FILENAME,
    REPORT_SCOPE_COMPLETE,
    REPORT_SCOPE_SELECTED,
    REQUIRED_REPORT_FILES,
    RESULTS_EXPORTS_DIRNAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STAGING_DIRNAME,
    STATUS_CANCELED,
    PiReportCancelledError,
    PiReportPublicationError,
    PiReportValidationError,
)
from behavioral_review.pi_report.media import (
    MediaMaterializationResult,
    materialize_report_media,
    materialize_report_overlay_media,
)
from behavioral_review.pi_report.source import (
    PiReportSource,
    resolve_pi_report_source,
    verify_pinned_source_integrity,
)
from behavioral_review.source import assert_no_upstream_writes, snapshot_upstream_fingerprints
from project.file_digest import sha256_file
from results_packaging.contracts import (
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from results_packaging.export import write_json, write_yaml

ProgressCallback = Callable[[str, int, int], None]
CancellationCallback = Callable[[], bool]

_PROTECTED_SESSION_SUBDIRS = (
    "preprocess",
    "pose_inference",
    "tracking_correction",
    "pose_finalization",
    "blob_features",
    "combined_features",
    "behavioral_inference",
    "behavioral_arbitration",
    "results_package",
    "behavioral_review",
)


@dataclass(frozen=True, slots=True)
class PiReportSettings:
    """Operator-visible PI report generation settings."""

    pre_context_sec: float = DEFAULT_PRE_CONTEXT_SEC
    post_context_sec: float = DEFAULT_POST_CONTEXT_SEC
    codec: str = DEFAULT_CODEC
    crf: int = DEFAULT_CRF
    pixel_format: str = DEFAULT_PIXEL_FORMAT
    movflags: str | None = DEFAULT_MOVFLAGS
    write_zip: bool = False
    ffmpeg_executable: Path | None = None
    # C3-6 opt-in overlays (default off). Annotated clips replace naked media when
    # enabled; never duplicates both representations for the same bout.
    overlays_enabled: bool = False
    overlay_include_behavior: bool = True
    overlay_include_pose: bool = False
    overlay_pose_run_dir: Path | str | None = None
    overlay_include_contact: bool = True
    overlay_include_sbs: bool = True
    overlay_draw_skeleton: bool = True
    overlay_hud_mode: str = "both"

    def to_dict(self) -> dict[str, Any]:
        overlays = bool(self.overlays_enabled)
        pose_dir = (
            str(Path(str(self.overlay_pose_run_dir)).expanduser().resolve())
            if overlays
            and self.overlay_include_pose
            and self.overlay_pose_run_dir is not None
            and str(self.overlay_pose_run_dir).strip()
            else None
        )
        return {
            "schema_version": PI_REPORT_SETTINGS_SCHEMA_VERSION,
            "pre_context_sec": float(self.pre_context_sec),
            "post_context_sec": float(self.post_context_sec),
            "codec": self.codec,
            "crf": int(self.crf),
            "pixel_format": self.pixel_format,
            "movflags": self.movflags,
            "write_zip": bool(self.write_zip),
            "ffmpeg_executable": (
                str(self.ffmpeg_executable) if self.ffmpeg_executable else None
            ),
            "gif_requested": False,
            "overlays_requested": overlays,
            "overlays_enabled": overlays,
            "overlay_include_behavior": bool(self.overlay_include_behavior),
            "overlay_include_pose": bool(self.overlay_include_pose),
            "overlay_pose_run_dir": pose_dir,
            "overlay_include_contact": bool(self.overlay_include_contact),
            "overlay_include_sbs": bool(self.overlay_include_sbs),
            "overlay_draw_skeleton": bool(self.overlay_draw_skeleton),
            "overlay_hud_mode": str(self.overlay_hud_mode),
            "default_channel_view": "exclusive",
            "include_contact_sbs": False,
        }


@dataclass(frozen=True, slots=True)
class PiReportResult:
    """Successful finalized PI report directory."""

    report_id: str
    report_dir: Path
    zip_path: Path | None
    source_type: str
    package_run_id: str
    review_run_id: str | None
    n_bouts: int
    n_media_succeeded: int
    n_media_failed: int
    manifest_path: Path
    report_data_path: Path
    qc_path: Path
    index_html_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_dir": str(self.report_dir),
            "zip_path": str(self.zip_path) if self.zip_path else None,
            "source_type": self.source_type,
            "package_run_id": self.package_run_id,
            "review_run_id": self.review_run_id,
            "n_bouts": self.n_bouts,
            "n_media_succeeded": self.n_media_succeeded,
            "n_media_failed": self.n_media_failed,
            "manifest_path": str(self.manifest_path),
            "report_data_path": str(self.report_data_path),
            "qc_path": str(self.qc_path),
            "index_html_path": str(self.index_html_path),
        }


def publish_pi_report(
    *,
    session_root: Path | str,
    source_type: str,
    package_dir: Path | str | None = None,
    review_run_dir: Path | str | None = None,
    settings: PiReportSettings | None = None,
    query: GalleryQuery | None = None,
    report_id: str | None = None,
    timestamp: str | None = None,
    output_root: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_requested: CancellationCallback | None = None,
) -> PiReportResult:
    """Generate an immutable PI HTML report under ``results_exports/pi_report__*``.

    Stages into a temporary directory, validates required artifacts, then
    atomically moves into the final report directory. Never overwrites an
    existing report. Never mutates S8/S9/S10 science trees.
    Cooperative cancel is honored through encode/bundle (before rename).
    """

    cfg = settings or PiReportSettings()
    root = Path(session_root).expanduser().resolve()
    exports_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (root / RESULTS_EXPORTS_DIRNAME)
    )
    exports_root.mkdir(parents=True, exist_ok=True)

    ts = _sanitize_timestamp(timestamp or _now_timestamp())
    job_id = report_id or f"{PI_REPORT_RUN_PREFIX}__{ts}"
    if not str(job_id).startswith(f"{PI_REPORT_RUN_PREFIX}__"):
        job_id = f"{PI_REPORT_RUN_PREFIX}__{ts}"
    final_dir = exports_root / job_id
    _reject_output_overlap(final_dir, root)
    if final_dir.exists():
        raise PiReportPublicationError(
            f"PI report already exists (immutability): {final_dir}"
        )

    _raise_if_cancelled(cancellation_requested)
    source = resolve_pi_report_source(
        source_type=source_type,
        package_dir=package_dir,
        review_run_dir=review_run_dir,
        query=query,
        pre_context_sec=cfg.pre_context_sec,
        post_context_sec=cfg.post_context_sec,
        require_video=True,
    )
    if source.gallery.n_cards < 1:
        raise PiReportPublicationError(
            "PI report selection includes no eligible exclusive bouts."
        )
    # Snapshot S8/S9/prepared-video fingerprints before export (no science writes).
    before = snapshot_upstream_fingerprints(source.package)
    report_scope = _report_scope_for_query(query)
    selected_bout_ids = (
        sorted(query.bout_ids)
        if query is not None and query.bout_ids is not None
        else None
    )

    staging_parent = exports_root / STAGING_DIRNAME
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"{job_id}__", dir=str(staging_parent))
    )
    log_lines: list[str] = []
    created_at = _iso_now()
    published = False

    try:
        _log(log_lines, f"PI report staging started: {job_id}")
        _log(log_lines, f"source_type={source.source_type}")
        _log(log_lines, f"report_scope={report_scope}")
        _emit(progress_callback, "resolve_source", 1, 4)

        settings_snapshot = {
            **cfg.to_dict(),
            "report_id": job_id,
            "source_type": source.source_type,
            "package_run_id": source.package_run_id,
            "review_run_id": source.review_run_id,
            "source_digests": dict(source.source_digests),
            "n_exclusive_bouts": source.gallery.n_cards,
            "report_scope": report_scope,
            "selected_bout_ids": selected_bout_ids,
            "subset_of_source": report_scope == REPORT_SCOPE_SELECTED,
            "query": (query.to_dict() if query is not None else GalleryQuery().to_dict()),
        }
        write_yaml(staging_dir / SETTINGS_USED_FILENAME, settings_snapshot)
        write_json(
            staging_dir / RUN_META_FILENAME,
            _run_meta(
                report_id=job_id,
                created_at=created_at,
                completed_at=None,
                status=STATUS_RUNNING,
                session_root=root,
                source=source,
                settings_snapshot=settings_snapshot,
                counts=None,
                error_message=None,
            ),
        )

        def _media_progress(done: int, total: int, bout_id: str) -> None:
            _log(log_lines, f"encode {done}/{total}: {bout_id}")
            _emit(progress_callback, "encode_media", done, max(total, 1))

        if cfg.overlays_enabled:
            if source.source_type != PI_SOURCE_REVIEWED:
                raise PiReportPublicationError(
                    "PI report overlays require a published S10 reviewed source "
                    "(refusing automated S8 label burn-in)."
                )
            if source.review_run_dir is None:
                raise PiReportPublicationError(
                    "PI report overlays require an explicit published review_run_dir."
                )
            _log(log_lines, "overlay media enabled (single annotated representation)")
            media = materialize_report_overlay_media(
                cards=source.gallery.cards,
                review_run_dir=source.review_run_dir,
                report_root=staging_dir,
                include_behavior=bool(cfg.overlay_include_behavior),
                include_pose=bool(cfg.overlay_include_pose),
                pose_run_dir=cfg.overlay_pose_run_dir,
                hud_mode=str(cfg.overlay_hud_mode),
                include_contact=bool(cfg.overlay_include_contact),
                include_sbs=bool(cfg.overlay_include_sbs),
                draw_skeleton=bool(cfg.overlay_draw_skeleton),
                crf=cfg.crf,
                codec=cfg.codec,
                pixel_format=cfg.pixel_format,
                fail_fast=True,
                progress_callback=_media_progress,
                cancellation_requested=cancellation_requested,
            )
        else:
            media = materialize_report_media(
                cards=source.gallery.cards,
                source_video=source.video_path,
                report_root=staging_dir,
                ffmpeg_executable=cfg.ffmpeg_executable,
                crf=cfg.crf,
                codec=cfg.codec,
                pixel_format=cfg.pixel_format,
                movflags=cfg.movflags,
                fail_fast=True,
                progress_callback=_media_progress,
                cancellation_requested=cancellation_requested,
            )
        if media.n_failed:
            raise PiReportPublicationError(
                f"PI report media encode failures: {media.n_failed}"
            )
        if media.n_succeeded != source.gallery.n_cards:
            raise PiReportPublicationError(
                "PI report media count does not match exclusive bout count "
                f"({media.n_succeeded} vs {source.gallery.n_cards})."
            )

        _raise_if_cancelled(cancellation_requested)
        _emit(progress_callback, "write_bundle", 3, 4)
        suite_version = _behavior_suite_version()
        report_data = build_report_data_payload(
            source_type=source.source_type,
            gallery=source.gallery,
            media=media,
            provenance={
                **source.provenance,
                "source_digests": dict(source.source_digests),
                "prepared_video_identity": {
                    "basename": source.video_path.name,
                    "sha256": source.video_sha256,
                    "fps": source.video_probe.fps,
                    "n_frames": source.video_probe.n_frames,
                    "width": source.video_probe.width,
                    "height": source.video_probe.height,
                },
                "report_id": job_id,
                "generated_at_utc": created_at,
                "behavior_suite_version": suite_version,
                "report_scope": report_scope,
                "selected_bout_ids": selected_bout_ids,
                "subset_of_source": report_scope == REPORT_SCOPE_SELECTED,
            },
            settings=settings_snapshot,
            generated_at_utc=created_at,
            report_id=job_id,
            behavior_suite_version=suite_version,
            report_scope=report_scope,
            selected_bout_ids=selected_bout_ids,
        )
        write_report_bundle_files(report_root=staging_dir, report_data=report_data)

        # Integrity gate before any presentation as a finalized report.
        _raise_if_cancelled(cancellation_requested)
        verify_pinned_source_integrity(source)

        completed_at = _iso_now()
        qc = _build_qc(
            source=source,
            media=media,
            report_data=report_data,
            staging_dir=staging_dir,
            warnings=[],
            report_scope=report_scope,
        )
        write_json(staging_dir / QC_FILENAME, qc)
        terminal_status = (
            STATUS_COMPLETED
            if not qc.get("warnings")
            else STATUS_COMPLETED_WITH_WARNINGS
        )
        write_json(
            staging_dir / RUN_META_FILENAME,
            _run_meta(
                report_id=job_id,
                created_at=created_at,
                completed_at=completed_at,
                status=terminal_status,
                session_root=root,
                source=source,
                settings_snapshot=settings_snapshot,
                counts={
                    "n_bouts": source.gallery.n_cards,
                    "n_media_succeeded": media.n_succeeded,
                    "n_media_failed": media.n_failed,
                    "report_scope": report_scope,
                },
                error_message=None,
            ),
        )
        _write_log(staging_dir / PROCESSING_LOG_FILENAME, log_lines)

        # Collect digests after all non-manifest artifacts are final. The
        # manifest digests itself after write and is excluded from self-check.
        artifact_digests = _collect_artifact_digests(staging_dir, media)
        manifest = {
            "schema_version": PI_REPORT_MANIFEST_SCHEMA_VERSION,
            "report_id": job_id,
            "job_kind": PI_REPORT_JOB_KIND,
            "created_utc": created_at,
            "completed_utc": completed_at,
            "behavior_suite_version": suite_version,
            "source_type": source.source_type,
            "source_package_run_id": source.package_run_id,
            "source_review_run_id": source.review_run_id,
            "source_digests": dict(source.source_digests),
            "prepared_video_identity": {
                "basename": source.video_path.name,
                "sha256": source.video_sha256,
                "fps": source.video_probe.fps,
                "n_frames": source.video_probe.n_frames,
            },
            "selection": settings_snapshot.get("query"),
            "report_scope": report_scope,
            "selected_bout_ids": selected_bout_ids,
            "subset_of_source": report_scope == REPORT_SCOPE_SELECTED,
            "report_settings": {
                "pre_context_sec": cfg.pre_context_sec,
                "post_context_sec": cfg.post_context_sec,
                "codec": cfg.codec,
                "crf": cfg.crf,
                "pixel_format": cfg.pixel_format,
                "movflags": cfg.movflags,
                "gif_requested": False,
                "overlays_requested": bool(cfg.overlays_enabled),
                "overlays_enabled": bool(cfg.overlays_enabled),
                "overlay_include_behavior": bool(cfg.overlay_include_behavior),
                "overlay_include_pose": bool(cfg.overlay_include_pose),
                "overlay_pose_run_dir": (
                    str(Path(str(cfg.overlay_pose_run_dir)).expanduser().resolve())
                    if cfg.overlays_enabled
                    and cfg.overlay_include_pose
                    and cfg.overlay_pose_run_dir is not None
                    and str(cfg.overlay_pose_run_dir).strip()
                    else None
                ),
                "overlay_include_contact": bool(cfg.overlay_include_contact),
                "overlay_include_sbs": bool(cfg.overlay_include_sbs),
                "overlay_draw_skeleton": bool(cfg.overlay_draw_skeleton),
                "overlay_hud_mode": str(cfg.overlay_hud_mode),
            },
            "media_policy": (
                "annotated_video_previews"
                if cfg.overlays_enabled
                else "video_previews"
            ),
            "overlay_provenance": (
                dict(getattr(media, "overlay_provenance", {}) or {})
                if cfg.overlays_enabled
                else None
            ),
            "media_mapping": dict(media.media_mapping),
            "n_bouts": source.gallery.n_cards,
            "n_media_succeeded": media.n_succeeded,
            "n_media_failed": media.n_failed,
            "artifact_digests": artifact_digests,
            "status": terminal_status,
        }
        write_json(staging_dir / REPORT_MANIFEST_FILENAME, manifest)

        validate_report_directory(staging_dir, expect_n_bouts=source.gallery.n_cards)

        _raise_if_cancelled(cancellation_requested)
        # Atomic publish: rename staging → final (never overwrite).
        if final_dir.exists():
            raise PiReportPublicationError(
                f"PI report appeared during staging (refusing overwrite): {final_dir}"
            )
        staging_dir.replace(final_dir)
        staging_dir = final_dir  # published
        published = True
        _emit(progress_callback, "finalize", 4, 4)

        zip_path: Path | None = None
        if cfg.write_zip:
            # Cancel after rename is too late to discard the report; finish ZIP.
            zip_path = write_pi_report_zip(final_dir)

        assert_no_upstream_writes(
            package_dir=source.package_dir,
            s8_run_dir=source.package.source_s8.run_dir,
            video_path=source.video_path,
            snapshots=before,
        )

        return PiReportResult(
            report_id=job_id,
            report_dir=final_dir,
            zip_path=zip_path,
            source_type=source.source_type,
            package_run_id=source.package_run_id,
            review_run_id=source.review_run_id,
            n_bouts=source.gallery.n_cards,
            n_media_succeeded=media.n_succeeded,
            n_media_failed=media.n_failed,
            manifest_path=final_dir / REPORT_MANIFEST_FILENAME,
            report_data_path=final_dir / REPORT_DATA_FILENAME,
            qc_path=final_dir / QC_FILENAME,
            index_html_path=final_dir / INDEX_HTML_FILENAME,
        )
    except PiReportCancelledError as exc:
        _log(log_lines, f"CANCELED: {exc}")
        if not published:
            try:
                write_json(
                    staging_dir / RUN_META_FILENAME,
                    {
                        "schema_version": PI_REPORT_RUN_META_SCHEMA_VERSION,
                        "report_id": job_id,
                        "status": STATUS_CANCELED,
                        "error_message": str(exc),
                        "created_at_utc": created_at,
                        "completed_at_utc": _iso_now(),
                    },
                )
                _write_log(staging_dir / PROCESSING_LOG_FILENAME, log_lines)
            except Exception:  # noqa: BLE001
                pass
            _cleanup_tree(staging_dir)
            if final_dir.exists() and final_dir != staging_dir:
                _cleanup_tree(final_dir)
        raise
    except Exception as exc:
        _log(log_lines, f"FAILED: {exc}")
        # Best-effort failure breadcrumbs inside staging, then wipe incomplete report.
        if not published:
            try:
                write_json(
                    staging_dir / RUN_META_FILENAME,
                    {
                        "schema_version": PI_REPORT_RUN_META_SCHEMA_VERSION,
                        "report_id": job_id,
                        "status": STATUS_FAILED,
                        "error_message": str(exc),
                        "created_at_utc": created_at,
                        "completed_at_utc": _iso_now(),
                    },
                )
                _write_log(staging_dir / PROCESSING_LOG_FILENAME, log_lines)
            except Exception:  # noqa: BLE001
                pass
            _cleanup_tree(staging_dir)
            # Ensure no incomplete final report is left presented as success.
            if final_dir.exists() and final_dir != staging_dir:
                _cleanup_tree(final_dir)
        if isinstance(exc, PiReportPublicationError):
            raise
        raise PiReportPublicationError(f"PI report export failed: {exc}") from exc
    finally:
        # Remove empty staging parent if unused.
        with_suppress_rmdir(staging_parent)


def write_pi_report_zip(report_dir: Path | str, zip_path: Path | str | None = None) -> Path:
    """Zip a finalized report directory with relative member paths only."""

    root = Path(report_dir).expanduser().resolve()
    if not root.is_dir():
        raise PiReportPublicationError(f"Report directory missing for ZIP: {root}")
    out = (
        Path(zip_path).expanduser().resolve()
        if zip_path is not None
        else root.with_suffix(root.suffix + ".zip")
        if root.suffix
        else root.parent / f"{root.name}.zip"
    )
    if out.exists():
        raise PiReportPublicationError(f"ZIP already exists (immutability): {out}")
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(root).as_posix()
                zf.write(path, arcname=arcname)
    # Completeness: every required file present in archive.
    with zipfile.ZipFile(out, mode="r") as zf:
        names = set(zf.namelist())
    for required in REQUIRED_REPORT_FILES:
        if required not in names:
            out.unlink(missing_ok=True)
            raise PiReportValidationError(
                f"ZIP missing required member {required!r}; archive removed."
            )
    media_files = [
        n for n in names if n.startswith(f"{MEDIA_DIRNAME}/") and n.endswith(".mp4")
    ]
    if not media_files:
        out.unlink(missing_ok=True)
        raise PiReportValidationError("ZIP contains no media previews; archive removed.")
    return out


def validate_report_directory(
    report_dir: Path | str,
    *,
    expect_n_bouts: int | None = None,
) -> dict[str, Any]:
    """Validate a staged or finalized PI report directory."""

    root = Path(report_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_REPORT_FILES if not (root / name).is_file()]
    if missing:
        raise PiReportValidationError(f"PI report missing required files: {missing}")

    data = _load_json(root / REPORT_DATA_FILENAME)
    manifest = _load_json(root / REPORT_MANIFEST_FILENAME)
    qc = _load_json(root / QC_FILENAME)

    bouts = data.get("bouts")
    if not isinstance(bouts, list):
        raise PiReportValidationError("report_data.bouts must be a list.")
    if expect_n_bouts is not None and len(bouts) != int(expect_n_bouts):
        raise PiReportValidationError(
            f"report_data bout count {len(bouts)} != expected {expect_n_bouts}."
        )

    for bout in bouts:
        if not isinstance(bout, Mapping):
            raise PiReportValidationError("Each bout entry must be an object.")
        rel = bout.get("media_relpath")
        if not isinstance(rel, str):
            raise PiReportValidationError(
                f"Bout {bout.get('bout_id')!r} missing media_relpath."
            )
        rel_norm = rel.replace("\\", "/")
        if rel_norm.startswith("/") or "://" in rel_norm or (
            len(rel_norm) >= 2 and rel_norm[1] == ":"
        ):
            raise PiReportValidationError(
                f"Absolute media path forbidden in report: {rel!r}"
            )
        media_path = root / Path(rel_norm)
        if not media_path.is_file():
            raise PiReportValidationError(f"Missing media preview file: {rel_norm}")

    # Manifest digests for core files must match live files.
    # Skip the manifest itself (written after digest collection).
    digests = manifest.get("artifact_digests")
    if isinstance(digests, Mapping):
        for role, recorded in digests.items():
            if recorded is None:
                continue
            rel = str(role).replace("\\", "/")
            if rel in {REPORT_MANIFEST_FILENAME, f"./{REPORT_MANIFEST_FILENAME}"}:
                continue
            path = root / rel
            if not path.is_file():
                continue
            live = sha256_file(path)
            if str(recorded) != live:
                raise PiReportValidationError(
                    f"Artifact digest mismatch for {rel}: recorded={recorded!r} live={live!r}."
                )

    if int(manifest.get("n_bouts", -1)) != len(bouts):
        raise PiReportValidationError("Manifest n_bouts does not match report_data.")
    if int(manifest.get("n_media_succeeded", -1)) != len(bouts):
        raise PiReportValidationError(
            "Manifest n_media_succeeded does not match bout count."
        )

    return {
        "report_dir": root,
        "n_bouts": len(bouts),
        "source_type": data.get("source_type"),
        "qc_status": qc.get("status"),
        "valid": True,
    }


def _build_qc(
    *,
    source: PiReportSource,
    media: MediaMaterializationResult,
    report_data: Mapping[str, Any],
    staging_dir: Path,
    warnings: list[str],
    report_scope: str = REPORT_SCOPE_COMPLETE,
) -> dict[str, Any]:
    return {
        "schema_version": PI_REPORT_QC_SCHEMA_VERSION,
        "status": STATUS_COMPLETED if not warnings else STATUS_COMPLETED_WITH_WARNINGS,
        "source_type": source.source_type,
        "report_scope": report_scope,
        "subset_of_source": report_scope == REPORT_SCOPE_SELECTED,
        "n_exclusive_bouts": source.gallery.n_cards,
        "n_groups": source.gallery.n_groups,
        "status_counts": dict(source.gallery.status_counts),
        "n_media_succeeded": media.n_succeeded,
        "n_media_failed": media.n_failed,
        "media_files_present": sum(
            1 for p in (staging_dir / MEDIA_DIRNAME).glob("*.mp4") if p.is_file()
        ),
        "report_data_bout_count": len(report_data.get("bouts", [])),
        "contact_sbs_in_default_view": False,
        "draft_source": False,
        "warnings": list(warnings),
        "blocking": [],
    }


def _report_scope_for_query(query: GalleryQuery | None) -> str:
    if query is not None and query.bout_ids is not None:
        return REPORT_SCOPE_SELECTED
    return REPORT_SCOPE_COMPLETE


def _raise_if_cancelled(callback: CancellationCallback | None) -> None:
    if callback is not None and callback():
        raise PiReportCancelledError("PI report export cancelled before publication.")


def _collect_artifact_digests(
    staging_dir: Path,
    media: MediaMaterializationResult,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for rel in REQUIRED_REPORT_FILES:
        if rel == REPORT_MANIFEST_FILENAME:
            continue  # manifest is written after this collection
        path = staging_dir / rel
        if path.is_file():
            digests[rel.replace("\\", "/")] = sha256_file(path)
    for clip in media.clips:
        path = staging_dir / Path(clip.media_relpath)
        if path.is_file():
            digests[clip.media_relpath.replace("\\", "/")] = (
                clip.sha256 or sha256_file(path)
            )
    return digests


def _run_meta(
    *,
    report_id: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    session_root: Path,
    source: PiReportSource,
    settings_snapshot: Mapping[str, Any],
    counts: Mapping[str, Any] | None,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": PI_REPORT_RUN_META_SCHEMA_VERSION,
        "report_id": report_id,
        "job_kind": PI_REPORT_JOB_KIND,
        "status": status,
        "created_at_utc": created_at,
        "completed_at_utc": completed_at,
        "session_root_name": session_root.name,
        "source_type": source.source_type,
        "package_run_id": source.package_run_id,
        "review_run_id": source.review_run_id,
        "source_digests": dict(source.source_digests),
        "settings": dict(settings_snapshot),
        "counts": dict(counts) if counts else None,
        "error_message": error_message,
    }


def _reject_output_overlap(job_dir: Path, session_root: Path) -> None:
    resolved = job_dir.resolve()
    session = session_root.resolve()
    for name in _PROTECTED_SESSION_SUBDIRS:
        protected = (session / name).resolve()
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        raise PiReportPublicationError(
            f"Refusing PI report output inside protected science tree: {protected}"
        )


def _behavior_suite_version() -> str | None:
    try:
        return str(pkg_version("behavior-suite"))
    except PackageNotFoundError:
        try:
            return str(pkg_version("behavior_suite"))
        except PackageNotFoundError:
            return "0.1.1"


def _sanitize_timestamp(raw: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_-]+", "", str(raw).strip())
    if not text:
        raise PiReportPublicationError("Empty PI report timestamp.")
    return text


def _now_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _log(lines: list[str], message: str) -> None:
    lines.append(f"{_iso_now()} {message}")


def _write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _emit(
    callback: ProgressCallback | None,
    phase: str,
    done: int,
    total: int,
) -> None:
    if callback is not None:
        callback(phase, done, total)


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PiReportValidationError(f"JSON artifact is not an object: {path}")
    return payload


def _cleanup_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def with_suppress_rmdir(path: Path) -> None:
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


__all__ = [
    "PiReportResult",
    "PiReportSettings",
    "publish_pi_report",
    "validate_report_directory",
    "write_pi_report_zip",
]
