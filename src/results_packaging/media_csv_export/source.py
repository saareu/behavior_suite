"""Validate eligible S9 packages as export sources (no package mutation)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from project.file_digest import sha256_file
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    SYNC_INDEX_FILENAME,
    VERIFICATION_STRENGTH_CONTENT_HASH,
    ProvenanceError,
    PublicationError,
)
from results_packaging.export import load_json_object
from results_packaging.media_csv_export.contracts import (
    SOURCE_VIDEO_KIND_PREPARED,
    ExportSourceError,
)
from results_packaging.media_csv_export.video import VideoProbe, probe_video
from results_packaging.publication import validate_s9_run_for_downstream


@dataclass(frozen=True, slots=True)
class ExportSourcePackage:
    """Validated S9 package snapshot for one export job."""

    package_dir: Path
    package_run_id: str
    package_status: str
    session_root: Path | None
    run_meta: dict[str, Any]
    package_manifest: dict[str, Any]
    behavior_bouts: pd.DataFrame
    behavior_bouts_sha256: str
    sync_index_sha256: str
    package_manifest_sha256: str
    source_s8_run_id: str | None
    source_s8_run_dir: Path | None
    source_s7_run_id: str | None
    source_s6_run_id: str | None
    source_video_path: Path | None
    source_video_kind: str
    source_video_fingerprint: list[Any] | None
    source_video_sha256: str | None
    source_video_verification_strength: str | None
    currently_eligible: bool
    stale_warning: str | None
    video_probe: VideoProbe | None


def validate_package_for_export(
    package_dir: Path | str,
    *,
    require_video: bool,
    allow_stale_source: bool = False,
) -> ExportSourcePackage:
    """Validate a completed S9 package for media/CSV export.

    MVP policy: normal create/retry exports require a **currently eligible**
    S9 package (``allow_stale_source=False`` by default). Do not silently
    export from stale packages or substitute another package/video. Historical
    job inspection may pass ``allow_stale_source=True`` to reopen completed
    exports without mutating them. Explicit archival export from stale
    packages is deferred. Missing prepared video still blocks media exports.
    """

    run_dir = Path(package_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ExportSourceError(f"S9 package directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise ExportSourceError(f"Missing run_meta.json: {meta_path}")
    meta = load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise ExportSourceError(
            f"S9 package is not completed (status={status or 'missing'!r})."
        )

    for name in (
        BEHAVIOR_BOUTS_FILENAME,
        SYNC_INDEX_FILENAME,
        PACKAGE_MANIFEST_FILENAME,
        QC_FILENAME,
    ):
        if not (run_dir / name).is_file():
            raise ExportSourceError(f"S9 package missing required artifact: {name}")

    bouts_path = run_dir / BEHAVIOR_BOUTS_FILENAME
    sync_path = run_dir / SYNC_INDEX_FILENAME
    manifest_path = run_dir / PACKAGE_MANIFEST_FILENAME
    bouts_sha = sha256_file(bouts_path)
    sync_sha = sha256_file(sync_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = load_json_object(manifest_path)

    # Prefer embedded package digests when present.
    _assert_embedded_digest(manifest, role="behavior_bouts", path=bouts_path, live=bouts_sha)
    _assert_embedded_digest(manifest, role="sync_index", path=sync_path, live=sync_sha)

    currently_eligible = True
    stale_warning: str | None = None
    try:
        validate_s9_run_for_downstream(run_dir)
    except (PublicationError, ProvenanceError) as exc:
        currently_eligible = False
        if not allow_stale_source:
            raise ExportSourceError(
                f"S9 package is not currently eligible for export: {exc}"
            ) from exc
        stale_warning = (
            "Source package is stale or upstream references drifted; export "
            f"allowed from immutable package bytes with warning: {exc}"
        )

    source_s8: dict[str, Any] = (
        dict(meta["source_s8"]) if isinstance(meta.get("source_s8"), Mapping) else {}
    )
    source_s7: dict[str, Any] = (
        dict(meta["source_s7"]) if isinstance(meta.get("source_s7"), Mapping) else {}
    )
    source_s6: dict[str, Any] = (
        dict(meta["source_s6"]) if isinstance(meta.get("source_s6"), Mapping) else {}
    )
    source_video: dict[str, Any] = (
        dict(meta["source_video"])
        if isinstance(meta.get("source_video"), Mapping)
        else {}
    )

    video_path_text = _optional_text(source_video.get("path"))
    video_path = Path(video_path_text).expanduser().resolve() if video_path_text else None
    video_probe: VideoProbe | None = None
    if require_video:
        if video_path is None:
            raise ExportSourceError(
                "Media export requires package source_video.path (prepared video)."
            )
        if not video_path.is_file():
            raise ExportSourceError(f"Prepared video missing for media export: {video_path}")
        _check_video_fingerprint(source_video, video_path)
        video_probe = probe_video(video_path)
    elif video_path is not None and video_path.is_file():
        try:
            _check_video_fingerprint(source_video, video_path)
        except ExportSourceError:
            if not allow_stale_source:
                raise

    session_root_text = _optional_text(meta.get("session_root"))
    package_run_id = _optional_text(meta.get("run_id")) or run_dir.name
    bouts = pd.read_parquet(bouts_path)
    fingerprint_raw = source_video.get("fingerprint")
    source_video_fingerprint: list[Any] | None
    if isinstance(fingerprint_raw, list):
        source_video_fingerprint = list(fingerprint_raw)
    elif video_path is not None:
        source_video_fingerprint = list(artifact_fingerprint(video_path))
    else:
        source_video_fingerprint = None

    return ExportSourcePackage(
        package_dir=run_dir,
        package_run_id=package_run_id,
        package_status=status or STATUS_COMPLETED,
        session_root=Path(session_root_text).resolve() if session_root_text else None,
        run_meta=meta,
        package_manifest=manifest,
        behavior_bouts=bouts,
        behavior_bouts_sha256=bouts_sha,
        sync_index_sha256=sync_sha,
        package_manifest_sha256=manifest_sha,
        source_s8_run_id=_optional_text(source_s8.get("run_id")),
        source_s8_run_dir=(
            Path(str(source_s8["run_dir"])).resolve()
            if _optional_text(source_s8.get("run_dir"))
            else None
        ),
        source_s7_run_id=_optional_text(source_s7.get("run_id")),
        source_s6_run_id=_optional_text(source_s6.get("run_id")),
        source_video_path=video_path,
        source_video_kind=SOURCE_VIDEO_KIND_PREPARED,
        source_video_fingerprint=source_video_fingerprint,
        source_video_sha256=_optional_text(source_video.get("sha256")),
        source_video_verification_strength=_optional_text(
            source_video.get("verification_strength")
        ),
        currently_eligible=currently_eligible,
        stale_warning=stale_warning,
        video_probe=video_probe,
    )


def assert_source_unchanged(
    source: ExportSourcePackage,
    *,
    require_video: bool,
) -> None:
    """Re-check digests before finalize / retry."""

    live = validate_package_for_export(
        source.package_dir,
        require_video=require_video,
        allow_stale_source=True,
    )
    if live.behavior_bouts_sha256.lower() != source.behavior_bouts_sha256.lower():
        raise ExportSourceError(
            "Package behavior_bouts.parquet digest changed since export started."
        )
    if live.sync_index_sha256.lower() != source.sync_index_sha256.lower():
        raise ExportSourceError(
            "Package sync_index.parquet digest changed since export started."
        )
    if live.package_manifest_sha256.lower() != source.package_manifest_sha256.lower():
        raise ExportSourceError(
            "Package package_manifest.json digest changed since export started."
        )
    if require_video:
        if live.source_video_path != source.source_video_path:
            raise ExportSourceError("Prepared video path changed since export started.")
        if source.source_video_path is None or not source.source_video_path.is_file():
            raise ExportSourceError("Prepared video missing at retry/finalize.")


def _assert_embedded_digest(
    manifest: Mapping[str, Any],
    *,
    role: str,
    path: Path,
    live: str,
) -> None:
    for entry in manifest.get("package_artifacts", []) or []:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("role") != role:
            continue
        recorded = _optional_text(entry.get("sha256"))
        if recorded is None:
            continue
        if recorded.lower() != live.lower():
            raise ExportSourceError(
                f"Package {role} SHA256 does not match manifest-recorded digest "
                f"({path})."
            )
        return


def _check_video_fingerprint(source_video: Mapping[str, Any], video_path: Path) -> None:
    strength = _optional_text(source_video.get("verification_strength"))
    recorded_sha = _optional_text(source_video.get("sha256"))
    if strength == VERIFICATION_STRENGTH_CONTENT_HASH and recorded_sha is not None:
        live = sha256_file(video_path)
        if live.lower() != recorded_sha.lower():
            raise ExportSourceError(
                "Prepared video SHA256 no longer matches package-recorded digest."
            )
        return
    recorded_fp = source_video.get("fingerprint")
    if recorded_fp is not None:
        live_fp = list(artifact_fingerprint(video_path))
        if list(recorded_fp) != live_fp:
            raise ExportSourceError(
                "Prepared video size/mtime fingerprint no longer matches the "
                "fingerprint recorded by the S9 package."
            )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
