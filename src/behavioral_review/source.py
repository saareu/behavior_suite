"""Open and validate an eligible S9 package for behavioral inspection.

Opening is inspection-only: it never creates a review run, never accepts
content, and never mutates S7/S8/S9 artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    BOUTS_SCHEMA_VERSION as S8_BOUTS_SCHEMA_VERSION,
)
from behavioral_arbitration.contracts import (
    STATES_SCHEMA_VERSION as S8_STATES_SCHEMA_VERSION,
)
from behavioral_review.contracts import (
    EXPECTED_PACKAGE_BOUTS_SCHEMA,
    EXPECTED_S8_BOUTS_SCHEMA,
    EXPECTED_S8_STATES_SCHEMA,
    REQUIRED_PACKAGE_BOUT_COLUMNS,
    SOURCE_SNAPSHOT_SCHEMA_VERSION,
    SourceError,
)
from project.file_digest import sha256_file
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    ETHOGRAM_SUMMARY_FILENAME,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    SYNC_INDEX_FILENAME,
    VERIFICATION_STRENGTH_CONTENT_HASH,
)
from results_packaging.export import load_json_object
from results_packaging.publication import (
    ProvenanceError,
    PublicationError,
    validate_s9_run_for_downstream,
)
from results_packaging.sync_index import validate_sync_index


@dataclass(frozen=True, slots=True)
class ResolvedVideoSource:
    """Prepared-video identity resolved from the S9 package (no substitutes)."""

    path: Path | None
    kind: str
    sha256: str | None
    fingerprint: tuple[Any, ...] | None
    verification_strength: str | None
    available: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedS8Dependency:
    """Exact S8 states/bouts references pinned by the package."""

    run_id: str | None
    run_dir: Path | None
    states_path: Path | None
    bouts_path: Path | None
    states_sha256_recorded: str | None
    bouts_sha256_recorded: str | None
    states_schema_version: str | None
    bouts_schema_version: str | None
    states_available: bool
    bouts_available: bool
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewSourcePackage:
    """Validated S9 package snapshot for read-only behavioral inspection."""

    schema_version: str
    package_dir: Path
    package_run_id: str
    package_status: str
    currently_eligible: bool
    session_root: Path | None
    run_meta: dict[str, Any]
    package_manifest: dict[str, Any]
    qc: dict[str, Any]
    sync_index: pd.DataFrame
    behavior_bouts: pd.DataFrame
    ethogram_summary: pd.DataFrame | None
    sync_index_sha256: str
    behavior_bouts_sha256: str
    package_manifest_sha256: str
    source_s8: ResolvedS8Dependency
    source_s7_run_id: str | None
    source_s7_run_dir: Path | None
    source_s6_run_id: str | None
    source_video: ResolvedVideoSource
    n_dense_samples: int
    n_bouts: int
    warnings: tuple[str, ...] = ()
    validation: dict[str, Any] = field(default_factory=dict)

    @property
    def video_available(self) -> bool:
        return self.source_video.available

    @property
    def s8_states_available(self) -> bool:
        return self.source_s8.states_available


def open_review_source(
    package_dir: Path | str,
    *,
    require_video: bool = True,
    require_s8_states: bool = False,
) -> ReviewSourcePackage:
    """Open an explicit S9 package identity for inspection.

    Requires **current** downstream eligibility via
    ``validate_s9_run_for_downstream``. Historical ``run_meta.status`` is never
    treated as current eligibility. Does not silently substitute another
    package, S8 run, or prepared video. Does not create review artifacts.
    """

    run_dir = Path(package_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SourceError(f"S9 package directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise SourceError(f"Missing run_meta.json: {meta_path}")
    meta = load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise SourceError(
            f"S9 package is not completed (historical status={status or 'missing'!r})."
        )

    for name in (
        SYNC_INDEX_FILENAME,
        BEHAVIOR_BOUTS_FILENAME,
        PACKAGE_MANIFEST_FILENAME,
        QC_FILENAME,
    ):
        if not (run_dir / name).is_file():
            raise SourceError(f"S9 package missing required artifact: {name}")

    sync_path = run_dir / SYNC_INDEX_FILENAME
    bouts_path = run_dir / BEHAVIOR_BOUTS_FILENAME
    manifest_path = run_dir / PACKAGE_MANIFEST_FILENAME
    qc_path = run_dir / QC_FILENAME
    summary_path = run_dir / ETHOGRAM_SUMMARY_FILENAME

    sync_sha = sha256_file(sync_path)
    bouts_sha = sha256_file(bouts_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = load_json_object(manifest_path)
    qc = load_json_object(qc_path)

    _assert_embedded_digest(manifest, role="sync_index", path=sync_path, live=sync_sha)
    _assert_embedded_digest(
        manifest, role="behavior_bouts", path=bouts_path, live=bouts_sha
    )

    try:
        validation = validate_s9_run_for_downstream(run_dir)
    except (PublicationError, ProvenanceError) as exc:
        raise SourceError(
            f"S9 package is not currently eligible for behavioral review: {exc}"
        ) from exc

    warnings: list[str] = []
    sync_index = pd.read_parquet(sync_path)
    try:
        validate_sync_index(sync_index)
    except Exception as exc:  # SyncIndexError or structural failure
        raise SourceError(f"Invalid sync_index.parquet: {exc}") from exc

    behavior_bouts = pd.read_parquet(bouts_path)
    _validate_packaged_bouts_schema(behavior_bouts)

    ethogram_summary: pd.DataFrame | None = None
    if summary_path.is_file():
        ethogram_summary = pd.read_parquet(summary_path)
    else:
        warnings.append("ethogram_summary.parquet missing; summary strip unavailable.")

    source_s8 = _resolve_s8_dependency(
        meta,
        manifest,
        require_states=require_s8_states,
        warnings=warnings,
    )
    source_video = _resolve_prepared_video(
        meta,
        require_video=require_video,
        warnings=warnings,
    )

    source_s7: dict[str, Any] = (
        dict(meta["source_s7"]) if isinstance(meta.get("source_s7"), Mapping) else {}
    )
    source_s6: dict[str, Any] = (
        dict(meta["source_s6"]) if isinstance(meta.get("source_s6"), Mapping) else {}
    )
    source_s7_run_dir_text = _optional_text(source_s7.get("run_dir"))
    source_s7_run_dir = (
        Path(source_s7_run_dir_text).expanduser().resolve()
        if source_s7_run_dir_text
        else None
    )

    package_run_id = (
        _optional_text(meta.get("run_id"))
        or _optional_text(manifest.get("package_run_id"))
        or run_dir.name
    )
    session_root_text = _optional_text(meta.get("session_root"))
    n_dense = int(len(sync_index))
    n_bouts = int(len(behavior_bouts))

    # Preserve exact identities from validation / meta (no alternate selection).
    validated_s8_id = _optional_text(validation.get("source_s8_run_id"))
    if (
        validated_s8_id is not None
        and source_s8.run_id is not None
        and validated_s8_id != source_s8.run_id
    ):
        raise SourceError(
            "S9 eligibility source_s8_run_id does not match package meta "
            f"({validated_s8_id!r} vs {source_s8.run_id!r})."
        )

    return ReviewSourcePackage(
        schema_version=SOURCE_SNAPSHOT_SCHEMA_VERSION,
        package_dir=run_dir,
        package_run_id=package_run_id,
        package_status=status or STATUS_COMPLETED,
        currently_eligible=True,
        session_root=Path(session_root_text).resolve() if session_root_text else None,
        run_meta=meta,
        package_manifest=manifest,
        qc=qc,
        sync_index=sync_index,
        behavior_bouts=behavior_bouts,
        ethogram_summary=ethogram_summary,
        sync_index_sha256=sync_sha,
        behavior_bouts_sha256=bouts_sha,
        package_manifest_sha256=manifest_sha,
        source_s8=source_s8,
        source_s7_run_id=_optional_text(source_s7.get("run_id")),
        source_s7_run_dir=source_s7_run_dir,
        source_s6_run_id=_optional_text(source_s6.get("run_id")),
        source_video=source_video,
        n_dense_samples=n_dense,
        n_bouts=n_bouts,
        warnings=tuple(warnings),
        validation=dict(validation),
    )


def assert_no_upstream_writes(
    *,
    package_dir: Path,
    s8_run_dir: Path | None,
    video_path: Path | None,
    snapshots: Mapping[Path, tuple[Any, ...]],
) -> None:
    """Raise if any snapshotted upstream path fingerprint changed."""

    for path, expected in snapshots.items():
        if not path.exists():
            raise SourceError(f"Upstream path disappeared during inspection: {path}")
        live = artifact_fingerprint(path)
        if tuple(live) != tuple(expected):
            raise SourceError(
                f"Upstream artifact changed during inspection (forbidden write?): {path}"
            )
    # Explicit identity pins must still resolve to the same paths.
    if not package_dir.is_dir():
        raise SourceError(f"S9 package directory disappeared: {package_dir}")
    if s8_run_dir is not None and not s8_run_dir.is_dir():
        raise SourceError(f"S8 run directory disappeared: {s8_run_dir}")
    if video_path is not None and not video_path.is_file():
        raise SourceError(f"Prepared video disappeared: {video_path}")


def snapshot_upstream_fingerprints(source: ReviewSourcePackage) -> dict[Path, tuple[Any, ...]]:
    """Capture size/mtime fingerprints for no-write assertions."""

    paths: list[Path] = [
        source.package_dir / SYNC_INDEX_FILENAME,
        source.package_dir / BEHAVIOR_BOUTS_FILENAME,
        source.package_dir / PACKAGE_MANIFEST_FILENAME,
        source.package_dir / RUN_META_FILENAME,
        source.package_dir / QC_FILENAME,
    ]
    if source.source_s8.states_path is not None and source.source_s8.states_path.is_file():
        paths.append(source.source_s8.states_path)
    if source.source_s8.bouts_path is not None and source.source_s8.bouts_path.is_file():
        paths.append(source.source_s8.bouts_path)
    if source.source_video.path is not None and source.source_video.path.is_file():
        paths.append(source.source_video.path)
    return {p: tuple(artifact_fingerprint(p)) for p in paths}


def _validate_packaged_bouts_schema(bouts: pd.DataFrame) -> None:
    if bouts is None or bouts.empty:
        raise SourceError("Packaged behavior_bouts.parquet is empty.")
    missing = [c for c in REQUIRED_PACKAGE_BOUT_COLUMNS if c not in bouts.columns]
    if missing:
        raise SourceError(f"Packaged bouts missing required columns: {missing}")
    schema_vals = {str(v) for v in bouts["schema_version"].astype(str).unique()}
    if schema_vals - {EXPECTED_PACKAGE_BOUTS_SCHEMA}:
        raise SourceError(
            f"Unsupported packaged bout schema_version values: {sorted(schema_vals)}"
        )
    s8_schema_vals = {
        str(v) for v in bouts["s8_bouts_schema_version"].astype(str).unique()
    }
    if s8_schema_vals - {EXPECTED_S8_BOUTS_SCHEMA, S8_BOUTS_SCHEMA_VERSION}:
        raise SourceError(
            "Unsupported s8_bouts_schema_version values: "
            f"{sorted(s8_schema_vals)}; expected {EXPECTED_S8_BOUTS_SCHEMA!r}."
        )


def _resolve_s8_dependency(
    meta: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    require_states: bool,
    warnings: list[str],
) -> ResolvedS8Dependency:
    source_s8 = (
        dict(meta["source_s8"]) if isinstance(meta.get("source_s8"), Mapping) else {}
    )
    run_id = _optional_text(source_s8.get("run_id"))
    run_dir_text = _optional_text(source_s8.get("run_dir"))
    run_dir = Path(run_dir_text).expanduser().resolve() if run_dir_text else None

    states_text = _optional_text(source_s8.get("behavior_states"))
    bouts_text = _optional_text(source_s8.get("behavior_bouts"))
    states_path = Path(states_text).expanduser().resolve() if states_text else None
    bouts_path = Path(bouts_text).expanduser().resolve() if bouts_text else None
    states_sha = _optional_text(source_s8.get("behavior_states_sha256"))
    bouts_sha = _optional_text(source_s8.get("behavior_bouts_sha256"))

    states_schema = _read_parquet_schema_version(states_path)
    bouts_schema = _read_parquet_schema_version(bouts_path)

    warning: str | None = None
    states_available = bool(states_path is not None and states_path.is_file())
    bouts_available = bool(bouts_path is not None and bouts_path.is_file())

    if states_available and states_sha is not None and states_path is not None:
        live = sha256_file(states_path)
        if live.lower() != states_sha.lower():
            raise SourceError(
                "S8 behavior_states SHA256 no longer matches package-recorded digest."
            )
    if bouts_available and bouts_sha is not None and bouts_path is not None:
        live = sha256_file(bouts_path)
        if live.lower() != bouts_sha.lower():
            raise SourceError(
                "S8 behavior_bouts SHA256 no longer matches package-recorded digest."
            )

    if states_schema is not None and states_schema not in (
        EXPECTED_S8_STATES_SCHEMA,
        S8_STATES_SCHEMA_VERSION,
    ):
        raise SourceError(
            f"Unsupported S8 states schema_version {states_schema!r}; "
            f"expected {EXPECTED_S8_STATES_SCHEMA!r}."
        )
    if bouts_schema is not None and bouts_schema not in (
        EXPECTED_S8_BOUTS_SCHEMA,
        S8_BOUTS_SCHEMA_VERSION,
    ):
        raise SourceError(
            f"Unsupported S8 bouts schema_version {bouts_schema!r}; "
            f"expected {EXPECTED_S8_BOUTS_SCHEMA!r}."
        )

    if not states_available:
        msg = "S8 behavior_states.parquet unresolved; framewise ethogram blocked."
        if require_states:
            raise SourceError(msg)
        warnings.append(msg)
        warning = msg
    if not bouts_available:
        # Packaged bouts remain authoritative for navigation; referenced S8
        # bouts are provenance pins.
        warnings.append(
            "Referenced S8 behavior_bouts.parquet path unavailable; "
            "using packaged bouts only."
        )

    # Prefer package source_references identity if present (must agree).
    refs = manifest.get("source_references")
    if isinstance(refs, Mapping):
        ref_s8 = refs.get("source_s8")
        if isinstance(ref_s8, Mapping):
            ref_id = _optional_text(ref_s8.get("run_id"))
            if ref_id is not None and run_id is not None and ref_id != run_id:
                raise SourceError(
                    "Manifest source_references.source_s8.run_id disagrees with "
                    f"run_meta ({ref_id!r} vs {run_id!r})."
                )

    return ResolvedS8Dependency(
        run_id=run_id,
        run_dir=run_dir,
        states_path=states_path,
        bouts_path=bouts_path,
        states_sha256_recorded=states_sha,
        bouts_sha256_recorded=bouts_sha,
        states_schema_version=states_schema,
        bouts_schema_version=bouts_schema,
        states_available=states_available,
        bouts_available=bouts_available,
        warning=warning,
    )


def _resolve_prepared_video(
    meta: Mapping[str, Any],
    *,
    require_video: bool,
    warnings: list[str],
) -> ResolvedVideoSource:
    source_video = (
        dict(meta["source_video"])
        if isinstance(meta.get("source_video"), Mapping)
        else {}
    )
    path_text = _optional_text(source_video.get("path"))
    path = Path(path_text).expanduser().resolve() if path_text else None
    sha = _optional_text(source_video.get("sha256"))
    strength = _optional_text(source_video.get("verification_strength"))
    fingerprint_raw = source_video.get("fingerprint")
    fingerprint: tuple[Any, ...] | None = (
        tuple(fingerprint_raw) if isinstance(fingerprint_raw, list) else None
    )

    if path is None:
        msg = "Prepared video path absent from package source_video."
        if require_video:
            raise SourceError(msg)
        warnings.append(msg)
        return ResolvedVideoSource(
            path=None,
            kind="prepared_video",
            sha256=sha,
            fingerprint=fingerprint,
            verification_strength=strength,
            available=False,
            warning=msg,
        )

    if not path.is_file():
        msg = f"Prepared video missing (no substitute): {path}"
        if require_video:
            raise SourceError(msg)
        warnings.append(msg)
        return ResolvedVideoSource(
            path=path,
            kind="prepared_video",
            sha256=sha,
            fingerprint=fingerprint,
            verification_strength=strength,
            available=False,
            warning=msg,
        )

    try:
        _check_video_fingerprint(source_video, path)
    except SourceError as exc:
        if require_video:
            raise
        warnings.append(str(exc))
        return ResolvedVideoSource(
            path=path,
            kind="prepared_video",
            sha256=sha,
            fingerprint=fingerprint,
            verification_strength=strength,
            available=False,
            warning=str(exc),
        )

    if fingerprint is None:
        fingerprint = tuple(artifact_fingerprint(path))

    return ResolvedVideoSource(
        path=path,
        kind="prepared_video",
        sha256=sha,
        fingerprint=fingerprint,
        verification_strength=strength,
        available=True,
        warning=None,
    )


def _check_video_fingerprint(source_video: Mapping[str, Any], video_path: Path) -> None:
    strength = _optional_text(source_video.get("verification_strength"))
    recorded_sha = _optional_text(source_video.get("sha256"))
    if strength == VERIFICATION_STRENGTH_CONTENT_HASH and recorded_sha is not None:
        live = sha256_file(video_path)
        if live.lower() != recorded_sha.lower():
            raise SourceError(
                "Prepared video SHA256 no longer matches package-recorded digest "
                "(no silent substitute)."
            )
        return
    recorded_fp = source_video.get("fingerprint")
    if recorded_fp is not None:
        live_fp = list(artifact_fingerprint(video_path))
        if list(recorded_fp) != live_fp:
            raise SourceError(
                "Prepared video size/mtime fingerprint no longer matches the "
                "fingerprint recorded by the S9 package (no silent substitute)."
            )


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
            raise SourceError(
                f"Package {role} SHA256 does not match manifest-recorded digest "
                f"({path})."
            )
        return


def _read_parquet_schema_version(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path, columns=["schema_version"])
    except Exception as exc:  # noqa: BLE001 - surface as source failure
        raise SourceError(f"Failed reading schema_version from {path}: {exc}") from exc
    if frame.empty:
        return None
    return str(frame["schema_version"].iloc[0])


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
