"""``package_manifest.json`` construction for Subsystem 09 runs.

Distinguishes embedded package artifacts from externally referenced
dependencies. Labels fingerprint type and verification strength so
size/mtime fingerprints are never silently equated with SHA256 (U-R6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project.file_digest import sha256_file
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BOUTS_SCHEMA_VERSION,
    DEFAULT_PROFILE_ID,
    ETHOGRAM_SUMMARY_FILENAME,
    ETHOGRAM_SUMMARY_SCHEMA_VERSION,
    FINGERPRINT_TYPE_SHA256,
    FINGERPRINT_TYPE_SIZE_MTIME,
    MANIFEST_SCHEMA_VERSION,
    PACKAGE_MANIFEST_FILENAME,
    PACKAGE_MODE_REFERENCE_LIGHTWEIGHT,
    PACKAGE_VERSION,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    QC_SCHEMA_VERSION,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    SYNC_INDEX_FILENAME,
    SYNC_SCHEMA_VERSION,
    TARGET_FPS,
    VERIFICATION_STRENGTH_CONTENT_HASH,
    VERIFICATION_STRENGTH_PATH_ONLY,
    VERIFICATION_STRENGTH_SIZE_MTIME,
    PublicationError,
)
from results_packaging.dependencies import DependencyManifest
from results_packaging.export_profiles import ExportProfileResult
from results_packaging.qc import QcSummary, packaged_bout_content_checksum


@dataclass(frozen=True, slots=True)
class ArtifactFingerprintRecord:
    """One artifact identity with explicit verification semantics."""

    role: str
    relative_path: str | None
    absolute_path: str | None
    embedded: bool
    sha256: str | None
    fingerprint_type: str
    verification_strength: str
    size_bytes: int | None
    mtime_ns: int | None
    schema_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "embedded": self.embedded,
            "sha256": self.sha256,
            "fingerprint_type": self.fingerprint_type,
            "verification_strength": self.verification_strength,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "schema_version": self.schema_version,
            "notes": _verification_notes(
                self.fingerprint_type, self.verification_strength
            ),
        }


def build_package_manifest(
    *,
    package_run_id: str,
    created_at: str,
    session_root: Path,
    run_dir: Path,
    dependencies: DependencyManifest,
    qc: QcSummary,
    settings_used: Mapping[str, Any],
    export_profiles: Mapping[str, ExportProfileResult] | None,
    packaged_bouts_checksum: str | None,
    n_sync_rows: int,
    n_bouts: int,
    n_ethogram_rows: int,
    source_s8_states_path: Path,
    source_s8_bouts_path: Path,
    source_s8_states_sha256: str,
    source_s8_bouts_sha256: str,
    source_s8_run_id: str,
    source_s8_run_dir: Path,
    source_s8_status: str,
    source_s7_run_id: str,
    source_s7_run_dir: Path,
    source_s6_run_id: str,
    source_s6_run_dir: Path,
    sync_construction_method: str,
    package_mode: str = PACKAGE_MODE_REFERENCE_LIGHTWEIGHT,
    selected_export_profiles: Sequence[str] = (),
    profile_artifact_paths: Mapping[str, Mapping[str, Path]] | None = None,
) -> dict[str, Any]:
    """Build the frozen ``package_manifest.json`` payload."""

    session_root = Path(session_root).resolve()
    run_dir = Path(run_dir).resolve()

    # Immutable science artifacts — content-hashed for reopen / eligibility.
    science_roles = {
        "sync_index": (run_dir / SYNC_INDEX_FILENAME, SYNC_SCHEMA_VERSION),
        "behavior_bouts": (run_dir / BEHAVIOR_BOUTS_FILENAME, BOUTS_SCHEMA_VERSION),
        "ethogram_summary": (
            run_dir / ETHOGRAM_SUMMARY_FILENAME,
            ETHOGRAM_SUMMARY_SCHEMA_VERSION,
        ),
        "qc": (run_dir / QC_FILENAME, QC_SCHEMA_VERSION),
    }
    # Lifecycle records are rewritten at finalize; path-listed without content
    # hash so eligibility does not treat status/settings updates as staleness.
    lifecycle_roles: dict[str, tuple[Path, str | None]] = {
        "package_manifest": (run_dir / PACKAGE_MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION),
        "settings_used": (run_dir / SETTINGS_USED_FILENAME, SETTINGS_SCHEMA_VERSION),
        "run_meta": (run_dir / RUN_META_FILENAME, RUN_META_SCHEMA_VERSION),
        "processing_log": (run_dir / PROCESSING_LOG_FILENAME, None),
    }

    package_artifacts: list[dict[str, Any]] = []
    for role, (path, schema_v) in science_roles.items():
        if not path.is_file():
            raise PublicationError(f"Required package artifact missing for manifest: {path}")
        package_artifacts.append(
            fingerprint_file(
                role=role,
                path=path,
                session_root=session_root,
                embedded=True,
                schema_version=schema_v,
                prefer_sha256=True,
            ).to_dict()
        )
    for role, (path, lifecycle_schema) in lifecycle_roles.items():
        package_artifacts.append(
            {
                "role": role,
                "relative_path": _relpath(path, session_root),
                "absolute_path": str(path),
                "embedded": True,
                "sha256": None,
                "fingerprint_type": FINGERPRINT_TYPE_SIZE_MTIME,
                "verification_strength": "lifecycle_path",
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
                "mtime_ns": (
                    int(
                        getattr(
                            path.stat(),
                            "st_mtime_ns",
                            int(path.stat().st_mtime * 1_000_000_000),
                        )
                    )
                    if path.is_file()
                    else None
                ),
                "schema_version": lifecycle_schema,
                "notes": (
                    "Lifecycle / self-describing record. Not content-hashed in the "
                    "manifest because finalize rewrites status/settings/log and a "
                    "self-digest of package_manifest.json is not well-defined."
                ),
            }
        )

    # Optional profile artifacts.
    profile_block: dict[str, Any] = {}
    for profile_id, paths in (profile_artifact_paths or {}).items():
        entries = []
        for role, path in paths.items():
            if path.is_file():
                entries.append(
                    fingerprint_file(
                        role=role,
                        path=path,
                        session_root=session_root,
                        embedded=True,
                        prefer_sha256=True,
                    ).to_dict()
                )
        result = (export_profiles or {}).get(profile_id)
        profile_block[profile_id] = {
            "artifacts": entries,
            "n_included": int(result.n_included) if result is not None else None,
            "n_excluded": int(result.n_excluded) if result is not None else None,
            "exclusion_reason_counts": (
                dict(result.exclusion_reason_counts) if result is not None else {}
            ),
        }

    external_refs: list[dict[str, Any]] = []
    for artifact in dependencies.artifacts:
        if artifact.path is None:
            continue
        if artifact.role.endswith("_run_dir"):
            external_refs.append(
                {
                    "role": artifact.role,
                    "relative_path": _relpath(artifact.path, session_root),
                    "absolute_path": str(artifact.path),
                    "embedded": False,
                    "sha256": None,
                    "fingerprint_type": FINGERPRINT_TYPE_SIZE_MTIME,
                    "verification_strength": VERIFICATION_STRENGTH_PATH_ONLY,
                    "size_bytes": None,
                    "mtime_ns": None,
                    "schema_version": artifact.schema_version,
                    "notes": (
                        "Directory identity only; nested file digests recorded separately."
                    ),
                }
            )
            continue
        prefer_sha = artifact.sha256 is not None
        record = fingerprint_file(
            role=artifact.role,
            path=artifact.path,
            session_root=session_root,
            embedded=False,
            schema_version=artifact.schema_version,
            prefer_sha256=prefer_sha,
            known_sha256=artifact.sha256,
        )
        external_refs.append(record.to_dict())

    # Explicit S8 states/bouts references (authoritative science remains upstream).
    external_refs.extend(
        [
            fingerprint_file(
                role="source_s8_behavior_states",
                path=source_s8_states_path,
                session_root=session_root,
                embedded=False,
                prefer_sha256=True,
                known_sha256=source_s8_states_sha256,
            ).to_dict(),
            fingerprint_file(
                role="source_s8_behavior_bouts",
                path=source_s8_bouts_path,
                session_root=session_root,
                embedded=False,
                prefer_sha256=True,
                known_sha256=source_s8_bouts_sha256,
            ).to_dict(),
        ]
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "package_run_id": package_run_id,
        "created_at": created_at,
        "session_root": str(session_root),
        "run_dir": str(run_dir),
        "run_dir_relative": _relpath(run_dir, session_root),
        "package_mode": package_mode,
        "default_profile_id": DEFAULT_PROFILE_ID,
        "selected_export_profiles": list(selected_export_profiles),
        "behavior_states_mode": "reference",
        "portability": {
            "self_contained": False,
            "package_mode": package_mode,
            "note": (
                "Default reference_lightweight package embeds sync/bouts/summary "
                "artifacts and references S8 states, prepared video, and upstream "
                "runs by path/digest. Portable embed is deferred (U-R3)."
            ),
        },
        "timeline": {
            "fps": TARGET_FPS,
            "sample_idx_equals_dense_frame_idx": True,
            "sync_construction_method": sync_construction_method,
            "n_sync_rows": int(n_sync_rows),
        },
        "row_counts": {
            "sync_index": int(n_sync_rows),
            "behavior_bouts": int(n_bouts),
            "ethogram_summary": int(n_ethogram_rows),
            "source_s8_bouts": int(n_bouts),
        },
        "full_bout_preservation": True,
        "authoritative_bout_content_sha256": packaged_bouts_checksum,
        "package_artifacts": package_artifacts,
        "export_profiles": profile_block,
        "source_references": {
            "source_s8": {
                "run_id": source_s8_run_id,
                "run_dir": str(source_s8_run_dir),
                "run_dir_relative": _relpath(source_s8_run_dir, session_root),
                "status_at_publish": source_s8_status,
                "behavior_states": str(source_s8_states_path),
                "behavior_states_sha256": source_s8_states_sha256,
                "behavior_bouts": str(source_s8_bouts_path),
                "behavior_bouts_sha256": source_s8_bouts_sha256,
            },
            "source_s7": {
                "run_id": source_s7_run_id,
                "run_dir": str(source_s7_run_dir),
                "run_dir_relative": _relpath(source_s7_run_dir, session_root),
            },
            "source_s6": {
                "run_id": source_s6_run_id,
                "run_dir": str(source_s6_run_dir),
                "run_dir_relative": _relpath(source_s6_run_dir, session_root),
            },
        },
        "external_references": external_refs,
        "dependency_snapshot": dependencies.to_dict(),
        "qc_summary": {
            "structural_ok": qc.structural_ok,
            "warning_count": len(qc.warnings),
            "failure_count": len(qc.failures),
            "informational_count": len(qc.informational),
        },
        "settings": dict(settings_used),
        "fingerprint_policy": {
            "content_hash": FINGERPRINT_TYPE_SHA256,
            "size_mtime": FINGERPRINT_TYPE_SIZE_MTIME,
            "rule": (
                "Reuse trusted existing content hashes when available. Use clearly "
                "labeled size/mtime fingerprints when content hashes are unavailable. "
                "Optional full video SHA256 may be requested explicitly and must not "
                "become an implicit expensive GUI-thread operation (U-R6)."
            ),
            "size_mtime_is_not_sha256": True,
        },
    }


def fingerprint_file(
    *,
    role: str,
    path: Path,
    session_root: Path,
    embedded: bool,
    prefer_sha256: bool,
    known_sha256: str | None = None,
    schema_version: str | None = None,
) -> ArtifactFingerprintRecord:
    """Build a labeled fingerprint for one file artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PublicationError(f"Cannot fingerprint missing file for {role}: {resolved}")
    stat = resolved.stat()
    size = int(stat.st_size)
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    # Touch fingerprint helper so size/mtime path stays consistent with memo keys.
    size_mtime_fp = artifact_fingerprint(resolved)
    if prefer_sha256:
        digest = known_sha256 or sha256_file(resolved)
        return ArtifactFingerprintRecord(
            role=role,
            relative_path=_relpath(resolved, session_root),
            absolute_path=str(resolved),
            embedded=embedded,
            sha256=digest,
            fingerprint_type=FINGERPRINT_TYPE_SHA256,
            verification_strength=VERIFICATION_STRENGTH_CONTENT_HASH,
            size_bytes=size,
            mtime_ns=mtime_ns,
            schema_version=schema_version,
        )

    # Size/mtime fingerprint only — explicitly weaker than content hash.
    if len(size_mtime_fp) >= 3 and size_mtime_fp[0] != "missing":
        size = int(size_mtime_fp[1])
        mtime_ns = int(size_mtime_fp[2])
    return ArtifactFingerprintRecord(
        role=role,
        relative_path=_relpath(resolved, session_root),
        absolute_path=str(resolved),
        embedded=embedded,
        sha256=known_sha256,
        fingerprint_type=FINGERPRINT_TYPE_SIZE_MTIME,
        verification_strength=(
            VERIFICATION_STRENGTH_CONTENT_HASH
            if known_sha256 is not None
            else VERIFICATION_STRENGTH_SIZE_MTIME
        ),
        size_bytes=size,
        mtime_ns=mtime_ns,
        schema_version=schema_version,
    )


def _relpath(path: Path, root: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace(
            "\\", "/"
        )
    except ValueError:
        return None


def _verification_notes(fingerprint_type: str, strength: str) -> str:
    if fingerprint_type == FINGERPRINT_TYPE_SHA256:
        return "Content hash (SHA256). Strong identity for staleness checks."
    if strength == VERIFICATION_STRENGTH_SIZE_MTIME:
        return (
            "Size/mtime fingerprint only. Detects replacement heuristics but is "
            "NOT equivalent to SHA256 content verification."
        )
    if strength == VERIFICATION_STRENGTH_PATH_ONLY:
        return "Path identity only; not a content hash."
    return "Mixed/legacy identity fields; inspect sha256 and fingerprint_type separately."


def compute_packaged_bout_checksum(packaged_bouts: Any) -> str:
    return packaged_bout_content_checksum(packaged_bouts)
