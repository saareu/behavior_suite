"""In-memory dependency / provenance identities for S9 packaging.

Records exact upstream identities for later publication. Does not publish
packages in Sprint 1. Uses ``project.file_digest`` / fingerprints; avoids
hashing large prepared videos unless explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project.file_digest import sha256_file
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import (
    DEPENDENCY_MANIFEST_SCHEMA_VERSION,
    PACKAGE_MODE_REFERENCE_LIGHTWEIGHT,
    SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC,
    DependencyError,
)
from results_packaging.eligibility import S9SourceValidationResult


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Path + optional digest for one dependency artifact."""

    role: str
    path: Path | None
    sha256: str | None = None
    schema_version: str | None = None
    required: bool = True
    fingerprint: tuple[Any, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "required": self.required,
            "fingerprint": list(self.fingerprint) if self.fingerprint is not None else None,
        }


@dataclass(frozen=True, slots=True)
class DependencyManifest:
    """Sprint 1 in-memory dependency snapshot (not yet written to disk)."""

    schema_version: str
    session_root: Path
    package_mode: str
    sync_construction_method: str
    s8_run_id: str
    s8_run_dir: Path
    s8_status: str
    s8_classification: str
    s7_run_id: str
    s7_run_dir: Path
    s6_run_id: str
    s6_run_dir: Path
    artifacts: tuple[ArtifactRef, ...]
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_root": str(self.session_root),
            "package_mode": self.package_mode,
            "sync_construction_method": self.sync_construction_method,
            "source_s8": {
                "run_id": self.s8_run_id,
                "run_dir": str(self.s8_run_dir),
                "status": self.s8_status,
                "classification": self.s8_classification,
            },
            "source_s7": {
                "run_id": self.s7_run_id,
                "run_dir": str(self.s7_run_dir),
            },
            "source_s6": {
                "run_id": self.s6_run_id,
                "run_dir": str(self.s6_run_dir),
            },
            "artifacts": [a.to_dict() for a in self.artifacts],
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }

    def artifact_map(self) -> dict[str, ArtifactRef]:
        return {a.role: a for a in self.artifacts}

    def assert_required_resolvable(self) -> None:
        """Fail if a required dependency path is missing or digest drifted."""

        for artifact in self.artifacts:
            if not artifact.required:
                continue
            if artifact.path is None:
                raise DependencyError(
                    f"Required dependency {artifact.role!r} has no path."
                )
            path = artifact.path
            # Run directories are path identities; artifacts are files.
            if artifact.role.endswith("_run_dir"):
                if not path.is_dir():
                    raise DependencyError(
                        f"Required dependency {artifact.role!r} is unavailable: "
                        f"{path}"
                    )
                continue
            if not path.is_file():
                raise DependencyError(
                    f"Required dependency {artifact.role!r} is unavailable: "
                    f"{path}"
                )
            if artifact.sha256 is not None:
                live = sha256_file(path)
                if live.lower() != artifact.sha256.lower():
                    raise DependencyError(
                        f"Required dependency {artifact.role!r} digest mismatch "
                        f"(recorded={artifact.sha256}, live={live})."
                    )


def build_dependency_manifest(
    source: S9SourceValidationResult,
    *,
    package_mode: str = PACKAGE_MODE_REFERENCE_LIGHTWEIGHT,
    sync_construction_method: str = SYNC_METHOD_REBUILD_FROM_PREPARED_SYNC,
    hash_prepared_video: bool = False,
) -> DependencyManifest:
    """Assemble dependency identities from a validated S9 source."""

    sync = source.sync_video
    notes = list(sync.lineage_notes)

    artifacts: list[ArtifactRef] = [
        ArtifactRef(
            role="s8_behavior_states",
            path=source.states_path,
            sha256=source.states_sha256,
            schema_version=source.states_schema_version,
            required=True,
            fingerprint=artifact_fingerprint(source.states_path),
        ),
        ArtifactRef(
            role="s8_behavior_bouts",
            path=source.bouts_path,
            sha256=source.bouts_sha256,
            schema_version=source.bouts_schema_version,
            required=True,
            fingerprint=artifact_fingerprint(source.bouts_path),
        ),
        ArtifactRef(
            role="prepared_sync",
            path=sync.prepared_sync_path,
            sha256=sync.prepared_sync_sha256,
            schema_version="prepared_sync_v1",
            required=True,
            fingerprint=artifact_fingerprint(sync.prepared_sync_path),
        ),
    ]

    video_sha: str | None = sync.prepared_video_sha256_recorded
    if sync.prepared_video_path is not None:
        if hash_prepared_video:
            video_sha = sha256_file(sync.prepared_video_path)
            if (
                sync.prepared_video_sha256_recorded is not None
                and video_sha.lower() != sync.prepared_video_sha256_recorded.lower()
            ):
                raise DependencyError(
                    "Prepared video digest does not match recorded lineage digest."
                )
        elif video_sha is None:
            notes.append(
                "Prepared video digest not hashed in Sprint 1 "
                "(hash_prepared_video=false); path + fingerprint recorded."
            )
        artifacts.append(
            ArtifactRef(
                role="prepared_video",
                path=sync.prepared_video_path,
                sha256=video_sha,
                schema_version=None,
                required=True,
                fingerprint=artifact_fingerprint(sync.prepared_video_path),
            )
        )
    else:
        artifacts.append(
            ArtifactRef(
                role="prepared_video",
                path=None,
                sha256=None,
                required=False,
                fingerprint=None,
            )
        )
        notes.append("Prepared video not attached to dependency manifest.")

    # Optional nested references (paths only; digests already validated upstream).
    for role, path_text in (
        ("s7_run_dir", str(source.source_s7_run_dir)),
        ("s6_run_dir", str(source.source_s6_run_dir)),
    ):
        path = Path(path_text)
        artifacts.append(
            ArtifactRef(
                role=role,
                path=path if path.exists() else None,
                sha256=None,
                required=True,
                fingerprint=artifact_fingerprint(path),
            )
        )

    manifest = DependencyManifest(
        schema_version=DEPENDENCY_MANIFEST_SCHEMA_VERSION,
        session_root=source.session_root,
        package_mode=package_mode,
        sync_construction_method=sync_construction_method,
        s8_run_id=source.s8_run_id,
        s8_run_dir=source.s8_run_dir,
        s8_status=source.s8_status,
        s8_classification=source.s8_classification,
        s7_run_id=source.source_s7_run_id,
        s7_run_dir=source.source_s7_run_dir,
        s6_run_id=source.source_s6_run_id,
        s6_run_dir=source.source_s6_run_dir,
        artifacts=tuple(artifacts),
        notes=tuple(notes),
        extra={
            "source_s4_run_id": sync.source_s4_run_id,
            "source_s5_run_id": sync.source_s5_run_id,
            "prepared_sync_sha256_recorded": sync.prepared_sync_sha256_recorded,
            "n_dense_samples": source.n_dense_samples,
            "n_bouts": source.n_bouts,
        },
    )
    manifest.assert_required_resolvable()
    return manifest


def dependency_fingerprint_token(manifest: DependencyManifest) -> tuple[Any, ...]:
    """Compact fingerprint for invalidation tests / later discovery."""

    parts: list[Any] = [
        manifest.schema_version,
        str(manifest.session_root),
        manifest.s8_run_id,
        manifest.s7_run_id,
        manifest.s6_run_id,
    ]
    for artifact in manifest.artifacts:
        parts.append(artifact.role)
        parts.append(artifact.fingerprint)
        parts.append(artifact.sha256)
    return tuple(parts)
