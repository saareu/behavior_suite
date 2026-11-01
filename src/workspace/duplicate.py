"""Duplicate and ambiguous-association detection for workspace registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from workspace.models import (
    SessionFingerprint,
    SessionRecord,
    WorkspaceCatalog,
    normalize_path,
    path_to_storage,
)
from workspace.scan import ProjectDiscovery, SessionDiscovery


class DuplicateKind(StrEnum):
    SAME_PATH = "same_path"
    FINGERPRINT_COPY = "fingerprint_copy"
    DISPLAY_NAME = "display_name"
    CROSS_CATALOG_PATH = "cross_catalog_path"


@dataclass(frozen=True)
class DuplicateCandidate:
    kind: DuplicateKind
    message: str
    candidate_path: Path
    existing_session_id: str | None = None
    existing_session_root: Path | None = None
    existing_catalog_path: Path | None = None
    fingerprint_sha256: str | None = None


@dataclass(frozen=True)
class AmbiguousAssociation:
    """Operator-facing ambiguity that must not be auto-committed."""

    code: str
    message: str
    paths: tuple[Path, ...]


def find_path_duplicate(
    catalog: WorkspaceCatalog,
    session_root: Path | str,
    *,
    ignore_session_id: str | None = None,
) -> DuplicateCandidate | None:
    key = path_to_storage(session_root)
    for session in catalog.sessions:
        if ignore_session_id is not None and session.session_id == ignore_session_id:
            continue
        if path_to_storage(session.session_root) == key:
            return DuplicateCandidate(
                kind=DuplicateKind.SAME_PATH,
                message=(
                    f"Session root already registered as {session.session_id}: "
                    f"{path_to_storage(session.session_root)}"
                ),
                candidate_path=normalize_path(session_root),
                existing_session_id=session.session_id,
                existing_session_root=session.session_root,
            )
    return None


def find_fingerprint_duplicates(
    catalog: WorkspaceCatalog,
    fingerprint: SessionFingerprint,
    *,
    ignore_session_id: str | None = None,
    candidate_path: Path | str | None = None,
) -> tuple[DuplicateCandidate, ...]:
    """Flag possible copy/relocation when prepared_video_sha256 matches.

    Never auto-merges; operator must confirm primary vs alias / rebind.
    """

    sha = fingerprint.prepared_video_sha256
    if not sha:
        return ()
    matches: list[DuplicateCandidate] = []
    for session in catalog.sessions:
        if ignore_session_id is not None and session.session_id == ignore_session_id:
            continue
        other = session.fingerprint.prepared_video_sha256
        if other and other.lower() == sha.lower():
            matches.append(
                DuplicateCandidate(
                    kind=DuplicateKind.FINGERPRINT_COPY,
                    message=(
                        "Same prepared_video_sha256 under a different path; "
                        "possible copy or relocation."
                    ),
                    candidate_path=normalize_path(candidate_path or session.session_root),
                    existing_session_id=session.session_id,
                    existing_session_root=session.session_root,
                    fingerprint_sha256=sha,
                )
            )
    return tuple(matches)


def find_display_name_collisions(
    catalog: WorkspaceCatalog,
    display_name: str,
    *,
    ignore_session_id: str | None = None,
    candidate_path: Path | str,
) -> tuple[DuplicateCandidate, ...]:
    """Same display name at different paths is allowed; surface for operator review."""

    hits: list[DuplicateCandidate] = []
    for session in catalog.sessions:
        if ignore_session_id is not None and session.session_id == ignore_session_id:
            continue
        if session.display_name == display_name and path_to_storage(
            session.session_root
        ) != path_to_storage(candidate_path):
            hits.append(
                DuplicateCandidate(
                    kind=DuplicateKind.DISPLAY_NAME,
                    message=(
                        f"Display name {display_name!r} already used by "
                        f"{session.session_id} at {session.session_root}"
                    ),
                    candidate_path=normalize_path(candidate_path),
                    existing_session_id=session.session_id,
                    existing_session_root=session.session_root,
                )
            )
    return tuple(hits)


def collect_duplicates_for_session(
    catalog: WorkspaceCatalog | None,
    discovery: SessionDiscovery,
) -> tuple[DuplicateCandidate, ...]:
    if catalog is None:
        return ()
    found: list[DuplicateCandidate] = []
    path_hit = find_path_duplicate(catalog, discovery.session_root)
    if path_hit is not None:
        found.append(path_hit)
        ignore = path_hit.existing_session_id
    else:
        ignore = None
    found.extend(
        find_fingerprint_duplicates(
            catalog,
            discovery.fingerprint,
            ignore_session_id=ignore,
            candidate_path=discovery.session_root,
        )
    )
    found.extend(
        find_display_name_collisions(
            catalog,
            discovery.display_name,
            ignore_session_id=ignore,
            candidate_path=discovery.session_root,
        )
    )
    return tuple(found)


def collect_ambiguous_associations(
    project: ProjectDiscovery,
) -> tuple[AmbiguousAssociation, ...]:
    ambiguous: list[AmbiguousAssociation] = []
    if project.kind == "ambiguous":
        ambiguous.append(
            AmbiguousAssociation(
                code="ambiguous_nesting",
                message=(
                    "Candidate has both a local preprocess/ tree and child session "
                    "directories."
                ),
                paths=(project.root_path, *(s.session_root for s in project.sessions)),
            )
        )
    if project.kind == "empty":
        ambiguous.append(
            AmbiguousAssociation(
                code="unrecognized_layout",
                message="No recognizable project/session layout at candidate path.",
                paths=(project.root_path,),
            )
        )
    # Multiple sessions sharing identical non-empty prepared_video digests.
    by_sha: dict[str, list[SessionDiscovery]] = {}
    for session in project.sessions:
        sha = session.fingerprint.prepared_video_sha256
        if sha:
            by_sha.setdefault(sha.lower(), []).append(session)
    for sha, group in by_sha.items():
        if len(group) > 1:
            ambiguous.append(
                AmbiguousAssociation(
                    code="duplicate_fingerprint_within_scan",
                    message=(
                        "Multiple discovered sessions share prepared_video_sha256="
                        f"{sha}."
                    ),
                    paths=tuple(item.session_root for item in group),
                )
            )
    return tuple(ambiguous)


def session_match_for_path(
    catalog: WorkspaceCatalog,
    session_root: Path | str,
) -> SessionRecord | None:
    return catalog.find_session_by_root(session_root)
