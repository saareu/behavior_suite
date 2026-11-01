"""Versioned workspace catalog models and identity helpers (D0-A).

Scientific S1–S10 artifact schemas are intentionally untouched. Catalog entries
reference filesystem analysis roots; they never rewrite science trees.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from workspace.errors import WorkspaceDuplicateError, WorkspaceValidationError

SCHEMA_VERSION = "workspace_catalog_v1"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({SCHEMA_VERSION})
# Future readers may accept older compatible payloads; writers always emit SCHEMA_VERSION.
COMPATIBLE_READ_SCHEMA_VERSIONS: frozenset[str] = frozenset({SCHEMA_VERSION})

BEHAVIOR_SUITE_DIRNAME = ".behavior_suite"
CATALOG_FILENAME = "workspace_catalog_v1.json"
CATALOG_BACKUP_SUFFIX = ".bak"
SUBSESSIONS_DIRNAME = "subsessions"

AnalysisTargetKind = Literal["session", "subsession"]
RegistrationMode = Literal["auto", "manual"]
PreprocessBinding = Literal[
    "session_default",
    "shared_parent",
    "independent",
    "unbound",
]
SubsessionKind = Literal["temporal_segment", "separate_recording", "hybrid"]
TemporalDomain = Literal["prepared", "raw_source"]


class PathStatus(StrEnum):
    """Filesystem availability of a registered path."""

    OK = "ok"
    MISSING = "missing"
    RELOCATED_CANDIDATE = "relocated_candidate"


def utc_now_iso() -> str:
    """Return an aware UTC timestamp in ISO-8601 form."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_workspace_id() -> str:
    """Mint a stable UUID string identity (never a human-readable slug)."""

    return str(uuid.uuid4())


def normalize_path(path: Path | str) -> Path:
    """Return an absolute, expanded path without requiring existence."""

    return Path(path).expanduser().resolve(strict=False)


def path_to_storage(path: Path | str) -> str:
    """Serialize a path deterministically for catalog JSON."""

    return str(normalize_path(path))


def catalog_dir_for_root(root: Path | str) -> Path:
    """Return `{root}/.behavior_suite` for the catalog-owning root."""

    return normalize_path(root) / BEHAVIOR_SUITE_DIRNAME


def catalog_path_for_root(root: Path | str) -> Path:
    """Deterministic catalog ownership path for a project root.

    Preferred locations (W1):
    - Multi-session / parent project: `{project.root_path}/.behavior_suite/workspace_catalog_v1.json`
    - Legacy self-project: `{session_root}/.behavior_suite/workspace_catalog_v1.json`
      (identical when ``project.root_path == session.session_root``)
    """

    return catalog_dir_for_root(root) / CATALOG_FILENAME


def session_analysis_target_id(session_id: str) -> str:
    return f"session:{session_id}"


def subsession_analysis_target_id(subsession_id: str) -> str:
    return f"subsession:{subsession_id}"


def parse_analysis_target_id(analysis_target_id: str) -> tuple[AnalysisTargetKind, str]:
    if analysis_target_id.startswith("session:"):
        identity = analysis_target_id.removeprefix("session:")
        if not identity:
            raise WorkspaceValidationError("session analysis_target_id is empty.")
        return "session", identity
    if analysis_target_id.startswith("subsession:"):
        identity = analysis_target_id.removeprefix("subsession:")
        if not identity:
            raise WorkspaceValidationError("subsession analysis_target_id is empty.")
        return "subsession", identity
    raise WorkspaceValidationError(
        f"Unrecognized analysis_target_id format: {analysis_target_id!r}"
    )


def validate_subsession_slug(slug: str) -> str:
    """Validate a human-readable subsession slug (never used as stable identity)."""

    if not isinstance(slug, str) or not slug or slug != slug.strip():
        raise WorkspaceValidationError("Subsession slug must be non-empty without outer whitespace.")
    if slug in {".", ".."}:
        raise WorkspaceValidationError("Subsession slug cannot be '.' or '..'.")
    if "/" in slug or "\\" in slug:
        raise WorkspaceValidationError("Subsession slug must be a single path component.")
    return slug


def default_subsession_analysis_root(session_root: Path | str, slug: str) -> Path:
    """W7(a): created subsessions live under ``subsessions/{slug}/``.

    Manual registration may point at arbitrary sibling folders (W7(b)); that path
    is stored on ``SubsessionRecord.analysis_root`` and does not require this helper.
    """

    return normalize_path(session_root) / SUBSESSIONS_DIRNAME / validate_subsession_slug(slug)


class TemporalBounds(BaseModel):
    """Optional temporal window (view/filter metadata for D0-A; science deferred)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: TemporalDomain
    start_idx: int = Field(ge=0)
    end_idx_exclusive: int | None = None

    @model_validator(mode="after")
    def _validate_span(self) -> TemporalBounds:
        if self.end_idx_exclusive is not None and self.end_idx_exclusive <= self.start_idx:
            raise WorkspaceValidationError(
                "temporal_bounds.end_idx_exclusive must be greater than start_idx."
            )
        return self


class SessionFingerprint(BaseModel):
    """Lightweight, rebuildable fingerprint for relocation matching (D0-B)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_present: bool = False
    prepared_video_sha256: str | None = None
    prepare_meta_mtime: str | None = None


class SubsessionRecord(BaseModel):
    """Optional child of a Session. Identity is UUID; slug is display/path only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subsession_id: str
    display_name: str = Field(min_length=1)
    kind: SubsessionKind
    preprocess_binding: PreprocessBinding
    slug: str | None = None
    analysis_root: Path | None = None
    temporal_bounds: TemporalBounds | None = None
    path_status: PathStatus = PathStatus.OK
    last_seen_utc: str | None = None
    notes: str | None = None

    @field_validator("analysis_root", mode="before")
    @classmethod
    def _coerce_analysis_root(cls, value: Any) -> Path | None:
        if value is None or value == "":
            return None
        return normalize_path(value)

    @model_validator(mode="after")
    def _validate_binding_paths(self) -> SubsessionRecord:
        if self.preprocess_binding == "independent" and self.analysis_root is None:
            raise WorkspaceValidationError(
                "independent subsessions require analysis_root."
            )
        if self.preprocess_binding == "shared_parent" and self.analysis_root is not None:
            # Shared-parent is read-only against the parent session tree initially (W3).
            # analysis_root may still be omitted; if present it must not imply a write root.
            pass
        if self.slug is not None:
            validate_subsession_slug(self.slug)
        return self


class SessionRecord(BaseModel):
    """First-class analysis target when ``subsessions`` is empty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    display_name: str = Field(min_length=1)
    session_root: Path
    registration_mode: RegistrationMode = "manual"
    legacy_self_project: bool = False
    has_preprocess_dir: bool = False
    path_status: PathStatus = PathStatus.OK
    last_seen_utc: str | None = None
    fingerprint: SessionFingerprint = Field(default_factory=SessionFingerprint)
    subsessions: tuple[SubsessionRecord, ...] = ()
    notes: str | None = None

    @field_validator("session_root", mode="before")
    @classmethod
    def _coerce_session_root(cls, value: Any) -> Path:
        return normalize_path(value)

    @model_validator(mode="after")
    def _validate_subsession_ids(self) -> SessionRecord:
        seen: set[str] = set()
        for sub in self.subsessions:
            if sub.subsession_id in seen:
                raise WorkspaceDuplicateError(
                    f"Duplicate subsession_id within session: {sub.subsession_id}"
                )
            seen.add(sub.subsession_id)
        return self

    def get_subsession(self, subsession_id: str) -> SubsessionRecord:
        for sub in self.subsessions:
            if sub.subsession_id == subsession_id:
                return sub
        raise WorkspaceValidationError(
            f"Unknown subsession_id {subsession_id!r} under session {self.session_id}."
        )


class ProjectRecord(BaseModel):
    """Logical project. Membership of sessions is by catalog link, not folder nesting (W1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    display_name: str = Field(min_length=1)
    root_path: Path
    registration_mode: RegistrationMode = "manual"
    notes: str | None = None

    @field_validator("root_path", mode="before")
    @classmethod
    def _coerce_root_path(cls, value: Any) -> Path:
        return normalize_path(value)


class WorkspaceCatalog(BaseModel):
    """Durable sidecar catalog. One project ownership root per catalog file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    catalog_id: str
    created_utc: str
    updated_utc: str
    behavior_suite_version_written: str | None = None
    project: ProjectRecord
    sessions: tuple[SessionRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_identities(self) -> WorkspaceCatalog:
        if self.schema_version not in COMPATIBLE_READ_SCHEMA_VERSIONS:
            # Defensive: constructors should not mint unsupported versions.
            raise WorkspaceValidationError(
                f"Unsupported workspace catalog schema_version: {self.schema_version!r}"
            )
        session_ids: set[str] = set()
        session_roots: dict[str, str] = {}
        subsession_ids: set[str] = set()
        analysis_roots: dict[str, str] = {}

        for session in self.sessions:
            if session.session_id in session_ids:
                raise WorkspaceDuplicateError(
                    f"Duplicate session_id in catalog: {session.session_id}"
                )
            session_ids.add(session.session_id)
            root_key = path_to_storage(session.session_root)
            if root_key in session_roots:
                raise WorkspaceDuplicateError(
                    "Duplicate session_root in catalog: "
                    f"{root_key} ({session_roots[root_key]} vs {session.session_id})"
                )
            session_roots[root_key] = session.session_id

            for sub in session.subsessions:
                if sub.subsession_id in subsession_ids:
                    raise WorkspaceDuplicateError(
                        f"Duplicate subsession_id in catalog: {sub.subsession_id}"
                    )
                subsession_ids.add(sub.subsession_id)
                if sub.analysis_root is not None:
                    ark = path_to_storage(sub.analysis_root)
                    if ark in analysis_roots:
                        raise WorkspaceDuplicateError(
                            f"Duplicate subsession analysis_root in catalog: {ark}"
                        )
                    analysis_roots[ark] = sub.subsession_id
        return self

    def get_session(self, session_id: str) -> SessionRecord:
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        raise WorkspaceValidationError(f"Unknown session_id: {session_id}")

    def find_session_by_root(self, session_root: Path | str) -> SessionRecord | None:
        key = path_to_storage(session_root)
        for session in self.sessions:
            if path_to_storage(session.session_root) == key:
                return session
        return None


class AnalysisTarget(BaseModel):
    """Resolved filesystem target for existing S2–S10 discovery scanners."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_target_id: str
    kind: AnalysisTargetKind
    discovery_root: Path
    preprocess_root: Path | None
    preprocess_binding: PreprocessBinding
    temporal_filter: TemporalBounds | None = None
    project_id: str
    session_id: str
    subsession_id: str | None = None
    science_writes_allowed: bool = True
    path_status: PathStatus = PathStatus.OK
    has_preprocess_dir: bool = False

    @field_validator("discovery_root", "preprocess_root", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> Path | None:
        if value is None or value == "":
            return None
        return normalize_path(value)


class WorkspaceRefV1(BaseModel):
    """Optional provenance block for *future* new-write run_meta only (W6).

    Defined in D0-A as a contract stub. Writers must not attach this to existing
    scientific artifacts; emission is deferred to D0-C.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["workspace_ref_v1"] = "workspace_ref_v1"
    project_id: str
    session_id: str
    subsession_id: str | None = None
    analysis_target_id: str
    preprocess_binding: PreprocessBinding


class RecentTargetEntry(BaseModel):
    """Remember-only launch/history entry (W4(c) / P11). Never stores run ids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    session_id: str
    subsession_id: str | None = None
    analysis_target_id: str
    project_root: Path
    session_root: Path
    discovery_root: Path
    display_name: str
    catalog_path: Path | None = None

    @field_validator("project_root", "session_root", "discovery_root", "catalog_path", mode="before")
    @classmethod
    def _coerce_recent_paths(cls, value: Any) -> Path | None:
        if value is None or value == "":
            return None
        return normalize_path(value)


def probe_path_status(path: Path | str) -> PathStatus:
    """Return ``ok`` when the path exists as a directory, else ``missing``."""

    resolved = normalize_path(path)
    if resolved.is_dir():
        return PathStatus.OK
    return PathStatus.MISSING


def probe_has_preprocess(session_root: Path | str) -> bool:
    return (normalize_path(session_root) / "preprocess").is_dir()


def build_legacy_self_project_catalog(
    session_root: Path | str,
    *,
    display_name: str | None = None,
    behavior_suite_version: str | None = None,
    registration_mode: RegistrationMode = "manual",
    allow_incomplete: bool = True,
) -> WorkspaceCatalog:
    """Build an in-memory 1:1 legacy self-project catalog for a session directory.

    Incomplete sessions (missing ``preprocess/``) may be registered when
    ``allow_incomplete`` is True (W2(b)). Analysis open remains separately gated.
    """

    root = normalize_path(session_root)
    has_preprocess = probe_has_preprocess(root)
    if not has_preprocess and not allow_incomplete:
        raise WorkspaceValidationError(
            f"Session is missing preprocess directory: {root / 'preprocess'}"
        )

    now = utc_now_iso()
    name = display_name or root.name or "session"
    project_id = new_workspace_id()
    session_id = new_workspace_id()
    path_status = probe_path_status(root)

    project = ProjectRecord(
        project_id=project_id,
        display_name=name,
        root_path=root,
        registration_mode=registration_mode,
    )
    session = SessionRecord(
        session_id=session_id,
        display_name=name,
        session_root=root,
        registration_mode=registration_mode,
        legacy_self_project=True,
        has_preprocess_dir=has_preprocess,
        path_status=path_status,
        last_seen_utc=now,
        fingerprint=SessionFingerprint(preprocess_present=has_preprocess),
        subsessions=(),
    )
    return WorkspaceCatalog(
        schema_version=SCHEMA_VERSION,
        catalog_id=new_workspace_id(),
        created_utc=now,
        updated_utc=now,
        behavior_suite_version_written=behavior_suite_version,
        project=project,
        sessions=(session,),
    )


def with_updated_path_status(catalog: WorkspaceCatalog) -> WorkspaceCatalog:
    """Return a copy with refreshed ``path_status`` / preprocess flags (no new ids)."""

    sessions: list[SessionRecord] = []
    now = utc_now_iso()
    for session in catalog.sessions:
        session_status = probe_path_status(session.session_root)
        has_preprocess = (
            probe_has_preprocess(session.session_root)
            if session_status is PathStatus.OK
            else False
        )
        subs: list[SubsessionRecord] = []
        for sub in session.subsessions:
            if sub.analysis_root is None:
                subs.append(sub)
                continue
            subs.append(
                sub.model_copy(
                    update={
                        "path_status": probe_path_status(sub.analysis_root),
                        "last_seen_utc": now,
                    }
                )
            )
        sessions.append(
            session.model_copy(
                update={
                    "path_status": session_status,
                    "has_preprocess_dir": has_preprocess,
                    "last_seen_utc": now,
                    "fingerprint": session.fingerprint.model_copy(
                        update={"preprocess_present": has_preprocess}
                    ),
                    "subsessions": tuple(subs),
                }
            )
        )
    return catalog.model_copy(
        update={"sessions": tuple(sessions), "updated_utc": now}
    )
