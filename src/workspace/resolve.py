"""Resolve analysis targets from catalog selection without launching science."""

from __future__ import annotations

from pathlib import Path

from workspace.errors import WorkspaceValidationError
from workspace.models import (
    AnalysisTarget,
    PathStatus,
    PreprocessBinding,
    SessionRecord,
    SubsessionRecord,
    WorkspaceCatalog,
    normalize_path,
    parse_analysis_target_id,
    session_analysis_target_id,
    subsession_analysis_target_id,
)


def science_writes_allowed(binding: PreprocessBinding) -> bool:
    """Return whether new S1–S9 science may be written for this binding (W3).

    ``shared_parent`` is read-only / filter-view only in this phase.
    ``unbound`` has no preprocess yet and cannot host science writes.
    """

    return binding in {"session_default", "independent"}


def session_preprocess_root(session: SessionRecord) -> Path | None:
    if not session.has_preprocess_dir and session.path_status is not PathStatus.OK:
        return None
    candidate = normalize_path(session.session_root) / "preprocess"
    if candidate.is_dir():
        return candidate
    return None


def resolve_session_target(
    catalog: WorkspaceCatalog,
    session: SessionRecord,
) -> AnalysisTarget:
    """Session without an active subsession remains a first-class analysis target."""

    preprocess_root = session_preprocess_root(session)
    has_preprocess = preprocess_root is not None
    return AnalysisTarget(
        analysis_target_id=session_analysis_target_id(session.session_id),
        kind="session",
        discovery_root=normalize_path(session.session_root),
        preprocess_root=preprocess_root,
        preprocess_binding="session_default",
        temporal_filter=None,
        project_id=catalog.project.project_id,
        session_id=session.session_id,
        subsession_id=None,
        science_writes_allowed=True,
        path_status=session.path_status,
        has_preprocess_dir=has_preprocess,
    )


def resolve_subsession_target(
    catalog: WorkspaceCatalog,
    session: SessionRecord,
    subsession: SubsessionRecord,
) -> AnalysisTarget:
    """Resolve a subsession target.

    Shared-parent targets reuse the parent session discovery/preprocess roots for
    read-only inspection. Independent targets use ``analysis_root``.
    """

    binding = subsession.preprocess_binding
    if binding == "shared_parent":
        discovery_root = normalize_path(session.session_root)
        preprocess_root = session_preprocess_root(session)
        path_status = session.path_status
    elif binding == "independent":
        if subsession.analysis_root is None:
            raise WorkspaceValidationError(
                "independent subsession is missing analysis_root."
            )
        discovery_root = normalize_path(subsession.analysis_root)
        preprocess_candidate = discovery_root / "preprocess"
        preprocess_root = preprocess_candidate if preprocess_candidate.is_dir() else None
        path_status = subsession.path_status
    elif binding == "unbound":
        discovery_root = normalize_path(
            subsession.analysis_root
            if subsession.analysis_root is not None
            else session.session_root
        )
        preprocess_root = None
        path_status = subsession.path_status
    else:
        # session_default on a subsession is unusual; treat as parent session root.
        discovery_root = normalize_path(session.session_root)
        preprocess_root = session_preprocess_root(session)
        path_status = session.path_status

    return AnalysisTarget(
        analysis_target_id=subsession_analysis_target_id(subsession.subsession_id),
        kind="subsession",
        discovery_root=discovery_root,
        preprocess_root=preprocess_root,
        preprocess_binding=binding,
        temporal_filter=subsession.temporal_bounds,
        project_id=catalog.project.project_id,
        session_id=session.session_id,
        subsession_id=subsession.subsession_id,
        science_writes_allowed=science_writes_allowed(binding),
        path_status=path_status,
        has_preprocess_dir=preprocess_root is not None,
    )


def resolve_analysis_target(
    catalog: WorkspaceCatalog,
    *,
    session_id: str,
    subsession_id: str | None = None,
) -> AnalysisTarget:
    """Resolve the active analysis target from catalog identities."""

    session = catalog.get_session(session_id)
    if subsession_id is None:
        return resolve_session_target(catalog, session)
    subsession = session.get_subsession(subsession_id)
    return resolve_subsession_target(catalog, session, subsession)


def resolve_analysis_target_id(
    catalog: WorkspaceCatalog,
    analysis_target_id: str,
) -> AnalysisTarget:
    kind, identity = parse_analysis_target_id(analysis_target_id)
    if kind == "session":
        return resolve_analysis_target(catalog, session_id=identity, subsession_id=None)
    for session in catalog.sessions:
        for sub in session.subsessions:
            if sub.subsession_id == identity:
                return resolve_subsession_target(catalog, session, sub)
    raise WorkspaceValidationError(f"Unknown subsession analysis target: {identity}")


def analysis_open_blocked_reason(target: AnalysisTarget) -> str | None:
    """Return a human-readable block reason, or None when analysis open is allowed.

    W2(b): incomplete sessions may be listed; analysis open requires preprocess.
    Missing paths always block analysis open.
    """

    if target.path_status is PathStatus.MISSING:
        return f"Analysis target path is missing: {target.discovery_root}"
    if target.path_status is PathStatus.RELOCATED_CANDIDATE:
        return (
            "Analysis target is a relocation candidate; confirm rebind before opening."
        )
    if not target.has_preprocess_dir:
        return (
            "Analysis open blocked: session is missing preprocess/ "
            f"under {target.discovery_root}"
        )
    return None
