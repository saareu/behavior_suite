"""Registration preview/commit, rebind, and safe reopen (D0-B).

Preview APIs never write catalogs. Commit/rebind require explicit operator
approval flags and validate the full catalog before every save. Scientific
source trees and immutable manifests are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from workspace.catalog import (
    catalog_to_jsonable,
    detect_duplicate_session_root,
    load_catalog,
    load_catalog_safe,
    parse_catalog_payload,
    write_catalog,
)
from workspace.duplicate import (
    AmbiguousAssociation,
    DuplicateCandidate,
    collect_ambiguous_associations,
    collect_duplicates_for_session,
)
from workspace.errors import (
    WorkspaceDuplicateError,
    WorkspaceValidationError,
)
from workspace.models import (
    AnalysisTarget,
    PathStatus,
    ProjectRecord,
    RegistrationMode,
    SessionFingerprint,
    SessionRecord,
    WorkspaceCatalog,
    build_legacy_self_project_catalog,
    catalog_path_for_root,
    new_workspace_id,
    normalize_path,
    path_to_storage,
    probe_has_preprocess,
    probe_path_status,
    utc_now_iso,
    with_updated_path_status,
)
from workspace.resolve import (
    analysis_open_blocked_reason,
    resolve_analysis_target,
)
from workspace.scan import (
    ProjectDiscovery,
    SessionDiscovery,
    discover_session_artifacts,
    scan_candidate,
)

ProposalAction = Literal["create", "reuse", "update", "skip"]


@dataclass(frozen=True)
class RegistrationProposal:
    """One proposed session registration row for operator review."""

    proposal_id: str
    action: ProposalAction
    project_display_name: str
    project_root: Path
    project_id: str
    session_display_name: str
    session_root: Path
    session_id: str
    legacy_self_project: bool
    has_preprocess_dir: bool
    path_status: PathStatus
    fingerprint: SessionFingerprint
    run_counts: dict[str, int]
    present_subsystems: tuple[str, ...]
    missing_or_incomplete_stages: tuple[str, ...]
    issues: tuple[str, ...]
    duplicates: tuple[DuplicateCandidate, ...]
    selected_by_default: bool = True
    registration_mode: RegistrationMode = "auto"


@dataclass(frozen=True)
class RegistrationPreview:
    """Proposed catalog built from discovery — no durable write."""

    catalog_path: Path
    project_root: Path
    project_id: str
    project_display_name: str
    legacy_self_project: bool
    discovery: ProjectDiscovery
    existing_catalog: WorkspaceCatalog | None
    proposals: tuple[RegistrationProposal, ...]
    ambiguous_associations: tuple[AmbiguousAssociation, ...]
    duplicate_candidates: tuple[DuplicateCandidate, ...]
    missing_or_relocated: tuple[str, ...]
    proposed_catalog: WorkspaceCatalog


@dataclass(frozen=True)
class RegistrationCommitResult:
    catalog: WorkspaceCatalog
    catalog_path: Path
    committed_session_ids: tuple[str, ...]
    skipped_proposal_ids: tuple[str, ...]


@dataclass(frozen=True)
class RebindPreview:
    """Preview of catalog-only path rebind for a relocated root."""

    session_id: str
    project_id: str
    old_session_root: Path
    new_session_root: Path
    old_project_root: Path
    new_project_root: Path | None
    rebind_project_root: bool
    fingerprint_match: bool
    affected_session_paths: tuple[Path, ...]
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]
    proposed_catalog: WorkspaceCatalog


@dataclass(frozen=True)
class ReopenResult:
    """Headless reopen of a registered session through the catalog resolver."""

    catalog: WorkspaceCatalog
    catalog_path: Path | None
    analysis_target: AnalysisTarget
    session: SessionRecord
    discovery: SessionDiscovery | None
    analysis_open_blocked: str | None


def validate_catalog(catalog: WorkspaceCatalog) -> WorkspaceCatalog:
    """Validate complete catalog identities/paths before every save."""

    return parse_catalog_payload(catalog_to_jsonable(catalog))


def _load_existing_catalog(path: Path) -> WorkspaceCatalog | None:
    catalog_file = catalog_path_for_root(path)
    if not catalog_file.is_file():
        return None
    result = load_catalog_safe(catalog_file)
    return result.catalog


def _proposal_from_session(
    *,
    discovery: SessionDiscovery,
    project: ProjectRecord,
    existing: WorkspaceCatalog | None,
    legacy_self_project: bool,
) -> RegistrationProposal:
    duplicates = collect_duplicates_for_session(existing, discovery)
    matched = existing.find_session_by_root(discovery.session_root) if existing else None
    if matched is not None:
        action: ProposalAction = "reuse"
        session_id = matched.session_id
        selected = True
    elif any(dup.kind.value == "same_path" for dup in duplicates):
        action = "skip"
        session_id = next(
            dup.existing_session_id
            for dup in duplicates
            if dup.existing_session_id is not None and dup.kind.value == "same_path"
        )
        selected = False
    else:
        action = "create"
        session_id = new_workspace_id()
        selected = True

    path_status = PathStatus(discovery.path_status)
    issues = tuple(issue.message for issue in discovery.issues)
    return RegistrationProposal(
        proposal_id=new_workspace_id(),
        action=action,
        project_display_name=project.display_name,
        project_root=project.root_path,
        project_id=project.project_id,
        session_display_name=discovery.display_name,
        session_root=discovery.session_root,
        session_id=session_id,
        legacy_self_project=legacy_self_project,
        has_preprocess_dir=discovery.has_preprocess_dir,
        path_status=path_status,
        fingerprint=discovery.fingerprint,
        run_counts=discovery.run_counts,
        present_subsystems=tuple(discovery.present_subsystems),
        missing_or_incomplete_stages=discovery.missing_or_incomplete_stages,
        issues=issues,
        duplicates=duplicates,
        selected_by_default=selected,
    )


def _build_proposed_catalog(
    *,
    project: ProjectRecord,
    catalog_id: str,
    created_utc: str,
    proposals: tuple[RegistrationProposal, ...],
    existing: WorkspaceCatalog | None,
    behavior_suite_version: str | None,
) -> WorkspaceCatalog:
    now = utc_now_iso()
    by_id: dict[str, SessionRecord] = {}
    if existing is not None:
        for session in existing.sessions:
            by_id[session.session_id] = session

    for proposal in proposals:
        if proposal.action == "skip":
            continue
        prior = by_id.get(proposal.session_id)
        if prior is not None:
            by_id[proposal.session_id] = prior.model_copy(
                update={
                    "display_name": proposal.session_display_name,
                    "session_root": proposal.session_root,
                    "registration_mode": proposal.registration_mode,
                    "legacy_self_project": proposal.legacy_self_project,
                    "has_preprocess_dir": proposal.has_preprocess_dir,
                    "path_status": proposal.path_status,
                    "last_seen_utc": now,
                    "fingerprint": proposal.fingerprint,
                    # Preserve parent-child (subsessions) associations.
                    "subsessions": prior.subsessions,
                    "notes": prior.notes,
                }
            )
        else:
            by_id[proposal.session_id] = SessionRecord(
                session_id=proposal.session_id,
                display_name=proposal.session_display_name,
                session_root=proposal.session_root,
                registration_mode=proposal.registration_mode,
                legacy_self_project=proposal.legacy_self_project,
                has_preprocess_dir=proposal.has_preprocess_dir,
                path_status=proposal.path_status,
                last_seen_utc=now,
                fingerprint=proposal.fingerprint,
                subsessions=(),
            )

    catalog = WorkspaceCatalog(
        catalog_id=catalog_id,
        created_utc=created_utc,
        updated_utc=now,
        behavior_suite_version_written=behavior_suite_version,
        project=project,
        sessions=tuple(by_id.values()),
    )
    return validate_catalog(catalog)


def preview_registration(
    path: Path | str,
    *,
    existing_catalog: WorkspaceCatalog | None = None,
    discover_runs: bool = True,
    allow_incomplete: bool = True,
    behavior_suite_version: str | None = None,
) -> RegistrationPreview:
    """Build a proposed catalog from discovered sessions without writing.

    Incomplete sessions are included when ``allow_incomplete`` is True (W2(b)).
    """

    discovery = scan_candidate(path, discover_runs=discover_runs)
    root = discovery.root_path
    catalog_file = catalog_path_for_root(root)
    existing = existing_catalog
    if existing is None:
        existing = _load_existing_catalog(root)

    if existing is not None:
        project = existing.project
        catalog_id = existing.catalog_id
        created_utc = existing.created_utc
        version = existing.behavior_suite_version_written or behavior_suite_version
    else:
        project = ProjectRecord(
            project_id=new_workspace_id(),
            display_name=discovery.display_name,
            root_path=root,
            registration_mode="auto",
        )
        catalog_id = new_workspace_id()
        created_utc = utc_now_iso()
        version = behavior_suite_version

    legacy = discovery.kind == "legacy_self_project"
    proposals: list[RegistrationProposal] = []
    for discovered in discovery.sessions:
        if not discovered.has_preprocess_dir and not allow_incomplete:
            continue
        proposals.append(
            _proposal_from_session(
                discovery=discovered,
                project=project,
                existing=existing,
                legacy_self_project=legacy
                and path_to_storage(discovered.session_root) == path_to_storage(root),
            )
        )

    ambiguous = collect_ambiguous_associations(discovery)
    duplicates: list[DuplicateCandidate] = []
    for proposal in proposals:
        duplicates.extend(proposal.duplicates)

    missing_or_relocated: list[str] = []
    if existing is not None:
        discovered_keys = {path_to_storage(p.session_root) for p in proposals}
        for prior in existing.sessions:
            status = probe_path_status(prior.session_root)
            if status is PathStatus.MISSING:
                missing_or_relocated.append(
                    f"missing:{prior.session_id}:{path_to_storage(prior.session_root)}"
                )
            elif path_to_storage(prior.session_root) not in discovered_keys:
                # Still present on disk but outside this scan — keep, note only.
                pass

    proposed = _build_proposed_catalog(
        project=project,
        catalog_id=catalog_id,
        created_utc=created_utc,
        proposals=tuple(proposals),
        existing=existing,
        behavior_suite_version=version,
    )
    # Ensure previously registered missing sessions remain visible.
    if existing is not None:
        proposed_ids = {s.session_id for s in proposed.sessions}
        retained = list(proposed.sessions)
        now = utc_now_iso()
        for prior in existing.sessions:
            if prior.session_id in proposed_ids:
                continue
            status = probe_path_status(prior.session_root)
            retained.append(
                prior.model_copy(
                    update={
                        "path_status": status,
                        "has_preprocess_dir": (
                            probe_has_preprocess(prior.session_root)
                            if status is PathStatus.OK
                            else False
                        ),
                        "last_seen_utc": now if status is PathStatus.OK else prior.last_seen_utc,
                    }
                )
            )
        proposed = validate_catalog(
            proposed.model_copy(update={"sessions": tuple(retained), "updated_utc": now})
        )

    return RegistrationPreview(
        catalog_path=catalog_file,
        project_root=root,
        project_id=project.project_id,
        project_display_name=project.display_name,
        legacy_self_project=legacy,
        discovery=discovery,
        existing_catalog=existing,
        proposals=tuple(proposals),
        ambiguous_associations=ambiguous,
        duplicate_candidates=tuple(duplicates),
        missing_or_relocated=tuple(missing_or_relocated),
        proposed_catalog=proposed,
    )


def commit_registration(
    preview: RegistrationPreview,
    *,
    approved_proposal_ids: tuple[str, ...] | list[str] | None = None,
    operator_approved: bool = False,
    write_backup: bool = True,
    behavior_suite_version: str | None = None,
    catalog_path: Path | str | None = None,
) -> RegistrationCommitResult:
    """Persist selected proposals. Requires ``operator_approved=True``.

    Separate from :func:`preview_registration` — preview never writes.
    ``catalog_path`` may override the default
    ``{project_root}/.behavior_suite/workspace_catalog_v1.json`` location (useful
    for read-only science trees / lab smoke tests).
    """

    if not operator_approved:
        raise WorkspaceValidationError(
            "Catalog registration requires explicit operator approval "
            "(operator_approved=True)."
        )

    if approved_proposal_ids is None:
        selected = {
            p.proposal_id for p in preview.proposals if p.selected_by_default
        }
    else:
        selected = set(approved_proposal_ids)

    unknown = selected - {p.proposal_id for p in preview.proposals}
    if unknown:
        raise WorkspaceValidationError(
            f"Unknown registration proposal ids: {sorted(unknown)}"
        )

    # Reject committing ambiguous scan kinds unless every selected proposal is reuse.
    if preview.discovery.kind == "ambiguous":
        non_reuse = [
            p
            for p in preview.proposals
            if p.proposal_id in selected and p.action not in {"reuse", "update"}
        ]
        if non_reuse:
            raise WorkspaceValidationError(
                "Ambiguous hierarchy requires operator-resolved associations before "
                "creating new catalog entries."
            )

    approved_proposals = tuple(
        p for p in preview.proposals if p.proposal_id in selected and p.action != "skip"
    )
    skipped = tuple(
        p.proposal_id
        for p in preview.proposals
        if p.proposal_id not in selected or p.action == "skip"
    )

    # Guard against duplicate path commits within the selection.
    seen_paths: dict[str, str] = {}
    for proposal in approved_proposals:
        key = path_to_storage(proposal.session_root)
        if key in seen_paths:
            raise WorkspaceDuplicateError(
                f"Duplicate path in approved proposals: {key} "
                f"({seen_paths[key]} vs {proposal.session_id})"
            )
        seen_paths[key] = proposal.session_id
        if preview.existing_catalog is not None:
            dup = detect_duplicate_session_root(
                preview.existing_catalog,
                proposal.session_root,
                ignore_session_id=proposal.session_id
                if proposal.action in {"reuse", "update"}
                else None,
            )
            if dup is not None and proposal.action == "create":
                raise WorkspaceDuplicateError(
                    f"Refusing to register duplicate path already owned by {dup}: {key}"
                )

    existing = preview.existing_catalog
    catalog_id = existing.catalog_id if existing is not None else preview.proposed_catalog.catalog_id
    created_utc = (
        existing.created_utc if existing is not None else preview.proposed_catalog.created_utc
    )
    project = preview.proposed_catalog.project
    version = (
        behavior_suite_version
        or preview.proposed_catalog.behavior_suite_version_written
    )

    catalog = _build_proposed_catalog(
        project=project,
        catalog_id=catalog_id,
        created_utc=created_utc,
        proposals=approved_proposals,
        existing=existing,
        behavior_suite_version=version,
    )
    # Retain missing/unselected prior sessions so they stay visible with status.
    if existing is not None:
        keep_ids = {s.session_id for s in catalog.sessions}
        retained = list(catalog.sessions)
        for session in existing.sessions:
            if session.session_id in keep_ids:
                continue
            retained.append(
                session.model_copy(
                    update={"path_status": probe_path_status(session.session_root)}
                )
            )
        catalog = validate_catalog(
            catalog.model_copy(
                update={"sessions": tuple(retained), "updated_utc": utc_now_iso()}
            )
        )

    catalog = validate_catalog(with_updated_path_status(catalog))
    target_path = (
        normalize_path(catalog_path) if catalog_path is not None else preview.catalog_path
    )
    path = write_catalog(
        catalog,
        target_path,
        refresh_paths=False,
        write_backup=write_backup,
    )
    loaded = load_catalog(path)
    return RegistrationCommitResult(
        catalog=loaded,
        catalog_path=path,
        committed_session_ids=tuple(p.session_id for p in approved_proposals),
        skipped_proposal_ids=skipped,
    )


def rescan_catalog(
    catalog: WorkspaceCatalog | Path | str,
    *,
    discover_runs: bool = True,
) -> RegistrationPreview:
    """Idempotent rescan: reuse UUIDs for known paths; propose incremental adds."""

    if isinstance(catalog, WorkspaceCatalog):
        bound = catalog
        root = bound.project.root_path
    else:
        path = normalize_path(catalog)
        if path.name.endswith(".json"):
            bound = load_catalog(path)
            root = bound.project.root_path
        else:
            bound = load_catalog(catalog_path_for_root(path))
            root = bound.project.root_path
    return preview_registration(
        root,
        existing_catalog=bound,
        discover_runs=discover_runs,
    )


def preview_rebind(
    catalog: WorkspaceCatalog,
    *,
    session_id: str,
    new_session_root: Path | str,
    rebind_project_root: bool | None = None,
) -> RebindPreview:
    """Validate a candidate replacement path and preview catalog-only updates.

    Does not reinterpret an unrelated folder as the same session without
    fingerprint evidence and operator confirmation at commit time.
    """

    session = catalog.get_session(session_id)
    new_root = normalize_path(new_session_root)
    conflicts: list[str] = []
    warnings: list[str] = []

    if probe_path_status(new_root) is not PathStatus.OK:
        conflicts.append(f"Replacement path is not an existing directory: {new_root}")

    dup = detect_duplicate_session_root(
        catalog, new_root, ignore_session_id=session_id
    )
    if dup is not None:
        conflicts.append(
            f"Replacement path already registered as session {dup}: {new_root}"
        )

    new_discovery = (
        discover_session_artifacts(new_root)
        if probe_path_status(new_root) is PathStatus.OK
        else None
    )
    fingerprint_match = False
    if new_discovery is not None:
        old_sha = session.fingerprint.prepared_video_sha256
        new_sha = new_discovery.fingerprint.prepared_video_sha256
        if old_sha and new_sha and old_sha.lower() == new_sha.lower():
            fingerprint_match = True
        elif old_sha and new_sha and old_sha.lower() != new_sha.lower():
            conflicts.append(
                "prepared_video_sha256 mismatch between registered session and "
                "replacement path; refusing silent reinterpretation."
            )
        elif session.fingerprint.preprocess_present and not new_discovery.has_preprocess_dir:
            warnings.append(
                "Registered session had preprocess/; replacement path does not."
            )
        if not fingerprint_match and old_sha and not new_sha:
            warnings.append(
                "No prepared_video_sha256 available on replacement path; "
                "operator must confirm this is the same session."
            )
        if not fingerprint_match and not old_sha:
            warnings.append(
                "No fingerprint digest on the registered session; confirm rebind carefully."
            )

    # Affected paths: session root (+ project root when legacy self-project).
    legacy = session.legacy_self_project or path_to_storage(
        catalog.project.root_path
    ) == path_to_storage(session.session_root)
    should_rebind_project = (
        rebind_project_root if rebind_project_root is not None else legacy
    )
    new_project_root = new_root if should_rebind_project else catalog.project.root_path
    if should_rebind_project and len(catalog.sessions) > 1:
        warnings.append(
            "Rebinding project.root_path while multiple sessions are registered; "
            "verify catalog ownership location."
        )

    affected = [session.session_root, new_root]
    if should_rebind_project:
        affected.append(catalog.project.root_path)

    now = utc_now_iso()
    updated_session = session.model_copy(
        update={
            "session_root": new_root,
            "path_status": (
                PathStatus.OK
                if probe_path_status(new_root) is PathStatus.OK
                else PathStatus.MISSING
            ),
            "has_preprocess_dir": (
                new_discovery.has_preprocess_dir if new_discovery is not None else False
            ),
            "fingerprint": (
                new_discovery.fingerprint
                if new_discovery is not None
                else session.fingerprint
            ),
            "last_seen_utc": now,
            # Retain stable identity + parent-child associations.
            "session_id": session.session_id,
            "subsessions": session.subsessions,
            "legacy_self_project": session.legacy_self_project,
        }
    )
    sessions = tuple(
        updated_session if item.session_id == session_id else item
        for item in catalog.sessions
    )
    project = catalog.project
    if should_rebind_project:
        project = project.model_copy(update={"root_path": new_project_root})

    proposed = validate_catalog(
        catalog.model_copy(
            update={
                "project": project,
                "sessions": sessions,
                "updated_utc": now,
            }
        )
    )

    return RebindPreview(
        session_id=session_id,
        project_id=catalog.project.project_id,
        old_session_root=session.session_root,
        new_session_root=new_root,
        old_project_root=catalog.project.root_path,
        new_project_root=new_project_root if should_rebind_project else None,
        rebind_project_root=should_rebind_project,
        fingerprint_match=fingerprint_match,
        affected_session_paths=tuple(normalize_path(p) for p in affected),
        conflicts=tuple(conflicts),
        warnings=tuple(warnings),
        proposed_catalog=proposed,
    )


def commit_rebind(
    catalog: WorkspaceCatalog,
    preview: RebindPreview,
    *,
    operator_approved: bool = False,
    catalog_path: Path | str | None = None,
    allow_without_fingerprint_match: bool = False,
    write_backup: bool = True,
) -> tuple[WorkspaceCatalog, Path]:
    """Commit a rebind preview. Updates catalog references only."""

    if not operator_approved:
        raise WorkspaceValidationError(
            "Catalog rebind requires explicit operator approval (operator_approved=True)."
        )
    if preview.conflicts:
        raise WorkspaceValidationError(
            "Cannot commit rebind with conflicts: " + "; ".join(preview.conflicts)
        )
    if (
        not preview.fingerprint_match
        and not allow_without_fingerprint_match
        and preview.warnings
    ):
        raise WorkspaceValidationError(
            "Rebind lacks fingerprint match; pass allow_without_fingerprint_match=True "
            "after operator confirmation. Warnings: "
            + "; ".join(preview.warnings)
        )

    if preview.session_id not in {s.session_id for s in catalog.sessions}:
        raise WorkspaceValidationError(
            f"Rebind preview session_id {preview.session_id} is not in the catalog."
        )
    if preview.project_id != catalog.project.project_id:
        raise WorkspaceValidationError("Rebind preview project_id does not match catalog.")

    to_write = validate_catalog(preview.proposed_catalog)
    target = (
        normalize_path(catalog_path)
        if catalog_path is not None
        else catalog_path_for_root(to_write.project.root_path)
    )
    path = write_catalog(to_write, target, refresh_paths=False, write_backup=write_backup)
    return load_catalog(path), path


def reopen_registered_session(
    catalog: WorkspaceCatalog | Path | str,
    session_id: str | None = None,
    *,
    session_root: Path | str | None = None,
    require_preprocess: bool = False,
    discover_runs: bool = False,
) -> ReopenResult:
    """Reopen a registered session and resolve its analysis root.

    Catalog registration is **not** an eligibility gate: missing preprocess only
    blocks when ``require_preprocess=True``. Missing paths remain visible with
    actionable ``path_status`` / ``analysis_open_blocked``.
    """

    catalog_path: Path | None
    if isinstance(catalog, WorkspaceCatalog):
        bound = with_updated_path_status(catalog)
        catalog_path = catalog_path_for_root(bound.project.root_path)
    else:
        path = normalize_path(catalog)
        if path.is_dir():
            catalog_path = catalog_path_for_root(path)
            bound = with_updated_path_status(load_catalog(catalog_path))
        else:
            catalog_path = path
            bound = with_updated_path_status(load_catalog(path))

    session: SessionRecord | None = None
    if session_id is not None:
        session = bound.get_session(session_id)
    elif session_root is not None:
        session = bound.find_session_by_root(session_root)
        if session is None:
            raise WorkspaceValidationError(
                f"No registered session for path: {normalize_path(session_root)}"
            )
    elif len(bound.sessions) == 1:
        session = bound.sessions[0]
    else:
        raise WorkspaceValidationError(
            "session_id or session_root is required when the catalog has multiple sessions."
        )

    assert session is not None
    target = resolve_analysis_target(bound, session_id=session.session_id)
    blocked = analysis_open_blocked_reason(target)
    if require_preprocess and blocked is not None:
        raise WorkspaceValidationError(blocked)

    discovery = (
        discover_session_artifacts(session.session_root)
        if discover_runs and session.path_status is PathStatus.OK
        else None
    )
    return ReopenResult(
        catalog=bound,
        catalog_path=catalog_path if catalog_path.is_file() else catalog_path,
        analysis_target=target,
        session=session,
        discovery=discovery,
        analysis_open_blocked=blocked,
    )


def register_legacy_self_project(
    session_root: Path | str,
    *,
    operator_approved: bool = False,
    discover_runs: bool = True,
) -> RegistrationCommitResult:
    """Convenience: preview + commit a single legacy session-as-project."""

    preview = preview_registration(session_root, discover_runs=discover_runs)
    return commit_registration(preview, operator_approved=operator_approved)


def ephemeral_open_without_catalog(
    session_root: Path | str,
    *,
    require_preprocess: bool = True,
) -> AnalysisTarget:
    """Backward-compatible path open that does not require registration."""

    catalog = build_legacy_self_project_catalog(
        session_root, allow_incomplete=not require_preprocess
    )
    session = catalog.sessions[0]
    target = resolve_analysis_target(catalog, session_id=session.session_id)
    if require_preprocess:
        blocked = analysis_open_blocked_reason(target)
        if blocked is not None:
            raise WorkspaceValidationError(blocked)
    return target
