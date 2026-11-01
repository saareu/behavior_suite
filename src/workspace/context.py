"""Central workspace selection state (headless; no automatic science launch)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workspace.catalog import (
    detect_duplicate_session_root,
    load_catalog,
    load_catalog_safe,
    write_catalog,
)
from workspace.errors import (
    WorkspaceContextError,
    WorkspaceValidationError,
)
from workspace.models import (
    AnalysisTarget,
    RecentTargetEntry,
    WorkspaceCatalog,
    build_legacy_self_project_catalog,
    catalog_path_for_root,
    normalize_path,
    probe_has_preprocess,
    probe_path_status,
    with_updated_path_status,
)
from workspace.policy import (
    ContextChangeDecision,
    ContextChangeRequest,
    JobBinding,
    evaluate_context_change,
    is_stale_job_result,
    require_context_change_allowed,
)
from workspace.resolve import (
    analysis_open_blocked_reason,
    resolve_analysis_target,
)

ContextListener = Callable[["WorkspaceContextSnapshot"], None]


@dataclass(frozen=True)
class WorkspaceContextSnapshot:
    """Immutable view emitted to listeners on context changes."""

    project_id: str | None
    session_id: str | None
    subsession_id: str | None
    analysis_target_id: str | None
    discovery_root: Path | None
    catalog_path: Path | None
    analysis_target: AnalysisTarget | None
    generation: int
    ephemeral: bool


class WorkspaceContext:
    """Shared application workspace selection.

    Separates global project/session/subsession selection from subsystem-specific
    ``selected_*_run_id`` fields (those remain on per-page controllers).

    Never launches scientific processing. Catalog IO is explicit via
    ``load_from_path`` / ``save_catalog``.
    """

    def __init__(self) -> None:
        self._catalog: WorkspaceCatalog | None = None
        self._catalog_path: Path | None = None
        self._ephemeral: bool = False
        self._project_id: str | None = None
        self._session_id: str | None = None
        self._subsession_id: str | None = None
        self._analysis_target: AnalysisTarget | None = None
        self._generation: int = 0
        self._listeners: list[ContextListener] = []
        self._recent: list[RecentTargetEntry] = []

    # --- read-only properties -------------------------------------------------

    @property
    def catalog(self) -> WorkspaceCatalog | None:
        return self._catalog

    @property
    def catalog_path(self) -> Path | None:
        return self._catalog_path

    @property
    def ephemeral(self) -> bool:
        return self._ephemeral

    @property
    def project_id(self) -> str | None:
        return self._project_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def subsession_id(self) -> str | None:
        return self._subsession_id

    @property
    def analysis_target_id(self) -> str | None:
        if self._analysis_target is None:
            return None
        return self._analysis_target.analysis_target_id

    @property
    def discovery_root(self) -> Path | None:
        if self._analysis_target is None:
            return None
        return self._analysis_target.discovery_root

    @property
    def analysis_target(self) -> AnalysisTarget | None:
        return self._analysis_target

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def recent_targets(self) -> tuple[RecentTargetEntry, ...]:
        return tuple(self._recent)

    def snapshot(self) -> WorkspaceContextSnapshot:
        return WorkspaceContextSnapshot(
            project_id=self._project_id,
            session_id=self._session_id,
            subsession_id=self._subsession_id,
            analysis_target_id=self.analysis_target_id,
            discovery_root=self.discovery_root,
            catalog_path=self._catalog_path,
            analysis_target=self._analysis_target,
            generation=self._generation,
            ephemeral=self._ephemeral,
        )

    # --- listeners ------------------------------------------------------------

    def add_listener(self, listener: ContextListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: ContextListener) -> None:
        self._listeners = [item for item in self._listeners if item is not listener]

    def _notify(self) -> None:
        snap = self.snapshot()
        for listener in list(self._listeners):
            listener(snap)

    # --- catalog lifecycle ----------------------------------------------------

    def clear(self, *, request: ContextChangeRequest | None = None) -> ContextChangeDecision:
        decision = self._evaluate(request or ContextChangeRequest(kind="clear"))
        self._catalog = None
        self._catalog_path = None
        self._ephemeral = False
        self._project_id = None
        self._session_id = None
        self._subsession_id = None
        self._analysis_target = None
        if decision.bump_generation:
            self._generation += 1
        self._notify()
        return decision

    def bind_catalog(
        self,
        catalog: WorkspaceCatalog,
        *,
        catalog_path: Path | str | None = None,
        ephemeral: bool = False,
        select_session_id: str | None = None,
        select_subsession_id: str | None = None,
        require_analysis_open: bool = False,
        request: ContextChangeRequest | None = None,
    ) -> AnalysisTarget | None:
        """Attach a catalog and optionally select a session/subsession."""

        decision = self._evaluate(request or ContextChangeRequest(kind="project"))
        path = (
            normalize_path(catalog_path)
            if catalog_path is not None
            else catalog_path_for_root(catalog.project.root_path)
        )
        self._catalog = catalog
        self._catalog_path = path
        self._ephemeral = ephemeral
        self._project_id = catalog.project.project_id
        self._session_id = None
        self._subsession_id = None
        self._analysis_target = None
        if decision.bump_generation:
            self._generation += 1

        target: AnalysisTarget | None = None
        if select_session_id is not None:
            target = self._select_session_unlocked(
                select_session_id,
                subsession_id=select_subsession_id,
                require_analysis_open=require_analysis_open,
                bump=False,
            )
        self._notify()
        return target

    def load_from_path(
        self,
        catalog_path: Path | str,
        *,
        safe: bool = True,
        select_first_session: bool = False,
        require_analysis_open: bool = False,
        request: ContextChangeRequest | None = None,
    ) -> WorkspaceCatalog | None:
        """Load a durable catalog. Safe mode quarantines malformed payloads."""

        path = normalize_path(catalog_path)
        if safe:
            result = load_catalog_safe(path)
            if result.catalog is None:
                return None
            catalog = result.catalog
        else:
            catalog = load_catalog(path)

        session_id = catalog.sessions[0].session_id if (
            select_first_session and catalog.sessions
        ) else None
        self.bind_catalog(
            catalog,
            catalog_path=path,
            ephemeral=False,
            select_session_id=session_id,
            require_analysis_open=require_analysis_open,
            request=request or ContextChangeRequest(kind="project"),
        )
        return catalog

    def save_catalog(self, *, refresh_paths: bool = True) -> Path:
        if self._catalog is None:
            raise WorkspaceContextError("No catalog is bound.")
        if self._ephemeral and self._catalog_path is None:
            self._catalog_path = catalog_path_for_root(self._catalog.project.root_path)
        assert self._catalog_path is not None
        path = write_catalog(
            self._catalog,
            self._catalog_path,
            refresh_paths=refresh_paths,
        )
        # Reload from what we wrote would race; keep in-memory and mark durable.
        self._catalog = with_updated_path_status(self._catalog) if refresh_paths else self._catalog
        self._catalog_path = path
        self._ephemeral = False
        return path

    # --- selection ------------------------------------------------------------

    def set_project(
        self,
        project_id: str,
        *,
        request: ContextChangeRequest | None = None,
    ) -> None:
        """Select project and clear session/subsession (inconsistent combos forbidden)."""

        if self._catalog is None:
            raise WorkspaceContextError("No catalog is bound.")
        if project_id != self._catalog.project.project_id:
            raise WorkspaceValidationError(
                f"project_id {project_id!r} does not match bound catalog "
                f"{self._catalog.project.project_id!r}."
            )
        decision = self._evaluate(request or ContextChangeRequest(kind="project"))
        self._project_id = project_id
        self._session_id = None
        self._subsession_id = None
        self._analysis_target = None
        if decision.bump_generation:
            self._generation += 1
        self._notify()

    def set_session(
        self,
        session_id: str,
        *,
        project_id: str | None = None,
        require_analysis_open: bool = False,
        request: ContextChangeRequest | None = None,
    ) -> AnalysisTarget:
        decision = self._evaluate(request or ContextChangeRequest(kind="session"))
        if self._catalog is None:
            raise WorkspaceContextError("No catalog is bound.")
        if project_id is not None and project_id != self._catalog.project.project_id:
            raise WorkspaceValidationError(
                "Cannot select session under a different project than the bound catalog."
            )
        target = self._select_session_unlocked(
            session_id,
            subsession_id=None,
            require_analysis_open=require_analysis_open,
            bump=decision.bump_generation,
        )
        self._notify()
        return target

    def set_subsession(
        self,
        subsession_id: str | None,
        *,
        require_analysis_open: bool = False,
        request: ContextChangeRequest | None = None,
    ) -> AnalysisTarget:
        if self._catalog is None or self._session_id is None:
            raise WorkspaceContextError("Select a session before setting a subsession.")
        decision = self._evaluate(request or ContextChangeRequest(kind="subsession"))
        target = self._select_session_unlocked(
            self._session_id,
            subsession_id=subsession_id,
            require_analysis_open=require_analysis_open,
            bump=decision.bump_generation,
        )
        self._notify()
        return target

    def clear_subsession(
        self,
        *,
        require_analysis_open: bool = False,
        request: ContextChangeRequest | None = None,
    ) -> AnalysisTarget:
        return self.set_subsession(
            None,
            require_analysis_open=require_analysis_open,
            request=request,
        )

    def open_session_path(
        self,
        path: Path | str,
        *,
        require_preprocess: bool = True,
        register_ephemeral: bool = True,
        persist: bool = False,
        request: ContextChangeRequest | None = None,
    ) -> AnalysisTarget:
        """Open a filesystem session root without launching science.

        If a durable catalog is already bound and the path is registered, selects it.
        Otherwise builds an ephemeral legacy self-project view (D0-A; migration is D0-B).
        """

        session_root = normalize_path(path)
        decision = self._evaluate(request or ContextChangeRequest(kind="session"))

        if self._catalog is not None:
            existing = self._catalog.find_session_by_root(session_root)
            if existing is not None:
                target = self._select_session_unlocked(
                    existing.session_id,
                    subsession_id=None,
                    require_analysis_open=require_preprocess,
                    bump=decision.bump_generation,
                )
                self._remember_current()
                self._notify()
                return target
            dup = detect_duplicate_session_root(self._catalog, session_root)
            if dup is not None:
                raise WorkspaceValidationError(
                    f"Session root already registered as {dup}: {session_root}"
                )

        if not register_ephemeral and self._catalog is None:
            raise WorkspaceContextError(
                "No catalog is bound and ephemeral registration is disabled."
            )

        catalog = build_legacy_self_project_catalog(
            session_root,
            allow_incomplete=True,
        )
        session = catalog.sessions[0]
        if require_preprocess and not session.has_preprocess_dir:
            raise WorkspaceValidationError(
                f"Analysis open blocked: missing preprocess/ under {session_root}"
            )

        # If a catalog is already bound under a different project, replace with ephemeral
        # view for D0-A (grouping / merge is D0-B).
        self._catalog = catalog
        self._catalog_path = catalog_path_for_root(catalog.project.root_path)
        self._ephemeral = True
        self._project_id = catalog.project.project_id
        target = self._select_session_unlocked(
            session.session_id,
            subsession_id=None,
            require_analysis_open=require_preprocess,
            bump=decision.bump_generation,
        )
        if persist:
            self.save_catalog(refresh_paths=True)
        self._remember_current()
        self._notify()
        return target

    def resolve_analysis_target(self) -> AnalysisTarget:
        if self._analysis_target is None:
            raise WorkspaceContextError("No analysis target is selected.")
        return self._analysis_target

    # --- remember / restore (W4) ---------------------------------------------

    def remember_current(self) -> RecentTargetEntry | None:
        return self._remember_current()

    def set_recent_targets(self, entries: Sequence[RecentTargetEntry]) -> None:
        """Replace remember-only history (never includes science run ids)."""

        self._recent = list(entries)

    def propose_restore(
        self,
        entry: RecentTargetEntry | None = None,
        *,
        require_paths_ok: bool = True,
        require_preprocess: bool = True,
    ) -> RecentTargetEntry | None:
        """Return a validation-gated restore candidate without applying it.

        W4 discrepancy note: default product policy remains remember-only (W4(c)/P11).
        Automatic launch restore is **not** performed by WorkspaceContext construction
        or GUI startup in D0-A. Callers may invoke ``restore_selection`` explicitly
        after presenting this proposal.
        """

        candidate = entry or (self._recent[0] if self._recent else None)
        if candidate is None:
            return None
        if require_paths_ok and (
            probe_path_status(candidate.session_root).value != "ok"
            or probe_path_status(candidate.discovery_root).value != "ok"
        ):
            return None
        # Shared-parent: preprocess on session_root. Independent: may use discovery_root.
        if require_preprocess and not (
            probe_has_preprocess(candidate.session_root)
            or (
                candidate.subsession_id is not None
                and probe_has_preprocess(candidate.discovery_root)
            )
        ):
            return None
        return candidate

    def restore_selection(
        self,
        entry: RecentTargetEntry,
        *,
        require_paths_ok: bool = True,
        require_preprocess: bool = True,
        request: ContextChangeRequest | None = None,
    ) -> AnalysisTarget:
        """Apply a validation-gated selection restore (explicit call only)."""

        proposal = self.propose_restore(
            entry,
            require_paths_ok=require_paths_ok,
            require_preprocess=require_preprocess,
        )
        if proposal is None:
            raise WorkspaceValidationError(
                "Safe selection restoration rejected: path or preprocess gate failed."
            )

        decision = self._evaluate(request or ContextChangeRequest(kind="restore"))
        catalog: WorkspaceCatalog | None = None
        if entry.catalog_path is not None and Path(entry.catalog_path).is_file():
            catalog = load_catalog(entry.catalog_path)
            self._catalog = catalog
            self._catalog_path = normalize_path(entry.catalog_path)
            self._ephemeral = False
        elif self._catalog is None or self._catalog.project.project_id != entry.project_id:
            catalog = build_legacy_self_project_catalog(
                entry.session_root,
                allow_incomplete=not require_preprocess,
            )
            # Preserve remembered ids when building ephemeral only if roots match;
            # otherwise use newly minted catalog and select by path.
            self._catalog = catalog
            self._catalog_path = catalog_path_for_root(catalog.project.root_path)
            self._ephemeral = True

        assert self._catalog is not None
        session = self._catalog.find_session_by_root(entry.session_root)
        if session is None and self._catalog.project.project_id == entry.project_id:
            # Fall back to remembered session_id when roots were rebound offline.
            try:
                session = self._catalog.get_session(entry.session_id)
            except WorkspaceValidationError:
                session = None
        if session is None:
            raise WorkspaceValidationError(
                f"Cannot restore session for path {entry.session_root}"
            )

        self._project_id = self._catalog.project.project_id
        target = self._select_session_unlocked(
            session.session_id,
            subsession_id=entry.subsession_id,
            require_analysis_open=require_preprocess,
            bump=decision.bump_generation,
        )
        self._remember_current()
        self._notify()
        return target

    # --- job safeguards -------------------------------------------------------

    def evaluate_navigation(
        self,
        request: ContextChangeRequest,
    ) -> ContextChangeDecision:
        return evaluate_context_change(request)

    def bind_job(self, *, job_kind: str = "unspecified") -> JobBinding:
        if self._analysis_target is None:
            raise WorkspaceContextError("Cannot bind a job without an analysis target.")
        return JobBinding(
            analysis_target_id=self._analysis_target.analysis_target_id,
            discovery_root=self._analysis_target.discovery_root,
            generation=self._generation,
            job_kind=job_kind,
        )

    def is_stale_job_result(self, binding: JobBinding) -> bool:
        return is_stale_job_result(
            binding,
            current_analysis_target_id=self.analysis_target_id,
            current_generation=self.generation,
        )

    # --- internals ------------------------------------------------------------

    def _evaluate(self, request: ContextChangeRequest) -> ContextChangeDecision:
        return require_context_change_allowed(request)

    def _select_session_unlocked(
        self,
        session_id: str,
        *,
        subsession_id: str | None,
        require_analysis_open: bool,
        bump: bool,
    ) -> AnalysisTarget:
        if self._catalog is None:
            raise WorkspaceContextError("No catalog is bound.")
        # Ensure session belongs to this catalog/project.
        self._catalog.get_session(session_id)
        target = resolve_analysis_target(
            self._catalog,
            session_id=session_id,
            subsession_id=subsession_id,
        )
        if require_analysis_open:
            blocked = analysis_open_blocked_reason(target)
            if blocked is not None:
                raise WorkspaceValidationError(blocked)

        self._project_id = self._catalog.project.project_id
        self._session_id = session_id
        self._subsession_id = subsession_id
        self._analysis_target = target
        if bump:
            self._generation += 1
        return target

    def _remember_current(self) -> RecentTargetEntry | None:
        if self._analysis_target is None or self._catalog is None:
            return None
        session = self._catalog.get_session(self._analysis_target.session_id)
        entry = RecentTargetEntry(
            project_id=self._catalog.project.project_id,
            session_id=session.session_id,
            subsession_id=self._subsession_id,
            analysis_target_id=self._analysis_target.analysis_target_id,
            project_root=self._catalog.project.root_path,
            session_root=session.session_root,
            discovery_root=self._analysis_target.discovery_root,
            display_name=session.display_name,
            catalog_path=self._catalog_path,
        )
        self._recent = [entry, *[item for item in self._recent if item != entry]]
        return entry

    def to_workspace_ref_dict(self) -> dict[str, Any] | None:
        """Optional W6 stub payload for *future* new writes only (not emitted here)."""

        target = self._analysis_target
        if target is None:
            return None
        return {
            "schema_version": "workspace_ref_v1",
            "project_id": target.project_id,
            "session_id": target.session_id,
            "subsession_id": target.subsession_id,
            "analysis_target_id": target.analysis_target_id,
            "preprocess_binding": target.preprocess_binding,
        }
