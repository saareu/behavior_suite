"""Shared workspace shell controller for MainWindow (D0-D1/D0-D2).

Owns orchestration around one :class:`WorkspaceContext`: remember-only recent
list, navigation guards, background discovery generation, register/rebind
preview commit plumbing. Page Browse write-back and analysis-target propagation
are coordinated by MainWindow using this controller. Does not call S1–S10
science runners or rewrite scientific artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from PySide6.QtCore import QObject, QSettings, Signal

from ui.tasks import GuiTaskContext, GuiTaskKind, GuiTaskRunner, TaskAlreadyRunningError
from ui.workspace.recent_store import WorkspaceRecentStore
from workspace.catalog import load_catalog_safe
from workspace.context import WorkspaceContext, WorkspaceContextSnapshot
from workspace.errors import (
    WorkspaceCatalogError,
    WorkspaceCatalogVersionError,
    WorkspaceDuplicateError,
    WorkspaceError,
    WorkspaceNavigationBlocked,
    WorkspaceValidationError,
)
from workspace.migrate import (
    RebindPreview,
    RegistrationCommitResult,
    RegistrationPreview,
    commit_rebind,
    commit_registration,
    preview_rebind,
    preview_registration,
    reopen_registered_session,
    rescan_catalog,
)
from workspace.models import AnalysisTarget, RecentTargetEntry, WorkspaceCatalog
from workspace.policy import (
    ContextChangeDecision,
    ContextChangeKind,
    ContextChangeRequest,
)
from workspace.qt_context import create_qt_workspace_bridge

NavProbe = Callable[[], tuple[bool, bool, bool]]
"""Return (hard_locked, soft_task_running, has_open_s10_draft)."""


@dataclass(frozen=True)
class WorkspaceScanRequest:
    """Identity for one background discovery/preview request."""

    request_id: str
    generation: int
    path: Path
    kind: Literal["registration_preview", "rescan", "rebind_preview"]


@dataclass(frozen=True)
class WorkspaceRegistrationStatus:
    """Compact registration state for the selector / status bar."""

    registered: bool
    ephemeral: bool
    catalog_path: Path | None
    path_status: str | None
    incomplete: bool
    display: str


class WorkspaceShellController(QObject):
    """GUI-facing controller around one shared :class:`WorkspaceContext`."""

    status_message = Signal(str)
    workspace_changed = Signal(object)  # WorkspaceContextSnapshot
    registration_preview_ready = Signal(object)  # RegistrationPreview
    rebind_preview_ready = Signal(object)  # RebindPreview
    background_busy_changed = Signal(bool)
    error_message = Signal(str, str)  # title, detail

    def __init__(
        self,
        context: WorkspaceContext | None = None,
        *,
        settings: QSettings | None = None,
        nav_probe: NavProbe | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context or WorkspaceContext()
        self._bridge = create_qt_workspace_bridge(self.context)
        self._bridge.setParent(self)
        self._recent_store = WorkspaceRecentStore(settings)
        self._nav_probe = nav_probe or (lambda: (False, False, False))
        self._scan_runner = GuiTaskRunner(self)
        self._scan_generation = 0
        self._active_scan: WorkspaceScanRequest | None = None
        self._pending_preview: RegistrationPreview | None = None
        self._pending_rebind: RebindPreview | None = None

        # Remember-only: load history into context without activating.
        loaded = self._recent_store.load()
        self.context.set_recent_targets(loaded)

        self.context.add_listener(self._on_context_snapshot)
        self._bridge.contextChanged.connect(self._emit_workspace_changed)
        self._scan_runner.finished.connect(self._on_scan_finished)

    # --- read API -------------------------------------------------------------

    @property
    def bridge(self) -> QObject:
        return self._bridge

    @property
    def discovery_root(self) -> Path | None:
        return self.context.discovery_root

    @property
    def project_id(self) -> str | None:
        return self.context.project_id

    @property
    def session_id(self) -> str | None:
        return self.context.session_id

    @property
    def subsession_id(self) -> str | None:
        return self.context.subsession_id

    @property
    def analysis_target(self) -> AnalysisTarget | None:
        return self.context.analysis_target

    @property
    def analysis_target_id(self) -> str | None:
        return self.context.analysis_target_id

    @property
    def generation(self) -> int:
        return self.context.generation

    @property
    def ephemeral(self) -> bool:
        return self.context.ephemeral

    @property
    def catalog(self) -> WorkspaceCatalog | None:
        return self.context.catalog

    @property
    def catalog_path(self) -> Path | None:
        return self.context.catalog_path

    @property
    def recent_targets(self) -> tuple[RecentTargetEntry, ...]:
        return self.context.recent_targets

    @property
    def is_background_busy(self) -> bool:
        return self._scan_runner.is_running

    @property
    def pending_registration_preview(self) -> RegistrationPreview | None:
        return self._pending_preview

    @property
    def pending_rebind_preview(self) -> RebindPreview | None:
        return self._pending_rebind

    def registration_status(self) -> WorkspaceRegistrationStatus:
        target = self.context.analysis_target
        catalog = self.context.catalog
        if target is None or catalog is None:
            return WorkspaceRegistrationStatus(
                registered=False,
                ephemeral=False,
                catalog_path=None,
                path_status=None,
                incomplete=False,
                display="No workspace selected",
            )
        session = catalog.get_session(target.session_id)
        registered = not self.context.ephemeral
        incomplete = not session.has_preprocess_dir
        label = session.display_name
        if incomplete:
            label = f"{label} (incomplete)"
        if session.path_status.value == "missing":
            label = f"{label} (missing path)"
        elif self.context.ephemeral:
            label = f"{label} (not registered)"
        return WorkspaceRegistrationStatus(
            registered=registered,
            ephemeral=self.context.ephemeral,
            catalog_path=self.context.catalog_path,
            path_status=session.path_status.value,
            incomplete=incomplete,
            display=label,
        )

    def status_line(self) -> str:
        status = self.registration_status()
        root = self.discovery_root
        if root is None:
            return "Workspace: none (remember-only recent list available)"
        bits = [f"Workspace: {status.display}", f"root={root}"]
        if status.catalog_path is not None and status.registered:
            bits.append(f"catalog={status.catalog_path}")
        if self.context.subsession_id:
            bits.append(f"subsession={self.context.subsession_id}")
        return " · ".join(bits)

    # --- navigation guards ----------------------------------------------------

    def build_change_request(
        self,
        kind: ContextChangeKind = "session",
        *,
        soft_task_running: bool | None = None,
    ) -> ContextChangeRequest:
        hard_locked, soft_running, has_draft = self._nav_probe()
        return ContextChangeRequest(
            kind=kind,
            hard_locked=hard_locked,
            page_task_running=(
                soft_running if soft_task_running is None else soft_task_running
            ),
            has_open_s10_draft=has_draft,
        )

    def evaluate_switch(
        self,
        kind: ContextChangeKind = "session",
    ) -> ContextChangeDecision:
        return self.context.evaluate_navigation(self.build_change_request(kind))

    def ensure_switch_allowed(
        self,
        kind: ContextChangeKind = "session",
    ) -> ContextChangeDecision:
        decision = self.evaluate_switch(kind)
        if not decision.allowed:
            raise WorkspaceNavigationBlocked(
                decision.reason or "Workspace switch blocked."
            )
        return decision

    def cancel_background_scan(self, *, reason: str = "cancelled") -> None:
        """Reject in-flight discovery results (navigate-away / supersede)."""

        self._scan_generation += 1
        self._pending_preview = None
        self._pending_rebind = None
        if self._scan_runner.is_running:
            self._scan_runner.request_cancellation()
            self.status_message.emit(f"Workspace discovery {reason}.")

    def is_stale_scan(self, request: WorkspaceScanRequest) -> bool:
        if self._active_scan is None:
            return True
        if request.request_id != self._active_scan.request_id:
            return True
        return request.generation != self._scan_generation

    # --- open / activate ------------------------------------------------------

    def open_session_path(
        self,
        path: Path | str,
        *,
        require_preprocess: bool = True,
        persist: bool = False,
        remember: bool = True,
    ) -> AnalysisTarget:
        """Browse/open without registration (ephemeral legacy self-project)."""

        decision = self.ensure_switch_allowed("session")
        target = self.context.open_session_path(
            path,
            require_preprocess=require_preprocess,
            register_ephemeral=True,
            persist=persist,
            request=self.build_change_request("session"),
        )
        if remember:
            self._persist_recent()
        if decision.release_s10_draft_lock:
            self.status_message.emit(
                "Workspace changed; release any open S10 draft lock before editing."
            )
        return target

    def activate_recent(self, entry: RecentTargetEntry) -> AnalysisTarget:
        decision = self.ensure_switch_allowed("restore")
        # Missing paths: propose_restore returns None when require_paths_ok.
        proposal = self.context.propose_restore(entry, require_paths_ok=True)
        if proposal is None:
            raise WorkspaceValidationError(
                f"Cannot open recent session; path missing or incomplete: "
                f"{entry.session_root}"
            )
        target = self.context.restore_selection(
            entry,
            request=self.build_change_request("restore"),
        )
        self._persist_recent()
        if decision.release_s10_draft_lock:
            self.status_message.emit(
                "Workspace restored; release any open S10 draft lock before editing."
            )
        return target

    def activate_registered_session(
        self,
        catalog_path: Path | str,
        *,
        session_id: str | None = None,
        require_preprocess: bool = False,
    ) -> AnalysisTarget:
        decision = self.ensure_switch_allowed("session")
        result = reopen_registered_session(
            catalog_path,
            session_id=session_id,
            require_preprocess=require_preprocess,
            discover_runs=False,
        )
        self.context.bind_catalog(
            result.catalog,
            catalog_path=result.catalog_path,
            ephemeral=False,
            select_session_id=result.session.session_id,
            require_analysis_open=require_preprocess,
            request=self.build_change_request("session"),
        )
        self._persist_recent()
        if decision.release_s10_draft_lock:
            self.status_message.emit(
                "Registered session activated; release any open S10 draft lock "
                "before editing."
            )
        assert self.context.analysis_target is not None
        return self.context.analysis_target

    def clear_workspace(self) -> None:
        self.ensure_switch_allowed("clear")
        self.cancel_background_scan(reason="cleared")
        self.context.clear(request=self.build_change_request("clear"))

    def sync_from_legacy_path(
        self,
        path: Path | str | None,
        *,
        require_preprocess: bool = False,
    ) -> AnalysisTarget | None:
        """Dual-write helper for existing MainWindow ``_remember_session`` paths.

        Does not start science. If navigation is blocked, leaves context unchanged
        and returns None (caller may still keep legacy ``_session_root``).
        """

        if path is None or not str(path).strip():
            return None
        decision = self.evaluate_switch("session")
        if not decision.allowed:
            self.status_message.emit(decision.reason or "Workspace sync blocked.")
            return None
        try:
            return self.open_session_path(
                path,
                require_preprocess=require_preprocess,
                persist=False,
                remember=True,
            )
        except WorkspaceError as exc:
            self.error_message.emit("Workspace open failed", str(exc))
            return None

    # --- registration / rescan (preview never writes) -------------------------

    def begin_registration_preview(self, path: Path | str) -> WorkspaceScanRequest:
        root = Path(path).expanduser().resolve(strict=False)
        request = self._begin_scan(root, kind="registration_preview")

        def operation() -> RegistrationPreview:
            return preview_registration(root, discover_runs=True)

        self._start_scan(
            operation,
            request=request,
            task_kind=GuiTaskKind.WORKSPACE_SCAN,
            task_name="workspace registration preview",
            on_success=lambda preview: self._accept_registration_preview(
                request, preview
            ),
        )
        return request

    def begin_rescan(self) -> WorkspaceScanRequest:
        catalog = self.context.catalog
        catalog_path = self.context.catalog_path
        if catalog is None and catalog_path is None:
            raise WorkspaceValidationError("No catalog is bound to rescan.")
        root = (
            catalog.project.root_path
            if catalog is not None
            else Path(catalog_path).parent.parent  # type: ignore[arg-type]
        )
        request = self._begin_scan(root, kind="rescan")
        source: WorkspaceCatalog | Path = catalog if catalog is not None else Path(
            catalog_path  # type: ignore[arg-type]
        )

        def operation() -> RegistrationPreview:
            return rescan_catalog(source)

        self._start_scan(
            operation,
            request=request,
            task_kind=GuiTaskKind.WORKSPACE_RESCAN,
            task_name="workspace catalog rescan",
            on_success=lambda preview: self._accept_registration_preview(
                request, preview
            ),
        )
        return request

    def commit_pending_registration(
        self,
        *,
        approved_proposal_ids: list[str] | tuple[str, ...] | None = None,
        operator_approved: bool = False,
        catalog_path: Path | str | None = None,
        activate: bool = True,
    ) -> RegistrationCommitResult:
        preview = self._pending_preview
        if preview is None:
            raise WorkspaceValidationError("No registration preview is pending.")
        if not operator_approved:
            raise WorkspaceValidationError(
                "Registration commit requires explicit operator confirmation."
            )
        # Ambiguous new creates: backend also rejects; surface clearly here too.
        if preview.discovery.kind == "ambiguous" or preview.ambiguous_associations:
            selected = (
                set(approved_proposal_ids)
                if approved_proposal_ids is not None
                else {
                    p.proposal_id
                    for p in preview.proposals
                    if p.selected_by_default
                }
            )
            creates = [
                p
                for p in preview.proposals
                if p.proposal_id in selected and p.action == "create"
            ]
            if creates:
                raise WorkspaceValidationError(
                    "Ambiguous folder layout blocks new registration until the "
                    "operator resolves project/session associations."
                )
        try:
            result = commit_registration(
                preview,
                approved_proposal_ids=approved_proposal_ids,
                operator_approved=True,
                catalog_path=catalog_path,
            )
        except (WorkspaceDuplicateError, WorkspaceCatalogError, WorkspaceValidationError):
            raise
        except OSError as exc:
            raise WorkspaceCatalogError(
                f"Catalog write failed (read-only or inaccessible location): {exc}"
            ) from exc

        self._pending_preview = None
        if activate and result.committed_session_ids:
            session_id = result.committed_session_ids[0]
            self.context.bind_catalog(
                result.catalog,
                catalog_path=result.catalog_path,
                ephemeral=False,
                select_session_id=session_id,
                require_analysis_open=False,
                request=self.build_change_request("session"),
            )
            self._persist_recent()
        self.status_message.emit(
            f"Registered {len(result.committed_session_ids)} session(s) to "
            f"{result.catalog_path}"
        )
        return result

    def discard_pending_registration(self) -> None:
        self._pending_preview = None
        self.cancel_background_scan(reason="discarded")

    # --- rebind ---------------------------------------------------------------

    def begin_rebind_preview(
        self,
        *,
        session_id: str,
        new_session_root: Path | str,
        rebind_project_root: bool | None = None,
    ) -> WorkspaceScanRequest:
        catalog = self.context.catalog
        if catalog is None:
            raise WorkspaceValidationError("Bind a catalog before rebinding paths.")
        new_root = Path(new_session_root).expanduser().resolve(strict=False)
        request = self._begin_scan(new_root, kind="rebind_preview")

        def operation() -> RebindPreview:
            return preview_rebind(
                catalog,
                session_id=session_id,
                new_session_root=new_root,
                rebind_project_root=rebind_project_root,
            )

        self._start_scan(
            operation,
            request=request,
            task_kind=GuiTaskKind.WORKSPACE_REBIND_PREVIEW,
            task_name="workspace rebind preview",
            on_success=lambda preview: self._accept_rebind_preview(request, preview),
        )
        return request

    def commit_pending_rebind(
        self,
        *,
        operator_approved: bool = False,
        allow_without_fingerprint_match: bool = False,
        catalog_path: Path | str | None = None,
    ) -> WorkspaceCatalog:
        preview = self._pending_rebind
        catalog = self.context.catalog
        if preview is None or catalog is None:
            raise WorkspaceValidationError("No rebind preview is pending.")
        if not operator_approved:
            raise WorkspaceValidationError(
                "Rebind commit requires explicit operator confirmation."
            )
        if preview.conflicts:
            raise WorkspaceValidationError(
                "Cannot commit rebind while conflicts remain "
                "(for example digest mismatch or path collision): "
                + "; ".join(preview.conflicts)
            )
        if (
            not preview.fingerprint_match
            and not allow_without_fingerprint_match
            and preview.warnings
        ):
            raise WorkspaceValidationError(
                "Rebind lacks fingerprint match; require explicit operator override."
            )
        try:
            updated, written_path = commit_rebind(
                catalog,
                preview,
                operator_approved=True,
                allow_without_fingerprint_match=allow_without_fingerprint_match,
                catalog_path=catalog_path or self.context.catalog_path,
            )
        except OSError as exc:
            raise WorkspaceCatalogError(
                f"Catalog rebind write failed (read-only or inaccessible): {exc}"
            ) from exc

        self._pending_rebind = None
        self.context.bind_catalog(
            updated,
            catalog_path=written_path,
            ephemeral=False,
            select_session_id=preview.session_id,
            require_analysis_open=False,
            request=self.build_change_request("session"),
        )
        self._persist_recent()
        self.status_message.emit(
            f"Rebound session {preview.session_id} → {preview.new_session_root}"
        )
        return updated

    def discard_pending_rebind(self) -> None:
        self._pending_rebind = None
        self.cancel_background_scan(reason="discarded")

    def load_catalog_for_selector(self, catalog_path: Path | str) -> WorkspaceCatalog | None:
        """Safe catalog load for selector; never auto-activates a session."""

        path = Path(catalog_path)
        try:
            result = load_catalog_safe(path)
        except WorkspaceCatalogVersionError as exc:
            self.error_message.emit("Unsupported catalog version", str(exc))
            return None
        except WorkspaceCatalogError as exc:
            self.error_message.emit("Catalog load failed", str(exc))
            return None
        if result.catalog is None:
            detail = result.error or "Catalog is missing or was quarantined as corrupt."
            self.error_message.emit("Catalog unavailable", detail)
            return None
        return result.catalog

    # --- internals ------------------------------------------------------------

    def _persist_recent(self) -> None:
        entry = self.context.remember_current()
        if entry is None:
            return
        updated = self._recent_store.remember(entry)
        self.context.set_recent_targets(updated)

    def _on_context_snapshot(self, snap: WorkspaceContextSnapshot) -> None:
        del snap  # Qt bridge mirrors; status refreshed by MainWindow listener.

    def _emit_workspace_changed(self, snap: object) -> None:
        self.workspace_changed.emit(snap)

    def _begin_scan(
        self,
        path: Path,
        *,
        kind: Literal["registration_preview", "rescan", "rebind_preview"],
    ) -> WorkspaceScanRequest:
        if self._scan_runner.is_running:
            self.cancel_background_scan(reason="superseded")
        self._scan_generation += 1
        request = WorkspaceScanRequest(
            request_id=uuid4().hex,
            generation=self._scan_generation,
            path=path,
            kind=kind,
        )
        self._active_scan = request
        self.background_busy_changed.emit(True)
        self.status_message.emit(f"Scanning workspace ({kind.replace('_', ' ')})…")
        return request

    def _start_scan(
        self,
        operation: Callable[[], Any],
        *,
        request: WorkspaceScanRequest,
        task_kind: GuiTaskKind,
        task_name: str,
        on_success: Callable[[Any], None],
    ) -> None:
        context = GuiTaskContext(
            task_id=request.request_id,
            task_kind=task_kind,
            project_dir=request.path,
            raw_video_path=None,
            generation=request.generation,
        )

        def _success(result: Any) -> None:
            if self.is_stale_scan(request) or context.is_cancellation_requested():
                self.status_message.emit("Ignored stale workspace discovery result.")
                return
            on_success(result)

        def _error(exc: BaseException) -> None:
            if self.is_stale_scan(request):
                return
            title = "Workspace discovery failed"
            if isinstance(exc, WorkspaceCatalogVersionError):
                title = "Unsupported catalog version"
            elif isinstance(exc, WorkspaceCatalogError):
                title = "Catalog error"
            elif isinstance(exc, WorkspaceDuplicateError):
                title = "Registration conflict"
            elif isinstance(exc, WorkspaceValidationError):
                title = "Workspace validation"
            self.error_message.emit(title, str(exc))

        try:
            self._scan_runner.start(
                operation,
                task_name=task_name,
                context=context,
                on_success=_success,
                on_error=_error,
            )
        except TaskAlreadyRunningError:
            self.error_message.emit(
                "Workspace busy",
                "A workspace discovery task is already running.",
            )

    def _accept_registration_preview(
        self,
        request: WorkspaceScanRequest,
        preview: RegistrationPreview,
    ) -> None:
        if self.is_stale_scan(request):
            return
        self._pending_preview = preview
        self.registration_preview_ready.emit(preview)
        self.status_message.emit(
            f"Registration preview ready ({len(preview.proposals)} proposal(s))."
        )

    def _accept_rebind_preview(
        self,
        request: WorkspaceScanRequest,
        preview: RebindPreview,
    ) -> None:
        if self.is_stale_scan(request):
            return
        self._pending_rebind = preview
        self.rebind_preview_ready.emit(preview)
        self.status_message.emit("Rebind preview ready.")

    def _on_scan_finished(self) -> None:
        self.background_busy_changed.emit(False)
        if self._active_scan is not None:
            # Keep request id for late stale checks until next begin.
            pass
