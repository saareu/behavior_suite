"""Headless coordination for Subsystem 10 Behavioral Review GUI.

Owns session discovery, S9 package / draft / reviewed-run selection, inspection
playhead state, editing-mode gates, and publish dispatch. All science and
correction logic stays in ``behavioral_review`` — never duplicated here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from behavioral_review import (
    EXCLUSIVE_ETHOGRAM_LABELS,
    MODE_EDITING,
    MODE_INSPECTION,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    STATUS_VOIDED,
    BehavioralReviewError,
    BoutFilter,
    EditingError,
    EditingSession,
    InspectionError,
    PublicationError,
    ReviewedProjectSummary,
    ReviewedPublicationResult,
    ReviewedRunSummary,
    SessionError,
    TimelineView,
    classify_analysis_source,
    open_editing_session,
    reopen_reviewed_run,
    summarize_behavioral_review_project,
)
from behavioral_review.contracts import (
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    CLASS_DRAFT,
)
from behavioral_review.eligibility import assert_draft_publishable
from results_packaging import (
    ResultsPackageProjectSummary,
    ResultsPackageRunSummary,
    summarize_results_package_project,
)
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

FolderOpener = Callable[[Path], bool]
OpenSessionFn = Callable[..., EditingSession]
PublishFn = Callable[..., ReviewedPublicationResult]
ReviewDiscoveryFn = Callable[..., ReviewedProjectSummary]
PackageDiscoveryFn = Callable[..., ResultsPackageProjectSummary]

READINESS_NO_SESSION = "no_session"
READINESS_DISCOVERING = "discovering"
READINESS_NO_PACKAGE = "no_package"
READINESS_READY = "ready"
READINESS_OTHER = "other"

VIEW_SOURCE_AUTOMATED = "automated"
VIEW_SOURCE_WORKING = "working_draft"
VIEW_SOURCE_REVIEWED = "reviewed_published"


class BehavioralReviewUiError(ValueError):
    """Expected, actionable Subsystem 10 UI validation error."""


@dataclass(frozen=True, slots=True)
class DiscoveryRequestIdentity:
    """Identity used to reject stale asynchronous discovery results."""

    request_id: str
    generation: int
    session_root: Path


@dataclass(frozen=True, slots=True)
class OpenSourceIdentity:
    """Identity used to reject stale package / draft open results."""

    request_id: str
    generation: int
    session_root: Path
    package_dir: Path
    source_kind: str
    review_run_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class DecodeRequestIdentity:
    """Identity used to reject stale video-decode results."""

    request_id: str
    generation: int
    package_run_id: str
    sample_idx: int
    prepared_frame_idx: int


@dataclass(frozen=True, slots=True)
class WindowRequestIdentity:
    """Identity used to reject stale ethogram / evidence window loads."""

    request_id: str
    generation: int
    package_run_id: str
    sample_idx: int
    half_window: int
    include_s7: bool


@dataclass(frozen=True, slots=True)
class PublishRequestIdentity:
    """Identity used to reject stale publication results."""

    request_id: str
    generation: int
    review_run_id: str
    parent_package_run_id: str


# Edit operation kinds dispatched through BEHAVIORAL_REVIEW_EDIT workers.
EDIT_OP_ENTER_EDITING = "enter_editing"
EDIT_OP_CREATE_DRAFT = "create_draft"
EDIT_OP_SET_EXCLUSIVE_LABEL = "set_exclusive_label"
EDIT_OP_SET_PARALLEL_PRED = "set_parallel_pred"
EDIT_OP_TRIM_BOUT = "trim_bout"
EDIT_OP_SPLIT_BOUT = "split_bout"
EDIT_OP_SET_STATUS = "set_bout_status"
EDIT_OP_ADD_NOTE = "add_note"
EDIT_OP_UNDO = "undo"
EDIT_OP_REDO = "redo"


@dataclass(frozen=True, slots=True)
class EditRequestIdentity:
    """Identity used to reject stale edit-worker results."""

    request_id: str
    generation: int
    session_root: Path
    package_run_id: str
    review_run_id: str | None
    op: str


@dataclass(frozen=True, slots=True)
class EditSnapshot:
    """Authoritative post-edit UI fields collected off the GUI thread."""

    identity: EditRequestIdentity
    review_run_id: str | None
    review_run_dir: Path | None
    draft_open: bool
    editing_active: bool
    mode: str
    view_source: str
    can_undo: bool
    can_redo: bool
    status_counts: dict[str, int]
    changed_intervals_text: str
    selected_bout_id: str | None
    n_reviewed_bouts: int | None
    run_stage: str


@dataclass(frozen=True, slots=True)
class _PendingEdit:
    """Captured edit kwargs bound to one active mutation request."""

    identity: EditRequestIdentity
    kwargs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class S9PackageOption:
    """One discovered S9 package offered as an S10 review parent."""

    run_id: str
    run_dir: Path
    historical_status: str | None
    classification: str
    currently_eligible: bool
    n_bouts: int | None
    n_dense_samples: int | None
    label: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReviewRunOption:
    """One discovered draft or published reviewed run."""

    run_id: str
    run_dir: Path
    classification: str
    currently_eligible: bool
    parent_package_run_id: str | None
    publication_state: str | None
    historical_status: str | None
    superseded_by: str | None
    label: str
    is_draft: bool


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Payload collected off the GUI thread for one discovery generation."""

    identity: DiscoveryRequestIdentity
    s9_options: tuple[S9PackageOption, ...]
    review_options: tuple[ReviewRunOption, ...]
    s9_summary: ResultsPackageProjectSummary
    review_summary: ReviewedProjectSummary


@dataclass(frozen=True, slots=True)
class OpenSessionSnapshot:
    """Opened editing/inspection session plus fingerprints for staleness."""

    identity: OpenSourceIdentity
    session: EditingSession
    package_run_id: str
    n_dense: int
    n_bouts: int
    directions: tuple[tuple[int, int], ...]
    video_available: bool
    s8_states_available: bool
    warnings: tuple[str, ...]
    view_source: str
    draft_resumed: bool = False


@dataclass(frozen=True, slots=True)
class DecodeSnapshot:
    """Decoded prepared frame for one request identity."""

    identity: DecodeRequestIdentity
    frame_bgr: np.ndarray | None
    prepared_frame_unchanged: bool
    plateau: bool


@dataclass(frozen=True, slots=True)
class WindowSnapshot:
    """Windowed inspection payload for ethogram / evidence panes."""

    identity: WindowRequestIdentity
    timeline: TimelineView | None
    evidence_text: str
    provenance_text: str
    reconstruction_text: str
    bout_rows: tuple[dict[str, Any], ...]
    status_counts: dict[str, int]
    changed_intervals_text: str
    compare_text: str


@dataclass(frozen=True, slots=True)
class PublishPreview:
    """Pre-publish summary shown before confirming publication."""

    parent_package_run_id: str
    parent_package_dir: Path
    reviewer_id: str
    review_run_id: str
    corrected_intervals_text: str
    status_counts: dict[str, int]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    eligible: bool


@dataclass(slots=True)
class BehavioralReviewUiState:
    """Mutable S10 workspace state."""

    session_root: Path | None = None
    s9_options: tuple[S9PackageOption, ...] = ()
    review_options: tuple[ReviewRunOption, ...] = ()
    selected_s9_run_id: str | None = None
    selected_review_run_id: str | None = None
    s9_selection_manual: bool = False
    review_selection_manual: bool = False
    s9_summary: ResultsPackageProjectSummary | None = None
    review_summary: ReviewedProjectSummary | None = None
    readiness_kind: str = READINESS_NO_SESSION
    readiness_error: str | None = None
    readiness_warnings: tuple[str, ...] = ()
    editing_session: EditingSession | None = None
    package_run_id: str | None = None
    package_dir: Path | None = None
    view_source: str = VIEW_SOURCE_AUTOMATED
    analysis_source_kind: str = SOURCE_TYPE_AUTOMATED
    mode: str = MODE_INSPECTION
    reviewer_id: str = ""
    editing_active: bool = False
    draft_open: bool = False
    review_run_id: str | None = None
    review_run_dir: Path | None = None
    n_dense: int = 0
    n_bouts: int = 0
    sample_idx: int = 0
    time_sec: float = 0.0
    prepared_frame_idx: int = 0
    prepared_time_sec: float = 0.0
    prepared_frame_unchanged: bool = False
    plateau: bool = False
    directions: tuple[tuple[int, int], ...] = ()
    selected_focal_id: int | None = None
    selected_partner_id: int | None = None
    selected_channel: str | None = None
    selected_bout_id: str | None = None
    bout_channel_filter: str | None = None
    bout_behavior_filter: str | None = None
    frame_bgr: np.ndarray | None = None
    timeline: TimelineView | None = None
    evidence_text: str = ""
    provenance_text: str = ""
    reconstruction_text: str = ""
    bout_rows: tuple[dict[str, Any], ...] = ()
    status_counts: dict[str, int] = field(default_factory=dict)
    changed_intervals_text: str = ""
    compare_text: str = ""
    can_undo: bool = False
    can_redo: bool = False
    video_available: bool = False
    s8_states_available: bool = False
    task_running: bool = False
    discovery_running: bool = False
    open_running: bool = False
    decode_running: bool = False
    window_running: bool = False
    edit_running: bool = False
    publish_running: bool = False
    discovery_generation: int = 0
    open_generation: int = 0
    decode_generation: int = 0
    window_generation: int = 0
    edit_generation: int = 0
    publish_generation: int = 0
    active_discovery_identity: DiscoveryRequestIdentity | None = None
    active_open_identity: OpenSourceIdentity | None = None
    active_decode_identity: DecodeRequestIdentity | None = None
    active_window_identity: WindowRequestIdentity | None = None
    active_edit_identity: EditRequestIdentity | None = None
    active_publish_identity: PublishRequestIdentity | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_publish_result: ReviewedPublicationResult | None = None
    last_publish_preview: PublishPreview | None = None
    unexpected_error_detail: str | None = None
    inspected_published_only: bool = False
    exclusive_labels: tuple[str, ...] = EXCLUSIVE_ETHOGRAM_LABELS


class BehavioralReviewController:
    """Session discovery, source selection, inspection, editing, and publish."""

    def __init__(
        self,
        *,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        open_session_fn: OpenSessionFn = open_editing_session,
        publish_fn: PublishFn | None = None,
        review_discovery_fn: ReviewDiscoveryFn = summarize_behavioral_review_project,
        package_discovery_fn: PackageDiscoveryFn = summarize_results_package_project,
    ) -> None:
        self.state = BehavioralReviewUiState()
        self._folder_opener = folder_opener
        self._open_session_fn = open_session_fn
        self._publish_fn = publish_fn
        self._review_discovery_fn = review_discovery_fn
        self._package_discovery_fn = package_discovery_fn
        self._pending_edit: _PendingEdit | None = None

    # --- session / discovery -------------------------------------------------

    def prepare_session(self, session_root: Path | str) -> DiscoveryRequestIdentity:
        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise BehavioralReviewUiError(f"Session directory does not exist: {root}")
        self._assert_no_active_mutation("change session")
        self.close_session(release_lock=True)
        self.state.session_root = root
        self.state.readiness_kind = READINESS_DISCOVERING
        self.state.readiness_error = None
        self.state.discovery_generation += 1
        identity = DiscoveryRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.discovery_generation,
            session_root=root,
        )
        self.state.active_discovery_identity = identity
        self.state.discovery_running = True
        return identity

    def set_session(self, session_root: Path | str) -> DiscoveryRequestIdentity:
        return self.prepare_session(session_root)

    def collect_discovery_snapshot(
        self, identity: DiscoveryRequestIdentity
    ) -> DiscoverySnapshot:
        s9_summary = self._package_discovery_fn(identity.session_root)
        review_summary = self._review_discovery_fn(identity.session_root)
        s9_options = tuple(_s9_option_from_summary(run) for run in s9_summary.runs)
        review_options = tuple(
            _review_option_from_summary(run) for run in review_summary.runs
        )
        return DiscoverySnapshot(
            identity=identity,
            s9_options=s9_options,
            review_options=review_options,
            s9_summary=s9_summary,
            review_summary=review_summary,
        )

    def apply_discovery_snapshot(self, snapshot: DiscoverySnapshot) -> bool:
        active = self.state.active_discovery_identity
        if (
            active is None
            or snapshot.identity.request_id != active.request_id
            or snapshot.identity.generation != active.generation
            or self.state.session_root is None
            or snapshot.identity.session_root.resolve(strict=False)
            != self.state.session_root.resolve(strict=False)
        ):
            return False
        self.state.s9_options = snapshot.s9_options
        self.state.review_options = snapshot.review_options
        self.state.s9_summary = snapshot.s9_summary
        self.state.review_summary = snapshot.review_summary
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        if not self.state.s9_selection_manual:
            eligible = [o for o in snapshot.s9_options if o.currently_eligible]
            self.state.selected_s9_run_id = eligible[0].run_id if eligible else None
        elif self.state.selected_s9_run_id not in {o.run_id for o in snapshot.s9_options}:
            self.state.selected_s9_run_id = None
            self.state.s9_selection_manual = False
        if not snapshot.s9_options:
            self.state.readiness_kind = READINESS_NO_PACKAGE
            self.state.readiness_error = "No S9 results packages found in this session."
        else:
            self.state.readiness_kind = READINESS_READY
            self.state.readiness_error = None
        return True

    def select_s9_package(self, run_id: str | None) -> None:
        self.state.selected_s9_run_id = run_id
        self.state.s9_selection_manual = bool(run_id)

    def select_review_run(self, run_id: str | None) -> None:
        self.state.selected_review_run_id = run_id
        self.state.review_selection_manual = bool(run_id)

    def selected_s9_option(self) -> S9PackageOption | None:
        rid = self.state.selected_s9_run_id
        if rid is None:
            return None
        for opt in self.state.s9_options:
            if opt.run_id == rid:
                return opt
        return None

    def selected_review_option(self) -> ReviewRunOption | None:
        rid = self.state.selected_review_run_id
        if rid is None:
            return None
        for opt in self.state.review_options:
            if opt.run_id == rid:
                return opt
        return None

    # --- open package / draft / reviewed ------------------------------------

    def begin_open_automated(self) -> OpenSourceIdentity:
        self._assert_no_active_mutation("open a package")
        opt = self.selected_s9_option()
        if opt is None:
            raise BehavioralReviewUiError("Select an eligible S9 package first.")
        if not opt.currently_eligible:
            raise BehavioralReviewUiError(
                f"S9 package {opt.run_id} is not currently eligible for review."
            )
        if self.state.session_root is None:
            raise BehavioralReviewUiError("No session is open.")
        self._close_editing_session(release_lock=True)
        self.state.open_generation += 1
        identity = OpenSourceIdentity(
            request_id=str(uuid4()),
            generation=self.state.open_generation,
            session_root=self.state.session_root,
            package_dir=opt.run_dir,
            source_kind=VIEW_SOURCE_AUTOMATED,
        )
        self.state.active_open_identity = identity
        self.state.open_running = True
        self.state.task_running = True
        self.state.run_stage = f"Opening automated package {opt.run_id}…"
        self.state.run_error = None
        return identity

    def begin_open_draft(self, *, force_stale_lock: bool = False) -> OpenSourceIdentity:
        self._assert_no_active_mutation("resume a draft")
        del force_stale_lock  # applied in worker via open kwargs
        opt = self.selected_review_option()
        if opt is None or not opt.is_draft:
            raise BehavioralReviewUiError("Select a working draft to resume.")
        parent = opt.parent_package_run_id
        parent_opt = None
        for cand in self.state.s9_options:
            if cand.run_id == parent:
                parent_opt = cand
                break
        if parent_opt is None:
            raise BehavioralReviewUiError(
                f"Draft parent package {parent!r} is not discoverable in this session."
            )
        if self.state.session_root is None:
            raise BehavioralReviewUiError("No session is open.")
        self._close_editing_session(release_lock=True)
        self.state.open_generation += 1
        identity = OpenSourceIdentity(
            request_id=str(uuid4()),
            generation=self.state.open_generation,
            session_root=self.state.session_root,
            package_dir=parent_opt.run_dir,
            source_kind=VIEW_SOURCE_WORKING,
            review_run_dir=opt.run_dir,
        )
        self.state.active_open_identity = identity
        self.state.open_running = True
        self.state.task_running = True
        self.state.run_stage = f"Opening working draft {opt.run_id}…"
        self.state.run_error = None
        return identity

    def begin_open_reviewed(self) -> OpenSourceIdentity:
        self._assert_no_active_mutation("open a reviewed run")
        opt = self.selected_review_option()
        if opt is None or opt.is_draft:
            raise BehavioralReviewUiError(
                "Select a published reviewed run to reopen for inspection."
            )
        parent = opt.parent_package_run_id
        parent_opt = None
        for cand in self.state.s9_options:
            if cand.run_id == parent:
                parent_opt = cand
                break
        if parent_opt is None:
            raise BehavioralReviewUiError(
                f"Reviewed-run parent package {parent!r} is not discoverable."
            )
        if self.state.session_root is None:
            raise BehavioralReviewUiError("No session is open.")
        self._close_editing_session(release_lock=True)
        self.state.open_generation += 1
        identity = OpenSourceIdentity(
            request_id=str(uuid4()),
            generation=self.state.open_generation,
            session_root=self.state.session_root,
            package_dir=parent_opt.run_dir,
            source_kind=VIEW_SOURCE_REVIEWED,
            review_run_dir=opt.run_dir,
        )
        self.state.active_open_identity = identity
        self.state.open_running = True
        self.state.task_running = True
        self.state.run_stage = f"Opening reviewed run {opt.run_id} (inspection only)…"
        self.state.run_error = None
        return identity

    def open_source_worker(
        self,
        identity: OpenSourceIdentity,
        *,
        reviewer_id: str | None = None,
        force_stale_lock: bool = False,
    ) -> OpenSessionSnapshot:
        session = self._open_session_fn(
            identity.package_dir,
            require_video=False,
            require_s8_states=True,
            initial_sample_idx=0,
        )
        directions = _discover_directions(session)
        view_source = identity.source_kind
        warnings = tuple(session.inspection.warnings)
        draft_resumed = False
        if identity.source_kind == VIEW_SOURCE_WORKING:
            # Resume draft off the GUI thread (projection rebuild is expensive).
            view_source = VIEW_SOURCE_WORKING
            rid = (reviewer_id or "").strip()
            if identity.review_run_dir is not None and rid:
                session.enter_editing_mode(rid)
                session.open_draft(
                    identity.review_run_dir,
                    force_stale_lock=force_stale_lock,
                )
                draft_resumed = True
            elif identity.review_run_dir is not None and not rid:
                warnings = warnings + (
                    "Reviewer identity is required to resume a working draft.",
                )
        elif identity.source_kind == VIEW_SOURCE_REVIEWED:
            # Validate reopenability without mutating.
            if identity.review_run_dir is not None:
                reopen_reviewed_run(identity.review_run_dir)
            view_source = VIEW_SOURCE_REVIEWED
            warnings = warnings + (
                "Inspecting a published reviewed run. Editing requires an explicit "
                "new working draft (never mutates the published version).",
            )
        return OpenSessionSnapshot(
            identity=identity,
            session=session,
            package_run_id=session.package_run_id,
            n_dense=session.n_dense_samples,
            n_bouts=session.inspection.n_bouts,
            directions=directions,
            video_available=session.inspection.video is not None,
            s8_states_available=session.inspection.source.s8_states_available,
            warnings=warnings,
            view_source=view_source,
            draft_resumed=draft_resumed,
        )

    def apply_open_snapshot(
        self,
        snapshot: OpenSessionSnapshot,
        *,
        reviewer_id: str | None = None,
        force_stale_lock: bool = False,
    ) -> bool:
        active = self.state.active_open_identity
        if (
            active is None
            or snapshot.identity.request_id != active.request_id
            or snapshot.identity.generation != active.generation
        ):
            with_contextlib_suppress_close(snapshot.session)
            return False
        self.state.editing_session = snapshot.session
        self.state.package_run_id = snapshot.package_run_id
        self.state.package_dir = snapshot.identity.package_dir
        self.state.view_source = snapshot.view_source
        self.state.analysis_source_kind = (
            SOURCE_TYPE_REVIEWED
            if snapshot.view_source == VIEW_SOURCE_REVIEWED
            else SOURCE_TYPE_AUTOMATED
        )
        self.state.n_dense = snapshot.n_dense
        self.state.n_bouts = snapshot.n_bouts
        self.state.directions = snapshot.directions
        self.state.video_available = snapshot.video_available
        self.state.s8_states_available = snapshot.s8_states_available
        self.state.readiness_warnings = snapshot.warnings
        self.state.inspected_published_only = (
            snapshot.view_source == VIEW_SOURCE_REVIEWED
        )
        self.state.mode = MODE_INSPECTION
        self.state.editing_active = False
        self.state.draft_open = False
        self.state.review_run_id = None
        self.state.review_run_dir = None
        if snapshot.directions:
            focal, partner = snapshot.directions[0]
            self.state.selected_focal_id = focal
            self.state.selected_partner_id = partner
            snapshot.session.inspection.select_direction(
                focal_id=focal, partner_id=partner
            )
        playhead = snapshot.session.inspection.playhead
        self._apply_playhead(playhead)
        if (
            snapshot.view_source == VIEW_SOURCE_WORKING
            and snapshot.identity.review_run_dir is not None
        ):
            rid = (reviewer_id or self.state.reviewer_id or "").strip()
            if snapshot.draft_resumed and snapshot.session.draft is not None:
                self.state.reviewer_id = rid or str(snapshot.session.reviewer_id or "")
                self.state.editing_active = True
                self.state.mode = MODE_EDITING
                self.state.draft_open = True
                self.state.review_run_id = snapshot.session.draft.review_run_id
                self.state.review_run_dir = snapshot.identity.review_run_dir
                self.state.view_source = VIEW_SOURCE_WORKING
                self._refresh_edit_flags()
            elif not rid:
                self.state.run_error = (
                    "Reviewer identity is required to resume a working draft."
                )
                self.state.open_running = False
                self.state.task_running = False
                self.state.active_open_identity = None
                return True
            else:
                # Fallback when worker could not resume (e.g. reviewer missing then).
                snapshot.session.enter_editing_mode(rid)
                snapshot.session.open_draft(
                    snapshot.identity.review_run_dir,
                    force_stale_lock=force_stale_lock,
                )
                self.state.reviewer_id = rid
                self.state.editing_active = True
                self.state.mode = MODE_EDITING
                self.state.draft_open = True
                self.state.review_run_id = snapshot.session.draft.review_run_id  # type: ignore[union-attr]
                self.state.review_run_dir = snapshot.identity.review_run_dir
                self.state.view_source = VIEW_SOURCE_WORKING
                self._refresh_edit_flags()
        self.state.open_running = False
        self.state.task_running = False
        self.state.active_open_identity = None
        self.state.run_stage = (
            f"Opened {snapshot.view_source} · {snapshot.package_run_id} · "
            f"{snapshot.n_dense} dense · {snapshot.n_bouts} bouts"
        )
        return True

    def fail_open(self, identity: OpenSourceIdentity, message: str) -> bool:
        active = self.state.active_open_identity
        if (
            active is None
            or identity.request_id != active.request_id
            or identity.generation != active.generation
        ):
            return False
        self.state.open_running = False
        self.state.task_running = False
        self.state.active_open_identity = None
        self.state.run_error = message
        self.state.run_stage = "Open failed"
        return True

    # --- playhead / decode / windows ----------------------------------------

    def seek_sample(self, sample_idx: int) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package before seeking.")
        mapping = session.seek_sample(int(sample_idx), decode=False)
        self._apply_playhead(mapping)
        return self.begin_decode_request()

    def seek_time_sec(self, time_sec: float) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package before seeking.")
        mapping = session.inspection.seek_time_sec(float(time_sec), decode=False)
        self._apply_playhead(mapping)
        return self.begin_decode_request()

    def step(self, delta: int = 1) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package before stepping.")
        mapping = session.inspection.step(int(delta), decode=False)
        self._apply_playhead(mapping)
        return self.begin_decode_request()

    def previous_bout(self) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        try:
            seek = session.inspection.previous_bout()
        except (SessionError, InspectionError, BehavioralReviewError) as exc:
            raise BehavioralReviewUiError(str(exc)) from exc
        if seek is None:
            return None
        self.state.selected_bout_id = seek.bout_id
        self._apply_playhead(seek.start)
        return self.begin_decode_request()

    def next_bout(self) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        try:
            seek = session.inspection.next_bout()
        except (SessionError, InspectionError, BehavioralReviewError) as exc:
            raise BehavioralReviewUiError(str(exc)) from exc
        if seek is None:
            return None
        self.state.selected_bout_id = seek.bout_id
        self._apply_playhead(seek.start)
        return self.begin_decode_request()

    def jump_bout_start(self, bout_id: str | None = None) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        target = bout_id or self.state.selected_bout_id
        if not target:
            raise BehavioralReviewUiError("Select a bout first.")
        try:
            seek = session.inspection.seek_bout_start(target, decode=False)
        except (SessionError, InspectionError, BehavioralReviewError) as exc:
            raise BehavioralReviewUiError(str(exc)) from exc
        self.state.selected_bout_id = seek.bout_id
        self._apply_playhead(seek.start)
        return self.begin_decode_request()

    def jump_bout_end(self, bout_id: str | None = None) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        target = bout_id or self.state.selected_bout_id
        if not target:
            raise BehavioralReviewUiError("Select a bout first.")
        try:
            seek = session.inspection.seek_bout_end(target, decode=False)
        except (SessionError, InspectionError, BehavioralReviewError) as exc:
            raise BehavioralReviewUiError(str(exc)) from exc
        self.state.selected_bout_id = seek.bout_id
        self._apply_playhead(seek.end)
        return self.begin_decode_request()

    def select_direction(self, focal_id: int, partner_id: int) -> None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        session.inspection.select_direction(focal_id=focal_id, partner_id=partner_id)
        self.state.selected_focal_id = int(focal_id)
        self.state.selected_partner_id = int(partner_id)

    def select_channel(self, channel: str | None) -> None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        session.inspection.select_channel(channel)
        self.state.selected_channel = channel

    def select_bout(self, bout_id: str | None) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        if bout_id:
            session.inspection.select_bout(bout_id)
            self.state.selected_bout_id = bout_id
            return self.jump_bout_start(bout_id)
        self.state.selected_bout_id = None
        return None

    def set_bout_filters(
        self,
        *,
        channel: str | None = None,
        behavior_name: str | None = None,
    ) -> None:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        self.state.bout_channel_filter = channel
        self.state.bout_behavior_filter = behavior_name
        channels = frozenset({channel}) if channel else None
        behaviors = frozenset({behavior_name}) if behavior_name else None
        session.inspection.set_bout_filter(
            BoutFilter(channels=channels, behaviors=behaviors)
        )

    def set_analysis_source_kind(self, kind: str) -> str:
        classified = classify_analysis_source(kind)
        self.state.analysis_source_kind = classified
        return classified

    def begin_decode_request(self) -> DecodeRequestIdentity | None:
        session = self.state.editing_session
        if session is None or not self.state.video_available:
            return None
        self.state.decode_generation += 1
        identity = DecodeRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.decode_generation,
            package_run_id=str(self.state.package_run_id or ""),
            sample_idx=int(self.state.sample_idx),
            prepared_frame_idx=int(self.state.prepared_frame_idx),
        )
        self.state.active_decode_identity = identity
        self.state.decode_running = True
        return identity

    def decode_worker(self, identity: DecodeRequestIdentity) -> DecodeSnapshot:
        session = self.state.editing_session
        if session is None or session.inspection.video is None:
            return DecodeSnapshot(
                identity=identity,
                frame_bgr=None,
                prepared_frame_unchanged=True,
                plateau=True,
            )
        # Keep pixmap on plateau: only decode when prepared index changes.
        # Never plateau without a pixmap (first frame / failed prior decode).
        previous = session.inspection._previous_prepared_frame_idx
        decoded = session.inspection.video.read_frame_if_changed(
            identity.prepared_frame_idx,
            previous_prepared_frame_idx=previous,
        )
        unchanged = decoded is None
        if decoded is None and session.inspection.last_decoded is None:
            decoded = session.inspection.video.read_frame(identity.prepared_frame_idx)
            unchanged = False
        plateau = unchanged
        frame = None
        if decoded is not None:
            frame = np.asarray(decoded.image_bgr)
            session.inspection.last_decoded = decoded
            session.inspection._previous_prepared_frame_idx = int(
                identity.prepared_frame_idx
            )
        elif session.inspection.last_decoded is not None:
            frame = np.asarray(session.inspection.last_decoded.image_bgr)
        return DecodeSnapshot(
            identity=identity,
            frame_bgr=frame,
            prepared_frame_unchanged=unchanged,
            plateau=plateau,
        )

    def apply_decode_snapshot(self, snapshot: DecodeSnapshot) -> bool:
        active = self.state.active_decode_identity
        if (
            active is None
            or snapshot.identity.request_id != active.request_id
            or snapshot.identity.generation != active.generation
            or snapshot.identity.sample_idx != self.state.sample_idx
            or snapshot.identity.prepared_frame_idx != self.state.prepared_frame_idx
            or snapshot.identity.package_run_id != str(self.state.package_run_id or "")
        ):
            return False
        if snapshot.frame_bgr is not None:
            self.state.frame_bgr = snapshot.frame_bgr
        self.state.prepared_frame_unchanged = snapshot.prepared_frame_unchanged
        self.state.plateau = snapshot.plateau
        self.state.decode_running = False
        # Keep latest decode identity only; older generations already rejected.
        if active.generation == self.state.decode_generation:
            self.state.active_decode_identity = None
        return True

    def begin_window_request(
        self, *, half_window: int = 240, include_s7: bool = False
    ) -> WindowRequestIdentity:
        if self.state.editing_session is None:
            raise BehavioralReviewUiError("Open a package first.")
        if self.state.edit_running:
            raise BehavioralReviewUiError(
                "Cannot refresh ethogram windows while an edit is in progress."
            )
        self.state.window_generation += 1
        identity = WindowRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.window_generation,
            package_run_id=str(self.state.package_run_id or ""),
            sample_idx=int(self.state.sample_idx),
            half_window=int(half_window),
            include_s7=bool(include_s7),
        )
        self.state.active_window_identity = identity
        self.state.window_running = True
        return identity

    def window_worker(self, identity: WindowRequestIdentity) -> WindowSnapshot:
        session = self.state.editing_session
        assert session is not None
        timeline = session.inspection.timeline_view(
            half_window=identity.half_window,
            include_probability_traces=True,
        )
        evidence_lines: list[str] = []
        s8 = session.inspection.s8_echoed_evidence(half_window=min(120, identity.half_window))
        evidence_lines.append(
            f"S8 echo window [{s8.start_sample_idx}…{s8.end_sample_idx}] "
            f"points={len(s8.points)}"
        )
        for pt in list(s8.points)[:12]:
            prob = pt.prob if pt.prob is not None else float("nan")
            evidence_lines.append(
                f"  {pt.detector_name}: prob={prob:.3f} "
                f"valid={pt.input_valid} origin={pt.score_origin} "
                f"display={pt.display_state}"
            )
        if identity.include_s7:
            s7 = session.inspection.s7_evidence(half_window=min(120, identity.half_window))
            evidence_lines.append(f"S7 availability: {s7.availability}")
            for pt in list(s7.points)[:8]:
                prob = pt.prob if pt.prob is not None else float("nan")
                evidence_lines.append(
                    f"  S7 {pt.detector_name}: prob={prob:.3f} "
                    f"valid={pt.input_valid} display={pt.display_state}"
                )
        prov = session.inspection.provenance_at_playhead()
        if prov is None:
            provenance_text = "No arbitration provenance at playhead."
            reconstruction_text = "—"
        else:
            provenance_text = (
                f"label={prov.behavior_label} recon={prov.label_reconstruction} "
                f"undefined_reason={prov.undefined_reason or '—'} "
                f"hierarchy={prov.hierarchy_residual_kind or '—'}"
            )
            reconstruction_text = str(prov.label_reconstruction or "—")

        bout_rows = _bout_table_rows(session)
        status_counts = _status_counts(session)
        changed = session.changed_intervals() if session.draft is not None else {}
        changed_text = _format_changed_intervals(changed)
        compare_text = ""
        if session.draft is not None and session.working_states is not None:
            cmp = session.compare_automated_versus_reviewed(
                start_sample_idx=max(0, identity.sample_idx - 5),
                end_sample_idx=identity.sample_idx + 5,
            )
            compare_text = (
                f"Automated vs reviewed rows in ±5 samples: {len(cmp)}"
                if cmp is not None
                else ""
            )
        return WindowSnapshot(
            identity=identity,
            timeline=timeline,
            evidence_text="\n".join(evidence_lines),
            provenance_text=provenance_text,
            reconstruction_text=reconstruction_text,
            bout_rows=bout_rows,
            status_counts=status_counts,
            changed_intervals_text=changed_text,
            compare_text=compare_text,
        )

    def apply_window_snapshot(self, snapshot: WindowSnapshot) -> bool:
        active = self.state.active_window_identity
        if (
            active is None
            or snapshot.identity.request_id != active.request_id
            or snapshot.identity.generation != active.generation
            or snapshot.identity.sample_idx != self.state.sample_idx
            or snapshot.identity.package_run_id != str(self.state.package_run_id or "")
        ):
            return False
        self.state.timeline = snapshot.timeline
        self.state.evidence_text = snapshot.evidence_text
        self.state.provenance_text = snapshot.provenance_text
        self.state.reconstruction_text = snapshot.reconstruction_text
        self.state.bout_rows = snapshot.bout_rows
        self.state.status_counts = snapshot.status_counts
        self.state.changed_intervals_text = snapshot.changed_intervals_text
        self.state.compare_text = snapshot.compare_text
        self.state.window_running = False
        if active.generation == self.state.window_generation:
            self.state.active_window_identity = None
        self._refresh_edit_flags()
        return True

    # --- editing ------------------------------------------------------------

    def set_reviewer_id(self, reviewer_id: str) -> None:
        self.state.reviewer_id = str(reviewer_id).strip()

    def enter_editing_mode(self) -> None:
        """Synchronous convenience wrapper (tests / headless)."""

        identity = self.begin_enter_editing()
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Enter-editing result was rejected as stale.")

    def create_draft(self, *, review_root: Path | str | None = None) -> None:
        """Synchronous convenience wrapper (tests / headless)."""

        identity = self.begin_create_draft(review_root=review_root)
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Create-draft result was rejected as stale.")

    def exit_editing_mode(self, *, release_lock: bool = False) -> None:
        self._assert_no_active_mutation("exit editing mode")
        session = self.state.editing_session
        if session is None:
            return
        session.exit_editing_mode(release_lock=release_lock)
        self.state.editing_active = False
        self.state.mode = MODE_INSPECTION
        if release_lock:
            self.state.draft_open = False
        self._refresh_edit_flags()

    def set_exclusive_label(
        self,
        *,
        start_sample_idx: int,
        end_sample_idx: int,
        label: str,
        focal_id: int | None = None,
        partner_id: int | None = None,
    ) -> None:
        identity = self.begin_set_exclusive_label(
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
            label=label,
            focal_id=focal_id,
            partner_id=partner_id,
        )
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Exclusive-label result was rejected as stale.")

    def set_parallel_pred(
        self,
        *,
        channel: str,
        start_sample_idx: int,
        end_sample_idx: int,
        pred_value: int,
    ) -> None:
        identity = self.begin_set_parallel_pred(
            channel=channel,
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
            pred_value=pred_value,
        )
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Parallel-pred result was rejected as stale.")

    def trim_selected_bout(
        self, *, new_start_sample_idx: int, new_end_sample_idx: int
    ) -> None:
        identity = self.begin_trim_selected_bout(
            new_start_sample_idx=new_start_sample_idx,
            new_end_sample_idx=new_end_sample_idx,
        )
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Trim result was rejected as stale.")

    def split_selected_bout(self, *, at_sample_idx: int) -> None:
        identity = self.begin_split_selected_bout(at_sample_idx=at_sample_idx)
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Split result was rejected as stale.")

    def set_bout_status(self, status: str) -> None:
        identity = self.begin_set_bout_status(status=status)
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Status result was rejected as stale.")

    def add_note(self, note_text: str, *, bout_id: str | None = None) -> None:
        identity = self.begin_add_note(note_text, bout_id=bout_id)
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Note result was rejected as stale.")

    def undo(self) -> None:
        identity = self.begin_undo()
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Undo result was rejected as stale.")

    def redo(self) -> None:
        identity = self.begin_redo()
        snapshot = self.edit_worker(identity)
        if not self.apply_edit_snapshot(snapshot):
            raise BehavioralReviewUiError("Redo result was rejected as stale.")

    def begin_enter_editing(self) -> EditRequestIdentity:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open an automated package before editing.")
        if self.state.inspected_published_only:
            raise BehavioralReviewUiError(
                "Published reviewed runs are inspection-only. Create a new working "
                "draft from the parent automated package to edit."
            )
        rid = self.state.reviewer_id.strip()
        if not rid:
            raise BehavioralReviewUiError("Enter a reviewer identity before editing.")
        return self._begin_edit(
            EDIT_OP_ENTER_EDITING,
            {"reviewer_id": rid},
            stage=f"Entering editing mode (reviewer={rid})…",
            require_writable=False,
            require_editing_active=False,
        )

    def begin_create_draft(
        self, *, review_root: Path | str | None = None
    ) -> EditRequestIdentity:
        if self.state.editing_session is None:
            raise BehavioralReviewUiError("Open a package first.")
        if not self.state.editing_active:
            raise BehavioralReviewUiError("Enter editing mode before creating a draft.")
        if self.state.draft_open:
            raise BehavioralReviewUiError("A working draft is already open.")
        return self._begin_edit(
            EDIT_OP_CREATE_DRAFT,
            {"review_root": review_root},
            stage="Creating working draft…",
            require_writable=False,
            require_editing_active=True,
        )

    def begin_set_exclusive_label(
        self,
        *,
        start_sample_idx: int,
        end_sample_idx: int,
        label: str,
        focal_id: int | None = None,
        partner_id: int | None = None,
    ) -> EditRequestIdentity:
        self._require_writable_session()
        focal = int(focal_id if focal_id is not None else self.state.selected_focal_id or 0)
        partner = int(
            partner_id if partner_id is not None else self.state.selected_partner_id or 1
        )
        return self._begin_edit(
            EDIT_OP_SET_EXCLUSIVE_LABEL,
            {
                "focal_id": focal,
                "partner_id": partner,
                "start_sample_idx": int(start_sample_idx),
                "end_sample_idx": int(end_sample_idx),
                "label": str(label),
                "client_op_id": str(uuid4()),
            },
            stage="Applying exclusive label correction…",
        )

    def begin_set_parallel_pred(
        self,
        *,
        channel: str,
        start_sample_idx: int,
        end_sample_idx: int,
        pred_value: int,
    ) -> EditRequestIdentity:
        self._require_writable_session()
        if channel not in (CHANNEL_CONTACT, CHANNEL_SIDE_BY_SIDE):
            raise BehavioralReviewUiError(
                "Parallel corrections require contact or side_by_side."
            )
        mouse_a, mouse_b = 0, 1
        if self.state.directions:
            a, b = self.state.directions[0]
            mouse_a, mouse_b = min(a, b), max(a, b)
        return self._begin_edit(
            EDIT_OP_SET_PARALLEL_PRED,
            {
                "channel": str(channel),
                "mouse_a_id": mouse_a,
                "mouse_b_id": mouse_b,
                "start_sample_idx": int(start_sample_idx),
                "end_sample_idx": int(end_sample_idx),
                "pred_value": int(pred_value),
                "client_op_id": str(uuid4()),
            },
            stage="Applying contact/SBS correction…",
        )

    def begin_trim_selected_bout(
        self, *, new_start_sample_idx: int, new_end_sample_idx: int
    ) -> EditRequestIdentity:
        self._require_writable_session()
        target = self._selected_source_bout_id()
        return self._begin_edit(
            EDIT_OP_TRIM_BOUT,
            {
                "target_bout_id": target,
                "new_start_sample_idx": int(new_start_sample_idx),
                "new_end_sample_idx": int(new_end_sample_idx),
                "client_op_id": str(uuid4()),
            },
            stage="Trimming bout boundary…",
        )

    def begin_split_selected_bout(self, *, at_sample_idx: int) -> EditRequestIdentity:
        self._require_writable_session()
        target = self._selected_source_bout_id()
        return self._begin_edit(
            EDIT_OP_SPLIT_BOUT,
            {
                "target_bout_id": target,
                "at_sample_idx": int(at_sample_idx),
                "client_op_id": str(uuid4()),
            },
            stage="Splitting bout…",
        )

    def begin_set_bout_status(self, status: str) -> EditRequestIdentity:
        self._require_writable_session()
        if status not in (STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNREVIEWED):
            raise BehavioralReviewUiError(
                "Status must be accepted, rejected, or unreviewed."
            )
        target = self._selected_source_bout_id()
        return self._begin_edit(
            EDIT_OP_SET_STATUS,
            {
                "target_bout_id": target,
                "status": status,
                "client_op_id": str(uuid4()),
            },
            stage=f"Setting bout status to {status}…",
        )

    def begin_add_note(
        self, note_text: str, *, bout_id: str | None = None
    ) -> EditRequestIdentity:
        self._require_writable_session()
        target = (
            bout_id
            if bout_id is not None
            else self._selected_source_bout_id(required=False) or None
        )
        return self._begin_edit(
            EDIT_OP_ADD_NOTE,
            {
                "note_text": str(note_text),
                "target_bout_id": target,
                "client_op_id": str(uuid4()),
            },
            stage="Adding reviewer note…",
        )

    def begin_undo(self) -> EditRequestIdentity:
        self._require_writable_session()
        if not self.state.can_undo:
            raise BehavioralReviewUiError("Nothing to undo.")
        return self._begin_edit(EDIT_OP_UNDO, {}, stage="Undoing last correction…")

    def begin_redo(self) -> EditRequestIdentity:
        self._require_writable_session()
        if not self.state.can_redo:
            raise BehavioralReviewUiError("Nothing to redo.")
        return self._begin_edit(EDIT_OP_REDO, {}, stage="Redoing correction…")

    def edit_worker(self, identity: EditRequestIdentity) -> EditSnapshot:
        pending = self._pending_edit
        if pending is None or pending.identity.request_id != identity.request_id:
            raise BehavioralReviewUiError("No pending edit matches this request.")
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Editing session closed before edit worker ran.")
        kwargs = dict(pending.kwargs)
        op = identity.op
        if op == EDIT_OP_ENTER_EDITING:
            session.enter_editing_mode(str(kwargs["reviewer_id"]))
            stage = f"Editing mode active (reviewer={kwargs['reviewer_id']})"
        elif op == EDIT_OP_CREATE_DRAFT:
            draft = session.create_draft(review_root=kwargs.get("review_root"))
            stage = f"Working draft created: {draft.review_run_id}"
        elif op == EDIT_OP_SET_EXCLUSIVE_LABEL:
            session.set_exclusive_label(**kwargs)
            stage = "Exclusive label correction applied"
        elif op == EDIT_OP_SET_PARALLEL_PRED:
            session.set_parallel_pred(**kwargs)
            stage = "Contact/SBS correction applied"
        elif op == EDIT_OP_TRIM_BOUT:
            session.trim_bout_boundary(**kwargs)
            stage = "Bout trim applied"
        elif op == EDIT_OP_SPLIT_BOUT:
            session.split_bout_at_sample(**kwargs)
            stage = "Bout split applied"
        elif op == EDIT_OP_SET_STATUS:
            session.set_bout_review_status(**kwargs)
            stage = f"Bout status set to {kwargs.get('status')}"
        elif op == EDIT_OP_ADD_NOTE:
            session.add_note(**kwargs)
            stage = "Reviewer note added"
        elif op == EDIT_OP_UNDO:
            session.undo()
            stage = "Undo applied"
        elif op == EDIT_OP_REDO:
            session.redo()
            stage = "Redo applied"
        else:
            raise BehavioralReviewUiError(f"Unknown edit operation: {op!r}")
        return self._build_edit_snapshot(identity, run_stage=stage)

    def apply_edit_snapshot(self, snapshot: EditSnapshot) -> bool:
        """Apply a committed edit result. Never rolls back journal commits."""

        active = self.state.active_edit_identity
        if (
            active is None
            or snapshot.identity.request_id != active.request_id
            or snapshot.identity.generation != active.generation
        ):
            self._clear_pending_edit_if_matches(snapshot.identity)
            return False
        if not self._edit_identity_matches_current_view(snapshot.identity):
            # Journal commit already durable; drop only the stale GUI callback.
            self._finish_edit_request(success_stage=None)
            return False
        self.state.editing_active = snapshot.editing_active
        self.state.mode = snapshot.mode
        self.state.draft_open = snapshot.draft_open
        self.state.review_run_id = snapshot.review_run_id
        self.state.review_run_dir = snapshot.review_run_dir
        self.state.view_source = snapshot.view_source
        self.state.can_undo = snapshot.can_undo
        self.state.can_redo = snapshot.can_redo
        self.state.status_counts = dict(snapshot.status_counts)
        self.state.changed_intervals_text = snapshot.changed_intervals_text
        if snapshot.selected_bout_id is not None:
            self.state.selected_bout_id = snapshot.selected_bout_id
        elif self.state.selected_bout_id:
            # Boundary ops may invalidate the previous display id.
            known = {row.get("bout_id") for row in self.state.bout_rows}
            if (
                self.state.bout_rows
                and self.state.selected_bout_id not in known
                and snapshot.identity.op
                in {EDIT_OP_TRIM_BOUT, EDIT_OP_SPLIT_BOUT, EDIT_OP_UNDO, EDIT_OP_REDO}
            ):
                self.state.selected_bout_id = None
        self.state.run_stage = snapshot.run_stage
        self.state.run_error = None
        self._finish_edit_request(success_stage=snapshot.run_stage)
        return True

    def fail_edit(self, identity: EditRequestIdentity, message: str) -> bool:
        """Record a pre-commit or worker failure without inventing a rollback."""

        active = self.state.active_edit_identity
        if (
            active is None
            or identity.request_id != active.request_id
            or identity.generation != active.generation
        ):
            return False
        self.state.run_error = message
        self.state.run_stage = f"Edit failed: {identity.op}"
        self._finish_edit_request(success_stage=None)
        return True

    def refresh_after_stale_edit(self) -> None:
        """Reload undo/status flags from the live session after a rejected callback."""

        if self.state.editing_session is None:
            return
        if self.state.editing_session.draft is not None:
            self.state.draft_open = True
            self.state.review_run_id = self.state.editing_session.draft.review_run_id
            self.state.review_run_dir = self.state.editing_session.draft.review_run_dir
            self.state.view_source = VIEW_SOURCE_WORKING
        self._refresh_edit_flags()

    # --- publish ------------------------------------------------------------

    def build_publish_preview(self) -> PublishPreview:
        session = self.state.editing_session
        if session is None or session.draft is None:
            raise BehavioralReviewUiError("Open a writable draft before publishing.")
        if not self.state.editing_active:
            raise BehavioralReviewUiError("Editing mode must be active to publish.")
        report = assert_draft_publishable(
            session.draft,
            source=session.inspection.source,
            require_writer_lock=True,
        )
        status_counts = _status_counts(session)
        preview = PublishPreview(
            parent_package_run_id=session.package_run_id,
            parent_package_dir=session.inspection.source.package_dir,
            reviewer_id=str(session.reviewer_id or ""),
            review_run_id=session.draft.review_run_id,
            corrected_intervals_text=_format_changed_intervals(
                session.changed_intervals()
            ),
            status_counts=status_counts,
            blockers=tuple(report.blockers),
            warnings=tuple(report.warnings),
            eligible=not report.blockers,
        )
        self.state.last_publish_preview = preview
        return preview

    def begin_publish(self) -> PublishRequestIdentity:
        self._assert_no_active_mutation("publish")
        preview = self.build_publish_preview()
        if preview.blockers:
            raise BehavioralReviewUiError(
                "Publication blocked:\n- " + "\n- ".join(preview.blockers)
            )
        session = self.state.editing_session
        assert session is not None and session.draft is not None
        self.state.publish_generation += 1
        identity = PublishRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.publish_generation,
            review_run_id=session.draft.review_run_id,
            parent_package_run_id=session.package_run_id,
        )
        self.state.active_publish_identity = identity
        self.state.publish_running = True
        self.state.task_running = True
        self.state.run_stage = "Publishing reviewed results…"
        self.state.run_error = None
        return identity

    def publish_worker(self, identity: PublishRequestIdentity) -> ReviewedPublicationResult:
        del identity
        session = self.state.editing_session
        assert session is not None
        if self._publish_fn is not None:
            return self._publish_fn(session)
        result: ReviewedPublicationResult = session.publish_reviewed_run()
        return result

    def apply_publish_result(
        self, identity: PublishRequestIdentity, result: ReviewedPublicationResult
    ) -> bool:
        active = self.state.active_publish_identity
        if (
            active is None
            or identity.request_id != active.request_id
            or identity.generation != active.generation
        ):
            return False
        self.state.last_publish_result = result
        self.state.publish_running = False
        self.state.task_running = False
        self.state.active_publish_identity = None
        if result.success:
            self.state.run_stage = (
                f"Published {result.review_run_id} "
                f"(eligible={result.currently_eligible})"
            )
            # Draft is now published; leave inspection of parent package.
            self.state.draft_open = False
            self.state.editing_active = False
            self.state.mode = MODE_INSPECTION
            if self.state.editing_session is not None:
                self.state.editing_session.exit_editing_mode(release_lock=False)
                self.state.editing_session.draft = None
        else:
            self.state.run_error = result.error_message or "Publication failed"
            self.state.run_stage = "Publication failed"
        return True

    def fail_publish(self, identity: PublishRequestIdentity, message: str) -> bool:
        active = self.state.active_publish_identity
        if (
            active is None
            or identity.request_id != active.request_id
            or identity.generation != active.generation
        ):
            return False
        self.state.publish_running = False
        self.state.task_running = False
        self.state.active_publish_identity = None
        self.state.run_error = message
        self.state.run_stage = "Publication failed"
        return True

    # --- housekeeping -------------------------------------------------------

    def open_folder(self, path: Path | str) -> bool:
        return bool(self._folder_opener(Path(path)))

    def close_session(self, *, release_lock: bool = True) -> None:
        self._assert_no_active_mutation("close the session")
        self._close_editing_session(release_lock=release_lock)
        self.state.session_root = None
        self.state.s9_options = ()
        self.state.review_options = ()
        self.state.selected_s9_run_id = None
        self.state.selected_review_run_id = None
        self.state.s9_selection_manual = False
        self.state.review_selection_manual = False
        self.state.s9_summary = None
        self.state.review_summary = None
        self.state.readiness_kind = READINESS_NO_SESSION
        self.state.readiness_error = None
        self.state.readiness_warnings = ()
        self.state.discovery_running = False
        self.state.active_discovery_identity = None

    def _close_editing_session(self, *, release_lock: bool) -> None:
        session = self.state.editing_session
        if session is not None:
            try:
                if session.draft is not None:
                    session.close_draft(release_lock=release_lock)
                session.exit_editing_mode(release_lock=False)
            except (EditingError, BehavioralReviewError, PublicationError):
                pass
            try:
                if session.inspection.video is not None:
                    session.inspection.video.clear_cache()
            except Exception:  # noqa: BLE001
                pass
        self.state.editing_session = None
        self.state.package_run_id = None
        self.state.package_dir = None
        self.state.view_source = VIEW_SOURCE_AUTOMATED
        self.state.analysis_source_kind = SOURCE_TYPE_AUTOMATED
        self.state.mode = MODE_INSPECTION
        self.state.editing_active = False
        self.state.draft_open = False
        self.state.review_run_id = None
        self.state.review_run_dir = None
        self.state.n_dense = 0
        self.state.n_bouts = 0
        self.state.sample_idx = 0
        self.state.time_sec = 0.0
        self.state.prepared_frame_idx = 0
        self.state.prepared_time_sec = 0.0
        self.state.frame_bgr = None
        self.state.timeline = None
        self.state.bout_rows = ()
        self.state.evidence_text = ""
        self.state.provenance_text = ""
        self.state.reconstruction_text = ""
        self.state.inspected_published_only = False
        self.state.can_undo = False
        self.state.can_redo = False
        self.state.open_running = False
        self.state.decode_running = False
        self.state.window_running = False
        self.state.edit_running = False
        self.state.publish_running = False
        self.state.task_running = False
        self.state.active_open_identity = None
        self.state.active_decode_identity = None
        self.state.active_window_identity = None
        self.state.active_edit_identity = None
        self.state.active_publish_identity = None
        self._pending_edit = None

    def _require_writable_session(self) -> EditingSession:
        session = self.state.editing_session
        if session is None:
            raise BehavioralReviewUiError("Open a package first.")
        if not self.state.editing_active or not self.state.draft_open:
            raise BehavioralReviewUiError(
                "Editing requires active editing mode and an open working draft."
            )
        return session

    def _assert_no_active_mutation(self, action: str) -> None:
        if self.state.edit_running or self.state.active_edit_identity is not None:
            raise BehavioralReviewUiError(
                f"Cannot {action} while an edit is in progress."
            )
        if self.state.publish_running or self.state.active_publish_identity is not None:
            raise BehavioralReviewUiError(
                f"Cannot {action} while publication is in progress."
            )

    def _begin_edit(
        self,
        op: str,
        kwargs: Mapping[str, Any],
        *,
        stage: str,
        require_writable: bool = True,
        require_editing_active: bool = False,
    ) -> EditRequestIdentity:
        if self.state.edit_running or self.state.active_edit_identity is not None:
            raise BehavioralReviewUiError(
                "Another edit is already running for this working draft."
            )
        if self.state.publish_running or self.state.active_publish_identity is not None:
            raise BehavioralReviewUiError(
                "Cannot edit while publication is in progress."
            )
        if self.state.open_running or self.state.active_open_identity is not None:
            raise BehavioralReviewUiError(
                "Cannot edit while a package/draft open task is running."
            )
        if self.state.session_root is None or self.state.package_run_id is None:
            raise BehavioralReviewUiError("Open a package first.")
        if require_writable:
            self._require_writable_session()
        elif require_editing_active and not self.state.editing_active:
            raise BehavioralReviewUiError("Enter editing mode before creating a draft.")
        self.state.edit_generation += 1
        identity = EditRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.edit_generation,
            session_root=self.state.session_root,
            package_run_id=str(self.state.package_run_id),
            review_run_id=self.state.review_run_id,
            op=op,
        )
        self._pending_edit = _PendingEdit(identity=identity, kwargs=kwargs)
        self.state.active_edit_identity = identity
        self.state.edit_running = True
        self.state.task_running = True
        self.state.run_stage = stage
        self.state.run_error = None
        return identity

    def _build_edit_snapshot(
        self, identity: EditRequestIdentity, *, run_stage: str
    ) -> EditSnapshot:
        session = self.state.editing_session
        assert session is not None
        draft = session.draft
        snap = session.snapshot()
        selected = self.state.selected_bout_id
        if selected and session.reviewed_bouts is not None:
            # Keep selection when still present in reviewed/source ids.
            reviewed = session.reviewed_bouts
            ids: set[str] = set()
            if "bout_id" in reviewed.columns:
                ids.update(str(v) for v in reviewed["bout_id"].tolist())
            if "source_s8_bout_id" in reviewed.columns:
                ids.update(
                    str(v)
                    for v in reviewed["source_s8_bout_id"].tolist()
                    if v is not None and str(v).strip()
                )
            if ids and selected not in ids:
                selected = None
        return EditSnapshot(
            identity=identity,
            review_run_id=draft.review_run_id if draft is not None else None,
            review_run_dir=draft.review_run_dir if draft is not None else None,
            draft_open=draft is not None,
            editing_active=bool(session.mode == MODE_EDITING),
            mode=str(session.mode),
            view_source=(
                VIEW_SOURCE_WORKING if draft is not None else self.state.view_source
            ),
            can_undo=bool(snap.get("can_undo")),
            can_redo=bool(snap.get("can_redo")),
            status_counts=_status_counts(session),
            changed_intervals_text=_format_changed_intervals(
                session.changed_intervals()
            ),
            selected_bout_id=selected,
            n_reviewed_bouts=(
                int(len(session.reviewed_bouts))
                if session.reviewed_bouts is not None
                else None
            ),
            run_stage=run_stage,
        )

    def _edit_identity_matches_current_view(self, identity: EditRequestIdentity) -> bool:
        if self.state.session_root is None or self.state.package_run_id is None:
            return False
        if identity.session_root.resolve(strict=False) != self.state.session_root.resolve(
            strict=False
        ):
            return False
        if identity.package_run_id != str(self.state.package_run_id):
            return False
        # create_draft starts with review_run_id=None; later ops must match draft.
        return (
            identity.op in {EDIT_OP_CREATE_DRAFT, EDIT_OP_ENTER_EDITING}
            or identity.review_run_id == self.state.review_run_id
        )

    def _finish_edit_request(self, *, success_stage: str | None) -> None:
        del success_stage
        self.state.edit_running = False
        self.state.task_running = bool(
            self.state.publish_running
            or self.state.open_running
            or self.state.discovery_running
        )
        self.state.active_edit_identity = None
        self._pending_edit = None

    def _clear_pending_edit_if_matches(self, identity: EditRequestIdentity) -> None:
        pending = self._pending_edit
        if pending is not None and pending.identity.request_id == identity.request_id:
            self._pending_edit = None
        if (
            self.state.active_edit_identity is not None
            and self.state.active_edit_identity.request_id == identity.request_id
        ):
            self.state.active_edit_identity = None
            self.state.edit_running = False
            self.state.task_running = bool(
                self.state.publish_running
                or self.state.open_running
                or self.state.discovery_running
            )

    def _selected_source_bout_id(self, *, required: bool = True) -> str:
        bout_id = self.state.selected_bout_id
        if not bout_id:
            if required:
                raise BehavioralReviewUiError("Select a bout first.")
            return ""
        for row in self.state.bout_rows:
            if row.get("bout_id") == bout_id:
                source = row.get("source_s8_bout_id") or bout_id
                return str(source)
        session = self.state.editing_session
        if session is not None:
            try:
                row = session.inspection.bouts.get(bout_id)
                source = row.get("source_s8_bout_id")
                if source is not None and str(source).strip():
                    return str(source)
            except Exception:  # noqa: BLE001
                pass
        return str(bout_id)

    def _apply_playhead(self, mapping: Any) -> None:
        self.state.sample_idx = int(mapping.sample_idx)
        self.state.time_sec = float(mapping.time_sec)
        self.state.prepared_frame_idx = int(mapping.prepared_frame_idx)
        self.state.prepared_time_sec = float(mapping.prepared_time_sec)

    def _refresh_edit_flags(self) -> None:
        session = self.state.editing_session
        if session is None or session.draft is None:
            self.state.can_undo = False
            self.state.can_redo = False
            return
        snap = session.snapshot()
        self.state.can_undo = bool(snap.get("can_undo"))
        self.state.can_redo = bool(snap.get("can_redo"))
        self.state.status_counts = _status_counts(session)
        self.state.changed_intervals_text = _format_changed_intervals(
            session.changed_intervals()
        )


def with_contextlib_suppress_close(session: EditingSession) -> None:
    try:
        if session.draft is not None:
            session.close_draft(release_lock=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        if session.inspection.video is not None:
            session.inspection.video.clear_cache()
    except Exception:  # noqa: BLE001
        pass


def _s9_option_from_summary(run: ResultsPackageRunSummary) -> S9PackageOption:
    from results_packaging.contracts import (
        COMPLETED_WITH_WARNINGS,
        ELIGIBLE_DOWNSTREAM,
    )

    classification = str(run.classification or "")
    eligible = classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
    status = run.status
    label = (
        f"{run.run_id} · {classification or status or 'unknown'}"
        f"{' · eligible' if eligible else ' · not eligible'}"
    )
    reasons = tuple(str(r) for r in (run.staleness_reasons or ()))
    return S9PackageOption(
        run_id=run.run_id,
        run_dir=Path(run.run_dir),
        historical_status=str(status) if status is not None else None,
        classification=classification,
        currently_eligible=eligible,
        n_bouts=None,
        n_dense_samples=None,
        label=label,
        reasons=reasons,
    )


def _review_option_from_summary(run: ReviewedRunSummary) -> ReviewRunOption:
    is_draft = run.classification == CLASS_DRAFT or str(
        run.publication_state or ""
    ).lower() in {"draft", "in_progress"}
    # Prefer draft classification; also treat missing science artifacts with working/.
    if run.classification == CLASS_DRAFT:
        is_draft = True
    label = (
        f"{run.review_run_id} · {run.classification}"
        f"{' · eligible' if run.currently_eligible else ''}"
        f"{' · superseded by ' + run.superseded_by if run.superseded_by else ''}"
    )
    return ReviewRunOption(
        run_id=run.review_run_id,
        run_dir=Path(run.run_dir),
        classification=run.classification,
        currently_eligible=bool(run.currently_eligible),
        parent_package_run_id=run.parent_package_run_id,
        publication_state=run.publication_state,
        historical_status=run.status,
        superseded_by=run.superseded_by,
        label=label,
        is_draft=is_draft,
    )


def _discover_directions(session: EditingSession) -> tuple[tuple[int, int], ...]:
    bouts = session.inspection.source.behavior_bouts
    if bouts is None or bouts.empty:
        return ((0, 1), (1, 0))
    exclusive = bouts[bouts["channel"] == CHANNEL_EXCLUSIVE]
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    if not exclusive.empty and "focal_id" in exclusive.columns:
        for _, row in exclusive.iterrows():
            if pd.isna(row.get("focal_id")) or pd.isna(row.get("partner_id")):
                continue
            pair = (int(row["focal_id"]), int(row["partner_id"]))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if not pairs:
        pairs = [(0, 1), (1, 0)]
    return tuple(pairs)


def _bout_table_rows(session: EditingSession) -> tuple[dict[str, Any], ...]:
    browser = session.inspection.bouts
    frame = browser.list_bouts()
    if frame is None or frame.empty:
        return ()
    # Prefer reviewed bouts when draft has regenerated table.
    if session.reviewed_bouts is not None and not session.reviewed_bouts.empty:
        # Display filter still comes from packaged browser; merge status overlay.
        pass
    statuses = session.current_statuses() if session.draft is not None else None
    status_map: dict[str, str] = {}
    if statuses is not None and not statuses.empty:
        current = statuses
        if "active" in current.columns:
            current = current[current["active"].fillna(False).astype(bool)]
        for _, row in current.iterrows():
            key = str(row.get("target_bout_id") or row.get("bout_id") or "")
            if key:
                status_map[key] = str(row.get("status") or STATUS_UNREVIEWED)
    # Cap rows for GUI responsiveness, but keep at least a few of each channel.
    selected = frame
    if len(frame) > 500 and "channel" in frame.columns:
        parts = []
        for channel, group in frame.groupby("channel", sort=False):
            del channel
            parts.append(group.head(80))
        selected = pd.concat(parts, ignore_index=True).head(500)
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        bout_id = str(row.get("bout_id") or "")
        source_id = str(row.get("source_s8_bout_id") or bout_id)
        status = status_map.get(bout_id) or status_map.get(source_id) or STATUS_UNREVIEWED
        rows.append(
            {
                "bout_id": bout_id,
                "source_s8_bout_id": source_id,
                "channel": str(row.get("channel") or ""),
                "behavior_name": str(row.get("behavior_name") or ""),
                "focal_id": row.get("focal_id"),
                "partner_id": row.get("partner_id"),
                "start_sample_idx": int(row.get("start_sample_idx") or 0),
                "end_sample_idx": int(row.get("end_sample_idx") or 0),
                "status": status,
            }
        )
    return tuple(rows)


def _status_counts(session: EditingSession) -> dict[str, int]:
    counts = {
        STATUS_ACCEPTED: 0,
        STATUS_REJECTED: 0,
        STATUS_UNREVIEWED: 0,
        STATUS_VOIDED: 0,
    }
    if session.draft is None:
        return counts
    statuses = session.current_statuses()
    if statuses is None or statuses.empty:
        n = (
            int(len(session.reviewed_bouts))
            if session.reviewed_bouts is not None
            else session.inspection.n_bouts
        )
        counts[STATUS_UNREVIEWED] = n
        return counts
    current = statuses
    if "active" in current.columns:
        current = current[current["active"].fillna(False).astype(bool)]
    for status in current["status"].tolist():
        key = str(status)
        counts[key] = counts.get(key, 0) + 1
    decided = counts.get(STATUS_ACCEPTED, 0) + counts.get(STATUS_REJECTED, 0)
    n_bouts = (
        int(len(session.reviewed_bouts))
        if session.reviewed_bouts is not None
        else session.inspection.n_bouts
    )
    counts[STATUS_UNREVIEWED] = max(0, n_bouts - decided - counts.get(STATUS_VOIDED, 0))
    return counts


def _format_changed_intervals(changed: Mapping[str, list[tuple[int, int]]]) -> str:
    if not changed:
        return "(no corrected intervals)"
    parts: list[str] = []
    for channel, spans in changed.items():
        for start, end in spans:
            parts.append(f"{channel}: [{start}…{end}]")
    return "; ".join(parts) if parts else "(no corrected intervals)"
