"""Subsystem 10 — Behavioral Review & Editing GUI page (Group 4 beside S9)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from behavioral_review import (
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    EXCLUSIVE_ETHOGRAM_LABELS,
    MODE_EDITING,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    BehavioralReviewError,
    EditingError,
    ReviewedPublicationResult,
)
from behavioral_review.contracts import REVIEW_TARGET_FPS
from ui.controllers.behavioral_review_controller import (
    BehavioralReviewController,
    BehavioralReviewUiError,
    DecodeRequestIdentity,
    DecodeSnapshot,
    DiscoveryRequestIdentity,
    DiscoverySnapshot,
    EditRequestIdentity,
    EditSnapshot,
    OpenSessionSnapshot,
    OpenSourceIdentity,
    PublishPreview,
    PublishRequestIdentity,
    WindowRequestIdentity,
    WindowSnapshot,
)
from ui.tasks import (
    GuiTaskContext,
    GuiTaskKind,
    GuiTaskRunner,
    TaskAlreadyRunningError,
)
from ui.widgets.ethogram_timeline_view import EthogramTimelineView
from ui.widgets.pose_video_view import PoseVideoView

_BOUT_COLUMNS = (
    "bout_id",
    "channel",
    "behavior_name",
    "focal_id",
    "partner_id",
    "start_sample_idx",
    "end_sample_idx",
    "status",
)

_PLAYBACK_TICK_MS = 50


class BehavioralReviewPage(QWidget):
    """Group 4 page for S10 inspection, optional editing, and publication."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: BehavioralReviewController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or BehavioralReviewController()
        self.discovery_runner = GuiTaskRunner(self)
        self.open_runner = GuiTaskRunner(self)
        self.decode_runner = GuiTaskRunner(self)
        self.window_runner = GuiTaskRunner(self)
        self.edit_runner = GuiTaskRunner(self)
        self.publish_runner = GuiTaskRunner(self)
        self._syncing_widgets = False
        self._last_running_state = False
        self._playing = False
        self._playback_speed = 1.0
        self._active_discovery_identity: DiscoveryRequestIdentity | None = None
        self._active_open_identity: OpenSourceIdentity | None = None
        self._active_decode_identity: DecodeRequestIdentity | None = None
        self._active_window_identity: WindowRequestIdentity | None = None
        self._active_edit_identity: EditRequestIdentity | None = None
        self._active_publish_identity: PublishRequestIdentity | None = None
        self._last_shown_prepared_frame_idx: int | None = None
        self._last_bout_table_fingerprint: tuple[object, ...] | None = None
        self._build_ui()
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(_PLAYBACK_TICK_MS)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self.refresh_from_state()

    @property
    def has_running_task(self) -> bool:
        state = self.controller.state
        return (
            self.discovery_runner.is_running
            or self.open_runner.is_running
            or self.decode_runner.is_running
            or self.window_runner.is_running
            or self.edit_runner.is_running
            or self.publish_runner.is_running
            or state.task_running
            or state.discovery_running
            or state.open_running
            or state.decode_running
            or state.window_running
            or state.edit_running
            or state.publish_running
        )

    def set_session(self, session_root: Path | str) -> None:
        self._stop_playback()
        try:
            identity = self.controller.set_session(session_root)
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot open session", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S10 session opened: {Path(session_root).name}; discovering…"
        )
        self._start_discovery_task(identity)

    def refresh_from_state(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            self.session_path.setText(
                str(state.session_root) if state.session_root is not None else ""
            )
            self._populate_s9_combo()
            self._populate_review_combo()
            self._sync_analysis_source_combo()
            self._sync_direction_combo()
            self._sync_channel_combo()
            self._sync_mode_banner()
            self.reviewer_edit.setText(state.reviewer_id)
            self._sync_clock_labels()
            self._sync_scrubber()
            self._populate_bout_table()
            self._apply_timeline_from_state()
            self.evidence_text.setPlainText(state.evidence_text)
            self.provenance_text.setPlainText(state.provenance_text)
            self.reconstruction_text.setPlainText(state.reconstruction_text)
            self.changed_intervals_label.setText(
                state.changed_intervals_text or "(no corrected intervals)"
            )
            self.compare_label.setText(state.compare_text or "")
            self._sync_publish_preview_labels(state.last_publish_preview)
            self.stage_label.setText(
                f"Failed: {state.run_error}" if state.run_error else state.run_stage
            )
            self.readiness_label.setText(self._readiness_text())
            if (
                state.frame_bgr is not None
                and state.prepared_frame_idx != self._last_shown_prepared_frame_idx
            ):
                self.video.set_frame(state.frame_bgr)
                self._last_shown_prepared_frame_idx = state.prepared_frame_idx
            elif state.editing_session is None:
                self.video.set_frame(None)
                self._last_shown_prepared_frame_idx = None
        finally:
            self._syncing_widgets = False
        self._update_enabled()
        self._emit_running_changed()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 10 — Behavioral Review & Editing")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Inspect an eligible Subsystem 9 results package with a dense (120 Hz) "
            "playhead and prepared-video dual clock. Inspection is the default; "
            "editing requires an explicit mode toggle and reviewer identity. "
            "No reciprocal edit tool, no uncertain status, and no reviewed CSV/media "
            "export in MVP. Corrections stay in the controller — this page only "
            "dispatches. Publishing creates an additive immutable reviewed run."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session")
        session_form = QFormLayout(session_group)
        self.session_path = QLineEdit()
        self.session_browse_button = QPushButton("Browse…")
        self.session_browse_button.clicked.connect(self._browse_session)
        self.open_session_button = QPushButton("Open session")
        self.open_session_button.clicked.connect(self._open_typed_session)
        self.refresh_discovery_button = QPushButton("Refresh discovery")
        self.refresh_discovery_button.clicked.connect(self._refresh_discovery)
        session_row = QHBoxLayout()
        session_row.addWidget(self.session_path, 1)
        session_row.addWidget(self.session_browse_button)
        session_row.addWidget(self.open_session_button)
        session_row.addWidget(self.refresh_discovery_button)
        self.readiness_label = QLabel("No session.")
        self.readiness_label.setWordWrap(True)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("Readiness", self.readiness_label)

        source_group = QGroupBox("S9 package / review drafts")
        source_form = QFormLayout(source_group)
        self.s9_combo = QComboBox()
        self.s9_combo.currentIndexChanged.connect(self._s9_selection_changed)
        self.open_automated_button = QPushButton("Open automated package")
        self.open_automated_button.clicked.connect(self._open_automated)
        self.review_combo = QComboBox()
        self.review_combo.currentIndexChanged.connect(self._review_selection_changed)
        self.resume_draft_button = QPushButton("Resume draft")
        self.resume_draft_button.clicked.connect(self._resume_draft)
        self.inspect_published_button = QPushButton("Inspect published")
        self.inspect_published_button.clicked.connect(self._inspect_published)
        self.analysis_source_combo = QComboBox()
        self.analysis_source_combo.addItem("Automated (S9)", SOURCE_TYPE_AUTOMATED)
        self.analysis_source_combo.addItem("Reviewed (S10)", SOURCE_TYPE_REVIEWED)
        self.analysis_source_combo.currentIndexChanged.connect(
            self._analysis_source_changed
        )
        source_form.addRow("S9 package", self.s9_combo)
        source_form.addRow(self.open_automated_button)
        source_form.addRow("Review drafts / runs", self.review_combo)
        review_buttons = QHBoxLayout()
        review_buttons.addWidget(self.resume_draft_button)
        review_buttons.addWidget(self.inspect_published_button)
        source_form.addRow(review_buttons)
        source_form.addRow("Analysis source (explicit)", self.analysis_source_combo)

        self.mode_banner = QLabel("MODE: INSPECTION")
        self.mode_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_banner.setStyleSheet(
            "font-size: 14px; font-weight: 700; padding: 6px; "
            "background: #37474f; color: #eceff1;"
        )

        editing_group = QGroupBox("Editing mode")
        editing_form = QFormLayout(editing_group)
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("reviewer identity (required for edits)")
        self.reviewer_edit.editingFinished.connect(self._reviewer_changed)
        self.enter_editing_button = QPushButton("Enter editing mode")
        self.enter_editing_button.clicked.connect(self._enter_editing)
        self.create_draft_button = QPushButton("Create draft")
        self.create_draft_button.clicked.connect(self._create_draft)
        self.exit_editing_button = QPushButton("Exit editing mode")
        self.exit_editing_button.clicked.connect(self._exit_editing)
        edit_buttons = QHBoxLayout()
        edit_buttons.addWidget(self.enter_editing_button)
        edit_buttons.addWidget(self.create_draft_button)
        edit_buttons.addWidget(self.exit_editing_button)
        editing_form.addRow("Reviewer identity", self.reviewer_edit)
        editing_form.addRow(edit_buttons)

        self.video = PoseVideoView()
        self.video.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.analysis_clock_label = QLabel("Analysis: sample — · t — s")
        self.prepared_clock_label = QLabel("Prepared: frame — · t — s")
        self.plateau_label = QLabel("Plateau: —")
        clock_row = QHBoxLayout()
        clock_row.addWidget(self.analysis_clock_label)
        clock_row.addWidget(self.prepared_clock_label)
        clock_row.addWidget(self.plateau_label)
        clock_row.addStretch(1)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        self.step_back_button = QPushButton("◀ −1")
        self.step_back_button.clicked.connect(lambda: self._step(-1))
        self.step_fwd_button = QPushButton("+1 ▶")
        self.step_fwd_button.clicked.connect(lambda: self._step(1))
        self.prev_bout_button = QPushButton("◀ Bout")
        self.prev_bout_button.clicked.connect(self._prev_bout)
        self.next_bout_button = QPushButton("Bout ▶")
        self.next_bout_button.clicked.connect(self._next_bout)
        self.jump_start_button = QPushButton("Bout start")
        self.jump_start_button.clicked.connect(self._jump_bout_start)
        self.jump_end_button = QPushButton("Bout end")
        self.jump_end_button.clicked.connect(self._jump_bout_end)
        self.speed_combo = QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        transport = QHBoxLayout()
        for widget in (
            self.play_button,
            self.step_back_button,
            self.step_fwd_button,
            self.prev_bout_button,
            self.next_bout_button,
            self.jump_start_button,
            self.jump_end_button,
        ):
            transport.addWidget(widget)
        transport.addWidget(QLabel("Speed"))
        transport.addWidget(self.speed_combo)
        transport.addStretch(1)

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.setMaximum(0)
        self.scrubber.valueChanged.connect(self._scrubber_changed)

        self.direction_combo = QComboBox()
        self.direction_combo.currentIndexChanged.connect(self._direction_changed)
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("(all channels)", None)
        self.channel_combo.addItem("exclusive", CHANNEL_EXCLUSIVE)
        self.channel_combo.addItem("contact", CHANNEL_CONTACT)
        self.channel_combo.addItem("side_by_side", CHANNEL_SIDE_BY_SIDE)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Direction"))
        nav_row.addWidget(self.direction_combo, 1)
        nav_row.addWidget(QLabel("Channel"))
        nav_row.addWidget(self.channel_combo, 1)

        viewer_group = QGroupBox("Video / playhead")
        viewer_layout = QVBoxLayout(viewer_group)
        viewer_layout.addWidget(self.video, 1)
        viewer_layout.addLayout(clock_row)
        viewer_layout.addWidget(self.scrubber)
        viewer_layout.addLayout(transport)
        viewer_layout.addLayout(nav_row)

        self.timeline = EthogramTimelineView()
        self.timeline.sample_seek_requested.connect(self._timeline_seek)
        timeline_group = QGroupBox("Ethogram timeline (dense)")
        timeline_layout = QVBoxLayout(timeline_group)
        timeline_layout.addWidget(self.timeline)

        bouts_group = QGroupBox("Bouts")
        bouts_layout = QVBoxLayout(bouts_group)
        filter_row = QHBoxLayout()
        self.bout_channel_filter = QComboBox()
        self.bout_channel_filter.addItem("(all channels)", None)
        self.bout_channel_filter.addItem("exclusive", CHANNEL_EXCLUSIVE)
        self.bout_channel_filter.addItem("contact", CHANNEL_CONTACT)
        self.bout_channel_filter.addItem("side_by_side", CHANNEL_SIDE_BY_SIDE)
        self.bout_channel_filter.currentIndexChanged.connect(self._bout_filters_changed)
        self.bout_behavior_filter = QLineEdit()
        self.bout_behavior_filter.setPlaceholderText("behavior filter (exact)")
        self.bout_behavior_filter.editingFinished.connect(self._bout_filters_changed)
        filter_row.addWidget(QLabel("Channel filter"))
        filter_row.addWidget(self.bout_channel_filter)
        filter_row.addWidget(QLabel("Behavior"))
        filter_row.addWidget(self.bout_behavior_filter, 1)
        self.bout_table = QTableWidget(0, len(_BOUT_COLUMNS))
        self.bout_table.setHorizontalHeaderLabels(list(_BOUT_COLUMNS))
        self.bout_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.bout_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.bout_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # Interactive (not ResizeToContents): content-sized columns on every
        # scrub rebuild were observed to dominate GUI-thread settle time.
        self.bout_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.bout_table.horizontalHeader().setStretchLastSection(True)
        self.bout_table.itemSelectionChanged.connect(self._bout_selection_changed)
        bouts_layout.addLayout(filter_row)
        bouts_layout.addWidget(self.bout_table)

        evidence_group = QGroupBox("Evidence / provenance / reconstruction")
        evidence_layout = QVBoxLayout(evidence_group)
        self.evidence_text = QTextEdit()
        self.evidence_text.setReadOnly(True)
        self.evidence_text.setMaximumHeight(120)
        self.provenance_text = QTextEdit()
        self.provenance_text.setReadOnly(True)
        self.provenance_text.setMaximumHeight(80)
        self.reconstruction_text = QTextEdit()
        self.reconstruction_text.setReadOnly(True)
        self.reconstruction_text.setMaximumHeight(60)
        evidence_layout.addWidget(QLabel("Evidence"))
        evidence_layout.addWidget(self.evidence_text)
        evidence_layout.addWidget(QLabel("Provenance"))
        evidence_layout.addWidget(self.provenance_text)
        evidence_layout.addWidget(QLabel("Reconstruction"))
        evidence_layout.addWidget(self.reconstruction_text)
        self.compare_label = QLabel("")
        self.compare_label.setWordWrap(True)
        evidence_layout.addWidget(self.compare_label)

        edits_group = QGroupBox("Edit controls (draft required)")
        edits_form = QFormLayout(edits_group)
        self.edit_start_spin = QSpinBox()
        self.edit_start_spin.setMaximum(10_000_000)
        self.edit_end_spin = QSpinBox()
        self.edit_end_spin.setMaximum(10_000_000)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start"))
        range_row.addWidget(self.edit_start_spin)
        range_row.addWidget(QLabel("End"))
        range_row.addWidget(self.edit_end_spin)
        self.use_playhead_start_button = QPushButton("Start = playhead")
        self.use_playhead_start_button.clicked.connect(self._use_playhead_as_start)
        self.use_playhead_end_button = QPushButton("End = playhead")
        self.use_playhead_end_button.clicked.connect(self._use_playhead_as_end)
        range_row.addWidget(self.use_playhead_start_button)
        range_row.addWidget(self.use_playhead_end_button)
        self.exclusive_label_combo = QComboBox()
        for label in EXCLUSIVE_ETHOGRAM_LABELS:
            self.exclusive_label_combo.addItem(label, label)
        self.apply_exclusive_button = QPushButton("Set exclusive label")
        self.apply_exclusive_button.clicked.connect(self._apply_exclusive_label)
        self.parallel_channel_combo = QComboBox()
        self.parallel_channel_combo.addItem("contact", CHANNEL_CONTACT)
        self.parallel_channel_combo.addItem("side_by_side", CHANNEL_SIDE_BY_SIDE)
        self.parallel_pred_spin = QSpinBox()
        self.parallel_pred_spin.setRange(0, 1)
        self.apply_parallel_button = QPushButton("Set contact/SBS pred")
        self.apply_parallel_button.clicked.connect(self._apply_parallel_pred)
        parallel_row = QHBoxLayout()
        parallel_row.addWidget(self.parallel_channel_combo)
        parallel_row.addWidget(QLabel("pred"))
        parallel_row.addWidget(self.parallel_pred_spin)
        parallel_row.addWidget(self.apply_parallel_button)
        self.trim_button = QPushButton("Trim selected bout")
        self.trim_button.clicked.connect(self._trim_bout)
        self.split_button = QPushButton("Split at playhead")
        self.split_button.clicked.connect(self._split_bout)
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(
            lambda: self._set_status(STATUS_ACCEPTED)
        )
        self.reject_button = QPushButton("Reject")
        self.reject_button.clicked.connect(
            lambda: self._set_status(STATUS_REJECTED)
        )
        self.unreviewed_button = QPushButton("Unreviewed")
        self.unreviewed_button.clicked.connect(
            lambda: self._set_status(STATUS_UNREVIEWED)
        )
        status_row = QHBoxLayout()
        status_row.addWidget(self.accept_button)
        status_row.addWidget(self.reject_button)
        status_row.addWidget(self.unreviewed_button)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("note text")
        self.add_note_button = QPushButton("Add note")
        self.add_note_button.clicked.connect(self._add_note)
        note_row = QHBoxLayout()
        note_row.addWidget(self.note_edit, 1)
        note_row.addWidget(self.add_note_button)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self._redo)
        undo_row = QHBoxLayout()
        undo_row.addWidget(self.undo_button)
        undo_row.addWidget(self.redo_button)
        self.changed_intervals_label = QLabel("(no corrected intervals)")
        self.changed_intervals_label.setWordWrap(True)
        edits_form.addRow("Interval", range_row)
        edits_form.addRow("Exclusive label", self.exclusive_label_combo)
        edits_form.addRow(self.apply_exclusive_button)
        edits_form.addRow("Parallel pred", parallel_row)
        edits_form.addRow(self.trim_button)
        edits_form.addRow(self.split_button)
        edits_form.addRow("Review status", status_row)
        edits_form.addRow("Note", note_row)
        edits_form.addRow(undo_row)
        edits_form.addRow("Changed intervals", self.changed_intervals_label)

        publish_group = QGroupBox("Publication")
        publish_layout = QVBoxLayout(publish_group)
        self.publish_preview_label = QLabel("No publish preview.")
        self.publish_preview_label.setWordWrap(True)
        self.publish_blockers_label = QLabel("")
        self.publish_blockers_label.setWordWrap(True)
        self.publish_blockers_label.setStyleSheet("color: #c62828;")
        self.publish_button = QPushButton("Publish Reviewed Results")
        self.publish_button.clicked.connect(self._publish)
        publish_layout.addWidget(self.publish_preview_label)
        publish_layout.addWidget(self.publish_blockers_label)
        publish_layout.addWidget(self.publish_button)

        self.stage_label = QLabel("Not started")
        self.stage_label.setWordWrap(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(viewer_group, 1)
        left_layout.addWidget(timeline_group)
        left_layout.addWidget(bouts_group)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(session_group)
        right_layout.addWidget(source_group)
        right_layout.addWidget(self.mode_banner)
        right_layout.addWidget(editing_group)
        right_layout.addWidget(evidence_group)
        right_layout.addWidget(edits_group)
        right_layout.addWidget(publish_group)
        right_layout.addWidget(self.stage_label)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(360)

        body = QHBoxLayout()
        body.addWidget(left, 3)
        body.addWidget(right_scroll, 2)

        root = QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(scope)
        root.addLayout(body, 1)

    # ----------------------------------------------------------- session I/O

    def _browse_session(self) -> None:
        if not self._guard_idle("browse session"):
            return
        path = QFileDialog.getExistingDirectory(self, "Select session / project root")
        if path:
            self.set_session(path)

    def _open_typed_session(self) -> None:
        if not self._guard_idle("open session"):
            return
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _refresh_discovery(self) -> None:
        if not self._guard_idle("refresh discovery"):
            return
        root = self.controller.state.session_root
        if root is None:
            QMessageBox.information(self, "No session", "Open a session first.")
            return
        self.set_session(root)

    def _start_discovery_task(self, identity: DiscoveryRequestIdentity) -> None:
        self._active_discovery_identity = identity
        self.stage_label.setText("Discovering S9 packages and review runs…")
        self._update_enabled()
        self._emit_running_changed()
        if self.discovery_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_REVIEW_DISCOVERY,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> DiscoverySnapshot:
            return self.controller.collect_discovery_snapshot(identity)

        try:
            self.discovery_runner.start(
                _operation,
                task_name="S10 discovery",
                context=context,
                on_success=lambda result: self._on_discovery_success(result, identity),
                on_error=lambda exc: self._on_discovery_error(exc, identity),
                on_finished=lambda: self._on_discovery_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_discovery_success(
        self, result: object, identity: DiscoveryRequestIdentity
    ) -> None:
        if not isinstance(result, DiscoverySnapshot):
            self.unexpected_error.emit(
                f"Unexpected S10 discovery result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_discovery_snapshot(result)
        if not accepted:
            self.status_message.emit(
                "Ignored stale S10 discovery result after session change."
            )
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S10 discovery complete for {identity.session_root.name}."
        )

    def _on_discovery_error(
        self, exc: BaseException, identity: DiscoveryRequestIdentity
    ) -> None:
        active = self.controller.state.active_discovery_identity
        if (
            active is None
            or active.request_id != identity.request_id
            or active.generation != identity.generation
        ):
            return
        # Controller has no fail_discovery helper — clear flags here.
        self.controller.state.discovery_running = False
        self.controller.state.active_discovery_identity = None
        self.controller.state.readiness_kind = "other"
        self.controller.state.readiness_error = str(exc)
        self.refresh_from_state()
        if isinstance(exc, BehavioralReviewUiError):
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(f"S10 discovery failed: {exc}")
        else:
            self.unexpected_error.emit(str(exc))

    def _on_discovery_finished(self, identity: DiscoveryRequestIdentity) -> None:
        latest = self.controller.state.active_discovery_identity
        if (
            self.controller.state.discovery_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            self._start_discovery_task(latest)
            return
        if (
            self._active_discovery_identity is not None
            and identity.request_id == self._active_discovery_identity.request_id
        ):
            self._active_discovery_identity = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    # ----------------------------------------------------------- open source

    def _open_automated(self) -> None:
        if not self._guard_idle("open automated package"):
            return
        self._stop_playback()
        try:
            identity = self.controller.begin_open_automated()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot open package", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_open_task(identity)

    def _resume_draft(self) -> None:
        if not self._guard_idle("resume draft"):
            return
        self._stop_playback()
        self.controller.set_reviewer_id(self.reviewer_edit.text())
        try:
            identity = self.controller.begin_open_draft()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot resume draft", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_open_task(identity)

    def _inspect_published(self) -> None:
        if not self._guard_idle("inspect published"):
            return
        self._stop_playback()
        try:
            identity = self.controller.begin_open_reviewed()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot open reviewed run", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_open_task(identity)

    def _start_open_task(self, identity: OpenSourceIdentity) -> None:
        self._active_open_identity = identity
        self.stage_label.setText(self.controller.state.run_stage)
        self._update_enabled()
        self._emit_running_changed()
        if self.open_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_REVIEW_OPEN,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        reviewer_id = self.reviewer_edit.text().strip() or None

        def _operation() -> OpenSessionSnapshot:
            return self.controller.open_source_worker(
                identity,
                reviewer_id=reviewer_id,
            )

        try:
            self.open_runner.start(
                _operation,
                task_name="S10 open source",
                context=context,
                on_success=lambda result: self._on_open_success(result, identity),
                on_error=lambda exc: self._on_open_error(exc, identity),
                on_finished=lambda: self._on_open_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_open_success(
        self, result: object, identity: OpenSourceIdentity
    ) -> None:
        if not isinstance(result, OpenSessionSnapshot):
            self.unexpected_error.emit(
                f"Unexpected S10 open result type: {type(result)!r}"
            )
            return
        reviewer_id = self.reviewer_edit.text().strip()
        accepted = self.controller.apply_open_snapshot(
            result,
            reviewer_id=reviewer_id or None,
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale S10 open result after session/source change."
            )
            return
        self._last_shown_prepared_frame_idx = None
        self._last_bout_table_fingerprint = None
        self.refresh_from_state()
        if self.controller.state.run_error:
            QMessageBox.warning(self, "Open incomplete", self.controller.state.run_error)
            self.status_message.emit(self.controller.state.run_error)
        else:
            self.status_message.emit(self.controller.state.run_stage)
            self._request_seek_workers()

    def _on_open_error(
        self, exc: BaseException, identity: OpenSourceIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.fail_open(identity, message)
        if not accepted:
            return
        self.refresh_from_state()
        if isinstance(exc, BehavioralReviewUiError):
            QMessageBox.warning(self, "Open failed", message)
            self.status_message.emit(f"S10 open failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_open_finished(self, identity: OpenSourceIdentity) -> None:
        latest = self.controller.state.active_open_identity
        if (
            self.controller.state.open_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            self._start_open_task(latest)
            return
        if (
            self._active_open_identity is not None
            and identity.request_id == self._active_open_identity.request_id
        ):
            self._active_open_identity = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    # ----------------------------------------------------------- seek / decode

    def _request_seek_workers(self) -> None:
        """Start decode + window workers for the current playhead (stale-safe)."""
        decode_id = self.controller.begin_decode_request()
        if decode_id is not None:
            self._start_decode_task(decode_id)
        if (
            self.controller.state.editing_session is not None
            and not self.controller.state.edit_running
        ):
            try:
                window_id = self.controller.begin_window_request()
            except BehavioralReviewUiError:
                return
            self._start_window_task(window_id)

    def _seek_sample(self, sample_idx: int) -> None:
        try:
            decode_id = self.controller.seek_sample(int(sample_idx))
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._sync_clock_labels()
        self._sync_scrubber()
        if decode_id is not None:
            self._start_decode_task(decode_id)
        if (
            self.controller.state.editing_session is not None
            and not self.controller.state.edit_running
        ):
            try:
                window_id = self.controller.begin_window_request()
            except BehavioralReviewUiError:
                return
            self._start_window_task(window_id)

    def _apply_decode_identity(
        self, decode_id: DecodeRequestIdentity | None
    ) -> None:
        self._sync_clock_labels()
        self._sync_scrubber()
        if decode_id is not None:
            self._start_decode_task(decode_id)
        if (
            self.controller.state.editing_session is not None
            and not self.controller.state.edit_running
        ):
            try:
                window_id = self.controller.begin_window_request()
            except BehavioralReviewUiError:
                return
            self._start_window_task(window_id)

    def _start_decode_task(self, identity: DecodeRequestIdentity) -> None:
        self._active_decode_identity = identity
        self._emit_running_changed()
        if self.decode_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_REVIEW_DECODE,
            project_dir=self.controller.state.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> DecodeSnapshot:
            return self.controller.decode_worker(identity)

        try:
            self.decode_runner.start(
                _operation,
                task_name="S10 decode",
                context=context,
                on_success=lambda result: self._on_decode_success(result, identity),
                on_error=lambda exc: self._on_decode_error(exc, identity),
                on_finished=lambda: self._on_decode_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_decode_success(
        self, result: object, identity: DecodeRequestIdentity
    ) -> None:
        if not isinstance(result, DecodeSnapshot):
            self.unexpected_error.emit(
                f"Unexpected S10 decode result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_decode_snapshot(result)
        if not accepted:
            return
        # Keep pixmap on plateau; only push when prepared frame changed.
        if (
            not result.prepared_frame_unchanged
            and result.frame_bgr is not None
            and result.identity.prepared_frame_idx
            != self._last_shown_prepared_frame_idx
        ) or (
            result.frame_bgr is not None
            and self._last_shown_prepared_frame_idx is None
        ):
            self.video.set_frame(result.frame_bgr)
            self._last_shown_prepared_frame_idx = result.identity.prepared_frame_idx
        self._sync_clock_labels()

    def _on_decode_error(
        self, exc: BaseException, identity: DecodeRequestIdentity
    ) -> None:
        active = self.controller.state.active_decode_identity
        if (
            active is None
            or active.request_id != identity.request_id
            or active.generation != identity.generation
        ):
            return
        self.controller.state.decode_running = False
        if active.generation == self.controller.state.decode_generation:
            self.controller.state.active_decode_identity = None
        self.status_message.emit(f"Decode failed: {exc}")

    def _on_decode_finished(self, identity: DecodeRequestIdentity) -> None:
        latest = self.controller.state.active_decode_identity
        if (
            self.controller.state.decode_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            self._start_decode_task(latest)
            return
        if (
            self._active_decode_identity is not None
            and identity.request_id == self._active_decode_identity.request_id
        ):
            self._active_decode_identity = None
        self._emit_running_changed()
        self.task_finished.emit()

    def _start_window_task(self, identity: WindowRequestIdentity) -> None:
        self._active_window_identity = identity
        self._emit_running_changed()
        if self.window_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_REVIEW_WINDOW,
            project_dir=self.controller.state.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> WindowSnapshot:
            return self.controller.window_worker(identity)

        try:
            self.window_runner.start(
                _operation,
                task_name="S10 window",
                context=context,
                on_success=lambda result: self._on_window_success(result, identity),
                on_error=lambda exc: self._on_window_error(exc, identity),
                on_finished=lambda: self._on_window_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_window_success(
        self, result: object, identity: WindowRequestIdentity
    ) -> None:
        if not isinstance(result, WindowSnapshot):
            self.unexpected_error.emit(
                f"Unexpected S10 window result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_window_snapshot(result)
        if not accepted:
            return
        self._syncing_widgets = True
        try:
            self._apply_timeline_from_state()
            # Bout list is filter/draft-driven, not playhead-driven; skip
            # redundant QTable rebuilds during scrub/seek window updates.
            self._populate_bout_table(only_if_changed=True)
            self.evidence_text.setPlainText(self.controller.state.evidence_text)
            self.provenance_text.setPlainText(self.controller.state.provenance_text)
            self.reconstruction_text.setPlainText(
                self.controller.state.reconstruction_text
            )
            self.changed_intervals_label.setText(
                self.controller.state.changed_intervals_text
                or "(no corrected intervals)"
            )
            self.compare_label.setText(self.controller.state.compare_text or "")
        finally:
            self._syncing_widgets = False
        self._update_enabled()

    def _on_window_error(
        self, exc: BaseException, identity: WindowRequestIdentity
    ) -> None:
        active = self.controller.state.active_window_identity
        if (
            active is None
            or active.request_id != identity.request_id
            or active.generation != identity.generation
        ):
            return
        self.controller.state.window_running = False
        if active.generation == self.controller.state.window_generation:
            self.controller.state.active_window_identity = None
        self.status_message.emit(f"Window load failed: {exc}")

    def _on_window_finished(self, identity: WindowRequestIdentity) -> None:
        latest = self.controller.state.active_window_identity
        if (
            self.controller.state.window_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            self._start_window_task(latest)
            return
        if (
            self._active_window_identity is not None
            and identity.request_id == self._active_window_identity.request_id
        ):
            self._active_window_identity = None
        self._emit_running_changed()
        self.task_finished.emit()

    # ----------------------------------------------------------- playback

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self.controller.state.editing_session is None:
            return
        if self.controller.state.n_dense <= 0:
            return
        self._playing = True
        self.play_button.setText("Pause")
        self._playback_timer.start()

    def _stop_playback(self) -> None:
        self._playing = False
        self.play_button.setText("Play")
        self._playback_timer.stop()

    def _on_playback_tick(self) -> None:
        if not self._playing:
            return
        state = self.controller.state
        if state.editing_session is None or state.n_dense <= 0:
            self._stop_playback()
            return
        # Advance dense sample_idx from wall clock rate × REVIEW_TARGET_FPS × speed.
        step = max(
            1,
            int(round(REVIEW_TARGET_FPS * (_PLAYBACK_TICK_MS / 1000.0) * self._playback_speed)),
        )
        next_idx = int(state.sample_idx) + step
        if next_idx >= state.n_dense:
            self._seek_sample(state.n_dense - 1)
            self._stop_playback()
            return
        self._seek_sample(next_idx)

    def _speed_changed(self) -> None:
        if self._syncing_widgets:
            return
        data = self.speed_combo.currentData()
        if data is not None:
            self._playback_speed = float(data)

    def _step(self, delta: int) -> None:
        self._stop_playback()
        try:
            decode_id = self.controller.step(delta)
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._apply_decode_identity(decode_id)

    def _prev_bout(self) -> None:
        self._stop_playback()
        try:
            decode_id = self.controller.previous_bout()
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        if decode_id is None:
            self.status_message.emit("No previous bout.")
            return
        self._apply_decode_identity(decode_id)

    def _next_bout(self) -> None:
        self._stop_playback()
        try:
            decode_id = self.controller.next_bout()
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        if decode_id is None:
            self.status_message.emit("No next bout.")
            return
        self._apply_decode_identity(decode_id)

    def _jump_bout_start(self) -> None:
        self._stop_playback()
        try:
            decode_id = self.controller.jump_bout_start()
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._apply_decode_identity(decode_id)

    def _jump_bout_end(self) -> None:
        self._stop_playback()
        try:
            decode_id = self.controller.jump_bout_end()
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._apply_decode_identity(decode_id)

    def _scrubber_changed(self, value: int) -> None:
        if self._syncing_widgets:
            return
        self._stop_playback()
        self._seek_sample(int(value))

    def _timeline_seek(self, sample_idx: int) -> None:
        self._stop_playback()
        self._seek_sample(int(sample_idx))

    # ----------------------------------------------------------- selections

    def _s9_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        run_id = self.s9_combo.currentData()
        self.controller.select_s9_package(
            str(run_id) if run_id is not None else None
        )
        self._update_enabled()

    def _review_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        run_id = self.review_combo.currentData()
        self.controller.select_review_run(
            str(run_id) if run_id is not None else None
        )
        self._update_enabled()

    def _analysis_source_changed(self) -> None:
        if self._syncing_widgets:
            return
        kind = self.analysis_source_combo.currentData()
        if kind is None:
            return
        try:
            self.controller.set_analysis_source_kind(str(kind))
        except Exception as exc:  # noqa: BLE001
            self.status_message.emit(str(exc))
            self.refresh_from_state()

    def _direction_changed(self) -> None:
        if self._syncing_widgets:
            return
        data = self.direction_combo.currentData()
        if data is None:
            return
        focal, partner = data
        try:
            self.controller.select_direction(int(focal), int(partner))
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._request_seek_workers()

    def _channel_changed(self) -> None:
        if self._syncing_widgets:
            return
        channel = self.channel_combo.currentData()
        try:
            self.controller.select_channel(
                str(channel) if channel is not None else None
            )
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._request_seek_workers()

    def _bout_filters_changed(self) -> None:
        if self._syncing_widgets:
            return
        channel = self.bout_channel_filter.currentData()
        behavior = self.bout_behavior_filter.text().strip() or None
        try:
            self.controller.set_bout_filters(
                channel=str(channel) if channel is not None else None,
                behavior_name=behavior,
            )
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._request_seek_workers()

    def _bout_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        rows = self.bout_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        item = self.bout_table.item(row, 0)
        if item is None:
            return
        bout_id = item.text()
        self._stop_playback()
        try:
            decode_id = self.controller.select_bout(bout_id)
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        # Prefill trim spins from selected row.
        start_item = self.bout_table.item(row, 5)
        end_item = self.bout_table.item(row, 6)
        if start_item is not None and end_item is not None:
            with _signals_blocked(self.edit_start_spin, self.edit_end_spin):
                self.edit_start_spin.setValue(int(start_item.text() or 0))
                self.edit_end_spin.setValue(int(end_item.text() or 0))
        self._apply_decode_identity(decode_id)

    # ----------------------------------------------------------- editing

    def _reviewer_changed(self) -> None:
        if self._syncing_widgets:
            return
        self.controller.set_reviewer_id(self.reviewer_edit.text())

    def _enter_editing(self) -> None:
        if not self._guard_edit_idle("enter editing"):
            return
        self.controller.set_reviewer_id(self.reviewer_edit.text())
        try:
            identity = self.controller.begin_enter_editing()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot enter editing", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _create_draft(self) -> None:
        if not self._guard_edit_idle("create draft"):
            return
        try:
            identity = self.controller.begin_create_draft()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot create draft", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _exit_editing(self) -> None:
        if not self._guard_edit_idle("exit editing"):
            return
        try:
            self.controller.exit_editing_mode(release_lock=False)
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot exit editing", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()

    def _use_playhead_as_start(self) -> None:
        self.edit_start_spin.setValue(int(self.controller.state.sample_idx))

    def _use_playhead_as_end(self) -> None:
        self.edit_end_spin.setValue(int(self.controller.state.sample_idx))

    def _apply_exclusive_label(self) -> None:
        if not self._guard_edit_idle("apply exclusive label"):
            return
        label = self.exclusive_label_combo.currentData()
        try:
            identity = self.controller.begin_set_exclusive_label(
                start_sample_idx=int(self.edit_start_spin.value()),
                end_sample_idx=int(self.edit_end_spin.value()),
                label=str(label),
            )
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Exclusive label failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _apply_parallel_pred(self) -> None:
        if not self._guard_edit_idle("apply contact/SBS"):
            return
        channel = self.parallel_channel_combo.currentData()
        try:
            identity = self.controller.begin_set_parallel_pred(
                channel=str(channel),
                start_sample_idx=int(self.edit_start_spin.value()),
                end_sample_idx=int(self.edit_end_spin.value()),
                pred_value=int(self.parallel_pred_spin.value()),
            )
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Parallel pred failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _trim_bout(self) -> None:
        if not self._guard_edit_idle("trim bout"):
            return
        try:
            identity = self.controller.begin_trim_selected_bout(
                new_start_sample_idx=int(self.edit_start_spin.value()),
                new_end_sample_idx=int(self.edit_end_spin.value()),
            )
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Trim failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _split_bout(self) -> None:
        if not self._guard_edit_idle("split bout"):
            return
        try:
            identity = self.controller.begin_split_selected_bout(
                at_sample_idx=int(self.controller.state.sample_idx)
            )
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Split failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _set_status(self, status: str) -> None:
        if not self._guard_edit_idle("update status"):
            return
        try:
            identity = self.controller.begin_set_bout_status(status)
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Status update failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _add_note(self) -> None:
        if not self._guard_edit_idle("add note"):
            return
        text = self.note_edit.text().strip()
        if not text:
            QMessageBox.information(self, "Note required", "Enter note text.")
            return
        try:
            identity = self.controller.begin_add_note(text)
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Note failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.note_edit.clear()
        self._start_edit_task(identity)

    def _undo(self) -> None:
        if not self._guard_edit_idle("undo"):
            return
        try:
            identity = self.controller.begin_undo()
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _redo(self) -> None:
        if not self._guard_edit_idle("redo"):
            return
        try:
            identity = self.controller.begin_redo()
        except BehavioralReviewUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._start_edit_task(identity)

    def _start_edit_task(self, identity: EditRequestIdentity) -> None:
        self._active_edit_identity = identity
        self.stage_label.setText(self.controller.state.run_stage)
        self._update_enabled()
        self._emit_running_changed()
        if self.edit_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_REVIEW_EDIT,
            project_dir=self.controller.state.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> EditSnapshot:
            return self.controller.edit_worker(identity)

        try:
            self.edit_runner.start(
                _operation,
                task_name=f"S10 edit ({identity.op})",
                context=context,
                on_success=lambda result: self._on_edit_success(result, identity),
                on_error=lambda exc: self._on_edit_error(exc, identity),
                on_finished=lambda: self._on_edit_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.status_message.emit("An edit task is already running.")
            self.controller.fail_edit(identity, "Edit runner already busy.")
            self.refresh_from_state()

    def _on_edit_success(self, result: object, identity: EditRequestIdentity) -> None:
        if not isinstance(result, EditSnapshot):
            self.unexpected_error.emit(
                f"Unexpected S10 edit result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_edit_snapshot(result)
        if not accepted:
            # Committed journal work is durable; refresh if the same draft is still open.
            self.controller.refresh_after_stale_edit()
            self.status_message.emit(
                "Ignored stale S10 edit UI callback; draft journal remains committed."
            )
            self.refresh_from_state()
            return
        self._last_bout_table_fingerprint = None
        self.refresh_from_state()
        self.status_message.emit(self.controller.state.run_stage)
        self._request_seek_workers()

    def _on_edit_error(
        self, exc: BaseException, identity: EditRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.fail_edit(identity, message)
        if not accepted:
            return
        # Failure before journal commit leaves prior authoritative state intact.
        self.controller.refresh_after_stale_edit()
        self.refresh_from_state()
        if isinstance(
            exc, (BehavioralReviewUiError, EditingError, BehavioralReviewError)
        ):
            QMessageBox.warning(self, "Edit failed", message)
            self.status_message.emit(f"S10 edit failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_edit_finished(self, identity: EditRequestIdentity) -> None:
        if (
            self._active_edit_identity is not None
            and identity.request_id == self._active_edit_identity.request_id
        ):
            self._active_edit_identity = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    # ----------------------------------------------------------- publish

    def _publish(self) -> None:
        if not self._guard_idle("publish"):
            return
        self._stop_playback()
        try:
            preview = self.controller.build_publish_preview()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot publish", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._sync_publish_preview_labels(preview)
        counts = preview.status_counts
        count_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        blockers = "\n".join(f"- {b}" for b in preview.blockers) or "(none)"
        warnings = "\n".join(f"- {w}" for w in preview.warnings) or "(none)"
        body = (
            f"Review run: {preview.review_run_id}\n"
            f"Parent package: {preview.parent_package_run_id}\n"
            f"Reviewer: {preview.reviewer_id}\n"
            f"Status counts: {count_text}\n"
            f"Corrected intervals: {preview.corrected_intervals_text}\n\n"
            f"Blockers:\n{blockers}\n\n"
            f"Warnings:\n{warnings}"
        )
        if preview.blockers:
            QMessageBox.warning(
                self,
                "Publication blocked",
                body + "\n\nResolve blockers before publishing.",
            )
            self.status_message.emit("Publication blocked by QC.")
            return
        confirm = QMessageBox.question(
            self,
            "Publish Reviewed Results",
            body + "\n\nPublish immutable reviewed results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            identity = self.controller.begin_publish()
        except BehavioralReviewUiError as exc:
            QMessageBox.warning(self, "Cannot publish", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._start_publish_task(identity)

    def _start_publish_task(self, identity: PublishRequestIdentity) -> None:
        self._active_publish_identity = identity
        self.stage_label.setText(self.controller.state.run_stage)
        self._update_enabled()
        self._emit_running_changed()
        if self.publish_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_REVIEW_PUBLISH,
            project_dir=self.controller.state.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> ReviewedPublicationResult:
            return self.controller.publish_worker(identity)

        try:
            self.publish_runner.start(
                _operation,
                task_name="S10 publish",
                context=context,
                on_success=lambda result: self._on_publish_success(result, identity),
                on_error=lambda exc: self._on_publish_error(exc, identity),
                on_finished=lambda: self._on_publish_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_publish_success(
        self, result: object, identity: PublishRequestIdentity
    ) -> None:
        if not isinstance(result, ReviewedPublicationResult):
            self.unexpected_error.emit(
                f"Unexpected S10 publish result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_publish_result(identity, result)
        if not accepted:
            self.status_message.emit("Ignored stale S10 publish result.")
            return
        self.refresh_from_state()
        if result.success:
            self.status_message.emit(self.controller.state.run_stage)
            QMessageBox.information(
                self,
                "Published",
                f"Published {result.review_run_id}\n"
                f"eligible={result.currently_eligible}",
            )
            root = self.controller.state.session_root
            if root is not None:
                try:
                    discovery = self.controller.prepare_session(root)
                except BehavioralReviewUiError:
                    return
                self._start_discovery_task(discovery)
        else:
            QMessageBox.warning(
                self,
                "Publication failed",
                result.error_message or "Publication failed",
            )

    def _on_publish_error(
        self, exc: BaseException, identity: PublishRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.fail_publish(identity, message)
        if not accepted:
            return
        self.refresh_from_state()
        if isinstance(exc, BehavioralReviewUiError):
            QMessageBox.warning(self, "Publication failed", message)
            self.status_message.emit(f"S10 publish failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_publish_finished(self, identity: PublishRequestIdentity) -> None:
        if (
            self._active_publish_identity is not None
            and identity.request_id == self._active_publish_identity.request_id
        ):
            self._active_publish_identity = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    # ----------------------------------------------------------- sync helpers

    def _populate_s9_combo(self) -> None:
        state = self.controller.state
        self.s9_combo.blockSignals(True)
        self.s9_combo.clear()
        self.s9_combo.addItem("(none)", None)
        selected = 0
        for i, opt in enumerate(state.s9_options, start=1):
            self.s9_combo.addItem(opt.label, opt.run_id)
            if opt.run_id == state.selected_s9_run_id:
                selected = i
        self.s9_combo.setCurrentIndex(selected)
        self.s9_combo.blockSignals(False)

    def _populate_review_combo(self) -> None:
        state = self.controller.state
        self.review_combo.blockSignals(True)
        self.review_combo.clear()
        self.review_combo.addItem("(none)", None)
        selected = 0
        for i, opt in enumerate(state.review_options, start=1):
            self.review_combo.addItem(opt.label, opt.run_id)
            if opt.run_id == state.selected_review_run_id:
                selected = i
        self.review_combo.setCurrentIndex(selected)
        self.review_combo.blockSignals(False)

    def _sync_analysis_source_combo(self) -> None:
        kind = self.controller.state.analysis_source_kind
        idx = self.analysis_source_combo.findData(kind)
        self.analysis_source_combo.blockSignals(True)
        self.analysis_source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.analysis_source_combo.blockSignals(False)

    def _sync_direction_combo(self) -> None:
        state = self.controller.state
        self.direction_combo.blockSignals(True)
        self.direction_combo.clear()
        selected = 0
        for i, (focal, partner) in enumerate(state.directions):
            self.direction_combo.addItem(f"{focal} → {partner}", (focal, partner))
            if (
                state.selected_focal_id == focal
                and state.selected_partner_id == partner
            ):
                selected = i
        if state.directions:
            self.direction_combo.setCurrentIndex(selected)
        self.direction_combo.blockSignals(False)

    def _sync_channel_combo(self) -> None:
        channel = self.controller.state.selected_channel
        idx = self.channel_combo.findData(channel)
        self.channel_combo.blockSignals(True)
        self.channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.channel_combo.blockSignals(False)

    def _sync_mode_banner(self) -> None:
        state = self.controller.state
        if state.edit_running or self.edit_runner.is_running:
            op = (
                state.active_edit_identity.op
                if state.active_edit_identity is not None
                else "edit"
            )
            self.mode_banner.setText(f"MODE: EDITING… ({op})")
            self.mode_banner.setStyleSheet(
                "font-size: 14px; font-weight: 700; padding: 6px; "
                "background: #e65100; color: #fff8e1;"
            )
        elif state.editing_active or state.mode == MODE_EDITING:
            self.mode_banner.setText("MODE: EDITING ACTIVE")
            self.mode_banner.setStyleSheet(
                "font-size: 14px; font-weight: 700; padding: 6px; "
                "background: #bf360c; color: #fff3e0;"
            )
        else:
            label = "MODE: INSPECTION"
            if state.inspected_published_only:
                label += " (published reviewed — read-only)"
            self.mode_banner.setText(label)
            self.mode_banner.setStyleSheet(
                "font-size: 14px; font-weight: 700; padding: 6px; "
                "background: #37474f; color: #eceff1;"
            )

    def _sync_clock_labels(self) -> None:
        state = self.controller.state
        self.analysis_clock_label.setText(
            f"Analysis: sample {state.sample_idx} · t {state.time_sec:.3f} s"
        )
        self.prepared_clock_label.setText(
            f"Prepared: frame {state.prepared_frame_idx} · "
            f"t {state.prepared_time_sec:.3f} s"
        )
        plateau = "yes" if state.plateau else "no"
        unchanged = "yes" if state.prepared_frame_unchanged else "no"
        self.plateau_label.setText(
            f"Plateau: {plateau} · prepared unchanged: {unchanged}"
        )

    def _sync_scrubber(self) -> None:
        state = self.controller.state
        self.scrubber.blockSignals(True)
        maximum = max(0, int(state.n_dense) - 1)
        self.scrubber.setMaximum(maximum)
        self.scrubber.setValue(min(int(state.sample_idx), maximum))
        self.scrubber.blockSignals(False)

    def _apply_timeline_from_state(self) -> None:
        timeline = self.controller.state.timeline
        if timeline is None:
            self.timeline.clear()
            return
        self.timeline.set_timeline(
            start_sample_idx=timeline.start_sample_idx,
            end_sample_idx=timeline.end_sample_idx,
            playhead_sample_idx=self.controller.state.sample_idx,
            lanes=timeline.lanes,
            segments=timeline.segments,
            lod_level=timeline.lod_level,
        )

    def _bout_table_fingerprint(
        self, rows: tuple[dict[str, object], ...]
    ) -> tuple[object, ...]:
        """Stable identity for bout table contents (excludes playhead)."""

        return tuple(
            (
                row.get("bout_id"),
                row.get("channel"),
                row.get("behavior_name"),
                row.get("start_sample_idx"),
                row.get("end_sample_idx"),
                row.get("status"),
                row.get("focal_id"),
                row.get("partner_id"),
            )
            for row in rows
        )

    def _populate_bout_table(self, *, only_if_changed: bool = False) -> None:
        rows = self.controller.state.bout_rows
        fingerprint = self._bout_table_fingerprint(rows)
        selected_id = self.controller.state.selected_bout_id
        if only_if_changed and fingerprint == self._last_bout_table_fingerprint:
            # Still sync selection highlight after bout-jump seeks.
            if selected_id is not None:
                self.bout_table.blockSignals(True)
                for r in range(self.bout_table.rowCount()):
                    item = self.bout_table.item(r, 0)
                    if item is not None and item.text() == selected_id:
                        self.bout_table.selectRow(r)
                        break
                self.bout_table.blockSignals(False)
            return
        self.bout_table.blockSignals(True)
        self.bout_table.setRowCount(len(rows))
        select_row = -1
        for r, row in enumerate(rows):
            values = [
                str(row.get(col, "") if row.get(col) is not None else "")
                for col in _BOUT_COLUMNS
            ]
            for c, value in enumerate(values):
                self.bout_table.setItem(r, c, QTableWidgetItem(value))
            if selected_id is not None and row.get("bout_id") == selected_id:
                select_row = r
        if select_row >= 0:
            self.bout_table.selectRow(select_row)
        self.bout_table.blockSignals(False)
        self._last_bout_table_fingerprint = fingerprint

    def _sync_publish_preview_labels(self, preview: PublishPreview | None) -> None:
        if preview is None:
            self.publish_preview_label.setText("No publish preview.")
            self.publish_blockers_label.setText("")
            return
        counts = ", ".join(
            f"{k}={v}" for k, v in sorted(preview.status_counts.items())
        )
        self.publish_preview_label.setText(
            f"{preview.review_run_id} · parent {preview.parent_package_run_id} · "
            f"reviewer {preview.reviewer_id} · {counts}\n"
            f"Intervals: {preview.corrected_intervals_text}"
        )
        if preview.blockers:
            self.publish_blockers_label.setText(
                "Blockers: " + "; ".join(preview.blockers)
            )
        elif preview.warnings:
            self.publish_blockers_label.setText(
                "Warnings: " + "; ".join(preview.warnings)
            )
            self.publish_blockers_label.setStyleSheet("color: #ef6c00;")
        else:
            self.publish_blockers_label.setText("No blockers.")
            self.publish_blockers_label.setStyleSheet("color: #2e7d32;")

    def _readiness_text(self) -> str:
        state = self.controller.state
        parts = [state.readiness_kind]
        if state.readiness_error:
            parts.append(state.readiness_error)
        if state.readiness_warnings:
            parts.append("; ".join(state.readiness_warnings[:3]))
        if state.package_run_id:
            parts.append(f"open={state.package_run_id}")
        if state.review_run_id:
            parts.append(f"draft={state.review_run_id}")
        return " · ".join(parts)

    def _update_enabled(self) -> None:
        busy = self.has_running_task
        state = self.controller.state
        has_session = state.session_root is not None
        has_package = state.editing_session is not None
        writable = bool(state.editing_active and state.draft_open)
        inspecting_published = bool(state.inspected_published_only)
        mutating = bool(
            state.edit_running
            or self.edit_runner.is_running
            or state.publish_running
            or self.publish_runner.is_running
        )
        # Keep scrub/play usable during edits; block source/draft/publish conflicts.
        source_locked = mutating or state.open_running or self.open_runner.is_running

        self.session_browse_button.setEnabled(not source_locked and not busy)
        self.open_session_button.setEnabled(not source_locked and not busy)
        self.refresh_discovery_button.setEnabled(
            not source_locked and not busy and has_session
        )
        self.s9_combo.setEnabled(not source_locked and not busy and has_session)
        self.review_combo.setEnabled(not source_locked and not busy and has_session)
        self.open_automated_button.setEnabled(
            not source_locked and not busy and has_session
        )
        self.resume_draft_button.setEnabled(
            not source_locked and not busy and has_session
        )
        self.inspect_published_button.setEnabled(
            not source_locked and not busy and has_session
        )
        self.analysis_source_combo.setEnabled(not source_locked and not busy)

        self.enter_editing_button.setEnabled(
            not mutating and has_package and not inspecting_published
        )
        self.create_draft_button.setEnabled(
            not mutating
            and has_package
            and state.editing_active
            and not state.draft_open
        )
        self.exit_editing_button.setEnabled(not mutating and state.editing_active)

        for widget in (
            self.play_button,
            self.step_back_button,
            self.step_fwd_button,
            self.prev_bout_button,
            self.next_bout_button,
            self.jump_start_button,
            self.jump_end_button,
            self.scrubber,
            self.direction_combo,
            self.channel_combo,
            self.bout_channel_filter,
            self.bout_behavior_filter,
            self.bout_table,
        ):
            widget.setEnabled(has_package and not self.publish_runner.is_running)

        for widget in (
            self.apply_exclusive_button,
            self.apply_parallel_button,
            self.trim_button,
            self.split_button,
            self.accept_button,
            self.reject_button,
            self.unreviewed_button,
            self.add_note_button,
            self.undo_button,
            self.redo_button,
        ):
            widget.setEnabled(writable and not mutating)

        self.undo_button.setEnabled(writable and not mutating and state.can_undo)
        self.redo_button.setEnabled(writable and not mutating and state.can_redo)
        self.publish_button.setEnabled(writable and not mutating)

    def _guard_idle(self, action: str) -> bool:
        if self.controller.state.edit_running or self.edit_runner.is_running:
            self.status_message.emit(
                f"Cannot {action} while an edit is in progress."
            )
            return False
        if self.has_running_task and (
            self.open_runner.is_running
            or self.publish_runner.is_running
            or self.discovery_runner.is_running
            or self.controller.state.open_running
            or self.controller.state.publish_running
            or self.controller.state.discovery_running
        ):
            self.status_message.emit(
                f"Cannot {action} while a discovery/open/publish task is running."
            )
            return False
        return True

    def _guard_edit_idle(self, action: str) -> bool:
        if self.controller.state.edit_running or self.edit_runner.is_running:
            self.status_message.emit(
                f"Cannot {action}: another edit is already running."
            )
            return False
        if self.controller.state.publish_running or self.publish_runner.is_running:
            self.status_message.emit(
                f"Cannot {action} while publication is in progress."
            )
            return False
        if self.controller.state.open_running or self.open_runner.is_running:
            self.status_message.emit(
                f"Cannot {action} while a package/draft open task is running."
            )
            return False
        return True

    def _emit_running_changed(self) -> None:
        running = self.has_running_task
        if running != self._last_running_state:
            self._last_running_state = running
            self.task_running_changed.emit(running)


class _signals_blocked:
    """Temporarily block signals on one or more widgets."""

    def __init__(self, *widgets: QWidget) -> None:
        self._widgets = widgets

    def __enter__(self) -> None:
        for widget in self._widgets:
            widget.blockSignals(True)

    def __exit__(self, *args: object) -> None:
        for widget in self._widgets:
            widget.blockSignals(False)
