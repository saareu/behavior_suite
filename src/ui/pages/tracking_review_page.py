"""Video-centered Subsystem 03 tracking review workspace."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from tracking_correction.contracts import (
    ACCEPTANCE_SUPERSEDED,
    TrackingCorrectionError,
    is_currently_accepted,
)
from tracking_correction.review import (
    CorrectionEpisode,
    PoseDisplaySource,
    track_legend_entries,
)
from tracking_correction.run_discovery import TrackingCorrectionRunSummary
from ui.controllers.tracking_review_controller import TrackingReviewController
from ui.widgets.pose_video_view import PoseVideoView

# Poll frequently; wall-clock math decides which prepared frame to show.
_PLAYBACK_POLL_INTERVAL_MS = 8


class TrackingReviewPage(QWidget):
    """Minimal local review UI for a completed automatic S3 run."""

    unexpected_error = Signal(str)
    status_message = Signal(str)
    accepted = Signal(object)

    def __init__(
        self,
        controller: TrackingReviewController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or TrackingReviewController()
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_playback_tick)
        self._reviewable_runs: tuple[TrackingCorrectionRunSummary, ...] = ()
        self._episode_list_items: list[CorrectionEpisode] = []

        self.video = PoseVideoView()
        self.video.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.run_label = QLabel("No project or S3 run loaded")
        self.run_label.setWordWrap(True)
        self.run_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.provenance_label = QLabel("S3 output: —")
        self.provenance_label.setWordWrap(True)
        self.provenance_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.frame_label = QLabel("Frame: —")
        self.correction_label = QLabel("Corrections: —")
        self.correction_label.setWordWrap(True)
        self.episode_label = QLabel("Correction episode: —")
        self.episode_label.setWordWrap(True)
        self.acceptance_label = QLabel("Acceptance: —")
        self.skeleton_label = QLabel("Skeleton: —")
        self.skeleton_label.setWordWrap(True)
        self.manual_edits_label = QLabel("Manual edits: 0")
        self.legend_widget = self._build_legend_widget()

        self.run_combo = QComboBox()
        self.run_combo.setVisible(False)
        self.run_combo.currentIndexChanged.connect(self._run_selection_changed)

        self.play_button = QPushButton("Play")
        self.back_button = QPushButton("◀ Frame")
        self.forward_button = QPushButton("Frame ▶")
        self.prev_correction_button = QPushButton("◀ Auto")
        self.next_correction_button = QPushButton("Auto ▶")
        self.accept_button = QPushButton("Accept Tracking")
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("frame index")
        self.jump_button = QPushButton("Jump")
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setMinimum(0)
        self.scrubber.setMaximum(0)
        self.source_combo = QComboBox()
        self.source_combo.addItem("S3 corrected working pose", PoseDisplaySource.CORRECTED.value)
        self.source_combo.addItem("S2 provisional pose", PoseDisplaySource.PROVISIONAL.value)
        self.speed_combo = QComboBox()
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)

        self.interval_start_spin = QSpinBox()
        self.interval_end_spin = QSpinBox()
        for spin in (self.interval_start_spin, self.interval_end_spin):
            spin.setMinimum(0)
            spin.setMaximum(0)
        self.use_current_start_button = QPushButton("Start = current")
        self.use_current_end_button = QPushButton("End = current")
        self.swap_identities_button = QPushButton("Swap identities")
        self.node_combo = QComboBox()
        self.track_combo = QComboBox()
        self.track_combo.addItem("Track 0", 0)
        self.track_combo.addItem("Track 1", 1)
        self.swap_node_button = QPushButton("Swap node")
        self.blank_node_button = QPushButton("Blank node")
        self.undo_button = QPushButton("Undo")
        self.reset_manual_button = QPushButton("Reset manual edits")

        self.episode_list = QListWidget()
        self.episode_list.setMinimumHeight(120)
        self.episode_list.currentRowChanged.connect(self._episode_list_selection_changed)

        transport = QHBoxLayout()
        for widget in (
            self.play_button,
            self.back_button,
            self.forward_button,
        ):
            transport.addWidget(widget)
        transport.addStretch(1)
        transport.addWidget(QLabel("Speed"))
        transport.addWidget(self.speed_combo)

        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Jump to frame"))
        jump_row.addWidget(self.jump_edit, 1)
        jump_row.addWidget(self.jump_button)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self.video, 1)
        viewer_layout.addWidget(self.scrubber)
        viewer_layout.addLayout(transport)

        corrections_panel = QWidget()
        corrections_layout = QVBoxLayout(corrections_panel)
        corrections_layout.setContentsMargins(0, 0, 0, 0)
        episode_nav = QHBoxLayout()
        episode_nav.addWidget(self.prev_correction_button)
        episode_nav.addWidget(self.next_correction_button)
        corrections_layout.addLayout(episode_nav)
        corrections_layout.addWidget(self.episode_label)
        corrections_layout.addWidget(self.correction_label)
        corrections_layout.addStretch(1)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Start"))
        interval_row.addWidget(self.interval_start_spin)
        interval_row.addWidget(self.use_current_start_button)
        interval_row.addWidget(QLabel("End"))
        interval_row.addWidget(self.interval_end_spin)
        interval_row.addWidget(self.use_current_end_button)

        node_row = QHBoxLayout()
        node_row.addWidget(QLabel("Node"))
        node_row.addWidget(self.node_combo, 1)
        node_row.addWidget(QLabel("Track"))
        node_row.addWidget(self.track_combo)
        node_row.addWidget(self.swap_node_button)
        node_row.addWidget(self.blank_node_button)

        undo_row = QHBoxLayout()
        undo_row.addWidget(self.undo_button)
        undo_row.addWidget(self.reset_manual_button)
        undo_row.addStretch(1)

        manual_panel = QWidget()
        manual_layout = QVBoxLayout(manual_panel)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.addLayout(interval_row)
        manual_layout.addWidget(self.swap_identities_button)
        manual_layout.addLayout(node_row)
        manual_layout.addLayout(undo_row)
        manual_layout.addWidget(self.manual_edits_label)
        manual_layout.addWidget(QLabel("# | action | frames | target"))
        manual_layout.addWidget(self.episode_list, 1)

        status_panel = QWidget()
        status_form = QFormLayout(status_panel)
        status_form.addRow("S3 run", self.run_combo)
        status_form.addRow("Pose source", self.source_combo)
        status_form.addRow(jump_row)
        status_form.addRow(self.frame_label)
        status_form.addRow(self.skeleton_label)
        status_form.addRow(self.acceptance_label)
        status_form.addRow(self.accept_button)
        status_form.addRow(self.provenance_label)

        side_toolbox = QToolBox()
        side_toolbox.addItem(corrections_panel, "Machine corrections")
        side_toolbox.addItem(manual_panel, "Manual correction")
        side_toolbox.addItem(status_panel, "Provenance / status")

        side_inner = QWidget()
        side_inner_layout = QVBoxLayout(side_inner)
        side_inner_layout.setContentsMargins(0, 0, 0, 0)
        side_inner_layout.addWidget(self.legend_widget)
        side_inner_layout.addWidget(side_toolbox, 1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side_inner)
        side_scroll.setMinimumWidth(280)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(viewer_panel)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.run_label)
        layout.addWidget(splitter, 1)

        self.play_button.clicked.connect(self._toggle_play)
        self.back_button.clicked.connect(self._step_backward)
        self.forward_button.clicked.connect(self._step_forward)
        self.prev_correction_button.clicked.connect(self._prev_correction)
        self.next_correction_button.clicked.connect(self._next_correction)
        self.jump_button.clicked.connect(self._jump_to_frame)
        self.jump_edit.returnPressed.connect(self._jump_to_frame)
        self.scrubber.valueChanged.connect(self._scrubber_changed)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.accept_button.clicked.connect(self._accept_tracking)
        self.use_current_start_button.clicked.connect(self._use_current_as_start)
        self.use_current_end_button.clicked.connect(self._use_current_as_end)
        self.swap_identities_button.clicked.connect(self._swap_identities)
        self.swap_node_button.clicked.connect(self._swap_node)
        self.blank_node_button.clicked.connect(self._blank_node)
        self.undo_button.clicked.connect(self._undo_manual_edit)
        self.reset_manual_button.clicked.connect(self._reset_manual_edits)
        self._set_controls_enabled(False)

    def _build_legend_widget(self) -> QGroupBox:
        box = QGroupBox("Track colors")
        layout = QVBoxLayout(box)
        for entry in track_legend_entries():
            row = QHBoxLayout()
            swatch = QLabel()
            pixmap = QPixmap(14, 14)
            pixmap.fill(QColor(*entry.color_rgb))
            swatch.setPixmap(pixmap)
            swatch.setFixedSize(14, 14)
            label = f"Track {entry.track}"
            if entry.role_label:
                label = f"{label} — {entry.role_label}"
            text = QLabel(label)
            row.addWidget(swatch)
            row.addWidget(text, 1)
            layout.addLayout(row)
        return box

    def open_project(self, session_root: Path) -> None:
        """Open a project/session directory and load a completed S3 run when possible."""

        self._stop_playback()
        try:
            summary = self.controller.open_project(session_root)
        except TrackingCorrectionError as exc:
            self._set_controls_enabled(False)
            QMessageBox.warning(self, "Cannot open project", str(exc))
            self.status_message.emit(str(exc))
            return

        presence = summary.artifact_presence
        self.run_label.setText(
            f"Project: {summary.session_root}\n"
            f"preprocess/: {'yes' if presence.preprocess else 'no'} · "
            f"pose_inference/: {'yes' if presence.pose_inference else 'no'} · "
            f"tracking_correction/: {'yes' if presence.tracking_correction else 'no'}"
        )
        reviewable = summary.reviewable_runs
        self._populate_run_combo(reviewable)
        if not reviewable:
            self._set_controls_enabled(False)
            self.provenance_label.setText("S3 output: (none)")
            message = (
                "No completed S3 runs found under "
                f"{summary.session_root / 'tracking_correction'}."
            )
            QMessageBox.information(self, "No S3 runs", message)
            self.status_message.emit(message)
            return
        # Visually selected first run must also be the loaded run.
        self.open_run(reviewable[0].run_dir)

    def open_path(self, path: Path) -> None:
        """Open either a session root or a direct completed S3 run directory."""

        from tracking_correction.run_discovery import is_completed_s3_run_dir

        target = Path(path)
        if is_completed_s3_run_dir(target):
            self.open_run(target)
            return
        self.open_project(target)

    def open_run(self, s3_run_dir: Path) -> None:
        """Open one completed automatic S3 run for review."""

        self._stop_playback()
        try:
            session = self.controller.open_run(s3_run_dir)
        except TrackingCorrectionError as exc:
            self._set_controls_enabled(False)
            QMessageBox.warning(self, "Cannot open S3 run", str(exc))
            self.status_message.emit(str(exc))
            return

        if self.controller.project is not None:
            reviewable = self.controller.project.reviewable_runs
            self._populate_run_combo(reviewable, selected=session.run_dir)
            presence = self.controller.project.artifact_presence
            self.run_label.setText(
                f"Project: {self.controller.project.session_root}\n"
                f"preprocess/: {'yes' if presence.preprocess else 'no'} · "
                f"pose_inference/: {'yes' if presence.pose_inference else 'no'} · "
                f"tracking_correction/: {'yes' if presence.tracking_correction else 'no'}\n"
                f"Selected S3 run: {session.run_id}"
            )
        else:
            self._populate_run_combo(())
            self.run_label.setText(f"S3 run: {session.run_id}")

        self.provenance_label.setText(
            f"S3 output: {session.run_dir}\n"
            f"Video: {session.prepared_video_path}\n"
            f"Working pose: {session.working_tracked_pose_path}\n"
            f"Playback FPS: {session.fps_header:g} Hz — {session.fps_source}"
            + (
                f" ({session.fps_source_detail})"
                if session.fps_source_detail
                else ""
            )
        )
        self.scrubber.blockSignals(True)
        self.scrubber.setMaximum(max(0, session.frame_count - 1))
        self.scrubber.setValue(session.current_frame)
        self.scrubber.blockSignals(False)
        max_frame = max(0, session.frame_count - 1)
        self.interval_start_spin.setMaximum(max_frame)
        self.interval_end_spin.setMaximum(max_frame)
        self.interval_start_spin.setValue(session.current_frame)
        self.interval_end_spin.setValue(session.current_frame)
        self._populate_node_combo(session.available_node_names)
        self._populate_episode_list(session.navigable_episodes)
        speed_index = self.speed_combo.findData(session.playback_speed)
        if speed_index >= 0:
            self.speed_combo.setCurrentIndex(speed_index)
        self._set_controls_enabled(True)
        self._refresh_view()
        self.status_message.emit(f"Opened S3 run {session.run_id}")

    def _populate_run_combo(
        self,
        runs: tuple[TrackingCorrectionRunSummary, ...],
        *,
        selected: Path | None = None,
    ) -> None:
        self._reviewable_runs = runs
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in runs:
            self.run_combo.addItem(run.run_id, str(run.run_dir))
        self.run_combo.setVisible(len(runs) > 1)
        if selected is not None and runs:
            selected_text = str(selected.resolve())
            index = self.run_combo.findData(selected_text)
            if index < 0:
                for i, run in enumerate(runs):
                    if run.run_dir.resolve() == selected.resolve():
                        index = i
                        break
            if index >= 0:
                self.run_combo.setCurrentIndex(index)
        elif runs:
            self.run_combo.setCurrentIndex(0)
        self.run_combo.blockSignals(False)

    def _populate_episode_list(self, episodes: tuple[CorrectionEpisode, ...]) -> None:
        self._episode_list_items = list(episodes)
        self.episode_list.blockSignals(True)
        self.episode_list.clear()
        for index, episode in enumerate(episodes, start=1):
            item = QListWidgetItem(_format_manual_edit_list_row(index, episode))
            item.setData(Qt.ItemDataRole.UserRole, index - 1)
            self.episode_list.addItem(item)
        self.episode_list.blockSignals(False)

    def _episode_list_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._episode_list_items):
            return
        episode = self._episode_list_items[row]
        session = self.controller.session
        if session is not None and session.current_frame == episode.start_frame:
            return
        self._stop_playback()
        try:
            self.controller.jump_to_episode(episode)
        except TrackingCorrectionError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_view(sync_episode_list=False)

    def _run_selection_changed(self, index: int) -> None:
        if index < 0 or not self._reviewable_runs:
            return
        path_text = self.run_combo.itemData(index)
        if not path_text:
            return
        chosen = Path(str(path_text))
        if (
            self.controller.session is not None
            and self.controller.session.run_dir.resolve() == chosen.resolve()
        ):
            return
        self.open_run(chosen)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.play_button,
            self.back_button,
            self.forward_button,
            self.prev_correction_button,
            self.next_correction_button,
            self.jump_button,
            self.jump_edit,
            self.scrubber,
            self.source_combo,
            self.speed_combo,
            self.accept_button,
            self.interval_start_spin,
            self.interval_end_spin,
            self.use_current_start_button,
            self.use_current_end_button,
            self.swap_identities_button,
            self.node_combo,
            self.track_combo,
            self.swap_node_button,
            self.blank_node_button,
            self.undo_button,
            self.reset_manual_button,
            self.episode_list,
        ):
            widget.setEnabled(enabled)

    def _populate_node_combo(self, nodes: tuple[str, ...]) -> None:
        current = self.node_combo.currentData()
        self.node_combo.blockSignals(True)
        self.node_combo.clear()
        for node in nodes:
            self.node_combo.addItem(node, node)
        if current is not None:
            index = self.node_combo.findData(current)
            if index >= 0:
                self.node_combo.setCurrentIndex(index)
        self.node_combo.blockSignals(False)

    def _refresh_view(self, *, sync_episode_list: bool = True) -> None:
        try:
            view = self.controller.current_view()
        except TrackingCorrectionError as exc:
            self.status_message.emit(str(exc))
            QMessageBox.warning(self, "Frame load failed", str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        session = self.controller.session
        self.video.set_frame(view.image_bgr)
        self.frame_label.setText(
            f"Prepared frame: {view.frame_idx} / {max(0, view.frame_count - 1)} "
            f"({view.frame_count} frames @ {session.fps_header:g} Hz — {session.fps_source})"
            if session is not None
            else f"Prepared frame: {view.frame_idx} / {max(0, view.frame_count - 1)}"
        )
        if view.correction_types:
            self.correction_label.setText(
                "Corrections on this frame: " + ", ".join(view.correction_types)
            )
        else:
            self.correction_label.setText("Corrections on this frame: none")
        episode = view.correction_episode
        if episode is None:
            self.episode_label.setText("Machine episode: none")
        else:
            types = ", ".join(episode.correction_types) if episode.correction_types else "none"
            self.episode_label.setText(
                f"Machine episode: frames {episode.start_frame}–{episode.end_frame} "
                f"({episode.frame_count} frames); types: {types}"
            )
        if session is not None:
            edge_count = len(session.skeleton_edges)
            self.skeleton_label.setText(
                f"Skeleton: {edge_count} edge(s) from {session.skeleton_source}"
            )
            if sync_episode_list:
                self._populate_episode_list(session.navigable_episodes)
                self._sync_episode_list_selection(session.current_frame)
        if view.acceptance_state == ACCEPTANCE_SUPERSEDED:
            self.acceptance_label.setText(
                "Acceptance: superseded — press Accept Tracking again"
            )
        else:
            self.acceptance_label.setText(f"Acceptance: {view.acceptance_state}")
        self.manual_edits_label.setText(f"Manual edits: {view.manual_edit_count}")
        self._populate_node_combo(view.available_nodes)
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(view.frame_idx)
        self.scrubber.blockSignals(False)
        already = is_currently_accepted(view.acceptance_state)
        if already:
            self.accept_button.setText("Accept Tracking (already accepted)")
        elif view.acceptance_state == ACCEPTANCE_SUPERSEDED:
            self.accept_button.setText("Accept Tracking (re-accept required)")
        else:
            self.accept_button.setText("Accept Tracking")
        can_undo = view.manual_edit_count > 0
        self.undo_button.setEnabled(self.accept_button.isEnabled() and can_undo)
        self.reset_manual_button.setEnabled(
            self.accept_button.isEnabled()
            and (
                can_undo
                or (session is not None and session.has_automatic_baseline)
            )
        )

    def _sync_episode_list_selection(self, frame_idx: int) -> None:
        selected = -1
        for index, episode in enumerate(self._episode_list_items):
            if episode.start_frame <= frame_idx <= episode.end_frame:
                selected = index
                break
        self.episode_list.blockSignals(True)
        self.episode_list.setCurrentRow(selected)
        self.episode_list.blockSignals(False)

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        session = self.controller.session
        if session is None:
            return
        if session.current_frame >= session.max_frame_idx:
            self.status_message.emit("Already at the last prepared frame.")
            return
        self.controller.begin_playback(now_monotonic=time.monotonic())
        self._playing = True
        self.play_button.setText("Pause")
        self._timer.start(_PLAYBACK_POLL_INTERVAL_MS)

    def _stop_playback(self) -> None:
        self._playing = False
        self.play_button.setText("Play")
        self._timer.stop()
        self.controller.stop_playback()

    def _on_playback_tick(self) -> None:
        if self.controller.session is None:
            self._stop_playback()
            return
        previous = self.controller.session.current_frame
        frame_idx = self.controller.sync_playback_frame(now_monotonic=time.monotonic())
        if frame_idx is None:
            self._refresh_view()
            self._stop_playback()
            self.status_message.emit("Reached end of prepared video.")
            return
        if frame_idx != previous:
            self._refresh_view()

    def _step_forward(self) -> None:
        self._stop_playback()
        self.controller.step_forward()
        self._refresh_view()

    def _step_backward(self) -> None:
        self._stop_playback()
        self.controller.step_backward()
        self._refresh_view()

    def _next_correction(self) -> None:
        self._stop_playback()
        episode = self.controller.next_correction_episode()
        if episode is None:
            self.status_message.emit("No later automatic correction episode.")
            return
        self._refresh_view()
        self.status_message.emit(
            f"Automatic episode {episode.start_frame}–{episode.end_frame}"
        )

    def _prev_correction(self) -> None:
        self._stop_playback()
        episode = self.controller.previous_correction_episode()
        if episode is None:
            self.status_message.emit("No earlier automatic correction episode.")
            return
        self._refresh_view()
        self.status_message.emit(
            f"Automatic episode {episode.start_frame}–{episode.end_frame}"
        )

    def _jump_to_frame(self) -> None:
        self._stop_playback()
        text = self.jump_edit.text().strip()
        try:
            frame_idx = int(text)
            self.controller.set_frame(frame_idx)
        except (ValueError, TrackingCorrectionError) as exc:
            QMessageBox.warning(self, "Invalid frame", str(exc))
            return
        self._refresh_view()

    def _scrubber_changed(self, value: int) -> None:
        if self.controller.session is None:
            return
        if value == self.controller.session.current_frame:
            return
        self._stop_playback()
        try:
            self.controller.set_frame(int(value))
        except TrackingCorrectionError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_view()

    def _source_changed(self) -> None:
        if self.controller.session is None:
            return
        source = self.source_combo.currentData()
        try:
            self.controller.set_pose_source(str(source))
        except TrackingCorrectionError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_view()

    def _speed_changed(self) -> None:
        if self.controller.session is None:
            return
        speed = self.speed_combo.currentData()
        try:
            self.controller.set_playback_speed(float(speed))
        except TrackingCorrectionError as exc:
            self.status_message.emit(str(exc))
            return
        if self._playing:
            self.controller.begin_playback(now_monotonic=time.monotonic())

    def _accept_tracking(self) -> None:
        self._stop_playback()
        reply = QMessageBox.question(
            self,
            "Accept Tracking",
            "Accept the current working corrected pose as the final S3 tracked result?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self.controller.accept_tracking()
        except TrackingCorrectionError as exc:
            QMessageBox.warning(self, "Acceptance failed", str(exc))
            self.status_message.emit(str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        message = (
            "Already accepted; tracked_pose.parquet unchanged."
            if result.already_accepted
            else f"Accepted tracking.\n{result.tracked_pose_path}"
        )
        QMessageBox.information(self, "Tracking accepted", message)
        self.status_message.emit(message)
        self.accepted.emit(result)
        self._refresh_view()

    def _use_current_as_start(self) -> None:
        session = self.controller.session
        if session is None:
            return
        self.interval_start_spin.setValue(session.current_frame)

    def _use_current_as_end(self) -> None:
        session = self.controller.session
        if session is None:
            return
        self.interval_end_spin.setValue(session.current_frame)

    def _swap_identities(self) -> None:
        self._stop_playback()
        try:
            result = self.controller.swap_identities(
                self.interval_start_spin.value(),
                self.interval_end_spin.value(),
            )
        except TrackingCorrectionError as exc:
            QMessageBox.warning(self, "Swap identities failed", str(exc))
            self.status_message.emit(str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        self._refresh_view()
        message = (
            f"Swapped identities on frames {result.start_frame}–{result.end_frame}."
        )
        if result.acceptance_invalidated:
            message += " Prior acceptance superseded."
        self.status_message.emit(message)

    def _selected_node(self) -> str | None:
        value = self.node_combo.currentData()
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _swap_node(self) -> None:
        self._stop_playback()
        node = self._selected_node()
        if node is None:
            QMessageBox.warning(self, "Swap node failed", "Select a node first.")
            return
        try:
            result = self.controller.swap_node(node)
        except TrackingCorrectionError as exc:
            QMessageBox.warning(self, "Swap node failed", str(exc))
            self.status_message.emit(str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        self._refresh_view()
        message = f"Swapped node {node!r} at frame {result.start_frame}."
        if result.acceptance_invalidated:
            message += " Prior acceptance superseded."
        self.status_message.emit(message)

    def _blank_node(self) -> None:
        self._stop_playback()
        node = self._selected_node()
        if node is None:
            QMessageBox.warning(self, "Blank node failed", "Select a node first.")
            return
        track = int(self.track_combo.currentData())
        try:
            result = self.controller.blank_node(node, track)
        except TrackingCorrectionError as exc:
            QMessageBox.warning(self, "Blank node failed", str(exc))
            self.status_message.emit(str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        self._refresh_view()
        message = f"Blanked node {node!r} on track {track} at frame {result.start_frame}."
        if result.acceptance_invalidated:
            message += " Prior acceptance superseded."
        self.status_message.emit(message)

    def _undo_manual_edit(self) -> None:
        self._stop_playback()
        try:
            result = self.controller.undo_last_manual_edit()
        except TrackingCorrectionError as exc:
            QMessageBox.warning(self, "Undo failed", str(exc))
            self.status_message.emit(str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        self._refresh_view()
        message = f"Undid last manual edit ({result.record and result.record.get('action')})."
        if result.acceptance_invalidated:
            message += " Prior acceptance superseded."
        self.status_message.emit(message)

    def _reset_manual_edits(self) -> None:
        self._stop_playback()
        reply = QMessageBox.question(
            self,
            "Reset manual edits",
            "Restore the automatic working pose and discard all manual edits?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            result = self.controller.reset_manual_edits()
        except TrackingCorrectionError as exc:
            QMessageBox.warning(self, "Reset failed", str(exc))
            self.status_message.emit(str(exc))
            return
        except Exception as exc:
            self.unexpected_error.emit(str(exc))
            return
        self._refresh_view()
        message = "Manual edits reset to the automatic baseline."
        if result.acceptance_invalidated:
            message += " Prior acceptance superseded."
        self.status_message.emit(message)


def _format_manual_edit_list_row(index: int, episode: CorrectionEpisode) -> str:
    """Compact one-line summary of a manual edit for the visible list."""

    action = episode.correction_types[0] if episode.correction_types else "manual_edit"
    if episode.is_single_frame:
        frames = f"f{episode.start_frame}"
    else:
        frames = f"{episode.start_frame}–{episode.end_frame}"
    target_parts: list[str] = []
    if episode.node:
        target_parts.append(episode.node)
    if episode.track is not None:
        target_parts.append(f"track {episode.track}")
    elif episode.tracks:
        track_text = ",".join(str(track) for track in episode.tracks)
        target_parts.append(f"tracks {track_text}")
    target = " · ".join(target_parts) if target_parts else "—"
    return f"{index} | {action} | {frames} | {target}"
