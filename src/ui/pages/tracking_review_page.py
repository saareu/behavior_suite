"""Video-centered Subsystem 03 tracking review workspace."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from tracking_correction.contracts import ACCEPTANCE_ACCEPTED, TrackingCorrectionError
from tracking_correction.review import PoseDisplaySource
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

        self.video = PoseVideoView()
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

        self.run_combo = QComboBox()
        self.run_combo.setVisible(False)
        self.run_combo.currentIndexChanged.connect(self._run_selection_changed)

        self.play_button = QPushButton("Play")
        self.back_button = QPushButton("◀ Frame")
        self.forward_button = QPushButton("Frame ▶")
        self.prev_correction_button = QPushButton("◀ Episode")
        self.next_correction_button = QPushButton("Episode ▶")
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

        transport = QHBoxLayout()
        for widget in (
            self.play_button,
            self.back_button,
            self.forward_button,
            self.prev_correction_button,
            self.next_correction_button,
        ):
            transport.addWidget(widget)
        transport.addStretch(1)
        transport.addWidget(QLabel("Speed"))
        transport.addWidget(self.speed_combo)

        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Jump to frame"))
        jump_row.addWidget(self.jump_edit, 1)
        jump_row.addWidget(self.jump_button)

        form = QFormLayout()
        form.addRow("S3 run", self.run_combo)
        form.addRow("Pose source", self.source_combo)
        form.addRow(jump_row)
        form.addRow(self.frame_label)
        form.addRow(self.correction_label)
        form.addRow(self.episode_label)
        form.addRow(self.skeleton_label)
        form.addRow(self.acceptance_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.run_label)
        layout.addWidget(self.provenance_label)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.scrubber)
        layout.addLayout(transport)
        layout.addLayout(form)
        layout.addWidget(self.accept_button, alignment=Qt.AlignmentFlag.AlignLeft)

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
        self._set_controls_enabled(False)

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
        if len(reviewable) == 1:
            self.open_run(reviewable[0].run_dir)
            return
        self.status_message.emit(
            f"Found {len(reviewable)} completed S3 runs; select one to review."
        )

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
                # Compare resolved paths loosely.
                for i, run in enumerate(runs):
                    if run.run_dir.resolve() == selected.resolve():
                        index = i
                        break
            if index >= 0:
                self.run_combo.setCurrentIndex(index)
        self.run_combo.blockSignals(False)

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
        ):
            widget.setEnabled(enabled)

    def _refresh_view(self) -> None:
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
            self.episode_label.setText("Correction episode: none")
        else:
            types = ", ".join(episode.correction_types) if episode.correction_types else "none"
            self.episode_label.setText(
                f"Correction episode: frames {episode.start_frame}–{episode.end_frame} "
                f"({episode.frame_count} frames); types: {types}"
            )
        if session is not None:
            edge_count = len(session.skeleton_edges)
            self.skeleton_label.setText(
                f"Skeleton: {edge_count} edge(s) from {session.skeleton_source}"
            )
        self.acceptance_label.setText(f"Acceptance: {view.acceptance_state}")
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(view.frame_idx)
        self.scrubber.blockSignals(False)
        already = view.acceptance_state == ACCEPTANCE_ACCEPTED
        self.accept_button.setText(
            "Accept Tracking (already accepted)" if already else "Accept Tracking"
        )

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
            self.status_message.emit("No later correction episode.")
            return
        self._refresh_view()
        self.status_message.emit(
            f"Correction episode {episode.start_frame}–{episode.end_frame}"
        )

    def _prev_correction(self) -> None:
        self._stop_playback()
        episode = self.controller.previous_correction_episode()
        if episode is None:
            self.status_message.emit("No earlier correction episode.")
            return
        self._refresh_view()
        self.status_message.emit(
            f"Correction episode {episode.start_frame}–{episode.end_frame}"
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
