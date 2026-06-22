"""Final preprocessing review, background execution, and result display."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from preprocess.models import PreprocessResult
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.controllers.run_preprocess_controller import (
    FinalReviewSummary,
    RunPreprocessController,
    RunPreprocessValidationError,
    RunResultSummary,
)
from ui.tasks import GuiTaskRunner


class RunValidatePage(QWidget):
    """Show the final review and dispatch one service run off the GUI thread."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)

    def __init__(self, setup_controller: PreprocessSetupController) -> None:
        super().__init__()
        self.setup_controller = setup_controller
        self.controller = RunPreprocessController(setup_controller.state)
        self.task_runner = GuiTaskRunner(self)

        title = QLabel("Run and Validate")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        explanation = QLabel(
            "Review the accepted inputs and official output paths before running the "
            "existing preprocessing service."
        )
        explanation.setWordWrap(True)

        review_group = QGroupBox("Final review")
        review_form = QFormLayout(review_group)
        self.review_labels: dict[str, QLabel] = {}
        for key, caption in (
            ("project", "Project directory"),
            ("raw", "Raw video path"),
            ("trim", "Trim range"),
            ("pre_crop", "Pre-crop"),
            ("timing", "Timing"),
            ("crop_mode", "Crop mode"),
            ("crop_size", "Crop output dimensions"),
            ("crop_accepted", "Accepted crop status"),
            ("canonical", "Canonical resolution"),
        ):
            label = self._selectable_label()
            self.review_labels[key] = label
            review_form.addRow(caption, label)

        output_group = QGroupBox("Official output paths")
        output_form = QFormLayout(output_group)
        self.review_output_labels: list[QLabel] = []
        for filename in (
            "prepared_video.mp4",
            "prepare_meta.json",
            "prepared_sync.npz",
            "cropped_background.png",
            "settings_used.yaml",
            "processing_log.txt",
        ):
            label = self._selectable_label()
            self.review_output_labels.append(label)
            output_form.addRow(filename, label)

        self.run_button = QPushButton("Run Preprocessing")
        self.run_button.clicked.connect(self._run_preprocessing)
        self.open_folder_button = QPushButton("Open Preprocess Folder")
        self.open_folder_button.clicked.connect(self._open_preprocess_folder)
        action_row = QHBoxLayout()
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.open_folder_button)
        action_row.addStretch(1)

        self.activity = QProgressBar()
        self.activity.setRange(0, 0)
        self.activity.setVisible(False)
        self.stage_label = QLabel("Not started")
        self.elapsed_label = QLabel("Elapsed: 0.0 s")
        activity_row = QHBoxLayout()
        activity_row.addWidget(self.activity)
        activity_row.addWidget(self.stage_label, 1)
        activity_row.addWidget(self.elapsed_label)

        result_group = QGroupBox("Run result")
        result_layout = QVBoxLayout(result_group)
        self.result_status = QLabel("Status: NOT RUN")
        self.result_status.setStyleSheet("font-weight: 600;")
        self.result_message = QLabel("Preprocessing has not run.")
        self.result_message.setWordWrap(True)
        self.result_details = QLabel()
        self.result_details.setWordWrap(True)
        self.result_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.result_warnings = QLabel()
        self.result_warnings.setWordWrap(True)
        self.result_paths = QLabel()
        self.result_paths.setWordWrap(True)
        self.result_paths.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        result_layout.addWidget(self.result_status)
        result_layout.addWidget(self.result_message)
        result_layout.addWidget(self.result_details)
        result_layout.addWidget(self.result_warnings)
        result_layout.addWidget(self.result_paths)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(review_group)
        content_layout.addWidget(output_group)
        content_layout.addLayout(action_row)
        content_layout.addLayout(activity_row)
        content_layout.addWidget(result_group)
        content_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.error_label)

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    @staticmethod
    def _selectable_label() -> QLabel:
        label = QLabel("—")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def on_activated(self) -> None:
        """Refresh the review whenever upstream state navigates here."""

        self.refresh_from_state()

    def commit(self) -> bool:
        """The final page has no Next action."""

        return True

    def refresh_from_state(self) -> None:
        """Render current review, task state, and typed result."""

        if self.controller.can_run():
            try:
                self._render_review(self.controller.build_final_review())
            except RunPreprocessValidationError as exc:
                self._show_expected_error(str(exc))
        else:
            self._clear_review()
        running = self.controller.state.preprocess_task_running
        self.activity.setVisible(running)
        self.run_button.setEnabled(self.controller.can_run())
        self.open_folder_button.setEnabled(
            not running and self.controller.can_open_preprocess_folder()
        )
        self.stage_label.setText(self.controller.state.run_status)
        self._update_elapsed()
        self._render_result(self.controller.result_summary())
        self.validity_changed.emit()

    def _render_review(self, summary: FinalReviewSummary) -> None:
        values = {
            "project": str(summary.project_directory),
            "raw": str(summary.raw_video_path),
            "trim": summary.trim_range,
            "pre_crop": summary.pre_crop_summary,
            "timing": summary.timing_summary,
            "crop_mode": summary.crop_mode,
            "crop_size": f"{summary.crop_output_dimensions[0]} × "
            f"{summary.crop_output_dimensions[1]}",
            "crop_accepted": "Accepted" if summary.crop_accepted else "Not accepted",
            "canonical": summary.canonical_resolution,
        }
        for key, value in values.items():
            self.review_labels[key].setText(value)
        for label, (_, path) in zip(
            self.review_output_labels,
            summary.official_outputs,
            strict=True,
        ):
            label.setText(str(path))

    def _clear_review(self) -> None:
        for label in self.review_labels.values():
            label.setText("—")
        for label in self.review_output_labels:
            label.setText("—")

    def _run_preprocessing(self) -> None:
        try:
            self.controller.start_run(
                self.task_runner,
                on_complete=self._task_complete,
                on_error=self._task_error,
                on_finished=self._task_finished,
                on_stage=self._stage_changed,
            )
        except RunPreprocessValidationError as exc:
            self._show_expected_error(str(exc))
            return
        except Exception as exc:
            self.setup_controller.record_unexpected_error(exc)
            self._show_expected_error(
                "An unexpected error occurred while starting preprocessing."
            )
            return
        self.error_label.clear()
        self.activity.setVisible(True)
        self.elapsed_timer.start()
        self.run_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.validity_changed.emit()

    def _task_complete(self, _result: PreprocessResult) -> None:
        self.refresh_from_state()

    def _task_error(self, _exc: BaseException) -> None:
        self.refresh_from_state()
        message = self.controller.state.run_error_message or "Preprocessing failed."
        self._show_expected_error(message)

    def _task_finished(self) -> None:
        self.elapsed_timer.stop()
        self.activity.setVisible(False)
        self.refresh_from_state()
        self.status_message.emit(self.result_status.text())

    def _stage_changed(self, stage: str) -> None:
        self.stage_label.setText(stage)
        self.status_message.emit(stage)
        self.validity_changed.emit()

    def _update_elapsed(self) -> None:
        elapsed = self.controller.update_elapsed_time()
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f} s")

    def _render_result(self, summary: RunResultSummary) -> None:
        self.result_status.setText(f"Status: {summary.status}")
        if summary.status == "SUCCESS":
            self.result_status.setStyleSheet("font-weight: 600; color: #176b2c;")
        elif summary.status == "FAILED":
            self.result_status.setStyleSheet("font-weight: 600; color: #b00020;")
        else:
            self.result_status.setStyleSheet("font-weight: 600;")
        self.result_message.setText(summary.message)
        details: list[str] = []
        if summary.prepared_frame_count is not None:
            details.append(f"Prepared frame count: {summary.prepared_frame_count}")
        if summary.prepared_size_wh is not None:
            details.append(
                f"Prepared video dimensions: {summary.prepared_size_wh[0]} × "
                f"{summary.prepared_size_wh[1]}"
            )
        if summary.prepared_fps is not None:
            details.append(f"Prepared FPS: {summary.prepared_fps:.6g}")
        details.append(f"Elapsed time: {summary.elapsed_sec:.1f} s")
        self.result_details.setText("\n".join(details))
        self.result_warnings.setText(
            "Warnings:\n" + "\n".join(f"- {item}" for item in summary.warnings)
            if summary.warnings
            else ""
        )
        paths = [f"{name}: {path}" for name, path in summary.official_outputs]
        if summary.status == "FAILED" and summary.processing_log_path is not None:
            paths.append(f"Processing log: {summary.processing_log_path}")
        self.result_paths.setText("\n".join(paths))

    def _open_preprocess_folder(self) -> None:
        try:
            folder = self.controller.open_preprocess_folder()
        except RunPreprocessValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self.error_label.clear()
        self.status_message.emit(f"Opened preprocess folder: {folder}")

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_message.emit(message)
        self.validity_changed.emit()
