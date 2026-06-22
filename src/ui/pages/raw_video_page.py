"""Raw-video selection and typed probe summary page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from preprocess.exceptions import PreprocessError
from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)
from ui.tasks import GuiTaskRunner, TaskAlreadyRunningError


class RawVideoPage(QWidget):
    """Select and probe a raw video without displaying frames."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)

    def __init__(self, controller: PreprocessSetupController) -> None:
        super().__init__()
        self.controller = controller
        self.task_runner = GuiTaskRunner(self)
        title = QLabel("Raw Video")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.video_path = QLineEdit()
        self.video_path.textEdited.connect(self._video_path_edited)
        self.browse_button = QPushButton("Browse for Raw Video")
        self.browse_button.clicked.connect(self._browse_video)
        path_row = QHBoxLayout()
        path_row.addWidget(self.video_path, 1)
        path_row.addWidget(self.browse_button)

        self.full_count = QCheckBox("Perform full sequential readable-frame count")
        warning = QLabel(
            "Warning: full sequential counting may take substantial time for a long video."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #8a5a00;")
        self.probe_button = QPushButton("Probe Selected Video")
        self.probe_button.clicked.connect(self._probe_video)
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setVisible(False)

        summary = QGroupBox("Probe summary")
        self.summary_values = {
            "source": QLabel("—"),
            "size": QLabel("—"),
            "reported": QLabel("—"),
            "readable": QLabel("Not counted"),
            "ffprobe_count": QLabel("—"),
            "fps": QLabel("—"),
            "codec": QLabel("—"),
            "duration": QLabel("—"),
        }
        summary_form = QFormLayout(summary)
        summary_form.addRow("Source path", self.summary_values["source"])
        summary_form.addRow("Dimensions", self.summary_values["size"])
        summary_form.addRow("OpenCV-reported frames", self.summary_values["reported"])
        summary_form.addRow("OpenCV-readable frames", self.summary_values["readable"])
        summary_form.addRow("ffprobe frames", self.summary_values["ffprobe_count"])
        summary_form.addRow("FPS", self.summary_values["fps"])
        summary_form.addRow("Codec", self.summary_values["codec"])
        summary_form.addRow("Duration", self.summary_values["duration"])

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(path_row)
        layout.addWidget(self.full_count)
        layout.addWidget(warning)
        layout.addWidget(self.probe_button)
        layout.addWidget(self.busy_indicator)
        layout.addWidget(summary)
        layout.addWidget(self.error_label)
        layout.addStretch(1)

    def is_valid(self) -> bool:
        """Return whether a raw video was successfully probed."""

        return self.controller.state.raw_probe is not None

    def commit(self) -> bool:
        """Validate that raw-video setup permits forward navigation."""

        try:
            self.controller.validate_step(1)
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
            return False
        return True

    def on_activated(self) -> None:
        """Refresh the selected path and any existing probe summary."""

        state = self.controller.state
        if state.raw_video_path is not None:
            self.video_path.setText(str(state.raw_video_path))
        self._refresh_summary()

    def _browse_video(self) -> None:
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose raw video",
            str(Path.cwd()),
            "Video files (*.avi *.mp4 *.mov *.mkv);;All files (*)",
        )
        if filename:
            self.video_path.setText(filename)
            self.controller.set_raw_video_path(Path(filename))
            self._refresh_summary()
            self.validity_changed.emit()

    def _video_path_edited(self, text: str) -> None:
        if text.strip():
            self.controller.set_raw_video_path(Path(text))
        else:
            self.controller.clear_raw_video_path()
        self._refresh_summary()
        self.validity_changed.emit()

    def _probe_video(self) -> None:
        if self.full_count.isChecked():
            self._probe_video_in_background()
            return
        try:
            self.controller.probe_raw_video(
                Path(self.video_path.text()),
                require_sequential_count=self.full_count.isChecked(),
            )
            self.error_label.clear()
            self._refresh_summary()
            self.status_message.emit("Raw video probe completed.")
            self.validity_changed.emit()
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._show_unexpected(exc)

    def _probe_video_in_background(self) -> None:
        try:
            path = self.controller.prepare_raw_video_probe(Path(self.video_path.text()))
            self._set_busy(True)
            self.task_runner.start(
                lambda: self.controller.compute_raw_video_probe(path, True),
                task_name="full raw-video probe",
                on_success=self._raw_probe_succeeded,
                on_error=self._raw_probe_failed,
                on_finished=lambda: self._set_busy(False),
            )
        except (SetupValidationError, TaskAlreadyRunningError) as exc:
            self._set_busy(False)
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._set_busy(False)
            self._show_unexpected(exc)

    def _raw_probe_succeeded(self, result: object) -> None:
        try:
            self.controller.apply_raw_video_probe(result)
        except Exception as exc:
            self._show_unexpected(exc)
            return
        self.error_label.clear()
        self._refresh_summary()
        self.status_message.emit("Raw video probe and sequential count completed.")
        self.validity_changed.emit()

    def _raw_probe_failed(self, exc: BaseException) -> None:
        message = self.controller.record_raw_video_probe_failure(exc)
        if isinstance(exc, (PreprocessError, OSError, ValueError)):
            self._show_expected_error(message)
        else:
            self._show_unexpected(exc)

    def _set_busy(self, busy: bool) -> None:
        self.busy_indicator.setVisible(busy)
        self.browse_button.setEnabled(not busy)
        self.probe_button.setEnabled(not busy)
        self.full_count.setEnabled(not busy)
        self.video_path.setEnabled(not busy)

    def _refresh_summary(self) -> None:
        probe = self.controller.state.raw_probe
        if probe is None:
            for key, label in self.summary_values.items():
                label.setText("Not counted" if key == "readable" else "—")
            return
        fps = probe.raw_fps_effective or probe.opencv_fps
        self.summary_values["source"].setText(str(probe.source_path))
        self.summary_values["size"].setText(f"{probe.width} × {probe.height}")
        self.summary_values["reported"].setText(str(probe.frame_count_opencv_reported))
        self.summary_values["readable"].setText(
            str(probe.frame_count_opencv_readable)
            if probe.frame_count_opencv_readable is not None
            else "Not counted"
        )
        self.summary_values["ffprobe_count"].setText(
            str(probe.frame_count_ffprobe)
            if probe.frame_count_ffprobe is not None
            else "Unavailable"
        )
        self.summary_values["fps"].setText(f"{fps:.6g}" if fps is not None else "Unavailable")
        self.summary_values["codec"].setText(probe.codec or "Unavailable")
        self.summary_values["duration"].setText(
            f"{probe.duration_sec:.3f} s" if probe.duration_sec is not None else "Unavailable"
        )

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_message.emit(message)
        self.validity_changed.emit()

    def _show_unexpected(self, exc: BaseException) -> None:
        self.controller.record_unexpected_error(exc)
        self.error_label.setText("An unexpected error occurred.")
        self.unexpected_error.emit(str(exc))
