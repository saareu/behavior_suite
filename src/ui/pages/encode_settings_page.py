"""Validated canonical and encoding settings page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.controllers.encode_settings_controller import (
    EncodeSettingsController,
    EncodeSettingsValidationError,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController

_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)
_MAX_DIMENSION = 2_147_483_647


class EncodeSettingsPage(QWidget):
    """Expose only config options already accepted by the core models."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)

    def __init__(self, setup_controller: PreprocessSetupController) -> None:
        super().__init__()
        self.setup_controller = setup_controller
        self.controller = EncodeSettingsController(setup_controller.state)
        self._refreshing = False

        title = QLabel("Encode Settings")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        explanation = QLabel(
            "Canonical resolution preserves aspect ratio through uniform scaling "
            "and centered black padding."
        )
        explanation.setWordWrap(True)
        warning = QLabel(
            "Changing encode settings after crop acceptance invalidates the accepted "
            "crop only when canonical geometry changes."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #8a5a00;")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self._build_canonical_group())
        content_layout.addWidget(self._build_ffmpeg_group())
        content_layout.addWidget(self._build_opencv_group())

        debug_group = QGroupBox("Debug")
        self.debug_enabled = QCheckBox("Retain internal debug artifacts")
        debug_layout = QVBoxLayout(debug_group)
        debug_layout.addWidget(self.debug_enabled)
        content_layout.addWidget(debug_group)

        self.reset_button = QPushButton("Reset to loaded config")
        self.reset_button.clicked.connect(self._reset_to_loaded)
        content_layout.addWidget(self.reset_button)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        self.dirty_label = QLabel()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #176b2c;")
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addWidget(warning)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.dirty_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.error_label)

        for widget in (
            self.canonical_enabled,
            self.canonical_width,
            self.canonical_height,
            self.ffmpeg_container,
            self.ffmpeg_codec,
            self.ffmpeg_pixel_format,
            self.ffmpeg_preset,
            self.ffmpeg_crf,
            self.ffmpeg_bframes,
            self.ffmpeg_faststart,
            self.opencv_container,
            self.opencv_fourcc,
            self.debug_enabled,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._settings_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._settings_changed)
            else:
                widget.valueChanged.connect(self._settings_changed)
        self.refresh_from_state()

    def _build_canonical_group(self) -> QGroupBox:
        group = QGroupBox("Canonical resolution")
        self.canonical_enabled = QCheckBox("Enabled")
        self.canonical_width = QSpinBox()
        self.canonical_width.setRange(1, _MAX_DIMENSION)
        self.canonical_height = QSpinBox()
        self.canonical_height.setRange(1, _MAX_DIMENSION)
        form = QFormLayout(group)
        form.addRow(self.canonical_enabled)
        form.addRow("Width", self.canonical_width)
        form.addRow("Height", self.canonical_height)
        return group

    def _build_ffmpeg_group(self) -> QGroupBox:
        group = QGroupBox("ffmpeg Stage A")
        self.ffmpeg_container = self._combo(("mp4",))
        self.ffmpeg_codec = self._combo(("libx264",))
        self.ffmpeg_pixel_format = self._combo(("yuv420p",))
        self.ffmpeg_preset = self._combo(_PRESETS)
        self.ffmpeg_crf = QSpinBox()
        self.ffmpeg_crf.setRange(0, 51)
        self.ffmpeg_bframes = QSpinBox()
        self.ffmpeg_bframes.setRange(0, 0)
        self.ffmpeg_bframes.setEnabled(False)
        self.ffmpeg_faststart = QCheckBox("Enabled")
        self.ffmpeg_faststart.setEnabled(False)
        form = QFormLayout(group)
        form.addRow("Container", self.ffmpeg_container)
        form.addRow("Codec", self.ffmpeg_codec)
        form.addRow("Pixel format", self.ffmpeg_pixel_format)
        form.addRow("Preset", self.ffmpeg_preset)
        form.addRow("CRF", self.ffmpeg_crf)
        form.addRow("B-frames", self.ffmpeg_bframes)
        form.addRow("Faststart", self.ffmpeg_faststart)
        return group

    def _build_opencv_group(self) -> QGroupBox:
        group = QGroupBox("OpenCV Stage B")
        self.opencv_container = self._combo(("mp4",))
        self.opencv_fourcc = self._combo(("mp4v",))
        form = QFormLayout(group)
        form.addRow("Container", self.opencv_container)
        form.addRow("FourCC", self.opencv_fourcc)
        return group

    @staticmethod
    def _combo(values: tuple[str, ...]) -> QComboBox:
        combo = QComboBox()
        for value in values:
            combo.addItem(value, value)
        return combo

    def on_activated(self) -> None:
        """Refresh current typed settings when the page becomes active."""

        self.refresh_from_state()

    def is_valid(self) -> bool:
        """Return whether settings and crop acceptance permit Run navigation."""

        return self.controller.can_advance()

    def commit(self) -> bool:
        """Block Next unless current settings validate and crop remains accepted."""

        try:
            self.controller.validate_navigation()
        except EncodeSettingsValidationError as exc:
            self._show_expected_error(str(exc))
            return False
        return True

    def refresh_from_state(self) -> None:
        """Render the active typed config without touching the source YAML."""

        config = self.controller.state.preprocess_config
        self._refreshing = True
        if config is not None:
            canonical = config.prepare.canonical_resolution
            ffmpeg = config.encoding.ffmpeg
            opencv = config.encoding.opencv
            self.canonical_enabled.setChecked(canonical.enabled)
            self.canonical_width.setValue(canonical.width)
            self.canonical_height.setValue(canonical.height)
            self._set_combo(self.ffmpeg_container, ffmpeg.container)
            self._set_combo(self.ffmpeg_codec, ffmpeg.codec)
            self._set_combo(self.ffmpeg_pixel_format, ffmpeg.pixel_format)
            self._set_combo(self.ffmpeg_preset, ffmpeg.preset)
            self.ffmpeg_crf.setValue(ffmpeg.crf)
            self.ffmpeg_bframes.setValue(ffmpeg.bframes)
            self.ffmpeg_faststart.setChecked(ffmpeg.faststart)
            self._set_combo(self.opencv_container, opencv.container)
            self._set_combo(self.opencv_fourcc, opencv.fourcc)
            self.debug_enabled.setChecked(config.debug.enabled)
        self.canonical_width.setEnabled(self.canonical_enabled.isChecked())
        self.canonical_height.setEnabled(self.canonical_enabled.isChecked())
        self._refreshing = False
        self.dirty_label.setText(
            "Configuration has unsaved GUI edits."
            if self.controller.state.config_dirty
            else "Configuration matches the loaded settings."
        )
        accepted = self.controller.state.accepted_crop_plan
        if accepted is None or not accepted.accepted_by_user:
            self.status_label.setText(
                "Crop review and explicit acceptance are required before Run and Validate."
            )
        else:
            self.status_label.setText("Encode settings are valid and the crop is accepted.")
        self.validity_changed.emit()

    def _settings_changed(self) -> None:
        if self._refreshing:
            return
        self.canonical_width.setEnabled(self.canonical_enabled.isChecked())
        self.canonical_height.setEnabled(self.canonical_enabled.isChecked())
        try:
            self.controller.update_settings(
                canonical_enabled=self.canonical_enabled.isChecked(),
                canonical_width=self.canonical_width.value(),
                canonical_height=self.canonical_height.value(),
                ffmpeg_container=self.ffmpeg_container.currentData(),
                ffmpeg_codec=self.ffmpeg_codec.currentData(),
                ffmpeg_pixel_format=self.ffmpeg_pixel_format.currentData(),
                ffmpeg_preset=self.ffmpeg_preset.currentData(),
                ffmpeg_crf=self.ffmpeg_crf.value(),
                ffmpeg_bframes=self.ffmpeg_bframes.value(),
                ffmpeg_faststart=self.ffmpeg_faststart.isChecked(),
                opencv_container=self.opencv_container.currentData(),
                opencv_fourcc=self.opencv_fourcc.currentData(),
                debug_enabled=self.debug_enabled.isChecked(),
            )
        except EncodeSettingsValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Encode settings updated and validated.")

    def _reset_to_loaded(self) -> None:
        try:
            self.controller.reset_to_loaded_config()
        except EncodeSettingsValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Encode settings reset to the loaded configuration.")

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(max(index, 0))

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.status_label.clear()
        self.error_message.emit(message)
        self.validity_changed.emit()
