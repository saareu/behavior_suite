"""Trim and typed pre-crop configuration page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from preprocess.pre_crop import PreCropMode
from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)

_MAX_FRAME_OR_COORDINATE = 2_147_483_647


class TrimPreCropPage(QWidget):
    """Collect trim/pre-crop values and display the core-resolved ROI."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)

    def __init__(self, controller: PreprocessSetupController) -> None:
        super().__init__()
        self.controller = controller
        self._refreshing = False

        title = QLabel("Trim and Pre-Crop")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        trim_group = QGroupBox("Decode-order frame selection")
        self.start_frame = QSpinBox()
        self.start_frame.setRange(0, _MAX_FRAME_OR_COORDINATE)
        self.end_frame = QLineEdit()
        self.end_frame.setPlaceholderText("blank = end of video")
        self.selection_semantics = QLabel()
        trim_form = QFormLayout(trim_group)
        trim_form.addRow("Start frame (inclusive)", self.start_frame)
        trim_form.addRow("End frame (exclusive, optional)", self.end_frame)
        trim_form.addRow(self.selection_semantics)

        pre_crop_group = QGroupBox("Detection pre-crop")
        self.mode = QComboBox()
        for mode in PreCropMode:
            self.mode.addItem(mode.value, mode.value)
        self.boundary = QSpinBox()
        self.boundary.setRange(0, _MAX_FRAME_OR_COORDINATE)
        self.boundary_label = QLabel("Boundary")
        boundary_row = QHBoxLayout()
        boundary_row.addWidget(self.boundary_label)
        boundary_row.addWidget(self.boundary, 1)

        self.rectangle_inputs: dict[str, QSpinBox] = {}
        rectangle_layout = QFormLayout()
        for name in ("x", "y", "width", "height"):
            spin_box = QSpinBox()
            spin_box.setRange(0, _MAX_FRAME_OR_COORDINATE)
            self.rectangle_inputs[name] = spin_box
            rectangle_layout.addRow(name, spin_box)
        self.rectangle_widget = QWidget()
        self.rectangle_widget.setLayout(rectangle_layout)

        self.resolved_roi = QLabel("Not resolved")
        self.resolved_roi.setStyleSheet("font-family: monospace;")
        pre_crop_layout = QVBoxLayout(pre_crop_group)
        pre_crop_layout.addWidget(QLabel("Mode"))
        pre_crop_layout.addWidget(self.mode)
        pre_crop_layout.addLayout(boundary_row)
        pre_crop_layout.addWidget(self.rectangle_widget)
        pre_crop_layout.addWidget(QLabel("Resolved ROI (x, y, width, height)"))
        pre_crop_layout.addWidget(self.resolved_roi)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(trim_group)
        layout.addWidget(pre_crop_group)
        layout.addWidget(self.error_label)
        layout.addStretch(1)

        self.start_frame.valueChanged.connect(self._inputs_changed)
        self.end_frame.textChanged.connect(self._inputs_changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.boundary.valueChanged.connect(self._inputs_changed)
        for spin_box in self.rectangle_inputs.values():
            spin_box.valueChanged.connect(self._inputs_changed)
        self.refresh_from_state()

    def on_activated(self) -> None:
        """Load config defaults/current state and immediately validate them."""

        self.refresh_from_state()
        self._validate_inputs()

    def refresh_from_state(self) -> None:
        """Refresh controls from typed setup state."""

        self._refreshing = True
        state = self.controller.state
        self.start_frame.setValue(state.start_frame or 0)
        self.end_frame.setText(
            "" if state.end_frame_exclusive is None else str(state.end_frame_exclusive)
        )
        mode = state.pre_crop_config.mode if state.pre_crop_config is not None else "none"
        index = self.mode.findData(mode)
        self.mode.setCurrentIndex(max(index, 0))
        if state.pre_crop_config is not None:
            if state.pre_crop_config.boundary_px is not None:
                self.boundary.setValue(state.pre_crop_config.boundary_px)
            if state.pre_crop_config.manual_rectangle is not None:
                for name, value in zip(
                    ("x", "y", "width", "height"),
                    state.pre_crop_config.manual_rectangle,
                    strict=True,
                ):
                    self.rectangle_inputs[name].setValue(value)
        self._refreshing = False
        self._update_control_visibility()
        self._update_semantics()
        self._refresh_resolved_roi()

    def is_valid(self) -> bool:
        """Return whether trim and pre-crop are currently valid."""

        return self.controller.state.trim_pre_crop_valid

    def commit(self) -> bool:
        """Revalidate current inputs before navigation."""

        return self._validate_inputs()

    def _mode_changed(self) -> None:
        self._update_control_visibility()
        self._inputs_changed()

    def _update_control_visibility(self) -> None:
        mode = PreCropMode(self.mode.currentData())
        directional = mode not in {PreCropMode.NONE, PreCropMode.MANUAL_RECTANGLE}
        self.boundary.setVisible(directional)
        self.boundary_label.setVisible(directional)
        self.rectangle_widget.setVisible(mode is PreCropMode.MANUAL_RECTANGLE)

    def _inputs_changed(self) -> None:
        if self._refreshing:
            return
        self._update_semantics()
        self._validate_inputs()

    def _end_frame_value(self) -> int | None:
        text = self.end_frame.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise SetupValidationError("End frame must be an integer or left blank.") from exc

    def _update_semantics(self) -> None:
        start = self.start_frame.value()
        end_text = self.end_frame.text().strip()
        if end_text:
            self.selection_semantics.setText(f"Selected raw frames: [{start}, {end_text})")
        else:
            self.selection_semantics.setText(f"Selected raw frames: [{start}, end of video)")

    def _validate_inputs(self) -> bool:
        try:
            mode = PreCropMode(self.mode.currentData())
            resolved = self.controller.configure_trim_and_pre_crop(
                start_frame=self.start_frame.value(),
                end_frame_exclusive=self._end_frame_value(),
                mode=mode,
                boundary_px=self.boundary.value(),
                rectangle_x=self.rectangle_inputs["x"].value(),
                rectangle_y=self.rectangle_inputs["y"].value(),
                rectangle_width=self.rectangle_inputs["width"].value(),
                rectangle_height=self.rectangle_inputs["height"].value(),
            )
            self.error_label.clear()
            self.resolved_roi.setText(
                f"{resolved.roi.x}, {resolved.roi.y}, {resolved.roi.width}, {resolved.roi.height}"
            )
            self.status_message.emit("Trim and pre-crop are valid.")
            self.validity_changed.emit()
            return True
        except SetupValidationError as exc:
            self.error_label.setText(str(exc))
            self.resolved_roi.setText("Not resolved")
            self.error_message.emit(str(exc))
            self.validity_changed.emit()
            return False
        except Exception as exc:
            self.controller.record_unexpected_error(exc)
            self.error_label.setText("An unexpected error occurred.")
            self.unexpected_error.emit(str(exc))
            self.validity_changed.emit()
            return False

    def _refresh_resolved_roi(self) -> None:
        resolved = self.controller.state.resolved_pre_crop
        if resolved is None:
            self.resolved_roi.setText("Not resolved")
        else:
            roi = resolved.roi
            self.resolved_roi.setText(f"{roi.x}, {roi.y}, {roi.width}, {roi.height}")
