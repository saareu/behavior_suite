"""Trim and typed pre-crop configuration page."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from preprocess.exceptions import PreprocessError
from preprocess.pre_crop import PreCropMode
from ui.controllers.preprocess_setup_controller import (
    SUPPORTED_RAW_FRAME_JUMP_STEPS,
    PreprocessSetupController,
    RawFrameReadResult,
    SetupValidationError,
)
from ui.tasks import GuiTaskContext, GuiTaskRunner, TaskAlreadyRunningError
from ui.widgets.video_frame_view import PreCropOverlay, VideoFrameView

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
        self.task_runner = GuiTaskRunner(self)
        self._refreshing = False
        self._pending_slider_frame_idx: int | None = None

        title = QLabel("Trim and Pre-Crop")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        viewer_group = QGroupBox("Raw decode-frame navigator")
        self.frame_view = VideoFrameView()
        self.raw_frame_status = QLabel("Raw frame: —")
        self.raw_frame_status.setStyleSheet("font-family: monospace;")
        self.raw_frame_loading = QLabel()
        self.raw_frame_loading.setWordWrap(True)
        self.raw_frame_loading.setVisible(False)
        self.first_frame_button = QPushButton("First")
        self.previous_frame_button = QPushButton("Previous")
        self.next_frame_button = QPushButton("Next")
        self.step_backward_button = QPushButton("− Step")
        self.step_forward_button = QPushButton("+ Step")
        self.jump_step = QComboBox()
        for step in SUPPORTED_RAW_FRAME_JUMP_STEPS:
            self.jump_step.addItem(f"{step:,} frame" + ("" if step == 1 else "s"), step)
        self.exact_jump = QSpinBox()
        self.exact_jump.setRange(0, _MAX_FRAME_OR_COORDINATE)
        self.exact_jump.setGroupSeparatorShown(True)
        self.jump_button = QPushButton("Jump")
        self.frame_scrubber = QSlider(Qt.Orientation.Horizontal)
        self.frame_scrubber.setRange(0, 0)
        self.frame_scrubber.setEnabled(False)
        self.pending_frame_status = QLabel()
        self.pending_frame_status.setStyleSheet("font-family: monospace;")
        self.pending_frame_status.setVisible(False)
        nav_row = QHBoxLayout()
        nav_row.addWidget(self.first_frame_button)
        nav_row.addWidget(self.previous_frame_button)
        nav_row.addWidget(self.next_frame_button)
        nav_row.addWidget(self.jump_step)
        nav_row.addWidget(self.step_backward_button)
        nav_row.addWidget(self.step_forward_button)
        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Exact raw frame"))
        jump_row.addWidget(self.exact_jump, 1)
        jump_row.addWidget(self.jump_button)
        viewer_layout = QVBoxLayout(viewer_group)
        viewer_layout.addWidget(self.frame_view)
        viewer_layout.addWidget(self.raw_frame_status)
        viewer_layout.addWidget(self.frame_scrubber)
        viewer_layout.addWidget(self.pending_frame_status)
        viewer_layout.addWidget(self.raw_frame_loading)
        viewer_layout.addLayout(nav_row)
        viewer_layout.addLayout(jump_row)

        trim_group = QGroupBox("Decode-order frame selection")
        self.start_frame = QSpinBox()
        self.start_frame.setRange(0, _MAX_FRAME_OR_COORDINATE)
        self.end_frame = QLineEdit()
        self.end_frame.setPlaceholderText("blank = end of video")
        self.selection_semantics = QLabel()
        self.selection_semantics.setWordWrap(True)
        trim_button_row = QHBoxLayout()
        self.set_start_button = QPushButton("Set start to current frame")
        self.set_end_button = QPushButton("Set end exclusive to current frame")
        self.set_end_plus_one_button = QPushButton("Set end exclusive to current + 1")
        self.clear_start_button = QPushButton("Clear start")
        self.clear_end_button = QPushButton("Clear end")
        trim_button_row.addWidget(self.set_start_button)
        trim_button_row.addWidget(self.set_end_button)
        trim_button_row.addWidget(self.set_end_plus_one_button)
        trim_button_row.addWidget(self.clear_start_button)
        trim_button_row.addWidget(self.clear_end_button)
        trim_form = QFormLayout(trim_group)
        trim_form.addRow("Start frame (inclusive)", self.start_frame)
        trim_form.addRow("End frame (exclusive, optional)", self.end_frame)
        trim_form.addRow(self.selection_semantics)
        trim_form.addRow(trim_button_row)

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

        self.disable_pre_crop_button = QPushButton("Disable pre-crop")
        self.reset_pre_crop_button = QPushButton("Reset pre-crop to full raw frame")
        reset_row = QHBoxLayout()
        reset_row.addWidget(self.disable_pre_crop_button)
        reset_row.addWidget(self.reset_pre_crop_button)
        self.pre_crop_visual_hint = QLabel(
            "Visual edits use raw decode-frame coordinates. Disable stores mode "
            "'none'; reset stores an active full-frame rectangle/span when possible."
        )
        self.pre_crop_visual_hint.setWordWrap(True)
        self.resolved_roi = QLabel("Not resolved")
        self.resolved_roi.setStyleSheet("font-family: monospace;")
        pre_crop_layout = QVBoxLayout(pre_crop_group)
        pre_crop_layout.addWidget(QLabel("Mode"))
        pre_crop_layout.addWidget(self.mode)
        pre_crop_layout.addLayout(boundary_row)
        pre_crop_layout.addWidget(self.rectangle_widget)
        pre_crop_layout.addLayout(reset_row)
        pre_crop_layout.addWidget(self.pre_crop_visual_hint)
        pre_crop_layout.addWidget(QLabel("Resolved ROI (x, y, width, height)"))
        pre_crop_layout.addWidget(self.resolved_roi)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(viewer_group)
        layout.addWidget(trim_group)
        layout.addWidget(pre_crop_group)
        layout.addWidget(self.error_label)
        layout.addStretch(1)

        self.start_frame.valueChanged.connect(self._inputs_changed)
        self.end_frame.textChanged.connect(self._inputs_changed)
        self.first_frame_button.clicked.connect(self._go_first_frame)
        self.previous_frame_button.clicked.connect(self._go_previous_frame)
        self.next_frame_button.clicked.connect(self._go_next_frame)
        self.step_backward_button.clicked.connect(self._step_backward)
        self.step_forward_button.clicked.connect(self._step_forward)
        self.jump_button.clicked.connect(self._jump_to_exact_frame)
        self.jump_step.currentIndexChanged.connect(self._jump_step_changed)
        self.frame_scrubber.sliderMoved.connect(self._raw_frame_scrubber_moved)
        self.frame_scrubber.sliderReleased.connect(self._raw_frame_scrubber_released)
        self.frame_view.pre_crop_geometry_changed.connect(
            self._visual_pre_crop_changed
        )
        self.set_start_button.clicked.connect(self._set_start_to_current)
        self.set_end_button.clicked.connect(self._set_end_to_current)
        self.set_end_plus_one_button.clicked.connect(self._set_end_to_current_plus_one)
        self.clear_start_button.clicked.connect(self._clear_start)
        self.clear_end_button.clicked.connect(self._clear_end)
        self.disable_pre_crop_button.clicked.connect(self._disable_pre_crop)
        self.reset_pre_crop_button.clicked.connect(self._reset_pre_crop_to_full_frame)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.boundary.valueChanged.connect(self._inputs_changed)
        for spin_box in self.rectangle_inputs.values():
            spin_box.valueChanged.connect(self._inputs_changed)
        self.refresh_from_state()

    def on_activated(self) -> None:
        """Load config defaults/current state and immediately validate them."""

        self.refresh_from_state()
        self._validate_inputs()
        self._load_initial_frame_if_needed()

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
        self._refresh_pre_crop_overlay()
        self._refresh_navigation_controls()

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
            selection = f"Selected trim: [{start}, {end_text})"
            try:
                frame_count = int(end_text) - start
            except ValueError:
                frame_count = None
            if frame_count is not None and frame_count > 0:
                selection += f" ({frame_count:,} frames)"
        else:
            selection = f"Selected trim: [{start}, end of video)"
        self.selection_semantics.setText(
            selection
            + "\nStart frame is included. End frame is excluded from preprocessing."
        )

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
            self._refresh_pre_crop_overlay()
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
        self._refresh_pre_crop_overlay()

    def _refresh_pre_crop_overlay(self) -> None:
        config = self.controller.state.pre_crop_config
        resolved = self.controller.state.resolved_pre_crop
        mode = config.mode if config is not None else str(self.mode.currentData())
        boundary = config.boundary_px if config is not None else None
        if boundary is None and PreCropMode(mode) not in {
            PreCropMode.NONE,
            PreCropMode.MANUAL_RECTANGLE,
        }:
            boundary = self.boundary.value()
        roi = None
        if resolved is not None:
            roi = (
                resolved.roi.x,
                resolved.roi.y,
                resolved.roi.width,
                resolved.roi.height,
            )
        self.frame_view.set_pre_crop_overlay(
            PreCropOverlay(
                mode=mode,
                roi=roi,
                boundary_px=boundary,
            )
        )

    def _load_initial_frame_if_needed(self) -> None:
        state = self.controller.state
        if (
            state.raw_probe is None
            or state.raw_video_path is None
            or state.last_successfully_displayed_frame_idx is not None
            or state.raw_frame_navigation_in_progress
            or self.task_runner.is_running
        ):
            return
        self._load_raw_frame(state.current_raw_frame_idx or 0)

    def _go_first_frame(self) -> None:
        self._navigate_to(lambda: self.controller.first_raw_frame_index())

    def _go_previous_frame(self) -> None:
        self._navigate_to(lambda: self.controller.previous_raw_frame_index())

    def _go_next_frame(self) -> None:
        self._navigate_to(lambda: self.controller.next_raw_frame_index())

    def _step_backward(self) -> None:
        self._navigate_to(
            lambda: self.controller.raw_frame_index_after_step(
                direction=-1,
                step=self.controller.state.raw_frame_navigation_step,
            )
        )

    def _step_forward(self) -> None:
        self._navigate_to(
            lambda: self.controller.raw_frame_index_after_step(
                direction=1,
                step=self.controller.state.raw_frame_navigation_step,
            )
        )

    def _jump_to_exact_frame(self) -> None:
        self._navigate_to(
            lambda: self.controller.validate_raw_frame_index(self.exact_jump.value())
        )

    def _jump_step_changed(self) -> None:
        try:
            self.controller.set_raw_frame_navigation_step(int(self.jump_step.currentData()))
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
        self._refresh_navigation_controls()

    def _navigate_to(self, target_factory) -> None:
        try:
            target = target_factory()
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self._clear_pending_scrubber_target()
        self._load_raw_frame(target)

    def _raw_frame_scrubber_moved(self, value: int) -> None:
        self._pending_slider_frame_idx = int(value)
        self.pending_frame_status.setText(
            f"Pending raw frame: {int(value):,} (release to load)"
        )
        self.pending_frame_status.setVisible(True)

    def _raw_frame_scrubber_released(self) -> None:
        if self.controller.trusted_raw_frame_upper_bound() is None:
            self._clear_pending_scrubber_target()
            self._refresh_navigation_controls()
            return
        target = int(self.frame_scrubber.value())
        self._pending_slider_frame_idx = target
        self.pending_frame_status.setText(f"Pending raw frame: {target:,}")
        self.pending_frame_status.setVisible(True)
        self._load_raw_frame(target)

    def _load_raw_frame(self, raw_decode_frame_idx: int) -> None:
        context: GuiTaskContext | None = None
        try:
            request, context = self.controller.begin_raw_frame_read(raw_decode_frame_idx)
            self._set_navigation_busy(True, raw_decode_frame_idx)
            self.task_runner.start(
                lambda: self.controller.compute_raw_frame_read(request, context),
                task_name="raw frame read",
                context=context,
                on_success=lambda result: self._raw_frame_read_succeeded(result, context),
                on_error=lambda exc: self._raw_frame_read_failed(
                    exc,
                    context,
                    raw_decode_frame_idx,
                ),
                on_finished=lambda: self._raw_frame_read_finished(context),
            )
        except (SetupValidationError, TaskAlreadyRunningError) as exc:
            if context is not None and not self.task_runner.is_running:
                self.controller.finish_task(context)
            self._clear_pending_scrubber_target()
            self._set_navigation_busy(False)
            self._show_expected_error(str(exc))
        except Exception as exc:
            if context is not None and not self.task_runner.is_running:
                self.controller.finish_task(context)
            self._clear_pending_scrubber_target()
            self._set_navigation_busy(False)
            self._show_unexpected(exc)

    def _raw_frame_read_succeeded(
        self,
        result: object,
        context: GuiTaskContext,
    ) -> None:
        if not isinstance(result, RawFrameReadResult):
            self._show_unexpected(TypeError("Raw frame read returned an invalid result."))
            return
        try:
            frame = self.controller.apply_raw_frame_read(result, context=context)
        except (SetupValidationError, PreprocessError, OSError, ValueError) as exc:
            self._clear_pending_scrubber_target()
            self._show_expected_error(str(exc))
            return
        except Exception as exc:
            self._clear_pending_scrubber_target()
            self._show_unexpected(exc)
            return
        if frame is None:
            self._clear_pending_scrubber_target()
            self._refresh_navigation_controls()
            return
        self._clear_pending_scrubber_target()
        self.frame_view.set_frame(frame)
        self.error_label.clear()
        self.status_message.emit(
            f"Loaded raw decode frame {result.raw_decode_frame_idx:,}."
        )
        self._refresh_navigation_controls()

    def _raw_frame_read_failed(
        self,
        exc: BaseException,
        context: GuiTaskContext,
        requested_frame_idx: int,
    ) -> None:
        message = self.controller.record_raw_frame_read_failure(
            exc,
            context=context,
            requested_frame_idx=requested_frame_idx,
        )
        if message is not None:
            self._show_expected_error(message)
        self._clear_pending_scrubber_target()
        self._refresh_navigation_controls()

    def _raw_frame_read_finished(self, context: GuiTaskContext) -> None:
        self.controller.finish_task(context)
        self._set_navigation_busy(False)
        self._refresh_navigation_controls()

    def _set_navigation_busy(
        self,
        busy: bool,
        requested_frame_idx: int | None = None,
    ) -> None:
        if busy and requested_frame_idx is not None:
            self.raw_frame_loading.setText(
                f"Loading raw frame {requested_frame_idx:,}…"
            )
            self.raw_frame_loading.setVisible(True)
        else:
            self.raw_frame_loading.clear()
            self.raw_frame_loading.setVisible(False)
        self._refresh_navigation_controls()

    def _refresh_navigation_controls(self) -> None:
        state = self.controller.state
        has_probe = state.raw_probe is not None and state.raw_video_path is not None
        busy = state.raw_frame_navigation_in_progress or self.task_runner.is_running
        current = state.current_raw_frame_idx
        upper_bound = self.controller.trusted_raw_frame_upper_bound()
        if current is None:
            self.raw_frame_status.setText("Raw frame: —")
        else:
            if upper_bound is None:
                self.raw_frame_status.setText(f"Raw frame: {current:,}")
            else:
                self.raw_frame_status.setText(
                    f"Raw frame: {current:,} / {upper_bound - 1:,}"
                )
            if not self._refreshing:
                self.exact_jump.setValue(current)
        self.exact_jump.setMaximum(_MAX_FRAME_OR_COORDINATE)
        self._refresh_scrubber(upper_bound=upper_bound, busy=busy)
        step_index = self.jump_step.findData(state.raw_frame_navigation_step)
        if step_index >= 0 and self.jump_step.currentIndex() != step_index:
            self.jump_step.setCurrentIndex(step_index)
        can_navigate = has_probe and not busy
        at_first = current is None or current <= 0
        at_known_last = (
            upper_bound is not None
            and current is not None
            and current >= upper_bound - 1
        )
        self.first_frame_button.setEnabled(can_navigate and not at_first)
        self.previous_frame_button.setEnabled(can_navigate and not at_first)
        self.next_frame_button.setEnabled(can_navigate and not at_known_last)
        self.step_backward_button.setEnabled(can_navigate and not at_first)
        self.step_forward_button.setEnabled(can_navigate and not at_known_last)
        self.jump_button.setEnabled(can_navigate)
        self.exact_jump.setEnabled(can_navigate)
        self.jump_step.setEnabled(can_navigate)
        trim_buttons_enabled = (
            can_navigate and state.last_successfully_displayed_frame_idx is not None
        )
        self.set_start_button.setEnabled(trim_buttons_enabled)
        self.set_end_button.setEnabled(trim_buttons_enabled)
        self.set_end_plus_one_button.setEnabled(trim_buttons_enabled)
        self.clear_start_button.setEnabled(not busy and has_probe)
        self.clear_end_button.setEnabled(not busy and has_probe)
        if state.raw_frame_navigation_error:
            self.error_label.setText(state.raw_frame_navigation_error)

    def _refresh_scrubber(self, *, upper_bound: int | None, busy: bool) -> None:
        state = self.controller.state
        authoritative_count_available = upper_bound is not None and upper_bound > 0
        value = self._displayed_raw_frame_index()
        if (
            authoritative_count_available
            and busy
            and self._pending_slider_frame_idx is not None
        ):
            value = self._pending_slider_frame_idx
        self.frame_scrubber.blockSignals(True)
        try:
            if authoritative_count_available:
                self.frame_scrubber.setRange(0, upper_bound - 1)
                self.frame_scrubber.setValue(max(0, min(value or 0, upper_bound - 1)))
            else:
                self.frame_scrubber.setRange(0, 0)
                self.frame_scrubber.setValue(0)
        finally:
            self.frame_scrubber.blockSignals(False)
        self.frame_scrubber.setEnabled(authoritative_count_available and not busy)
        if self._pending_slider_frame_idx is None:
            self.pending_frame_status.setVisible(False)
            self.pending_frame_status.clear()
        elif not authoritative_count_available:
            self.pending_frame_status.setText(
                "Scrubber is disabled until a verified raw-frame count is available."
            )
            self.pending_frame_status.setVisible(True)
        elif state.raw_frame_navigation_in_progress:
            self.pending_frame_status.setText(
                f"Pending raw frame: {self._pending_slider_frame_idx:,}"
            )
            self.pending_frame_status.setVisible(True)

    def _displayed_raw_frame_index(self) -> int | None:
        state = self.controller.state
        if state.last_successfully_displayed_frame_idx is not None:
            return state.last_successfully_displayed_frame_idx
        return state.current_raw_frame_idx

    def _clear_pending_scrubber_target(self) -> None:
        self._pending_slider_frame_idx = None
        self.pending_frame_status.clear()
        self.pending_frame_status.setVisible(False)

    def _set_start_to_current(self) -> None:
        self._apply_trim_action(self.controller.set_start_frame_to_current)

    def _set_end_to_current(self) -> None:
        self._apply_trim_action(self.controller.set_end_frame_exclusive_to_current)

    def _set_end_to_current_plus_one(self) -> None:
        self._apply_trim_action(
            self.controller.set_end_frame_exclusive_to_current_plus_one
        )

    def _clear_start(self) -> None:
        self._apply_trim_action(self.controller.clear_start_frame_selection)

    def _clear_end(self) -> None:
        self._apply_trim_action(self.controller.clear_end_frame_selection)

    def _apply_trim_action(self, action) -> None:
        try:
            action()
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        except Exception as exc:
            self._show_unexpected(exc)
            self.refresh_from_state()
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Trim and pre-crop are valid.")
        self.validity_changed.emit()

    def _visual_pre_crop_changed(self, mode_value: str, geometry: object) -> None:
        try:
            mode = PreCropMode(mode_value)
        except ValueError:
            self._show_expected_error(f"Unsupported visual pre-crop mode: {mode_value}")
            return
        if mode is PreCropMode.MANUAL_RECTANGLE:
            if not (
                isinstance(geometry, tuple)
                and len(geometry) == 4
                and all(isinstance(value, int) for value in geometry)
            ):
                self._show_expected_error("Visual rectangle edit returned invalid geometry.")
                return
            self._apply_pre_crop_controls(mode=mode, rectangle=geometry)
            return
        if not isinstance(geometry, int):
            self._show_expected_error("Visual split edit returned invalid geometry.")
            return
        self._apply_pre_crop_controls(mode=mode, boundary_px=geometry)

    def _disable_pre_crop(self) -> None:
        self._apply_pre_crop_controls(mode=PreCropMode.NONE)

    def _reset_pre_crop_to_full_frame(self) -> None:
        raw_probe = self.controller.state.raw_probe
        if raw_probe is None:
            self._show_expected_error("Probe a raw video before resetting pre-crop.")
            return
        mode = PreCropMode(self.mode.currentData())
        if mode is PreCropMode.VERTICAL_KEEP_LEFT:
            self._apply_pre_crop_controls(mode=mode, boundary_px=raw_probe.width)
        elif mode is PreCropMode.VERTICAL_KEEP_RIGHT:
            self._apply_pre_crop_controls(mode=mode, boundary_px=0)
        elif mode is PreCropMode.HORIZONTAL_KEEP_UPPER:
            self._apply_pre_crop_controls(mode=mode, boundary_px=raw_probe.height)
        elif mode is PreCropMode.HORIZONTAL_KEEP_LOWER:
            self._apply_pre_crop_controls(mode=mode, boundary_px=0)
        else:
            self._apply_pre_crop_controls(
                mode=PreCropMode.MANUAL_RECTANGLE,
                rectangle=(0, 0, raw_probe.width, raw_probe.height),
            )

    def _apply_pre_crop_controls(
        self,
        *,
        mode: PreCropMode,
        boundary_px: int | None = None,
        rectangle: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._refreshing = True
        try:
            index = self.mode.findData(mode.value)
            self.mode.setCurrentIndex(max(index, 0))
            if boundary_px is not None:
                self.boundary.setValue(boundary_px)
            if rectangle is not None:
                for name, value in zip(
                    ("x", "y", "width", "height"),
                    rectangle,
                    strict=True,
                ):
                    self.rectangle_inputs[name].setValue(value)
        finally:
            self._refreshing = False
        self._update_control_visibility()
        self._update_semantics()
        self._validate_inputs()

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_message.emit(message)
        self.validity_changed.emit()

    def _show_unexpected(self, exc: BaseException) -> None:
        self.controller.record_unexpected_error(exc)
        self.error_label.setText("An unexpected error occurred.")
        self.unexpected_error.emit(str(exc))
        self.validity_changed.emit()
