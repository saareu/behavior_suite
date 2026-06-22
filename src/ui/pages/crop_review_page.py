"""Automatic/manual cage crop review with explicit acceptance."""

from __future__ import annotations

import cv2
from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from preprocess.exceptions import PreprocessError
from ui.controllers.crop_review_controller import (
    CropReviewComputation,
    CropReviewController,
    CropReviewValidationError,
)
from ui.controllers.preprocess_setup_controller import (
    PreprocessSetupController,
    SetupValidationError,
)
from ui.state import CropReviewMode, WorkflowStep
from ui.tasks import GuiTaskRunner, TaskAlreadyRunningError
from ui.widgets.crop_overlay_view import CropOverlayView
from ui.widgets.video_frame_view import VideoFrameView

_MAX_INT = 2_147_483_647


class CropReviewPage(QWidget):
    """Render CropPlan candidates while the controller owns all decisions."""

    validity_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)
    unexpected_error = Signal(str)

    def __init__(self, setup_controller: PreprocessSetupController) -> None:
        super().__init__()
        self.setup_controller = setup_controller
        self.controller = CropReviewController(setup_controller.state)
        self.task_runner = GuiTaskRunner(self)
        self._refreshing = False
        self._busy = False

        title = QLabel("Crop Review")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        explanation = QLabel(
            "Automatic and manual results remain candidates until the prepared "
            "preview is explicitly accepted."
        )
        explanation.setWordWrap(True)

        self.automatic_mode = QRadioButton("Automatic detection / retry")
        self.manual_mode = QRadioButton("Manual four-corner crop")
        self.automatic_mode.toggled.connect(self._mode_changed)
        self.manual_mode.toggled.connect(self._mode_changed)
        self.load_frame_button = QPushButton("Load Representative Frame")
        self.load_frame_button.clicked.connect(self._load_frame)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.automatic_mode)
        mode_row.addWidget(self.manual_mode)
        mode_row.addStretch(1)
        mode_row.addWidget(self.load_frame_button)

        self.raw_view = CropOverlayView()
        self.raw_view.raw_point_clicked.connect(self._manual_point_clicked)
        self.preview_view = VideoFrameView()
        raw_group = QGroupBox("Raw review frame")
        raw_layout = QVBoxLayout(raw_group)
        self.frame_status = QLabel("No representative frame loaded.")
        raw_layout.addWidget(self.frame_status)
        raw_layout.addWidget(self.raw_view, 1)
        preview_group = QGroupBox("Prepared crop preview (review only)")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview_view, 1)
        views = QSplitter(Qt.Orientation.Vertical)
        views.addWidget(raw_group)
        views.addWidget(preview_group)
        views.setSizes([360, 280])

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self._build_automatic_group())
        controls_layout.addWidget(self._build_manual_group())
        controls_layout.addWidget(self._build_diagnostics_group())
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setVisible(False)
        controls_layout.addWidget(self.busy_indicator)
        self.accept_button = QPushButton("Accept Crop")
        self.accept_button.clicked.connect(self._accept_crop)
        controls_layout.addWidget(self.accept_button)
        controls_layout.addStretch(1)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(345)
        controls_scroll.setMaximumWidth(430)
        controls_scroll.setWidget(controls)

        content = QHBoxLayout()
        content.addWidget(controls_scroll)
        content.addWidget(views, 1)

        self.success_label = QLabel()
        self.success_label.setWordWrap(True)
        self.success_label.setStyleSheet("color: #176b2c;")
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(explanation)
        layout.addLayout(mode_row)
        layout.addLayout(content, 1)
        layout.addWidget(self.success_label)
        layout.addWidget(self.error_label)
        self.refresh_from_state()

    def _build_automatic_group(self) -> QGroupBox:
        self.automatic_group = QGroupBox("Automatic detector settings")
        form = QFormLayout()
        self.sample_step = self._int_spin(1, _MAX_INT)
        self.pad_px = self._int_spin(0, 100_000)
        self.threshold = self._int_spin(0, 255)
        self.roi_margin_px = self._int_spin(0, 100_000)
        self.interpolation = QComboBox()
        self.interpolation.addItem("linear", "linear")
        self.interpolation.addItem("cubic", "cubic")
        self.canonical_enabled = QCheckBox("Use canonical resolution")
        self.canonical_width = self._int_spin(2, _MAX_INT, step=2)
        self.canonical_height = self._int_spin(2, _MAX_INT, step=2)
        form.addRow("Sample step", self.sample_step)
        form.addRow("Padding (px)", self.pad_px)
        form.addRow("Threshold", self.threshold)
        form.addRow("ROI margin (px)", self.roi_margin_px)
        form.addRow("Perspective interpolation", self.interpolation)
        form.addRow(self.canonical_enabled)
        form.addRow("Canonical width", self.canonical_width)
        form.addRow("Canonical height", self.canonical_height)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced detector settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.advanced_widget = QWidget()
        advanced_form = QFormLayout(self.advanced_widget)
        self.pre_crop_expansion = self._double_spin(0.0, 200.0)
        self.dilate_kernel = self._int_spin(1, 999, step=2)
        self.erode_kernel = self._int_spin(1, 999, step=2)
        self.rim_close_kernel = self._int_spin(1, 999, step=2)
        self.minimum_width_fraction = self._double_spin(0.001, 1.0, decimals=3)
        self.minimum_height_fraction = self._double_spin(0.001, 1.0, decimals=3)
        self.minimum_contour_area = self._double_spin(0.001, 1_000_000_000.0)
        self.fit_tolerance = self._int_spin(0, 100_000)
        advanced_form.addRow("Pre-crop expansion (%)", self.pre_crop_expansion)
        advanced_form.addRow("Dilate kernel", self.dilate_kernel)
        advanced_form.addRow("Erode kernel", self.erode_kernel)
        advanced_form.addRow("Rim close kernel", self.rim_close_kernel)
        advanced_form.addRow("Minimum width fraction", self.minimum_width_fraction)
        advanced_form.addRow("Minimum height fraction", self.minimum_height_fraction)
        advanced_form.addRow("Minimum contour area", self.minimum_contour_area)
        advanced_form.addRow("Fit tolerance (px)", self.fit_tolerance)
        self.advanced_widget.setVisible(False)

        self.find_button = QPushButton("Find Cage")
        self.retry_button = QPushButton("Retry")
        self.find_button.clicked.connect(self._run_detection)
        self.retry_button.clicked.connect(self._run_detection)
        actions = QHBoxLayout()
        actions.addWidget(self.find_button)
        actions.addWidget(self.retry_button)

        layout = QVBoxLayout(self.automatic_group)
        layout.addLayout(form)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_widget)
        layout.addLayout(actions)

        setting_widgets = (
            self.sample_step,
            self.pad_px,
            self.threshold,
            self.roi_margin_px,
            self.interpolation,
            self.canonical_enabled,
            self.canonical_width,
            self.canonical_height,
            self.pre_crop_expansion,
            self.dilate_kernel,
            self.erode_kernel,
            self.rim_close_kernel,
            self.minimum_width_fraction,
            self.minimum_height_fraction,
            self.minimum_contour_area,
            self.fit_tolerance,
        )
        for widget in setting_widgets:
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._settings_changed)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._settings_changed)
            else:
                widget.valueChanged.connect(self._settings_changed)
        return self.automatic_group

    def _build_manual_group(self) -> QGroupBox:
        self.manual_group = QGroupBox("Manual four-corner selection")
        instructions = QLabel(
            "Click exactly: top-left → top-right → bottom-right → bottom-left. "
            "Points are kept in raw-frame coordinates and are never reordered."
        )
        instructions.setWordWrap(True)
        self.next_point_label = QLabel()
        self.next_point_label.setStyleSheet("font-weight: 600;")
        self.clear_points_button = QPushButton("Clear / Reset Points")
        self.clear_points_button.clicked.connect(self._clear_manual_points)
        layout = QVBoxLayout(self.manual_group)
        layout.addWidget(instructions)
        layout.addWidget(self.next_point_label)
        layout.addWidget(self.clear_points_button)
        return self.manual_group

    def _build_diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("Candidate diagnostics")
        self.diagnostics = {
            "size": QLabel("—"),
            "fit": QLabel("—"),
            "rim": QLabel("—"),
            "rotation": QLabel("—"),
            "roi": QLabel("—"),
        }
        form = QFormLayout(group)
        form.addRow("Detected output size", self.diagnostics["size"])
        form.addRow("Fit score", self.diagnostics["fit"])
        form.addRow("Rim density", self.diagnostics["rim"])
        form.addRow("Rotation status", self.diagnostics["rotation"])
        form.addRow("Pre-crop ROI", self.diagnostics["roi"])
        return group

    @staticmethod
    def _int_spin(minimum: int, maximum: int, *, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        return spin

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        *,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        return spin

    def on_activated(self) -> None:
        """Synchronize upstream inputs and load the deterministic review frame."""

        self.controller.synchronize_upstream_context()
        self.refresh_from_state()
        if self.controller.representative_frame is None:
            self._load_frame()

    def is_valid(self) -> bool:
        """Return whether an explicitly accepted CropPlan enables navigation."""

        return self.controller.can_advance()

    def commit(self) -> bool:
        """Validate Crop Review navigation through shared setup state."""

        try:
            self.setup_controller.validate_step(WorkflowStep.CROP_REVIEW)
        except SetupValidationError as exc:
            self._show_expected_error(str(exc))
            return False
        return True

    def refresh_from_state(self) -> None:
        """Refresh settings, frames, overlays, diagnostics, and navigation state."""

        self._refreshing = True
        state = self.controller.state
        self.automatic_mode.setChecked(state.crop_mode is CropReviewMode.AUTOMATIC)
        self.manual_mode.setChecked(state.crop_mode is CropReviewMode.MANUAL)
        config = state.preprocess_config
        if config is not None:
            detector = config.cage_detect
            prepare = config.prepare
            canonical = prepare.canonical_resolution
            self.sample_step.setValue(detector.sample_step)
            self.pad_px.setValue(detector.pad_px)
            self.threshold.setValue(detector.threshold)
            self.roi_margin_px.setValue(prepare.roi_margin_px)
            self.interpolation.setCurrentIndex(
                max(self.interpolation.findData(prepare.perspective_interpolation), 0)
            )
            self.canonical_enabled.setChecked(canonical.enabled)
            self.canonical_width.setValue(canonical.width)
            self.canonical_height.setValue(canonical.height)
            self.pre_crop_expansion.setValue(detector.pre_crop_expansion_percent)
            self.dilate_kernel.setValue(detector.dilate_kernel_size)
            self.erode_kernel.setValue(detector.erode_kernel_size)
            self.rim_close_kernel.setValue(detector.rim_close_kernel_size)
            self.minimum_width_fraction.setValue(
                detector.minimum_cage_width_fraction
            )
            self.minimum_height_fraction.setValue(
                detector.minimum_cage_height_fraction
            )
            self.minimum_contour_area.setValue(detector.minimum_contour_area)
            self.fit_tolerance.setValue(detector.fit_tolerance_px)
        self.canonical_width.setEnabled(self.canonical_enabled.isChecked())
        self.canonical_height.setEnabled(self.canonical_enabled.isChecked())
        self._refreshing = False
        self.automatic_group.setEnabled(
            state.crop_mode is CropReviewMode.AUTOMATIC and not self._busy
        )
        self.manual_group.setEnabled(
            state.crop_mode is CropReviewMode.MANUAL and not self._busy
        )
        self.next_point_label.setText(
            f"Next required point: {self.controller.next_manual_point_label}"
        )
        self.raw_view.set_frame(self.controller.representative_frame)
        self.preview_view.set_frame(self.controller.prepared_preview)
        frame_index = self.controller.representative_frame_index
        self.frame_status.setText(
            "No representative frame loaded."
            if frame_index is None
            else f"Raw decode frame {frame_index}"
        )
        candidate = state.candidate_crop_plan
        pre_crop = state.resolved_pre_crop
        candidate_quad = None
        if candidate is not None and state.crop_mode is CropReviewMode.AUTOMATIC:
            candidate_quad = candidate.quad_raw_tl_tr_br_bl
        self.raw_view.set_overlays(
            pre_crop_roi=None if pre_crop is None else pre_crop.roi,
            candidate_quad=candidate_quad,
            manual_points=self.controller.manual_points,
            manual_input_enabled=(
                state.crop_mode is CropReviewMode.MANUAL
                and not self._busy
            ),
        )
        self._refresh_diagnostics()
        self.accept_button.setEnabled(
            candidate is not None
            and self.controller.prepared_preview is not None
            and not self._busy
        )
        if self.controller.can_advance():
            self.success_label.setText("Crop accepted. Ready for encode settings.")
        elif candidate is not None:
            self.success_label.setText(
                "Candidate preview ready. Review it, then press Accept Crop."
            )
        else:
            self.success_label.clear()
        self.validity_changed.emit()

    def _load_frame(self) -> None:
        try:
            self.controller.load_representative_frame()
            self.error_label.clear()
            self.refresh_from_state()
            self.status_message.emit("Representative raw frame loaded.")
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._show_unexpected(exc)

    def _run_detection(self) -> None:
        try:
            request = self.controller.begin_automatic_detection()
            self._set_busy(True)
            self.refresh_from_state()
            self.task_runner.start(
                lambda: self.controller.compute_automatic_detection(request),
                task_name="automatic cage detection",
                on_success=self._detection_succeeded,
                on_error=self._detection_failed,
                on_finished=lambda: self._set_busy(False),
            )
        except (CropReviewValidationError, TaskAlreadyRunningError) as exc:
            self._set_busy(False)
            self._show_expected_error(str(exc))
        except Exception as exc:
            self._set_busy(False)
            self._show_unexpected(exc)

    def _detection_succeeded(self, result: object) -> None:
        try:
            if not isinstance(result, CropReviewComputation):
                raise TypeError("Automatic detection returned an unexpected result type.")
            self.controller.apply_automatic_detection(result)
        except (CropReviewValidationError, ValueError) as exc:
            self._show_expected_error(str(exc))
            return
        except Exception as exc:
            self._show_unexpected(exc)
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit(
            "Automatic crop candidate is ready for review; it is not accepted."
        )

    def _detection_failed(self, exc: BaseException) -> None:
        self.controller.record_automatic_detection_failure(exc)
        if isinstance(
            exc,
            (PreprocessError, ValidationError, OSError, ValueError, cv2.error),
        ):
            self._show_expected_error(
                self.controller.state.last_validation_error
                or "Automatic cage detection failed."
            )
        else:
            self._show_unexpected(exc)
        self.refresh_from_state()

    def _settings_changed(self) -> None:
        if self._refreshing:
            return
        try:
            self.controller.update_detector_settings(
                sample_step=self.sample_step.value(),
                pad_px=self.pad_px.value(),
                threshold=self.threshold.value(),
                roi_margin_px=self.roi_margin_px.value(),
                perspective_interpolation=self.interpolation.currentData(),
                canonical_enabled=self.canonical_enabled.isChecked(),
                canonical_width=self.canonical_width.value(),
                canonical_height=self.canonical_height.value(),
                pre_crop_expansion_percent=self.pre_crop_expansion.value(),
                dilate_kernel_size=self.dilate_kernel.value(),
                erode_kernel_size=self.erode_kernel.value(),
                rim_close_kernel_size=self.rim_close_kernel.value(),
                minimum_cage_width_fraction=self.minimum_width_fraction.value(),
                minimum_cage_height_fraction=self.minimum_height_fraction.value(),
                minimum_contour_area=self.minimum_contour_area.value(),
                fit_tolerance_px=self.fit_tolerance.value(),
            )
            self.error_label.clear()
            self.canonical_width.setEnabled(self.canonical_enabled.isChecked())
            self.canonical_height.setEnabled(self.canonical_enabled.isChecked())
            self.refresh_from_state()
            self.status_message.emit("Detector settings changed; retry detection.")
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))

    def _mode_changed(self) -> None:
        if self._refreshing:
            return
        mode = (
            CropReviewMode.MANUAL
            if self.manual_mode.isChecked()
            else CropReviewMode.AUTOMATIC
        )
        try:
            self.controller.set_crop_mode(mode)
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self.error_label.clear()
        self.refresh_from_state()
        if mode is CropReviewMode.MANUAL:
            self.status_message.emit("Select four manual points in the prescribed order.")

    def _manual_point_clicked(self, x: float, y: float) -> None:
        try:
            plan = self.controller.add_manual_point(x, y)
            self.error_label.clear()
            self.refresh_from_state()
            if plan is None:
                self.status_message.emit(
                    f"Manual point stored; next: {self.controller.next_manual_point_label}."
                )
            else:
                self.status_message.emit(
                    "Manual crop candidate is ready for review; it is not accepted."
                )
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
        except Exception as exc:
            self._show_unexpected(exc)

    def _clear_manual_points(self) -> None:
        self.controller.clear_manual_points()
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Manual crop points cleared.")

    def _accept_crop(self) -> None:
        try:
            self.controller.accept_crop()
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Crop accepted. Ready for encode settings.")

    def _refresh_diagnostics(self) -> None:
        plan = self.controller.state.candidate_crop_plan
        if plan is None:
            for label in self.diagnostics.values():
                label.setText("—")
            return
        width, height = plan.prepared_size_wh
        roi = plan.pre_crop_roi
        self.diagnostics["size"].setText(f"{width} × {height}")
        self.diagnostics["fit"].setText(
            "n/a (manual)" if plan.fit_score is None else f"{plan.fit_score:.5g}"
        )
        self.diagnostics["rim"].setText(
            "n/a (manual)" if plan.rim_density is None else f"{plan.rim_density:.5g}"
        )
        self.diagnostics["rotation"].setText(
            "clockwise 90°" if plan.rotated_90 else "none"
        )
        self.diagnostics["roi"].setText(
            f"({roi.x}, {roi.y}, {roi.width}, {roi.height})"
        )

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.advanced_widget.setVisible(checked)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_indicator.setVisible(busy)
        self.load_frame_button.setEnabled(not busy)
        self.find_button.setEnabled(not busy)
        self.retry_button.setEnabled(not busy)
        self.automatic_mode.setEnabled(not busy)
        self.manual_mode.setEnabled(not busy)
        self.clear_points_button.setEnabled(not busy)
        self.accept_button.setEnabled(not busy and self.accept_button.isEnabled())
        self.raw_view.setEnabled(not busy)
        self.automatic_group.setEnabled(
            not busy
            and self.controller.state.crop_mode is CropReviewMode.AUTOMATIC
        )
        self.manual_group.setEnabled(
            not busy and self.controller.state.crop_mode is CropReviewMode.MANUAL
        )
        if not busy:
            self.refresh_from_state()

    def _show_expected_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.success_label.clear()
        self.error_message.emit(message)
        self.validity_changed.emit()

    def _show_unexpected(self, exc: BaseException) -> None:
        self.setup_controller.record_unexpected_error(exc)
        self.error_label.setText("An unexpected crop-review error occurred.")
        self.success_label.clear()
        self.unexpected_error.emit(str(exc))
        self.validity_changed.emit()
