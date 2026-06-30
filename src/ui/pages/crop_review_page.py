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

from preprocess.config import MaskRectangleConfig
from preprocess.exceptions import CageDetectionCancelledError, PreprocessError
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
from ui.tasks import (
    GuiTaskContext,
    GuiTaskKind,
    GuiTaskProgressEvent,
    GuiTaskRunner,
    TaskAlreadyRunningError,
)
from ui.widgets.crop_overlay_view import CropOverlayView
from ui.widgets.video_frame_view import MaskOverlay, VideoFrameView

_MAX_INT = 2_147_483_647
_DETECTOR_FIELD_LABELS = {
    "sample_step": "sample step",
    "pad_px": "padding",
    "threshold": "threshold",
    "pre_crop_expansion_percent": "pre-crop expansion",
    "dilate_kernel_size": "dilate kernel",
    "erode_kernel_size": "erode kernel",
    "rim_close_kernel_size": "rim close kernel",
    "minimum_cage_width_fraction": "minimum width fraction",
    "minimum_cage_height_fraction": "minimum height fraction",
    "minimum_contour_area": "minimum contour area",
    "fit_tolerance_px": "fit tolerance",
}


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
        self._selected_mask_shape_index: int | None = None
        self._mask_drawing_mode = "select"
        self._pending_polygon_vertices: list[tuple[int, int]] = []
        self.preview_view.mask_shape_selected.connect(self._select_mask_shape_from_view)
        self.preview_view.mask_rectangle_created.connect(self._add_mask_rectangle)
        self.preview_view.mask_rectangle_changed.connect(self._replace_mask_rectangle)
        self.preview_view.mask_polygon_vertex_added.connect(self._add_mask_polygon_vertex)
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
        controls_layout.addWidget(self._build_mask_group())
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setVisible(False)
        controls_layout.addWidget(self.busy_indicator)
        self.detection_progress_label = QLabel()
        self.detection_progress_label.setWordWrap(True)
        self.detection_progress_label.setVisible(False)
        controls_layout.addWidget(self.detection_progress_label)
        self.cancel_detection_button = QPushButton("Cancel Cage Detection")
        self.cancel_detection_button.clicked.connect(self._cancel_detection)
        self.cancel_detection_button.setVisible(False)
        controls_layout.addWidget(self.cancel_detection_button)
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
        self.reset_detector_defaults_button = QPushButton(
            "Reset detector settings to defaults"
        )
        self.detector_defaults_status = QLabel()
        self.detector_defaults_status.setWordWrap(True)
        self.find_button.clicked.connect(self._run_detection)
        self.retry_button.clicked.connect(self._run_detection)
        self.reset_detector_defaults_button.clicked.connect(
            self._reset_detector_settings_to_defaults
        )
        actions = QHBoxLayout()
        actions.addWidget(self.find_button)
        actions.addWidget(self.retry_button)

        layout = QVBoxLayout(self.automatic_group)
        layout.addLayout(form)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_widget)
        layout.addWidget(self.detector_defaults_status)
        layout.addWidget(self.reset_detector_defaults_button)
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

    def _build_mask_group(self) -> QGroupBox:
        self.mask_group = QGroupBox("Static prepared-coordinate exclusion mask")
        self.mask_enabled = QCheckBox("Enable static exclusion mask")
        self.mask_enabled.toggled.connect(self._mask_enabled_changed)
        self.add_rectangle_button = QPushButton("Add rectangle")
        self.add_polygon_button = QPushButton("Add polygon")
        self.finish_polygon_button = QPushButton("Finish polygon")
        self.cancel_polygon_button = QPushButton("Cancel polygon")
        self.delete_shape_button = QPushButton("Delete selected shape")
        self.clear_masks_button = QPushButton("Clear all masks")
        self.mask_shape_select = QComboBox()
        self.mask_status = QLabel("No mask shapes.")
        self.mask_status.setWordWrap(True)
        self.add_rectangle_button.clicked.connect(self._begin_mask_rectangle)
        self.add_polygon_button.clicked.connect(self._begin_mask_polygon)
        self.finish_polygon_button.clicked.connect(self._finish_mask_polygon)
        self.cancel_polygon_button.clicked.connect(self._cancel_mask_polygon)
        self.delete_shape_button.clicked.connect(self._delete_selected_mask_shape)
        self.clear_masks_button.clicked.connect(self._clear_all_mask_shapes)
        self.mask_shape_select.currentIndexChanged.connect(self._mask_selection_changed)

        layout = QVBoxLayout(self.mask_group)
        layout.addWidget(self.mask_enabled)
        row = QHBoxLayout()
        row.addWidget(self.add_rectangle_button)
        row.addWidget(self.add_polygon_button)
        layout.addLayout(row)
        polygon_row = QHBoxLayout()
        polygon_row.addWidget(self.finish_polygon_button)
        polygon_row.addWidget(self.cancel_polygon_button)
        layout.addLayout(polygon_row)
        layout.addWidget(QLabel("Select shape"))
        layout.addWidget(self.mask_shape_select)
        layout.addWidget(self.delete_shape_button)
        layout.addWidget(self.clear_masks_button)
        layout.addWidget(self.mask_status)
        return self.mask_group

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
        self._refresh_detector_defaults_status()
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
        prepared_preview = self.controller.prepared_preview
        self.preview_view.set_frame(prepared_preview)
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
        self._refresh_mask_controls(prepared_preview is not None)
        self._refresh_diagnostics()
        self.accept_button.setEnabled(
            candidate is not None
            and prepared_preview is not None
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
        context: GuiTaskContext | None = None
        try:
            request = self.controller.begin_automatic_detection()
            context = self.setup_controller.begin_task(GuiTaskKind.CAGE_DETECTION)
            self._set_busy(True)
            self.refresh_from_state()
            self.task_runner.start_progressive(
                lambda report: self.controller.compute_automatic_detection(
                    request,
                    progress_callback=report,
                    cancellation_requested=context.is_cancellation_requested,
                ),
                task_name="automatic cage detection",
                context=context,
                on_progress=lambda event: self._detection_progress(event, context),
                on_success=lambda result: self._detection_succeeded(result, context),
                on_error=lambda exc: self._detection_failed(exc, context),
                on_finished=lambda: self._detection_finished(context),
            )
        except (CropReviewValidationError, TaskAlreadyRunningError) as exc:
            if context is not None and not self.task_runner.is_running:
                self.setup_controller.finish_task(context)
            self._set_busy(False)
            self._show_expected_error(str(exc))
        except Exception as exc:
            if context is not None and not self.task_runner.is_running:
                self.setup_controller.finish_task(context)
            self._set_busy(False)
            self._show_unexpected(exc)

    def _detection_succeeded(
        self,
        result: object,
        context: GuiTaskContext,
    ) -> None:
        if not self.setup_controller.task_is_current(context):
            return
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
        self.detection_progress_label.setText("Finding cage: Complete")
        self.detection_progress_label.setVisible(True)
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit(
            "Automatic crop candidate is ready for review; it is not accepted."
        )

    def _detection_failed(
        self,
        exc: BaseException,
        context: GuiTaskContext,
    ) -> None:
        if not self.setup_controller.task_belongs_to_current_inputs(context):
            return
        if isinstance(exc, CageDetectionCancelledError):
            self.controller.record_automatic_detection_cancellation()
            self.detection_progress_label.setText("Finding cage: Cancelled")
            self.detection_progress_label.setVisible(True)
            self.error_label.clear()
            self.status_message.emit("Automatic cage detection cancelled.")
            self.validity_changed.emit()
            return
        self.controller.record_automatic_detection_failure(exc)
        self.detection_progress_label.setText("Finding cage: Failed")
        self.detection_progress_label.setVisible(True)
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

    def _detection_finished(self, context: GuiTaskContext) -> None:
        self.setup_controller.finish_task(context)
        self._set_busy(False)
        self.refresh_from_state()

    def _detection_progress(
        self,
        event: GuiTaskProgressEvent,
        context: GuiTaskContext,
    ) -> None:
        if not self.setup_controller.task_progress_is_current(event, context):
            return
        self.detection_progress_label.setText(f"Finding cage: {event.phase}")
        self.detection_progress_label.setVisible(True)

    def _cancel_detection(self) -> None:
        self.task_runner.request_cancellation()
        self.cancel_detection_button.setEnabled(False)
        self.detection_progress_label.setText(
            "Finding cage: Cancellation requested"
        )
        self.detection_progress_label.setVisible(True)

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

    def _reset_detector_settings_to_defaults(self) -> None:
        try:
            result = self.controller.reset_detector_settings_to_defaults()
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            return
        self.error_label.clear()
        self.refresh_from_state()
        if result.already_at_defaults:
            self.status_message.emit("Detector settings already using defaults.")
        elif result.invalidated_accepted:
            self.status_message.emit(
                "Detector settings reset to defaults; automatic crop invalidated. "
                "Rerun automatic detection."
            )
        elif result.invalidated_candidate:
            self.status_message.emit(
                "Detector settings reset to defaults; automatic candidate invalidated. "
                "Rerun automatic detection."
            )
        elif result.preserved_manual_acceptance:
            self.status_message.emit(
                "Detector settings reset to defaults; accepted manual crop preserved."
            )
        else:
            self.status_message.emit(
                "Detector settings reset to defaults; automatic detection must be rerun."
            )

    def _refresh_mask_controls(self, preview_available: bool) -> None:
        config = self.controller.state.preprocess_config
        mask = None if config is None else config.mask
        self.mask_enabled.blockSignals(True)
        self.mask_shape_select.blockSignals(True)
        try:
            self.mask_enabled.setChecked(False if mask is None else mask.enabled)
            self.mask_shape_select.clear()
            shapes = [] if mask is None else mask.shapes
            for index, shape in enumerate(shapes):
                self.mask_shape_select.addItem(_mask_shape_label(index, shape), index)
            if (
                self._selected_mask_shape_index is not None
                and self._selected_mask_shape_index < len(shapes)
            ):
                self.mask_shape_select.setCurrentIndex(self._selected_mask_shape_index)
            else:
                self._selected_mask_shape_index = None
        finally:
            self.mask_enabled.blockSignals(False)
            self.mask_shape_select.blockSignals(False)
        shapes_payload = (
            ()
            if mask is None
            else tuple(shape.model_dump(mode="python") for shape in mask.shapes)
        )
        self.preview_view.set_mask_overlay(
            MaskOverlay(
                shapes=shapes_payload,
                selected_index=self._selected_mask_shape_index,
                pending_polygon=tuple(self._pending_polygon_vertices),
                editing_mode=(
                    "disabled"
                    if not preview_available or self._busy
                    else self._mask_drawing_mode
                ),
                enabled=False if mask is None else mask.enabled,
            )
        )
        self.mask_group.setEnabled(not self._busy)
        self.add_rectangle_button.setEnabled(preview_available and not self._busy)
        self.add_polygon_button.setEnabled(preview_available and not self._busy)
        self.finish_polygon_button.setEnabled(
            preview_available
            and not self._busy
            and self._mask_drawing_mode == "polygon"
        )
        self.cancel_polygon_button.setEnabled(
            preview_available
            and not self._busy
            and self._mask_drawing_mode == "polygon"
        )
        has_selection = self._selected_mask_shape_index is not None
        self.delete_shape_button.setEnabled(has_selection and not self._busy)
        self.clear_masks_button.setEnabled(bool(shapes_payload) and not self._busy)
        if self._mask_drawing_mode == "polygon":
            self.mask_status.setText(
                f"Drawing polygon: {len(self._pending_polygon_vertices)} vertices. "
                "Click the prepared preview, then Finish polygon."
            )
        elif self._mask_drawing_mode == "rectangle":
            self.mask_status.setText("Drag on the prepared preview to add a rectangle.")
        elif not shapes_payload:
            self.mask_status.setText("No mask shapes.")
        elif has_selection:
            self.mask_status.setText(
                f"Selected mask shape {self._selected_mask_shape_index + 1} of "
                f"{len(shapes_payload)}."
            )
        else:
            self.mask_status.setText(f"{len(shapes_payload)} mask shape(s).")

    def _mask_enabled_changed(self, enabled: bool) -> None:
        if self._refreshing:
            return
        try:
            changed = self.controller.set_static_mask_enabled(enabled)
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        self.error_label.clear()
        self.refresh_from_state()
        if changed:
            self.status_message.emit("Static exclusion mask changed; rerun preprocessing.")
        else:
            self.status_message.emit("Static exclusion mask already unchanged.")

    def _begin_mask_rectangle(self) -> None:
        self._mask_drawing_mode = "rectangle"
        self._pending_polygon_vertices.clear()
        self.refresh_from_state()

    def _begin_mask_polygon(self) -> None:
        self._mask_drawing_mode = "polygon"
        self._pending_polygon_vertices.clear()
        self.refresh_from_state()

    def _finish_mask_polygon(self) -> None:
        try:
            self._selected_mask_shape_index = self.controller.add_static_mask_polygon(
                tuple(self._pending_polygon_vertices)
            )
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        self._pending_polygon_vertices.clear()
        self._mask_drawing_mode = "select"
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Static exclusion polygon added; rerun preprocessing.")

    def _cancel_mask_polygon(self) -> None:
        self._pending_polygon_vertices.clear()
        self._mask_drawing_mode = "select"
        self.refresh_from_state()
        self.status_message.emit("In-progress polygon mask cancelled.")

    def _select_mask_shape_from_view(self, index: int) -> None:
        self._selected_mask_shape_index = index
        self._mask_drawing_mode = "select"
        self.refresh_from_state()

    def _mask_selection_changed(self) -> None:
        if self._refreshing:
            return
        self._selected_mask_shape_index = self.mask_shape_select.currentData()
        self._mask_drawing_mode = "select"
        self.refresh_from_state()

    def _add_mask_rectangle(self, geometry: object) -> None:
        if not _is_rectangle_tuple(geometry):
            self._show_expected_error("Rectangle mask geometry is invalid.")
            return
        try:
            self._selected_mask_shape_index = self.controller.add_static_mask_rectangle(
                geometry
            )
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        self._mask_drawing_mode = "select"
        self.error_label.clear()
        self.refresh_from_state()
        self.status_message.emit("Static exclusion rectangle added; rerun preprocessing.")

    def _replace_mask_rectangle(self, index: int, geometry: object) -> None:
        if not _is_rectangle_tuple(geometry):
            self._show_expected_error("Rectangle mask geometry is invalid.")
            return
        try:
            changed = self.controller.replace_static_mask_shape(
                index,
                MaskRectangleConfig(
                    type="rectangle",
                    x=geometry[0],
                    y=geometry[1],
                    width=geometry[2],
                    height=geometry[3],
                ),
            )
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        self._selected_mask_shape_index = index
        self.error_label.clear()
        self.refresh_from_state()
        if changed:
            self.status_message.emit("Static exclusion rectangle changed; rerun preprocessing.")

    def _add_mask_polygon_vertex(self, vertex: object) -> None:
        if (
            not isinstance(vertex, tuple)
            or len(vertex) != 2
            or not all(isinstance(value, int) for value in vertex)
        ):
            self._show_expected_error("Polygon vertex is invalid.")
            return
        self._pending_polygon_vertices.append(vertex)
        self.refresh_from_state()

    def _delete_selected_mask_shape(self) -> None:
        if self._selected_mask_shape_index is None:
            self._show_expected_error("Select a mask shape to delete.")
            return
        try:
            changed = self.controller.delete_static_mask_shape(
                self._selected_mask_shape_index
            )
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        self._selected_mask_shape_index = None
        self.error_label.clear()
        self.refresh_from_state()
        if changed:
            self.status_message.emit("Static exclusion mask shape deleted; rerun preprocessing.")

    def _clear_all_mask_shapes(self) -> None:
        try:
            changed = self.controller.clear_static_mask_shapes()
        except CropReviewValidationError as exc:
            self._show_expected_error(str(exc))
            self.refresh_from_state()
            return
        self._selected_mask_shape_index = None
        self._pending_polygon_vertices.clear()
        self._mask_drawing_mode = "select"
        self.error_label.clear()
        self.refresh_from_state()
        if changed:
            self.status_message.emit("Static exclusion masks cleared; rerun preprocessing.")

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

    def _refresh_detector_defaults_status(self) -> None:
        modified = self.controller.detector_settings_modified_fields()
        if not modified:
            self.detector_defaults_status.setText("Detector settings use defaults.")
            return
        labels = ", ".join(_DETECTOR_FIELD_LABELS.get(field, field) for field in modified)
        self.detector_defaults_status.setText(f"Modified from defaults: {labels}.")

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.advanced_widget.setVisible(checked)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_indicator.setVisible(busy)
        self.cancel_detection_button.setVisible(busy)
        self.cancel_detection_button.setEnabled(busy)
        if busy:
            self.detection_progress_label.clear()
            self.detection_progress_label.setVisible(True)
        self.load_frame_button.setEnabled(not busy)
        self.find_button.setEnabled(not busy)
        self.retry_button.setEnabled(not busy)
        self.automatic_mode.setEnabled(not busy)
        self.manual_mode.setEnabled(not busy)
        self.clear_points_button.setEnabled(not busy)
        self.mask_group.setEnabled(not busy)
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


def _mask_shape_label(index: int, shape: object) -> str:
    shape_type = getattr(shape, "type", "shape")
    if shape_type == "rectangle":
        return (
            f"{index + 1}: rectangle "
            f"({shape.x}, {shape.y}, {shape.width}, {shape.height})"
        )
    if shape_type == "polygon":
        return f"{index + 1}: polygon ({len(shape.vertices)} vertices)"
    return f"{index + 1}: mask shape"


def _is_rectangle_tuple(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 4
        and all(isinstance(item, int) for item in value)
    )
